"""
Main hybrid tracking loop: SAM2 propagation + state machine.
Ported from bjj-pose-estimation (sam2_tracking.py + run_sam2_hybrid.py)

Adapted for Mac M4 Max: MPS/CPU, YOLO26 for on-demand detection, RTMPose for keypoints.
SAM2 is the sole tracker; YOLO26 only runs when a track is lost (user intervention required).

Usage:
    Imported by ``tracking_pipeline.pipeline``; not meant to be run standalone.
"""

import gc
import json
import math
import os
import subprocess
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .device import get_device, empty_cache
from .sam2_manager import SAM2Manager
from .pose import PoseEstimator
from .identity_manager import IdentityManager
from .state_machine import StateMachine, TrackingState
from .human_verification_suspend import HumanVerificationSuspend
from .video_io import VideoIO
from .smoothing import BoxSmoother, KeypointSmoother


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class YOLO26Detector:
    """Persistent YOLO26 model for repeated per-frame detection."""

    def __init__(self, yolo_model="yolo26m", threshold=0.5, device=None):
        self.threshold = threshold
        model_name = yolo_model if yolo_model.endswith(".pt") else f"{yolo_model}.pt"
        self.device = str(device) if device is not None else "cpu"
        print(f"[detect] Loading {model_name} on {self.device} (persistent)...")
        self.model = YOLO(model_name)
        print("[detect] YOLO26 loaded.")

    def detect_persons(self, frame_bgr, min_area=2000):
        """Detect persons in a BGR frame. Returns list of {box, confidence}."""
        results = self.model(
            frame_bgr,
            conf=self.threshold,
            classes=[0],  # COCO class 0 = person
            verbose=False,
            device=self.device,
        )[0]

        persons = []
        boxes = results.boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().tolist()
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < min_area:
                continue
            persons.append({
                "box": box,
                "confidence": float(boxes.conf[i].cpu()),
            })
        # Explicitly release GPU tensors held by Results object
        del boxes, results
        return persons


def run_tracking(
    video_path,
    output_dir,
    box_a,
    box_b,
    start_frame=0,
    end_frame=None,
    sam2_model_id="facebook/sam2.1-hiera-base-plus",
    step_size=60,
    max_history=8,
    detection_threshold=0.5,
    yolo_model="yolo26m",
    max_missing_frames=15,
    force_cpu=False,
    save_json=True,
    detection_callback=None,
    progress_callback=None,
    frame_callback=None,
    should_stop=None,
    frame_stride=1,
    prop_stride=1,
    enable_pose=True,
):
    """
    Main tracking loop: SAM2 propagation with user intervention on track loss.

    SAM2 is the sole tracker. YOLO26 is only loaded on-demand when an athlete's
    track is lost (mask collapse for more than max_missing_frames). When that
    happens, the user must re-select both athletes via the same UI as the
    initial frame.

    Flow:
        1. Init managers (SAM2, RTMPose, DINOv2 identity gallery, StateMachine)
        2. Extract video frames, initialize SAM2 with user-verified boxes
        3. Build identity gallery from initial frame
        4. Main loop: SAM2-only propagation + loss detection + user intervention
        5. Save output video (H.264) and tracking JSON

    Args:
        video_path: Path to input video.
        output_dir: Output directory for results.
        box_a: [x1, y1, x2, y2] for athlete A (track_id=1).
        box_b: [x1, y1, x2, y2] for athlete B (track_id=2).
        start_frame: Global frame index to start tracking.
        end_frame: Global frame index to stop (None = end of video).
        sam2_model_id: SAM2 model identifier.
        step_size: Frames per SAM2 propagation step (default 60).
        max_history: SAM2 memory pruning window (default 8).
        detection_threshold: YOLO26 confidence threshold (used on loss).
        yolo_model: YOLO26 model variant (lazy-loaded only when needed).
        max_missing_frames: Frames of mask collapse before declaring track lost
            and triggering YOLO + user intervention (default 15, ~0.5s at 30fps).
        force_cpu: Force CPU device.
        save_json: Whether to save tracking JSON.
        frame_stride: Only write every Nth frame to the tracking JSON output.
            SAM2 propagates ALL frames for mask continuity; only the JSON output
            is filtered. Real frame_idx values are preserved. Default 1 = no stride.
        prop_stride: M4 propagation stride. SAM2 only propagates every Nth real frame.
            ffmpeg's select filter extracts only those frames; SAM2 sees them as
            consecutive. Output global_idx values are multiplied by prop_stride so
            they map back to real frame positions. Default 1 = no stride (all frames).
        enable_pose: If True (default), RTMPose estimates keypoints each frame.
            Set False for fast mode to skip pose estimation entirely (~120ms/frame saved).
        detection_callback: Optional callable(reason, frame_jpeg, **kwargs) -> (box_a, box_b) | None.
            Called when human input is needed mid-tracking (BLACKOUT / tracking_lost).
            Receives yolo_detections=[{box, confidence}, ...] as a keyword arg.
            Return None for CLI "still waiting" / service suspend-not-ready.
            Raise HumanVerificationSuspend after persisting state to stop this run_tracking
            immediately (service mode: release worker semaphore after checkpoint).
        progress_callback: Optional callable(frames_done, total_frames, global_idx).
            ``frames_done`` / ``total_frames`` are segment-local (this ``run_tracking``
            invocation).             ``global_idx`` is the absolute video frame index just processed.
            Invoked on the first processed frame and every 30 frames thereafter.
        frame_callback: Optional callable(frame_bgr, global_idx, athletes).
            Called after per-frame results are computed, before writing viz frame.
            ``athletes`` is a list of dicts with keys: track_id, box, keypoints, source.
            Allows inline processing (e.g. upscaling) without a second video pass.
        should_stop: Optional callable() -> bool. If provided and returns True,
            the loop exits cleanly and returns None (e.g. for job cancellation).

    Returns:
        Path to tracking JSON file, or None.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(force_cpu)
    t_start = time.time()

    # ===== 1. Initialize all managers =====
    print("=" * 60)
    print("Initializing tracking managers...")
    print("=" * 60)

    sam2_mgr = SAM2Manager(model_id=sam2_model_id, device=device)
    # M4: conditionally load PoseEstimator. Fast mode skips pose to save ~120ms/frame.
    pose_est = PoseEstimator() if enable_pose else None

    # Shared DINOv2 on CPU (used by identity_mgr for post-scramble verification)
    dino_device = torch.device("cpu")
    print(f"[dino] Loading shared DINOv2 vits14 on {dino_device}...")
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", skip_validation=True)
    dino_model.to(dino_device)
    dino_model.eval()
    print("[dino] Shared DINOv2 loaded.")

    identity_mgr = IdentityManager(device=dino_device, dino_model=dino_model)
    state_machine = StateMachine()
    video_io = VideoIO(video_path)

    # Temporal smoothing filters (OneEuro adaptive low-pass)
    box_smoother = BoxSmoother(min_cutoff=1.7, beta=0.01, blend_frames=5)
    kpt_smoother = KeypointSmoother(
        min_cutoff=0.8, beta=0.015, min_confidence=0.3, max_velocity=200.0,
    )

    # YOLO26 detector: lazy-loaded on first track loss (saves ~200-400MB at startup)
    detector = None

    fps = video_io.fps
    total_frames = video_io.total_frames
    if end_frame is None:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)
    total_local = end_frame - start_frame

    # ===== 2. Init SAM2 (batch-based, memory-bounded) =====
    # M4: BATCH_SIZE is in REAL frame units. With prop_stride=12, a batch of 60
    # real frames yields only 5 SAM2-visible frames (60/12). Keep batch in real
    # frames so batch_end_local arithmetic (real frame units) stays consistent.
    BATCH_SIZE = 60  # real frames per batch

    print()
    print("=" * 60)
    print("Initializing SAM2 (batch mode)...")
    print("=" * 60)
    sam2_mgr.init_video_meta(video_path, start_frame, end_frame)
    first_batch = min(BATCH_SIZE, total_local)
    sam2_mgr.load_batch(0, first_batch, prop_stride=prop_stride)

    # Read initial frame
    video_io.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame0 = video_io.cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame {start_frame}")
    frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)

    # ===== 3. Initialize tracks =====
    print()
    print("=" * 60)
    print("Initializing tracks & identity gallery...")
    print("=" * 60)

    init_boxes = {1: box_a, 2: box_b}

    # Add initial boxes to SAM2 + build identity gallery
    for track_id, box in init_boxes.items():
        mask = sam2_mgr.add_initial_box(0, track_id, box)
        if pose_est is not None:
            kpts, scores = pose_est.estimate(frame0, box)
        else:
            kpts, scores = np.zeros((17, 2)), np.zeros(17)
        identity_mgr.update_gallery(
            track_id, frame0_rgb,
            mask=mask, box=box,
            keypoints=kpts, scores=scores,
        )
        label = "A" if track_id == 1 else "B"
        print(f"  Athlete {label} (track {track_id}): "
              f"box={[round(c) for c in box]}, mask={mask.sum()} px")

    # ===== 4. Setup output =====
    raw_output = os.path.join(output_dir, "tracked_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        raw_output, fourcc, fps, (video_io.width, video_io.height),
    )

    debug_dir = os.path.join(output_dir, "debug_frames")
    os.makedirs(debug_dir, exist_ok=True)

    # Streaming JSON writer: write frame entries to disk as they're produced
    # instead of accumulating all in memory.
    json_path = os.path.join(output_dir, "tracking.json") if save_json else None
    _json_file = None
    _json_first_frame = True
    if save_json:
        _json_file = open(json_path, "w")
        _json_file.write(json.dumps({
            "video": video_path,
            "fps": fps,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }, cls=NumpyEncoder)[:-1])  # strip closing }
        _json_file.write(', "frames": [\n')

    display_map = {1: 1, 2: 2}

    # ===== 5. Main tracking loop =====
    print()
    print("=" * 60)
    print(f"Tracking: frames {start_frame} -> {end_frame} "
          f"({total_local} frames)")
    print(f"  step_size={step_size}, max_history={max_history}, "
          f"max_missing_frames={max_missing_frames}, prop_stride={prop_stride}, "
          f"enable_pose={enable_pose}")
    print("=" * 60)

    current_local = 0
    last_known_boxes = {}  # {track_id: box} for batch carry-over
    frames_processed = 0
    user_cancelled = False
    human_suspend = False

    # Loss detection state
    missing_frames = {1: 0, 2: 0}
    initial_mask_areas = {}  # {track_id: pixel_count} from first valid mask
    MIN_MASK_PIXELS = 50
    MASK_AREA_COLLAPSE_RATIO = 0.10  # lost if < 10% of initial area

    # Track batch boundary
    batch_end_local = sam2_mgr.batch_offset + first_batch

    while current_local < total_local:
        if user_cancelled:
            break
        # --- Load next batch if needed ---
        if current_local >= batch_end_local:
            next_size = min(BATCH_SIZE, total_local - current_local)
            sam2_mgr.load_batch(current_local, next_size, prop_stride=prop_stride)
            if last_known_boxes:
                sam2_mgr.reprompt_boxes(last_known_boxes)
            batch_end_local = current_local + next_size

        # M4: convert real-frame step budget to SAM2-visible frame count.
        # batch_end_local and current_local are in real frame units.
        real_steps = min(step_size, batch_end_local - current_local)
        steps = max(1, int(math.ceil(real_steps / prop_stride)))
        if real_steps <= 0:
            break

        generator = sam2_mgr.propagate_step(current_local, steps=steps)

        for batch_rel_idx, obj_ids, mask_logits in generator:
            if should_stop is not None and should_stop():
                user_cancelled = True
                break
            if sam2_mgr.is_reset:
                sam2_mgr.is_reset = False
                break

            # M4: batch_offset is in real-frame units; batch_rel_idx is in SAM2-frame units.
            # Real frame position = batch_offset (real) + batch_rel_idx (SAM2) * prop_stride.
            # Example: batch_offset=60, batch_rel_idx=3, prop_stride=12
            #   → local_idx = 60 + 3*12 = 96 (real frame 96 from segment start)
            local_idx = sam2_mgr.batch_offset + batch_rel_idx * prop_stride
            global_idx = start_frame + local_idx

            # Load frame from temp dir (batch-relative path)
            frame_path = os.path.join(
                sam2_mgr.temp_dir, f"{batch_rel_idx:05d}.jpg",
            )
            if not os.path.exists(frame_path):
                continue
            frame_bgr = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # --- SAM2 masks ---
            masks_sam2 = {}
            for i, track_id in enumerate(obj_ids):
                m = (mask_logits[i] > 0.0).cpu().numpy().squeeze()
                if m.shape != (video_io.height, video_io.width):
                    m = cv2.resize(
                        m.astype(np.uint8),
                        (video_io.width, video_io.height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                if m.sum() > 50:
                    masks_sam2[track_id] = m
            del mask_logits  # Release GPU tensor immediately

            # --- Scene/fade detection ---
            is_cut = video_io.detect_scene_change(frame_bgr)
            fade_state = video_io.update_fade_state(frame_bgr)

            # --- Mask IoU between athletes ---
            iou = 0.0
            if 1 in masks_sam2 and 2 in masks_sam2:
                inter = np.logical_and(masks_sam2[1], masks_sam2[2]).sum()
                union = np.logical_or(masks_sam2[1], masks_sam2[2]).sum()
                iou = inter / union if union > 0 else 0

            # --- State machine update ---
            prev_state = state_machine.state
            state_machine.update(iou, is_cut, fade_state)
            curr_state = state_machine.state

            if prev_state != curr_state:
                print(f"  Frame {global_idx}: {prev_state.name} -> "
                      f"{curr_state.name} (IoU={iou:.2f})")

            # --- Build per-frame results ---
            frame_boxes = {}
            frame_kpts = {}
            frame_scores = {}
            frame_masks = {}
            frame_sources = {}

            # ==============================
            # RE_ID_MODE: user intervention required
            # ==============================
            if curr_state == TrackingState.RE_ID_MODE:
                try:
                    new_boxes = _detect_and_request_boxes(
                        frame_bgr, global_idx, detection_callback,
                        detector, yolo_model, detection_threshold, device,
                    )
                except HumanVerificationSuspend:
                    human_suspend = True
                    break
                if new_boxes is not None:
                    detector = new_boxes[2]  # updated detector ref
                    new_box_a, new_box_b = new_boxes[0], new_boxes[1]
                    sam2_mgr.reset_state()
                    sam2_mgr.add_initial_box(batch_rel_idx, 1, new_box_a)
                    sam2_mgr.add_initial_box(batch_rel_idx, 2, new_box_b)
                    last_known_boxes[1] = new_box_a
                    last_known_boxes[2] = new_box_b
                    box_smoother.reset()
                    kpt_smoother.reset()
                    missing_frames = {1: 0, 2: 0}
                    initial_mask_areas.clear()
                    state_machine.set_tracking()
                    current_local = local_idx + prop_stride
                    break  # Restart propagation after user intervention

                # User cancelled — write empty frame and stay in RE_ID
                _write_viz_frame(
                    out, frame_bgr, frame_boxes, frame_masks,
                    frame_kpts, frame_scores, frame_sources, display_map,
                    curr_state, iou, global_idx, debug_dir, local_idx,
                )
                if _json_file is not None and global_idx % frame_stride == 0:
                    _json_first_frame = _append_frame_to_json(
                        _json_file, _json_first_frame,
                        global_idx, local_idx, fps, curr_state, iou,
                        frame_boxes, frame_kpts, frame_scores,
                        frame_sources, display_map,
                    )
                frames_processed += 1
                if progress_callback is not None and (
                    frames_processed == 1 or frames_processed % 30 == 0
                ):
                    try:
                        progress_callback(frames_processed, total_local, global_idx)
                    except Exception as e:
                        print(f"[tracking] progress_callback error (non-fatal): {e}")
                del frame_bgr, frame_rgb, masks_sam2
                del frame_boxes, frame_kpts, frame_scores, frame_masks, frame_sources
                current_local = local_idx + prop_stride
                sam2_mgr.prune_memory(max_history, flush_cache=False)
                continue  # Stay in RE_ID mode

            # ==============================
            # BLACKOUT entry: clear SAM2 state, optionally request human boxes
            # ==============================
            if (curr_state == TrackingState.BLACKOUT
                    and prev_state != TrackingState.BLACKOUT):
                print(f"  Frame {global_idx}: BLACKOUT — clearing SAM2")
                sam2_mgr.reset_state()
                box_smoother.reset()
                kpt_smoother.reset()
                missing_frames = {1: 0, 2: 0}
                initial_mask_areas.clear()
                if detection_callback is not None:
                    try:
                        _, frame_jpeg_enc = cv2.imencode(".jpg", frame_bgr)
                        cb_result = detection_callback(
                            "transition", frame_jpeg_enc.tobytes(),
                            yolo_detections=[],
                            frame_idx=global_idx,
                        )
                        if cb_result is not None:
                            new_box_a, new_box_b = cb_result
                            sam2_mgr.add_initial_box(batch_rel_idx, 1, new_box_a)
                            sam2_mgr.add_initial_box(batch_rel_idx, 2, new_box_b)
                            state_machine.set_tracking()
                            current_local = local_idx + prop_stride
                            break
                    except HumanVerificationSuspend:
                        human_suspend = True
                        break
                    except Exception as _cb_err:
                        print(f"  [detection_callback] BLACKOUT callback failed: {_cb_err}")

            # ==============================
            # SAM2 PRIMARY: derive boxes from masks every frame
            # ==============================
            timestamp = global_idx / fps  # seconds, for OneEuro filters

            for track_id in [1, 2]:
                if track_id in masks_sam2:
                    mask = masks_sam2[track_id]
                    mask_area = int(mask.sum())
                    ys, xs = np.where(mask)

                    # Record initial mask area for collapse detection
                    if track_id not in initial_mask_areas and mask_area > MIN_MASK_PIXELS:
                        initial_mask_areas[track_id] = mask_area

                    # Check for mask collapse (track loss detection)
                    init_area = initial_mask_areas.get(track_id, mask_area)
                    area_ratio = mask_area / max(init_area, 1)
                    if mask_area < MIN_MASK_PIXELS or area_ratio < MASK_AREA_COLLAPSE_RATIO:
                        missing_frames[track_id] += 1
                    else:
                        missing_frames[track_id] = 0

                    if len(ys) > 0 and mask_area >= MIN_MASK_PIXELS:
                        raw_box = [float(xs.min()), float(ys.min()),
                                   float(xs.max()), float(ys.max())]
                        # Smooth bounding box (reduces mask-edge jitter)
                        box = box_smoother.smooth(track_id, raw_box, timestamp)
                        frame_boxes[track_id] = box
                        frame_masks[track_id] = mask
                        frame_sources[track_id] = "SAM2"

                        # M4: skip pose estimation in fast mode (enable_pose=False)
                        if pose_est is not None:
                            # Pose on smoothed box → more stable crop
                            kpts, sc = pose_est.estimate(frame_bgr, box)
                            # Smooth keypoints with confidence filtering
                            kpts_s, sc_adj = kpt_smoother.smooth(
                                track_id, kpts, sc, timestamp,
                            )
                            frame_kpts[track_id] = kpts_s
                            frame_scores[track_id] = sc_adj
                else:
                    missing_frames[track_id] += 1

            # ==============================
            # Identity verification after scramble exit
            # ==============================
            if (prev_state == TrackingState.SCRAMBLE
                    and curr_state == TrackingState.TRACKING):
                _correct_identity_swap_after_scramble(
                    frame_boxes, frame_masks, frame_rgb,
                    identity_mgr, display_map, global_idx,
                )

            # ==============================
            # Loss detection: trigger user intervention when track lost
            # ==============================
            any_lost = any(
                missing_frames[track_id] > max_missing_frames for track_id in [1, 2]
            )
            if any_lost and curr_state == TrackingState.TRACKING:
                lost_ids = [track_id for track_id in [1, 2]
                            if missing_frames[track_id] > max_missing_frames]
                print(f"  Frame {global_idx}: TRACK LOST for athlete(s) "
                      f"{lost_ids} (missing > {max_missing_frames} frames)")
                state_machine.state = TrackingState.RE_ID_MODE

                try:
                    new_boxes = _detect_and_request_boxes(
                        frame_bgr, global_idx, detection_callback,
                        detector, yolo_model, detection_threshold, device,
                    )
                except HumanVerificationSuspend:
                    human_suspend = True
                    break
                if new_boxes is not None:
                    detector = new_boxes[2]  # updated detector ref
                    new_box_a, new_box_b = new_boxes[0], new_boxes[1]
                    sam2_mgr.reset_state()
                    sam2_mgr.add_initial_box(batch_rel_idx, 1, new_box_a)
                    sam2_mgr.add_initial_box(batch_rel_idx, 2, new_box_b)
                    last_known_boxes[1] = new_box_a
                    last_known_boxes[2] = new_box_b
                    box_smoother.reset()
                    kpt_smoother.reset()
                    missing_frames = {1: 0, 2: 0}
                    initial_mask_areas.clear()
                    state_machine.set_tracking()
                    current_local = local_idx + prop_stride
                    break  # Restart propagation after user intervention

            # ==============================
            # Write output frame
            # ==============================
            _write_viz_frame(
                out, frame_bgr, frame_boxes, frame_masks,
                frame_kpts, frame_scores, frame_sources, display_map,
                curr_state, iou, global_idx, debug_dir, local_idx,
            )

            if _json_file is not None and global_idx % frame_stride == 0:
                _json_first_frame = _append_frame_to_json(
                    _json_file, _json_first_frame,
                    global_idx, local_idx, fps, curr_state, iou,
                    frame_boxes, frame_kpts, frame_scores,
                    frame_sources, display_map,
                )

            if frame_callback is not None and frame_boxes:
                try:
                    athletes = _build_athlete_dicts(
                        frame_boxes, frame_kpts, frame_scores,
                        frame_sources, display_map,
                    )
                    frame_callback(frame_bgr, global_idx, athletes)
                except Exception as _cb_err:
                    print(f"  [frame_callback] error at frame {global_idx}: {_cb_err}")

            # Track last-known boxes for batch carry-over
            if frame_boxes:
                for track_id, box in frame_boxes.items():
                    last_known_boxes[track_id] = box

            frames_processed += 1
            if frames_processed % 30 == 0:
                elapsed = time.time() - t_start
                pct = frames_processed / total_local * 100
                print(f"  [{pct:.0f}%] Frame {global_idx} | "
                      f"{curr_state.value} | "
                      f"Tracks: {len(frame_boxes)} | {elapsed:.0f}s")
                if should_stop is not None and should_stop():
                    user_cancelled = True
                    break
            if progress_callback is not None and (
                frames_processed == 1 or frames_processed % 30 == 0
            ):
                try:
                    progress_callback(frames_processed, total_local, global_idx)
                except Exception as e:
                    print(f"[tracking] progress_callback error (non-fatal): {e}")

            # Release per-frame objects to reduce GC pressure
            del frame_bgr, frame_rgb, masks_sam2
            del frame_boxes, frame_kpts, frame_scores, frame_masks, frame_sources

            # M4: advance current_local by prop_stride real frames.
            # local_idx = (batch_offset + batch_rel_idx) * prop_stride
            # So the next real frame position is local_idx + prop_stride.
            current_local = local_idx + prop_stride

            # Prune SAM2 memory every frame (one entry removed at a time,
            # constant cost). flush_cache=False defers MPS heap compaction to
            # batch boundaries — calling empty_cache mid-propagation on every
            # frame caused progressive MPS fragmentation spikes.
            sam2_mgr.prune_memory(max_history, flush_cache=False)

        if user_cancelled or human_suspend:
            break
        # End of propagation step: heavy cleanup at batch boundary only.
        # flush_cache=True triggers MPS empty_cache here (not mid-propagation).
        sam2_mgr.prune_memory(max_history, flush_cache=True)
        gc.collect()   # Expensive heap scan — safe here between batches
        empty_cache()

    # ===== 6. Finalize =====
    out.release()
    video_io.release()

    if user_cancelled or human_suspend:
        if _json_file is not None:
            _json_file.write("\n]}\n")
            _json_file.close()
        sam2_mgr.cleanup()
        print(
            "Tracking cancelled by user."
            if user_cancelled
            else "Tracking suspended for human verification.",
        )
        return None

    # Close streaming JSON FIRST before any cleanup that might crash.
    if _json_file is not None:
        _json_file.write("\n]}\n")
        _json_file.close()
        print(f"Tracking JSON: {json_path}")

    # Re-encode with H.264 for compatibility
    output_video = os.path.join(output_dir, "tracked_output.mp4")
    print(f"\nRe-encoding: {output_video}")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_output,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-loglevel", "error",
                output_video,
            ],
            check=True,
        )
        os.remove(raw_output)
    except Exception as e:
        print(f"FFmpeg failed ({e}). Raw video at: {raw_output}")
        output_video = raw_output

    sam2_mgr.cleanup()

    elapsed = time.time() - t_start
    n = frames_processed
    rate = n / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed:.1f}s ({n} frames, {rate:.1f} fps)")
    print(f"Output: {output_video}")

    return json_path


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_athlete_dicts(frame_boxes, frame_kpts, frame_scores,
                         frame_sources, display_map):
    """Build the athlete dict list for frame_callback."""
    athletes = []
    for track_id, box in frame_boxes.items():
        disp_id = display_map.get(track_id, track_id)
        athlete = {
            "track_id": disp_id,
            "box": [round(c, 1) for c in box],
            "source": frame_sources.get(track_id, "unknown"),
        }
        kpts = frame_kpts.get(track_id)
        if kpts is not None:
            if hasattr(kpts, "tolist"):
                athlete["keypoints"] = kpts.tolist()
            else:
                athlete["keypoints"] = kpts
        kpt_sc = frame_scores.get(track_id)
        if kpt_sc is not None:
            if hasattr(kpt_sc, "tolist"):
                athlete["keypoint_scores"] = kpt_sc.tolist()
            else:
                athlete["keypoint_scores"] = kpt_sc
        athletes.append(athlete)
    return athletes


def _detect_and_request_boxes(frame_bgr, global_idx, detection_callback,
                        detector, yolo_model, detection_threshold, device):
    """Lazy-load YOLO, detect persons, and request user-verified boxes.

    Returns (box_a, box_b, detector) on success, or None if user cancelled.
    The detector is returned so the caller can keep the reference for future use.
    """
    # Lazy-load YOLO detector
    if detector is None:
        print(f"  Frame {global_idx}: Loading YOLO detector on demand...")
        detector = YOLO26Detector(
            yolo_model=yolo_model, threshold=detection_threshold, device=device,
        )

    yolo_detections = detector.detect_persons(frame_bgr)

    if detection_callback is not None:
        # Service mode: send to human via WebSocket
        _, frame_jpeg_enc = cv2.imencode(".jpg", frame_bgr)
        try:
            cb_result = detection_callback(
                "tracking_lost", frame_jpeg_enc.tobytes(),
                yolo_detections=yolo_detections,
                frame_idx=global_idx,
            )
            if cb_result is not None:
                return (cb_result[0], cb_result[1], detector)
        except HumanVerificationSuspend:
            raise
        except Exception as e:
            print(f"  [detection_callback] failed: {e}")
        return None
    else:
        # No callback: parallel-segment mode or headless run.
        # Return None to signal track loss; caller handles via max_missing_frames.
        # Do NOT try CLI select_boxes — not available in server context.
        print(f"  Frame {global_idx}: Track lost, no detection_callback — continuing")
        return None


def _correct_identity_swap_after_scramble(frame_boxes, frame_masks, frame_rgb,
                                    identity_mgr, display_map, global_idx):
    """After exiting SCRAMBLE, check if identities got swapped."""
    print(f"  Frame {global_idx}: Exiting SCRAMBLE — verifying identities...")

    if 1 not in frame_boxes or 2 not in frame_boxes:
        return

    s1 = identity_mgr.query_identity_scores(
        frame_rgb, frame_boxes[1], mask=frame_masks.get(1),
    )
    s2 = identity_mgr.query_identity_scores(
        frame_rgb, frame_boxes[2], mask=frame_masks.get(2),
    )

    s1_as_1 = s1.get(1, 0)
    s1_as_2 = s1.get(2, 0)
    s2_as_1 = s2.get(1, 0)
    s2_as_2 = s2.get(2, 0)

    print(f"    Track 1: ID1={s1_as_1:.2f}, ID2={s1_as_2:.2f}")
    print(f"    Track 2: ID1={s2_as_1:.2f}, ID2={s2_as_2:.2f}")

    if s1_as_2 > s1_as_1 and s2_as_1 > s2_as_2:
        print("    Identity SWAP detected! Correcting display labels.")
        display_map[1], display_map[2] = display_map[2], display_map[1]


# COCO 17-joint skeleton connections for pose visualization
_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # upper body
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]


def _write_viz_frame(out, frame_bgr, frame_boxes, frame_masks,
                     frame_kpts, frame_scores, frame_sources, display_map,
                     state, iou, global_idx, debug_dir, local_idx):
    """Draw masks, boxes, skeleton, labels, and status on frame, then write to video."""
    viz = frame_bgr.copy()
    colors = {1: (0, 255, 0), 2: (255, 0, 0)}  # Green=A, Blue=B

    for track_id, box in frame_boxes.items():
        disp_id = display_map.get(track_id, track_id)
        color = colors.get(disp_id, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box)

        # Semi-transparent mask overlay
        if track_id in frame_masks:
            overlay = np.zeros_like(viz)
            overlay[frame_masks[track_id]] = color
            viz = cv2.addWeighted(viz, 1.0, overlay, 0.4, 0)

        # Pose skeleton (confidence-aware: suppress low-confidence limbs)
        kpts = frame_kpts.get(track_id)
        kpt_sc = frame_scores.get(track_id)
        if kpts is not None:
            pts = kpts if isinstance(kpts, list) else kpts.tolist()
            scs = None
            if kpt_sc is not None:
                scs = kpt_sc if isinstance(kpt_sc, list) else kpt_sc.tolist()
            # Draw limb connections (only if both endpoints confident)
            for i, j in _SKELETON_EDGES:
                if i < len(pts) and j < len(pts):
                    if scs and (scs[i] <= 0 or scs[j] <= 0):
                        continue  # skip limbs with suppressed endpoints
                    p1 = (int(pts[i][0]), int(pts[i][1]))
                    p2 = (int(pts[j][0]), int(pts[j][1]))
                    if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                        cv2.line(viz, p1, p2, color, 2, cv2.LINE_AA)
            # Draw joint dots (skip suppressed joints)
            for idx_j, pt in enumerate(pts):
                px, py = int(pt[0]), int(pt[1])
                if px > 0 and py > 0:
                    if scs and idx_j < len(scs) and scs[idx_j] <= 0:
                        continue
                    cv2.circle(viz, (px, py), 3, color, -1, cv2.LINE_AA)

        # Bounding box + label
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
        source = frame_sources.get(track_id, "?")
        label = f"ID {disp_id} ({source})"
        cv2.putText(viz, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Status bar
    status = f"{state.value} | IoU:{iou:.2f} | F:{global_idx}"
    cv2.putText(viz, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out.write(viz)

    # Save debug frame every 120 frames
    if local_idx % 120 == 0:
        cv2.imwrite(
            os.path.join(debug_dir, f"debug_{global_idx:05d}.jpg"), viz,
        )


def _append_frame_to_json(json_file, is_first, global_idx, local_idx, fps,
                       state, iou, frame_boxes, frame_kpts, frame_scores,
                       frame_sources, display_map):
    """Write one frame's data directly to the JSON file (streaming).

    Returns updated is_first flag.
    """
    frame_entry = {
        "frame_idx": global_idx,
        "local_idx": local_idx,
        "timestamp": round(global_idx / fps, 4),
        "state": state.value,
        "iou": round(iou, 4),
        "athletes": _build_athlete_dicts(
            frame_boxes, frame_kpts, frame_scores, frame_sources, display_map
        ),
    }

    if not is_first:
        json_file.write(",\n")
    json_file.write(json.dumps(frame_entry, cls=NumpyEncoder))
    json_file.flush()
    return False
