"""
Main hybrid tracking orchestration.
"""
import gc
import json
import os
import subprocess
import time

import cv2
import numpy as np
import torch

from tracking_pipeline.device import get_device, empty_cache
from tracking_pipeline.sam2_manager import SAM2Manager
from tracking_pipeline.pose import PoseEstimator
from tracking_pipeline.identity_manager import IdentityManager
from tracking_pipeline.state_machine import StateMachine
from tracking_pipeline.video_io import VideoIO
from tracking_pipeline.smoothing import BoxSmoother, KeypointSmoother
from tracking_pipeline.hybrid.yolo26_detector import NumpyEncoder
from tracking_pipeline.hybrid.tracking_loop import execute_tracking_loop
from tracking_pipeline.hybrid.finalize import finalize_tracking


class _TrackingContext:
    """Mutable bag for tracking run state passed to loop/finalize."""

    pass


def run_tracking(
    video_path,
    output_dir,
    # LEGACY: box_a/box_b superseded by player_mapping (track_id<->player_id). Remove once all paths consume bindings.
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
    athlete_bindings=None,
    player_mapping=None,
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

    # Seed init_boxes from the human-confirmed identity binding so track_ids never
    # flip across the resume job chain. Three sources, in precedence order:
    #
    #   1. athlete_bindings (Stream 2, CANONICAL, N-athlete ready: {track_id: box}).
    #      Carries track_id↔player_id↔box directly — the single source of truth when
    #      present; drives both init_boxes seeding and identity grounding.
    #   2. player_mapping (Stream 0b, resume-path binding one hop upstream).
    #      { "<obj_id>": "<player_id>" } with obj_id "1" == box_a, "2" == box_b
    #      (the correction-modal contract). Honored when the canonical
    #      athlete_bindings is absent — keys init_boxes by the CONFIRMED obj_ids so
    #      the human correction still drives seeding on resume.
    #   3. LEGACY positional {1: box_a, 2: box_b} — initial submit, or any resume
    #      that carried neither binding. Remove once all paths consume bindings.
    init_boxes = {
        b.track_id: b.box
        for b in (athlete_bindings or [])
        if getattr(b, "box", None)
    }
    if not init_boxes and player_mapping:
        obj_id_to_box = {1: box_a, 2: box_b}
        init_boxes = {
            int(obj_id): obj_id_to_box[int(obj_id)]
            for obj_id in player_mapping
            if int(obj_id) in obj_id_to_box
        }
    if not init_boxes:
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
        label = chr(ord("A") + track_id - 1) if 1 <= track_id <= 26 else str(track_id)
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

    ctx = _TrackingContext()
    ctx.output_dir = output_dir
    ctx.device = device
    ctx.t_start = t_start
    ctx.sam2_mgr = sam2_mgr
    ctx.pose_est = pose_est
    ctx.identity_mgr = identity_mgr
    ctx.state_machine = state_machine
    ctx.video_io = video_io
    ctx.box_smoother = box_smoother
    ctx.kpt_smoother = kpt_smoother
    ctx.detector = detector
    ctx.fps = fps
    ctx.total_frames = total_frames
    ctx.start_frame = start_frame
    ctx.end_frame = end_frame
    ctx.total_local = total_local
    ctx.BATCH_SIZE = BATCH_SIZE
    ctx.first_batch = first_batch
    ctx.raw_output = raw_output
    ctx.out = out
    ctx.debug_dir = debug_dir
    ctx.json_path = json_path
    ctx._json_file = _json_file
    ctx._json_first_frame = _json_first_frame
    ctx.display_map = display_map
    ctx.current_local = current_local
    ctx.last_known_boxes = last_known_boxes
    ctx.frames_processed = frames_processed
    ctx.user_cancelled = user_cancelled
    ctx.human_suspend = human_suspend
    ctx.missing_frames = missing_frames
    ctx.initial_mask_areas = initial_mask_areas
    ctx.batch_end_local = batch_end_local
    ctx.step_size = step_size
    ctx.max_history = max_history
    ctx.max_missing_frames = max_missing_frames
    ctx.prop_stride = prop_stride
    ctx.frame_stride = frame_stride
    ctx.enable_pose = enable_pose
    ctx.detection_callback = detection_callback
    ctx.progress_callback = progress_callback
    ctx.frame_callback = frame_callback
    ctx.should_stop = should_stop
    ctx.yolo_model = yolo_model
    ctx.detection_threshold = detection_threshold
    ctx.MIN_MASK_PIXELS = MIN_MASK_PIXELS
    ctx.MASK_AREA_COLLAPSE_RATIO = MASK_AREA_COLLAPSE_RATIO

    execute_tracking_loop(ctx)
    return finalize_tracking(ctx)
