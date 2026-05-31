"""Main SAM2 propagation loop."""
import gc
import math
import os
import time

import cv2
import numpy as np

from tracking_pipeline.device import empty_cache
from tracking_pipeline.state_machine import TrackingState
from tracking_pipeline.human_verification_suspend import HumanVerificationSuspend
from tracking_pipeline.hybrid.intervention import (
    _correct_identity_swap_after_scramble,
    _detect_and_request_boxes,
)
from tracking_pipeline.hybrid.output import (
    _append_frame_to_json,
    _build_athlete_dicts,
    _write_viz_frame,
)


def execute_tracking_loop(ctx):
    """Run the main tracking loop; mutates ``ctx`` in place."""

    while ctx.current_local < ctx.total_local:
        if ctx.user_cancelled:
            break
        # --- Load next batch if needed ---
        if ctx.current_local >= ctx.batch_end_local:
            next_size = min(ctx.BATCH_SIZE, ctx.total_local - ctx.current_local)
            ctx.sam2_mgr.load_batch(ctx.current_local, next_size, prop_stride=ctx.prop_stride)
            if ctx.last_known_boxes:
                ctx.sam2_mgr.reprompt_boxes(ctx.last_known_boxes)
            ctx.batch_end_local = ctx.current_local + next_size

        # M4: convert real-frame step budget to SAM2-visible frame count.
        # ctx.batch_end_local and ctx.current_local are in real frame units.
        real_steps = min(ctx.step_size, ctx.batch_end_local - ctx.current_local)
        steps = max(1, int(math.ceil(real_steps / ctx.prop_stride)))
        if real_steps <= 0:
            break

        generator = ctx.sam2_mgr.propagate_step(ctx.current_local, steps=steps)

        for batch_rel_idx, obj_ids, mask_logits in generator:
            if ctx.should_stop is not None and ctx.should_stop():
                ctx.user_cancelled = True
                break
            if ctx.sam2_mgr.is_reset:
                ctx.sam2_mgr.is_reset = False
                break

            # M4: batch_offset is in real-frame units; batch_rel_idx is in SAM2-frame units.
            # Real frame position = batch_offset (real) + batch_rel_idx (SAM2) * ctx.prop_stride.
            # Example: batch_offset=60, batch_rel_idx=3, ctx.prop_stride=12
            #   → local_idx = 60 + 3*12 = 96 (real frame 96 from segment start)
            local_idx = ctx.sam2_mgr.batch_offset + batch_rel_idx * ctx.prop_stride
            global_idx = ctx.start_frame + local_idx

            # Load frame from temp dir (batch-relative path)
            frame_path = os.path.join(
                ctx.sam2_mgr.temp_dir, f"{batch_rel_idx:05d}.jpg",
            )
            if not os.path.exists(frame_path):
                continue
            frame_bgr = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # --- SAM2 masks ---
            masks_sam2 = {}
            for i, track_id in enumerate(obj_ids):
                m = (mask_logits[i] > 0.0).cpu().numpy().squeeze()
                if m.shape != (ctx.video_io.height, ctx.video_io.width):
                    m = cv2.resize(
                        m.astype(np.uint8),
                        (ctx.video_io.width, ctx.video_io.height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                if m.sum() > 50:
                    masks_sam2[track_id] = m
            del mask_logits  # Release GPU tensor immediately

            # --- Scene/fade detection ---
            is_cut = ctx.video_io.detect_scene_change(frame_bgr)
            fade_state = ctx.video_io.update_fade_state(frame_bgr)

            # --- Mask IoU between athletes ---
            iou = 0.0
            if 1 in masks_sam2 and 2 in masks_sam2:
                inter = np.logical_and(masks_sam2[1], masks_sam2[2]).sum()
                union = np.logical_or(masks_sam2[1], masks_sam2[2]).sum()
                iou = inter / union if union > 0 else 0

            # --- State machine update ---
            prev_state = ctx.state_machine.state
            ctx.state_machine.update(iou, is_cut, fade_state)
            curr_state = ctx.state_machine.state

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
                        frame_bgr, global_idx, ctx.detection_callback,
                        ctx.detector, ctx.yolo_model, ctx.detection_threshold, ctx.device,
                    )
                except HumanVerificationSuspend:
                    ctx.human_suspend = True
                    break
                if new_boxes is not None:
                    ctx.detector = new_boxes[2]  # updated ctx.detector ref
                    new_box_a, new_box_b = new_boxes[0], new_boxes[1]
                    ctx.sam2_mgr.reset_state()
                    ctx.sam2_mgr.add_initial_box(batch_rel_idx, 1, new_box_a)
                    ctx.sam2_mgr.add_initial_box(batch_rel_idx, 2, new_box_b)
                    ctx.last_known_boxes[1] = new_box_a
                    ctx.last_known_boxes[2] = new_box_b
                    ctx.box_smoother.reset()
                    ctx.kpt_smoother.reset()
                    ctx.missing_frames = {1: 0, 2: 0}
                    ctx.initial_mask_areas.clear()
                    ctx.state_machine.set_tracking()
                    ctx.current_local = local_idx + ctx.prop_stride
                    break  # Restart propagation after user intervention

                # User cancelled — write empty frame and stay in RE_ID
                _write_viz_frame(
                    ctx.out, frame_bgr, frame_boxes, frame_masks,
                    frame_kpts, frame_scores, frame_sources, ctx.display_map,
                    curr_state, iou, global_idx, ctx.debug_dir, local_idx,
                )
                if ctx._json_file is not None and global_idx % ctx.frame_stride == 0:
                    ctx._json_first_frame = _append_frame_to_json(
                        ctx._json_file, ctx._json_first_frame,
                        global_idx, local_idx, ctx.fps, curr_state, iou,
                        frame_boxes, frame_kpts, frame_scores,
                        frame_sources, ctx.display_map,
                    )
                ctx.frames_processed += 1
                if ctx.progress_callback is not None and (
                    ctx.frames_processed == 1 or ctx.frames_processed % 30 == 0
                ):
                    try:
                        ctx.progress_callback(ctx.frames_processed, ctx.total_local, global_idx)
                    except Exception as e:
                        print(f"[tracking] ctx.progress_callback error (non-fatal): {e}")
                del frame_bgr, frame_rgb, masks_sam2
                del frame_boxes, frame_kpts, frame_scores, frame_masks, frame_sources
                ctx.current_local = local_idx + ctx.prop_stride
                ctx.sam2_mgr.prune_memory(ctx.max_history, flush_cache=False)
                continue  # Stay in RE_ID mode

            # ==============================
            # BLACKOUT entry: clear SAM2 state, optionally request human boxes
            # ==============================
            if (curr_state == TrackingState.BLACKOUT
                    and prev_state != TrackingState.BLACKOUT):
                print(f"  Frame {global_idx}: BLACKOUT — clearing SAM2")
                ctx.sam2_mgr.reset_state()
                ctx.box_smoother.reset()
                ctx.kpt_smoother.reset()
                ctx.missing_frames = {1: 0, 2: 0}
                ctx.initial_mask_areas.clear()
                if ctx.detection_callback is not None:
                    try:
                        _, frame_jpeg_enc = cv2.imencode(".jpg", frame_bgr)
                        cb_result = ctx.detection_callback(
                            "transition", frame_jpeg_enc.tobytes(),
                            yolo_detections=[],
                            frame_idx=global_idx,
                        )
                        if cb_result is not None:
                            new_box_a, new_box_b = cb_result
                            ctx.sam2_mgr.add_initial_box(batch_rel_idx, 1, new_box_a)
                            ctx.sam2_mgr.add_initial_box(batch_rel_idx, 2, new_box_b)
                            ctx.state_machine.set_tracking()
                            ctx.current_local = local_idx + ctx.prop_stride
                            break
                    except HumanVerificationSuspend:
                        ctx.human_suspend = True
                        break
                    except Exception as _cb_err:
                        print(f"  [ctx.detection_callback] BLACKOUT callback failed: {_cb_err}")

            # ==============================
            # SAM2 PRIMARY: derive boxes from masks every frame
            # ==============================
            timestamp = global_idx / ctx.fps  # seconds, for OneEuro filters

            for track_id in [1, 2]:
                if track_id in masks_sam2:
                    mask = masks_sam2[track_id]
                    mask_area = int(mask.sum())
                    ys, xs = np.where(mask)

                    # Record initial mask area for collapse detection
                    if track_id not in ctx.initial_mask_areas and mask_area > ctx.MIN_MASK_PIXELS:
                        ctx.initial_mask_areas[track_id] = mask_area

                    # Check for mask collapse (track loss detection)
                    init_area = ctx.initial_mask_areas.get(track_id, mask_area)
                    area_ratio = mask_area / max(init_area, 1)
                    if mask_area < ctx.MIN_MASK_PIXELS or area_ratio < ctx.MASK_AREA_COLLAPSE_RATIO:
                        ctx.missing_frames[track_id] += 1
                    else:
                        ctx.missing_frames[track_id] = 0

                    if len(ys) > 0 and mask_area >= ctx.MIN_MASK_PIXELS:
                        raw_box = [float(xs.min()), float(ys.min()),
                                   float(xs.max()), float(ys.max())]
                        # Smooth bounding box (reduces mask-edge jitter)
                        box = ctx.box_smoother.smooth(track_id, raw_box, timestamp)
                        frame_boxes[track_id] = box
                        frame_masks[track_id] = mask
                        frame_sources[track_id] = "SAM2"

                        # M4: skip pose estimation in fast mode (ctx.enable_pose=False)
                        if ctx.pose_est is not None:
                            # Pose on smoothed box → more stable crop
                            kpts, sc = ctx.pose_est.estimate(frame_bgr, box)
                            # Smooth keypoints with confidence filtering
                            kpts_s, sc_adj = ctx.kpt_smoother.smooth(
                                track_id, kpts, sc, timestamp,
                            )
                            frame_kpts[track_id] = kpts_s
                            frame_scores[track_id] = sc_adj
                else:
                    ctx.missing_frames[track_id] += 1

            # ==============================
            # Identity verification after scramble exit
            # ==============================
            if (prev_state == TrackingState.SCRAMBLE
                    and curr_state == TrackingState.TRACKING):
                _correct_identity_swap_after_scramble(
                    frame_boxes, frame_masks, frame_rgb,
                    ctx.identity_mgr, ctx.display_map, global_idx,
                )

            # ==============================
            # Loss detection: trigger user intervention when track lost
            # ==============================
            any_lost = any(
                ctx.missing_frames[track_id] > ctx.max_missing_frames for track_id in [1, 2]
            )
            if any_lost and curr_state == TrackingState.TRACKING:
                lost_ids = [track_id for track_id in [1, 2]
                            if ctx.missing_frames[track_id] > ctx.max_missing_frames]
                print(f"  Frame {global_idx}: TRACK LOST for athlete(s) "
                      f"{lost_ids} (missing > {ctx.max_missing_frames} frames)")
                ctx.state_machine.state = TrackingState.RE_ID_MODE

                try:
                    new_boxes = _detect_and_request_boxes(
                        frame_bgr, global_idx, ctx.detection_callback,
                        ctx.detector, ctx.yolo_model, ctx.detection_threshold, ctx.device,
                    )
                except HumanVerificationSuspend:
                    ctx.human_suspend = True
                    break
                if new_boxes is not None:
                    ctx.detector = new_boxes[2]  # updated ctx.detector ref
                    new_box_a, new_box_b = new_boxes[0], new_boxes[1]
                    ctx.sam2_mgr.reset_state()
                    ctx.sam2_mgr.add_initial_box(batch_rel_idx, 1, new_box_a)
                    ctx.sam2_mgr.add_initial_box(batch_rel_idx, 2, new_box_b)
                    ctx.last_known_boxes[1] = new_box_a
                    ctx.last_known_boxes[2] = new_box_b
                    ctx.box_smoother.reset()
                    ctx.kpt_smoother.reset()
                    ctx.missing_frames = {1: 0, 2: 0}
                    ctx.initial_mask_areas.clear()
                    ctx.state_machine.set_tracking()
                    ctx.current_local = local_idx + ctx.prop_stride
                    break  # Restart propagation after user intervention

            # ==============================
            # Write output frame
            # ==============================
            _write_viz_frame(
                ctx.out, frame_bgr, frame_boxes, frame_masks,
                frame_kpts, frame_scores, frame_sources, ctx.display_map,
                curr_state, iou, global_idx, ctx.debug_dir, local_idx,
            )

            if ctx._json_file is not None and global_idx % ctx.frame_stride == 0:
                ctx._json_first_frame = _append_frame_to_json(
                    ctx._json_file, ctx._json_first_frame,
                    global_idx, local_idx, ctx.fps, curr_state, iou,
                    frame_boxes, frame_kpts, frame_scores,
                    frame_sources, ctx.display_map,
                )

            if ctx.frame_callback is not None and frame_boxes:
                try:
                    athletes = _build_athlete_dicts(
                        frame_boxes, frame_kpts, frame_scores,
                        frame_sources, ctx.display_map,
                    )
                    ctx.frame_callback(frame_bgr, global_idx, athletes)
                except Exception as _cb_err:
                    print(f"  [ctx.frame_callback] error at frame {global_idx}: {_cb_err}")

            # Track last-known boxes for batch carry-over
            if frame_boxes:
                for track_id, box in frame_boxes.items():
                    ctx.last_known_boxes[track_id] = box

            ctx.frames_processed += 1
            if ctx.frames_processed % 30 == 0:
                elapsed = time.time() - ctx.t_start
                pct = ctx.frames_processed / ctx.total_local * 100
                print(f"  [{pct:.0f}%] Frame {global_idx} | "
                      f"{curr_state.value} | "
                      f"Tracks: {len(frame_boxes)} | {elapsed:.0f}s")
                if ctx.should_stop is not None and ctx.should_stop():
                    ctx.user_cancelled = True
                    break
            if ctx.progress_callback is not None and (
                ctx.frames_processed == 1 or ctx.frames_processed % 30 == 0
            ):
                try:
                    ctx.progress_callback(ctx.frames_processed, ctx.total_local, global_idx)
                except Exception as e:
                    print(f"[tracking] ctx.progress_callback error (non-fatal): {e}")

            # Release per-frame objects to reduce GC pressure
            del frame_bgr, frame_rgb, masks_sam2
            del frame_boxes, frame_kpts, frame_scores, frame_masks, frame_sources

            # M4: advance ctx.current_local by ctx.prop_stride real frames.
            # local_idx = (batch_offset + batch_rel_idx) * ctx.prop_stride
            # So the next real frame position is local_idx + ctx.prop_stride.
            ctx.current_local = local_idx + ctx.prop_stride

            # Prune SAM2 memory every frame (one entry removed at a time,
            # constant cost). flush_cache=False defers MPS heap compaction to
            # batch boundaries — calling empty_cache mid-propagation on every
            # frame caused progressive MPS fragmentation spikes.
            ctx.sam2_mgr.prune_memory(ctx.max_history, flush_cache=False)

        if ctx.user_cancelled or ctx.human_suspend:
            break
        # End of propagation step: heavy cleanup at batch boundary only.
        # flush_cache=True triggers MPS empty_cache here (not mid-propagation).
        ctx.sam2_mgr.prune_memory(ctx.max_history, flush_cache=True)
        gc.collect()   # Expensive heap scan — safe here between batches
        empty_cache()
