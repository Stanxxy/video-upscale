"""
Unified pipeline worker: detect → verify → (track → detect)[inloop] → upscale → upload → publish.

Orchestrates the full job lifecycle. Progress is persisted to Keyspaces
via ``JobsStore`` (replacing the former WebSocket-based delivery).
During tracking, ``job_lifecycle.current_frame`` / ``total_frames`` are
**clip-global**: ``total_frames`` is the requested clip span
(``resolved_end − clip_start``), and ``current_frame`` is a 1-based count
of how far along that clip the tracker has advanced (using absolute
``frame_idx``), including after mid-track resume.
When human input is needed (bounding-box correction) the worker writes a
checkpoint, sets the job state to AWAITING_CORRECTION, and raises
``JobSuspendedError`` so models are released and the process can exit
cleanly.  A later ``/resume`` call re-creates the job with the corrected
boxes.
"""
import asyncio
import json
import logging
import os
import re
import shutil
import time
from uuid import uuid4

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import (
    END_OF_TRACKING_SENTINEL,
    WorkerStateSnapshot,
    build_annotate_completed,
    build_detect_initial_pending,
    build_download_completed,
    build_publish_completed,
    build_tracking_started,
    build_track_completed,
    build_track_mid_loss,
    build_track_progress,
    build_upload_incremental,
    build_upscale_started,
    build_upscale_window_progress,
    build_verified_boxes_checkpoint,
    should_flush_analysis,
)
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.models import JobCancelledError, JobSuspendedError, JobStatus, TrackRequest
from tracking_pipeline.human_verification_suspend import HumanVerificationSuspend
from service.s3 import S3Client
from service.sns import SNSPublisher
from service.tracking_chain_merge import consolidate_tracking_json_with_job_chain

from pipeline import deduplicate_clips

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU / model cleanup
# ---------------------------------------------------------------------------

PARTIAL_UPLOAD_INTERVAL = 30.0
LIFECYCLE_HEARTBEAT_INTERVAL = 1.0


def _tracking_progress_flags(
    now: float,
    last_ks_write: float,
    last_partial_upload: float,
    *,
    ks_interval: float = LIFECYCLE_HEARTBEAT_INTERVAL,
    partial_interval: float = PARTIAL_UPLOAD_INTERVAL,
) -> tuple[bool, bool]:
    """Decide whether the tracking-progress callback should write to the
    lifecycle row (1s cadence) and/or upload a partial-tracking checkpoint
    (30s cadence). Pure function — easy to unit test without a fake clock."""
    return (
        (now - last_ks_write) >= ks_interval,
        (now - last_partial_upload) >= partial_interval,
    )


def _make_worker_state(
    *,
    progress_percent: float,
    current_frame: int = 0,
    total_frames: int = 0,
    stage_progress_fraction: float = 0.0,
) -> WorkerStateSnapshot:
    """Snapshot the in-memory worker progress for a checkpoint write."""
    return WorkerStateSnapshot(
        progress_percent=progress_percent,
        current_frame=current_frame,
        total_frames=total_frames,
        stage_progress_fraction=stage_progress_fraction,
    )


def _pct_at_least(pct: float, floor: float) -> float:
    """Replacement jobs seed lifecycle progress — never regress below ``floor``."""
    return max(pct, floor)


def _video_frame_cap(video_path: str) -> int:
    """Total frames reported by OpenCV (at least 1)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        return max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 1)
    finally:
        cap.release()


def _resolved_clip_end_and_total(
    clip_start_frame: int,
    end_frame: int | None,
    video_frame_cap: int,
) -> tuple[int, int]:
    """Return ``(clip_end_resolved, clip_total_frames)`` for the requested clip."""
    raw_end = end_frame if end_frame is not None else video_frame_cap
    clip_end = min(max(raw_end, clip_start_frame + 1), video_frame_cap)
    clip_total = max(clip_end - clip_start_frame, 1)
    return clip_end, clip_total


def _clip_done_inclusive_through_global(
    global_idx: int,
    clip_start_frame: int,
    clip_total_frames: int,
) -> int:
    """1-based count of clip frames from ``clip_start`` through ``global_idx`` inclusive."""
    return min(max(global_idx - clip_start_frame + 1, 0), clip_total_frames)


def _tracking_progress_pct_clip(
    global_idx: int,
    clip_start_frame: int,
    clip_total_frames: int,
    progress_floor: float,
) -> tuple[int, float]:
    """Return (clip_done_1based, tracking-stage percent) for Keyspaces/UI."""
    done = _clip_done_inclusive_through_global(
        global_idx, clip_start_frame, clip_total_frames,
    )
    frac = done / max(clip_total_frames, 1)
    pct = _pct_at_least(15.0 + frac * 40.0, progress_floor)
    return done, pct


def _last_global_frame_idx_from_tracking(track_data: dict) -> int | None:
    frames = track_data.get("frames") or []
    if not frames:
        return None
    last = frames[-1]
    if "frame_idx" in last:
        return int(last["frame_idx"])
    if "frame" in last:
        return int(last["frame"])
    return None


def _track_completed_clip_worker_state(
    track_data: dict,
    clip_start_frame: int,
    clip_total_frames: int,
    *,
    progress_percent: float,
) -> WorkerStateSnapshot:
    last_g = _last_global_frame_idx_from_tracking(track_data)
    if last_g is not None:
        done = _clip_done_inclusive_through_global(
            last_g, clip_start_frame, clip_total_frames,
        )
    else:
        done = clip_total_frames
    return _make_worker_state(
        progress_percent=progress_percent,
        current_frame=done,
        total_frames=clip_total_frames,
        stage_progress_fraction=(done / max(clip_total_frames, 1)),
    )


def _ensure_models_released():
    """Release all ML models from GPU/RAM."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Partial-tracking JSON loader (tolerates truncated streaming output)
# ---------------------------------------------------------------------------

def _load_partial_tracking_dict(partial_path: str) -> dict:
    """Parse tracking.json; tolerate truncated streaming output from abrupt cancellation."""
    with open(partial_path, encoding="utf-8") as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r",\s*\"frames\"\s*:\s*\[", content)
    if not m:
        raise json.JSONDecodeError("no frames array in partial tracking", content, 0)
    header_raw = content[: m.start()] + "}"
    header = json.loads(header_raw)
    tail = content[m.end() :]
    dec = json.JSONDecoder()
    frames = []
    pos = 0
    n = len(tail)
    while pos < n:
        while pos < n and tail[pos] in " \t\n\r,":
            pos += 1
        if pos >= n:
            break
        try:
            obj, end = dec.raw_decode(tail, pos)
            frames.append(obj)
            pos = end
        except json.JSONDecodeError:
            break
    return {**header, "frames": frames}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_job(
    job_id: str,
    request: TrackRequest,
    config: ServiceConfig,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
):
    """Run the full tracking + upscale pipeline for a single job."""
    # TODO: encapsulate the logic of each stage to a method.
    work_dir = os.path.join(config.temp_dir, job_id)
    os.makedirs(work_dir, exist_ok=True)
    loop = asyncio.get_event_loop()
    clip_start_frame = 0
    clip_end_resolved = 1
    clip_total_frames = 1
    logger.info(
        "Job %s: worker starting work_dir=%s input=s3://%s/%s skip_upscale=%s",
        job_id,
        work_dir,
        request.bucket,
        request.key,
        request.skip_upscale,
    )

    try:
        # Mark job as running in Keyspaces (set_state is authoritative for state)
        await jobs_store.set_state(job_id, JobState.RUNNING)

        _lifecycle_row = await jobs_store.get_lifecycle(job_id)
        progress_floor = float(
            (_lifecycle_row or {}).get("progress_percent") or 0.0,
        )

        # ============================================================
        # STAGE 1: DOWNLOAD (0-10%) [for both new and resumed jobs]
        # ============================================================
        logger.info("Job %s: stage download (0-10%%)", job_id)
        await job_store.update_job(
            job_id,
            status=JobStatus.DOWNLOADING,
            progress_percent=_pct_at_least(2.0, progress_floor),
        )
        await jobs_store.update_progress(
            job_id, PipelineStage.DOWNLOAD,
            _pct_at_least(2.0, progress_floor),
        )

        s3: S3Client = _make_s3(config)
        s3.ensure_bucket(request.bucket)

        # TODO: video path should be unique for each video and stay on disk for 24h. 
        # This is to help resume the job from the checkpoint. 
        video_path = await loop.run_in_executor(
            None,
            s3.download_file,
            request.bucket,
            request.key,
            os.path.join(work_dir, "video.mp4"), # TODO: video path should be saved in the checkpoint.  
        )

        dl_pct = _pct_at_least(10.0, progress_floor)
        await job_store.update_job(job_id, progress_percent=dl_pct)
        await jobs_store.update_progress(job_id, PipelineStage.DOWNLOAD, dl_pct)
        await jobs_store.write_checkpoint(
            job_id, PipelineStage.DOWNLOAD, False,
            build_download_completed(
                worker_state=_make_worker_state(
                    progress_percent=dl_pct, stage_progress_fraction=1.0,
                ),
            ),
        )
        logger.info("Job %s: download finished local_path=%s", job_id, video_path)

        if _is_cancelled(job_id, job_store):
            return

        # Parse time range BEFORE detection so clip bounds are available
        clip_start_frame, end_frame = _parse_time_range(
            video_path, request.start_time, request.end_time,
        )
        logger.info(
            "Job %s: time range clip_start_frame=%d end_frame=%s",
            job_id,
            clip_start_frame,
            end_frame if end_frame is not None else "full_video",
        )

        _vid_frame_cap = await loop.run_in_executor(None, _video_frame_cap, video_path)
        clip_end_resolved, clip_total_frames = _resolved_clip_end_and_total(
            clip_start_frame, end_frame, _vid_frame_cap,
        )
        logger.info(
            "Job %s: clip_end_resolved=%d clip_total_frames=%d (video_cap=%d)",
            job_id,
            clip_end_resolved,
            clip_total_frames,
            _vid_frame_cap,
        )

        # SAM2 resumes mid-clip from checkpoint; annotate/upscale clip logic uses clip_start_frame.
        tracking_start_frame = clip_start_frame
        if (
            request.resume_from_frame is not None
            and request.resume_from_frame != END_OF_TRACKING_SENTINEL
        ):
            tracking_start_frame = request.resume_from_frame
            logger.info(
                "Job %s: resuming tracking from frame %d",
                job_id,
                tracking_start_frame,
            )

        # ============================================================
        # STAGE 2: DETECT + HUMAN VERIFY (10-15%)
        # ============================================================
        logger.info("Job %s: stage detect/verify (10-15%%)", job_id)
        if request.box_a is not None and request.box_b is not None:
            # Boxes provided in request -- skip detection entirely
            box_a = request.box_a
            box_b = request.box_b
            logger.info(
                "Job %s using provided boxes (skipping detection): "
                "box_a=%s box_b=%s",
                job_id, box_a, box_b,
            )
            det_pct = _pct_at_least(15.0, progress_floor)
            det_cf = _clip_done_inclusive_through_global(
                tracking_start_frame, clip_start_frame, clip_total_frames,
            )
            await job_store.update_job(
                job_id, status=JobStatus.DETECTING, progress_percent=det_pct,
                current_frame=det_cf, total_frames=clip_total_frames,
            )
            await jobs_store.update_progress(
                job_id, PipelineStage.DETECT, det_pct,
                current_frame=det_cf, total_frames=clip_total_frames,
            )
            await jobs_store.write_checkpoint(
                job_id, PipelineStage.DETECT, False,
                build_verified_boxes_checkpoint(
                    box_a,
                    box_b,
                    PipelineStage.DETECT,
                    worker_state=_make_worker_state(
                        progress_percent=det_pct,
                        current_frame=det_cf,
                        total_frames=clip_total_frames,
                        stage_progress_fraction=1.0,
                    ),
                ),
            )
        else:
            det_pct_lo = _pct_at_least(10.0, progress_floor)
            await job_store.update_job(
                job_id, status=JobStatus.DETECTING, progress_percent=det_pct_lo,
            )
            await jobs_store.update_progress(
                job_id, PipelineStage.DETECT, det_pct_lo,
            )

            # Import heavy ML deps lazily
            # TODO: the dependency should be imported only once at the start of vision engine.
            # Do not import it here. Also encapsulate the detection logic to a method.
            from service.tracking_runner import run_detect, capture_frame_jpeg

            # Detect at the start frame (first frame of the requested range)
            frame_idx = tracking_start_frame
            candidates = await loop.run_in_executor(
                None,
                lambda: run_detect(
                    video_path,
                    frame_idx=frame_idx,
                    threshold=request.detection_threshold,
                    yolo_model=request.yolo_model,
                ),
            )

            frame_jpeg = await loop.run_in_executor(
                None, lambda: capture_frame_jpeg(video_path, frame_idx),
            )
            if frame_jpeg is None:
                raise RuntimeError(f"Failed to read frame {frame_idx} from video")

            # Optional Gemini hint if >2 candidates
            suggested_boxes = None
            if len(candidates) > 2:
                try:
                    from service.vllm_selector import suggest_athletes
                    suggested_boxes = await suggest_athletes(
                        frame_jpeg, candidates, config,
                    )
                except Exception as e:
                    logger.warning("Athlete suggestion (Gemini) failed: %s", e)

            # Upload detection frame to S3 for human review
            frame_s3_key = f"checkpoints/{job_id}/frame_{frame_idx}.jpg"
            await loop.run_in_executor(
                None,
                lambda: s3.put_object(
                    request.bucket,
                    frame_s3_key,
                    frame_jpeg,
                    "image/jpeg",
                ),
            )

            # Write checkpoint to Keyspaces and suspend
            det_cf0 = _clip_done_inclusive_through_global(
                frame_idx, clip_start_frame, clip_total_frames,
            )
            await jobs_store.write_checkpoint(
                job_id, PipelineStage.DETECT, False,
                build_detect_initial_pending(
                    frame_idx=frame_idx,
                    frame_s3_key=frame_s3_key,
                    frame_bucket=request.bucket,
                    candidates=candidates,
                    suggested_boxes=suggested_boxes,
                    worker_state=_make_worker_state(
                        progress_percent=det_pct_lo,
                        current_frame=det_cf0,
                        total_frames=clip_total_frames,
                        stage_progress_fraction=0.0,
                    ),
                ),
            )
            await jobs_store.set_state(job_id, JobState.AWAITING_CORRECTION)
            raise JobSuspendedError("Awaiting initial detection verification")

        if _is_cancelled(job_id, job_store):
            return

        # ============================================================
        # STAGE 3: TRACKING (15-55%)
        # ============================================================
        logger.info("Job %s: stage tracking (15-55%%)", job_id)
        track_stage_pct = _pct_at_least(15.0, progress_floor)
        track_started_cf = min(
            max(tracking_start_frame - clip_start_frame + 1, 1),
            clip_total_frames,
        )
        await job_store.update_job(
            job_id, status=JobStatus.TRACKING, progress_percent=track_stage_pct,
            current_frame=track_started_cf,
            total_frames=clip_total_frames,
        )
        await jobs_store.update_progress(
            job_id, PipelineStage.TRACK, track_stage_pct,
            current_frame=track_started_cf,
            total_frames=clip_total_frames,
        )

        from service.tracking_runner import run_tracking_job

        tracking_output_dir = os.path.join(work_dir, "tracking")
        os.makedirs(tracking_output_dir, exist_ok=True)

        skip_tracking_runner = (
            request.resume_from_frame == END_OF_TRACKING_SENTINEL
            and bool(request.resume_tracking_s3_key)
        )

        partial_tracking_data = None
        if request.resume_tracking_s3_key and not skip_tracking_runner:
            logger.info(
                "Job %s: downloading partial tracking from %s",
                job_id,
                request.resume_tracking_s3_key,
            )
            partial_bucket = request.output_bucket or request.bucket
            partial_tracking_data = await loop.run_in_executor(
                None,
                s3.download_json,
                partial_bucket,
                request.resume_tracking_s3_key,
            )
            logger.info(
                "Job %s: partial tracking has %d frames",
                job_id,
                len(partial_tracking_data.get("frames", [])),
            )

        tracking_json_path: str | None = None
        if skip_tracking_runner:
            logger.info(
                "Job %s: skipping run_tracking_job — loading tracking JSON from %s",
                job_id,
                request.resume_tracking_s3_key,
            )
            load_bucket = request.output_bucket or request.bucket
            full_tracking = await loop.run_in_executor(
                None,
                s3.download_json,
                load_bucket,
                request.resume_tracking_s3_key,
            )
            tracking_json_path = os.path.join(tracking_output_dir, "tracking.json")
            with open(tracking_json_path, "w") as f:
                json.dump(full_tracking, f)
        else:
            # Build sync progress callback -> async Keyspaces (two cadences):
            #   1-second  → lifecycle row update (heartbeat)
            #   30-second → partial-tracking S3 upload + V1 track_progress checkpoint
            #   immediate → first partial checkpoint once at least one frame exists
            _last_ks_write = 0.0
            _last_partial_upload = 0.0
            _first_partial_upload = False

            def tracking_progress_cb(frames_done: int, total: int, global_idx: int):
                nonlocal _last_ks_write, _last_partial_upload, _first_partial_upload
                clip_done, pct = _tracking_progress_pct_clip(
                    global_idx, clip_start_frame, clip_total_frames, progress_floor,
                )
                resume_next_global = global_idx + 1
                now = time.monotonic()
                write_lifecycle, upload_partial = _tracking_progress_flags(
                    now, _last_ks_write, _last_partial_upload,
                )
                if frames_done >= 1 and not _first_partial_upload:
                    upload_partial = True
                    _first_partial_upload = True
                if write_lifecycle:
                    _last_ks_write = now
                if upload_partial:
                    _last_partial_upload = now
                asyncio.run_coroutine_threadsafe(
                    _update_tracking_progress_with_partial(
                        job_id, clip_done, clip_total_frames, pct,
                        job_store, jobs_store,
                        request, work_dir, s3,
                        resume_next_global=resume_next_global,
                        write_lifecycle=write_lifecycle,
                        upload_partial=upload_partial,
                    ),
                    loop,
                )

            detection_cb = _make_detection_cb(
                job_id, loop, jobs_store, s3, config, request, work_dir,
                clip_start_frame=clip_start_frame,
                clip_total_frames=clip_total_frames,
                progress_floor=progress_floor,
            )
            eff_step_size = request.step_size or config.tracking_step_size
            eff_max_history = request.max_history or config.tracking_max_history
            eff_max_missing_frames = (
                request.max_missing_frames
                if request.max_missing_frames is not None
                else config.tracking_max_missing_frames
            )
            logger.info(
                "Job %s tracking config: step_size=%s max_history=%s "
                "max_missing_frames=%s",
                job_id,
                eff_step_size,
                eff_max_history,
                eff_max_missing_frames,
            )

            await jobs_store.write_checkpoint(
                job_id, PipelineStage.TRACK, False,
                build_tracking_started(
                    clip_start_frame=clip_start_frame,
                    clip_end_frame=clip_end_resolved,
                    worker_state=_make_worker_state(
                        progress_percent=track_stage_pct,
                        current_frame=track_started_cf,
                        total_frames=clip_total_frames,
                        stage_progress_fraction=(
                            track_started_cf / max(clip_total_frames, 1)
                        ),
                    ),
                ),
            )

            tracking_json_path = await loop.run_in_executor(
                None,
                lambda: run_tracking_job(
                    video_path,
                    box_a,
                    box_b,
                    tracking_output_dir,
                    sam2_model_id=request.sam2_model,
                    yolo_model=request.yolo_model,
                    detection_threshold=request.detection_threshold,
                    start_frame=tracking_start_frame,
                    end_frame=end_frame,
                    step_size=eff_step_size,
                    max_history=eff_max_history,
                    max_missing_frames=eff_max_missing_frames,
                    progress_cb=tracking_progress_cb,
                    detection_cb=detection_cb,
                    should_stop=lambda: job_store.is_cancelled(job_id),
                ),
            )
        logger.info(
            "Job %s: tracking executor returned path=%s",
            job_id,
            tracking_json_path,
        )

        # This is for mid-track persistance. should be saved with checkpoint.
        # ==============================================================
        # If tracking_json_path is None, the detection_cb triggered a suspend
        if tracking_json_path is None:
            raise JobSuspendedError("Awaiting mid-tracking detection correction")

        # Phase 3: Merge partial tracking with new tracking
        if partial_tracking_data:
            with open(tracking_json_path) as f:
                new_tracking = json.load(f)
            merged_frames = partial_tracking_data.get("frames", []) + new_tracking.get("frames", [])
            merged = {
                **new_tracking,
                "frames": merged_frames,
                "start_frame": partial_tracking_data.get(
                    "start_frame", clip_start_frame,
                ),
            }
            with open(tracking_json_path, "w") as f:
                json.dump(merged, f)
            logger.info("Job %s: merged %d partial + %d new = %d total frames",
                        job_id, len(partial_tracking_data.get("frames", [])),
                        len(new_tracking.get("frames", [])), len(merged_frames))

        await consolidate_tracking_json_with_job_chain(
            jobs_store=jobs_store,
            s3=s3,
            bucket=request.output_bucket or request.bucket,
            leaf_job_id=job_id,
            local_tracking_json_path=tracking_json_path,
            clip_start_frame=clip_start_frame,
        )

        # Write completed tracking checkpoint (pre-upload; the post-upload
        # re-write in the upload stage adds artifacts.tracking_s3_key).
        with open(tracking_json_path) as _f:
            _track_data = json.load(_f)
        _frame_count = len(_track_data.get("frames", []))
        await jobs_store.write_checkpoint(
            job_id, PipelineStage.TRACK, False,
            build_track_completed(
                start_frame=clip_start_frame,
                frame_count=_frame_count,
                worker_state=_track_completed_clip_worker_state(
                    _track_data, clip_start_frame, clip_total_frames,
                    progress_percent=55.0,
                ),
            ),
        )

        # ==============================================================
        # TRACKING-ONLY SHORT-CIRCUIT (skip_upscale=True)
        # Upload tracking results to S3 then mark completed.
        # only used in QA client.
        # TODO: encapsulate the logic of this short-circuit to a method.
        # ==============================================================
        if request.skip_upscale:
            await job_store.update_job(
                job_id, status=JobStatus.UPLOADING, progress_percent=80.0,
            )
            await jobs_store.update_progress(job_id, PipelineStage.UPLOAD, 80.0)

            output_bucket = request.output_bucket or request.bucket
            base_key = os.path.splitext(request.key)[0]
            tracking_result_key = f"{base_key}_tracked.json"
            tracked_video_key = f"{base_key}_tracked.mp4"

            s3.ensure_bucket(output_bucket)

            # Upload tracking JSON
            with open(tracking_json_path) as f:
                tracking_data = json.load(f)
            if request.resume_existing_upload_tracking_key != tracking_result_key:
                await loop.run_in_executor(
                    None, s3.upload_json, tracking_data,
                    output_bucket, tracking_result_key,
                )

            # Re-write the track row now that we know the durable key.
            await jobs_store.write_checkpoint(
                job_id, PipelineStage.TRACK, False,
                build_track_completed(
                    start_frame=clip_start_frame,
                    frame_count=_frame_count,
                    tracking_s3_key=tracking_result_key,
                    worker_state=_track_completed_clip_worker_state(
                        tracking_data, clip_start_frame, clip_total_frames,
                        progress_percent=55.0,
                    ),
                ),
            )

            # Upload tracked video (if it exists)
            tracked_video = os.path.join(
                tracking_output_dir, "tracked_output.mp4",
            )
            if os.path.isfile(tracked_video):
                await loop.run_in_executor(
                    None, s3.upload_file, tracked_video,
                    output_bucket, tracked_video_key, "video/mp4",
                )

            # Terminal write for the skip_upscale path: the upload row carries
            # completed=True because there is no publish stage in this branch.
            await jobs_store.write_checkpoint(
                job_id, PipelineStage.UPLOAD, True,
                build_upload_incremental(
                    tracking_s3_key=tracking_result_key,
                    worker_state=_make_worker_state(
                        progress_percent=100.0, stage_progress_fraction=1.0,
                    ),
                ),
            )

            await job_store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress_percent=100.0,
                result_bucket=output_bucket,
                result_key=tracking_result_key,
            )
            await jobs_store.set_state(job_id, JobState.COMPLETED)
            logger.info(
                "Job %s completed (tracking only -> s3://%s/%s)",
                job_id, output_bucket, tracking_result_key,
            )
            return

        # ============================================================
        # Pre-upscale: upload tracking JSON so the durable input is in S3
        # before the upscale stage starts. The upscale checkpoint contract
        # requires artifacts.tracking_s3_key, so this upload must happen
        # before _run_upscale_analysis writes its first checkpoint.
        # ============================================================
        output_bucket = request.output_bucket or request.bucket
        base_key = os.path.splitext(request.key)[0]
        tracking_result_key = f"{base_key}_tracked.json"
        analysis_result_key = f"{base_key}_analysis.json"
        annotated_video_key = f"{base_key}_annotated.mp4"

        s3.ensure_bucket(output_bucket)
        with open(tracking_json_path) as f:
            _full_tracking = json.load(f)
        if request.resume_existing_upload_tracking_key == tracking_result_key:
            logger.info(
                "Job %s: skipping tracking JSON re-upload (key already durable)",
                job_id,
            )
        else:
            await loop.run_in_executor(
                None, s3.upload_json, _full_tracking,
                output_bucket, tracking_result_key,
            )

        # Re-write the track row with the durable tracking_s3_key.
        await jobs_store.write_checkpoint(
            job_id, PipelineStage.TRACK, False,
            build_track_completed(
                start_frame=clip_start_frame,
                frame_count=_frame_count,
                tracking_s3_key=tracking_result_key,
                worker_state=_track_completed_clip_worker_state(
                    _full_tracking, clip_start_frame, clip_total_frames,
                    progress_percent=55.0,
                ),
            ),
        )

        # First upload row (additive on artifacts as later artifacts land).
        await jobs_store.write_checkpoint(
            job_id, PipelineStage.UPLOAD, False,
            build_upload_incremental(
                tracking_s3_key=tracking_result_key,
                worker_state=_make_worker_state(
                    progress_percent=55.0, stage_progress_fraction=0.33,
                ),
            ),
        )

        # ============================================================
        # STAGE 4: UPSCALE + ANALYSIS -- second pass (55-80%)
        # ============================================================
        logger.info("Job %s: stage upscale+analysis (55-80%%)", job_id)
        await job_store.update_job(
            job_id, status=JobStatus.UPSCALING, progress_percent=55.0,
        )
        await jobs_store.update_progress(job_id, PipelineStage.UPSCALE_ANALYZE, 55.0)

        _last_upscale_ks_write = 0.0

        async def _update_upscale_progress(pct: float):
            nonlocal _last_upscale_ks_write
            await job_store.update_job(job_id, progress_percent=pct)
            now = time.monotonic()
            if (now - _last_upscale_ks_write) >= 1.0:
                _last_upscale_ks_write = now
                await jobs_store.update_progress(job_id, PipelineStage.UPSCALE_ANALYZE, pct)

        def _upscale_progress(pct_within_stage: float):
            overall = 55.0 + pct_within_stage * 25.0  # 55%-80%
            asyncio.run_coroutine_threadsafe(
                _update_upscale_progress(overall),
                loop,
            )

        analysis_result, fps = await loop.run_in_executor(
            None,
            lambda: _run_upscale_analysis(
                video_path, tracking_json_path, config, request, work_dir,
                job_id=job_id,
                jobs_store=jobs_store,
                loop=loop,
                tracking_s3_key=tracking_result_key,
                progress_cb=_upscale_progress,
            ),
        )
        logger.info(
            "Job %s: upscale+analysis finished fps=%s has_analysis=%s",
            job_id,
            fps,
            analysis_result is not None,
        )

        # ============================================================
        # STAGE 4.5: ANNOTATE VIDEO (80-85%)
        # TODO: annotate the video in a coroutine and dont block the upload.
        # ============================================================
        logger.info("Job %s: stage annotate (80-85%%)", job_id)
        annotated_video_path = None
        tracked_video_path = os.path.join(tracking_output_dir, "tracked_output.mp4")

        if analysis_result and os.path.isfile(tracked_video_path):
            await job_store.update_job(job_id, progress_percent=80.0)
            await jobs_store.update_progress(job_id, PipelineStage.ANNOTATE, 80.0)

            from service.video_annotator import annotate_video
            annotated_path = os.path.join(work_dir, "annotated_output.mp4")
            try:
                annotated_video_path = await loop.run_in_executor(
                    None,
                    lambda: annotate_video(
                        tracked_video_path, analysis_result,
                        annotated_path, fps, clip_start_frame,
                    ),
                )
            except Exception as e:
                logger.warning("Video annotation failed (non-fatal): %s", e)
        elif os.path.isfile(tracked_video_path):
            # No analysis but tracked video exists -- upload the tracked video
            annotated_video_path = tracked_video_path

        await job_store.update_job(job_id, progress_percent=85.0)
        await jobs_store.update_progress(job_id, PipelineStage.ANNOTATE, 85.0)

        if _is_cancelled(job_id, job_store):
            return

        # ============================================================
        # STAGE 5: UPLOAD (85-90%)
        # ============================================================
        logger.info(
            "Job %s: stage upload (85-90%%) bucket=%s",
            job_id,
            output_bucket,
        )
        await job_store.update_job(
            job_id, status=JobStatus.UPLOADING, progress_percent=85.0,
        )
        await jobs_store.update_progress(job_id, PipelineStage.UPLOAD, 85.0)

        s3.ensure_bucket(output_bucket)

        # Tracking JSON was already uploaded before the upscale stage so the
        # upscale_analyze checkpoint could record artifacts.tracking_s3_key.

        # Upload analysis JSON (if upscaling was done) and update upload row.
        analysis_uploaded_key: str | None = None
        if analysis_result is not None:
            if request.resume_existing_upload_analysis_key == analysis_result_key:
                logger.info(
                    "Job %s: skipping analysis JSON re-upload (recovery)",
                    job_id,
                )
                analysis_uploaded_key = analysis_result_key
            else:
                await loop.run_in_executor(
                    None,
                    s3.upload_json,
                    analysis_result,
                    output_bucket,
                    analysis_result_key,
                )
                analysis_uploaded_key = analysis_result_key
            await jobs_store.write_checkpoint(
                job_id, PipelineStage.UPLOAD, False,
                build_upload_incremental(
                    tracking_s3_key=tracking_result_key,
                    analysis_s3_key=analysis_uploaded_key,
                    worker_state=_make_worker_state(
                        progress_percent=88.3, stage_progress_fraction=0.66,
                    ),
                ),
            )

        # Upload annotated video and update upload row + write annotate row.
        annotated_uploaded_key: str | None = None
        if annotated_video_path and os.path.isfile(annotated_video_path):
            if request.resume_existing_upload_annotated_key == annotated_video_key:
                logger.info(
                    "Job %s: skipping annotated video re-upload (recovery)",
                    job_id,
                )
                annotated_uploaded_key = annotated_video_key
            else:
                await loop.run_in_executor(
                    None,
                    s3.upload_file,
                    annotated_video_path,
                    output_bucket,
                    annotated_video_key,
                    "video/mp4",
                )
                annotated_uploaded_key = annotated_video_key
            await jobs_store.write_checkpoint(
                job_id, PipelineStage.UPLOAD, False,
                build_upload_incremental(
                    tracking_s3_key=tracking_result_key,
                    analysis_s3_key=analysis_uploaded_key,
                    annotated_video_s3_key=annotated_uploaded_key,
                    worker_state=_make_worker_state(
                        progress_percent=90.0, stage_progress_fraction=1.0,
                    ),
                ),
            )

        # Annotate stage's checkpoint reflects the durable result of the
        # annotation step — empty artifacts if annotation failed/skipped.
        await jobs_store.write_checkpoint(
            job_id, PipelineStage.ANNOTATE, False,
            build_annotate_completed(
                annotated_video_s3_key=annotated_uploaded_key,
                worker_state=_make_worker_state(
                    progress_percent=85.0, stage_progress_fraction=1.0,
                ),
            ),
        )

        await job_store.update_job(job_id, progress_percent=90.0)
        await jobs_store.update_progress(job_id, PipelineStage.UPLOAD, 90.0)

        if _is_cancelled(job_id, job_store):
            return

        # ============================================================
        # STAGE 6: PUBLISH SNS (90-95%) -- optional
        # ============================================================
        topic_arn = request.sns_topic_arn or config.sns_topic_arn
        event_count = 0
        sns_completion_sent = False
        if (
            topic_arn
            and analysis_result is not None
            and not request.resume_terminal_publish_done
        ):
            logger.info("Job %s: stage SNS publish topic configured", job_id)
            try:
                await job_store.update_job(
                    job_id, status=JobStatus.PUBLISHING, progress_percent=92.0,
                )
                await jobs_store.update_progress(job_id, PipelineStage.PUBLISH, 92.0)

                video_id = request.video_id or uuid4()
                sns = SNSPublisher(
                    config.aws_region,
                    topic_arn,
                    endpoint_url=config.s3_endpoint_url or None,
                    access_key_id=config.aws_access_key_id or None,
                    secret_access_key=config.aws_secret_access_key or None,
                )
                result_uri = f"s3://{output_bucket}/{analysis_result_key}"
                event_count = sns.publish_events(
                    analysis_result, video_id, fps,
                    job_id=job_id, result_s3_uri=result_uri,
                )
                sns_completion_sent = True
            except Exception as e:
                logger.warning("SNS publish failed (non-fatal): %s", e)
        else:
            if request.resume_terminal_publish_done:
                logger.info(
                    "Job %s: SNS skipped — resume_terminal_publish_done set",
                    job_id,
                )
            else:
                logger.info(
                    "Job %s: stage SNS skipped (topic_configured=%s has_analysis=%s)",
                    job_id,
                    bool(topic_arn),
                    analysis_result is not None,
                )

        # Terminal publish row — completed=True closes out the job pipeline.
        await jobs_store.write_checkpoint(
            job_id, PipelineStage.PUBLISH, True,
            build_publish_completed(
                sns_topic_arn=topic_arn or "",
                sns_event_count=event_count,
                sns_completion_sent=sns_completion_sent,
                worker_state=_make_worker_state(
                    progress_percent=100.0, stage_progress_fraction=1.0,
                ),
            ),
        )

        # ============================================================
        # COMPLETED
        # ============================================================
        result_key = analysis_result_key if analysis_result else tracking_result_key
        await job_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress_percent=100.0,
            result_bucket=output_bucket,
            result_key=result_key,
        )
        await jobs_store.set_state(job_id, JobState.COMPLETED)
        logger.info("Job %s completed (%d SNS events published)", job_id, event_count)

    except asyncio.CancelledError:
        logger.info("Job %s cancelled (client disconnected)", job_id)
        await job_store.update_job(job_id, status=JobStatus.CANCELLED)
        await jobs_store.set_state(job_id, JobState.CANCELLED)

    except JobCancelledError:
        # Only mark CANCELLED if not already AWAITING_CORRECTION (mid-track suspend)
        lifecycle = await jobs_store.get_lifecycle(job_id)
        if lifecycle and lifecycle.get("job_state") == JobState.AWAITING_CORRECTION.value:
            logger.info("Job %s: JobCancelledError after mid-track suspend — keeping AWAITING_CORRECTION", job_id)
        else:
            logger.info("Job %s cancelled (DELETE requested)", job_id)
            await job_store.update_job(job_id, status=JobStatus.CANCELLED)
            await jobs_store.set_state(job_id, JobState.CANCELLED)

    except JobSuspendedError as e:
        logger.info("Job %s suspended: %s", job_id, e)
        # State already set to AWAITING_CORRECTION by the suspend code.
        # Don't send error -- this is a clean suspension.

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        await job_store.update_job(
            job_id, status=JobStatus.FAILED, error_message=str(e),
        )
        await jobs_store.set_state(job_id, JobState.FAILED, error_message=str(e))

    finally:
        # Release GPU/CPU resources
        # TODO: only clean the data but keep the model in GPU/CPU. Let the worker to be 
        # ready for next job.
        _ensure_models_released()

        # Save partial tracking on FAILED/CANCELLED only — AWAITING_CORRECTION
        # suspend already persisted partial tracking via detection_cb.
        try:
            ks_lc = await jobs_store.get_lifecycle(job_id)
            ks_state = ks_lc.get("job_state") if ks_lc else None
            job_status = await job_store.get_job(job_id)
            partial_tracking = os.path.join(work_dir, "tracking", "tracking.json")
            _s3 = locals().get("s3")
            if (
                _s3 is not None
                and ks_state != JobState.AWAITING_CORRECTION.value
                and job_status
                and job_status.status in (JobStatus.FAILED, JobStatus.CANCELLED)
                and os.path.isfile(partial_tracking)
                and os.path.getsize(partial_tracking) > 100
            ):
                partial_key = f"checkpoints/{job_id}/partial_tracking.json"
                bucket = request.output_bucket or request.bucket
                partial_data = _load_partial_tracking_dict(partial_tracking)
                await loop.run_in_executor(
                    None, _s3.upload_json, partial_data, bucket, partial_key,
                )
                logger.info(
                    "Job %s: saved partial tracking to s3://%s/%s",
                    job_id, bucket, partial_key,
                )
                frames = partial_data.get("frames") or []
                if frames:
                    last_g = int(frames[-1].get("frame_idx", len(frames) - 1))
                else:
                    last_g = clip_start_frame - 1
                resume_frm = last_g + 1
                clip_done_fail = _clip_done_inclusive_through_global(
                    last_g, clip_start_frame, clip_total_frames,
                )
                pct_fail = float(
                    ks_lc.get("progress_percent", 0.0) if ks_lc else 0.0,
                )
                await jobs_store.write_checkpoint(
                    job_id, PipelineStage.TRACK, False,
                    build_track_progress(
                        partial_tracking_s3_key=partial_key,
                        resume_from_frame=resume_frm,
                        worker_state=_make_worker_state(
                            progress_percent=pct_fail,
                            current_frame=clip_done_fail,
                            total_frames=clip_total_frames,
                            stage_progress_fraction=(
                                clip_done_fail / max(clip_total_frames, 1)
                            ),
                        ),
                    ),
                )
        except Exception as e:
            logger.warning("Job %s: failed to save partial tracking: %s", job_id, e)

        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_s3(config: ServiceConfig) -> S3Client:
    return S3Client(
        region=config.aws_region,
        endpoint_url=config.s3_endpoint_url or None,
        access_key_id=config.aws_access_key_id or None,
        secret_access_key=config.aws_secret_access_key or None,
    )


def _is_cancelled(job_id: str, job_store: InMemoryJobStore) -> bool:
    if job_store.is_cancelled(job_id):
        logger.info("Job %s cancelled", job_id)
        return True
    return False


async def _flush_analysis_checkpoint(
    *,
    job_id: str,
    jobs_store: JobsStore,
    s3: S3Client,
    output_bucket: str,
    output_dir: str,
    tracking_s3_key: str,
    analysis_results: list[dict],
    current_context: str,
    next_frame_idx: int,
    progress_percent: float,
    total_tracking_frames: int,
    stage_progress_fraction: float,
) -> str:
    """Persist analysis_raw.json locally + S3 + V1 upscale_analyze checkpoint.

    Called every ``should_flush_analysis(window_count)`` and once at the
    final flush. Returns the S3 key the raw analysis was uploaded to.
    """
    raw_path = os.path.join(output_dir, "analysis_raw.json")
    with open(raw_path, "w") as f:
        json.dump(analysis_results, f, indent=2)
    raw_key = f"checkpoints/{job_id}/analysis_raw.json"
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, s3.upload_json, analysis_results, output_bucket, raw_key,
    )
    ws = _make_worker_state(
        progress_percent=progress_percent,
        current_frame=next_frame_idx,
        total_frames=total_tracking_frames,
        stage_progress_fraction=stage_progress_fraction,
    )
    await jobs_store.write_checkpoint(
        job_id, PipelineStage.UPSCALE_ANALYZE, False,
        build_upscale_window_progress(
            frame_idx=next_frame_idx,
            analysis_window_count=len(analysis_results),
            analysis_current_context=current_context,
            tracking_s3_key=tracking_s3_key,
            analysis_raw_s3_key=raw_key,
            worker_state=ws,
        ),
    )
    return raw_key


async def _update_tracking_progress_with_partial(
    job_id: str,
    clip_done: int,
    clip_total: int,
    pct: float,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
    request: TrackRequest,
    work_dir: str,
    s3: S3Client,
    *,
    resume_next_global: int,
    write_lifecycle: bool,
    upload_partial: bool,
):
    """Lifecycle progress + (every 30s) partial-tracking S3 upload + V1 checkpoint.

    The two cadences are independent: ``write_lifecycle`` is the 1-second
    Keyspaces lifecycle heartbeat; ``upload_partial`` is the 30-second
    partial-tracking durable checkpoint.

    ``clip_done`` / ``clip_total`` are 1-based position within the requested clip
    (same semantics as ``job_lifecycle.current_frame`` / ``total_frames``).
    ``resume_next_global`` is the absolute video frame index to resume SAM2 at
    (written into checkpoint artifacts).
    """
    if write_lifecycle:
        await _update_tracking_progress(
            job_id, clip_done, clip_total, pct, job_store, jobs_store, write_ks=True,
        )
    if not upload_partial:
        return
    tracking_json_path = os.path.join(work_dir, "tracking", "tracking.json")
    if not os.path.isfile(tracking_json_path):
        return
    partial_key = f"checkpoints/{job_id}/partial_tracking.json"
    upload_bucket = request.output_bucket or request.bucket
    try:
        partial_data = _load_partial_tracking_dict(tracking_json_path)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, s3.upload_json, partial_data, upload_bucket, partial_key,
        )
    except Exception as e:
        logger.warning("Periodic partial-tracking upload failed: %s", e)
        return
    ws = _make_worker_state(
        progress_percent=pct,
        current_frame=clip_done,
        total_frames=clip_total,
        stage_progress_fraction=(clip_done / max(clip_total, 1)),
    )
    await jobs_store.write_checkpoint(
        job_id, PipelineStage.TRACK, False,
        build_track_progress(
            partial_tracking_s3_key=partial_key,
            resume_from_frame=resume_next_global,
            worker_state=ws,
        ),
    )


async def _update_tracking_progress(
    job_id: str,
    clip_done: int,
    clip_total: int,
    pct: float,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
    write_ks: bool = True,
):
    await job_store.update_job(
        job_id,
        progress_percent=pct,
        current_frame=clip_done,
        total_frames=clip_total,
    )
    if write_ks:
        await jobs_store.update_progress(
            job_id, PipelineStage.TRACK, pct,
            current_frame=clip_done, total_frames=clip_total,
        )


def _make_detection_cb(
    job_id: str,
    loop: asyncio.AbstractEventLoop,
    jobs_store: JobsStore,
    s3: S3Client,
    config: ServiceConfig,
    request: TrackRequest,
    work_dir: str,
    *,
    clip_start_frame: int,
    clip_total_frames: int,
    progress_floor: float,
):
    """
    Create a sync detection_callback for run_tracking's detection_callback parameter.

    Instead of blocking on a WebSocket response, this writes a checkpoint to
    Keyspaces and returns None, which signals the tracking loop to stop.
    The job will be resumed later via /resume with corrected bounding boxes.
    """

    def detection_cb(
        reason: str, frame_jpeg: bytes, **kwargs
    ) -> tuple | None:
        yolo_detections = kwargs.get("yolo_detections", [])
        global_frame_idx = kwargs.get("frame_idx", 0)

        async def _async_suspend():
            # Convert YOLO detections to candidate format
            candidates = [
                {
                    "candidate_id": i,
                    "box": d["box"],
                    "confidence": d["confidence"],
                }
                for i, d in enumerate(yolo_detections)
            ]

            # Optional Gemini hint if >2 candidates
            suggested_boxes = None
            if len(candidates) > 2:
                try:
                    from service.vllm_selector import suggest_athletes
                    suggested_boxes = await suggest_athletes(
                        frame_jpeg, candidates, config,
                    )
                except Exception as e:
                    logger.warning("Mid-track athlete suggestion (Gemini) failed: %s", e)

            # Upload detection frame to S3 — put_object takes raw bytes;
            # upload_file would treat the JPEG bytes as a local path.
            frame_s3_key = f"checkpoints/{job_id}/frame_{global_frame_idx}.jpg"
            await loop.run_in_executor(
                None,
                lambda: s3.put_object(
                    request.bucket,
                    frame_s3_key,
                    frame_jpeg,
                    "image/jpeg",
                ),
            )

            # Save partial tracking to S3 if available. Use the same bucket
            # the resume route reads from (output_bucket when set).
            tracking_json_path = os.path.join(work_dir, "tracking", "tracking.json")
            partial_key: str | None = None
            if os.path.isfile(tracking_json_path):
                try:
                    partial_key = f"checkpoints/{job_id}/partial_tracking.json"
                    partial_data = _load_partial_tracking_dict(tracking_json_path)
                    upload_bucket = request.output_bucket or request.bucket
                    await loop.run_in_executor(
                        None, s3.upload_json, partial_data, upload_bucket, partial_key,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to save partial tracking during mid-track suspend: %s", e,
                    )
                    partial_key = None

            mid_clip_done, mid_pct = _tracking_progress_pct_clip(
                global_frame_idx, clip_start_frame, clip_total_frames, progress_floor,
            )
            mid_loss_ws = _make_worker_state(
                progress_percent=mid_pct,
                current_frame=mid_clip_done,
                total_frames=clip_total_frames,
                stage_progress_fraction=(
                    mid_clip_done / max(clip_total_frames, 1)
                ),
            )
            await jobs_store.write_checkpoint(
                job_id, PipelineStage.TRACK, False,
                build_track_mid_loss(
                    frame_idx=global_frame_idx,
                    frame_s3_key=frame_s3_key,
                    frame_bucket=request.bucket,
                    candidates=candidates,
                    suggested_boxes=suggested_boxes,
                    partial_tracking_s3_key=partial_key,
                    resume_from_frame=global_frame_idx,
                    worker_state=mid_loss_ws,
                ),
            )
            await jobs_store.set_state(job_id, JobState.AWAITING_CORRECTION)

        future = asyncio.run_coroutine_threadsafe(_async_suspend(), loop)
        try:
            future.result(timeout=60)
        except Exception as e:
            logger.warning("Mid-tracking suspend failed: %s", e)
            return None

        raise HumanVerificationSuspend()

    return detection_cb


def _run_upscale_analysis(
    video_path: str,
    tracking_json_path: str,
    config: ServiceConfig,
    request: TrackRequest,
    work_dir: str,
    *,
    job_id: str,
    jobs_store: JobsStore,
    loop: asyncio.AbstractEventLoop,
    tracking_s3_key: str,
    progress_cb=None,
):
    """
    Second-pass: read tracking.json, re-open video, crop + upscale + analyse.

    Runs AFTER tracking is complete so SAM2/RTMPose are fully unloaded and
    the upscaler can use all available GPU memory.

    Writes V1 ``upscale_analyze`` checkpoints:
      - one ``analysis_started`` row at the top of the stage
      - one ``analysis_window_completed`` row every 5 windows
      - one ``analysis_window_completed`` row at the final flush

    Returns (analysis_result, fps):
      - analysis_result: dict with clips or None if no Gemini key
      - fps: video frame rate
    """
    import cv2
    from PIL import Image
    from utils import get_union_box, get_padded_square_box

    # -- Load tracking data --------------------------------------------------
    with open(tracking_json_path) as f:
        tracking_data = json.load(f)

    output_bucket = request.output_bucket or request.bucket
    s3_for_writes = _make_s3(config)
    total_tracking_frames = len(tracking_data.get("frames", []))

    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # -- Initialize restorer -------------------------------------------------
    if request.method == "diffusion":
        from diffusion_restorer import DiffusionRestorer
        restorer = DiffusionRestorer()
    else:
        from restorer import RealESRGANRestorer
        restorer = RealESRGANRestorer(config.model_path)

    # -- Download player reference images (optional) -------------------------
    player_ref_images = None
    if request.player_references:
        from PIL import Image as _PILImage
        import io

        player_ref_images = []
        s3_for_refs = _make_s3(config)
        ref_bucket = request.bucket  # references stored in same bucket
        for ref in request.player_references:
            try:
                ref_resp = s3_for_refs.get_object(ref_bucket, ref["s3_key"])
                ref_data = ref_resp["Body"].read()
                img = _PILImage.open(io.BytesIO(ref_data))
                player_ref_images.append({
                    "player_name": ref.get("player_name", "Unknown"),
                    "image": img,
                })
                logger.info(
                    "Downloaded player reference for %s: %s",
                    ref.get("player_name"), ref["s3_key"],
                )
            except Exception as e:
                logger.warning(
                    "Failed to download player reference %s: %s",
                    ref.get("s3_key"), e,
                )
        if not player_ref_images:
            player_ref_images = None

    # -- Initialize analyser (optional) --------------------------------------
    analyze_fn = None
    if config.gemini_api_key:
        taxonomy_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "bjj_analysis_taxonomy.md",
        )
        if request.analyzer_mode == "multi":
            from analyzer import BJJMultiAgentAnalyzer, analyze_sequence_sync
            _analyzer = BJJMultiAgentAnalyzer(
                config.gemini_api_key,
                taxonomy_path=taxonomy_path,
                request_timeout_ms=config.gemini_request_timeout_ms,
            )
            analyze_fn = lambda frames, indices, ctx: analyze_sequence_sync(
                _analyzer, frames, indices, ctx,
                player_references=player_ref_images,
            )
        else:
            from analyzer import BJJTechniqueAnalyzer
            _analyzer = BJJTechniqueAnalyzer(
                config.gemini_api_key,
                taxonomy_path=taxonomy_path,
                request_timeout_ms=config.gemini_request_timeout_ms,
            )
            analyze_fn = lambda frames, indices, ctx: _analyzer.analyze_sequence(
                frames, indices, ctx,
                player_references=player_ref_images,
            )

    # -- Sliding-window analysis state ---------------------------------------
    sliding_buffer = []
    analysis_results = []
    current_context = "Start of match."
    WINDOW_SIZE = 30
    STRIDE = 15

    # -- Resume from analysis checkpoint if available ------------------------
    resume_start_frame = 0
    if request.analysis_raw_s3_key:
        try:
            s3_for_resume = _make_s3(config)
            raw_bucket = request.output_bucket or request.bucket
            prev_raw = s3_for_resume.download_json(raw_bucket, request.analysis_raw_s3_key)
            if prev_raw and isinstance(prev_raw, list):
                analysis_results = prev_raw
                # Find last window's max frame to resume from
                last_window = prev_raw[-1] if prev_raw else None
                if last_window and "frames" in last_window:
                    resume_start_frame = max(last_window["frames"]) + 1
                # Restore context
                if last_window and "analysis" in last_window:
                    ctx = last_window["analysis"].get("current_context_summary", "")
                    if ctx:
                        current_context = ctx
                logger.info(
                    "Resumed analysis from checkpoint: %d windows, resuming from frame %d",
                    len(analysis_results), resume_start_frame,
                )
        except Exception as e:
            logger.warning("Failed to load analysis checkpoint: %s", e)

    # -- Write analysis_started checkpoint -----------------------------------
    started_ws = _make_worker_state(
        progress_percent=55.0,
        current_frame=resume_start_frame,
        total_frames=total_tracking_frames,
        stage_progress_fraction=(
            resume_start_frame / max(total_tracking_frames, 1)
        ),
    )
    try:
        asyncio.run_coroutine_threadsafe(
            jobs_store.write_checkpoint(
                job_id, PipelineStage.UPSCALE_ANALYZE, False,
                build_upscale_started(
                    tracking_s3_key=tracking_s3_key,
                    analysis_raw_s3_key=request.analysis_raw_s3_key or None,
                    resume_from_frame=resume_start_frame,
                    analysis_window_count=len(analysis_results),
                    analysis_current_context=current_context,
                    worker_state=started_ws,
                ),
            ),
            loop,
        ).result(timeout=10)
    except Exception as e:
        logger.warning("analysis_started checkpoint write failed: %s", e)

    method_prefix = {
        "diffusion": "diff_", "swinir": "swinir_", "hat": "hat_",
    }.get(request.method, "esrgan_")

    def _analyze_window(window):
        nonlocal current_context
        batch_frames = [x[1] for x in window]
        batch_indices = [x[0] for x in window]
        chunk_idx = len(analysis_results) + 1
        logger.info(
            "Analyzing window %d (frames %d-%d, %d images)",
            chunk_idx, batch_indices[0], batch_indices[-1], len(batch_frames),
        )
        try:
            result_str = analyze_fn(
                batch_frames, batch_indices, current_context,
            )
            logger.info(
                "Window %d raw response length: %d chars",
                chunk_idx, len(result_str) if result_str else 0,
            )
            logger.debug("Window %d raw response: %.500s", chunk_idx, result_str)
            result_data = json.loads(result_str)

            if "error" in result_data:
                logger.error(
                    "Window %d: Gemini returned error: %s",
                    chunk_idx, result_data["error"],
                )
            elif "clips" not in result_data:
                logger.warning(
                    "Window %d: Gemini response missing 'clips' key. Keys: %s",
                    chunk_idx, list(result_data.keys()),
                )
            else:
                logger.info(
                    "Window %d: Gemini returned %d clips",
                    chunk_idx, len(result_data["clips"]),
                )

            if "current_context_summary" in result_data:
                current_context = result_data["current_context_summary"]
            analysis_results.append({
                "window": chunk_idx,
                "frames": batch_indices,
                "analysis": result_data,
            })
        except Exception as e:
            logger.error(
                "Analysis window %d failed: %s", chunk_idx, e, exc_info=True,
            )
            analysis_results.append({"window": chunk_idx, "raw_error": str(e)})

    # -- Main upscale loop ---------------------------------------------------
    frames = tracking_data.get("frames", [])
    total = len(frames)
    processed = 0
    _last_upscale_hb = time.monotonic()
    _hb_interval = float(config.upscale_heartbeat_interval_sec)

    try:
        for entry in frames:
            frame_idx = entry["frame_idx"]
            _now = time.monotonic()
            if _now - _last_upscale_hb >= _hb_interval:
                logger.info(
                    "Job %s upscale heartbeat: frame_idx=%s processed=%d/%d buffer=%d",
                    job_id, frame_idx, processed, total, len(sliding_buffer),
                )
                _last_upscale_hb = _now

            athletes = entry.get("athletes", [])

            # Skip frames before resume point
            if frame_idx < resume_start_frame:
                processed += 1
                continue

            if frame_idx % request.sampling_rate != 0:
                processed += 1
                continue
            if not athletes:
                processed += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                logger.warning("Could not read frame %d", frame_idx)
                processed += 1
                continue

            h, w = frame_bgr.shape[:2]
            boxes = [a["box"] for a in athletes]
            union_box = get_union_box(boxes)
            square_box = get_padded_square_box(
                union_box, padding=0.2, img_shape=(h, w),
            )
            x1, y1, x2, y2 = square_box
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                processed += 1
                continue

            try:
                _t_enh = time.perf_counter()
                logger.debug(
                    "Job %s upscale: enhance start frame_idx=%d crop_hw=%dx%d",
                    job_id, frame_idx, crop.shape[0], crop.shape[1],
                )
                if request.method == "diffusion":
                    h_crop, w_crop = crop.shape[:2]
                    if max(h_crop, w_crop) > 768:
                        scale = 768 / max(h_crop, w_crop)
                        crop = cv2.resize(
                            crop,
                            (int(w_crop * scale), int(h_crop * scale)),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                    enhanced = restorer.enhance(crop, strength=0.5)
                else:
                    enhanced = restorer.enhance(crop, target_size=1024)
                logger.debug(
                    "Job %s upscale: enhance done frame_idx=%d in %.2fs",
                    job_id, frame_idx, time.perf_counter() - _t_enh,
                )
            except Exception as e:
                logger.warning("Upscale failed at frame %d: %s", frame_idx, e)
                processed += 1
                continue

            out_path = os.path.join(
                output_dir, f"{method_prefix}frame_{frame_idx:06d}.jpg",
            )
            cv2.imwrite(out_path, enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # Analysis buffer
            if analyze_fn is not None:
                img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                sliding_buffer.append((frame_idx, pil_img))

                if len(sliding_buffer) >= WINDOW_SIZE:
                    _analyze_window(sliding_buffer[:WINDOW_SIZE])
                    del sliding_buffer[:STRIDE]

                    with open(
                        os.path.join(output_dir, "analysis_raw.json"), "w",
                    ) as f:
                        json.dump(analysis_results, f, indent=2)

                    # Periodic durable flush every 5 windows.
                    if should_flush_analysis(len(analysis_results)):
                        last_frame = (
                            (analysis_results[-1].get("frames") or [resume_start_frame])[-1]
                        )
                        progress_pct = 55.0 + (processed / max(total, 1)) * 25.0
                        try:
                            asyncio.run_coroutine_threadsafe(
                                _flush_analysis_checkpoint(
                                    job_id=job_id,
                                    jobs_store=jobs_store,
                                    s3=s3_for_writes,
                                    output_bucket=output_bucket,
                                    output_dir=output_dir,
                                    tracking_s3_key=tracking_s3_key,
                                    analysis_results=analysis_results,
                                    current_context=current_context,
                                    next_frame_idx=last_frame + 1,
                                    progress_percent=progress_pct,
                                    total_tracking_frames=total_tracking_frames,
                                    stage_progress_fraction=(
                                        processed / max(total, 1)
                                    ),
                                ),
                                loop,
                            ).result(timeout=30)
                        except Exception as e:
                            logger.warning(
                                "Periodic analysis flush failed: %s", e,
                            )

            processed += 1
            if progress_cb and total > 0:
                progress_cb(processed / total)
    finally:
        cap.release()

    # -- Finalize analysis ---------------------------------------------------
    if analyze_fn is None:
        logger.info("Upscale pass complete (no Gemini key -- skipping analysis)")
        return None, fps

    logger.info(
        "Finalize: %d windows analysed, %d frames in remaining buffer",
        len(analysis_results), len(sliding_buffer),
    )
    if sliding_buffer:
        _analyze_window(sliding_buffer)

    # Final durable flush — always write a checkpoint at the end of the stage,
    # whether or not we hit the every-5 boundary on the last window.
    if analysis_results:
        last_frame_final = (
            (analysis_results[-1].get("frames") or [resume_start_frame])[-1]
        )
        try:
            asyncio.run_coroutine_threadsafe(
                _flush_analysis_checkpoint(
                    job_id=job_id,
                    jobs_store=jobs_store,
                    s3=s3_for_writes,
                    output_bucket=output_bucket,
                    output_dir=output_dir,
                    tracking_s3_key=tracking_s3_key,
                    analysis_results=analysis_results,
                    current_context=current_context,
                    next_frame_idx=last_frame_final + 1,
                    progress_percent=80.0,
                    total_tracking_frames=total_tracking_frames,
                    stage_progress_fraction=1.0,
                ),
                loop,
            ).result(timeout=30)
        except Exception as e:
            logger.warning("Final analysis flush failed: %s", e)

    ok_windows = sum(
        1 for r in analysis_results
        if "analysis" in r and "clips" in r["analysis"]
    )
    err_windows = sum(1 for r in analysis_results if "raw_error" in r)
    api_err_windows = sum(
        1 for r in analysis_results
        if "analysis" in r and "error" in r.get("analysis", {})
    )
    logger.info(
        "Finalize summary: %d total windows, %d with clips, "
        "%d with raw_error, %d with Gemini API error",
        len(analysis_results), ok_windows, err_windows, api_err_windows,
    )

    final_clips = deduplicate_clips(analysis_results)
    mode = "Multi-Agent" if request.analyzer_mode == "multi" else "Single-Agent"
    result = {
        "match_summary": f"Analysis generated via {mode}",
        "clips": final_clips,
        "fps": fps,
    }
    with open(os.path.join(output_dir, "analysis_final.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result, fps


def _parse_time_range(
    video_path: str,
    start_time: str | None,
    end_time: str | None,
) -> tuple[int, int | None]:
    """Convert MM:SS or HH:MM:SS time strings to frame indices."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    def _to_seconds(ts: str) -> float:
        parts = ts.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(ts)

    start_frame = 0
    if start_time:
        start_frame = int(_to_seconds(start_time) * fps)

    end_frame = None
    if end_time:
        end_frame = int(_to_seconds(end_time) * fps)

    return start_frame, end_frame
