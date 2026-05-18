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
from service.models import JobCancelledError, JobSuspendedError, JobStatus, ProcessingMode, TrackRequest
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
        # M3 S9: K-segment parallel path (fresh jobs, standard_segments > 1)
        # Sequential path runs when standard_segments == 1 or a resume job.
        # ============================================================
        _use_parallel_segments = (
            config.standard_segments > 1
            and not request.resume_tracking_s3_key
            and not request.skip_upscale  # skip_upscale=True is QA-only; use sequential
        )

        if _use_parallel_segments:
            # ---- Parallel path: delegate to _run_parallel_segments ----
            import torch as _torch_ps
            _is_cuda_ps = _torch_ps.cuda.is_available()
            _eff_step_ps = request.step_size or (
                120 if _is_cuda_ps else config.tracking_step_size
            )
            _eff_max_history_ps = (
                request.max_history or config.tracking_max_history
            )
            _eff_max_missing_ps = (
                request.max_missing_frames
                if request.max_missing_frames is not None
                else config.tracking_max_missing_frames
            )
            if request.frame_stride > 0:
                _eff_frame_stride_ps = request.frame_stride
            else:
                import cv2 as _cv2_ps
                _cap_fps_ps = _cv2_ps.VideoCapture(video_path)
                _src_fps_ps = _cap_fps_ps.get(_cv2_ps.CAP_PROP_FPS) or 30.0
                _cap_fps_ps.release()
                _eff_frame_stride_ps = max(1, round(_src_fps_ps / 10))
            logger.info(
                "Job %s: parallel-segment mode k=%d "
                "step_size=%s max_history=%s max_missing=%s frame_stride=%s",
                job_id, config.standard_segments,
                _eff_step_ps, _eff_max_history_ps,
                _eff_max_missing_ps, _eff_frame_stride_ps,
            )
            _par_track_pct = _pct_at_least(15.0, progress_floor)
            _par_started_cf = min(
                max(tracking_start_frame - clip_start_frame + 1, 1),
                clip_total_frames,
            )
            await job_store.update_job(
                job_id, status=JobStatus.TRACKING,
                progress_percent=_par_track_pct,
                current_frame=_par_started_cf,
                total_frames=clip_total_frames,
            )
            await jobs_store.update_progress(
                job_id, PipelineStage.TRACK, _par_track_pct,
                current_frame=_par_started_cf,
                total_frames=clip_total_frames,
            )
            tracking_json_path, analysis_result, fps = await _run_parallel_segments(
                job_id=job_id,
                video_path=video_path,
                box_a=box_a,
                box_b=box_b,
                clip_start_frame=clip_start_frame,
                clip_end_resolved=clip_end_resolved,
                clip_total_frames=clip_total_frames,
                config=config,
                request=request,
                work_dir=work_dir,
                job_store=job_store,
                jobs_store=jobs_store,
                loop=loop,
                eff_step_size=_eff_step_ps,
                eff_max_history=_eff_max_history_ps,
                eff_max_missing_frames=_eff_max_missing_ps,
                eff_frame_stride=_eff_frame_stride_ps,
                progress_floor=progress_floor,
            )
            logger.info(
                "Job %s: parallel-segment done fps=%s has_analysis=%s",
                job_id, fps, analysis_result is not None,
            )
            # Set S3-key variables used by annotate/upload/publish below.
            output_bucket = request.output_bucket or request.bucket
            base_key = os.path.splitext(request.key)[0]
            tracking_result_key = f"{base_key}_tracked.json"
            analysis_result_key = f"{base_key}_analysis.json"
            annotated_video_key = f"{base_key}_annotated.mp4"
            tracking_output_dir = os.path.join(work_dir, "tracking")
            # _run_parallel_segments already handled the skip_upscale-equivalent path;
            # skip_upscale is only relevant for the sequential path below.
        else:
            # ---- Sequential path: original stage 3 + stage 4 --------
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
                job_id, status=JobStatus.TRACKING,
                progress_percent=track_stage_pct,
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
                # S7: on CUDA (GB10) default step_size to 120 (vs 60 on CPU/MPS).
                # Larger steps reduce batch-boundary overhead; 128 GB memory is fine.
                import torch as _torch_s7
                _is_cuda_s7 = _torch_s7.cuda.is_available()
                _default_step = 120 if _is_cuda_s7 else config.tracking_step_size
                eff_step_size = request.step_size or _default_step
                eff_max_history = request.max_history or config.tracking_max_history
                eff_max_missing_frames = (
                    request.max_missing_frames
                    if request.max_missing_frames is not None
                    else config.tracking_max_missing_frames
                )
                # M4: apply fast mode presets for tracking parameters.
                _is_fast_mode = request.processing_mode == ProcessingMode.FAST
                if _is_fast_mode:
                    # Fast mode: SAM2-tiny + propagation stride 12 + no pose
                    _eff_sam2_model = "facebook/sam2.1-hiera-tiny"
                    _eff_prop_stride = 12
                    _eff_enable_pose = False
                else:
                    # Standard mode: SAM2 base-plus + no propagation stride + pose enabled
                    _eff_sam2_model = request.sam2_model
                    _eff_prop_stride = 1
                    _eff_enable_pose = True

                # S11: stride-N frame sampling. Auto-compute from source fps when 0.
                # 60fps → stride=6 → ~300 frames from 1800; 30fps → stride=3.
                # Fast mode: frame_stride = prop_stride (no benefit computing more frames than propagated)
                if _is_fast_mode:
                    eff_frame_stride = _eff_prop_stride
                elif request.frame_stride > 0:
                    eff_frame_stride = request.frame_stride
                else:
                    import cv2 as _cv2
                    _cap_fps = _cv2.VideoCapture(video_path)
                    _src_fps = _cap_fps.get(_cv2.CAP_PROP_FPS) or 30.0
                    _cap_fps.release()
                    eff_frame_stride = max(1, round(_src_fps / 10))
                logger.info(
                    "Job %s tracking config: processing_mode=%s sam2=%s "
                    "step_size=%s max_history=%s max_missing_frames=%s "
                    "frame_stride=%s prop_stride=%s enable_pose=%s",
                    job_id,
                    request.processing_mode,
                    _eff_sam2_model,
                    eff_step_size,
                    eff_max_history,
                    eff_max_missing_frames,
                    eff_frame_stride,
                    _eff_prop_stride,
                    _eff_enable_pose,
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
                        sam2_model_id=_eff_sam2_model,
                        yolo_model=request.yolo_model,
                        detection_threshold=request.detection_threshold,
                        start_frame=tracking_start_frame,
                        end_frame=end_frame,
                        step_size=eff_step_size,
                        max_history=eff_max_history,
                        max_missing_frames=eff_max_missing_frames,
                        frame_stride=eff_frame_stride,
                        prop_stride=_eff_prop_stride,
                        enable_pose=_eff_enable_pose,
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

    # -- Off-thread JPEG writer (S2) -----------------------------------------
    # The per-frame upscale write blocks the main loop on disk. Push encode +
    # write to a small thread pool. Bound the queue with a semaphore so a
    # stalled disk applies backpressure instead of growing memory unbounded.
    import concurrent.futures
    import threading
    _jpeg_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="upscale-jpeg"
    )
    _jpeg_pending: list[concurrent.futures.Future] = []
    _jpeg_pending_lock = threading.Lock()
    _jpeg_inflight_sem = threading.BoundedSemaphore(value=16)

    def _jpeg_write_task(out_path: str, img):
        try:
            ok, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            if not ok:
                logger.warning("JPEG encode failed for %s", out_path)
                return
            with open(out_path, "wb") as f:
                f.write(buf.tobytes())
        except Exception as e:
            logger.warning("JPEG write failed for %s: %s", out_path, e)
        finally:
            _jpeg_inflight_sem.release()

    def _submit_jpeg_write(out_path: str, img):
        # Block here when 16 writes are in flight -- backpressure.
        _jpeg_inflight_sem.acquire()
        fut = _jpeg_executor.submit(_jpeg_write_task, out_path, img)
        with _jpeg_pending_lock:
            # Periodic prune of completed futures so the list doesn't grow.
            if len(_jpeg_pending) > 32:
                _jpeg_pending[:] = [f for f in _jpeg_pending if not f.done()]
            _jpeg_pending.append(fut)

    def _drain_jpeg_writes(timeout: float | None = None):
        """Block until all submitted writes complete."""
        with _jpeg_pending_lock:
            pending = list(_jpeg_pending)
            _jpeg_pending.clear()
        if not pending:
            return
        logger.info("Draining %d pending JPEG writes", len(pending))
        for f in concurrent.futures.as_completed(pending, timeout=timeout):
            # Surface exceptions if any; _jpeg_write_task already logs them.
            try:
                f.result()
            except Exception:
                pass

    # -- Initialize restorer -------------------------------------------------
    # M4: fast mode uses BicubicRestorer (no neural network; negligible latency).
    if request.processing_mode == ProcessingMode.FAST:
        from restorer import BicubicRestorer
        restorer = BicubicRestorer()
        logger.info("Job %s upscale: fast mode — using BicubicRestorer (LANCZOS4)", job_id)
    elif request.method == "diffusion":
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
    # M4 F2: fast mode uses non-overlapping windows (WINDOW_SIZE=STRIDE=20)
    # to halve the window count vs standard overlapping windows (30/15).
    if request.processing_mode == ProcessingMode.FAST:
        WINDOW_SIZE = 20
        STRIDE = 20
    else:
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
        # M4 F5: fast mode drops context chain to allow full Gemini parallelism
        ctx = current_context if _use_context_chain else None
        logger.info(
            "Analyzing window %d (frames %d-%d, %d images, ctx_chain=%s)",
            chunk_idx, batch_indices[0], batch_indices[-1], len(batch_frames),
            _use_context_chain,
        )
        try:
            result_str = analyze_fn(
                batch_frames, batch_indices, ctx,
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

            if _use_context_chain and "current_context_summary" in result_data:
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

    # -- S4: async analyzer consumer thread (M2) ---------------------------------
    # Replaces M1's sync consumer thread with an async-capable consumer that owns
    # its own event loop. This allows using analyze_sequence_async (non-blocking
    # Gemini I/O) while keeping the upscale loop (producer) running concurrently
    # on a separate thread.
    #
    # gemini_max_inflight=1 (default): sequential context chain is preserved.
    # Upscale doesn't block waiting for Gemini — the producer dispatches windows
    # and continues; the async consumer processes them with the async Gemini SDK.
    # This achieves upscale/Gemini overlap even with inflight=1.
    #
    # M4 F5: fast mode raises gemini_max_inflight to 24 and drops the context chain
    # (each window gets previous_context=None). This parallelizes Gemini calls
    # dramatically at the cost of window-to-window context continuity.
    _is_fast_upscale = request.processing_mode == ProcessingMode.FAST
    _eff_gemini_max_inflight = 24 if _is_fast_upscale else config.gemini_max_inflight
    _use_context_chain = not _is_fast_upscale  # fast mode: no context chain
    if _is_fast_upscale:
        logger.info(
            "Job %s: fast mode Gemini fanout=%d (no context chain)",
            job_id, _eff_gemini_max_inflight,
        )

    import queue as _stdlib_queue
    analysis_queue: _stdlib_queue.Queue = _stdlib_queue.Queue(
        maxsize=_eff_gemini_max_inflight + 1
    )
    _analysis_consumer_stop = object()  # sentinel
    _analysis_consumer_error: list[BaseException] = []

    # Determine whether we can use async Gemini for single-agent mode.
    # BJJTechniqueAnalyzer has analyze_sequence_async; BJJMultiAgentAnalyzer
    # already has it too. We store a reference to the analyzer for async dispatch.
    # Guard: only enable async if the method is a real coroutine function (not a
    # MagicMock auto-attribute or other non-async callable).
    import inspect as _inspect
    _async_analyzer = locals().get("_analyzer") if config.gemini_api_key else None
    _async_method = getattr(_async_analyzer, "analyze_sequence_async", None) if _async_analyzer else None
    _use_async = (
        _async_analyzer is not None
        and _async_method is not None
        and _inspect.iscoroutinefunction(_async_method)
        and request.analyzer_mode != "multi"  # multi uses its own async internally
    )

    def _trigger_periodic_flush():
        if not should_flush_analysis(len(analysis_results)):
            return
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
                    stage_progress_fraction=(processed / max(total, 1)),
                ),
                loop,
            ).result(timeout=30)
        except Exception as e:
            logger.warning("Periodic analysis flush failed: %s", e)

    async def _async_analyze_window(window_payload):
        """Async version of _analyze_window using analyze_sequence_async."""
        nonlocal current_context
        batch_frames = [x[1] for x in window_payload]
        batch_indices = [x[0] for x in window_payload]
        chunk_idx = len(analysis_results) + 1
        # M4 F5: fast mode drops context chain for full Gemini parallelism
        ctx = current_context if _use_context_chain else None
        logger.info(
            "Analyzing window %d async (frames %d-%d, %d images, ctx_chain=%s)",
            chunk_idx, batch_indices[0], batch_indices[-1], len(batch_frames),
            _use_context_chain,
        )
        try:
            result_str = await _async_analyzer.analyze_sequence_async(
                batch_frames, batch_indices, ctx,
                player_references=player_ref_images,
            )
            logger.info(
                "Window %d async raw response length: %d chars",
                chunk_idx, len(result_str) if result_str else 0,
            )
            logger.debug("Window %d async raw response: %.500s", chunk_idx, result_str)
            result_data = json.loads(result_str)

            if "error" in result_data:
                logger.error("Window %d: Gemini returned error: %s", chunk_idx, result_data["error"])
            elif "clips" not in result_data:
                logger.warning(
                    "Window %d: Gemini response missing 'clips' key. Keys: %s",
                    chunk_idx, list(result_data.keys()),
                )
            else:
                logger.info("Window %d: Gemini returned %d clips", chunk_idx, len(result_data["clips"]))

            if _use_context_chain and "current_context_summary" in result_data:
                current_context = result_data["current_context_summary"]
            analysis_results.append({
                "window": chunk_idx,
                "frames": batch_indices,
                "analysis": result_data,
            })
        except Exception as e:
            logger.error("Async analysis window %d failed: %s", chunk_idx, e, exc_info=True)
            analysis_results.append({"window": chunk_idx, "raw_error": str(e)})

    def _analysis_consumer_loop():
        """Consumer thread: owns its own asyncio event loop for async Gemini calls.

        M4 F5 (fast mode): when _eff_gemini_max_inflight > 1 and _use_async is True,
        drains up to _eff_gemini_max_inflight windows from the queue and runs them
        concurrently via asyncio.gather. This multiplies Gemini throughput by fanout
        at the cost of no context chain (already disabled in fast mode).
        """
        _consumer_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_consumer_loop)
        try:
            while True:
                # Drain up to _eff_gemini_max_inflight items for batch dispatch
                items = []
                while len(items) < _eff_gemini_max_inflight:
                    if items:
                        # Non-blocking get for additional items beyond the first
                        try:
                            item = analysis_queue.get_nowait()
                        except Exception:
                            break
                    else:
                        # Blocking get for the first item
                        item = analysis_queue.get()
                    if item is _analysis_consumer_stop:
                        analysis_queue.task_done()
                        # Drain and process any already collected items before stopping
                        if items:
                            try:
                                if _use_async and len(items) > 1:
                                    _consumer_loop.run_until_complete(
                                        asyncio.gather(*[_async_analyze_window(w) for w in items])
                                    )
                                elif _use_async:
                                    _consumer_loop.run_until_complete(_async_analyze_window(items[0]))
                                else:
                                    for w in items:
                                        _analyze_window(w)
                            except Exception as e:
                                logger.error("Consumer stop-drain error: %s", e, exc_info=True)
                            finally:
                                for _ in items:
                                    analysis_queue.task_done()
                        return
                    items.append(item)

                if not items:
                    continue
                try:
                    if _use_async and len(items) > 1:
                        # M4 F5: concurrent dispatch via asyncio.gather
                        _consumer_loop.run_until_complete(
                            asyncio.gather(*[_async_analyze_window(w) for w in items])
                        )
                    elif _use_async:
                        _consumer_loop.run_until_complete(_async_analyze_window(items[0]))
                    else:
                        for w in items:
                            _analyze_window(w)
                    try:
                        with open(
                            os.path.join(output_dir, "analysis_raw.json"), "w"
                        ) as f:
                            json.dump(analysis_results, f, indent=2)
                    except Exception as e:
                        logger.warning("analysis_raw.json write failed: %s", e)
                    _trigger_periodic_flush()
                finally:
                    for _ in items:
                        analysis_queue.task_done()
        except BaseException as e:
            logger.error("Analysis consumer crashed: %s", e, exc_info=True)
            _analysis_consumer_error.append(e)
        finally:
            _consumer_loop.close()

    _analysis_consumer_thread = None
    if analyze_fn is not None:
        _analysis_consumer_thread = threading.Thread(
            target=_analysis_consumer_loop,
            name="upscale-analyzer-consumer",
            daemon=True,
        )
        _analysis_consumer_thread.start()

    def _dispatch_window(window):
        """Hand a window off to the consumer thread. Blocks if queue is full (backpressure)."""
        analysis_queue.put(window)

    # -- Main upscale loop ---------------------------------------------------
    frames = tracking_data.get("frames", [])
    total = len(frames)
    processed = 0
    _last_upscale_hb = time.monotonic()
    _hb_interval = float(config.upscale_heartbeat_interval_sec)
    # M4 F2: fast mode applies effective_sampling_rate=max(request.sampling_rate, 2)
    # so we skip every other upscale frame, halving the upscale workload.
    _eff_sampling_rate = (
        max(request.sampling_rate, 2)
        if request.processing_mode == ProcessingMode.FAST
        else request.sampling_rate
    )

    # Sequential VideoCapture cursor. tracking_data['frames'] is in ascending
    # frame_idx order (asserted below). Calling cap.set(POS_FRAMES) per frame
    # forces an O(GOP) random seek on long-GOP codecs (VP9, H.264). Instead we
    # advance cap.read() sequentially and only fall back to cap.set() when the
    # requested frame is not the next contiguous frame (e.g., sampling_rate>1
    # gaps, resume from a checkpoint past frame 0, or out-of-order entries).
    _cap_cursor = -1  # idx of last frame consumed by cap.read(); -1 means nothing read yet

    def _read_frame_at(target_idx: int):
        """Return (ret, frame_bgr) for target_idx using sequential reads when possible."""
        nonlocal _cap_cursor
        next_expected = _cap_cursor + 1
        if target_idx == next_expected:
            ret, frame_bgr = cap.read()
            if ret:
                _cap_cursor = target_idx
            return ret, frame_bgr
        if target_idx > next_expected:
            # Skip the gap with cheap sequential reads (decoding-cheaper than a
            # set() seek for small gaps; for large gaps cap.set is faster, so
            # we threshold).
            gap = target_idx - next_expected
            if gap <= 8:
                # Burn through the gap with cap.read() (no decoding savings
                # available without a seek anyway, but stays warm).
                for _ in range(gap):
                    ok, _ = cap.read()
                    if not ok:
                        _cap_cursor = target_idx  # advance cursor anyway to avoid infinite loop
                        return False, None
                    _cap_cursor += 1
                ret, frame_bgr = cap.read()
                if ret:
                    _cap_cursor = target_idx
                return ret, frame_bgr
        # Fall back to a true seek (backwards jump or large forward gap).
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame_bgr = cap.read()
        if ret:
            _cap_cursor = target_idx
        return ret, frame_bgr

    # Sanity assert: frames are ascending. Log and fall back if not, so the
    # sequential path is safe.
    _ascending = all(
        frames[i]["frame_idx"] <= frames[i + 1]["frame_idx"]
        for i in range(len(frames) - 1)
    )
    if not _ascending:
        logger.warning(
            "tracking_data frames are not in ascending frame_idx order; sequential "
            "read optimization will fall back to seeks on every out-of-order entry"
        )

    # S1: batch upscale crops. ESRGAN forward passes are GPU-bound and small;
    # batching N crops into a single forward pass amortizes Python/launch
    # overhead 8x. Diffusion path retains per-call semantics (it takes
    # 'strength', not a list).
    BATCH_SIZE = 8 if request.method != "diffusion" else 1
    pending_crops: list[tuple[int, "np.ndarray"]] = []  # (frame_idx, crop_bgr)

    def _flush_batch():
        nonlocal processed
        if not pending_crops:
            return
        try:
            _t_enh = time.perf_counter()
            if request.method == "diffusion":
                # 1-element list in diffusion path; preserve old semantics.
                fidx, crop = pending_crops[0]
                h_crop, w_crop = crop.shape[:2]
                if max(h_crop, w_crop) > 768:
                    scale = 768 / max(h_crop, w_crop)
                    crop = cv2.resize(
                        crop,
                        (int(w_crop * scale), int(h_crop * scale)),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                enhanced_list = [restorer.enhance(crop, strength=0.5)]
                batch_indices = [fidx]
            else:
                batch_indices = [fi for fi, _ in pending_crops]
                # S5: pre-scale crops to cap long edge at 288px before ESRGAN.
                # x4 produces ~1152px, then target_size caps to 768px.
                # ~3.5x fewer FLOPs vs sending full-size crops to ESRGAN.
                _S5_MAX_INPUT_EDGE = 288
                prescaled_crops = []
                for c in (c for _, c in pending_crops):
                    h_c, w_c = c.shape[:2]
                    long_edge = max(h_c, w_c)
                    if long_edge > _S5_MAX_INPUT_EDGE:
                        _scale = _S5_MAX_INPUT_EDGE / long_edge
                        c = cv2.resize(
                            c,
                            (max(1, int(w_c * _scale)), max(1, int(h_c * _scale))),
                            interpolation=cv2.INTER_AREA,
                        )
                    prescaled_crops.append(c)
                batch_crops = prescaled_crops
                _target_sz = config.upscale_target_size
                enhance_batch_fn = getattr(restorer, "enhance_batch", None)
                use_batched = (
                    callable(enhance_batch_fn)
                    # Restorer classes implement enhance_batch as a method on
                    # the class. MagicMock auto-creates attributes, so guard
                    # by checking the attribute is defined on the class (not
                    # an auto-mocked instance attribute).
                    and hasattr(type(restorer), "enhance_batch")
                )
                if use_batched:
                    enhanced_list = enhance_batch_fn(
                        batch_crops, target_size=_target_sz
                    )
                else:
                    enhanced_list = [
                        restorer.enhance(c, target_size=_target_sz)
                        for c in batch_crops
                    ]
            logger.debug(
                "Job %s upscale: batch of %d done in %.2fs (%.2f ms/frame)",
                job_id, len(pending_crops),
                time.perf_counter() - _t_enh,
                (time.perf_counter() - _t_enh) * 1000.0 / max(len(pending_crops), 1),
            )
        except Exception as e:
            logger.warning(
                "Batch upscale failed for frames %s: %s",
                [fi for fi, _ in pending_crops], e,
            )
            pending_crops.clear()
            return

        # Emit results in order.
        for fidx, enhanced in zip(batch_indices, enhanced_list):
            out_path = os.path.join(
                output_dir, f"{method_prefix}frame_{fidx:06d}.jpg",
            )
            _submit_jpeg_write(out_path, enhanced)

            if analyze_fn is not None:
                img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                sliding_buffer.append((fidx, pil_img))

                if len(sliding_buffer) >= WINDOW_SIZE:
                    # S3: dispatch to the analyzer consumer thread instead of
                    # blocking the upscale loop. Pass a copy so the producer
                    # can mutate sliding_buffer freely after handing it off.
                    _dispatch_window(list(sliding_buffer[:WINDOW_SIZE]))
                    del sliding_buffer[:STRIDE]

        # S1: amortized empty_cache. Reclaim cached blocks once per batch
        # instead of once per crop.
        if hasattr(restorer, "flush_cache"):
            restorer.flush_cache()
        pending_crops.clear()

    try:
        for entry in frames:
            frame_idx = entry["frame_idx"]
            _now = time.monotonic()
            if _now - _last_upscale_hb >= _hb_interval:
                logger.info(
                    "Job %s upscale heartbeat: frame_idx=%s processed=%d/%d buffer=%d batch=%d",
                    job_id, frame_idx, processed, total,
                    len(sliding_buffer), len(pending_crops),
                )
                _last_upscale_hb = _now

            athletes = entry.get("athletes", [])

            # Skip frames before resume point
            if frame_idx < resume_start_frame:
                processed += 1
                continue

            if frame_idx % _eff_sampling_rate != 0:
                processed += 1
                continue
            if not athletes:
                processed += 1
                continue

            ret, frame_bgr = _read_frame_at(frame_idx)
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

            pending_crops.append((frame_idx, crop))
            processed += 1

            if len(pending_crops) >= BATCH_SIZE:
                _flush_batch()

            if progress_cb and total > 0:
                progress_cb(processed / total)

        # Drain any straggler crops at end of stream.
        _flush_batch()
    finally:
        cap.release()
        # S2: drain pending JPEG writes before stage exit so resume sees a
        # consistent on-disk artifact set.
        try:
            _drain_jpeg_writes(timeout=120.0)
        except Exception as e:
            logger.warning("JPEG drain failed: %s", e)
        _jpeg_executor.shutdown(wait=True)

    # -- Finalize analysis ---------------------------------------------------
    if analyze_fn is None:
        logger.info("Upscale pass complete (no Gemini key -- skipping analysis)")
        return None, fps

    # S3: flush the remaining buffer through the consumer queue so the
    # final window is processed in-order with everything else, then
    # signal the consumer to stop and wait for it to drain.
    if sliding_buffer:
        _dispatch_window(list(sliding_buffer))
        sliding_buffer.clear()
    if _analysis_consumer_thread is not None:
        analysis_queue.put(_analysis_consumer_stop)
        _analysis_consumer_thread.join(timeout=300.0)
        if _analysis_consumer_thread.is_alive():
            logger.warning(
                "Analysis consumer thread did not exit within 300s drain"
            )
        if _analysis_consumer_error:
            logger.error(
                "Analysis consumer reported error during run: %s",
                _analysis_consumer_error[0],
            )

    logger.info(
        "Finalize: %d windows analysed (consumer drained)",
        len(analysis_results),
    )

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


# ---------------------------------------------------------------------------
# M3 S9: K-segment parallel helper
# ---------------------------------------------------------------------------

async def _run_parallel_segments(
    *,
    job_id: str,
    video_path: str,
    box_a: list,
    box_b: list,
    clip_start_frame: int,
    clip_end_resolved: int,
    clip_total_frames: int,
    config: ServiceConfig,
    request: TrackRequest,
    work_dir: str,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
    loop: asyncio.AbstractEventLoop,
    eff_step_size: int,
    eff_max_history: int,
    eff_max_missing_frames: int,
    eff_frame_stride: int,
    progress_floor: float,
) -> tuple[str, dict | None, float]:
    """
    Run K segments in parallel (tracking + upscale + analyze each segment).

    Returns (merged_tracking_json_path, merged_analysis_result, fps).

    Each segment runs in a thread executor via asyncio.gather.
    Identity stitching is applied at each boundary before frames are merged.
    The merged tracking JSON is written to ``work_dir/tracking/tracking.json``.

    Writes the TRACK completed checkpoint and uploads the merged tracking JSON
    before starting the upscale stage.
    """
    from service.segment_runner import (
        split_segments,
        stitch_segment_identity,
        merge_tracking_results,
        merge_analysis_results,
        OVERLAP_FRAMES,
    )
    from service.tracking_runner import run_tracking_job

    k = config.standard_segments
    if k <= 1:
        raise ValueError("_run_parallel_segments called with k<=1; use sequential path")

    logger.info(
        "Job %s: K-segment parallel mode k=%d clip=[%d, %d) total_frames=%d",
        job_id, k, clip_start_frame, clip_end_resolved, clip_total_frames,
    )

    segments = split_segments(clip_start_frame, clip_end_resolved, k, OVERLAP_FRAMES)

    # Per-segment output directories
    seg_output_dirs = [
        os.path.join(work_dir, f"tracking_seg{s.segment_id}") for s in segments
    ]
    for d in seg_output_dirs:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Launch K tracking tasks in parallel via thread executor
    # ------------------------------------------------------------------
    def _track_segment(seg_idx: int) -> str | None:
        """Thread: run SAM2 tracking for one segment. Returns tracking.json path or None."""
        seg = segments[seg_idx]
        out_dir = seg_output_dirs[seg_idx]
        logger.info(
            "Job %s seg%d: tracking start_frame=%d end_frame=%d length=%d",
            job_id, seg_idx, seg.start_frame, seg.end_frame, seg.length,
        )
        try:
            path = run_tracking_job(
                video_path=video_path,
                box_a=box_a,
                box_b=box_b,
                output_dir=out_dir,
                sam2_model_id=request.sam2_model,
                yolo_model=request.yolo_model,
                detection_threshold=request.detection_threshold,
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
                step_size=eff_step_size,
                max_history=eff_max_history,
                max_missing_frames=eff_max_missing_frames,
                frame_stride=eff_frame_stride,
                # Per-segment segments don't do mid-tracking suspends (no detection_cb).
                # Detection/YOLO is not re-run within a segment if tracking stays healthy.
                detection_cb=None,
                progress_cb=None,
                should_stop=lambda: job_store.is_cancelled(job_id),
            )
            logger.info("Job %s seg%d: tracking done path=%s", job_id, seg_idx, path)
            return path
        except Exception as e:
            logger.error(
                "Job %s seg%d: tracking failed: %s", job_id, seg_idx, e, exc_info=True,
            )
            raise

    # Gather K tracking tasks concurrently
    tracking_futures = [
        loop.run_in_executor(None, _track_segment, i) for i in range(k)
    ]
    tracking_paths: list[str | None] = list(await asyncio.gather(*tracking_futures))

    # Check for cancellation
    if any(p is None for p in tracking_paths):
        raise JobCancelledError(
            "A tracking segment returned None (job was cancelled or suspended)"
        )

    # ------------------------------------------------------------------
    # Load per-segment tracking JSON
    # ------------------------------------------------------------------
    seg_tracking_frames: list[list[dict]] = []
    seg_fps = 30.0
    for i, path in enumerate(tracking_paths):
        with open(path) as f:
            data = json.load(f)
        seg_fps = data.get("fps", seg_fps)
        frames = data.get("frames", [])
        seg_tracking_frames.append(frames)
        logger.info(
            "Job %s seg%d: loaded %d tracking frames", job_id, i, len(frames),
        )

    # ------------------------------------------------------------------
    # Identity stitching at each segment boundary
    # ------------------------------------------------------------------
    stitched_frames: list[list[dict]] = [seg_tracking_frames[0]]
    for i in range(1, k):
        remapped = stitch_segment_identity(
            stitched_frames[i - 1],
            seg_tracking_frames[i],
            OVERLAP_FRAMES,
        )
        stitched_frames.append(remapped)
        logger.info(
            "Job %s: stitched seg%d identity (%d frames)", job_id, i, len(remapped),
        )

    # ------------------------------------------------------------------
    # Merge tracking frames (de-overlap)
    # ------------------------------------------------------------------
    merged_tracking = merge_tracking_results(segments, stitched_frames, OVERLAP_FRAMES)
    # Augment with metadata from seg0's data
    with open(tracking_paths[0]) as f:
        first_seg_data = json.load(f)
    merged_tracking_full = {
        "video": first_seg_data.get("video", video_path),
        "fps": seg_fps,
        "start_frame": clip_start_frame,
        "end_frame": clip_end_resolved,
        "frames": merged_tracking["frames"],
    }

    merged_tracking_dir = os.path.join(work_dir, "tracking")
    os.makedirs(merged_tracking_dir, exist_ok=True)
    merged_tracking_path = os.path.join(merged_tracking_dir, "tracking.json")
    with open(merged_tracking_path, "w") as f:
        json.dump(merged_tracking_full, f)

    _frame_count = len(merged_tracking_full["frames"])
    logger.info(
        "Job %s: merged tracking: %d total frames from %d segments",
        job_id, _frame_count, k,
    )

    # ------------------------------------------------------------------
    # Write tracking-complete checkpoint and upload merged tracking JSON
    # ------------------------------------------------------------------
    await jobs_store.write_checkpoint(
        job_id, PipelineStage.TRACK, False,
        build_track_completed(
            start_frame=clip_start_frame,
            frame_count=_frame_count,
            worker_state=_make_worker_state(
                progress_percent=55.0,
                current_frame=_frame_count,
                total_frames=clip_total_frames,
                stage_progress_fraction=1.0,
            ),
        ),
    )

    output_bucket = request.output_bucket or request.bucket
    base_key = os.path.splitext(request.key)[0]
    tracking_result_key = f"{base_key}_tracked.json"

    s3 = _make_s3(config)
    s3.ensure_bucket(output_bucket)
    await loop.run_in_executor(
        None, s3.upload_json, merged_tracking_full, output_bucket, tracking_result_key,
    )
    await jobs_store.write_checkpoint(
        job_id, PipelineStage.TRACK, False,
        build_track_completed(
            start_frame=clip_start_frame,
            frame_count=_frame_count,
            tracking_s3_key=tracking_result_key,
            worker_state=_make_worker_state(
                progress_percent=55.0,
                current_frame=_frame_count,
                total_frames=clip_total_frames,
                stage_progress_fraction=1.0,
            ),
        ),
    )
    await jobs_store.write_checkpoint(
        job_id, PipelineStage.UPLOAD, False,
        build_upload_incremental(
            tracking_s3_key=tracking_result_key,
            worker_state=_make_worker_state(
                progress_percent=55.0, stage_progress_fraction=0.33,
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Upscale + analyze each segment in parallel
    # ------------------------------------------------------------------
    await job_store.update_job(job_id, status=JobStatus.UPSCALING, progress_percent=55.0)
    await jobs_store.update_progress(job_id, PipelineStage.UPSCALE_ANALYZE, 55.0)

    # Build per-segment tracking JSON (stitched frames, clipped to segment range)
    seg_upscale_work_dirs = [
        os.path.join(work_dir, f"upscale_seg{s.segment_id}") for s in segments
    ]
    seg_tracking_paths: list[str] = []
    for i, (seg, frames) in enumerate(zip(segments, stitched_frames)):
        os.makedirs(seg_upscale_work_dirs[i], exist_ok=True)
        seg_track_data = {
            **merged_tracking_full,
            "frames": frames,
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
        }
        seg_track_path = os.path.join(seg_upscale_work_dirs[i], "tracking.json")
        with open(seg_track_path, "w") as f:
            json.dump(seg_track_data, f)
        seg_tracking_paths.append(seg_track_path)

    def _upscale_segment(seg_idx: int) -> tuple[dict | None, float]:
        """Thread: run upscale+analyze for one segment."""
        try:
            result, fps_val = _run_upscale_analysis(
                video_path,
                seg_tracking_paths[seg_idx],
                config,
                request,
                seg_upscale_work_dirs[seg_idx],
                job_id=job_id,
                jobs_store=jobs_store,
                loop=loop,
                tracking_s3_key=tracking_result_key,
                progress_cb=None,
            )
            logger.info(
                "Job %s seg%d: upscale+analyze done has_analysis=%s",
                job_id, seg_idx, result is not None,
            )
            return result, fps_val
        except Exception as e:
            logger.error(
                "Job %s seg%d: upscale failed: %s",
                job_id, seg_idx, e, exc_info=True,
            )
            raise

    upscale_futures = [
        loop.run_in_executor(None, _upscale_segment, i) for i in range(k)
    ]
    upscale_results: list[tuple[dict | None, float]] = list(
        await asyncio.gather(*upscale_futures)
    )

    # ------------------------------------------------------------------
    # Merge analysis results
    # ------------------------------------------------------------------
    seg_analyses = [r[0] for r in upscale_results]
    fps_values = [r[1] for r in upscale_results if r[1] and r[1] > 0]
    final_fps = fps_values[0] if fps_values else seg_fps

    merged_analysis = merge_analysis_results(seg_analyses)
    if merged_analysis:
        merged_analysis["fps"] = final_fps

    logger.info(
        "Job %s: K-segment complete fps=%.2f has_analysis=%s merged_frames=%d",
        job_id, final_fps, merged_analysis is not None, _frame_count,
    )

    return merged_tracking_path, merged_analysis, final_fps
