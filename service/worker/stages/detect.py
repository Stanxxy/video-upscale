"""STAGE 2: DETECT + HUMAN VERIFY (10-15%)."""
from __future__ import annotations

import logging

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import (
    build_detect_initial_pending,
    build_verified_boxes_checkpoint,
)
from service.models import JobStatus, JobSuspendedError

from service.worker.context import WorkerRunContext
from service.worker.helpers import _is_cancelled
from service.worker.progress import _clip_done_inclusive_through_global, _make_worker_state, _pct_at_least

logger = logging.getLogger("service.worker")


async def run_detect_stage(ctx: WorkerRunContext) -> bool:
    """Returns True if pipeline should stop (cancelled). May raise JobSuspendedError."""
    job_id = ctx.job_id
    request = ctx.request
    config = ctx.config
    job_store = ctx.job_store
    jobs_store = ctx.jobs_store
    loop = ctx.loop
    progress_floor = ctx.progress_floor
    clip_start_frame = ctx.clip_start_frame
    clip_total_frames = ctx.clip_total_frames
    tracking_start_frame = ctx.tracking_start_frame
    video_path = ctx.video_path
    s3 = ctx.s3

    logger.info("Job %s: stage detect/verify (10-15%%)", job_id)
    if request.box_a is not None and request.box_b is not None:
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

        from service.tracking_runner import run_detect, capture_frame_jpeg

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

        suggested_boxes = None
        if len(candidates) > 2:
            try:
                from service.vllm_selector import suggest_athletes
                suggested_boxes = await suggest_athletes(
                    frame_jpeg, candidates, config,
                )
            except Exception as e:
                logger.warning("Athlete suggestion (Gemini) failed: %s", e)

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
        return True

    ctx.box_a = box_a
    ctx.box_b = box_b
    return False
