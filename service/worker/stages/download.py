"""STAGE 1: DOWNLOAD (0-10%)."""
from __future__ import annotations

import logging
import os

from service.analysis_keyspaces_enums import PipelineStage
from service.checkpoints import END_OF_TRACKING_SENTINEL, build_download_completed

from service.worker.context import WorkerRunContext
from service.worker.helpers import _is_cancelled, _make_s3
from service.worker.progress import (
    _make_worker_state,
    _pct_at_least,
    _resolved_clip_end_and_total,
    _video_frame_cap,
)
from service.worker.stages.upscale_parse import _parse_time_range

logger = logging.getLogger("service.worker")


async def run_download_stage(ctx: WorkerRunContext) -> bool:
    """Download source video and resolve clip bounds. Returns True if pipeline should stop."""
    job_id = ctx.job_id
    request = ctx.request
    config = ctx.config
    job_store = ctx.job_store
    jobs_store = ctx.jobs_store
    work_dir = ctx.work_dir
    loop = ctx.loop
    progress_floor = ctx.progress_floor

    from service.models import JobStatus

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

    s3 = _make_s3(config)
    s3.ensure_bucket(request.bucket)

    video_path = await loop.run_in_executor(
        None,
        s3.download_file,
        request.bucket,
        request.key,
        os.path.join(work_dir, "video.mp4"),
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
        return True

    clip_start_frame, end_frame = _parse_time_range(
        video_path, request.start_time, request.end_time,
    )
    logger.info(
        "Job %s: time range clip_start_frame=%d end_frame=%s",
        job_id,
        clip_start_frame,
        end_frame if end_frame is not None else "full_video",
    )

    vid_frame_cap = await loop.run_in_executor(None, _video_frame_cap, video_path)
    clip_end_resolved, clip_total_frames = _resolved_clip_end_and_total(
        clip_start_frame, end_frame, vid_frame_cap,
    )
    logger.info(
        "Job %s: clip_end_resolved=%d clip_total_frames=%d (video_cap=%d)",
        job_id,
        clip_end_resolved,
        clip_total_frames,
        vid_frame_cap,
    )

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

    ctx.video_path = video_path
    ctx.s3 = s3
    ctx.clip_start_frame = clip_start_frame
    ctx.end_frame = end_frame
    ctx.clip_end_resolved = clip_end_resolved
    ctx.clip_total_frames = clip_total_frames
    ctx.tracking_start_frame = tracking_start_frame
    return False
