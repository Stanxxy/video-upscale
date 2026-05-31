"""STAGE 4: UPSCALE + ANALYSIS orchestration (55-80%)."""
from __future__ import annotations

import logging
import time

from service.analysis_keyspaces_enums import PipelineStage
from service.models import JobStatus

from service.worker.context import WorkerRunContext
from service.worker.progress import _schedule_background_coro
from service.worker.stages.parallel_upscale import _run_parallel_upscale
from service.worker.stages.upscale_analysis import _run_upscale_analysis

logger = logging.getLogger("service.worker")


async def run_upscale_stage(ctx: WorkerRunContext) -> None:
    job_id = ctx.job_id
    request = ctx.request
    config = ctx.config
    job_store = ctx.job_store
    jobs_store = ctx.jobs_store
    work_dir = ctx.work_dir
    loop = ctx.loop
    progress_floor = ctx.progress_floor
    video_path = ctx.video_path
    tracking_json_path = ctx.tracking_json_path
    clip_start_frame = ctx.clip_start_frame
    clip_end_resolved = ctx.clip_end_resolved
    clip_total_frames = ctx.clip_total_frames
    tracking_result_key = ctx.tracking_result_key
    use_parallel_upscale = ctx.use_parallel_upscale
    k_needed = ctx.k_needed

    logger.info("Job %s: stage upscale+analysis (55-80%%)", job_id)
    await job_store.update_job(
        job_id, status=JobStatus.UPSCALING, progress_percent=55.0,
    )
    await jobs_store.update_progress(job_id, PipelineStage.UPSCALE_ANALYZE, 55.0)

    if use_parallel_upscale:
        logger.info(
            "Job %s: parallel-upscale mode k_needed=%d standard_segments=%d",
            job_id, k_needed, config.standard_segments,
        )
        analysis_result, fps = await _run_parallel_upscale(
            job_id=job_id,
            video_path=video_path,
            merged_tracking_json_path=tracking_json_path,
            clip_start_frame=clip_start_frame,
            clip_end_resolved=clip_end_resolved,
            clip_total_frames=clip_total_frames,
            config=config,
            request=request,
            work_dir=work_dir,
            job_store=job_store,
            jobs_store=jobs_store,
            loop=loop,
            tracking_s3_key=tracking_result_key,
            progress_floor=progress_floor,
        )
    else:
        last_upscale_ks_write = 0.0

        async def update_upscale_progress(pct: float):
            nonlocal last_upscale_ks_write
            await job_store.update_job(job_id, progress_percent=pct)
            now = time.monotonic()
            if (now - last_upscale_ks_write) >= 1.0:
                last_upscale_ks_write = now
                await jobs_store.update_progress(
                    job_id, PipelineStage.UPSCALE_ANALYZE, pct,
                )

        def upscale_progress(pct_within_stage: float):
            overall = 55.0 + pct_within_stage * 25.0
            _schedule_background_coro(
                update_upscale_progress(overall),
                loop,
                context="upscale progress write",
            )

        analysis_result, fps = await loop.run_in_executor(
            None,
            lambda: _run_upscale_analysis(
                video_path, tracking_json_path, config, request, work_dir,
                job_id=job_id,
                jobs_store=jobs_store,
                loop=loop,
                tracking_s3_key=tracking_result_key,
                progress_cb=upscale_progress,
            ),
        )
    logger.info(
        "Job %s: upscale+analysis finished fps=%s has_analysis=%s",
        job_id,
        fps,
        analysis_result is not None,
    )

    ctx.analysis_result = analysis_result
    ctx.fps = fps
