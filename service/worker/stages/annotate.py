"""STAGE 4.5: ANNOTATE VIDEO (80-85%)."""
from __future__ import annotations

import logging
import os

from service.analysis_keyspaces_enums import PipelineStage

from service.worker.context import WorkerRunContext

logger = logging.getLogger("service.worker")


async def run_annotate_stage(ctx: WorkerRunContext) -> None:
    job_id = ctx.job_id
    job_store = ctx.job_store
    jobs_store = ctx.jobs_store
    work_dir = ctx.work_dir
    loop = ctx.loop
    clip_start_frame = ctx.clip_start_frame
    tracking_output_dir = ctx.tracking_output_dir
    analysis_result = ctx.analysis_result
    fps = ctx.fps

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
        annotated_video_path = tracked_video_path

    await job_store.update_job(job_id, progress_percent=85.0)
    await jobs_store.update_progress(job_id, PipelineStage.ANNOTATE, 85.0)

    ctx.annotated_video_path = annotated_video_path
