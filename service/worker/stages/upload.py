"""Pre-upscale tracking upload and STAGE 5: UPLOAD (85-90%)."""
from __future__ import annotations

import json
import logging
import os

from service.analysis_keyspaces_enums import PipelineStage
from service.checkpoints import build_track_completed, build_upload_incremental
from service.models import JobStatus

from service.worker.context import WorkerRunContext
from service.worker.helpers import _is_cancelled
from service.worker.progress import _make_worker_state, _track_completed_clip_worker_state

logger = logging.getLogger("service.worker")


async def run_pre_upscale_tracking_upload(ctx: WorkerRunContext) -> None:
    """Upload tracking JSON before upscale so checkpoints can record durable keys."""
    job_id = ctx.job_id
    request = ctx.request
    jobs_store = ctx.jobs_store
    loop = ctx.loop
    clip_start_frame = ctx.clip_start_frame
    clip_total_frames = ctx.clip_total_frames
    tracking_json_path = ctx.tracking_json_path
    s3 = ctx.s3
    frame_count = ctx.frame_count

    output_bucket = request.output_bucket or request.bucket
    base_key = os.path.splitext(request.key)[0]
    tracking_result_key = f"{base_key}_tracked.json"
    analysis_result_key = f"{base_key}_analysis.json"
    annotated_video_key = f"{base_key}_annotated.mp4"

    s3.ensure_bucket(output_bucket)
    with open(tracking_json_path) as f:
        full_tracking = json.load(f)
    if request.resume_existing_upload_tracking_key == tracking_result_key:
        logger.info(
            "Job %s: skipping tracking JSON re-upload (key already durable)",
            job_id,
        )
    else:
        await loop.run_in_executor(
            None, s3.upload_json, full_tracking,
            output_bucket, tracking_result_key,
        )

    await jobs_store.write_checkpoint(
        job_id, PipelineStage.TRACK, False,
        build_track_completed(
            start_frame=clip_start_frame,
            frame_count=frame_count,
            tracking_s3_key=tracking_result_key,
            worker_state=_track_completed_clip_worker_state(
                full_tracking, clip_start_frame, clip_total_frames,
                progress_percent=55.0,
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

    ctx.output_bucket = output_bucket
    ctx.base_key = base_key
    ctx.tracking_result_key = tracking_result_key
    ctx.analysis_result_key = analysis_result_key
    ctx.annotated_video_key = annotated_video_key


async def run_upload_stage(ctx: WorkerRunContext) -> bool:
    """Upload analysis and annotated artifacts. Returns True if cancelled."""
    job_id = ctx.job_id
    request = ctx.request
    job_store = ctx.job_store
    jobs_store = ctx.jobs_store
    loop = ctx.loop
    output_bucket = ctx.output_bucket
    tracking_result_key = ctx.tracking_result_key
    analysis_result_key = ctx.analysis_result_key
    annotated_video_key = ctx.annotated_video_key
    analysis_result = ctx.analysis_result
    annotated_video_path = ctx.annotated_video_path
    s3 = ctx.s3

    from service.checkpoints import build_annotate_completed

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
        return True
    return False
