"""STAGE 6: PUBLISH SNS (90-95%) and job completion."""
from __future__ import annotations

import logging
from uuid import uuid4

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import build_publish_completed
from service.models import JobStatus
from service.sns import SNSPublisher

from service.worker.context import WorkerRunContext
from service.worker.progress import _make_worker_state

logger = logging.getLogger("service.worker")


async def run_publish_and_complete_stage(ctx: WorkerRunContext) -> None:
    job_id = ctx.job_id
    request = ctx.request
    config = ctx.config
    job_store = ctx.job_store
    jobs_store = ctx.jobs_store
    output_bucket = ctx.output_bucket
    tracking_result_key = ctx.tracking_result_key
    analysis_result_key = ctx.analysis_result_key
    analysis_result = ctx.analysis_result
    fps = ctx.fps

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
            # Reuse the SAME output_bucket and tracking_result_key the upload
            # stage persisted on the context (upload.py set ctx.output_bucket /
            # ctx.tracking_result_key from one base_key). No recomputation here,
            # so the published URI is byte-identical to the uploaded artifact.
            tracking_uri = f"s3://{output_bucket}/{tracking_result_key}"
            event_count = sns.publish_events(
                analysis_result, video_id, fps,
                job_id=job_id, result_s3_uri=result_uri,
                tracking_s3_uri=tracking_uri,
                athlete_bindings=request.athlete_bindings,
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

    ctx.event_count = event_count
