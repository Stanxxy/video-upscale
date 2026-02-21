import asyncio
import os
import shutil
import logging

from service.models import JobRequest, JobStatus
from service.job_store import InMemoryJobStore
from service.s3 import S3Client
from service.sns import SNSPublisher
from service.config import ServiceConfig

logger = logging.getLogger(__name__)


async def run_job(
    job_id: str,
    request: JobRequest,
    config: ServiceConfig,
    job_store: InMemoryJobStore,
):
    work_dir = os.path.join(config.temp_dir, job_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        # Stage 1: Download from S3
        await job_store.update_job(
            job_id, status=JobStatus.DOWNLOADING, progress=0.05, message="Downloading input files from S3"
        )
        s3 = S3Client(config.aws_region)
        video_path = s3.download_file(
            request.video_s3_bucket, request.video_s3_key, os.path.join(work_dir, "video.mp4")
        )
        json_path = s3.download_file(
            request.tracking_json_s3_bucket, request.tracking_json_s3_key, os.path.join(work_dir, "tracking.json")
        )

        # Stage 2: Process
        await job_store.update_job(
            job_id, status=JobStatus.PROCESSING, progress=0.1, message="Starting pipeline"
        )
        output_dir = os.path.join(work_dir, "output")
        taxonomy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bjj_analysis_taxonomy.md")

        loop = asyncio.get_event_loop()

        def _progress_callback(stage: str, pct: float):
            asyncio.run_coroutine_threadsafe(
                job_store.update_job(
                    job_id,
                    progress=0.1 + pct * 0.7,
                    message=f"{stage} ({pct:.0%})",
                ),
                loop,
            )

        # Import here to avoid loading heavy ML deps at import time
        from pipeline import process_video

        result = await loop.run_in_executor(
            None,
            lambda: process_video(
                video_path=video_path,
                json_path=json_path,
                output_dir=output_dir,
                model_path=config.model_path,
                method=request.method,
                sampling_rate=request.sampling_rate,
                analyze=True,
                api_key=config.gemini_api_key,
                multi_agent=(request.analyzer_mode == "multi"),
                taxonomy_path=taxonomy_path,
                progress_callback=_progress_callback,
            ),
        )

        if result is None:
            raise RuntimeError("Pipeline returned no analysis result")

        fps = result.get("fps", 30.0)

        # Stage 3: Upload to S3
        await job_store.update_job(
            job_id, status=JobStatus.UPLOADING, progress=0.85, message="Uploading results to S3"
        )
        result_uri = s3.upload_json(result, request.output_s3_bucket, request.output_s3_key)

        # Stage 4: Publish to SNS
        topic_arn = request.sns_topic_arn or config.sns_topic_arn
        event_count = 0
        if topic_arn:
            await job_store.update_job(
                job_id, status=JobStatus.PUBLISHING, progress=0.95, message="Publishing events to SNS"
            )
            sns = SNSPublisher(config.aws_region, topic_arn)
            event_count = sns.publish_events(
                result, request.video_id, fps,
                job_id=job_id, result_s3_uri=result_uri,
            )

        # Done
        clip_count = len(result.get("clips", []))
        await job_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            result_s3_uri=result_uri,
            message=f"Completed. {clip_count} clips detected, {event_count} events published.",
        )
        logger.info("Job %s completed: %d clips, %d events", job_id, clip_count, event_count)

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        await job_store.update_job(
            job_id, status=JobStatus.FAILED, error=str(e)
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
