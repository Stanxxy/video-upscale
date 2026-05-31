"""Unified pipeline worker orchestrator."""
import asyncio
import logging
import os
import shutil

from service.analysis_keyspaces_enums import JobState
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.models import (
    JobCancelledError,
    JobStatus,
    JobSuspendedError,
    TrackRequest,
)

from service.worker.context import WorkerRunContext
from service.worker.gpu import _ensure_models_released, _load_partial_tracking_dict
from service.worker.helpers import _is_cancelled
from service.worker.progress import _clip_done_inclusive_through_global, _make_worker_state
from service.worker.stages.annotate import run_annotate_stage
from service.worker.stages.download import run_download_stage
from service.worker.stages.detect import run_detect_stage
from service.worker.stages.publish import run_publish_and_complete_stage
from service.worker.stages.track import run_track_stage
from service.worker.stages.upload import run_pre_upscale_tracking_upload, run_upload_stage
from service.worker.stages.upscale_run import run_upscale_stage

from service.analysis_keyspaces_enums import PipelineStage
from service.checkpoints import build_track_progress

logger = logging.getLogger("service.worker")


async def run_job(
    job_id: str,
    request: TrackRequest,
    config: ServiceConfig,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
):
    """Run the full tracking + upscale pipeline for a single job."""
    work_dir = os.path.join(config.temp_dir, job_id)
    os.makedirs(work_dir, exist_ok=True)
    loop = asyncio.get_event_loop()
    logger.info(
        "Job %s: worker starting work_dir=%s input=s3://%s/%s skip_upscale=%s",
        job_id,
        work_dir,
        request.bucket,
        request.key,
        request.skip_upscale,
    )

    ctx = WorkerRunContext(
        job_id=job_id,
        request=request,
        config=config,
        job_store=job_store,
        jobs_store=jobs_store,
        work_dir=work_dir,
        loop=loop,
    )

    try:
        await jobs_store.set_state(job_id, JobState.RUNNING)

        lifecycle_row = await jobs_store.get_lifecycle(job_id)
        ctx.progress_floor = float(
            (lifecycle_row or {}).get("progress_percent") or 0.0,
        )

        if await run_download_stage(ctx):
            return
        if await run_detect_stage(ctx):
            return

        await run_track_stage(ctx)
        if ctx.pipeline_complete:
            return

        await run_pre_upscale_tracking_upload(ctx)
        await run_upscale_stage(ctx)
        await run_annotate_stage(ctx)
        if _is_cancelled(job_id, job_store):
            return
        if await run_upload_stage(ctx):
            return
        await run_publish_and_complete_stage(ctx)

    except asyncio.CancelledError:
        logger.info("Job %s cancelled (client disconnected)", job_id)
        await job_store.update_job(job_id, status=JobStatus.CANCELLED)
        await jobs_store.set_state(job_id, JobState.CANCELLED)

    except JobCancelledError:
        lifecycle = await jobs_store.get_lifecycle(job_id)
        if lifecycle and lifecycle.get("job_state") == JobState.AWAITING_CORRECTION.value:
            logger.info(
                "Job %s: JobCancelledError after mid-track suspend — keeping AWAITING_CORRECTION",
                job_id,
            )
        else:
            logger.info("Job %s cancelled (DELETE requested)", job_id)
            await job_store.update_job(job_id, status=JobStatus.CANCELLED)
            await jobs_store.set_state(job_id, JobState.CANCELLED)

    except JobSuspendedError as e:
        logger.info("Job %s suspended: %s", job_id, e)

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        await job_store.update_job(
            job_id, status=JobStatus.FAILED, error_message=str(e),
        )
        await jobs_store.set_state(job_id, JobState.FAILED, error_message=str(e))

    finally:
        _ensure_models_released()

        try:
            ks_lc = await jobs_store.get_lifecycle(job_id)
            ks_state = ks_lc.get("job_state") if ks_lc else None
            job_status = await job_store.get_job(job_id)
            partial_tracking = os.path.join(work_dir, "tracking", "tracking.json")
            s3 = ctx.s3
            if (
                s3 is not None
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
                    None, s3.upload_json, partial_data, bucket, partial_key,
                )
                logger.info(
                    "Job %s: saved partial tracking to s3://%s/%s",
                    job_id, bucket, partial_key,
                )
                frames = partial_data.get("frames") or []
                if frames:
                    last_g = int(frames[-1].get("frame_idx", len(frames) - 1))
                else:
                    last_g = ctx.clip_start_frame - 1
                resume_frm = last_g + 1
                clip_done_fail = _clip_done_inclusive_through_global(
                    last_g, ctx.clip_start_frame, ctx.clip_total_frames,
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
                            total_frames=ctx.clip_total_frames,
                            stage_progress_fraction=(
                                clip_done_fail / max(ctx.clip_total_frames, 1)
                            ),
                        ),
                    ),
                )
        except Exception as e:
            logger.warning("Job %s: failed to save partial tracking: %s", job_id, e)

        shutil.rmtree(work_dir, ignore_errors=True)
