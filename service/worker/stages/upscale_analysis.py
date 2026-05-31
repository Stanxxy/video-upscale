"""Sequential upscale + analysis stage."""
import asyncio

from service.config import ServiceConfig
from service.jobs_store import JobsStore
from service.models import TrackRequest

from service.worker.stages.upscale_setup import _upscale_analysis_setup
from service.worker.stages.upscale_loop import _upscale_analysis_loop_finalize


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
    """
    g = _upscale_analysis_setup(
        video_path,
        tracking_json_path,
        config,
        request,
        work_dir,
        job_id=job_id,
        jobs_store=jobs_store,
        loop=loop,
        tracking_s3_key=tracking_s3_key,
        progress_cb=progress_cb,
    )
    return _upscale_analysis_loop_finalize(g)


__all__ = ["_run_upscale_analysis"]
