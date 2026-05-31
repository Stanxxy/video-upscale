"""Analysis checkpoint flush during upscale+analyze."""
import asyncio
import json
import logging
import os

from service.analysis_keyspaces_enums import PipelineStage
from service.checkpoints import build_upscale_window_progress
from service.jobs_store import JobsStore
from service.s3 import S3Client

from service.worker.progress import _make_worker_state

logger = logging.getLogger("service.worker")

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
    os.makedirs(output_dir, exist_ok=True)
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
