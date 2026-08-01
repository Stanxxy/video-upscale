"""HIGHLIGHT_INGEST stage (S12 Phase 1b production wiring design §1.3/§2.3):
S3 download + Gemini Files API upload/poll + reference-image fetch — one
cohesive "ingest" stage, not three separate worker-stage files, per the
design's own reasoning (three sequential I/O steps that always run together).
"""
from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from google import genai

from service.analysis_keyspaces_enums import PipelineStage
from service.checkpoints import build_highlight_ingest_completed
from service.checkpoints.highlight_resume import build_highlight_resume_plan
from service.config import ServiceConfig
from service.jobs_store import JobsStore
from service.models import TrackRequest
from service.pipelines import gemini_upload
from service.worker.helpers import _make_s3
from service.worker.highlight_progress import (
    PREPARING,
    HighlightProgressWriter,
)
from service.worker.progress import _make_worker_state

logger = logging.getLogger("service.worker")

# Belt-and-suspenders safety margin (design §2.3) — a resumed job reuses an
# existing Gemini Files API upload only if it has at least this much time
# left before the server's 48h TTL, so an in-flight resumed run never races
# expiration mid-job.
_REUSE_SAFETY_MARGIN = timedelta(minutes=10)


@dataclass
class HighlightIngestResult:
    video_path: str
    gemini_file_uri: str
    gemini_file_name: str
    gemini_file_mime_type: Optional[str]
    gemini_file_expiration: Optional[datetime]
    video_duration_sec: float
    player_references: list = field(default_factory=list)


def _probe_duration_sec(video_path: str) -> float:
    """OpenCV-based duration probe (frame_count / fps) — kept local to this
    stage (not shared with frame_source.py's ffprobe-based native_fps probe,
    which is a QA-playground-only concern for raw frame sampling)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0:
            raise RuntimeError(f"could not determine fps for {video_path!r} (fps={fps})")
        return frame_count / fps
    finally:
        cap.release()


def _binding_field(binding, name: str):
    """Support both AthleteBinding pydantic objects and plain dicts (same
    duck-typing convention already used by sns.py::_bindings_by_player_id)."""
    if isinstance(binding, dict):
        return binding.get(name)
    return getattr(binding, name, None)


async def _fetch_player_references(s3, bucket: str, athlete_bindings, loop) -> list[dict]:
    """Download each ``athlete_bindings[].s3_key`` reference-capture crop
    (design §4.2) — inline bytes, fetched ONCE per job. A binding missing
    ``s3_key``/``player_id`` is skipped (logged), never fabricated."""
    refs: list[dict] = []
    for binding in athlete_bindings or []:
        player_id = _binding_field(binding, "player_id")
        s3_key = _binding_field(binding, "s3_key")
        player_name = _binding_field(binding, "player_name")
        if not player_id or not s3_key:
            logger.warning(
                "highlight_ingest: athlete_binding missing player_id/s3_key, skipping "
                "reference image: %r",
                binding,
            )
            continue
        image_bytes = await loop.run_in_executor(None, s3.get_object, bucket, s3_key)
        mime_type = mimetypes.guess_type(s3_key)[0] or "image/jpeg"
        refs.append({
            "player_id": player_id,
            "player_name": player_name,
            "image_bytes": image_bytes,
            "mime_type": mime_type,
        })
    return refs


async def run_highlight_ingest_stage(
    job_id: str,
    request: TrackRequest,
    config: ServiceConfig,
    jobs_store: JobsStore,
    work_dir: str,
    gemini_client: genai.Client,
    *,
    progress_writer: HighlightProgressWriter | None = None,
) -> HighlightIngestResult:
    """S3 download -> Gemini Files API upload/poll (or reuse a non-expired
    upload from a prior HIGHLIGHT_INGEST checkpoint on resume) -> reference-
    image fetch -> writes the HIGHLIGHT_INGEST checkpoint.

    Non-ACTIVE Files-API terminal states RAISE (via ``gemini_upload.
    poll_until_active``) — never proceeds with a fabricated "ready" state.
    """
    loop = asyncio.get_event_loop()
    progress = progress_writer or HighlightProgressWriter(job_id, jobs_store)
    await progress.write(PipelineStage.HIGHLIGHT_INGEST, 1.0, phase=PREPARING)

    s3 = _make_s3(config)
    s3.ensure_bucket(request.bucket)

    video_path = await loop.run_in_executor(
        None, s3.download_file, request.bucket, request.key,
        os.path.join(work_dir, "video.mp4"),
    )
    duration_sec = await loop.run_in_executor(None, _probe_duration_sec, video_path)
    logger.info(
        "Job %s: highlight_ingest downloaded video_path=%s duration_sec=%.1f",
        job_id, video_path, duration_sec,
    )
    await progress.write(PipelineStage.HIGHLIGHT_INGEST, 4.0, phase=PREPARING)

    checkpoints = await jobs_store.get_all_checkpoints(job_id)
    resume_plan = build_highlight_resume_plan(checkpoints)

    now = datetime.now(timezone.utc)
    reuse = (
        resume_plan["gemini_file_uri"]
        and resume_plan["gemini_file_name"]
        and resume_plan["gemini_file_expiration"] is not None
        and resume_plan["gemini_file_expiration"] > now + _REUSE_SAFETY_MARGIN
    )

    if reuse:
        gemini_file_uri = resume_plan["gemini_file_uri"]
        gemini_file_name = resume_plan["gemini_file_name"]
        gemini_file_mime_type = resume_plan["gemini_file_mime_type"]
        gemini_file_expiration = resume_plan["gemini_file_expiration"]
        logger.info(
            "Job %s: highlight_ingest reusing non-expired Gemini Files API upload "
            "%s (expires %s) — skipping re-upload",
            job_id, gemini_file_name, gemini_file_expiration.isoformat(),
        )
    else:
        uploaded = await gemini_upload.upload_video_to_gemini(gemini_client, video_path)
        active = await gemini_upload.poll_until_active(
            gemini_client, uploaded.name,
            poll_interval_sec=config.gemini_upload_poll_interval_sec,
            timeout_sec=config.gemini_upload_poll_timeout_sec,
        )
        gemini_file_uri = active.uri
        gemini_file_name = active.name
        gemini_file_mime_type = active.mime_type
        gemini_file_expiration = active.expiration_time
        logger.info(
            "Job %s: highlight_ingest uploaded video to Gemini Files API name=%s uri=%s",
            job_id, gemini_file_name, gemini_file_uri,
        )
    await progress.write(PipelineStage.HIGHLIGHT_INGEST, 7.0, phase=PREPARING)

    # References are stored in the SAME bucket as the source video — same
    # established convention as upscale_setup.py:82 (`ref_bucket =
    # request.bucket  # references stored in same bucket`), NOT
    # output_bucket (a bucket-split deployment has output_bucket != bucket;
    # reference crops were never written to output_bucket).
    player_references = await _fetch_player_references(
        s3, request.bucket, request.athlete_bindings, loop,
    )
    await progress.write(PipelineStage.HIGHLIGHT_INGEST, 10.0, phase=PREPARING)

    await jobs_store.write_checkpoint(
        job_id, PipelineStage.HIGHLIGHT_INGEST, True,
        build_highlight_ingest_completed(
            gemini_file_uri=gemini_file_uri,
            gemini_file_name=gemini_file_name,
            gemini_file_mime_type=gemini_file_mime_type,
            gemini_file_expiration=(
                gemini_file_expiration.isoformat() if gemini_file_expiration else None
            ),
            player_references_ready=True,
            worker_state=_make_worker_state(
                progress_percent=10.0, stage_progress_fraction=1.0,
            ),
        ),
    )

    return HighlightIngestResult(
        video_path=video_path,
        gemini_file_uri=gemini_file_uri,
        gemini_file_name=gemini_file_name,
        gemini_file_mime_type=gemini_file_mime_type,
        gemini_file_expiration=gemini_file_expiration,
        video_duration_sec=duration_sec,
        player_references=player_references,
    )
