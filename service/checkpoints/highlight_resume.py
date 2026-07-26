"""Resume plan composition for the v2 highlight job (S12 Phase 1b design §6.3).

Dramatically simpler than ``service/checkpoints/resume.py``'s
``build_resume_plan`` (the tracking pipeline's own resume model): no SAM2
frame-cursor math, no upscale-window cursor, no partial-tracking-JSON merge.
Resume = re-invoke ``run_highlight_job`` for the SAME ``job_id``, skipping
outer chunks already checkpointed as complete.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, TypedDict

from service.analysis_keyspaces_enums import PipelineStage


class HighlightResumePlan(TypedDict):
    resume_from_chunk_index: int
    gemini_file_uri: Optional[str]
    gemini_file_name: Optional[str]
    gemini_file_mime_type: Optional[str]
    gemini_file_expiration: Optional[datetime]
    player_references_ready: bool


def _parse_iso(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def build_highlight_resume_plan(checkpoints: list[dict[str, Any]]) -> HighlightResumePlan:
    """Compose the v2 resume plan from durable HIGHLIGHT_INGEST/HIGHLIGHT_CHUNK
    checkpoints.

    ``resume_from_chunk_index`` = ``1 + max(chunk_index for completed
    HIGHLIGHT_CHUNK checkpoints, default -1)`` — i.e. chunk 0 when no chunk
    has completed yet. The Gemini Files API identity (uri/name/mime_type/
    expiration) and ``player_references_ready`` are read from the LATEST
    ``HIGHLIGHT_INGEST`` checkpoint, if any — the caller (``run_highlight_job``)
    uses ``gemini_file_expiration`` to decide whether to reuse the existing
    upload or re-upload (design §2.3's "safety margin" check).
    """
    max_chunk_index = -1
    for cp in checkpoints:
        if cp.get("stage_name") != PipelineStage.HIGHLIGHT_CHUNK.value:
            continue
        if not cp.get("completed"):
            continue
        data = cp.get("checkpoint_data") or {}
        idx = data.get("chunk_index")
        if isinstance(idx, int) and idx > max_chunk_index:
            max_chunk_index = idx

    ingest_cp = None
    for cp in checkpoints:
        if cp.get("stage_name") == PipelineStage.HIGHLIGHT_INGEST.value:
            ingest_cp = cp  # last one wins if duplicated — same "latest write" convention as elsewhere

    gemini_file_uri: Optional[str] = None
    gemini_file_name: Optional[str] = None
    gemini_file_mime_type: Optional[str] = None
    gemini_file_expiration: Optional[datetime] = None
    player_references_ready = False

    if ingest_cp is not None:
        data = ingest_cp.get("checkpoint_data") or {}
        artifacts = data.get("artifacts") or {}
        gemini_file_uri = artifacts.get("gemini_file_uri")
        gemini_file_name = artifacts.get("gemini_file_name")
        gemini_file_mime_type = artifacts.get("gemini_file_mime_type")
        gemini_file_expiration = _parse_iso(artifacts.get("gemini_file_expiration"))
        player_references_ready = bool(data.get("player_references_ready"))

    return HighlightResumePlan(
        resume_from_chunk_index=max_chunk_index + 1,
        gemini_file_uri=gemini_file_uri,
        gemini_file_name=gemini_file_name,
        gemini_file_mime_type=gemini_file_mime_type,
        gemini_file_expiration=gemini_file_expiration,
        player_references_ready=player_references_ready,
    )
