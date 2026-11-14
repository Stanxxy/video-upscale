"""Checkpoint helpers for durable highlight-job resume decisions.

All builders emit the V1 envelope:
    {
        "schema_version": 1,
        "artifacts": dict,
        "worker_state": {...},
        ... stage-specific scalars ...
    }
"""

from service.checkpoints.builders import (
    build_cancellation_checkpoint,
    build_highlight_chunk_completed,
    build_highlight_ingest_completed,
    build_highlight_publish_completed,
    build_highlight_publish_progress,
)
from service.checkpoints.constants import HIGHLIGHT_STAGE_ORDER
from service.checkpoints.envelope import WorkerStateSnapshot, make_envelope
from service.checkpoints.highlight_resume import build_highlight_resume_plan

__all__ = [
    "HIGHLIGHT_STAGE_ORDER",
    "WorkerStateSnapshot",
    "make_envelope",
    "build_cancellation_checkpoint",
    "build_highlight_ingest_completed",
    "build_highlight_chunk_completed",
    "build_highlight_publish_completed",
    "build_highlight_publish_progress",
    "build_highlight_resume_plan",
]
