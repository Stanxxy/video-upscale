"""Mirrors shared_lib.models.analysis_keyspaces_enums — standalone copy for the engine."""

from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def states_with_completed_at() -> frozenset[JobState]:
    return frozenset(
        {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    )


def states_that_sync_latest_job_row() -> frozenset[JobState]:
    return frozenset(
        {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.INTERRUPTED,
        }
    )


class PipelineStage(StrEnum):
    # The highlight-scan-critique-analyze job lifecycle's stage set.
    HIGHLIGHT_INGEST = "highlight_ingest"   # S3 download + Gemini Files API upload + reference images
    HIGHLIGHT_CHUNK = "highlight_chunk"     # one checkpoint per completed outer chunk
    HIGHLIGHT_PUBLISH = "highlight_publish"  # terminal: analysis_complete sent
    CANCELLATION = "cancellation"           # durable record of a user cancel request


# Historical Keyspaces rows written by the retired local tracking/upscale
# pipeline. They are read explicitly as a retired, unrecoverable job (see
# ``jobs_store.lifecycle.LifecycleMixin.get_lifecycle``); no code resumes them.
LEGACY_AWAITING_CORRECTION_STATE = "AWAITING_CORRECTION"
RETIRED_TRACKING_STAGES = frozenset(
    {"download", "detect", "track", "upscale_analyze", "annotate", "upload", "publish"}
)
RETIRED_TRACKING_MESSAGE = (
    "This job ran on the retired local tracking pipeline and cannot be resumed. "
    "Start a new analysis."
)


def parse_job_state(raw: str | None) -> JobState:
    if raw is None or raw == "":
        return JobState.PENDING
    try:
        return JobState(raw)
    except ValueError:
        raise ValueError(f"invalid Keyspaces job_state value: {raw!r}") from None


def parse_pipeline_stage_optional(raw: str | None) -> PipelineStage | None:
    if raw is None or raw == "":
        return None
    try:
        return PipelineStage(raw)
    except ValueError:
        raise ValueError(f"invalid Keyspaces lifecycle.stage value: {raw!r}") from None


def parse_pipeline_stage_strict(raw: str | None, *, label: str) -> PipelineStage:
    if raw is None or raw == "":
        raise ValueError(f"missing pipeline stage ({label})")
    try:
        return PipelineStage(raw)
    except ValueError:
        raise ValueError(f"invalid Keyspaces checkpoint stage_name ({label}): {raw!r}") from None
