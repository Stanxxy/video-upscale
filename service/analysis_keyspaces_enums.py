"""Mirrors shared_lib.models.analysis_keyspaces_enums — standalone copy for the engine."""

from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_CORRECTION = "AWAITING_CORRECTION"
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
            JobState.AWAITING_CORRECTION,
        }
    )


class PipelineStage(StrEnum):
    DOWNLOAD = "download"
    DETECT = "detect"
    TRACK = "track"
    UPSCALE_ANALYZE = "upscale_analyze"
    ANNOTATE = "annotate"
    UPLOAD = "upload"
    PUBLISH = "publish"


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
