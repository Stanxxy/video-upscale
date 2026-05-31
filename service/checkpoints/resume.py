"""Resume plan composition from durable checkpoints."""

from __future__ import annotations

from typing import Any, NamedTuple

from service.analysis_keyspaces_enums import PipelineStage

from service.checkpoints.constants import END_OF_TRACKING_SENTINEL
from service.checkpoints.query import (
    latest_checkpoint_data_by_stage,
    next_unprocessed_frame,
    worker_state_from,
)


class ResumePlan(NamedTuple):
    """Deterministic resume decisions from durable checkpoints."""

    track_request_overrides: dict[str, Any]
    """Fields to merge into ``TrackRequest`` (resume keys, analysis hints)."""

    skip_tracking_runner: bool
    """If True, worker must not call ``run_tracking_job`` — load tracking JSON from S3."""

    pipeline_already_complete: bool
    """Job finished successfully; recovery should not spawn a replacement worker."""

    seed_progress_floor: float
    """Minimum lifecycle progress_percent — replacement jobs must not regress below this."""

    existing_upload_artifacts: dict[str, str | None]
    """Latest cumulative upload-row artifact keys (skip re-upload when they match)."""

    terminal_publish_complete: bool
    """Latest publish checkpoint indicates SNS stage already finished."""


def build_resume_plan(checkpoints: list[dict[str, Any]]) -> ResumePlan:
    """Compose resume routing from the latest checkpoint row per stage."""
    from service.tracking_chain_merge import resolve_best_tracking_keys_from_checkpoints

    by_stage = latest_checkpoint_data_by_stage(checkpoints)
    overrides: dict[str, Any] = {}

    track_cp = by_stage.get(PipelineStage.TRACK.value, {})
    full_tracking_key, partial_key = resolve_best_tracking_keys_from_checkpoints(
        checkpoints,
    )

    upscale_cp = by_stage.get(PipelineStage.UPSCALE_ANALYZE.value, {})
    upscale_artifacts = upscale_cp.get("artifacts") or {}
    upscale_cursor = upscale_cp.get("resume_cursor") or {}

    pipeline_already_complete = False
    for cp in checkpoints:
        sn = cp.get("stage_name")
        if sn == PipelineStage.PUBLISH.value and cp.get("completed"):
            pipeline_already_complete = True
            break
        if sn == PipelineStage.UPLOAD.value and cp.get("completed"):
            pipeline_already_complete = True
            break

    seed_ws = worker_state_from(checkpoints) or {}
    seed_floor = float(seed_ws.get("progress_percent") or 0.0)

    upload_data = by_stage.get(PipelineStage.UPLOAD.value, {})
    uarts = upload_data.get("artifacts") or {}
    existing_upload_artifacts: dict[str, str | None] = {
        "tracking_s3_key": uarts.get("tracking_s3_key"),
        "analysis_s3_key": uarts.get("analysis_s3_key"),
        "annotated_video_s3_key": uarts.get("annotated_video_s3_key"),
    }

    terminal_publish_complete = False
    pub_latest = by_stage.get(PipelineStage.PUBLISH.value, {})
    if pub_latest.get("reason") == "publish_completed":
        terminal_publish_complete = True

    skip_tracking_runner = False

    if upscale_artifacts.get("analysis_raw_s3_key"):
        upscale_tracking_key = upscale_artifacts.get("tracking_s3_key")
        if upscale_tracking_key:
            overrides["resume_tracking_s3_key"] = upscale_tracking_key
        overrides["resume_from_frame"] = END_OF_TRACKING_SENTINEL
        overrides["analysis_raw_s3_key"] = upscale_artifacts["analysis_raw_s3_key"]
        overrides["analysis_window_count"] = int(
            upscale_cursor.get("analysis_window_count", 0)
        )
        overrides["analysis_current_context"] = upscale_cp.get(
            "analysis_current_context", ""
        )
        skip_tracking_runner = True

    elif full_tracking_key:
        overrides["resume_tracking_s3_key"] = full_tracking_key
        overrides["resume_from_frame"] = END_OF_TRACKING_SENTINEL
        skip_tracking_runner = True

    elif partial_key:
        overrides["resume_tracking_s3_key"] = partial_key
        overrides["resume_from_frame"] = next_unprocessed_frame(track_cp)

    return ResumePlan(
        track_request_overrides=overrides,
        skip_tracking_runner=skip_tracking_runner,
        pipeline_already_complete=pipeline_already_complete,
        seed_progress_floor=seed_floor,
        existing_upload_artifacts=existing_upload_artifacts,
        terminal_publish_complete=terminal_publish_complete,
    )


def resume_plan_to_request_fields(plan: ResumePlan) -> dict[str, Any]:
    """Flatten upload/publish hints for :class:`TrackRequest` merge."""
    eu = plan.existing_upload_artifacts
    return {
        "resume_existing_upload_tracking_key": eu.get("tracking_s3_key"),
        "resume_existing_upload_analysis_key": eu.get("analysis_s3_key"),
        "resume_existing_upload_annotated_key": eu.get("annotated_video_s3_key"),
        "resume_terminal_publish_done": plan.terminal_publish_complete,
    }


def build_resume_overrides(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Compose TrackRequest field overrides from the latest resumable checkpoint state.

    Used by both submit_detection_response (manual resume) and
    recover_interrupted_job (automatic recovery) so they stay in sync.

    Delegates to :func:`build_resume_plan` — see ``ResumePlan`` for precedence.
    """
    return dict(build_resume_plan(checkpoints).track_request_overrides)
