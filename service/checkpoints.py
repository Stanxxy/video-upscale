"""Checkpoint helpers for durable job resume decisions.

All builders emit the V1 envelope:
    {
        "schema_version": 1,
        "pending_detection": null | dict,
        "artifacts": dict,           # S3 keys live here
        "worker_state": {...},        # required progress snapshot
        ... stage-specific scalars ...
    }
"""

from __future__ import annotations

from typing import Any, NamedTuple

from service.analysis_keyspaces_enums import PipelineStage


STAGE_ORDER = [
    PipelineStage.DOWNLOAD,
    PipelineStage.DETECT,
    PipelineStage.TRACK,
    PipelineStage.UPSCALE_ANALYZE,
    PipelineStage.ANNOTATE,
    PipelineStage.UPLOAD,
    PipelineStage.PUBLISH,
]


# Sentinel used to short-circuit the tracking pass when resuming after an
# upscale/analysis crash — see CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md "Resume
# parameter forwarding". Any value past plausible video lengths works.
END_OF_TRACKING_SENTINEL = 10**9


# ---------------------------------------------------------------------------
# Envelope helper + worker_state snapshot
# ---------------------------------------------------------------------------


class WorkerStateSnapshot(NamedTuple):
    """In-memory worker progress at the time a checkpoint is written."""

    progress_percent: float
    current_frame: int
    total_frames: int
    stage_progress_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "progress_percent": float(self.progress_percent),
            "current_frame": int(self.current_frame),
            "total_frames": int(self.total_frames),
            "stage_progress_fraction": float(self.stage_progress_fraction),
        }


def make_envelope(
    *,
    worker_state: WorkerStateSnapshot,
    pending_detection: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    **extras: Any,
) -> dict[str, Any]:
    cp: dict[str, Any] = {
        "schema_version": 1,
        "pending_detection": pending_detection,
        "artifacts": dict(artifacts or {}),
        "worker_state": worker_state.to_dict(),
    }
    cp.update(extras)
    return cp


# ---------------------------------------------------------------------------
# Per-stage builders
# ---------------------------------------------------------------------------


def build_download_completed(*, worker_state: WorkerStateSnapshot) -> dict[str, Any]:
    return make_envelope(worker_state=worker_state, reason="download_completed")


def build_detect_initial_pending(
    *,
    frame_idx: int,
    frame_s3_key: str,
    frame_bucket: str,
    candidates: list[dict[str, Any]],
    suggested_boxes: list[list[float]] | None,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    return make_envelope(
        worker_state=worker_state,
        pending_detection={
            "reason": "initial",
            "frame_idx": frame_idx,
            "frame_s3_key": frame_s3_key,
            "frame_bucket": frame_bucket,
            "candidates": candidates,
            "suggested_boxes": suggested_boxes,
        },
    )


def build_track_progress(
    *,
    partial_tracking_s3_key: str | None,
    resume_from_frame: int,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"resume_from_frame": resume_from_frame}
    if partial_tracking_s3_key:
        artifacts["partial_tracking_s3_key"] = partial_tracking_s3_key
    return make_envelope(
        worker_state=worker_state,
        artifacts=artifacts,
        reason="tracking_progress",
        resume_cursor={"frame_idx": resume_from_frame},
    )


def build_track_mid_loss(
    *,
    frame_idx: int,
    frame_s3_key: str,
    frame_bucket: str,
    candidates: list[dict[str, Any]],
    suggested_boxes: list[list[float]] | None,
    partial_tracking_s3_key: str | None,
    resume_from_frame: int,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"resume_from_frame": resume_from_frame}
    if partial_tracking_s3_key:
        artifacts["partial_tracking_s3_key"] = partial_tracking_s3_key
    return make_envelope(
        worker_state=worker_state,
        pending_detection={
            "reason": "tracking_lost",
            "frame_idx": frame_idx,
            "frame_s3_key": frame_s3_key,
            "frame_bucket": frame_bucket,
            "candidates": candidates,
            "suggested_boxes": suggested_boxes,
        },
        artifacts=artifacts,
        resume_cursor={"frame_idx": resume_from_frame},
    )


def build_track_completed(
    *,
    start_frame: int,
    frame_count: int,
    worker_state: WorkerStateSnapshot,
    tracking_s3_key: str | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    if tracking_s3_key:
        artifacts["tracking_s3_key"] = tracking_s3_key
    return make_envelope(
        worker_state=worker_state,
        artifacts=artifacts,
        reason="track_completed",
        start_frame=start_frame,
        frame_count=frame_count,
    )


def build_upscale_started(
    *,
    tracking_s3_key: str,
    worker_state: WorkerStateSnapshot,
    analysis_raw_s3_key: str | None = None,
    resume_from_frame: int = 0,
    analysis_window_count: int = 0,
    analysis_current_context: str = "",
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"tracking_s3_key": tracking_s3_key}
    if analysis_raw_s3_key:
        artifacts["analysis_raw_s3_key"] = analysis_raw_s3_key
    return make_envelope(
        worker_state=worker_state,
        artifacts=artifacts,
        reason="analysis_started",
        resume_cursor={
            "frame_idx": resume_from_frame,
            "analysis_window_count": analysis_window_count,
        },
        analysis_current_context=analysis_current_context,
    )


def build_upscale_window_progress(
    *,
    frame_idx: int,
    analysis_window_count: int,
    analysis_current_context: str,
    tracking_s3_key: str,
    analysis_raw_s3_key: str,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    return make_envelope(
        worker_state=worker_state,
        artifacts={
            "tracking_s3_key": tracking_s3_key,
            "analysis_raw_s3_key": analysis_raw_s3_key,
        },
        reason="analysis_window_completed",
        resume_cursor={
            "frame_idx": frame_idx,
            "analysis_window_count": analysis_window_count,
        },
        analysis_current_context=analysis_current_context,
    )


def build_annotate_completed(
    *,
    annotated_video_s3_key: str | None,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    if annotated_video_s3_key:
        artifacts["annotated_video_s3_key"] = annotated_video_s3_key
    return make_envelope(
        worker_state=worker_state,
        artifacts=artifacts,
        reason="annotate_completed",
    )


_UPLOAD_REASON_BY_LATEST_KEY = (
    ("annotated_video_s3_key", "annotated_video_uploaded"),
    ("analysis_s3_key", "analysis_uploaded"),
    ("tracking_s3_key", "tracking_uploaded"),
)


def build_upload_incremental(
    *,
    worker_state: WorkerStateSnapshot,
    tracking_s3_key: str | None = None,
    analysis_s3_key: str | None = None,
    annotated_video_s3_key: str | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    if tracking_s3_key:
        artifacts["tracking_s3_key"] = tracking_s3_key
    if analysis_s3_key:
        artifacts["analysis_s3_key"] = analysis_s3_key
    if annotated_video_s3_key:
        artifacts["annotated_video_s3_key"] = annotated_video_s3_key
    reason = "upload_started"
    for key, candidate_reason in _UPLOAD_REASON_BY_LATEST_KEY:
        if key in artifacts:
            reason = candidate_reason
            break
    return make_envelope(
        worker_state=worker_state,
        artifacts=artifacts,
        reason=reason,
    )


def build_publish_completed(
    *,
    sns_topic_arn: str,
    sns_event_count: int,
    sns_completion_sent: bool,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    return make_envelope(
        worker_state=worker_state,
        artifacts={
            "sns_topic_arn": sns_topic_arn,
            "sns_event_count": sns_event_count,
            "sns_completion_sent": sns_completion_sent,
        },
        reason="publish_completed",
    )


def should_flush_analysis(window_count: int, every_n: int = 5) -> bool:
    """Return True when the analysis loop should write a periodic checkpoint."""
    return window_count > 0 and window_count % every_n == 0


def build_replaced_by_new_job(
    *,
    replacement_job_id: str,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    return make_envelope(
        worker_state=worker_state,
        artifacts={"replacement_job_id": replacement_job_id},
        reason="replaced_by_new_job",
    )


# ---------------------------------------------------------------------------
# Migrated existing builders (now require worker_state)
# ---------------------------------------------------------------------------


def build_verified_boxes_checkpoint(
    box_a: list[float],
    box_b: list[float],
    source_stage: PipelineStage | None,
    *,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    """Checkpoint data for boxes supplied by a correction resume request."""
    return make_envelope(
        worker_state=worker_state,
        reason="detection_correction_resume",
        source_stage=source_stage.value if source_stage else "",
        verified_box_a=box_a,
        verified_box_b=box_b,
    )


def build_cancellation_checkpoint(
    *,
    reason: str,
    worker_state: WorkerStateSnapshot,
    frame_idx: int = 0,
    progress_percent: float = 0.0,
) -> dict[str, Any]:
    """Final checkpoint data for terminal cancellation."""
    return make_envelope(
        worker_state=worker_state,
        reason=reason,
        resume_cursor={"frame_idx": frame_idx},
        progress_percent=progress_percent,
    )


# ---------------------------------------------------------------------------
# Checkpoint readers
# ---------------------------------------------------------------------------


def checkpoint_by_stage(checkpoints: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return checkpoint data keyed by stage name."""
    return {cp["stage_name"]: cp.get("checkpoint_data", {}) for cp in checkpoints}


def select_correction_checkpoint(
    checkpoints: list[dict[str, Any]],
) -> tuple[PipelineStage | None, dict[str, Any]]:
    """Prefer mid-track correction context, then initial detection context.

    Reads `artifacts.partial_tracking_s3_key` first (V1 schema) with root-level
    fallback for back-compat.
    """
    by_stage = checkpoint_by_stage(checkpoints)
    track_cp = by_stage.get(PipelineStage.TRACK.value, {})
    track_artifacts = track_cp.get("artifacts") or {}
    if (
        track_cp.get("pending_detection")
        or track_artifacts.get("partial_tracking_s3_key")
        or track_cp.get("partial_tracking_s3_key")
    ):
        return PipelineStage.TRACK, track_cp

    detect_cp = by_stage.get(PipelineStage.DETECT.value, {})
    if detect_cp.get("pending_detection"):
        return PipelineStage.DETECT, detect_cp

    return None, {}


def next_unprocessed_frame(checkpoint_data: dict[str, Any]) -> int:
    """Resolve the resume frame cursor; prefers V1 artifact location."""
    artifacts = checkpoint_data.get("artifacts") or {}
    artifact_frame = artifacts.get("resume_from_frame")
    if artifact_frame is not None:
        return int(artifact_frame)

    resume_cursor = checkpoint_data.get("resume_cursor") or {}
    cursor_frame = resume_cursor.get("frame_idx")
    if cursor_frame is not None:
        return int(cursor_frame)

    pending = checkpoint_data.get("pending_detection") or {}
    pending_frame = pending.get("frame_idx")
    if pending_frame is not None:
        return int(pending_frame)

    return int(checkpoint_data.get("frame_count", 0) or 0)


def worker_state_from(checkpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the latest pipeline-stage checkpoint's worker_state block, if any."""
    by_stage = checkpoint_by_stage(checkpoints)
    for stage in reversed(STAGE_ORDER):
        data = by_stage.get(stage.value)
        if data and data.get("worker_state"):
            return dict(data["worker_state"])
    return None


def build_resume_overrides(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Compose TrackRequest field overrides from the latest resumable checkpoint state.

    Used by both submit_detection_response (manual resume) and
    recover_interrupted_job (automatic recovery) so they stay in sync.

    Priority: when both a track checkpoint with `partial_tracking_s3_key`
    AND an upscale_analyze checkpoint with `analysis_raw_s3_key` are
    present, the upscale_analyze checkpoint TAKES PRECEDENCE because
    tracking is logically complete (the upscale stage already started).
    `resume_tracking_s3_key` is overwritten to point at the full tracking
    JSON, and `resume_from_frame` is set to `END_OF_TRACKING_SENTINEL`
    so the worker's tracking pass becomes a no-op.
    """
    overrides: dict[str, Any] = {}
    by_stage = checkpoint_by_stage(checkpoints)

    # Mid-track partial — set from track stage if either pending_detection or
    # tracking_progress wrote an artifact pointer.
    track_cp = by_stage.get(PipelineStage.TRACK.value, {})
    track_artifacts = track_cp.get("artifacts") or {}
    partial_key = track_artifacts.get("partial_tracking_s3_key") or track_cp.get(
        "partial_tracking_s3_key"
    )
    if partial_key:
        overrides["resume_tracking_s3_key"] = partial_key
        overrides["resume_from_frame"] = next_unprocessed_frame(track_cp)

    # Crash during/after upscale_analyze takes precedence: full tracking is
    # complete, so set the sentinel and forward analysis fields.
    upscale_cp = by_stage.get(PipelineStage.UPSCALE_ANALYZE.value, {})
    upscale_artifacts = upscale_cp.get("artifacts") or {}
    upscale_cursor = upscale_cp.get("resume_cursor") or {}
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

    return overrides
