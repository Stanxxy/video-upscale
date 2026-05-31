"""Per-stage V1 checkpoint builders."""

from __future__ import annotations

from typing import Any

from service.analysis_keyspaces_enums import PipelineStage

from service.checkpoints.envelope import WorkerStateSnapshot, make_envelope


def build_download_completed(*, worker_state: WorkerStateSnapshot) -> dict[str, Any]:
    return make_envelope(worker_state=worker_state, reason="download_completed")


def build_detect_initial_pending(
    *,
    frame_idx: int,
    frame_s3_key: str,
    frame_bucket: str,
    candidates: list[dict[str, Any]],
    suggested_boxes: Any | None,
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


def build_tracking_started(
    *,
    clip_start_frame: int,
    clip_end_frame: int | None,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    """Lightweight checkpoint when SAM2 tracking begins (visibility only)."""
    extras: dict[str, Any] = {
        "reason": "tracking_started",
        "resume_cursor": {"frame_idx": clip_start_frame},
    }
    if clip_end_frame is not None:
        extras["clip_end_frame"] = clip_end_frame
    return make_envelope(worker_state=worker_state, **extras)


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
    suggested_boxes: Any | None,
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
