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


def build_highlight_ingest_completed(
    *,
    gemini_file_uri: str,
    gemini_file_name: str,
    gemini_file_mime_type: str | None,
    gemini_file_expiration: str | None,
    player_references_ready: bool,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    """S12 Phase 1b (design §6.1) — terminal checkpoint for the
    ``HIGHLIGHT_INGEST`` stage: S3 download + Gemini Files API upload +
    reference-image fetch. ``gemini_file_expiration`` is an ISO-8601 string
    (never a raw ``datetime`` — ``write_checkpoint`` JSON-serializes
    ``checkpoint_data`` and datetimes are not JSON-serializable)."""
    return make_envelope(
        worker_state=worker_state,
        artifacts={
            "gemini_file_uri": gemini_file_uri,
            "gemini_file_name": gemini_file_name,
            "gemini_file_mime_type": gemini_file_mime_type,
            "gemini_file_expiration": gemini_file_expiration,
        },
        reason="highlight_ingest_completed",
        player_references_ready=player_references_ready,
    )


def build_highlight_chunk_completed(
    *,
    chunk_index: int,
    chunks_total: int,
    highlights_scanned: int,
    highlights_analyzed: int,
    highlights_ditched: int,
    highlights_published: int,
    gemini_file_uri: str,
    worker_state: WorkerStateSnapshot,
    clips: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """S12 Phase 1b (design §3.4/§6.3) — written after ONE outer chunk's
    full scan->critique->analyze sequence completes. Chunk-granularity
    resume identity (``build_highlight_resume_plan`` reads ``chunk_index``
    off this checkpoint's own top-level key, not nested under
    ``artifacts`` — mirrors ``build_track_completed``'s
    ``start_frame``/``frame_count`` top-level scalars).

    ``highlights_published`` is now ALWAYS 0 (2026-07-26 CEO batched-publish
    re-scope, AC8-11): publish moved out of the per-chunk loop entirely, to
    one finalize pass after majority-vote reconciliation (see
    ``worker/highlight_orchestrator.py::run_highlight_job``) — nothing is
    ever published at chunk-completion time anymore. The field is kept
    (never removed — additive-only checkpoint-schema discipline; an old
    consumer reading this key sees an honest 0, not a missing key) rather
    than repurposed to mean something else under the same name.

    ``clips`` (additive, AC8-11): this chunk's own collected-and-seam-
    deduped-eligible clip dicts (raw actor-axis output included, BEFORE
    match-wide majority-vote reconciliation) — the durable, checkpoint-
    backed source of truth the finalize step reconstructs the FULL match's
    candidate set from, regardless of whether a given chunk was freshly
    analyzed this run or its clips are being recovered from an earlier,
    resumed-past run's completed checkpoint. ``None``/omitted (an
    OLDER-format checkpoint written before this field existed) is treated
    as an empty list downstream — additive-only, never a crash, never
    fabricated clips."""
    return make_envelope(
        worker_state=worker_state,
        artifacts={"gemini_file_uri": gemini_file_uri, "clips": clips or []},
        reason="highlight_chunk_completed",
        chunk_index=chunk_index,
        chunks_total=chunks_total,
        highlights_scanned=highlights_scanned,
        highlights_analyzed=highlights_analyzed,
        highlights_ditched=highlights_ditched,
        highlights_published=highlights_published,
    )


def build_highlight_publish_completed(
    *,
    sns_topic_arn: str,
    sns_event_count: int,
    sns_completion_sent: bool,
    result_s3_uri: str | None,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    """S12 Phase 1b (design §5.4) — terminal checkpoint for the
    ``HIGHLIGHT_PUBLISH`` stage: the final ``analysis_complete`` SNS event
    has been sent. Mirrors ``build_publish_completed`` (the tracking
    pipeline's own terminal-publish builder) but has no
    ``tracking_s3_uri`` — v2 has no tracking artifact at all (decision 3)."""
    return make_envelope(
        worker_state=worker_state,
        artifacts={
            "sns_topic_arn": sns_topic_arn,
            "sns_event_count": sns_event_count,
            "sns_completion_sent": sns_completion_sent,
            "result_s3_uri": result_s3_uri,
        },
        reason="highlight_publish_completed",
    )


def build_highlight_publish_progress(
    *,
    candidate_key: str,
    event_index: int,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    """2026-07-26 CEO batched-publish re-scope (AC8-11) — Brooks's named new
    requirement: written ONCE per candidate successfully published during
    the finalize-publish loop, so a crash mid-batch (e.g. after 30 of 50)
    can resume publishing only the remaining candidates, never double-sends.

    ``completed=False`` on EVERY row this builder produces (passed by the
    caller to ``jobs_store.write_checkpoint`` — never ``True``) — this is
    deliberately NOT the terminal ``HIGHLIGHT_PUBLISH`` checkpoint
    (``build_highlight_publish_completed``, unchanged, still written exactly
    once, with ``completed=True``, after every candidate has published).
    Multiple rows accumulate under the SAME ``HIGHLIGHT_PUBLISH`` stage name
    (the schema already supports this — ``HIGHLIGHT_CHUNK`` does the exact
    same one-row-per-chunk-index thing); a resume reads ALL of them and
    treats every row's own ``candidate_key`` as "already published,"
    regardless of the ``completed`` flag, while the terminal row (a
    different ``reason``, no ``candidate_key`` field at all) never collides
    with this shape.

    ``candidate_key`` (additive, new field, no collision with anything
    existing): a stable per-highlight identity — see
    ``worker/highlight_orchestrator.py``'s own construction
    (``f"{chunk_index}:{highlight_index}"``) — deterministic across a
    resume because it is derived from data already persisted on the
    ``HIGHLIGHT_CHUNK`` checkpoint, never re-derived from in-memory-only
    state that a resume wouldn't have."""
    return make_envelope(
        worker_state=worker_state,
        artifacts={"candidate_key": candidate_key, "event_index": event_index},
        reason="highlight_publish_candidate",
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
