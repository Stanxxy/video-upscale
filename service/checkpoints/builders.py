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
    clips_by_chunk: dict[str, list[dict[str, Any]]] | None = None,
    highlights_collected_by_chunk: dict[str, int] | None = None,
) -> dict[str, Any]:
    """S12 Phase 1b (design §3.4/§6.3) — written after ONE outer chunk's
    full scan->critique->analyze sequence completes. Chunk-granularity
    resume identity (``build_highlight_resume_plan`` reads ``chunk_index``
    off this checkpoint's own top-level key, not nested under
    ``artifacts`` — mirrors ``build_track_completed``'s
    ``start_frame``/``frame_count`` top-level scalars).

    **CORRECTED 2026-07-26 (Evaluator REJECT on the first version of this
    field, `clips` — that version was wrong):** ``job_stage_checkpoints``'s
    real schema is ``PRIMARY KEY (job_id, stage_name)``
    (``bjj-vision-backend/infrastructure/keyspaces/migrations/
    005_job_stage_checkpoints.cql``) — an ``INSERT`` is an UPSERT on that
    key, so AT MOST ONE ROW SURVIVES per ``(job_id, stage_name)``; every
    ``write_checkpoint(job_id, HIGHLIGHT_CHUNK, ...)`` call OVERWRITES the
    previous one. The documented convention for this exact table
    (``working_log/knowledge-base/references/
    2026-05-02-checkpoint-artifacts-v1-addendum.md``: "Cassandra/Keyspaces
    overwrites the previous JSON value when (job_id, stage_name) repeats,
    so the latest row contains the cumulative state") is READ-LATEST-THEN-
    MERGE-CUMULATIVE-WRITE, never one row per chunk. ``clips_by_chunk``
    (keyed by ``str(chunk_index)``) and ``highlights_collected_by_chunk``
    (same keys, ``len(that chunk's clips)`` — an independent integrity
    signal, never derived from ``clips_by_chunk`` itself, so Phase-2
    reconstruction can detect data loss rather than silently trusting its
    own input) are the FULL, merged maps for every chunk completed so far
    — the caller (``worker/highlight_orchestrator.py::run_highlight_job``)
    is responsible for reading the current latest row, merging THIS
    chunk's entry in, and passing the complete merged maps here before
    every write. Safe under this project's existing single-writer-per-job
    invariant (see ``run_highlight_job``'s own docstring) — never safe
    under concurrent writers to the same ``job_id``, which this codebase
    does not have.

    ``highlights_published`` is now ALWAYS 0 (2026-07-26 CEO batched-publish
    re-scope, AC8-11): publish moved out of the per-chunk loop entirely, to
    one finalize pass after majority-vote reconciliation — nothing is ever
    published at chunk-completion time anymore. The field is kept (never
    removed — additive-only checkpoint-schema discipline; an old consumer
    reading this key sees an honest 0, not a missing key) rather than
    repurposed to mean something else under the same name."""
    return make_envelope(
        worker_state=worker_state,
        artifacts={
            "gemini_file_uri": gemini_file_uri,
            "clips_by_chunk": clips_by_chunk or {},
            "highlights_collected_by_chunk": highlights_collected_by_chunk or {},
        },
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
    published_candidate_keys: list[str],
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    """2026-07-26 CEO batched-publish re-scope (AC8-11) — Brooks's named new
    requirement: written after EVERY candidate successfully published during
    the finalize-publish loop, so a crash mid-batch (e.g. after 30 of 50)
    can resume publishing only the remaining candidates, never double-sends.

    **CORRECTED 2026-07-26 (Evaluator REJECT on the first version of this
    builder — that version's own docstring claim, "multiple rows accumulate
    under the SAME stage name, the schema already supports this," was
    FALSE):** ``job_stage_checkpoints`` has ``PRIMARY KEY (job_id,
    stage_name)`` — an ``INSERT`` is an UPSERT, so at most ONE row survives
    per ``(job_id, HIGHLIGHT_PUBLISH)``. Writing one row per candidate (the
    original design) would have each write OVERWRITE the previous one,
    losing every earlier candidate's "published" record except the very
    last — the exact double-send bug this checkpoint exists to prevent.

    ``published_candidate_keys`` is now the FULL, cumulative list of every
    candidate_key published so far THIS job (not just this one) — the
    caller reads the current latest ``HIGHLIGHT_PUBLISH`` row, appends this
    candidate's key, and passes the complete merged list here before every
    write (read-latest-then-merge-cumulative-write, the documented
    convention for this table — see ``build_highlight_chunk_completed``'s
    own docstring for the KB reference). ``candidate_key``/``event_index``
    are kept as "most recently added" fields (useful for a human reading
    one row without decoding the whole list), but ``published_candidate_keys``
    is the load-bearing field a resume actually reads.

    ``completed=False`` on EVERY row this builder produces (passed by the
    caller to ``jobs_store.write_checkpoint`` — never ``True``) — this is
    deliberately NOT the terminal ``HIGHLIGHT_PUBLISH`` checkpoint
    (``build_highlight_publish_completed``, unchanged, still written exactly
    once, with ``completed=True``, after every candidate has published,
    OVERWRITING the last progress row — the terminal state is the final
    state, that overwrite is intentional and correct)."""
    return make_envelope(
        worker_state=worker_state,
        artifacts={
            "candidate_key": candidate_key,
            "event_index": event_index,
            "published_candidate_keys": list(published_candidate_keys),
        },
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
