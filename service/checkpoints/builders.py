"""Per-stage V1 checkpoint builders."""

from __future__ import annotations

from typing import Any

from service.checkpoints.envelope import WorkerStateSnapshot, make_envelope


def build_highlight_ingest_completed(
    *,
    gemini_file_uri: str,
    gemini_file_name: str,
    gemini_file_mime_type: str | None,
    gemini_file_expiration: str | None,
    worker_state: WorkerStateSnapshot,
) -> dict[str, Any]:
    """S12 Phase 1b (design §6.1) — terminal checkpoint for the
    ``HIGHLIGHT_INGEST`` stage: S3 download + Gemini Files API upload.
    ``gemini_file_expiration`` is an ISO-8601 string (never a raw
    ``datetime`` — ``write_checkpoint`` JSON-serializes ``checkpoint_data``
    and datetimes are not JSON-serializable)."""
    return make_envelope(
        worker_state=worker_state,
        artifacts={
            "gemini_file_uri": gemini_file_uri,
            "gemini_file_name": gemini_file_name,
            "gemini_file_mime_type": gemini_file_mime_type,
            "gemini_file_expiration": gemini_file_expiration,
        },
        reason="highlight_ingest_completed",
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
    ``artifacts``).

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
    has been sent."""
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
