"""Unit tests for service.checkpoints builders — V1 envelope conformance.

Every builder MUST emit the V1 envelope shape:
{
    "schema_version": 1,
    "artifacts": dict,
    "worker_state": {
        "progress_percent": float,
        "current_frame": int,
        "total_frames": int,
        "stage_progress_fraction": float,
    },
    ... stage-specific scalar fields ...
}
"""
from service.analysis_keyspaces_enums import PipelineStage

from service.checkpoints import (
    WorkerStateSnapshot,
    make_envelope,
    build_cancellation_checkpoint,
    build_highlight_ingest_completed,
    build_highlight_chunk_completed,
    build_highlight_publish_completed,
    build_highlight_publish_progress,
)
from service.checkpoints.constants import HIGHLIGHT_STAGE_ORDER


V1_KEYS = {"schema_version", "artifacts", "worker_state"}


def _ws(progress_percent=10.0, current_frame=0, total_frames=0, stage_progress_fraction=1.0):
    return WorkerStateSnapshot(
        progress_percent=progress_percent,
        current_frame=current_frame,
        total_frames=total_frames,
        stage_progress_fraction=stage_progress_fraction,
    )


def _assert_envelope(cp: dict) -> None:
    assert V1_KEYS <= cp.keys(), f"missing envelope keys: {V1_KEYS - cp.keys()}"
    assert cp["schema_version"] == 1
    assert isinstance(cp["artifacts"], dict)
    ws = cp["worker_state"]
    assert {
        "progress_percent",
        "current_frame",
        "total_frames",
        "stage_progress_fraction",
    } <= ws.keys()
    assert isinstance(ws["progress_percent"], (int, float))
    assert isinstance(ws["current_frame"], int)
    assert isinstance(ws["total_frames"], int)
    assert isinstance(ws["stage_progress_fraction"], (int, float))


# ---------------------------------------------------------------------------
# Envelope helper
# ---------------------------------------------------------------------------


def test_make_envelope_requires_worker_state():
    cp = make_envelope(worker_state=_ws())
    _assert_envelope(cp)
    assert cp["artifacts"] == {}


def test_make_envelope_passes_through_extras():
    cp = make_envelope(worker_state=_ws(), reason="custom", resume_cursor={"frame_idx": 7})
    _assert_envelope(cp)
    assert cp["reason"] == "custom"
    assert cp["resume_cursor"] == {"frame_idx": 7}


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancellation_envelope_keeps_resume_cursor():
    cp = build_cancellation_checkpoint(
        reason="user_cancelled",
        frame_idx=42,
        progress_percent=33.3,
        worker_state=_ws(
            progress_percent=33.3,
            current_frame=42,
            total_frames=100,
            stage_progress_fraction=0.42,
        ),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "user_cancelled"
    assert cp["resume_cursor"] == {"frame_idx": 42}
    assert cp["progress_percent"] == 33.3


# ---------------------------------------------------------------------------
# S12 Phase 1b — v2 highlight job checkpoint builders (item 7).
# ---------------------------------------------------------------------------
def test_highlight_ingest_completed_envelope_shape():
    cp = build_highlight_ingest_completed(
        gemini_file_uri="files/abc", gemini_file_name="files/abc",
        gemini_file_mime_type="video/mp4", gemini_file_expiration="2026-07-22T12:00:00+00:00",
        worker_state=_ws(progress_percent=5.0),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "highlight_ingest_completed"
    assert cp["artifacts"]["gemini_file_uri"] == "files/abc"
    assert cp["artifacts"]["gemini_file_mime_type"] == "video/mp4"


def test_highlight_chunk_completed_envelope_shape():
    cp = build_highlight_chunk_completed(
        chunk_index=2, chunks_total=5, highlights_scanned=4,
        highlights_analyzed=3, highlights_ditched=1, highlights_published=3,
        gemini_file_uri="files/abc",
        worker_state=_ws(progress_percent=40.0),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "highlight_chunk_completed"
    assert cp["chunk_index"] == 2
    assert cp["chunks_total"] == 5
    assert cp["highlights_scanned"] == 4
    assert cp["highlights_analyzed"] == 3
    assert cp["highlights_ditched"] == 1
    assert cp["highlights_published"] == 3
    assert cp["artifacts"]["gemini_file_uri"] == "files/abc"


def test_highlight_publish_completed_envelope_shape():
    cp = build_highlight_publish_completed(
        sns_topic_arn="arn:aws:sns:x", sns_event_count=7, sns_completion_sent=True,
        result_s3_uri="s3://bucket/base_key_v2_events.json",
        worker_state=_ws(progress_percent=100.0, stage_progress_fraction=1.0),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "highlight_publish_completed"
    assert cp["artifacts"]["sns_event_count"] == 7
    assert cp["artifacts"]["result_s3_uri"] == "s3://bucket/base_key_v2_events.json"
    assert "tracking_s3_uri" not in cp["artifacts"]


# ---------------------------------------------------------------------------
# 2026-07-26 CEO batched-publish re-scope (AC8-11) — additive clips field on
# HIGHLIGHT_CHUNK, and the new per-candidate HIGHLIGHT_PUBLISH progress row.
# ---------------------------------------------------------------------------
def test_highlight_chunk_completed_clips_by_chunk_field_defaults_to_empty_map():
    """Additive-only: a caller that doesn't pass `clips_by_chunk` at all (an
    OLDER call site, or a legitimate zero-highlight chunk) gets {}, never a
    missing key or a crash."""
    cp = build_highlight_chunk_completed(
        chunk_index=0, chunks_total=1, highlights_scanned=0,
        highlights_analyzed=0, highlights_ditched=0, highlights_published=0,
        gemini_file_uri="files/abc", worker_state=_ws(progress_percent=100.0),
    )
    assert cp["artifacts"]["clips_by_chunk"] == {}
    assert cp["artifacts"]["highlights_collected_by_chunk"] == {}


def test_highlight_chunk_completed_clips_by_chunk_field_carries_the_full_cumulative_map():
    """2026-07-26 Evaluator REJECT fix: this field is the FULL, merged
    cumulative map across every chunk completed so far (real
    job_stage_checkpoints PRIMARY KEY (job_id, stage_name) — an INSERT is
    an UPSERT, so the caller must pass the complete merged state on every
    write, never just this one chunk's own slice)."""
    clips_by_chunk = {
        "0": [{"start_s": 10.0, "end_s": 20.0, "action_class": "guard_pass", "_candidate_key": "0:1"}],
        "1": [{"start_s": 800.0, "end_s": 810.0, "action_class": "sweep", "_candidate_key": "1:1"}],
    }
    highlights_collected_by_chunk = {"0": 1, "1": 1}
    cp = build_highlight_chunk_completed(
        chunk_index=1, chunks_total=2, highlights_scanned=1,
        highlights_analyzed=1, highlights_ditched=0, highlights_published=0,
        gemini_file_uri="files/abc", worker_state=_ws(progress_percent=100.0),
        clips_by_chunk=clips_by_chunk, highlights_collected_by_chunk=highlights_collected_by_chunk,
    )
    assert cp["artifacts"]["clips_by_chunk"] == clips_by_chunk
    assert cp["artifacts"]["highlights_collected_by_chunk"] == highlights_collected_by_chunk


def test_highlight_publish_progress_envelope_shape():
    cp = build_highlight_publish_progress(
        candidate_key="2:3", event_index=17, published_candidate_keys=["0:1", "1:1", "2:3"],
        worker_state=_ws(progress_percent=34.0, stage_progress_fraction=0.34),
    )
    _assert_envelope(cp)
    assert cp["reason"] == "highlight_publish_candidate"
    assert cp["artifacts"]["candidate_key"] == "2:3"
    assert cp["artifacts"]["event_index"] == 17
    # 2026-07-26 Evaluator REJECT fix: the FULL cumulative list, not just
    # this one candidate — real job_stage_checkpoints UPSERT semantics mean
    # a resume can only ever see the LATEST row, so it must carry every
    # candidate published so far, not just the most recent one.
    assert cp["artifacts"]["published_candidate_keys"] == ["0:1", "1:1", "2:3"]
    # Never the terminal-checkpoint shape — caller writes this with
    # completed=False, and this builder never claims otherwise itself
    # (completed is the write_checkpoint caller's own argument, not part
    # of the envelope this builder returns).
    assert cp["reason"] != "highlight_publish_completed"


# ---------------------------------------------------------------------------
# Highlight stage order
# ---------------------------------------------------------------------------
def test_highlight_stage_order_is_the_three_highlight_stages():
    assert len(set(HIGHLIGHT_STAGE_ORDER)) == 3
    assert HIGHLIGHT_STAGE_ORDER == [
        PipelineStage.HIGHLIGHT_INGEST,
        PipelineStage.HIGHLIGHT_CHUNK,
        PipelineStage.HIGHLIGHT_PUBLISH,
    ]
