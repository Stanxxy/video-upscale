"""S12 Phase 1b — service/checkpoints/highlight_resume.py (item 8).

Dramatically simpler than the tracking pipeline's build_resume_plan — no
SAM2-style cursor math, just "which chunk comes next" + the Gemini Files
API identity carried on the latest HIGHLIGHT_INGEST checkpoint.
"""
from __future__ import annotations

from datetime import datetime, timezone

from service.checkpoints.highlight_resume import build_highlight_resume_plan


def _chunk_cp(chunk_index: int, *, completed: bool = True) -> dict:
    return {
        "stage_name": "highlight_chunk",
        "completed": completed,
        "checkpoint_data": {"chunk_index": chunk_index, "chunks_total": 5},
    }


def _ingest_cp(
    *, gemini_file_uri="files/abc", gemini_file_name="files/abc",
    gemini_file_mime_type="video/mp4", gemini_file_expiration=None,
    player_references_ready=True,
) -> dict:
    return {
        "stage_name": "highlight_ingest",
        "completed": True,
        "checkpoint_data": {
            "artifacts": {
                "gemini_file_uri": gemini_file_uri,
                "gemini_file_name": gemini_file_name,
                "gemini_file_mime_type": gemini_file_mime_type,
                "gemini_file_expiration": gemini_file_expiration,
            },
            "player_references_ready": player_references_ready,
        },
    }


def test_no_checkpoints_resumes_from_chunk_zero():
    plan = build_highlight_resume_plan([])
    assert plan["resume_from_chunk_index"] == 0
    assert plan["gemini_file_uri"] is None
    assert plan["player_references_ready"] is False


def test_n_completed_chunks_resumes_from_chunk_n():
    plan = build_highlight_resume_plan([_chunk_cp(0), _chunk_cp(1), _chunk_cp(2)])
    assert plan["resume_from_chunk_index"] == 3


def test_out_of_order_chunk_checkpoints_use_max_not_last_written():
    plan = build_highlight_resume_plan([_chunk_cp(2), _chunk_cp(0), _chunk_cp(1)])
    assert plan["resume_from_chunk_index"] == 3


def test_incomplete_chunk_checkpoint_does_not_count():
    plan = build_highlight_resume_plan([_chunk_cp(0), _chunk_cp(1, completed=False)])
    assert plan["resume_from_chunk_index"] == 1  # chunk 1 not yet complete -> resume there


def test_ingest_checkpoint_carries_gemini_file_identity():
    exp = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc).isoformat()
    plan = build_highlight_resume_plan([_ingest_cp(gemini_file_expiration=exp)])
    assert plan["gemini_file_uri"] == "files/abc"
    assert plan["gemini_file_name"] == "files/abc"
    assert plan["gemini_file_mime_type"] == "video/mp4"
    assert plan["gemini_file_expiration"] == datetime.fromisoformat(exp)
    assert plan["player_references_ready"] is True


def test_ingest_and_chunk_checkpoints_together():
    plan = build_highlight_resume_plan([_ingest_cp(), _chunk_cp(0), _chunk_cp(1)])
    assert plan["resume_from_chunk_index"] == 2
    assert plan["gemini_file_uri"] == "files/abc"


def test_no_ingest_checkpoint_leaves_gemini_fields_none():
    plan = build_highlight_resume_plan([_chunk_cp(0)])
    assert plan["gemini_file_uri"] is None
    assert plan["gemini_file_expiration"] is None
    assert plan["player_references_ready"] is False


def test_malformed_expiration_string_does_not_raise():
    plan = build_highlight_resume_plan([_ingest_cp(gemini_file_expiration="not-a-date")])
    assert plan["gemini_file_expiration"] is None
