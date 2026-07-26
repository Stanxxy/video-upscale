"""S12 Phase 1b — ServiceConfig defaults for the v2 production knobs (item 14)."""
from __future__ import annotations

from service.config import ServiceConfig


def test_outer_chunk_scope_sec_default_matches_qa_playground_max_scope_sec():
    config = ServiceConfig()
    # Same default VALUE as chunk_segment.MAX_SCOPE_SEC (a reasonable,
    # already-informally-validated starting point) — but a first-class,
    # independent production knob (design §3.1), not a borrowed constant.
    assert config.outer_chunk_scope_sec == 720


def test_highlight_pipeline_budget_cap_default():
    config = ServiceConfig()
    assert config.highlight_pipeline_budget_cap == 60


def test_gemini_upload_poll_defaults():
    config = ServiceConfig()
    assert config.gemini_upload_poll_interval_sec == 5.0
    assert config.gemini_upload_poll_timeout_sec == 600.0


def test_outer_chunk_scope_sec_env_override():
    config = ServiceConfig(_env_file=None, outer_chunk_scope_sec=300)
    assert config.outer_chunk_scope_sec == 300
