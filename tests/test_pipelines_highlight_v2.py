"""v2 build plan (``2026-07-18-highlight-scan-critique-analyze-v2-plan.md``):
``highlight_axes.py`` pure helpers (schema + Gracie's real prompts),
``highlight_critique.py`` pure helpers (schema + Gracie's real prompt, the
ABSOLUTE-seconds contract), ``executors.highlight_critique_node``, and
``executors.highlight_analyze_node``'s v2 restructure (authoritative
corrected bounds, ONE flat position/technique verdict per highlight, the
bounded validator loop, and the "ditched" contract). All Gemini calls are
mocked — no live network access anywhere in this file.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types

from service.pipelines import executors, highlight_axes, highlight_critique, simplified_tags
from service.pipelines.executors import RunContext, estimate_run_plan, run_pipeline
from service.pipelines.models import HighlightAnalyzeConfig, HighlightCritiqueConfig


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", AsyncMock())


# =============================================================================== #
# highlight_axes.py — pure helpers (schema/prompt), Gracie's real text.
# =============================================================================== #
def test_position_schema_shape_flat_object_not_a_list():
    schema = highlight_axes.build_position_schema()
    assert schema.type == types.Type.OBJECT
    assert set(schema.required) == {"position", "justification"}
    assert "clips" not in schema.properties  # NOT a list — one flat verdict
    assert "actor" not in schema.properties
    assert "action_class" not in schema.properties
    assert "start_s" not in schema.properties  # no timestamps reported
    assert list(schema.properties["position"].enum) == simplified_tags.POSITION_VALUES


def test_technique_schema_shape_flat_object_no_position_field():
    schema = highlight_axes.build_technique_schema()
    assert schema.type == types.Type.OBJECT
    assert set(schema.required) == {"action_class", "outcome", "justification"}
    assert "clips" not in schema.properties
    assert "position" not in schema.properties
    assert "actor" not in schema.properties  # dropped in Gracie's real schema
    assert "start_s" not in schema.properties
    assert list(schema.properties["action_class"].enum) == simplified_tags.ACTION_CLASS_VALUES
    assert list(schema.properties["outcome"].enum) == simplified_tags.OUTCOME_VALUES


def test_validator_schema_is_highlight_level_flat_object():
    schema = highlight_axes.build_validator_schema()
    assert schema.type == types.Type.OBJECT
    assert set(schema.required) == {"adversarial_case", "verdict", "evidence"}
    assert list(schema.properties["verdict"].enum) == ["agree", "disagree", "ditch"]
    assert "clips" not in schema.properties  # no per-candidate-index verdict list


def test_validator_schema_final_label_fields_are_nullable():
    schema = highlight_axes.build_validator_schema()
    for field in ("final_position", "final_action_class", "final_outcome", "wrong_label", "reason"):
        assert schema.properties[field].nullable is True
        assert field not in schema.required


def test_position_prompt_embeds_description_via_template():
    prompt = highlight_axes.build_position_prompt("a grip fight into a takedown")
    assert "a grip fight into a takedown" in prompt


def test_technique_prompt_embeds_description_via_template():
    prompt = highlight_axes.build_technique_prompt("a scramble to back control")
    assert "a scramble to back control" in prompt


def test_validator_prompt_embeds_all_six_tokens():
    prompt = highlight_axes.build_validator_prompt(
        description="d-marker", position_label="mount-marker",
        position_justification="pj-marker", technique_label="guard_pass-marker",
        outcome_label="successful-marker", technique_justification="tj-marker",
    )
    for marker in ("d-marker", "mount-marker", "pj-marker", "guard_pass-marker", "successful-marker", "tj-marker"):
        assert marker in prompt


# --------------------------------------------------------------------------- #
# Leftover-$-token regression guard (coordinator-mandated, 2026-07-18):
# Template.safe_substitute silently leaves an unrecognized/misnamed token as
# a literal "$token" string — never an error. These tests render every v2
# prompt with realistic substitution values and assert NO "$" character
# survives, catching a name-mismatch between the prompt text and the
# executor's substitution dict as a hard test failure, not a silent bug.
# --------------------------------------------------------------------------- #
def test_position_prompt_leaves_no_leftover_tokens():
    prompt = highlight_axes.build_position_prompt("a body-movement description")
    assert "$" not in prompt


def test_technique_prompt_leaves_no_leftover_tokens():
    prompt = highlight_axes.build_technique_prompt("a body-movement description")
    assert "$" not in prompt


def test_validator_prompt_leaves_no_leftover_tokens():
    prompt = highlight_axes.build_validator_prompt(
        description="d", position_label="mount", position_justification="pj",
        technique_label="guard_pass", outcome_label="successful", technique_justification="tj",
    )
    assert "$" not in prompt


def test_critique_prompt_leaves_no_leftover_tokens():
    prompt = highlight_critique.build_critique_prompt(
        window_start_sec=10.0, window_end_sec=30.0, critique_backpad_s=6.0,
        scan_start_sec=16.0, scan_end_sec=30.0, description="d",
    )
    assert "$" not in prompt


def test_position_prompt_leaves_no_leftover_tokens_even_with_none_description():
    prompt = highlight_axes.build_position_prompt(None)
    assert "$" not in prompt


def test_technique_prompt_leaves_no_leftover_tokens_even_with_none_description():
    prompt = highlight_axes.build_technique_prompt(None)
    assert "$" not in prompt


def test_critique_prompt_leaves_no_leftover_tokens_even_with_none_description():
    prompt = highlight_critique.build_critique_prompt(
        window_start_sec=0.0, window_end_sec=10.0, critique_backpad_s=6.0,
        scan_start_sec=6.0, scan_end_sec=10.0, description=None,
    )
    assert "$" not in prompt


def test_critique_prompt_all_six_tokens_actually_appear_in_rendered_output():
    """Direct proof each of the six named values ends up in the rendered
    text (not just "no $ left over" — that alone wouldn't catch a token
    silently rendered as empty/missing due to a KeyError-free but wrong
    substitution key)."""
    prompt = highlight_critique.build_critique_prompt(
        window_start_sec=11.25, window_end_sec=33.5, critique_backpad_s=7.75,
        scan_start_sec=17.25, scan_end_sec=33.5, description="MARKER-DESCRIPTION",
    )
    assert "11.25" in prompt
    assert "33.5" in prompt
    assert "7.75" in prompt
    assert "17.25" in prompt
    assert "MARKER-DESCRIPTION" in prompt


# =============================================================================== #
# highlight_critique.py — pure helpers.
# =============================================================================== #
def test_critique_schema_required_fields():
    schema = highlight_critique.build_critique_schema()
    assert set(schema.required) == {"movement_confirmed", "corrected_start_s", "corrected_end_s", "note"}


def test_critique_prompt_embeds_description():
    prompt = highlight_critique.build_critique_prompt(
        window_start_sec=0.0, window_end_sec=10.0, critique_backpad_s=6.0,
        scan_start_sec=6.0, scan_end_sec=10.0, description="a grip fight into a single leg",
    )
    assert "a grip fight into a single leg" in prompt


def test_critique_prompt_handles_missing_description_never_crashes():
    prompt = highlight_critique.build_critique_prompt(
        window_start_sec=0.0, window_end_sec=10.0, critique_backpad_s=6.0,
        scan_start_sec=6.0, scan_end_sec=10.0, description=None,
    )
    assert isinstance(prompt, str) and len(prompt) > 0


# =============================================================================== #
# executors.highlight_critique_node
# =============================================================================== #
@pytest.mark.asyncio
async def test_highlight_critique_node_sends_backward_padded_window_only(monkeypatch):
    """The critique window is [max(scope_start, start_s - backpad), end_s] —
    NEVER extends past the scan's own end_s (backward pad ONLY, per Gracie's
    VTG-drift finding: search earlier, not later)."""
    captured = []

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured.append((start_sec, end_sec))
        return json.dumps({"movement_confirmed": True, "corrected_start_s": 15.0, "corrected_end_s": 28.0, "note": "n"})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 20.0, "end_s": 30.0, "adjustment": None}]
    cfg = HighlightCritiqueConfig(critique_backpad_s=6.0).model_dump()

    events = [e async for e in executors.highlight_critique_node(ctx, cfg)]

    assert captured == [(14.0, 30.0)]  # 20-6=14, end stays at scan's own 30 (never extended)
    result_event = next(e for e in events if e["type"] == "highlight_critique_result")
    assert result_event["highlight_index"] == 1


@pytest.mark.asyncio
async def test_highlight_critique_node_clamps_backpad_to_scope_start(monkeypatch):
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return json.dumps({"movement_confirmed": True, "corrected_start_s": 10.0, "corrected_end_s": 18.0, "note": "n"})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=10, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 12.0, "end_s": 20.0, "adjustment": None}]
    cfg = HighlightCritiqueConfig(critique_backpad_s=6.0).model_dump()

    events = [e async for e in executors.highlight_critique_node(ctx, cfg)]
    start_evt = next(e for e in events if e["type"] == "highlight_critique_start")
    assert start_evt["scope"] == [10.0, 20.0]  # 12-6=6 clamped up to ctx.start_sec=10


@pytest.mark.asyncio
async def test_highlight_critique_node_additive_never_overwrites_start_s_end_s(monkeypatch):
    """ABSOLUTE-seconds contract (module docstring): Gemini reports
    corrected_start_s/corrected_end_s directly on the match's absolute clock
    — NEVER chunk-relative — so the response value flows through with only
    sanity-clamping, no offset-addition conversion."""
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return json.dumps({
            "movement_confirmed": True, "corrected_start_s": 16.0, "corrected_end_s": 29.0,
            "note": "earlier setup found",
        })

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    highlight = {"index": 1, "start_s": 20.0, "end_s": 30.0, "adjustment": None}
    ctx.highlights = [highlight]
    cfg = HighlightCritiqueConfig(critique_backpad_s=6.0).model_dump()

    [e async for e in executors.highlight_critique_node(ctx, cfg)]

    # Original scan bounds are UNTOUCHED.
    assert highlight["start_s"] == 20.0
    assert highlight["end_s"] == 30.0
    # Corrected bounds are ADDITIVE and used VERBATIM (absolute, not offset-added).
    assert highlight["corrected_start_s"] == 16.0
    assert highlight["corrected_end_s"] == 29.0
    assert highlight["critique_note"] == "earlier setup found"


@pytest.mark.asyncio
async def test_highlight_critique_node_inverted_window_skipped_never_calls_gemini():
    """Reuse of the inverted-window guard (INS-107 / the highlight_analyze
    fix): a highlight whose own bounds sit at/before ctx.start_sec such that
    the backward-padded window inverts must be skipped, zero Gemini calls."""
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        return json.dumps({"movement_confirmed": True, "corrected_start_s": 0.0, "corrected_end_s": 1.0, "note": "n"})

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=20, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    # start_s == end_s == ctx.start_sec -> window_start = max(20, 20-6)=20,
    # window_end = 20 -> inverted/zero-length, must be skipped.
    ctx.highlights = [{"index": 1, "start_s": 20.0, "end_s": 20.0, "adjustment": None}]
    cfg = HighlightCritiqueConfig(critique_backpad_s=6.0).model_dump()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_critique_node(ctx, cfg)]

    assert call_count["n"] == 0
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["highlight_index"] == 1
    assert "corrected_start_s" not in ctx.highlights[0]


@pytest.mark.asyncio
async def test_highlight_critique_node_no_highlights_errors_never_crashes():
    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=50,
                      gemini_api_key="k", request_timeout_ms=1000)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_critique_node(ctx, HighlightCritiqueConfig().model_dump())]
    assert events[0]["type"] == "error"
    assert events[0]["highlight_index"] is None


@pytest.mark.asyncio
async def test_highlight_critique_node_garbage_correction_dropped_never_fabricated(monkeypatch):
    """A correction whose end <= start (after clamping) is DROPPED — the
    highlight falls back to its original bounds, never a fabricated/garbage
    corrected span."""
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        # corrected_end_s (9.0) < corrected_start_s (20.0), ABSOLUTE seconds —
        # nonsensical; must be rejected, not silently coerced.
        return json.dumps({"movement_confirmed": False, "corrected_start_s": 20.0, "corrected_end_s": 9.0, "note": "n"})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    highlight = {"index": 1, "start_s": 20.0, "end_s": 30.0, "adjustment": None}
    ctx.highlights = [highlight]
    cfg = HighlightCritiqueConfig(critique_backpad_s=6.0).model_dump()

    events = [e async for e in executors.highlight_critique_node(ctx, cfg)]
    assert "corrected_start_s" not in highlight
    assert "corrected_end_s" not in highlight
    result_event = next(e for e in events if e["type"] == "highlight_critique_result")
    assert result_event["applied"] is False


@pytest.mark.asyncio
async def test_highlight_critique_node_missing_correction_fields_dropped_never_fabricated(monkeypatch):
    """A response missing corrected_start_s/corrected_end_s entirely (e.g. a
    malformed/mock response) must never fabricate a correction."""
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return json.dumps({"movement_confirmed": False, "note": "clip did not match description at all"})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    highlight = {"index": 1, "start_s": 20.0, "end_s": 30.0, "adjustment": None}
    ctx.highlights = [highlight]

    events = [e async for e in executors.highlight_critique_node(ctx, HighlightCritiqueConfig().model_dump())]
    assert "corrected_start_s" not in highlight
    result_event = next(e for e in events if e["type"] == "highlight_critique_result")
    assert result_event["applied"] is False
    assert result_event["movement_confirmed"] is False


@pytest.mark.asyncio
async def test_highlight_critique_node_movement_not_confirmed_ignores_well_formed_correction(monkeypatch):
    """Evaluator LOW 1 (2026-07-18): the schema does NOT make
    corrected_start_s/corrected_end_s conditional on movement_confirmed — a
    real Gemini response can (and did, live) return movement_confirmed=false
    ALONGSIDE well-formed corrected_start_s/corrected_end_s. That correction
    must be IGNORED — the scan's ORIGINAL start_s/end_s stay authoritative —
    and the mismatch must be surfaced via critique_note."""
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        # Well-formed, in-bounds correction — the OLD code would have
        # applied this despite movement_confirmed=false.
        return json.dumps({
            "movement_confirmed": False, "corrected_start_s": 16.0, "corrected_end_s": 29.0,
            "note": "description does not match this clip — no arm-lock setup visible anywhere",
        })

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    highlight = {"index": 1, "start_s": 20.0, "end_s": 30.0, "adjustment": None}
    ctx.highlights = [highlight]
    cfg = HighlightCritiqueConfig(critique_backpad_s=6.0).model_dump()

    events = [e async for e in executors.highlight_critique_node(ctx, cfg)]

    # Correction NEVER applied despite being well-formed/in-bounds.
    assert "corrected_start_s" not in highlight
    assert "corrected_end_s" not in highlight
    # Original scan bounds remain the ONLY bounds on the highlight record.
    assert highlight["start_s"] == 20.0
    assert highlight["end_s"] == 30.0
    # Mismatch surfaced via critique_note/telemetry, not silently dropped.
    assert "arm-lock setup visible" in highlight["critique_note"]

    result_event = next(e for e in events if e["type"] == "highlight_critique_result")
    assert result_event["applied"] is False
    assert result_event["movement_confirmed"] is False
    assert result_event["corrected_start_s"] is None
    assert result_event["corrected_end_s"] is None


@pytest.mark.asyncio
async def test_highlight_critique_node_movement_not_confirmed_without_note_gets_default_mismatch_note(monkeypatch):
    """Same movement_confirmed=false gate, but the response omits `note`
    entirely — the mismatch must still be surfaced (a sensible default
    note), never silently swallowed."""
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return json.dumps({"movement_confirmed": False, "corrected_start_s": 5.0, "corrected_end_s": 9.0})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    highlight = {"index": 1, "start_s": 20.0, "end_s": 30.0, "adjustment": None}
    ctx.highlights = [highlight]

    [e async for e in executors.highlight_critique_node(ctx, HighlightCritiqueConfig().model_dump())]

    assert "corrected_start_s" not in highlight
    assert highlight["critique_note"]  # non-empty default, never None/missing
    assert "not applied" in highlight["critique_note"] or "movement_confirmed" in highlight["critique_note"]


# =============================================================================== #
# executors.highlight_analyze_node — v2 restructure: authoritative corrected
# bounds, ONE flat position/technique verdict, the bounded validator loop,
# ditch contract, bounded-loop termination.
# =============================================================================== #
def _agree_response():
    return json.dumps({
        "adversarial_case": "considered the opposite, evidence did not support it", "verdict": "agree",
        "final_position": None, "final_action_class": None, "final_outcome": None,
        "wrong_label": None, "reason": None, "evidence": "e",
    })


def _position_response(position="mount"):
    return json.dumps({"position": position, "justification": "j"})


def _technique_response(action_class="submission_choke", outcome="successful"):
    return json.dumps({"action_class": action_class, "outcome": outcome, "justification": "j"})


@pytest.mark.asyncio
async def test_highlight_analyze_node_reads_corrected_bounds_as_authoritative(monkeypatch):
    """corrected_start_s/corrected_end_s (set by highlight_critique) override
    the scan's own start_s/end_s for window expansion — and the synthesized
    clip's own start_s/end_s land on the AUTHORITATIVE bounds, not the
    scan's original ones."""
    captured_windows = []
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_windows.append((start_sec, end_sec))
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _position_response()
        if call_count["n"] == 2:
            return _technique_response()
        return _agree_response()

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    # Scan's own bounds [20, 35]; critique corrected to [15, 32] — the
    # AUTHORITATIVE bounds this test proves get used instead.
    ctx.highlights = [{
        "index": 1, "start_s": 20.0, "end_s": 35.0, "adjustment": None,
        "corrected_start_s": 15.0, "corrected_end_s": 32.0,
    }]
    cfg = HighlightAnalyzeConfig(preroll_s=5.0, postroll_s=4.0).model_dump()
    events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    # Window built from CORRECTED bounds: 15-5=10, 32+4=36 (not 20-5=15/35+4=39).
    assert all(w == (10.0, 36.0) for w in captured_windows)

    start_event = next(e for e in events if e["type"] == "highlight_start")
    assert start_event["highlight_bounds"] == [20.0, 35.0]  # provenance: scan's own, unchanged meaning
    assert start_event["authoritative_bounds"] == [15.0, 32.0]  # NEW: what actually drove the math

    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert result_event["status"] == "analyzed"
    assert len(result_event["clips"]) == 1
    assert result_event["clips"][0]["start_s"] == 15.0  # authoritative, not scan's own 20.0
    assert result_event["clips"][0]["end_s"] == 32.0


@pytest.mark.asyncio
async def test_highlight_analyze_node_falls_back_to_scan_bounds_when_uncorrected(monkeypatch):
    """A highlight with no corrected_start_s/corrected_end_s (e.g. the older
    highlight-scan-analyze pipeline, or a critique call that failed) behaves
    byte-identically to before — authoritative == scan's own bounds."""
    captured_windows = []
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_windows.append((start_sec, end_sec))
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _position_response()
        if call_count["n"] == 2:
            return _technique_response()
        return _agree_response()

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 20.0, "end_s": 35.0, "adjustment": None}]
    cfg = HighlightAnalyzeConfig(preroll_s=5.0, postroll_s=4.0).model_dump()
    events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    assert all(w == (15.0, 39.0) for w in captured_windows)  # 20-5, 35+4 — unchanged from pre-v2 math

    start_event = next(e for e in events if e["type"] == "highlight_start")
    assert start_event["authoritative_bounds"] == [20.0, 35.0] == start_event["highlight_bounds"]

    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert result_event["status"] == "analyzed"
    assert result_event["clips"][0]["start_s"] == 20.0
    assert result_event["clips"][0]["end_s"] == 35.0


@pytest.mark.asyncio
async def test_highlight_analyze_node_validator_ditch_yields_zero_clips_and_status():
    """Strict-ditch path (mocked validator): status='ditched' + ditch_reason,
    clips=[] by design — the highlight-level 'sibling to clips' contract."""
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _position_response("closed_guard")
        if call_count["n"] == 2:
            return _technique_response("guard_pass", "unsuccessful")
        return json.dumps({
            "adversarial_case": "no intentional movement by either athlete", "verdict": "ditch",
            "final_position": None, "final_action_class": None, "final_outcome": None,
            "wrong_label": None, "reason": "no intentional grip/weight change", "evidence": "e",
        })

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert result_event["status"] == "ditched"
    assert result_event["ditch_reason"] == "no intentional grip/weight change"
    assert result_event["clips"] == []
    assert result_event["verdict"] == "ditch"
    # position, technique (once each) + validator (1 round, default) = 3.
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_highlight_analyze_node_disagree_only_changes_wrong_label_field():
    """Evaluator MEDIUM (2026-07-18, blocking): Gracie's prompt names EXACTLY
    ONE wrong_label on 'disagree'. A validator whose adversarial case only
    doubted action_class but ALSO returned a changed final_position (a live-
    demonstrated risk) must NOT have that final_position take effect —
    final_position must stay the ORIGINAL analyzer label; ONLY
    final_action_class (the field wrong_label actually names) may change."""
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _position_response("mount")
        if call_count["n"] == 2:
            return _technique_response("submission_arm_lock", "successful")
        # wrong_label="action_class", but ALSO smuggles a changed
        # final_position ("side_control") — the fix must ignore that stray
        # final_position entirely and keep the ORIGINAL "mount".
        return json.dumps({
            "adversarial_case": "the arm looked trapped for a choke, not a lock",
            "verdict": "disagree",
            "final_position": "side_control",  # MUST be ignored — wrong_label doesn't name position
            "final_action_class": "submission_choke",  # the ONE field wrong_label names — MUST apply
            "final_outcome": "unclear",  # MUST be ignored — wrong_label doesn't name outcome
            "wrong_label": "action_class",
            "reason": "collar grip visible, not an arm isolation", "evidence": "e",
        })

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert result_event["status"] == "analyzed"
    assert result_event["verdict"] == "disagree"
    clip = result_event["clips"][0]
    assert clip["position"] == "mount"  # UNCHANGED — original, despite the validator's stray final_position
    assert clip["action_class"] == "submission_choke"  # CORRECTED — the one field wrong_label named
    assert clip["outcome"] == "successful"  # UNCHANGED — original, despite the validator's stray final_outcome


@pytest.mark.asyncio
async def test_highlight_analyze_node_disagree_with_no_wrong_label_keeps_all_originals():
    """A 'disagree' verdict with a null/missing wrong_label names no axis to
    correct — treated as NO CHANGE (every original label kept) rather than
    guessing which field to trust."""
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _position_response("half_guard")
        if call_count["n"] == 2:
            return _technique_response("sweep", "unsuccessful")
        return json.dumps({
            "adversarial_case": "considered other reads but none stuck",
            "verdict": "disagree", "final_position": "mount", "final_action_class": "guard_pass",
            "final_outcome": "successful", "wrong_label": None,
            "reason": "unclear which axis, if any, is actually wrong", "evidence": "e",
        })

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    result_event = next(e for e in events if e["type"] == "highlight_result")
    clip = result_event["clips"][0]
    assert clip["position"] == "half_guard"
    assert clip["action_class"] == "sweep"
    assert clip["outcome"] == "unsuccessful"


@pytest.mark.asyncio
async def test_highlight_analyze_node_agree_ignores_any_stray_final_fields():
    """On 'agree', the ORIGINAL analyzer labels ship regardless of whatever
    (possibly stray/inconsistent) final_* values the validator echoes."""
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _position_response("standing")
        if call_count["n"] == 2:
            return _technique_response("takedown_attempt", "unsuccessful")
        return json.dumps({
            "adversarial_case": "case considered, both labels survive",
            "verdict": "agree",
            # Stray/inconsistent final_* values a validator should never
            # send on "agree" (per Gracie's prompt, they should echo the
            # originals) — but even if it did send something else, "agree"
            # must not let it through.
            "final_position": "mount", "final_action_class": "guard_pass", "final_outcome": "successful",
            "wrong_label": None, "reason": None, "evidence": "e",
        })

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    result_event = next(e for e in events if e["type"] == "highlight_result")
    clip = result_event["clips"][0]
    assert clip["position"] == "standing"
    assert clip["action_class"] == "takedown_attempt"
    assert clip["outcome"] == "unsuccessful"


@pytest.mark.asyncio
async def test_highlight_analyze_node_position_and_technique_called_exactly_once_never_per_iteration():
    """LeCun's own cost model (build plan design-pass, '1 + N(3+L)'):
    position/technique are a FLAT +2 per highlight — only the validator
    loops. Even with max_validator_iterations=3 and a validator that keeps
    disagreeing, position/technique must be called EXACTLY ONCE each, not
    once per validator round."""
    call_count = {"n": 0}
    position_calls = {"n": 0}
    technique_calls = {"n": 0}
    validator_calls = {"n": 0}
    actor_calls = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        # Deterministic call ORDER (position, then technique, then N
        # validator rounds, then ONE actor call once the validator
        # resolves) — never sniffing prompt text content, which would make
        # this test fragile against a future prompt wording edit.
        call_count["n"] += 1
        if call_count["n"] == 1:
            position_calls["n"] += 1
            return _position_response()
        if call_count["n"] == 2:
            technique_calls["n"] += 1
            return _technique_response()
        if call_count["n"] == 3:
            validator_calls["n"] += 1
            return json.dumps({"adversarial_case": "ac", "verdict": "disagree",
                                "final_position": "mount", "final_action_class": "submission_choke",
                                "final_outcome": "successful", "wrong_label": "action_class",
                                "reason": "r", "evidence": "e"})
        actor_calls["n"] += 1
        return json.dumps({"actor": "unclear", "identity_uncertain": True, "justification": "j"})

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]
    cfg = HighlightAnalyzeConfig(max_validator_iterations=3).model_dump()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    assert position_calls["n"] == 1
    assert technique_calls["n"] == 1
    assert validator_calls["n"] == 1  # "disagree" resolves on the FIRST round — no need to loop further
    assert actor_calls["n"] == 1  # flat, once — never re-invoked by the validator loop
    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert result_event["status"] == "analyzed"
    assert result_event["verdict"] == "disagree"


@pytest.mark.asyncio
async def test_highlight_analyze_node_bounded_loop_terminates_on_permanent_disagreement():
    """PROOF the validator loop cannot hang: a validator that NEVER reaches
    agree/disagree/ditch must still terminate after EXACTLY
    max_validator_iterations rounds (never more, never fewer) and ditch via
    the for-loop's own 'exhaustion' fallthrough — a `for...else`, never a
    `while True`. Position/technique are called ONCE regardless (LeCun cost
    model) — only the validator call repeats."""
    call_count = {"n": 0}
    max_iterations = 3

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _position_response()
        if call_count["n"] == 2:
            return _technique_response()
        # Every subsequent call is a validator round; it PERMANENTLY
        # returns an unrecognized verdict (neither agree, disagree, nor
        # ditch) — the adversarial case the bounded loop must survive
        # without hanging.
        return json.dumps({"adversarial_case": "ac", "verdict": "still_thinking", "evidence": "e"})

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]
    cfg = HighlightAnalyzeConfig(max_validator_iterations=max_iterations).model_dump()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    # 1 position + 1 technique + max_iterations validator rounds — never
    # more (proves termination), never fewer (proves every available round
    # was used before giving up).
    assert call_count["n"] == 2 + max_iterations

    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert result_event["status"] == "ditched"
    assert "unresolved" in result_event["ditch_reason"]
    assert str(max_iterations) in result_event["ditch_reason"]
    assert result_event["clips"] == []
    assert result_event["validator_rounds"] == max_iterations


@pytest.mark.asyncio
async def test_highlight_analyze_node_v1_default_single_pass_unresolved_ditches_immediately():
    """v1 founder decision: max_validator_iterations defaults to 1 — a SINGLE
    unresolved verdict must ditch immediately, not loop."""
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _position_response()
        if call_count["n"] == 2:
            return _technique_response()
        return json.dumps({"adversarial_case": "ac", "verdict": "still_thinking", "evidence": "e"})

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]
    cfg = HighlightAnalyzeConfig().model_dump()  # max_validator_iterations default = 1
    assert HighlightAnalyzeConfig.model_validate(cfg).max_validator_iterations == 1

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    assert call_count["n"] == 3  # position + technique + a single validator round
    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert result_event["status"] == "ditched"


@pytest.mark.asyncio
async def test_highlight_analyze_node_position_call_transport_error_skips_highlight_never_calls_technique():
    """A transport-level failure on the FIRST (position) call must abort just
    THIS highlight with an error event — never call technique/validator for
    it, never crash the whole run."""
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        raise RuntimeError("simulated transport failure")

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    assert call_count["n"] == 1  # NEVER attempted technique after position failed
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["highlight_index"] == 1
    assert "position call" in error_events[0]["message"]
    assert not [e for e in events if e["type"] == "highlight_result"]


@pytest.mark.asyncio
async def test_highlight_analyze_node_technique_call_transport_error_skips_highlight():
    call_count = {"n": 0}

    async def _fake(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _position_response()
        raise RuntimeError("simulated technique transport failure")

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    assert call_count["n"] == 2  # position succeeded, technique failed — validator NEVER attempted
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "technique call" in error_events[0]["message"]


@pytest.mark.asyncio
async def test_highlight_analyze_node_validator_call_transport_error_skips_highlight_never_hangs():
    """Evaluator LOW 2 (2026-07-18): position and technique both succeed,
    but the validator call itself raises a transport-level error — the
    highlight must be skipped with an error event (never a fabricated
    result), the loop must continue to any remaining highlight, and this
    must never hang regardless of max_validator_iterations (the transport
    exception is raised on the FIRST validator attempt here, well before any
    iteration budget would matter)."""
    call_count = {"n": 0}

    async def _fake(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        # Per-highlight call order is position, technique, validator — use
        # modulo so this resets correctly across BOTH highlights (a plain
        # "== 1"/"== 2" check would only match highlight 1's first two
        # calls and never highlight 2's).
        slot = call_count["n"] % 3
        if slot == 1:
            return _position_response()
        if slot == 2:
            return _technique_response()
        # Every validator attempt (slot 0) raises. It never retries within
        # a highlight since the transport exception aborts it outright.
        raise RuntimeError("simulated validator transport failure")

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [
        {"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None},
        {"index": 2, "start_s": 50.0, "end_s": 55.0, "adjustment": None},
    ]
    # A generous iteration budget — proves the transport-error break exits
    # the validator loop immediately rather than retrying up to the cap.
    cfg = HighlightAnalyzeConfig(max_validator_iterations=5).model_dump()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    # 3 calls for highlight 1 (position, technique, validator-raises) + 3
    # for highlight 2 = 6 — NEVER 5 rounds' worth of validator retries (10+),
    # proving the transport-error path breaks out on the FIRST failure.
    assert call_count["n"] == 6
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 2
    assert all("validator call" in e["message"] for e in error_events)
    assert error_events[0]["highlight_index"] == 1
    assert error_events[1]["highlight_index"] == 2
    # No fabricated result for either highlight — the loop continued
    # cleanly to highlight 2 and also aborted it, never crashing/hanging.
    assert not [e for e in events if e["type"] == "highlight_result"]
    assert ctx.final_clips == []


# =============================================================================== #
# Budget-gate factor — the REAL two-phase gate inside run_pipeline (not just
# estimate_run_plan's pre-flight upper bound — see test_pipelines_executors.py
# for that half). Factor = 3 (position+technique+actor, fixed) +
# max_validator_iterations (S12 Phase 1b: was 2 fixed, now 3 with the actor
# axis — see _highlight_per_highlight_call_factor).
# =============================================================================== #
@pytest.mark.asyncio
async def test_run_pipeline_real_two_phase_gate_uses_per_highlight_factor(monkeypatch):
    """2 highlights x factor(3 fixed + 3 validator iterations = 6)
    = 12 real calls > budget_cap=8 -> must abort BEFORE the deep loop, zero
    real Gemini calls spent (build plan item 8: never under-count)."""
    from service.pipelines import registry

    pdef = registry.get_default("highlight-scan-analyze")
    next(s for s in pdef.stages if s.type == "highlight_analyze").config["max_validator_iterations"] = 3

    async def _fake_scan_generate(*, model, contents, config):
        from types import SimpleNamespace
        return SimpleNamespace(text=json.dumps({"highlights": [
            {"start_s": 0, "end_s": 10}, {"start_s": 20, "end_s": 30},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_scan_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    analyze_calls = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        analyze_calls["n"] += 1
        return json.dumps({"clips": []})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
                      start_sec=0, end_sec=90, gemini_api_key="fake-key", request_timeout_ms=60_000)
    planned = estimate_run_plan(pdef, duration_sec=90)
    events = [e async for e in run_pipeline(pdef, ctx, planned, budget_cap=8)]

    assert analyze_calls["n"] == 0  # aborted before any real PASS-2+ call
    assert events[-1]["type"] == "error"
    assert "2" in events[-1]["message"] and "6" in events[-1]["message"] and "12" in events[-1]["message"]


@pytest.mark.asyncio
async def test_run_pipeline_real_two_phase_gate_within_cap_runs_normally(monkeypatch):
    """Same shape, but budget_cap large enough for the PRE-FLIGHT gate's
    pessimistic factor (2 highlights x 6 = 12) — the deep loop then runs
    with an IMMEDIATELY-agreeing validator (round 1), so each highlight's
    REAL spend is position+technique+validator+actor = 4 calls (actor fires
    because the highlight is not ditched), 8 total — well under cap=12."""
    from service.pipelines import registry

    pdef = registry.get_default("highlight-scan-analyze")
    next(s for s in pdef.stages if s.type == "highlight_analyze").config["max_validator_iterations"] = 3

    async def _fake_scan_generate(*, model, contents, config):
        from types import SimpleNamespace
        return SimpleNamespace(text=json.dumps({"highlights": [
            {"start_s": 0, "end_s": 10}, {"start_s": 20, "end_s": 30},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_scan_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    analyze_calls = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        analyze_calls["n"] += 1
        n = analyze_calls["n"]
        # Per highlight: call 1 = position, call 2 = technique, call 3 =
        # validator (agrees immediately), call 4 = actor (any valid JSON —
        # the actor mapping is permissive about a missing "actor" key).
        slot = n % 4
        if slot == 1:
            return _position_response()
        if slot == 2:
            return _technique_response()
        if slot == 3:
            return _agree_response()
        return json.dumps({"actor": "unclear", "identity_uncertain": True, "justification": "j"})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
                      start_sec=0, end_sec=90, gemini_api_key="fake-key", request_timeout_ms=60_000)
    planned = estimate_run_plan(pdef, duration_sec=90)
    events = [e async for e in run_pipeline(pdef, ctx, planned, budget_cap=12)]

    types_seen = [e["type"] for e in events]
    assert types_seen[-1] == "run_complete"
    assert "error" not in types_seen
    assert analyze_calls["n"] == 8  # 2 highlights x 4 (position+technique+validator+actor)
