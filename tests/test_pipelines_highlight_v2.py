"""``highlight_axes.py`` pure helpers (schema + prompt), ``highlight_critique.py``
pure helpers (schema + prompt, the ABSOLUTE-seconds contract),
``executors.highlight_critique_node``, and ``executors.highlight_analyze_node``
(2026-07-26 single-call cutover: authoritative corrected bounds, ONE flat
taxonomy verdict per highlight replacing the deleted position/technique/
validator triple, plus the unchanged actor axis). All Gemini calls are
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
# highlight_axes.py — pure helpers (schema/prompt).
# =============================================================================== #
def test_single_call_schema_shape_flat_object_not_a_list():
    schema = highlight_axes.build_single_call_schema()
    assert schema.type == types.Type.OBJECT
    assert set(schema.required) == {"position", "action_class", "outcome", "justification"}
    assert "clips" not in schema.properties  # NOT a list — one flat verdict
    assert "actor" not in schema.properties  # identity is a separate, independent call
    assert "start_s" not in schema.properties  # no timestamps reported
    assert list(schema.properties["position"].enum) == simplified_tags.POSITION_VALUES
    assert list(schema.properties["action_class"].enum) == simplified_tags.ACTION_CLASS_VALUES
    assert list(schema.properties["outcome"].enum) == simplified_tags.OUTCOME_VALUES


def test_single_call_prompt_embeds_description_via_template():
    prompt = highlight_axes.build_single_call_prompt("a grip fight into a takedown")
    assert "a grip fight into a takedown" in prompt


# --------------------------------------------------------------------------- #
# Leftover-$-token regression guard (coordinator-mandated, 2026-07-18):
# Template.safe_substitute silently leaves an unrecognized/misnamed token as
# a literal "$token" string — never an error. These tests render every
# prompt with realistic substitution values and assert NO "$" character
# survives, catching a name-mismatch between the prompt text and the
# executor's substitution dict as a hard test failure, not a silent bug.
# --------------------------------------------------------------------------- #
def test_single_call_prompt_leaves_no_leftover_tokens():
    prompt = highlight_axes.build_single_call_prompt("a body-movement description")
    assert "$" not in prompt


def test_critique_prompt_leaves_no_leftover_tokens():
    prompt = highlight_critique.build_critique_prompt(
        window_start_sec=10.0, window_end_sec=30.0, critique_backpad_s=6.0,
        scan_start_sec=16.0, scan_end_sec=30.0, description="d",
    )
    assert "$" not in prompt


def test_single_call_prompt_leaves_no_leftover_tokens_even_with_none_description():
    prompt = highlight_axes.build_single_call_prompt(None)
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


# --------------------------------------------------------------------------- #
# INS-140 regression (2026-07-26 re-scope AC12/AC13): `mime_type` MUST reach
# `FileData` on every ingest-path `analyze_chunk` call — confirmed missing at
# highlight_critique_node (this call site) and highlight_analyze_node's axis
# call(s) before this fix. `highlight_scan_node` already did this correctly
# (not regression-tested here — that path was never broken).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_highlight_critique_node_forwards_video_mime_type_to_gemini(monkeypatch):
    """INS-140: `ctx.video_mime_type` (set by the production ingest stage for
    a Gemini Files API URI) MUST reach `analyze_chunk`'s `mime_type=` kwarg —
    SimplifiedTagsTimeAnalyzer.analyze_chunk's own docstring says this "MUST"
    be non-None for Files API URIs, or Gemini silently 500s."""
    captured_kwargs = []

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_kwargs.append(kw)
        return json.dumps({"movement_confirmed": True, "corrected_start_s": 15.0, "corrected_end_s": 28.0, "note": "n"})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(
        youtube_id="x", youtube_url="files/abc123", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000, video_mime_type="video/mp4",
    )
    ctx.highlights = [{"index": 1, "start_s": 20.0, "end_s": 30.0, "adjustment": None}]
    cfg = HighlightCritiqueConfig(critique_backpad_s=6.0).model_dump()

    [e async for e in executors.highlight_critique_node(ctx, cfg)]

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["mime_type"] == "video/mp4"


@pytest.mark.asyncio
async def test_highlight_critique_node_qa_playground_mime_type_stays_none(monkeypatch):
    """QA playground (no ingest stage, ctx.video_mime_type never set) —
    byte-identical prior behavior: mime_type stays None, never fabricated."""
    captured_kwargs = []

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_kwargs.append(kw)
        return json.dumps({"movement_confirmed": True, "corrected_start_s": 15.0, "corrected_end_s": 28.0, "note": "n"})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)  # video_mime_type defaults None
    ctx.highlights = [{"index": 1, "start_s": 20.0, "end_s": 30.0, "adjustment": None}]
    cfg = HighlightCritiqueConfig(critique_backpad_s=6.0).model_dump()

    [e async for e in executors.highlight_critique_node(ctx, cfg)]

    assert captured_kwargs[0]["mime_type"] is None


@pytest.mark.asyncio
async def test_highlight_critique_node_offsets_are_file_relative_not_rebased_per_chunk(monkeypatch):
    """AC13 / INS-140 finding (c): a real production job re-uses the SAME
    uploaded whole-match Gemini file across every outer chunk (see
    ``highlight_orchestrator._outer_chunks`` — one job, one Files API upload,
    N chunks of the SAME file) — every ``video_metadata`` offset sent for
    chunk k>0 MUST stay absolute (file-relative), never re-based to that
    chunk's own local 0:00. Simulated here via a RunContext representing the
    SECOND outer chunk of a job (start_sec=720, matching a 720s
    outer_chunk_scope_sec grid) — the sent offsets must land inside
    [720, ...], never [0, ...] (which would silently target the WRONG span of
    the single uploaded file — INS-140's exact "500 INTERNAL" failure mode)."""
    captured = []

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured.append((youtube_url, start_sec, end_sec))
        return json.dumps({"movement_confirmed": True, "corrected_start_s": 740.0, "corrected_end_s": 750.0, "note": "n"})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    # Chunk index 1 of a 720s-grid job — the SAME gemini_file_uri as chunk 0,
    # scoped [720, 1440] on the file's own absolute clock.
    ctx = RunContext(
        youtube_id="x", youtube_url="files/whole-match-upload", start_sec=720, end_sec=1440,
        gemini_api_key="k", request_timeout_ms=1000, video_mime_type="video/mp4",
    )
    ctx.highlights = [{"index": 1, "start_s": 740.0, "end_s": 750.0, "adjustment": None}]
    cfg = HighlightCritiqueConfig(critique_backpad_s=6.0).model_dump()

    [e async for e in executors.highlight_critique_node(ctx, cfg)]

    assert len(captured) == 1
    youtube_url, start_sec, end_sec = captured[0]
    assert youtube_url == "files/whole-match-upload"  # same file, never re-uploaded/re-split per chunk
    assert start_sec >= 720.0  # never rebased to this chunk's own local 0:00
    assert end_sec <= 1440.0


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
# executors.highlight_analyze_node — 2026-07-26 single-call cutover:
# authoritative corrected bounds, ONE flat taxonomy verdict (position +
# action_class + outcome) replacing the deleted position/technique/validator
# triple, plus the unchanged independent actor axis. Every successful
# taxonomy call produces a synthesized highlight.
# =============================================================================== #
def _taxonomy_response(position="mount", action_class="submission_choke", outcome="successful", justification="j"):
    return json.dumps({
        "position": position, "action_class": action_class, "outcome": outcome,
        "justification": justification,
    })


def _actor_response(actor="unclear", identity_uncertain=True, justification="j"):
    return json.dumps({"actor": actor, "identity_uncertain": identity_uncertain, "justification": justification})


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
            return _taxonomy_response()
        return _actor_response()

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
            return _taxonomy_response()
        return _actor_response()

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
async def test_highlight_analyze_node_synthesizes_clip_from_single_taxonomy_verdict():
    """2026-07-26 cutover: no more validator reconciliation — the taxonomy
    call's own position/action_class/outcome ship directly, unmodified."""
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _taxonomy_response(position="side_control", action_class="submission_arm_lock", outcome="successful")
        return _actor_response(actor="p1", identity_uncertain=False)

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]
    ctx.player_references = [{"player_id": "p1", "player_name": "Alice"}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert call_count["n"] == 2  # ONE taxonomy call + ONE actor call — no validator round(s)
    assert result_event["status"] == "analyzed"
    clip = result_event["clips"][0]
    assert clip["position"] == "side_control"
    assert clip["action_class"] == "submission_arm_lock"
    assert clip["outcome"] == "successful"
    assert clip["player_id"] == "p1"
    assert clip["player_name"] == "Alice"
    assert clip["identity_uncertain"] is False
    # Successful taxonomy calls always emit an analyzed result.
    assert "ditch_reason" not in result_event
    assert "verdict" not in result_event
    assert "validator_rounds" not in result_event


@pytest.mark.asyncio
async def test_highlight_analyze_node_notes_concatenates_taxonomy_and_actor_justification():
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _taxonomy_response(justification="arm trapped, hips elevated")
        return _actor_response(justification="blue gi matches reference image 1")

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert result_event["clips"][0]["notes"] == "arm trapped, hips elevated | blue gi matches reference image 1"


@pytest.mark.asyncio
async def test_highlight_analyze_node_taxonomy_and_actor_each_called_exactly_once():
    """Cost model after the 2026-07-26 cutover: exactly 2 calls per highlight
    (taxonomy + actor) — no validator loop to inflate the count regardless of
    how many highlights are analyzed."""
    call_count = {"n": 0}
    taxonomy_calls = {"n": 0}
    actor_calls = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] % 2 == 1:
            taxonomy_calls["n"] += 1
            return _taxonomy_response()
        actor_calls["n"] += 1
        return _actor_response()

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [
        {"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None},
        {"index": 2, "start_s": 50.0, "end_s": 55.0, "adjustment": None},
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    assert taxonomy_calls["n"] == 2  # once per highlight
    assert actor_calls["n"] == 2  # once per highlight
    result_events = [e for e in events if e["type"] == "highlight_result"]
    assert len(result_events) == 2
    assert all(e["status"] == "analyzed" for e in result_events)


@pytest.mark.asyncio
async def test_highlight_analyze_node_taxonomy_call_transport_error_skips_highlight_never_calls_actor():
    """A transport-level failure on the taxonomy call must abort just THIS
    highlight with an error event — never call the actor axis for it, never
    crash the whole run."""
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

    assert call_count["n"] == 1  # NEVER attempted the actor call after taxonomy failed
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["highlight_index"] == 1
    assert "taxonomy call" in error_events[0]["message"]
    assert not [e for e in events if e["type"] == "highlight_result"]


@pytest.mark.asyncio
async def test_highlight_analyze_node_actor_call_transport_error_skips_highlight_never_hangs():
    """The taxonomy call succeeds, but the actor call itself raises a
    transport-level error — the highlight must be skipped with an error
    event (never a fabricated result), and the loop must continue to any
    remaining highlight."""
    call_count = {"n": 0}

    async def _fake(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        # Per-highlight call order is taxonomy, actor — use modulo so this
        # resets correctly across BOTH highlights.
        slot = call_count["n"] % 2
        if slot == 1:
            return _taxonomy_response()
        raise RuntimeError("simulated actor transport failure")

    ctx = RunContext(youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
                      gemini_api_key="k", request_timeout_ms=1000)
    ctx.highlights = [
        {"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None},
        {"index": 2, "start_s": 50.0, "end_s": 55.0, "adjustment": None},
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    # 2 calls for highlight 1 (taxonomy, actor-raises) + 2 for highlight 2 = 4.
    assert call_count["n"] == 4
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 2
    assert all("actor call" in e["message"] for e in error_events)
    assert error_events[0]["highlight_index"] == 1
    assert error_events[1]["highlight_index"] == 2
    assert not [e for e in events if e["type"] == "highlight_result"]
    assert ctx.final_clips == []


@pytest.mark.asyncio
async def test_highlight_analyze_node_forwards_video_mime_type_on_both_calls(monkeypatch):
    """INS-140 (2026-07-26 re-scope AC12): the taxonomy call is BRAND NEW at
    this cutover — must not silently reintroduce the missing-mime_type bug
    the actor call already had fixed."""
    captured_kwargs = []

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_kwargs.append(kw)
        if len(captured_kwargs) == 1:
            return _taxonomy_response()
        return _actor_response()

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(
        youtube_id="x", youtube_url="files/abc123", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000, video_mime_type="video/mp4",
    )
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]

    [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    assert len(captured_kwargs) == 2
    assert captured_kwargs[0]["mime_type"] == "video/mp4"  # taxonomy call
    assert captured_kwargs[1]["mime_type"] == "video/mp4"  # actor call


# =============================================================================== #
# Budget-gate factor — the REAL two-phase gate inside run_pipeline (not just
# estimate_run_plan's pre-flight upper bound — see test_pipelines_executors.py
# for that half). Factor = 2 (taxonomy + actor, fixed — no more validator
# iterations to inflate it; see _highlight_per_highlight_call_factor).
# =============================================================================== #
@pytest.mark.asyncio
async def test_run_pipeline_real_two_phase_gate_uses_per_highlight_factor(monkeypatch):
    """2 highlights x factor(2: taxonomy + actor) = 4 real calls > budget_cap=3
    -> must abort BEFORE the deep loop, zero real Gemini calls spent (build
    plan item 8: never under-count)."""
    from service.pipelines import registry

    pdef = registry.get_default("highlight-scan-analyze")

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
    events = [e async for e in run_pipeline(pdef, ctx, planned, budget_cap=3)]

    assert analyze_calls["n"] == 0  # aborted before any real PASS-2+ call
    assert events[-1]["type"] == "error"
    assert "2" in events[-1]["message"] and "4" in events[-1]["message"]


@pytest.mark.asyncio
async def test_run_pipeline_real_two_phase_gate_within_cap_runs_normally(monkeypatch):
    """Same shape, but budget_cap=4 (exactly the pre-flight factor) — the
    deep loop runs normally: each highlight's REAL spend is taxonomy + actor
    = 2 calls, 4 total, at cap but never over."""
    from service.pipelines import registry

    pdef = registry.get_default("highlight-scan-analyze")

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
        # Per highlight: call 1 = taxonomy, call 2 = actor.
        slot = n % 2
        if slot == 1:
            return _taxonomy_response()
        return _actor_response()

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
                      start_sec=0, end_sec=90, gemini_api_key="fake-key", request_timeout_ms=60_000)
    planned = estimate_run_plan(pdef, duration_sec=90)
    events = [e async for e in run_pipeline(pdef, ctx, planned, budget_cap=4)]

    types_seen = [e["type"] for e in events]
    assert types_seen[-1] == "run_complete"
    assert "error" not in types_seen
    assert analyze_calls["n"] == 4  # 2 highlights x 2 (taxonomy+actor)
