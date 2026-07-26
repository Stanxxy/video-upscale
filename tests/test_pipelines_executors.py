"""``service/pipelines/executors.py`` — budget guard estimation, dedup ``key_by``
fix (Brooks R6), return_format schema selection, and an end-to-end mocked run
of the ``vision-mimic`` pipeline through the streaming run loop.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors as genai_errors
from PIL import Image

from service.pipelines import executors, frame_source, gemini_retry, highlight_scan, registry, simplified_tags
from service.pipelines.executors import (
    DEFAULT_BUDGET_CAP,
    RunContext,
    _apply_dedup_bucket_key,
    _apply_simplified_tags_dedup_keys,
    _clip_events_to_highlight_bounds,
    _convert_chunk_relative_to_absolute,
    _response_schema_for,
    estimate_run_plan,
    run_pipeline,
)
from service.pipelines.models import (
    ChunkAnalyzeConfig,
    ChunkSegmentConfig,
    DedupConfig,
    DetectCropConfig,
    HighlightAnalyzeConfig,
    HighlightScanConfig,
)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retry tests in this module use real transient google.genai.errors and
    the real gemini_retry.call_with_retry backoff loop — mock asyncio.sleep
    process-wide so nothing in this file ever actually sleeps."""
    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", AsyncMock())


def _server_error(code: int = 503) -> genai_errors.ServerError:
    return genai_errors.ServerError(code, {"message": f"{code} error", "status": "UNAVAILABLE"}, None)


# --------------------------------------------------------------------------- #
# Budget estimation (pre-flight guard)
# --------------------------------------------------------------------------- #
def test_estimate_run_plan_single_shot_is_one_call():
    pdef = registry.get_default("single-shot")
    planned = estimate_run_plan(pdef, duration_sec=30)
    assert planned["planned_gemini_calls"] == 1
    assert planned["planned_windows"] == 1
    assert planned["planned_frames"] == 0


def test_estimate_run_plan_single_shot_multi_agent_is_four_calls():
    pdef = registry.get_default("single-shot")
    analyze_stage = next(s for s in pdef.stages if s.type == "analyze")
    analyze_stage.config["agent_mode"] = "multi"
    planned = estimate_run_plan(pdef, duration_sec=30)
    assert planned["planned_gemini_calls"] == 4


def test_estimate_run_plan_vision_mimic_counts_detect_plus_analyze():
    pdef = registry.get_default("vision-mimic")
    # 10s @ 5fps = 50 frames; window=30/stride=15 -> windows: [0:30],[15:45],[30:50] = 3
    planned = estimate_run_plan(pdef, duration_sec=10)
    assert planned["planned_frames"] == 50
    assert planned["planned_windows"] == 3
    assert planned["planned_gemini_calls"] == 50 + 3  # 1 detect/frame + 1 analyze/window


def test_estimate_run_plan_vision_mimic_multi_multiplies_analyze_calls_by_four():
    pdef = registry.get_default("vision-mimic-multi")
    planned = estimate_run_plan(pdef, duration_sec=10)
    assert planned["planned_gemini_calls"] == 50 + 3 * 4


def test_estimate_run_plan_detect_crop_disabled_drops_detect_calls():
    pdef = registry.get_default("vision-mimic")
    next(s for s in pdef.stages if s.type == "detect_crop").enabled = False
    planned = estimate_run_plan(pdef, duration_sec=10)
    assert planned["planned_gemini_calls"] == 3  # analyze calls only, no detect calls


def test_default_budget_cap_rejects_a_large_scope():
    pdef = registry.get_default("vision-mimic")
    planned = estimate_run_plan(pdef, duration_sec=600)  # 10 minutes
    assert planned["planned_gemini_calls"] > DEFAULT_BUDGET_CAP


# --------------------------------------------------------------------------- #
# chunk-segment-tags budget guard — Brooks seam #2: a REAL upper bound
# (1 + ceil(duration/min_chunk_s)), not the flat "1" a non-frame pipeline
# falls through to (that flat estimate is exactly the bypass this closes).
# --------------------------------------------------------------------------- #
def test_estimate_run_plan_chunk_segment_tags_is_real_upper_bound():
    import math

    pdef = registry.get_default("chunk-segment-tags")
    planned = estimate_run_plan(pdef, duration_sec=97)
    expected_chunks = math.ceil(97 / 5.0)  # default min_chunk_s=5.0
    assert planned["planned_windows"] == expected_chunks
    assert planned["planned_gemini_calls"] == 1 + expected_chunks
    assert planned["planned_frames"] == 0


def test_estimate_run_plan_chunk_segment_tags_scales_with_min_chunk_s():
    pdef = registry.get_default("chunk-segment-tags")
    next(s for s in pdef.stages if s.type == "chunk_segment").config["min_chunk_s"] = 10.0
    planned = estimate_run_plan(pdef, duration_sec=100)
    assert planned["planned_windows"] == 10
    assert planned["planned_gemini_calls"] == 11


def test_estimate_run_plan_chunk_segment_tags_never_returns_flat_one():
    """Regression guard for the exact bypass this seam closes: a non-trivial
    duration must NOT fall through to the generic non-frame-pipeline
    ``planned_gemini_calls: multiplier`` (== 1) default."""
    pdef = registry.get_default("chunk-segment-tags")
    planned = estimate_run_plan(pdef, duration_sec=600)
    assert planned["planned_gemini_calls"] != 1
    assert planned["planned_gemini_calls"] > DEFAULT_BUDGET_CAP  # a long scope fails closed pre-stream


# --------------------------------------------------------------------------- #
# highlight-scan-analyze budget guard — same real-upper-bound discipline as
# the chunk_segment branch above (1 + ceil(duration/min_highlight_s)), never
# the flat "1" a non-frame pipeline falls through to.
# --------------------------------------------------------------------------- #
def test_estimate_run_plan_highlight_scan_analyze_is_real_upper_bound():
    import math

    pdef = registry.get_default("highlight-scan-analyze")
    planned = estimate_run_plan(pdef, duration_sec=97)
    expected_highlights = math.ceil(97 / 3.0)  # default min_highlight_s=3.0
    # 2026-07-26 single-call re-scope: highlight_analyze now costs 2 calls
    # per highlight (ONE taxonomy call + ONE actor call), down from
    # 3 (position+technique+actor) + max_validator_iterations — this
    # pipeline has no highlight_critique stage, so the factor is exactly 2
    # (see _highlight_per_highlight_call_factor).
    assert planned["planned_windows"] == expected_highlights
    assert planned["planned_gemini_calls"] == 1 + expected_highlights * 2
    assert planned["planned_frames"] == 0


def test_estimate_run_plan_highlight_scan_analyze_scales_with_min_highlight_s():
    pdef = registry.get_default("highlight-scan-analyze")
    next(s for s in pdef.stages if s.type == "highlight_scan").config["min_highlight_s"] = 10.0
    planned = estimate_run_plan(pdef, duration_sec=100)
    assert planned["planned_windows"] == 10
    assert planned["planned_gemini_calls"] == 1 + 10 * 2  # factor=2, see test above


# --------------------------------------------------------------------------- #
# highlight-scan-critique-analyze budget guard — same real-upper-bound
# discipline, PLUS the highlight_critique stage's +1 per-highlight factor
# (build plan item 8).
# --------------------------------------------------------------------------- #
def test_estimate_run_plan_highlight_scan_critique_analyze_default_factor_is_three():
    import math

    pdef = registry.get_default("highlight-scan-critique-analyze")
    planned = estimate_run_plan(pdef, duration_sec=97)
    expected_highlights = math.ceil(97 / 3.0)  # default min_highlight_s=3.0
    # factor = 1 (critique) + 2 (taxonomy + actor) = 3.
    assert planned["planned_windows"] == expected_highlights
    assert planned["planned_gemini_calls"] == 1 + expected_highlights * 3
    assert planned["planned_frames"] == 0


# --------------------------------------------------------------------------- #
# Dedup key_by fix (Brooks R6)
# --------------------------------------------------------------------------- #
def test_dedup_bucket_key_actor_player_id_overrides_absent_role():
    raw_results = [{
        "window": 1, "frames": [0, 1],
        "analysis": {"clips": [
            {"start_frame": 0, "end_frame": 10, "action": "mount", "actor_player_id": "p1", "confidence": 0.9},
            {"start_frame": 5, "end_frame": 15, "action": "mount", "actor_player_id": "p2", "confidence": 0.8},
        ]},
    }]
    prepared = _apply_dedup_bucket_key(raw_results, "actor_player_id")
    roles = {c["role"] for c in prepared[0]["analysis"]["clips"]}
    assert roles == {"p1", "p2"}  # bucketed by actor_player_id, not the (absent) schema role


def test_dedup_bucket_key_falls_back_to_role_when_no_actor_player_id():
    raw_results = [{
        "window": 1, "frames": [0],
        "analysis": {"clips": [{"start_frame": 0, "end_frame": 10, "action": "mount", "role": "blue gi"}]},
    }]
    prepared = _apply_dedup_bucket_key(raw_results, "actor_player_id")
    assert prepared[0]["analysis"]["clips"][0]["role"] == "blue gi"


def test_dedup_bucket_key_role_mode_leaves_role_as_is():
    raw_results = [{
        "window": 1, "frames": [0],
        "analysis": {"clips": [{"start_frame": 0, "end_frame": 10, "role": "blue gi", "actor_player_id": "p1"}]},
    }]
    prepared = _apply_dedup_bucket_key(raw_results, "role")
    assert prepared[0]["analysis"]["clips"][0]["role"] == "blue gi"


def test_dedup_bucket_key_does_not_mutate_input():
    raw_results = [{
        "window": 1, "frames": [0],
        "analysis": {"clips": [{"start_frame": 0, "end_frame": 10, "actor_player_id": "p1"}]},
    }]
    _apply_dedup_bucket_key(raw_results, "actor_player_id")
    assert "role" not in raw_results[0]["analysis"]["clips"][0]  # deep-copied, original untouched


@pytest.mark.asyncio
async def test_dedup_node_merges_clips_bucketed_by_actor_player_id():
    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.final_format = "clips-v1"
    ctx.raw_results = [
        {"window": 1, "frames": [0, 1], "analysis": {"clips": [
            {"start_frame": 0, "end_frame": 10, "action": "mount", "actor_player_id": "p1", "confidence": 0.9},
        ]}},
        {"window": 2, "frames": [1, 2], "analysis": {"clips": [
            {"start_frame": 8, "end_frame": 20, "action": "mount", "actor_player_id": "p1", "confidence": 0.8},
        ]}},
    ]
    events = [e async for e in executors.dedup_node(ctx, DedupConfig().model_dump())]
    assert ctx.dedup_applied is True
    assert len(ctx.final_clips) == 1  # merged: same actor, same action, overlapping
    assert ctx.final_clips[0]["start_frame"] == 0
    assert ctx.final_clips[0]["end_frame"] == 20
    assert any(e["type"] == "progress" for e in events)


# --------------------------------------------------------------------------- #
# simplified-tags-v1 dedup adaptation
# --------------------------------------------------------------------------- #
def test_apply_simplified_tags_dedup_keys_injects_role_and_action_aliases():
    raw_results = [{
        "window": 1, "frames": [0, 1],
        "analysis": {"clips": [
            {"start_frame": 0, "end_frame": 10, "position": "mount", "actor": "p1",
             "action_class": "submission_arm_lock", "outcome": "successful", "confidence": 0.9},
        ]},
    }]
    prepared = _apply_simplified_tags_dedup_keys(raw_results)
    clip = prepared[0]["analysis"]["clips"][0]
    assert clip["role"] == "p1"
    assert clip["action"] == "submission_arm_lock"
    # originals untouched
    assert "role" not in raw_results[0]["analysis"]["clips"][0]
    assert "action" not in raw_results[0]["analysis"]["clips"][0]


def test_apply_simplified_tags_dedup_keys_handles_contested_actor():
    raw_results = [{
        "window": 1, "frames": [0],
        "analysis": {"clips": [
            {"start_frame": 0, "end_frame": 5, "position": "scramble", "actor": "contested",
             "action_class": "transition", "outcome": "unclear", "confidence": 0.4},
        ]},
    }]
    prepared = _apply_simplified_tags_dedup_keys(raw_results)
    assert prepared[0]["analysis"]["clips"][0]["role"] == "contested"


@pytest.mark.asyncio
async def test_dedup_node_merges_simplified_tags_clips_bucketed_by_actor():
    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.final_format = "simplified-tags-v1"
    ctx.raw_results = [
        {"window": 1, "frames": [0, 1], "analysis": {"clips": [
            {"start_frame": 0, "end_frame": 10, "position": "mount", "actor": "p1",
             "action_class": "submission_arm_lock", "outcome": "unsuccessful", "confidence": 0.9},
        ]}},
        {"window": 2, "frames": [1, 2], "analysis": {"clips": [
            {"start_frame": 8, "end_frame": 20, "position": "mount", "actor": "p1",
             "action_class": "submission_arm_lock", "outcome": "successful", "confidence": 0.85},
        ]}},
    ]
    events = [e async for e in executors.dedup_node(ctx, DedupConfig().model_dump())]
    assert ctx.dedup_applied is True
    assert len(ctx.final_clips) == 1  # merged: same actor, same action_class, overlapping
    assert ctx.final_clips[0]["start_frame"] == 0
    assert ctx.final_clips[0]["end_frame"] == 20
    assert any("actor (fixed for simplified-tags-v1)" in e.get("message", "") for e in events)


@pytest.mark.asyncio
async def test_dedup_node_skips_when_upstream_format_is_raw_text():
    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.final_format = "raw-text"
    ctx.final_clips = []
    events = [e async for e in executors.dedup_node(ctx, DedupConfig().model_dump())]
    assert ctx.dedup_applied is False
    assert "raw-text" in ctx.dedup_skip_reason
    assert any("skipped" in e.get("message", "") for e in events)


@pytest.mark.asyncio
async def test_dedup_node_skips_when_upstream_format_is_events_v1():
    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.final_format = "events-v1"
    ctx.final_clips = [{"start": "00:01", "end": "00:02", "label": "x", "description": "d", "confidence": 0.5}]
    events = [e async for e in executors.dedup_node(ctx, DedupConfig().model_dump())]
    assert ctx.dedup_applied is False
    assert ctx.dedup_skip_reason is not None
    assert "events-v1" in ctx.dedup_skip_reason
    assert any("skipped" in e.get("message", "") for e in events)
    # final_clips untouched (still the raw events list, not deduped/mutated).
    assert ctx.final_clips[0]["label"] == "x"


# --------------------------------------------------------------------------- #
# return_format schema selection
# --------------------------------------------------------------------------- #
def test_response_schema_for_clips_v1_is_none():
    assert _response_schema_for("clips-v1") is None


def test_response_schema_for_events_v1_is_event_schema():
    from google.genai import types as genai_types

    schema = _response_schema_for("events-v1")
    assert isinstance(schema, genai_types.Schema)


def test_response_schema_for_raw_text_is_none():
    assert _response_schema_for("raw-text") is None


def test_response_schema_for_simplified_tags_v1_is_the_4axis_schema():
    from google.genai import types as genai_types

    schema = _response_schema_for("simplified-tags-v1")
    assert isinstance(schema, genai_types.Schema)
    tag_schema = schema.properties["clips"].items
    assert list(tag_schema.properties["position"].enum) == simplified_tags.POSITION_VALUES


# --------------------------------------------------------------------------- #
# analyze routing: simplified-tags-v1 uses SimplifiedTagsAnalyzer, NEVER
# analyzer.BJJTechniqueAnalyzer/BJJMultiAgentAnalyzer (clips-v1/events-v1 paths
# stay byte-identical / unchanged).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_simplified_tags_v1_analyze_uses_simplified_tags_analyzer_not_bjj_analyzer(monkeypatch):
    from service.pipelines.models import AnalyzeConfig

    bjj_analyzer_constructed = {"called": False}

    def _spy_bjj_init(self, *a, **kw):
        bjj_analyzer_constructed["called"] = True

    monkeypatch.setattr(executors.BJJTechniqueAnalyzer, "__init__", _spy_bjj_init)
    monkeypatch.setattr(executors.BJJMultiAgentAnalyzer, "__init__", _spy_bjj_init)

    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["model"] = model
        captured["config"] = config
        return SimpleNamespace(text=json.dumps({"current_context_summary": "s", "clips": []}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.windows = [[frame_source.CropFrame(frame_index=0, image=_tiny_image(), degraded=None)]]
    cfg = AnalyzeConfig(return_format="simplified-tags-v1")
    events = [e async for e in executors._analyze_frame_windows(ctx, cfg)]

    assert bjj_analyzer_constructed["called"] is False  # production analyzer classes never touched
    assert captured["config"].system_instruction == simplified_tags.DEFAULT_TAXONOMY_TEXT
    assert any(e["type"] == "window_result" and e["format"] == "simplified-tags-v1" for e in events)
    assert ctx.final_format == "simplified-tags-v1"


@pytest.mark.asyncio
async def test_simplified_tags_v1_with_multi_agent_yields_error_and_never_calls_gemini():
    from service.pipelines.models import AnalyzeConfig

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.windows = [[frame_source.CropFrame(frame_index=0, image=_tiny_image(), degraded=None)]]
    cfg = AnalyzeConfig(return_format="simplified-tags-v1", agent_mode="multi")
    events = [e async for e in executors._analyze_frame_windows(ctx, cfg)]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "agent_mode='multi'" in events[0]["message"]


@pytest.mark.asyncio
async def test_clips_v1_analyze_still_uses_bjj_technique_analyzer_unchanged(monkeypatch):
    """Regression guard: clips-v1 path must still construct BJJTechniqueAnalyzer,
    never SimplifiedTagsAnalyzer — the new format is fully additive."""
    from service.pipelines.models import AnalyzeConfig

    constructed = {"bjj": False, "simplified": False}

    def _spy_bjj_init(self, *a, **kw):
        constructed["bjj"] = True
        self.client = MagicMock()
        self.model_id = "gemini-3.1-flash-lite"
        self.temperature = 0.2
        self.thinking_config = None
        self.taxonomy_text = None
        self.taxonomy_path = None
        self._request_timeout_ms = 60_000

    async def _fake_analyze(self, frames, indices, ctx, *, response_schema=None):
        return json.dumps({"clips": []})

    monkeypatch.setattr(executors.BJJTechniqueAnalyzer, "__init__", _spy_bjj_init)
    monkeypatch.setattr(executors.BJJTechniqueAnalyzer, "analyze_sequence_async", _fake_analyze)

    orig_simplified_init = executors.simplified_tags.SimplifiedTagsAnalyzer.__init__

    def _spy_simplified_init(self, *a, **kw):
        constructed["simplified"] = True
        return orig_simplified_init(self, *a, **kw)

    monkeypatch.setattr(executors.simplified_tags.SimplifiedTagsAnalyzer, "__init__", _spy_simplified_init)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.windows = [[frame_source.CropFrame(frame_index=0, image=_tiny_image(), degraded=None)]]
    cfg = AnalyzeConfig(return_format="clips-v1")
    [e async for e in executors._analyze_frame_windows(ctx, cfg)]

    assert constructed["bjj"] is True
    assert constructed["simplified"] is False


# --------------------------------------------------------------------------- #
# system_instruction wiring (analyze node -> analyzer taxonomy_text)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_frame_based_analyze_forwards_system_instruction_as_taxonomy_text(monkeypatch):
    from service.pipelines.models import AnalyzeConfig

    captured_kwargs = {}

    def _spy_init(self, *a, **kw):
        captured_kwargs.update(kw)
        # Skip the real network client construction; still exercise real __init__ logic.
        self.client = MagicMock()
        self.model_id = kw.get("model_id", "gemini-3.1-flash-lite")
        self.temperature = kw.get("temperature", 0.2)
        self.thinking_config = kw.get("thinking_config")
        self.taxonomy_text = kw.get("taxonomy_text")
        self.taxonomy_path = a[1] if len(a) > 1 else kw.get("taxonomy_path")
        self._request_timeout_ms = 60_000

    monkeypatch.setattr(executors.BJJTechniqueAnalyzer, "__init__", _spy_init)

    async def _fake_analyze(self, frames, indices, ctx, *, response_schema=None):
        return json.dumps({"clips": []})

    monkeypatch.setattr(executors.BJJTechniqueAnalyzer, "analyze_sequence_async", _fake_analyze)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.windows = []  # no windows to iterate; we only care about analyzer construction
    cfg = AnalyzeConfig(system_instruction="  CUSTOM TAXONOMY TEXT  ")
    events = [e async for e in executors._analyze_frame_windows(ctx, cfg)]
    assert events == []  # no windows -> no per-window events, just the constructor call
    assert captured_kwargs["taxonomy_text"] == "CUSTOM TAXONOMY TEXT"  # whitespace-trimmed


@pytest.mark.asyncio
async def test_frame_based_analyze_omits_taxonomy_text_when_system_instruction_blank():
    from service.pipelines.models import AnalyzeConfig

    captured_kwargs = {}
    orig_init = executors.BJJTechniqueAnalyzer.__init__

    def _spy_init(self, *a, **kw):
        captured_kwargs.update(kw)
        self.client = MagicMock()
        self.model_id = "gemini-3.1-flash-lite"
        self.temperature = 0.2
        self.thinking_config = None
        self.taxonomy_text = kw.get("taxonomy_text")
        self.taxonomy_path = None
        self._request_timeout_ms = 60_000

    executors.BJJTechniqueAnalyzer.__init__ = _spy_init
    try:
        ctx = RunContext(
            youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
            gemini_api_key="k", request_timeout_ms=1000,
        )
        ctx.windows = []
        cfg = AnalyzeConfig(system_instruction="   ")  # whitespace-only
        [e async for e in executors._analyze_frame_windows(ctx, cfg)]
        assert captured_kwargs["taxonomy_text"] is None
    finally:
        executors.BJJTechniqueAnalyzer.__init__ = orig_init


@pytest.mark.asyncio
async def test_video_native_analyze_forwards_system_instruction_to_gemini_config(monkeypatch):
    from service.pipelines.models import AnalyzeConfig

    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["config"] = config
        return SimpleNamespace(text="[]")

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    cfg = AnalyzeConfig(system_instruction="CUSTOM TAXONOMY", return_format="events-v1")
    [e async for e in executors._analyze_video_window(ctx, cfg)]
    assert captured["config"].system_instruction == "CUSTOM TAXONOMY"


# --------------------------------------------------------------------------- #
# detect_crop_node — Bug regression: a malformed 5-element ``box_2d`` gets
# dropped by the REAL (unmocked) ``_parse_detect_entries``/``_run_detect``
# path, leaving <2 usable athletes for that frame; the frame after it is back
# to a full pair. Real traceback was an IndexError inside
# ``frame_source._pair_continuity`` via ``AthleteTracker.resolve``. Only the
# Gemini transport (``generate_content``) is mocked here — the malformed-box
# drop, role parsing, top-2 selection, and tracker hysteresis all run for
# real, so this exercises the FULL live crash path, not just the pure
# tracker unit tests in test_pipelines_frame_source.py.
# --------------------------------------------------------------------------- #
def _tiny_image():
    return Image.new("RGB", (100, 100), color="gray")


@pytest.mark.asyncio
async def test_detect_crop_node_malformed_box_dropped_to_single_athlete_never_crashes(monkeypatch):
    responses = [
        # Frame 0: clean 2-athlete pair — establishes the tracker's held pair.
        json.dumps([
            {"box_2d": [100, 100, 400, 300], "label": "athlete: white gi", "confidence": 0.9},
            {"box_2d": [500, 100, 800, 300], "label": "athlete: blue gi", "confidence": 0.9},
        ]),
        # Frame 1: one real athlete + one MALFORMED 5-element box_2d (mirrors
        # the live log: "dropping malformed entry ... box_2d: [210, 271, 0,
        # 693, 467]"). The real _parse_detect_entries drops the malformed
        # entry, leaving exactly 1 usable athlete -> select_top2 degrades to
        # "single_athlete". This is the frame that used to crash the NEXT
        # frame's tracker.resolve() call.
        json.dumps([
            {"box_2d": [105, 100, 405, 300], "label": "athlete: white gi", "confidence": 0.9},
            {"box_2d": [210, 271, 0, 693, 467], "label": "athlete: blue gi", "confidence": 0.85},
        ]),
        # Frame 2: back to a clean 2-athlete pair — this is the exact call
        # that raised "IndexError: list index out of range" before the fix
        # (held_boxes had collapsed to length 1 on frame 1).
        json.dumps([
            {"box_2d": [108, 100, 408, 300], "label": "athlete: white gi", "confidence": 0.9},
            {"box_2d": [505, 100, 805, 300], "label": "athlete: blue gi", "confidence": 0.9},
        ]),
    ]
    call_state = {"i": 0}

    async def _fake_generate(*, model, contents, config):
        text = responses[call_state["i"]]
        call_state["i"] += 1
        return SimpleNamespace(text=text)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=1,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.crops = [
        frame_source.CropFrame(frame_index=0, image=_tiny_image()),
        frame_source.CropFrame(frame_index=10, image=_tiny_image()),
        frame_source.CropFrame(frame_index=20, image=_tiny_image()),
    ]

    cfg = DetectCropConfig().model_dump()
    events = [e async for e in executors.detect_crop_node(ctx, cfg)]  # must not raise IndexError

    progress = [e for e in events if e["type"] == "progress"]
    assert [e["degraded"] for e in progress] == [None, "single_athlete", None]
    assert not any(e["type"] == "error" for e in events)

    # All 3 frames survive (single_athlete crops the one real box, never
    # fabricating/dropping a frame that has at least 1 usable athlete).
    assert len(ctx.crops) == 3
    assert [c.degraded for c in ctx.crops] == [None, "single_athlete", None]


# --------------------------------------------------------------------------- #
# detect_crop_node — transient-retry resilience (root cause: frame 1116
# "503 UNAVAILABLE" bug report). Runs the REAL _run_detect/gemini_retry code
# path (only client.aio.models.generate_content is mocked) so this pins the
# actual retry wiring through RunContext.retry_config, not just the pure
# gemini_retry unit tests.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_detect_crop_node_retries_transient_503_then_succeeds(monkeypatch):
    good_entries = json.dumps([
        {"box_2d": [100, 100, 400, 300], "label": "athlete: white gi", "confidence": 0.9},
        {"box_2d": [500, 100, 800, 300], "label": "athlete: blue gi", "confidence": 0.9},
    ])
    call_count = {"n": 0}

    async def _fake_generate(*, model, contents, config):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise _server_error(503)
        return SimpleNamespace(text=good_entries)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=1,
        gemini_api_key="k", request_timeout_ms=1000,
        retry_config=gemini_retry.GeminiRetryConfig(max_attempts=5, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0),
    )
    ctx.crops = [frame_source.CropFrame(frame_index=0, image=_tiny_image())]

    events = [e async for e in executors.detect_crop_node(ctx, DetectCropConfig().model_dump())]

    assert not any(e["type"] == "error" for e in events)
    assert len(ctx.crops) == 1  # the frame survives — retry succeeded, nothing dropped/fabricated
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_detect_crop_node_exhausted_retries_drops_frame_as_error_not_fabricated(monkeypatch):
    """Persistent 503s exhaust every retry attempt: the frame is surfaced via
    the existing ``error`` NDJSON event and dropped from ctx.crops (the
    pre-existing honest-failure path) — never a fabricated detection."""
    async def _fake_generate(*, model, contents, config):
        raise _server_error(503)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=1,
        gemini_api_key="k", request_timeout_ms=1000,
        retry_config=gemini_retry.GeminiRetryConfig(max_attempts=3, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0),
    )
    ctx.crops = [frame_source.CropFrame(frame_index=1116, image=_tiny_image())]

    events = [e async for e in executors.detect_crop_node(ctx, DetectCropConfig().model_dump())]

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "frame 1116" in error_events[0]["message"]
    assert "503" in error_events[0]["message"]
    assert len(ctx.crops) == 0  # dropped honestly, never fabricated


@pytest.mark.asyncio
async def test_chunk_segment_node_retries_transient_error_then_succeeds(monkeypatch):
    call_count = {"n": 0}

    async def _fake_generate(*, model, contents, config):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise _server_error(500)
        return SimpleNamespace(text=json.dumps({"chunks": [
            {"start_s": 0, "end_s": 20, "reason": "match_action"},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=20,
        gemini_api_key="k", request_timeout_ms=1000,
        retry_config=gemini_retry.GeminiRetryConfig(max_attempts=3, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0),
    )
    events = [e async for e in executors.chunk_segment_node(ctx, ChunkSegmentConfig().model_dump())]

    assert not any(e["type"] == "error" for e in events)
    assert call_count["n"] == 2
    assert len(ctx.chunks) == 1


@pytest.mark.asyncio
async def test_analyze_video_window_retries_transient_error_then_succeeds(monkeypatch):
    from service.pipelines.models import AnalyzeConfig

    call_count = {"n": 0}

    async def _fake_generate(*, model, contents, config):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise _server_error(429)
        return SimpleNamespace(text="[]")

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
        retry_config=gemini_retry.GeminiRetryConfig(max_attempts=3, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0),
    )
    cfg = AnalyzeConfig(return_format="events-v1")
    events = [e async for e in executors._analyze_video_window(ctx, cfg)]

    assert not any(e["type"] == "error" for e in events)
    assert call_count["n"] == 2
    assert ctx.final_clips == []


# --------------------------------------------------------------------------- #
# End-to-end mocked run — vision-mimic pipeline through run_pipeline()
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_pipeline_vision_mimic_end_to_end_mocked(monkeypatch):
    """Full run loop, all I/O mocked: ffmpeg sampling, Gemini detect, Gemini analyze.
    Verifies event sequence, degraded frame drop, dedup application."""
    pdef = registry.get_default("vision-mimic")
    # Shrink window config so 3 sampled frames -> 1 window (keeps the test fast).
    next(s for s in pdef.stages if s.type == "window").config = {"window_size": 3, "stride": 3}
    next(s for s in pdef.stages if s.type == "frame_sample").config = {"sample_fps": 3}

    monkeypatch.setattr(frame_source, "probe_native_fps", lambda url, headers: 30.0)
    monkeypatch.setattr(
        executors, "_resolve_stream_url", lambda url: ("http://stream", {}),
    )
    monkeypatch.setattr(
        frame_source, "sample_frames",
        lambda *a, **kw: [
            frame_source.SampledFrame(frame_index=0, image=_tiny_image()),
            frame_source.SampledFrame(frame_index=10, image=_tiny_image()),
            frame_source.SampledFrame(frame_index=20, image=_tiny_image()),
        ],
    )

    # Frame 1: 2 athletes detected+cropped. Frame 2: dropped (no_detection). Frame 3: 2 athletes.
    detect_results = [
        ([
            {"box_2d": [100, 100, 400, 300], "label": "athlete: white gi", "confidence": 0.9},
            {"box_2d": [500, 100, 800, 300], "label": "athlete: blue gi", "confidence": 0.9},
        ], "gemini-3.5-flash", 5.0),
        ([{"box_2d": [0, 0, 10, 10], "label": "referee: black", "confidence": 0.9}], "gemini-3.5-flash", 5.0),
        ([
            {"box_2d": [110, 100, 410, 300], "label": "athlete: white gi", "confidence": 0.9},
            {"box_2d": [510, 100, 810, 300], "label": "athlete: blue gi", "confidence": 0.9},
        ], "gemini-3.5-flash", 5.0),
    ]
    call_state = {"i": 0}

    async def _fake_run_detect(client, image, model=None, thinking_config=None, prompt=None, retry_config=None):
        result = detect_results[call_state["i"]]
        call_state["i"] += 1
        return result

    monkeypatch.setattr(executors, "_run_detect", _fake_run_detect)

    async def _fake_analyze_sequence_async(self, frames, indices, ctx, *, response_schema=None):
        return json.dumps({
            "current_context_summary": "ctx-1",
            "clips": [{
                "start_frame": indices[0], "end_frame": indices[-1], "action": "mount",
                "actor_player_id": "p1", "confidence": 0.8,
            }],
        })

    monkeypatch.setattr(executors.BJJTechniqueAnalyzer, "analyze_sequence_async", _fake_analyze_sequence_async)
    monkeypatch.setattr(executors.BJJTechniqueAnalyzer, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(
        youtube_id="jNQXAC9IVRw", youtube_url="https://youtube.com/watch?v=jNQXAC9IVRw",
        start_sec=0, end_sec=1, gemini_api_key="fake-key", request_timeout_ms=60_000,
    )
    planned = estimate_run_plan(pdef, duration_sec=1)
    events = [e async for e in run_pipeline(pdef, ctx, planned)]

    types_seen = [e["type"] for e in events]
    assert types_seen[0] == "run_start"
    assert types_seen[-1] == "run_complete"
    assert "window_result" in types_seen
    assert "crops" in types_seen

    run_complete = events[-1]
    assert run_complete["dedup_applied"] is True
    assert len(run_complete["clips"]) == 1
    assert run_complete["degraded_counts"].get("no_detection") == 1
    assert run_complete["degraded_counts"].get("clean") == 2


@pytest.mark.asyncio
async def test_run_pipeline_reports_error_event_on_gemini_failure(monkeypatch):
    pdef = registry.get_default("single-shot")

    async def _boom(*a, **kw):
        raise RuntimeError("simulated Gemini outage")

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("simulated Gemini outage"))
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    planned = estimate_run_plan(pdef, duration_sec=10)
    events = [e async for e in run_pipeline(pdef, ctx, planned)]
    assert any(e["type"] == "error" for e in events)
    assert events[-1]["type"] == "run_complete"  # run still terminates cleanly


@pytest.mark.asyncio
async def test_run_pipeline_vision_mimic_simplified_end_to_end_mocked(monkeypatch):
    """Full run loop for the new simplified-taxonomy pipeline, all I/O mocked.
    Verifies enum values flow through window_result -> dedup -> run_complete
    untouched, and the format tag is 'simplified-tags-v1' throughout."""
    pdef = registry.get_default("vision-mimic-simplified")
    next(s for s in pdef.stages if s.type == "window").config = {"window_size": 3, "stride": 3}
    next(s for s in pdef.stages if s.type == "frame_sample").config = {"sample_fps": 3}

    monkeypatch.setattr(frame_source, "probe_native_fps", lambda url, headers: 30.0)
    monkeypatch.setattr(executors, "_resolve_stream_url", lambda url: ("http://stream", {}))
    monkeypatch.setattr(
        frame_source, "sample_frames",
        lambda *a, **kw: [
            frame_source.SampledFrame(frame_index=0, image=_tiny_image()),
            frame_source.SampledFrame(frame_index=10, image=_tiny_image()),
            frame_source.SampledFrame(frame_index=20, image=_tiny_image()),
        ],
    )

    async def _fake_run_detect(client, image, model=None, thinking_config=None, prompt=None, retry_config=None):
        return (
            [
                {"box_2d": [100, 100, 400, 300], "label": "athlete: white gi", "confidence": 0.9},
                {"box_2d": [500, 100, 800, 300], "label": "athlete: blue gi", "confidence": 0.9},
            ],
            "gemini-3.5-flash", 5.0,
        )

    monkeypatch.setattr(executors, "_run_detect", _fake_run_detect)

    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({
            "current_context_summary": "ctx-1",
            "clips": [{
                "start_frame": 0, "end_frame": 20,
                "position": "mount", "actor": "p1",
                "action_class": "submission_arm_lock", "outcome": "unsuccessful",
                "specific_technique_guess": "possibly a kimura", "confidence": 0.75,
            }],
        }))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="jNQXAC9IVRw", youtube_url="https://youtube.com/watch?v=jNQXAC9IVRw",
        start_sec=0, end_sec=1, gemini_api_key="fake-key", request_timeout_ms=60_000,
    )
    planned = estimate_run_plan(pdef, duration_sec=1)
    events = [e async for e in run_pipeline(pdef, ctx, planned)]

    window_results = [e for e in events if e["type"] == "window_result"]
    assert window_results, "expected at least one window_result event"
    for wr in window_results:
        assert wr["format"] == "simplified-tags-v1"
        for tag in wr["clips"]:
            assert tag["position"] in simplified_tags.POSITION_VALUES
            assert tag["action_class"] in simplified_tags.ACTION_CLASS_VALUES
            assert tag["outcome"] in simplified_tags.OUTCOME_VALUES

    run_complete = events[-1]
    assert run_complete["format"] == "simplified-tags-v1"
    assert run_complete["dedup_applied"] is True
    assert len(run_complete["clips"]) == 1
    assert run_complete["clips"][0]["position"] == "mount"
    assert run_complete["clips"][0]["specific_technique_guess"] == "possibly a kimura"


# =============================================================================== #
# chunk-segment-tags — chunk_segment_node / chunk_analyze_node / time dedup /
# end-to-end mocked run. Brooks seams: native_fps stays None end-to-end,
# chunk-relative -> absolute conversion is real (never fabricated), context
# threading honors context_chain.enabled, skipped chunks make ZERO deep calls.
# =============================================================================== #
def test_convert_chunk_relative_to_absolute_adds_effective_start():
    raw_clips = [
        {"start_s": 2.0, "end_s": 7.0, "actor": "top", "action_class": "sweep", "confidence": 0.8},
    ]
    out = _convert_chunk_relative_to_absolute(raw_clips, effective_start=100.0, chunk_index=1)
    assert out[0]["start_s"] == 102.0
    assert out[0]["end_s"] == 107.0


def test_convert_chunk_relative_to_absolute_drops_malformed_clip_never_fabricates():
    raw_clips = [
        {"start_s": 2.0, "end_s": 7.0, "actor": "top", "action_class": "sweep", "confidence": 0.8},
        {"actor": "bottom", "action_class": "escape", "confidence": 0.5},  # missing start_s/end_s
    ]
    out = _convert_chunk_relative_to_absolute(raw_clips, effective_start=0.0, chunk_index=2)
    assert len(out) == 1


@pytest.mark.asyncio
async def test_chunk_segment_node_produces_chunk_map_with_worth_analysis(monkeypatch):
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"chunks": [
            {"start_s": 0, "end_s": 20, "reason": "pre_match"},
            {"start_s": 20, "end_s": 80, "reason": "match_action"},
            {"start_s": 80, "end_s": 100, "reason": "broadcast"},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    cfg = ChunkSegmentConfig().model_dump()
    events = [e async for e in executors.chunk_segment_node(ctx, cfg)]

    chunk_map_events = [e for e in events if e["type"] == "chunk_map"]
    assert len(chunk_map_events) == 1
    chunks = chunk_map_events[0]["chunks"]
    assert [c["reason"] for c in chunks] == ["pre_match", "match_action", "broadcast"]
    assert [c["worth_analysis"] for c in chunks] == [False, True, False]
    assert [c["index"] for c in chunks] == [1, 2, 3]
    assert ctx.chunks == chunks
    assert ctx.native_fps is None  # this pipeline never samples frames


def _existing_shape_chunk_map_consumer(event: dict) -> dict:
    """Mimics a consumer written BEFORE Task 1 (ADCC eval dashboard additive
    per-call emit) — the QA VLM Studio frontend's chunk_map handler only ever
    reads these three keys. Used to prove the additive change is byte-
    identical for anything that doesn't know the new keys exist."""
    return {"type": event["type"], "chunks": event["chunks"], "timing": event["timing"]}


@pytest.mark.asyncio
async def test_chunk_map_additive_emit_existing_consumer_byte_identical_plus_new_fields(monkeypatch):
    """Task 1 regression: chunk_map must gain prompt_text/raw_response_text/
    usage_metadata WITHOUT changing any pre-existing key's value — an
    existing-shape consumer that only reads {type, chunks, timing} sees
    EXACTLY what it saw before this change."""
    raw_text = json.dumps({"chunks": [
        {"start_s": 0, "end_s": 20, "reason": "pre_match"},
        {"start_s": 20, "end_s": 80, "reason": "match_action"},
        {"start_s": 80, "end_s": 100, "reason": "broadcast"},
    ]})

    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(
            text=raw_text,
            usage_metadata=SimpleNamespace(prompt_token_count=1234, candidates_token_count=56, total_token_count=1290),
        )

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    cfg = ChunkSegmentConfig().model_dump()
    events = [e async for e in executors.chunk_segment_node(ctx, cfg)]
    chunk_map_event = next(e for e in events if e["type"] == "chunk_map")

    # Existing-shape consumer: byte-identical to pre-Task-1 behavior.
    existing_view = _existing_shape_chunk_map_consumer(chunk_map_event)
    assert existing_view == {
        "type": "chunk_map",
        "chunks": ctx.chunks,
        "timing": chunk_map_event["timing"],
    }
    assert [c["reason"] for c in existing_view["chunks"]] == ["pre_match", "match_action", "broadcast"]

    # New additive fields, real values (never fabricated/None when the SDK
    # response actually carried them).
    assert chunk_map_event["prompt_text"].strip().startswith("You are scanning a BJJ video")
    assert "0" in chunk_map_event["prompt_text"] and "100" in chunk_map_event["prompt_text"]
    assert chunk_map_event["raw_response_text"] == raw_text
    assert chunk_map_event["usage_metadata"] == {
        "prompt_token_count": 1234, "candidates_token_count": 56, "total_token_count": 1290,
    }


@pytest.mark.asyncio
async def test_chunk_map_additive_emit_usage_metadata_none_when_sdk_response_lacks_it(monkeypatch):
    """A response double with no usage_metadata attribute at all (e.g. a bare
    SimpleNamespace(text=...), same shape as every pre-existing test in this
    file) must yield usage_metadata=None, never an AttributeError and never a
    fabricated value."""
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"chunks": [{"start_s": 0, "end_s": 20, "reason": "match_action"}]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=20,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    events = [e async for e in executors.chunk_segment_node(ctx, ChunkSegmentConfig().model_dump())]
    chunk_map_event = next(e for e in events if e["type"] == "chunk_map")
    assert chunk_map_event["usage_metadata"] is None


@pytest.mark.asyncio
async def test_chunk_segment_node_surfaces_gemini_error_never_fabricates(monkeypatch):
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("simulated outage"))
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=60,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    events = [e async for e in executors.chunk_segment_node(ctx, ChunkSegmentConfig().model_dump())]
    assert any(e["type"] == "error" for e in events)
    assert ctx.chunks == []


@pytest.mark.asyncio
async def test_chunk_analyze_node_skips_non_worthy_chunks_with_zero_deep_calls_and_reason():
    call_count = {"n": 0}

    class _CountingAnalyzer:
        def __init__(self, *a, **kw):
            pass

        async def analyze_chunk(self, *a, **kw):
            call_count["n"] += 1
            return json.dumps({"current_context_summary": "s", "clips": []})

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 0, "end_s": 20, "reason": "pre_match", "worth_analysis": False, "adjustment": None},
        {"index": 2, "start_s": 20, "end_s": 80, "reason": "match_action", "worth_analysis": True, "adjustment": None},
        {"index": 3, "start_s": 80, "end_s": 100, "reason": "broadcast", "worth_analysis": False, "adjustment": None},
    ]

    import service.pipelines.simplified_tags as st_module
    orig_cls = st_module.SimplifiedTagsTimeAnalyzer
    st_module.SimplifiedTagsTimeAnalyzer = _CountingAnalyzer
    try:
        events = [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig().model_dump())]
    finally:
        st_module.SimplifiedTagsTimeAnalyzer = orig_cls

    assert call_count["n"] == 1  # only the ONE worth_analysis chunk got a real Gemini call
    skip_events = [e for e in events if e.get("skipped_chunk")]
    assert len(skip_events) == 2
    assert {e["reason"] for e in skip_events} == {"pre_match", "broadcast"}
    assert all(e["type"] == "progress" for e in skip_events)
    chunk_starts = [e for e in events if e["type"] == "chunk_start"]
    assert len(chunk_starts) == 1  # only the worthy chunk gets a chunk_start/chunk_result pair
    assert ctx.native_fps is None


@pytest.mark.asyncio
async def test_chunk_analyze_node_converts_chunk_relative_to_absolute_with_overlap(monkeypatch):
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["video_part"] = contents[0].parts[1]
        return SimpleNamespace(text=json.dumps({
            "current_context_summary": "s",
            "clips": [{"start_s": 2.0, "end_s": 6.0, "position": "mount", "actor": "top",
                       "action_class": "submission_arm_lock", "outcome": "successful", "confidence": 0.9}],
        }))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 30, "end_s": 60, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]
    cfg = ChunkAnalyzeConfig(overlap_s=5.0).model_dump()
    events = [e async for e in executors.chunk_analyze_node(ctx, cfg)]

    # effective_start = max(0, 30 - 5) = 25 -> video_metadata sent [25, 60]
    assert captured["video_part"].video_metadata.start_offset == "25.000s"
    assert captured["video_part"].video_metadata.end_offset == "60.000s"

    result_events = [e for e in events if e["type"] == "chunk_result"]
    assert len(result_events) == 1
    clip = result_events[0]["clips"][0]
    # chunk-relative [2.0, 6.0] + effective_start(25) -> absolute [27.0, 31.0]
    assert clip["start_s"] == 27.0
    assert clip["end_s"] == 31.0
    assert ctx.final_format == "simplified-tags-time-v1"
    assert ctx.native_fps is None


def _existing_shape_chunk_result_consumer(event: dict) -> dict:
    """Mimics a pre-Task-1 consumer of chunk_result — only these keys were
    ever read (mirrors ``_existing_shape_chunk_map_consumer`` above)."""
    return {
        "type": event["type"], "chunk_index": event["chunk_index"], "format": event["format"],
        "clips": event["clips"], "context_summary": event["context_summary"], "timing": event["timing"],
    }


@pytest.mark.asyncio
async def test_chunk_result_additive_emit_existing_consumer_byte_identical_plus_new_fields(monkeypatch):
    """Task 1 regression (PASS-2 sibling of the chunk_map test above):
    chunk_result must gain prompt_text/raw_response_text/usage_metadata
    WITHOUT changing any pre-existing key — read off the REAL
    SimplifiedTagsTimeAnalyzer's post-call instrumentation slots, not a
    change to analyze_chunk's return contract."""
    raw_text = json.dumps({
        "current_context_summary": "s",
        "clips": [{"start_s": 2.0, "end_s": 6.0, "position": "mount", "actor": "top",
                   "action_class": "submission_arm_lock", "outcome": "successful", "confidence": 0.9}],
    })

    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(
            text=raw_text,
            usage_metadata=SimpleNamespace(prompt_token_count=987, candidates_token_count=65, total_token_count=1052),
        )

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 30, "end_s": 60, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]
    events = [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig(overlap_s=5.0).model_dump())]
    result_event = next(e for e in events if e["type"] == "chunk_result")

    existing_view = _existing_shape_chunk_result_consumer(result_event)
    assert existing_view == {
        "type": "chunk_result", "chunk_index": 1, "format": "simplified-tags-time-v1",
        "clips": result_event["clips"], "context_summary": "s", "timing": result_event["timing"],
    }
    assert result_event["clips"][0]["start_s"] == 27.0  # unaffected by the additive fields

    assert "NATIVE video clip" in result_event["prompt_text"]
    assert result_event["raw_response_text"] == raw_text
    assert result_event["usage_metadata"] == {
        "prompt_token_count": 987, "candidates_token_count": 65, "total_token_count": 1052,
    }


@pytest.mark.asyncio
async def test_chunk_result_additive_emit_none_when_analyzer_replaced_by_test_double():
    """When a caller monkeypatches SimplifiedTagsTimeAnalyzer entirely (e.g.
    the existing ``_CountingAnalyzer``/plain-function test doubles elsewhere
    in this file), the analyzer instance never sets the instrumentation
    slots — chunk_analyze_node must degrade to None, never raise
    AttributeError, and never fabricate a value."""
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return json.dumps({"current_context_summary": "s", "clips": []})

    from service.pipelines.simplified_tags import SimplifiedTagsTimeAnalyzer

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 0, "end_s": 20, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig().model_dump())]

    result_event = next(e for e in events if e["type"] == "chunk_result")
    assert result_event["prompt_text"] is None
    assert result_event["raw_response_text"] is None
    assert result_event["usage_metadata"] is None


@pytest.mark.asyncio
async def test_chunk_analyze_node_overlap_never_goes_before_scope_start():
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"current_context_summary": "s", "clips": []}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=10, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 12, "end_s": 40, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]
    captured = {}

    async def _capture_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured["start_sec"] = start_sec
        return json.dumps({"current_context_summary": "s", "clips": []})

    from service.pipelines.simplified_tags import SimplifiedTagsTimeAnalyzer

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SimplifiedTagsTimeAnalyzer, "analyze_chunk", _capture_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: fake_client)
        [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig(overlap_s=5.0).model_dump())]

    # start_s(12) - overlap(5) = 7, clamped up to ctx.start_sec(10)
    assert captured["start_sec"] == 10


# --------------------------------------------------------------------------- #
# Bug regression — "chunk-segment-tags emitted a chunk OVER max length".
# Root cause: chunk_analyze_node's own back-overlap (`chunk["start_s"] -
# overlap_s`) was applied on top of a base chunk that repair_chunk_bounds had
# already capped AT max_chunk_s (one of its over-max split pieces), silently
# growing the REAL analyzed span (`chunk_end - effective_start`, what the
# `chunk_start` NDJSON event's "scope" reports) past max_chunk_s. Fix: thread
# PASS 1's max_chunk_s onto ctx.chunk_max_s and clamp effective_start so the
# with-overlap span never exceeds it.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_chunk_analyze_node_overlap_never_pushes_analyzed_span_over_max_chunk_s(monkeypatch):
    """Regression for the live bug: a base chunk sitting exactly AT
    max_chunk_s (as repair_chunk_bounds's over-max split would produce) must
    not grow past max_chunk_s once the 5s back-overlap is applied."""
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["video_part"] = contents[0].parts[1]
        return SimpleNamespace(text=json.dumps({"current_context_summary": "s", "clips": []}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=200,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    # A split-piece chunk exactly AT the 60s cap (e.g. repair_chunk_bounds's
    # second over-max split piece: [55, 115]) — before the fix, applying the
    # chunk_analyze back-overlap on top pushed the sent span to [50, 115] = 65s.
    ctx.chunks = [
        {"index": 1, "start_s": 55, "end_s": 115, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]
    ctx.chunk_max_s = 60.0  # set by chunk_segment_node from ChunkSegmentConfig.max_chunk_s

    events = [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig(overlap_s=5.0).model_dump())]

    chunk_start_events = [e for e in events if e["type"] == "chunk_start"]
    assert len(chunk_start_events) == 1
    scope_start, scope_end = chunk_start_events[0]["scope"]
    assert scope_end - scope_start <= 60.0  # NEVER exceeds max_chunk_s
    assert scope_start == 55.0  # clamped to chunk["start_s"] itself, not start_s - overlap_s(50)
    assert scope_end == 115.0

    assert captured["video_part"].video_metadata.start_offset == "55.000s"
    assert captured["video_part"].video_metadata.end_offset == "115.000s"


@pytest.mark.asyncio
async def test_chunk_analyze_node_overlap_fits_under_cap_for_a_short_base_chunk(monkeypatch):
    """A base chunk well under max_chunk_s still gets its full back-overlap —
    the clamp only trims overlap that would exceed the cap, never overlap
    that comfortably fits under it (no regression of the normal case)."""
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["video_part"] = contents[0].parts[1]
        return SimpleNamespace(text=json.dumps({"current_context_summary": "s", "clips": []}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=200,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 30, "end_s": 40, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]
    ctx.chunk_max_s = 60.0

    [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig(overlap_s=5.0).model_dump())]

    # 10s base chunk + full 5s overlap = 15s, nowhere near the 60s cap.
    assert captured["video_part"].video_metadata.start_offset == "25.000s"
    assert captured["video_part"].video_metadata.end_offset == "40.000s"


@pytest.mark.asyncio
async def test_chunk_analyze_node_no_chunk_max_s_set_preserves_prior_behavior(monkeypatch):
    """``ctx.chunk_max_s`` defaults to None (e.g. a caller that never ran
    chunk_segment_node first) — the new clamp must be a strict no-op then,
    identical to the pre-fix behavior."""
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["video_part"] = contents[0].parts[1]
        return SimpleNamespace(text=json.dumps({"current_context_summary": "s", "clips": []}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=200,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 55, "end_s": 115, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]
    assert ctx.chunk_max_s is None

    [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig(overlap_s=5.0).model_dump())]

    assert captured["video_part"].video_metadata.start_offset == "50.000s"  # unclamped: 55 - 5
    assert captured["video_part"].video_metadata.end_offset == "115.000s"


@pytest.mark.asyncio
async def test_chunk_segment_node_sets_ctx_chunk_max_s_from_config(monkeypatch):
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"chunks": [
            {"start_s": 0, "end_s": 50, "reason": "match_action"},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=50,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    cfg = ChunkSegmentConfig(max_chunk_s=45.0).model_dump()
    [e async for e in executors.chunk_segment_node(ctx, cfg)]

    assert ctx.chunk_max_s == 45.0


@pytest.mark.asyncio
async def test_chunk_analyze_node_context_threading_honors_context_chain_disabled():
    responses = iter([
        json.dumps({"current_context_summary": "chained after chunk 1", "clips": []}),
        json.dumps({"current_context_summary": "chained after chunk 2", "clips": []}),
    ])
    captured_contexts = []

    async def _capture_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_contexts.append(previous_context)
        return next(responses)

    from service.pipelines.simplified_tags import SimplifiedTagsTimeAnalyzer

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 0, "end_s": 20, "reason": "match_action", "worth_analysis": True, "adjustment": None},
        {"index": 2, "start_s": 20, "end_s": 40, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]
    ctx.stage_enabled = {"context_chain": False}  # DISABLED

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SimplifiedTagsTimeAnalyzer, "analyze_chunk", _capture_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig().model_dump())]

    # context_chain disabled -> every chunk gets "Start of match." regardless
    # of the previous chunk's current_context_summary.
    assert captured_contexts == ["Start of match.", "Start of match."]


@pytest.mark.asyncio
async def test_chunk_analyze_node_context_threading_honors_context_chain_enabled():
    responses = iter([
        json.dumps({"current_context_summary": "chained after chunk 1", "clips": []}),
        json.dumps({"current_context_summary": "chained after chunk 2", "clips": []}),
    ])
    captured_contexts = []

    async def _capture_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_contexts.append(previous_context)
        return next(responses)

    from service.pipelines.simplified_tags import SimplifiedTagsTimeAnalyzer

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 0, "end_s": 20, "reason": "match_action", "worth_analysis": True, "adjustment": None},
        {"index": 2, "start_s": 20, "end_s": 40, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]
    ctx.stage_enabled = {"context_chain": True}  # ENABLED

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SimplifiedTagsTimeAnalyzer, "analyze_chunk", _capture_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig().model_dump())]

    assert captured_contexts == ["Start of match.", "chained after chunk 1"]


@pytest.mark.asyncio
async def test_chunk_analyze_node_surfaces_pass2_gemini_outage_never_silent_empty_clips():
    """R1 regression: SimplifiedTagsTimeAnalyzer.analyze_chunk() swallows its
    own Gemini/transport exception and returns ``{"error": "..."}`` as valid
    JSON (it never raises) — so chunk_analyze_node must explicitly detect
    that "error" key and surface a type:"error" event for THAT chunk instead
    of silently falling through to ``.get("clips", [])`` -> [] (which would
    be indistinguishable from a real clean chunk with 0 events). The loop
    must CONTINUE to the next chunk (chunks are independent), not abort the
    whole run."""
    responses = iter([
        json.dumps({"error": "simulated PASS-2 outage: 503 UNAVAILABLE"}),
        json.dumps({
            "current_context_summary": "s",
            "clips": [{"start_s": 1.0, "end_s": 4.0, "position": "mount", "actor": "top",
                       "action_class": "submission_arm_lock", "outcome": "successful", "confidence": 0.9}],
        }),
    ])

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return next(responses)

    from service.pipelines.simplified_tags import SimplifiedTagsTimeAnalyzer

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 0, "end_s": 20, "reason": "match_action", "worth_analysis": True, "adjustment": None},
        {"index": 2, "start_s": 20, "end_s": 40, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig().model_dump())]

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["chunk_index"] == 1
    assert "PASS-2 Gemini call failed" in error_events[0]["message"]
    assert "simulated PASS-2 outage" in error_events[0]["message"]

    # The errored chunk must NOT get a chunk_result (never rendered as a
    # silent "0 events found" clean result) — only chunk 2 (which succeeded)
    # produces one, proving the loop CONTINUED past the error rather than
    # aborting the whole run.
    result_events = [e for e in events if e["type"] == "chunk_result"]
    assert len(result_events) == 1
    assert result_events[0]["chunk_index"] == 2

    # final_clips reflects only the real clip from the surviving chunk — the
    # errored chunk contributes nothing (never fabricated).
    assert len(ctx.final_clips) == 1
    # chunk 2 effective_start = max(0, 20 - default overlap_s(5)) = 15, + rel(1.0)
    assert ctx.final_clips[0]["start_s"] == 16.0


@pytest.mark.asyncio
async def test_chunk_analyze_node_no_chunks_errors_never_fabricates():
    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = []
    events = [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig().model_dump())]
    assert any(e["type"] == "error" for e in events)
    # Contract fix (2026-07-18): chunk_index is ALWAYS a key on a chunk_analyze
    # error event, even with no specific chunk to attribute (None here) —
    # same discipline as the analogous highlight_analyze no-highlights guard.
    error_event = next(e for e in events if e["type"] == "error")
    assert error_event["chunk_index"] is None
    assert ctx.final_clips == []
    assert ctx.final_format == "simplified-tags-time-v1"


@pytest.mark.asyncio
async def test_chunk_analyze_node_transport_exception_error_event_attaches_chunk_index():
    """Contract fix (2026-07-18): the transport-exception error-yield site
    (analyzer.analyze_chunk RAISES rather than returning an {"error": ...}
    payload) previously omitted chunk_index — only the dict-shaped-error path
    attached it. Both paths must now be attributable to their chunk."""
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        raise RuntimeError("simulated transport failure")

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 0, "end_s": 20, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig().model_dump())]

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["stage_id"] == "chunk_analyze"
    assert error_events[0]["chunk_index"] == 1
    assert "simulated transport failure" in error_events[0]["message"]


@pytest.mark.asyncio
async def test_chunk_analyze_node_non_json_response_error_event_attaches_chunk_index():
    """Same contract fix, non-JSON-response error-yield site."""
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return "not valid json {{{"

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.chunks = [
        {"index": 1, "start_s": 0, "end_s": 20, "reason": "match_action", "worth_analysis": True, "adjustment": None},
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.chunk_analyze_node(ctx, ChunkAnalyzeConfig().model_dump())]

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["chunk_index"] == 1
    assert "non-JSON response" in error_events[0]["message"]


# --------------------------------------------------------------------------- #
# dedup_node — simplified-tags-time-v1 seconds-based branch
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_dedup_node_merges_time_v1_clips_by_iou_and_axis_identity():
    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.final_format = "simplified-tags-time-v1"
    ctx.raw_results = [
        {"window": 1, "frames": [], "analysis": {"clips": [
            {"start_s": 58.0, "end_s": 63.0, "actor": "bottom", "action_class": "escape", "confidence": 0.7},
        ]}},
        {"window": 2, "frames": [], "analysis": {"clips": [
            {"start_s": 59.0, "end_s": 64.0, "actor": "bottom", "action_class": "escape", "confidence": 0.9},
        ]}},
    ]
    events = [e async for e in executors.dedup_node(ctx, DedupConfig().model_dump())]
    assert ctx.dedup_applied is True
    assert ctx.native_fps is None  # never fabricated — this format has no frame unit at all
    assert len(ctx.final_clips) == 1
    assert ctx.final_clips[0]["start_s"] == 58.0
    assert ctx.final_clips[0]["end_s"] == 64.0
    assert any("time-based dedup" in e.get("message", "") for e in events)


@pytest.mark.asyncio
async def test_dedup_node_time_v1_empty_clips_applies_cleanly_no_skip_reason():
    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.final_format = "simplified-tags-time-v1"
    ctx.raw_results = []
    [e async for e in executors.dedup_node(ctx, DedupConfig().model_dump())]
    assert ctx.dedup_applied is True
    assert ctx.final_clips == []
    assert ctx.dedup_skip_reason is None


# --------------------------------------------------------------------------- #
# End-to-end mocked run — chunk-segment-tags through run_pipeline()
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_run_pipeline_chunk_segment_tags_end_to_end_mocked(monkeypatch):
    pdef = registry.get_default("chunk-segment-tags")

    call_log = {"segment": 0, "analyze": 0}

    async def _fake_segment_generate(*, model, contents, config):
        call_log["segment"] += 1
        return SimpleNamespace(text=json.dumps({"chunks": [
            {"start_s": 0, "end_s": 10, "reason": "pre_match"},
            {"start_s": 10, "end_s": 50, "reason": "match_action"},
            {"start_s": 50, "end_s": 55, "reason": "restart_reset"},
            {"start_s": 55, "end_s": 90, "reason": "match_action"},
            {"start_s": 90, "end_s": 100, "reason": "broadcast"},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_segment_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_log["analyze"] += 1
        return json.dumps({
            "current_context_summary": f"after chunk at {start_sec}",
            "clips": [{
                "start_s": 1.0, "end_s": 4.0, "position": "mount", "actor": "top",
                "action_class": "submission_arm_lock", "outcome": "successful", "confidence": 0.8,
            }],
        })

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
        start_sec=0, end_sec=100, gemini_api_key="fake-key", request_timeout_ms=60_000,
    )
    planned = estimate_run_plan(pdef, duration_sec=100)
    events = [e async for e in run_pipeline(pdef, ctx, planned)]

    types_seen = [e["type"] for e in events]
    assert types_seen[0] == "run_start"
    assert types_seen[-1] == "run_complete"
    assert "chunk_map" in types_seen
    assert "chunk_start" in types_seen
    assert "chunk_result" in types_seen

    # Only the 2 match_action chunks made a REAL PASS-2 call; the other 3 did not.
    assert call_log["analyze"] == 2
    assert call_log["segment"] == 1

    skip_progress = [
        e for e in events if e["type"] == "progress" and e.get("stage_id") == "chunk_analyze" and e.get("skipped_chunk")
    ]
    assert len(skip_progress) == 3
    assert {e["reason"] for e in skip_progress} == {"pre_match", "restart_reset", "broadcast"}

    run_complete = events[-1]
    assert run_complete["format"] == "simplified-tags-time-v1"
    assert run_complete["native_fps"] is None  # NEVER fabricated — this pipeline is entirely native-video
    assert run_complete["dedup_applied"] is True
    assert len(run_complete["clips"]) == 2  # both chunk clips, non-overlapping in time -> not merged
    for clip in run_complete["clips"]:
        assert clip["position"] in simplified_tags.POSITION_VALUES
        assert clip["action_class"] in simplified_tags.ACTION_CLASS_VALUES


# --------------------------------------------------------------------------- #
# Task 1 — durable budget fix: two-phase gate on the REAL worthy-chunk count,
# checked INSIDE run_pipeline right after chunk_segment's PASS 1 completes,
# BEFORE chunk_analyze's sequential deep loop runs.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_run_pipeline_chunk_segment_tags_aborts_before_deep_loop_when_worthy_count_over_budget_cap(monkeypatch):
    """2 match_action (worthy) chunks + budget_cap=1 -> the run must emit an
    `error` event and abort BEFORE chunk_analyze's deep loop — ZERO real
    SimplifiedTagsTimeAnalyzer.analyze_chunk() calls, never a wasted Gemini call."""
    pdef = registry.get_default("chunk-segment-tags")

    async def _fake_segment_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"chunks": [
            {"start_s": 0, "end_s": 10, "reason": "pre_match"},
            {"start_s": 10, "end_s": 50, "reason": "match_action"},
            {"start_s": 50, "end_s": 55, "reason": "restart_reset"},
            {"start_s": 55, "end_s": 90, "reason": "match_action"},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_segment_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    analyze_calls = {"count": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        analyze_calls["count"] += 1
        return json.dumps({"clips": []})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
        start_sec=0, end_sec=90, gemini_api_key="fake-key", request_timeout_ms=60_000,
    )
    planned = estimate_run_plan(pdef, duration_sec=90)
    # 2 worthy (match_action) chunks in the fake chunk map above; cap=1 forces the abort.
    events = [e async for e in run_pipeline(pdef, ctx, planned, budget_cap=1)]

    assert analyze_calls["count"] == 0  # zero deep-analyze calls happened
    types_seen = [e["type"] for e in events]
    assert "chunk_map" in types_seen  # PASS 1 still streamed before the abort
    assert "chunk_start" not in types_seen  # PASS 2 never started
    assert "chunk_result" not in types_seen
    assert types_seen[-1] == "error"  # aborts on the error event, no run_complete follows
    error_event = events[-1]
    assert error_event["stage_id"] == "chunk_analyze"
    assert "budget cap" in error_event["message"]
    assert "min_chunk_s" in error_event["message"]  # pipeline-aware advice, not sample_fps/window/agent_mode
    assert "sample_fps" not in error_event["message"]
    assert "agent_mode" not in error_event["message"]


@pytest.mark.asyncio
async def test_run_pipeline_chunk_segment_tags_within_budget_cap_runs_the_deep_loop_normally(monkeypatch):
    """Same 2-worthy-chunk chunk map, but budget_cap=60 (the default) —
    the deep loop runs normally through to run_complete, no abort."""
    pdef = registry.get_default("chunk-segment-tags")

    async def _fake_segment_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"chunks": [
            {"start_s": 0, "end_s": 10, "reason": "pre_match"},
            {"start_s": 10, "end_s": 50, "reason": "match_action"},
            {"start_s": 50, "end_s": 55, "reason": "restart_reset"},
            {"start_s": 55, "end_s": 90, "reason": "match_action"},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_segment_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    analyze_calls = {"count": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        analyze_calls["count"] += 1
        return json.dumps({"clips": []})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
        start_sec=0, end_sec=90, gemini_api_key="fake-key", request_timeout_ms=60_000,
    )
    planned = estimate_run_plan(pdef, duration_sec=90)
    events = [e async for e in run_pipeline(pdef, ctx, planned, budget_cap=DEFAULT_BUDGET_CAP)]

    assert analyze_calls["count"] == 2
    types_seen = [e["type"] for e in events]
    assert types_seen[-1] == "run_complete"
    assert "error" not in types_seen


@pytest.mark.asyncio
async def test_run_pipeline_chunk_segment_tags_no_budget_cap_arg_skips_the_gate_entirely(monkeypatch):
    """budget_cap=None (default, e.g. a direct caller that never resolves an
    effective cap) means the post-PASS-1 gate is simply not consulted — the
    run proceeds through the deep loop unconditionally. The real caller
    (qa_pipeline.pipeline_run) always resolves and passes an effective cap;
    this pins the function's own default-arg behavior."""
    pdef = registry.get_default("chunk-segment-tags")

    async def _fake_segment_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"chunks": [
            {"start_s": 0, "end_s": 50, "reason": "match_action"},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_segment_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return json.dumps({"clips": []})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
        start_sec=0, end_sec=50, gemini_api_key="fake-key", request_timeout_ms=60_000,
    )
    planned = estimate_run_plan(pdef, duration_sec=50)
    events = [e async for e in run_pipeline(pdef, ctx, planned)]  # no budget_cap kwarg at all

    types_seen = [e["type"] for e in events]
    assert types_seen[-1] == "run_complete"


# =============================================================================== #
# highlight-scan-analyze — highlight_scan_node / highlight_analyze_node /
# event clipping / end-to-end mocked run. Brooks-style seams mirrored from the
# chunk-segment-tags block above: native_fps stays None end-to-end,
# chunk-relative -> absolute conversion is real (reused, never duplicated),
# and — the pipeline's OWN new invariant — event clipping replaces dedup.
# =============================================================================== #
@pytest.mark.asyncio
async def test_highlight_scan_node_produces_highlight_map_no_reason_axis(monkeypatch):
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"highlights": [
            {"start_s": 20, "end_s": 35},
            {"start_s": 60, "end_s": 72},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    cfg = HighlightScanConfig().model_dump()
    events = [e async for e in executors.highlight_scan_node(ctx, cfg)]

    highlight_map_events = [e for e in events if e["type"] == "highlight_map"]
    assert len(highlight_map_events) == 1
    highlights = highlight_map_events[0]["highlights"]
    assert [h["start_s"] for h in highlights] == [20.0, 60.0]
    assert [h["index"] for h in highlights] == [1, 2]
    for h in highlights:
        # v2 (2026-07-18 plan): description/reasoning are additive keys
        # (None here since the fake Gemini response didn't include them).
        assert set(h) == {
            "index", "start_s", "end_s", "description", "reasoning",
            "adjustment", "start_is_synthetic", "end_is_synthetic",
        }  # no "reason"/"worth_analysis"
    assert ctx.highlights == highlights
    assert ctx.native_fps is None  # this pipeline never samples frames


@pytest.mark.asyncio
async def test_highlight_scan_node_emits_prompt_and_usage_metadata(monkeypatch):
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(
            text=json.dumps({"highlights": [{"start_s": 0, "end_s": 10}]}),
            usage_metadata=SimpleNamespace(prompt_token_count=500, candidates_token_count=20, total_token_count=520),
        )

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    events = [e async for e in executors.highlight_scan_node(ctx, HighlightScanConfig().model_dump())]
    highlight_map_event = next(e for e in events if e["type"] == "highlight_map")
    assert highlight_map_event["prompt_text"].strip().startswith("Now watch this match from")
    assert "0" in highlight_map_event["prompt_text"] and "10" in highlight_map_event["prompt_text"]
    assert highlight_map_event["usage_metadata"] == {
        "prompt_token_count": 500, "candidates_token_count": 20, "total_token_count": 520,
    }


@pytest.mark.asyncio
async def test_highlight_scan_node_surfaces_gemini_error_never_fabricates(monkeypatch):
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("simulated outage"))
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=50,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    events = [e async for e in executors.highlight_scan_node(ctx, HighlightScanConfig().model_dump())]
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "simulated outage" in error_events[0]["message"]
    assert ctx.highlights == []


@pytest.mark.asyncio
async def test_highlight_scan_node_uses_custom_system_and_initial_prompt(monkeypatch):
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["system_instruction"] = config.system_instruction
        captured["prompt_text"] = contents[0].parts[0].text
        return SimpleNamespace(text=json.dumps({"highlights": []}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=5, end_sec=15,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    cfg = HighlightScanConfig(
        system_prompt="Only flag submissions.",
        initial_prompt="Scan $start_sec to $end_sec for submissions only.",
    ).model_dump()
    [e async for e in executors.highlight_scan_node(ctx, cfg)]

    assert captured["system_instruction"] == "Only flag submissions."
    assert captured["prompt_text"] == "Scan 5 to 15 for submissions only."


# --------------------------------------------------------------------------- #
# Event clipping (replaces dedup) — the pipeline's own new invariant.
# --------------------------------------------------------------------------- #
def test_clip_events_drops_event_entirely_in_pre_roll_region():
    """An event reported only because pre-roll context video was sent, but
    that never overlaps the highlight's OWN bounds, must be DROPPED."""
    events = [{"start_s": 10.0, "end_s": 14.0, "action_class": "escape", "actor": "top"}]
    kept, dropped = _clip_events_to_highlight_bounds(events, bound_start=20.0, bound_end=35.0)
    assert kept == []
    assert dropped == 1


def test_clip_events_drops_event_entirely_in_post_roll_region():
    events = [{"start_s": 36.0, "end_s": 39.0, "action_class": "transition", "actor": "bottom"}]
    kept, dropped = _clip_events_to_highlight_bounds(events, bound_start=20.0, bound_end=35.0)
    assert kept == []
    assert dropped == 1


def test_clip_events_keeps_event_entirely_inside_bounds_unmodified():
    events = [{"start_s": 22.0, "end_s": 30.0, "action_class": "sweep", "actor": "top"}]
    kept, dropped = _clip_events_to_highlight_bounds(events, bound_start=20.0, bound_end=35.0)
    assert dropped == 0
    assert kept == events


def test_clip_events_truncates_event_straddling_the_pre_roll_boundary():
    """An event that genuinely belongs to the highlight but whose reported
    start overshoots into the pre-roll region is CLIPPED, not dropped."""
    events = [{"start_s": 17.0, "end_s": 25.0, "action_class": "guard_pass", "actor": "top"}]
    kept, dropped = _clip_events_to_highlight_bounds(events, bound_start=20.0, bound_end=35.0)
    assert dropped == 0
    assert len(kept) == 1
    assert kept[0]["start_s"] == 20.0  # clamped up to the bound
    assert kept[0]["end_s"] == 25.0  # unaffected, already inside bounds


def test_clip_events_truncates_event_straddling_the_post_roll_boundary():
    events = [{"start_s": 30.0, "end_s": 38.0, "action_class": "submission_choke", "actor": "top"}]
    kept, dropped = _clip_events_to_highlight_bounds(events, bound_start=20.0, bound_end=35.0)
    assert dropped == 0
    assert len(kept) == 1
    assert kept[0]["start_s"] == 30.0
    assert kept[0]["end_s"] == 35.0  # clamped down to the bound


def test_clip_events_drops_event_with_malformed_start_or_end_never_fabricates():
    events = [
        {"start_s": 22.0, "end_s": 30.0, "action_class": "sweep", "actor": "top"},
        {"action_class": "escape", "actor": "bottom"},  # missing start_s/end_s
    ]
    kept, dropped = _clip_events_to_highlight_bounds(events, bound_start=20.0, bound_end=35.0)
    assert len(kept) == 1
    assert dropped == 1


def test_clip_events_boundary_touching_events_are_dropped_zero_overlap():
    """An event ending exactly at bound_start (or starting exactly at
    bound_end) has ZERO real overlap with [bound_start, bound_end) and is
    dropped, not kept as a zero-duration clip."""
    events = [
        {"start_s": 10.0, "end_s": 20.0, "action_class": "escape", "actor": "top"},  # ends exactly at bound_start
        {"start_s": 35.0, "end_s": 40.0, "action_class": "escape", "actor": "top"},  # starts exactly at bound_end
    ]
    kept, dropped = _clip_events_to_highlight_bounds(events, bound_start=20.0, bound_end=35.0)
    assert kept == []
    assert dropped == 2


# --------------------------------------------------------------------------- #
# highlight_analyze_node — pre/post-roll window construction, absolute
# conversion (reused from chunk_analyze), clipping wired into the loop.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_highlight_analyze_node_sends_preroll_postroll_expanded_window(monkeypatch):
    """2026-07-26 single-call cutover: ONE taxonomy call, then ONE actor
    call — both get the SAME preroll/postroll-expanded window + shared fps."""
    captured_windows = []
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_windows.append((start_sec, end_sec, kw.get("fps")))
        call_count["n"] += 1
        if call_count["n"] == 1:
            return json.dumps({
                "position": "mount", "action_class": "submission_choke", "outcome": "successful",
                "justification": "j",
            })
        return json.dumps({"actor": "unclear", "identity_uncertain": True, "justification": "e"})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.highlights = [{"index": 1, "start_s": 20.0, "end_s": 35.0, "adjustment": None}]
    cfg = HighlightAnalyzeConfig(fps=10, preroll_s=5.0, postroll_s=4.0).model_dump()
    events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    # taxonomy call, actor call — both with the SAME window/fps.
    assert captured_windows == [(15.0, 39.0, 10)] * 2
    start_event = next(e for e in events if e["type"] == "highlight_start")
    assert start_event["scope"] == [15.0, 39.0]
    assert start_event["highlight_bounds"] == [20.0, 35.0]
    assert start_event["authoritative_bounds"] == [20.0, 35.0]  # no critique -> same as scan bounds
    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert result_event["status"] == "analyzed"
    assert result_event["clips"] == [{
        "start_s": 20.0, "end_s": 35.0,
        "position": "mount", "action_class": "submission_choke", "outcome": "successful",
        "player_id": None, "player_name": None, "identity_uncertain": True, "actor_sentinel": "unclear",
        "notes": "j | e",
    }]


@pytest.mark.asyncio
async def test_highlight_analyze_node_sends_valid_duration_offset_for_float_imprecise_window(monkeypatch):
    """Real production regression (2026-07-18): Gemini rejected a live request
    with ``400 INVALID_ARGUMENT ... Field 'start_offset' ... Invalid duration
    format, failed to parse nano seconds`` because ``window_start = max(ctx.
    start_sec, highlight["start_s"] - cfg.preroll_s)`` on a highlight starting
    at 33.3s with a 5.0s preroll computes ``33.3 - 5.0 ==
    28.299999999999997`` — 15 fractional digits, over protobuf Duration's
    9-digit limit.

    This exercises the REAL ``SimplifiedTagsTimeAnalyzer.analyze_chunk`` (only
    the transport-level ``generate_content`` call is mocked, never
    ``analyze_chunk`` itself) so the fix is proven end-to-end through the
    actual serialization boundary, not tautologically. Non-tautology verified
    manually: reverting ``simplified_tags._format_offset_seconds``'s call site
    to the old ``f"{start_sec}s"``/``f"{end_sec}s"`` makes this test FAIL
    (``start_offset == "28.299999999999997s"``, 15 fractional digits); the
    fix makes it PASS.
    """
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["video_part"] = contents[0].parts[1]
        return SimpleNamespace(text=json.dumps({"clips": []}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    # Real repro numbers: highlight starts at 33.3s, preroll_s=5.0 ->
    # window_start = 33.3 - 5.0 == 28.299999999999997 (raw float).
    ctx.highlights = [{"index": 1, "start_s": 33.3, "end_s": 40.0, "adjustment": None}]
    cfg = HighlightAnalyzeConfig(preroll_s=5.0, postroll_s=4.0).model_dump()

    events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    start_offset = captured["video_part"].video_metadata.start_offset
    end_offset = captured["video_part"].video_metadata.end_offset

    for offset in (start_offset, end_offset):
        assert offset.endswith("s")
        body = offset[:-1]
        frac_digits = body.split(".")[1] if "." in body else ""
        assert len(frac_digits) <= 9, (
            f"{offset!r} has {len(frac_digits)} fractional digits — Gemini's "
            "protobuf Duration parser rejects anything over 9 (nanoseconds)"
        )
    # Pins the exact quantized value too — never the raw 15-digit float.
    assert start_offset == "28.300s"

    error_events = [e for e in events if e["type"] == "error"]
    assert error_events == []  # the (mocked) Gemini call succeeded — no 400


@pytest.mark.asyncio
async def test_highlight_analyze_node_clamps_window_to_requested_scope():
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return json.dumps({"clips": []})

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=10, end_sec=40,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.highlights = [{"index": 1, "start_s": 12.0, "end_s": 38.0, "adjustment": None}]
    cfg = HighlightAnalyzeConfig(preroll_s=15.0, postroll_s=10.0).model_dump()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    start_event = next(e for e in events if e["type"] == "highlight_start")
    # 12-15=-3 clamped to ctx.start_sec=10; 38+10=48 clamped to ctx.end_sec=40.
    assert start_event["scope"] == [10.0, 40.0]


@pytest.mark.asyncio
async def test_highlight_analyze_node_synthesizes_one_clip_at_authoritative_bounds(monkeypatch):
    """2026-07-26 single-call cutover: the taxonomy call reports no
    timestamp — the synthesized clip's start_s/end_s are the highlight's own
    AUTHORITATIVE bounds directly, not derived from any Gemini-reported
    offset."""
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return json.dumps({
                "position": "mount", "action_class": "submission_arm_lock", "outcome": "successful",
                "justification": "arm trapped",
            })
        return json.dumps({"actor": "contested", "identity_uncertain": True, "justification": "collar grip visible"})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.highlights = [{"index": 1, "start_s": 20.0, "end_s": 35.0, "adjustment": None}]
    cfg = HighlightAnalyzeConfig(preroll_s=5.0, postroll_s=4.0).model_dump()
    events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    result_event = next(e for e in events if e["type"] == "highlight_result")
    assert result_event["status"] == "analyzed"
    assert len(result_event["clips"]) == 1
    clip = result_event["clips"][0]
    # Bounds are the highlight's OWN authoritative bounds directly (20/35),
    # never a Gemini-reported/converted offset (the taxonomy call reports no
    # timestamps at all).
    assert clip["start_s"] == 20.0
    assert clip["end_s"] == 35.0
    assert clip["position"] == "mount"
    assert clip["action_class"] == "submission_arm_lock"
    assert clip["outcome"] == "successful"
    assert ctx.final_format == "simplified-tags-time-v1"
    assert ctx.final_clips == result_event["clips"]
    assert ctx.native_fps is None


@pytest.mark.asyncio
async def test_highlight_analyze_node_no_highlights_errors_never_crashes(monkeypatch):
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())
    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=50,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]
    assert events[0]["type"] == "error"
    # Contract fix (2026-07-18): highlight_index is ALWAYS a key on a
    # highlight_analyze error event, even when there's no specific highlight
    # to attribute to (None here) — the frontend can rely on the key's
    # presence unconditionally.
    assert events[0]["highlight_index"] is None
    assert ctx.final_clips == []
    assert ctx.final_format == "simplified-tags-time-v1"


@pytest.mark.asyncio
async def test_highlight_analyze_node_swallowed_analyzer_error_surfaces_as_error_event():
    """analyze_chunk never raises on a Gemini/transport failure — it returns
    {"error": ...} as its JSON payload. highlight_analyze_node must surface
    that explicitly rather than silently rendering "0 events"."""
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return json.dumps({"error": "simulated PASS-2 outage"})

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=50,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "simulated PASS-2 outage" in error_events[0]["message"]
    assert ctx.final_clips == []


@pytest.mark.asyncio
async def test_highlight_analyze_node_continues_past_one_highlight_error():
    """Highlight 1's taxonomy call raises -> the whole highlight is skipped
    with an error event WITHOUT ever attempting the actor call (never a
    wasted Gemini call past the first failure); the loop continues to
    highlight 2, whose taxonomy+actor calls (2, 3) both succeed."""
    call_count = {"n": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient failure on highlight 1")
        if call_count["n"] == 2:  # taxonomy call for highlight 2
            return json.dumps({
                "position": "mount", "action_class": "submission_choke", "outcome": "successful",
                "justification": "j",
            })
        return json.dumps({  # actor call for highlight 2
            "actor": "unclear", "identity_uncertain": True, "justification": "j",
        })

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.highlights = [
        {"index": 1, "start_s": 5.0, "end_s": 10.0, "adjustment": None},
        {"index": 2, "start_s": 50.0, "end_s": 55.0, "adjustment": None},
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    assert call_count["n"] == 3  # highlight 1: 1 call (taxonomy, raised); highlight 2: 2 calls
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "highlight 1" in error_events[0]["message"]
    # Contract fix (2026-07-18): the transport-exception error-yield site
    # must attach highlight_index too (previously only the dict-shaped
    # {"error": ...} path did) — this is exactly that path, since the fake
    # analyzer RAISES rather than returning an {"error": ...} payload.
    assert error_events[0]["stage_id"] == "highlight_analyze"
    assert error_events[0]["highlight_index"] == 1
    result_events = [e for e in events if e["type"] == "highlight_result"]
    assert len(result_events) == 1
    assert result_events[0]["highlight_index"] == 2
    assert result_events[0]["status"] == "analyzed"
    assert ctx.final_clips == result_events[0]["clips"]


@pytest.mark.asyncio
async def test_highlight_analyze_node_no_context_chain_always_passes_none_previous_context():
    """Build-plan invariant: no text context_chain in this pipeline — every
    highlight gets a fresh previous_context (None), never a threaded summary
    from a prior highlight."""
    captured_contexts = []

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_contexts.append(previous_context)
        return json.dumps({"current_context_summary": "some narrative state"})

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.highlights = [
        {"index": 1, "start_s": 5.0, "end_s": 10.0, "adjustment": None},
        {"index": 2, "start_s": 50.0, "end_s": 55.0, "adjustment": None},
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    # Taxonomy + actor calls both fire per highlight = 2 calls x 2
    # highlights = 4 — never threaded, even though the fake returned a
    # summary.
    assert captured_contexts == [None] * 4


@pytest.mark.asyncio
async def test_highlight_analyze_node_non_json_response_error_event_attaches_highlight_index():
    """Contract fix (2026-07-18): the non-JSON-response error-yield site also
    must attach highlight_index (previously only the dict-shaped-error path did)."""
    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return "not valid json {{{"

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.highlights = [{"index": 1, "start_s": 5.0, "end_s": 15.0, "adjustment": None}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, HighlightAnalyzeConfig().model_dump())]

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["stage_id"] == "highlight_analyze"
    assert error_events[0]["highlight_index"] == 1
    assert "non-JSON response" in error_events[0]["message"]


# --------------------------------------------------------------------------- #
# Belt-and-suspenders defensive skip (live-bug fix, 2026-07-18) — never send
# Gemini an inverted or zero-length native-video offset. The REAL fix is
# highlight_scan.py's scope-clamp (see test_pipelines_highlight_scan.py and
# the executors-level scope-clamp wiring test below); this guards the case
# where an invalid window reaches highlight_analyze_node anyway (e.g. a
# directly-constructed ctx.highlights list, as in this unit test, bypassing
# the PASS-1 scope-clamp entirely).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_highlight_analyze_node_skips_highlight_with_inverted_window_never_calls_gemini():
    """Direct regression for the live repro: scope=40-85s, a highlight fully
    outside scope (start_s=111, end_s=124 relative to this ctx's own
    start/end below) would otherwise produce window_start > window_end.
    Reproduced directly here via a highlight whose OWN bounds are entirely
    past ctx.end_sec — preroll/postroll math then inverts."""
    analyze_calls = {"count": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        analyze_calls["count"] += 1
        return json.dumps({"clips": []})

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=40, end_sec=85,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    # This highlight should have been dropped by highlight_scan.py's
    # scope-clamp already (real fix); constructed directly here to exercise
    # the executors-level belt-and-suspenders guard in isolation.
    ctx.highlights = [{"index": 3, "start_s": 111.0, "end_s": 124.0, "adjustment": None}]
    cfg = HighlightAnalyzeConfig(preroll_s=5.0, postroll_s=4.0).model_dump()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    assert analyze_calls["count"] == 0  # NEVER sent to Gemini
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["stage_id"] == "highlight_analyze"
    assert error_events[0]["highlight_index"] == 3
    assert "invalid" in error_events[0]["message"].lower()
    assert ctx.final_clips == []  # no results at all — the only highlight was skipped


@pytest.mark.asyncio
async def test_highlight_analyze_node_zero_length_window_also_skipped():
    """window_start == window_end (zero-length, not just inverted) must also
    be skipped — the guard is `>=`, not `>`."""
    analyze_calls = {"count": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        analyze_calls["count"] += 1
        return json.dumps({"clips": []})

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=10,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    # Both edges synthetic -> zero preroll/postroll -> window == highlight's
    # own bounds; a zero-duration highlight would produce window_start ==
    # window_end. (Zero-duration highlights don't occur via finalize_highlights
    # in practice — this exercises the guard directly.)
    ctx.highlights = [{
        "index": 1, "start_s": 5.0, "end_s": 5.0, "adjustment": None,
        "start_is_synthetic": True, "end_is_synthetic": True,
    }]
    cfg = HighlightAnalyzeConfig(preroll_s=5.0, postroll_s=4.0).model_dump()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        events = [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    assert analyze_calls["count"] == 0
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["highlight_index"] == 1


# --------------------------------------------------------------------------- #
# highlight_scan_node wiring — scope_start/scope_end (live-bug fix, 2026-07-18)
# are threaded from ctx.start_sec/ctx.end_sec into finalize_highlights, so the
# PASS-1 highlight_map streamed to the UI is scope-valid end-to-end.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_highlight_scan_node_drops_out_of_scope_highlight_from_the_stream(monkeypatch):
    """Real repro numbers: scope=40-85s, PASS 1 returns an in-scope highlight
    plus an entirely-out-of-scope one (index-would-be 3, start_s=111,
    end_s=124) — the streamed highlight_map must never include it."""
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"highlights": [
            {"start_s": 50, "end_s": 60},
            {"start_s": 111, "end_s": 124},  # entirely outside [40, 85]
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=40, end_sec=85,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    events = [e async for e in executors.highlight_scan_node(ctx, HighlightScanConfig().model_dump())]
    highlight_map_event = next(e for e in events if e["type"] == "highlight_map")

    assert len(highlight_map_event["highlights"]) == 1
    assert highlight_map_event["highlights"][0]["start_s"] == 50.0
    assert highlight_map_event["highlights"][0]["end_s"] == 60.0
    assert ctx.highlights == highlight_map_event["highlights"]


# --------------------------------------------------------------------------- #
# End-to-end mocked run of the full highlight-scan-analyze pipeline through
# run_pipeline — no context_chain/dedup stages at all in this pipeline's
# fixed stage set.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_run_pipeline_highlight_scan_analyze_end_to_end_mocked(monkeypatch):
    pdef = registry.get_default("highlight-scan-analyze")
    assert [s.type for s in pdef.stages] == ["video_window", "highlight_scan", "highlight_analyze"]

    async def _fake_scan_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"highlights": [
            {"start_s": 10, "end_s": 20},
            {"start_s": 50, "end_s": 58},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_scan_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    analyze_calls = {"count": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        analyze_calls["count"] += 1
        return json.dumps({"clips": []})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
        start_sec=0, end_sec=90, gemini_api_key="fake-key", request_timeout_ms=60_000,
    )
    planned = estimate_run_plan(pdef, duration_sec=90)
    events = [e async for e in run_pipeline(pdef, ctx, planned, budget_cap=DEFAULT_BUDGET_CAP)]

    # 2 highlights x 2 calls each (taxonomy + actor, both fire — the fake's
    # {"clips": []} response parses fine on both, just carries no
    # position/action_class/outcome/actor keys) = 4.
    assert analyze_calls["count"] == 4
    types_seen = [e["type"] for e in events]
    assert types_seen[-1] == "run_complete"
    assert "error" not in types_seen
    # No context_chain/dedup stage events at all in this pipeline's fixed set.
    stage_types_seen = {e["stage_type"] for e in events if e["type"] == "stage_start"}
    assert stage_types_seen == {"video_window", "highlight_scan", "highlight_analyze"}


# =============================================================================== #
# Evaluator CHANGES-REQUIRED item 1 (2026-07-18) — highlight-scan-analyze's own
# two-phase gate, reapplied verbatim from the chunk-segment-tags pattern above:
# gate the REAL highlight count against budget_cap INSIDE run_pipeline right
# after highlight_scan's PASS 1 completes, BEFORE highlight_analyze's
# sequential deep loop runs.
# =============================================================================== #
@pytest.mark.asyncio
async def test_run_pipeline_highlight_scan_analyze_aborts_before_deep_loop_when_count_over_budget_cap(monkeypatch):
    """3 highlights + budget_cap=1 -> the run must emit an `error` event and
    abort BEFORE highlight_analyze's deep loop — ZERO real
    SimplifiedTagsTimeAnalyzer.analyze_chunk() calls, never a wasted Gemini call.
    highlight_scan has no worth_analysis prefilter, so ALL 3 count against cap
    (unlike chunk-segment-tags, where only match_action chunks count)."""
    pdef = registry.get_default("highlight-scan-analyze")

    async def _fake_scan_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"highlights": [
            {"start_s": 0, "end_s": 10},
            {"start_s": 20, "end_s": 30},
            {"start_s": 40, "end_s": 50},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_scan_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    analyze_calls = {"count": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        analyze_calls["count"] += 1
        return json.dumps({"clips": []})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
        start_sec=0, end_sec=90, gemini_api_key="fake-key", request_timeout_ms=60_000,
    )
    planned = estimate_run_plan(pdef, duration_sec=90)
    events = [e async for e in run_pipeline(pdef, ctx, planned, budget_cap=1)]

    assert analyze_calls["count"] == 0  # zero deep-analyze calls happened
    types_seen = [e["type"] for e in events]
    assert "highlight_map" in types_seen  # PASS 1 still streamed before the abort
    assert "highlight_start" not in types_seen  # PASS 2 never started
    assert "highlight_result" not in types_seen
    assert types_seen[-1] == "error"  # aborts on the error event, no run_complete follows
    error_event = events[-1]
    assert error_event["stage_id"] == "highlight_analyze"
    assert "budget cap" in error_event["message"]
    assert "min_highlight_s" in error_event["message"]  # pipeline-aware advice
    assert "min_chunk_s" not in error_event["message"]
    assert "sample_fps" not in error_event["message"]


@pytest.mark.asyncio
async def test_run_pipeline_highlight_scan_analyze_within_budget_cap_runs_the_deep_loop_normally(monkeypatch):
    """Same 3-highlight map, but budget_cap=60 (the default) — the deep loop
    runs normally through to run_complete, no abort."""
    pdef = registry.get_default("highlight-scan-analyze")

    async def _fake_scan_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"highlights": [
            {"start_s": 0, "end_s": 10},
            {"start_s": 20, "end_s": 30},
            {"start_s": 40, "end_s": 50},
        ]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_scan_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    analyze_calls = {"count": 0}

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        analyze_calls["count"] += 1
        return json.dumps({"clips": []})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
        start_sec=0, end_sec=90, gemini_api_key="fake-key", request_timeout_ms=60_000,
    )
    planned = estimate_run_plan(pdef, duration_sec=90)
    events = [e async for e in run_pipeline(pdef, ctx, planned, budget_cap=DEFAULT_BUDGET_CAP)]

    # 3 highlights x 2 calls each (taxonomy + actor) = 6.
    assert analyze_calls["count"] == 6
    types_seen = [e["type"] for e in events]
    assert types_seen[-1] == "run_complete"
    assert "error" not in types_seen


@pytest.mark.asyncio
async def test_run_pipeline_highlight_scan_analyze_no_budget_cap_arg_skips_the_gate_entirely(monkeypatch):
    """budget_cap=None (default) means the post-PASS-1 gate is simply not
    consulted — the run proceeds through the deep loop unconditionally (same
    pinned default-arg behavior as chunk-segment-tags' analogous test)."""
    pdef = registry.get_default("highlight-scan-analyze")

    async def _fake_scan_generate(*, model, contents, config):
        return SimpleNamespace(text=json.dumps({"highlights": [{"start_s": 0, "end_s": 10}]}))

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _fake_scan_generate
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: fake_client)

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        return json.dumps({"clips": []})

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)

    ctx = RunContext(
        youtube_id="x", youtube_url="https://youtube.com/watch?v=x",
        start_sec=0, end_sec=10, gemini_api_key="fake-key", request_timeout_ms=60_000,
    )
    planned = estimate_run_plan(pdef, duration_sec=10)
    events = [e async for e in run_pipeline(pdef, ctx, planned)]  # no budget_cap kwarg at all

    types_seen = [e["type"] for e in events]
    assert types_seen[-1] == "run_complete"


# =============================================================================== #
# Evaluator CHANGES-REQUIRED item 2 (2026-07-18) — the synthetic-seam
# preroll/postroll clamp fix, and a PINNED regression demonstrating the
# accepted (not fixed) genuinely-close-distinct-highlights fragmentation case.
# =============================================================================== #
@pytest.mark.asyncio
async def test_highlight_analyze_node_clamps_preroll_to_zero_at_synthetic_start_seam():
    """A highlight whose start_is_synthetic=True (the second sibling of a
    max-length split) must get ZERO preroll — its sent window must not cross
    back into the FIRST sibling's territory at all."""
    captured_windows = []

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_windows.append((start_sec, end_sec))
        return json.dumps({"clips": []})

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.highlights = [
        {"index": 1, "start_s": 0.0, "end_s": 35.0, "adjustment": None,
         "start_is_synthetic": False, "end_is_synthetic": True},
        {"index": 2, "start_s": 35.0, "end_s": 70.0, "adjustment": None,
         "start_is_synthetic": True, "end_is_synthetic": False},
    ]
    cfg = HighlightAnalyzeConfig(preroll_s=5.0, postroll_s=4.0).model_dump()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    # Taxonomy + actor calls both fire per highlight, sharing the same
    # window — dedupe to the per-highlight window sequence.
    unique_windows = list(dict.fromkeys(captured_windows))
    # Piece 1: genuine start (0, no preroll needed since already at scope start),
    # SYNTHETIC end (35) -> postroll clamped to 0, NOT 35+4=39.
    assert unique_windows[0] == (0.0, 35.0)
    # Piece 2: SYNTHETIC start (35) -> preroll clamped to 0, NOT 35-5=30;
    # genuine end (70) -> full postroll applies (70+4=74).
    assert unique_windows[1] == (35.0, 74.0)
    # The two sent windows share NO frames at all — the manufactured seam
    # duplicate the evaluator's repro found (piece1=[0,39] vs piece2=[30,70],
    # a 9s overlap) is structurally eliminated.
    assert unique_windows[0][1] <= unique_windows[1][0]


@pytest.mark.asyncio
async def test_highlight_analyze_node_synthetic_seam_fix_eliminates_evaluator_repro_overlap(monkeypatch):
    """Direct regression for the evaluator's exact repro (70s highlight split
    at 35s, preroll=5/postroll=4): BEFORE the fix, sent windows were
    piece1=[0,39] and piece2=[30,70] (9s of shared video content — the
    manufactured double-vision). AFTER the fix, sent windows must not overlap
    at all."""
    highlights = highlight_scan.finalize_highlights(
        [{"start_s": 0.0, "end_s": 70.0}], min_highlight_s=1.0, max_highlight_s=35.0,
    )
    assert len(highlights) == 2  # split into exactly 2 siblings at the 35s seam
    assert highlights[0]["end_is_synthetic"] is True
    assert highlights[1]["start_is_synthetic"] is True

    captured_windows = []

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_windows.append((start_sec, end_sec))
        return json.dumps({"clips": []})

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    ctx.highlights = highlights
    cfg = HighlightAnalyzeConfig(preroll_s=5.0, postroll_s=4.0).model_dump()

    monkeypatch.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())
    [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    # Taxonomy + actor calls both fire per highlight, sharing the same window — dedupe.
    unique_windows = list(dict.fromkeys(captured_windows))
    assert unique_windows == [(0.0, 35.0), (35.0, 74.0)]
    # No shared video content across the manufactured seam (contiguous, not
    # overlapping) — this is the concrete "eliminate the duplicate at the
    # source" claim; it is NOT a claim that Gemini can never report adjacent
    # truncated fragments of one action straddling 35s (see docstrings).
    assert unique_windows[0][1] == unique_windows[1][0]


@pytest.mark.asyncio
async def test_highlight_analyze_node_genuinely_close_highlights_fragmentation_is_accepted_not_fixed():
    """PINNED regression (evaluator CHANGES-REQUIRED item 2, CEO product call
    2026-07-18, case (c)): two GENUINELY DISTINCT PASS-1 highlights (neither
    is a synthetic split sibling of the other) sitting closer together than
    preroll_s + postroll_s DO get overlapping sent windows — this is real
    pre/post-roll context doing its intended job, ACCEPTED as a known
    playground-scope limitation (precision over recall, human confirms
    events), NOT built around. This test exists to PIN that behavior so a
    future change to it is a deliberate, visible decision, not a silent
    regression in either direction."""
    captured_windows = []

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kw):
        captured_windows.append((start_sec, end_sec))
        return json.dumps({"clips": []})

    ctx = RunContext(
        youtube_id="x", youtube_url="https://y", start_sec=0, end_sec=100,
        gemini_api_key="k", request_timeout_ms=1000,
    )
    # Two genuinely distinct PASS-1 highlights, 6s apart — closer than
    # preroll_s(5) + postroll_s(4) = 9s, and NEITHER carries a synthetic flag
    # (both came straight from PASS 1, not from this pipeline's own split).
    ctx.highlights = [
        {"index": 1, "start_s": 10.0, "end_s": 20.0, "adjustment": None,
         "start_is_synthetic": False, "end_is_synthetic": False},
        {"index": 2, "start_s": 26.0, "end_s": 40.0, "adjustment": None,
         "start_is_synthetic": False, "end_is_synthetic": False},
    ]
    cfg = HighlightAnalyzeConfig(preroll_s=5.0, postroll_s=4.0).model_dump()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _fake_analyze_chunk)
        mp.setattr(RunContext, "gemini_client", lambda self: MagicMock())
        [e async for e in executors.highlight_analyze_node(ctx, cfg)]

    # Highlight 1's postroll (20+4=24) and highlight 2's preroll (26-5=21)
    # OVERLAP in [21, 24] — genuine pre/post-roll context video shared by two
    # DISTINCT highlights. This is the accepted residual, pinned here.
    # Taxonomy + actor calls both fire per highlight, sharing the same window — dedupe.
    unique_windows = list(dict.fromkeys(captured_windows))
    assert unique_windows == [(5.0, 24.0), (21.0, 44.0)]
    assert unique_windows[0][1] > unique_windows[1][0]  # windows DO overlap — accepted, not a bug
