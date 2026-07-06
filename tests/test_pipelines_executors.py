"""``service/pipelines/executors.py`` — budget guard estimation, dedup ``key_by``
fix (Brooks R6), return_format schema selection, and an end-to-end mocked run
of the ``vision-mimic`` pipeline through the streaming run loop.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from service.pipelines import executors, frame_source, registry, simplified_tags
from service.pipelines.executors import (
    DEFAULT_BUDGET_CAP,
    RunContext,
    _apply_dedup_bucket_key,
    _apply_simplified_tags_dedup_keys,
    _convert_chunk_relative_to_absolute,
    _response_schema_for,
    estimate_run_plan,
    run_pipeline,
)
from service.pipelines.models import ChunkAnalyzeConfig, ChunkSegmentConfig, DedupConfig


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
# End-to-end mocked run — vision-mimic pipeline through run_pipeline()
# --------------------------------------------------------------------------- #
def _tiny_image():
    return Image.new("RGB", (100, 100), color="gray")


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

    async def _fake_run_detect(client, image, model=None, thinking_config=None, prompt=None):
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

    async def _fake_run_detect(client, image, model=None, thinking_config=None, prompt=None):
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
    assert captured["video_part"].video_metadata.start_offset == "25.0s"
    assert captured["video_part"].video_metadata.end_offset == "60s"

    result_events = [e for e in events if e["type"] == "chunk_result"]
    assert len(result_events) == 1
    clip = result_events[0]["clips"][0]
    # chunk-relative [2.0, 6.0] + effective_start(25) -> absolute [27.0, 31.0]
    assert clip["start_s"] == 27.0
    assert clip["end_s"] == 31.0
    assert ctx.final_format == "simplified-tags-time-v1"
    assert ctx.native_fps is None


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
    assert ctx.final_clips == []
    assert ctx.final_format == "simplified-tags-time-v1"


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
