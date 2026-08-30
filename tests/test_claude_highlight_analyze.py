"""Claude Highlight Analyze allowlist + routing unit tests (no live network)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from service.pipelines import claude_frames, executors
from service.pipelines.models import HighlightAnalyzeConfig
from service.pipelines.registry import (
    HIGHLIGHT_ANALYZE_MODELS,
    PipelineValidationError,
    get_default,
    validate_pipeline_def,
)
from service.routes.qa_vlm import EVENT_MODELS


def test_highlight_analyze_models_include_claude_opus():
    assert "claude-opus-4-8" in HIGHLIGHT_ANALYZE_MODELS
    for m in EVENT_MODELS:
        assert m in HIGHLIGHT_ANALYZE_MODELS


def test_highlight_analyze_accepts_claude_top_level_and_axes():
    pdef = get_default("highlight-scan-critique-analyze")
    ha = next(s for s in pdef.stages if s.type == "highlight_analyze")
    ha.config["model"] = "claude-opus-4-8"
    for axis in ("position", "technique", "validator"):
        ha.config[axis]["model"] = "claude-opus-4-8"
    validate_pipeline_def(pdef)  # must not raise


def test_highlight_scan_rejects_claude():
    pdef = get_default("highlight-scan-critique-analyze")
    next(s for s in pdef.stages if s.type == "highlight_scan").config["model"] = "claude-opus-4-8"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_highlight_critique_rejects_claude():
    pdef = get_default("highlight-scan-critique-analyze")
    next(s for s in pdef.stages if s.type == "highlight_critique").config["model"] = "claude-opus-4-8"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_chunk_analyze_rejects_claude():
    pdef = get_default("chunk-segment-tags")
    next(s for s in pdef.stages if s.type == "chunk_analyze").config["model"] = "claude-opus-4-8"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_subsample_evenly_caps_at_20_and_keeps_endpoints():
    items = list(range(100))
    kept = claude_frames.subsample_evenly(items, max_n=20)
    assert len(kept) <= 20
    assert kept[0] == 0
    assert kept[-1] == 99
    assert len(kept) == len(set(kept))


def test_subsample_evenly_noop_when_under_cap():
    items = list(range(5))
    assert claude_frames.subsample_evenly(items, max_n=20) == items


def test_opus_thinking_kwargs_off_omits_thinking():
    assert claude_frames.opus_thinking_kwargs("off") == {}
    assert claude_frames.opus_thinking_kwargs(None) == {}


def test_opus_thinking_kwargs_adaptive_effort():
    kw = claude_frames.opus_thinking_kwargs("high")
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"]["effort"] == "high"


def test_effective_axis_model_cascades_top_level_claude():
    cfg = HighlightAnalyzeConfig(model="claude-opus-4-8")
    assert executors._effective_highlight_axis_model(cfg, "gemini-3.1-flash-lite") == "claude-opus-4-8"
    cfg_g = HighlightAnalyzeConfig(model="gemini-3.5-flash")
    assert executors._effective_highlight_axis_model(cfg_g, "gemini-3.1-flash-lite") == "gemini-3.1-flash-lite"


@pytest.mark.asyncio
async def test_call_axis_json_routes_claude_without_gemini_analyzer(monkeypatch):
    """Claude model_id must hit claude_frames, not require a Gemini analyzer."""
    called = {}

    async def fake_analyze(ctx, **kwargs):
        called["kwargs"] = kwargs
        return {"position": "mount", "justification": "hips high"}, (
            '{"position":"mount","justification":"hips high"}'
        )

    monkeypatch.setattr(claude_frames, "analyze_axis_json", fake_analyze)
    ctx = executors.RunContext(
        youtube_id="x", youtube_url="https://youtu.be/x",
        start_sec=0, end_sec=60, gemini_api_key="g", request_timeout_ms=1000,
        anthropic_api_key="test-key",
    )
    telem = SimpleNamespace(
        last_prompt_text=None, last_raw_response_text=None, last_usage_metadata=None,
    )
    data, err = await executors._call_axis_json(
        telem, ctx.youtube_url, 1.0, 5.0,
        response_schema=None, prompt_text="prompt", fps=10, media_resolution=None,
        ctx=ctx, model_id="claude-opus-4-8",
        system_instruction="sys", thinking="off",
    )
    assert err is None
    assert data["position"] == "mount"
    assert called["kwargs"]["model_id"] == "claude-opus-4-8"
    assert called["kwargs"]["fps"] == 10
    assert telem.last_prompt_text == "prompt"
    assert telem.last_raw_response_text == '{"position":"mount","justification":"hips high"}'


@pytest.mark.asyncio
async def test_analyze_axis_json_mocked_anthropic_no_network(monkeypatch):
    from PIL import Image

    from service.pipelines.frame_source import SampledFrame

    fake_frames = [
        SampledFrame(frame_index=i, image=Image.new("RGB", (8, 8), color=(i % 255, 0, 0)))
        for i in range(25)
    ]
    monkeypatch.setattr(claude_frames.frame_source, "sample_frames", lambda *_a, **_k: fake_frames)
    monkeypatch.setattr(claude_frames, "_ensure_stream", lambda ctx: None)
    monkeypatch.setattr(claude_frames, "_jpeg_b64", lambda _img: "aaa")

    mock_resp = SimpleNamespace(
        content=[SimpleNamespace(text='{"position":"mount","justification":"ok"}')]
    )
    mock_create = AsyncMock(return_value=mock_resp)
    mock_client = MagicMock()
    mock_client.messages.create = mock_create

    import sys
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AsyncAnthropic=lambda api_key=None: mock_client),
    )

    ctx = executors.RunContext(
        youtube_id="x", youtube_url="https://youtu.be/x",
        start_sec=0, end_sec=60, gemini_api_key="g", request_timeout_ms=1000,
        anthropic_api_key="test-key",
        stream_url="http://stream", stream_headers={}, native_fps=30.0,
    )
    data, raw = await claude_frames.analyze_axis_json(
        ctx,
        model_id="claude-opus-4-8",
        window_start=1.0,
        window_end=5.0,
        prompt_text='Return {"position":"mount","justification":"ok"}',
        system_instruction="sys",
        fps=10,
        thinking="off",
    )
    assert data["position"] == "mount"
    assert '"position"' in raw
    assert mock_create.await_count == 1
    call_kwargs = mock_create.await_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-8"
    # 20 images (subsampled from 25) + 1 text block
    assert len(call_kwargs["messages"][0]["content"]) == 21
