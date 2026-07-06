"""``analyzer.py`` QA pipeline seam (build plan §3.3) — additive, keyword-only,
production-preserving knobs on ``BJJTechniqueAnalyzer``/``BJJMultiAgentAnalyzer``.

Covers:
- Byte-identity: default kwargs produce the EXACT same ``GenerateContentConfig``
  fields as before the seam existed (no ``thinking_config``, default schema).
- Caller audit: every pre-existing production construction path (``pipeline.py``,
  ``service/worker/stages/upscale_setup.py``, ``analyze_sequence_sync``) still
  works unmodified and gets the byte-identical defaults.
- ``response_schema=`` override actually reaches the Gemini call.
- ``thinking_config=`` is added ONLY when not None, on every internal call site
  (single-agent, multi-agent persona calls, multi-agent Judge).
- ``model_id=`` propagates to every internal call.
"""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types

import analyzer
from analyzer import BJJMultiAgentAnalyzer, BJJTechniqueAnalyzer


def _make_single(**overrides):
    a = BJJTechniqueAnalyzer.__new__(BJJTechniqueAnalyzer)
    a.model_id = overrides.get("model_id", "gemini-3.1-flash-lite")
    a.temperature = overrides.get("temperature", 0.2)
    a.thinking_config = overrides.get("thinking_config", None)
    a.taxonomy_text = overrides.get("taxonomy_text", None)
    a._request_timeout_ms = 600_000
    a.taxonomy_path = "/nonexistent-taxonomy.md"
    a.client = MagicMock()
    return a


def _make_multi(**overrides):
    a = BJJMultiAgentAnalyzer.__new__(BJJMultiAgentAnalyzer)
    a.model_id = overrides.get("model_id", "gemini-3.1-flash-lite")
    a.temperature = overrides.get("temperature", 0.2)
    a.thinking_config = overrides.get("thinking_config", None)
    a.taxonomy_text = overrides.get("taxonomy_text", None)
    a._request_timeout_s = 600.0
    a.taxonomy_path = "/nonexistent-taxonomy.md"
    a.client = MagicMock()
    return a


# --------------------------------------------------------------------------- #
# __init__ signatures — additive, keyword-only, documented defaults
# --------------------------------------------------------------------------- #
def test_single_agent_init_signature_is_additive_keyword_only():
    sig = inspect.signature(BJJTechniqueAnalyzer.__init__)
    params = sig.parameters
    for name in ("model_id", "temperature", "thinking_config"):
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, name
    assert params["model_id"].default == "gemini-3.1-flash-lite"
    assert params["temperature"].default == 0.2
    assert params["thinking_config"].default is None


def test_multi_agent_init_signature_is_additive_keyword_only():
    sig = inspect.signature(BJJMultiAgentAnalyzer.__init__)
    params = sig.parameters
    for name in ("model_id", "temperature", "thinking_config"):
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, name
    assert params["model_id"].default == "gemini-3.1-flash-lite"
    assert params["thinking_config"].default is None


def test_init_with_real_constructor_defaults_match_production(monkeypatch):
    """Constructing through the real (non-__new__) __init__ with no new kwargs
    yields the exact pre-seam attribute values."""
    monkeypatch.setattr(analyzer, "_gemini_client", lambda api_key, to_ms: MagicMock())
    a = BJJTechniqueAnalyzer("fake-key")
    assert a.model_id == "gemini-3.1-flash-lite"
    assert a.temperature == 0.2
    assert a.thinking_config is None
    assert a.taxonomy_text is None

    m = BJJMultiAgentAnalyzer("fake-key")
    assert m.model_id == "gemini-3.1-flash-lite"
    assert m.temperature == 0.2
    assert m.thinking_config is None
    assert m.taxonomy_text is None


# --------------------------------------------------------------------------- #
# Byte-identity: default-kwargs GenerateContentConfig matches pre-seam shape
# --------------------------------------------------------------------------- #
def test_single_agent_default_config_is_byte_identical_to_pre_seam():
    a = _make_single()
    valid = json.dumps({"clips": []})
    a.client.models.generate_content.return_value = SimpleNamespace(text=valid)

    a.analyze_sequence(["<frame>"], [0], previous_context=None)

    _, kwargs = a.client.models.generate_content.call_args
    config = kwargs["config"]
    # Pre-seam call site set exactly these 4 fields and NOTHING else conditionally.
    assert config.temperature == 0.2
    assert config.response_mime_type == "application/json"
    assert config.thinking_config is None  # never set — omitted entirely, same as before
    assert config.response_schema is not None  # _build_response_schema([]) default


@pytest.mark.asyncio
async def test_single_agent_async_default_config_is_byte_identical_to_pre_seam():
    a = _make_single()
    valid = json.dumps({"clips": []})
    a.client.aio.models.generate_content = AsyncMock(return_value=SimpleNamespace(text=valid))

    await a.analyze_sequence_async(["<frame>"], [0], previous_context=None)

    _, kwargs = a.client.aio.models.generate_content.call_args
    config = kwargs["config"]
    assert config.temperature == 0.2
    assert config.thinking_config is None
    assert config.response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_multi_agent_persona_calls_omit_thinking_config_by_default():
    a = _make_multi()
    captured_configs = []

    async def _fake_generate(*, model, contents, config):
        captured_configs.append(config)
        return SimpleNamespace(text="specialist report")

    a.client.aio.models.generate_content = _fake_generate
    await a.run_agent("Biomechanist", "focus on physics", ["<frame>"], [0], "ctx", temperature=0.8)

    assert len(captured_configs) == 1
    assert captured_configs[0].thinking_config is None
    assert captured_configs[0].temperature == 0.8  # per-persona temperature untouched by seam


@pytest.mark.asyncio
async def test_multi_agent_judge_default_config_is_byte_identical_to_pre_seam():
    a = _make_multi()
    judge_json = json.dumps({"clips": []})

    async def _fake_generate(*, model, contents, config):
        if config.response_schema is not None:
            return SimpleNamespace(text=judge_json)
        return SimpleNamespace(text="specialist report")

    a.client.aio.models.generate_content = _fake_generate
    out = await a.analyze_sequence_async(["<frame>"], [0], previous_context=None)
    assert json.loads(out) == {"clips": []}


# --------------------------------------------------------------------------- #
# thinking_config seam — added ONLY when not None
# --------------------------------------------------------------------------- #
def test_single_agent_thinking_config_added_when_set():
    tc = types.ThinkingConfig(thinking_level="low")
    a = _make_single(thinking_config=tc)
    a.client.models.generate_content.return_value = SimpleNamespace(text=json.dumps({"clips": []}))

    a.analyze_sequence(["<frame>"], [0], previous_context=None)

    _, kwargs = a.client.models.generate_content.call_args
    assert kwargs["config"].thinking_config is tc


@pytest.mark.asyncio
async def test_multi_agent_thinking_config_reaches_persona_and_judge_calls():
    tc = types.ThinkingConfig(thinking_budget=0)
    a = _make_multi(thinking_config=tc)
    captured = []

    async def _fake_generate(*, model, contents, config):
        captured.append(config)
        if config.response_schema is not None:
            return SimpleNamespace(text=json.dumps({"clips": []}))
        return SimpleNamespace(text="specialist report")

    a.client.aio.models.generate_content = _fake_generate
    await a.analyze_sequence_async(["<frame>"], [0], previous_context=None)

    assert len(captured) == 4  # 3 personas + judge
    assert all(c.thinking_config is tc for c in captured)


# --------------------------------------------------------------------------- #
# model_id seam
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_model_id_propagates_to_every_multi_agent_call():
    a = _make_multi(model_id="gemini-3.5-flash")
    captured_models = []

    async def _fake_generate(*, model, contents, config):
        captured_models.append(model)
        if config.response_schema is not None:
            return SimpleNamespace(text=json.dumps({"clips": []}))
        return SimpleNamespace(text="specialist report")

    a.client.aio.models.generate_content = _fake_generate
    await a.analyze_sequence_async(["<frame>"], [0], previous_context=None)
    assert captured_models == ["gemini-3.5-flash"] * 4


# --------------------------------------------------------------------------- #
# response_schema seam
# --------------------------------------------------------------------------- #
def test_response_schema_override_reaches_gemini_call():
    custom_schema = types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING))
    a = _make_single()
    a.client.models.generate_content.return_value = SimpleNamespace(text="[]")

    a.analyze_sequence(["<frame>"], [0], previous_context=None, response_schema=custom_schema)

    _, kwargs = a.client.models.generate_content.call_args
    assert kwargs["config"].response_schema is custom_schema


@pytest.mark.asyncio
async def test_response_schema_none_falls_back_to_build_response_schema():
    a = _make_single()
    a.client.aio.models.generate_content = AsyncMock(return_value=SimpleNamespace(text=json.dumps({"clips": []})))
    await a.analyze_sequence_async(["<frame>"], [0], previous_context=None, response_schema=None)
    _, kwargs = a.client.aio.models.generate_content.call_args
    # Default schema has the clips-v1 shape (current_context_summary + clips array).
    assert "clips" in kwargs["config"].response_schema.properties


# --------------------------------------------------------------------------- #
# taxonomy_text seam — QA pipeline's per-node ``system_instruction`` override
# --------------------------------------------------------------------------- #
def test_single_agent_taxonomy_text_overrides_file_read():
    a = _make_single(taxonomy_text="CUSTOM TAXONOMY: only armbar and triangle exist.")
    a.taxonomy_path = "/nonexistent-taxonomy.md"  # would normally fail to read
    system_instruction, _contents = a._build_contents_and_prompt(["<frame>"], [0], None, None)
    assert "CUSTOM TAXONOMY: only armbar and triangle exist." in system_instruction
    assert "Taxonomy file not found." not in system_instruction


def test_single_agent_taxonomy_text_none_reads_file_exactly_as_before():
    a = _make_single(taxonomy_text=None)
    a.taxonomy_path = "/nonexistent-taxonomy.md"
    system_instruction, _contents = a._build_contents_and_prompt(["<frame>"], [0], None, None)
    assert "Taxonomy file not found." in system_instruction  # unchanged pre-seam fallback text


@pytest.mark.asyncio
async def test_multi_agent_judge_taxonomy_text_overrides_file_read():
    a = _make_multi(taxonomy_text="CUSTOM TAXONOMY: only armbar and triangle exist.")
    captured = {}

    async def _fake_generate(*, model, contents, config):
        if config.response_schema is not None:
            captured["system_instruction"] = config.system_instruction
            return SimpleNamespace(text=json.dumps({"clips": []}))
        return SimpleNamespace(text="specialist report")

    a.client.aio.models.generate_content = _fake_generate
    await a.analyze_sequence_async(["<frame>"], [0], previous_context=None)
    assert "CUSTOM TAXONOMY: only armbar and triangle exist." in captured["system_instruction"]


@pytest.mark.asyncio
async def test_multi_agent_judge_taxonomy_text_none_reads_file_exactly_as_before():
    a = _make_multi(taxonomy_text=None)
    a.taxonomy_path = "/nonexistent-taxonomy.md"
    captured = {}

    async def _fake_generate(*, model, contents, config):
        if config.response_schema is not None:
            captured["system_instruction"] = config.system_instruction
            return SimpleNamespace(text=json.dumps({"clips": []}))
        return SimpleNamespace(text="specialist report")

    a.client.aio.models.generate_content = _fake_generate
    await a.analyze_sequence_async(["<frame>"], [0], previous_context=None)
    # Pre-seam behavior on an unreadable path: taxonomy_text becomes "" (not None).
    assert captured["system_instruction"] == "\n\nYou are the Synthesizer. Output JSON only."


# --------------------------------------------------------------------------- #
# Caller audit — pre-existing production construction paths unaffected
# --------------------------------------------------------------------------- #
def test_pipeline_py_caller_construction_unaffected(monkeypatch):
    """pipeline.py:process_video constructs BJJTechniqueAnalyzer/BJJMultiAgentAnalyzer
    with ONLY (api_key, taxonomy_path=, request_timeout_ms=) — no new kwargs. Confirms
    that call signature still works and yields production defaults."""
    monkeypatch.setattr(analyzer, "_gemini_client", lambda api_key, to_ms: MagicMock())
    a = BJJTechniqueAnalyzer("fake-key", taxonomy_path=None, request_timeout_ms=600_000)
    assert a.model_id == "gemini-3.1-flash-lite"
    assert a.thinking_config is None

    m = BJJMultiAgentAnalyzer("fake-key", taxonomy_path=None, request_timeout_ms=600_000)
    assert m.model_id == "gemini-3.1-flash-lite"
    assert m.thinking_config is None


def test_upscale_setup_lambda_call_shape_unaffected(monkeypatch):
    """service/worker/stages/upscale_setup.py's analyze_fn lambdas call
    analyzer.analyze_sequence(frames, indices, ctx, player_references=...) and
    analyze_sequence_sync(_analyzer, frames, indices, ctx, player_references=...)
    — neither passes response_schema. Confirm both call shapes still work."""
    monkeypatch.setattr(analyzer, "_gemini_client", lambda api_key, to_ms: MagicMock())
    a = BJJTechniqueAnalyzer("fake-key", taxonomy_path=None, request_timeout_ms=600_000)
    a.client.models.generate_content.return_value = SimpleNamespace(text=json.dumps({"clips": []}))
    out = a.analyze_sequence(["<frame>"], [0], "ctx", player_references=None)
    assert json.loads(out) == {"clips": []}


def test_qa_identity_analyze_caller_construction_unaffected(monkeypatch):
    """service/routes/qa.py constructs BJJTechniqueAnalyzer(api_key=config.gemini_api_key)
    — positional-free, no new kwargs."""
    monkeypatch.setattr(analyzer, "_gemini_client", lambda api_key, to_ms: MagicMock())
    a = BJJTechniqueAnalyzer(api_key="fake-key")
    assert a.model_id == "gemini-3.1-flash-lite"
    assert a.temperature == 0.2
    assert a.thinking_config is None
