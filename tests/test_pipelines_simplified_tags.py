"""``service/pipelines/simplified_tags.py`` — the QA-EXPERIMENT ``simplified-tags-v1``
4-axis Gemini tagging format: enum-constrained response schema, prompt content,
bundled default taxonomy text, and the direct-Gemini analyzer wrapper. Kept
entirely separate from ``analyzer.py``'s production ``clips-v1`` prompt/schema
(never imported/branched here).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types

from service.pipelines import simplified_tags


# --------------------------------------------------------------------------- #
# Frozen enum value sets (task contract)
# --------------------------------------------------------------------------- #
def test_position_values_frozen_set_of_9():
    assert simplified_tags.POSITION_VALUES == [
        "standing", "closed_guard", "open_guard", "half_guard", "side_control",
        "mount", "back_control", "turtle", "scramble",
    ]


def test_action_class_values_frozen_set_of_10_includes_all_escapes_and_submissions():
    assert simplified_tags.ACTION_CLASS_VALUES == [
        "takedown_attempt", "guard_pull", "sweep", "guard_pass", "back_take",
        "submission_choke", "submission_arm_lock", "submission_leg_lock",
        "escape", "transition",
    ]


def test_outcome_values_frozen_set_of_3():
    assert simplified_tags.OUTCOME_VALUES == ["successful", "unsuccessful", "unclear"]


# --------------------------------------------------------------------------- #
# Response schema — enum enforcement + free-text fields
# --------------------------------------------------------------------------- #
def test_schema_top_level_envelope_matches_clips_v1_shape():
    schema = simplified_tags.build_response_schema()
    assert schema.type == types.Type.OBJECT
    assert schema.required == ["clips"]
    assert set(schema.properties) == {"current_context_summary", "clips"}
    assert schema.properties["current_context_summary"].type == types.Type.STRING
    assert schema.properties["clips"].type == types.Type.ARRAY


def _tag_schema():
    return simplified_tags.build_response_schema().properties["clips"].items


def test_schema_tag_required_fields():
    tag = _tag_schema()
    assert set(tag.required) == {
        "start_frame", "end_frame", "position", "actor", "action_class", "outcome", "confidence",
    }
    # specific_technique_guess is optional (present as a property, absent from required)
    assert "specific_technique_guess" in tag.properties
    assert "specific_technique_guess" not in tag.required


def test_schema_position_is_string_enum_of_exact_9_values():
    tag = _tag_schema()
    position = tag.properties["position"]
    assert position.type == types.Type.STRING
    assert list(position.enum) == simplified_tags.POSITION_VALUES


def test_schema_action_class_is_string_enum_of_exact_10_values():
    tag = _tag_schema()
    action_class = tag.properties["action_class"]
    assert action_class.type == types.Type.STRING
    assert list(action_class.enum) == simplified_tags.ACTION_CLASS_VALUES


def test_schema_outcome_is_string_enum_of_exact_3_values():
    tag = _tag_schema()
    outcome = tag.properties["outcome"]
    assert outcome.type == types.Type.STRING
    assert list(outcome.enum) == simplified_tags.OUTCOME_VALUES


def test_schema_actor_is_free_text_not_an_enum():
    tag = _tag_schema()
    actor = tag.properties["actor"]
    assert actor.type == types.Type.STRING
    assert not actor.enum  # None or empty — never gated


def test_schema_specific_technique_guess_is_free_text_not_an_enum():
    tag = _tag_schema()
    guess = tag.properties["specific_technique_guess"]
    assert guess.type == types.Type.STRING
    assert not guess.enum


def test_schema_confidence_is_number():
    tag = _tag_schema()
    assert tag.properties["confidence"].type == types.Type.NUMBER


# --------------------------------------------------------------------------- #
# Prompt content — demands exact enums, absolute frame numbers
# --------------------------------------------------------------------------- #
def test_prompt_demands_absolute_frame_numbers():
    prompt = simplified_tags.build_prompt([100, 115, 130], previous_context=None)
    assert "[100, 115, 130]" in prompt
    assert "ABSOLUTE" in prompt


def test_prompt_names_all_four_fields_and_demands_exact_enum_values():
    prompt = simplified_tags.build_prompt([0, 1], previous_context="ctx")
    for field in ("position", "actor", "action_class", "outcome", "specific_technique_guess", "confidence"):
        assert field in prompt
    assert "EXACTLY one" in prompt or "EXACTLY the" in prompt


def test_prompt_forwards_previous_context():
    prompt = simplified_tags.build_prompt([0], previous_context="Blue gi passed to mount.")
    assert "Blue gi passed to mount." in prompt


def test_prompt_defaults_context_to_start_of_match_when_none():
    prompt = simplified_tags.build_prompt([0], previous_context=None)
    assert "Start of the match." in prompt


# --------------------------------------------------------------------------- #
# Bundled default taxonomy text
# --------------------------------------------------------------------------- #
def test_default_taxonomy_text_loaded_from_bundled_file_and_mentions_all_axes():
    text = simplified_tags.DEFAULT_TAXONOMY_TEXT
    assert "Simplified taxonomy file not found." not in text  # file must actually be present/readable
    for axis_value in ("closed_guard", "submission_arm_lock", "unsuccessful", "specific_technique_guess"):
        assert axis_value in text


def test_resolve_system_instruction_none_returns_bundled_default():
    assert simplified_tags.resolve_system_instruction(None) == simplified_tags.DEFAULT_TAXONOMY_TEXT


def test_resolve_system_instruction_whitespace_only_returns_bundled_default():
    assert simplified_tags.resolve_system_instruction("   \n  ") == simplified_tags.DEFAULT_TAXONOMY_TEXT


def test_resolve_system_instruction_override_replaces_default():
    assert simplified_tags.resolve_system_instruction("  CUSTOM TEXT  ") == "CUSTOM TEXT"


# --------------------------------------------------------------------------- #
# SimplifiedTagsAnalyzer — direct Gemini call, mirrors BJJTechniqueAnalyzer's
# call signature but never touches analyzer.py.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_analyzer_no_frames_returns_no_frames_marker():
    analyzer = simplified_tags.SimplifiedTagsAnalyzer(MagicMock(), model_id="gemini-3.1-flash-lite")
    result = await analyzer.analyze_sequence_async([], [], None)
    assert result == "No frames."


@pytest.mark.asyncio
async def test_analyzer_sends_prompt_and_schema_and_uses_shared_client(monkeypatch):
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return SimpleNamespace(text=json.dumps({"current_context_summary": "s", "clips": []}))

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate

    analyzer = simplified_tags.SimplifiedTagsAnalyzer(
        client, model_id="gemini-3.1-flash-lite", thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    result = await analyzer.analyze_sequence_async(["frame1", "frame2"], [10, 20], "prev ctx")

    assert captured["model"] == "gemini-3.1-flash-lite"
    assert captured["contents"][0].startswith("\n") or "10, 20" in captured["contents"][0]
    assert captured["contents"][1:] == ["frame1", "frame2"]
    assert captured["config"].response_schema is not None
    assert captured["config"].system_instruction == simplified_tags.DEFAULT_TAXONOMY_TEXT
    assert captured["config"].thinking_config.thinking_budget == 0
    assert json.loads(result) == {"current_context_summary": "s", "clips": []}


@pytest.mark.asyncio
async def test_analyzer_honors_system_instruction_override():
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["config"] = config
        return SimpleNamespace(text="{}")

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate

    analyzer = simplified_tags.SimplifiedTagsAnalyzer(
        client, model_id="gemini-3.1-flash-lite", system_instruction="MY CUSTOM TAXONOMY",
    )
    await analyzer.analyze_sequence_async(["f"], [0], None)
    assert captured["config"].system_instruction == "MY CUSTOM TAXONOMY"


@pytest.mark.asyncio
async def test_analyzer_strips_markdown_fences():
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text='```json\n{"clips": []}\n```')

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate
    analyzer = simplified_tags.SimplifiedTagsAnalyzer(client, model_id="gemini-3.1-flash-lite")
    result = await analyzer.analyze_sequence_async(["f"], [0], None)
    assert json.loads(result) == {"clips": []}


@pytest.mark.asyncio
async def test_analyzer_surfaces_gemini_error_as_json_error_never_fabricates():
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("simulated Gemini outage"))
    analyzer = simplified_tags.SimplifiedTagsAnalyzer(client, model_id="gemini-3.1-flash-lite")
    result = await analyzer.analyze_sequence_async(["f"], [0], None)
    parsed = json.loads(result)
    assert "simulated Gemini outage" in parsed["error"]


# =============================================================================== #
# simplified-tags-time-v1 — NATIVE-VIDEO sibling (chunk-segment-tags PASS 2).
# Distinct from the frame-keyed format above: start_s/end_s (seconds), and an
# INVERTED actor axis (enum-constrained top/bottom/contested, never a name).
# =============================================================================== #
def _time_tag_schema():
    return simplified_tags.build_video_response_schema().properties["clips"].items


def test_time_actor_values_frozen_set_of_3():
    assert simplified_tags.TIME_ACTOR_VALUES == ["top", "bottom", "contested"]


def test_video_schema_top_level_envelope_matches_time_v1_shape():
    schema = simplified_tags.build_video_response_schema()
    assert schema.type == types.Type.OBJECT
    assert schema.required == ["clips"]
    assert set(schema.properties) == {"current_context_summary", "clips"}


def test_video_schema_tag_required_fields_are_seconds_not_frames():
    tag = _time_tag_schema()
    assert set(tag.required) == {
        "start_s", "end_s", "position", "actor", "action_class", "outcome", "confidence",
    }
    assert "start_frame" not in tag.properties
    assert "end_frame" not in tag.properties
    assert tag.properties["start_s"].type == types.Type.NUMBER
    assert tag.properties["end_s"].type == types.Type.NUMBER


def test_video_schema_actor_is_a_hard_enum_never_free_text():
    """Gracie domain veto: no player reference images on this path, so the
    schema itself (not just the prompt) forbids a name/player_id."""
    tag = _time_tag_schema()
    actor = tag.properties["actor"]
    assert actor.type == types.Type.STRING
    assert list(actor.enum) == simplified_tags.TIME_ACTOR_VALUES


def test_video_schema_actor_gi_color_is_optional_free_text():
    tag = _time_tag_schema()
    assert "actor_gi_color" in tag.properties
    assert "actor_gi_color" not in tag.required
    assert not tag.properties["actor_gi_color"].enum


def test_video_schema_position_action_class_outcome_enums_match_frame_based_format():
    """Same 4-axis taxonomy as simplified-tags-v1 — only the time unit and
    actor axis differ."""
    tag = _time_tag_schema()
    assert list(tag.properties["position"].enum) == simplified_tags.POSITION_VALUES
    assert list(tag.properties["action_class"].enum) == simplified_tags.ACTION_CLASS_VALUES
    assert list(tag.properties["outcome"].enum) == simplified_tags.OUTCOME_VALUES


def test_video_prompt_demands_chunk_relative_seconds_not_absolute():
    prompt = simplified_tags.build_video_prompt(previous_context=None)
    assert "RELATIVE TO THIS CLIP" in prompt
    assert "0:00" in prompt


def test_video_prompt_inverts_actor_instruction_forbids_names():
    prompt = simplified_tags.build_video_prompt(previous_context=None)
    assert "top" in prompt and "bottom" in prompt and "contested" in prompt
    assert "NO player reference images" in prompt
    assert "NEVER output a name" in prompt


def test_video_prompt_instructs_not_to_suppress_overlap_events():
    prompt = simplified_tags.build_video_prompt(previous_context=None)
    assert "Do NOT suppress or skip an event" in prompt or "do not suppress" in prompt.lower()


def test_video_prompt_forwards_previous_context():
    prompt = simplified_tags.build_video_prompt(previous_context="Blue on top in mount.")
    assert "Blue on top in mount." in prompt


@pytest.mark.asyncio
async def test_time_analyzer_sends_native_video_part_with_offsets(monkeypatch):
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return SimpleNamespace(text=json.dumps({"current_context_summary": "s", "clips": []}))

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate

    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(client, model_id="gemini-3.1-flash-lite")
    result = await analyzer.analyze_chunk("https://youtube.com/watch?v=x", 12.0, 42.0, "prev ctx")

    assert captured["model"] == "gemini-3.1-flash-lite"
    content = captured["contents"][0]
    video_part = content.parts[1]
    assert video_part.file_data.file_uri == "https://youtube.com/watch?v=x"
    assert video_part.video_metadata.start_offset == "12.0s"
    assert video_part.video_metadata.end_offset == "42.0s"
    assert captured["config"].response_schema is not None
    assert json.loads(result) == {"current_context_summary": "s", "clips": []}


@pytest.mark.asyncio
async def test_time_analyzer_strips_markdown_fences():
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(text='```json\n{"clips": []}\n```')

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(client, model_id="gemini-3.1-flash-lite")
    result = await analyzer.analyze_chunk("https://y", 0.0, 10.0, None)
    assert json.loads(result) == {"clips": []}


@pytest.mark.asyncio
async def test_time_analyzer_surfaces_gemini_error_as_json_error_never_fabricates():
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("simulated Gemini outage"))
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(client, model_id="gemini-3.1-flash-lite")
    result = await analyzer.analyze_chunk("https://y", 0.0, 10.0, None)
    parsed = json.loads(result)
    assert "simulated Gemini outage" in parsed["error"]
