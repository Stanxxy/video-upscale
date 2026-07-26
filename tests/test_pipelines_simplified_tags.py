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
from google.genai import errors as genai_errors
from google.genai import types

from service.pipelines import gemini_retry, simplified_tags


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retry tests below use the real gemini_retry.call_with_retry backoff
    loop with real transient google.genai.errors — mock asyncio.sleep so
    nothing in this file ever actually sleeps."""
    monkeypatch.setattr("service.pipelines.gemini_retry.asyncio.sleep", AsyncMock())


def _server_error(code: int = 503) -> genai_errors.ServerError:
    return genai_errors.ServerError(code, {"message": f"{code} error", "status": "UNAVAILABLE"}, None)


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


@pytest.mark.asyncio
async def test_analyzer_retries_transient_503_then_succeeds():
    call_count = {"n": 0}

    async def _fake_generate(*, model, contents, config):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise _server_error(503)
        return SimpleNamespace(text=json.dumps({"clips": []}))

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate
    analyzer = simplified_tags.SimplifiedTagsAnalyzer(
        client, model_id="gemini-3.1-flash-lite",
        retry_config=gemini_retry.GeminiRetryConfig(max_attempts=3, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0),
    )
    result = await analyzer.analyze_sequence_async(["f"], [0], None)
    assert json.loads(result) == {"clips": []}
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_analyzer_exhausted_transient_retries_surfaces_json_error_never_fabricates():
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=_server_error(503))
    analyzer = simplified_tags.SimplifiedTagsAnalyzer(
        client, model_id="gemini-3.1-flash-lite",
        retry_config=gemini_retry.GeminiRetryConfig(max_attempts=3, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0),
    )
    result = await analyzer.analyze_sequence_async(["f"], [0], None)
    parsed = json.loads(result)
    assert "503" in parsed["error"]
    assert client.aio.models.generate_content.call_count == 3


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
    assert video_part.video_metadata.start_offset == "12.000s"
    assert video_part.video_metadata.end_offset == "42.000s"
    assert captured["config"].response_schema is not None
    assert json.loads(result) == {"current_context_summary": "s", "clips": []}


# --------------------------------------------------------------------------- #
# `_format_offset_seconds` — live-bug regression (2026-07-18). Gemini rejected
# a real production request with `400 INVALID_ARGUMENT ... Field 'start_offset'
# ... Invalid duration format, failed to parse nano seconds` because protobuf
# `Duration` accepts AT MOST 9 fractional digits, and `highlight_analyze_node`'s
# preroll subtraction (`33.3 - 5.0`) produces a float with 15 fractional digits
# when naively formatted (`f"{x}s"`). These tests pin the formatter's own
# digit-count invariant against the exact adversarial floats from the repro.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw_seconds",
    [
        33.3 - 5.0,  # == 28.299999999999997 (real repro: preroll subtraction)
        2 / 3,  # == 0.6666666666666666
        0.1 + 0.2,  # == 0.30000000000000004
        599.9999999999999,  # large value, rounds up across an integer boundary
        35.0,
        0,
        12.5,
    ],
)
def test_format_offset_seconds_never_exceeds_9_fractional_digits(raw_seconds):
    formatted = simplified_tags._format_offset_seconds(raw_seconds)

    assert formatted.endswith("s")
    assert "e" not in formatted.lower()  # never scientific notation
    body = formatted[:-1]  # strip trailing "s"
    if "." in body:
        frac_digits = body.split(".")[1]
        assert len(frac_digits) <= 9, (
            f"{formatted!r} has {len(frac_digits)} fractional digits — "
            "protobuf Duration accepts at most 9 (nanoseconds)"
        )
    # Sanity: the formatted value round-trips to (approximately) the input,
    # i.e. this is a precision fix, not a silent value-mangling one.
    assert abs(float(body) - float(raw_seconds)) < 0.01


def test_format_offset_seconds_adversarial_floats_produce_expected_strings():
    """Pins the EXACT output for the repro's specific floats (millisecond/
    3-fractional-digit precision, never stripped) — guards against a future
    "simplification" reintroducing the raw-float bug."""
    assert simplified_tags._format_offset_seconds(33.3 - 5.0) == "28.300s"
    assert simplified_tags._format_offset_seconds(2 / 3) == "0.667s"
    assert simplified_tags._format_offset_seconds(0.1 + 0.2) == "0.300s"
    assert simplified_tags._format_offset_seconds(599.9999999999999) == "600.000s"
    assert simplified_tags._format_offset_seconds(35.0) == "35.000s"
    assert simplified_tags._format_offset_seconds(0) == "0.000s"
    assert simplified_tags._format_offset_seconds(12.5) == "12.500s"


# --------------------------------------------------------------------------- #
# `mime_type`/`extra_parts` kwargs (S12 Phase 1b production wiring, item 4) —
# additive, `None`/empty preserves every existing caller byte-identical
# (INS-107 single-formatter discipline, mirrors `fps`/`media_resolution`).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_analyze_chunk_mime_type_none_omits_field_from_file_data(monkeypatch):
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["contents"] = contents
        return SimpleNamespace(text=json.dumps({"clips": []}))

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(client, model_id="gemini-3.1-flash-lite")

    await analyzer.analyze_chunk("https://y", 0.0, 10.0, None)

    video_part = captured["contents"][0].parts[1]
    assert video_part.file_data.mime_type is None


@pytest.mark.asyncio
async def test_analyze_chunk_mime_type_lands_in_file_data_when_set(monkeypatch):
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["contents"] = contents
        return SimpleNamespace(text=json.dumps({"clips": []}))

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(client, model_id="gemini-3.1-flash-lite")

    await analyzer.analyze_chunk(
        "files/abc123", 0.0, 10.0, None, mime_type="video/mp4",
    )

    video_part = captured["contents"][0].parts[1]
    assert video_part.file_data.mime_type == "video/mp4"
    assert video_part.file_data.file_uri == "files/abc123"


@pytest.mark.asyncio
async def test_analyze_chunk_extra_parts_none_leaves_contents_as_text_and_video_only(monkeypatch):
    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["contents"] = contents
        return SimpleNamespace(text=json.dumps({"clips": []}))

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(client, model_id="gemini-3.1-flash-lite")

    await analyzer.analyze_chunk("https://y", 0.0, 10.0, None)

    parts = captured["contents"][0].parts
    assert len(parts) == 2  # text, video — no extra_parts appended


@pytest.mark.asyncio
async def test_analyze_chunk_extra_parts_appended_after_video_part(monkeypatch):
    from google.genai import types as genai_types

    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["contents"] = contents
        return SimpleNamespace(text=json.dumps({"clips": []}))

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(client, model_id="gemini-3.1-flash-lite")

    ref_part = genai_types.Part(inline_data=genai_types.Blob(data=b"fake-jpeg-bytes", mime_type="image/jpeg"))
    await analyzer.analyze_chunk("https://y", 0.0, 10.0, None, extra_parts=[ref_part])

    parts = captured["contents"][0].parts
    assert len(parts) == 3
    assert parts[2] is ref_part


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


@pytest.mark.asyncio
async def test_time_analyzer_retries_transient_429_then_succeeds():
    call_count = {"n": 0}

    async def _fake_generate(*, model, contents, config):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise _server_error(429)
        return SimpleNamespace(text=json.dumps({"clips": []}))

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
        client, model_id="gemini-3.1-flash-lite",
        retry_config=gemini_retry.GeminiRetryConfig(max_attempts=5, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0),
    )
    result = await analyzer.analyze_chunk("https://y", 0.0, 10.0, None)
    assert json.loads(result) == {"clips": []}
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_time_analyzer_non_transient_404_fails_fast_no_retry():
    client = MagicMock()
    not_found = genai_errors.ClientError(404, {"message": "model not found", "status": "NOT_FOUND"}, None)
    client.aio.models.generate_content = AsyncMock(side_effect=not_found)
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
        client, model_id="gemini-3.1-flash-lite",
        retry_config=gemini_retry.GeminiRetryConfig(max_attempts=5, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0),
    )
    result = await analyzer.analyze_chunk("https://y", 0.0, 10.0, None)
    parsed = json.loads(result)
    assert "model not found" in parsed["error"]
    assert client.aio.models.generate_content.call_count == 1  # non-transient — no retry attempted


# --------------------------------------------------------------------------- #
# Additive instrumentation (ADCC eval dashboard, Task 1) — new attributes only,
# analyze_chunk's return contract (a plain str) is unchanged.
# --------------------------------------------------------------------------- #
def test_extract_usage_metadata_reads_real_fields():
    response = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=100, candidates_token_count=20, total_token_count=120,
    ))
    assert simplified_tags.extract_usage_metadata(response) == {
        "prompt_token_count": 100, "candidates_token_count": 20, "total_token_count": 120,
    }


def test_extract_usage_metadata_none_when_absent():
    assert simplified_tags.extract_usage_metadata(SimpleNamespace(text="x")) is None


@pytest.mark.asyncio
async def test_time_analyzer_records_instrumentation_slots_on_success():
    async def _fake_generate(*, model, contents, config):
        return SimpleNamespace(
            text=json.dumps({"clips": []}),
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=2, total_token_count=12),
        )

    client = MagicMock()
    client.aio.models.generate_content = _fake_generate
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(client, model_id="gemini-3.1-flash-lite")
    assert analyzer.last_prompt_text is None  # unset before any call

    result = await analyzer.analyze_chunk("https://y", 5.0, 15.0, "prev")
    assert json.loads(result) == {"clips": []}
    assert "NATIVE video clip" in analyzer.last_prompt_text
    assert analyzer.last_raw_response_text == json.dumps({"clips": []})
    assert analyzer.last_usage_metadata == {
        "prompt_token_count": 10, "candidates_token_count": 2, "total_token_count": 12,
    }


@pytest.mark.asyncio
async def test_time_analyzer_instrumentation_prompt_set_but_response_none_on_failure():
    """A failed call still records the prompt that was sent (it was built and
    sent before the exception), but raw_response_text/usage_metadata stay
    None — never fabricated from a call that never got a response."""
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("simulated outage"))
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(client, model_id="gemini-3.1-flash-lite")

    result = await analyzer.analyze_chunk("https://y", 0.0, 10.0, None)
    assert "simulated outage" in json.loads(result)["error"]
    assert analyzer.last_prompt_text is not None
    assert analyzer.last_raw_response_text is None
    assert analyzer.last_usage_metadata is None
