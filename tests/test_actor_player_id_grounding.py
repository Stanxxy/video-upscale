"""Stream 2 (Bug A): structured-output grounding of athlete identity.

The vision engine must ground candidate identity by emitting ``actor_player_id``
(a STRING enum of the video's player_ids) via Gemini structured output, NOT
free-text gi-color. These tests pin the contract end-to-end with a MOCKED Gemini
client (no real API call):

- ``_build_response_schema`` produces a valid ``genai.types.Schema`` with
  ``actor_player_id`` as a STRING enum of the player_ids, plus ``identity_uncertain``.
- The analyzer sets ``response_schema`` + ``response_mime_type='application/json'``.
- Schema-shaped JSON is accepted; an out-of-enum ``actor_player_id`` is REJECTED
  (raises) — no silent gi-color fallback.
- ``init_boxes`` generalizes to N bindings ({track_id: box}).
- ``clip_to_event`` resolves track_id/player_name from the binding; reasoning →
  role descriptor.
- The prompt labels refs by player name/player_id and contains NO gi-color
  identity instruction.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from google.genai import types

from analyzer import (
    BJJTechniqueAnalyzer,
    _build_response_schema,
    _player_ids_from_refs,
)
from service.sns import clip_to_event


PID_A = "11111111-1111-1111-1111-111111111111"
PID_B = "22222222-2222-2222-2222-222222222222"


# --------------------------------------------------------------------------- #
# _build_response_schema
# --------------------------------------------------------------------------- #
def test_build_response_schema_is_valid_genai_schema():
    schema = _build_response_schema([PID_A, PID_B])
    assert isinstance(schema, types.Schema)
    assert schema.type == types.Type.OBJECT
    assert "clips" in schema.properties


def test_actor_player_id_is_string_enum_of_player_ids():
    schema = _build_response_schema([PID_A, PID_B])
    clip_schema = schema.properties["clips"].items
    actor = clip_schema.properties["actor_player_id"]
    assert actor.type == types.Type.STRING
    assert list(actor.enum) == [PID_A, PID_B]


def test_schema_includes_identity_uncertain_and_reasoning():
    schema = _build_response_schema([PID_A, PID_B])
    clip_props = schema.properties["clips"].items.properties
    assert clip_props["identity_uncertain"].type == types.Type.BOOLEAN
    assert "reasoning" in clip_props
    assert "role" not in clip_props  # identity no longer carried by free-text role


def test_schema_without_player_ids_has_no_enum():
    schema = _build_response_schema([])
    actor = schema.properties["clips"].items.properties["actor_player_id"]
    assert actor.type == types.Type.STRING
    assert not actor.enum


def test_player_ids_from_refs_extracts_ids():
    refs = [
        {"player_id": PID_A, "player_name": "Stan", "image": object()},
        {"player_id": PID_B, "player_name": "Joe", "image": object()},
    ]
    assert _player_ids_from_refs(refs) == [PID_A, PID_B]
    # Legacy unlabelled refs (no player_id) → empty enum.
    assert _player_ids_from_refs([{"player_name": "Stan"}]) == []
    assert _player_ids_from_refs(None) == []


# --------------------------------------------------------------------------- #
# Structured-output contract — MOCKED Gemini client
# --------------------------------------------------------------------------- #
def _make_analyzer_with_mock(response_text: str):
    """Build a BJJTechniqueAnalyzer whose Gemini client is fully mocked."""
    analyzer = BJJTechniqueAnalyzer.__new__(BJJTechniqueAnalyzer)
    analyzer.model_id = "gemini-3.1-flash-lite"
    analyzer._request_timeout_ms = 600_000
    analyzer.taxonomy_path = "/nonexistent-taxonomy.md"

    mock_client = MagicMock()
    mock_response = SimpleNamespace(text=response_text)
    mock_client.models.generate_content.return_value = mock_response
    analyzer.client = mock_client
    return analyzer, mock_client


def _refs():
    return [
        {"track_id": 1, "player_id": PID_A, "player_name": "Stan", "image": object()},
        {"track_id": 2, "player_id": PID_B, "player_name": "Joe", "image": object()},
    ]


def test_sync_sets_response_schema_and_mime_type():
    valid = json.dumps({
        "current_context_summary": "x",
        "clips": [{
            "start_frame": 0, "end_frame": 5, "action": "guard_pass",
            "technique": "other", "actor_player_id": PID_A,
            "identity_uncertain": False, "reasoning": "white gi on top", "confidence": 0.7,
        }],
    })
    analyzer, mock_client = _make_analyzer_with_mock(valid)

    out = analyzer.analyze_sequence(
        ["<frame>"], [0], previous_context=None, player_references=_refs(),
    )

    # Structured output config asserted on the actual call.
    _, kwargs = mock_client.models.generate_content.call_args
    config = kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert isinstance(config.response_schema, types.Schema)
    actor = config.response_schema.properties["clips"].items.properties["actor_player_id"]
    assert list(actor.enum) == [PID_A, PID_B]

    # Parser accepts schema-shaped JSON and keeps the grounded player_id.
    parsed = json.loads(out)
    assert parsed["clips"][0]["actor_player_id"] == PID_A


@pytest.mark.asyncio
async def test_async_single_agent_sets_response_schema():
    valid = json.dumps({"clips": [{"actor_player_id": PID_A, "action": "mount"}]})
    analyzer = BJJTechniqueAnalyzer.__new__(BJJTechniqueAnalyzer)
    analyzer.model_id = "gemini-3.1-flash-lite"
    analyzer._request_timeout_ms = 600_000
    analyzer.taxonomy_path = "/nonexistent-taxonomy.md"

    captured = {}

    async def _fake_generate(*, model, contents, config):
        captured["config"] = config
        return SimpleNamespace(text=valid)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = _fake_generate
    analyzer.client = mock_client

    out = await analyzer.analyze_sequence_async(
        ["<frame>"], [0], previous_context=None, player_references=_refs(),
    )
    config = captured["config"]
    assert config.response_mime_type == "application/json"
    actor = config.response_schema.properties["clips"].items.properties["actor_player_id"]
    assert list(actor.enum) == [PID_A, PID_B]
    assert json.loads(out)["clips"][0]["actor_player_id"] == PID_A


def test_out_of_enum_actor_player_id_is_rejected_no_fallback():
    """Strict validator: an out-of-enum actor_player_id RAISES (no gi-color fallback)."""
    bad = json.dumps({
        "clips": [{
            "start_frame": 0, "end_frame": 5, "action": "guard_pass",
            "technique": "other", "actor_player_id": "athlete in black gi",
            "reasoning": "black gi", "confidence": 0.5,
        }],
    })
    analyzer, _ = _make_analyzer_with_mock(bad)

    with pytest.raises(ValueError, match="out-of-enum actor_player_id"):
        analyzer.analyze_sequence(
            ["<frame>"], [0], previous_context=None, player_references=_refs(),
        )


def test_in_enum_actor_player_id_passes_validation():
    good = json.dumps({"clips": [{"actor_player_id": PID_B, "action": "mount"}]})
    analyzer, _ = _make_analyzer_with_mock(good)
    out = analyzer.analyze_sequence(
        ["<frame>"], [0], previous_context=None, player_references=_refs(),
    )
    assert json.loads(out)["clips"][0]["actor_player_id"] == PID_B


# --------------------------------------------------------------------------- #
# Prompt: refs labelled by identity, NO gi-color identity instruction
# --------------------------------------------------------------------------- #
def test_prompt_labels_refs_by_player_name_and_id():
    analyzer = BJJTechniqueAnalyzer.__new__(BJJTechniqueAnalyzer)
    analyzer.taxonomy_path = "/nonexistent-taxonomy.md"

    _, contents = analyzer._build_contents_and_prompt(
        ["<frame>"], [0], None, _refs(),
    )
    joined = "\n".join(c for c in contents if isinstance(c, str))
    # References labelled by identity.
    assert f"Player Stan (player_id {PID_A}):" in joined
    assert f"Player Joe (player_id {PID_B}):" in joined
    # Output schema switched to grounded identity.
    assert "actor_player_id" in joined


def test_prompt_has_no_gi_color_identity_instruction():
    analyzer = BJJTechniqueAnalyzer.__new__(BJJTechniqueAnalyzer)
    analyzer.taxonomy_path = "/nonexistent-taxonomy.md"

    _, contents = analyzer._build_contents_and_prompt(
        ["<frame>"], [0], None, _refs(),
    )
    joined = "\n".join(c for c in contents if isinstance(c, str))
    # The old gi-color identity instruction is gone.
    assert "athlete in white gi" not in joined
    assert "athlete in blue gi" not in joined
    # gi color survives ONLY as a descriptor-in-reasoning note, never as identity.
    assert 'set `actor_player_id`' in joined or "actor_player_id to the player_id" in joined


# --------------------------------------------------------------------------- #
# init_boxes generalization (run_tracking)
# --------------------------------------------------------------------------- #
def _init_boxes(athlete_bindings, box_a=None, box_b=None):
    """Mirror of run_tracking.py init_boxes derivation."""
    return {
        b.track_id: b.box
        for b in (athlete_bindings or [])
        if getattr(b, "box", None)
    } or {1: box_a, 2: box_b}


def test_init_boxes_two_bindings():
    from service.models import AthleteBinding
    bindings = [
        AthleteBinding(track_id=1, player_id=PID_A, player_name="Stan", box=[0, 0, 10, 10]),
        AthleteBinding(track_id=2, player_id=PID_B, player_name="Joe", box=[20, 20, 30, 30]),
    ]
    assert _init_boxes(bindings) == {1: [0, 0, 10, 10], 2: [20, 20, 30, 30]}


def test_init_boxes_three_bindings():
    from service.models import AthleteBinding
    pid_c = "33333333-3333-3333-3333-333333333333"
    bindings = [
        AthleteBinding(track_id=1, player_id=PID_A, player_name="A", box=[0, 0, 1, 1]),
        AthleteBinding(track_id=2, player_id=PID_B, player_name="B", box=[2, 2, 3, 3]),
        AthleteBinding(track_id=3, player_id=pid_c, player_name="C", box=[4, 4, 5, 5]),
    ]
    assert _init_boxes(bindings) == {
        1: [0, 0, 1, 1], 2: [2, 2, 3, 3], 3: [4, 4, 5, 5],
    }


def test_init_boxes_legacy_fallback_when_no_bindings():
    assert _init_boxes(None, box_a=[1, 1, 2, 2], box_b=[3, 3, 4, 4]) == {
        1: [1, 1, 2, 2], 2: [3, 3, 4, 4],
    }


# --------------------------------------------------------------------------- #
# clip_to_event grounding
# --------------------------------------------------------------------------- #
def test_clip_to_event_resolves_track_id_and_name_from_binding():
    from service.models import AthleteBinding
    bindings = [
        AthleteBinding(track_id=1, player_id=PID_A, player_name="Stan", box=[0, 0, 1, 1]),
        AthleteBinding(track_id=2, player_id=PID_B, player_name="Joe", box=[2, 2, 3, 3]),
    ]
    clip = {
        "start_frame": 0, "end_frame": 30, "action": "mount", "technique": "other",
        "actor_player_id": PID_B, "identity_uncertain": False,
        "reasoning": "blue gi athlete on top secures mount", "confidence": 0.8,
    }
    event = clip_to_event(clip, uuid4(), fps=30.0, athlete_bindings=bindings)
    cand = event.event_candidates[0]

    assert cand.player_name == "Joe"
    grounded = cand._grounded_identity
    assert grounded["player_id"] == PID_B
    assert grounded["track_id"] == 2  # resolved from binding[player_id=B]
    assert grounded["identity_uncertain"] is False
    # gi-color reasoning becomes the role descriptor only — never identity.
    assert cand.role.startswith("blue gi athlete")


@pytest.mark.asyncio
async def test_multi_agent_judge_sets_schema_and_validates():
    """The multi-agent Judge must also set response_schema + enum and reject out-of-enum."""
    from analyzer import BJJMultiAgentAnalyzer

    analyzer = BJJMultiAgentAnalyzer.__new__(BJJMultiAgentAnalyzer)
    analyzer.model_id = "gemini-3.1-flash-lite"
    analyzer._request_timeout_s = 600.0
    analyzer.taxonomy_path = "/nonexistent-taxonomy.md"

    judge_json = json.dumps({"clips": [{"actor_player_id": PID_A, "action": "mount"}]})
    captured = {}

    async def _fake_generate(*, model, contents, config):
        # Specialists output free text; the Judge gets the structured config.
        if config.response_schema is not None:
            captured["config"] = config
            return SimpleNamespace(text=judge_json)
        return SimpleNamespace(text="specialist report")

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = _fake_generate
    analyzer.client = mock_client

    out = await analyzer.analyze_sequence_async(
        ["<frame>"], [0], previous_context=None, player_references=_refs(),
    )

    config = captured["config"]
    assert config.response_mime_type == "application/json"
    actor = config.response_schema.properties["clips"].items.properties["actor_player_id"]
    assert list(actor.enum) == [PID_A, PID_B]
    assert json.loads(out)["clips"][0]["actor_player_id"] == PID_A


@pytest.mark.asyncio
async def test_multi_agent_judge_rejects_out_of_enum():
    from analyzer import BJJMultiAgentAnalyzer

    analyzer = BJJMultiAgentAnalyzer.__new__(BJJMultiAgentAnalyzer)
    analyzer.model_id = "gemini-3.1-flash-lite"
    analyzer._request_timeout_s = 600.0
    analyzer.taxonomy_path = "/nonexistent-taxonomy.md"

    bad_json = json.dumps({"clips": [{"actor_player_id": "athlete in white gi"}]})

    async def _fake_generate(*, model, contents, config):
        if config.response_schema is not None:
            return SimpleNamespace(text=bad_json)
        return SimpleNamespace(text="specialist report")

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = _fake_generate
    analyzer.client = mock_client

    with pytest.raises(ValueError, match="out-of-enum actor_player_id"):
        await analyzer.analyze_sequence_async(
            ["<frame>"], [0], previous_context=None, player_references=_refs(),
        )


class _FakeSNSClient:
    def __init__(self):
        self.published = []

    def publish(self, *, TopicArn, Message, MessageAttributes):
        self.published.append({"Message": Message, "MessageAttributes": MessageAttributes})
        return {"MessageId": "fake"}


def test_publish_events_emits_grounded_identity_keys():
    """The emitted SNS candidate JSON must carry player_id/track_id/identity_uncertain
    so the backend listener / from_dynamodb_dict can persist grounded identity."""
    from service.models import AthleteBinding
    from service.sns import SNSPublisher

    pub = SNSPublisher.__new__(SNSPublisher)
    pub.client = _FakeSNSClient()
    pub.topic_arn = "arn:aws:sns:us-east-1:000000000000:bjj-events"

    bindings = [
        AthleteBinding(track_id=1, player_id=PID_A, player_name="Stan", box=[0, 0, 1, 1]),
        AthleteBinding(track_id=2, player_id=PID_B, player_name="Joe", box=[2, 2, 3, 3]),
    ]
    analysis = {"clips": [{
        "action": "mount", "technique": "other", "actor_player_id": PID_B,
        "identity_uncertain": True, "reasoning": "blue gi", "confidence": 0.6,
        "start_frame": 0, "end_frame": 10,
    }]}

    pub.publish_events(analysis, uuid4(), fps=30.0, athlete_bindings=bindings)

    # First published message is the event (second is analysis_complete boundary).
    event_msg = None
    for call in pub.client.published:
        if call["MessageAttributes"].get("event_type", {}).get("StringValue") == "bjj_event_detected":
            event_msg = json.loads(call["Message"])
            break
    assert event_msg is not None
    cand = event_msg["event_candidates"][0]
    assert cand["player_id"] == PID_B
    assert cand["track_id"] == 2
    assert cand["identity_uncertain"] is True
    assert cand["player_name"] == "Joe"
    assert cand["role"] == "blue gi"  # descriptor only


def test_clip_to_event_role_is_descriptor_not_identity():
    from service.models import AthleteBinding
    bindings = [
        AthleteBinding(track_id=1, player_id=PID_A, player_name="Stan", box=[0, 0, 1, 1]),
    ]
    clip = {"actor_player_id": PID_A, "reasoning": "white gi", "action": "guard_pass"}
    event = clip_to_event(clip, uuid4(), fps=30.0, athlete_bindings=bindings)
    cand = event.event_candidates[0]
    assert cand.player_name == "Stan"
    assert cand.role == "white gi"  # descriptor, not "Stan"
    assert cand._grounded_identity["player_id"] == PID_A
