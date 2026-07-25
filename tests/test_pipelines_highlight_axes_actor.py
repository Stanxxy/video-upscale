"""S12 Phase 1b production wiring — the actor axis (item 3): schema/prompt
pure helpers in ``service/pipelines/highlight_axes.py``. Same leftover-$-
token regression discipline as ``tests/test_pipelines_highlight_v2.py``.
"""
from __future__ import annotations

from google.genai import types

from service.pipelines import highlight_axes
from shared_lib.models.simplified_taxonomy import AXIS2_ACTOR_SENTINELS


def test_actor_schema_enum_reflects_dynamic_player_choices_plus_sentinels():
    schema = highlight_axes.build_actor_schema(["player-a", "player-b"])
    assert schema.type == types.Type.OBJECT
    assert set(schema.required) == {"actor", "identity_uncertain", "justification"}
    assert list(schema.properties["actor"].enum) == [
        "player-a", "player-b", *AXIS2_ACTOR_SENTINELS,
    ]
    assert schema.properties["identity_uncertain"].type == types.Type.BOOLEAN


def test_actor_schema_with_no_player_choices_is_sentinel_only():
    """QA playground path (no player_references) — the enum degrades to
    sentinels only, never fabricates a player_id choice."""
    schema = highlight_axes.build_actor_schema([])
    assert list(schema.properties["actor"].enum) == list(AXIS2_ACTOR_SENTINELS)


def test_actor_schema_is_flat_not_a_list():
    schema = highlight_axes.build_actor_schema(["p1"])
    assert "clips" not in schema.properties
    assert "start_s" not in schema.properties
    assert "position" not in schema.properties
    assert "action_class" not in schema.properties


def test_actor_prompt_embeds_reference_labels():
    prompt = highlight_axes.build_actor_prompt(
        "a scramble to back control",
        [
            {"player_id": "p1", "player_name": "Alice"},
            {"player_id": "p2", "player_name": "Bob"},
        ],
    )
    assert "Alice" in prompt
    assert "p1" in prompt
    assert "Bob" in prompt
    assert "p2" in prompt
    assert "a scramble to back control" in prompt


def test_actor_prompt_with_no_references_states_none_available():
    prompt = highlight_axes.build_actor_prompt("some description", [])
    assert "No reference images are available" in prompt


def test_actor_prompt_falls_back_to_player_id_when_name_missing():
    prompt = highlight_axes.build_actor_prompt("d", [{"player_id": "p1", "player_name": None}])
    assert "p1" in prompt


def test_actor_prompt_leaves_no_leftover_tokens():
    prompt = highlight_axes.build_actor_prompt(
        "a body-movement description",
        [{"player_id": "p1", "player_name": "Alice"}],
    )
    assert "$" not in prompt


def test_actor_prompt_leaves_no_leftover_tokens_with_no_references():
    prompt = highlight_axes.build_actor_prompt(None, [])
    assert "$" not in prompt


def test_actor_system_prompt_never_instructs_bare_positional_guess():
    """Domain veto (mirrors simplified_tags.py's TIME_ACTOR_VALUES docstring
    for the OTHER direction): this call HAS reference images, so it must not
    be told to fall back to a bare top/bottom positional guess."""
    assert "top" not in highlight_axes.ACTOR_SYSTEM_PROMPT.lower().split()
