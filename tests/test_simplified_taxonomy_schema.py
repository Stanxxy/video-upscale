"""``analyzer._build_response_schema`` emits the ENUM-constrained
axis1_position/axis3_action/axis4_outcome shape, with no technique list or
guess field, while ``actor_player_id``/``identity_uncertain``/``reasoning``
stay as before. Coverage for the actor_player_id enum machinery itself lives
in ``tests/test_actor_player_id_grounding.py``.
"""
from __future__ import annotations

from google.genai import types

from analyzer import _build_response_schema
from service import taxonomy_mapper as tm

PID_A = "11111111-1111-1111-1111-111111111111"
PID_B = "22222222-2222-2222-2222-222222222222"


# --------------------------------------------------------------------------- #
# analyzer._build_response_schema — new axis fields (item 2)
# --------------------------------------------------------------------------- #

def test_schema_no_longer_has_free_string_action_technique():
    schema = _build_response_schema([PID_A, PID_B])
    clip_props = schema.properties["clips"].items.properties
    assert "action" not in clip_props
    assert "technique" not in clip_props
    assert "specific_technique" not in clip_props


def test_schema_axis1_position_is_array_of_string_enum():
    schema = _build_response_schema([PID_A, PID_B])
    clip_props = schema.properties["clips"].items.properties
    axis1 = clip_props["axis1_position"]
    assert axis1.type == types.Type.ARRAY
    assert axis1.items.type == types.Type.STRING
    assert set(axis1.items.enum) == set(tm.AXIS1_POSITION)


def test_schema_axis3_action_is_array_of_string_enum():
    schema = _build_response_schema([PID_A, PID_B])
    clip_props = schema.properties["clips"].items.properties
    axis3 = clip_props["axis3_action"]
    assert axis3.type == types.Type.ARRAY
    assert axis3.items.type == types.Type.STRING
    assert set(axis3.items.enum) == set(tm.AXIS3_ACTION)


def test_schema_axis4_outcome_is_scalar_string_enum():
    schema = _build_response_schema([PID_A, PID_B])
    clip_props = schema.properties["clips"].items.properties
    axis4 = clip_props["axis4_outcome"]
    assert axis4.type == types.Type.STRING
    assert set(axis4.enum) == set(tm.AXIS4_OUTCOME)


def test_schema_has_no_technique_guess_or_shortlist_field():
    schema = _build_response_schema([PID_A, PID_B])
    clip_props = schema.properties["clips"].items.properties
    assert "technique_guess" not in clip_props
    assert "technique_shortlist" not in clip_props


def test_schema_required_fields_use_new_axes():
    schema = _build_response_schema([PID_A, PID_B])
    clip_schema = schema.properties["clips"].items
    assert set(clip_schema.required) == {
        "start_frame", "end_frame", "axis1_position", "axis3_action",
        "axis4_outcome", "actor_player_id",
    }


def test_schema_actor_player_id_machinery_unchanged():
    """D5: actor_player_id enum machinery must be byte-for-byte identical to
    pre-T2 behavior — this pins the exact same assertions as the pre-existing
    test_actor_player_id_grounding.py::test_actor_player_id_is_string_enum_of_player_ids."""
    schema = _build_response_schema([PID_A, PID_B])
    clip_props = schema.properties["clips"].items.properties
    actor = clip_props["actor_player_id"]
    assert actor.type == types.Type.STRING
    assert list(actor.enum) == [PID_A, PID_B]
    assert clip_props["identity_uncertain"].type == types.Type.BOOLEAN
    assert "reasoning" in clip_props


def test_schema_without_player_ids_actor_still_unconstrained_string():
    schema = _build_response_schema([])
    actor = schema.properties["clips"].items.properties["actor_player_id"]
    assert actor.type == types.Type.STRING
    assert not actor.enum
