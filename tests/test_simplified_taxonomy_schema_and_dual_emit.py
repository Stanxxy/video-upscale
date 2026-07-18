"""T2 (simplified-4-axis-taxonomy adoption, 2026-07-12) — production seam
tests for items 2 and 4:

- ``analyzer._build_response_schema`` emits the ENUM-constrained
  axis1_position/axis3_action/axis4_outcome/technique_guess shape (D5),
  while ``actor_player_id``/``identity_uncertain``/``reasoning`` stay
  EXACTLY as before (unchanged machinery, D5).
- ``service.sns.clip_to_event`` dual-emits: schema_version=2 + populated new
  axis fields, AND legacy action/technique/result derived via
  ``taxonomy_mapper.dual_emit_legacy_fields`` — never independently read off
  the clip dict.

Existing coverage for the actor_player_id enum machinery itself
(``tests/test_actor_player_id_grounding.py``) is untouched and still green;
this file covers what's NEW.
"""
from __future__ import annotations

from uuid import uuid4

from google.genai import types

from analyzer import _build_response_schema
from service.sns import clip_to_event
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


def test_schema_technique_guess_is_unconstrained_free_string():
    schema = _build_response_schema([PID_A, PID_B])
    clip_props = schema.properties["clips"].items.properties
    guess = clip_props["technique_guess"]
    assert guess.type == types.Type.STRING
    assert not guess.enum  # deliberately unconstrained per D1


def test_schema_required_fields_use_new_axes():
    schema = _build_response_schema([PID_A, PID_B])
    clip_schema = schema.properties["clips"].items
    assert set(clip_schema.required) == {
        "start_frame", "end_frame", "axis1_position", "axis3_action",
        "axis4_outcome", "actor_player_id",
    }
    # technique_guess is explicitly NOT required (D1: optional, ungated).
    assert "technique_guess" not in clip_schema.required


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


# --------------------------------------------------------------------------- #
# service.sns.clip_to_event — dual emit (item 4)
# --------------------------------------------------------------------------- #

def _new_shape_clip(**overrides) -> dict:
    base = {
        "start_frame": 0, "end_frame": 30,
        "axis1_position": ["closed_guard"],
        "axis3_action": ["submission_arm_lock"],
        "axis4_outcome": "successful",
        "technique_guess": "armbar",
        "actor_player_id": PID_A,
        "identity_uncertain": False,
        "reasoning": "white gi athlete on top isolates the arm",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


def test_clip_to_event_sets_schema_version_2():
    event = clip_to_event(_new_shape_clip(), uuid4(), fps=30.0)
    cand = event.event_candidates[0]
    assert cand.schema_version == 2


def test_clip_to_event_populates_new_axis_fields():
    event = clip_to_event(_new_shape_clip(), uuid4(), fps=30.0)
    cand = event.event_candidates[0]
    assert cand.axis1_position == ["closed_guard"]
    assert cand.axis3_action == ["submission_arm_lock"]
    assert cand.axis4_outcome == "successful"
    assert cand.technique_guess == "armbar"
    assert cand.technique_shortlist == "armbar"


def test_clip_to_event_dual_emits_legacy_fields_from_new_axes():
    event = clip_to_event(_new_shape_clip(), uuid4(), fps=30.0)
    cand = event.event_candidates[0]
    # Derived via taxonomy_mapper, not read off the (nonexistent) clip["action"].
    assert cand.action == "submission_attempt"
    assert cand.technique == "armbar"
    assert cand.result == "success"


def test_clip_to_event_legacy_values_always_pass_old_enums():
    event = clip_to_event(_new_shape_clip(
        axis3_action=["submission_leg_lock"], axis4_outcome="unsuccessful",
        technique_guess="heel hook",
    ), uuid4(), fps=30.0)
    cand = event.event_candidates[0]
    assert cand.action in tm.VALID_ACTIONS
    assert cand.technique in tm.VALID_TECHNIQUES
    assert cand.result in tm.VALID_RESULTS


def test_clip_to_event_handles_missing_axis_fields_gracefully():
    """A clip dict with none of the new axis keys (e.g. a hand-authored test
    fixture, or a malformed upstream response) must not crash — sanitize_*
    fallbacks apply, dual-emit still yields a legal legacy triple."""
    clip = {
        "start_frame": 0, "end_frame": 5, "actor_player_id": PID_A,
        "reasoning": "test", "confidence": 0.5,
    }
    event = clip_to_event(clip, uuid4(), fps=30.0)
    cand = event.event_candidates[0]
    assert cand.schema_version == 2
    assert cand.axis1_position == []
    assert cand.axis3_action == []
    assert cand.axis4_outcome == "unclear"
    assert cand.action == "other"
    assert cand.technique == "other"
    assert cand.result == "neutral"


def test_clip_to_event_multi_label_axis3_dual_emits_via_precedence():
    event = clip_to_event(_new_shape_clip(
        axis3_action=["sweep", "submission_choke"], technique_guess="rnc",
    ), uuid4(), fps=30.0)
    cand = event.event_candidates[0]
    assert cand.axis3_action == ["sweep", "submission_choke"]
    assert cand.action == "submission_attempt"  # submission precedence wins
    assert cand.technique == "rear_naked_choke"


def test_clip_to_event_grounding_unaffected_by_dual_emit(caplog):
    """Sanity: the pre-existing identity-grounding behavior (player_name/
    track_id resolution, role-as-descriptor) is untouched by the dual-emit
    seam — same assertions as test_actor_player_id_grounding.py but on a
    new-shape clip."""
    from service.models import AthleteBinding

    bindings = [AthleteBinding(track_id=2, player_id=PID_A, player_name="Stan", box=[0, 0, 1, 1])]
    event = clip_to_event(_new_shape_clip(), uuid4(), fps=30.0, athlete_bindings=bindings)
    cand = event.event_candidates[0]
    assert cand.player_name == "Stan"
    assert cand._grounded_identity["player_id"] == PID_A
    assert cand._grounded_identity["track_id"] == 2
    assert cand.role.startswith("white gi athlete")


def test_clip_to_event_logs_transition_for_d6_monitoring(caplog):
    with caplog.at_level("INFO"):
        clip_to_event(_new_shape_clip(axis3_action=["transition"]), uuid4(), fps=30.0)
    assert "taxonomy_monitor" in caplog.text
    assert "transition" in caplog.text


def test_clip_to_event_logs_scramble_for_d6_monitoring(caplog):
    with caplog.at_level("INFO"):
        clip_to_event(_new_shape_clip(axis1_position=["scramble"]), uuid4(), fps=30.0)
    assert "taxonomy_monitor" in caplog.text
    assert "scramble" in caplog.text
