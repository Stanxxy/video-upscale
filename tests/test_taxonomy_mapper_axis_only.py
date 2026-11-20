"""service/taxonomy_mapper.py::build_axis_only_candidate.

The mapping LOGIC is unit-tested via a duck-typed ``candidate_cls`` seam,
which keeps these tests independent of shared_lib's installed version; the
real class is exercised in ``test_real_shared_lib_construction_succeeds``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from service import taxonomy_mapper


@dataclass
class _DuckCandidate:
    """A permissive stand-in for shared_lib's VideoEventCandidate — accepts
    every kwarg build_axis_only_candidate passes, with NO schema enforcement
    (no required-field/enum validation at all). Used to test the MAPPING
    logic (which clip field lands in which kwarg) independently of
    shared_lib's installed version."""

    role: str
    player_name: Optional[str] = None
    player_id: Optional[str] = None
    track_id: Optional[int] = None
    identity_uncertain: Optional[bool] = None
    notes: str = ""
    schema_version: int = 1
    axis1_position: Optional[List[str]] = None
    axis3_action: Optional[List[str]] = None
    axis4_outcome: Optional[str] = None
    actor_sentinel: Optional[str] = None


def _clip(**overrides) -> dict:
    base = {
        "start_s": 10.0, "end_s": 20.0,
        "position": "mount", "action_class": "submission_arm_lock", "outcome": "successful",
        "player_id": "p1", "player_name": "Alice",
        "identity_uncertain": False, "actor_sentinel": None,
        "notes": "arm trapped | tap confirmed | evidence text",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Mapping logic — always testable against ANY schema (duck-typed injection).
# --------------------------------------------------------------------------- #
def test_passes_only_axis_and_actor_kwargs():
    """``_DuckCandidate`` declares no legacy or technique-list field, so any
    such kwarg raises ``TypeError``."""
    cand = taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1", candidate_cls=_DuckCandidate)
    assert cand.axis3_action == ["submission_arm_lock"]


def test_schema_version_is_three():
    cand = taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1", candidate_cls=_DuckCandidate)
    assert cand.schema_version == 3


def test_axis1_and_axis3_are_wrapped_in_single_item_lists():
    cand = taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1", candidate_cls=_DuckCandidate)
    assert cand.axis1_position == ["mount"]
    assert cand.axis3_action == ["submission_arm_lock"]


def test_axis4_outcome_is_scalar_direct_assignment():
    cand = taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1", candidate_cls=_DuckCandidate)
    assert cand.axis4_outcome == "successful"


def test_missing_outcome_sanitizes_to_unclear_not_none():
    cand = taxonomy_mapper.build_axis_only_candidate(
        _clip(outcome=None), "vid-1", candidate_cls=_DuckCandidate,
    )
    assert cand.axis4_outcome == "unclear"  # D4: never null/blank


def test_real_player_id_maps_to_grounded_identity_fields():
    cand = taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1", candidate_cls=_DuckCandidate)
    assert cand.player_id == "p1"
    assert cand.player_name == "Alice"
    assert cand.identity_uncertain is False
    assert cand.actor_sentinel is None
    assert cand.role == "Alice"
    assert cand.track_id is None


def test_sentinel_actor_maps_to_sentinel_fields_and_null_identity():
    cand = taxonomy_mapper.build_axis_only_candidate(
        _clip(player_id=None, player_name=None, identity_uncertain=True, actor_sentinel="contested"),
        "vid-1", candidate_cls=_DuckCandidate,
    )
    assert cand.player_id is None
    assert cand.player_name is None
    assert cand.actor_sentinel == "contested"
    assert cand.identity_uncertain is True
    assert cand.role == "contested"


def test_notes_carries_the_clip_notes_text_verbatim():
    cand = taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1", candidate_cls=_DuckCandidate)
    assert cand.notes == "arm trapped | tap confirmed | evidence text"


def test_missing_notes_defaults_to_empty_string_not_none():
    cand = taxonomy_mapper.build_axis_only_candidate(
        _clip(notes=None), "vid-1", candidate_cls=_DuckCandidate,
    )
    assert cand.notes == ""


def test_no_grounded_identity_at_all_falls_back_to_unknown_role():
    cand = taxonomy_mapper.build_axis_only_candidate(
        _clip(player_id=None, player_name=None, actor_sentinel=None),
        "vid-1", candidate_cls=_DuckCandidate,
    )
    assert cand.role == "Unknown"


# --------------------------------------------------------------------------- #
# Real shared_lib construction.
# --------------------------------------------------------------------------- #
def test_real_shared_lib_construction_succeeds():
    """The REAL (default, non-duck-typed) VideoEventCandidate class
    constructs successfully from this function's kwargs."""
    cand = taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1")
    assert cand.schema_version == 3
    assert cand.axis3_action == ["submission_arm_lock"]


# --------------------------------------------------------------------------- #
# 2026-07-26 single-call cutover (AC2) — pinned regression against the REAL
# clip dict shape executors.highlight_analyze_node now emits (ONE taxonomy
# call producing position/action_class/outcome + the unchanged actor axis
# resolving player_id/player_name/identity_uncertain/actor_sentinel), not
# the deleted two-call+validator design's shape. build_axis_only_candidate's
# own mapping logic is unchanged (call-count-agnostic — see its docstring);
# this test proves that claim against the shape that's ACTUALLY produced now.
# --------------------------------------------------------------------------- #
def test_maps_the_single_call_executor_output_shape():
    single_call_clip = {
        "start_s": 15.0, "end_s": 32.0,
        "position": "side_control", "action_class": "submission_arm_lock", "outcome": "successful",
        "player_id": "p1", "player_name": "Alice",
        "identity_uncertain": False, "actor_sentinel": None,
        # Notes format post-cutover: "<taxonomy justification> | <actor justification>"
        # — two segments, not the old three (position | technique | validator evidence).
        "notes": "arm trapped, hips elevated | blue gi matches reference image 1",
    }
    cand = taxonomy_mapper.build_axis_only_candidate(single_call_clip, "vid-1", candidate_cls=_DuckCandidate)
    assert cand.schema_version == 3
    assert cand.axis1_position == ["side_control"]
    assert cand.axis3_action == ["submission_arm_lock"]
    assert cand.axis4_outcome == "successful"
    assert cand.player_id == "p1"
    assert cand.player_name == "Alice"
    assert cand.identity_uncertain is False
    assert cand.actor_sentinel is None
    assert cand.notes == "arm trapped, hips elevated | blue gi matches reference image 1"
