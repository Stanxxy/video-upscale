"""S12 Phase 1b — service/taxonomy_mapper.py::build_axis_only_candidate
(item 13, design §5.2/§5.5).

The installed shared_lib (1.2.0 as of this stream) still declares
``VideoEventCandidate.action``/``.confidence`` as REQUIRED, non-``None``
fields — the axis-only mapping LOGIC is unit-tested here via a duck-typed
``candidate_cls`` seam (never by fudging the ``None`` legacy values to make
the OLD schema accept them). The tests that require the REAL relaxed model
to actually construct are marked skip, pending the shared_lib 1.3.0 PR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest
from pydantic import ValidationError

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
    action: Optional[str] = None
    technique: Optional[str] = None
    result: Optional[str] = None
    confidence: Optional[float] = None
    notes: str = ""
    schema_version: int = 1
    axis1_position: Optional[List[str]] = None
    axis3_action: Optional[List[str]] = None
    axis4_outcome: Optional[str] = None
    actor_sentinel: Optional[str] = None
    technique_shortlist: Optional[str] = None
    technique_guess: Optional[str] = None


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
def test_legacy_fields_are_none_never_fabricated():
    cand = taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1", candidate_cls=_DuckCandidate)
    assert cand.action is None
    assert cand.technique is None
    assert cand.result is None
    assert cand.confidence is None
    assert cand.technique_shortlist is None
    assert cand.technique_guess is None


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
    assert cand.track_id is None  # always None for v2 — no tracking exists


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


def test_never_calls_dual_emit_legacy_fields(monkeypatch):
    """Regression guard for the module docstring's contract: the axis-only
    path must NEVER touch the dual-emit legacy tables."""
    def _boom(*a, **kw):
        raise AssertionError("build_axis_only_candidate must never call dual_emit_legacy_fields")

    monkeypatch.setattr(taxonomy_mapper, "dual_emit_legacy_fields", _boom)
    taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1", candidate_cls=_DuckCandidate)  # must not raise


# --------------------------------------------------------------------------- #
# Real shared_lib construction — BLOCKED on the 1.3.0 relaxation (§8.1).
# --------------------------------------------------------------------------- #
def test_real_shared_lib_1_2_0_rejects_none_action_and_confidence():
    """Proves the hard sequencing constraint is REAL, not silently bypassed:
    the installed shared_lib (1.2.0) still requires action/confidence, so
    calling this function with the REAL VideoEventCandidate class raises."""
    with pytest.raises(ValidationError):
        taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1")


@pytest.mark.skip(reason="requires shared_lib>=1.3.0 axis-only relaxation, PR pending")
def test_real_shared_lib_construction_succeeds_after_relaxation():
    """Once shared_lib ships the relaxed VideoEventCandidate (action/
    technique/result/confidence -> Optional, schema_version==3 documented
    legal), this must construct successfully with the default (real) class."""
    cand = taxonomy_mapper.build_axis_only_candidate(_clip(), "vid-1")
    assert cand.schema_version == 3
    assert cand.action is None
    assert cand.confidence is None
