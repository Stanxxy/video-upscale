"""Tests for the re-authored ``service/taxonomy_mapper.py`` (T2, simplified
4-axis taxonomy adoption, 2026-07-12).

Replaces the old ``map_to_frontend_taxonomy``-era test suite (dead code,
deleted alongside the pipeline-category heuristics it exercised — see the
module docstring in ``service/taxonomy_mapper.py``). Covers:

- Manifest-driven axis vocabulary loaded correctly.
- Sanitization fail-loud-with-explicit-sentinel behavior (item 5).
- ``technique_guess`` -> curated shortlist resolution, incl. aliases (item 3).
- New -> legacy dual-emit mapping: EVERY possible new-taxonomy emission
  produces legacy action/technique/result values that pass the OLD enums
  (item 4 — the search-422 seam, plan §7 risk #2).
"""
from __future__ import annotations

import itertools

import pytest

from service import taxonomy_mapper as tm


# ---------------------------------------------------------------------------
# Manifest-driven vocabulary
# ---------------------------------------------------------------------------

class TestManifestVocabulary:
    def test_axis1_position_has_nine_values(self):
        assert len(tm.AXIS1_POSITION) == 9
        assert "scramble" in tm.AXIS1_POSITION

    def test_axis3_action_has_ten_values(self):
        assert len(tm.AXIS3_ACTION) == 10
        assert "transition" in tm.AXIS3_ACTION

    def test_axis4_outcome_has_three_values(self):
        assert set(tm.AXIS4_OUTCOME) == {"successful", "unsuccessful", "unclear"}

    def test_every_axis3_class_has_a_shortlist(self):
        assert set(tm.TECHNIQUE_SHORTLISTS) == set(tm.AXIS3_ACTION)
        for cls, shortlist in tm.TECHNIQUE_SHORTLISTS.items():
            assert "other" in shortlist, f"{cls} shortlist missing 'other'"

    def test_is_valid_axis1_axis3_axis4(self):
        assert tm.is_valid_axis1(["standing", "mount"])
        assert not tm.is_valid_axis1(["bogus"])
        assert tm.is_valid_axis3(["sweep"])
        assert not tm.is_valid_axis3(["bogus"])
        assert tm.is_valid_axis4("successful")
        assert not tm.is_valid_axis4("bogus")

    def test_is_valid_technique_for_class(self):
        assert tm.is_valid_technique_for_class("submission_arm_lock", "armbar")
        assert not tm.is_valid_technique_for_class("submission_arm_lock", "heel_hook")
        # Unknown class -> False, never raises.
        assert not tm.is_valid_technique_for_class("bogus_class", "armbar")


# ---------------------------------------------------------------------------
# Sanitization (item 5)
# ---------------------------------------------------------------------------

class TestSanitizeAxis1Position:
    def test_valid_values_pass_through(self):
        assert tm.sanitize_axis1_position(["standing", "mount"]) == ["standing", "mount"]

    def test_none_becomes_empty_list(self):
        assert tm.sanitize_axis1_position(None) == []

    def test_invalid_values_are_dropped_not_raised(self, caplog):
        with caplog.at_level("WARNING"):
            result = tm.sanitize_axis1_position(["standing", "bogus_position"])
        assert result == ["standing"]
        assert "dropped invalid axis1_position" in caplog.text

    def test_all_invalid_becomes_empty_not_fabricated(self):
        # Unlike axis3, axis1 has no forced fallback — an all-invalid input
        # is dropped to [] (unlabeled), never guessed at.
        assert tm.sanitize_axis1_position(["bogus"]) == []


class TestSanitizeAxis3Action:
    def test_valid_values_pass_through(self):
        assert tm.sanitize_axis3_action(["sweep", "escape"]) == ["sweep", "escape"]

    def test_none_becomes_empty_list(self):
        assert tm.sanitize_axis3_action(None) == []

    def test_all_invalid_falls_back_to_transition(self, caplog):
        with caplog.at_level("WARNING"):
            result = tm.sanitize_axis3_action(["bogus_class"])
        assert result == ["transition"]
        assert "falling back to ['transition']" in caplog.text

    def test_partial_invalid_keeps_valid_subset(self):
        assert tm.sanitize_axis3_action(["sweep", "bogus"]) == ["sweep"]


class TestSanitizeAxis4Outcome:
    @pytest.mark.parametrize("value", ["successful", "unsuccessful", "unclear"])
    def test_valid_values_pass_through(self, value):
        assert tm.sanitize_axis4_outcome(value) == value

    def test_none_becomes_unclear(self):
        assert tm.sanitize_axis4_outcome(None) == "unclear"

    def test_invalid_becomes_unclear_and_logs(self, caplog):
        with caplog.at_level("WARNING"):
            result = tm.sanitize_axis4_outcome("bogus_outcome")
        assert result == "unclear"
        assert "falling back to 'unclear'" in caplog.text


# ---------------------------------------------------------------------------
# technique_guess -> curated shortlist (item 3, D1)
# ---------------------------------------------------------------------------

class TestResolveTechniqueShortlist:
    def test_empty_axis3_action_returns_none(self):
        assert tm.resolve_technique_shortlist("armbar", []) is None

    def test_exact_match_in_scope(self):
        assert tm.resolve_technique_shortlist("armbar", ["submission_arm_lock"]) == "armbar"

    def test_case_insensitive_exact_match(self):
        assert tm.resolve_technique_shortlist("Armbar", ["submission_arm_lock"]) == "armbar"
        assert tm.resolve_technique_shortlist("  ARMBAR  ", ["submission_arm_lock"]) == "armbar"

    def test_alias_rnc(self):
        assert tm.resolve_technique_shortlist("RNC", ["submission_choke"]) == "rear_naked_choke"

    def test_alias_dearce_to_darce_anaconda(self):
        assert tm.resolve_technique_shortlist("D'Arce Choke", ["submission_choke"]) == "darce_anaconda"

    def test_unmatched_guess_falls_back_to_other(self):
        assert tm.resolve_technique_shortlist("some totally novel lock", ["submission_arm_lock"]) == "other"

    def test_none_guess_falls_back_to_other_when_class_has_shortlist(self):
        assert tm.resolve_technique_shortlist(None, ["sweep"]) == "other"

    def test_transition_class_only_has_other(self):
        assert tm.resolve_technique_shortlist("anything", ["transition"]) == "other"

    def test_alias_scoped_to_axis3_not_returned_out_of_scope(self):
        # "rnc" aliases to rear_naked_choke, a submission_choke-only value —
        # asking for it under sweep (no overlap) should NOT return it.
        assert tm.resolve_technique_shortlist("rnc", ["sweep"]) == "other"

    def test_every_shortlist_value_resolves_to_itself(self):
        for cls, shortlist in tm.TECHNIQUE_SHORTLISTS.items():
            for value in shortlist:
                assert tm.resolve_technique_shortlist(value, [cls]) == value


# ---------------------------------------------------------------------------
# New -> legacy dual-emit mapping (item 4 — the search-422 seam)
# ---------------------------------------------------------------------------

class TestDualEmitLegacyFields:
    def test_every_single_axis3_class_maps_to_a_valid_legacy_action(self):
        for cls in tm.AXIS3_ACTION:
            action = tm.axis3_to_legacy_action([cls])
            assert action in tm.VALID_ACTIONS, f"{cls} -> {action} not in VALID_ACTIONS"

    def test_empty_axis3_maps_to_other(self):
        assert tm.axis3_to_legacy_action([]) == "other"
        assert tm.axis3_to_legacy_action(None) == "other"

    def test_multi_label_axis3_uses_precedence(self):
        # submission_* always wins over any co-occurring non-submission class.
        assert tm.axis3_to_legacy_action(["sweep", "submission_arm_lock"]) == "submission_attempt"
        assert tm.axis3_to_legacy_action(["transition", "escape"]) == "escape"

    def test_every_axis4_outcome_maps_to_a_valid_legacy_result(self):
        for outcome in tm.AXIS4_OUTCOME:
            result = tm.axis4_to_legacy_result(outcome)
            assert result in tm.VALID_RESULTS, f"{outcome} -> {result} not in VALID_RESULTS"

    def test_axis4_mapping_values(self):
        assert tm.axis4_to_legacy_result("successful") == "success"
        assert tm.axis4_to_legacy_result("unsuccessful") == "failed"
        assert tm.axis4_to_legacy_result("unclear") == "neutral"

    def test_none_outcome_maps_to_neutral(self):
        assert tm.axis4_to_legacy_result(None) == "neutral"

    def test_every_shortlist_value_maps_to_a_valid_legacy_technique(self):
        """Exhaustive: every value the engine could EVER pass as
        technique_shortlist (every class's shortlist, incl. every "other")
        must resolve to a legacy TechniqueType member — the core guarantee
        behind the T2 search-422 mitigation."""
        for shortlist in tm.TECHNIQUE_SHORTLISTS.values():
            for value in shortlist:
                technique = tm.technique_shortlist_to_legacy_technique(value)
                assert technique in tm.VALID_TECHNIQUES, f"{value} -> {technique} not in VALID_TECHNIQUES"

    def test_none_shortlist_maps_to_other(self):
        assert tm.technique_shortlist_to_legacy_technique(None) == "other"

    def test_verbatim_twins_pass_through_unchanged(self):
        for shortlist_value, legacy_value in tm.TECHNIQUE_SHORTLIST_TO_LEGACY_TECHNIQUE.items():
            assert shortlist_value == legacy_value, (
                f"{shortlist_value} -> {legacy_value} is not a verbatim twin"
            )

    def test_thirty_one_verbatim_twins_documented(self):
        """Pin the actual verified count (31) — the plan's own estimate of
        "20 values" undercounted; see the module's table docstring."""
        assert len(tm.TECHNIQUE_SHORTLIST_TO_LEGACY_TECHNIQUE) == 31

    def test_exhaustive_dual_emit_over_every_axis3_x_axis4_combo(self):
        """Full cartesian product of every single-label axis3 class x axis4
        outcome x every shortlist value in that class (plus None) — every
        combination must legally dual-emit."""
        for cls, outcome in itertools.product(tm.AXIS3_ACTION, tm.AXIS4_OUTCOME):
            shortlist_values = [None, *tm.TECHNIQUE_SHORTLISTS.get(cls, [])]
            for shortlist_value in shortlist_values:
                legacy = tm.dual_emit_legacy_fields([cls], outcome, shortlist_value)
                assert legacy["action"] in tm.VALID_ACTIONS
                assert legacy["technique"] in tm.VALID_TECHNIQUES
                assert legacy["result"] in tm.VALID_RESULTS

    def test_multi_label_axis3_combinations_all_dual_emit_legally(self):
        """Every 2-combination of axis3 classes (the realistic multi-label
        case) must also dual-emit legal legacy values."""
        for combo in itertools.combinations(tm.AXIS3_ACTION, 2):
            for outcome in tm.AXIS4_OUTCOME:
                legacy = tm.dual_emit_legacy_fields(list(combo), outcome, None)
                assert legacy["action"] in tm.VALID_ACTIONS
                assert legacy["technique"] in tm.VALID_TECHNIQUES
                assert legacy["result"] in tm.VALID_RESULTS


# ---------------------------------------------------------------------------
# Dead-code removal guard: the old pipeline-category API must be gone.
# ---------------------------------------------------------------------------

def test_map_to_frontend_taxonomy_removed():
    assert not hasattr(tm, "map_to_frontend_taxonomy")
    assert not hasattr(tm, "_infer_standup_action")
    assert not hasattr(tm, "_infer_guard_play_action")
    assert not hasattr(tm, "_infer_positional_action")
    assert not hasattr(tm, "_infer_defense_action")
