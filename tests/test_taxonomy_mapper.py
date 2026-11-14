"""Tests for ``service/taxonomy_mapper.py``: the axis vocabulary re-exported
from shared_lib and the fail-loud-with-explicit-sentinel sanitizers."""
from __future__ import annotations

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

    def test_is_valid_axis1_axis3_axis4(self):
        assert tm.is_valid_axis1(["standing", "mount"])
        assert not tm.is_valid_axis1(["bogus"])
        assert tm.is_valid_axis3(["sweep"])
        assert not tm.is_valid_axis3(["bogus"])
        assert tm.is_valid_axis4("successful")
        assert not tm.is_valid_axis4("bogus")


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
