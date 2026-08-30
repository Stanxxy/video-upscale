import pytest

from eval.adcc.score.highlight_baselines import (
    TARGET_SPANS_PER_MATCH,
    uniform_span_baseline,
    whole_match_baseline,
)


class TestWholeMatchBaseline:
    def test_one_span_covering_the_whole_window(self):
        windows = {"m1": (10.0, 610.0)}
        result = whole_match_baseline(windows)
        assert result["m1"] == [{"start_s": 10.0, "end_s": 610.0, "action_class": "highlight"}]

    def test_one_entry_per_match_id(self):
        windows = {"m1": (0.0, 100.0), "m2": (50.0, 250.0)}
        result = whole_match_baseline(windows)
        assert set(result.keys()) == {"m1", "m2"}
        assert len(result["m1"]) == 1
        assert len(result["m2"]) == 1


class TestUniformSpanBaseline:
    def test_span_count_matches_rounded_target_per_match(self):
        windows = {"m1": (0.0, 1000.0)}
        result = uniform_span_baseline(windows)
        assert len(result["m1"]) == round(TARGET_SPANS_PER_MATCH) == 2

    def test_points_are_evenly_spaced_within_the_window(self):
        windows = {"m1": (0.0, 100.0)}
        result = uniform_span_baseline(windows, target_spans_per_match=2.0)
        points = sorted(p["start_s"] for p in result["m1"])
        assert points == pytest.approx([25.0, 75.0])

    def test_every_prediction_is_a_point_start_equals_end(self):
        windows = {"m1": (0.0, 300.0)}
        result = uniform_span_baseline(windows)
        for p in result["m1"]:
            assert p["start_s"] == p["end_s"]
            assert p["action_class"] == "highlight"

    def test_non_positive_duration_yields_empty_list_not_a_crash(self):
        windows = {"m1": (100.0, 100.0)}
        result = uniform_span_baseline(windows)
        assert result["m1"] == []

    def test_custom_target_spans_per_match_is_honored(self):
        windows = {"m1": (0.0, 400.0)}
        result = uniform_span_baseline(windows, target_spans_per_match=4.0)
        assert len(result["m1"]) == 4

    def test_multiple_matches_each_get_their_own_span_count(self):
        windows = {"m1": (0.0, 200.0), "m2": (0.0, 200.0)}
        result = uniform_span_baseline(windows, target_spans_per_match=1.0)
        assert len(result["m1"]) == 1
        assert len(result["m2"]) == 1
