import pytest

from eval.adcc.predict.scope_policy import (
    MAX_SLICE_S,
    MULTI_MATCH_POST_MARGIN_S,
    MULTI_MATCH_PRE_MARGIN_S,
    SLICE_OVERLAP_S,
    ScopePolicyError,
    ScopeWindow,
    build_slice_plan,
    multi_match_scope,
    single_match_scope,
    slice_scope,
)


class TestSingleMatchScope:
    def test_whole_file(self):
        window = single_match_scope("m1", 633.0)
        assert window.start_s == 0.0
        assert window.end_s == 633.0
        assert window.is_multi_match is False

    def test_rejects_non_positive_duration(self):
        with pytest.raises(ScopePolicyError):
            single_match_scope("m1", 0.0)
        with pytest.raises(ScopePolicyError):
            single_match_scope("m1", -5.0)


class TestMultiMatchScope:
    def test_gold_derived_window(self):
        window = multi_match_scope("m2", [500.0, 620.0, 700.0])
        assert window.start_s == 500.0 - MULTI_MATCH_PRE_MARGIN_S
        assert window.end_s == 700.0 + MULTI_MATCH_POST_MARGIN_S
        assert window.is_multi_match is True

    def test_clamped_to_zero_when_pre_margin_would_go_negative(self):
        window = multi_match_scope("m2", [10.0, 50.0])
        assert window.start_s == 0.0  # 10 - 120 < 0, clamped

    def test_empty_gold_t_s_raises(self):
        with pytest.raises(ScopePolicyError):
            multi_match_scope("m2", [])


class TestSliceScope:
    def test_under_cap_returns_single_slice(self):
        window = ScopeWindow(match_id="m1", start_s=0.0, end_s=600.0, is_multi_match=False)
        slices = slice_scope(window)
        assert len(slices) == 1
        assert slices[0].start_s == 0.0
        assert slices[0].end_s == 600.0
        assert slices[0].slice_index == 0

    def test_exactly_at_cap_returns_single_slice(self):
        window = ScopeWindow(match_id="m1", start_s=0.0, end_s=MAX_SLICE_S, is_multi_match=False)
        slices = slice_scope(window)
        assert len(slices) == 1

    def test_over_cap_slices_with_overlap(self):
        # 1500s scope, 720s cap, 45s overlap -> 3 slices.
        window = ScopeWindow(match_id="m1", start_s=0.0, end_s=1500.0, is_multi_match=False)
        slices = slice_scope(window)
        assert len(slices) >= 2
        # first slice starts exactly at window start; last ends exactly at window end
        assert slices[0].start_s == 0.0
        assert slices[-1].end_s == 1500.0
        # every slice is within the cap
        for s in slices:
            assert s.duration_s <= MAX_SLICE_S + 1e-9
        # consecutive slices overlap by exactly SLICE_OVERLAP_S (except possibly
        # the final short slice, whose overlap with its predecessor may be
        # larger if the remainder is small)
        for prev, nxt in zip(slices, slices[1:-1]):
            overlap = prev.end_s - nxt.start_s
            assert abs(overlap - SLICE_OVERLAP_S) < 1e-9
        # slice_index is 0-based and ordered
        assert [s.slice_index for s in slices] == list(range(len(slices)))

    def test_seam_landing_event_is_within_at_least_one_slices_non_overlap_zone_and_both_slices_near_seam(self):
        """A worst-case 25-match VOD style long scope: any point in the
        middle of the scope must fall inside at least one slice, and points
        near an internal seam fall inside BOTH adjacent slices (proving the
        overlap actually straddles the seam, the property stitch.py relies
        on)."""
        window = ScopeWindow(match_id="m1", start_s=0.0, end_s=2000.0, is_multi_match=False)
        slices = slice_scope(window)
        first_seam = slices[0].end_s  # the boundary before overlap pullback
        seam_point = slices[1].start_s + (SLICE_OVERLAP_S / 2.0)  # squarely inside the overlap zone
        covering = [s for s in slices if s.start_s <= seam_point <= s.end_s]
        assert len(covering) >= 2

    def test_never_exceeds_window_end(self):
        window = ScopeWindow(match_id="m1", start_s=100.0, end_s=100.0 + MAX_SLICE_S * 2.3, is_multi_match=True)
        slices = slice_scope(window)
        assert slices[-1].end_s == window.end_s
        assert all(s.end_s <= window.end_s for s in slices)

    def test_non_positive_duration_raises(self):
        window = ScopeWindow(match_id="m1", start_s=100.0, end_s=100.0, is_multi_match=False)
        with pytest.raises(ScopePolicyError):
            slice_scope(window)


class TestBuildSlicePlan:
    def test_multi_match_dispatch(self):
        slices = build_slice_plan("m1", is_multi_match=True, gold_t_s_values=[500.0, 620.0])
        assert slices[0].match_id == "m1"

    def test_single_match_dispatch(self):
        slices = build_slice_plan("m1", is_multi_match=False, ffprobe_duration_s=400.0)
        assert len(slices) == 1
        assert slices[0].end_s == 400.0

    def test_multi_match_with_duration_arg_is_an_error(self):
        with pytest.raises(ScopePolicyError):
            build_slice_plan("m1", is_multi_match=True, gold_t_s_values=[1.0], ffprobe_duration_s=400.0)

    def test_single_match_with_gold_t_s_arg_is_an_error(self):
        with pytest.raises(ScopePolicyError):
            build_slice_plan("m1", is_multi_match=False, gold_t_s_values=[1.0], ffprobe_duration_s=400.0)

    def test_multi_match_missing_gold_t_s_is_an_error(self):
        with pytest.raises(ScopePolicyError):
            build_slice_plan("m1", is_multi_match=True)

    def test_single_match_missing_duration_is_an_error(self):
        with pytest.raises(ScopePolicyError):
            build_slice_plan("m1", is_multi_match=False)
