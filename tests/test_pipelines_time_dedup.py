"""``service/pipelines/time_dedup.py`` — the seconds-based dedup branch for
``simplified-tags-time-v1`` (Brooks HIGH seam: real seconds merge, no
fabricated frame numbers, ``ctx.native_fps`` stays ``None`` end-to-end on this
pipeline — the caller never derives a frame count from these seconds).
"""
from __future__ import annotations

from service.pipelines import time_dedup


def _chunk(clips):
    return {"window": 1, "frames": [], "analysis": {"clips": clips}}


# --------------------------------------------------------------------------- #
# time-IoU
# --------------------------------------------------------------------------- #
def test_time_iou_no_overlap_is_zero():
    assert time_dedup._time_iou(0, 10, 20, 30) == 0.0


def test_time_iou_full_overlap_is_one():
    assert time_dedup._time_iou(0, 10, 0, 10) == 1.0


def test_time_iou_partial_overlap_is_between_zero_and_one():
    iou = time_dedup._time_iou(0, 10, 5, 15)
    assert 0.0 < iou < 1.0
    assert iou == 5 / 15  # overlap=5, union=15


# --------------------------------------------------------------------------- #
# Merge on time-IoU + exact axis identity (action_class + actor)
# --------------------------------------------------------------------------- #
def test_merges_overlapping_clips_same_actor_and_action_class():
    raw_results = [_chunk([
        {"start_s": 10.0, "end_s": 20.0, "actor": "top", "action_class": "submission_arm_lock", "confidence": 0.6},
        {"start_s": 18.0, "end_s": 25.0, "actor": "top", "action_class": "submission_arm_lock", "confidence": 0.9},
    ])]
    merged = time_dedup.deduplicate_clips_by_time(raw_results)
    assert len(merged) == 1
    assert merged[0]["start_s"] == 10.0
    assert merged[0]["end_s"] == 25.0
    assert merged[0]["confidence"] == 0.9  # keep-higher-confidence clip's fields


def test_does_not_merge_across_different_action_class():
    raw_results = [_chunk([
        {"start_s": 10.0, "end_s": 20.0, "actor": "top", "action_class": "submission_arm_lock", "confidence": 0.6},
        {"start_s": 12.0, "end_s": 22.0, "actor": "top", "action_class": "guard_pass", "confidence": 0.9},
    ])]
    merged = time_dedup.deduplicate_clips_by_time(raw_results)
    assert len(merged) == 2  # different axis identity — kept distinct even though they overlap in time


def test_does_not_merge_across_different_actor():
    raw_results = [_chunk([
        {"start_s": 10.0, "end_s": 20.0, "actor": "top", "action_class": "guard_pass", "confidence": 0.6},
        {"start_s": 12.0, "end_s": 22.0, "actor": "bottom", "action_class": "guard_pass", "confidence": 0.9},
    ])]
    merged = time_dedup.deduplicate_clips_by_time(raw_results)
    assert len(merged) == 2


def test_does_not_merge_non_overlapping_clips_even_same_identity():
    raw_results = [_chunk([
        {"start_s": 0.0, "end_s": 5.0, "actor": "top", "action_class": "guard_pass", "confidence": 0.6},
        {"start_s": 50.0, "end_s": 55.0, "actor": "top", "action_class": "guard_pass", "confidence": 0.9},
    ])]
    merged = time_dedup.deduplicate_clips_by_time(raw_results)
    assert len(merged) == 2


def test_merges_across_chunk_boundaries_overlap_reporting():
    """The realistic case: PASS 2's overlap_s back-overlap causes the SAME
    real event to be re-reported once near the end of chunk 1 and once near
    the start of chunk 2 (both already converted to ABSOLUTE seconds by the
    caller) — dedup must collapse them."""
    raw_results = [
        _chunk([{"start_s": 58.0, "end_s": 63.0, "actor": "bottom", "action_class": "escape", "confidence": 0.7}]),
        _chunk([{"start_s": 59.0, "end_s": 64.0, "actor": "bottom", "action_class": "escape", "confidence": 0.85}]),
    ]
    merged = time_dedup.deduplicate_clips_by_time(raw_results)
    assert len(merged) == 1
    assert merged[0]["start_s"] == 58.0
    assert merged[0]["end_s"] == 64.0
    assert merged[0]["confidence"] == 0.85


def test_result_sorted_by_start_s():
    raw_results = [_chunk([
        {"start_s": 40.0, "end_s": 45.0, "actor": "top", "action_class": "sweep", "confidence": 0.5},
        {"start_s": 5.0, "end_s": 10.0, "actor": "bottom", "action_class": "escape", "confidence": 0.5},
    ])]
    merged = time_dedup.deduplicate_clips_by_time(raw_results)
    assert [c["start_s"] for c in merged] == [5.0, 40.0]


# --------------------------------------------------------------------------- #
# Malformed clips — dropped, never fabricated into a mergeable shape
# --------------------------------------------------------------------------- #
def test_drops_clip_missing_start_s_or_end_s():
    raw_results = [_chunk([
        {"start_s": 0.0, "end_s": 5.0, "actor": "top", "action_class": "sweep", "confidence": 0.5},
        {"actor": "top", "action_class": "sweep", "confidence": 0.9},  # missing start_s/end_s
    ])]
    merged = time_dedup.deduplicate_clips_by_time(raw_results)
    assert len(merged) == 1
    assert merged[0]["start_s"] == 0.0


def test_drops_clip_with_non_numeric_start_s():
    raw_results = [_chunk([
        {"start_s": "oops", "end_s": 5.0, "actor": "top", "action_class": "sweep", "confidence": 0.5},
    ])]
    merged = time_dedup.deduplicate_clips_by_time(raw_results)
    assert merged == []


def test_empty_raw_results_returns_empty_list():
    assert time_dedup.deduplicate_clips_by_time([]) == []
    assert time_dedup.deduplicate_clips_by_time([_chunk([])]) == []


def test_does_not_mutate_input():
    raw_results = [_chunk([
        {"start_s": 0.0, "end_s": 5.0, "actor": "top", "action_class": "sweep", "confidence": 0.5},
    ])]
    time_dedup.deduplicate_clips_by_time(raw_results)
    assert raw_results[0]["analysis"]["clips"][0]["start_s"] == 0.0  # untouched original
