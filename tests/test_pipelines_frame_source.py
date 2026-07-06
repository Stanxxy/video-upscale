"""``service/pipelines/frame_source.py`` — top-2 selection determinism + degraded
branches, min-crop-size floor, identity-stability hysteresis (build plan §4.3/§4.4).
"""
from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from service.pipelines.frame_source import (
    AthleteTracker,
    Detection,
    build_sliding_windows,
    compute_crop_box,
    count_sliding_windows,
    parse_role,
    probe_native_fps,
    select_top2,
)

IMG_W, IMG_H = 1000, 1000


def _ffprobe_stdout(text):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=text.encode(), stderr=b"")


# --------------------------------------------------------------------------- #
# probe_native_fps — robust to multi-line ffprobe output
# --------------------------------------------------------------------------- #
def test_probe_native_fps_single_line_fraction():
    with patch("subprocess.run", return_value=_ffprobe_stdout("30000/1001")):
        assert probe_native_fps("u", {}) == pytest.approx(29.97, abs=0.01)


def test_probe_native_fps_handles_duplicated_line():
    # ffprobe can emit the rate twice (e.g. "25/1\n\n25/1") — must not blow up
    # on split(); take the first non-empty line. Regression for the
    # "too many values to unpack" crash seen on real YouTube streams.
    with patch("subprocess.run", return_value=_ffprobe_stdout("25/1\n\n25/1")):
        assert probe_native_fps("u", {}) == pytest.approx(25.0)


def test_probe_native_fps_empty_payload_raises_no_guess():
    with patch("subprocess.run", return_value=_ffprobe_stdout("   \n\n  ")):
        with pytest.raises(RuntimeError):
            probe_native_fps("u", {})


def _det(box, role, label=None, confidence=0.9):
    return Detection(box_px=list(box), role=role, label=label or f"{role}: x", confidence=confidence)


# --------------------------------------------------------------------------- #
# parse_role
# --------------------------------------------------------------------------- #
def test_parse_role_recognizes_all_four_tokens():
    assert parse_role("athlete: white gi") == "athlete"
    assert parse_role("referee: black shirt") == "referee"
    assert parse_role("coach: mat edge") == "coach"
    assert parse_role("bystander: crowd") == "bystander"


def test_parse_role_unrecognized_prefix_is_unlabeled():
    assert parse_role("person in red") == "unlabeled"
    assert parse_role("") == "unlabeled"


# --------------------------------------------------------------------------- #
# select_top2 — 4-branch degrade ladder (plan §4.3)
# --------------------------------------------------------------------------- #
def test_two_athletes_no_degradation():
    dets = [
        _det([100, 100, 300, 400], "athlete"),
        _det([400, 100, 600, 400], "athlete"),
    ]
    result = select_top2(dets, IMG_W, IMG_H)
    assert result.degraded is None
    assert len(result.candidates) == 2


def test_three_athletes_picks_the_contact_pair():
    """Two overlapping athletes + one far-away athlete: the overlapping pair
    must win over any pairing with the isolated one, even though the isolated
    one alone might individually score higher on centrality."""
    a = _det([100, 100, 400, 400], "athlete")  # overlaps with b
    b = _det([350, 100, 650, 400], "athlete")  # overlaps with a
    c = _det([700, 700, 900, 900], "athlete")  # isolated, far corner
    result = select_top2([a, b, c], IMG_W, IMG_H)
    assert result.degraded is None
    chosen_boxes = {tuple(d.box_px) for d in result.candidates}
    assert tuple(a.box_px) in chosen_boxes
    assert tuple(b.box_px) in chosen_boxes
    assert tuple(c.box_px) not in chosen_boxes


def test_single_athlete_degrades_to_single_athlete():
    dets = [_det([100, 100, 300, 400], "athlete"), _det([0, 0, 50, 50], "referee")]
    result = select_top2(dets, IMG_W, IMG_H)
    assert result.degraded == "single_athlete"
    assert len(result.candidates) == 1


def test_zero_athletes_two_persons_degrades_to_role_uncertain():
    dets = [_det([100, 100, 300, 400], "referee"), _det([400, 100, 600, 400], "coach")]
    result = select_top2(dets, IMG_W, IMG_H)
    assert result.degraded == "role_uncertain"
    assert len(result.candidates) == 2


def test_fewer_than_two_persons_drops_frame_no_detection():
    result = select_top2([_det([100, 100, 300, 400], "referee")], IMG_W, IMG_H)
    assert result.degraded == "no_detection"
    assert result.candidates is None
    assert result.boxes_px is None


def test_no_detections_at_all_drops_frame():
    result = select_top2([], IMG_W, IMG_H)
    assert result.degraded == "no_detection"
    assert result.candidates is None


def test_selection_is_deterministic_across_repeated_calls():
    dets = [
        _det([100, 100, 300, 400], "athlete"),
        _det([350, 120, 650, 420], "athlete"),
        _det([50, 700, 200, 900], "athlete"),
    ]
    results = [select_top2(dets, IMG_W, IMG_H) for _ in range(5)]
    boxes = [tuple(sorted(tuple(c.box_px) for c in r.candidates)) for r in results]
    assert len(set(boxes)) == 1  # identical every time


def test_selection_order_independent_of_input_order():
    a = _det([100, 100, 300, 400], "athlete")
    b = _det([350, 120, 650, 420], "athlete")
    r1 = select_top2([a, b], IMG_W, IMG_H)
    r2 = select_top2([b, a], IMG_W, IMG_H)
    assert {tuple(c.box_px) for c in r1.candidates} == {tuple(c.box_px) for c in r2.candidates}


# --------------------------------------------------------------------------- #
# Crop geometry — min-crop-size floor (plan §4.4 step 3)
# --------------------------------------------------------------------------- #
def test_crop_box_expands_to_floor_when_union_is_tiny():
    tiny_box = [[490, 490, 510, 510]]  # 20x20 px union, way below any floor
    square = compute_crop_box(tiny_box, img_shape=(1000, 1000), padding=0.2, min_crop_fraction=0.4)
    side = square[2] - square[0]
    assert side >= 0.4 * 1000 - 1  # allow rounding


def test_crop_box_does_not_shrink_a_large_union_below_its_own_size():
    """A union already bigger than the floor must NOT be tightened — floor only
    expands, never zooms in (plan: 'cap zoom-in, always allow zoom-out')."""
    big_box = [[0, 0, 900, 900]]
    square = compute_crop_box(big_box, img_shape=(1000, 1000), padding=0.0, min_crop_fraction=0.4)
    side = square[2] - square[0]
    assert side >= 900


def test_crop_box_stays_within_image_bounds():
    box = [[0, 0, 50, 50]]
    square = compute_crop_box(box, img_shape=(200, 200), padding=0.2, min_crop_fraction=0.9)
    x1, y1, x2, y2 = square
    assert x1 >= 0 and y1 >= 0 and x2 <= 200 and y2 <= 200


# --------------------------------------------------------------------------- #
# Identity stability — hysteresis (plan §4.4 step 5)
# --------------------------------------------------------------------------- #
def test_tracker_holds_pair_across_minor_movement():
    tracker = AthleteTracker(margin=0.15, min_streak=3)
    a1 = _det([100, 100, 300, 400], "athlete")
    b1 = _det([400, 100, 600, 400], "athlete")
    sel1 = select_top2([a1, b1], IMG_W, IMG_H)
    resolved1 = tracker.resolve(sel1, [a1, b1], IMG_W, IMG_H)
    assert resolved1.degraded is None

    # Same two people, slightly moved — should just track continuity (high IoU).
    a2 = _det([105, 105, 305, 405], "athlete")
    b2 = _det([405, 105, 605, 405], "athlete")
    sel2 = select_top2([a2, b2], IMG_W, IMG_H)
    resolved2 = tracker.resolve(sel2, [a2, b2], IMG_W, IMG_H)
    assert {tuple(b) for b in resolved2.boxes_px} == {tuple(a2.box_px), tuple(b2.box_px)}


def test_tracker_does_not_swap_on_single_frame_challenger():
    """A one-frame-only higher-contact/higher-score challenger pair must NOT
    immediately swap the held pair (hysteresis requires a sustained margin over
    several frames)."""
    tracker = AthleteTracker(margin=0.15, min_streak=3)
    a = _det([300, 400, 400, 500], "athlete")  # held pair: modest overlap
    b = _det([370, 400, 470, 500], "athlete")
    held = tracker.resolve(select_top2([a, b], IMG_W, IMG_H), [a, b], IMG_W, IMG_H)
    assert {tuple(x) for x in held.boxes_px} == {tuple(a.box_px), tuple(b.box_px)}

    # Challenger pair (c, d): bigger, more central, AND higher mutual overlap
    # than (a, b) — select_top2 picks (c, d) independently every frame, but
    # hysteresis must still not swap on the FIRST such frame.
    c = _det([450, 450, 650, 650], "athlete")
    d = _det([550, 450, 750, 650], "athlete")
    assert {tuple(x) for x in select_top2([a, b, c, d], IMG_W, IMG_H).boxes_px} == {
        tuple(c.box_px), tuple(d.box_px),
    }
    challenged = tracker.resolve(select_top2([a, b, c, d], IMG_W, IMG_H), [a, b, c, d], IMG_W, IMG_H)
    # Held pair (a, b) must still be what's returned — challenger hasn't sustained yet.
    assert {tuple(x) for x in challenged.boxes_px} == {tuple(a.box_px), tuple(b.box_px)}


def test_tracker_swaps_after_sustained_challenger_margin():
    tracker = AthleteTracker(margin=0.15, min_streak=3)
    a = _det([300, 400, 400, 500], "athlete")
    b = _det([370, 400, 470, 500], "athlete")
    tracker.resolve(select_top2([a, b], IMG_W, IMG_H), [a, b], IMG_W, IMG_H)

    c = _det([450, 450, 650, 650], "athlete")
    d = _det([550, 450, 750, 650], "athlete")
    last = None
    for _ in range(4):
        last = tracker.resolve(select_top2([a, b, c, d], IMG_W, IMG_H), [a, b, c, d], IMG_W, IMG_H)
    assert {tuple(x) for x in last.boxes_px} == {tuple(c.box_px), tuple(d.box_px)}


def test_tracker_accepts_immediately_when_held_pair_vanishes():
    tracker = AthleteTracker(margin=0.15, min_streak=3)
    a = _det([100, 100, 300, 400], "athlete")
    b = _det([400, 100, 600, 400], "athlete")
    tracker.resolve(select_top2([a, b], IMG_W, IMG_H), [a, b], IMG_W, IMG_H)

    c = _det([700, 700, 900, 900], "athlete")
    d = _det([600, 700, 800, 900], "athlete")
    resolved = tracker.resolve(select_top2([c, d], IMG_W, IMG_H), [c, d], IMG_W, IMG_H)
    assert {tuple(x) for x in resolved.boxes_px} == {tuple(c.box_px), tuple(d.box_px)}


def test_tracker_passes_through_dropped_frames_untouched():
    tracker = AthleteTracker()
    from service.pipelines.frame_source import SelectionResult

    dropped = SelectionResult(candidates=None, degraded="no_detection")
    assert tracker.resolve(dropped, [], IMG_W, IMG_H) is dropped
    assert tracker.held_boxes is None


# --------------------------------------------------------------------------- #
# Sliding windows (shared by real windowing + budget estimation)
# --------------------------------------------------------------------------- #
def test_build_sliding_windows_matches_production_accumulation():
    items = list(range(35))
    windows = build_sliding_windows(items, window_size=30, stride=15)
    assert windows[0] == list(range(0, 30))
    assert windows[1] == list(range(15, 35))  # trailing partial window (20 items)


def test_count_sliding_windows_matches_build_sliding_windows_length():
    for n in (0, 1, 29, 30, 44, 45, 100):
        assert count_sliding_windows(n, 30, 15) == len(build_sliding_windows(list(range(n)), 30, 15))
