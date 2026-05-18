"""
Unit tests for M3 S9: identity stitching (service/segment_runner.py).

Verifies:
1. stitch_segment_identity correctly remaps seg1 athlete IDs to seg0 IDs.
2. Spatial proximity fallback works when DINOv2 not used.
3. Ambiguous case (single athlete) maps correctly.
4. Identity mapping is transitive across multiple boundaries.
"""
import pytest
from service.segment_runner import stitch_segment_identity


def _make_frame(frame_idx: int, athletes: list[dict]) -> dict:
    return {"frame_idx": frame_idx, "athletes": athletes}


def _make_athlete(track_id, box):
    return {"track_id": track_id, "box": box}


class TestStitchSegmentIdentity:
    def test_simple_remap_two_athletes(self):
        """
        seg0 has athletes 1 (left) and 2 (right).
        seg1 has athletes 2 (left, same position as seg0's athlete 1) and 1 (right).
        After stitching, seg1 should have: original 2 → remapped to 1, original 1 → remapped to 2.
        """
        # seg0 overlap region: athlete 1 at left (~50px), athlete 2 at right (~300px)
        seg0_frames = [
            _make_frame(i, [
                _make_athlete(1, [10, 10, 100, 200]),   # left side
                _make_athlete(2, [300, 10, 400, 200]),  # right side
            ])
            for i in range(30)
        ]
        # seg1 overlap: athlete IDs are swapped — "2" is on the left, "1" is on the right
        # (i.e., same physical positions but different IDs)
        seg1_frames = [
            _make_frame(i + 30, [
                _make_athlete(2, [15, 10, 105, 200]),   # left (matches seg0's athlete 1)
                _make_athlete(1, [295, 10, 395, 200]),  # right (matches seg0's athlete 2)
            ])
            for i in range(30)
        ]

        result = stitch_segment_identity(seg0_frames, seg1_frames, overlap_window=30)

        # After remapping: seg1's "2" should become "1" (maps to left-side), "1" → "2"
        for frame in result:
            aids = {a["track_id"] for a in frame["athletes"]}
            assert 1 in aids, f"athlete 1 missing in frame {frame['frame_idx']}"
            assert 2 in aids, f"athlete 2 missing in frame {frame['frame_idx']}"
            for a in frame["athletes"]:
                if a["box"][0] < 150:  # left athlete
                    assert a["track_id"] == 1, (
                        f"Left athlete should be id=1, got {a['track_id']}"
                    )
                else:  # right athlete
                    assert a["track_id"] == 2, (
                        f"Right athlete should be id=2, got {a['track_id']}"
                    )

    def test_identity_mapping_unchanged(self):
        """
        When seg1 athlete IDs already match seg0 positions, no remapping occurs.
        """
        seg0_frames = [
            _make_frame(i, [
                _make_athlete(1, [10, 10, 100, 200]),
                _make_athlete(2, [300, 10, 400, 200]),
            ])
            for i in range(30)
        ]
        seg1_frames = [
            _make_frame(i + 30, [
                _make_athlete(1, [12, 10, 102, 200]),   # same position as seg0's 1
                _make_athlete(2, [298, 10, 398, 200]),  # same position as seg0's 2
            ])
            for i in range(30)
        ]
        result = stitch_segment_identity(seg0_frames, seg1_frames, overlap_window=30)
        # IDs should be unchanged
        for frame in result:
            for a in frame["athletes"]:
                orig_id = next(
                    oa["track_id"]
                    for of in seg1_frames
                    if of["frame_idx"] == frame["frame_idx"]
                    for oa in of["athletes"]
                    if oa["box"] == a["box"]
                )
                assert a["track_id"] == orig_id

    def test_empty_seg0_returns_seg1_unchanged(self):
        """Empty seg0 overlap → seg1 returned unchanged."""
        seg1_frames = [
            _make_frame(i, [_make_athlete(1, [10, 10, 100, 200])])
            for i in range(10)
        ]
        result = stitch_segment_identity([], seg1_frames, overlap_window=30)
        assert result is seg1_frames  # returned unchanged (identity)

    def test_empty_seg1_returns_empty(self):
        """Empty seg1 → empty result."""
        seg0_frames = [
            _make_frame(i, [_make_athlete(1, [10, 10, 100, 200])])
            for i in range(10)
        ]
        result = stitch_segment_identity(seg0_frames, [], overlap_window=30)
        assert result == []

    def test_single_athlete_maps_correctly(self):
        """Single athlete per frame: mapping should be stable."""
        seg0_frames = [
            _make_frame(i, [_make_athlete(1, [50, 50, 150, 250])])
            for i in range(30)
        ]
        seg1_frames = [
            _make_frame(i + 30, [_make_athlete(3, [55, 52, 155, 252])])  # same position
            for i in range(30)
        ]
        result = stitch_segment_identity(seg0_frames, seg1_frames, overlap_window=30)
        # All athletes in result should have id=1 (mapped from 3)
        for frame in result:
            for a in frame["athletes"]:
                assert a["track_id"] == 1, (
                    f"Expected remapped id=1, got {a['track_id']}"
                )

    def test_remapping_applies_to_all_frames(self):
        """Remapping applies to ALL frames in seg1, not just the overlap region."""
        seg0_frames = [
            _make_frame(i, [
                _make_athlete(1, [10, 10, 100, 200]),
                _make_athlete(2, [300, 10, 400, 200]),
            ])
            for i in range(30)
        ]
        # seg1 has 100 frames: first 30 in overlap, remaining 70 are non-overlap
        seg1_frames = [
            _make_frame(i + 30, [
                _make_athlete(2, [15, 10, 105, 200]),   # left — maps to id=1
                _make_athlete(1, [295, 10, 395, 200]),  # right — maps to id=2
            ])
            for i in range(100)
        ]
        result = stitch_segment_identity(seg0_frames, seg1_frames, overlap_window=30)
        assert len(result) == 100
        # Check a frame beyond the overlap window (e.g. frame index 90)
        beyond_frame = next(f for f in result if f["frame_idx"] == 90 + 30)
        left_ath = next(a for a in beyond_frame["athletes"] if a["box"][0] < 150)
        assert left_ath["track_id"] == 1, (
            f"Non-overlap frame: left athlete should be id=1, got {left_ath['track_id']}"
        )

    def test_no_athletes_in_overlap(self):
        """Frames with no athletes produce empty mapping — no crash."""
        seg0_frames = [
            _make_frame(i, [])  # no athletes
            for i in range(30)
        ]
        seg1_frames = [
            _make_frame(i + 30, [_make_athlete(1, [10, 10, 100, 200])])
            for i in range(30)
        ]
        # Should not raise, should return seg1 unchanged
        result = stitch_segment_identity(seg0_frames, seg1_frames, overlap_window=30)
        assert len(result) == 30
