"""
Unit tests for M3 S9: K-segment boundary splitting (service/segment_runner.py).

Verifies:
1. split_segments produces correct boundaries for K=4 / 7956 frames.
2. No frame gap: every frame in [clip_start, clip_end) is covered by at least one segment.
3. Overlap regions are correct.
4. K=1 returns a single segment spanning the full range.
5. merge_tracking_results de-duplicates overlap frames correctly.
6. merge_analysis_results correctly passes clips through deduplicate_clips.
"""
import pytest
from service.segment_runner import (
    split_segments,
    merge_tracking_results,
    merge_analysis_results,
    OVERLAP_FRAMES,
    SegmentBounds,
)


class TestSplitSegments:
    def test_k4_7956_frames_no_gaps(self):
        """K=4, 7956 frames: every frame covered by at least one segment."""
        clip_start = 0
        clip_end = 7956
        segments = split_segments(clip_start, clip_end, k=4, overlap=OVERLAP_FRAMES)
        assert len(segments) == 4

        # Every frame index must be covered by at least one segment
        covered = set()
        for seg in segments:
            for f in range(seg.start_frame, seg.end_frame):
                covered.add(f)
        assert covered == set(range(clip_start, clip_end)), (
            f"Frames not covered: {set(range(clip_start, clip_end)) - covered}"
        )

    def test_k4_7956_segment_ids(self):
        """Segment IDs are 0..K-1."""
        segments = split_segments(0, 7956, k=4)
        assert [s.segment_id for s in segments] == [0, 1, 2, 3]

    def test_k4_7956_boundaries(self):
        """K=4, 7956 frames: first segment starts at 0, last ends at 7956."""
        segments = split_segments(0, 7956, k=4, overlap=OVERLAP_FRAMES)
        assert segments[0].start_frame == 0
        assert segments[-1].end_frame == 7956

    def test_k4_overlap_regions(self):
        """Interior segments have overlap padding at both ends."""
        segments = split_segments(0, 7956, k=4, overlap=30)
        # Segment 1 should start before the canonical boundary
        # Canonical boundary for seg1 = 0 + 1*1989 = 1989
        # With overlap: start = 1989 - 30 = 1959
        assert segments[1].start_frame < 1989
        assert segments[1].start_frame == max(0, 1989 - 30)
        # Segment 1 end should extend 30 beyond canonical end (3978)
        assert segments[1].end_frame > 3978
        assert segments[1].end_frame == min(7956, 3978 + 30)

    def test_k1_single_segment(self):
        """K=1 returns exactly one segment covering the full range, no overlap."""
        segments = split_segments(100, 1000, k=1, overlap=30)
        assert len(segments) == 1
        assert segments[0].start_frame == 100
        assert segments[0].end_frame == 1000
        assert segments[0].segment_id == 0

    def test_k2_symmetric_overlap(self):
        """K=2: first segment has overlap at end, second at start only."""
        segments = split_segments(0, 200, k=2, overlap=10)
        # seg0: [0, 100+10) = [0, 110)
        assert segments[0].start_frame == 0
        assert segments[0].end_frame == 110
        # seg1: [100-10, 200) = [90, 200)
        assert segments[1].start_frame == 90
        assert segments[1].end_frame == 200

    def test_k4_non_zero_start(self):
        """Clip not starting at 0."""
        segments = split_segments(1000, 5000, k=4, overlap=20)
        assert segments[0].start_frame == 1000
        assert segments[-1].end_frame == 5000
        # Check no frame gaps
        covered = set()
        for seg in segments:
            covered.update(range(seg.start_frame, seg.end_frame))
        assert covered.issuperset(set(range(1000, 5000)))

    def test_k1_raises_on_invalid(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            split_segments(0, 100, k=0)

    def test_invalid_clip_range(self):
        with pytest.raises(ValueError, match="clip_end"):
            split_segments(100, 50, k=2)

    def test_smoke_k4_frame_count(self):
        """Smoke fixture: K=4, 1798 frames → 4 segments of ~450 each."""
        segments = split_segments(0, 1798, k=4, overlap=OVERLAP_FRAMES)
        assert len(segments) == 4
        # All 1798 frames covered
        covered = set()
        for seg in segments:
            covered.update(range(seg.start_frame, seg.end_frame))
        assert set(range(0, 1798)).issubset(covered)


class TestMergeTrackingResults:
    def _make_frames(self, frame_indices: list[int], track_id: int = 1) -> list[dict]:
        return [
            {
                "frame_idx": idx,
                "athletes": [{"track_id": track_id, "box": [0, 0, 100, 200]}],
            }
            for idx in frame_indices
        ]

    def test_k1_merge_returns_all_frames(self):
        """K=1 merge: single segment returns all frames unchanged."""
        segments = split_segments(0, 100, k=1, overlap=30)
        frames = self._make_frames(list(range(100)))
        result = merge_tracking_results(segments, [frames], overlap=30)
        assert len(result["frames"]) == 100

    def test_k2_merge_no_duplicates(self):
        """K=2 merge: overlap frames are not duplicated."""
        segments = split_segments(0, 100, k=2, overlap=10)
        # seg0: [0, 60), seg1: [40, 100)
        seg0_frames = self._make_frames(list(range(segments[0].start_frame, segments[0].end_frame)))
        seg1_frames = self._make_frames(list(range(segments[1].start_frame, segments[1].end_frame)))
        result = merge_tracking_results(segments, [seg0_frames, seg1_frames], overlap=10)
        frame_indices = [f["frame_idx"] for f in result["frames"]]
        # No duplicates
        assert len(frame_indices) == len(set(frame_indices)), "Duplicate frame indices found"
        # All original frames covered
        assert set(range(0, 100)).issubset(set(frame_indices))

    def test_k4_merge_sorted(self):
        """K=4 merge: result is sorted by frame_idx."""
        segments = split_segments(0, 400, k=4, overlap=10)
        segment_frames = []
        for seg in segments:
            segment_frames.append(
                self._make_frames(list(range(seg.start_frame, seg.end_frame)))
            )
        result = merge_tracking_results(segments, segment_frames, overlap=10)
        indices = [f["frame_idx"] for f in result["frames"]]
        assert indices == sorted(indices)


class TestMergeAnalysisResults:
    """Tests for merge_analysis_results: K-segment clip merge correctness.

    Regression guard for the bug where merge_analysis_results passed a flat
    list of clip dicts to deduplicate_clips(), which expects window-level
    dicts, causing all clips to be silently dropped.
    """

    def _make_clip(self, start: int, end: int, role: str = "athlete_a", action: str = "takedown") -> dict:
        return {
            "start_frame": start,
            "end_frame": end,
            "role": role,
            "action": action,
            "technique": "double_leg",
            "specific_technique": "Double Leg Takedown",
            "confidence": 0.9,
        }

    def test_single_segment_clips_preserved(self):
        """Single segment with clips: clips must pass through to final output."""
        seg_analysis = {
            "match_summary": "test",
            "clips": [self._make_clip(0, 100), self._make_clip(200, 300)],
            "fps": 60.0,
        }
        result = merge_analysis_results([seg_analysis])
        assert result is not None
        assert len(result["clips"]) == 2, (
            f"Expected 2 clips, got {len(result['clips'])}. "
            "merge_analysis_results may be passing clips in wrong format to deduplicate_clips."
        )

    def test_two_segments_clips_merged(self):
        """Two segments with distinct clips: all clips present in merged output."""
        seg0 = {
            "clips": [self._make_clip(0, 100, role="athlete_a")],
            "fps": 60.0,
        }
        seg1 = {
            "clips": [self._make_clip(2000, 2100, role="athlete_b")],
            "fps": 60.0,
        }
        result = merge_analysis_results([seg0, seg1])
        assert result is not None
        assert len(result["clips"]) >= 2, (
            f"Expected >=2 clips from 2 segments, got {len(result['clips'])}."
        )

    def test_none_segments_ignored(self):
        """None entries in segment_analyses are gracefully ignored."""
        seg = {
            "clips": [self._make_clip(0, 100)],
            "fps": 60.0,
        }
        result = merge_analysis_results([None, seg, None])
        assert result is not None
        assert len(result["clips"]) >= 1

    def test_all_none_returns_none(self):
        """All-None segments returns None."""
        result = merge_analysis_results([None, None])
        assert result is None

    def test_empty_input_returns_none(self):
        """Empty segment list returns None."""
        result = merge_analysis_results([])
        assert result is None

    def test_fps_propagated(self):
        """fps from segment analyses is preserved in merged result."""
        seg = {"clips": [self._make_clip(0, 100)], "fps": 59.94}
        result = merge_analysis_results([seg])
        assert result is not None
        assert abs(result["fps"] - 59.94) < 0.01
