from eval.adcc.predict.stitch import stitch_match_predictions


def _clip(start_s, end_s, action_class, confidence=0.8, actor="top", **extra):
    clip = {"start_s": start_s, "end_s": end_s, "action_class": action_class, "confidence": confidence, "actor": actor}
    clip.update(extra)
    return clip


class TestOutputShape:
    def test_matches_report_predictions_contract(self):
        result = stitch_match_predictions("m1", [[_clip(0, 5, "guard_pass")]])
        assert set(result.keys()) == {"match_id", "clips"}
        assert result["match_id"] == "m1"
        assert isinstance(result["clips"], list)

    def test_empty_input_yields_empty_clips(self):
        result = stitch_match_predictions("m1", [[], []])
        assert result == {"match_id": "m1", "clips": []}


class TestSeamDuplicateInOverlapZone:
    def test_duplicate_in_overlap_zone_merges_to_one_survivor(self):
        """Two slices, overlap zone [670, 715]: BOTH slices independently
        report the same real guard_pass sitting inside the overlap -> 1
        survivor, not 2."""
        slice_a_clips = [_clip(700, 706, "guard_pass", confidence=0.7)]  # tail of slice A
        slice_b_clips = [_clip(701, 707, "guard_pass", confidence=0.9)]  # head of slice B, same real event
        result = stitch_match_predictions("m1", [slice_a_clips, slice_b_clips])
        assert len(result["clips"]) == 1
        # higher-confidence half's fields survive
        assert result["clips"][0]["confidence"] == 0.9

    def test_distinct_events_in_overlap_zone_are_not_merged(self):
        """Two REAL, distinct events that both happen to sit in the overlap
        zone (different action_class, far apart in time) must NOT merge."""
        slice_a_clips = [_clip(700, 703, "guard_pass")]
        slice_b_clips = [_clip(710, 713, "sweep")]  # different class, temporally far (>10s midpoint gap)
        result = stitch_match_predictions("m1", [slice_a_clips, slice_b_clips])
        assert len(result["clips"]) == 2


class TestSeamBoundaryCompletion:
    def test_event_completing_exactly_on_seam_boundary_captured_once_not_lost_not_doubled(self):
        """Slice A's own scope ends at t=720 and truncates the event right at
        its own edge; slice B (whose scope starts at 675, i.e. the 45s
        overlap) reports the SAME event in full. Stitch must produce exactly
        ONE clip spanning the full technique -- not 2 (double-counted) and
        not 0 (lost)."""
        slice_a_clips = [_clip(715, 720, "submission_choke", confidence=0.6)]  # truncated at slice A's own edge
        slice_b_clips = [_clip(714, 723, "submission_choke", confidence=0.85)]  # full report from slice B
        result = stitch_match_predictions("m1", [slice_a_clips, slice_b_clips])
        assert len(result["clips"]) == 1
        merged = result["clips"][0]
        # union span captures the FULL technique, not the truncated half
        assert merged["start_s"] == 714
        assert merged["end_s"] == 723
        assert merged["confidence"] == 0.85


class TestActorWobbleAcrossSeam:
    def test_actor_wobble_duplicate_across_seam_is_merged_not_kept_as_two(self):
        """The exact dedup-bug-D scenario at a seam: same real exchange,
        actor flips top<->contested across the two independent slice runs.
        time_dedup.py's (action_class, actor) bucketing would keep both;
        stitch.py must merge them into one survivor."""
        slice_a_clips = [_clip(500, 505, "sweep", confidence=0.5, actor="top")]
        slice_b_clips = [_clip(501, 506, "sweep", confidence=0.75, actor="contested")]
        result = stitch_match_predictions("m1", [slice_a_clips, slice_b_clips])
        assert len(result["clips"]) == 1
        assert result["clips"][0]["actor"] == "contested"  # higher-confidence half wins


class TestSubmissionFamilyCompat:
    def test_submission_subtype_flip_across_seam_still_merges(self):
        """A occlusion-driven subtype flip (choke vs arm_lock) for the SAME
        seam-straddling finish is still one real event under the
        submission-family compat rule."""
        slice_a_clips = [_clip(300, 306, "submission_choke", confidence=0.55)]
        slice_b_clips = [_clip(301, 307, "submission_arm_lock", confidence=0.8)]
        result = stitch_match_predictions("m1", [slice_a_clips, slice_b_clips])
        assert len(result["clips"]) == 1
        assert result["clips"][0]["action_class"] == "submission_arm_lock"

    def test_non_submission_classes_require_exact_match_not_family_compat(self):
        """guard_pass and sweep are NOT family-compatible with each other --
        only the three submission_* classes get that treatment."""
        slice_a_clips = [_clip(300, 306, "guard_pass", confidence=0.55)]
        slice_b_clips = [_clip(301, 307, "sweep", confidence=0.8)]
        result = stitch_match_predictions("m1", [slice_a_clips, slice_b_clips])
        assert len(result["clips"]) == 2


class TestMalformedClipHandling:
    def test_malformed_clip_is_dropped_not_fabricated(self, caplog):
        slice_a_clips = [_clip(10, 15, "guard_pass"), {"start_s": "not-a-number", "end_s": 5, "action_class": "sweep"}]
        with caplog.at_level("WARNING"):
            result = stitch_match_predictions("m1", [slice_a_clips])
        assert len(result["clips"]) == 1
        assert any("dropping malformed clip" in rec.message for rec in caplog.records)

    def test_inverted_span_is_dropped(self):
        slice_a_clips = [{"start_s": 20, "end_s": 10, "action_class": "guard_pass", "confidence": 0.5}]
        result = stitch_match_predictions("m1", [slice_a_clips])
        assert result["clips"] == []


class TestMultiSliceThreeWay:
    def test_three_slices_middle_one_bridges_a_chain(self):
        """A three-slice run (worst-case 25-match VOD style): the SAME real
        event partially reported by slice 0 and slice 2's neighbors, fully
        bridged by slice 1's overlap coverage -- still collapses to one
        survivor even with 3 sources."""
        s0 = [_clip(710, 715, "back_take", confidence=0.5)]
        s1 = [_clip(712, 718, "back_take", confidence=0.9)]
        s2 = [_clip(716, 720, "back_take", confidence=0.6)]
        result = stitch_match_predictions("m1", [s0, s1, s2])
        assert len(result["clips"]) == 1
        assert result["clips"][0]["start_s"] == 710
        assert result["clips"][0]["end_s"] == 720
