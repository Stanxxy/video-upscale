"""``service/worker/seam_dedup.py`` — pure temporal-proximity +
class-compatibility dedup helpers for the 45s outer-chunk overlap
(2026-07-26-engine13-rescope-single-call-cutover.md AC6; Brooks's
architecture ruling §3, refusing a ``time_dedup.py``-style (action_class,
actor) bucketing port for this specific seam)."""
from __future__ import annotations

from service.worker import seam_dedup


# --------------------------------------------------------------------------- #
# class_compatible
# --------------------------------------------------------------------------- #
def test_class_compatible_exact_match():
    assert seam_dedup.class_compatible("guard_pass", "guard_pass") is True


def test_class_compatible_submission_family_cross_compat():
    assert seam_dedup.class_compatible("submission_choke", "submission_arm_lock") is True
    assert seam_dedup.class_compatible("submission_leg_lock", "submission_choke") is True


def test_class_compatible_non_submission_mismatch_rejected():
    assert seam_dedup.class_compatible("guard_pass", "takedown_attempt") is False


def test_class_compatible_submission_vs_non_submission_rejected():
    assert seam_dedup.class_compatible("submission_choke", "guard_pass") is False


def test_class_compatible_none_values_never_fabricated_leniency():
    assert seam_dedup.class_compatible(None, None) is True  # only equal-None is "compatible"
    assert seam_dedup.class_compatible(None, "guard_pass") is False
    assert seam_dedup.class_compatible("guard_pass", None) is False


# --------------------------------------------------------------------------- #
# spans_close
# --------------------------------------------------------------------------- #
def test_spans_close_overlapping_spans():
    assert seam_dedup.spans_close(700.0, 715.0, 710.0, 720.0, 10.0) is True


def test_spans_close_non_overlapping_within_proximity_window():
    # midpoints: 707.5 and 720.5 -> 13.0 apart, still <= window(15.0)
    assert seam_dedup.spans_close(700.0, 715.0, 718.0, 723.0, 15.0) is True


def test_spans_close_non_overlapping_beyond_proximity_window():
    assert seam_dedup.spans_close(700.0, 705.0, 750.0, 755.0, 10.0) is False


def test_spans_close_exact_boundary_touch_counts_as_close():
    # Touching (not overlapping: overlap == 0), midpoints exactly
    # proximity_window_s apart (5 vs 15 -> diff 10) -> `<=`, inclusive.
    assert seam_dedup.spans_close(0.0, 10.0, 10.0, 20.0, 10.0) is True


# --------------------------------------------------------------------------- #
# in_seam_band
# --------------------------------------------------------------------------- #
def test_in_seam_band_highlight_fully_inside_band():
    assert seam_dedup.in_seam_band(700.0, 715.0, 675.0, 720.0) is True


def test_in_seam_band_highlight_straddling_band_edge():
    assert seam_dedup.in_seam_band(670.0, 680.0, 675.0, 720.0) is True  # overlaps [675,680]


def test_in_seam_band_highlight_entirely_before_band():
    assert seam_dedup.in_seam_band(100.0, 115.0, 675.0, 720.0) is False


def test_in_seam_band_highlight_entirely_after_band():
    assert seam_dedup.in_seam_band(800.0, 815.0, 675.0, 720.0) is False


def test_in_seam_band_highlight_touching_boundary_exactly_excluded():
    # end_s == seam_start -> zero-width intersection, excluded (`>` not `>=`).
    assert seam_dedup.in_seam_band(600.0, 675.0, 675.0, 720.0) is False


# --------------------------------------------------------------------------- #
# find_seam_duplicate
# --------------------------------------------------------------------------- #
def test_find_seam_duplicate_matches_class_compatible_close_clip():
    prior = [{"start_s": 700.0, "end_s": 715.0, "action_class": "submission_choke"}]
    candidate = {"start_s": 703.0, "end_s": 716.0, "action_class": "submission_arm_lock"}
    match = seam_dedup.find_seam_duplicate(candidate, prior)
    assert match == prior[0]


def test_find_seam_duplicate_no_match_returns_none():
    prior = [{"start_s": 700.0, "end_s": 715.0, "action_class": "submission_choke"}]
    candidate = {"start_s": 703.0, "end_s": 716.0, "action_class": "takedown_attempt"}
    assert seam_dedup.find_seam_duplicate(candidate, prior) is None


def test_find_seam_duplicate_empty_prior_list_returns_none():
    candidate = {"start_s": 703.0, "end_s": 716.0, "action_class": "submission_choke"}
    assert seam_dedup.find_seam_duplicate(candidate, []) is None


def test_find_seam_duplicate_never_consults_actor_or_player_id():
    """Core Brooks condition (dedup-bug-D): a real duplicate whose actor axis
    FLIPPED across the seam must still merge — actor/player_id is never part
    of the comparison."""
    prior = [{
        "start_s": 700.0, "end_s": 715.0, "action_class": "submission_choke",
        "player_id": "p1", "player_name": "Alice", "actor_sentinel": None,
    }]
    candidate = {
        "start_s": 703.0, "end_s": 716.0, "action_class": "submission_choke",
        "player_id": None, "player_name": None, "actor_sentinel": "contested",
    }
    match = seam_dedup.find_seam_duplicate(candidate, prior)
    assert match == prior[0]


def test_find_seam_duplicate_malformed_prior_clip_skipped_never_raises():
    prior = [{"start_s": "not-a-number", "end_s": 715.0, "action_class": "submission_choke"}]
    candidate = {"start_s": 703.0, "end_s": 716.0, "action_class": "submission_choke"}
    assert seam_dedup.find_seam_duplicate(candidate, prior) is None


def test_find_seam_duplicate_malformed_candidate_returns_none_never_raises():
    prior = [{"start_s": 700.0, "end_s": 715.0, "action_class": "submission_choke"}]
    candidate = {"start_s": None, "end_s": 716.0, "action_class": "submission_choke"}
    assert seam_dedup.find_seam_duplicate(candidate, prior) is None


def test_find_seam_duplicate_returns_first_match_in_order():
    prior = [
        {"start_s": 700.0, "end_s": 705.0, "action_class": "guard_pass"},
        {"start_s": 703.0, "end_s": 716.0, "action_class": "submission_choke"},
    ]
    candidate = {"start_s": 704.0, "end_s": 717.0, "action_class": "submission_arm_lock"}
    match = seam_dedup.find_seam_duplicate(candidate, prior)
    assert match == prior[1]


# --------------------------------------------------------------------------- #
# dedup_match_clips — 2026-07-26 CEO batched-publish re-scope, item 6:
# whole-match post-hoc selection pass (not per-chunk-incremental anymore).
# --------------------------------------------------------------------------- #
def _chunks_1500_720():
    # 1500s / 720s -> (0,720), (675,1440), (1395,1500) — matches
    # highlight_orchestrator._outer_chunks(1500.0, 720) exactly.
    return [(0.0, 720.0), (675.0, 1440.0), (1395.0, 1500.0)]


def test_dedup_match_clips_drops_seam_duplicate_keeps_earlier_chunk_version():
    clips_by_chunk = [
        [{"start_s": 700.0, "end_s": 715.0, "action_class": "submission_choke", "highlight_index": 1}],
        [{"start_s": 703.0, "end_s": 716.0, "action_class": "submission_arm_lock", "highlight_index": 1}],
        [],
    ]
    survivors, dropped = seam_dedup.dedup_match_clips(clips_by_chunk, _chunks_1500_720())
    assert dropped == 1
    assert len(survivors) == 1
    assert survivors[0]["action_class"] == "submission_choke"  # chunk 0's (earlier) version survives


def test_dedup_match_clips_no_duplicates_keeps_everything_in_chunk_order():
    clips_by_chunk = [
        [{"start_s": 100.0, "end_s": 110.0, "action_class": "guard_pass"}],
        [{"start_s": 800.0, "end_s": 810.0, "action_class": "takedown_attempt"}],
        [{"start_s": 1450.0, "end_s": 1460.0, "action_class": "sweep"}],
    ]
    survivors, dropped = seam_dedup.dedup_match_clips(clips_by_chunk, _chunks_1500_720())
    assert dropped == 0
    assert [c["action_class"] for c in survivors] == ["guard_pass", "takedown_attempt", "sweep"]


def test_dedup_match_clips_class_incompatible_seam_neighbors_both_survive():
    clips_by_chunk = [
        [{"start_s": 700.0, "end_s": 715.0, "action_class": "guard_pass"}],
        [{"start_s": 703.0, "end_s": 716.0, "action_class": "takedown_attempt"}],
    ]
    survivors, dropped = seam_dedup.dedup_match_clips(clips_by_chunk, [(0.0, 720.0), (675.0, 1000.0)])
    assert dropped == 0
    assert len(survivors) == 2


def test_dedup_match_clips_actor_never_consulted_still_dedupes_on_flip():
    """Core Brooks condition, end-to-end at the orchestration-facing API:
    a seam duplicate whose actor axis flipped between chunks still merges."""
    clips_by_chunk = [
        [{
            "start_s": 700.0, "end_s": 715.0, "action_class": "submission_choke",
            "player_id": "p1", "actor_sentinel": None,
        }],
        [{
            "start_s": 703.0, "end_s": 716.0, "action_class": "submission_choke",
            "player_id": None, "actor_sentinel": "contested",
        }],
    ]
    survivors, dropped = seam_dedup.dedup_match_clips(clips_by_chunk, [(0.0, 720.0), (675.0, 1000.0)])
    assert dropped == 1
    assert len(survivors) == 1
    assert survivors[0]["player_id"] == "p1"


def test_dedup_match_clips_empty_input_returns_empty():
    survivors, dropped = seam_dedup.dedup_match_clips([], [])
    assert survivors == []
    assert dropped == 0


def test_dedup_match_clips_single_chunk_no_seam_to_check():
    clips_by_chunk = [[{"start_s": 10.0, "end_s": 20.0, "action_class": "guard_pass"}]]
    survivors, dropped = seam_dedup.dedup_match_clips(clips_by_chunk, [(0.0, 100.0)])
    assert dropped == 0
    assert len(survivors) == 1


def test_dedup_match_clips_preserves_original_per_chunk_order_not_resorted_by_time():
    clips_by_chunk = [
        [
            {"start_s": 50.0, "end_s": 60.0, "action_class": "sweep", "highlight_index": "a"},
            {"start_s": 10.0, "end_s": 20.0, "action_class": "guard_pass", "highlight_index": "b"},
        ],
        [{"start_s": 1450.0, "end_s": 1460.0, "action_class": "takedown_attempt", "highlight_index": "c"}],
    ]
    survivors, _ = seam_dedup.dedup_match_clips(clips_by_chunk, _chunks_1500_720())
    assert [c["highlight_index"] for c in survivors] == ["a", "b", "c"]


def test_dedup_match_clips_three_way_seam_only_checks_adjacent_pairs():
    """A clip in chunk 2's leading seam band with chunk 1 must NOT be
    checked against chunk 0's clips at all (chunk 0 and chunk 2 don't
    share a seam band under a normal 720s/45s grid) — only genuinely
    adjacent chunk pairs are ever compared."""
    clips_by_chunk = [
        [{"start_s": 700.0, "end_s": 715.0, "action_class": "submission_choke"}],  # chunk 0 trailing
        [],  # chunk 1 has nothing near either of its own seams
        [{"start_s": 1400.0, "end_s": 1415.0, "action_class": "submission_choke"}],  # chunk 2 leading (own seam w/ chunk1)
    ]
    survivors, dropped = seam_dedup.dedup_match_clips(clips_by_chunk, _chunks_1500_720())
    assert dropped == 0  # chunk 2's clip is only checked against chunk 1 (empty), never chunk 0
    assert len(survivors) == 2
