"""``service/pipelines/chunk_segment.py`` — PASS-1 pure helpers: response
schema, prompt content, deterministic bound repair (merge-under-min,
overlap-split-over-max), ``worth_analysis`` derivation, and the v1 hard
scope cap check. All pure functions — no Gemini client, no network.
"""
from __future__ import annotations

from google.genai import types

from service.pipelines import chunk_segment


# --------------------------------------------------------------------------- #
# worth_analysis derivation — never asked of Gemini directly
# --------------------------------------------------------------------------- #
def test_worth_analysis_true_only_for_match_action():
    assert chunk_segment.worth_analysis("match_action") is True
    for reason in ("restart_reset", "stoppage", "pre_match", "broadcast"):
        assert chunk_segment.worth_analysis(reason) is False


# --------------------------------------------------------------------------- #
# Response schema
# --------------------------------------------------------------------------- #
def test_schema_top_level_envelope():
    schema = chunk_segment.build_chunk_segment_schema()
    assert schema.type == types.Type.OBJECT
    assert schema.required == ["chunks"]
    chunk_item = schema.properties["chunks"].items
    assert set(chunk_item.required) == {"start_s", "end_s", "reason"}


def test_schema_reason_is_hard_enum_of_exact_5_values():
    schema = chunk_segment.build_chunk_segment_schema()
    chunk_item = schema.properties["chunks"].items
    reason = chunk_item.properties["reason"]
    assert reason.type == types.Type.STRING
    assert list(reason.enum) == chunk_segment.REASON_VALUES
    assert chunk_segment.REASON_VALUES == [
        "match_action", "restart_reset", "stoppage", "pre_match", "broadcast",
    ]


def test_schema_start_end_are_numbers_not_frames():
    schema = chunk_segment.build_chunk_segment_schema()
    chunk_item = schema.properties["chunks"].items
    assert chunk_item.properties["start_s"].type == types.Type.NUMBER
    assert chunk_item.properties["end_s"].type == types.Type.NUMBER
    assert "start_frame" not in chunk_item.properties


# --------------------------------------------------------------------------- #
# Prompt content
# --------------------------------------------------------------------------- #
def test_prompt_bakes_in_skip_broadcast_never_skip_live_rule():
    prompt = chunk_segment.build_segment_prompt(10, 100)
    assert "skip broadcast filler" in prompt
    assert "NEVER skip live mat time" in prompt


def test_prompt_bakes_in_clean_cut_point_guidance():
    prompt = chunk_segment.build_segment_prompt(0, 60)
    assert "static for at least" in prompt
    assert "scramble is still resolving" in prompt


def test_prompt_demands_absolute_seconds_in_scope_bounds():
    prompt = chunk_segment.build_segment_prompt(30, 90)
    assert "[30, 90]" in prompt
    assert "ABSOLUTE" in prompt


# --------------------------------------------------------------------------- #
# Bound-repair — merge sub-min chunks into a neighbor
# --------------------------------------------------------------------------- #
def test_repair_merges_sub_min_chunk_into_previous_neighbor():
    raw = [
        {"start_s": 0, "end_s": 20, "reason": "match_action"},
        {"start_s": 20, "end_s": 22, "reason": "restart_reset"},  # 2s < min 5s
        {"start_s": 22, "end_s": 50, "reason": "match_action"},
    ]
    repaired, notes = chunk_segment.repair_chunk_bounds(raw, min_chunk_s=5, max_chunk_s=60)
    assert len(repaired) == 2  # the sub-min chunk merged into its (previous) neighbor only
    assert repaired[0]["start_s"] == 0
    assert repaired[0]["end_s"] == 22  # first chunk absorbed the sub-min restart_reset
    assert repaired[1]["start_s"] == 22
    assert repaired[1]["end_s"] == 50
    assert notes  # a note was recorded


def test_repair_merge_prefers_match_action_reason_never_skip_live_mat_time():
    """When a match_action chunk merges with a sub-min non-match_action
    neighbor, the merged reason stays match_action — 'never skip live mat
    time' bias."""
    raw = [
        {"start_s": 0, "end_s": 20, "reason": "match_action"},
        {"start_s": 20, "end_s": 22, "reason": "broadcast"},  # sub-min, lower priority
    ]
    repaired, _ = chunk_segment.repair_chunk_bounds(raw, min_chunk_s=5, max_chunk_s=60)
    assert len(repaired) == 1
    assert repaired[0]["reason"] == "match_action"


def test_repair_merges_leading_sub_min_chunk_forward():
    raw = [
        {"start_s": 0, "end_s": 2, "reason": "pre_match"},  # sub-min, no earlier neighbor
        {"start_s": 2, "end_s": 40, "reason": "match_action"},
    ]
    repaired, notes = chunk_segment.repair_chunk_bounds(raw, min_chunk_s=5, max_chunk_s=60)
    assert len(repaired) == 1
    assert repaired[0]["start_s"] == 0
    assert repaired[0]["end_s"] == 40
    assert notes


def test_repair_leaves_well_formed_chunks_untouched():
    raw = [
        {"start_s": 0, "end_s": 30, "reason": "match_action"},
        {"start_s": 30, "end_s": 60, "reason": "broadcast"},
    ]
    repaired, notes = chunk_segment.repair_chunk_bounds(raw, min_chunk_s=5, max_chunk_s=60)
    assert repaired == raw
    assert notes == []


# --------------------------------------------------------------------------- #
# Bound-repair — split over-max chunks WITH overlap (never a naked cut)
# --------------------------------------------------------------------------- #
def test_repair_splits_over_max_chunk_with_overlap():
    raw = [{"start_s": 0, "end_s": 130, "reason": "match_action"}]  # 130s > max 60s
    repaired, notes = chunk_segment.repair_chunk_bounds(
        raw, min_chunk_s=5, max_chunk_s=60, split_overlap_s=5,
    )
    assert len(repaired) == 3  # 0-60, 55-115, 110-130
    assert repaired[0]["start_s"] == 0 and repaired[0]["end_s"] == 60
    assert repaired[1]["start_s"] == 55 and repaired[1]["end_s"] == 115  # 5s overlap into prior piece
    assert repaired[2]["start_s"] == 110 and repaired[2]["end_s"] == 130
    assert all(c["reason"] == "match_action" for c in repaired)
    assert notes  # adjustment recorded


def test_repair_split_pieces_cover_the_full_original_span_with_no_gap():
    raw = [{"start_s": 10, "end_s": 100, "reason": "match_action"}]
    repaired, _ = chunk_segment.repair_chunk_bounds(raw, min_chunk_s=5, max_chunk_s=30, split_overlap_s=5)
    assert repaired[0]["start_s"] == 10
    assert repaired[-1]["end_s"] == 100
    # No gap: each next piece starts at or before the previous piece's end.
    for a, b in zip(repaired, repaired[1:]):
        assert b["start_s"] <= a["end_s"]


def test_repair_never_naked_cuts_exactly_at_max_chunk_s_boundary():
    """A naked cut at exactly max_chunk_s (no overlap) would guillotine a
    submission/transition mid-action — every split boundary must carry the
    overlap."""
    raw = [{"start_s": 0, "end_s": 61, "reason": "match_action"}]
    repaired, _ = chunk_segment.repair_chunk_bounds(raw, min_chunk_s=5, max_chunk_s=60, split_overlap_s=5)
    assert len(repaired) == 2
    assert repaired[1]["start_s"] == 55  # not 60 — carries the overlap


# --------------------------------------------------------------------------- #
# Malformed raw entries — dropped, never fabricated
# --------------------------------------------------------------------------- #
def test_repair_drops_malformed_entries_never_fabricates():
    raw = [
        {"start_s": 0, "end_s": 30, "reason": "match_action"},
        {"start_s": "oops", "end_s": 40, "reason": "match_action"},  # non-numeric
        {"start_s": 40, "end_s": 50},  # missing reason
        {"start_s": 50, "end_s": 60, "reason": "not-a-real-reason"},  # off-enum
        {"start_s": 60, "end_s": 55, "reason": "match_action"},  # end < start
    ]
    repaired, _ = chunk_segment.repair_chunk_bounds(raw, min_chunk_s=5, max_chunk_s=60)
    assert len(repaired) == 1
    assert repaired[0]["start_s"] == 0 and repaired[0]["end_s"] == 30


# --------------------------------------------------------------------------- #
# finalize_chunks — indexing, worth_analysis derivation, analyze_all override
# --------------------------------------------------------------------------- #
def test_finalize_chunks_assigns_1_based_index_and_derives_worth_analysis():
    raw = [
        {"start_s": 0, "end_s": 30, "reason": "match_action"},
        {"start_s": 30, "end_s": 60, "reason": "broadcast"},
    ]
    chunks = chunk_segment.finalize_chunks(raw, min_chunk_s=5, max_chunk_s=90, analyze_all=False)
    assert [c["index"] for c in chunks] == [1, 2]
    assert chunks[0]["worth_analysis"] is True
    assert chunks[1]["worth_analysis"] is False


def test_finalize_chunks_analyze_all_forces_every_chunk_worthy():
    raw = [
        {"start_s": 0, "end_s": 30, "reason": "broadcast"},
        {"start_s": 30, "end_s": 60, "reason": "pre_match"},
    ]
    chunks = chunk_segment.finalize_chunks(raw, min_chunk_s=5, max_chunk_s=90, analyze_all=True)
    assert all(c["worth_analysis"] is True for c in chunks)
    # reason itself is untouched by the ablation override — only the derived
    # worth_analysis flag is forced.
    assert chunks[0]["reason"] == "broadcast"


def test_finalize_chunks_carries_adjustment_notes_through():
    raw = [
        {"start_s": 0, "end_s": 20, "reason": "match_action"},
        {"start_s": 20, "end_s": 22, "reason": "restart_reset"},
    ]
    chunks = chunk_segment.finalize_chunks(raw, min_chunk_s=5, max_chunk_s=60, analyze_all=False)
    assert len(chunks) == 1
    assert chunks[0]["adjustment"] is not None


def test_finalize_chunks_no_adjustment_is_none_not_empty_string():
    raw = [{"start_s": 0, "end_s": 30, "reason": "match_action"}]
    chunks = chunk_segment.finalize_chunks(raw, min_chunk_s=5, max_chunk_s=60, analyze_all=False)
    assert chunks[0]["adjustment"] is None


# --------------------------------------------------------------------------- #
# v1 hard scope cap
# --------------------------------------------------------------------------- #
def test_check_scope_cap_within_bound_returns_none():
    assert chunk_segment.check_scope_cap(300) is None
    assert chunk_segment.check_scope_cap(chunk_segment.MAX_SCOPE_SEC) is None


def test_check_scope_cap_over_bound_returns_actionable_message():
    msg = chunk_segment.check_scope_cap(chunk_segment.MAX_SCOPE_SEC + 1)
    assert msg is not None
    assert "exceeds" in msg
    assert "Narrow" in msg
