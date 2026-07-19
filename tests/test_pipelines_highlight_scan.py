"""``service/pipelines/highlight_scan.py`` — PASS-1 pure helpers: response
schema, prompt content ($start_sec/$end_sec safe-substitution), and
deterministic bound repair (merge-under-min, CONTIGUOUS split-over-max — no
overlap, unlike ``chunk_segment``). All pure functions — no Gemini client, no
network.
"""
from __future__ import annotations

from google.genai import types

from service.pipelines import highlight_scan


# --------------------------------------------------------------------------- #
# Response schema
# --------------------------------------------------------------------------- #
def test_schema_top_level_envelope():
    schema = highlight_scan.build_highlight_schema()
    assert schema.type == types.Type.OBJECT
    assert schema.required == ["highlights"]
    highlight_item = schema.properties["highlights"].items
    # v2 (2026-07-18-highlight-scan-critique-analyze-v2-plan.md, "Node 1"):
    # description/reasoning are ADDITIVE required fields — see
    # test_schema_has_description_and_reasoning_axis below for the dedicated
    # regression pinning their content/type.
    assert set(highlight_item.required) == {"start_s", "end_s", "description", "reasoning"}


def test_schema_has_no_reason_enum_or_worth_analysis_axis():
    """Stage-1 contract: no ``reason`` ENUM, no ``worth_analysis``, no
    per-clip technique/position detail (unlike chunk_segment's PASS 1) —
    updated 2026-07-18 (v2 plan) to allow the new description/reasoning
    free-text fields, which are NOT the enum/worth_analysis this test
    guards against."""
    schema = highlight_scan.build_highlight_schema()
    highlight_item = schema.properties["highlights"].items
    assert set(highlight_item.properties) == {"start_s", "end_s", "description", "reasoning"}
    assert not highlight_item.properties["description"].enum
    assert not highlight_item.properties["reasoning"].enum
    assert "worth_analysis" not in highlight_item.properties
    assert "reason" not in highlight_item.properties  # no reason ENUM (distinct from the "reasoning" free-text field)


def test_schema_description_and_reasoning_are_free_text_strings():
    schema = highlight_scan.build_highlight_schema()
    highlight_item = schema.properties["highlights"].items
    assert highlight_item.properties["description"].type == types.Type.STRING
    assert highlight_item.properties["reasoning"].type == types.Type.STRING


def test_schema_start_end_are_numbers():
    schema = highlight_scan.build_highlight_schema()
    highlight_item = schema.properties["highlights"].items
    assert highlight_item.properties["start_s"].type == types.Type.NUMBER
    assert highlight_item.properties["end_s"].type == types.Type.NUMBER


# --------------------------------------------------------------------------- #
# Prompt content — system prompt (full override) + initial prompt ($-substituted)
# --------------------------------------------------------------------------- #
def test_resolve_system_prompt_defaults_to_bundled_text_when_none():
    assert highlight_scan.resolve_system_prompt(None) == highlight_scan.DEFAULT_SYSTEM_PROMPT


def test_resolve_system_prompt_defaults_to_bundled_text_when_whitespace_only():
    assert highlight_scan.resolve_system_prompt("   \n  ") == highlight_scan.DEFAULT_SYSTEM_PROMPT


def test_resolve_system_prompt_override_replaces_default_entirely():
    custom = "Only flag submissions, nothing else."
    assert highlight_scan.resolve_system_prompt(custom) == custom


def test_build_scan_prompt_default_substitutes_absolute_bounds():
    prompt = highlight_scan.build_scan_prompt(10, 100)
    assert "10" in prompt and "100" in prompt
    assert "$start_sec" not in prompt and "$end_sec" not in prompt  # substituted, not literal


def test_build_scan_prompt_custom_override_also_gets_substituted():
    """A playground-edited initial_prompt may still reference $start_sec/
    $end_sec — safe_substitute applies to the override too, not just the
    bundled default."""
    prompt = highlight_scan.build_scan_prompt(5, 55, initial_prompt="Scan from $start_sec to $end_sec only.")
    assert prompt == "Scan from 5 to 55 only."


def test_build_scan_prompt_custom_override_without_tokens_is_a_safe_noop():
    """safe_substitute never raises on a template with no recognized tokens —
    a playground edit that drops $start_sec/$end_sec is a valid (if
    less-precise) override, never a hard error."""
    prompt = highlight_scan.build_scan_prompt(5, 55, initial_prompt="Just find the highlights, no scope needed.")
    assert prompt == "Just find the highlights, no scope needed."


def test_build_scan_prompt_empty_string_normalizes_to_default():
    default_prompt = highlight_scan.build_scan_prompt(0, 30, initial_prompt=None)
    empty_override_prompt = highlight_scan.build_scan_prompt(0, 30, initial_prompt="   ")
    assert empty_override_prompt == default_prompt


# --------------------------------------------------------------------------- #
# Bound repair — merge sub-min into neighbor (same bias as chunk_segment)
# --------------------------------------------------------------------------- #
def test_repair_merges_sub_min_highlight_into_previous_neighbor():
    raw = [
        {"start_s": 0.0, "end_s": 20.0},
        {"start_s": 20.0, "end_s": 21.5},  # 1.5s < min 3.0s
        {"start_s": 21.5, "end_s": 40.0},
    ]
    repaired, notes = highlight_scan.repair_highlight_bounds(raw, min_highlight_s=3.0, max_highlight_s=60.0)
    assert len(repaired) == 2
    # The sub-min middle entry [20-21.5s] merges INTO its previous neighbor
    # [0-20s], extending it to [0-21.5s]; the trailing entry [21.5-40s] is
    # untouched (it is not itself sub-min).
    assert repaired[0]["start_s"] == 0.0 and repaired[0]["end_s"] == 21.5
    assert repaired[1]["start_s"] == 21.5 and repaired[1]["end_s"] == 40.0
    assert notes and "merged sub-min highlight" in notes[0]


def test_repair_merges_leading_sub_min_highlight_forward():
    raw = [
        {"start_s": 0.0, "end_s": 1.0},  # 1.0s < min 3.0s, no earlier neighbor
        {"start_s": 1.0, "end_s": 30.0},
    ]
    repaired, notes = highlight_scan.repair_highlight_bounds(raw, min_highlight_s=3.0, max_highlight_s=60.0)
    assert len(repaired) == 1
    assert repaired[0]["start_s"] == 0.0 and repaired[0]["end_s"] == 30.0
    assert any("leading sub-min" in n for n in notes)


def test_repair_drops_malformed_entries_never_fabricates():
    raw = [
        {"start_s": 0.0, "end_s": 20.0},
        {"actor": "top"},  # missing start_s/end_s
        {"start_s": "not-a-number", "end_s": 30.0},
        {"start_s": 40.0, "end_s": 30.0},  # non-positive duration
    ]
    repaired, _notes = highlight_scan.repair_highlight_bounds(raw, min_highlight_s=1.0, max_highlight_s=60.0)
    assert len(repaired) == 1
    assert repaired[0]["start_s"] == 0.0 and repaired[0]["end_s"] == 20.0


# --------------------------------------------------------------------------- #
# Scope clamp/drop (live-bug fix, 2026-07-18) — PASS 1 can emit a highlight
# entirely outside the requested scope (real repro: scope=40-85s, PASS 1
# returned index 3 start_s=111/end_s=124), which must never flow untouched
# into highlight_analyze_node's preroll/postroll window math (which can
# invert -> a genuine Gemini 500). scope_start/scope_end default to None
# (skip entirely) so every OTHER test in this file above, which doesn't pass
# scope, is completely unaffected by this addition.
# --------------------------------------------------------------------------- #
def test_repair_drops_highlight_entirely_outside_scope():
    raw = [
        {"start_s": 50.0, "end_s": 60.0},  # inside scope
        {"start_s": 111.0, "end_s": 124.0},  # entirely outside scope [40, 85] — real repro numbers
    ]
    repaired, notes = highlight_scan.repair_highlight_bounds(
        raw, min_highlight_s=1.0, max_highlight_s=60.0, scope_start=40.0, scope_end=85.0,
    )
    assert len(repaired) == 1
    assert repaired[0]["start_s"] == 50.0 and repaired[0]["end_s"] == 60.0


def test_repair_drops_highlight_touching_scope_boundary_zero_overlap():
    """Touching the scope boundary exactly is zero real overlap — dropped,
    not kept as a zero-duration clamp (same convention as
    executors._clip_events_to_highlight_bounds's boundary-touching case)."""
    raw = [{"start_s": 85.0, "end_s": 95.0}]  # starts exactly at scope_end
    repaired, _notes = highlight_scan.repair_highlight_bounds(
        raw, min_highlight_s=1.0, max_highlight_s=60.0, scope_start=40.0, scope_end=85.0,
    )
    assert repaired == []


def test_repair_clamps_highlight_straddling_scope_start():
    raw = [{"start_s": 35.0, "end_s": 50.0}]  # starts before scope, overlaps into it
    repaired, _notes = highlight_scan.repair_highlight_bounds(
        raw, min_highlight_s=1.0, max_highlight_s=60.0, scope_start=40.0, scope_end=85.0,
    )
    assert len(repaired) == 1
    assert repaired[0]["start_s"] == 40.0  # clamped up
    assert repaired[0]["end_s"] == 50.0  # unaffected, already inside scope


def test_repair_clamps_highlight_straddling_scope_end():
    raw = [{"start_s": 80.0, "end_s": 95.0}]  # continues past scope end
    repaired, _notes = highlight_scan.repair_highlight_bounds(
        raw, min_highlight_s=1.0, max_highlight_s=60.0, scope_start=40.0, scope_end=85.0,
    )
    assert len(repaired) == 1
    assert repaired[0]["start_s"] == 80.0
    assert repaired[0]["end_s"] == 85.0  # clamped down


def test_repair_scope_none_default_skips_clamping_entirely():
    """Backward-compatibility guard: omitting scope_start/scope_end (the
    default) must reproduce EXACTLY the pre-fix behavior — a highlight far
    outside any reasonable scope survives untouched when no scope is given."""
    raw = [{"start_s": 111.0, "end_s": 124.0}]
    repaired, _notes = highlight_scan.repair_highlight_bounds(raw, min_highlight_s=1.0, max_highlight_s=60.0)
    assert repaired[0]["start_s"] == 111.0 and repaired[0]["end_s"] == 124.0


def test_finalize_highlights_drops_out_of_scope_and_clamps_straddling():
    """End-to-end (repair + index assignment) with the real repro numbers."""
    raw = [
        {"start_s": 50.0, "end_s": 60.0},
        {"start_s": 111.0, "end_s": 124.0},  # dropped
        {"start_s": 80.0, "end_s": 95.0},  # clamped to end at 85
    ]
    out = highlight_scan.finalize_highlights(
        raw, min_highlight_s=1.0, max_highlight_s=60.0, scope_start=40.0, scope_end=85.0,
    )
    assert len(out) == 2
    assert [h["index"] for h in out] == [1, 2]
    assert out[0]["start_s"] == 50.0 and out[0]["end_s"] == 60.0
    assert out[1]["start_s"] == 80.0 and out[1]["end_s"] == 85.0


# --------------------------------------------------------------------------- #
# Bound repair — split over-max into CONTIGUOUS pieces (NO overlap — load-
# bearing for the downstream event-clipping invariant, see module docstring).
# --------------------------------------------------------------------------- #
def test_repair_splits_over_max_highlight_into_contiguous_pieces_no_overlap():
    raw = [{"start_s": 0.0, "end_s": 70.0}]  # 70s > max 30.0s
    repaired, notes = highlight_scan.repair_highlight_bounds(raw, min_highlight_s=1.0, max_highlight_s=30.0)
    assert len(repaired) == 3
    assert repaired[0]["start_s"] == 0.0 and repaired[0]["end_s"] == 30.0
    assert repaired[1]["start_s"] == 30.0 and repaired[1]["end_s"] == 60.0
    assert repaired[2]["start_s"] == 60.0 and repaired[2]["end_s"] == 70.0
    # CONTIGUOUS, not overlapping: piece[i].end_s == piece[i+1].start_s exactly.
    for a, b in zip(repaired, repaired[1:]):
        assert a["end_s"] == b["start_s"]
    assert any("CONTIGUOUS" in n for n in notes)


def test_repair_split_pieces_tag_synthetic_vs_genuine_edges():
    """Evaluator CHANGES-REQUIRED item 2 fix: only the FIRST piece's start_s
    and the LAST piece's end_s are genuine (PASS-1-reported) edges — every
    edge introduced by the split itself is a manufactured seam, tagged
    synthetic so executors.highlight_analyze_node can clamp preroll/postroll
    to zero across it."""
    raw = [{"start_s": 0.0, "end_s": 70.0}]
    repaired, _notes = highlight_scan.repair_highlight_bounds(raw, min_highlight_s=1.0, max_highlight_s=30.0)
    assert len(repaired) == 3
    assert repaired[0]["start_is_synthetic"] is False  # genuine original start
    assert repaired[0]["end_is_synthetic"] is True  # manufactured seam
    assert repaired[1]["start_is_synthetic"] is True  # manufactured seam (shared with piece 0)
    assert repaired[1]["end_is_synthetic"] is True  # manufactured seam (shared with piece 2)
    assert repaired[2]["start_is_synthetic"] is True  # manufactured seam (shared with piece 1)
    assert repaired[2]["end_is_synthetic"] is False  # genuine original end


def test_repair_non_split_highlight_has_no_synthetic_edges():
    raw = [{"start_s": 10.0, "end_s": 20.0}]  # well under max_highlight_s, never split
    repaired, _notes = highlight_scan.repair_highlight_bounds(raw, min_highlight_s=1.0, max_highlight_s=60.0)
    assert repaired[0]["start_is_synthetic"] is False
    assert repaired[0]["end_is_synthetic"] is False


def test_repair_split_default_overlap_constant_is_zero():
    """Regression guard: HIGHLIGHT_SPLIT_OVERLAP_S must stay 0.0 — a nonzero
    default would silently reintroduce double-counting risk across split
    highlight pieces (see module docstring / executors.highlight_analyze_node)."""
    assert highlight_scan.HIGHLIGHT_SPLIT_OVERLAP_S == 0.0


def test_repair_split_no_pieces_exceed_max():
    raw = [{"start_s": 0.0, "end_s": 145.0}]
    repaired, _notes = highlight_scan.repair_highlight_bounds(raw, min_highlight_s=1.0, max_highlight_s=30.0)
    for piece in repaired:
        assert (piece["end_s"] - piece["start_s"]) <= 30.0 + 1e-9


def test_repair_split_pieces_inherit_parent_description_and_reasoning():
    """v2 (2026-07-18 plan): a split piece is a mechanical cut of ONE
    continuous highlight, not N distinct judgments — every piece inherits
    the SAME parent description/reasoning."""
    raw = [{"start_s": 0.0, "end_s": 70.0, "description": "long scramble", "reasoning": "sustained exchange"}]
    repaired, _notes = highlight_scan.repair_highlight_bounds(raw, min_highlight_s=1.0, max_highlight_s=30.0)
    assert len(repaired) == 3
    for piece in repaired:
        assert piece["description"] == "long scramble"
        assert piece["reasoning"] == "sustained exchange"


def test_repair_split_with_explicit_nonzero_overlap_param_still_works():
    """The overlap behavior itself remains directly testable via the
    parameter, even though production callers never pass a nonzero value."""
    raw = [{"start_s": 0.0, "end_s": 70.0}]
    repaired, _notes = highlight_scan.repair_highlight_bounds(
        raw, min_highlight_s=1.0, max_highlight_s=30.0, split_overlap_s=5.0,
    )
    assert repaired[1]["start_s"] == 25.0  # 30 - 5 overlap


# --------------------------------------------------------------------------- #
# finalize_highlights — repair + 1-based index, the exact highlight_map shape
# --------------------------------------------------------------------------- #
def test_finalize_highlights_assigns_1_based_index_and_no_reason_key():
    raw = [
        {"start_s": 10.0, "end_s": 20.0, "description": "grip fight", "reasoning": "live exchange"},
        {"start_s": 30.0, "end_s": 40.0, "description": "sweep attempt", "reasoning": "scramble"},
    ]
    out = highlight_scan.finalize_highlights(raw, min_highlight_s=1.0, max_highlight_s=60.0)
    assert [h["index"] for h in out] == [1, 2]
    for h in out:
        # v2 (2026-07-18 plan): description/reasoning are additive pass-through keys.
        assert set(h) == {
            "index", "start_s", "end_s", "description", "reasoning",
            "adjustment", "start_is_synthetic", "end_is_synthetic",
        }
        assert h["start_is_synthetic"] is False
        assert h["end_is_synthetic"] is False
    assert out[0]["description"] == "grip fight" and out[0]["reasoning"] == "live exchange"
    assert out[1]["description"] == "sweep attempt" and out[1]["reasoning"] == "scramble"


def test_finalize_highlights_description_reasoning_none_when_absent_never_fabricated():
    """A raw highlight missing description/reasoning (e.g. a test double, or
    a malformed real response) gets None, never a fabricated string."""
    raw = [{"start_s": 10.0, "end_s": 20.0}]
    out = highlight_scan.finalize_highlights(raw, min_highlight_s=1.0, max_highlight_s=60.0)
    assert out[0]["description"] is None
    assert out[0]["reasoning"] is None
