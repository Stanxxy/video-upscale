"""The two MANDATORY highlight-eval baselines (frozen contract §C, LeCun):
"reported alongside the real F1, or the number is unfalsifiable" — a headline
F1 with no null-model comparison invites over-reading a middling Gemini score
as impressive when a trivial baseline would have scored nearly as well (or
the reverse). Both baselines are consumed by the SAME unchanged scoring path
(``highlight_scoring.score_highlights`` -> ``matcher.match_all`` ->
``metrics.summarize``) as the real Gemini predictions — they are just a
different ``pred_by_match`` generator, never a parallel scoring
implementation.

**Baseline 1 — whole-match span.** "Predict the entire match window is one
highlight." The null model with perfect recall by construction (every GT
highlight point falls inside a span covering the whole match) and precision
bounded only by how many DISTINCT GT highlights compete for that one span
under the one-to-one Hungarian assignment (§5.2's "competed away" FN) —
this baseline's real value is exposing how much of a real system's precision
edge over "just watch the whole match" actually exists.

**Baseline 2 — uniform span.** "Predict a highlight every N seconds,"
tuned so the average number of predicted spans per match matches the real
corpus's own highlight density (frozen contract: "span-count ≈ 2.16/match" —
557 highlights / 258 matches from ``build_highlight_gold.py``'s real
manifest). This is the "same guess-rate, zero video understanding" null
model — a real system beating this baseline is the more informative bar,
since it controls for "predicting *some* fixed number of things, regardless
of content" rather than for "predicting everything."

``N`` is derived PER MATCH (``duration_s / target_spans_per_match``) rather
than as one global across-corpus constant: this eval surface tests one match
at a time (frozen contract's "over ONE ADCC corpus video" framing, and
section F's single-select match picker), so a per-match derivation is what
makes the baseline meaningful for whichever single match is under test,
while still averaging to the frozen ``TARGET_SPANS_PER_MATCH`` rate by
construction for any match. This is a documented interpretation of the
contract's literal "N seconds" phrasing, not a silent deviation — flagged
in the implementation report.
"""
from __future__ import annotations

__all__ = ["TARGET_SPANS_PER_MATCH", "whole_match_baseline", "uniform_span_baseline"]

# 557 real highlights / 258 real matches (build_highlight_gold.py's manifest,
# INCLUDE_TAKEDOWNS=False) — frozen contract §C: "N tuned so span-count ≈
# 2.16/match".
TARGET_SPANS_PER_MATCH: float = 2.16


def whole_match_baseline(match_windows: dict[str, tuple[float, float]]) -> dict[str, list[dict]]:
    """Baseline 1: one span, ``[window_start_s, window_end_s]``, per
    match_id. ``match_windows`` maps ``match_id -> (start_s, end_s)`` — the
    SAME scope-policy window a real highlight-discovery Gemini call is scoped
    to (``scope_policy.single_match_scope``/``multi_match_scope``), never a
    re-derived or hardcoded window, so this baseline is scored against
    exactly the same clock the real predictions and the GT share."""
    return {
        match_id: [{"start_s": start_s, "end_s": end_s, "action_class": "highlight"}]
        for match_id, (start_s, end_s) in match_windows.items()
    }


def uniform_span_baseline(
    match_windows: dict[str, tuple[float, float]],
    *,
    target_spans_per_match: float = TARGET_SPANS_PER_MATCH,
) -> dict[str, list[dict]]:
    """Baseline 2: ``round(target_spans_per_match)`` evenly-spaced POINT
    predictions (``start_s == end_s``) per match_id, centered within
    ``round(target_spans_per_match)`` equal-width sub-intervals of the match
    window. A non-positive window duration (malformed input, never observed
    for a real ``ScopeWindow``) degrades to an empty prediction list for that
    match_id rather than raising — this is a NULL-MODEL baseline generator,
    not a run-gating path, so a defensive empty-list is the right failure
    mode here (mirrors ``match_catalog._resolve_window``'s "flag, don't
    raise, for a display/reference-only path" discipline)."""
    span_count = max(1, round(target_spans_per_match))
    pred_by_match: dict[str, list[dict]] = {}
    for match_id, (start_s, end_s) in match_windows.items():
        duration = end_s - start_s
        if duration <= 0:
            pred_by_match[match_id] = []
            continue
        step = duration / span_count
        points = [start_s + step * (i + 0.5) for i in range(span_count)]
        pred_by_match[match_id] = [{"start_s": t, "end_s": t, "action_class": "highlight"} for t in points]
    return pred_by_match
