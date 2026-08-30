"""Highlight-Eval scoring (frozen contract §C) — reuses ``matcher.py`` 100%
UNCHANGED and ``metrics.summarize`` verbatim. This module's only job is the
highlight-specific glue around those two: loading the provenanced GT artifact
(``eval/adcc/data_prep/build_highlight_gold.py``'s output, never recomputing
the predicate here) and the ``filter_to_highlights()`` tagging step that makes
``matcher.match_one``'s class-compatibility check degrade to a no-op.

**Why the class check degrades to a no-op**: every GT highlight event gets a
singleton ``compat = frozenset({"highlight"})`` attached (below); every
Gemini highlight-discovery prediction gets its ``action_class`` overwritten
to ``"highlight"`` (``filter_to_highlights``). ``matcher.match_one``'s
``class_ok = p["action_class"] in g["compat"]`` is therefore always
``True`` — matching becomes pure temporal point-in-span + tau_t Hungarian
one-to-one per ``match_id``, exactly as the frozen contract specifies. Filter
BEFORE matching, never after (mirrors ``metrics.filter_scoreable``'s
discipline — a filter applied after ``match_all`` would create phantom
FN/FP, same failure mode ``filter_scoreable``'s docstring warns about).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from eval.adcc.score.matcher import MatchOutcome, match_all
from eval.adcc.score.metrics import summarize

__all__ = [
    "HIGHLIGHT_CLASS",
    "load_highlight_gold_by_match",
    "filter_to_highlights",
    "score_highlights",
    "render_timeline_marks",
    "effective_pred_events",
    "original_pred_events",
    "mean_abs_center_error",
]

# The one class label every GT highlight/Gemini prediction is tagged with —
# never a real taxonomy class (§C: "the class check degrades to a no-op").
HIGHLIGHT_CLASS = "highlight"


def load_highlight_gold_by_match(highlight_gold_path: Path) -> dict[str, list[dict]]:
    """Loads ``gold_snapshot/highlight_gold.jsonl`` (the ONLY legal source of
    highlight GT — never recompute ``is_highlight`` here, per
    ``build_highlight_gold.py``'s module docstring) and attaches the
    singleton ``compat`` set every event needs to satisfy ``matcher.py``'s
    event-dict contract. Every record's ``match_id`` is present in the
    returned dict even when its ``events`` list is empty (the ~17
    zero-highlight matches) — an empty gold list for a match_id is a real,
    scoreable state (any prediction under that match_id is then 100% FP),
    never absent from the dict entirely."""
    gold_by_match: dict[str, list[dict]] = {}
    with open(highlight_gold_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gold_by_match[rec["match_id"]] = [
                {**e, "match_id": rec["match_id"], "compat": frozenset({HIGHLIGHT_CLASS})}
                for e in rec["events"]
            ]
    return gold_by_match


def filter_to_highlights(pred_by_match: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """The ``filter_to_highlights()`` analog the frozen contract calls for
    (§C): tags every prediction event's ``action_class`` as
    ``"highlight"`` — a Gemini highlight-discovery call has no notion of a
    real taxonomy class in the first place (it was only ever asked "is this a
    highlight"), so this is a pure relabel, not a drop-some-keep-some filter
    like ``metrics.filter_scoreable``. Still named/positioned as a
    filter-before-match step (mirrors that module's discipline) because a
    caller MUST run this before ``match_all``, never retrofit the tag onto an
    already-matched result — the class-compat no-op only holds if every
    prediction carries this tag going INTO the Hungarian assignment."""
    return {
        match_id: [{**p, "match_id": match_id, "action_class": HIGHLIGHT_CLASS} for p in events]
        for match_id, events in pred_by_match.items()
    }


def score_highlights(
    gold_by_match: dict[str, list[dict]], pred_by_match_raw: dict[str, list[dict]], tau_t: float,
) -> dict:
    """Runs the UNCHANGED ``matcher.match_all`` + ``metrics.summarize`` over
    the highlight-tagged gold/pred sets.

    ``gold_by_match`` must already be the ``compat``-tagged output of
    ``load_highlight_gold_by_match`` (never re-tagged here — this function
    trusts its caller loaded the real provenanced artifact, it does not
    re-derive or re-validate the predicate). ``pred_by_match_raw`` is the
    RAW (untagged) highlight-discovery output — ``start_s``/``end_s`` pairs
    with no ``action_class`` yet; this function is the one and only place
    that calls ``filter_to_highlights`` on it, so a caller never has to
    remember the filter-then-match ordering itself.

    Returns ``{"summary": <metrics.summarize's dict>, "outcomes": <per-match
    MatchOutcome dict>}`` — ``outcomes`` is what ``render_timeline_marks``
    below re-slices for the per-mark client detail; ``summary`` is the P/R/F1
    + FN/FP-subtype breakdown card."""
    pred_by_match = filter_to_highlights(pred_by_match_raw)
    outcomes = match_all(gold_by_match, pred_by_match, tau_t)
    summary = summarize(outcomes, gold_by_match, pred_by_match)
    return {"summary": summary, "outcomes": outcomes, "gold_by_match": gold_by_match, "pred_by_match": pred_by_match}


def effective_pred_events(enriched_highlights: list[dict]) -> list[dict]:
    """Timestamp critique/refinement (frozen contract §G, LeCun's scoring
    policy): derives the EFFECTIVE (scoreable) ``{start_s, end_s}`` per
    persisted highlight — ``refined_*`` when ``refined=True``, else
    ``original_*``.

    A highlight with ``rejected=True`` (AC1-6's diagnostic-only field) is
    SKIPPED entirely — never falls back to ``original_start_s``/``end_s``.
    This closes the root Bug C (an out-of-window discovery timestamp,
    persisted with ``refined=False``, was previously scored via the
    ``original_*`` fallback with no bounds check at all). NOT 1:1
    order-preserving with ``enriched_highlights`` when any entry is
    rejected — callers needing index correspondence (e.g. the caller of
    ``render_timeline_marks``) must apply the SAME ``not h.get("rejected")``
    filter to ``enriched_highlights`` before zipping with this function's
    output — see ``dashboard_app.py``'s ``/highlight/score`` handler
    (Task 5).

    Tolerates a LEGACY persisted highlight that predates this increment
    (the flat ``{start_s, end_s, reason}`` shape, no ``original_start_s``/
    ``refined``/``rejected`` keys at all) by falling back to its own
    ``start_s``/``end_s`` directly — never crashes on an older persisted
    file, and never filters absences of ``rejected`` (only ``rejected=True``
    is skipped)."""
    events = []
    for h in enriched_highlights:
        if h.get("rejected"):
            continue
        if "original_start_s" in h:
            if h.get("refined"):
                start_s, end_s = h["refined_start_s"], h["refined_end_s"]
            else:
                start_s, end_s = h["original_start_s"], h["original_end_s"]
        else:
            start_s, end_s = h["start_s"], h["end_s"]
        events.append({"start_s": start_s, "end_s": end_s})
    return events


def original_pred_events(enriched_highlights: list[dict]) -> list[dict]:
    """The ORIGINAL (pre-refinement) discovery ``{start_s, end_s}`` per
    persisted highlight — same ``rejected=True`` skip as
    ``effective_pred_events`` (diagnostic-only rows never enter the
    before/after comparison), same legacy fallback otherwise. Exists so
    LeCun's before/after 3-way comparison (original-only P/R/F1 vs.
    effective P/R/F1) can be computed directly from ONE persisted file,
    never a second Gemini run. NOT 1:1 with ``enriched_highlights`` when
    any entry is rejected — same caller filter obligation as
    ``effective_pred_events``."""
    events = []
    for h in enriched_highlights:
        if h.get("rejected"):
            continue
        if "original_start_s" in h:
            start_s, end_s = h["original_start_s"], h["original_end_s"]
        else:
            start_s, end_s = h["start_s"], h["end_s"]
        events.append({"start_s": start_s, "end_s": end_s})
    return events


def mean_abs_center_error(
    gold_events: list[dict], pred_events: list[dict], outcome: MatchOutcome,
) -> Optional[float]:
    """A small LeCun-requested diagnostic (frozen contract §G): mean
    ``|pred_center - gold_t|`` over TP pairs only (the only pairs with a
    defined gold<->pred correspondence — mirrors ``metrics.
    bootstrap_match_ci``'s "never invent a correspondence" discipline).
    Never recomputes matching — purely re-slices an ALREADY-DECIDED
    ``MatchOutcome``, same pattern as ``render_timeline_marks``. Lets a
    caller compute this for BOTH ``original_pred_events`` and
    ``effective_pred_events`` outputs (run this function twice, once per
    prediction set, against outcomes computed from each) to get LeCun's
    before/after center-accuracy comparison — no UI wiring, just a
    runnable helper."""
    diffs = []
    for gi, pi in outcome.tp:
        gold_t = gold_events[gi]["t_s"]
        pred_center = (pred_events[pi]["start_s"] + pred_events[pi]["end_s"]) / 2.0
        diffs.append(abs(pred_center - gold_t))
    if not diffs:
        return None
    return sum(diffs) / len(diffs)


def render_timeline_marks(
    match_id: str, gold_events: list[dict], pred_events: list[dict], outcome: MatchOutcome,
) -> dict:
    """One match's dual-rail-timeline render payload (frozen contract §F):
    every GT/prediction event tagged with its TP/FN/FP status (+ subtype),
    plus the TP index pairs the client draws a connecting tick between.

    Never recomputes matching — this purely re-slices the ALREADY-DECIDED
    ``outcome`` (mirrors ``metrics.position_entry_rollup``'s "filter+count
    over an already-computed outcome, not a new matching pass" discipline).
    Every gold/pred index is asserted present in exactly one status bucket
    (matcher.match_one's own postcondition — every index it produces lands in
    exactly one of tp/fn_*/fp_* by construction) rather than defaulted, so a
    matcher regression that leaves an index unclassified fails loudly here
    instead of silently rendering a wrong color.
    """
    gold_status: dict[int, str] = {}
    for gi, _pi in outcome.tp:
        gold_status[gi] = "tp"
    for gi in outcome.fn_silent:
        gold_status[gi] = "fn_silent"
    for gi in outcome.fn_misclassified:
        gold_status[gi] = "fn_misclassified"
    for gi in outcome.fn_competed_away:
        gold_status[gi] = "fn_competed_away"
    assert set(gold_status) == set(range(len(gold_events))), (
        f"match {match_id!r}: matcher left gold index/indices unclassified — "
        f"{set(range(len(gold_events))) - set(gold_status)}"
    )

    pred_status: dict[int, str] = {}
    for _gi, pi in outcome.tp:
        pred_status[pi] = "tp"
    for pi in outcome.fp_hallucinated:
        pred_status[pi] = "fp_hallucinated"
    for pi in outcome.fp_duplicate:
        pred_status[pi] = "fp_duplicate"
    assert set(pred_status) == set(range(len(pred_events))), (
        f"match {match_id!r}: matcher left prediction index/indices unclassified — "
        f"{set(range(len(pred_events))) - set(pred_status)}"
    )

    gold_marks = [
        {"index": gi, "t_s": e["t_s"], "action_class": e.get("action_class"), "status": gold_status[gi]}
        for gi, e in enumerate(gold_events)
    ]
    pred_marks = [
        {"index": pi, "start_s": p["start_s"], "end_s": p["end_s"], "status": pred_status[pi]}
        for pi, p in enumerate(pred_events)
    ]
    pairs = [{"gold_index": gi, "pred_index": pi} for gi, pi in outcome.tp]

    return {"match_id": match_id, "gold_marks": gold_marks, "pred_marks": pred_marks, "pairs": pairs}
