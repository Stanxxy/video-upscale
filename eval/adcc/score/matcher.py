"""Per-match one-to-one Hungarian assignment (plan §5.2).

**Contract** (documented here since no ``predict/`` capture module is built
in this offline half of the harness — Phase 2 prediction-capture is
explicitly out of scope for this task):

  - a **gold event** is a dict with (at least) keys ``t_s`` (int seconds),
    ``compat`` (``frozenset[str]`` of acceptable prediction ``action_class``
    values — see ``apply_gracie_mapping.compat_classes_for``), and whatever
    bookkeeping fields the caller wants preserved (``action_class``,
    ``match_id``, etc., used by ``metrics.py`` for confusion-matrix
    attribution, not read by this module).
  - a **prediction event** is a dict with keys ``start_s``/``end_s`` (float
    seconds, ABSOLUTE match clock — already converted from chunk-relative by
    the prediction-capture layer), ``action_class`` (str), and optionally
    ``confidence``/``match_id``.

Both lists passed to ``match_one`` MUST already belong to exactly ONE
``match_id`` — the caller (``match_all`` below, or ``metrics.py``) is
responsible for partitioning gold and predictions per match before calling
this function. ``match_all`` enforces this by construction (it loops one
match_id bucket at a time) plus a defensive per-event tag assertion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["MatchOutcome", "match_one", "match_all"]

# Sentinel cost for class/time-infeasible (gold, pred) pairs — must exceed any
# real temporal distance in this corpus (matches run at most a few thousand
# seconds; the longest single video in the corpus is ~30,573s/8.5h, so 1e9 is
# comfortably unreachable by a real cost).
_INFEASIBLE_COST = 1e9


@dataclass(frozen=True)
class MatchOutcome:
    """Indices are positions into the ``gold_events``/``pred_events`` lists
    passed to ``match_one`` for this one match_id."""

    tp: list[tuple[int, int]] = field(default_factory=list)             # (gold_idx, pred_idx)
    fn_silent: list[int] = field(default_factory=list)                  # gold_idx
    fn_misclassified: list[int] = field(default_factory=list)           # gold_idx
    # gold_idx -> pred_idx of the nearest TIME-feasible-but-CLASS-incompatible
    # prediction — the plan §5.6 "crux smoking gun" diagnostic (what class did
    # the model call this real event instead?).
    fn_misclassified_nearest_pred: dict = field(default_factory=dict)
    fn_competed_away: list[int] = field(default_factory=list)           # gold_idx
    fp_hallucinated: list[int] = field(default_factory=list)            # pred_idx
    fp_duplicate: list[int] = field(default_factory=list)               # pred_idx


def _time_feasible(gold_t_s: float, pred_start: float, pred_end: float, tau_t: float) -> bool:
    return (pred_start - tau_t) <= gold_t_s <= (pred_end + tau_t)


def _cost(gold_t_s: float, pred_start: float, pred_end: float) -> float:
    """Temporal distance from the gold point to the nearest span edge, 0 if
    inside the span (plan §5.2)."""
    if pred_start <= gold_t_s <= pred_end:
        return 0.0
    if gold_t_s < pred_start:
        return pred_start - gold_t_s
    return gold_t_s - pred_end


def match_one(gold_events: list[dict], pred_events: list[dict], tau_t: float) -> MatchOutcome:
    """Hungarian-assign ONE match_id's gold events against ONE match_id's
    predictions. See module docstring for the event-dict contract."""
    n_g, n_p = len(gold_events), len(pred_events)
    if n_g == 0 and n_p == 0:
        return MatchOutcome()

    feasible = [[False] * n_p for _ in range(n_g)]
    cost_matrix = np.full((n_g, n_p), _INFEASIBLE_COST, dtype=float)
    for gi, g in enumerate(gold_events):
        for pi, p in enumerate(pred_events):
            time_ok = _time_feasible(g["t_s"], p["start_s"], p["end_s"], tau_t)
            class_ok = p["action_class"] in g["compat"]
            if time_ok and class_ok:
                feasible[gi][pi] = True
                cost_matrix[gi, pi] = _cost(g["t_s"], p["start_s"], p["end_s"])

    tp: list[tuple[int, int]] = []
    assigned_gold: set[int] = set()
    assigned_pred: set[int] = set()
    if n_g and n_p:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for gi, pi in zip(row_ind, col_ind):
            gi, pi = int(gi), int(pi)
            if feasible[gi][pi]:
                tp.append((gi, pi))
                assigned_gold.add(gi)
                assigned_pred.add(pi)

    fn_silent: list[int] = []
    fn_misclassified: list[int] = []
    fn_misclassified_nearest_pred: dict[int, int] = {}
    fn_competed_away: list[int] = []
    for gi in range(n_g):
        if gi in assigned_gold:
            continue
        if any(feasible[gi]):
            # A class+time-feasible prediction existed but lost the
            # one-to-one assignment to a DIFFERENT gold event.
            fn_competed_away.append(gi)
            continue
        # Relax the class requirement: was ANYTHING time-feasible here?
        time_only_candidates = [
            pi for pi, p in enumerate(pred_events)
            if _time_feasible(gold_events[gi]["t_s"], p["start_s"], p["end_s"], tau_t)
        ]
        if time_only_candidates:
            fn_misclassified.append(gi)
            nearest_pi = min(
                time_only_candidates,
                key=lambda pi: _cost(gold_events[gi]["t_s"], pred_events[pi]["start_s"], pred_events[pi]["end_s"]),
            )
            fn_misclassified_nearest_pred[gi] = nearest_pi
        else:
            fn_silent.append(gi)

    fp_hallucinated: list[int] = []
    fp_duplicate: list[int] = []
    for pi in range(n_p):
        if pi in assigned_pred:
            continue
        had_feasible_gold = any(feasible[gi][pi] for gi in range(n_g))
        if had_feasible_gold:
            fp_duplicate.append(pi)
        else:
            fp_hallucinated.append(pi)

    return MatchOutcome(
        tp=tp,
        fn_silent=fn_silent,
        fn_misclassified=fn_misclassified,
        fn_misclassified_nearest_pred=fn_misclassified_nearest_pred,
        fn_competed_away=fn_competed_away,
        fp_hallucinated=fp_hallucinated,
        fp_duplicate=fp_duplicate,
    )


def match_all(
    gold_by_match: dict[str, list[dict]],
    pred_by_match: dict[str, list[dict]],
    tau_t: float,
) -> dict[str, MatchOutcome]:
    """Runs ``match_one`` per ``match_id``. HARD ASSERTS that no event
    carries a ``match_id`` tag disagreeing with the bucket it was filed
    under — the structural guarantee (plan §5.2: "never match across
    match_ids") that this loop, by construction, never mixes two matches'
    events into one assignment problem.
    """
    results: dict[str, MatchOutcome] = {}
    for match_id in sorted(set(gold_by_match) | set(pred_by_match)):
        gold_events = gold_by_match.get(match_id, [])
        pred_events = pred_by_match.get(match_id, [])
        for g in gold_events:
            tag = g.get("match_id", match_id)
            assert tag == match_id, f"gold event tagged match_id={tag!r} found under bucket {match_id!r}"
        for p in pred_events:
            tag = p.get("match_id", match_id)
            assert tag == match_id, f"pred event tagged match_id={tag!r} found under bucket {match_id!r}"
        results[match_id] = match_one(gold_events, pred_events, tau_t)
    return results
