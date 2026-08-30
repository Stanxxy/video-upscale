"""P/R/F1 (micro+macro), event-count ratio, per-class confusion matrix, and
the confidence PR curve (plan §5.3/§5.4), built on top of ``matcher.py``'s
per-match ``MatchOutcome``s.

**Two-track discipline (§5.3, never blend the two numbers)**:
  - Track A (full taxonomy): every gold event, matched against ALL predictions.
  - Track B (scoreable subset, the ship gate): FILTER-THEN-MATCH —
    ``filter_scoreable`` drops non-scoreable gold/predictions from BOTH sides
    BEFORE ``matcher.match_all`` ever runs, never after (filter-after-match
    creates phantom FN/FP — a non-scoreable gold with no nearby prediction
    must NOT count as an FN once dropped from the denominator; a
    non-scoreable prediction near a real gold must NOT count as a competing
    FP once dropped from the numerator).

**TP/FP class attribution** (documented choice, since ``compat`` is
many-to-many for SUBMISSION gold rows — a gold ``submission_choke`` matched by
a predicted ``submission_arm_lock`` IS a real TP under §5.1's family-wide
compat rule, but the two class labels disagree): recall-side per-class counts
key off the GOLD event's own resolved ``action_class`` (was this true class
covered at all); precision-side per-class counts key off the PREDICTION's
claimed ``action_class`` (was this claimed class right, given the benefit of
the doubt on gold's own subtype uncertainty). This is the standard
many-to-many confusion-matrix convention and is documented here rather than
silently baked in.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Optional

from eval.adcc.data_prep.apply_gracie_mapping import compat_classes_for
from eval.adcc.score.matcher import MatchOutcome, match_all

__all__ = [
    "NO_MATCH",
    "attach_compat",
    "filter_scoreable",
    "prf1",
    "summarize",
    "run_track",
    "pr_curve_over_confidence",
    "auc_pr",
    "max_f1_point",
    "recall_at_precision_floor",
    "sweep_tau_t",
    "bootstrap_match_ci",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_BOOTSTRAP_N",
    "position_entry_rollup",
]

NO_MATCH = "NO_MATCH"

DEFAULT_TAU_T_SWEEP: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0, 10.0)
DEFAULT_CONFIDENCE_THRESHOLDS: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(21))  # 0.00..1.00


def attach_compat(gold_by_match: dict[str, list[dict]], mapping: dict) -> dict[str, list[dict]]:
    """Returns a NEW ``gold_by_match`` with a ``compat`` (``frozenset[str]``)
    key attached to every event, re-derived fresh from the versioned mapping
    artifact (never persisted in the committed snapshot — keeps the compat
    rule tied to whatever mapping version is active at score time, not frozen
    at gold-build time)."""
    out: dict[str, list[dict]] = {}
    for match_id, events in gold_by_match.items():
        out[match_id] = [
            {**event, "match_id": match_id, "compat": compat_classes_for(event, mapping)}
            for event in events
        ]
    return out


def filter_scoreable(
    gold_by_match: dict[str, list[dict]],
    pred_by_match: dict[str, list[dict]],
    non_scoreable_pred_classes: frozenset[str],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Track B's FILTER-THEN-MATCH step: drop non-scoreable gold events
    (``scoreable is False``, e.g. residual ``transition``) and non-scoreable
    predictions (``action_class`` in ``non_scoreable_pred_classes``, e.g.
    ``transition``/``guard_pull``) BEFORE any matching happens."""
    gold_filtered = {
        match_id: [e for e in events if e.get("scoreable", True)]
        for match_id, events in gold_by_match.items()
    }
    pred_filtered = {
        match_id: [p for p in events if p["action_class"] not in non_scoreable_pred_classes]
        for match_id, events in pred_by_match.items()
    }
    return gold_filtered, pred_filtered


def prf1(tp: int, fp: int, fn: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns ``(precision, recall, f1)``. ``None`` for any metric whose
    denominator is 0 (undefined, e.g. a class predicted zero times) — the
    caller decides how to treat ``None`` for macro-averaging (excluded, not
    silently coerced to 0.0 — see ``summarize``'s docstring)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def summarize(
    outcomes: dict[str, MatchOutcome],
    gold_by_match: dict[str, list[dict]],
    pred_by_match: dict[str, list[dict]],
) -> dict:
    """Aggregates every match's ``MatchOutcome`` into the report-ready shape:
    totals, micro P/R/F1, per-class P/R/F1 (+macro = mean over classes with a
    DEFINED metric, per-class undefined metrics excluded from the average
    rather than coerced to 0), confusion matrix (with an explicit
    ``NO_MATCH`` row/col), and the FN/FP subtype breakdown (§5.2)."""
    tp_for_recall: Counter = Counter()
    fn_by_class: Counter = Counter()
    tp_for_precision: Counter = Counter()
    fp_by_class: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)

    total_tp = total_fn = total_fp = 0
    fn_silent_n = fn_misclassified_n = fn_competed_away_n = 0
    fp_hallucinated_n = fp_duplicate_n = 0

    for match_id, outcome in outcomes.items():
        gold_events = gold_by_match.get(match_id, [])
        pred_events = pred_by_match.get(match_id, [])

        for gi, pi in outcome.tp:
            gold_class = gold_events[gi]["action_class"]
            pred_class = pred_events[pi]["action_class"]
            tp_for_recall[gold_class] += 1
            tp_for_precision[pred_class] += 1
            confusion[gold_class][pred_class] += 1
            total_tp += 1

        for gi in outcome.fn_silent:
            gold_class = gold_events[gi]["action_class"]
            fn_by_class[gold_class] += 1
            confusion[gold_class][NO_MATCH] += 1
            total_fn += 1
            fn_silent_n += 1

        for gi in outcome.fn_misclassified:
            gold_class = gold_events[gi]["action_class"]
            fn_by_class[gold_class] += 1
            nearest_pi = outcome.fn_misclassified_nearest_pred[gi]
            nearest_class = pred_events[nearest_pi]["action_class"]
            confusion[gold_class][nearest_class] += 1  # §5.6 "crux smoking gun" detail, not lumped into NO_MATCH
            total_fn += 1
            fn_misclassified_n += 1

        for gi in outcome.fn_competed_away:
            gold_class = gold_events[gi]["action_class"]
            fn_by_class[gold_class] += 1
            confusion[gold_class][NO_MATCH] += 1
            total_fn += 1
            fn_competed_away_n += 1

        for pi in outcome.fp_hallucinated:
            pred_class = pred_events[pi]["action_class"]
            fp_by_class[pred_class] += 1
            confusion[NO_MATCH][pred_class] += 1
            total_fp += 1
            fp_hallucinated_n += 1

        for pi in outcome.fp_duplicate:
            pred_class = pred_events[pi]["action_class"]
            fp_by_class[pred_class] += 1
            confusion[NO_MATCH][pred_class] += 1
            total_fp += 1
            fp_duplicate_n += 1

    micro_p, micro_r, micro_f1 = prf1(total_tp, total_fp, total_fn)

    all_classes = sorted(set(tp_for_recall) | set(fn_by_class) | set(tp_for_precision) | set(fp_by_class))
    per_class: dict[str, dict] = {}
    macro_p_values, macro_r_values, macro_f1_values = [], [], []
    for cls in all_classes:
        tp_p, fp = tp_for_precision[cls], fp_by_class[cls]
        tp_r, fn = tp_for_recall[cls], fn_by_class[cls]
        p = tp_p / (tp_p + fp) if (tp_p + fp) > 0 else None
        r = tp_r / (tp_r + fn) if (tp_r + fn) > 0 else None
        f1 = None
        if p is not None and r is not None and (p + r) > 0:
            f1 = 2 * p * r / (p + r)
        per_class[cls] = {
            "tp_for_precision": tp_for_precision[cls], "fp": fp_by_class[cls], "precision": p,
            "tp_for_recall": tp_for_recall[cls], "fn": fn_by_class[cls], "recall": r,
            "f1": f1,
        }
        if p is not None:
            macro_p_values.append(p)
        if r is not None:
            macro_r_values.append(r)
        if f1 is not None:
            macro_f1_values.append(f1)

    macro_p = sum(macro_p_values) / len(macro_p_values) if macro_p_values else None
    macro_r = sum(macro_r_values) / len(macro_r_values) if macro_r_values else None
    macro_f1 = sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else None

    gold_count = sum(len(events) for events in gold_by_match.values())
    pred_count = sum(len(events) for events in pred_by_match.values())

    return {
        "totals": {"tp": total_tp, "fn": total_fn, "fp": total_fp, "gold_count": gold_count, "pred_count": pred_count},
        "event_count_ratio": (pred_count / gold_count) if gold_count > 0 else None,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_class": per_class,
        "confusion_matrix": {row: dict(cols) for row, cols in confusion.items()},
        "fn_breakdown": {"silent": fn_silent_n, "misclassified": fn_misclassified_n, "competed_away": fn_competed_away_n},
        "fp_breakdown": {"hallucinated": fp_hallucinated_n, "duplicate": fp_duplicate_n},
    }


def run_track(
    gold_by_match: dict[str, list[dict]],
    pred_by_match: dict[str, list[dict]],
    tau_t: float,
) -> dict:
    """Runs the matcher + summarizer for one track (caller has already
    decided whether ``gold_by_match``/``pred_by_match`` are the full-taxonomy
    (Track A) or scoreable-filtered (Track B) event sets)."""
    outcomes = match_all(gold_by_match, pred_by_match, tau_t)
    return summarize(outcomes, gold_by_match, pred_by_match)


def position_entry_rollup(outcomes: dict[str, MatchOutcome], gold_by_match: dict[str, list[dict]]) -> dict:
    """Plan §7's third rollup (context/bookkeeping ONLY, never a ship gate):
    isolates recall for just the gold events whose ``raw_sequenceType ==
    "POSITION ENTRY"`` (the §4.3 split router's back_take/guard_pass/sweep/
    transition source rows), reported overall and per resolved
    ``action_class``.

    This is a FILTER+COUNT over ``outcomes`` that the caller already computed
    from a real ``match_all`` run (Track B, typically) — NOT a new matching
    pass and NOT a scoring-logic change. A POSITION-ENTRY-derived gold
    event's TP/FN status is exactly whatever the full assignment already
    decided for it; this function only re-slices those already-decided
    outcomes by ``raw_sequenceType``.

    Precision is deliberately NOT reported here: predictions carry no
    ``raw_sequenceType`` tag, so there is no principled way to say a given
    predicted clip was "competing specifically against a POSITION-ENTRY-
    derived gold" — recall-side TP/FN counts are the only well-defined slice.
    """
    tp_by_class: Counter = Counter()
    fn_by_class: Counter = Counter()

    for match_id, outcome in outcomes.items():
        events = gold_by_match.get(match_id, [])
        for gi, _pi in outcome.tp:
            event = events[gi]
            if event.get("raw_sequenceType") == "POSITION ENTRY":
                tp_by_class[event["action_class"]] += 1
        for gi in outcome.fn_silent + outcome.fn_misclassified + outcome.fn_competed_away:
            event = events[gi]
            if event.get("raw_sequenceType") == "POSITION ENTRY":
                fn_by_class[event["action_class"]] += 1

    classes = sorted(set(tp_by_class) | set(fn_by_class))
    per_class: dict[str, dict] = {}
    total_tp = total_fn = 0
    for cls in classes:
        tp, fn = tp_by_class[cls], fn_by_class[cls]
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        per_class[cls] = {"tp": tp, "fn": fn, "recall": recall}
        total_tp += tp
        total_fn += fn

    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
    return {
        "per_class": per_class,
        "totals": {"tp": total_tp, "fn": total_fn, "recall": total_recall},
    }


def sweep_tau_t(
    gold_by_match: dict[str, list[dict]],
    pred_by_match: dict[str, list[dict]],
    tau_t_values: tuple[float, ...] = DEFAULT_TAU_T_SWEEP,
) -> dict[float, dict]:
    """§5.1: τ_t is swept, never asserted. Returns ``{tau_t: summary}`` — a
    flat recall-vs-tau_t curve diagnosing temporal-offset (recall climbs with
    tau_t) vs class-confusion (recall plateaus early)."""
    return {tau_t: run_track(gold_by_match, pred_by_match, tau_t) for tau_t in tau_t_values}


def pr_curve_over_confidence(
    gold_by_match: dict[str, list[dict]],
    pred_by_match: dict[str, list[dict]],
    tau_t: float,
    thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS,
) -> list[dict]:
    """§5.4: PR curve over prediction ``confidence`` τ_c, 0->1 step 0.05.
    At each threshold, predictions with ``confidence < tau_c`` are dropped
    BEFORE matching (consistent filter-then-match discipline), then Track-
    whatever-was-passed-in P/R/F1 is recomputed from scratch. A flat curve is
    itself a finding (confidence isn't calibrated enough to gate anything)."""
    curve: list[dict] = []
    for tau_c in thresholds:
        filtered_preds = {
            match_id: [p for p in events if float(p.get("confidence", 0.0)) >= tau_c]
            for match_id, events in pred_by_match.items()
        }
        summary = run_track(gold_by_match, filtered_preds, tau_t)
        curve.append({
            "tau_c": tau_c,
            "precision": summary["micro"]["precision"],
            "recall": summary["micro"]["recall"],
            "f1": summary["micro"]["f1"],
            "tp": summary["totals"]["tp"],
            "fp": summary["totals"]["fp"],
            "fn": summary["totals"]["fn"],
        })
    return curve


def auc_pr(curve: list[dict]) -> Optional[float]:
    """Trapezoidal AUC over the (recall, precision) points of a PR curve,
    sorted by recall ascending. Points with an undefined precision/recall
    (``None``, e.g. zero predictions survive the threshold) are skipped."""
    points = sorted(
        ((c["recall"], c["precision"]) for c in curve if c["recall"] is not None and c["precision"] is not None),
        key=lambda pair: pair[0],
    )
    if len(points) < 2:
        return None
    area = 0.0
    for (r0, p0), (r1, p1) in zip(points, points[1:]):
        area += (r1 - r0) * (p0 + p1) / 2.0
    return area


def max_f1_point(curve: list[dict]) -> Optional[dict]:
    defined = [c for c in curve if c["f1"] is not None]
    if not defined:
        return None
    return max(defined, key=lambda c: c["f1"])


def recall_at_precision_floor(curve: list[dict], precision_floor: float = 0.9) -> Optional[dict]:
    """Highest-recall point on the curve whose precision is >= ``precision_floor``.
    ``None`` if no threshold reaches the floor (itself a reportable finding)."""
    candidates = [c for c in curve if c["precision"] is not None and c["precision"] >= precision_floor]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["recall"] if c["recall"] is not None else -1.0)


# --------------------------------------------------------------------------- #
# §5.4 bootstrap 95% CI, resampled over MATCHES (Bezos fix, Part B1).
#
# Events within a match are NOT independent (a match's gold events share one
# video, one referee/style, one Gemini run) — resampling individual EVENTS
# would understate the true variance. Resampling whole MATCH_IDs with
# replacement, and recomputing Track-B P/R/F1 fresh on each resample, is the
# standard cluster-bootstrap fix for this non-independence.
# --------------------------------------------------------------------------- #
DEFAULT_BOOTSTRAP_SEED = 20260712  # frozen, matches build_split.DEFAULT_SEED's date-derived convention
DEFAULT_BOOTSTRAP_N = 1000


def _percentile(sorted_values: list[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile (numpy's default ``'linear'`` method)
    over an ALREADY-SORTED list. ``None`` for an empty list (e.g. every
    resample produced an undefined metric — itself a reportable finding, not
    silently coerced to 0)."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (pct / 100.0) * (len(sorted_values) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_match_ci(
    gold_by_match: dict[str, list[dict]],
    pred_by_match: dict[str, list[dict]],
    tau_t: float,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci_low_pct: float = 2.5,
    ci_high_pct: float = 97.5,
) -> dict:
    """Bootstrap ``(ci_low_pct, ci_high_pct)`` CI for micro P/R/F1 on
    ``gold_by_match``/``pred_by_match`` (caller decides which track — this is
    plumbed in from Track B, the ship-gate track, by ``report.py``).

    Resamples the set of ``match_id`` KEYS with replacement (``len(match_ids)``
    draws per resample, plan §5.4's "resample matches with replacement"), not
    individual events. A resampled match_id can appear more than once in one
    resample; each occurrence is re-tagged with a synthetic per-occurrence
    match_id (``f"{match_id}__{occurrence_index}"``) so ``match_all``'s
    per-match_id bucketing (and its own-tag assertion) stays correct — the
    duplicate occurrence is scored as if it were an independent extra match,
    which is exactly the point of the match-level bootstrap.

    Deterministic: same ``seed`` -> byte-identical resample draws -> identical
    CI, every time (``random.Random(seed)``, never the shared global RNG).
    The **point estimate** returned is the real (non-bootstrapped) Track-B
    micro P/R/F1 on the FULL sample — the bootstrap only estimates the CI
    around it, never replaces it with a bootstrap mean.
    """
    match_ids = sorted(gold_by_match)  # deterministic universe, sorted for reproducibility
    point = run_track(gold_by_match, pred_by_match, tau_t)["micro"]

    if not match_ids:
        return {
            "point": point, "n_resamples": n_resamples, "seed": seed,
            "precision_ci": None, "recall_ci": None, "f1_ci": None,
        }

    rng = random.Random(seed)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for _ in range(n_resamples):
        drawn = [rng.choice(match_ids) for _ in match_ids]
        resampled_gold: dict[str, list[dict]] = {}
        resampled_pred: dict[str, list[dict]] = {}
        for occurrence_idx, mid in enumerate(drawn):
            synthetic_id = f"{mid}__{occurrence_idx}"
            resampled_gold[synthetic_id] = [{**e, "match_id": synthetic_id} for e in gold_by_match[mid]]
            resampled_pred[synthetic_id] = [{**p, "match_id": synthetic_id} for p in pred_by_match.get(mid, [])]

        resample_micro = run_track(resampled_gold, resampled_pred, tau_t)["micro"]
        if resample_micro["precision"] is not None:
            precisions.append(resample_micro["precision"])
        if resample_micro["recall"] is not None:
            recalls.append(resample_micro["recall"])
        if resample_micro["f1"] is not None:
            f1s.append(resample_micro["f1"])

    precisions.sort()
    recalls.sort()
    f1s.sort()

    return {
        "point": point,
        "n_resamples": n_resamples,
        "seed": seed,
        "precision_ci": [_percentile(precisions, ci_low_pct), _percentile(precisions, ci_high_pct)],
        "recall_ci": [_percentile(recalls, ci_low_pct), _percentile(recalls, ci_high_pct)],
        "f1_ci": [_percentile(f1s, ci_low_pct), _percentile(f1s, ci_high_pct)],
    }
