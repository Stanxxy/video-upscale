"""``python -m eval.adcc.score.report`` — the founder-facing offline report
CLI (plan §6.3/§9). Zero network, zero Gemini calls: a pure function over the
committed gold snapshot (``eval/adcc/gold_snapshot/normalized_gold.jsonl`` +
``split_assignment.json``) and an already-captured predictions directory.

**Predictions directory contract** (documented here since no ``predict/``
prediction-capture module exists in this offline build — that is Phase 2
work, explicitly out of scope for this task): one JSON file per ``match_id``,
named ``<match_id>.json``, shaped::

    {
      "match_id": "<match_id>",
      "video_url_md5": "<32-char md5 hex>",
      "clips": [
        {"start_s": 12.4, "end_s": 15.1, "action_class": "guard_pass",
         "actor": "top", "outcome": "successful", "confidence": 0.83},
        ...
      ]
    }

A ``predict/`` module that partitions a multi-match VOD's raw
``simplified-tags-time-v1`` output into per-match files in this exact shape
is the natural Phase-2 companion to this scorer; it is not built here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

from eval.adcc.data_prep.apply_gracie_mapping import load_mapping
from eval.adcc.score.metrics import (
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_TAU_T_SWEEP,
    attach_compat,
    auc_pr,
    bootstrap_match_ci,
    filter_scoreable,
    max_f1_point,
    position_entry_rollup,
    pr_curve_over_confidence,
    recall_at_precision_floor,
    run_track,
    sweep_tau_t,
)

__all__ = ["load_gold_snapshot", "load_split_assignment", "load_predictions", "build_report", "main"]

DEFAULT_FROZEN_TAU_T = 5.0
DEFAULT_PRECISION_FLOOR = 0.9

_SPLIT_ALIASES = {"dev": ["dev"], "held_out": ["test"], "all": ["dev", "test"]}

# The three submission_* subtypes are the scoreable class where a miss is
# least forgivable (founder-trust ask) — combined recall is surfaced as its
# own headline number rather than left for the reader to hand-sum from the
# per-class table.
SUBMISSION_CLASSES: tuple[str, ...] = ("submission_choke", "submission_arm_lock", "submission_leg_lock")


def load_gold_snapshot(path: Path) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_split_assignment(path: Path) -> dict:
    return json.loads(path.read_text())


def load_predictions(predictions_dir: Path, match_ids: set[str]) -> dict[str, list[dict]]:
    """Loads ``<match_id>.json`` prediction files for the requested
    ``match_ids`` only. A match_id with NO prediction file on disk gets an
    empty prediction list (every gold event in that match becomes an FN) —
    never silently excluded from the gold denominator."""
    pred_by_match: dict[str, list[dict]] = {}
    for match_id in match_ids:
        path = predictions_dir / f"{match_id}.json"
        if not path.exists():
            pred_by_match[match_id] = []
            continue
        data = json.loads(path.read_text())
        clips = data.get("clips", [])
        pred_by_match[match_id] = [{**clip, "match_id": match_id} for clip in clips]
    return pred_by_match


def _match_ids_for_split(split_assignment: dict, split_labels: list[str]) -> set[str]:
    ids: set[str] = set()
    for entry in split_assignment["assignment"].values():
        if entry["split"] in split_labels:
            ids.update(entry["match_ids"])
    return ids


def _select_gold(gold_records: list[dict], match_ids: set[str]) -> dict[str, list[dict]]:
    return {rec["match_id"]: rec["events"] for rec in gold_records if rec["match_id"] in match_ids}


def _git_sha(cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 — best-effort provenance stamp, never fatal to the report
        return None


def _named_miss_list(
    gold_by_match: dict[str, list[dict]],
    track_b_summary: dict,
    outcomes_gi_by_match: dict[str, list[int]],
) -> list[dict]:
    misses = []
    for match_id, gold_indices in outcomes_gi_by_match.items():
        events = gold_by_match[match_id]
        for gi in gold_indices:
            event = events[gi]
            misses.append({
                "match_id": match_id, "t_s": event["t_s"], "action_class": event["action_class"],
                "raw_sequenceType": event["raw_sequenceType"],
            })
    misses.sort(key=lambda m: (m["match_id"], m["t_s"]))
    return misses


def _fn_gi_by_class(
    gold_by_match: dict[str, list[dict]],
    outcomes: dict,
) -> dict[str, dict[str, list[int]]]:
    """Groups every Track-B FN (any subtype: silent/misclassified/
    competed_away) by the missed gold event's own ``action_class``, then by
    ``match_id`` — the shape ``_named_miss_list`` expects per class. Feeds
    both the submission-focused callout (highest prominence, per Jobs
    acceptance item 6) and the full scoreable named-miss list (item 4: every
    dropped scoreable class, not just submissions — LeCun §5.6 bets the real
    recall cliff is on sweeps/back-takes/scrambles, so those must be
    individually visible too)."""
    by_class: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for match_id, outcome in outcomes.items():
        events = gold_by_match.get(match_id, [])
        for gi in outcome.fn_silent + outcome.fn_misclassified + outcome.fn_competed_away:
            cls = events[gi]["action_class"]
            by_class[cls][match_id].append(gi)
    return by_class


def _submission_family_recall(per_class: dict) -> dict:
    """Standalone headline (item 2): combined recall across the three
    submission_* classes, computed from the SAME per-class TP/FN counts
    ``metrics.summarize`` already produced — no new scoring, just a sum over
    three already-computed rows so the founder doesn't have to hand-divide
    the miss list."""
    tp = sum(per_class.get(c, {}).get("tp_for_recall", 0) for c in SUBMISSION_CLASSES)
    fn = sum(per_class.get(c, {}).get("fn", 0) for c in SUBMISSION_CLASSES)
    gold = tp + fn
    recall = tp / gold if gold > 0 else None
    return {"tp": tp, "fn": fn, "gold": gold, "recall": recall}


def build_report(
    gold_records: list[dict],
    split_assignment: dict,
    predictions_dir: Path,
    mapping: dict,
    *,
    split: str = "all",
    tau_t: float = DEFAULT_FROZEN_TAU_T,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
    non_scoreable_pred_classes: Optional[frozenset[str]] = None,
    bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Pure function: gold + split + predictions -> the full report payload
    (everything ``report.md``/``report.json`` render from). Runs Track A/B at
    the frozen ``tau_t``, the tau_t sweep (Track B), and the confidence PR
    curve (Track B), for EACH requested split label separately — never a
    single blended dev+held-out number (plan §9)."""
    if non_scoreable_pred_classes is None:
        non_scoreable_pred_classes = frozenset(mapping["non_scoreable_prediction_classes"])

    split_labels = _SPLIT_ALIASES[split]
    per_split: dict[str, dict] = {}

    from eval.adcc.score.matcher import match_all  # local import: avoids a cycle with metrics at module load

    for label in split_labels:
        match_ids = _match_ids_for_split(split_assignment, [label])
        gold_by_match_raw = _select_gold(gold_records, match_ids)
        gold_by_match = attach_compat(gold_by_match_raw, mapping)
        pred_by_match = load_predictions(predictions_dir, match_ids)

        track_a = run_track(gold_by_match, pred_by_match, tau_t)
        gold_b, pred_b = filter_scoreable(gold_by_match, pred_by_match, non_scoreable_pred_classes)
        track_b = run_track(gold_b, pred_b, tau_t)

        # B1 (Bezos): bootstrap 95% CI over MATCHES for Track-B (the ship
        # gate) micro P/R/F1 — never over individual events (plan §5.4: a
        # match's events aren't independent draws). Seeded/deterministic.
        track_b_ci = bootstrap_match_ci(gold_b, pred_b, tau_t, n_resamples=bootstrap_n, seed=bootstrap_seed)

        tau_t_sweep = sweep_tau_t(gold_b, pred_b, DEFAULT_TAU_T_SWEEP)
        pr_curve = pr_curve_over_confidence(gold_b, pred_b, tau_t)

        # Named-miss lists (per Jobs acceptance item 6 + the item-4 extension
        # to every scoreable class): group every Track-B FN (any subtype) by
        # the missed gold event's own action_class, then render both the
        # submission-focused callout (highest prominence) and the full
        # per-class breakdown from the SAME grouping (no duplicate FN walk).
        outcomes_b = match_all(gold_b, pred_b, tau_t)
        fn_gi_by_class = _fn_gi_by_class(gold_b, outcomes_b)

        submission_miss_gi: dict[str, list[int]] = defaultdict(list)
        for cls in SUBMISSION_CLASSES:
            for match_id, gis in fn_gi_by_class.get(cls, {}).items():
                submission_miss_gi[match_id].extend(gis)
        submission_misses = _named_miss_list(gold_b, track_b, submission_miss_gi)

        named_misses_by_class = {
            cls: _named_miss_list(gold_b, track_b, gi_by_match)
            for cls, gi_by_match in sorted(fn_gi_by_class.items())
        }

        # Item 2: submission-family recall as an explicit headline number.
        submission_family_recall = _submission_family_recall(track_b["per_class"])

        # Item 3: POSITION-ENTRY-alone rollup (context/bookkeeping, not a
        # ship gate) — a filter+count re-slice of the SAME Track-B outcomes
        # already computed above, never a new match run.
        pe_rollup = position_entry_rollup(outcomes_b, gold_b)

        per_split[label] = {
            "match_count": len(match_ids),
            "track_a": track_a,
            "track_b": track_b,
            "track_b_ci": track_b_ci,
            "tau_t_sweep": {str(k): v for k, v in tau_t_sweep.items()},
            "pr_curve": pr_curve,
            "pr_curve_auc": auc_pr(pr_curve),
            "pr_curve_max_f1": max_f1_point(pr_curve),
            "pr_curve_recall_at_precision_floor": recall_at_precision_floor(pr_curve, precision_floor),
            "submission_misses": submission_misses,
            "submission_family_recall": submission_family_recall,
            "position_entry_rollup": pe_rollup,
            "named_misses_by_class": named_misses_by_class,
        }

    return {
        "split_requested": split,
        "frozen_tau_t": tau_t,
        "precision_floor": precision_floor,
        "per_split": per_split,
    }


def _fmt(x: Optional[float]) -> str:
    return f"{x:.3f}" if x is not None else "N/A"


def _fmt_ci(ci: Optional[list]) -> str:
    if ci is None or ci[0] is None or ci[1] is None:
        return "N/A"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _match_count_asymmetry_caveat(per_split: dict) -> Optional[str]:
    """B2 (Bezos): the dev/test split is stratified at the VIDEO level (plan
    §5.5), and match counts inherit whatever match density each video happens
    to carry — so dev and test can (and, under the frozen seed, do) land on
    noticeably different NUMBERS OF MATCHES even at nominally-equal ~15%
    video shares. Surfaced as an explicit one-line caveat so a "dev improved /
    held-out didn't" delta isn't over-read as if the two splits were
    equal-weight samples."""
    if "dev" not in per_split or "test" not in per_split:
        return None
    dev_n = per_split["dev"]["match_count"]
    test_n = per_split["test"]["match_count"]
    if dev_n == test_n:
        return None
    larger, smaller = ("dev", "test") if dev_n > test_n else ("test", "dev")
    return (
        f"**Caveat — dev/test match-count asymmetry:** dev has {dev_n} matches vs. test's {test_n} "
        f"under the frozen split seed (stratification is at the VIDEO-URL level, plan §5.5 — a "
        f"video's match count, not a fixed match quota, decides how many matches land in each "
        f"~15% video share). {larger} currently has more matches than {smaller}. A P/R/F1 delta "
        "between dev and held-out should be read against this imbalance, not treated as two "
        "equal-weight samples — see each split's own bootstrap CI (over matches) for the "
        "uncertainty this asymmetry itself implies."
    )


def _render_confusion_matrix_md(confusion: dict[str, dict]) -> str:
    rows = sorted(confusion.keys())
    cols = sorted({c for row in confusion.values() for c in row.keys()})
    header = "| gold \\ pred | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines = [header, sep]
    for row in rows:
        cells = [str(confusion[row].get(col, 0)) for col in cols]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_per_class_table_md(per_class: dict) -> str:
    """One row per class with a DEFINED-or-attempted metric (item 1): P/R/F1
    plus the raw TP/FN/FP counts each ratio was computed from, straight out
    of ``metrics.summarize``'s ``per_class`` dict — nothing recomputed here.
    Recall-side TP (``tp_for_recall``, keyed off the gold event's own class)
    and precision-side TP (``tp_for_precision``, keyed off the prediction's
    claimed class) are shown as separate columns since they can genuinely
    differ under SUBMISSION's many-to-many compat rule (metrics.py's module
    docstring, §5.1)."""
    classes = sorted(per_class.keys())
    header = "| class | precision | recall | f1 | tp (recall-side) | fn | tp (precision-side) | fp |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for cls in classes:
        row = per_class[cls]
        lines.append(
            f"| {cls} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} | "
            f"{row['tp_for_recall']} | {row['fn']} | {row['tp_for_precision']} | {row['fp']} |"
        )
    return "\n".join(lines)


def _render_report_md(report: dict) -> str:
    lines = [
        "# ADCC Eval Harness — Report",
        "",
        f"Frozen tau_t: {report['frozen_tau_t']}s | Precision floor: {report['precision_floor']}",
        "",
    ]

    asymmetry_caveat = _match_count_asymmetry_caveat(report["per_split"])
    if asymmetry_caveat is not None:
        lines += [asymmetry_caveat, ""]

    for label, split_report in report["per_split"].items():
        ta, tb = split_report["track_a"], split_report["track_b"]
        tb_ci = split_report["track_b_ci"]
        sub_recall = split_report["submission_family_recall"]
        pe = split_report["position_entry_rollup"]
        lines += [
            f"## Split: {label} ({split_report['match_count']} matches)",
            "",
            "### Headline (Track B — scoreable subset, the ship gate)",
            f"- Precision (micro): {_fmt(tb['micro']['precision'])} "
            f"(95% CI over matches, n={tb_ci['n_resamples']} resamples, seed={tb_ci['seed']}: "
            f"{_fmt_ci(tb_ci['precision_ci'])})",
            f"- Recall (micro): {_fmt(tb['micro']['recall'])} (95% CI: {_fmt_ci(tb_ci['recall_ci'])})",
            f"- F1 (micro): {_fmt(tb['micro']['f1'])} (95% CI: {_fmt_ci(tb_ci['f1_ci'])})",
            f"- Precision (macro): {_fmt(tb['macro']['precision'])}",
            f"- Recall (macro): {_fmt(tb['macro']['recall'])}",
            f"- F1 (macro): {_fmt(tb['macro']['f1'])}",
            f"- Event-count ratio (pred/gold): {_fmt(tb['event_count_ratio'])}",
            f"- FN breakdown: silent={tb['fn_breakdown']['silent']} "
            f"misclassified={tb['fn_breakdown']['misclassified']} competed_away={tb['fn_breakdown']['competed_away']}",
            f"- FP breakdown: hallucinated={tb['fp_breakdown']['hallucinated']} duplicate={tb['fp_breakdown']['duplicate']}",
            f"- **Submission-family recall** (submission_choke+arm_lock+leg_lock combined — "
            f"the least-forgivable miss): {_fmt(sub_recall['recall'])} "
            f"(tp={sub_recall['tp']}/gold={sub_recall['gold']}, fn={sub_recall['fn']})",
            "",
            "### Per-class P/R/F1 (Track B — scoreable subset, the ship gate)",
            _render_per_class_table_md(tb["per_class"]),
            "",
            "### Track A (full taxonomy — context/sanity only, NOT the optimization target)",
            f"- Precision (micro): {_fmt(ta['micro']['precision'])} | Recall (micro): {_fmt(ta['micro']['recall'])} | F1 (micro): {_fmt(ta['micro']['f1'])}",
            "",
            "#### Per-class P/R/F1 (Track A, full taxonomy — context only)",
            _render_per_class_table_md(ta["per_class"]),
            "",
            "### PR curve over confidence (Track B)",
            f"- AUC-PR: {_fmt(split_report['pr_curve_auc'])}",
            f"- Max-F1 point: {split_report['pr_curve_max_f1']}",
            f"- Recall @ precision >= {report['precision_floor']}: {split_report['pr_curve_recall_at_precision_floor']}",
            "",
            "### tau_t sweep (Track B, recall vs. temporal tolerance)",
        ]
        for tau_t_str, summary in split_report["tau_t_sweep"].items():
            lines.append(f"- tau_t={tau_t_str}s: recall={_fmt(summary['micro']['recall'])} precision={_fmt(summary['micro']['precision'])}")
        lines += [
            "",
            "### Confusion matrix (Track A, gold class vs. predicted class, NO_MATCH = FN/FP row-col)",
            _render_confusion_matrix_md(ta["confusion_matrix"]),
            "",
            "### POSITION-ENTRY-alone rollup (context/bookkeeping only — NOT a ship gate)",
            "POSITION ENTRY is the annotators' catch-all sequenceType (§4.3 split router); this "
            "rollup re-slices the SAME Track-B match outcomes above (no re-matching) down to just "
            "the gold events whose raw_sequenceType == POSITION ENTRY that resolved to a scoreable "
            "class (back_take/guard_pass/sweep). Precision is not defined here — predictions carry "
            "no raw_sequenceType tag to attribute against.",
            f"- Overall: recall={_fmt(pe['totals']['recall'])} (tp={pe['totals']['tp']}, fn={pe['totals']['fn']})",
        ]
        for cls in sorted(pe["per_class"].keys()):
            row = pe["per_class"][cls]
            lines.append(f"- {cls}: recall={_fmt(row['recall'])} (tp={row['tp']}, fn={row['fn']})")
        lines += [
            "",
            f"### Named submission misses ({len(split_report['submission_misses'])})",
        ]
        for miss in split_report["submission_misses"][:200]:
            lines.append(f"- match={miss['match_id']} t_s={miss['t_s']} class={miss['action_class']} raw={miss['raw_sequenceType']}")
        lines += [
            "",
            "### Named scoreable misses by class (Track B FN, all classes — submissions called out "
            "above with highest prominence; every other scoreable class, including sweeps/back-takes/"
            "escapes, is individually listed below, not just an aggregate FN count)",
        ]
        for cls, misses in split_report["named_misses_by_class"].items():
            lines.append(f"#### {cls} ({len(misses)})")
            for miss in misses[:200]:
                lines.append(f"- match={miss['match_id']} t_s={miss['t_s']} class={miss['action_class']} raw={miss['raw_sequenceType']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "held_out", "all"], default="all")
    parser.add_argument("--gold-snapshot", type=Path, default=Path("eval/adcc/gold_snapshot/normalized_gold.jsonl"))
    parser.add_argument("--split-assignment", type=Path, default=Path("eval/adcc/gold_snapshot/split_assignment.json"))
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tau-t", type=float, default=DEFAULT_FROZEN_TAU_T)
    parser.add_argument("--precision-floor", type=float, default=DEFAULT_PRECISION_FLOOR)
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N, help="bootstrap resample count for the Track-B match-level CI")
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED, help="seed for the Track-B bootstrap CI (deterministic)")
    args = parser.parse_args(argv)

    gold_records = load_gold_snapshot(args.gold_snapshot)
    split_assignment = load_split_assignment(args.split_assignment)
    mapping = load_mapping()

    report = build_report(
        gold_records, split_assignment, args.predictions, mapping,
        split=args.split, tau_t=args.tau_t, precision_floor=args.precision_floor,
        bootstrap_n=args.bootstrap_n, bootstrap_seed=args.bootstrap_seed,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=list) + "\n")
    (args.out / "report.md").write_text(_render_report_md(report))

    run_manifest = {
        "gold_snapshot_sha256": __import__("hashlib").sha256(args.gold_snapshot.read_bytes()).hexdigest(),
        "predictions_dir": str(args.predictions),
        "worktree_git_sha": _git_sha(Path(__file__).resolve().parent.parent.parent.parent),
        "mapping_version": mapping["version"],
        "split_requested": args.split,
        "frozen_tau_t": args.tau_t,
    }
    (args.out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
