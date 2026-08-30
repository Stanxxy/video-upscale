"""Gate 4 scoring — attribution self-consistency/honesty signals,
`GATES_3_4_DESIGN.md` §I.6. Pure post-processing over `gate4_predictions.json`
— NO `service.pipelines` dependency, runs fine under EITHER worktree's venv
(run under wt-vlmstudio's own venv for consistency with every other scorer).

**§I.1's framing, restated as code discipline, not just prose:** this module
computes sentinel/uncertainty/flip-instability signals only. It NEVER
computes or reports an "accuracy" number — there is no ground-truth
player-identity label anywhere in the gold corpus to compare against.

**Cross-highlight flip rate — operationalization note (design ambiguity,
flagged):** §I.6's exact denominator wording ("matches with >=2 non-sentinel
highlights") could mean >=2 non-sentinel highlights OVERALL, or specifically
>=2 non-sentinel highlights among the `outcome=="successful"` subset (the
only subset the flip CHECK itself actually uses). This module uses the
latter, stricter reading: a match is only ELIGIBLE for the flip check if it
has >=2 `outcome=="successful"` windows that BOTH resolved to a non-sentinel
actor — anything less has literally nothing to compare (a single data point,
or zero, cannot show a flip either way), and scoring it as a vacuous
"consistent" would inflate the reported rate with matches that never
actually tested anything.

Run from the repo root (after `run_gate4.py --execute`, from wt-s12):

    .venv/bin/python -m eval.adcc.s12_gates.score_gate4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from eval.adcc.s12_gates import gate_common as gc

# Verbatim data value from wt-s12's shared_lib.models.simplified_taxonomy.AXIS2_ACTOR_SENTINELS
# (confirmed via direct smoke import before this module was written) — a data constant, not code,
# duplicated here so score_gate4.py has zero service.pipelines/shared_lib dependency.
AXIS2_ACTOR_SENTINELS: list[str] = ["contested", "unclear"]


def score_gate4(rerun: bool = False) -> dict:
    """``rerun=True`` [RE-RUN 2026-07-26]: reads Gate 4's VERSIONED
    (qualified-model) predictions file instead of the flash-lite v1
    canonical."""
    predictions_path = gc.S12_GATES_DIR / "gate4_predictions.json"
    if rerun:
        predictions_path = gc.versioned_path(predictions_path)
    predictions = gc.read_json(predictions_path)
    n_total = len(predictions)
    n_scored = sum(1 for p in predictions if p["status"] != "call_failed")
    excluded = [p["window_id"] for p in predictions if p["status"] == "call_failed"]

    scored = [p for p in predictions if p["status"] != "call_failed"]

    with_refs_sentinel = sum(1 for p in scored if p["with_refs"]["actor"] in AXIS2_ACTOR_SENTINELS)
    without_refs_sentinel = sum(1 for p in scored if p["without_refs"]["actor"] in AXIS2_ACTOR_SENTINELS)
    sentinel_rate_with_refs = (with_refs_sentinel / n_scored) if n_scored else None
    sentinel_rate_without_refs = (without_refs_sentinel / n_scored) if n_scored else None

    identity_uncertain_count = sum(1 for p in scored if p["with_refs"].get("identity_uncertain") is True)
    identity_uncertain_rate = (identity_uncertain_count / n_scored) if n_scored else None

    # --- Cross-highlight flip rate (with-refs, successful-outcome subset only — see module docstring) ---
    by_match: dict[str, list[dict]] = {}
    for p in scored:
        by_match.setdefault(p["match_id"], []).append(p)

    per_match_detail = {}
    eligible_matches = 0
    flipped_matches = 0
    for mid, preds in by_match.items():
        successful_non_sentinel_actors = [
            p["with_refs"]["actor"] for p in preds
            if p["gold_event"]["outcome"] == "successful" and p["with_refs"]["actor"] not in AXIS2_ACTOR_SENTINELS
        ]
        n = len(successful_non_sentinel_actors)
        eligible = n >= 2
        flipped = (len(set(successful_non_sentinel_actors)) > 1) if eligible else None
        per_match_detail[mid] = {
            "n_successful_non_sentinel": n, "actors": successful_non_sentinel_actors,
            "eligible": eligible, "flipped": flipped,
        }
        if eligible:
            eligible_matches += 1
            if flipped:
                flipped_matches += 1

    flip_rate = (flipped_matches / eligible_matches) if eligible_matches else None

    return {
        "gate": "gate4", "rerun": rerun,
        "n_windows_total": n_total, "n_scored": n_scored, "excluded_call_failed": excluded,
        "sentinel_rate_with_refs": sentinel_rate_with_refs,
        "sentinel_rate_with_refs_count": with_refs_sentinel,
        "sentinel_rate_without_refs": sentinel_rate_without_refs,
        "sentinel_rate_without_refs_count": without_refs_sentinel,
        "identity_uncertain_rate": identity_uncertain_rate,
        "identity_uncertain_count": identity_uncertain_count,
        "cross_highlight_flip_rate": flip_rate,
        "cross_highlight_flip_eligible_matches": eligible_matches,
        "cross_highlight_flip_flipped_matches": flipped_matches,
        "cross_highlight_flip_per_match_detail": per_match_detail,
        "accuracy_vs_gold": None,  # explicitly never computed — see module docstring
    }


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def write_report(result: dict, predictions: list[dict], out_path: Path) -> None:
    lines = []
    lines.append("# Gate 4 Report — Attribution (actor axis + self-referential reference-image crops)")
    lines.append("")
    lines.append(
        "**NO ACCURACY-VS-GOLD NUMBER EXISTS OR IS CLAIMED ANYWHERE IN THIS REPORT (§I.1).** There is "
        "no player-identity ground truth anywhere in the gold corpus — every metric below is a "
        "self-consistency/honesty-channel signal, not a correctness measurement. Reference crops are "
        "SELF-REFERENTIAL left/right half-frame stills from the SAME match's own video, a disclosed "
        "coarse proxy for a real per-athlete bounding-box crop — NOT a person-detector output, and NOT "
        "verified-mapped to the real athlete names in gold's `name` field (labeled `ref_left`/`ref_right` "
        "only)."
    )
    lines.append("")
    lines.append("## Settings tuple")
    lines.append("")
    lines.append(f"fps={gc.FPS}, model={gc.MODEL_ID}, media_resolution=None (LOW, ActorAxisConfig default), "
                  "thinking=off, description=None throughout.")
    lines.append("")
    lines.append("## Per-match table")
    lines.append("")
    lines.append("| window | gold_class | gold_outcome | with_refs actor | with_refs identity_uncertain | "
                  "without_refs actor |")
    lines.append("|---|---|---|---|---|---|")
    for p in predictions:
        if p["status"] == "call_failed":
            lines.append(f"| {p['window_id']} | {p['gold_event']['action_class']} | "
                          f"{p['gold_event']['outcome']} | _(call_failed)_ | | |")
            continue
        lines.append(
            f"| {p['window_id']} | {p['gold_event']['action_class']} | {p['gold_event']['outcome']} | "
            f"{p['with_refs']['actor']} | {p['with_refs'].get('identity_uncertain')} | "
            f"{p['without_refs']['actor']} |"
        )
    lines.append("")
    lines.append(f"N scored: {result['n_scored']}/{result['n_windows_total']} "
                  f"(excluded call_failed: {result['excluded_call_failed']}).")
    lines.append("")
    lines.append("## Rates")
    lines.append("")
    lines.append(
        f"**sentinel_rate_with_refs** = {_fmt_pct(result['sentinel_rate_with_refs'])} "
        f"({result['sentinel_rate_with_refs_count']}/{result['n_scored']}). INS-073 re-confirmation: "
        f"**sentinel_rate_without_refs** = {_fmt_pct(result['sentinel_rate_without_refs'])} "
        f"({result['sentinel_rate_without_refs_count']}/{result['n_scored']}) — expected ~100% by "
        "construction (no real choices exist in the enum without refs); a structural check, not a finding."
    )
    lines.append("")
    lines.append(
        f"**identity_uncertain_rate** (with-refs only) = {_fmt_pct(result['identity_uncertain_rate'])} "
        f"({result['identity_uncertain_count']}/{result['n_scored']})."
    )
    lines.append("")
    lines.append(
        f"**cross_highlight_flip_rate** = {_fmt_pct(result['cross_highlight_flip_rate'])} "
        f"({result['cross_highlight_flip_flipped_matches']}/{result['cross_highlight_flip_eligible_matches']} "
        "eligible matches — see module docstring for the exact eligibility rule: >=2 `outcome==successful` "
        "windows that both resolved to a non-sentinel actor). **Caveat, stated plainly (design §I.6): a "
        "genuinely back-and-forth match makes SOME flipping correct, not an error — this is a coarse "
        "instability signal, not a precision measurement.**"
    )
    lines.append("")
    lines.append("### Per-match flip detail")
    lines.append("")
    lines.append("| match_id | n_successful_non_sentinel | actors | eligible | flipped |")
    lines.append("|---|---|---|---|---|")
    for mid, d in result["cross_highlight_flip_per_match_detail"].items():
        lines.append(f"| {mid} | {d['n_successful_non_sentinel']} | {d['actors']} | {d['eligible']} | {d['flipped']} |")
    lines.append("")
    lines.append("## What this gate does NOT settle (§I.8)")
    lines.append("")
    lines.append(
        "This design gives sentinel/uncertainty/flip-rate signals and an INS-073 re-confirmation — enough "
        "to judge whether the actor axis behaves sanely and honestly when it has SOME visual anchor vs. "
        "none. It does NOT give a true per-player attribution accuracy number. That would require real "
        "athlete reference photos (or human-verified name-to-crop labels) and hand-labeled ground truth — "
        "out of scope here, a genuine not-yet-closed gap, not solved by this design."
    )
    lines.append("")
    out_path.write_text("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun", action="store_true",
                        help="[RE-RUN 2026-07-26] Score the qualified-model re-run — reads versioned "
                             "predictions, writes gc.RERUN_LABEL-suffixed outputs, never touches v1 canonicals.")
    args = parser.parse_args(argv)

    predictions_path = gc.S12_GATES_DIR / "gate4_predictions.json"
    if args.rerun:
        predictions_path = gc.versioned_path(predictions_path)
    predictions = gc.read_json(predictions_path)
    result = score_gate4(rerun=args.rerun)
    suffix = f".{gc.RERUN_LABEL}" if args.rerun else ""
    scoring_path = gc.S12_GATES_DIR / f"gate4_scoring{suffix}.json"
    report_path = gc.S12_GATES_DIR / f"GATE4_REPORT{suffix}.md"
    gc.write_json(scoring_path, result)
    write_report(result, predictions, report_path)
    print(f"wrote {scoring_path}", file=sys.stderr)
    print(f"wrote {report_path}", file=sys.stderr)
    print(f"sentinel_with_refs={result['sentinel_rate_with_refs']} "
          f"identity_uncertain_rate={result['identity_uncertain_rate']} "
          f"flip_rate={result['cross_highlight_flip_rate']}", file=sys.stderr)


if __name__ == "__main__":
    main()
