"""LeCun cutover Gate 2 scoring (``GATES_1_2_DESIGN.md`` §C) — validator
``media_resolution`` (low vs high) EXACT-CLASS-AGREEMENT scoring on the
16-clip occlusion subset. Deliberately does NOT use ``matcher.py``/
``metrics.py`` — the module-level warning at the top of the design doc
explains why: the standard Hungarian matcher's family-wide SUBMISSION
compat rule would score a predicted ``submission_leg_lock`` against a gold
``submission_arm_lock`` as a TP, which is exactly the confusion this gate
exists to detect (§C.1).

Bezos condition 4 (2026-07-25 pass-with-conditions review): the
``(directional, n=16)`` qualifier is printed INLINE on the verdict line in
``GATE2_REPORT.md``, not only in surrounding caveat text.

Bezos precondition 2 (2026-07-26 authorize-with-preconditions review): §G.1
point 4 dual reporting — headline recomputed with
``ingestion_path=="files_api"`` clips excluded, plus an explicit verdict-
sensitivity statement. UNLIKE Gate 1 (whose sole reroute, `w37`, is a
straddler already outside the core-30 headline), Gate 2's two reroutes
(`occ0`, `occ11`) ARE both inside the main 16-clip scoreable pool AND are
both FORCED `unsuccessful`-outcome slots (§A.3 — exactly the occluded/
hard-failure population Gate 2 exists to oversample) — excluding them is
NOT a structural no-op here and can plausibly move the verdict.

Run from the repo root (after ``run_gate2.py --execute``):

    .venv/bin/python -m eval.adcc.s12_gates.score_gate2 [--label smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from eval.adcc.s12_gates import gate_common as gc

NET_DELTA_PASS_THRESHOLD = 2  # design §C.3


def _score_clip_pool(clips: list[dict], records_by_id: dict) -> dict:
    """Core per-clip scoring loop (§C.1 exact-class agreement), factored out
    so both the full 16-clip pool and its files_api-excluded sibling (§G.1
    point 4) are computed by the identical code — never two hand-maintained
    copies that could silently drift."""
    rows: list[dict] = []
    excluded: list[str] = []
    correct_low_count = correct_high_count = 0
    n_scored = 0

    for c in clips:
        wid = c["window_id"]
        rec = records_by_id.get(wid)
        gold = c["gold_event"]
        if rec is None or rec["status"] == "call_failed":
            excluded.append(wid)
            rows.append({
                "window_id": wid, "match_id": c["match_id"], "gold_class": gold["action_class"],
                "gold_outcome": gold["outcome"], "pos_label": None, "tech_label": None,
                "val_low_verdict": None, "val_low_final_class": None, "correct_low": None,
                "val_high_verdict": None, "val_high_final_class": None, "correct_high": None,
                "scored": False,
            })
            continue

        # §C.1: exact string equality, not family membership. A 'ditch' verdict
        # (val_*_final_class is None per run_gate2.py's _validator_final_class
        # contract) scores correct=False, never a fabricated match.
        correct_low = (rec["val_low_final_class"] == gold["action_class"])
        correct_high = (rec["val_high_final_class"] == gold["action_class"])
        correct_low_count += int(correct_low)
        correct_high_count += int(correct_high)
        n_scored += 1

        rows.append({
            "window_id": wid, "match_id": c["match_id"], "gold_class": gold["action_class"],
            "gold_outcome": gold["outcome"], "pos_label": rec["position_label"], "tech_label": rec["technique_label"],
            "val_low_verdict": rec["val_low_verdict"], "val_low_final_class": rec["val_low_final_class"],
            "correct_low": correct_low,
            "val_high_verdict": rec["val_high_verdict"], "val_high_final_class": rec["val_high_final_class"],
            "correct_high": correct_high,
            "scored": True,
        })

    accuracy_low = (correct_low_count / n_scored) if n_scored else None
    accuracy_high = (correct_high_count / n_scored) if n_scored else None
    net_delta = correct_high_count - correct_low_count if n_scored else None
    verdict = None
    if net_delta is not None:
        verdict = "PASS" if net_delta >= NET_DELTA_PASS_THRESHOLD else "NO_GO"

    return {
        "n_clips_in_pool": len(clips), "n_scored": n_scored, "excluded_call_failed_or_unrun": excluded,
        "correct_low_count": correct_low_count, "correct_high_count": correct_high_count,
        "accuracy_low": accuracy_low, "accuracy_high": accuracy_high, "net_delta": net_delta,
        "verdict": verdict, "rows": rows,
    }


def score_gate2(label: str = "", rerun: bool = False) -> dict:
    """``rerun=True`` [RE-RUN 2026-07-26]: reads Gate 2's VERSIONED
    (qualified-model) predictions file instead of the flash-lite v1
    canonical."""
    manifest = gc.read_json(gc.GATE2_MANIFEST_PATH)
    clips = manifest["clips"]
    predictions_path = gc.versioned_path(gc.GATE2_PREDICTIONS_PATH) if rerun else gc.GATE2_PREDICTIONS_PATH
    records = gc.read_json(predictions_path) if predictions_path.exists() else []
    records_by_id = {r["window_id"]: r for r in records}

    full = _score_clip_pool(clips, records_by_id)

    # --- §G.1 point 4 dual reporting (Bezos precondition 2, 2026-07-26) ---
    files_api_ids = {r["window_id"] for r in records if r.get("ingestion_path") == "files_api"}
    clips_excl = [c for c in clips if c["window_id"] not in files_api_ids]
    excl = _score_clip_pool(clips_excl, records_by_id)
    n_files_api = len(clips) - len(clips_excl)
    verdict_flips = full["verdict"] != excl["verdict"]
    if n_files_api == 0:
        verdict_sensitivity_note = (
            "NO-OP: zero files_api-ingested clips are present in this run's predictions yet "
            f"(none have been scored) — verdict is identical with/without exclusion: {full['verdict']!r}."
        )
    else:
        verdict_sensitivity_note = (
            f"{n_files_api} files_api clip(s) excluded (of {len(clips)}) -> "
            f"verdict {full['verdict']!r} (full) vs {excl['verdict']!r} (excl. files_api). "
            + ("VERDICT FLIPS ON EXCLUSION — reportable finding, not a footnote."
               if verdict_flips else "Verdict does NOT flip on exclusion.")
        )

    return {
        "gate": "gate2", "label": label, "rerun": rerun,
        "n_clips_in_manifest": len(clips),
        "n_scored": full["n_scored"], "excluded_call_failed_or_unrun": full["excluded_call_failed_or_unrun"],
        "correct_low_count": full["correct_low_count"], "correct_high_count": full["correct_high_count"],
        "accuracy_low": full["accuracy_low"], "accuracy_high": full["accuracy_high"],
        "net_delta": full["net_delta"], "verdict": full["verdict"], "rows": full["rows"],
        "files_api_dual_reporting": {
            "files_api_window_ids": sorted(files_api_ids),
            "n_files_api_clips": n_files_api,
            "n_clips_excl_files_api": len(clips_excl),
            "n_scored_excl_files_api": excl["n_scored"],
            "accuracy_low_excl_files_api": excl["accuracy_low"],
            "accuracy_high_excl_files_api": excl["accuracy_high"],
            "net_delta_excl_files_api": excl["net_delta"],
            "verdict_excl_files_api": excl["verdict"],
            "verdict_flips_on_exclusion": verdict_flips,
            "verdict_sensitivity_note": verdict_sensitivity_note,
        },
    }


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def write_report(result: dict, out_path: Path) -> None:
    lines = []
    lines.append("# Gate 2 Report — validator media_resolution (LOW vs HIGH) on the 16-clip occlusion subset")
    lines.append("")
    if result["label"]:
        lines.append(f"**RUN LABEL: `{result['label']}`** — n_scored={result['n_scored']} of "
                      f"{result['n_clips_in_manifest']} manifest clips. This is NOT the full 16-clip batch; "
                      "the verdict below demonstrates the scoring PIPELINE only, not a real gate decision.")
        lines.append("")
    lines.append("## Settings tuple (design §D)")
    lines.append("")
    lines.append("| | fps | media_resolution | model | thinking |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| position/technique | {gc.FPS} | None (LOW) | {gc.MODEL_ID} | off |")
    lines.append(f"| validator-low | {gc.FPS} | low | {gc.MODEL_ID} | off |")
    lines.append(f"| validator-high | {gc.FPS} | high | {gc.MODEL_ID} | off |")
    lines.append("")
    lines.append("## Per-clip table (§C.2, mandatory — no bare aggregate)")
    lines.append("")
    lines.append("| match | gold_class | gold_outcome | pos_label | tech_label | val_low_verdict | "
                  "val_low_final_class | correct_low | val_high_verdict | val_high_final_class | correct_high |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in result["rows"]:
        if not r["scored"]:
            lines.append(f"| {r['match_id']} | {r['gold_class']} | {r['gold_outcome']} | "
                          "_(call_failed / not run)_ | | | | | | | |")
            continue
        lines.append(
            f"| {r['match_id']} | {r['gold_class']} | {r['gold_outcome']} | {r['pos_label']} | {r['tech_label']} | "
            f"{r['val_low_verdict']} | {r['val_low_final_class']} | {r['correct_low']} | "
            f"{r['val_high_verdict']} | {r['val_high_final_class']} | {r['correct_high']} |"
        )
    lines.append("")
    lines.append(f"N scored: {result['n_scored']}/{result['n_clips_in_manifest']} "
                  f"(excluded call_failed/unrun: {result['excluded_call_failed_or_unrun']}).")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if result["verdict"] is None:
        lines.append("**NOT SCOREABLE — n_scored=0.**")
    else:
        lines.append(
            f"accuracy_low={_fmt_pct(result['accuracy_low'])} ({result['correct_low_count']}/{result['n_scored']}), "
            f"accuracy_high={_fmt_pct(result['accuracy_high'])} ({result['correct_high_count']}/{result['n_scored']}), "
            f"net_delta={result['net_delta']} (pass threshold: >= {NET_DELTA_PASS_THRESHOLD}) -> "
            f"**{result['verdict']} (directional, n={result['n_scored']})**"
        )
    lines.append("")
    lines.append(
        "**Directional, small-n caveat:** this gate's n=16-by-design sample is deliberately tiny "
        "(design §C.3) — a `net_delta` swing of 1-2 clips is the resolution limit at this n. The "
        "PASS/NO-GO call above is binary per the design's own threshold, but should be read as "
        "directional evidence, not a high-confidence estimate; a larger occlusion-subset re-run is "
        "reasonable post-launch tuning, not a cutover blocker."
    )
    lines.append("")
    lines.append("## §G.1 point 4 dual reporting — headline recomputed excluding files_api-ingested clips")
    lines.append("")
    lines.append(
        "Unlike Gate 1 (whose sole reroute, `w37`, is a straddler already outside the headline), Gate "
        "2's two reroutes (`occ0`/`occ11`) are BOTH inside the main 16-clip pool AND are both forced "
        "`unsuccessful`-outcome slots (§A.3's oversampled occluded population) — exclusion is not a "
        "structural no-op here."
    )
    lines.append("")
    dr = result["files_api_dual_reporting"]
    lines.append(f"files_api-ingested clip(s) in this run: `{dr['files_api_window_ids']}`.")
    lines.append("")
    if dr["verdict_excl_files_api"] is None:
        lines.append(f"EXCLUDING files_api clips (n_clips={dr['n_clips_excl_files_api']}, "
                      f"n_scored={dr['n_scored_excl_files_api']}): **NOT SCOREABLE — n_scored=0.**")
    else:
        lines.append(
            f"EXCLUDING files_api clips (n_clips={dr['n_clips_excl_files_api']}, "
            f"n_scored={dr['n_scored_excl_files_api']}): "
            f"accuracy_low={_fmt_pct(dr['accuracy_low_excl_files_api'])}, "
            f"accuracy_high={_fmt_pct(dr['accuracy_high_excl_files_api'])}, "
            f"net_delta={dr['net_delta_excl_files_api']} -> "
            f"**verdict: {dr['verdict_excl_files_api']} (directional, n={dr['n_scored_excl_files_api']})**"
        )
    lines.append("")
    lines.append(f"**VERDICT SENSITIVITY: {dr['verdict_sensitivity_note']}**")
    lines.append("")
    out_path.write_text("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="", help="Non-empty for a partial/smoke run — suffixes output filenames.")
    parser.add_argument("--rerun", action="store_true",
                        help="[RE-RUN 2026-07-26] Score the qualified-model re-run — reads versioned predictions, "
                             "writes gc.RERUN_LABEL-suffixed outputs, never touches v1 canonicals.")
    args = parser.parse_args(argv)

    result = score_gate2(label=args.label, rerun=args.rerun)
    suffix = f".{args.label}" if args.label else (f".{gc.RERUN_LABEL}" if args.rerun else "")
    scoring_path = gc.S12_GATES_DIR / f"gate2_scoring{suffix}.json"
    report_path = gc.S12_GATES_DIR / f"GATE2_REPORT{suffix}.md"

    gc.write_json(scoring_path, result)
    write_report(result, report_path)

    print(f"wrote {scoring_path}", file=sys.stderr)
    print(f"wrote {report_path}", file=sys.stderr)
    print(f"verdict={result['verdict']} n_scored={result['n_scored']}", file=sys.stderr)


if __name__ == "__main__":
    main()
