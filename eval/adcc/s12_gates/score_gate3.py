"""Gate 3 scoring — Files-API ingestion-path fidelity, per `GATES_3_4_DESIGN.md`
§H.5 as amended (`VERDICTS.md` §4): primary pairing is Arm S (YouTube,
already-captured in Gate 1, vs. new Files-API run); secondary is a reduced
Arm-T spot-check (n=3); plus the fps-honoring re-confirmation (§H.3) and the
zero-incremental-cost bonus finding (§H.4).

**Per-window "final_action_class" for Arm S (design ambiguity, flagged —
§H.5's original text only defines this for Arm T's single flat verdict; Arm
S emits a LIST of clips, which has no built-in single "verdict").**
Conservative, code-reused resolution: run `matcher.match_one` on the
window's one gold event vs. that path's emitted events (same
`filter_scoreable`/`compat_classes_for` discipline Gate 1's own Track B
uses) — if a TP exists, `final_action_class` is the TP-matched prediction's
class; if only a time-feasible-but-class-incompatible prediction exists
(`fn_misclassified`), use its nearest predicted class (the same "smoking
gun" diagnostic `metrics.py` already surfaces); if nothing is
time-feasible at all, `final_action_class=None`. This reuses the EXACT
matching primitive Gate 1 already trusts, rather than inventing a new
"most confident clip" heuristic.

Run from the repo root (after `run_gate3.py --execute`):

    .venv/bin/python -m eval.adcc.s12_gates.score_gate3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from eval.adcc.data_prep.apply_gracie_mapping import compat_classes_for, load_mapping
from eval.adcc.s12_gates import gate_common as gc
from eval.adcc.score import matcher

AGREEMENT_RATE_BAR = 6 / 8  # §H.5 PASS condition (a)
ACCURACY_SLACK = 1  # §H.5 PASS condition (b) — files_api may be at most 1 window worse


def _window_outcome(gold_event: dict, emitted_events: list[dict], mapping: dict) -> tuple[bool, Optional[str]]:
    """Returns ``(hit, final_action_class)`` for ONE window/path, per the
    module docstring's matcher-reuse resolution."""
    non_scoreable = frozenset(mapping["non_scoreable_prediction_classes"])
    preds = [e for e in emitted_events if e.get("action_class") not in non_scoreable]
    compat = compat_classes_for(gold_event, mapping)
    gold = {**gold_event, "compat": compat, "match_id": "w"}
    tagged_preds = [{**p, "match_id": "w"} for p in preds]
    outcome = matcher.match_one([gold], tagged_preds, tau_t=gc.TAU_T)
    if outcome.tp:
        _, pi = outcome.tp[0]
        return True, tagged_preds[pi]["action_class"]
    if outcome.fn_misclassified:
        nearest_pi = outcome.fn_misclassified_nearest_pred[0]
        return False, tagged_preds[nearest_pi]["action_class"]
    return False, None


def _pair_rows(pairs: list[dict], youtube_key: str, files_api_key: str, mapping: dict) -> list[dict]:
    """Shared per-window pairing logic for BOTH the Arm-S primary pool and
    the Arm-T spot-check pool — same code, different input records."""
    rows = []
    for p in pairs:
        gold = p["gold_event"]
        yt_rec = p[youtube_key]
        fa_rec = p[files_api_key]
        yt_status = yt_rec["status"] if yt_rec else "missing"
        fa_status = fa_rec["status"] if fa_rec else "missing"
        yt_hit = yt_class = fa_hit = fa_class = None
        scored = yt_status not in ("call_failed", "missing") and fa_status not in ("call_failed", "missing")
        if scored:
            yt_hit, yt_class = _window_outcome(gold, yt_rec["emitted_events"], mapping)
            fa_hit, fa_class = _window_outcome(gold, fa_rec["emitted_events"], mapping)
        rows.append({
            "window_id": p["window_id"], "match_id": p["match_id"], "gold_class": gold["action_class"],
            "gold_outcome": gold["outcome"], "youtube_status": yt_status, "files_api_status": fa_status,
            "scored": scored,
            "youtube_final_class": yt_class, "youtube_hit": yt_hit,
            "files_api_final_class": fa_class, "files_api_hit": fa_hit,
            "agree": (yt_class == fa_class) if scored else None,
            "is_replacement": p.get("is_replacement", False),
            "replaced_window_id": p.get("replaced_window_id"),
        })
    return rows


def score_gate3(rerun: bool = False) -> dict:
    """``rerun=True`` [RE-RUN 2026-07-26]: reads Gate 3's VERSIONED
    (qualified-model) predictions file instead of the flash-lite v1
    canonical."""
    predictions_path = gc.S12_GATES_DIR / "gate3_predictions.json"
    upload_metrics_path = gc.S12_GATES_DIR / "gate3_upload_metrics.json"
    if rerun:
        predictions_path = gc.versioned_path(predictions_path)
        upload_metrics_path = gc.versioned_path(upload_metrics_path)
    predictions = gc.read_json(predictions_path)
    mapping = load_mapping()

    primary_rows = _pair_rows(predictions["primary_arm_s"], "youtube_arm_s", "files_api_arm_s", mapping)
    n_primary_scored = sum(1 for r in primary_rows if r["scored"])
    agreement_count = sum(1 for r in primary_rows if r["scored"] and r["agree"])
    accuracy_youtube_count = sum(1 for r in primary_rows if r["scored"] and r["youtube_hit"])
    accuracy_files_api_count = sum(1 for r in primary_rows if r["scored"] and r["files_api_hit"])
    agreement_rate = (agreement_count / n_primary_scored) if n_primary_scored else None
    accuracy_youtube = (accuracy_youtube_count / n_primary_scored) if n_primary_scored else None
    accuracy_files_api = (accuracy_files_api_count / n_primary_scored) if n_primary_scored else None
    accuracy_delta = (accuracy_files_api - accuracy_youtube) if n_primary_scored else None

    pass_condition_a = (agreement_rate is not None and agreement_rate >= AGREEMENT_RATE_BAR)
    pass_condition_b = (
        n_primary_scored > 0 and accuracy_files_api_count >= accuracy_youtube_count - ACCURACY_SLACK
    )
    design_pass_bar = bool(pass_condition_a and pass_condition_b) if n_primary_scored else None

    spot_check_rows = _pair_rows(predictions["spot_check_arm_t"], "youtube_arm_t", "files_api_arm_t", mapping)
    spot_check_agree = sum(1 for r in spot_check_rows if r["scored"] and r["agree"])
    spot_check_n_scored = sum(1 for r in spot_check_rows if r["scored"])

    # --- §H.3 fps-honoring re-confirmation ---
    fps_rows = []
    n_detectable_strong = 0
    for f in predictions["fps_check"]:
        fps1 = f["fps1"]["parsed"] if f["fps1"]["status"] == "ok" else None
        fps10 = f["fps10"]["parsed"] if f["fps10"]["status"] == "ok" else None
        class1 = fps1.get("action_class") if fps1 else None
        class10 = fps10.get("action_class") if fps10 else None
        just1 = (fps1.get("justification") or "").strip() if fps1 else None
        just10 = (fps10.get("justification") or "").strip() if fps10 else None
        detectable_strong = bool(fps1 and fps10 and class1 != class10)
        detectable_weak = bool(fps1 and fps10 and just1 != just10)
        if detectable_strong:
            n_detectable_strong += 1
        fps_rows.append({
            "window_id": f["window_id"], "match_id": f["match_id"],
            "fps1_status": f["fps1"]["status"], "fps1_action_class": class1, "fps1_justification": just1,
            "fps10_status": f["fps10"]["status"], "fps10_action_class": class10, "fps10_justification": just10,
            "detectable_diff_strong_action_class": detectable_strong,
            "detectable_diff_weak_justification_text": detectable_weak,
        })
    fps_honoring_confirmed = n_detectable_strong >= 1  # §H.3's stated bar: >=1 of 2 clips

    return {
        "gate": "gate3", "rerun": rerun,
        "primary_arm_s": {
            "n_windows": len(primary_rows), "n_scored": n_primary_scored,
            "agreement_count": agreement_count, "agreement_rate": agreement_rate,
            "accuracy_youtube_count": accuracy_youtube_count, "accuracy_youtube": accuracy_youtube,
            "accuracy_files_api_count": accuracy_files_api_count, "accuracy_files_api": accuracy_files_api,
            "accuracy_delta": accuracy_delta,
            "design_pass_bar_agreement_ge_6_of_8": pass_condition_a,
            "design_pass_bar_accuracy_no_material_regression": pass_condition_b,
            "design_pass_bar_met": design_pass_bar,
            "rows": primary_rows,
        },
        "spot_check_arm_t": {
            "n_windows": len(spot_check_rows), "n_scored": spot_check_n_scored,
            "agreement_count": spot_check_agree, "rows": spot_check_rows,
        },
        "fps_honoring": {
            "n_clips": len(fps_rows), "n_detectable_diff_strong": n_detectable_strong,
            "confirmed": fps_honoring_confirmed, "rows": fps_rows,
        },
        "bonus_finding": predictions["bonus_finding"],
        "replacement": predictions.get("replacement"),
        "upload_metrics": (
            gc.read_json(upload_metrics_path) if upload_metrics_path.exists() else []
        ),
    }


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def write_report(result: dict, out_path: Path) -> None:
    lines = []
    lines.append("# Gate 3 Report — Files-API ingestion-path fidelity (Arm-S primary, per VERDICTS.md §4 amendment)")
    lines.append("")
    repl = result.get("replacement")
    if repl:
        lines.append(
            f"**§G.2 REPLACEMENT IN THIS RUN — NOT a silent substitution.** `{repl['replaced_window_id']}` "
            f"was replaced by `{repl['replacement_window_id']}`. Reason: {repl['reason']}"
        )
        lines.append("")
    lines.append("## Upload metrics (upload_latency_ms / active_wait_ms per window)")
    lines.append("")
    lines.append("| window | file_name | upload_latency_ms | active_wait_ms | sec_start_s | is_replacement |")
    lines.append("|---|---|---|---|---|---|")
    for u in result.get("upload_metrics", []):
        lines.append(f"| {u['window_id']} | {u['file_name']} | {u['upload_latency_ms']:.0f} | "
                      f"{u['active_wait_ms']:.0f} | {u['sec_start_s']} | {u.get('is_replacement', False)} |")
    lines.append("")
    lines.append("## Settings tuple")
    lines.append("")
    lines.append("| | fps | media_resolution | model | thinking | ingestion (Files-API side) |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| Arm S (primary) | 10 | None (LOW) | {gc.MODEL_ID} | off | files_api (mime_type+file-relative offsets) |")
    lines.append(f"| Arm T (spot-check) position/technique | 10 | None (LOW) | {gc.MODEL_ID} | off | files_api |")
    lines.append(f"| Arm T (spot-check) validator | 10 | high (prod default) | {gc.MODEL_ID} | off | files_api |")
    lines.append(f"| fps-honoring (technique only) | 1 and 10 | None (LOW) | {gc.MODEL_ID} | off | files_api |")
    lines.append("")
    p = result["primary_arm_s"]
    lines.append("## PRIMARY — Arm S, YouTube-URI (Gate 1, already-captured) vs Files-API (new)")
    lines.append("")
    lines.append("| window | gold_class | youtube_final_class | youtube_hit | files_api_final_class | "
                  "files_api_hit | agree |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in p["rows"]:
        wid_label = f"{r['window_id']} **[§G.2 REPLACEMENT]**" if r.get("is_replacement") else r["window_id"]
        if not r["scored"]:
            lines.append(f"| {wid_label} | {r['gold_class']} | _(youtube={r['youtube_status']}, "
                          f"files_api={r['files_api_status']})_ | | | | |")
            continue
        lines.append(
            f"| {wid_label} | {r['gold_class']} | {r['youtube_final_class']} | {r['youtube_hit']} | "
            f"{r['files_api_final_class']} | {r['files_api_hit']} | {r['agree']} |"
        )
    lines.append("")
    lines.append(
        f"N scored: {p['n_scored']}/{p['n_windows']}. agreement_rate={_fmt_pct(p['agreement_rate'])} "
        f"({p['agreement_count']}/{p['n_scored'] if p['n_scored'] else 0}), "
        f"accuracy_youtube={_fmt_pct(p['accuracy_youtube'])} ({p['accuracy_youtube_count']}), "
        f"accuracy_files_api={_fmt_pct(p['accuracy_files_api'])} ({p['accuracy_files_api_count']}), "
        f"accuracy_delta={_fmt_pct(p['accuracy_delta']) if p['accuracy_delta'] is not None else 'n/a'}."
    )
    lines.append("")
    lines.append(
        f"**Design's own numeric PASS bar (§H.5, reported mechanically — NOT a rendered gate verdict, "
        f"that is LeCun's call):** condition (a) agreement_rate>=75%: {p['design_pass_bar_agreement_ge_6_of_8']}; "
        f"condition (b) accuracy_files_api >= accuracy_youtube-1: "
        f"{p['design_pass_bar_accuracy_no_material_regression']}; both required: **{p['design_pass_bar_met']}**."
    )
    lines.append("")
    s = result["spot_check_arm_t"]
    lines.append("## SECONDARY — Arm T spot-check (n=3, reduced per VERDICTS.md §4, insurance only)")
    lines.append("")
    lines.append("| window | gold_class | youtube_final_class | youtube_hit | files_api_final_class | "
                  "files_api_hit | agree |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in s["rows"]:
        if not r["scored"]:
            lines.append(f"| {r['window_id']} | {r['gold_class']} | _(youtube={r['youtube_status']}, "
                          f"files_api={r['files_api_status']})_ | | | | |")
            continue
        lines.append(
            f"| {r['window_id']} | {r['gold_class']} | {r['youtube_final_class']} | {r['youtube_hit']} | "
            f"{r['files_api_final_class']} | {r['files_api_hit']} | {r['agree']} |"
        )
    lines.append("")
    lines.append(f"N scored: {s['n_scored']}/{s['n_windows']}, agreement: {s['agreement_count']}/{s['n_scored']}. "
                  "NOT part of the n=8 fidelity sample — insurance-only secondary check.")
    lines.append("")
    f = result["fps_honoring"]
    lines.append("## §H.3 — INS-096 fps-honoring re-confirmation on the Files-API path")
    lines.append("")
    lines.append(
        "**Operationalization note (design says 'compare the two responses'; this scorer's precise "
        "rule, flagged):** `detectable_diff_strong` = the two fps' `action_class` differ (a real judgment "
        "change — the strong, primary signal). `detectable_diff_weak` = only the `justification` text "
        "differs (action_class the same) — reported as supplementary color, NOT treated as dispositive on "
        "its own, since `temperature=0.2` calls can produce different wording by sampling noise alone even "
        "with zero real fps effect; using text-difference as the primary bar would make this check "
        "trivially always pass."
    )
    lines.append("")
    lines.append("| window | fps1_action_class | fps10_action_class | detectable_diff (action_class) | "
                  "detectable_diff (justification text) |")
    lines.append("|---|---|---|---|---|")
    for r in f["rows"]:
        lines.append(f"| {r['window_id']} | {r['fps1_action_class']} | {r['fps10_action_class']} | "
                      f"{r['detectable_diff_strong_action_class']} | {r['detectable_diff_weak_justification_text']} |")
    lines.append("")
    lines.append(f"{f['n_detectable_diff_strong']}/{f['n_clips']} clips show a strong (action_class) detectable "
                  f"difference. Design's confirmation bar (>=1 of 2): **{'CONFIRMED' if f['confirmed'] else 'NOT CONFIRMED'}**.")
    lines.append("")
    lines.append("## §H.4 — Bonus finding (zero incremental cost, NOT part of the n=8 sample)")
    lines.append("")
    lines.append(result["bonus_finding"]["note"])
    lines.append(f"Windows: `{result['bonus_finding']['windows']}`. Source artifacts: "
                  f"`{result['bonus_finding']['source_artifacts']}`.")
    lines.append("")
    out_path.write_text("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun", action="store_true",
                        help="[RE-RUN 2026-07-26] Score the qualified-model re-run — reads versioned "
                             "predictions, writes gc.RERUN_LABEL-suffixed outputs, never touches v1 canonicals.")
    args = parser.parse_args(argv)

    result = score_gate3(rerun=args.rerun)
    suffix = f".{gc.RERUN_LABEL}" if args.rerun else ""
    scoring_path = gc.S12_GATES_DIR / f"gate3_scoring{suffix}.json"
    report_path = gc.S12_GATES_DIR / f"GATE3_REPORT{suffix}.md"
    gc.write_json(scoring_path, result)
    write_report(result, report_path)
    print(f"wrote {scoring_path}", file=sys.stderr)
    print(f"wrote {report_path}", file=sys.stderr)
    print(f"agreement_rate={result['primary_arm_s']['agreement_rate']} "
          f"accuracy_delta={result['primary_arm_s']['accuracy_delta']} "
          f"fps_honoring_confirmed={result['fps_honoring']['confirmed']}", file=sys.stderr)


if __name__ == "__main__":
    main()
