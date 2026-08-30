"""LeCun cutover Gate 1 scoring (``GATES_1_2_DESIGN.md`` §B) + Gate 5
addendum (§E.3), as amended by Bezos's 2026-07-25 pass-with-conditions
review (conditions 1-4 below; condition 5 was a sampling-RNG change already
folded into ``gate_common.build_gate2_subset`` before any manifest was
frozen and does not affect this scoring module).

**Bezos condition 2 (headline scope).** The headline PASS/NO-GO is computed
on the 30-event STRATIFIED CORE alone, never the full 38 (the 8 forced
Gate-5 straddlers are a deliberately class-skewed, non-representative
addition — see design §A.2 Step 2 — and mixing them into the headline
accuracy comparison would bias it). The required margin is re-derived
PROPORTIONALLY from the design's own ``hit_T - hit_S >= 4`` rule on n=38:
``required_margin(n) = ceil((4/38) * n)``, which evaluates to 4 again at
n=30 (3.16 -> ceil -> 4) — coincidence of the arithmetic, not a hardcoded
re-statement (see ``required_margin_for_n``). The full-38 number (including
the straddlers) is still computed and reported, but labeled explicitly as
SECONDARY, never the pass bar.

**Bezos condition 3 (Gate 1/Gate 2 coupling).** If Gate 1's OWN criteria
would PASS on the core-30 but Gate 2 (``gate2_scoring.json``, if already
written) returned NO-GO (HIGH validator resolution not justified — the
production default would drop to LOW), Gate 1's verdict is downgraded to
``PASS_PROVISIONAL`` — a distinct, structured ``coupled_verdict`` field, not
prose — because Arm T's real-contract validator ran at HIGH (§B.2, the
production default), and a production cutover that ships LOW instead would
not be the exact same system Gate 1 measured. The cross-check evidence is
Gate 2's OWN already-collected validator-low numbers on the 16-clip
occlusion subset (``accuracy_low``/``accuracy_high``/``net_delta``) — ZERO
new Gemini calls, per the founder's explicit "no extra spend" instruction.

**Bezos condition 4 (inline qualifiers).** The sign-test p-value is printed
INLINE on Gate 1's verdict line in ``GATE1_REPORT.md``, not only in a
separate table.

Run from the repo root (after ``run_gate1.py --execute`` has produced
predictions — including a partial/smoke run, which this script scores
honestly as a partial-N sample, never silently presented as a full-N gate
verdict):

    .venv/bin/python -m eval.adcc.s12_gates.score_gate1 [--label smoke]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

from scipy.stats import binomtest

from eval.adcc.data_prep.apply_gracie_mapping import load_mapping
from eval.adcc.s12_gates import gate_common as gc
from eval.adcc.score import highlight_baselines, matcher, metrics

DESIGN_MARGIN_HIT_DELTA = 4
DESIGN_MARGIN_N = 38
DESIGN_FP_DUP_CEILING = 0.05
GATE5_FLIP_THRESHOLD_PP = 25.0
GATE5_FALLBACK_MIN_SURVIVORS = 5


def required_margin_for_n(n: int) -> int:
    """Bezos condition 2's proportional re-derivation: the smallest integer
    margin that is >= the design's own ~10.5 percentage-point bar
    (``4/38``), applied to a subset of size ``n``. ``ceil`` (never floor/
    round) so the re-derived bar is never LOOSER than the original design's
    intent at a smaller n."""
    return math.ceil((DESIGN_MARGIN_HIT_DELTA / DESIGN_MARGIN_N) * n)


def _build_arm_buckets(windows: list[dict], arm_records: list[dict]) -> tuple[dict, dict, list[str]]:
    """Per-window synthetic-match-id buckets (§B.3: ``f"{match_id}__w{i}"``
    — already exactly ``window['window_id']``, minted at manifest-freeze
    time). Windows whose arm record is missing (not yet run — smoke/partial
    scoring) or ``status == 'call_failed'`` are EXCLUDED from both
    dicts (§D.1) and returned in the third element."""
    records_by_id = {r["window_id"]: r for r in arm_records}
    gold_by_match: dict[str, list[dict]] = {}
    pred_by_match: dict[str, list[dict]] = {}
    excluded: list[str] = []
    for w in windows:
        wid = w["window_id"]
        rec = records_by_id.get(wid)
        if rec is None:
            excluded.append(wid)
            continue
        if rec["status"] == "call_failed":
            excluded.append(wid)
            continue
        g = w["gold_event"]
        gold_by_match[wid] = [{**g, "match_id": wid}]
        pred_by_match[wid] = [{**e, "match_id": wid} for e in rec["emitted_events"]]
    return gold_by_match, pred_by_match, excluded


def _score_track_b(gold_by_match: dict, pred_by_match: dict, mapping: dict) -> tuple[dict, dict]:
    """§B.3: attach_compat -> filter_scoreable -> match_all(tau_t=5.0) ->
    summarize. Returns ``(outcomes, summary)``."""
    gold_c = metrics.attach_compat(gold_by_match, mapping)
    non_scoreable = frozenset(mapping["non_scoreable_prediction_classes"])
    gold_f, pred_f = metrics.filter_scoreable(gold_c, pred_by_match, non_scoreable)
    outcomes = matcher.match_all(gold_f, pred_f, gc.TAU_T)
    summary = metrics.summarize(outcomes, gold_f, pred_f)
    return outcomes, summary


def _window_hits(outcomes: dict) -> dict[str, bool]:
    return {wid: bool(outcome.tp) for wid, outcome in outcomes.items()}


def _fp_dup_rate(summary: dict) -> Optional[float]:
    pred_count = summary["totals"]["pred_count"]
    if pred_count == 0:
        return None
    return summary["fp_breakdown"]["duplicate"] / pred_count


def apply_gate2_coupling(own_verdict: str, gate2_scoring_path: Path) -> tuple[str, str]:
    """Bezos condition 3 (2026-07-25 pass-with-conditions review): if Gate
    1's own-criteria verdict is PASS, but Gate 2's ALREADY-WRITTEN
    ``gate2_scoring.json`` (the canonical, non-labeled path — a smoke/
    partial-run's ``--label``-suffixed file never satisfies this coupling
    check by design, see ``score_gate2.py``) says NO_GO, downgrade to a
    distinct ``PASS_PROVISIONAL_PENDING_GATE2_CONFIRM`` state. Pulled out as
    a standalone, directly-unit-testable function (isolated from I/O other
    than the one path-existence + read) rather than inlined — a coupling
    rule this load-bearing deserves to be independently verifiable without
    needing a full real Gate 1 PASS run to exercise it. NO_GO's own-verdict
    never triggers coupling (a NO_GO stays NO_GO regardless of Gate 2)."""
    if own_verdict != "PASS":
        return own_verdict, "own_verdict is not PASS — coupling check does not apply."
    if not gate2_scoring_path.exists():
        return own_verdict, "gate2_scoring.json not found — coupling check pending (run score_gate2.py first)."
    gate2 = gc.read_json(gate2_scoring_path)
    gate2_verdict = gate2.get("verdict")
    if gate2_verdict != "NO_GO":
        return own_verdict, f"Gate 2 verdict={gate2_verdict!r} — no coupling downgrade."
    note = (
        f"Gate 2 returned NO_GO (net_delta={gate2.get('net_delta')}, "
        f"accuracy_low={gate2.get('accuracy_low')}, accuracy_high={gate2.get('accuracy_high')} "
        f"on n={gate2.get('n_scored')}) — production would ship validator media_resolution=LOW, "
        "not the HIGH Arm T actually used here. Cross-check with Gate 2's own validator-low "
        "numbers on the occlusion subset before treating this PASS as final; a confirmatory "
        "re-run is recommended (zero new spend needed for THIS flag — Gate 2's calls are "
        "already collected)."
    )
    return "PASS_PROVISIONAL_PENDING_GATE2_CONFIRM", note


def _sign_test(hit_t: dict[str, bool], hit_s: dict[str, bool], window_ids: list[str]) -> dict:
    """Two-sided exact sign test on paired (T-hit, S-hit) outcomes over
    ``window_ids`` present in BOTH ``hit_t`` and ``hit_s`` (a window scored
    for only one arm cannot be paired — excluded, not treated as a tie).
    Ties (both hit or both miss) are excluded per design §B.4."""
    n_plus = n_minus = n_tie = n_unpaired = 0
    for wid in window_ids:
        if wid not in hit_t or wid not in hit_s:
            n_unpaired += 1
            continue
        t, s = hit_t[wid], hit_s[wid]
        if t and not s:
            n_plus += 1
        elif s and not t:
            n_minus += 1
        else:
            n_tie += 1
    n = n_plus + n_minus
    p_value = binomtest(min(n_plus, n_minus), n, 0.5, alternative="two-sided").pvalue if n > 0 else None
    return {
        "n_plus_t_wins": n_plus, "n_minus_s_wins": n_minus, "n_ties_excluded": n_tie,
        "n_unpaired_excluded": n_unpaired, "n_effective": n, "p_value": p_value,
    }


def _files_api_window_ids(*record_lists: list[dict]) -> set[str]:
    """§G.1 point 4 dual reporting: the set of window_ids where ANY arm's
    record used the Files API reroute lane (``ingestion_path=="files_api"``)
    — union across arm_s/arm_t, so the excluded-scope denominator is the
    SAME for both arms (a window rerouted for one arm and not the other
    should never happen by construction, §G.1 point 3, but this stays
    correct even if it somehow did)."""
    ids: set[str] = set()
    for records in record_lists:
        for rec in records:
            if rec.get("ingestion_path") == "files_api":
                ids.add(rec["window_id"])
    return ids


def _score_scope(scope_windows: list[dict], arm_s_records: list[dict], arm_t_records: list[dict], mapping: dict) -> dict:
    """One scope's full score bundle (hit_S/hit_T, fp_dup rates, sign test,
    per-arm ``metrics.summarize``, null baselines) — factored out so both
    the raw scope and its files_api-excluded sibling (§G.1 point 4) are
    computed by the exact same code, never two hand-maintained copies."""
    gold_s, pred_s, excl_s = _build_arm_buckets(scope_windows, arm_s_records)
    gold_t, pred_t, excl_t = _build_arm_buckets(scope_windows, arm_t_records)
    outcomes_s, summary_s = _score_track_b(gold_s, pred_s, mapping) if gold_s else ({}, None)
    outcomes_t, summary_t = _score_track_b(gold_t, pred_t, mapping) if gold_t else ({}, None)
    hit_s = _window_hits(outcomes_s)
    hit_t = _window_hits(outcomes_t)
    paired_ids = [w["window_id"] for w in scope_windows]
    sign_test = _sign_test(hit_t, hit_s, paired_ids)
    fp_dup_s = _fp_dup_rate(summary_s) if summary_s else None
    fp_dup_t = _fp_dup_rate(summary_t) if summary_t else None
    baselines = _baseline_summaries(scope_windows, gold_s, mapping) if gold_s else None

    return {
        "n_windows_in_scope": len(scope_windows),
        "n_scored_arm_s": len(gold_s), "n_scored_arm_t": len(gold_t),
        "excluded_arm_s_call_failed_or_unrun": excl_s,
        "excluded_arm_t_call_failed_or_unrun": excl_t,
        "hit_s": sum(hit_s.values()), "hit_t": sum(hit_t.values()),
        "hit_delta": sum(hit_t.values()) - sum(hit_s.values()),
        "fp_dup_rate_s": fp_dup_s, "fp_dup_rate_t": fp_dup_t,
        "sign_test": sign_test,
        "summary_arm_s": summary_s, "summary_arm_t": summary_t,
        "baselines": baselines,
    }


def _own_verdict_from_scope(scope: dict) -> tuple[str, list[str]]:
    """§B.4's numeric PASS rule, applied to ANY scope dict (headline core30,
    or its files_api-excluded sibling) — factored out so both are judged by
    the identical rule, never a hand-duplicated copy that could drift."""
    required_margin = required_margin_for_n(scope["n_windows_in_scope"])
    hit_delta_ok = scope["hit_delta"] >= required_margin
    fp_dup_ok = (
        scope["fp_dup_rate_t"] is not None and scope["fp_dup_rate_s"] is not None
        and scope["fp_dup_rate_t"] <= scope["fp_dup_rate_s"] + DESIGN_FP_DUP_CEILING
    ) if (scope["fp_dup_rate_s"] is not None) else (scope["fp_dup_rate_t"] in (None, 0.0))
    verdict = "PASS" if (hit_delta_ok and fp_dup_ok) else "NO_GO"
    reasons = []
    if not hit_delta_ok:
        reasons.append(
            f"hit_T-hit_S={scope['hit_delta']} < required_margin={required_margin} (n={scope['n_windows_in_scope']})"
        )
    if not fp_dup_ok:
        reasons.append(
            f"fp_dup_rate_T={scope['fp_dup_rate_t']} > fp_dup_rate_S={scope['fp_dup_rate_s']} + {DESIGN_FP_DUP_CEILING}"
        )
    return verdict, reasons


def _baseline_summaries(windows: list[dict], gold_by_match: dict, mapping: dict) -> dict:
    """§F.3: whole_match/uniform_span null baselines over the SAME window
    pool actually scored for this scope (core or full)."""
    match_windows = {w["window_id"]: (w["window_start_s"], w["window_end_s"])
                      for w in windows if w["window_id"] in gold_by_match}
    results = {}
    for name, builder in (("whole_match", highlight_baselines.whole_match_baseline),
                           ("uniform_span", highlight_baselines.uniform_span_baseline)):
        pred_by_match = builder(match_windows)
        _outcomes, summary = _score_track_b(gold_by_match, pred_by_match, mapping)
        results[name] = summary
    return results


def score_gate1(label: str = "", rerun: bool = False) -> dict:
    """``rerun=True`` [RE-RUN 2026-07-26]: reads Gate 1's VERSIONED
    (qualified-model) predictions files (`gc.versioned_path(...)`) instead
    of the flash-lite v1 canonicals — never mixes v1 predictions with a
    rerun scoring pass."""
    manifest = gc.read_json(gc.GATE1_MANIFEST_PATH)
    windows = manifest["windows"]
    arm_s_path = gc.versioned_path(gc.GATE1_ARM_S_PATH) if rerun else gc.GATE1_ARM_S_PATH
    arm_t_path = gc.versioned_path(gc.GATE1_ARM_T_PATH) if rerun else gc.GATE1_ARM_T_PATH
    arm_s_records = gc.read_json(arm_s_path) if arm_s_path.exists() else []
    arm_t_records = gc.read_json(arm_t_path) if arm_t_path.exists() else []
    mapping = load_mapping()

    core_windows = [w for w in windows if not w["gate5_straddler"]]
    straddler_windows = [w for w in windows if w["gate5_straddler"]]

    files_api_ids = _files_api_window_ids(arm_s_records, arm_t_records)
    core_windows_excl = [w for w in core_windows if w["window_id"] not in files_api_ids]
    full_windows_excl = [w for w in windows if w["window_id"] not in files_api_ids]

    scopes = {}
    for scope_name, scope_windows in (
        ("core30", core_windows), ("full38", windows),
        ("core30_excl_files_api", core_windows_excl), ("full38_excl_files_api", full_windows_excl),
    ):
        scopes[scope_name] = _score_scope(scope_windows, arm_s_records, arm_t_records, mapping)

    # --- Bezos condition 2: headline verdict on core30 ALONE ---
    core = scopes["core30"]
    own_verdict, own_verdict_reasons = _own_verdict_from_scope(core)
    required_margin = required_margin_for_n(core["n_windows_in_scope"])

    # --- §G.1 point 4 dual reporting (Bezos precondition 2, 2026-07-26):
    # headline recomputed with ingestion_path=="files_api" windows excluded,
    # PLUS an explicit verdict-sensitivity statement. Gate 1's only §G.1
    # reroute is `w37`, which is a gate5_straddler (excluded from core30 by
    # construction already) — core30 therefore contains ZERO files_api
    # windows, so this exclusion is a STRUCTURAL NO-OP for Gate 1's own
    # headline (reported anyway, for completeness/contract-compliance, with
    # that fact stated explicitly rather than silently producing an
    # identical-looking number). The exclusion DOES have real effect on the
    # full38 secondary figure (which loses w37).
    core_excl = scopes["core30_excl_files_api"]
    own_verdict_excl, own_verdict_excl_reasons = _own_verdict_from_scope(core_excl)
    verdict_flips_on_exclusion = own_verdict != own_verdict_excl
    n_files_api_in_core = core["n_windows_in_scope"] - core_excl["n_windows_in_scope"]
    if n_files_api_in_core == 0:
        verdict_sensitivity_note = (
            "NO-OP: the core-30 headline contains ZERO files_api-ingested windows (Gate 1's only "
            "§G.1 reroute, w37, is a gate5_straddler — already outside the core-30 by construction). "
            f"own_verdict is therefore identical with/without exclusion: {own_verdict!r}. Verdict does "
            "NOT flip on exclusion (trivially — there was nothing to exclude)."
        )
    else:
        verdict_sensitivity_note = (
            f"{n_files_api_in_core} files_api window(s) excluded from core-30 -> "
            f"own_verdict {own_verdict!r} (full) vs {own_verdict_excl!r} (excl. files_api). "
            + ("VERDICT FLIPS ON EXCLUSION — reportable finding, not a footnote."
               if verdict_flips_on_exclusion else "Verdict does NOT flip on exclusion.")
        )

    # --- Bezos condition 3: Gate 1 x Gate 2 coupling ---
    gate2_scoring_path = gc.versioned_path(gc.GATE2_SCORING_PATH) if rerun else gc.GATE2_SCORING_PATH
    coupled_verdict, coupling_note = apply_gate2_coupling(own_verdict, gate2_scoring_path)

    # --- Gate 5 addendum data (straddler vs non-straddler, Arm T hit rate) — Bezos condition 1 reframe applied in the .md writer ---
    gold_t_straddlers, pred_t_straddlers, excl_t_straddlers = _build_arm_buckets(straddler_windows, arm_t_records)
    outcomes_t_straddlers, _ = _score_track_b(gold_t_straddlers, pred_t_straddlers, mapping) if gold_t_straddlers else ({}, None)
    hit_t_straddlers = _window_hits(outcomes_t_straddlers)
    n_straddlers_scored = len(hit_t_straddlers)
    straddler_hit_rate = (sum(hit_t_straddlers.values()) / n_straddlers_scored) if n_straddlers_scored else None

    gold_t_core, pred_t_core, _ = _build_arm_buckets(core_windows, arm_t_records)
    outcomes_t_core, _ = _score_track_b(gold_t_core, pred_t_core, mapping) if gold_t_core else ({}, None)
    hit_t_core = _window_hits(outcomes_t_core)
    n_core_scored = len(hit_t_core)
    core_hit_rate = (sum(hit_t_core.values()) / n_core_scored) if n_core_scored else None

    gate5_path = "primary" if n_straddlers_scored >= GATE5_FALLBACK_MIN_SURVIVORS else "fallback_count_only"
    gate5_gap_pp = None
    gate5_flip = None
    if gate5_path == "primary" and straddler_hit_rate is not None and core_hit_rate is not None:
        gate5_gap_pp = (core_hit_rate - straddler_hit_rate) * 100.0
        gate5_flip = gate5_gap_pp > GATE5_FLIP_THRESHOLD_PP

    result = {
        "gate": "gate1", "label": label, "rerun": rerun,
        "settings": {"model_id": gc.MODEL_ID, "thinking": "high" if rerun else None} if rerun else None,
        "design_margin_reference": {"hit_delta": DESIGN_MARGIN_HIT_DELTA, "n": DESIGN_MARGIN_N},
        "required_margin_core30": required_margin,
        "scopes": scopes,
        "own_verdict": own_verdict, "own_verdict_reasons": own_verdict_reasons,
        "verdict": coupled_verdict, "coupling_note": coupling_note,
        "files_api_dual_reporting": {  # §G.1 point 4, Bezos precondition 2 (2026-07-26)
            "files_api_window_ids": sorted(files_api_ids),
            "n_files_api_in_core30": n_files_api_in_core,
            "own_verdict_excl_files_api": own_verdict_excl,
            "own_verdict_excl_files_api_reasons": own_verdict_excl_reasons,
            "verdict_flips_on_exclusion": verdict_flips_on_exclusion,
            "verdict_sensitivity_note": verdict_sensitivity_note,
        },
        "gate5_addendum_data": {
            "n_straddlers_total": len(straddler_windows), "n_straddlers_scored": n_straddlers_scored,
            "excluded_straddlers": excl_t_straddlers, "straddler_arm_t_hit_rate": straddler_hit_rate,
            "n_core_scored": n_core_scored, "core_arm_t_hit_rate": core_hit_rate,
            "gap_pp_core_minus_straddler": gate5_gap_pp, "flip_threshold_pp": GATE5_FLIP_THRESHOLD_PP,
            "path": gate5_path, "flip_to_no_go": gate5_flip,
        },
    }
    return result


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _fmt_p(x: Optional[float]) -> str:
    return "n/a (n=0 effective pairs)" if x is None else f"p={x:.4f}"


def _fmt_excluded(ids: list[str], max_show: int = 5) -> str:
    if len(ids) <= max_show:
        return str(ids)
    return f"{ids[:max_show]} ... (+{len(ids) - max_show} more)"


def write_report(result: dict, out_path: Path) -> None:
    core = result["scopes"]["core30"]
    full = result["scopes"]["full38"]
    lines = []
    lines.append("# Gate 1 Report — position+technique+validator (Arm T) vs simplified-tags (Arm S)")
    lines.append("")
    if result["label"]:
        lines.append(f"**RUN LABEL: `{result['label']}`** — see N-scored figures below before reading any "
                      "verdict as authoritative. A label other than empty means this is NOT the full "
                      "38-window batch (e.g. a smoke-test partial run) and the verdict below is "
                      "illustrative of the scoring PIPELINE only, not a real gate decision.")
        lines.append("")
    lines.append("## Settings tuple (design §D)")
    lines.append("")
    lines.append("| | fps | media_resolution | model | thinking | tau_t | max_validator_iterations |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| Arm S | {gc.FPS} | None (LOW) | {gc.MODEL_ID} | off | {gc.TAU_T}s | n/a |")
    lines.append(f"| Arm T position | {gc.FPS} | None (LOW) | {gc.MODEL_ID} | off | {gc.TAU_T}s | n/a |")
    lines.append(f"| Arm T technique | {gc.FPS} | None (LOW) | {gc.MODEL_ID} | off | {gc.TAU_T}s | n/a |")
    lines.append(f"| Arm T validator | {gc.FPS} | high (prod default) | {gc.MODEL_ID} | off | {gc.TAU_T}s | 1 |")
    lines.append("")
    lines.append("## Subset composition (design §A.2)")
    lines.append("")
    lines.append("30 stratified core + 8 forced Gate-5 straddlers = 38 windows. Quotas: "
                  f"`{gc.GATE1_QUOTAS}`.")
    lines.append("")
    lines.append("## HEADLINE verdict — 30-event stratified CORE only (Bezos condition 2)")
    lines.append("")
    st = core["sign_test"]
    lines.append(
        f"hit_S={core['hit_s']}, hit_T={core['hit_t']}, hit_T-hit_S={core['hit_delta']} "
        f"(required >= {result['required_margin_core30']}, re-derived from design's 4/38 rule: "
        f"ceil(4/38*{core['n_windows_in_scope']})={result['required_margin_core30']}), "
        f"fp_dup_rate_S={_fmt_pct(core['fp_dup_rate_s'])}, fp_dup_rate_T={_fmt_pct(core['fp_dup_rate_t'])} "
        f"(ceiling +{DESIGN_FP_DUP_CEILING*100:.0f}pp) -> "
        f"**verdict: {result['own_verdict']}** ({', '.join(result['own_verdict_reasons']) or 'both criteria satisfied'}), "
        f"**sign-test: {_fmt_p(st['p_value'])}** (n_effective={st['n_effective']}, "
        f"T-wins={st['n_plus_t_wins']}, S-wins={st['n_minus_s_wins']}, ties_excluded={st['n_ties_excluded']})"
    )
    lines.append("")
    lines.append(f"**COUPLED VERDICT (Bezos condition 3): `{result['verdict']}`** — {result['coupling_note']}")
    lines.append("")
    lines.append(f"N scored: Arm S {core['n_scored_arm_s']}/{core['n_windows_in_scope']}, "
                  f"Arm T {core['n_scored_arm_t']}/{core['n_windows_in_scope']} "
                  f"(excluded call_failed/unrun — Arm S: {_fmt_excluded(core['excluded_arm_s_call_failed_or_unrun'])}, "
                  f"Arm T: {_fmt_excluded(core['excluded_arm_t_call_failed_or_unrun'])}; full list in "
                  "gate1_scoring.json).")
    lines.append("")
    lines.append("## §G.1 point 4 dual reporting — headline recomputed excluding files_api-ingested windows")
    lines.append("")
    dr = result["files_api_dual_reporting"]
    core_excl = result["scopes"]["core30_excl_files_api"]
    lines.append(f"files_api-ingested window(s) in this run: `{dr['files_api_window_ids']}`.")
    lines.append("")
    lines.append(
        f"core-30 EXCLUDING files_api windows (n={core_excl['n_windows_in_scope']}): "
        f"hit_S={core_excl['hit_s']}, hit_T={core_excl['hit_t']}, hit_T-hit_S={core_excl['hit_delta']} -> "
        f"**verdict: {dr['own_verdict_excl_files_api']}** "
        f"({', '.join(dr['own_verdict_excl_files_api_reasons']) or 'both criteria satisfied'})"
    )
    lines.append("")
    lines.append(f"**VERDICT SENSITIVITY: {dr['verdict_sensitivity_note']}**")
    lines.append("")
    lines.append("## SECONDARY figure — full 38-window pool (core + 8 straddlers, informational only)")
    lines.append("")
    lines.append(
        f"hit_S={full['hit_s']}, hit_T={full['hit_t']}, hit_T-hit_S={full['hit_delta']}, "
        f"fp_dup_rate_S={_fmt_pct(full['fp_dup_rate_s'])}, fp_dup_rate_T={_fmt_pct(full['fp_dup_rate_t'])}. "
        "NOT the pass bar — the 8 straddlers are a deliberately class-skewed, non-representative addition."
    )
    full_excl = result["scopes"]["full38_excl_files_api"]
    lines.append(
        f"full-38 EXCLUDING files_api windows (n={full_excl['n_windows_in_scope']}, drops `w37`): "
        f"hit_S={full_excl['hit_s']}, hit_T={full_excl['hit_t']}, hit_T-hit_S={full_excl['hit_delta']} "
        "(§G.1 point 4 — this is where files_api exclusion has real effect, unlike the core-30 headline "
        "above, since w37 is a full-38-only straddler)."
    )
    lines.append("")
    for scope_name, scope in (("Core-30", core), ("Full-38", full)):
        for arm_name, key in (("Arm S", "summary_arm_s"), ("Arm T", "summary_arm_t")):
            summ = scope[key]
            lines.append(f"### {scope_name} — {arm_name} — metrics.summarize")
            lines.append("")
            if summ is None:
                lines.append("_no windows scored for this arm in this scope._")
            else:
                lines.append(f"- micro P/R/F1: {summ['micro']}")
                lines.append(f"- totals: {summ['totals']}")
                lines.append(f"- fp_breakdown: {summ['fp_breakdown']}")
                lines.append(f"- fn_breakdown: {summ['fn_breakdown']}")
            lines.append("")
        if scope["baselines"]:
            lines.append(f"### {scope_name} — null baselines (highlight_baselines.py, same window pool)")
            lines.append("")
            for bname, bsumm in scope["baselines"].items():
                lines.append(f"- **{bname}**: micro P/R/F1: {bsumm['micro']}, totals: {bsumm['totals']}")
            lines.append("")
    lines.append("## 63% baseline (footnote, continuity only)")
    lines.append("")
    lines.append(
        "The historically-cited 63% `action_class` @ fps=10 prose number (INS-096/097/102/103 lineage) has "
        "no backing artifact and cannot be recomputed — it is reported here for continuity ONLY, and is "
        "NOT the pass bar for this gate (design §B.1's explicit ruling; Arm S's re-run above IS the "
        "official baseline)."
    )
    lines.append("")
    out_path.write_text("\n".join(lines))


def write_gate5_addendum(result: dict, out_path: Path) -> None:
    d = result["gate5_addendum_data"]
    lines = []
    lines.append("# Gate 5 Addendum — straddler vs non-straddler Arm T hit-rate comparison")
    lines.append("")
    lines.append(
        "**What this measures (Bezos reframe, condition 1, 2026-07-25 pass-with-conditions review):** "
        "this addendum compares Arm T's hit rate on the 8 W=60 flagged \"straddler\" events "
        "(`GATE5_REPORT.md`'s production-anchoring table) against the 30 non-straddler core events, "
        "**under this harness's own window construction — full, UNTRUNCATED "
        "`[t_s-5s, t_s+4s]` native-video context for every window, identical for straddlers and "
        "non-straddlers alike.**"
    )
    lines.append("")
    lines.append(
        "**This is explicitly NOT a measurement of whether production's real 720s chunk-boundary "
        "truncation causes accuracy loss on these events.** Production's `highlight_analyze_node` "
        "clamps each highlight's analyze window to the run's own outer chunk scope "
        "(`service/pipelines/executors.py:1286`: `window_start = max(ctx.start_sec, authoritative_start - "
        "cfg.preroll_s)`; `:1290`: `window_end = min(ctx.end_sec, authoritative_end + cfg.postroll_s)`) — "
        "a straddling event's setup evidence sitting in the PREVIOUS 720s chunk is genuinely invisible to "
        "the chunk that reports it, in production. This harness's own window clamp "
        "(`gate_common.compute_window`) only clamps to `[0, video_duration_s]`, never to a 720s chunk "
        "boundary — every window here gets its full 9s of native video regardless of where a real "
        "720s grid seam would fall. **Known gap:** if Gate 1's Arm T does WORSE on the 8 straddlers "
        "even under this harness's full context, that is a signal the straddling events are simply "
        "harder cases on their own merits (occlusion, longer setups, etc.) — not evidence about "
        "truncation loss specifically, which this addendum cannot measure without re-running the "
        "SAME 8 events through the real chunked production pipeline (a separate, not-yet-built probe)."
    )
    lines.append("")
    lines.append(
        f"straddler Arm T hit rate: {_fmt_pct(d['straddler_arm_t_hit_rate'])} "
        f"(n={d['n_straddlers_scored']}/{d['n_straddlers_total']} scored, excluded: {d['excluded_straddlers']})"
    )
    lines.append(f"non-straddler (core-30) Arm T hit rate: {_fmt_pct(d['core_arm_t_hit_rate'])} "
                  f"(n={d['n_core_scored']}/30 scored)")
    lines.append("")
    if d["path"] == "fallback_count_only":
        lines.append(
            f"**FALLBACK PATH** — only {d['n_straddlers_scored']} of 8 straddlers survived call failures "
            f"(< {GATE5_FALLBACK_MIN_SURVIVORS} minimum for a statistically meaningful 2-event-swing "
            "comparison, per design §E.3). Gate 5 closes on COUNT ALONE: 1.4% (8/557) at the "
            "decision-relevant W=60 is small enough to PASS provisionally. **Mandatory follow-up ticket "
            "required** (per design §E.3): once the full 557-event corpus runs through the real production "
            "path, pull the 8 straddling events + an equal-size random non-straddling sample and compare "
            "recall — this is a monitoring item, not a pre-cutover blocker, but MUST be filed, not "
            "silently dropped."
        )
    else:
        gap = d["gap_pp_core_minus_straddler"]
        verdict = "FLIP TO NO-GO" if d["flip_to_no_go"] else "no flip"
        lines.append(
            f"gap (core - straddler) = {gap:.1f}pp (flip threshold: >{d['flip_threshold_pp']}pp) -> **{verdict}**"
        )
        if d["flip_to_no_go"]:
            lines.append("")
            lines.append(
                "Per design §E.3: the zero-overlap 720s chunking must be revisited (add cross-chunk "
                "overlap + dedup) before cutover — but see the reframe above: this harness measured "
                "straddler-vs-core difficulty under FULL context, not the real truncation effect, so a "
                "flip here is a signal to investigate straddler difficulty, and the truncation-specific "
                "claim in the original design text should be validated against a truncated-window probe "
                "before acting on it as a truncation finding specifically."
            )
    lines.append("")
    out_path.write_text("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="", help="Non-empty for a partial/smoke run — suffixes output filenames "
                                                      "(e.g. --label smoke -> gate1_scoring.smoke.json) so canonical "
                                                      "gate1_scoring.json/GATE1_REPORT.md are reserved for a real "
                                                      "full-batch run.")
    parser.add_argument("--rerun", action="store_true",
                        help="[RE-RUN 2026-07-26] Score the qualified-model (gemini-3.5-flash, thinking=high) "
                             "re-run: reads gc.versioned_path(...) predictions instead of the flash-lite v1 "
                             "canonicals, and writes to gc.RERUN_LABEL-suffixed output files (never overwrites "
                             "v1). Mutually orthogonal to --label (which still governs partial/smoke labeling "
                             "within a rerun, if ever needed).")
    args = parser.parse_args(argv)

    result = score_gate1(label=args.label, rerun=args.rerun)
    suffix = f".{args.label}" if args.label else (f".{gc.RERUN_LABEL}" if args.rerun else "")
    scoring_path = gc.S12_GATES_DIR / f"gate1_scoring{suffix}.json"
    report_path = gc.S12_GATES_DIR / f"GATE1_REPORT{suffix}.md"
    addendum_path = gc.S12_GATES_DIR / f"GATE5_ADDENDUM{suffix}.md"

    gc.write_json(scoring_path, result)
    write_report(result, report_path)
    write_gate5_addendum(result, addendum_path)

    print(f"wrote {scoring_path}", file=sys.stderr)
    print(f"wrote {report_path}", file=sys.stderr)
    print(f"wrote {addendum_path}", file=sys.stderr)
    print(f"own_verdict={result['own_verdict']} coupled_verdict={result['verdict']}", file=sys.stderr)


if __name__ == "__main__":
    main()
