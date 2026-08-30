"""Per-clip scoring vs gold + go/no-go report for SAM3 scout arms."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from eval.adcc.score.matcher import match_one

DEFAULT_TAU_T = 2.0


def gold_event_for_clip(clip: dict) -> dict:
    """Build a matcher-compatible gold event for one scout clip."""
    return {
        "t_s": float(clip["t_s"]),
        "action_class": clip["action_class"],
        "outcome": clip.get("outcome"),
        "match_id": clip["match_id"],
        "compat": frozenset({clip["action_class"]}),
        "clip_id": clip["clip_id"],
        "occlusion_heavy": bool(clip.get("occlusion_heavy", False)),
        "stratum": clip.get("stratum"),
    }


def score_arm_events(
    gold: dict,
    pred_events: list[dict],
    tau_t: float = DEFAULT_TAU_T,
) -> dict[str, Any]:
    outcome = match_one([gold], pred_events, tau_t)
    detected = bool(outcome.tp)
    pred = None
    action_ok = False
    outcome_ok = False
    if outcome.tp:
        _, pi = outcome.tp[0]
        pred = pred_events[pi]
        action_ok = pred.get("action_class") == gold["action_class"]
        outcome_ok = pred.get("outcome") == gold.get("outcome")
    return {
        "detected": detected,
        "action_class_ok": action_ok,
        "outcome_ok": outcome_ok,
        "tp": outcome.tp,
        "fn_silent": outcome.fn_silent,
        "fn_misclassified": outcome.fn_misclassified,
        "fp_hallucinated": outcome.fp_hallucinated,
        "pred": pred,
    }


def score_clip_arms(
    clip: dict,
    arms_payload: dict,
    tau_t: float = DEFAULT_TAU_T,
) -> dict[str, Any]:
    gold = gold_event_for_clip(clip)
    per_arm = {}
    for arm_name, arm_body in arms_payload.get("arms", {}).items():
        per_arm[arm_name] = score_arm_events(gold, arm_body.get("events") or [], tau_t=tau_t)
    return {
        "clip_id": clip["clip_id"],
        "match_id": clip["match_id"],
        "stratum": clip.get("stratum"),
        "occlusion_heavy": bool(clip.get("occlusion_heavy", False)),
        "gold": {k: (list(v) if isinstance(v, frozenset) else v) for k, v in gold.items()},
        "arms": per_arm,
    }


def _arm_score_tuple(arm_result: dict) -> tuple[int, int, int]:
    """Lexicographic score: detection, action_class, outcome."""
    return (
        int(bool(arm_result.get("detected"))),
        int(bool(arm_result.get("action_class_ok"))),
        int(bool(arm_result.get("outcome_ok"))),
    )


def decide_go_no_go(per_clip_scores: list[dict]) -> dict[str, Any]:
    """Apply founder go/no-go rules (small-n, per-clip, not aggregate %).

    No-go: on hard/occlusion subset, both B and C ≤ A, OR caller flags
    unusable eyeball (passed separately via ``eyeball_unusable``).
    """
    hard = [c for c in per_clip_scores if c.get("occlusion_heavy") or c.get("stratum") == "submission"]
    b_beats_a = 0
    c_beats_a = 0
    b_le_a_hard = 0
    c_le_a_hard = 0
    for c in per_clip_scores:
        a = _arm_score_tuple(c["arms"].get("A", {}))
        b = _arm_score_tuple(c["arms"].get("B", {}))
        c_sc = _arm_score_tuple(c["arms"].get("C", {}))
        if b > a:
            b_beats_a += 1
        if c_sc > a:
            c_beats_a += 1
        if c.get("occlusion_heavy") or c.get("stratum") == "submission":
            if b <= a:
                b_le_a_hard += 1
            if c_sc <= a:
                c_le_a_hard += 1

    n_hard = max(1, len(hard))
    both_le_on_hard = (b_le_a_hard == len(hard) and c_le_a_hard == len(hard) and len(hard) > 0)
    verdict = "investigate"
    reason = "partial lift or mixed arms — investigate further"
    if both_le_on_hard and b_beats_a == 0 and c_beats_a == 0:
        verdict = "no-go"
        reason = "Arms B and C both ≤ Arm A on hard subset and no lift elsewhere"
    elif b_beats_a > 0 or c_beats_a > 0:
        # Justified if any clear beat on hard subset
        hard_lift = False
        for c in hard:
            a = _arm_score_tuple(c["arms"].get("A", {}))
            b = _arm_score_tuple(c["arms"].get("B", {}))
            c_sc = _arm_score_tuple(c["arms"].get("C", {}))
            if b > a or c_sc > a:
                hard_lift = True
                break
        if hard_lift:
            verdict = "justified"
            reason = "Arm B and/or C beat A on occlusion-heavy/hard cases (eyeball still required)"
        else:
            verdict = "investigate"
            reason = "lift only on non-hard subset"

    return {
        "verdict": verdict,
        "reason": reason,
        "n_clips": len(per_clip_scores),
        "n_hard": len(hard),
        "b_beats_a": b_beats_a,
        "c_beats_a": c_beats_a,
        "both_le_on_hard": both_le_on_hard,
        "note": "Eyeball unusable athlete separation forces no-go regardless of scores.",
    }


def write_report(
    per_clip_scores: list[dict],
    out_path: str | Path,
    eyeball_unusable: bool = False,
) -> dict[str, Any]:
    decision = decide_go_no_go(per_clip_scores)
    if eyeball_unusable:
        decision["verdict"] = "no-go"
        decision["reason"] = "founder eyeball: unusable athlete separation"
    report = {"clips": per_clip_scores, "decision": decision}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
