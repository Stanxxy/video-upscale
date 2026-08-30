"""Full frontier-lane batch driver — Bezos frontier-checkpoint authorization
(2026-07-26): "AUTHORIZE-WITH-PRECONDITIONS — route ruling (Opus=
direct_anthropic, Kimi=OpenRouter) CONFIRMED... RUN BOTH FULL BATCHES this
turn." Kimi K3 (via OpenRouter, frames-only — §6.2 probe result) and Opus
4.8 (via direct Anthropic, real usage-surfaced cost — precondition 1) each
run a position -> technique -> validator Arm-T-style triple per window, on
the SAME 30 core windows Gate 1's qualified `gemini-3.5-flash` Arm T ran on
(deterministic — `gate5_straddler == False` in the FROZEN Gate 1 manifest,
no new draw). Headline: each lane's hit rate vs Gemini's own Arm T on the
identical window subset — LeCun's falsifiable prediction, GATES_RERUN_
DESIGN.md §8.

Reuses (never reimplements):
- `run_gate1._reconcile_validator` — byte-faithful validator reconciliation
  switch (agree/disagree/ditch), model-agnostic (pure function over label
  strings, no analyzer/ctx dependency).
- `score_gate1._build_arm_buckets` / `_score_track_b` / `_window_hits` — the
  SAME clip-matching scorer (`matcher.match_all` at `TAU_T`) Gate 1 itself
  uses, so a frontier-lane "hit" means exactly what a Gate-1 "hit" means.
  Frontier records are built in `_arm_t_record`'s exact shape
  (`_frontier_record` below) specifically so these scorer functions work on
  them with zero modification.
- `openrouter_client.call_kimi_frames_axis` (Kimi lane, precondition 2's
  fixed framing/ordering already landed there) and
  `anthropic_direct.analyze_axis_json_with_usage` (Opus lane, precondition
  1 — real per-call `cost_usd` from `response.usage`, never estimated).

Per-lane hard cap ($40 Kimi, $40 Opus, GATES_RERUN_DESIGN.md §6.4) is
checked BEFORE each window's 3-call batch (not after) using a running
per-call-cost average (a conservative flat estimate before any real data
exists for that lane) — if the projection would push the lane over its cap,
the lane HALTS immediately; already-collected windows are still scored and
reported, and the halt is logged explicitly (never a silent truncation).

A separate, INFERRED failure-halt threshold (`FRONTIER_FAILURE_HALT_
THRESHOLD = 2` of 30, proportionally analogous to Gate 1's own explicit
">3 of 38" rule) applies per lane — GATES_RERUN_DESIGN.md §6 does not state
an explicit numeric threshold for this lane, so this is FLAGGED here as an
inferred conservative choice, the same treatment Gate 3/4's own inferred
thresholds got.

Dry-run by default; `--execute` required to spend real money.

Run from the repo root:

    .venv/bin/python -m eval.adcc.s12_gates.run_frontier_batch                       # dry-run plan
    .venv/bin/python -m eval.adcc.s12_gates.run_frontier_batch --execute             # FULL BATCH, both lanes
    .venv/bin/python -m eval.adcc.s12_gates.run_frontier_batch --execute --lane kimi # one lane only
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from typing import Optional

from eval.adcc.data_prep.apply_gracie_mapping import load_mapping
from eval.adcc.s12_gates import anthropic_direct as ad
from eval.adcc.s12_gates import gate_common as gc
from eval.adcc.s12_gates import openrouter_client as orc
from eval.adcc.s12_gates import score_gate1 as sg1
from eval.adcc.s12_gates.run_gate1 import _reconcile_validator
from service.pipelines import highlight_axes
from service.pipelines.executors import RunContext

LANE_CAP_USD = 40.0
CORE30_WINDOW_COUNT = 30
FRONTIER_FAILURE_HALT_THRESHOLD = 2  # INFERRED — see module docstring
FIRST_WINDOW_PROJECTED_COST_USD = 0.15  # conservative pre-data estimate (smoke pilot showed ~$0.04/call max)

FRONTIER_PREDICTIONS_PATH = {
    "kimi": gc.S12_GATES_DIR / "frontier_kimi_predictions.json",
    "opus": gc.S12_GATES_DIR / "frontier_opus_predictions.json",
}
FRONTIER_SCORING_PATH = gc.S12_GATES_DIR / "frontier_scoring.json"
FRONTIER_REPORT_PATH = gc.S12_GATES_DIR / "FRONTIER_LANE_REPORT.md"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _core30_windows() -> list[dict]:
    manifest = gc.read_json(gc.GATE1_MANIFEST_PATH)
    core = [w for w in manifest["windows"] if not w["gate5_straddler"]]
    assert len(core) == CORE30_WINDOW_COUNT, f"expected {CORE30_WINDOW_COUNT} core windows, got {len(core)}"
    return core


def _load_openrouter_key() -> str:
    env_path = gc.S12_GATES_DIR.parent.parent.parent / ".env"
    for line in open(env_path):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise RuntimeError(f"run_frontier_batch: OPENROUTER_API_KEY not found in {env_path}")


def _frontier_record(
    window: dict, calls: list[dict], *, status: str, final_action_class: Optional[str],
    final_position: Optional[str] = None, final_outcome: Optional[str] = None,
    verdict: Optional[str] = None, ditch_reason: Optional[str] = None,
    model_id: str = "", provider_route: str = "",
) -> dict:
    """Same shape as `run_gate1._arm_t_record` — deliberately, so
    `score_gate1`'s scoring helpers work on frontier records unmodified."""
    window_start, window_end = window["window_start_s"], window["window_end_s"]
    emitted_events: list[dict] = []
    if status == "analyzed":
        emitted_events = [{
            "start_s": window_start, "end_s": window_end,
            "action_class": final_action_class, "confidence": None,
        }]
    return {
        "window_id": window["window_id"], "match_id": window["match_id"],
        "gate5_straddler": window["gate5_straddler"], "gold_event": window["gold_event"],
        "video_url_watch": window["video_url_watch"],
        "window_start_s": window_start, "window_end_s": window_end,
        "settings": {"fps": gc.FPS, "model": model_id, "provider_route": provider_route, "thinking": "high"},
        "calls": calls,
        "final_position": final_position, "final_outcome": final_outcome, "verdict": verdict,
        "ditch_reason": ditch_reason,
        "emitted_events": emitted_events,
        "status": status,  # "analyzed" | "ditched" | "call_failed"
    }


async def _lane_axis_call(
    lane: str, *, window: dict, call_name: str, system_instruction: str, prompt_text: str,
    openrouter_key: Optional[str], anthropic_key: Optional[str],
) -> tuple[Optional[dict], Optional[str], dict]:
    """One real axis call for the given lane. Returns ``(data, error,
    ledger_kwargs)`` — ``ledger_kwargs`` always carries real
    ``cost_usd``/``model_id``/``provider_route``/``thinking_level``/
    ``latency_ms`` (0.0 default) regardless of success/failure, so the
    caller can build a `gc.LedgerEntry` uniformly."""
    window_start, window_end = window["window_start_s"], window["window_end_s"]
    provider_route = "openrouter" if lane == "kimi" else "direct_anthropic"
    model_id = orc.KIMI_K3_MODEL_ID if lane == "kimi" else orc.OPUS_48_ANTHROPIC_MODEL_ID
    try:
        if lane == "kimi":
            data, _raw, usage, latency_ms = await orc.call_kimi_frames_axis(
                api_key=openrouter_key, youtube_url=window["video_url_watch"],
                window_start=window_start, window_end=window_end, fps=gc.FPS,
                system_instruction=system_instruction, prompt_text=prompt_text, thinking="high",
            )
            cost_usd = usage.get("cost_usd")
            if cost_usd is None:
                raise RuntimeError(
                    f"run_frontier_batch: OpenRouter usage.cost missing for kimi/{call_name} on "
                    f"{window['window_id']} — cannot monitor real spend; failing loud rather than "
                    "silently treating the call as free."
                )
            thinking_level = "high" if (usage.get("reasoning_tokens") or 0) > 0 else "requested_zero_reasoning_tokens"
        else:
            ctx = RunContext(
                youtube_id="", youtube_url=window["video_url_watch"], start_sec=0, end_sec=0,
                gemini_api_key="", request_timeout_ms=600_000, anthropic_api_key=anthropic_key,
            )
            data, _raw, usage, latency_ms = await ad.analyze_axis_json_with_usage(
                ctx, model_id=model_id, window_start=window_start, window_end=window_end,
                prompt_text=prompt_text, system_instruction=system_instruction, fps=gc.FPS, thinking="high",
            )
            cost_usd = usage["cost_usd"]
            thinking_level = "high" if (usage.get("thinking_tokens") or 0) > 0 else "requested_zero_thinking_tokens"
        return data, None, {
            "cost_usd": cost_usd, "latency_ms": latency_ms, "model_id": model_id,
            "provider_route": provider_route, "thinking_level": thinking_level, "usage_metadata": usage,
        }
    except Exception as e:  # real transport/parse/ffmpeg/API failure — recorded, never swallowed
        return None, str(e), {
            "cost_usd": 0.0, "latency_ms": 0.0, "model_id": model_id,
            "provider_route": provider_route, "thinking_level": None, "usage_metadata": None,
        }


async def run_lane_window(
    *, lane: str, window: dict, ledger: gc.CallLedger,
    openrouter_key: Optional[str], anthropic_key: Optional[str],
) -> tuple[dict, float]:
    window_id, match_id = window["window_id"], window["match_id"]
    description = None
    calls: list[dict] = []
    window_cost = 0.0

    async def _do_call(call_name: str, system_instruction: str, prompt_text: str) -> Optional[dict]:
        nonlocal window_cost
        data, err, meta = await _lane_axis_call(
            lane, window=window, call_name=call_name, system_instruction=system_instruction,
            prompt_text=prompt_text, openrouter_key=openrouter_key, anthropic_key=anthropic_key,
        )
        window_cost += meta["cost_usd"]
        calls.append({
            "call_name": call_name, "status": "call_failed" if err else "ok", "error": err,
            "parsed": data, "latency_ms": meta["latency_ms"], "cost_usd": meta["cost_usd"],
            "model_id": meta["model_id"],
        })
        ledger.record(gc.LedgerEntry(
            gate="frontier_batch", window_id=window_id, match_id=match_id, arm=f"frontier_{lane}",
            call_name=call_name, model_id=meta["model_id"], status="call_failed" if err else "ok", error=err,
            http_status=None, latency_ms=meta["latency_ms"], usage_metadata=meta["usage_metadata"],
            cost_usd=meta["cost_usd"], timestamp_utc=_now_utc(), thinking_level=meta["thinking_level"],
            provider_route=meta["provider_route"],
        ))
        return data

    pos_data = await _do_call("position", highlight_axes.POSITION_SYSTEM_PROMPT, highlight_axes.build_position_prompt(description))
    if calls[-1]["status"] == "call_failed":
        return _frontier_record(window, calls, status="call_failed", final_action_class=None), window_cost

    tech_data = await _do_call("technique", highlight_axes.TECHNIQUE_SYSTEM_PROMPT, highlight_axes.build_technique_prompt(description))
    if calls[-1]["status"] == "call_failed":
        return _frontier_record(window, calls, status="call_failed", final_action_class=None), window_cost

    position_label = pos_data.get("position") if isinstance(pos_data, dict) else None
    position_justification = pos_data.get("justification") if isinstance(pos_data, dict) else None
    technique_label = tech_data.get("action_class") if isinstance(tech_data, dict) else None
    outcome_label = tech_data.get("outcome") if isinstance(tech_data, dict) else None
    technique_justification = tech_data.get("justification") if isinstance(tech_data, dict) else None

    validator_prompt = highlight_axes.build_validator_prompt(
        description=description, position_label=position_label, position_justification=position_justification,
        technique_label=technique_label, outcome_label=outcome_label, technique_justification=technique_justification,
    )
    val_data = await _do_call("validator", highlight_axes.VALIDATOR_SYSTEM_PROMPT, validator_prompt)
    if calls[-1]["status"] == "call_failed":
        return _frontier_record(window, calls, status="call_failed", final_action_class=None), window_cost

    status, final_position, final_action_class, final_outcome, verdict, ditch_reason = _reconcile_validator(
        position_label, position_justification, technique_label, outcome_label, technique_justification, val_data,
    )
    provider_route = "openrouter" if lane == "kimi" else "direct_anthropic"
    model_id = orc.KIMI_K3_MODEL_ID if lane == "kimi" else orc.OPUS_48_ANTHROPIC_MODEL_ID
    record = _frontier_record(
        window, calls, status=status, final_action_class=final_action_class, final_position=final_position,
        final_outcome=final_outcome, verdict=verdict, ditch_reason=ditch_reason,
        model_id=model_id, provider_route=provider_route,
    )
    return record, window_cost


async def run_lane(
    lane: str, windows: list[dict], ledger: gc.CallLedger,
    openrouter_key: Optional[str], anthropic_key: Optional[str], cap_usd: float = LANE_CAP_USD,
) -> tuple[list[dict], float, bool]:
    records: list[dict] = []
    running_cost = 0.0
    calls_made = 0
    halted = False
    for i, w in enumerate(windows):
        projected_window_cost = (
            FIRST_WINDOW_PROJECTED_COST_USD if calls_made == 0 else (running_cost / calls_made) * 3
        )
        if running_cost + projected_window_cost > cap_usd:
            print(
                f"[frontier:{lane}] CAP CHECK: stopping BEFORE window {i} ({w['window_id']}) — "
                f"running_cost=${running_cost:.4f} + projected_window_cost=${projected_window_cost:.4f} "
                f"would exceed cap=${cap_usd:.2f}. {len(windows) - i} window(s) NOT attempted "
                "(no silent truncation — logged here explicitly).", file=sys.stderr,
            )
            halted = True
            break
        print(f"[frontier:{lane}] window {i + 1}/{len(windows)} {w['window_id']} "
              f"({w['gold_event']['action_class']})...", file=sys.stderr)
        record, window_cost = await run_lane_window(
            lane=lane, window=w, ledger=ledger, openrouter_key=openrouter_key, anthropic_key=anthropic_key,
        )
        records.append(record)
        running_cost += window_cost
        calls_made += len(record["calls"])
        emitted = record["emitted_events"][0]["action_class"] if record["emitted_events"] else None
        print(f"  status={record['status']} verdict={record.get('verdict')} "
              f"final_action_class={emitted} running_cost=${running_cost:.4f}",
              file=sys.stderr)
        gc.write_json(FRONTIER_PREDICTIONS_PATH[lane], records)
        ledger.flush()

        failed = sum(1 for r in records if r["status"] == "call_failed")
        if failed > FRONTIER_FAILURE_HALT_THRESHOLD:
            print(
                f"[frontier:{lane}] HALT-AND-ESCALATE tripped after window {i} ({w['window_id']}) — "
                f"{failed} window(s) failed (INFERRED threshold: >{FRONTIER_FAILURE_HALT_THRESHOLD} of "
                f"{len(windows)}). STOPPING — {len(windows) - (i + 1)} window(s) never attempted.",
                file=sys.stderr,
            )
            halted = True
            break

    print(f"[frontier:{lane}] done. {len(records)}/{len(windows)} window(s) attempted, "
          f"{calls_made} call(s), ${running_cost:.4f} real measured spend, halted={halted}.", file=sys.stderr)
    return records, running_cost, halted


def score_lane_vs_gemini(lane_records: list[dict], core_windows: list[dict], mapping: dict) -> dict:
    """Restricts BOTH the lane's own scoring AND Gemini's Arm T baseline
    scoring to the EXACT SAME attempted-window subset (fair paired
    comparison even if a lane halted early with < 30 windows)."""
    attempted_ids = {r["window_id"] for r in lane_records}
    scoped_windows = [w for w in core_windows if w["window_id"] in attempted_ids]

    gemini_arm_t_path = gc.versioned_path(gc.GATE1_ARM_T_PATH)
    gemini_arm_t_records = gc.read_json(gemini_arm_t_path)

    gold_lane, pred_lane, excl_lane = sg1._build_arm_buckets(scoped_windows, lane_records)
    gold_gem, pred_gem, excl_gem = sg1._build_arm_buckets(scoped_windows, gemini_arm_t_records)
    _outcomes_lane, summary_lane = sg1._score_track_b(gold_lane, pred_lane, mapping) if gold_lane else ({}, None)
    _outcomes_gem, summary_gem = sg1._score_track_b(gold_gem, pred_gem, mapping) if gold_gem else ({}, None)
    hits_lane = sg1._window_hits(_outcomes_lane)
    hits_gem = sg1._window_hits(_outcomes_gem)
    hit_count_lane = sum(1 for v in hits_lane.values() if v)
    hit_count_gem = sum(1 for v in hits_gem.values() if v)

    return {
        "n_windows_attempted": len(lane_records), "n_scored_lane": len(gold_lane), "n_scored_gemini": len(gold_gem),
        "excluded_lane_call_failed_or_unrun": excl_lane, "excluded_gemini_call_failed_or_unrun": excl_gem,
        "hit_count_lane": hit_count_lane, "hit_count_gemini_same_windows": hit_count_gem,
        "hit_rate_lane": (hit_count_lane / len(gold_lane)) if gold_lane else None,
        "hit_rate_gemini_same_windows": (hit_count_gem / len(gold_gem)) if gold_gem else None,
        "per_window_hits_lane": hits_lane, "per_window_hits_gemini": hits_gem,
    }


def _print_dry_run_plan(windows: list[dict], lanes: list[str]) -> None:
    print(
        f"FRONTIER BATCH DRY-RUN PLAN — {len(windows)} core window(s) x 3 calls "
        f"(position/technique/validator) x {len(lanes)} lane(s) = "
        f"{len(windows) * 3 * len(lanes)} call(s) planned (subject to each lane's own ${LANE_CAP_USD:.0f} cap).",
        file=sys.stderr,
    )
    for lane in lanes:
        model_id = orc.KIMI_K3_MODEL_ID if lane == "kimi" else orc.OPUS_48_ANTHROPIC_MODEL_ID
        route = "openrouter" if lane == "kimi" else "direct_anthropic"
        print(f"  lane={lane} model={model_id} provider_route={route} cap=${LANE_CAP_USD:.0f}", file=sys.stderr)
    print("Zero real calls made. Pass --execute to spend real money.", file=sys.stderr)


async def _amain(args: argparse.Namespace) -> None:
    windows = _core30_windows()
    lanes = [args.lane] if args.lane else ["kimi", "opus"]

    if not args.execute:
        _print_dry_run_plan(windows, lanes)
        return

    cfg = gc.load_service_config()
    openrouter_key = _load_openrouter_key() if "kimi" in lanes or "opus" in lanes else None
    anthropic_key = cfg.anthropic_api_key
    if "opus" in lanes and not anthropic_key:
        raise RuntimeError("run_frontier_batch: ANTHROPIC_API_KEY/BJJ_ANTHROPIC_API_KEY not configured.")

    print(
        f"WARNING: --execute fires the FULL frontier batch (up to {len(windows) * 3} calls per lane, "
        f"lanes={lanes}). This is Bezos-gated (2026-07-26 AUTHORIZE-WITH-PRECONDITIONS). "
        "Ctrl-C now if this was not explicitly authorized. Proceeding in 3 real seconds...",
        file=sys.stderr,
    )
    await asyncio.sleep(3)

    ledger = gc.CallLedger()
    results: dict = {}
    for lane in lanes:
        existing = gc.read_json(FRONTIER_PREDICTIONS_PATH[lane]) if FRONTIER_PREDICTIONS_PATH[lane].exists() else []
        already_done = {r["window_id"] for r in existing if r["status"] != "call_failed"}
        remaining = [w for w in windows if w["window_id"] not in already_done]
        if len(remaining) < len(windows):
            print(f"[frontier:{lane}] resume: {len(windows) - len(remaining)} window(s) already done, "
                  f"{len(remaining)} remaining.", file=sys.stderr)
        records, cost, halted = await run_lane(lane, remaining, ledger, openrouter_key, anthropic_key)
        merged = {r["window_id"]: r for r in existing}
        for r in records:
            merged[r["window_id"]] = r
        merged_records = [merged[w["window_id"]] for w in windows if w["window_id"] in merged]
        gc.write_json(FRONTIER_PREDICTIONS_PATH[lane], merged_records)
        results[lane] = {"records": merged_records, "spend_usd_this_run": cost, "halted_this_run": halted}

    # [FIX] Score ALL lanes with a predictions file on disk, not just the lane(s)
    # run THIS invocation — otherwise a single-lane invocation (`--lane kimi` then,
    # separately, `--lane opus`) silently overwrites frontier_scoring.json with
    # only its own lane's data, discarding the other lane's already-collected
    # results from the combined artifact (found live when finalizing both lanes'
    # results after they were run sequentially in separate invocations).
    mapping = load_mapping()
    scoring: dict = {"generated_at_utc": _now_utc(), "lanes": {}}
    for lane in ("kimi", "opus"):
        if lane in results:
            lane_records = results[lane]["records"]
            spend_this_run = results[lane]["spend_usd_this_run"]
            halted_this_run = results[lane]["halted_this_run"]
        elif FRONTIER_PREDICTIONS_PATH[lane].exists():
            lane_records = gc.read_json(FRONTIER_PREDICTIONS_PATH[lane])
            spend_this_run = 0.0
            halted_this_run = None  # not run this invocation — unknown here, see ledger for that lane's own last run
        else:
            continue
        scoring["lanes"][lane] = score_lane_vs_gemini(lane_records, windows, mapping)
        scoring["lanes"][lane]["spend_usd_this_run"] = spend_this_run
        scoring["lanes"][lane]["halted_this_run"] = halted_this_run
    gc.write_json(FRONTIER_SCORING_PATH, scoring)

    print(f"\n[frontier] DONE. Scoring written to {FRONTIER_SCORING_PATH}.", file=sys.stderr)
    for lane, s in scoring["lanes"].items():
        print(f"  {lane}: hit_rate={s['hit_rate_lane']}, gemini_same_windows={s['hit_rate_gemini_same_windows']}, "
              f"n={s['n_scored_lane']}, spend_this_run=${s['spend_usd_this_run']:.4f}", file=sys.stderr)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="DANGER: spends real money. Default is dry-run.")
    parser.add_argument("--lane", choices=["kimi", "opus"], default=None, help="Restrict to one lane (default: both).")
    args = parser.parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
