"""Kimi-native (direct Moonshot) full batch driver — `GATES_RERUN_DESIGN.md`
§9.2/§9.5. Runs, in order, within one invocation:

1. **Pilot** (§9.5): first 3 PRIMARY-arm windows, real cost + parsing
   verified before continuing. Auto-continues to the full batch unless the
   pilot itself shows an anomaly (any call_failed, or per-call cost wildly
   outside the expected range) — per LeCun's explicit pre-authorization
   ("gated only on the pilot").
2. **PRIMARY arm** (§9.2): Kimi-native single-call, mirroring Arm S exactly
   — `simplified_tags.build_video_prompt`/`build_video_response_schema`'s
   PROMPT TEXT sent verbatim (no formal schema object — Moonshot has no
   Gemini-`types.Schema` equivalent; the prompt is fully self-describing,
   same cross-provider discipline already used for every other lane). Runs
   on the core-30 MINUS `w17` (pre-excluded — see `kimi_native_client.py`'s
   module docstring for the §9.1 probe finding that structurally rules it
   out for every mechanism tried). n<=29, HALT >3 (Gate 1's own precedent,
   §9.3).
3. **SECONDARY arm** (§9.2): Kimi-native triple (position/technique/
   validator), reusing `run_gate1._reconcile_validator` unmodified. Reuses
   the SAME per-window downloaded+normalized clip across all 3 calls (one
   download per window, not three) — same "one upload per window" discipline
   `RerouteUploadCache` already established. n=12 (first-12-by-window_id,
   `w17` not in this slice — no exclusion needed here), HALT >1 (§9.3).
4. **Sub-lane C** (§9.2): pure re-aggregation, ZERO new calls — Kimi-native
   triple vs already-captured Kimi-frames triple (`frontier_kimi_
   predictions.json`) on whatever windows both actually scored.

Budget: real per-call cost tracked from `kimi_native_client`'s verified rate
card; hard cap **$20** (checked BEFORE each window, same discipline as
`run_frontier_batch.py`) — if a lane would project past the cap, it HALTS,
already-collected windows are scored, and the halt is logged explicitly.

Dry-run by default; `--execute` required to spend real money.

Run from the repo root:

    .venv/bin/python -m eval.adcc.s12_gates.run_kimi_native_batch                # dry-run plan
    .venv/bin/python -m eval.adcc.s12_gates.run_kimi_native_batch --execute      # pilot + full batch
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from typing import Optional

from eval.adcc.data_prep.apply_gracie_mapping import load_mapping
from eval.adcc.s12_gates import gate_common as gc
from eval.adcc.s12_gates import kimi_native_client as knc
from eval.adcc.s12_gates import score_gate1 as sg1
from eval.adcc.s12_gates.run_gate1 import _reconcile_validator
from service.pipelines import highlight_axes, simplified_tags

CAP_USD = 20.0
PRIMARY_HALT_THRESHOLD = 3   # §9.3, matches Gate 1's own >3-of-38 precedent exactly
SECONDARY_HALT_THRESHOLD = 1  # §9.3, ~10% of 12
SECONDARY_N = 12
PILOT_N = 3
W17_WINDOW_ID = "650d20c745120c8433e9d059__w17"
FIRST_WINDOW_PROJECTED_COST_USD = 0.30  # conservative pre-data estimate (pilot showed ~$0.10/call)

PRIMARY_PREDICTIONS_PATH = gc.S12_GATES_DIR / "kimi_native_primary_predictions.json"
SECONDARY_PREDICTIONS_PATH = gc.S12_GATES_DIR / "kimi_native_secondary_predictions.json"
KIMI_NATIVE_SCORING_PATH = gc.S12_GATES_DIR / "kimi_native_scoring.json"
KIMI_NATIVE_REPORT_PATH = gc.S12_GATES_DIR / "KIMI_NATIVE_REPORT.md"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _core30_windows_excluding_w17() -> list[dict]:
    manifest = gc.read_json(gc.GATE1_MANIFEST_PATH)
    core = [w for w in manifest["windows"] if not w["gate5_straddler"]]
    excluded = [w for w in core if w["window_id"] == W17_WINDOW_ID]
    kept = [w for w in core if w["window_id"] != W17_WINDOW_ID]
    if excluded:
        print(f"[kimi_native] PRE-EXCLUDED {W17_WINDOW_ID} from the primary-arm window set — "
              "§9.1 probe confirmed no working native-video mechanism can reach this video "
              "(external-URL fetch AND file-reference both fail; the only working mechanism "
              "requires local video bytes, which requires yt-dlp, permanently blocked for this "
              "SME-claimed video). See kimi_native_client.py module docstring.", file=sys.stderr)
    assert len(kept) == 29, f"expected 29 windows after w17 exclusion, got {len(kept)}"
    return kept


def _first12_windows() -> list[dict]:
    manifest = gc.read_json(gc.GATE1_MANIFEST_PATH)
    core = [w for w in manifest["windows"] if not w["gate5_straddler"]]
    first12 = core[:SECONDARY_N]
    assert not any(w["window_id"] == W17_WINDOW_ID for w in first12), "w17 unexpectedly in first-12 slice"
    return first12


def _ledger_entry(
    *, gate: str, window_id: str, match_id: str, arm: str, call_name: str,
    status: str, error: Optional[str], latency_ms: float, usage: Optional[dict],
    cost_usd: Optional[float], thinking_level: Optional[str],
) -> gc.LedgerEntry:
    return gc.LedgerEntry(
        gate=gate, window_id=window_id, match_id=match_id, arm=arm, call_name=call_name,
        model_id=knc.KIMI_K3_MODEL_ID, status=status, error=error, http_status=None,
        latency_ms=latency_ms, usage_metadata=usage, cost_usd=cost_usd, timestamp_utc=_now_utc(),
        thinking_level=thinking_level, provider_route=knc.PROVIDER_ROUTE,
    )


def _thinking_level_from_usage(usage: Optional[dict]) -> Optional[str]:
    if not usage:
        return None
    rt = usage.get("reasoning_tokens")
    return "high" if (rt or 0) > 0 else "requested_zero_reasoning_tokens"


async def _primary_arm_window(
    window: dict, *, api_key: str, ledger: gc.CallLedger,
) -> tuple[dict, float]:
    """Mirrors `run_gate1.run_arm_s_window`'s output CONTRACT exactly (same
    dict shape) so `score_gate1`'s scoring helpers work unmodified. Downloads
    +normalizes ONE local clip for this window (cache-aware — reused if
    already present), sends `simplified_tags.build_video_prompt(None)`
    VERBATIM (§9.2's explicit "verbatim, unmodified" requirement) with NO
    system instruction (byte-matches Arm S's own `system_instruction=None`).
    """
    window_id, match_id = window["window_id"], window["match_id"]
    window_start = window["window_start_s"]
    cost = 0.0
    try:
        clip_path, sec_start = await knc.ensure_local_clip(window)
        video_url, video_bytes = knc.video_data_url(clip_path)
        data, raw, usage, latency_ms = await knc.call_moonshot_chat(
            api_key=api_key, model=knc.KIMI_K3_MODEL_ID, system_instruction=None,
            prompt_text=simplified_tags.build_video_prompt(None), video_data_url=video_url,
        )
        cost = usage["cost_usd"]
        status_ledger, error = "ok", None
    except Exception as e:
        data, usage, latency_ms = None, None, 0.0
        status_ledger, error = "call_failed", str(e)

    ledger.record(_ledger_entry(
        gate="kimi_native", window_id=window_id, match_id=match_id, arm="primary", call_name="arm_s",
        status=status_ledger, error=error, latency_ms=latency_ms, usage=usage, cost_usd=(cost or None),
        thinking_level=_thinking_level_from_usage(usage),
    ))

    call_record = {
        "call_name": "arm_s", "status": status_ledger, "error": error, "parsed": data,
        "latency_ms": latency_ms, "cost_usd": cost, "model_id": knc.KIMI_K3_MODEL_ID,
    }
    emitted_events: list[dict] = []
    record_status = "call_failed" if error else "ok"
    if error is None:
        clips = data.get("clips", []) if isinstance(data, dict) else []
        for clip in clips:
            # clip-relative -> absolute: the downloaded clip's own 0:00 is `sec_start`
            # (window_start - 2s buffer, clamped), NOT window_start itself — same
            # conversion `files_api_reroute.ReroutedUpload` already established.
            emitted_events.append({
                "start_s": sec_start + clip["start_s"], "end_s": sec_start + clip["end_s"],
                "action_class": clip["action_class"], "confidence": clip.get("confidence"),
            })
    record = {
        "window_id": window_id, "match_id": match_id, "gate5_straddler": window["gate5_straddler"],
        "gold_event": window["gold_event"], "video_url_watch": window["video_url_watch"],
        "window_start_s": window_start, "window_end_s": window["window_end_s"],
        "settings": {
            "native_video_mechanism": "inline data:video/mp4;base64 URI in a video_url content block",
            "fps": "not controllable, provider default",  # §9.2 explicit disclosure requirement
            "thinking": knc.REASONING_EFFORT, "model": knc.KIMI_K3_MODEL_ID, "provider_route": knc.PROVIDER_ROUTE,
        },
        "calls": [call_record], "emitted_events": emitted_events, "status": record_status,
    }
    return record, cost


async def _secondary_arm_window(
    window: dict, *, api_key: str, ledger: gc.CallLedger,
) -> tuple[dict, float]:
    """Mirrors `run_gate1.run_arm_t_window`'s output contract. Downloads the
    clip ONCE, reuses it across all 3 axis calls (one download per window,
    not three — `RerouteUploadCache`'s "one upload per window" discipline)."""
    window_id, match_id = window["window_id"], window["match_id"]
    description = None
    calls: list[dict] = []
    cost = 0.0

    try:
        clip_path, _sec_start = await knc.ensure_local_clip(window)
        video_url, _video_bytes = knc.video_data_url(clip_path)
    except Exception as e:
        ledger.record(_ledger_entry(
            gate="kimi_native", window_id=window_id, match_id=match_id, arm="secondary", call_name="download",
            status="call_failed", error=str(e), latency_ms=0.0, usage=None, cost_usd=None, thinking_level=None,
        ))
        return _secondary_record(window, [{"call_name": "download", "status": "call_failed", "error": str(e)}],
                                  status="call_failed", final_action_class=None), cost

    async def _do_call(call_name: str, prompt_text: str) -> Optional[dict]:
        nonlocal cost
        try:
            data, raw, usage, latency_ms = await knc.call_moonshot_chat(
                api_key=api_key, model=knc.KIMI_K3_MODEL_ID, system_instruction=None,
                prompt_text=prompt_text, video_data_url=video_url,
            )
            cost += usage["cost_usd"]
            status, error = "ok", None
        except Exception as e:
            data, usage, latency_ms = None, None, 0.0
            status, error = "call_failed", str(e)
        calls.append({"call_name": call_name, "status": status, "error": error, "parsed": data,
                       "latency_ms": latency_ms, "cost_usd": usage["cost_usd"] if usage else 0.0,
                       "model_id": knc.KIMI_K3_MODEL_ID})
        ledger.record(_ledger_entry(
            gate="kimi_native", window_id=window_id, match_id=match_id, arm="secondary", call_name=call_name,
            status=status, error=error, latency_ms=latency_ms, usage=usage,
            cost_usd=(usage["cost_usd"] if usage else None), thinking_level=_thinking_level_from_usage(usage),
        ))
        return data

    # NOTE: the axis prompts (highlight_axes.*) assume ABSOLUTE match-clock
    # framing is irrelevant to them (they only reason about what's visible,
    # not timestamps) — verbatim reuse is safe without a clip-relative variant,
    # unlike the primary arm which needs `sec_start` conversion for start_s/end_s.
    pos_data = await _do_call("position", highlight_axes.build_position_prompt(description))
    if calls[-1]["status"] == "call_failed":
        return _secondary_record(window, calls, status="call_failed", final_action_class=None), cost

    tech_data = await _do_call("technique", highlight_axes.build_technique_prompt(description))
    if calls[-1]["status"] == "call_failed":
        return _secondary_record(window, calls, status="call_failed", final_action_class=None), cost

    position_label = pos_data.get("position") if isinstance(pos_data, dict) else None
    position_justification = pos_data.get("justification") if isinstance(pos_data, dict) else None
    technique_label = tech_data.get("action_class") if isinstance(tech_data, dict) else None
    outcome_label = tech_data.get("outcome") if isinstance(tech_data, dict) else None
    technique_justification = tech_data.get("justification") if isinstance(tech_data, dict) else None

    validator_prompt = highlight_axes.build_validator_prompt(
        description=description, position_label=position_label, position_justification=position_justification,
        technique_label=technique_label, outcome_label=outcome_label, technique_justification=technique_justification,
    )
    val_data = await _do_call("validator", validator_prompt)
    if calls[-1]["status"] == "call_failed":
        return _secondary_record(window, calls, status="call_failed", final_action_class=None), cost

    status, final_position, final_action_class, final_outcome, verdict, ditch_reason = _reconcile_validator(
        position_label, position_justification, technique_label, outcome_label, technique_justification, val_data,
    )
    return _secondary_record(
        window, calls, status=status, final_action_class=final_action_class, final_position=final_position,
        final_outcome=final_outcome, verdict=verdict, ditch_reason=ditch_reason,
    ), cost


def _secondary_record(
    window: dict, calls: list[dict], *, status: str, final_action_class: Optional[str],
    final_position: Optional[str] = None, final_outcome: Optional[str] = None,
    verdict: Optional[str] = None, ditch_reason: Optional[str] = None,
) -> dict:
    window_start, window_end = window["window_start_s"], window["window_end_s"]
    emitted_events: list[dict] = []
    if status == "analyzed":
        emitted_events = [{"start_s": window_start, "end_s": window_end,
                            "action_class": final_action_class, "confidence": None}]
    return {
        "window_id": window["window_id"], "match_id": window["match_id"],
        "gate5_straddler": window["gate5_straddler"], "gold_event": window["gold_event"],
        "video_url_watch": window["video_url_watch"], "window_start_s": window_start, "window_end_s": window_end,
        "settings": {
            "native_video_mechanism": "inline data:video/mp4;base64 URI in a video_url content block",
            "fps": "not controllable, provider default", "thinking": knc.REASONING_EFFORT,
            "model": knc.KIMI_K3_MODEL_ID, "provider_route": knc.PROVIDER_ROUTE,
        },
        "calls": calls, "final_position": final_position, "final_outcome": final_outcome, "verdict": verdict,
        "ditch_reason": ditch_reason, "emitted_events": emitted_events, "status": status,
    }


async def _run_arm(
    arm: str, windows: list[dict], *, api_key: str, ledger: gc.CallLedger,
    halt_threshold: int, cap_remaining_usd: float, out_path,
) -> tuple[list[dict], float, bool]:
    records: list[dict] = []
    running_cost = 0.0
    calls_made = 0
    halted = False
    window_fn = _primary_arm_window if arm == "primary" else _secondary_arm_window
    for i, w in enumerate(windows):
        projected = FIRST_WINDOW_PROJECTED_COST_USD if calls_made == 0 else (
            (running_cost / calls_made) * (1 if arm == "primary" else 3)
        )
        if running_cost + projected > cap_remaining_usd:
            print(f"[kimi_native:{arm}] CAP CHECK: stopping BEFORE window {i} ({w['window_id']}) — "
                  f"running_cost=${running_cost:.4f} + projected=${projected:.4f} would exceed "
                  f"remaining_cap=${cap_remaining_usd:.2f}. {len(windows) - i} window(s) NOT attempted.",
                  file=sys.stderr)
            halted = True
            break
        print(f"[kimi_native:{arm}] window {i + 1}/{len(windows)} {w['window_id']} "
              f"({w['gold_event']['action_class']})...", file=sys.stderr)
        record, window_cost = await window_fn(w, api_key=api_key, ledger=ledger)
        records.append(record)
        running_cost += window_cost
        calls_made += len(record["calls"])
        print(f"  status={record['status']} running_cost=${running_cost:.4f}", file=sys.stderr)
        gc.write_json(out_path, records)
        ledger.flush()

        failed = sum(1 for r in records if r["status"] == "call_failed")
        if failed > halt_threshold:
            print(f"[kimi_native:{arm}] HALT-AND-ESCALATE — {failed} window(s) failed "
                  f"(threshold >{halt_threshold} of {len(windows)}). STOPPING — "
                  f"{len(windows) - (i + 1)} window(s) never attempted.", file=sys.stderr)
            halted = True
            break
    return records, running_cost, halted


def _print_dry_run_plan() -> None:
    primary = _core30_windows_excluding_w17()
    secondary = _first12_windows()
    print(f"KIMI-NATIVE DRY-RUN PLAN — pilot={PILOT_N} calls, "
          f"primary={len(primary)} windows x 1 call = {len(primary)} calls, "
          f"secondary={len(secondary)} windows x 3 calls = {len(secondary) * 3} calls, "
          f"total<= {len(primary) + len(secondary) * 3} calls, cap=${CAP_USD:.0f}.", file=sys.stderr)
    print("Zero real calls made. Pass --execute to spend real money.", file=sys.stderr)


async def _amain(args: argparse.Namespace) -> None:
    if not args.execute:
        _print_dry_run_plan()
        return

    api_key = knc._load_moonshot_key()
    ledger = gc.CallLedger()
    primary_windows = _core30_windows_excluding_w17()
    secondary_windows = _first12_windows()

    print(f"WARNING: --execute fires the Kimi-native pilot + full batch (up to "
          f"{PILOT_N} pilot + {len(primary_windows)} primary + {len(secondary_windows) * 3} secondary calls). "
          "Bezos/LeCun-gated (2026-07-26). Ctrl-C now if not explicitly authorized. Proceeding in 3s...",
          file=sys.stderr)
    await asyncio.sleep(3)

    # --- Pilot (§9.5) ---
    print(f"\n[kimi_native] === PILOT: {PILOT_N} primary-arm calls ===", file=sys.stderr)
    pilot_windows = primary_windows[:PILOT_N]
    pilot_records, pilot_cost, _halted = await _run_arm(
        "primary", pilot_windows, api_key=api_key, ledger=ledger,
        halt_threshold=PILOT_N,  # pilot itself never halts early — we want to SEE all N outcomes
        cap_remaining_usd=CAP_USD, out_path=PRIMARY_PREDICTIONS_PATH,
    )
    pilot_failed = sum(1 for r in pilot_records if r["status"] == "call_failed")
    pilot_avg_cost = pilot_cost / max(1, len(pilot_records))
    print(f"[kimi_native] PILOT RESULT: {len(pilot_records)}/{PILOT_N} attempted, "
          f"{pilot_failed} failed, ${pilot_cost:.4f} total (${pilot_avg_cost:.4f}/call avg).", file=sys.stderr)
    if pilot_failed > 0 or pilot_avg_cost > 0.50:
        print(f"[kimi_native] PILOT ANOMALY — {pilot_failed} failure(s) and/or per-call cost "
              f"${pilot_avg_cost:.4f} outside the expected ~$0.10/call range. STOPPING per §9.5's "
              "pilot-gate discipline — NOT proceeding to the full batch automatically.", file=sys.stderr)
        gc.write_json(PRIMARY_PREDICTIONS_PATH, pilot_records)
        return
    print("[kimi_native] Pilot clean — proceeding directly to the full batch (LeCun pre-authorized).",
          file=sys.stderr)

    # --- PRIMARY arm (remaining windows) ---
    remaining_primary = [w for w in primary_windows if w["window_id"] not in {r["window_id"] for r in pilot_records}]
    print(f"\n[kimi_native] === PRIMARY ARM: {len(remaining_primary)} remaining window(s) ===", file=sys.stderr)
    rest_records, rest_cost, primary_halted = await _run_arm(
        "primary", remaining_primary, api_key=api_key, ledger=ledger,
        halt_threshold=PRIMARY_HALT_THRESHOLD, cap_remaining_usd=CAP_USD - pilot_cost, out_path=PRIMARY_PREDICTIONS_PATH,
    )
    primary_records = pilot_records + rest_records
    gc.write_json(PRIMARY_PREDICTIONS_PATH, primary_records)
    primary_cost = pilot_cost + rest_cost
    print(f"[kimi_native] PRIMARY done: {len(primary_records)}/{len(primary_windows)} attempted, "
          f"${primary_cost:.4f}, halted={primary_halted}.", file=sys.stderr)

    # --- SECONDARY arm ---
    print(f"\n[kimi_native] === SECONDARY ARM: {len(secondary_windows)} window(s) ===", file=sys.stderr)
    secondary_records, secondary_cost, secondary_halted = await _run_arm(
        "secondary", secondary_windows, api_key=api_key, ledger=ledger,
        halt_threshold=SECONDARY_HALT_THRESHOLD, cap_remaining_usd=CAP_USD - primary_cost,
        out_path=SECONDARY_PREDICTIONS_PATH,
    )
    gc.write_json(SECONDARY_PREDICTIONS_PATH, secondary_records)
    print(f"[kimi_native] SECONDARY done: {len(secondary_records)}/{len(secondary_windows)} attempted, "
          f"${secondary_cost:.4f}, halted={secondary_halted}.", file=sys.stderr)

    total_cost = primary_cost + secondary_cost
    print(f"\n[kimi_native] TOTAL real measured spend this run: ${total_cost:.4f} (cap ${CAP_USD:.0f}).",
          file=sys.stderr)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="DANGER: spends real money. Default is dry-run.")
    args = parser.parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
