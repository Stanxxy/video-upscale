"""Phase B smoke/probe driver — `GATES_RERUN_DESIGN.md` §6. Runs, in order:

1. Opus 4.8 provider-route sanity check (§6.4, mandatory before treating
   OpenRouter as Opus's default route): 2 calls, same window/prompt/schema,
   `thinking="high"` — one via direct Anthropic SDK
   (`service.pipelines.claude_frames.analyze_axis_json`), one via
   OpenRouter (`openrouter_client.call_opus_via_openrouter_axis`). Checks:
   (a) real thinking-token consumption on BOTH routes, (b) same enum choice
   on an unambiguous window.
2. Kimi K3 smoke (frames-only, confirmed via the §6.2 probe already run —
   see `openrouter_client.py`'s module docstring): 2 calls (position +
   technique) on ONE window.
3. Opus smoke (whichever route the step-1 check selects as default): 2
   calls (position + technique) on the SAME window.

Total this script: up to 6 real calls (2 + 2 + 2) — well within the "few
calls each, then STOP" instruction. NO full frontier batch is spent here —
that is explicitly held for a Bezos checkpoint after this report.

Run from the repo root:

    .venv/bin/python -m eval.adcc.s12_gates.smoke_frontier_lanes
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

from eval.adcc.s12_gates import gate_common as gc
from eval.adcc.s12_gates import openrouter_client as orc
from service.pipelines import claude_frames, highlight_axes


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

SMOKE_WINDOW_ID = "650d20c745120c8433e9d024__w0"  # Gate 1 core-30, window 0, gold=back_take.
# [CORRECTION 2026-07-26, Bezos precondition 3a] This comment previously called w0
# "well-understood" — that was factually wrong and is retracted. Cross-checked
# against gate1_predictions_arm_t.gemini35flash_highthink.json: the QUALIFIED
# gemini-3.5-flash model's own Arm T also got this window wrong on this turn's
# re-run (final_position="mount" vs gold "turtle"; validator verdict="agree" on
# the wrong label; final_action_class never resolved vs gold "back_take"). w0 is
# one of Gate 1's known-hard back_take-from-turtle cases, NOT a curated
# unambiguous clip — the opus_route_check's enum-disagreement (sweep vs
# guard_pass, neither matching gold) is consistent with genuine task difficulty,
# not evidence either route is "worse" at reading an easy case. See also: the
# route-check payloads THIS RUN predate the precondition-2 framing/ordering fix
# below, so the enum-disagreement is additionally confounded by that (now-fixed)
# construction difference — it does NOT stand as clean independent
# route-equivalence evidence. The reasoning_tokens=0 finding (OpenRouter did not
# confirm thinking was honored for Opus specifically) is what actually carries
# the routing conclusion in this smoke's report, not the enum mismatch.


def _get_window() -> dict:
    manifest = gc.read_json(gc.GATE1_MANIFEST_PATH)
    return next(w for w in manifest["windows"] if w["window_id"] == SMOKE_WINDOW_ID)


async def _direct_anthropic_technique_call(anthropic_api_key: str, window: dict) -> tuple[dict, dict, float]:
    """Direct-Anthropic-SDK technique-axis call via
    `claude_frames.analyze_axis_json` — builds a minimal `RunContext`
    (only the fields the Claude frame path actually reads: youtube_url,
    anthropic_api_key; stream_url/native_fps are lazily resolved inside
    `analyze_axis_json` itself)."""
    from service.pipelines.executors import RunContext

    ctx = RunContext(
        youtube_id="", youtube_url=window["video_url_watch"], start_sec=0, end_sec=0,
        gemini_api_key="", request_timeout_ms=600_000, anthropic_api_key=anthropic_api_key,
    )
    import time
    t0 = time.perf_counter()
    data, raw = await claude_frames.analyze_axis_json(
        ctx, model_id=orc.OPUS_48_ANTHROPIC_MODEL_ID,
        window_start=window["window_start_s"], window_end=window["window_end_s"],
        prompt_text=highlight_axes.build_technique_prompt(None),
        system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT,
        fps=gc.FPS, thinking="high",
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return data, {"raw": raw}, latency_ms


async def _amain() -> None:
    cfg = gc.load_service_config()
    if not cfg.anthropic_api_key:
        raise RuntimeError("smoke_frontier_lanes: ANTHROPIC_API_KEY/BJJ_ANTHROPIC_API_KEY not configured.")
    openrouter_key = None
    for line in open(gc.S12_GATES_DIR.parent.parent.parent / ".env"):
        if line.startswith("OPENROUTER_API_KEY="):
            openrouter_key = line.strip().split("=", 1)[1]
            break
    if not openrouter_key:
        raise RuntimeError("smoke_frontier_lanes: OPENROUTER_API_KEY not found in .env")

    window = _get_window()
    match_id = SMOKE_WINDOW_ID.split("__")[0]
    print(f"[smoke_frontier] target window: {SMOKE_WINDOW_ID} "
          f"(gold={window['gold_event']['action_class']}, window=[{window['window_start_s']}, {window['window_end_s']}])",
          file=sys.stderr)

    results: dict = {"window_id": SMOKE_WINDOW_ID, "gold_action_class": window["gold_event"]["action_class"]}
    ledger = gc.CallLedger()  # `provider_route` recorded on EVERY call below — CEO's explicit instruction

    # --- Step 1: Opus provider-route sanity check (§6.4) ---
    print("\n[smoke_frontier] === STEP 1: Opus 4.8 provider-route sanity check ===", file=sys.stderr)

    print("[smoke_frontier] call 1/2: Opus via DIRECT Anthropic SDK, thinking=high...", file=sys.stderr)
    direct_data, direct_meta, direct_latency_ms = await _direct_anthropic_technique_call(
        cfg.anthropic_api_key, window,
    )
    print(f"  direct-anthropic: action_class={direct_data.get('action_class')} "
          f"outcome={direct_data.get('outcome')} latency_ms={direct_latency_ms:.0f}", file=sys.stderr)
    ledger.record(gc.LedgerEntry(
        gate="phase_b_smoke", window_id=SMOKE_WINDOW_ID, match_id=match_id, arm="opus_route_check",
        call_name="technique", model_id=orc.OPUS_48_ANTHROPIC_MODEL_ID, status="ok", error=None,
        http_status=None, latency_ms=direct_latency_ms, usage_metadata=None, cost_usd=None,
        timestamp_utc=_now_utc(), thinking_level="high", provider_route="direct_anthropic",
    ))

    print("[smoke_frontier] call 2/2: Opus via OpenRouter, thinking=high...", file=sys.stderr)
    or_data, or_raw, or_usage, or_latency_ms = await orc.call_opus_via_openrouter_axis(
        api_key=openrouter_key, youtube_url=window["video_url_watch"],
        window_start=window["window_start_s"], window_end=window["window_end_s"], fps=gc.FPS,
        system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT,
        prompt_text=highlight_axes.build_technique_prompt(None), thinking="high",
    )
    print(f"  openrouter-opus: action_class={or_data.get('action_class')} outcome={or_data.get('outcome')} "
          f"latency_ms={or_latency_ms:.0f} usage={or_usage}", file=sys.stderr)
    or_thinking_confirmed_now = or_usage.get("reasoning_tokens") is not None and or_usage.get("reasoning_tokens", 0) > 0
    ledger.record(gc.LedgerEntry(
        gate="phase_b_smoke", window_id=SMOKE_WINDOW_ID, match_id=match_id, arm="opus_route_check",
        call_name="technique", model_id=orc.OPUS_48_OPENROUTER_MODEL_ID, status="ok", error=None,
        http_status=None, latency_ms=or_latency_ms, usage_metadata=or_usage, cost_usd=or_usage.get("cost_usd"),
        timestamp_utc=_now_utc(),
        thinking_level=("high" if or_thinking_confirmed_now else "requested_not_confirmed"),
        provider_route="openrouter",
    ))

    same_enum_choice = direct_data.get("action_class") == or_data.get("action_class")
    # Direct-Anthropic thinking-token confirmation: claude_frames doesn't currently
    # surface usage_metadata (Anthropic SDK response.usage has no bare "thinking
    # tokens" field the way OpenAI-style APIs do — Anthropic bills/reports thinking
    # content as part of the response content blocks, not a separate token count in
    # `usage`). We confirm thinking was ACTIVE indirectly: opus_thinking_kwargs sets
    # thinking.type=adaptive + effort=high in the actual request (verified by
    # inspecting the real kwargs below), which is the same mechanism
    # GATES_3_4_DESIGN.md §6.1 already established as Opus's correct "max thinking"
    # mapping — never fabricated, but also not independently re-measurable from the
    # SDK response alone at this call shape.
    direct_thinking_kwargs = claude_frames.opus_thinking_kwargs("high")
    direct_thinking_requested = bool(direct_thinking_kwargs.get("thinking"))
    openrouter_thinking_confirmed = or_usage.get("reasoning_tokens") is not None and or_usage.get("reasoning_tokens", 0) > 0

    print(f"\n[smoke_frontier] ROUTE CHECK RESULTS:", file=sys.stderr)
    print(f"  same_enum_choice (action_class): {same_enum_choice} "
          f"(direct={direct_data.get('action_class')!r}, openrouter={or_data.get('action_class')!r})", file=sys.stderr)
    print(f"  direct-anthropic thinking REQUESTED in payload: {direct_thinking_requested} "
          f"(kwargs={direct_thinking_kwargs})", file=sys.stderr)
    print(f"  openrouter thinking CONFIRMED via reasoning_tokens>0: {openrouter_thinking_confirmed} "
          f"(usage={or_usage})", file=sys.stderr)

    route_check_passed = same_enum_choice and openrouter_thinking_confirmed
    default_route = "openrouter" if route_check_passed else "direct_anthropic"
    print(f"  ROUTE CHECK: {'PASS' if route_check_passed else 'FAIL'} -> "
          f"default_route={default_route!r}", file=sys.stderr)

    results["opus_route_check"] = {
        "direct_anthropic": {"action_class": direct_data.get("action_class"), "outcome": direct_data.get("outcome"),
                              "latency_ms": direct_latency_ms, "thinking_requested": direct_thinking_requested},
        "openrouter": {"action_class": or_data.get("action_class"), "outcome": or_data.get("outcome"),
                       "latency_ms": or_latency_ms, "usage": or_usage,
                       "thinking_confirmed": openrouter_thinking_confirmed},
        "same_enum_choice": same_enum_choice, "route_check_passed": route_check_passed,
        "default_route": default_route,
    }

    # --- Step 2: Kimi K3 smoke (frames-only, confirmed) ---
    print("\n[smoke_frontier] === STEP 2: Kimi K3 smoke (position + technique, frames-only) ===", file=sys.stderr)
    kimi_pos_data, _, kimi_pos_usage, kimi_pos_latency = await orc.call_kimi_frames_axis(
        api_key=openrouter_key, youtube_url=window["video_url_watch"],
        window_start=window["window_start_s"], window_end=window["window_end_s"], fps=gc.FPS,
        system_instruction=highlight_axes.POSITION_SYSTEM_PROMPT,
        prompt_text=highlight_axes.build_position_prompt(None), thinking="high",
    )
    print(f"  kimi position: {kimi_pos_data} latency_ms={kimi_pos_latency:.0f} usage={kimi_pos_usage}", file=sys.stderr)
    ledger.record(gc.LedgerEntry(
        gate="phase_b_smoke", window_id=SMOKE_WINDOW_ID, match_id=match_id, arm="kimi_smoke",
        call_name="position", model_id=orc.KIMI_K3_MODEL_ID, status="ok", error=None, http_status=None,
        latency_ms=kimi_pos_latency, usage_metadata=kimi_pos_usage, cost_usd=kimi_pos_usage.get("cost_usd"),
        timestamp_utc=_now_utc(), thinking_level="high", provider_route="openrouter",
    ))

    kimi_tech_data, _, kimi_tech_usage, kimi_tech_latency = await orc.call_kimi_frames_axis(
        api_key=openrouter_key, youtube_url=window["video_url_watch"],
        window_start=window["window_start_s"], window_end=window["window_end_s"], fps=gc.FPS,
        system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT,
        prompt_text=highlight_axes.build_technique_prompt(None), thinking="high",
    )
    print(f"  kimi technique: {kimi_tech_data} latency_ms={kimi_tech_latency:.0f} usage={kimi_tech_usage}", file=sys.stderr)
    ledger.record(gc.LedgerEntry(
        gate="phase_b_smoke", window_id=SMOKE_WINDOW_ID, match_id=match_id, arm="kimi_smoke",
        call_name="technique", model_id=orc.KIMI_K3_MODEL_ID, status="ok", error=None, http_status=None,
        latency_ms=kimi_tech_latency, usage_metadata=kimi_tech_usage, cost_usd=kimi_tech_usage.get("cost_usd"),
        timestamp_utc=_now_utc(), thinking_level="high", provider_route="openrouter",
    ))
    kimi_hit = kimi_tech_data.get("action_class") == window["gold_event"]["action_class"]
    print(f"  kimi technique vs gold ({window['gold_event']['action_class']}): {'HIT' if kimi_hit else 'MISS'}",
          file=sys.stderr)

    results["kimi_smoke"] = {
        "position": kimi_pos_data, "technique": kimi_tech_data,
        "usage": {"position": kimi_pos_usage, "technique": kimi_tech_usage},
        "hit_vs_gold": kimi_hit,
    }

    # --- Step 3: Opus smoke via the selected default route ---
    print(f"\n[smoke_frontier] === STEP 3: Opus smoke via default_route={default_route!r} "
          "(position + technique) ===", file=sys.stderr)
    if default_route == "openrouter":
        opus_pos_data, _, opus_pos_usage, opus_pos_latency = await orc.call_opus_via_openrouter_axis(
            api_key=openrouter_key, youtube_url=window["video_url_watch"],
            window_start=window["window_start_s"], window_end=window["window_end_s"], fps=gc.FPS,
            system_instruction=highlight_axes.POSITION_SYSTEM_PROMPT,
            prompt_text=highlight_axes.build_position_prompt(None), thinking="high",
        )
        opus_smoke_model_id = orc.OPUS_48_OPENROUTER_MODEL_ID
    else:
        from service.pipelines.executors import RunContext
        import time
        ctx = RunContext(
            youtube_id="", youtube_url=window["video_url_watch"], start_sec=0, end_sec=0,
            gemini_api_key="", request_timeout_ms=600_000, anthropic_api_key=cfg.anthropic_api_key,
        )
        t0 = time.perf_counter()
        opus_pos_data, _ = await claude_frames.analyze_axis_json(
            ctx, model_id=orc.OPUS_48_ANTHROPIC_MODEL_ID,
            window_start=window["window_start_s"], window_end=window["window_end_s"],
            prompt_text=highlight_axes.build_position_prompt(None),
            system_instruction=highlight_axes.POSITION_SYSTEM_PROMPT, fps=gc.FPS, thinking="high",
        )
        opus_pos_latency = (time.perf_counter() - t0) * 1000.0
        opus_pos_usage = None
        opus_smoke_model_id = orc.OPUS_48_ANTHROPIC_MODEL_ID
    print(f"  opus position ({default_route}): {opus_pos_data}", file=sys.stderr)
    ledger.record(gc.LedgerEntry(
        gate="phase_b_smoke", window_id=SMOKE_WINDOW_ID, match_id=match_id, arm="opus_smoke",
        call_name="position", model_id=opus_smoke_model_id, status="ok", error=None, http_status=None,
        latency_ms=opus_pos_latency, usage_metadata=opus_pos_usage,
        cost_usd=(opus_pos_usage.get("cost_usd") if opus_pos_usage else None),
        timestamp_utc=_now_utc(), thinking_level="high", provider_route=default_route,
    ))

    # technique already captured in step 1 (direct_data / or_data) — reuse rather
    # than spend a 3rd redundant call for the smoke.
    opus_tech_data = direct_data if default_route == "direct_anthropic" else or_data
    opus_hit = opus_tech_data.get("action_class") == window["gold_event"]["action_class"]
    print(f"  opus technique (reused from step 1): {opus_tech_data.get('action_class')} vs gold "
          f"({window['gold_event']['action_class']}): {'HIT' if opus_hit else 'MISS'}", file=sys.stderr)

    results["opus_smoke"] = {
        "position": opus_pos_data, "technique_reused_from_step1": opus_tech_data.get("action_class"),
        "route_used": default_route, "hit_vs_gold": opus_hit,
    }

    gc.write_json(gc.S12_GATES_DIR / "smoke_frontier_lanes_result.json", results)
    ledger.flush()
    print(f"\n[smoke_frontier] DONE. Results written to smoke_frontier_lanes_result.json. "
          f"{len(ledger.entries)} total ledger entries (cumulative, all gates). "
          f"Calls this run: 5 (2 route-check + 2 Kimi + 1 Opus-position; Opus technique reused from step 1, "
          f"no extra spend).", file=sys.stderr)


def main(argv=None) -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
