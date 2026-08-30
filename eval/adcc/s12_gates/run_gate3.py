"""Gate 3 driver — Files-API ingestion-path fidelity probe.

Binding spec: `GATES_3_4_DESIGN.md` §H, AMENDED by `VERDICTS.md` §4
(2026-07-26, after Gate 1 returned NO-GO): primary architecture-under-test
repointed from Arm T to Arm S (the presumptive shipping baseline). Read
`VERDICTS.md` §4 together with §H — §H's original text is the mechanics
(call shape, output contract), §4 is the corrected pairing/cost.

**Revised spend (VERDICTS.md §4): 8 (Arm-S primary, Files-API side only —
YouTube side reused from Gate 1's `gate1_predictions_arm_s.json` at zero
extra cost) + 9 (Arm-T spot-check, 3 windows x 3 calls, Files-API side only
— YouTube side reused from `gate1_predictions_arm_t.json`) + 4 (fps-honoring
re-confirmation, technique-only) = 21 calls.** Exactly ONE upload per
window is needed for the whole gate (8 total) — the spot-check windows are
a SUBSET of the same 8-window draw (first 3 by window_id) and the
fps-honoring windows are also a subset (first 2 by window_id), so their
uploads are reused, never re-uploaded.

**§H.2 correction (VERDICTS.md §4, folded in per the CEO's explicit
instruction):** every Files-API call in this gate MUST go through
`files_api_reroute.FilesApiTimeAnalyzer` (never the base
`SimplifiedTagsTimeAnalyzer`) and `ReroutedUpload.call_window_bounds()`
(never a hand-derived offset) — these are the two real bugs
(missing `mime_type`, absolute-vs-file-relative `video_metadata` offsets)
found and fixed during the §G.1 reroute build; §H's original draft assumed
neither existed.

Reuses `run_gate1.run_arm_s_window`/`run_arm_t_window`/`_reconcile_validator`
directly (byte-identical call/reconciliation logic to Gate 1 — never a
second, hand-copied implementation) and `files_api_reroute.RerouteUploadCache`
for upload management (same still-ACTIVE-reuse discipline as Gates 1/2).

**Halt policy (design does not state one explicitly for Gate 3 — inferred,
conservative reading, flagged in the handoff report): ANY call_failed
anywhere in this gate's 21-call core spend halts immediately** (mirrors
Gate 2's "scarcity of n" precedent — Gate 3's n=8/3/2 sub-samples are all
too small to absorb even one exclusion without materially changing what's
measured).

Run from the repo root:

    .venv/bin/python -m eval.adcc.s12_gates.run_gate3                # dry-run plan only
    .venv/bin/python -m eval.adcc.s12_gates.run_gate3 --execute       # spends real Gemini quota (21 calls)
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections import Counter
from typing import Optional

from eval.adcc.s12_gates import files_api_reroute as reroute
from eval.adcc.s12_gates import gate_common as gc
from eval.adcc.s12_gates.run_gate1 import run_arm_s_window, run_arm_t_window

GATE3_SEED = 20260726
GATE3_N = 8
SPOT_CHECK_N = 3
FPS_CHECK_N = 2
GATE3_PREDICTIONS_PATH = gc.S12_GATES_DIR / "gate3_predictions.json"

# --------------------------------------------------------------------------- #
# §G.2 replacement — window `w17`'s source video (`FO4jG04aGqc`) is
# YouTube-unavailable ("blocked due to claimed content by SME", yt-dlp-
# reproduced 2026-07-26 — a real, external, permanent failure, not a code
# bug). CEO-authorized (2026-07-26), option (a): spend the replacement per
# §G.2's procedure. The replacement draw is FULLY DETERMINISTIC and was
# computed BEFORE this authorization (fresh `random.Random(SEED_REPLACEMENT)`
# over the remaining 29-window eligible pool, i.e. the original 8-window draw
# removed from the 37-window eligible set) — reported here, never re-derived
# ad hoc. Every artifact this substitution touches carries an explicit
# `replaced_window_id`/`replacement_reason` tag — NEVER a silent swap.
# --------------------------------------------------------------------------- #
REPLACED_WINDOW_ID = "650d20c745120c8433e9d059__w17"
REPLACEMENT_WINDOW_ID = "6661e503716f12758093d038__w1"
REPLACEMENT_REASON = (
    "w17 (650d20c745120c8433e9d059__w17, guard_pass, video FO4jG04aGqc) was dropped from "
    "the §H.1 draw: its source video is YouTube-unavailable ('ERROR: [youtube] FO4jG04aGqc: "
    "Video unavailable. It was blocked due to the claimed content by SME.', yt-dlp-reproduced "
    "2026-07-26) — an external, permanent failure, not a Files-API/Gemini issue. Replaced per "
    "GATES_1_2_DESIGN.md §G.2's procedure: a FRESH random.Random(gate_common.SEED_REPLACEMENT="
    "20260727) instance sampling 1 window from the remaining 29-window eligible pool (the "
    "original 37-window §H.1-eligible set minus the 8 already drawn) selected "
    "6661e503716f12758093d038__w1 (back_take, video dsldlavgLz4). CEO-authorized (option a), "
    "Bezos-reviewed, 2026-07-26 — never a silent substitution."
)


class Gate3HaltError(RuntimeError):
    """Raised on ANY call_failed within Gate 3's core spend — see module
    docstring's halt-policy note."""


def select_gate3_windows() -> tuple[list[str], dict]:
    """§H.1 — deterministic n=8 draw, WITH the §G.2 `w17` -> `w1` replacement
    applied (see module-level constants above). Returns ``(sorted_window_ids,
    windows_by_id)`` — ``sorted_window_ids`` is the FINAL, post-replacement
    list actually run. Re-draws once (fresh `Random(GATE3_SEED)`
    re-invoked — there is no other well-defined "next" draw, same reasoning
    as §G.2's replacement-draw correction) if any `action_class` exceeds 3
    of 8 in the ORIGINAL draw — a purely defensive check; this run's actual
    draw does not trigger it (verified: max class count = 3)."""
    arm_t_records = gc.read_json(gc.GATE1_ARM_T_PATH)
    eligible = sorted(
        r["window_id"] for r in arm_t_records
        if r["status"] != "call_failed" and r.get("ingestion_path") != "files_api"
    )
    manifest = gc.read_json(gc.GATE1_MANIFEST_PATH)
    windows_by_id = {w["window_id"]: w for w in manifest["windows"]}

    rng = random.Random(GATE3_SEED)
    draw = rng.sample(eligible, GATE3_N)
    classes = Counter(windows_by_id[w]["gold_event"]["action_class"] for w in draw)
    if max(classes.values()) > 3:
        print(f"select_gate3_windows: degenerate draw {classes} — re-drawing once.", file=sys.stderr)
        rng = random.Random(GATE3_SEED)
        draw = rng.sample(eligible, GATE3_N)
        classes = Counter(windows_by_id[w]["gold_event"]["action_class"] for w in draw)
        if max(classes.values()) > 3:
            raise RuntimeError(f"select_gate3_windows: still degenerate after one re-draw: {classes}")

    if REPLACED_WINDOW_ID not in draw:
        raise RuntimeError(
            f"select_gate3_windows: expected {REPLACED_WINDOW_ID!r} in the original draw to replace — "
            f"not found. Original draw: {sorted(draw)}. Refusing to silently apply a stale substitution."
        )
    remaining_pool = sorted(set(eligible) - set(draw))
    replacement_rng = random.Random(20260727)  # gate_common.SEED_REPLACEMENT
    computed_replacement = replacement_rng.sample(remaining_pool, 1)[0]
    if computed_replacement != REPLACEMENT_WINDOW_ID:
        raise RuntimeError(
            f"select_gate3_windows: re-derived replacement {computed_replacement!r} does not match the "
            f"pre-authorized {REPLACEMENT_WINDOW_ID!r} — corpus/eligibility drift since authorization. "
            "Refusing to proceed on a mismatch."
        )
    final_draw = [w if w != REPLACED_WINDOW_ID else REPLACEMENT_WINDOW_ID for w in draw]

    return sorted(final_draw), windows_by_id


def _print_dry_run_plan(window_ids: list[str], windows_by_id: dict) -> None:
    gc.assert_thinking_high_in_generate_config()  # [RE-RUN] fail loud before any plan is even printed
    spot_check = window_ids[:SPOT_CHECK_N]
    fps_check = window_ids[:FPS_CHECK_N]
    print(f"GATE 3 DRY-RUN PLAN — {len(window_ids)} windows drawn, "
          f"{len(window_ids)} Arm-S primary (1 call each) + {SPOT_CHECK_N} Arm-T spot-check (3 calls each) "
          f"+ {FPS_CHECK_N} fps-honoring (2 calls each) = "
          f"{len(window_ids) + SPOT_CHECK_N * 3 + FPS_CHECK_N * 2} Gemini call(s), "
          f"{len(window_ids)} upload(s) (reused across sub-experiments).", file=sys.stderr)
    for wid in window_ids:
        w = windows_by_id[wid]
        tags = []
        if wid in spot_check:
            tags.append("spot_check")
        if wid in fps_check:
            tags.append("fps_check")
        print(f"  {wid}  action_class={w['gold_event']['action_class']}  "
              f"window=[{w['window_start_s']}, {w['window_end_s']}]  {' '.join(tags)}", file=sys.stderr)
    print("Zero Gemini calls made. Pass --execute to spend real quota.", file=sys.stderr)


async def _amain(args: argparse.Namespace) -> None:
    window_ids, windows_by_id = select_gate3_windows()

    if not args.execute:
        _print_dry_run_plan(window_ids, windows_by_id)
        return

    gc.assert_thinking_high_in_generate_config()  # [RE-RUN] belt-and-suspenders — re-verified right before real spend

    print("WARNING: --execute fires Gate 3's real 21-call spend. Ctrl-C now if not authorized. "
          "Proceeding in 3 real seconds...", file=sys.stderr)
    await asyncio.sleep(3)

    cfg = gc.load_service_config()
    client = gc.build_gemini_client(cfg)
    retry_config = gc.build_retry_config(cfg)
    ledger = gc.CallLedger()
    upload_cache = reroute.RerouteUploadCache(client, ledger, gate="gate3")

    # Reroute analyzers (mime_type + file-relative offsets, §H.2 correction)
    from service.pipelines import highlight_axes, simplified_tags  # noqa: F401 (schema/prompt builders)

    reroute_arm_s_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=None, retry_config=retry_config,
    )
    reroute_position_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=highlight_axes.POSITION_SYSTEM_PROMPT, retry_config=retry_config,
    )
    reroute_technique_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT, retry_config=retry_config,
    )
    reroute_validator_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=highlight_axes.VALIDATOR_SYSTEM_PROMPT, retry_config=retry_config,
    )
    # Dedicated technique analyzer for the fps-honoring check (separate instance,
    # never shares last_* instrumentation state with the spot-check's technique calls).
    reroute_fps_technique_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT, retry_config=retry_config,
    )

    # [RE-RUN 2026-07-26] Pair against the SAME model's Gate 1 output — the qualified
    # re-run's Gate 1 predictions (versioned), NOT the v1 flash-lite canonicals. A fair
    # paired comparison requires Gate 3's "already-captured YouTube-URI baseline" to be
    # from the SAME model the Files-API side is now using.
    gate1_arm_t_source = gc.versioned_path(gc.GATE1_ARM_T_PATH)
    gate1_arm_s_source = gc.versioned_path(gc.GATE1_ARM_S_PATH)
    if not gate1_arm_t_source.exists() or not gate1_arm_s_source.exists():
        raise RuntimeError(
            f"run_gate3: re-run requires Gate 1's qualified-model outputs to exist first — "
            f"missing {gate1_arm_t_source if not gate1_arm_t_source.exists() else gate1_arm_s_source}. "
            "Run run_gate1.py --execute (qualified model) before Gate 3."
        )
    arm_t_records = {r["window_id"]: r for r in gc.read_json(gate1_arm_t_source)}
    arm_s_records = {r["window_id"]: r for r in gc.read_json(gate1_arm_s_source)}

    # [RE-RUN] versioned Gate-3 output paths (never overwrite the flash-lite v1 files).
    primary_out_path = gc.versioned_path(gc.S12_GATES_DIR / "gate3_predictions_primary.json")
    spot_check_out_path = gc.versioned_path(gc.S12_GATES_DIR / "gate3_predictions_spot_check.json")
    fps_check_out_path = gc.versioned_path(gc.S12_GATES_DIR / "gate3_predictions_fps_check.json")
    upload_metrics_out_path = gc.versioned_path(gc.S12_GATES_DIR / "gate3_upload_metrics.json")
    predictions_out_path = gc.versioned_path(GATE3_PREDICTIONS_PATH)

    # Resume-on-rerun (same discipline as run_gate4.py — a prior partial run may have
    # already completed some windows; never re-spend on those). A window counts as
    # "already done" for a given sub-experiment if its record's own status is not
    # call_failed (the halt policy means call_failed should never actually be persisted
    # here, but the check is defensive).
    primary: list[dict] = gc.read_json(primary_out_path) if primary_out_path.exists() else []
    spot_check: list[dict] = gc.read_json(spot_check_out_path) if spot_check_out_path.exists() else []
    fps_check: list[dict] = gc.read_json(fps_check_out_path) if fps_check_out_path.exists() else []
    uploads_meta: list[dict] = gc.read_json(upload_metrics_out_path) if upload_metrics_out_path.exists() else []

    primary_done = {p["window_id"] for p in primary if p["files_api_arm_s"]["status"] != "call_failed"}
    spot_check_done = {p["window_id"] for p in spot_check if p["files_api_arm_t"]["status"] != "call_failed"}
    fps_check_done = {p["window_id"] for p in fps_check}
    uploads_done = {u["window_id"] for u in uploads_meta}
    if primary_done:
        print(f"[gate3] resume: primary already done for {sorted(primary_done)}", file=sys.stderr)
    if spot_check_done:
        print(f"[gate3] resume: spot_check already done for {sorted(spot_check_done)}", file=sys.stderr)
    if fps_check_done:
        print(f"[gate3] resume: fps_check already done for {sorted(fps_check_done)}", file=sys.stderr)

    def _halt_if_failed(records: list[dict], where: str) -> None:
        for rec in records:
            if rec.get("status") == "call_failed":
                raise Gate3HaltError(f"Gate 3 HALT: call_failed in {where} for {rec.get('window_id')}: {rec}")

    spot_check_ids = window_ids[:SPOT_CHECK_N]
    fps_check_ids = window_ids[:FPS_CHECK_N]

    for wid in window_ids:
        if wid in primary_done and (wid not in spot_check_ids or wid in spot_check_done) \
                and (wid not in fps_check_ids or wid in fps_check_done):
            print(f"[gate3] window {wid} — fully done in a prior run, skipping entirely.", file=sys.stderr)
            continue

        w = windows_by_id[wid]
        is_replacement = wid == REPLACEMENT_WINDOW_ID

        if wid in uploads_done:
            print(f"[gate3] window {wid} — reusing already-recorded upload from a prior run...", file=sys.stderr)
            upload = await upload_cache.ensure(w)  # cheap: RerouteUploadCache re-verifies/reuses if still ACTIVE
            call_bounds = upload.call_window_bounds(w["window_start_s"], w["window_end_s"])
        else:
            print(f"[gate3] window {wid} ({w['gold_event']['action_class']})"
                  f"{' [§G.2 REPLACEMENT for ' + REPLACED_WINDOW_ID + ']' if is_replacement else ''}"
                  " — ensuring Files-API upload...", file=sys.stderr)
            upload = await upload_cache.ensure(w)
            call_bounds = upload.call_window_bounds(w["window_start_s"], w["window_end_s"])
            uploads_meta.append({
                "window_id": wid, "file_uri": upload.file_uri, "file_name": upload.file_name,
                "upload_latency_ms": upload.upload_latency_ms, "active_wait_ms": upload.active_wait_ms,
                "sec_start_s": upload.sec_start_s, "call_bounds": list(call_bounds),
                "is_replacement": is_replacement,
            })
            gc.write_json(upload_metrics_out_path, uploads_meta)

        # --- Primary: Arm S, Files-API side (1 call) ---
        if wid not in primary_done:
            arm_s_rec = await run_arm_s_window(
                analyzer=reroute_arm_s_analyzer, ledger=ledger, window=w,
                ingestion_uri=upload.file_uri, ingestion_path="files_api", call_bounds=call_bounds,
                gate="gate3",
            )
            primary.append({
                "window_id": wid, "match_id": w["match_id"], "gold_event": w["gold_event"],
                "youtube_arm_s": arm_s_records.get(wid),
                "files_api_arm_s": arm_s_rec,
                "is_replacement": is_replacement,
                "replaced_window_id": REPLACED_WINDOW_ID if is_replacement else None,
                "replacement_reason": REPLACEMENT_REASON if is_replacement else None,
            })
            gc.write_json(primary_out_path, primary)
            ledger.flush()
            _halt_if_failed([arm_s_rec], "primary Arm-S")
            print(f"  primary arm_s: {arm_s_rec['status']}, {len(arm_s_rec['emitted_events'])} emitted event(s)",
                  file=sys.stderr)

        # --- Spot-check: Arm T, Files-API side (3 calls) — only first 3 windows ---
        if wid in spot_check_ids and wid not in spot_check_done:
            arm_t_rec = await run_arm_t_window(
                position_analyzer=reroute_position_analyzer, technique_analyzer=reroute_technique_analyzer,
                validator_analyzer=reroute_validator_analyzer, ledger=ledger, window=w,
                ingestion_uri=upload.file_uri, ingestion_path="files_api", call_bounds=call_bounds,
                gate="gate3",
            )
            spot_check.append({
                "window_id": wid, "match_id": w["match_id"], "gold_event": w["gold_event"],
                "youtube_arm_t": arm_t_records.get(wid),
                "files_api_arm_t": arm_t_rec,
            })
            gc.write_json(spot_check_out_path, spot_check)
            ledger.flush()
            _halt_if_failed([arm_t_rec], "spot-check Arm-T")
            print(f"  spot_check arm_t: {arm_t_rec['status']}, verdict={arm_t_rec.get('verdict')}",
                  file=sys.stderr)

        # --- fps-honoring: technique-only @ fps=1 and fps=10 — only first 2 windows ---
        if wid in fps_check_ids and wid not in fps_check_done:
            fps_results = {}
            for fps_val in (1, 10):
                data, err, call_record = await gc.timed_axis_call(
                    ledger=ledger, gate="gate3", window_id=wid, match_id=w["match_id"],
                    arm="fps_check", call_name=f"technique_fps{fps_val}",
                    analyzer=reroute_fps_technique_analyzer, youtube_url=upload.file_uri,
                    window_start=call_bounds[0], window_end=call_bounds[1],
                    response_schema=highlight_axes.build_technique_schema(),
                    prompt_text=highlight_axes.build_technique_prompt(None),
                    fps=fps_val, media_resolution=None, ingestion_path="files_api",
                )
                fps_results[f"fps{fps_val}"] = call_record
                _halt_if_failed([{"status": call_record["status"], "window_id": wid}], f"fps-honoring fps={fps_val}")
            fps_check.append({"window_id": wid, "match_id": w["match_id"], **fps_results})
            gc.write_json(fps_check_out_path, fps_check)
            ledger.flush()
            print(f"  fps_check: fps1={fps_results['fps1']['parsed']} fps10={fps_results['fps10']['parsed']}",
                  file=sys.stderr)

    # Bonus finding — zero incremental cost, reused from Gate 1/2's already-captured data.
    bonus = {
        "note": (
            "Bonus finding (§H.4) — NOT part of the n=8 fidelity sample. For w37/occ0/occ11, "
            "Files-API succeeded where native YouTube ingestion 400'd unconditionally on every "
            "probed offset/fps. Evidence that Files-API is a viable fallback/workaround for the "
            "undocumented duration-cap bug — a claim about VIABILITY, not FIDELITY; not blended "
            "into the n=8 agreement_rate/accuracy numbers above."
        ),
        "windows": ["66b009481c62a1db7a53d504__w37", "66afe5ab1c62a1db7a53b80b__occ0", "66a6ff021c62a1db7a5133c0__occ11"],
        "source_artifacts": ["gate1_predictions_arm_s.json", "gate1_predictions_arm_t.json", "gate2_predictions.json",
                              "reroute_manifest.json"],
    }

    predictions = {
        "primary_arm_s": primary, "spot_check_arm_t": spot_check, "fps_check": fps_check,
        "bonus_finding": bonus, "window_ids_draw": window_ids,
        "spot_check_ids": spot_check_ids, "fps_check_ids": fps_check_ids,
        "replacement": {
            "replaced_window_id": REPLACED_WINDOW_ID, "replacement_window_id": REPLACEMENT_WINDOW_ID,
            "reason": REPLACEMENT_REASON,
        },
    }
    gc.write_json(predictions_out_path, predictions)

    total_cost = ledger.total_cost_usd()
    print(f"[gate3] done. {len(window_ids)} windows x 1 (primary) + {SPOT_CHECK_N} x 3 (spot-check) + "
          f"{FPS_CHECK_N} x 2 (fps-check) = {len(window_ids) + SPOT_CHECK_N*3 + FPS_CHECK_N*2} call(s) attempted. "
          f"Ledger total (cumulative): {ledger.total_calls()} call(s), ${total_cost:.4f}.", file=sys.stderr)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="DANGER: spends real Gemini quota. Default is dry-run.")
    args = parser.parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
