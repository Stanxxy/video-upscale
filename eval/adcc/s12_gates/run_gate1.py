"""LeCun cutover Gate 1 driver (``GATES_1_2_DESIGN.md`` §B) — Arm S
(``simplified-tags-time-v1``, single call) vs Arm T (position + technique +
validator, production's real ``highlight_analyze`` emission contract).

**Dry-run by default (mirrors ``eval/adcc/predict/run_predictions.py``'s
``--execute`` gate discipline) — ZERO Gemini calls unless ``--execute`` is
passed.** ``--limit-windows N`` restricts a run to the FIRST N windows of
the frozen manifest (index order) — this is the SMOKE-TEST knob; the
founder-approved smoke budget for this gate is 2 windows (8 calls, out of
the 34-call buffer). ``--execute`` with NO ``--limit-windows`` runs the
FULL 38-window / 152-call batch and is a SEPARATE, founder-gated action —
do not invoke it without explicit authorization (LeCun/Bezos sign-off).

Reuses (never reimplements) ``service.pipelines.simplified_tags.
SimplifiedTagsTimeAnalyzer`` and ``service.pipelines.executors._call_axis_json``
for the actual Gemini transport/retry/JSON-parse plumbing, and
``service.pipelines.highlight_axes`` for the position/technique/validator
prompt+schema builders — per the design doc's own "Summary for the
engineer". The validator reconciliation logic below
(``_reconcile_validator``) is a byte-faithful port of
``executors.highlight_analyze_node``'s real reconciliation switch (verified
read, that module's lines ~1440-1500), per design §B.2's explicit ruling
("score the validator-FINAL verdict... per executors.highlight_analyze_node's
real reconciliation logic").

Run from the repo root:

    .venv/bin/python -m eval.adcc.s12_gates.run_gate1                       # dry-run plan only
    .venv/bin/python -m eval.adcc.s12_gates.run_gate1 --execute --limit-windows 2   # SMOKE TEST
    .venv/bin/python -m eval.adcc.s12_gates.run_gate1 --execute             # FULL BATCH — founder-gated
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

from eval.adcc.s12_gates import files_api_reroute as reroute
from eval.adcc.s12_gates import gate_common as gc
from service.pipelines import highlight_axes, simplified_tags

CALLS_PER_WINDOW = 4  # 1 Arm S + 3 Arm T (position, technique, validator)


def _reconcile_validator(
    position_label: Optional[str], position_justification: Optional[str],
    technique_label: Optional[str], outcome_label: Optional[str], technique_justification: Optional[str],
    val_data: Optional[dict],
) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Returns ``(status, final_position, final_action_class, final_outcome,
    verdict, ditch_reason)``. Byte-faithful port of
    ``executors.highlight_analyze_node``'s reconciliation switch for
    ``max_validator_iterations=1`` (v1 shipped default, per design §B.2):
    single-field-correction on 'disagree', unchanged labels on 'agree', zero
    emission on 'ditch' OR on a missing/unrecognized verdict (the for-loop's
    own ``else`` clause at iteration exhaustion — this driver never loops,
    since design §D fixes ``max_validator_iterations=1`` for Gate 1)."""
    verdict = val_data.get("verdict") if isinstance(val_data, dict) else None
    if verdict == "ditch":
        reason = (
            (val_data.get("reason") if isinstance(val_data, dict) else None)
            or "validator invoked the strict ditch criterion"
        )
        return "ditched", None, None, None, verdict, reason
    if verdict == "agree":
        return "analyzed", position_label, technique_label, outcome_label, verdict, None
    if verdict == "disagree":
        wrong_label = val_data.get("wrong_label") if isinstance(val_data, dict) else None
        if wrong_label == "position":
            fp = (val_data.get("final_position") if isinstance(val_data, dict) else None) or position_label
            return "analyzed", fp, technique_label, outcome_label, verdict, None
        if wrong_label == "action_class":
            fac = (val_data.get("final_action_class") if isinstance(val_data, dict) else None) or technique_label
            return "analyzed", position_label, fac, outcome_label, verdict, None
        if wrong_label == "outcome":
            fo = (val_data.get("final_outcome") if isinstance(val_data, dict) else None) or outcome_label
            return "analyzed", position_label, technique_label, fo, verdict, None
        # wrong_label null/absent/unrecognized on 'disagree' -> NO CHANGE (executors.py's
        # own documented fallback), not a fabricated correction.
        return "analyzed", position_label, technique_label, outcome_label, verdict, None
    # verdict missing/unrecognized -> unresolved -> ditch (for...else at max_validator_iterations=1).
    return "ditched", None, None, None, verdict, "validator disagreement unresolved after 1 iteration(s)"


async def run_arm_s_window(
    *, analyzer: "simplified_tags.SimplifiedTagsTimeAnalyzer", ledger: gc.CallLedger,
    window: dict, ingestion_uri: str, ingestion_path: str = "youtube_uri",
    call_bounds: Optional[tuple[float, float]] = None, gate: str = "gate1",
) -> dict:
    """``call_bounds``, when provided, are the FILE-RELATIVE
    ``video_metadata`` offsets actually sent to Gemini (§G.1 reroute lane —
    see ``files_api_reroute.py``'s "live bug #2" writeup: an uploaded,
    pre-cropped file's clock starts at 0:00, not the original video's
    absolute match-clock second). ``window["window_start_s"]``/
    ``["window_end_s"]`` (ABSOLUTE match-clock) are used for everything
    else — the output record, and the emitted-event conversion below, which
    stays correct regardless of ingestion path because Gemini's returned
    ``clip["start_s"]``/``["end_s"]`` are always relative to whatever
    ``start_offset`` was actually sent, and
    ``absolute_window_start + clip_relative_s == sec_start + call_start +
    clip_relative_s`` by construction — no special-casing needed there.

    ``gate`` (additive — default ``"gate1"`` preserves this driver's own
    prior ledger-tagging behavior byte-for-byte) lets OTHER drivers reuse
    this function without their calls being mislabeled in
    ``gate_budget_ledger.json`` as Gate 1 spend (live bug found + fixed
    2026-07-26, during Gate 3's reuse of this function — Gate 3's first run
    logged its primary/spot-check calls under ``gate="gate1"`` until this
    parameter was added)."""
    window_id = window["window_id"]
    match_id = window["match_id"]
    window_start, window_end = window["window_start_s"], window["window_end_s"]
    call_start, call_end = call_bounds if call_bounds is not None else (window_start, window_end)

    data, err, call_record = await gc.timed_axis_call(
        ledger=ledger, gate=gate, window_id=window_id, match_id=match_id, arm="arm_s", call_name="arm_s",
        analyzer=analyzer, youtube_url=ingestion_uri,
        window_start=call_start, window_end=call_end,
        response_schema=simplified_tags.build_video_response_schema(),
        prompt_text=simplified_tags.build_video_prompt(None),
        fps=gc.FPS, media_resolution=None, ingestion_path=ingestion_path,
    )

    emitted_events: list[dict] = []
    status = "call_failed" if err is not None else "ok"
    if err is None:
        clips = data.get("clips", []) if isinstance(data, dict) else []
        for clip in clips:
            emitted_events.append({
                "start_s": window_start + clip["start_s"],
                "end_s": window_start + clip["end_s"],
                "action_class": clip["action_class"],
                "confidence": clip.get("confidence"),
            })

    return {
        "window_id": window_id, "match_id": match_id, "gate5_straddler": window["gate5_straddler"],
        "gold_event": window["gold_event"], "video_url_watch": window["video_url_watch"],
        "ingestion_path": ingestion_path,
        "window_start_s": window_start, "window_end_s": window_end,
        "settings": {"fps": gc.FPS, "model": gc.MODEL_ID, "thinking": None, "media_resolution": None},
        "calls": [call_record],
        "emitted_events": emitted_events,
        "status": status,
    }


async def run_arm_t_window(
    *, position_analyzer, technique_analyzer, validator_analyzer, ledger: gc.CallLedger, window: dict,
    ingestion_uri: str, ingestion_path: str = "youtube_uri",
    call_bounds: Optional[tuple[float, float]] = None, gate: str = "gate1",
) -> dict:
    """``call_bounds`` — see ``run_arm_s_window``'s docstring (§G.1 reroute
    lane, file-relative offsets). Arm T's emitted event is the window's own
    (ABSOLUTE) span regardless — none of position/technique/validator
    return their own timestamps — so only the THREE ``gc.timed_axis_call``
    invocations below use ``call_start``/``call_end``; ``_arm_t_record``
    still receives the unmodified ``window`` dict. ``gate`` — see
    ``run_arm_s_window``'s docstring (ledger-tagging fix for reuse by other
    drivers, e.g. Gate 3)."""
    window_id = window["window_id"]
    match_id = window["match_id"]
    window_start, window_end = window["window_start_s"], window["window_end_s"]
    call_start, call_end = call_bounds if call_bounds is not None else (window_start, window_end)
    youtube_url = ingestion_uri
    description = None  # isolated-window eval — no upstream scan description exists here
    calls: list[dict] = []

    pos_data, pos_err, pos_record = await gc.timed_axis_call(
        ledger=ledger, gate=gate, window_id=window_id, match_id=match_id, arm="arm_t", call_name="position",
        analyzer=position_analyzer, youtube_url=youtube_url, window_start=call_start, window_end=call_end,
        response_schema=highlight_axes.build_position_schema(),
        prompt_text=highlight_axes.build_position_prompt(description),
        fps=gc.FPS, media_resolution=None, ingestion_path=ingestion_path,
    )
    calls.append(pos_record)
    if pos_err is not None:
        return _arm_t_record(window, calls, status="call_failed", final_action_class=None, ingestion_path=ingestion_path)

    tech_data, tech_err, tech_record = await gc.timed_axis_call(
        ledger=ledger, gate=gate, window_id=window_id, match_id=match_id, arm="arm_t", call_name="technique",
        analyzer=technique_analyzer, youtube_url=youtube_url, window_start=call_start, window_end=call_end,
        response_schema=highlight_axes.build_technique_schema(),
        prompt_text=highlight_axes.build_technique_prompt(description),
        fps=gc.FPS, media_resolution=None, ingestion_path=ingestion_path,
    )
    calls.append(tech_record)
    if tech_err is not None:
        return _arm_t_record(window, calls, status="call_failed", final_action_class=None, ingestion_path=ingestion_path)

    position_label = pos_data.get("position") if isinstance(pos_data, dict) else None
    position_justification = pos_data.get("justification") if isinstance(pos_data, dict) else None
    technique_label = tech_data.get("action_class") if isinstance(tech_data, dict) else None
    outcome_label = tech_data.get("outcome") if isinstance(tech_data, dict) else None
    technique_justification = tech_data.get("justification") if isinstance(tech_data, dict) else None

    val_data, val_err, val_record = await gc.timed_axis_call(
        ledger=ledger, gate=gate, window_id=window_id, match_id=match_id, arm="arm_t", call_name="validator",
        analyzer=validator_analyzer, youtube_url=youtube_url, window_start=call_start, window_end=call_end,
        response_schema=highlight_axes.build_validator_schema(),
        prompt_text=highlight_axes.build_validator_prompt(
            description=description, position_label=position_label, position_justification=position_justification,
            technique_label=technique_label, outcome_label=outcome_label,
            technique_justification=technique_justification,
        ),
        fps=gc.FPS, media_resolution="high",  # production default — the one lever NOT varied by Gate 1 (§B.2)
        ingestion_path=ingestion_path,
    )
    calls.append(val_record)
    if val_err is not None:
        return _arm_t_record(window, calls, status="call_failed", final_action_class=None, ingestion_path=ingestion_path)

    status, final_position, final_action_class, final_outcome, verdict, ditch_reason = _reconcile_validator(
        position_label, position_justification, technique_label, outcome_label, technique_justification, val_data,
    )
    return _arm_t_record(
        window, calls, status=status, final_action_class=final_action_class,
        final_position=final_position, final_outcome=final_outcome, verdict=verdict, ditch_reason=ditch_reason,
        ingestion_path=ingestion_path,
    )


def _arm_t_record(
    window: dict, calls: list[dict], *, status: str, final_action_class: Optional[str],
    final_position: Optional[str] = None, final_outcome: Optional[str] = None,
    verdict: Optional[str] = None, ditch_reason: Optional[str] = None, ingestion_path: str = "youtube_uri",
) -> dict:
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
        "ingestion_path": ingestion_path,
        "window_start_s": window_start, "window_end_s": window_end,
        "settings": {
            "fps": gc.FPS, "model": gc.MODEL_ID, "thinking": None,
            "media_resolution": {"position": None, "technique": None, "validator": "high"},
        },
        "calls": calls,
        "final_position": final_position, "final_outcome": final_outcome, "verdict": verdict,
        "ditch_reason": ditch_reason,
        "emitted_events": emitted_events,
        "status": status,  # "analyzed" | "ditched" | "call_failed"
    }


def _merge_by_window_id(existing: list[dict], new_records: list[dict]) -> list[dict]:
    by_id = {r["window_id"]: r for r in existing}
    for r in new_records:
        by_id[r["window_id"]] = r
    return list(by_id.values())


def _print_dry_run_plan(windows: list[dict]) -> None:
    gc.assert_thinking_high_in_generate_config()  # [RE-RUN] fail loud before any plan is even printed
    print(f"GATE 1 DRY-RUN PLAN — {len(windows)} window(s), {len(windows) * CALLS_PER_WINDOW} Gemini call(s) "
          f"(1 Arm S + 3 Arm T per window), model={gc.MODEL_ID}, fps={gc.FPS}", file=sys.stderr)
    for w in windows:
        print(f"  [{w['index']:02d}] {w['window_id']}  action_class={w['gold_event']['action_class']}  "
              f"straddler={w['gate5_straddler']}  window=[{w['window_start_s']}, {w['window_end_s']}]",
              file=sys.stderr)
    print("Zero Gemini calls made. Pass --execute to spend real quota.", file=sys.stderr)


async def _amain(args: argparse.Namespace) -> None:
    manifest = gc.read_json(gc.GATE1_MANIFEST_PATH)
    windows = manifest["windows"]
    if args.limit_windows is not None:
        windows = windows[: args.limit_windows]

    if not args.execute:
        _print_dry_run_plan(windows)
        return

    gc.assert_thinking_high_in_generate_config()  # [RE-RUN] belt-and-suspenders — re-verified right before real spend

    if args.limit_windows is None:
        print(
            "WARNING: --execute with no --limit-windows fires the FULL 38-window / 152-call batch. "
            "This is a founder-gated action per the LeCun/Bezos sign-off protocol — Ctrl-C now if this "
            "was not explicitly authorized. Proceeding in 3 real seconds...", file=sys.stderr,
        )
        await asyncio.sleep(3)

    cfg = gc.load_service_config()
    client = gc.build_gemini_client(cfg)
    retry_config = gc.build_retry_config(cfg)
    ledger = gc.CallLedger()

    arm_s_analyzer = gc.build_time_analyzer(client, system_instruction=None, retry_config=retry_config)
    position_analyzer = gc.build_time_analyzer(
        client, system_instruction=highlight_axes.POSITION_SYSTEM_PROMPT, retry_config=retry_config,
    )
    technique_analyzer = gc.build_time_analyzer(
        client, system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT, retry_config=retry_config,
    )
    validator_analyzer = gc.build_time_analyzer(
        client, system_instruction=highlight_axes.VALIDATOR_SYSTEM_PROMPT, retry_config=retry_config,
    )
    # §G.1 reroute-lane analyzer siblings — same system instructions, but
    # FilesApiTimeAnalyzer (mime_type-carrying, see files_api_reroute.py's
    # "Live bug found" writeup). Used ONLY for windows in REROUTE_WINDOW_IDS.
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

    # [RE-RUN 2026-07-26] versioned output paths — NEVER overwrite the flash-lite
    # v1 canonicals (gc.GATE1_ARM_S_PATH/GATE1_ARM_T_PATH stay untouched).
    arm_s_out_path = gc.versioned_path(gc.GATE1_ARM_S_PATH)
    arm_t_out_path = gc.versioned_path(gc.GATE1_ARM_T_PATH)
    existing_arm_s = gc.read_json(arm_s_out_path) if arm_s_out_path.exists() else []
    existing_arm_t = gc.read_json(arm_t_out_path) if arm_t_out_path.exists() else []
    new_arm_s: list[dict] = []
    new_arm_t: list[dict] = []

    # §G.1 reroute lane — the pwGbW5GZgfc-video windows (w37, occ0, occ11)
    # get a Files API upload instead of the YouTube watch URL. ONE upload
    # per window, cached and reused across ALL of that window's calls
    # (§G.1 point 3 — mandatory for cross-arm paired validity).
    upload_cache = reroute.RerouteUploadCache(client, ledger, gate="gate1")

    for w in windows:
        print(f"[gate1] window {w['index']:02d} {w['window_id']} ({w['gold_event']['action_class']})...",
              file=sys.stderr)

        is_rerouted = w["window_id"] in reroute.REROUTE_WINDOW_IDS
        call_bounds = None
        if is_rerouted:
            print(f"  [reroute] {w['window_id']} routed through Files API (§G.1) — "
                  "downloading/normalizing/uploading...", file=sys.stderr)
            upload = await upload_cache.ensure(w)
            ingestion_uri, ingestion_path = upload.file_uri, "files_api"
            call_bounds = upload.call_window_bounds(w["window_start_s"], w["window_end_s"])
            print(f"  [reroute] upload_latency_ms={upload.upload_latency_ms:.0f} "
                  f"active_wait_ms={upload.active_wait_ms:.0f} file={upload.file_name} "
                  f"call_bounds(file-relative)={call_bounds}", file=sys.stderr)
        else:
            ingestion_uri, ingestion_path = w["video_url_watch"], "youtube_uri"

        eff_arm_s_analyzer = reroute_arm_s_analyzer if is_rerouted else arm_s_analyzer
        eff_position_analyzer = reroute_position_analyzer if is_rerouted else position_analyzer
        eff_technique_analyzer = reroute_technique_analyzer if is_rerouted else technique_analyzer
        eff_validator_analyzer = reroute_validator_analyzer if is_rerouted else validator_analyzer

        arm_s_rec = await run_arm_s_window(
            analyzer=eff_arm_s_analyzer, ledger=ledger, window=w,
            ingestion_uri=ingestion_uri, ingestion_path=ingestion_path, call_bounds=call_bounds,
        )
        new_arm_s.append(arm_s_rec)
        print(f"  arm_s: {arm_s_rec['status']}, {len(arm_s_rec['emitted_events'])} emitted event(s)",
              file=sys.stderr)

        arm_t_rec = await run_arm_t_window(
            position_analyzer=eff_position_analyzer, technique_analyzer=eff_technique_analyzer,
            validator_analyzer=eff_validator_analyzer, ledger=ledger, window=w,
            ingestion_uri=ingestion_uri, ingestion_path=ingestion_path, call_bounds=call_bounds,
        )
        new_arm_t.append(arm_t_rec)
        print(f"  arm_t: {arm_t_rec['status']}, verdict={arm_t_rec.get('verdict')}, "
              f"{len(arm_t_rec['emitted_events'])} emitted event(s)", file=sys.stderr)

        # Flush after every window — a crash mid-run never loses already-spent calls.
        gc.write_json(arm_s_out_path, _merge_by_window_id(existing_arm_s, new_arm_s))
        gc.write_json(arm_t_out_path, _merge_by_window_id(existing_arm_t, new_arm_t))
        ledger.flush()

        # §D.1 halt-and-escalate — checked INSIDE the loop, after EVERY window
        # (both arms), so a trip stops the batch IMMEDIATELY (never after
        # already spending the remaining windows' calls). Only meaningful
        # (and only enforced) on a run against the FULL manifest; a
        # --limit-windows smoke slice must not trip this.
        if args.limit_windows is None:
            failed_windows = {
                r["window_id"] for r in new_arm_s if r["status"] == "call_failed"
            } | {
                r["window_id"] for r in new_arm_t if r["status"] == "call_failed"
            }
            if len(failed_windows) > gc.GATE1_FAILURE_HALT_THRESHOLD:
                print(
                    f"[gate1] HALT-AND-ESCALATE tripped after window {w['index']:02d} {w['window_id']} — "
                    f"{len(failed_windows)} window(s) failed (threshold: >{gc.GATE1_FAILURE_HALT_THRESHOLD} of "
                    f"{gc.GATE1_SUBSET_N}). STOPPING — no further windows will be attempted this run.",
                    file=sys.stderr,
                )
                raise gc.GateHaltError(
                    f"Gate 1 HALT-AND-ESCALATE: {len(failed_windows)} window(s) had a call_failed arm "
                    f"(threshold: >{gc.GATE1_FAILURE_HALT_THRESHOLD} of {gc.GATE1_SUBSET_N}). "
                    f"Failed windows: {sorted(failed_windows)}. Halted after window index {w['index']} "
                    f"({w['window_id']}) — {len(windows) - (w['index'] + 1)} window(s) never attempted."
                )

    total_cost = ledger.total_cost_usd()
    print(f"[gate1] done. {len(windows)} window(s) x {CALLS_PER_WINDOW} calls = "
          f"{len(windows) * CALLS_PER_WINDOW} call(s) attempted this run. "
          f"Ledger total (all gates, cumulative): {ledger.total_calls()} call(s), "
          f"${total_cost:.4f} (real, from usage_metadata).", file=sys.stderr)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="DANGER: spends real Gemini quota. Default is dry-run.")
    parser.add_argument(
        "--limit-windows", type=int, default=None,
        help="Restrict to the first N manifest windows (index order) — SMOKE TEST knob. "
             "Omit for the full 38-window batch (founder-gated).",
    )
    args = parser.parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
