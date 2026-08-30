"""LeCun cutover Gate 2 driver (``GATES_1_2_DESIGN.md`` §C) — validator
``media_resolution`` A/B (low vs high) on the 16-clip occlusion subset.

**Dry-run by default — ZERO Gemini calls unless ``--execute`` is passed.**
``--limit-clips N`` restricts a run to the FIRST N manifest clips (index
order) — SMOKE-TEST knob; the founder-approved smoke budget for this gate is
1 clip (4 calls). ``--execute`` with no ``--limit-clips`` runs the FULL
16-clip / 64-call batch and is founder-gated — do not invoke without
explicit authorization.

Per clip: position (once) + technique (once) + validator-low (once) +
validator-high (once), independently invoked (§C: "not chained — validator-
high does not see validator-low's output"). Position/technique results are
NOT re-run per validator condition (§C: "there is no need to re-run them per
validator-resolution condition").
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

from eval.adcc.s12_gates import files_api_reroute as reroute
from eval.adcc.s12_gates import gate_common as gc
from service.pipelines import highlight_axes

CALLS_PER_CLIP = 4  # position + technique + validator_low + validator_high


def _validator_final_class(val_data) -> tuple[str, str]:
    """Returns ``(verdict, final_action_class)``. §C.1: a 'ditch' verdict
    scores as ``correct=False`` (handled by the scorer, not here) — this
    just extracts what the validator actually said, verbatim, never
    substituting a fabricated class on ditch/disagree-with-no-target."""
    verdict = val_data.get("verdict") if isinstance(val_data, dict) else None
    if verdict == "agree":
        return verdict, None  # caller fills in technique_label (unchanged) for 'agree'
    if verdict == "disagree":
        wrong_label = val_data.get("wrong_label") if isinstance(val_data, dict) else None
        if wrong_label == "action_class":
            return verdict, (val_data.get("final_action_class") if isinstance(val_data, dict) else None)
        return verdict, None  # disagree on position/outcome/unrecognized -> action_class unchanged
    return verdict, None  # ditch / missing / unrecognized


def _print_dry_run_plan(clips: list[dict]) -> None:
    gc.assert_thinking_high_in_generate_config()  # [RE-RUN] fail loud before any plan is even printed
    print(f"GATE 2 DRY-RUN PLAN — {len(clips)} clip(s), {len(clips) * CALLS_PER_CLIP} Gemini call(s) "
          f"(position + technique + validator_low + validator_high per clip), model={gc.MODEL_ID}, fps={gc.FPS}",
          file=sys.stderr)
    for c in clips:
        g = c["gold_event"]
        print(f"  [{c['index']:02d}] {c['window_id']}  action_class={g['action_class']}  outcome={g['outcome']}  "
              f"window=[{c['window_start_s']}, {c['window_end_s']}]", file=sys.stderr)
    print("Zero Gemini calls made. Pass --execute to spend real quota.", file=sys.stderr)


def _merge_by_window_id(existing: list[dict], new_records: list[dict]) -> list[dict]:
    by_id = {r["window_id"]: r for r in existing}
    for r in new_records:
        by_id[r["window_id"]] = r
    return list(by_id.values())


async def run_clip(
    *, position_analyzer, technique_analyzer, validator_low_analyzer, validator_high_analyzer,
    ledger: gc.CallLedger, clip: dict, ingestion_uri: str, ingestion_path: str = "youtube_uri",
    call_bounds: Optional[tuple[float, float]] = None,
) -> dict:
    """``call_bounds`` — §G.1 reroute lane, file-relative offsets (see
    ``run_gate1.run_arm_s_window``'s docstring / ``files_api_reroute.py``'s
    "live bug #2" writeup). ``clip["window_start_s"]``/``["window_end_s"]``
    (ABSOLUTE) are still used for the output record."""
    window_id = clip["window_id"]
    match_id = clip["match_id"]
    window_start, window_end = clip["window_start_s"], clip["window_end_s"]
    call_start, call_end = call_bounds if call_bounds is not None else (window_start, window_end)
    youtube_url = ingestion_uri
    description = None
    calls: list[dict] = []

    pos_data, pos_err, pos_record = await gc.timed_axis_call(
        ledger=ledger, gate="gate2", window_id=window_id, match_id=match_id, arm="gate2", call_name="position",
        analyzer=position_analyzer, youtube_url=youtube_url, window_start=call_start, window_end=call_end,
        response_schema=highlight_axes.build_position_schema(),
        prompt_text=highlight_axes.build_position_prompt(description),
        fps=gc.FPS, media_resolution=None, ingestion_path=ingestion_path,
    )
    calls.append(pos_record)
    if pos_err is not None:
        return _clip_record(clip, calls, status="call_failed", ingestion_path=ingestion_path)

    tech_data, tech_err, tech_record = await gc.timed_axis_call(
        ledger=ledger, gate="gate2", window_id=window_id, match_id=match_id, arm="gate2", call_name="technique",
        analyzer=technique_analyzer, youtube_url=youtube_url, window_start=call_start, window_end=call_end,
        response_schema=highlight_axes.build_technique_schema(),
        prompt_text=highlight_axes.build_technique_prompt(description),
        fps=gc.FPS, media_resolution=None, ingestion_path=ingestion_path,
    )
    calls.append(tech_record)
    if tech_err is not None:
        return _clip_record(clip, calls, status="call_failed", position_label=(
            pos_data.get("position") if isinstance(pos_data, dict) else None
        ), ingestion_path=ingestion_path)

    position_label = pos_data.get("position") if isinstance(pos_data, dict) else None
    position_justification = pos_data.get("justification") if isinstance(pos_data, dict) else None
    technique_label = tech_data.get("action_class") if isinstance(tech_data, dict) else None
    outcome_label = tech_data.get("outcome") if isinstance(tech_data, dict) else None
    technique_justification = tech_data.get("justification") if isinstance(tech_data, dict) else None

    validator_prompt = highlight_axes.build_validator_prompt(
        description=description, position_label=position_label, position_justification=position_justification,
        technique_label=technique_label, outcome_label=outcome_label,
        technique_justification=technique_justification,
    )
    validator_schema = highlight_axes.build_validator_schema()

    val_low_data, val_low_err, val_low_record = await gc.timed_axis_call(
        ledger=ledger, gate="gate2", window_id=window_id, match_id=match_id, arm="gate2", call_name="validator_low",
        analyzer=validator_low_analyzer, youtube_url=youtube_url, window_start=call_start, window_end=call_end,
        response_schema=validator_schema, prompt_text=validator_prompt,
        fps=gc.FPS, media_resolution="low", ingestion_path=ingestion_path,
    )
    calls.append(val_low_record)

    val_high_data, val_high_err, val_high_record = await gc.timed_axis_call(
        ledger=ledger, gate="gate2", window_id=window_id, match_id=match_id, arm="gate2", call_name="validator_high",
        analyzer=validator_high_analyzer, youtube_url=youtube_url, window_start=call_start, window_end=call_end,
        response_schema=validator_schema, prompt_text=validator_prompt,
        fps=gc.FPS, media_resolution="high", ingestion_path=ingestion_path,
    )
    calls.append(val_high_record)

    if val_low_err is not None or val_high_err is not None:
        return _clip_record(
            clip, calls, status="call_failed", position_label=position_label, technique_label=technique_label,
            ingestion_path=ingestion_path,
        )

    def _final_class(val_data) -> tuple[str, str]:
        verdict, corrected = _validator_final_class(val_data)
        return verdict, (corrected if corrected else technique_label)

    val_low_verdict, val_low_final_class = _final_class(val_low_data)
    val_high_verdict, val_high_final_class = _final_class(val_high_data)
    # §C.1: a 'ditch' verdict scores correct=False regardless of technique_label — represent
    # this explicitly in the record (final_class None on ditch, never a fabricated label).
    if val_low_verdict == "ditch":
        val_low_final_class = None
    if val_high_verdict == "ditch":
        val_high_final_class = None

    return _clip_record(
        clip, calls, status="ok", position_label=position_label, technique_label=technique_label,
        outcome_label=outcome_label, val_low_verdict=val_low_verdict, val_low_final_class=val_low_final_class,
        val_high_verdict=val_high_verdict, val_high_final_class=val_high_final_class,
        ingestion_path=ingestion_path,
    )


def _clip_record(
    clip: dict, calls: list[dict], *, status: str, position_label=None, technique_label=None,
    outcome_label=None, val_low_verdict=None, val_low_final_class=None,
    val_high_verdict=None, val_high_final_class=None, ingestion_path: str = "youtube_uri",
) -> dict:
    return {
        "window_id": clip["window_id"], "match_id": clip["match_id"], "gold_event": clip["gold_event"],
        "video_url_watch": clip["video_url_watch"],
        "ingestion_path": ingestion_path,
        "window_start_s": clip["window_start_s"], "window_end_s": clip["window_end_s"],
        "settings": {
            "fps": gc.FPS, "model": gc.MODEL_ID, "thinking": None,
            "media_resolution": {"position": None, "technique": None, "validator_low": "low", "validator_high": "high"},
        },
        "calls": calls,
        "position_label": position_label, "technique_label": technique_label, "outcome_label": outcome_label,
        "val_low_verdict": val_low_verdict, "val_low_final_class": val_low_final_class,
        "val_high_verdict": val_high_verdict, "val_high_final_class": val_high_final_class,
        "status": status,  # "ok" | "call_failed"
    }


async def _amain(args: argparse.Namespace) -> None:
    manifest = gc.read_json(gc.GATE2_MANIFEST_PATH)
    clips = manifest["clips"]
    if args.limit_clips is not None:
        clips = clips[: args.limit_clips]

    if not args.execute:
        _print_dry_run_plan(clips)
        return

    gc.assert_thinking_high_in_generate_config()  # [RE-RUN] belt-and-suspenders — re-verified right before real spend

    if args.limit_clips is None:
        print(
            "WARNING: --execute with no --limit-clips fires the FULL 16-clip / 64-call batch. "
            "This is a founder-gated action — Ctrl-C now if this was not explicitly authorized. "
            "Proceeding in 3 real seconds...", file=sys.stderr,
        )
        await asyncio.sleep(3)

    cfg = gc.load_service_config()
    client = gc.build_gemini_client(cfg)
    retry_config = gc.build_retry_config(cfg)
    ledger = gc.CallLedger()

    position_analyzer = gc.build_time_analyzer(
        client, system_instruction=highlight_axes.POSITION_SYSTEM_PROMPT, retry_config=retry_config,
    )
    technique_analyzer = gc.build_time_analyzer(
        client, system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT, retry_config=retry_config,
    )
    validator_low_analyzer = gc.build_time_analyzer(
        client, system_instruction=highlight_axes.VALIDATOR_SYSTEM_PROMPT, retry_config=retry_config,
    )
    validator_high_analyzer = gc.build_time_analyzer(
        client, system_instruction=highlight_axes.VALIDATOR_SYSTEM_PROMPT, retry_config=retry_config,
    )
    # §G.1 reroute-lane analyzer siblings (see run_gate1.py's identical comment
    # + files_api_reroute.py's "Live bug found" writeup) — used ONLY for
    # windows in REROUTE_WINDOW_IDS.
    reroute_position_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=highlight_axes.POSITION_SYSTEM_PROMPT, retry_config=retry_config,
    )
    reroute_technique_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT, retry_config=retry_config,
    )
    reroute_validator_low_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=highlight_axes.VALIDATOR_SYSTEM_PROMPT, retry_config=retry_config,
    )
    reroute_validator_high_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=highlight_axes.VALIDATOR_SYSTEM_PROMPT, retry_config=retry_config,
    )

    # [RE-RUN 2026-07-26] versioned output path — never overwrite the flash-lite v1 canonical.
    predictions_out_path = gc.versioned_path(gc.GATE2_PREDICTIONS_PATH)
    existing = gc.read_json(predictions_out_path) if predictions_out_path.exists() else []
    new_records: list[dict] = []

    # §G.1 reroute lane — see run_gate1.py's identical comment. Shared cache
    # class, but a SEPARATE instance per gate (gate="gate2" tag on ledger rows) —
    # occ0/occ11 each get their OWN upload even though w37 (Gate 1) shares the
    # same source video, since §G.1 point 3 scopes reuse to "ALL of that
    # window's calls", not across gates/windows.
    upload_cache = reroute.RerouteUploadCache(client, ledger, gate="gate2")

    for c in clips:
        g = c["gold_event"]
        print(f"[gate2] clip {c['index']:02d} {c['window_id']} ({g['action_class']}/{g['outcome']})...",
              file=sys.stderr)

        is_rerouted = c["window_id"] in reroute.REROUTE_WINDOW_IDS
        call_bounds = None
        if is_rerouted:
            print(f"  [reroute] {c['window_id']} routed through Files API (§G.1) — "
                  "downloading/normalizing/uploading...", file=sys.stderr)
            upload = await upload_cache.ensure(c)
            ingestion_uri, ingestion_path = upload.file_uri, "files_api"
            call_bounds = upload.call_window_bounds(c["window_start_s"], c["window_end_s"])
            print(f"  [reroute] upload_latency_ms={upload.upload_latency_ms:.0f} "
                  f"active_wait_ms={upload.active_wait_ms:.0f} file={upload.file_name} "
                  f"call_bounds(file-relative)={call_bounds}", file=sys.stderr)
        else:
            ingestion_uri, ingestion_path = c["video_url_watch"], "youtube_uri"

        eff_position_analyzer = reroute_position_analyzer if is_rerouted else position_analyzer
        eff_technique_analyzer = reroute_technique_analyzer if is_rerouted else technique_analyzer
        eff_validator_low_analyzer = reroute_validator_low_analyzer if is_rerouted else validator_low_analyzer
        eff_validator_high_analyzer = reroute_validator_high_analyzer if is_rerouted else validator_high_analyzer

        rec = await run_clip(
            position_analyzer=eff_position_analyzer, technique_analyzer=eff_technique_analyzer,
            validator_low_analyzer=eff_validator_low_analyzer, validator_high_analyzer=eff_validator_high_analyzer,
            ledger=ledger, clip=c, ingestion_uri=ingestion_uri, ingestion_path=ingestion_path,
            call_bounds=call_bounds,
        )
        new_records.append(rec)
        print(f"  status={rec['status']} tech={rec['technique_label']} "
              f"low=({rec['val_low_verdict']},{rec['val_low_final_class']}) "
              f"high=({rec['val_high_verdict']},{rec['val_high_final_class']})", file=sys.stderr)

        gc.write_json(predictions_out_path, _merge_by_window_id(existing, new_records))
        ledger.flush()

        # §D.1/§G.2 halt-and-escalate — checked INSIDE the loop, after EVERY
        # clip, so a trip stops the batch IMMEDIATELY (never after already
        # spending the remaining clips' calls). A --limit-clips smoke slice
        # is, by definition, not a statistically meaningful sample of the
        # full-batch failure rate and must not trip this (unchanged).
        if args.limit_clips is None:
            failed_clips = {r["window_id"] for r in new_records if r["status"] == "call_failed"}
            if len(failed_clips) > gc.GATE2_FAILURE_HALT_THRESHOLD:
                print(
                    f"[gate2] HALT-AND-ESCALATE tripped after clip {c['index']:02d} {c['window_id']} — "
                    f"{len(failed_clips)} clip(s) failed (threshold: >{gc.GATE2_FAILURE_HALT_THRESHOLD} of "
                    f"{gc.GATE2_SUBSET_N}). STOPPING — no further clips will be attempted this run.",
                    file=sys.stderr,
                )
                raise gc.GateHaltError(
                    f"Gate 2 HALT-AND-ESCALATE: {len(failed_clips)} clip(s) had a call_failed status "
                    f"(threshold: >{gc.GATE2_FAILURE_HALT_THRESHOLD} of {gc.GATE2_SUBSET_N} — per Bezos "
                    "condition 5's conservative reading, ANY Gate-2 failure trips this). "
                    f"Failed clips: {sorted(failed_clips)}. Halted after clip index {c['index']} "
                    f"({c['window_id']}) — {len(clips) - (c['index'] + 1)} clip(s) never attempted."
                )

    total_cost = ledger.total_cost_usd()
    print(f"[gate2] done. {len(clips)} clip(s) x {CALLS_PER_CLIP} calls = "
          f"{len(clips) * CALLS_PER_CLIP} call(s) attempted this run. "
          f"Ledger total (all gates, cumulative): {ledger.total_calls()} call(s), "
          f"${total_cost:.4f} (real, from usage_metadata).", file=sys.stderr)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="DANGER: spends real Gemini quota. Default is dry-run.")
    parser.add_argument(
        "--limit-clips", type=int, default=None,
        help="Restrict to the first N manifest clips (index order) — SMOKE TEST knob. "
             "Omit for the full 16-clip batch (founder-gated).",
    )
    args = parser.parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
