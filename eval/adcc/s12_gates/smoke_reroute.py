"""Standalone smoke test for the §G.1 Files API reroute lane ONLY (CEO
instruction, 2026-07-26: "cheapest slice: download+upload+poll ACTIVE for
ONE of the 3 windows, run its Arm-T position call or Gate-2
position/technique — keep it under ~5 Gemini calls").

Exercises the REAL production code path (`files_api_reroute.
RerouteUploadCache`/`prepare_files_api_upload`, `gate_common.timed_axis_call`)
against ONE of the 3 rerouted windows — `66afe5ab1c62a1db7a53b80b__occ0`
(Gate 2's forced-unsuccessful `submission_arm_lock` clip, the cheapest of
the 3: shortest window, already proven to 400 on native YouTube ingestion
during the original smoke test) — running just its `position` + `technique`
calls (2 Gemini calls) against the uploaded Files API URI, where the
YouTube-URI path previously 400'd unconditionally.

This is a STANDALONE script, not a `--limit-clips`/`--limit-windows` mode of
`run_gate1.py`/`run_gate2.py` — those drivers always run a window/clip's
FULL call set (4 calls each) once reroute is engaged; adding a partial-call
smoke flag to the production drivers would be scope creep on code that will
run the real full batch. This script duplicates only the minimal call
sequence (2 of `run_gate2.run_clip`'s 4 calls), reusing every underlying
primitive (`RerouteUploadCache`, `timed_axis_call`, `highlight_axes`
prompt/schema builders) byte-for-byte from the real drivers — never a
second, parallel implementation of the reroute/call mechanics themselves.

Run from the repo root:

    .venv/bin/python -m eval.adcc.s12_gates.smoke_reroute
"""
from __future__ import annotations

import asyncio
import sys

from eval.adcc.s12_gates import files_api_reroute as reroute
from eval.adcc.s12_gates import gate_common as gc
from service.pipelines import highlight_axes

SMOKE_WINDOW_ID = "66afe5ab1c62a1db7a53b80b__occ0"


async def _amain() -> None:
    manifest = gc.read_json(gc.GATE2_MANIFEST_PATH)
    clip = next((c for c in manifest["clips"] if c["window_id"] == SMOKE_WINDOW_ID), None)
    if clip is None:
        raise RuntimeError(f"smoke_reroute: {SMOKE_WINDOW_ID!r} not found in gate2_subset_manifest.json")
    assert SMOKE_WINDOW_ID in reroute.REROUTE_WINDOW_IDS, "smoke target must be a §G.1 rerouted window"

    print(f"[smoke_reroute] target: {SMOKE_WINDOW_ID} "
          f"(video={clip['video_url_watch']}, window=[{clip['window_start_s']}, {clip['window_end_s']}])",
          file=sys.stderr)

    cfg = gc.load_service_config()
    client = gc.build_gemini_client(cfg)
    retry_config = gc.build_retry_config(cfg)
    ledger = gc.CallLedger()

    upload_cache = reroute.RerouteUploadCache(client, ledger, gate="gate2_smoke")

    print("[smoke_reroute] downloading section + normalizing + uploading + polling ACTIVE...", file=sys.stderr)
    t_total0 = asyncio.get_event_loop().time()
    upload = await upload_cache.ensure(clip)
    t_total_ms = (asyncio.get_event_loop().time() - t_total0) * 1000.0

    print(f"[smoke_reroute] REROUTE PIPELINE RESULT:", file=sys.stderr)
    print(f"  local source:      {upload.local_source_path} (sha256={upload.source_sha256[:16]}...)", file=sys.stderr)
    print(f"  local normalized:  {upload.local_normalized_path} (sha256={upload.normalized_sha256[:16]}...)",
          file=sys.stderr)
    print(f"  yt-dlp format:     {upload.ytdlp_format}", file=sys.stderr)
    print(f"  yt-dlp section:    {upload.ytdlp_section_spec}", file=sys.stderr)
    print(f"  download_latency_ms: {upload.download_latency_ms:.0f}", file=sys.stderr)
    print(f"  normalize_latency_ms: {upload.normalize_latency_ms:.0f}", file=sys.stderr)
    print(f"  UPLOAD_LATENCY_MS:  {upload.upload_latency_ms:.0f}", file=sys.stderr)
    print(f"  ACTIVE_WAIT_MS (PROCESSING->ACTIVE): {upload.active_wait_ms:.0f}", file=sys.stderr)
    print(f"  files_api uri:     {upload.file_uri}", file=sys.stderr)
    print(f"  files_api name:    {upload.file_name}", file=sys.stderr)
    print(f"  total pipeline wall time: {t_total_ms:.0f}ms", file=sys.stderr)

    position_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=highlight_axes.POSITION_SYSTEM_PROMPT, retry_config=retry_config,
    )
    technique_analyzer = reroute.build_files_api_time_analyzer(
        client, system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT, retry_config=retry_config,
    )

    # §G.1 reroute lane bug #2 (files_api_reroute.py's "live bug" writeup):
    # the uploaded file's OWN clock starts at 0:00 — video_metadata offsets
    # MUST be file-relative, never the window's absolute match-clock seconds.
    call_start, call_end = upload.call_window_bounds(clip["window_start_s"], clip["window_end_s"])
    print(f"[smoke_reroute] call_bounds (file-relative): [{call_start}, {call_end}] "
          f"(absolute window was [{clip['window_start_s']}, {clip['window_end_s']}], "
          f"upload.sec_start_s={upload.sec_start_s})", file=sys.stderr)

    print("[smoke_reroute] calling position axis against the Files API URI (where YouTube-URI 400'd)...",
          file=sys.stderr)
    pos_data, pos_err, pos_record = await gc.timed_axis_call(
        ledger=ledger, gate="gate2_smoke", window_id=SMOKE_WINDOW_ID, match_id=clip["match_id"],
        arm="gate2_smoke", call_name="position", analyzer=position_analyzer, youtube_url=upload.file_uri,
        window_start=call_start, window_end=call_end,
        response_schema=highlight_axes.build_position_schema(),
        prompt_text=highlight_axes.build_position_prompt(None),
        fps=gc.FPS, media_resolution=None, ingestion_path="files_api",
    )
    print(f"  position: status={pos_record['status']} parsed={pos_data}", file=sys.stderr)

    print("[smoke_reroute] calling technique axis against the Files API URI...", file=sys.stderr)
    tech_data, tech_err, tech_record = await gc.timed_axis_call(
        ledger=ledger, gate="gate2_smoke", window_id=SMOKE_WINDOW_ID, match_id=clip["match_id"],
        arm="gate2_smoke", call_name="technique", analyzer=technique_analyzer, youtube_url=upload.file_uri,
        window_start=call_start, window_end=call_end,
        response_schema=highlight_axes.build_technique_schema(),
        prompt_text=highlight_axes.build_technique_prompt(None),
        fps=gc.FPS, media_resolution=None, ingestion_path="files_api",
    )
    print(f"  technique: status={tech_record['status']} parsed={tech_data}", file=sys.stderr)

    ledger.flush()

    gold = clip["gold_event"]
    print("", file=sys.stderr)
    print(f"[smoke_reroute] gold: action_class={gold['action_class']} outcome={gold['outcome']}", file=sys.stderr)
    print(f"[smoke_reroute] DONE. 2 Gemini calls attempted via files_api ingestion_path "
          f"(pos={pos_record['status']}, tech={tech_record['status']}). "
          f"Ledger entries this run: upload + 2 calls = {len([e for e in ledger.entries if e['window_id'] == SMOKE_WINDOW_ID])}.",
          file=sys.stderr)

    print("[smoke_reroute] cleaning up (best-effort delete of uploaded file)...", file=sys.stderr)
    await upload_cache.cleanup()


def main(argv=None) -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
