"""Gate 4 driver — Attribution accuracy (actor axis + self-referential
reference-image crops), `GATES_3_4_DESIGN.md` §I. Unchanged by the
`VERDICTS.md` amendment (Gate 4 was never coupled to Arm T's fate).

**MANDATORY cross-worktree invocation — this worktree's own
`service.pipelines.highlight_axes` has NO actor axis at all** (verified:
only position/technique/validator). The actor axis (`build_actor_schema`/
`build_actor_prompt`, `ACTOR_SYSTEM_PROMPT`) only exists in
`whole-video-analysis-wt-s12`. Run EXACTLY as:

    cd /Users/stanliu/bjj-proj/whole-video-analysis-wt-s12
    PYTHONPATH=/Users/stanliu/bjj-proj/whole-video-analysis-wt-vlmstudio \\
      .venv/bin/python -m eval.adcc.s12_gates.run_gate4 [--execute]

This physically resolves `service.pipelines.*` (incl. the actor axis)
against wt-s12 (cwd precedence in Python's import search) and
`eval.adcc.s12_gates.*` (this script itself, `gate_common`,
`files_api_reroute`) against wt-vlmstudio (via `PYTHONPATH`) — verified by a
direct smoke import before any Gate-4 code was written. The script file
lives under wt-vlmstudio's `eval/adcc/s12_gates/` per the CEO's explicit
"scripts still live in s12_gates" instruction; only the INTERPRETER and CWD
are wt-s12's. A runtime guard below refuses to proceed if invoked any other
way (never silently falls back to the wrong `highlight_axes`).

Gate 4 does NOT use the Files-API reroute lane at all — every one of its 8
matches' native videos is analyzed via a plain YouTube watch URL (none of
the 8 is the `pwGbW5GZgfc` blocker). A short LOCAL download is still needed,
but ONLY to extract one still frame per match for the reference crops (see
`prepare_reference_crops`'s docstring for the one flagged deviation from the
design's literal download-span wording).

Run from the repo root (module invocation, see above for the exact command):

    ... run_gate4                # dry-run plan only, zero calls
    ... run_gate4 --execute      # spends real Gemini quota (46 calls)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Runtime guard — refuse to proceed against the wrong `service.pipelines`
# (see module docstring). Checked BEFORE any other import that could shadow
# a clear error with a confusing one.
# --------------------------------------------------------------------------- #
try:
    from service.pipelines.highlight_axes import build_actor_schema  # noqa: F401
except ImportError as e:  # pragma: no cover — exercised only by a wrong invocation
    raise RuntimeError(
        "run_gate4: service.pipelines.highlight_axes has no build_actor_schema — this means "
        "you are NOT running from whole-video-analysis-wt-s12's interpreter/cwd. See this "
        "module's docstring for the mandatory cross-worktree invocation command."
    ) from e

from eval.adcc.s12_gates import files_api_reroute as reroute
from eval.adcc.s12_gates import gate_common as gc
from service.pipelines import highlight_axes
from service.pipelines.executors import _call_axis_json

# --------------------------------------------------------------------------- #
# §I.2 — frozen n=8 match draw (self-verified against the live gold
# snapshot before this driver was written — all 8 match_id -> event-count
# pairs and the seeded random.Random(20260726).sample(eligible, 8) draw
# reproduce the design's own table exactly).
# --------------------------------------------------------------------------- #
GATE4_MATCHES: list[tuple[str, str]] = [
    ("650d20c745120c8433e9d018", "ATkSimsKwtg"),
    ("650d20c745120c8433e9d027", "n_dkNnfjx4s"),
    ("65396bbbb7741492b97777fc", "KoPxgjLbJmA"),
    ("6567972aa4a13c99c4fe36ca", "oHb7bR0epUI"),
    ("65dbe62da2434704339da256", "BvAQDVP29E0"),
    ("662ae0ce532ae8f232657f96", "srZxKZ70eW0"),
    ("664b79b5532ae8f2326a66b7", "vCyb33erhsc"),
    ("6665f04d4828f704f1b3afd8", "dsldlavgLz4"),
]

GATE4_REF_CACHE_DIR = reroute.VLM_EVAL_HOME / "media_cache" / "s12_gates_gate4_refs"
GATE4_PREDICTIONS_PATH = gc.S12_GATES_DIR / "gate4_predictions.json"
GATE4_FAILURE_HALT_THRESHOLD = 2  # inferred, conservative — see handoff report; design states no explicit rule


class Gate4HaltError(RuntimeError):
    pass


def build_gate4_windows() -> list[dict]:
    """§I.2's window construction: per match, the first 3 (2 for the last
    row) hard-track events sorted by `t_s` ascending, same `[t_s-5, t_s+4]`
    window cut as §A.4. Returns a flat list of 23 window dicts."""
    events = gc.load_hard_track_events()
    by_match: dict[str, list[dict]] = {}
    for e in events:
        by_match.setdefault(e["match_id"], []).append(e)

    windows: list[dict] = []
    for match_id, video_id in GATE4_MATCHES:
        match_events = sorted(by_match.get(match_id, []), key=lambda e: e["t_s"])
        if len(match_events) < 2:
            raise ValueError(f"build_gate4_windows: {match_id} has only {len(match_events)} hard events (need >=2)")
        selected = match_events[:3]
        video_url_watch = f"https://www.youtube.com/watch?v={video_id}"
        for i, e in enumerate(selected):
            window_start_s, window_end_s, clamp_warning = gc.compute_window(float(e["t_s"]))
            windows.append({
                "window_id": f"{match_id}__g4_{i}", "match_id": match_id, "video_url_watch": video_url_watch,
                "gold_event": {
                    "t_s": e["t_s"], "action_class": e["action_class"], "outcome": e["outcome"],
                    "position_bucket": e["position_bucket"], "track": e["track"], "scoreable": e["scoreable"],
                },
                "window_start_s": window_start_s, "window_end_s": window_end_s, "clamp_warning": clamp_warning,
            })
    return windows


def reference_offset_for_match(match_windows: list[dict]) -> float:
    """§I.3's `reference_offset_s` formula, applied to a match's own
    earliest selected window."""
    first_window_start_s = min(w["window_start_s"] for w in match_windows)
    return 3.0 if first_window_start_s >= 8.0 else first_window_start_s / 2.0


async def prepare_reference_crops(video_url_watch: str, ref_offset_s: float, out_dir: Path, match_id: str) -> tuple[bytes, bytes, float]:
    """§I.3 reference-crop production, with ONE FLAGGED DEVIATION from the
    design's literal text: the design says download "a slightly wider local
    section covering `[0, last_selected_window_end_s]` once per match" — but
    Gate 4's actual highlight-window CALLS go straight to Gemini via native
    YouTube `file_uri` (no Files-API reroute, no local file needed for
    those at all); the ONLY thing that ever needs a local file is this ONE
    still frame at `reference_offset_s` (always <=3.0s into the video, per
    the formula). Downloading `[0, last_selected_window_end_s]` would mean
    downloading up to ~2.4 HOURS of video (match `n_dkNnfjx4s`, last window
    ends at t~8807s) to extract one frame near t=0-3s — no correctness
    benefit, only cost/latency. This function downloads only
    `[0, ref_offset_s+2]` (a few seconds) instead — the frame's actual pixel
    content at `ref_offset_s` is byte-identical either way. Flagged per the
    "implement the conservative reading, flag it, continue" instruction.

    Extracts LEFT/RIGHT half-frame JPEGs via ffmpeg's `crop` filter directly
    (no H.264 normalization needed — a single-frame JPEG extract works on
    any codec ffmpeg can decode; `media_resolver.build_mp4_derivative` is
    for producing a Files-API-uploadable video, irrelevant here).

    Returns ``(left_jpeg_bytes, right_jpeg_bytes, download_latency_ms)``.

    **[RE-RUN 2026-07-26, GATES_RERUN_DESIGN.md §1] reuse-as-is fix.** The
    design is explicit: Gate 4's reference-image crops "contain no model
    output whatsoever, and are reused as-is (real time/cost saving: no need
    to re-download or re-crop anything for Gate 4)." This function now
    checks for already-extracted `{match_id}.ref_left.jpg`/`.ref_right.jpg`
    FIRST and returns them directly (zero download, zero ffmpeg call) if
    both exist — only falling through to a fresh download+extract for a
    match that has never been processed before. This is a real, necessary
    fix (found live, 2026-07-26): the driver originally re-downloaded on
    every invocation regardless, which is both a design violation (§1) and
    what caused a second real yt-dlp/ffmpeg transient failure to be hit
    unnecessarily on a match whose crops were already sitting on disk from
    the v1 flash-lite run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    left_path = out_dir / f"{match_id}.ref_left.jpg"
    right_path = out_dir / f"{match_id}.ref_right.jpg"
    if left_path.exists() and right_path.exists():
        return left_path.read_bytes(), right_path.read_bytes(), 0.0

    sec_end = ref_offset_s + 2.0
    section_spec = f"*0.000-{sec_end:.3f}"
    raw_template = str(out_dir / f"{match_id}.ref_raw.%(ext)s")
    for stale in out_dir.glob(f"{match_id}.ref_raw.*"):
        stale.unlink()

    import time
    t0 = time.perf_counter()
    cmd = [
        reroute.YTDLP_BIN, "--download-sections", section_spec, "-f", reroute.YTDLP_FORMAT,
        "-o", raw_template, "--no-playlist", "--quiet", "--no-warnings", video_url_watch,
    ]
    await reroute._run_subprocess(cmd, op_name=f"yt-dlp reference-section download ({match_id})")
    download_latency_ms = (time.perf_counter() - t0) * 1000.0

    matches = sorted(out_dir.glob(f"{match_id}.ref_raw.*"))
    if not matches:
        raise RuntimeError(f"prepare_reference_crops: no output file for {match_id} (template {raw_template})")
    raw_path = matches[0]

    for out_path, crop_x in ((left_path, "0"), (right_path, "iw/2")):
        cmd = [
            reroute.FFMPEG_BIN, "-y", "-ss", str(ref_offset_s), "-i", str(raw_path),
            "-frames:v", "1", "-vf", f"crop=iw/2:ih:{crop_x}:0", str(out_path),
        ]
        await reroute._run_subprocess(cmd, op_name=f"ffmpeg reference crop ({match_id}, {out_path.name})")

    return left_path.read_bytes(), right_path.read_bytes(), download_latency_ms


async def timed_actor_call(
    *, ledger: gc.CallLedger, window_id: str, match_id: str, call_name: str, analyzer,
    youtube_url: str, window_start: float, window_end: float, response_schema, prompt_text: str,
    extra_parts: Optional[list],
) -> tuple[Optional[dict], Optional[str], dict]:
    """Gate-4-local call wrapper — deliberately NOT `gate_common.
    timed_axis_call` (that function's `_call_axis_json` binding does not
    accept `extra_parts` in wt-vlmstudio's own `service.pipelines.executors`;
    modifying the shared module to add a cross-worktree-only kwarg would risk
    Gates 1/2/3, which never need it). This wrapper calls THIS process's
    bound `_call_axis_json` (wt-s12's version, confirmed to support
    `extra_parts`/`mime_type` via the mandatory cross-worktree invocation)
    directly, reusing `gate_common.LedgerEntry`/`CallLedger`/
    `extract_http_status` for identical ledger bookkeeping to every other
    gate."""
    import time
    t0 = time.perf_counter()
    data, err = await _call_axis_json(
        analyzer, youtube_url, window_start, window_end,
        response_schema=response_schema, prompt_text=prompt_text,
        fps=gc.FPS, media_resolution=None, extra_parts=extra_parts,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    usage = getattr(analyzer, "last_usage_metadata", None)
    status = "ok" if err is None else "call_failed"
    cost_usd = usage["total_token_count"] * (0.60 / 1_000_000) if usage and usage.get("total_token_count") else None

    # [RE-RUN 2026-07-26] live gap found — this local wrapper predates
    # thinking_level tracking and was silently omitting it from every Gate 4
    # ledger row (unlike gc.timed_axis_call, which all other gates use).
    # Fixed the same way: read the REAL level off the analyzer's own
    # thinking_config, never a second independently-passed value.
    thinking_cfg = getattr(analyzer, "thinking_config", None)
    thinking_level_raw = getattr(thinking_cfg, "thinking_level", None)
    thinking_level = (
        thinking_level_raw.name.lower() if hasattr(thinking_level_raw, "name") else thinking_level_raw
    )

    ledger.record(gc.LedgerEntry(
        gate="gate4", window_id=window_id, match_id=match_id, arm="actor", call_name=call_name,
        model_id=gc.MODEL_ID, status=status, error=err, http_status=gc.extract_http_status(err),
        latency_ms=latency_ms, usage_metadata=usage, cost_usd=cost_usd,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), ingestion_path="youtube_uri",
        thinking_level=thinking_level,
    ))
    call_record = {
        "call_name": call_name, "status": status, "error": err,
        "raw_response_text": getattr(analyzer, "last_raw_response_text", None),
        "parsed": data, "usage_metadata": usage, "latency_ms": round(latency_ms, 1),
    }
    return data, err, call_record


def _print_dry_run_plan(windows: list[dict]) -> None:
    gc.assert_thinking_high_in_generate_config()  # [RE-RUN] fail loud before any plan is even printed
    print(f"GATE 4 DRY-RUN PLAN — {len(windows)} window(s), {len(windows) * 2} Gemini call(s) "
          f"(WITH-refs + WITHOUT-refs per window), model={gc.MODEL_ID}, fps={gc.FPS}, "
          f"8 reference-crop extractions (2 halves each, zero Gemini calls).", file=sys.stderr)
    for w in windows:
        print(f"  {w['window_id']}  action_class={w['gold_event']['action_class']}  "
              f"window=[{w['window_start_s']}, {w['window_end_s']}]", file=sys.stderr)
    print("Zero Gemini calls made. Pass --execute to spend real quota.", file=sys.stderr)


async def _amain(args: argparse.Namespace) -> None:
    windows = build_gate4_windows()
    if not args.execute:
        _print_dry_run_plan(windows)
        return

    gc.assert_thinking_high_in_generate_config()  # [RE-RUN] belt-and-suspenders — re-verified right before real spend

    print("WARNING: --execute fires Gate 4's real 46-call spend. Ctrl-C now if not authorized. "
          "Proceeding in 3 real seconds...", file=sys.stderr)
    await asyncio.sleep(3)

    cfg = gc.load_service_config()
    client = gc.build_gemini_client(cfg)
    retry_config = gc.build_retry_config(cfg)
    ledger = gc.CallLedger()

    with_refs_analyzer = gc.build_time_analyzer(
        client, system_instruction=highlight_axes.ACTOR_SYSTEM_PROMPT, retry_config=retry_config,
    )
    without_refs_analyzer = gc.build_time_analyzer(
        client, system_instruction=highlight_axes.ACTOR_SYSTEM_PROMPT, retry_config=retry_config,
    )

    windows_by_match: dict[str, list[dict]] = {}
    for w in windows:
        windows_by_match.setdefault(w["match_id"], []).append(w)

    # Resume-on-rerun (added after a real transient yt-dlp/network failure mid-batch,
    # 2026-07-26 — manually confirmed the SAME download succeeds on retry; without this,
    # re-invoking --execute would blindly re-spend every already-successful window's 2
    # calls from scratch, blowing well past the 46-call design budget). A window already
    # recorded with status="ok" is skipped entirely (crops not re-downloaded, no new
    # Gemini calls) and its EXISTING record is carried forward unchanged.
    # [RE-RUN 2026-07-26] versioned output path — never overwrite the flash-lite v1 canonical.
    predictions_out_path = gc.versioned_path(GATE4_PREDICTIONS_PATH)
    existing_predictions: list[dict] = (
        gc.read_json(predictions_out_path) if predictions_out_path.exists() else []
    )
    already_ok: dict[str, dict] = {p["window_id"]: p for p in existing_predictions if p["status"] == "ok"}
    if already_ok:
        print(f"[gate4] resume: {len(already_ok)} window(s) already status=ok in "
              f"{predictions_out_path.name} — skipping their crops/calls, carrying records forward.",
              file=sys.stderr)

    predictions: list[dict] = []
    failed_windows: set[str] = set()

    for match_id, video_id in GATE4_MATCHES:
        match_windows = windows_by_match[match_id]
        match_window_ids = {w["window_id"] for w in match_windows}
        if match_window_ids <= already_ok.keys():
            print(f"[gate4] match {match_id} — all {len(match_window_ids)} window(s) already status=ok, "
                  "skipping crop download entirely.", file=sys.stderr)
            for wid in sorted(match_window_ids):
                predictions.append(already_ok[wid])
            gc.write_json(predictions_out_path, predictions)
            continue

        ref_offset_s = reference_offset_for_match(match_windows)
        video_url_watch = match_windows[0]["video_url_watch"]
        print(f"[gate4] match {match_id} — preparing reference crops (ref_offset_s={ref_offset_s})...",
              file=sys.stderr)
        left_bytes, right_bytes, dl_ms = await prepare_reference_crops(
            video_url_watch, ref_offset_s, GATE4_REF_CACHE_DIR, match_id,
        )
        print(f"  reference crops ready ({len(left_bytes)}B left, {len(right_bytes)}B right, "
              f"download_latency_ms={dl_ms:.0f})", file=sys.stderr)

        player_refs = [
            {"player_id": "ref_left", "player_name": "ref_left", "image_bytes": left_bytes, "mime_type": "image/jpeg"},
            {"player_id": "ref_right", "player_name": "ref_right", "image_bytes": right_bytes, "mime_type": "image/jpeg"},
        ]
        from google.genai import types
        extra_parts = [
            types.Part(inline_data=types.Blob(data=ref["image_bytes"], mime_type=ref["mime_type"]))
            for ref in player_refs
        ]
        actor_schema_with_refs = highlight_axes.build_actor_schema(["ref_left", "ref_right"])
        actor_schema_without_refs = highlight_axes.build_actor_schema([])
        prompt_with_refs = highlight_axes.build_actor_prompt(None, player_refs)
        prompt_without_refs = highlight_axes.build_actor_prompt(None, [])

        for w in match_windows:
            wid = w["window_id"]
            if wid in already_ok:
                print(f"[gate4] window {wid} — already status=ok, skipping.", file=sys.stderr)
                predictions.append(already_ok[wid])
                gc.write_json(predictions_out_path, predictions)
                continue
            print(f"[gate4] window {wid} ({w['gold_event']['action_class']}/{w['gold_event']['outcome']})...",
                  file=sys.stderr)

            with_data, with_err, with_record = await timed_actor_call(
                ledger=ledger, window_id=wid, match_id=match_id, call_name="actor_with_refs",
                analyzer=with_refs_analyzer, youtube_url=w["video_url_watch"],
                window_start=w["window_start_s"], window_end=w["window_end_s"],
                response_schema=actor_schema_with_refs, prompt_text=prompt_with_refs, extra_parts=extra_parts,
            )
            without_data, without_err, without_record = await timed_actor_call(
                ledger=ledger, window_id=wid, match_id=match_id, call_name="actor_without_refs",
                analyzer=without_refs_analyzer, youtube_url=w["video_url_watch"],
                window_start=w["window_start_s"], window_end=w["window_end_s"],
                response_schema=actor_schema_without_refs, prompt_text=prompt_without_refs, extra_parts=None,
            )
            print(f"  with_refs: {with_record['status']} {with_data}", file=sys.stderr)
            print(f"  without_refs: {without_record['status']} {without_data}", file=sys.stderr)

            if with_err is not None or without_err is not None:
                failed_windows.add(wid)

            predictions.append({
                "window_id": wid, "match_id": match_id, "gold_event": w["gold_event"],
                "video_url_watch": w["video_url_watch"],
                "window_start_s": w["window_start_s"], "window_end_s": w["window_end_s"],
                "ref_offset_s": ref_offset_s,
                "with_refs": {**with_record, "actor": with_data.get("actor") if with_data else None,
                              "identity_uncertain": with_data.get("identity_uncertain") if with_data else None},
                "without_refs": {**without_record, "actor": without_data.get("actor") if without_data else None},
                "status": "call_failed" if wid in failed_windows else "ok",
            })
            gc.write_json(predictions_out_path, predictions)
            ledger.flush()

            if len(failed_windows) > GATE4_FAILURE_HALT_THRESHOLD:
                print(f"[gate4] HALT-AND-ESCALATE tripped after window {wid} — {len(failed_windows)} window(s) "
                      f"failed (threshold: >{GATE4_FAILURE_HALT_THRESHOLD} of {len(windows)}). STOPPING.",
                      file=sys.stderr)
                raise Gate4HaltError(
                    f"Gate 4 HALT-AND-ESCALATE: {len(failed_windows)} window(s) had a call_failed axis "
                    f"(threshold: >{GATE4_FAILURE_HALT_THRESHOLD} of {len(windows)}). "
                    f"Failed windows: {sorted(failed_windows)}"
                )

    total_cost = ledger.total_cost_usd()
    print(f"[gate4] done. {len(windows)} window(s) x 2 calls = {len(windows) * 2} call(s) attempted. "
          f"Ledger total (cumulative): {ledger.total_calls()} call(s), ${total_cost:.4f}.", file=sys.stderr)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="DANGER: spends real Gemini quota. Default is dry-run.")
    args = parser.parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
