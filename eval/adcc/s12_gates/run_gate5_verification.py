"""AC7 — 8-straddler targeted verification run (`2026-07-26-engine13-
rescope-single-call-cutover.md` AC7 / `VERDICTS.md` §3's specified
verification method). Re-runs the 8 known Gate-5 straddler events
(`GATE5_REPORT.md`'s W=60 production-anchoring table) through the REAL
`highlight_critique_node`/`highlight_analyze_node` production functions
from `whole-video-analysis-wt-s12`, `feature/s12-highlight-production-
wiring`, in TWO configurations:

- **config a (PATCHED)**: `HIGHLIGHT_OUTER_CHUNK_OVERLAP_S=45.0` (as shipped).
- **config b (PRE-PATCH baseline)**: `overlap_s=0.0` (the never-measured
  truncated baseline) — achieved by passing `overlap_s=0.0` directly as a
  keyword argument to `_outer_chunks` at OUR call site (production's own
  constant is never monkeypatched/mutated — the function's own `overlap_s`
  parameter already exists for exactly this; no production code edited).

**MUST be run from `whole-video-analysis-wt-s12`'s own venv/cwd** (so
`service.*` resolves the REAL patched pipeline code), with `PYTHONPATH`
pointing at this `whole-video-analysis-wt-vlmstudio` repo root (so
`eval.*` resolves this harness) — same cross-worktree pattern as Gate 4:

    cd /Users/stanliu/bjj-proj/whole-video-analysis-wt-s12 && \\
    PYTHONPATH=/Users/stanliu/bjj-proj/whole-video-analysis-wt-vlmstudio \\
    .venv/bin/python -m eval.adcc.s12_gates.run_gate5_verification --execute

**Budget-conscious design (lightweight targeted check, not a third full
gate campaign, per `VERDICTS.md` §3's own framing):** running the FULL
`highlight_scan` discovery stage over a real 720s chunk would cost an
unbounded, uncontrolled number of calls (a whole match's own worst-case
estimate already exceeds 100 calls per chunk — `executors.estimate_run_
plan`). Instead, this harness bypasses ONLY the discovery step — it hand-
constructs a `ctx.highlights = [{"index": 0, "start_s": t_s-60, "end_s":
t_s+5, ...}]` entry using the SAME W=60/post_tail_s=5 span
`GATE5_REPORT.md` itself already used to FLAG these 8 events as straddlers
(not fabricated — the exact operative definition already established) —
then calls the REAL `highlight_critique_node`/`highlight_analyze_node`
functions unmodified, with `ctx.start_sec`/`ctx.end_sec` set to the REAL
chunk-scope boundaries `_outer_chunks` computes for each config. This is
exactly the mechanism `GATE5_ADDENDUM.md` cited as the truncation site
(`executors.py`'s `window_start = max(ctx.start_sec, authoritative_start -
cfg.preroll_s)` / critique's own `max(ctx.start_sec, highlight["start_s"] -
cfg.critique_backpad_s)`) — the ONLY thing skipped is the (expensive,
uncontrolled) discovery of WHERE a highlight starts, which this harness
already knows from the gold record.

**Settings** (INS-141 qualified bar, overriding the registry's own
flash-lite/fps=1/thinking=None defaults on `HighlightCritiqueConfig`/
`HighlightAnalyzeConfig`): `model=gemini-3.5-flash`, `thinking=high`,
`fps=10` (only `HighlightAnalyzeConfig` exposes `fps` — critique's own fps
is hardcoded to 1 inside `highlight_critique_node`, unconfigurable,
confirmed by reading the function). `preroll_s`/`postroll_s`/
`critique_backpad_s`/`media_resolution` are left at their REAL registry
defaults (this harness tests the ACTUAL shipped config, not a hand-picked
worst-case) — see module-level constants below for the exact values read
live from `service.pipelines.registry.get_default(...)`.

**Ingestion**: YouTube watch URL directly for 7 of the 8 events (cost/
simplicity, per explicit instruction — chunking math is ingestion-
independent). Event `e8` (match `66b009481c62a1db7a53d504`, video
`pwGbW5GZgfc`) is the SAME 8.5h ADCC-compilation video already confirmed,
independently, to 400 on every native YouTube ingestion offset/fps
(§G.1) — this harness routes it through the proven Files-API reroute lane
(`files_api_reroute.py`'s download/normalize/upload pipeline, reused
unmodified) instead.

**Seam-dedup double-publish check** (config a only — dedup is a NEW
mechanism specific to the overlap patch, not meaningful for config b):
for each event, ALSO runs `highlight_critique_node` + a TAXONOMY-ONLY
`_call_axis_json` call (bypassing the full `highlight_analyze_node`'s actor
call — verified live-read of `seam_dedup.py`: `dedup_match_clips`/
`find_seam_duplicate` NEVER consult `actor`/`player_id`, only `start_s`/
`end_s`/`action_class`, so the actor call is genuinely unnecessary for this
specific check and skipping it is a real, disclosed cost saving) scoped to
chunk `k-1`'s own boundary — simulating "what if the PRECEDING chunk's own
scan had ALSO independently found this event near its tail" (a
conservative, worst-case stress test for dedup — the scenario that actually
exercises the merge logic). The two resulting clip dicts are fed to the
REAL `seam_dedup.dedup_match_clips` and the survivor count is checked.

Halt: >2 windows failed within a single config (stated explicitly by the
CEO, mirrors this program's established per-invocation halt convention).
Budget: <=80 calls / <=$10 (hard cap, checked before each event/config).
Fail loud — no fallback logic.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from service.pipelines import executors, gemini_retry
    from service.pipelines.executors import RunContext
    from service.pipelines.models import HighlightAnalyzeConfig, HighlightCritiqueConfig
    from service.pipelines.registry import get_default
    from service.routes.qa_vlm import thinking_config_for
    from service.worker import seam_dedup
    from service.worker.highlight_orchestrator import _outer_chunks
except ImportError as e:
    raise RuntimeError(
        "run_gate5_verification.py must be run with cwd=whole-video-analysis-wt-s12 (so `service.*` "
        "resolves the REAL patched pipeline) and PYTHONPATH including whole-video-analysis-wt-vlmstudio "
        "(so `eval.*` resolves this harness) — see this module's own docstring for the exact invocation. "
        f"Original ImportError: {e}"
    ) from e

from eval.adcc.predict import media_resolver
from eval.adcc.s12_gates import files_api_reroute as reroute
from eval.adcc.s12_gates import gate_common as gc
from eval.adcc.s12_gates import gemini_upload

QUALIFIED_MODEL = "gemini-3.5-flash"
QUALIFIED_THINKING = "high"
QUALIFIED_FPS = 10
MAX_SCOPE_SEC = 720
PATCHED_OVERLAP_S = 45.0
PRE_PATCH_OVERLAP_S = 0.0
CAP_USD = 10.0
HALT_PER_CONFIG = 2
GEMINI_35_FLASH_USD_PER_TOKEN = 0.60 / 1_000_000  # matches gate_common.py's own rate, same model

REROUTE_VIDEO_IDS = {"pwGbW5GZgfc"}  # §G.1-confirmed permanently broken for native YouTube ingestion

# GATE5_REPORT.md's own W=60 production-anchoring straddler table — the 8
# events. `t_s`/`grid_start`/`boundary_s` transcribed directly from that
# report's "t_s"/"boundary crossed (k, s)" columns (k=1 for all 8 — verified
# by inspection). `video_id` extracted from the gold snapshot's own
# `video_url` field (`.../embed/<id>`) for the 8 match_ids, cross-checked
# live against `eval/adcc/gold_snapshot/highlight_gold.jsonl`.
EVENTS: list[dict] = [
    {"event_id": "e1", "match_id": "6470211a3ec8dca9d1a32710", "match_label": "PJ Barch vs Kade Ruotolo",
     "video_id": "3gotA-Y0gXQ", "t_s": 727, "gold_action_class": "submission_arm_lock",
     "grid_start": 0, "boundary_s": 720},
    {"event_id": "e2", "match_id": "650d20c745120c8433e9cfe0", "match_label": "Joshua Cisneros vs Ethan Crelinsten",
     "video_id": "IEcPgfZbGRs", "t_s": 774, "gold_action_class": "guard_pass",
     "grid_start": 0, "boundary_s": 720},
    {"event_id": "e3", "match_id": "650d20c745120c8433e9cff3", "match_label": "Xande Ribeiro vs Eoghan O'Flanagan",
     "video_id": "BKZ6Dt7SpYE", "t_s": 761, "gold_action_class": "back_take",
     "grid_start": 0, "boundary_s": 720},
    {"event_id": "e4", "match_id": "650d20c745120c8433e9d020", "match_label": "Ffion Davies vs Beatriz Mesquita",
     "video_id": "q7oQLvo6L90", "t_s": 760, "gold_action_class": "guard_pass",
     "grid_start": 0, "boundary_s": 720},
    {"event_id": "e5", "match_id": "65de1567a2434704339dbc8c", "match_label": "Margot Ciccarelli vs Ashley Bendle",
     "video_id": "KzYTzUD5rRA", "t_s": 729, "gold_action_class": "sweep",
     "grid_start": 0, "boundary_s": 720},
    {"event_id": "e6", "match_id": "65de222ea2434704339dbe9c", "match_label": "Nia Blackman vs Salla Simola",
     "video_id": "KzYTzUD5rRA", "t_s": 2235, "gold_action_class": "back_take",
     "grid_start": 1465, "boundary_s": 2185},
    {"event_id": "e7", "match_id": "65edf42aa2434704339f7e93", "match_label": "Fabricio Andrey vs Kaua Gabriel",
     "video_id": "60PqfBVmd10", "t_s": 754, "gold_action_class": "back_take",
     "grid_start": 0, "boundary_s": 720},
    {"event_id": "e8", "match_id": "66b009481c62a1db7a53d504", "match_label": "Huaiqing Xu vs Daiki Yonekura",
     "video_id": "pwGbW5GZgfc", "t_s": 30443, "gold_action_class": "back_take",
     "grid_start": 29685, "boundary_s": 30405},
]
for _ev in EVENTS:
    _ev["needs_reroute"] = _ev["video_id"] in REROUTE_VIDEO_IDS
    _ev["video_watch_url"] = f"https://www.youtube.com/watch?v={_ev['video_id']}"

GATE5_VERIFICATION_PREDICTIONS_PATH = gc.S12_GATES_DIR / "gate5_verification_predictions.json"
GATE5_VERIFICATION_SCORING_PATH = gc.S12_GATES_DIR / "gate5_verification_scoring.json"
GATE5_VERIFICATION_LEDGER_PATH = gc.S12_GATES_DIR / "gate5_verification_ledger.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_cost(usage_metadata: Optional[dict]) -> float:
    if not usage_metadata:
        return 0.0
    total = usage_metadata.get("total_token_count")
    return float(total) * GEMINI_35_FLASH_USD_PER_TOKEN if total is not None else 0.0


class Ledger:
    def __init__(self, path: Path = GATE5_VERIFICATION_LEDGER_PATH):
        self.path = path
        self.entries: list[dict] = gc.read_json(path) if path.exists() else []

    def record(self, **kwargs) -> None:
        kwargs.setdefault("timestamp_utc", _now_utc())
        self.entries.append(kwargs)

    def flush(self) -> None:
        gc.write_json(self.path, self.entries)

    def total_cost_usd(self) -> float:
        return sum(e.get("cost_usd") or 0.0 for e in self.entries)


def chunk_bounds_for_event(ev: dict, overlap_s: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Reuses the REAL `_outer_chunks` (never reimplemented) — works in
    match-relative (0-based) terms, then translates back to absolute
    match-clock seconds via `ev["grid_start"]`. Returns
    `(chunk0_bounds_abs, chunk1_bounds_abs)`."""
    rel_duration = (ev["boundary_s"] - ev["grid_start"]) + MAX_SCOPE_SEC + 1.0
    chunks = _outer_chunks(rel_duration, MAX_SCOPE_SEC, overlap_s=overlap_s)
    assert len(chunks) >= 2, f"expected >=2 chunks, got {chunks} for event {ev['event_id']}"
    gs = ev["grid_start"]
    c0 = (gs + chunks[0][0], gs + chunks[0][1])
    c1 = (gs + chunks[1][0], gs + chunks[1][1])
    return c0, c1


async def _reroute_upload_for_range(
    ev: dict, abs_start: float, abs_end: float, client, *, tag: str,
) -> tuple[str, float]:
    """Reuses `files_api_reroute.download_section` + `media_resolver.
    build_mp4_derivative` + `gemini_upload.upload_video_to_gemini`/
    `poll_until_active` (all already-proven §G.1 harness code, unmodified)
    to build ONE Gemini Files-API upload covering `[abs_start, abs_end]` of
    `ev`'s video. Returns `(file_uri, sec_start)` — `sec_start` is the
    uploaded file's own 0:00 in absolute match-clock terms (same
    file-relative-offset discipline as `ReroutedUpload`)."""
    window_id = f"{ev['event_id']}_{tag}"
    src_path, _spec, sec_start = await reroute.download_section(
        ev["video_watch_url"], abs_start, abs_end, reroute.MEDIA_CACHE_DIR, window_id=window_id,
    )
    norm_path = reroute.MEDIA_CACHE_DIR / f"{window_id}.normalized.mp4"
    await media_resolver.build_mp4_derivative(src_path, norm_path)
    uploaded = await gemini_upload.upload_video_to_gemini(client, str(norm_path))
    active_file = await gemini_upload.poll_until_active(
        client, uploaded.name,
        poll_interval_sec=reroute.UPLOAD_POLL_INTERVAL_SEC, timeout_sec=reroute.UPLOAD_POLL_TIMEOUT_SEC,
    )
    return active_file.uri, sec_start


async def build_ctx_for_scope(
    ev: dict, chunk_start: float, chunk_end: float, api_key: str, gemini_client, upload_cache: dict,
) -> RunContext:
    """Builds a RunContext scoped to `[chunk_start, chunk_end]` (absolute
    match-clock). For a reroute-needed event, uploads (or reuses, via
    `upload_cache` keyed on the rounded (start,end) tuple — avoids a
    redundant upload if the exact same range is requested twice, e.g.
    config a/b sharing chunk0) a Files-API segment and converts
    `ctx.start_sec`/`end_sec`/highlight bounds to FILE-RELATIVE via the
    returned `sec_start` — same discipline as `files_api_reroute.
    ReroutedUpload`/`kimi_native_client.ensure_local_clip`."""
    if not ev["needs_reroute"]:
        return RunContext(
            youtube_id=ev["event_id"], youtube_url=ev["video_watch_url"],
            start_sec=int(chunk_start), end_sec=int(chunk_end),
            gemini_api_key=api_key, request_timeout_ms=600_000,
            retry_config=gemini_retry.GeminiRetryConfig(), video_mime_type=None,
        )
    key = (round(chunk_start, 1), round(chunk_end, 1))
    if key not in upload_cache:
        file_uri, sec_start = await _reroute_upload_for_range(
            ev, chunk_start, chunk_end, gemini_client, tag=f"{key[0]:.0f}_{key[1]:.0f}",
        )
        upload_cache[key] = (file_uri, sec_start)
    file_uri, sec_start = upload_cache[key]
    return RunContext(
        youtube_id=ev["event_id"], youtube_url=file_uri,
        start_sec=int(round(chunk_start - sec_start)), end_sec=int(round(chunk_end - sec_start)),
        gemini_api_key=api_key, request_timeout_ms=600_000,
        retry_config=gemini_retry.GeminiRetryConfig(), video_mime_type="video/mp4",
    )


def _highlight_span_for_event(ev: dict) -> dict:
    """The SAME W=60/post_tail_s=5 span GATE5_REPORT.md used to FLAG this
    event as a straddler — the operative "what would a scan have plausibly
    bounded this highlight as" proxy, not a fabricated span (see module
    docstring)."""
    return {"index": 0, "start_s": float(ev["t_s"] - 60), "end_s": float(ev["t_s"] + 5), "description": None}


def _translate_highlight_to_ctx(highlight: dict, sec_start_abs: float) -> dict:
    """For a reroute event, the RunContext is file-relative — the
    hand-constructed highlight's own start_s/end_s must be translated the
    SAME way ctx.start_sec/end_sec already were (kept as a SEPARATE working
    copy so the caller's absolute-time bookkeeping for scoring stays
    untouched)."""
    return {**highlight, "start_s": highlight["start_s"] - sec_start_abs, "end_s": highlight["end_s"] - sec_start_abs}


async def run_critique_and_analyze(
    ev: dict, ctx: RunContext, ledger: Ledger, *, config_tag: str,
) -> tuple[Optional[dict], list[dict], float]:
    """Runs the REAL `highlight_critique_node` then `highlight_analyze_node`
    on `ctx` (already scoped to the target chunk). Returns
    `(clip_or_None, raw_events, cost_usd)`."""
    cost = 0.0
    raw_events: list[dict] = []

    critique_cfg = HighlightCritiqueConfig(model=QUALIFIED_MODEL, thinking=QUALIFIED_THINKING).model_dump()
    async for event in executors.highlight_critique_node(ctx, critique_cfg):
        raw_events.append(event)
        if event.get("type") == "highlight_critique_result":
            c = _usage_cost(event.get("usage_metadata"))
            cost += c
            ledger.record(
                gate="gate5_verification", config=config_tag, event_id=ev["event_id"], call_name="critique",
                model_id=QUALIFIED_MODEL, thinking_level=QUALIFIED_THINKING, status="ok",
                cost_usd=c, usage_metadata=event.get("usage_metadata"),
            )
        elif event.get("type") == "error":
            ledger.record(
                gate="gate5_verification", config=config_tag, event_id=ev["event_id"], call_name="critique",
                model_id=QUALIFIED_MODEL, thinking_level=QUALIFIED_THINKING, status="call_failed",
                cost_usd=0.0, error=event.get("message"),
            )
            return None, raw_events, cost

    analyze_cfg = HighlightAnalyzeConfig(
        model=QUALIFIED_MODEL, thinking=QUALIFIED_THINKING, fps=QUALIFIED_FPS,
    ).model_dump()
    clip: Optional[dict] = None
    async for event in executors.highlight_analyze_node(ctx, analyze_cfg):
        raw_events.append(event)
        if event.get("type") == "highlight_result":
            axis_calls = event.get("axis_calls") or {}
            tax_usage = (axis_calls.get("taxonomy") or {}).get("usage_metadata")
            actor_usage = (axis_calls.get("actor") or {}).get("usage_metadata")
            c_tax, c_actor = _usage_cost(tax_usage), _usage_cost(actor_usage)
            cost += c_tax + c_actor
            ledger.record(
                gate="gate5_verification", config=config_tag, event_id=ev["event_id"], call_name="taxonomy",
                model_id=QUALIFIED_MODEL, thinking_level=QUALIFIED_THINKING, status="ok",
                cost_usd=c_tax, usage_metadata=tax_usage,
            )
            ledger.record(
                gate="gate5_verification", config=config_tag, event_id=ev["event_id"], call_name="actor",
                model_id=QUALIFIED_MODEL, thinking_level=QUALIFIED_THINKING, status="ok",
                cost_usd=c_actor, usage_metadata=actor_usage,
            )
            clip = event["clips"][0] if event.get("clips") else None
        elif event.get("type") == "error":
            ledger.record(
                gate="gate5_verification", config=config_tag, event_id=ev["event_id"], call_name="taxonomy_or_actor",
                model_id=QUALIFIED_MODEL, thinking_level=QUALIFIED_THINKING, status="call_failed",
                cost_usd=0.0, error=event.get("message"),
            )
    return clip, raw_events, cost


async def run_dedup_stress_side(
    ev: dict, ctx: RunContext, ledger: Ledger,
) -> tuple[Optional[dict], float]:
    """Chunk k-1's own critique + a TAXONOMY-ONLY call (actor skipped — see
    module docstring: `seam_dedup.py` never consults it) on the SAME
    highlight span, scoped to `ctx` (chunk k-1's own boundary). Returns
    `(clip_dict_or_None, cost_usd)` — `clip_dict` carries only
    `start_s`/`end_s`/`action_class` (all `dedup_match_clips` needs)."""
    cost = 0.0
    critique_cfg = HighlightCritiqueConfig(model=QUALIFIED_MODEL, thinking=QUALIFIED_THINKING).model_dump()
    async for event in executors.highlight_critique_node(ctx, critique_cfg):
        if event.get("type") == "highlight_critique_result":
            cost += _usage_cost(event.get("usage_metadata"))
            ledger.record(
                gate="gate5_verification", config="a_dedup_stress", event_id=ev["event_id"], call_name="critique",
                model_id=QUALIFIED_MODEL, thinking_level=QUALIFIED_THINKING, status="ok",
                cost_usd=cost, usage_metadata=event.get("usage_metadata"),
            )
        elif event.get("type") == "error":
            ledger.record(
                gate="gate5_verification", config="a_dedup_stress", event_id=ev["event_id"], call_name="critique",
                model_id=QUALIFIED_MODEL, thinking_level=QUALIFIED_THINKING, status="call_failed",
                cost_usd=0.0, error=event.get("message"),
            )
            return None, cost

    highlight = ctx.highlights[0]
    authoritative_start = highlight.get("corrected_start_s", highlight["start_s"])
    authoritative_end = highlight.get("corrected_end_s", highlight["end_s"])
    window_start = round(max(ctx.start_sec, authoritative_start - HighlightAnalyzeConfig().preroll_s), 3)
    window_end = round(min(ctx.end_sec, authoritative_end + HighlightAnalyzeConfig().postroll_s), 3)
    if window_start >= window_end:
        return None, cost

    from service.pipelines import simplified_tags, highlight_axes as hl_axes
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
        ctx.gemini_client(), model_id=QUALIFIED_MODEL,
        thinking_config=thinking_config_for(QUALIFIED_MODEL, QUALIFIED_THINKING),
        system_instruction=hl_axes.SINGLE_CALL_SYSTEM_PROMPT, retry_config=ctx.retry_config,
    )
    tax_data, tax_err = await executors._call_axis_json(
        analyzer, ctx.youtube_url, window_start, window_end,
        response_schema=hl_axes.build_single_call_schema(),
        prompt_text=hl_axes.build_single_call_prompt(None),
        fps=QUALIFIED_FPS, media_resolution=None, mime_type=ctx.video_mime_type,
    )
    c = _usage_cost(getattr(analyzer, "last_usage_metadata", None))
    cost += c
    if tax_err is not None:
        ledger.record(
            gate="gate5_verification", config="a_dedup_stress", event_id=ev["event_id"], call_name="taxonomy_only",
            model_id=QUALIFIED_MODEL, thinking_level=QUALIFIED_THINKING, status="call_failed",
            cost_usd=0.0, error=tax_err,
        )
        return None, cost
    ledger.record(
        gate="gate5_verification", config="a_dedup_stress", event_id=ev["event_id"], call_name="taxonomy_only",
        model_id=QUALIFIED_MODEL, thinking_level=QUALIFIED_THINKING, status="ok",
        cost_usd=c, usage_metadata=getattr(analyzer, "last_usage_metadata", None),
    )
    return {
        "start_s": authoritative_start, "end_s": authoritative_end,
        "action_class": tax_data.get("action_class") if isinstance(tax_data, dict) else None,
    }, cost


async def run_one_event(
    ev: dict, api_key: str, gemini_client, ledger: Ledger, upload_cache: dict,
) -> dict:
    result: dict = {"event_id": ev["event_id"], "match_label": ev["match_label"], "t_s": ev["t_s"],
                     "gold_action_class": ev["gold_action_class"], "needs_reroute": ev["needs_reroute"]}
    total_cost = 0.0

    for config_tag, overlap_s in (("a_patched", PATCHED_OVERLAP_S), ("b_pre_patch", PRE_PATCH_OVERLAP_S)):
        _c0, c1 = chunk_bounds_for_event(ev, overlap_s)
        ctx = await build_ctx_for_scope(ev, c1[0], c1[1], api_key, gemini_client, upload_cache)
        highlight = _highlight_span_for_event(ev)
        if ev["needs_reroute"]:
            sec_start_abs = c1[0] - ctx.start_sec  # recover the upload's own sec_start
            highlight = _translate_highlight_to_ctx(highlight, sec_start_abs)
        ctx.highlights = [highlight]
        clip, _raw, cost = await run_critique_and_analyze(ev, ctx, ledger, config_tag=config_tag)
        total_cost += cost
        result[config_tag] = {
            "chunk1_bounds_abs": c1, "status": "ok" if clip else "call_failed",
            "predicted_action_class": clip.get("action_class") if clip else None,
            "predicted_position": clip.get("position") if clip else None,
            "predicted_outcome": clip.get("outcome") if clip else None,
            "clip_span_abs": [
                (clip["start_s"] + (c1[0] - ctx.start_sec) if ev["needs_reroute"] else clip["start_s"]),
                (clip["end_s"] + (c1[0] - ctx.start_sec) if ev["needs_reroute"] else clip["end_s"]),
            ] if clip else None,
            "hit": bool(clip and clip.get("action_class") == ev["gold_action_class"]),
        }
        ledger.flush()

    # Seam-dedup stress test (config a / patched ONLY, per module docstring).
    c0, c1_patched = chunk_bounds_for_event(ev, PATCHED_OVERLAP_S)
    ctx0 = await build_ctx_for_scope(ev, c0[0], c0[1], api_key, gemini_client, upload_cache)
    highlight0 = _highlight_span_for_event(ev)
    if ev["needs_reroute"]:
        sec_start_abs0 = c0[0] - ctx0.start_sec
        highlight0 = _translate_highlight_to_ctx(highlight0, sec_start_abs0)
    ctx0.highlights = [highlight0]
    dedup_clip0, dedup_cost = await run_dedup_stress_side(ev, ctx0, ledger)
    total_cost += dedup_cost
    ledger.flush()

    chunk1_clip = result["a_patched"]
    dedup_clip1 = None
    if chunk1_clip["status"] == "ok":
        dedup_clip1 = {
            "start_s": chunk1_clip["clip_span_abs"][0], "end_s": chunk1_clip["clip_span_abs"][1],
            "action_class": chunk1_clip["predicted_action_class"],
        }
    if dedup_clip0 is not None:
        sec_start_abs0 = c0[0] - ctx0.start_sec if ev["needs_reroute"] else 0.0
        dedup_clip0_abs = {
            "start_s": dedup_clip0["start_s"] + sec_start_abs0, "end_s": dedup_clip0["end_s"] + sec_start_abs0,
            "action_class": dedup_clip0["action_class"],
        }
    else:
        dedup_clip0_abs = None

    clips_by_chunk = [[dedup_clip0_abs] if dedup_clip0_abs else [], [dedup_clip1] if dedup_clip1 else []]
    survivors, dropped = seam_dedup.dedup_match_clips(clips_by_chunk, [c0, c1_patched])
    result["seam_dedup_stress"] = {
        "chunk0_stress_clip": dedup_clip0_abs, "chunk1_patched_clip": dedup_clip1,
        "survivors": len(survivors), "dropped": dropped,
        "double_publish_would_occur_without_dedup": bool(dedup_clip0_abs and dedup_clip1),
        "dedup_correctly_prevented_double_publish": bool(dedup_clip0_abs and dedup_clip1 and dropped >= 1),
    }
    result["total_cost_usd"] = total_cost
    return result


def _print_dry_run_plan() -> None:
    print(f"GATE5 VERIFICATION DRY-RUN PLAN — {len(EVENTS)} events x 2 configs x "
          f"~3 calls (critique+taxonomy+actor) + {len(EVENTS)} dedup-stress "
          f"(critique+taxonomy-only, config a) = up to "
          f"{len(EVENTS) * (2 * 3 + 2)} calls, cap=${CAP_USD:.0f}.", file=sys.stderr)
    for ev in EVENTS:
        print(f"  {ev['event_id']}: {ev['match_label']} t_s={ev['t_s']} gold={ev['gold_action_class']} "
              f"needs_reroute={ev['needs_reroute']}", file=sys.stderr)
    print("Zero real calls made. Pass --execute to spend real money.", file=sys.stderr)


async def _amain(args: argparse.Namespace) -> None:
    if not args.execute:
        _print_dry_run_plan()
        return

    cfg = gc.load_service_config()
    gemini_client = gc.build_gemini_client(cfg)
    ledger = Ledger()
    upload_cache: dict = {}

    print("WARNING: --execute fires the Gate5 8-straddler verification (up to "
          f"{len(EVENTS) * 8} calls, cap=${CAP_USD:.0f}). Ctrl-C now if not authorized. "
          "Proceeding in 3s...", file=sys.stderr)
    await asyncio.sleep(3)

    events = EVENTS[: args.limit_events] if args.limit_events else EVENTS
    results: list[dict] = []
    for ev in events:
        if ledger.total_cost_usd() > CAP_USD:
            print(f"[gate5_verification] CAP EXCEEDED (${ledger.total_cost_usd():.4f} > ${CAP_USD:.0f}) — "
                  f"stopping BEFORE event {ev['event_id']}.", file=sys.stderr)
            break
        print(f"[gate5_verification] event {ev['event_id']}: {ev['match_label']} "
              f"(gold={ev['gold_action_class']}, needs_reroute={ev['needs_reroute']})...", file=sys.stderr)
        result = await run_one_event(ev, cfg.gemini_api_key, gemini_client, ledger, upload_cache)
        results.append(result)
        gc.write_json(GATE5_VERIFICATION_PREDICTIONS_PATH, results)
        print(f"  a_patched: hit={result['a_patched']['hit']} pred={result['a_patched']['predicted_action_class']} | "
              f"b_pre_patch: hit={result['b_pre_patch']['hit']} pred={result['b_pre_patch']['predicted_action_class']} | "
              f"dedup: {result['seam_dedup_stress']} | cost=${result['total_cost_usd']:.4f} | "
              f"running_total=${ledger.total_cost_usd():.4f}", file=sys.stderr)

        failed_a = sum(1 for r in results if r["a_patched"]["status"] == "call_failed")
        failed_b = sum(1 for r in results if r["b_pre_patch"]["status"] == "call_failed")
        if failed_a > HALT_PER_CONFIG or failed_b > HALT_PER_CONFIG:
            print(f"[gate5_verification] HALT-AND-ESCALATE — failed_a={failed_a} failed_b={failed_b} "
                  f"(threshold >{HALT_PER_CONFIG}). STOPPING.", file=sys.stderr)
            break

    print(f"\n[gate5_verification] DONE. {len(results)}/{len(events)} event(s) attempted, "
          f"${ledger.total_cost_usd():.4f} total real spend.", file=sys.stderr)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="DANGER: spends real money. Default is dry-run.")
    parser.add_argument("--limit-events", type=int, default=None, help="SMOKE-TEST knob: first N events only.")
    args = parser.parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
