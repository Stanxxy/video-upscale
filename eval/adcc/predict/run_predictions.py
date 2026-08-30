"""In-process prediction-capture driver for the ADCC eval harness (plan §6.2).

Imports ``service.pipelines.registry.get_default("chunk-segment-tags")`` +
``service.pipelines.executors.run_pipeline``/``RunContext`` DIRECTLY — no live
uvicorn, no SSE parsing, no HTTP round-trip. Gemini ingests the YouTube URL
natively (``FileData(file_uri=...)``, already how ``chunk_segment_node``/
``chunk_analyze_node`` build their video parts) — no local video download.
These are READ-ONLY imports of ``service/pipelines/*``: this module never
modifies chunk_segment.py/executors.py/registry.py (hard constraint).

Each match's scope is computed by ``scope_policy.build_slice_plan`` and, when
over ``MAX_SCOPE_SEC``, split into multiple slices; each slice is run as its
own independent ``chunk-segment-tags`` pipeline invocation (its own PASS-1
rough scan + PASS-2 deep loop, no shared context across slices — a genuinely
separate Gemini run per slice, which is exactly why ``stitch.py``'s seam-dedup
exists). ``chunk_analyze_node`` already converts chunk-relative Gemini seconds
to ABSOLUTE match-clock seconds using the slice's own ``start_sec`` (verified:
each slice's ``ctx.start_sec``/``ctx.end_sec`` ARE the absolute video-clock
offsets sent as ``video_metadata`` — see ``executors.chunk_segment_node``/
``chunk_analyze_node``), so this module does ZERO additional offset math on
the clips it collects — it only concatenates per-slice clip lists and hands
them to ``stitch.stitch_match_predictions``.

**--execute guard (mandatory, Part C2).** Default is DRY RUN: this module
computes and prints the full per-match/per-slice run plan (scope windows,
slice counts, per-slice ``estimate_run_plan`` call-budget estimate,
aggregate totals) — ZERO Gemini calls, safe to import/exercise in tests with
no network access and no API key. ``--execute`` is the ONLY code path that
can spend real Gemini quota, gated a SECOND time at the
``capture_match_predictions`` function boundary (raises without
``execute=True`` even if called directly, not just via the CLI flag) — this
task explicitly forbids ever invoking that path; the real 103-video (well,
258-match) batch is founder-gated, same as ``CALL_BUDGET_ESTIMATE.md``
documents.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eval.adcc.predict.scope_policy import Slice, build_slice_plan
from eval.adcc.predict.stitch import stitch_match_predictions
from eval.adcc.predict.youtube_url import normalize_youtube_watch_url
from service.pipelines import chunk_segment, registry
from service.pipelines.executors import RunContext, estimate_run_plan, run_pipeline
from service.pipelines.models import PipelineDef

logger = logging.getLogger("eval.adcc.predict.run_predictions")

__all__ = [
    "RunPredictionsError",
    "MatchTarget",
    "build_match_targets",
    "load_video_durations_json",
    "plan_match_slices",
    "estimate_slice_calls",
    "build_dry_run_plan",
    "capture_match_predictions",
    "main",
]

# service/config.py's own default (10 min) — reused, not re-invented, so a
# real (--execute, founder-run) invocation matches the live service's own
# per-call HTTP timeout behavior.
DEFAULT_REQUEST_TIMEOUT_MS = 600_000

PIPELINE_ID = "chunk-segment-tags"


class RunPredictionsError(ValueError):
    """Raised for a mis-configured run (missing duration for a single-match
    video, --execute called without an API key, execute=True bypassed
    without going through the CLI's own --execute flag) — never silently
    defaulted around."""


@dataclass(frozen=True)
class MatchTarget:
    """One match_id's identity + scope-policy inputs, derived fresh from the
    gold snapshot every time (never hardcoded) — mirrors
    ``group_by_video``'s "first-class asserted artifact, not an implicit
    assumption" discipline (plan §6.5) for the predict/ layer."""

    match_id: str
    video_url: str
    video_url_md5: str
    is_multi_match: bool
    gold_t_s_values: tuple[float, ...]


def build_match_targets(gold_records: list[dict]) -> dict[str, MatchTarget]:
    """One ``MatchTarget`` per ``match_id`` in ``gold_records``.

    ``is_multi_match`` is derived from whether >1 match_id shares this
    match's ``video_url_md5`` WITHIN THE PASSED-IN ``gold_records`` LIST —
    never hardcoded. Caller contract: pass the FULL set of gold records for
    every match whose video is in scope (e.g. one split's full record list,
    never an arbitrary match_id subset that might exclude a sibling match on
    the same shared VOD) — ``build_split.py`` already guarantees sibling
    matches on one video always land in the same split, so filtering by
    split (not by ad-hoc match_id list) preserves this correctly.
    """
    video_to_matches: dict[str, list[str]] = defaultdict(list)
    for rec in gold_records:
        video_to_matches[rec["video_url_md5"]].append(rec["match_id"])

    targets: dict[str, MatchTarget] = {}
    for rec in gold_records:
        stem = rec["video_url_md5"]
        is_multi = len(video_to_matches[stem]) > 1
        t_s_values = tuple(sorted(e["t_s"] for e in rec["events"]))
        targets[rec["match_id"]] = MatchTarget(
            match_id=rec["match_id"], video_url=rec["video_url"], video_url_md5=stem,
            is_multi_match=is_multi, gold_t_s_values=t_s_values,
        )
    return targets


def load_video_durations_json(path: Path) -> dict[str, float]:
    """Loads a ``{video_url_md5: ffprobe_duration_seconds}`` JSON mapping —
    the pre-gathered, metadata-only ffprobe durations (``CALL_BUDGET_ESTIMATE.md``
    §7: "reads only the file's container header... transfers zero video
    bytes"). This module never invokes ffprobe itself (keeps it network-free
    and unit-testable); the mapping is a caller-supplied input."""
    data = json.loads(path.read_text())
    return {str(k): float(v) for k, v in data.items()}


def plan_match_slices(target: MatchTarget, video_durations: dict[str, float]) -> list[Slice]:
    """Dispatches to ``scope_policy.build_slice_plan`` for one match. Raises
    ``RunPredictionsError`` (never a silent 0-duration guess) if a
    single-match video's duration is missing from ``video_durations``."""
    if target.is_multi_match:
        return build_slice_plan(target.match_id, is_multi_match=True, gold_t_s_values=target.gold_t_s_values)

    if target.video_url_md5 not in video_durations:
        raise RunPredictionsError(
            f"match {target.match_id!r}: single-match video {target.video_url_md5!r} has no ffprobe "
            "duration in --video-durations. Supply it (metadata-only ffprobe probe, no video bytes) "
            "before this match's scope can be computed."
        )
    return build_slice_plan(
        target.match_id, is_multi_match=False, ffprobe_duration_s=video_durations[target.video_url_md5],
    )


def estimate_slice_calls(pipeline_def: PipelineDef, sl: Slice) -> dict:
    """Real-code-formula call-budget estimate for ONE slice (never a
    guess) — reuses ``executors.estimate_run_plan`` verbatim, plus a
    defensive re-check that this slice is actually under
    ``chunk_segment.MAX_SCOPE_SEC`` (a ``scope_policy`` bug, not a real-world
    condition, would be the only way this ever fires)."""
    cap_error = chunk_segment.check_scope_cap(sl.duration_s)
    if cap_error is not None:  # pragma: no cover — scope_policy.slice_scope guarantees this never fires
        raise RunPredictionsError(
            f"slice {sl.match_id!r}#{sl.slice_index} still exceeds MAX_SCOPE_SEC after slicing "
            f"(scope_policy bug, not a real-world condition): {cap_error}"
        )
    return estimate_run_plan(pipeline_def, duration_sec=sl.duration_s)


def build_dry_run_plan(
    targets: list[MatchTarget], video_durations: dict[str, float],
) -> dict:
    """Zero-Gemini-call dry-run plan: per-match slice list + per-slice
    ``estimate_run_plan`` call-budget estimate + aggregate totals. This is
    the ONLY thing ``main()`` does unless ``--execute`` is passed."""
    pipeline_def = registry.get_default(PIPELINE_ID)
    registry.validate_pipeline_def(pipeline_def)  # sanity check on the registry's own default, should never fail

    per_match: list[dict] = []
    total_calls = 0
    total_slices = 0
    over_cap_match_count = 0

    for target in targets:
        slices = plan_match_slices(target, video_durations)
        if len(slices) > 1:
            over_cap_match_count += 1

        slice_plans = []
        match_calls = 0
        for sl in slices:
            planned = estimate_slice_calls(pipeline_def, sl)
            match_calls += planned["planned_gemini_calls"]
            slice_plans.append({
                "slice_index": sl.slice_index,
                "start_s": sl.start_s, "end_s": sl.end_s, "duration_s": sl.duration_s,
                "planned_gemini_calls": planned["planned_gemini_calls"],
            })

        total_calls += match_calls
        total_slices += len(slices)
        per_match.append({
            "match_id": target.match_id,
            "video_url_md5": target.video_url_md5,
            "is_multi_match": target.is_multi_match,
            "slice_count": len(slices),
            "slices": slice_plans,
            "total_planned_gemini_calls_worst_case": match_calls,
        })

    return {
        "pipeline_id": pipeline_def.id,
        "match_count": len(targets),
        "over_cap_match_count": over_cap_match_count,
        "total_slices": total_slices,
        "total_planned_gemini_calls_worst_case": total_calls,
        "per_match": per_match,
    }


async def _run_one_slice(pipeline_def: PipelineDef, ctx: RunContext, budget_cap: Optional[int]) -> list[dict]:
    """Drains ``run_pipeline``'s async generator for one slice, returning the
    ``run_complete`` event's ``clips`` (already absolute match seconds — see
    module docstring). Logs (never raises) a mid-run ``error`` event, same
    "run continues past one failure" discipline ``qa_pipeline.py`` already
    uses — a single slice's Gemini hiccup should not abort every other
    slice's already-in-flight capture."""
    planned = estimate_run_plan(pipeline_def, duration_sec=float(ctx.end_sec - ctx.start_sec))
    clips: list[dict] = []
    async for event in run_pipeline(pipeline_def, ctx, planned, budget_cap=budget_cap):
        if event.get("type") == "error":
            logger.error(
                "run_predictions: slice error (video=%s scope=%d-%ds): %s",
                ctx.youtube_id, ctx.start_sec, ctx.end_sec, event.get("message"),
            )
        elif event.get("type") == "run_complete":
            clips = event.get("clips", [])
    return clips


async def capture_match_predictions(
    target: MatchTarget,
    slices: list[Slice],
    *,
    gemini_api_key: str,
    request_timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    budget_cap: Optional[int] = None,
    execute: bool,
) -> dict:
    """Runs every slice for ONE match (sequentially — no shared context
    across slices by design, so there is no correctness reason to serialize
    beyond simplicity/rate-limit friendliness) and stitches the results.

    **SECOND --execute gate** (defense in depth beyond ``main()``'s CLI flag
    parsing): raises ``RunPredictionsError`` unless the caller explicitly
    passes ``execute=True`` — a direct (non-CLI) caller cannot accidentally
    spend Gemini quota by omission.
    """
    if not execute:
        raise RunPredictionsError(
            "capture_match_predictions called without execute=True -- this would spend real Gemini "
            "quota. This path is founder-gated; pass --execute explicitly on the CLI (see "
            "CALL_BUDGET_ESTIMATE.md) or execute=True directly if you are certain."
        )

    pipeline_def = registry.get_default(PIPELINE_ID)
    # LIVE-BUG FIX (2026-07-18): target.video_url is the ADCC gold snapshot's
    # raw /embed/<id> YouTube URL (INS-084) — Gemini's native file_uri
    # ingestion only accepts the canonical watch?v=<id> form (see
    # youtube_url.py's module docstring for the full incident writeup).
    # Normalized HERE, upstream of RunContext, so service/pipelines/
    # executors.py (co-tenant, byte-untouchable) receives an already-good
    # URL and needs zero changes. Fail-closed: a malformed URL raises here,
    # BEFORE any Gemini call — zero spend on a bad URL. Computed ONCE outside
    # the per-slice loop (the URL is match-invariant, never slice-dependent).
    canonical_youtube_url = normalize_youtube_watch_url(target.video_url)
    per_slice_clips: list[list[dict]] = []
    for sl in slices:
        ctx = RunContext(
            youtube_id=target.video_url_md5,
            youtube_url=canonical_youtube_url,
            start_sec=int(sl.start_s),
            end_sec=int(round(sl.end_s)),
            gemini_api_key=gemini_api_key,
            request_timeout_ms=request_timeout_ms,
        )
        clips = await _run_one_slice(pipeline_def, ctx, budget_cap)
        per_slice_clips.append(clips)

    return stitch_match_predictions(target.match_id, per_slice_clips)


def _load_gold_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _filter_gold_records(
    gold_records: list[dict], split_assignment: Optional[dict], split: str, match_ids: Optional[set[str]],
) -> list[dict]:
    if match_ids is not None:
        return [rec for rec in gold_records if rec["match_id"] in match_ids]
    if split == "all" or split_assignment is None:
        return gold_records
    wanted_ids: set[str] = set()
    for entry in split_assignment["assignment"].values():
        if entry["split"] == split:
            wanted_ids.update(entry["match_ids"])
    return [rec for rec in gold_records if rec["match_id"] in wanted_ids]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-snapshot", type=Path, default=Path("eval/adcc/gold_snapshot/normalized_gold.jsonl"))
    parser.add_argument("--split-assignment", type=Path, default=Path("eval/adcc/gold_snapshot/split_assignment.json"))
    parser.add_argument("--split", choices=["sandbox", "dev", "test", "all"], default="all")
    parser.add_argument("--match-ids", default=None, help="comma-separated match_id allowlist, overrides --split")
    parser.add_argument(
        "--video-durations", type=Path, default=None,
        help="JSON {video_url_md5: ffprobe_duration_seconds}; required for any single-match video in scope",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("eval/adcc/predictions"))
    parser.add_argument("--budget-cap", type=int, default=None, help="per-slice PASS-2 call-count cap override")
    parser.add_argument(
        "--execute", action="store_true",
        help="DANGER: spends real Gemini quota. Default is dry-run (plan + budget only, zero calls). "
             "Founder-gated -- do not pass this without explicit sign-off.",
    )
    args = parser.parse_args(argv)

    gold_records = _load_gold_records(args.gold_snapshot)
    split_assignment = json.loads(args.split_assignment.read_text()) if args.split_assignment.exists() else None
    match_ids = set(args.match_ids.split(",")) if args.match_ids else None
    gold_records = _filter_gold_records(gold_records, split_assignment, args.split, match_ids)

    targets = build_match_targets(gold_records)
    video_durations = load_video_durations_json(args.video_durations) if args.video_durations else {}

    plan = build_dry_run_plan(list(targets.values()), video_durations)

    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(
            f"\nDRY RUN ONLY — zero Gemini calls spent. {plan['match_count']} match(es), "
            f"{plan['total_slices']} slice(s) ({plan['over_cap_match_count']} match(es) needed slicing), "
            f"worst-case planned calls: {plan['total_planned_gemini_calls_worst_case']}. "
            "Pass --execute to actually run (founder-gated, real Gemini spend)."
        )
        return

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RunPredictionsError("--execute requires a GEMINI_API_KEY environment variable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for target in targets.values():
        slices = plan_match_slices(target, video_durations)
        result = asyncio.run(
            capture_match_predictions(
                target, slices, gemini_api_key=api_key, budget_cap=args.budget_cap, execute=True,
            )
        )
        out_path = args.out_dir / f"{target.match_id}.json"
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out_path} ({len(result['clips'])} clips from {len(slices)} slice(s))")


if __name__ == "__main__":
    main()
