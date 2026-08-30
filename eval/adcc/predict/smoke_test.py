"""3-video go/no-go smoke test (plan §6.2) — the cheap pre-flight gate before
the real 258-match batch runs. Selects exactly:

  1. **1 single-match video** — the smallest (fewest gold events) single-
     match video in the corpus. Sanity-checks the plain ``[0,
     ffprobe_duration]`` scope path + a single-slice run.
  2. **1 small multi-match video** — the multi-match VOD with the FEWEST
     sibling matches (still >1, so the N:1 grouping/gold-derived-window path
     is genuinely exercised, cheaply).
  3. **1 worst-case 25-match VOD** — the multi-match VOD with the MOST
     sibling matches in the corpus (plan §3/§6.5's "2 of them pack 25
     matches each into one ~8-hour livestream"). Only ONE representative
     match from this VOD is actually run (the go/no-go question is "does
     Gemini-YouTube-ingestion + gold-derived scoping + over-cap slicing work
     at all on this VOD," not "run all 25 of its matches").

This catches, cheaply, BEFORE the real batch: Gemini YouTube-ingestion
failure, the >720s duration-ceiling/slicing path, and multi-match scope-
alignment bugs (plan §6.2's stated purpose for the smoke test).

Same ``--execute`` discipline as ``run_predictions.py``: default is DRY RUN
(prints the 3-target plan, spends nothing); ``--execute`` is founder-gated
and this task explicitly forbids running it — this module is produced, not
run, per Part C4.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from eval.adcc.predict.run_predictions import (
    RunPredictionsError,
    MatchTarget,
    build_dry_run_plan,
    build_match_targets,
    capture_match_predictions,
    load_video_durations_json,
    plan_match_slices,
)

__all__ = ["select_smoke_targets", "main"]


def select_smoke_targets(targets: dict[str, MatchTarget]) -> list[MatchTarget]:
    """Deterministically picks the 3 go/no-go targets from whatever
    ``targets`` (built by ``build_match_targets`` from the FULL gold
    snapshot — never a pre-filtered subset, or the "worst-case 25-match VOD"
    criterion can't be evaluated) is passed in. Raises ``RunPredictionsError``
    if the corpus doesn't have the shape the smoke test needs (no
    single-match video, no multi-match video, or fewer than 2 DISTINCT
    multi-match videos to tell "small" from "worst-case" apart) — never
    silently substitutes a different shape of test.
    """
    video_to_targets: dict[str, list[MatchTarget]] = defaultdict(list)
    for t in targets.values():
        video_to_targets[t.video_url_md5].append(t)

    single_match_videos = {stem: ts for stem, ts in video_to_targets.items() if len(ts) == 1}
    multi_match_videos = {stem: ts for stem, ts in video_to_targets.items() if len(ts) > 1}

    if not single_match_videos:
        raise RunPredictionsError("smoke_test: no single-match video found in the provided gold records")
    if len(multi_match_videos) < 2:
        raise RunPredictionsError(
            "smoke_test: need >=2 distinct multi-match videos (one 'small', one 'worst-case') — "
            f"found {len(multi_match_videos)}"
        )

    single_stem = min(single_match_videos, key=lambda stem: len(single_match_videos[stem][0].gold_t_s_values))
    small_multi_stem = min(multi_match_videos, key=lambda stem: len(multi_match_videos[stem]))
    worst_case_stem = max(multi_match_videos, key=lambda stem: len(multi_match_videos[stem]))

    single_target = single_match_videos[single_stem][0]
    small_multi_target = multi_match_videos[small_multi_stem][0]
    worst_case_target = multi_match_videos[worst_case_stem][0]

    return [single_target, small_multi_target, worst_case_target]


def _load_gold_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-snapshot", type=Path, default=Path("eval/adcc/gold_snapshot/normalized_gold.jsonl"))
    parser.add_argument(
        "--video-durations", type=Path, default=None,
        help="JSON {video_url_md5: ffprobe_duration_seconds}; required for the selected single-match video",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("eval/adcc/predictions_smoke"))
    parser.add_argument("--budget-cap", type=int, default=None)
    parser.add_argument(
        "--execute", action="store_true",
        help="DANGER: spends real Gemini quota (3 matches, one an 8-hour VOD's worth of slices). "
             "Default is dry-run. Founder-gated — do not pass this without explicit sign-off.",
    )
    args = parser.parse_args(argv)

    gold_records = _load_gold_records(args.gold_snapshot)
    targets = build_match_targets(gold_records)  # FULL corpus — needed to find the true worst-case VOD
    smoke_targets = select_smoke_targets(targets)

    video_durations = load_video_durations_json(args.video_durations) if args.video_durations else {}
    plan = build_dry_run_plan(smoke_targets, video_durations)

    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(
            "\nSMOKE TEST DRY RUN ONLY — zero Gemini calls spent. Targets: "
            + ", ".join(f"{m['match_id']} ({'multi' if m['is_multi_match'] else 'single'}-match, "
                        f"{m['slice_count']} slice(s))" for m in plan["per_match"])
            + f". Worst-case planned calls: {plan['total_planned_gemini_calls_worst_case']}. "
            "Pass --execute to actually run the 3-video go/no-go (founder-gated)."
        )
        return

    import os
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RunPredictionsError("--execute requires a GEMINI_API_KEY environment variable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for target in smoke_targets:
        slices = plan_match_slices(target, video_durations)
        result = asyncio.run(
            capture_match_predictions(
                target, slices, gemini_api_key=api_key, budget_cap=args.budget_cap, execute=True,
            )
        )
        out_path = args.out_dir / f"{target.match_id}.json"
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append((target, slices, result))
        print(f"wrote {out_path} ({len(result['clips'])} clips from {len(slices)} slice(s))")

    print("\n--- SMOKE TEST GO/NO-GO SUMMARY ---")
    for target, slices, result in results:
        status = "GO" if result["clips"] or len(slices) == 1 else "INSPECT"
        print(f"{target.match_id}: {status} — {len(slices)} slice(s), {len(result['clips'])} clip(s)")


if __name__ == "__main__":
    main()
