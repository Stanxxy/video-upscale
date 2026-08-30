"""Builds the frozen ``split_assignment.json`` — sandbox/dev/test split BY
``video_url_md5`` (never by ``match_id``), per plan §5.5/§8.3: match_ids
inherit their video's split, else two matches sharing a VOD land on opposite
sides and contaminate the boundary.

Stratified by per-video event-count tercile (aggregate event count across all
matches on that video — the natural video-level generalization of the plan's
"per-match event-count tercile" language for videos that host >1 match),
deterministic seeded shuffle within each tercile, then a 70/15/15 slice.
Frozen: re-running with the same gold snapshot + seed reproduces byte-
identical output; changing it is an explicit, logged action (plan §6.4).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

__all__ = ["build_split", "DEFAULT_SEED", "SPLIT_RATIOS"]

DEFAULT_SEED = 20260712  # date-derived, frozen — changing this is a logged action, not a tweak
SPLIT_RATIOS = {"sandbox": 0.70, "dev": 0.15, "test": 0.15}


def _tercile_of(rank_fraction: float) -> int:
    """0/1/2 tercile from a [0,1) rank fraction (low/med/high event count)."""
    if rank_fraction < 1 / 3:
        return 0
    if rank_fraction < 2 / 3:
        return 1
    return 2


def _slice_ratios(n: int) -> dict[str, int]:
    """Deterministic integer slice sizes summing to exactly ``n`` (largest-
    remainder method) — avoids the classic "70%+15%+15% of 7 videos doesn't
    sum to 7" rounding bug."""
    raw = {label: n * ratio for label, ratio in SPLIT_RATIOS.items()}
    floors = {label: int(v) for label, v in raw.items()}
    remainder = n - sum(floors.values())
    # distribute the remainder to the labels with the largest fractional part
    fractional_order = sorted(raw, key=lambda label: raw[label] - floors[label], reverse=True)
    for label in fractional_order[:remainder]:
        floors[label] += 1
    return floors


def build_split(
    gold_records: list[dict],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict]:
    """``gold_records`` is the list read from ``normalized_gold.jsonl``.
    Returns ``{video_url_md5: {"split": "sandbox"|"dev"|"test",
    "match_ids": [...], "event_count": int}}``.
    """
    per_video_matches: dict[str, list[str]] = defaultdict(list)
    per_video_events: dict[str, int] = defaultdict(int)
    for rec in gold_records:
        stem = rec["video_url_md5"]
        per_video_matches[stem].append(rec["match_id"])
        per_video_events[stem] += len(rec["events"])

    stems_by_event_count = sorted(per_video_events, key=lambda s: per_video_events[s])
    n = len(stems_by_event_count)
    tercile_of_stem = {
        stem: _tercile_of(rank / n) for rank, stem in enumerate(stems_by_event_count)
    }

    stems_by_tercile: dict[int, list[str]] = defaultdict(list)
    for stem, tercile in tercile_of_stem.items():
        stems_by_tercile[tercile].append(stem)

    rng = random.Random(seed)
    assignment: dict[str, dict] = {}
    for tercile in sorted(stems_by_tercile):
        stems = sorted(stems_by_tercile[tercile])  # deterministic pre-shuffle order
        rng.shuffle(stems)
        sizes = _slice_ratios(len(stems))
        cursor = 0
        for label in ("sandbox", "dev", "test"):
            count = sizes[label]
            for stem in stems[cursor:cursor + count]:
                assignment[stem] = {
                    "split": label,
                    "match_ids": sorted(per_video_matches[stem]),
                    "event_count": per_video_events[stem],
                }
            cursor += count

    return assignment


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-snapshot", required=True, type=Path, help="path to normalized_gold.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="output split_assignment.json path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in args.gold_snapshot.read_text().splitlines() if line.strip()]
    assignment = build_split(records, seed=args.seed)

    counts = defaultdict(int)
    for entry in assignment.values():
        counts[entry["split"]] += 1

    args.out.write_text(json.dumps({
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "video_counts": dict(counts),
        "assignment": assignment,
    }, indent=2, sort_keys=True) + "\n")
    print(f"split {len(assignment)} videos: {dict(counts)} -> {args.out}")


if __name__ == "__main__":
    main()
