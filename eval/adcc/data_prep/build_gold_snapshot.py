"""Orchestrates data-prep: raw ADCC JSON -> committed gold snapshot.

Reads ONLY the two small source JSONs (``metadata_and_events.json``,
``video_links.json`` — pulled once from ``bjj-server`` via ``scp``/``ssh cat``,
never the ~100GB ``videos/`` directory) plus an on-disk video-stem listing
(``ssh ... ls videos/`` — filenames only, also cheap), and produces:

  - ``normalized_gold.jsonl`` — one JSON record per usable match_id.
  - ``manifest.json`` — source sha256s, asserted corpus counts, pull
    timestamp (passed in explicitly, never ``datetime.now()`` — this module
    is a pure function of its inputs plus that one caller-supplied string).

Zero Gemini calls, zero network beyond the one-time SSH pull that produced
the local input files this script reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

from eval.adcc.data_prep.apply_gracie_mapping import apply_mapping, load_mapping
from eval.adcc.data_prep.group_by_video import assert_corpus_counts, compute_video_stem
from eval.adcc.data_prep.parse_timestamp import parse_timestamp

__all__ = ["build_gold_snapshot", "sha256_of_file", "load_on_disk_stems", "main"]


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_on_disk_stems(listing_path: Path) -> set[str]:
    """Parses one-filename-per-line output of ``ls videos/`` (as captured by
    ``ssh bjj-server "ls '.../videos/'" > listing.txt``) into a set of
    on-disk stems (filename minus extension). Blank lines ignored."""
    stems: set[str] = set()
    with open(listing_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stem = line.rsplit(".", 1)[0]
            stems.add(stem)
    return stems


def _build_match_events(raw_events: list[dict], mapping: dict) -> list[dict]:
    """Sort one match's raw events chronologically (dual-format parse, fail-
    loud on malformed timestamps — no fallback), then apply the §4 mapping
    row by row, threading the immediately-prior RAW event for the POSITION
    ENTRY back_take dedup rule (§4.3 rule 1). Dropped rows (administrative
    no-ops, dedup-suppressed repeat back_takes) are simply absent from the
    output — never emitted as a placeholder."""
    parsed = [(parse_timestamp(e["timestamp"]), e) for e in raw_events]
    parsed.sort(key=lambda pair: pair[0])

    normalized: list[dict] = []
    prev_raw: Optional[dict] = None
    for t_s, raw_event in parsed:
        partial = apply_mapping(raw_event, prev_raw, mapping)
        prev_raw = raw_event  # advances regardless of drop — "immediately prior GOLD event"
        if partial is None:
            continue
        normalized.append({"t_s": t_s, **partial})
    return normalized


def build_gold_snapshot(
    metadata_and_events: dict,
    video_links: dict[str, str],
    on_disk_stems: set[str],
    mapping: dict,
    *,
    expected_usable_matches: int = 258,
    expected_unique_videos: int = 103,
) -> list[dict]:
    """Pure function: raw JSON dicts + mapping -> list of normalized gold
    match records (the exact rows to be written to ``normalized_gold.jsonl``,
    one per line). Corpus counts are asserted (never hardcoded past this
    point) before any record is built.
    """
    metadata = metadata_and_events["metadata"]
    events = metadata_and_events["events"]

    usable_ids, _unique_stems = assert_corpus_counts(
        video_links, events, on_disk_stems,
        expected_usable_matches=expected_usable_matches,
        expected_unique_videos=expected_unique_videos,
    )

    records: list[dict] = []
    for match_id in sorted(usable_ids):
        embed_url = video_links[match_id]
        stem = compute_video_stem(embed_url)
        normalized_events = _build_match_events(events[match_id], mapping)
        records.append({
            "match_id": match_id,
            "video_url": embed_url,
            "video_url_md5": stem,
            "name": metadata.get(match_id, {}).get("name", ""),
            "events": normalized_events,
        })
    return records


def _write_jsonl(records: list[dict], out_path: Path) -> None:
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path, help="path to local metadata_and_events.json copy")
    parser.add_argument("--links", required=True, type=Path, help="path to local video_links.json copy")
    parser.add_argument("--on-disk-stems-file", required=True, type=Path, help="path to `ls videos/` output")
    parser.add_argument("--mapping", default=None, type=Path, help="override gracie_mapping/v1.yaml path (tests only)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pulled-at", required=True, help="explicit ISO8601 pull timestamp string (no datetime.now())")
    parser.add_argument("--expected-usable-matches", type=int, default=258)
    parser.add_argument("--expected-unique-videos", type=int, default=103)
    args = parser.parse_args(argv)

    metadata_and_events = json.loads(args.metadata.read_text())
    video_links = json.loads(args.links.read_text())
    on_disk_stems = load_on_disk_stems(args.on_disk_stems_file)
    mapping = load_mapping(str(args.mapping) if args.mapping else None)

    records = build_gold_snapshot(
        metadata_and_events, video_links, on_disk_stems, mapping,
        expected_usable_matches=args.expected_usable_matches,
        expected_unique_videos=args.expected_unique_videos,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "normalized_gold.jsonl"
    _write_jsonl(records, jsonl_path)

    total_events = sum(len(r["events"]) for r in records)
    scoreable_events = sum(1 for r in records for e in r["events"] if e["scoreable"])
    unique_videos = {r["video_url_md5"] for r in records}

    manifest = {
        "mapping_version": mapping["version"],
        "pulled_at": args.pulled_at,
        "source_sha256": {
            "metadata_and_events.json": sha256_of_file(args.metadata),
            "video_links.json": sha256_of_file(args.links),
        },
        "asserted_counts": {
            "usable_match_ids": len(records),
            "unique_videos": len(unique_videos),
            "expected_usable_match_ids": args.expected_usable_matches,
            "expected_unique_videos": args.expected_unique_videos,
        },
        "derived_counts": {
            "total_normalized_events": total_events,
            "scoreable_events": scoreable_events,
            "non_scoreable_events": total_events - scoreable_events,
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote {len(records)} match records ({total_events} events, {scoreable_events} scoreable) to {jsonl_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
