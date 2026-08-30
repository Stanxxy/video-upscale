"""LeCun cutover Gates 1 & 2 — subset freeze (``GATES_1_2_DESIGN.md`` §A).

**Zero Gemini calls, zero network.** Pure, deterministic selection over the
already-committed gold snapshot. Writes
``eval/adcc/s12_gates/gate1_subset_manifest.json`` (38 windows) and
``eval/adcc/s12_gates/gate2_subset_manifest.json`` (16 clips) per the design
doc's §F.1 file layout. Re-running this script is idempotent — same seed,
same live gold snapshot -> byte-identical manifest content (modulo the
``generated_at`` timestamp), by construction.

Run from the repo root (``whole-video-analysis-wt-vlmstudio/``):

    .venv/bin/python -m eval.adcc.s12_gates.freeze_subsets
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from eval.adcc.s12_gates import gate_common as gc


def _build_gate1_manifest(events: list[dict]) -> dict:
    subset = gc.build_gate1_subset(events)
    video_durations = gc.load_video_durations()
    windows = []
    for i, event in enumerate(subset):
        video_url_watch, window_start_s, window_end_s, clamp_warning = gc.resolve_window_for_event(
            event, video_durations,
        )
        windows.append({
            "window_id": f"{event['match_id']}__w{i}",
            "index": i,
            "match_id": event["match_id"],
            "gate5_straddler": event["gate5_straddler"],
            "gold_event": {
                "t_s": event["t_s"], "action_class": event["action_class"], "outcome": event["outcome"],
                "position_bucket": event["position_bucket"], "raw_sequenceType": event["raw_sequenceType"],
                "track": event["track"], "scoreable": event["scoreable"],
            },
            "video_url_watch": video_url_watch,
            "window_start_s": window_start_s, "window_end_s": window_end_s,
            "clamp_warning": clamp_warning,
        })
    body = {
        "gate": "gate1",
        "design_doc": "eval/adcc/s12_gates/GATES_1_2_DESIGN.md",
        "seed": gc.SEED,
        "settings": {
            "fps": gc.FPS, "model": gc.MODEL_ID, "thinking": None,
            "preroll_s": gc.PREROLL_S, "postroll_s": gc.POSTROLL_S, "tau_t": gc.TAU_T,
            "media_resolution": {"arm_s": None, "position": None, "technique": None, "validator": "high"},
        },
        "quotas": gc.GATE1_QUOTAS,
        "frame_counts_post_straddler_removal": gc.GATE1_EXPECTED_FRAME_COUNTS,
        "gate5_straddlers": [{"match_id": m, "action_class": c, "t_s": t} for m, c, t in gc.GATE5_STRADDLERS],
        "n_windows": len(windows),
        "windows": windows,
        "gold_snapshot_path": str(gc.GOLD_PATH),
        "gold_snapshot_sha256": gc.sha256_file(gc.GOLD_PATH),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    body["manifest_sha256"] = gc.sha256_json_obj(body)
    return body


def _build_gate2_manifest(events: list[dict]) -> dict:
    subset = gc.build_gate2_subset(events)
    video_durations = gc.load_video_durations()
    clips = []
    for i, event in enumerate(subset):
        video_url_watch, window_start_s, window_end_s, clamp_warning = gc.resolve_window_for_event(
            event, video_durations,
        )
        clips.append({
            "window_id": f"{event['match_id']}__occ{i}",
            "index": i,
            "match_id": event["match_id"],
            "gold_event": {
                "t_s": event["t_s"], "action_class": event["action_class"], "outcome": event["outcome"],
                "position_bucket": event["position_bucket"], "raw_sequenceType": event["raw_sequenceType"],
                "track": event["track"], "scoreable": event["scoreable"],
            },
            "video_url_watch": video_url_watch,
            "window_start_s": window_start_s, "window_end_s": window_end_s,
            "clamp_warning": clamp_warning,
        })
    body = {
        "gate": "gate2",
        "design_doc": "eval/adcc/s12_gates/GATES_1_2_DESIGN.md",
        "seed": gc.SEED,
        "rng_procedure": (
            "Bezos condition 5 (founder pass-with-conditions, 2026-07-25) supersedes the design doc's "
            "original 'one shared Random instance' text: arm_lock's and leg_lock's successful-pool "
            "samples each use their OWN freshly-seeded random.Random(20260726) instance, drawn in class "
            "order (arm_lock, then leg_lock)."
        ),
        "settings": {
            "fps": gc.FPS, "model": gc.MODEL_ID, "thinking": None,
            "preroll_s": gc.PREROLL_S, "postroll_s": gc.POSTROLL_S,
            "media_resolution": {"position": None, "technique": None, "validator_low": "low", "validator_high": "high"},
        },
        "quota_per_class": gc.GATE2_QUOTA_PER_CLASS,
        "forced_unsuccessful": {
            cls: [{"match_id": m, "t_s": t} for m, t in pairs]
            for cls, pairs in gc.GATE2_FORCED_UNSUCCESSFUL.items()
        },
        "n_clips": len(clips),
        "clips": clips,
        "gold_snapshot_path": str(gc.GOLD_PATH),
        "gold_snapshot_sha256": gc.sha256_file(gc.GOLD_PATH),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    body["manifest_sha256"] = gc.sha256_json_obj(body)
    return body


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--print-only", action="store_true", help="Print summaries, do not write files.")
    args = parser.parse_args(argv)

    events = gc.load_hard_track_events()
    print(f"loaded {len(events)} hard-track gold events from {gc.GOLD_PATH}", file=sys.stderr)

    gate1 = _build_gate1_manifest(events)
    gate2 = _build_gate2_manifest(events)

    print(f"gate1: {gate1['n_windows']} windows (30 stratified + 8 forced straddlers)", file=sys.stderr)
    print(f"gate2: {gate2['n_clips']} clips (8 arm_lock [1 forced+7 sampled] + 8 leg_lock [5 forced+3 sampled])",
          file=sys.stderr)

    if args.print_only:
        return

    gc.write_json(gc.GATE1_MANIFEST_PATH, gate1)
    gc.write_json(gc.GATE2_MANIFEST_PATH, gate2)
    print(f"wrote {gc.GATE1_MANIFEST_PATH}", file=sys.stderr)
    print(f"wrote {gc.GATE2_MANIFEST_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
