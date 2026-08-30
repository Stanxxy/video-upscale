#!/usr/bin/env python3
"""Build Arm B/C videos + viewer JSON for one clip (post-SAM3)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.adcc.sam3_scout.advisory_metrics import compute_advisory_metrics
from eval.adcc.sam3_scout.build_arm_videos import build_arm_videos
from eval.adcc.sam3_scout.viewer_manifest import build_viewer_entry, write_viewer_bundle


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--clip-id", required=True)
    p.add_argument("--source-mp4", required=True)
    p.add_argument("--tracks-jsonl", default=None)
    p.add_argument("--masks-dir", default=None)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    tracks = args.tracks_jsonl or str(run_dir / f"{args.clip_id}_tracks.jsonl")
    masks = args.masks_dir or str(run_dir / f"{args.clip_id}_masks")
    arms = build_arm_videos(
        args.source_mp4, tracks, masks, run_dir, args.clip_id,
    )
    adv = compute_advisory_metrics(tracks)
    entry = build_viewer_entry(
        clip_id=args.clip_id,
        raw_mp4=args.source_mp4,
        mask_preview_mp4=str(run_dir / f"{args.clip_id}_mask_preview.mp4"),
        arm_b_mp4=arms["arm_b"],
        arm_c_mp4=arms["arm_c"],
        advisory=adv,
    )
    out = write_viewer_bundle([entry], run_dir / f"{args.clip_id}_viewer.json")
    print(json.dumps({"viewer": str(out), "arms": arms, "advisory": adv}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
