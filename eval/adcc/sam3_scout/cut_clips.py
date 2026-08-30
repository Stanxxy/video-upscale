#!/usr/bin/env python3
"""Cut gold windows from ADCC media root and optionally rsync to gx10."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from eval.adcc.sam3_scout.clips_io import load_clips


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov"}


def find_source(media_root: Path, stem: str) -> Path | None:
    if not media_root.is_dir():
        return None
    hits = [p for p in media_root.glob(f"{stem}.*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not hits:
        hits = [p for p in media_root.rglob(f"{stem}.*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not hits:
        return None
    return sorted(hits, key=lambda p: p.stat().st_size, reverse=True)[0]


def cut_clip(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.0, end_s - start_s)
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_s), "-i", str(src), "-t", str(dur),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", default=str(Path(__file__).with_name("clips.json")))
    p.add_argument("--media-root", default="/mnt/e/assets/data/bjj-vision/all|ADCC|all/videos")
    p.add_argument("--out-dir", default="/tmp/sam3_scout_clips")
    p.add_argument("--rsync-to-gx10", action="store_true")
    p.add_argument("--gx10-host", default="gx10")
    p.add_argument("--gx10-dir", default="/home/sliu1143/sam3_scout/clips")
    args = p.parse_args(argv)

    data = load_clips(args.clips)
    media = Path(args.media_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    missing = []
    for c in data["clips"]:
        src = find_source(media, c["video_url_md5"])
        dst = out_dir / f"{c['clip_id']}.mp4"
        if src is None:
            missing.append(c["clip_id"])
            print(f"MISSING {c['clip_id']} stem={c['video_url_md5']}", file=sys.stderr)
            continue
        print(f"CUT {c['clip_id']} from {src.name} [{c['window_start_s']},{c['window_end_s']}]")
        cut_clip(src, dst, float(c["window_start_s"]), float(c["window_end_s"]))
        written.append(dst)

    if args.rsync_to_gx10 and written:
        subprocess.check_call(["ssh", args.gx10_host, f"mkdir -p {args.gx10_dir}"])
        subprocess.check_call(
            ["rsync", "-avP"] + [str(p) for p in written] + [f"{args.gx10_host}:{args.gx10_dir}/"]
        )
    print(f"wrote {len(written)} clips; missing={len(missing)}")
    return 0 if written or not data["clips"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
