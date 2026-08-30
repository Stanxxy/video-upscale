#!/usr/bin/env bash
# Cut ADCC gold windows and rsync trimmed clips to gx10 for SAM3 scout.
# Does NOT run SAM3 on bjj-server — GPU work stays on gx10.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CLIPS_YAML="${CLIPS_YAML:-$(cd "$(dirname "$0")" && pwd)/clips.yaml}"
MEDIA_ROOT="${ADCC_MEDIA_ROOT:-/mnt/e/assets/data/bjj-vision/all|ADCC|all/videos}"
OUT_DIR="${SCOUT_CLIPS_OUT:-/tmp/sam3_scout_clips}"
GX10_HOST="${GX10_HOST:-gx10}"
GX10_DIR="${GX10_SCOUT_CLIPS_DIR:-/home/sliu1143/sam3_scout/clips}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUT_DIR"

python3 - <<'PY' "$CLIPS_YAML" "$MEDIA_ROOT" "$OUT_DIR"
import sys, os, subprocess, glob
from pathlib import Path

try:
    import yaml
except ImportError:
    # Minimal YAML subset loader for our clips.yaml (no nested complexity beyond lists/dicts)
    raise SystemExit("PyYAML required: pip install pyyaml")

clips_yaml, media_root, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
data = yaml.safe_load(Path(clips_yaml).read_text())
clips = data["clips"]
media = Path(media_root)
if not media.is_dir():
    print(f"WARNING: media root missing locally: {media}", file=sys.stderr)
    print("Will still emit ffmpeg commands / fail per clip if stems absent.", file=sys.stderr)

def find_source(stem: str) -> Path | None:
    patterns = [f"{stem}.*", f"**/{stem}.*"]
    for pat in patterns:
        hits = list(media.glob(pat)) if media.is_dir() else []
        hits = [h for h in hits if h.is_file() and h.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
        if hits:
            return sorted(hits, key=lambda p: p.stat().st_size, reverse=True)[0]
    return None

for c in clips:
    stem = c["video_url_md5"]
    src = find_source(stem)
    dst = Path(out_dir) / f"{c['clip_id']}.mp4"
    if src is None:
        print(f"MISSING source for {c['clip_id']} stem={stem}")
        continue
    start = float(c["window_start_s"])
    dur = float(c["window_end_s"]) - start
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", str(src), "-t", str(dur),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ]
    print("CUT", c["clip_id"], "from", src.name, f"[{start},{start+dur}]")
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(" ->", dst)
PY

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 — skip rsync"
  exit 0
fi

echo "Rsync clips -> ${GX10_HOST}:${GX10_DIR}"
ssh "$GX10_HOST" "mkdir -p '$GX10_DIR'"
rsync -avP --include='*.mp4' --exclude='*' "$OUT_DIR"/ "${GX10_HOST}:${GX10_DIR}/"
echo "Done. Clips on gx10 at ${GX10_DIR}"
