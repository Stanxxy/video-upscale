"""Window clips cut from a source video with ffmpeg (cached by source, span and format)."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "temp_data"
MEDIA = ROOT / "vlm-core" / "media"
CACHE = ROOT / "x-mmagold" / "clips"
_LOCK = threading.Lock()  # one cut at a time: parallel callers want the same clip


def source_path(source: str) -> Path:
    path = MEDIA / f"{source}.mp4"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def cut(source: Path, start_ms: int, end_ms: int, *, height: int = 480, fps: int = 10, crf: int = 26,
        out: Path | None = None) -> Path:
    out = out or CACHE / f"{source.stem}_{start_ms}_{end_ms}_{height}p{fps}.mp4"
    with _LOCK:
        return _cut(source, start_ms, end_ms, height, fps, crf, out)


def _cut(source, start_ms, end_ms, height, fps, crf, out):
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start_ms / 1000:.3f}", "-to", f"{end_ms / 1000:.3f}",
         "-i", str(source), "-an", "-vf", f"scale=-2:{height},fps={fps}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        check=True,
    )
    frames = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                             "stream=nb_read_packets", "-of", "csv=p=0", str(tmp)], capture_output=True, text=True).stdout.strip()
    if not frames.isdigit() or int(frames) == 0:
        raise RuntimeError(f"cut produced no video frames: {tmp}")  # a model answers a broken clip confidently
    tmp.replace(out)
    return out
