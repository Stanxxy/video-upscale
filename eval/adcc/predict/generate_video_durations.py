"""One-off, SERVER-SIDE-ONLY script (Task 5): probes real per-video durations
via ``ffprobe`` for every on-disk video file (any container — the corpus mixes
``.mp4``/``.mkv``) under the ADCC videos directory, and writes
``~/vlm-eval/video_durations.json`` (``{video_url_md5: duration_seconds}``,
the exact shape ``run_predictions.load_video_durations_json``/
``dashboard_scope``/``dashboard_app`` expect).

Metadata-only: ``ffprobe -show_entries format=duration`` reads only the
container header, transfers zero video bytes — same discipline as
``CALL_BUDGET_ESTIMATE.md`` §7's original duration-gathering pass. Run this
ON bjj-server (the videos live at
``/mnt/e/assets/data/bjj-vision/all|ADCC|all/videos/``, not on the Mac) via:

    ~/vlm-eval/venv/bin/python eval/adcc/predict/generate_video_durations.py

Re-run any time the on-disk corpus changes (new videos added) — safe to
re-run, always overwrites the full mapping from scratch (never merges stale
entries silently).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEFAULT_VIDEOS_DIR = Path("/mnt/e/assets/data/bjj-vision/all|ADCC|all/videos")
DEFAULT_OUT_PATH = Path.home() / "vlm-eval" / "video_durations.json"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def probe_duration_seconds(video_path: Path, timeout_s: float = 30.0) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, timeout=timeout_s, check=True,
    )
    return float(out.stdout.strip())


def probe_all(videos_dir: Path) -> tuple[dict, list]:
    durations: dict = {}
    errors: list = []
    files = sorted(p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    print(f"probing {len(files)} files (metadata-only ffprobe, zero video bytes transferred)...", file=sys.stderr)
    for f in files:
        try:
            durations[f.stem] = probe_duration_seconds(f)
        except Exception as e:  # noqa: BLE001 — log the real failure per file, keep probing the rest
            errors.append((f.stem, str(e)))
    return durations, errors


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args(argv)

    durations, errors = probe_all(args.videos_dir)
    args.out.write_text(json.dumps(durations, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} with {len(durations)} entries ({len(errors)} errors)", file=sys.stderr)
    for stem, err in errors:
        print(f"  ERROR {stem}: {err}", file=sys.stderr)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
