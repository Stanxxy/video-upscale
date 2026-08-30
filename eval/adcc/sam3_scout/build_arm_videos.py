"""Build Arm B / Arm C continuous frame MP4s from SAM3 tracks + source clip."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from eval.adcc.sam3_scout.crop_upscale import (
    DEFAULT_PADDING,
    SCOUT_UPSCALE_TARGET,
    crop_and_upscale_frame,
)
from eval.adcc.sam3_scout.highlight_composite import (
    HIGHLIGHT_FILL_ALPHA,
    composite_athlete_highlight,
)


def _load_tracks(tracks_jsonl: Path) -> list[dict]:
    rows = []
    with open(tracks_jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_arm_videos(
    source_mp4: str,
    tracks_jsonl: str,
    masks_dir: Optional[str],
    out_dir: Path,
    clip_id: str,
    padding: float = DEFAULT_PADDING,
    target: int = SCOUT_UPSCALE_TARGET,
    fill_alpha: float = HIGHLIGHT_FILL_ALPHA,
    restorer=None,
    fps_out: float = 10.0,
) -> dict[str, str]:
    """Emit ``{clip}_arm_b.mp4`` and ``{clip}_arm_c.mp4`` (and frame dirs).

    Frames without boxes fall back to a letterboxed full-frame resize so the
    sequence stays continuous for Gemini.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = _load_tracks(Path(tracks_jsonl))
    by_idx = {t["frame_index"]: t for t in tracks}
    masks_path = Path(masks_dir) if masks_dir else None

    arm_b_dir = out_dir / f"{clip_id}_arm_b_frames"
    arm_c_dir = out_dir / f"{clip_id}_arm_c_frames"
    arm_b_dir.mkdir(exist_ok=True)
    arm_c_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(source_mp4)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    # Sample at ~fps_out from source
    step = max(1, int(round(src_fps / float(fps_out))))
    fidx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % step != 0:
            fidx += 1
            continue
        t = by_idx.get(fidx, {"boxes": [], "mask_file": None})
        boxes = t.get("boxes") or []
        masks = None
        if masks_path and t.get("mask_file"):
            data = np.load(masks_path / t["mask_file"])
            masks = data["masks"]

        # Arm B
        b_frame = crop_and_upscale_frame(frame, boxes, padding=padding, target=target, restorer=restorer)
        if b_frame is None:
            b_frame = cv2.resize(frame, (target, target), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(arm_b_dir / f"{written:06d}.jpg"), b_frame)

        # Arm C — highlight then crop
        highlighted = composite_athlete_highlight(frame, masks if masks is not None else np.zeros((0, *frame.shape[:2]), dtype=np.uint8), fill_alpha=fill_alpha)
        c_frame = crop_and_upscale_frame(highlighted, boxes, padding=padding, target=target, restorer=restorer)
        if c_frame is None:
            c_frame = cv2.resize(highlighted, (target, target), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(arm_c_dir / f"{written:06d}.jpg"), c_frame)

        written += 1
        fidx += 1
    cap.release()

    arm_b_mp4 = str(out_dir / f"{clip_id}_arm_b.mp4")
    arm_c_mp4 = str(out_dir / f"{clip_id}_arm_c.mp4")
    _frames_to_mp4(arm_b_dir, arm_b_mp4, fps_out)
    _frames_to_mp4(arm_c_dir, arm_c_mp4, fps_out)
    return {"arm_b": arm_b_mp4, "arm_c": arm_c_mp4, "arm_b_frames": str(arm_b_dir), "arm_c_frames": str(arm_c_dir)}


def _frames_to_mp4(frame_dir: Path, out_mp4: str, fps: float) -> None:
    pattern = str(frame_dir / "%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out_mp4) or os.path.getsize(out_mp4) == 0:
        raise RuntimeError(f"failed to build {out_mp4}")
