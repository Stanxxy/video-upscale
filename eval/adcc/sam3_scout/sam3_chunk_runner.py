"""Chunked SAM3 runner for the scout (ported from gx10 INS-056 harness).

Designed to run **on gx10** inside the SAM3 venv. Local Mac unit tests cover
``chunk_math`` only; this module is exercised via dry-run on gx10.

Persists per-frame boxes (JSONL) + optional mask NPZ + mask-overlay preview MP4
so Arms B/C and the HTML viewer can consume tracks offline.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from eval.adcc.sam3_scout.chunk_math import plan_chunks
from eval.adcc.sam3_scout.crop_upscale import boxes_from_masks

# Lazy heavy imports happen inside run paths so local unit tests can import
# chunk helpers without CUDA/SAM3.


@dataclass
class ScoutRunManifest:
    clip_id: str
    video_path: str
    text_prompt: str
    duration_s: float
    chunked: bool
    n_chunks: int
    concurrency: int
    chunk_seconds: float
    threshold_seconds: float
    tracks_jsonl: str
    preview_mp4: Optional[str]
    masks_dir: Optional[str]


def probe_duration_seconds(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def ensure_h264(src: str, dst: str) -> str:
    """Unconditionally re-encode to H.264/yuv420p (INS-056 AV1 gotcha)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an", dst,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def trim_window(src: str, dst: str, start_s: float, end_s: float) -> str:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.0, float(end_s) - float(start_s))
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_s), "-i", src, "-t", str(dur),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an", dst,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def segment_video(video_path: str, chunk_dir: Path, chunk_seconds: float) -> list[str]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(chunk_dir / "chunk_%04d.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-force_key_frames", f"expr:gte(t,n_forced*{chunk_seconds})",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-f", "segment", "-segment_time", str(chunk_seconds), "-reset_timestamps", "1",
        pattern,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    paths = sorted(str(p) for p in chunk_dir.glob("chunk_*.mp4"))
    if not paths:
        raise RuntimeError(f"ffmpeg segmentation produced no chunks under {chunk_dir}")
    return paths


def _mask_overlay_frame(frame_bgr, masks, alpha: float = 0.45):
    import cv2
    import numpy as np

    colors = [
        (40, 180, 255), (255, 160, 40), (80, 220, 80), (200, 80, 220),
        (0, 255, 255), (255, 0, 128),
    ]
    overlay = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    for i, m in enumerate(masks):
        mask = m.astype(bool)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        color = np.asarray(colors[i % len(colors)], dtype=np.float32)
        overlay[mask] = (color * alpha + overlay[mask].astype(np.float32) * (1 - alpha)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in color), 2)
    return overlay


class Sam3SessionRunner:
    """Thin wrapper around Sam3VideoPredictor with a process-wide semaphore."""

    def __init__(self, bpe_path: Optional[str] = None, concurrency: int = 1):
        if concurrency != 1:
            raise ValueError("sam3_scout requires concurrency=1 (INS-056)")
        self._sem = threading.BoundedSemaphore(1)
        self._lock = threading.Lock()
        self._predictor = None
        self._bpe_path = bpe_path

    def _get_predictor(self):
        if self._predictor is None:
            with self._lock:
                if self._predictor is None:
                    import torch
                    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

                    bpe = self._bpe_path
                    if bpe is None:
                        # Default relative to common gx10 layout
                        bpe = os.environ.get(
                            "SAM3_BPE_PATH",
                            "/home/sliu1143/Documents/vision/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                        )
                    self._predictor = Sam3VideoPredictor(
                        bpe_path=bpe,
                        apply_temporal_disambiguation=True,
                    )
                    self._torch = torch
        return self._predictor

    def run_on_video(
        self,
        video_path: str,
        text_prompt: str,
        frame_offset: int = 0,
        time_offset_s: float = 0.0,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> list[dict]:
        """Return list of {frame_index, t_s, obj_ids, boxes, masks?} per frame with detections."""
        import cv2
        import numpy as np
        import torch

        pred = self._get_predictor()
        records: list[dict] = []
        with self._sem:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                response = pred.handle_request({"type": "start_session", "resource_path": video_path})
                session_id = response["session_id"]
                try:
                    cap = cv2.VideoCapture(video_path)
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    pred.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": text_prompt,
                    })
                    frame_outputs = {}
                    for resp in pred.handle_stream_request({
                        "type": "propagate_in_video",
                        "session_id": session_id,
                    }):
                        fidx = int(resp["frame_index"])
                        frame_outputs[fidx] = resp["outputs"]
                        if on_progress:
                            on_progress(len(frame_outputs))

                    for fidx in range(total):
                        out = frame_outputs.get(fidx)
                        if out is None or len(out.get("out_obj_ids", [])) == 0:
                            records.append({
                                "frame_index": frame_offset + fidx,
                                "chunk_frame_index": fidx,
                                "t_s": time_offset_s + (fidx / fps),
                                "obj_ids": [],
                                "boxes": [],
                                "masks": None,
                            })
                            continue
                        masks = np.asarray(out["out_binary_masks"])
                        obj_ids = [int(x) for x in out["out_obj_ids"]]
                        boxes = boxes_from_masks(masks)
                        records.append({
                            "frame_index": frame_offset + fidx,
                            "chunk_frame_index": fidx,
                            "t_s": time_offset_s + (fidx / fps),
                            "obj_ids": obj_ids,
                            "boxes": boxes,
                            "masks": masks,  # caller may drop before JSON serialize
                        })
                finally:
                    pred.handle_request({"type": "close_session", "session_id": session_id})
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        return records


def run_chunked_sam3(
    video_path: str,
    out_dir: Path,
    clip_id: str,
    text_prompt: str = "person",
    chunk_seconds: float | None = None,
    persist_masks: bool = True,
    write_preview: bool = True,
    runner: Optional[Sam3SessionRunner] = None,
) -> ScoutRunManifest:
    """Plan chunks, run SAM3 serially, stitch track JSONL (+ optional preview)."""
    import cv2
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h264 = str(out_dir / f"{clip_id}_h264.mp4")
    ensure_h264(video_path, h264)
    duration = probe_duration_seconds(h264)
    plan = plan_chunks(duration, chunk_seconds=chunk_seconds, concurrency=1)

    runner = runner or Sam3SessionRunner(concurrency=1)
    all_records: list[dict] = []
    chunk_work = out_dir / f"{clip_id}_chunks"
    if chunk_work.exists():
        shutil.rmtree(chunk_work)

    if plan.chunked:
        chunk_paths = segment_video(h264, chunk_work, plan.chunk_seconds)
        frame_offset = 0
        time_offset = 0.0
        for i, cpath in enumerate(chunk_paths):
            recs = runner.run_on_video(
                cpath, text_prompt,
                frame_offset=frame_offset,
                time_offset_s=time_offset,
            )
            # Advance offsets from actual chunk length
            cap = cv2.VideoCapture(cpath)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            cap.release()
            frame_offset += n
            time_offset += n / fps
            all_records.extend(recs)
    else:
        all_records = runner.run_on_video(h264, text_prompt)

    tracks_path = out_dir / f"{clip_id}_tracks.jsonl"
    masks_dir = out_dir / f"{clip_id}_masks" if persist_masks else None
    if masks_dir is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)

    with open(tracks_path, "w") as f:
        for rec in all_records:
            masks = rec.get("masks")
            row = {k: v for k, v in rec.items() if k != "masks"}
            if persist_masks and masks is not None and masks_dir is not None and len(rec.get("boxes") or []):
                mask_path = masks_dir / f"frame_{rec['frame_index']:06d}.npz"
                np.savez_compressed(mask_path, masks=masks.astype(np.uint8), obj_ids=np.asarray(rec["obj_ids"]))
                row["mask_file"] = str(mask_path.name)
            f.write(json.dumps(row) + "\n")

    preview_path = None
    if write_preview:
        preview_path = str(out_dir / f"{clip_id}_mask_preview.mp4")
        _write_preview(h264, tracks_path, masks_dir, preview_path)

    manifest = ScoutRunManifest(
        clip_id=clip_id,
        video_path=h264,
        text_prompt=text_prompt,
        duration_s=duration,
        chunked=plan.chunked,
        n_chunks=plan.n_chunks,
        concurrency=plan.concurrency,
        chunk_seconds=plan.chunk_seconds,
        threshold_seconds=plan.threshold_seconds,
        tracks_jsonl=str(tracks_path),
        preview_mp4=preview_path,
        masks_dir=str(masks_dir) if masks_dir else None,
    )
    with open(out_dir / f"{clip_id}_manifest.json", "w") as f:
        json.dump(asdict(manifest), f, indent=2)
    return manifest


def _write_preview(video_path: str, tracks_jsonl: Path, masks_dir: Optional[Path], out_mp4: str) -> None:
    import cv2
    import numpy as np

    tracks = []
    with open(tracks_jsonl) as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    by_idx = {t["frame_index"]: t for t in tracks}
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = out_mp4 + ".raw.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = by_idx.get(fidx)
        if t and masks_dir and t.get("mask_file"):
            data = np.load(masks_dir / t["mask_file"])
            frame = _mask_overlay_frame(frame, data["masks"])
        writer.write(frame)
        fidx += 1
    cap.release()
    writer.release()
    subprocess.check_call(
        ["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.remove(raw)
