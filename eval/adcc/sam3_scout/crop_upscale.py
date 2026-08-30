"""Crop + pad + upscale helpers for Arms B/C.

Mirrors ``utils.get_union_box`` / ``get_padded_square_box`` from the vision
engine. Scout upscale target is founder-locked at 1080×1080 (not production 768).
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

DEFAULT_PADDING = 0.2
SCOUT_UPSCALE_TARGET = 1080


def get_union_box(boxes: Sequence[Sequence[float]]) -> Optional[list[float]]:
    if not boxes:
        return None
    arr = np.asarray(boxes, dtype=float)
    return [
        float(np.min(arr[:, 0])),
        float(np.min(arr[:, 1])),
        float(np.max(arr[:, 2])),
        float(np.max(arr[:, 3])),
    ]


def get_padded_square_box(
    box: Sequence[float],
    padding: float = DEFAULT_PADDING,
    img_shape: Optional[tuple[int, int]] = None,
) -> list[int]:
    x_min, y_min, x_max, y_max = [float(v) for v in box]
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2
    cy = y_min + h / 2
    side = max(w, h) * (1 + padding)
    hs = side / 2
    new_x1, new_y1 = cx - hs, cy - hs
    new_x2, new_y2 = cx + hs, cy + hs

    if img_shape is not None:
        img_h, img_w = img_shape
        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0
        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0
        if new_x2 > img_w:
            new_x1 -= new_x2 - img_w
            new_x2 = img_w
        if new_y2 > img_h:
            new_y1 -= new_y2 - img_h
            new_y2 = img_h
        new_x1 = max(0, int(new_x1))
        new_y1 = max(0, int(new_y1))
        new_x2 = min(img_w, int(new_x2))
        new_y2 = min(img_h, int(new_y2))
    else:
        new_x1, new_y1, new_x2, new_y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)
    return [new_x1, new_y1, new_x2, new_y2]


def boxes_from_masks(masks: np.ndarray) -> list[list[float]]:
    """``masks``: (N, H, W) bool/uint8 → list of [x1,y1,x2,y2]."""
    boxes: list[list[float]] = []
    if masks is None or len(masks) == 0:
        return boxes
    for m in masks:
        ys, xs = np.where(m.astype(bool))
        if len(xs) == 0:
            continue
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1])
    return boxes


def crop_square(frame_bgr: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty crop box {[x1, y1, x2, y2]} for frame shape {(h, w)}")
    crop = frame_bgr[y1:y2, x1:x2]
    side = max(crop.shape[0], crop.shape[1])
    square = np.zeros((side, side, 3), dtype=frame_bgr.dtype)
    # Center the (possibly non-square after clamp) crop into a square canvas.
    oy = (side - crop.shape[0]) // 2
    ox = (side - crop.shape[1]) // 2
    square[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
    return square


def upscale_to_square(
    crop_bgr: np.ndarray,
    target: int = SCOUT_UPSCALE_TARGET,
    restorer=None,
) -> np.ndarray:
    """Upscale crop to ``target×target``.

    Prefer ``restorer.enhance`` (RealESRGANRestorer) when provided; otherwise
    Fall back to Lanczos resize (FastUpscaler-style substitute — no separate
    FastUpscaler class exists in this repo).
    """
    if target <= 0:
        raise ValueError(f"target must be > 0, got {target!r}")
    if restorer is not None:
        enhanced = restorer.enhance(crop_bgr, target_size=target)
        h, w = enhanced.shape[:2]
        if h != target or w != target:
            enhanced = cv2.resize(enhanced, (target, target), interpolation=cv2.INTER_LANCZOS4)
        return enhanced
    return cv2.resize(crop_bgr, (target, target), interpolation=cv2.INTER_LANCZOS4)


def crop_and_upscale_frame(
    frame_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    padding: float = DEFAULT_PADDING,
    target: int = SCOUT_UPSCALE_TARGET,
    restorer=None,
) -> Optional[np.ndarray]:
    """Union → pad square → crop → upscale. Returns None if no boxes."""
    union = get_union_box(boxes)
    if union is None:
        return None
    h, w = frame_bgr.shape[:2]
    sq = get_padded_square_box(union, padding=padding, img_shape=(h, w))
    cropped = crop_square(frame_bgr, sq)
    return upscale_to_square(cropped, target=target, restorer=restorer)
