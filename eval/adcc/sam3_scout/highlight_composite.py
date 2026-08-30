"""Arm C — low-alpha whole-athlete highlight composite (before crop+upscale).

Fill alpha is intentionally low so athletes remain visible ("do not block").
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

# Founder-tweakable after eyeball. Plan default band: 0.25–0.35.
HIGHLIGHT_FILL_ALPHA = 0.30
HIGHLIGHT_CONTOUR_THICKNESS = 2

# Distinct but muted BGR tints per instance index.
_COLORS = [
    (40, 180, 255),   # amber
    (255, 160, 40),   # blue-ish
    (80, 220, 80),    # green
    (200, 80, 220),   # magenta
]


def composite_athlete_highlight(
    frame_bgr: np.ndarray,
    masks: np.ndarray,
    fill_alpha: float = HIGHLIGHT_FILL_ALPHA,
    contour_thickness: int = HIGHLIGHT_CONTOUR_THICKNESS,
    colors: Sequence[tuple[int, int, int]] = _COLORS,
) -> np.ndarray:
    """Overlay semi-transparent fills + thin contours on whole-athlete masks.

    ``masks``: (N, H, W) bool/uint8. Empty masks → returns a copy of the frame.
    Raises if ``fill_alpha`` is not in (0, 1).
    """
    if not (0.0 < float(fill_alpha) < 1.0):
        raise ValueError(f"fill_alpha must be in (0,1), got {fill_alpha!r}")
    out = frame_bgr.copy()
    if masks is None or len(masks) == 0:
        return out
    h, w = frame_bgr.shape[:2]
    for i, m in enumerate(masks):
        mask = m.astype(bool)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mask.any():
            continue
        color = np.asarray(colors[i % len(colors)], dtype=np.float32)
        region = out[mask].astype(np.float32)
        blended = color * float(fill_alpha) + region * (1.0 - float(fill_alpha))
        out[mask] = blended.astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), int(contour_thickness))
    return out
