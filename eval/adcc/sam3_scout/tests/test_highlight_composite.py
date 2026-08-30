"""Arm C highlight composite tests."""
from __future__ import annotations

import numpy as np
import pytest

from eval.adcc.sam3_scout.highlight_composite import (
    HIGHLIGHT_FILL_ALPHA,
    composite_athlete_highlight,
)


def test_default_alpha_in_band():
    assert 0.25 <= HIGHLIGHT_FILL_ALPHA <= 0.35


def test_highlight_does_not_fully_occlude():
    frame = np.full((40, 40, 3), 200, dtype=np.uint8)
    masks = np.zeros((1, 40, 40), dtype=np.uint8)
    masks[0, 10:30, 10:30] = 1
    out = composite_athlete_highlight(frame, masks, fill_alpha=0.30)
    # Center pixel should still show substantial original luminance (not solid tint).
    center = out[20, 20].astype(float)
    assert center.mean() > 100  # not crushed to dark/opaque overlay
    assert not np.array_equal(out[20, 20], frame[20, 20])  # but did change


def test_empty_masks_copy():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out = composite_athlete_highlight(frame, np.zeros((0, 10, 10), dtype=np.uint8))
    assert np.array_equal(out, frame)


def test_bad_alpha_raises():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        composite_athlete_highlight(frame, np.zeros((0, 8, 8), dtype=np.uint8), fill_alpha=1.0)
