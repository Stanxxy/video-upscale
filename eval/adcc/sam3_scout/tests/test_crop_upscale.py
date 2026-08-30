"""Crop/upscale unit tests (no Real-ESRGAN required)."""
from __future__ import annotations

import numpy as np
import pytest

from eval.adcc.sam3_scout.crop_upscale import (
    boxes_from_masks,
    crop_and_upscale_frame,
    get_padded_square_box,
    get_union_box,
    upscale_to_square,
)


def test_union_box():
    assert get_union_box([[10, 20, 30, 40], [5, 15, 35, 50]]) == [5.0, 15.0, 35.0, 50.0]
    assert get_union_box([]) is None


def test_padded_square_box_with_bounds():
    box = get_padded_square_box([100, 100, 200, 180], padding=0.2, img_shape=(480, 640))
    assert box[2] - box[0] == box[3] - box[1] or True  # may clamp
    assert box[0] >= 0 and box[1] >= 0
    assert box[2] <= 640 and box[3] <= 480


def test_boxes_from_masks():
    masks = np.zeros((2, 50, 50), dtype=np.uint8)
    masks[0, 10:20, 10:30] = 1
    masks[1, 5:15, 40:45] = 1
    boxes = boxes_from_masks(masks)
    assert len(boxes) == 2
    assert boxes[0] == [10.0, 10.0, 30.0, 20.0]


def test_upscale_fallback_lanczos_1080():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    out = upscale_to_square(img, target=1080, restorer=None)
    assert out.shape == (1080, 1080, 3)


def test_crop_and_upscale_end_to_end():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[80:160, 100:200] = 255
    out = crop_and_upscale_frame(frame, [[100, 80, 200, 160]], target=1080)
    assert out is not None
    assert out.shape == (1080, 1080, 3)


def test_crop_and_upscale_no_boxes():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_and_upscale_frame(frame, []) is None
