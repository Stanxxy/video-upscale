"""Regression: PoseEstimator must not call rtmlib on empty crops (ZeroDivisionError)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from tracking_pipeline.pose import PoseEstimator


@pytest.fixture
def pose_without_rtmpose():
    """PoseEstimator body stub — avoids loading ONNX / rtmlib weights."""
    est = PoseEstimator.__new__(PoseEstimator)
    est.body = MagicMock()
    return est


def test_zero_width_box_returns_zeros_without_calling_body(pose_without_rtmpose):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    # Thin vertical strip in float space collapses to zero width after int crop
    box = [32.0, 10.0, 32.0, 50.0]

    kpts, scores = pose_without_rtmpose.estimate(frame, box)

    assert kpts.shape == (17, 2)
    assert scores.shape == (17,)
    assert np.allclose(kpts, 0)
    assert np.allclose(scores, 0)
    pose_without_rtmpose.body.assert_not_called()


def test_zero_height_box_returns_zeros_without_calling_body(pose_without_rtmpose):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    box = [10.0, 32.0, 50.0, 32.0]

    kpts, scores = pose_without_rtmpose.estimate(frame, box)

    assert np.allclose(kpts, 0)
    assert np.allclose(scores, 0)
    pose_without_rtmpose.body.assert_not_called()


def test_out_of_frame_box_collapses_to_empty_crop(pose_without_rtmpose):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    # Entirely past right edge — clamping gives cx2 < cx1-style empty slice
    box = [100.0, 10.0, 110.0, 50.0]

    kpts, scores = pose_without_rtmpose.estimate(frame, box)

    assert np.allclose(kpts, 0)
    assert np.allclose(scores, 0)
    pose_without_rtmpose.body.assert_not_called()


def test_valid_box_calls_body(pose_without_rtmpose):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    box = [10.0, 10.0, 50.0, 50.0]
    pose_without_rtmpose.body.return_value = (
        [np.zeros((17, 2))],
        [np.zeros(17)],
    )

    pose_without_rtmpose.estimate(frame, box)

    pose_without_rtmpose.body.assert_called_once()
