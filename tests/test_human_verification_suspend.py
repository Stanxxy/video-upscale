"""HumanVerificationSuspend propagation from service-mode detection callbacks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tracking_pipeline.human_verification_suspend import HumanVerificationSuspend
from tracking_pipeline.hybrid_tracking import _detect_and_request_boxes


def test_detect_and_request_boxes_propagates_suspend():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def cb(*_a, **_kw):
        raise HumanVerificationSuspend()

    detector = MagicMock()
    detector.detect_persons.return_value = [
        {"box": [1.0, 2.0, 50.0, 100.0], "confidence": 0.95},
    ]

    with pytest.raises(HumanVerificationSuspend):
        _detect_and_request_boxes(
            frame, 0, cb, detector, "yolo26m", 0.5, "cpu",
        )


def test_detect_and_request_boxes_no_callback_short_circuits_without_yolo():
    """Bug fix: when ``detection_callback`` is None (parallel-segment / headless
    mode), the helper must NOT load YOLO. Previously the YOLO26Detector was
    constructed on every track-loss frame because the locally-loaded detector
    was discarded on the no-callback return path. Symptom in production logs:
    ``[detect] Loading yolo26m.pt on mps (persistent)...`` every few frames."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    with patch("tracking_pipeline.hybrid_tracking.YOLO26Detector") as yolo_ctor:
        result = _detect_and_request_boxes(
            frame, 100,
            None,  # detection_callback
            None,  # detector
            "yolo26m", 0.5, "cpu",
        )

    assert result is None
    yolo_ctor.assert_not_called()


def test_detect_and_request_boxes_no_callback_does_not_use_existing_detector():
    """Even when a previously-loaded detector is passed in, the no-callback
    path must short-circuit without running a forward pass — the callee has no
    way to consume YOLO results in parallel-segment mode."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    detector = MagicMock()
    detector.detect_persons.return_value = [
        {"box": [1.0, 2.0, 50.0, 100.0], "confidence": 0.95},
    ]

    result = _detect_and_request_boxes(
        frame, 100,
        None,  # detection_callback
        detector,
        "yolo26m", 0.5, "cpu",
    )

    assert result is None
    detector.detect_persons.assert_not_called()
