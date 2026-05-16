"""HumanVerificationSuspend propagation from service-mode detection callbacks."""

from __future__ import annotations

from unittest.mock import MagicMock

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
