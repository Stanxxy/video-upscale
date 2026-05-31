"""Lock public tracking import surface before hybrid_tracking refactor."""
from __future__ import annotations

import tracking
import tracking_pipeline.hybrid_tracking as hybrid_tracking


def test_tracking_public_exports():
    assert set(tracking.__all__) == {
        "detect_persons",
        "HumanVerificationSuspend",
        "run_tracking",
        "run_pipeline",
    }
    assert tracking.run_tracking is hybrid_tracking.run_tracking
    assert tracking.run_pipeline.__module__.startswith("tracking_pipeline")
    assert tracking.detect_persons.__module__.startswith("tracking_pipeline")


def test_hybrid_tracking_exports_run_tracking():
    assert callable(hybrid_tracking.run_tracking)
