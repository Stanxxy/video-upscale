"""
Root-level tracking package shim.

Inserts tracking_pipeline/ into sys.path so service/ code can import modules like
`detect`, etc.  The `run_tracking` function lives in tracking_pipeline/tracking.py
which shares its base name with this package — we use importlib to load it by
file path to avoid circular resolution.

Usage:
    from tracking import detect_persons, run_tracking
"""
import importlib.util
import os
import sys

_TEST_TRACKING_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "tracking_pipeline")
)

if _TEST_TRACKING_DIR not in sys.path:
    sys.path.insert(0, _TEST_TRACKING_DIR)

# tracking_pipeline/tracking.py — load FIRST via importlib (name collision).
# This must happen before importing pipeline.py, which itself does
# `from tracking import run_tracking` and expects to find it here.
_tracking_path = os.path.join(_TEST_TRACKING_DIR, "tracking.py")
_spec = importlib.util.spec_from_file_location("_tracking_pipeline", _tracking_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_tracking = _mod.run_tracking

# These modules have unique names — regular import is fine.
from detect import detect_persons  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

__all__ = ["detect_persons", "run_tracking", "run_pipeline"]
