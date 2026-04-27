"""
Public tracking API for the vision service.

Re-exports ``tracking_pipeline`` entry points so callers can use
``from tracking import run_tracking`` without touching ``sys.path``.
"""
from tracking_pipeline.detect import detect_persons
from tracking_pipeline.hybrid_tracking import run_tracking
from tracking_pipeline.pipeline import run_pipeline

__all__ = ["detect_persons", "run_tracking", "run_pipeline"]
