"""
Main hybrid tracking loop: SAM2 propagation + state machine.

Public API preserved for ``from tracking_pipeline.hybrid_tracking import run_tracking``.
"""
from tracking_pipeline.hybrid.intervention import _detect_and_request_boxes
from tracking_pipeline.hybrid.run_tracking import run_tracking
from tracking_pipeline.hybrid.yolo26_detector import NumpyEncoder, YOLO26Detector

__all__ = [
    "NumpyEncoder",
    "YOLO26Detector",
    "run_tracking",
    "_detect_and_request_boxes",
]
