"""Box selection utilities."""
from tracking_pipeline.select_boxes.web_ui import select_boxes_web
from tracking_pipeline.select_boxes.cv2_ui import (
    draw_detections,
    manual_draw_boxes,
    read_frame,
    select_boxes_from_detections,
)

__all__ = [
    "select_boxes_web",
    "draw_detections",
    "select_boxes_from_detections",
    "manual_draw_boxes",
    "read_frame",
]
