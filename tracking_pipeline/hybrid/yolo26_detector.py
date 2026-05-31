"""YOLO26 on-demand detection for hybrid tracking."""
import json
import numpy as np
from ultralytics import YOLO

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class YOLO26Detector:
    """Persistent YOLO26 model for repeated per-frame detection."""

    def __init__(self, yolo_model="yolo26m", threshold=0.5, device=None):
        self.threshold = threshold
        model_name = yolo_model if yolo_model.endswith(".pt") else f"{yolo_model}.pt"
        self.device = str(device) if device is not None else "cpu"
        print(f"[detect] Loading {model_name} on {self.device} (persistent)...")
        self.model = YOLO(model_name)
        print("[detect] YOLO26 loaded.")

    def detect_persons(self, frame_bgr, min_area=2000):
        """Detect persons in a BGR frame. Returns list of {box, confidence}."""
        results = self.model(
            frame_bgr,
            conf=self.threshold,
            classes=[0],  # COCO class 0 = person
            verbose=False,
            device=self.device,
        )[0]

        persons = []
        boxes = results.boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().tolist()
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < min_area:
                continue
            persons.append({
                "box": box,
                "confidence": float(boxes.conf[i].cpu()),
            })
        # Explicitly release GPU tensors held by Results object
        del boxes, results
        return persons
