"""
Video I/O and scene detection (cuts/fades).
Ported from bjj-pose-estimation/bjj_pipeline/core/video_io.py
"""
import cv2
import numpy as np
from collections import deque


class VideoIO:
    """Handles video metadata and robust scene detection (cuts/fades)."""

    def __init__(self, video_path, window_size=20):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.prev_hist = None

        # Moving window buffers for fade detection
        self.window_size = window_size
        self.luma_history = deque(maxlen=window_size)
        self.edge_history = deque(maxlen=window_size)
        self.fade_state = "ACTIVE"

    def detect_scene_change(self, frame, threshold=0.6):
        """Detect scene cuts using histogram correlation."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        is_cut = False
        if self.prev_hist is not None:
            score = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
            if score < threshold:
                print(f"SCENE CUT DETECTED! Score: {score:.4f} < {threshold}")
                is_cut = True

        self.prev_hist = hist
        return is_cut

    def update_fade_state(self, frame):
        """Detect fades using moving window of luma and edge density."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        luma = np.mean(gray)
        edges = np.sum(cv2.Canny(gray, 100, 200) > 0)

        self.luma_history.append(luma)
        self.edge_history.append(edges)

        if len(self.luma_history) < self.luma_history.maxlen:
            return "WARMING_UP"

        luma_trend = self.luma_history[-1] - self.luma_history[0]
        max_edge = max(self.edge_history) if max(self.edge_history) > 0 else 1
        edge_drop_ratio = self.edge_history[-1] / max_edge

        if luma_trend < -20.0 and edge_drop_ratio < 0.3:
            self.fade_state = "FADING_OUT"
        elif luma < 15.0 and edge_drop_ratio < 0.1:
            self.fade_state = "BLACKOUT"
        elif self.fade_state == "BLACKOUT" and luma > 25.0 and edge_drop_ratio > 0.2:
            self.fade_state = "RECOVERING"
        elif (self.fade_state == "RECOVERING"
              and luma_trend > -5.0 and edge_drop_ratio > 0.5):
            self.fade_state = "ACTIVE"
        elif self.fade_state == "FADING_OUT" and luma_trend > 0:
            self.fade_state = "ACTIVE"

        return self.fade_state

    def release(self):
        self.cap.release()
