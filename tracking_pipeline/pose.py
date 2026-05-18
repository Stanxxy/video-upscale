"""
RTMPose keypoint estimation from bounding boxes.
Used alongside YOLO26 (detection only) and SAM2 (mask propagation).

Uses rtmlib for top-down pose estimation.
S8: Auto-selects GPU (onnxruntime CUDA EP) when available, falls back to CPU.
On Mac M4 Max (MPS) force_cpu remains True — onnxruntime MPS EP is not stable.
"""
import numpy as np
import torch

from .device import get_device


class PoseEstimator:
    """Top-down pose estimator using RTMPose via rtmlib."""

    def __init__(self, device=None):
        from rtmlib import Body

        # S8: Use CUDA EP on Blackwell/CUDA GPUs; stay on CPU for MPS/CPU-only.
        # onnxruntime-gpu falls back to CPU EP gracefully when CUDA EP is absent.
        _cuda_available = torch.cuda.is_available()
        _force_cpu = not _cuda_available  # True on M4 Max (MPS), False on Spark CUDA

        if device is None:
            device = get_device(force_cpu=_force_cpu)

        backend = "onnxruntime"
        device_str = "cuda" if _cuda_available else "cpu"

        print(f"[pose] Loading RTMPose on {device_str} (force_cpu={_force_cpu})...")
        try:
            self.body = Body(
                mode="lightweight",  # RTMPose-m, good balance of speed/accuracy
                backend=backend,
                device=device_str,
            )
        except Exception as e:
            if device_str != "cpu":
                print(f"[pose] RTMPose on {device_str} failed ({e}), falling back to CPU...")
                self.body = Body(
                    mode="lightweight",
                    backend=backend,
                    device="cpu",
                )
            else:
                raise
        print(f"[pose] RTMPose loaded on {device_str}.")

    def estimate(self, frame, box=None):
        """
        Estimate keypoints for a single person.

        Args:
            frame: BGR frame (numpy array)
            box: [x1, y1, x2, y2] bounding box (optional, for cropping)

        Returns:
            keypoints: (17, 2) array of [x, y] coordinates (in full-frame space)
            scores: (17,) array of confidence scores
        """
        if box is not None:
            # Crop to the bounding box with padding so RTMPose only sees
            # the target athlete, not spectators in the background.
            h, w = frame.shape[:2]
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            pad_x = int(bw * 0.25)
            pad_y = int(bh * 0.25)
            cx1 = max(0, int(box[0]) - pad_x)
            cy1 = max(0, int(box[1]) - pad_y)
            cx2 = min(w, int(box[2]) + pad_x)
            cy2 = min(h, int(box[3]) + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]

            # Degenerate box / clamping can yield empty slices; rtmlib YOLOX
            # preprocess divides by img width and crashes on zero-sized crops.
            if crop.ndim < 2 or crop.shape[0] < 1 or crop.shape[1] < 1:
                return np.zeros((17, 2)), np.zeros(17)

            keypoints, scores = self.body(crop)

            if keypoints is None or len(keypoints) == 0:
                return np.zeros((17, 2)), np.zeros(17)

            # Translate keypoints back to full-frame coordinates
            for i in range(len(keypoints)):
                keypoints[i][:, 0] += cx1
                keypoints[i][:, 1] += cy1

            if len(keypoints) > 1:
                # Multiple detections inside the crop: pick the one with
                # most keypoints inside the original bounding box
                best_idx = 0
                best_inside = -1
                for i, kp in enumerate(keypoints):
                    inside = np.sum(
                        (kp[:, 0] >= box[0]) & (kp[:, 0] <= box[2]) &
                        (kp[:, 1] >= box[1]) & (kp[:, 1] <= box[3])
                    )
                    if inside > best_inside:
                        best_inside = inside
                        best_idx = i
                return keypoints[best_idx][:, :2], scores[best_idx]

            return keypoints[0][:, :2], scores[0]

        # No box: run on full frame
        keypoints, scores = self.body(frame)

        if keypoints is None or len(keypoints) == 0:
            return np.zeros((17, 2)), np.zeros(17)

        return keypoints[0][:, :2], scores[0]

