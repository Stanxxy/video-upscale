"""MLP / feature models for legacy advanced tracker."""
"""
Online learning tracker with DINOv2 features and MLP classifiers.
Ported from bjj-pose-estimation/bjj_pipeline/advanced_tracking.py

Adapted for Mac M4 Max: MPS device support.
"""
import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tracking_pipeline.device import get_device, empty_cache

logger = logging.getLogger("bjj_tracking")


def calculate_torso_angle(keypoints):
    """Calculate torso angle (0=vertical/upright, 90=horizontal/on ground)."""
    if keypoints is None or len(keypoints) < 13:
        return None

    ls, rs = keypoints[5], keypoints[6]
    lh, rh = keypoints[11], keypoints[12]

    if ls[0] <= 0 or rs[0] <= 0 or lh[0] <= 0 or rh[0] <= 0:
        return None

    mid_shoulder = (ls[:2] + rs[:2]) / 2
    mid_hip = (lh[:2] + rh[:2]) / 2

    vec = mid_shoulder - mid_hip
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None

    angle = np.degrees(np.arccos(np.abs(vec[1]) / norm))
    return angle


class TorchMLP(nn.Module):
    """Binary classifier: target athlete vs other."""

    def __init__(self, input_dim=384, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class FeatureExtractor:
    """Extracts DINOv2 features from specific body joint patches."""

    def __init__(self, device=None, model_type="dinov2_vits14", dino_model=None):
        if device is None:
            device = get_device()
        self.device = device
        if dino_model is not None:
            self.model = dino_model
            logger.info(f"Using shared DINOv2 on {device}")
        else:
            logger.info(f"Loading DINOv2 ({model_type}) on {device}...")
            self.model = torch.hub.load("facebookresearch/dinov2", model_type).to(device)
            self.model.eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.patch_size = 64

    def extract_joint_features(self, frame, keypoints, joints_to_track=None):
        """
        Extract features for specified joints.
        Returns: dict {joint_idx: torch.Tensor(1, 384) on device}
        """
        if joints_to_track is None:
            joints_to_track = [5, 6, 11, 12, 13, 14]

        features = {}
        H, W = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        batch_patches = []
        valid_indices = []

        for j_idx in joints_to_track:
            if j_idx >= len(keypoints):
                continue
            x, y = keypoints[j_idx][:2]
            conf = keypoints[j_idx][2] if len(keypoints[j_idx]) > 2 else 1.0

            if conf < 0.3 or x <= 0 or y <= 0 or x >= W or y >= H:
                continue

            x1 = max(0, int(x - self.patch_size // 2))
            y1 = max(0, int(y - self.patch_size // 2))
            x2 = min(W, int(x + self.patch_size // 2))
            y2 = min(H, int(y + self.patch_size // 2))

            patch = frame_rgb[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            patch_resized = cv2.resize(patch, (224, 224))
            patch_norm = (patch_resized - self.mean) / self.std
            batch_patches.append(patch_norm.transpose(2, 0, 1))
            valid_indices.append(j_idx)

        if not batch_patches:
            return {}

        batch_np = np.array(batch_patches)
        batch_tensor = torch.from_numpy(batch_np).to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch_tensor)

        for i, j_idx in enumerate(valid_indices):
            features[j_idx] = embeddings[i].unsqueeze(0)

        return features


class TorchOnlineClassifier:
    """Per-track online learner using PyTorch MLPs on device."""

    def __init__(self, track_id, joints_to_track=None, device=None):
        if joints_to_track is None:
            joints_to_track = [5, 6, 11, 12, 13, 14]
        if device is None:
            device = get_device()

        self.track_id = track_id
        self.joints_to_track = joints_to_track
        self.device = device

        self.classifiers = {
            j: TorchMLP(input_dim=384, hidden_dim=64).to(device)
            for j in joints_to_track
        }
        self.optimizers = {
            j: optim.Adam(self.classifiers[j].parameters(), lr=0.001)
            for j in joints_to_track
        }
        self.criterion = nn.CrossEntropyLoss()
        self.history = {j: {"X": [], "y": []} for j in joints_to_track}
        self.buffer_size = 200
        self.is_trained = False

    def update(self, features, label):
        """Add new sample. label: 1=target, 0=other."""
        for j_idx, feat in features.items():
            if j_idx in self.history:
                # Keep history on CPU to avoid long-lived device tensor retention.
                self.history[j_idx]["X"].append(feat.detach().cpu().clone())
                self.history[j_idx]["y"].append(int(label))
                if len(self.history[j_idx]["X"]) > self.buffer_size:
                    self.history[j_idx]["X"].pop(0)
                    self.history[j_idx]["y"].pop(0)

    def fit(self, epochs=5, snapshot=None):
        """Train classifiers on accumulated history."""
        source = snapshot if snapshot is not None else self.history
        trained_count = 0
        for j_idx, clf in self.classifiers.items():
            X_list = source[j_idx]["X"]
            y_list = source[j_idx]["y"]
            if len(X_list) < 10:
                continue

            # Build batches locally, then release right after each joint fit.
            X_batch = torch.cat(X_list, dim=0).to(self.device)
            y_batch = torch.tensor(y_list, dtype=torch.long, device=self.device)

            optimizer = self.optimizers[j_idx]
            clf.train()
            outputs = None
            loss = None
            for _ in range(epochs):
                optimizer.zero_grad()
                outputs = clf(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            trained_count += 1
            del X_batch, y_batch
            if outputs is not None:
                del outputs
            if loss is not None:
                del loss

        self.is_trained = trained_count > 0
        return self.is_trained

    def get_training_snapshot(self):
        """Return immutable CPU snapshot of training history."""
        snap = {}
        for j_idx in self.joints_to_track:
            snap[j_idx] = {
                "X": [x.clone() for x in self.history[j_idx]["X"]],
                "y": list(self.history[j_idx]["y"]),
            }
        return snap

    def export_model_state(self):
        """Export only model weights (optimizer state intentionally excluded)."""
        return {
            j_idx: clf.state_dict()
            for j_idx, clf in self.classifiers.items()
        }

    def load_model_state(self, state_by_joint):
        """Load model weights produced by async retraining worker."""
        for j_idx, state in state_by_joint.items():
            if j_idx in self.classifiers:
                self.classifiers[j_idx].load_state_dict(state)
        self.is_trained = True

    def predict_score(self, features):
        """Returns average probability of being the target athlete."""
        if not self.is_trained:
            return 0.5
        scores = []
        with torch.no_grad():
            for j_idx, feat in features.items():
                if j_idx in self.classifiers:
                    clf = self.classifiers[j_idx]
                    clf.eval()
                    output = clf(feat.to(self.device))
                    prob = F.softmax(output, dim=1)[0, 1].item()
                    scores.append(prob)
        if not scores:
            return 0.5
        return np.mean(scores)


def _train_models_from_snapshot(track_payloads, joints, epochs):
    """Background trainer entrypoint (CPU)."""
    results = {}
    for payload in track_payloads:
        track_id = payload["track_id"]
        snapshot = payload["snapshot"]
        state_by_joint = payload["state_by_joint"]

        trainer = TorchOnlineClassifier(track_id, joints_to_track=joints, device=torch.device("cpu"))
        for j_idx, state in state_by_joint.items():
            if j_idx in trainer.classifiers:
                trainer.classifiers[j_idx].load_state_dict(state)

        trainer.fit(epochs=epochs, snapshot=snapshot)
        results[track_id] = trainer.export_model_state()
        del trainer
    gc.collect()
    return results


class AsyncRetrainWorker:
    """Latest-only async retrain scheduler with atomic result handoff."""

    def __init__(self, enabled=True, epochs=2):
        self.enabled = enabled
        self.epochs = epochs
        self._executor = ThreadPoolExecutor(max_workers=1) if enabled else None
        self._future = None
        self._pending_payload = None
        self._lock = threading.Lock()

    def schedule(self, payload):
        if not self.enabled:
            return False
        with self._lock:
            if self._future is not None and not self._future.done():
                # Keep only latest snapshot while worker is busy.
                self._pending_payload = payload
                return False
            self._future = self._executor.submit(
                _train_models_from_snapshot,
                payload["track_payloads"],
                payload["joints"],
                payload["epochs"],
            )
            return True

    def poll(self):
        """Return completed result dict or None; auto-schedules pending latest payload."""
        if not self.enabled:
            return None
        with self._lock:
            if self._future is None or not self._future.done():
                return None
            try:
                result = self._future.result()
            except Exception as e:
                logger.warning("Async retrain worker failed: %s", e)
                result = None
            self._future = None
            if self._pending_payload is not None:
                pending = self._pending_payload
                self._pending_payload = None
                self._future = self._executor.submit(
                    _train_models_from_snapshot,
                    pending["track_payloads"],
                    pending["joints"],
                    pending["epochs"],
                )
            return result

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

