"""
DINOv2-based athlete identity management and re-identification.
Ported from bjj-pose-estimation/bjj_pipeline/models/identity_manager.py

Supports:
- Full-body DINOv2 embeddings (global)
- Multi-bin body part features (head/torso/legs from keypoints)
- HSV color histograms (gi color constraint)
"""
import cv2
import numpy as np
import torch

from device import get_device


class IdentityManager:
    """Manages athlete identities using DINOv2 embeddings and color histograms."""

    def __init__(self, device=None, dino_model=None):
        if device is None:
            device = get_device()
        self.device = device
        if dino_model is not None:
            self.model = dino_model
            print(f"[identity] Using shared DINOv2 on {device}")
        else:
            print(f"[identity] Loading DINOv2 vits14 on {device}...")
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.model.to(device)
            self.model.eval()

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        # Gallery: {athlete_id: {'full': feat, 'torso': feat, 'hist': hist}}
        self.gallery = {}

    def _preprocess_crop(self, crop):
        """Resize and normalize crop for DINOv2."""
        if crop.size == 0:
            return None
        resized = cv2.resize(crop, (224, 224))
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def extract_global_features(self, frame, box=None, mask=None):
        """Extract DINOv2 features from a box or mask region."""
        if mask is not None:
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                return None
            box = [x_indices.min(), y_indices.min(),
                   x_indices.max(), y_indices.max()]

        if box is None:
            return None

        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            crop = crop * mask_crop[:, :, np.newaxis]

        tensor = self._preprocess_crop(crop)
        if tensor is None:
            return None

        with torch.no_grad():
            features = self.model(tensor)

        return torch.nn.functional.normalize(features, dim=1).cpu()

    def extract_regional_features(self, frame, keypoints, scores=None):
        """
        Extract features from body parts based on keypoints.

        Args:
            keypoints: (17, 2) or (17, 3) array
            scores: (17,) confidence array (if keypoints is (17, 2))
        """
        bins = {}
        h, w = frame.shape[:2]

        if keypoints.shape[-1] == 3:
            pts = keypoints[:, :2]
            confs = keypoints[:, 2]
        elif keypoints.shape[-1] == 2:
            pts = keypoints
            confs = scores if scores is not None else np.ones(len(keypoints))
        else:
            return {}

        parts = {
            "head": [0, 1, 2, 3, 4],
            "torso": [5, 6, 11, 12],
            "legs": [13, 14, 15, 16],
        }

        for part_name, indices in parts.items():
            valid_pts = []
            for idx in indices:
                if idx < len(confs) and confs[idx] > 0.3:
                    valid_pts.append(pts[idx])

            if len(valid_pts) > 0:
                valid_pts = np.array(valid_pts)
                x_min, y_min = np.min(valid_pts, axis=0)
                x_max, y_max = np.max(valid_pts, axis=0)

                pad = 20
                x1 = max(0, int(x_min - pad))
                y1 = max(0, int(y_min - pad))
                x2 = min(w, int(x_max + pad))
                y2 = min(h, int(y_max + pad))

                crop = frame[y1:y2, x1:x2]
                feat = self._preprocess_crop(crop)
                if feat is not None:
                    with torch.no_grad():
                        f = self.model(feat)
                    bins[part_name] = torch.nn.functional.normalize(f, dim=1).cpu()

        return bins

    def compute_color_histogram(self, frame, mask=None, box=None):
        """Compute HSV histogram for gi color matching."""
        if mask is None and box is not None:
            x1, y1, x2, y2 = map(int, box)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1

        if mask is None or np.sum(mask) == 0:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], mask.astype(np.uint8), [8, 4], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def update_gallery(self, obj_id, frame, mask=None, box=None,
                       keypoints=None, scores=None):
        """Update the identity gallery for an athlete."""
        entry = {}
        entry["full"] = self.extract_global_features(frame, box, mask)
        if keypoints is not None:
            bins = self.extract_regional_features(frame, keypoints, scores)
            entry.update(bins)
        entry["hist"] = self.compute_color_histogram(frame, mask, box)
        self.gallery[obj_id] = entry

    def query_identity_scores(self, frame, box, mask=None,
                              keypoints=None, scores=None):
        """
        Return similarity scores for all gallery identities.
        Scoring: 0.5×global_dino + 0.3×color + 0.2×torso
        """
        query_full = self.extract_global_features(frame, box, mask)
        query_hist = self.compute_color_histogram(frame, mask, box)
        query_bins = (
            self.extract_regional_features(frame, keypoints, scores)
            if keypoints is not None else {}
        )

        results = {}
        for ref_id, gallery_entry in self.gallery.items():
            score = 0.0
            weights = 0.0

            # Global DINOv2
            if query_full is not None and gallery_entry.get("full") is not None:
                sim = torch.nn.functional.cosine_similarity(
                    query_full, gallery_entry["full"]
                ).item()
                sim = (sim + 1) / 2
                score += 0.5 * sim
                weights += 0.5

            # Color histogram
            if query_hist is not None and gallery_entry.get("hist") is not None:
                sim = cv2.compareHist(
                    query_hist, gallery_entry["hist"], cv2.HISTCMP_CORREL
                )
                sim = max(0, sim)
                score += 0.3 * sim
                weights += 0.3

            # Torso features
            if "torso" in query_bins and "torso" in gallery_entry:
                sim = torch.nn.functional.cosine_similarity(
                    query_bins["torso"], gallery_entry["torso"]
                ).item()
                sim = (sim + 1) / 2
                score += 0.2 * sim
                weights += 0.2

            results[ref_id] = score / weights if weights > 0 else 0

        return results

    def query_identity(self, frame, box, mask=None,
                       keypoints=None, scores=None):
        """Identify a person against the gallery. Returns (best_id, best_score)."""
        scores_map = self.query_identity_scores(
            frame, box, mask, keypoints, scores
        )
        if not scores_map:
            return -1, -1.0
        best_id = max(scores_map, key=scores_map.get)
        return best_id, scores_map[best_id]
