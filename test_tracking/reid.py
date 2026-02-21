"""
DINOv2 + Color Histogram Re-Identification for BJJ athletes.

Maintains an identity gallery built from user-verified initial boxes.
When tracking is lost or identities may have swapped, queries the gallery
to re-assign correct track IDs.

Simplified port of bjj-pose-estimation/bjj_pipeline/models/identity_manager.py.
"""
import cv2
import numpy as np
import torch

from device import get_device, get_dtype


class ReIDGallery:
    """
    Maintains a gallery of athlete appearances for re-identification.

    Uses:
    - DINOv2 vits14 (384-dim embeddings) for visual similarity
    - HSV color histograms for gi color matching
    """

    def __init__(self, device=None):
        if device is None:
            device = get_device()
        self.device = device

        print("[reid] Loading DINOv2 vits14...")
        self.model = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14', verbose=False
        )
        self.model.to(device)
        self.model.eval()

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        # Gallery: {track_id: {"features": Tensor, "histogram": ndarray}}
        self.gallery = {}

    def _extract_features(self, frame_rgb, box):
        """Extract DINOv2 embedding from a crop defined by box."""
        x1, y1, x2, y2 = [int(c) for c in box]
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        resized = cv2.resize(crop, (224, 224))
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std

        with torch.no_grad():
            features = self.model(tensor)

        return torch.nn.functional.normalize(features, dim=1).cpu()

    def _compute_histogram(self, frame_rgb, box):
        """Compute HSV color histogram for gi color matching."""
        x1, y1, x2, y2 = [int(c) for c in box]
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def build_gallery(self, frame_rgb, verified_boxes):
        """
        Build the identity gallery from user-verified boxes.

        Args:
            frame_rgb: RGB frame (numpy array)
            verified_boxes: List of {"box": [x1,y1,x2,y2], "track_id": int}
        """
        self.gallery.clear()

        for det in verified_boxes:
            track_id = det["track_id"]
            box = det["box"]

            features = self._extract_features(frame_rgb, box)
            histogram = self._compute_histogram(frame_rgb, box)

            self.gallery[track_id] = {
                "features": features,
                "histogram": histogram,
            }
            print(f"[reid] Gallery: ID {track_id} registered "
                  f"(features={'OK' if features is not None else 'FAIL'}, "
                  f"histogram={'OK' if histogram is not None else 'FAIL'})")

    def query(self, frame_rgb, candidate_boxes):
        """
        Match candidate boxes against the gallery.

        Args:
            frame_rgb: RGB frame
            candidate_boxes: List of [x1, y1, x2, y2] boxes to identify

        Returns:
            List of (track_id, confidence) for each candidate, assigned via
            greedy best-match. Unmatched candidates get track_id=-1.
        """
        if not self.gallery or not candidate_boxes:
            return [(-1, 0.0)] * len(candidate_boxes)

        # Compute score matrix: gallery IDs x candidates
        gallery_ids = sorted(self.gallery.keys())
        score_matrix = np.zeros((len(gallery_ids), len(candidate_boxes)))

        for g_idx, gal_id in enumerate(gallery_ids):
            gal_entry = self.gallery[gal_id]
            for c_idx, box in enumerate(candidate_boxes):
                score_matrix[g_idx, c_idx] = self._compute_score(
                    frame_rgb, box, gal_entry
                )

        # Greedy assignment (maximize score)
        assignments = {}  # candidate_idx -> (track_id, score)
        used_gallery = set()
        used_candidates = set()

        # Flatten and sort all (gallery, candidate) pairs by score
        pairs = []
        for g_idx in range(len(gallery_ids)):
            for c_idx in range(len(candidate_boxes)):
                pairs.append((g_idx, c_idx, score_matrix[g_idx, c_idx]))
        pairs.sort(key=lambda x: x[2], reverse=True)

        for g_idx, c_idx, score in pairs:
            if g_idx in used_gallery or c_idx in used_candidates:
                continue
            if score > 0.4:  # Minimum threshold
                assignments[c_idx] = (gallery_ids[g_idx], score)
                used_gallery.add(g_idx)
                used_candidates.add(c_idx)

        # Build result list
        results = []
        for c_idx in range(len(candidate_boxes)):
            if c_idx in assignments:
                results.append(assignments[c_idx])
            else:
                results.append((-1, 0.0))

        return results

    def check_identity_swap(self, frame_rgb, current_boxes):
        """
        Check if tracked athlete identities look swapped.

        Args:
            frame_rgb: RGB frame
            current_boxes: Dict {track_id: [x1,y1,x2,y2]}

        Returns:
            True if identities appear swapped, False otherwise.
        """
        if len(current_boxes) < 2 or len(self.gallery) < 2:
            return False

        track_ids = sorted(current_boxes.keys())
        if len(track_ids) < 2:
            return False

        id_a, id_b = track_ids[0], track_ids[1]

        # Score each box against each gallery entry
        score_a_as_a = self._compute_score(
            frame_rgb, current_boxes[id_a], self.gallery.get(id_a, {})
        )
        score_a_as_b = self._compute_score(
            frame_rgb, current_boxes[id_a], self.gallery.get(id_b, {})
        )
        score_b_as_a = self._compute_score(
            frame_rgb, current_boxes[id_b], self.gallery.get(id_a, {})
        )
        score_b_as_b = self._compute_score(
            frame_rgb, current_boxes[id_b], self.gallery.get(id_b, {})
        )

        # Swap detected if A looks more like B and B looks more like A
        is_swapped = (score_a_as_b > score_a_as_a and
                      score_b_as_a > score_b_as_b)

        if is_swapped:
            print(f"[reid] SWAP DETECTED: "
                  f"ID{id_a} looks like ID{id_b} ({score_a_as_b:.2f} vs {score_a_as_a:.2f}), "
                  f"ID{id_b} looks like ID{id_a} ({score_b_as_a:.2f} vs {score_b_as_b:.2f})")

        return is_swapped

    def _compute_score(self, frame_rgb, box, gallery_entry):
        """Compute weighted similarity between a box and a gallery entry."""
        if not gallery_entry:
            return 0.0

        score = 0.0
        weights = 0.0

        # DINOv2 similarity (50% weight)
        query_feat = self._extract_features(frame_rgb, box)
        gal_feat = gallery_entry.get("features")
        if query_feat is not None and gal_feat is not None:
            sim = torch.nn.functional.cosine_similarity(
                query_feat, gal_feat
            ).item()
            sim = (sim + 1) / 2  # Normalize from [-1,1] to [0,1]
            score += 0.5 * sim
            weights += 0.5

        # Color histogram similarity (50% weight since no keypoints)
        query_hist = self._compute_histogram(frame_rgb, box)
        gal_hist = gallery_entry.get("histogram")
        if query_hist is not None and gal_hist is not None:
            sim = cv2.compareHist(query_hist, gal_hist, cv2.HISTCMP_CORREL)
            sim = max(0, sim)
            score += 0.5 * sim
            weights += 0.5

        return score / weights if weights > 0 else 0.0
