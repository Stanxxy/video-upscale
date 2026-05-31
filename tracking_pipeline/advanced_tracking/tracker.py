"""Legacy BJJ advanced tracker orchestration."""
from tracking_pipeline.advanced_tracking.models import (
    AsyncRetrainWorker,
    FeatureExtractor,
    TorchOnlineClassifier,
    calculate_torso_angle,
    get_device,
)


class BJJAdvancedTracker:
    """
    Orchestrates tracking using geometry (IOU) + visual (online MLP) matching.
    Cost = (1-visual_weight)*IOU + visual_weight*MLP_score - pose_penalty
    """

    def __init__(
        self,
        max_missed_frames=30,
        visual_weight=0.7,
        device=None,
        async_retrain_enabled=True,
        retrain_interval=45,
        retrain_epochs=2,
    ):
        if device is None:
            device = get_device()
        self.tracks = {}
        self.next_id = 1
        self.max_missed_frames = max_missed_frames
        self.visual_weight = visual_weight
        self.device = device
        self.joints = [5, 6, 11, 12, 13, 14]
        self.retrain_interval = max(1, int(retrain_interval))
        self.retrain_epochs = max(1, int(retrain_epochs))
        self.retrain_worker = AsyncRetrainWorker(
            enabled=async_retrain_enabled,
            epochs=self.retrain_epochs,
        )

    def initialize(self, detections, all_features):
        """Initialize tracks from first confident frame."""
        logger.info(f"Initializing tracks with {len(detections)} detections")
        self.tracks = {}

        for i, (det, feats) in enumerate(zip(detections, all_features)):
            track_id = self.next_id
            self.next_id += 1

            clf = TorchOnlineClassifier(track_id, self.joints, device=self.device)
            clf.update(feats, 1)

            self.tracks[track_id] = {
                "box": det["box"],
                "keypoints": det.get("keypoints"),
                "classifier": clf,
                "missed": 0,
                "active": True,
                "history_features": [feats],
            }

        # Cross-train: A's features are negative for B and vice versa
        track_ids = list(self.tracks.keys())
        if len(track_ids) >= 2:
            id_a, id_b = track_ids[0], track_ids[1]
            self.tracks[id_a]["classifier"].update(
                self.tracks[id_b]["history_features"][0], 0
            )
            self.tracks[id_b]["classifier"].update(
                self.tracks[id_a]["history_features"][0], 0
            )
            self.tracks[id_a]["classifier"].fit()
            self.tracks[id_b]["classifier"].fit()

    def update(self, detections, all_features, frame_idx=0):
        """Update tracks with new detections using cost matrix + assignment."""
        self._poll_retrain_results()

        if not self.tracks:
            if all_features:
                self.initialize(detections, all_features)
            return self.tracks

        track_ids = list(self.tracks.keys())

        # Build cost matrix
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            last_box = track["box"]

            for j, (det, feats) in enumerate(zip(detections, all_features)):
                iou_score = self._compute_iou(last_box, det["box"])
                visual_score = (
                    track["classifier"].predict_score(feats)
                    if feats else 0.5
                )

                # Pose penalty: upright = likely referee
                pose_penalty = 0.0
                kpts = det.get("keypoints")
                if kpts is not None:
                    angle = calculate_torso_angle(kpts)
                    if angle is not None and angle < 20:
                        pose_penalty = 0.5

                total = (
                    (1 - self.visual_weight) * iou_score
                    + self.visual_weight * visual_score
                    - pose_penalty
                )
                if pose_penalty > 0 and visual_score < 0.7:
                    total = -1.0

                cost_matrix[i, j] = total

        # Greedy assignment (sorted by score, descending)
        assignments = {}
        assigned_dets = set()
        matches = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                matches.append((i, j, cost_matrix[i, j]))
        matches.sort(key=lambda x: x[2], reverse=True)

        for t_idx, d_idx, score in matches:
            if t_idx not in assignments and d_idx not in assigned_dets:
                if score > 0.4:
                    assignments[t_idx] = d_idx
                    assigned_dets.add(d_idx)

        # Update matched tracks
        matched_track_ids = []
        for t_idx, tid in enumerate(track_ids):
            track = self.tracks[tid]
            if t_idx in assignments:
                d_idx = assignments[t_idx]
                det = detections[d_idx]

                track["box"] = det["box"]
                track["keypoints"] = det.get("keypoints")
                track["missed"] = 0
                track["active"] = True

                if all_features[d_idx]:
                    feats = all_features[d_idx]
                    track["classifier"].update(feats, 1)
                    for other_d_idx, other_feats in enumerate(all_features):
                        if other_d_idx != d_idx and other_feats:
                            track["classifier"].update(other_feats, 0)

                matched_track_ids.append(tid)
            else:
                track["missed"] += 1
                if track["missed"] > self.max_missed_frames:
                    track["active"] = False

        # Retrain classifiers asynchronously at a lower cadence.
        if matched_track_ids and frame_idx % self.retrain_interval == 0:
            self._schedule_retrain(matched_track_ids)

        return self.tracks

    def _schedule_retrain(self, matched_track_ids):
        track_payloads = []
        for tid in matched_track_ids:
            track = self.tracks.get(tid)
            if track is None:
                continue
            clf = track["classifier"]
            track_payloads.append(
                {
                    "track_id": tid,
                    "snapshot": clf.get_training_snapshot(),
                    "state_by_joint": clf.export_model_state(),
                }
            )
        if not track_payloads:
            return
        self.retrain_worker.schedule(
            {
                "track_payloads": track_payloads,
                "joints": self.joints,
                "epochs": self.retrain_epochs,
            }
        )

    def _poll_retrain_results(self):
        result = self.retrain_worker.poll()
        if not result:
            return
        for tid, state_by_joint in result.items():
            track = self.tracks.get(tid)
            if track is None:
                continue
            track["classifier"].load_model_state(state_by_joint)
        gc.collect()

    def close(self):
        self.retrain_worker.shutdown()

    @staticmethod
    def _compute_iou(box_a, box_b):
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)
