"""
Temporal smoothing filters for bounding boxes and keypoints.

Uses the 1-Euro adaptive low-pass filter (Casiez et al., CHI 2012) which
smooths aggressively when the signal is slow and responds quickly during
fast movement — ideal for combat sports where athletes alternate between
stillness and explosive motion.
"""

import numpy as np


class OneEuroFilter:
    """Adaptive low-pass filter for real-time signal smoothing.

    Parameters
    ----------
    min_cutoff : float
        Minimum cutoff frequency (Hz). Lower = more smoothing when stationary.
    beta : float
        Speed coefficient. Higher = less lag during fast motion.
    d_cutoff : float
        Cutoff frequency for derivative estimation.
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        """Filter a value.

        Parameters
        ----------
        x : array-like
            Values to filter.
        t : float, optional
            Timestamp in seconds. If None, assumes constant dt = 1/30.

        Returns
        -------
        numpy.ndarray
            Filtered values (same shape as input).
        """
        x = np.asarray(x, dtype=np.float64)

        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t if t is not None else 0.0
            return x.copy()

        dt = (t - self.t_prev) if t is not None else (1.0 / 30.0)
        dt = max(dt, 1e-6)

        # Estimate derivative (speed)
        dx = (x - self.x_prev) / dt

        # Smooth the derivative
        a_d = self._smoothing_factor(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Smooth the signal
        a = self._smoothing_factor(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t if t is not None else (self.t_prev + dt)

        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class BoxSmoother:
    """Per-object bounding box smoother with gradual YOLO26 re-anchoring.

    Applies a OneEuro filter to the 4 box coordinates independently.
    When YOLO26 overrides SAM2, blends over N frames instead of jumping.

    Parameters
    ----------
    min_cutoff : float
        OneEuro min_cutoff for box coords. Higher = more responsive.
    beta : float
        OneEuro beta for box coords.
    blend_frames : int
        Number of frames to blend during YOLO26 re-anchoring.
    """

    def __init__(self, min_cutoff=1.7, beta=0.01, blend_frames=5):
        self.filters = {}        # {obj_id: OneEuroFilter}
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.blend_frames = blend_frames
        self._blend_state = {}   # {obj_id: {"target", "remaining", "start"}}

    def smooth(self, obj_id, box, timestamp=None):
        """Smooth a bounding box.

        Parameters
        ----------
        obj_id : int
            Track ID.
        box : list
            [x1, y1, x2, y2].
        timestamp : float, optional
            Frame timestamp in seconds.

        Returns
        -------
        list
            Smoothed [x1, y1, x2, y2].
        """
        if obj_id not in self.filters:
            self.filters[obj_id] = OneEuroFilter(
                min_cutoff=self.min_cutoff, beta=self.beta,
            )

        box_arr = np.array(box, dtype=np.float64)

        # If in a blend transition, interpolate toward target
        if obj_id in self._blend_state:
            state = self._blend_state[obj_id]
            if state["remaining"] > 0:
                alpha = 1.0 - (state["remaining"] / self.blend_frames)
                box_arr = ((1.0 - alpha) * np.array(state["start"])
                           + alpha * np.array(state["target"]))
                state["remaining"] -= 1
                if state["remaining"] <= 0:
                    del self._blend_state[obj_id]

        smoothed = self.filters[obj_id](box_arr, timestamp)
        return smoothed.tolist()

    def start_blend(self, obj_id, current_box, target_box):
        """Initiate gradual blend from current_box to target_box.

        Called when YOLO26 re-anchoring overrides SAM2 to avoid sudden jumps.
        """
        self._blend_state[obj_id] = {
            "start": list(current_box),
            "target": list(target_box),
            "remaining": self.blend_frames,
        }

    def reset(self, obj_id=None):
        if obj_id is not None:
            self.filters.pop(obj_id, None)
            self._blend_state.pop(obj_id, None)
        else:
            self.filters.clear()
            self._blend_state.clear()


class KeypointSmoother:
    """Per-object, per-joint keypoint smoother with confidence filtering.

    Features:
    - OneEuro filter per joint per object
    - Confidence gating: low-confidence keypoints use last known position
    - Velocity clamping: rejects physically impossible jumps
    - Confidence-weighted blending: uncertain keypoints pulled toward prediction

    Parameters
    ----------
    min_cutoff : float
        OneEuro min_cutoff for keypoints (lower = more smoothing).
    beta : float
        OneEuro beta (higher = faster response to movement).
    min_confidence : float
        Minimum confidence to trust a raw keypoint.
    max_velocity : float
        Max pixels/frame before velocity clamping kicks in.
    num_joints : int
        Number of keypoints (COCO = 17).
    """

    def __init__(self, min_cutoff=0.8, beta=0.015, min_confidence=0.3,
                 max_velocity=200.0, num_joints=17):
        self.filters = {}       # {obj_id: {joint_idx: OneEuroFilter}}
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.min_confidence = min_confidence
        self.max_velocity = max_velocity
        self.num_joints = num_joints
        self._last_good = {}    # {obj_id: (N, 2) last good positions}
        self._last_scores = {}  # {obj_id: (N,) last scores}

    def smooth(self, obj_id, keypoints, scores, timestamp=None):
        """Smooth keypoints with confidence weighting.

        Parameters
        ----------
        obj_id : int
            Track ID.
        keypoints : numpy.ndarray
            (N, 2) array of [x, y] per joint.
        scores : numpy.ndarray
            (N,) confidence per joint.
        timestamp : float, optional
            Frame timestamp in seconds.

        Returns
        -------
        smoothed_kpts : numpy.ndarray
            (N, 2) smoothed keypoints.
        adjusted_scores : numpy.ndarray
            (N,) scores with suppressed low-confidence joints set to 0.
        """
        kpts = np.array(keypoints, dtype=np.float64)
        sc = np.array(scores, dtype=np.float64)

        if obj_id not in self.filters:
            self.filters[obj_id] = {}
            self._last_good[obj_id] = kpts.copy()
            self._last_scores[obj_id] = sc.copy()

        n_joints = min(self.num_joints, len(kpts))
        smoothed = np.zeros_like(kpts)
        adjusted_scores = sc.copy()

        for j in range(n_joints):
            if j not in self.filters[obj_id]:
                self.filters[obj_id][j] = OneEuroFilter(
                    min_cutoff=self.min_cutoff, beta=self.beta,
                )

            filt = self.filters[obj_id][j]
            raw_pt = kpts[j]
            conf = sc[j] if j < len(sc) else 0.0
            last_good = self._last_good[obj_id][j]

            # Confidence gating: suppress unreliable keypoints
            if conf < self.min_confidence or raw_pt[0] <= 0 or raw_pt[1] <= 0:
                smoothed[j] = filt(last_good, timestamp)
                adjusted_scores[j] = 0.0
                continue

            # Velocity clamping: reject impossible jumps
            if last_good[0] > 0 and last_good[1] > 0:
                velocity = np.linalg.norm(raw_pt - last_good)
                if velocity > self.max_velocity:
                    direction = (raw_pt - last_good) / (velocity + 1e-6)
                    raw_pt = last_good + direction * self.max_velocity
                    adjusted_scores[j] *= 0.5

            # Confidence-weighted blending: uncertain points pulled to prediction
            if filt.x_prev is not None and conf < 0.8:
                effective_pt = conf * raw_pt + (1.0 - conf) * filt.x_prev
            else:
                effective_pt = raw_pt

            smoothed[j] = filt(effective_pt, timestamp)
            self._last_good[obj_id][j] = smoothed[j].copy()
            self._last_scores[obj_id][j] = adjusted_scores[j]

        return smoothed, adjusted_scores

    def reset(self, obj_id=None):
        if obj_id is not None:
            self.filters.pop(obj_id, None)
            self._last_good.pop(obj_id, None)
            self._last_scores.pop(obj_id, None)
        else:
            self.filters.clear()
            self._last_good.clear()
            self._last_scores.clear()
