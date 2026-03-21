"""
Tracking state machine for BJJ athlete tracking.
Ported from bjj-pose-estimation/bjj_pipeline/core/state_machine.py
"""
from enum import Enum


class TrackingState(Enum):
    TRACKING = "TRACKING"
    SCRAMBLE = "SCRAMBLE"
    LOST = "LOST"
    RE_ID_MODE = "RE_ID_MODE"
    FADING_OUT = "FADING_OUT"
    BLACKOUT = "BLACKOUT"
    RECOVERING = "RECOVERING"


class StateMachine:
    """
    Manages high-level tracking state transitions.

    Transitions:
    - TRACKING -> SCRAMBLE: When IoU > 0.7
    - TRACKING -> RE_ID_MODE: When scene cut detected
    - SCRAMBLE -> TRACKING: When IoU < 0.6 (triggers identity verify)
    - * -> FADING_OUT -> BLACKOUT -> RECOVERING -> RE_ID_MODE
    """
    def __init__(self):
        self.state = TrackingState.TRACKING
        self.scramble_iou_threshold = 0.7
        self.stable_iou_threshold = 0.6

    def update(self, iou, is_scene_cut, fade_state="ACTIVE"):
        # 1. Handle Fade Transitions
        if fade_state == "FADING_OUT":
            self.state = TrackingState.FADING_OUT
            return
        elif fade_state == "BLACKOUT":
            self.state = TrackingState.BLACKOUT
            return
        elif fade_state == "RECOVERING":
            self.state = TrackingState.RECOVERING
            return
        elif fade_state == "ACTIVE" and self.state == TrackingState.RECOVERING:
            self.state = TrackingState.RE_ID_MODE
            return

        # 2. Handle Hard Cuts
        if is_scene_cut:
            self.state = TrackingState.RE_ID_MODE
            return

        # 3. Handle Tracking Logic
        if self.state in [TrackingState.TRACKING, TrackingState.SCRAMBLE]:
            if self.state == TrackingState.TRACKING:
                if iou > self.scramble_iou_threshold:
                    self.state = TrackingState.SCRAMBLE
            elif self.state == TrackingState.SCRAMBLE:
                if iou < self.stable_iou_threshold:
                    self.state = TrackingState.TRACKING
        elif self.state == TrackingState.RE_ID_MODE:
            pass  # Caller handles reset

    def set_tracking(self):
        self.state = TrackingState.TRACKING
