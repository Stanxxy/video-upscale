"""Human-in-the-loop detection helpers."""
import cv2

from tracking_pipeline.human_verification_suspend import HumanVerificationSuspend
from tracking_pipeline.hybrid.yolo26_detector import YOLO26Detector

def _detect_and_request_boxes(frame_bgr, global_idx, detection_callback,
                        detector, yolo_model, detection_threshold, device):
    """Lazy-load YOLO, detect persons, and request user-verified boxes.

    Returns (box_a, box_b, detector) on success, or None if user cancelled
    OR if there is no ``detection_callback`` to consume YOLO results.
    The detector is returned so the caller can keep the reference for future use.

    No-op short-circuit: when ``detection_callback`` is ``None`` (CLI tests
    and other headless callers) there is no consumer for YOLO output —
    mid-track suspend cannot be raised without a callback. After the
    2026-05-25 refactor production tracking always runs sequentially with a
    non-None ``detection_callback`` (see ``service/worker.py:_make_detection_cb``
    + the call site in ``run_job``), so this short-circuit is now a
    defense-in-depth safety net rather than a hot path. Loading YOLO +
    running a forward pass per track-loss frame is pure waste when the
    callback is ``None``, and previously the locally-loaded detector was
    thrown away on the no-callback return path so YOLO was reloaded every
    track-loss (logs flooded with `[detect] Loading yolo26m.pt on mps
    (persistent)...` every few frames in the prior parallel-tracking
    design). Short-circuit before touching YOLO.
    """
    if detection_callback is None:
        # Parallel-segment mode / headless run: nothing to do with YOLO output.
        # Return None to signal track loss; caller handles via max_missing_frames.
        # Do NOT try CLI select_boxes — not available in server context.
        print(f"  Frame {global_idx}: Track lost, no detection_callback — continuing")
        return None

    # Service mode with a human-in-the-loop callback.
    # Lazy-load YOLO detector
    if detector is None:
        print(f"  Frame {global_idx}: Loading YOLO detector on demand...")
        detector = YOLO26Detector(
            yolo_model=yolo_model, threshold=detection_threshold, device=device,
        )

    yolo_detections = detector.detect_persons(frame_bgr)

    _, frame_jpeg_enc = cv2.imencode(".jpg", frame_bgr)
    try:
        cb_result = detection_callback(
            "tracking_lost", frame_jpeg_enc.tobytes(),
            yolo_detections=yolo_detections,
            frame_idx=global_idx,
        )
        if cb_result is not None:
            return (cb_result[0], cb_result[1], detector)
    except HumanVerificationSuspend:
        raise
    except Exception as e:
        print(f"  [detection_callback] failed: {e}")
    return None


def _correct_identity_swap_after_scramble(frame_boxes, frame_masks, frame_rgb,
                                    identity_mgr, display_map, global_idx):
    """After exiting SCRAMBLE, check if identities got swapped."""
    print(f"  Frame {global_idx}: Exiting SCRAMBLE — verifying identities...")

    if 1 not in frame_boxes or 2 not in frame_boxes:
        return

    s1 = identity_mgr.query_identity_scores(
        frame_rgb, frame_boxes[1], mask=frame_masks.get(1),
    )
    s2 = identity_mgr.query_identity_scores(
        frame_rgb, frame_boxes[2], mask=frame_masks.get(2),
    )

    s1_as_1 = s1.get(1, 0)
    s1_as_2 = s1.get(2, 0)
    s2_as_1 = s2.get(1, 0)
    s2_as_2 = s2.get(2, 0)

    print(f"    Track 1: ID1={s1_as_1:.2f}, ID2={s1_as_2:.2f}")
    print(f"    Track 2: ID1={s2_as_1:.2f}, ID2={s2_as_2:.2f}")

    if s1_as_2 > s1_as_1 and s2_as_1 > s2_as_2:
        print("    Identity SWAP detected! Correcting display labels.")
        display_map[1], display_map[2] = display_map[2], display_map[1]

