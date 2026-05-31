"""Time-range parsing for clip bounds."""

def _parse_time_range(
    video_path: str,
    start_time: str | None,
    end_time: str | None,
) -> tuple[int, int | None]:
    """Convert MM:SS or HH:MM:SS time strings to frame indices."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    def _to_seconds(ts: str) -> float:
        parts = ts.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(ts)

    start_frame = 0
    if start_time:
        start_frame = int(_to_seconds(start_time) * fps)

    end_frame = None
    if end_time:
        end_frame = int(_to_seconds(end_time) * fps)

    return start_frame, end_frame
