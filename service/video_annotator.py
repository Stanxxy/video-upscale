"""
Post-process a tracked video to overlay analysis event captions.

Reads the tracked video (which already has masks, boxes, and pose skeletons),
and adds semi-transparent caption banners showing technique metadata during
the frame ranges where each clip event is active.
"""
import logging
import os
import subprocess

import cv2

logger = logging.getLogger(__name__)


def _resolve_actor_label(clip: dict, bindings_by_player_id: dict | None) -> str:
    """Burn-in identity label: prefer grounded player_name / track_id over the
    free-text `role` descriptor (gi color etc.).

    Order: binding player_name → track_id ("Athlete N") → actor_player_id →
    `role` descriptor (last resort).
    """
    pid = clip.get("actor_player_id")
    if pid and bindings_by_player_id:
        binding = bindings_by_player_id.get(pid)
        if binding is not None:
            name = (
                binding.get("player_name") if isinstance(binding, dict)
                else getattr(binding, "player_name", None)
            )
            if name:
                return name
            track_id = (
                binding.get("track_id") if isinstance(binding, dict)
                else getattr(binding, "track_id", None)
            )
            if track_id is not None:
                return f"Athlete {track_id}"
    if pid:
        return str(pid)
    return clip.get("role", "") or ""


def annotate_video(
    tracked_video_path: str,
    analysis_result: dict,
    output_path: str,
    fps: float,
    start_frame: int = 0,
    athlete_bindings=None,
) -> str:
    """
    Overlay analysis clip captions onto the tracked video.

    Args:
        tracked_video_path: Path to tracked_output.mp4 (has masks/boxes/skeleton).
        analysis_result: Dict with "clips" list and "fps".
        output_path: Where to write the final annotated video.
        fps: Video FPS (used for frame-to-time mapping).
        start_frame: The global start frame offset used during tracking.

    Returns:
        Path to the annotated video file.
    """
    clips = analysis_result.get("clips", [])
    if not clips:
        logger.info("No analysis clips to overlay, copying tracked video as-is")
        # Just copy/link the tracked video if no clips
        import shutil
        shutil.copy2(tracked_video_path, output_path)
        return output_path

    cap = cv2.VideoCapture(tracked_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open tracked video: {tracked_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    logger.info(
        "Annotating video: %dx%d, %d frames, %.1f fps, %d clips",
        width, height, total_frames, video_fps, len(clips),
    )

    # Write to temp file first (MP4V), then re-encode to H.264
    raw_output = output_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(raw_output, fourcc, video_fps, (width, height))

    # Build frame-to-clips lookup for fast access
    # The tracked video frame index maps to global_idx = start_frame + local_idx
    clip_lookup = _build_clip_lookup(clips, start_frame, total_frames)

    # Index bindings by player_id so the caption can show grounded player names.
    bindings_by_player_id = {}
    for b in (athlete_bindings or []):
        bpid = b.get("player_id") if isinstance(b, dict) else getattr(b, "player_id", None)
        if bpid:
            bindings_by_player_id[bpid] = b

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if any clips are active for this frame
        global_idx = start_frame + frame_idx
        active_clips = clip_lookup.get(global_idx, [])

        if active_clips:
            frame = _draw_caption_banner(
                frame, active_clips, width, height,
                bindings_by_player_id=bindings_by_player_id,
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    logger.info("Wrote %d annotated frames", frame_idx)

    # Re-encode with H.264 for compatibility
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_output,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-loglevel", "error",
                output_path,
            ],
            check=True,
        )
        os.remove(raw_output)
    except Exception as e:
        logger.warning("FFmpeg re-encode failed (%s), using raw MP4V", e)
        os.rename(raw_output, output_path)

    return output_path


def _build_clip_lookup(
    clips: list[dict],
    start_frame: int,
    total_frames: int,
) -> dict[int, list[dict]]:
    """
    Build a mapping from global frame index to list of active clips.

    Only populates entries for frames that have at least one active clip,
    keeping memory usage proportional to clip coverage rather than total frames.
    """
    lookup: dict[int, list[dict]] = {}

    for clip in clips:
        sf = clip.get("start_frame", 0)
        ef = clip.get("end_frame", sf)
        for fidx in range(sf, ef + 1):
            if fidx not in lookup:
                lookup[fidx] = []
            lookup[fidx].append(clip)

    return lookup


def _draw_caption_banner(
    frame, active_clips: list[dict], width: int, height: int,
    bindings_by_player_id: dict | None = None,
):
    """Draw semi-transparent banner at the bottom with clip metadata."""
    # Calculate banner height based on number of clips
    line_height = 28
    padding = 12
    num_clips = min(len(active_clips), 3)  # Show at most 3 stacked
    banner_height = padding * 2 + num_clips * (line_height * 2 + 8)

    # Draw semi-transparent dark banner at bottom
    overlay = frame.copy()
    banner_top = height - banner_height
    cv2.rectangle(overlay, (0, banner_top), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)

    # Draw clip info
    y = banner_top + padding + line_height
    for i, clip in enumerate(active_clips[:3]):
        action = clip.get("action", clip.get("category", "other"))
        technique = clip.get("specific_technique", clip.get("technique", ""))
        # Prefer grounded player_name / track_id over the free-text role descriptor.
        role = _resolve_actor_label(clip, bindings_by_player_id)
        confidence = clip.get("confidence", 0)

        # Line 1: action + technique (bold/large)
        text1 = f"{_format_category(action)}: {technique}"
        cv2.putText(
            frame, text1, (padding, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Line 2: role + confidence (smaller)
        conf_pct = f"{confidence * 100:.0f}%" if confidence else ""
        text2 = f"{role}  {conf_pct}" if role else conf_pct
        cv2.putText(
            frame, text2, (padding, y + line_height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA,
        )

        y += line_height * 2 + 8

    return frame


def _format_category(category: str) -> str:
    """Convert SCREAMING_SNAKE to Title Case."""
    return category.replace("_", " ").title()
