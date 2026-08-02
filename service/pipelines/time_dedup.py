"""Seconds-based dedup for ``simplified-tags-time-v1`` (Brooks HIGH seam).

``frame_dedup.deduplicate_clips`` is FRAME-keyed (``start_frame``/``end_frame``)
and this pipeline never has a ``ctx.native_fps`` (no frame sampling happens —
``chunk-segment-tags`` is entirely native-video). Rather than fabricate frames
from seconds (banned) or force the frame-keyed function to accept a mixed
unit, this module is a standalone seconds-native merge: bucket by EXACT axis
identity (``action_class`` + ``actor``), merge same-bucket clips whose
time-IoU exceeds a small overlap threshold, keep the higher-confidence clip's
fields on merge (widening the time span to the union).

Malformed clips (missing/non-numeric ``start_s``/``end_s``) are DROPPED
(logged), never fabricated into a mergeable shape.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger("service.pipelines.time_dedup")

# Any positive time-IoU above this merges two same-bucket clips. Deliberately
# low (not a strict "mostly overlapping" bar): the main duplicate source this
# guards against is PASS 2's fixed-overlap_s back-overlap window between
# adjacent chunks re-reporting the same short exchange near the seam, which
# produces LOW absolute IoU on clips that are otherwise the same real event.
TIME_IOU_MERGE_THRESHOLD = 0.05


def _time_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return overlap / union


def _valid_clip(clip: dict) -> bool:
    if not isinstance(clip, dict):
        return False
    try:
        start_s = float(clip.get("start_s"))
        end_s = float(clip.get("end_s"))
    except (TypeError, ValueError):
        return False
    return end_s > start_s


def deduplicate_clips_by_time(raw_results: list[dict]) -> list[dict]:
    """``raw_results`` mirrors ``RunContext.raw_results``'s shape:
    ``[{"window":, "frames":, "analysis": {"clips":[...]}}]`` — flattens every
    clip, drops malformed ones, buckets by exact ``(action_class, actor)``
    identity, merges same-bucket time-IoU-overlapping clips (keep the
    higher-confidence clip's non-time fields; widen the span to the union),
    and returns a flat list sorted by ``start_s``.
    """
    all_clips: list[dict] = []
    for chunk in raw_results:
        for clip in chunk.get("analysis", {}).get("clips", []):
            if _valid_clip(clip):
                all_clips.append(clip)
            else:
                logger.warning("time_dedup: dropping malformed clip %r", clip)

    if not all_clips:
        return []

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for clip in all_clips:
        key = (clip.get("action_class", ""), clip.get("actor", ""))
        buckets[key].append(clip)

    merged_all: list[dict] = []
    for _key, clips in buckets.items():
        clips = sorted(clips, key=lambda c: float(c["start_s"]))
        merged = [dict(clips[0])]
        for nxt in clips[1:]:
            cur = merged[-1]
            iou = _time_iou(float(cur["start_s"]), float(cur["end_s"]), float(nxt["start_s"]), float(nxt["end_s"]))
            if iou > TIME_IOU_MERGE_THRESHOLD:
                winner = nxt if float(nxt.get("confidence", 0) or 0) > float(cur.get("confidence", 0) or 0) else cur
                new_start = min(float(cur["start_s"]), float(nxt["start_s"]))
                new_end = max(float(cur["end_s"]), float(nxt["end_s"]))
                merged[-1] = dict(winner)
                merged[-1]["start_s"] = new_start
                merged[-1]["end_s"] = new_end
            else:
                merged.append(dict(nxt))
        merged_all.extend(merged)

    merged_all.sort(key=lambda c: float(c["start_s"]))
    return merged_all


__all__ = ["TIME_IOU_MERGE_THRESHOLD", "deduplicate_clips_by_time"]
