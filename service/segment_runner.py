"""
Intra-job K-segment parallel runner for standard mode (M3 S9).

Splits the video clip into K roughly equal segments and runs each segment's
full pipeline (tracking → upscale → analyze) in parallel using asyncio + thread
executors.  Identity stitching reassigns athlete IDs at segment boundaries via
spatial centroid matching (with optional DINOv2 cosine similarity upgrade).

Public API
----------
- split_segments(clip_start, clip_end, k, overlap) → list[SegmentBounds]
- stitch_segment_identity(seg0_frames, seg1_frames, overlap_window) → seg1_frames_remapped
- merge_tracking_results(segments_results) → dict
- merge_analysis_results(segments_results) → dict
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segment boundary helpers
# ---------------------------------------------------------------------------

OVERLAP_FRAMES = 30  # frames shared at each boundary for identity stitching


@dataclass
class SegmentBounds:
    """Frame range for a single segment (both ends inclusive-exclusive)."""

    segment_id: int
    start_frame: int  # absolute video frame index (inclusive)
    end_frame: int    # absolute video frame index (exclusive)

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def split_segments(
    clip_start: int,
    clip_end: int,
    k: int,
    overlap: int = OVERLAP_FRAMES,
) -> list[SegmentBounds]:
    """Divide [clip_start, clip_end) into K overlapping segments.

    Each interior segment receives `overlap` extra frames on each side so the
    identity-stitching algorithm has context at the join.  The caller is
    responsible for deduplicating overlap frames in merge_tracking_results().

    K must be >= 1.  If K=1 the single segment spans the full range with no
    overlap (identical to the existing sequential path).

    Example (K=4, 7956 frames, overlap=30):
        seg 0: [0,    1989+30)  = [0, 2019)
        seg 1: [1989-30, 3978+30) = [1959, 4008)
        seg 2: [3978-30, 5967+30) = [3948, 5997)
        seg 3: [5967-30, 7956)   = [5937, 7956)
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    total = clip_end - clip_start
    if total <= 0:
        raise ValueError(f"clip_end ({clip_end}) must be > clip_start ({clip_start})")

    # Base segment length (rounded down); last segment absorbs remainder.
    base = total // k

    segments: list[SegmentBounds] = []
    for i in range(k):
        # Canonical boundary (no overlap)
        seg_start_canon = clip_start + i * base
        seg_end_canon = clip_start + (i + 1) * base if i < k - 1 else clip_end

        # Add overlap padding (clamped to clip bounds)
        seg_start = max(clip_start, seg_start_canon - (overlap if i > 0 else 0))
        seg_end = min(clip_end, seg_end_canon + (overlap if i < k - 1 else 0))

        segments.append(SegmentBounds(
            segment_id=i,
            start_frame=seg_start,
            end_frame=seg_end,
        ))

    return segments


# ---------------------------------------------------------------------------
# Identity stitching
# ---------------------------------------------------------------------------

def _centroid(box: list[float]) -> tuple[float, float]:
    """Return (cx, cy) from [x1,y1,x2,y2]."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _centroid_distance(box_a: list[float], box_b: list[float]) -> float:
    ax, ay = _centroid(box_a)
    bx, by = _centroid(box_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _build_id_mapping_spatial(
    ref_frames: list[dict],
    seg_frames: list[dict],
) -> dict[Any, Any]:
    """
    Build a mapping {seg_athlete_id → ref_athlete_id} using spatial proximity.

    For each athlete in seg_frames (appearing in the overlap region), find the
    closest athlete in ref_frames by average centroid distance over shared
    frame range, then map to the best-matching ref_id.

    Returns an empty dict if either side has no athletes.
    """
    # Collect (frame_idx, athlete_id, box) tuples from ref and seg
    ref_positions: dict[Any, list[tuple[float, float]]] = {}
    for f in ref_frames:
        for a in f.get("athletes", []):
            aid = a.get("track_id") or a.get("athlete_id")
            box = a.get("box")
            if aid is not None and box and len(box) == 4:
                ref_positions.setdefault(aid, []).append(_centroid(box))

    seg_positions: dict[Any, list[tuple[float, float]]] = {}
    for f in seg_frames:
        for a in f.get("athletes", []):
            aid = a.get("track_id") or a.get("athlete_id")
            box = a.get("box")
            if aid is not None and box and len(box) == 4:
                seg_positions.setdefault(aid, []).append(_centroid(box))

    if not ref_positions or not seg_positions:
        logger.warning(
            "stitch: empty athlete positions (ref=%d ids, seg=%d ids) — no remapping",
            len(ref_positions), len(seg_positions),
        )
        return {}

    # Compute mean centroid per athlete in each side
    def mean_centroid(positions: list[tuple[float, float]]) -> tuple[float, float]:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    ref_means = {aid: mean_centroid(pts) for aid, pts in ref_positions.items()}
    seg_means = {aid: mean_centroid(pts) for aid, pts in seg_positions.items()}

    # Greedy nearest-neighbour matching (sufficient for 2-athlete case)
    mapping: dict[Any, Any] = {}
    used_ref_ids: set[Any] = set()
    for seg_id, seg_c in seg_means.items():
        best_ref_id = None
        best_dist = float("inf")
        for ref_id, ref_c in ref_means.items():
            if ref_id in used_ref_ids:
                continue
            d = math.sqrt((seg_c[0] - ref_c[0]) ** 2 + (seg_c[1] - ref_c[1]) ** 2)
            if d < best_dist:
                best_dist = d
                best_ref_id = ref_id
        if best_ref_id is not None:
            mapping[seg_id] = best_ref_id
            used_ref_ids.add(best_ref_id)
            logger.info(
                "stitch: seg_id=%s → ref_id=%s (centroid_dist=%.1f px)",
                seg_id, best_ref_id, best_dist,
            )

    return mapping


def stitch_segment_identity(
    seg0_frames: list[dict],
    seg1_frames: list[dict],
    overlap_window: int = OVERLAP_FRAMES,
    *,
    dino_similarity_threshold: float = 0.7,
) -> list[dict]:
    """
    Remap athlete IDs in seg1_frames to match seg0_frames canonical IDs.

    Algorithm:
    1. Extract the last `overlap_window` frames from seg0 and the first
       `overlap_window` frames from seg1 as the overlap region.
    2. Build an athlete-id mapping via spatial centroid proximity.
    3. Apply the mapping to ALL frames in seg1 (not just overlap).

    Returns a new list of frame dicts (deep-copied metadata, same objects for
    non-athlete fields) with remapped athlete IDs.

    If the mapping is empty or identity (no change needed) the original list
    is returned unchanged.
    """
    if not seg0_frames or not seg1_frames:
        return seg1_frames

    ref_overlap = seg0_frames[-overlap_window:] if len(seg0_frames) >= overlap_window else seg0_frames
    seg_overlap = seg1_frames[:overlap_window] if len(seg1_frames) >= overlap_window else seg1_frames

    mapping = _build_id_mapping_spatial(ref_overlap, seg_overlap)

    if not mapping:
        logger.info("stitch: no ID mapping found — seg1 IDs unchanged")
        return seg1_frames

    # Check whether mapping is identity (all seg_id == ref_id)
    if all(k == v for k, v in mapping.items()):
        logger.info("stitch: mapping is identity — seg1 IDs unchanged")
        return seg1_frames

    logger.info("stitch: applying mapping %s to %d seg1 frames", mapping, len(seg1_frames))

    remapped: list[dict] = []
    for frame in seg1_frames:
        new_frame = {k: v for k, v in frame.items() if k != "athletes"}
        new_athletes = []
        for athlete in frame.get("athletes", []):
            new_ath = dict(athlete)
            # Support both track_id and athlete_id fields
            for id_key in ("track_id", "athlete_id"):
                if id_key in new_ath and new_ath[id_key] in mapping:
                    new_ath[id_key] = mapping[new_ath[id_key]]
            new_athletes.append(new_ath)
        new_frame["athletes"] = new_athletes
        remapped.append(new_frame)

    return remapped


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _canonical_boundary(segments: list[SegmentBounds], i: int) -> int:
    """Return the canonical (non-overlapping) start of segment i."""
    # Canonical boundary is midway between seg[i-1] canonical end and seg[i] canonical end
    # We re-derive from the original split: canonical start = clip_start + i * base
    # Since we don't store that, use the non-overlapping midpoint heuristic:
    # canonical cutoff = start of seg[i] + overlap (i.e. where overlap ends for seg[i])
    if i == 0:
        return segments[0].start_frame
    # The overlap at the start of seg[i] was added because i > 0.
    # Canonical start was originally: seg[i-1].start_frame + (seg[i-1].length_without_overlap)
    # Simplified: canonical_start_of_seg_i = segments[i].start_frame + OVERLAP_FRAMES
    # But we must keep exactly the canonical boundary, not re-add overlap again.
    # Use the simpler rule: keep frames from seg_i where frame_idx >= seg_i.start_frame + OVERLAP_FRAMES
    # (this is where the overlap padding begins on the left side).
    return segments[i].start_frame + OVERLAP_FRAMES


def merge_tracking_results(
    segment_bounds: list[SegmentBounds],
    segment_frames: list[list[dict]],
    overlap: int = OVERLAP_FRAMES,
) -> dict:
    """
    Concatenate per-segment frames lists into a single de-overlapped list.

    For each segment (except the first), frames whose frame_idx falls in the
    overlap region at the start of that segment are dropped (they were already
    emitted by the previous segment).  For the first segment, frames in its
    tail overlap are kept (they are canonical for seg 0).

    The resulting list is sorted by frame_idx.
    """
    if not segment_bounds or not segment_frames:
        return {"frames": []}

    merged: list[dict] = []
    for i, (bounds, frames) in enumerate(zip(segment_bounds, segment_frames)):
        if i == 0:
            # Keep all frames up to (but not including) the start of seg1's
            # canonical region, i.e. keep everything from seg0.
            # Drop duplicates that will come from seg1's overlap.
            # Rule: keep frames with frame_idx < (canonical start of seg1)
            if len(segment_bounds) > 1:
                cutoff = segment_bounds[1].start_frame + overlap
                kept = [f for f in frames if f.get("frame_idx", 0) < cutoff]
            else:
                kept = list(frames)
        else:
            # Drop the left-side overlap region (already in prev segment's output)
            cutoff_lo = bounds.start_frame + overlap
            if i < len(segment_bounds) - 1:
                cutoff_hi = segment_bounds[i + 1].start_frame + overlap
                kept = [
                    f for f in frames
                    if cutoff_lo <= f.get("frame_idx", 0) < cutoff_hi
                ]
            else:
                # Last segment: keep everything from cutoff_lo to end
                kept = [f for f in frames if f.get("frame_idx", 0) >= cutoff_lo]
        merged.extend(kept)

    # Sort by frame_idx to be safe
    merged.sort(key=lambda f: f.get("frame_idx", 0))

    # Use metadata from first segment (fps, video path, etc.)
    return {"frames": merged}


def merge_analysis_results(
    segment_analyses: list[dict | None],
) -> dict | None:
    """
    Merge per-segment analysis dicts into a single result dict.

    Each segment produces {"match_summary": ..., "clips": [...], "fps": ...}.
    We concatenate the clips arrays and deduplicate by (start_frame, end_frame).
    """
    from pipeline import deduplicate_clips

    if not segment_analyses:
        return None

    all_clips: list[dict] = []
    fps = 30.0
    for analysis in segment_analyses:
        if analysis is None:
            continue
        fps = analysis.get("fps", fps)
        all_clips.extend(analysis.get("clips", []) or [])

    if not all_clips:
        # All segments returned no analysis (no Gemini key or no clips)
        return None

    # deduplicate_clips expects the window-level list format:
    # [{"analysis": {"clips": [...]}}]. Wrap each clip individually so the
    # per-segment clips go through the same dedup/merge logic as the
    # single-segment path.
    wrapped = [{"analysis": {"clips": [c]}} for c in all_clips]
    final_clips = deduplicate_clips(wrapped)
    return {
        "match_summary": "Analysis generated via K-segment parallel pipeline",
        "clips": final_clips,
        "fps": fps,
    }
