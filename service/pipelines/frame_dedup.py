"""Dependency-neutral frame-based clip deduplication."""
from __future__ import annotations

from collections import defaultdict


def deduplicate_clips(all_analysis_results):
    all_clips = []
    for chunk in all_analysis_results:
        if "analysis" in chunk and "clips" in chunk["analysis"]:
            all_clips.extend(chunk["analysis"]["clips"])

    if not all_clips:
        return []

    # Group by role (athlete) so overlaps are resolved per-athlete
    by_role = defaultdict(list)
    for clip in all_clips:
        by_role[clip.get("role", "")].append(clip)

    merged_all = []
    for role, clips in by_role.items():
        clips.sort(key=lambda x: x["start_frame"])

        # Phase 1: merge same-action adjacent/overlapping clips (original logic, now per-athlete)
        merged = [clips[0].copy()]
        for nxt in clips[1:]:
            cur = merged[-1]
            overlap_len = max(0, min(cur["end_frame"], nxt["end_frame"])
                               - max(cur["start_frame"], nxt["start_frame"]))
            is_same_cat = cur.get("action", cur.get("category", "")) == nxt.get("action", nxt.get("category", ""))
            is_close = (nxt["start_frame"] - cur["end_frame"]) < 30
            if is_same_cat and (overlap_len > 0 or is_close):
                new_start = min(cur["start_frame"], nxt["start_frame"])
                new_end = max(cur["end_frame"], nxt["end_frame"])
                if nxt.get("confidence", 0) > cur.get("confidence", 0):
                    merged[-1] = nxt.copy()
                merged[-1]["start_frame"] = new_start
                merged[-1]["end_frame"] = new_end
            else:
                merged.append(nxt.copy())

        # Phase 2: resolve remaining overlaps (different-category, same athlete)
        # Keep the higher-confidence clip; trim or discard the lower one.
        resolved = [merged[0]]
        for nxt in merged[1:]:
            cur = resolved[-1]
            overlap_len = max(0, min(cur["end_frame"], nxt["end_frame"])
                               - max(cur["start_frame"], nxt["start_frame"]))
            if overlap_len > 0:
                if nxt.get("confidence", 0) > cur.get("confidence", 0):
                    # cur loses: trim it to end just before nxt starts
                    cur["end_frame"] = nxt["start_frame"] - 1
                    if cur["end_frame"] <= cur["start_frame"]:
                        resolved.pop()
                    resolved.append(nxt)
                else:
                    # nxt loses: trim it to start just after cur ends
                    nxt["start_frame"] = cur["end_frame"] + 1
                    if nxt["start_frame"] < nxt["end_frame"]:
                        resolved.append(nxt)
            else:
                resolved.append(nxt)

        merged_all.extend(resolved)

    merged_all.sort(key=lambda x: x["start_frame"])
    return merged_all


__all__ = ["deduplicate_clips"]
