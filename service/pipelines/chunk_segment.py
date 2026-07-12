"""QA-layer ``chunk_segment`` (PASS 1 of the ``chunk-segment-tags`` pipeline) —
pure helpers: the Gemini structured-output schema/prompt for the ONE cheap
rough-segmentation call, and deterministic chunk-bound repair.

Kept as pure functions (no Gemini client here) so the bound-repair logic is
unit-testable without a network call — ``service/pipelines/executors.py``'s
``chunk_segment_node`` is the only caller that actually talks to Gemini.

Domain rule baked into the prompt (Gracie): "skip broadcast filler, never skip
live mat time." ``reason`` is a closed enum; ``worth_analysis`` is DERIVED in
Python (``reason == "match_action"``), never asked of Gemini directly — an
enum-constrained axis is cheaper to get right and impossible to drift off
vocabulary (same discipline as ``simplified_tags.py``'s enum axes).
"""
from __future__ import annotations

import logging
from typing import Optional

from google.genai import types

logger = logging.getLogger("service.pipelines.chunk_segment")

# Frozen enum — keep in lockstep with the prompt text below and with
# ``build_chunk_segment_schema``'s ``response_schema`` enum list.
REASON_VALUES: list[str] = [
    "match_action",
    "restart_reset",
    "stoppage",
    "pre_match",
    "broadcast",
]

# Reasons that are KEPT in the emitted chunk map but never deep-analyzed by
# PASS 2 (shown + flagged, not skipped/deleted from the map).
_KEPT_NOT_ANALYZED_REASONS = frozenset({"restart_reset", "stoppage"})
# Reasons that are skipped entirely from downstream analysis.
_SKIPPED_REASONS = frozenset({"pre_match", "broadcast"})

# Reason priority when a bound-repair merge combines two chunks with
# DIFFERENT reasons — highest-priority reason wins for the merged chunk.
# "never skip live mat time" means match_action always wins; the two
# kept-but-not-analyzed reasons come next (still real mat time, just not
# deep-analyzed); broadcast/pre_match are the least important to preserve.
_REASON_MERGE_PRIORITY: dict[str, int] = {
    "match_action": 0,
    "restart_reset": 1,
    "stoppage": 1,
    "pre_match": 2,
    "broadcast": 2,
}

# v1 hard cap on the run's TOTAL scoped duration (§ spec: "~10-12 min").
# Beyond this, ``executors``/``qa_pipeline`` fail closed with an actionable
# message rather than silently truncating or attempting an unvalidated
# pre-slice-and-stitch. 12 minutes.
MAX_SCOPE_SEC: int = 12 * 60

# Overlap used ONLY for the over-max split repair below — deliberately the
# same value as ``ChunkAnalyzeConfig.overlap_s``'s default (5s) so a
# bound-repair split seam-heals via the SAME dedup overlap window PASS 2
# already carries. Not read from ``ChunkAnalyzeConfig`` at repair time (the
# two nodes validate independently) — if a future caller needs this
# configurable, promote it to a real field; for v1 it is a documented
# constant matching the locked default.
SEGMENT_SPLIT_OVERLAP_S: float = 5.0


def worth_analysis(reason: str) -> bool:
    """Pure derivation, never asked of Gemini: only ``match_action`` chunks
    get PASS-2 deep analysis."""
    return reason == "match_action"


def build_chunk_segment_schema() -> types.Schema:
    chunk_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["start_s", "end_s", "reason"],
        properties={
            "start_s": types.Schema(
                type=types.Type.NUMBER, format="float",
                description="Chunk start, ABSOLUTE match seconds (same clock as the requested scope).",
            ),
            "end_s": types.Schema(
                type=types.Type.NUMBER, format="float",
                description="Chunk end, ABSOLUTE match seconds (same clock as the requested scope).",
            ),
            "reason": types.Schema(
                type=types.Type.STRING,
                enum=list(REASON_VALUES),
                description="Exactly one scene-class enum value — see instructions.",
            ),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["chunks"],
        properties={
            "chunks": types.Schema(
                type=types.Type.ARRAY,
                description="ORDERED, non-overlapping chunks covering the full requested scope.",
                items=chunk_schema,
            ),
        },
    )


def build_segment_prompt(start_sec: int, end_sec: int) -> str:
    """Per-run prompt text for the PASS-1 rough scan.

    Timestamps are explicitly ABSOLUTE match seconds within
    ``[start_sec, end_sec]`` — the whole scoped video is Gemini's only
    reference frame here (no prior-chunk context threading in PASS 1, unlike
    PASS 2)."""
    return f"""
    You are scanning a BJJ video from {start_sec}s to {end_sec}s (absolute match
    seconds) at LOW sampling resolution to classify the SCENE CLASS of each
    contiguous segment — you are NOT reading grips or techniques yet, only
    deciding what KIND of footage each span is.

    Break the ENTIRE {start_sec}-{end_sec}s span into an ORDERED, non-overlapping
    list of chunks. For each chunk, classify `reason` as EXACTLY ONE of:
      - "match_action": live grappling/rolling is actively happening.
      - "restart_reset": referee stand-up, stall reset, or out-of-bounds restart
         (no live exchange right now, but still part of the match).
      - "stoppage": injury timeout, points/penalty conference, or another real
         stoppage (still part of the match, no live exchange right now).
      - "pre_match": walkouts, introductions, staredowns — before the match starts.
      - "broadcast": commentary desk, replays, sponsor reads, crowd shots,
         anything that is not live match footage or an in-match pause.

    HARD RULE: skip broadcast filler, but NEVER skip live mat time. When in
    doubt between "match_action" and anything else, prefer "match_action" —
    false negatives here (calling live action something else) cost more than
    false positives.

    CLEAN CUT POINTS ONLY: cut a chunk boundary only when the position has
    been static for at least ~3 seconds, or the athletes have reset to
    standing. NEVER cut while a limb is being isolated, a pass is clearing the
    legs, a takedown is in the air, a back-take is mid-transition, or a
    scramble is still resolving — extend the current chunk through those
    moments instead of slicing them.

    Timestamps (`start_s`/`end_s`) must be ABSOLUTE match seconds in
    [{start_sec}, {end_sec}], never relative to a sub-window.

    Return ONLY valid JSON matching this shape — no prose:
    {{
        "chunks": [
            {{ "start_s": {start_sec}, "end_s": 0, "reason": "<enum value>" }}
        ]
    }}
    """


def _reason_priority(reason: str) -> int:
    return _REASON_MERGE_PRIORITY.get(reason, 2)


def _merge_reason(a: str, b: str) -> str:
    return a if _reason_priority(a) <= _reason_priority(b) else b


def repair_chunk_bounds(
    raw_chunks: list[dict], min_chunk_s: float, max_chunk_s: float,
    split_overlap_s: float = SEGMENT_SPLIT_OVERLAP_S,
) -> tuple[list[dict], list[str]]:
    """Deterministic bound repair (LeCun): merge sub-min chunks into a
    neighbor (safe — the dominant failure mode is under-segmentation/lumping,
    not hallucinated chunks, so being generous about merging is the right
    bias); split over-max chunks WITH ``split_overlap_s`` overlap (never a
    naked cut at ``max_chunk_s`` — that guillotines a submission/transition
    mid-action) so PASS 2's dedup seam-heals the split.

    Returns ``(repaired_chunks, adjustment_notes)`` — ``adjustment_notes`` is
    a flat list of human-readable strings (also attached per-chunk via the
    ``adjustment`` key) so the emitted ``chunk_map`` is never silent about a
    repair.

    Malformed raw entries (missing/non-numeric ``start_s``/``end_s``, or an
    off-enum ``reason``) are DROPPED (logged), never fabricated into a
    plausible-looking chunk.
    """
    cleaned: list[dict] = []
    for entry in raw_chunks:
        if not isinstance(entry, dict):
            continue
        try:
            start_s = float(entry["start_s"])
            end_s = float(entry["end_s"])
        except (KeyError, TypeError, ValueError):
            logger.warning("chunk_segment: dropping malformed chunk entry %r", entry)
            continue
        reason = entry.get("reason")
        if reason not in REASON_VALUES:
            logger.warning("chunk_segment: dropping chunk with off-enum reason %r", entry)
            continue
        if end_s <= start_s:
            logger.warning("chunk_segment: dropping non-positive-duration chunk %r", entry)
            continue
        cleaned.append({"start_s": start_s, "end_s": end_s, "reason": reason})

    cleaned.sort(key=lambda c: c["start_s"])

    # --- Pass A: merge sub-min chunks into a neighbor -------------------- #
    merged: list[dict] = []
    notes: list[str] = []
    for chunk in cleaned:
        duration = chunk["end_s"] - chunk["start_s"]
        if duration >= min_chunk_s or not merged:
            merged.append(dict(chunk))
            continue
        # Merge into the PREVIOUS chunk (its neighbor) — safe per the
        # under-segmentation-bias rationale above.
        prev = merged[-1]
        note = (
            f"merged sub-min chunk [{chunk['start_s']:.1f}-{chunk['end_s']:.1f}s, "
            f"{duration:.1f}s < min {min_chunk_s:.1f}s] into neighbor "
            f"[{prev['start_s']:.1f}-{prev['end_s']:.1f}s]"
        )
        prev["end_s"] = max(prev["end_s"], chunk["end_s"])
        prev["reason"] = _merge_reason(prev["reason"], chunk["reason"])
        prev["adjustment"] = (prev.get("adjustment") + "; " + note) if prev.get("adjustment") else note
        notes.append(note)

    # If the FIRST chunk itself is sub-min (no earlier neighbor to merge
    # into), merge it forward into the next one instead.
    if merged and (merged[0]["end_s"] - merged[0]["start_s"]) < min_chunk_s and len(merged) > 1:
        first = merged.pop(0)
        nxt = merged[0]
        note = (
            f"merged leading sub-min chunk [{first['start_s']:.1f}-{first['end_s']:.1f}s] "
            f"into neighbor [{nxt['start_s']:.1f}-{nxt['end_s']:.1f}s]"
        )
        nxt["start_s"] = min(nxt["start_s"], first["start_s"])
        nxt["reason"] = _merge_reason(nxt["reason"], first["reason"])
        nxt["adjustment"] = (nxt.get("adjustment") + "; " + note) if nxt.get("adjustment") else note
        notes.append(note)

    # --- Pass B: split over-max chunks WITH overlap ---------------------- #
    final: list[dict] = []
    for chunk in merged:
        duration = chunk["end_s"] - chunk["start_s"]
        if duration <= max_chunk_s:
            final.append(chunk)
            continue
        note = (
            f"split over-max chunk [{chunk['start_s']:.1f}-{chunk['end_s']:.1f}s, "
            f"{duration:.1f}s > max {max_chunk_s:.1f}s] into sub-chunks with "
            f"{split_overlap_s:.1f}s overlap"
        )
        notes.append(note)
        cursor = chunk["start_s"]
        first_piece = True
        while cursor < chunk["end_s"]:
            piece_end = min(chunk["end_s"], cursor + max_chunk_s)
            piece = {
                "start_s": cursor, "end_s": piece_end, "reason": chunk["reason"],
                "adjustment": note if first_piece else (note + " (continuation)"),
            }
            final.append(piece)
            first_piece = False
            if piece_end >= chunk["end_s"]:
                break
            cursor = piece_end - split_overlap_s

    return final, notes


def finalize_chunks(raw_chunks: list[dict], min_chunk_s: float, max_chunk_s: float, analyze_all: bool) -> list[dict]:
    """Repair bounds, derive ``worth_analysis``, apply the ``analyze_all``
    ablation override, and assign 1-based ``index`` — the exact shape emitted
    in the ``chunk_map`` NDJSON event and consumed by ``chunk_analyze_node``.
    """
    repaired, _notes = repair_chunk_bounds(raw_chunks, min_chunk_s, max_chunk_s)
    out: list[dict] = []
    for i, chunk in enumerate(repaired, start=1):
        reason = chunk["reason"]
        worth = True if analyze_all else worth_analysis(reason)
        out.append({
            "index": i,
            "start_s": chunk["start_s"],
            "end_s": chunk["end_s"],
            "reason": reason,
            "worth_analysis": worth,
            "adjustment": chunk.get("adjustment"),
        })
    return out


def check_scope_cap(duration_sec: float) -> Optional[str]:
    """Fail-closed v1 hard cap check (no silent truncation, no pre-slice
    stitching). Returns an actionable error message string if ``duration_sec``
    exceeds ``MAX_SCOPE_SEC``, else ``None``."""
    if duration_sec > MAX_SCOPE_SEC:
        return (
            f"Scoped duration ({duration_sec:.0f}s) exceeds the chunk-segment-tags "
            f"v1 hard cap of {MAX_SCOPE_SEC}s (~{MAX_SCOPE_SEC // 60} min). Narrow "
            "the run scope — pre-slicing/stitching a longer video is not "
            "supported in v1 (no silent truncation)."
        )
    return None


__all__ = [
    "REASON_VALUES",
    "MAX_SCOPE_SEC",
    "SEGMENT_SPLIT_OVERLAP_S",
    "worth_analysis",
    "build_chunk_segment_schema",
    "build_segment_prompt",
    "repair_chunk_bounds",
    "finalize_chunks",
    "check_scope_cap",
]
