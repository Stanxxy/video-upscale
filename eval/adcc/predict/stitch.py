"""Seam-dedup for slice-and-stitch prediction capture (plan §6.2, Bezos's
slice-and-stitch ruling; ``scope_policy.py``'s companion).

``scope_policy.slice_scope`` breaks any over-``MAX_SCOPE_SEC`` match scope
into an ORDERED list of overlapping <= 720s slices, each analyzed by its own
independent ``chunk-segment-tags`` run (``run_predictions.py``). Every slice's
``clips`` are ALREADY converted to absolute match seconds (that conversion is
``run_predictions.py``'s job — chunk-relative-to-slice + the slice's own
absolute ``start_s`` offset — this module never sees chunk-relative seconds).
This module's ONE job: concatenate those per-slice clip lists and dedup the
seam duplicates the ``SLICE_OVERLAP_S`` overlap zone deliberately produces.

**Why NOT reuse ``service/pipelines/time_dedup.py``'s bucketing:**
``time_dedup.deduplicate_clips_by_time`` buckets by EXACT ``(action_class,
actor)`` identity before ever checking temporal overlap — this is documented
dedup-bug-D (plan §1(D), §5.2 "FP-duplicate... directly quantifies the dedup
actor-wobble bug"): when Gemini flips ``actor`` (top/bottom/contested) across
an analysis seam for the SAME real exchange, the two reports land in
DIFFERENT buckets and both survive as duplicates. A slice seam is exactly the
kind of boundary where this flip is likely (two independent Gemini calls, no
shared ``previous_context`` between slices — ``run_predictions.py`` restarts
context at every slice boundary by design, since each slice is a genuinely
separate PASS-1+PASS-2 run). This module's merge key is deliberately
TEMPORAL-PROXIMITY + CLASS-COMPATIBILITY instead, with ``actor`` never
consulted at all — a duplicate that only differs by ``actor`` still merges.

**Class-compatibility** (not exact ``action_class`` identity): the three
``submission_*`` classes are treated as mutually compatible, mirroring
``apply_gracie_mapping``'s SUBMISSION family-wide compat rule (§5.1) — a
model that calls the same seam-straddling finish ``submission_choke`` from
one slice's viewpoint and ``submission_arm_lock`` from the other's (occlusion
right at the seam) is still describing ONE real event. Every other
``action_class`` requires an exact match (they aren't prone to this specific
occlusion-driven subtype flip the way submission-lock identification is).

**Span-widening on merge**: mirrors ``time_dedup.py``'s own established
convention (kept for codebase consistency, a considered extension of the
plan's literal "keep the higher-confidence survivor" — see
``_merge_pair``'s docstring) — the merged clip's OTHER fields (action_class,
actor, outcome, confidence, actor_gi_color) come from the higher-confidence
half, but ``start_s``/``end_s`` widen to the UNION of both spans. This is
what makes the "completion lands exactly on a seam boundary" case work: a
truncated clip from one slice (cut off at that slice's own scope edge) merged
with the same event's fuller reporting from the overlapping neighbor slice
produces ONE clip spanning the full technique, not a truncated survivor.

Output shape matches ``report.py``'s predictions contract exactly: ``{
"match_id": ..., "clips": [...] }``.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("eval.adcc.predict.stitch")

__all__ = [
    "DEFAULT_PROXIMITY_WINDOW_S",
    "SUBMISSION_FAMILY_CLASSES",
    "StitchError",
    "stitch_match_predictions",
]

# "midpoint within a small window" (plan §6.2/Part C3). Generous enough to
# absorb ordinary few-second Gemini timestamp jitter between two INDEPENDENT
# calls describing the same real exchange (no shared context to align them),
# small enough that it won't accidentally fuse two genuinely different
# back-to-back techniques in a fast scramble. Deliberately smaller than
# SLICE_OVERLAP_S (45s, scope_policy.py) — the overlap window is how wide a
# net the SLICING casts to guarantee an event lands in two slices at all;
# this is how close two REPORTS of that event need to be, timestamp-wise, to
# be considered the same exchange once both slices have reported it.
DEFAULT_PROXIMITY_WINDOW_S: float = 10.0

# Mirrors apply_gracie_mapping's §5.1 SUBMISSION family-wide compat rule —
# an occlusion-driven subtype flip AT the seam is still one real event.
SUBMISSION_FAMILY_CLASSES: frozenset[str] = frozenset(
    {"submission_choke", "submission_arm_lock", "submission_leg_lock"}
)


class StitchError(ValueError):
    """Raised for malformed slice-clip input (missing/non-numeric
    start_s/end_s) that this module refuses to silently coerce or drop
    without logging."""


def _valid_clip(clip: dict) -> bool:
    if not isinstance(clip, dict):
        return False
    try:
        start_s = float(clip.get("start_s"))
        end_s = float(clip.get("end_s"))
    except (TypeError, ValueError):
        return False
    return end_s > start_s


def _class_compatible(a: str, b: str) -> bool:
    if a == b:
        return True
    return a in SUBMISSION_FAMILY_CLASSES and b in SUBMISSION_FAMILY_CLASSES


def _spans_close(a: dict, b: dict, proximity_window_s: float) -> bool:
    a_start, a_end = float(a["start_s"]), float(a["end_s"])
    b_start, b_end = float(b["start_s"]), float(b["end_s"])

    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap > 0:
        return True

    mid_a = (a_start + a_end) / 2.0
    mid_b = (b_start + b_end) / 2.0
    return abs(mid_a - mid_b) <= proximity_window_s


def _merge_pair(survivor: dict, candidate: dict) -> dict:
    """Merges ``candidate`` into ``survivor`` (both already known temporally
    close + class-compatible). Non-span fields come from whichever of the two
    has the higher ``confidence`` (plan's literal instruction: "keep the
    higher-confidence survivor"); ``start_s``/``end_s`` widen to the UNION of
    both spans (documented extension, see module docstring's "Span-widening
    on merge" — this is what prevents a real event captured in truncated
    form by one slice, and in full by its overlapping neighbor, from
    surviving as the truncated half)."""
    survivor_conf = float(survivor.get("confidence", 0.0) or 0.0)
    candidate_conf = float(candidate.get("confidence", 0.0) or 0.0)
    winner = candidate if candidate_conf > survivor_conf else survivor

    merged = dict(winner)
    merged["start_s"] = min(float(survivor["start_s"]), float(candidate["start_s"]))
    merged["end_s"] = max(float(survivor["end_s"]), float(candidate["end_s"]))
    # Provenance: which slice(s) contributed to this merged clip (debugging /
    # smoke-test aid only — never consulted by the merge decision itself).
    sources = set()
    for half in (survivor, candidate):
        src = half.get("source_slice_index")
        if src is not None:
            sources.add(src)
        for prior in half.get("source_slice_indices", []):
            sources.add(prior)
    if sources:
        merged["source_slice_indices"] = sorted(sources)
        merged.pop("source_slice_index", None)
    return merged


def stitch_match_predictions(
    match_id: str,
    per_slice_clips: list[list[dict]],
    *,
    proximity_window_s: float = DEFAULT_PROXIMITY_WINDOW_S,
) -> dict:
    """Concatenates ``per_slice_clips`` (one clip list per slice, in slice
    order — clips already carrying ABSOLUTE match seconds) and dedups seam
    duplicates by temporal-proximity + class-compatibility (never ``actor``).

    Malformed clips (missing/non-numeric ``start_s``/``end_s``) are DROPPED
    and logged, never fabricated into a mergeable shape (mirrors
    ``time_dedup.py``'s own discipline).

    Returns ``{"match_id": match_id, "clips": [...]}`` — the exact shape
    ``report.py.load_predictions`` reads from ``<match_id>.json``.
    """
    all_clips: list[dict] = []
    for slice_index, clips in enumerate(per_slice_clips):
        for clip in clips:
            if not _valid_clip(clip):
                logger.warning(
                    "stitch: dropping malformed clip from slice %d of match %r: %r",
                    slice_index, match_id, clip,
                )
                continue
            tagged = dict(clip)
            tagged.setdefault("source_slice_index", slice_index)
            all_clips.append(tagged)

    if not all_clips:
        return {"match_id": match_id, "clips": []}

    all_clips.sort(key=lambda c: float(c["start_s"]))

    survivors: list[dict] = [all_clips[0]]
    for candidate in all_clips[1:]:
        merged_into_existing = False
        # Compare against every current survivor (not just the last one):
        # sorting by start_s puts seam duplicates near each other in the
        # overwhelming case, but class-compatible proximity is not always
        # adjacent-only once multiple classes interleave in the overlap zone
        # — checking the full survivor set is correct and, at per-match
        # event-count scale (tens, not thousands), cheap.
        for i, survivor in enumerate(survivors):
            if _class_compatible(survivor["action_class"], candidate["action_class"]) and _spans_close(
                survivor, candidate, proximity_window_s
            ):
                survivors[i] = _merge_pair(survivor, candidate)
                merged_into_existing = True
                break
        if not merged_into_existing:
            survivors.append(candidate)

    survivors.sort(key=lambda c: float(c["start_s"]))
    return {"match_id": match_id, "clips": survivors}
