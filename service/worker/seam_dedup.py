"""Seam dedup for the 45s backward outer-chunk overlap
(2026-07-26-engine13-rescope-single-call-cutover.md AC5/AC6; Brooks's
architecture ruling §3).

``highlight_orchestrator._outer_chunks`` gives each outer chunk after the
first a 45s BACKWARD-extended start (Gate 5, ``VERDICTS_V2.md``) so
``highlight_critique_node``'s own backward pad (``critique_backpad_s``, max
30s) and ``highlight_analyze_node``'s own ``preroll_s`` (max 15s) have room
to reach past a chunk seam that would otherwise truncate them exactly at the
grid boundary — losing setup-cue evidence for a highlight straddling the
seam. That overlap means a real event sitting in the seam band can be
independently discovered and reported by BOTH the trailing highlight_scan of
chunk k and the leading highlight_scan of chunk k+1 — two independent
Gemini calls, no shared ``previous_context`` between outer chunks (each
chunk is its own fresh ``RunContext``).

**Why NOT reuse ``service/pipelines/time_dedup.py``'s bucketing (Brooks's
ruling §3, refusing option (d)):** ``time_dedup.deduplicate_clips_by_time``
buckets by EXACT ``(action_class, actor)`` identity before ever checking
temporal overlap. If the actor axis flips across the seam (plausible — each
outer chunk gets a fresh, independent reference-image call with no shared
context), the two reports land in different buckets and BOTH survive as
duplicates — the exact "dedup-bug-D" ``eval/adcc/predict/stitch.py`` (the
ADCC eval harness's own slice-seam dedup) was built to survive. This module
is a structural, production-side PORT of ``stitch.py``'s temporal-proximity
+ class-compatibility approach — ``actor``/``player_id`` is NEVER consulted
in the merge key — not a call into ``time_dedup.py``.

**Structurally simpler than ``stitch.py`` by design:** ``stitch.py`` MERGES
two seam halves (span-widening, "keep the higher-confidence non-span
fields") because its caller (the eval harness) collects an entire match's
clips before "publishing" anything. Production publishes per-highlight,
INCREMENTALLY, as each highlight is analyzed (Brooks's ruling §2 — SNS is an
immutable, append-only, one-highlight-at-a-time record, NEVER rewritten once
published). By the time chunk k+1's leading highlight is analyzed, chunk
k's trailing highlight (if it is the same real event) has ALREADY been
published — there is nothing to merge INTO. This module's only job is
deciding whether to SKIP publishing chunk k+1's highlight because it is a
seam-duplicate of something chunk k already published; the original,
already-published record is never touched, corrected, or widened. The
single-call taxonomy verdict also carries no ``confidence`` field (unlike
``stitch.py``'s eval-only schema), so there is no higher-confidence
tie-break to make in the first place — the earlier chunk's already-published
version simply stands.
"""
from __future__ import annotations

from typing import Optional

# Mirrors eval/adcc/predict/stitch.py's SUBMISSION_FAMILY_CLASSES exactly —
# an occlusion-driven submission-subtype flip AT the seam is still one real
# event (apply_gracie_mapping's own §5.1 family-wide compat rule, ADCC eval
# harness precedent).
SUBMISSION_FAMILY_CLASSES: frozenset[str] = frozenset(
    {"submission_choke", "submission_arm_lock", "submission_leg_lock"}
)

# "midpoint within a small window" — same value + reasoning as stitch.py's
# DEFAULT_PROXIMITY_WINDOW_S: generous enough to absorb ordinary few-second
# Gemini timestamp jitter between two INDEPENDENT chunk-scoped calls
# describing the same real exchange, small enough not to accidentally fuse
# two genuinely different back-to-back techniques in a fast scramble.
DEFAULT_PROXIMITY_WINDOW_S: float = 10.0


def class_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """``None``/mismatched-but-unknown values are never silently treated as
    compatible unless literally equal (never fabricated leniency)."""
    if a is None or b is None:
        return a == b
    if a == b:
        return True
    return a in SUBMISSION_FAMILY_CLASSES and b in SUBMISSION_FAMILY_CLASSES


def spans_close(
    a_start: float, a_end: float, b_start: float, b_end: float, proximity_window_s: float,
) -> bool:
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap > 0:
        return True
    mid_a = (a_start + a_end) / 2.0
    mid_b = (b_start + b_end) / 2.0
    return abs(mid_a - mid_b) <= proximity_window_s


def in_seam_band(start_s: float, end_s: float, seam_start: float, seam_end: float) -> bool:
    """Whether a highlight's own ``[start_s, end_s)`` bounds intersect the
    seam band ``[seam_start, seam_end)`` between two adjacent outer chunks
    (the 45s-wide overlap zone ``_outer_chunks`` creates at every boundary
    after the first). A highlight entirely outside this band never needs a
    cross-chunk dedup check at all — this is a cheap pre-filter, not the
    dedup decision itself."""
    return end_s > seam_start and start_s < seam_end


def find_seam_duplicate(
    candidate: dict, prior_clips: list[dict], *, proximity_window_s: float = DEFAULT_PROXIMITY_WINDOW_S,
) -> Optional[dict]:
    """Returns the FIRST already-published clip (from the previous chunk's
    own trailing seam-band clips, ``prior_clips``) that ``candidate`` (a
    highlight from the CURRENT chunk, already known to be in-band via
    ``in_seam_band``) is a temporal-proximity + class-compatible duplicate
    of — or ``None`` if no match. ``actor``/``player_id`` is NEVER consulted
    (see module docstring). Malformed clips (missing/non-numeric
    ``start_s``/``end_s``) are skipped, never fabricated into a comparable
    shape."""
    try:
        cand_start = float(candidate["start_s"])
        cand_end = float(candidate["end_s"])
    except (KeyError, TypeError, ValueError):
        return None
    for prior in prior_clips:
        try:
            prior_start = float(prior["start_s"])
            prior_end = float(prior["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if class_compatible(prior.get("action_class"), candidate.get("action_class")) and spans_close(
            prior_start, prior_end, cand_start, cand_end, proximity_window_s,
        ):
            return prior
    return None


__all__ = [
    "SUBMISSION_FAMILY_CLASSES",
    "DEFAULT_PROXIMITY_WINDOW_S",
    "class_compatible",
    "spans_close",
    "in_seam_band",
    "find_seam_duplicate",
]
