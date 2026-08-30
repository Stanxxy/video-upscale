"""Per-match scope-window policy + over-cap slicing (plan §6.2, founder
decision §8.1).

**Scope window** (fixed, non-tunable constants — documented openly as a
locator, NOT a technique-leak per §6.2):
  - single-match video: ``[0, ffprobe_duration]`` — the whole file.
  - multi-match VOD: ``[min(gold t_s) - PRE_MARGIN_S, max(gold t_s) +
    POST_MARGIN_S]`` per match, clamped to ``>= 0`` — ACCEPTED gold-derived
    window (plan §8.1).

**Slicing.** ``service/pipelines/chunk_segment.py::MAX_SCOPE_SEC`` (720s) is a
hard, fail-closed cap on a SINGLE ``chunk-segment-tags`` run (read-only import
here — this module never modifies that production file, and never raises
``MAX_SCOPE_SEC`` itself). Round-1's ``CALL_BUDGET_ESTIMATE.md`` found 66/258
match-runs individually exceed that cap. Rather than skip those matches or
touch the production scope cap, this module SLICES any over-cap scope into an
ORDERED list of overlapping <= ``MAX_SCOPE_SEC``-second windows, each one a
legal ``chunk-segment-tags`` run on its own. ``stitch.py`` is the seam-dedup
companion that recombines the per-slice predictions.

**Why the overlap margin is 45s** (``SLICE_OVERLAP_S``): the value must
exceed a plausible single-technique duration so a submission/sweep/pass whose
completion lands exactly on a seam is captured whole in at least one slice's
NON-overlap zone and, crucially, captured in BOTH adjacent slices somewhere
near the seam (which is what lets ``stitch.py``'s proximity-based seam-dedup
work at all -- with zero overlap there is nothing to dedup, and a seam-
straddling event is simply lost). 45s is a considered middle point: (a) most
gold SUBMISSION rows resolve in well under 45s from the "ATTEMPT (...)"-typed
entry to the finish (the raw ADCC events are POINT timestamps, not spans, so
this is a generous upper bound, not a measured one); (b) `chunk-segment-tags`
PASS 1 already segments its OWN scope on ``restart_reset``/``stoppage``
boundaries at the true continuity breaks, so a 45s margin only needs to
outlast a single uninterrupted live-mat exchange, not an entire match; (c) it
keeps the overlap-overhead cost real but modest: 45s of double-analyzed time
per internal seam is ~6.25% of one 720s slice, not a multiplier on total
spend (quantified in ``CALL_BUDGET_ESTIMATE.md`` §_Overlap re-price_). This is
deliberately LARGER than
``service/pipelines/chunk_segment.py::SEGMENT_SPLIT_OVERLAP_S`` (5s, PASS 1's
OWN internal over-max chunk-repair overlap, a completely different mechanism
operating one layer below this one) — that overlap only needs to seam-heal a
Gemini-invented chunk boundary; this one needs to seam-heal an eval-harness-
invented SLICE boundary across which an entire independent Gemini run (PASS 1
+ PASS 2) restarts from zero context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from service.pipelines.chunk_segment import MAX_SCOPE_SEC

__all__ = [
    "MULTI_MATCH_PRE_MARGIN_S",
    "MULTI_MATCH_POST_MARGIN_S",
    "SLICE_OVERLAP_S",
    "MAX_SLICE_S",
    "ScopeWindow",
    "Slice",
    "ScopePolicyError",
    "single_match_scope",
    "multi_match_scope",
    "slice_scope",
    "build_slice_plan",
]

# Gold-derived multi-match window margins — ACCEPTED founder decision (plan
# §8.1). Fixed, not tunable per-run: a documented "which chapter of the VOD"
# locator, never a technique-timing hint (§6.2).
MULTI_MATCH_PRE_MARGIN_S: float = 120.0
MULTI_MATCH_POST_MARGIN_S: float = 180.0

# The over-cap slicing max window size. Re-exported (not re-defined) from the
# production module so this stays in lockstep with the real cap if it ever
# changes there — this module still never WRITES to chunk_segment.py.
MAX_SLICE_S: float = float(MAX_SCOPE_SEC)

# See module docstring "Why the overlap margin is 45s".
SLICE_OVERLAP_S: float = 45.0

if SLICE_OVERLAP_S >= MAX_SLICE_S:  # pragma: no cover — defensive, never true for the frozen constants above
    raise ValueError(
        f"SLICE_OVERLAP_S ({SLICE_OVERLAP_S}) must be strictly less than MAX_SLICE_S ({MAX_SLICE_S}) "
        "or slice_scope's cursor never advances."
    )


class ScopePolicyError(ValueError):
    """Raised for malformed scope inputs (empty gold t_s list, inverted
    window, negative duration) — never silently coerced/clamped into a
    plausible-looking-but-wrong scope."""


@dataclass(frozen=True)
class ScopeWindow:
    """A match's full analysis window, BEFORE any over-cap slicing.
    ``start_s``/``end_s`` are absolute seconds into the video's OWN clock
    (per-match clock for a single-match video; the shared VOD clock for a
    multi-match video — never per-match-relative for the latter, per plan
    §3's "Shared-VOD timestamps are absolute into the stream's own clock")."""

    match_id: str
    start_s: float
    end_s: float
    is_multi_match: bool

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class Slice:
    """One <= ``MAX_SLICE_S``-second, individually-runnable
    ``chunk-segment-tags`` scope, tagged with its position in the ordered
    slice list for the match (``slice_index``, 0-based) so
    ``run_predictions.py``/``stitch.py`` can reconstruct adjacency without
    re-deriving it from timestamps alone."""

    match_id: str
    slice_index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def single_match_scope(match_id: str, ffprobe_duration_s: float) -> ScopeWindow:
    """Single-match video scope: ``[0, ffprobe_duration]`` — the whole file
    (plan §6.2). ``ffprobe_duration_s`` must be a real, positive, already-
    probed duration (this function never invokes ffprobe itself — see
    ``run_predictions.py`` for the injectable duration-lookup seam that keeps
    this module network-free and unit-testable)."""
    if ffprobe_duration_s <= 0:
        raise ScopePolicyError(f"ffprobe_duration_s must be positive, got {ffprobe_duration_s!r}")
    return ScopeWindow(match_id=match_id, start_s=0.0, end_s=float(ffprobe_duration_s), is_multi_match=False)


def multi_match_scope(match_id: str, gold_t_s_values: Sequence[float]) -> ScopeWindow:
    """Multi-match VOD per-match scope: ``[min(gold t_s) - 120s, max(gold
    t_s) + 180s]``, clamped to ``>= 0`` (plan §6.2/§8.1). ``gold_t_s_values``
    is that ONE match's own gold event timestamps (already the match's own
    absolute-into-the-VOD seconds, per plan §3) — never another match's."""
    if not gold_t_s_values:
        raise ScopePolicyError(f"match {match_id!r}: multi-match scope requires >=1 gold t_s value, got none")
    start_s = max(0.0, min(gold_t_s_values) - MULTI_MATCH_PRE_MARGIN_S)
    end_s = max(gold_t_s_values) + MULTI_MATCH_POST_MARGIN_S
    return ScopeWindow(match_id=match_id, start_s=start_s, end_s=end_s, is_multi_match=True)


def slice_scope(
    window: ScopeWindow, *, max_slice_s: float = MAX_SLICE_S, overlap_s: float = SLICE_OVERLAP_S,
) -> list[Slice]:
    """Slices ``window`` into an ORDERED list of <= ``max_slice_s``-second
    windows with ``overlap_s`` of back-overlap between consecutive slices
    (plan §6.2's pre-slice-and-stitch, Bezos's ruling — see module docstring
    for the overlap-margin justification). A scope already within the cap
    returns a single one-element list (no gratuitous slicing).

    The LAST slice always ends exactly at ``window.end_s`` (never runs past
    it, never leaves a dangling gap before it) — the second-to-last slice's
    end is pulled back by ``overlap_s`` to become the next slice's start, so
    every internal seam carries the same overlap margin.
    """
    if window.duration_s <= 0:
        raise ScopePolicyError(f"match {window.match_id!r}: non-positive scope duration {window.duration_s!r}")

    if window.duration_s <= max_slice_s:
        return [Slice(match_id=window.match_id, slice_index=0, start_s=window.start_s, end_s=window.end_s)]

    slices: list[Slice] = []
    cursor = window.start_s
    idx = 0
    while cursor < window.end_s:
        slice_end = min(cursor + max_slice_s, window.end_s)
        slices.append(Slice(match_id=window.match_id, slice_index=idx, start_s=cursor, end_s=slice_end))
        if slice_end >= window.end_s:
            break
        cursor = slice_end - overlap_s
        idx += 1
    return slices


def build_slice_plan(
    match_id: str,
    *,
    is_multi_match: bool,
    gold_t_s_values: Optional[Sequence[float]] = None,
    ffprobe_duration_s: Optional[float] = None,
    max_slice_s: float = MAX_SLICE_S,
    overlap_s: float = SLICE_OVERLAP_S,
) -> list[Slice]:
    """One-call convenience: scope-window policy + slicing for one match.
    Exactly one of ``gold_t_s_values`` (multi-match) / ``ffprobe_duration_s``
    (single-match) must be supplied, matching ``is_multi_match`` — supplying
    the wrong one, or both, or neither, is a hard error (no silent
    fallback/guess between the two policies)."""
    if is_multi_match:
        if gold_t_s_values is None or ffprobe_duration_s is not None:
            raise ScopePolicyError(
                f"match {match_id!r}: is_multi_match=True requires gold_t_s_values only "
                f"(got gold_t_s_values={gold_t_s_values!r}, ffprobe_duration_s={ffprobe_duration_s!r})"
            )
        window = multi_match_scope(match_id, gold_t_s_values)
    else:
        if ffprobe_duration_s is None or gold_t_s_values is not None:
            raise ScopePolicyError(
                f"match {match_id!r}: is_multi_match=False requires ffprobe_duration_s only "
                f"(got gold_t_s_values={gold_t_s_values!r}, ffprobe_duration_s={ffprobe_duration_s!r})"
            )
        window = single_match_scope(match_id, ffprobe_duration_s)

    return slice_scope(window, max_slice_s=max_slice_s, overlap_s=overlap_s)
