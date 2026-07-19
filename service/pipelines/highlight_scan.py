"""QA-layer ``highlight_scan`` (PASS 1 of the ``highlight-scan-analyze``
pipeline) — pure helpers: the Gemini structured-output schema/prompt for the
ONE cheap rough highlight-scan call, and deterministic highlight-bound
repair.

Kept as pure functions (no Gemini client here) so the bound-repair logic is
unit-testable without a network call — ``service/pipelines/executors.py``'s
``highlight_scan_node`` is the only caller that actually talks to Gemini.

Deliberately SIMPLER than ``chunk_segment.py``'s PASS 1: no ``reason`` enum,
no ``worth_analysis`` derivation, no ``analyze_all`` ablation. This pass is
pure judgment of "is this span worth a closer, finer-grained look" — see the
build plan's "Product Decision" (precision over recall, accepted for this
playground surface).

**Split-repair overlap is deliberately ZERO (contiguous split), unlike
``chunk_segment.SEGMENT_SPLIT_OVERLAP_S``'s 5s back-overlap.** ``chunk_segment``
can afford an overlap-healed split because ``chunk-segment-tags`` still runs
a downstream time-based dedup stage that merges the resulting duplicate
events at the seam. ``highlight-scan-analyze`` has NO dedup stage, and its
own bounds being contiguous (not overlapping) is necessary but, on its own,
NOT SUFFICIENT for correctness — see the honest correctness statement below.

**Correctness statement (corrected 2026-07-18, evaluator CHANGES-REQUIRED
item 2 — the prior text here overclaimed "safe with no dedup").** Event
clipping (``executors._clip_events_to_highlight_bounds``) bounds every
emitted event to its OWN highlight's ``[start_s, end_s]``, which DOES prevent
one specific failure mode: two highlights' OWN bounds are never contiguous-
but-overlapping (merge only extends outward; split pieces here are
zero-overlap), so an event cannot simultaneously satisfy two DIFFERENT
highlights' clip tests from bounds alone.

It does NOT, by itself, prevent a real event from being reported as two
adjacent fragments by two DIFFERENT highlight_analyze calls, because
``preroll_s``/``postroll_s`` EXPAND each highlight's SENT video window beyond
its own bounds — so neighboring highlights' sent windows routinely overlap in
actual video content even though their clipping bounds do not. A single real
action whose true span crosses into that overlapping video region can be
independently seen (and reported) by both calls; clipping then truncates each
report to its own highlight's bounds, yielding two adjacent partial-event
records instead of one. Two distinct residual cases, handled differently
(evaluator CHANGES-REQUIRED item 2, CEO product call 2026-07-18):

1. **Split siblings of ONE over-long PASS-1 highlight (this module's own
   max-length split) — FIXED, not just documented.** A synthetic seam we
   manufacture ourselves must not itself cause double-vision. Each split
   piece here is tagged ``start_is_synthetic``/``end_is_synthetic`` (True on
   the internal cut edge(s) introduced by the split, False on the piece's
   genuine original edge). ``executors.highlight_analyze_node`` reads these
   flags and clamps ``preroll_s``/``postroll_s`` to ZERO on a synthetic
   edge — the two sibling calls' SENT video windows then share no frames
   across the manufactured seam at all, eliminating the specific
   overlapping-content duplicate this fix targets. A single real action that
   genuinely straddles the synthetic cut point may still be reported as two
   abutting fragments (one honestly truncated at the seam on each side) —
   that residual is a property of cutting one real span in half, not a
   manufactured double-report of overlapping content.
2. **Two GENUINELY DISTINCT PASS-1 highlights that happen to sit closer
   together than ``preroll_s + postroll_s`` — ACCEPTED, not fixed.** This is
   real pre/post-roll doing its intended job (giving Gemini surrounding
   context) on two highlights PASS 1 judged independently worth reporting.
   Building a merge/dedup pass for this would reintroduce the mechanical
   dedup machinery this pipeline was built to remove. Accepted for this
   playground surface (precision over recall, human-in-the-loop confirms
   events) — see ``tests/test_pipelines_executors.py``'s pinned regression
   test demonstrating this exact case, so the limitation is documented and
   observable, not silent.
"""
from __future__ import annotations

import logging
from string import Template
from typing import Optional

from google.genai import types

logger = logging.getLogger("service.pipelines.highlight_scan")

# See module docstring: MUST stay 0.0 for the event-clipping invariant to
# hold across split highlight pieces. Kept as a named constant (rather than a
# bare literal in the split loop below) so the rationale has one place to
# live and a future caller cannot casually pass a nonzero value without
# reading this comment.
HIGHLIGHT_SPLIT_OVERLAP_S: float = 0.0

# Bundled default Gemini ``system_instruction`` for the PASS-1 rough scan.
# Overridable via ``HighlightScanConfig.system_prompt`` (full replacement,
# same "empty means default" discipline as ``AnalyzeConfig.system_instruction``).
DEFAULT_SYSTEM_PROMPT: str = """
You are scanning a BJJ (Brazilian Jiu-Jitsu) match video to find HIGHLIGHTS —
spans of live grappling action worth a closer, finer-grained look later. You
are NOT reading grips or naming techniques yet, only judging WHERE the
interesting live action is.

A "highlight" is a contiguous span of live grappling exchange: a takedown
attempt, a guard pass, a sweep, a submission attempt, a scramble, a
back-take, or any other active technique exchange. It is NOT: walkouts,
introductions, staredowns, referee stand-ups with no exchange, stall resets,
injury timeouts, points/penalty conferences, commentary desk cutaways,
replays, sponsor reads, or crowd shots.

HARD RULE: precision over recall is fine here (a human confirms every
highlight afterward) — but never invent a highlight where nothing is
happening, and never merge two clearly separate exchanges into one span just
because they are close together in time.

CLEAN CUT POINTS ONLY: start a highlight only when live action visibly
begins, and end it only once the position has settled/reset or the exchange
has clearly concluded — at least ~2 seconds of stillness, or a stoppage.
NEVER cut while a limb is being isolated, a pass is clearing the legs, a
takedown is in the air, a back-take is mid-transition, or a scramble is
still resolving.
""".strip()

# Bundled default per-run "user turn" prompt for the PASS-1 rough scan.
# ``$start_sec``/``$end_sec`` are substituted via ``string.Template.safe_substitute``
# (never a KeyError on an unrecognized/absent token — see ``build_scan_prompt``)
# with the run's ABSOLUTE match seconds. Literal JSON braces below need no
# escaping (unlike ``str.format`` templates) — ``Template`` only treats ``$``
# specially.
DEFAULT_INITIAL_PROMPT: str = """
Scan this BJJ video from $start_sec s to $end_sec s (absolute match seconds)
at LOW sampling resolution and find every HIGHLIGHT per the system
instructions.

Break the $start_sec-$end_sec s span into an ORDERED list of highlight spans.
Do NOT try to cover the entire span (this is not a scene segmentation) —
only report the spans that are genuinely worth a closer look.

Timestamps (`start_s`/`end_s`) must be ABSOLUTE match seconds in
[$start_sec, $end_sec], never relative to a sub-window.

For each highlight also report:
- `description`: a body-movement description ONLY (grips, level changes,
  positions) — NEVER a technique name/label. This is a recall lever, not a
  classification. Exclude broadcast REPLAYS (a cut back to footage already
  shown) — do not report the same exchange twice because it was replayed.
- `reasoning`: a short free-text justification for why you flagged this span.

Return ONLY valid JSON matching this shape — no prose:
{
    "highlights": [
        { "start_s": $start_sec, "end_s": 0, "description": "string", "reasoning": "string" }
    ]
}
""".strip()


def resolve_system_prompt(override: Optional[str]) -> str:
    """``None``/whitespace-only -> ``DEFAULT_SYSTEM_PROMPT``. A non-empty
    override REPLACES it entirely (same discipline as
    ``simplified_tags.resolve_system_instruction`` /
    ``qa_vlm._clean_system_instruction``)."""
    cleaned = override.strip() if override else None
    return cleaned or DEFAULT_SYSTEM_PROMPT


def build_highlight_schema() -> types.Schema:
    """Gemini structured-output schema for PASS 1: an ORDERED list of
    ``{start_s, end_s, description, reasoning}`` — no ``reason`` ENUM, no
    ``worth_analysis``, no per-clip technique/position detail (that stays
    PASS 2's job). The schema/prompt surface stays small so this rough-scan
    call stays cheap (build plan cost model).

    ``description``/``reasoning`` (v2 build plan,
    ``2026-07-18-highlight-scan-critique-analyze-v2-plan.md``, "Node 1"):
    ADDITIVE to the v1 ``{start_s, end_s}`` shape. ``description`` is a
    body-movement string (NO technique label — recall-biased, "what
    physically happens", not "what this is called"); ``reasoning`` is a
    short free-text justification for WHY this span was flagged (forces the
    model to justify its own recall judgment rather than pattern-matching
    silently). Both REQUIRED at the Gemini schema level (forces the model to
    always produce them) but treated as optional/pass-through Python-side by
    ``repair_highlight_bounds``/``finalize_highlights`` (``.get(...)``, never
    a hard failure if a malformed response omits them)."""
    highlight_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["start_s", "end_s", "description", "reasoning"],
        properties={
            "start_s": types.Schema(
                type=types.Type.NUMBER, format="float",
                description="Highlight start, ABSOLUTE match seconds (same clock as the requested scope).",
            ),
            "end_s": types.Schema(
                type=types.Type.NUMBER, format="float",
                description="Highlight end, ABSOLUTE match seconds (same clock as the requested scope).",
            ),
            "description": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Body-movement description ONLY — what physically happens (grips, level "
                    "changes, positions), NEVER a technique name/label. Recall lever, not a "
                    "classification."
                ),
            ),
            "reasoning": types.Schema(
                type=types.Type.STRING,
                description="Short free-text justification for why this span was flagged as a highlight.",
            ),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["highlights"],
        properties={
            "highlights": types.Schema(
                type=types.Type.ARRAY,
                description="ORDERED list of highlight spans worth a closer look — not a full scene segmentation.",
                items=highlight_schema,
            ),
        },
    )


def build_scan_prompt(start_sec: int, end_sec: int, initial_prompt: Optional[str] = None) -> str:
    """Per-run "user turn" prompt text for the PASS-1 rough scan.

    ``initial_prompt`` (``HighlightScanConfig.initial_prompt``) ``None``/
    whitespace-only -> ``DEFAULT_INITIAL_PROMPT``; a non-empty override
    REPLACES the default template text (same "empty means default" full-
    replacement discipline as ``system_prompt``/``AnalyzeConfig.system_instruction``
    elsewhere in this package). EITHER text is then substituted via
    ``string.Template.safe_substitute`` with the run's ABSOLUTE
    ``start_sec``/``end_sec`` — ``safe_substitute`` never raises on a
    template with no ``$start_sec``/``$end_sec`` tokens at all (a playground
    edit that drops them is a valid, if less-precise, override; it is never a
    hard error), and never treats a literal ``{``/``}`` specially (unlike
    ``str.format``, which would require the JSON illustration braces above to
    be doubled)."""
    cleaned = initial_prompt.strip() if initial_prompt else None
    template = cleaned or DEFAULT_INITIAL_PROMPT
    return Template(template).safe_substitute(start_sec=start_sec, end_sec=end_sec)


def repair_highlight_bounds(
    raw_highlights: list[dict], min_highlight_s: float, max_highlight_s: float,
    split_overlap_s: float = HIGHLIGHT_SPLIT_OVERLAP_S,
    scope_start: Optional[float] = None, scope_end: Optional[float] = None,
) -> tuple[list[dict], list[str]]:
    """Deterministic bound repair, adapted from
    ``chunk_segment.repair_chunk_bounds`` (same merge-under-min bias; see
    module docstring for why the split-over-max behavior differs): merge
    sub-min highlights into a neighbor; split over-max highlights into
    CONTIGUOUS pieces (``split_overlap_s`` defaults to 0.0 — see
    ``HIGHLIGHT_SPLIT_OVERLAP_S``).

    Returns ``(repaired_highlights, adjustment_notes)`` — ``adjustment_notes``
    is a flat list of human-readable strings (also attached per-highlight via
    the ``adjustment`` key) so the emitted highlight map is never silent
    about a repair.

    Malformed raw entries (missing/non-numeric ``start_s``/``end_s``, or a
    non-positive duration) are DROPPED (logged), never fabricated into a
    plausible-looking highlight.

    ``scope_start``/``scope_end`` (optional; live-bug fix, 2026-07-18 —
    scope=40-85s repro, PASS 1 returned a highlight at ``start_s=111,
    end_s=124``, entirely outside the requested scope): when BOTH are
    provided, this is applied FIRST, before the merge/split passes below —
    a raw highlight entirely outside ``[scope_start, scope_end]`` (zero
    overlap, touching-at-the-boundary counts as zero overlap, same
    convention as ``executors._clip_events_to_highlight_bounds``) is DROPPED
    (logged); a highlight that only PARTIALLY overlaps the scope is CLAMPED
    to ``[scope_start, scope_end]``. Without this, an out-of-scope highlight
    flows untouched into ``executors.highlight_analyze_node``'s
    preroll/postroll window math, which can invert (``window_start >
    window_end``) and send Gemini a genuinely malformed native-video
    offset (observed: real ``500 INTERNAL``). ``None`` (the default) skips
    this step entirely, so every existing caller of this pure-function
    surface that doesn't pass scope is unaffected; the one production caller
    (``executors.highlight_scan_node``) always passes the run's real
    ``ctx.start_sec``/``ctx.end_sec``.
    """
    cleaned: list[dict] = []
    for entry in raw_highlights:
        if not isinstance(entry, dict):
            continue
        try:
            start_s = float(entry["start_s"])
            end_s = float(entry["end_s"])
        except (KeyError, TypeError, ValueError):
            logger.warning("highlight_scan: dropping malformed highlight entry %r", entry)
            continue
        if end_s <= start_s:
            logger.warning("highlight_scan: dropping non-positive-duration highlight %r", entry)
            continue
        if scope_start is not None and scope_end is not None:
            if end_s <= scope_start or start_s >= scope_end:
                logger.warning(
                    "highlight_scan: dropping highlight [%.1f-%.1f]s entirely outside "
                    "requested scope [%.1f-%.1f]s", start_s, end_s, scope_start, scope_end,
                )
                continue
            clamped_start, clamped_end = max(start_s, scope_start), min(end_s, scope_end)
            if clamped_start != start_s or clamped_end != end_s:
                logger.warning(
                    "highlight_scan: clamping highlight [%.1f-%.1f]s to requested scope "
                    "[%.1f-%.1f]s -> [%.1f-%.1f]s",
                    start_s, end_s, scope_start, scope_end, clamped_start, clamped_end,
                )
            start_s, end_s = clamped_start, clamped_end
        # description/reasoning (v2, additive): pass-through only, never
        # validated/fabricated — a malformed/omitted value from Gemini
        # (schema requires them, but a response could still fail schema
        # enforcement upstream) becomes None here, not a fabricated string.
        cleaned.append({
            "start_s": start_s, "end_s": end_s,
            "description": entry.get("description"), "reasoning": entry.get("reasoning"),
        })

    cleaned.sort(key=lambda h: h["start_s"])

    # --- Pass A: merge sub-min highlights into a neighbor ---------------- #
    merged: list[dict] = []
    notes: list[str] = []
    for highlight in cleaned:
        duration = highlight["end_s"] - highlight["start_s"]
        if duration >= min_highlight_s or not merged:
            merged.append(dict(highlight))
            continue
        prev = merged[-1]
        note = (
            f"merged sub-min highlight [{highlight['start_s']:.1f}-{highlight['end_s']:.1f}s, "
            f"{duration:.1f}s < min {min_highlight_s:.1f}s] into neighbor "
            f"[{prev['start_s']:.1f}-{prev['end_s']:.1f}s]"
        )
        prev["end_s"] = max(prev["end_s"], highlight["end_s"])
        prev["adjustment"] = (prev.get("adjustment") + "; " + note) if prev.get("adjustment") else note
        notes.append(note)

    # If the FIRST highlight itself is sub-min (no earlier neighbor to merge
    # into), merge it forward into the next one instead.
    if merged and (merged[0]["end_s"] - merged[0]["start_s"]) < min_highlight_s and len(merged) > 1:
        first = merged.pop(0)
        nxt = merged[0]
        note = (
            f"merged leading sub-min highlight [{first['start_s']:.1f}-{first['end_s']:.1f}s] "
            f"into neighbor [{nxt['start_s']:.1f}-{nxt['end_s']:.1f}s]"
        )
        nxt["start_s"] = min(nxt["start_s"], first["start_s"])
        nxt["adjustment"] = (nxt.get("adjustment") + "; " + note) if nxt.get("adjustment") else note
        notes.append(note)

    # --- Pass B: split over-max highlights into CONTIGUOUS pieces -------- #
    # See module docstring — ``split_overlap_s`` MUST stay 0.0 in production
    # use so a split piece's OWN bounds never overlap its sibling's; kept as
    # a parameter only so the contiguous-vs-overlap behavior itself is
    # directly unit-testable.
    #
    # ``start_is_synthetic``/``end_is_synthetic`` (evaluator CHANGES-REQUIRED
    # item 2 fix, 2026-07-18): mark which edge(s) of each piece are a
    # manufactured internal cut point (``True``) vs. the highlight's genuine
    # original edge (``False``) — the FIRST piece's ``start_s`` and the LAST
    # piece's ``end_s`` are genuine; every other edge introduced by this split
    # is synthetic. ``executors.highlight_analyze_node`` reads these flags to
    # clamp ``preroll_s``/``postroll_s`` to zero across a synthetic edge (see
    # module docstring's "Correctness statement").
    final: list[dict] = []
    for highlight in merged:
        duration = highlight["end_s"] - highlight["start_s"]
        if duration <= max_highlight_s:
            final.append({**highlight, "start_is_synthetic": False, "end_is_synthetic": False})
            continue
        note = (
            f"split over-max highlight [{highlight['start_s']:.1f}-{highlight['end_s']:.1f}s, "
            f"{duration:.1f}s > max {max_highlight_s:.1f}s] into CONTIGUOUS sub-highlights "
            f"(0s overlap; sent-window preroll/postroll additionally clamped to zero across "
            f"each synthetic seam — see highlight_analyze_node)"
        )
        notes.append(note)
        cursor = highlight["start_s"]
        first_piece = True
        while cursor < highlight["end_s"]:
            piece_end = min(highlight["end_s"], cursor + max_highlight_s)
            is_last_piece = piece_end >= highlight["end_s"]
            piece = {
                "start_s": cursor, "end_s": piece_end,
                # description/reasoning (v2, additive): each split piece
                # inherits the SAME parent description/reasoning — they
                # describe one continuous highlight that this split
                # mechanically cut into contiguous pieces, not N distinct
                # judgments.
                "description": highlight.get("description"), "reasoning": highlight.get("reasoning"),
                "adjustment": note if first_piece else (note + " (continuation)"),
                "start_is_synthetic": not first_piece,
                "end_is_synthetic": not is_last_piece,
            }
            final.append(piece)
            first_piece = False
            if is_last_piece:
                break
            cursor = piece_end - split_overlap_s

    return final, notes


def finalize_highlights(
    raw_highlights: list[dict], min_highlight_s: float, max_highlight_s: float,
    scope_start: Optional[float] = None, scope_end: Optional[float] = None,
) -> list[dict]:
    """Repair bounds and assign 1-based ``index`` — the exact shape emitted
    in the ``highlight_map`` NDJSON event and consumed by
    ``executors.highlight_analyze_node``. Carries ``start_is_synthetic``/
    ``end_is_synthetic`` through from ``repair_highlight_bounds`` (see that
    function's docstring) — ``False``/``False`` for any highlight that was
    never split.

    ``scope_start``/``scope_end`` (optional; live-bug fix, 2026-07-18) are
    forwarded as-is to ``repair_highlight_bounds`` — see that function's
    docstring for the out-of-scope drop/clamp behavior. The emitted
    ``highlight_map`` streamed to the UI is scope-valid whenever the caller
    passes these (``executors.highlight_scan_node`` always does)."""
    repaired, _notes = repair_highlight_bounds(
        raw_highlights, min_highlight_s, max_highlight_s,
        scope_start=scope_start, scope_end=scope_end,
    )
    out: list[dict] = []
    for i, highlight in enumerate(repaired, start=1):
        out.append({
            "index": i,
            "start_s": highlight["start_s"],
            "end_s": highlight["end_s"],
            "description": highlight.get("description"),
            "reasoning": highlight.get("reasoning"),
            "adjustment": highlight.get("adjustment"),
            "start_is_synthetic": highlight.get("start_is_synthetic", False),
            "end_is_synthetic": highlight.get("end_is_synthetic", False),
        })
    return out


__all__ = [
    "HIGHLIGHT_SPLIT_OVERLAP_S",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_INITIAL_PROMPT",
    "resolve_system_prompt",
    "build_highlight_schema",
    "build_scan_prompt",
    "repair_highlight_bounds",
    "finalize_highlights",
]
