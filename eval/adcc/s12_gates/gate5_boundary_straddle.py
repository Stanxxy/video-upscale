"""LeCun cutover Gate 5 — boundary-straddle count (s12 gates).

**Zero Gemini calls, zero network.** Pure arithmetic over the already-
committed gold snapshot (``eval/adcc/gold_snapshot/highlight_gold.jsonl`` +
``normalized_gold.jsonl``) and the production chunking constant
(``service.pipelines.chunk_segment.MAX_SCOPE_SEC``, 720s, zero overlap —
mirrors ``HIGHLIGHT_SPLIT_OVERLAP_S=0.0``).

**The question.** vision-engine v2 slices a match's analysis scope into
non-overlapping 720s outer chunks. Each chunk is analyzed independently by
Gemini — there is NO cross-chunk context. If a highlight event's setup
evidence (the exchange leading up to a submission/sweep/back-take/pass) falls
in the PREVIOUS chunk while the event itself lands in the NEXT chunk, that
setup evidence is invisible to whichever chunk actually reports the event.
Per INS-097 (truncated-window findings), this loss is directionally biased:
gold ``t_s`` marks the event's trigger/completion instant, so the AT-RISK
evidence is almost entirely the window BEFORE ``t_s``, not after it. This
gate quantifies how many gold events sit close enough to a 720s boundary for
that loss to plausibly happen.

**Modeling a point event as a span.** Gold events are point timestamps, not
intervals, so we model event ``e``'s evidence span as::

    [t_s - W, t_s + POST_TAIL_S]

for a sweep of setup-window widths ``W in {30, 60, 120}`` seconds, plus a
small fixed tail ``POST_TAIL_S`` after ``t_s`` so a boundary landing exactly
on the finish itself still counts.

*Why these three ``W`` values, not one invented number:* this same codebase
already carries a considered estimate for "how long does one submission/
sweep/pass setup-to-finish take" — ``eval/adcc/predict/scope_policy.py``'s
``SLICE_OVERLAP_S = 45.0`` docstring: "most gold SUBMISSION rows resolve in
well under 45s from the ATTEMPT-typed entry to the finish... a generous upper
bound, not a measured one." We sweep AROUND that existing 45s reference
rather than inventing a fresh number:
  - ``W=30``: below the 45s reference — a conservative/tight setup window.
  - ``W=60``: just above the 45s reference — adds margin for slower setups.
  - ``W=120``: matches ``scope_policy.MULTI_MATCH_PRE_MARGIN_S`` (also
    already load-bearing elsewhere in this repo, as the multi-match scope's
    OWN pre-event margin) — an intentionally aggressive upper bound for
    worst-case sensitivity.

*Why ``POST_TAIL_S = 5.0``:* mirrors
``service.pipelines.chunk_segment.SEGMENT_SPLIT_OVERLAP_S`` (5.0s), the
production module's own small repair-overlap constant — reused here for
consistency rather than inventing a new constant, and to allow for ``t_s``
being logged at the trigger frame rather than full completion.

**Boundary counted:** only INTERIOR chunk seams (``k=1,2,3,...`` chunk
starts from the scope's own start), never the scope's own ``k=0`` start
itself — that edge is where analysis begins, not a place where evidence
could have been split across two independently-analyzed chunks.

**Two grid anchoring choices** (production applies the first; the second is
a sensitivity check only):
  - ``production``: chunk boundaries at ``scope_start + k*720``, where
    ``scope_start`` is computed exactly the way
    ``eval/adcc/predict/run_predictions.py``/``scope_policy.py`` compute it
    for a real run — ``0.0`` for a single-match video (a video's own
    ffprobe duration only affects where the window ENDS, never where it
    STARTS, so no ffprobe lookup is needed for this gate), or
    ``max(0, min(all-gold t_s for this match) - 120)`` for a multi-match VOD
    (``multi_match_scope``, using the match's FULL gold event set from
    ``normalized_gold.jsonl`` — the same "gold_t_s_values" ``run_predictions
    .build_match_targets`` computes production-side, not just the highlight
    subset).
  - ``naive_t0``: pretends every match's grid starts at raw video ``t=0``,
    ignoring the multi-match scope-window origin entirely. Reported purely
    as a sensitivity check per the task brief.

**Offset sweep.** For each ``(anchor_mode, W)`` pair we also sweep an
additional uniform grid-phase offset ``0..719s`` (step 1s) added to every
match's own anchor, and report the min/mean/max straddle COUNT (summed
across all gold events) over that 720-value sweep. This bounds how sensitive
the headline number is to the exact scope-start second (e.g. an off-by-a-
few-seconds video seek or scope-window rounding) rather than relying on one
single lucky/unlucky phase.

No fallback logic anywhere: malformed/missing gold rows raise immediately
(``KeyError``/``json.JSONDecodeError``/explicit ``ValueError``), never
silently skipped or defaulted.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from eval.adcc.predict.run_predictions import MatchTarget, build_match_targets
from eval.adcc.predict.scope_policy import multi_match_scope
from service.pipelines.chunk_segment import MAX_SCOPE_SEC, SEGMENT_SPLIT_OVERLAP_S

__all__ = [
    "CHUNK_S",
    "W_VALUES_S",
    "POST_TAIL_S",
    "OFFSET_SWEEP_S",
    "StraddleEvent",
    "load_jsonl",
    "build_grid_starts",
    "boundaries_in_span",
    "find_straddling_events",
    "sweep_offsets",
    "run_gate5",
    "main",
]

# Re-exported from the production module (read-only import, never redefined)
# so this gate stays in lockstep with the real cap if it ever changes.
CHUNK_S: float = float(MAX_SCOPE_SEC)

W_VALUES_S: tuple[float, ...] = (30.0, 60.0, 120.0)

# Mirrors chunk_segment.SEGMENT_SPLIT_OVERLAP_S (5.0s) — see module docstring.
POST_TAIL_S: float = float(SEGMENT_SPLIT_OVERLAP_S)

# Grid-phase sweep: integer offsets 0..719s inclusive, step 1s.
OFFSET_SWEEP_S: tuple[int, ...] = tuple(range(0, int(CHUNK_S)))

ANCHOR_MODES: tuple[str, ...] = ("production", "naive_t0")


class Gate5DataError(ValueError):
    """Raised for malformed/inconsistent gold data — never silently
    skipped or coerced into a plausible-looking-but-wrong record."""


@dataclass(frozen=True)
class StraddleEvent:
    match_id: str
    match_name: str
    action_class: str
    t_s: float
    w_s: float
    span_start_s: float
    span_end_s: float
    boundary_s: float
    boundary_k: int
    grid_start_s: float


def load_jsonl(path: Path) -> list[dict]:
    """Loads a JSONL gold file. Fails loud: missing file -> FileNotFoundError
    (no ``try/except``, no default-to-empty); a malformed line ->
    ``json.JSONDecodeError`` propagates as-is; an empty file is itself an
    error (a real gold snapshot is never empty)."""
    text = path.read_text()
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise Gate5DataError(f"{path}: file is empty — a real gold snapshot must have >=1 record")
    return [json.loads(line) for line in lines]


def build_grid_starts(normalized_gold_records: list[dict]) -> dict[str, dict]:
    """One entry per ``match_id``: ``{"is_multi_match", "production_grid_start_s",
    "naive_t0_grid_start_s"}``. Reuses ``run_predictions.build_match_targets`` +
    ``scope_policy.multi_match_scope`` verbatim (the SAME functions a real
    prediction run uses) so this gate's grid can never drift from what
    production actually computes. ``normalized_gold_records`` must be the
    FULL 258-match corpus (not the highlight-filtered subset) — sibling
    multi-match detection and the pre-margin's ``min(gold t_s)`` both need
    every event, not just the highlight-predicate-passing subset."""
    targets: dict[str, MatchTarget] = build_match_targets(normalized_gold_records)
    grid_starts: dict[str, dict] = {}
    for match_id, target in targets.items():
        if target.is_multi_match:
            if not target.gold_t_s_values:
                raise Gate5DataError(
                    f"match {match_id!r}: flagged is_multi_match=True but has zero gold t_s values "
                    "— cannot compute a scope window"
                )
            window = multi_match_scope(match_id, target.gold_t_s_values)
            production_start = window.start_s
        else:
            # single_match_scope always returns start_s=0.0 regardless of
            # duration (duration only bounds where the window ENDS) — so we
            # never need an ffprobe lookup for this gate's grid-origin math.
            production_start = 0.0
        grid_starts[match_id] = {
            "is_multi_match": target.is_multi_match,
            "production_grid_start_s": production_start,
            "naive_t0_grid_start_s": 0.0,
        }
    return grid_starts


def boundaries_in_span(span_start: float, span_end: float, anchor: float, chunk_s: float = CHUNK_S) -> list[tuple[int, float]]:
    """Interior chunk-seam boundaries (``anchor + k*chunk_s``, ``k=1,2,...``
    — never ``k=0``, the scope's own start, see module docstring) that fall
    within ``[span_start, span_end]`` inclusive. Returns ``[(k, boundary_s), ...]``
    sorted by ``k``. Direct, closed-form ``k`` range — no unbounded loop."""
    if span_end < span_start:
        raise Gate5DataError(f"span_end ({span_end}) < span_start ({span_start})")
    k_min = max(1, int((span_start - anchor) // chunk_s) - 1)
    hits: list[tuple[int, float]] = []
    k = k_min
    while True:
        boundary = anchor + k * chunk_s
        if boundary > span_end:
            break
        if boundary >= span_start:
            hits.append((k, boundary))
        k += 1
    return hits


def find_straddling_events(
    highlight_records: list[dict], grid_starts: dict[str, dict], *, anchor_mode: str, w_s: float,
) -> list[StraddleEvent]:
    if anchor_mode not in ANCHOR_MODES:
        raise Gate5DataError(f"unknown anchor_mode {anchor_mode!r}, expected one of {ANCHOR_MODES}")
    anchor_key = f"{anchor_mode}_grid_start_s"

    straddles: list[StraddleEvent] = []
    for rec in highlight_records:
        match_id = rec["match_id"]
        if match_id not in grid_starts:
            raise Gate5DataError(
                f"match {match_id!r} present in highlight_gold.jsonl but missing from "
                "normalized_gold.jsonl-derived grid starts — gold snapshots are out of sync"
            )
        grid_start = grid_starts[match_id][anchor_key]
        for event in rec["events"]:
            t_s = float(event["t_s"])
            span_start = t_s - w_s
            span_end = t_s + POST_TAIL_S
            for k, boundary in boundaries_in_span(span_start, span_end, grid_start):
                straddles.append(
                    StraddleEvent(
                        match_id=match_id,
                        match_name=rec["name"],
                        action_class=event["action_class"],
                        t_s=t_s,
                        w_s=w_s,
                        span_start_s=span_start,
                        span_end_s=span_end,
                        boundary_s=boundary,
                        boundary_k=k,
                        grid_start_s=grid_start,
                    )
                )
    return straddles


def sweep_offsets(
    highlight_records: list[dict], grid_starts: dict[str, dict], *, anchor_mode: str, w_s: float,
) -> dict:
    """For each integer offset in ``OFFSET_SWEEP_S``, adds it to every
    match's own anchor and counts total straddling events across the whole
    corpus. Returns ``{"counts_by_offset": [...], "min", "mean", "max"}``."""
    if anchor_mode not in ANCHOR_MODES:
        raise Gate5DataError(f"unknown anchor_mode {anchor_mode!r}, expected one of {ANCHOR_MODES}")
    anchor_key = f"{anchor_mode}_grid_start_s"

    # Precompute (grid_start, t_s) pairs once — the offset sweep re-derives
    # only the cheap boundary check per offset, not the whole record walk.
    events: list[tuple[float, float]] = []
    for rec in highlight_records:
        match_id = rec["match_id"]
        if match_id not in grid_starts:
            raise Gate5DataError(f"match {match_id!r} missing from grid_starts during offset sweep")
        base_anchor = grid_starts[match_id][anchor_key]
        for event in rec["events"]:
            events.append((base_anchor, float(event["t_s"])))

    counts_by_offset: list[int] = []
    for offset in OFFSET_SWEEP_S:
        count = 0
        for base_anchor, t_s in events:
            span_start = t_s - w_s
            span_end = t_s + POST_TAIL_S
            if boundaries_in_span(span_start, span_end, base_anchor + offset):
                count += 1
        counts_by_offset.append(count)

    return {
        "counts_by_offset": counts_by_offset,
        "min": min(counts_by_offset),
        "mean": statistics.mean(counts_by_offset),
        "max": max(counts_by_offset),
    }


def _scan_existing_predictions(adcc_root: Path) -> list[str]:
    """Deliverable #4: search ``eval/adcc/`` for any already-captured
    prediction-output JSON that would let a LATER pass measure an accuracy
    delta (straddling vs non-straddling events) — reported only, never
    executed/loaded/run here (zero Gemini calls in this gate)."""
    skip_names = {
        "highlight_gold.jsonl", "normalized_gold.jsonl", "highlight_gold_manifest.json",
        "manifest.json", "split_assignment.json",
    }
    hits: list[str] = []
    for path in sorted(adcc_root.rglob("*.json")):
        if "__pycache__" in path.parts:
            continue
        if path.name in skip_names:
            continue
        if "s12_gates" in path.parts:
            continue
        hits.append(str(path.relative_to(adcc_root.parent.parent)))
    return hits


def run_gate5(*, gold_dir: Path, adcc_root: Path) -> dict:
    highlight_records = load_jsonl(gold_dir / "highlight_gold.jsonl")
    normalized_records = load_jsonl(gold_dir / "normalized_gold.jsonl")

    grid_starts = build_grid_starts(normalized_records)

    total_events = sum(len(rec["events"]) for rec in highlight_records)
    matches_with_events = sum(1 for rec in highlight_records if rec["events"])
    total_matches = len(highlight_records)
    multi_match_count = sum(1 for v in grid_starts.values() if v["is_multi_match"])

    per_anchor: dict[str, dict] = {}
    for anchor_mode in ANCHOR_MODES:
        per_w: dict[str, dict] = {}
        for w_s in W_VALUES_S:
            straddles = find_straddling_events(highlight_records, grid_starts, anchor_mode=anchor_mode, w_s=w_s)
            sweep = sweep_offsets(highlight_records, grid_starts, anchor_mode=anchor_mode, w_s=w_s)
            per_w[str(w_s)] = {
                "straddle_count": len(straddles),
                "fraction_of_gold_events": (len(straddles) / total_events) if total_events else 0.0,
                "straddling_events": [asdict(s) for s in straddles],
                "offset_sweep": sweep,
            }
        per_anchor[anchor_mode] = {"per_w": per_w}

    existing_predictions = _scan_existing_predictions(adcc_root)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_s": CHUNK_S,
        "post_tail_s": POST_TAIL_S,
        "w_values_s": list(W_VALUES_S),
        "offset_sweep_range_s": [OFFSET_SWEEP_S[0], OFFSET_SWEEP_S[-1]],
        "total_gold_events": total_events,
        "total_matches": total_matches,
        "matches_with_highlight_events": matches_with_events,
        "multi_match_count": multi_match_count,
        "single_match_count": total_matches - multi_match_count,
        "per_anchor": per_anchor,
        "existing_prediction_outputs_found": existing_predictions,
    }


def _fmt_pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


def _render_report_md(results: dict) -> str:
    lines: list[str] = []
    lines.append("# Gate 5 — Boundary-Straddle Count")
    lines.append("")
    lines.append(
        f"Generated: {results['generated_at_utc']} | chunk_s={results['chunk_s']:.0f} | "
        f"post_tail_s={results['post_tail_s']:.0f}"
    )
    lines.append("")
    lines.append(
        f"Total gold (highlight-predicate) events: **{results['total_gold_events']}** across "
        f"{results['total_matches']} matches ({results['matches_with_highlight_events']} with >=1 "
        f"highlight event, {results['multi_match_count']} multi-match / {results['single_match_count']} "
        "single-match)."
    )
    lines.append("")
    lines.append(
        "Zero Gemini/network calls. Pure arithmetic over "
        "`eval/adcc/gold_snapshot/{highlight_gold,normalized_gold}.jsonl` and the production 720s "
        "chunk constant (`service.pipelines.chunk_segment.MAX_SCOPE_SEC`)."
    )
    lines.append("")

    lines.append("## Headline: straddle count per setup-window W, per anchoring choice")
    lines.append("")
    lines.append("| Anchor | W (s) | Straddle count | Fraction of gold events | Offset-sweep min/mean/max |")
    lines.append("|---|---|---|---|---|")
    for anchor_mode in ANCHOR_MODES:
        for w_s in W_VALUES_S:
            entry = results["per_anchor"][anchor_mode]["per_w"][str(w_s)]
            sweep = entry["offset_sweep"]
            lines.append(
                f"| {anchor_mode} | {w_s:.0f} | {entry['straddle_count']} | "
                f"{_fmt_pct(entry['fraction_of_gold_events'])} | "
                f"{sweep['min']} / {sweep['mean']:.1f} / {sweep['max']} |"
            )
    lines.append("")

    lines.append(
        "`production` anchoring = the grid production actually applies (per-match "
        "`scope_policy.multi_match_scope`/single-match `start_s=0`, exactly mirroring "
        "`run_predictions.build_match_targets`). `naive_t0` = video-`t=0`-anchored sensitivity check "
        "only (ignores the multi-match scope-window origin)."
    )
    lines.append("")

    lines.append("## Straddling events — production anchoring, per W")
    for w_s in W_VALUES_S:
        entry = results["per_anchor"]["production"]["per_w"][str(w_s)]
        lines.append("")
        lines.append(f"### W={w_s:.0f}s ({entry['straddle_count']} straddling event(s))")
        if not entry["straddling_events"]:
            lines.append("")
            lines.append("_None._")
            continue
        lines.append("")
        lines.append("| Match | action_class | t_s | span | boundary crossed (k, s) |")
        lines.append("|---|---|---|---|---|")
        for ev in entry["straddling_events"]:
            lines.append(
                f"| {ev['match_name']} ({ev['match_id']}) | {ev['action_class']} | {ev['t_s']:.0f} | "
                f"[{ev['span_start_s']:.0f}, {ev['span_end_s']:.0f}] | "
                f"k={ev['boundary_k']}, {ev['boundary_s']:.0f}s (grid_start={ev['grid_start_s']:.0f}) |"
            )
    lines.append("")

    lines.append("## Existing prediction outputs under eval/adcc/ (accuracy-delta measurement)")
    lines.append("")
    if results["existing_prediction_outputs_found"]:
        lines.append(
            "The following JSON artifacts exist and COULD, in a later pass, be joined against this "
            "gate's straddling-event list to measure an accuracy delta (straddling vs non-straddling). "
            "None were loaded, parsed, or executed by this gate (zero-Gemini scope):"
        )
        lines.append("")
        for p in results["existing_prediction_outputs_found"]:
            lines.append(f"- `{p}`")
    else:
        lines.append(
            "**None found.** No captured Gemini prediction-output JSON exists anywhere under "
            "`eval/adcc/` at the time this gate ran (only gold snapshots, `sam3_scout/clips.json`, and "
            "its `reports/template_pre_gemini.json` template were present, neither of which is a real "
            "prediction run). An accuracy-delta measurement (straddling vs non-straddling events) is "
            "**not currently possible** without first running real predictions — which this gate "
            "deliberately does not do."
        )
    lines.append("")

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=Path("eval/adcc/gold_snapshot"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval/adcc/s12_gates"))
    args = parser.parse_args(argv)

    adcc_root = args.gold_dir.parent  # eval/adcc
    results = run_gate5(gold_dir=args.gold_dir, adcc_root=adcc_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "gate5_results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=False) + "\n")

    report_path = args.out_dir / "GATE5_REPORT.md"
    report_path.write_text(_render_report_md(results))

    prod_120 = results["per_anchor"]["production"]["per_w"][str(120.0)]
    print(
        f"Gate 5: {results['total_gold_events']} gold events, "
        f"W=120s production straddle count={prod_120['straddle_count']} "
        f"({_fmt_pct(prod_120['fraction_of_gold_events'])}), "
        f"offset-sweep max={prod_120['offset_sweep']['max']}"
    )
    print(f"wrote {results_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
