"""Technique sequences (workflow ``mma-sequences-ground``): pure functions only.

The scope is tiled into short overlapping windows; one census call per window returns
every fighter's chain of techniques: a time span plus the techniques in order, not timed
one by one (founder 2026-11-15: report exchanges). This module tiles the scope, merges
the windows' answers and turns them into published technique events with steps.

Merge rules (plan 2026-11-15-mma-technique-sequences):

1. Copies of one sequence: sequences of the same fighter whose spans overlap, or are
   less than ``SAME_GAP_MS`` apart, are one sequence (seen twice in the overlap of two
   windows, or listed twice by one answer). One copy is kept: the one farthest from its
   own window's edges, which saw the most before and after it; a tie keeps the copy
   with more steps.
2. A sequence in footage that is not live (it starts in an inactive position span) is
   rejected.
"""

from uuid import uuid5

from bjj_analysis_core.mma_outputs import MergedSequenceV1, SequenceBatchV1, SequenceCensusV1
from shared_lib.models.analysis_contracts import TimeRangeMs
from shared_lib.models.analysis_results import TechniqueEventV1, sequence_event_fields

SAME_GAP_MS = 300


def sequence_windows(run, config):
    """Tile the selected scope into overlapping windows, in order."""
    scope = run.spec.scope
    size, overlap = config.sequence_window_ms, config.sequence_overlap_ms
    windows, start = [], scope.start_ms
    while True:
        end = min(start + size, scope.end_ms)
        windows.append(TimeRangeMs(start_ms=start, end_ms=end))
        if end == scope.end_ms:
            return tuple(windows)
        start = end - overlap


def _not_live(positions, at_ms):
    return any(
        p.observation == "inactive" and p.bounds.start_ms <= at_ms < p.bounds.end_ms
        for p in positions.positions or ()
    )


def merge_sequences(run, config, censuses, positions):
    """One census per window -> final sequences in time order (SequenceBatchV1)."""
    windows = sequence_windows(run, config)
    censuses = tuple(SequenceCensusV1.model_validate(c) for c in censuses)
    if [c.window_index for c in censuses] != list(range(len(windows))):
        raise ValueError("exactly one census is required for every sequence window")
    copies = []  # dict(actor, span, margin, seq)
    for index, census in enumerate(censuses):
        window = windows[index]
        for sequence in census.sequences:
            if not window.start_ms <= sequence.start_ms < sequence.end_ms <= window.end_ms:
                raise ValueError("a census sequence falls outside its window")
            copies.append(dict(
                actor=sequence.actor_player_id or sequence.actor_sentinel,
                start=sequence.start_ms, end=sequence.end_ms,
                margin=min(sequence.start_ms - window.start_ms, window.end_ms - sequence.end_ms),
                seq=sequence,
            ))
    # Union copies of one sequence: same fighter, spans overlapping or nearly touching.
    parent = list(range(len(copies)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(copies):
        for j in range(i + 1, len(copies)):
            b = copies[j]
            if a["actor"] == b["actor"] and a["start"] < b["end"] + SAME_GAP_MS and b["start"] < a["end"] + SAME_GAP_MS:
                parent[root(j)] = root(i)
    groups = {}
    for i in range(len(copies)):
        groups.setdefault(root(i), []).append(copies[i])
    rejected = 0
    sequences = []
    for members in groups.values():
        best = max(members, key=lambda c: (c["margin"], len(c["seq"].steps), -c["start"]))
        if _not_live(positions, best["start"]):
            rejected += 1
            continue
        seq = best["seq"]
        sequences.append(MergedSequenceV1(
            actor_player_id=seq.actor_player_id,
            actor_sentinel=seq.actor_sentinel,
            summary=seq.summary,
            bounds=TimeRangeMs(start_ms=seq.start_ms, end_ms=seq.end_ms),
            steps=tuple(s.published() for s in seq.steps),
        ))
    sequences.sort(key=lambda s: (s.bounds.start_ms, str(s.actor_player_id or s.actor_sentinel)))
    return SequenceBatchV1(
        sequences=tuple(sequences),
        census_sequences=len(copies),
        merged_copies=len(copies) - len(groups),
        rejected_not_live=rejected,
    )


def sequence_events(run, batch, common):
    """Published technique events, one per merged sequence (``common`` from normalization)."""
    events = []
    for index, sequence in enumerate(SequenceBatchV1.model_validate(batch).sequences):
        steps, bounds = sequence.steps, sequence.bounds
        derived = sequence_event_fields(steps)
        outcome = derived["outcome"]
        if not run.spec.scope.contains(bounds):
            raise ValueError("sequence bounds exceed the selected scope")
        decider = next((s for s in steps if s.outcome.value == "successful"), steps[-1])
        events.append(
            TechniqueEventV1(
                **common(run, uuid5(run.spec.run_id, f"sequence:v1:{index}"), bounds),
                **derived,
                actor_player_id=sequence.actor_player_id,
                actor_sentinel=sequence.actor_sentinel,
                notes=sequence.summary,
                evidence=sequence.summary,
                outcome_evidence=decider.evidence or sequence.summary,
                # Steps are not timed: the whole sequence is the outcome's evidence window.
                outcome_bounds=None if outcome == "unclear" else bounds,
                steps=steps,
            )
        )
    return events
