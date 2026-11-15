"""Technique sequences (workflow ``mma-sequences-ground``): pure functions only.

The scope is tiled into short overlapping windows; one census call per window returns
every fighter's chain of techniques. This module tiles the scope, merges the windows'
answers and turns them into published technique events with steps.

Merge rules (plan 2026-11-15-mma-technique-sequences):

1. Copies of one sequence: two sequences from adjacent windows with the same fighter
   that share a step (same class, starts within ``SAME_STRIKE_MS`` for a strike or
   ``SAME_STEP_MS`` otherwise) are one sequence seen twice. Their steps are joined;
   of two copies of one step, the copy farther from its own window's edges is kept
   (it saw more before and after it, e.g. the outcome of a step cut off at a window's
   end). Steps of one answer are never twins: within one window a fast combination
   is real.
2. Steps in footage that is not live (an inactive position span) are rejected.
3. A pause longer than ``SEQUENCE_GAP_MS`` starts a new sequence: that is the
   definition of a sequence, so a census sequence with such a pause is split.
"""

from uuid import uuid5

from bjj_analysis_core.mma_outputs import MergedSequenceV1, SequenceBatchV1, SequenceCensusV1
from shared_lib.models.analysis_contracts import TimeRangeMs
from shared_lib.models.sport_taxonomy import vocabulary
from shared_lib.models.analysis_results import (
    SEQUENCE_GAP_MS,
    TechniqueEventV1,
    sequence_classes,
    sequence_outcome,
)

SAME_STRIKE_MS = 400
SAME_STEP_MS = 1000
STRIKES = frozenset(vocabulary("mma.v1").strike_actions)


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


def _actor(sequence):
    return sequence.actor_player_id or sequence.actor_sentinel


def _same_step(a, b):
    tolerance = SAME_STRIKE_MS if a.action.value in STRIKES else SAME_STEP_MS
    return a.action == b.action and abs(a.bounds.start_ms - b.bounds.start_ms) <= tolerance


def _margin(step, window):
    return min(step.bounds.start_ms - window.start_ms, window.end_ms - step.bounds.end_ms)


def _not_live(positions, step):
    return any(
        p.observation == "inactive"
        and p.bounds.start_ms <= step.bounds.start_ms < p.bounds.end_ms
        for p in positions.positions or ()
    )


def _split(steps):
    """Runs of steps whose pauses are at most SEQUENCE_GAP_MS."""
    runs, reach = [[steps[0]]], steps[0].bounds.end_ms
    for step in steps[1:]:
        if step.bounds.start_ms - reach > SEQUENCE_GAP_MS:
            runs.append([])
        runs[-1].append(step)
        reach = step.bounds.end_ms if len(runs[-1]) == 1 else max(reach, step.bounds.end_ms)
    return runs


def merge_sequences(run, config, censuses, positions):
    """One census per window -> final sequences in time order (SequenceBatchV1)."""
    windows = sequence_windows(run, config)
    censuses = tuple(SequenceCensusV1.model_validate(c) for c in censuses)
    if [c.window_index for c in censuses] != list(range(len(windows))):
        raise ValueError("exactly one census is required for every sequence window")
    copies = []  # (window index, actor, summary, [(step, margin)])
    for index, census in enumerate(censuses):
        window = windows[index]
        for sequence in census.sequences:
            steps = [s.published() for s in sequence.steps]
            if not all(window.start_ms <= s.bounds.start_ms and s.bounds.end_ms <= window.end_ms for s in steps):
                raise ValueError("a census step falls outside its window")
            copies.append(
                (index, _actor(sequence), sequence, [(s, _margin(s, window)) for s in steps])
            )
    # Union copies of one sequence across adjacent windows (union-find over copies).
    parent = list(range(len(copies)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (wi, actor_i, _, steps_i) in enumerate(copies):
        for j in range(i + 1, len(copies)):
            wj, actor_j, _, steps_j = copies[j]
            if wj - wi != 1 or actor_i != actor_j:
                continue
            if any(_same_step(a, b) for a, _ in steps_i for b, _ in steps_j):
                parent[root(j)] = root(i)
    groups = {}
    for i in range(len(copies)):
        groups.setdefault(root(i), []).append(i)
    merged_copies = len(copies) - len(groups)
    rejected = splits = 0
    sequences = []
    for members in groups.values():
        first = copies[members[0]][2]
        kept = []  # [(step, margin, copy)]
        for i in members:
            earlier = len(kept)  # only steps of earlier copies can be twins
            taken = set()
            for step, margin in copies[i][3]:
                twins = [
                    k for k in range(earlier)
                    if k not in taken and _same_step(step, kept[k][0])
                ]
                if not twins:
                    kept.append((step, margin, i))
                    continue
                k = min(twins, key=lambda k: abs(kept[k][0].bounds.start_ms - step.bounds.start_ms))
                taken.add(k)
                if margin > kept[k][1]:
                    kept[k] = (step, margin, i)
        steps = sorted((s for s, _, _ in kept), key=lambda s: (s.bounds.start_ms, s.bounds.end_ms))
        live = [s for s in steps if not _not_live(positions, s)]
        rejected += len(steps) - len(live)
        if not live:
            continue
        runs = _split(live)
        splits += len(runs) - 1
        summary = max((copies[i][2] for i in members), key=lambda s: len(s.steps)).summary
        for steps_run in runs:
            sequences.append(
                MergedSequenceV1(
                    actor_player_id=first.actor_player_id,
                    actor_sentinel=first.actor_sentinel,
                    summary=summary,
                    steps=tuple(steps_run),
                )
            )
    sequences.sort(key=lambda s: (s.steps[0].bounds.start_ms, str(s.actor_player_id or s.actor_sentinel)))
    return SequenceBatchV1(
        sequences=tuple(sequences),
        census_sequences=len(copies),
        merged_copies=merged_copies,
        rejected_not_live_steps=rejected,
        splits=splits,
    )


def _deciding_step(steps, outcome):
    if outcome == "successful":
        return next(s for s in steps if s.outcome.value == "successful")
    if outcome == "unsuccessful":
        return steps[-1]
    return None


def sequence_events(run, batch, common):
    """Published technique events, one per merged sequence (``common`` from normalization)."""
    events = []
    for index, sequence in enumerate(SequenceBatchV1.model_validate(batch).sequences):
        steps = sequence.steps
        outcome = sequence_outcome(steps)
        decider = _deciding_step(steps, outcome)
        bounds = TimeRangeMs(
            start_ms=steps[0].bounds.start_ms,
            end_ms=max(s.bounds.end_ms for s in steps),
        )
        if not run.spec.scope.contains(bounds):
            raise ValueError("sequence bounds exceed the selected scope")
        events.append(
            TechniqueEventV1(
                **common(run, uuid5(run.spec.run_id, f"sequence:v1:{index}"), bounds),
                action_classes=sequence_classes(steps),
                actor_player_id=sequence.actor_player_id,
                actor_sentinel=sequence.actor_sentinel,
                outcome=outcome,
                notes=sequence.summary,
                evidence=sequence.summary,
                outcome_evidence=decider.evidence if decider is not None else sequence.summary,
                outcome_bounds=None if decider is None else decider.bounds,
                steps=steps,
            )
        )
    return events
