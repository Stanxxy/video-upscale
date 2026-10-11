"""Zero-provider-call candidate union and outcome-aware temporal handoff.

Adapted from the lab's one-to-one nearest pairing; integer source milliseconds,
strict production roles and immutable admitted configuration replace lab payloads.
"""

from uuid import uuid5

from bjj_analysis_core.outputs import (
    CandidateBatchV1,
    DiscoveryCandidateV1,
    TechniqueAnalysisInputV1,
    TechniqueScanV1,
    TemporalCritiqueV1,
)
from bjj_analysis_core.requests import canonical_json, load_config
from shared_lib.models.analysis_contracts import TimeRangeMs
from shared_lib.models.analysis_results import AnalysisRevisionV1
from shared_lib.models.analysis_timeline import content_sha256


def position_context(run, revision, bounds):
    """Recompute the fallible before/after context at the current corrected bounds."""
    revision = AnalysisRevisionV1.model_validate(revision)
    revision.validate_against_run(run.spec)
    bounds = TimeRangeMs.model_validate(bounds)
    if revision.positions is None or revision.techniques is not None:
        raise ValueError("context requires a position-only revision")
    if not run.spec.scope.contains(bounds):
        raise ValueError("context bounds exceed selected scope")
    intervals = revision.positions

    def state_at(at):
        return next(
            p
            for p in intervals
            if p.bounds.start_ms <= at < p.bounds.end_ms
            or (at == run.spec.scope.end_ms == p.bounds.end_ms)
        ).model_dump(mode="json")

    return dict(
        position_before=state_at(bounds.start_ms),
        position_after=state_at(bounds.end_ms),
        overlapping_positions=[
            p.model_dump(mode="json")
            for p in intervals
            if p.bounds.start_ms < bounds.end_ms and p.bounds.end_ms > bounds.start_ms
        ],
    )


def build_candidates(run, config_bytes, scan, positions):
    run, config = load_config(run, config_bytes)
    cfg = config.candidates
    scan = TechniqueScanV1.model_validate(scan)
    positions = AnalysisRevisionV1.model_validate(positions)
    positions.validate_against_run(run.spec)
    if positions.positions is None or positions.techniques is not None:
        raise ValueError("candidate union requires a position-only revision")
    previous_start = run.spec.scope.start_ms
    for event in scan.events:
        if (
            not run.spec.scope.contains(event.bounds)
            or event.bounds.start_ms < previous_start
        ):
            raise ValueError("scan events must be ordered inside selected scope")
        previous_start = event.bounds.start_ms
    transitions = []
    for before, after in zip(positions.positions, positions.positions[1:]):
        # No reset/cutaway candidates, nor an evidence-only duplicate state.
        if before.observation == after.observation == "observed" and (
            before.position != after.position
            or before.roles != after.roles
            or before.notes != after.notes
        ):
            transitions.append((before, after))
    pairs = []
    for ti, (_, after) in enumerate(transitions):
        at = after.bounds.start_ms
        for si, event in enumerate(scan.events):
            b = event.bounds
            if (
                b.start_ms - cfg.merge_tolerance_ms
                <= at
                <= b.end_ms + cfg.merge_tolerance_ms
            ):
                # Twice the midpoint distance avoids float rounding and preserves ties.
                pairs.append((abs(b.start_ms + b.end_ms - 2 * at), ti, si))
    matched, used = {}, set()
    for _, ti, si in sorted(pairs):
        if si not in matched and ti not in used:
            matched[si] = ti
            used.add(ti)
    candidates = []
    for si, event in enumerate(scan.events):
        ti = matched.get(si)
        candidates.append(
            DiscoveryCandidateV1(
                candidate_id=str(
                    uuid5(run.spec.run_id, f"scan:v1:{si}:{content_sha256(event)}")
                ),
                **event.model_dump(),
                scan_index=si,
                transition_at_ms=(
                    None if ti is None else transitions[ti][1].bounds.start_ms
                ),
                sources=(
                    ("technique_scan",)
                    if ti is None
                    else ("technique_scan", "position_transition")
                ),
            )
        )
    for ti, (before, after) in enumerate(transitions):
        if ti in used:
            continue
        at = after.bounds.start_ms
        bounds = TimeRangeMs(
            start_ms=max(run.spec.scope.start_ms, at - cfg.transition_pre_ms),
            end_ms=min(run.spec.scope.end_ms, at + cfg.transition_post_ms),
        )
        candidates.append(
            DiscoveryCandidateV1(
                candidate_id=str(
                    uuid5(
                        run.spec.run_id,
                        f"transition:v1:{before.item_id}:{after.item_id}",
                    )
                ),
                bounds=bounds,
                description=(
                    f"Inspect movement near {at} ms: "
                    f"before={canonical_json(before.model_dump(mode='json', include={'position', 'roles', 'notes'}))}; "
                    f"after={canonical_json(after.model_dump(mode='json', include={'position', 'roles', 'notes'}))}. "
                    "Position change alone is not proof of an intentional technique."
                ),
                reasoning="Secondary candidate from a critiqued active-to-active position/role transition; confirm intentional body movement in the video.",
                scan_index=None,
                transition_at_ms=at,
                sources=("position_transition",),
            )
        )
    # Critique replay identity includes all persisted inputs, not just a scan
    # index. A changed position prior/config must not reuse an old decision.
    position_digest = content_sha256(positions)
    candidates = [
        DiscoveryCandidateV1.model_validate(
            {
                **c.model_dump(),
                "candidate_id": str(
                    uuid5(
                        run.spec.run_id,
                        f"candidate:v2:{run.spec.effective_config.sha256}:{position_digest}:{content_sha256(c)}",
                    )
                ),
            }
        )
        for c in candidates
    ]
    candidates.sort(key=lambda c: (c.bounds.start_ms, c.bounds.end_ms, c.candidate_id))
    return CandidateBatchV1(
        candidates=tuple(candidates),
        scan_count=len(scan.events),
        transition_count=len(transitions),
        matched_count=len(matched),
    )


def critique_window(run, config_bytes, candidate):
    run, config = load_config(run, config_bytes)
    candidate = DiscoveryCandidateV1.model_validate(candidate)
    if not run.spec.scope.contains(candidate.bounds):
        raise ValueError("candidate exceeds selected scope")
    return TimeRangeMs(
        start_ms=max(
            run.spec.scope.start_ms,
            candidate.bounds.start_ms - config.candidates.critique_back_ms,
        ),
        end_ms=min(
            run.spec.scope.end_ms,
            candidate.bounds.end_ms + config.candidates.critique_forward_ms,
        ),
    )


def resolve_temporal_critiques(run, config_bytes, candidates, decisions):
    """No missing critique becomes an empty event; explicit rejects remain auditable.

    The host persists the original candidates AND all decisions before using this
    surviving analysis worklist. Inspection padding is never exported as an
    outcome-complete timestamp unless the critique actually observed that outcome.
    """
    run, _ = load_config(run, config_bytes)
    candidates = tuple(DiscoveryCandidateV1.model_validate(c) for c in candidates)
    decisions = tuple(TemporalCritiqueV1.model_validate(d) for d in decisions)
    by_id = {d.candidate_id: d for d in decisions}
    if (
        len({c.candidate_id for c in candidates}) != len(candidates)
        or len(by_id) != len(decisions)
        or set(by_id) != {c.candidate_id for c in candidates}
    ):
        raise ValueError("exactly one temporal critique is required per candidate")
    work = []
    for candidate in candidates:
        window = critique_window(run, config_bytes, candidate)
        decision = by_id[candidate.candidate_id]
        if not decision.confirmed:
            continue
        if not window.contains(decision.bounds):
            raise ValueError("temporal correction exceeds observed window")
        if (
            decision.end_status == "window_exhausted"
            and decision.bounds.end_ms != window.end_ms
        ):
            raise ValueError(
                "exhausted outcome must retain evidence through window end"
            )
        work.append(
            TechniqueAnalysisInputV1(
                candidate_id=candidate.candidate_id,
                description=candidate.description,
                bounds=decision.bounds,
                end_status=decision.end_status,
                temporal_reason=decision.reason,
            )
        )
    return tuple(work)
