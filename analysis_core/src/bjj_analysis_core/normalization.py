"""Exact-coverage normalization into publishable production components.

Only the trusted host supplies a registered run and persisted completion time.
No database/provider/route imports. IDs derive from run + component + content;
replayed outputs produce identical revisions and never receive review acceptance.
"""

from uuid import uuid5

from bjj_analysis_core import CORE_VERSION
from bjj_analysis_core.packs import pack_for
from pydantic import AwareDatetime, TypeAdapter
from shared_lib.models.analysis_contracts import TimeRangeMs
from shared_lib.models.analysis_results import (
    AnalysisRevisionV1,
    PositionIntervalV1,
    TechniqueEventV1,
)
from shared_lib.models.analysis_run_record import RegisteredAIRunV1
from shared_lib.models.analysis_timeline import content_sha256

from shared_lib.models.sport_taxonomy import vocabulary as _vocabulary

STRIKE_ACTIONS = frozenset(_vocabulary("mma.v1").strike_actions)


def _context(run, completed_at):
    run = RegisteredAIRunV1.model_validate(run)
    completed_at = TypeAdapter(AwareDatetime).validate_python(completed_at)
    if run.spec.core_version != CORE_VERSION:
        raise ValueError("registered run requires a different core version")
    pack_for(run)  # an unknown workflow or taxonomy mismatch fails here
    if completed_at < run.registered_at:
        raise ValueError("component completion precedes registered run")
    return run, completed_at


def _common(run, item_id, bounds):
    return dict(
        analysis_id=run.spec.analysis_id,
        scope_id=run.spec.scope_id,
        source_run_id=run.spec.run_id,
        origin="ai",
        item_id=item_id,
        bounds=bounds,
        player_ids=tuple(a.player_id for a in run.spec.athletes),
        taxonomy_id=run.spec.taxonomy_id,
    )


def _revision(run, completed_at, component, items):
    # The temporary revision ID is not emitted or included in final ID input.
    # Derive the final ID from every semantic field, including persisted time.
    draft = AnalysisRevisionV1(
        revision_id=uuid5(run.spec.run_id, f"normalization:v1:{component}"),
        parent_revision_id=run.baseline_revision_id,
        analysis_id=run.spec.analysis_id,
        edit_version=run.spec.base_edit_version,
        scope_id=run.spec.scope_id,
        source_sha256=run.spec.source.sha256,
        scope=run.spec.scope,
        player_ids=tuple(a.player_id for a in run.spec.athletes),
        created_at=completed_at,
        author=run.spec.owner,
        taxonomy_id=run.spec.taxonomy_id,
        **{component: tuple(items)},
    )
    final = AnalysisRevisionV1.model_validate(
        {
            **draft.model_dump(),
            "revision_id": uuid5(
                run.spec.run_id, f"normalized:v1:{component}:{content_sha256(draft)}"
            ),
        }
    )
    final.validate_against_run(run.spec)
    return final


def normalize_positions(run, scan, critiques, *, completed_at):
    """Every rough transition needs one decision; no missing critiques or sorting.

    Rejected transitions vanish only by explicit decision. Confirmed corrected
    boundaries stay in scan order and must be strictly inside scope/increasing.
    Do not merge adjacent states: doing so can discard evidence or a reset cue.
    """
    run, completed_at = _context(run, completed_at)
    pack = pack_for(run)
    scan = pack.position_scan.model_validate(scan)
    # critiques=None: the run has no position check; the scan's transitions stand.
    decisions = None if critiques is None else tuple(
        pack.position_critique.model_validate(c) for c in critiques
    )
    by_index = {} if decisions is None else {c.transition_index: c for c in decisions}
    if decisions is not None and (len(by_index) != len(decisions) or set(by_index) != set(
        range(len(scan.transitions))
    )):
        raise ValueError(
            "exactly one critique is required for every scanned transition"
        )
    previous_rough = run.spec.scope.start_ms
    for transition in scan.transitions:
        if not previous_rough < transition.at_ms < run.spec.scope.end_ms:
            raise ValueError(
                "rough boundaries must be strictly increasing inside scope"
            )
        previous_rough = transition.at_ms
    start, state, intervals = run.spec.scope.start_ms, scan.initial_state, []
    boundaries = []
    for index, transition in enumerate(scan.transitions):
        if decisions is None:
            boundaries.append((transition.at_ms, transition.state_after))
            continue
        decision = by_index[index]
        if decision.confirmed:
            boundaries.append((decision.corrected_at_ms, decision.state_after))
    boundaries.append((run.spec.scope.end_ms, None))
    for end, after in boundaries:
        if not start < end <= run.spec.scope.end_ms:
            raise ValueError(
                "corrected boundaries must be strictly increasing inside scope"
            )
        if after is not None and end == run.spec.scope.end_ms:
            raise ValueError("a transition must occur before scope end")
        bounds = TimeRangeMs(start_ms=start, end_ms=end)
        seed = PositionIntervalV1(
            **_common(run, uuid5(run.spec.run_id, "position:seed"), bounds),
            **state.model_dump(),
        )
        interval = PositionIntervalV1.model_validate(
            {
                **seed.model_dump(),
                "item_id": uuid5(
                    run.spec.run_id, f"position:v1:{content_sha256(seed)}"
                ),
            }
        )
        intervals.append(interval)
        start, state = end, after
    return _revision(run, completed_at, "positions", intervals)


def normalize_techniques(run, candidates, decisions, *, completed_at, strikes=()):
    """One terminal decision per candidate; no silent drops or inferred outcomes.

    Candidates/decisions must be checkpointed by the host. Input order is kept;
    overlapping distinct attempts survive. Known outcome evidence outside an
    event's clip is rejected, never clamped or assigned a fixed padding tail.
    """
    run, completed_at = _context(run, completed_at)
    pack = pack_for(run)
    candidates = tuple(pack.technique_candidate.model_validate(c) for c in candidates)
    candidate_ids = tuple(c.candidate_id for c in candidates)
    # decisions=None: the run has no validator; each analysis stands as proposed.
    unchecked = decisions is None
    decisions = () if unchecked else tuple(
        pack.technique_decision.model_validate(d) for d in decisions
    )
    by_id = {d.candidate_id: d for d in decisions}
    if len(set(candidate_ids)) != len(candidate_ids) or (not unchecked and (
        len(by_id) != len(decisions)
        or set(by_id) != set(candidate_ids)
    )):
        raise ValueError(
            "exactly one decision is required for every technique candidate"
        )
    events = []
    for candidate in candidates:
        if not run.spec.scope.contains(candidate.observation_window):
            raise ValueError("candidate observation window exceeds selected scope")
        if unchecked:
            data = candidate.proposed
        else:
            decision = by_id[candidate.candidate_id]
            if decision.verdict == "ditch":
                continue
            data = decision.technique
        if pack.windowed_strikes and any(
            a.value in STRIKE_ACTIONS for a in data.action_classes
        ):
            # Windowed strike workflow: strikes come only from the census. A position-
            # change candidate the chain labels as a strike is the census's strike seen
            # a second time (it duplicated knockdowns in the Round 1 run), not a new one.
            continue
        if not unchecked and decision.verdict == "agree" and data != candidate.proposed:
            raise ValueError("agree must preserve the proposed technique exactly")
        if not candidate.observation_window.contains(data.bounds):
            raise ValueError("technique bounds exceed the supplied observation window")
        event = TechniqueEventV1(
            **_common(
                run,
                uuid5(run.spec.run_id, f"technique:v1:{candidate.candidate_id}"),
                data.bounds,
            ),
            **data.model_dump(exclude={"schema_version", "bounds"}),
        )
        events.append(event)
    # Windowed strike census (mma-windowed-strikes): already merged, one event each.
    if strikes and not pack.windowed_strikes:
        raise ValueError("only the windowed strike workflow publishes census strikes")
    for index, data in enumerate(strikes):
        data = pack.technique_analysis.model_validate(data)
        if not run.spec.scope.contains(data.bounds):
            raise ValueError("strike bounds exceed the selected scope")
        events.append(
            TechniqueEventV1(
                **_common(run, uuid5(run.spec.run_id, f"strike:v1:{index}"), data.bounds),
                **data.model_dump(exclude={"schema_version", "bounds"}),
            )
        )
    return _revision(run, completed_at, "techniques", events)
