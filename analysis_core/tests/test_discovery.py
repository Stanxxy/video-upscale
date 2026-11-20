import hashlib
import json

import pytest
from bjj_analysis_core.discovery import (
    build_candidates,
    critique_window,
    position_context,
    resolve_temporal_critiques,
)
from bjj_analysis_core.normalization import normalize_positions, normalize_techniques
from bjj_analysis_core.outputs import CandidateConfigV1
from bjj_analysis_core.requests import parse_response, prepare_call, prompt_bundle
from conftest import state, technique


def event(start=2000, end=4000, description="A circles behind B"):
    return dict(
        bounds=dict(start_ms=start, end_ms=end),
        description=description,
        reasoning="visible deliberate body movement",
    )


def timeline(run, completed_at, changes=()):
    return normalize_positions(
        run,
        dict(
            initial_state=state(),
            transitions=[dict(at_ms=t, state_after=s) for t, s in changes],
        ),
        [
            dict(
                transition_index=i,
                confirmed=True,
                corrected_at_ms=t,
                state_after=s,
                reason="visible positional change",
            )
            for i, (t, s) in enumerate(changes)
        ],
        completed_at=completed_at,
    )


def temporal(candidate, **changes):
    data = dict(
        candidate_id=candidate.candidate_id,
        confirmed=True,
        bounds=dict(start_ms=2000, end_ms=7000),
        end_status="outcome_visible",
        reason="setup visible at start; defense completed at end",
    )
    data.update(changes)
    return data


def test_one_to_one_union_keeps_scan_movement_and_each_distinct_transition(
    run, configuration, completed_at
):
    positions = timeline(
        run,
        completed_at,
        [
            (4000, state("turtle", ("turtled_player", "turtle_opponent"))),
            (6000, state("back_control", ("back_defender", "back_controller"))),
        ],
    )
    scan = dict(events=[event(2000, 8000)])
    batch = build_candidates(run, configuration, scan, positions)
    assert (batch.scan_count, batch.transition_count, batch.matched_count) == (1, 2, 1)
    assert batch.provider_call_count == 0 and len(batch.candidates) == 2
    scanned = next(c for c in batch.candidates if c.scan_index == 0)
    assert scanned.description == scan["events"][0]["description"]
    assert scanned.bounds.model_dump() == scan["events"][0]["bounds"]
    assert scanned.transition_at_ms == 4000  # deterministic equal-distance tie
    assert scanned.sources == ("technique_scan", "position_transition")
    other = next(c for c in batch.candidates if c.scan_index is None)
    assert other.transition_at_ms == 6000 and other.bounds.start_ms == 2000
    assert build_candidates(run, configuration, scan, positions) == batch


@pytest.mark.parametrize("observation", ["inactive", "unobservable", "unchanged"])
def test_no_synthetic_attempt_for_reset_cutaway_or_evidence_only_boundary(
    run, configuration, completed_at, observation
):
    after = state()
    if observation != "unchanged":
        after.update(
            observation=observation,
            position=None,
            roles=(
                ("not_applicable", "not_applicable")
                if observation == "inactive"
                else ("unknown", "unknown")
            ),
            uncertainty_note=(
                "camera loses both athletes" if observation == "unobservable" else None
            ),
            interruption_note="referee reset" if observation == "inactive" else None,
        )
    positions = timeline(run, completed_at, [(4000, after), (7000, state())])
    batch = build_candidates(run, configuration, dict(events=[event()]), positions)
    assert batch.transition_count == 0 and len(batch.candidates) == 1
    assert batch.candidates[0].sources == ("technique_scan",)


def test_role_change_is_a_transition_and_bounds_clip_only_to_scope(
    run, configuration, completed_at
):
    positions = timeline(
        run,
        completed_at,
        [
            (1500, state("turtle", ("turtled_player", "turtle_opponent"))),
            (10500, state("turtle", ("turtle_opponent", "turtled_player"))),
        ],
    )
    batch = build_candidates(run, configuration, dict(events=[]), positions)
    assert batch.transition_count == 2
    assert [c.bounds.model_dump() for c in batch.candidates] == [
        dict(start_ms=1000, end_ms=3500),
        dict(start_ms=6500, end_ms=11000),
    ]


def test_two_overlapping_scans_are_never_collapsed(run, configuration, completed_at):
    positions = timeline(
        run,
        completed_at,
        [(4000, state("turtle", ("turtled_player", "turtle_opponent")))],
    )
    batch = build_candidates(
        run,
        configuration,
        dict(events=[event(), event(description="B counters A")]),
        positions,
    )
    assert len(batch.candidates) == 2 and batch.matched_count == 1
    assert len({c.candidate_id for c in batch.candidates}) == 2
    assert {c.description for c in batch.candidates} == {
        "A circles behind B",
        "B counters A",
    }


@pytest.mark.parametrize(
    "events",
    [
        [event(0, 4000)],
        [event(2000, 12000)],
        [event(4000, 6000), event()],
        [event(True, 4000)],
        [event(2000.5, 4000)],
    ],
)
def test_scan_invalid_times_reject_at_parser_and_union(
    run, configuration, completed_at, events
):
    request = prepare_call(
        run, configuration, "technique_scan", input_window=run.spec.scope, context={}
    )
    with pytest.raises(ValueError):
        parse_response(request, json.dumps(dict(events=events)))
    with pytest.raises(ValueError):
        build_candidates(
            run, configuration, dict(events=events), timeline(run, completed_at)
        )


def test_independent_scan_and_candidate_config_binding(
    run, configuration, completed_at
):
    positions = timeline(run, completed_at)
    with pytest.raises(ValueError, match="independent"):
        prepare_call(
            run,
            configuration,
            "technique_scan",
            input_window=run.spec.scope,
            context={},
            advisory_positions=positions,
        )
    with pytest.raises(ValueError, match="entire"):
        prepare_call(
            run,
            configuration,
            "technique_scan",
            input_window=dict(start_ms=2000, end_ms=4000),
            context={},
        )
    changed = json.loads(configuration)
    changed["candidates"]["merge_tolerance_ms"] = 0
    changed_bytes = json.dumps(changed).encode()
    with pytest.raises(ValueError, match="configuration bytes"):
        build_candidates(run, changed_bytes, dict(events=[]), positions)
    # A newly admitted artifact can legitimately change the deterministic knobs.
    newrun = run.model_copy(
        update={
            "spec": run.spec.model_copy(
                update={
                    "effective_config": run.spec.effective_config.model_copy(
                        update={"sha256": hashlib.sha256(changed_bytes).hexdigest()}
                    )
                }
            )
        }
    )
    assert (
        build_candidates(newrun, changed_bytes, dict(events=[]), positions).candidates
        == ()
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(transition_pre_ms=0, transition_post_ms=0),
        dict(merge_tolerance_ms=-1),
        dict(critique_forward_ms=4.0),
        dict(critique_back_ms=True),
    ],
)
def test_candidate_parameters_are_strict(change):
    with pytest.raises(ValueError):
        CandidateConfigV1(**change)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate",
        "foreign",
        "outside",
        "exhausted_early",
        "rejected_with_bounds",
    ],
)
def test_temporal_decisions_are_complete_and_window_bound(
    run, configuration, completed_at, failure
):
    candidate = build_candidates(
        run, configuration, dict(events=[event()]), timeline(run, completed_at)
    ).candidates[0]
    d = temporal(candidate)
    decisions = [d]
    if failure == "missing":
        decisions = []
    elif failure == "duplicate":
        decisions *= 2
    elif failure == "foreign":
        d["candidate_id"] = "foreign"
    elif failure == "outside":
        d["bounds"]["end_ms"] = 9000  # window ends at 8000
    elif failure == "exhausted_early":
        d["end_status"] = "window_exhausted"
    elif failure == "rejected_with_bounds":
        d["confirmed"] = False
    with pytest.raises(ValueError):
        resolve_temporal_critiques(run, configuration, [candidate], decisions)
    if failure not in ("missing", "duplicate"):
        request = prepare_call(
            run,
            configuration,
            "technique_critique",
            input_window=critique_window(run, configuration, candidate),
            context=dict(candidate=candidate),
        )
        with pytest.raises(ValueError):
            parse_response(request, json.dumps(d))


def test_exhausted_window_is_explicit_and_analysis_cannot_shorten_clip(
    run, configuration, completed_at
):
    positions = timeline(
        run,
        completed_at,
        [(6000, state("turtle", ("turtled_player", "turtle_opponent")))],
    )
    candidate = build_candidates(
        run, configuration, dict(events=[event()]), positions
    ).candidates[0]
    window = critique_window(run, configuration, candidate)
    request = prepare_call(
        run,
        configuration,
        "technique_critique",
        input_window=window,
        context=dict(candidate=candidate),
        advisory_positions=positions,
    )
    data = temporal(
        candidate,
        bounds=dict(start_ms=2000, end_ms=window.end_ms),
        end_status="window_exhausted",
        reason="OUTCOME_NOT_VISIBLE_BY_WINDOW_END",
    )
    critique = parse_response(request, json.dumps(data))
    work = resolve_temporal_critiques(run, configuration, [candidate], [critique])[0]
    assert work.end_status == "window_exhausted" and work.bounds.end_ms == 8000
    analysis = prepare_call(
        run,
        configuration,
        "technique_analyze",
        input_window=work.bounds,
        context=work,
        advisory_positions=positions,
    )
    context = json.loads(analysis.initial_prompt)["advisory_positions"]
    assert context["position_before"]["position"] == "standing"
    assert context["position_after"]["position"] == "turtle"
    proposed = technique(
        bounds=work.bounds.model_dump(),
        outcome="unclear",
        outcome_bounds=None,
        outcome_evidence="outcome not visible before inspection ended",
    )
    parsed = parse_response(analysis, json.dumps(proposed))
    assert parsed.bounds == work.bounds
    with pytest.raises(ValueError, match="authoritative"):
        parse_response(analysis, json.dumps(technique()))
    normalized = normalize_techniques(
        run,
        [
            dict(
                candidate_id=work.candidate_id,
                description=work.description,
                observation_window=work.bounds,
                proposed=proposed,
            )
        ],
        [
            dict(
                candidate_id=work.candidate_id,
                verdict="agree",
                intentional_technique_visible=True,
                technique=proposed,
                reason="still unclear in visible window",
            )
        ],
        completed_at=completed_at,
    )
    assert normalized.techniques[0].bounds.end_ms == 8000
    assert normalized.techniques[0].outcome == "unclear"


def test_rejected_non_event_and_empty_scan_need_no_analysis(
    run, configuration, completed_at
):
    positions = timeline(run, completed_at)
    assert (
        build_candidates(run, configuration, dict(events=[]), positions).candidates
        == ()
    )
    assert resolve_temporal_critiques(run, configuration, [], []) == ()
    candidate = build_candidates(
        run, configuration, dict(events=[event()]), positions
    ).candidates[0]
    d = temporal(
        candidate,
        confirmed=False,
        bounds=None,
        end_status=None,
        reason="ordinary grip adjustment only",
    )
    assert resolve_temporal_critiques(run, configuration, [candidate], [d]) == ()


def test_context_half_open_boundary_and_scope_end(run, completed_at):
    positions = timeline(
        run,
        completed_at,
        [(6000, state("turtle", ("turtled_player", "turtle_opponent")))],
    )
    context = position_context(run, positions, dict(start_ms=2000, end_ms=6000))
    assert context["position_after"]["position"] == "turtle"
    assert len(context["overlapping_positions"]) == 1
    assert (
        position_context(run, positions, run.spec.scope)["position_after"]["position"]
        == "turtle"
    )


def test_prompt_bundle_carries_temporal_and_independence_contracts():
    prompts = prompt_bundle()["prompts"]
    assert "Do not depend on a position timeline" in prompts["technique_scan"]
    assert "OUTCOME_NOT_VISIBLE_BY_WINDOW_END" in prompts["technique_critique"]
    assert "Do not blindly append a fixed tail" in prompts["technique_critique"]
    assert "corrected bounds exactly" in prompts["technique_analyze"]


def test_changed_position_prior_invalidates_cached_temporal_decision(
    run, configuration, completed_at
):
    original = timeline(run, completed_at)
    changed = timeline(
        run,
        completed_at,
        [(6000, state("turtle", ("turtled_player", "turtle_opponent")))],
    )
    scan = dict(events=[event()])
    first = build_candidates(run, configuration, scan, original).candidates[0]
    second = next(
        c
        for c in build_candidates(run, configuration, scan, changed).candidates
        if c.scan_index == 0
    )
    assert first.candidate_id != second.candidate_id
    assert first.description == second.description and first.bounds == second.bounds
    with pytest.raises(ValueError, match="exactly one"):
        resolve_temporal_critiques(run, configuration, [second], [temporal(first)])


@pytest.mark.parametrize(
    "position,roles",
    [
        ("other", ("neutral", "neutral")),
        ("standing", ("neutral", "neutral")),
    ],
)
def test_subposition_notes_retain_a_candidate_without_new_taxonomy_label(
    run, configuration, completed_at, position, roles
):
    before = state(position, roles)
    before["notes"] = "Both athletes face one another, standing with feet planted."
    after = {
        **before,
        "notes": "A kneels on both knees in front of standing B, without leg contact.",
    }
    # Domain taxonomy validity remains the shared validator's job; notes carry
    # geometric distinctions not encoded by a new enum.
    scan = dict(initial_state=before, transitions=[dict(at_ms=6000, state_after=after)])
    positions = normalize_positions(
        run,
        scan,
        [
            dict(
                transition_index=0,
                confirmed=True,
                corrected_at_ms=6000,
                state_after=after,
                reason="visible geometry change",
            )
        ],
        completed_at=completed_at,
    )
    batch = build_candidates(run, configuration, dict(events=[]), positions)
    assert batch.transition_count == 1 and batch.candidates[0].sources == (
        "position_transition",
    )
    assert before["notes"] in batch.candidates[0].description
    assert after["notes"] in batch.candidates[0].description


def test_candidate_parameters_change_pairing_only_under_new_admitted_config(
    run, configuration, completed_at
):
    positions = timeline(
        run,
        completed_at,
        [(5000, state("turtle", ("turtled_player", "turtle_opponent")))],
    )
    scan = dict(events=[event()])  # ends1000 ms before boundary
    assert build_candidates(run, configuration, scan, positions).matched_count == 1
    data = json.loads(configuration)
    data["candidates"]["merge_tolerance_ms"] = 0
    new_bytes = json.dumps(data).encode()
    new_run = run.model_copy(
        update={
            "spec": run.spec.model_copy(
                update={
                    "effective_config": run.spec.effective_config.model_copy(
                        update={"sha256": hashlib.sha256(new_bytes).hexdigest()}
                    )
                }
            )
        }
    )
    batch = build_candidates(new_run, new_bytes, scan, positions)
    assert batch.matched_count == 0 and len(batch.candidates) == 2
    old_id = (
        build_candidates(run, configuration, scan, positions).candidates[0].candidate_id
    )
    assert all(c.candidate_id != old_id for c in batch.candidates)


def test_old_core_and_unpadded_critique_request_fail_closed(
    run, configuration, completed_at
):
    positions = timeline(run, completed_at)
    old = run.model_copy(
        update={"spec": run.spec.model_copy(update={"core_version": "0.1.0"})}
    )
    with pytest.raises(ValueError, match="core or prompt"):
        build_candidates(old, configuration, dict(events=[]), positions)
    candidate = build_candidates(
        run, configuration, dict(events=[event()]), positions
    ).candidates[0]
    with pytest.raises(ValueError, match="padded"):
        prepare_call(
            run,
            configuration,
            "technique_critique",
            input_window=candidate.bounds,
            context=dict(candidate=candidate),
        )


def test_temporal_reordering_replays_and_foreign_position_context_rejects(
    run, configuration, completed_at
):
    positions = timeline(run, completed_at)
    candidates = build_candidates(
        run,
        configuration,
        dict(events=[event(), event(description="B counters")]),
        positions,
    ).candidates
    decisions = [temporal(c) for c in candidates]
    assert resolve_temporal_critiques(
        run, configuration, candidates, decisions
    ) == resolve_temporal_critiques(run, configuration, candidates, decisions[::-1])
    from uuid import uuid4

    wrong = positions.model_copy(update={"analysis_id": uuid4()})
    with pytest.raises(ValueError):
        build_candidates(run, configuration, dict(events=[]), wrong)
