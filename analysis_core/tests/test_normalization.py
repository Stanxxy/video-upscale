import pytest
from bjj_analysis_core.normalization import normalize_positions, normalize_techniques
from conftest import candidate, decision, state, technique


def position_inputs():
    scan = dict(
        initial_state=state(),
        transitions=[
            dict(
                at_ms=2500,
                state_after=state("back_control", ("back_controller", "back_defender")),
            ),
            dict(
                at_ms=8000,
                state_after=state("turtle", ("turtle_opponent", "turtled_player")),
            ),
        ],
    )
    critiques = [
        dict(
            transition_index=i,
            confirmed=True,
            corrected_at_ms=t,
            state_after=scan["transitions"][i]["state_after"],
            reason="corrected visible boundary",
        )
        for i, t in enumerate((3000, 7000))
    ]
    return scan, critiques


def test_corrected_positions_exact_coverage_semantic_roles_and_replay(
    run, completed_at
):
    scan, critiques = position_inputs()
    result = normalize_positions(run, scan, critiques, completed_at=completed_at)
    assert [
        (p.bounds.start_ms, p.bounds.end_ms, p.position.value) for p in result.positions
    ] == [
        (1000, 3000, "standing"),
        (3000, 7000, "back_control"),
        (7000, 11000, "turtle"),
    ]
    assert tuple(r.value for r in result.positions[1].roles) == (
        "back_controller",
        "back_defender",
    )
    assert sum(p.bounds.duration_ms for p in result.positions) == 10000
    assert result.techniques is None and result.latest_reviews == ()
    assert (
        result.parent_revision_id == run.baseline_revision_id
        and result.edit_version == 2
    )
    assert all(
        p.source_run_id == run.spec.run_id and p.origin == "ai"
        for p in result.positions
    )
    assert result == normalize_positions(
        run, scan, tuple(reversed(critiques)), completed_at=completed_at
    )


def test_explicit_rejection_drops_only_that_boundary(run, completed_at):
    scan, critiques = position_inputs()
    critiques[0].update(confirmed=False, corrected_at_ms=None, state_after=None)
    result = normalize_positions(run, scan, critiques, completed_at=completed_at)
    assert [p.position.value for p in result.positions] == ["standing", "turtle"]


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate",
        "foreign_index",
        "crossed",
        "scope_start",
        "scope_end",
        "relative_seconds",
        "bool_time",
    ],
)
def test_bad_boundaries_and_incomplete_critique_fail(run, completed_at, failure):
    scan, critiques = position_inputs()
    if failure == "missing":
        critiques.pop()
    elif failure == "duplicate":
        critiques.append(critiques[0])
    elif failure == "foreign_index":
        critiques[1]["transition_index"] = 7
    else:
        critiques[0]["corrected_at_ms"] = {
            "crossed": 9000,
            "scope_start": 1000,
            "scope_end": 11000,
            "relative_seconds": 3.0,
            "bool_time": True,
        }[failure]
    with pytest.raises(ValueError):
        normalize_positions(run, scan, critiques, completed_at=completed_at)


@pytest.mark.parametrize(
    "position,roles,changes",
    [
        ("back_control", ("top", "bottom"), {}),
        ("turtle", ("pinning_player", "pinned_player"), {}),
        ("other", ("neutral", "neutral"), {"notes": "back mount"}),
        ("other", ("neutral", "neutral"), {"notes": "armbar"}),
        ("standing", ("unknown", "unknown"), {}),
    ],
)
def test_no_role_guess_or_other_alias_conversion(
    run, completed_at, position, roles, changes
):
    value = state(position, roles)
    value.update(changes)
    with pytest.raises(ValueError):
        normalize_positions(
            run,
            dict(initial_state=value, transitions=[]),
            [],
            completed_at=completed_at,
        )


@pytest.mark.parametrize(
    "observation,note_field,roles",
    [
        ("inactive", "interruption_note", ("not_applicable", "not_applicable")),
        ("unobservable", "uncertainty_note", ("unknown", "unknown")),
    ],
)
def test_reset_and_unobservable_not_conflated(
    run, completed_at, observation, note_field, roles
):
    value = state(None, roles)
    value.update(
        observation=observation,
        **{
            note_field: (
                "visible reset" if observation == "inactive" else "camera cutaway"
            )
        },
    )
    result = normalize_positions(
        run, dict(initial_state=value, transitions=[]), [], completed_at=completed_at
    )
    assert (
        result.positions[0].observation == observation
        and result.positions[0].position is None
    )


def test_preserves_repeated_state_boundaries_and_evidence(run, completed_at):
    first, after = state(), state()
    after["evidence"] = "second camera-visible relationship"
    scan = dict(initial_state=first, transitions=[dict(at_ms=5000, state_after=after)])
    critiques = [
        dict(
            transition_index=0,
            confirmed=True,
            corrected_at_ms=5000,
            state_after=after,
            reason="retained evidence",
        )
    ]
    result = normalize_positions(run, scan, critiques, completed_at=completed_at)
    assert [p.evidence for p in result.positions] == [
        first["evidence"],
        after["evidence"],
    ]


def test_outcome_complete_techniques_and_zero_lane(run, completed_at):
    result = normalize_techniques(
        run, [candidate()], [decision()], completed_at=completed_at
    )
    event = result.techniques[0]
    assert event.bounds.end_ms == 6000 and event.outcome_bounds.end_ms == 5900
    assert result.positions is None and result.latest_reviews == ()
    assert result == normalize_techniques(
        run, [candidate()], [decision()], completed_at=completed_at
    )
    empty = normalize_techniques(run, [], [], completed_at=completed_at)
    assert empty.techniques == () and empty.positions is None


@pytest.mark.parametrize(
    "failure",
    [
        "early_end",
        "missing_outcome",
        "unknown_actor",
        "false_agree",
        "outside_window",
        "failed_ditch",
        "missing_decision",
        "extra_decision",
    ],
)
def test_bad_technique_results_never_clamped_or_silently_dropped(
    run, completed_at, failure
):
    source, final = candidate(), decision()
    if failure == "early_end":
        final.update(
            verdict="correct",
            technique=technique(bounds=dict(start_ms=2000, end_ms=5000)),
        )
    elif failure == "missing_outcome":
        final.update(verdict="correct", technique=technique(outcome_bounds=None))
    elif failure == "unknown_actor":
        final.update(verdict="correct", technique=technique(actor_player_id="stranger"))
    elif failure == "false_agree":
        final["technique"]["notes"] = "different proposal"
    elif failure == "outside_window":
        final.update(
            verdict="correct", technique=technique(bounds=dict(start_ms=0, end_ms=6000))
        )
    elif failure == "failed_ditch":
        final.update(verdict="ditch")
    elif failure == "extra_decision":
        final["candidate_id"] = "not-this-candidate"
    with pytest.raises(ValueError):
        normalize_techniques(
            run,
            [source],
            [] if failure == "missing_decision" else [final],
            completed_at=completed_at,
        )


def test_explicit_non_event_ditch_and_unclear_outcome(run, completed_at):
    discarded = decision(
        verdict="ditch",
        intentional_technique_visible=False,
        technique=None,
        reason="only static holding",
    )
    assert (
        normalize_techniques(
            run, [candidate()], [discarded], completed_at=completed_at
        ).techniques
        == ()
    )
    unclear = technique(
        outcome="unclear",
        outcome_bounds=None,
        outcome_evidence="footage ends before result",
    )
    result = normalize_techniques(
        run,
        [candidate()],
        [decision(verdict="correct", technique=unclear)],
        completed_at=completed_at,
    )
    assert result.techniques[0].outcome == "unclear"


def test_distinct_overlapping_attempts_survive(run, completed_at):
    a, b = candidate(), candidate(candidate_id="scan-2")
    d1, d2 = decision(), decision(candidate_id="scan-2")
    result = normalize_techniques(run, [a, b], [d1, d2], completed_at=completed_at)
    assert len(result.techniques) == 2
    assert result.techniques[0].item_id != result.techniques[1].item_id


def test_wrong_core_or_pre_registration_completion_rejected(run, completed_at):
    with pytest.raises(ValueError):
        normalize_techniques(run, [], [], completed_at=run.spec.admitted_at)
    changed = run.model_copy(
        update={"spec": run.spec.model_copy(update={"core_version": "different"})}
    )
    with pytest.raises(ValueError):
        normalize_techniques(changed, [], [], completed_at=completed_at)
