"""Workflow mma-windowed-strikes-ground: ground phases, the re-scan call and its merge."""

import asyncio
import json
from datetime import timedelta

import pytest
from bjj_analysis_core import MMA_GROUND_WORKFLOW_ID
from bjj_analysis_core.execution import ProviderReplyV1
from bjj_analysis_core.executor import execute_pipeline
from bjj_analysis_core.grounds import apply_ground_rescans, ground_phases
from bjj_analysis_core.packs import MMA_GROUND, MMA_STRIKES
from bjj_analysis_core.requests import load_config, parse_response, prepare_call
from journal_fixture import DiskJournal
from shared_lib.models.analysis_contracts import CallUsageV1
from test_windowed_strikes import NODE, Sink, config_bytes, strikes_run


def ground_config(**changes):
    from bjj_analysis_core.requests import CoreConfigV1, prompt_bundle_sha256

    base = json.loads(config_bytes(MMA_STRIKES))
    base.update(prompt_bundle_sha256=prompt_bundle_sha256(MMA_GROUND), ground_rescan=NODE.model_dump(),
                ground_window_ms=5000, ground_lookback_ms=2000)
    base.update(changes)
    return CoreConfigV1.model_validate(base).model_dump_json().encode()


def state(position, roles, observation="observed"):
    return dict(observation=observation, position=position, roles=list(roles), notes="", evidence="seen",
                uncertainty_note=None, interruption_note=None, interruption_kind=None, cage_contact=None)


STANDING = state("striking_range", ("neutral", "neutral"))
# The rough scan puts b on top in side control (the error the re-scan fixes).
SCAN = dict(schema_version=1, initial_state=STANDING, transitions=[
    dict(at_ms=3000, state_after=state("side_control", ("pinned_player", "pinning_player"))),
    dict(at_ms=6000, state_after=state("scramble", ("neutral", "neutral"))),
    dict(at_ms=7000, state_after=STANDING),
])


def ground(run):
    data = ground_config()
    return strikes_run(run, data, MMA_GROUND_WORKFLOW_ID), data


def test_the_ground_node_belongs_exactly_to_the_ground_workflow(run):
    ground_run, data = ground(run)
    load_config(ground_run, data)
    without = ground_config(ground_rescan=None)
    with pytest.raises(ValueError, match="ground re-scan node"):
        load_config(strikes_run(run, without, MMA_GROUND_WORKFLOW_ID), without)
    plain = config_bytes()
    assert load_config(strikes_run(run, plain), plain)[1].ground_rescan is None


def test_ground_spans_merge_and_split_into_lookback_windows(run):
    ground_run, data = ground(run)
    _, config = load_config(ground_run, data)
    scan = MMA_GROUND.position_scan.model_validate(SCAN)
    phases = ground_phases(ground_run, config, scan)
    # Side control 3000-6000 and the scramble 6000-7000 are one phase, split at 5000 ms.
    assert [(c.start_ms, c.end_ms, w.start_ms, w.end_ms) for c, w in phases] == [(3000, 7000, 1000, 9000)]
    # A longer phase is split into windows of ground_window_ms, each with its own look-back.
    long = dict(SCAN, transitions=SCAN["transitions"][:1] + [dict(at_ms=10000, state_after=STANDING)])
    split = ground_phases(ground_run, config, MMA_GROUND.position_scan.model_validate(long))
    assert [(c.start_ms, c.end_ms, w.start_ms) for c, w in split] == [(3000, 8000, 1000), (8000, 10000, 6000)]


def test_the_merge_replaces_the_phase_and_restores_the_scan_after_it(run):
    ground_run, data = ground(run)
    _, config = load_config(ground_run, data)
    scan = MMA_GROUND.position_scan.model_validate(SCAN)
    phases = ground_phases(ground_run, config, scan)
    rescans = [dict(schema_version=1, phase_index=i, states=[
        dict(at_ms=w.start_ms, position="clinch", top_player_id=None, evidence="standing tie-up"),
        dict(at_ms=c.start_ms + 500, position="side_control", top_player_id="a", evidence="a in grey shorts on top"),
    ]) for i, (c, w) in enumerate(phases)]
    merged = apply_ground_rescans(MMA_GROUND, ground_run, config, scan, rescans)
    timeline = [(t.at_ms, t.state_after.position.value, tuple(r.value for r in t.state_after.roles))
                for t in merged.transitions]
    assert timeline[0] == (3000, "clinch", ("neutral", "neutral"))
    assert (3500, "side_control", ("pinning_player", "pinned_player")) in timeline  # a on top now
    assert timeline[-1][0] == 7000 and timeline[-1][1] == "striking_range"  # the scan resumes
    with pytest.raises(ValueError, match="every ground phase"):
        apply_ground_rescans(MMA_GROUND, ground_run, config, scan, rescans[:-1])


def test_the_ground_call_window_and_answers_are_checked(run):
    ground_run, data = ground(run)
    _, config = load_config(ground_run, data)
    core, window = ground_phases(ground_run, config, MMA_GROUND.position_scan.model_validate(SCAN))[0]
    context = dict(phase_index=0, core=core, window=window)
    call = prepare_call(ground_run, data, "ground_rescan", input_window=window, context=context)
    late = dict(start_ms=window.start_ms + 500, end_ms=window.end_ms)
    with pytest.raises(ValueError, match="look-back"):
        prepare_call(ground_run, data, "ground_rescan", input_window=late, context={**context, "window": late})

    def answer(*states):
        return json.dumps(dict(schema_version=1, phase_index=0, states=list(states)))

    ok = dict(at_ms=window.start_ms, position="side_control", top_player_id="b", evidence="b on top")
    assert parse_response(call, answer(ok)).states[0].top_player_id == "b"
    for bad, message in (
        (dict(ok, top_player_id="referee"), "top fighter"),
        (dict(ok, top_player_id=None), "top fighter"),
        (dict(ok, position="scramble"), "neutral position"),
        (dict(ok, at_ms=core.start_ms + 1), "from the start"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_response(call, answer(bad))


class Provider:
    name = "fixture"

    def __init__(self, clock):
        self.clock, self.calls = clock, []

    async def generate(self, run, request, execution_id, call_id):
        self.calls.append(request.node_id)
        context = json.loads(request.initial_prompt)["stage_context"]
        if request.node_id == "position_scan":
            data = SCAN
        elif request.node_id == "ground_rescan":
            data = dict(phase_index=context["phase_index"], states=[dict(
                at_ms=context["window"]["start_ms"], position="side_control", top_player_id="a",
                evidence="a on top")])
        elif request.node_id == "position_critique":
            data = dict(transition_index=context["transition_index"], confirmed=True,
                        corrected_at_ms=context["proposed"]["at_ms"] + 3000,
                        state_after=context["proposed"]["state_after"], reason="looks later")
        elif request.node_id == "technique_scan":
            data = dict(events=[])
        elif request.node_id == "technique_analyze":
            from conftest import technique

            bounds = context["bounds"]
            data = technique(bounds=bounds, action_classes=["transition"], strike_target=None, significant=None,
                             knockdown=False, outcome_bounds=dict(start_ms=bounds["end_ms"] - 100, end_ms=bounds["end_ms"]))
        elif request.node_id == "strike_census":
            data = dict(window_index=context["window_index"], strikes=[])
        else:
            raise AssertionError(request.node_id)
        data = {k: v for k, v in data.items() if k != "schema_version"}
        usage = CallUsageV1(run_id=run.spec.run_id, execution_id=execution_id, call_id=call_id, recorded_at=self.clock(),
                            kind="model", provider=self.name, model=request.config.model, usage_status="complete",
                            input_tokens=1, output_tokens=1, cost_status="unknown")
        return ProviderReplyV1(text=json.dumps(dict(schema_version=1, **data)), usage=usage)


def test_full_dag_publishes_the_re_scanned_top_fighter(run, tmp_path):
    data = ground_config(position_critique=None, technique_critique=None, technique_validate=None,
                         strike_audit=None, knockdown_check=None)
    ground_run = strikes_run(run, data, MMA_GROUND_WORKFLOW_ID)
    at = [ground_run.registered_at]

    def clock():
        at[0] += timedelta(milliseconds=1)
        return at[0]

    async def guard():
        return None

    provider, journal = Provider(clock), DiskJournal(tmp_path / "j.sqlite")
    result = asyncio.run(execute_pipeline(ground_run, data, journal=journal, provider=provider, sink=Sink(),
                                          guard=guard, clock=clock, max_parallel_calls=4))
    assert provider.calls.count("ground_rescan") == 1
    positions = result.positions.positions
    side = [p for p in positions if p.position == "side_control"]
    assert side and all(p.roles == ("pinning_player", "pinned_player") for p in side)
    assert positions[-1].position == "striking_range"
    journal.close()


def test_position_checks_skip_transitions_the_ground_re_scan_already_checked(run, tmp_path):
    """A critique retime near a re-scanned phase could cross its high-fps boundaries."""
    data = ground_config(technique_critique=None, technique_validate=None, strike_audit=None,
                         knockdown_check=None)  # position_critique stays configured
    ground_run = strikes_run(run, data, MMA_GROUND_WORKFLOW_ID)
    at = [ground_run.registered_at]

    def clock():
        at[0] += timedelta(milliseconds=1)
        return at[0]

    async def guard():
        return None

    provider, journal = Provider(clock), DiskJournal(tmp_path / "j.sqlite")
    result = asyncio.run(execute_pipeline(ground_run, data, journal=journal, provider=provider, sink=Sink(),
                                          guard=guard, clock=clock, max_parallel_calls=4))
    # Every transition of this short scope is inside the phase or within a critique's reach.
    assert "position_critique" not in provider.calls
    assert any(p.position == "side_control" for p in result.positions.positions)
    journal.close()


def test_normalization_accepts_exactly_the_declared_unchecked_transitions(run, completed_at):
    from bjj_analysis_core.normalization import normalize_positions

    ground_run, _ = ground(run)
    scan = MMA_GROUND.position_scan.model_validate(SCAN)
    critique = dict(schema_version=1, transition_index=2, confirmed=True, corrected_at_ms=7200,
                    state_after=SCAN["transitions"][2]["state_after"], reason="stands up")
    revision = normalize_positions(ground_run, scan, [critique], completed_at=completed_at, unchecked=(0, 1))
    assert [p.bounds.start_ms for p in revision.positions] == [1000, 3000, 6000, 7200]
    for bad in ((0,), (0, 1, 2)):
        with pytest.raises(ValueError, match="exactly one critique"):
            normalize_positions(ground_run, scan, [critique], completed_at=completed_at, unchecked=bad)
