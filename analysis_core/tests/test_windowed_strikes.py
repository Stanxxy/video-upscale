"""Workflow mma-windowed-strikes: tiling, request/response rules, merge and the full DAG."""

import asyncio
import hashlib
import json
from datetime import timedelta

import pytest
from bjj_analysis_core import MMA_STRIKES_WORKFLOW_ID, MMA_WORKFLOW_ID
from bjj_analysis_core.execution import ProviderReplyV1
from bjj_analysis_core.executor import execute_pipeline
from bjj_analysis_core.mma_outputs import StrikeAuditV1, StrikeCensusV1
from bjj_analysis_core.packs import MMA, MMA_STRIKES
from bjj_analysis_core.requests import (
    CoreConfigV1,
    NodeConfigV1,
    load_config,
    parse_response,
    prepare_call,
    prompt_bundle_sha256,
)
from bjj_analysis_core.strikes import merge_strikes, strike_windows
from journal_fixture import DiskJournal
from shared_lib.models.analysis_contracts import CallUsageV1, TimeRangeMs
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

NODE = NodeConfigV1(model="fixture-model", media_resolution="low", fps=10)


def config_bytes(pack=MMA_STRIKES, *, audit=True, **changes):
    fields = dict(
        prompt_bundle_sha256=prompt_bundle_sha256(pack),
        position_scan=NODE.model_copy(update={"fps": 1}),
        position_critique=NODE,
        technique_scan=NODE.model_copy(update={"fps": 1}),
        technique_critique=NODE,
        technique_analyze=NODE,
        technique_validate=NODE,
        candidates=dict(),
        strike_census=NODE if pack is MMA_STRIKES else None,
        strike_audit=NODE if (pack is MMA_STRIKES and audit) else None,
        strike_window_ms=5000,
        strike_overlap_ms=1000,
    )
    fields.update(changes)
    return CoreConfigV1(**fields).model_dump_json().encode()


def strikes_run(run, data, workflow_id=MMA_STRIKES_WORKFLOW_ID):
    spec = run.spec.model_copy(
        update={
            "workflow_id": workflow_id,
            "taxonomy_id": "mma.v1",
            "effective_config": run.spec.effective_config.model_copy(
                update={"sha256": hashlib.sha256(data).hexdigest()}
            ),
        }
    )
    return RegisteredAIRunV1.model_validate({**run.model_dump(), "spec": spec.model_dump()})


def strike(at, actor="b", action="strike_punch", **changes):
    value = dict(at_ms=at, action=action, actor_player_id=actor, actor_sentinel=None, strike_target="head",
                 outcome="successful", significant=True, knockdown=False, evidence="glove meets the jaw")
    value.update(changes)
    return value


def test_scope_is_tiled_into_overlapping_windows_ending_at_scope_end(run):
    data = config_bytes()
    run = strikes_run(run, data)
    _, config = load_config(run, data)
    assert [(w.start_ms, w.end_ms) for w in strike_windows(run, config)] == [(1000, 6000), (5000, 10000), (9000, 11000)]


def test_census_nodes_belong_only_to_the_windowed_workflow(run):
    mma = config_bytes(MMA, strike_census=NODE)
    with pytest.raises(ValueError, match="only to the windowed"):
        load_config(strikes_run(run, mma, MMA_WORKFLOW_ID), mma)
    audit_off = config_bytes(audit=False)
    assert load_config(strikes_run(run, audit_off), audit_off)[1].strike_audit is None
    no_census = CoreConfigV1.model_validate_json(config_bytes(audit=False)).model_copy(update={"strike_census": None})
    raw = no_census.model_dump_json().encode()
    with pytest.raises(ValueError, match="needs a strike_census"):
        load_config(strikes_run(run, raw), raw)


def test_census_call_must_use_its_admitted_window_and_answers_inside_it(run):
    data = config_bytes()
    run = strikes_run(run, data)
    window = TimeRangeMs(start_ms=5000, end_ms=10000)
    with pytest.raises(ValueError, match="admitted tiling"):
        prepare_call(run, data, "strike_census", input_window=dict(start_ms=5000, end_ms=9000),
                     context=dict(window_index=1, window=dict(start_ms=5000, end_ms=9000)))
    call = prepare_call(run, data, "strike_census", input_window=window, context=dict(window_index=1, window=window))
    ok = dict(schema_version=1, window_index=1, strikes=[strike(5200), strike(5900, actor="a")])
    assert len(parse_response(call, json.dumps(ok)).strikes) == 2
    assert parse_response(call, json.dumps(dict(ok, strikes=[strike(10000)])))  # last visible frame
    for bad, message in ((dict(ok, strikes=[strike(10001)]), "inside the window"),
                         (dict(ok, strikes=[strike(5900), strike(5200)]), "inside the window"),
                         (dict(ok, window_index=0), "different window")):
        with pytest.raises(ValueError, match=message):
            parse_response(call, json.dumps(bad))


def test_audit_decides_every_census_strike_once_in_order(run):
    data = config_bytes()
    run = strikes_run(run, data)
    window = TimeRangeMs(start_ms=1000, end_ms=6000)
    census = dict(schema_version=1, window_index=0, strikes=[strike(2000), strike(3000)])
    call = prepare_call(run, data, "strike_audit", input_window=window, context=dict(window_index=0, census=census))
    keep = dict(census_index=0, verdict="keep", at_ms=None, reason="clear jab")
    retime = dict(census_index=1, verdict="retime", at_ms=3400, reason="contact later")
    assert parse_response(call, json.dumps(dict(schema_version=1, window_index=0, decisions=[keep, retime], added=[])))
    with pytest.raises(ValueError, match="every census strike once"):
        parse_response(call, json.dumps(dict(schema_version=1, window_index=0, decisions=[keep], added=[])))
    with pytest.raises(ValueError, match="outside the window"):
        parse_response(call, json.dumps(dict(schema_version=1, window_index=0, decisions=[keep, dict(retime, at_ms=7000)], added=[])))
    echo = dict(keep, at_ms=2000)
    assert parse_response(call, json.dumps(dict(schema_version=1, window_index=0, decisions=[echo, retime], added=[])))
    with pytest.raises(ValueError, match="keeps its census time"):
        parse_response(call, json.dumps(dict(schema_version=1, window_index=0, decisions=[dict(keep, at_ms=2100), retime], added=[])))


def _positions(run, inactive):
    from bjj_analysis_core.normalization import normalize_positions  # noqa: F401  (shape reference)

    class P:
        def __init__(self, s, e, obs):
            self.bounds, self.observation = TimeRangeMs(start_ms=s, end_ms=e), obs

    class R:
        positions = (P(1000, inactive[0], "observed"), P(*inactive, "inactive"), P(inactive[1], 11000, "observed"))

    return R()


def test_merge_applies_audit_rejects_not_live_and_merges_overlap_twins_only(run):
    data = config_bytes()
    run = strikes_run(run, data)
    _, config = load_config(run, data)
    censuses = [
        StrikeCensusV1(window_index=0, strikes=[strike(2000), strike(2300), strike(5500), strike(5800, action="strike_knee")]),
        StrikeCensusV1(window_index=1, strikes=[strike(5700), strike(8500)]),
        StrikeCensusV1(window_index=2, strikes=[]),
    ]
    audits = [
        StrikeAuditV1(window_index=0, decisions=[dict(census_index=0, verdict="keep", at_ms=None, reason="r"),
                                                 dict(census_index=1, verdict="keep", at_ms=None, reason="r"),
                                                 dict(census_index=2, verdict="keep", at_ms=None, reason="r"),
                                                 dict(census_index=3, verdict="drop", at_ms=None, reason="feint")],
                      added=[strike(4000, actor="a", outcome="unclear"), strike(2100)]),
        StrikeAuditV1(window_index=1, decisions=[dict(census_index=0, verdict="keep", at_ms=None, reason="r"),
                                                 dict(census_index=1, verdict="retime", at_ms=8600, reason="r")], added=[]),
        StrikeAuditV1(window_index=2, decisions=[], added=[]),
    ]
    batch = merge_strikes(run, config, censuses, audits, _positions(run, (8000, 9000)))
    # 2000 and 2300 stay (same window: a fast combination); 5500/5700 are one strike seen
    # from two windows; the knee was dropped; 8600 sits in the replay span; 4000 was added.
    assert [s.bounds.start_ms + 250 for s in batch.strikes] == [2000, 2300, 4000, 5500]
    # The audit's 2100 re-adds the census 2000 punch: one strike, not a combination.
    assert (batch.dropped_by_audit, batch.added_by_audit, batch.rejected_not_live, batch.merged_duplicates) == (1, 2, 1, 2)
    unclear = batch.strikes[2]
    assert unclear.outcome == "unclear" and unclear.outcome_bounds is None and unclear.actor_player_id == "a"
    assert batch.strikes[0].outcome_bounds == TimeRangeMs(start_ms=1950, end_ms=2050)
    without_audit = merge_strikes(run, config, censuses, None, _positions(run, (8000, 9000)))
    # Unaudited: the knee stays, 8500 is in the replay, 5700 is 5500 seen again.
    assert len(without_audit.strikes) == 4 and without_audit.dropped_by_audit == 0


class Provider:
    name = "fixture"

    def __init__(self, clock):
        self.clock, self.calls = clock, []

    async def generate(self, run, request, execution_id, call_id):
        self.calls.append(request.node_id)
        context = json.loads(request.initial_prompt)["stage_context"]
        live = dict(observation="observed", position="striking_range", roles=["neutral", "neutral"], notes="",
                    evidence="both standing", uncertainty_note=None, interruption_note=None, interruption_kind=None,
                    cage_contact=None)
        replay = dict(live, observation="inactive", position=None, roles=["not_applicable", "not_applicable"],
                      evidence="slow-motion replay graphic", interruption_note="broadcast replay", interruption_kind="replay")
        if request.node_id == "position_scan":
            data = dict(initial_state=live, transitions=[dict(at_ms=8000, state_after=replay)])
        elif request.node_id == "position_critique":
            data = dict(transition_index=0, confirmed=True, corrected_at_ms=8000, state_after=replay, reason="replay starts")
        elif request.node_id == "technique_scan":
            data = dict(events=[])
        elif request.node_id == "technique_critique":
            data = dict(candidate_id=context["candidate"]["candidate_id"], confirmed=False, bounds=None, end_status=None,
                        reason="no grappling attack at the cut")
        elif request.node_id == "strike_census":
            start = context["window"]["start_ms"]
            data = dict(window_index=context["window_index"], strikes=[strike(start + 600), strike(start + 1500, actor="a")])
        else:
            data = dict(window_index=context["window_index"], added=[],
                        decisions=[dict(census_index=i, verdict="keep", at_ms=None, reason="visible")
                                   for i, _ in enumerate(context["census"]["strikes"])])
        usage = CallUsageV1(run_id=run.spec.run_id, execution_id=execution_id, call_id=call_id, recorded_at=self.clock(),
                            kind="model", provider=self.name, model=request.config.model, usage_status="complete",
                            input_tokens=1, output_tokens=1, cost_status="unknown")
        return ProviderReplyV1(text=json.dumps(dict(schema_version=1, **data)), usage=usage)


class Sink:
    def __init__(self):
        self.outputs = {}

    async def publish(self, run, component, revision):
        self.outputs[component] = revision


@pytest.mark.parametrize("parallel", [1, 6])
def test_full_dag_publishes_census_strikes_as_mma_techniques_without_per_strike_calls(run, tmp_path, parallel):
    data = config_bytes()
    run = strikes_run(run, data)
    at = [run.registered_at]

    def clock():
        at[0] += timedelta(milliseconds=1)
        return at[0]

    provider, sink = Provider(clock), Sink()

    async def guard():
        return None

    journal = DiskJournal(tmp_path / "j.sqlite")
    result = asyncio.run(execute_pipeline(run, data, journal=journal, provider=provider, sink=sink, guard=guard,
                                          clock=clock, max_parallel_calls=parallel))
    calls = sorted(provider.calls)
    assert calls.count("strike_census") == 3 and calls.count("strike_audit") == 3
    assert "technique_analyze" not in calls and "technique_validate" not in calls
    events = result.techniques.techniques
    # Windows start 1000/5000/9000; the replay runs 8000 to scope end, so 9600 and 10500 are rejected.
    assert [(e.bounds.start_ms + 250, e.actor_player_id) for e in events] == [(1600, "b"), (2500, "a"), (5600, "b"), (6500, "a")]
    assert all(e.taxonomy_id == "mma.v1" and e.strike_target is not None and e.origin == "ai" for e in events)
    assert sink.outputs["techniques"] == result.techniques
    # Resume replays every step from the journal: no provider call.
    again = Provider(clock)
    asyncio.run(execute_pipeline(run, data, journal=journal, provider=again, sink=Sink(), guard=guard, clock=clock,
                                 max_parallel_calls=parallel))
    assert again.calls == []
    journal.close()


def test_chain_strikes_are_not_published_in_the_windowed_workflow(run, completed_at):
    from bjj_analysis_core.normalization import normalize_techniques
    from conftest import technique

    data = config_bytes()
    run = strikes_run(run, data)
    grapple = technique(action_classes=["takedown_attempt"], strike_target=None, significant=None, knockdown=False)
    punch = technique(action_classes=["strike_punch"], strike_target="head", significant=True, knockdown=True,
                      notes="right hand drops him", outcome_evidence="falls", outcome_bounds=dict(start_ms=5500, end_ms=5900))
    candidates, decisions = [], []
    for name, proposal in (("grapple", grapple), ("punch", punch)):
        candidates.append(dict(candidate_id=name, description="seen", observation_window=dict(start_ms=1000, end_ms=11000), proposed=proposal))
        decisions.append(dict(schema_version=1, candidate_id=name, verdict="agree", intentional_technique_visible=True,
                              technique=proposal, reason="visible"))
    revision = normalize_techniques(run, candidates, decisions, completed_at=completed_at)
    assert [e.action_classes[0].value for e in revision.techniques] == ["takedown_attempt"]
