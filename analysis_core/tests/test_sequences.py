"""Workflow mma-sequences-ground: tiling, request/response rules, merge and the full DAG."""

import asyncio
import json
from datetime import timedelta

import pytest
from bjj_analysis_core import MMA_GROUND_WORKFLOW_ID, MMA_SEQUENCES_WORKFLOW_ID
from bjj_analysis_core.execution import ProviderReplyV1
from bjj_analysis_core.executor import execute_pipeline
from bjj_analysis_core.mma_outputs import SequenceCensusV1
from bjj_analysis_core.normalization import normalize_sequences
from bjj_analysis_core.packs import MMA_SEQUENCES
from bjj_analysis_core.requests import (
    CoreConfigV1,
    load_config,
    parse_response,
    prepare_call,
    prompt_bundle_sha256,
)
from bjj_analysis_core.sequences import merge_sequences, sequence_windows
from journal_fixture import DiskJournal
from shared_lib.models.analysis_contracts import CallUsageV1, TimeRangeMs
from test_grounds import SCAN, ground_config
from test_windowed_strikes import NODE, Sink, _positions, strikes_run


def sequence_config(**changes):
    fields = dict(
        prompt_bundle_sha256=prompt_bundle_sha256(MMA_SEQUENCES),
        position_scan=NODE.model_copy(update={"fps": 1}),
        position_critique=None,
        technique_scan=None,
        technique_critique=None,
        technique_analyze=None,
        technique_validate=None,
        candidates=None,
        ground_rescan=NODE,
        ground_window_ms=5000,
        ground_lookback_ms=2000,
        sequence_census=NODE,
        sequence_window_ms=5000,
        sequence_overlap_ms=2000,
    )
    fields.update(changes)
    return CoreConfigV1(**fields).model_dump_json().encode()


def sequences_run(run, data=None):
    data = data or sequence_config()
    return strikes_run(run, data, MMA_SEQUENCES_WORKFLOW_ID), data


def step(action="strike_punch", technique="jab", outcome="successful", **changes):
    strike = action.startswith("strike_") or action == "ground_strike"
    value = dict(
        evidence="seen on the main angle", action=action, technique=technique, note=None, outcome=outcome,
        strike_target="head" if strike else None, significant=True if strike else None, knockdown=False,
        repeat=1, landed=(int(outcome == "successful") if action == "ground_strike" else None),
    )
    value.update(changes)
    return value


def seq(start, end, *steps, actor="a", summary="a chain"):
    return dict(summary=summary, actor_appearance=f"{actor} shorts", actor_player_id=actor, actor_sentinel=None,
                start_ms=start, end_ms=end, steps=list(steps or (step(),)))


def census(index, *sequences):
    return SequenceCensusV1(window_index=index, sequences=list(sequences))


def test_scope_is_tiled_into_windows_with_the_sequence_overlap(run):
    run, data = sequences_run(run)
    _, config = load_config(run, data)
    assert [(w.start_ms, w.end_ms) for w in sequence_windows(run, config)] == [
        (1000, 6000), (4000, 9000), (7000, 11000),
    ]


def test_the_sequence_workflow_has_a_census_and_no_technique_chain(run):
    seq_run, data = sequences_run(run)
    load_config(seq_run, data)
    without = sequence_config(sequence_census=None)
    with pytest.raises(ValueError, match="needs a sequence_census"):
        load_config(sequences_run(run, without)[0], without)
    with_scan = sequence_config(technique_scan=NODE)
    with pytest.raises(ValueError, match="no technique scan"):
        load_config(sequences_run(run, with_scan)[0], with_scan)
    ground = json.loads(ground_config())
    ground.update(sequence_census=NODE.model_dump())
    ground = CoreConfigV1.model_validate(ground).model_dump_json().encode()
    with pytest.raises(ValueError, match="only to the sequence workflow"):
        load_config(strikes_run(run, ground, MMA_GROUND_WORKFLOW_ID), ground)
    chainless = json.loads(ground_config())
    chainless.update(technique_scan=None)
    chainless = CoreConfigV1.model_validate(chainless).model_dump_json().encode()
    with pytest.raises(ValueError, match="needs its technique scan"):
        load_config(strikes_run(run, chainless, MMA_GROUND_WORKFLOW_ID), chainless)


def test_the_sequence_prompt_carries_the_technique_names(run):
    seq_run, data = sequences_run(run)
    window = TimeRangeMs(start_ms=1000, end_ms=6000)
    call = prepare_call(seq_run, data, "sequence_census", input_window=window,
                        context=dict(window_index=0, window=window))
    names = json.loads(call.initial_prompt)["technique_names"]
    assert names["takedown_attempt"][0] == "double_leg" and "sweep" not in names
    assert "takedown_attempt" in call.system_prompt
    other = TimeRangeMs(start_ms=1500, end_ms=6000)
    with pytest.raises(ValueError, match="admitted tiling"):
        prepare_call(seq_run, data, "sequence_census", input_window=other,
                     context=dict(window_index=0, window=other))


def test_census_answers_are_checked_inside_the_window_and_by_the_step_rules(run):
    seq_run, data = sequences_run(run)
    window = TimeRangeMs(start_ms=1000, end_ms=6000)
    call = prepare_call(seq_run, data, "sequence_census", input_window=window,
                        context=dict(window_index=0, window=window))

    def answer(*sequences, index=0):
        return json.dumps(dict(schema_version=1, window_index=index, sequences=list(sequences)))

    good = seq(1200, 5000, step(), step(technique="cross"), step("takedown_attempt", "double_leg"),
               step("ground_strike", "punch", repeat=5, landed=3))
    parsed = parse_response(call, answer(good)).sequences[0]
    assert [s.technique.value for s in parsed.steps] == ["jab", "cross", "double_leg", "punch"]
    for bad, message in (
        (answer(good, index=1), "different window"),
        (answer(seq(5900, 6200)), "inside the window"),
        (answer(seq(1500, 1200)), "ends after it starts"),
        (answer(seq(1200, 1500, actor="referee")), "player id"),
        (answer(seq(1200, 1500, step(technique="double_leg"))), "technique names"),
        (answer(seq(1200, 1500, step("takedown_attempt", "jab"))), "technique names"),
        (answer(seq(1200, 1500, step(technique="other"))), "needs a note"),
        (answer(seq(1200, 1500, step("ground_strike", "punch", repeat=2, landed=3))), "at most repeat"),
        (answer(seq(1200, 1500, step("takedown_attempt", "trip", strike_target="leg"))), "only to strike steps"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_response(call, bad)


def test_a_sequence_seen_in_two_overlapping_windows_is_kept_once(run):
    seq_run, data = sequences_run(run)
    _, config = load_config(seq_run, data)
    censuses = [
        # Window 0 (1000-6000) is cut off at its end; window 1 (4000-9000) sees the whole chain.
        census(0, seq(4800, 6000, step(), step("takedown_attempt", "double_leg", outcome="unclear"))),
        census(1, seq(4900, 7500, step(), step("takedown_attempt", "double_leg"), step("ground_strike", "punch", repeat=4, landed=2)),
               seq(5000, 5400, step(technique="cross"), actor="b")),  # the other fighter is not a copy
        census(2),
    ]
    batch = merge_sequences(seq_run, config, censuses, _positions(seq_run, (9500, 10000)))
    assert (batch.census_sequences, batch.merged_copies) == (3, 1)
    joined = next(q for q in batch.sequences if q.actor_player_id == "a")
    assert [s.technique for s in joined.steps] == ["jab", "double_leg", "punch"]  # window 1's copy saw more
    assert (joined.bounds.start_ms, joined.bounds.end_ms) == (4900, 7500)
    assert [q.actor_player_id for q in batch.sequences] == ["a", "b"]


def test_one_answer_listing_a_sequence_twice_keeps_it_once(run):
    seq_run, data = sequences_run(run)
    _, config = load_config(seq_run, data)
    censuses = [census(0, seq(2000, 3000, step()), seq(2100, 3200, step(), step(technique="cross"))), census(1), census(2)]
    batch = merge_sequences(seq_run, config, censuses, _positions(seq_run, (9500, 10000)))
    assert batch.merged_copies == 1 and len(batch.sequences) == 1
    apart = [census(0, seq(2000, 2500), seq(3500, 4000)), census(1), census(2)]
    assert len(merge_sequences(seq_run, config, apart, _positions(seq_run, (9500, 10000))).sequences) == 2


def test_merge_rejects_sequences_in_footage_that_is_not_live(run):
    seq_run, data = sequences_run(run)
    _, config = load_config(seq_run, data)
    censuses = [census(0, seq(2000, 2300), seq(2500, 2900, actor="b")), census(1), census(2)]
    batch = merge_sequences(seq_run, config, censuses, _positions(seq_run, (2400, 3000)))
    assert batch.rejected_not_live == 1 and [q.actor_player_id for q in batch.sequences] == ["a"]
    with pytest.raises(ValueError, match="every sequence window"):
        merge_sequences(seq_run, config, censuses[:2], _positions(seq_run, (2200, 3000)))


def test_published_sequences_list_untimed_techniques_with_derived_classes_and_outcome(run, completed_at):
    seq_run, data = sequences_run(run)
    _, config = load_config(seq_run, data)
    censuses = [
        census(0, seq(1200, 3500, step(outcome="unsuccessful"), step(technique="cross"),
                      step("takedown_attempt", "double_leg", outcome="unsuccessful"),
                      summary="jab, cross, then a stuffed double leg")),
        census(1, seq(8000, 8400, step(outcome="unsuccessful"), actor="b")),
        census(2),
    ]
    batch = merge_sequences(seq_run, config, censuses, _positions(seq_run, (10000, 10500)))
    first, second = normalize_sequences(seq_run, batch, completed_at=completed_at).techniques
    assert [a.value for a in first.action_classes] == ["strike_punch", "takedown_attempt"]
    assert first.outcome.value == "successful" and first.outcome_bounds == first.bounds
    assert (first.bounds.start_ms, first.bounds.end_ms) == (1200, 3500) and "bounds" not in first.steps[0].model_dump()
    assert first.notes == "jab, cross, then a stuffed double leg"
    assert second.outcome.value == "unsuccessful" and second.actor_player_id == "b"


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
                evidence="a on top", geometry=None)])
        elif request.node_id == "sequence_census":
            index, start = context["window_index"], context["window"]["start_ms"]
            sequences = [seq(start + 200, start + 800, step(), step(technique="cross"))] if index == 0 else []
            data = dict(window_index=index, sequences=sequences)
        else:
            raise AssertionError(request.node_id)
        data = {k: v for k, v in data.items() if k != "schema_version"}
        usage = CallUsageV1(run_id=run.spec.run_id, execution_id=execution_id, call_id=call_id, recorded_at=self.clock(),
                            kind="model", provider=self.name, model=request.config.model, usage_status="complete",
                            input_tokens=1, output_tokens=1, cost_status="unknown")
        return ProviderReplyV1(text=json.dumps(dict(schema_version=1, **data)), usage=usage)


@pytest.mark.parametrize("parallel", [1, 6])
def test_full_dag_publishes_sequences_without_technique_calls(run, tmp_path, parallel):
    seq_run, data = sequences_run(run)
    at = [seq_run.registered_at]

    def clock():
        at[0] += timedelta(milliseconds=1)
        return at[0]

    async def guard():
        return None

    provider, journal, sink = Provider(clock), DiskJournal(tmp_path / "j.sqlite"), Sink()
    result = asyncio.run(execute_pipeline(seq_run, data, journal=journal, provider=provider, sink=sink,
                                          guard=guard, clock=clock, max_parallel_calls=parallel))
    assert sorted(set(provider.calls)) == ["ground_rescan", "position_scan", "sequence_census"]
    assert provider.calls.count("sequence_census") == 3
    (event,) = result.techniques.techniques
    assert [s.technique for s in event.steps] == ["jab", "cross"]
    assert set(sink.outputs) == {"positions", "techniques"}
    journal.close()
