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


def step(start, end=None, action="strike_punch", technique="jab", outcome="successful", **changes):
    strike = action.startswith("strike_") or action == "ground_strike"
    value = dict(
        start_ms=start, end_ms=end or start + 300, action=action, technique=technique, note=None,
        outcome=outcome, strike_target="head" if strike else None, significant=True if strike else None,
        knockdown=False, repeat=1, landed=(int(outcome == "successful") if action == "ground_strike" else None),
        evidence="seen on the main angle",
    )
    value.update(changes)
    return value


def seq(*steps, actor="a", summary="a chain"):
    return dict(summary=summary, actor_appearance=f"{actor} shorts", actor_player_id=actor, actor_sentinel=None,
                steps=list(steps))


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

    good = seq(step(1200), step(1500, technique="cross"), step(2000, 3500, "takedown_attempt", "double_leg"),
               step(3800, 5000, "ground_strike", "punch", repeat=5, landed=3))
    assert len(parse_response(call, answer(good)).sequences[0].steps) == 4
    for bad, message in (
        (answer(good, index=1), "different window"),
        (answer(seq(step(5900, 6200))), "inside the window"),
        (answer(seq(step(1200), actor="referee")), "player id"),
        (answer(seq(step(1200, technique="double_leg"))), "technique names"),
        (answer(seq(step(1200, action="takedown_attempt", technique="jab"))), "technique names"),
        (answer(seq(step(1200, technique="other"))), "needs a note"),
        (answer(seq(step(1200, 2000, "ground_strike", "punch", repeat=2, landed=3))), "at most repeat"),
        (answer(seq(step(1200, 2000, "takedown_attempt", "trip", strike_target="leg"))), "only to strike steps"),
        (answer(seq(step(2000), step(1200))), "ordered by start"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_response(call, bad)


def test_merge_joins_overlap_copies_keeps_the_better_seen_step_and_splits_long_pauses(run):
    seq_run, data = sequences_run(run)
    _, config = load_config(seq_run, data)
    censuses = [
        census(
            0,
            # Cut off by window 0's end (unclear); window 1 sees the takedown land.
            seq(step(4800), step(5100, 6000, "takedown_attempt", "double_leg", outcome="unclear")),
            # A pause of 2 s inside one census sequence: two sequences.
            seq(step(1200), step(3500, technique="hook"), actor="b"),
        ),
        census(
            1,
            seq(step(4800), step(5150, 6500, "takedown_attempt", "double_leg"),
                step(6700, 8000, "ground_strike", "punch", repeat=4, landed=2)),
            # The other fighter at the same moment is not a copy.
            seq(step(4900, technique="cross"), actor="b"),
        ),
        census(2),
    ]
    live = _positions(seq_run, (9500, 10000))
    batch = merge_sequences(seq_run, config, censuses, live)
    assert (batch.census_sequences, batch.merged_copies, batch.splits) == (4, 1, 1)
    chains = [([s.action.value for s in q.steps], q.actor_player_id) for q in batch.sequences]
    assert chains == [
        (["strike_punch"], "b"),
        (["strike_punch"], "b"),
        (["strike_punch"], "b"),
        (["strike_punch", "takedown_attempt", "ground_strike"], "a"),
    ] or chains == [
        (["strike_punch"], "b"),
        (["strike_punch"], "b"),
        (["strike_punch", "takedown_attempt", "ground_strike"], "a"),
        (["strike_punch"], "b"),
    ]
    joined = next(q for q in batch.sequences if q.actor_player_id == "a")
    takedown = joined.steps[1]
    assert takedown.outcome.value == "successful" and takedown.bounds.end_ms == 6500


def test_merge_rejects_steps_in_footage_that_is_not_live(run):
    seq_run, data = sequences_run(run)
    _, config = load_config(seq_run, data)
    censuses = [census(0, seq(step(2000), step(2300, technique="cross"))), census(1), census(2)]
    batch = merge_sequences(seq_run, config, censuses, _positions(seq_run, (2200, 3000)))
    assert batch.rejected_not_live_steps == 1
    assert [[s.technique for s in q.steps] for q in batch.sequences] == [["jab"]]
    with pytest.raises(ValueError, match="every sequence window"):
        merge_sequences(seq_run, config, censuses[:2], _positions(seq_run, (2200, 3000)))


def test_published_sequences_derive_classes_outcome_and_evidence(run, completed_at):
    seq_run, data = sequences_run(run)
    _, config = load_config(seq_run, data)
    censuses = [
        census(0, seq(step(1200, outcome="unsuccessful"), step(1500, technique="cross"),
                      step(2000, 3500, "takedown_attempt", "double_leg", outcome="unsuccessful"),
                      summary="jab, cross, then a stuffed double leg")),
        census(1, seq(step(8000, outcome="unsuccessful"), actor="b")),
        census(2),
    ]
    batch = merge_sequences(seq_run, config, censuses, _positions(seq_run, (10000, 10500)))
    revision = normalize_sequences(seq_run, batch, completed_at=completed_at)
    first, second = revision.techniques
    assert [a.value for a in first.action_classes] == ["strike_punch", "takedown_attempt"]
    assert first.outcome.value == "successful" and first.outcome_bounds == first.steps[1].bounds
    assert (first.bounds.start_ms, first.bounds.end_ms) == (1200, 3500)
    assert first.notes == "jab, cross, then a stuffed double leg"
    assert second.outcome.value == "unsuccessful" and second.actor_player_id == "b"
    assert first.strike_target is None and first.significant is None


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
            sequences = [seq(step(start + 200), step(start + 500, technique="cross"))] if index == 0 else []
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
