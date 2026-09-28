import asyncio
import json
from datetime import timedelta
from uuid import uuid4

import pytest
from bjj_analysis_core.execution import (
    CheckpointConflict,
    ExecutionCancelled,
    NotDispatched,
    ProviderReplyV1,
    RecoveryRequired,
    StepRunner,
)
from bjj_analysis_core.executor import execute_pipeline as run_pipeline
from bjj_analysis_core.reporting import report_execution
from bjj_analysis_core.requests import prepare_call
from conftest import state, technique
from journal_fixture import DiskJournal
from shared_lib.models.analysis_contracts import CallUsageV1, summarize_costs


async def fixture_authority():
    """Explicit trusted local fixture authority; real hosts must reauthorize."""


async def execute_pipeline(*args, guard=fixture_authority, **kwargs):
    return await run_pipeline(*args, guard=guard, **kwargs)


class Clock:
    def __init__(self, run):
        self.at = run.registered_at

    def __call__(self):
        self.at += timedelta(milliseconds=1)
        return self.at


class Sink:
    def __init__(self):
        self.outputs = {}

    async def publish(self, run, component, revision):
        if component in self.outputs:
            assert self.outputs[component] == revision
        self.outputs[component] = revision


class FixtureProvider:
    name = "fixture"

    def __init__(self, clock, sink, *, empty=False):
        self.clock, self.sink, self.empty = clock, sink, empty
        self.calls, self.before, self.failures = [], [], {}

    async def generate(self, run, request, execution_id, call_id):
        self.calls.append((request.node_id, call_id))
        if request.node_id in self.failures:
            failure = self.failures[request.node_id]
            if callable(failure):
                return await failure(run, request, execution_id, call_id)
            raise failure
        context = json.loads(request.initial_prompt)["stage_context"]
        if request.node_id == "position_scan":
            data = dict(
                initial_state=state(),
                transitions=(
                    []
                    if self.empty
                    else [
                        dict(
                            at_ms=4000,
                            state_after=state(
                                "back_control", ("back_controller", "back_defender")
                            ),
                        )
                    ]
                ),
            )
        elif request.node_id == "position_critique":
            self.before.append(context["state_before"])
            data = dict(
                transition_index=context["transition_index"],
                confirmed=True,
                corrected_at_ms=4500,
                state_after=context["proposed"]["state_after"],
                reason="rear control visible",
            )
        elif request.node_id == "technique_scan":
            assert "positions" in self.sink.outputs  # independent publication order
            assert json.loads(request.initial_prompt)["advisory_positions"] is None
            data = dict(
                events=(
                    []
                    if self.empty
                    else [
                        dict(
                            bounds=dict(start_ms=2000, end_ms=5000),
                            description="A applies a rear choke",
                            reasoning="grip and defense visible",
                        )
                    ]
                )
            )
        elif request.node_id == "technique_critique":
            data = dict(
                candidate_id=context["candidate"]["candidate_id"],
                confirmed=True,
                bounds=dict(start_ms=2000, end_ms=8000),
                end_status="outcome_visible",
                reason="tap and release visible after rough end",
            )
        elif request.node_id == "technique_analyze":
            data = technique(
                bounds=context["bounds"],
                outcome_bounds=dict(start_ms=7400, end_ms=7500),
            )
        else:
            data = dict(
                candidate_id=context["candidate_id"],
                verdict="agree",
                intentional_technique_visible=True,
                technique=context["proposed"],
                reason="tap visible",
            )
        return self.reply(run, request, execution_id, call_id, json.dumps(data))

    def reply(self, run, request, execution_id, call_id, text):
        return ProviderReplyV1(
            text=text,
            usage=CallUsageV1(
                run_id=run.spec.run_id,
                execution_id=execution_id,
                call_id=call_id,
                recorded_at=self.clock(),
                kind="model",
                provider=self.name,
                model=request.config.model,
                provider_request_id=str(call_id),
                usage_status="complete",
                input_tokens=10,
                output_tokens=5,
                thinking_tokens=2,
                cost_status="reported",
                cost_nano_usd=12,
                raw_usage=dict(bucket="fixture-private", key="usage", sha256="a" * 64),
            ),
        )


@pytest.fixture
def setup(run, tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    journal, clock, sink = DiskJournal(path), Clock(run), Sink()
    provider = FixtureProvider(clock, sink)
    yield path, journal, clock, sink, provider
    journal.close()


@pytest.mark.asyncio
async def test_full_dag_on_disk_resume_has_no_provider_replay(
    run, configuration, setup
):
    path, journal, clock, sink, provider = setup
    result = await execute_pipeline(
        run, configuration, journal=journal, provider=provider, sink=sink, clock=clock
    )
    assert len(provider.calls) == 6
    assert result.techniques.techniques[0].bounds.end_ms == 8000
    assert set(sink.outputs) == {"positions", "techniques"}
    restarted = DiskJournal(path)
    again = await execute_pipeline(
        run, configuration, journal=restarted, provider=provider, sink=sink, clock=clock
    )
    restarted.close()
    assert again == result and len(provider.calls) == 6
    records = journal.records()
    assert all(r.phase == "succeeded" for r in records)
    summary = summarize_costs(tuple(r.usage for r in records))
    assert summary.known_cost_nano_usd == 72 and summary.unknown_cost_calls == 0
    assert all(r.usage.cost_nano_usd == 0 for r in records if r.request is None)
    history = journal.records(history=True)
    assert sum(r.phase == "responded" for r in history) == 6
    assert all(r.progress.completed_items == 0 for r in history if r.phase == "running")
    assert all(r.progress.completed_items == r.progress.total_items for r in records)


@pytest.mark.asyncio
async def test_empty_fanouts_are_explicit_zero_not_missing(run, configuration, setup):
    _, journal, clock, sink, provider = setup
    provider.empty = True
    result = await execute_pipeline(
        run, configuration, journal=journal, provider=provider, sink=sink, clock=clock
    )
    assert result.techniques.techniques == () and len(provider.calls) == 2
    for node in (
        "position_critique",
        "technique_critique",
        "technique_analyze",
        "technique_validate",
    ):
        record = next(r for r in journal.records() if r.progress.node_id == node)
        assert record.progress.total_items == record.progress.completed_items == 0
        assert record.usage.kind == "deterministic" and record.phase == "succeeded"


@pytest.mark.asyncio
async def test_positions_survive_technique_failure_and_only_failed_call_retries(
    run, configuration, setup
):
    _, journal, clock, sink, provider = setup
    provider.failures["technique_scan"] = NotDispatched("connection never opened")
    with pytest.raises(NotDispatched):
        await execute_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            clock=clock,
        )
    positions = sink.outputs["positions"]
    assert "techniques" not in sink.outputs
    with pytest.raises(RecoveryRequired):
        await execute_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            clock=clock,
        )
    del provider.failures["technique_scan"]
    result = await execute_pipeline(
        run,
        configuration,
        journal=journal,
        provider=provider,
        sink=sink,
        clock=clock,
        retry_failed=True,
    )
    assert result.positions == positions
    assert [n for n, _ in provider.calls].count("position_scan") == 1
    assert [n for n, _ in provider.calls].count("technique_scan") == 2
    scan_records = [
        r
        for r in journal.records(history=True)
        if r.progress.node_id == "technique_scan"
    ]
    assert {r.progress.attempt for r in scan_records} == {1, 2}
    assert any(
        r.phase == "failed" and r.usage.cost_status == "unknown" for r in scan_records
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [TimeoutError("unknown dispatch"), asyncio.CancelledError()]
)
async def test_uncertain_or_cancelled_provider_is_never_silently_retried(
    run, configuration, setup, failure
):
    _, journal, clock, sink, provider = setup
    provider.failures["position_scan"] = failure
    with pytest.raises(type(failure)):
        await execute_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            clock=clock,
        )
    record = journal.records()[0]
    assert record.usage.cost_nano_usd is None and not record.retryable
    with pytest.raises(RecoveryRequired):
        await execute_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            clock=clock,
            retry_failed=True,
        )
    assert len(provider.calls) == 1 and not sink.outputs


@pytest.mark.asyncio
async def test_malformed_reply_retains_paid_usage_and_attempt_budget(
    run, configuration, setup
):
    _, journal, clock, sink, provider = setup

    async def malformed(run, req, eid, cid):
        return provider.reply(run, req, eid, cid, "not json")

    provider.failures["position_scan"] = malformed
    for retry in (False, True):
        with pytest.raises(ValueError):
            await execute_pipeline(
                run,
                configuration,
                journal=journal,
                provider=provider,
                sink=sink,
                clock=clock,
                retry_failed=retry,
            )
    with pytest.raises(RecoveryRequired):
        await execute_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            clock=clock,
            retry_failed=True,
        )
    failed = [r for r in journal.records(history=True) if r.phase == "failed"]
    assert len(provider.calls) == len(failed) == 2
    assert all(r.reply.text == "not json" for r in failed)
    assert summarize_costs(tuple(r.usage for r in failed)).known_cost_nano_usd == 24
    report = report_execution(journal.records(history=True))
    assert report.total_costs.known_cost_nano_usd == 24
    assert report.node_costs[0].node_id == "position_scan"
    assert report.node_costs[0].costs.known_cost_nano_usd == 24
    with pytest.raises(ValueError, match="missing"):
        report_execution(journal.records())  # latest heads must not erase retries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase,after",
    [
        ("running", False),
        ("running", True),
        ("responded", False),
        ("responded", True),
        ("succeeded", True),
    ],
)
async def test_checkpoint_commit_faults_never_blindly_rebill(
    run, configuration, setup, phase, after
):
    path, journal, clock, sink, provider = setup

    class Fault:
        fired = False

        async def read(self, key):
            return await journal.read(key)

        async def compare_and_swap(self, expected, record):
            if not self.fired and record.phase == phase:
                self.fired = True
                if after:
                    await journal.compare_and_swap(expected, record)
                raise OSError("checkpoint commit response lost")
            return await journal.compare_and_swap(expected, record)

    request = prepare_call(
        run, configuration, "position_scan", input_window=run.spec.scope, context={}
    )
    with pytest.raises(OSError):
        await StepRunner(run, configuration, Fault(), provider, clock=clock).model(
            request
        )
    restarted = DiskJournal(path)
    runner = StepRunner(
        run, configuration, restarted, provider, clock=clock, retry_failed=True
    )
    ambiguous = (phase == "running" and after) or (phase == "responded" and not after)
    if ambiguous:
        with pytest.raises(RecoveryRequired):
            await runner.model(request)
    else:
        assert (await runner.model(request)).initial_state.position == "standing"
    restarted.close()
    assert len(provider.calls) == (0 if phase == "running" and after else 1)


@pytest.mark.asyncio
async def test_competing_workers_reserve_one_provider_call(run, configuration, setup):
    _, journal, clock, sink, provider = setup
    request = prepare_call(
        run, configuration, "position_scan", input_window=run.spec.scope, context={}
    )
    original = provider.generate
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        entered.set()
        await release.wait()
        return await original(*args)

    provider.generate = blocked
    one = asyncio.create_task(
        StepRunner(run, configuration, journal, provider, clock=clock).model(request)
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    with pytest.raises(RecoveryRequired):
        await StepRunner(run, configuration, journal, provider, clock=clock).model(
            request
        )
    release.set()
    await one
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    ["config_sha256", "prompt_bundle_sha256", "system_prompt", "model", "scope"],
)
async def test_unadmitted_prepared_call_rejects_before_reservation(
    run, configuration, setup, field
):
    _, journal, clock, sink, provider = setup
    request = prepare_call(
        run, configuration, "position_scan", input_window=run.spec.scope, context={}
    )
    if field == "model":
        request = request.model_copy(
            update={
                "config": request.config.model_copy(update={"model": "foreign-model"})
            }
        )
    elif field == "scope":
        request = request.model_copy(
            update={"input_window": dict(start_ms=0, end_ms=20000)}
        )
    else:
        request = request.model_copy(update={field: "b" * 64})
    with pytest.raises(RecoveryRequired):
        await StepRunner(run, configuration, journal, provider, clock=clock).model(
            request
        )
    assert not journal.records() and not provider.calls


@pytest.mark.asyncio
async def test_explicit_authority_is_required_and_rechecked_on_cached_replay(
    run, configuration, setup
):
    _, journal, clock, sink, provider = setup
    with pytest.raises(ValueError, match="guard"):
        await run_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            guard=None,
            clock=clock,
        )
    await execute_pipeline(
        run, configuration, journal=journal, provider=provider, sink=sink, clock=clock
    )
    count = len(provider.calls)

    async def revoked():
        raise ExecutionCancelled("revoked after prior completion")

    with pytest.raises(ExecutionCancelled):
        await execute_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            clock=clock,
            guard=revoked,
        )
    assert len(provider.calls) == count


@pytest.mark.asyncio
async def test_revoked_during_call_retains_usage_but_never_publishes(
    run, configuration, setup
):
    _, journal, clock, sink, provider = setup
    allowed = True

    async def guard():
        if not allowed:
            raise ExecutionCancelled("revoked while provider running")

    original = provider.generate

    async def revoke(*args):
        nonlocal allowed
        reply = await original(*args)
        allowed = False
        return reply

    provider.generate = revoke
    with pytest.raises(ExecutionCancelled):
        await execute_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            clock=clock,
            guard=guard,
        )
    report = report_execution(journal.records(history=True))
    assert report.total_costs.known_cost_nano_usd == 12
    assert len(provider.calls) == 1 and not sink.outputs


@pytest.mark.asyncio
async def test_bad_usage_is_retained_but_cannot_poison_accounting(
    run, configuration, setup
):
    _, journal, clock, sink, provider = setup

    async def wrong(run, req, eid, cid):
        return provider.reply(run, req, eid, uuid4(), '{"initial_state":null}')

    provider.failures["position_scan"] = wrong
    with pytest.raises(RecoveryRequired, match="usage"):
        await execute_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            clock=clock,
        )
    record = journal.records()[0]
    assert record.reply.usage.call_id != record.usage.call_id
    assert record.usage.cost_status == "unknown" and not record.retryable


@pytest.mark.asyncio
async def test_current_authority_guard_stops_before_cached_or_new_work(
    run, configuration, setup
):
    _, journal, clock, sink, provider = setup

    async def revoked():
        raise ExecutionCancelled("workspace revoked")

    with pytest.raises(ExecutionCancelled):
        await execute_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            clock=clock,
            guard=revoked,
        )
    assert not journal.records() and not provider.calls


@pytest.mark.asyncio
async def test_sink_recovery_reuses_results_without_model_calls(
    run, configuration, setup
):
    _, journal, clock, sink, provider = setup
    original = sink.publish

    async def interrupted(run, component, revision):
        await original(run, component, revision)
        raise OSError("ack lost after durable publication")

    sink.publish = interrupted
    with pytest.raises(OSError):
        await execute_pipeline(
            run,
            configuration,
            journal=journal,
            provider=provider,
            sink=sink,
            clock=clock,
        )
    assert len(provider.calls) == 2 and "positions" in sink.outputs
    sink.publish = original
    await execute_pipeline(
        run,
        configuration,
        journal=journal,
        provider=provider,
        sink=sink,
        clock=clock,
        retry_failed=True,
    )
    assert len(provider.calls) == 6


@pytest.mark.asyncio
async def test_changed_input_and_corrupt_journal_fail_closed(run, configuration, setup):
    _, journal, clock, sink, provider = setup
    request = prepare_call(
        run, configuration, "position_scan", input_window=run.spec.scope, context={}
    )
    runner = StepRunner(run, configuration, journal, provider, clock=clock)
    await runner.model(request)
    changed = request.model_copy(
        update={"system_prompt": request.system_prompt + " changed"}
    )
    with pytest.raises(RecoveryRequired):
        await runner.model(changed)
    journal.db.execute("UPDATE heads SET body='{}'")
    journal.db.commit()
    with pytest.raises(ValueError, match="checksum"):
        await runner.model(request)
    assert len(provider.calls) == 1
