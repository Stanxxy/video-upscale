import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4, uuid5

import pytest
from bjj_analysis_core.execution import (
    ProviderReplyV1,
    RecoveryRequired,
    StepCheckpointV1,
    StepRunner,
)
from bjj_analysis_core.executor import execute_pipeline
from bjj_analysis_core.reporting import report_execution
from bjj_analysis_core.requests import prepare_call
from conftest import authorized
from shared_lib.models.analysis_contracts import CallUsageV1, NodeProgressV1


def checkpoint(run):
    step = uuid4()
    now = datetime.now(timezone.utc)
    return StepCheckpointV1(
        step_id=step,
        input_sha256="f" * 64,
        revision=1,
        phase="running",
        request=None,
        progress=NodeProgressV1(
            run_id=run.spec.run_id,
            execution_id=step,
            node_id="deterministic",
            attempt=1,
            sequence=1,
            status="running",
            started_at=now,
            updated_at=now,
            total_items=1,
        ),
        usage=CallUsageV1(
            run_id=run.spec.run_id,
            execution_id=step,
            call_id=uuid5(step, "attempt:1"),
            recorded_at=now,
            kind="deterministic",
            usage_status="not_applicable",
            cost_status="not_applicable",
            cost_nano_usd=0,
        ),
    )


def finished(record):
    now = datetime.now(timezone.utc)
    return StepCheckpointV1.model_validate(
        {
            **record.model_dump(),
            "revision": 2,
            "phase": "succeeded",
            "result_json": '"ok"',
            "progress": {
                **record.progress.model_dump(),
                "sequence": 2,
                "status": "succeeded",
                "completed_items": 1,
                "updated_at": now,
                "finished_at": now,
            },
        }
    )


@pytest.mark.asyncio
async def test_real_conditional_history_and_reopen(adapter, resources, run, journal):
    first = checkpoint(run)
    assert await journal.read(first.step_id) is None
    assert await journal.compare_and_swap(None, first)
    assert not await journal.compare_and_swap(None, first)
    second = finished(first)
    assert await journal.compare_and_swap(1, second)
    session, s3, keyspace, bucket = resources
    reopened = adapter.KeyspacesCheckpointJournal(
        run, session, s3, bucket=bucket, keyspace=keyspace, guard=authorized
    )
    assert await reopened.read(first.step_id) == second
    page, cursor = await reopened.history_page(first.step_id, limit=1)
    assert page == (first,) and cursor == 1
    page2, cursor2 = await reopened.history_page(
        first.step_id, after_revision=cursor, limit=1
    )
    assert page2 == (second,) and cursor2 is None
    report = report_execution(page + page2)
    assert report.total_costs.unknown_cost_calls == 0
    row = session.execute(
        f"SELECT artifact_key FROM {keyspace}.analysis_execution_checkpoints WHERE run_id=%s AND step_id=%s AND revision=0",
        [run.spec.run_id, first.step_id],
    ).one()
    metadata = s3.head_object(Bucket=bucket, Key=row.artifact_key)
    assert metadata["ServerSideEncryption"] == "AES256"
    assert (
        s3.get_object_acl(Bucket=bucket, Key=row.artifact_key)["Grants"][0][
            "Permission"
        ]
        == "FULL_CONTROL"
    )


@pytest.mark.asyncio
async def test_concurrent_reservations_atomic_no_losing_history(journal, run):
    first = checkpoint(run)
    competitor = first.model_copy(update={"input_sha256": "e" * 64})
    barrier, execute = asyncio.Barrier(2), journal._execute

    async def interleaved(query, values, *, conditional=False):
        if conditional:
            await asyncio.wait_for(barrier.wait(), 5)
        return await execute(query, values, conditional=conditional)

    journal._execute = interleaved
    outcomes = await asyncio.gather(
        journal.compare_and_swap(None, first),
        journal.compare_and_swap(None, competitor),
    )
    assert sum(outcomes) == 1
    winner = first if outcomes[0] else competitor
    assert await journal.read(first.step_id) == winner
    rows, cursor = await journal.history_page(first.step_id)
    assert rows == (winner,) and cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [False, True])
async def test_lost_cql_commit_propagates_and_recovers_by_read(journal, run, after):
    record = checkpoint(run)
    execute = journal._execute

    async def interrupted(query, values, *, conditional=False):
        if conditional:
            if after:
                await execute(query, values, conditional=True)
            raise TimeoutError("conditional response unknown")
        return await execute(query, values, conditional=conditional)

    journal._execute = interrupted
    with pytest.raises(TimeoutError):
        await journal.compare_and_swap(None, record)
    journal._execute = execute
    assert await journal.read(record.step_id) == (record if after else None)
    rows, _ = await journal.history_page(record.step_id)
    assert rows == ((record,) if after else ())


@pytest.mark.asyncio
async def test_read_errors_are_not_absent_checkpoints(journal):
    async def down(*args, **kwargs):
        raise TimeoutError("Cassandra unavailable")

    journal._execute = down
    with pytest.raises(TimeoutError):
        await journal.read(uuid4())


@pytest.mark.asyncio
async def test_checksum_namespace_and_missing_history_fail_closed(
    journal, run, resources, adapter
):
    session, s3, keyspace, bucket = resources
    record = checkpoint(run)
    assert await journal.compare_and_swap(None, record)
    assert await journal.compare_and_swap(1, finished(record))
    session.execute(
        f"DELETE FROM {keyspace}.analysis_execution_checkpoints WHERE run_id=%s AND step_id=%s AND revision=1",
        [run.spec.run_id, record.step_id],
    )
    with pytest.raises(adapter.CheckpointIntegrityError, match="gap"):
        await journal.history_page(record.step_id)
    row = session.execute(
        f"SELECT artifact_key FROM {keyspace}.analysis_execution_checkpoints WHERE run_id=%s AND step_id=%s AND revision=0",
        [run.spec.run_id, record.step_id],
    ).one()
    s3.put_object(Bucket=bucket, Key=row.artifact_key, Body=b"corrupt")
    with pytest.raises(adapter.CheckpointIntegrityError, match="checksum"):
        await journal.read(record.step_id)
    session.execute(
        f"UPDATE {keyspace}.analysis_execution_checkpoints SET artifact_key=%s WHERE run_id=%s AND step_id=%s AND revision=0",
        ["foreign/run/object", run.spec.run_id, record.step_id],
    )
    with pytest.raises(adapter.CheckpointIntegrityError, match="outside"):
        await journal.read(record.step_id)


@pytest.mark.asyncio
async def test_guards_revision_types_and_payload_cap(
    journal, run, adapter, monkeypatch
):
    record = checkpoint(run)
    with pytest.raises(ValueError):
        await journal.compare_and_swap(True, record)
    with pytest.raises(ValueError):
        await journal.history_page(record.step_id, limit=True)
    monkeypatch.setattr(adapter, "MAX_CHECKPOINT_BYTES", 128)
    with pytest.raises(adapter.CheckpointIntegrityError, match="limit"):
        await journal.compare_and_swap(None, record)
    assert await journal.read(record.step_id) is None

    async def revoked():
        raise PermissionError("run revoked")

    journal.guard = revoked
    with pytest.raises(PermissionError):
        await journal.read(record.step_id)
    with pytest.raises(PermissionError):
        await journal.compare_and_swap(None, record)


@pytest.mark.asyncio
async def test_full_empty_dag_real_journal_replay_and_history(
    journal, run, configuration, resources
):
    calls, outputs = [], {}

    class Provider:
        name = "fixture"

        async def generate(self, run, request, execution_id, call_id):
            calls.append(request.node_id)
            if request.node_id == "position_scan":
                data = dict(
                    initial_state=dict(
                        observation="observed",
                        position="standing",
                        roles=["neutral", "neutral"],
                        notes="",
                        evidence="both standing",
                        uncertainty_note=None,
                        interruption_note=None,
                    ),
                    transitions=[],
                )
            else:
                assert request.node_id == "technique_scan" and "positions" in outputs
                data = dict(events=[])
            return ProviderReplyV1(
                text=json.dumps(data),
                usage=CallUsageV1(
                    run_id=run.spec.run_id,
                    execution_id=execution_id,
                    call_id=call_id,
                    recorded_at=datetime.now(timezone.utc),
                    kind="model",
                    provider=self.name,
                    model=request.config.model,
                    usage_status="unknown",
                    cost_status="unknown",
                ),
            )

    class Sink:
        async def publish(self, run, component, revision):
            outputs[component] = revision

    first = await execute_pipeline(
        run,
        configuration,
        journal=journal,
        provider=Provider(),
        sink=Sink(),
        guard=authorized,
    )
    second = await execute_pipeline(
        run,
        configuration,
        journal=journal,
        provider=Provider(),
        sink=Sink(),
        guard=authorized,
    )
    assert first == second and first.techniques.techniques == ()
    assert calls == ["position_scan", "technique_scan"]
    # Inventory is test-owned; product node inventory/projections remain a host concern.
    session, _, keyspace, _ = resources
    ids = {
        r.step_id
        for r in session.execute(
            f"SELECT step_id FROM {keyspace}.analysis_execution_checkpoints"
        )
    }
    history = []
    for step in ids:
        page, cursor = await journal.history_page(step)
        assert cursor is None
        history.extend(page)
    assert report_execution(history).total_costs.unknown_cost_calls == 2


def test_migration_matches_adapter_ddl(adapter):
    path = (
        Path(__file__).resolve().parents[3]
        / "bjj-vision-backend-wt-s39/infrastructure/keyspaces/migrations/010_analysis_execution_checkpoints.cql"
    )
    sql = "\n".join(
        line for line in path.read_text().splitlines() if not line.startswith("--")
    )
    assert "".join(sql.split()).rstrip(";") == "".join(
        adapter.checkpoint_table_ddl("video_analysis").split()
    )
    with pytest.raises(ValueError):
        adapter.checkpoint_table_ddl("a; DROP KEYSPACE b")


@pytest.mark.asyncio
async def test_racing_updates_keep_one_complete_history(journal, run):
    first = checkpoint(run)
    assert await journal.compare_and_swap(None, first)
    second = finished(first)
    competitor = second.model_copy(update={"result_json": '"other"'})
    barrier, execute = asyncio.Barrier(2), journal._execute

    async def race(query, values, *, conditional=False):
        if conditional:
            await asyncio.wait_for(barrier.wait(), 5)
        return await execute(query, values, conditional=conditional)

    journal._execute = race
    results = await asyncio.gather(
        journal.compare_and_swap(1, second), journal.compare_and_swap(1, competitor)
    )
    assert sum(results) == 1
    winner = second if results[0] else competitor
    rows, cursor = await journal.history_page(first.step_id)
    assert rows == (first, winner) and cursor is None
    assert await journal.read(first.step_id) == winner


@pytest.mark.asyncio
@pytest.mark.parametrize("after_put", [False, True])
async def test_s3_failure_cannot_commit_checkpoint_and_exact_object_reuse(
    journal, run, after_put
):
    record = checkpoint(run)
    put = journal._put_bytes

    def lost_reply(checkpoint):
        if after_put:
            put(checkpoint)
        raise OSError("S3 response unavailable")

    journal._put_bytes = lost_reply
    with pytest.raises(OSError):
        await journal.compare_and_swap(None, record)
    assert await journal.read(record.step_id) is None
    journal._put_bytes = put
    assert await journal.compare_and_swap(None, record)
    assert await journal.read(record.step_id) == record


@pytest.mark.asyncio
async def test_malformed_model_receipt_survives_restart_and_retry(
    journal, run, configuration, adapter, resources
):
    calls = []

    class Provider:
        name = "fixture"

        async def generate(self, run, request, execution_id, call_id):
            calls.append(call_id)
            # Reservation is visible in the real database BEFORE model dispatch.
            assert (await journal.read(execution_id)).phase == "running"
            data = (
                "not-json"
                if len(calls) == 1
                else json.dumps(
                    dict(
                        initial_state=dict(
                            observation="observed",
                            position="standing",
                            roles=["neutral", "neutral"],
                            notes="",
                            evidence="visible standing athletes",
                            uncertainty_note=None,
                            interruption_note=None,
                        ),
                        transitions=[],
                    )
                )
            )
            return ProviderReplyV1(
                text=data,
                usage=CallUsageV1(
                    run_id=run.spec.run_id,
                    execution_id=execution_id,
                    call_id=call_id,
                    recorded_at=datetime.now(timezone.utc),
                    kind="model",
                    provider=self.name,
                    model=request.config.model,
                    usage_status="complete",
                    input_tokens=10,
                    output_tokens=5,
                    cost_status="unknown",
                ),
            )

    request = prepare_call(
        run, configuration, "position_scan", input_window=run.spec.scope, context={}
    )
    provider = Provider()
    with pytest.raises(ValueError):
        await StepRunner(run, configuration, journal, provider).model(request)
    session, s3, keyspace, bucket = resources
    reopened = adapter.KeyspacesCheckpointJournal(
        run, session, s3, bucket=bucket, keyspace=keyspace, guard=authorized
    )
    with pytest.raises(RecoveryRequired):
        await StepRunner(run, configuration, reopened, provider).model(request)
    assert len(calls) == 1
    assert (
        await StepRunner(
            run, configuration, reopened, provider, retry_failed=True
        ).model(request)
    ).transitions == ()
    assert len(calls) == 2 and calls[0] != calls[1]
    step = uuid5(run.spec.run_id, "step:v1:position_scan:")
    history, cursor = [], 0
    while cursor is not None:
        page, cursor = await reopened.history_page(step, after_revision=cursor, limit=1)
        history.extend(page)
    failed = next(r for r in history if r.phase == "failed")
    assert failed.reply.text == "not-json" and failed.usage.input_tokens == 10
    assert report_execution(history).total_costs.unknown_cost_calls == 2


@pytest.mark.asyncio
async def test_revocation_after_artifact_write_prevents_cas(journal, run):
    record = checkpoint(run)
    put, revoked = journal._put_bytes, False

    def revoke_after_write(value):
        nonlocal revoked
        ref = put(value)
        revoked = True
        return ref

    async def guard():
        if revoked:
            raise PermissionError("lease revoked before conditional commit")

    journal._put_bytes, journal.guard = revoke_after_write, guard
    with pytest.raises(PermissionError):
        await journal.compare_and_swap(None, record)
    journal.guard = authorized
    assert await journal.read(record.step_id) is None


@pytest.mark.asyncio
async def test_mid_cql_revocation_denies_success_but_retains_commit(journal, run):
    record = checkpoint(run)
    execute, revoked = journal._execute, False

    async def revoke_on_commit(query, values, *, conditional=False):
        nonlocal revoked
        rows = await execute(query, values, conditional=conditional)
        if conditional:
            revoked = True
        return rows

    async def guard():
        if revoked:
            raise PermissionError("lease revoked while Cassandra was executing")

    journal._execute, journal.guard = revoke_on_commit, guard
    with pytest.raises(PermissionError):
        await journal.compare_and_swap(None, record)
    journal.guard = authorized
    assert await journal.read(record.step_id) == record


@pytest.mark.asyncio
async def test_actual_task_cancellation_during_driver_call_preserves_history(
    journal, run
):
    record = checkpoint(run)
    committed, release, exited = threading.Event(), threading.Event(), threading.Event()
    execute = journal._execute_sync

    def delayed_return(query, values, *, conditional=False):
        result = execute(query, values, conditional=conditional)
        if conditional:
            committed.set()
            try:
                assert release.wait(5)
            finally:
                exited.set()
        return result

    journal._execute_sync = delayed_return
    task = asyncio.create_task(journal.compare_and_swap(None, record))
    try:
        assert await asyncio.to_thread(committed.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        assert await asyncio.to_thread(exited.wait, 5)
    journal._execute_sync = execute
    assert await journal.read(record.step_id) == record
    assert (await journal.history_page(record.step_id))[0] == (record,)


@pytest.mark.asyncio
async def test_missing_head_is_not_empty_history(journal, run, resources, adapter):
    record = checkpoint(run)
    assert await journal.compare_and_swap(None, record)
    session, _, keyspace, _ = resources
    session.execute(
        f"DELETE FROM {keyspace}.analysis_execution_checkpoints WHERE run_id=%s AND step_id=%s AND revision=0",
        [run.spec.run_id, record.step_id],
    )
    with pytest.raises(adapter.CheckpointIntegrityError, match="no checkpoint head"):
        await journal.read(record.step_id)
    with pytest.raises(adapter.CheckpointIntegrityError, match="no checkpoint head"):
        await journal.history_page(record.step_id)


@pytest.mark.asyncio
async def test_concurrent_first_commit_during_missing_head_probe_is_valid(
    journal, run, resources, adapter
):
    record = checkpoint(run)
    session, s3, keyspace, bucket = resources
    writer = adapter.KeyspacesCheckpointJournal(
        run, session, s3, bucket=bucket, keyspace=keyspace, guard=authorized
    )
    execute, inserted = journal._execute, False

    async def insert_after_empty_head(query, values, *, conditional=False):
        nonlocal inserted
        rows = await execute(query, values, conditional=conditional)
        if not inserted and "revision=0" in query and not rows:
            inserted = True
            assert await writer.compare_and_swap(None, record)
        return rows

    journal._execute = insert_after_empty_head
    assert await journal.read(record.step_id) == record
