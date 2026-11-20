import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from bjj_analysis_core.execution import CheckpointConflict, RecoveryRequired
from conftest import Wire

from service.analysis_gemini import GeminiProvider, S3GeminiArtifacts
from service.analysis_run_host import DurableGeminiRun


@asynccontextmanager
async def case(resolver, inputs, s3, journal):
    async with resolver.open(inputs[0]) as media:
        wire, publications = Wire(media), []

        async def handler(request):
            if ":generateContent" in str(request.url):
                data = json.loads(await request.aread())
                schema = data["generationConfig"]["responseJsonSchema"]
                output = dict(events=[])
                if "initial_state" in schema["properties"]:
                    output = dict(
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
                wire.result["candidates"][0]["content"]["parts"] = [
                    {"text": json.dumps(output)}
                ]
            return await wire(request)

        provider = GeminiProvider(
            media,
            S3GeminiArtifacts(media.run, s3[0], bucket=s3[1]),
            api_key="fixture-secret",
            qualified_models={"fixture-model": {"fixture-model-r1"}},
            guard=resolver.guard,
            transport=httpx.MockTransport(handler),
        )

        class Sink:
            async def publish(self, run, component, revision):
                publications.append((component, revision))

        async def guard():
            await resolver.guard(media.run)

        host = DurableGeminiRun(
            media.run,
            media.config_bytes,
            journal=journal,
            provider=provider,
            sink=Sink(),
            guard=guard,
            mime_type="video/mp4",
        )
        try:
            yield host, wire, publications
        finally:
            await provider.aclose()


@pytest.mark.asyncio
async def test_real_dag_replay_and_cleanup_are_separate_durable_operations(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, publications):
        result = await host.execute()
        assert [p[0] for p in publications] == ["positions", "techniques"]
        assert len(wire.generates()) == 2
        assert result.techniques.techniques == ()
        count = len(wire.calls)
        host.provider.uploaded = None  # simulated fresh provider binding after restart
        assert await host.execute() == result
        assert len(wire.calls) == count
        receipt = await host.cleanup()
        assert len([c for c in wire.calls if c[0] == "DELETE"]) == 1
        count = len(wire.calls)
        assert await host.cleanup() == receipt
        assert await host.execute() == result
        assert len(wire.calls) == count  # no fresh media or model after deletion
        for node in ("provider_upload", "provider_cleanup", "dag_complete"):
            history, cursor = await journal.history_page(host._identity(node)[0])
            assert [r.phase for r in history] == ["running", "succeeded"]
            assert cursor is None and all(r.usage.cost_nano_usd == 0 for r in history)
            assert all(r.usage.kind == "deterministic" for r in history)


@pytest.mark.asyncio
async def test_upload_cache_rebinds_without_http(resolver, inputs, s3, journal):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        upload = await host.ensure_upload()
        count = len(wire.calls)
        host.provider.uploaded = None
        assert await host.ensure_upload() == upload
        assert host.provider.uploaded == upload and len(wire.calls) == count


@pytest.mark.asyncio
async def test_concurrent_upload_reservations_have_one_remote_winner(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        barrier, save = asyncio.Barrier(2), journal.compare_and_swap

        async def race(expected, record):
            if (
                record.progress.node_id == "provider_upload"
                and record.phase == "running"
            ):
                await asyncio.wait_for(barrier.wait(), 5)
            return await save(expected, record)

        journal.compare_and_swap = race
        outcomes = await asyncio.gather(
            host.ensure_upload(), host.ensure_upload(), return_exceptions=True
        )
        assert sum(not isinstance(r, BaseException) for r in outcomes) == 1
        assert any(
            isinstance(r, (CheckpointConflict, RecoveryRequired)) for r in outcomes
        )
        assert len(wire.calls) == 2
        assert (await host._read("provider_upload")).phase == "succeeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("after_commit", [False, True])
async def test_lost_upload_completion_ack_never_reuploads(
    resolver, inputs, s3, journal, after_commit
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        save = journal.compare_and_swap

        async def lost(expected, record):
            if (
                record.progress.node_id == "provider_upload"
                and record.phase == "succeeded"
            ):
                if after_commit:
                    await save(expected, record)
                raise TimeoutError("checkpoint ack lost")
            return await save(expected, record)

        journal.compare_and_swap = lost
        with pytest.raises(TimeoutError):
            await host.ensure_upload()
        assert len(wire.calls) == 2
        journal.compare_and_swap = save
        host.provider.uploaded = None
        if after_commit:
            await host.ensure_upload()
        else:
            with pytest.raises(RecoveryRequired):
                await host.ensure_upload()
        assert len(wire.calls) == 2


@pytest.mark.asyncio
async def test_remote_upload_failure_leaves_uncertain_reservation(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):

        async def uncertain(*args, **kw):
            raise httpx.ReadTimeout("upload result unknown")

        host.provider.upload = uncertain
        with pytest.raises(httpx.ReadTimeout):
            await host.ensure_upload()
        assert (await host._read("provider_upload")).phase == "running"
        with pytest.raises(RecoveryRequired):
            await host.ensure_upload()
        assert wire.calls == []


@pytest.mark.asyncio
async def test_cleanup_before_dag_completion_is_forbidden(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        await host.ensure_upload()
        with pytest.raises(RecoveryRequired, match="completed DAG"):
            await host.cleanup()
        assert not any(c[0] == "DELETE" for c in wire.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("after_commit", [False, True])
async def test_cleanup_commit_loss_never_repeats_delete_or_paid_work(
    resolver, inputs, s3, journal, after_commit
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        result = await host.execute()
        save = journal.compare_and_swap

        async def lost(expected, record):
            if (
                record.progress.node_id == "provider_cleanup"
                and record.phase == "succeeded"
            ):
                if after_commit:
                    await save(expected, record)
                raise TimeoutError("cleanup commit unknown")
            return await save(expected, record)

        journal.compare_and_swap = lost
        with pytest.raises(TimeoutError):
            await host.cleanup()
        journal.compare_and_swap = save
        if after_commit:
            await host.cleanup()
        else:
            with pytest.raises(RecoveryRequired):
                await host.cleanup()
        assert await host.execute() == result
        with pytest.raises(RecoveryRequired, match="cleanup began"):
            await host.ensure_upload()
        assert len([c for c in wire.calls if c[0] == "DELETE"]) == 1
        assert len(wire.generates()) == 2


@pytest.mark.asyncio
async def test_failed_result_marker_is_rebuilt_without_model_or_publication(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, publications):
        save = journal.compare_and_swap

        async def lost(expected, record):
            if (
                record.progress.node_id == "dag_complete"
                and record.phase == "succeeded"
            ):
                raise TimeoutError("pure result marker not committed")
            return await save(expected, record)

        journal.compare_and_swap = lost
        with pytest.raises(TimeoutError):
            await host.execute()
        journal.compare_and_swap = save
        count = len(wire.calls)
        result = await host.execute()
        assert len(wire.calls) == count and len(publications) == 2
        assert result.techniques.techniques == ()
        assert (await host._read("dag_complete")).progress.attempt == 2


@pytest.mark.asyncio
async def test_revocation_after_reservation_prevents_external_action(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        save, revoked = journal.compare_and_swap, False

        async def committed(expected, record):
            nonlocal revoked
            result = await save(expected, record)
            revoked = True
            return result

        async def guard():
            if revoked:
                raise PermissionError("lease revoked")

        host.guard, journal.compare_and_swap = guard, committed
        with pytest.raises(PermissionError):
            await host.ensure_upload()
        assert wire.calls == []


@pytest.mark.asyncio
async def test_cancelled_upload_reservation_is_not_reclaimed(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        entered = asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        host.provider.upload = blocked
        task = asyncio.create_task(host.ensure_upload())
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(RecoveryRequired):
            await host.ensure_upload()
        assert wire.calls == []


@pytest.mark.asyncio
async def test_lost_reservation_ack_stops_before_upload(resolver, inputs, s3, journal):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        save = journal.compare_and_swap

        async def lost(expected, record):
            await save(expected, record)
            raise TimeoutError("reservation committed, response lost")

        journal.compare_and_swap = lost
        with pytest.raises(TimeoutError):
            await host.ensure_upload()
        journal.compare_and_swap = save
        with pytest.raises(RecoveryRequired):
            await host.ensure_upload()
        assert wire.calls == []


@pytest.mark.asyncio
async def test_partial_result_stays_available_without_terminal_cleanup(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, publications):
        generate = host.provider.generate

        async def failed_technique(run, request, execution_id, call_id):
            if request.node_id == "technique_scan":
                wire.status = 503
            return await generate(run, request, execution_id, call_id)

        host.provider.generate = failed_technique
        with pytest.raises(RuntimeError, match="HTTP 503"):
            await host.execute()
        assert [p[0] for p in publications] == ["positions"]
        assert await host._read("dag_complete") is None
        with pytest.raises(RecoveryRequired, match="completed DAG"):
            await host.cleanup()
        with pytest.raises(RecoveryRequired):
            await host.execute(
                retry_failed=True
            )  # uncertain model call is not retryable
        assert len(wire.generates()) == 2 and not any(
            c[0] == "DELETE" for c in wire.calls
        )


@pytest.mark.asyncio
async def test_publication_failure_cannot_mark_dag_complete(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, publications):
        publish, blocked = host.sink.publish, True

        async def unavailable(run, component, revision):
            if component == "techniques" and blocked:
                raise TimeoutError("sink not durably acknowledged")
            await publish(run, component, revision)

        host.sink.publish = unavailable
        with pytest.raises(TimeoutError):
            await host.execute()
        assert await host._read("dag_complete") is None
        with pytest.raises(RecoveryRequired):
            await host.cleanup()
        count = len(wire.calls)
        blocked = False
        await host.execute(retry_failed=True)
        assert len(wire.calls) == count and [p[0] for p in publications] == [
            "positions",
            "techniques",
        ]


@pytest.mark.asyncio
async def test_changed_host_inputs_reject_cached_external_operation(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        await host.ensure_upload()
        host.mime_type = "video/quicktime"
        count = len(wire.calls)
        with pytest.raises(RecoveryRequired, match="identity changed"):
            await host.ensure_upload()
        assert len(wire.calls) == count


@pytest.mark.asyncio
async def test_concurrent_cleanup_has_one_delete(resolver, inputs, s3, journal):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        await host.execute()
        barrier, save = asyncio.Barrier(2), journal.compare_and_swap

        async def race(expected, record):
            if (
                record.progress.node_id == "provider_cleanup"
                and record.phase == "running"
            ):
                await asyncio.wait_for(barrier.wait(), 5)
            return await save(expected, record)

        journal.compare_and_swap = race
        results = await asyncio.gather(
            host.cleanup(), host.cleanup(), return_exceptions=True
        )
        assert sum(not isinstance(r, BaseException) for r in results) == 1
        assert any(isinstance(r, CheckpointConflict) for r in results)
        assert len([c for c in wire.calls if c[0] == "DELETE"]) == 1


@pytest.mark.asyncio
async def test_cancelled_cleanup_does_not_redelete_or_lose_result(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        result = await host.execute()
        entered = asyncio.Event()

        async def uncertain(operation_id):
            entered.set()
            await asyncio.Event().wait()

        host.provider.delete_upload = uncertain
        task = asyncio.create_task(host.cleanup())
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(RecoveryRequired):
            await host.cleanup()
        count = len(wire.calls)
        assert await host.execute() == result and len(wire.calls) == count
