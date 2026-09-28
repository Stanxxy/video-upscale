import asyncio
import base64
import json
from datetime import timedelta
from uuid import uuid4

import pytest
from bjj_analysis_core.execution import CheckpointConflict, RecoveryRequired
from botocore.exceptions import ClientError
from test_host import case

from service.analysis_gemini import GeminiBoundaryError


async def lose_completion(host, node):
    """Let real provider/S3 operations succeed, lose only the journal completion."""
    save, retain, receipts = (
        host.journal.compare_and_swap,
        host.provider.artifacts.retain,
        {},
    )

    async def recorded(kind, identity, value):
        ref = await retain(kind, identity, value)
        receipts[kind] = ref
        return ref

    async def lost(expected, record):
        if record.progress.node_id == node and record.phase == "succeeded":
            raise TimeoutError("completion acknowledgment unavailable")
        return await save(expected, record)

    host.provider.artifacts.retain = recorded
    host.journal.compare_and_swap = lost
    try:
        with pytest.raises(TimeoutError):
            if node == "provider_upload":
                await host.ensure_upload()
            else:
                await host.cleanup()
    finally:
        host.journal.compare_and_swap = save
        host.provider.artifacts.retain = retain
    assert (await host._read(node)).phase == "running"
    host.provider.uploaded = None
    return receipts["upload_finalize" if node == "provider_upload" else "file_delete"]


@pytest.mark.asyncio
async def test_upload_receipt_recovery_rechecks_file_and_keeps_attempt(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        receipt = await lose_completion(host, "provider_upload")
        upload = await host.reconcile_upload(receipt)
        assert [call[0] for call in wire.calls] == ["POST", "POST", "GET"]
        assert upload.receipt != receipt
        response = s3[0].get_object(
            Bucket=upload.receipt.bucket, Key=upload.receipt.key
        )
        with response["Body"] as body:
            evidence = json.loads(body.read())
        assert evidence["context"]["reconciled_from"] == receipt.model_dump(mode="json")
        count = len(wire.calls)
        assert await host.reconcile_upload(receipt) == upload
        assert await host.ensure_upload() == upload
        assert len(wire.calls) == count
        history, _ = await journal.history_page(host._identity("provider_upload")[0])
        assert [item.phase for item in history] == ["running", "succeeded"]
        assert {item.progress.attempt for item in history} == {1}
        assert len({item.usage.call_id for item in history}) == 1


@pytest.mark.asyncio
async def test_cleanup_receipt_recovery_never_redeletes_or_reexecutes(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        result = await host.execute()
        receipt = await lose_completion(host, "provider_cleanup")
        count = len(wire.calls)
        assert await host.reconcile_cleanup(receipt) == receipt
        assert host.provider.uploaded is None
        assert await host.reconcile_cleanup(receipt) == receipt
        assert await host.cleanup() == receipt
        assert await host.execute() == result
        assert len(wire.calls) == count
        assert sum(call[0] == "DELETE" for call in wire.calls) == 1
        assert len(wire.generates()) == 2
        with pytest.raises(RecoveryRequired):
            await host.reconcile_upload(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["identity", "run", "kind", "status", "before", "future", "bytes"]
)
async def test_untrusted_upload_receipt_cannot_complete_reservation(
    resolver, inputs, s3, journal, mutation
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        receipt = await lose_completion(host, "provider_upload")
        record = await host._read("provider_upload")
        artifacts = host.provider.artifacts
        value = await artifacts.read(
            receipt, kind="upload_finalize", identity=record.usage.call_id
        )
        if mutation == "identity":
            value["identity"] = str(uuid4())
        elif mutation == "run":
            value["run_id"] = str(uuid4())
        elif mutation == "kind":
            value["kind"] = "file_get"
        elif mutation == "status":
            value["status"] = 503
        elif mutation == "before":
            value["recorded_at"] = (
                record.progress.started_at - timedelta(seconds=1)
            ).isoformat()
        elif mutation == "future":
            value["recorded_at"] = (host.provider.now() + timedelta(days=1)).isoformat()
        else:
            body = json.loads(base64.b64decode(value["body_base64"]))
            body["file"]["sizeBytes"] = "1"
            value["body_base64"] = base64.b64encode(json.dumps(body).encode()).decode()
        bad = await artifacts.retain("upload_finalize", record.usage.call_id, value)
        count = len(wire.calls)
        with pytest.raises(GeminiBoundaryError):
            await host.reconcile_upload(bad)
        assert len(wire.calls) == count
        assert (await host._read("provider_upload")).phase == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["key", "bucket", "checksum"])
async def test_receipt_location_and_bytes_are_verified(
    resolver, inputs, s3, journal, mutation
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        receipt = await lose_completion(host, "provider_upload")
        if mutation == "key":
            receipt = receipt.model_copy(update={"key": "another-run/receipt.json"})
        elif mutation == "bucket":
            receipt = receipt.model_copy(update={"bucket": "another-private-bucket"})
        else:
            # Only this test's disposable receipt object is corrupted.
            s3[0].put_object(Bucket=receipt.bucket, Key=receipt.key, Body=b"{}")
        count = len(wire.calls)
        with pytest.raises(GeminiBoundaryError):
            await host.reconcile_upload(receipt)
        assert len(wire.calls) == count
        assert (await host._read("provider_upload")).phase == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["name", "state", "expiry"])
async def test_fresh_file_must_still_be_usable(resolver, inputs, s3, journal, mutation):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        receipt = await lose_completion(host, "provider_upload")
        if mutation == "name":
            wire.file["name"] = "files/unrelated"
        elif mutation == "state":
            wire.file["state"] = "PROCESSING"
        else:
            wire.file["expirationTime"] = host.provider.now().isoformat()
        with pytest.raises(GeminiBoundaryError):
            await host.reconcile_upload(receipt)
        assert [call[0] for call in wire.calls] == ["POST", "POST", "GET"]
        assert (await host._read("provider_upload")).phase == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context", [None, {"uploaded_name": "files/wrong", "source_sha256": "0" * 64}]
)
async def test_cleanup_requires_exact_target_evidence(
    resolver, inputs, s3, journal, context
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        await host.execute()
        receipt = await lose_completion(host, "provider_cleanup")
        record = await host._read("provider_cleanup")
        artifacts = host.provider.artifacts
        value = await artifacts.read(
            receipt, kind="file_delete", identity=record.usage.call_id
        )
        value["context"] = context
        bad = await artifacts.retain("file_delete", record.usage.call_id, value)
        count = len(wire.calls)
        with pytest.raises(GeminiBoundaryError, match="target mismatch"):
            await host.reconcile_cleanup(bad)
        assert len(wire.calls) == count
        assert (await host._read("provider_cleanup")).phase == "running"


@pytest.mark.asyncio
async def test_reconciliation_requires_prior_reservation_and_complete_dag(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        with pytest.raises(RecoveryRequired, match="reservation"):
            await host.reconcile_upload(None)
        with pytest.raises(RecoveryRequired, match="completed DAG"):
            await host.reconcile_cleanup(None)
        assert wire.calls == []


@pytest.mark.asyncio
async def test_reconciliation_revocation_after_get_prevents_commit(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        receipt = await lose_completion(host, "provider_upload")
        request, guard = host.provider._request, host.guard

        async def denied():
            raise PermissionError("authority revoked")

        async def revoke(*args, **kwargs):
            result = await request(*args, **kwargs)
            host.guard = denied
            return result

        host.provider._request = revoke
        with pytest.raises(PermissionError):
            await host.reconcile_upload(receipt)
        host.guard = guard
        assert (await host._read("provider_upload")).phase == "running"
        assert host.provider.uploaded is None
        assert len(wire.calls) == 3


@pytest.mark.asyncio
async def test_cancelled_reconciliation_can_repeat_read_but_not_upload(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        receipt = await lose_completion(host, "provider_upload")
        request, entered = host.provider._request, asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        host.provider._request = blocked
        task = asyncio.create_task(host.reconcile_upload(receipt))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await host._read("provider_upload")).phase == "running"
        host.provider._request = request
        await host.reconcile_upload(receipt)
        assert [call[0] for call in wire.calls] == ["POST", "POST", "GET"]


@pytest.mark.asyncio
async def test_competing_receipt_reconciliation_has_one_checkpoint_winner(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        receipt = await lose_completion(host, "provider_upload")
        save, barrier = journal.compare_and_swap, asyncio.Barrier(2)

        async def race(expected, record):
            await asyncio.wait_for(barrier.wait(), 5)
            return await save(expected, record)

        journal.compare_and_swap = race
        results = await asyncio.gather(
            host.reconcile_upload(receipt),
            host.reconcile_upload(receipt),
            return_exceptions=True,
        )
        journal.compare_and_swap = save
        assert sum(isinstance(result, CheckpointConflict) for result in results) == 1
        assert sum(not isinstance(result, BaseException) for result in results) == 1
        assert [call[0] for call in wire.calls] == ["POST", "POST", "GET", "GET"]
        assert (await host._read("provider_upload")).revision == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("node", ["provider_upload", "provider_cleanup"])
@pytest.mark.parametrize("after_commit", [False, True])
async def test_reconciliation_commit_loss_reuses_evidence_without_external_retry(
    resolver, inputs, s3, journal, node, after_commit
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        if node == "provider_cleanup":
            await host.execute()
        receipt = await lose_completion(host, node)
        reconcile = (
            host.reconcile_upload
            if node == "provider_upload"
            else host.reconcile_cleanup
        )
        save = journal.compare_and_swap

        async def lost(expected, record):
            if after_commit:
                await save(expected, record)
            raise TimeoutError("reconciliation commit unknown")

        journal.compare_and_swap = lost
        with pytest.raises(TimeoutError):
            await reconcile(receipt)
        journal.compare_and_swap = save
        count = len(wire.calls)
        await reconcile(receipt)
        additional_gets = 1 if node == "provider_upload" and not after_commit else 0
        assert len(wire.calls) == count + additional_gets
        assert (
            sum(call[0] == "POST" and "/upload/" in call[1] for call in wire.calls) == 2
        )
        assert sum(call[0] == "DELETE" for call in wire.calls) == (
            node == "provider_cleanup"
        )
        assert (await host._read(node)).revision == 2


@pytest.mark.asyncio
async def test_processing_finalize_receipt_recovers_only_after_fresh_active_get(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        receipt = await lose_completion(host, "provider_upload")
        record = await host._read("provider_upload")
        artifacts = host.provider.artifacts
        value = await artifacts.read(
            receipt, kind="upload_finalize", identity=record.usage.call_id
        )
        body = json.loads(base64.b64decode(value["body_base64"]))
        body["file"]["state"] = "PROCESSING"
        value["body_base64"] = base64.b64encode(json.dumps(body).encode()).decode()
        processing = await artifacts.retain(
            "upload_finalize", record.usage.call_id, value
        )
        await host.reconcile_upload(processing)
        assert [call[0] for call in wire.calls] == ["POST", "POST", "GET"]


@pytest.mark.asyncio
async def test_missing_receipt_is_not_evidence_of_failed_upload(
    resolver, inputs, s3, journal
):
    async with case(resolver, inputs, s3, journal) as (host, wire, _):
        receipt = await lose_completion(host, "provider_upload")
        missing = receipt.model_copy(
            update={
                "key": receipt.key.replace(receipt.sha256, "0" * 64),
                "sha256": "0" * 64,
            }
        )
        count = len(wire.calls)
        with pytest.raises(ClientError):
            await host.reconcile_upload(missing)
        with pytest.raises(RecoveryRequired):
            await host.ensure_upload()
        assert len(wire.calls) == count
