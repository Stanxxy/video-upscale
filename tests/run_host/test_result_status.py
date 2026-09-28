import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from shared_lib.infrastructure.annotation_store import (
    AnnotationIntegrityError,
    AnnotationNotFound,
    encode_item,
)
from shared_lib.models.analysis_availability import ProposalReadinessV1
from shared_lib.models.analysis_timeline import content_sha256
from test_delivery import delivery_queue, receive
from test_dispatch import dispatch_tables, setup_dispatch
from test_worker import leases

from service.result_delivery_supervisor import ResultDeliverySupervisor


def lanes(view):
    return {item.component: item for item in view.components}


async def ingest(dispatch, run, component, *, stage=False):
    publication, _ = await dispatch.outbox.read(run.spec.run_id, component)
    store = dispatch.outbox.components
    receipt = await store.ingest(
        run.spec.analysis_id, publication.message, actor=run.spec.owner
    )
    if stage:
        await store.stage_proposals(
            run.spec.analysis_id, run.spec.run_id, component, actor=run.spec.owner
        )
    return receipt


@pytest.mark.asyncio
async def test_unstarted_read_is_empty_without_claims_or_writes(setup_dispatch):
    dispatch, run, wire, _, annotations, _ = setup_dispatch
    before = await annotations.read(run.spec.analysis_id, actor=run.spec.owner)
    view = await dispatch.result_status(run.spec.run_id)
    assert view.run_id == run.spec.run_id
    assert tuple(lanes(view)) == ("positions", "techniques")
    assert all(
        c.publication_state == "not_prepared"
        and c.declared_item_count is None
        and c.review.state == "not_received"
        for c in view.components
    )
    assert await annotations.read(run.spec.analysis_id, actor=run.spec.owner) == before
    assert wire.calls == [] and list(dispatch.scratch_root.iterdir()) == []
    assert (
        dispatch.session.execute(
            f"SELECT * FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
            [str(run.spec.run_id)],
        ).one()
        is None
    )


@pytest.mark.asyncio
async def test_real_delivery_intake_and_staging_are_separate_with_explicit_zero(
    setup_dispatch, delivery_queue
):
    dispatch, run, wire, components, annotations, _ = setup_dispatch
    sns, sqs, topic, queue = delivery_queue
    dispatch.outbox.topic_arn = topic
    await dispatch.dispatch(run.spec.run_id)
    count = len(wire.calls)
    prepared = lanes(await dispatch.result_status(run.spec.run_id))
    assert [c.declared_item_count for c in prepared.values()] == [1, 0]
    assert all(
        c.publication_state == "prepared" and c.review.state == "not_received"
        for c in prepared.values()
    )
    sender = ResultDeliverySupervisor(
        dispatch.outbox, sns, authority=dispatch.authority
    )
    await sender.recover_page(run.spec.run_id.int % 16)
    acknowledged = await dispatch.result_status(run.spec.run_id)
    assert all(
        c.publication_state == "acknowledged"
        and c.published_at is not None
        and c.review.state == "not_received"
        for c in acknowledged.components
    )
    messages = await receive(sqs, queue, 2)
    for message in messages:
        await components.ingest(run.spec.analysis_id, message, actor=run.spec.owner)
    received = await dispatch.result_status(run.spec.run_id)
    assert all(
        c.review.state == "received" and c.review.proposal_count is None
        for c in received.components
    )
    await components.stage_proposals(
        run.spec.analysis_id, run.spec.run_id, "positions", actor=run.spec.owner
    )
    partial = lanes(await dispatch.result_status(run.spec.run_id))
    assert partial["positions"].review.state == "ready"
    assert partial["techniques"].review.state == "received"
    await components.stage_proposals(
        run.spec.analysis_id, run.spec.run_id, "techniques", actor=run.spec.owner
    )
    ready = await dispatch.result_status(run.spec.run_id)
    assert all(
        c.review.state == "ready" and c.review.ready_at is not None
        for c in ready.components
    )
    assert lanes(ready)["techniques"].review.proposal_count == 0
    assert (
        await annotations.read(run.spec.analysis_id, actor=run.spec.owner)
    ).metadata.edit_version == 0
    assert len(wire.calls) == count and list(dispatch.scratch_root.iterdir()) == []
    rendered = str(asdict(ready))
    assert not any(
        secret in rendered
        for secret in (
            "artifact",
            "bucket",
            "fixture-secret",
            "system_prompt",
            "sns_message_id",
            "position-attempt:",
        )
    )


@pytest.mark.asyncio
async def test_positions_can_be_review_ready_while_technique_scan_is_running(
    setup_dispatch,
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    original = dispatch.transport
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        if ":generateContent" in str(request.url):
            payload = json.loads(await request.aread())
            if (
                "events"
                in payload["generationConfig"]["responseJsonSchema"]["properties"]
            ):
                entered.set()
                await release.wait()
        return await original.handle_async_request(request)

    dispatch.transport = httpx.MockTransport(handler)
    task = asyncio.create_task(dispatch.dispatch(run.spec.run_id))
    waiting = asyncio.create_task(entered.wait())
    try:
        done, _ = await asyncio.wait(
            (task, waiting), timeout=30, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            await task
        assert waiting in done
        await ingest(dispatch, run, "positions", stage=True)
        view = lanes(await dispatch.result_status(run.spec.run_id))
        assert view["positions"].review.state == "ready"
        assert view["techniques"].publication_state == "not_prepared"
        assert view["techniques"].review.state == "not_received"
        assert not task.done()
        release.set()
        await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        waiting.cancel()
        await asyncio.gather(task, waiting, return_exceptions=True)


@pytest.mark.asyncio
async def test_lost_send_ack_does_not_hide_real_consumer_readiness(
    setup_dispatch, delivery_queue
):
    dispatch, run, wire, components, _, _ = setup_dispatch
    sns, sqs, topic, queue = delivery_queue
    dispatch.outbox.topic_arn = topic
    await dispatch.dispatch(run.spec.run_id)
    count = len(wire.calls)

    class LostAck:
        def publish(self, **kwargs):
            sns.publish(**kwargs)
            raise TimeoutError("accepted but acknowledgement lost")

    sender = ResultDeliverySupervisor(
        dispatch.outbox, LostAck(), authority=dispatch.authority
    )
    with pytest.raises(TimeoutError):
        await sender.publish_one(run.spec.run_id, "positions")
    (message,) = await receive(sqs, queue, 1)
    await components.ingest(run.spec.analysis_id, message, actor=run.spec.owner)
    await components.stage_proposals(
        run.spec.analysis_id, run.spec.run_id, "positions", actor=run.spec.owner
    )
    view = lanes(await dispatch.result_status(run.spec.run_id))["positions"]
    assert view.publication_state == "prepared" and view.published_at is None
    assert view.review.state == "ready"
    assert len(wire.calls) == count


@pytest.mark.asyncio
async def test_concurrent_intake_and_readiness_share_the_earlier_consumer_snapshot(
    setup_dispatch, monkeypatch
):
    dispatch, run, _, components, _, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    original, armed = components.receipt, True

    async def racing(*args, **kwargs):
        nonlocal armed
        if armed and kwargs.get("_row") is not None:
            armed = False
            await ingest(dispatch, run, "positions", stage=True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(components, "receipt", racing)
    first = lanes(await dispatch.result_status(run.spec.run_id))
    assert first["positions"].review.state == "not_received"
    second = lanes(await dispatch.result_status(run.spec.run_id))
    assert second["positions"].review.state == "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["artifact", "readiness_hash", "receipt_digest", "missing_publication"]
)
async def test_corrupt_result_evidence_fails_closed(setup_dispatch, fault):
    dispatch, run, _, components, annotations, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    receipt = await ingest(dispatch, run, "positions", stage=fault == "readiness_hash")
    publication, _ = await dispatch.outbox.read(run.spec.run_id, "positions")
    if fault == "artifact":
        ref = publication.message.payload.artifact
        dispatch.s3.put_object(Bucket=ref.bucket, Key=ref.key, Body=b"corrupt")
    elif fault == "missing_publication":
        annotations.client.delete_item(
            TableName=dispatch.outbox.table.name,
            Key=encode_item({"run_id": str(run.spec.run_id), "component": "positions"}),
        )
    else:
        row = await annotations._get(
            dispatch.proposals.tables.runs, {"run_id": str(run.spec.run_id)}
        )
        if fault == "readiness_hash":
            row["positions_readiness_hash"] = "0" * 64
        else:
            receipt = receipt.model_copy(update={"result_sha256": "0" * 64})
            row["positions_result"] = receipt.model_dump_json()
            row["positions_result_hash"] = content_sha256(receipt)
        annotations.client.put_item(
            TableName=dispatch.proposals.tables.runs, Item=encode_item(row)
        )
    with pytest.raises(AnnotationIntegrityError):
        await dispatch.result_status(run.spec.run_id)


@pytest.mark.asyncio
async def test_revocation_after_last_publication_read_returns_no_partial_status(
    setup_dispatch, monkeypatch
):
    dispatch, run, _, _, _, revoke = setup_dispatch
    original = dispatch.outbox.read

    async def revoked(run_id, lane):
        result = await original(run_id, lane)
        if lane == "techniques":
            revoke()
        return result

    monkeypatch.setattr(dispatch.outbox, "read", revoked)
    with pytest.raises(PermissionError, match="revoked"):
        await dispatch.result_status(run.spec.run_id)


@pytest.mark.asyncio
async def test_unknown_run_never_looks_like_missing_results(setup_dispatch):
    dispatch, _, wire, _, _, _ = setup_dispatch
    with pytest.raises(AnnotationNotFound):
        await dispatch.result_status(uuid4())
    assert wire.calls == []


@pytest.mark.asyncio
async def test_new_publication_after_consumer_snapshot_is_not_false_corruption(
    setup_dispatch, monkeypatch
):
    dispatch, run, _, components, _, _ = setup_dispatch
    original, armed = components.receipt, True

    async def racing(*args, **kwargs):
        nonlocal armed
        if armed and kwargs.get("_row") is not None:
            armed = False
            await dispatch.dispatch(run.spec.run_id)
        return await original(*args, **kwargs)

    monkeypatch.setattr(components, "receipt", racing)
    view = await dispatch.result_status(run.spec.run_id)
    assert all(
        c.publication_state == "prepared" and c.review.state == "not_received"
        for c in view.components
    )


@pytest.mark.asyncio
async def test_future_readiness_certificate_is_not_advertised(setup_dispatch):
    dispatch, run, _, _, annotations, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    await ingest(dispatch, run, "positions", stage=True)
    row = await annotations._get(
        dispatch.proposals.tables.runs, {"run_id": str(run.spec.run_id)}
    )
    marker = ProposalReadinessV1.model_validate_json(row["positions_readiness"])
    marker = marker.model_copy(
        update={"ready_at": datetime.now(timezone.utc) + timedelta(hours=1)}
    )
    row["positions_readiness"] = marker.model_dump_json()
    row["positions_readiness_hash"] = content_sha256(marker)
    annotations.client.put_item(
        TableName=dispatch.proposals.tables.runs, Item=encode_item(row)
    )
    with pytest.raises(AnnotationIntegrityError, match="timestamp is in the future"):
        await dispatch.result_status(run.spec.run_id)


@pytest.mark.asyncio
async def test_review_decisions_do_not_reinterpret_original_ready_count(setup_dispatch):
    dispatch, run, wire, _, annotations, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    await ingest(dispatch, run, "positions", stage=True)
    before = lanes(await dispatch.result_status(run.spec.run_id))["positions"]
    proposals = dispatch.proposals
    page = await proposals.list_page(
        run.spec.analysis_id, "position", actor=run.spec.owner
    )
    assert len(page.items) == 1
    decision = await proposals.resolve(
        run.spec.analysis_id,
        "position",
        page.items[0].proposal_id,
        {"change_id": uuid4(), "action": "rejected"},
        actor=run.spec.owner,
    )
    assert decision.proposal.resolution.action == "rejected"
    count = len(wire.calls)
    after = lanes(await dispatch.result_status(run.spec.run_id))["positions"]
    assert after == before and after.review.proposal_count == 1
    assert (
        await annotations.read(run.spec.analysis_id, actor=run.spec.owner)
    ).metadata.edit_version == 0
    assert len(wire.calls) == count
