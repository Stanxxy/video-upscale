import asyncio
import json
import os
from uuid import uuid4

import boto3
import pytest
from botocore.config import Config
from shared_lib.infrastructure.annotation_store import encode_item
from shared_lib.models.analysis_messages import AnalysisMessageV1
from test_dispatch import dispatch_tables, setup_dispatch
from test_worker import leases

from service.result_delivery_supervisor import ResultDeliverySupervisor


@pytest.fixture
def delivery_queue(s3):
    kwargs = dict(
        endpoint_url=os.environ["ANNOTATION_TEST_ENDPOINT"],
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 0}),
    )
    sns, sqs = boto3.client("sns", **kwargs), boto3.client("sqs", **kwargs)
    topic = sns.create_topic(Name="s39-delivery-" + uuid4().hex)["TopicArn"]
    queue = None
    try:
        queue = sqs.create_queue(QueueName="s39-delivery-" + uuid4().hex)["QueueUrl"]
        arn = sqs.get_queue_attributes(QueueUrl=queue, AttributeNames=["QueueArn"])[
            "Attributes"
        ]["QueueArn"]
        sns.subscribe(TopicArn=topic, Protocol="sqs", Endpoint=arn)
        yield sns, sqs, topic, queue
    finally:
        if queue:
            sqs.delete_queue(QueueUrl=queue)
        sns.delete_topic(TopicArn=topic)
        sns.close()
        sqs.close()


async def receive(sqs, queue, count):
    result = []
    for _ in range(50):
        page = await asyncio.to_thread(
            sqs.receive_message, QueueUrl=queue, MaxNumberOfMessages=10
        )
        for raw in page.get("Messages", []):
            result.append(
                AnalysisMessageV1.model_validate_json(
                    json.loads(raw["Body"])["Message"]
                )
            )
            await asyncio.to_thread(
                sqs.delete_message, QueueUrl=queue, ReceiptHandle=raw["ReceiptHandle"]
            )
        if len(result) >= count:
            break
        await asyncio.sleep(0.05)
    assert len(result) == count
    return result


@pytest.mark.asyncio
async def test_dag_pending_index_to_real_sns_sqs_and_idempotent_intake(
    setup_dispatch, delivery_queue
):
    dispatch, run, wire, components, _, _ = setup_dispatch
    sns, sqs, topic, queue = delivery_queue
    dispatch.outbox.topic_arn = topic
    await dispatch.dispatch(run.spec.run_id)
    count = len(wire.calls)
    supervisor = ResultDeliverySupervisor(
        dispatch.outbox, sns, authority=dispatch.authority
    )
    report = await supervisor.recover_page(run.spec.run_id.int % 16)
    assert len(report.attempts) == 2
    assert all(a.status == "acknowledged" for a in report.attempts)
    messages = await receive(sqs, queue, 2)
    for message in messages:
        await components.ingest(run.spec.analysis_id, message, actor=run.spec.owner)
        await components.ingest(run.spec.analysis_id, message, actor=run.spec.owner)
    assert {m.payload.component for m in messages} == {"positions", "techniques"}
    assert len(wire.calls) == count
    assert (await supervisor.recover_page(run.spec.run_id.int % 16)).attempts == ()


@pytest.mark.asyncio
async def test_lost_sns_ack_does_not_block_other_result_and_redelivers_exact_envelope(
    setup_dispatch, delivery_queue
):
    dispatch, run, wire, components, _, _ = setup_dispatch
    sns, sqs, topic, queue = delivery_queue
    dispatch.outbox.topic_arn = topic
    await dispatch.dispatch(run.spec.run_id)
    count = len(wire.calls)

    class LostAck:
        calls = 0

        def publish(self, **kwargs):
            result = sns.publish(**kwargs)
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("SNS accepted; acknowledgment lost")
            return result

    sender = LostAck()
    supervisor = ResultDeliverySupervisor(
        dispatch.outbox, sender, authority=dispatch.authority
    )
    report = await supervisor.recover_page(run.spec.run_id.int % 16)
    assert [a.status for a in report.attempts] == ["unconfirmed", "acknowledged"]
    retry = await supervisor.recover_page(run.spec.run_id.int % 16)
    assert all(a.status == "acknowledged" for a in retry.attempts)
    messages = await receive(sqs, queue, 3)
    first_id = messages[0].message_id
    repeated = [m for m in messages if m.message_id == first_id]
    assert len(repeated) == 2 and repeated[0] == repeated[1]
    for message in messages:
        await components.ingest(run.spec.analysis_id, message, actor=run.spec.owner)
    assert len(wire.calls) == count and sender.calls == 3


@pytest.mark.asyncio
async def test_corrupt_index_key_does_not_starve_valid_neighbors_or_cursor(
    setup_dispatch, delivery_queue
):
    dispatch, run, _, _, _, _ = setup_dispatch
    sns, _, topic, _ = delivery_queue
    dispatch.outbox.topic_arn = topic
    await dispatch.dispatch(run.spec.run_id)
    shard = run.spec.run_id.int % 16
    dispatch.outbox.annotations.client.put_item(
        TableName=dispatch.outbox.table.name,
        Item=encode_item(
            {
                "run_id": "invalid-uuid",
                "component": "positions",
                "pending_shard": f"pending#{shard:02d}",
                "created_order": "0000-corrupt",
            }
        ),
    )
    supervisor = ResultDeliverySupervisor(
        dispatch.outbox, sns, authority=dispatch.authority
    )
    page = await supervisor.recover_page(shard, limit=2)
    assert [a.status for a in page.attempts] == ["unconfirmed", "acknowledged"]
    assert page.attempts[0].error_code == "invalid_index_entry"
    assert page.next_cursor is not None
    following = await supervisor.recover_page(shard, after=page.next_cursor, limit=2)
    assert (
        len(following.attempts) == 1 and following.attempts[0].status == "acknowledged"
    )
    assert following.next_cursor is None
