"""Isolated new publisher boundary; full durable path is a LocalStack contract test."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from shared_lib.infrastructure.annotation_store import AnnotationNotFound

from service.result_publication import ResultPublisher


@pytest.mark.asyncio
async def test_missing_intent_is_not_sent():
    class Outbox:
        async def read(self, run_id, component):
            return None

    class SNS:
        def publish(self, **kwargs):
            pytest.fail("missing intent sent")

    with pytest.raises(AnnotationNotFound):
        await ResultPublisher(Outbox(), SNS()).publish(uuid4(), "positions")


@pytest.mark.asyncio
async def test_already_sent_returns_original_receipt_without_sns():
    receipt = object()

    class Outbox:
        async def read(self, run_id, component):
            return object(), receipt

    class SNS:
        def publish(self, **kwargs):
            pytest.fail("already sent intent republished")

    assert (
        await ResultPublisher(Outbox(), SNS()).publish(uuid4(), "positions") is receipt
    )


@pytest.mark.asyncio
async def test_bounded_page_passes_cursor_and_stops_on_error():
    ids = uuid4(), uuid4()
    cursor = {"internal": "cursor"}

    class Outbox:
        seen = []

        async def pending_page(self, shard, *, after, limit):
            assert (shard, after, limit) == (2, cursor, 3)
            return ((ids[0], "positions"), (ids[1], "techniques")), cursor

        async def read(self, run_id, component):
            self.seen.append(run_id)
            raise RuntimeError("storage unavailable")

    outbox = Outbox()
    with pytest.raises(RuntimeError, match="storage unavailable"):
        await ResultPublisher(outbox, None).publish_page(2, after=cursor, limit=3)
    assert outbox.seen == [ids[0]]


@pytest.mark.asyncio
async def test_actual_send_uses_exact_persisted_body_topic_and_attribute():
    message = SimpleNamespace(
        payload=SimpleNamespace(message_type="analysis.result.available.v1"),
        model_dump_json=lambda: '{"persisted":"message"}',
    )
    record = SimpleNamespace(
        topic_arn="arn:aws:sns:us-east-1:123456789012:results", message=message
    )

    class Outbox:
        async def read(self, run_id, component):
            return record, None

        async def mark_published(self, saved, message_id):
            assert saved is record and message_id == "sns-id"
            return "acknowledged"

    class SNS:
        def publish(self, **kwargs):
            assert kwargs == {
                "TopicArn": record.topic_arn,
                "Message": message.model_dump_json(),
                "MessageAttributes": {
                    "event_type": {
                        "DataType": "String",
                        "StringValue": message.payload.message_type,
                    }
                },
            }
            return {"MessageId": "sns-id"}

    assert (
        await ResultPublisher(Outbox(), SNS()).publish(uuid4(), "positions")
        == "acknowledged"
    )
