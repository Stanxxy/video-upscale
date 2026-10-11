"""Explicit new-workflow publication adapter; never called by legacy orchestration.

The durable store pins the SNS topic and message before sending. No background
worker, provider call or resource provisioning is started by importing this module.
"""

import asyncio

from shared_lib.infrastructure.annotation_store import AnnotationNotFound


class ResultPublisher:
    def __init__(self, outbox, sns_client):
        self.outbox, self.sns = outbox, sns_client

    async def publish(self, run_id, component):
        saved = await self.outbox.read(run_id, component)
        if saved is None:
            raise AnnotationNotFound("result publication intent not found")
        record, published = saved
        if published is not None:
            return published
        response = await asyncio.to_thread(
            self.sns.publish,
            TopicArn=record.topic_arn,
            Message=record.message.model_dump_json(),
            MessageAttributes={
                "event_type": {
                    "DataType": "String",
                    "StringValue": record.message.payload.message_type,
                }
            },
        )
        # Missing/invalid SNS acknowledgement is an error, never presumed sent.
        return await self.outbox.mark_published(record, response["MessageId"])

    async def publish_page(self, shard, *, after=None, limit=20):
        """One bounded explicit recovery page; restart cursor on the next sweep.

        Errors propagate with pending intent retained. The future supervisor must
        report/quarantine poisoned entries and continue scheduling other pages;
        this helper makes no claim of a continuously running dispatcher.
        """
        keys, cursor = await self.outbox.pending_page(shard, after=after, limit=limit)
        receipts = []
        for run_id, component in keys:
            receipts.append(await self.publish(run_id, component))
        return tuple(receipts), cursor
