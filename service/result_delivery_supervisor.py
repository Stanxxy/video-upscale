"""Opt-in recovery of already durable result intents; never invokes inference.

Reports are internal operational values, not a new public or persisted schema.
Pending DynamoDB records remain the durable inventory of unacknowledged intents.
"""

import asyncio
import math
import threading
from dataclasses import dataclass
from uuid import UUID

from shared_lib.infrastructure.annotation_store import (
    AnnotationIntegrityError,
    AnnotationNotFound,
    AnnotationStoreUnavailable,
    decode_item,
)
from shared_lib.models.analysis_timeline import content_sha256

from service.result_publication import ResultPublisher


@dataclass(frozen=True)
class DeliveryAttempt:
    run_id: UUID | None
    component: str | None
    status: str
    error_code: str | None = None


@dataclass(frozen=True)
class DeliveryPage:
    shard: int
    attempts: tuple[DeliveryAttempt, ...]
    next_cursor: dict | None
    error_code: str | None = None


def _error_code(error):
    # Never put exception text (possibly containing URLs/credentials) in reports.
    if isinstance(error, (PermissionError, AnnotationNotFound)):
        return "access_or_record_unavailable"
    if isinstance(error, (AnnotationIntegrityError, ValueError)):
        return "invalid_record"
    if isinstance(error, (AnnotationStoreUnavailable, TimeoutError)):
        return "storage_or_delivery_unavailable"
    return "delivery_error"


async def _drain(task):
    # A blocking SNS thread cannot be cancelled. Keep its enclosing publication
    # task alive to retain an acknowledgment where authority still permits it.
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not task.cancelled():
        task.exception()


class _AuthorizedOutbox:
    def __init__(self, outbox, run, authority, cancelled):
        self.outbox, self.run, self.authority = outbox, run, authority
        self.cancelled = cancelled

    async def _guard(self):
        current = await self.outbox.proposals.resolve_delivery_run(self.run.spec.run_id)
        if current != self.run:
            raise AnnotationIntegrityError("registered delivery run changed")
        await self.authority(current)

    async def read(self, run_id, component):
        if self.cancelled.is_set():
            raise asyncio.CancelledError
        if run_id != self.run.spec.run_id:
            raise AnnotationIntegrityError("delivery run mismatch")
        await self._guard()
        saved = await self.outbox.read(run_id, component)
        if saved is not None and saved[0].run_sha256 != content_sha256(self.run):
            raise AnnotationIntegrityError("delivery intent differs from run")
        await self._guard()
        if self.cancelled.is_set():
            raise asyncio.CancelledError
        return saved

    async def mark_published(self, record, message_id):
        await self._guard()
        result = await self.outbox.mark_published(record, message_id)
        await self._guard()
        return result


class _CancellableSNS:
    def __init__(self, client, cancelled):
        self.client, self.cancelled = client, cancelled

    def publish(self, **kwargs):
        # Runs in the executor thread, not before submission: queued work must
        # not start a network send after cancellation has already been observed.
        if self.cancelled.is_set():
            raise asyncio.CancelledError
        return self.client.publish(**kwargs)


class ResultDeliverySupervisor:
    def __init__(self, outbox, sns_client, *, authority):
        if not callable(authority):
            raise ValueError("live delivery authority required")
        self.outbox, self.sns, self.authority = outbox, sns_client, authority

    async def publish_one(self, run_id, component):
        """One explicit send/reuse; never retries within this operation."""
        run_id = UUID(str(run_id))
        if component not in {"positions", "techniques"}:
            raise ValueError("unknown component")
        run = await self.outbox.proposals.resolve_delivery_run(run_id)
        cancelled = threading.Event()
        guarded = _AuthorizedOutbox(self.outbox, run, self.authority, cancelled)
        task = asyncio.create_task(
            ResultPublisher(guarded, _CancellableSNS(self.sns, cancelled)).publish(
                run_id, component
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            await _drain(task)
            raise

    async def recover_page(self, shard, *, after=None, limit=20, stop=None):
        """Continue after item failures; page-read failures retain their cursor."""
        if (
            type(shard) is not int
            or not 0 <= shard < 16
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise ValueError("bounded shard and page size required")
        try:
            keys, cursor = await self.outbox.pending_candidates_page(
                shard, after=after, limit=limit
            )
        except Exception as error:
            return DeliveryPage(shard, (), after, _error_code(error))
        attempts = []
        for candidate in keys:
            if stop is not None and stop.is_set():
                # The page was not exhausted. Restart it rather than skip unsent keys.
                return DeliveryPage(shard, tuple(attempts), after)
            run_id, component = None, None
            try:
                row = decode_item(candidate)
                run_id = UUID(row["run_id"])
                if row["component"] not in {"positions", "techniques"}:
                    raise ValueError("invalid component index key")
                component = row["component"]
            except (ValueError, TypeError, KeyError, AttributeError):
                attempts.append(
                    DeliveryAttempt(
                        run_id, component, "unconfirmed", "invalid_index_entry"
                    )
                )
                continue
            try:
                await self.publish_one(run_id, component)
            except Exception as error:
                attempts.append(
                    DeliveryAttempt(
                        run_id, component, "unconfirmed", _error_code(error)
                    )
                )
            else:
                attempts.append(DeliveryAttempt(run_id, component, "acknowledged"))
        return DeliveryPage(shard, tuple(attempts), cursor)

    async def run(self, stop, *, observe, interval_seconds, page_size=20):
        """Explicit lifecycle: one page per shard per pass, then paced polling.

        Cursors advance across passes, including past failed entries. At EOF they
        reset so older failures/new earlier inserts are reconsidered. Restarting
        this supervisor also starts from the beginning of the durable index.
        """
        if (
            not callable(observe)
            or not math.isfinite(interval_seconds)
            or not 0 < interval_seconds <= 60
            or type(page_size) is not int
            or not 1 <= page_size <= 100
        ):
            raise ValueError(
                "observer, bounded polling interval and page size required"
            )
        cursors = {shard: None for shard in range(16)}
        while not stop.is_set():
            for shard in range(16):
                if stop.is_set():
                    return
                page = await self.recover_page(
                    shard, after=cursors[shard], limit=page_size, stop=stop
                )
                await observe(page)  # observer failure is fatal, not an invisible stall
                cursors[shard] = page.next_cursor
            try:
                await asyncio.wait_for(stop.wait(), interval_seconds)
            except TimeoutError:
                pass
