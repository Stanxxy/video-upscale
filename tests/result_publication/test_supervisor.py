import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from shared_lib.infrastructure.annotation_store import encode_item
from shared_lib.models.analysis_timeline import content_sha256

from service.result_delivery_supervisor import ResultDeliverySupervisor


class Spec(BaseModel):
    run_id: UUID


class Run(BaseModel):
    spec: Spec


def boundary():
    run = Run(spec=Spec(run_id=uuid4()))
    record = SimpleNamespace(
        run_sha256=content_sha256(run),
        topic_arn="fixture-topic",
        message=SimpleNamespace(
            model_dump_json=lambda: '{"immutable":true}',
            payload=SimpleNamespace(message_type="analysis.result.available.v1"),
        ),
    )

    class Outbox:
        marker = None

        async def resolve_delivery_run(self, run_id):
            assert run_id == run.spec.run_id
            return run

        async def read(self, run_id, component):
            return record, self.marker

        async def mark_published(self, record, message_id):
            self.marker = message_id
            return message_id

        async def pending_page(self, shard, **kwargs):
            return ((run.spec.run_id, "positions"),), None

        async def pending_candidates_page(self, shard, **kwargs):
            keys, cursor = await self.pending_page(shard, **kwargs)
            return (
                tuple(
                    encode_item({"run_id": str(run_id), "component": component})
                    for run_id, component in keys
                ),
                cursor,
            )

    class SNS:
        calls = 0

        def publish(self, **kwargs):
            self.calls += 1
            return {"MessageId": "fixture-ack"}

    async def authority(candidate):
        assert candidate == run

    outbox, sns = Outbox(), SNS()
    outbox.proposals = outbox
    return ResultDeliverySupervisor(outbox, sns, authority=authority), run, outbox, sns


@pytest.mark.asyncio
async def test_success_and_stale_index_reuse_ack_without_resend():
    supervisor, run, outbox, sns = boundary()
    for _ in range(2):
        page = await supervisor.recover_page(0)
        assert page.attempts[0].status == "acknowledged"
    assert sns.calls == 1 and outbox.marker == "fixture-ack"


@pytest.mark.asyncio
async def test_denied_item_does_not_block_following_item_or_expose_error_text():
    supervisor, run, outbox, sns = boundary()
    bad = uuid4()

    async def page(*args, **kwargs):
        return ((bad, "positions"), (run.spec.run_id, "techniques")), {"next": "page"}

    resolve = outbox.resolve_delivery_run

    async def denied(run_id):
        if run_id == bad:
            raise PermissionError("secret-token https://private.example")
        return await resolve(run_id)

    outbox.pending_page, outbox.resolve_delivery_run = page, denied
    result = await supervisor.recover_page(0)
    assert [r.status for r in result.attempts] == ["unconfirmed", "acknowledged"]
    assert result.attempts[0].error_code == "access_or_record_unavailable"
    assert "secret" not in repr(result) and sns.calls == 1
    assert result.next_cursor == {"next": "page"}


@pytest.mark.asyncio
async def test_query_failure_retains_cursor_and_is_not_empty_success():
    supervisor, _, outbox, _ = boundary()

    async def unavailable(*args, **kwargs):
        raise TimeoutError("database unavailable")

    outbox.pending_page = unavailable
    cursor = {"prior": "position"}
    page = await supervisor.recover_page(3, after=cursor)
    assert page.error_code == "storage_or_delivery_unavailable"
    assert page.attempts == () and page.next_cursor == cursor


@pytest.mark.asyncio
async def test_revoked_authority_never_sends():
    supervisor, run, outbox, sns = boundary()

    async def denied(run):
        raise PermissionError("revoked")

    supervisor.authority = denied
    page = await supervisor.recover_page(0)
    assert page.attempts[0].status == "unconfirmed"
    assert sns.calls == 0 and outbox.marker is None


@pytest.mark.asyncio
async def test_cancellation_before_send_prevents_network_after_authority_returns():
    supervisor, run, _, sns = boundary()
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(run):
        entered.set()
        await release.wait()

    supervisor.authority = slow
    task = asyncio.create_task(supervisor.publish_one(run.spec.run_id, "positions"))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sns.calls == 0


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_blocking_send_and_retains_ack():
    supervisor, run, outbox, sns = boundary()
    entered, release = threading.Event(), threading.Event()

    def blocked(**kwargs):
        entered.set()
        assert release.wait(5)
        return {"MessageId": "saved-after-cancel"}

    sns.publish = blocked
    task = asyncio.create_task(supervisor.publish_one(run.spec.run_id, "positions"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert outbox.marker == "saved-after-cancel"


@pytest.mark.asyncio
async def test_queued_executor_send_cannot_start_after_cancel():
    supervisor, run, outbox, sns = boundary()
    loop = asyncio.get_running_loop()
    previous = loop._default_executor
    pool = ThreadPoolExecutor(max_workers=1)
    occupied, release = threading.Event(), threading.Event()

    def occupy():
        occupied.set()
        assert release.wait(5)

    loop.set_default_executor(pool)
    blocker = loop.run_in_executor(None, occupy)
    task = None
    try:
        while not occupied.is_set():
            await asyncio.sleep(0)
        task = asyncio.create_task(supervisor.publish_one(run.spec.run_id, "positions"))
        for _ in range(1000):
            if pool._work_queue.qsize():
                break
            await asyncio.sleep(0)
        assert pool._work_queue.qsize() == 1
        task.cancel()
        await asyncio.sleep(0)  # supervisor observes cancellation while SNS is queued
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sns.calls == 0 and outbox.marker is None
    finally:
        release.set()
        await blocker
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        loop._default_executor = previous
        pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_stop_mid_page_restarts_cursor_without_skipping_unvisited_keys():
    supervisor, run, outbox, _ = boundary()
    stop = asyncio.Event()

    async def page(*args, **kwargs):
        return ((run.spec.run_id, "positions"), (run.spec.run_id, "techniques")), {
            "end": 1
        }

    mark = outbox.mark_published

    async def stopped(*args):
        result = await mark(*args)
        stop.set()
        return result

    outbox.pending_page, outbox.mark_published = page, stopped
    result = await supervisor.recover_page(0, after={"prior": 1}, stop=stop)
    assert len(result.attempts) == 1 and result.next_cursor == {"prior": 1}


@pytest.mark.asyncio
async def test_loop_advances_past_failed_items_and_resets_at_eof():
    supervisor, _, outbox, _ = boundary()
    stop, seen, reports = asyncio.Event(), [], []

    async def page(shard, *, after, limit):
        if shard == 0:
            seen.append(after)
            return (), {"next": 1} if after is None else None
        return (), None

    async def observe(report):
        reports.append(report)
        if len(reports) == 33:
            stop.set()

    outbox.pending_page = page
    await supervisor.run(stop, observe=observe, interval_seconds=0.001, page_size=1)
    assert seen == [None, {"next": 1}, None]
    assert [r.shard for r in reports[:16]] == list(range(16))


@pytest.mark.asyncio
async def test_observer_failure_stops_loop():
    supervisor, _, outbox, _ = boundary()

    async def empty(*args, **kwargs):
        return (), None

    async def failed(report):
        raise RuntimeError("operator reporting failed")

    outbox.pending_page = empty
    with pytest.raises(RuntimeError, match="operator reporting failed"):
        await supervisor.run(asyncio.Event(), observe=failed, interval_seconds=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("interval", [0, -1, 61, float("nan"), float("inf")])
async def test_invalid_polling_policy_never_queries(interval):
    supervisor, _, _, _ = boundary()
    with pytest.raises(ValueError):
        await supervisor.run(
            asyncio.Event(), observe=lambda page: None, interval_seconds=interval
        )
