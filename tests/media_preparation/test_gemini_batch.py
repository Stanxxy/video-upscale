"""Gemini Batch API delivery (x-aiquota M3.0): one job per stage, durable ledger, same replies."""

import asyncio
import json
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest

from service.analysis_gemini import GeminiBoundaryError, S3GeminiArtifacts
from bjj_analysis_core.execution import ProviderWaiting
from service.analysis_gemini_batch import MAX_WAIT, BatchGeminiProvider, wake_after

from test_gemini import Wire, receipt_value, request_for, uploaded


class BatchWire(Wire):
    """Adds batchGenerateContent, job polling and the results download to the fixture."""

    def __init__(self, media):
        super().__init__(media)
        self.inputs = {}  # input file name -> JSONL bytes
        self.jobs = {}  # job name -> input file name
        self.polls_before_done = 1
        self.job_state = "BATCH_STATE_SUCCEEDED"
        self.item_error = None

    async def __call__(self, request):
        path = request.url.path
        if path == "/upload/v1beta/files" and (
            request.headers.get("x-goog-upload-header-content-type") == "application/jsonl"
            or request.url.params.get("upload_id") == "jsonl"
        ):
            body = await request.aread()
            self.calls.append((request.method, str(request.url), body, dict(request.headers)))
            if request.url.params.get("upload_id") == "jsonl":
                name = f"files/input{len(self.inputs) + 1}"
                self.inputs[name] = body
                return httpx.Response(200, json={"file": {"name": name}})
            return httpx.Response(200, json={}, headers={
                "x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=jsonl"})
        if path.endswith(":batchGenerateContent"):
            body = await request.aread()
            self.calls.append((request.method, str(request.url), body, dict(request.headers)))
            name = f"batches/b{len(self.jobs) + 1}"
            self.jobs[name] = json.loads(body)["batch"]["input_config"]["file_name"]
            return httpx.Response(200, json={"name": name, "metadata": {"state": "BATCH_STATE_PENDING"}})
        if path.startswith("/v1beta/batches/"):
            self.calls.append((request.method, str(request.url), b"", dict(request.headers)))
            job = path.removeprefix("/v1beta/")
            if self.polls_before_done > 0:
                self.polls_before_done -= 1
                return httpx.Response(200, json={"name": job, "metadata": {"state": "BATCH_STATE_RUNNING"}})
            return httpx.Response(200, json={"name": job, "metadata": {
                "state": self.job_state, "output": {"responsesFile": f"files/out-{job.split('/')[-1]}"}}})
        if path.startswith("/download/v1beta/files/out-"):
            self.calls.append((request.method, str(request.url), b"", dict(request.headers)))
            job = "batches/" + path.split("out-")[1].split(":")[0]
            lines = []
            for line in self.inputs[self.jobs[job]].decode().splitlines():
                key = json.loads(line)["key"]
                if self.item_error:
                    lines.append(json.dumps({"key": key, "error": self.item_error}))
                else:
                    lines.append(json.dumps({"key": key, "response": self.result}))
            return httpx.Response(200, text="\n".join(lines) + "\n")
        return await super().__call__(request)

    def batch_creates(self):
        return [c for c in self.calls if ":batchGenerateContent" in c[1]]


@asynccontextmanager
async def batch_case(resolver, inputs, s3, wire=None):
    async with resolver.open(inputs[0]) as media:
        wire = wire or BatchWire(media)
        artifacts = S3GeminiArtifacts(media.run, s3[0], bucket=s3[1])
        provider = BatchGeminiProvider(
            media,
            artifacts,
            api_key="fixture-secret",
            qualified_models={"fixture-model": {"fixture-model-r1"}},
            guard=resolver.guard,
            transport=httpx.MockTransport(wire),
            settle_seconds=0.05,
        )
        try:
            yield provider, wire, media
        finally:
            await provider.aclose()


def ids(n):
    return [(uuid4(), uuid4()) for _ in range(n)]


async def answered(provider, run, request, execution_id, call_id):
    """Ask again after each ProviderWaiting, as the scheduler does when the run wakes."""
    for _ in range(20):
        try:
            return await provider.generate(run, request, execution_id, call_id)
        except ProviderWaiting:
            continue
    raise AssertionError("batch never answered")


@pytest.mark.asyncio
async def test_a_stage_is_one_batch_job_with_the_interactive_reply(resolver, inputs, s3):
    async with batch_case(resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        request = request_for(media)
        steps = ids(3)
        replies = await asyncio.gather(*(
            answered(provider, media.run, request, execution_id, call_id) for execution_id, call_id in steps
        ))
        assert len(wire.batch_creates()) == 1
        assert wire.generates() == []  # no interactive generateContent
        submitted = [json.loads(line) for line in next(iter(wire.inputs.values())).decode().splitlines()]
        assert sorted(item["key"] for item in submitted) == sorted(str(e) for e, _ in steps)
        assert all(item["request"] == provider._request_body(request) for item in submitted)
        for (execution_id, call_id), reply in zip(steps, replies, strict=True):
            assert reply.text == '{"schema_version":1,"events":[]}'
            assert reply.usage.call_id == call_id and reply.usage.execution_id == execution_id
            assert reply.usage.provider == "google-gemini-batch-v1"
            assert (reply.usage.input_tokens, reply.usage.thinking_tokens) == (101, 25)
            receipt = receipt_value(s3, reply.usage.raw_usage)
            assert receipt["context"]["delivery"] == "batch"
            assert receipt["context"]["batch"] == "batches/b1"


@pytest.mark.asyncio
async def test_a_resumed_step_waits_for_its_recorded_job_and_never_resubmits(resolver, inputs, s3):
    async with batch_case(resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        request = request_for(media)
        execution_id = uuid4()
        wire.polls_before_done = 10**6  # the job is still running when the engine dies
        with pytest.raises(ProviderWaiting):
            await provider.generate(media.run, request, execution_id, uuid4())
        wire.polls_before_done = 1
        # A fresh provider (engine restart) with a new attempt id for the same step.
        async with batch_case(resolver, inputs, s3, wire=wire) as (resumed, _, _):
            await uploaded(resumed)
            reply = await answered(resumed, media.run, request, execution_id, call_id := uuid4())
        assert len(wire.batch_creates()) == 1
        assert reply.usage.call_id == call_id


@pytest.mark.asyncio
async def test_a_reask_after_a_delivered_answer_is_a_new_job(resolver, inputs, s3):
    async with batch_case(resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        request = request_for(media)
        execution_id = uuid4()
        await answered(provider, media.run, request, execution_id, uuid4())
        await answered(provider, media.run, request, execution_id, uuid4())  # invalid answer: ask again
        assert len(wire.batch_creates()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["job", "item"])
async def test_failed_jobs_and_items_fail_the_step(resolver, inputs, s3, failure):
    async with batch_case(resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        if failure == "job":
            wire.job_state = "BATCH_STATE_FAILED"
        else:
            wire.item_error = {"code": 400, "message": "fixture item failure"}
        with pytest.raises(GeminiBoundaryError):
            await answered(provider, media.run, request_for(media), uuid4(), uuid4())
        assert wire.generates() == []


@pytest.mark.asyncio
async def test_a_pending_job_parks_the_step_with_a_backoff_and_no_second_submission(resolver, inputs, s3):
    async with batch_case(resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        request, execution_id, call_id = request_for(media), uuid4(), uuid4()
        wire.polls_before_done = 3
        for _ in range(3):
            before = provider.now()
            with pytest.raises(ProviderWaiting) as waiting:
                await provider.generate(media.run, request, execution_id, call_id)
            assert waiting.value.wake_at - before >= wake_after(provider.now() - before)
        reply = await provider.generate(media.run, request, execution_id, call_id)
        assert reply.usage.call_id == call_id
        assert len(wire.batch_creates()) == 1
        assert sum(1 for c in wire.calls if "/v1beta/batches/" in c[1]) == 4  # one check per ask


@pytest.mark.asyncio
async def test_an_unanswered_job_fails_the_step_after_the_cutoff(resolver, inputs, s3):
    async with batch_case(resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        request, execution_id = request_for(media), uuid4()
        wire.polls_before_done = 10**6
        with pytest.raises(ProviderWaiting):
            await provider.generate(media.run, request, execution_id, uuid4())
        later = provider.now() + MAX_WAIT + wake_after(MAX_WAIT)
        provider.clock = lambda: later
        with pytest.raises(GeminiBoundaryError, match="unanswered"):
            await provider.generate(media.run, request, execution_id, uuid4())
        assert len(wire.batch_creates()) == 1


def test_checks_back_off_as_a_job_ages():
    from datetime import timedelta as td
    waits = [wake_after(td(minutes=m)) for m in (0, 4, 6, 20, 120, 2000)]
    assert waits == sorted(waits) and waits[0] == td(minutes=1) and waits[-1] == td(minutes=10)
