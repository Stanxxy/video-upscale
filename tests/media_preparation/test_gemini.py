import asyncio
import base64
import hashlib
import importlib.util
import json
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from bjj_analysis_core.execution import RecoveryRequired, StepRunner
from bjj_analysis_core.requests import prepare_call
from shared_lib.models.analysis_run_record import RegisteredAIRunV1


class Wire:
    """Provider HTTP fixture only; media verification/artifact S3 remain real."""

    def __init__(self, media):
        self.calls = []
        self.media = media
        self.upload_url = "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=test"
        self.file = dict(
            name="files/fixture",
            uri="https://generativelanguage.googleapis.com/v1beta/files/fixture",
            mimeType="video/mp4",
            sizeBytes=str(media.source_size),
            sha256Hash=base64.b64encode(
                media.run.spec.source.sha256.encode("ascii")
            ).decode(),
            state="ACTIVE",
            expirationTime=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        )
        self.result = dict(
            modelVersion="fixture-model-r1",
            responseId="provider-response-1",
            usageMetadata=dict(
                promptTokenCount=101,
                candidatesTokenCount=9,
                thoughtsTokenCount=25,
                cachedContentTokenCount=0,
                promptTokensDetails=[{"modality": "VIDEO", "tokenCount": 90}],
            ),
            candidates=[
                dict(
                    finishReason="STOP",
                    content=dict(parts=[{"text": '{"schema_version":1,"events":[]}'}]),
                )
            ],
        )
        self.status = 200
        self.error = None
        self.after_generate = None

    async def __call__(self, request):
        body = await request.aread()
        self.calls.append(
            (request.method, str(request.url), body, dict(request.headers))
        )
        assert request.headers["x-goog-api-key"] == "fixture-secret"
        if request.url.path == "/upload/v1beta/files":
            if request.url.query:
                assert (
                    hashlib.sha256(body).hexdigest()
                    == self.media.run.spec.source.sha256
                )
                return httpx.Response(200, json={"file": self.file})
            return httpx.Response(
                200, json={}, headers={"x-goog-upload-url": self.upload_url}
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if request.method == "GET":
            return httpx.Response(200, json=self.file)
        if self.error:
            raise self.error
        if self.after_generate:
            self.after_generate()
        return httpx.Response(self.status, json=self.result)

    def generates(self):
        return [c for c in self.calls if ":generateContent" in c[1]]


@asynccontextmanager
async def provider_case(gemini, resolver, inputs, s3):
    async with resolver.open(inputs[0]) as media:
        wire = Wire(media)
        artifacts = gemini.S3GeminiArtifacts(media.run, s3[0], bucket=s3[1])
        provider = gemini.GeminiProvider(
            media,
            artifacts,
            api_key="fixture-secret",
            qualified_models={"fixture-model": {"fixture-model-r1"}},
            guard=resolver.guard,
            transport=httpx.MockTransport(wire),
        )
        try:
            yield provider, wire, media
        finally:
            await provider.aclose()


def request_for(media, node="technique_scan"):
    return prepare_call(
        media.run,
        media.config_bytes,
        node,
        input_window=media.run.spec.scope,
        context={},
    )


async def uploaded(provider):
    return await provider.upload(uuid4(), mime_type="video/mp4", max_polls=0)


def receipt_value(s3, ref):
    client, _ = s3
    response = client.get_object(Bucket=ref.bucket, Key=ref.key)
    with response["Body"] as stream:
        data = stream.read()
    assert hashlib.sha256(data).hexdigest() == ref.sha256
    assert response["ServerSideEncryption"] == "AES256"
    return json.loads(data)


@pytest.mark.asyncio
async def test_native_payload_upload_usage_and_private_raw_receipt(
    gemini, resolver, inputs, s3
):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        upload = await uploaded(provider)
        assert upload.run_id == media.run.spec.run_id
        assert receipt_value(s3, upload.receipt)["kind"] == "upload_finalize"
        request = request_for(media)
        execution, call = uuid4(), uuid4()
        reply = await provider.generate(media.run, request, execution, call)
        assert reply.text == '{"schema_version":1,"events":[]}'
        assert reply.usage.execution_id == execution and reply.usage.call_id == call
        assert reply.usage.provider_request_id == "provider-response-1"
        assert reply.usage.input_tokens == 101 and reply.usage.thinking_tokens == 25
        assert (
            reply.usage.usage_status == "complete"
            and reply.usage.cost_status == "unknown"
        )
        assert reply.usage.cost_nano_usd is None
        payload = json.loads(wire.generates()[0][2])
        parts = payload["contents"][0]["parts"]
        assert json.loads(parts[0]["text"])["reference_player_id"] == "z"
        assert json.loads(parts[2]["text"])["reference_player_id"] == "a"
        assert base64.b64decode(parts[1]["inlineData"]["data"]) == media.crops[0].data
        assert base64.b64decode(parts[3]["inlineData"]["data"]) == media.crops[1].data
        assert parts[4]["videoMetadata"] == dict(
            startOffset="0.500s", endOffset="2.500s", fps=10
        )
        assert parts[5]["text"] == request.initial_prompt
        assert payload["systemInstruction"]["parts"][0]["text"] == request.system_prompt
        generation = payload["generationConfig"]
        assert generation["responseJsonSchema"] == request.response_schema
        assert generation["thinkingConfig"] == dict(
            thinkingLevel="HIGH", includeThoughts=False
        )
        assert generation["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"
        raw = receipt_value(s3, reply.usage.raw_usage)
        assert json.loads(base64.b64decode(raw["body_base64"])) == wire.result
        assert "fixture-secret" not in json.dumps(raw)
        assert raw["context"]["execution_id"] == str(execution)
        assert len(wire.generates()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["503", "timeout", "redirect"])
async def test_generation_is_one_http_attempt(gemini, resolver, inputs, s3, case):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        if case == "timeout":
            wire.error = httpx.ReadTimeout("fixture lost response")
        else:
            wire.status = 503 if case == "503" else 302
        with pytest.raises((gemini.GeminiBoundaryError, httpx.ReadTimeout)):
            await provider.generate(media.run, request_for(media), uuid4(), uuid4())
        assert len(wire.generates()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["digest", "expiry", "state", "name", "uri", "size"])
async def test_invalid_live_file_prevents_paid_dispatch(
    gemini, resolver, inputs, s3, case
):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        mutations = {
            "digest": ("sha256Hash", base64.b64encode(b"x" * 32).decode()),
            "expiry": ("expirationTime", datetime.now(timezone.utc).isoformat()),
            "state": ("state", "FAILED"),
            "name": ("name", "files/another"),
            "uri": ("uri", "https://attacker.invalid/file"),
            "size": ("sizeBytes", "1"),
        }
        key, value = mutations[case]
        wire.file[key] = value
        with pytest.raises(gemini.GeminiBoundaryError):
            await provider.generate(media.run, request_for(media), uuid4(), uuid4())
        assert wire.generates() == []
        assert len([c for c in wire.calls if c[0] == "POST"]) == 2  # never re-upload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.invalid/upload/x",
        "http://generativelanguage.googleapis.com/upload/x",
        "https://generativelanguage.googleapis.com.evil/upload/x",
    ],
)
async def test_upload_url_cannot_redirect_private_video(
    gemini, resolver, inputs, s3, url
):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        wire.upload_url = url
        with pytest.raises(gemini.GeminiBoundaryError):
            await uploaded(provider)
        assert len(wire.calls) == 1  # only metadata, no video body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["missing", "partial", "blocked", "truncated", "malformed"]
)
async def test_usage_survives_nonusable_output(gemini, resolver, inputs, s3, mode):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        if mode == "missing":
            del wire.result["usageMetadata"]
        elif mode == "partial":
            wire.result["usageMetadata"] = {"promptTokenCount": 10}
        elif mode == "blocked":
            wire.result["candidates"] = []
            wire.result["promptFeedback"] = {"blockReason": "SAFETY"}
        elif mode == "truncated":
            wire.result["candidates"][0]["finishReason"] = "MAX_TOKENS"
        else:
            wire.result["candidates"][0]["content"]["parts"] = [{"text": "not JSON"}]
        reply = await provider.generate(media.run, request_for(media), uuid4(), uuid4())
        assert reply.usage.cost_nano_usd is None
        assert reply.usage.usage_status == {
            "missing": "unknown",
            "partial": "partial",
        }.get(mode, "complete")
        if mode in {"blocked", "truncated"}:
            assert reply.text == ""  # parser must fail, not infer zero events
        elif mode == "malformed":
            assert reply.text == "not JSON"
        assert (
            json.loads(
                base64.b64decode(
                    receipt_value(s3, reply.usage.raw_usage)["body_base64"]
                )
            )
            == wire.result
        )


@pytest.mark.asyncio
async def test_revocation_after_paid_response_keeps_usage(gemini, resolver, inputs, s3):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        revoked = False

        def revoke():
            nonlocal revoked
            revoked = True

        async def guard(run):
            if revoked:
                raise PermissionError("revoked")

        provider.guard, wire.after_generate = guard, revoke
        reply = await provider.generate(media.run, request_for(media), uuid4(), uuid4())
        assert reply.usage.input_tokens == 101
        with pytest.raises(PermissionError):
            await provider.generate(media.run, request_for(media), uuid4(), uuid4())
        assert len(wire.generates()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["prompt", "model", "window", "crop"])
async def test_tampered_request_or_identity_has_no_provider_io(
    gemini, resolver, inputs, s3, mutation
):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        request = request_for(media)
        if mutation == "prompt":
            request = request.model_copy(update={"system_prompt": "different prompt"})
        elif mutation == "model":
            provider.models = {}
        elif mutation == "window":
            request = request.model_copy(
                update={"input_window": dict(start_ms=0, end_ms=3000)}
            )
        else:
            provider.media = replace(media, crops=tuple(reversed(media.crops)))
        calls = len(wire.calls)
        with pytest.raises((gemini.GeminiBoundaryError, ValueError)):
            await provider.generate(media.run, request, uuid4(), uuid4())
        assert len(wire.calls) == calls


@pytest.mark.asyncio
async def test_reopen_upload_and_explicit_delete(gemini, resolver, inputs, s3):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        checkpoint = await uploaded(provider)
        provider.uploaded = None
        provider.bind_upload(checkpoint)
        with pytest.raises(gemini.GeminiBoundaryError):
            provider.bind_upload(
                replace(checkpoint, uri="https://attacker.invalid/video")
            )
        receipt = await provider.delete_upload(uuid4())
        assert (
            provider.uploaded is None
            and receipt_value(s3, receipt)["kind"] == "file_delete"
        )
        assert len([c for c in wire.calls if c[0] == "DELETE"]) == 1


@pytest.mark.asyncio
async def test_executor_retains_malformed_usage_and_never_retries_implicitly(
    gemini, resolver, inputs, s3, tmp_path
):
    path = (
        Path(__file__).resolve().parents[2] / "analysis_core/tests/journal_fixture.py"
    )
    spec = importlib.util.spec_from_file_location("s39_provider_test_journal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        journal = module.DiskJournal(tmp_path / "provider.sqlite")

        async def guard():
            await resolver.guard(media.run)

        runner = StepRunner(
            media.run, media.config_bytes, journal, provider, guard=guard
        )
        wire.result["candidates"][0]["content"]["parts"] = [{"text": "not JSON"}]
        with pytest.raises(ValueError):
            await runner.model(request_for(media))
        with pytest.raises(RecoveryRequired):
            await runner.model(request_for(media))
        assert len(wire.generates()) == 1
        history = journal.records(history=True)
        assert history[-1].usage.input_tokens == 101
        assert history[-1].reply.text == "not JSON"
        assert receipt_value(s3, history[-1].usage.raw_usage)["kind"] == "generate"
        journal.close()


@pytest.mark.asyncio
async def test_processing_poll_is_bounded_and_not_another_upload(
    gemini, resolver, inputs, s3
):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        wire.file["state"] = "PROCESSING"
        with pytest.raises(gemini.GeminiBoundaryError, match="not usable"):
            await provider.upload(uuid4(), mime_type="video/mp4", max_polls=1)
        assert [c[0] for c in wire.calls] == ["POST", "POST", "GET"]
        assert provider.uploaded is None


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["model", "tokens", "multiple", "invalid_json"])
async def test_invalid_provider_response_is_retained_before_validation(
    gemini, resolver, inputs, s3, case
):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        if case == "model":
            wire.result["modelVersion"] = "unqualified-revision"
        elif case == "tokens":
            wire.result["usageMetadata"]["promptTokenCount"] = True
        elif case == "multiple":
            wire.result["candidates"] *= 2
        else:
            wire.result = "not a response object"
        retained, retain = [], provider.artifacts.retain

        async def observe(kind, identity, value):
            ref = await retain(kind, identity, value)
            if kind == "generate":
                retained.append(ref)
            return ref

        provider.artifacts.retain = observe
        with pytest.raises(gemini.GeminiBoundaryError):
            await provider.generate(media.run, request_for(media), uuid4(), uuid4())
        assert len(wire.generates()) == 1 and len(retained) == 1
        assert (
            json.loads(base64.b64decode(receipt_value(s3, retained[0])["body_base64"]))
            == wire.result
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("after_commit", [False, True])
async def test_raw_receipt_commit_failure_does_not_retry_model(
    gemini, resolver, inputs, s3, after_commit
):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        retain, receipts = provider.artifacts.retain, []

        async def lost(kind, identity, value):
            if kind != "generate":
                return await retain(kind, identity, value)
            if after_commit:
                receipts.append(await retain(kind, identity, value))
            raise TimeoutError("raw receipt commit response lost")

        provider.artifacts.retain = lost
        with pytest.raises(TimeoutError):
            await provider.generate(media.run, request_for(media), uuid4(), uuid4())
        assert len(wire.generates()) == 1
        if after_commit:
            assert receipt_value(s3, receipts[0])["kind"] == "generate"


@pytest.mark.asyncio
async def test_revocation_after_file_get_prevents_generate(
    gemini, resolver, inputs, s3
):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)

        async def guard(run):
            if any(c[0] == "GET" for c in wire.calls):
                raise PermissionError("revoked during lookup")

        provider.guard = guard
        with pytest.raises(PermissionError):
            await provider.generate(media.run, request_for(media), uuid4(), uuid4())
        assert wire.generates() == []


@pytest.mark.asyncio
async def test_cancel_during_source_verification_closes_descriptor(
    gemini, resolver, inputs, s3
):
    import threading

    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        entered, release, closed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original = media.open_source

        @contextmanager
        def blocked():
            with original() as stream:
                entered.set()
                assert release.wait(5)
                try:
                    yield stream
                finally:
                    closed.set()

        # A tiny host wrapper changes only acquisition timing, not media bytes.
        class SlowMedia:
            open_source = staticmethod(blocked)

        provider.media = SlowMedia()
        task = asyncio.create_task(provider.upload(uuid4(), mime_type="video/mp4"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert closed.is_set() and wire.calls == []


@pytest.mark.asyncio
async def test_response_byte_limit_and_no_retry(
    gemini, resolver, inputs, s3, monkeypatch
):
    async with provider_case(gemini, resolver, inputs, s3) as (provider, wire, media):
        await uploaded(provider)
        # Keep file_get below this cap; make only the generation response oversized.
        monkeypatch.setattr(gemini, "MAX_RESPONSE_BYTES", 1024)
        wire.result["candidates"][0]["content"]["parts"] = [{"text": "x" * 2048}]
        with pytest.raises(gemini.GeminiBoundaryError, match="byte limit"):
            await provider.generate(media.run, request_for(media), uuid4(), uuid4())
        assert len(wire.generates()) == 1


def test_documented_and_observed_sha_encodings_are_exact(gemini):
    digest = "183650e2def63712de785b641973235b1d8c7212344b0b36ca6972bfb6e15a8e"
    # Published actual API response: googleapis/js-genai issue381.
    observed = "MTgzNjUwZTJkZWY2MzcxMmRlNzg1YjY0MTk3MzIzNWIxZDhjNzIxMjM0NGIwYjM2Y2E2OTcyYmZiNmUxNWE4ZQ=="
    assert gemini.file_sha256(observed) == digest
    assert (
        gemini.file_sha256(base64.b64encode(bytes.fromhex(digest)).decode()) == digest
    )
    for invalid in (
        base64.b64encode(b"x" * 64).decode(),
        base64.b64encode(b"a" * 31).decode(),
        None,
    ):
        with pytest.raises(gemini.GeminiBoundaryError):
            gemini.file_sha256(invalid)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thinking,resolution,fps",
    [
        ("minimal", "low", 1),
        ("low", "medium", 30),
        ("medium", "high", 10),
        ("high", "low", 5),
    ],
)
async def test_admitted_parameters_map_without_sdk_defaults(
    gemini, resolver, inputs, s3, thinking, resolution, fps
):
    run, config, put = inputs
    settings = json.loads(config)
    for node in (
        "position_scan",
        "position_critique",
        "technique_scan",
        "technique_critique",
        "technique_analyze",
        "technique_validate",
    ):
        settings[node].update(thinking=thinking, media_resolution=resolution, fps=fps)
    encoded = json.dumps(settings).encode()
    data = run.model_dump()
    data["spec"]["effective_config"] = put(encoded, "configured.json")
    variant = (RegisteredAIRunV1.model_validate(data), encoded, put)
    async with provider_case(gemini, resolver, variant, s3) as (provider, wire, media):
        await uploaded(provider)
        await provider.generate(media.run, request_for(media), uuid4(), uuid4())
        body = json.loads(wire.generates()[0][2])
        assert (
            body["generationConfig"]["thinkingConfig"]["thinkingLevel"]
            == thinking.upper()
        )
        assert (
            body["generationConfig"]["mediaResolution"]
            == "MEDIA_RESOLUTION_" + resolution.upper()
        )
        assert body["contents"][0]["parts"][4]["videoMetadata"]["fps"] == fps
