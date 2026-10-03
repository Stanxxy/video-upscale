"""Opt-in Gemini Developer REST adapter; one HTTP attempt, private raw receipts.

No SDK retries, credential discovery, admission, scheduler or active runner hook.
The host checkpoints uploads and owns qualification, leases and reconciliation.
"""

import asyncio
import base64
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from bjj_analysis_core.execution import ProviderReplyV1
from bjj_analysis_core.requests import (
    PreparedCallV1,
    canonical_json,
    load_config,
    prepare_call,
    strict_json,
)
from botocore.exceptions import ClientError
from pydantic import AwareDatetime, TypeAdapter
from shared_lib.models.analysis_contracts import CallUsageV1, PrivateArtifactRef
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

ORIGIN = "https://generativelanguage.googleapis.com"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
logger = logging.getLogger(__name__)


class GeminiBoundaryError(RuntimeError):
    """Failure is explicit; never authorize an implicit provider retry."""


def file_sha256(encoded):
    """Decode exact SHA representations, never bypass a mismatch.

    Files API documents base64 bytes; observed API records also wrap an ASCII
    hex digest in base64. Both representations identify the same 256-bit value.
    """
    if not isinstance(encoded, str):
        raise GeminiBoundaryError("missing provider SHA-256")
    value = base64.b64decode(encoded, validate=True)
    if len(value) == 32:
        return value.hex()
    if re.fullmatch(b"[0-9a-f]{64}", value):
        return value.decode("ascii")
    raise GeminiBoundaryError("invalid provider SHA-256 representation")


async def _blocking(operation, *args):
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


class S3GeminiArtifacts:
    """Immutable artifacts only, never another upload/execution state machine."""

    def __init__(self, run, client, *, bucket):
        self.run = RegisteredAIRunV1.model_validate(run)
        if not isinstance(bucket, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket
        ):
            raise ValueError("explicit private artifact bucket required")
        self.client, self.bucket = client, bucket

    def _retain(self, kind, identity, value):
        if kind not in {
            "upload_start",
            "upload_finalize",
            "file_get",
            "file_delete",
            "generate",
        }:
            raise ValueError("unknown receipt kind")
        identity = UUID(str(identity))
        data = canonical_json(value).encode()
        if len(data) > 2 * MAX_RESPONSE_BYTES:
            raise GeminiBoundaryError("private provider receipt exceeds limit")
        digest = hashlib.sha256(data).hexdigest()
        ref = PrivateArtifactRef(
            bucket=self.bucket,
            key=f"analysis-provider/v1/{self.run.spec.analysis_id}/{self.run.spec.run_id}/{kind}/{identity}/{digest}.json",
            sha256=digest,
        )
        try:
            self.client.put_object(
                Bucket=ref.bucket,
                Key=ref.key,
                Body=data,
                ContentType="application/json",
                ACL="private",
                ServerSideEncryption="AES256",
                IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
                raise
            response = self.client.get_object(Bucket=ref.bucket, Key=ref.key)
            body = response["Body"]
            try:
                if (
                    response.get("ContentLength") != len(data)
                    or body.read(len(data) + 1) != data
                ):
                    raise GeminiBoundaryError(
                        "immutable provider receipt changed"
                    ) from error
            finally:
                body.close()
        return ref

    async def retain(self, kind, identity, value):
        # Retain returned private usage even if authority was revoked mid-call.
        # The executor checks authority before any subsequent action.
        return await _blocking(self._retain, kind, identity, value)

    async def read(self, ref, *, kind, identity):
        """Read exact private evidence; callers cannot redirect to another run/key."""
        ref = PrivateArtifactRef.model_validate(ref)
        identity = UUID(str(identity))
        if kind not in {"upload_finalize", "file_delete"} or (
            ref.bucket != self.bucket
            or ref.key
            != (
                f"analysis-provider/v1/{self.run.spec.analysis_id}/{self.run.spec.run_id}/"
                f"{kind}/{identity}/{ref.sha256}.json"
            )
        ):
            raise GeminiBoundaryError("receipt outside registered operation")

        def read():
            response = self.client.get_object(Bucket=ref.bucket, Key=ref.key)
            body = response["Body"]
            try:
                if response.get("ContentLength", 0) > 2 * MAX_RESPONSE_BYTES:
                    raise GeminiBoundaryError("provider receipt exceeds limit")
                data = body.read(2 * MAX_RESPONSE_BYTES + 1)
            finally:
                body.close()
            if (
                len(data) > 2 * MAX_RESPONSE_BYTES
                or hashlib.sha256(data).hexdigest() != ref.sha256
            ):
                raise GeminiBoundaryError("provider receipt checksum mismatch")
            value = strict_json(data)
            if not isinstance(value, dict) or (
                value.get("run_id") != str(self.run.spec.run_id)
                or value.get("identity") != str(identity)
                or value.get("kind") != kind
            ):
                raise GeminiBoundaryError("provider receipt identity mismatch")
            return value

        return await _blocking(read)


@dataclass(frozen=True)
class UploadedVideo:
    run_id: UUID
    source_sha256: str
    name: str
    uri: str
    mime_type: str
    receipt: PrivateArtifactRef


def qualified_model_versions(qualified_models):
    """Validate explicit host qualification input; never qualify a model here."""
    if not isinstance(qualified_models, dict) or not qualified_models:
        raise ValueError("explicit qualified model/version pairs required")
    models = {}
    for model, versions in qualified_models.items():
        if not isinstance(model, str) or not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
            raise ValueError("explicit model ID required")
        if (
            not isinstance(versions, (set, frozenset, tuple, list))
            or not versions
            or any(not isinstance(v, str) or not v for v in versions)
        ):
            raise ValueError("qualified returned model versions required")
        models[model] = frozenset(versions)
    return models


class GeminiProvider:
    name = "google-gemini-native-v1"

    def __init__(
        self,
        media,
        artifacts,
        *,
        api_key,
        qualified_models,
        guard,
        timeout_seconds=180,
        clock=None,
        transport=None,
    ):
        self.run, self.config = load_config(media.run, media.config_bytes)
        if artifacts.run != self.run or not callable(guard):
            raise ValueError("matching artifacts and live run guard required")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("explicit Gemini credential required")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 1800:
            raise ValueError("bounded provider timeout required")
        self.models = qualified_model_versions(qualified_models)
        self.media, self.artifacts, self.guard = media, artifacts, guard
        self.timeout = timeout_seconds
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.uploaded = None
        self.http = httpx.AsyncClient(
            headers={"x-goog-api-key": api_key},
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_seconds, connect=10),
            transport=(
                transport
                if transport is not None
                else httpx.AsyncHTTPTransport(retries=0)
            ),
        )

    async def aclose(self):
        await self.http.aclose()

    def now(self):
        return TypeAdapter(AwareDatetime).validate_python(self.clock())

    def _url(self, url, *, upload=False):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "generativelanguage.googleapis.com"
            or parsed.fragment
            or (not upload and parsed.query)
            or not parsed.path.startswith("/upload/" if upload else "/v1beta/")
        ):
            raise GeminiBoundaryError("provider URL outside pinned endpoint")
        return url

    async def _request(
        self, method, url, kind, identity, *, upload=False, context=None, **kwargs
    ):
        url = self._url(url, upload=upload)
        # Bound both total wall time and streamed response size. No redirect,
        # reconnect or retry loop. A lost response remains an uncertain outcome.
        transport_completed = False
        try:
            async with asyncio.timeout(self.timeout):
                async with self.http.stream(method, url, **kwargs) as response:
                    parts, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise GeminiBoundaryError(
                                "provider response exceeds byte limit"
                            )
                        parts.append(chunk)
                    raw = b"".join(parts)
                transport_completed = True

                envelope = dict(
                    run_id=str(self.run.spec.run_id),
                    identity=str(identity),
                    kind=kind,
                    recorded_at=self.now().isoformat(),
                    status=response.status_code,
                    body_base64=base64.b64encode(raw).decode(),
                    context=context,
                )
                ref = await self.artifacts.retain(kind, identity, envelope)
                if not 200 <= response.status_code < 300:
                    raise GeminiBoundaryError(
                        f"provider HTTP {response.status_code}; raw receipt retained"
                    )
                return raw, dict(response.headers), ref
        except GeminiBoundaryError:
            raise
        except Exception as error:
            if not transport_completed:
                operation = (
                    kind
                    if kind
                    in {
                        "upload_start",
                        "upload_finalize",
                        "file_get",
                        "file_delete",
                        "generate",
                    }
                    else "provider_request"
                )
                # Exception messages and request objects can contain credentials,
                # signed upload URLs, or submitted media details. Log only safe tags.
                logger.warning(
                    "Gemini provider transport failed operation=%s exception_class=%s",
                    operation,
                    type(error).__name__,
                )
            raise

    def _file(self, value, *, mime_type, expected_name=None, active=True):
        if not isinstance(value, dict):
            raise GeminiBoundaryError("provider file response must be an object")
        name = value.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"files/[a-z0-9-]+", name):
            raise GeminiBoundaryError("invalid provider file name")
        if expected_name is not None and name != expected_name:
            raise GeminiBoundaryError("provider returned another file")
        if (
            value.get("uri") != f"{ORIGIN}/v1beta/{name}"
            or value.get("mimeType") != mime_type
        ):
            raise GeminiBoundaryError("provider file URI/MIME mismatch")
        digest = file_sha256(value.get("sha256Hash"))
        size = value.get("sizeBytes")
        if isinstance(size, str) and re.fullmatch(r"[0-9]+", size):
            size = int(size)
        if (
            type(size) is not int
            or size != self.media.source_size
            or digest != self.run.spec.source.sha256
        ):
            raise GeminiBoundaryError("uploaded bytes differ from verified source")
        expiry = TypeAdapter(AwareDatetime).validate_python(value.get("expirationTime"))
        if expiry <= self.now() + timedelta(seconds=self.timeout):
            raise GeminiBoundaryError(
                "provider file expires before safe dispatch window"
            )
        state = value.get("state")
        if state not in ({"ACTIVE"} if active else {"ACTIVE", "PROCESSING"}):
            raise GeminiBoundaryError("provider file is not usable")
        return name

    async def upload(self, upload_id, *, mime_type, max_polls=60, poll_seconds=1):
        """Explicit one upload. Host reserves/checkpoints this operation separately.

        Never call automatically on failed file lookup, expiry, timeout or restart.
        A lost acknowledgment requires reconciliation, not another upload call.
        """
        upload_id = UUID(str(upload_id))
        if self.uploaded is not None:
            raise GeminiBoundaryError("upload already bound")
        if mime_type not in {
            "video/mp4",
            "video/quicktime",
            "video/webm",
            "video/x-msvideo",
            "video/mpeg",
        }:
            raise ValueError("explicit supported native MIME required")
        if (
            type(max_polls) is not int
            or not 0 <= max_polls <= 120
            or type(poll_seconds) is not int
            or not 1 <= poll_seconds <= 5
        ):
            raise ValueError("bounded processing polling required")
        await self.guard(self.run)
        manager = self.media.open_source()
        try:
            source = await _blocking(manager.__enter__)
            await self.guard(self.run)
            raw, headers, _ = await self._request(
                "POST",
                ORIGIN + "/upload/v1beta/files",
                "upload_start",
                upload_id,
                upload=True,
                headers={
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(self.media.source_size),
                    "X-Goog-Upload-Header-Content-Type": mime_type,
                },
                json={
                    "file": {"displayName": f"bjj-{self.run.spec.run_id}-{upload_id}"}
                },
            )
            url = self._url(headers.get("x-goog-upload-url", ""), upload=True)

            async def chunks():
                while chunk := await _blocking(source.read, 1024 * 1024):
                    await self.guard(self.run)
                    yield chunk

            await self.guard(self.run)
            raw, _, receipt = await self._request(
                "POST",
                url,
                "upload_finalize",
                upload_id,
                upload=True,
                headers={
                    "Content-Length": str(self.media.source_size),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
                content=chunks(),
            )
        finally:
            manager.__exit__(None, None, None)
        value = strict_json(raw)["file"]
        name = self._file(value, mime_type=mime_type, active=False)
        for _ in range(max_polls):
            if value["state"] == "ACTIVE":
                break
            await asyncio.sleep(poll_seconds)
            await self.guard(self.run)
            raw, _, receipt = await self._request(
                "GET", f"{ORIGIN}/v1beta/{name}", "file_get", upload_id
            )
            value = strict_json(raw)
            self._file(value, mime_type=mime_type, expected_name=name, active=False)
        self._file(value, mime_type=mime_type, expected_name=name)
        await self.guard(self.run)
        self.uploaded = UploadedVideo(
            self.run.spec.run_id,
            self.run.spec.source.sha256,
            name,
            value["uri"],
            mime_type,
            receipt,
        )
        return self.uploaded

    def bind_upload(self, uploaded):
        """Host passes its trusted checkpoint, not a client-supplied file URI."""
        if (
            not isinstance(uploaded, UploadedVideo)
            or uploaded.run_id != self.run.spec.run_id
            or uploaded.source_sha256 != self.run.spec.source.sha256
        ):
            raise GeminiBoundaryError("upload checkpoint belongs to another source/run")
        if (
            not re.fullmatch(r"files/[a-z0-9-]+", uploaded.name)
            or uploaded.uri != f"{ORIGIN}/v1beta/{uploaded.name}"
            or uploaded.mime_type
            not in {
                "video/mp4",
                "video/quicktime",
                "video/webm",
                "video/x-msvideo",
                "video/mpeg",
            }
        ):
            raise GeminiBoundaryError("invalid upload checkpoint target")
        self.uploaded = uploaded  # always revalidated via files.get before dispatch

    async def _successful_receipt(self, ref, *, kind, operation_id, not_before):
        await self.guard(self.run)
        value = await self.artifacts.read(ref, kind=kind, identity=operation_id)
        recorded = TypeAdapter(AwareDatetime).validate_python(value.get("recorded_at"))
        start = TypeAdapter(AwareDatetime).validate_python(not_before)
        status = value.get("status")
        if (
            type(status) is not int
            or not 200 <= status < 300
            or not start <= recorded <= self.now()
        ):
            raise GeminiBoundaryError("receipt is not a successful reserved operation")
        await self.guard(self.run)
        return value

    async def reconcile_upload(self, operation_id, receipt, *, mime_type, not_before):
        """Explicit evidence recovery: one fresh GET, never a second upload."""
        operation_id = UUID(str(operation_id))
        value = await self._successful_receipt(
            receipt,
            kind="upload_finalize",
            operation_id=operation_id,
            not_before=not_before,
        )
        body = strict_json(base64.b64decode(value["body_base64"], validate=True))
        name = self._file(body["file"], mime_type=mime_type, active=False)
        await self.guard(self.run)
        raw, _, current_receipt = await self._request(
            "GET",
            f"{ORIGIN}/v1beta/{name}",
            "file_get",
            operation_id,
            context={
                "reconciled_from": PrivateArtifactRef.model_validate(
                    receipt
                ).model_dump(mode="json")
            },
        )
        current = strict_json(raw)
        self._file(current, mime_type=mime_type, expected_name=name)
        await self.guard(self.run)
        # Binding is done only by the host after its checkpoint CAS succeeds.
        return UploadedVideo(
            self.run.spec.run_id,
            self.run.spec.source.sha256,
            name,
            current["uri"],
            mime_type,
            current_receipt,
        )

    async def reconcile_cleanup(self, operation_id, receipt, *, not_before):
        """Accept a saved successful DELETE for the exact checkpointed target."""
        if self.uploaded is None:
            raise GeminiBoundaryError("no committed cleanup target")
        value = await self._successful_receipt(
            receipt,
            kind="file_delete",
            operation_id=operation_id,
            not_before=not_before,
        )
        if value.get("context") != {
            "uploaded_name": self.uploaded.name,
            "source_sha256": self.run.spec.source.sha256,
        }:
            raise GeminiBoundaryError("cleanup receipt target mismatch")
        await self.guard(self.run)
        return PrivateArtifactRef.model_validate(receipt)

    def _request_body(self, request):
        request = PreparedCallV1.model_validate(request)
        initial = strict_json(request.initial_prompt)
        rebuilt = prepare_call(
            self.run,
            self.media.config_bytes,
            request.node_id,
            input_window=request.input_window,
            context=initial["stage_context"],
        )
        # Advisory positions originate in trusted persisted core output. Other
        # prompt/config/schema/context fields must equal the admitted builder.
        actual, expected = dict(initial), strict_json(rebuilt.initial_prompt)
        advisory = actual.pop("advisory_positions")
        expected.pop("advisory_positions")
        if actual != expected or (
            advisory is not None
            and request.node_id
            in {"position_scan", "technique_scan", "position_critique"}
        ):
            raise GeminiBoundaryError("prepared context differs from admitted request")
        if request.model_dump(exclude={"initial_prompt"}) != rebuilt.model_dump(
            exclude={"initial_prompt"}
        ):
            raise GeminiBoundaryError(
                "prepared configuration differs from admitted request"
            )
        if request.config.model not in self.models:
            raise GeminiBoundaryError("model has no explicit qualification")
        if request.config.model == "gemini-3.8-flash" and request.config.thinking == "minimal":
            raise GeminiBoundaryError("gemini-3.8-flash requires low, medium or high thinking")
        if self.uploaded is None:
            raise GeminiBoundaryError("no checkpointed provider upload bound")
        parts = []
        for crop, athlete in zip(self.media.crops, self.run.spec.athletes, strict=True):
            if (
                crop.player_id != athlete.player_id
                or crop.display_name != athlete.display_name
                or hashlib.sha256(crop.data).hexdigest()
                != athlete.reference_crop.sha256
            ):
                raise GeminiBoundaryError("ordered crop binding differs from run")
            parts += [
                {
                    "text": canonical_json(
                        {
                            "reference_player_id": crop.player_id,
                            "display_name": crop.display_name,
                        }
                    )
                },
                {
                    "inlineData": {
                        "mimeType": crop.mime_type,
                        "data": base64.b64encode(crop.data).decode(),
                    }
                },
            ]
        window = request.input_window
        offset = lambda ms: f"{ms // 1000}.{ms % 1000:03d}s"
        parts += [
            {
                "fileData": {
                    "fileUri": self.uploaded.uri,
                    "mimeType": self.uploaded.mime_type,
                },
                "videoMetadata": {
                    "startOffset": offset(window.start_ms),
                    "endOffset": offset(window.end_ms),
                    "fps": request.config.fps,
                },
            },
            {"text": request.initial_prompt},
        ]
        return dict(
            contents=[{"role": "user", "parts": parts}],
            systemInstruction={"parts": [{"text": request.system_prompt}]},
            generationConfig={
                "candidateCount": 1,
                "responseMimeType": "application/json",
                "responseJsonSchema": request.response_schema,
                "thinkingConfig": {
                    "thinkingLevel": request.config.thinking.upper(),
                    "includeThoughts": False,
                },
                "mediaResolution": "MEDIA_RESOLUTION_"
                + request.config.media_resolution.upper(),
            },
        )

    async def generate(self, run, request, execution_id, call_id):
        if RegisteredAIRunV1.model_validate(run) != self.run:
            raise GeminiBoundaryError("provider run binding differs")
        request = PreparedCallV1.model_validate(request)
        execution_id, call_id = UUID(str(execution_id)), UUID(str(call_id))
        body = self._request_body(request)
        await self.guard(self.run)
        raw, _, _ = await self._request(
            "GET", f"{ORIGIN}/v1beta/{self.uploaded.name}", "file_get", call_id
        )
        self._file(
            strict_json(raw),
            mime_type=self.uploaded.mime_type,
            expected_name=self.uploaded.name,
        )
        await self.guard(self.run)
        raw, _, receipt = await self._request(
            "POST",
            f"{ORIGIN}/v1beta/models/{request.config.model}:generateContent",
            "generate",
            call_id,
            json=body,
            context={
                "execution_id": str(execution_id),
                "request": request.model_dump(mode="json"),
                "uploaded_name": self.uploaded.name,
                "source_sha256": self.run.spec.source.sha256,
            },
        )
        # Retain first, parse second. No post-call guard discards paid usage;
        # the checkpoint executor reauthorizes before subsequent work/publication.
        value = strict_json(raw)
        if not isinstance(value, dict):
            raise GeminiBoundaryError(
                "provider response must be an object; raw receipt retained"
            )
        if value.get("modelVersion") not in self.models[request.config.model]:
            raise GeminiBoundaryError(
                "returned model version is not qualified; raw receipt retained"
            )
        metadata = value.get("usageMetadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise GeminiBoundaryError("invalid usage object; raw receipt retained")
        tokens = {}
        for local, remote in (
            ("input_tokens", "promptTokenCount"),
            ("output_tokens", "candidatesTokenCount"),
            ("thinking_tokens", "thoughtsTokenCount"),
            ("cached_input_tokens", "cachedContentTokenCount"),
        ):
            count = None if metadata is None else metadata.get(remote)
            if count is not None and (type(count) is not int or count < 0):
                raise GeminiBoundaryError("invalid token usage; raw receipt retained")
            tokens[local] = count
        status = (
            "complete"
            if all(v is not None for v in tokens.values())
            else "partial" if metadata is not None else "unknown"
        )
        usage = CallUsageV1(
            run_id=self.run.spec.run_id,
            execution_id=execution_id,
            call_id=call_id,
            recorded_at=self.now(),
            kind="model",
            provider=self.name,
            model=request.config.model,
            provider_request_id=value.get("responseId"),
            raw_usage=receipt,
            usage_status=status,
            cost_status="unknown",
            cost_nano_usd=None,
            **tokens,
        )
        candidates = value.get("candidates", [])
        if len(candidates) > 1:
            raise GeminiBoundaryError(
                "multiple provider candidates; raw receipt retained"
            )
        text = ""
        if candidates and candidates[0].get("finishReason") == "STOP":
            pieces = []
            for part in candidates[0].get("content", {}).get("parts", []):
                if part.get("thought") is True:
                    continue
                if not isinstance(part.get("text"), str):
                    raise GeminiBoundaryError(
                        "non-text response part; raw receipt retained"
                    )
                pieces.append(part["text"])
            text = "".join(pieces)
        # Blocked/truncated output is explicitly no usable JSON, never an empty
        # event list. The executor persists usage then fails strict response parse.
        return ProviderReplyV1(text=text, usage=usage)

    async def delete_upload(self, operation_id):
        """Explicit cleanup only; host durably schedules and records this action."""
        if self.uploaded is None:
            raise GeminiBoundaryError("no upload bound for cleanup")
        name = self.uploaded.name
        if not re.fullmatch(r"files/[a-z0-9-]+", name):
            raise GeminiBoundaryError("invalid cleanup target")
        await self.guard(self.run)
        _, _, receipt = await self._request(
            "DELETE",
            f"{ORIGIN}/v1beta/{name}",
            "file_delete",
            UUID(str(operation_id)),
            context={
                "uploaded_name": name,
                "source_sha256": self.run.spec.source.sha256,
            },
        )
        self.uploaded = None
        return receipt
