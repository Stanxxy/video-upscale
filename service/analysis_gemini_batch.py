"""Gemini Batch API delivery for "Ready later" analysis (x-aiquota M3.0 spike).

Same requests as ``GeminiProvider`` (``_request_body``), same response handling
(``_reply``); only the delivery differs. The calls a stage dispatches together are
collected for ``settle_seconds`` after the last arrival, written as one JSONL file,
and submitted as one ``batchGenerateContent`` job (half the interactive price).

Durability: before a job is created, a ``batch_ledger`` receipt is retained for
every step (keyed by ``execution_id``, which survives a resume; ``call_id`` does
not). A resumed ``generate`` for a step already in a ledger polls that job and
never submits the call again. Each answer is retained as an ordinary ``generate``
receipt whose context says ``delivery: batch``, so cost reports price it at the
batch rate.
"""

import asyncio
import base64
import hashlib
import json
import logging
import time
from uuid import UUID, uuid4

from bjj_analysis_core.requests import PreparedCallV1, canonical_json, strict_json
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

from service.analysis_gemini import (
    MAX_RESPONSE_BYTES,
    ORIGIN,
    GeminiBoundaryError,
    GeminiProvider,
    _blocking,
)

logger = logging.getLogger("service.analysis_gemini_batch")

SUCCEEDED = {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}
ENDED = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
}
MAX_BATCH_INPUT_BYTES = 512 * 1024 * 1024
MAX_BATCH_OUTPUT_BYTES = 256 * 1024 * 1024


class BatchGeminiProvider(GeminiProvider):
    name = "google-gemini-batch-v1"
    collects_batches = True  # the executor dispatches whole stages at once

    def __init__(self, *args, settle_seconds=3.0, poll_seconds=20.0, max_wait_seconds=48 * 3600, **kwargs):
        super().__init__(*args, **kwargs)
        self.settle_seconds, self.poll_seconds, self.max_wait = settle_seconds, poll_seconds, max_wait_seconds
        self._pending = {}  # execution_id -> (model, body, future)
        self._timer = None
        self._jobs = {}  # job name -> task resolving {execution_id: raw response bytes}
        self.submitted = []  # job names, in order (spike evidence)

    # -- generate ---------------------------------------------------------------
    async def generate(self, run, request, execution_id, call_id):
        if RegisteredAIRunV1.model_validate(run) != self.run:
            raise GeminiBoundaryError("provider run binding differs")
        request = PreparedCallV1.model_validate(request)
        execution_id, call_id = UUID(str(execution_id)), UUID(str(call_id))
        body = self._request_body(request)
        await self.guard(self.run)
        job = await self._ledger_job(execution_id)
        if job is None:
            future = asyncio.get_running_loop().create_future()
            self._pending[execution_id] = (request.config.model, body, future)
            self._arm()
            job = await future
        answers = await self._job_answers(job)
        item = answers.get(str(execution_id))
        if item is None:
            raise GeminiBoundaryError(f"batch {job} returned no answer for this step")
        if "error" in item or "response" not in item:
            raise GeminiBoundaryError(f"batch item failed: {json.dumps(item.get('error'))[:300]}")
        raw = canonical_json(item["response"]).encode()
        receipt = await self.artifacts.retain(
            "generate",
            call_id,
            dict(
                run_id=str(self.run.spec.run_id),
                identity=str(call_id),
                kind="generate",
                recorded_at=self.now().isoformat(),
                status=200,
                body_base64=base64.b64encode(raw).decode(),
                context={
                    "delivery": "batch",
                    "batch": job,
                    "execution_id": str(execution_id),
                    "request": request.model_dump(mode="json"),
                    "uploaded_name": self.uploaded.name,
                    "source_sha256": self.run.spec.source.sha256,
                },
            ),
        )
        return self._reply(raw, receipt, request, execution_id, call_id)

    # -- collection ---------------------------------------------------------------
    def _arm(self):
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.get_running_loop().call_later(
            self.settle_seconds, lambda: asyncio.ensure_future(self._flush())
        )

    async def _flush(self):
        pending, self._pending, self._timer = self._pending, {}, None
        by_model = {}
        for execution_id, (model, body, future) in pending.items():
            by_model.setdefault(model, []).append((execution_id, body, future))
        for model, items in by_model.items():
            try:
                job = await self._submit(model, items)
            except BaseException as error:  # every waiting step sees the failure
                for _, _, future in items:
                    if not future.done():
                        future.set_exception(error)
                continue
            for _, _, future in items:
                if not future.done():
                    future.set_result(job)

    async def _submit(self, model, items):
        await self.guard(self.run)
        raw, _, _ = await self._request("GET", f"{ORIGIN}/v1beta/{self.uploaded.name}", "file_get", uuid4())
        self._file(strict_json(raw), mime_type=self.uploaded.mime_type, expected_name=self.uploaded.name)
        lines = "".join(
            json.dumps({"key": str(execution_id), "request": body}, separators=(",", ":")) + "\n"
            for execution_id, body, _ in items
        ).encode()
        if len(lines) > MAX_BATCH_INPUT_BYTES:
            raise GeminiBoundaryError("batch input exceeds limit")
        input_file = await self._upload_jsonl(lines)
        response = await self.http.post(
            self._url(f"{ORIGIN}/v1beta/models/{model}:batchGenerateContent"),
            json={"batch": {"display_name": f"run-{self.run.spec.run_id}", "input_config": {"file_name": input_file}}},
        )
        created = await self.artifacts.retain(
            "batch_create",
            uuid4(),
            dict(run_id=str(self.run.spec.run_id), status=response.status_code, model=model,
                 input_file=input_file, input_sha256=hashlib.sha256(lines).hexdigest(),
                 keys=[str(e) for e, _, _ in items], body=response.text[:MAX_RESPONSE_BYTES]),
        )
        if response.status_code != 200:
            raise GeminiBoundaryError(f"batch create HTTP {response.status_code}; receipt retained")
        job = response.json()["name"]
        # Ledger before anyone waits: a resume finds the job instead of resubmitting.
        await asyncio.gather(*(
            self.artifacts.retain("batch_ledger", execution_id,
                                  dict(run_id=str(self.run.spec.run_id), execution_id=str(execution_id),
                                       batch=job, create_receipt=created.key))
            for execution_id, _, _ in items
        ))
        self.submitted.append(job)
        logger.info("batch %s submitted: %d requests, model %s, receipt %s", job, len(items), model, created.key)
        return job

    async def _upload_jsonl(self, data):
        start = await self.http.post(
            self._url(f"{ORIGIN}/upload/v1beta/files", upload=True),
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(data)),
                "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            },
            json={"file": {"display_name": f"batch-{self.run.spec.run_id}.jsonl"}},
        )
        if start.status_code != 200:
            raise GeminiBoundaryError(f"batch input upload start HTTP {start.status_code}")
        session = self._url(start.headers["x-goog-upload-url"], upload=True)
        done = await self.http.post(
            session,
            headers={"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"},
            content=data,
        )
        if done.status_code != 200:
            raise GeminiBoundaryError(f"batch input upload HTTP {done.status_code}")
        return done.json()["file"]["name"]

    # -- ledger and results ---------------------------------------------------------
    async def _ledger_job(self, execution_id):
        prefix = (
            f"analysis-provider/v1/{self.run.spec.analysis_id}/{self.run.spec.run_id}/batch_ledger/{execution_id}/"
        )

        def find():
            listed = self.artifacts.client.list_objects_v2(Bucket=self.artifacts.bucket, Prefix=prefix)
            jobs = set()
            for item in listed.get("Contents", []):
                body = self.artifacts.client.get_object(Bucket=self.artifacts.bucket, Key=item["Key"])["Body"]
                try:
                    jobs.add(strict_json(body.read(MAX_RESPONSE_BYTES))["batch"])
                finally:
                    body.close()
            if len(jobs) > 1:
                raise GeminiBoundaryError("step recorded in more than one batch")
            return next(iter(jobs), None)

        return await _blocking(find)

    def _job_answers(self, job):
        if job not in self._jobs:
            self._jobs[job] = asyncio.ensure_future(self._wait(job))
        return self._jobs[job]

    async def _wait(self, job):
        started = time.monotonic()
        while True:
            await self.guard(self.run)
            response = await self.http.get(self._url(f"{ORIGIN}/v1beta/{job}"))
            if response.status_code != 200:
                raise GeminiBoundaryError(f"batch get HTTP {response.status_code}")
            value = response.json()
            meta = value.get("metadata", value)
            state = meta.get("state") or value.get("state")
            if state in SUCCEEDED:
                break
            if state in ENDED:
                await self.artifacts.retain("batch_get", uuid4(), dict(run_id=str(self.run.spec.run_id), body=value))
                raise GeminiBoundaryError(f"batch {job} ended {state}")
            if time.monotonic() - started > self.max_wait:
                raise GeminiBoundaryError(f"batch {job} still {state} after {self.max_wait}s")
            await asyncio.sleep(self.poll_seconds)
        await self.artifacts.retain("batch_get", uuid4(), dict(run_id=str(self.run.spec.run_id), body=value))
        output = value.get("response") or meta.get("output") or {}
        name = output.get("responsesFile") or (meta.get("dest") or {}).get("fileName")
        if not name:
            raise GeminiBoundaryError(f"batch {job} has no responses file")
        download = await self.http.get(
            f"{ORIGIN}/download/v1beta/{name}:download", params={"alt": "media"}
        )
        if download.status_code != 200 or len(download.content) > MAX_BATCH_OUTPUT_BYTES:
            raise GeminiBoundaryError(f"batch {job} results download HTTP {download.status_code}")
        answers = {}
        for line in download.text.splitlines():
            if line.strip():
                item = json.loads(line)
                answers[item.get("key") or (item.get("metadata") or {}).get("key")] = item
        logger.info("batch %s done after %.1f min: %d answers", job, (time.monotonic() - started) / 60, len(answers))
        return answers
