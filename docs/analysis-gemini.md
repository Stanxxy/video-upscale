# Gemini native provider boundary (opt-in)

`service/analysis_gemini.py` implements the installed core0.3 provider port through
Gemini Developer REST endpoints. It uses the already installed HTTPX runtime,
now an explicit service dependency. The active runner, default models and admission
routes are unchanged. There were no live Gemini requests during local verification.

## What the host must provide

- An open `VerifiedRunMedia` context for the exact authoritative registered run.
- `S3GeminiArtifacts` with an explicit private bucket/client, and `guard(run)` that
  checks current run ownership, cancellation and worker lease.
- An explicit API key and map of admitted model IDs to qualified returned model
  versions. No model list, fallback model or prices are guessed by the adapter.
- A durable Keyspaces reservation/checkpoint for upload/cleanup, and the existing
  `StepRunner` journal for every model attempt. The S3 receipts are immutable
  artifacts, not scheduling/retry authority or a second execution state machine.

The provider must stay inside the media context and be closed with `aclose()`.
Only trusted host code may inject an HTTP transport (used by local wire tests);
no browser-supplied client, endpoint, file checkpoint or qualification map is valid.

## File lifecycle

`upload(upload_id, mime_type=...)` is one explicit upload start/finalize operation.
It rehashes and uploads the same verified descriptor, then checks provider file
name/URI/MIME, exact source size/SHA, processing state and expiry. Processing polls
are bounded; they do not retry failed transport calls. Source reads run off the
event loop and cancellation drains the read/verification worker before closing its
descriptor. Return metadata is private, not an expiring public URL for clients.

The host checkpoints the returned `UploadedVideo`. On restart it may pass its
trusted checkpoint to `bind_upload`; every model call rechecks that exact file
through `files.get`. Failed/expired/mismatched files do not trigger automatic
re-upload. Lost upload acknowledgments require host reconciliation; this adapter
now supports [explicit saved-receipt reconciliation](analysis-provider-reconciliation.md).
It does not implement resumable-upload recovery or inventory after a process crash.

`delete_upload(operation_id)` is explicit, guarded and receipted. It never deletes
the original S3 video/crops. New deletion receipts include the exact uploaded file
name and source hash so a lost checkpoint can be reconciled against its target.
No automatic best-effort deletion masks a failure.
Host retention/revocation cleanup must run with appropriate internal authority,
including when ordinary user-run authority is already revoked.

The pinned endpoint rejects returned foreign/insecure upload URLs and redirects.
Credentials are headers, never query parameters. Client construction disables
environment proxy/auth discovery, automatic redirects and transport retries; total
wall time and streamed response size are bounded. The installed SDK's internal
upload retry loop is not used. See the documented
[REST Files protocol](https://ai.google.dev/api/files).

`sha256Hash` accepts strict base64 of either 32 digest bytes or 64 lowercase ASCII
hex bytes. Both compare to the same admitted SHA, never bypass verification.
The latter representation appears in a
[recorded API response](https://github.com/googleapis/js-genai/issues/381);
the current target's representation still belongs in the live canary.

## Native request and response

Prepared config/prompt/schema/stage inputs are rebuilt against the admitted core
bundle before I/O. Only trusted persisted advisory position output may differ
from an independently rebuilt request. Each crop is paired with its ordered stable
player ID and display name, with exact inline bytes. The full uploaded video uses
integer-millisecond offsets formatted as protobuf durations (e.g. `0.500s`), exact
FPS, admitted thinking/media resolution, one candidate, and structured JSON output.
No clip-origin rebasing, arbitrary padding or timestamp repair is performed.

The request keeps the core's absolute-millisecond prompts unchanged. Qualification
must prove the selected model's response clock, native sampling, crop identity and
all six output schemas; HTTP fixtures cannot establish those behaviors. See
[generateContent / VideoMetadata / GenerationConfig](https://ai.google.dev/api/generate-content).

Every bounded received response (including HTTP errors) is retained privately
before parsing, with run/call identity, response bytes/status and the prepared
request context for generations. No credentials/upload-session headers are stored.
Transport loss/oversize responses remain uncertain; the adapter does not fabricate
a usable response or retry. A failed raw-receipt commit also stops the attempt.

Known provider token fields, response ID and the raw receipt populate `CallUsageV1`.
Missing fields remain null/partial/unknown. Modality detail and actual model version
remain in the raw artifact. Returned model versions outside the explicit qualified
set fail after raw retention. No rate estimation, user credit debit or invented
zero cost is supplied: `cost_status=unknown` until a qualified pricing projector
can interpret the retained usage.

Malformed model text is returned unchanged for executor persistence before strict
parsing. Blocked/truncated generations have no usable JSON (`text=""`); their usage
is retained and the parser fails, rather than interpreting them as zero techniques.
Multiple candidates/non-text responses and malformed usage fail with retained raw
evidence. A post-dispatch authority change does not discard already paid usage;
the executor checks authority before subsequent calls/publication.

## Verification and remaining integration

From the engine worktree:

```sh
ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 ../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/media_preparation tests/media_preparation -q --tb=short
```

Tests use real generated FFmpeg media and LocalStack S3, with the actual HTTPX
serialization/streaming layer intercepted at its transport. They assert exact
uploads, native payloads, all thinking levels, credential isolation, no retry after
503/timeout/redirect, file expiry/checksum rejection, private raw retention, unknown
cost, cancellation and executor malformed-response history. No full legacy/GPU
suite or live model accuracy is inferred from those checks.

The opt-in [run host](analysis-run-host.md) now supplies Keyspaces-owned upload/cleanup
reservations and complete-result coordination, including saved-receipt recovery.
Remaining integration: receipt-less orphan/expiry reconciliation and maintenance authority,
active runner lifecycle/lease wiring, model/schema/clock/crop canaries on approved media,
qualified rate snapshots and durable progress/usage projections. No production
activation should occur before these gates. Code rollback leaves private receipts
and existing results intact; do not reinterpret prior attempts under another
provider/core version or automatically rerun paid work.
