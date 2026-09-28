# Saved provider receipt reconciliation (opt-in)

This boundary repairs a specific crash window: Gemini returned a response and the
private S3 receipt was retained, but the host's Keyspaces completion was not committed.
It does not resolve every upload/delete failure or retry an external write.

## Explicit host operations

Trusted host code, with verified media and a **currently authorized live guard**, may
call `DurableGeminiRun.reconcile_upload(receipt_ref)` or
`DurableGeminiRun.reconcile_cleanup(receipt_ref)`. The reference identifies private
evidence; it grants no authority. These methods are not public routes or background
tasks and are not automatically called by the worker/scheduler.

| Operation | Required evidence and state | External work during recovery |
| --- | --- | --- |
| Upload | Existing running upload reservation and successful finalize receipt for its exact call ID; no cleanup started | One GET of that exact file, validating active state, source bytes, MIME, name/URI and safe expiry |
| Cleanup | Completed DAG, committed upload target, running cleanup reservation, successful DELETE receipt naming that exact target/source | None |

Receipt reads require the configured private bucket and exact
analysis/run/kind/operation/SHA-addressed key. Reads are bounded, bodies are closed,
SHA-256 is verified, and the envelope must match run, operation and kind. Status must
be a successful HTTP integer and recorded time must fall between the reservation
start and the current clock. Caller-supplied public URLs or arbitrary object keys
are rejected before reading them.

Upload recovery accepts an original ACTIVE or PROCESSING finalize response only
when its file metadata passes the existing integrity/expiry checks. The fresh GET
must be ACTIVE and retain the same identity. It is receipted privately and links
back to the exact finalize receipt in its context. PROCESSING, failed, missing,
expired or mismatched current files remain blocked; there is no polling or re-upload
inside reconciliation. Normal later model dispatch still performs its own fresh GET.

Cleanup recovery requires the original delete receipt's `uploaded_name` and
`source_sha256` context to equal the committed upload. Older receipts with no target
context are insufficient for this helper; they are not silently treated as proof.
Already-succeeded cleanup checkpoints remain replayable as before. No 404 or failed
request is converted into inferred deletion success.

## Persistence, cancellation and cost

Recovery completes the **same reservation and attempt** using the existing journal
CAS, preserving call ID and immutable history. It does not allocate a new model
attempt, admission or user-use allowance. Upload/delete remain zero *model* cost;
that is not a statement that storage/network infrastructure costs nothing.

Two reconcilers may issue read-only GETs, but only one checkpoint CAS can win. A
loser reports conflict. Lost CAS acknowledgment is resolved by reading the current
head: a committed result replays, while an uncommitted one requires another explicit
evidence recovery. Only the GET may repeat—never upload, DELETE or generation.
Revocation/cancellation prevents subsequent authorized actions. Blocking S3 reads
drain and close safely; returned receipts remain private audit evidence.

No checkpoint result is bound for subsequent use until completion CAS and authority
checks succeed. As elsewhere, guards do not retroactively cancel HTTP already sent
or an already committed storage write; immutable evidence is retained.

## Remaining operational work

The [registered-run dispatcher](analysis-run-dispatch.md) now supports explicit
`ProviderActions` within the existing worker heartbeat and resource lifetime. Receipt
recovery requires `recover=True`; requested cleanup runs before lifecycle completion.
An uncertain cleanup cannot be bypassed by omitting the action on a later recovery.
A closed completed-worker lease still is not cleanup authority: terminal/revoked-run
maintenance needs a separately approved integration. Current account access,
cancellation and lifecycle ownership are not bypassed.

If finalize/delete responses were never durably retained, these helpers cannot prove
the outcome. Receipt inventory/discovery, resumable-session or orphan lookup,
expired-upload policy and cleanup of failed/cancelled/revoked runs remain separate
gates. No arbitrary retry, deletion, refund or retention policy is invented. Live
provider and AWS qualification are still required before mounting this path.

## Verification

```sh
CHECKPOINT_CASSANDRA_HOST=127.0.0.1 ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 ../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/run_host tests/run_host -q --tb=short
```

Native tests cover valid evidence, altered location/body/envelope, wrong target,
missing prerequisites/evidence, processing-to-active recovery, fresh file failures,
revocation, cancellation, competing CAS and before/after completion acknowledgment
loss. Storage/media/journal are real disposable local resources; Gemini HTTP is
intercepted. No live model, production deletion or UI behavior is claimed.
