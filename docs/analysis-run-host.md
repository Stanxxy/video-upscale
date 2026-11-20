# Durable Gemini DAG host (opt-in)

`service/analysis_run_host.py` connects the installed core0.3 executor, verified
media/Gemini provider, Keyspaces checkpoint journal and an explicit component sink.
It is not mounted in the active worker and does not claim a worker lease, provision
resources, settle credits or write legacy `job_lifecycle` terminal status.

## Operations and authority

The host receives the authoritative registered run/config, matching journal/provider,
an idempotent durable publication sink, native source MIME and current authority/
cancellation/lease guard. Provider and media contexts remain caller-owned and must
stay open; the caller closes them on every exit. The opt-in
[worker attempt adapter](analysis-run-worker.md) now supplies conditional lifecycle
ownership and heartbeat/cancellation supervision; a guarded production factory and
version-aware scheduler remain required before production wiring.

Three stable host step IDs share the existing checkpoint table/model:

- `provider_upload`: reserve with CAS before calling upload, then persist returned
  private upload metadata. A succeeded record rebinds without another HTTP upload.
- `dag_complete`: the typed final result, written only after both core components
  reached their sink boundary and the final core guard passed. It does not mean
  the account consumer is already review-ready or that the engine lifecycle ended.
- `provider_cleanup`: reserve with CAS before an explicit provider-file deletion.
  It requires the committed complete DAG result and exact committed upload target.

All use immutable history and run/config/provider/MIME-bound digests. They are
model-free work: checkpoint usage records zero *model* cost, not free storage,
network or infrastructure. No new DynamoDB/Keyspaces table or public API is added.

## Recovery contract

The normal core deterministic helper is intentionally not used for upload/delete:
it can reclaim interrupted pure work, whereas an external operation may already
have happened. For upload and cleanup, a running reservation after any exception,
timeout or cancellation is uncertain and cannot be automatically reclaimed.
A reservation commit with a lost acknowledgment stops before remote dispatch;
a completion commit with a lost acknowledgment is resolved by reading its head.
Uncommitted remote outcomes need explicit reconciliation, never guessed retries.

Only the pure `dag_complete` marker may be reconstructed: replay goes through the
core's existing step checkpoints, preserving its paid-call uncertainty rules.
Once completion is committed, `execute()` returns its saved typed result without
upload, generation or publication, including after cleanup. Cleanup can be invoked
separately and replayed safely after a committed receipt. Starting cleanup bars new
upload acquisition for this run, even if cleanup's outcome later becomes uncertain.

Partial position publication remains usable when technique analysis fails. A missing
complete marker never permits ordinary cleanup. Failed/cancelled-run or revoked-owner
retention cleanup is separate internal-authority work, not an implicit action in
this coordinator. Explicit [saved-receipt reconciliation](analysis-provider-reconciliation.md)
is available under a live guard; no public reconciliation API or abandoned-upload
inventory is included.

`retry_failed=True` is forwarded only to the core's existing bounded known-failure
recovery. It neither retries uncertain model dispatch nor unlocks upload/delete.
Manual edits, acceptance and usage admission are outside this host; it consumes no
new admission record or user-use allowance during replay.

## Verification / next integration

From the engine worktree:

```sh
CHECKPOINT_CASSANDRA_HOST=127.0.0.1 ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 ../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/run_host tests/run_host -q --tb=short
```

Tests use generated real video/crops, actual local Cassandra/S3, installed core0.3
and the actual Gemini HTTP client with an intercepted provider transport. A complete
zero-technique DAG demonstrates ordered publication, durable replay and separate
cleanup. Fault tests cover competing reservations, before/after commit loss,
cancellation, revocation, changed identities, partial technique failure and a sink
failure that must not create completion. The sink is a fixture here; previous
publication/intake checks cover its production implementation independently.

Next: receipt-less orphan/expiry reconciliation, maintenance authority and active host integration;
provider model/schema/clock/rate canaries; durable queryable progress/usage projections.
Only then activate the new workflow. Rollback retains history and outputs, drains or
version-dispatches jobs, and never reinterprets/repeats paid attempts automatically.
