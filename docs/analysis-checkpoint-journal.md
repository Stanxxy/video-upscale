# Position DAG checkpoint journal (opt-in)

`service/analysis_checkpoint_journal.py` implements the installed core0.3 journal
port. It is not mounted in the production runner. The host supplies the registered
run, a verified Cassandra/Keyspaces session, a private S3 client/bucket, and a live
authority/cancellation/worker-lease guard. No credentials, provisioning, scheduler,
provider implementation or legacy checkpoint conversion is hidden in this adapter.

Read/integrity/history operations are inherited from the separately installable
`bjj_analysis_observer.checkpoints.KeyspacesCheckpointReader`. That package rejects
non-read CQL and contains no checkpoint upload, conditional-write or provisioning
methods. The engine subclass retains its original write execution, upload and CAS
methods; HTTP hosts must use the reader, not import the engine writer.

## Storage and write protocol

The additive backend migration is
`infrastructure/keyspaces/migrations/010_analysis_execution_checkpoints.cql`.
One partition `(run_id, step_id)` contains one checkpoint model:

- `revision=0`: mutable head pointing to the latest committed record revision.
- `revision>=1`: immutable successive checkpoint revisions, including reservations,
  provider responses/usage, parsing failures and completed results.
- CQL stores only bounded revision, input digest and artifact pointer/digest fields.
  Full JSON is private S3 data, capped at 16 MiB per checkpoint.

Each write uploads a checksummed immutable object first, then atomically inserts
history and conditionally advances the head in one same-partition logged batch.
An explicit false conditional result is contention; transport/ambiguous failures
propagate. AWS documents support for logged batches with conditional statements;
the local tests exercise native Cassandra 4.1, not AWS account capability or IAM.
See [AWS batch semantics](https://docs.aws.amazon.com/keyspaces/latest/devguide/batchStatements.html).

Keys are pinned to `analysis-checkpoints/v1/{analysis_id}/{run_id}/{step_id}/{revision}/{sha}.json`
in the injected bucket. Uploads request private ACL, AES256 encryption and
`IfNoneMatch=*`. Only an exact verified pre-existing object is reusable. There is
no presumed-success recovery, public URL or raw payload in CQL. Production bucket
ownership/encryption policy compatibility must be qualified before enabling this.

The adapter uses LOCAL_QUORUM reads/writes, LOCAL_SERIAL conditional writes and
FallthroughRetryPolicy. It does not use the old execute wrapper that swallows read
failures. The host must configure verified TLS/authentication, request timeouts,
SDK retry behavior and speculative execution consistently with the journal contract.

## Recovery, access and history

Reads verify namespace, SHA-256, strict JSON, run/step/revision and input identity.
A missing head with retained history is corruption, not a fresh step. A second
head read distinguishes this from a concurrent first commit. History reads are
bounded to 1–100 records/page and reject gaps or changed inputs. Page cursors are
per-step revisions, not an all-run inventory or a cross-page snapshot token.

The guard runs around reads and before/after conditional commits. Revocation while
a driver write is blocked denies the return, but keeps the committed audit record.
Task cancellation cannot undo a blocking driver commit: a restart must read the
head and preserve the executor's uncertain-dispatch stop. Atomic host lease fencing
and provider pre-dispatch authority checks remain separate activation requirements.

Committed responses allow parsing/replay without a new provider request. The core
keeps malformed paid responses and prior attempts, so cost reporting must consume
complete history for every run step. Run/step inventory and durable DynamoDB
progress/cost projections are not implemented here; the journal is not a user API.

No TTL or automatic deletion is introduced. Failed/contended writes can leave
unreferenced private S3 objects. Retain these until a scoped reconciliation/retention
policy exists; never delete history as a way to retry paid work. Storage billing and
snapshot growth need production-size measurement. Annotation tables remain separate
from this execution-only journal.

## Verification and rollout

From the engine worktree, with the backend environment's installed core0.3/shared1.5:

```sh
CHECKPOINT_CASSANDRA_HOST=127.0.0.1 ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 ../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/checkpoint_journal tests/checkpoint_journal -q --tb=short
```

Tests create/delete only UUID-owned local keyspaces and buckets. They cover real
CAS contention, atomic history, S3/CQL lost acknowledgments, malformed-response
restart/retry, cancellation during real driver I/O, revocation, corrupt history,
DDL parity and a full zero-technique DAG replay without extra fixture-provider calls.
Fixtures do not establish Gemini compatibility, media grounding or model accuracy.

Before activation: provision additive DDL with approved retention/access policies;
canary conditional batches, consistency, TLS/IAM and S3 conditions/encryption against
the actual AWS target; wire host lease/lifecycle, verified media/provider ports,
inventory/projections and operational reconciliation. Code rollback leaves new
history intact and drains/version-dispatches in-flight jobs. Old code must never
reinterpret a new workflow checkpoint. No production workflow switch is included.
