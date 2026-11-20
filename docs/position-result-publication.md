# Position-analysis result publication (s39, additive)

This is a trusted normalized-result delivery seam, not an activated inference
workflow. Legacy `SNSPublisher`, job scheduling, Keyspaces lifecycle and VLM Studio
are unchanged. No automatic worker starts on import. Use the packaged s39 shared
library (1.5.0) before explicitly constructing this adapter.

## Sequence

1. Admission has already registered the immutable run/baseline. The portable core
   produces a strict, single-lane `AnalysisRevisionV1` with stable item/revision
   identities, original owner/baseline and absolute source milliseconds.
2. `DynamoResultPublicationStore.prepare(run_id, component, revision)` validates
   the run and current workspace authority, then writes the bounded private
   component artifact. Positions must cover the whole selected scope; techniques
   may be empty. No padding/coercion/raw-Gemini conversion is done here.
3. A DynamoDB transaction reserves the next run message sequence, exact lane
   message hash, component mode and immutable outbox intent. SNS sees nothing
   before this transaction confirms. A failed commit may leave a private orphan;
   same-output retry recovers without provider work.
4. `ResultPublisher(outbox, sns_client).publish(run_id, component)` reloads and
   checks the stored intent/artifact/current authority and sends its exact message
   body to its pinned standard SNS result topic. Changing topic requires explicit
   operator migration; configuration drift fails closed.
5. After SNS returns a message ID, a conditional acknowledgment removes the
   pending index fields. A lost send/ack response may cause duplicate delivery,
   always with the same application message ID, sequence, timestamp and artifact.
   SNS acceptance is not backend readiness or run success.

Consumer intake atomically checks the reserved message hash. Foreign messages
cannot win after reservation; existing externally received declarations cannot be
silently adopted by this producer. Complete-result mode is mutually exclusive.
Independent components may be prepared/received in either order. The counter is
checked against persisted peer intent so a reset cannot silently duplicate sequence.
Future progress/terminal publishers must allocate from the same run counter, not
invent independent sequence streams. No such publisher is implemented here.

## Storage and recovery

`ResultPublicationTable.with_prefix(prefix)` defines one additional model-specific
PAY_PER_REQUEST table: `{prefix}_result_publications`, key `(run_id, component)`.
The record is one immutable `ResultPublicationV1` plus optional
`ResultPublishedV1` acknowledgment; artifact bytes remain private S3. A sparse
`pending_publications` keys-only GSI uses 16 shards and timestamp/message ordering.
Current annotation tables do not change. Original registered-run content/hash is
unchanged; sequence/mode/expected-message fields are infrastructure metadata.

`publish_page(shard, after=cursor, limit=20)` performs one bounded recovery page.
The GSI can be stale: every key is rehydrated through strong base reads, and
already-acknowledged records do not resend. Restart from the first page on each
new sweep; an earlier insertion must not be permanently skipped by a high watermark.
Concurrent sends remain safe duplicates, not exactly-once sends or billed AI calls.

The original page helper propagates errors. The opt-in
[delivery supervisor](result-delivery-supervisor.md) now adds paced shard pagination,
individual failure reporting, live authority checks and cancellation-safe sends.
Application lifecycle mounting, operator alerting/repair and retention still remain
required before activation. No pending row is silently dropped, acknowledged or downgraded
to the legacy workflow. Revoked/expired workspaces cannot prepare or publish through
the adapter; deletion must revoke parent access before deleting retained children.
Consumer account host separately rechecks real video ownership.

Production gates: table creation/IAM/PITR/encryption/retention, private S3 policy,
engine-only SNS Publish, SNS-source-only SQS policy, dedicated topic/subscription,
consumer deployment ordering, package release pins, supervisor recovery and canary.
Deploy the new consumer validation before enabling producers; stop/drain producers
before rolling consumers back. Do not remove pending data during code rollback.

## Local checks

The [portable core package](../analysis_core/README.md) now supplies strict
normalizers, candidate union and six provider-neutral prompt contracts. It is not the full DAG or
a running model adapter. Install its wheel before the backend acceptance suite;
the core integration test feeds normalized outputs into this publication seam.

The new module has no ML/server imports; its narrow tests use the backend's installed
shared wheel without changing the existing engine environment:

```sh
# engine worktree cwd
../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/result_publication tests/result_publication -q
```

Full persistence/wire integration lives at the owning backend's existing LocalStack
boundary. `ENGINE_RESULT_PUBLISHER_FILE` is an explicit test artifact path, not a
runtime sibling import or shared-library source injection:

```sh
# backend worktree cwd; only UUID-owned disposable resources
ENGINE_RESULT_PUBLISHER_FILE=/Users/stanliu/bjj-proj/whole-video-analysis-wt-s39/service/result_publication.py ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 .venv/bin/python -m pytest tests/annotation_persistence -q --tb=short
```

Real local SNS/SQS routing is tested through account intake to review readiness.
Live AWS signatures/IAM are not proved by LocalStack; signed-fixture consumer tests
are a separate gate. No model call, account admission or raw-output normalization
is claimed by the delivery test.
