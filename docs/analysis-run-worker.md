# Position DAG worker attempt (opt-in)

`service/analysis_run_worker.py` connects `DurableGeminiRun` to conditional ownership
and completion on the existing Keyspaces `job_lifecycle` table. It creates no schema,
starts no service and is **not mounted in production or the legacy scheduler**.

## Authority and lifecycle

`KeyspacesRunLease` accepts a trusted `RegisteredAIRunV1`, native Cassandra session,
explicit keyspace, live authority callback and finite lease/I/O timeout policy.
The callback must verify the authoritative registration and current owner/scope
access (including principal kind), not merely validate a caller-supplied model.
The lifecycle identity is run UUID as job ID, video ID, principal ID as user ID, and
the admitted workflow ID as pipeline kind. Checkpoints bind the complete registered
run/config separately. No new user authentication mechanism is introduced.

`register()` is an explicit conditional insert. Existing rows are validated, never
reset. Each lease object generates a unique, single-attempt token in the existing
`owner_instance_id` column; it is not the engine's shared instance ID.

- Normal acquisition claims only `PENDING`.
- Explicit `recover=True` can claim `INTERRUPTED` or an expired `RUNNING` row.
  It cannot claim a fresh running, completed, failed or cancelled row.
- Claims compare prior owner, state, heartbeat and immutable lifecycle identity.
- Heartbeats and terminal writes compare the same fields. A prior owner cannot
  overwrite a replacement's state, including when takeover happens after its read.
- Guards check current authority and ownership, then check time/closure again after
  the last authority I/O. Clock rollback, future heartbeat, missing/invalid row,
  expiry or ambiguous CQL responses are errors, not empty/successful reads.

Cassandra timestamps are normalized to milliseconds. Deploying workers must share
one lease policy and synchronized clocks with an operational safety margin; client
clock-based expiry is not a server-time lease. Driver retries are disabled and
conditional operations use LOCAL_SERIAL/LOCAL_QUORUM. Actual AWS Keyspaces behavior
and IAM still require qualification.

These are fenced lifecycle writes plus live dispatch guards, **not atomic fencing
of external HTTP calls**. An already dispatched request may finish after ownership
changes. Existing durable per-step reservations prevent an uncertain upload/model
call from being automatically repeated by its replacement. Raw spend/results must
still be retained where possible even if subsequent publication is forbidden.

## One supervised attempt

`run_position_attempt(lease, host_factory, heartbeat_seconds=..., recover=False,
retry_failed=False)` acquires ownership before opening media/provider resources.
`host_factory(guard)` is an async context manager that assembles the admitted host,
journal, media, provider and sink with that guard and owns their teardown. Adapt the
zero-argument guard for provider/media interfaces receiving the registered run.
The factory must not replace the leased run or pass an unguarded publication port.
The opt-in [registered-run dispatcher](analysis-run-dispatch.md) now supplies this
concrete guarded factory. The opt-in [new-run scheduler](analysis-run-schedule.md)
provides isolated durable discovery; public request routing remains absent.
Explicit dispatcher `ProviderActions` can wrap receipt recovery and requested cleanup
around DAG execution within this same heartbeat/teardown lifetime. They do not alter
lease acquisition or authorize terminal takeover; ordinary dispatch defaults stay
unchanged. Pending cleanup blocks dispatcher lifecycle completion without proof.

The heartbeat runs while media preparation, DAG execution and context teardown are
in progress. Completion is written only after the durable complete-DAG result and
successful resource teardown. Heartbeat/storage/authority failure or caller
cancellation closes dispatch permission before cancelling and draining the work.
Repeated cancellation waits for teardown; it does not abandon open media contexts.

On error, the supervisor leaves the lifecycle RUNNING until expiry; it does not
invent permanent FAILED/CANCELLED or usage/refund policy. Error/checkpoint inspection
and an explicit recovery command remain required. Recovery passes known-failure retry
permission separately to the core; it never converts an uncertain paid call into a
safe retry. A completed DAG whose lifecycle completion failed can be recovered by a
new lease and completed without new inference or publication.

This lifecycle COMPLETED means the durable core/sink boundary finished, not that the
consumer already made the account workspace review-ready. It does not update legacy
latest/recovery-index projections, SSE, DAG progress or user balances. Finished leases
cannot authorize provider cleanup; separately authorized cleanup/reconciliation is
still needed. No implicit deletion occurs on success or failure.

## Isolation and activation gates

Do not add these jobs to legacy `owned_jobs` or the legacy `ACTIVE`/calendar-hour
partitions of `job_recovery_index`: legacy heartbeat writes are unconditional and
its recovery dispatch does not recognize this workflow. The new-run scheduler uses
disjoint `POSITION_NEW_RUN_V1` fixed-shard partitions in that existing table and
routes only to this worker path. Production host wiring, projections and
reconciliation/retention tooling remain prerequisites for admission activation.
No polling loop or automatic takeover is started by these worker helpers.

## Verification

```sh
CHECKPOINT_CASSANDRA_HOST=127.0.0.1 ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 ../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/run_host tests/run_host -q --tb=short
```

The worker tests use actual local Cassandra lifecycle CAS and actual journal/S3/media
with a transport-intercepted Gemini provider. They cover claim races, stale-owner
completion and heartbeat rejection, renewal versus takeover, lost/cancelled claim
acknowledgments, exact expiry during authority I/O, closure during authority I/O,
clock rollback, heartbeat failure, repeated cancellation/teardown, partial results,
and recovery after lifecycle completion loss without another model request. No live
model, AWS production, accuracy, user-facing progress or admission claim follows.
