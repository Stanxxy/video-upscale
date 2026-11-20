# Registered new-run scheduling (opt-in)

`PositionRunSchedule` in `service/analysis_run_schedule.py` durably discovers
already-admitted position runs and invokes the registered-run dispatcher. It creates
no table, admits no browser input and starts no background task on import.

## Account/session runtime (approved September 14)

Deployment composition uses `OwnerPositionRunSchedule` through
`partitioned_engine_service_runtime`, with explicit account and session dispatchers.
The single-store namespace below remains available to its isolated harness only;
the partitioned runtime never reads or retires those old hints.

| Signed host credential | Registration/result stores | `recovery_state` | `heartbeat_bucket` |
| --- | --- | --- | --- |
| account | account dispatcher only | `POSITION_NEW_RUN_ACCOUNT_V1` | `position-new-account-v1#00` … `#15` |
| session | session dispatcher only | `POSITION_NEW_RUN_SESSION_V1` | `position-new-session-v1#00` … `#15` |

The private request still contains only the stored run UUID. Verified credential
ownership chooses the store before lookup; there is no browser owner field,
cross-store probe, scan, shared registry or automatic migration. A stored owner
that disagrees with its lane is rejected. Run UUID derivation is unchanged.

Both lanes have their own schedule and result-delivery workers. They must have
distinct registration and publication table names. Both dispatchers share **one**
execution semaphore, so adding the second lane does not double DAG concurrency
within this runtime. This is not a fleet-wide permit or a legacy-engine limit.
Each publication is owner-checked and currently authorized before delivery.

Scheduling queries join their native driver work even after cancellation before
returning to the client-owning host. Worker/listener failure uses the existing
supervised runtime shutdown; parent clients must remain open through drainage.
Observers receive `(owner_kind, page)`, and all four workers are supervised.

This factory is opt-in and does not mount itself in the engine entrypoint.
Existing hints require an explicit, separately reviewed cutover if they are ever
to be moved; restarting the new runtime is not a migration. Host composition,
live model qualification and deployment gates remain required.

## Discovery is a hint, not execution authority

The existing Keyspaces `job_recovery_index` supplies the physical discovery index.
Immutable registration, current application authorization and conditional
`job_lifecycle` ownership still decide whether a run may execute.

| Existing column | New-run namespace meaning |
| --- | --- |
| `recovery_state` | `POSITION_NEW_RUN_V1`, disjoint from legacy recovery states |
| `heartbeat_bucket` | `position-new-v1#00` through `#15`, selected by run UUID modulo 16 |
| `last_heartbeat_at` | Immutable registration timestamp, normalized to CQL milliseconds |
| `job_id` | Registered run UUID; second clustering key breaks timestamp ties |
| `video_id`, `job_state`, `owner_instance_id` | Exact admitted video, `PENDING` hint, empty owner |

Despite inherited column names, these are not live heartbeat projections. Stable
shards keep older pending work discoverable without an age-based lookback window.
Legacy readers use their own state/calendar-hour partitions; do not register these
runs in legacy `owned_jobs` or `ACTIVE` recovery partitions.

## Admission handoff and processing

A trusted internal caller invokes `enqueue(registered_run_uuid)`. The scheduler
resolves the stored registration, checks supported workflow/core versions and current
authority, conditionally inserts the hint and reads it back for exact validation.
No lifecycle row or model call is required: a scheduler can resume discovery after
a process dies between enqueue and dispatcher registration. A lost enqueue
acknowledgment is retried with the **same registered UUID**, not a new admission.
If a concurrent scheduler completes and retires the hint before confirmation,
enqueue reports an unconfirmed handoff; callers must reconcile that run's state.

This does not make application admission and enqueue atomic. Host integration must
durably retry/reconcile an admission whose enqueue handoff never completed.

Each discovered candidate is rehydrated and validated before lifecycle inspection:

- Absent lifecycle or `PENDING`: normal dispatch only, with the existing lifecycle
  conditional claim preventing duplicate execution by competing schedulers.
- Fresh `RUNNING`: report `running`, without takeover.
- Expired `RUNNING`, `INTERRUPTED`, `FAILED`, `CANCELLED` or `AWAITING_CORRECTION`:
  report `explicit_recovery_required`. No automatic recovery/paid retry is granted.
- `COMPLETED`: reauthorize and confirm completion, then remove only the exact
  discovery hint. Lifecycle, checkpoint, usage and result history remain intact.
- Invalid registration, revoked access, malformed hints or uncertain failures:
  report `unconfirmed` and retain the entry for inspection/reconciliation.

A completed hint left behind by failed retirement can be removed on the next pass
without inference. This completion means the durable DAG/publication-intent boundary,
not SNS delivery, consumer review-readiness or human acceptance.

## Loop and operational limits

`process_page` reads at most 100 candidates plus one lookahead. A tuple cursor
includes timestamp and raw job ID so an invalid UUID or equal-timestamp neighbor
does not prevent traversal. Per-candidate errors do not block later candidates;
query errors preserve the prior cursor and report `discovery_unavailable`.

`run(stop, observe=..., interval_seconds=..., page_size=...)` explicitly starts a
round-robin loop over 16 shards. Cursors advance between passes and reset at EOF;
observer failure propagates rather than pretending monitoring succeeded. The
required async observer must be supplied by the host. Reports are internal values,
not a persisted/public schema. Graceful stop finishes the current attempt; task
cancellation propagates through existing worker fencing and resource teardown.

Dispatch is **serial through the full DAG** in this initial loop. The configured
interval (at most 60 seconds) is a delay between passes, not a discovery-latency
guarantee. Deployment must qualify throughput/fairness and worker capacity. Retained
failed/revoked/corrupt hints can grow fixed partitions; capacity, monitoring and an
approved retention/remediation policy are production gates, not implicit deletion.

No HTTP route, service-lifespan mount, quota debit, automatic provider cleanup,
permanent-failure policy or legacy production switch is introduced. Admission
handoff, upload/delete reconciliation, progress projections and supported host
authority wiring remain open. Local Cassandra validates the CQL behavior, not AWS
Keyspaces tuple-range/LWT capability, IAM, clock safety or production capacity.

## Local verification

```sh
CHECKPOINT_CASSANDRA_HOST=127.0.0.1 ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 ../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/run_host tests/run_host -q --tb=short
```

Scheduler tests create the existing recovery-index migration in a disposable native
Cassandra keyspace. Actual DynamoDB/S3, media preparation, lifecycle/checkpoints and
durable publication run with an intercepted Gemini HTTP transport. Cases include
restart discovery, claim races, lost enqueue acknowledgment, equal-timestamp paging,
revocation, uncertain paid work, completed-hint repair and dispatch cancellation.
