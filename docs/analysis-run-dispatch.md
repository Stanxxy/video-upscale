# Registered position-run dispatch (opt-in)

`PositionRunDispatcher` in `service/analysis_run_dispatch.py` supplies the concrete
resource factory missing from the leased worker. A trusted internal caller supplies
only a registered run UUID plus explicit recovery/known-failure retry flags. The
dispatcher loads immutable registration through `DynamoProposalStore`, rejects
uninstalled workflow/core versions and rechecks current access before claiming.
There is no legacy fallback or caller-supplied prompt/model/run payload.

Server-owned construction supplies the existing proposal and publication stores,
native Cassandra session, S3 client, explicit private bucket allowlist and scratch
root, source size limit, native MIME, credential, qualified model/version map,
authority callback, lease/heartbeat policy and bounded concurrency. Clients and
their credentials/timeouts remain application-owned. MIME must match the trusted
source; this helper does not infer it from a filename. Model qualification entries
must be backed by live canaries before activation, not copied from fixture names.

Capacity is acquired before the lease and media. A waiter rechecks access after
capacity becomes available. The shared dispatcher instance bounds simultaneous
media/provider contexts; deployment still owns process-level resource limits and
clock/lease policy. Separate instances do not share an in-memory semaphore.

The factory wires the same lease guard into source/config/crop preparation, provider
calls, checkpoint journal and result sink. It checks all six configured model IDs
against qualification before upload. Provider HTTP closure and local media teardown
are supervised on success, errors and cancellation. S3/Cassandra clients remain open
for the application to close at shutdown.

The sink acknowledges `DynamoResultPublicationStore.prepare`: immutable private
component artifacts and a durable publication intent. It does **not** acknowledge
SNS delivery, consumer review-readiness, human acceptance or a current-timeline edit.
Position output can be stored before technique work completes. A failed technique
publication preserves positions; explicit safe recovery reuses paid results and
completes the missing intent without new model calls.

Registration identity and workspace access are rehydrated on guards. The additional
application authority callback is mandatory for current video ownership, cancellation
and other host policy; the stored principal is not a substitute for authentication.
An external request already sent when access changes may still finish. The provider
retains its private response evidence, while subsequent publication is denied.

## Explicit provider actions inside the leased attempt

Trusted internal callers can pass `provider_actions=ProviderActions(...)` to
`dispatch`. These are in-process instructions, not a public request, stored admission
schema or permission to retry unknown model calls. Defaults remain unchanged.
Options apply to this invocation, not a newly persisted cleanup policy; the host
must explicitly request cleanup on the attempt that will finish the run.

| Option | Behavior |
| --- | --- |
| `cleanup=True` | Delete the committed provider upload only after the complete DAG and durable outbox intents, before resource teardown and lifecycle completion |
| `upload_receipt=ref` | With explicit `recover=True`, reconcile a reserved upload from exact private evidence before continuing the existing DAG checkpoints |
| `cleanup_receipt=ref` | With `recover=True` and `cleanup=True`, reconcile the exact saved deletion outcome before replaying the completed DAG/cleanup |

Only one receipt operation can be supplied per attempt. Invalid combinations fail
before registration/claim. Registered UUID/version/current-authority checks and the
existing lease claim still apply: active running and terminal runs cannot be seized.
The receipt is evidence, not permission; the host validates its exact run/operation,
target and bytes. A recovered upload may proceed to previously unexecuted DAG nodes;
it does not repeat the upload or bypass uncertain-model checkpoints.

The dispatcher wraps its real host within the existing worker heartbeat lifetime.
Receipt recovery, DAG execution and requested cleanup all run with the same live
guard. The worker finishes the lifecycle only after those actions and supervised
resource teardown succeed. Failure/cancellation retains the existing RUNNING-until-
expiry policy for explicit recovery, with no invented refund/permanent-failure policy.

Once a cleanup reservation exists, omitting cleanup options does **not** permit an
uncertain cleanup to be ignored during lifecycle completion. Recovery without its
proof remains blocked; a saved success can finish with no additional DELETE or model
call. Partial positions remain available if technique work/publication fails, and no
cleanup is started before both result-publication intents are durable.

Cleanup is not retroactive: if a default attempt already reached COMPLETED without
requesting cleanup, this path cannot reopen it. Separate terminal/revoked-run
maintenance authority is still required. Ordinary new-run scheduling supplies none
of these actions and never automatically starts recovery or cleanup.

## Remaining activation work

An internal [durable progress reader](analysis-progress.md) now rehydrates exact
checkpoint states/counts without claiming a worker or opening media. Browser delivery,
lifecycle/heartbeat composition and UI rendering are still separate integration work.

This is a callable internal dispatcher, **not an HTTP route, queue consumer or durable
job-discovery scheduler**. No background loop, legacy heartbeat/recovery registration,
automatic takeover, automatic cleanup, admission, quota debit or production switch is introduced.
Completed runs are not reclaimed by this worker; results remain available through
stored artifacts/intents. Failed attempts still use the prior worker's expiry and
explicit checkpoint-recovery rules.

Saved result delivery now has an opt-in [recovery supervisor](result-delivery-supervisor.md).
New-run discovery now has an opt-in [isolated scheduler](analysis-run-schedule.md).
Next: admission handoff and service-lifespan wiring, upload/delete
reconciliation, progress/usage projections, supported host authority wiring,
and live provider/AWS qualification. Public request admission and production activation
must wait for those boundaries. This code does not broaden trial or account access.

## Local verification

```sh
CHECKPOINT_CASSANDRA_HOST=127.0.0.1 ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 ../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/run_host tests/run_host -q --tb=short
```

Tests use actual disposable DynamoDB registration/outbox/consumer intake, Cassandra
lifecycle/checkpoints, S3 artifacts and FFmpeg media; only Gemini HTTP transport is
intercepted. The zero-technique DAG verifies both durable lanes without auto-editing
the workspace. Additional cases cover missing/future-version runs, access revocation
before and during work, queued access checks, unqualified models, scratch teardown,
partial outbox failure and explicit recovery without further HTTP calls.
