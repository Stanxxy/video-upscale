# Durable DAG progress reads (internal, opt-in)

`PositionRunDispatcher.progress(registered_run_uuid, max_item_reads=2000)` reconstructs
progress from existing Keyspaces checkpoint heads and their verified private S3
payloads. It opens no media, claims no lease, writes no state, sends no provider call
and does not alter usage. It can read an admitted run before lifecycle registration
or after completion. No new table, global checkpoint scan or public API is introduced.

This is a trusted server interface. The configured authority callback and stored-run
rehydration guard every read and the final return. A future HTTP/SSE adapter must
authenticate and authorize its actual viewer; merely knowing the stored owner's
identity is not viewer authorization. These helpers do not create an auth mechanism.

## Evidence and discovery

`bjj_analysis_observer.progress` is explicitly qualified for installed core **0.3.0**.
Fifteen known core/host summary keys are derived from existing stable step UUIDs.
Committed position scan, candidate batch and worklist outputs supply fanout identities:
position critique uses transition indexes; technique critique/analyze/validate use
their exact candidate IDs. No inference/replay is used to recreate missing outputs.

The internal frozen view contains run ID, observation time and ordered node records:
last recorded status, completed/total item counts and existing `NodeProgressV1`
records with per-execution sequence/attempt/timestamps. It excludes raw results,
prompts, private object references, receipt bodies and prices.

- No checkpoint → `not_started`; an undiscovered fanout total stays null.
- Committed discovery gives its real item count, including zero.
- Each succeeded item contributes exactly one completed item, independently of its
  checkpoint revision count or prior attempts.
- Even all finished items, or an empty fanout, require the durable summary before
  the node is labeled succeeded. A known empty fanout with no summary is not success.
- A recorded failed or cancelled item remains failed/cancelled. Unknown dispatch
  outcomes may be recorded as `failed / DISPATCH_UNCERTAIN`; this is not retry permission.
- Summary/identity/count disagreement, duplicate item IDs, malformed/corrupt artifacts,
  unavailable storage and revoked access raise errors rather than empty success.

The complete nonempty native DAG test compares every persisted execution key with
the projected keys; topology drift must be qualified before changing the frozen
supported version or adding a new core node.

## Consistency and limits

These are **sequential per-step observations, not an atomic run snapshot**. The
observation timestamp is read completion, not proof that all nodes shared one instant.
A source/summary race that cannot justify the topology fails explicitly so a caller
may request another read. No hidden polling/retry is performed. Client ordering must
use the existing per-execution sequence within its bound run, not a fabricated global
sequence or timestamp ordering across nodes.

Recorded `running` does not prove a worker is alive. The internal lifecycle
composition below adds observed heartbeat freshness; connection freshness and
independently available/delivered results still need service integration.
`dag_complete` is not consumer review-readiness or
human acceptance. Optional unrequested cleanup can remain not-started after the DAG
and lifecycle complete. No synthetic percentage, ETA or heartbeat is returned.

Item reads are bounded across all four fanouts, with a configurable limit from 0
through 2,000. Exceeding the budget returns an error, not truncated apparent progress.
Each checkpoint also retains existing byte limits, authority and integrity checks.
This correctness-first reader can make many authorized storage calls; a production
SSE polling cadence, latency/capacity budget and large-fanout paging/indexing strategy
must be qualified rather than silently increasing limits. No high-frequency stream
is mounted here. Historical costs remain in the existing full-history core reporting
path; latest heads cannot be used to price retries correctly.

## Lifecycle and heartbeat composition

`PositionRunDispatcher.status(registered_run_uuid, max_item_reads=2000)` wraps the
same progress read with two exact-key `job_lifecycle` reads at LOCAL_QUORUM. It
validates the stored video/principal/workflow identity and timestamp order. The
internal view contains the unchanged progress view plus recorded lifecycle state,
last heartbeat, lease expiry for RUNNING only, completion time and final observation
time. Worker owner tokens are used for consistency checks but never returned.

- No lifecycle row: `NOT_REGISTERED`; a pending row: `PENDING`. Neither claims work.
- `RUNNING`: heartbeat is `recent` only before the configured lease expiry; at or
  after expiry it is `expired`. Both preserve the recorded RUNNING lifecycle.
- `INTERRUPTED` and `COMPLETED`: heartbeat freshness is `not_running`, not expired.
- A recent heartbeat proves only a recent lease renewal, **not** model progress or
  guaranteed worker survival. An uncertain provider failure can coexist with a
  recently heartbeating RUNNING lifecycle and a recorded failed node. No automatic
  recovery, failure conversion, cancellation or usage change follows from this view.
- COMPLETED requires a succeeded durable DAG marker and either no cleanup attempt
  or succeeded cleanup. It does not imply SNS delivery, consumer intake, review
  readiness, human acceptance or finished billing.

Lifecycle reads bracket the node reads. Appearance/disappearance, owner or state
change raises `StatusObservationChanged`; callers may explicitly read again. Same
attempt heartbeat renewal is allowed, using the later heartbeat. Missing/corrupt
identity, unsupported lifecycle states, backwards/future timestamps and revoked
authority fail closed. No hidden reread, best-effort partial view or write occurs.
Only the position worker's existing PENDING/RUNNING/INTERRUPTED/COMPLETED contract is
supported; permanent FAILED/CANCELLED semantics require a separate approved extension.

Freshness is calculated after the final awaited authority check, so time spent in
storage/authorization cannot return an already expired lease as recent. As with the
worker, clocks must be synchronized; inconsistent timestamps raise errors. This
remains a sequence of observations, not a database-wide snapshot or proof against
changes after the final read. A transient lifecycle change can require another
caller-initiated request. A broken browser connection is a separate transport state,
not something this server-side reader can infer.

The existing current-run authority callback is still an internal execution-access
boundary, **not viewer authentication**. No HTTP/SSE route or polling loop is mounted.
The lifecycle read adds two key reads to the existing bounded checkpoint read cost;
production polling and large-DAG performance remain unqualified.

## Verification and remaining integration

```sh
CHECKPOINT_CASSANDRA_HOST=127.0.0.1 ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 ../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/run_host tests/run_host -q --tb=short
```

Native fixtures cover unstarted/completed reads, active and completed nonempty DAGs,
empty fanouts before/after summary commit, uncertain failure, revocation, corrupted
artifacts, missing runs, source/summary races, counts, duplicate identities, version
qualification and item budgets. Gemini HTTP is intercepted; storage, journal,
executor and publication are real disposable local dependencies.

Independent result publication, intake and proposal readiness can now be read through
the [internal result-status reader](analysis-result-status.md); it does not equate
SNS acknowledgement with consumer readiness or accepted annotations.

Next: authenticated browser delivery and composition of these observations,
durable admission/service startup integration and actual DAG animation/acceptance.
No production switch, live-model accuracy or UI completion is claimed.

The readers now ship in the separate [observer package](../analysis_observer/README.md).
The engine dispatcher and future hosts use the same installed implementation.
`AuthorizedRunObserver.status` additionally binds the actual verified viewer and
rechecks host resource authorization; it does not authenticate browser credentials
or expose the internal view as a public DTO. No HTTP/SSE route is mounted yet.
