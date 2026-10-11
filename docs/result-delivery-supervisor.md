# Result-delivery recovery supervisor (opt-in)

`ResultDeliverySupervisor` in `service/result_delivery_supervisor.py` delivers
existing immutable result intents. It cannot start inference, create admission,
change annotations, debit usage or delete provider uploads. Nothing starts on import
and it is not mounted in the production service lifespan.

## Operations

- `publish_one(run_id, component)` rehydrates the registered run, checks current
  application authority, verifies the stored intent/artifact and sends the exact
  pinned message through the existing publisher. A saved acknowledgment is reused.
- `recover_page(shard, after=..., limit=...)` processes one bounded pending-index
  page. Each entry is decoded independently and every failure is reported without
  stopping valid neighbors. Query failure preserves the prior cursor and returns a
  page error, never a successful empty page.
- `run(stop, observe=..., interval_seconds=..., page_size=...)` explicitly runs a
  paced loop, one page for each of 16 shards per pass. Cursors advance across passes,
  including past failed entries, and reset at EOF. A restart begins at the start of
  the durable index. The required async observer receives each internal page report;
  observer failure stops the loop rather than hiding an operational stall.

The shared store's additive `pending_candidates_page` returns raw DynamoDB index
keys with their cursor for individual validation. They are untrusted discovery hints,
not send authority. The existing strict `pending_page` API remains unchanged.
Use the rebuilt shared library with this method before constructing the supervisor;
do not inject sibling source paths or fall back to a stale package.

## Failure and cancellation semantics

Reports use `acknowledged` or `unconfirmed` plus bounded error codes; they never
include exception messages, source URLs, credentials or model output. Invalid index
entries may lack a valid run/component identity and are reported in page order.
Reports are internal values, not a new database or public schema. The application
must persist/alert on the observer output according to its operations policy.

Unconfirmed does not necessarily mean still pending: SNS or the acknowledgment
transaction may have succeeded before an error. Subsequent reads resolve durable
state. An uncertain send may be delivered again with the same application message
ID, sequence, timestamp and artifact. Consumers must deduplicate; this is not
exactly-once delivery. An acknowledgment is not consumer review-readiness.

Cancellation before dispatch prevents a new send, including one queued in a busy
thread executor. Cancellation after the synchronous send begins waits for it to
finish and preserves its acknowledgment if authority still permits. Repeated
cancellation cannot abandon the in-flight thread. A graceful stop completes the
current item and resumes an unexhausted page from its previous cursor; no unvisited
key is skipped. Server-owned SNS client timeouts/retry policy must bound shutdown.

Current host/video/cancellation policy is a mandatory authority callback, separate
from the finished inference lease. It is checked before sending and acknowledgment.
An already dispatched network request cannot be atomically revoked. Denied access
does not authorize dropping the saved intent or bypassing workspace revocation.

## Limits / activation

Pending inventory remains durable in DynamoDB; in-memory pagination is not another
retry database. Failed/corrupt rows are reported and retained, not automatically
quarantined, deleted or marked successful. A damaged index entry may require operator
repair; other entries/shards continue. Empty eventual GSI results are not proof that
all work is globally drained. Production metrics, alarms, retry pacing, retention and
manual repair procedures still need wiring into the application's lifecycle.

New-run discovery and paid-dispatch scheduling, upload/delete reconciliation,
progress/cost projections, live AWS/provider canaries and production activation
remain separate slices. No UI, billing or live model qualification is claimed here.

## Verification

Focused tests cover bounded cursor advancement/reset, item/query failures, access
denial, observer failure, graceful stop and pre-send/queued/in-flight cancellation.
Local integration runs a real stored DAG into DynamoDB/S3 outbox, SNS→SQS and real
component intake; the Gemini transport alone is intercepted. A lost SNS acknowledgment
redelivers the exact saved envelope without inference. A persisted malformed UUID
does not block valid neighboring entries or later pages. No live AWS send occurred.
