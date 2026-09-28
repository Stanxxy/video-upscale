# Result delivery and review status (internal, opt-in)

`PositionRunDispatcher.result_status(registered_run_uuid)` reads saved evidence for
positions and techniques independently. It neither calls Gemini nor sends messages,
stages proposals, accepts annotations, claims a lease, changes usage or writes status.
There is no new table, persisted schema, HTTP route or service-startup hook.

Each component exposes:

- `publication_state`: `not_prepared`, `prepared`, or `acknowledged`. Acknowledged
  means the saved SNS acknowledgement, **not** confirmed consumer delivery.
- `declared_item_count`: the immutable publication's item count, including zero;
  null when no publication has been observed.
- `published_at`: saved SNS acknowledgement time, null without that receipt.
- `review`: existing `ComponentAvailabilityV1`, with `not_received`, `received`, or
  `ready`, and the existing item/proposal counts and intake/readiness timestamps.

`ready` requires the consumer's durable proposal-staging completion marker. Merely
finishing the DAG, preparing an outbox entry, receiving an SNS acknowledgement, or
ingesting an artifact is not enough. An empty technique result becomes ready only
after an explicit zero-proposal readiness marker. Positions need not wait for the
technique lane or whole-run completion. These counts describe the original staged
output, not today's pending suggestions or human-accepted annotations; per-item
review and accepted-statistics behavior remain unchanged.

The publication and review axes intentionally remain separate. A lost SNS response
can leave publication `prepared` even though delivery succeeded and the consumer
has already staged review proposals. This valid state neither hides readiness nor
authorizes a model retry. Existing delivery recovery handles saved-message replay.

## Evidence and consistency

The reader first obtains one strongly consistent registered-run row. Both consumer
receipts and readiness markers use **that same row**, preventing concurrent intake
and staging from looking like an orphan marker. It then reads each outbox entry
through the existing store, which checks the publication/ack digests, run binding
and actual private S3 result artifact. Any observed consumer receipt must match that
exact publication message and result digest. Missing/corrupt artifacts, orphaned
consumer evidence, mismatched receipts, unsupported complete-result intake, future
timestamps, unavailable storage or revoked access raise errors; none becomes an
empty or optimistically successful status.

Publication observations may be newer than the earlier consumer row; newly staged
results can appear on the next request. This is not an atomic snapshot across S3,
DynamoDB and Keyspaces. No global sequence, hidden polling, reread or fallback is
introduced. The observer does not scan proposals or reconstruct a readiness marker;
it relies on the existing retention invariant that proposals/decisions remain with
the workspace and cleanup revokes the parent before removing children.

Current registered-run authority is checked around reads and before returning.
The output excludes private artifact references, prompts, message bodies, SNS IDs,
worker tokens and credentials. This is still a trusted internal interface: its
execution-authority callback is **not browser viewer authentication**. The existing
account-host availability API continues to authorize the actual viewer and video;
it is not replaced or bypassed here.

## Verification and next boundary

```sh
CHECKPOINT_CASSANDRA_HOST=127.0.0.1 ANNOTATION_TEST_ENDPOINT=http://127.0.0.1:4566 ../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest --confcutdir=tests/run_host tests/run_host/test_result_status.py -q --tb=short
```

Native fixtures exercise real DAG publication, SNS→SQS delivery, consumer intake,
proposal staging and the read interface. Provider HTTP is intercepted; persistence,
queue delivery, proposals and readiness writes are real disposable local resources.

Next: integrate progress/lifecycle and result observations into authenticated browser
transport, with ordering/connection freshness and a qualified polling/load budget.
Durable admission/service startup and the existing rollout gates remain open. No
browser UX acceptance, live model qualification or production activation is claimed.
