# Execution and host contract (core 0.3)

`execute_pipeline(run, config_bytes, journal=..., provider=..., sink=..., guard=...)`
is the complete **serial** stage sequence, not a deployed worker. No hidden model
fan-out, provider retry, extra admission/use debit, startup hook or fallback path.

Position scan → per-transition critique → normalize → position publication;
then independent technique scan → position/technique union → temporal critiques →
analysis worklist → per-item analysis → validation → normalize → technique publication.
Rejected rough position states do not become subsequent critique context.

Position critique defaults to 4000ms backward and 6000ms forward inspection,
clipped to selected scope. `max_call_attempts` defaults2 (maximum3); these fields
are part of the effective config. Changed model/prompts/config/core require a new
admitted run or exact version dispatch, never reinterpretation of saved work.

## Required production ports

- **Journal:** strong read plus atomic compare-and-swap of the current head and
  retained immutable revision history. Raw requests/replies and every failed attempt
  remain available privately. Externalize large payloads to private checksummed S3;
  never put unbounded prompt/video results into one DynamoDB row. The approved
  Keyspaces lifecycle/checkpoint authority is unchanged. Its existing plain
  overwriting stage helper is not this journal contract; adapter/storage capability
  verification is a required next step. The SQLite adapter is test-only evidence,
  not a proposed production migration. Active history cannot be silently expired.
- **Provider:** one dispatch, qualified source/crop checksum and ordered identity
  grounding, exact media window/time conversion, actual response/usage identity,
  and explicit unknown accounting. No SDK-internal retry that escapes attempt
  recording. `NotDispatched` may be raised only when the adapter can establish
  that no request left the process. Generic exceptions/timeouts are uncertain.
  Native provider compatibility, upload expiry and model/rate qualification are
  not established by fixture providers.
- **Guard:** mandatory current authorization, cancellation and worker-lease check,
  including cached replay. Called before steps/publication and final handoff.
  If revoked during a provider call, the returned private response/usage is retained,
  then the next guard prevents subsequent calls/publication. A long-call heartbeat
  and prompt cancellation are host supervision concerns, not simulated percentages.
- **Sink:** idempotent exact-revision publication through the existing durable
  producer outbox, with current authority checked on real sends. Cached publication
  is reauthorized by the guard without resending; a lost acknowledgment may repeat
  the same sink operation. Returning means its durable host boundary was reached,
  not that the account consumer staged all proposals. No automatic item acceptance.

The executor returns two typed component revisions only after both sink steps.
It does not write a successful engine lifecycle/terminal message. The owning host
must combine its execution result with durable publication/consumer requirements.
Positions already published remain usable when subsequent technique work fails.

## Checkpoint and retry semantics

Each logical node/item has a run-scoped stable step ID. Each attempt has a distinct
call ID; input digest binds the complete registered run, request, provider identity
and deterministic inputs. A CAS reservation with unknown usage precedes dispatch.
Concurrent workers cannot both claim a fresh model step.

The returned raw reply is saved as `responded` **before parsing**. Malformed JSON
still retains its actual provider usage. A committed response can be parsed again
after restart with no provider call; a succeeded step reuses its exact result.
Commit failures propagate, not masquerade as provider failures.

- Running model step after restart: dispatch may have happened; reconcile first.
- Timeout/generic provider exception or in-flight cancellation: unknown spend stays
  unknown; no implicit retry, even with `retry_failed=True`.
- Explicit non-dispatch or malformed returned output: a trusted host may request
  bounded failed-work retry. Previous receipts remain in history and spend totals.
- Pure/idempotent deterministic work: safe to recompute after an abandoned in-flight
  reservation; CAS fences stale finishers. Publication uses the same immutable
  revision/outbox identity. Completed upstream model work is not repeated.
- Missing/changed input, corrupted checkpoint or invalid usage identity: fail
  visibly; do not invent output, zero cost, or new request identity.

No external reconciliation endpoint, automatic retry loop/backoff, scheduler or
lease implementation is supplied by this package. Hosts must bound/reserve actual
model calls globally and per account before enabling it.

## Progress and costs

`StepCheckpointV1.progress` uses the existing shared `NodeProgressV1`. Per-item
records distinguish running from succeeded/failed/cancelled; separate zero-cost
stage summaries carry actual completed/total counts, including zero candidates.
No elapsed-time percentage or fake internal model progress is generated.

`report_execution(history)` requires all contiguous revisions for each included
step. It chooses the latest receipt of **each attempt**, not just the latest step
head, so failed paid attempts survive successful retry. It returns current per-step
progress and per-node/run `CostSummaryV1`, with no prompts/crops/provider raw text.
Unknown spend remains an explicit count alongside known nano-USD subtotal;
deterministic zero means zero model cost, not zero infrastructure overhead.
Pricing requires versioned rate evidence; no prices or membership charges are
invented here. A host must provide complete history for the run, authorize the
projection and connect it to durable usage storage and UI separately.

## Concrete host journal

The engine's opt-in [Keyspaces/S3 adapter](../docs/analysis-checkpoint-journal.md)
stores atomic heads/history and checksummed private payloads. It is outside this
portable package and does not change core0.3. Local Cassandra/S3 integration tests
exercise the installed executor against it. Production host wiring, lease fencing,
AWS policy/capability qualification and all-run inventory/projections remain open.

The host also has [exact-byte source/crop preparation](../docs/analysis-media.md),
tested with real local S3 and FFmpeg. It preserves source time and player order;
it is not the provider port, upload receipt lifecycle or production activation.

The next host increment provides an opt-in [Gemini REST provider](../docs/analysis-gemini.md)
with explicit file lifecycle operations, native request mapping and raw usage
receipts. Local HTTP fixtures and real S3 verify the boundary; Keyspaces upload
reservations, host wiring and model/schema/clock qualification remain gates.

An opt-in [durable run host](../docs/analysis-run-host.md) now connects those ports,
reserves external upload/cleanup separately from replay-safe calculations, and
stores a complete result after the core sink boundary. Active-worker lease/lifecycle
wiring, operational reconciliation and live qualification remain open.
