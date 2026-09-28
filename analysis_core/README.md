# Portable analysis core — checkpoint-aware DAG

`bjj-analysis-core` 0.3.0 is an installable, provider-neutral package. It contains
six prompt/request contracts (position scan/critique and technique scan/critique/
analyze/validate), deterministic candidate union and pure normalization into the approved shared production
position/technique revision models. The full serial executor now connects these
stages through explicit provider, checkpoint and publication ports. It is **not
activated in production**; concrete production adapters remain required.

This adapts the position-timeline experiment at VLM Studio baseline `b362d8a`
(`feature/qa-vlm-studio`). It is not a verbatim extraction or accuracy/parity claim:
production requires standardized position-relative roles, absolute integer
milliseconds and timestamped outcome evidence that the old lab payloads lack.
Old top/bottom or seconds payloads are not silently upgraded. Lab files are unchanged.

## Boundary and use

1. The trusted host registers a run using this package's `CORE_VERSION` and
   `WORKFLOW_ID`, ordered athlete bindings and private crop/source references.
   Persist exact `CoreConfigV1` JSON bytes, including `prompt_bundle_sha256()`.
   All six nodes default to high thinking; model, media resolution and FPS must
   be explicit. Provider/model capability validation is the future adapter's job.
2. Call `prepare_call(run, config_bytes, node_id, input_window=..., context=...)`.
   It checks the admitted artifact digest, bundled prompts/taxonomy/output schemas,
   core/workflow version, typed stage inputs and scope. It returns system/initial
   prompts, output JSON schema and node parameters without network I/O.
3. The host verifies source/crop artifact bytes, attaches media to its provider
   request and retains the actual request, raw response, usage and stage checkpoint.
   Private artifact locators are not inserted into text prompts. There is no
   Gemini SDK dependency in this package; the executor calls only its supplied
   provider port, which must enforce verified media and one-dispatch semantics.
4. `parse_response(prepared, response_text)` rejects malformed/duplicate-key JSON,
   code fences, nonfinite numbers, schema violations, wrong candidate/transition
   identity and out-of-window responses. It is structural stage validation, **not
   sufficient validation for publication**.
5. Call `normalize_positions(run, scan, critiques, completed_at=...)` or
   `normalize_techniques(run, candidates, decisions, completed_at=...)`. Persisted
   host completion time makes replay deterministic. These functions enforce final
   shared semantic validators, original baseline/edit version, run-owned IDs and
   unreviewed AI provenance. Send their single-lane revisions to the existing
   durable result publication adapter; never publish parsed stage DTOs directly.

## Discovery and temporal handoff

`technique_scan` observes the full scope independently: supplying position context
or a partial scan window fails. It returns ordered movement descriptions and rough
bounds, not classified techniques. Empty scans and overlapping attempts are valid.

`build_candidates(run, config_bytes, scan, positions)` unions all scan findings
with critiqued observed-to-observed changes in position, roles or subposition notes.
Reset/cutaway boundaries and evidence-only repeats do not create attempts. Like
the lab, a one-to-one nearest pairing enriches one scan with one transition; it
does not drop/relabel scans or collapse several transitions into one broad event.
Ties use transition then scan order. Synthetic descriptions retain before/after
subposition notes, including `other` geometry. Batch counts and sources are explicit;
`provider_call_count=0` describes this stage, not a persisted financial cost record.

`CoreConfigV1.candidates` pins editable deterministic parameters in the same exact
admitted artifact: transition pre/post 4000/2000ms, merge tolerance 2000ms, technique
critique back/forward inspection 6000/4000ms. Only observation windows are clipped
to scope. Padding is not blindly appended to final technique clips. Candidate IDs
bind run, effective config, complete position revision and candidate payload, so
changed priors/config cannot reuse old critique decisions.

Use `critique_window(...)` to prepare each `technique_critique` request, then
`resolve_temporal_critiques(...)` to require exactly one explicit decision per
candidate and obtain the surviving analysis worklist. A rejected non-event has
null timing; failed attempts remain confirmed. `window_exhausted` requires the
last inspected time as the end and exposes `OUTCOME_NOT_VISIBLE_BY_WINDOW_END` in
the prompt. It does not imply automatic extension, another provider call, or a
known outcome. Retain all candidate/decision records, including rejects, in host
checkpoints; the worklist is not the audit log.

Each work item is the `technique_analyze` context and its exact corrected bounds
are the observation window. Analysis must preserve those bounds. Known outcome
claims still need visible timestamped evidence; otherwise report unclear.
Technique critique/analyze/validate use fallible before/after context recomputed
at the relevant candidate/corrected/proposed bounds. End-point context uses the
state at that time (or the last state at scope end); occupancy remains half-open.
If validation subsequently changes bounds, recompute context again for downstream
display rather than retaining the earlier context. No timeline is an event gate.

Version 0.3.0 deliberately rejects registered 0.1/0.2 runs/configs. No resume-time
upgrade or old-prompt replay is implied; production dispatch must use the exact
recorded package version. This package has not been activated in either host.

Positions require one explicit critique decision per rough boundary and partition
the entire scope. Invalid/crossed boundaries fail rather than being sorted or
clamped; rejected boundaries are explicit. Repeated states retain their evidence.
Back control/turtle use semantic roles rather than visual vertical order; genuine
uncertainty and interruptions remain explicit. No dominance score is inferred.

Techniques require one decision per candidate. Failed attempts remain techniques;
only explicit non-events are ditched. Known outcomes need timestamped evidence
inside the final clip, not a guessed padding duration. Distinct overlapping
attempts survive. An empty technique component is available zero, not missing data.
Position context is advisory and cannot gate attempts within an unchanged position.

The host must preserve raw scan/critique decisions and their observation windows:
final normalized intervals do not themselves retain every rejected boundary or
critique reason. Normalization is pure and accepts trusted persisted stage inputs;
it does not authenticate provider responses or fetch their media.

## Installed-package verification

From the engine worktree, with the s39 shared 1.5.0 wheel already installed:

```sh
../bjj-vision-backend-wt-s39/.venv/bin/python -m pip wheel --no-deps ./analysis_core -w analysis_core/dist
../bjj-vision-backend-wt-s39/.venv/bin/python -m pip install --no-deps --force-reinstall analysis_core/dist/bjj_analysis_core-0.3.0-py3-none-any.whl
../bjj-vision-backend-wt-s39/.venv/bin/python -m pytest analysis_core/tests -q
```

Tests assert imports resolve from `site-packages`, packaged prompt resources exist,
and the core imports no host/lab/infrastructure modules. The backend's
`test_core_normalization_localstack.py` exercises installed scan, candidate union,
temporal correction, analysis and validation through normalization, durable publication preparation and independent
review readiness while preserving a concurrent human edit.
`test_core_executor_localstack.py` drives that real storage boundary through the
complete installed executor. Core tests separately exercise on-disk CAS/history,
reopen, commit-loss, malformed paid responses, bounded retry and cancellation.

## Executor and usage reporting

See [execution and host contract](execution.md). `execute_pipeline(...)` publishes
positions before technique work, persists responses before parsing, reuses completed
work and requires an explicit authority/cancellation guard. Uncertain paid dispatch
is a reconciliation stop, not an automatic retry. Stage summaries represent even
zero-item fan-outs without fake calls. `report_execution(full_history)` provides
per-step progress plus per-node/run costs including prior failed attempts, keeping
unknown spend distinct from the known subtotal. Neither helper settles credits,
publishes SSE nor declares consumer review readiness or engine lifecycle success.

## Remaining work

The host now has an opt-in [Keyspaces/S3 journal adapter](../docs/analysis-checkpoint-journal.md)
with real local persistence/recovery tests, plus [verified source/crop preparation](../docs/analysis-media.md).
An opt-in [Gemini REST provider](../docs/analysis-gemini.md) now has local wire/media/
artifact verification. An opt-in [durable host coordinator](../docs/analysis-run-host.md)
now reserves upload/cleanup and records the complete DAG result in the existing journal.
An opt-in [worker attempt adapter](../docs/analysis-run-worker.md) now adds conditional
lifecycle ownership, heartbeats, cancellation draining and durable completion.
The [registered-run dispatcher](../docs/analysis-run-dispatch.md) now assembles concrete
guarded media/provider/journal and durable outbox ports with exact version gating.
Live provider/model/clock qualification, durable scheduler/public routing integration,
progress/usage projections and operational reconciliation
and lab adoption still need integration. The test SQLite journal is not a new
production database choice.
Prompt editing must eventually produce an
explicitly versioned effective bundle; arbitrary prompt overrides are not accepted
by this initial immutable production bundle. No active workflow is switched, no
live model compatibility or accuracy is verified, and no production release pin
is established by a local wheel build.
