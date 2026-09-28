# BJJ analysis observer 0.1.0

Installable read-only durable observation for the position DAG, qualified against
`bjj-analysis-core==0.3.0` and `shared-lib>=1.5.0,<1.6`. Hosts need no engine process,
engine source imports, media preparation or model client to read existing work.

Build and install the qualified local shared/core wheels first, then this wheel:

This is an opt-in DAG runtime installation, not an addition to the legacy service
requirements. Host packaging/deployment must explicitly supply these qualified
wheels before mounting the new DAG; no public package-index substitution is intended.

```sh
python -m pip wheel --no-deps --no-build-isolation --no-index ./analysis_observer --wheel-dir ./analysis_observer/dist
python -m pip install --no-deps --no-index ./analysis_observer/dist/bjj_analysis_observer-0.1.0-py3-none-any.whl
```

Use leaf imports. `checkpoints` contains verified bounded checkpoint/history reads;
`progress`, `lifecycle`, and `results` contain the existing internal observations.
`observer.AuthorizedRunObserver.status(analysis_id, run_id, actor=verified_actor)`
adds actual-viewer binding to lifecycle/progress reads.

The host must authenticate `actor` from verified credentials, never a URL/body owner,
IP address, guessed run ID or stored run owner. The required async `authorize(run,
actor)` callback must recheck current account/video access or standalone session/source
access and expiry. The observer checks registered run ownership and invokes that
callback repeatedly, including before returning. Callback failures propagate.

Standalone credentials are to be server-issued under the approved new DAG contract.
This package neither issues nor validates cookies and must not be wired to the legacy
client-chosen session-ID signing route. IP-based quota is not result authorization.

Inject approved read-only storage clients with least-privilege access. The package
does not provision resources, upload checkpoints, acquire leases, admit work, call
providers, retry analysis or charge usage. Python method separation is not a substitute
for IAM. The engine-only journal subclass owns writes.

Views are internal, sequential observations rather than atomic snapshots. Authority,
corruption and topology races fail explicitly. No automatic retry or empty-success
fallback is supplied. The host still owns public DTO redaction, sanitized errors,
stream pacing/concurrency and authentication expiry. Direct reads can be expensive:
qualify large-fanout latency/cost before enabling production SSE.

Verification: run `python -m pytest --confcutdir=analysis_observer/tests
analysis_observer/tests -q` against an installed, noneditable wheel; the engine's native
`tests/run_host` and `tests/checkpoint_journal` cover storage/execution compatibility.
