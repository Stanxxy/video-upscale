# Opt-in position DAG engine entrypoint

`python -m service.dag_server` runs the existing public engine application with
the approved separate internal signed listener and four supervised owner workers.
It is not enabled by changing the legacy Uvicorn command alone: enabling
`BJJ_DAG_ENGINE_ENABLED=1` without the owning entrypoint fails before legacy startup
I/O. Existing service scripts/container commands remain unchanged/default-off.

Export deployment settings before starting the module. It validates configuration
before importing the legacy application; it does not load DAG secrets from `.env`.
The public host/port still use `BJJ_HOST`/`BJJ_PORT` (defaults0.0.0.0/8000).
Proxy rewriting and access logs are disabled in the owning public Uvicorn server.
The private run-ID-only API never mounts on that public listener.

## Explicit settings

| Environment setting | Meaning |
| --- | --- |
| `BJJ_DAG_ENGINE_ENABLED` | Must be exactly `1` for this entrypoint |
| `BJJ_DAG_QUALIFIED_MODELS_JSON` | Nonempty model-ID → returned-version-list JSON, supplied only after live qualification; no model default |
| `BJJ_DAG_GEMINI_API_KEY` | Dedicated provider credential; never logged |
| `BJJ_DAG_INTERNAL_HOST`, `BJJ_DAG_INTERNAL_PORT` | Exact private bind IP and port; not wildcard/public |
| `BJJ_DAG_INTERNAL_CERTIFICATE`, `BJJ_DAG_INTERNAL_PRIVATE_KEY` | Existing private listener TLS files; no environment bypass |
| `BJJ_DAG_ENQUEUE_AUDIENCE` | Must match both admission hosts |
| `BJJ_DAG_AWS_REGION` | Native AWS client region; standard SDK credential chain/IAM |
| `BJJ_DAG_KEYSPACE` | Existing engine checkpoint/lifecycle/recovery-index keyspace |
| `BJJ_DAG_CASSANDRA_HOST`, `BJJ_DAG_CASSANDRA_PORT` | Dedicated verified-TLS Cassandra/Keyspaces connection |
| `BJJ_DAG_CASSANDRA_USERNAME`, `BJJ_DAG_CASSANDRA_PASSWORD` | Dedicated database credentials |
| `BJJ_DAG_CASSANDRA_CA_FILE` | Trusted CA PEM for database TLS; certificate and hostname verification remain enabled |
| `BJJ_DAG_MAX_SOURCE_BYTES` | Positive maximum exact-source download size |

Supply each suffix below twice, under `BJJ_DAG_ACCOUNT_` and `BJJ_DAG_SESSION_`:

- `ANNOTATION_TABLE_PREFIX`, `PROPOSAL_TABLE_PREFIX`, `PUBLICATION_TABLE`: existing
  model-specific owner tables. All model table names must be disjoint.
- `ARTIFACT_BUCKET`, `SOURCE_BUCKETS_JSON`: explicit artifact bucket and nonempty
  JSON list of allowed private source buckets.
- `SCRATCH_ROOT`: existing absolute scratch directory, never the workspace root.
- `RESULT_TOPIC_ARN`: the corresponding owner's result publication topic.
- `ACCESS_ORIGIN`, `ACCESS_AUDIENCE`, `ACCESS_KEY_ID`, `ACCESS_SECRET`: exact HTTPS
  current-access endpoint configuration for that host.
- `ENQUEUE_KEY_ID`, `ENQUEUE_SECRET`: owner-bound admission signing credential.
  The four access/enqueue secrets must be distinct; do not reuse quota/IP secrets.

## Ownership and qualification

The runtime factory constructs additional native clients, checks active DynamoDB
tables/indexes against the existing definitions, checks buckets/topics, and prepares
the required CQL columns without scans or DDL. Run inputs/preset bytes remain the
immutable registered snapshots and are checked again by the existing dispatcher.
Providing a model allowlist is not evidence that the model has been qualified.

Each store uses its own dispatcher/scheduler/publication worker, with one shared
process-local execution semaphore. Worker failure reaches the public host's shutdown
callback; public requests and workers drain before newly owned native/HTTP clients
close. Partial startup also closes all constructed clients. Cancellation joins
blocking SDK/driver work before releasing its owner.

Legacy recovery still uses the legacy application and its existing clients. The
new DAG uses a separate verified-TLS connection, not the legacy client's permissive
TLS wrapper. The new partitions do not migrate or consume old scheduling hints.

Native local tests inject explicit fixture clients/transports into the same factory;
there is no environment switch for fixture providers or disabling internal TLS.
This is local implementation, not production deployment or live model qualification.

## Approved Gemini 3.8 production setup

The founder approved Gemini3.8 production selection on September21 after the
real-video run. `config/analysis/gemini38.json` is the canonical new-run preset:
all six nodes use `gemini-3.8-flash`, high thinking, low media resolution,
1fps scans and10fps detailed stages. The high-resolution experiment is excluded.
`config/analysis/qualified-models.json` binds the model ID to the returned version
observed in all21 calls of the completed real-video native DAG.

Publish the exact preset bytes into each owner's existing private artifact bucket
under a content-addressed immutable key. Set that owner's `BJJ_DAG_PRESET_REF_JSON`
to its bucket/key and SHA256 of those bytes. Set engine
`BJJ_DAG_QUALIFIED_MODELS_JSON` to the exact version-map file contents. Existing
runs retain their registered configuration reference; never overwrite an old
preset object or rewrite retry snapshots. The admission host and engine still
require their existing explicit infrastructure configuration and opt-in activation.

The setup is tested against the advertised legacy model and portable preset.
Approval is model selection, not a claim of perfect BJJ accuracy: the sample
missed half guard at low resolution. See umbrella S39 real-video qualification
receipt. Publishing these files alone does not deploy or enable a remote host.
