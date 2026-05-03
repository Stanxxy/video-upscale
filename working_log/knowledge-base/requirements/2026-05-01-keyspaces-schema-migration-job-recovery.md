---

## date: 2026-05-01
category: requirement
tags: [keyspaces, schema, migration, job-lifecycle, crash-recovery, ops]
status: active

# Keyspaces schema migration — job lineage and recovery index

## Purpose

The vision engine `JobsStore` (`service/jobs_store.py`) persists manual resume lineage and crash-recovery discovery in Amazon Keyspaces (Apache Cassandra–compatible API). Deployments that predate the job lifecycle refactor must apply this DDL **before** rolling out code that inserts or reads `parent_job_id`, `replacement_job_id`, or `job_recovery_index`.

## Preconditions

- Amazon Keyspaces (or a Cassandra cluster with the same CQL semantics) reachable from the operator network.
- IAM credentials (SigV4) or username/password as required by your environment. The engine’s `KeyspacesClient` uses `PlainTextAuthProvider` with `KEYSPACES_USERNAME` / `KEYSPACES_PASSWORD` (Keyspaces service-specific credentials).
- Environment variables aligned with `service/keyspaces_client.py`:


| Variable                                    | Default          | Notes                                                                     |
| ------------------------------------------- | ---------------- | ------------------------------------------------------------------------- |
| `KEYSPACE_NAME`                             | `video_analysis` | Must match the keyspace used by `bjj-vision-backend` / shared job tables. |
| `KEYSPACES_CONTACT_POINTS`                  | *(empty)*        | Comma-separated contact points (Keyspaces regional endpoint hostnames).   |
| `KEYSPACES_PORT`                            | `9142`           | Keyspaces TLS port.                                                       |
| `KEYSPACES_USERNAME` / `KEYSPACES_PASSWORD` |                  | Service-specific username and password from IAM.                          |
| `KEYSPACES_USE_SSL`                         | `true`           | Script mirrors engine: TLS with `CERT_NONE` for Keyspaces endpoints.      |


- **Companion services:** Any service that shares `job_lifecycle` (for example `bjj-vision-backend` / `shared_lib`) must use the same keyspace and table definitions. Coordinate a single migration window or apply the same DDL from one canonical source.

## DDL (canonical)

### 1. Lifecycle lineage columns

```cql
ALTER TABLE <keyspace>.job_lifecycle ADD parent_job_id text;
ALTER TABLE <keyspace>.job_lifecycle ADD replacement_job_id text;
```

- `origin_job_id` is assumed to already exist on `job_lifecycle` from prior schema; if not, add it in the same rollout as this migration.
- Re-running `ADD` after a column exists fails; the bundled `scripts/keyspaces/apply_migrations.py` treats duplicate-column errors as success (idempotent apply).

### 2. Recovery index table

Used by `JobsStore.upsert_recovery_index`, `list_stale_recovery_candidates`, and `remove_recovery_index`. Partition key bounds scans to `recovery_state` + `heartbeat_bucket` (values `ACTIVE`, `AWAITING_CORRECTION`, `TERMINAL` and `YYYYMMDDHH`).

```cql
CREATE TABLE IF NOT EXISTS <keyspace>.job_recovery_index (
  recovery_state text,
  heartbeat_bucket text,
  last_heartbeat_at timestamp,
  job_id text,
  owner_instance_id text,
  video_id text,
  job_state text,
  updated_at timestamp,
  PRIMARY KEY ((recovery_state, heartbeat_bucket), last_heartbeat_at, job_id)
) WITH CLUSTERING ORDER BY (last_heartbeat_at ASC, job_id ASC);
```

## Raw CQL files (cqlsh / review)

Checked-in statements mirror the Python runner:

- `scripts/keyspaces/migrations/001_job_lifecycle_lineage.cql`
- `scripts/keyspaces/migrations/002_job_recovery_index.cql`

Edit the `video_analysis` keyspace name in those files if your deployment differs.

## How to apply (engine repo)

From the repo root with `cassandra-driver` installed (see `requirements-service.txt`):

```bash
source venv/bin/activate
export KEYSPACE_NAME=video_analysis
export KEYSPACES_CONTACT_POINTS=cassandra.<region>.amazonaws.com
export KEYSPACES_USERNAME=...
export KEYSPACES_PASSWORD=...
python scripts/keyspaces/apply_migrations.py
```

Options:

- `--dry-run` — print statements only; no cluster connection.
- `--verify-only` — connect and describe that expected columns/table exist (read-only checks where supported).

## Rollback

- Dropping added columns or the recovery table is **not** required for rollback of application code alone; older code ignores extra columns and tables.
- If you must remove schema, Keyspaces/Cassandra column drop is operationally risky; prefer leaving columns unused until a planned compaction/maintenance window per AWS guidance.

## Verification

1. Run `python scripts/keyspaces/apply_migrations.py --verify-only` after apply.
2. Smoke: start the service, create a job, confirm `INSERT` into `job_lifecycle` with `parent_job_id` / `replacement_job_id` and rows in `job_recovery_index` after `create_lifecycle` / `heartbeat` / `set_state`.

## Acceptance criteria

- DDL applied to every production/staging Keyspaces keyspace that backs vision-engine job persistence.
- Companion backend keyspace (if shared) updated in the same change window.
- Idempotent apply script committed and documented (this file + `scripts/keyspaces/README.md`).

## References

- Implementation plan: [2026-04-26 - Job Lifecycle Resume Refactor Plan](2026-04-26-job-lifecycle-resume-refactor-plan.md)
- Workflow reference: [2026-04-25 - Job Start and Resume Workflow Reference](../insights/2026-04-25-job-start-resume-workflow-reference.md)

