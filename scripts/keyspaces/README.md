# Keyspaces schema helpers

## Apply job-recovery migration (2026-05-01)

See [Keyspaces schema migration — job lineage and recovery index](../../working_log/knowledge-base/requirements/2026-05-01-keyspaces-schema-migration-job-recovery.md) for full requirements, env vars, and rollout notes.

Quick apply (from repo root so `.env` is found; the script calls `load_dotenv()` like `service/app.py`):

```bash
cd /path/to/whole-video-analysis
source venv/bin/activate
pip install -r requirements-service.txt   # cassandra-driver + python-dotenv
python scripts/keyspaces/apply_migrations.py
```

To rely only on shell exports (CI): `python scripts/keyspaces/apply_migrations.py --no-dotenv`.

`apply_migrations.py` is idempotent: duplicate `ALTER ADD` and existing `CREATE TABLE IF NOT EXISTS` are handled safely.

Raw CQL for `cqlsh` or review: `migrations/001_job_lifecycle_lineage.cql`, `migrations/002_job_recovery_index.cql` (adjust keyspace name if needed).
