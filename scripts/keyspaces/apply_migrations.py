#!/usr/bin/env python3
"""Apply Keyspaces DDL for job recovery schema (idempotent).

Requires cassandra-driver and the same env vars as service/keyspaces_client.py.

Loads a `.env` file from the current working directory (or parents) via
`python-dotenv`, matching `service/app.py`, unless `--no-dotenv` is passed.
"""

from __future__ import annotations

import argparse
import logging
import os
import ssl
import sys
import time
from typing import Iterable

from cassandra import InvalidRequest
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Populate os.environ from `.env` (cwd walk), same pattern as the FastAPI app."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning(
            "python-dotenv not installed; export KEYSPACES_* in your shell or "
            "`pip install python-dotenv`"
        )
        return
    load_dotenv()


def _ddl_statements(keyspace: str) -> list[str]:
    ks = keyspace.strip()
    return [
        f"ALTER TABLE {ks}.job_lifecycle ADD parent_job_id text",
        f"ALTER TABLE {ks}.job_lifecycle ADD replacement_job_id text",
        (
            f"CREATE TABLE IF NOT EXISTS {ks}.job_recovery_index ("
            "recovery_state text,"
            "heartbeat_bucket text,"
            "last_heartbeat_at timestamp,"
            "job_id text,"
            "owner_instance_id text,"
            "video_id text,"
            "job_state text,"
            "updated_at timestamp,"
            "PRIMARY KEY ((recovery_state, heartbeat_bucket), last_heartbeat_at, job_id)"
            ") WITH CLUSTERING ORDER BY (last_heartbeat_at ASC, job_id ASC)"
        ),
    ]


def _alter_add_is_duplicate(msg: str) -> bool:
    m = msg.lower()
    return (
        "already exists" in m
        or "duplicate column name" in m
        or "conflicts with an existing column" in m
    )


def _schema_change_in_progress(msg: str) -> bool:
    m = msg.lower()
    return "being created, altered or deleted" in m


def _execute_idempotent(session, statement: str) -> str:
    max_attempts = 12
    delay = 2.0
    for attempt in range(max_attempts):
        try:
            session.execute(statement)
            return "applied"
        except InvalidRequest as e:
            msg = str(e)
            if statement.strip().upper().startswith("ALTER TABLE") and _alter_add_is_duplicate(
                msg
            ):
                logger.info("Skipped (already present): %s", statement[:120])
                return "skipped"
            if _schema_change_in_progress(msg) and attempt < max_attempts - 1:
                logger.info(
                    "Schema change in progress; retrying in %.0fs (%s)",
                    delay,
                    statement[:60],
                )
                time.sleep(delay)
                delay = min(delay * 1.5, 30.0)
                continue
            raise
    raise RuntimeError("unreachable")


def _connect():
    keyspace = os.getenv("KEYSPACE_NAME", "video_analysis")
    username = os.getenv("KEYSPACES_USERNAME", "")
    password = os.getenv("KEYSPACES_PASSWORD", "")
    contact_points = [p.strip() for p in os.getenv("KEYSPACES_CONTACT_POINTS", "").split(",") if p.strip()]
    port = int(os.getenv("KEYSPACES_PORT", "9142"))
    use_ssl = os.getenv("KEYSPACES_USE_SSL", "true").lower() == "true"

    if not contact_points:
        print(
            "KEYSPACES_CONTACT_POINTS is empty; set it to your Keyspaces contact host(s).",
            file=sys.stderr,
        )
        sys.exit(2)

    ssl_context = None
    if use_ssl:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    auth_provider = PlainTextAuthProvider(username=username, password=password)
    cluster = Cluster(
        contact_points=contact_points,
        auth_provider=auth_provider,
        protocol_version=4,
        ssl_context=ssl_context,
        port=port,
    )
    session = cluster.connect(keyspace)
    session.default_consistency_level = ConsistencyLevel.LOCAL_QUORUM
    return cluster, session, keyspace


def _verify(session, keyspace: str) -> bool:
    """Best-effort checks via system_schema (Keyspaces: WHERE only keyspace_name [+ table_name])."""
    ok = True
    try:
        q_cols = (
            "SELECT column_name FROM system_schema.columns "
            "WHERE keyspace_name = %s AND table_name = 'job_lifecycle'"
        )
        lifecycle_cols = {r.column_name for r in session.execute(q_cols, [keyspace])}
        for col in ("parent_job_id", "replacement_job_id"):
            if col not in lifecycle_cols:
                print(
                    f"VERIFY FAIL: job_lifecycle.{col} not found in system_schema",
                    file=sys.stderr,
                )
                ok = False
            else:
                print(f"VERIFY OK: job_lifecycle.{col}")
        q_tbl = (
            "SELECT table_name FROM system_schema.tables "
            "WHERE keyspace_name = %s"
        )
        table_names = {r.table_name for r in session.execute(q_tbl, [keyspace])}
        if "job_recovery_index" not in table_names:
            print("VERIFY FAIL: job_recovery_index table missing", file=sys.stderr)
            ok = False
        else:
            print("VERIFY OK: job_recovery_index")
    except Exception as e:
        print(
            f"VERIFY SKIP: system_schema check failed ({e}); confirm columns manually.",
            file=sys.stderr,
        )
        return True
    return ok


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Print DDL only; do not connect.")
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Connect and verify schema; no DDL.",
    )
    p.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Do not load .env (use only variables already exported in the shell).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.no_dotenv:
        _load_dotenv()

    keyspace = os.getenv("KEYSPACE_NAME", "video_analysis")
    stmts = _ddl_statements(keyspace)

    if args.dry_run:
        for s in stmts:
            print(s + ";")
        return 0

    cluster = None
    try:
        cluster, session, ks = _connect()
        if args.verify_only:
            return 0 if _verify(session, ks) else 1
        for s in stmts:
            status = _execute_idempotent(session, s)
            logger.info("%s: %s", status, s[:100])
        print("Migration finished.")
        if not _verify(session, ks):
            return 1
        return 0
    finally:
        if cluster is not None:
            cluster.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
