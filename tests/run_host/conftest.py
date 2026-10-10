"""Reuse exact-media fixtures, adding a disposable real checkpoint keyspace."""

import importlib.util
import logging
import os
from pathlib import Path
from uuid import uuid4

import pytest
from cassandra.cluster import Cluster

from service.analysis_checkpoint_journal import (
    KeyspacesCheckpointJournal,
    checkpoint_table_ddl,
)


def load_fixture_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "media_preparation" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


media_fixtures = load_fixture_module("s39_host_media_fixtures", "conftest.py")
adapter = media_fixtures.adapter
s3 = media_fixtures.s3
media_bytes = media_fixtures.media_bytes
inputs = media_fixtures.inputs
resolver = media_fixtures.resolver
Wire = load_fixture_module("s39_host_wire_fixture", "test_gemini.py").Wire


@pytest.fixture(autouse=True)
def service_logs_reach_caplog(monkeypatch):
    # Importing service.app (test_server) detaches the "service" logger from root.
    monkeypatch.setattr(logging.getLogger("service"), "propagate", True)


@pytest.fixture(scope="module")
def checkpoint_database():
    host = os.environ.get("CHECKPOINT_CASSANDRA_HOST")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise pytest.UsageError("explicit loopback Cassandra required")
    cluster = Cluster([host], port=9042, protocol_version=4)
    session = cluster.connect()
    keyspace = "s39_host_" + uuid4().hex
    created = False
    try:
        session.execute(
            f"CREATE KEYSPACE {keyspace} WITH replication = {{'class':'SimpleStrategy','replication_factor':1}}"
        )
        created = True
        session.execute(checkpoint_table_ddl(keyspace))
        yield session, keyspace
    finally:
        if created:
            session.execute(f"DROP KEYSPACE {keyspace}")
        cluster.shutdown()


@pytest.fixture
def journal(checkpoint_database, s3, inputs):
    session, keyspace = checkpoint_database
    client, bucket = s3

    async def guard():
        """Fixture only; deployment host supplies current authority and lease."""

    return KeyspacesCheckpointJournal(
        inputs[0], session, client, bucket=bucket, keyspace=keyspace, guard=guard
    )
