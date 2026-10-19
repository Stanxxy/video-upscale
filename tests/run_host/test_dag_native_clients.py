"""The engine's production Keyspaces client verifies certificates against the service hostname.

Release-0032 (2026-10-19): without ``server_hostname`` the driver checked each discovered
node IP against Amazon Keyspaces' certificate and the trial engine refused to start.
"""
import ssl
from contextlib import ExitStack

import cassandra.cluster
import pytest

from service import dag_server


class FakeCluster:
    created = []

    def __init__(self, contact_points, **kwargs):
        self.contact_points, self.kwargs = contact_points, kwargs
        FakeCluster.created.append(self)

    def connect(self, keyspace):
        self.keyspace = keyspace
        return "session"

    def shutdown(self):
        pass


@pytest.fixture
def settings():
    return {
        "region": "us-east-1",
        "cassandra_ca": None,
        "cassandra_host": "cassandra.us-east-1.amazonaws.com",
        "cassandra_port": 9142,
        "cassandra_username": "user",
        "cassandra_password": "secret",
        "keyspace": "bjj_vision",
    }


def test_keyspaces_tls_verifies_the_service_hostname(monkeypatch, settings):
    monkeypatch.setattr(cassandra.cluster, "Cluster", FakeCluster)
    FakeCluster.created.clear()
    with ExitStack() as stack:
        clients = dag_server._native_clients(settings, stack)
    (cluster,) = FakeCluster.created
    assert clients["session"] == "session" and cluster.keyspace == "bjj_vision"
    assert cluster.contact_points == ["cassandra.us-east-1.amazonaws.com"] and cluster.kwargs["port"] == 9142
    tls = cluster.kwargs["ssl_context"]
    assert tls.check_hostname is True and tls.verify_mode == ssl.CERT_REQUIRED
    assert cluster.kwargs["ssl_options"] == {"server_hostname": "cassandra.us-east-1.amazonaws.com"}
