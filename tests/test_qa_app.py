"""Standalone QA VLM Studio app (``service/qa_app.py``) — boots WITHOUT Cassandra.

Regression guard for the bug this module fixes: ``service.app:app``'s lifespan
unconditionally constructs ``KeyspacesClient()`` (Cassandra) at startup, so it
cannot run standalone for the QA VLM lane. ``service.qa_app:app`` must import
and serve ``/health`` with zero Keyspaces/Cassandra/SAM2 involvement.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_qa_app_imports_without_cassandra_and_health_ok():
    # Import happens here (not at module scope) so a failure surfaces inside
    # the test rather than during collection — this import is the exact path
    # that would raise cassandra.cluster.NoHostAvailable if qa_app touched
    # Keyspaces the way service.app does.
    from service.qa_app import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mode"] == "qa-standalone"


def test_qa_app_boots_even_if_cassandra_connect_would_raise():
    """Regression guard for the actual bug: ``service.app``'s lifespan calls
    ``KeyspacesClient()``, whose ``__init__`` calls ``Cluster.connect()``
    synchronously — that raises ``NoHostAvailable`` with no Cassandra running,
    which is why the full app can't serve the QA VLM lane standalone.

    ``service.routes.qa_vlm``/``service.routes.qa`` are submodules of the
    ``service.routes`` package, so importing them still executes
    ``service/routes/__init__.py`` (Python always initializes parent packages
    before a submodule) — that barrel imports ``service.jobs_store`` ->
    ``service.keyspaces_client`` for its *class* definitions, but never
    constructs a ``KeyspacesClient`` instance at import time. This test proves
    that: even with ``Cluster.connect`` patched to explode (simulating "no
    Cassandra"), importing ``service.qa_app`` and hitting ``/health`` still
    succeeds, because ``qa_app`` never instantiates ``KeyspacesClient``.

    Run in a subprocess so the patch + fresh import isn't contaminated by
    other test modules/conftest that already imported ``service.routes`` (and
    therefore ``service.keyspaces_client``) earlier in the same session.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        from cassandra.cluster import Cluster

        def _boom(self, *a, **kw):
            raise RuntimeError("simulated: no Cassandra available")

        Cluster.connect = _boom  # any real KeyspacesClient() construction now explodes

        from fastapi.testclient import TestClient
        from service.qa_app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200, resp.status_code
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=__file__.rsplit("/tests/", 1)[0],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
