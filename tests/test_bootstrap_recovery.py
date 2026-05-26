"""Tests for single-shot recovery sweep at FastAPI lifespan startup.

These exercise ``service.routes.bootstrap_recovery_on_startup`` end to end
against a real ``RecoveryManager`` instance and the same ``FakeJobsStore``
shape used by ``tests/test_reconciler.py``.
"""

import logging
from datetime import datetime, timedelta, timezone

import pytest

from service.analysis_keyspaces_enums import JobState
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.reconciler import RecoveryManager
from service import routes as routes_mod


class FakeJobsStore:
    """Mirror of ``tests/test_reconciler.py::FakeJobsStore`` for bootstrap.

    Implements the subset of ``JobsStore`` used by ``RecoveryManager``: the
    candidates listing, lifecycle lookup, claim, and (optional) state update.
    The bucket/heartbeat filter mirrors the production query.
    """

    def __init__(self, candidates):
        self.candidates = candidates
        self.lifecycles = {c["job_id"]: dict(c) for c in candidates}
        self.claimed: list[tuple] = []
        self.states: list[tuple] = []

    async def list_stale_recovery_candidates(self, heartbeat_buckets, stale_before):
        return [
            c
            for c in self.candidates
            if c["heartbeat_bucket"] in heartbeat_buckets
            and c["last_heartbeat_at"] < stale_before
        ]

    async def get_lifecycle(self, job_id):
        return self.lifecycles.get(job_id)

    async def claim_job_for_recovery(
        self,
        job_id,
        owner_instance_id,
        expected_owner_instance_id,
        expected_state=JobState.RUNNING,
        expected_last_heartbeat_at=None,
    ):
        self.claimed.append((
            job_id,
            owner_instance_id,
            expected_owner_instance_id,
            expected_state,
            expected_last_heartbeat_at,
        ))
        return True

    async def set_state(self, job_id, state, error_message=""):
        self.states.append((job_id, state, error_message))
        self.lifecycles[job_id]["job_state"] = state.value
        return True


def _wire_routes(jobs_store, instance_id: str = "new-1") -> None:
    cfg = ServiceConfig(
        detection_timeout=86400.0,
        s3_endpoint_url="http://localhost:4566",
    )
    routes_mod.init_routes(cfg, InMemoryJobStore(), jobs_store, instance_id=instance_id)


@pytest.mark.asyncio
async def test_bootstrap_recovery_finds_running_orphan_with_fresh_heartbeat():
    """A RUNNING orphan whose heartbeat is only 5s old MUST be claimed at startup."""
    now = datetime.now(timezone.utc)
    fresh_heartbeat = now - timedelta(seconds=5)  # not stale under default 90s
    bucket = now.strftime("%Y%m%d%H")
    store = FakeJobsStore([
        {
            "job_id": "orphan-running",
            "job_state": JobState.RUNNING.value,
            "owner_instance_id": "previous-process",
            "last_heartbeat_at": fresh_heartbeat,
            "heartbeat_bucket": bucket,
        }
    ])
    _wire_routes(store, instance_id="bootstrap-1")

    recovered: list[str] = []

    async def recover_job(lifecycle):
        recovered.append(lifecycle["job_id"])

    await routes_mod.bootstrap_recovery_on_startup(
        "bootstrap-1",
        recover_job,
        heartbeat_bucket_hours=24,
    )

    assert recovered == ["orphan-running"]
    assert store.claimed and store.claimed[0][0] == "orphan-running"
    # RUNNING -> INTERRUPTED transition recorded before recover_job dispatch.
    assert store.states == [
        ("orphan-running", JobState.INTERRUPTED, "Worker heartbeat stale; scheduling recovery"),
    ]


@pytest.mark.asyncio
async def test_periodic_reconcile_default_does_not_pick_up_fresh_heartbeat():
    """Control case: default ``reconcile_once()`` (stale_after=90s) leaves it alone.

    This proves the bootstrap override is what enables claiming a freshly
    orphaned job at restart; periodic-loop defaults are unchanged.
    """
    now = datetime.now(timezone.utc)
    fresh_heartbeat = now - timedelta(seconds=5)
    bucket = now.strftime("%Y%m%d%H")
    store = FakeJobsStore([
        {
            "job_id": "orphan-running",
            "job_state": JobState.RUNNING.value,
            "owner_instance_id": "previous-process",
            "last_heartbeat_at": fresh_heartbeat,
            "heartbeat_bucket": bucket,
        }
    ])

    recovered: list[str] = []

    async def recover_job(lifecycle):
        recovered.append(lifecycle["job_id"])

    manager = RecoveryManager(
        store,
        "new-worker",
        recover_job=recover_job,
        stale_after=90.0,
        heartbeat_bucket_hours=24,
        now_fn=lambda: now,
    )
    await manager.reconcile_once()

    assert recovered == []
    assert store.claimed == []
    assert store.states == []


@pytest.mark.asyncio
async def test_bootstrap_recovery_swallows_recover_job_exception(caplog):
    """If ``recover_job`` raises, bootstrap MUST log with exc_info and not propagate."""
    now = datetime.now(timezone.utc)
    fresh_heartbeat = now - timedelta(seconds=5)
    bucket = now.strftime("%Y%m%d%H")
    store = FakeJobsStore([
        {
            "job_id": "bad-orphan",
            "job_state": JobState.INTERRUPTED.value,  # skip RUNNING->INTERRUPTED extra write
            "owner_instance_id": "previous-process",
            "replacement_job_id": "",
            "last_heartbeat_at": fresh_heartbeat,
            "heartbeat_bucket": bucket,
        }
    ])
    _wire_routes(store, instance_id="bootstrap-2")

    async def recover_job(lifecycle):
        raise RuntimeError("synthetic recover failure")

    caplog.set_level(logging.ERROR, logger="service.routes")

    # MUST NOT raise.
    await routes_mod.bootstrap_recovery_on_startup(
        "bootstrap-2",
        recover_job,
        heartbeat_bucket_hours=24,
    )

    # Error message captured with stack trace.
    failure_records = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and "Bootstrap recovery" in r.getMessage()
    ]
    assert failure_records, "expected bootstrap failure log"
    rec = failure_records[0]
    assert rec.exc_info is not None, "exc_info must be attached for stack traces"
    assert "synthetic recover failure" in rec.getMessage() or any(
        "synthetic recover failure" in (r.exc_text or "") for r in caplog.records if r.exc_text
    )


@pytest.mark.asyncio
async def test_bootstrap_recovery_emits_dispatch_and_tick_info_logs(caplog):
    """The reconciler's per-tick and per-dispatch INFO lines MUST emit so
    operators can confirm the bootstrap sweep saw and acted on candidates.
    """
    now = datetime.now(timezone.utc)
    fresh_heartbeat = now - timedelta(seconds=5)
    bucket = now.strftime("%Y%m%d%H")
    store = FakeJobsStore([
        {
            "job_id": "obs-orphan",
            "job_state": JobState.INTERRUPTED.value,
            "owner_instance_id": "previous-process",
            "replacement_job_id": "",
            "last_heartbeat_at": fresh_heartbeat,
            "heartbeat_bucket": bucket,
        }
    ])
    _wire_routes(store, instance_id="bootstrap-obs")

    async def recover_job(lifecycle):
        return None

    caplog.set_level(logging.INFO, logger="service.reconciler")
    caplog.set_level(logging.INFO, logger="service.routes")

    await routes_mod.bootstrap_recovery_on_startup(
        "bootstrap-obs",
        recover_job,
        heartbeat_bucket_hours=24,
    )

    messages = [r.getMessage() for r in caplog.records]
    assert any("Recovery tick: candidates=" in m for m in messages), messages
    assert any(
        "Recovery: claimed and dispatched job_id=obs-orphan" in m for m in messages
    ), messages


@pytest.mark.asyncio
async def test_bootstrap_recovery_logs_scan_and_completion(caplog):
    """Operators rely on the scan/complete log lines to confirm bootstrap ran."""
    store = FakeJobsStore([])
    _wire_routes(store, instance_id="bootstrap-3")

    async def recover_job(lifecycle):
        return None

    caplog.set_level(logging.INFO, logger="service.routes")

    await routes_mod.bootstrap_recovery_on_startup(
        "bootstrap-3",
        recover_job,
        heartbeat_bucket_hours=24,
    )

    messages = [r.getMessage() for r in caplog.records if r.name == "service.routes"]
    assert any(
        "Bootstrap recovery: scanning 24 buckets" in m for m in messages
    ), messages
    assert any("Bootstrap recovery: complete" in m for m in messages), messages
