"""Tests for scheduled worker crash recovery."""

from datetime import datetime, timedelta, timezone

import pytest

from service.analysis_keyspaces_enums import JobState
from service.reconciler import RecoveryManager


class FakeJobsStore:
    def __init__(self, candidates):
        self.candidates = candidates
        self.lifecycles = {candidate["job_id"]: dict(candidate) for candidate in candidates}
        self.claimed = []
        self.states = []

    async def list_stale_recovery_candidates(self, heartbeat_buckets, stale_before):
        return [
            candidate
            for candidate in self.candidates
            if candidate["heartbeat_bucket"] in heartbeat_buckets
            and candidate["last_heartbeat_at"] < stale_before
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


@pytest.mark.asyncio
async def test_reconcile_once_interrupts_and_dispatches_stale_running_job():
    now = datetime(2026, 4, 26, 20, 0, tzinfo=timezone.utc)
    stale_heartbeat = now - timedelta(minutes=3)
    recovered = []
    store = FakeJobsStore([
        {
            "job_id": "stale-job",
            "job_state": JobState.RUNNING.value,
            "owner_instance_id": "old-worker",
            "last_heartbeat_at": stale_heartbeat,
            "heartbeat_bucket": "2026042619",
        }
    ])

    async def recover_job(lifecycle):
        recovered.append(lifecycle["job_id"])

    manager = RecoveryManager(
        store,
        "new-worker",
        recover_job= recover_job,
        stale_after=90.0,
        now_fn=lambda: now,
    )

    await manager.reconcile_once()

    assert store.claimed == [(
        "stale-job",
        "new-worker",
        "old-worker",
        JobState.RUNNING,
        stale_heartbeat,
    )]
    assert store.states == [(
        "stale-job",
        JobState.INTERRUPTED,
        "Worker heartbeat stale; scheduling recovery",
    )]
    assert recovered == ["stale-job"]


@pytest.mark.asyncio
async def test_reconcile_once_ignores_awaiting_correction_jobs():
    now = datetime(2026, 4, 26, 20, 0, tzinfo=timezone.utc)
    store = FakeJobsStore([
        {
            "job_id": "awaiting-job",
            "job_state": JobState.AWAITING_CORRECTION.value,
            "owner_instance_id": "old-worker",
            "last_heartbeat_at": now - timedelta(minutes=10),
            "heartbeat_bucket": "2026042619",
        }
    ])
    manager = RecoveryManager(store, "new-worker", now_fn=lambda: now)

    await manager.reconcile_once()

    assert store.claimed == []
    assert store.states == []


@pytest.mark.asyncio
async def test_reconcile_once_recovers_interrupted_job_without_replacement():
    now = datetime(2026, 4, 26, 20, 0, tzinfo=timezone.utc)
    stale_heartbeat = now - timedelta(minutes=3)
    recovered = []
    store = FakeJobsStore([
        {
            "job_id": "interrupted-job",
            "job_state": JobState.INTERRUPTED.value,
            "owner_instance_id": "old-worker",
            "replacement_job_id": "",
            "last_heartbeat_at": stale_heartbeat,
            "heartbeat_bucket": "2026042619",
        }
    ])

    async def recover_job(lifecycle):
        recovered.append(lifecycle["job_id"])

    manager = RecoveryManager(
        store,
        "new-worker",
        recover_job=recover_job,
        stale_after=90.0,
        now_fn=lambda: now,
    )

    await manager.reconcile_once()

    assert store.claimed == [(
        "interrupted-job",
        "new-worker",
        "old-worker",
        JobState.INTERRUPTED,
        stale_heartbeat,
    )]
    assert store.states == []
    assert recovered == ["interrupted-job"]


@pytest.mark.asyncio
async def test_reconcile_once_ignores_stale_index_when_lifecycle_is_fresh():
    now = datetime(2026, 4, 26, 20, 0, tzinfo=timezone.utc)
    old_heartbeat = now - timedelta(minutes=10)
    fresh_heartbeat = now - timedelta(seconds=10)
    store = FakeJobsStore([
        {
            "job_id": "healthy-job",
            "job_state": JobState.RUNNING.value,
            "owner_instance_id": "worker",
            "last_heartbeat_at": old_heartbeat,
            "heartbeat_bucket": "2026042619",
        }
    ])
    store.lifecycles["healthy-job"]["last_heartbeat_at"] = fresh_heartbeat
    manager = RecoveryManager(store, "new-worker", now_fn=lambda: now)

    await manager.reconcile_once()

    assert store.claimed == []
    assert store.states == []
