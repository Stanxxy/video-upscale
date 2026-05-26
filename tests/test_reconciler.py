"""Tests for scheduled worker crash recovery."""

from datetime import datetime, timedelta, timezone

import pytest

from service.analysis_keyspaces_enums import JobState
from service.reconciler import RecoveryManager


def test_heartbeat_buckets_for_scan_default_trailing_window():
    now = datetime(2026, 4, 26, 15, 30, tzinfo=timezone.utc)
    buckets = RecoveryManager.heartbeat_buckets_for_scan(now, hours=24)
    assert len(buckets) == 24
    assert buckets[0] == "2026042615"
    assert buckets[-1] == "2026042516"
    assert "2026042608" in buckets  # 7 hours before anchor hour


def test_heartbeat_buckets_for_scan_dedupes_dst_safe():
    """Consecutive hours always yield distinct bucket strings in UTC."""
    now = datetime(2026, 4, 26, 20, 0, tzinfo=timezone.utc)
    buckets = RecoveryManager.heartbeat_buckets_for_scan(now, hours=3)
    assert buckets == ["2026042620", "2026042619", "2026042618"]


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
async def test_reconcile_once_finds_stale_job_in_bucket_many_hours_behind():
    """Index partition follows last heartbeat hour; wide scan must still see the row."""
    now = datetime(2026, 4, 26, 20, 0, tzinfo=timezone.utc)
    stale_heartbeat = now - timedelta(minutes=3)
    old_bucket = (now - timedelta(hours=12)).strftime("%Y%m%d%H")
    recovered = []
    store = FakeJobsStore([
        {
            "job_id": "overnight-stale",
            "job_state": JobState.RUNNING.value,
            "owner_instance_id": "dead-worker",
            "last_heartbeat_at": stale_heartbeat,
            "heartbeat_bucket": old_bucket,
        }
    ])

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

    assert recovered == ["overnight-stale"]
    assert store.claimed[0][0] == "overnight-stale"


@pytest.mark.asyncio
async def test_reconcile_once_narrow_bucket_hours_misses_distant_partition():
    now = datetime(2026, 4, 26, 20, 0, tzinfo=timezone.utc)
    stale_heartbeat = now - timedelta(minutes=3)
    old_bucket = (now - timedelta(hours=12)).strftime("%Y%m%d%H")
    store = FakeJobsStore([
        {
            "job_id": "too-old-for-two-hour-scan",
            "job_state": JobState.RUNNING.value,
            "owner_instance_id": "dead-worker",
            "last_heartbeat_at": stale_heartbeat,
            "heartbeat_bucket": old_bucket,
        }
    ])
    manager = RecoveryManager(
        store,
        "new-worker",
        stale_after=90.0,
        heartbeat_bucket_hours=2,
        now_fn=lambda: now,
    )

    await manager.reconcile_once()

    assert store.claimed == []


@pytest.mark.asyncio
async def test_reconcile_once_stale_after_override_picks_up_fresh_heartbeats():
    """``stale_after_override=0`` claims any candidate whose heartbeat is in the past;
    the default (90s) leaves the same fresh-heartbeat candidate alone."""
    now = datetime(2026, 4, 26, 20, 0, tzinfo=timezone.utc)
    fresh_heartbeat = now - timedelta(seconds=5)  # NOT stale by default 90s
    recovered: list[str] = []
    store = FakeJobsStore([
        {
            "job_id": "fresh-but-orphaned",
            "job_state": JobState.RUNNING.value,
            "owner_instance_id": "previous-process",
            "last_heartbeat_at": fresh_heartbeat,
            "heartbeat_bucket": "2026042620",
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

    # Default pass: heartbeat is fresh -> skipped.
    await manager.reconcile_once()
    assert recovered == []
    assert store.claimed == []

    # Override: any heartbeat strictly in the past is stale -> claimed.
    await manager.reconcile_once(stale_after_override=0.0)
    assert recovered == ["fresh-but-orphaned"]
    assert store.claimed and store.claimed[0][0] == "fresh-but-orphaned"


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
