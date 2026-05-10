"""Shared pytest fixtures for service tests.

Provides:
- ``make_mock_jobs_store()`` — factory returning an in-memory mock of
  ``service.jobs_store.JobsStore`` covering the methods routes touch.
- ``mock_jobs_store`` fixture — fresh mock per test.
- ``service_components`` fixture — ``(config, job_store, jobs_store)`` tuple
  with FastAPI routes initialised against the mock.
- ``service_app`` / ``service_client`` fixtures — ASGI client wired to the
  service routes.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.routes import init_routes, router


def make_mock_jobs_store() -> MagicMock:
    """Build an in-memory mock of JobsStore for route-level tests.

    The mock supports race-test injection via ``store._claim_replacement_result``
    — set to ``False`` before a request to simulate a lost replacement claim.
    """
    store = MagicMock()
    _lifecycles: dict[str, dict] = {}
    _requests: dict[str, str] = {}
    _checkpoints: dict[tuple[str, str], dict] = {}
    # Append-only history of every checkpoint write, keyed by (job_id,
    # stage_name). Lets tests assert the *sequence* of incremental writes
    # (e.g. tracking_uploaded → analysis_uploaded → annotated_video_uploaded
    # for the upload row) without losing data when the same row is overwritten.
    _checkpoint_history: dict[tuple[str, str], list[dict]] = {}
    _latest: dict[str, dict] = {}

    async def create_lifecycle(
        job_id,
        video_id="",
        user_id="",
        origin_job_id="",
        parent_job_id="",
        replacement_job_id="",
        owner_instance_id="",
        progress_percent=0.0,
        current_frame=0,
        total_frames=0,
    ):
        _lifecycles[job_id] = {
            "job_id": job_id,
            "video_id": video_id,
            "user_id": user_id,
            "origin_job_id": origin_job_id,
            "parent_job_id": parent_job_id,
            "replacement_job_id": replacement_job_id,
            "job_state": "PENDING",
            "stage": "",
            "progress_percent": progress_percent,
            "current_frame": current_frame,
            "total_frames": total_frames,
            "stage_message": "",
            "error_message": "",
            "owner_instance_id": owner_instance_id,
        }
        return True

    async def get_lifecycle(job_id):
        return _lifecycles.get(job_id)

    async def set_state(job_id, state, error_message="", sync_latest=True):
        if job_id in _lifecycles:
            sv = state.value if hasattr(state, "value") else state
            _lifecycles[job_id]["job_state"] = sv
            if (
                sync_latest
                and sv in {
                    "AWAITING_CORRECTION",
                    "INTERRUPTED",
                    "COMPLETED",
                    "FAILED",
                    "CANCELLED",
                }
                and _lifecycles[job_id].get("video_id")
            ):
                await set_latest(_lifecycles[job_id]["video_id"], job_id, state)
        return True

    async def save_request(job_id, request_json):
        _requests[job_id] = request_json
        return True

    async def get_request(job_id):
        return _requests.get(job_id)

    async def write_checkpoint(job_id, stage_name, completed, data):
        sn = stage_name.value if hasattr(stage_name, "value") else stage_name
        record = {
            "stage_name": sn,
            "completed": completed,
            "checkpoint_data": data,
        }
        _checkpoints[(job_id, sn)] = record
        _checkpoint_history.setdefault((job_id, sn), []).append(record)
        return True

    async def get_checkpoint(job_id, stage_name):
        sn = stage_name.value if hasattr(stage_name, "value") else stage_name
        return _checkpoints.get((job_id, sn))

    async def get_all_checkpoints(job_id):
        return [v for (jid, _), v in _checkpoints.items() if jid == job_id]

    async def set_latest(video_id, job_id, state):
        st = state.value if hasattr(state, "value") else state
        _latest[video_id] = {"job_id": job_id, "job_state": st}
        return True

    async def set_replacement(job_id, replacement_job_id):
        if job_id in _lifecycles:
            _lifecycles[job_id]["replacement_job_id"] = replacement_job_id
        return True

    async def claim_replacement(job_id, replacement_job_id, expected_state=None):
        if getattr(store, "_claim_replacement_result", None) is False:
            return False
        if job_id not in _lifecycles or _lifecycles[job_id].get("replacement_job_id"):
            return False
        _lifecycles[job_id]["replacement_job_id"] = replacement_job_id
        return True

    async def update_progress(*args, **kwargs):
        pass

    async def list_active_recovery_index_rows_newest_first(
        heartbeat_buckets, *, limit_per_bucket=1000
    ):
        return []

    async def claim_pending_job_takeover(
        job_id, new_owner_instance_id, *, expected_owner_instance_id
    ):
        return False

    store.create_lifecycle = create_lifecycle
    store.get_lifecycle = get_lifecycle
    store.set_state = set_state
    store.save_request = save_request
    store.get_request = get_request
    store.write_checkpoint = write_checkpoint
    store.get_checkpoint = get_checkpoint
    store.get_all_checkpoints = get_all_checkpoints
    store.set_latest = set_latest
    store.set_replacement = set_replacement
    store.claim_replacement = claim_replacement
    store.update_progress = update_progress
    store.list_active_recovery_index_rows_newest_first = (
        list_active_recovery_index_rows_newest_first
    )
    store.claim_pending_job_takeover = claim_pending_job_takeover
    store.register_owned_job = MagicMock()
    store.unregister_owned_job = MagicMock()
    store.owned_jobs = set()
    store._lifecycles = _lifecycles
    store._requests = _requests
    store._checkpoints = _checkpoints
    store._checkpoint_history = _checkpoint_history
    store._latest = _latest
    return store


@pytest.fixture()
def mock_jobs_store():
    return make_mock_jobs_store()


@pytest.fixture()
def service_components(mock_jobs_store):
    config = ServiceConfig(
        detection_timeout=86400.0,
        s3_endpoint_url="http://localhost:4566",
    )
    job_store = InMemoryJobStore()
    init_routes(config, job_store, mock_jobs_store)
    return config, job_store, mock_jobs_store


@pytest.fixture()
def service_app(service_components):
    app = FastAPI()
    app.include_router(router)
    return app


@pytest_asyncio.fixture()
async def service_client(service_app):
    transport = ASGITransport(app=service_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
