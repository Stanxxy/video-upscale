"""S12 Phase 1b — service/worker/stages/highlight_ingest.py (item 6).

Mocked S3 + mocked Gemini client throughout — no live network access.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from service.config import ServiceConfig
from service.models import TrackRequest
from service.worker.stages import highlight_ingest


class FakeJobsStore:
    def __init__(self, checkpoints=None):
        self._checkpoints = checkpoints or []
        self.written: list[tuple] = []
        self.lifecycle = {"progress_percent": 0.0}

    async def get_all_checkpoints(self, job_id):
        return self._checkpoints

    async def write_checkpoint(self, job_id, stage_name, completed, data):
        self.written.append((job_id, stage_name, completed, data))
        return True

    async def get_lifecycle(self, job_id):
        return dict(self.lifecycle)

    async def update_highlight_progress(self, job_id, stage, percent, **kwargs):
        self.lifecycle.update({"progress_percent": percent, **kwargs})
        return True


class FakeS3:
    """Objects are keyed on (bucket, key) — NOT key alone — so a
    bucket-mismatch bug raises a KeyError instead of silently succeeding."""

    def __init__(self, objects: dict[tuple[str, str], bytes]):
        self._objects = objects
        self.ensure_bucket_calls: list[str] = []
        self.downloaded: list[tuple] = []
        self.get_object_calls: list[tuple[str, str]] = []

    def ensure_bucket(self, bucket):
        self.ensure_bucket_calls.append(bucket)

    def download_file(self, bucket, key, local_path):
        self.downloaded.append((bucket, key, local_path))
        return local_path

    def get_object(self, bucket, key):
        self.get_object_calls.append((bucket, key))
        return self._objects[(bucket, key)]


@pytest.fixture(autouse=True)
def _mock_duration(monkeypatch):
    monkeypatch.setattr(highlight_ingest, "_probe_duration_sec", lambda path: 123.4)


def _config():
    return ServiceConfig(gemini_upload_poll_interval_sec=0.01, gemini_upload_poll_timeout_sec=5.0)


def _request(**kwargs):
    return TrackRequest(bucket="src-bucket", key="videos/match.mp4", **kwargs)


async def _run_ingest(job_id, request, config, jobs_store, work_dir):
    """Exercise the explicit source-preparation and completion phases."""
    source = await highlight_ingest.prepare_highlight_source(
        job_id, request, config, jobs_store, work_dir,
    )
    return await highlight_ingest.complete_highlight_ingest_stage(
        job_id, request, config, jobs_store, work_dir, source, MagicMock(),
    )


@pytest.mark.asyncio
async def test_fresh_job_uploads_and_polls_to_active(monkeypatch, tmp_path):
    monkeypatch.setattr(highlight_ingest, "_make_s3_for_bucket", lambda config, bucket: FakeS3({}))

    uploaded = SimpleNamespace(name="files/abc")
    active = SimpleNamespace(
        name="files/abc", uri="https://generativelanguage.googleapis.com/v1/files/abc",
        mime_type="video/mp4", expiration_time=datetime.now(timezone.utc) + timedelta(hours=48),
    )
    monkeypatch.setattr(highlight_ingest.gemini_upload, "upload_video_to_gemini", AsyncMock(return_value=uploaded))
    monkeypatch.setattr(highlight_ingest.gemini_upload, "poll_until_active", AsyncMock(return_value=active))

    jobs_store = FakeJobsStore(checkpoints=[])
    result = await _run_ingest("job-1", _request(), _config(), jobs_store, str(tmp_path))

    assert result.gemini_file_uri == active.uri
    assert result.gemini_file_name == "files/abc"
    assert result.gemini_file_mime_type == "video/mp4"
    assert result.video_duration_sec == 123.4

    highlight_ingest.gemini_upload.upload_video_to_gemini.assert_awaited_once()
    highlight_ingest.gemini_upload.poll_until_active.assert_awaited_once()

    assert len(jobs_store.written) == 1
    _, stage_name, completed, data = jobs_store.written[0]
    assert stage_name.value == "highlight_ingest"
    assert completed is True
    assert data["artifacts"]["gemini_file_uri"] == active.uri


@pytest.mark.asyncio
async def test_resume_with_non_expired_checkpoint_skips_reupload(monkeypatch, tmp_path):
    monkeypatch.setattr(highlight_ingest, "_make_s3_for_bucket", lambda config, bucket: FakeS3({}))

    upload_mock = AsyncMock()
    poll_mock = AsyncMock()
    monkeypatch.setattr(highlight_ingest.gemini_upload, "upload_video_to_gemini", upload_mock)
    monkeypatch.setattr(highlight_ingest.gemini_upload, "poll_until_active", poll_mock)

    future_expiry = (datetime.now(timezone.utc) + timedelta(hours=40)).isoformat()
    existing_checkpoint = {
        "stage_name": "highlight_ingest",
        "completed": True,
        "checkpoint_data": {
            "artifacts": {
                "gemini_file_uri": "https://.../files/existing",
                "gemini_file_name": "files/existing",
                "gemini_file_mime_type": "video/mp4",
                "gemini_file_expiration": future_expiry,
            },
        },
    }
    jobs_store = FakeJobsStore(checkpoints=[existing_checkpoint])

    result = await _run_ingest("job-1", _request(), _config(), jobs_store, str(tmp_path))

    assert result.gemini_file_uri == "https://.../files/existing"
    assert result.gemini_file_name == "files/existing"
    upload_mock.assert_not_awaited()
    poll_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_with_expired_checkpoint_reuploads(monkeypatch, tmp_path):
    monkeypatch.setattr(highlight_ingest, "_make_s3_for_bucket", lambda config, bucket: FakeS3({}))

    uploaded = SimpleNamespace(name="files/new")
    active = SimpleNamespace(
        name="files/new", uri="https://.../files/new", mime_type="video/mp4",
        expiration_time=datetime.now(timezone.utc) + timedelta(hours=48),
    )
    monkeypatch.setattr(highlight_ingest.gemini_upload, "upload_video_to_gemini", AsyncMock(return_value=uploaded))
    monkeypatch.setattr(highlight_ingest.gemini_upload, "poll_until_active", AsyncMock(return_value=active))

    # Expires in 2 minutes — under the 10-minute reuse safety margin.
    near_expiry = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
    existing_checkpoint = {
        "stage_name": "highlight_ingest",
        "completed": True,
        "checkpoint_data": {
            "artifacts": {
                "gemini_file_uri": "https://.../files/old",
                "gemini_file_name": "files/old",
                "gemini_file_mime_type": "video/mp4",
                "gemini_file_expiration": near_expiry,
            },
        },
    }
    jobs_store = FakeJobsStore(checkpoints=[existing_checkpoint])

    result = await _run_ingest("job-1", _request(), _config(), jobs_store, str(tmp_path))

    assert result.gemini_file_uri == "https://.../files/new"
    highlight_ingest.gemini_upload.upload_video_to_gemini.assert_awaited_once()
    highlight_ingest.gemini_upload.poll_until_active.assert_awaited_once()


@pytest.mark.asyncio
async def test_poll_failure_propagates_never_swallowed(monkeypatch, tmp_path):
    """A FAILED/timeout Files-API state must raise all the way out of the
    ingest stage — no fallback, no fabricated ACTIVE state."""
    monkeypatch.setattr(highlight_ingest, "_make_s3_for_bucket", lambda config, bucket: FakeS3({}))

    uploaded = SimpleNamespace(name="files/abc")
    monkeypatch.setattr(highlight_ingest.gemini_upload, "upload_video_to_gemini", AsyncMock(return_value=uploaded))
    monkeypatch.setattr(
        highlight_ingest.gemini_upload, "poll_until_active",
        AsyncMock(side_effect=RuntimeError("state=FAILED")),
    )

    jobs_store = FakeJobsStore(checkpoints=[])

    with pytest.raises(RuntimeError, match="FAILED"):
        await _run_ingest("job-1", _request(), _config(), jobs_store, str(tmp_path))

    assert jobs_store.written == []  # no checkpoint written on failure
