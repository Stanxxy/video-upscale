"""Shared module state and helpers for route handlers."""

import asyncio
import logging
import os
from datetime import datetime, timezone

from fastapi import HTTPException

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.s3 import S3Client

QA_HTML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "qa_client",
    "index.html",
)

logger = logging.getLogger("service.routes")

_config: ServiceConfig | None = None
_job_store: InMemoryJobStore | None = None
_jobs_store: JobsStore | None = None
_job_semaphore: asyncio.Semaphore | None = None
_instance_id: str = ""
_active_tasks: dict[str, asyncio.Task] = {}


def init_routes(
    config: ServiceConfig,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
    instance_id: str = "",
):
    global _config, _job_store, _jobs_store, _job_semaphore, _instance_id
    _config = config
    _job_store = job_store
    _jobs_store = jobs_store
    _job_semaphore = asyncio.Semaphore(config.max_concurrent_jobs)
    _instance_id = instance_id


# G7 cost guardrails (in-memory).
# Counter lives in process memory and RESETS on restart. Accepted tradeoff;
# with max_concurrent_jobs=1 and a single engine instance the count is
# authoritative for the running process.
_daily_count_date: str = ""
_daily_count: int = 0


def _check_kill_switch() -> None:
    """Reject all admission when the global kill switch is engaged."""
    if _config is not None and _config.analysis_disabled:
        raise HTTPException(503, "Analysis temporarily disabled")


def _admit_daily() -> None:
    """Admit one NEW analysis against today's UTC daily cap, or reject with 429.

    Only NEW analyses call this; resume/correction paths are exempt.
    """
    global _daily_count_date, _daily_count
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _daily_count_date:
        _daily_count_date = today
        _daily_count = 0
    limit = _config.max_daily_analyses if _config is not None else 0
    if _daily_count + 1 > limit:
        raise HTTPException(429, "Daily analysis limit reached")
    _daily_count += 1


def _require_write(ok: bool, operation: str) -> None:
    if not ok:
        raise HTTPException(500, f"Failed to persist {operation}")


def _s3_client() -> S3Client:
    assert _config is not None
    return S3Client(
        region=_config.aws_region,
        endpoint_url=_config.s3_endpoint_url or None,
        access_key_id=_config.aws_access_key_id or None,
        secret_access_key=_config.aws_secret_access_key or None,
    )
