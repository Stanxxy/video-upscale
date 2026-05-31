"""Shared module state and helpers for route handlers."""

import asyncio
import logging
import os

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
