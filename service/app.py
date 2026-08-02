import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

# SAM2 uses tqdm in propagate_in_video; disable bar output so service.log stays readable.
os.environ.setdefault("TQDM_DISABLE", "1")

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Load .env early so non-prefixed vars (KEYSPACES_*, etc.) are available
load_dotenv()

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.keyspaces_client import KeyspacesClient
from service.jobs_store import JobsStore
from service.heartbeat import HeartbeatTask
from service.reconciler import RecoveryManager
from service.routes import (
    router,
    init_routes,
    recover_interrupted_job,
    drain_orphan_pending_jobs_on_startup,
    bootstrap_recovery_on_startup,
)

class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record — unified format for Grafana/Loki ingestion."""

    _SERVICE = "bjj-vision-engine"

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%f+00:00"),
            "level": record.levelname.lower(),
            "service": self._SERVICE,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# Install JSON formatter on root (uvicorn, fastapi) and service logger
_json_fmt = _JsonFormatter()
_root = logging.getLogger()
if not any(getattr(h, "_bjj_json", False) for h in _root.handlers):
    for h in list(_root.handlers):
        _root.removeHandler(h)
    _rh = logging.StreamHandler()
    _rh.setFormatter(_json_fmt)
    _rh._bjj_json = True  # type: ignore[attr-defined]
    _root.setLevel(logging.INFO)
    _root.addHandler(_rh)

service_logger = logging.getLogger("service")
service_logger.setLevel(logging.INFO)
service_logger.propagate = False
if not any(getattr(h, "_bjj_json", False) for h in service_logger.handlers):
    _sh = logging.StreamHandler()
    _sh.setFormatter(_json_fmt)
    _sh._bjj_json = True  # type: ignore[attr-defined]
    service_logger.addHandler(_sh)

INSTANCE_ID = str(uuid.uuid4())[:8]  # unique per process


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ServiceConfig()
    job_store = InMemoryJobStore()

    # Initialize Keyspaces
    ks_client = KeyspacesClient()
    jobs_store = JobsStore(ks_client)

    init_routes(config, job_store, jobs_store, instance_id=INSTANCE_ID)

    await drain_orphan_pending_jobs_on_startup(
        INSTANCE_ID,
        heartbeat_bucket_hours=config.recovery_heartbeat_bucket_hours,
    )

    # Catch RUNNING/INTERRUPTED jobs orphaned by the previous process before
    # the periodic reconciler's 90s stale-window would otherwise wait. Runs
    # BEFORE HeartbeatTask.start() and RecoveryManager.start() so the
    # periodic loops do not interleave with bootstrap.
    await bootstrap_recovery_on_startup(
        INSTANCE_ID,
        recover_interrupted_job,
        heartbeat_bucket_hours=config.recovery_heartbeat_bucket_hours,
        max_heartbeat_age_hours=config.recovery_max_heartbeat_age_hours,
    )

    # Start heartbeat
    heartbeat = HeartbeatTask(jobs_store, INSTANCE_ID)
    heartbeat.start()

    # Start recovery manager for stale worker-owned jobs
    recovery = RecoveryManager(
        jobs_store,
        INSTANCE_ID,
        heartbeat_bucket_hours=config.recovery_heartbeat_bucket_hours,
        max_heartbeat_age_seconds=(
            None if config.recovery_max_heartbeat_age_hours == 0
            else config.recovery_max_heartbeat_age_hours * 3600.0
        ),
        recover_job=recover_interrupted_job,
    )
    recovery.start()
    service_logger.info("RecoveryManager + HeartbeatTask started")

    try:
        yield
    finally:
        heartbeat.stop()
        recovery.stop()
        ks_client.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="BJJ Video Tracking & Analysis Service",
        version="2.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_unhandled_http_exceptions(request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException:
            raise
        except Exception:
            service_logger.exception(
                "Unhandled exception for %s %s",
                request.method,
                request.url.path,
            )
            raise

    # QA client (qa_client/index.html) is served on a separate origin during local dev.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:8765",
            "http://localhost:8765",
            "http://127.0.0.1:5500",
            "http://localhost:5500",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
