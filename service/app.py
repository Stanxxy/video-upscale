import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

# SAM2 uses tqdm in propagate_in_video; disable bar output so service.log stays readable.
os.environ.setdefault("TQDM_DISABLE", "1")

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

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

# Ensure service loggers emit INFO (uvicorn only configures its own loggers)
service_logger = logging.getLogger("service")
service_logger.setLevel(logging.INFO)
service_logger.propagate = False
if not service_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    service_logger.addHandler(_h)

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
    )

    # Start heartbeat
    heartbeat = HeartbeatTask(jobs_store, INSTANCE_ID)
    heartbeat.start()

    # Start recovery manager for stale worker-owned jobs
    recovery = RecoveryManager(
        jobs_store,
        INSTANCE_ID,
        heartbeat_bucket_hours=config.recovery_heartbeat_bucket_hours,
        recover_job=recover_interrupted_job,
    )
    recovery.start()
    service_logger.info("RecoveryManager + HeartbeatTask started")

    # Pre-warm SAM2 models to eliminate cold-start latency on first job.
    # SAM2Manager.__init__ calls SAM2VideoPredictor.from_pretrained(), which
    # downloads/caches weights and loads them onto the device — no video file
    # needed. Running in a thread pool avoids blocking the async event loop.
    async def _prewarm_sam2():
        import asyncio

        def _load_models():
            try:
                from tracking_pipeline.sam2_manager import SAM2Manager

                for model_id in [
                    "facebook/sam2.1-hiera-tiny",       # fast mode
                    "facebook/sam2.1-hiera-base-plus",  # standard mode
                ]:
                    service_logger.info("Pre-warming SAM2 model: %s", model_id)
                    mgr = SAM2Manager(model_id=model_id)
                    del mgr
                    service_logger.info("Pre-warm complete: %s", model_id)
            except Exception as e:
                service_logger.warning("SAM2 pre-warm failed (non-fatal): %s", e)

        await asyncio.get_event_loop().run_in_executor(None, _load_models)

    if not config.disable_prewarm:
        asyncio.ensure_future(_prewarm_sam2())

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

    app.include_router(router)
    return app


app = create_app()
