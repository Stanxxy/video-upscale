import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

# SAM2 uses tqdm in propagate_in_video; disable bar output so service.log stays readable.
os.environ.setdefault("TQDM_DISABLE", "1")

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env early so non-prefixed vars (KEYSPACES_*, etc.) are available
load_dotenv()

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.keyspaces_client import KeyspacesClient
from service.jobs_store import JobsStore
from service.heartbeat import HeartbeatTask
from service.reconciler import Reconciler
from service.routes import router, init_routes

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

    # Start heartbeat
    heartbeat = HeartbeatTask(jobs_store, INSTANCE_ID)
    heartbeat.start()

    # Run reconciler
    reconciler = Reconciler(jobs_store, INSTANCE_ID)
    await reconciler.run_on_startup()

    try:
        yield
    finally:
        heartbeat.stop()
        ks_client.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="BJJ Video Tracking & Analysis Service",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
