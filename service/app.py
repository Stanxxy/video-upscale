import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.ws_manager import WSManager
from service.routes import router, init_routes

# Ensure service loggers emit INFO (uvicorn only configures its own loggers)
logging.getLogger("service").setLevel(logging.INFO)
if not logging.getLogger("service").handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.getLogger("service").addHandler(_h)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ServiceConfig()
    job_store = InMemoryJobStore()
    ws_manager = WSManager()
    init_routes(config, job_store, ws_manager)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="BJJ Video Tracking & Analysis Service",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
