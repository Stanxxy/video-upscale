from contextlib import asynccontextmanager
from fastapi import FastAPI

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.routes import router, init_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ServiceConfig()
    job_store = InMemoryJobStore()
    init_routes(config, job_store)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="BJJ Video Analysis Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
