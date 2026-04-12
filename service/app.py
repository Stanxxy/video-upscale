import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.ws_manager import WSManager
from service.routes import router, init_routes

# Ensure service loggers emit INFO (uvicorn only configures its own loggers)
service_logger = logging.getLogger("service")
service_logger.setLevel(logging.INFO)
service_logger.propagate = False
if not service_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    service_logger.addHandler(_h)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ServiceConfig()
    job_store = InMemoryJobStore()
    ws_manager = WSManager()
    init_routes(config, job_store, ws_manager)

    async def _periodic_cleanup():
        from service.routes import _cleanup_orphaned_tasks
        while True:
            await asyncio.sleep(30)
            await _cleanup_orphaned_tasks()

    import asyncio
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="BJJ Video Tracking & Analysis Service",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
