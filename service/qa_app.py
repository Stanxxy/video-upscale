"""Standalone ASGI app for the QA VLM Studio lane — Gemini-only, no infra.

``service.app:app`` (the full production app) is not usable for local QA VLM
work: its ``lifespan`` unconditionally connects to Cassandra/Keyspaces
(``KeyspacesClient()``), starts ``HeartbeatTask``/``RecoveryManager``, and
pre-warms SAM2 — none of which the QA VLM Studio page
(``qa_client/vlm_studio.html``) needs. Without a running Cassandra cluster,
that startup hard-fails with ``cassandra.cluster.NoHostAvailable`` before the
process ever binds a port.

This module mounts ONLY the QA routes (``/qa/vlm-chat``, ``/qa/vlm-image``,
``/qa/identity-analyze``) plus a trivial ``/health``, with the SAME CORS
policy as ``service.app``. It deliberately imports the route callables
directly from their leaf modules (``service.routes.qa_vlm``, ``service.routes.qa``)
rather than via the ``service.routes`` package barrel — the barrel eagerly
imports ``service.routes.jobs``/``human_loop``/``health`` (which pull in
``torch`` and the tracking/worker stack) at package-import time.

Only requirement to run: ``BJJ_GEMINI_API_KEY``/``GEMINI_API_KEY`` in the
engine ``.env`` (loaded via ``service.config.ServiceConfig``), plus yt-dlp +
ffmpeg on PATH for ``/qa/vlm-image``'s frame extraction.

Production ``service/app.py`` is UNCHANGED by this file — it is a fully
separate app, not a modification of the existing one.
"""
from __future__ import annotations

import logging

from dotenv import load_dotenv

# Load .env early, same as service/app.py, so BJJ_GEMINI_API_KEY / GEMINI_API_KEY
# and any BJJ_-prefixed ServiceConfig overrides are available.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.routes.qa import identity_analyze
from service.routes.qa_vlm import vlm_chat, vlm_image
from service.routes.state import init_routes

logger = logging.getLogger("service.qa_app")


def create_qa_app() -> FastAPI:
    app = FastAPI(
        title="BJJ QA VLM Studio Backend (standalone, no infra)",
        version="1.0.0",
    )

    # Same CORS policy as service/app.py's production CORSMiddleware block —
    # the QA client (qa_client/vlm_studio.html) is served on a separate origin
    # during local dev via `qa_client/serve.sh` (port 8765) or a static file
    # server (e.g. VS Code Live Server, port 5500).
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

    # QA routes only need `route_state._config` (for `config.gemini_api_key`)
    # and, for `/qa/identity-analyze`, `route_state._s3_client()` (built lazily
    # from `_config` on demand). They never touch `_job_store`/`_jobs_store`/
    # `_job_semaphore`, so an in-memory job store is wired only to satisfy
    # `init_routes`'s signature — no Keyspaces/JobsStore is constructed here.
    config = ServiceConfig()
    init_routes(config, InMemoryJobStore(), None, instance_id="qa-standalone")

    if not config.gemini_api_key:
        logger.warning(
            "GEMINI_API_KEY / BJJ_GEMINI_API_KEY is not set — /qa/vlm-chat and "
            "/qa/vlm-image will 400 until the engine .env is configured."
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": "qa-standalone", "gemini_key_configured": bool(config.gemini_api_key)}

    app.post("/qa/vlm-chat")(vlm_chat)
    app.post("/qa/vlm-image")(vlm_image)
    app.post("/qa/identity-analyze")(identity_analyze)

    return app


app = create_qa_app()
