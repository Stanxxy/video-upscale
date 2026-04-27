"""Thin async Keyspaces (Cassandra) client for the engine repo.

Mirrors shared_lib's AsyncKeyspacesClient but is fully standalone
so the engine repo does not need to import shared_lib.
"""

import asyncio
import logging
import os
import ssl
from functools import partial
from typing import Any

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel

logger = logging.getLogger(__name__)


class KeyspacesClient:
    """Synchronous cassandra-driver wrapped with run_in_executor for async use."""

    def __init__(self) -> None:
        self.keyspace = os.getenv("KEYSPACE_NAME", "video_analysis")
        username = os.getenv("KEYSPACES_USERNAME", "")
        password = os.getenv("KEYSPACES_PASSWORD", "")
        contact_points = os.getenv("KEYSPACES_CONTACT_POINTS", "").split(",")
        port = int(os.getenv("KEYSPACES_PORT", "9142"))
        use_ssl = os.getenv("KEYSPACES_USE_SSL", "true").lower() == "true"

        ssl_context = None
        if use_ssl:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        auth_provider = PlainTextAuthProvider(
            username=username,
            password=password,
        )

        self.cluster = Cluster(
            contact_points=contact_points,
            auth_provider=auth_provider,
            protocol_version=4,
            ssl_context=ssl_context,
            port=port,
        )
        self.session = self.cluster.connect(self.keyspace)
        # Amazon Keyspaces requires LOCAL_QUORUM for all operations
        self.session.default_consistency_level = ConsistencyLevel.LOCAL_QUORUM
        logger.info(
            "KeyspacesClient connected to %s (keyspace=%s)",
            contact_points,
            self.keyspace,
        )

    # ── async helpers ────────────────────────────────────────────────

    async def execute(
        self, query: str, params: list[Any] | None = None
    ) -> list[Any]:
        """Execute a read query and return all rows."""
        loop = asyncio.get_event_loop()
        try:
            if params:
                result = await loop.run_in_executor(
                    None, partial(self.session.execute, query, params)
                )
            else:
                result = await loop.run_in_executor(
                    None, partial(self.session.execute, query)
                )
            return result.all()
        except Exception as e:
            logger.error("Keyspaces read error: %s", e)
            return []

    async def execute_write(
        self, query: str, params: list[Any] | None = None
    ) -> bool:
        """Execute a write query. Returns True on success."""
        loop = asyncio.get_event_loop()
        try:
            if params:
                await loop.run_in_executor(
                    None, partial(self.session.execute, query, params)
                )
            else:
                await loop.run_in_executor(
                    None, partial(self.session.execute, query)
                )
            return True
        except Exception as e:
            logger.error("Keyspaces write error: %s", e)
            return False

    def close(self) -> None:
        """Shut down the driver connection."""
        try:
            self.session.shutdown()
            self.cluster.shutdown()
            logger.info("KeyspacesClient connection closed.")
        except Exception as e:
            logger.warning("Error closing KeyspacesClient: %s", e)
