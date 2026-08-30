"""Pure chunk-planning math for gx10 SAM3 (INS-056).

Defaults match the live gx10 harness:
  SAM3_CHUNK_SECONDS=30
  CHUNK_THRESHOLD_SECONDS = chunk * 1.5  (= 45 when chunk=30)
  concurrency forced to 1 for the scout (never raise without upstream proof)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os

DEFAULT_CHUNK_SECONDS = 30.0
DEFAULT_THRESHOLD_MULTIPLIER = 1.5
SCOUT_MAX_CONCURRENCY = 1


@dataclass(frozen=True)
class ChunkPlan:
    duration_s: float
    chunk_seconds: float
    threshold_seconds: float
    chunked: bool
    n_chunks: int
    concurrency: int

    @property
    def chunk_boundaries(self) -> list[tuple[float, float]]:
        """Inclusive start / exclusive-ish end seconds for each planned chunk."""
        if not self.chunked:
            return [(0.0, self.duration_s)]
        bounds: list[tuple[float, float]] = []
        t = 0.0
        while t < self.duration_s - 1e-9:
            end = min(t + self.chunk_seconds, self.duration_s)
            bounds.append((t, end))
            t = end
        return bounds


def chunk_threshold_seconds(chunk_seconds: float = DEFAULT_CHUNK_SECONDS) -> float:
    return float(chunk_seconds) * DEFAULT_THRESHOLD_MULTIPLIER


def plan_chunks(
    duration_s: float,
    chunk_seconds: float | None = None,
    concurrency: int | None = None,
) -> ChunkPlan:
    """Decide whether to chunk and how many windows.

    Videos with duration <= threshold are a single SAM3 session.
    Longer videos are split into fixed ``chunk_seconds`` windows (no overlap).
    Scout always forces concurrency=1 regardless of env.
    """
    if duration_s < 0:
        raise ValueError(f"duration_s must be >= 0, got {duration_s!r}")
    cs = float(chunk_seconds if chunk_seconds is not None else os.environ.get("SAM3_CHUNK_SECONDS", DEFAULT_CHUNK_SECONDS))
    if cs <= 0:
        raise ValueError(f"chunk_seconds must be > 0, got {cs!r}")
    threshold = chunk_threshold_seconds(cs)
    # Scout lock: never honor a raised MAX_CONCURRENT_SAM3_SESSIONS for this package.
    conc = SCOUT_MAX_CONCURRENCY
    if concurrency is not None and int(concurrency) != SCOUT_MAX_CONCURRENCY:
        raise ValueError(
            f"sam3_scout concurrency must be {SCOUT_MAX_CONCURRENCY} (got {concurrency}); "
            "raising concurrency requires independent SAM3 thread-safety proof (INS-056)."
        )
    if duration_s <= threshold:
        return ChunkPlan(
            duration_s=duration_s,
            chunk_seconds=cs,
            threshold_seconds=threshold,
            chunked=False,
            n_chunks=1 if duration_s > 0 else 0,
            concurrency=conc,
        )
    n = int(math.ceil(duration_s / cs))
    return ChunkPlan(
        duration_s=duration_s,
        chunk_seconds=cs,
        threshold_seconds=threshold,
        chunked=True,
        n_chunks=n,
        concurrency=conc,
    )
