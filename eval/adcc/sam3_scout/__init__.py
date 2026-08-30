"""SAM3 scout package — gx10 offline arms A/B/C vs ADCC gold windows.

See ``working_log/plans/2026-07-20-sam3-scout-implementation.md``.
"""

__all__ = [
    "SCOUT_GEMINI_MODEL",
    "SCOUT_FPS",
    "SCOUT_FRAME_QUALITY",
    "FORBIDDEN_GEMINI_MODELS",
]

SCOUT_GEMINI_MODEL = "gemini-3.5-flash"
SCOUT_FPS = 10.0
SCOUT_FRAME_QUALITY = "high"
FORBIDDEN_GEMINI_MODELS = frozenset({"gemini-3.1-flash-lite"})
