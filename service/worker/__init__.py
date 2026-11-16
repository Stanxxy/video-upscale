"""Lazy exports for the highlight worker; keeps service startup light."""
from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "SNSPublisher": ("service.sns", "SNSPublisher"),
    "run_highlight_job": ("service.worker.highlight_orchestrator", "run_highlight_job"),
    "_make_s3": ("service.worker.helpers", "_make_s3"),
    "_is_cancelled": ("service.worker.helpers", "_is_cancelled"),
    "_make_worker_state": ("service.worker.progress", "_make_worker_state"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
