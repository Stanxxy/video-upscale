"""Request compatibility for Gemini models exposed by the QA VLM Studio.

Gemini 3.6/3.7 use level-based thinking and do not accept the legacy sampling
controls used by older Studio models. Keep those differences at request-build
boundaries so adding the models to an allowlist produces runnable experiments,
not dropdown entries that fail at the Gemini API.
"""
from __future__ import annotations

from typing import Any


LATEST_FLASH_MODELS: tuple[str, ...] = ("gemini-3.7-flash", "gemini-3.6-flash")
_DEFAULT_SAMPLING_ONLY_MODELS = frozenset(LATEST_FLASH_MODELS)


def generation_sampling_kwargs(model_id: str, *, temperature: float) -> dict[str, Any]:
    """Return sampling kwargs supported by ``model_id``.

    Existing models retain their exact temperature contract. Gemini 3.6/3.7
    intentionally use the provider default and therefore omit ``temperature``.
    """
    if model_id in _DEFAULT_SAMPLING_ONLY_MODELS:
        return {}
    return {"temperature": temperature}
