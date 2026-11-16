"""OpenRouter chat completions: one call, billed cost and latency recorded.

Privacy: every call asks for providers that neither retain nor train on prompts
(``data_collection: deny`` and ``zdr: true``); a model without such a provider fails
instead of falling back to one that keeps the data.
"""
from __future__ import annotations

import os
import time

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"


class ProviderError(RuntimeError):
    pass


def complete(model: str, messages: list, *, schema: dict | None, reasoning: str = "high",
             max_tokens: int = 32000, timeout_s: float = 600) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ProviderError("OPENROUTER_API_KEY is not set")
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning": {"effort": reasoning},
        "provider": {"data_collection": "deny", "zdr": True},
        "usage": {"include": True},
    }
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "answer", "strict": True, "schema": schema}}
    started = time.monotonic()
    response = httpx.post(URL, json=body, timeout=timeout_s,
                          headers={"Authorization": f"Bearer {key}", "X-Title": "bjj-golden-prelabel"})
    latency = time.monotonic() - started
    if response.status_code != 200:
        raise ProviderError(f"{response.status_code}: {response.text[:400]}")
    data = response.json()
    if "error" in data:
        raise ProviderError(str(data["error"])[:400])
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return dict(
        text=choice["message"].get("content") or "",
        finish_reason=choice.get("finish_reason"),
        provider=data.get("provider"),
        model=data.get("model"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        cost=usage.get("cost"),
        latency_s=round(latency, 2),
        id=data.get("id"),
    )
