"""Measured Gemini token-cost accounting for the QA Position Timeline DAG.

The pipeline runner owns node attribution because it is the one layer that
sees every streamed call result together with the stage that produced it.
USD values use returned usage metadata; missing metadata never becomes a
zero-cost call.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Iterator, Optional


PRICING_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_TIER = "standard_paid"

_PROMOTIONAL_END = date(2026, 12, 31)
_PROMOTIONAL_EFFECTIVE_FROM_BY_MODEL = {
    "gemini-3.8-flash": "2026-09-02",
    "gemini-3.7-flash": "2026-08-26",
    "gemini-3.6-flash": "2026-08-26",
}
_LATEST_FLASH_MODELS = frozenset(_PROMOTIONAL_EFFECTIVE_FROM_BY_MODEL)


def _rate_for(model: Optional[str], on_date: date) -> Optional[dict]:
    """Return Google's published Standard paid-tier rate for supported models."""
    if model not in _LATEST_FLASH_MODELS:
        return None
    if on_date <= _PROMOTIONAL_END:
        return {
            "input_usd_per_million": 0.75,
            "output_usd_per_million": 3.75,
            "cached_input_usd_per_million": 0.075,
            "rate_effective_from": _PROMOTIONAL_EFFECTIVE_FROM_BY_MODEL[model],
            "rate_valid_through": "2026-12-31",
        }
    return {
        "input_usd_per_million": 1.50,
        "output_usd_per_million": 7.50,
        "cached_input_usd_per_million": 0.15,
        "rate_effective_from": "2027-01-01",
        "rate_valid_through": None,
    }


def new_node_accumulator(stage) -> dict:
    """Create mutable accounting state for one pipeline stage."""
    return {
        "stage_id": stage.id,
        "stage_type": stage.type,
        "label": stage.label,
        "model": stage.config.get("model"),
        "call_count": 0,
        "metered_call_count": 0,
        "unmetered_call_count": 0,
        "prompt_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "cached_content_token_count": 0,
        "total_token_count": 0,
    }


def _usage_entries(value: Any) -> Iterator[Any]:
    """Yield each call-level ``usage_metadata`` slot in a streamed event."""
    if isinstance(value, dict):
        if "usage_metadata" in value:
            yield value["usage_metadata"]
        for key, child in value.items():
            if key != "usage_metadata":
                yield from _usage_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _usage_entries(child)


def record_event_usage(accumulator: dict, event: dict) -> None:
    """Attribute every call-level usage slot in ``event`` to one stage."""
    for usage in _usage_entries(event):
        accumulator["call_count"] += 1
        if not isinstance(usage, dict):
            accumulator["unmetered_call_count"] += 1
            continue
        prompt = usage.get("prompt_token_count")
        candidates = usage.get("candidates_token_count")
        if not isinstance(prompt, int) or not isinstance(candidates, int):
            accumulator["unmetered_call_count"] += 1
            continue
        accumulator["metered_call_count"] += 1
        accumulator["prompt_token_count"] += prompt
        accumulator["candidates_token_count"] += candidates
        accumulator["thoughts_token_count"] += int(usage.get("thoughts_token_count") or 0)
        accumulator["cached_content_token_count"] += int(
            usage.get("cached_content_token_count") or 0
        )
        total = usage.get("total_token_count")
        accumulator["total_token_count"] += (
            total
            if isinstance(total, int)
            else prompt
            + candidates
            + int(usage.get("thoughts_token_count") or 0)
        )


def node_cost_snapshot(accumulator: dict, *, on_date: Optional[date] = None) -> dict:
    """Return the public immutable-shaped ledger row for one stage."""
    snapshot = dict(accumulator)
    snapshot.update(
        {
            "pricing_tier": PRICING_TIER,
            "pricing_source_url": PRICING_SOURCE_URL,
            "estimated_usd": 0.0,
            "pricing_status": "zero_calls",
            "input_usd_per_million": None,
            "output_usd_per_million": None,
            "cached_input_usd_per_million": None,
            "rate_effective_from": None,
            "rate_valid_through": None,
        }
    )
    if snapshot["call_count"] == 0:
        if snapshot["model"] is None:
            snapshot["pricing_status"] = "not_applicable"
        return snapshot
    if snapshot["unmetered_call_count"]:
        snapshot["estimated_usd"] = None
        snapshot["pricing_status"] = "usage_unavailable"
        return snapshot

    rate = _rate_for(snapshot["model"], on_date or date.today())
    if rate is None:
        snapshot["estimated_usd"] = None
        snapshot["pricing_status"] = "unpriced_model"
        return snapshot

    cached_tokens = min(
        snapshot["cached_content_token_count"], snapshot["prompt_token_count"]
    )
    regular_input_tokens = snapshot["prompt_token_count"] - cached_tokens
    output_tokens = snapshot["candidates_token_count"] + snapshot["thoughts_token_count"]
    estimated = (
        regular_input_tokens * rate["input_usd_per_million"]
        + cached_tokens * rate["cached_input_usd_per_million"]
        + output_tokens * rate["output_usd_per_million"]
    ) / 1_000_000
    snapshot.update(rate)
    snapshot["estimated_usd"] = round(estimated, 12)
    snapshot["pricing_status"] = "priced"
    return snapshot


def summarize_node_costs(node_costs: list[dict]) -> dict:
    """Build a run total without hiding any unpriced or unmetered call."""
    has_unknown_spend = any(
        row["call_count"] > 0 and row["estimated_usd"] is None for row in node_costs
    )
    rate_rows = [row for row in node_costs if row["pricing_status"] == "priced"]
    return {
        "call_count": sum(row["call_count"] for row in node_costs),
        "metered_call_count": sum(row["metered_call_count"] for row in node_costs),
        "unmetered_call_count": sum(row["unmetered_call_count"] for row in node_costs),
        "prompt_token_count": sum(row["prompt_token_count"] for row in node_costs),
        "candidates_token_count": sum(row["candidates_token_count"] for row in node_costs),
        "thoughts_token_count": sum(row["thoughts_token_count"] for row in node_costs),
        "total_token_count": sum(row["total_token_count"] for row in node_costs),
        "estimated_usd": (
            None
            if has_unknown_spend
            else round(sum(row["estimated_usd"] or 0.0 for row in node_costs), 12)
        ),
        "pricing_tier": PRICING_TIER,
        "pricing_source_url": PRICING_SOURCE_URL,
        "rate_effective_from": (
            rate_rows[0]["rate_effective_from"] if rate_rows else None
        ),
        "rate_valid_through": (
            rate_rows[0]["rate_valid_through"] if rate_rows else None
        ),
        "has_unknown_spend": has_unknown_spend,
    }


__all__ = [
    "PRICING_SOURCE_URL",
    "PRICING_TIER",
    "new_node_accumulator",
    "node_cost_snapshot",
    "record_event_usage",
    "summarize_node_costs",
]
