"""Exact source<->provider millisecond mapping at the Gemini boundary.

Core requests, checkpoints and results stay on the Analyzer source timeline. Gemini
sees the file with its first video frame at 0, so only the adapter shifts: prompt
and window times down by the measured offset, returned times back up by it.
"""

# Every time-bearing field of the core prompt context and of every node output.
TIME_FIELDS = frozenset(
    {"start_ms", "end_ms", "at_ms", "corrected_at_ms", "transition_at_ms"}
)


def _walk(value, change, strict):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in TIME_FIELDS and type(item) is int:
                out[key] = change(item)
            elif strict and key.endswith("_ms") and key not in TIME_FIELDS:
                raise ValueError(f"unmapped time field {key}")
            else:
                out[key] = _walk(item, change, strict)
        return out
    if isinstance(value, list):
        return [_walk(item, change, strict) for item in value]
    return value


def to_provider(value, offset_ms):
    """Source ms -> provider ms; frames before the first video frame clamp to 0."""
    return _walk(value, lambda ms: max(0, ms - offset_ms), True)


def to_source(value, offset_ms):
    """Provider ms -> source ms. Unknown shapes are left for strict validation."""
    return _walk(value, lambda ms: ms + offset_ms, False)
