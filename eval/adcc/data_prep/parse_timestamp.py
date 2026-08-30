"""Dual-format ADCC gold timestamp parser.

The raw ADCC ``metadata_and_events.json`` events carry ``timestamp`` as either
``MM:SS`` or ``HH:MM:SS`` (verified: 615/3,260 events, ~19%, use the 3-field
form — a naive 2-field parser silently corrupts a fifth of the gold set, see
plan §3/§6.5). This module does exactly one thing: parse that string into
integer total seconds, branching on field count, and RAISING (never
coercing/defaulting) on anything else — per CLAUDE.md's no-fallback-logic
policy and the plan's explicit "fail-loud" requirement (§6.1).
"""
from __future__ import annotations

__all__ = ["parse_timestamp", "TimestampParseError"]


class TimestampParseError(ValueError):
    """Raised for any timestamp that is not a well-formed ``MM:SS``/``HH:MM:SS``
    string of non-negative integer fields. Never caught-and-coerced by callers —
    a malformed timestamp is a real data-quality defect to surface, not paper
    over (plan §4.5: "Malformed timestamp/position — data-quality drop — fail
    loud, list them")."""


def parse_timestamp(ts: str) -> int:
    """Parse an ADCC gold ``timestamp`` string into total seconds.

    Accepts exactly two shapes:
      - ``MM:SS``    (2 colon-separated fields)
      - ``HH:MM:SS`` (3 colon-separated fields)

    Every field must be a base-10 non-negative integer (no floats, no signs,
    no whitespace-only fields). ``MM``/``SS`` must be in ``[0, 59]`` — the
    plan's samples show minutes running past 59 rolling into the ``HH`` field
    (e.g. an 8-hour livestream's multi-match VOD timestamps), so a malformed
    ``MM``/``SS`` >= 60 is a real data defect, not a valid "long" duration.

    Raises ``TimestampParseError`` on anything else — any other field count,
    a non-integer field, or an out-of-range field. Never silently coerces.
    """
    if not isinstance(ts, str):
        raise TimestampParseError(f"timestamp must be a str, got {type(ts).__name__}: {ts!r}")

    fields = ts.split(":")
    if len(fields) not in (2, 3):
        raise TimestampParseError(
            f"malformed timestamp {ts!r}: expected 'MM:SS' or 'HH:MM:SS' "
            f"(2 or 3 colon-separated fields), got {len(fields)} field(s)"
        )

    parsed: list[int] = []
    for field in fields:
        # int() itself would accept a leading '+'/'-' or surrounding
        # whitespace ("  12".strip() semantics) — reject anything that isn't
        # a bare run of digits so "malformed" truly means malformed, not
        # "int() was lenient about it".
        if not field.isdigit():
            raise TimestampParseError(
                f"malformed timestamp {ts!r}: field {field!r} is not a bare non-negative integer"
            )
        parsed.append(int(field))

    if len(parsed) == 2:
        hours, minutes, seconds = 0, parsed[0], parsed[1]
    else:
        hours, minutes, seconds = parsed

    if not (0 <= minutes <= 59):
        raise TimestampParseError(f"malformed timestamp {ts!r}: minutes field {minutes} out of [0,59]")
    if not (0 <= seconds <= 59):
        raise TimestampParseError(f"malformed timestamp {ts!r}: seconds field {seconds} out of [0,59]")

    return hours * 3600 + minutes * 60 + seconds
