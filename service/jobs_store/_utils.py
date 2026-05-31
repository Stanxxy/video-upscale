"""Shared Keyspaces store utilities."""

from datetime import datetime, timezone


def as_utc_aware(dt: datetime | None) -> datetime | None:
    """cassandra-driver returns naive UTC wall times from Keyspaces; normalize for Python compares."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
