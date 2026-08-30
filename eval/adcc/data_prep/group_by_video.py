"""``video_links.json`` -> ``{embed_url: [match_id, ...]}`` grouping + the
corpus-count assertions that de-risk the whole harness (plan §6.5): gold is
keyed by ``match_id``, video is keyed by URL, N:1 video->match, and a harness
built the "obvious" way (per-match loop, one-video-per-match assumption)
misaligns silently. This module makes the N:1 grouping a first-class, asserted
artifact rather than an implicit assumption buried in a loop.

``video_file_stem = md5(embed_url).hexdigest()`` (verified against all 259
links in ``video_links.json``, plan §3).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

__all__ = [
    "compute_video_stem",
    "group_by_video",
    "usable_match_ids",
    "assert_corpus_counts",
    "CorpusAssertionError",
]


class CorpusAssertionError(AssertionError):
    """Raised when the reproduced corpus counts (usable match_ids / unique
    videos) don't match the expected values. This is a hard stop, not a
    warning — a mismatch means either the source JSONs changed underneath us
    or the on-disk stem set is wrong, either of which silently misaligns every
    downstream score."""


def compute_video_stem(embed_url: str) -> str:
    """``video_file_stem = md5(embed_url).hexdigest()`` — the cracked
    URL->on-disk-filename encoding (plan §3), re-derived here rather than
    hardcoded per video so it stays correct if the corpus grows."""
    return hashlib.md5(embed_url.encode("utf-8")).hexdigest()


def group_by_video(video_links: dict[str, str]) -> dict[str, list[str]]:
    """All ``match_id``s in ``video_links.json``, grouped by their embed URL —
    UNFILTERED (no events/on-disk check yet; that's ``usable_match_ids``
    below). ``video_links`` maps ``match_id -> embed_url`` 1:1; this inverts
    it to ``embed_url -> [match_id, ...]`` (N:1), which is the whole point —
    losing this grouping is exactly the "obvious way" bug the plan warns
    about (§6.5)."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for match_id, embed_url in video_links.items():
        grouped[embed_url].append(match_id)
    return dict(grouped)


def usable_match_ids(
    video_links: dict[str, str],
    events: dict[str, list],
    on_disk_stems: set[str],
) -> set[str]:
    """A ``match_id`` is "usable" iff it has (a) a non-empty ``events`` entry,
    (b) a ``video_links`` entry, and (c) that entry's md5 stem is present on
    disk (``on_disk_stems``). All three conditions are asserted, never
    assumed — a match with events but no video (or vice versa) is silently
    dropped from the harness rather than crashing at score time on a missing
    file."""
    usable: set[str] = set()
    for match_id, embed_url in video_links.items():
        if not events.get(match_id):
            continue
        stem = compute_video_stem(embed_url)
        if stem not in on_disk_stems:
            continue
        usable.add(match_id)
    return usable


def assert_corpus_counts(
    video_links: dict[str, str],
    events: dict[str, list],
    on_disk_stems: set[str],
    *,
    expected_usable_matches: int = 258,
    expected_unique_videos: int = 103,
) -> tuple[set[str], set[str]]:
    """Re-derive the usable match_id set + unique on-disk video stem set and
    hard-assert them against the plan's verified corpus facts (§3): 258
    usable match_ids on 103 unique videos. Never hardcode these numbers
    downstream — always re-derive-and-assert here first, per the plan's
    explicit "all re-asserted at build time, never hardcoded" instruction.

    Returns ``(usable_ids, unique_video_stems)`` for the caller to build the
    gold snapshot from, only after both counts check out.
    """
    usable = usable_match_ids(video_links, events, on_disk_stems)
    unique_stems = {compute_video_stem(video_links[mid]) for mid in usable}

    if len(usable) != expected_usable_matches:
        raise CorpusAssertionError(
            f"usable match_id count mismatch: got {len(usable)}, expected {expected_usable_matches}. "
            "Source JSONs or on-disk stem set changed underneath the harness — investigate before "
            "trusting any downstream score."
        )
    if len(unique_stems) != expected_unique_videos:
        raise CorpusAssertionError(
            f"unique video count mismatch: got {len(unique_stems)}, expected {expected_unique_videos}. "
            "Source JSONs or on-disk stem set changed underneath the harness — investigate before "
            "trusting any downstream score."
        )
    return usable, unique_stems
