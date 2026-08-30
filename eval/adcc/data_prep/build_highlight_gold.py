"""Highlight-Eval GT-provenance builder (frozen contract §B, NON-NEGOTIABLE —
Brooks's blocker). The highlight-reel ground-truth subset does not exist as a
standalone artifact anywhere upstream; it is DERIVED, deterministically, from
the already-committed ``normalized_gold.jsonl`` by this module, through the
SAME provenanced-artifact pattern as ``build_gold_snapshot.py`` (versioned
predicate, sha256'd source, asserted counts, a checked-in manifest).

**The frozen predicate** (LeCun+Rener, verbatim — do not relitigate)::

    is_highlight(event) :=
        action_class in {submission_choke, submission_arm_lock,
                          submission_leg_lock, submission_unclear}
        OR action_class == back_take
        OR action_class == sweep
        OR (action_class == guard_pass AND outcome == "successful")

Yields ~557 highlights over 258 matches (median 2/match, 17 zero-highlight
matches — a real edge case in the corpus, not a bug: some matches genuinely
have no clean submission/back-take/sweep/successful-pass in the raw ADCC
annotation). ``INCLUDE_TAKEDOWNS`` (default ``False``) optionally also counts
a successful ``takedown_attempt`` as a highlight (~+136) — its state is
recorded in the manifest below, never silently assumed by a downstream
consumer.

**Why this is a separate module, not a code path inside
``build_gold_snapshot.py``**: the highlight predicate is a SCORING-time
policy decision (what counts as "highlight-worthy"), not a raw-data
normalization rule (§4's sequenceType/position/subtype mapping) — keeping it
here means changing the highlight definition never touches the frozen,
already-shipped ``normalized_gold.jsonl``/``manifest.json`` artifacts, and
this module's own output/manifest carries its own independent provenance.

**The one rule that matters for correctness (Brooks):** ``/highlight/score``
and every scorer in ``eval/adcc/score/highlight_scoring.py`` MUST load
``highlight_gold.jsonl`` from disk — NEVER recompute ``is_highlight`` ad hoc
against ``normalized_gold.jsonl`` at scoring time. Reproducibility (same
artifact -> same score, forever, independent of whatever predicate tweak
might be mid-flight in this file) is the entire point of building this as a
committed artifact instead of an inline filter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

__all__ = [
    "PREDICATE_VERSION",
    "SUBMISSION_HIGHLIGHT_CLASSES",
    "INCLUDE_TAKEDOWNS_DEFAULT",
    "is_highlight",
    "build_highlight_gold",
    "sha256_of_file",
    "main",
]

# Bump this string any time ``is_highlight``'s logic changes — the manifest
# records it so a stale ``highlight_gold.jsonl`` built under an older
# predicate version is detectable, never silently trusted as current.
PREDICATE_VERSION = "highlight_predicate_v1"

# The four submission_* subtypes are family-wide highlight-worthy regardless
# of whether the specific lock/choke subtype was resolved (mirrors
# apply_gracie_mapping.compat_classes_for's SUBMISSION family-wide compat
# rule — a submission is a submission for highlight purposes even when
# ``submission_unclear``, the "gave up on subtype, still definitely a
# submission" bucket, per resolve_submission_subtype's own docstring).
SUBMISSION_HIGHLIGHT_CLASSES: frozenset[str] = frozenset(
    {"submission_choke", "submission_arm_lock", "submission_leg_lock", "submission_unclear"}
)

INCLUDE_TAKEDOWNS_DEFAULT: bool = False


def is_highlight(event: dict, *, include_takedowns: bool = INCLUDE_TAKEDOWNS_DEFAULT) -> bool:
    """The frozen predicate, verbatim (see module docstring). ``event`` is one
    already-normalized gold event dict (``normalized_gold.jsonl``'s per-event
    shape: ``action_class``/``outcome``/... — see ``apply_gracie_mapping.py``
    for the full contract). Never mutates ``event``, never guesses a missing
    ``outcome`` (``event["outcome"]`` — no ``.get`` default; every real
    normalized event carries this key, a record missing it is a real
    upstream bug that must surface loudly)."""
    action_class = event["action_class"]
    if action_class in SUBMISSION_HIGHLIGHT_CLASSES:
        return True
    if action_class == "back_take":
        return True
    if action_class == "sweep":
        return True
    if action_class == "guard_pass" and event["outcome"] == "successful":
        return True
    if include_takedowns and action_class == "takedown_attempt" and event["outcome"] == "successful":
        return True
    return False


def build_highlight_gold(
    gold_records: list[dict], *, include_takedowns: bool = INCLUDE_TAKEDOWNS_DEFAULT,
) -> list[dict]:
    """Pure function: the full ``normalized_gold.jsonl`` record list -> one
    highlight-gold record per match_id (same identity fields as the source
    record — ``match_id``/``video_url``/``video_url_md5``/``name`` — plus an
    ``events`` list containing ONLY the highlight-predicate-passing subset,
    filtered in place with each event's original fields intact so
    ``score_highlights``/``render_timeline_marks`` can still show the real
    ``action_class``/``t_s`` per highlight mark).

    Every ``match_id`` from the source is represented, including the ~17
    zero-highlight matches (an empty ``events`` list, never dropped from the
    output entirely — a real, expected edge case per the module docstring,
    not something to silently exclude from the artifact)."""
    records: list[dict] = []
    for rec in gold_records:
        highlight_events = [e for e in rec["events"] if is_highlight(e, include_takedowns=include_takedowns)]
        records.append({
            "match_id": rec["match_id"],
            "video_url": rec["video_url"],
            "video_url_md5": rec["video_url_md5"],
            "name": rec.get("name", ""),
            "events": highlight_events,
        })
    return records


def _write_jsonl(records: list[dict], out_path: Path) -> None:
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-snapshot", type=Path, default=Path("eval/adcc/gold_snapshot/normalized_gold.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("eval/adcc/gold_snapshot"))
    parser.add_argument(
        "--include-takedowns", action="store_true",
        help="also count a successful takedown_attempt as a highlight (default False, ~+136 over the "
             "258-match corpus) — the flag's state is recorded in the manifest, never left implicit.",
    )
    args = parser.parse_args(argv)

    gold_records = [json.loads(line) for line in args.gold_snapshot.read_text().splitlines() if line.strip()]
    records = build_highlight_gold(gold_records, include_takedowns=args.include_takedowns)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "highlight_gold.jsonl"
    _write_jsonl(records, out_path)

    total_highlights = sum(len(r["events"]) for r in records)
    zero_highlight_matches = sum(1 for r in records if not r["events"])
    per_match_counts = sorted(len(r["events"]) for r in records)
    median_highlights = per_match_counts[len(per_match_counts) // 2] if per_match_counts else None

    manifest = {
        "predicate_version": PREDICATE_VERSION,
        "include_takedowns": args.include_takedowns,
        "source_gold_snapshot": str(args.gold_snapshot),
        "source_gold_snapshot_sha256": sha256_of_file(args.gold_snapshot),
        "asserted_counts": {
            "match_count": len(records),
            "total_highlights": total_highlights,
            "zero_highlight_match_count": zero_highlight_matches,
            "median_highlights_per_match": median_highlights,
        },
    }
    manifest_path = args.out_dir / "highlight_gold_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        f"wrote {len(records)} match record(s) ({total_highlights} highlight(s), "
        f"{zero_highlight_matches} zero-highlight match(es), median={median_highlights}/match, "
        f"include_takedowns={args.include_takedowns}) to {out_path}"
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
