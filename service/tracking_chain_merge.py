"""
Merge durable tracking JSON across replacement-job chains before upscale/analysis.

Walks ``job_lifecycle.parent_job_id`` to the root, loads each ancestor's best
available tracking artifact from Keyspaces + S3 (prefer post-upload
``*_tracked.json``, fallback ``partial_tracking.json``), then merges with the
current job's local ``tracking.json`` using **last writer wins** per
``frame_idx`` (later chain positions override earlier).

Only the leaf job's final merged JSON is written locally and uploaded as
``{base}_tracked.json`` — no extra per-ancestor materialized keys.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from botocore.exceptions import ClientError

from service.analysis_keyspaces_enums import PipelineStage
from service.jobs_store import JobsStore
from service.s3 import S3Client

logger = logging.getLogger(__name__)


def merge_tracking_frames_last_writer(
    frame_segments: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Combine frame lists in order; duplicate ``frame_idx`` keeps the **last** segment's row."""
    by_idx: dict[int, dict[str, Any]] = {}
    for segment in frame_segments:
        for entry in segment:
            fi = entry.get("frame_idx")
            if fi is None:
                continue
            by_idx[int(fi)] = entry
    return [by_idx[k] for k in sorted(by_idx)]


def _checkpoint_ts(cp: dict[str, Any]) -> float:
    ts = cp.get("updated_at")
    if isinstance(ts, datetime):
        return ts.timestamp()
    return 0.0


def resolve_best_tracking_keys_from_checkpoints(
    checkpoints: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    """Return ``(full_tracked_s3_key, partial_tracking_s3_key)`` from checkpoint history.

    Prefers the **newest** artifact-bearing row with ``tracking_s3_key``, then
    newest ``partial_tracking_s3_key``, scanning newest-first so terminal
    ``replaced_by_new_job`` rows (which omit artifact keys) do not erase older
    rows that still carry ``tracking_s3_key``.

    For each key, reads ``checkpoint_data.artifacts.*`` first, then the same
    field at checkpoint root (legacy / tests) — aligned with
    :func:`service.checkpoints.select_correction_checkpoint`.
    """
    sorted_cps = sorted(checkpoints, key=_checkpoint_ts, reverse=True)
    full_k: str | None = None
    partial_k: str | None = None
    for cp in sorted_cps:
        sn = cp.get("stage_name") or ""
        if sn not in (
            PipelineStage.TRACK.value,
            PipelineStage.UPLOAD.value,
            PipelineStage.UPSCALE_ANALYZE.value,
        ):
            continue
        data = cp.get("checkpoint_data") or {}
        arts = data.get("artifacts") or {}
        full_candidate = arts.get("tracking_s3_key") or data.get("tracking_s3_key")
        if full_candidate:
            full_k = full_candidate
            break
    for cp in sorted_cps:
        sn = cp.get("stage_name") or ""
        if sn != PipelineStage.TRACK.value:
            continue
        data = cp.get("checkpoint_data") or {}
        arts = data.get("artifacts") or {}
        partial_candidate = arts.get("partial_tracking_s3_key") or data.get(
            "partial_tracking_s3_key"
        )
        if partial_candidate:
            partial_k = partial_candidate
            break
    return full_k, partial_k


def _is_s3_not_found(exc: BaseException) -> bool:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in ("404", "NoSuchKey", "NotFound")
    return False


def download_tracking_json_best_effort(
    s3: S3Client,
    bucket: str,
    *,
    full_key: str | None,
    partial_key: str | None,
    context: str,
) -> dict[str, Any] | None:
    """Try ``full_key`` then ``partial_key``; log and return ``None`` if both missing."""
    for label, key in (("full", full_key), ("partial", partial_key)):
        if not key:
            continue
        try:
            return s3.download_json(bucket, key)
        except ClientError as e:
            if _is_s3_not_found(e):
                logger.warning(
                    "%s: S3 miss (%s) s3://%s/%s — trying fallback",
                    context,
                    label,
                    bucket,
                    key,
                )
                continue
            raise
        except Exception as e:
            logger.warning("%s: failed to download %s: %s", context, key, e)
            continue
    return None


async def walk_job_chain_leaf_to_root(
    jobs_store: JobsStore,
    leaf_job_id: str,
) -> list[str]:
    """Return ``[leaf, parent, …, root]`` following ``parent_job_id``."""
    chain: list[str] = []
    jid = leaf_job_id
    seen: set[str] = set()
    while jid and jid not in seen:
        seen.add(jid)
        chain.append(jid)
        lc = await jobs_store.get_lifecycle(jid)
        if not lc:
            break
        parent = (lc.get("parent_job_id") or "").strip()
        if not parent:
            break
        jid = parent
    return chain


async def consolidate_tracking_json_with_job_chain(
    *,
    jobs_store: JobsStore,
    s3: S3Client,
    bucket: str,
    leaf_job_id: str,
    local_tracking_json_path: str,
    clip_start_frame: int,
) -> dict[str, Any]:
    """Merge ancestor artifacts into ``local_tracking_json_path``; return stats dict.

    Writes the merged JSON back to ``local_tracking_json_path``.
    """
    with open(local_tracking_json_path, encoding="utf-8") as f:
        local_doc = json.load(f)

    chain = await walk_job_chain_leaf_to_root(jobs_store, leaf_job_id)
    ancestors_root_first = list(reversed(chain[1:]))

    local_frames = local_doc.get("frames") or []
    if not ancestors_root_first:
        logger.info(
            "Job %s: chain merge skipped — single-job chain (no parents)",
            leaf_job_id,
        )
        return {
            "chain_length": 1,
            "ancestors_loaded": 0,
            "ancestors_missing": 0,
            "frame_count_before": len(local_frames),
            "frame_count_after": len(local_frames),
        }

    loop = asyncio.get_event_loop()
    segments: list[list[dict[str, Any]]] = []
    loaded = 0
    missing = 0

    for ancestor_id in ancestors_root_first:
        cps = await jobs_store.get_all_checkpoints(ancestor_id)
        fk, pk = resolve_best_tracking_keys_from_checkpoints(cps)

        def _download(
            *,
            _fk: str | None = fk,
            _pk: str | None = pk,
            _aid: str = ancestor_id,
        ) -> dict[str, Any] | None:
            return download_tracking_json_best_effort(
                s3,
                bucket,
                full_key=_fk,
                partial_key=_pk,
                context=f"ancestor={_aid}",
            )

        blob = await loop.run_in_executor(None, _download)
        if blob is None:
            missing += 1
            logger.warning(
                "Job %s: chain merge — no usable tracking JSON for ancestor %s",
                leaf_job_id,
                ancestor_id,
            )
            continue
        fr = blob.get("frames") or []
        if fr:
            segments.append(fr)
            loaded += 1

    segments.append(local_frames)
    merged_frames = merge_tracking_frames_last_writer(segments)

    min_idx: int | None = None
    for e in merged_frames:
        fi = e.get("frame_idx")
        if fi is not None:
            min_idx = int(fi) if min_idx is None else min(min_idx, int(fi))

    out_doc = {
        **local_doc,
        "frames": merged_frames,
        "start_frame": min_idx if min_idx is not None else local_doc.get(
            "start_frame", clip_start_frame,
        ),
    }

    with open(local_tracking_json_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f)

    logger.info(
        "Job %s: chain merge chain_jobs=%d ancestors_loaded=%d ancestors_missing=%d "
        "frames %d -> %d",
        leaf_job_id,
        len(chain),
        loaded,
        missing,
        len(local_frames),
        len(merged_frames),
    )

    return {
        "chain_length": len(chain),
        "ancestors_loaded": loaded,
        "ancestors_missing": missing,
        "frame_count_before": len(local_frames),
        "frame_count_after": len(merged_frames),
    }


def preflight_resume_tracking_overrides(
    overrides: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    s3: S3Client,
    bucket: str,
) -> dict[str, Any]:
    """If ``resume_tracking_s3_key`` is missing in S3, fall back to another durable key.

    Uses :func:`resolve_best_tracking_keys_from_checkpoints` and HEAD checks.
    When switching to a **full** ``*_tracked.json`` key, sets
    ``resume_from_frame`` to :data:`~service.checkpoints.END_OF_TRACKING_SENTINEL`.

    If HEAD fails (network / emulator down), leaves overrides unchanged so the
    worker download step stays authoritative.
    """
    from service.checkpoints import END_OF_TRACKING_SENTINEL

    def _head_ok(k: str) -> bool | None:
        """True = exists, False = missing, None = inconclusive."""
        try:
            return s3.object_exists(bucket, k)
        except Exception as e:
            logger.warning(
                "resume preflight: HEAD inconclusive for s3://%s/%s: %s",
                bucket,
                k,
                e,
            )
            return None

    out = dict(overrides)
    key = out.get("resume_tracking_s3_key")
    if not key or not str(bucket).strip():
        return out
    primary = _head_ok(key)
    if primary is True:
        return out
    if primary is None:
        return out

    logger.warning(
        "resume_tracking_s3_key not found s3://%s/%s — attempting fallback keys",
        bucket,
        key,
    )
    fk, pk = resolve_best_tracking_keys_from_checkpoints(checkpoints)
    candidates: list[tuple[str, str, bool]] = []
    if fk:
        candidates.append((fk, "full", True))
    if pk:
        candidates.append((pk, "partial", False))
    for cand, label, is_full in candidates:
        if cand == key:
            continue
        st = _head_ok(cand)
        if st is None:
            logger.warning(
                "resume preflight: abort fallback chain (inconclusive HEAD on %s)",
                cand,
            )
            return out
        if not st:
            continue
        out["resume_tracking_s3_key"] = cand
        if is_full:
            out["resume_from_frame"] = END_OF_TRACKING_SENTINEL
        logger.info(
            "resume preflight: switched to %s key s3://%s/%s",
            label,
            bucket,
            cand,
        )
        return out

    logger.error(
        "resume preflight: no fallback tracking artifact in bucket=%s — "
        "replacement job may fail at download",
        bucket,
    )
    return out
