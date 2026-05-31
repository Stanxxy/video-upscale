"""Parallel upscale across memory-bounded segments."""
import asyncio
import json
import logging
import os

from service.analysis_keyspaces_enums import PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.models import TrackRequest

from service.worker.progress import (
    LIFECYCLE_HEARTBEAT_INTERVAL,
    _pct_at_least,
)
from service.worker.stages.upscale import _run_upscale_analysis

logger = logging.getLogger("service.worker")

async def _run_parallel_upscale(
    *,
    job_id: str,
    video_path: str,
    merged_tracking_json_path: str,
    clip_start_frame: int,
    clip_end_resolved: int,
    clip_total_frames: int,
    config: ServiceConfig,
    request: TrackRequest,
    work_dir: str,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
    loop: asyncio.AbstractEventLoop,
    tracking_s3_key: str,
    progress_floor: float,
) -> tuple[dict | None, float]:
    """
    Run the UPSCALE_ANALYZE stage in parallel across N memory-bounded
    segments. Each segment processes only the tracking frames whose
    ``frame_idx`` falls inside its absolute frame range.

    Returns ``(merged_analysis_result, fps)`` to match the sequential
    ``_run_upscale_analysis`` shape so the caller in ``run_job`` can hand
    the result straight to the annotate/upload/publish stages.

    Progress: each segment's ``progress_cb`` reports a per-segment
    ``processed/total`` ratio to a shared aggregator. The aggregator runs
    on the worker event loop, computes a clip-weighted overall fraction,
    and writes ``PipelineStage.UPSCALE_ANALYZE`` lifecycle progress in the
    55%-80% band at the ``LIFECYCLE_HEARTBEAT_INTERVAL`` cadence (1Hz).
    The aggregator never writes any ``job_state`` transition or checkpoint
    artifact — those remain the sole responsibility of
    ``_run_upscale_analysis`` (it writes the V1 ``upscale_analyze``
    checkpoint rows per the addendum) and the caller (it writes
    track-complete + upload rows around this helper).
    """
    from service.segment_runner import (
        SegmentBounds,
        compute_segment_ranges,
        merge_analysis_results,
    )

    # Dynamic segment count: bounded by segment_max_frames, parallelism by standard_segments.
    all_ranges = compute_segment_ranges(clip_total_frames, config.segment_max_frames)
    n_total = len(all_ranges)
    k_parallel = min(n_total, config.standard_segments)
    segments = [
        SegmentBounds(
            segment_id=i,
            start_frame=clip_start_frame + s,
            end_frame=clip_start_frame + e,
        )
        for i, (s, e) in enumerate(all_ranges)
    ]

    logger.info(
        "Job %s: parallel-upscale n_segments=%d k_parallel=%d clip=[%d, %d) total_frames=%d",
        job_id, n_total, k_parallel, clip_start_frame, clip_end_resolved, clip_total_frames,
    )

    # Load the merged tracking JSON produced by the sequential tracking stage.
    with open(merged_tracking_json_path) as f:
        merged_tracking_full = json.load(f)
    all_frames = merged_tracking_full.get("frames", [])
    seg_fps = merged_tracking_full.get("fps", 30.0)

    # ------------------------------------------------------------------
    # Per-segment tracking JSON (slice merged frames by absolute frame_idx)
    # ------------------------------------------------------------------
    seg_upscale_work_dirs = [
        os.path.join(work_dir, f"upscale_seg{s.segment_id}") for s in segments
    ]
    seg_tracking_paths: list[str] = []
    seg_frame_counts: list[int] = []
    for i, seg in enumerate(segments):
        os.makedirs(seg_upscale_work_dirs[i], exist_ok=True)
        seg_frames = [
            fr for fr in all_frames
            if seg.start_frame <= int(fr.get("frame_idx", fr.get("frame", -1))) < seg.end_frame
        ]
        seg_frame_counts.append(len(seg_frames))
        seg_track_data = {
            **merged_tracking_full,
            "frames": seg_frames,
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
        }
        seg_track_path = os.path.join(seg_upscale_work_dirs[i], "tracking.json")
        with open(seg_track_path, "w") as f:
            json.dump(seg_track_data, f)
        seg_tracking_paths.append(seg_track_path)
        logger.info(
            "Job %s seg%d: upscale slice frames=%d range=[%d, %d)",
            job_id, i, len(seg_frames), seg.start_frame, seg.end_frame,
        )

    total_seg_frames = max(sum(seg_frame_counts), 1)

    # ------------------------------------------------------------------
    # Per-segment progress aggregator (writes PipelineStage.UPSCALE_ANALYZE)
    # ------------------------------------------------------------------
    # Each segment runs in its own thread executor with its own internal
    # upscale loop. ``_run_upscale_analysis`` already accepts a
    # ``progress_cb(fraction)`` that reports the per-segment processed/total
    # ratio in [0, 1]. We store ``frames_done[seg_idx] = round(fraction * seg_len)``
    # in a shared dict (each thread owns its own key; int assignment is
    # atomic under the GIL so no lock is required) and a background asyncio
    # task aggregates the sum every ``LIFECYCLE_HEARTBEAT_INTERVAL``,
    # computing overall progress weighted by per-segment frame counts.
    frames_done: dict[int, int] = {i: 0 for i in range(n_total)}

    def _make_seg_progress_cb(seg_idx: int):
        seg_len = seg_frame_counts[seg_idx]

        def _seg_progress_cb(fraction: float):
            # ``fraction`` is the per-segment processed/total reported by
            # ``_run_upscale_analysis`` (line ~2384). Convert to absolute
            # frames-done within that segment so the aggregator can sum.
            done = int(round(max(0.0, min(1.0, fraction)) * seg_len))
            frames_done[seg_idx] = done
        return _seg_progress_cb

    aggregator_stop = asyncio.Event()

    async def _aggregate_and_write_lifecycle():
        """Compute clip-weighted upscale progress and write the lifecycle row.

        ``_upscale_analyze`` band is 55%-80%. With per-segment frame counts
        ``seg_frame_counts`` summing to ``total_seg_frames``, the overall
        progress is ``55.0 + (sum(frames_done.values()) / total_seg_frames) * 25.0``
        clamped to [55.0, 80.0]. ``progress_floor`` keeps replacement jobs
        from regressing.
        """
        total_done = sum(frames_done.values())
        frac = min(max(total_done / total_seg_frames, 0.0), 1.0)
        pct = _pct_at_least(55.0 + frac * 25.0, progress_floor)
        # Clip just below the band ceiling so the stage-complete write owns
        # the 80.0 boundary cleanly.
        pct = min(pct, 80.0)
        try:
            await job_store.update_job(job_id, progress_percent=pct)
            await jobs_store.update_progress(
                job_id, PipelineStage.UPSCALE_ANALYZE, pct,
            )
        except Exception as e:
            # Best-effort heartbeat; never raise from the aggregator.
            logger.error(
                "parallel-upscale progress aggregator write failed: %s",
                e,
                exc_info=e,
            )

    async def _aggregator_task():
        try:
            while not aggregator_stop.is_set():
                try:
                    await asyncio.wait_for(
                        aggregator_stop.wait(),
                        timeout=LIFECYCLE_HEARTBEAT_INTERVAL,
                    )
                    break  # stop event set during the wait — exit loop
                except asyncio.TimeoutError:
                    pass
                await _aggregate_and_write_lifecycle()
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # Run each segment's upscale+analyze in a thread executor
    # ------------------------------------------------------------------
    def _upscale_segment(seg_idx: int) -> tuple[dict | None, float]:
        try:
            result, fps_val = _run_upscale_analysis(
                video_path,
                seg_tracking_paths[seg_idx],
                config,
                request,
                seg_upscale_work_dirs[seg_idx],
                job_id=job_id,
                jobs_store=jobs_store,
                loop=loop,
                tracking_s3_key=tracking_s3_key,
                progress_cb=_make_seg_progress_cb(seg_idx),
            )
            logger.info(
                "Job %s seg%d: upscale+analyze done has_analysis=%s",
                job_id, seg_idx, result is not None,
            )
            return result, fps_val
        except Exception as e:
            logger.error(
                "Job %s seg%d: upscale failed: %s",
                job_id, seg_idx, e, exc_info=True,
            )
            raise

    upscale_results: list[tuple[dict | None, float]] = [(None, 0.0)] * n_total
    aggregator = asyncio.create_task(_aggregator_task())
    try:
        # Run all segments in sequential batches of k_parallel to bound
        # peak GPU memory.
        for _batch_start in range(0, n_total, k_parallel):
            _batch_end = min(_batch_start + k_parallel, n_total)
            _batch_futures = [
                loop.run_in_executor(None, _upscale_segment, i)
                for i in range(_batch_start, _batch_end)
            ]
            _batch_res = list(await asyncio.gather(*_batch_futures))
            for _bi, _res in zip(range(_batch_start, _batch_end), _batch_res):
                upscale_results[_bi] = _res
    finally:
        aggregator_stop.set()
        aggregator.cancel()
        try:
            await aggregator
        except (asyncio.CancelledError, Exception):
            pass
        # Final aggregated write so any frames processed after the last
        # 1Hz tick are reflected before the next stage takes over.
        try:
            await _aggregate_and_write_lifecycle()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Merge per-segment analysis results
    # ------------------------------------------------------------------
    seg_analyses = [r[0] for r in upscale_results]
    fps_values = [r[1] for r in upscale_results if r[1] and r[1] > 0]
    final_fps = fps_values[0] if fps_values else seg_fps

    merged_analysis = merge_analysis_results(seg_analyses)
    if merged_analysis:
        merged_analysis["fps"] = final_fps

    logger.info(
        "Job %s: parallel-upscale complete n_segments=%d k_parallel=%d fps=%.2f has_analysis=%s",
        job_id, n_total, k_parallel, final_fps, merged_analysis is not None,
    )

    return merged_analysis, final_fps
