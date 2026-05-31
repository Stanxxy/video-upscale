"""Final analysis drain and artifact write for upscale stage."""
from __future__ import annotations

import asyncio
import json
import logging
import os

from pipeline import deduplicate_clips
from service.worker.callbacks.analysis_checkpoint import _flush_analysis_checkpoint

logger = logging.getLogger("service.worker")


def shutdown_upscale_analysis_consumer(
    *,
    analyze_fn,
    sliding_buffer: list,
    dispatch_window,
    analysis_consumer_thread,
    analysis_queue,
    analysis_consumer_stop,
    analysis_consumer_error: list[BaseException],
) -> None:
    """Drain the analyzer queue and join the consumer thread (idempotent)."""
    if analyze_fn is None or analysis_consumer_thread is None:
        return
    if not analysis_consumer_thread.is_alive():
        return
    if sliding_buffer:
        dispatch_window(list(sliding_buffer))
        sliding_buffer.clear()
    analysis_queue.put(analysis_consumer_stop)
    analysis_consumer_thread.join(timeout=300.0)
    if analysis_consumer_thread.is_alive():
        logger.warning("Analysis consumer thread did not exit within 300s drain")
    if analysis_consumer_error:
        logger.error(
            "Analysis consumer reported error during run: %s",
            analysis_consumer_error[0],
        )


def finalize_upscale_analysis(
    *,
    analyze_fn,
    fps: float,
    sliding_buffer: list,
    dispatch_window,
    analysis_consumer_thread,
    analysis_queue,
    analysis_consumer_stop,
    analysis_consumer_error: list[BaseException],
    analysis_results: list[dict],
    resume_start_frame: int,
    job_id: str,
    jobs_store,
    s3_for_writes,
    output_bucket: str,
    output_dir: str,
    tracking_s3_key: str,
    current_context: str,
    total_tracking_frames: int,
    loop,
    request,
) -> tuple[dict | None, float]:
    if analyze_fn is None:
        logger.info("Upscale pass complete (no Gemini key -- skipping analysis)")
        return None, fps

    shutdown_upscale_analysis_consumer(
        analyze_fn=analyze_fn,
        sliding_buffer=sliding_buffer,
        dispatch_window=dispatch_window,
        analysis_consumer_thread=analysis_consumer_thread,
        analysis_queue=analysis_queue,
        analysis_consumer_stop=analysis_consumer_stop,
        analysis_consumer_error=analysis_consumer_error,
    )

    logger.info("Finalize: %d windows analysed (consumer drained)", len(analysis_results))

    if analysis_results:
        last_frame_final = (analysis_results[-1].get("frames") or [resume_start_frame])[-1]
        try:
            asyncio.run_coroutine_threadsafe(
                _flush_analysis_checkpoint(
                    job_id=job_id,
                    jobs_store=jobs_store,
                    s3=s3_for_writes,
                    output_bucket=output_bucket,
                    output_dir=output_dir,
                    tracking_s3_key=tracking_s3_key,
                    analysis_results=analysis_results,
                    current_context=current_context,
                    next_frame_idx=last_frame_final + 1,
                    progress_percent=80.0,
                    total_tracking_frames=total_tracking_frames,
                    stage_progress_fraction=1.0,
                ),
                loop,
            ).result(timeout=30)
        except Exception as exc:
            logger.warning("Final analysis flush failed: %s", exc)

    ok_windows = sum(1 for row in analysis_results if "analysis" in row and "clips" in row["analysis"])
    err_windows = sum(1 for row in analysis_results if "raw_error" in row)
    api_err_windows = sum(
        1 for row in analysis_results
        if "analysis" in row and "error" in row.get("analysis", {})
    )
    logger.info(
        "Finalize summary: %d total windows, %d with clips, %d with raw_error, %d with Gemini API error",
        len(analysis_results), ok_windows, err_windows, api_err_windows,
    )

    mode = "Multi-Agent" if request.analyzer_mode == "multi" else "Single-Agent"
    result = {
        "match_summary": f"Analysis generated via {mode}",
        "clips": deduplicate_clips(analysis_results),
        "fps": fps,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "analysis_final.json"), "w") as out_file:
        json.dump(result, out_file, indent=2)
    return result, fps
