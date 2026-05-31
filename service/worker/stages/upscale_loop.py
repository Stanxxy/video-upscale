"""Sequential upscale + analysis (loop + finalize half)."""
from __future__ import annotations
import asyncio
import json
import logging
import os
import threading
import time
import cv2
from service.checkpoints import should_flush_analysis
from service.models import ProcessingMode
from service.worker.callbacks.analysis_checkpoint import _flush_analysis_checkpoint
from service.worker.stages.upscale_batch import flush_upscale_batch
from service.worker.stages.upscale_finalize_stage import (
    finalize_upscale_analysis,
    shutdown_upscale_analysis_consumer,
)
logger = logging.getLogger("service.worker")
def _upscale_analysis_loop_finalize(g: dict):
    def _impl():
        tracking_data = g["tracking_data"]
        config = g["config"]
        request = g["request"]
        job_id = g["job_id"]
        jobs_store = g["jobs_store"]
        loop = g["loop"]
        tracking_s3_key = g["tracking_s3_key"]
        progress_cb = g.get("progress_cb")
        output_dir = g["output_dir"]
        cap = g["cap"]
        fps = g["fps"]
        jpeg_pool = g["jpeg_pool"]
        _submit_jpeg_write = g["_submit_jpeg_write"]
        _drain_jpeg_writes = g["_drain_jpeg_writes"]
        restorer = g["restorer"]
        analyze_fn = g["analyze_fn"]
        sliding_buffer = g["sliding_buffer"]
        analysis_results = g["analysis_results"]
        current_context = g["current_context"]
        WINDOW_SIZE = g["WINDOW_SIZE"]
        STRIDE = g["STRIDE"]
        resume_start_frame = g["resume_start_frame"]
        method_prefix = g["method_prefix"]
        output_bucket = g["output_bucket"]
        s3_for_writes = g["s3_for_writes"]
        total_tracking_frames = g["total_tracking_frames"]
        player_ref_images = g.get("player_ref_images")
        _analyzer = g.get("_analyzer")
        from utils import get_union_box, get_padded_square_box
        def _analyze_window(window):
            nonlocal current_context
            batch_frames = [x[1] for x in window]
            batch_indices = [x[0] for x in window]
            chunk_idx = len(analysis_results) + 1
            ctx = current_context if _use_context_chain else None
            logger.info(
                "Analyzing window %d (frames %d-%d, %d images, ctx_chain=%s)",
                chunk_idx, batch_indices[0], batch_indices[-1], len(batch_frames),
                _use_context_chain,
            )
            try:
                result_str = analyze_fn(
                    batch_frames, batch_indices, ctx,
                )
                logger.info(
                    "Window %d raw response length: %d chars",
                    chunk_idx, len(result_str) if result_str else 0,
                )
                logger.debug("Window %d raw response: %.500s", chunk_idx, result_str)
                result_data = json.loads(result_str)
                if "error" in result_data:
                    logger.error(
                        "Window %d: Gemini returned error: %s",
                        chunk_idx, result_data["error"],
                    )
                elif "clips" not in result_data:
                    logger.warning(
                        "Window %d: Gemini response missing 'clips' key. Keys: %s",
                        chunk_idx, list(result_data.keys()),
                    )
                else:
                    logger.info(
                        "Window %d: Gemini returned %d clips",
                        chunk_idx, len(result_data["clips"]),
                    )
                if _use_context_chain and "current_context_summary" in result_data:
                    current_context = result_data["current_context_summary"]
                analysis_results.append({
                    "window": chunk_idx,
                    "frames": batch_indices,
                    "analysis": result_data,
                })
            except Exception as e:
                logger.error(
                    "Analysis window %d failed: %s", chunk_idx, e, exc_info=True,
                )
                analysis_results.append({"window": chunk_idx, "raw_error": str(e)})
        _is_fast_upscale = request.processing_mode == ProcessingMode.FAST
        _eff_gemini_max_inflight = 24 if _is_fast_upscale else config.gemini_max_inflight
        _use_context_chain = not _is_fast_upscale  # fast mode: no context chain
        _fast_anchor_context: str | None = None
        if _is_fast_upscale:
            logger.info(
                "Job %s: fast mode Gemini fanout=%d (anchor-context: first window sequential, rest parallel)",
                job_id, _eff_gemini_max_inflight,
            )
        import queue as _stdlib_queue
        analysis_queue: _stdlib_queue.Queue = _stdlib_queue.Queue(
            maxsize=_eff_gemini_max_inflight + 1
        )
        _analysis_consumer_stop = object()  # sentinel
        _analysis_consumer_error: list[BaseException] = []
        # already has it too. We store a reference to the analyzer for async dispatch.
        # Guard: only enable async if the method is a real coroutine function (not a
        # MagicMock auto-attribute or other non-async callable).
        import inspect as _inspect
        _async_analyzer = locals().get("_analyzer") if config.gemini_api_key else None
        _async_method = getattr(_async_analyzer, "analyze_sequence_async", None) if _async_analyzer else None
        _use_async = (
            _async_analyzer is not None
            and _async_method is not None
            and _inspect.iscoroutinefunction(_async_method)
            and request.analyzer_mode != "multi"  # multi uses its own async internally
        )

        def _trigger_periodic_flush():
            if not should_flush_analysis(len(analysis_results)):
                return
            last_frame = (
                (analysis_results[-1].get("frames") or [resume_start_frame])[-1]
            )
            progress_pct = 55.0 + (processed / max(total, 1)) * 25.0
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
                        next_frame_idx=last_frame + 1,
                        progress_percent=progress_pct,
                        total_tracking_frames=total_tracking_frames,
                        stage_progress_fraction=(processed / max(total, 1)),
                    ),
                    loop,
                ).result(timeout=30)
            except Exception as e:
                logger.warning("Periodic analysis flush failed: %s", e)

        async def _async_analyze_window(window_payload):
            """Async version of _analyze_window using analyze_sequence_async."""
            nonlocal current_context, _fast_anchor_context
            batch_frames = [x[1] for x in window_payload]
            batch_indices = [x[0] for x in window_payload]
            chunk_idx = len(analysis_results) + 1
            # Fast mode: use anchor context from first window to preserve role labels.
            # First window gets ctx=None (establishes anchor); subsequent windows get anchor.
            ctx = current_context if _use_context_chain else _fast_anchor_context
            logger.info(
                "Analyzing window %d async (frames %d-%d, %d images, ctx_chain=%s)",
                chunk_idx, batch_indices[0], batch_indices[-1], len(batch_frames),
                _use_context_chain,
            )
            try:
                result_str = await _async_analyzer.analyze_sequence_async(
                    batch_frames, batch_indices, ctx,
                    player_references=player_ref_images,
                )
                logger.info(
                    "Window %d async raw response length: %d chars",
                    chunk_idx, len(result_str) if result_str else 0,
                )
                logger.debug("Window %d async raw response: %.500s", chunk_idx, result_str)
                result_data = json.loads(result_str)

                if "error" in result_data:
                    logger.error("Window %d: Gemini returned error: %s", chunk_idx, result_data["error"])
                elif "clips" not in result_data:
                    logger.warning(
                        "Window %d: Gemini response missing 'clips' key. Keys: %s",
                        chunk_idx, list(result_data.keys()),
                    )
                else:
                    logger.info("Window %d: Gemini returned %d clips", chunk_idx, len(result_data["clips"]))

                if _use_context_chain and "current_context_summary" in result_data:
                    current_context = result_data["current_context_summary"]
                elif not _use_context_chain and _fast_anchor_context is None and "current_context_summary" in result_data:
                    _fast_anchor_context = result_data["current_context_summary"]
                analysis_results.append({
                    "window": chunk_idx,
                    "frames": batch_indices,
                    "analysis": result_data,
                })
            except Exception as e:
                logger.error("Async analysis window %d failed: %s", chunk_idx, e, exc_info=True)
                analysis_results.append({"window": chunk_idx, "raw_error": str(e)})

        def _analysis_consumer_loop():
            """Consumer thread: owns its own asyncio event loop for async Gemini calls.

            M4 F5 (fast mode): when _eff_gemini_max_inflight > 1 and _use_async is True,
            drains up to _eff_gemini_max_inflight windows from the queue and runs them
            concurrently via asyncio.gather. This multiplies Gemini throughput by fanout
            at the cost of no context chain (already disabled in fast mode).
            """
            _consumer_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_consumer_loop)
            try:
                while True:
                    # Drain up to _eff_gemini_max_inflight items for batch dispatch
                    items = []
                    while len(items) < _eff_gemini_max_inflight:
                        if items:
                            # Non-blocking get for additional items beyond the first
                            try:
                                item = analysis_queue.get_nowait()
                            except Exception:
                                break
                        else:
                            # Blocking get for the first item
                            item = analysis_queue.get()
                        if item is _analysis_consumer_stop:
                            analysis_queue.task_done()
                            # Drain and process any already collected items before stopping
                            if items:
                                try:
                                    if _use_async and len(items) > 1:
                                        if _is_fast_upscale and _fast_anchor_context is None:
                                            _consumer_loop.run_until_complete(_async_analyze_window(items[0]))
                                            if len(items) > 1:
                                                _consumer_loop.run_until_complete(
                                                    asyncio.gather(*[_async_analyze_window(w) for w in items[1:]])
                                                )
                                        else:
                                            _consumer_loop.run_until_complete(
                                                asyncio.gather(*[_async_analyze_window(w) for w in items])
                                            )
                                    elif _use_async:
                                        _consumer_loop.run_until_complete(_async_analyze_window(items[0]))
                                    else:
                                        for w in items:
                                            _analyze_window(w)
                                except Exception as e:
                                    logger.error("Consumer stop-drain error: %s", e, exc_info=True)
                                finally:
                                    for _ in items:
                                        analysis_queue.task_done()
                            return
                        items.append(item)

                    if not items:
                        continue
                    try:
                        if _use_async and len(items) > 1:
                            if _is_fast_upscale and _fast_anchor_context is None:
                                # Fast mode phase 1: first window alone establishes role anchor
                                _consumer_loop.run_until_complete(_async_analyze_window(items[0]))
                                if len(items) > 1:
                                    _consumer_loop.run_until_complete(
                                        asyncio.gather(*[_async_analyze_window(w) for w in items[1:]])
                                    )
                            else:
                                # Standard mode or anchor already set: full concurrent dispatch
                                _consumer_loop.run_until_complete(
                                    asyncio.gather(*[_async_analyze_window(w) for w in items])
                                )
                        elif _use_async:
                            _consumer_loop.run_until_complete(_async_analyze_window(items[0]))
                        else:
                            for w in items:
                                _analyze_window(w)
                        try:
                            os.makedirs(output_dir, exist_ok=True)
                            with open(
                                os.path.join(output_dir, "analysis_raw.json"), "w"
                            ) as f:
                                json.dump(analysis_results, f, indent=2)
                        except Exception as e:
                            logger.warning("analysis_raw.json write failed: %s", e)
                        _trigger_periodic_flush()
                    finally:
                        for _ in items:
                            analysis_queue.task_done()
            except BaseException as e:
                logger.error("Analysis consumer crashed: %s", e, exc_info=True)
                _analysis_consumer_error.append(e)
            finally:
                _consumer_loop.close()

        _analysis_consumer_thread = None
        if analyze_fn is not None:
            _analysis_consumer_thread = threading.Thread(
                target=_analysis_consumer_loop,
                name="upscale-analyzer-consumer",
                daemon=True,
            )
            _analysis_consumer_thread.start()

        def _dispatch_window(window):
            """Hand a window off to the consumer thread. Blocks if queue is full (backpressure)."""
            analysis_queue.put(window)

        # -- Main upscale loop ---------------------------------------------------
        frames = tracking_data.get("frames", [])
        total = len(frames)
        processed = 0
        _last_upscale_hb = time.monotonic()
        _hb_interval = float(config.upscale_heartbeat_interval_sec)
        # M4 F2: fast mode applies effective_sampling_rate=max(request.sampling_rate, 2)
        # so we skip every other upscale frame, halving the upscale workload.
        _eff_sampling_rate = (
            max(request.sampling_rate, 2)
            if request.processing_mode == ProcessingMode.FAST
            else request.sampling_rate
        )

        # Sequential VideoCapture cursor. tracking_data['frames'] is in ascending
        # frame_idx order (asserted below). Calling cap.set(POS_FRAMES) per frame
        # forces an O(GOP) random seek on long-GOP codecs (VP9, H.264). Instead we
        # advance cap.read() sequentially and only fall back to cap.set() when the
        # requested frame is not the next contiguous frame (e.g., sampling_rate>1
        # gaps, resume from a checkpoint past frame 0, or out-of-order entries).
        _cap_cursor = -1  # idx of last frame consumed by cap.read(); -1 means nothing read yet

        def _read_frame_at(target_idx: int):
            """Return (ret, frame_bgr) for target_idx using sequential reads when possible."""
            nonlocal _cap_cursor
            next_expected = _cap_cursor + 1
            if target_idx == next_expected:
                ret, frame_bgr = cap.read()
                if ret:
                    _cap_cursor = target_idx
                return ret, frame_bgr
            if target_idx > next_expected:
                # Skip the gap with cheap sequential reads (decoding-cheaper than a
                # set() seek for small gaps; for large gaps cap.set is faster, so
                # we threshold).
                gap = target_idx - next_expected
                if gap <= 8:
                    # Burn through the gap with cap.read() (no decoding savings
                    # available without a seek anyway, but stays warm).
                    for _ in range(gap):
                        ok, _ = cap.read()
                        if not ok:
                            _cap_cursor = target_idx  # advance cursor anyway to avoid infinite loop
                            return False, None
                        _cap_cursor += 1
                    ret, frame_bgr = cap.read()
                    if ret:
                        _cap_cursor = target_idx
                    return ret, frame_bgr
            # Fall back to a true seek (backwards jump or large forward gap).
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame_bgr = cap.read()
            if ret:
                _cap_cursor = target_idx
            return ret, frame_bgr

        # Sanity assert: frames are ascending. Log and fall back if not, so the
        # sequential path is safe.
        _ascending = all(
            frames[i]["frame_idx"] <= frames[i + 1]["frame_idx"]
            for i in range(len(frames) - 1)
        )
        if not _ascending:
            logger.warning(
                "tracking_data frames are not in ascending frame_idx order; sequential "
                "read optimization will fall back to seeks on every out-of-order entry"
            )

        # S1: batch upscale crops. ESRGAN forward passes are GPU-bound and small;
        # batching N crops into a single forward pass amortizes Python/launch
        # overhead 8x. Diffusion path retains per-call semantics (it takes
        # 'strength', not a list).
        BATCH_SIZE = 8 if request.method != "diffusion" else 1
        pending_crops: list[tuple[int, "np.ndarray"]] = []  # (frame_idx, crop_bgr)

        def _flush_batch():
            flush_upscale_batch(
                pending_crops=pending_crops,
                request=request,
                restorer=restorer,
                config=config,
                job_id=job_id,
                output_dir=output_dir,
                method_prefix=method_prefix,
                submit_jpeg_write=_submit_jpeg_write,
                analyze_fn=analyze_fn,
                sliding_buffer=sliding_buffer,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
                dispatch_window=_dispatch_window,
            )

        try:
            for entry in frames:
                frame_idx = entry["frame_idx"]
                _now = time.monotonic()
                if _now - _last_upscale_hb >= _hb_interval:
                    logger.info(
                        "Job %s upscale heartbeat: frame_idx=%s processed=%d/%d buffer=%d batch=%d",
                        job_id, frame_idx, processed, total,
                        len(sliding_buffer), len(pending_crops),
                    )
                    _last_upscale_hb = _now

                athletes = entry.get("athletes", [])

                # Skip frames before resume point
                if frame_idx < resume_start_frame:
                    processed += 1
                    continue

                if frame_idx % _eff_sampling_rate != 0:
                    processed += 1
                    continue
                if not athletes:
                    processed += 1
                    continue

                ret, frame_bgr = _read_frame_at(frame_idx)
                if not ret:
                    logger.warning("Could not read frame %d", frame_idx)
                    processed += 1
                    continue

                h, w = frame_bgr.shape[:2]
                boxes = [a["box"] for a in athletes]
                union_box = get_union_box(boxes)
                square_box = get_padded_square_box(
                    union_box, padding=0.2, img_shape=(h, w),
                )
                x1, y1, x2, y2 = square_box
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    processed += 1
                    continue

                pending_crops.append((frame_idx, crop))
                processed += 1

                if len(pending_crops) >= BATCH_SIZE:
                    _flush_batch()

                if progress_cb and total > 0:
                    progress_cb(processed / total)

            # Drain any straggler crops at end of stream.
            _flush_batch()
        finally:
            # Stop the analyzer consumer first so a JPEG/cap cleanup failure cannot
            # leave a daemon thread writing analysis_raw.json after work_dir cleanup.
            shutdown_upscale_analysis_consumer(
                analyze_fn=analyze_fn,
                sliding_buffer=sliding_buffer,
                dispatch_window=_dispatch_window,
                analysis_consumer_thread=_analysis_consumer_thread,
                analysis_queue=analysis_queue,
                analysis_consumer_stop=_analysis_consumer_stop,
                analysis_consumer_error=_analysis_consumer_error,
            )
            cap.release()
            # S2: drain pending JPEG writes before stage exit so resume sees a
            # consistent on-disk artifact set.
            try:
                _drain_jpeg_writes(timeout=120.0)
            except Exception as e:
                logger.warning("JPEG drain failed: %s", e)
            jpeg_pool.shutdown()

        return finalize_upscale_analysis(
            analyze_fn=analyze_fn,
            fps=fps,
            sliding_buffer=sliding_buffer,
            dispatch_window=_dispatch_window,
            analysis_consumer_thread=_analysis_consumer_thread,
            analysis_queue=analysis_queue,
            analysis_consumer_stop=_analysis_consumer_stop,
            analysis_consumer_error=_analysis_consumer_error,
            analysis_results=analysis_results,
            resume_start_frame=resume_start_frame,
            job_id=job_id,
            jobs_store=jobs_store,
            s3_for_writes=s3_for_writes,
            output_bucket=output_bucket,
            output_dir=output_dir,
            tracking_s3_key=tracking_s3_key,
            current_context=current_context,
            total_tracking_frames=total_tracking_frames,
            loop=loop,
            request=request,
        )

    return _impl()
