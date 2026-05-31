"""Batch crop enhancement and analyzer dispatch for upscale stage."""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

import cv2
from PIL import Image

logger = logging.getLogger("service.worker")


def flush_upscale_batch(
    *,
    pending_crops: list[tuple[int, Any]],
    request,
    restorer,
    config,
    job_id: str,
    output_dir: str,
    method_prefix: str,
    submit_jpeg_write: Callable[[str, Any], None],
    analyze_fn,
    sliding_buffer: list[tuple[int, Image.Image]],
    window_size: int,
    stride: int,
    dispatch_window: Callable[[list[tuple[int, Image.Image]]], None],
) -> None:
    if not pending_crops:
        return
    try:
        started = time.perf_counter()
        if request.method == "diffusion":
            fidx, crop = pending_crops[0]
            h_crop, w_crop = crop.shape[:2]
            if max(h_crop, w_crop) > 768:
                scale = 768 / max(h_crop, w_crop)
                crop = cv2.resize(
                    crop,
                    (int(w_crop * scale), int(h_crop * scale)),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            enhanced_list = [restorer.enhance(crop, strength=0.5)]
            batch_indices = [fidx]
        else:
            batch_indices = [fi for fi, _ in pending_crops]
            max_input_edge = 288
            batch_crops = []
            for crop in (c for _, c in pending_crops):
                h_c, w_c = crop.shape[:2]
                long_edge = max(h_c, w_c)
                if long_edge > max_input_edge:
                    scale = max_input_edge / long_edge
                    crop = cv2.resize(
                        crop,
                        (max(1, int(w_c * scale)), max(1, int(h_c * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                batch_crops.append(crop)
            enhance_batch_fn = getattr(restorer, "enhance_batch", None)
            use_batched = callable(enhance_batch_fn) and hasattr(type(restorer), "enhance_batch")
            if use_batched:
                enhanced_list = enhance_batch_fn(batch_crops, target_size=config.upscale_target_size)
            else:
                enhanced_list = [
                    restorer.enhance(crop, target_size=config.upscale_target_size)
                    for crop in batch_crops
                ]
        elapsed = time.perf_counter() - started
        logger.debug(
            "Job %s upscale: batch of %d done in %.2fs (%.2f ms/frame)",
            job_id, len(pending_crops), elapsed,
            elapsed * 1000.0 / max(len(pending_crops), 1),
        )
    except Exception as exc:
        logger.warning(
            "Batch upscale failed for frames %s: %s",
            [fi for fi, _ in pending_crops], exc,
        )
        pending_crops.clear()
        return

    for fidx, enhanced in zip(batch_indices, enhanced_list):
        out_path = os.path.join(output_dir, f"{method_prefix}frame_{fidx:06d}.jpg")
        submit_jpeg_write(out_path, enhanced)
        if analyze_fn is not None:
            img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            sliding_buffer.append((fidx, Image.fromarray(img_rgb)))
            if len(sliding_buffer) >= window_size:
                dispatch_window(list(sliding_buffer[:window_size]))
                del sliding_buffer[:stride]

    if hasattr(restorer, "flush_cache"):
        restorer.flush_cache()
    pending_crops.clear()
