"""
CLI analysis pipeline for offline debugging and isolated testing.

Provides process_video() for end-to-end upscale + Gemini analysis via CLI (main.py).
NOT used by the production service — service/worker.py handles that code path.

The live run_pipeline() for tracking jobs lives in tracking_pipeline/pipeline.py,
imported as ``tracking_pipeline.pipeline`` or via the ``tracking`` re-export package.
"""
import cv2
import json
import os
from tqdm import tqdm
from utils import get_union_box, get_padded_square_box


def deduplicate_clips(all_analysis_results):
    from collections import defaultdict

    all_clips = []
    for chunk in all_analysis_results:
        if "analysis" in chunk and "clips" in chunk["analysis"]:
            all_clips.extend(chunk["analysis"]["clips"])

    if not all_clips:
        return []

    # Group by role (athlete) so overlaps are resolved per-athlete
    by_role = defaultdict(list)
    for clip in all_clips:
        by_role[clip.get("role", "")].append(clip)

    merged_all = []
    for role, clips in by_role.items():
        clips.sort(key=lambda x: x["start_frame"])

        # Phase 1: merge same-action adjacent/overlapping clips (original logic, now per-athlete)
        merged = [clips[0].copy()]
        for nxt in clips[1:]:
            cur = merged[-1]
            overlap_len = max(0, min(cur["end_frame"], nxt["end_frame"])
                               - max(cur["start_frame"], nxt["start_frame"]))
            is_same_cat = cur.get("action", cur.get("category", "")) == nxt.get("action", nxt.get("category", ""))
            is_close = (nxt["start_frame"] - cur["end_frame"]) < 30
            if is_same_cat and (overlap_len > 0 or is_close):
                new_start = min(cur["start_frame"], nxt["start_frame"])
                new_end = max(cur["end_frame"], nxt["end_frame"])
                if nxt.get("confidence", 0) > cur.get("confidence", 0):
                    merged[-1] = nxt.copy()
                merged[-1]["start_frame"] = new_start
                merged[-1]["end_frame"] = new_end
            else:
                merged.append(nxt.copy())

        # Phase 2: resolve remaining overlaps (different-category, same athlete)
        # Keep the higher-confidence clip; trim or discard the lower one.
        resolved = [merged[0]]
        for nxt in merged[1:]:
            cur = resolved[-1]
            overlap_len = max(0, min(cur["end_frame"], nxt["end_frame"])
                               - max(cur["start_frame"], nxt["start_frame"]))
            if overlap_len > 0:
                if nxt.get("confidence", 0) > cur.get("confidence", 0):
                    # cur loses: trim it to end just before nxt starts
                    cur["end_frame"] = nxt["start_frame"] - 1
                    if cur["end_frame"] <= cur["start_frame"]:
                        resolved.pop()
                    resolved.append(nxt)
                else:
                    # nxt loses: trim it to start just after cur ends
                    nxt["start_frame"] = cur["end_frame"] + 1
                    if nxt["start_frame"] < nxt["end_frame"]:
                        resolved.append(nxt)
            else:
                resolved.append(nxt)

        merged_all.extend(resolved)

    merged_all.sort(key=lambda x: x["start_frame"])
    return merged_all


def process_video(
    video_path,
    json_path,
    output_dir,
    model_path,
    method="esrgan",
    sampling_rate=1,
    max_frames=None,
    target_size=1024,
    diffusion_strength=0.5,
    analyze=False,
    api_key=None,
    multi_agent=False,
    taxonomy_path=None,
    progress_callback=None,
):
    """
    Run the full upscale + analysis pipeline.

    Returns a dict with keys: match_summary, clips, fps (if analyze=True), else None.
    """
    # Load JSON
    print(f"Loading detection data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create a lookup for frame data
    frames_data = {f["frame"]: f["athletes"] for f in data["frames"]}

    # Setup Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {width}x{height}, Total frames: {total_frames}, FPS: {fps}")

    # Initialize Restorer based on method
    if method == "diffusion":
        from diffusion_restorer import DiffusionRestorer

        restorer = DiffusionRestorer()
        print("Using Diffusion (ControlNet-Tile) for enhancement.")
    elif method in ("swinir", "hat"):
        from restorer import RealESRGANRestorer

        restorer = RealESRGANRestorer(model_path)
        print(f"Using {method.upper()} for enhancement.")
    else:
        from restorer import RealESRGANRestorer

        restorer = RealESRGANRestorer(model_path)
        print("Using Real-ESRGAN for enhancement.")

    # Initialize Analyzer if requested
    analyzer = None
    if analyze:
        if not api_key:
            raise ValueError("API Key required for analysis.")

        if multi_agent:
            from analyzer import BJJMultiAgentAnalyzer, analyze_sequence_sync
            from service.config import ServiceConfig

            _g_timeout_ms = ServiceConfig().gemini_request_timeout_ms
            analyzer = BJJMultiAgentAnalyzer(
                api_key,
                taxonomy_path=taxonomy_path,
                request_timeout_ms=_g_timeout_ms,
            )
            print("Initialized Gemini BJJ Multi-Agent System (3 Agents + Judge).")
        else:
            from analyzer import BJJTechniqueAnalyzer
            from service.config import ServiceConfig

            _g_timeout_ms = ServiceConfig().gemini_request_timeout_ms
            analyzer = BJJTechniqueAnalyzer(
                api_key,
                taxonomy_path=taxonomy_path,
                request_timeout_ms=_g_timeout_ms,
            )
            print("Initialized Gemini BJJ Single Agent (Thinking Mode).")

    os.makedirs(output_dir, exist_ok=True)

    # Determine which frames to process
    target_frames = sorted(
        [f for f in frames_data.keys() if f % sampling_rate == 0]
    )
    if max_frames:
        target_frames = target_frames[:max_frames]

    print(f"Processing {len(target_frames)} frames using {method}...")

    pbar = tqdm(total=len(target_frames), desc="Enhancing Crops")

    sliding_buffer = []
    analysis_results = []
    current_context = "Start of match."

    # Sliding Window Config
    WINDOW_SIZE = 30
    STRIDE = 15

    current_frame = 0

    for i, frame_idx in enumerate(target_frames):
        # 1. Seek & Capture
        if frame_idx < current_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_frame = frame_idx
        while current_frame < frame_idx:
            cap.grab()
            current_frame += 1
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        athletes = frames_data[frame_idx]
        if not athletes:
            pbar.update(1)
            continue

        # 2. Enhance
        boxes = [a["box"] for a in athletes]
        union_box = get_union_box(boxes)
        square_box = get_padded_square_box(
            union_box, padding=0.2, img_shape=(height, width)
        )
        x1, y1, x2, y2 = square_box
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            pbar.update(1)
            continue

        try:
            enhanced_crop = None
            if method == "diffusion":
                h_crop, w_crop = crop.shape[:2]
                if max(h_crop, w_crop) > 768:
                    scale = 768 / max(h_crop, w_crop)
                    crop = cv2.resize(
                        crop,
                        (int(w_crop * scale), int(h_crop * scale)),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                enhanced_crop = restorer.enhance(crop, strength=diffusion_strength)
            else:
                enhanced_crop = restorer.enhance(crop, target_size=target_size)

            # Save to disk
            if method == "diffusion":
                prefix = "diff_"
            elif method == "swinir":
                prefix = "swinir_"
            elif method == "hat":
                prefix = "hat_"
            else:
                prefix = "esrgan_"
            output_path = os.path.join(
                output_dir, f"{prefix}frame_{frame_idx:06d}.jpg"
            )
            cv2.imwrite(
                output_path, enhanced_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )

            # Add to Buffer
            if analyze:
                from PIL import Image

                img_rgb = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                sliding_buffer.append((frame_idx, pil_img))

                if len(sliding_buffer) >= WINDOW_SIZE:
                    batch_frames = [x[1] for x in sliding_buffer[:WINDOW_SIZE]]
                    batch_indices = [x[0] for x in sliding_buffer[:WINDOW_SIZE]]

                    chunk_idx = len(analysis_results) + 1
                    tqdm.write(
                        f"Analyzing Window {chunk_idx} ({batch_indices[0]}-{batch_indices[-1]})..."
                    )

                    # Choose analysis method
                    if multi_agent:
                        result_json_str = analyze_sequence_sync(
                            analyzer, batch_frames, batch_indices, current_context
                        )
                    else:
                        result_json_str = analyzer.analyze_sequence(
                            batch_frames, batch_indices, current_context
                        )

                    try:
                        result_data = json.loads(result_json_str)
                        if "current_context_summary" in result_data:
                            current_context = result_data["current_context_summary"]

                        analysis_results.append(
                            {
                                "window": chunk_idx,
                                "frames": batch_indices,
                                "analysis": result_data,
                            }
                        )
                    except Exception:
                        tqdm.write("JSON Error")
                        analysis_results.append(
                            {"window": chunk_idx, "raw_error": result_json_str}
                        )

                    sliding_buffer = sliding_buffer[STRIDE:]

                    with open(
                        os.path.join(output_dir, "analysis_raw.json"), "w"
                    ) as f:
                        json.dump(analysis_results, f, indent=2)

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

        pbar.update(1)

        # Report progress
        if progress_callback:
            progress_callback("enhancing", (i + 1) / len(target_frames))

    # Process Final Buffer
    if analyze and len(sliding_buffer) > 0:
        tqdm.write("Analyzing Final Partial Window...")
        batch_frames = [x[1] for x in sliding_buffer]
        batch_indices = [x[0] for x in sliding_buffer]
        if multi_agent:
            result_json_str = analyze_sequence_sync(
                analyzer, batch_frames, batch_indices, current_context
            )
        else:
            result_json_str = analyzer.analyze_sequence(
                batch_frames, batch_indices, current_context
            )

        try:
            result_data = json.loads(result_json_str)
            analysis_results.append(
                {"window": "final", "frames": batch_indices, "analysis": result_data}
            )
        except Exception:
            pass
        with open(os.path.join(output_dir, "analysis_raw.json"), "w") as f:
            json.dump(analysis_results, f, indent=2)

    # Final Deduplication
    result = None
    if analyze:
        tqdm.write("Deduplicating clips...")
        final_clips = deduplicate_clips(analysis_results)

        mode = "Multi-Agent" if multi_agent else "Single-Agent"
        result = {
            "match_summary": f"Analysis generated via {mode}",
            "clips": final_clips,
            "fps": fps,
        }

        with open(os.path.join(output_dir, "analysis_final.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"Final Analysis saved to {os.path.join(output_dir, 'analysis_final.json')}"
        )

    cap.release()
    pbar.close()

    return result
