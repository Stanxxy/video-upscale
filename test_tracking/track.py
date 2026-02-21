"""
SAM 2.1 Video Object Tracking via HuggingFace Transformers.

Usage:
    python track.py --video ../input_video.mp4 --detections output/detections.json --max_frames 100
    python track.py --video ../input_video.mp4 --detections output/detections.json --cpu
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from device import get_device, get_dtype

SAM2_MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


def extract_frames(video_path, max_frames=None, sampling_rate=1):
    """Extract frames from video as PIL Images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    frame_indices = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sampling_rate == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            frame_indices.append(idx)
        idx += 1
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    print(f"[track] Extracted {len(frames)} frames from {total} total "
          f"({width}x{height} @ {fps:.1f}fps)")
    return frames, frame_indices, (height, width), fps


def mask_to_bbox(mask):
    """Convert binary mask to [x1, y1, x2, y2] bounding box."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def prune_memory(inference_session, max_prompts=30):
    """
    Trim old conditioning frames to bound memory growth.

    SAM 2's conditioning frames (cond_frame_outputs) accumulate indefinitely
    during propagation. This function keeps only the last max_prompts frames,
    preventing unbounded memory growth and maintaining fast inference.

    Ported from bjj-pose-estimation/bjj_pipeline/run_sam2_hybrid.py:260-285.
    """
    for obj_idx in list(inference_session.point_inputs_per_obj.keys()):
        point_inputs = inference_session.point_inputs_per_obj.get(obj_idx, {})
        mask_inputs = inference_session.mask_inputs_per_obj.get(obj_idx, {})

        prompted_frames = sorted(
            set(point_inputs.keys()) | set(mask_inputs.keys())
        )

        if len(prompted_frames) > max_prompts:
            to_remove = prompted_frames[:-max_prompts]
            for f_idx in to_remove:
                point_inputs.pop(f_idx, None)
                mask_inputs.pop(f_idx, None)
                # Also trim conditioning frame outputs
                obj_output = inference_session.output_dict_per_obj.get(
                    obj_idx, {}
                )
                obj_output.get("cond_frame_outputs", {}).pop(f_idx, None)
                # Trim frames_tracked
                inference_session.frames_tracked_per_obj.get(
                    obj_idx, {}
                ).pop(f_idx, None)


def track_objects(video_path, detections, model_name="tiny", max_frames=None,
                  sampling_rate=1, force_cpu=False, max_prompts=30):
    """
    Track detected persons through video using SAM 2.1.

    Args:
        video_path: Path to video file
        detections: List of dicts with "box" and "track_id" keys
        model_name: SAM2 model size ("tiny", "small", "base_plus", "large")
        max_frames: Limit frames to process
        sampling_rate: Process every Nth frame
        force_cpu: Force CPU inference
        max_prompts: Max conditioning frames to keep in memory (sliding window)

    Returns:
        (tracking_json, raw_results) where tracking_json is compatible with
        the existing tracking format and raw_results contains masks for
        visualization.
    """
    device = get_device(force_cpu)
    dtype = get_dtype(device)

    try:
        return _track_impl(video_path, detections, model_name, max_frames,
                           sampling_rate, device, dtype, max_prompts)
    except RuntimeError as e:
        if "MPS" in str(e) or "not implemented" in str(e).lower():
            print(f"[track] MPS error: {e}")
            print("[track] Falling back to CPU...")
            return _track_impl(video_path, detections, model_name, max_frames,
                               sampling_rate, torch.device("cpu"),
                               torch.float32, max_prompts)
        raise


def _track_impl(video_path, detections, model_name, max_frames,
                sampling_rate, device, dtype, max_prompts=30):
    """Internal tracking implementation."""
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    model_id = SAM2_MODELS[model_name]
    print(f"[track] Loading {model_id} on {device} ({dtype})...")
    t0 = time.time()

    model = Sam2VideoModel.from_pretrained(model_id).to(device, dtype=dtype)
    processor = Sam2VideoProcessor.from_pretrained(model_id)
    print(f"[track] Model loaded in {time.time() - t0:.1f}s")

    # Extract video frames
    frames, frame_indices, (vid_h, vid_w), fps = extract_frames(
        video_path, max_frames, sampling_rate
    )

    # Initialize video session
    print("[track] Initializing video session...")
    inference_session = processor.init_video_session(
        video=frames,
        inference_device=device,
        dtype=dtype,
    )

    # Add detection boxes on the first frame
    ann_frame_idx = 0
    obj_ids = [d["track_id"] for d in detections]
    input_boxes = [[d["box"] for d in detections]]

    print(f"[track] Adding {len(detections)} object(s) on frame {ann_frame_idx}...")
    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=obj_ids,
        input_boxes=input_boxes,
    )

    # Run initial segmentation on annotation frame
    with torch.inference_mode():
        model(inference_session=inference_session, frame_idx=ann_frame_idx)

    # Propagate through all frames
    print("[track] Propagating tracking through video...")
    tracking_results = {}

    with torch.inference_mode():
        for output in tqdm(
            model.propagate_in_video_iterator(inference_session),
            total=len(frames),
            desc="Tracking",
        ):
            frame_masks = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[vid_h, vid_w]],
                binarize=True,
            )[0]  # (num_objects, 1, H, W)

            athletes = []
            for obj_i, obj_id in enumerate(inference_session.obj_ids):
                mask = frame_masks[obj_i, 0]  # (H, W) binary
                bbox = mask_to_bbox(mask)
                if bbox is not None:
                    athletes.append({
                        "track_id": int(obj_id),
                        "box": bbox,
                        "mask_area": int(mask.sum()),
                    })

            tracking_results[output.frame_idx] = {
                "masks": frame_masks.cpu(),
                "athletes": athletes,
            }

            # Prune old conditioning frames to keep memory bounded
            prune_memory(inference_session, max_prompts)

    # Build output JSON compatible with existing format
    output_json = {
        "video": video_path,
        "model": model_id,
        "fps": fps,
        "video_size": [vid_w, vid_h],
        "num_objects": len(obj_ids),
        "frames": [],
    }

    for local_idx in sorted(tracking_results.keys()):
        actual_frame = (frame_indices[local_idx]
                        if local_idx < len(frame_indices) else local_idx)
        output_json["frames"].append({
            "frame": actual_frame,
            "athletes": tracking_results[local_idx]["athletes"],
        })

    print(f"[track] Tracked {len(obj_ids)} object(s) across "
          f"{len(tracking_results)} frames")
    return output_json, tracking_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 2.1 Video Tracking")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--detections", required=True,
                        help="Path to detections JSON from detect.py")
    parser.add_argument("--model", default="tiny",
                        choices=list(SAM2_MODELS.keys()),
                        help="SAM2 model size")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames to process")
    parser.add_argument("--sampling_rate", type=int, default=1,
                        help="Process every Nth frame")
    parser.add_argument("--max_prompts", type=int, default=30,
                        help="Max conditioning frames in memory (sliding window)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--output", default="output/tracking.json")
    args = parser.parse_args()

    with open(args.detections) as f:
        det_data = json.load(f)

    tracking_json, _ = track_objects(
        args.video, det_data["persons"], args.model,
        args.max_frames, args.sampling_rate, args.cpu,
        args.max_prompts,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(tracking_json, f, indent=2)
    print(f"[track] Saved to {args.output}")
