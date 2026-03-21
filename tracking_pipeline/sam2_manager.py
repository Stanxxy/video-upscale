"""
SAM2 Video Predictor manager with memory optimization.
Ported from bjj-pose-estimation/bjj_pipeline/models/sam2_manager.py

Adapted for Mac M4 Max: batch-based frame extraction via ffmpeg,
MPS/CPU device support, empty_cache() for MPS.

Memory-bounded: only one batch of frames (~120) is loaded at a time,
regardless of total video length.
"""
import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

from device import get_device, empty_cache


def _apply_mps_float32_patch():
    """
    Patch SAM2VideoPredictor._get_image_feature so images are cast to float32
    before being moved to MPS. MPS does not support float64 tensors, but SAM2's
    lazy frame loader can produce them. Moving float32 → MPS works fine.
    """
    _orig = SAM2VideoPredictor._get_image_feature

    def _patched(self, inference_state, frame_idx, batch_size):
        orig_images = inference_state.get("images")
        if orig_images is not None:
            class _F32Images:
                def __getitem__(_, i):
                    t = orig_images[i]
                    return t.float() if isinstance(t, torch.Tensor) else t
                def __len__(_):
                    return len(orig_images) if hasattr(orig_images, "__len__") else 0
            inference_state["images"] = _F32Images()
        try:
            return _orig(self, inference_state, frame_idx, batch_size)
        finally:
            if orig_images is not None:
                inference_state["images"] = orig_images

    SAM2VideoPredictor._get_image_feature = _patched


# Apply patch at import time so it's active before any predictor is loaded.
_apply_mps_float32_patch()


def _apply_fill_holes_noop_patch():
    """
    Patch fill_holes_in_mask_scores to be a no-op on non-CUDA platforms.

    SAM2's fill_holes_in_mask_scores calls get_connected_components which does
    `from sam2 import _C` — a CUDA-only C extension.  On Mac MPS this import
    fails with ImportError on EVERY frame, creating 120+ exception cycles per
    60-frame batch.  The function is already designed to gracefully skip when
    the extension is missing, so replacing it with a no-op is safe and avoids
    the per-frame exception overhead entirely.
    """
    try:
        from sam2 import _C  # noqa: F401
        return  # C extension available — keep original behaviour
    except ImportError:
        pass

    import sam2.utils.misc as _misc

    def _noop_fill_holes(mask, max_area):
        return mask

    _misc.fill_holes_in_mask_scores = _noop_fill_holes
    print("[sam2] Patched fill_holes_in_mask_scores → no-op (CUDA _C unavailable)")


_apply_fill_holes_noop_patch()


class SAM2Manager:
    """Manages SAM2 Video Predictor state, propagation, and memory.

    Uses batch-based frame loading: only a window of frames is extracted
    and loaded at a time, keeping memory bounded regardless of video length.
    """

    def __init__(self, model_id="facebook/sam2.1-hiera-base-plus", device=None):
        if device is None:
            device = get_device()
        self.device = device
        self.model_id = model_id
        print(f"[sam2] Loading {model_id} on {device}...")
        t0 = time.time()
        try:
            self.predictor = SAM2VideoPredictor.from_pretrained(
                model_id, device=str(device),
            )
        except (RuntimeError, NotImplementedError, TypeError) as e:
            if hasattr(device, "type") and device.type != "cpu":
                print(f"[sam2] {device} failed ({e}), falling back to CPU...")
                device = torch.device("cpu")
                self.device = device
                self.predictor = SAM2VideoPredictor.from_pretrained(
                    model_id, device="cpu",
                )
            else:
                raise
        print(f"[sam2] Model loaded on {self.device} in {time.time() - t0:.1f}s")

        self.inference_state = None
        self.temp_dir = None
        self.is_reset = False

        # Batch loading state
        self.video_path = None
        self.video_fps = 30.0
        self.global_start_frame = 0
        self.total_local_frames = 0
        self.batch_offset = 0  # local_idx offset of the current batch

    # ------------------------------------------------------------------
    # Video metadata (no frame extraction)
    # ------------------------------------------------------------------

    def init_video_meta(self, video_path, start_frame=0, end_frame=None):
        """Capture video metadata without extracting frames.

        Call load_batch() afterwards to extract + init SAM2 for a window.
        """
        self.video_path = video_path
        self.global_start_frame = start_frame

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if end_frame is None:
            end_frame = total
        end_frame = min(end_frame, total)
        self.total_local_frames = end_frame - start_frame

        self.temp_dir = "temp_sam2_frames"
        print(f"[sam2] Video meta: fps={self.video_fps:.1f}, "
              f"frames {start_frame}-{end_frame} "
              f"({self.total_local_frames} total)")

    # ------------------------------------------------------------------
    # Batch frame extraction + SAM2 init
    # ------------------------------------------------------------------

    def load_batch(self, batch_start_local, batch_size):
        """Extract a batch of frames and initialize SAM2 state.

        Args:
            batch_start_local: Start index relative to the tracking range
                               (0-based from global_start_frame).
            batch_size: Number of frames to extract.
        """
        # Clean previous batch
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.batch_offset = batch_start_local
        global_start = self.global_start_frame + batch_start_local

        t0 = time.time()
        actual_count = self._extract_frames_ffmpeg(global_start, batch_size)
        if actual_count == 0:
            # Fallback to cv2 if ffmpeg fails
            actual_count = self._extract_frames_cv2(global_start, batch_size)
        elapsed = time.time() - t0
        print(f"[sam2] Batch: extracted {actual_count} frames "
              f"(local {batch_start_local}-{batch_start_local + actual_count - 1}) "
              f"in {elapsed:.1f}s")

        # Initialize SAM2 state for this batch
        self._init_state()

    def _extract_frames_ffmpeg(self, global_start, count):
        """Extract frames using ffmpeg (fast, hardware-accelerated)."""
        start_sec = global_start / self.video_fps
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{start_sec:.4f}",
                    "-i", self.video_path,
                    "-frames:v", str(count),
                    "-q:v", "5",
                    "-loglevel", "error",
                    os.path.join(self.temp_dir, "%05d.jpg"),
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[sam2] ffmpeg extraction failed: {e}")
            return 0

        # ffmpeg names files starting from 00001.jpg; rename to 00000.jpg
        files = sorted(f for f in os.listdir(self.temp_dir) if f.endswith(".jpg"))
        for i, fname in enumerate(files):
            expected = f"{i:05d}.jpg"
            if fname != expected:
                os.rename(
                    os.path.join(self.temp_dir, fname),
                    os.path.join(self.temp_dir, expected),
                )
        return len(files)

    def _extract_frames_cv2(self, global_start, count):
        """Fallback: extract frames using cv2 (slower but always works)."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {self.video_path}")
        if global_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, global_start)
        idx = 0
        while idx < count:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(
                os.path.join(self.temp_dir, f"{idx:05d}.jpg"), frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )
            idx += 1
        cap.release()
        return idx

    def _init_state(self):
        """Initialize SAM2 inference state from the current temp dir."""
        self.is_reset = False
        try:
            self.inference_state = self.predictor.init_state(
                video_path=self.temp_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,  # sync: batch is small enough
            )
        except (TypeError, RuntimeError) as e:
            if hasattr(self.device, "type") and self.device.type != "cpu":
                print(f"[sam2] init_state failed on {self.device} ({e}), "
                      "falling back to CPU for SAM2...")
                self.device = torch.device("cpu")
                self.predictor = SAM2VideoPredictor.from_pretrained(
                    self.model_id, device="cpu",
                )
                self.inference_state = self.predictor.init_state(
                    video_path=self.temp_dir,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=False,
                )
            else:
                raise
        empty_cache()

    # ------------------------------------------------------------------
    # Re-prompting (for batch boundary carry-over)
    # ------------------------------------------------------------------

    def reprompt_boxes(self, boxes_dict):
        """Re-prompt SAM2 with bounding boxes on frame 0 of the current batch.

        Called after load_batch() to carry tracking state across batch boundaries.

        Args:
            boxes_dict: {obj_id: [x1, y1, x2, y2]} from end of previous batch.
        """
        for obj_id, box in boxes_dict.items():
            try:
                with torch.no_grad():
                    self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=np.array(box),
                    )
            except Exception as e:
                print(f"[sam2] reprompt obj {obj_id} failed: {e}")

    # ------------------------------------------------------------------
    # Legacy init_video (kept for standalone test_tracking usage)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Tracking operations
    # ------------------------------------------------------------------

    def add_initial_box(self, frame_idx, obj_id, box):
        """Initialize a track with a bounding box prompt. Returns initial mask."""
        with torch.no_grad():
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=np.array(box),
            )
            return (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

    def propagate_step(self, start_frame_local, steps=60):
        """
        Run propagation for a limited number of steps.

        Args:
            start_frame_local: Frame index relative to the full tracking range.
            steps: Number of frames to propagate.

        Yields: (batch_relative_frame_idx, obj_ids, mask_logits)
        """
        batch_rel = start_frame_local - self.batch_offset
        with torch.no_grad():
            return self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=batch_rel,
                max_frame_num_to_track=steps,
            )

    def prune_memory(self, max_history=8, max_cond_history=6, flush_cache=False):
        """Prune old prompts AND conditioning feature maps to keep memory bounded.

        Safe to call mid-propagation: SAM2 uses .get() with None fallback
        for non_cond_frame_outputs, so pruned entries are gracefully skipped.

        Frame 0 (the initial prompt / anchor frame for each batch) is always
        protected from pruning to preserve the identity reference.

        Args:
            max_history: Maximum non-conditioning frames to retain per object.
                SAM2 selects at most num_maskmem-1=6 non-cond frames during
                attention (consecutive with stride=1), so storing more than ~8
                wastes RAM. The +2 buffer covers frames cleared by
                _clear_non_cond_mem_around_input near each conditioning frame.
            max_cond_history: Maximum conditioning frames to retain per object.
                SAM2 uses max_cond_frames_in_attn=-1 (attends to ALL cond frames),
                so capping this is critical. 6 matches the non-cond budget
                (num_maskmem-1) for balanced attention.
            flush_cache: If True, call empty_cache() after pruning.
                Keep False during mid-propagation to avoid expensive MPS heap
                compaction on every frame; only set True at batch boundaries.
        """
        if self.inference_state is None:
            return

        # Prune old prompt inputs
        try:
            for obj_idx in self.inference_state["point_inputs_per_obj"]:
                point_inputs = self.inference_state["point_inputs_per_obj"][obj_idx]
                mask_inputs = self.inference_state["mask_inputs_per_obj"].get(obj_idx, {})

                frames = sorted(
                    set(point_inputs.keys()) | set(mask_inputs.keys())
                )
                if len(frames) > max_history:
                    to_remove = frames[:-max_history]
                    for f in to_remove:
                        point_inputs.pop(f, None)
                        mask_inputs.pop(f, None)
                        try:
                            ft = self.inference_state["frames_tracked_per_obj"]
                            if obj_idx in ft:
                                ft[obj_idx].pop(f, None)
                        except (KeyError, TypeError):
                            pass
        except Exception as e:
            print(f"[sam2] WARNING: prune_memory (prompts) failed: {e}")

        # Prune heavy conditioning feature maps (main memory consumer).
        # SAM2 stores per-object output dicts in output_dict_per_obj.
        # cond_frame_outputs and non_cond_frame_outputs are pruned separately
        # because SAM2 attends to ALL cond frames (max_cond_frames_in_attn=-1),
        # making it critical to cap cond history independently.
        try:
            output_dict_per_obj = self.inference_state.get("output_dict_per_obj")
            if output_dict_per_obj is not None:
                for obj_idx in list(output_dict_per_obj.keys()):
                    obj_dict = output_dict_per_obj[obj_idx]

                    cond_d = obj_dict.get("cond_frame_outputs", {})
                    cond_frames = sorted(cond_d.keys())
                    if len(cond_frames) > max_cond_history:
                        # Protect frame 0 (initial anchor) from pruning
                        prunable = [f for f in cond_frames if f != 0]
                        n_to_remove = len(cond_frames) - max_cond_history
                        for f in prunable[:n_to_remove]:
                            del cond_d[f]
                    # pred_masks is not read during propagation (only maskmem_features
                    # is used in memory attention), so null it out to save ~0.25MB/frame.
                    for entry in cond_d.values():
                        entry["pred_masks"] = None

                    non_cond_d = obj_dict.get("non_cond_frame_outputs", {})
                    non_cond_frames = sorted(non_cond_d.keys())
                    if len(non_cond_frames) > max_history:
                        for f in non_cond_frames[:-max_history]:
                            del non_cond_d[f]
                    for entry in non_cond_d.values():
                        entry["pred_masks"] = None
        except Exception as e:
            print(f"[sam2] WARNING: prune_memory (output_dict) failed: {e}")

        if flush_cache:
            empty_cache()

    def reset_state(self):
        """Reset inference state — crucial for scene cuts."""
        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
            self.is_reset = True
            print("[sam2] State reset (memory cleared).")

    def cleanup(self):
        """Remove temp frames and free memory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.inference_state = None
        empty_cache()
