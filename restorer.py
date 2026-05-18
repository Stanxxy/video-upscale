import torch
import cv2
import numpy as np
from spandrel import ModelLoader
from spandrel.__helpers.model_descriptor import UnsupportedDtypeError

# Module-level CUDA tuning (S1). These are idempotent and cheap. allow_tf32
# enables TF32 matmuls on Ampere+; cudnn.benchmark autotunes conv kernels for
# the first forward pass at each input shape and caches the choice.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class RealESRGANRestorer:
    def __init__(self, model_path, device=None):
        if device is None:
            # S1: prefer CUDA over MPS when both available. Production deploy
            # is DGX Spark (CUDA); the prior MPS-first order was M4-Max-biased.
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model using spandrel
        loader = ModelLoader()
        self.model = loader.load_from_file(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Optimization: Convert to half precision if supported
        self._is_half = False
        if self.device.type in ["mps", "cuda"]:
            try:
                self.model.half()
                self._is_half = True
                print("Model converted to half-precision (FP16)")
            except UnsupportedDtypeError:
                print("Model does not support half-precision (FP16). Keeping in FP32.")
            except Exception as e:
                print(f"Warning: Could not convert to half precision: {e}")

        # S1: channels_last layout on CUDA for better tensor-core utilization
        # on conv-heavy workloads. spandrel wraps the underlying nn.Module
        # under `.model`; convert whichever attribute holds the parameters.
        if self.device.type == "cuda":
            try:
                inner = getattr(self.model, "model", self.model)
                inner.to(memory_format=torch.channels_last)
            except Exception as e:
                print(f"Warning: channels_last conversion failed: {e}")

        self.scale = getattr(self.model, 'scale', 4)

    def flush_cache(self):
        """Release cached blocks back to the allocator.

        S1: per-call empty_cache() inside enhance() was wasteful at >0.3 fps
        upscale rate. The worker now calls this once per analyzer-window
        flush instead of once per crop. The OOM guard is preserved -- just
        amortized.
        """
        if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def enhance(self, img, target_size=1024):
        """
        Enhance a single image.
        img: numpy array (BGR)
        target_size: max dimension for the final output
        """
        # Convert BGR to RGB and then to tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        if self._is_half:
            img_tensor = img_tensor.half()
        if self.device.type == "cuda":
            img_tensor = img_tensor.contiguous(memory_format=torch.channels_last)

        # Inference
        output = self.model(img_tensor)

        # Convert back to numpy
        output = output.squeeze(0).permute(1, 2, 0).float().cpu().clamp(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Optimization: Resize to target size if necessary
        h, w = output_bgr.shape[:2]
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            output_bgr = cv2.resize(output_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # S1: empty_cache moved off the hot path. Caller invokes
        # restorer.flush_cache() once per batch / analyzer-window flush.
        return output_bgr

    @torch.no_grad()
    def enhance_batch(self, crops, target_size=1024):
        """Batch-enhance a list of crops with a single forward pass.

        S1: stack N crops via letterbox padding to a common input size, run
        one forward pass, then unpad and resize each output back to its
        per-crop aspect. Same public contract as ``enhance`` but accepts a
        list of BGR numpy arrays and returns a list of BGR numpy arrays in
        the same order.

        Sizing notes:
          - Letterbox to a square equal to the max(H, W) across the batch
            (rounded up to a multiple of 8 so spandrel's strided convs are
            happy). This keeps memory bounded at ~N * (max_side)^2 * 4 ch.
          - For 8 crops at ~860px (the M0 fixture's union-box size) FP16
            activations are ~80-120 MB -- comfortable on 128 GB unified.
        """
        if not crops:
            return []
        if len(crops) == 1:
            return [self.enhance(crops[0], target_size=target_size)]

        # 1. Compute the letterbox target (multiple of 8).
        max_h = max(c.shape[0] for c in crops)
        max_w = max(c.shape[1] for c in crops)
        side = max(max_h, max_w)
        # Round up to a multiple of 8.
        side = ((side + 7) // 8) * 8

        # 2. Build batched tensor on device. Use edge-replication padding (not
        #    zeros): ESRGAN's later layers leak signal from a "hard zero"
        #    letterbox boundary up to ~50 px into the crop. Edge replication
        #    keeps the boundary smooth so the un-padded output matches the
        #    single-crop output within FP16 rounding.
        batch = torch.zeros(
            (len(crops), 3, side, side),
            dtype=torch.float32,
            device=self.device,
        )
        orig_hw: list[tuple[int, int]] = []
        for i, crop in enumerate(crops):
            h, w = crop.shape[:2]
            orig_hw.append((h, w))
            pad_bottom = side - h
            pad_right = side - w
            if pad_bottom > 0 or pad_right > 0:
                padded = cv2.copyMakeBorder(
                    crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE
                )
            else:
                padded = crop
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            batch[i] = t.to(self.device, non_blocking=True)

        if self._is_half:
            batch = batch.half()
        if self.device.type == "cuda":
            batch = batch.contiguous(memory_format=torch.channels_last)

        # 3. Single forward pass.
        out = self.model(batch)
        # out shape: (N, 3, side*scale, side*scale)
        out = out.float().clamp(0, 1).cpu().numpy()

        # 4. Unpad each output back to its source aspect, then resize to
        #    target_size.
        scale_factor = self.scale
        results = []
        for i, (h, w) in enumerate(orig_hw):
            out_h = int(h * scale_factor)
            out_w = int(w * scale_factor)
            arr = out[i, :, :out_h, :out_w]  # (3, out_h, out_w) RGB float32
            arr = (arr * 255.0).round().astype(np.uint8)
            arr = np.transpose(arr, (1, 2, 0))  # HWC RGB
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if max(out_h, out_w) > target_size:
                s = target_size / max(out_h, out_w)
                bgr = cv2.resize(
                    bgr,
                    (int(out_w * s), int(out_h * s)),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            results.append(bgr)
        return results
