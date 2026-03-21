"""
Device detection with MPS fallback for Apple Silicon.
Sets PYTORCH_ENABLE_MPS_FALLBACK=1 before any torch import
to handle unimplemented MPS operators in YOLO26 and SAM2.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch


def get_device(force_cpu=False):
    """Return best available device: mps > cuda > cpu."""
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype(device):
    """Return appropriate dtype. bfloat16 for MPS/CUDA, float32 for CPU."""
    if device.type in ("mps", "cuda"):
        return torch.bfloat16
    return torch.float32


def empty_cache():
    """Free device memory (works on CUDA, MPS, or no-op on CPU).

    On MPS, synchronize first to ensure all pending GPU operations complete
    before releasing memory — otherwise async ops may keep tensors alive.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
