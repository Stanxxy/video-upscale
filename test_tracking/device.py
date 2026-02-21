"""
Device detection with MPS fallback for Apple Silicon.
Sets PYTORCH_ENABLE_MPS_FALLBACK=1 before any torch import
to handle unimplemented MPS operators in RF-DETR and SAM2.
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
