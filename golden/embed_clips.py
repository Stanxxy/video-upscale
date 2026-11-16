"""Frozen V-JEPA 2 (ViT-L, 256 px) clip embeddings for the technique-class probe (runs on gx10).

Each task clip -> 32 frames sampled evenly -> mean-pooled last hidden state (1024-d). Identity-free: it sees
the whole exchange, so it can say what was thrown but not by whom.
Usage: embed_clips.py CLIP_DIR OUT_NPZ
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

MODEL, FRAMES = "facebook/vjepa2-vitl-fpc64-256", 32


def frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = set(np.linspace(0, max(n - 1, 0), FRAMES).round().astype(int).tolist())
    out, i, last = [], 0, None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in idx:
            last = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        while len(out) < FRAMES and i >= np.linspace(0, max(n - 1, 0), FRAMES).round()[len(out)] and last is not None:
            out.append(last)
        i += 1
    while out and len(out) < FRAMES:
        out.append(out[-1])
    return np.stack(out) if out else None


def main(clip_dir, out):
    proc = AutoVideoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    done = dict(np.load(out)) if os.path.exists(out) else {}
    for k, path in enumerate(sorted(glob.glob(f"{clip_dir}/*.mp4"))):
        tid = os.path.basename(path)[:-4]
        if tid in done:
            continue
        video = frames(path)
        if video is None:
            continue
        inputs = proc(video, return_tensors="pt").to("cuda")
        inputs = {k2: (v.half() if v.is_floating_point() else v) for k2, v in inputs.items()}
        with torch.no_grad():
            h = model.get_vision_features(**inputs) if hasattr(model, "get_vision_features") else model(**inputs).last_hidden_state
        done[tid] = h.float().mean(dim=1)[0].cpu().numpy()
        if k % 100 == 0:
            print(k, flush=True)
    np.savez(out, **done)
    print("done", len(done))


if __name__ == "__main__":
    main(*sys.argv[1:3])
