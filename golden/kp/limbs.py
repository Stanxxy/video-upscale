"""Stage 4: TAPNext++ point tracks for wrists, elbows, knees and ankles, fused with RTMW-x pose.

The video is cut into 1 s chunks. Each chunk seeds the 8 limb joints of both fighters from the pose at its first
frame and tracks them forward (online, 512 px), with a grid of support points. Fusion per frame and joint: the
pose point when its score (SimCC maximum) is at least 2.5, otherwise the tracked point when it is visible and its seed was
confident. Pose and tracker disagreeing by more than 0.15 torso length is flagged.

Usage: limbs.py SEG_DIR CKPT    (reads kp_rtmw.npz; writes limbs.npz)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path.home() / "bjj/x-mmagold/kp/repos/tapnet"))
from tapnet.tapnextpp.votsp2026.model import TAPNextPP  # noqa: E402

LIMBS = [9, 10, 7, 8, 15, 16, 13, 14]   # L/R wrist, L/R elbow, L/R ankle, L/R knee (COCO body)
CHUNK = 25
POSE_OK, SEED_OK, DISAGREE = 2.5, 2.5, 0.15   # RTMW SimCC maxima (about 1.5-7), not 0-1


def grid_support_points(n, w, h):
    """n points on an aspect-matched grid over the frame (as in tapnet's VOT runner, which needs the vot package)."""
    cols = max(1, round(float(np.sqrt(n * w / h))))
    rows = max(1, int(np.ceil(n / cols)))
    gx, gy = np.meshgrid((np.arange(cols) + 0.5) * (w / cols), (np.arange(rows) + 0.5) * (h / rows))
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)[:n]


def torso_len(kp):
    sh, hp = (kp[5] + kp[6]) / 2, (kp[11] + kp[12]) / 2
    return float(np.linalg.norm(sh - hp))


def main(seg_dir, ckpt):
    seg = Path(seg_dir)
    z = np.load(seg / "kp_rtmw.npz")
    kp, sc = z["kp"], z["score"]
    n = kp.shape[0]
    model = TAPNextPP.from_checkpoint(ckpt, device="cuda", input_resolution=512)
    cap = cv2.VideoCapture(str(seg.parent.parent / "segs" / f"{seg.name}.mp4"))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    frames = frames[:n]
    H, W = frames[0].shape[:2]
    support = grid_support_points(64, W, H)
    trk = np.full((n, 2, 8, 2), np.nan, np.float32)
    vis = np.zeros((n, 2, 8), bool)
    seed_score = np.zeros((n, 2, 8), np.float32)
    for c0 in range(0, n, CHUNK):
        q, where = [], []
        for j in range(2):
            if np.isnan(kp[c0, j, 0, 0]):
                continue
            for li, k in enumerate(LIMBS):
                q.append(kp[c0, j, k]); where.append((j, li))
        if not q:
            continue
        query = np.concatenate([np.array(q, np.float32), support])
        state = None
        for f in range(c0, min(c0 + CHUNK, n)):
            if state is None:
                pos, v, state = model.track_frame(frames[f], query_points_xy=query)
            else:
                pos, v, state = model.track_frame(frames[f], state=state)
            for qi, (j, li) in enumerate(where):
                trk[f, j, li], vis[f, j, li] = pos[qi], v[qi]
                seed_score[f, j, li] = sc[c0, j, LIMBS[li]]
        if c0 % 250 == 0:
            print(c0, flush=True)
    fused = np.full((n, 2, 8, 2), np.nan, np.float32)
    source = np.zeros((n, 2, 8), np.uint8)          # 0 none, 1 pose, 2 tracker
    disagree = np.zeros((n, 2, 8), bool)
    for f in range(n):
        for j in range(2):
            if np.isnan(kp[f, j, 0, 0]):
                continue
            tl = max(torso_len(kp[f, j]), 1.0)
            for li, k in enumerate(LIMBS):
                p_ok = sc[f, j, k] >= POSE_OK
                t_ok = vis[f, j, li] and seed_score[f, j, li] >= SEED_OK and not np.isnan(trk[f, j, li, 0])
                if p_ok:
                    fused[f, j, li], source[f, j, li] = kp[f, j, k], 1
                elif t_ok:
                    fused[f, j, li], source[f, j, li] = trk[f, j, li], 2
                if p_ok and t_ok and np.linalg.norm(kp[f, j, k] - trk[f, j, li]) > DISAGREE * tl:
                    disagree[f, j, li] = True
    np.savez_compressed(seg / "limbs.npz", trk=trk, vis=vis, seed_score=seed_score, fused=fused, source=source,
                        disagree=disagree)
    have = ~np.isnan(kp[:, :, 0, 0])
    print(json.dumps(dict(frames=n, pose_pct=round(float((source[have] == 1).mean()), 3),
                          tracker_fill_pct=round(float((source[have] == 2).mean()), 3),
                          none_pct=round(float((source[have] == 0).mean()), 3),
                          disagree_pct=round(float(disagree[have].mean()), 3))))


if __name__ == "__main__":
    main(*sys.argv[1:3])
