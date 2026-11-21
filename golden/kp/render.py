"""Frame packs for the classifier ablation and the judges.

For each window: frames at about 12 fps (at most 16), cropped to both fighters, 640 px, each stamped with its time
from the window start, in four conditions:
  raw    no overlay
  id     identity boxes (A red, B blue) with the letter
  pose   id + RTMW-x skeletons (joints with SimCC score under 2.0 hidden)
  trace  pose + the last 0.4 s of each fused wrist/ankle path (a thick trail for the candidate's limb)
plus clip.mp4 (the raw window at 25 fps, same crop) for the video condition.

Usage: render.py WINDOWS_JSON OUT_DIR [DET CONDS MAX_FRAMES]   (window: id, seg, f0, f1, optional focus {fighter, li})
With DET (light, rtmo, ...) the boxes and skeletons come from identity_<DET>.json and kp_<DET>.npz (round 2: no
limb tracker, so no trace condition) and only the frames of the comma-separated CONDS are written.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

K = Path.home() / "bjj/x-mmagold/kp"
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 5), (0, 6)]
COL = {0: (40, 40, 230), 1: (230, 120, 30)}     # BGR: A red, B blue
TRAIL = [0, 1, 4, 5]                             # wrists, ankles in limbs.npz order
FPS = 25


def load(seg, det=None):
    d = K / "out" / seg
    cap = cv2.VideoCapture(str(K / "segs" / f"{seg}.mp4"))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    if det:
        return dict(frames=frames, ident=json.load(open(d / f"identity_{det}.json"))["frames"],
                    kp=np.load(d / f"kp_{det}.npz"))
    return dict(frames=frames, ident=json.load(open(d / "identity.json"))["frames"],
                kp=np.load(d / "kp_rtmw.npz"), limbs=np.load(d / "limbs.npz"))


def crop_box(data, f0, f1):
    xs, ys = [], []
    for f in range(f0, f1 + 1):
        for s in ("a", "b"):
            b = data["ident"][str(f)].get(s)
            if b:
                x, y, w, h = b["box"]
                xs += [x, x + w]; ys += [y, y + h]
    H, W = data["frames"][0].shape[:2]
    if not xs:
        return 0, 0, W, H
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pw, ph = (x1 - x0) * 0.12 + 20, (y1 - y0) * 0.12 + 20
    x0, x1, y0, y1 = x0 - pw, x1 + pw, y0 - ph, y1 + ph
    w, h = max(x1 - x0, 360), max(y1 - y0, 360)
    if w / h < 4 / 3:
        w = h * 4 / 3
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = min(w, W), min(h, H)
    x0, y0 = int(np.clip(cx - w / 2, 0, W - w)), int(np.clip(cy - h / 2, 0, H - h))
    return x0, y0, int(w), int(h)


def draw(img, data, f, cond, focus):
    out = img.copy()
    if cond == "raw":
        return out
    fr = data["ident"][str(f)]
    kp, sc = data["kp"]["kp"], data["kp"]["score"]
    fused = data["limbs"]["fused"] if cond == "trace" else None
    for j, s in enumerate(("a", "b")):
        if s not in fr:
            continue
        x, y, w, h = map(int, fr[s]["box"])
        cv2.rectangle(out, (x, y), (x + w, y + h), COL[j], 2)
        cv2.putText(out, s.upper(), (x + 4, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL[j], 2)
        if cond in ("pose", "trace"):
            for a, b in EDGES:
                if sc[f, j, a] >= 2.0 and sc[f, j, b] >= 2.0:
                    cv2.line(out, tuple(map(int, kp[f, j, a])), tuple(map(int, kp[f, j, b])), COL[j], 2)
            for k in (9, 10, 15, 16):
                if sc[f, j, k] >= 2.0:
                    cv2.circle(out, tuple(map(int, kp[f, j, k])), 5, COL[j], -1)
        if cond == "trace":
            trail = TRAIL + ([focus["li"]] if focus and focus["fighter"] == s and focus["li"] not in TRAIL else [])
            for li in trail:
                pts = [fused[g, j, li] for g in range(max(0, f - 10), f + 1)]
                pts = [tuple(map(int, p)) for p in pts if not np.isnan(p).any()]
                thick = 5 if focus and focus["fighter"] == s and focus["li"] == li else 2
                for p, q in zip(pts, pts[1:]):
                    cv2.line(out, p, q, (60, 230, 255) if thick > 2 else (200, 255, 255), thick)
    return out


def main(windows_path, out_dir, det=None, conds="raw,id,pose,trace", max_frames="16"):
    wins = json.load(open(windows_path))
    out_dir = Path(out_dir)
    cache = {}
    index = {}
    for wdef in wins:
        seg = wdef["seg"]
        if seg not in cache:
            cache.clear(); cache[seg] = load(seg, det)
        data = cache[seg]
        f0, f1 = max(0, wdef["f0"]), min(len(data["frames"]) - 1, wdef["f1"])
        x0, y0, w, h = crop_box(data, f0, f1)
        step = max(2, int(np.ceil((f1 - f0 + 1) / int(max_frames))))
        picks = list(range(f0, f1 + 1, step))
        d = out_dir / wdef["id"]
        d.mkdir(parents=True, exist_ok=True)
        scale = 640 / w
        for cond in conds.split(","):
            for i, f in enumerate(picks):
                img = draw(data["frames"][f], data, f, cond, wdef.get("focus"))[y0:y0 + h, x0:x0 + w]
                img = cv2.resize(img, (640, int(h * scale)))
                cv2.putText(img, f"t={(f - f0) / FPS:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(img, f"t={(f - f0) / FPS:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imwrite(str(d / f"{cond}_{i:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        index[wdef["id"]] = dict(seg=seg, f0=f0, f1=f1, picks=picks, crop=[x0, y0, w, h],
                                 times=[round((f - f0) / FPS, 2) for f in picks])
        if det:
            continue
        # judge sheet: 6 trace frames around the flagged peak (or spread over the window), 3x2 at 480 px
        peak = wdef.get("peak_f")
        sel = [peak + o for o in (-12, -6, -2, 0, 4, 8)] if peak is not None else list(np.linspace(f0, f1, 6).astype(int))
        tiles = []
        for f in sel:
            f = int(np.clip(f, f0, f1))
            img = draw(data["frames"][f], data, f, "trace", wdef.get("focus"))[y0:y0 + h, x0:x0 + w]
            img = cv2.resize(img, (480, int(h * 480 / w)))
            lab = f"t={(f - f0) / FPS:.2f}s" + (" PEAK" if f == peak else "")
            cv2.putText(img, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(img, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(img)
        sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        cv2.imwrite(str(d / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        vw = cv2.VideoWriter(str(d / "clip_raw.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (640, int(h * scale)))
        for f in range(f0, f1 + 1):
            vw.write(cv2.resize(data["frames"][f][y0:y0 + h, x0:x0 + w], (640, int(h * scale))))
        vw.release()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "clip_raw.avi"), "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", str(d / "clip.mp4")], check=True)
        (d / "clip_raw.avi").unlink()
    json.dump(index, open(out_dir / f"index_{Path(windows_path).stem}.json", "w"))
    print("rendered", len(index))


if __name__ == "__main__":
    main(*sys.argv[1:6])
