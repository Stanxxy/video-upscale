"""Dense frame sheets for strike annotators (agents or people): every 2nd frame of a window, cropped to both
fighters, six 480 px tiles per sheet, each stamped with its time from the window start. No overlays.

Usage: annot.py WINDOWS_JSON OUT_DIR [IDENTITY_FILE]   (window: id, seg, f0, f1; IDENTITY_FILE defaults to
identity.json in each segment's folder; writes OUT_DIR/<id>/d01.jpg ... and OUT_DIR/index.json)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from render import FPS, K, crop_box

STEP, TILE, COLS, ROWS = 2, 480, 3, 2


def main(windows, out, ident_name="identity.json"):
    out = Path(out)
    wins = json.load(open(windows))
    index, cache = {}, {}
    for w in wins:
        seg = w["seg"]
        if seg not in cache:
            cap = cv2.VideoCapture(str(K / "segs" / f"{seg}.mp4"))
            frames = []
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                frames.append(img)
            cache = {seg: dict(frames=frames, ident=json.load(open(K / "out" / seg / ident_name))["frames"])}
        data = cache[seg]
        f1 = min(w["f1"], len(data["frames"]) - 1)
        x0, y0, cw, ch = crop_box(data, w["f0"], f1)
        th = int(TILE * ch / cw)
        tiles, times = [], []
        for f in range(w["f0"], f1 + 1, STEP):
            t = (f - w["f0"]) / FPS
            tile = cv2.resize(data["frames"][f][y0:y0 + ch, x0:x0 + cw], (TILE, th), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (118, 26), (0, 0, 0), -1)
            cv2.putText(tile, f"t={t:.2f}s", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(tile); times.append(round(t, 2))
        d = out / w["id"]
        d.mkdir(parents=True, exist_ok=True)
        sheets = []
        for i in range(0, len(tiles), COLS * ROWS):
            chunk = tiles[i:i + COLS * ROWS]
            chunk += [np.zeros_like(tiles[0])] * (COLS * ROWS - len(chunk))
            rows = [np.hstack(chunk[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
            name = f"d{i // (COLS * ROWS) + 1:02d}.jpg"
            cv2.imwrite(str(d / name), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
            sheets.append(name)
        index[w["id"]] = dict(seg=seg, f0=w["f0"], f1=f1, sheets=sheets, times=times, duration_s=round((f1 - w["f0"]) / FPS, 2))
    json.dump(index, open(out / "index.json", "w"), indent=1)
    print(len(index), "windows", sum(len(v["sheets"]) for v in index.values()), "sheets")


if __name__ == "__main__":
    main(*sys.argv[1:4])
