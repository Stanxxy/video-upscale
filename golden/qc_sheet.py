"""QC contact sheet: N frames of a source video with the tracker's A (red) / B (blue) boxes.
Usage: python -m golden.qc_sheet SOURCE BOXES_JSON OUT_JPG [N] [TRACK_FPS]"""
import json
import sys

import cv2
import numpy as np


def main(source, boxes_path, out, n=12, track_fps=10):
    n, track_fps = int(n), int(track_fps)
    boxes = json.load(open(boxes_path))["boxes"]
    cap = cv2.VideoCapture(source)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    tiles = []
    for k in range(n):
        t = total * (0.15 + 0.7 * k / max(n - 1, 1))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            continue
        for lab, box in boxes.get(str(int(round(t * track_fps)) + 1), {}).items():
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 225) if lab == "a" else (230, 130, 30), 5)
        cv2.putText(img, f"{int(t // 60)}:{int(t % 60):02d}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        tiles.append(cv2.resize(img, (480, 270)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(out, np.vstack(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
