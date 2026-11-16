"""Copy a source video with the fighters' boxes drawn in: red A (ordered_players[0]), blue B.

Boxes come from ``track_assign.py`` on a TRACK_FPS copy of the whole video (frame 1 = time 0). A gap of up to
GAP tracking frames between two boxes of one fighter is filled by linear interpolation. Audio is dropped.
Also writes one reference crop per fighter (the frame where both are boxed and largest).
Usage: python -m golden.render_boxes SOURCE BOXES_JSON OUT_MP4 CROP_PREFIX [TRACK_FPS]
"""
import json
import subprocess
import sys

import cv2

GAP = 5
STYLE = {"a": ((40, 40, 225), "A"), "b": ((230, 130, 30), "B")}


def filled_boxes(data):
    raw = {int(f): d for f, d in data["boxes"].items()}
    frames = data["frames"]
    filled = {f: dict(raw.get(f, {})) for f in range(1, frames + 1)}
    for lab in ("a", "b"):
        seen = sorted(f for f, d in raw.items() if lab in d)
        for f0, f1 in zip(seen, seen[1:]):
            if 1 < f1 - f0 <= GAP + 1:
                b0, b1 = raw[f0][lab], raw[f1][lab]
                for f in range(f0 + 1, f1):
                    k = (f - f0) / (f1 - f0)
                    filled[f][lab] = [v0 + (v1 - v0) * k for v0, v1 in zip(b0, b1)]
    return filled


def presence(data, track_fps):
    """Per tracking frame: how many fighters are boxed (0-2), after gap filling."""
    return {f: len(d) for f, d in filled_boxes(data).items()}


def main(source, boxes_path, out, crop_prefix, track_fps=10):
    data = json.load(open(boxes_path))
    filled = filled_boxes(data)
    best = max((f for f, d in filled.items() if len(d) == 2),
               key=lambda f: min(d[2] * d[3] for d in filled[f].values()), default=None)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    index = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f = int(round(index / fps * track_fps)) + 1
        index += 1
        if f == best:
            for lab, box in filled[f].items():
                x, y, w, h = (int(v) for v in box)
                pad = int(0.08 * max(w, h))
                crop = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
                cv2.imwrite(f"{crop_prefix}-{0 if lab == 'a' else 1}.png", crop)
        for lab, box in filled.get(f, {}).items():
            x, y, w, h = (int(v) for v in box)
            colour, text = STYLE[lab]
            cv2.rectangle(img, (x, y), (x + w, y + h), colour, 4)
            cv2.rectangle(img, (x, max(0, y - 34)), (x + 40, max(34, y)), colour, -1)
            cv2.putText(img, text, (x + 8, max(26, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    if enc.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {enc.returncode}")
    if best is None:
        raise SystemExit("no frame has both fighters boxed; no reference crops")
    print(json.dumps({"frames": index, "crop_frame": best}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], *(int(x) for x in sys.argv[5:6]))
