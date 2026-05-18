"""Detect the top-2 person bounding boxes on a frame for unattended M0 launch."""
import argparse, json, cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = YOLO(args.model)
    img = cv2.imread(args.frame)
    h, w = img.shape[:2]
    res = m.predict(img, classes=[0], conf=args.conf, device=args.device, verbose=False)
    boxes = res[0].boxes
    data = []
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().tolist()
        conf = float(boxes.conf[i].cpu())
        data.append({"xyxy": xyxy, "conf": conf})
    data.sort(key=lambda d: -d["conf"])
    out = {
        "frame": args.frame,
        "frame_shape": [h, w],
        "all_detections": data,
        "box_a": data[0]["xyxy"] if len(data) >= 1 else None,
        "box_b": data[1]["xyxy"] if len(data) >= 2 else None,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
