"""
Human verification/correction of detected bounding boxes.

Primary: Flask web UI with canvas-based drawing (browser opens automatically).
Fallback: cv2 window UI if Flask is not installed.

Usage:
    python select_boxes.py --video ../input_video.mp4 --frame 0
"""
import argparse
import base64
import json
import socket
import threading
import time
import webbrowser

import cv2
import numpy as np


COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]


# ---------------------------------------------------------------------------
# Web UI helpers
# ---------------------------------------------------------------------------

def _find_free_port():
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _frame_to_data_uri(frame):
    """Convert a BGR frame to a base64 JPEG data URI."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_html(data_uri, detections, width, height):
    """Build the HTML page for bounding box selection."""
    det_json = json.dumps(detections or [])
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Select Athletes</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; color: #eee; font-family: system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; padding: 16px; }}
  h2 {{ margin-bottom: 8px; }}
  #status {{ margin-bottom: 10px; font-size: 15px; min-height: 22px; color: #adf; }}
  #canvas-wrap {{ position: relative; cursor: crosshair; }}
  canvas {{ display: block; border: 2px solid #444; }}
  .btn-row {{ margin-top: 12px; display: flex; gap: 12px; }}
  button {{ padding: 10px 28px; font-size: 15px; border: none; border-radius: 6px;
           cursor: pointer; font-weight: 600; }}
  #resetBtn {{ background: #e74c3c; color: #fff; }}
  #submitBtn {{ background: #27ae60; color: #fff; }}
  #submitBtn:disabled {{ background: #555; cursor: not-allowed; }}
  .instructions {{ margin-bottom: 12px; font-size: 13px; color: #aaa;
                   max-width: {min(width, 900)}px; line-height: 1.5; }}
</style>
</head>
<body>
<h2>Select 2 Athletes</h2>
<div class="instructions">
  Click on a yellow detection box to select it, or click-and-drag anywhere to draw a custom box.<br>
  First selection = <b style="color:#2ecc71">Athlete A (green)</b>,
  second = <b style="color:#e74c3c">Athlete B (red)</b>.
</div>
<div id="status">Click a detection or draw a box for Athlete A</div>
<div id="canvas-wrap">
  <canvas id="c" width="{width}" height="{height}"></canvas>
</div>
<div class="btn-row">
  <button id="resetBtn" onclick="resetAll()">Reset</button>
  <button id="submitBtn" disabled onclick="submitBoxes()">Submit (need 2)</button>
</div>

<script>
const IMG_SRC = "{data_uri}";
const DETECTIONS = {det_json};
const W = {width}, H = {height};

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const img = new Image();
let selected = [];      // [{"{"}box:[x1,y1,x2,y2], track_id:1|2, src:"det"|"draw"{"}"}]
let dragging = false;
let dragStart = null;
let dragEnd = null;

img.onload = () => redraw();
img.src = IMG_SRC;

function redraw() {{
  ctx.drawImage(img, 0, 0, W, H);

  // Draw detections (yellow dashed)
  DETECTIONS.forEach((d, i) => {{
    const [x1, y1, x2, y2] = d.box;
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#f1c40f";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(241,196,15,0.7)";
    ctx.font = "bold 16px system-ui";
    ctx.fillText("[" + i + "] " + (d.confidence ? d.confidence.toFixed(2) : ""), x1 + 4, y1 - 6);
  }});

  // Draw selected boxes
  selected.forEach(s => {{
    const [x1, y1, x2, y2] = s.box;
    const color = s.track_id === 1 ? "#2ecc71" : "#e74c3c";
    const label = s.track_id === 1 ? "Athlete A" : "Athlete B";
    ctx.setLineDash([]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = color;
    ctx.font = "bold 18px system-ui";
    ctx.fillText(label, x1 + 4, y2 + 20);
  }});

  // Draw current drag rect
  if (dragging && dragStart && dragEnd) {{
    const x = Math.min(dragStart.x, dragEnd.x);
    const y = Math.min(dragStart.y, dragEnd.y);
    const w = Math.abs(dragEnd.x - dragStart.x);
    const h = Math.abs(dragEnd.y - dragStart.y);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }}
}}

function getPos(e) {{
  const r = canvas.getBoundingClientRect();
  return {{
    x: Math.round((e.clientX - r.left) * (W / r.width)),
    y: Math.round((e.clientY - r.top) * (H / r.height))
  }};
}}

function hitDetection(px, py) {{
  for (let i = 0; i < DETECTIONS.length; i++) {{
    const [x1, y1, x2, y2] = DETECTIONS[i].box;
    if (px >= x1 && px <= x2 && py >= y1 && py <= y2) return i;
  }}
  return -1;
}}

canvas.addEventListener("mousedown", e => {{
  if (selected.length >= 2) return;
  const p = getPos(e);
  dragging = true;
  dragStart = p;
  dragEnd = p;
}});

canvas.addEventListener("mousemove", e => {{
  if (!dragging) return;
  dragEnd = getPos(e);
  redraw();
}});

canvas.addEventListener("mouseup", e => {{
  if (!dragging) return;
  dragging = false;
  const p = getPos(e);
  const dx = Math.abs(p.x - dragStart.x);
  const dy = Math.abs(p.y - dragStart.y);

  if (dx < 10 && dy < 10) {{
    // Click — check if inside a detection
    const idx = hitDetection(p.x, p.y);
    if (idx >= 0) {{
      // Check not already selected
      if (!selected.some(s => s.src === "det" && s._idx === idx)) {{
        addSelection(DETECTIONS[idx].box.slice(), "det", idx);
      }}
    }}
  }} else {{
    // Drag — create custom box
    const x1 = Math.min(dragStart.x, p.x);
    const y1 = Math.min(dragStart.y, p.y);
    const x2 = Math.max(dragStart.x, p.x);
    const y2 = Math.max(dragStart.y, p.y);
    addSelection([x1, y1, x2, y2], "draw", -1);
  }}

  dragStart = null;
  dragEnd = null;
  redraw();
}});

function addSelection(box, src, idx) {{
  if (selected.length >= 2) return;
  const tid = selected.length + 1;
  selected.push({{box, track_id: tid, src, _idx: idx}});
  updateUI();
}}

function updateUI() {{
  const st = document.getElementById("status");
  const btn = document.getElementById("submitBtn");
  if (selected.length === 0) {{
    st.textContent = "Click a detection or draw a box for Athlete A";
    btn.disabled = true;
    btn.textContent = "Submit (need 2)";
  }} else if (selected.length === 1) {{
    st.textContent = "Athlete A selected. Now select Athlete B.";
    btn.disabled = true;
    btn.textContent = "Submit (need 1 more)";
  }} else {{
    st.textContent = "Both athletes selected! Click Submit or Reset.";
    btn.disabled = false;
    btn.textContent = "Submit";
  }}
}}

function resetAll() {{
  selected = [];
  updateUI();
  redraw();
}}

function submitBoxes() {{
  if (selected.length !== 2) return;
  const payload = selected.map(s => ({{
    box: s.box.map(v => Math.round(v * 10) / 10),
    track_id: s.track_id
  }}));
  fetch("/submit", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify(payload)
  }}).then(() => {{
    document.getElementById("status").textContent = "Submitted! You can close this tab.";
    document.getElementById("submitBtn").disabled = true;
    document.getElementById("submitBtn").textContent = "Done";
  }});
}}
</script>
</body>
</html>"""


def select_boxes_web(frame, detections=None):
    """
    Launch a Flask web server for bounding box selection.

    Opens the browser with a canvas showing the frame and detection overlays.
    Returns list of 2 dicts with box and track_id, or None if server fails.
    """
    from flask import Flask, request, Response

    state = {"result": None, "done": False}
    h, w = frame.shape[:2]

    # Scale down for browser if very large
    max_dim = 1200
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        disp_w, disp_h = int(w * scale), int(h * scale)
        disp_frame = cv2.resize(frame, (disp_w, disp_h))
    else:
        disp_w, disp_h = w, h
        disp_frame = frame

    # Scale detection boxes to display size
    scaled_dets = []
    for d in (detections or []):
        scaled_box = [round(c * scale, 1) for c in d["box"]]
        scaled_dets.append({
            "box": scaled_box,
            "confidence": d.get("confidence", 0),
        })

    data_uri = _frame_to_data_uri(disp_frame)
    html = _build_html(data_uri, scaled_dets, disp_w, disp_h)

    app = Flask(__name__)
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return Response(html, content_type="text/html")

    @app.route("/submit", methods=["POST"])
    def submit():
        boxes = request.get_json()
        # Scale boxes back to original frame size
        if scale != 1.0:
            for b in boxes:
                b["box"] = [round(c / scale, 1) for c in b["box"]]
        state["result"] = boxes
        state["done"] = True
        return "ok"

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False),
        daemon=True,
    )
    thread.start()

    # Give server a moment to start, then open browser
    time.sleep(0.5)
    print(f"[select] Web UI opened at {url}")
    webbrowser.open(url)

    # Block until user submits
    while not state["done"]:
        time.sleep(0.2)

    print("[select] Web UI: boxes received.")
    return state["result"]


def draw_detections(frame, detections):
    """Draw numbered detection boxes on a frame."""
    viz = frame.copy()
    for i, det in enumerate(detections):
        box = det["box"]
        x1, y1, x2, y2 = [int(c) for c in box]
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)

        label = f"[{i}] conf={det.get('confidence', 0):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(viz, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(viz, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return viz


def select_boxes_from_detections(frame, detections):
    """
    Show detections and let user accept or redraw.

    Args:
        frame: BGR frame (numpy array)
        detections: List of {"box": [x1,y1,x2,y2], "confidence": float, ...}

    Returns:
        List of 2 dicts: [{"box": [...], "track_id": 1}, {"box": [...], "track_id": 2}]
        or None if cancelled.
    """
    # Try web UI first
    try:
        result = select_boxes_web(frame, detections or [])
        if result is not None:
            return result
    except ImportError:
        print("[select] Flask not installed, using cv2 UI.")
    except Exception as e:
        print(f"[select] Web UI failed ({e}), falling back to cv2.")

    if not detections:
        print("[select] No detections to verify. Drawing manually.")
        return manual_draw_boxes(frame)

    viz = draw_detections(frame, detections)

    # Add instructions
    instructions = [
        "DETECTION VERIFICATION",
        f"Found {len(detections)} person(s). Options:",
        "  Press digit keys to select Athlete A, then Athlete B",
        "  Press 'm' to manually draw both boxes",
        "  Press 'q' or ESC to cancel",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(viz, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("Verify Detections", viz)
    cv2.setWindowProperty("Verify Detections", cv2.WND_PROP_TOPMOST, 1)

    selected = []
    print(f"\n[select] {len(detections)} detection(s) shown.")
    print("[select] Press digit key to select Athlete A, then Athlete B.")
    print("[select] Press 'm' to manually draw. Press 'q'/ESC to cancel.")

    while len(selected) < 2:
        key = cv2.waitKey(0) & 0xFF

        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return None

        if key == ord('m'):
            cv2.destroyAllWindows()
            return manual_draw_boxes(frame)

        # Digit keys 0-9
        if ord('0') <= key <= ord('9'):
            idx = key - ord('0')
            if idx < len(detections):
                if idx not in [s["_idx"] for s in selected]:
                    track_id = len(selected) + 1
                    selected.append({
                        "box": detections[idx]["box"],
                        "track_id": track_id,
                        "_idx": idx,
                    })
                    label = "A" if track_id == 1 else "B"
                    print(f"[select] Athlete {label} = detection [{idx}]")

                    # Redraw with selection highlighted
                    viz2 = draw_detections(frame, detections)
                    for s in selected:
                        si = s["_idx"]
                        box = detections[si]["box"]
                        x1, y1, x2, y2 = [int(c) for c in box]
                        label_text = f"Athlete {'A' if s['track_id'] == 1 else 'B'}"
                        cv2.rectangle(viz2, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        cv2.putText(viz2, label_text, (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 2)
                    cv2.imshow("Verify Detections", viz2)
                else:
                    print(f"[select] Detection [{idx}] already selected.")
            else:
                print(f"[select] No detection [{idx}]. Max index: {len(detections) - 1}")

    cv2.destroyAllWindows()

    # Clean up internal keys
    result = []
    for s in selected:
        result.append({
            "box": [round(c, 1) for c in s["box"]],
            "track_id": s["track_id"],
        })
    return result


def manual_draw_boxes(frame):
    """Let user draw two boxes manually using cv2.selectROI."""
    # Try web UI first
    try:
        result = select_boxes_web(frame, [])
        if result is not None:
            return result
    except ImportError:
        print("[select] Flask not installed, using cv2 UI.")
    except Exception as e:
        print(f"[select] Web UI failed ({e}), falling back to cv2.")

    print("\n[select] MANUAL MODE: Draw bounding box for Athlete A")
    print("[select] Click and drag to draw. Press ENTER to confirm, ESC to cancel.")

    box_a = cv2.selectROI("Draw Athlete A", frame, fromCenter=False,
                          showCrosshair=True)
    cv2.destroyAllWindows()

    if box_a == (0, 0, 0, 0):
        print("[select] Cancelled.")
        return None

    print("[select] Draw bounding box for Athlete B")
    # Show frame with Athlete A drawn
    viz = frame.copy()
    x, y, w, h = box_a
    cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(viz, "Athlete A", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    box_b = cv2.selectROI("Draw Athlete B", viz, fromCenter=False,
                          showCrosshair=True)
    cv2.destroyAllWindows()

    if box_b == (0, 0, 0, 0):
        print("[select] Cancelled.")
        return None

    def roi_to_xyxy(roi):
        x, y, w, h = roi
        return [int(x), int(y), int(x + w), int(y + h)]

    return [
        {"box": roi_to_xyxy(box_a), "track_id": 1},
        {"box": roi_to_xyxy(box_b), "track_id": 2},
    ]


def read_frame(video_path, frame_idx=0):
    """Read a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Box Selection")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    args = parser.parse_args()

    frame = read_frame(args.video, args.frame)
    result = manual_draw_boxes(frame)

    if result:
        for r in result:
            label = "A" if r["track_id"] == 1 else "B"
            print(f"Athlete {label}: {r['box']}")
    else:
        print("Selection cancelled.")
