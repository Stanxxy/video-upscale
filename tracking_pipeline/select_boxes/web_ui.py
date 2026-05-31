"""Web-based box selection UI."""
import base64
import json
import socket
import threading
import time
import webbrowser

import cv2
import numpy as np

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

