"""Name the two fighters of a video by what tells them apart (glove, shin-guard, shorts colour).

One image call per video on frames spread over its middle; the names become ``ordered_players`` display names
for the census and the A/B labels on the correction page. Usage: python -m golden.name_fighters VIDEO OUT_JSON
"""
import base64
import json
import sys

import cv2
import yaml

from golden.openrouter import complete
from pathlib import Path

MODEL = "gemini-3.8-flash"
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["fighter_a", "fighter_b", "confident", "evidence"],
          "properties": {"fighter_a": {"type": "string"}, "fighter_b": {"type": "string"},
                         "confident": {"type": "boolean"}, "evidence": {"type": "string"}}}
PROMPT = ("These frames come from one MMA fight video. Name the two fighters (never the referee, cornermen or crowd) by "
          "what tells them apart at a glance in every frame: glove colour, shin-guard colour, shorts colour, rash guard. "
          "Each name at most 5 words, starting with the most visible difference, e.g. 'Red gloves, black shorts'. "
          "fighter_a is the fighter whose kit is red or darker when there is such a contrast. Set confident false when "
          "the two fighters cannot be told apart reliably. evidence: one sentence.")


def main(video, out):
    cap = cv2.VideoCapture(video)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    content = [{"type": "text", "text": PROMPT}]
    for k in range(8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (0.2 + 0.6 * k / 7)))
        ok, img = cap.read()
        if ok:
            ok, jpg = cv2.imencode(".jpg", cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 85])
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}})
    entry = yaml.safe_load((Path(__file__).parent / "models.yaml").read_text())[MODEL]
    reply = complete(entry["model"], [{"role": "user", "content": content}], schema=SCHEMA, reasoning="medium", max_tokens=8000)
    names = json.loads(reply["text"])
    names["cost"] = reply["cost"]
    Path(out).write_text(json.dumps(names, indent=1))
    print(json.dumps(names))


if __name__ == "__main__":
    main(*sys.argv[1:3])
