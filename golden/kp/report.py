"""Build the experiment report page (HTML + sheet images) from the run outputs.

Inputs under DATA: out/<seg>/{identity.json,candidates.json,stats.json}, packs/<win>/sheet.jpg, runs/*.jsonl,
judges/*.jsonl, summary.json (numbers computed by analyse.py), answers.json
(the written findings). Writes SITE/index.html and SITE/img/<win>.jpg.
Usage: report.py DATA SITE
"""
import html
import json
import sys
from pathlib import Path

import cv2


def esc(x):
    return html.escape(str(x))


def main(data, site):
    data, site = Path(data), Path(site)
    (site / "img").mkdir(parents=True, exist_ok=True)
    S = json.load(open(data / "summary.json"))
    S.update(json.load(open(data / "answers.json")))   # the written findings (date, lede, answers, notes)
    cards = S["cards"]
    for c in cards:
        src = data / "packs" / c["window"] / "sheet.jpg"
        img = cv2.imread(str(src))
        img = cv2.resize(img, (1080, int(img.shape[0] * 1080 / img.shape[1])))
        cv2.imwrite(str(site / "img" / f"{c['window']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 78])
    tpl = (Path(__file__).parent / "report_template.html").read_text()
    page = tpl.replace("/*DATA*/null", json.dumps(S))
    (site / "index.html").write_text(page)
    print("wrote", site / "index.html", len(cards), "cards")


if __name__ == "__main__":
    main(*sys.argv[1:3])
