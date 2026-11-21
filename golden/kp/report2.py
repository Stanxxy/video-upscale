"""Build the round 2 report page from summary2.json (analyse2.py) and answers2.json (the written findings).
Writes SITE/index.html and copies the bad-case pictures to SITE/img/.  Usage: report2.py DATA SITE"""
import json
import shutil
import sys
from pathlib import Path


def main(data, site):
    data, site = Path(data), Path(site)
    (site / "img").mkdir(parents=True, exist_ok=True)
    S = json.load(open(data / "summary2.json"))
    S.update(json.load(open(data / "answers2.json")))
    for c in S["cards"]:
        shutil.copy(data / "idcards" / c["img"], site / "img" / c["img"])
    tpl = (Path(__file__).parent / "report2_template.html").read_text()
    (site / "index.html").write_text(tpl.replace("/*DATA*/null", json.dumps(S)))
    print("wrote", site / "index.html", len(S["cards"]), "cards")


if __name__ == "__main__":
    main(*sys.argv[1:3])
