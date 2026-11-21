"""One line per segment from identity_<DET>.json summaries. Usage: idsum.py DATA DET SEG..."""
import json
import sys
from pathlib import Path

for seg in sys.argv[3:]:
    s = json.load(open(Path(sys.argv[1]) / "out" / seg / f"identity_{sys.argv[2]}.json"))["summary"]
    w = " ".join(f"{k[:3]}{v:.1f}" for k, v in s["kit"]["weight"].items())
    print(f"{seg:5s} {sys.argv[2]:8s} both {s['both_pct']:.2f} one {s['one_pct']:.2f} cuts {s['cuts']:2d} merged {s['merged']:3d} "
          f"bare {int(s['bare_fight'])} tracks {s['fighter_tracks']}/{s['all_tracks']} kit[{s['kit']['source']}] {w}")
