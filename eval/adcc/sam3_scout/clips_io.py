"""Load ``clips.yaml`` without requiring PyYAML when possible."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_clips(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        # Fallback: expect a sibling clips.json generated for ops scripts.
        sibling = path.with_suffix(".json")
        if sibling.is_file():
            return json.loads(sibling.read_text())
        raise SystemExit(
            "PyYAML not installed and clips.json missing. "
            "pip install pyyaml OR generate clips.json from clips.yaml."
        )


def dump_clips_json(yaml_path: str | Path, json_path: str | Path | None = None) -> Path:
    data = load_clips(yaml_path)
    out = Path(json_path) if json_path else Path(yaml_path).with_suffix(".json")
    out.write_text(json.dumps(data, indent=2))
    return out
