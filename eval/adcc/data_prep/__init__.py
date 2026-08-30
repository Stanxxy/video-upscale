"""Data-prep: ADCC raw JSON (``metadata_and_events.json`` / ``video_links.json``)
-> committed gold snapshot (``eval/adcc/gold_snapshot/normalized_gold.jsonl`` +
``manifest.json``) + frozen ``split_assignment.json``.

Runs OFFLINE, no Gemini calls. Reads the two small source JSONs (pulled from
``bjj-server:/mnt/e/assets/data/bjj-vision/all|ADCC|all/`` — the ~100GB
``videos/`` directory itself is never copied)."""
