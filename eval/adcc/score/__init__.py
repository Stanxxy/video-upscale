"""Offline scoring: Hungarian matcher (``matcher.py``), P/R/F1 + confusion
matrix + PR-curve metrics (``metrics.py``), and the founder-facing
``report.md``/``report.json`` CLI (``report.py``). Zero network, zero Gemini
calls — a pure function over the committed gold snapshot and a directory of
already-captured prediction JSON files."""
