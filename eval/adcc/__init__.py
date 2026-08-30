"""ADCC ground-truth eval harness — offline half (data-prep + scoring).

See ``working_log/plans/2026-07-12-adcc-eval-harness-precision-plan.md`` (umbrella
repo) for the full design. This package never touches production surface
(``service/app.py``, ``service/worker/**``, ``analyzer.py``, ``taxonomy_mapper.py``,
``VideoEvent``) and spends zero Gemini calls — pure data transformation + offline
scoring over a committed gold snapshot and (separately captured) prediction JSON.
"""
