# Vision engine (whole-video-analysis)

The Gemini-native analysis engine behind BJJ Vision. It runs the
position / actor / action-class / outcome analysis over a match video and
publishes the resulting events. The specific move lives in each item's notes;
there are no technique lists.

## Parts

- `service/` — FastAPI service: `POST /track` (highlight analysis), job status, SSE events, QA routes, and the account DAG entry points (`dag_server`, `current_account_dag_server`, `analysis_run_dispatch`).
- `analysis_core/` — the DAG prompts, outputs and presets (`bjj_analysis_core`).
- `analysis_observer/` — read-only lifecycle and checkpoint observer.
- `analyzer.py` — Gemini sliding-window analyzer used by the QA identity routes.
- `contracts/` — OpenAPI and backend checkpoint contracts. `docs/` — architecture and per-module notes.

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-service.txt
cp .env.example .env   # set BJJ_GEMINI_API_KEY and AWS values
./service.sh start     # port 8000; ./service.sh dev for --reload
curl -s http://localhost:8000/health
```

System `ffmpeg` is required for `/qa/vlm-image`.

## Tests

```bash
pytest tests --ignore=tests/run_host --ignore=tests/checkpoint_journal --ignore=tests/media_preparation
pytest analysis_core analysis_observer
```

`tests/run_host`, `tests/checkpoint_journal` and `tests/media_preparation` need local Cassandra and S3 endpoints.
