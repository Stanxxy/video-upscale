# SAM3 Scout — Evaluator / PM gate checklist

**Date:** 2026-07-20
**Package:** `whole-video-analysis-wt-vlmstudio/eval/adcc/sam3_scout/`
**Plan:** `working_log/plans/2026-07-20-sam3-scout-implementation.md`

## Hard requirements (must all be true)

| Check | Evidence | Pass? |
|-------|----------|-------|
| No `gemini-3.1-flash-lite` | `gemini_arms.assert_scout_gemini_contract` + `tests/test_gemini_arms_guard.py` | YES |
| Gemini knobs = `gemini-3.5-flash` / fps=10 / high | `SCOUT_*` constants in `__init__.py` | YES |
| SAM3 host = gx10 only | README + deploy/dry-run scripts; no bjj-server SAM3 path | YES |
| Concurrency = 1 | `chunk_math.SCOUT_MAX_CONCURRENCY` rejects raises | YES |
| Chunking = 30s / 45s threshold | `chunk_math` + unit tests | YES |
| No fidelity stop-gate | Advisory metrics are soft; `score_report` go/no-go is score+eyeball, not 80% track gate | YES |
| Arms match design A/B/C | Baseline / crop+1080 / highlight→crop+1080; structured-evidence + fixed-zoom-without-SAM3 absent | YES |
| Upscale target 1080 (not prod 768) | `SCOUT_UPSCALE_TARGET = 1080` | YES |
| Unit tests green | `pytest eval/adcc/sam3_scout/tests/` → 25 passed | YES |

## PM acceptance notes

- Clip set: 8 windows in `clips.yaml` / `clips.json` (2 sub / 2 sweep / 2 pass / 2 escape).
- Eyeball viewer: `viewer/index.html` + soft advisory panel.
- Full Gemini spend + gold score report is an **ops run** after clips exist on gx10; driver + scorer are implemented and gated.

## Verdict

**PASS for implementation gate** (code + docs + local tests + gx10 dry-run plumbing).
Live SAM3-on-real-ADCC + paid Gemini arm matrix remains the founder ops step using the shipped scripts.

### gx10 dry-run evidence (2026-07-20)

- Clip: `650d20c7_escape_8372.mp4` (40.04s H.264), concurrency 1, chunked=false (under 45s threshold)
- Artifacts: `/home/sliu1143/sam3_scout/runs/dry_run/{dry_run_tracks.jsonl,dry_run_masks/,dry_run_mask_preview.mp4,dry_run_manifest.json}`
- Advisory: `detection_coverage=1.0`, `ge_2_coverage=1.0`, mean_instances≈4.7
- Ops note: temporarily stopped Docker `vllm-spark` + `vllm-spark-27b` to free unified memory for SAM3; restarted after run.
