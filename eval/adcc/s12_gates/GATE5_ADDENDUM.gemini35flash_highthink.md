# Gate 5 Addendum — straddler vs non-straddler Arm T hit-rate comparison

**What this measures (Bezos reframe, condition 1, 2026-07-25 pass-with-conditions review):** this addendum compares Arm T's hit rate on the 8 W=60 flagged "straddler" events (`GATE5_REPORT.md`'s production-anchoring table) against the 30 non-straddler core events, **under this harness's own window construction — full, UNTRUNCATED `[t_s-5s, t_s+4s]` native-video context for every window, identical for straddlers and non-straddlers alike.**

**This is explicitly NOT a measurement of whether production's real 720s chunk-boundary truncation causes accuracy loss on these events.** Production's `highlight_analyze_node` clamps each highlight's analyze window to the run's own outer chunk scope (`service/pipelines/executors.py:1286`: `window_start = max(ctx.start_sec, authoritative_start - cfg.preroll_s)`; `:1290`: `window_end = min(ctx.end_sec, authoritative_end + cfg.postroll_s)`) — a straddling event's setup evidence sitting in the PREVIOUS 720s chunk is genuinely invisible to the chunk that reports it, in production. This harness's own window clamp (`gate_common.compute_window`) only clamps to `[0, video_duration_s]`, never to a 720s chunk boundary — every window here gets its full 9s of native video regardless of where a real 720s grid seam would fall. **Known gap:** if Gate 1's Arm T does WORSE on the 8 straddlers even under this harness's full context, that is a signal the straddling events are simply harder cases on their own merits (occlusion, longer setups, etc.) — not evidence about truncation loss specifically, which this addendum cannot measure without re-running the SAME 8 events through the real chunked production pipeline (a separate, not-yet-built probe).

straddler Arm T hit rate: 62.5% (n=8/8 scored, excluded: [])
non-straddler (core-30) Arm T hit rate: 36.7% (n=30/30 scored)

gap (core - straddler) = -25.8pp (flip threshold: >25.0pp) -> **no flip**
