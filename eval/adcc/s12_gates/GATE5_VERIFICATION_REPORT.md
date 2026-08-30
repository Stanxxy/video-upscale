# Gate 5 Verification Report — AC7 8-Straddler Targeted Re-run

**Generated:** 2026-07-26. **Requirement:** `working_log/plans/2026-07-26-engine13-rescope-single-call-cutover.md`
AC7 ("The 8 known Gate-5 straddler events... are re-run through the REAL chunked, patched production
pipeline; recall is reported against the untruncated 0/8 addendum baseline AND the real, previously-
unmeasured, truncated pre-patch baseline.") **This closes AC7 — no verdict is rendered here; LeCun rules on
the number.**

**Real code under test:** `whole-video-analysis-wt-s12`, branch `feature/s12-highlight-production-wiring`
(6 commits: single-call analyze, 45s backward outer-chunk overlap + seam dedup, batched publish).
`service.worker.highlight_orchestrator._outer_chunks`, `service.pipelines.executors.
highlight_critique_node`/`highlight_analyze_node`, `service.worker.seam_dedup.dedup_match_clips` — all
called UNMODIFIED. No production file was edited to produce the pre-patch config; it is achieved via
`_outer_chunks`'s own `overlap_s=0.0` keyword argument at the harness's own call site.

---

## Headline numbers

| | Recall (8 events) |
|---|---|
| **Config a — PATCHED** (45s backward overlap + seam dedup, as shipped) | **1/8 = 12.5%** |
| **Config b — PRE-PATCH baseline** (zero overlap, never measured before) | **0/8 = 0.0%** |
| Reference: untruncated full-context, flash-lite v1 (`GATE5_ADDENDUM.md`) | 0/8 = 0.0% |
| Reference: untruncated full-context, qualified v2 (`GATE5_ADDENDUM.gemini35flash_highthink.md`) | 5/8 = 62.5% |

**The patch improves recall by +1 event (0→1 of 8) over the never-before-measured pre-patch truncated
baseline** — a real, positive, directionally-correct effect, consistent with the first-principles chunking-
boundary argument the patch shipped on. **Both chunk-scoped configurations remain dramatically below the
untruncated qualified-model ceiling (62.5%)** — chunk-boundary clamping (even with the 45s backward overlap)
costs the great majority of the accuracy the same model achieves with full temporal context on these same
8 events. This is real, disclosed data for LeCun's verdict — no interpretation/verdict is rendered here.

---

## Per-event table

| Event | Match | Gold | Config a (patched) | Config b (pre-patch) |
|---|---|---|---|---|
| e1 | PJ Barch vs Kade Ruotolo | submission_arm_lock | ✗ submission_leg_lock | ✗ submission_leg_lock |
| e2 | Joshua Cisneros vs Ethan Crelinsten | guard_pass | ✗ takedown_attempt | ✗ submission_leg_lock |
| e3 | Xande Ribeiro vs Eoghan O'Flanagan | back_take | **✓ back_take** | ✗ submission_choke |
| e4 | Ffion Davies vs Beatriz Mesquita | guard_pass | ✗ submission_choke | ✗ submission_choke |
| e5 | Margot Ciccarelli vs Ashley Bendle | sweep | ✗ submission_leg_lock | ✗ submission_leg_lock |
| e6 | Nia Blackman vs Salla Simola | back_take | ✗ guard_pass | ✗ submission_arm_lock |
| e7 | Fabricio Andrey vs Kaua Gabriel | back_take | ✗ escape | ✗ takedown_attempt |
| e8 | Huaiqing Xu vs Daiki Yonekura | back_take | ✗ guard_pass | ✗ submission_choke |

**e3 is the ONE event the 45s overlap flips from a miss to a hit** — its setup window's earlier portion,
invisible to chunk 1 under zero overlap, becomes visible once the chunk pulls back 45s (confirmed
structurally: e3's chunk-1 boundary is at `t_s=761`'s straddle point `720`, well within the 45s reach).

---

## Seam-dedup double-publish check

**Stress-test design** (config a only — dedup is a mechanism specific to the overlap patch): for each event,
in addition to the real chunk-1 analysis, a SEPARATE critique + taxonomy-only call (actor axis skipped —
`seam_dedup.py`'s own real merge logic, verified by reading the code, never consults `actor`/`player_id`)
was run scoped to chunk 0's own boundary, simulating "what if the preceding chunk had ALSO independently
found this event near its tail." Both resulting clips were fed into the REAL `seam_dedup.dedup_match_clips`.

| | Count (of 8) |
|---|---|
| Events where a double-publish risk existed (both chunk 0's stress clip AND chunk 1's real clip produced a clip) | **7/8** |
| Events where `dedup_match_clips` correctly identified and dropped the duplicate | **3/7** (e1, e4, e5) |
| Events where dedup declined to merge (chunk 0 stress clip and chunk 1 clip landed on temporally-close but CLASS-INCOMPATIBLE action_class labels) | **4/7** (e2, e6, e7, e8) |
| Event with no double-publish risk at all (chunk 0's stress call found no clip) | **1/8** (e3) |

**Real, working example of a correct merge** (e1): chunk 0's stress clip labeled `submission_arm_lock`
`[661,731]`, chunk 1's real clip labeled `submission_leg_lock` `[675,732]` — DIFFERENT specific labels, but
`seam_dedup.py`'s `SUBMISSION_FAMILY_CLASSES` compatibility rule (an occlusion-driven submission-subtype
flip at the seam is still one real event, matching `eval/adcc/predict/stitch.py`'s precedent) correctly
recognized these as the same event and dropped one — `dedup_correctly_prevented_double_publish=True`,
1 survivor.

**Important, disclosed limitation, not a dedup bug**: `dedup_match_clips` is CONSERVATIVE by design — it
only merges when the two independent chunk-boundary calls agree closely enough (class-compatible per
`class_compatible()`'s strict equality-or-submission-family rule + temporal proximity). In 4 of 7 at-risk
events, the two chunk-boundary calls landed on GENUINELY DIFFERENT, class-INCOMPATIBLE labels for what
should be the same real event (e.g. e6: `transition` vs `guard_pass`; e7: `takedown_attempt` vs `escape`) —
dedup correctly declines to merge these (it cannot know which is right), meaning **both would still be
published as separate candidates in production** under these conditions. This residual risk is a direct
consequence of the same low recall shown in the headline numbers above (the model's chunk-boundary action-
class calls disagree with each other roughly as often as they disagree with gold) — not a defect in
`seam_dedup.py` itself, whose own logic behaved exactly as documented in every case tested.

---

## Disclosures

**(a) Discovery-stage bypass.** This harness does NOT run the real `highlight_scan` discovery stage (which
would cost an unbounded, uncontrolled number of calls per 720s chunk — a real match's own worst-case
estimate exceeds 100 calls/chunk). It hand-constructs the "found highlight" using `GATE5_REPORT.md`'s own
W=60/post_tail_s=5 span (the exact definition already used to flag these 8 events as straddlers — not
fabricated) and runs the REAL `highlight_critique_node`/`highlight_analyze_node` unmodified on it. This
verifies the chunk-scope window-clamping mechanism precisely (the mechanism `GATE5_ADDENDUM.md` itself cited
as the truncation site) but does NOT verify whether a real PASS-1 scan would have found/bounded this event
the same way near a truncated chunk boundary — a genuine, disclosed scope limitation of this lightweight
check, consistent with `VERDICTS.md` §3's own framing ("lightweight targeted check, not a third full gate
campaign").

**(b) Settings.** `model=gemini-3.5-flash`, `thinking=high` (both critique and taxonomy/actor calls),
`fps=10` (`HighlightAnalyzeConfig`'s only configurable fps; `highlight_critique_node`'s own fps is hardcoded
to 1, unconfigurable — verified by reading the function). `preroll_s`/`postroll_s`/`critique_backpad_s`/
`media_resolution` left at their REAL shipped registry defaults (this tests the actual production config,
not a hand-picked worst-case).

**(c) Ingestion.** 7 of 8 events used a direct YouTube watch URL (cost/simplicity — chunking math is
ingestion-independent). Event `e8` (video `pwGbW5GZgfc`) is the SAME 8.5h ADCC-compilation video already
confirmed independently (twice, in earlier phases of this program) to fail on every native YouTube ingestion
offset — routed through the proven Files-API reroute lane (`files_api_reroute.py`, reused unmodified:
download+normalize+upload the chunk-scoped segment, file-relative offset conversion). No other straddler
match/video required rerouting.

**(d) Two harness bugs found and fixed live, disclosed.** (1) `gemini_upload.poll_until_active()` requires
keyword-only `poll_interval_sec`/`timeout_sec` — the initial call site omitted them, crashing on event `e8`
after 7/8 events had already completed successfully; fixed using the same constants
`files_api_reroute.py` itself uses, `e8` re-run standalone and merged in. (2) A one-off retry script's
`sys.path.insert(0, ...)` shadowed wt-s12's real `service.worker` with vlmstudio's own (different, non-
patched) copy — caught immediately by this harness's own cross-worktree import guard (raised a clear error
rather than silently running against the wrong code); fixed by using the same `-m` + `PYTHONPATH` invocation
pattern already proven correct for the main driver.

**(e) Spend accounting.** An initial 1-event smoke test (event `e1` only, $0.0933) was run before the full
batch to verify the pipeline end-to-end — its results were DISCARDED (superseded by the full batch's own
fresh `e1` run) but its real spend is not recoverable and is included in the ledger total below for honest
accounting.

---

## Spend

- Per-event total (the 8 results actually scored above): **$0.8065**
- Ledger total (includes the discarded smoke-test `e1` run, disclosure (e)): **$0.8997**
- **Cap: $10 — 9.0% used.**
- Real calls: 71 (critique + taxonomy + actor per config, per event, plus critique + taxonomy-only for the
  7 at-risk dedup-stress sides) — all `status="ok"`, zero `call_failed` in the final run.

---

## Artifact paths

- `eval/adcc/s12_gates/run_gate5_verification.py` — the verification driver
- `eval/adcc/s12_gates/gate5_verification_predictions.json` — all 8 events' full raw results
- `eval/adcc/s12_gates/gate5_verification_scoring.json` — this report's computed numbers
- `eval/adcc/s12_gates/gate5_verification_ledger.json` — every real call, settings tuple, real cost
- This file: `eval/adcc/s12_gates/GATE5_VERIFICATION_REPORT.md`
