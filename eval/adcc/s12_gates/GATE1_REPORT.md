# Gate 1 Report — position+technique+validator (Arm T) vs simplified-tags (Arm S)

## Settings tuple (design §D)

| | fps | media_resolution | model | thinking | tau_t | max_validator_iterations |
|---|---|---|---|---|---|---|
| Arm S | 10 | None (LOW) | gemini-3.1-flash-lite | off | 5.0s | n/a |
| Arm T position | 10 | None (LOW) | gemini-3.1-flash-lite | off | 5.0s | n/a |
| Arm T technique | 10 | None (LOW) | gemini-3.1-flash-lite | off | 5.0s | n/a |
| Arm T validator | 10 | high (prod default) | gemini-3.1-flash-lite | off | 5.0s | 1 |

## Subset composition (design §A.2)

30 stratified core + 8 forced Gate-5 straddlers = 38 windows. Quotas: `{'guard_pass': 11, 'back_take': 7, 'sweep': 4, 'submission_choke': 4, 'submission_leg_lock': 3, 'submission_arm_lock': 1}`.

## HEADLINE verdict — 30-event stratified CORE only (Bezos condition 2)

hit_S=13, hit_T=12, hit_T-hit_S=-1 (required >= 4, re-derived from design's 4/38 rule: ceil(4/38*30)=4), fp_dup_rate_S=6.5%, fp_dup_rate_T=0.0% (ceiling +5pp) -> **verdict: NO_GO** (hit_T-hit_S=-1 < required_margin=4 (n=30)), **sign-test: p=1.0000** (n_effective=9, T-wins=4, S-wins=5, ties_excluded=21)

**COUPLED VERDICT (Bezos condition 3): `NO_GO`** — own_verdict is not PASS — coupling check does not apply.

N scored: Arm S 30/30, Arm T 30/30 (excluded call_failed/unrun — Arm S: [], Arm T: []; full list in gate1_scoring.json).

## §G.1 point 4 dual reporting — headline recomputed excluding files_api-ingested windows

files_api-ingested window(s) in this run: `['66b009481c62a1db7a53d504__w37']`.

core-30 EXCLUDING files_api windows (n=30): hit_S=13, hit_T=12, hit_T-hit_S=-1 -> **verdict: NO_GO** (hit_T-hit_S=-1 < required_margin=4 (n=30))

**VERDICT SENSITIVITY: NO-OP: the core-30 headline contains ZERO files_api-ingested windows (Gate 1's only §G.1 reroute, w37, is a gate5_straddler — already outside the core-30 by construction). own_verdict is therefore identical with/without exclusion: 'NO_GO'. Verdict does NOT flip on exclusion (trivially — there was nothing to exclude).**

## SECONDARY figure — full 38-window pool (core + 8 straddlers, informational only)

hit_S=16, hit_T=12, hit_T-hit_S=-4, fp_dup_rate_S=4.7%, fp_dup_rate_T=0.0%. NOT the pass bar — the 8 straddlers are a deliberately class-skewed, non-representative addition.
full-38 EXCLUDING files_api windows (n=37, drops `w37`): hit_S=16, hit_T=12, hit_T-hit_S=-4 (§G.1 point 4 — this is where files_api exclusion has real effect, unlike the core-30 headline above, since w37 is a full-38-only straddler).

### Core-30 — Arm S — metrics.summarize

- micro P/R/F1: {'precision': 0.41935483870967744, 'recall': 0.43333333333333335, 'f1': 0.4262295081967213}
- totals: {'tp': 13, 'fn': 17, 'fp': 18, 'gold_count': 30, 'pred_count': 31}
- fp_breakdown: {'hallucinated': 16, 'duplicate': 2}
- fn_breakdown: {'silent': 6, 'misclassified': 11, 'competed_away': 0}

### Core-30 — Arm T — metrics.summarize

- micro P/R/F1: {'precision': 0.4444444444444444, 'recall': 0.4, 'f1': 0.4210526315789474}
- totals: {'tp': 12, 'fn': 18, 'fp': 15, 'gold_count': 30, 'pred_count': 27}
- fp_breakdown: {'hallucinated': 15, 'duplicate': 0}
- fn_breakdown: {'silent': 3, 'misclassified': 15, 'competed_away': 0}

### Core-30 — null baselines (highlight_baselines.py, same window pool)

- **whole_match**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 30, 'fp': 30, 'gold_count': 30, 'pred_count': 30}
- **uniform_span**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 30, 'fp': 60, 'gold_count': 30, 'pred_count': 60}

### Full-38 — Arm S — metrics.summarize

- micro P/R/F1: {'precision': 0.37209302325581395, 'recall': 0.42105263157894735, 'f1': 0.3950617283950617}
- totals: {'tp': 16, 'fn': 22, 'fp': 27, 'gold_count': 38, 'pred_count': 43}
- fp_breakdown: {'hallucinated': 25, 'duplicate': 2}
- fn_breakdown: {'silent': 7, 'misclassified': 15, 'competed_away': 0}

### Full-38 — Arm T — metrics.summarize

- micro P/R/F1: {'precision': 0.35294117647058826, 'recall': 0.3157894736842105, 'f1': 0.33333333333333337}
- totals: {'tp': 12, 'fn': 26, 'fp': 22, 'gold_count': 38, 'pred_count': 34}
- fp_breakdown: {'hallucinated': 22, 'duplicate': 0}
- fn_breakdown: {'silent': 4, 'misclassified': 22, 'competed_away': 0}

### Full-38 — null baselines (highlight_baselines.py, same window pool)

- **whole_match**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 38, 'fp': 38, 'gold_count': 38, 'pred_count': 38}
- **uniform_span**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 38, 'fp': 76, 'gold_count': 38, 'pred_count': 76}

## 63% baseline (footnote, continuity only)

The historically-cited 63% `action_class` @ fps=10 prose number (INS-096/097/102/103 lineage) has no backing artifact and cannot be recomputed — it is reported here for continuity ONLY, and is NOT the pass bar for this gate (design §B.1's explicit ruling; Arm S's re-run above IS the official baseline).
