# Gate 1 Report — position+technique+validator (Arm T) vs simplified-tags (Arm S)

## Settings tuple (design §D)

| | fps | media_resolution | model | thinking | tau_t | max_validator_iterations |
|---|---|---|---|---|---|---|
| Arm S | 10 | None (LOW) | gemini-3.5-flash | off | 5.0s | n/a |
| Arm T position | 10 | None (LOW) | gemini-3.5-flash | off | 5.0s | n/a |
| Arm T technique | 10 | None (LOW) | gemini-3.5-flash | off | 5.0s | n/a |
| Arm T validator | 10 | high (prod default) | gemini-3.5-flash | off | 5.0s | 1 |

## Subset composition (design §A.2)

30 stratified core + 8 forced Gate-5 straddlers = 38 windows. Quotas: `{'guard_pass': 11, 'back_take': 7, 'sweep': 4, 'submission_choke': 4, 'submission_leg_lock': 3, 'submission_arm_lock': 1}`.

## HEADLINE verdict — 30-event stratified CORE only (Bezos condition 2)

hit_S=16, hit_T=11, hit_T-hit_S=-5 (required >= 4, re-derived from design's 4/38 rule: ceil(4/38*30)=4), fp_dup_rate_S=2.3%, fp_dup_rate_T=0.0% (ceiling +5pp) -> **verdict: NO_GO** (hit_T-hit_S=-5 < required_margin=4 (n=30)), **sign-test: p=0.0625** (n_effective=5, T-wins=0, S-wins=5, ties_excluded=25)

**COUPLED VERDICT (Bezos condition 3): `NO_GO`** — own_verdict is not PASS — coupling check does not apply.

N scored: Arm S 30/30, Arm T 30/30 (excluded call_failed/unrun — Arm S: [], Arm T: []; full list in gate1_scoring.json).

## §G.1 point 4 dual reporting — headline recomputed excluding files_api-ingested windows

files_api-ingested window(s) in this run: `['66b009481c62a1db7a53d504__w37']`.

core-30 EXCLUDING files_api windows (n=30): hit_S=16, hit_T=11, hit_T-hit_S=-5 -> **verdict: NO_GO** (hit_T-hit_S=-5 < required_margin=4 (n=30))

**VERDICT SENSITIVITY: NO-OP: the core-30 headline contains ZERO files_api-ingested windows (Gate 1's only §G.1 reroute, w37, is a gate5_straddler — already outside the core-30 by construction). own_verdict is therefore identical with/without exclusion: 'NO_GO'. Verdict does NOT flip on exclusion (trivially — there was nothing to exclude).**

## SECONDARY figure — full 38-window pool (core + 8 straddlers, informational only)

hit_S=21, hit_T=16, hit_T-hit_S=-5, fp_dup_rate_S=3.3%, fp_dup_rate_T=0.0%. NOT the pass bar — the 8 straddlers are a deliberately class-skewed, non-representative addition.
full-38 EXCLUDING files_api windows (n=37, drops `w37`): hit_S=20, hit_T=16, hit_T-hit_S=-4 (§G.1 point 4 — this is where files_api exclusion has real effect, unlike the core-30 headline above, since w37 is a full-38-only straddler).

### Core-30 — Arm S — metrics.summarize

- micro P/R/F1: {'precision': 0.37209302325581395, 'recall': 0.5333333333333333, 'f1': 0.4383561643835616}
- totals: {'tp': 16, 'fn': 14, 'fp': 27, 'gold_count': 30, 'pred_count': 43}
- fp_breakdown: {'hallucinated': 26, 'duplicate': 1}
- fn_breakdown: {'silent': 0, 'misclassified': 14, 'competed_away': 0}

### Core-30 — Arm T — metrics.summarize

- micro P/R/F1: {'precision': 0.4074074074074074, 'recall': 0.36666666666666664, 'f1': 0.38596491228070173}
- totals: {'tp': 11, 'fn': 19, 'fp': 16, 'gold_count': 30, 'pred_count': 27}
- fp_breakdown: {'hallucinated': 16, 'duplicate': 0}
- fn_breakdown: {'silent': 3, 'misclassified': 16, 'competed_away': 0}

### Core-30 — null baselines (highlight_baselines.py, same window pool)

- **whole_match**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 30, 'fp': 30, 'gold_count': 30, 'pred_count': 30}
- **uniform_span**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 30, 'fp': 60, 'gold_count': 30, 'pred_count': 60}

### Full-38 — Arm S — metrics.summarize

- micro P/R/F1: {'precision': 0.3442622950819672, 'recall': 0.5526315789473685, 'f1': 0.4242424242424242}
- totals: {'tp': 21, 'fn': 17, 'fp': 40, 'gold_count': 38, 'pred_count': 61}
- fp_breakdown: {'hallucinated': 38, 'duplicate': 2}
- fn_breakdown: {'silent': 0, 'misclassified': 17, 'competed_away': 0}

### Full-38 — Arm T — metrics.summarize

- micro P/R/F1: {'precision': 0.45714285714285713, 'recall': 0.42105263157894735, 'f1': 0.4383561643835617}
- totals: {'tp': 16, 'fn': 22, 'fp': 19, 'gold_count': 38, 'pred_count': 35}
- fp_breakdown: {'hallucinated': 19, 'duplicate': 0}
- fn_breakdown: {'silent': 3, 'misclassified': 19, 'competed_away': 0}

### Full-38 — null baselines (highlight_baselines.py, same window pool)

- **whole_match**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 38, 'fp': 38, 'gold_count': 38, 'pred_count': 38}
- **uniform_span**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 38, 'fp': 76, 'gold_count': 38, 'pred_count': 76}

## 63% baseline (footnote, continuity only)

The historically-cited 63% `action_class` @ fps=10 prose number (INS-096/097/102/103 lineage) has no backing artifact and cannot be recomputed — it is reported here for continuity ONLY, and is NOT the pass bar for this gate (design §B.1's explicit ruling; Arm S's re-run above IS the official baseline).
