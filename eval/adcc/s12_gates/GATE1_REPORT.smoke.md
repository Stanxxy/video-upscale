# Gate 1 Report — position+technique+validator (Arm T) vs simplified-tags (Arm S)

**RUN LABEL: `smoke`** — see N-scored figures below before reading any verdict as authoritative. A label other than empty means this is NOT the full 38-window batch (e.g. a smoke-test partial run) and the verdict below is illustrative of the scoring PIPELINE only, not a real gate decision.

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

hit_S=0, hit_T=0, hit_T-hit_S=0 (required >= 4, re-derived from design's 4/38 rule: ceil(4/38*30)=4), fp_dup_rate_S=0.0%, fp_dup_rate_T=0.0% (ceiling +5pp) -> **verdict: NO_GO** (hit_T-hit_S=0 < required_margin=4 (n=30)), **sign-test: n/a (n=0 effective pairs)** (n_effective=0, T-wins=0, S-wins=0, ties_excluded=2)

**COUPLED VERDICT (Bezos condition 3): `NO_GO`** — own_verdict is not PASS — coupling check does not apply.

N scored: Arm S 2/30, Arm T 2/30 (excluded call_failed/unrun — Arm S: ['6643b593532ae8f232692d70__w2', '6580c285827346aa0335a86b__w3', '654152332d513512a36b6212__w4', '65dfbed6a2434704339de9a3__w5', '650d20c745120c8433e9d00e__w6'] ... (+23 more), Arm T: ['6643b593532ae8f232692d70__w2', '6580c285827346aa0335a86b__w3', '654152332d513512a36b6212__w4', '65dfbed6a2434704339de9a3__w5', '650d20c745120c8433e9d00e__w6'] ... (+23 more); full list in gate1_scoring.json).

## §G.1 point 4 dual reporting — headline recomputed excluding files_api-ingested windows

files_api-ingested window(s) in this run: `[]`.

core-30 EXCLUDING files_api windows (n=30): hit_S=0, hit_T=0, hit_T-hit_S=0 -> **verdict: NO_GO** (hit_T-hit_S=0 < required_margin=4 (n=30))

**VERDICT SENSITIVITY: NO-OP: the core-30 headline contains ZERO files_api-ingested windows (Gate 1's only §G.1 reroute, w37, is a gate5_straddler — already outside the core-30 by construction). own_verdict is therefore identical with/without exclusion: 'NO_GO'. Verdict does NOT flip on exclusion (trivially — there was nothing to exclude).**

## SECONDARY figure — full 38-window pool (core + 8 straddlers, informational only)

hit_S=0, hit_T=0, hit_T-hit_S=0, fp_dup_rate_S=0.0%, fp_dup_rate_T=0.0%. NOT the pass bar — the 8 straddlers are a deliberately class-skewed, non-representative addition.
full-38 EXCLUDING files_api windows (n=38, drops `w37`): hit_S=0, hit_T=0, hit_T-hit_S=0 (§G.1 point 4 — this is where files_api exclusion has real effect, unlike the core-30 headline above, since w37 is a full-38-only straddler).

### Core-30 — Arm S — metrics.summarize

- micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}
- totals: {'tp': 0, 'fn': 2, 'fp': 2, 'gold_count': 2, 'pred_count': 2}
- fp_breakdown: {'hallucinated': 2, 'duplicate': 0}
- fn_breakdown: {'silent': 0, 'misclassified': 2, 'competed_away': 0}

### Core-30 — Arm T — metrics.summarize

- micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}
- totals: {'tp': 0, 'fn': 2, 'fp': 2, 'gold_count': 2, 'pred_count': 2}
- fp_breakdown: {'hallucinated': 2, 'duplicate': 0}
- fn_breakdown: {'silent': 0, 'misclassified': 2, 'competed_away': 0}

### Core-30 — null baselines (highlight_baselines.py, same window pool)

- **whole_match**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 2, 'fp': 2, 'gold_count': 2, 'pred_count': 2}
- **uniform_span**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 2, 'fp': 4, 'gold_count': 2, 'pred_count': 4}

### Full-38 — Arm S — metrics.summarize

- micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}
- totals: {'tp': 0, 'fn': 2, 'fp': 2, 'gold_count': 2, 'pred_count': 2}
- fp_breakdown: {'hallucinated': 2, 'duplicate': 0}
- fn_breakdown: {'silent': 0, 'misclassified': 2, 'competed_away': 0}

### Full-38 — Arm T — metrics.summarize

- micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}
- totals: {'tp': 0, 'fn': 2, 'fp': 2, 'gold_count': 2, 'pred_count': 2}
- fp_breakdown: {'hallucinated': 2, 'duplicate': 0}
- fn_breakdown: {'silent': 0, 'misclassified': 2, 'competed_away': 0}

### Full-38 — null baselines (highlight_baselines.py, same window pool)

- **whole_match**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 2, 'fp': 2, 'gold_count': 2, 'pred_count': 2}
- **uniform_span**: micro P/R/F1: {'precision': 0.0, 'recall': 0.0, 'f1': None}, totals: {'tp': 0, 'fn': 2, 'fp': 4, 'gold_count': 2, 'pred_count': 4}

## 63% baseline (footnote, continuity only)

The historically-cited 63% `action_class` @ fps=10 prose number (INS-096/097/102/103 lineage) has no backing artifact and cannot be recomputed — it is reported here for continuity ONLY, and is NOT the pass bar for this gate (design §B.1's explicit ruling; Arm S's re-run above IS the official baseline).
