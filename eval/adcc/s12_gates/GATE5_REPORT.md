# Gate 5 — Boundary-Straddle Count

Generated: 2026-07-26T01:06:02.866052+00:00 | chunk_s=720 | post_tail_s=5

Total gold (highlight-predicate) events: **557** across 258 matches (241 with >=1 highlight event, 178 multi-match / 80 single-match).

Zero Gemini/network calls. Pure arithmetic over `eval/adcc/gold_snapshot/{highlight_gold,normalized_gold}.jsonl` and the production 720s chunk constant (`service.pipelines.chunk_segment.MAX_SCOPE_SEC`).

## Headline: straddle count per setup-window W, per anchoring choice

| Anchor | W (s) | Straddle count | Fraction of gold events | Offset-sweep min/mean/max |
|---|---|---|---|---|
| production | 30 | 2 | 0.4% | 0 / 1.8 / 13 |
| production | 60 | 8 | 1.4% | 0 / 3.2 / 18 |
| production | 120 | 21 | 3.8% | 0 / 5.0 / 24 |
| naive_t0 | 30 | 11 | 2.0% | 3 / 14.1 / 30 |
| naive_t0 | 60 | 27 | 4.8% | 8 / 25.5 / 50 |
| naive_t0 | 120 | 68 | 12.2% | 27 / 47.2 / 77 |

`production` anchoring = the grid production actually applies (per-match `scope_policy.multi_match_scope`/single-match `start_s=0`, exactly mirroring `run_predictions.build_match_targets`). `naive_t0` = video-`t=0`-anchored sensitivity check only (ignores the multi-match scope-window origin).

## Straddling events — production anchoring, per W

### W=30s (2 straddling event(s))

| Match | action_class | t_s | span | boundary crossed (k, s) |
|---|---|---|---|---|
| PJ Barch vs Kade Ruotolo (6470211a3ec8dca9d1a32710) | submission_arm_lock | 727 | [697, 732] | k=1, 720s (grid_start=0) |
| Margot Ciccarelli vs Ashley Bendle (65de1567a2434704339dbc8c) | sweep | 729 | [699, 734] | k=1, 720s (grid_start=0) |

### W=60s (8 straddling event(s))

| Match | action_class | t_s | span | boundary crossed (k, s) |
|---|---|---|---|---|
| PJ Barch vs Kade Ruotolo (6470211a3ec8dca9d1a32710) | submission_arm_lock | 727 | [667, 732] | k=1, 720s (grid_start=0) |
| Joshua Cisneros vs Ethan Crelinsten (650d20c745120c8433e9cfe0) | guard_pass | 774 | [714, 779] | k=1, 720s (grid_start=0) |
| Xande Ribeiro vs Eoghan O'Flanagan (650d20c745120c8433e9cff3) | back_take | 761 | [701, 766] | k=1, 720s (grid_start=0) |
| Ffion Davies vs Beatriz Mesquita (650d20c745120c8433e9d020) | guard_pass | 760 | [700, 765] | k=1, 720s (grid_start=0) |
| Margot Ciccarelli vs Ashley Bendle (65de1567a2434704339dbc8c) | sweep | 729 | [669, 734] | k=1, 720s (grid_start=0) |
| Nia Blackman vs Salla Simola (65de222ea2434704339dbe9c) | back_take | 2235 | [2175, 2240] | k=1, 2185s (grid_start=1465) |
| Fabricio Andrey vs Kaua Gabriel (65edf42aa2434704339f7e93) | back_take | 754 | [694, 759] | k=1, 720s (grid_start=0) |
| Huaiqing Xu vs Daiki Yonekura (66b009481c62a1db7a53d504) | back_take | 30443 | [30383, 30448] | k=1, 30405s (grid_start=29685) |

### W=120s (21 straddling event(s))

| Match | action_class | t_s | span | boundary crossed (k, s) |
|---|---|---|---|---|
| PJ Barch vs Kade Ruotolo (6470211a3ec8dca9d1a32710) | submission_arm_lock | 727 | [607, 732] | k=1, 720s (grid_start=0) |
| Joshua Cisneros vs Ethan Crelinsten (650d20c745120c8433e9cfe0) | guard_pass | 774 | [654, 779] | k=1, 720s (grid_start=0) |
| Gabriel Sousa vs Ruan Alvarenga (650d20c745120c8433e9cfeb) | guard_pass | 809 | [689, 814] | k=1, 720s (grid_start=0) |
| Gabriel Sousa vs Ruan Alvarenga (650d20c745120c8433e9cfeb) | back_take | 818 | [698, 823] | k=1, 720s (grid_start=0) |
| Xande Ribeiro vs Eoghan O'Flanagan (650d20c745120c8433e9cff3) | back_take | 761 | [641, 766] | k=1, 720s (grid_start=0) |
| Rafael Lovato Jr. vs Perttu Tepponen (650d20c745120c8433e9d001) | back_take | 798 | [678, 803] | k=1, 720s (grid_start=0) |
| Rafael Lovato Jr. vs Perttu Tepponen (650d20c745120c8433e9d001) | submission_choke | 810 | [690, 815] | k=1, 720s (grid_start=0) |
| Ffion Davies vs Brianna Ste-Marie (650d20c745120c8433e9d01e) | guard_pass | 5691 | [5571, 5696] | k=1, 5619s (grid_start=4899) |
| Ffion Davies vs Beatriz Mesquita (650d20c745120c8433e9d020) | guard_pass | 760 | [640, 765] | k=1, 720s (grid_start=0) |
| Rafaela Guedes vs Eleftheria Christodolou (650d20c745120c8433e9d027) | sweep | 9075 | [8955, 9080] | k=1, 9002s (grid_start=8282) |
| Nick Rodriguez vs Andy Varela (650d20c745120c8433e9d029) | guard_pass | 794 | [674, 799] | k=1, 720s (grid_start=0) |
| Nick Rodriguez vs Andy Varela (650d20c745120c8433e9d029) | guard_pass | 809 | [689, 814] | k=1, 720s (grid_start=0) |
| Nick Rodriguez vs Andy Varela (650d20c745120c8433e9d029) | guard_pass | 822 | [702, 827] | k=1, 720s (grid_start=0) |
| Gordon Ryan vs Andre Galvao (650d20c745120c8433e9d059) | sweep | 822 | [702, 827] | k=1, 720s (grid_start=0) |
| Margot Ciccarelli vs Ashley Bendle (65de1567a2434704339dbc8c) | sweep | 729 | [609, 734] | k=1, 720s (grid_start=0) |
| Nia Blackman vs Salla Simola (65de222ea2434704339dbe9c) | back_take | 2235 | [2115, 2240] | k=1, 2185s (grid_start=1465) |
| Fabricio Andrey vs Kaua Gabriel (65edf42aa2434704339f7e93) | back_take | 754 | [634, 759] | k=1, 720s (grid_start=0) |
| Fabricio Andrey vs Kaua Gabriel (65edf42aa2434704339f7e93) | back_take | 798 | [678, 803] | k=1, 720s (grid_start=0) |
| Huaiqing Xu vs Daiki Yonekura (66b009481c62a1db7a53d504) | back_take | 30443 | [30323, 30448] | k=1, 30405s (grid_start=29685) |
| Jacob Rodriguez vs Achilles Rocha (66c53c471c62a1db7a571063) | submission_choke | 794 | [674, 799] | k=1, 720s (grid_start=0) |
| Mateusz Szczecinski vs Garry Tonon (66d75c251c7649ed30cc536d) | submission_arm_lock | 796 | [676, 801] | k=1, 720s (grid_start=0) |

## Existing prediction outputs under eval/adcc/ (accuracy-delta measurement)

The following JSON artifacts exist and COULD, in a later pass, be joined against this gate's straddling-event list to measure an accuracy delta (straddling vs non-straddling). None were loaded, parsed, or executed by this gate (zero-Gemini scope):

- `eval/adcc/sam3_scout/clips.json`
- `eval/adcc/sam3_scout/reports/template_pre_gemini.json`
