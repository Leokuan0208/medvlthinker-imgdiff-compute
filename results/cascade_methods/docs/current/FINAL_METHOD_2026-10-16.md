# Final method, 2026-10-16 — the experiment round of 2026-10-15/16

**Branch** `exp-2026-10-15` (worktree `.claude/worktrees/exp-2026-10-15`). Every number below names the
artifact it came from, all under `results/cascade_methods/artifacts/`. Judge of record = **MedVLThinker-32B**
(`run_judge.py` default); cross-family judge = **MedGemma-27B-it**. "EM" = the project's lenient exact match
(`run_openvqa.py` scorer). "LLaVA recall" = LLaVA-Med v1.0.0's open-question recall, code imported verbatim.

---

## 0. The short version

1. **The final method** = Lingshu-7B **cap320** best-of-8 pools, sampled at a **single temperature T = 0.7**,
   scored by the shipped probe recipe (24 MLP probes, layers 18/20/22, class-weighted BCE), read out by
   **rank average with a vote-count tie-break**. Frozen probes: `ckpts/train/final_probe_2026-10-16/fold{0,1}`.
   *(Corrected 2026-10-18: this line first shipped a **frozen per-benchmark temperature**. Leo reversed it
   because choosing T per benchmark needs each question's dataset identity at test time, which a deployed
   system does not have; the probe itself needs no dataset label. The map is kept below as the
   **domain-calibrated variant, not shipped**. Shipped numbers, `final_eval_FINAL_cap320_T07.json`
   `macro_over_open8.all` and `final_fold1_headline_2026-10-16.json` `policies.T07.macro`:)*

   | currency | gain over greedy, both folds | image half 1 only |
   |---|---|---|
   | judge (MedVLThinker-32B, also the training-label judge) | **+0.0798 [+0.0711, +0.0887]** | **+0.0846 [+0.0729, +0.0965]** |
   | cross-family judge (MedGemma-27B) | **+0.0710 [+0.0622, +0.0797]** | **+0.0721 [+0.0601, +0.0841]** |
   | lenient EM | **+0.0121 [+0.0030, +0.0210]** | **+0.0165 [+0.0026, +0.0299]** |
   | token-F1 | +0.0232 [+0.0160, +0.0303] | +0.0288 [+0.0182, +0.0388] |
   | LLaVA-Med open recall | +0.0324 [+0.0250, +0.0396] | +0.0389 |

   Per benchmark (T = 0.7): judge 6 significant wins, 2 ties (VQA-RAD +0.0000, VQA-Med −0.0038); lenient EM
   significantly negative on Kvasir-x1, GEMeX and VQA-Med.

   The rest of §0 item 2 and the §3 table describe the **per-benchmark-temperature variant**:
2. **Its macro gain over greedy decoding is significant in every currency**, every question scored by a probe
   that never saw its image (two-fold image cross-fitting). "n/8" counts benchmarks with a positive point
   estimate, not significant wins: per benchmark the judge gives 6 wins and 2 ties (VQA-RAD, VQA-Med), and
   exact match gives 3 wins, 3 ties and 2 losses (Kvasir-x1, GEMeX) — §3. *(Corrected 2026-10-18: this line
   first said "on all eight open benchmarks", which the per-benchmark intervals do not support.)*

   | currency | gain over greedy, both folds | image half 1 only (out of sample for every choice) |
   |---|---|---|
   | judge (MedVLThinker-32B) | **+0.0897 [+0.0831, +0.0966]**, 8/8 | **+0.0936 [+0.0849, +0.1026]**, 7/8 |
   | cross-family judge (MedGemma-27B) | **+0.0783 [+0.0715, +0.0853]**, 8/8 | **+0.0811 [+0.0724, +0.0898]**, 7/8 |
   | lenient EM | **+0.0165 [+0.0107, +0.0224]**, 5/8 | **+0.0206 [+0.0120, +0.0291]**, 5/8 |
   | token-F1 | **+0.0253 [+0.0206, +0.0301]**, 8/8 | **+0.0289 [+0.0223, +0.0355]**, 7/8 |
   | LLaVA-Med open recall | **+0.0347 [+0.0297, +0.0399]**, 8/8 | **+0.0375 [+0.0304, +0.0447]**, 7/8 |

   (`final_eval_FINAL_cap320_Tmap.json`, `final_fold1_headline_2026-10-16.json`.) The September headline was
   +0.0737 judge and an EM **tie** on half the questions; cross-fitting doubles the sample and the gain is now
   significant in all five currencies.
3. **Two changes earned their place** (each under a rule fixed before the result): the per-benchmark temperature
   (+0.0091 judge / +0.0090 MedGemma on the half that played no part in choosing it) and the vote-count
   tie-break (+0.0006, free). **Three did not**: full-resolution pools, a wider probe, calibrated score averaging.
4. **Full resolution, decided on 2026-09-17, is reversed by measurement.** It helps the generator on PathVQA and
   Kvasir-x1 but costs GEMeX 4 points of greedy accuracy; the method is better at cap320 (+0.0790 vs +0.0745).
5. **Comparability.** The final method is now also scored on the full official VQA-RAD (451), SLAKE-English
   (1,061) and PathVQA (6,719) test splits (closed + open), on the official VQA-Med 2019 test (500), in the
   field's metrics, and head-to-head against WED (arXiv 2605.18313), the only published test-time-selection
   method on our exact open subsets: **on identical candidates the probe beats WED on PathVQA (+0.0843 judge)
   and SLAKE (+0.0574) in every currency including WED's own exact match, and ties on VQA-RAD.** §5.

---

## 1. What was run

| step | what | artifact |
|---|---|---|
| geometry | image sizes on all eight benchmarks: cap320 binds on 7/8; RadImageNet never | `image_geometry_all8_2026-10-15.json` |
| full-res regeneration | N=8 + greedy at max_pixels 1,003,520 on 7 benchmarks + 4 train pools, judged, re-extracted; **same-session cap320 greedy control** | `resolution_check_2026-10-15.json` |
| readout | 7 ways to combine the 24 probes, 4 currencies, pre-specified adoption rule | `readout_variants_2026-10-15.json` |
| temperature | per-benchmark T chosen by cross-fitting inside the held-out half; map frozen | `temp_crossfit_2026-10-15.json`, `frozen_temperature_map_2026-10-15.json` |
| width | hidden 1024 / 60 ep / cosine vs a same-session control | `head_final_stack_08{ctrl,wide}_2026-10-15.json` |
| cross-fitted eval | every question of every cell scored by a probe that never saw its image | `final_eval_*.json` |
| comparability cells | closed halves of VQA-RAD / PathVQA / SLAKE-en; VQA-Med 2019 official test | `/data/dan/dataset/comparability_cells/` |
| cross-family labels | MedGemma-27B on all 254,625 cap320 candidates (169,017 new, 85,608 reused) | `/data/dan/exp_2026_10/xjudge_c320/merged*.jsonl` |
| WED | baseline selector run on our identical pools | `wed_vs_probe_2026-10-16.json` |

---

## 2. Verdicts on each candidate change

| change | result | verdict |
|---|---|---|
| **per-benchmark temperature** (frozen map: GEMeX / OmniMedVQA / RadImageNet 1.0, VQA-RAD 0.2, VQA-Med 0.4, others 0.7) | map − fixed 0.7, **image half 1 (out of sample)**: judge **+0.0091 [+0.0007, +0.0182]**, MedGemma **+0.0090 [+0.0010, +0.0175]**, EM +0.0041 (tie), LLaVA recall −0.0015 (tie); all five changed benchmarks positive under both judges (`tmap_vs_t07_2026-10-16.json`) | adopted 2026-10-16, **reversed 2026-10-18** (needs the dataset label at test time) — kept as the domain-calibrated variant |
| **rank_vote readout** (ties broken by vote count) | +0.0006 judge WIN, +0.0006 EM WIN, MedGemma +0.0005 (`readout_variants_2026-10-15.json`) | **ADOPTED** (pre-specified rule) |
| full-resolution pools | method +0.0745 vs +0.0790 at cap320; per-benchmark choice of resolution −0.0023 vs cap320, tie (`select_policy_cap320_vs_fullres_T07.json`) | rejected |
| wider probe (1024, 60 ep, cosine) | +0.0021 over a same-session control, below the ~0.003 refit floor; mostly 5 SLAKE questions | rejected |
| calibrated averaging (z-score, Platt, prob) | ties under both judges (−0.0022 … +0.0027; the −0.0022 is Platt under MedGemma); z-score wins EM +0.0032 [+0.00001, +0.0073] | rejected under the pre-specified rule — backlog 0.4 closed |
| probe-weighted vote (`vote_prob`) | EM **+0.0135** WIN and F1 +0.0094 WIN, but judge −0.0040 / MedGemma −0.0058 | not adopted; it is a **currency trade**, the clearest D2 evidence so far |

**Why full resolution was reversed.** Judge currency, greedy, full resolution minus a cap320 control generated
in the same session (serving drift ≈ 0 everywhere): VQA-RAD +0.0300 (tie), SLAKE +0.0202 (tie), **PathVQA +0.0200
WIN**, **Kvasir-x1 +0.0091 WIN**, VQA-Med +0.0035, OmniMedVQA −0.0037, **GEMeX −0.0414 LOSS** (oracle@8 −0.0279).
GEMeX's chest X-rays (median 7.8 MP) go from 303 to 1,237 vision tokens and the model does worse. The 2026-09-17
decision rested on a sweep of the three benchmarks where full resolution happens to help.

---

## 3. The final method, per benchmark

`final_eval_FINAL_cap320_Tmap.json`, both folds, judge of record unless noted:

| benchmark | n | T | greedy | final | Δ judge [95% CI] | Δ MedGemma | Δ EM | Δ LLaVA recall | oracle@8 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| PathVQA-open | 3,357 | 0.7 | 0.3140 | 0.3664 | +0.0524 [+0.0389, +0.0661] | +0.0420 | +0.0185 | +0.0278 | 0.4808 |
| SLAKE-open | 645 | 0.7 | 0.7302 | 0.7876 | +0.0574 [+0.0291, +0.0868] | +0.0419 | +0.0171 | +0.0300 | 0.8791 |
| VQA-RAD-open | 200 | 0.2 | 0.4900 | 0.5150 | +0.0250 [−0.0103, +0.0628] | +0.0250 | +0.0100 | +0.0122 | 0.5650 |
| RadImageNet | 2,000 | 1.0 | 0.3210 | 0.4690 | +0.1480 [+0.1310, +0.1650] | +0.1260 | +0.0790 | +0.0838 | 0.5505 |
| Kvasir-x1 | 10,121 | 0.7 | 0.2849 | 0.3992 | +0.1143 [+0.1068, +0.1218] | +0.1048 | −0.0381 | +0.0339 | 0.4696 |
| OmniMedVQA | 8,883 | 1.0 | 0.5164 | 0.6894 | +0.1730 [+0.1647, +0.1813] | +0.1594 | +0.0643 | +0.0572 | 0.7230 |
| VQA-Med C4 | 3,663 | 0.4 | 0.0947 | 0.1007 | +0.0060 [−0.0022, +0.0139] | +0.0087 | −0.0014 | +0.0011 | 0.1873 |
| GEMeX | 8,000 | 1.0 | 0.3974 | 0.5391 | +0.1417 [+0.1321, +0.1513] | +0.1182 | −0.0174 | +0.0316 | 0.6241 |

The two benchmarks that **lost** under the shipped method (VQA-RAD, VQA-Med) now have positive point estimates,
but both are ties, and VQA-Med is −0.0005 on image half 1 (`final_fold1_headline_2026-10-16.json`). EM is
significantly negative on Kvasir-x1 [−0.0441, −0.0322] and GEMeX [−0.0251, −0.0096], whose golds are sentences:
there EM penalises correct paraphrases (judge and MedGemma both agree the picks are better).

---

## 4. Cost

Unchanged from the shipped method: N = 8 samples per question from one prefill (vLLM `n=8`), probes on states
already extracted. The temperature map changes only the sampling temperature, not the budget. Measured cost of
best-of-8 at cap320 (`bestofn_vllm_2026-09-16.json`): 5,831 → 6,589 GFLOPs (1.13× FLOP-eq), 174.0 → 476.6 ms
(2.74×), 34.3 → 123.2 J (3.59×). Keeping cap320 instead of full resolution avoids a 2.76× increase in vision
tokens (`image_geometry_all8_2026-10-15.json`). Nothing has been run as one end-to-end deployed system.

---

## 5. Comparability with published work

### 5.1 Full official test splits (closed + open), final method

`final_eval_FINAL_cap320_Tmap.json` → `full_official_splits` (judge accuracy over all questions; LLaVA-Med
protocol columns computed with its own code):

| split | n (open + closed) | judge: greedy → final | LLaVA open recall: greedy → final | LLaVA closed acc: greedy → final |
|---|---|---|---|---|
| VQA-RAD test | 451 (200 + 251) | 0.6408 → **0.6608** | 0.3852 → 0.3974 | 0.7570 → 0.7729 |
| PathVQA test | 6,719 (3,357 + 3,362) | 0.5912 → **0.6322** | 0.3065 → 0.3343 | 0.8676 → 0.8974 |
| SLAKE test (English) | 1,061 (645 + 416) | 0.7710 → **0.8002** | 0.7815 → 0.8115 | 0.7212 → 0.7163 |

Closed (yes/no) halves alone, judge, final − greedy: PathVQA +0.0298 [+0.0197, +0.0397] WIN; VQA-RAD +0.0159
(tie); SLAKE −0.0144 (tie). These closed-half gains are **in-domain, not transfer**: the cross-fitted run trains
each fold's probes on the other image half of the three closed splits and of the VQA-Med 2019 test as well
(3,366 / 3,325 such rows in folds 0 / 1). *(Corrected 2026-10-19: this sentence first said "the probe was trained
only on open answers; it transfers to yes/no without harm". A deck review found that the FINAL run's training
rows only sum to the logged 115,934 / 116,000 when the closed and VQA-Med-2019 cells are included —
`deck/review/R_results.md`, R_method.md.)*

### 5.2 Published numbers (verbatim from each paper; protocols differ — read the caveats)

| source | VQA-RAD | SLAKE | PathVQA | protocol |
|---|---|---|---|---|
| Lingshu-7B (2506.07044v5, Table I) | 67.9 | 83.1 | 61.9 | one combined open+closed accuracy; open judged by **GPT-4.1**; SLAKE = 2,094 items (appears to include Chinese) |
| Lingshu-32B (same table) | 76.5 | 89.2 | 65.9 | same |
| Qwen2.5-VL-7B (same table) | 64.5 | 67.2 | 44.1 | same |
| MedGemma-4B-IT (same table) | 72.5 | 76.4 | 48.8 | same |
| **ours, Lingshu-7B greedy** (§5.1) | 64.1 | 77.1 | 59.1 | combined; **MedVLThinker-32B** judge; SLAKE English only; cap320 |
| **ours, final method** (§5.1) | **66.1** | **80.0** | **63.2** | same as the row above |

The gap between our greedy and Lingshu's published row is a protocol gap (judge model, SLAKE language, image
resolution), not a reproduction failure; the like-for-like comparison is our two rows, same protocol.

LLaVA-Med protocol (2306.00890v1, Table 4; open = recall, closed = accuracy; **its test counts differ from ours**:
VQA-RAD 179 open / 272 closed, PathVQA 6,761):

| model | RAD open / closed | SLAKE open / closed | Path open / closed | training |
|---|---|---|---|---|
| LLaVA-Med zero-shot rows (Table 4b) | 28.23–37.84 / 60.66–61.40 | 37.71–39.73 / 49.76–54.33 | 11.34–12.30 / 49.07–54.05 | none (zero-shot) |
| LLaVA-Med (From LLaVA), fine-tuned (Table 4a) | 61.52 / 84.19 | 83.08 / 85.34 | 37.95 / 91.21 | fine-tuned on each train split |
| **ours, final method** (§5.1) | 39.74 / 77.29 | 81.15 / 71.63 | 33.43 / 89.74 | Lingshu-7B off the shelf + a probe |

VQA-Med 2019 official test (500), ImageCLEF-style strict accuracy (**re-implemented**, not the official
evaluator), `vqamed2019_by_category_2026-10-16.json`: greedy 0.346, final 0.348 (judge 0.530 → 0.538). The
2019 winners (CEUR Vol-2380, Table 3) reached 0.624 with systems trained on the category vocabularies; our gap
is largest on *organ* (strict 0.096 vs judge 0.472 — right answers in the wrong words).

### 5.3 Head-to-head: WED (arXiv 2605.18313) on identical pools

WED (Wasserstein Equilibrium Decoding) is the only published test-time selection method evaluated on our exact
open subsets (VQA-RAD and PathVQA, yes/no excluded). It was run as a selector over **the same cap320 T=0.7
pools** the probe chooses from (`src/exp_2026_10/wed/wed_select.py`, authors' code imported), so the only
difference is the selection rule. Probe picks come from the cross-fitted FINAL records (T=0.7 policy, so the
pools are identical). `wed_vs_probe_2026-10-16.json`; "WED-EM / WED-F1" are WED's own metrics, computed with
its own `canonicalize_answer` + `evaluate_answer` logic (repo `utils.py`):

| benchmark | n | currency | greedy | WED | probe | probe − WED [95% CI] | WED − greedy |
|---|---:|---|---:|---:|---:|---|---|
| PathVQA-open | 3,357 | judge | 0.3140 | 0.2821 | **0.3664** | **+0.0843 [+0.0711, +0.0978]** | −0.0319 LOSS |
| | | MedGemma-27B | 0.3441 | 0.3077 | **0.3861** | **+0.0783 [+0.0647, +0.0918]** | −0.0363 LOSS |
| | | WED-EM | 0.2601 | 0.2237 | **0.2845** | **+0.0608 [+0.0497, +0.0722]** | −0.0363 LOSS |
| | | WED-F1 | 0.3090 | 0.2671 | **0.3285** | **+0.0615 [+0.0498, +0.0730]** | −0.0419 LOSS |
| SLAKE-open | 645 | judge | 0.7302 | 0.7302 | **0.7876** | **+0.0574 [+0.0326, +0.0818]** | 0.0000 |
| | | MedGemma-27B | 0.7566 | 0.7488 | **0.7984** | **+0.0496 [+0.0216, +0.0775]** | −0.0078 |
| | | WED-EM | 0.6357 | 0.6264 | **0.6558** | **+0.0295 [+0.0048, +0.0546]** | −0.0093 |
| | | WED-F1 | 0.7226 | 0.7131 | **0.7393** | **+0.0262 [+0.0057, +0.0476]** | −0.0095 |
| VQA-RAD-open | 200 | judge | 0.4900 | 0.4850 | 0.4900 | +0.0050 [−0.0515, +0.0580] | −0.0050 |
| | | MedGemma-27B | 0.5200 | 0.5200 | 0.5300 | +0.0100 [−0.0479, +0.0663] | 0.0000 |
| | | WED-EM | 0.2600 | 0.2200 | 0.2600 | +0.0400 [−0.0053, +0.0826] | −0.0400 LOSS |
| | | WED-F1 | 0.3694 | 0.3450 | 0.3660 | +0.0210 [−0.0268, +0.0655] | −0.0245 |

**On identical candidates the probe beats WED on PathVQA and SLAKE in every currency, including WED's own
metrics; VQA-RAD (200 questions) is a tie throughout.** WED does not beat greedy decoding on any of the three
benchmarks with Lingshu-7B's pools, and is significantly worse than greedy on PathVQA. For orientation, WED's
paper reports VQA-RAD-open greedy EM 21.00 / 29.00 for Qwen3-VL-8B / MedGemma-4B (2605.18313v2, Table 3);
our Lingshu-7B greedy is 0.2600 on the same metric. **Caveat in WED's favour:** it was adapted to select from
our 8 samples; its own pipeline samples up to 16 times at T ∈ {0.5, 1.0} with its own prompts, which was not
reproduced. It fell back to the majority answer on 12/3,357 PathVQA, 55/645 SLAKE and 0/200 VQA-RAD questions
where its filters removed every candidate.

---

## 6. Caveats

1. **Judges.** Both judges see the gold answer and are text-only. MedVLThinker-32B shares Lingshu's Qwen2.5-VL
   lineage; MedGemma-27B does not, and agrees on the sign everywhere it matters.
2. **The temperature map was chosen on image half 0**; the out-of-sample evidence is image half 1 (§2, §0).
3. **EM** is lenient (containment) and penalises paraphrase on sentence-gold benchmarks; it is reported, not
   used to choose anything.
4. **Our VQA-Med cell** (3,663) is mostly the 2019 *training* split; the official test (500) is now separate (§5.2).
5. **WED** was adapted to fixed pools (our 8 samples instead of its own sampling; always-answer fallback on
   55/645 SLAKE questions where its filters removed every candidate); its own code is imported, not copied.
6. **MedEvalKit's open-question scorer has four bugs** in our local copy (survey 2026-10-15) — its open token
   metrics are not used anywhere here.
7. Nothing here is an end-to-end deployed run; cost numbers are per-component measurements (§4).
