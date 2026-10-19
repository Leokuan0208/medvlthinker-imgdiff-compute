# R2_numbers — round-2 number audit of the corrected deck (72 slides) and speaker script

Inputs: `out/slide_text.md`, `out/probe_verifier_report_2026-10-19_script.md`, round-1 `review/R_numbers.md`.
Artifacts were read from `results/cascade_methods/artifacts/` on exp-2026-10-15; the MIX files were read from the main checkout.
Severity: HIGH = wrong number or wrong label · MED = misleading, inconsistent or under-sourced · LOW = rounding, wording or a pointer.

## 0. Bottom line

- **All 26 round-1 findings are fixed** (§1), including all 6 HIGH and all 12 MED.
- **No HIGH findings in round 2.** There are 2 MED and 9 LOW (§3).
- **The review-computed numbers reproduce exactly.** I re-ran the producing scripts with their output redirected to `review/scr/r2_*_out.json`; the originals were not touched:
  - `r2_w1_micro.py` matches `numbers_review_part1.json`: micro, the two selection-efficiency ratios, and the t-intervals.
  - `r2_w1_pi.py` matches π and pos_weight: 0.2071 / 0.2085 and 3.83 / 3.80 on 115,934 / 116,000 rows.
  - `r2_part2_numbers.py` matches `numbers_review_part2.json`: all four blocks.
  - `r2_part3_numbers.py` matches all 27 keys of `numbers_review_part3.json` bit for bit (same seed). That covers LOBO, Bonferroni, the fold macros, headroom, the correlations, the answer-changed rate and the WED game-active subset.
  - `review/rr_rows.py` matches `training_rows_by_source`. The per-cell sums reproduce 81,331 / 81,554.
  - `w1_count.py` gives 194 cards and 186 distinct papers.
  - `wed_variants.py` gives the WED variants at 0.270–0.286 vs greedy 0.314 on PathVQA.
- **Not-shipped Tmap numbers (task 4): clean.**
  - The Tmap figures 0.0897 / 0.0783 / 0.0165 / 0.0347 appear only where labelled as not shipped: slide 46 ("FINAL_METHOD shipped it (+0.0897)… reversed"), slide 69 (backup) and script 46.
  - Other occurrences of 0.0783 and 0.0165 are different quantities that are labelled as such:
    - 0.0783 on slide 57 is the WED PathVQA MedGemma probe − WED delta.
    - +0.0165 on slides 37 and 49 and in scripts 29, 34 and 37 is the half-1 EM gain from `final_fold1_headline_2026-10-16.json`.
  - 0.0347 (Tmap LLaVA recall) appears nowhere.
- **Source lines (task 5).** Every cited artifact, key, code line and doc exists, with one exception: `DOMAIN_GUIDE_2026-09-30.md` (finding 10). Checked against code:
  - `run_openvqa.py:186` (SamplingParams) and `:51` (MIN_PX = 4·28·28 = 3,136, which resolves round-1 "could not verify").
  - `head_sweep.py:172–174` (pos_weight).
  - `final_method_eval.py:287` (std + 1e-6) and `:348–350` (vote tie-break).
  - `extract_generator_hidden.py:67` (norm = strip + lower) and `:309` (rows sorted by answer, so the double-tie goes to the alphabetically first answer, as stated).
  - The AUDIT §5 cites are correct: the layer argmax is at line 239 and the refit floor at line 245, both inside §5.

## 1. Round-1 HIGH/MED findings — status

| # | round-1 finding | status (where it now stands) |
|---|---|---|
| 1 | GEMeX answer prior −0.0425 | fixed. Slide 53, script 38 and script 53 now say +0.054 vs the probe's +0.120 (audit_stats_baselines … answer_prior_LOO_optimistic.per_benchmark.gemex_open 0.0538; probe 0.1197) |
| 2 | "validated by humans" | fixed. Slide 62 says "No human has rated this method's picks; … LLM adjudicator … hand-checked PathVQA items". Slide 63 no longer says "human". Slide 66 lists "humans validated the judge" under *do not say*. Script 11 is fixed |
| 3 | script 47 "four times the image tokens" | fixed: "2.8 times … four times on GEMeX" |
| 4 | slide 39 cost labelled as "scoring" | fixed: "generate 8 + LoRA-verify 8: 3.68× / 5.77×" and "LoRA scoring alone: 1.69× / 2.82×"; script 39 matches |
| 5 | slide 44 "6 of 8 with zero labels" | fixed: "5 of 8 beat greedy, a 6th after 50 labels" |
| 6 | 46–66% presented as general | fixed: "on RadImageNet and OmniMedVQA … 46% and 66% … (3 seeds)" |
| 7 | cap320 +0.0790 next to +0.0798 | fixed: footnote "earlier probe fit (2026-10-15) … the shipped fit gives +0.0798"; verdict now "not worse" |
| 8 | selection efficiency 0.81 vs 0.85 | fixed. Slide 8 gives both (0.81 per-benchmark mean, 0.85 ratio of the bars); script 8 explains |
| 9 | GFLOPs labelled measured | fixed: "GFLOPs (modelled, shared prefill)" and the source says "modelled" |
| 10 | 4.4 distinct (81,331 / 18,417) | fixed: 4.42 = 162,885 / 36,869 (numbers_review_part2.json, reproduced) |
| 11 | slide 46 chart/table probes | fixed: chart "September pooled probe v2", table "August pooled probe v1" |
| 12 | "September-era" on Aug-25 artifacts | fixed on slide 29 and slide 44 ("August era") |
| 13 | WED N wording | fixed and consistent on slides 22 / 57 / 60 (≤ 8 distinct, ≤ 16 draws) |
| 14 | WED +1.2 to +5.0 | fixed: "−1.2 to +5.0 pts (9 of 10 cells positive)" on slides 22 and 60 |
| 15 | answering vs grading prompt | fixed: "at layer 21 … at layer 28 it reverses (0.7895 vs 0.7922)" |
| 16 | z-score "all tie" | fixed: slide 43 and script 43 say "z-score EM +0.0032 (significant), judges tie — fails the rule" |
| 17 | probe dir | fixed: `ckpts/train/final_probe_2026-10-16/fold{0,1}` |
| 18 | AUDIT § cites | fixed (script 25 says "statistical-validity table"); the remaining §5 cites are correct (see §0) |
| 19–26 (LOW) | | all fixed: MedGemma range +0.10 to +0.15; ≈274 per question; DualRead OOD average; "highest point estimate"; Qwen run named; "7-billion"; "at most four answers"; MIX files in the main checkout |

## 2. Numbers re-verified this round (correct — not repeated below)

- **Headline and half-1 (slides 2, 49, 66; scripts 2, 49).** Checked against `final_eval_FINAL_cap320_T07.json` and `final_fold1_headline_2026-10-16.json`:
  - all macros and CIs;
  - all n_positive counts (7/8, 6/8, 4/8, 6/8, 7/8);
  - all half-1 values;
  - the 0.0089 judge − MedGemma gap.
- **Per benchmark (slides 18, 50, 52, 68).** All 8 × 6 deltas, the greedy / pick / oracle levels and the vote − greedy values.
- **Statistics slide 63 and its script.** Checked against `numbers_review_part3.json`, reproduced:
  - t-intervals;
  - LOBO ranges and 8/8, 8/8, 6/8;
  - micro values;
  - fold values 0.0747 / 0.0846.
- **Bonferroni (slide 50, script 50).** SLAKE ×40 lower bound +0.0141; MedGemma-SLAKE ×40 [−0.0046, +0.0912]; GEMeX EM ×8 [−0.0191, +0.0006].
- **Fixes vs harms (slide 51).** All rows, and the totals 4,890 / 847 / 149 / 0.73 / 0.064; VQA-Med changes its answer on 72%.
- **Headroom (slide 52).** The captured shares, macro 0.44, r = 0.92 / 0.30, VQA-Med 0.116.
- **WED (slide 57).** All 48 table cells, "7 / 6-of-7 / 0 currencies", the game-active subset, and 490 / 392 candidates ≤ 1.
- **LLaVA-Med and official splits (slides 58–59).**
  - full_official_splits values; PathVQA-closed +0.0297 [+0.0197, +0.0397].
  - Counts 179/272, 6,761, 645/416 against PAPER_NOTES_GH.
  - VQA-Med 2019: imageclef 0.346 / 0.340 and judge 0.530 / 0.538. Rows 702 / 749.
  - Lingshu Table 6 (7B and 32B).
- **Training rows (slides 26, 31, 70).** 115,934 / 116,000; 81,331 / 81,554; 3,366 / 3,325; 31,237 / 31,121 of 31,498; 6,029 questions; π.
- **Design-why slides 37–47.**
  - judge_2x2 and consensus: EM +0.0026 / +0.0101 / +0.0033, cross-judge +0.0565 / +0.0658, consensus +0.0703 / +0.0733, the 7–30% share.
  - readout_variants: every bar; vote_prob EM +0.0135 [+0.0036, +0.0237].
  - head_ens_width: L18–L22 and the ensemble; PVFIXED: +0.0182, +0.0736, +0.0007.
  - head_lobo_pooled (+0.0279, +0.0223, +0.0707, −0.0056, +0.0485); head_price +0.0196; Qwen +0.0227 → +0.0820.
  - coverage_sc16: +0.0167; wins on RadImageNet, Kvasir, OmniMed and GEMeX.
  - budget_conv: RadImageNet +0.122 / +0.154 / +0.193; VQA-Med oracle 0.199 / 0.348.
  - MIX_T07_16: +0.0984 / +0.0140. MIX8 +0.0726 is lower in all five currencies.
  - temp_crossfit +0.0045 and the EM ordering; head_temp_ensemble table (all 32 cells); tmap_vs_t07 +0.0091 / +0.0090 / +0.0041.
  - select_policy and resolution_check values; image_geometry 2.76×, GEMeX 7.8 MP and 303 tokens.
  - Sweep journals: linear 0.664, the widths, dropout, epochs, wd, depth, BCE / BT / listwise / hybrid, BUD 0.676 / 0.696, clean CV 0.680 / 0.655 (both Bradley–Terry; sd 0.016 / 0.009).
  - verifarch: all 6 values.
  - audit_stats_baselines: +0.0185 (t-interval [−0.0295, +0.0664], so "interval includes 0" holds), −0.0271, −0.0294; EM −0.0208 / −0.0307 / −0.0573 / +0.0047.
  - verifier_32b_gpu: +0.067 [+0.038, +0.095] and +0.005 [−0.023, +0.032].
  - free_head: 14.90 / 15.18; 3.81 × 1.81 = 6.90; head − LoRA +0.0162 / −0.0145.
  - CHEAP_VERIFIER: N = 4 values; retrospective N32: 0.715 / 0.720 and 0.319 / 0.343.
- **Worked example (slides 25, 32; scripts 25, 30, 32).** 0.917 / 0.583 / 0.000; 20 and 4 probes ranked first; spleen −37.429 / −7.209; Large Bowel scores in [−8.712, −1.451].
- **Paper figures spot-checked against PAPER_NOTES.**
  - Mirage: 62.2–72.5%, 69.5–87.1%, p = 0.782.
  - CASE: 0.610 / 0.542 / 0.502, 100.8 / 58.5 / r 0.75, +19.1, 0.596 / 0.537, −2.0.
  - DualRead 0.952 / 0.666.
  - MedProb App. H 0.09 → 0.13 (100 items).
  - WED −1.2 to +5.0 and the Grok judge.
  - HSRM: 88.3 / 85.2, 85.93 / 83.91, 86.1 / 82.2, 2.1–3.4 M, 15/16.
  - SWIFT: 57.5, 51.0, 1.8–3.0×10⁵, +2.7. The 52.9 is not in the notes but R_papers verified it from the PDF.
  - Q-Probe: 0.29 → 0.46 (greedy 0.38), 4,096, k = 48.
  - Brown: 82.9 → 98.44, 40.50 → 41.41. Wang: 56.5 → 74.4. Cobbe: 30× and 400.
  - Liu: 0.7192. Contamination: 4.2% / 19.8%. Lingshu Table 10: 45.2 / 69.0.

## 3. New findings (round 2)

### MED

**1. Slide 39 (and script 39): the "FLOP-eq per question, as run" rows use the HF repeat-8 generation convention. This contradicts slides 27, 35 and 62.**
- **As written.** "generate 8 + probe (separate full-resolution re-read) 14.90×", "generate 8 + LoRA verifier 15.18×", "probe with states captured during generation ≈ 8 + 0".
- **Correct value + source.**
  - In `free_head_2026-08-16.json` `DELIVERABLE_3_cost.arms_fixed_bestof8`, `bo8_head_TODAY` has `gen_passes 8.0` + `head_extraction_passes 3.8136` × 1.8095 = `flopeq_measured_geometry 14.9008`. So the 14.90 charges **8 full generation passes**.
  - `bestofn_vllm_2026-09-16.json` `forward_tokens_per_question` calls this `flopeq_as_charged 8.0`, "the HF repeat-8 convention". The pools were actually generated under vLLM with a shared prefill: `flopeq_shared 1.13`.
  - Slides 27, 35 and 62 say sampling as run = 1.13× and re-read = 6.90. On their accounting the probe as run is ≈ 1.13 + 6.90 ≈ 8.0×, not 14.90× (derived by adding the two artifacts' terms; not in one artifact).
- **Exact fix.**
  - Relabel the block "FLOP-eq per question, HF accounting (8 independent generation passes, free_head_2026-08-16.json)".
  - Change the third row to "states captured during generation: 8 (HF) / ≈ 1.13 (vLLM shared prefill) + 0".
  - Add one line: "on the vLLM path the pools were generated with, sampling is 1.13× (slide 27), so probe as run ≈ 1.13 + 6.90 (derived); the LoRA verifier cannot be served on vLLM, so its row stays HF".
  - The script's "about two percent less" (14.90 / 15.18 = 0.981) is true only under HF accounting. Add "under the same HF accounting".

**2. Slides 27, 35, 62 and scripts 27, 35, 62: the measured cost ratios come from 20 VQA-RAD questions with ≤ 32 new tokens, but are presented as general "per question" costs.**
- **As written.** "per question, batch 1, A100: latency 174.0 / 476.6 ms (2.74×), energy 34.3 / 123.2 J (3.59×)"; the source says "NVML, 20 timed calls after 3 warm-ups".
- **Correct value + source.** `bestofn_vllm_2026-09-16.json`: `harness` says "real vqa_rad open items"; `max_new` 32 (the shipped run uses 64, run_openvqa / slide 27); `n_kept` 20. The numbers are right, but they are for short VQA-RAD answers. They are not measured on the sentence-answer benchmarks (Kvasir-x1, GEMeX), where 8 sampled continuations are longer.
- **Exact fix.** Add to the slide 27 table header or source: "measured on 20 VQA-RAD open questions, ≤ 32 new tokens, one request in flight; not measured on the sentence-answer benchmarks". Add "(VQA-RAD items)" after "2.74× latency, 3.59× energy" on slide 62.

### LOW

**3. Script 11 (If asked) and script 62 (If asked): wrong Wilson interval for "50 hand-checked judge decisions".**
- **As written.** Script 62: "50 hand-checked judge decisions (error 0.04, Wilson 95% [0.0071, 0.1954])". Script 11: "judge error 0.04 among 50, Wilson upper bound ~0.20".
- **Correct value + source.**
  - In `pathvqa_judge_audit.json` `judge_validation.cells`, `direct_accept` (n 25, 1 wrong) and `direct_reject` (n 25, 1 wrong) each carry Wilson [0.0071, 0.1954]. That is the interval for **1/25**, one cell.
  - Pooled, 2/50 = 0.04 has Wilson 95% [0.011, 0.135] (computed here, z = 1.96).
  - The same block also audits 90 more judge decisions on reasoning-mode answers (`reason_accept`, `reason_reject`, `reason_reject_bothwrong_extra`), with a fair error rate up to 0.28 on `reason_reject`. The "50" is the direct-answer subset only.
- **Exact fix.** Script 62: "50 judge decisions on direct (short) answers: 2 wrong, 0.04, Wilson 95% [0.011, 0.135]; the same audit found more judge errors on long reasoning-mode answers (up to 0.28 on rejections)". Script 11: "Wilson upper bound ~0.13".

**4. Slide 43: the lower bound is quoted for one currency only.**
- **As written.** "+0.0006 judge and +0.0006 EM, both significant (lower bound +0.00005)".
- **Correct value + source.** In `readout_variants_2026-10-15.json`, `rank_vote_minus_rank_j` has CI [+0.00005, +0.00152] and `rank_vote_minus_rank_em` has CI [+0.00003, +0.00154].
- **Exact fix.** "(lower bounds +0.00005 judge, +0.00003 EM)".

**5. Slide 47: "others: ties" includes a benchmark that was not measured.**
- **As written.** In "greedy, full-res − cap320, same session", the "others" row says "ties".
- **Correct value + source.** `resolution_check_2026-10-15.json` has 7 cells and **no radimagenet_open**. The ties are SLAKE (+0.0202 [0.0000, +0.0406]), VQA-RAD, OmniMed and VQA-Med.
- **Exact fix.** "SLAKE, VQA-RAD, OmniMed, VQA-Med: ties; RadImageNet: not measured".

**6. Slide 41: "147-configuration sweep" cannot be reproduced from the cited journals.**
- **As written.** "147-configuration sweep, 5-fold CV …"; the source is `_head_sweep_journal*.jsonl`.
- **Correct value + source.** Concatenating all `_head_sweep_journal*.jsonl` gives 150 rows, 149 distinct tags and 128 distinct `_key` configurations. 147 comes from `DESIGN_EVIDENCE.md` line 20, not from the journals.
- **Exact fix.** Either say "a ~150-run sweep (128 distinct configurations)", or keep 147 and cite DESIGN_EVIDENCE.md for the count. Script 41 ("a sweep of 147 configurations") needs the same change.

**7. Script 44: the spoken ratio does not match the chart on screen.**
- **As written.** "four general domains gains about 0.02; … all eight benchmarks, about 0.07 — more than three times as much".
- **Correct value + source.** The text line is right: +0.0182 → +0.0736, 4.0× (`head_final_stack_PVFIXED_2026-09-13.json` macro.deployed_4dom_L21ish / pooled_ens). But the bar chart the audience is looking at shows +0.0279 → +0.0707, 2.5× (`head_lobo_pooled_2026-08-25.json` macro.four_domain / pooled).
- **Exact fix.** "about 0.02 to 0.03 → about 0.07 — two and a half to four times, depending on the run".

**8. Slide 35: the comparison uses the greedy GFLOPs, not the best-of-8 GFLOPs.**
- **As written.** "Negligible next to ≈ 5,831 GFLOPs (modelled) of generation."
- **Correct value + source.** 5,831 is one greedy pass. Best-of-8 generation is 6,589 (modelled), per the slide 27 table.
- **Exact fix.** "… next to ≈ 6,589 GFLOPs (modelled) of best-of-8 generation".

**9. Slide 59: the winner's team name is not in the notes.**
- **As written.** "2019 winner (Hanlin) 0.624".
- **Correct value + source.** PAPER_NOTES_GH line 496 records only "best team 0.624 acc / 0.644 BLEU [p.1, p.6]". The team name is not recorded there.
- **Exact fix.** Either confirm "Hanlin" on CEUR Vol-2380 overview p.7 and add it to the source line, or write "2019 best team".

**10. Slides 23 and 62: the domain-guide source is not on this branch.**
- **As written.** Source "DOMAIN_GUIDE_2026-09-30.md §4.1" (slide 23) and "§2.5 holes" (slide 62).
- **Correct value + source.** The file is not on branch exp-2026-10-15. It exists at `.claude/worktrees/lit-domain-package/literature/DOMAIN_GUIDE_2026-09-30.md` (branch worktree-lit-domain-package, unmerged).
- **Exact fix.** Cite it as "literature/DOMAIN_GUIDE_2026-09-30.md (branch worktree-lit-domain-package)".

**11. Slide 33 / script 33: the N = 16 MedGemma value is omitted (informational).**
- **As written.** "N = 16 with the final probes: judge +0.0984 vs +0.0798; lenient EM +0.0140 vs +0.0121 … (MedGemma labels incomplete for 16-sample pools)".
- **Correct value + source.** These are correct (`final_eval_FINAL_cap320_MIX_T07_16.json`, main checkout). The same file's MedGemma macro is +0.0674 [+0.0566, +0.0782], *below* N = 8's +0.0710, computed on incomplete labels.
- **Exact fix.** None needed on the slide. Do not say in Q&A that N = 16 is better "in every currency"; the cross-family number is not yet valid.

## 4. Script = slide (task 3)

Apart from findings 1, 3, 6 and 7 above, every number in the script matches its slide and its source. Spot-checked across all 72 sections:
- scripts 2, 7, 8, 25, 27, 29, 32, 33, 35, 37–39, 43–47, 49–53, 57, 60 and 62–66;
- the If-asked answers.

Two If-asked figures that are not on any slide were verified:
- 16.3% single-distinct questions (0.7% VQA-Med … 56.6% SLAKE), from numbers_review_part2;
- 261 / 377 overlap rows, all VQA-RAD-train images shared with VQA-Med / VQA-RAD-closed / VQA-Med-2019-test, re-derived with rr_rows.py.

## 5. Could not verify (unchanged from round 1 unless noted)

- Slide 6: ViT "32 layers · width 1280".
- Slide 14: "read in full" count.
- Slide 58: LLaVA-Med Table 4 ranges (not in PAPER_NOTES; R_papers' domain).
- Slide 59: "Hanlin" (finding 9).
- Slide 23 / script 23: "24 of 2,345 picks". `free_head_2026-08-16.json` DELIVERABLE_2 was not opened to the pick-change key.
- Resolved this round: min_pixels 3,136 (`run_openvqa.py:51`) and the alphabetical double-tie (`extract_generator_hidden.py:309`).
