# R3_numbers — round-3 number audit (72 slides + speaker script)

Inputs: `out/slide_text.md`, `out/probe_verifier_report_2026-10-19_script.md`, with `review/R2_numbers.md` as the baseline (numbers it lists as verified are not repeated).
Artifacts were read from `results/cascade_methods/artifacts/` on exp-2026-10-15. The MIX files were read from the main checkout.
New recomputation: `review/scr/r3_checks.py` → `review/scr/r3_checks_out.json`. It is read-only and writes only its own output; no numbers_review JSON was modified.
Severity: HIGH = wrong number or wrong label · MED = misleading, inconsistent or under-sourced · LOW = wording, scope or pointer.

## 0. Bottom line

- **No HIGH findings.** Every number checked this round equals its source. There are 3 MED and 8 LOW findings (§2), all about labels or consistency.
- **Not-shipped per-benchmark temperature (Tmap) numbers: clean.**
  - +0.0897, +0.0783 (as a Tmap value) and +0.0091/+0.0090 appear only on slide 46, slide 69 and in the script for 46 and 69, and every occurrence is labelled "adopted 10-16, reversed 10-18" or appears in a question about it.
  - 0.0347 appears nowhere.
  - The frozen map on slide 69 matches `frozen_temperature_map_2026-10-15.json`, and +0.0091 matches `tmap_vs_t07 … judge.macro` 0.009058.
- **Script = slide.** Every script number that corresponds to a slide number matches it, with one exception: the exact-match loss count (MED-1). Spot-checked: scripts 2, 7, 27, 28, 33, 35, 37–39, 44–47, 49–53, 57, 60, 62 and 66.
- **Resolved from R2 §5 "could not verify".** The "24 of 2,345 picks" is confirmed. Source: `free_head_2026-08-16.json` DELIVERABLE_2_equivalence.picks_and_endpoint.AR_vs_TF_same_resolution."head_only::cap320_ar__vs__head_only::cap320_tf": picks_changed 24, judge d_selected 0.0 [−0.00213, +0.00213]. The capture was at cap320 vs cap320.

## 1. Numbers newly verified this round (correct)

| item | where | source |
|---|---|---|
| Answer-prior shares: OmniMed 60%, RadImageNet 67%, GEMeX 45%, Kvasir-x1 23%. Prior below greedy on PathVQA, SLAKE and VQA-Med. Macro ~25% (0.0185/0.0737 = 0.251) | slides 38, 53; scripts 38, 53 | `audit_stats_baselines_2026-09-20.json` per_benchmark.*.{probe_minus_greedy, answer_prior_LOO_optimistic_minus_greedy}. Recomputed shares: 0.5963 / 0.6748 / 0.4496 / 0.2347 |
| Prior vs probe: +0.0944 vs +0.1583 (OmniMed), +0.0827 vs +0.1225 (RadImageNet). Probe − prior on RadImageNet +0.0398 [+0.0219, +0.0578]. RadImageNet EM tie +0.0050 [−0.0080, +0.0179]. "Probe significantly ahead on all four" holds (GEMeX [+0.056, +0.0759], Kvasir [+0.0857, +0.1043], OmniMed [+0.0561, +0.0717]) | script 38, 53 | same artifact, probe_minus_prior_ci |
| Judge agreement 92.9% on 112,770 rows; κ 0.79; positive rates 19.4% / 23.9% | scripts 11, 34, 62; slide 64 | `judge_2x2_2026-09-20.json` train_label_agreement_AB 0.92852, train_label_pos_rate A 0.19419 / B 0.23880, train_rows. κ recomputed from the marginals: 0.78991 |
| Wilson 95% for 2/50 = [0.011, 0.135] | scripts 11, 62 | recomputed: [0.01104, 0.13460] (r3_checks_out.json); equals numbers_review_part3 `judge_audit.wilson95_2_of_50` |
| 90-item disagreement sample: clear judge error 7.8% [0.0382, 0.1519]; reasoning-mode rejections up to 0.28 | script 62 | `pathvqa_judge_audit.json` disagreement_hand_labels.frac_clear_judge_error_c_only 0.0778 (Wilson [0.0382, 0.1519]); judge_validation.cells.reason_reject.error_rate_fair 0.28 |
| Label-source EM: +0.0026 / +0.0101 / +0.0033 / +0.0057 | slide 37; script 37 | `judge_2x2` macro.probe_{A,B}_graded_EM 0.00257 / 0.01010; `judge_consensus` probe_{AND,SOFT}_graded_EM 0.00329 / 0.00572 (layer 20, 3 seeds). For the label, see MED-2 |
| One layer vs three: −0.0007 | script 33 | PVFIXED pooled_ens − pooled_singlelayer (R2) |
| 2 vs 8 seeds: 0.8120 vs 0.8283; thread-count span up to 0.035 | script 33 | `OPENTEXT_CORRECTIONS_2026-08-19.md` §2 table ("control sel_eff") and §7 item 3 (0.7936 … 0.8283 → span 0.0347) |
| N = 16 EM +0.0140 vs +0.0121; judge +0.0984; MedGemma +0.0674 (incomplete labels) | slides 33, 37; script 33 | main-checkout `final_eval_FINAL_cap320_MIX_T07_16.json` 0.01400 / 0.09836 / 0.06743; `_MIX_T07` 0.01206 |
| MIX16 judge +0.0927, EM +0.0108; MIX8 +0.0726 | slide 46; script 46 | `_MIX_MIX16` 0.09272 / 0.01076; `_MIX_MIX8` 0.07263 |
| GEMeX tokens 303 / 1,237 / 9,092; 4× on GEMeX; pooled 2.76×; median 7.8 MP | slide 47; scripts 6, 47 | `image_geometry_all8_2026-10-15.json` benchmarks.gemex_open.mean_tokens 303.1 / 1236.9 / 9092.4, fullres_over_cap320 4.081, median_px 7,774,464; pooled 724.7 / 262.8 = 2.758 |
| Slide 35: ≈ 6,589 GFLOPs best-of-8 (modelled) | slide 35 | 5,831.44 (free_head null_tests.flop_model_rebuild) × 371.2 / 328.5 = 6,589.6 |
| 3.81 passes × 1.81 = 6.90 (HF); 14.90× / 15.18× | slides 35, 39, 62 | `free_head` DELIVERABLE_3_cost: 3.8136 × 1.8095 = 6.9008; bo8_head_TODAY.flopeq_measured_geometry 14.9008 |
| LoRA path, HF: 3.68× / 5.77× and 1.69× / 2.82× | slide 39; script 39 | `bestofn_latency_energy_2026-08-03.json` measured: bo8_total 1325.7 ms / 322.72 J, verify8 608.9 / 157.63, gen1 360.4 / 55.89. Harness: "HF batch-1 … real vqa_rad open items", n_kept 20, max_new 32 |
| Judge sign agreement: 7 of 8; VQA-RAD +0.0000 vs +0.0100 | scripts 11, 62 | `final_eval_FINAL_cap320_T07.json` cells.*.all.pick_minus_greedy_{judge,xjudge}. Only VQA-RAD differs: the judge delta is exactly 0 |
| Temperature table macro at 0.7 = +0.0773 (August v1); distinct 2.03 → 5.12; oracle 0.463 → 0.565 | scripts 7, 46 | `head_temp_ensemble_2026-08-30.json` (verifier genframe_head_pooled_ens) — recomputed means |
| Probe-version labels | slides 29, 36, 39, 43, 45, 46, 53, 54 | Each era label matches the artifact's verifier: free_head → genframe_head_ens8 (Aug.); head_temp_ensemble → genframe_head_pooled_ens (Aug. v1); readout_variants, temp_crossfit, template_stratified, budget_conversion_sc32 → genframe_head_pooled_ens_v2 (Sept.) |
| Template / gold-seen table (11,630 / 6,822 / 11,764 / 6,688; +0.108 … −0.039 with CIs; PathVQA 1,285 / 338) | slide 53 | `template_stratified_gain_2026-09-20.json` pooled_micro.*.{n,dA,dEM}; cells.pathvqa_open |
| Readout: probe-weighted vote "minus 0.004 and minus 0.006"; z-score EM +0.0032 (sig.) | script 43 | `readout_variants` vote_prob_minus_rank_{j,x} −0.00398 / −0.00577; z_mean_minus_rank_em 0.00322 [+0.00001, …] |
| VQA-Med EM −0.0137 [−0.0227, −0.0049]; PathVQA-closed training rows 2,246 / 2,123; VQA-RAD 200 / 251; PathVQA 6,719 | scripts 51, 58 | final_eval cells.vqamed_open, full_official_splits; numbers_review_part3 training_rows_by_source |
| EM drop OmniMed → +0.0032 [−0.0074, +0.0136]; drop RadImageNet → +0.0034 [−0.0071, +0.0136] | script 49 | numbers_review_part3 em.leave_one_benchmark_out |
| Rank ties 0.58% (September probe) | script 25 | `AUDIT_2026-09-18.md` lines 242, 263 |
| BCE controls −0.0020 [−0.0116, +0.0075] and −0.0817 [−0.1042, −0.0586]; 0.7507 vs 0.7895 | scripts 41, 42 | `VERIFIER_ARCHITECTURES_2026-08-04.md` l.217, l.335–336 |
| Width re-test +0.0021 (0.0761 vs 0.0740) | slide 41 | head_final_stack_08wide / 08ctrl pooled_ens 0.07609 / 0.07401 |
| N = 4 (LoRA era): judge +0.00159, EM +0.00825; 8 seeds ≥ best single seed | slide 45; script 37 | `CHEAP_VERIFIER_ON_7B_2026-08-16.md` §4 table (the "incumbent LoRA alone"); §6: 0.801090 vs per-seed max 0.800409 |

## 2. Findings

### MED

**MED-1. Slides 49 and 62, scripts 49 and 62: the lenient-EM loss count is 2 here and 3 elsewhere.**
- **As written.**
  - Slide 49: "lenient exact match: small gain; loses on Kvasir-x1 and VQA-Med". Script 49: "per benchmark it loses on Kvasir and VQA-Med".
  - Slide 62: "lenient EM small, loses on 2". Script 62: "Exact match loses on Kvasir and VQA-Med".
  - Slide 2, by contrast, says "loses on 3 benchmarks", and slide 50 lists Kvasir-x1, GEMeX and VQA-Med.
- **Correct value + source.** In `final_eval_FINAL_cap320_T07.json` cells.*.all.ci_pick_minus_greedy_em, three CIs exclude 0:
  - Kvasir-x1 [−0.0441, −0.0322]
  - VQA-Med [−0.0227, −0.0049]
  - GEMeX [−0.0162, −0.0021]

  After Bonferroni ×8 the GEMeX loss becomes a tie, [−0.0191, +0.0006] (numbers_review_part3). So the count is 3 unadjusted and 2 after the correction.
- **Fix.**
  - Slide 49: "loses on Kvasir-x1, VQA-Med and GEMeX (GEMeX a tie after Bonferroni ×8)".
  - Slide 62: "loses on 3 (2 after Bonferroni)".
  - Scripts 49 and 62: the same change.

**MED-2. Slide 37, label-source row, lenient-EM cell: "shipped +0.0026" reads as the shipped method's EM gain.**
- **As written.** "shipped +0.0026 · MedGemma labels +0.0101 · consensus +0.0033 · soft +0.0057"
- **Correct value + source.**
  - +0.0026 is `judge_2x2_2026-09-20.json` macro.probe_A_graded_EM. That is a September layer-20, 3-seed probe trained on MedVLThinker labels.
  - The shipped method's EM gain is +0.0121 (final_eval_FINAL_cap320_T07.json).
  - The value itself is correct; the label is ambiguous. The era note "(layer 20, 3 seeds)" appears only in the source line.
- **Fix.** "Sept. layer-20 probe: MedVLThinker labels (shipped source) +0.0026 · MedGemma labels +0.0101 · consensus +0.0033 · soft +0.0057".

**MED-3. Slide 37, decision table: most rows carry no era label, and two rows mix eras. This breaks R2 shared rule 2.**
- **As written.**
  - "training data | pooling +0.055 (no CI)" carries no era label or source. Slide 44's chart shows the August run, +0.0279 → +0.0707 (Δ +0.0428). The +0.055 is the September PVFIXED run, +0.0736 − 0.0182 = +0.0554, which is not in slide 37's source line.
  - In the "N" row, the judge cell is "16: +0.0167" (September probe, `coverage_sc16_ci_ALL_2026-09-16.json`, 16 − 8). The EM cell is "+0.0140 vs +0.0121" (October final probes). The October judge equivalent would be +0.0984 − 0.0798 = +0.0186 (derived from `final_eval_FINAL_cap320_MIX_T07_16.json` and `_MIX_T07.json`).
  - Rows for readout, temperature and label source (Sept., half A), layers (Aug., half A) and what is read (July / August CV) carry no on-row era tag. Only "(Aug. probe)" and "¹ July" are tagged.
- **Correct value + source.** The values are correct for their own artifacts. The problem is the missing era tags.
- **Fix.** Append a short era tag to each verdict cell:
  - "pooling +0.055 (Sept., no CI)", and add `head_final_stack_PVFIXED_2026-09-13.json` to the source line; or use slide 44's August +0.043 instead.
  - "16: +0.0167 (Sept.)", or replace it with the October "+0.0984 vs +0.0798", so that both cells in the row come from the October probes.
  - "(Sept.)" on the readout, temperature and label-source rows; "(Aug.)" on layers; "(July CV)" on what is read.

### LOW

**L1. Script 11 and script 62, If-asked: the audited PathVQA answers are Lingshu-32B outputs, not Lingshu-7B candidates.**
- `pathvqa_judge_audit.json` _meta.dumps lists `ckpts/openvqa/strong_lingshu/ckpt_pathvqa_open_lingshu32b.jsonl` and `…_think.jsonl`.
- The script says "on July pools", which a listener will take to mean the 7B's sampled pools.
- **Fix.** "on Lingshu-32B direct / reasoning outputs (July)".

**L2. Script 2, If-asked "Macro vs micro": "(no CI computed …)" is now stale.**
- Slides 63 and 68 show the micro CIs: judge +0.1097 [+0.1058, +0.1136], MedGemma +0.0985 [+0.0944, +0.1025], EM +0.0098 [+0.0064, +0.0131] (numbers_review_part3 *.micro_question_weighted).
- **Fix.** Quote these CIs and cite numbers_review_part3.json.

**L3. Slide 39, "cost vs greedy (HF path, measured)": the measurement scope is not stated on the slide.**
- R2 rule 4 requires the scope for slide 27 and it is applied there; this table has no equivalent.
- `bestofn_latency_energy_2026-08-03.json` harness: HF batch-1, 20 VQA-RAD open items, max_new 32. Its greedy baseline is the HF greedy call (360.4 ms), not the vLLM 174.0 ms on slide 27.
- **Fix.** Add "(HF batch 1, 20 VQA-RAD questions, ≤ 32 new tokens; greedy = HF greedy)".

**L4. Slides 38 and 53 cite different files for the same answer-prior shares.**
- Slide 38 cites numbers_review_part2.json answer_prior_share (r2w2_prior_share.py). Slide 53 cites numbers_review_part3.json (r2_part3_extra.py).
- Both exist and agree to the rounding shown.
- **Fix.** Cite one of them on both slides, for consistency.

**L5. Slide 33 / slide 37 / script 33: "at about twice the generation" for N = 16 is not a measured cost.**
- Under vLLM shared prefill, FLOPs do not double, and latency and energy for N = 16 were not measured.
- **Fix.** "twice the samples (cost not measured)".

**L6. Script 46, If-asked MIX16: "lower again" holds only for the two metrics quoted.**
- In the same files, MedGemma has MIX16 +0.0778 vs T07_16 +0.0674 — higher. Both are on incomplete labels, as the script already says.
- **Fix.** "lower under the training judge and EM (MedGemma labels incomplete, not compared)".

**L7. Slide 28 / script 28: "about 274 [image tokens] on average" is the mean over 2,345 questions from 3 benchmarks, not over all 8.**
- Source: `free_head` DELIVERABLE_3_cost.geometry.generation_cap320_per_question.M = 274.13, n = 2,345.
- The all-8 question-weighted mean at cap320 is 262.8 (`image_geometry_all8_2026-10-15.json` pooled_question_weighted.cap320).
- **Fix.** Either say "≈ 274 on 3 benchmarks", or use 263 (all 8, image_geometry).

**L8. Slides 41 and 42 carry no on-screen era label for the sweep.**
- Script 41 says the sweep used the "four-domain, layer-21 setup"; the slides do not.
- **Fix.** Add "(August, 4-domain layer-21 setup)" to the chart captions on both slides.

## 3. Could not verify (unchanged)

- Slide 6: ViT "32 layers · width 1280".
- Slide 14: "read in full".
- Slide 58: LLaVA-Med Table 4 ranges.
- Paper-side figures are R_papers' domain. No new ones were introduced since R2, apart from those already listed as verified.
