# R_numbers — number-accuracy audit of the 2026-10-19 professor deck + speaker script

Scope: all 71 slides (`out/slide_text.md`) and the full script (`out/probe_verifier_report_2026-10-19_script.md`).
Each number was traced to the artifact under `results/cascade_methods/artifacts/` (worktree exp-2026-10-15),
`extra_numbers.json`, `worked_example_12231.json`, the PAPER_NOTES_*, or the docs. **About 330 numbers checked
and correct** (they are not listed here). The shipped headline (+0.0798 / +0.0710 / +0.0121 and their CIs, the
per-benchmark table, fixes/harms, the WED table, the LLaVA-Med and VQA-Med/Lingshu comparisons, the readout
variants, the sweep journal, N=16/32, Qwen replication) is clean. The Tmap figures (0.0897 / 0.0783 / 0.0165)
appear only on backup slide 68, labelled "not shipped". That is correct. (The 0.0783 on slide 57 is the WED
PathVQA MedGemma delta and the 0.0165 on slide 49 is the fold-1 EM gain. Both are correct and coincide with Tmap only by chance.)

Severity: HIGH = wrong number or wrong label · MED = unsourced, mis-sourced or misleading · LOW = rounding or format.

---

## HIGH

1. **Slide 53 + script 38 (If asked) + script 53 — GEMeX answer-prior "−0.0425" is from a different era and a different prior.**
   - As written: "answer-frequency control … recovers ~25% of the gain; on GEMeX (4,315 distinct golds) it loses to greedy (−0.0425) while the probe gains". The source line is `audit_stats_baselines_2026-09-20.json`.
   - In the cited artifact, the September prior on the held-out half *beats* greedy on GEMeX: `summary_by_currency.judge.answer_prior_LOO_optimistic.per_benchmark.gemex_open = +0.0538` (the probe there is +0.1197).
   - −0.0425 comes from `free_signal_bakeoff_2026-08-21.json` / `TRANSFER_WALL_2026-08-21.md` §10. That is the August string prior on the full 8,000-question GEMeX, with the frozen four-domain head and GEMeX out of training. `AUDIT_2026-09-18.md` (baselines table) and `CHECK_2026-09-28.md` lines 834–836 already flag this sentence as "about a different probe".
   - Fix: replace it with "on GEMeX the prior gains +0.054 vs the probe's +0.120 (same halves)", or keep −0.0425 but label it "August four-domain probe, full GEMeX, GEMeX not in training — free_signal_bakeoff_2026-08-21.json". Make the same change in the script 38 If-asked answer and the script 53 text ("on GEMeX … it loses to greedy").

2. **Slide 62 ("judge validated by humans on ~11% of questions") + slide 63 ("judge vs human agreement on 5 of 8") + script 11 If-asked ("validated against humans?") — the rater is not documented as human.**
   - The source, `AUDIT_2026-09-18.md` line 233, says "Judge validated against an **independent rater** | SLAKE / VQA-RAD / PathVQA only = 11.1 % of questions". `DESIGN_EVIDENCE.md` §9 gap 20 also says "independent rater".
   - The only rater artifact on disk is `results/cascade_methods/claude_judge/verdicts_lingshu32b/` (an LLM rater, apparently Claude), not humans.
   - Fix: "judge checked against an independent (LLM) rater on 11.1% of questions (SLAKE, VQA-RAD, PathVQA)", and "independent-rater agreement on 5 of 8". In script 11, answer "No human validation; an independent LLM rater on 11.1%…". The script cites "AUDIT §5". The fact is in the baselines table at line 233.

3. **Script 47 — "four times the image tokens" contradicts slide 47 ("2.76× the vision tokens per question").**
   - Correct: pooled question-weighted 724.7 / 262.8 = 2.76× (`image_geometry_all8_2026-10-15.json` `pooled_question_weighted`). 4.08× holds for GEMeX only (`benchmarks.gemex_open.fullres_over_cap320_tokens`). The 4× is the pixel cap ratio (1,003,520 / 250,880).
   - Fix: "about 2.8 times the image tokens (4 times on GEMeX)".

4. **Slide 39 table "cost of scoring 8 answers (HF path): LoRA verifier vs greedy 3.68× / 5.77×" + script 39 ("scoring eight answers costs 3.7 times the latency and 5.8 times the energy of greedy").**
   - 3.68× = `bo8_total.lat_ms_mean` 1325.7 / `gen1` 360.4, and 5.77× = 322.72 / 55.89 J (`bestofn_latency_energy_2026-08-03.json` measured). That is the *whole* pipeline (generate 8 + verify 8) vs greedy, not the cost of scoring. Scoring alone (`verify8`) is 608.9 ms / 157.63 J = 1.69× / 2.82× of greedy.
   - Fix: relabel as "generate 8 + LoRA-verify 8, vs greedy (HF): 3.68× latency, 5.77× energy". Alternatively quote verify-only 1.69× / 2.82×. The probe row should then say "generate 8 + probe" for a like-for-like comparison: HF gen8 alone is 1.99× / 2.95× (`bestofn_vllm_2026-09-16.json` `HF_COMPARISON`).

5. **Slide 44 ("onboarding cost: a new benchmark with zero labels: +0.0196 macro (6 of 8 already beat greedy)") — wrong count.**
   - `head_price_from_lobo_2026-08-30.json`: at k = 0, 5 of 8 cells are positive (PathVQA, SLAKE, RadImageNet, Kvasir-x1, GEMeX). OmniMedVQA crosses only at k = 50 (`cells.omnimed_open.crossover_k = 50`). `summary.crossed = 6` counts crossings at *any* k.
   - Fix: "+0.0196 macro; 5 of 8 beat greedy with zero labels, a 6th after 50".

6. **Slide 44 + script 44 ("~100 labelled questions recover 46–66% of the full gain" / "roughly half to two-thirds") — a two-benchmark range presented as general.**
   - `DESIGN_EVIDENCE.md` line 208: 46.4% is RadImageNet and 65.9% is OmniMedVQA, the two cells with weak zero-shot. The same artifact gives SLAKE 0.86, GEMeX 0.72 and Kvasir 0.81 of full at k = 100. VQA-RAD and VQA-Med are not meaningful (full gain ≤ 0).
   - Fix: "on RadImageNet and OmniMedVQA, ~100 labelled questions recover 46% and 66% of the full gain (3 seeds)".

## MED

7. **Slide 47 table "cap320 +0.0790 [+0.0708, +0.0874]" sits beside the headline +0.0798 for what looks like the same configuration.**
   - It comes from `select_policy_cap320_vs_fullres_T07.json` `macro.cap320_minus_ref_greedy_judge`, which reads `final_eval_cap320_crossfit_rankvote` (an earlier probe fit on 112,770 rows). The shipped fit has 115,934 / 116,000 rows.
   - Fix: footnote it as "earlier probe fit (2026-10-15), same protocol; the shipped fit gives +0.0798". Also, "cap320 better" (slide 37, green) and "slightly worse" (script 47) are point estimates. No paired cap320 − fullres CI exists in the artifact. Only selected − cap320 (−0.0023, tie) does. Mark the verdict as a point estimate or as "cheaper, not worse".

8. **Script 8 — "Greedy … 0.394 … oracle 0.559 … probe 0.473 … The ratio of picked to oracle … is 0.81."**
   - 0.473 / 0.559 = 0.847. The 0.81 is the mean of the per-benchmark ratios (0.8077). The audience can do the arithmetic. Slide 8 has the same issue.
   - Fix: say "averaged benchmark by benchmark, selection efficiency is 0.81", and define it that way on slide 8 ("mean over benchmarks of pick ÷ oracle@8").

9. **Slide 27 / 35 — "GFLOPs (vLLM, shared prefill) 5,831 | 6,589", with the source line "(measured: NVML…)".**
   - The GFLOPs are not measured. 5,831 is the analytic FLOP model at a June operating point (`free_head_2026-08-16.json` `null_tests.flop_model_rebuild.rebuilt.TOTAL` 5831.44). 6,589 = 5,831 × 371.2 / 328.5 forward tokens (derived). `bestofn_vllm_2026-09-16.json` holds only `flopeq_shared 1.13`.
   - Fix: label the GFLOPs row "modelled (FLOP model × forward-token ratio)". Keep "measured" for ms and J only.

10. **Slide 27 source — "≈4.4 distinct = 81,331 rows / 18,417 questions (recipe.json)".**
    - Neither number is in recipe.json. 81,331 = `training_rows` 112,770 − `original_domain_rows` 31,439, and 18,417 = 36,869 − 18,452. Both are derived, and both describe the September probe's training half, not the shipped T07 pools.
    - Slide 7's chart shows ≈4.17 at T = 0.7 (`head_temp_ensemble` macro of `mean_distinct`).
    - Fix: state the derivation, or recompute from `final_eval_FINAL_cap320_T07_records.json`.

11. **Slide 46 — the chart and the table come from different probes, and the label is wrong.**
    - The chart: `temp_crossfit_2026-10-15.json` (probe_dir `genframe_head_pooled_ens_v2`, T = 0.7 macro +0.0736).
    - The per-benchmark table: `head_temp_ensemble_2026-08-30.json`, whose `verifier = genframe_head_pooled_ens` (the August v1 probe, macro at 0.7 = +0.0773).
    - The source line calls both "September pooled probe".
    - Fix: label the table "August pooled probe (v1), held-out halves", or rebuild it from `temp_crossfit...per_benchmark.*.fixed_*_minus_greedy_judge`.

12. **Slide 29 ("head_ens_width_2026-08-25.json … September-era pooled probe") and slide 44 ("September era, head_lobo_pooled_2026-08-25.json").**
    - Both artifacts are dated 2026-08-25 (August).
    - Fix: "August-era pooled probe".

13. **Slide 22 ("sample up to 8 candidates at T ∈ {0.5, 1.0}") vs slide 57 ("its own pipeline samples up to 16 at T ∈ {0.5, 1.0}") vs slide 60 ("≤ 8").**
    - PAPER_NOTES_DE line 350 says "up to 8 candidates" (p.5), and line 429 says "up to 16 calls".
    - Fix: make all three say "up to 8 candidates (up to 16 sampling calls) at T ∈ {0.5, 1.0}".

14. **Slide 22 ("judge gains +1.2 to +5.0 pts over greedy on 2–8B VLMs") + script 22 ("one to five points over greedy").**
    - WED Table 1 includes Qwen-8B VQA-RAD −1.2 (PAPER_NOTES_DE line 400). Slide 60 states it, but slide 22 hides it.
    - Fix: "−1.2 to +5.0 pts (9 of 10 cells positive)".

15. **Slide 40 reading ("the model 'knows' more while producing an answer than when asked to grade it").**
    - True at layer 21 (span 0.7956 vs grading 0.7827). Reversed at layer 28 (grader span 0.7922 > generator span 0.7895, `verifarch_hidden_promptform_indomain_2026-08-04.json` table).
    - Fix: "at layer 21".

16. **Slide 43 / script 43 ("logit / z-score / probability / Platt averaging … all tie").**
    - z-score averaging is a significant EM win: `z_mean_minus_rank_em` +0.0032, verdict WIN (`readout_variants_2026-10-15.json`). It ties only under both judges.
    - Fix: "tie under both judges (z-score: +0.0032 EM, significant, but fails the rule)".

17. **Slide 30 source ("ckpts/train/genframe_head_pooled_ens_v2 … head_L18_seed0.pt shapes") and slide 31 "recipe.json".**
    - That is the September probe directory. The shipped probes are `ckpts/train/final_probe_2026-10-16/fold{0,1}`. They have the same file size (3,674,659 B), so the shape is the same, and the parameter count is correct.
    - Fix: cite the final_probe dir.

18. **Script 11 If-asked cites "AUDIT_2026-09-18.md §5" for the 11%, and script 25 If-asked cites "§5" for the 0.58% tie rate.**
    - Both facts sit in the baselines / statistical-validity block (lines 233 and 242). Check the section number before quoting.

## LOW

19. **Slide 50 ("+0.11 to +0.16, both judges").** MedGemma gives Kvasir +0.1048 and GEMeX +0.1072, so its range is +0.10 to +0.15. Fix: "+0.11 to +0.16 judge, +0.10 to +0.15 MedGemma".
20. **Slide 28 ("≈ 274 image / ≈ 5 answer tokens over 8,943 candidates").** 274.1 is per *question* (n = 2,345, `DELIVERABLE_3_cost.geometry.generation_cap320_per_question.M`). Per candidate row (n = 8,943) it is 277.3 image and 5.28 answer tokens.
21. **Slide 21 ("DualRead … SLAKE AUROC 0.952 in-distribution, 0.666 out of distribution").** 0.666 is the OOD *average* (Qwen3-VL-2B), not SLAKE (PAPER_NOTES_I line 620).
22. **Slide 37 ("0.7 best single T").** It is a point estimate: 0.7 beats 1.0 by +0.0045 (temp_crossfit) / +0.0063 (head_temp_ensemble), and AUDIT says this is at the edge of the tie band. Fix: "0.7 highest point estimate".
23. **Slide 44 vs slide 54 — Qwen "+0.0820" vs "+0.0814".** Both are correct but different runs: `head_final_stack_qwen` pooled_ens vs the single-layer-20 refit in `replication_currency`. Add the run name to one of them.
24. **Script 30 ("the 8-billion-parameter model").** Not in any artifact (Lingshu-7B / Qwen2.5-VL-7B ≈ 8.3B per model card). Either source it or say "the 7B model".
25. **Script 33 ("with four samples every probe scores four answers").** Probes score *distinct* answers (≤ N). This is a wording issue, not a number error.
26. **Slide 33 source `final_eval_FINAL_cap320_MIX_*.json`.** These summary JSONs exist only in the main checkout (`/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/`), not on branch exp-2026-10-15. Copy them in or note the path.

---

## Could not verify

- Slide 69: "vLLM 0.9.0.1", "min_pixels 3,136", HF snapshot "b98aecd4…" (run_openvqa.py not inspected).
- Slide 14: "34 core papers read in full" (194 catalogued matches memory; 34 not traced).
- Slide 56: "71.9 … (boxed MCQ prompt)". The 0.7192 is in PAPER_NOTES_GH; the prompt description is not.
- Slide 6: ViT "32 layers · width 1280" (Qwen2.5-VL Table 1). Not in the notes I searched; it is standard but unchecked here.
- Slide 21: MedProb "probe beats prompting by >20 points 'in several cases'"; slide 70 "Kvasir-VQA only in Lingshu-RL", "deduplicated against MedEvalKit test sets (not VQA-Med 2019)".
- Slide 53: assumed `dA` in `template_stratified_gain_2026-09-20.json` = judge of record (not documented in the file).
- Slide 46: "0.7 was the project default before this curve was measured" (historical claim, not checked).
- Slide 60: Mirage grader "Gemini-2.5" (PAPER_NOTES_BC line 650 names Gemini-2.5-Flash in a nearby section; not confirmed for Mirage).
