# R_design — a sceptical review of Part IV ("Why each choice", slides 36–47, plus slide 29 on layers)

Reviewer stance: an ML professor who has read the artifacts, not only the slides. Every number below is
copied from the file named next to it. EXP = `.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/`;
MAIN = `~/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/`. "not measured" means I found no artifact.

---

## 0. Problems that run through the whole section (fix these first)

**S1. Nearly every verdict is in one currency, and it is the currency the probe was trained on.** The probe is
trained on MedVLThinker-32B judge labels, and the project's own standing caveat says a newly trained verifier
gains about +0.006–0.009 under that judge for free (`coadapt_verifier_T04_2026-08-14.json`). Only three Part IV
comparisons carry exact match (EM): readout (slide 43), temperature (slide 46) and probe-vs-LoRA (slide 39).
Token reading, layers, width, pooling, N and cap320 are judge-only on the slides. For cap320 and the selector the
EM numbers **exist and are not shown** (see §1 and §11).
*Fix:* add an "EM" column to the slide 37 table. Each cell holds the EM delta, or "judge only — not measured".

**S2. Eras and interval types are mixed on one axis.** Slide 38 plots the October T=0.7 probe (+0.0798, image-clustered
bootstrap, both halves) beside September baselines (one held-out half, benchmark t-intervals, df=7) on the same
scale. It also fills the dot for the answer-string prior, whose interval includes 0: [−0.0295, +0.0664]
(`audit_stats_baselines_2026-09-20.json:summary_by_currency.judge.answer_prior_LOO_optimistic.benchmark_t_interval_df7`).
"Filled dot = CI excludes 0" is therefore false for that row.

**S3. The deck ships T=0.7 (+0.0798). The project's own final-method document adopted the per-benchmark temperature
map (+0.0897).** The deck never says so (`FINAL_METHOD_2026-10-16.md` §0 item 3 and §2: "per-benchmark temperature …
ADOPTED"; CLAUDE.md §0a quotes +0.0897). A reader who has seen either document will ask why. See §10 and Q1.

**S4. The cost claims compare a design with a measurement.** "Probe ≈ 0 extra by design" sits next to the LoRA
verifier's *measured* 3.68× latency and 5.77× energy. As run, the probe needed a separate teacher-forced pass at full
resolution. The as-run FLOP figure is 14.90× (measured geometry), against 15.18× for the LoRA verifier: a saving of
about 1.9%, not "far cheaper" (`free_head_2026-08-16.json:DELIVERABLE_3_cost.arms_fixed_bestof8.bo8_head_TODAY` vs
`.bo8_incumbent_LoRA`; `CHECK_2026-09-28.md` N2). The extraction pass was never timed.

**S5. Where the design choices were selected, and what is truly out of sample.** The September choices (layers,
the fixed-T curve, readout variants, LOBO) were all read off the September held-out half. That half is image half 0:
the frozen v2 probes reproduce +0.0737 on fold 0 (`final_eval_validate_cap320_frozenv2.json:macro_over_open8.fold0`).
Image half 1 is therefore out of sample for both the September and the October choices, and it is the strongest
defence available. The T=0.7 method on half 1 alone: judge **+0.0846 [+0.0729, +0.0965]**, MedGemma +0.0721
[+0.0601, +0.0841], EM **+0.0165 [+0.0026, +0.0299]** (`final_fold1_headline_2026-10-16.json:policies.T07.macro`).
*Fix:* say this once on slide 36 or 37: "Design choices were made on image half 0. Image half 1 is out of sample for
all of them: judge +0.0846, EM +0.0165, both significant." One caveat remains. The rank-average convention dates
from August, on the July 2,345-question cells (all images) (`OPENTEXT_CORRECTIONS_2026-08-19.md` §7 item 4).

---

## 1. Train a selector at all (slide 38) — **SUPPORTED** for "a trained probe beats every untrained selector"; **OVERCLAIMED** in its conclusion

- **Tested:** majority vote, answer-string prior, random pick, pick-longest, zero-shot self-verification, zero-shot 32B verifier.
- **Isolation and controls:** probe vs vote is a clean paired test on identical pools, folds and questions, in both currencies. Judge +0.0893
  [+0.0811, +0.0972], EM +0.0136 [+0.0053, +0.0216] (`extra_numbers.json:probe_minus_vote_*`, from
  `/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json`). That is the comparison to lead with, and it is not on the slide.
- **Problems.**
  (a) The 32B result does not support "not size". The same artifact gives the pure capacity effect, 32B zero-shot minus
  7B zero-shot: **+0.067 [+0.038, +0.095], significant** (`verifier_32b_gpu.json:pooled.deltas_vs_32B.32B_minus_7B-zeroshot`;
  its `confound` field says "Pure capacity read = 32B-zeroshot vs 7B-zeroshot"). The tie with the trained 7B
  (+0.005 [−0.023, +0.032]) cannot rule out a ±3-point gap at n=600. It is also EM/substring currency, on a July
  diverse pool (5 prompts × 3 temperatures), against the July LoRA verifier rather than the probe.
  (b) The prior's dot is filled although its CI includes 0 (S2).
  (c) The baselines are judge-only, but EM exists and *strengthens* the case. September lenient-EM macros: prior −0.0208,
  random −0.0307, longest −0.0573, probe +0.0047 (`audit_stats_baselines_2026-09-20.json:summary_by_currency.lenientEM.*`).
  (d) The self-verification number (0.319 vs 0.343) comes from a retrospective table row with no named artifact
  (`PROJECT_RETROSPECTIVE_2026-07-29.md` line 1642, N32).
- **Fix (slide text):** conclusion box → "Training on labels is what makes a 7B selector work. A 4.5× larger model
  helps a zero-shot verifier (+0.067) and, untrained, only ties a trained 7B (EM/substring, n=600, July)." Add a
  line: "probe − majority vote, same pools: judge +0.0893, EM +0.0136, both significant." Draw the prior as a hollow
  dot. Put the September rows under a divider headed "September probe, image half 0, t-intervals over 8 benchmarks".
- **Slide 37 row:** "probe best" → "beats every untrained selector (judge and EM); vs WED: 2 wins, 1 tie (3 benchmarks)".

## 2. Probe on hidden states vs LoRA verifier (slide 39) — **WEAKLY SUPPORTED**

- **Tested:** the July LoRA Yes/No verifier against the **August single-layer, four-domain, Bradley–Terry head**
  `genframe_head_ens8`, on the same 2,345 questions. This is not the shipped probe (`free_head_2026-08-16.json` endpoint.selector).
- **Result:** judge +0.0162 [+0.0026, +0.0299]; EM −0.0145 [−0.0281, −0.0004]. Under EM the August probe (0.4529) is
  *below greedy* (0.4550) (`CHECK_2026-09-28.md` N3). A judge-up / EM-down split of this size matches the paraphrase-drift
  pattern the project documented. Neither result is guardrail-clean (VQA-RAD negative in both).
- **Problems.** Slide 37 says "probe: same judge accuracy, far cheaper". Both halves of that are wrong: the judge says
  the probe is better, EM says it is worse, and "far cheaper" holds only by design (S4). The slide-39 table header
  "same 2,345 questions, August" does not name the predecessor head.
- **Fix:**
  - **Slide 37:** "accuracy a wash (judge +, EM −); cheaper only once states are captured during generation".
  - **Slide 39 header:** "August single-layer probe (`genframe_head_ens8`) vs July LoRA, same 2,345 questions".
  - **Slide 39 cost table, new row:** "probe as run (separate re-read pass): 14.90× FLOP-eq vs LoRA 15.18× (−1.9%); latency not timed".
  - **Script, add:** "As we actually ran it, the probe needed its own re-reading pass, and then it costs almost the same FLOPs as the LoRA verifier. The saving is real only once the states are captured during generation. We showed that capture is equivalent only for an earlier single-layer probe."

## 3. Read the answer tokens, with the answering prompt (slide 40) — **WEAKLY SUPPORTED**

- **Tested:** span mean, last token and grading prompt (July features, layer 21). Then span + question tokens, + image
  tokens, image tokens alone (September; judge only; 3 seeds; no CI; `head_visual_features_2026-09-01.json`).
- **Problems.**
  (a) The span-vs-last numbers come from an artifact that labels itself "CONTAMINATED in-domain probe (5-fold CV INSIDE eval) …
  Optimistic upper bound" (`verifarch_hidden_promptform_indomain_2026-08-04.json:what`). The gaps are 0.015 and 0.013, with no CI.
  (b) A cleaner comparison exists and is not used: training-pool CV, span 0.68047 vs last 0.65456 (sd 0.016 / 0.009)
  (`_head_sweep_journal*.jsonl` `A_objbt_h256`, `B_L21_last`; DESIGN_EVIDENCE §2).
  (c) "The model knows more while producing an answer than when asked to grade it" is an interpretation of a 0.013 gap
  with no CI from a contaminated upper bound.
  (d) The extras rest on 3 seeds and are judge-only. The image-only arm (−0.0361) is the most informative row.
- **Fix:** slide 37 "span best" → "span ≥ last (≈1–2 sd in CV); extras ≤ +0.002 (judge, 3 seeds)". On slide 40,
  change "reading" to "consistent with the answer's own tokens carrying the signal (upper-bound CV, no CI)". Add
  the clean CV row (0.680 vs 0.655).

## 4. Layers 18/20/22 (slide 29; slide 37 row) — **WEAKLY SUPPORTED**, honestly framed

- **Tested:** single layers 18–22 and three sets (`head_ens_width_2026-08-25.json`), 5 seeds, **5 of 8 benchmarks**,
  judge only, no CI. Concatenations of layers 7/14/21/28 in training-pool CV.
- **Problems.** The set was chosen on the evaluation half: "argmax of 8 sets on the held-out halves" (`AUDIT_2026-09-18.md`
  §5). The per-layer numbers for all 8 benchmarks were withdrawn as unreproducible (`OPENTEXT_FULL_RUNDOWN_2026-09-04.md`
  §5.6). "A hedge, not a gain" is the right message, and the slide already says it.
- **Fix:** add to slide 29: "chosen on image half 0 (5 of 8 benchmarks, judge only); half 1 is out of sample (S5)".

## 5. MLP: one hidden layer, width 256 (slide 41) — **WEAKLY SUPPORTED**

- **Tested:** a 147-configuration sweep, scored by 5-fold CV inside the **four-domain, layer-21** training pool. Then one
  end-to-end check: the sweep winner (1024 wide, 60 epochs, cosine) against a same-session control, 8 seeds, 3 threads.
  It gained +0.0021, judge only, no CI (`head_final_stack_08{wide,ctrl}_2026-10-15.json:macro.pooled_ens` 0.0761 vs 0.0740).
- **Problems.**
  (a) The project documented that model selection by this CV does not transfer to eval: "CV chose grader/bce/h256 …
  scored 0.7507 on eval, while listwise (cv 0.6583) scored 0.7895" (`VERIFIER_ARCHITECTURES_2026-08-04.md` §6 item 3,
  "UNRESOLVED"). The slide does not mention this.
  (b) Slide 37 lists LayerNorm among "rest flat". In CV it was the best direction, +0.005 to +0.010 (`R2_normz_ln1`
  0.70400 vs 0.69408), and it was never tested end to end.
  (c) The learning rate was swept only for Bradley–Terry, never for BCE.
  (d) "Our own linear probe is −0.03 worse" is a CV selection-efficiency number, not an end-to-end one.
- **Fix:** slide 37 → "linear worse in CV; ≥128 flat in CV; widest re-tested end-to-end: +0.0021 (judge, noise)". Add a
  slide-41 footnote: "CV→eval model selection failed once on this split (VERIFIER_ARCHITECTURES §6.3). LayerNorm
  (+~1 sd in CV) and the BCE learning rate were never tested end-to-end."

## 6. BCE loss (slide 42) — **SUPPORTED** as framed ("no claim")

- **Good:** the slide states the row and step confound and declines to claim a win. That is the right call.
- **Missing evidence on both sides.**
  - *For BCE:* with the row confound removed, the July feature-arm control trained BCE on exactly the ranking losses'
    restricted subset. Restricting rows cost −0.0020 [−0.0116, +0.0075]; swapping the loss on the same rows cost
    **−0.0817 [−0.1042, −0.0586]** (`VERIFIER_ARCHITECTURES_2026-08-04.md` §4.3).
  - *Against BCE:* in that era, listwise scored *higher on eval* than the CV-chosen BCE head (0.7895 vs 0.7507, §6 item 3).
  - The August `genframe_head_ens8` head shown on slide 39 used Bradley–Terry.
- **Fix:** add a Q&A entry with both data points. Keep "open: a ranking loss on the final recipe has not been tested".

## 7. Rank average + vote tie-break (slide 43) — **SUPPORTED** (the best-controlled slide in Part IV)

- **What makes it strong:** the rule was fixed in advance, there are three currencies, the questions are identical, the
  CIs are paired and image-clustered, and the probe scores are fixed, so thread and refit noise cannot enter
  (`readout_variants_2026-10-15.json:adoption_rule`, `no_refit: true`).
- **Problems.**
  (a) The footer says "held-out halves". It is **one** half, image half 0: PathVQA n=1,623 (`per_benchmark.pathvqa_open.n`).
  (b) The test used the September probes (`probe_dir … genframe_head_pooled_ens_v2`), not the final ones, and the rule
  was not re-checked on half 1 or on the final probes. No other artifact contains `rank_vote_minus_rank`.
  (c) The adopted gain is +0.00059 [+0.00005, +0.00152]. The lower bound is 0.00005, so call it "significant but tiny".
  (d) The rank-average incumbent was itself picked on evaluation data in August: average ranks 0.806540 vs argsort
  0.798365 (`OPENTEXT_CORRECTIONS_2026-08-19.md` §7 item 4).
- **Fix:** footer → "September pooled probe, image half 0 (n = 18,452)". Script: "The tie-break gain is six hundredths of a
  point. It is free, and it passed a rule we wrote down first."

## 8. Pooling all 8 benchmarks + 4 domains (slide 44) — **SUPPORTED** as the main lever; **judge-only**

- **Isolation:** good. In `head_final_stack.py` the arms `deployed_4dom_L21ish` and `pooled_singlelayer` differ only in
  their training rows: same BCE recipe, same layer 20. +0.0182 → +0.0729, and the ensemble gives +0.0736
  (`head_final_stack_PVFIXED_2026-09-13.json:macro`). The result replicates on Qwen2.5-VL-7B.
- **Problems.**
  (a) Judge only and no CI (`AUDIT_2026-09-18.md` §5 lists PVFIXED and LOBO among the artifacts with no interval key).
  (b) Pooling adds in-domain *judge* labels, which is exactly where a probe learns its judge's taste. The 2×2 shows that
  7–30% of a judge-currency gain is judge-specific (`judge_2x2_2026-09-20.json`). The lever has not been measured in EM or
  MedGemma for Lingshu.
  (c) The slide shows two different four-domain baselines without explanation: +0.0279 in the bar chart
  (`head_lobo_pooled`, 5 seeds, 103,550-row era) and +0.0182 in the text box (PVFIXED).
  (d) The Kvasir-x1 LOBO fold leaks: it keeps same-source `kvasir_open` rows (`AUDIT_2026-09-18.md` §5). That makes LOBO
  optimistic. The "breadth does not transfer" conclusion survives the leak.
  (e) The price curve uses 3 seeds, below the project's ≥10-seed rule.
- **Fix:** add a footnote: "judge only, no interval; the two four-domain baselines come from different runs (5-seed LOBO run vs
  PVFIXED); Kvasir-x1 LOBO fold leaks (optimistic)". Slide 37 "pooling +0.055" → "pooling +0.055 (judge; EM not measured)".

## 9. N = 8 (slides 45, 33) — **WEAKLY SUPPORTED** (inherited, not chosen)

- **Tested:** only N=16 and N=32 (September probe, judge, image-clustered CIs; `coverage_sc16_ci_ALL_2026-09-16.json`), and
  N=16 with the final probes.
- **Problems.**
  (a) N=8 was never compared with a *smaller* N using the probe. The only smaller-N evidence points the other way: N=4 at
  T=0.4 tied N=8 at T=0.7 under the judge (+0.00159 [−0.0051, +0.0083]) and **won under EM** (+0.00825
  [+0.0015, +0.0151]) at half the generation, with the July LoRA verifier (`CHEAP_VERIFIER_ON_7B_2026-08-16.md` §4,
  `decoding_ladder_cold_2026-08-14.json`). It is not on the slide.
  (b) "16 better" is judge-only. With the final probes, MedGemma *falls* (+0.0674 vs +0.0710) because the MedGemma labels
  for the 16-sample pools are incomplete. EM is +0.0140 vs +0.0121 (point estimates; no paired CI computed;
  `final_eval_FINAL_cap320_MIX_T07_16.json`, in MAIN).
  (c) The RadImageNet N=32 cell was sampled at T=1.0 (`CHECK_2026-09-28.md`), which is not mentioned.
- **Fix:** slide 45 "so" box → "N = 8 is the project's inherited default, where cost was measured. N = 4 was never tested with the
  probe (with the July verifier it tied under the judge and won under EM at half the cost). N = 16 is +0.0167 (judge only)
  at 2× generation."

## 10. One temperature, T = 0.7 (slide 46) — **OVERCLAIMED**

- **Tested:** fixed T ∈ {0.2, 0.4, 0.7, 1.0} with the September probe on half 0 (judge and EM); a per-benchmark map,
  chosen on half 0 and tested on half 1; a mixed 4×2 policy.
- **Problems.**
  (a) "0.7 best single T" holds only under the judge. Under EM, 0.7 is third of four: 0.2 +0.0100, 0.4 +0.0095, 0.7 +0.0044,
  1.0 −0.0096 (`temp_crossfit_2026-10-15.json:macro.fixed_*_minus_greedy_em`). Under the judge, 0.7 beats 1.0 by only
  +0.0045, with no CI. The audit put the comparable gap "at the edge of the … tie-break band" (`AUDIT_2026-09-18.md` §5).
  (b) The reason for not shipping the map, "needs to know which dataset … not available at test time", conflicts with
  slide 44, which says the method "needs labelled questions from each benchmark". There is a coherent version: the pooled
  probe needs no dataset ID at inference, while the map would add one. Say that instead.
  (c) The map passed an out-of-sample test on half 1: judge +0.0091 [+0.0007, +0.0182], MedGemma +0.0090 [+0.0010, +0.0175]
  (`tmap_vs_t07_2026-10-16.json:macro.fold1_out_of_sample`), and `FINAL_METHOD_2026-10-16.md` §2 marks it **ADOPTED**.
  The deck reverses that decision silently (S3).
  (d) MIX8 vs T07 ("lower in every currency") is unpaired; no paired CI was computed. It is also confounded: the final
  probes were trained only on T=0.7 pools, so any non-0.7 policy is scored off its training distribution. The one
  matched-temperature check was August and the 4-domain probe: +0.0013 / −0.0028 (`head_temp_matched_2026-08-24.json`).
  (e) The per-benchmark table uses the pre-v2 probe (`head_temp_ensemble_2026-08-30.json`, 0.7 = +0.0773), while the
  chart uses v2 (+0.0736). It is labelled "September pooled probe", but there are two different probes.
- **Fix:**
  - **Slide 37:** → "0.7: best fixed T under the judge (EM prefers 0.2–0.4); map +0.009 out of sample, not shipped (needs a dataset ID at inference)".
  - **Slide 46 box:** "per-benchmark temperature: +0.009 under both judges on the unseen half (EM tie). It is the final method in FINAL_METHOD_2026-10-16.md (+0.0897). We ship the simpler T = 0.7 so that inference needs no dataset label."
  - **Slide 46 MIX:** "lower point estimate in every currency (no paired CI; probes were trained on T = 0.7 pools)".

## 11. cap320 vs full resolution (slide 47) — **OVERCLAIMED** ("cap320 better")

- **Isolation:** the whole method at both resolutions, cross-fitted, same reference greedy. That part is good. But
  +0.0790 [+0.0708, +0.0874] vs +0.0745 [+0.0651, +0.0835] are two vs-greedy intervals. **No paired full-res − cap320
  interval exists** in `select_policy_cap320_vs_fullres_T07.json:macro`. The intervals overlap, and the difference is 0.0046.
  The per-benchmark-choice arm is a tie: −0.0023 [−0.0087, +0.0037].
- **EM exists and is not shown:** cap320 +0.0122 vs full-res +0.0111 (same file, `*_minus_ref_greedy_em`). Also a tie.
- **Feature mismatch:** in the cap320 arm the probe reads features extracted at full resolution, so the comparison is
  not fully like-for-like. The re-extraction is in flight.
- **Script error:** "four times the image tokens" is wrong. Pixels are capped at 4×; vision tokens are **2.76×**
  (`FINAL_METHOD_2026-10-16.md` §4, `image_geometry_all8_2026-10-15.json`). The slide itself says 2.76×.
- **Minor:** this slide's cap320 arm (+0.0790) and the headline (+0.0798) come from different record files (crossfit_rankvote
  vs the FINAL T07 refit). Have an answer ready.
- **Fix:**
  - **Slide 37:** "cap320 better" → "cap320: no worse (judge +0.0790 vs +0.0745; EM +0.0122 vs +0.0111) at 2.76× fewer vision tokens".
  - **Slide 47 verdict:** "cap320 is cheaper and not worse; full resolution hurts GEMeX greedy by 4 points".
  - **Script:** "four times the image tokens" → "2.76 times the vision tokens".

## 12. Anything else — choices Part IV does not defend at all

- **Label source (which judge trains the probe):** a real design choice, with no slide. Evidence (layer 20, 3 seeds, judge-currency macros):
  - trained on MedVLThinker labels → graded A +0.0710 / B +0.0658 / EM +0.0026;
  - trained on MedGemma labels → A +0.0565 / B +0.0804 / EM +0.0101;
  - consensus labels → A +0.0703 / B +0.0733 / EM +0.0033 (`judge_2x2_2026-09-20.json`, `judge_consensus_2026-09-20.json`).
  - **Fix:** add a slide-37 row "labels: MedVLThinker-32B judge | MedGemma, consensus, soft | at noise floor; 7–30% of judge gain is judge-specific".
- **24 probes = 3 layers × 8 seeds:** the number of seeds was never swept. 8 seeds ≥ the best single seed (August,
  `CHEAP_VERIFIER_ON_7B_2026-08-16.md` §6). The CIs hold the probes fixed, so refit variance is not in any interval.
- **z-standardisation:** "+0.0075 SE" exists only as a docstring (`genframe_selector.py:15-17`), with no artifact. Under
  rule 7 it must not appear on a slide.
- **Features at 1,003,520 px while generating at cap320:** an untested mismatch on every slide (re-extraction running).

---

## 13. The 8 hardest questions about Part IV, and the honest answers

**Q1. Your final-method document adopted a per-benchmark temperature (+0.0897), and it passed your own out-of-sample test.
Why does the deck ship +0.0798? And how can "the dataset is unknown at test time" be true when slide 44 says you need labels
from every benchmark?**
A: "The map is real: +0.0091 judge [+0.0007, +0.0182] and +0.0090 MedGemma on image half 1, which played no part in
choosing it, with EM a tie (`tmap_vs_t07_2026-10-16.json`). The distinction is inference-time input. The pooled probe
scores any question without knowing its source; the map needs a dataset label on every incoming question. Training needs
labelled data per domain, but deployment then needs no routing. We ship the version with fewer inference-time
assumptions and report the map as the accuracy-max variant (`FINAL_METHOD_2026-10-16.md`)."

**Q2. Almost every comparison here uses the judge whose labels trained the probe. Which design verdicts survive exact match?**
A:
- **Measured in both currencies:**
  - probe vs majority vote: EM +0.0136 [+0.0053, +0.0216] (`extra_numbers.json`);
  - readout tie-break: EM +0.0006, significant (`readout_variants_2026-10-15.json`);
  - cap320 vs full-res: tie in EM too, +0.0122 vs +0.0111 (`select_policy_cap320_vs_fullres_T07.json`);
  - probe vs LoRA: **EM loss** −0.0145 (`free_head_2026-08-16.json`);
  - temperature: EM prefers 0.2–0.4, *not* 0.7 (`temp_crossfit_2026-10-15.json`).
- **Judge only:** token reading, layers, width, the pooling lever (EM not measured for Lingshu), N=16 at the September stage.
- **Overall:** the final T=0.7 method is significant in EM on both halves (+0.0121 [+0.0030, +0.0210]) and on half 1
  alone (+0.0165) (`final_eval_FINAL_cap320_T07.json`, `final_fold1_headline_2026-10-16.json`).

**Q3. "The probe is far cheaper than the LoRA verifier." As you actually ran it?**
A: "No. As run, the probe needed a separate teacher-forced pass at full resolution: 14.90× FLOP-eq vs the LoRA's 15.18×,
about 1.9% cheaper (`free_head_2026-08-16.json:DELIVERABLE_3_cost`; `CHECK_2026-09-28.md` N2). That pass was never timed.
The large saving needs the states to be captured during generation. We showed that capture is equivalent (24 of 2,345
picks change, 0.0000 [−0.0021, +0.0021]) only for the August single-layer head. The shipped 24-probe ensemble has not been
run that way." On accuracy: a judge win and an EM loss, against an earlier probe.

**Q4. You chose layers, readout, temperature and recipe on held-out data. Which number is out of sample for every choice?**
A: "Image half 1. Every September choice was read off image half 0 (the frozen v2 probes reproduce +0.0737 on fold 0,
`final_eval_validate_cap320_frozenv2.json`), and the October map was chosen on half 0. On half 1 alone the shipped method gives judge +0.0846
[+0.0729, +0.0965], MedGemma +0.0721, EM +0.0165 [+0.0026, +0.0299] (`final_fold1_headline_2026-10-16.json:policies.T07`).
One caveat: the rank-average convention was set in August on July cells that include half-1 images."

**Q5. Is N = 8 a measured choice? What happens at N = 4?**
A: "It is inherited. With the probe it was compared only upward: N=16 adds +0.0167 judge (4 wins, 4 ties) at twice the
generation. N=4 was never run with the probe. With the July LoRA verifier, N=4 at T=0.4 tied N=8 at T=0.7 under the judge
and won under EM (+0.00825 [+0.0015, +0.0151]) at half the generation (`CHEAP_VERIFIER_ON_7B_2026-08-16.md` §4).
Probe at N=4: not measured."

**Q6. The width and loss sweep selects by CV inside a four-domain training pool. You documented that this CV→eval selection
failed before. Why trust slides 41–42?**
A: "Only as evidence of a plateau, not of an optimum. CV picked BCE/h256 in July and it scored 0.7507 on eval, below
listwise at 0.7895 (`VERIFIER_ARCHITECTURES_2026-08-04.md` §6.3, unresolved). The one end-to-end check was the sweep's best
cell against a same-session control: +0.0021, judge only, inside the ~0.003 refit floor (`head_final_stack_08{wide,ctrl}`).
LayerNorm (~1 sd better in CV) and the BCE learning rate were never tested end-to-end. On the loss, the cleanest evidence
favours BCE: on identical restricted rows, swapping to a ranking loss cost −0.0817 [−0.1042, −0.0586] (§4.3). We still
make no claim for the final recipe."

**Q7. "Training, not size, is the active ingredient." Your own artifact says 32B zero-shot beats 7B zero-shot by 6.7 points.**
A: "Correct. Size helps a zero-shot verifier: +0.067 [+0.038, +0.095] (`verifier_32b_gpu.json`). The claim is narrower:
untrained, a 4.5× larger model only ties a trained 7B, +0.005 [−0.023, +0.032]. That is n=600, EM/substring, July pools,
and the trained side was the LoRA verifier, not the probe. A tie with that interval cannot exclude a 3-point difference."

**Q8. Is cap320 better than full resolution, or just cheaper?**
A: "Cheaper and not worse. We did not compute a paired interval for full-res minus cap320. The two vs-greedy results
are +0.0790 [+0.0708, +0.0874] and +0.0745 [+0.0651, +0.0835] under the judge, and +0.0122 vs +0.0111 under EM. Choosing
resolution per benchmark ties cap320 (−0.0023 [−0.0087, +0.0037]) (`select_policy_cap320_vs_fullres_T07.json`). The one
significant per-benchmark effect is GEMeX greedy, −0.0414 [−0.0488, −0.0341] at full resolution (`resolution_check_2026-10-15.json`).
Full resolution costs 2.76× the vision tokens. In the cap320 arm the probe still reads full-resolution features;
re-extraction at cap320 is running."

---

## 14. Verdict summary

| choice | verdict | the one-line fix |
|---|---|---|
| train a selector | SUPPORTED (conclusion OVERCLAIMED) | add probe − vote EM +0.0136; drop "not size"; hollow prior dot |
| probe vs LoRA | WEAKLY SUPPORTED | "accuracy a wash; cheaper only with capture-in-generation (as run: −1.9% FLOPs)"; name the August head |
| answer tokens | WEAKLY SUPPORTED | label the upper-bound CV; add the clean CV row; soften "knows more" |
| layers | WEAKLY SUPPORTED | "chosen on half 0, 5/8 benchmarks, judge only" |
| MLP 256 | WEAKLY SUPPORTED | disclose the CV→eval failure, LayerNorm untested, lr unswept |
| BCE | SUPPORTED (as "no claim") | add the restricted-row control and the contrary eval datum to Q&A |
| readout | SUPPORTED | "one held-out half (half 0), September probes"; gain is tiny |
| pooling | SUPPORTED, judge-only | footnote: no CI, EM not measured, two baselines, LOBO leak |
| N = 8 | WEAKLY SUPPORTED | "inherited; N=4 never tested with the probe" |
| T = 0.7 | OVERCLAIMED | "best under the judge only"; say the map was adopted in FINAL_METHOD and why it is not shipped |
| cap320 | OVERCLAIMED | "not worse, 2.76× cheaper"; fix "four times the tokens" in the script |
| label source / seeds / z-score | UNSUPPORTED on slides | add the label-source row; drop the docstring-only +0.0075 |
