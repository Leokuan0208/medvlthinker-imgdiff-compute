# Design evidence for the final method — every choice, what it was tested against, and why

Compiled 2026-10-18 for the presentation deck. Read-only research over both artifact trees:
`MAIN` = `/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/` and
`EXP` = `/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/`
(artifacts under `artifacts/`, docs under `docs/current/`). Every number below is copied from the named file
and key (or the named doc section where the artifact has no matching key). Nothing is estimated. Where a
choice was never tested, the row says **not tested**.

---

## 0. How to read the tables

**Eras / probes** (a number from one era must never be presented as a property of another):

| tag | what | evaluated on |
|---|---|---|
| **JUL-LoRA** | July generative verifier: Lingshu-7B + LoRA emitting Yes/No (`ckpts/train/lora_verifier_pooled4` / `_disjoint`) | 2,345 open questions, 3 benchmarks (SLAKE/VQA-RAD/PathVQA-1,500) |
| **AUG-4dom** | August probe on frozen hidden states: four July training domains, layer 21, Bradley–Terry, 8 seeds (`genframe_head_ens8`) | same 2,345, or the 8 full benchmarks |
| **SWEEP** | `_head_sweep_journal*.jsonl` (MAIN): 147 configurations, four-domain pool, layer 21, 5-fold image-grouped CV **inside the training pool**; metric `cv_sel_eff` (selection efficiency = selected/oracle@N), sd across folds ≈ 0.007–0.02 | training-pool CV, not the evaluation benchmarks |
| **SEP-pooled** | September shipped probe `genframe_head_pooled_ens_v2`: 24 MLPs (L18/20/22 × 8 seeds), BCE, 112,770 rows, T=0.7 pools, rank average | one held-out image half, 18,452 questions |
| **OCT-final** | final method: same recipe refit per fold (`ckpts/train/final_probe_2026-10-16/fold{0,1}`), frozen per-benchmark T, rank average + vote tie-break, two-fold image cross-fitting | all 36,869 questions of the 8 open benchmarks |

**Currencies:** **J** = judge of record MedVLThinker-32B (text-only, sees the gold); **MG** = cross-family judge
MedGemma-27B-it; **EM** = lenient exact match; **F1** = token-F1; **LLaVA** = LLaVA-Med open recall; **SE** =
selection efficiency (selected / oracle@N — *not* share of headroom). "n/8" = benchmarks with a positive
**point estimate**, not significant wins. Default CI = paired bootstrap, 10,000 resamples, clustered by image
within benchmark, unless the row says otherwise.

**Two facts to correct in the brief before they reach a slide:**
- The final probes were **not** trained on 112,770 rows. That is the September probe. Final fold 0 trained on
  **115,934** rows and fold 1 on **116,000** (EXP `final_eval_FINAL_cap320_Tmap.json:folds.0.train_rows`,
  `folds.1.train_rows`).
- The final probes were fitted on **T=0.7** pools and on features extracted at **max_pixels 1,003,520**
  (`legacy_cap320 = True`; `AUDIT_2026-09-18.md` §4), not at the generation resolution or the frozen
  temperatures. Both are runs still in flight (DOMAIN_GUIDE §2.6 items i and iii).

---

## 1. Verifier family — why a trained probe on frozen hidden states

| alternative | measured | currency | CI | era | verdict | source file:key |
|---|---|---|---|---|---|---|
| **final method (probe)** vs greedy | **+0.0897** | J | [+0.0831, +0.0966], 8/8 | OCT-final | — | EXP `final_eval_FINAL_cap320_Tmap.json:macro_over_open8.all.pick_minus_greedy_judge` |
| same | +0.0783 / +0.0165 / +0.0253 / +0.0347 | MG / EM / F1 / LLaVA | [+0.0715,+0.0853] / [+0.0107,+0.0224] / [+0.0206,+0.0301] / [+0.0297,+0.0399] | OCT-final | — | same file, `pick_minus_greedy_{xjudge,em,f1,llava_recall}` |
| **majority vote (self-consistency)** vs greedy, final pools | −0.0070 | J | [−0.0111, −0.0029] LOSS | OCT-final pools | rejected | same file, `macro_over_open8.all.sc_minus_greedy_judge` |
| same | −0.0067 / −0.0014 | MG / EM | [−0.0110,−0.0023] LOSS / [−0.0050,+0.0024] TIE | OCT-final pools | rejected | `sc_minus_greedy_xjudge`, `sc_minus_greedy_em` |
| majority vote minus probe | −0.0895 / −0.0124 | J / EM | [−0.1017,−0.0774] / [−0.0244,−0.0011], both LOSS | SEP-pooled | rejected | EXP `readout_variants_2026-10-15.json:macro.sc_minus_rank_j`, `sc_minus_rank_em` |
| majority vote as N grows (fixed 225 Kvasir q) | 0.0933 (N=2) → 0.0800 (N=16) vs greedy 0.1244; oracle 0.1558 → 0.4178 | J | none | AUG | flat at random floor | `TRANSFER_WALL_2026-08-21.md` §4 |
| **answer-string prior** (counter over answer strings; no image, no hidden state) | macro +0.0185 vs greedy (probe +0.0737 on the same halves) | J | benchmark t-interval [−0.0295, +0.0664] | SEP-pooled | recovers ~25% of the gain; rejected | EXP `audit_stats_baselines_2026-09-20.json:summary_by_currency.judge.answer_prior_LOO_optimistic.macro` |
| answer prior on GEMeX (least memorisable) | prior −0.0425 vs greedy; head +0.0148; head − prior +0.0573 | J | head [+0.0057,+0.0238]; head−prior [+0.0483,+0.0663] | AUG-4dom | probe does verification, not memorisation | `TRANSFER_WALL_2026-08-21.md` §10 (from `free_signal_bakeoff_2026-08-21.json`) |
| random pick / pick-longest | −0.0271 / −0.0294 | J | — | SEP-pooled | controls | EXP `audit_stats_baselines_2026-09-20.json:summary_by_currency.judge.random_pick.macro`, `.longest.macro` |
| **mean log-prob / perplexity / P(True)** as selector | **not tested** — uncomputable: dumps store one scalar per question, not per candidate | — | — | — | gap | `AUDIT_2026-09-18.md` §5 |
| **semantic entropy / semantic consistency** as selector | **not tested** as a best-of-N selector. (In July it was tested only as an *error-detection* signal: AUROC 0.807 vs confidence 0.866 — a different task.) A similarity-weighted vote (`soft_vote`) exists only inside the July 39-feature model below | — | — | JUL | gap | `PROJECT_RETROSPECTIVE_2026-07-29.md` line 382 |
| 39-feature pool-structure model (consensus + prior, zero model calls) | SE 0.7636 vs LoRA 0.7752: −0.0116 | J (SE) | [−0.0334, +0.0103] TIE; PathVQA −0.0400 (guardrail dirty) | JUL | rejected | `VERIFIER_ARCHITECTURES_2026-08-04.md` §3 row B1 |
| **zero-shot self-verification** (Lingshu-7B asked "is this correct?", P(Yes)) | SE 0.7058 vs trained LoRA 0.7752: −0.0695 | J (SE) | [−0.0906, −0.0484] | JUL | rejected | `VERIFIER_ARCHITECTURES_2026-08-04.md` §3 (control row) |
| zero-shot 7B verifier, other pool | sel_acc 0.4133 vs greedy 0.44 (below greedy) | EM/substring | — | JUL, n=600 | rejected | MAIN `verifier_32b_gpu.json:pooled.verifiers.7B-zeroshot.sel_acc`, `pooled.greedy` |
| **32B as verifier** (zero-shot Lingshu-32B) vs trained 7B LoRA verifier | 0.480 vs 0.475: +0.005 | EM/substring | [−0.0233, +0.0317] n.s. | JUL, n=600 (VQA-RAD, SLAKE, PMC content) | 4.5× larger model ties a trained 7B → rejected | MAIN `verifier_32b_gpu.json:pooled.deltas_vs_32B.32B_minus_7B-trained` |
| zero-shot CLIP-family alignment (SigLIP / PubMedCLIP / BiomedCLIP) | SE 0.6247–0.6608, below random pick 0.6763 | J (SE) | — | JUL | rejected | `VERIFIER_ARCHITECTURES_2026-08-04.md` §3 rows D4–D7 |
| learning-to-rank losses on the LoRA-era score function | best +0.0014 [−0.0061, +0.0089]; worst −0.1117 | J (SE) | as shown | JUL | rejected | `VERIFIER_ARCHITECTURES_2026-08-04.md` §3 rows A1, A3 |
| **July LoRA generative verifier** vs August probe (same 2,345 q) | probe − LoRA: **+0.0162** | J | [+0.0026, +0.0299] WIN | AUG-4dom vs JUL-LoRA | — | MAIN/EXP `free_head_2026-08-16.json:DELIVERABLE_4_head_only.head_vs_incumbent.judge.d_selected` |
| same comparison | probe − LoRA: **−0.0145** | **EM** | [−0.0281, −0.0004] **LOSS** | AUG-4dom vs JUL-LoRA | ⚠️ the "probe beats LoRA" claim is judge-only | same key, `.em.d_selected`; `CHECK_2026-09-28.md` line 656 |
| LoRA ⊕ probe rank fusion | SE 0.7752 → 0.8065: +0.0313 | J (SE) | [+0.0163, +0.0463] | JUL+AUG | was the August deployable; superseded by pooling | `VERIFIER_ARCHITECTURES_2026-08-04.md` §3 row C1 |
| **generator LoRA-SFT** (fine-tune the 7B, greedy) | +0.0142 at 1× inference; +0.0374 combined with best-of-8 + LoRA verifier | currency not stated in the doc | none | JUL-era, 3 benchmarks | not compared against the final method | `AUDIT_2026-09-18.md` §5 (from `ckpts/cheapleg/scores_{base7b,adapt7b_s0}/score_summary.json`) |
| **7B + probe vs Lingshu-32B greedy** (4 benchmarks) | probe − 32B: SLAKE −0.0303, VQA-RAD −0.1134, PathVQA +0.0357 (700 matched), RadImageNet +0.1484; macro +0.0101 (J), −0.0305 (EM) | J (EM in brackets) | SLAKE [−0.0716,+0.0119]; VQA-RAD [−0.1868,−0.0495]; PathVQA [−0.0061,+0.0779]; RadImageNet [+0.1215,+0.1763] | SEP-pooled | J: 1 loss, 2 ties, 1 win; EM: 2 losses, 1 tie, 1 win. Not re-run on final method | EXP `audit_stats_baselines_2026-09-20.json:bigger_model_reference_32B.cells.*.judge.probe_minus_32b(_ci)`; macros from `AUDIT_2026-09-18.md` §5 |
| **WED** (published training-free selector, arXiv 2605.18313) on identical pools | probe − WED: PathVQA +0.0843, SLAKE +0.0574, VQA-RAD +0.0050 | J (also wins in MG, WED-EM, WED-F1 on PathVQA/SLAKE) | [+0.0711,+0.0978] / [+0.0326,+0.0818] / [−0.0515,+0.0580] | OCT-final, T=0.7 pools | probe wins 2, ties 1 | EXP `wed_vs_probe_2026-10-16.json` (table in `FINAL_METHOD_2026-10-16.md` §5.3) |

**Why we chose this.** Every way of picking among the 8 samples *without training* — majority vote, an
answer-frequency counter, picking at random or the longest, asking the model itself, or asking a 4.5×
bigger model — either loses to greedy decoding or ties a small trained verifier. Training is the active
ingredient, not size (the 32B zero-shot verifier ties the trained 7B). Between the two trained verifiers,
the probe reads the generator's own hidden states with a 918k-parameter MLP instead of running a
LoRA-tuned 7B forward per candidate, so it is far cheaper; and once trained on all eight benchmarks it
reaches +0.0897 (judge) / +0.0783 (MedGemma) / +0.0165 (EM). **Honest caveats:** (i) on the one like-for-like
comparison against the July LoRA (August probe, 2,345 questions) the probe wins under the judge but
**loses under exact match** (−0.0145, CI excludes 0); (ii) a LoRA verifier trained on the pooled 8-benchmark
data was never built, so "probe beats LoRA" at the final configuration is **not tested**; (iii) the 32B
comparison was done with exact-match labels in July on a different pool; (iv) logprob and semantic-entropy
selectors were never run as selectors.

---

## 2. Feature — which tokens, which layers, what else is appended

| alternative | measured | currency | CI / sd | era | verdict | source file:key |
|---|---|---|---|---|---|---|
| **answer-span mean (h_span)** vs **last answer token (h_last)**, answering prompt | L21 0.7956 vs 0.7807; L28 0.7895 vs 0.7772; L14 0.7698 vs 0.7616; L7 0.7405 vs 0.7486 (last better at L7) | J (SE) | none; ⚠️ "contaminated" in-eval CV, upper bound | JUL features | span ≥ last at layers ≥14 | MAIN/EXP `verifarch_hidden_promptform_indomain_2026-08-04.json:table` (mode generator) |
| span vs last vs span+last (BT, h256, L21) | 0.68047 (`A_objbt_h256`) vs 0.65456 (`B_L21_last`) vs 0.67006 (`B_L21_span+last`) | SE (CV) | sd 0.016 / 0.009 / 0.020 | SWEEP | span alone best | MAIN `_head_sweep_journal*.jsonl` tags as named |
| span vs span+last (BCE, h1024, L21) | 0.69408 (`R2_L21_span`) vs 0.69048 (`R2_L21_span+last`) | SE (CV) | sd 0.0125 / 0.0220 | SWEEP | span alone | same |
| **answering prompt** vs **grader prompt** (candidate as assistant turn vs "is this correct?") | in-domain L21/span 0.7956 vs 0.7827; CV-selected grader head alone 0.7507, −0.0245 [−0.0470, −0.0020] vs LoRA | J (SE) | as shown | JUL features | answering prompt | `VERIFIER_ARCHITECTURES_2026-08-04.md` §5, §3 row C6 |
| + question-token mean (h_q) | +0.0825 vs h_span +0.0806 (+0.0018) | J | none; 3 seeds | SEP-era, 108,126 rows (pre-PathVQA-backfill) | below 0.003 floor; not adopted | EXP `head_visual_features_2026-09-01.json:macro.h_span+h_q`, `context_control_gain` |
| + image-token mean (h_img) | +0.0808 (+0.0002); image tokens alone −0.0361 | J | none; 3 seeds | same | rejected | same file, `macro.h_span+h_img`, `macro.h_img_only`, `visual_gain` |
| + cosine(h_span, h_img) | +0.0807 | J | none | same | rejected | same file, `macro.h_span+cos` |
| question-only / prompt-only probe (no answer) | **not tested** | — | — | — | gap | — |
| max / attention pooling over the span | **not tested** (mean only) | — | — | — | gap | — |
| + self-consistency count as input feature | four-domain base +0.0098 (A→C); pooled −0.0016 (+0.0720 vs +0.0736) | J | none | AUG / SEP | helps only without pooled data; not shipped. Note: thread-count reruns favour the SC arm on 5 of 6 runs → a tie | EXP `head_best_config_2026-08-24.json:macro_minus_greedy.A/.C`; `head_final_stack_PVFIXED_2026-09-13.json:macro.pooled_ens_sc`; `repro_threading_2026-09-13.json:VERDICT` |
| **single layer vs layer ensemble** (pooled) | +0.0729 (one layer) vs **+0.0736** (18/20/22) = +0.0007 | J | none; 0.0007 is 4.7× below the 0.0033 thread floor | SEP | a hedge, not a gain | EXP `head_final_stack_PVFIXED_2026-09-13.json:macro.pooled_singlelayer`, `macro.pooled_ens` |
| **which layers** (single L18…L22, ensembles) | L18 +0.1036, L19 +0.1020, L20 +0.1007, L21 +0.1008, L22 +0.1028; {18,20,22} **+0.1051**; all five +0.1040; {19,20,21} +0.1033 | J | none; 5 seeds; **5 of 8 benchmarks only** (RadImageNet, Kvasir-x1, OmniMedVQA, VQA-Med, GEMeX) | SEP-era pooled, 103,550 rows | shipped set is the argmax, within noise | EXP `head_ens_width_2026-08-25.json:macro.*`, `cells` |
| layers 7/14/21/28 concatenated (BCE h1024) | L21 0.69408; L14+21 0.69536; L21+28 0.69257; L14+21+28 0.69273; L7+14+21+28 0.68787 | SE (CV) | sd 0.010–0.016 | SWEEP | no gain from concatenation | MAIN `_head_sweep_journal*.jsonl` `R2_L*_span` |
| linear probe L28 vs L21 | 0.64733 vs 0.66426 | SE (CV) | sd 0.016 / 0.008 | SWEEP | late layer worse | MAIN `_head_sweep_journal.jsonl` `ctl_L28span_bce_h0`, `ctl_L21span_bce_h0` |
| per-layer end-to-end on all 8 (L18–L22) | **withdrawn — unreproducible** (re-created artifacts cover 5 of 8 benchmarks) | — | — | AUG | do not quote | `OPENTEXT_FULL_RUNDOWN_2026-09-04.md` §5.6 banner |
| **z-score standardisation** vs none | omitting it costs 0.0075 SE | SE | none | AUG-4dom | adopted | `src/training_methods/genframe_selector.py:15-17` docstring (no artifact) |
| z vs l2-then-z | 0.70400 (`R2_normz_ln1`) vs 0.69970 (`R2_norml2z_ln1`); 0.69408 (z) vs 0.69470 (l2z), no LN | SE (CV) | sd ≈ 0.008–0.012 | SWEEP | tie | MAIN `_head_sweep_journal*.jsonl` |
| transfer-motivated variants (within-pool norm, PCA-32/128, rank-only, domain-adversarial) | head − string prior, out-of-domain: raw **+0.0734**; domadv +0.0558; poolnorm+raw +0.0527; poolnorm +0.0409; pca32 +0.0386; pca128 +0.0339; rankonly +0.0302 | SE | none | AUG (leave-one-domain-out) | plain features best | EXP `head_arch_transfer_2026-08-19.json:results.raw.MEAN_head_minus_prior`; `TRANSFER_WALL_2026-08-21.md` §2 |
| **states captured during generation** vs separate teacher-forced pass, same resolution | 24 of 2,345 picks change; 0.0000 | J | [−0.0021, +0.0021] TIE (EM +0.0013 [0.0000, +0.0030] TIE) | AUG-4dom single layer (L21), HF `generate()` | equivalent | MAIN/EXP `free_head_2026-08-16.json:DELIVERABLE_2_equivalence.picks_and_endpoint.AR_vs_TF_same_resolution.head_only::cap320_ar__vs__head_only::cap320_tf` |
| captured at cap320 vs deployed (full-res teacher-forced) features | 153 picks change; −0.0009 | J | [−0.0060, +0.0043] TIE | AUG-4dom | tie | same file, `VERDICT.1_…cap320_captured_vs_deployed.judge` |
| feature extraction at the generation resolution (cap320) for the pooled/final probe | **not tested** (features at 1,003,520 px, pools at 250,880 px); re-extraction in flight | — | — | SEP/OCT | gap | `AUDIT_2026-09-18.md` §4; DOMAIN_GUIDE §2.6 |

**Why we chose this.** The mean over the candidate's own answer tokens carries more correctness signal
than the last token at every layer from 14 up, and the state the model computes *while answering* beats
the state it computes when *asked to grade* — the model "knows" more while producing the answer. Adding
the question tokens, the image tokens, their cosine, or a self-consistency count adds at most +0.0018,
which is below the refit floor, so the input stays one vector per layer. Standardising with frozen
training statistics is cheap insurance (+0.0075 SE when it was measured). **Honest caveats:** the layers
18/20/22 were chosen by scoring layer sets end-to-end on the held-out halves (`AUDIT_2026-09-18.md` §5), the
layer-set comparison covers only 5 of 8 benchmarks, and the spread across sets (≈0.004) is within noise —
the 3-layer ensemble is a hedge against picking the wrong single layer, not a measured gain (+0.0007).
Capture-during-generation was shown equivalent only for an August single-layer predecessor; the
shipped 24-probe ensemble has never been run that way.

---

## 3. Probe architecture and optimisation — MLP 256, BCE, AdamW 1e-3, wd 1e-2, 30 epochs

All rows except the last two are **SWEEP** (four-domain pool, layer 21, 5-fold CV `cv_sel_eff`, sd ≈ 0.01;
MAIN `_head_sweep_journal*.jsonl`, by tag). Differences under ~0.015 are about one sd.

| axis | values measured (`cv_sel_eff`) | shipped value | verdict | tags |
|---|---|---|---|---|
| **width** (BCE, depth 1, matched "A" stage) | linear 0.66426 · 128 0.69029 · **256 0.69396** · 512 0.69075 · 1024 0.69742 | 256 | linear is clearly worse (−0.030); hidden ≥128 all within ~1 sd | `A_objbce_h{0,128,256,512,1024}` |
| width (BCE, "REG" stage) | 16 0.68314 · 32 0.70090 · 64 0.68840 · 128 0.69298 · **256 0.69623** | 256 | capacity irrelevant above 32 | `REG_h{16,32,64,128}`, `REG_drop0.0` |
| **depth** (h1024) | 1 hidden layer 0.69742 / 0.69408 · 2 hidden layers 0.69777 | 1 | tie | `A_objbce_h1024`, `R2_L21_span`, `R2_d2_dr0.0` |
| **dropout** (h256 BCE) | **0.0 0.69623** · 0.1 0.68469 · 0.3 0.68930 · 0.5 0.69260 · 0.7 0.67781 | 0.0 (vestigial `Dropout(0.0)` layer) | no value beats 0; within ~1 sd | `REG_drop*` |
| **epochs** (h256, no schedule) | 5 0.66796 · 10 0.68714 · 20 0.69273 · **30 0.69623** · 60 0.69935 | 30 | <20 under-fits; 30 vs 60 tie | `REG_ep*`, `REG_drop0.0` |
| **schedule** (h1024) | cosine 60 ep **0.70598** (sweep winner) · cosine 120 ep 0.70350 · none 30 ep 0.69742 | none | ~1 sd; chased end-to-end in October (row below) | `R2_ep60_cos`, `R2_ep120_cos` |
| **learning rate** | **BCE: not tested.** BT only: 3e-4 0.68890 · 1e-3 0.68047 · 3e-3 0.66908 | 1e-3 | gap — the lr grid was run only for the BT objective, where 3e-4 scored highest | `E_lr0.0003`, `A_objbt_h256`, `E_lr0.003` |
| **weight decay** (h256 BCE) | 1e-3 0.69587 · **1e-2 0.69623** · 1e-1 0.69985 · 0.3 0.69468 · 1.0 0.69191 | 1e-2 | flat | `REG_wd*` |
| LayerNorm inside the probe (h1024) | with LN 0.70400 / 0.69837 (rerun) vs without 0.69408 / 0.69423 (rerun) | no LN | ~+0.005–0.010 in CV, never tested end-to-end | `R2_normz_ln1`, `F_R2_normz_ln1`, `R2_L21_span`, `F_A_objbce_h1024` |
| **objective**, matched representation (h256) | **BCE 0.69396** · Bradley–Terry 0.68047 · listwise 0.67386 · hybrid 0.68828 | BCE | ⚠️ confounded: BT/listwise train on 38.9% of rows and get 3.26× fewer steps (`OPENTEXT_CORRECTIONS_2026-08-19.md` §3b); step-matched BT (98 ep) 0.67623 vs BCE (30 ep) 0.69623 | `A_obj{bce,bt,listwise}_h256`, `F_hybrid1.0`, `BUD_bt_ep98`, `BUD_bce_ep30` |
| objective, JUL features (restricted-set control) | pointwise BCE 0.7704 > BT 0.7459 > listwise 0.6907; swapping the loss on the *same* restricted rows costs −0.0817 [−0.1042, −0.0586] | BCE | ranking losses hurt | `VERIFIER_ARCHITECTURES_2026-08-04.md` §4.3 |
| **set-awareness** (candidate features relative to its pool) | BCE h256: `srboth` 0.66696, `srresid` 0.66687 vs 0.69396; h1024: 0.67873 / 0.66733 vs 0.69408 | none | worse | `G_sr*_bce`, `R2_sr*` |
| **per-dataset reweighting** | BCE h256 0.68975 vs matched control 0.69396; h1024 0.68309 vs 0.69408 | none | slightly worse, ~1 sd | `H_rwper_ds_bce`, `A_objbce_h256`, `R2_rw_per_ds`, `R2_L21_span` |
| **OCT end-to-end: wide probe** (1024, 60 ep, cosine — the sweep winner) vs same-session control | +0.0761 vs +0.0740 = +0.0021 | J | below the ~0.003 refit floor | rejected | EXP `head_final_stack_08wide_2026-10-15.json:macro.pooled_ens`, `head_final_stack_08ctrl_2026-10-15.json:macro.pooled_ens` |
| model class on identical features (JUL feature arm) | MLP 0.7704 vs gradient-boosted trees 0.7078 | J (SE) | none | MLP | `VERIFIER_ARCHITECTURES_2026-08-04.md` §6 item 2 |

**Why we chose this.** A linear probe is clearly too weak (−0.03 SE), but once there is one hidden layer of
≥32 units, nothing else in a 147-configuration grid — width, depth, dropout, weight decay, epochs, schedule,
LayerNorm, set-aware features, reweighting — moves CV selection efficiency by more than about one standard
deviation. So the recipe is the simplest point on a flat plateau. When the single best grid cell (1024 wide,
60 epochs, cosine) was refitted end-to-end in October against a control fitted in the same session, it gained
+0.0021 — inside the ~0.003 floor — so 256 stayed. BCE is kept because it uses every candidate row
(including all-right and all-wrong questions, which ranking losses must discard) and consumes exactly the
binary label the judge produces. **Honest caveats:** (i) the sweep ran on the four-domain, layer-21 era, so it
*informed* rather than *selected* the pooled recipe; (ii) the BCE-vs-BT gap is confounded by data volume —
do not claim BCE beats BT; (iii) the learning rate was never swept for BCE; (iv) LayerNorm looked ~1 sd better
in CV and was never tested end-to-end; (v) the CV→eval transfer of model selection is known to fail on this
split (`VERIFIER_ARCHITECTURES_2026-08-04.md` §6 item 3).

---

## 4. Ensemble and readout — 24 probes, rank average, vote-count tie-break

All rows in the first block: SEP-pooled probes, identical held-out questions, pre-specified adoption rule
("replace rank only if the macro delta is > 0 under BOTH judges, the CI excludes 0 under at least one, and the
EM delta is not a significant loss" — EXP `readout_variants_2026-10-15.json:adoption_rule`). Values are
variant **minus rank average**.

| alternative | J | MG | EM | verdict | source key (EXP `readout_variants_2026-10-15.json`) |
|---|---|---|---|---|---|
| raw logit average | +0.0012 [−0.0009, +0.0044] | +0.0008 [−0.0015, +0.0041] | +0.0030 [−0.0001, +0.0071] | tie — rejected | `macro.logit_mean_minus_rank_{j,x,em}` |
| z-scored score average | +0.0013 [−0.0009, +0.0045] | +0.0008 [−0.0016, +0.0040] | **+0.0032 [+0.00001, +0.0073] WIN** | rejected under rule (judges tie) | `macro.z_mean_minus_rank_*` |
| probability average | +0.0027 [−0.0008, +0.0071] | +0.0003 [−0.0026, +0.0039] | −0.00005 [−0.0049, +0.0051] | rejected | `macro.prob_mean_minus_rank_*` |
| Platt-calibrated average | +0.0002 [−0.0029, +0.0039] | −0.0022 [−0.0046, +0.0002] | +0.0004 [−0.0028, +0.0042] | rejected | `macro.platt_mean_minus_rank_*` |
| **rank average + vote-count tie-break** | **+0.00059 [+0.00005, +0.00152] WIN** | +0.00050 [−0.00006, +0.00143] | **+0.00060 [+0.00003, +0.00154] WIN** | **ADOPTED** (`adopt.rank_vote = True`) | `macro.rank_vote_minus_rank_*` |
| probe-weighted majority vote | −0.0040 [−0.0134, +0.0057] | −0.0058 [−0.0155, +0.0040] | **+0.0135 [+0.0036, +0.0237] WIN** (F1 +0.0094 WIN) | not adopted — a currency trade | `macro.vote_prob_minus_rank_*` |

| other readout / ensemble question | measured | currency | era | verdict | source |
|---|---|---|---|---|---|
| average ranks vs `argsort` ranks | 0.806540 vs 0.798365 SE | J (SE) | AUG (LoRA ⊕ BT-head fusion) | average ranks; ⚠️ chosen on eval data | `genframe_data.py:640-641` docstring; `OPENTEXT_CORRECTIONS_2026-08-19.md` §7 item 4 |
| tie-break rules (lexical / most-sampled / longest / first row) | ambiguity band +0.0061; spread across rules +0.0010; best − first_row +0.00018 | J | AUG-4dom, full benchmarks (tie rate 2.62%) | at that time: reproducibility fix only | EXP `tiebreak_2026-09-13.json`; `TRANSFER_WALL_2026-08-21.md` §11 |
| tie rate of the shipped probe | 0.58% of questions | — | SEP | small; band never measured on this probe | `AUDIT_2026-09-18.md` §5–§6 |
| **8 seeds vs 1 seed** | per-seed SE 0.788147–0.800409 (sd 0.004456) vs 8-seed ensemble 0.801090 | J (SE) | AUG-4dom | ensemble ≥ best single seed | `CHEAP_VERIFIER_ON_7B_2026-08-16.md` §6 item 2 |
| 2 seeds vs 8 seeds (control arm) | 0.8120 vs 0.8283 SE; a "+0.0196 WIN" at 2 seeds became +0.0072 [−0.0021, +0.0166] TIE at 8 | J (SE) | AUG | seed depth is load-bearing | `OPENTEXT_CORRECTIONS_2026-08-19.md` §2 (`head_eval_bigtrain_2026-08-19.json`) |
| number of seeds (4 / 8 / 16 …) as a swept axis | **not tested** | — | — | gap | — |
| layer ensemble vs single layer | see §2 (+0.0007; hedge) | | | | |

**Why we chose this.** The 24 probes are fitted independently and their raw outputs are not on a common
scale, so the readout averages each probe's *within-question rank* — only the order inside one question can
matter for an argmax. In October every calibrated alternative (z-score, Platt, probability, raw logits) was
tested on identical questions in four currencies under a rule fixed in advance: all tie under both judges,
so rank averaging stays. Breaking exact rank ties by how many of the 8 samples produced the string was the
one change that passed the rule (+0.0006 judge and EM, both significant, free). A probe-weighted vote raises
exact match by +0.0135 but lowers both judges — it trades one currency for another, so it was not adopted.
Eight seeds are used because seed noise is comparable to the effects being claimed. **Honest caveats:** the
original rank-average convention was picked on evaluation data (the October test is the first held-out
check); the number of seeds was never swept.

---

## 5. Training data — pool all eight benchmarks' training halves plus the four July domains

| alternative | measured | currency | CI | era | verdict | source file:key |
|---|---|---|---|---|---|---|
| **four July domains** (31,439 rows) vs **pooled eight + four** (112,770 rows) | +0.0182 → +0.0729 (single layer) → +0.0736 (ensemble); pooled lever +0.0554 | J | none | SEP | **the largest lever in the project** — adopted | EXP `head_final_stack_PVFIXED_2026-09-13.json:macro.deployed_4dom_L21ish`, `.pooled_singlelayer`, `.pooled_ens` |
| same lever on Qwen2.5-VL-7B | +0.0227 → +0.0820, 8/8; lever +0.0594 | J (EM +0.0276, strict EM +0.0187, F1 +0.0317) | J [+0.0708, +0.0924] | SEP recipe | replicates on the base model (same LM family) | `TRANSFER_WALL_2026-08-21.md` §12; `AUDIT_2026-09-18.md` §3 (`replication_currency_2026-09-20.json`) |
| **breadth only** (leave-one-benchmark-out) | four-domain +0.0279 → LOBO +0.0223 → pooled +0.0707; breadth −0.0056; own training half +0.0485 | J | none | SEP-era | per-benchmark labels are what matter | EXP `head_lobo_pooled_2026-08-25.json:macro.four_domain`, `.lobo`, `.pooled`, `macro_breadth_gain`, `macro_own_data_gain` |
| domain breadth at fixed row budget | mean slope +0.00868 per added domain | SE | [−0.00143, +0.01997] (Kvasir pair collapsed) | AUG | breadth not reliable | EXP `head_domain_scaling_MERGED_2026-08-21.json:collapsed_kvasir_family.mean_slope`, `.ci` |
| **onboarding price** (start from the other seven) | zero-shot macro +0.0196; first ~100 labelled q recover 46.4% (RadImageNet) / 65.9% (OmniMedVQA) of the gain | J | none; **3 seeds** | SEP-era | ~100 labels per new benchmark | EXP `head_price_from_lobo_2026-08-30.json:summary.zero_shot_macro`; `AUDIT_2026-09-18.md` §5 |
| per-dataset reweighting | see §3 (−0.004 to −0.011 CV, ~1 sd) | SE | — | SWEEP | not adopted | §3 |
| row-capped fit (cap each benchmark) | **not tested** | — | — | — | gap | DOMAIN_GUIDE §2.3.1 |
| does it just memorise templates / answers? | gain on novel question strings +0.1162 vs seen +0.1077; gold never seen in training +0.0918 | J (micro) | novel [+0.1071, +0.1255]; gold-unseen [+0.0824, +0.1010] | SEP | not a template prior (PathVQA is the exception) | EXP `template_stratified_gain_2026-09-20.json` via `AUDIT_2026-09-18.md` §2.5 |
| **labels from which judge** (2×2: train on A or B, grade with A or B) | train MedVLThinker → graded A +0.0710 / B +0.0658 / EM +0.0026; train MedGemma → A +0.0565 / B +0.0804 / EM +0.0101 | J / MG / EM | none; layer 20, 3 seeds | SEP-era | 7–30% of a judge-currency gain is judge-specific; cross-judge gain +0.057 to +0.066 | EXP `judge_2x2_2026-09-20.json:macro.probe_{A,B}_graded_{A,B,EM}` |
| consensus labels (both judges say correct) | A +0.0703 / B +0.0733 / EM +0.0033; soft (A+B)/2: +0.0663 / +0.0748 / +0.0057 | J / MG / EM | none; 3 seeds | SEP-era | at least as judge-robust; margin at the noise floor; not adopted | EXP `judge_consensus_2026-09-20.json:macro.probe_AND_graded_*`, `probe_SOFT_graded_*` |
| training-pool temperature matched to deployment | +0.0013 (eval T=0.2), −0.0028 (eval T=0.7) | J | none | AUG | no advantage | EXP `head_temp_matched_2026-08-24.json:VERDICT` |
| training on frozen-temperature pools (OCT) | **not tested yet** — in flight | — | — | OCT | gap | DOMAIN_GUIDE §2.6 item (i) |

**Why we chose this.** Retraining the probe on the training half of every benchmark is worth more than every
other design choice combined (+0.0554 judge), and it replicates at the same size on Qwen2.5-VL-7B. Breadth
alone does nothing on a benchmark the probe has never seen (−0.0056), so the honest framing is: **the method
needs a labelled split per benchmark, about 100 labelled questions to get roughly half to two-thirds of the
gain.** The gain is as large on question templates and gold answers never seen in training, so it is not
memorisation. Labels come from the MedVLThinker-32B judge; the 2×2 shows the probe partly learns its judge's
taste, which is why the cross-family judge is always reported beside it. **Honest caveats:** the price curve is
3 seeds against the project's ≥10-seed rule; reweighting and row-capping were tested only in CV or not at all.

---

## 6. Sampling — N = 8, per-benchmark temperature, no other knobs, cap320 resolution

### 6a. Number of samples N

| alternative | measured | currency | CI | era | verdict | source file:key |
|---|---|---|---|---|---|---|
| N=16 vs N=8 (real pools, matched questions) | macro +0.0167; 4 WIN (RadImageNet, Kvasir-x1, OmniMedVQA, GEMeX), 0 LOSS, 4 TIE | J | per-benchmark, image-clustered | SEP-pooled, T=0.7 | a genuine accuracy/cost point, costs a doubling of generation; not adopted for the base method; N=16 at the frozen T is in flight | EXP `coverage_sc16_ci_ALL_2026-09-16.json:macro_budget_16_minus_8`, `VERDICT` |
| N=8/16/32, RadImageNet | probe − greedy +0.1215 → +0.1544 → +0.1932; headroom converted 66.7% → 59.8% → 52.9% | J | none | SEP-pooled | more samples pay where selection skill exists | EXP `budget_conversion_sc32_2026-09-17.json:cells.radimagenet_open.*` |
| N=8/16/32, VQA-Med (coverage-limited) | −0.0033 → +0.0011 → +0.0044 while oracle rises 0.1987 → 0.3475 | J | none | SEP-pooled | extra samples wasted where skill is absent | same file, `cells.vqamed_open.*` |
| union of temperature pools, budget-matched to 8 | +0.0021 macro | J | none | AUG-4dom | gain was extra generation, not a better pool | EXP `mixed_temperature_2026-08-22.json:VERDICT` |
| union of the whole decoding grid (29.97 distinct / item) | oracle +0.1633, but selected −0.004975, SE 0.765 → 0.601 | J | none | JUL-LoRA | more candidates is not the fix | `INFERENCE_PARAMS_2026-08-13.md` §6 (`_infparams_ceiling.json`) |
| pool pruning (5 pruners × keep 2/3/4/6) | best fixed rule −0.0000 | J | none | AUG | rejected | EXP `pool_pruning_2026-08-24.json:VERDICT` |
| N=4 at T=0.4 vs N=8 at T=0.7 | +0.00159 [−0.0051, +0.0083] TIE (J); +0.00825 [+0.0015, +0.0151] WIN (EM), at half the generation | J / EM | as shown | JUL-LoRA | never re-tested on the probe | `CHEAP_VERIFIER_ON_7B_2026-08-16.md` §4 (`decoding_ladder_cold_2026-08-14.json`) |
| adaptive / dynamic N (Weitzman stopping) | **not tested with the probe** (July cascade only) | — | — | — | gap | DOMAIN_GUIDE §8 item 14 |

### 6b. Temperature

| alternative | measured | currency | CI | era | verdict | source file:key |
|---|---|---|---|---|---|---|
| one fixed T for all benchmarks | T=0.2 +0.0377 · 0.4 +0.0507 · **0.7 +0.0736** · 1.0 +0.0691 | J | none | SEP-pooled | 0.7 best under the judge; 0.7 vs 1.0 = +0.0045, near noise | EXP `temp_crossfit_2026-10-15.json:macro.fixed_{0.2,0.4,0.7,1.0}_minus_greedy_judge` |
| same, exact match | T=0.2 +0.0100 · 0.4 +0.0095 · 0.7 +0.0044 · 1.0 −0.0096 | EM | none | SEP-pooled | ⚠️ EM prefers cold — the optimum depends on the currency | same file, `macro.fixed_*_minus_greedy_em` |
| per-benchmark T chosen in-sample | +0.0877 vs +0.0773 at T=0.7 = +0.0103 | J | none — upper bound, chosen on eval | SEP-era pre-v2 probe | motivated the test below | EXP `head_temp_ensemble_2026-08-30.json:value_of_tuning_T_per_benchmark` |
| per-benchmark T, cross-fitted inside the held-out half | +0.0088 over fixed 0.7 | J | [−0.0003, +0.0177] TIE (CI "somewhat anticonservative": T not re-chosen inside resamples) | SEP-pooled | map then frozen | EXP `temp_crossfit_2026-10-15.json:macro.crossfit_minus_fixed07_judge`, `ci_caveat` |
| **frozen map** (GEMeX/OmniMedVQA/RadImageNet 1.0, VQA-RAD 0.2, VQA-Med 0.4, rest 0.7) vs fixed 0.7, **image half 1 (played no part in choosing)** | **J +0.0091**, **MG +0.0090**, EM +0.0041, F1 +0.0002, LLaVA −0.0015 | J / MG / EM / F1 / LLaVA | J [+0.0007, +0.0182] WIN; MG [+0.0010, +0.0175] WIN; EM [−0.0062, +0.0152] TIE; F1 [−0.0080, +0.0089]; LLaVA [−0.0100, +0.0077] | OCT-final probes | **ADOPTED** | EXP `tmap_vs_t07_2026-10-16.json:macro.fold1_out_of_sample.*`; map in `frozen_temperature_map_2026-10-15.json:frozen_T` |
| T grid finer than {0.2, 0.4, 0.7, 1.0} | **not tested** | — | — | — | gap | — |
| JUL-era decoding ladder (LoRA verifier) | T=0.3 +0.0054 TIE, T=0.5 +0.0057 TIE (seed-aware), T=1.0 −0.0165 LOSS, T=1.3 −0.0456 LOSS vs T=0.7 | J | T=0.5 [−0.0026, +0.0141]; T=1.0 [−0.0286, −0.0053]; T=1.3 [−0.0590, −0.0323] | JUL-LoRA | optimum depends on the verifier: a weaker verifier wanted colder samples | `INFERENCE_PARAMS_2026-08-13.md` §2 (`_infparams_seedaware.json`) |

### 6c. Other decoding knobs (all JUL-LoRA era, 2,345 questions, 3 generation seeds; never re-run on the probe)

| knob | measured vs T=0.7 baseline | currency | CI (seed-aware) | verdict | source |
|---|---|---|---|---|---|
| min-p 0.10 | +0.0040 (J); +0.0097 (EM) | J / EM | J [−0.0051, +0.0132] TIE; EM [+0.0017, +0.0178] SIG | not adopted; untested on probe | `INFERENCE_PARAMS_2026-08-13.md` §2, §8 |
| repetition penalty 1.10 | +0.0048 (J); −0.0077 (EM) | J / EM | J [−0.0055, +0.0154] TIE; EM [−0.0148, −0.0006] LOSS | rejected — currency trap | same |
| repetition penalty 1.05 | +0.0013 | J | [−0.0070, +0.0098] TIE | rejected | same §2 |
| top-p, top-k | **not measured** (runs started and abandoned) | — | — | gap | same §8 |

### 6d. Image resolution

| alternative | measured | currency | CI | era | verdict | source file:key |
|---|---|---|---|---|---|---|
| **full resolution (1,003,520 px) vs cap320 (250,880 px)**, whole method, T=0.7, cross-fitted | method +0.0745 vs **+0.0790** | J | full-res [+0.0651, +0.0835]; cap320 [+0.0708, +0.0874] | OCT | cap320 kept; 2026-09-17 decision reversed | EXP `select_policy_cap320_vs_fullres_T07.json:macro.fullres_minus_ref_greedy_judge`, `.cap320_minus_ref_greedy_judge` |
| choose resolution per benchmark (cross-fitted) | −0.0023 vs cap320 | J | [−0.0087, +0.0037] TIE | OCT | rejected | same file, `macro.selected_minus_cap320_judge` |
| greedy at full-res vs same-session cap320 control, GEMeX | −0.0414 (0.356 vs 0.397); oracle@8 −0.0279 | J | [−0.0488, −0.0341] LOSS; oracle [−0.0360, −0.0200] | OCT | full-res hurts chest X-rays | EXP `resolution_check_2026-10-15.json:cells.gemex_open.greedy_fullres_minus_cap320`, `.oracle8_fullres_minus_cap320` |
| greedy full-res vs cap320, others | PathVQA +0.0200 WIN, Kvasir-x1 +0.0091 WIN, VQA-RAD +0.0300, SLAKE +0.0202, VQA-Med +0.0035, OmniMedVQA −0.0037 (ties) | J | per-cell in artifact | OCT | mixed | same file, `cells.*`; `FINAL_METHOD_2026-10-16.md` §2 |
| vision tokens | full-res = 2.76× the vision tokens of cap320 | — | — | OCT | cost | `FINAL_METHOD_2026-10-16.md` §4 (`image_geometry_all8_2026-10-15.json`) |

**Why we chose this.** Eight samples is where the project's cost was measured and where a probe still
converts most of its headroom; going to 16 is a real but paid improvement (+0.0167, generation doubled) that
pays only on the four benchmarks where the probe already selects well — a separate accuracy-max operating
point, not the base method. Temperature matters (macro spread 0.0381 across the four fixed temperatures,
`head_temp_ensemble_2026-08-30.json:macro_spread_across_T`), and the benchmarks the
probe wins most want *hotter* samples while the two it used to lose want *colder* ones; a per-benchmark map,
chosen on one image half and frozen, gained +0.0091 judge / +0.0090 MedGemma on the other half. min-p and
repetition penalty were neutral under the judge and repetition penalty loses under exact match. Full
resolution was decided in September from three benchmarks where it helps, then measured on all eight: it
costs GEMeX 4 points of greedy accuracy and the method is better at cap320, at a quarter of the image
tokens. **Honest caveats:** the T grid has only four values; the temperature effect is judge-favoured (EM
prefers T=0.2–0.4, and LLaVA recall is −0.0015 for the map); min-p, repetition penalty, top-p and adaptive N
were never run with the probe.

---

## 7. Evaluation protocol

| choice | alternative / check | measured | currency | era | verdict | source file:key |
|---|---|---|---|---|---|---|
| **judge = MedVLThinker-32B** | — | it was the script default, never deliberately chosen; text-only, sees the gold; MedEvalKit's own open-question judge is GPT-4.1 | — | all | stated, not selected | `AUDIT_2026-09-18.md` §1 (`src/labeling/run_judge.py:21`) |
| judge determinism | re-judge 4,047 cached labels | 4,047 / 4,047 agree | J | JUL | deterministic | `INFERENCE_PARAMS_2026-08-13.md` §3 (`_infparams_rejudge.json`) |
| **cross-family judge MedGemma-27B** | re-grade identical picks | SEP +0.0679 (7/8 positive, 5/8 significant), κ 0.714–0.901 per benchmark; OCT +0.0783 [+0.0715, +0.0853] | MG | SEP / OCT | 92% of the judge gain survives | `AUDIT_2026-09-18.md` §2.3 (`xjudge_rescore_medgemma27b_2026-09-20.json`); EXP `final_eval_FINAL_cap320_Tmap.json:…pick_minus_greedy_xjudge` |
| self-preference check | P(judge=1 \| EM=0): Lingshu 0.1389, Qwen 0.1674, MedGemma-4b 0.1659 | J | SEP | no same-family preference visible | `AUDIT_2026-09-18.md` §1 (`replication_currency_2026-09-20.json`) |
| length bias | pick-longest −0.0294, pick-shortest −0.0301; length coefficient −0.157 | J | SEP | judge does not reward length | `AUDIT_2026-09-18.md` §2.2 (`judge_length_bias_2026-09-20.json`) |
| independent human/other-rater validation of the judge | only SLAKE / VQA-RAD / PathVQA = 11.1% of questions | — | — | gap on the other five benchmarks | `AUDIT_2026-09-18.md` §5 |
| **currencies reported** | J, MG, EM, F1, LLaVA recall (+ ImageCLEF-style strict on VQA-Med 2019) | all five significant at OCT; per benchmark EM loses on Kvasir-x1 [−0.0441, −0.0322] and GEMeX [−0.0251, −0.0096] (sentence golds) | all | OCT | report all | `FINAL_METHOD_2026-10-16.md` §0, §3; VQA-Med 2019 strict 0.346 → 0.348 in `vqamed2019_by_category_2026-10-16.json` |
| **CI clustered by image** vs i.i.d. over questions | a "router WIN" +0.0133 [+0.0109, +0.0156] i.i.d. became +0.0119 [−0.0032, +0.0297] clustered by benchmark; at ρ=0.2, 4.44 q/image widens every CI ×1.30 | J | AUG | always cluster | `OPENTEXT_FULL_RUNDOWN_2026-09-04.md` §6 item 2; `OPENTEXT_CORRECTIONS_2026-08-19.md` §7 item 2 |
| benchmark-level interval (n=8 clusters) | SEP judge macro [+0.0260, +0.1165]; t-interval [+0.0156, +0.1319] | J | SEP | wider, still > 0 | `AUDIT_2026-09-18.md` §2.1; EXP `audit_stats_baselines_2026-09-20.json:summary_by_currency.judge.probe.benchmark_t_interval_df7` |
| seed-aware CI (generation seed variance) | T=0.5 "WIN" [+0.0010, +0.0104] → [−0.0026, +0.0141] TIE | J | JUL | item-only CIs are anticonservative for sampling claims | `INFERENCE_PARAMS_2026-08-13.md` §2 |
| **two-fold image cross-fitting** vs one held-out half | SEP one half: +0.0737 [+0.0608, +0.0864] (n 18,452), EM tie; OCT both halves: +0.0897 [+0.0831, +0.0966], EM +0.0165 WIN; image half 1 alone (out of sample for every October choice) +0.0936 [+0.0849, +0.1026] | J / EM | SEP / OCT | doubles n, every question scored by a probe that never saw its image | `FINAL_METHOD_2026-10-16.md` §0; EXP `final_fold1_headline_2026-10-16.json` |
| frozen-probe reproduction | frozen v2 probes reproduce the headline: +0.073720 vs recipe +0.073607 | J | SEP | reproducible | `AUDIT_2026-09-18.md` §2.1; EXP `final_eval_validate_cap320_frozenv2.json:macro_over_open8.fold0.pick_minus_greedy_judge` (+0.0737) |
| **thread count** of the CPU fit | 1/2/4/8 threads → +0.072615 / +0.074448 / +0.071169 / +0.071660; spread 0.0033; bitwise deterministic at fixed threads | J | SEP refit, one arm | arm differences < ~0.003 across runs are noise | EXP `repro_threading_2026-09-13.json:thread_sweep_spread`, `VERDICT` |
| **seed depth** of verifier claims | co-adaptation gain +0.00384 (1 seed) → +0.00014 (4) → −0.00173 (10) | J | AUG | ≥10 seeds for small claims | `coadapt_verifier_T04_2026-08-14.json` (CLAUDE.md §0) |
| macro vs micro | judge +0.0737 macro vs +0.1109 micro | J | SEP | macro reported (conservative) | EXP `audit_stats_baselines_2026-09-20.json:summary_by_currency.judge.probe.micro_question_weighted` |
| pre-specified adoption rules | October readout rule (above), temperature map frozen before full-res pools, ~0.003 floor for the wide probe | — | OCT | the first round chosen this way | `readout_variants_2026-10-15.json:adoption_rule`; `frozen_temperature_map_2026-10-15.json:frozen_at` |
| probe-refit (seed) variance inside the final CI | **not included** — the bootstrap resamples images with probes fixed | — | OCT | gap | `final_eval_FINAL_cap320_Tmap.json:bootstrap` |
| control task (shuffled labels, selectivity) / within-question AUROC | **not run**; only a feature-permutation null (13.35 σ above null, AUG) | — | — | gap | DOMAIN_GUIDE §2.5 holes 8–9; `CHEAP_VERIFIER_ON_7B_2026-08-16.md` §8 |

**Why we chose this.** The judge is an LLM that sees the gold answer, which is the only grader that accepts
correct paraphrases ("Oculus dexter" for "right eye"); exact match is reported beside it because it cannot be
gamed by paraphrase, and it is known to undercount on sentence-gold benchmarks. Because a probe trained on one
judge's labels partly learns that judge, every headline is re-graded by MedGemma-27B, a judge from a different
model family, which keeps 92% of the gain. Intervals are clustered by image, because several questions share
one image and an i.i.d. bootstrap manufactured a false win once. Cross-fitting scores every question with a
probe that never saw its image, doubling the sample over one held-out half. **Honest caveats:** the judge
itself was a default, not a choice; independent validation of it covers 11% of questions; the final CIs hold
the probes fixed (no refit variance); and the protocol differs from published work (judge model, SLAKE
English only, cap320) — `FINAL_METHOD_2026-10-16.md` §5.2.

---

## 8. Cost

| quantity | greedy | best-of-8 (final method's sampling) | ratio | source |
|---|---:|---:|---:|---|
| GFLOPs per question (vLLM, shared prefill) | 5,831 | 6,589 | 1.13× | EXP `bestofn_vllm_2026-09-16.json` via `FINAL_METHOD_2026-10-16.md` §4 |
| latency per question (batch 1, A100) | 174.0 ms | 476.6 ms | 2.74× | same, `ratios_vs_greedy.latency` = 2.739 |
| energy per question | 34.3 J | 123.2 J | 3.59× | same, `measured.gen1/gen8.energy_j.mean`, `ratios_vs_greedy.energy` = 3.594 |
| HF `num_return_sequences=8` path | — | — | 1.99× latency / 2.95× energy | same, `HF_COMPARISON.hf_ratio_*` |
| July LoRA verifier (HF, re-encodes the image per candidate) | — | — | 3.68× latency / 5.77× energy vs HF greedy | DOMAIN_GUIDE §2.4.3 (`bestofn_latency_energy_2026-08-03.json`) |
| probe itself (design) | — | 24 × ~1.8 MFLOP per candidate on already-computed states | negligible | DOMAIN_GUIDE §2.4.6 |
| capture during generation (instead of a separate pass) | — | equivalence shown for an August single-layer head (24 / 153 picks change, ties) | — | `free_head_2026-08-16.json` (see §2) |
| full resolution instead of cap320 | — | — | 2.76× vision tokens | `FINAL_METHOD_2026-10-16.md` §4 |
| per-benchmark temperature | — | — | no budget change | `FINAL_METHOD_2026-10-16.md` §4 |

**Why we chose this.** Sampling 8 answers from one image encoding costs only 1.13× the arithmetic of greedy
because ~98% of the tokens are the shared prompt and image — but 2.74× the latency and 3.59× the energy, so
quote the measured milliseconds and joules, never FLOPs alone. The probe was chosen partly because it reads
states the generator already computes, instead of a LoRA verifier that re-encodes the image for every
candidate (3.68× latency, 5.77× energy) and cannot be served under vLLM (vLLM drops its visual LoRA modules).
**Honest caveats:** nothing has been run as one end-to-end deployed system; the probe's features were
extracted in a separate teacher-forced pass, so "no extra forward pass" describes the design, not the run;
VRAM was not measured.

---

## 9. Gaps — choices never tested (or tested only in an earlier era)

1. **LoRA verifier retrained on the pooled 8-benchmark data** vs the pooled probe — never built. The only
   like-for-like comparison (August probe vs July LoRA) is a judge WIN and an **EM LOSS**.
2. **Logprob / perplexity / P(True) / self-certainty selectors** — uncomputable from the stored dumps.
3. **Semantic entropy / semantic-cluster vote** as a best-of-N selector — never run (only as July error detection).
4. Published hidden-state verifiers re-implemented on our data (ELHSR, LiLaVe, CASE) — missing (`AUDIT_2026-09-18.md` §5).
5. **Learning rate for the BCE probe** — the lr grid was run only for Bradley–Terry.
6. **LayerNorm in the probe** — ~1 sd better in CV, never tested end-to-end.
7. **Number of seeds** (e.g. 4 / 8 / 16) as a swept axis.
8. **Question-only / prompt-only probe**, max or attention pooling over the answer span.
9. **Layer choice on all 8 benchmarks** — the layer-set comparison covers 5 of 8; per-layer end-to-end numbers were withdrawn; layers 18/20/22 were chosen on the held-out halves.
10. **Feature extraction at the generation resolution (cap320)** and **capture inside the vLLM sampling pass** for the shipped 24-probe ensemble — in flight / not done; capture equivalence is shown only for an August single-layer head.
11. **Probes trained on the frozen-temperature pools** (final probes are trained on T=0.7 pools) — in flight.
12. **N = 16 at the frozen temperatures** as an accuracy-max point — in flight; **adaptive N** never run with the probe.
13. **min-p, repetition penalty, top-p, top-k** with the probe — min-p/rp only tested with the July LoRA; top-p/top-k never measured. **Temperature grid** has only 0.2/0.4/0.7/1.0.
14. **Row-capped training** (does Kvasir-x1's 23,221 rows vs SLAKE's 635 dominate the probe?).
15. **Price curve at ≥10 seeds** (current: 3 seeds).
16. **Generator LoRA-SFT, 7B+probe vs Lingshu-32B greedy, and the answer-string prior** — measured on the September probe, never re-run against the final method; 32B greedy exists on only 4 of 8 benchmarks.
17. **Cross-family generator replication** — none valid (MedGemma-4b pools broken and withdrawn; Qwen2.5-VL is Lingshu's own base model).
18. **Control task / selectivity and within-question AUROC** — never run.
19. **Probe-refit variance inside the final CIs** — bootstrap holds probes fixed.
20. **Judge validation against an independent rater** on 5 of 8 benchmarks.
21. **End-to-end deployed latency/energy/VRAM** of the whole method.

### Discrepancies found while compiling (fix before the deck)

- **Training rows:** the brief and DOMAIN_GUIDE give 112,770 for the final probes; that is the September
  probe. The final folds trained on 115,934 / 116,000 rows (`final_eval_FINAL_cap320_Tmap.json:folds`).
- **Reweighting control:** DOMAIN_GUIDE §2.3.1 compares `H_rwper_ds_bce` 0.68975 with a control of
  **0.68980** that comes from a code comment (`src/training_methods/head_sweep.py:300`, "published 0.6898"),
  not the journal. In the journal `ctl_bce_h256` is **0.69786** and the matched-stage `A_objbce_h256` is
  **0.69396**, so reweighting is −0.004 to −0.008 (about 1 sd), not "exactly nothing (0.00005)".
- **Probe vs July LoRA:** quote it as "judge +0.0162 WIN / EM −0.0145 LOSS", and label it the August
  single-layer head, not the shipped probe (`CHECK_2026-09-28.md` line 656–658).
- `audit_stats_baselines_2026-09-20.json:bigger_model_reference_32B.note` still calls the judge
  "Lingshu-32B judge"; the judge is MedVLThinker-32B (`AUDIT_2026-09-18.md` §1).
