# A unified, vision-aware, 7B-only pipeline CANNOT match always-32B-direct: it reaches macro **0.616278**, a shortfall of **−0.040395 [−0.052275, −0.028427]**, and injecting the vision signal into the verifier contributes **nothing** (+0.002044 sel_eff [−0.011580, +0.015668], n.s.).

**Round of 2026-08-12.** Synthesis + adversarial verification of three attacks (vision-aware verifier,
7B-only frontier, VRAM levers) plus three sibling artifacts from the same day (unified pipeline,
minimum escalation, pre-generation router). Every number below names its artifact. Abstention appears
nowhere: every arm discussed returns an answer on every item.

> **The one-sentence answer to the user's question.** *A 7B+verifier pipeline with no 32B at test time
> closes **32.2% [20.1%, 44.8%]** of the 0.0596 gap to always-32B-direct and remains **−0.0404**
> significantly short; the vision-injection idea is refuted with a mechanism (the language-side
> representation the verifier already reads **is** image-dependent, +0.024035 cv_sel_eff real-vs-noise,
> 10/10 seeds); but the pipeline runs in **23.42 GiB** of VRAM against the 32B's **72.60 GiB**
> (**3.10×** whole-suite, **6.61×** at 4-bit weights), and that ratio — not the accuracy — is the
> defensible claim.*

---

## 1. What I re-derived independently before believing anything

I did not import any of the three attacks' analysis code for the headline checks. GPUs were both busy
(load average 22, other rounds resident), so all verification below is CPU-side and reuses only frozen
artifacts on disk. No GPU job was launched, nothing was killed, `MedEvalKit/` untouched,
`src/training_methods/freeze_selector.py` **not** run.

| # | check | result | verdict |
|---|---|---|---|
| V1 | **Null test, frozen selector.** Re-ran `python3 src/training_methods/genframe_selector.py` from disk | `pass=True`, max abs deviation **4.47971781336598e-07**; reloaded heads reproduce the 2026-08-04 training-run eval logits at **0.0 — bit-exact** | PASS |
| V2 | **Macro baselines** recomputed cell-by-cell from `_selector_rerun_parts/vec_disjoint.npz` | always-7B **0.597087**, always-32B-direct **0.656672**, gap **0.059586** — identical to published | PASS |
| V3 | **Frontier headline re-derived from raw per-item vectors**, my own script, not `sevenb_only_frontier.py` | 7B-only macro **0.616278**, deviation from the artifact **−3.233e-07** (6-dp rounding) | CONFIRMED |
| V4 | **My own paired item bootstrap**, nboot=10,000, rng seed 0, resampling within each of the 8 cells | vs 32B-direct **−0.040395 [−0.052302, −0.028510]** (artifact: [−0.052275, −0.028427]); vs always-7B **+0.019191 [+0.011863, +0.026714]** (artifact: [+0.012003, +0.026682]) | CONFIRMED, differences are bootstrap RNG only |
| V5 | **Cross-source alignment audit** — do the selector's item order and `vec_disjoint.npz` agree? Compared `greedy_ok` from `genframe_data.load_items()` against `<cell>\|always_7b` element-wise | max abs deviation **0** on all three open cells (645 / 200 / 1500) | PASS — the pairing in V4 is real, not assumed |
| V6 | **Per-cell selector accuracy** derived independently from the frozen heads | slake_open **0.778295**, vqa_rad_open **0.510000**, pathvqa_open **0.390667**; pooled **0.507463** = `recipe.json:must_reproduce.acc` reached by a *different* aggregation | PASS |

**Traps from the brief, each explicitly cleared:**

- **A vision scorer that gains without using the image.** The image-ablation control *was* run, and
  properly: a separate noise-image feature cache (`feats_hidden_noise`), image-grouped 5-fold CV, folds
  computed **once from the real cache and reused verbatim** for the noise arm, 10 seeds.
  [`_visverif_parts/langside_image_dependence.json`]. Vision-only features (`Vmean`) score
  cv_AUROC **0.5022922980778332** — chance — so no arm can be gaining from a vision feature that carries
  correctness signal, because it carries none.
- **MCQ gain that is option-coverage luck.** The luck-floor control is present per cell:
  PMC random-gold permutation **0.2497**, MedXpert **0.2001**
  [`sevenb_only_frontier_2026-08-12.json:PART1.MCQ_MENU_UPDATE...`]. It is moot in the end — the
  verifier-over-options arm **loses**, so there is no gain to attribute.
- **A VRAM number that is a pool reservation.** The convention is explicit and correct: HF transformers,
  never vLLM ("vLLM reserves a pool and hides true allocation"), with (a) weights resident /
  (b) `max_memory_allocated` / (c) `max_memory_reserved` / (d) process footprint kept separate, and every
  (d) taken on a shared card flagged as a reconstruction [`vram_levers_2026-08-12.json:_meta`]. The VRAM
  instrument's own null test reproduces the 2026-08-11 S1 scenario at **max abs deviation 0.0000**.
  **But see §6 — the headline picked the wrong (d).**
- **Eval-visible selection.** The vision round's primary arm `L_Vmean` was named by a **pre-registered
  train-only 5-fold image-grouped CV** (`train_cv_concat.json`; `L_Vmean` is indeed the CV maximum at
  0.6750161364339039), and the eval-best arm `L_prod_sim` is reported *separately* as a declared
  anti-conservative second rule. Both null. The frontier's per-cell arm choice is 5-fold cross-fit over
  12 fold seeds with **seed sd 0.0** — the chosen arm was a singleton in every fold of every seed, so the
  headline reduces to an exact identity, which is what V3 verified.
- **A single lucky seed.** Every trained arm carries 10 seeds with mean/sd/range and a **seed-matched
  paired** contrast alongside the item bootstrap. The primary vision arm's per-seed paired mean is
  **−0.001771**, 5/10 positive, sign-test **p = 0.80**.

---

## 2. The frontier — accuracy × VRAM × compute

**Accuracy** = 8-cell macro, Variant B, equal weight 1/8, n = 42,224, paired item bootstrap nboot=10,000
against **always-32B-direct 0.656672**.
**VRAM** = whole-process footprint (d), **peak over the worst item of the whole suite** — this is the
column a deployer provisions, and §6 explains why it is not the number the VRAM round headlined.
**Compute** = macro-weighted FLOP-eq at the project's as-charged R32 = 4.57, `x_direct` = ÷4.57.

| # | configuration | macro | Δ vs 32B-direct [95% CI] | VRAM (d) GiB | ×smaller | compute x_direct | **32B resident?** |
|---|---|---:|---|---:|---:|---:|:--:|
| 1 | **always-7B greedy** (zero-32B floor) | 0.597087 | **−0.059586 [−0.0718, −0.0479]** LOSS | **23.4206** ᵐ | 3.10× | 0.2188 | **NO** |
| 2 | 7B + incumbent verifier best-of-k, cross-fit k | 0.605320 | **−0.0514 [−0.0634, −0.0398]** LOSS | 23.4206 ᵐ | 3.10× | 1.3403 | **NO** |
| 3 | **7B + frozen 8-seed selector, best-of-8 on the 3 open cells ← BEST 7B-ONLY** | **0.616278** | **−0.040395 [−0.052275, −0.028427]** LOSS | **23.4206** ᵐ | **3.10×** | **1.4199** ᵈ | **NO** |
| 3b | row 3 with **4-bit (nf4) weights** | not measured on macro-8 ᴺ | — | **10.9792** ʳ | **6.61×** | 1.4199 (weight-only quant = 1.0× FLOPs) | **NO** |
| 4 | + escalate VQA_RAD_open only (12.5% macro wt / 0.47% items) | 0.627528 | −0.029144 | ≥ 84.7452 ʳ | 0.86× | — | **YES** |
| 5 | + 4 cells escalated (50% macro wt / 13.8% items) | 0.648140 | −0.008532 | ≥ 84.7452 ʳ | 0.86× | — | **YES** |
| 6 | **minimum escalation that TIES: 6 of 8 cells (75% macro wt / 17.3% items)** | **0.657365** | **+0.000693 [−0.002214, +0.003646]** TIE | ≥ 84.7452 ʳ | 0.86× | — | **YES** |
| 7 | pre-generation router, honest nested CV, seed 0 | 0.655945 | −0.000728 [−0.006614, +0.005268] | 77.999 (weights) | — | **0.7427** | **YES** |
| 8 | **SHIPPED accuracy-max** (the current two-arm method) | 0.657500 | +0.0008 [−0.0022, +0.0037] TIE | 77.999 (weights) | — | 1.7400 | **YES** |
| 9 | **always-32B-direct (THE BAR)** | **0.656672** | 0 (reference) | **72.6023** ᵐ | 1.00× | 1.0000 | **YES** |
| — | *ceiling: perfect selector over the current 8-sample pool, MCQ at 7B* | *0.659625* | *+0.002953* | *23.4206* | *3.10×* | — | *NO* |

ᵐ **measured** on a clean exclusive card [`vram_testtime_2026-08-11.json`: S1 7B-direct-MCQ **23.4206**,
S4 open-text best-of-8 **18.7644**, S2 32B-direct **72.6023**].
ʳ **reconstructed** — (c) peak reserved + the 1.3835 GiB CUDA-context offset (shared card), or derived
arithmetic over measured weights; flagged as such at every use.
ᵈ **derived**: 8 generations + **7.636674** full-7B-pass-equivalents per open question for the selector
stack [`ckpts/train/genframe_head_ens8/recipe.json:cost`] ⇒ macro (5×1.0 + 3×15.636674)/8 = **6.488753**
FLOP-eq ⇒ 6.488753/4.57 = **1.4199×** always-32B-direct.
ᴺ 4-bit accuracy is measured on the **open half only** (§4); the macro-8 composite is deliberately
**not measured** because the two halves sit on different evaluation tracks.

**Sources.** Rows 1–3, 4–6, and the ceiling: `sevenb_only_frontier_2026-08-12.json`
(PART1 `honest_crossfit`, PART3 `a_exact_cell_subset_enumeration`). Row 2:
`min_escalation_2026-08-12.json:PART2_zero_32b`. Row 3b: `vram_levers_2026-08-12.json:bnb_quantised_unified_arm`.
Row 7: `pregen_router_2026-08-12.json:NESTED_HONEST_PRIMARY.NESTED_seed0`. Rows 8–9:
`min_escalation_2026-08-12.json:HEADLINE_TABLE`.

### THE HARDWARE-CLASS CLIFF — the single most important structural fact of this round

**Accuracy and compute are continuous in the escalation rate. VRAM is not.** Rows 1–3b never load the
32B and fit a **24 GB** board (a 12 GB board at 4-bit). Row 4 escalates **0.47% of items** and is
already in a different hardware class, because the 32B has to be *somewhere*:

- 7B weights **15.4937** + 32B weights **62.3125** = **77.8062 GiB** (both MEASURED), + the 32B's derived
  peak activations 5.5555 = **83.3617**, + CUDA context = **84.7452** — against a **79.1384 GiB** usable
  A100. **Does not fit.**
- The cleanest measured-only statement needs no derived term: 7B open-arm measured process peak
  **18.7644** + 32B measured resident weights **62.3125** = **81.0769 GiB**, over capacity by **1.0769**
  *before the 32B runs a single forward pass* [`cost_decomposition_2026-08-12.json`].
- **Resolution cannot rescue it.** The co-residency wall is a *weights* wall: bf16 7B+32B fails at every
  cap, still short by **0.2579 GiB** at cap20 (360 vision tokens)
  [`vram_levers_2026-08-12.json:coresidency_direct`].
- **Load-on-demand is not viable**, and fails the brief's own threshold on both terms: measured
  62.3107 GiB on disk, most-favourable measured load **139.8 s** (an independent instrument — a real
  `from_pretrained`) = **74.4× one 32B forward pass** (1.88 s), against a measured tie escalation of
  **17.3% of items** ⇒ **24.15 s** added latency averaged over all traffic, versus **0.325 s** with the
  32B resident on a second card. Worse than the thing it avoids by ~2 orders of magnitude
  [`_sevenb_frontier_parts/load_on_demand.json`]. *(0.1519 GiB/s cold read was measured on a contended
  `/data` mount with two sibling GPU jobs live — not an idle-system number.)*
- **The one lever that could have built a middle rung is measured and it fails.** 7B-bf16 + 32B-**nf4**
  co-loads at **33.7431 GiB** resident / **41.7839 (d)** — fits one 80 GB card with 37.35 GiB spare, and
  fits a 48 GB L40S. But swapping NF4 into the row-6 tie policy on the 3 cells where its accuracy is
  measured takes macro **0.657365 → 0.655034** and the non-inferiority CI lower bound from −0.002214 to
  **−0.006266: it breaks the pre-registered tie** [`_frontier_verify_parts/nf4_tie.json`]. **Read it
  correctly** — the paired swap delta is **−0.002331 [−0.006028, +0.001417]** and *spans zero*. The honest
  statement is *"the tie was marginal (0.000686 of slack) and NF4 is not certified to preserve it"*, **not**
  "NF4 damages accuracy". 5 of 8 cells, including all 3 open cells, are unmeasured at 4-bit.

⇒ **There is no measured operating point between 0.616278 at 23.42 GiB and a tie at ≥84.75 GiB.** The
frontier is a genuine either/or, not a smooth trade-off.

---

## 3. The per-cell shortfall, and why 5 of 8 cells have no 7B-side lever

My own derivation (V3/V6), 7B-only best policy minus always-32B-direct:

| cell | n | 7B-only best | always-32B-direct | shortfall | limited by |
|---|---:|---:|---:|---:|---|
| PMC_VQA | 33,430 | 0.542656 | 0.551780 | **−0.009124** | capability |
| SLAKE_closed | 836 | 0.825359 | 0.858852 | **−0.033493** | capability |
| VQA_RAD_closed | 251 | 0.780876 | 0.852590 | **−0.071713** | capability |
| PATH_VQA_closed | 3,362 | 0.840869 | 0.889054 | **−0.048186** | capability |
| MedXpertQA-MM | 2,000 | 0.261500 | 0.306500 | **−0.045000** | capability |
| SLAKE_open | 645 | 0.778295 | 0.818605 | **−0.040310** | selection |
| VQA_RAD_open | 200 | 0.510000 | 0.600000 | **−0.090000** | selection |
| PATH_VQA_open | 1,500 | 0.390667 | 0.376000 | **+0.014667 ← the 7B WINS** | — |

**"Capability-limited" is a measurement here, not an assumption.** On every MCQ cell the candidate set is
**complete by construction** (gold is always among the options, or in {yes, no}), so the coverage wall is
**0** and the oracle-over-candidates is a vacuous **1.0**. The honest MCQ ceiling is therefore the best
*measured* 7B-only mechanism — and every mechanism ever tried on these cells is ≤ greedy:

- **Verifier over the given options** (the unification the brief singled out as never tried) — measured
  this round and it **loses**: PMC **−0.0758 [−0.0897, −0.0625] SIG**, PATH_VQA_closed
  **−0.1689 [−0.1886, −0.1499] SIG**, MedXpert +0.0025 n.s., VQA_RAD_closed −0.0518 n.s. Zero wins.
  It is **not** a floor artifact: candidate-level AUROC is 0.583–0.800, so the verifier *has* ranking
  signal over the options — it is simply worse than the generator's own argmax
  [`unified_pipeline_2026-08-12.json:VERDICT.Q1`].
- **MCQ test-time augmentation**: always-K summed gain **−0.0078**; gated cross-fit **+0.0000062**
  [`mcq_tta_2026-08-10.json`].
- **Sampling best-of-N on MCQ**: structurally dead (PMC verifier pick 0.4325 **below** greedy 0.5060;
  MedXpert oracle@8 0.5365 **below** its own luck floor 0.6808).

Positive-part decomposition of the −0.0404 residual: **selection 0.0163 / coverage 0.0000 / pure
capability 0.0259** — **~64% of what is left is capability the 7B does not have.**

**The framing number the brief asked for.** With the 5 MCQ cells at always-7B, **oracle@8** selection on
the 3 open cells closes only **104.95%** of the gap, and parity needs a **uniform open sel_eff of
0.988342** [`vision_verifier_2026-08-12.json:macro`]. Against a **field constant of 0.78–0.81** across
~27 architectures. **Open-text selection alone cannot carry a no-32B pipeline to parity, even perfectly.**

---

## 4. The user's hypothesis, answered — and refuted with a mechanism, not a flat null

> *"we seem to be ignoring the Vision part for our verifier... maybe if we inject the vision signal into
> the verifier, it can help."*

**The verifier is not ignoring the image. The premise is false, and that is why every injection arm is
null.** Confound-free test, both arms trained **and** tested in-distribution on their own cache, identical
image-grouped folds, identical trainer, 10 seeds
[`_visverif_parts/langside_image_dependence.json`, `src/training_methods/langside_image_dependence_cv.py`]:

| arm | cv_sel_eff | cv_AUROC |
|---|---:|---:|
| real images | **0.804084** (sd 0.005545) | 0.773203 |
| noise images | **0.780050** (sd 0.008283) | 0.745898 |
| **contrast** | **+0.024035, 10/10 seeds positive** | **+0.027305, 10/10 positive** |

A causal LM attending over vision tokens puts the image into the language-side vector. There is no vision
blindness left to fix, so explicit vision features are largely redundant.

**Every one of the four sub-attacks returns a null**, including the two designed to be maximally
favourable to the hypothesis:

| arm (10 seeds each) | mean sel_eff | vs bar L 0.793869 (sd 0.005835) |
|---|---:|---|
| **L (language-side bar)** | 0.793869 | — |
| **L_Vmean (PRIMARY, train-CV-selected)** | 0.792098 | **+0.002044 [−0.011580, +0.015668] n.s.** |
| L_prod_sim (eval-best, declared second rule) | 0.801975 | +0.003406 [−0.009537, +0.016349] n.s. |
| L_prod / L_maxsim / L_simgrid | 0.797956 / 0.798501 / 0.792711 | all inside the seed spread |
| xattn (learned cross-attention over the 6×6 grid) | 0.783106 | **below the bar** |

Both **pre-registered falsification conditions triggered**: the primary CI spans zero, and the
mechanism endpoint — laterality, n=230 — is **−0.013043 [−0.052174, +0.026087]**, a *negative* point
estimate. Macro contribution of vision injection: **+0.000681 [−0.003831, +0.005217]**, i.e. **1.14% of
the 0.0596 gap with a CI of [−6.4%, +8.8%]** — indistinguishable from zero.

**Four supporting measurements:**

1. **Capacity ablation of the deployed clean LoRA**, all 253 mixed items, fidelity gate PASS (max dev
   5.0e-06 over 2,192 scores): zeroing **all 96 vision-tower LoRA modules** = **+0.015810
   [−0.019763, +0.055336]** — *no harm, point estimate positive*. Vision-only capacity =
   **−0.079051 [−0.142292, −0.019763] SIG LOSS**. The incidental 15.2% of adapter capacity sitting on the
   ViT contributes nothing measurable. **Structural finding:** all 192 vision tensors are
   `visual.blocks.*.mlp.{down,gate,up}_proj` — Qwen2.5-VL names ViT attention `attn.qkv`/`attn.proj`, so
   **the ViT's attention was never adaptable under this recipe** and the spatial-mixing part a laterality
   question needs had **zero** capacity.
2. **xattn localises hard but not sensibly**: 99.6–99.8% of attention mass on **one of 36 patches**
   (entropy 0.18% of uniform), and the attended position does not shift sign-consistently between
   'left'- and 'right'-bearing candidates on the same image (seed-mean −0.001977; 2/10 seeds nominally
   p<0.05 **with opposing signs**).
3. **Permutation null** (real images, deranged correspondence) barely moves — L_prod −0.010899 n.s. —
   while blank/noise collapse hard (L_prod_sim blank **0.614441**, *below* the 0.676260 random-pick
   floor). That is the signature of a **distributional**, not informational, effect.
4. **Similarity in the generator's own vision space scores correctness at 0.4346–0.4426 AUROC —
   below chance** — while *relevance* scores 0.581–0.598. This reproduces the external-encoder failure
   (SigLIP / PubMedCLIP / BiomedCLIP) **inside the model's own representation space**.

**Positive control** (so the null is about the task, not broken plumbing): dataset identity reads off the
cached image vector at **0.998106**, collapsing to 0.284091 on noise — below the 0.590909 majority
baseline. Features are real and correctly row-aligned.

### The honest caveat, and the one place the hypothesis survives

**"Already present" is not "well used."** My own arithmetic on the round's numbers: the language-side
head sits **0.127824** above the random-pick floor (0.804084 vs the measured floor **0.676260**,
`genframe_data.random_pick()`), and the image accounts for **0.024035 of that — 18.8%**. So the image is
a real but **minority** contributor, and destroying it costs less than a fifth of the head's skill.

Two independent facts say the *resolution* of that image signal is a live lever even though *injection*
is not:

- **Laterality remains the weakest stratum at 0.613043**, against 0.817186 on short non-laterality items,
  and **nothing in this round moved it.**
- The VRAM round found the verifier degrades **monotonically** as the image is shrunk — sel_eff
  0.818402 → 0.803944 (cap320) → 0.791837 (cap80), candidate AUROC 0.876769 → 0.871792 → 0.864212,
  guardrail dirty on 3/3 cells at cap80, and the **≤3-word-gold stratum significantly down at
  −0.0339 [−0.0627, −0.0052]** [`_vram_levers_parts/verifier_grid.json`]. A verifier that did not use the
  image would not care what resolution it was shown at.

⇒ The open question is **not** *how to get the vision signal into the verifier*. It is **why a
representation that demonstrably contains it still cannot separate "Right." from "Left."** That is a
different attack and must not be conflated with the one this round ran.

---

## 5. Guardrails

**Row 3 (the best 7B-only point): 0 of 8 cells damaged versus always-7B — by construction, not by luck.**
On the 5 MCQ cells the chosen arm **is** the always-7B vector element-wise (max abs deviation 0, verified
in V3). On the 3 open cells the frozen selector strictly improves on 7B greedy: SLAKE_open **+0.041861**,
VQA_RAD_open **+0.045000**, PATH_VQA_open **+0.066667** (my derivation). No seed-noise caveat is needed:
nothing is fitted per cell at deployment time (cross-fit seed sd **0.0**).

**The guardrail that binds is the one against always-32B-direct: 7 of 8 cells are below it.** Only
PATH_VQA_open is above (**+0.014667**). That asymmetry *is* the result.

**Vision arms: guardrail FAILS on the primary arm, within seed noise.** Seed-matched 10-seed ensembles,
L_Vmean vs L: slake_open 0.871252 vs 0.873016 (**down 0.0018**), vqa_rad_open 0.730159 vs 0.722222 (up),
pathvqa_open 0.765161 vs 0.761290 (up) ⇒ `guardrail_clean_vs_L = false`. The slake_open regression is far
inside the bar's own per-seed sd (0.005835) and **the arm has no pooled gain to protect in the first
place**, so this is reported, not treated as a finding. No vision arm is recommended for deployment.

**VRAM levers: the guardrail separates the two levers cleanly.** Quantisation arms stay at 1/3 cells worse
with the other two at or above control; the resolution ladder walks **0 → 2/3 → 3/3** as the cap drops.
vqa_rad_open (n=200, the laterality-heavy cell) moves first and hardest under resolution — consistent
with the visual-grounding reading. Contested stratum (≥2 distinct candidates, ~1.6× more sensitive)
tracks the same ordering: nf4 −0.0041, bf16@cap320 −0.0247, bf16@cap80 −0.0453.

---

## 6. Corrections I am making this round (rule 7)

**C1 — the VRAM headline understates the unified pipeline by ~28%, and the "×smaller" ratio by 2.2×.**
`vram_levers_2026-08-12.json:headline_round2` reports the nf4 unified arm at **(d) 8.5905 GiB** and
**8.45× smaller** than always-32B-direct. That (d) is the peak over **12 open-text items**, whose worst
driver carries **1,200 vision tokens**. The 8-cell macro also requires the **MCQ leg**, whose worst item
(MedXpert MM-1561) carries **46,816 vision tokens** and whose nf4 footprint is, in the same file's own
`bnb_quantised_unified_arm.arms.nf4.mcq_by_cap.cap16384`, **(b) 8.0563 / (c) 9.5957 / (d) 10.9792 GiB**.

> **The honest whole-suite figures are: bf16 23.4206 GiB (measured, S1) = 3.10× smaller; nf4 10.9792 GiB
> (reconstructed) = 6.61× smaller.** Not 18.7644 / 3.87×, and not 8.5905 / 8.45×.

The *fit* verdicts survive (10.9792 < 11.63 GiB usable on a 12 GB board), but with only **0.65 GiB** of
headroom — so the safe claim is a **16 GB** board, not a 12 GB one. The brief's own "measured 3.9× VRAM
advantage of the 7B side" is likewise the open-arm-only ratio; **use 3.10×.** This is the same failure
mode CLAUDE.md §9.6 names: a number correct in its own scope, quoted outside it.

**C2 — the "short-answer failure mode" in the brief does not hold on the clean pool.** It came from the
**contaminated** `pooled4` verifier on a different n=1,064 pool. On the clean disjoint verifier over
n=2,345 the length pattern is **monotone decreasing** — 1-word golds are the verifier's *strongest*
stratum (**0.826396**), 2–3 words 0.713178, 4–8 words 0.500000. The genuinely weak stratum is
**laterality (0.613043)**; laterality items merely happen to be short. **Length was a confounder, not the
mechanism** [`_visverif_parts/causal_null_length.json`].

**C3 — a grader defect inflates one arm and, unpropagated, understates the gap.** The option branch is
graded `pick == gold` while the baselines go through MedEvalKit's extractor, which reduces a bare `"C:"`
response to the empty string (`utils/utils.py:111-112`) and falls through to difflib similarity. On 6,000
PMC items the two graders disagree on **69** items for the 7B and **94** for the 32B. Consequence: the
fusion arm's PMC win is **+0.0132 [+0.0072, +0.0192] SIG** against the harness grader and
**+0.0030 [−0.0023, +0.0083] NOT SIGNIFICANT** against a repaired one — **the entire apparent win was the
grader** [`unified_pipeline_2026-08-12.json:THE_GRADER_DEFECT...`, `pmcvqa_grader_defect_2026-08-12.json`].
*My arithmetic on the knock-on effect:* under the repaired grader on that subsample the PMC 7B/32B pair is
0.5493333/0.5670 instead of 0.5391667/0.5523333, so the PMC gap widens by **+0.0045** and the macro gap
would move **0.059586 → 0.060148**. **This is NOT propagated** — the macro baselines remain harness-graded
and the whole frontier above is internally consistent on that basis. Flagged so it is not rediscovered as
a discrepancy. **`MedEvalKit/` was not modified.**

**C4 — do not quote `ens8_scaled`, and do not quote the round-1 retracted rows.** Standing from
2026-08-05: `ens8_scaled`'s 0.746× compute is a Weitzman-controller collapse, not a win.
`vram_levers_2026-08-12.json:retracted` lists six round-1 rows that loaded several models per process.

---

## 7. Recommendation — **ship the VRAM/deployability claim, not the accuracy claim**

The user asked to choose between *"beat 32B outright"* and *"match 32B cheaply."* **The evidence says:
neither, as stated — but a third framing is measured, robust, and unclaimed by anyone.**

**Why "beat 32B outright" is not the direction.** The target is **0.0596**. Eight pre-registered attacks
over two prior rounds failed to close **0.0029** — a target **20× smaller**. This round adds **eight more
failures** at the same wall: seven verifier architectures (six concat arms + xattn) landing inside a
**±0.021 seed spread that is a documented field constant across ~27 architectures**, and the
verifier-over-options unification losing significantly on 2 of 4 option cells. And **~64% of the residual
is pure capability**, on cells where the candidate set is provably complete and the coverage wall is
exactly 0 — there is no 7B-side lever there at all. Selection efficiency 0.78–0.81 is not a number that
one more architecture moves; the coverage wall is already **4.5× the selection wall**.

**Why "match 32B cheaply" is nearly won already and is not news.** Row 8 (shipped accuracy-max) is a tie
at **+0.0008 [−0.0022, +0.0037]**, and row 7 reaches **0.655945 at 0.7427× compute**. But both keep the
32B resident, so *cheaply* means FLOPs, not hardware — the box is still a 78 GiB box. And the honest
non-inferiority reading of row 7 is that its CI lower bound (−0.006614) **fails** the pre-registered
tolerance, so it is "not significantly worse", not "certified equal".

**The claim the data actually supports — and it is a good one:**

> **A single 7B model plus a 47.6M-parameter LoRA verifier, with no large model anywhere in the system,
> recovers 32.2% [20.1%, 44.8%] of the 7B→32B accuracy gap and runs in 23.42 GiB — 3.10× less VRAM than
> always-32B-direct's measured 72.60 GiB, and 6.61× less at 4-bit weights (10.98 GiB), which moves the
> deployment from an 80 GB datacentre card to a 24 GB (or 16 GB) board. The cost is a residual −0.0404
> [−0.0523, −0.0284] in macro accuracy, of which ~64% is generator capability and 0% is coverage. VRAM is
> discontinuous in the escalation rate: escalating even 0.47% of items requires ≥84.75 GiB and changes the
> hardware class, and the two mitigations — load-on-demand (139.8 s, 74.4× a forward pass) and a 4-bit
> strong leg (breaks a tie with only 0.000686 of slack) — are both measured and both fail.**

That is a **deployability frontier with a hardware-class discontinuity**, backed by measured VRAM on a
clean card, an exhausted-architecture null with 10 seeds and a pre-registered CV, and a capability
decomposition that says *where* the remaining gap lives. Two of its three legs are **negative results
with mechanisms** — which is this project's established contribution shape (two walls, ~90 negatives),
and the discontinuity claim is genuinely new. It needs **no** win over 32B-direct to be true, and it is
falsifiable, reproducible from frozen artifacts, and directly actionable for a deployer.

**What the paper must not say:** that the 7B-only pipeline matches always-32B-direct (it does not,
−0.0404 significant); that it is *cheaper in FLOPs* (**1.4199×**, it is not); that vision injection helps
(it does not); or that the whole pipeline is 8.45× smaller (**3.10× / 6.61×** — see C1).

---

## 8. Ranked next steps

1. **Generator capability on the 4 low-hanging MCQ cells — the only lever with 0.0207 of macro sitting
   behind it and no 7B-side mechanism at all.** VQA_RAD_closed (−0.0717), PATH_VQA_closed (−0.0482),
   MedXpert (−0.0450), SLAKE_closed (−0.0335). The candidate set is complete, so this is *not* a
   selection problem and no verifier can touch it. Domain-adaptive fine-tuning or distillation from the
   32B on these families is the only thing that moves it. **Generator work outranks verifier work** —
   this round is the fourth independent confirmation.
2. **Laterality as its own attack, framed correctly.** The representation *contains* the image
   (+0.024035, 10/10 seeds) and still scores laterality at **0.613043** vs 0.817186. The structural clue
   is concrete: **the ViT's attention was never adaptable** — all 192 vision LoRA tensors are
   `visual.blocks.*.mlp.*` because Qwen2.5-VL names attention `attn.qkv`/`attn.proj`. Adding those names
   to `target_modules` is a one-line change that gives spatial mixing non-zero capacity for the first
   time. Cheap, well-motivated, and **not** the arm this round refuted. ~30.5% of vqa_rad_open is decided
   by a laterality token.
3. **Close C1 properly: measure the unified nf4 arm's (d) on a clean exclusive card at the MedEvalKit
   default cap, over an item pool that includes MedXpert.** The 10.9792 figure is a reconstruction and
   the 12 GB fit has 0.65 GiB of headroom. This is the load-bearing number of the recommended paper claim
   and it deserves a direct measurement.
4. **Measure 4-bit accuracy on the 5 unmeasured cells, including all 3 open cells.** Row 3b's accuracy is
   currently `not measured` on macro-8; if nf4 is free on the full suite (the open half says
   −0.0017 [−0.0133, +0.0117] on n=600), the headline VRAM number legitimately becomes 10.98 GiB rather
   than 23.42, and 6.61× rather than 3.10×. Note the open-cell NF4 rows scored with
   `use_llm_judge=False` read 0.000 for **both** arms — those are not accuracy and must not be quoted.
5. **A verifier-resolution sweep with a matched control on the full n=2,345 pool.** The n=600 subsample
   found a monotone degradation with a significant hit on the ≤3-word stratum but could not resolve the
   cap320 rung. This is the cheapest measurement that would confirm the visual-grounding reading of the
   laterality failure.
6. **Do NOT spend further compute on vision-injection architectures for the verifier**, on listwise /
   pairwise / set-aware scorers, or on (choice)(why) unification. All measured, all inside the 0.78–0.81
   field constant or significantly worse.

---

## 9. Not measured — stated, not hidden

- **Row 3b's macro-8 accuracy at 4-bit.** The MCQ half (int8wo −0.001333 macro5, 0/5 cells significantly
  worse; int4wo −0.028, 1/5) and the open half (−0.0017 [−0.0133, +0.0117], n=600) sit on **different
  evaluation tracks** (internal 7-cell harness with PMC `test_clean` vs the 3-cell transfer pool), and
  CLAUDE.md forbids cross-multiplying across evaluation contexts. The composite stays unmeasured.
- **The 4-bit 32B's accuracy on 5 of 8 cells**, including all 3 open cells.
- **Arms B0/B of the unified pipeline** (the option-trained and jointly-trained verifiers) never finished
  under shared-GPU contention; arm B stopped at ~1,200/20,728 steps. The leakage-checked 20,728-example
  training set survives at `ckpts/train/lora_verifier_unified_s0/unified_manifest.json`. They owe ≥10
  seeds if ever run. *(Their pre-registered prediction — that they fall short of always-7B on the option
  cells — is unfalsified, not confirmed.)*
- **The open-text generator's resolution** was never swept; only the verifier's. The coverage wall is
  untested against resolution.
- **Rows 4–6 are eval-visible lower bounds** on the escalation needed (the cell subset is chosen on eval).
  A deployable gate needs *more* escalation, not less — item-level cross-fit gating at a 10% budget still
  sits at **−0.0285 [−0.0392, −0.0177]**.
- **No new VRAM measurement was taken by the vision round**, and none is claimed; structurally it adds no
  model at test time (a 3584→256→1 MLP over states the 7B forward already produces).
- **PMC-VQA numbers in the unification arm are on a pre-registered 6,000-item subsample** of `test_2`
  (seed 20260810), not all 33,430.
- **Environment caveat, stated because it widens CIs:** both A100s carried other rounds' jobs all session
  (load average 22; a 403-token 7B forward at 397 ms against ~40 ms idle). Every load waited for free VRAM
  and claimed it atomically; **no other process was killed**.

---

### Reproduce

```bash
cd ~/medvlthinker-imgdiff-compute
python3 src/training_methods/genframe_selector.py        # V1: reload + verify (READ ONLY)
python3 src/cascade_methods/frontier_verify2.py          # V3/V4 the sibling's independent re-derivation
```

⚠️ **Never run `src/training_methods/freeze_selector.py`** — it *rewrites* `ckpts/train/genframe_head_ens8/`,
and a refit is a fresh seed draw. The frozen `.pt` files are the artifact of record, not the recipe.
⚠️ **Never score a visual LoRA under vLLM** — it drops all 192 `visual.*` modules (0.775204 HF vs
0.702997 vLLM). HF transformers only.
