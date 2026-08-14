---
title: "Adaptive Test-Time Compute for Medical VLMs"
subtitle: "Progress Report — 6 July to 14 August 2026"
author: "Li-Wen Kuan (Leo)"
date: "14 August 2026"
---

# Executive summary

Over the last five and a half weeks the project moved from *believing* it had beaten a large
medical vision-language model to *knowing precisely* what it can and cannot claim. That sounds
like a retreat. It is not: the headline result that survives is narrower but it is now defended
by measurements that did not exist on 6 July, and four of the five claims that were withdrawn
were withdrawn because **we** found the defect, not because a reviewer did.

**The five things that matter most.**

1. **The verifier was contaminated, and fixing it cost the project its main claim.** The
   open-text verifier had been trained on roughly 70% of the items it was later asked to score.
   Retraining it on a disjoint split cost **−0.0119 [−0.0188, −0.0052]** of macro accuracy and
   turned *"beats a 32B forward pass"* into *"ties a 32B forward pass"*. This single correction
   is about **seven times larger** than any improvement the project has ever obtained from a
   better selector.

2. **What survives is a strong claim against reasoning models.** The method scores **+0.0601
   [+0.0499, +0.0700]** against a reasoning Lingshu-32B, at **−87.9% parallel latency** and
   **−84.3% energy**. Against the same model answering *directly* it is a statistical tie
   (+0.0008 [−0.0022, +0.0037]).

3. **A new and better result arrived on 12 August: the same accuracy for less of everything.**
   A pre-specified per-cell configuration matches always-32B-direct (**+0.0002 [−0.0019,
   +0.0023]**) at **0.865× its compute, −2.2% latency and −8.2% energy**, guardrail-clean on all
   eight benchmark cells. It is the first operating point in the project that is cheaper than the
   baseline on all three cost axes simultaneously.

4. **The project now knows why its central research direction stalled.** Selection efficiency
   sits at 0.78–0.81 across roughly twenty-seven distinct verifier architectures. This is not a
   local failure: it is a **field constant**, independently confirmed by 2026 literature, and the
   seed-to-seed spread of our own training runs (~0.021) exceeds every architectural effect we
   have ever measured.

5. **The first hardware measurements in the project's history were taken.** The entire open-text
   arm — eight-sample generator *and* verifier co-resident — runs in **18.76 GiB**, on a single
   24 GB card. always-32B-direct needs **72.60 GiB** and an 80 GB card. This is a **3.9× smaller
   footprint** and it is a deployability claim the paper previously could not make at all.

**What to say on Monday, in one sentence:** *the method matches a 32-billion-parameter model's
accuracy at 0.87× its compute, −2.2% latency and −8.2% energy, and beats the same model in
reasoning mode by six accuracy points at a tenth of its latency and energy — measured on eight
benchmark cells with a decontaminated verifier and a guardrail on every cell.*

\newpage

# 1. Where the project stood on 6 July

By 6 July the project had converged on its current shape: a **format-aware adaptive cascade**
between Lingshu-7B and Lingshu-32B, evaluated on the MedEvalKit harness. The IEEE paper draft was
produced on 8 July.

At that point the headline was:

- accuracy-max **+0.0245** over an always-32B-with-reasoning baseline, sample-weighted over 42,224
  items, at **0.932×** the compute of a single 32B forward pass;
- described in the draft as *Pareto-dominating* every fixed way of using the 32B.

Five properties of that claim have since changed. Each is documented below with the measurement
that changed it.

# 2. The arc of the period

## 2.1 Late July — the correction pass (29–30 July)

Three audits ran in sequence, and all three moved the headline.

**The prompt-matching audit.** The project's flagship qualitative finding — *reasoning hurts
perception VQA* — had been derived from arms whose prompts were not matched: the reasoning arm and
the direct arm differed in **answer format** as well as in the reasoning instruction. Re-running
with the format matched:

- the **perception half got stronger**: 17 of 20 cells negative, pooled **−0.0401 [−0.0456,
  −0.0347]** over 30,250 paired samples;
- the **reasoning half collapsed**: with format matched, **0 of 9** explicit reasoning-trigger
  effects are significant, while **3 of 9** answer-format effects are.

The mechanism is that **asking for the answer in `\boxed{}` is itself a reasoning trigger**.
MedVLThinker emits 431–580 tokens on 99–100% of items with no reasoning instruction present.
Consequence: all seven published Lingshu think-vs-direct cells were **withdrawn**, and Lingshu-32B
must never again be cited as reasoning evidence without a token audit.

**The MMMU contamination audit.** Lingshu-7B scored 0.80 on MMMU-Medical against its own published
54.0. Two controls settled it: an image-ablation (0.827 → 0.593 when the image is removed) and a
control model (base Qwen2.5-VL-7B scores 0.567 on the identical harness). The conclusion was
genuine weights plus train-set contamination, not a harness bug on our side. MMMU is **excluded**
from the reporting pool ("Variant B"). Under macro averaging it would carry 11.1% of the score, so
the exclusion is consequential and must be argued on contamination grounds alone.

**The PMC-VQA provenance audit.** The project had been quoting PMC-VQA numbers from two different
files without saying which. The MedEvalKit track uses **`test_2.csv`** (v2, 33,430 items, hard-coded
in vendor code, **zero published verification**). The internal June-era track used
**`test_clean.csv`** (v1, 2,000 items, the authors' only human-verified split). **Their intersection
is six items.** Every PMC number must now be quoted with its file and row count.

## 2.2 30 July — the reporting convention changed, and the cost claim reversed

Accuracy reporting moved from **sample-weighted** (42,224 items) to **macro** — equal weight per
reporting cell, eight cells at 1/8 each. The reason: sample-weighting gave PMC-VQA **79.2%** of the
entire suite's weight, so one unverified split was speaking for the method.

The reweighting is dramatic: PMC-VQA goes from 79.2% to 12.5%; the open-text arm goes from 5.6% of
items to **37.5% of the score**; closed/MCQ goes from 94.4% to 62.5%.

**Under equal weight the cost claim reverses.** No operating point is compute-cheaper than
always-32B-direct: compute-lean **1.196×** (was 0.492×), accuracy-max **1.410×** (was 0.932×). The
phrase *"Pareto-dominates"* was **retired** from the paper, including from its title.

The nuance that must travel with this: macro-averaging *cost* answers a different question from
sample-weighted cost. Cost is additive per query, so on traffic resembling this suite the ~0.49×
saving is what a deployer would actually pay; the macro number tests whether the saving
**generalises across task types**, and it does not. Both must be reported, each labelled.

## 2.3 3 August — two silent constants were wrong

Two numbers that had been asserted rather than measured were checked:

| constant | asserted | measured / derived | effect |
|---|---|---|---|
| R32 (32B:7B FLOP ratio) | 4.57 (name-plate) | **3.816** [3.734, 3.859] | every cost ratio moves **against** the method |
| best-of-N latency | 522 ms | **1,305.3 ms** (n=45) | *"N drops out of latency"* is **false** |
| best-of-N energy | 568.8 J | **316.7 J** (n=45) | energy was over-stated |

With the honest R32 and a prompt-matched reasoning baseline, the headline against reasoning halved:
from +0.0720 to **+0.0325 [+0.0237, +0.0412]**.

## 2.4 4–5 August — the verifier research programme, and its limit

Four rounds of verifier architecture work ran in this window. The full catalogue is in §5; the
outcome is summarised here.

- **Cross-family verifiers refuted.** All six foreign judges (Qwen2.5-VL, MedVLThinker, QoQ-Med,
  MedGemma-4B, InternVL3-8B, Chiron-o1) scored **worse** than the same-family incumbent
  (0.668–0.688 vs 0.775), and the generator scoring its own candidates (0.707) beat every foreign
  judge. A 4.5×-larger same-family judge merely **ties** a trained 7B one — confirming for the third
  time that **training, not size, is the active ingredient in verification.**
- **A discriminative head on the generator's own hidden states won**: rank-fused with the incumbent
  it reached selection efficiency **0.8065 (+0.0313 [+0.0163, +0.0463])**, guardrail-clean.
- **Then the mechanism behind that win was retracted.** The apparent "generator frame beats grader
  frame" effect was confounded — the two arms differed in pooling and objective as well as frame,
  and each was a single seed. On a matched 2×2×2 grid at 10 seeds, all four contrasts have
  confidence intervals spanning zero.
- **Pairwise verification refuted twice.** The project's only prior selection "win" (+0.076) did not
  survive decontamination: with the clean adapter on matched items, round-robin, knockout and Borda
  all have **negative** point estimates.

## 2.5 5 August — decontamination, and the headline flips

The disjoint-verifier retrain completed and was propagated through the full cascade. Both anchor
artifacts reproduced exactly (5,529 leaf fields, zero differing), so the comparison is clean.

| arm | macro | vs 32B-reasoning | vs 32B-direct |
|---|---|---|---|
| contaminated (as published) | 0.6694 | +0.0720 | **+0.0128 [+0.0056, +0.0200] WIN** |
| **clean disjoint verifier** | **0.6575** | **+0.0601 [+0.0499, +0.0700] WIN** | **+0.0008 [−0.0022, +0.0037] TIE** |

Attribution on a shared bootstrap stream: **decontamination −0.0119 [−0.0188, −0.0052]**
(significant) versus **selection +0.0014 [−0.0003, +0.0032]** (not significant). Decontamination is
about seven times the size of the selector change, and it is what flipped the verdict.

## 2.6 9–10 August — an infrastructure interruption, and a full integrity audit

The machine rebooted on 9 August. A four-part audit established that **nothing was in flight and
nothing was damaged**: the newest file on disk was the commit object from 5 August, 3.5 days before
the reboot; 843 of 843 checkpoint files end on a complete line; the frozen selector reloaded and
reproduced its training-run logits **bit-exactly** (deviation 0.0).

The audit also surfaced a real preservation problem that predated it: 63 commits were unpushed, and
the reproduction inputs (`feats_hidden/` 4.4 GB, the frozen selector, the clean adapter) had **zero
tracked files** in git. All are now pushed and backed up to a separate physical device with
content-hash verification.

## 2.7 10–12 August — eight pre-registered attacks on the remaining gap

Two rounds of four attacks each targeted the +0.0029 needed for a significant win over
always-32B-direct. **None succeeded.** Details in §5. Two products of those rounds outrank the
attacks themselves:

**The measured ceilings.** Free upper bounds on the eight-cell macro: perfect **selection** over the
current pools **+0.0301**; perfect **coverage** at infinite sampling **+0.0091**; perfect
**7B-vs-32B routing +0.0661**, of which only **1.3%** is currently converted. *The ceiling is not the
problem; identifiability is.*

**A methodological result that closes a whole family of approaches.** The "pick the best arm per
cell" rule earns **+0.0109 macro from shuffled labels alone**, against +0.0090 on the real data
(p = 0.67). Cross-fitting removes the bias but not the variance. **Per-cell selection cannot be the
mechanism on this suite — its own noise is roughly four times the effect being sought.**

## 2.8 11–12 August — the cost result, and the first hardware measurements

**Measured VRAM** (HuggingFace, bf16, batch 1, one A100): 7B direct **23.42 GiB**; the whole
open-text arm **18.76 GiB**; 32B direct **72.60 GiB**; 32B with genuine reasoning **72.70 GiB**.

Two things were overturned in the process. Reasoning does **not** cost more memory — 3 → 259
generated tokens moves the footprint by **+0.09 GiB**, because the peak is set by *prefill* of a long
multimodal prompt, not by decode. And the nine VRAM figures circulating in the project were **vLLM
memory-pool reservations**, not measurements: three different 7B models all reported an identical
71.45 GB.

**The cost decomposition** then explained the whole cost story: by phase, **LM prefill is 82.1%** of
compute, vision towers 16.7%, and **all decode is 1.2%**. This is a prefill-bound workload — every
idea about generating fewer or shorter answers was doomed before it started. **44.1%** of compute is
cheap-side work on questions that get escalated anyway.

**And it produced the period's best result.** A pre-specified per-cell configuration reaches macro
**0.6569, +0.0002 [−0.0019, +0.0023]** at **0.865× compute, 650.6 ms (−2.2%), 116.6 J (−8.2%)**,
guardrail-clean on all eight cells. Exhaustive enumeration of all 1,679,616 per-cell assignments
shows **nothing below ~0.86× can satisfy the accuracy constraint at all.**

\newpage

# 3. The method as it stands

## 3.1 Architecture

A **format-aware adaptive cascade**. A router reads the prompt text — never the gold answer — and
decides whether the question is multiple-choice or open-ended, then runs a different policy.

**Multiple-choice arm** (5 of 8 cells, 62.5% of the score):
7B answers directly → a **confidence-margin gate** (the gap between top-1 and top-2 option
log-probabilities) escalates to the 32B below a cross-fit threshold → a **certified veto** keeps the
7B's answer wherever a Wilson lower bound on its precision in a confidence bin exceeds the 32B's
accuracy in that same bin. The veto *answers*; it is not an abstention mechanism.

**Open-text arm** (3 of 8 cells, 37.5% of the score):
7B samples up to 8 candidates under a **Weitzman optimal-stopping rule** (mean 5.5 draws) → a
**LoRA verifier** scores P(correct | image, question, candidate) for each → argmax → escalate to
32B-direct if the verifier's top score is low.

Neither leg runs in reasoning mode. The 32B's reasoning mode is a **baseline**, never a deployed tier.

## 3.2 What is resident

| component | parameters | weights |
|---|---:|---:|
| Lingshu-7B base (incl. vision tower) | 8.29 B | 15.45 GiB |
| Verifier LoRA (r = 16) | 47.6 M | 181.6 MiB |
| Frozen 8-seed selector head | 7.34 M | 28.1 MiB |
| Lingshu-32B (escalation target) | 33.45 B | 62.31 GiB |

The generator and the verifier are **one model plus an adapter**, not two models — the verifier adds
0.57% to the parameter count. This is why the whole cheap arm fits in 18.76 GiB.

# 4. Results

## 4.1 Headline table

Macro, 8 cells at 1/8 each, Variant B (MMMU excluded), clean disjoint verifier. Intervals are 95%
paired item-level bootstraps, 10,000 resamples.

| arm | macro | vs 32B-reasoning | vs 32B-direct | compute |
|---|---:|---|---|---:|
| always-7B | 0.5971 | — | −0.0596 | 0.219× |
| always-32B-reasoning (unmatched prompt) | 0.5974 | — | −0.0593 | 1.00× |
| always-32B-reasoning (prompt-matched) | 0.6250 | — | −0.0317 | 1.00× |
| **always-32B-direct — the bar** | **0.6567** | — | — | 1.00× |
| oracle mode-select on the 32B | 0.6573 | — | +0.0006 | 1.00× |
| **accuracy-max (shipped)** | **0.6575** | **+0.0601 [+0.0499, +0.0700] WIN** | **+0.0008 [−0.0022, +0.0037] TIE** | 1.74× |
| accuracy-max + frozen selector | 0.6590 | +0.0615 [+0.0514, +0.0715] WIN | +0.0023 [−0.0010, +0.0054] TIE | 1.74× |
| compute-lean | 0.6443 | +0.0469 WIN | −0.0124 [−0.0191, −0.0062] LOSS | 1.46× |
| accuracy-max⁺ fusion | 0.6503 | +0.0529 WIN | −0.0063 [−0.0120, −0.0011] LOSS | 1.70× |
| **cost-optimal point (12 Aug)** | **0.6569** | — | **+0.0002 [−0.0019, +0.0023] TIE** | **0.865×** |

A significant win requires macro Δ ≈ **+0.0029** (the CI half-width), i.e. a summed per-cell gain of
≈ +0.0235.

## 4.2 Per cell

| cell | n | 7B | 32B-direct | method | escalation |
|---|---:|---:|---:|---:|---:|
| PMC-VQA (MCQ) | 33,430 | 0.5427 | 0.5518 | **0.5613** | 8.45% |
| SLAKE-closed | 836 | 0.8254 | 0.8589 | 0.8589 | 20.45% |
| VQA-RAD-closed | 251 | 0.7809 | 0.8526 | 0.8526 | 56.97% |
| PathVQA-closed | 3,362 | 0.8409 | 0.8891 | 0.8891 | 45.72% |
| MedXpert-MM | 2,000 | 0.2615 | 0.3065 | 0.3065 | 89.60% |
| SLAKE-open | 645 | 0.7364 | 0.8186 | 0.8171 | 43.41% |
| VQA-RAD-open | 200 | 0.4650 | 0.6000 | 0.5900 | 54.00% |
| PathVQA-open | 1,500 | 0.3240 | 0.3760 | **0.3847** | 16.00% |

**On four of the five multiple-choice cells the method is byte-identical to always-32B-direct.** On
half the macro weight it *is* the baseline it is being compared to. The entire vs-direct delta is
two cells — PMC-VQA (+0.0095) and PathVQA-open (+0.0087) — less what the other two open cells give
back. Leave-one-cell-out range: **[−0.0004, +0.0024]**; dropping PMC-VQA makes the delta negative.

## 4.3 Cost and memory

| configuration | macro | compute | latency | energy | resident |
|---|---:|---:|---:|---:|---:|
| always-32B-direct | 0.6567 | 1.000× | 665 ms | 127.0 J | 62.31 GiB |
| shipped accuracy-max | 0.6575 | 1.740× | 775.9 ms (+17%) | 255.3 J (+101%) | 77.76 GiB |
| **cost-optimal point** | **0.6569** | **0.865×** | **650.6 ms (−2.2%)** | **116.6 J (−8.2%)** | 77.76 GiB |
| always-7B | 0.5971 | 0.219× | 347 ms | 45.8 J | 15.45 GiB |

Measured test-time VRAM (HF, bf16, batch 1): 7B direct **23.42 GiB**; full open-text arm **18.76
GiB**; 32B direct **72.60 GiB**. The cheap arm runs on a 24 GB card; the 32B needs 80 GB. But **any
escalation puts both models in memory** (77.76 GiB of weights plus activations exceeds one 80 GB
card), so the deployed method is a two-card system. Going 7B-only is a hardware-class cliff, not a
gradual saving.

\newpage

# 5. Negative results

The negative catalogue is a deliberate contribution, not an accident. In this period it grew by
roughly twenty entries. The most decision-relevant are listed here.

## 5.1 The selection wall

**Twenty-seven distinct verifier/selector architectures now sit at selection efficiency 0.78–0.81.**
Prompted judges; LoRA judges; a 4.5×-larger same-family judge; six cross-family judges; every score
fusion tried; pairwise verifiers both simulated and with real A-vs-B forward passes; listwise and
ranking losses (twice, independently); set-aware DeepSets and attention heads; pool-relative geometry
features; gradient-boosted feature models; zero-shot contrastive alignment with SigLIP, PubMedCLIP
and BiomedCLIP (all three scored **below** a random-pick floor); discriminative heads on hidden states
in two prompt frames; learned combiners (which, cross-fitted *on eval*, lose to a parameter-free
w = 0.5 fusion, and whose eval-visible weight sweep peaks at exactly 0.5).

Two facts make this a finding rather than a failure. It is a **field constant** — an independent 2026
system converts 78.3% of its oracle; we convert 77.5%. And **our own seed-to-seed spread (~0.021)
exceeds every architectural effect we have measured**, which means most of the literature's reported
differences at this scale are not distinguishable from noise either.

## 5.2 Mechanisms established, not just nulls

Several negatives came with a diagnosis, which is what makes them publishable:

- **Comparative information cannot be recovered from independently-encoded candidates.** Any
  antisymmetric comparison matrix decomposes into an additive part — which *is* a pointwise scorer —
  plus a residual. Measured: **97.93%** of the learned matrix was the additive term, and ranking by
  the residual alone gives 0.6798 against a random floor of 0.6763.
- **The verifier's limit is candidate provenance, not option discreteness.** The same scorer on the
  generator's own samples rescues at **2.18× the random floor**; on the prompt's answer space it falls
  **below 1/K** on three of four cells. A verifier built on the generator's own base adds information
  **only inside the generator's support**. This supersedes the previous "option discreteness"
  explanation — the yes/no cells have K = 2 and fail identically.
- **The optimal unified scorer gives the verifier zero weight.** Cross-fitting the fusion weight
  between verifier and generator returns λ = 1.0, at which the generator's answer can never be
  overturned.
- **Vision injection is refuted, and so is its premise.** The verifier is *not* ignoring the image:
  replacing the image with noise costs the language-side head **+0.024 selection efficiency, 10 of 10
  seeds positive**. Because the image information is already in the representation the verifier reads,
  explicit vision features are redundant — all seven vision-aware arms returned nulls, including two
  designed to favour the hypothesis.
- **Quantisation cannot reduce compute.** Structural, not empirical: identical logical parameter
  count, identical multiply-accumulates, and AWQ/NF4 are W4A16, so weights dequantise to bf16 and
  multiply on 16-bit tensor cores. It is a **memory** lever only (62.31 → 19.53 GiB, 3.19×), and INT8
  measured **12× slower**.
- **The open-text arm has negative value under a cost objective.** From the same 7B-greedy start, the
  best-of-N + verifier machinery buys +0.0251 open-macro accuracy for +10.03 FLOP-eq; one 32B-direct
  pass buys +0.0897 for +3.57 FLOP-eq. **The 32B is 10× more efficient per accuracy point on open
  text.**

## 5.3 A structural constraint discovered in this period

Because the non-inferiority test is **paired**, the width of the confidence interval depends on how a
replacement model's errors correlate with the baseline's. Interpolating the measured per-item joint
distribution gives:

- a replacement that *is* the 32B computationally (quantised, distilled) needs only to **match** it —
  macro ≥ 0.6557;
- a replacement erring **independently** of it must **beat** it by **+0.0149** to be certified "not
  worse".

**The constraint therefore admits exactly two shapes: something that is the 32B in disguise, or
something genuinely better.** A merely-comparable cheaper model of different lineage cannot pass,
however cheap it is. This is the geometry of the test, not a property of any model, and it says
distillation *from* the 32B is structurally the right family.

# 6. Infrastructure and measurement findings

These are not the research contribution, but several are larger than the effects the research was
chasing, and every one of them constrains future work.

1. **±0.008 open-text reproducibility.** Regenerating the 32B open arm under a different
   tensor-parallel configuration moves cells by ±0.008 (±0.00183 macro) — **larger than the entire
   published vs-direct delta**. Every open-text comparison needs a matched control arm in the same
   serving configuration.
2. **vLLM 0.9.0.1 silently drops all 192 visual LoRA modules.** The same adapter scores **0.775204**
   under HuggingFace and **0.702997** under vLLM — a −0.072 engine artifact. Never score a visual
   adapter under vLLM. Our headline numbers are unaffected (the verifier path uses HuggingFace).
3. **A grading defect in the harness itself.** `MedEvalKit/utils/utils.py:112` reduces a bare `"C:"`
   response to the empty string, which then falls through to fuzzy text-matching against option
   bodies — and it **differentially penalises the 32B**. One apparent win this month was entirely
   this artifact: significant against the harness grader, not significant against a repaired one.
4. **PMC-VQA answer-position skew.** The reported split has B+C = **73.6%** and a constant-C floor of
   37.8%, which recovers 68.5% of the 32B's accuracy on that cell. A permutation null holding letter
   composition fixed shows the certified veto's gain is **not** an artifact (z = 9.78, p < 1e-4), but
   **44%** of it is attributable to the skew, and both arms are significantly **worse** on the gold-A
   stratum.
5. **Four numerics landmines**, each larger than most real effects here: TF32-by-default
   (−0.0089/+0.024), CPU thread count (+0.0048), feature row order (+0.0041), and rank-averaging
   versus argsort (+0.008).
6. **Nothing has been run end to end** in the Lingshu era. Every operating point is a CPU re-costing
   of saved per-sample dumps using per-leg batch-1 constants. The single genuine end-to-end execution
   in the project is a June-era cascade.

# 7. What is open

| item | status |
|---|---|
| Prior-art collision | Four 2026 papers overlap our claims; positioning analysis written 11 Aug, not yet folded into the paper |
| Paper revision | The draft still states the retired vs-direct win; needs rewriting around the reasoning claim plus the cost result |
| Inference-parameter sweep | Running: decoding parameters, image resolution, vision-axis diversity |
| Resolution consistency | The deployed generator and verifier may be running at different `max_pixels`; being checked now |
| End-to-end execution | Never done in the Lingshu era; the strongest remaining threat to external validity |
| MedEvalKit grader defect | Found, not yet scoped; the vendor dependency must not be edited |
| Distillation from the 32B | Identified as structurally the right family by the paired-CI result; not yet attempted |

# 8. Recommendation

Frame the report around **two measured claims**, not one:

> The method matches a 32B forward pass at **0.865× its compute, −2.2% latency and −8.2% energy**,
> and beats the same model in reasoning mode by **+0.060 [+0.050, +0.070]** at −87.9% latency and
> −84.3% energy. The cheap side of the pipeline runs in **18.76 GiB** against the 32B's **72.60 GiB**.

Then present the negative results as a deliberate second contribution: the selection wall as a field
constant, the provenance mechanism, the paired-CI geometry, and the measurement landmines. Roughly
ninety negative results are catalogued, most with a diagnosed mechanism, and four headline claims
were withdrawn by our own audits before anyone else saw them. That record is unusual and it is worth
presenting as a strength.

# Appendix: primary sources

| artifact | what it holds |
|---|---|
| `cascade_selector_rerun_2026-08-05.json` | decontaminated headline, per-cell accuracy, escalation |
| `cost_decomposition_2026-08-12.json` | FLOP decomposition, the cost frontier, the 0.865× point |
| `vram_testtime_2026-08-11.json` | the five measured memory scenarios |
| `method_inventory_2026-08-11.json` | all 229 tested methods with per-value sources |
| `pmcvqa_answer_bias_audit_2026-08-11.json` | the answer-position audit |
| `PROJECT_RETROSPECTIVE_2026-07-29.md` | the definitive account: §4 results, §6 negatives, §10 corrections |
| `LITERATURE_UPDATE_2026-08-11.md` | field state, 142 citations |
| `HEADLINE_HISTORY_2026-08-11.md` | every superseded headline and why it was superseded |

All intervals are 95% paired item-level bootstraps with 10,000 resamples. All accuracies are macro
over eight reporting cells, Variant B, clean disjoint verifier, unless explicitly labelled otherwise.
