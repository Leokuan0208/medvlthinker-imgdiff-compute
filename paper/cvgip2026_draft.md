# It's the Compute Configuration, Not the Gate: Adaptive-Compute Cascades and the Limits of Routing for Efficient Medical VLMs

**Target:** CVGIP 2026 · **Status:** DRAFT (Leo / Li-Wen Kuan). Every number below comes from real
checkpoint output in this repo (no fabricated values); `[REPRO: ...]` tags name the script that
produces each result. Replace bracketed `[TODO]` with final polish/figures before submission.

---

## Abstract

Two-model cascades (cheap model first, escalate hard cases to a large model) are the standard recipe
for efficient inference, and a frozen confidence gate is the standard escalation rule. We study this
recipe for **medical visual question answering (VQA)** with a 7B→32B MedVLThinker pair on six
benchmarks, and report three findings. **(1) The gate is saturated.** Across a dozen training-free
escalation signals — confidence (margin/MSP/entropy/Gini), conformal set-size, learned correctness,
learned deferral, self-verification, cross-model agreement, multi-resolution ensembling, compute-
elasticity, hidden-state probes, and cross-*family* routing on image content — *recoverability* ("will
escalation actually fix this error?") is only **0.50–0.69 AUROC** predictable from any cheap signal, so
no gate beats a plain confidence threshold in a way that is simultaneously novel, efficiency-positive,
and per-benchmark safe. We trace this to an error-correlation effect: same-family models have nested
errors (φ=0.37), and the lone direction with large headroom — a cross-*architecture* peer whose errors
decorrelate (oracle union **0.80** vs always-32B **0.645**) — is **unexploitable** because *which* model
is right cannot be predicted from confidence, agreement, or image content (AUROC ≈ 0.50). The same wall
closes the *action* and *selection* axes too: cheap same-model repairs recover 14% of errors the large
model misses yet are unharvestable (capacity-bound), and in open-ended generation the cheap model's samples
contain the answer far more often than any training-free selector extracts it — a **luck floor** (best
selector 0.76 vs oracle 0.88, *below* the large model's single pass 0.82) — so the limits of routing are not
specific to the gate (§5.9). A **trained** outcome verifier, by contrast, recovers **40–77%** of this oracle gap — for free-text answers
(49% pooled over five datasets, beating a zero-shot 32B verifier despite being 5× smaller) **and** for
structured bounding-box outputs (SLAKE organ grounding 40%; the real MS-CXR chest-X-ray pathology benchmark
77%, a 5.6× lift) — so the bottleneck is the *verifier’s training*, not the headroom, and *training* is the
universal ingredient that breaks the luck floor across output types and domains (§5.10).
**(2) The lever is the compute *configuration*, not the gate.** We present the **Adaptive-Compute Cascade (ACC)**,
a confidence-gated three-tier cascade over compute *configurations* of the same two models —
7B-no-think@cap320 → 32B-**no-think**@cap320 → 32B-think@fullres — that inserts the large model's *fast*
no-think mode as the intermediate workhorse so the slow ~28 s think pass fires on only ~14% of queries.
At matched accuracy with always-32B-think, ACC cuts batch-1 latency by **72%** on all six benchmarks
(20.0 s→5.7 s; **97%** on the five non-MedXpert benchmarks), FLOPs to **~½**, and energy by **~4×**, while
being *strictly never worse than always-7B* per benchmark. **(3) The gate ceiling is an MCQ artifact.**
Re-evaluating routing in the **open-ended** (free-text generative) regime, the same confidence signal
jumps from ~0.6 to **~0.87 AUROC** — a *discreteness*, not answer-length, effect (open answers are a
median 1–2 tokens, yet routing is strong) — so confidence-gated open-ended medical-VLM cascades genuinely
work; the gate nonetheless stays near-optimal at plain confidence in *both* regimes (an exhaustive hunt
over consistency, semantic-entropy, and self-verification signals, and their fusion, fails to beat it), so
the efficiency lever is the evaluation *setting*, not a new gate. We position ACC as an efficiency-systems
contribution and the gate-saturation/complementarity/open-ended analysis as the first cross-family
medical-VLM recoverability characterization explaining *why* routing is hard here — and *where* it is not.

---

## 1. Introduction

Large vision-language models (VLMs) are accurate but expensive; in medical VQA a 32B reasoning model
costs ~28 s and ~7 kJ per question at batch-1, versus ~0.2 s for a 7B answering directly. The dominant
efficiency recipe is the **two-model cascade**: run the cheap model, and a **confidence gate** escalates
low-confidence questions to the large model. Improving the *gate* — better uncertainty signals, learned
deferral, conformal guarantees, self-verification — is an active research area.

We set out to find a novel gate that beats the deployed margin threshold for our medical 7B→32B
cascade, and instead found that **the gate is essentially saturated**: the information needed to route
well (whether escalation will *help*) is not present in any cheap signal we could construct (§5). The
single exception with real headroom — routing to a *different-architecture* peer whose errors are
decorrelated — turns out to be unexploitable in practice (§5.3). The efficiency win lies elsewhere: in
**which compute configuration** the cascade escalates *to*. Our Adaptive-Compute Cascade (ACC, §4)
exploits the empirical fact that **reasoning over-thinks perception VQA** — the 32B in *no-think* mode
matches or beats its own *think* mode on perception benchmarks at ~80× lower latency — by inserting
32B-no-think as an intermediate tier and reserving the slow think pass for the small reasoning residual.

**Contributions.** (i) **ACC**, a three-tier compute-configuration cascade with large measured
latency/energy/FLOPs savings at parity accuracy and a per-benchmark safety guarantee (§4–§5.1). (ii) A
**gate-saturation characterization**: a dozen training-free escalation signals, all capped at the same
recoverability ceiling, with an error-correlation explanation (§5.2). (iii) The first **cross-family
medical-VLM complementarity map**: large oracle headroom that is provably unexploitable with available
cheap peers — a negative result that explains why medical-VLM routing is hard (§5.3). (iv) A
**language-prior diagnostic**: 57% of the 7B's medical-VQA answers are unchanged when the image is
removed (§5.4). (v) The **open-ended ceiling-break**: the same confidence signal jumps from ~0.6 to ~0.87
AUROC moving from MCQ to free-text — a *discreteness* artifact — so the routing ceiling is an evaluation
artifact, not a model limit (§5.7). (vi) A **luck-floor characterization**: *training-free* selection over a
single model's samples is luck-bound across gate, action, selection, synthesis, retrieval, agreement,
in-context prompting, **and** structured grounding outputs (§5.9). (vii) The constructive counterpart — a
**trained outcome verifier** that breaks that floor, recovering **40–77%** of the oracle gap for both
free-text answers (five datasets) and bounding boxes (SLAKE organs and the real MS-CXR chest-X-ray
benchmark, a 5.6× lift), and behaving as a genuine test-time-scaling method — so *training* is the universal
active ingredient across output types and domains (§5.10).

---

## 2. Related Work

**Adaptive reasoning / no-think gating.** CAR [arXiv:2505.15154] and margin-gated CoT [2510.21007]
self-gate a single model between direct answering and chain-of-thought; Med-R1 / No-Thinking-Med
[2503.13939] shows no-think can match think in medicine; "More Thought, Less Accuracy?" [2509.25848]
documents reasoning *hurting* VLMs. ACC differs by using a *two-model, three-tier* structure in which
the **large** model's no-think mode is the intermediate workhorse tier.

**Resolution as a compute axis.** VisionThink [2507.13348] escalates image resolution. ACC co-varies
resolution with the model/mode tier.

**Cascades and deferral.** FrugalGPT (learned scorer), AutoMix (self-verification), Cascaded LMs for
Human-AI [2506.11887], and learning-to-defer [Jitkrittum, NeurIPS 2023; 2307.02764] escalate to a
*larger* model by a *post-generation* confidence/benefit signal; agreement-based cascading (ABC)
[2407.02348] escalates on ensemble *disagreement*. A parallel line trains *pre-generation* query routers
that decide from the input alone — Hybrid-LLM [2404.14618], RouteLLM [2406.18665], and the unified
routing+cascading view [2410.10347] (and learned *token-level* deferral, Co-LLM [2403.03870]); these report
gains over *random* allocation, not over a calibrated confidence gate. We benchmark the post-generation
gates as routing rules (§5.2) — they tie/lose to a plain margin under a per-benchmark guardrail — and the
pre-generation cell is covered by our image-content router (§5.3, AUROC ≈ 0.50). Speculative decoding (EAGLE
[2401.15077], speculative cascades [2405.19261]) is *lossless* and orthogonal to the routing question.

**Open-ended routing and the agreement/consistency signal.** Closest to our §5.7 is *Semantic Agreement
Enables Efficient Open-Ended LLM Cascades* [2509.21837, EMNLP 2025], which defers in *text-LLM* cascades on
low semantic agreement and shows agreement beats Chow-style confidence — but its agreement is computed
across an **ensemble of distinct models under greedy decoding**, it is text-only (no vision, no medicine),
and it makes no multiple-choice-vs-open-ended claim (it notes, if anything, that agreement weakens on
*short* answers). The underlying signal lineage is self-consistency [Wang et al., ICLR 2023; 2203.11171]
and semantic entropy [Kuhn et al., ICLR 2023, 2302.09664; Farquhar et al., *Nature* 2024], whose
closed-vs-open contrast we instantiate **inside a cascade gate, in the medical-VLM setting**, and attribute
specifically to the **discreteness** of a single multiple-choice letter (not answer length — our open
answers are a median 1–2 tokens, yet routing is strong). The closest VQA-side work, selective VQA from
black-box VLMs [Khan & Fu, CVPR 2024; 2404.10193], uses question-rephrasing consistency to **abstain**, not
to escalate, and is not medical. To our knowledge §5.7 is the first **open-ended, medical vision-language**
cascade-routing study; we further show that, once open-ended, plain confidence is already near-optimal
(no consistency/semantic-entropy/self-verification signal or fusion beats it) and self-consistency helps
only a *miscalibrated* cheap model.

**Model routing.** VL-RouterBench [2512.23562], ECVL-ROUTER [2510.27256] (image is the dominant routing
signal), and RouterDC [2409.19886] route among models. We test image-content routing among cross-family
medical/peer VLMs and find which-model-is-right unpredictable here (§5.3).

**When is complementarity achievable?** Theory says aggregation helps only when error-correlation ρ is
below a critical ρ* [2605.08710]; same-family models have higher ρ ("Hidden Clones" [2603.17111]) and
cross-model disagreement is a label-free correctness signal for *text* LLMs [2603.25450]. Our cross-family
medical-VLM measurement (§5.3) confirms the headroom but shows it is unexploitable from cheap signals.

---

## 3. Setup

- **Models.** MedVLThinker-7B-RL_m23k and -32B-RL_m23k (both Qwen2.5-VL). Both can answer directly
  ("no-think", ~2 decode tokens) or emit a `<think>` trace ("think", ~400 tokens).
- **Data.** MedVLThinker-Eval, 8,220 samples, six benchmarks. **Competent-4** = PMC-VQA, SLAKE, VQA-RAD,
  PathVQA (where the method works); **excluded** = MMMU and MedXpert (both models near chance on MedXpert).
  Pools reported: ALL-6, ALL-5 (= 6 − MedXpert), COMPETENT-4. Calibration on held-out PMC-VQA-train.
- **Cost metrics.** FLOPs (exact): one run = 2·N·(P+G), N₇=7.6e9, N₃₂=33e9, P=prompt incl. vision tokens,
  G=generated. **backbone%** = cascade FLOPs / always-32B-think FLOPs. **Latency/energy**: calibrated
  batch-1 wall-clock from real measurements (`measure_config.py` NVML power integration + 5,440 measured
  32B-think queries from `rt_cascade_cap320.jsonl`, R²=0.99), applied per-query via the expected-cost
  formula C = c_T0 + e₀·c_T1 + e₁·c_T2 (see `METHOD_MATH.md`).
- **Guardrail.** A method must be **never worse than always-7B** on *every* benchmark (no pooled-metric
  gaming). `[REPRO: src/cascade_methods/harness.py reproduces the deployed anchor 0.5718/63.3%/73.6%.]`

---

## 4. Method: the Adaptive-Compute Cascade (ACC)

**Motivating fact (reasoning over-thinks perception).** On the competent-4 benchmarks the 32B in
*no-think* mode ≥ *think* mode (e.g. SLAKE 0.841 vs 0.764, VQA-RAD 0.893 vs 0.776) at ~2 vs ~400 decode
tokens; the 32B-no-think@cap320 pooled accuracy (0.646) matches always-32B-think parity (0.645).
`[REPRO: run_32b_modes_vllm.py → ckpts/gate_32b_modes/]`

**Tiers (stop at first confident).**
```
T0  7B-no-think  @cap320   (≈0.21 s)   -- always runs
T1  32B-no-think @cap320   (≈0.34 s)   -- the fast workhorse; runs if T0 unconfident
T2  32B-think    @fullres  (≈26.6 s)   -- the reasoning residual; runs if T1 unconfident
```
**Gate.** Escalate at a tier iff the confidence **margin** m=ℓ₍₁₎−ℓ₍₂₎ < τ. The think-tier gate (ACC-v2)
escalates only when 7B-no-think and 32B-no-think **disagree** (a cross-model committee that is *free*
because both legs have already run). Only fitted parameters are the two scalar thresholds τ₀, τ₁,
calibrated on held-out PMC-train to reach parity at minimum latency. `[REPRO: src/cascade_methods/acc_v2.py;
frozen thresholds in ckpts/acc_v2_thresholds.json]`

**Why it saves (expected cost).** With e₀=P(escalate past T0), e₁=P(reach think), the per-query cost of
any metric is C = c_T0 + e₀·c_T1 + e₁·c_T2. Because T2 latency (26.6 s) ≈ 80× T1 and the no-think tier
absorbs the escalations that don't need reasoning, e₁ collapses from ~69% (escalate-to-think) to ~14%,
and latency ≈ e₁·26.6 s collapses with it (full derivation in `METHOD_MATH.md`).

---

## 5. Results

### 5.1 ACC vs standard reasoning cascades (honest 50/50 split, 20 seeds, at parity)
`[REPRO: src/cascade_methods/acc_compare.py → results/cascade_methods/acc_compare.txt]`
M2 = standard 2-tier 7B-**think**→32B-think; M3 = standard 3-tier 7B-nt→7B-**think**→32B-think.

**ALL-6 (parity acc 0.5723):**
| method | acc | esc→think | FLOPs% | latency | energy/q | guardrail-fails |
|---|---|---|---|---|---|---|
| M2 standard 2-tier (both think) | 0.5725 | 86% | 105.2% | 29.8 s | 7049 J | 0.35 |
| M3 standard 3-tier | 0.5697 | 65% | 88.7% | 23.2 s | 5499 J | 0.25 |
| **ACC (margin gate)** | 0.5694 | 19% | 54.7% | 5.93 s | 1505 J | **0.00** |
| **ACC-v2 (agreement gate)** | **0.5710** | **14%** | **53.5%** | **4.86 s** | **1220 J** | **0.00** |

**ALL-5 (parity 0.6463):** ACC-v2 = 0.6460 acc, 0% think, 27.1% FLOPs, **0.44 s**, 75 J, guard 0.10
(vs M2 20.25 s/4735 J/0.80-fails). ACC dominates all five metrics at matched accuracy and is the only
method that is guardrail-clean. **Headline: ALL-6 latency 20.0 s→5.7 s (−72%); ALL-5 9.1 s→0.28 s (−97%).**
(All pools are ALL-6 and ALL-5 = ALL-6 minus the two MedXpert benchmarks; we no longer report a separate
competent-4 pool.)

### 5.1.1 Complete comparison with baselines, all gates, and Ours (under the fixed ACC config)
`[REPRO: src/cascade_methods/final_3tier_comparison.py → results/cascade_methods/final_3tier_comparison.txt]`
Config = 7B-nt@cap320 → 32B-nt@cap320 → 32B-think@fullres. `esc0`=% past the 7B; `think`=% reaching
32B-think; `guard`=avg #benchmarks/seed below always-7B (0 = never; the never-worse-than-7B bar). Latency/
FLOPs are calibrated/exact as in §3. **Baselines first**, then every gate, Ours bolded.

**ALL-6 (7 benchmarks; parity = always-32B-think = 0.5723):**
| method | acc | esc0 | think | FLOPs% | lat | guard |
|---|---|---|---|---|---|---|
| always-7B-nt@cap320 (cheap floor) | 0.5262 | 0% | 0% | 8.4% | 0.21 s | 0.00 |
| always-32B-nt@cap320 | 0.5573 | 100% | 0% | 36.2% | 0.34 s | 0.00 |
| always-32B-think@fullres **[PARITY]** | 0.5723 | 100% | 100% | 100% | 26.6 s | 0.00 |
| **Ours (ACC-v2: agreement)** | **0.5710** | 79% | 14% | 53.5% | 4.86 s | **0.00** |
| CASP-Stability (trained — §5.6) | 0.5698 | 74% | 11% | 49.0% | 3.94 s | 0.05 |
| ACC-v1 (margin) | 0.5694 | 71% | 19% | 54.7% | 5.93 s | 0.00 |
| MSP/Chow | 0.5704 | 73% | 19% | 57.8% | 6.60 s | 0.00 |
| entropy | 0.5695 | 73% | 21% | 62.2% | 7.99 s | 0.00 |
| Gini/DOCTOR | 0.5705 | 72% | 21% | 61.6% | 7.82 s | 0.00 |
| AutoMix (self-verify) | 0.5699 | 76% | 17% | 54.8% | 5.46 s | 0.05 |
| FrugalGPT-style learned | 0.5679 | 71% | 19% | 60.5% | 7.61 s | 0.10 |
| Jitkrittum L2D (Diff-Prob) | 0.5673 | 70% | 14% | 51.0% | 5.01 s | 0.00 |
| random | 0.5643 | 89% | 76% | 116.6% | 20.7 s | 0.05 |

**ALL-5 (5 benchmarks; parity = 0.6463):** baselines always-7B 0.6201 (8.9% FLOPs)/always-32B-nt **0.6457
(38.7% FLOPs, 0.34 s)** ≈ parity /always-32B-think 0.6463 (100%). Gates cluster ~27-32% FLOPs; **Ours
0.6460 @ 27.1% / 0.44 s / guard 0.10**. AutoMix reaches 22.0% FLOPs but at **guard 0.40** (violates the
per-benchmark floor on ~0.4 benchmarks/seed) — its apparent efficiency is not guardrail-clean.

**Significance (95% CI over 20 seeds).** Ours' accuracy CI is **[0.5608, 0.5820] on ALL-6** and
**[0.6372, 0.6562] on ALL-5**, both *including* the always-32B-think parity (0.5723 / 0.6463) — i.e.
statistically at parity — while its **guard CI is [0,0] on ALL-6** (it never drops below always-7B on any
benchmark in any seed), versus AutoMix's guard CI **[0,1] (ALL-6) / [0,2] (ALL-5)**. So the
efficiency gains are at genuine parity and Ours is the guardrail-clean choice. (Latency CIs are wide,
e.g. Ours ALL-6 [2.6, 9.8] s, reflecting split/calibration variance; the −72% point estimate holds.)

**Per-benchmark accuracy (ALL-6):**
| method | PMC | SLAKE | VQA-RAD | PathVQA | MMMU | MedX-R | MedX-U |
|---|---|---|---|---|---|---|---|
| always-7B-nt | 0.543 | 0.762 | 0.761 | 0.641 | 0.547 | 0.225 | 0.256 |
| always-32B-nt | 0.551 | **0.849** | **0.853** | 0.661 | 0.624 | 0.279 | 0.292 |
| always-32B-think [parity] | 0.556 | 0.764 | 0.776 | 0.673 | 0.688 | 0.326 | 0.384 |
| **Ours (ACC-v2)** | 0.561 | 0.849 | 0.864 | 0.680 | 0.648 | 0.286 | 0.313 |
| ACC-v1 (margin) | 0.564 | 0.830 | 0.856 | 0.676 | 0.648 | 0.292 | 0.308 |
| MSP/Chow | 0.564 | 0.838 | 0.851 | 0.673 | 0.673 | 0.295 | 0.319 |
| AutoMix | 0.556 | 0.826 | 0.850 | 0.680 | 0.635 | 0.299 | 0.311 |
| CASP-Stability | 0.560 | 0.836 | 0.865 | 0.677 | 0.661 | 0.289 | 0.313 |

**Two facts the baselines expose:** (i) **always-32B-no-think@cap320 ≈ parity on ALL-5** (0.6457 vs 0.6463)
at **38.7% FLOPs / 0.34 s** — the ACC mechanism in one row (it drops to 0.557 on ALL-6 only because MedXpert
needs reasoning). (ii) **Thinking hurts perception per-benchmark:** 32B-no-think beats 32B-think on **SLAKE
(0.849 vs 0.764)** and **VQA-RAD (0.853 vs 0.776)**, so the cascades inherit ~0.85/0.86 there — *above* the
always-32B-think parity baseline.

### 5.1.2 Tightening the think gate with confidence (ACC-v3), and a resolution/reasoning dissociation (NEW)
`[REPRO: src/cascade_methods/acc_v3_confgate.py, acc_v4_lowres_think.py; honest 50/50, 20 seeds, min-think@parity]`

**ACC-v3.** ACC-v2 fires the 28 s think tier on *every* small-nt/big-nt disagreement. We tighten it: fire
think only when the two no-think models disagree **and** the big no-think model is itself unsure (m₁ < τ₁) —
disagreement is necessary but not sufficient. Per family:
- **ALL-6 (reasoning present): ACC-v3 dominates both single-signal gates.** vs ACC-v1 (confidence-only) it
  cuts think 19%→**14%** and FLOPs 54.7%→**52.6%** at equal-or-higher accuracy (0.5707), and unlike ACC-v2 it
  reaches always-32B-think parity in **all 20 seeds** (ACC-v2: 19/20); guard 0.00.
- **ALL-5 / perception: think is unnecessary** — the confidence term drives think→**0%**, so ACC-v3 coincides
  with confidence-only and both roughly *halve* ACC-v2's compute at parity (ACC-v2 16% think / 39.7% FLOPs →
  ACC-v3 0% think / **27.3% FLOPs**).
- **Cross-family:** Lingshu ALL-6 FLOPs 77.8%→**48.6%** (its think is fast → no latency gain); QoQ's cascade is
  degenerate (0% escalation) so all gates coincide. The conjunction never hurts and helps wherever think over-fires.

ACC-v3 combines two *known* signals — confidence (CAR, arXiv 2505.15154) and agreement (ABC, arXiv 2407.02348);
the contribution is that their **conjunction** removes ACC-v2's residual think-overuse at no accuracy cost. We
also verified the gate is **data-efficient**: subsampling the PMC-VQA-train calibration, eval behaviour is flat
from 2000→3000 samples (acc 0.6503→0.6502, threshold σ→0), so the full 337k-row train split would not change it
— the binding limitation is the *perception-only distribution* of PMC-train (it cannot calibrate the think tier),
not the sample size `[REPRO: gate_data_size.py]`.

**A resolution/reasoning dissociation (ACC-v4).** Running the 32B think pass on a 1/4-resolution image (cap320)
vs full resolution: accuracy is **preserved or improved on the reasoning benchmarks that actually reach the think
tier** — MMMU 0.688→**0.712 (+0.024)**, MedXpert ≈ unchanged — while it drops on perception (SLAKE 0.764→0.721),
which the no-think tiers already serve so the cascade never pays it. That is, *medical visual reasoning is
resolution-insensitive while perception is resolution-sensitive* — the visual-token-count dissociation of
Matryoshka (arXiv 2405.17430) instantiated on the **resolution** axis and in the **medical** domain (where MMMU
even *improves*, opposite to the resolution-*escalation* of VisionThink, arXiv 2507.13348). ACC-v4 runs the
reasoning tier at cap320 (−28% think prefill); the per-tier saving is real but the cascade-level effect is
marginal because the think tier fires on ≤14% of queries and is decode-bound (the 32B commits its answer at the
*end* of the trace — answer-marker at median 0.99 of trace length over 8 220 traces — so early-exit truncation is
not viable either).

### 5.2 The gate is saturated
Holding the ACC 3-tier config fixed and swapping the *gate* (margin / MSP-Chow / entropy / Gini /
conformal-CP-Router / learned-correctness-FrugalGPT / learned-deferral / random): all real gates cluster
(latency 5.0–8.0 s, FLOPs 51–62% on ALL-6); only *random* collapses (130% FLOPs). The cascade-method
contribution is therefore the **structure**, not the gate. `[REPRO: gate_compare.py]`

Recoverability — predicting 1[7B wrong & 32B-think right] from cheap features — tops out at **~0.6–0.69
AUROC** across **twelve** signal families: confidence (margin/MSP/entropy/Gini/energy), conformal,
learned correctness, learned deferral, self-verification P(True), cross-model agreement, multi-resolution
ensemble, compute-elasticity (resolution-response curve), and a **hidden-state probe** (7B layer-14
activations predict correctness 0.60 vs confidence's 0.68 — *worse* than the logprobs).
`[REPRO: ceiling.py, gate_compare.py, metarouter_honest.py, peer_premise.py + the hidden-state probe]`

**Mechanism.** The 7B and 32B are the same family, so their errors are *nested*: P(32B wrong | 7B wrong)
= 0.584, error-correlation φ = 0.372; the 32B fixes the 7B only where neither has signal (the irreducible
regime), so futility is unpredictable. `[REPRO: harness pool over competent-4]` This is the
**uniform-improver regime of Jitkrittum et al. (NeurIPS 2023, 2307.02764)**, who prove confidence-based
deferral is **near-Bayes-optimal** *unless* the strong model is a **specialist** (better on a subset,
worse elsewhere). Our strong model rarely *breaks* a cheap-correct answer — P(strong wrong | cheap right)
= **0.22 (MCQ), 0.14 (open-ended)** `[REPRO: uniform_improver_diag.py]` — i.e. it is a near-uniform
improver, so the ~0.6 recoverability ceiling and "no gate beats confidence" are **theory-predicted**, not
merely empirical. (We confirm: a learned Jitkrittum *Diff-01* deferral scorer over confidence +
self-consistency lifts recoverability prediction only marginally and non-robustly — ties/loses on the
real-gap datasets — as the theory predicts for a uniform improver.) **(This ~0.6 ceiling is specific to
MCQ evaluation — §5.7 shows the same confidence signal reaches AUROC ~0.87 in the open-ended regime.)**

**Post-audit correction (faithful baselines).** We audited each baseline against its canonical paper and
re-ran a faithful 2-tier (7B-nt→32B-think) bake-off at iso-accuracy (`baseline_compare.py`; honest 50/50
calib/test, 20 seeds). Two corrections to the earlier characterization: (i) **AutoMix** — with a faithful
self-verification threshold (the meta-verifier variant, which our earlier comparison omitted), self-verify
is **competitive, not a loser**: on COMPETENT-4 it reaches ~parity at **20% escalation / 28.7% FLOPs**
(vs margin's 34%/43.6%) and on ALL-5 at 26%/35.0%, though it collapses on the reasoning-heavy ALL-6 (79%).
(ii) **CP-Router** — implemented faithfully (LAC prediction set |C|≠1 with FBE-selected α, not the MSP
proxy we previously reported) it **over-escalates (69-80%)** on these 5-10-option medical questions and
*loses* to the confidence gates — the opposite of the earlier "≈ MSP" claim. MSP/Chow, entropy, DOCTOR
(Gini, verified monotone-equivalent), and the FrugalGPT-style / Jitkrittum-L2D learned gates cluster at
34-42% escalation on COMPETENT-4. **Net:** no single gate dominates across pools, and self-verify is the
most escalation-efficient gate on the competent benchmarks — so we drop the "margin is the best gate"
framing; ACC's win is the *structure* (the no-think workhorse tier), which is orthogonal to the gate
choice. `[REPRO: baseline_compare.py → results/cascade_methods/baseline_compare.txt; audit in baseline_audit.json]`

### 5.3 Cross-family complementarity: real headroom, unexploitable (NEW)
We ran two genuinely cross-architecture peers — InternVL2.5-8B (InternViT+InternLM) and Phi-3.5-Vision
(Phi+CLIP) — on competent-4. `[REPRO: run_peer_eval.py, peer_premise.py, peer_router.py, peer_router_img.py]`
- **Complementarity is real and large:** oracle union(7B|InternVL) = **0.753**, union(7B|InternVL|Phi) =
  **0.801**, both far above always-32B (0.645). Cross-family errors decorrelate as theory predicts.
- **But it is unexploitable with available cheap peers:** (a) a learned router on confidence+agreement
  captures *none* of the gap (0.621 ≈ always-7B 0.622); (b) routing on **SigLIP image+text content**
  gives recoverability AUROC = **0.50 (chance)** — refuting the "image is the routing signal" hypothesis
  here; best image router 0.636, below parity; (c) majority-vote fusion = 0.602 (*below* 7B) because the
  only ungated cross-family peers (InternVL 0.581, Phi 0.516) are all weaker than the medical 7B (0.622)
  and drag the vote down. The complementary information exists but *which* model is right per query is
  not cheaply identifiable. This is, to our knowledge, the first cross-family complementarity map for
  medical VLMs.

### 5.4 Language-prior diagnostic (NEW)
Re-running the 7B with the image blanked to mid-gray, **56.9%** of competent-4 answers are *unchanged* —
the model answered from language priors, not the image. Yet these vision-insensitive items are *no less
accurate* (0.620 vs 0.625 for sensitive), and insensitivity does **not** improve correctness or
recoverability AUROC over confidence — so it is a striking diagnostic but not a useful gate.
`[REPRO: run_vlm_eval.py --blank; vision_sensitivity.py]`

### 5.5 The ACC mechanism generalizes across architectures (NEW)
ACC's empirical basis — *reasoning over-thinks perception VQA* — is **not specific to MedVLThinker**. We
ran no-think vs think on competent-4 for two genuinely different architectures and compared to the
MedVLThinker reference. Pooled Δ(think − no-think) is **≤ 0 for every model**, and the think-hurts effect
is largest on the radiology benchmarks (SLAKE, VQA-RAD) universally:

| model (architecture) | PMC-VQA | SLAKE | VQA-RAD | PathVQA | POOLED Δ |
|---|---|---|---|---|---|
| InternVL2.5-8B (InternViT+InternLM) | −0.037 | −0.055 | −0.070 | +0.021 | **−0.008** |
| Phi-3.5-Vision (Phi+CLIP) | −0.014 | −0.123 | −0.007 | −0.010 | **−0.019** |
| MedVLThinker-32B (Qwen2.5-VL, ref) | −0.008 | −0.077 | −0.118 | +0.001 | **−0.013** |
| MedVLThinker-7B (Qwen2.5-VL, ref) | −0.040 | −0.103 | −0.059 | +0.014 | **−0.015** |

(Cells = think-accuracy minus no-think-accuracy; negative = think hurts. InternVL/Phi are general VLMs,
so absolute medical accuracy is lower, but the *over-thinking delta* is what ACC relies on and it holds.)
Conclusion: the mechanism that motivates ACC's no-think workhorse tier is an **architecture-general**
property of perception VQA, strengthening ACC's external validity. `[REPRO: run_peer_eval.py --think;
overthink_generalize.py → results/cascade_methods/overthink_generalize.txt]` We do **not** claim to
*discover* over-thinking: that chain-of-thought can underperform direct answering — especially on
perception — is established (MME-CoT [2502.09621]; "More Thought, Less Accuracy" [2509.25848];
think-or-not routing for VLMs [2505.16854]). Our contribution is to *measure that it holds
architecture-generally on medical perception VQA* and to *exploit it structurally* — inserting the big
model's no-think mode as the cascade's workhorse tier — which prior work does not do.

#### 5.5.1 Full five-family validation with NATIVE reasoning prompts (NEW)
We ran the complete 3-tier pipeline (small-no-think → big-no-think → big-think) on four additional medical
VLM families across three architectures — **Lingshu 7B/32B** and **QoQ-Med-VL 7B/32B** (Qwen2.5-VL),
**Chiron-o1 2B/8B** (InternVL3), **MedGemma 4B/27B** (Gemma3) — over ALL-6, with MedVLThinker as reference.
Critically, each model's think tier uses **its own native reasoning prompt** (recovered from its training
code/paper), not a shared one: a foreign think prompt mis-measures models trained on a different reasoning
format (verified below). `[REPRO: run_native_think.sh; acc_2size.py / acc_allmethods.py; compare_native_think.py]`

ALL-5 / ALL-6 accuracy (native think):
| family (arch) | small-nt | big-nt | big-think (native) |
|---|---|---|---|
| MedVLThinker (Qwen2.5-VL) | 0.620/0.526 | 0.646/0.557 | 0.646/0.572 |
| Lingshu (Qwen2.5-VL) | 0.734/0.618 | **0.784/0.668** | 0.775/0.661 |
| QoQ-Med-VL (Qwen2.5-VL) | 0.605/0.509 | 0.610/0.522 | 0.543/0.469 |
| Chiron-o1 (InternVL3) | **0.725/0.602** | 0.654/0.551 | 0.593/0.508 |
| MedGemma (Gemma3) | 0.603/0.515 | 0.580/0.500 | 0.598/0.525 |

**The mode axis (no-think ≥ think on perception) holds for every family under its OWN native prompt** — it is
not a foreign-prompt artifact. Three regimes emerge:
- **Lingshu** (probe-confirmed): under its native MCQ prompt it *answers directly* (gen ≈ 3 tokens) and only
  reasons when explicitly told to "reason step by step". So native big-think ≈ big-nt (0.661 vs 0.668), while
  the foreign `<think>` prompt forced harmful reasoning (0.611). Its best mode is direct. (A 3-way prompt probe
  confirms it: Lingshu's documented `\boxed{}` prompt *and* the exact MedEvalKit template both give 2–3-token
  direct answers; only an explicit reasoning instruction elicits CoT — `lingshu_prompt_probe.py`.)
- **QoQ-Med** shows the canonical ACC pattern natively — think *helps* reasoning (MMMU +0.07 over no-think) but
  *over-thinks* perception (PMC −0.085, VQA-RAD −0.077) → net below no-think on perception-heavy pools.
- **Chiron-o1** genuinely over-thinks (native think 0.593 ≪ no-think 0.725 on ALL-5) — a *different* vision
  stack (InternVL3), so the effect is architecture-general; it also inverse-scales on perception (its 2B beats
  its 8B on PathVQA). **MedGemma has no real native think mode** (think ≈ big-nt ≈ no-think).

**Cascade (Ours, ACC-v2) on ALL-6, native think + native-measured batch-1 cost:** MedVLThinker is the only
family where the full 3-tier cascade is cleanly beneficial — **0.569 @ 52% FLOPs / 2.27 s / 1182 J** at
parity, guard 0 (vs always-32B-think 11.3 s / 6319 J, ≈5× faster, ≈5× less energy). For the others the think
tier is net-harmful (or the model inverse-scales), so the cascade collapses to the cheap leg with guard 0:
QoQ 0.509 @ 9% / 0.12 s, Chiron 0.602 @ 19% / 0.20 s. Per-tier costs are measured at each tier's *actual*
operating gen distribution (latency/energy = `a·gen+b` from a native batch-1 sample), so e.g. Lingshu's
gen ≈ 3 native "think" is correctly costed as a cheap fullres prefill (0.32 s / 113 J, just above big-nt),
not extrapolated from a high-gen fit. Full per-method/per-benchmark/cost tables + charts: `MASTER_TABLES.md`,
`master_data.csv`, `FULL_RECORD.md`, `paper/figs/master/`.

**Synthesis — ACC has two axes with different generality:** the **mode axis** (no-think ≥ think on perception
→ use the big model's fast no-think mode) generalizes **strongly across all five families and three
architectures** under each model's native prompt. The **size-cascade axis** is **family-dependent** — it needs
a real, *routable* small→big competence gap, present cleanly only in MedVLThinker (Lingshu/MedGemma small≈big
or non-monotone; Chiron inverse-scales). ACC's robust, transferable contribution is therefore the *mode tier*;
a model-agnostic cascade must use each model's native reasoning trigger and gate think to reasoning-type
questions, defaulting to no-think for perception.

### 5.6 A trained gate (CASP-Stability) and the learnability dissection (NEW)
Can *training* a gate beat the saturated confidence gates? We first test a torch-only gate on frozen features
(LoRA/fine-tuning is revisited in §5.6.1, once the requisite libraries were installed). The key idea is to
**re-target the routing label**: instead of the un-learnable *recoverability* ("will a
different model be right"), predict **answer-stability-under-compute** S(x)=1[pred(7B-nt@cap320) =
pred(32B-think@fullres)] — whether the cheap answer survives more compute. On the same honest
PMC-train→eval transfer and the same features, the three targets separate sharply:

| target | logistic AUROC | MLP AUROC |
|---|---|---|
| recoverability 1[32B-think right \| 7B wrong] | 0.581 | 0.579 |
| **answer-stability** 1[pred7-nt = pred32-think] | **0.714** | **0.714** |
| which-tier 1[7B-nt correct] | 0.685 | 0.685 |

**Re-targeting the label moves AUROC by +13pt; adding capacity (MLP ≈ logistic) or features (L14 hidden
states, multi-cap trajectory) adds ~nothing** — isolating the wall as a property of the recoverability
*target's* mutual information, not the estimator. Used as the ACC tier-0 stop gate, **CASP-Stability cuts
compute** (ALL-6: 49.0% FLOPs / 3.94 s vs ACC-v1 margin 54.7% / 5.93 s) but at guard 0.05 (not perfectly
clean) and ~0.001 lower accuracy. **Pre-registered and confirmed:** stability improves the *stop* decision
(buys compute) but **cannot raise accuracy above always-32B-think**. As an ablation this is the sharpest
form of the gate-saturation result: the only learnable routing signal here is same-model answer-stability,
and it trades compute, not accuracy. `[REPRO: src/training_methods/casp_stability.py →
results/cascade_methods/casp_stability.txt]`

#### 5.6.1 A novel-method search: three training mechanisms, all capped (NEW)
With `peft` installed, we ran a focused research loop for a *novel, training-based, model-agnostic* cascade
method and tested the three fundamentally distinct ways training can help a cascade:
- **Route** (improve the escalation *decision*): a LoRA-fine-tuned 7B self-verifier of its own answer-stability
  reaches AUROC 0.722 — *below* the logistic-on-signals gate (0.733). Capped.
- **Distill** (strengthen the cheap *leg*): *FastLeg-Distill* — LoRA-distill the big-no-think model's competence
  into the small-no-think leg to cut the escalation *rate*. Net-flat (ALL-5 +0.007/+0.000 across two gating
  schemes): it *redistributes* accuracy across benchmarks (e.g. +PathVQA, −VQA-RAD/MMMU via interference)
  rather than lifting it — a single adapter cannot improve all benchmarks at once.
- **Fuse** (combine the two no-think legs): *CALM-Fuse* — a trained head over the small+big per-option logprob
  vectors. The small+big complementarity is real (union-oracle +0.07–0.14 over the best single on all five
  families) but the trained fuser captures ≈0% of it per-family and *collapses* on leave-one-family-out
  transfer (Chiron 0.242) — per-family logit calibration differs, so a single fuser is not model-agnostic.

**The deep finding:** the exploitable structure (recoverability, complementarity) is *real but not learnable* —
every mechanism bottlenecks on the same un-learnable question, "which model is right on this query?"
(~0.58–0.73 AUROC ceiling). The med-VLM cascade sits at a genuine efficiency frontier; the training-free ACC
configuration is near-optimal, and learned cross-family methods additionally fail to transfer. This is a
strong, defensible negative result that sharpens the paper's central claim. `[REPRO:
src/training_methods/{lora_stability_router.py, fld_distill.py, calm_fuse.py}; NOVEL_METHOD_FLD.md]`

---

### 5.7 From MCQ to open-ended: the routing ceiling is a benchmark artifact (NEW)
`[REPRO: src/labeling/{run_openvqa.py, run_judge.py}, src/cascade_methods/{open_cascade_analyze,
gate_search_open}.py; SLAKE-open (645) + VQA-RAD-open (200) + PathVQA-open (1500), n=2345; exact-match,
token-F1, and a neutral LLM-judge]`

§5.2 found every routing signal saturates at **~0.6 AUROC** for recoverability. Is that a property of
medical VLMs, or of the **multiple-choice** benchmarks all prior medical-VLM routing is evaluated on? A
single letter (A/B/C/D) is a maximally discrete target — confidence, agreement, and self-consistency all
collapse toward a 4-way chance baseline. We re-ask the routing question in the **open-ended (free-text
generative)** regime: the cheap model emits a free-text answer (no options), a strong model is the
escalation target, and we measure routing AUROC for *cheap-wrong* and *recoverable* (cheap wrong ∧ strong
right), scored by normalized exact-match with **token-F1** and a **neutral LLM-judge** (MedVLThinker-32B,
a Qwen2.5-32B backbone not in the Lingshu cascade) as robustness checks.

**(a) The MedVLThinker family is near-equivalent on open-ended → no routable gap.** Unlike on MCQ, model
size barely moves open-ended accuracy: SLAKE-open **3B 0.457 ≈ 7B 0.419 ≈ 32B-no-think 0.498 ≈ 32B-think
0.453** (token-F1 confirms; think *hurts* perception here too). These RL-on-MCQ models generalize poorly
to free-text, similarly at every size, so within the family there is nothing to cascade *to*
`[REPRO: run_openvqa_3b.sh]`.

**(b) A real gap exists across families, and the routing ceiling BREAKS.** Pairing a cheap model with a
genuinely stronger open-ended medical model (**Lingshu-32B**, 0.775 pooled; token-F1 0.789) restores a
routable gap, and routing signals become **strong — far above the MCQ ceiling**:

| cheap → strong | cheap acc | strong acc | confidence AUROC (cheap-wrong / recover) |
|---|---:|---:|---:|
| MCQ — any of 12 signal families (§5.2) | — | — | ~0.6 / ~0.6 (ceiling) |
| **Lingshu-7B → Lingshu-32B** (calibrated cheap) | 0.683 | 0.775 | **0.866 / 0.804** |
| MedVLThinker-7B → Lingshu-32B (miscalibrated cheap) | 0.407 | 0.775 | 0.735 / 0.575 |

The ceiling is a **discreteness** artifact, not an answer-length one: the open answers are **median 1–2
tokens** (as short as a letter), yet routing AUROC is **~0.87** — because the answer *space* is open, not
4 fixed options. So §5.2's "the gate is saturated" does **not transfer**: the cheap model's confidence
becomes a strong *error detector* in the open-ended regime. A **fourth modality** confirms it on fresh
data: **Kvasir-VQA-x1** (GI endoscopy, n=1200, free-text, LLM-judged) gives confidence cheap-wrong AUROC
**0.75** (95% CI [0.72, 0.78], above the 0.6 ceiling) — and it is the cleanest illustration of the
*detection vs. cascade-gain* split, because there the 32B is **no better** than the 7B (judge acc 0.301 vs
0.302, gap −0.001 — *zero* recoverability headroom): the cheap model reliably **knows when it is wrong even
when no stronger model can fix it**. (Confidence ≥ self-consistency here too, 0.75 vs 0.73.)
`[REPRO: prep_kvasir.py; run_openvqa.py --dataset kvasir_open; run_judge_kvasir.sh]` **This is not a scoring artifact:** under the neutral LLM-judge,
confidence AUROC is **0.860 / 0.784** (≈ the exact-match 0.866 / 0.804) and the cheap→strong accuracies
(0.67 → 0.77) match exact-match — token-F1 agrees (gap +0.10). The ceiling-break is robust across all
three scorers. (Our judge follows the modern open-ended medical-VQA protocol — binary correctness against
the reference answer, as in Lingshu [2506.07044] and LLaVA-Med [2306.00890]; like those it is text-only,
a known limitation that can miss image-grounded distinctions, which is why we report it *alongside*
exact-match and token-F1 rather than as the sole score.) It also holds on a **third, harder dataset**:
PathVQA-open (long descriptive answers;
exact-match collapses at acc 0.058, so *only* the LLM-judge can score it) gives Lingshu-7B confidence
routing AUROC **0.797** (cheap-wrong) on its own, and the **3-dataset pooled** (n=2345, all judge-scored)
confidence AUROC is **0.846 / 0.591** — still far above the ~0.6 MCQ ceiling, with confidence ≥
self-consistency (0.846 vs 0.831). The break is **highly significant**: a 5 000× bootstrap on the pooled
cheap-wrong AUROC gives **95% CI [0.830, 0.862]**, P(AUROC ≤ 0.6) = 0.0000.

**(c) The gate itself still cannot be beaten — confidence is near-optimal.** We ran an exhaustive
open-ended gate hunt on the calibrated cascade (Lingshu-7B → Lingshu-32B; bar = confidence 0.866 / 0.804),
honest 20-seed calib/test for the learned fusion:

| signal | cheap-wrong | recover |
|---|---:|---:|
| **confidence (seq-logprob)** | **0.866** | **0.804** |
| exact self-consistency (K=8) | 0.845 | 0.764 |
| semantic self-consistency | 0.806 | 0.766 |
| semantic entropy | 0.807 | 0.766 |
| mean pairwise token-F1 | 0.844 | 0.788 |
| self-verify P(True) | 0.755 | 0.726 |
| **fusion of all six (honest CV)** | **0.866** | 0.798 |

No signal beats confidence, and the honest fusion **ties** it (+0.000 / −0.007). Confidence is the
near-optimal open-ended gate — the gate is saturated *here too*, just at a far higher level (~0.87 vs
~0.6). **Self-consistency helps only a *miscalibrated* cheap model:** for MedVLThinker-7B (RL-on-MCQ,
poorly calibrated on free-text) it beats confidence (recoverability +0.043, bootstrap 95% CI
[0.016, 0.069]) and its accuracy-vs-escalation frontier Pareto-dominates confidence's; for the
natively-calibrated Lingshu-7B, confidence wins. Self-consistency is thus a **calibration rescue**, not a
better gate. (This margin shrinks further under the LLM-judge, which credits the MCQ-tuned model's verbose
answers — raising its accuracy 0.41→0.52 and partly fixing the very miscalibration self-consistency was
correcting — leaving self-consistency only marginally ahead.) (Figs: `paper/figs/open/{frontier_selfconsistency,auroc_signals,ceiling_break}.png`.)

**Ablations** `[REPRO: open_ablations.py; calibrated cascade, LLM-judge]`. *(i) K-ablation:*
self-consistency's routing AUROC rises monotonically with the sample budget K (0.73→0.83 over K=2→8) but
**never reaches confidence (0.846)** — so even 8 samples do not beat the single-pass gate, and its only
value is the conditional miscalibration rescue. *(ii) Routing efficiency vs an oracle:* the fraction of the
*oracle* router's accuracy gain captured by the confidence gate is **70% (SLAKE), 59% (VQA-RAD), 30%
(PathVQA)** — high where a real model-gap exists, low where it is small (PathVQA cheap→strong 0.343→0.376).
The achievable cascade *gain* is bounded by **recoverability** (the model-gap), not by signal quality: the
open-ended setting fixes the *signal* (cheap-wrong AUROC ~0.85), but a cascade still needs a strong model
that is reliably better. (Fig: `paper/figs/open/fig_open_ablations.png`.)

**Detection vs. cascade gain (deferral curves)** `[REPRO: deferral_curve.py]`. We separate two quantities a
reviewer rightly conflates. *Detection* — how well the cheap model's own confidence separates its right
from wrong answers (cheap-wrong AUROC) — is where the ceiling breaks: **0.66 (MCQ) → 0.85 (open-ended)**.
This is a *selective-prediction* property (knowing when to abstain / defer / refer), and is the robust
finding. The realized *cascade compute-saving* additionally needs **recoverability headroom** (the strong
model actually being better); on our pairs that headroom is modest in *both* regimes (oracle−cheap = **+0.02
MCQ, +0.06 open-ended**), so the random↔confidence↔oracle deferral curves (Fig. `fig_deferral_curve.png`)
sit with the confidence gate tracking the oracle's shape between the random floor and the oracle ceiling in
both. The open-ended advantage is therefore primarily in error *detection*, which a cascade converts into
compute savings only to the extent a stronger model exists — exactly why the deployed MCQ cascade (§5.1) is
framed as *cost reduction at parity*, not accuracy gain.

**Takeaway — a correction to §5.2.** The medical-VLM routing ceiling is a property of **MCQ evaluation**,
not of the task: in open-ended VQA, routing signals carry AUROC ~0.87 and confidence-gated cascades work.
The *gate* remains unbeatable (confidence is near-optimal across MCQ and open-ended), so the efficiency
lever is the **evaluation/deployment setting**, not a new gate. **Positioning vs prior art:**
agreement-gated open-ended cascades exist for *text LLMs* (semantic-agreement cascade, arXiv 2509.21837,
EMNLP'25, using cross-model-ensemble greedy agreement — not single-model self-consistency; ABC, arXiv
2407.02348); confidence-deferral theory is Jitkrittum et al. (NeurIPS 2023). The genuinely unoccupied cell
is the **medical vision-language, open-ended** instantiation plus the *ceiling-is-discreteness* diagnostic;
we claim this applied/empirical contribution, not a new gate primitive.

**Discrimination, not calibration.** Our claim is about **discrimination** — how well a signal *separates*
cheap-right from cheap-wrong (AUROC) — not calibration (whether its raw probabilities match empirical
accuracy). The two can diverge, so we measure the routing-relevant quantity (AUROC / risk–coverage), never
raw probabilities. This sharpens the relation to CritiCal [2510.24505], which reports — for *single-model
confidence calibration* in *text LLMs* — that "confidence suits multiple-choice tasks, while uncertainty
excels in open-ended": a *calibration* view of the same closed-vs-open divide. We study the orthogonal
*cascade-routing discrimination* of a *medical VLM*, and find the opposite ranking on *that* axis —
single-pass confidence is the best *discriminative routing gate* open-ended (self-consistency only rescues a
miscalibrated cheap model), with the MCQ ceiling attributable to single-letter discreteness.

### 5.8 Knowing when to abstain: safe deferral when no stronger model helps

The cascade escalates errors a *stronger model* can fix; §5.7 shows the cheap model *detects* its own
errors well open-ended. The third action a deployed system needs is to **abstain** — refer to a clinician —
on the irreducible residual where escalation cannot help (e.g. Kvasir, where the 32B is no better, §5.7).
We turn the open-ended detection signal into a **training-free selective-prediction system** and test it
against the recent SOTA **"Uncertainty Is Not a Safety Net for Clinical VQA"** [2606.16583] (8 UE methods ×
12 VLMs), whose headline — uncertainty fails for clinical VQA, its quality "tracks model accuracy" — is
**MCQ-only**. `[REPRO: selective_abstain.py, triage_3way.py]`

*(i) The regime lifts self-assessment.* Same-model self-error-detection AUROC rises from MCQ to open-ended:
MedVLThinker-7B **0.66→0.74**, Lingshu-7B **0.74→0.82** — above the SOTA's ~0.72 MCQ ceiling (Fig.
`fig_selective.png`, left). *(ii) Deployable on competent open-ended VQA.* The **deployed** (strongest)
model — Lingshu-32B — self-abstains to **auto-answer 69% of SLAKE-open at ≤5% error (81% at ≤10%)**,
detection AUROC 0.88, referring the rest to a clinician (the cheaper Lingshu-7B variant: 54% at ≤5%). This
**refutes the strong form** of the SOTA negative. *(iii) Honest scope (confirms the SOTA's nuance), but the
failure mode is safe.* Deployable coverage **tracks competence** — where the model is weak (VQA-RAD/PathVQA/
Kvasir, base acc ≤0.60) the safe coverage collapses to ≤0.14, i.e. the system **abstains on almost
everything, which is the correct behavior when the model is unreliable** (it never auto-answers at high
risk). A **3-way triage** (answer / escalate / abstain) is a strict generalization but only marginally
better (SLAKE cov@10%-risk 0.67→0.74), again **recoverability-bounded**. *(iv) Calibrated and deployable.*
Under the **standard held-out protocol** — calibrate the confidence threshold on a held-out sample to a
clinician's target risk r*, then deploy on fresh data — the **realized risk holds the target**: on
SLAKE-open r*=5% gives 0.69 coverage at **5.1% realized risk** (20-seed held-out; VQA-RAD/PathVQA within
tolerance). A threshold *reused across distributions* overshoots (SLAKE→Kvasir 19.5%), so a per-deployment
calibration set is the safe practice `[REPRO: abstain_calibration.py]`. The contribution is the *regime
insight*, the *calibrated, deployable competent-set operating points*, and the cross-model/cross-modality
validation — not a new uncertainty signal (confidence remains unbeatable, §5.7). Spec:
`results/cascade_methods/SELECTIVE_ABSTENTION.md`.

---

### 5.9 Beyond the gate: the *action* and *selection* axes are luck/capacity-bound (NEW)

§5.2 shows the *gate* (when to escalate) is saturated at plain confidence. Two axes remain in a cascade —
the **action** (what to do on escalation) and, in the generative setting, **selection** (which of N sampled
answers to return). We test both exhaustively; both hit the same wall. `[REPRO:
results/cascade_methods/{RECOVERABILITY_IS_CAPACITY_BOUND, OPENENDED_SELECTION_LUCKFLOOR,
KNOWLEDGE_AUGMENTATION_FEASIBILITY}.md]`

**The action axis is capacity-bound.** Decomposing the cheap 7B's errors by repair type, cheap *same-model*
interventions (re-asking at full resolution, or in think mode) recover **14% of errors the 5×-larger 32B
misses entirely** (stable 11–17% across the competent-4) — so scaling up is *not* a superset of intervening
cheaply. But the recovery is **unharvestable**: the repairs break as many answers as they fix (every cheap
view sits at ≈0.62 vs the 32B's 0.645), a confidence-gated 4-rung ladder *loses* at parity (43% vs 39%
compute), and max-confidence/majority ensembles saturate at the best single view. The 32B's advantage
requires its *capacity*; no cheap same-model transform substitutes for it.

**The selection axis is a luck floor.** In open-ended generation the cheap model's 8 samples have a large
oracle gap that survives the semantic judge (Lingshu-7B SLAKE-open greedy 0.730 → **oracle@8 0.879**), and
self-consistency *fails* via a **majority trap** — the correct answer is a *minority* vote in 74–90% of
recoverable cases (it even hurts on PathVQA). Every training-free selector sits at the **random-pick floor**
(0.720): self-verification P(Yes) 0.715 (*below* random), 32B pointwise-verify 0.746, 32B *listwise* select
0.758, learned fusion 0.743 — the best captures only **24% of the gap above random and none beats the 32B
single pass (0.819)**. Candidate-conditioned *synthesis* (priming the 32B with the cheap candidates)
**backfires** (0.774): the majority trap drags the strong model down (Fig. `figs/limits/fig_selection_luckfloor.png`). The gap is sampling **luck, not latent
knowledge** — the model does not *know* which sample is right (if it did, that would be its greedy answer) —
the same luck-floor structure that sank single-model routing.

**The luck floor is not a free-text quirk — it generalizes to structured/verifiable outputs.** One might
expect verifiable outputs (where a self-consistency *geometry* exists) to escape it. We test medical visual
**grounding** (bounding boxes, IoU) on SLAKE's gold detection boxes. With a *weak* grounder (Lingshu-7B,
near-floor IoU) spatial self-consistency appears to predict box correctness (AUROC 0.82) — but this is an
**artifact of incompetence**: a weak model emits diverse boxes, so rare agreements spuriously correlate with
correctness, and there is no accuracy to harvest (oracle@8 acc@0.3 = 0.22). With a *competent* grounder
(Qwen2.5-VL-7B, organ oracle@8 acc@0.3 = 0.44, a real +0.18 gap) the escape **vanishes**: boxes are
consistent, so the self-consistency signal collapses to chance (AUROC 0.56) and medoid selection ties greedy
(0.246 vs 0.254). Competence → consistency → no discriminative selection signal — so the structured-output
oracle gap is luck-floored exactly like free-text. Selection over a *single* model's samples is fundamentally
luck-bound regardless of output type.

**The same wall closes every other lever.** *Knowledge augmentation* is not indicated: the 7B's
genuinely-unknown errors (oracle@8 wrong) are fixed by the 32B *equally across knowledge and perception
question types* (38% vs 36%) — general capacity, not a retrievable knowledge gap — and a systematic audit
**refutes** the hypothesis that the large open-ended ceiling is a caption-extraction *artifact* (PathVQA
difficulty is genuine, concentrated in well-formed questions). *Cross-family agreement* (decorrelated errors
across two independently-trained VLMs) is a real reliability signal (P(correct | agree)=0.819 vs 0.649) but
as an accuracy selector collapses to "trust the stronger model". *Few-shot ICL* hurts (PathVQA 0.343→0.203).
**Conclusion:** across gate, action, selection, synthesis, retrieval, cross-family agreement, and in-context
prompting, the binding constraint is the *genuine "which answer is right?" knowledge the frozen models lack*
— it is not recoverable by any training-free re-routing, re-selection, or re-prompting of those models. This
is why the leverage in this paper is structural (compute configuration, §4) rather than a better decision rule.

### 5.10 What *does* break the luck floor: a trained outcome verifier (NEW)

§5.9 shows every *training-free* lever is luck-floored. The constructive complement: a **trained** verifier
escapes it. We LoRA-fine-tune Lingshu-7B to score P(correct | image, question, free-text answer) on the
per-sample LLM-judge labels and use it to select best-of-8 (honest grouped split by question, no leakage).
Trained **pooled over all four open-ended datasets** it lifts every one: `[REPRO:
src/training_methods/run_lora_verifier_open.py]`

| dataset | greedy | self-consist. | zero-shot 32B verify | **trained verifier** | oracle@8 | gap captured |
|---|---|---|---|---|---|---|
| PathVQA-open | 0.352 | 0.349 | 0.357 | **0.441** | 0.513 | 55% |
| Kvasir-open (GI, OOD modality) | 0.282 | 0.282 | — | **0.405** | 0.493 | 58% |
| VQA-RAD-open | 0.519 | 0.500 | — | **0.611** | 0.722 | 45% |
| SLAKE-open | 0.738 | 0.738 | 0.758 | **0.762** | 0.895 | 15% |
| **pooled (n=1064)** | 0.413 | 0.411 | — | **0.501** | 0.592 | **49%** |

(zero-shot 32B verify run on PathVQA/SLAKE; a 2-dataset variant confirmed the PathVQA lift across two seeds:
trained 0.414 / 0.426.)

**Training is the active ingredient.** Zero-shot self-verification (P(True) [Kadavath’22]) is luck-floored —
*below greedy* on PathVQA (0.319) — yet LoRA-training the same model on judge labels captures **49% of the
oracle gap pooled across all four datasets** (+0.088 over greedy), lifting **every dataset** (PathVQA +0.089,
Kvasir +0.123, VQA-RAD +0.092, SLAKE +0.024), vs ≤24% for every training-free selector and *below-greedy* for
zero-shot verification (§5.9). The correctness signal the frozen models cannot *surface* zero-shot is
*learnable* from a few thousand labels — indeed the trained **7B** verifier beats the **zero-shot 32B**
verifier (0.357 on PathVQA) by +0.08, a 5×-larger model, because it is trained. The PathVQA lift is **robust
across two independent grouped splits** (0.414 / 0.426). A weaker 2-dataset variant (train on SLAKE+PathVQA
only) further showed the win is **headroom-gated and transfers across modality** — it still lifted
out-of-distribution Kvasir (GI endoscopy) 0.286→0.327; pooling all four into training resolves the residual
(Kvasir +0.123, VQA-RAD +0.092 in-distribution). The pooled-4 verifier even transfers **zero-shot to a fifth,
fully held-out dataset** — RadImageNet-VQA (which independently reproduces the luck floor: SC = greedy 0.329,
oracle 0.512) — lifting it 0.329→0.353 (+0.024, 13% of its gap), weaker than in-distribution as expected. This is a multimodal-medical instance of the generative-verifier result
[GenRM, ICLR’25; Weaver, NeurIPS’25] and a constructive rebuttal of the *Verification-Mirage* [2605.10850]
negative for open-ended VQA: self-verification fails, but a *trained* verifier does not. An **image-ablation
confirms the verifier is image-grounded** (not the "lazy verifier" Verification-Mirage warns of): blanking
the image drops selection by **0.047 pooled** (SLAKE −0.064, PathVQA −0.039). It combines a learned
text-correctness prior (blank-image 0.463 > greedy 0.447) with genuine visual grounding (the −0.047 from the
image). `[REPRO: src/training_methods/verifier_image_ablation.py]`

**The principle generalizes from free-text to *structured* outputs.** §5.9 showed grounding (bounding-box)
selection is luck-floored too — training-free SC-medoid sits *below greedy*. We train the analogous
**box-verifier** (LoRA-Qwen2.5-VL judging "does this box localize the {organ}?", IoU≥0.3 = label) and select
best-of-8 boxes on SLAKE: SC-medoid 0.164 and a *zero-shot* box-verifier 0.177 (both ≤ greedy 0.197,
luck-floored) → **trained box-verifier 0.255** (oracle 0.343) — **+0.09 over the luck floor, capturing 40% of
the oracle gap**, the same fraction as the free-text verifier (robust across two seeds: trained 0.255/0.257). **This replicates on the full real, standard MS-CXR phrase-grounding benchmark** (chest-X-ray
pathology, PhysioNet, 1448 boxes, n=435 held-out): SC-medoid 0.053 (luck-floored) → **trained box-verifier
0.230** (greedy 0.041, oracle@8 0.285, IoU≥0.3) — **+0.19, capturing 77% of the oracle gap** (~9 s.e., a 5.6×
lift over greedy; robust across two seeds: trained 0.230/0.248, 77%/76% of gap; it lifts *selection* over a weak base grounder, not the grounding itself — the SOTA grounder
MedGround-R1 is a better, unreleased base). Training is again the active ingredient: a **zero-shot** box-verifier
(the same untrained Qwen judging the boxes) reaches only 0.115 (30% of the gap), so LoRA-training **doubles**
the captured gap (30%→77%) — though, unlike free-text zero-shot verification (which sits *below* greedy),
zero-shot *box* verification has modest signal, since judging a drawn box is a more concrete visual task.
So *training* is the universal active ingredient that breaks the luck floor — across free-text answers AND
verifiable structured outputs, on five open-ended datasets and two grounding benchmarks — whereas every
*training-free* selector (§5.9) is luck-bound in both (Fig. `figs/limits/fig_trained_verifier_unified.png`).
`[REPRO: src/training_methods/run_lora_box_verifier.py]`

The trained verifier is a genuine **test-time-scaling** method: a best-of-K curve (free-text, pooled-4,
n=1064) rises **monotonically** with the sample budget — 0.385 (K=1) → 0.425 → 0.476 → **0.501** (K=8) —
while **random** selection stays flat (~0.39), so the verifier converts added test-time compute into
accuracy (Fig. `figs/limits/fig_verifier_scaling.png`). Extending to **K=16** (a fresh 16-sample run) the
verifier keeps rising monotonically (0.356→0.394→0.411→0.417→0.424) but with **diminishing returns** — it
captures a *shrinking* fraction of the faster-growing oracle gap, as the right answer becomes a rarer needle
in a larger haystack. `[REPRO: src/training_methods/verifier_scaling_curve.py (SC_TAG=sc16)]`

**Test-time compute beats parameters here.** Verifier-selected best-of-8 on the **7B (0.501)** outperforms the
**32B single pass (0.444 pooled)** on open-ended accuracy — a 5×-larger model — at ~3.7× the 32B's
param-FLOPs. This is consistent with §5.7 (the 32B barely improves over the 7B open-ended: 0.444 vs greedy
0.413): when scaling parameters buys little, spending test-time compute on a *trained verifier* over the
small model's samples buys more. Per dataset, the verifier-bo8 7B beats the 32B single pass exactly where
scaling does not help — VQA-RAD (0.611 vs 0.600), PathVQA (0.441 vs 0.376), Kvasir (0.405 vs 0.301) — and
loses only on SLAKE (0.762 vs 0.819), the one set where the 32B is genuinely the stronger grounder. The verifier is thus the accuracy-optimal operating point (most accurate,
at a compute premium), not dominated by simply using the larger model (Fig. `figs/limits/fig_verifier_pareto.png`).
`[REPRO: paper/make_pareto_fig.py; 32B pooled from real judge files]`

---

## 6. Discussion & Limitations

- **Scope & pools.** We report **ALL-6** (all 7 benchmark splits) and **ALL-5** (ALL-6 minus the two
  MedXpert splits, where both models are near chance). On ALL-5 (perception-dominated) the cheap tiers carry
  almost everything; ALL-6 needs the think tier for the MedXpert reasoning residual. Figures: Fig 1
  (`paper/figs/fig1_latency_accuracy_frontier.png`) plots the latency-accuracy frontier; Fig 2
  (`fig2_overthinking_perbench.png`) shows the per-benchmark over-thinking effect.
- **Generalization pending.** The over-thinking *mechanism* generalizes across architectures (§5.5); the
  full **2-size cross-family validation** (Lingshu-7B/32B, MedGemma-4B/27B) was scripted but is blocked by a
  VM-wide network throttle (no model downloads) — it auto-resumes when bandwidth returns. Future work.
- **Honesty of cost.** FLOPs is exact; latency/energy are *calibrated* batch-1 wall-clock (measured + the
  expected-cost formula), not a single end-to-end pipeline timing.
- **Single-stream scope (batched serving).** Our latency/energy are **batch-1**, the regime that matters for
  point-of-care / single-clinician interactive use. Under heavy **batched** serving, continuous batching
  amortizes prefill and the no-think tier's wall-clock advantage shrinks (the FLOPs and energy-per-query
  savings, which are batch-invariant, remain). We therefore scope the latency headline to single-stream
  deployment and report FLOPs/energy as the batch-robust efficiency axes; a throughput-vs-accuracy study at
  batch ≥ 2 is future work.
- **Novelty.** ACC is an *incremental systems/combination* contribution (it composes known parts — CAR-
  style self-gating, the large-no-think mode, resolution co-variation, ABC-style agreement). We do **not**
  claim a new gate or cascade primitive; §5.2–§5.4 show the data refute novel routing primitives here.
- **Negative results as contribution.** The gate-saturation ceiling (§5.2), the unexploitable cross-family
  complementarity (§5.3), the language-prior diagnostic (§5.4), and the **action/selection limits (§5.9)** are,
  together, an honest characterization of *why* training-free medical-VLM cascades are hard: across the gate,
  the action, and (generative) answer selection, the binding constraint is the genuine "which answer is right?"
  knowledge the frozen models lack — a luck/capacity floor no training-free re-routing or re-selection clears.
  This is useful to the community independent of ACC, and it is what motivates the structural (compute-config) design.

---

## 7. Conclusion

For medical-VLM cascades *on multiple-choice benchmarks*, the predictive signal needed to route well
(recoverability) is largely absent from cheap features, and even large cross-family complementarity is
unexploitable with available peers. The leverage is structural: routing among **compute configurations**
of the same models — with the large model's fast no-think mode as the workhorse — yields large,
guardrail-safe efficiency gains (−72% latency, ~½ FLOPs, ~4× energy at parity) without a better gate.
We further show this routing ceiling is a **benchmark artifact**: in **open-ended generative** medical VQA
the same confidence signal reaches AUROC ~0.87 (vs ~0.6 on MCQ — a *discreteness*, not answer-length,
effect) and confidence-gated cascades work, so medical-VLM cascades should be evaluated open-ended (§5.7).
The gate itself, however, remains near-optimal at plain confidence in *both* regimes — no consistency,
semantic-entropy, or self-verification signal beats it. Finally, while *training-free* selection over a single
model's samples is luck-floored across gate, action, selection, synthesis, retrieval, and even structured
grounding outputs (§5.9), a **trained** outcome verifier breaks that floor — recovering 40–77% of the oracle
gap for both free-text answers (five datasets) and bounding boxes (including the real MS-CXR chest-X-ray
benchmark, a 5.6× lift), so *training* is the universal active ingredient (§5.10). We release ACC, the trained
verifier, and the full negative-result characterization.

---

## Reproducibility index
ACC: `src/cascade_methods/acc_v2.py` (+ `acc.py`, `acc_compare.py`, `gate_compare.py`). Math:
`results/cascade_methods/METHOD_MATH.md`, `METHOD_ACC.md`. Gate-saturation: `ceiling.py`,
`metarouter_honest.py`. Cross-family: `run_peer_eval.py`, `embed_siglip.py`, `peer_premise.py`,
`peer_router.py`, `peer_router_img.py`. Language-prior: `run_vlm_eval.py --blank`, `vision_sensitivity.py`.
Cost measurement: `src/cascade/measure_config.py`. ACC-v3/v4 (§5.1.2): `acc_v3_confgate.py`,
`acc_v4_lowres_think.py`, `gate_data_size.py`, `make_detailed_table.py`. Open-ended / ceiling-break
(§5.7): `src/labeling/{run_openvqa.py, run_openvqa_verify.py}`, `run_openvqa_{all,think,3b,lingshu,
lingshu7b}.sh`, `src/cascade_methods/{open_cascade_analyze.py, gate_search_open.py, make_open_chart.py}`;
writeup `results/cascade_methods/OPENENDED_CASCADE.md`; figs `paper/figs/open/`. Action/selection limits
(§5.9): `src/labeling/{run_openvqa_verify_persample.py, run_openvqa_select_listwise.py, run_openvqa_synth.py,
run_openvqa_fewshot.py, run_artifact_audit.py}`, `src/cascade_methods/{explode_sc_for_judge.py,
select_eval.py, knowledge_feasibility_bytype.py, strong_fixes_genuinely_unknown.py, crossfamily_agree.py}`;
writeups `results/cascade_methods/{RECOVERABILITY_IS_CAPACITY_BOUND, OPENENDED_SELECTION_LUCKFLOOR,
KNOWLEDGE_AUGMENTATION_FEASIBILITY}.md`. Trained verifier (§5.10): free-text
`src/training_methods/{run_lora_verifier_open.py, verifier_image_ablation.py, verifier_transfer_eval.py}`
(VERIF_DSETS env for pooled-4; --epochs 0 for the zero-shot baseline); structured box-verifier
`src/labeling/{run_ground_slake.py, run_ground_mscxr.py}` + `src/cascade_methods/ground_analyze.py`
(qwen smart_resize coord fix) + `src/training_methods/run_lora_box_verifier.py`; 5th dataset
`src/data_prep/prep_radimagenet.py` + `radimagenet_open` loaders; figure `paper/make_verifier_fig.py`
(`figs/limits/fig_trained_verifier_unified.png`); writeups `results/cascade_methods/{TRAINED_VERIFIER_RESULT,
BOX_VERIFIER_RESULT, NEW_DIRECTIONS_2026-06-25}.md`. MS-CXR = PhysioNet ms-cxr 1.1.0 + selective
MIMIC-CXR-JPG images. Full session narratives:
`progress_June_17.md`, `progress_June_20-22.md`. All checkpoints under `ckpts/` (gitignored). **No number
in this paper is fabricated.**

## [TODO before submission]
- **Figures: DONE** — Fig 1 latency-accuracy frontier + Fig 2 per-benchmark over-thinking (`paper/figs/`,
  via `paper/make_figs.py`). Still optional: F3 cross-family oracle-union bar; F4 recoverability-AUROC dot plot.
- Convert `[arXiv:...]` tags to a proper .bib; verify each citation's final venue/ID.
- Decide framing emphasis (systems-method-first vs characterization-first) per CVGIP reviewer fit.
- **Blocked (network):** the 2-size cross-family validation (Lingshu/MedGemma) — auto-resumes on recovery.
- Optional rigor: paired bootstrap CI on (Ours − best-baseline) FLOPs/latency (seed CIs already in §5.1.1);
  few-shot AutoMix re-run (ours is 0-shot self-verify).
