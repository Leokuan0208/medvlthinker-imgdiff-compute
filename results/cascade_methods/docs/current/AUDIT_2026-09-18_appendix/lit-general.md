# lit-general REPORT — beyond medical, beyond "yet another hidden-state verifier"
Agent `lit-general` · 2026-09-20 · APPEND-ONLY, written incrementally.

TOPIC: G1-G6 — science of correctness probes (transfer, label-efficiency, scaling, what they read),
general-domain VLM verifier setting, judge-bias inheritance, template/answer-prior leakage.

Status: IN PROGRESS

## 1. Verified paper cards

### G1 — cross-task/dataset generalization of correctness probes

**P1 · arXiv:2506.08572 · "The Geometries of Truth Are Orthogonal Across Tasks"**
Azizian, Kirchhof, Ndiaye, Bethune, Klein, Ablin, Cuturi (Apple) · v1 2025-06-10, v2 2025-07-04 ·
https://arxiv.org/abs/2506.08572
Trains linear truth/false probes per-task, shows probes across tasks "share little similarity", have
"almost disjoint supports" under sparsity, and even "mixtures of probes and tasks" (i.e. an explicit
multi-task / mixture-of-experts probe) FAIL to recover cross-task performance; activations cluster by
task, not by truth value, when viewed across tasks.
**Relation to F1/F2:** this is the closest published statement of our F1 (per-benchmark, LOBO fails) for
plain true/false statement probes (not generated-answer correctness, not VLM, not best-of-N). It also
already tried the exact fix an "adapt F1 into a method" plan would reach for first (probe mixtures) and
reports it doesn't work. **DONE** for "does task-specific truth geometry replicate" in text LLMs;
**OPEN** for whether it holds for (a) outcome/correctness-of-a-full-answer probes (not truth-of-a-bare-
statement), (b) VLMs, (c) best-of-N selection rather than classification accuracy as the metric.

**P2 · arXiv:2604.03754 · "Testing the Limits of Truth Directions in LLMs"**
Poulis, Crovella, Terzi · v1 2026-04-04, rev 2026-09-11 · https://arxiv.org/abs/2604.03754
Shows truth directions are layer-, task-type-, difficulty-, and PROMPT-template-dependent; "universality
claims... are more limited than previously known." No VLM, no best-of-N.
**Relation:** ADJACENT — another nail against a single universal direction; supports F2 (no property of
a benchmark predicts probe transfer) by showing the dependency is on many confounded axes at once, not
one interpretable one either.

**P3 · arXiv:2407.08582 · "On the Universal Truthfulness Hyperplane Inside LLMs"**
Liu, Chen, Cheng, He · EMNLP 2024 · https://arxiv.org/abs/2407.08582
The pro-transfer counter-case named in our own briefing: pooling >40 datasets to train ONE hyperplane
gets ~70% cross-task accuracy (per search-result synopsis, unverified against full text) vs single-
dataset probes, i.e. diversity of TRAINING data, not architecture, drives transfer — consonant with our
F4 (architecture doesn't matter, data does). Does not test best-of-N / BoN selection, is text-only.
**Relation:** ADJACENT — shows pooled-training helps some, but P1/P2 (2025-26, later) directly contest
the "universal" framing with harder tests; the field has NOT converged.

**P4 · arXiv:2505.16520 · "Are the Hidden States Hiding Something? Testing the Limits of Factuality-
Encoding Capabilities in LLMs"**
Servedio, De Bellis, Di Palma, Anelli, Di Noia · v1 2025-05-22, v3 2025-05-30 ·
https://arxiv.org/abs/2505.16520
Realistic (non-synthetic, LLM-generated) factuality datasets make probe performance drop sharply vs the
synthetic benchmarks most probing papers use.
**Relation:** ADJACENT — a distinct axis of the "what predicts probe success" question (G2 in F2's
language): dataset REALISM/provenance, not the six regime detectors we tried.

### G2 — cross-model / fine-tuning-lineage transfer of probes

**P5 · arXiv:2605.11448 · "Deep Minds and Shallow Probes"**
Lee, Kondor · 2026-05-12 · https://arxiv.org/abs/2605.11448
Theory paper: probes should be invariant to representation-basis symmetries; proposes a "shared
probe-visible quotient" through which probes transfer between models, giving "coverage-aware monitor
portability across model families." Text-only, no empirical VLM fine-tuning-lineage test in the parts
fetched.
**Relation:** ADJACENT — gives a theoretical vocabulary for WHY zero-shot cross-model transfer (our F3,
55% retained Lingshu→Qwen) works at all, but doesn't measure a fine-tuning lineage or a VLM.

**P6 · arXiv:2512.14880 · "Task Matrices: Linear Maps for Cross-Model Finetuning Transfer"**
O'Brien, Gajulapalli, Xia · 2025-12-16 · https://arxiv.org/abs/2512.14880
Learns a linear map between a base model's and its fine-tune's embeddings ("task matrix"); a base model
+ task matrix "surpasses linear probes, sometimes approaching finetuned levels." Tested across vision
and text models, 10 datasets.
**Relation:** ADJACENT and important — this is the natural NEXT step on our F3 zero-shot-transfer result:
a learned linear map (not raw zero-shot copy) between Lingshu-7B and Qwen2.5-VL-7B hidden states might
recover more than 55% of the gain. Not tested for verifier/BoN transfer or same our exact model pair.

**P7 · arXiv:2603.24787 · "ReLope: KL-Regularized LoRA Probes for Multimodal LLM Routing"**
Zeng, Wang, Chen, Lin · v1 2026-03-25, v2 2026-07-14 · https://arxiv.org/abs/2603.24787
States plainly: "probe routing... provides an effective solution in text-only LLMs. However, we observe
that these probes degrade substantially when applied to multimodal LLMs (MLLMs)... the presence of
visual inputs weakens the separability of correctness signals in hidden states." Fixes: an Attention
Probe (aggregate prior-layer states by attention) and a KL-regularized LoRA probe.
**Relation:** DONE for "do vanilla correctness probes get worse on VLMs than on text LLMs, because of
the image tokens" — this is independent, converging support for our F5 (image-token features add
nothing) from a different angle (routing/gating rather than BoN verification), and a ready-made
stronger-probe recipe to try against our F4 null (architecture doesn't matter) if it also fails on our
setup that would strengthen F4 considerably.

### G3 — probe/verifiability skill vs. scale and post-training

**P8 · arXiv:2504.05419 · "Reasoning Models Know When They're Right: Probing Hidden States for
Self-Verification"**
Zhang, Chen, Pan, Zhao, Panda, Li, He · 2025-04-07 · https://arxiv.org/abs/2504.05419
MLP probes on reasoning-model hidden states verify intermediate answers with high accuracy and
calibrated scores; used to early-exit reasoning, cutting inference tokens 24% at no accuracy cost. No
scale sweep or GRPO-before/after comparison confirmed in the fetched abstract.
**Relation:** DONE for "hidden states encode correctness that the model itself doesn't exploit" in text
reasoning models (matches our framing that the probe recovers signal the greedy decode doesn't use);
OPEN for VLMs, open-ended VQA, and explicit scale/post-training dependence.

**P9 · arXiv:2604.13386 · "Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer
Ensembling"**
Nordby, Pais, Parrack · 2026-04-15 · https://arxiv.org/abs/2604.13386
0.5B-176B, 12 models, deception-detection (Insider Trading / Harm-Pressure Knowledge) tasks: "probe
accuracy improves with scale: ~5% AUROC per 10x parameters (R=0.81)"; multi-layer ensembling recovers
single-layer failures (+29pp / +78pp AUROC) because "deception directions rotate gradually across
layers."
**Relation:** DONE, and the single closest quantitative law in the field for "does probe skill scale
with model size" — but for a DIFFERENT probing target (deception, not answer correctness), text-only,
not BoN/verifier framing. Gives a concrete external number (~5%/10x, R=0.81) our own Qwen/Lingshu
7B→32B(→72B-AWQ) sweep could be checked against.

**P10 · arXiv:2605.09502 · "Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is
Diagnostic, Not Causal"**
(authors not re-confirmed beyond search synopsis — VERIFY before citing) · 2026-05 ·
https://arxiv.org/abs/2605.09502
Reasoning-error probe AUROC non-monotonic across scale: 7B 0.669 → 14B 0.762 → 32B/72B recovering to
~0.9-0.977; causal patching of hidden states does NOT fix reasoning (diagnostic, not causal).
**Relation:** ADJACENT — a second, non-monotonic scaling curve (contradicts P9's clean log-linear fit),
math/CoT domain. Together P9+P10 show scale-vs-probe-skill is NOT a settled law even in text — a
VLM/BoN-verifier version is genuinely open and directly buildable from the project's cached Qwen sizes.

**P11 · arXiv:2604.23318 · "Hidden States Know Where Reasoning Diverges: Credit Assignment via
Span-Level Wasserstein Distance"**
(per search synopsis) · 2026-04-25 · https://arxiv.org/abs/2604.23318
Within-GRPO-group Wasserstein distance between correct/incorrect rollout hidden-state distributions is
usable as a dense credit-assignment signal, tested with GRPO training itself (Qwen2.5-Math-7B,
Llama3.1-8B, Qwen2.5-14B).
**Relation:** ADJACENT — probes a signal DURING RL, not a before/after comparison of correctness-probe
skill across training stage. Does not answer "does GRPO sharpen or erase the frozen correctness
signal" (our G3 question) directly.

### G4 — general-domain VLM BoN / multimodal reward models vs. a tiny probe

**P12 · arXiv:2503.10291 · "VisualPRM: An Effective Process Reward Model for Multimodal Reasoning"**
(InternVL team) · 2025-03 · https://arxiv.org/abs/2503.10291
8B-parameter multimodal PRM (VisualPRM400K training set), improves MiniCPM-V2.6/QwenVL2.5-7B/
InternVL2.5-8B/-78B by +8.0/+3.7/+8.4/+5.9 points over 7 REASONING benchmarks in BoN; beats ORM and
self-consistency. Evaluated on multimodal *reasoning* (MathVista-style), not everyday open-ended VQA.
**Relation:** ADJACENT — an 8B RM, ~8,700x our probe's ~918k params, never tested on VQAv2/GQA/
TextVQA/DocVQA/MM-Vet-style open-ended VQA, only reasoning benchmarks.

**P13 · arXiv:2505.07263 · "Skywork-VL Reward: An Effective Reward Model for Multimodal Understanding
and Reasoning"**
Wang, Wang, Pei, Shen, Peng, Hao, Qiu, Jian, Xie, Song, Liu, Zhou · v1 2025-05-12, v2 2025-06-09 ·
https://arxiv.org/abs/2505.07263
Reward model built by adding a reward head to **Qwen2.5-VL-7B-Instruct** (multi-stage fine-tuning,
pairwise ranking loss on a large preference dataset covering both standard VLM and VLM-reasoner
responses).
**Relation:** ADJACENT, but load-bearing for a direction: this RM shares its EXACT backbone with our
project's own Qwen2.5-VL-7B-Instruct generator/probe host. No open-ended-VQA-BoN comparison against a
lightweight hidden-state probe on the same backbone exists in what we could verify.

**P14 · arXiv:2502.14191 · "Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision
Language Models"**
Yasunaga, Zettlemoyer, Ghazvininejad · 2025-02-20 · https://arxiv.org/abs/2502.14191
5,211 expert-annotated triplets across 6 domains incl. a VQA slice; best judges (Gemini 1.5 Pro, Claude
3.5 Sonnet) only 72% accuracy. Benchmarks RMs generically, not open-ended-VQA-BoN gain specifically, and
does not compare tiny probes to billion-param RMs.
**Relation:** ADJACENT — the right BENCHMARK-of-benchmarks to cite for "what a multimodal RM is
supposed to be good at," but not the comparison we need.
**G4 verdict: OPEN.** No verified paper compares a <1M-param hidden-state probe against a multi-billion-
parameter multimodal RM for BoN accuracy on standard open-ended general-domain VQA. Skywork-VL-Reward's
shared Qwen2.5-VL-7B-Instruct backbone makes this the cheapest fair fight to run.

### G5 — predictive law for BoN gain from verifier quality, coverage, N

**P15 · arXiv:2507.12399 · "ROC-n-reroll: How verifier imperfection affects test-time scaling"**
Dorner, Chen, Cruz, Yang · 2025-07-16 · https://arxiv.org/abs/2507.12399
Proves instance-level BoN/rejection-sampling accuracy is "precisely characterized by the geometry of the
verifier's ROC curve"; rejection sampling beats BoN at fixed compute for concave ROC, both converge at
infinite compute; high-compute performance is NOT predictable from low-compute observations in general.
Text-domain theory.
**Relation:** DONE for the theoretical machinery (ROC-curve-drives-BoN-accuracy); NOT empirically fit
against a grid like the project's (2 generators × 8 benchmarks × 4 temperatures × N∈{8,16,32}), and not
applied to coverage as a separate axis. The clearest theory paper to test our grid against.

**P16 · arXiv:2606.02981 · "Predicting Inference-Time Scaling Gains from Labeled Validation-Set Output
Statistics"**
Zhang, Li · 2026-06-02 · https://arxiv.org/abs/2606.02981
Ridge regression on 3 cheap validation statistics (prompt-level agreement spread, label-assisted
first-correct-sample position, completion-length variance) + entropy predicts realized BoN gain,
ρ=0.90 with actual reward-model-verified gains, from a SINGLE labeled validation pass. Text-domain.
**Relation:** ADJACENT — a genuinely different, non-AUROC-based feature set that already gets ρ=0.90;
worth testing directly against our data as a strong external baseline before claiming our own law.

**P17 · arXiv:2607.17531 · "Oracle Gap and Signal Fidelity: A Fixed-Pool Diagnostic for Test-Time
Collaboration"**
Hu · 2026-07-20 · https://arxiv.org/abs/2607.17531
Decomposes net BoN/collaboration gain into recoverable mass (oracle gap) × verification-signal coverage
× selection quality × harm; code/math domains (LiveCodeBench, MATH-L5, GPQA-Diamond); e.g. a public-test
verifier (MCC 0.825) gains +8.14pp, a generated-test verifier (MCC 0.248) gains +2.70pp with near-zero
harm. No closed-form AUROC/coverage/N formula — a measured decomposition, not a predictive law.
**Relation:** ADJACENT — closest in SPIRIT to our F6 (selection efficiency ~0.78-0.81 regardless of
method): "oracle gap is a joint property of task, model, and sampling config," matching our own framing.
Not VLM, not open-ended-text, no closed-form law.
**G5 verdict: OPEN for VLMs / open-ended VQA and for a validated law combining AUROC+coverage+N**; three
different, competing partial theories exist in text/code (ROC-geometry, cheap-statistics-ridge, oracle-
gap-decomposition) and none has been cross-tested against the others or against a multimodal grid.

### G6 — evaluation-currency sensitivity (F8) and template/answer-prior leakage (F9)

**P18 · arXiv:2603.12520 · "When LLM Judge Scores Look Good but Best-of-N Decisions Fail"**
Landesberg · 2026-03-12 · https://arxiv.org/abs/2603.12520
5,000-prompt best-of-4 Chatbot Arena study: a judge with moderate GLOBAL correlation (r=0.47) captures
only 21.0% of the improvement perfect selection would give over random choice, because within-prompt
correlation is only r_within=0.27 and 67% of pairwise comparisons tie; pairwise judging recovers 21.1%→
61.2%. Prescribes reporting within-prompt signal/tie-rate/recovery instead of global agreement.
**Relation:** DONE for "a judge's global correlation overstates its real best-of-N selection value" —
but this is about the JUDGE acting as the selector itself, not about a SEPARATELY TRAINED verifier whose
measured GAIN differs >10x depending on which metric scores its picks (our F8). Landesberg's
within-prompt/tie-rate toolkit is directly reusable as an instrument for our probe, but the specific
question — "does a probe trained on judge labels partially learn the judge's accepted PHRASINGS, and by
how much does that inflate its judge-currency gain vs. its EM-currency gain" — is not asked here.
**OPEN** in this precise form.

**P19 · arXiv:2603.08091 · "Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing
Optimization"**
Zhou, Huang, Zhang, Chen, Xu, Zhu, Zhao, Yang · 2026-03-09 · https://arxiv.org/abs/2603.08091
JudgeBiasBench: 12 bias types across 4 dimensions, bias-injection pipeline, bias-aware RL/contrastive
debiasing for judges. Fetched abstract does not state whether downstream models TRAINED on a biased
judge's labels inherit the bias (the paper's focus is the judge itself, not a distilled verifier).
**Relation:** ADJACENT — gives a taxonomy/benchmark for judge bias but not the "downstream verifier
inherits it" claim; would need the full text to confirm either way.

**P20 · arXiv:2008.02637 · "Question and Answer Test-Train Overlap in Open-Domain Question Answering
Datasets"**
Lewis, Stenetorp, Riedel · 2020-08-06 (EACL 2021) · https://arxiv.org/abs/2008.02637
Foundational: 60-70% of test-time answers also appear in train sets, 30% of test questions have a
near-duplicate paraphrase in train; models perform 63pp worse (absolute) on non-memorizable
(non-repeated) questions than on repeated ones, across open-domain QA.
**Relation:** DONE for the GENERAL phenomenon (train/test template-and-answer overlap inflates apparent
model skill) in text QA — never applied to a BEST-OF-N VERIFIER's measured gain (our F9's specific
target), and predates VQA/VLM entirely. The direct methodological ancestor of F9.

**P21 · arXiv:2210.04692 · "Language Prior Is Not the Only Shortcut: A Benchmark for Shortcut Learning
in VQA" (VQA-VS)**
(EMNLP Findings 2022, per search synopsis — not independently re-fetched, cite cautiously) ·
https://arxiv.org/pdf/2210.04692
Extends VQA-CP: builds multiple distribution-shift OOD test sets (not just language-prior) to stress
different shortcut types in VQA; question-only "blind" baselines score 15.95-58.61% depending on split.
**Relation:** ADJACENT — establishes that VQA benchmarks carry MULTIPLE shortcut types (not just
language priors), a good citation for why a bag-of-answer-strings prior (F9) recovering 25-30% of gain
is unsurprising in this literature's terms, but again never connected to a trained VERIFIER's gain.
**G6 verdict: OPEN for both F8 and F9 in the precise "trained probe as intermediary" form.** The nearest
prior art (P18 for F8, P20/P21 for F9) supplies exactly the right MEASUREMENT TOOLS (within-prompt
correlation/tie-rate; train-test overlap rate; blind/prior baselines) but nobody has pointed them at a
small trained hidden-state verifier's best-of-N gain specifically.

Also flagged (already covered/verified by sibling agents, not re-verified here, cited for completeness):
ELHSR/SWIFT arXiv:2505.12225, HSRM arXiv:2608.30841, CASE arXiv:2608.17124, LiLaVe arXiv:2504.16760,
MedProb arXiv:2609.04336 (`lit-vlm-med/REPORT.md` — confirms MedProb Appendix H already does probe-as-
BoN-selector in a VLM, killing the "first VLM application" framing but NOT the general-domain,
transfer/label-efficiency/scaling/judge-bias questions this report covers), DualRead arXiv:2609.06419,
PAIR arXiv:2605.17877, SR-GRPO arXiv:2512.02807 (`lit-finetune/REPORT.md`).

## 2. Sub-question verdicts (summary)

| Q | Verdict | Closest papers | What's still new |
|---|---|---|---|
| G1 | ADJACENT (text-only, bare-statement truth, not BoN) | P1, P2, P3 | Same test on outcome/BoN-correctness probes, in a VLM; "probe mixtures fail" (P1) already rules out the obvious method fix |
| G2 | OPEN for VLM fine-tuning lineages, zero-shot | P6, P5, P7 | Zero-shot % on a real base→fine-tune lineage (Qwen2.5-VL-7B→Lingshu-7B) + whether a learned linear map (P6's mechanism) closes the remaining 45% |
| G3 | OPEN for VLMs / BoN; text-only laws disagree (P9 vs P10) | P9, P10, P8 | Scale sweep (3B/7B/32B/72B-AWQ) of BoN-verifier AUROC on a fixed open-ended-VQA task, plus explicit pre/post-RLVR comparison |
| G4 | OPEN | P12, P13, P14 | Tiny probe vs multi-billion-param RM (esp. Skywork-VL-Reward, same Qwen2.5-VL-7B-Instruct backbone) for BoN accuracy AND $/point on standard open-ended general VQA |
| G5 | OPEN (3 competing partial theories, none cross-tested, none multimodal) | P15, P16, P17 | Fit/refute ROC-geometry (P15) and oracle-gap (P17) against the project's own 2-generator×8-benchmark×4-temp×N grid |
| G6 (F8) | OPEN in the "trained verifier inherits judge phrasing" form | P18, P19 | Apply P18's within-prompt/tie-rate instrument to the PROJECT'S probe (not the judge) across currencies; quantify phrasing-inheritance directly |
| G6 (F9) | OPEN in the "verifier gain vs. template leakage" form | P20, P21 | Apply P20's train/test-overlap methodology to a trained BoN verifier's realized gain, stratified by template-novel vs. template-repeat |

## 3. Proposed directions (survive the search)

**D1 — A predictive law for BoN gain, tested against three competing text/code-domain theories (G5).**
Pitch: fit realized macro BoN gain (already measured, per benchmark×model×N×T cell) against
theory-motivated features — within-question AUROC + coverage (oracle−greedy) from ROC-n-reroll (P15)
and the oracle-gap/signal-fidelity decomposition (P17) — and separately against P16's cheap
validation-statistics feature set (ρ=0.90 in text); report which (if any) generalizes to a multimodal,
open-ended-VQA grid, where our own F2 already found nothing among six ad hoc regime detectors.
Nearest prior art / delta: P15/P16/P17 (text/code only, never cross-tested against each other or against
VQA). First experiment: CPU-only. Recompute within-question AUROC + coverage per cell from
`feats_hidden/` and `ckpts/openvqa/` judge labels already on disk (Lingshu-7B + Qwen2.5-VL-7B, 8
benchmarks, existing N/T sweep incl. `coverage_sc16_ci_ALL_2026-09-16.json`); fit ridge/lasso; a few
hours, no GPU. Kill result: R²/ρ indistinguishable from a mean-only baseline across ALL three feature
sets — upgrades F2 from "we didn't find a predictor" to "provably unpredictable even under the field's
best current theories," itself a strong, honest negative-results contribution. Main risk: likely
REPLICATES F2's null rather than breaking it; frame the paper as "we exhaustively tested the field's
three existing predictive theories against a multimodal, non-text BoN grid, and all three fail," not as
"we found the law."

**D2 — Fine-tuning-lineage probe transfer, zero-shot vs. a learned linear map (G2, extends F3).**
Pitch: our F3 shows a Lingshu-7B probe applied zero-shot to Qwen2.5-VL-7B hidden states keeps ~55% of
the native-probe gain (Lingshu is a Qwen2.5-VL-7B fine-tune). Test whether a cheap closed-form linear
map (à la Task Matrices, P6) between the two models' hidden-state spaces, fit on the ALREADY-HELD-OUT
labelled half (no new generation), recovers more of the missing 45%.
Nearest prior art / delta: P6 (base/fine-tune linear maps, vision+text, 10 datasets, never for a
same-family VLM verifier or BoN task); P5 (theory for why raw zero-shot works at all).
First experiment: CPU-only if paired hidden states for the same questions/images already exist for both
generators (verify first — `code-audit/out/s02.txt` confirms row-count alignment across generators on
PathVQA/SLAKE/VQA-RAD/8-bench, so this is plausible); fit ridge-regression linear map Qwen-space→
Lingshu-space on train half, re-score held-out half with the frozen Lingshu probe through the map; a few
hours. Kill result: linear map recovers <10% of the remaining 45% gap — shows the fine-tuning-induced
representation shift is NOT linearly recoverable, a clean negative result either way. Main risk: if
hidden states for the two generators were never extracted on IDENTICAL held-out items, this needs new
(cheap, CPU-feasible if only a forward pass on cached generations, else small-GPU) feature extraction —
check before promising zero-GPU-cost.

**D3 — A tiny probe vs. a multi-billion-parameter multimodal reward model, on general open-ended VQA
(G4).** Pitch: no verified paper compares a <1M-parameter hidden-state probe to a multi-billion-parameter
multimodal RM for BoN accuracy on standard open-ended VQA (VQAv2/GQA/OK-VQA/TextVQA/DocVQA/ChartQA/
MM-Vet free-form). Skywork-VL-Reward (P13, 2505.07263) is built directly on Qwen2.5-VL-7B-Instruct — the
SAME backbone the project already has a native probe for — making this the cheapest fair fight available.
Report accuracy AND $/point (RM forward passes for N candidates vs. probe's ~918k-param head on hidden
states already produced by generation).
Nearest prior art / delta: P12 (VisualPRM, 8B, reasoning benchmarks only), P13 (Skywork-VL-Reward, same
backbone, preference data not open VQA), P14 (Multimodal RewardBench, benchmarks RMs generically, not
this comparison). Delta: nobody has run this specific probe-vs-RM, same-backbone, open-VQA-BoN
comparison. First experiment: needs (a) Skywork-VL-Reward weights (~7-8B, check license/availability —
HF `Skywork/Skywork-VL-Reward` if it exists) loadable on one A100-80GB, and (b) a small existing or
cheaply-generated open-ended-VQA candidate pool (project has none non-medical on disk per the briefing —
this is the one direction that plausibly needs a modest GPU generation run, e.g. 1-2k questions from a
public open-ended VQA split × N=8, on Qwen2.5-VL-7B, which the project already runs routinely). Kill
result: Skywork-VL-Reward beats the probe by a wide, significant margin on accuracy per point AND per
dollar — the "tiny probe is competitive with billion-param RMs" claim dies for the general domain even
if it holds in medical VQA. Main risk: the only direction here requiring new GPU generation + a new
downloaded RM checkpoint, i.e. the highest-cost of the four; do D1/D2 first as CPU-only proof the probe
generalizes at all before spending GPU time here.

**D4 — Does the verifier's headline gain survive on genuinely novel (non-template-repeat) questions?
(G6, F9, the cheapest and most original.)** Pitch: F9 already shows 55-100% of held-out questions on 5/8
benchmarks reuse a training-seen question template, and a bag-of-answer-strings prior alone recovers
25-30% of the judge-currency gain. Apply Lewis et al.'s (P20) train/test-overlap methodology, but to the
VERIFIER's realized macro BoN gain rather than to raw QA accuracy: stratify held-out questions into
template-novel vs. template-repeat, report the probe's gain on each stratum separately, in BOTH judge and
EM currency (closing the loop with F8). Cite VQA-CP/VQA-VS (P21) as the general-VQA precedent for why
this kind of shortcut is expected, to frame the medical result as an instance of a known general-domain
phenomenon rather than a medical peculiarity — the cleanest "beyond medical" framing of the four without
needing new non-medical data for its FIRST experiment.
Nearest prior art / delta: P20 (open-domain QA, generator accuracy, not applied to a verifier); P21
(VQA shortcut benchmark, generator accuracy, not applied to a verifier). Delta: nobody has stratified a
trained BoN verifier's gain by train/held-out template overlap.
First experiment: pure CPU, reuses artifacts already computed for F9 (need to locate/confirm the exact
source file — the template-clustering and bag-of-answer-strings numbers are already stated as measured
in the LIT_BRIEFING project facts) plus the em-rescore outputs (`em-rescore/em_rescore_pooled_probe_
2026-09-18.json`); a few hours. Kill result: template-novel-only subset shows the SAME gain as the full
set within CI — template leakage is NOT the explanation, F9's correlation was coincidental, a clean fast
falsification. Main risk: lowest cost/risk of the four but weakest standalone "beyond medical" claim; its
generalization argument rests on citing VQA-CP/VQA-VS rather than a fresh general-domain replication — a
natural (optional, GPU-light) follow-up would rerun the same stratification on a small public open-ended
VQA pool with Qwen2.5-VL-7B, reusing whatever generation the project stands up for D3.

**Recommended order:** D1 and D4 first (CPU-only, reuse artifacts already on disk, days not weeks;
either could be written up as a negative-results contribution even on failure); D2 next (CPU-only if
hidden-state alignment checks out, otherwise flag before promising); D3 last and only if D1-D2 show the
probe's signal is genuinely robust, since it is the only one needing new GPU generation and a new
downloaded checkpoint.

## 4. Query log

WebSearch: "LLM hidden state truthfulness probe generalization across datasets tasks 2025 2026" ·
"\"universal truthfulness\" hyperplane direction LLM hidden states arxiv" · "probe transfer fine-tuned
model hidden states base model \"linear probe\" transfer arxiv 2026" · "reward model best-of-N verifier
LLM judge bias inherited trained arxiv 2026" · "multimodal reward model best-of-N VQAv2 TextVQA DocVQA
open-ended visual question answering 2026 VisualPRM LLaVA-Critic" · "hidden state probe vs reward model
comparison best-of-N cost efficiency small probe large reward model arxiv 2026" · "reasoning models know
when they're right calibration hidden states 2025 arxiv Zhang" · "RLVR GRPO effect on hidden state
correctness signal probe accuracy before after RL arxiv 2026" · "VQA-CP language prior shortcut
benchmark blind baseline question-only VQA answer template 2026" · "generator verifier gap best-of-N
scaling law AUROC predicts accuracy gain arxiv 2026" · "reward model overfits LLM judge artifacts RLAIF
bias amplification distillation arxiv 2026" · "benchmark question template overlap train test
contamination VQA shortcut answer memorization 2026 arxiv" · "few-shot probe adaptation meta-learning
correctness probe new task small labeled set arxiv 2026" (no specific hit) · "probe verifier accuracy
scales with model size 3B 7B 32B 72B hidden state correctness arxiv 2026" · "VisualPRM arxiv InternVL
process reward model best-of-N multimodal" · "Skywork-VL-Reward arxiv parameters benchmark best-of-N VQA
multimodal reward model" (budget exhausted mid-query, resolved via known URL instead).
WebFetch (arxiv abs pages, all verified as shown above): 2506.08572, 2604.03754, 2407.08582, 2505.16520,
2605.11448, 2512.14880, 2603.12520, 2504.05419, 2603.24787, 2607.17531, 2603.08091, 2502.14191,
2507.12399, 2606.02981, 2008.02637, 2604.13386, 2505.07263. WebSearch session budget (200 calls, shared
across the whole job's agents) was exhausted before two planned queries ("fine-tuning barely moves the
truth direction" full search, additional Skywork-VL-Reward search) — resolved the second from an already-
returned URL; the first (a phrase-level check for one more G2 paper) was left undone, see "Could not
verify" below.

## 5. Could not verify / left undone
- The exact phrase-search for "fine-tuning barely moves the truth direction" (named in our own prompt) —
  WebSearch budget ran out before this query. P5/P6 partially cover the same ground but are not a
  confirmed match for that specific claim/paper.
- P10 (2605.09502) and P11 (2604.23318) author lists were taken from WebSearch synopses, not independently
  re-confirmed via WebFetch of the abs page — treat author attribution as provisional, the arXiv ID/title/
  numbers as search-engine-reported (not full-text verified).
- P21 (VQA-VS, 2210.04692) was not independently re-fetched via WebFetch; venue/numbers are from the
  WebSearch synopsis only.
- P3's "~70% cross-task accuracy, +14pp over single-dataset probes" number came from a WebFetch synopsis
  of the abs page, not a table read from the full PDF — treat as approximately right, not exact.
- Did not verify JudgeBiasBench (P19)'s downstream-inheritance claim (full text needed, not fetched).
- Did not deep-read beyond abstract for any paper (time budget) — all cards are abstract-level. Any
  direction that proceeds should read the specific 2-3 closest papers (D1: P15+P17; D2: P6; D4: P20) in
  full before writing.

Status: COMPLETE (abstract-level verification; deep reads deferred per note above)
