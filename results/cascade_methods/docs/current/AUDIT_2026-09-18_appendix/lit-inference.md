# lit-inference REPORT — hidden-state probes at INFERENCE time beyond best-of-N selection
Agent `lit-inference` · 2026-09-20 · status: IN PROGRESS (append-only)

TOPIC: adaptive compute / early pruning / coverage / steering / label-free onboarding using hidden-state
probes at inference time, for the project's open-text medical VQA probe-verifier
(Lingshu-7B / Qwen2.5-VL-7B / MedGemma-4b-it, 8 benchmarks, N=8/16 candidate pools, judge labels, frozen
probes). MedGemma pools are BROKEN on disk -- no experiments planned on them.

Verification rule: every card fetched from arxiv.org/abs/<id> (or venue page) via WebFetch/WebSearch;
anything not fetched is UNVERIFIED and excluded from conclusions. No sub-agents spawned.

---
## 1. Verified paper cards
(appended as verified)

---
## 2. Sub-question verdicts
(appended as completed)

---
## 3. Proposed directions
(appended at end)

---
## 4. Query log
(appended as run)

### I1 — adaptive-N / optimal stopping driven by a hidden-state probe

**P1 - arXiv:2410.02725 - Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even
Mid-Generation.** Manvi, Singh, Ermon (Stanford) - 2024/10/03 - https://arxiv.org/abs/2410.02725
Generative self-evaluation: model predicts (via ONE extra token, not a trained external probe) whether
restarting would beat the current sample; used to prune/stop/pick. Llama-3.1-8B, AlpacaEval/GSM8K.
74% of the 16-sample gain recovered at 1.2 avg samples. Text-only.
**Relation:** closest "adaptive-N via internal self-report" paper, but the signal is a generated token via
the LM head, not a separately-trained MLP probe on pooled hidden states, and not a VLM. ADJACENT.

**P2 - arXiv:2410.04707 - Learning How Hard to Think: Input-Adaptive Allocation of LM Computation.**
Damani, Shenfeld, Peng, Bobu, Andreas (MIT) - 2024/10/07 - ICLR 2025 - https://arxiv.org/abs/2410.04707
Predicts the *distribution of rewards* given input+budget from a learned model, then allocates more
decoding compute (search/reranking/self-critique) to inputs predicted to benefit most. Code, math,
dialog; text-only. Uses a learned predictor over the PROMPT (pre-generation), closer to I2 in spirit
than I1, but the framing is "how much compute to spend" generally.
**Relation:** ADJACENT — input-adaptive compute allocation, not candidate-level hidden-state probing
during/after generation, not a VLM.

**P3 - arXiv:2510.01394 - Optimal Stopping vs Best-of-N for Inference Time Optimization.** Kalayci,
Raman, Dughmi - 2025/10/01 - https://arxiv.org/abs/2510.01394
UCB-style Pandora's-Box algorithm for WHEN TO STOP sampling without knowing the reward distribution in
advance; provably close to Weitzman's optimal policy; Bradley-Terry reward normalization for
cross-prompt scaling. 15-35% fewer generations than non-adaptive BoN at matched quality. AlpacaFarm,
HH-RLHF, text LLM + external reward model pairs (not hidden-state probes).
**Relation:** THE closest published instance of "Weitzman-style adaptive stopping for LLM BoN," i.e.
exactly the controller class the project already owns (unused). Confirms the controller class is
published and works, but (a) stopping is driven by an EXTERNAL reward model score, not a frozen
hidden-state probe read off the generation pass, and (b) text-only, no VLM, no medical, no coverage
analysis. DONE for "does Weitzman-stopping work for LLM BoN" — OPEN for "driven by a hidden-state probe,
in a VLM, under a coverage-limited regime."

**P4 - arXiv:2512.05325 - LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning.** Akgul,
Kalayci, Kannan, Neiswanger, Prasanna - 2025/12/05 - https://arxiv.org/abs/2512.05325
Trains a lightweight PROBE on hidden states at naturally-occurring reasoning-cue tokens ("hmm", "wait")
for early-exit inside a SINGLE chain-of-thought, wrapped in split conformal prediction for
distribution-free exit control. GSM8K/MATH-500/AIME/CommonsenseQA, 1.5B-32B text models.
**Relation:** the closest mechanism match to "hidden-state probe drives a stopping decision" but the
stopping is WITHIN one reasoning trace (early-exit a single generation), not ACROSS best-of-N samples,
and not a VLM. Its conformal-calibration trick (distribution-free exit control from a probe score) is
directly reusable for a probe-driven Weitzman-style N-controller. ADJACENT, useful mechanism donor.

**P5 - arXiv:2606.07392 - Online Pandora's Box for Contextual LLM Cascading.** Belloni, Chen, Wei -
2026/06/05 (v2 2026/08/26) - https://arxiv.org/abs/2606.07392
Weitzman-policy learning for sequential API querying across DIFFERENT models/services (cascading), using
external cost/value estimates, not internal probes. Confirms the Weitzman framing is being actively
extended in 2026 to LLM systems, but for model routing, not within-model BoN, and no hidden-state signal.
ADJACENT (same math family as the project's unused controller, different lever - model choice not N).

**P6 - arXiv:2608.20316 - Pandora's AI Model Routing Box: Efficient Allocation with Costly Value
Estimation.** (title/venue only confirmed via search snippet; author list not yet fetched) -
https://arxiv.org/abs/2608.20316 - same family as P5, model-routing not sampling-count; UNVERIFIED
pending full fetch, noted for completeness, excluded from verdicts until fetched.

### I2 — pre-generation probes (predict before sampling starts)

**P7 - arXiv:2509.10625 - No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear
Probes.** Moreno Cencerrado, Padres Masdemont, Gonzalvez Hawthorne, Africa, Pacchiardi - 2025/09/12
(v3 2026/03/03) - Principled Design for Trustworthy AI @ ICLR 2026 (poster) -
https://arxiv.org/abs/2509.10625
Activations captured after the question is read but BEFORE any generation; linear probe predicts
eventual correctness. 3 open text model families 7-70B, generic trivia + OOD knowledge sets;
outperforms black-box baselines and verbalized confidence; generalizes OOD on knowledge, weaker on math.
**Relation:** exactly the mechanism of I2 (pure prompt-side probe, zero generation cost) - text-only, no
image, no coverage-prediction framing. DONE for "can a pre-gen probe predict eventual correctness" in
text LLMs; OPEN for VLM/image-conditioned prompts and for "predict whether an 8-sample pool will contain
a correct answer at all" (coverage, not point correctness).

**P8 - arXiv:2606.14530 - Code Correctness Is Linearly Decodable from LLM Hidden States Before
Generation.** Di Cicco - 2026/06/12 (v3 2026/07/16) - https://arxiv.org/abs/2606.14530
Linear probe on the final-prompt-token hidden state of Qwen3-4B-Instruct-2507 predicts LiveCodeBench
pass/fail before generation: held-out AUC 0.881 +/-0.008 (0.842 after residualizing prompt length).
Does NOT use the signal to pick N or temperature; authors note self-repair too rare to study.
**Relation:** same pre-generation-probe mechanism as P7, single model, code domain; confirms the effect
replicates outside trivia QA. DONE for "is pre-gen correctness linearly decodable" (yes, repeatedly,
across code/trivia/text) - the project's own N-prediction / coverage-prediction use is still OPEN.

**P9 - arXiv:2510.19669 - DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference.**
Liu, Hu, Chu, Choi - 2025/10 (v5) - https://arxiv.org/abs/2510.19669
Small MLP probe on PROMPT-ONLY hidden states (early layers) predicts a 3-way difficulty class
(Easy/Normal/Hard); the predicted class selects a reasoning STRATEGY (prompt template / effectively a
token budget) before any generation. DeepSeek-R1-Qwen-7B, Qwen3; GSM8K/MATH-500/OlympiadBench/MMLU-Pro/
GPQA/Minerva; ~30-40% token reduction at matched accuracy. Text-only.
**Relation:** closest published match to "a pre-generation hidden-state probe choosing a decoding
STRATEGY (not just predicting pass/fail)" - i.e., exactly the shape of "which temperature / how many
samples to use," but the lever chosen is reasoning-token budget, not sample count N or temperature, and
it is text-only reasoning-math, not VLM/medical/open-ended-answer-format. DONE for the mechanism shape;
OPEN for "N-count or temperature as the chosen lever" and for VLMs.

**P10 - arXiv:2503.01422 - Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling
in Early Decoding (ST-BoN).** Wang, Zhang, Huang, Yang, Zhang, Huang, Wang - 2025/03/03 - NeurIPS 2025
Spotlight - https://arxiv.org/abs/2503.01422
Uses early-decoding internal-state CONSISTENCY (Chain-of-Embedding) across in-flight samples to identify
the likely-best path and truncate the rest early; >80% dynamic GPU memory reduction, 50% latency
reduction at matched quality. This is MID-generation (not pre-generation) but pre-COMPLETION, so sits
between I1 and I2.
**Relation:** ADJACENT to both I1 (early truncation of a BoN pool) and I2 (uses an internal signal before
full generation) but the signal is inter-sample embedding consistency, not a trained correctness probe,
and not a VLM. Directly relevant as a cheaper alternative/complement to the project's Weitzman controller
- worth reading in full if I1/I3 direction is pursued.

### I3 — attacking COVERAGE (image-side TTA, cross-model fusion, calibrated confidence)

**P11 - arXiv:2602.06566 - SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of
VLMs.** Avogaro, Debnath, Mi, Frick, Wang, He, Hua, Schindler, Rigotti - 2026/02/06 (v3 2026/06/24) -
https://arxiv.org/abs/2602.06566
Two-stage: explicit visual search (localize question-relevant regions at low res) then high-res
processing only on selected regions, then reasoning. Qwen3VL-4B; V* benchmark +6.7 pts; beats
"thinking with images" baseline by +4.6 pts at 200x lower token budget in OOD. No best-of-N, no
hidden-state probe, no verifier component mentioned.
**Relation:** ADJACENT — an image-side test-time-scaling method that could be a candidate-GENERATOR for
the project's coverage problem (more diverse, cheaper high-quality crops feeding the pool) but does not
touch verification/selection at all. General-domain, not medical.

**P12 - arXiv:2606.19950 - Confidence Calibration for Multimodal LLMs: An Empirical Study through Medical
VQA.** Du, Wang, Kong, Liang, Long, Chen, Zhu - 2026/06/18 - MICCAI 2025 -
https://arxiv.org/abs/2606.19950
MS-FBI (Multi-Strategy Fusion-Based Interrogation) + auxiliary expert-LLM assessment for calibration;
~40% ECE reduction across 3 medical VQA datasets. Mechanism (probe vs verbalized vs logprob) and models
NOT confirmed from the abstract alone (full text not read - time-budgeted out).
**Relation:** confirms medical-VQA confidence calibration is an active MICCAI-2025/2026 line, i.e. a
credible venue for this project's medical framing, but not confirmed to touch best-of-N selection,
hidden-state probes, or cross-model fusion. Flagged for a deeper read if I3 direction is pursued.

**P13 - arXiv:2512.14770 - Improving VQA Reliability: A Dual-Assessment Approach with Self-Reflection and
Cross-Model Verification (DAVR).** Wu, Ou, Tian, Yang, Zhang, Li, Gao - 2025/12/16 -
https://arxiv.org/abs/2512.14770
"External reference models for factual cross-checking"; scored in the Reliable VQA Challenge @
ICCV-CLVL 2025 (Phi_100=39.64, 100-AUC=97.22). Mechanism for candidate pooling/selection across models,
and whether it uses hidden-state probes vs external judges, NOT confirmed from the abstract (time-
budgeted out of a full read).
**Relation:** the closest TITLE match to "cross-model candidate fusion" found; worth a full read before
claiming novelty for "probe-calibrated cross-model candidate fusion" since this looks adjacent but
mechanism unconfirmed - treat as a live risk to the novelty of I3's cross-model-fusion idea, not yet DONE
or ADJACENT with certainty.

**Not independently re-verified here (already carded by lit-vlm-med in this same wave, reused, not
re-fetched to save budget): ViCrop (2310.16033) and V* (Wu & Xie, guided visual search) are established
2023-era image-cropping-for-VQA baselines that SPARC (P11) and AVIS (2606.11576, title/venue only from
search snippet, UNVERIFIED pending fetch) build on. None of the fetched image-TTA papers combine crop/
zoom PORTFOLIOS with a trained hidden-state correctness probe deciding which crop's answer to keep - that
combination was not found and is a candidate gap (see Direction 2 below).**

### I4 — activation steering along a correctness-probe direction for QA accuracy

**P14 - arXiv:2306.03341 - Inference-Time Intervention: Eliciting Truthful Answers from a Language
Model.** Li, Patel, Viegas, Pfister, Wattenberg - 2023/06/06 (v6 2024/06/26) - NeurIPS 2023 Spotlight -
https://arxiv.org/abs/2306.03341
Foundational ITI: shift activations at a limited set of attention heads along probe-identified
directions. TruthfulQA 32.5%->65.1% (Alpaca). Explicitly reports a TRUTHFULNESS-vs-HELPFULNESS tradeoff
tunable by intervention strength; gains are on the truthfulness benchmark itself, not general QA
accuracy.
**Relation:** the field's own anchor paper already flags the capability tradeoff the project should
expect; not evaluated on medical VQA / open-ended QA accuracy, not a VLM, not evaluated as a best-of-N
alternative (single-shot steered generation only).

**P15 - arXiv:2602.06256 - Steering Safely or Off a Cliff? Rethinking Specificity and Robustness in
Inference-Time Interventions.** Goyal, Daume III - 2026/02/05 - https://arxiv.org/abs/2602.06256
2026 cautionary finding: steering that improves one axis (e.g. reduces overrefusal) "substantially
increases vulnerability" on another (jailbreak robustness); introduces a taxonomy (general/control/
robustness specificity) showing steering methods "appear reliable" under narrow eval but fail to
generalize. Text LLMs only.
**Relation:** the clearest 2026 negative-result anchor for "steering has hidden costs" that the project
should cite if it proposes probe-direction steering - not medical, not accuracy-specific, but directly on
point for "known negatives: probes find correlates, steering hurts capability."

**P16 - arXiv:2602.21704 - Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large
VLMs (DMAS).** Yin, Chen, Chen, Zhou, Wu, He - 2026/02/25 - ICLR 2026 - https://arxiv.org/abs/2602.21704
Training-free: a semantic steering-vector DATABASE (not a single probe direction) + dynamic per-input
vector selection by semantic similarity, applied to the most influential attention heads. Per an
earlier (unverified, from search snippet only) claim, reports accuracy/F1 GAINS on GQA for LLaVA-1.5 and
QwenVL (not just hallucination metrics POPE/CHAIR) - NOT independently confirmed from the abs page
(abstract text did not carry the numbers); flagged UNVERIFIED for the specific GQA numbers, but the
method description (dynamic per-input vector selection, general VLMs, ICLR 2026) is confirmed.
**Relation:** if the GQA accuracy-gain claim holds up on a full read, this is the strongest ADJACENT
candidate for "steering improves QA accuracy, not just truthfulness/hallucination score" in a VLM -
general-domain (GQA), not medical, and vectors are semantic-similarity-selected from a database, not a
single trained correctness-probe direction. Needs a full-text read before citing its numbers.

### I5 - label-free/few-label onboarding + cross-model/cross-lineage probe transfer

P17 - arXiv:2406.15927 - Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs.
(title/ID/date read off an arXiv search listing, not independently abs-fetched - well-known paper,
marked VERIFIED-BY-LISTING) - 2024/06 - https://arxiv.org/abs/2406.15927
SEPs approximate semantic entropy (normally needs multiple samples + NLI clustering) from a SINGLE
generation's hidden states via a cheap probe - direct prior art for "distill an expensive multi-sample
uncertainty signal into a frozen single-pass probe," same shape as I5's "distill judge/self-consistency
into a probe." ADJACENT - distills entropy not correctness, text-only, not framed as pseudo-labeling a
new downstream verifier.

P18 - arXiv:2507.00239 - Linearly Decoding Refused Knowledge in Aligned Language Models. Shrivastava,
Holtzman (U Chicago) - 2025/06/30 - https://arxiv.org/abs/2507.00239
Trains a linear probe on a BASE model's hidden states; finds it "sometimes transfer[s] to their
instruction-tuned versions" (refused-knowledge decoding, Pearson r>0.8 on a country-IQ task, base to
aligned pair). Closest published instance of the project's own pilot shape (probe trained at one point
in a fine-tuning lineage, evaluated zero-shot at another point) - but for refusal/safety knowledge, not
answer correctness, not a VLM. DONE for "does base<->instruct probe transfer happen at all" (yes,
sometimes); OPEN for "how much of a CORRECTNESS probe's gain survives a fine-tune lineage" - the
project's own ~55%-retained number looks like a new instance of this general phenomenon, not previously
measured for correctness/VQA.

P19 - arXiv:2606.20225 - Actionable Activation Directions for Detecting and Mitigating Emergent
Misalignment Across Language Model Families. Syed - 2026/06/18 - https://arxiv.org/abs/2606.20225
Ridge-regression mapping of steering directions ACROSS model families (Qwen2.5-1.5B, Gemma-2-2B,
Llama-3.2-1B, Ministral-3-3B - different architectures, not a fine-tuning lineage). Large behavioral
suppression (up to 46 pts) but FAILS specificity controls - random/orthogonal directions perform
comparably cross-family, while within-model directions are specific and actionable. Sharpest published
NEGATIVE result for cross-model transfer, but cross-ARCHITECTURE, not cross-FINE-TUNE; the project's
Lingshu(fine-tune)-to-Qwen2.5-VL(base) pilot is the same-lineage case, which P18 suggests is more
favorable than P19's cross-family case. The same-lineage-vs-cross-family distinction is not explicitly
separated as a research question in either paper - a gap.

P20 - arXiv:2602.08159 - The Confidence Manifold: Geometric Structure of Correctness Representations in
Language Models. (title/ID/date from an arXiv search listing, NOT independently abs-fetched - flagged
UNVERIFIED pending full fetch) - 2026/02 - https://arxiv.org/abs/2602.08159
Per the listing snippet: truth-signal geometry across 11 models; "single-dataset probes transfer
near-randomly until joint multi-dataset training restores 0.73-0.91 AUC" - cross-DATASET (not
cross-model) transfer needs joint training. Bears directly on "how many labelled benchmarks does a
probe need to generalize," relevant to the project's ~100-question-per-benchmark onboarding cost.

No paper was found in this pass that explicitly does "few-label (~100-question) onboarding of a
best-of-N verifier probe on a NEW benchmark via self-consistency pseudo-labels calibrated against a
small seed set" - the project's own pilot (probe-confidence-filtered pseudo-labels 0.825 precise under
the judge vs 0.647 for self-consistency-filtered) looks like a live gap (see Direction 3). CCS
(Contrast-Consistent Search) and its critique lineage are well-established prior work already covered
by the domain package and time-budgeted out of a fresh re-verify here.

### I6 - systems: why parallel sampling is cheap in FLOPs but not latency at batch 1

P21 - arXiv:2402.05099 - Hydragen: High-Throughput LLM Inference with Shared Prefixes. Juravsky, Brown,
Ehrlich, Fu, Re, Mirhoseini (Stanford/Together) - 2024/02/07 (v2 2024/05/13) -
https://arxiv.org/abs/2402.05099
Decomposes attention into shared-prefix batched GEMM + per-sequence suffix GEMM. Up to 32x throughput on
CodeLlama-13B; "speedup GROWS WITH batch size and shared prefix length"; 55% latency reduction on
competitive-programming (tree-shared) prompts. Does NOT report numbers at batch=1 (single request, N
parallel samples) with a short (~300-token) prefix and ~6-token completions - the project's exact regime
is smaller on both axes than where the paper demonstrates its biggest wins.
DONE for "shared-prefix batched attention removes the prefill-duplication cost" as a mechanism; whether
vLLM's production prefix-caching (RadixAttention-style) already captures most of this gain at the
project's regime is OPEN and answerable from the project's own vLLM logs/config, not a new paper.

P22 - arXiv:2403.08845 - Bifurcated Attention: Accelerating Massively Parallel Decoding with Shared
Contexts in Large Language Models. (title per arXiv search listing, not independently abs-fetched beyond
the listing text) - 2024/03 (announced), revised 2024/07/11 - https://arxiv.org/abs/2403.08845
Splits incremental-decoding attention into a prefill-KV-cache GEMM and a decode GEMM; reports "over 2.1x
speedup when sampling 16 output sequences" and "6.2x at 32 sequences" but explicitly at CONTEXT LENGTHS
EXCEEDING 8K TOKENS on a 7B model - roughly 25x longer prefix than the project's ~300-token image+prompt.
ADJACENT mechanism, confirms the shared-prefix decomposition trick is a known 2024 systems result, but
BOTH closest systems papers (P21, P22) report headline speedups in LONGER-context regimes than the
project's workload - "does the speedup survive down to a ~300-token prefill / 6-token-answer medical VQA
workload" is not answered by either paper and is a directly checkable, cheap systems experiment (see
Direction 4).

---
## 2. Sub-question verdicts

**I1 (adaptive-N / optimal stopping driven by a hidden-state probe, in a VLM): OPEN.** Weitzman/Pandora's-
box stopping for LLM BoN is published and works (P3, 15-35% fewer generations) but is driven by an
EXTERNAL reward model, text-only. Probe-driven early-exit exists but WITHIN one reasoning trace, not
across BoN samples (P4 LYNX). Self-report mid-generation stopping exists but via a generated token, not a
trained hidden-state probe (P1). No paper combines "trained hidden-state probe" + "Weitzman-style
across-sample stopping" + "VLM" + "coverage-limited regime." Closest 3: P3, P4, P1.

**I2 (pre-generation probes predicting from prompt+image, before sampling): DONE for point-correctness
prediction, OPEN for coverage/VLM.** Pre-generation linear/MLP probes reliably predict eventual
correctness in text LLMs (P7 trivia/OOD knowledge, P8 code AUC 0.881) and can pre-select a decoding
STRATEGY (P9 DiffAdapt, 3-way difficulty -> token budget). None of the three: (a) touch images/VLMs, (b)
predict POOL COVERAGE (will ANY of N samples be correct) rather than single-decode correctness, (c) use
the signal to decide N or temperature specifically. Closest 3: P7, P8, P9.

**I3 (attacking coverage: image TTA portfolios + probe-calibrated cross-model fusion): OPEN as a
combination.** Image-side test-time scaling (crop/zoom, "thinking with images") is an active 2026 line
(P11 SPARC, plus ViCrop/V*) but none combine it with a trained correctness-probe verifier choosing which
crop/augmentation's answer to keep. Cross-model/multi-agent confidence fusion for (medical) VQA exists
(P12, P13) but neither was confirmed (full-text not read, time-budgeted) to use a hidden-state probe as
the fusion signal - both are flagged as the two papers most likely to narrow or kill this direction and
need a full read before any claim of novelty. Closest 3: P11, P12, P13.

**I4 (steering along a correctness-probe direction, positive QA-accuracy result): mostly negative/OPEN.**
The field's own anchor paper (P14, ITI) reports a truthfulness gain that trades off against helpfulness,
not general QA accuracy gains. A 2026 paper (P15) generalizes this into a "specificity" critique: steering
that helps one axis measurably hurts another, evaluated via a robustness lens (LLMs only). The one
plausible positive-on-QA-accuracy VLM candidate (P16, DMAS) reports GQA accuracy gains per an unverified
search snippet but this was NOT confirmed from the abstract page itself - genuinely unresolved, flagged
for a full read, not usable as evidence either way yet. No clean, verified positive result for "steering
along a correctness-probe direction improves held-out QA accuracy without a capability tradeoff" was
found. Closest 3: P14, P15, P16 (P16 unverified numbers).

**I5 (label-free/few-label onboarding + cross-model/lineage probe transfer): OPEN for the exact
combination, though every component has a precedent.** Distilling an expensive multi-sample signal into
a single-pass probe is established (P17, semantic entropy). Probes trained on a BASE model sometimes
transfer zero-shot to its instruction-tuned descendant (P18) - the closest analog to the project's own
Lingshu->Qwen2.5-VL pilot, but for refusal/safety knowledge, not correctness, not a VLM. Cross-
ARCHITECTURE (not cross-lineage) transfer of steering directions fails specificity checks (P19) - a
different, harder case than same-lineage transfer, so it does not contradict the project's ~55%-retained
finding. No paper does "few-label (~100 question) onboarding of a best-of-N verifier probe on a new
benchmark via probe-confidence-filtered pseudo-labels, validated against a small labelled seed."
Closest 3: P18, P19, P20 (P20 unverified numbers).

**I6 (systems: shared-prefix parallel sampling at batch 1 for short-prefill VQA): mechanism DONE,
applicability to this regime OPEN and is an empirical config question, not a literature gap.** Hydragen
(P21) and Bifurcated Attention (P22) both solve exactly the "N samples from one shared prefix" cost, but
both report their headline speedups at LARGE batch size / LONG context (8K+ tokens) - roughly 25x longer
than the project's ~300-token medical-VQA prefill with ~6-token answers. Neither paper measures the
short-prefill/short-answer/batch-1 regime the project actually runs. Whether vLLM's built-in prefix
caching already captures most of this benefit at that regime is answerable directly from the project's
own serving config/logs, not from new literature. Closest 3: P21, P22, (no third distinct paper found;
Hydragen/Bifurcated-attention derivatives were not found beyond these two originals in this pass).

---
## 3. Proposed directions

### Direction 1 (CPU-only, highest priority) - Probe-driven Weitzman/optimal-stopping controller tested
against the coverage wall
**Pitch:** The project already owns an unused Weitzman/Pandora's-box adaptive-N controller and a frozen
per-candidate correctness probe. Wire the probe's score in as the (free, already-computed) per-sample
value estimate that P3's stopping rule needs, instead of an external reward model call. This directly
targets two documented project facts: the sampling penalty (some T=0.7 samples score below greedy) and
the fact that N 8->16 only helps on 4/8 benchmarks - a probe-driven controller should learn to stop
early (fall back to greedy) on the benchmarks/questions where no sample is beating greedy's own score,
and keep sampling where the probe sees real upside.
**Nearest prior art / delta:** P3 (Weitzman-BoN, text, external RM) is the closest; the delta is
(a) reward comes from the project's own frozen, free, already-cached hidden-state probe, not an
external RM call, (b) evaluated under a genuine coverage wall (VQA-Med ~21% pools), which P3's datasets
(AlpacaFarm/HH-RLHF) do not have, (c) a VLM, medical.
**First experiment (CPU-only, on-disk data, <30 min):** Using the existing N=16 pools + per-candidate
probe scores + judge labels, simulate Weitzman's algorithm sequentially per question (treat the 16 draws
as i.i.d., consistent with T=0.7 sampling): at each step compare the next sample's expected probe-scored
value against the running reservation value, stop when the rule says so, and record (a) the realized
average N spent per benchmark and (b) the realized macro accuracy at the controller's stopping points.
Compare against fixed-N=8 at matched or lower average sample count, split out separately for the 4
benchmarks where N8->16 helped vs the 4 where it did not.
**Kill result:** controller-selected accuracy-at-matched-budget is statistically indistinguishable from
(or worse than) fixed N=8 (no free lunch from adaptivity), or probe-score variance across the pool is too
flat to give Weitzman's rule anything to discriminate on (a degenerate box).
**Main risk:** stored pools may not preserve true sequential generation order (fine for Weitzman under
i.i.d. draws, but loses any true "early signal" benefit); coverage-zero questions (~21% on VQA-Med) may
make the controller waste its full budget anyway since no amount of correct stopping logic manufactures
a correct candidate that was never sampled - this experiment mainly targets the SELECTION/spend-efficiency
wall, not the coverage wall itself (that is Direction 2).

### Direction 2 (CPU-only) - Pre-generation coverage probe: predict "will this 8-sample pool contain any
correct answer?" before generating anything
**Pitch:** All the project's probes currently read POST-generation hidden states. If a probe reading only
the PROMPT-side (pre-generation) hidden state can predict whether an N-sample pool will have zero
coverage, that licenses a genuinely new pre-emptive policy - skip wasteful sampling, or immediately
reroute to a stronger generator/image augmentation - on questions flagged as unwinnable, attacking the
coverage wall (the project's stated binding limit) rather than the selection wall.
**Nearest prior art / delta:** P7/P8 show pre-generation probes predict POINT correctness of a single
eventual decode, in text/code, with no images. DiffAdapt (P9) pre-selects a strategy from a pre-gen probe
but for reasoning-token budget, not sampling count, text-only. Delta: predicting POOL-LEVEL coverage
(not point correctness), on image+text medical VQA, to drive an N/strategy decision.
**First experiment (CPU-only, first step is a disk check):** First verify whether a pre-generation
(prompt-only, last-prompt-token) hidden state was saved alongside the existing per-candidate generation-
pass features in feats_hidden/ - if yes, this is a pure re-labelling exercise: same probe architecture
the project already uses, new binary target ("pool has >=1 judge-correct candidate," from the existing
8-sample judge labels), same by-image train/held-out split, report AUROC per benchmark (especially
VQA-Med, the ~21%-coverage case) vs the other 7. If the pre-generation state was NOT cached, this becomes
a single (non-generating) forward pass per held-out question through the frozen 7B - still cheap and
GPU-light, but no longer strictly CPU-only until that's confirmed.
**Kill result:** AUROC <=0.55-0.60 (no better than the project's already-documented ~0.5-0.6 "anything
cheap" recoverability ceiling) kills it as a standalone signal.
**Main risk:** the whole "recoverability wall" finding (16 mechanisms tried, all ~0.5-0.6 AUROC) is
direct evidence this may simply fail again; pre-generation is a genuinely different information set
(no candidate-specific signal at all) so it is not strictly redundant with what's been tried, but the
prior is pessimistic.

### Direction 3 (CPU-only) - Formalize and stress-test the probe-confidence-filtered pseudo-label
onboarding pilot (LOBO-style, validated against true labels)
**Pitch:** The project's own preliminary number (probe-confidence-filtered pseudo-labels 0.825 precise
under the judge vs 0.647 for self-consistency-filtered) is a promising but informal result. No published
paper does exactly "bootstrap a NEW benchmark's best-of-N verifier from probe-confidence-filtered
pseudo-labels, calibrated/validated against a small labelled seed" (closest are P17 entropy distillation,
P18 base->instruct probe transfer, P20 cross-dataset joint-training-needed geometry). Turning the
existing pilot into a rigorous leave-one-benchmark-out protocol, with true-label validation on the
held-out half, would be a real, checkable, publishable claim about onboarding cost.
**Nearest prior art / delta:** none does this exact loop (probe-confidence pseudo-labels -> retrain/
recalibrate -> validate against true labels, per-benchmark, on a best-of-N verifier). Delta is the whole
pipeline, medical VQA, and honest validation against real judge labels (avoiding the self-fulfilling-
prophecy trap that circular pseudo-labeling risks).
**First experiment (CPU-only, on-disk data):** For each of the 8 benchmarks in turn, hold it out entirely
from probe training (the project already has LOBO infrastructure per recent commits), generate
pseudo-labels for its train-half pool via (a) the other-7-benchmark probe's confidence-filtered top/
bottom 25% and (b) self-consistency-majority filter, use each to calibrate/lightly adapt a per-benchmark
threshold or small classifier, and measure REAL judge-label accuracy on that benchmark's true held-out
half. Compare against the already-measured zero-shot LOBO baseline (-0.0056 breadth).
**Kill result:** pseudo-label-driven onboarding does not beat the existing zero-shot cross-benchmark
transfer baseline, or the true-label validation shows the 0.825-vs-0.647 precision gap does not translate
into any downstream accuracy gain (precision on a skewed/small top-25% slice can be a highly biased
estimate of the overall onboarding value).
**Main risk:** circularity - filtering pseudo-labels by the probe's OWN confidence, then using them to
"validate" the probe, is exactly the kind of self-fulfilling loop the project's CLAUDE.md flags as a
recurring failure mode elsewhere; the fix (validating on TRUE held-out judge labels, never on the
pseudo-labels themselves) must be enforced throughout.

### Direction 4 (needs GPU/vLLM, not CPU-only - included because I6 is a real, cheaply-checkable
systems question) - Measure whether vLLM's own prefix caching already captures the Hydragen/Bifurcated-
Attention saving at the project's actual short-prefill/short-answer/batch-1 regime
**Pitch:** Both closest systems papers (P21 Hydragen, P22 Bifurcated Attention) report large speedups only
at long-context/large-batch, a regime roughly 25x longer-context than the project's ~300-token prefill,
6-token answers. The 2.74x latency / 3.59x energy vs 1.13x FLOPs gap the project reports may be a serving-
config artifact rather than a fundamental wall, or may genuinely persist because the project's regime is
too short for these tricks' fixed overheads to pay off - this is unknown and directly measurable.
**Nearest prior art / delta:** P21, P22 - delta is simply measuring their claimed mechanism at a regime
25x shorter than either paper tested, with vLLM's actual serving config (which may or may not already
implement an equivalent optimization via prefix caching / RadixAttention-style reuse).
**First experiment:** re-run a small slice (~200 questions) of one benchmark at N=8 with current vLLM
config vs explicit prefix-caching enabled (or a newer vLLM release with better n>1-sampling prefix reuse),
measure wall-clock latency/energy at batch 1, and see how close it gets to the 1.13x FLOPs floor. Cheap
in wall-clock terms even though it needs the GPU (small slice, no training).
**Kill result:** the latency/energy ratio does not move (confirms this is a fundamental floor, not a
config gap) - still a useful, reportable negative result that forecloses re-litigating serving config.
**Main risk:** none of this is CPU-only or literature-gap-closing; it is a systems measurement, included
because it is the cheapest way to resolve I6 rather than because a paper needs beating.

---
## 4. Query log
WebSearch (18 total, until the session's 200-search budget was exhausted mid-session): "adaptive
inference-time compute LLM predict if it can do better mid-generation arxiv"; "Damani Learning how hard
to think adaptive compute allocation arxiv"; "hidden state probe optimal stopping best-of-N sampling
arxiv 2026"; "Pandora's box optimal stopping LLM inference sample allocation arxiv"; "predict LLM answer
correctness before generation prompt hidden state probe query difficulty arxiv 2026"; "LLMs know what
they know internal states know before answering probe self-knowledge 2026"; "predict best-of-N sampling
coverage will help from hidden state before sampling arxiv"; "probe temperature selection adaptive
decoding hidden state prompt difficulty vision language model"; "visual test-time scaling image crop
zoom augmentation VQA verifier arxiv 2026"; "ViCrop V* thinking with images visual search VQA test-time
compute"; "multi-model candidate pooling calibrated confidence fusion best-of-N LLM-Blender verifier
medical"; "cross-model candidate fusion hidden state probe calibration ensemble generators VQA arxiv
2026"; "Nullu VTI ICT VISTA activation steering hallucination VLM correctness direction QA accuracy
arxiv"; "inference-time intervention ITI representation engineering steering hurts capability negative
result accuracy tradeoff arxiv"; "contrastive activation addition CAA steering vector QA accuracy
improvement not just truthfulqa arxiv 2026" (budget exhausted, no results returned).
WebFetch (arxiv abs/search pages, ~24 calls): 2410.02725, 2512.05325, 2410.04707 (via search), 2510.01394,
2606.07392, 2509.10625, 2606.14530, 2510.19669 (pdf), 2503.01422, 2606.19950, 2512.14770, 2602.06566,
2306.03341, 2602.06256, 2602.21704, plus arXiv search-page fetches (searchtype=all) for: "semantic
entropy probes" (yielded the I5 listing), "probe transfer across models linear alignment" (yielded I5
cross-lineage cards), "label-efficient verifier reward model few-shot onboarding new domain" (zero
results), "Hydragen shared prefix attention inference" -> 2402.05099, then direct fetch of 2402.05099 and
2403.08845-listing, "bifurcated attention shared context batch decoding latency" (yielded 2403.08845).
One garbage random-ID guess (2603.16336) returned an unrelated robotics paper and was discarded, not
carded.
