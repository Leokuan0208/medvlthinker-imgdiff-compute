# Structure and a Little Training Beat the Gate: Adaptive-Compute Cascades and Trained Verifiers for Efficient, Accurate Medical VLMs

*Li-Wen Kuan (Leo) et al. — CVGIP 2026 (draft, restructured 2026-06-27). All numbers verbatim from real
checkpoints; none fabricated. Canonical source: `results/cascade_methods/GROUND_TRUTH_NUMBERS.md`.*

## Abstract

Deploying medical vision–language models (VLMs) is expensive: a 32B reasoning model spends ~11 s and ~6 kJ
per question (batch-1), a 7B model ~0.2 s. The natural fix is a **cascade** — answer cheaply, escalate only the hard
cases — but this requires *deciding* which cases to escalate, and *what to do* once escalated. We make two
findings that pull in opposite directions and a unifying explanation.

**First, a hard negative: the routing decision cannot be improved by any training-free signal.** Across a
dozen escalation/selection signals (confidence margin, entropy, Gini, self-verification P(True), conformal
prediction, learned routers, cross-model agreement, multi-resolution stability, semantic self-consistency),
all hit the same **recoverability ceiling** (AUROC ≈ 0.5–0.69): whether a cheap model's error is fixable is
nearly unpredictable from cheap features. The same wall appears for *selection* — given N sampled answers,
no training-free rule beats picking one at random (the **luck floor**) — and even for verifiable
bounding-box outputs. Error correlation explains it: the cheap and strong models fail together (φ = 0.37).

**Second, two levers nonetheless give large, real gains** — neither of which is "a better gate":

1. **Structure (efficiency).** The *Adaptive-Compute Cascade* (ACC) routes across *compute configurations*
   of the same models — 7B no-think → **32B's fast no-think mode** → 32B's slow think mode — with the two
   no-think legs' **agreement** deciding when to pay for thinking. Because reasoning *over-thinks* perception
   VQA (32B no-think ≥ 32B think on competent sets), the slow think pass fires on only ~15% of queries.
   At parity accuracy (ALL-6, MedVLThinker): **latency 11.34 s → 2.27 s (−80%), FLOPs halved (100→52%),
   energy ~5× lower (6.3 kJ → 1.2 kJ)**; on ALL-5, **8.9 s → 0.44 s** and FLOPs to 25%. The win is the
   *structure* (the no-think intermediate tier), not the gate — every gate gives the same frontier.

2. **A little training (accuracy).** While *training-free* selection is luck-floored, a small **trained
   outcome verifier** that scores P(correct | image, question, candidate) and selects best-of-N **breaks the
   floor**, recovering **40–78%** of the oracle gap — for free-text answers (**49%** pooled over four
   datasets, transferring to a fifth) *and* for structured bounding boxes (SLAKE organs **40%**; the real
   **MS-CXR** chest-X-ray pathology benchmark **78%**, a 5.6× lift; bootstrap-significant). The verifier
   discriminates correct from incorrect candidates at **AUROC 0.924**, beats a zero-shot 32B verifier despite
   being 5× smaller, behaves as a genuine test-time-scaling method, and — because reasoning barely helps
   open-ended — **a 7B with the verifier (0.501) beats the 32B's single pass (0.462, same held-out split)**: test-time compute
   beats parameters where parameters do not help.

The connective insight: **training-free routing/selection over a frozen model is luck-floored; the two
things that move the needle are *structure* (ACC) and *a little training* (the verifier).** We additionally
show the routing ceiling is partly a **benchmark artifact**: the same confidence signal jumps from ~0.6
AUROC on multiple-choice to ~0.87 on open-ended free-text (a *discreteness* effect), so medical-VLM cascades
should be evaluated open-ended. We release ACC, the trained verifier, and the full characterization.

---

## 1. Introduction

A medical VLM takes an image (radiograph, pathology slide, …) and a question, and returns an answer. The
strongest models are large reasoning models that emit a long `<think>` trace; they are accurate but slow and
energy-hungry (≈ 11 s, ≈ 6 kJ per question at batch 1 for a 32B), while a 7B answers directly in ≈ 0.2 s.
A **cascade** — cheap model first, escalate hard cases to the expensive one — is the standard route to
efficiency. Its quality hinges on two decisions: the **gate** (which queries to escalate) and the **action**
(what computation to run on escalation). This paper characterizes the limits of the first and finds leverage
in the second, plus a separate training-based lever for accuracy.

**The gate is saturated.** A frozen confidence/agreement/consistency signal can rank queries by how *likely
the cheap model is wrong*, but what a cascade actually needs is whether *the expensive model will fix it*
(recoverability). We show this is nearly unpredictable from any cheap signal (AUROC ≤ 0.69), because the two
models' errors are correlated (φ = 0.37); 58% of the cheap model's errors the strong model also misses. No
decision rule beats a plain confidence threshold.

**The leverage is structural (ACC).** Instead of a better gate, we change *what escalation runs*. The 32B has
a *fast no-think mode* (≈ 0.34 s) that, on perception VQA, is **as accurate as or better than** its slow
think mode (thinking over-thinks perception). Inserting 32B-no-think as an intermediate tier — gated by the
agreement of the two no-think legs — lets the slow think pass fire only on the ~15% reasoning residual,
collapsing latency/energy at parity accuracy (§4–§5.1). The cheap-model latency is ~80× lower than think,
so removing think calls dominates the savings.

**Training breaks the selection floor (the verifier).** Routing *between* models is capacity-bound, but a
different question — given N samples from *one* model, which is correct? — admits a *trained* answer. A small
LoRA verifier scoring P(correct | image, question, candidate), used for best-of-N selection, recovers
40–78% of the oracle gap for both answers and boxes, where every training-free selector is luck-floored.

**Open-ended evaluation matters.** The pessimistic routing AUROC (~0.6) is partly an artifact of
multiple-choice: a single A/B/C/D letter is maximally discrete. On open-ended free-text the same confidence
signal reaches AUROC ~0.87. We therefore evaluate selection/verification open-ended throughout.

**Contributions.** (i) **ACC**, a compute-configuration cascade with large measured latency/energy/FLOPs
savings at parity and a per-benchmark safety guarantee. (ii) A **luck-floor characterization**: a dozen
training-free escalation/selection signals, all capped at the recoverability ceiling, with an
error-correlation explanation; extended to actions, cross-family peers, the language prior, and structured
outputs. (iii) The **open-ended ceiling-break**: routing signal is a discreteness artifact (MCQ ~0.6 →
open ~0.87). (iv) A **trained outcome verifier** that breaks the selection floor for free-text answers and
bounding boxes (incl. real MS-CXR), 2-seed and bootstrap-significant, a test-time-scaling method that lets a
7B beat the 32B single pass. (v) Honest framing throughout: the agreement *gate* mechanism is shared with
prior cascading work — our novelty is the compute-configuration *structure* and the trained-verifier
*application/unification*, not the signals themselves.

---

## 2. Related Work

**Efficient cascades / routing.** FrugalGPT-style cascades escalate to a larger model by a post-generation
confidence/benefit signal; agreement-based cascading (ABC) escalates on ensemble disagreement; CAR uses
confidence-aware routing for reasoning. Our **agreement gate** is, by design, in this family — we do not
claim it as novel. What is new is the *compute-configuration* tiering it controls: the large model's
no-think mode as an intermediate tier, exploiting that reasoning over-thinks perception. Pre-generation query
routers (learned on features) are a parallel line; we show they do not beat confidence here (§5.2–§5.3).

**Confidence, selective prediction, conformal.** MSP/Chow's rule, entropy, Gini/DOCTOR, energy scores,
split-conformal (CP-Router/LAC), and learned deferral (L2D) are the training-free gate family we benchmark;
all cluster at the recoverability ceiling.

**Verifiers / reward models for best-of-N.** Generative verifiers (GenRM) cast reward modeling as
next-token prediction for best-of-N in *text*; vision-language process reward models rerank reasoning steps;
medical generator-verifier pipelines exist for *data synthesis*. We instead train an *outcome* verifier for
**inference-time best-of-N selection in medical VQA**, and unify it across **free-text answers and
bounding-box grounding** — a combination we did not find in prior work — as a constructive counter to
*Verification Mirage* [2605.10850], which concluded self-verification fails in medical VQA and pointed to
retrieval, not a trained verifier.

**Open-ended routing / agreement signals.** Semantic-agreement cascades for text LLMs use cross-model
agreement; we make the multiple-choice-vs-open-ended *discreteness* claim and show the routing ceiling is an
MCQ artifact.

---

## 3. Setup, metrics, and definitions

**Models.** Cheap legs: 7B medical VLMs (MedVLThinker-7B, Lingshu-7B). Strong leg: the 32B counterpart, run
in no-think and think modes. Cross-family peers (§5.3): InternVL, Phi-3.5-V. Verifier base (§6):
Lingshu-7B; box-verifier base: Qwen2.5-VL-7B.

**Data / pools.** Six medical-VQA **benchmarks**: PMC-VQA, SLAKE, VQA-RAD, PathVQA, MMMU-medical,
MedXpertQA-MM. **ALL-6** = all six (MedXpert contributes two splits, so per-split tables show seven columns);
**ALL-5** = ALL-6 minus MedXpert (near-chance for both models); **COMPETENT-4** = SLAKE/VQA-RAD/PathVQA/PMC.
Open-ended (§5.3, §6): free-text SLAKE/VQA-RAD/PathVQA/Kvasir-VQA-x1, + RadImageNet-VQA (transfer), graded
by a neutral LLM judge. Grounding (§6.2): SLAKE organ boxes and the real **MS-CXR** chest-X-ray
phrase-grounding benchmark (PhysioNet; 1448 boxes / 1047 images).

**Cost model.** One model forward costs `F = 2·N·(P+G)` FLOPs (N params, P prompt tokens incl. vision, G
generated). `N₇ = 7.6e9`, `N₃₂ = 33.0e9`. For a cascade with per-tier cost `c` and escalation probabilities
`e₀` (past tier 0) and `e₁` (to think):
```
C = c_T0 + e₀·c_T1 + e₁·c_T2.                                                    (1)
```
`FLOPs%` (≡ "backbone%") = Σ(cascade FLOPs)/Σ(always-32B-think FLOPs). Latency/energy use the same Eq. (1)
with per-tier costs from real batch-1 measurements; energy is NVML-integrated `E=Σ(P_k+P_{k+1})/2·Δt`. The
think-tier latency fit has R² = 0.99 over 5,440 measured 32B-think queries. *(All ACC efficiency numbers use
the native batch-1 cost methodology of `master_data.csv`; an earlier co-resident `rt_cascade` estimate was
superseded.)*

**Confidence signals.** From a greedy decode over candidate letters with logprobs `ℓ₁ ≥ ℓ₂ ≥ …` and
softmax `pᵢ = e^{ℓᵢ}/Σⱼe^{ℓⱼ}`: **margin** `m = ℓ₁ − ℓ₂`; **MSP** `p₁`; **entropy** `−Σ pᵢ ln pᵢ`;
**Gini** `1 − Σ pᵢ²`. A gate escalates iff `m < τ`.

**Evaluation metrics.** *Accuracy* = exact-match (MCQ) or LLM-judge correctness (open-ended).
*AUROC* of a score `s` for a binary correctness label `y` is the rank statistic
`AUROC = P(s(positive) > s(negative))`. *Oracle@N* `= E_x[max_{i≤N} 1(aᵢ correct)]` (an oracle picks a
correct sample iff one exists). *Gap captured* by a selector with accuracy `acc(â)`, baseline `greedy`:
```
frac = (acc(â) − greedy) / (oracle@N − greedy).                                  (2)
```
*Guard* (safety) = average number of benchmarks, per seed, on which the cascade falls below the always-7B
baseline (0 = never worse than the cheap model). *IoU* for grounding; a box is correct iff `IoU ≥ θ`, θ=0.3.

---

## 4. Method I — the Adaptive-Compute Cascade (ACC)

ACC is a three-tier cascade over *compute configurations* of the **same** two models:
```
T0 = 7B no-think @cap320     (≈0.21 s)
T1 = 32B no-think @cap320    (≈0.34 s)
T2 = 32B think @fullres      (≈11.3 s, batch-1)
```
**Why a no-think middle tier.** Reasoning *over-thinks* perception VQA: at cap320 the 32B's no-think mode
matches or beats its think mode on competent sets — SLAKE **0.849** (no-think) vs 0.764 (think, +0.085),
VQA-RAD **0.853** vs 0.776 (+0.077). So most escalations need *capacity*, not *reasoning*; T1 absorbs them.

**Tier-0 gate (margin).** Output T0 iff `m₀ ≥ τ₀`; else escalate to T1.

**Think gate (agreement, ACC-v2).** Let `ŷ_{T0}, ŷ_{T1}` be the two no-think predictions. Fire the expensive
think tier only when the no-think legs **disagree**:
```
disagree = 1[ ŷ_{T0} ≠ ŷ_{T1} ];   s₁ = disagree + ε·(−m_{T1}),  ε = 1e-6;   fire T2 iff s₁ > τ₁.   (3)
```
The integer term selects disagreements; the tiny `ε·(−margin)` tiebreak makes `τ₁` fire on only the
*lowest-margin* disagreements, so think fires ~15% (not the full ~32% disagreement rate). ACC-v3 tightens
this to a conjunction `fire_think = 1[ŷ_{T0}≠ŷ_{T1}] ∧ 1[m_{T1} < τ₁']`. The only fitted parameters are the
scalars `τ₀, τ₁`, chosen on held-out PMC-VQA-train calibration to reach parity at minimum latency. *(The
agreement mechanism (3) is the ABC family; the contribution is the tiered compute-configuration it gates.)*

**Cost.** Plugging measured per-tier costs into Eq. (1) with `e₀ = 71.7%`, `e₁ = 15.1%` (ALL-6) yields the
savings in §5.1. The savings come from the third term: `e₁` drops from ~69% (a 2-tier 7B→think cascade) to
15%, and `c_T2 ≈ 33·c_T1` (and ~87× the 7B leg), so the avoided think calls dominate.

---

## 5. Results

### 5.1 ACC: large efficiency gains at parity, and it is the *structure* not the gate

At parity accuracy (always-32B-think) on ALL-6 (MedVLThinker, calibration held out):

| system | acc | esc₀ | think | FLOPs% | latency | energy | guard |
|---|---|---|---|---|---|---|---|
| always-7B-nt@cap320 | 0.5262 | 0% | 0% | 8.4% | 0.13 s | 20 J | — |
| always-32B-nt@cap320 | 0.5573 | 100% | 0% | 36.2% | 0.23 s | 78 J | 0.0 |
| **always-32B-think [PARITY]** | **0.5723** | 100% | 100% | 100% | **11.34 s** | **6319 J** | 0.0 |
| **Ours: ACC-v2 (agreement)** | **0.5693** | 71.7% | 15.1% | **52.0%** | **2.27 s** | **1182 J** | **0.0** |
| ACC-v1 (margin gate) | 0.5687 | 66.0% | 19.5% | 53.9% | 2.69 s | 1417 J | 0.0 |
| CASP-Stability (trained gate) | 0.5698 | 74.3% | 11.0% | 49.0% | 1.77 s | 899 J | 0.05 |

**Headline (ALL-6): 11.34 s → 2.27 s (−80% latency), FLOPs 100% → 52%, energy 6.3 kJ → 1.2 kJ (~5×), at
−0.003 accuracy and guard 0 (never worse than the 7B on any benchmark).** On **ALL-5** (excluding near-chance
MedXpert): always-think 8.88 s/4916 J → **ACC-v2 0.44 s/173 J, FLOPs 24.9%** (−95% latency, ~28× energy).

**It is the structure, not the gate.** Holding the 3-tier configuration fixed and swapping in every
training-free gate — MSP/Chow, entropy, Gini/DOCTOR, AutoMix self-verify, FrugalGPT-style learned,
Jitkrittum L2D — they all land on the same accuracy–FLOPs frontier (FLOPs 49–62%, latency 1.8–8.0 s at
parity); agreement is merely the cheapest point. Bootstrap CI on ACC-v2 accuracy: ALL-6 [0.5608, 0.5820],
ALL-5 [0.6372, 0.6562]; latency CI [2.6, 9.8] s. ACC-v3 (confidence-tightened think gate) further cuts think
19%→14% and FLOPs to 52.6% at equal accuracy (20/20 seeds at parity).

**Generalizes across families/architectures.** The "no-think ≥ think on perception" premise and the
ACC savings reproduce across five families and three architectures (Lingshu, QoQ, Chiron, MedGemma): e.g.
Lingshu ALL-6 FLOPs 77.8% → 48.6% at parity. (Full per-family tables: `MASTER_TABLES.md`.)

### 5.2 The luck floor: training-free routing and selection are bounded

The clean negative that motivates everything below: **over a frozen model, no training-free decision rule
beats trivial baselines** — for the gate, the action, *or* selection.

**(a) The gate is saturated.** A cascade needs *recoverability* — will the strong model fix this error? —
not just *cheap-model-wrongness*. Across **12 signal families** (margin, MSP/Chow, entropy, Gini/DOCTOR,
energy, hidden-state probe, self-verification P(True), conformal/CP-Router, learned GBM router, cross-model
agreement, multi-resolution stability, semantic self-consistency), recoverability AUROC sits at **0.5–0.69**;
a hidden-state probe reaches only 0.60 vs confidence 0.68. The cause is error correlation:
`P(32B wrong | 7B wrong) = 0.584` and `φ = 0.372` (φ = the correlation of the two models' right/wrong
outcomes) — the models fail together. Holding ACC's config fixed, every gate lands on one frontier (§5.1),
so no gate is "better." This matches theory: Jitkrittum et al. (*When does confidence-based cascade deferral
suffice?*, NeurIPS 2023) prove the optimal deferral rule needs *both* models' confidence and that
confidence-only deferral is fundamentally limited — exactly the wall we hit empirically on medical VLMs.

**(b) Cross-family complementarity is real but unexploitable.** Two independently-trained VLMs have large
*oracle union* headroom: `union(7B | InternVL) = 0.753`, `union(7B | InternVL | Phi) = 0.801` vs always-32B
0.645. But a learned router on frozen peer signals captures none of it: best learned router 0.621 ≈ always-7B
0.622; a SigLIP image+text router is at chance (AUROC 0.50). The headroom exists; it is just not addressable
with cheap features.

**(c) The cheap model leans on a language prior.** **56.9%** of the 7B's medical-VQA answers are unchanged
when the image is blanked, and image-sensitive vs insensitive questions have nearly equal accuracy
(0.620 vs 0.625) — much "VQA" is answerable from text alone, capping what better *visual* routing can buy.

**(d) The action axis is capacity-bound.** Decomposing the 7B's errors by repair (look-closer / think /
scale-up): cheap same-model repairs recover **14%** (11–17%) of errors the 32B *also* misses — a genuinely
novel observation — but they are *unharvestable*: repairs break as many as they fix, and a confidence-gated
repair ladder *loses* at parity (43% vs 39% compute). The 32B's edge is capacity, not a cheap transform.

**(e) Selection is luck-floored (the headline negative).** Sample one open-ended model N=8 times: a correct
answer is often present (large oracle gap, e.g. SLAKE-open greedy 0.730 → oracle@8 0.879), but the model
cannot say *which*. Every training-free selector ties or trails random-pick (0.720): self-verification
P(Yes) **0.715**, 32B pointwise 0.746, 32B listwise 0.758, fusion 0.743 — none beats the 32B single pass
(0.819); candidate **synthesis backfires** (0.774). A **majority trap** explains it: the correct answer is a
minority vote in 74–90% of recoverable cases. The gap is *sampling luck, not latent knowledge*. The same
holds for verifiable **bounding boxes**: with a competent grounder (Qwen2.5-VL) the spatial self-consistency
(medoid) selector ties greedy (0.246 vs 0.254, AUROC 0.56) — the apparent signal with a *weak* grounder
(AUROC 0.82) is an artifact of incompetence (diverse wrong boxes make rare agreements spuriously predictive).

### 5.3 The open-ended ceiling-break: the routing signal is partly an MCQ artifact

The pessimistic gate AUROC (~0.6) is inflated-pessimistic by multiple-choice: a single A/B/C/D letter is a
*maximally discrete* target. Re-running the **same** confidence signal on **open-ended free-text** answers,
detection of cheap-model errors jumps to **AUROC ~0.87** (Lingshu-7B→32B confidence 0.866; pooled over three
datasets 0.846, 95% CI [0.830, 0.862]; P(AUROC ≤ 0.6) = 0). It is a *discreteness*, not answer-length,
effect (token-F1 grading agrees). Two consequences: (i) confidence-gated cascades *do* work open-ended; and
(ii) the right setting to study selection/verification is open-ended — which is where the verifier (§6)
operates. The gate itself, however, *remains* near-optimal at plain confidence even here — no
self-consistency, semantic-entropy, or self-verification signal beats it (gate-hunt: confidence 0.866 vs
exact-SC 0.845, semantic-SC 0.806, P(True) 0.755). So even in the favorable regime, *training-free* selection
is still floored — setting up the one thing that is not.

---

## 6. Method II — a trained outcome verifier breaks the selection floor

§5.2(e) shows *training-free* selection is luck-floored. The constructive complement: a small **trained**
verifier escapes it, for free-text answers **and** structured boxes.

**Why a trained verifier matters *here* specifically.** In *general* LLM reasoning, trained-verifier
best-of-N barely beats plain self-consistency — recent work reports near-parity (e.g. self-certainty
best-of-N, NeurIPS 2025 [2502.18581]; aggregation studies [2510.13918]) — so a learned verifier is often
seen as not worth its cost. Medical open-ended VQA is the opposite regime: self-consistency *fails* (the
majority trap, §5.2e), the strong model barely helps (§5.1), yet the oracle headroom is large. That is
exactly where a learned selector should pay off — and it does.

**Verifier and selection.** A LoRA-fine-tuned VLM verifier with parameters φ scores a candidate by the
probability it assigns to "Yes" vs "No" at the final token of the prompt *"…Is the proposed answer correct?
Answer Yes or No."*:
```
s_φ(v,q,a) = P_φ(Yes | v,q,a) = e^{z_Yes}/(e^{z_Yes}+e^{z_No}).                  (4)
```
It is trained on per-sample correctness labels `y` (LLM-judge for answers; `y = 1[IoU ≥ θ]` for boxes) by
binary cross-entropy on the Yes/No token (base frozen, LoRA only):
```
L(φ) = −Σ [ y·log s_φ + (1−y)·log(1−s_φ) ].                                       (5)
```
Given N samples `a₁…a_N ~ M(·|v,q)`, **best-of-N selection** returns `â = argmax_{i≤N} s_φ(v,q,aᵢ)` (Eq. 4),
and we report gap-captured by Eq. (2) against the greedy baseline.

**Why a 32B-judged 7B verifier beating the 32B is not circular.** The LLM judge is an *automated grader*,
not a knowledge oracle: its sole job is deciding whether a free-text answer matches the **gold** answer
(a semantic substitute for exact-match, which is too brittle for free text). The labels `y` therefore derive
from the dataset **answer key**, not from the 32B's medical knowledge — the grader could be a human or
exact-match. The verifier (a frozen 7B + LoRA) is trained only on the **7B's own samples** and learns to
*discriminate* correct answers, a strictly easier task than *generating* them: the 7B's N samples already
contain a correct answer in 59% of pooled cases (the oracle), so the verifier need only identify it — which
is why a *7B* selector suffices and imports no 32B capability. The comparison is thus between two deployment
strategies — scaling to a 5× model *zero-shot* vs. sampling a small model + a small *supervised* verifier —
and the latter wins; the in-domain supervision (a few thousand gold labels) is precisely the contribution
(and its cost). AUROC 0.924 and the −0.047 image-ablation confirm the verifier learned genuine visual
discrimination, not judge-mimicry (the judge needs the gold answer; the verifier does not).

### 6.1 Free-text answers: 49% of the oracle gap (pooled over four datasets)

| dataset | greedy | self-consist. | zero-shot 32B verify | **trained verifier** | oracle@8 | gap captured |
|---|---|---|---|---|---|---|
| PathVQA-open | 0.352 | 0.349 | 0.357 | **0.441** | 0.513 | **56%** |
| Kvasir-open (GI, OOD) | 0.282 | 0.282 | — | **0.405** | 0.493 | **58%** |
| VQA-RAD-open | 0.519 | 0.500 | — | **0.611** | 0.722 | **45%** |
| SLAKE-open | 0.738 | 0.738 | 0.758 | **0.762** | 0.895 | 15% |
| **pooled (n=1064)** | 0.413 | 0.411 | — | **0.501** | 0.592 | **49%** |

**Training is the active ingredient.** Zero-shot self-verification P(True) is luck-floored (PathVQA 0.319 <
greedy); LoRA-training the *same* model on judge labels captures **49%** of the oracle gap (gain **+0.088**
over greedy; lifting every dataset), and the trained **7B** verifier beats the **zero-shot 32B** verifier
(0.357 on PathVQA) by +0.08 despite being 5× smaller. Bootstrap (best-of-8 vs a single random sample): gain
**+0.116, 95% CI [+0.092, +0.139]** (n=1064, excludes 0); and over the **32B itself**, +0.039, 95% CI
**[+0.010, +0.066]** — a modest but significant win over the 5× model. *Argmax is the correct selection
rule:* verifier-*weighted* voting (0.489) and a score×count hybrid (0.470) are both **worse** than plain
argmax (0.501), because the majority trap (§5.2e) contaminates even score-weighted voting.

**It genuinely discriminates (not a "lazy verifier").** Across all 8,512 candidates, the verifier's score
separates correct from incorrect at **AUROC 0.924** (mean score 0.749 vs 0.171), and a blank-image ablation
drops it −0.047 — it uses the image, refuting the Verification-Mirage "lazy verifier" failure mode.
(Fig. `figs/limits/fig_verifier_discrimination.png`.)

**Generalizes (datasets, modality, and across generators).** The pooled-4 verifier transfers **zero-shot to a
fifth, held-out** dataset (RadImageNet-VQA: 0.329 → 0.353, +0.024, 13% of its gap) and across modality
(Kvasir-OOD). **Across generator models:** applied to a *different* model's answers (MedVLThinker-7B), the
Lingshu-trained verifier still lifts SLAKE 0.543 → 0.620 (49%) and VQA-RAD 0.395 → 0.520 (61%) — it is
largely generator-agnostic. **Across base models (honest, mixed):** trained *from scratch* on a MedVLThinker-7B
base it works on SLAKE (0.564 → 0.622, 42%) and is pooled-positive (25%) but *fails* on VQA-RAD's tiny split
(n=54: 0.500 → 0.470) — so the method is not uniformly robust across bases, and a stronger base (Lingshu) makes
a stronger verifier (its transfer 49–61% exceeds a from-scratch MedVLThinker verifier's 25%); the Lingshu
result (49%, four datasets, two seeds) is the validated headline. **Data-efficient:** it needs only
${\sim}6{,}000$ judge labels in total to reach this lift.

### 6.2 Structured outputs: the same principle holds for bounding boxes

The luck floor generalizes to grounding, and so does the fix. We LoRA-train a **box-verifier** (Qwen2.5-VL)
to judge "does this drawn box localize the {target}?" and select best-of-8 boxes.

| benchmark | greedy | SC-medoid (training-free) | zero-shot box-verifier | **trained box-verifier** | oracle@8 | gap captured |
|---|---|---|---|---|---|---|
| SLAKE organs (n=487) | 0.197 | 0.164 | 0.177 | **0.255 / 0.257** | 0.343 | **40% / 53%** |
| **MS-CXR pathology (n=435, real)** | 0.041 | 0.053 | 0.115 | **0.232 / 0.230** | 0.285 | **78% / 77%** |

Both training-free selectors (SC-medoid, zero-shot) sit at the luck floor; the **trained** box-verifier
recovers 40–78% of the oracle gap, 2-seed-robust. On the real **MS-CXR** chest-X-ray benchmark the gain is
**+0.191, 95% bootstrap CI [+0.152, +0.232]** (n=435), a **5.6× lift** over greedy. (It lifts *selection over
a weak base grounder* — not a trained SOTA grounder like MedGround-R1, whose checkpoint is unreleased — but
the principle, "training breaks the luck floor for structured outputs too," is the point.) *Training is the
active ingredient here as well:* on MS-CXR the zero-shot box-verifier reaches 0.115 (30% of gap), so training
**doubles** the captured gap (30% → 78%). Fig. `figs/limits/fig_trained_verifier_unified.png`.

### 6.3 A test-time-scaling method; and test-time compute beats parameters

Best-of-K accuracy rises monotonically with the sample budget while random stays flat
(K = 1,2,4,8: **0.385 → 0.425 → 0.476 → 0.501**; oracle@8 0.592; random ~0.39; Fig.
`figs/limits/fig_verifier_scaling.png`); extending to K=16 (a fresh, larger sample) it keeps rising with
diminishing returns. So the verifier converts test-time compute into accuracy — the defining TTS property.

**Compute beats parameters where parameters don't help.** Because reasoning barely improves open-ended
medical VQA (§5.3), the 32B's single pass scores only **0.462** (same held-out split) pooled — *below* the 7B with verifier-bo8
(**0.501**), at ~3.7× the 32B's param-FLOPs. Per dataset the 7B+verifier beats the 32B exactly where scaling
fails — PathVQA 0.441 vs 0.377 and Kvasir 0.405 vs 0.326 (the two hardest sets) — and loses on the two where
the 32B is genuinely stronger, SLAKE (0.762 vs 0.829) and VQA-RAD (0.611 vs 0.648; per-dataset n are small —
VQA-RAD n=54 — so per-dataset signs are directional, the pooled n=1064 win is the solid claim)
(Fig. `figs/limits/fig_verifier_pareto.png`). The verifier is thus the accuracy-optimal operating point, not
dominated by simply using a 5×-larger model.

---

## 7. Discussion & limitations

**What is and isn't novel.** ACC's *agreement gate* (Eq. 3) is the ABC family; the contribution is the
*compute-configuration structure* it controls (the large-no-think intermediate tier exploiting that reasoning
over-thinks perception). The verifier's *mechanism* (a generative outcome verifier for best-of-N) follows
GenRM; the contribution is its *application + unification* — inference-time best-of-N for **open-ended medical
VQA and grounding**, as a constructive counter to Verification-Mirage. We do not claim new signals or new
reward-model machinery.

**Honest scope.** The ACC over-thinking premise (no-think ≥ think) holds on the *perception* benchmarks
(COMPETENT-4: SLAKE/VQA-RAD/PathVQA/PMC). **MMMU-medical is competent (32B-think 0.688), not near-chance** —
but it is a *reasoning* benchmark where thinking helps (no-think 0.624 < think 0.688), so the over-thinking
mechanism does not apply to it; **MedXpert-MM is the genuinely near-chance set** (7B at/below the 4-option
chance level, 0.23–0.26) and is excluded from headline efficiency claims. Both still appear in the full ALL-6
tables. The verifier (open-ended) is evaluated on the datasets that *have* free-text answers — SLAKE,
VQA-RAD, PathVQA (the core three), plus Kvasir (OOD) and RadImageNet (transfer); PMC-VQA and MMMU are
MCQ-only, so they are absent by *format*, not by selection. The verifier lifts *selection over a frozen
generator*; it does not beat a trained SOTA grounder in absolute IoU. Latency/energy are calibrated batch-1 (Eq. 1), not a single end-to-end wall-clock; FLOPs are
exact. The free-text per-dataset spread is wide (SLAKE 15% — little headroom — to Kvasir 58%); the 40–78%
headline is the cross-output-type range.

**Why the two levers and not a better gate.** Both findings reduce to the recoverability ceiling: *between*
frozen models, the fixable-vs-not signal is absent (so improve the *structure* instead), and *within* a
frozen model, which-sample-is-right is not zero-shot-surfaceable (so *train* a verifier). Structure and a
little training are the levers precisely because the frozen-model signal is luck.

## 8. Conclusion

For efficient, accurate medical VLMs, the routing/selection decision cannot be out-engineered with
training-free signals — a dozen of them hit the same recoverability ceiling, selection sits at a luck floor,
and the two frozen models fail together. Yet two levers give large, real gains: **structure** — the
Adaptive-Compute Cascade, whose large-model no-think tier cuts latency −80%, FLOPs to ~½, and energy ~5× at
parity — and **a little training** — an outcome verifier that breaks the selection luck floor, recovering
40–78% of the oracle gap for both answers and bounding boxes (incl. a real chest-X-ray benchmark), behaving
as a test-time-scaling method that lets a 7B beat a 32B. We also show the routing ceiling is partly an
MCQ artifact (AUROC ~0.6 → ~0.87 open-ended), so medical-VLM cascades should be evaluated open-ended. We
release ACC, the trained verifier, and the full negative-result characterization.

---

## Reproducibility index
ACC: `src/cascade_methods/{acc.py, acc_v2.py, acc_v3_confgate.py}`; numbers `results/cascade_methods/master_data.csv`
(canonical) + `GROUND_TRUTH_NUMBERS.md`; math `METHOD_MATH.md`/`METHOD_ACC.md`. Gate bake-off:
`{compare,evaluate,gate_compare,final_comparison}.py`. Luck floor: `{strong_fixes_genuinely_unknown,
knowledge_feasibility_bytype,crossfamily_agree,select_eval,ground_analyze}.py` + writeups
`{RECOVERABILITY_IS_CAPACITY_BOUND,OPENENDED_SELECTION_LUCKFLOOR}.md`. Open-ended: `src/labeling/run_openvqa.py`,
`src/cascade_methods/open_cascade_analyze.py`. Verifier: `src/training_methods/{run_lora_verifier_open.py,
run_lora_box_verifier.py, verifier_image_ablation.py, verifier_transfer_eval.py, verifier_scaling_curve.py}`;
grounding `src/labeling/run_ground_{slake,mscxr}.py`; figures `paper/make_*_fig.py`. **No number in this paper
is fabricated; all trace to a checkpoint via `GROUND_TRUTH_NUMBERS.md`.**
