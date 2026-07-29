# Structure and a Little Training Beat the Gate: Adaptive-Compute Cascades and Trained Verifiers for Efficient, Accurate Medical VLMs

*Li-Wen Kuan (Leo) et al. — conference version (distilled from the long-form manuscript), 2026-07.
All numbers verbatim from real checkpoints; none fabricated.*

## Abstract

Deploying medical vision–language models (VLMs) is expensive: a 32B reasoning model spends ≈ 11 s and ≈ 6 kJ
per question (batch-1), a 7B model ≈ 0.2 s. A **cascade** — answer cheaply, escalate only the hard cases — is
the standard route to efficiency, but it requires *deciding* which cases to escalate and *what to do* on
escalation. We report a hard negative, two positive levers, and a unifying explanation. **The luck floor:**
over a *frozen* model, no *training-free* signal beats trivial baselines — a dozen escalation signals hit the
same recoverability ceiling (AUROC ≈ 0.5–0.69) because the two models fail together (φ = 0.372; competent-set
P(32B wrong | 7B wrong) = 0.584), and the same floor holds for selection and even bounding boxes. **Two
levers move the needle, neither a "better gate."** (1) *Structure:* the Adaptive-Compute Cascade (ACC) routes
across compute configurations — 7B-no-think → the 32B's fast no-think mode → 32B-think — exploiting that
reasoning over-thinks perception, cutting latency 11.34 s → 2.27 s (−80 %), FLOPs to 52 %, energy ≈ 5× at
parity; and a faithful cross-family MCQ margin cascade matches the 32B at −17 … −69 % FLOPs, with a reasoning
think tier adding MMMU accuracy (+0.03 … +0.12) across three families. (2) *A little training:* a small
trained outcome verifier for best-of-N selection **breaks the luck floor** and **beats the strong 32B/38B on
accuracy across three families** (Lingshu 0.421 vs 0.331, MedVLThinker 0.344 vs 0.277, InternVL3 0.255 vs
0.218) and held-out OOD, for free-text answers (35–49 % of the oracle gap) and boxes (SLAKE 40–53 %; the real
MS-CXR chest-X-ray benchmark 77–78 %, a 5.6× lift). It discriminates correct candidates at AUROC 0.924, beats
a zero-shot 32B verifier despite being 5× smaller, and transfers zero-shot across architectures. We also show
the routing ceiling is partly an MCQ artifact (AUROC ≈ 0.6 → ≈ 0.87 open-ended). We release ACC, the
verifier, and the full characterization.

**Keywords:** medical VLM, cascade, test-time scaling, trained verifier, best-of-N, efficiency, grounding.

## 1. Introduction

A medical VLM maps an image (radiograph, pathology slide, …) and a question to an answer. The strongest
models are large reasoning models emitting a `<think>` trace — accurate but slow and energy-hungry (≈ 11 s,
≈ 6 kJ per question at batch-1 for a 32B) versus ≈ 0.2 s for a 7B. A **cascade** — cheap model first,
escalate the hard cases — is the standard route to efficiency; its quality hinges on the **gate** (which
queries to escalate) and the **action** (what computation to run on escalation). We characterize the limits
of the first and find leverage in the second, plus a separate training-based lever for accuracy.

**The gate is saturated.** A cascade needs *recoverability* — will the strong model fix this error? — not
just cheap-model wrongness. This is near-unpredictable from any frozen signal (AUROC ≤ 0.69) because the two
models' errors are correlated (φ = 0.372): on the competent benchmarks 58.4 % of the cheap model's errors the
strong model also misses. No decision rule beats a plain confidence threshold.

**The leverage is structural (ACC).** Instead of a better gate, we change *what escalation runs*. The 32B's
fast no-think mode (≈ 0.34 s) is as accurate as or better than its slow think mode on perception VQA (thinking
over-thinks perception: 32B-no-think beats 32B-think by +0.085 on SLAKE and +0.077 on VQA-RAD). Inserting
32B-no-think as an intermediate tier — gated by the agreement of the two no-think legs — lets the slow think
pass fire on only the ≈ 15 % reasoning residual.

**Training breaks the selection floor.** Routing *between* models is capacity-bound, but "given N samples
from *one* model, which is correct?" admits a *trained* answer: a small LoRA verifier used for best-of-N
selection recovers 35–78 % of the oracle gap for answers and boxes, where every training-free selector is
luck-floored — and, across three families, it beats the strong model on accuracy.

**Open-ended evaluation matters.** The pessimistic routing AUROC (≈ 0.6) is partly a multiple-choice artifact:
a single A/B/C/D letter is maximally discrete. On open-ended free-text the same confidence signal reaches
AUROC ≈ 0.87, so we evaluate selection/verification open-ended.

**Contributions.** (i) **ACC**, a compute-configuration cascade with large measured latency/energy/FLOPs
savings at parity and a per-benchmark safety guarantee. (ii) A **luck-floor characterization**: a dozen
training-free escalation/selection signals capped at the recoverability ceiling, explained by error
correlation, extended to actions, cross-family peers, the language prior, and structured outputs. (iii) The
**open-ended ceiling-break**: the routing signal is a discreteness artifact (MCQ ≈ 0.6 → open ≈ 0.87). (iv) A
**trained outcome verifier** that breaks the selection floor and **beats the 32B/38B across three families +
held-out OOD**, for free-text answers and bounding boxes (incl. real MS-CXR), bootstrap-significant, a
test-time-scaling method that transfers cross-architecture. (v) A **faithful cross-family MCQ cascade + think
tier**, reproducing published numbers (Lingshu-32B MMMU 0.633 = paper 62.3) and matching the 32B at −17 …
−69 % FLOPs. (vi) Honest framing: the agreement *gate* is shared with prior cascading; our novelty is the
compute-configuration *structure*, the verifier's *application/unification*, and the characterization.

## 2. Related Work

**Efficient cascades / routing.** FrugalGPT-style cascades escalate by a post-generation confidence/benefit
signal; agreement-based cascading (ABC) escalates on ensemble disagreement; CAR uses confidence-aware routing.
Our agreement gate is, by design, in this family — we do not claim it novel; the contribution is the
compute-configuration tiering it controls. **Confidence / selective prediction / conformal.** MSP/Chow,
entropy, Gini/DOCTOR, split-conformal (CP-Router/LAC), and learned deferral (L2D) are the training-free gate
family we benchmark; all cluster at the recoverability ceiling, consistent with Jitkrittum et al.
(NeurIPS 2023), who prove confidence-only deferral is fundamentally limited. **Verifiers for best-of-N.**
Generative verifiers (GenRM) cast reward modeling as next-token prediction in text; vision-language process
reward models rerank reasoning steps; medical generator–verifier pipelines exist for *data synthesis*. We
train an *outcome* verifier for *inference-time* best-of-N in medical VQA and unify it across free-text
answers and bounding-box grounding — a combination absent from prior work — as a constructive counter to
*Verification Mirage* [2605.10850], which concluded self-verification fails in medical VQA and pointed to
retrieval, not a trained verifier. Verifier-score-as-gate precedents (CCPS [2505.21772], Self-REF
[2410.13284], Kiyani [2602.17633]) do not train an outcome verifier for a medical-VQA cascade. The
multiple-choice-vs-open-ended discreteness claim is ours.

## 3. Setup, Method, and Definitions

**Models.** Three families, each a cheap 7B/8B and a strong 32B/38B: **Lingshu-7B/32B**,
**MedVLThinker-7B/32B**, **InternVL3-8B/38B**. Verifier base: Lingshu-7B; box-verifier base: Qwen2.5-VL-7B.

**Data.** Seven MCQ benchmarks: PMC-VQA, SLAKE, VQA-RAD, PathVQA, MMMU-medical, MedXpertQA-MM, OmniMedVQA.
ALL-6 = all but OmniMed; ALL-5 = ALL-6 − MedXpert (near-chance); COMPETENT-4 = SLAKE/VQA-RAD/PathVQA/PMC.
Open-ended sets (LLM-judge graded): SLAKE/VQA-RAD/PathVQA/Kvasir-VQA-x1, + RadImageNet-VQA (transfer/OOD).
Grounding: SLAKE organ boxes and the real **MS-CXR** chest-X-ray benchmark (PhysioNet). The MCQ efficiency
results use the **faithful MedEvalKit protocol** (the model authors' harness; isolated vLLM 0.9.0.1 + the
correct wrappers), under which Lingshu-32B MMMU = **0.633 = paper 62.3** (exact); our internal harness is
*not* faithful. The LLM judge (MedVLThinker-32B, cross-checked by a Claude-Sonnet-5 judge) is validated:
100 % exact-match anchor and independent-judge κ = 0.85–0.96.

**Cost model.** One forward = F = 2·N·(P+G) FLOPs (N₇ = 7.6e9, N₃₂ = 33.0e9). A cascade with per-tier cost c
and escalation probabilities e₀, e₁ costs C = c_T0 + e₀·c_T1 + e₁·c_T2 (Eq. 1); FLOPs% = Σcascade / Σalways-
32B-think; latency/energy use Eq. 1 with measured batch-1 per-tier costs (energy NVML-integrated). Best-of-K
FLOP cost in 7B-forward-equivalents: K = 1 → 1×, 2 → 4×, 4 → 8×, 8 → 16×; always-32B = 4.57×.

**Signals, verifier, metrics.** Confidence signals from candidate-letter logprobs: margin m = ℓ₁ − ℓ₂; MSP;
entropy; Gini. A gate escalates iff m < τ. The **verifier** s_φ(v,q,a) = P_φ(Yes | v,q,a) = softmax(z)_Yes
(Eq. 4) is a LoRA-fine-tuned VLM trained by binary cross-entropy on per-sample correctness labels y (Eq. 5;
LLM-judge for answers, y = 1[IoU ≥ 0.3] for boxes); **best-of-N** returns â = argmax_i s_φ. Accuracy =
exact-match (MCQ) or LLM-judge (open); AUROC; oracle@N; gap-captured = (acc(â) − greedy)/(oracle@N − greedy)
(Eq. 2); guard = # benchmarks worse than always-7B (0 = never worse).

**Method I — ACC.** A three-tier cascade over compute configurations of the *same* two models:
T0 = 7B-no-think @cap320 (≈ 0.21 s) → T1 = 32B-no-think @cap320 (≈ 0.34 s) → T2 = 32B-think @fullres
(≈ 11.3 s). T0 outputs iff margin m₀ ≥ τ₀. The think tier fires only when the two no-think legs *disagree*,
with an ε-margin tiebreak so it triggers on the lowest-margin disagreements (≈ 15 %, not the full ≈ 32 %
disagreement rate). The only fitted parameters are the scalars τ₀, τ₁, set on held-out PMC-VQA-train.

**Method II — the verifier.** Best-of-N selection with Eq. 4. It is *not* circular: the judge is an automated
grader of match-to-*gold* (labels from the answer key, not the 32B's knowledge); the verifier trains on the
7B's *own* samples and only *discriminates* correct answers (strictly easier than generating them), which is
why a 7B suffices and imports no 32B capability. AUROC 0.924 and a −0.047 blank-image ablation confirm
genuine visual discrimination.

## 4. The Luck Floor

Over a frozen model, no training-free decision rule beats trivial baselines — for the gate, the action, *or*
selection.

**(a) The gate is saturated.** Across 12 signal families (margin, MSP/Chow, entropy, Gini/DOCTOR, energy,
hidden-state probe, self-verification P(True), conformal/CP-Router, learned GBM router, cross-model
agreement, multi-resolution stability, semantic self-consistency), recoverability AUROC sits at **0.5–0.69**
(hidden-state probe 0.60 vs confidence 0.68). The cause is error correlation: ALL-6 recoverability
P(32B right | 7B wrong) = 37.2 % (φ = 0.372); COMPETENT-4 P(32B wrong | 7B wrong) = 0.584 — the models fail
together. **(b) Cross-family complementarity is real but unexploitable:** oracle union(7B | InternVL) = 0.753,
+Phi = 0.801 vs always-32B 0.645, yet the best learned router (0.621) ≈ always-7B (0.622) and a SigLIP
image+text router is at chance. **(c) A language prior:** over half the 7B's answers are unchanged when the
image is blanked, and image-sensitive vs insensitive questions have near-equal accuracy (0.620 vs 0.625).
**(d) The action axis is capacity-bound:** cheap same-model repairs recover 14.3 % of errors the 32B *also*
misses (a novel observation) but are unharvestable — a confidence-gated repair ladder *loses* at parity
(43 % vs 39 % compute). **(e) Selection is luck-floored (the headline negative):** sampling one model N = 8
times leaves a large oracle gap (SLAKE-open greedy 0.730 → oracle@8 0.879), but the model cannot say *which*
sample is right. Every training-free selector ties or trails random-pick (0.720): self-verify P(Yes) 0.715,
32B pointwise 0.746, 32B listwise 0.758 (best, 24 % of gap), fusion 0.743 — none beats the 32B single pass
(0.819); synthesis *backfires* (0.774). A **majority trap** explains it: the correct answer is a minority vote
in 74–90 % of recoverable cases. The same holds for **bounding boxes** (SC-medoid at the luck floor).

## 5. The Open-Ended Ceiling-Break

The pessimistic gate AUROC (≈ 0.6) is inflated-pessimistic by multiple-choice. Re-running the *same*
confidence signal on open-ended free-text, detection of cheap-model errors jumps to **AUROC ≈ 0.87**
(Lingshu-7B→32B confidence 0.866; pooled 0.846, 95 % CI [0.830, 0.862]) — a *discreteness*, not answer-length,
effect (token-F1 grading agrees). The gate itself, however, *remains* near-optimal at plain confidence even
here (confidence 0.866 vs exact-SC 0.845, semantic-SC 0.806, P(True) 0.755) — so even in the favorable regime,
*training-free* selection is still floored, motivating the trained verifier.

## 6. Results

### 6.1 ACC: large efficiency gains at parity, and it is the structure not the gate

At parity accuracy (always-32B-think) on ALL-6 (MedVLThinker, calibration held out):

| system | acc | esc₀ | think | FLOPs% | latency | energy | guard |
|---|---|---|---|---|---|---|---|
| always-7B-nt @cap320 | 0.5262 | 0 % | 0 % | 8.4 % | 0.13 s | 20 J | — |
| always-32B-nt @cap320 | 0.5573 | 100 % | 0 % | 36.2 % | 0.23 s | 78 J | 0.0 |
| **always-32B-think [PARITY]** | **0.5723** | 100 % | 100 % | 100 % | **11.34 s** | **6319 J** | 0.0 |
| **Ours: ACC-v2 (agreement)** | **0.5693** | 71.7 % | 15.1 % | **52.0 %** | **2.27 s** | **1182 J** | **0.0** |
| ACC-v1 (margin) | 0.5687 | 66 % | 19.5 % | 53.9 % | 2.69 s | 1417 J | 0.0 |

**ALL-6: 11.34 s → 2.27 s (−80 % latency), FLOPs 100 → 52 %, energy ≈ 5× (6.3 kJ → 1.2 kJ), at −0.003 accuracy
and guard 0** (never worse than the 7B on any benchmark). On **ALL-5** (excluding near-chance MedXpert):
8.88 s → **0.44 s**, FLOPs to **24.9 %**, energy to 173 J. **It is the structure, not the gate:** holding the
3-tier configuration fixed and swapping in every training-free gate (MSP/Chow, entropy, Gini/DOCTOR, AutoMix
self-verify, FrugalGPT-learned, Jitkrittum L2D) lands them all on the same accuracy–FLOPs frontier (FLOPs
49–62 %, latency 1.8–8.0 s); agreement is merely the cheapest point. The premise and savings reproduce across
five families/three architectures (e.g. Lingshu ALL-5 FLOPs 77.8 % → 38.9 %); where thinking *hurts*
(QoQ-Med, Chiron), ACC gracefully collapses to the cheap 7B (guard 0).

### 6.2 The verifier beats the 32B across three families

Trained outcome-verifier best-of-8 + verifier-confidence gate, full open-ended sets (InternVL3 uses the
*Lingshu-trained* verifier — cross-architecture transfer). "cascade-best" = best point on the
accuracy-vs-escalation frontier.

| family | dataset | cheap(SC) | STRONG | verifier-bo8 | **cascade-best @esc** | oracle@8 |
|---|---|---|---|---|---|---|
| **Lingshu** | POOLED | 0.322 | 0.331 | 0.414 | **0.421 @12 %** | 0.513 |
| (32B) | VQA-RAD | 0.465 | 0.600 | 0.575 | **0.625 @24 %** | 0.630 |
| | PathVQA | 0.324 | 0.376 | 0.453 | **0.469 @33 %** | 0.517 |
| | RadImageNet-OOD | 0.329 | 0.289 | 0.353 | **0.353 @0 %** | 0.512 |
| **MedVLThinker** | POOLED | 0.266 | 0.277 | 0.339 | **0.344 @14 %** | 0.416 |
| (32B) | VQA-RAD | 0.420 | 0.525 | 0.490 | **0.555 @54 %** | 0.600 |
| | RadImageNet-OOD | 0.204 | 0.202 | 0.241 | **0.243 @13 %** | 0.317 |
| **InternVL3** | POOLED | 0.202 | 0.218 | 0.249 | **0.255 @36 %** | 0.337 |
| (38B, x-arch) | VQA-RAD | 0.445 | 0.415 | 0.570 | **0.580 @12 %** | 0.620 |
| | RadImageNet-OOD | 0.285 | 0.304 | 0.302 | **0.313 @52 %** | 0.398 |

The cascade beats the strong model on **every family × dataset cell**, including held-out OOD, and the
Lingshu verifier **transfers cross-architecture** to lift a 38B InternVL3 it never trained on. The gate does
real work where the 32B is genuinely better (pure best-of-8 loses on the VQA-RAD cells; the gate escalates
more there and climbs above the 32B by *knowing when to defer*).

**A strict same-split test, honestly.** Pooling the four Lingshu sets and holding out 30 % (n = 1064):
verifier **0.501 > 32B 0.462 > SC 0.411 ≈ greedy 0.413** — the trained 7B captures **49 %** of the oracle gap
and, pooled, edges past the 32B while 5× smaller, beating it on the hard sets (PathVQA 0.441 vs 0.377, Kvasir
0.405 vs 0.326) and losing where the 32B is stronger (SLAKE 0.762 vs 0.829) or n is tiny (VQA-RAD). Across two
seeds the win vs the 32B is **+0.039 (95 % CI [+0.010, +0.066], significant)** on seed-0 and **−0.005 (a tie)**
on seed-1 — so versus the 32B the honest claim is **matches / modest-win (mean +0.017)**, while the margin
over *training-free* selection is robust (+0.088 / +0.066 over greedy; bootstrap vs first-sample +0.116
[+0.092, +0.140]), gap-captured **35–49 %**.

**Training is the active ingredient; candidate quality is the ceiling.** Zero-shot self-verification is
luck-floored; the trained 7B verifier beats a **zero-shot 32B** verifier (0.403 vs 0.355), argmax beats any
vote-flavored rule (0.501 > weighted 0.489 > hybrid 0.470), and best-of-K rises monotonically with the budget
(K = 1,2,4,8: 0.385 → 0.425 → 0.476 → 0.501; random flat ≈ 0.39) — a genuine test-time-scaling method. It
transfers zero-shot to a fifth dataset (RadImageNet 0.329 → 0.353) and across generators (Lingshu verifier →
MedVLThinker answers: SLAKE 49 %, VQA-RAD 61 %), though a from-scratch MedVLThinker verifier is weaker (SLAKE
42 %, pooled 25 %, and *fails* VQA-RAD at n = 54 — an honest negative). The binding limit is *candidate
quality*: selection efficiency is 74–82 % and a ranking loss lifts per-answer AUROC 0.90 → 0.93 without moving
selection, whereas a **cross-model candidate pool raises oracle@N by +0.11–0.15**.

**Structured outputs.** A LoRA box-verifier selecting best-of-8 boxes recovers **40–53 %** (SLAKE organs) and
**77–78 %** of the oracle gap on the real **MS-CXR** chest-X-ray benchmark (gain **+0.191, 95 % CI [+0.152,
+0.232]**, n = 435, a 5.6× lift over greedy) — both training-free selectors sit at the luck floor.

**The gate cannot be improved.** Verifier-confidence is the best cascade gate (pick-correctness AUROC 0.853 /
0.885 / 0.875 across families); the best *trained* gate (GBM/MLP on verifier±cheap features) adds ≤ +0.008
(noise) and the SOTA post-hoc recoverability gate (Jitkrittum Diff-Prob, AUROC 0.708) is worse — because the
recoverability wall persists (the strong model fixes only 6–10 % of the verifier's errors, 26 % where it is
competent, and *which* is near-unlearnable).

### 6.3 The faithful cross-family MCQ cascade and a reasoning think tier

Under the faithful MedEvalKit protocol, the 2-tier margin cascade matches always-32B accuracy at large FLOPs
savings wherever the small model is competitive (FLOPs saved vs always-32B; keep-cheap / no-win noted):

| benchmark | Lingshu | MedVLThinker | InternVL3 |
|---|---|---|---|
| MMMU-Med | keep-cheap (7B anomaly) | −14 % | **−62 %** |
| PMC-VQA (33k) | **−69 %** | −49 % | −16 % |
| SLAKE | −56 % | no win (7B weak) | keep-cheap (8B ≥ 38B) |
| VQA-RAD | −17 % | −41 % | **−67 %** |
| PathVQA | −31 % | **−68 %** | −20 % |
| MedXpert-MM | no win (floor) | no win | *(in progress)* |

Win magnitude ≈ the (32B − 7B) accuracy gap; a held-out-τ honesty check keeps PMC-VQA at −57 % (Lingshu) /
−49 % (MedVLThinker) with a fair threshold. A 3-tier **think tier** (adding 32B-think for reasoning) then
raises accuracy on the reasoning benchmark MMMU across **all three** families — **Lingshu +0.034 (0.633 →
0.667), MedVLThinker +0.107 (0.613 → 0.720), InternVL3 +0.120 (0.633 → 0.753)** — and MedVLThinker also gains
on MedXpert (+0.045); it is regime-adaptive (think where there is headroom, not at the floor). All three
families reason on a *generic* "reason step by step" prompt (gen_toks 3 → 275/561/368; MedVLThinker gains
+0.107 while emitting *zero* `<think>` tags), correcting an earlier "no promptable think mode" claim.

## 7. Discussion and Limitations

**Novelty, honestly.** ACC's agreement gate is the ABC family and the verifier's mechanism follows GenRM; the
contributions are the *compute-configuration structure*, the verifier's *application + unification* over
answers and boxes, and the *characterization* (the luck floor + the open-ended ceiling-break). **Scope.** The
over-thinking premise holds on *perception* (COMPETENT-4); MMMU is competent-but-reasoning (thinking helps, so
the ACC over-thinking mechanism does not apply, but the think tier does); MedXpert is near-chance and excluded
from headline efficiency. Versus the 32B the verifier is matches-to-modest-win (robust only vs training-free
selection); the per-dataset spread is wide (SLAKE 15 % → Kvasir 58 %); a from-scratch verifier is not
uniformly robust; the box-verifier lifts *selection over a frozen generator*, not a trained SOTA grounder.
Latency/energy are calibrated batch-1 (Eq. 1); FLOPs are exact. **In progress:** the 7th benchmark
(OmniMedVQA, ≈ 89k) is still running, InternVL3-MedXpert awaits a context-length fix, and some July latency
was measured under GPU contention and is re-measured serially (accuracy/FLOPs unaffected). Candidate quality
is the ceiling the verifier now points to — the highest-value next lever.

## 8. Conclusion

For efficient, accurate medical VLMs, the routing/selection decision cannot be out-engineered with
training-free signals — a dozen hit the same recoverability ceiling, selection sits at a luck floor, and the
two frozen models fail together. Two levers give large, real gains: **structure** — ACC cuts latency −80 %,
FLOPs to ~½, energy ~5× at parity, and a faithful cross-family MCQ cascade matches the 32B at −17 … −69 %
FLOPs with a reasoning think tier — and **a little training** — an outcome verifier that breaks the selection
luck floor and **beats the 32B/38B across three families + OOD**, for answers and boxes (incl. real MS-CXR),
transferring cross-architecture. We also show the routing ceiling is partly an MCQ artifact (≈ 0.6 → ≈ 0.87
open-ended), so medical-VLM cascades should be evaluated open-ended. We release ACC, the trained verifier, and
the full negative-result characterization. *All numbers trace to a checkpoint; none are fabricated.*
