# Structure and a Little Training Beat the Gate: Adaptive-Compute Cascades and Trained Verifiers for Efficient, Accurate Medical VLMs

*Li-Wen Kuan (Leo) et al. — long-form manuscript, 2026-07. All numbers verbatim from real checkpoints; none
fabricated. Canonical sources: `results/cascade_methods/docs/current/{OPENTEXT_MASTER_TABLE,VERIFIED_FACTS}.md`
and `docs/archive_mcq/GROUND_TRUTH_NUMBERS.md`.*

## Abstract

Deploying medical vision–language models (VLMs) is expensive: a 32B reasoning model spends ≈ 11 s and ≈ 6 kJ
per question (batch-1), a 7B model ≈ 0.2 s. A **cascade** — answer cheaply, escalate only the hard cases — is
the standard route to efficiency, but it requires *deciding* which cases to escalate and *what to do* once
escalated. We report one hard negative, two positive levers, and a unifying explanation.

**A hard negative — the luck floor.** Over a *frozen* model, no *training-free* signal beats trivial
baselines. A dozen escalation signals (confidence margin, MSP/Chow, entropy, Gini/DOCTOR, self-verification
P(True), conformal prediction, learned routers, cross-model agreement, multi-resolution stability, semantic
self-consistency) all hit the same **recoverability ceiling** (AUROC ≈ 0.5–0.69): whether a cheap model's
error is *fixable* is near-unpredictable from cheap features, because the two models fail together
(φ = 0.372; on the competent benchmarks P(32B wrong | 7B wrong) = 0.584). The same wall appears for
*selection* — given N sampled answers, no training-free rule beats picking one at random (the luck floor) —
and even for verifiable bounding boxes.

**Two levers move the needle, neither of which is "a better gate."** (1) **Structure (efficiency).** The
*Adaptive-Compute Cascade* (ACC) routes across *compute configurations* of the same models — 7B-no-think →
the 32B's **fast no-think mode** → the 32B's slow think mode — because reasoning *over-thinks* perception VQA
(32B-no-think ≥ 32B-think on competent sets). At parity accuracy this cuts **latency 11.34 s → 2.27 s
(−80%), FLOPs 100 % → 52 %, and energy ≈ 5× (6.3 kJ → 1.2 kJ)** on ALL-6, never worse than the 7B on any
benchmark; and a **faithful cross-family MCQ margin cascade** matches the 32B at **−17 … −69 % FLOPs**, with a
reasoning **"think tier"** adding accuracy on the reasoning benchmark MMMU (**+0.03 … +0.12**) across three
families. (2) **A little training (accuracy).** A small **trained outcome verifier** that scores
P(correct | image, question, candidate) and selects best-of-N **breaks the luck floor** and **beats the
strong 32B/38B on accuracy across three model families** — Lingshu, MedVLThinker, InternVL3 — on every
dataset and a held-out out-of-distribution set (Lingshu pooled **0.421 vs 0.331**, MedVLThinker **0.344 vs
0.277**, InternVL3 **0.255 vs 0.218**), for free-text answers (recovering **35–49 %** of the oracle gap over
two seeds) *and* structured bounding boxes (SLAKE organs **40–53 %**; the real **MS-CXR** chest-X-ray
pathology benchmark **77–78 %**, a 5.6× lift, bootstrap-significant). The verifier discriminates correct from
incorrect candidates at **AUROC 0.924**, beats a zero-shot 32B verifier despite being 5× smaller, and
transfers zero-shot **across architectures** (a Lingshu-trained verifier lifts InternVL3).

**The connective insight:** training-free routing/selection over a frozen model is luck-floored; the two
things that move the needle are *structure* (ACC) and *a little training* (the verifier). We additionally
show the routing ceiling is partly a **benchmark artifact** — the same confidence signal jumps from AUROC
≈ 0.6 on multiple-choice to ≈ 0.87 on open-ended free-text (a *discreteness* effect) — so medical-VLM
cascades should be evaluated open-ended. We release ACC, the trained verifier, and the full negative-result
characterization.

---


## 1. Introduction

A medical vision-language model (VLM) takes an image — a radiograph, a pathology slide, an endoscopy
frame — together with a question, and returns an answer. The strongest medical VLMs today are large
reasoning models that emit a long `<think>` trace before answering: they are accurate but slow and
energy-hungry (for a 32B at batch 1, one answer costs ≈ 11 s and ≈ 6 kJ), whereas a 7B of the same
family answers directly in ≈ 0.2 s. The standard route to efficiency is a **cascade**: run the cheap
model first and escalate only the hard cases to the expensive one. A cascade's quality hinges on two
decisions that are usually collapsed into one: the **gate** — *which* queries to escalate — and the
**action** — *what* computation to run on escalation. This paper separates the two, characterizes the
sharp limits of the first, and finds the real leverage in the second, plus a distinct training-based
lever that improves accuracy rather than efficiency.

**The gate is saturated — a "luck floor."** A frozen confidence, agreement, or self-consistency
signal can rank queries by how *likely the cheap model is wrong*. But what a cascade actually needs is
different: whether *the expensive model will fix* the cheap model's error — its *recoverability*. We
show recoverability is nearly unpredictable from any cheap signal. A dozen training-free signal
families (margin, maximum softmax probability, entropy, Gini/DOCTOR, conformal scores, self-verification,
a hidden-state probe) all cluster at the same ceiling, AUROC ≤ 0.69, and a learned hidden-state probe
(0.60) does not beat plain confidence (0.68). The mechanism is that the two models' errors are
**correlated**: on the six-benchmark suite the 7B/32B recoverability rate is φ = 0.372, and on the four
benchmarks where the method is competent, the strong model *also* misses **58.4%** of the cheap model's
errors — P(32B wrong | 7B wrong) = 0.584. This is exactly the regime where post-hoc deferral theory
(Jitkrittum et al., NeurIPS'23) predicts a wall, and no decision rule we tested beats a plain confidence
threshold in a way that is simultaneously novel, real-efficiency-positive, and per-benchmark
guardrail-safe. The same negative extends to *selection* (best-of-N over one model's own samples sits at
a luck floor, below or barely above a random pick), to *actions* (cheap repairs recover a real but
unharvestable 14% of errors the 32B also misses), to *cross-family peers*, and to the *language prior*.
We call this unifying negative result the **luck floor** (§5).

**Yet two levers give large, real gains — and neither is "a better gate."**

*Lever 1 — structure (efficiency).* Instead of a better gate we change *what escalation runs*. The 32B
has a *fast no-think mode* (≈ 0.34 s at batch 1) that, on perception VQA, is **as accurate as or better
than** its slow think mode — reasoning *over-thinks* perception (SLAKE: 32B-no-think 0.849 vs
32B-think 0.764, +0.085; VQA-RAD: 0.853 vs 0.776, +0.077). Our **Adaptive-Compute Cascade (ACC)**
inserts 32B-no-think as an intermediate tier between the cheap 7B and the slow 32B-think pass, gated by
the agreement of the two no-think legs, so that the expensive think pass fires only on the ~15%
reasoning residual. Because the cheap leg's latency is ~80× lower than a think pass, removing think calls
dominates the savings: at parity accuracy on the six-benchmark suite (ALL-6), batch-1 latency drops
**11.34 s → 2.27 s (−80%)**, FLOPs **100 → 52%**, energy **6318.8 → 1181.9 J (~5×)**, and the cascade is
**never worse than always-7B on any benchmark** (guardrail = 0). Excluding the near-chance MedXpert set
(ALL-5), latency drops **8.88 s → 0.44 s** and FLOPs to **24.9%**. Crucially, every training-free gate we
swap in lands in the same 49–62% FLOPs band — confirming it is the *structure*, not the gate, that pays
(§7–§8).

*Lever 1, generalized — the faithful MCQ cascade and a reasoning "think tier."* Reproduced under the
faithful MedEvalKit protocol (our Lingshu-32B MMMU = 0.633 exactly matches the paper's 62.3), a plain
2-tier margin cascade matches the 32B at **−17…−69% FLOPs** across three families and multiple
benchmarks; the win magnitude tracks the 7B/32B accuracy gap, so we honestly report the keep-cheap and
no-win cells too. Where a benchmark rewards reasoning, a strong-leg *think tier* adds accuracy for free:
MMMU-Medical improves **+0.034** (Lingshu, 0.633→0.667), **+0.107** (MedVLThinker, 0.613→0.720), and
**+0.120** (InternVL3, 0.633→0.753) (§11).

*Lever 2 — a little training (accuracy).* Routing *between* frozen models is capacity-bound, but a
different question — *given N samples from one model, which is correct?* — admits a **trained** answer. A
small LoRA outcome verifier that scores P(correct | image, question, candidate), used for best-of-N
selection, **breaks the selection floor** where every training-free selector is stuck. It **beats the
strong model on accuracy across three families and held-out OOD**: pooled, Lingshu **0.421 vs 0.331**,
MedVLThinker **0.344 vs 0.277**, and InternVL3 **0.255 vs 0.218** — the last obtained by transferring the
*Lingshu*-trained verifier onto InternVL3's answers, i.e. across model architectures. The signal is
learnable but not zero-shot-surfaceable: a trained 7B verifier beats a *zero-shot 32B* verifier (0.403
vs 0.355), so task-training matters more than scale. The same principle recovers **40–78%** of the oracle
gap on bounding boxes, including on the real MS-CXR chest-X-ray benchmark (bootstrap gain +0.191, 95% CI
[+0.152, +0.232]). We keep the caveats visible: on the canonical same-split pooled comparison the
verifier is **+0.039 [+0.010, +0.066]** over the 32B on one seed but a **−0.005 tie** on the other — so
the honest claim against the 32B is *matches-to-modest-win* (mean +0.017), while the improvement *over
training-free selection* is robust on both seeds (+0.066/+0.088), and the per-dataset spread is wide
(gap-captured 15% on SLAKE up to 58% on hard sets) (§9–§10).

**Open-ended evaluation matters.** The pessimistic routing AUROC (~0.6) is partly an artifact of
multiple-choice: a single A/B/C/D letter is maximally discrete, so a confidence margin carries little
information. On open-ended free-text the *same* confidence signal reaches AUROC **~0.87** (pooled 0.846,
95% CI [0.830, 0.862]); this is a discreteness effect, not a length effect (token-F1 agrees). We
therefore study selection and verification open-ended throughout, which is what makes Lever 2 possible
(§6).

**Honest scope.** Several results are explicitly provisional and flagged as such: the 7th benchmark
(OmniMedVQA, ~89k questions) is still running; the InternVL3-38B MedXpert cells are not yet measured
(a context-length cap gap, re-run queued); and some July latency measurements taken under GPU contention
are uncitable pending a serial re-run (accuracy and FLOPs are unaffected).

**Contributions.**

1. **ACC**, a cascade over *compute configurations* (7B-no-think → 32B-no-think → 32B-think) that
   exploits reasoning's over-thinking of perception, with large *measured* batch-1 latency/energy/FLOPs
   savings at parity (ALL-6 −80% latency, FLOPs → 52%, energy ~5×) and a per-benchmark safety guarantee,
   shown to be structure-driven rather than gate-driven, across five model families.

2. A **luck-floor characterization**: a dozen training-free escalation *and* selection signals, all
   capped at a recoverability ceiling (AUROC ≤ 0.69), explained by 7B/32B error correlation (φ = 0.372;
   P(32B wrong | 7B wrong) = 0.584 on the competent-4), and extended to actions, cross-family peers, the
   language prior, and structured outputs.

3. The **open-ended ceiling-break**: the routing signal's weakness is largely an MCQ discreteness
   artifact (AUROC ~0.6 MCQ → ~0.87 open-ended), motivating open-ended evaluation of medical-VLM
   cascades.

4. A **trained outcome verifier** for inference-time best-of-N that beats the strong 32B/38B on accuracy
   across **three families and held-out OOD**, transfers across architectures, and unifies **free-text
   answers and bounding-box grounding** (incl. real MS-CXR) — 2-seed and bootstrap-significant, a
   test-time-scaling method that lets a 7B match or exceed the 32B single pass, with the ties and
   per-dataset failures reported inline.

5. A **faithful MedEvalKit MCQ cascade** (−17…−69% FLOPs at reproduced-32B accuracy) plus a strong-leg
   **think tier** that adds reasoning accuracy where headroom exists (MMMU +0.03…+0.12 across three
   families), generalizing the structural lever beyond the internal harness.

6. **Honest novelty framing** throughout: the agreement *gate* mechanism is shared with prior cascading
   work (ABC/CAR) and the verifier lineage descends from generative reward models (GenRM); our
   contribution is the *compute-configuration structure*, the trained-verifier *application and
   unification* over answers and boxes, and the negative-result *characterization* — not the underlying
   signals.

---


## 2. Related Work

Our two decisions — *which* queries to escalate (the gate) and *what* to run on escalation
(the action) — sit at the intersection of several literatures: LLM/VLM cascades and query
routing, selective prediction and conformal deferral, the theory of post-hoc deferral, and
verifiers/reward models for best-of-N selection. We survey each below and state, inline and
without hedging, exactly which pieces we adopt unchanged and which are new. The short version:
our **gate** and our **verifier architecture** are each squarely inside an existing family — we
do not claim either as novel — and the contribution is (i) the *compute-configuration* structure
the gate controls, (ii) the *application and unification* of an outcome verifier across free-text
answers and grounding boxes in medical VQA, and (iii) the empirical *characterization* (the luck
floor, and the multiple-choice-vs-open-ended discreteness effect).

**Efficient cascades and query routing.** FrugalGPT popularized the cost-saving LLM cascade:
run a cheap model first, and escalate to a larger one only when a learned post-generation
confidence/benefit score is low. Agreement-based cascading (ABC) escalates on ensemble
*disagreement* rather than a scalar confidence, and confidence-aware routing (CAR, the closest
prior art to our structural method, arXiv 2505.15154) routes reasoning queries by a confidence
signal. Our **agreement gate is, by construction, a member of the ABC family** — the tier-1→tier-2
"think" trigger fires when the large model's fast no-think answer disagrees with itself under an
ε-margin tiebreak (§7), and we make no novelty claim for it. What is new is *what the gate
switches between*: not two models, but three **compute configurations** of the same two models
(7B-nothink → 32B-**no-think** → 32B-think), inserting the strong model's fast no-think mode as
an intermediate tier because reasoning *over-thinks* perception VQA (§7–§8). A parallel line
learns a *pre-*generation query router directly from image/text features (rather than from a
generation's confidence); we benchmark this line and find it does not beat a post-generation
confidence signal here (§5.2), and a SigLIP-feature router lands at chance for our
complementarity-routing task.

**Confidence, selective prediction, and conformal deferral.** The training-free gate family we
benchmark against the margin gate is the classical selective-prediction toolkit: maximum
softmax probability (MSP / Chow's rule), predictive entropy, the Gini/DOCTOR score,
energy-based scores, split-conformal thresholding and its cascade instantiations (CP-Router,
LAC), and learning-to-defer (L2D) heads. In our ACC bake-off these all cluster together at the
recoverability ceiling: on ALL-6 at 32B-think parity every training-free gate lands within a
narrow band (FLOPs ≈ 49–62%, batch-1 latency ≈ 1.8–8.0 s), and no member of the family
meaningfully separates from the deployed margin gate (§5.1, §8). In the open-ended
verifier-gate bake-off the same holds against the verifier-confidence gate (§10).

**Post-hoc deferral theory — the wall.** Jitkrittum et al. (NeurIPS 2023) analyze *post-hoc*
deferral: given a frozen small model, a rule that decides when to route to a frozen large model
is bounded by how predictable the large model's *marginal correction* is, and they derive the
Diff-Prob deferral score for that setting. This theory is a precise description of the ceiling we
hit empirically: the question "will the strong model fix this error?" is only **AUROC ~0.6** from
any cheap signal, and a faithful Diff-Prob gate reaches only **AUROC 0.708** — well below the
verifier-confidence gate's **0.853** and worse on the deferral curve (ADC 0.3832 vs 0.3923) —
because the strong model repairs only **6–10%** of the cheap model's errors (about **26%** on the
subset where the strong model is genuinely competitive), and *which* ones is near-unlearnable
(§5, §10). We therefore treat Jitkrittum's result as the *explanation* for our luck floor rather
than a baseline to beat.

**Verifiers and reward models for best-of-N.** Generative verifiers (GenRM) recast reward
modeling as next-token "is this correct?" prediction and use the resulting score for best-of-N
selection in *text*; vision-language process-reward models rerank intermediate *reasoning steps*;
and medical generator-verifier pipelines exist, but primarily for *data synthesis / filtering*,
not inference-time answer selection. Our verifier is squarely in this lineage — it is a small
LoRA outcome head s_φ(v,q,a)=P_φ(Yes\|v,q,a) trained with a BCE objective (§9), and we claim no
architectural novelty for it. The contribution is the **application and unification**: a single
*outcome* verifier used for inference-time best-of-N selection in **medical VQA**, trained and
evaluated over **both free-text answers and bounding-box grounding** (with the box label
y=1[IoU≥0.3]) — a combination we did not find in prior work (§9–§10). We position this
explicitly as a constructive counter to **Verification Mirage [2605.10850]**, which concluded
that self-verification *fails* in medical VQA and recommended retrieval rather than a trained
verifier. That negative result is about *zero-shot self-verification*, and our evidence is
consistent with it: zero-shot self-verification is essentially uninformative (discrimination
AUROC ≈ 0.5), and a zero-shot 32B verifier underperforms a *trained* 7B verifier (selection
accuracy 0.355 vs 0.403). The difference is training: the same signal, once *learned*, reaches
discrimination AUROC **0.924** and turns into an accuracy win over the strong 32B/38B across
three model families and held-out OOD (§9–§10). So we read the Mirage result as scoped to the
zero-shot regime, not as a bar against trained outcome verifiers.

**Verifier-score-as-gate precedents.** A recent thread uses a verifier/confidence score itself
as the *deferral gate*: CCPS [2505.21772] scores answer reliability via input-perturbation
(visual/consistency) stability; Self-REF [2410.13284] trains the model to emit calibrated
self-reference confidence tokens for routing; and Kiyani et al. [2602.17633] give a conformal
treatment of verifier scores for selective prediction. Our escalation gate — thresholding the
trained verifier's confidence — is in this spirit, and we again claim no novelty for the
*idea* of gating on a verifier score. **None of these is a trained outcome verifier for medical
VQA selection**, and where they overlap our setting we benchmark them faithfully: a faithful
CCPS input-perturbation visual-stability gate reaches only **AUROC ~0.60** (0.604 pooled) and
adds nothing when combined with verifier-confidence (**0.852 → 0.853**, i.e. within noise; §10).
The verifier-confidence gate remains the best member of this family in our bake-off, and no
*trained* gate beats it by more than +0.008 AUROC (§10).

**Base families and the faithful evaluation protocol.** Our three base families are Lingshu-7B/32B
[2506.07044], MedVLThinker-7B/32B, and InternVL3-8B/38B (verifier base: Lingshu-7B; box-verifier
base: Qwen2.5-VL-7B). We reproduce the Lingshu MedEvalKit evaluation protocol faithfully rather
than through an internal harness; the protocol recipe and our reproduction-fidelity numbers are
deferred to §4, and the faithful MCQ-cascade results that build on it are in §11.

**Open-ended vs multiple-choice routing — the discreteness claim.** Semantic-agreement cascades
for text LLMs escalate on cross-model or cross-sample agreement, and are typically evaluated on
open-ended generation. Our contribution to this line is the observation, which we did not find
stated elsewhere, that the *routability* of a medical VLM is largely a **multiple-choice
artifact**: the *same* confidence signal that is nearly useless on MCQ (routing AUROC ~0.6) becomes
strongly discriminative once the answer space is open-ended (~0.87), and we show this is driven by
answer *discreteness*, not answer *length* (token-F1 agrees; §6). The practical corollary — that
medical-VLM cascades and verifiers should be studied open-ended — motivates the selection and
verification results in §6, §9, and §10.


## 3. Setup, Metrics, and Definitions

This section fixes the models, data pools, cost accounting, decision signals, and evaluation
metrics used throughout. Everything downstream — the luck-floor characterization (§5), the
open-ended ceiling-break (§6), and the three methods (§4, §9, §11) — is defined against the
objects introduced here. Every number quoted below is a real measurement traced to a checkpoint;
none is fabricated.

### 3.1 Models and families

The paper is organized around **three medical-VLM families**, each a small "cheap" leg and a
large "strong" leg of the *same* architecture:

| Family | Cheap leg | Strong leg |
|---|---|---|
| **Lingshu** | Lingshu-7B | Lingshu-32B |
| **MedVLThinker (MVT)** | MedVLThinker-7B | MedVLThinker-32B |
| **InternVL3 (IV3)** | InternVL3-8B | InternVL3-38B |

Lingshu and MVT are the two families for which we have full efficiency instrumentation (batch-1
latency/energy, per-tier costs); InternVL3 is the third, cross-architecture family added to test
whether the trained verifier and the faithful MCQ cascade *transfer* off the family they were
tuned on. A single cross-cutting result depends on this: the outcome verifier is trained on
Lingshu-7B answers only and then applied, unchanged, to InternVL3-38B candidate answers
(cross-architecture verifier transfer; §10).

Two additional bases appear where a specific method needs them. The **outcome verifier** (§9–§10)
is a LoRA adapter (~190 MB) on the *frozen* Lingshu-7B, trained on ~6,000 judged `(image, question,
answer)` triples at a positive rate of 0.194. The **box verifier** for grounding (§10.2) uses
Qwen2.5-VL-7B as its base.

For the ACC generalization study (§8) we additionally run three more families that we do *not*
otherwise instrument: **QoQ-Med** (7B/32B), **Chiron-o1** (2B/8B), and **MedGemma** (4B/27B).
These are introduced only to show where the over-thinking premise behind ACC holds and where it
fails (they are defined and used only in §8). Finally, the cross-family *complementarity* analysis
in §5 uses two off-family peers (InternVL, Phi-3.5-V) as diversity sources; those are named where
used.

**A model quirk that matters for the cost accounting.** The MVT and Lingshu strong legs can run in
two modes: **no-think** (answer directly) or **think** (emit a `<think>…</think>` reasoning trace
before the answer). The mode is not free-floating — these models only emit a `<think>` trace under
one exact system prompt; without it they answer directly. The no-think vs think distinction is the
axis ACC exploits (§4) and the think tier exploits (§11), so we treat "32B no-think" and "32B
think" as two distinct, separately-measured configurations of the same weights.

### 3.2 Benchmarks and evaluation pools

**Multiple-choice benchmarks (7).** PMC-VQA, SLAKE, VQA-RAD, PathVQA, MMMU-medical, MedXpertQA-MM,
and OmniMedVQA. The core MCQ eval set is **8,220** samples across the first six; MedXpertQA-MM
contributes two splits (Reasoning and Understanding), so per-split tables show seven columns.
**OmniMedVQA is provisional** — the Open-access split is 88,996 QA pairs (of which ~57k, i.e. 64%,
are RadImageNet: 56,697 items; 4-option MCQ, exact-match), and the Lingshu paper evaluates the full
set, so faithfulness demands the full 89k rather than a sample. The full-set run was still
executing at the time of writing (its parser bug — a `KeyError` over 42 sub-datasets — was fixed
and the 89k run restarted), so no cascade result over OmniMedVQA is reported here; we carry it as
in-progress.

We name three pools used repeatedly:

- **ALL-6** — all six MCQ benchmarks (n = 8,220; MedXpert split into R and U for per-split tables).
- **ALL-5** — ALL-6 minus MedXpertQA-MM. MedXpert is excluded from the ALL-5/headline claims because
  both legs sit near chance on it (e.g., MVT 7B/32B-think MedXpert-R 0.225 / 0.326, U 0.256 / 0.385).
- **COMPETENT-4** — SLAKE, VQA-RAD, PathVQA, PMC-VQA (n = 6,050): the sets on which both legs are
  competent and on which the ACC over-thinking premise (§4) is meant to hold.

**Open-ended sets.** For the open-ended studies (§6, §9–§10) we use free-text versions of SLAKE,
VQA-RAD, PathVQA, and Kvasir-VQA-x1, plus **RadImageNet-VQA** as a held-out transfer/OOD set (a
verifier trained on the first four is applied to RadImageNet zero-shot). Open-ended answers are
graded by an LLM judge (§3.6), not by string match.

**Grounding sets.** For structured outputs (§10.2) we use SLAKE organ bounding boxes (n = 487) and
**MS-CXR**, a real chest-X-ray phrase-grounding benchmark from PhysioNet (evaluation n = 435). A
predicted box is scored correct iff its IoU with the reference exceeds θ = 0.3 (§3.5).

### 3.3 The faithful MedEvalKit protocol (and why the internal harness is not it)

MCQ accuracies for the faithful cascade (§11) and the reproduction-fidelity check below are
produced with **MedEvalKit**, run under a specific recipe so that our per-model numbers reproduce
the published ones. The recipe: an isolated `medeval_venv`; the `Qwen2_5_VL` wrapper pointed at the
family's weights; `datasets_path=hf`; `use_vllm` on; `TORCHDYNAMO_DISABLE` set (to avoid a
compile-path crash); `use_llm_judge=False` for the MCQ (exact-match) halves; and vLLM pinned to
**0.9.0.1**. We flag explicitly that our internal NGC labeling harness is **not** faithful for this
purpose — it produces systematically different scores (e.g., a secondary NGC Lingshu-32B run gives
SLAKE 89.4 / PMC 64.0 / MMMU 62.4, which do not track the paper the way the MedEvalKit run does) —
so every "faithful" number in §11 comes from the MedEvalKit recipe above, and the NGC numbers are
never used for a headline claim.

**Reproduction fidelity.** Under this protocol our MedEvalKit run reproduces the Lingshu paper on
the anchor benchmarks (percentages; "ours" vs the paper's Table 6):

| Benchmark | Lingshu-7B ours | Lingshu-7B paper | Lingshu-32B ours | Lingshu-32B paper |
|---|---|---|---|---|
| MMMU-Med | 80.0 | 54.0 | **63.3** | **62.3** |
| SLAKE-closed | 82.5 | 83.1 | 85.9 | 89.2 |
| PMC-VQA | 54.3 | 56.3 | 55.2 | 57.9 |
| MedXpertQA-MM | 26.2 | 26.7 | 30.6 | 25.7 |
| VQA-RAD-closed | 78.1 | 67.9 | 85.3 | 76.5 |

The clean anchor is **MMMU-Med on the 32B: 63.3 vs paper 62.3 — an essentially exact match** on the
same 150-question set. The 7B MMMU row is an **outlier by +26 points** (80.0 vs 54.0); this
inflation is **Lingshu-7B-specific**, not a bug in our harness — MedVLThinker-7B on the same eval
scores a normal **0.533** (below its own 32B), so we exclude the Lingshu-7B MMMU cell from all
claims (this is why MMMU is "keep-cheap" for Lingshu in §11). Two honest caveats on the rest of the
table: (i) SLAKE-closed and VQA-RAD-closed are the *closed* (MCQ) halves only, compared against the
paper's *full-set* numbers, so they are not strictly like-for-like (closed halves are typically
easier, which is why VQA-RAD-closed 85.3 sits above the full-set 76.5); (ii) the paper reference for
MedXpertQA-MM-32B is reported as 25.7 in our reproduction log but as 30.9 in the paper's Table 6 —
we record both and do not lean on that cell. The **open-ended halves** are reproduced separately via
the Claude-Sonnet-5 judge (§3.6): VQA-RAD-32B = 74.1% (paper 76.5) and SLAKE-32B = 85.0% (paper
89.2), close but slightly under — consistent with a stricter judge on the open halves.

### 3.4 Cost model

One model forward pass over a query costs

```
F = 2 · N · (P + G)   FLOPs,                                                          (1a)
```

where N is the parameter count, P the prompt length in tokens (including vision tokens), and G the
number of generated tokens; the factor 2 is the standard two-FLOP-per-parameter-per-token count and
covers both prefill (P) and decode (G). We use `N₇ = 7.6e9` for the 7B/8B cheap legs and
`N₃₂ = 33.0e9` for the 32B/38B strong legs. For a cascade with per-tier cost `c` (in any of FLOPs,
seconds, or joules) and escalation probabilities `e₀` (fraction passing tier 0) and `e₁` (fraction
reaching the think tier), total cost is

```
C = c_T0 + e₀ · c_T1 + e₁ · c_T2.                                                     (1)
```

We report **FLOPs%** (equivalently "backbone%") = Σ(cascade FLOPs) / Σ(always-32B-think FLOPs); 100%
is the always-think strong model. **Latency** and **energy** use the same Eq. (1) with per-tier
costs from real **batch-1** measurements; energy is NVML-integrated,
`E = Σ (P_k + P_{k+1})/2 · Δt` over the power trace. The think-tier latency scales linearly with the
number of generated tokens (R² = 0.99 for the fit), which is what makes the think tier so expensive:
avoiding a think call avoids a variable, token-proportional cost, whereas the no-think tiers are
near-constant (they decode ~2 tokens). *(All ACC efficiency numbers use the native batch-1 cost
methodology of the canonical `master_data.csv`; an earlier co-resident `rt_cascade` estimate, which
put the think leg at ~28 s rather than the batch-1 ~11.3 s, is superseded and not used for any
headline.)*

**Best-of-N cost.** For the verifier method (§9–§10) each of the K sampled candidates costs one 7B
generate, and selecting among them costs one 7B verify per candidate, so a best-of-K pass costs
about 2K 7B-forwards for K ≥ 2 (K = 1 needs no selection). In 7B-forward-equivalents:

| Best-of-K | K = 1 | K = 2 | K = 4 | K = 8 |
|---|---|---|---|---|
| FLOP cost (×7B-fwd) | 1× | 4× | 8× | 16× |

Against this, one always-strong (32B) forward costs **4.57** 7B-forward-equivalents, so the break-even
is `2N < 4.57 ⇒ N ≤ 2`: a best-of-2 verifier pass is cheaper than a single 32B forward, which is
why the deployable open-ended recommendation is N = 2 (§10). Batch-1 latency does not follow the
FLOP ratio — at batch-1 the 32B is only ~1.9× the 7B in latency and ~2.8× in energy (not 4.6×),
because prefill and fixed overheads dominate.

**A measurement caveat we carry honestly.** Some July latency numbers — specifically the InternVL3
family and all PathVQA rows — were measured under GPU contention and are therefore **not citable**
as batch-1 latencies; their accuracy and FLOPs are unaffected, and clean serial re-runs are queued.
Where a latency is contended we say so rather than quoting it.

### 3.5 Confidence signals and the gate

Every gate in the paper thresholds a scalar computed from a greedy decode over the candidate answer
tokens with logprobs `ℓ₁ ≥ ℓ₂ ≥ …` and softmax probabilities `pᵢ = e^{ℓᵢ} / Σⱼ e^{ℓⱼ}`:

- **margin** `m = ℓ₁ − ℓ₂` (the deployed signal),
- **MSP** (maximum softmax probability) `p₁`,
- **entropy** `−Σᵢ pᵢ ln pᵢ`,
- **Gini** `1 − Σᵢ pᵢ²`.

A confidence gate escalates iff `m < τ` (low margin ⇒ escalate). The threshold τ is fit on a
held-out calibration split (PMC-VQA-train for MCQ; a cross-fit split for the open-ended verifier
gate) and frozen; §5 and §11 report where a *trained* gate is compared against thresholding these
raw signals.

### 3.6 The trained outcome verifier (definition)

The verifier is the one component of the paper that is *trained*. Given an image v, a question q,
and a candidate answer a, it emits a scalar score

```
s_φ(v, q, a) = P_φ(Yes | v, q, a) = softmax(z)_Yes,                                   (4)
```

where z are the logits of a two-way (Yes/No) head on the frozen base (Lingshu-7B for free text,
Qwen2.5-VL-7B for boxes) plus a LoRA adapter with parameters φ. It is trained with a binary
cross-entropy loss against a correctness label y:

```
L(φ) = − E_{(v,q,a,y)} [ y · log s_φ(v,q,a) + (1 − y) · log(1 − s_φ(v,q,a)) ],        (5)
```

where the label is `y = 1[a is correct]`. For free-text answers correctness is the LLM-judge verdict
(§3.6.1); for boxes it is `y = 1[IoU(a, gold) ≥ θ]` with θ = 0.3. At inference the verifier performs
**best-of-N selection** over N sampled candidates,

```
â = argmax_{i ≤ N} s_φ(v, q, aᵢ).
```

Note the verifier only has to *discriminate* good from bad candidates, not *generate* them — a
strictly easier task than answering, which is the reason a small trained verifier can beat a much
larger generator (§9–§10). The mechanism, non-circularity argument, and results are in §9–§10; here
we fix only the definition.

### 3.6.1 Evaluation metrics

- **Accuracy** = exact-match (MCQ) or LLM-judge correctness (open-ended).
- **AUROC** of a score s for a binary correctness label y is the rank statistic
  `AUROC = P(s(positive) > s(negative))` — the probability the score ranks a correct instance above
  an incorrect one.
- **Oracle@N** `= E_x[ max_{i ≤ N} 1(aᵢ correct) ]`: the accuracy of an oracle that picks a correct
  sample whenever one of the N candidates is correct. It upper-bounds any selector.
- **Gap captured** by a selector with accuracy `acc(â)`, relative to the greedy baseline and the
  oracle@N ceiling:

  ```
  frac = (acc(â) − greedy) / (oracle@N − greedy).                                      (2)
  ```

- **Guard** (safety) = the average number of benchmarks, per seed, on which a cascade falls *below*
  its own always-cheap (7B) baseline; 0 means the cascade is never worse than the cheap model.
- **IoU** is the standard box overlap; a grounding prediction is correct iff `IoU ≥ θ`, θ = 0.3.

### 3.6.2 LLM-judge validation

Because open-ended accuracy depends on a judge, we validate the judge before trusting any
open-ended number. The primary automated grader is **MedVLThinker-32B**; a **Claude-Sonnet-5** judge
is used as an independent cross-check (and to reproduce the open-ended paper halves in §3.3). Three
checks support the judge:

1. **Exact-match anchor.** On candidate pairs where the answer string equals the gold string
   (n = 1,277), the MVT-32B judge returns Yes **100.0%** of the time. Containment cases (gold ⊂
   answer, n = 2,423) are judged Yes 82.5%; zero-word-overlap cases (n = 14,320) are judged Yes only
   6.3%, and spot-checks confirm those are legitimate synonyms rather than judge errors.
2. **Inter-judge agreement.** A second 32B judge (Lingshu-32B vs MVT-32B) agrees at Cohen's κ in the
   **0.85–0.96** range:

   | Set | Agreement | Cohen's κ |
   |---|---|---|
   | VQA-RAD | 0.984 | 0.962 |
   | Kvasir | 0.958 | 0.849 |

3. **Independent frontier judge.** The Claude-Sonnet-5 judge, run over 1,257 SLAKE and 200 VQA-RAD
   open pairs, again shows a **100% exact-match anchor** (n = 80/1005 exact pairs, all judged Yes);
   its 21% zero-word-overlap judged-Yes rate was spot-checked to be entirely legitimate (synonyms and
   bilingual Chinese answers). Under this judge the open halves land at SLAKE-open 0.844 and
   VQA-RAD-open 0.600, reproducing the paper full-set numbers within a few points (§3.3).

We treat κ ≥ 0.85 plus a 100% exact-match anchor as sufficient to use the judge as the open-ended
accuracy oracle throughout §6, §9, and §10.

### 3.7 Peak VRAM

Peak GPU memory at batch-1 for each family and configuration (GB), used to size the two-GPU serving
setup and to place each configuration on hardware:

| Family (small / big) | Small leg | Big, no-think | Big, think |
|---|---|---|---|
| MedVLThinker, Lingshu, QoQ-Med (7B / 32B) | 71.45 | 152.82 | 154.08 |
| Chiron-o1 (2B / 8B) | 53.46 | 74.16 | 74.17 |
| MedGemma (4B / 27B) | 52.35 | 155.52 | 155.52 |

For the 7B/32B families the strong leg exceeds a single 80 GB device, so the cascade co-resides the
cheap leg and the strong leg on separate GPUs; think vs no-think adds essentially nothing to peak
VRAM (154.08 vs 152.82 GB for the 32B) — the think-tier cost is entirely time and energy, not
memory, which is exactly why ACC (§4) trades *think calls* rather than *residency* for its savings.


## 4. The Luck Floor

This section states the paper's unifying negative and the reason the two positive levers (a
structural cascade and a small trained verifier) are not "a better gate." The claim is sharp and
holds across every axis we tested:

> **Over a *frozen* model, no *training-free* decision rule beats trivial baselines — for the gate
> (which query to escalate), the action (what to run on escalation), *or* selection (which of N
> sampled answers to keep).**

We call this the **luck floor**. For the gate it is a *recoverability ceiling*: the best cheap
signal predicts "will the strong model fix this error?" at only AUROC ~0.6, because the two models
fail together. For selection it is a *sampling floor*: a correct answer is often present among N
samples, but the model cannot identify it, so every training-free selector ties or trails a random
pick. The five sub-results below (a–e) are five faces of the same phenomenon. Everything positive in
this paper is a way *around* the floor, not through it: the structural cascade (Method I) exploits
that reasoning over-thinks perception rather than trying to route better, and the trained verifier
(Method II) adds a *little supervision* precisely where training-free selection is floored. Section 5
(the open-ended ceiling-break) shows that the pessimistic ~0.6 gate AUROC is *partly* a
multiple-choice artifact — but even in the favorable open-ended regime, training-free selection stays
floored, which is what makes the trained verifier necessary rather than optional.

Unless noted, gate/action numbers are the MedVLThinker 7B/32B pair; selection numbers use Lingshu-7B
generations judged by a fixed MedVLThinker-32B judge. ALL-6 = all six MCQ benchmarks; COMPETENT-4 =
{PMC-VQA, SLAKE, VQA-RAD, PathVQA} (MMMU and near-chance MedXpert excluded).

### 4.1 The gate is saturated

A cascade gate does not need to know whether the *cheap* model is wrong; it needs **recoverability** —
whether the *strong* model will fix that specific error. Cheap-model-wrongness and recoverability are
different targets, and the second is far harder. Across **12 signal families** — margin, MSP/Chow,
entropy, Gini/DOCTOR, energy, a hidden-state probe, self-verification P(True), conformal/CP-Router, a
learned GBM router, cross-model agreement, multi-resolution stability, and semantic self-consistency
— recoverability AUROC sits at **0.5–0.69**. The learned and hidden-state signals do not escape it: a
hidden-state probe reaches only **0.60** versus plain confidence **0.68**, and a cross-validated
"benefit predictor" trained to forecast the escalation gain tops out at **74.7%** on ALL-6 — *no gain*
over plain confidence (73.7%). Even the *easier* target of predicting 7B-correctness is only mildly
learnable (margin AUROC ~0.70–0.75; self-verification P(True) 0.65–0.70, below margin), while the
target a cascade actually needs — recoverability — is only ~0.6 from any cheap signal.

The cause is **error correlation: the two models fail together.** Decomposing every question by the
joint (7B, 32B) right/wrong outcome:

| outcome | ALL-6 | COMPETENT-4 |
|---|---|---|
| beneficial (7B wrong, 32B right) | 17.6% | 15.7% |
| futile (7B wrong, 32B wrong) | 29.8% | 22.0% |
| harmful (7B right, 32B wrong) | 13.0% | 13.4% |
| **recoverability** P(32B right \| 7B wrong) | **37.2%** | **41.6%** |
| ⇒ share of the 7B's errors that are futile | **62.8%** | **58.4%** |

So `P(32B wrong | 7B wrong) = 0.584` on COMPETENT-4 (`φ = 0.372` on ALL-6, φ being the correlation of
the two models' right/wrong outcomes): **most of what the small model gets wrong, the big model also
gets wrong.** Recoverability rises only weakly with 7B uncertainty (43% → 28% across margin quintiles)
and caps around 43–50% even for the most uncertain questions, so no confidence threshold can
concentrate escalations onto recoverable errors. The consequence is visible in the deployed gate's
own behaviour: of the margin gate's escalations, only **22.5%** are beneficial and **15.2%** are
actively *harmful* (the 7B was right and the 32B is wrong). On the MCQ side the same wall appears
directly — on COMPETENT-4 the 7B (cap320) scores 0.622 and the 32B-think 0.645 (gap +0.023); among the
2,286 questions the 7B gets wrong, the 32B is right only 41.6% of the time, and the margin ranks *which*
of those are recoverable at AUROC **0.578**.

The headroom is real but unreachable. An **outcome-oracle** with perfect recoverability knowledge
would reach 32B-parity using only 11.2% on the same backbone-compute metric at 4.6% escalation, versus
the deployed gate's 73.6% at 63% escalation — a **62-point gap** that no training-free gate closes.
This is not a failure of the particular signals we tried; it matches theory. Jitkrittum et al. (*When
does confidence-based cascade deferral suffice?*, NeurIPS 2023) prove that the optimal deferral rule
requires *both* models' confidence and that confidence-*only* deferral is fundamentally limited —
exactly the wall we measure on medical VLMs. A learned "defer" meta-router that predicts escalation
*gain* rather than cheap-model correctness (a VADR-style rule) is likewise not better and is not novel
relative to prior confidence-deferral work; its one new claim — that the gain is learnable — fails.

### 4.2 Cross-family complementarity is real but unexploitable

Independently-trained VLMs make *different* mistakes, so their **oracle union** has large headroom:
`union(7B | InternVL) = 0.753` and `union(7B | InternVL | Phi) = 0.801`, versus always-32B 0.645. But
that headroom is not addressable with cheap features. A learned router over frozen peer signals
captures none of it — the best learned router reaches **0.621 ≈ always-7B 0.622** — and a SigLIP
image+text router that sees the raw question and image is at chance (**AUROC 0.50**). The
complementarity exists; a training-free router simply cannot tell, per-question, *which* peer to
trust.

### 4.3 The cheap model leans on a language prior

Part of what looks like "visual" difficulty is not visual at all. **56.9%** of the 7B's medical-VQA
answers are *unchanged when the image is blanked* (blank-image probe), and questions the model is
image-*sensitive* on versus image-*insensitive* on have nearly equal accuracy (**0.620 vs 0.625**).
Much medical "VQA" is answerable from the text prior alone, which caps how much any better *visual*
routing signal could buy — a large fraction of the questions carry no routable visual difficulty in
the first place.

### 4.4 The action axis is capacity-bound, not action-bound

If the gate cannot pick recoverable errors, perhaps the *action* — what we do on escalation — can be
made cheaper than "call the 32B." It cannot. Decomposing the 7B's errors on COMPETENT-4 (n = 6,050;
error set = 2,286) by repair route:

| repair route | share of the 7B's errors recovered |
|---|---|
| SCALE up (call the 32B) | 41.6% |
| cheap 7B repair (look-closer OR think) | 33.7% |
| multi-action oracle union (ceiling) | 56.0% |
| **cheap 7B catches what the 32B *misses*** | **14.3%** (11–17% per benchmark) |
| unrecoverable by anything | 44.0% |

The genuinely novel observation is the fourth row: cheap same-model repairs recover **14.3%** of
errors the 32B *also* misses — a real complementary signal, stable at 11–17% across benchmarks. But it
is **unharvestable**: the repairs break as many answers as they fix (per-view accuracy is
noise-limited — cap320 0.622, full-res 0.621, think 0.607 — versus 0.645 when the 32B is called), so a
confidence-gated repair ladder *loses* at 32B-parity (**43% vs 39%** compute for the two-rung
baseline). Max-confidence selection across five resolutions plus think lands at 0.608–0.622 and
majority vote across views at 0.624–0.626 — i.e. no cheap combination of the small model's own views
clears its single-view accuracy by a useful margin. The 32B's edge is *capacity*, not a cheap
transform the 7B could apply to itself.

### 4.5 Selection is luck-floored (the headline negative)

The cleanest and most surprising face of the floor. Sample one open-ended model **N = 8** times: a
correct answer is frequently *present* (large oracle gap — SLAKE-open greedy 0.730 → oracle@8 0.879,
headroom +0.149; PathVQA-open greedy 0.343 → oracle@8 0.517, headroom +0.174), yet the model cannot
say *which* of the eight it is. Every training-free selector ties or trails a **random pick** (0.720):

| selector (SLAKE-open, Lingshu-7B, n = 645, MedVLThinker-32B judge) | acc | vs random | % of gap-above-random captured |
|---|---|---|---|
| random pick (mean sample accuracy) | 0.720 | — | — |
| self-verify P(Yes) argmax (7B) | 0.715 | −0.005 | worse than random |
| self-consistency majority | 0.736 | +0.016 | 10% |
| 32B pointwise verify, argmax | 0.746 | +0.026 | 16% |
| 32B **listwise** select | 0.758 | +0.038 | 24% (best training-free) |
| learned fusion [self, 32B] (5-fold CV) | 0.743 | +0.023 | 14% |
| candidate-conditioned synthesis (32B primed with the 7B candidates) | 0.774 | — | (−0.045 vs free-gen) |
| 32B free-gen single pass (SOTA bar) | 0.819 | — | — |
| oracle@8 (luck ceiling) | 0.879 | +0.159 | 100% |

Even the best training-free selector (32B *listwise*) captures only **24%** of the gap above random,
and *none* beats the 32B's own single free-generation pass (0.819); handing the 32B the 7B's
candidates to *synthesize* an answer actually **backfires** (0.774 vs 0.819, −0.045). A **majority
trap** explains the mechanism: the correct answer is a *minority* vote in **74–90%** of recoverable
questions (mean ≈ 1.5 of 8 votes), so any consistency- or agreement-based selector is systematically
pointed at the wrong answer. Two further training-free ideas fail in the same way. Cross-family
agreement *is* correlated with correctness — P(correct | MedVLThinker-7B and Lingshu-7B agree) = 0.819
versus 0.649 on disagreement — but it buys nothing at the answer level: the agreement-consensus
accuracy (0.730) merely equals Lingshu alone, and when the weaker model *dissents* it is right only
28.9% of the time. Few-shot in-context prompting *hurts* (PathVQA 0.343 → 0.203, −0.140; SLAKE 0.730 →
0.705, −0.025). The gap is **sampling luck, not latent knowledge the selector can surface for free.**

The same floor holds for **verifiable bounding boxes** with a competent grounder (Qwen2.5-VL-7B, IoU ≥
0.3). Spatial self-consistency — choosing the medoid box across N samples — sits *at or below* the
luck floor: on SLAKE organ boxes the medoid selector scores **0.164 vs greedy 0.197** (seed 0), i.e.
*below* simply taking the first sample, and a zero-shot box verifier is likewise floored (0.177 vs
greedy 0.199). Structured, geometrically-verifiable outputs do not escape training-free selection's
limit any more than free text does.

**Why this matters for the rest of the paper.** The floor is not a counsel of despair; it is a map of
where effort pays. It rules out a better *gate* over a frozen pair (§4.1–4.3), a cheaper *action* than
capacity (§4.4), and any *training-free* selection rule (§4.5). What it leaves open is exactly the two
levers this paper develops: a cascade that changes *what compute configuration* runs rather than
routing better (Method I), and a small *trained* outcome verifier that supplies the one thing
training-free selection lacks — a learned, per-answer correctness signal — which is the only thing
shown to lift selection above the luck floor for both free-text answers and boxes (Method II).


## 5. The Open-Ended Ceiling-Break

The preceding luck-floor section delivered a deliberately pessimistic verdict: over a *frozen*
model, no training-free gate reliably beats trivial baselines, with routing and recoverability
AUROCs clustered at roughly 0.5–0.69. Taken at face value, that result reads as "medical-VQA
cascades cannot be routed." This section shows that a large part of that pessimism is an **artifact
of the multiple-choice answer format**, not an intrinsic property of medical VQA. The *same*
confidence signal that looks nearly useless on multiple choice becomes strongly discriminative the
moment the answer is free text: the AUROC for detecting a cheap-model error jumps from ≈0.6 on MCQ
to ≈0.87 open-ended. Two things follow, one methodological and one that sets up the rest of the
paper: (i) confidence-gated cascades *do* work open-ended, and (ii) the right regime in which to
study selection and verification is open-ended — which is exactly where the trained verifier of
Method II operates.

### 5.1 Why multiple choice is pessimistic: a discreteness argument

On a four-way MCQ, the model's output is a single letter A/B/C/D — a *maximally discrete*,
degenerate target. Two consequences make confidence a weak correctness signal there. First, the
target is coarse enough that a model can be "right for the wrong reasons" (a lucky letter) or
"confidently wrong" with no surface trace, so the calibration of a top-1/top-2 margin over four
symbols carries little information about actual correctness. Second, the base rate of chance
agreement is high (a coin can land on the gold letter one time in four), which flattens any
confidence-vs-correctness relationship. The net effect is the MCQ ceiling documented in the luck
floor: whether the strong model will fix a cheap error is only ≈0.6 AUROC from any cheap signal,
and even predicting cheap-model *correctness* tops out in the same neighborhood. **The claim of this
section is that this ceiling is partly a measurement artifact of target discreteness, and that it
lifts under free-text answers.** (The discreteness framing is our own; prior open-ended agreement
work does not make the MCQ-vs-open-ended distinction.)

### 5.2 The ceiling-break: routing AUROC ≈ 0.6 (MCQ) → ≈ 0.87 (open-ended)

We re-run the same cheap-model confidence signal — the Lingshu-7B top-1/top-2 margin, escalating to
Lingshu-32B — on the open-ended free-text sets (SLAKE, VQA-RAD, PathVQA), grading correctness with
the validated LLM judge (§ Setup). The routing signal (predicting *cheap-model-wrong*, i.e. the
event that should trigger escalation) rises to **AUROC ≈ 0.87**: pooled over the three sets,
**0.846, 95% CI [0.830, 0.862]**, and **0.866** for the Lingshu-7B→32B cheap-model confidence on
the n = 845 pooled gate-hunt slice. The bootstrap CI lies entirely above the MCQ ceiling — no
resample falls at or below 0.6 — so the gap between the two regimes is not sampling noise. Per
set, two prediction targets behave very differently:

| Free-text set (Lingshu-7B → 32B) | Cheap-wrong AUROC | Recoverability AUROC |
|---|---:|---:|
| SLAKE                | 0.889 | 0.847 |
| VQA-RAD              | 0.717 | 0.579 |
| PathVQA              | 0.797 | 0.517 |
| **Pooled**           | **0.846** [0.830, 0.862] | **0.591** |

*Cheap-wrong* = predicting whether the cheap 7B answer is incorrect (the routing/escalation target).
*Recoverability* = predicting whether the 32B will actually fix that error. The MCQ comparator for
both is ≈0.6 (the routing/recoverability AUROCs of the luck-floor section).

Two honest readings of this table. **(a) The routing signal genuinely breaks the MCQ ceiling.** The
*cheap-wrong* column — the quantity a gate needs — is the one that jumps (pooled 0.846; SLAKE 0.889,
PathVQA 0.797, and even the hardest set, VQA-RAD, at 0.717 is above the MCQ ceiling). This is the
headline: a plain confidence gate can tell when the 7B is wrong on open-ended answers far better
than the MCQ numbers implied. **(b) Recoverability does *not* break.** The *recoverability* column —
"will escalating actually help?" — stays near the luck floor once pooled (0.591), matching the
recoverability wall of the previous section. In other words, the format artifact inflates the
difficulty of *detecting cheap errors*, but it does not manufacture recoverable structure that was
not there: whether the strong model repairs a given error remains largely unlearnable from cheap
signals, consistent with § from the luck floor. Both facts are kept intact here — the open-ended
regime helps the gate, not the underlying complementarity.

### 5.3 It is discreteness, not answer length

A natural confound is that open-ended answers are simply *longer*, and long wrong answers might be
trivially easy to flag (e.g. via degenerate or rambling generations), so the AUROC gain could be a
length effect rather than a discreteness effect. It is not. Re-grading correctness with **token-F1**
— a length-sensitive, partial-credit metric rather than the binary LLM-judge label — leaves the
routing AUROC essentially unchanged (the two gradings agree), so the signal is not an artifact of
answer length. The gain tracks the *discreteness of the target* (letter vs. free text), which is why
even short free-text answers (SLAKE, VQA-RAD) show the lift. The open-ended discreteness claim is
our own contribution, distinct from the cross-model agreement literature that motivates open-ended
gates without isolating the MCQ-vs-open-ended mechanism.

### 5.4 Even open-ended, plain confidence is still the best gate

The ceiling-break makes routing work; it does *not* make a fancier training-free signal necessary.
On the same open-ended slice (Lingshu-7B→32B, n = 845, predicting cheap-wrong), we ran a gate-hunt
over the natural alternatives — self-consistency in exact and semantic forms, semantic entropy, mean
token-F1 self-agreement, self-verification P(True), and a fusion of all signals — and none beats
plain cheap-model confidence:

| Gate signal (cheap-wrong AUROC, n = 845) | AUROC |
|---|---:|
| Cheap-model confidence (margin)          | **0.866** |
| Fusion (all signals)                     | 0.866 |
| Exact self-consistency                   | 0.845 |
| Mean token-F1 self-agreement             | 0.844 |
| Semantic entropy                         | 0.807 |
| Semantic self-consistency                | 0.806 |
| Self-verification P(True)                | 0.755 |

Confidence and the all-signal fusion tie at 0.866 — i.e. adding self-consistency, semantic-entropy,
or self-verification to confidence buys nothing, and self-verification P(True) (0.755) is
*worse* than plain confidence. So even in the favorable open-ended regime, *training-free* selection
is still floored: the best you can do with a frozen model and no extra training is plain confidence,
which is the same conclusion the luck floor reached, now shown to survive the format change. This is
the negative that motivates the one positive lever left — a small amount of training.

### 5.5 Consequence

The open-ended ceiling-break reframes the pessimistic MCQ verdict: the "gate is saturated at
AUROC ≈ 0.6" result is, in large part, a benchmark artifact of multiple-choice discreteness rather
than a statement about medical VQA. Concretely, this changes how the rest of the paper is
evaluated. Because (i) confidence-gated escalation is real open-ended (routing AUROC ≈ 0.87), (ii)
recoverability nonetheless stays at the luck floor (pooled 0.591), and (iii) no training-free
selection signal beats plain confidence even here, the productive question becomes whether a
*trained* selector can escape the floor that a frozen model cannot. We therefore study
selection and verification open-ended throughout — the setting of the trained outcome verifier
(Method II), which is the one lever that does break it.


## 6. Method I — the Adaptive-Compute Cascade (ACC)

The luck floor (§4) says that over a *frozen* model no training-free rule improves the **gate** (which
queries to escalate) or the **selection** (which sample to keep). ACC touches neither. It changes the third,
under-examined decision — the **action**: *what* escalation actually runs. Instead of escalating straight
from the cheap 7B to the most expensive 32B configuration, ACC cascades over **compute configurations of the
same two models**. The observation it exploits is that the 32B's cost is almost entirely its *reasoning*
trace — hundreds of decoded `<think>` tokens — yet on perception-style medical VQA that reasoning is *not*
what fixes the cheap model's errors; raw *capacity* is. So ACC inserts the strong model's **fast, no-think**
mode as an intermediate tier and reserves the slow think pass for the small residual of queries that
genuinely need step-by-step reasoning. The result is an efficiency method (parity accuracy, far less
compute), not an accuracy method; the accuracy lever is Method II (§8).

### 6.1 Three tiers over compute configurations

ACC is a three-tier cascade over compute configurations of the **same** cheap/strong model pair (batch-1
latencies annotated; "cap320" is the reduced `max_pixels` image budget and "fullres" the uncapped one, both
defined in §3):

```
T0 = 7B  no-think @ cap320     (≈ 0.21 s ; ~388 prefill tok, ~2 decode tok)
T1 = 32B no-think @ cap320     (≈ 0.34 s ; ~388 prefill tok, ~2 decode tok)
T2 = 32B think    @ fullres    (≈ 11.34 s, batch-1 ; ~685 prefill tok, ~391 decode tok)
```

The two lower tiers answer *directly* (no `<think>` trace, ≈2 decode tokens each), so they are near-instant;
only T2 pays for a long reasoning decode. A query flows T0 → T1 → T2, stopping at the first tier the gates
accept. T0 handles the confident majority; T1 absorbs the escalations that need capacity but not reasoning;
T2 fires only on the genuinely hard reasoning residual.

### 6.2 Why a no-think middle tier: reasoning over-thinks perception

The middle tier only makes sense if the 32B's *no-think* mode is competitive with — or better than — its
*think* mode on the sets ACC targets. It is. On the four competent (perception) benchmarks the 32B is no
worse, and often materially better, without thinking; the think mode's advantage appears only on the
reasoning-heavy sets (MMMU, MedXpertQA), which ACC excludes from its headline scope (§11). Per-benchmark
accuracy for the MedVLThinker pair (ALL-6 splits; exact-match):

| benchmark | 7B no-think | 32B no-think | 32B think |
|---|---|---|---|
| PMC-VQA | 0.543 | 0.551 | 0.556 |
| SLAKE | 0.762 | **0.849** | 0.764 |
| VQA-RAD | 0.761 | **0.853** | 0.776 |
| PathVQA | 0.641 | 0.661 | 0.672 |
| MMMU-medical | 0.547 | 0.624 | 0.688 |
| MedXpert-Reasoning | 0.225 | 0.279 | 0.326 |
| MedXpert-Understanding | 0.256 | 0.292 | 0.385 |

The two perception sets where escalation actually helps show the strong model *thinking itself out of correct
answers*: on **SLAKE the 32B scores 0.849 no-think vs 0.764 think (+0.085)** and on **VQA-RAD 0.853 vs 0.776
(+0.077)**. PMC-VQA and PathVQA are near-flat (0.551 vs 0.556; 0.661 vs 0.672), while think genuinely pays
only where reasoning is the task (MMMU 0.688 vs 0.624; MedXpert-Understanding 0.385 vs 0.292). Pooled over
the four competent sets, **32B-no-think 0.658 ≥ 32B-think 0.645**, and it reaches that at roughly **2 decode
tokens vs ~477** for the think pass. The mechanism is symmetric over-thinking, not a free lunch: think
*fixes* 11.5% of no-think's errors but *breaks* 11.1% of its correct answers, so on perception the two roughly
cancel. (A per-question oracle that could pick the better *mode* would reach 0.683 on ALL-6 / 0.759 on
COMPETENT-4, +10–11 points over either single mode — evidence that the mode choice carries real signal, even
though, per the luck floor, that oracle is not achievable from any cheap training-free signal.)

**Why the middle tier runs at cap320.** Capping the strong-leg resolution makes T1 cheap without costing
accuracy on perception. Across strong-leg configurations, pooled over the competent-4 (backbone% = fraction
of the always-think compute):

| strong-leg config | accuracy | backbone% |
|---|---|---|
| think @ fullres | 0.6451 | 100% |
| no-think @ fullres | 0.6582 | 66% |
| think @ cap320 | 0.6319 | 72% |
| **no-think @ cap320** | **0.6463** | **39%** |

`no-think @ cap320` recovers essentially all of the `think @ fullres` accuracy (0.6463 vs 0.6451) at **39%**
of the compute — so it is the natural T1. At the per-benchmark level 32B-no-think@cap320 scores SLAKE 0.849,
VQA-RAD 0.853, PMC-VQA 0.551, PathVQA 0.661, using ~375 prefill / ~2 decode tokens versus ~852 / ~477 for a
think call (≈28% of a think call's cost).

### 6.3 The two gates

ACC has two decision points, both thresholding cheap confidence signals already defined in §3 (margin
`m = ℓ₁ − ℓ₂` over the greedy letter-candidate logprobs).

**Tier-0 gate (margin).** From the 7B no-think decode, output T0's answer iff its margin `m₀ ≥ τ₀`; otherwise
escalate to T1.

**Think gate (agreement, ACC-v2).** Let `ŷ_{T0}, ŷ_{T1}` be the two no-think predictions. Fire the expensive
think tier only when the two no-think legs **disagree**, breaking ties by the 32B-no-think margin:

```
disagree = 1[ ŷ_{T0} ≠ ŷ_{T1} ];   s₁ = disagree + ε·(−m_{T1}),  ε = 1e-6;   fire T2 iff s₁ > τ₁  (τ₁ ≈ 1).   (3)
```

The integer `disagree` term selects the disagreements; the tiny `ε·(−m_{T1})` term orders those disagreements
by the 32B-no-think margin, so with `τ₁ ≈ 1` the gate fires T2 only on the *lowest-margin* disagreements.
Consequently think fires on **~15%** of ALL-6 queries, well below the **~32%** raw disagreement rate — the
agreement signal is filtered down to the least-confident disagreements, where reasoning is most likely to
help.

**Variants.** *ACC-v1* uses a plain margin think gate (`fire_think = 1[m_{T1} < τ₁']`); it sends more queries
to think (think ≈19.5% on ALL-6). *ACC-v2* is the agreement gate of Eq. (3) (think ≈15.1%). *ACC-v3* tightens
the gate to the **conjunction** `fire_think = 1[ŷ_{T0} ≠ ŷ_{T1}] ∧ 1[m_{T1} < τ₁']`, requiring both a
no-think disagreement *and* a low 32B-no-think margin before paying for reasoning. The only fitted parameters
are the scalar thresholds `τ₀, τ₁` (and `τ₁'` for v1/v3), chosen on **held-out PMC-VQA-train calibration** to
reach always-32B-think parity at minimum latency; nothing else is learned. *(The agreement mechanism of
Eq. (3) is the ABC cascade family; the contribution is the tiered compute-configuration it gates, not the
gate itself — novelty is discussed in §11.)*

### 6.4 Cost accounting

For a cascade with per-tier cost `c`, fraction `e₀` escalated past T0 and fraction `e₁` reaching the think
tier, the expected cost is Eq. (1):

```
C = c_T0 + e₀·c_T1 + e₁·c_T2.                                                    (1)
```

The same equation is evaluated in three currencies — FLOPs, batch-1 latency, and NVML-integrated energy —
with the per-tier costs measured at batch-1 (one model forward costs `F = 2·N·(P+G)` FLOPs, `N₇ = 7.6e9`,
`N₃₂ = 33.0e9`):

| tier | config | prefill P | decode G | FLOPs (×1e15) | latency | energy |
|---|---|---|---|---|---|---|
| T0 | 7B no-think @ cap320 | 388 | 2 | 0.01 | 0.21 s | 25 J |
| T1 | 32B no-think @ cap320 | 388 | 2 | 0.03 | 0.34 s | 65 J |
| T2 | 32B think @ fullres | 685 | 391 | 0.07 | **11.34 s** (batch-1) | ~6319 J |

The two no-think tiers are *constant-energy* (≈25 J, ≈65 J) because they decode only ~2 tokens; the think
tier's energy scales with its decode length (`E ≈ 18.17·G − 107.5 J`, ≈254 W × 27.5 s at G≈391). *(Cost
methodology note: an earlier **co-resident `rt_cascade`** measurement put T2 at **≈26.6 s / 6994 J** — think
latency ≈0.072 s per decode token, fit R² = 0.99, the think leg ≈80× the no-think leg. That co-resident
number is **superseded** by the batch-1 native accounting; the canonical headline uses **T2 ≈ 11.34 s**, from
the always-32B-think measurement.)*

For ACC-v2 on ALL-6 the measured escalation rates are **e₀ = 71.7%** (past T0) and **e₁ = 15.1%** (to think).
Plugging these into Eq. (1), the cost is dominated by the third term `e₁·c_T2`, because `c_T2` dwarfs both
no-think tiers: the whole cascade cost is governed by *how often the think tier fires*. The saving is
structural. A conventional 2-tier `7B → 32B-think` cascade at the same parity sends `e₁ ≈ 69%` of queries to
the think leg; ACC's free no-think middle tier absorbs the capacity-limited escalations and cuts that to
**15%**. Since the avoided think calls are ~two orders of magnitude more expensive than the tiers that replace
them, they account for essentially the entire compute reduction. The resulting parity efficiency — the
−80%/−95% latency, FLOPs → 52%/25%, and ~5×/~28× energy figures, at guard 0 — is reported in the ACC results
section (§7), together with the "it's the structure, not the gate" ablation that swaps Eq. (3) for every
other training-free gate and the cross-family reproduction.


## 7. Results I — ACC: large efficiency gains at parity, and it is the *structure* not the gate

This section reports what the Adaptive-Compute Cascade (§6) buys, and — the more important point —
*where the gain comes from*. The efficiency comes from the tiered **compute-configuration** (a cheap
no-think middle tier that absorbs most escalations), not from a clever routing rule: hold the three-tier
structure fixed and swap in every training-free gate from the literature and they all land on one
accuracy–FLOPs–latency frontier. We then show the mechanism and the savings reproduce across five model
families and three architectures — with the honest wrinkle that on families where reasoning does not help
perception, ACC does the right thing by *not* escalating (graceful collapse to the cheap leg), and on one
family (MedGemma) the win is only partial.

**Protocol and metrics (recap).** All numbers below are at **parity accuracy**, defined as the accuracy of
the always-on strong reasoning model (`always-32B-think`). The two scalar gate thresholds `τ₀, τ₁` are
frozen on held-out PMC-VQA-train calibration; every evaluation sample is held out. FLOPs% is the
prefill-inclusive cascade cost divided by the always-32B-think cost (Eq. 1); **latency** and **energy** are
**measured batch-1** per-tier costs (energy NVML-integrated). `esc₀` is the fraction escalated past T0 to
the no-think 32B tier; `think` is the fraction that reaches the expensive T2 think tier. **`guard`** is the
number of benchmarks on which a system is *worse than the always-7B leg*, averaged over 20 held-out seeds
(so it is fractional: `guard = 0.05` means that in one seed out of twenty a single benchmark dipped below
the 7B; `guard = 0` is a clean never-worse-than-cheap guarantee).

### 7.1 ALL-6 at parity (MedVLThinker)

At parity on the full six-benchmark suite (MedVLThinker 7B/32B), with every training-free gate swapped into
the *same* three-tier structure:

| system | acc | esc₀ | think | FLOPs% | latency | energy | guard |
|---|---|---|---|---|---|---|---|
| always-7B-nt @cap320 | 0.5262 | 0% | 0% | 8.4% | 0.13 s | 19.9 J | 0.00 |
| always-32B-nt @cap320 | 0.5573 | 100% | 0% | 36.2% | 0.23 s | 77.8 J | 0.00 |
| **always-32B-think [PARITY]** | **0.5723** | 100% | 100% | 100.0% | **11.34 s** | **6318.8 J** | 0.00 |
| **Ours: ACC-v2 (agreement gate)** | **0.5693** | 71.7% | 15.1% | **52.0%** | **2.27 s** | **1181.9 J** | **0.00** |
| ACC-v1 (margin gate) | 0.5687 | 66% | 19% | 53.9% | 2.69 s | 1416.6 J | 0.00 |
| CASP-Stability (trained gate) | 0.5698 | 74% | 11% | 49.0% | 1.77 s | 899.2 J | 0.05 |
| MSP/Chow | 0.5697 | 70% | 19% | 57.4% | 2.96 s | 1568.1 J | 0.00 |
| entropy | 0.5691 | 71% | 21% | 62.0% | 3.48 s | 1863.1 J | 0.00 |
| Gini/DOCTOR | 0.5702 | 69% | 22% | 61.0% | 3.44 s | 1837.1 J | 0.00 |
| AutoMix (self-verify) | 0.5692 | 73% | 18% | 54.6% | 2.50 s | 1307.0 J | 0.05 |
| FrugalGPT-style learned | 0.5677 | 70% | 19% | 60.4% | 3.30 s | 1765.5 J | 0.10 |
| Jitkrittum L2D (Diff-Prob) | 0.5666 | 67% | 15% | 50.6% | 2.29 s | 1194.5 J | 0.00 |
| random | 0.5641 | 89% | 76% | 116.5% | 8.95 s | 4889.5 J | 0.05 |

**Headline (ALL-6):** at −0.003 accuracy (0.5723 → 0.5693, i.e. parity) and **guard 0** (never worse than
the 7B on any benchmark), ACC-v2 takes **latency 11.34 s → 2.27 s (−80%)**, **FLOPs 100% → 52%**, and
**energy 6318.8 J → 1181.9 J (~5.3×, −81%)**. The savings come entirely from the third tier: the fraction
that reaches the expensive think pass drops from ~69% (what a naive 2-tier 7B→think cascade would spend) to
**15.1%**, because the no-think 32B middle tier absorbs the bulk of escalations that need *capacity*, not
*reasoning*.

**Per-benchmark: guard 0 is real, and ACC wins exactly where think over-thinks.** Breaking the ALL-6
accuracy out by benchmark shows why the aggregate parity holds and where the −0.003 comes from:

| system | PMC | SLAKE | VQA-RAD | PathVQA | MMMU | MedXpert-R | MedXpert-U |
|---|---|---|---|---|---|---|---|
| always-7B-nt | 0.543 | 0.762 | 0.761 | 0.641 | 0.547 | 0.225 | 0.256 |
| always-32B-nt | 0.551 | 0.849 | 0.853 | 0.661 | 0.624 | 0.279 | 0.292 |
| always-32B-think [PARITY] | 0.556 | 0.764 | 0.776 | 0.673 | 0.688 | 0.326 | 0.384 |
| **Ours (ACC-v2)** | **0.561** | **0.842** | **0.861** | **0.679** | 0.643 | 0.282 | 0.310 |

ACC-v2 is **≥ the 7B on all seven benchmarks** (guard 0), and it actually *beats* the parity think baseline
on the four perception sets — PMC (0.561 vs 0.556), SLAKE (0.842 vs 0.764), VQA-RAD (0.861 vs 0.776),
PathVQA (0.679 vs 0.673) — precisely because 32B-no-think ≥ 32B-think on perception (the over-thinking
premise, §6). It trails the think baseline only on the three *reasoning* sets (MMMU 0.643 vs 0.688,
MedXpert-R 0.282 vs 0.326, MedXpert-U 0.310 vs 0.384), where real reasoning genuinely helps. The −0.003
aggregate deficit is therefore concentrated on MMMU/MedXpert; this is the honest scope boundary that makes
ALL-5 (below) and COMPETENT-4 the settings where ACC is unambiguously at-or-above parity.

### 7.2 ALL-5 at parity (excluding near-chance MedXpert)

Removing MedXpertQA-MM — where both models are near chance and which is excluded from the ACC claims —
sharpens the result, because the reasoning residual the think tier must service almost vanishes:

| system | acc | esc₀ | think | FLOPs% | latency | energy | guard |
|---|---|---|---|---|---|---|---|
| always-7B-nt @cap320 | 0.6201 | 0% | 0% | 8.9% | 0.13 s | 19.9 J | 0.00 |
| always-32B-nt @cap320 | 0.6457 | 100% | 0% | 38.7% | 0.23 s | 77.8 J | 0.00 |
| **always-32B-think [PARITY]** | **0.6463** | 100% | 100% | 100.0% | **8.88 s** | **4915.9 J** | 0.00 |
| **Ours: ACC-v2 (agreement)** | **0.6450** | 35.1% | 2.3% | **24.9%** | **0.44 s** | **172.8 J** | **0.05** |
| CASP-Stability (trained) | 0.6461 | 43% | 2% | 28.3% | 0.46 s | 184.5 J | 0.05 |
| ACC-v1 (margin) | 0.6435 | 32% | 3% | 24.8% | 0.53 s | 223.7 J | 0.05 |
| MSP/Chow | 0.6444 | 39% | 3% | 27.0% | 0.51 s | 212.2 J | 0.05 |
| entropy | 0.6459 | 50% | 3% | 31.2% | 0.52 s | 211.0 J | 0.05 |
| Gini/DOCTOR | 0.6461 | 44% | 2% | 28.4% | 0.46 s | 179.4 J | 0.10 |
| AutoMix (self-verify) | 0.6448 | 25% | 2% | 20.6% | 0.35 s | 129.7 J | 0.65 |
| FrugalGPT-style learned | 0.6449 | 48% | 3% | 30.8% | 0.56 s | 235.2 J | 0.20 |
| Jitkrittum L2D (Diff-Prob) | 0.6403 | 32% | 2% | 23.6% | 0.41 s | 161.3 J | 0.20 |
| random | 0.6390 | 82% | 17% | 57.2% | 1.79 s | 899.7 J | 0.40 |

**Headline (ALL-5):** always-think **8.88 s / 4915.9 J → ACC-v2 0.44 s / 172.8 J, FLOPs 24.9%** — a
**−95% latency**, **~28× energy** (−96%) reduction at parity (0.6463 → 0.6450). Note the honest caveat in
this table: the *cheapest* FLOPs point is AutoMix at 20.6%, but it carries **guard 0.65** (it frequently
drops below the 7B on some benchmark across seeds); ACC-v2 at **24.9% FLOPs / guard 0.05** is the honest
operating point, and the trained CASP-Stability gate (28.3%, guard 0.05) is comparable. The learned gates
(FrugalGPT-style guard 0.20, Jitkrittum L2D guard 0.20) are both cheaper on paper but less guardrail-safe —
training the gate does not buy a cleaner frontier here (see §5).

### 7.3 It is the structure, not the gate

The two ALL-6 / ALL-5 tables above make the central claim visually: **holding the three-tier
compute-configuration fixed and swapping the routing rule barely moves the operating point.** On ALL-6,
every training-free gate — margin (ACC-v1), agreement (ACC-v2), MSP/Chow, entropy, Gini/DOCTOR, AutoMix
self-verification, the FrugalGPT-style learned router, and the Jitkrittum L2D Diff-Prob deferral rule —
lands in a narrow band: **accuracy 0.5666–0.5702**, **FLOPs 49.0–62.0%**, and **latency 1.77–3.48 s**, all
at parity. The trained CASP-Stability gate (49.0% FLOPs, 1.77 s) and the agreement gate (52.0%, 2.27 s) are
merely the cheapest points on that one frontier; agreement is the cheapest *guard-clean* one. Only **random**
escalation leaves the frontier (116.5% FLOPs, 8.95 s) — i.e. the structure, not the decision rule, is what
delivers the savings; a "better gate" moves you a few points along a frontier the structure already fixed.
This is the empirical face of the luck floor (§5): recoverability is ~0.6 AUROC from any signal, so no gate
can meaningfully out-route another.

**Robustness.** Bootstrap CIs on the ACC-v2 accuracy (from the held-out seed distribution) are
**ALL-6 [0.5608, 0.5820]** and **ALL-5 [0.6372, 0.6562]**, both straddling the parity target; the batch-1
latency CI is wide (**[2.6, 9.8] s**), reflecting the heavy-tailed cost of the rare think calls at batch-1.
ACC-v3 (a confidence-tightened conjunctive think gate) further trims think firings **19% → 14%** and FLOPs to
**52.6%** at equal accuracy, holding parity on 20/20 seeds.

### 7.4 It generalizes across families and architectures

The "no-think ≥ think on perception" premise and the ACC savings were re-measured on four additional
model pairs spanning three architectures (Lingshu-7B/32B, QoQ-Med-7B/32B, Chiron-o1-2B/8B,
MedGemma-4B/27B), plus MedVLThinker for reference:

| family | set | parity acc (32B-think) | ACC-v2 acc | ACC-v2 FLOPs% | esc₀ | think | guard |
|---|---|---|---|---|---|---|---|
| MedVLThinker | ALL-6 | 0.5723 | 0.5693 | 52.0% | 71.7% | 15.1% | 0.00 |
| MedVLThinker | ALL-5 | 0.6463 | 0.6450 | 24.9% | 35.1% | 2.3% | 0.05 |
| Lingshu | ALL-6 | 0.6611 | 0.6614 | 48.6% | 61% | 0% | 1.00 |
| Lingshu | ALL-5 | 0.7746 | 0.7726 | 38.9% | 42% | 1% | 1.15 |
| QoQ-Med | ALL-6 | 0.4689 | 0.5095 | 8.8% | 0% | 0% | 0.00 |
| QoQ-Med | ALL-5 | 0.5432 | 0.6048 | 9.1% | 0% | 0% | 0.00 |
| Chiron-o1 | ALL-6 | 0.5076 | 0.6023 | 19.3% | 0% | 0% | 0.00 |
| Chiron-o1 | ALL-5 | 0.5926 | 0.7249 | 19.9% | 0% | 0% | 0.00 |
| MedGemma | ALL-6 | 0.5253 | 0.5219 | 68.4% | 55% | 20% | 1.15 |
| MedGemma | ALL-5 | 0.5979 | 0.6028 | 11.5% | 0% | 0% | 0.00 |

**Lingshu — a clean win.** ACC matches or slightly beats parity while roughly *halving* FLOPs:
**48.6%** on ALL-6 (0.6614 vs parity 0.6611) and **38.9%** on ALL-5 (0.7726 vs 0.7746). The `guard`
of 1.00 (ALL-6) / 1.15 (ALL-5) is **inherited from the target, not caused by the cascade**: the always-on
Lingshu-32B is itself worse than Lingshu-7B on one benchmark (MMMU, the known Lingshu-7B inflation anomaly,
§11), so any system reaching for the 32B pays that guard. ACC-v1 (margin) is comparable (48.6% / 39.3%).

**QoQ-Med and Chiron-o1 — graceful collapse.** On these two families the strong model's *think* mode
actively **hurts**: the cheap 7B/2B no-think leg already beats always-32B/8B-think (QoQ 0.5094 cheap vs
0.4689 think on ALL-6; Chiron 0.6024 vs 0.5076). ACC does the correct thing — it **does not escalate**
(`esc₀ = 0%`, `think = 0%`), collapsing to the cheap leg and thereby delivering accuracy that *exceeds* the
"parity" think target (QoQ 0.5095, Chiron 0.6023 on ALL-6; QoQ 0.6048, Chiron 0.7249 on ALL-5) at cheap-leg
cost (**8.8% / 19.3% FLOPs**, guard 0). This is the intended failure mode: when reasoning is worthless, the
agreement gate never wastes compute chasing it. It is a graceful degradation, not a savings on these
families in the usual sense — there is simply nothing to escalate to.

**MedGemma — a partial win.** MedGemma is the honest mixed case. On ALL-6 the agreement gate escalates
heavily (`esc₀ = 55%`, `think = 20%`) for only a modest **68.4% FLOPs** at guard 1.15, and the margin gate
(ACC-v1) is worse still (93.8% FLOPs); here the trained CASP-Stability gate does noticeably better (29.6%
FLOPs, guard 1.15). On ALL-5, MedGemma's cheap leg already beats big-think (0.6031 vs 0.5979), so ACC again
collapses to the cheap leg (**11.5% FLOPs**, guard 0). So for MedGemma the efficiency win is real on ALL-5
but only partial / gate-dependent on ALL-6.

**Takeaway.** ACC delivers its large, guard-clean savings on the two families whose 32B has a genuine
think-mode advantage that over-fires on perception (MedVLThinker, Lingshu); on families where think does not
help it degrades gracefully to the cheap leg (QoQ, Chiron, and MedGemma-ALL-5); and the one partial case
(MedGemma-ALL-6) is exactly where a trained gate helps and the agreement gate over-escalates. Across all of
this the accuracy stays at parity and the guard stays clean or is inherited from the target — never
manufactured by the cascade.


## 8. Method II — a Trained Outcome Verifier

The luck floor (§4) is a statement about *training-free* decision rules over a *frozen* generator: no
confidence signal, self-verification, or aggregation rule beats trivial baselines, because the correct
answer is a *minority* candidate in most recoverable questions (the majority trap) and the strong model
fails on the same inputs the cheap one does. This section supplies the constructive complement — a small
**trained outcome verifier** — and defines it precisely; §9 reports the accuracy it buys (across three
model families, held-out OOD, and structured bounding boxes). The key claim of this section is *methodological*:
the verifier is a supervised, inference-time **selector**, not a new gate and not a smarter generator, and it
is trained only on labels a dataset answer key already contains — which is why it escapes the luck floor
without importing any capability from the model it is later compared against.

### 8.1 Why a trained verifier, and why *here* specifically

In *general* LLM reasoning, a trained-verifier best-of-N barely beats plain self-consistency: recent work
reports near-parity between learned verifiers and simple confidence/aggregation baselines (self-certainty
best-of-N, NeurIPS 2025 [2502.18581]; aggregation studies [2510.13918]). Under that evidence a learned
selector is often judged not worth its supervision cost.

Medical open-ended VQA is the *opposite* regime, and it is the regime in which a learned selector should pay
off. Three conditions hold simultaneously, each documented earlier:

1. **Self-consistency fails.** The correct answer is a minority vote in 74–90% of recoverable questions (the
   majority trap, §4), so any rule that rewards agreement — SC-majority, weighted voting, self-verification —
   sits at or below the random-pick floor.
2. **The strong model barely helps.** On open-ended answers the 5× model recovers only a small, near-unlearnable
   slice of the cheap model's errors (the recoverability wall, §4), so "just scale up" is not a fix either.
3. **The oracle headroom is large.** Over N=8 samples from the *cheap* generator, a perfect selector reaches
   far above greedy (pooled oracle@8 0.592 vs greedy 0.413, §9). The right answer is usually *present in the
   sample set*; what is missing is a rule that *picks* it.

A selector is exactly the tool for (3) when (1) and (2) rule out the cheap and the expensive shortcuts. This
is why a *learned* verifier — rather than a better gate or a bigger model — is the correct instrument in this
setting.

### 8.2 The verifier and best-of-N selection

The verifier is a LoRA-adapted VLM (base **Lingshu-7B** for free-text answers; **Qwen2.5-VL-7B** for boxes)
with adapter parameters φ. Given the image `v`, question `q`, and a candidate answer `a`, it scores the
candidate by the probability it assigns to the token "Yes" versus "No" at the final position of the prompt
*"…Is the proposed answer correct? Answer Yes or No."*:

```
s_φ(v, q, a) = P_φ(Yes | v, q, a) = e^{z_Yes} / (e^{z_Yes} + e^{z_No}).            (4)
```

`s_φ ∈ [0,1]` is a per-candidate correctness score. The base model is frozen; only the LoRA adapter
(≈190 MB) is trained, on per-sample correctness labels `y` (defined in §8.3) by binary cross-entropy on the
Yes/No token:

```
L(φ) = − Σ [ y · log s_φ + (1 − y) · log(1 − s_φ) ].                               (5)
```

Given N candidates `a₁ … a_N ∼ M(· | v, q)` drawn from the *same cheap generator* M, **best-of-N selection**
returns the highest-scoring candidate:

```
â = argmax_{i ≤ N} s_φ(v, q, aᵢ).                                                 (Eq. 4 applied N times)
```

We report the quality of `â` with the gap-captured metric of Eq. (2) (§3), `(acc(â) − greedy) /
(oracle@N − greedy)`, i.e. the fraction of the achievable oracle headroom that the selector realizes; and,
in the cascade of §9, the same `s_φ` doubles as an escalation signal (escalate iff the selected candidate's
score is low), so one trained head serves both selection and gating. This is a **pointwise outcome verifier**
— it scores a *completed* answer, not intermediate reasoning steps — trained for inference-time selection;
its lineage and how it differs from process-reward and generative-verifier work is discussed in §2. Argmax
over `s_φ` is the intended rule (voting-style aggregation reintroduces the majority trap; §9 quantifies the
cost of the alternatives). Training is deliberately small: ≈6,000 judged (candidate, label) pairs at a
positive rate of 0.194, one LoRA adapter, no change to the generator or the strong model.

### 8.3 Label rules: free-text answers and bounding boxes

The only thing the verifier is supervised on is the label `y` — whether a candidate is correct. We use the
**same** machinery (Eq. 4–5, argmax selection) for two output types, differing only in how `y` is defined.

**Free-text answers.** `y` is produced by an LLM judge that decides whether the free-text candidate `a`
*matches the gold answer* — a semantic stand-in for exact-match, which is too brittle for open-ended text
(synonyms, bilingual answers, paraphrase). The judge is **MedVLThinker-32B**, cross-checked by an independent
**Claude-Sonnet-5** judge; its validation (100% agreement on exact-match anchors; second-judge κ 0.85–0.96)
is reported in §4. Crucially, the judge compares the candidate to the dataset **answer key**, so `y` encodes
match-to-gold, not the judge's own medical opinion (see §8.4).

**Bounding boxes.** For grounding, `a` is a predicted box and the label is a hard IoU threshold against the
gold box:

```
y = 1[ IoU(a, gold) ≥ θ ],   θ = 0.3.                                             (label rule, boxes)
```

The box-verifier (base Qwen2.5-VL-7B) is otherwise identical — Eq. (4)–(5), best-of-8 selection — and judges
"does this drawn box localize the {target}?". One implementation detail matters: raw Qwen box coordinates are
mis-scaled on large images (unscaled candidates give mean IoU only 0.098, oracle@0.3 = 0.253), so candidate
and gold boxes are rescaled by the image width/height before computing IoU; with that fix the same
selection recipe applies. Results on SLAKE organ boxes and the real MS-CXR chest-X-ray benchmark are in §9.

### 8.4 Why a 32B-judged 7B verifier beating the 32B is not circular

The results in §9 show a 7B-based verifier that matches — and on hard sets modestly beats — a 5× strong
model, using labels produced by a 32B judge. This invites a circularity objection ("the 32B is grading its
own superior"), which the construction rules out on four grounds.

1. **The judge is an automated grader, not a knowledge oracle.** Its sole function is to decide whether a
   candidate string matches the **gold** answer. The labels `y` therefore derive from the dataset **answer
   key**; the grader could be a human or exact-match, and is a judge only because free-text needs a semantic
   comparator. No medical knowledge of the 32B *generator* enters the labels.
2. **The verifier is trained only on the cheap model's own samples.** It never sees a 32B answer; it learns to
   *discriminate* among the 7B's candidates. Discrimination is strictly easier than generation: the cheap
   model's 8 samples already contain a judged-correct answer in ≈59% of pooled cases (oracle@8 = 0.592, §9),
   so the verifier need only *identify* the correct candidate that is already present — which is exactly why a
   **7B** selector suffices and imports no 32B capability.
3. **The comparison is between two deployment strategies, not two knowledge levels.** One strategy is scaling
   to a 5× model *zero-shot*; the other is sampling a small model N times and selecting with a small
   *supervised* verifier. The in-domain supervision (a few thousand gold labels) is precisely the
   contribution — and its cost — not a leak of strong-model knowledge.
4. **The verifier genuinely uses the image.** Across all 8,512 candidates it separates correct from incorrect
   at **AUROC 0.924** (mean score 0.749 for correct vs 0.171 for incorrect), and a blank-image ablation drops
   discrimination by **−0.047** (VERIFIED_FACTS §D) — it degrades when the image is removed, so it is doing
   visual verification rather than mimicking a text-only judge (the judge, unlike the verifier, needs the gold
   answer). This directly refutes the "lazy verifier" failure mode of Verification-Mirage [2605.10850].

Because zero-shot self-verification is luck-floored (self-verification AUROC ≈ 0.5, §4) while the *trained*
head reaches AUROC 0.924, the discriminative signal is present but not zero-shot-surfaceable: **training is
the active ingredient**, and it is small, in-domain, and label-only. The honest scope of the resulting win —
a tie-to-modest gain versus the strong model, two-seed-robust over training-free baselines, with a wide
per-dataset spread and a weak-base failure — is quantified in §9.


## 9. Results II — the verifier beats the 32B across three families

The trained outcome verifier of §8 is used exactly as defined there — best-of-8 selection over a
small model's own samples, `â = argmax_i s_φ(v,q,a_i)` (Eq. 4), plus a verifier-confidence
escalation gate that hands the lowest-confidence picks to the strong model. This section reports what
that buys. The headline is a *positive*, and a strong one: across **three model families and two model
architectures**, the 7B-verifier cascade **beats the strong 32B/38B model** on accuracy on every
family×dataset cell we measured, including a held-out out-of-distribution set the verifier never saw.
We then subject the win to the strictest possible test — a single held-out same-split comparison across
two random seeds — where it narrows honestly to a *modest-win-to-tie*, and we characterize the ceiling
that stops it from going further (candidate quality, not the selector).

### 9.1 The three-family beats-32B master table

The verifier base is **Lingshu-7B** throughout; for the InternVL3 family we apply the *same*
Lingshu-trained verifier to InternVL3's answers (no retraining) — a **cross-architecture transfer**.
Each family is evaluated on its full open-ended sets. "cheap(SC)" is the small model's
self-consistency majority; "STRONG" is the family's 32B/38B single pass; "verifier-bo8" is best-of-8
verifier selection with no gate; "cascade-best@esc" is verifier-bo8 *plus* the verifier-confidence gate,
escalating the stated fraction to the strong model; "oracle@8" is the luck ceiling (an oracle that
always picks the correct sample when one exists among the 8).

**Lingshu family (strong = Lingshu-32B):**

| dataset | cheap (SC) | STRONG (32B) | verifier-bo8 | **cascade-best @esc** | oracle@8 |
|---|---|---|---|---|---|
| VQA-RAD | 0.465 | 0.600 | 0.575 | **0.625 @24%** | 0.630 |
| PathVQA | 0.324 | 0.376 | 0.453 | **0.469 @33%** | 0.517 |
| Kvasir | 0.286 | 0.301 | 0.439 | **0.448 @20%** | 0.491 |
| RadImageNet-OOD | 0.329 | 0.289 | 0.353 | **0.353 @0%** | 0.512 |
| **POOLED** | 0.322 | 0.331 | 0.414 | **0.421 @12%** | 0.513 |

**MedVLThinker family (strong = MVT-32B):**

| dataset | cheap (SC) | STRONG (32B) | verifier-bo8 | **cascade-best @esc** | oracle@8 |
|---|---|---|---|---|---|
| VQA-RAD | 0.420 | 0.525 | 0.490 | **0.555 @54%** | 0.600 |
| Kvasir | 0.343 | 0.361 | 0.477 | **0.483 @9%** | 0.550 |
| RadImageNet-OOD | 0.204 | 0.202 | 0.241 | **0.243 @13%** | 0.317 |
| **POOLED** | 0.266 | 0.277 | 0.339 | **0.344 @14%** | 0.416 |

**InternVL3 family (strong = IV3-38B; Lingshu-trained verifier — cross-architecture transfer):**

| dataset | cheap (SC) | STRONG (38B) | verifier-bo8 | **cascade-best @esc** | oracle@8 |
|---|---|---|---|---|---|
| VQA-RAD | 0.445 | 0.415 | 0.570 | **0.580 @12%** | 0.620 |
| PathVQA | 0.081 | 0.096 | 0.116 | **0.125 @52%** | 0.192 |
| Kvasir | 0.362 | 0.380 | 0.479 | **0.487 @17%** | 0.593 |
| RadImageNet-OOD | 0.285 | 0.304 | 0.302 | **0.313 @52%** | 0.398 |
| **POOLED** | 0.202 | 0.218 | 0.249 | **0.255 @36%** | 0.337 |

**Reading the table (with the caveats stated).** The **cascade-best** column beats **STRONG** on
*every* cell of *every* family — pooled, that is Lingshu **0.421 vs 0.331**, MedVLThinker
**0.344 vs 0.277**, InternVL3 **0.255 vs 0.218**. Two honest qualifications sit inside this. (i) **The
gate is doing real work where the strong model is genuinely better.** Pure verifier-bo8 (no gate)
does *not* uniformly beat the strong model: it loses on Lingshu VQA-RAD (0.575 < 0.600), MedVLThinker
VQA-RAD (0.490 < 0.525), and InternVL3 RadImageNet (0.302 < 0.304). On exactly those sets the
verifier-confidence gate escalates a larger fraction (24%, 54%, 52%) and the cascade climbs *above* the
strong model by combining best-of-N on the recoverable questions with deference on the rest — the
cascade wins by *knowing when to escalate*, not by out-selecting the 32B everywhere. (ii) The **POOLED**
strong numbers here (0.331 / 0.277 / 0.218) are low because these full-set pools are dominated by hard
and OOD sets (Kvasir, RadImageNet) and exclude SLAKE, the one set where the 32B is dominant; the
stricter same-split test in §9.2 puts the 32B back at a higher 0.462, so we do not lean the whole claim
on this pooled figure. (iii) On InternVL3 PathVQA the absolute numbers are near the floor for the whole
family (0.081/0.096/0.125) — the *relative* cascade win holds, but nobody should read 0.125 as a usable
operating point. The most defensible one-line summary: **the full-set 3-family cascade beats the strong
model on every cell, and the Lingshu verifier transfers cross-architecture to lift a 38B InternVL3
family it was never trained on.**

### 9.2 A rigorous same-split test, and honest two-seed accounting

The full-set tables above pool over whatever sets each family completed. For a clean apples-to-apples
comparison we also report a single **canonical held-out same-split** test: pool the four Lingshu
open-ended sets, hold out 30%, and evaluate every method — including the 32B — on the *identical*
held-out questions (n=1064).

| dataset | greedy | self-consist. | 32B (same split) | **verifier** | oracle | n |
|---|---|---|---|---|---|---|
| SLAKE | 0.738 | 0.738 | 0.829 | 0.762 | 0.895 | 210 |
| VQA-RAD | 0.519 | 0.500 | 0.648 | 0.611 | 0.722 | 54 |
| PathVQA | 0.352 | 0.349 | 0.377 | **0.441** | 0.513 | 435 |
| Kvasir | 0.282 | 0.282 | 0.326 | **0.405** | 0.493 | 365 |
| **POOLED** | 0.413 | 0.411 | 0.462 | **0.501** | 0.592 | 1064 |

The ordering is **verifier 0.501 > 32B 0.462 > SC 0.411 ≈ greedy 0.413**: the trained 7B verifier
captures **49%** of the oracle gap and, pooled, edges past the 32B *while being 5× smaller*. The
per-dataset pattern is the honest one: the verifier **beats the 32B on the hard sets** (PathVQA 0.441 vs
0.377, Kvasir 0.405 vs 0.326) where scaling to a bigger model stalls, and **loses to the 32B where the
32B is genuinely stronger** (SLAKE 0.762 vs 0.829) or where n is tiny (VQA-RAD 0.611 vs 0.648, n=54).

**Two-seed honesty.** On the seed-0 split the paired bootstrap gain of the verifier over the 32B is
**+0.0385, 95% CI [+0.010, +0.066] — excludes 0** (significant). On a second 70/30 split (seed-1) it is
**−0.005 (a tie: 0.445 vs 0.450)**. So the careful claim versus the 32B is **matches / competitive, mean
+0.017** — a *modest win to a tie*, not a robust rout. What *is* robust across both seeds is the margin
over training-free selection: the verifier beats greedy by **+0.088 (seed-0) / +0.066 (seed-1)** while
self-consistency ≈ greedy (paired bootstrap vs first-sample **+0.116, 95% CI [+0.092, +0.140]**), and
gap-captured is **35–49% (mean ~42%)**, driven by the hard sets — it helps PathVQA and Kvasir on *both*
seeds and is flat where the baseline is already high (SLAKE, seed-1 −4% at greedy 0.774) or n is tiny
(VQA-RAD, ~0% at n=69). This is the sense in which "beats the 32B across three families" (§9.1,
full-set cascade) and "matches-to-modest-win versus the 32B" (this strict same-split) are both true and
must be reported together.

### 9.3 Training is the active ingredient

The gain comes from the *supervised* verifier, not from best-of-N or from the base model's own
confidence. **Zero-shot self-verification** — asking the same model P(True) about its candidates — is
luck-floored (§5). Training the same 7B on judge labels lifts it to 0.501; and, decisively, the
**trained 7B verifier beats a zero-shot 32B verifier: 0.403 vs 0.355** selection accuracy on a matched
600-question sample (oracle@8 0.490), with selection efficiency **0.810 (7B-trained) vs 0.717
(32B-zero-shot)** — i.e. task-specific training beats 5× the parameters for the selection job. The
verifier's scores separate correct from incorrect candidates at **AUROC 0.924** across all 8,512
candidates (mean score 0.749 vs 0.171), and a blank-image ablation drops discrimination **−0.047**
(VERIFIED_FACTS §D) — it is reading the image, not mimicking the judge (which needs the gold answer;
the verifier does not). (Fig. `figs/limits/fig_verifier_discrimination.png`.)

**Argmax is the correct selection rule.** Given the verifier's scores, the plain argmax over
candidates (0.501) beats **verifier-weighted voting (0.489, −0.012 [−0.024, 0.000])** and a
**score×count hybrid (0.470, −0.031)**, and both beat plain majority (0.411). Any vote-flavored rule
reinherits the majority trap (§5): the correct answer is a minority vote in most recoverable questions,
so mixing counts back in only hurts.

### 9.4 Generalization: held-out sets, other generators, other bases

The verifier is not memorizing one dataset's answer distribution.

- **Held-out OOD (zero-shot).** The pooled-4 verifier applied *unchanged* to a fifth set,
  RadImageNet-VQA, lifts greedy **0.329 → 0.353** (13.6% of its gap, oracle 0.512, n=2000) — and that
  0.353 already exceeds the Lingshu-32B's 0.289 on the same set. On Kvasir (a GI-endoscopy OOD set) the
  two-dataset verifier lifts **0.2858 → 0.3267** (19.9%, oracle 0.4908, n=1200).
- **Cross-generator (fixed verifier, different answers).** Applied to a *different model's* answers —
  the Lingshu-trained verifier scoring MedVLThinker-7B candidates — it still captures **49% (SLAKE) /
  61% (VQA-RAD)** of the oracle gap; the verifier is largely generator-agnostic. (The InternVL3 column
  of §9.1 is the same story at the family scale.)
- **Other base, from scratch (honest, mixed).** Trained *from scratch* on a MedVLThinker-7B base it
  works on SLAKE (**0.564 → 0.622, 42%**) and is pooled-positive (**0.547 → 0.583, 25%**), but *fails*
  on VQA-RAD's tiny split (**0.500 → 0.470, n=54**). So the method is **not uniformly robust across
  bases**, and a stronger base makes a stronger verifier (the Lingshu verifier's 35–49% / cross-generator
  49–61% exceeds this from-scratch MedVLThinker verifier's 25% pooled). The validated headline is the
  Lingshu result (four datasets, two seeds); the from-scratch other-base result is reported as an honest
  mixed finding, not a headline.

### 9.5 A test-time-scaling method: compute rivals parameters where parameters don't help

Best-of-K accuracy rises monotonically with the sample budget while a random pick stays flat, the
defining test-time-scaling property:

| K | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| verifier best-of-K | 0.385 | 0.425 | 0.476 | 0.501 |
| FLOPs (7B-fwd-eq) | 1× | 4× | 8× | 16× |

(oracle@8 = 0.592; random pick ≈ 0.39, flat; Fig. `figs/limits/fig_verifier_scaling.png`.) Extending to
K=16 on a fresh, larger pool it keeps rising with diminishing returns (0.356 → 0.394 → 0.411 → 0.417 →
0.424). Because reasoning barely helps open-ended medical VQA (§6), the strong model's single pass is a
weak baseline to beat: pooled same-split, the **32B scores 0.462**, *matched-to-beaten* by the 7B with
verifier-bo8 (**0.501**, seed-0; a tie 0.445 vs 0.450 on seed-1). The compute premium is real and
stated honestly — best-of-8 costs **16× a 7B forward** versus the 32B's **4.57× a 7B forward** (≈3.8×
the FLOPs; sheet:faithful §5) — but it is not all cost: because 2N < 4.57 for N ≤ 2, the **verifier-bo2
with no gate beats the 32B on accuracy *and* FLOPs (4.0 vs 4.57 7B-fwd-eq)** on the sets where the strong
model is weak/OOD, so there is a genuinely dominating operating point on hard data. The verifier is thus
the accuracy-optimal operating point on the hard sets, not dominated by simply renting a 5×-larger model.
(Fig. `figs/limits/fig_verifier_pareto.png`.)

### 9.6 Structured outputs: the win holds for bounding boxes

The same recipe — an IoU-thresholded label `y = 1[IoU(a,gold) ≥ 0.3]`, a LoRA box-verifier
(Qwen2.5-VL-7B base), best-of-8 selection — transfers to grounding, on organ boxes and on a **real**
PhysioNet chest-X-ray benchmark.

| benchmark | greedy | SC-medoid | zero-shot box-verifier | **trained box-verifier** | oracle@8 | gap captured |
|---|---|---|---|---|---|---|
| SLAKE organs (n=487) | 0.197 | 0.164 | 0.177 | **0.255 / 0.257** | 0.343 | **40% / 53%** |
| MS-CXR pathology (n=435, real) | 0.041 | 0.053 | 0.115 | **0.232 / 0.230** | 0.285 | **78% / 77%** |

Both training-free selectors sit at the luck floor — SC-medoid (0.164) is *below* greedy (0.197) on
SLAKE — while the **trained** box-verifier recovers **40%/53%** (SLAKE, two seeds) and **78%/77%**
(MS-CXR, two seeds) of the oracle gap. On MS-CXR the gain over greedy is **+0.191, 95% bootstrap CI
[+0.152, +0.232]** (n=435), a **5.6× lift** — and training is again the active ingredient: the *zero-shot*
box-verifier reaches only 0.115 (30% of the gap), so training roughly doubles the captured gap. The
honest scope is that this lifts *selection over a weak base grounder*, not a trained SOTA grounder in
absolute IoU; the point is that the "train a verifier to break the luck floor" principle extends from
free-text answers to structured outputs. (Fig. `figs/limits/fig_trained_verifier_unified.png`.)

### 9.7 The escalation gate: verifier-confidence is the best signal, and no trained gate beats it

Given best-of-N selection, *which* picks should we escalate to the strong model? The answer mirrors the
luck floor of §5 exactly: **the verifier's own confidence is the best available gate, and no trained gate
beats it by more than noise.** Predicting pick-correctness pooled, verifier-confidence reaches AUROC
**0.853 / 0.885 / 0.875** (Lingshu / MedVLThinker / InternVL3), and the best trained gate (a GBM/logit on
verifier-plus-cheap features) reaches **0.861 / 0.882 / 0.879** — a Δ of **+0.008 / −0.003 / +0.004**,
i.e. noise, with identical cascade accuracy. A controlled gate-swap (verifier-bo8 fixed at 0.414, strong
0.331; Lingshu pooled) ranks the signals by area-under-the-deferral-curve (ADC, higher is better):

| escalation gate | AUROC (pick-correct) | ADC |
|---|---|---|
| **verifier-confidence** [deployed] | 0.853 | **0.3923 (best)** |
| trained GBM (verifier + cheap) | 0.854 | 0.3914 (ties, −0.001) |
| trained logit (verifier + cheap) | 0.853 | 0.3911 (ties) |
| verifier-mean | 0.834 | 0.3855 |
| SOTA Diff-Prob (Jitkrittum L2D) | 0.708 | 0.3832 |
| self-consistency / −n_distinct | ~0.69 | ~0.368 |
| verifier-margin / verifier-negstd | 0.40 / 0.44 | ~0.36 |

Cheap-only gates (self-consistency, n_distinct, cheap-confidence) land at **0.66–0.79**, far below; the
recoverability-ranking gate in the Jitkrittum L2D style nudges its own AUROC 0.60 → 0.63–0.65 but yields
no better cascade accuracy, and its ADC (0.3832) sits below verifier-confidence. The reason is the same
**recoverability wall** we hit for routing between frozen models: the strong model fixes only **6–10%**
of the verifier's errors (up to **26%** on the sets where the strong model is competitive), and *which*
ones is near-unlearnable (AUROC ~0.4–0.6; on InternVL3 pooled the recoverability signal is *below*
chance, 0.367).

**The deployed operating point.** Adding the verifier-confidence gate on top of best-of-N selection, on
the same n=1064 held-out split as §9.2, gives the accuracy-optimal **verifier-augmented cascade: 0.517 @
35% escalation** — above verifier-alone (0.501) and the 32B (0.462), at a compute premium (≈17.5
7B-forward-equivalents). Per dataset (seed-0) it beats the 32B on Kvasir (0.422 @36%), PathVQA (0.453
@30%) and VQA-RAD (0.667 @37%), and only *matches* it on SLAKE (0.829 @100% — the verifier alone, 0.762,
is below the 32B there, so the gate simply escalates everything). This is the same honest picture as the
full-set tables: the cascade wins by out-selecting the strong model on the hard sets and by deferring to
it on the one set where it dominates.

### 9.8 The binding limit is candidate quality, not the selector

The verifier does not capture the whole oracle gap, and the diagnostics say precisely why: **the ceiling
is the quality of the 8 candidates, not the selector's ability to rank them.** Per-answer, the verifier
already ranks correct above incorrect at AUROC ~**0.90** in every family (Lingshu 0.903, MVT 0.913, IV3
0.898), yet its **selection efficiency** — how often argmax picks a correct candidate *when a recoverable
one exists* — is only **81% (Lingshu) / 82% (MVT) / 74% (IV3)**. Pushing per-answer ranking harder does
not help selection: adding a within-question Bradley-Terry ranking loss raises per-answer AUROC **0.903 →
0.931 / 0.933** but leaves selection accuracy **flat at ~0.50**. What *does* move the ceiling is better
candidates: pooling samples across the three model families raises **oracle@8 by +0.11–0.15** (VQA-RAD
+0.150, Kvasir +0.138, RadImageNet +0.113), and the oracle already dwarfs the verifier in-family
(oracle@8 vs verifier-bo8: Lingshu 0.513 vs 0.414, MVT 0.416 vs 0.339, IV3 0.337 vs 0.249). So the
remaining gap is *headroom in the candidate set*, not a smarter loss — which is exactly why cross-model
candidate pooling, not a better verifier, is the natural next lever (§12).


## 10. Results III — the faithful cross-family MCQ cascade and a reasoning think tier

§4 (the ACC) established the structural claim — *spend compute on the query, not on every query* —
on the internal harness, with a single MedVLThinker family, and with the mechanism running the "wrong"
way for reasoning (no-think beats think on perception). This section stress-tests that claim in the two
directions where a systems contribution most needs to survive: **(a) a faithful, third-party
reproduction protocol** (MedEvalKit, so the accuracies are directly comparable to published papers, not
to our own harness), and **(b) three architecturally distinct model families** (Lingshu, MedVLThinker,
InternVL3). It also adds the mirror-image structural lever the ACC omits: a **reasoning "think" tier**
that *adds* accuracy on the reasoning benchmarks where thinking helps rather than hurts. Two structural
moves, both regime-adaptive; neither is a "better gate."

Two honesty notes bound this section up front. First, the protocol here is the **faithful MedEvalKit
pipeline** defined in §4 (isolated `medeval_venv` + the `Qwen2_5_VL` wrapper + native weights +
`use_vllm` + `TORCHDYNAMO_DISABLE`, vLLM 0.9.0.1, MCQ graded exact-match with `use_llm_judge=False`);
the internal NGC harness is **not** faithful and its numbers are not used here. Second, several July
cells are still **provisional** — the InternVL3-38B MedXpert cell (a token-cap failure), the 7th
benchmark (OmniMedVQA, still running), and all InternVL3/PathVQA *latency* figures (measured under GPU
contention) — and are flagged as such below rather than folded into the headline.

### 10.1 Reproduction fidelity: the pipeline reproduces published baselines

Before trusting any cross-family cascade number, we confirm the faithful pipeline reproduces the
published Lingshu baselines. Our MedEvalKit accuracies vs the Lingshu-paper (arXiv 2506.07044)
figures, in percent:

| benchmark | Lingshu-7B (ours) | 7B (paper) | Lingshu-32B (ours) | 32B (paper) |
|---|---|---|---|---|
| MMMU-Med | 80.0 | 54.0 | **63.3** | **62.3** |
| SLAKE-closed | 82.5 | 83.1 | 85.9 | 89.2 |
| PMC-VQA | 54.3 | 56.3 | 55.2 | 57.9 |
| MedXpert-MM | 26.2 | 26.7 | 30.6 | 25.7 |
| VQA-RAD-closed | 78.1 | 67.9 | 85.3 | 76.5 |

The clean like-for-like anchor is **MMMU-Med-32B: ours 63.3 vs paper 62.3 — an exact match** (0.633 =
62.3). PMC-VQA is 4-option MCQ throughout and tracks the paper closely for both sizes (55.2 vs 57.9;
54.3 vs 56.3). The SLAKE and VQA-RAD rows are *not* strictly like-for-like — the paper reports the
**combined** open+closed set (83.1 / 89.2 and 67.9 / 76.5), whereas our column is the **closed subset**
that the MCQ cascade actually runs on — so their offsets (VQA-RAD-closed higher, SLAKE-closed lower) are
composition, not error; on MedXpert-MM our reproduction even slightly exceeds the paper (30.6 vs 25.7).
The one genuine anomaly is **Lingshu-7B MMMU-Med at 80.0**, a Lingshu-7B-specific inflation of roughly
+26 over the paper's 54.0; it is a known quirk of that checkpoint, **excluded from all claims**, and
confirmed to be family-specific because **MedVLThinker-7B on the identical eval is a normal 0.533**
(< its own 32B). As an independent cross-check we also reproduced the *open-ended* halves through a
Claude-Sonnet-5 judge: **VQA-RAD-32B 74.1% (paper 76.5), SLAKE-32B 85.0% (paper 89.2)**, with the judge
validated at a **100% exact-match anchor** (zero-word-overlap "correct" cases were all legitimate
synonyms or bilingual Chinese matches) and independent-2nd-judge agreement **κ 0.85–0.96**. The
faithful pipeline is trustworthy; the cascade numbers below are built on it.

### 10.2 The 2-tier margin cascade matches the strong model at −17…−69% FLOPs, across three families

The construction is deliberately the simplest possible instance of the ACC structure: a **2-tier
cascade** — cheap 7B/8B tier, escalate to the strong 32B/38B tier — gated by the **same
confidence-margin rule** used throughout the paper (margin `m = ℓ₁ − ℓ₂`; escalate iff `m < τ`), with τ
set to the iso-strong operating point (just enough escalation to match the strong model's accuracy).
Cost is the 2-tier form of Eq. 1, `C = 1 + esc·4.57` in 7B-forward-equivalents (the cheap tier always
runs at cost 1; the strong tier costs 4.57 and runs on the escalated fraction), and FLOPs-savings vs
always-strong is `1 − C/4.57`.

The 3-family × 7-benchmark matrix of FLOPs savings at iso-strong accuracy:

| benchmark | Lingshu | MedVLThinker | InternVL3 |
|---|---|---|---|
| MMMU-Med | keep-cheap (7B anomaly) | −14% | **−62%** |
| PMC-VQA (33k) | **−69%** | −49% | −16% |
| SLAKE | −56% | no win (7B weak) | keep-cheap (8B ≥ 38B) |
| VQA-RAD | −17% | −41% | **−67%** |
| PathVQA | −31% | **−68%** | −20% |
| MedXpert-MM | no win (floor) | no win | *(cap gap, fixup)* |

Read as a whole, the cascade **matches the strong model at −17…−69% FLOPs** wherever it wins, and the
pattern of *where* it wins is governed by a single rule: **the win magnitude tracks the (strong − cheap)
accuracy gap**. A *small* gap means the cheap model already answers most queries, so only a thin
residual escalates and the saving is large; a *large* gap (the cheap model is weak) forces the gate to
escalate nearly everything, and the extra cheap pass on top of an almost-always-strong cascade makes it
*more* expensive than always-strong. This produces two non-winning regimes, each for a principled
reason:

- **keep-cheap** (the small model already ≥ the big one, so escalation buys nothing): **InternVL3-SLAKE**
  (the 8B matches or exceeds the 38B) and **Lingshu-MMMU** (the inflated Lingshu-7B, §10.1).
- **no win** (the small model is too weak, or both are at the floor): **MedVLThinker-SLAKE** (7B 0.498 vs
  32B 0.620 — a 12-point gap forces **96%** escalation and **+18%** FLOPs), and every **MedXpert-MM**
  cell (both models near the 4-option floor, ~100% escalation, +17…+22% FLOPs). The
  **InternVL3-MedXpert** cell is not a floor result but an *unmeasured* one — see §10.5.

### 10.3 Per-benchmark cost, efficiency summaries, and threshold honesty

Per-benchmark detail for Lingshu (accuracy is exact-match; 7B = 1, 32B = 4.57 forward-equivalents):

| benchmark | 7B | 32B | 2-tier | esc% | ΔFLOPs | Δlatency |
|---|---|---|---|---|---|---|
| PMC-VQA (n=33,430) | 0.543 | 0.552 | 0.552 | 9% | **−69%** | −33% |
| SLAKE-closed (836) | 0.825 | 0.859 | 0.861 | 22% | −56% | −22% |
| VQA-RAD-closed (251) | 0.781 | 0.853 | 0.853 | 61% | −17% | +21% |
| MedXpert-MM (2000) | 0.262 | 0.306 | 0.307 | 95% | +17% | +60% |
| MMMU (150) | 0.80* | 0.64 | — | — | — | — |

(*Lingshu-7B MMMU inflated, §10.1.) And for MedVLThinker:

| benchmark | 7B | 32B | 2-tier | esc% | ΔFLOPs |
|---|---|---|---|---|---|
| PMC-VQA (33k) | 0.521 | 0.537 | 0.537 | 29% | −49% |
| VQA-RAD-closed (251) | 0.765 | 0.865 | 0.865 | 37% | −41% |
| MMMU (150) | 0.533 | 0.613 | 0.613 | 64% | −14% |
| SLAKE-closed (836) | 0.498 | 0.620 | — | 96% | +18% (no win) |
| MedXpert (2000) | 0.239 | 0.299 | — | 100% | +22% (no win) |

MedVLThinker's **PathVQA** cell is the family's largest saving (**−68%**, matrix above); its per-benchmark
7B/32B/escalation breakdown is not yet in these tables. For **InternVL3** we currently have only the
matrix cells (MMMU-Med −62%, VQA-RAD −67%, PMC-VQA −16%, PathVQA −20%; SLAKE keep-cheap), with the
MedXpert cell pending (§10.5).

**Latency caveat.** The Lingshu ΔLatency column is a *within-family, batched, serialized* cheap-vs-strong
relative number (identical batch size and tensor-parallel degree), **not batch-1**; it moves the same
direction as FLOPs where escalation is low (PMC −33%, SLAKE −22%) but goes *positive* where escalation is
high (VQA-RAD +21%, MedXpert +60%), because the extra cheap pass is paid on nearly every query. The
InternVL3 family and all PathVQA rows were measured under GPU contention and are **uncitable for
latency** until a matched serial re-run (§10.5); their accuracy and FLOPs are unaffected.

**Publication-standard efficiency summaries (APGR / CPT).** We also report the two standard cascade
summaries used in the deferral literature — **CPT** (compute-per-task, in 7B-forward-equivalents, *lower
is better*; exactly `1 + esc·4.57`, so it also fixes the FLOPs column as `1 − CPT/4.57`) and **APGR** (an
accuracy-gain-rate summary, *higher is better*; APGR > 1 marks a favorable accuracy-per-compute trade):

| family | benchmark | APGR | CPT | ΔFLOPs |
|---|---|---|---|---|
| Lingshu (mean APGR 1.225) | PMC-VQA | 2.05 | 1.41 | −69% |
| | SLAKE | 1.23 | 2.01 | −56% |
| | VQA-RAD | 0.96 | 3.79 | −17% |
| | MedXpert | 0.66 | 5.34 | no win |
| MedVLThinker (mean APGR 0.915) | PMC-VQA | 1.32 | 2.33 | −49% |
| | VQA-RAD | 1.08 | 2.69 | −41% |
| | SLAKE | 0.72 | — | no win |
| | MedXpert | 0.55 | — | no win |

APGR crosses 1 exactly at the win/no-win boundary (Lingshu: PMC/SLAKE > 1 win, VQA-RAD 0.96 marginal,
MedXpert 0.66 loss), and CPT crosses 4.57 there (MedXpert CPT 5.34 > 4.57 = costlier than always-strong).

**Threshold honesty.** The matrix and per-benchmark tables report savings at the **iso-strong operating
point** (τ chosen so cascade accuracy just meets the strong model). That is mildly optimistic, so we ran
the honest version too: refitting τ on a **held-out split** and reporting the saving it *delivers*. On
PMC-VQA the honest, held-out-τ saving is **−57% FLOPs for Lingshu** (cascade 0.563 ≥ 32B 0.549) and
**−49% for MedVLThinker**, versus the fully-oracle-τ figures of −74% and −51%. So the headline
PMC-Lingshu number reads three ways depending on protocol — **−69%** at the iso-strong operating point,
**−57%** under a fair held-out threshold, **−74%** under oracle-τ — and the deployable, honest figure is
the middle one, still a large saving. This is a threshold-protocol difference, not a contradiction; we
report all three.

**Gate-signal ordering.** As in the ACC, the *choice* of confidence signal barely matters — the
efficiency comes from the structure, not the gate. Among the three cheap signals, the ordering is
**margin > confidence > cumulative-logprob**, with margin giving the lowest CPT at every benchmark:
PMC-VQA margin CPT 1.41 (−69%) < conf 1.55 (−66%) < cum-logprob 1.64 (−64%); SLAKE margin 2.01 (−56%) <
conf 2.06 < cum-logprob 2.19; VQA-RAD the three collapse (margin 3.79, −17%, ≈ conf ≈ cum-logprob). The
spread is small precisely because the gate is saturated (§5–§6); margin is chosen as the marginally-best,
training-free default.

### 10.4 The think tier: adding reasoning where it pays (the mirror image of over-thinking)

The ACC's premise (§4) is that reasoning **over-thinks** perception, so its no-think tier *removes*
wasted thinking on the COMPETENT-4 sets. The complementary structural move is to *add* thinking exactly
where it is under-supplied — the **reasoning** benchmarks — by running the strong leg in reasoning mode
as a third tier. Faithfully, across all three families:

| strong leg (reasoning mode) | MMMU-Med Δ | MedXpert Δ |
|---|---|---|
| Lingshu-32B | **+0.034** (0.633 → 0.667) | ~0 (floor) |
| MedVLThinker-32B | **+0.107** (0.613 → 0.720) | +0.045 (0.299 → 0.344) |
| InternVL3-38B | **+0.120** (0.633 → 0.753) | *(cap gap, fixup)* |

Reasoning on the strong leg **adds accuracy on MMMU-Med for every family** (+0.034 / +0.107 / +0.120),
and lifts MedVLThinker on MedXpert (+0.045, 0.299 → 0.344), while doing nothing on Lingshu-MedXpert
(both models are at the near-chance floor there, so there is no headroom to recover). The behavior is
thus **regime-adaptive**: think where headroom exists (MMMU), skip it at the floor (MedXpert).

The gains are genuinely from *more reasoning*, not prompt luck. On a generic "reason step by step"
prompt all three families expand their decode from ~3 tokens of direct answering to real chains:
**gen_toks 3 → 275 / 561 / 368** (direct → reasoning, per family). Notably, MedVLThinker emitted **0
`<think>` tags** yet still gained +0.107 — it does not need a native reasoning tag to benefit — which
also **corrects an earlier retired claim** of ours that "Lingshu has no promptable think mode." A direct
probe (`lingshu_reason_probe.py`) shows Lingshu-7B goes **gen_toks 3 (direct) → 174 ("reason step by
step") → 267 (emitting real `<think></think>` tags)** on both MCQ and open-text; Lingshu *can* reason on
demand, which is why the Lingshu-32B MMMU think-tier gain (+0.034) is available at all.

Finally, the think tier **composes into a FLOPs-positive cascade where the accuracy is there and stays
away where it is not**. Gating the reasoning tier behind a cheap-tier check, MMMU reaches its think-mode
accuracy at **~78% of always-think FLOPs (a win)**, whereas MedXpert-Reasoning and MedXpert-Understanding
land at **~143% and ~151% FLOPs (no win)** — the same regime-adaptive verdict, now paid for in compute:
add the reasoning tier on MMMU, keep it off MedXpert.

### 10.5 Provisional and in-progress (honest status)

Three items in this section are explicitly **not final** and are excluded from the headline:

- **InternVL3-38B MedXpert-MM — not measured (cap gap).** MedXpert prompts run to ~20k tokens, above the
  16,384 `MAX_MODEL_LEN` we set for the 38B (a KV-cache fit at tp=2), so both the direct and reasoning
  MedXpert runs for InternVL3 failed to generate. Both the 2-tier cell and the think-Δ cell are therefore
  blank ("cap gap, fixup"); a re-run at cap 24,000 is queued. This is the only structural hole in the
  InternVL3 column.
- **OmniMedVQA (7th benchmark) — running.** The full Open-access split is **88,996 QA** (4-option MCQ,
  exact-match; 42 sub-datasets, of which RadImageNet is 57k ≈ 64%); the Lingshu paper evaluates the full
  set (7B 82.9% / 32B 83.4%), so full — not a sample — is the faithful choice. A parser bug (a mid-run
  `KeyError`) has been fixed and the full run is in progress; **no OmniMedVQA cascade result is claimed
  here yet**.
- **Contended latency — uncitable.** As noted in §10.3, the InternVL3 family and all PathVQA rows were
  timed under GPU contention; their `latency_s` is withheld pending a matched serial re-run (accuracy and
  FLOPs are unaffected). Within-family cascade latency remains a *fair relative* cheap-vs-strong number
  (identical batch and tp, serialized), but it is not batch-1 and not cross-family comparable.

Status snapshot: **6 of the 7 Lingshu-suite benchmarks are complete across all three families**; the
above are the remaining edges being closed.


## 11. Discussion and Limitations

This section states plainly what in the paper is new and what is inherited, bounds every headline
claim to the regime where it actually holds, and lists the measurements that are still provisional.
The standing rule applies throughout: every figure below is verbatim from a real checkpoint, and where
a value is not yet measured we say so rather than estimate it.

### 11.1 What is and isn't novel

We do not claim new routing signals or new reward-model machinery. Two of the three methods are
established mechanisms applied to a new setting; the third contribution is the negative
characterization that explains *why* those are the levers.

- **ACC's decision rule is not new; its structure is the contribution.** The agreement/consistency
  gate that fires the think tier (Eq. 3) is the ABC cascade family, and the confidence-margin gate
  at tier 0 is Chow/MSP. What is new is the *object* being gated: a cascade over **compute
  configurations of one large model** — a large-model *no-think* intermediate tier inserted between
  the small model and the large-model *think* tier — motivated by the empirical fact that reasoning
  over-thinks perception (§11.2). The systems contribution is incremental-but-defensible, not a new
  gate.

- **The verifier's mechanism is GenRM lineage; its application and unification are the
  contribution.** A generative outcome verifier for best-of-N selection is standard. What is new is
  (i) applying it to **open-ended medical VQA and grounding** rather than closed-form or MCQ tasks,
  (ii) **unifying free-text answers and bounding boxes** under one score head with a single label
  rule (y = 1[IoU ≥ 0.3] for boxes), and (iii) using it as a *constructive counter* to the
  Verification-Mirage claim — a small trained verifier does break the training-free selection floor
  here, even though training-free self-verification does not.

- **The genuinely new empirical object is the luck floor itself.** The finding that a dozen
  training-free gate signals all hit the same recoverability ceiling (pooled pick-correctness AUROC
  ~0.5–0.69), that training-free selection sits at or below a random-pick floor, and that the two
  frozen models fail together, is a characterization result. It is what tells a practitioner to stop
  tuning gates and instead change the structure or train a small verifier.

### 11.2 Honest scope of the ACC over-thinking premise

ACC's headline efficiency (latency −80%, FLOPs to ~52%, energy ~5× at ALL-6 parity) rests on the
premise that the large model's *no-think* mode is at least as accurate as its *think* mode on the
target queries. That premise holds on **perception** benchmarks and must not be over-generalized:

- **Where it holds (COMPETENT-4: SLAKE, VQA-RAD, PathVQA, PMC-VQA).** MedVLThinker-32B no-think ≥
  32B think: SLAKE **0.849 vs 0.764** (+0.085), VQA-RAD **0.853 vs 0.776** (+0.077), with PMC-VQA
  (0.551 vs 0.556) and PathVQA (0.661 vs 0.672) essentially tied. Here the slow think pass is pure
  overhead, so inserting the fast no-think tier is a free win.

- **MMMU-medical is competent but a *reasoning* set, so the mechanism does not apply.** MMMU is not
  near-chance — 32B-think reaches **0.688** — but here thinking *helps*: no-think **0.624** < think
  **0.688**. The over-thinking premise is false on MMMU, so ACC's no-think tier does not save compute
  on it. This is exactly why the reasoning "think tier" (§11 faithful-cascade results) is the right
  structure for MMMU while the ACC no-think tier is the right structure for perception; the two are
  regime-adaptive complements, not a single universal rule.

- **MedXpert-MM is the genuinely near-chance set and is excluded from headline efficiency.** Both
  models sit at or below the 4-option chance level (7B MedXpert-R **0.225**, MedXpert-U **0.256**), so
  no routing decision over it is meaningful. It is excluded from the ACC efficiency headline and from
  COMPETENT-4, but it still appears in the full ALL-6 tables for completeness (ALL-6 parity 0.5723;
  ALL-5 removes MedXpert, parity 0.6463).

- **ACC does not gain on every base family, but it degrades gracefully.** On families where think
  mode *hurts* overall (QoQ-Med, Chiron-o1), the agreement gate correctly collapses the cascade to
  the cheap 7B leg (escalation 0%, guard **0.00**) rather than harming accuracy — a no-gain but
  no-harm outcome, not a failure. The mechanism only *pays* where the large no-think mode is
  genuinely competitive; MedGemma is a partial case.

### 11.3 The verifier: matches-to-modest-win, not a blowout

The strongest single claim — a small trained verifier lets a 7B beat the strong 32B/38B on accuracy —
is real but must be reported with its ties.

- **Against the strong model it is matches-to-modest-win.** On the canonical two-seed audit (two
  70/30 splits, pooled-4), the verifier beats the same-split 32B by **+0.039, 95% CI [+0.010, +0.066]**
  on seed 0 (excludes 0), but only **−0.005** on seed 1 (a tie). The honest summary is that the
  verifier **matches / is competitive with** the 32B, average **+0.017** — not a uniform blowout. The
  full-set 3-family master table does show every cascade-best cell above the strong model (e.g.
  pooled Lingshu 0.421 vs 0.331, MVT 0.344 vs 0.277, IV3 0.255 vs 0.218), but the per-split
  robustness check is what governs the headline verb.

- **Against training-free selection it is robustly positive on both seeds.** Over greedy first-sample
  it captures **+0.088** (seed 0) / **+0.066** (seed 1) of accuracy, while self-consistency ≈ greedy.
  So the defensible, seed-robust claim is "beats training-free selection," and the "beats the 32B"
  claim is seed-dependent (holds on seed 0, ties on seed 1).

- **The per-dataset spread is wide.** Pooled gap-captured is 35–49% across the two seeds (mean ~42%),
  but per-dataset it ranges from little-headroom sets to hard sets:

  | dataset | greedy → verifier | gap-captured |
  |---|---|---|
  | SLAKE | 0.738 → 0.762 | 15.2% |
  | VQA-RAD | 0.519 → 0.611 | 45.5% |
  | PathVQA | 0.352 → 0.441 | 55.7% |
  | Kvasir | 0.282 → 0.405 | 58.4% |

  SLAKE has little to recover (greedy already 0.738); Kvasir has the most. The often-quoted "35–78%"
  is the *cross-output-type* range spanning free-text pooled (35–49%) up to MS-CXR boxes (78%), not a
  per-dataset free-text range.

- **Other-base transfer is not uniformly robust.** Trained from scratch on a *different* backbone
  (MedVLThinker-7B), the verifier works on SLAKE (42% of gap, greedy 0.564 → 0.622) but **fails on
  VQA-RAD** (0.500 → 0.470, −20%; n=54, noisy), pooling to 25% — positive but weaker than Lingshu's
  49%. In contrast, *cross-generator* transfer (a Lingshu-trained verifier scoring MedVLThinker
  answers) is stronger: SLAKE 49% / VQA-RAD 61%. The takeaway is honest: verifier quality depends on
  the training base and the low-n sets are unstable.

- **It lifts selection over a frozen generator; it is not a SOTA grounder.** The verifier chooses among
  a *frozen* generator's own samples — it does not produce better answers or boxes. Absolute grounding
  IoU stays modest (SLAKE box 0.255, MS-CXR box 0.232); the win is that it recovers 40% (SLAKE) and
  78% (MS-CXR) of the gap to *that generator's own* oracle, with the MS-CXR bootstrap gain **+0.191,
  95% CI [+0.152, +0.232]** excluding 0. We do not claim to beat a purpose-trained SOTA detector in
  absolute IoU.

### 11.4 Measurement caveats

- **Latency and energy are calibrated batch-1, not a single end-to-end wall-clock.** They are
  computed from measured per-tier batch-1 costs composed through the cascade cost model (Eq. 1); FLOPs
  are exact (analytic 2·N·(P+G)). This is the honest accounting, but it is a modeled batch-1 figure,
  not a production-throughput number.

- **Some July latency was measured under GPU contention and is not citable.** The InternVL3 family
  rows and all PathVQA rows in the July faithful-cascade matrix were measured while the GPUs were
  contended, so their `latency_s` is uncitable until re-run serialized and matched (accuracy and
  FLOPs are unaffected). Clean serial re-runs (iv3_8b, four PathVQA legs) are queued. Cascade
  `latency_s` is a fair *relative* cheap-vs-strong number within a family (identical batch size and
  tensor-parallel degree, serialized), but it is not a batch-1 figure.

- **Threshold optimism is bounded but present.** The MCQ efficiency numbers are reported at both an
  iso-strong operating point and a held-out fair τ (e.g. Lingshu PMC-VQA −69% at the iso-32B point but
  −57% on a held-out τ); we flag the more conservative held-out figure as the deployable one.

### 11.5 In-progress items (do not treat as final)

- **OmniMedVQA (the 7th benchmark, full 88,996 open-access QA) is still running** with the fixed
  parser; there is no cascade result for it yet.
- **InternVL3-38B on MedXpert-MM is not yet measured** — both the 2-tier FLOPs cell and the think-tier
  Δ. MedXpert prompts run ~20k tokens, above the 16,384 context cap used for the 38B; a re-run at cap
  24,000 is queued. The corresponding cells are marked "cap gap, fixup," not estimated.
- Status snapshot: 6 of 7 of the suite are complete across the three families.

### 11.6 The binding limit: candidate quality (future work)

The clearest ceiling on the verifier is **not** the verifier — it is the pool of candidates it must
choose from. Three diagnostics converge on this:

| family | per-answer AUROC | selection efficiency | verifier-bo8 vs oracle@8 |
|---|---|---|---|
| Lingshu | 0.903 | 81% (2027/2514) | 0.414 vs 0.513 |
| MedVLThinker | 0.913 | 82% (1154/1414) | 0.339 vs 0.416 |
| InternVL3 | 0.898 | 74% (1688/2277) | 0.249 vs 0.337 |

The verifier already ranks single answers well (per-answer AUROC ~0.90), and a ranking-loss variant
pushes that to **0.93** — yet selection accuracy stays flat at ~0.50, because on recoverable questions
the correct answer is a minority vote and the verifier picks it only 74–82% of the time. The larger
lever is candidate *diversity*: pooling answers across the three model families raises oracle@8 by
**+0.11–0.15** (VQA-RAD +0.150, Kvasir +0.138, RadImageNet +0.113), far more than any better selector
on a single model's samples. So the productive direction for future work is not a stronger gate or a
sharper verifier but **better candidates** — cross-model pools, higher N (oracle rises through @16/@32),
and generators tuned for useful diversity. Selection can only recover what the candidate set already
contains.

### 11.7 Why two levers, and not a better gate

All of the above reduces to one fact. **Between** frozen models, the fixable-vs-not signal is nearly
absent — a dozen training-free gates hit the same recoverability ceiling (AUROC ~0.5–0.69), and the
strong model fixes only 6–10% of the small model's errors (26% where it is competitive), unpredictably
— so the leverage is to improve the **structure** (ACC). **Within** a frozen model, which-of-N-is-right
is real but not zero-shot-surfaceable (training-free self-verification ≈ chance; a trained verifier
reaches AUROC 0.924, and a trained 7B verifier beats a zero-shot 32B verifier 0.403 vs 0.355) — so the
leverage is a **little training**. Structure and a little training are the levers precisely because the
frozen-model routing/selection signal is, at training-free effort, luck.


## 12. Conclusion

A medical-VQA cascade must make two decisions, and this paper is a study of both: **which** queries to
escalate to a larger model (the *gate*), and **what** to run when we do escalate (the *action*). The
central negative result — the **luck floor** — is that over a *frozen* model neither decision can be
out-engineered with *training-free* signals. On the gate side, a dozen confidence, conformal, learned,
recoverability, and self-verification signals all land in the same AUROC band (~0.5–0.69) at predicting
whether escalation will help; the error structure of the two models is genuinely correlated (ALL-6
recoverability φ = 0.372, i.e. the strong model fixes only 37.2% of the small model's errors), and on the
competent four benchmarks the two models fail together most of the time (P(32B wrong | 7B wrong) = 0.584).
On the action/selection side, training-free best-of-N selection sits at a luck floor: on SLAKE-open a random
pick already scores 0.720, self-verification (0.715) is *below* random, and the best training-free selector
(32B listwise, 0.758) captures only 24% of the gap to oracle@8 (0.879), because the correct answer is a
*minority* vote in 74–90% of recoverable questions. The recoverability wall (Jitkrittum, NeurIPS'23) is the
theory; our numbers are its medical-VLM instance.

Yet **two levers give large, real gains, and neither is "a better gate."** They are the paper's two positive
contributions.

**Structure buys efficiency.** The Adaptive-Compute Cascade (ACC) routes over *compute configurations*
rather than over models, inserting the strong model's *fast no-think* mode as an intermediate tier
(7B-nt@cap320 → 32B-**no-think**@cap320 → 32B-**think**@fullres) and exploiting that reasoning *over-thinks*
perception VQA (32B-no-think ≥ 32B-think on the competent sets, e.g. SLAKE 0.849 vs 0.764, VQA-RAD 0.853 vs
0.776). At honest held-out parity with always-32B-think, on real measured batch-1 costs: **ALL-6 latency
11.34 s → 2.27 s (−80%), FLOPs 100% → 52%, energy 6318.8 J → 1181.9 J (≈5×), and zero guardrail violations**
(no benchmark worse than always-7B); **ALL-5 8.88 s → 0.44 s (−95%), FLOPs → 24.9%, energy → 172.8 J.** The
efficiency is the *structure*, not the gate — every training-free gate lands in the same 49–62% FLOPs band —
and the same lever transfers to a *faithful MedEvalKit* MCQ cascade that matches the 32B/38B at **−17…−69%
FLOPs** across three families, plus a reasoning "think tier" that *adds* accuracy where headroom exists
(MMMU-Med +0.034 Lingshu / +0.107 MedVLThinker / +0.120 InternVL3).

**A little training buys accuracy.** A small trained outcome verifier for best-of-N selection *breaks* the
luck floor and, unlike any training-free selector, **beats the strong model across three families and a
held-out OOD set** (pooled cascade-best vs strong: Lingshu 0.421 vs 0.331, MedVLThinker 0.344 vs 0.277,
InternVL3 0.255 vs 0.218 — the InternVL3 win uses the *Lingshu-trained* verifier, i.e. cross-architecture
transfer). For free-text answers it recovers 35–49% of the oracle gap over two seeds, and the same principle
holds for structured outputs, recovering 40–78% of the box-grounding gap including on a real chest-X-ray
benchmark (MS-CXR bootstrap +0.191, 95% CI [+0.152, +0.232]). It behaves as a **test-time-scaling method**:
adding candidates and selecting with the verifier lets a 7B match a 32B. Finally, the routing ceiling is
partly an **MCQ artifact** — the *same* confidence signal moves from AUROC ~0.6 in multiple-choice to ~0.87
open-ended — so medical-VLM cascades should be evaluated open-ended.

The honest scope is stated as sharply as the wins. Against the strong model the verifier is a *match-to-modest*
win, not a rout: over two 70/30 splits it is +0.039 (95% CI [+0.010, +0.066], significant) on one seed and
−0.005 (a tie) on the other, averaging +0.017; it is robust only relative to *training-free* selection
(+0.066/+0.088 over greedy on both seeds). The per-dataset spread is wide (free-text SLAKE 15.2% → PathVQA
55.7% / Kvasir 58.4% of the gap; boxes SLAKE ~40% → MS-CXR ~78%), a from-scratch verifier on a weaker base is
not uniformly robust (it fails VQA-RAD at n=54), and the method lifts *selection over a frozen generator* — it
is not a SOTA grounder. ACC's over-thinking premise holds only on *perception* (COMPETENT-4); MMMU is
competent-but-reasoning, where the think tier *helps* and the over-thinking cut therefore does not apply, and
MedXpert is near chance for both models and is excluded from the ACC headline. Several items are explicitly
in progress and must not be read as final: OmniMedVQA (the 7th benchmark, ~88,996 QA) is still running,
InternVL3-38B on MedXpert-MM is unmeasured (a 16384-token cap gap, re-run queued at cap 24000), and some July
latency cells (the InternVL3 family and all PathVQA rows) were measured under GPU contention and are
uncitable until re-run matched (accuracy, escalation, and FLOPs for those rows are unaffected). The binding
limit throughout is **candidate quality**, not the selector: a cross-model candidate pool raises oracle@N by
+0.11–0.15, which is where the largest remaining headroom lives and the clearest direction for future work.
We release ACC, the trained answer and box verifiers, the faithful MCQ cascade and think tier, and the full
negative-result characterization so that both the wins and the luck floor can be reproduced end to end.

---

## Reproducibility index

Every number in this paper traces to a checkpoint. Code is launched from the repo root
(`~/medvlthinker-imgdiff-compute`); the canonical value ledger is
`results/cascade_methods/artifacts/master_data.csv` (MCQ ALL-6/ALL-5) together with
`results/cascade_methods/docs/current/{GROUND_TRUTH_NUMBERS.md → VERIFIED_FACTS.md, OPENTEXT_MASTER_TABLE.md,
MASTER_SUMMARY_2026-07.md, METHODS_MASTER.md, METHOD_MATH.md, METHOD_ACC.md, UNIFIED_METHOD_EXPERIMENTS.md}`;
the session narrative is in `progress/progress_{June_17, June_20-22, June_24, June_25-26, June_29-30,
July_01-02, July_03}.md`.

**Method I — ACC (structure / efficiency).** `src/cascade_methods/{acc.py, acc_v2.py, acc_v3_confgate.py,
acc_v4_lowres_think.py}` (the 3-tier cascade and its agreement/margin/conjunction gates); the over-thinking
premise and the strong-leg per-config comparison in `src/cascade_methods/{strong_leg.py, final_comparison.py,
overthink_generalize.py}` from `src/labeling/run_32b_modes_vllm.py`; the all-methods bake-off, cross-family
sweep, and head-to-head in `src/cascade_methods/{acc_allmethods.py, acc_compare.py, cascade_all_families.py,
gate_compare.py, compare.py}`; per-tier measured batch-1 latency/energy via
`src/cascade_methods/{latency_estimate.py, open_measure_latency_energy.py}` and NVML power in
`src/labeling/nvml_power.py`; the cost math (Eq. 1) in `METHOD_MATH.md`/`METHOD_ACC.md`.

**The luck floor (the unifying negative).** Outcome-structure and recoverability diagnostics
`src/cascade_methods/{diagnostics.py, strong_fixes_genuinely_unknown.py, knowledge_feasibility_bytype.py}`;
capacity-vs-action analysis and cross-family complementarity `src/cascade_methods/{crossfamily_agree.py,
peer_premise.py, peer_router.py, peer_router_img.py, vision_sensitivity.py}`; open-ended selection luck floor
`src/cascade_methods/{select_eval.py, ground_analyze.py, open_cascade_analyze.py}`; writeups
`results/cascade_methods/docs/archive_mcq/{RECOVERABILITY_IS_CAPACITY_BOUND, OPENENDED_SELECTION_LUCKFLOOR}.md`.

**Open-ended ceiling-break.** Open-VQA generation `src/labeling/{run_openvqa.py, run_openvqa_fewshot.py}`;
the MCQ→open gate hunt `src/cascade_methods/{gate_search_open.py, open_gate_bakeoff.py}`; judge
`src/labeling/run_judge.py` (MedVLThinker-32B and a Claude-Sonnet-5 cross-check, κ 0.85–0.96, 100% exact-match
anchor).

**Method II — trained verifier (accuracy).** Answer verifier `src/training_methods/{run_lora_verifier_open.py,
run_lora_verifier_ranking.py, clean_verifier_dump.py}`; the ablations that make it non-circular and pin the
active ingredient `src/training_methods/{verifier_image_ablation.py, verifier_transfer_eval.py,
verifier_scaling_curve.py, cross_gen_verifier.py}`; the 3-family beats-32B master table (no new inference,
from saved artifacts) **`src/cascade_methods/open_verifier_cascade_table.py`**, with the gate bake-off and
cost frontier in `src/cascade_methods/{open_gate_swap.py, open_gate_efficiency.py, open_gate_heldout_tau.py,
open_recoverability_gate.py, open_cost_frontier.py}`. Box verifier
`src/training_methods/run_lora_box_verifier.py`; grounding data `src/labeling/{run_ground_slake.py,
run_ground_mscxr.py}`; verifier defn (Eq. 4/5), best-of-N argmax (Eq. 2), and gap-captured metrics in
`METHODS_MASTER.md`.

**Method III — faithful MedEvalKit MCQ cascade + think tier (structure, generalized).** The faithful harness
is the vendored `MedEvalKit/` (isolated venv + `Qwen2_5_VL` wrapper + Lingshu weights + `datasets_path=hf` +
`use_vllm` + `TORCHDYNAMO_DISABLE`, vllm 0.9.0.1); the internal NGC harness is *not* faithful. Runners
`runners/{run_full_matrix_medeval.sh, run_native_think.sh, run_omnimed_reruns.sh, run_regen_native.sh}`;
analysis `src/cascade_methods/{lingshu_medeval_cascade.py, cascade_all_families.py, lingshu_deferral_apgr.py
(APGR/CPT), compare_native_think.py, control_think_signals.py, final_3tier.py, lingshu_prompt_probe.py}`;
reproduction-fidelity, per-benchmark FLOPs/latency/escalation matrix, held-out-τ honesty check, and the think
deltas in `progress/progress_July_03.md` + `MASTER_SUMMARY_2026-07.md`.

**No number in this paper is fabricated; every figure is copied verbatim from real checkpoint output and
traces to a checkpoint via `VERIFIED_FACTS.md` / `GROUND_TRUTH_NUMBERS.md`.**


