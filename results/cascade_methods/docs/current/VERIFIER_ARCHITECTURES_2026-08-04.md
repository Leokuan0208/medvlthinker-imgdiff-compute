# Verifier architectures for best-of-8 selection — round synthesis (2026-08-04)

**Question this round asked.** Every verifier tried in this project so far has been a *generative*
language model producing an opinion (prompted, or LoRA-tuned to emit `Yes`/`No`). Sixteen distinct
attempts to improve best-of-8 selection had failed. This round tests whether the *computation*, not
the source, was the problem: discriminative heads, learning-to-rank objectives, feature-based
classifiers, and contrastive image-text alignment scorers — four families that are not "another
model's opinion."

**Answer, in one line.** Three of the four families produced no deployable win and are now closed.
The fourth — a **discriminative head on the frozen generator's own hidden states** — produced the
first result in this programme that clears the pre-registered detection floor *and* survives its own
falsification control: **sel_eff 0.7752 → 0.8065, Δ +0.0313 [+0.0163, +0.0463], guardrail-clean**,
as a parameter-free rank fusion of the incumbent with the head. It is a real improvement, it is
modest, it is one arm among ~20, and the confirmation experiments are named in §8.

Nothing in this round abstains, defers, or rejects. Every configuration always returns one of the 8
candidates.

---

## 1. Shared ground truth: baseline, harness, disjointness, detection floor

All four arms re-derived the baseline independently from
`ckpts/train/lora_verifier_disjoint/transfer_dump_*.json` before running anything, and **all four
reproduced every published cell exactly** (max abs deviation 0.0 for three arms; |Δ| < 1e-4 for the
alignment arm's AUROC at published precision).

| quantity | value |
|---|---:|
| eval items | 2,345 (slake_open 645 / vqa_rad_open 200 / pathvqa_open 1,500) |
| distinct eval images | 528 |
| oracle@8 | 0.626013 |
| greedy (1 sample) | 0.449467 |
| incumbent selected accuracy | 0.485288 |
| **incumbent selection efficiency** | **0.775204** |
| incumbent candidate AUROC | 0.885592 |
| incumbent per-set sel_eff | 0.8501 / 0.7619 / 0.7226 |
| random pick | 0.6763–0.6826 (arms differ on tie handling) |
| self-consistency (plurality) | 0.7139–0.7146 |
| generator zero-shot P(Yes), no adapter | 0.7071 (published, vLLM) / 0.7058 (HF, this round) |

**Convention that matters and is now settled.** Tie-breaking must be first-index `argmax` over the
stored `scores`, not the stored `pick` field. They disagree on 26/2345 items (5-dp rounding ties) and
the stored `pick` yields 0.774523. Two arms found this independently. `argmax` is what reproduces the
published cells and is used throughout.

**Disjointness was re-proven, not trusted.** Every arm asserted it in code on the md5 of *decoded RGB
pixels*: features arm 528 eval vs 5,229 train images → 0 intersection, 0 shared items; hidden arm
528 vs 3,457 → 0; alignment arm 528 vs 3,181 → 0. Builds fail otherwise.

**Detection floor (pre-registered).** With n=2345 and a plausible ~10% discordance rate, the paired
SE on accuracy is ≈0.0065, i.e. ≈0.0104 in sel_eff. **A win must be ≥ +0.021 sel_eff to clear 2σ.**
This was fixed before results were read and is applied uniformly below.

**The sensitive endpoint, derived exactly.** Of the 1,468 recoverable items (the sel_eff denominator),
**616 are all-correct lists** — every selector scores them by construction — and **852 are
label-mixed**; the remaining 877 items are all-wrong and are outside the denominator. So

> sel_eff = (616 + 852 · eff_mixed) / 1468

is an identity, and the only winnable stratum is the 852. The incumbent's efficiency there is
**0.6127**. Clearing +0.021 pooled means flipping **31 of 852 net**; the mixed stratum is ≈1.7× more
sensitive than pooled sel_eff and every arm below is also reported on it (derived through the
identity above — the derivation returns exact integer counts for every arm, which is itself a
consistency check).

---

## 2. What was tested, and why each is a genuinely different computation

| arm | computation | why it is not "another opinion" |
|---|---|---|
| **A. Learning-to-rank objectives** (T1.B) | Same score function, different loss: ListNet softmax-CE, "anypos" (−log Σ_correct softmax — the exact sel_eff surrogate), RankNet/Bradley–Terry over all within-question pos×neg pairs | The incumbent is trained pointwise and used argmax-within-question. This tests the *objective mismatch* hypothesis directly, with the loss as the only variable. |
| **B. Feature-based discriminative selector** | 39 hand-built features (pool duplicate structure, similarity structure, answer shape, question relation, train-derived answer prior) → GBM / MLP. **Zero model calls at inference.** | No language model runs at all. Nothing is prompted, nothing generates. Correctness is predicted from the *shape of the candidate pool*. |
| **C. Discriminative head on frozen generator hidden states** | Linear / 1-hidden-layer head reading `h_last` and `h_span` at layers 7/14/21/28 of the **frozen, unadapted** Lingshu-7B, under two prompt forms: the incumbent's *grader* prompt verbatim, and the model's own *answering* prompt with the candidate as the assistant turn. Score = head(h), **not** P("Yes"). | Same network, same prompt, same training draw as the incumbent — the *only* variable is the readout: a trained head on frozen features vs a LoRA-tuned LM head. This is the cleanest possible test of "is the information there but the readout wrong?" |
| **D. Contrastive image-text alignment** | Zero-shot cosine in a frozen dual-encoder space (SigLIP-so400m, PubMedCLIP, BiomedCLIP), raw and PMI-normalised; plus a trained MLP head on frozen `[v, t, v⊙t, \|v−t\|]` features | A geometric similarity, not a token distribution. The only family in the programme whose score never passes through an autoregressive head. |

Arm C's feature extraction was a fresh GPU pass — **80,882 forward passes** (40,441 rows × 2 prompt
forms), 0 failures. The pre-existing `feats_full/feat_*_L14.npz` caches are **unusable** for this
problem: they are per-*question* (416 rows for SLAKE, keyed by the closed-MCQ index), not
per-candidate, and come from MedVLThinker-7B, not the Lingshu-7B that generated these pools. That
blocker is now documented rather than assumed.

---

## 3. Results table

Selection efficiency, n=2345 (1,468 recoverable), paired bootstrap over items, nboot=10,000, seed 0.
"Mixed" = derived efficiency on the 852 winnable items via the §1 identity. Guardrail = never worse
on any of the three sets.

| # | selector | sel_eff | Δ vs incumbent [95% CI] | mixed (852) | guardrail | deployable? | verdict |
|---|---|---:|---|---:|:--:|:--:|---|
| — | **incumbent LoRA verifier** (Lingshu-7B + pooled disjoint LoRA) | 0.7752 | — | 0.6127 | — | yes | baseline |
| — | oracle@8 | 1.0000 | — | 1.0000 | — | — | ceiling |
| — | self-consistency (plurality) | 0.7139 | −0.0613 | 0.5082 | — | yes | control |
| — | generator zero-shot P(Yes) | 0.7058 | −0.0695 [−0.0906, −0.0484] | — | no | yes | control |
| — | random pick | 0.6763 | −0.0989 | 0.4531 | — | — | control |
| **C1** | **incumbent ⊕ generator-prompt head (rank avg)** | **0.8065** | **+0.0313 [+0.0163, +0.0463]** | **0.6667** | **clean** | **yes** | **WIN — the round's only one** |
| C2 | incumbent ⊕ per-benchmark grader head | 0.8038 | +0.0286 [+0.0129, +0.0436] | 0.6620 | clean | yes | corroborating |
| C3 | incumbent ⊕ CV-selected grader head | 0.7990 | +0.0238 [+0.0082, +0.0402] | 0.6538 | clean | yes | corroborating |
| C4 | generator-prompt head **alone** (CV-selected) | 0.7956 | +0.0204 [−0.0014, +0.0416] | 0.6479 | clean | yes | near-miss, n.s. |
| C5 | grader-prompt head, per-benchmark | 0.7759 | +0.0007 [−0.0218, +0.0232] | 0.6139 | no | yes | exact tie with a full LoRA |
| C6 | grader-prompt head, **CV-selected (pre-registered primary)** | 0.7507 | −0.0245 [−0.0470, −0.0020] | 0.5704 | no | yes | **LOSES, CI excludes 0** |
| D1 | trained SigLIP head (image+text) | 0.8038 | +0.0286 [+0.0068, +0.0511] | 0.6620 | **dirty** (0/5 seeds) | yes* | not a win (§5) |
| D2 | same head at the incumbent's 10,364-example budget | 0.7278 ± 0.0139 | −0.047 | — | — | yes | loses at matched supervision |
| D3 | trained SigLIP head, text-blinded | 0.7507 | −0.0245 | 0.5704 | — | — | ablation |
| D4 | zero-shot BiomedCLIP cosine | 0.6608 | −0.1144 | — | no | yes | **below random pick** |
| D5 | zero-shot PubMedCLIP cosine | 0.6417 | −0.1335 | — | no | yes | below random pick |
| D6 | zero-shot SigLIP cosine (raw) | 0.6247 | −0.1505 | — | no | yes | below random pick |
| D7 | zero-shot SigLIP, PMI-normalised | 0.6328 | −0.1424 | — | no | yes | PMI does not rescue it |
| B1 | 39-feature MLP, 10-seed ensemble, frozen off-eval | 0.7636 | −0.0116 [−0.0334, +0.0103] | 0.5927 | **dirty** (PathVQA −0.0400) | yes | statistical **tie**, zero inference cost |
| B2 | 39-feature MLP, seed 0 (top of the seed range) | 0.7704 | −0.0048 [−0.0266, +0.0174] | 0.6045 | — | yes | seed artefact, see §4 |
| B3 | 39-feature HGB (trees), matched capacity | 0.7078 | −0.0674 | — | — | yes | model class matters |
| A1 | best ranking arm (linear/FULL/anypos) | 0.7766 | +0.0014 [−0.0061, +0.0089] | 0.6150 | — | diagnostic† | no effect |
| A2 | best arm of any kind in the loss grid (gbm/FULL/**pointwise**) | 0.7813 | +0.0061 [−0.0102, +0.0232] | 0.6232 | clean | diagnostic† | n.s.; and it is pointwise |
| A3 | worst significant ranking contrast | — | −0.1117 [−0.1417, −0.0817] | — | — | diagnostic† | ranking losses actively hurt |
| **X1** | features + incumbent, **cross-fitted on eval** | 0.7970 | +0.0218 [+0.0040, +0.0401] | 0.6502 | clean | **NO** | diagnostic upper bound (§4) |
| X2 | features only, cross-fitted on eval | 0.7881 | +0.0129 [−0.0102, +0.0353] | 0.6350 | — | **NO** | measures the cross-fit optimism: +0.0177 |
| X3 | SigLIP head + incumbent, cross-fitted on eval | — | +0.0381 [+0.0225, +0.0531] | — | — | **NO** | diagnostic upper bound |
| **K1** | **incumbent ⊕ base zero-shot P(Yes)** [key control] | 0.7554 | **−0.0198 [−0.0354, −0.0048]** | 0.5786 | no | — | fusing a second *opinion* HURTS |
| K2 | incumbent ⊕ self-consistency count | 0.7554 | −0.0198 [−0.0368, −0.0027] | 0.5786 | no | — | hurts |
| K3 | incumbent ⊕ random | 0.7343 | −0.0409 [−0.0599, −0.0218] | 0.5423 | no | — | hurts |

\* D1 is off-eval-trained and therefore technically deployable, but it fails the project's
never-worse-on-any-set guardrail, so it does not qualify. † A-arms using the incumbent score as a
feature are cross-fitted diagnostics; the between-arm *contrast* is internally valid regardless, and
the no-incumbent-feature arms (which carry no caveat) are uniformly worse.

---

## 4. What won, what tied, what lost — and the diagnosis for each

### 4.1 WIN — discriminative head on the generator's hidden states, fused with the incumbent (C1)

**The result.** Rank-average of the incumbent's score with a CV-selected head on the generator's
*answering-prompt* hidden state: **0.8065 (+0.0313 [+0.0163, +0.0463])**, selected accuracy
0.4853 → 0.5049 (+0.0196 [+0.0098, +0.0294]), per-set 0.8836 / 0.8016 / 0.7510 — **all three above
the incumbent**. Mixed-stratum efficiency 0.6127 → 0.6667, i.e. **46 of the 852 winnable items
flipped net**, against the 31 the detection floor requires. Contested-stratum 0.6397 → 0.6900.

**Why this is not a fusion artefact — the control pattern is the evidence, not the CI.** The obvious
deflationary reading is "averaging any second score with the incumbent helps." It does not. Three
non-head second scores were fused identically: the base model's own zero-shot P(Yes) (a *generative*
opinion from the *same frozen network on the same prompt*), the self-consistency count, and random.
**All three make the incumbent significantly worse** (−0.0198, −0.0198, −0.0409; all CIs exclude
zero). Three different heads all make it better. The active ingredient is the **different
computation**, not the extra source. This is the first fusion in the project's history to survive its
own falsification control, and it partially overturns the standing "all fusions fail" entry — those
fusions all combined *generative* scorers.

It is also robust to the mechanics: rank-argsort fusion +0.0211, z-scored mean +0.0184, both still
excluding zero.

**What it is deployable on.** Both components are trained on strictly image-disjoint data; the head's
config was chosen by grouped CV *inside* the training pool before eval was touched; and **no
parameter is fitted at fusion time**. That is what separates C1 from X1/X3, which are fitted on eval.

**Honest deductions.**
- The **pre-registered primary arm loses.** The config CV chose on the grader prompt scored 0.7507,
  Δ −0.0245, CI excluding zero. The one-line answer to "does a discriminative head beat the incumbent
  standalone?" is **no**.
- C1 is **one arm among ~20** and its nominal CI is not multiplicity-corrected.
- **Cost is not free as measured.** The grader-prompt head needs an extra *base*-model forward pass
  (the incumbent's pass runs the *adapted* model), so that fusion costs ≈2× verifier prefill. The
  generator-prompt head's features are the states the model computes while decoding each candidate,
  so they *could* be cached during best-of-8 generation at near-zero extra cost — but they were
  measured in a separate teacher-forced pass, so **free-at-generation is an inference from the
  architecture, not a measurement.**

**Diagnosis of why it works where 16 things failed:** it is the only scorer that adds *no source*. It
reads the generator's own state with a different function. The information was already in the
network; the LM head was not the only way to get it out.

### 4.2 TIE — the feature-based selector (B1), and the head at parity (C5)

Two independent results say the same thing from opposite directions.

- A 39-feature MLP that makes **zero model calls**, trained in ~10 CPU-seconds, scores 0.7636
  (10-seed mean 0.7618, sd 0.0050) — Δ −0.0116 [−0.0334, +0.0103] vs a trained 7B judge. A
  statistical tie with the point estimate *below*. It beats the incumbent on candidate AUROC
  (0.8968 vs 0.8856) and beats self-consistency by +0.0565 [+0.0390, +0.0743].
- A near-linear head on **frozen** hidden states, per-benchmark, matches a full LoRA fine-tune to
  three decimals: 0.775886 vs 0.775204 sel_eff; 0.885550 vs 0.885592 AUROC.

**Diagnosis.** A generative judge is not required for this problem. Most of what the LoRA verifier
learned is recoverable either from pool structure alone or from a linear readout of the frozen
network. That collapses the *cost* side of best-of-N with no measured accuracy loss on SLAKE and
VQA-RAD — but not on PathVQA, where the feature model loses significantly (−0.0400 [−0.0730,
−0.0075]), so B1 is **not guardrail-clean and not a drop-in replacement**.

**What the feature model actually is.** Permutation importance on the *selection* endpoint:
`soft_vote` (similarity-weighted within-pool agreement) +0.0761, train-derived answer-string
correctness prior +0.0414, raw duplicate count +0.0181. Group-alone: no feature group reaches the
incumbent (best 0.7187). It is a **consensus-plus-prior machine with no image information at all**,
which is exactly why it wins on SLAKE (+0.0282) and loses on PathVQA (−0.0400). One immediately
actionable sub-result: `soft_vote` is worth 4× the exact-string duplicate count, which quantifies
what exact-string clustering leaves on the table.

**A seed-honesty note that should propagate.** The feature model's sel_eff sd across 10 seeds is
0.0050. Seed 0 (0.7704) is the **maximum** of 10. Any single-seed feature-model comparison below
≈+0.015 is noise.

### 4.3 LOSS — learning-to-rank objectives (A). The objective question is now closed at two levels.

15 objective contrasts holding features, folds, architecture, optimiser, epoch-selection criterion
and ensembling fixed. **None positive-significant.** Best +0.0082 [−0.0054, +0.0225]. **Five
significantly negative**, worst −0.1117. The best arm of any kind in the grid was *pointwise*.
On identical features in the feature arm: pointwise BCE 0.7704 > within-question BT 0.7459 >
listwise softmax 0.6907, and tuned listwise still lands at 0.6860 (−0.0892 [−0.1159, −0.0635]) with a
monotonically flat-to-worse held-out curve, so it is not under-fitted.

**The control that makes it airtight.** Listwise/pairwise losses are undefined on all-correct and
all-wrong lists, so they train on a subset. Adding `pointwise_bce_nondeg` — pointwise BCE on exactly
that subset — separates the two effects: restricting the training set alone costs −0.0020
[−0.0116, +0.0075]; swapping the loss on the *same* restricted set costs **−0.0817 [−0.1042,
−0.0586]**. The loss swap is doing the damage.

**Mechanism.** The listwise objective is not optimising the wrong thing — it optimises the right
thing *better* and generalises worse. With label leakage and unlimited capacity it beats pointwise
in-sample on identical features (0.956 ListNet / 0.971 anypos vs 0.918 pointwise); out of fold that
reverses (0.645 / 0.657 vs 0.729). It is the higher-variance estimator: **1,493 of 2,345 lists
(63.7%) are degenerate and carry zero gradient for it**, so it trains on 36.3% of the questions while
pointwise BCE uses all 18,760 candidate labels.

**And the ceiling is not a mis-ordering.** On the 330 mixed-list questions the incumbent gets wrong,
the best correct candidate sits at rank 2 only 22.1% of the time; the histogram is nearly flat to
rank 8 (mean rank 4.46, median score gap 0.132). Ranking losses repair near-ties; **these are not
near-ties.** Judge-label noise is ruled out: **0 of 2,345** questions contain the same candidate
string with two different judge labels (0 inconsistent pairs out of 30,326 duplicate pairs).

**Closes:** T1.B's remaining variants and plan candidates 3, 24 and the loss-half of 10 — permanently,
and now from two independent levels (LoRA-level N25, head/feature-level here). **Does not close**
set-aware *architectures* (Set-Encoder, pairwise distillation), where candidate *i*'s score is a
function of all 8. That is a different computation, not a different loss.

### 4.4 LOSS — zero-shot contrastive alignment (D). Branch closed, with the mechanism measured.

All three encoders are **at chance for correctness** (candidate AUROC 0.486 / 0.547 / 0.572) and
select **below random pick**. PMI normalisation does not rescue it. Medical pretraining does not
help: BiomedCLIP and PubMedCLIP are no better than general-domain SigLIP.

**The control that distinguishes relevance from correctness** — two paired tests on the *identical*
score function:

| | on-topic **correctness** (correct vs wrong candidate, same pool) | off-topic **relevance** (own candidate vs a foreign one) |
|---|---:|---:|
| SigLIP | 0.540 | 0.764 |
| PubMedCLIP | 0.527 | 0.651 |
| BiomedCLIP | 0.554 | 0.726 |
| incumbent (reference) | 0.756 | — |

So the scorer has real resolution for "is this text about this image" (up to +0.26 over chance) and
essentially none for "is this the *right answer*" (+0.04). Where it breaks is exactly where the repo
already localised the residual: on the laterality slice (n=342) zero-shot SigLIP scores 0.464 vs the
incumbent's 0.635; on negation (n=76) 0.368 vs 0.658 — **below chance on the slice, i.e. actively
anti-correlated with correctness** where a one-token contrast decides the item. Candidates in a pool
differ by a word or two, so their text embeddings are near-identical and the cosine to the image
varies by less than encoder noise. The contrastive objective never had to encode left/right or
present/absent.

**The caveated positive, and why it is not a win.** A trained MLP on frozen SigLIP features reaches
0.8038 (+0.0286 [+0.0068, +0.0511]), 5/5 seeds beating the incumbent. Three reasons it fails the bar:
(a) **guardrail-dirty** — loses on vqa_rad_open (−0.0397), 0/5 seeds clean; (b) **data-volume
confound, resolved against the method** — it sees 85,544 labelled examples to the incumbent's 10,364,
and the traced curve is 0.7037 (5k) / 0.7278 (10,364) / 0.7422 (20k) / 0.7781 (40k) / 0.8011 (85.5k),
so at matched supervision it **loses by −0.047**; (c) **shortcut exposure** — under a strict no-eval-
question-text split it collapses to 0.5974 while a size-matched control gets 0.7480, so ≈−0.15 of the
drop is a question-text→answer prior rather than data size.

What survives: the frozen features **do** carry usable visual evidence — permuting in a random image
drops the trained head 0.804 → 0.702 (AUROC 0.910 → 0.796), and a text-blinded copy only reaches
0.751. It is the **zero-shot cosine geometry**, not the features, that is the wrong readout.

**Closes:** T1.D and with it plan candidates 6, 11, 12, 14, 20, 21, and the BiomedCLIP / CONCH /
MedSigLIP download programme. **Does not close** a true cross-attention image-text-matching head
(T2.C / T3.A) — everything here is a readout over frozen *dual-encoder* features, and the fact that a
trained readout recovers +0.18 AUROC over the raw cosine on those same features is the argument for
trying cross-attention rather than more encoders.

---

## 5. Did the standing interpretation survive?

**The standing interpretation:** the failures share a shape — each added a *source* when what is
missing is per-item knowledge of *which source to believe*; pair-oracle headroom exists and no router
reaches it.

**Verdict: it survives, with one correction and one sharpening.**

*Survives.* The headroom is still there and still mostly uncashed. Pair-oracle ceilings measured this
round: incumbent × generator-prompt head **0.8774**; incumbent × grader-prompt head 0.8583; incumbent
× feature model 0.8672 (agreement on picks only 58.1%, each uniquely right on ~6% of items);
incumbent × SigLIP head 0.8842. The best deployable arm captures **≈28% of its pair-oracle headroom**
(0.7752 → 0.8065 against 0.8774). The remaining ≈0.071 still needs a per-item router, and no router
was built that reaches it. The wall is where it was.

*Correction — "adding a source always fails" is false as stated.* The failures were all *generative*
sources. A source that is a **different computation** on the *same* network does not collapse under
fusion: three heads all fuse positively, three non-head scores all fuse negatively (K1–K3). The
project's decorrelation law — "selection quality tracks agreement with the generator, ρ = +0.76" —
was confounded with "generative judge from a different family." Two independent counterexamples now
exist: the generator-prompt head has Spearman ρ 0.3671 with the incumbent (vs 0.7820 for the grader
prompt) yet is *better* than it; the feature model agrees on 58.1% of picks and ties it. **Restated
law: what fails is a second opinion; what helps is a second computation over the generator's own
evidence.**

*Sharpening — the generator holds the right information, and the readout was the bottleneck.* The
strongest evidence: on the model's own **answering** prompt, the hidden state is both a better and a
far more decorrelated correctness signal than on the **grading** prompt (in-domain matched probe
L21/span 0.7956 vs 0.7827; pair-oracle 0.8774 vs 0.8583). The state the model computes while
*producing* the answer knows more about whether the answer is right than the state it computes while
being *asked* whether the answer is right. That is a direct confirmation of the brief's hypothesis
and it is the one genuinely new mechanistic fact this round produced.

---

## 6. Cross-cutting mechanism findings (these change how the project should measure things)

1. **Candidate-level AUROC is not a valid proxy for selection efficiency.** The dissociation is
   bidirectional and large: a generator-prompt BT head has AUROC 0.678 but sel_eff 0.7956; its BCE
   sibling on identical features has AUROC 0.904 and sel_eff 0.8025; grader-prompt listwise 0.760
   AUROC / 0.7895 sel_eff vs grader-prompt BCE 0.872 / 0.7507 — i.e. **−0.112 AUROC bought +0.039
   selection**. In the other direction, `mlp_FULL_pointwise_bce` lifts AUROC 0.8856 → 0.9005 and moves
   sel_eff by exactly **+0.0000**. Selection is a within-question problem; global ordering is close to
   independent of it. This re-reads retrospective N25: "+0.030 AUROC, +0.000 selection" was not proof
   that ranking losses do nothing — AUROC was the wrong instrument. **Stop reporting AUROC as a
   verifier-quality endpoint.**
2. **Model class matters more than the objective.** Trees vs MLP on identical features at matched
   default capacity: HGB 0.7078 / AUROC 0.7135 vs MLP 0.7704 / AUROC 0.8971 — a 0.06 sel_eff gap from
   model class alone, larger than any objective effect measured. (The training mixture is 60%
   PathVQA; trees over-fit it, the heavily regularised near-linear MLP transfers.)
3. **Model selection does not transfer from the disjoint training pool to eval.** CV chose
   grader/bce/h256 (cv_sel_eff 0.6898), which scored 0.7507 on eval, while listwise (cv 0.6583) scored
   0.7895. The staged-grid gap was closed explicitly (listwise×h256 cv 0.6678, BT×h256 cv 0.6756) —
   CV would still have chosen pointwise. This is a genuine CV→eval transfer failure, not a grid
   omission. Likely cause: the composition-matched disjoint pool (PathVQA-heavy plus Kvasir /
   RadImageNet OOD) has different within-question group structure from the eval pools. **UNRESOLVED**,
   and it caps what any head-selection protocol on this split can deliver — it is precisely why the
   pre-registered arm lost while a post-hoc sibling tied.
4. **Cross-fit optimism is now measured, not guessed: +0.0177 sel_eff.** The same feature set fitted
   two ways (frozen off-eval 0.7704 vs cross-fitted on eval 0.7881) gives the size of the inflation
   directly. Applied to the feature+incumbent fusion, +0.0218 → **≈+0.0041**, below the detection
   floor. Every cross-fitted "gain" in this project should be discounted by this figure until the
   incumbent is scored off-eval.
5. **Stratum definitions are not harmonised across arms and should be.** "Contested" is defined as
   ≥2 distinct normalised candidates in three arms (n = 1,714 / 1,725 depending on normalisation) and
   as contested-within-recoverable in a fourth (n = 916). The **mixed-label stratum (n=852)** is
   unambiguous, exactly derivable from pooled sel_eff, and ≈1.7× more sensitive. **Adopt the 852 as
   the standard secondary endpoint.**

---

## 7. Is this a solid improvement? — direct answer

**Yes, one, and it is narrower than the headline number sounds.**

What the project can now claim, precisely:

> On the three open-ended benchmarks (n=2,345), fusing the deployed LoRA verifier's score with a
> discriminative head trained on the frozen generator's answering-prompt hidden states, by
> parameter-free rank averaging, raises best-of-8 selection efficiency from 0.7752 to **0.8065**
> (Δ +0.0313, 95% paired-bootstrap CI [+0.0163, +0.0463]) and selected accuracy from 0.4853 to
> **0.5049** (Δ +0.0196 [+0.0098, +0.0294]), improving all three benchmarks individually. Both
> components are trained on strictly image-disjoint data and no parameter is fitted at fusion time.
> Fusing the incumbent with a *generative* second opinion from the same frozen model, with
> self-consistency, or with noise all make it significantly **worse**, so the gain is attributable to
> the discriminative readout and not to score averaging.

What the project **cannot** claim: that a non-generative verifier beats the incumbent standalone (the
pre-registered arm lost, −0.0245); that the head is free (not measured); that this closes the
selection gap (it captures ≈28% of its own pair-oracle headroom and ≈29% of the oracle@8 prize).

**What must be verified before this goes in a paper**, in order:

1. **Clean disjoint retrain of the winner.** Re-extract features and re-fit the generator-prompt head
   from scratch under a *new* seed and a selection split drawn to match the eval pools' question mix
   (finding 6.3), then re-measure the fusion. If the +0.031 is a CV-transfer accident it will not
   reappear.
2. **Significance under multiplicity.** ~20 arms were reported; the nominal CI is not corrected. Run
   a permutation test on the fusion Δ and report a corrected p (Holm over the fusion family, or the
   max-Δ null across all arms). The control pattern (3/3 heads up, 3/3 non-heads down) is currently
   doing the work the CI cannot.
3. **Cost, measured.** Capture hidden states during the *actual* best-of-8 sampling run rather than a
   separate teacher-forced pass. If they match, the fusion is genuinely free at generation time and
   the claim becomes "+0.031 sel_eff at zero added inference cost." If they do not, the honest cost is
   ~2× verifier prefill and the claim must say so. Also read the features off the **adapted** model's
   forward pass: if the signal survives, one pass yields both P(Yes) and h; if it collapses, the
   complementarity came from using an unadapted network — which is itself the explanation.

**If the confirmation fails**, the accumulated evidence is now unambiguous. Twenty-plus distinct
approaches — six cross-family judges, scale, pairwise/tournament, more samples, richer answers,
diverse generation, Dawid-Skene, bandits, portfolios, logit fusion, slice discovery, shrinkage, TTA,
neuro-symbolic gates, ranking objectives at two levels, feature-based selectors, zero-shot contrastive
alignment, discriminative heads — converge on the same shape: **within-question selection is capped
near 0.78–0.81 by information the 7B does not have, not by the selector's form.** The residual is not
a mis-ordering (best correct candidate at mean rank 4.46 when the incumbent errs, median gap 0.132),
not label noise (0/2345 inconsistently labelled duplicate strings), not the objective (closed at two
levels), and not surface relevance (measured). At that point the finding *is* the pattern, and the
right paper is the negative-result-plus-limit characterisation, with the +0.031 fusion as the one
positive and the pair-oracle gap (≈0.071–0.109 across every scorer pair tried) as the measured wall.

---

## 8. Ranked next round

Re-ranked from the plan's Tier 2/3 given what was learned. Items 1–3 are cheap and decisive.

1. **Score the incumbent adapter over the 16,621 disjoint train items** (~132K pointwise forwards,
   one pass, no generation, <1 GPU-day). This single missing measurement is the blocker on *three*
   separate diagnostics at once: it converts the feature+incumbent fusion (+0.0218), the SigLIP-head
   fusion (+0.0381) and any per-item arbitration router from eval-fitted upper bounds into fittable,
   freezable, deployable numbers, and it removes the +0.0177 optimism correction from all of them.
   **Highest value per GPU-hour on the list.**
2. **Confirm the winner** — the three verification steps in §7 (clean retrain on a matched selection
   split; multiplicity-corrected significance; hidden states captured during real sampling). Until
   these land, C1 is a promising single measurement, not a result.
3. **T2.0 — re-measure real pairwise honestly.** Knockout (~14 forwards/q × 2,345 ≈ 33K forwards,
   ≈1.75× one pointwise pass) with the **clean disjoint adapter** on the full 2,345. The +0.076
   sel_eff win recorded in `RESEARCH_RESULTS_2026-07.md` was measured with the *contaminated* pooled4
   weights on n=578. If it holds, the bar for every future architecture is ~0.85, not 0.7752 — and
   every number in this document is being compared against the wrong baseline. This must be settled
   before the next architecture round is designed.
4. **T2.C, re-specified — a permutation-invariant set encoder over the 8 cached hidden vectors.** The
   hidden-state cache already exists (2.2 GB, `feats_hidden/`), so a small transformer over the 8
   per-candidate vectors with inter-candidate attention costs CPU-minutes, not GPU-hours. This is the
   cheapest possible test of the one hypothesis that §4.3 explicitly did **not** close: set-aware
   *architecture* (score of *i* depends on all 8) as opposed to set-aware *loss*. It is also the
   principled amortisation of item 3's comparative win into one pass, and it removes the position-bias
   confound that plausibly explains why LLM listwise *prompting* failed.
5. **T1.A stage 2 — distil the pairwise verdicts.** Train p(i≻j) from pair-relative features and apply
   Copeland everywhere. Now better-informed than in the plan: the feature set should include the
   hidden-state head score, which is the only pair-relative signal in the project that is both good
   and decorrelated. Conditional on item 3 confirming.
6. **T1.C — semantic re-clustering, promoted.** The feature arm found `soft_vote`
   (similarity-weighted consensus) is worth +0.0761 on the selection endpoint against +0.0181 for the
   exact-string duplicate count. Exact-string clustering is demonstrably leaving consensus signal on
   the table. Use the judge labels as the equivalence oracle first (a merge uniting a judge_ok=1 and a
   judge_ok=0 candidate is a detectable error) — free, and it gates everything downstream.
7. **Fix the CV→eval transfer failure** (finding 6.3) as a standalone protocol fix. Until it is fixed,
   *every* head-architecture selection on this split is unreliable, including the one that won.
8. **T3.A — medical visual-entailment head with cross-attention**, three-way entail/contradict/neutral,
   trained on synthesized laterality and negation contradictions. Still the only architecture aimed
   squarely at the measured failure mode, and §4.4 strengthened the case (frozen features carry visual
   evidence; the cosine readout is what fails). But it is a GPU-days bet with two priors against it
   (N41; generic ITM objectives do not track hallucination), so it goes after the cheap items. The
   check that decides it is the image-permutation null.
9. **T2.B — regenerate eval pools with per-candidate logprobs** (18,760 short vLLM generations, <1 h).
   Expected value has *dropped*: the consensus/likelihood feature family is now demonstrably saturated
   (a 39-feature consensus machine only ties). Run it as a feature source for items 5 and 7, not as a
   method.
10. **T2.E — vision-only kNN over labelled train golds.** Unchanged low prior, and the only remaining
    scorer whose evidence is ground truth rather than a model's opinion. Keep last.

**Closed by this round — do not re-propose:** ranking/listwise/pairwise *losses* on any existing score
function (plan candidates 3, 24, loss-half of 10); zero-shot contrastive alignment scoring, including
medical encoders and PMI normalisation (T1.D; candidates 6, 11, 12, 14, 20, 21, and the
BiomedCLIP/CONCH/MedSigLIP download programme); fusing a second *generative* opinion with the
incumbent (K1–K3); AUROC as a verifier-selection endpoint.

---

## 9. Artifacts and code

**Artifacts** (`results/cascade_methods/artifacts/`):
`verifarch_hidden_2026-08-04.json`, `verifarch_hidden_fusion_controls_2026-08-04.json`,
`verifarch_hidden_generatorprompt_2026-08-04.json`,
`verifarch_hidden_promptform_indomain_2026-08-04.json`, `verifarch_hidden_cvgap_2026-08-04.json`,
`verifarch_features_2026-08-04.json`, `verifarch_features_seeds_2026-08-04.json`,
`verifarch_features_followup_2026-08-04.json`, `verifarch_disjoint_assert.json`,
`verifarch_listwise_2026-08-04.json`, `verifarch_alignment_2026-08-04.json`.

**Code.**
Hidden-state arm — `src/training_methods/extract_generator_hidden.py`, `fit_hidden_head.py`,
`fuse_hidden_head_controls.py`, `cv_gap_listwise_capacity.py`, `finalize_hidden_head_artifact.py`.
Ranking-objective arm — `src/training_methods/verifarch_listwise.py`, `verifarch_listwise_diag.py`,
`verifarch_eval_imghash.py`.
Feature arm — `src/cascade_methods/verifarch_features.py`, `verifarch_features_seeds.py`,
`verifarch_features_followup.py`, `verifarch_features_merge.py`,
`src/training_methods/verifarch_assert_disjoint.py`.
Alignment arm — `src/verifier_arch/align_build_manifest.py`, `align_embed_all.py`, `align_score.py`,
`align_followup.py`, `align_datascaling.py`, `align_verdict.py`.

**Caches (gitignored):** `feats_hidden/` (2.2 GB, Lingshu-7B hidden states, both prompt forms),
`data/align_cache/` (3.1 GB, 3,709 PNGs + three encoders' embeddings).

Nothing under `MedEvalKit/`, `MedVLThinker/` or `MedRAG/` was modified. No abstention, deferral or
reject-option mechanism was proposed, built or evaluated anywhere in this round.
