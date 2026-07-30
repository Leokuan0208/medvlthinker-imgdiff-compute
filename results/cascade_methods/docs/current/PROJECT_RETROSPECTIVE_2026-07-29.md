# PROJECT RETROSPECTIVE — `medvlthinker-imgdiff-compute`
### Test-time compute for medical vision-language models: what actually helps
**Written 2026-07-29. Covers 2026-06-17 → 2026-07-29.**
**Author of the work:** Li-Wen Kuan (Leo). **This document:** the definitive account of the project.

---

## HOW TO READ THIS DOCUMENT

If you read only this file, you should come away able to (a) explain what was attempted and why,
(b) state the method precisely enough to re-implement it, (c) quote the results with their
confidence intervals and their weaknesses, (d) avoid re-running any of ~90 experiments that
already failed, and (e) know what to do next.

**Ground rules used throughout.** Every number is followed by the file it came from. Where a
number was corrected, the corrected value is used and the superseded one is named in §10. Where
something is unknown or was never measured, it says so explicitly rather than guessing. Nothing
here is estimated unless labelled `[estimated]`, and nothing is arithmetic I did not do
(labelled `[derived]`).

**Terminology.** This document deliberately avoids the project's internal codenames (ACC, VADR,
FALC, CASP, F3/F8/F10, Pandora, G1–G8, H1–H9) in the main text. Where a codename is unavoidable
because it names a file, it appears in parentheses after a plain-language description. A glossary
of the codenames is in §9.5 so that the older documents remain readable.

**A note on document status.** As of 2026-07-29, several standing documents in this repository
are numerically stale (`CLAUDE.md`, `README.md`, `RESULTS.md`, `READING_GUIDE.md`,
`PROJECT_OVERVIEW.md`, `TECHNICAL_REPORT_2026-07.md`, `METHOD_FINAL_2026-07.md`,
`RESEARCH_RESULTS_2026-07.md`). This retrospective supersedes their headline numbers. §9.4 lists
what to read in what order, and §10 lists what each stale document gets wrong.

> **Late addition, 2026-07-29 — Finding 1 was re-derived after a prompt-matching audit.** The flagship
> "reasoning hurts perception" table was built from **prompt-unmatched** think/direct arms. Correcting it
> makes the perception half **stronger** (**17/20**, not 15/20) and the reasoning half **weaker**
> (model-dependent, not universal); all 7 Lingshu-32B cells and QoQ-Med-VL-32B's reasoning cells are
> **withdrawn**; the **open-text** half is **provisional** pending a re-run. **§5.1 carries the corrected
> statement and table; §10.1 C20–C25 and §10.2 X15–X19 carry the corrections; §10.5 records an open
> `MedEvalKit` dependency issue.** Artifacts: `artifacts/finding1_corrected_2026-07-29.json` and
> `artifacts/finding1_prompt_matching_audit.json`.

---

# 1. WHAT THIS PROJECT SET OUT TO DO, AND HOW THE QUESTION EVOLVED

## 1.1 The original question

Medical vision-language models (VLMs) take a medical image plus a text question and produce a text
answer. Bigger models are more accurate and much more expensive. The project's founding question
was the standard efficiency question:

> **Can we get large-model accuracy at small-model cost by spending compute selectively?**

The specific vehicle was a **cascade**: run a cheap 7-billion-parameter model on every question,
and call an expensive 32-billion-parameter model only on the questions the cheap model is likely to
get wrong. The decision rule that chooses when to escalate is called the **gate**.

## 1.2 How the question changed, four times

The project's real intellectual content is that this question turned out to be the wrong one, and
was replaced four times.

**Change 1 — from "which model?" to "which configuration of which model?"** (2026-06-17). The
gate was found to be essentially un-improvable: the property it needs to predict — *will the big
model fix this particular error?* — is close to unpredictable from anything cheap
(~0.6 AUROC, see §5.2). But a different lever appeared: the big model's *reasoning mode* was
actively hurting it on perception questions. The question became **what compute configuration
should be used**, not which model.

**Change 2 — from multiple-choice to open-ended answers** (2026-06-24). Every routing signal was
degenerate on multiple-choice questions because the answer is one letter out of four. On free-text
answers, the same signals jumped from ~0.6 to ~0.87 AUROC. The question became **what does the
answer format do to the routing problem**.

**Change 3 — from reading a frozen model to training a small one** (2026-06-25/26). Every
training-free selector sat at the random-pick floor. A *trained* verifier — a small model
fine-tuned to score "is this answer correct?" — broke that floor for the first and only time in
the project. The question became **what is worth training, given that the models are frozen**.

**Change 4 — from "what does the method cost?" to "what is the honest baseline?"**
(2026-07-01, and decisively 2026-07-07). Two shifts here. First, the project abandoned its own
internal evaluation harness for **MedEvalKit**, the harness that faithfully reproduces the
published numbers of the models being used, so every claim became anchored to a public baseline.
Second, the project discovered it had been comparing itself against the *cheapest* way of running
the big model rather than the way a user would actually deploy it.

## 1.3 The final question, and the final answer

> **The accuracy–cost tension in medical VQA is not a law. It is a consequence of spending
> test-time compute uniformly — the same amount of reasoning, the same number of samples, the
> same model, on every question regardless of what the question needs.**

The deliverable is a **format-aware adaptive cascade**: it detects, from the prompt alone, whether
a question is multiple-choice or open-ended, and runs a different policy for each. It has one knob
with two settings — *compute-lean* and *accuracy-max* — and both settings use **less compute than
a single forward pass of the 32B model** while matching or beating it.

---

# 2. THE ARC — EVERY PHASE AND PIVOT, WITH THE DECISIVE NUMBER

## 2.0 Pre-history (before the daily logs begin)

The repository name, `imgdiff-compute`, is a fossil of the first project. Four directions were
killed before the first daily diary. They are documented in `CLAUDE.md` §2 and their code is in
`archive/`.

**P0.1 — Question-aware visual-token pruning.** Drop unimportant image tokens to save compute.
*Killed:* "did not yield a usable accuracy cliff" (`CLAUDE.md` §2). **No surviving number.** The
raw CSVs are in `archive/image-difficulty/`; a real figure may be recoverable there but was never
extracted. (The idea returned six weeks later as a projection — see §6.9 item N78 — and is still
unmeasured.)

**P0.2 — Image-difficulty-driven adaptive compute.** Spend more compute on "hard-looking" images.
*Killed:* "correlations were the wrong sign / near zero" (`CLAUDE.md` §2). **No surviving number.**
Echoed much later: unsupervised reliability estimation tracks *self-agreement* (~0.52), not
accuracy (~0.29) — `artifacts/dawid_skene_aggregate.json`.

**P0.3 — Single-model routing.** Route *within one model* across its own configurations
(reasoning on/off × retrieval on/off). *Killed by a permutation control:* with 2,000 shuffles
preserving marginal accuracy but destroying per-question complementarity, the oracle sat
**~29 standard deviations BELOW the random-allocation floor** — i.e. one model's configurations
are mutually redundant and carry no routable signal (`CLAUDE.md` §2, script
`archive/single-model-routing/oracle_luck_floor.py`, re-cited `docs/archive_mcq/FINDINGS.md` L70).
**Caveat:** the −29σ figure is quoted from documentation; the script's output artifact was not
located in `results/cascade_methods/artifacts/`. Treat it as documented-but-unverified.

*Why P0.3 matters more than the other three:* it is the **structural template for the entire rest
of the project**. Every later negative has the same shape — a large oracle gap that no frozen-model
signal can harvest. The project came to call this the **luck floor**.

**P0.4 — Cross-model 7B→32B cascade.** The structural fix was to route *between two different
models*. Models: `MedVLThinker-7B-RL_m23k` → `MedVLThinker-32B-RL_m23k`. Frozen confidence-margin
gate at τ = 0.426, calibrated on a held-out PMC-VQA training sample (v1 `train.csv`, n = 3,000);
cheap leg served at a reduced image resolution (cap320). Deployed anchor, reproduced exactly on
2026-06-17:
**accuracy 0.5718, escalation 63.3%, prefill-inclusive compute 73.6% of always-32B**, over
6 benchmarks / 8,220 samples (`progress/progress_June_17.md` §0).

---

## 2.1 Phase 1 — The cascade research loop (June 17–18)
*Source: `progress/progress_June_17.md`, 375 lines — the densest single log.*

### The founding negative: the gate is signal-limited

Three measurements, all on the deployed anchor, that ended the "build a better gate" program on
day one:

1. **Oracle headroom.** Escalating *only* the questions where the 7B is wrong and the 32B is right,
   cheapest-first, costs **11.2% of always-32B compute** (4.6% escalation) versus the deployed
   gate's 73.6%. A ~60-point gap the gate cannot reach. (`progress_June_17.md` §1.)
2. **Outcome structure.** Of the 7B's errors, **62.8% are futile to escalate** (the 32B is also
   wrong) across all 6 benchmarks; 58.4% on the four competent ones. Of the deployed gate's
   escalations, only **22.5% are beneficial** and **15.2% are actively harmful**.
   (`progress_June_17.md` §2; `docs/archive_mcq/FINDINGS.md`.)
3. **Recoverability is nearly flat in uncertainty.** P(32B right | 7B wrong) rises only from 28% to
   43% across margin quintiles and caps at ~43–50%.

A literature sweep catalogued **84 methods across 8 families** (`artifacts/literature_raw.json`).
Two data facts collapsed whole families at once: stored log-probabilities are normalized
log-softmax, so energy-based and max-logit scores are *identical* to maximum softmax probability;
and semantic entropy / self-consistency collapse to analytic option-entropy when the answer is a
single letter.

**Verdict:** among training-free gates built from cheap-model features, the plain confidence margin
is approximately optimal.

### Turning point 1 — the strong leg was running in the wrong mode

Labelling the 32B in every mode revealed that **reasoning over-thinks perception questions**
(`progress_June_17.md` §3):

| benchmark | 32B with reasoning | 32B without reasoning | Δ |
|---|---|---|---|
| SLAKE | 0.764 | 0.841 | **+0.077** |
| VQA-RAD | 0.776 | 0.893 | **+0.117** |
| PMC-VQA *(`test_clean.csv`, n = 2,000)* | 0.556 | 0.565 | +0.009 |
| PathVQA | 0.673 | 0.672 | −0.001 |
| MMMU | 0.688 | 0.629 | −0.059 (reasoning needed) |
| MedXpert-R / -U | 0.326 / 0.384 | 0.288 / 0.301 | reasoning needed |

*(Internal-harness numbers: this whole era's PMC-VQA cell is the human-verified v1
`test_clean.csv`, 2,000 items — **not** the 33,430-item `test_2.csv` of the later MedEvalKit
tables. See §9.3.)*

…at roughly **2 generated tokens versus ~477**. Swapping the escalation target from
reasoning-mode to direct-mode dropped compute from **69.5% → 48.6% and simultaneously raised
accuracy 0.653 → 0.660** on the competent four.

*(The operative deltas for the deployed cheap-leg resolution are slightly smaller:
SLAKE +0.085, VQA-RAD +0.077 at cap320 — `INCONSISTENCIES.md` X6.)*

### The three-tier compute-configuration cascade

The structural result of Phase 1 (internally "ACC"): a cascade over *compute configurations*
rather than models —
**7B-direct@cap320 → 32B-DIRECT@cap320 → 32B-reasoning@full-resolution**, each tier gated by its
own confidence margin. The big model's *fast* mode is inserted as an intermediate tier, so the slow
reasoning pass fires only on the ~18% reasoning residual.

**Canonical measured numbers** (`INCONSISTENCIES.md` X1 → `artifacts/master_data.csv`), 6
benchmarks, at parity with always-32B-reasoning:

| system | accuracy | latency | energy | compute |
|---|---|---|---|---|
| always-32B-reasoning | 0.5723 | 11.34 s | 6,318.8 J | 100% |
| 3-tier cascade | 0.5693 | **2.27 s (−80%)** | **1,181.9 J (~5.3×)** | **52%** |

On the 5-benchmark pool (excluding MedXpert): 8.88 s → **0.44 s**, compute 24.9%.

**The win is the structure, not the gate.** Holding the 3-tier configuration fixed and swapping the
gate, every real gate clusters at 5.0–8.0 s and 51–62% compute; only *random* collapses
(23.4 s / 130%) (`progress_June_17.md` §6b).

### Turning point 2 — the honest novelty assessment

The same log contains the project's first major self-correction. An adversarial prior-art check
found that every building block already exists: reasoning-hurts-medical-VQA (Med-R1);
large-model self-gated fast→slow (CAR, arXiv 2505.15154); resolution escalation (VisionThink);
multi-tier cascades (FrugalGPT, AutoMix); and decisively, the cross-model agreement gate is
**Agreement-Based Cascading (arXiv 2407.02348)**. The log retracts in place:
*"Earlier I called it 'the one genuine improvement'; that was wrong."*

Also killed in this window (Appendix C, June 18):
- **Cross-family complementarity routing.** Complementarity is real and huge (3-model oracle union
  0.801 vs always-32B 0.645) but **unexploitable**: a learned router scores 0.621 ≈ always-7B;
  a CLIP-style image+text recoverability predictor scores **AUROC 0.50 — chance**.
- **Vision-sensitivity gating.** **56.9%** of 7B answers are unchanged when the image is blanked
  (a striking language-prior diagnostic) but the image-insensitive items are **no less accurate**
  (0.620 vs 0.625).
- **Speculative decoding.** Infeasible: vLLM 0.10 rejects draft-model speculative decoding.

**Signal search declared exhausted:** 9 families tried, all capped at ~0.6–0.69 recoverability
AUROC, including a hidden-state probe on layer-14 activations that is *worse* than the
log-probabilities (0.53 vs 0.63).

---

## 2.2 Phase 2 — Cross-family validation and the cost bug (June 20–22)
*Source: `progress/progress_June_20-22.md`.*

**Why:** the "reasoning hurts perception" premise had been proven on one model pair. If it were
specific to that pair there is no paper.

Four more medical VLM families were downloaded → **5 families across 3 architectures**:
MedVLThinker, Lingshu, QoQ-Med-VL (all Qwen2.5-VL), Chiron-o1 (InternVL3), MedGemma (Gemma3).

**The foreign-prompt confound and its fix.** The reasoning tier had been run with *MedVLThinker's*
`<think>` system prompt on all five families. Each model's native reasoning recipe was recovered
and everything re-run at n=8,220. Result: **the effect is real, not a prompt artifact** — direct
mode ≥ native reasoning mode on perception for all five families — but the foreign prompt had
*inflated its magnitude*. Example: Lingshu's big-model reasoning accuracy moved 0.611 → 0.661.

> **⚠️ Reversed 2026-07-29 — this "fix" is what broke Finding 1.** Switching to each family's *native*
> recipe made the **think arm** differ from the **direct arm** by more than the reasoning instruction:
> different personas, different answer-format clauses, and for Lingshu and QoQ a different image
> resolution. Worse, two of the native recipes contain **no reasoning trigger at all** — Lingshu's
> generated 3.0 tokens (C22). The corrected analysis therefore goes **back to the superseded
> "foreign-think" dumps** for Lingshu / QoQ / Chiron / MedGemma, because those keep the answer-format
> constraint in both arms and actually reason. Net: the perception effect is **larger** than the native
> re-runs said, not smaller (§5.1, C20). The lesson is not "use native prompts" but **"match the two
> arms, and verify the think arm actually reasoned by counting generated tokens."**

**The cost-methodology bug (a rigor turning point).** Symptom: Lingshu's always-reasoning
configuration appeared *cheaper* (0.12 s / 33 J) than always-direct (0.28 s / 90 J) — physically
impossible. Cause: Lingshu's native reasoning emits ~3 tokens, but the latency/energy model
`a·gen + b` had been fitted on the *foreign*-prompt run (70–407 generated tokens) and then
extrapolated down to gen=3, producing an energy intercept of −16 J. Fixed by using a
median-with-zero-slope estimator where generation length is near-constant, **plus re-measuring
every reasoning tier under its native prompt**. All five families then became cost-monotone. This
fix is why the canonical 11.34 → 2.27 s numbers are correct and the diary's earlier
20.0 → 5.7 s numbers are retired.

**Training-based methods: a robust negative across all three ways training can help a cascade.**
- *Route:* a LoRA-fine-tuned stability router scores AUROC **0.7226 < a logistic regression on
  scalar signals, 0.7328**.
- *Distill:* distilling the big model's direct-mode competence into the cheap leg is **net-flat**
  (+0.007 / +0.000); it redistributes accuracy across benchmarks (PathVQA +0.099, VQA-RAD −0.086)
  rather than lifting it.
- *Fuse:* a trained fusion head captures ≈0–14% of the real union-oracle headroom and **collapses
  on leave-one-family-out transfer** (0.242 on Chiron) because logit scales differ per family.

All three bottleneck on the *same* quantity — "which model is right on this query?" — at
~0.58–0.73 AUROC.

**An uncomfortable standing conclusion:** the full size+mode cascade pays off cleanly **only for
MedVLThinker**. For Lingshu, QoQ and Chiron, reasoning is net-harmful, so the cascade collapses to
the cheap leg and every method in the comparison table produces identical numbers at 0%
escalation.

---

## 2.3 Phase 3 — The open-ended ceiling break (June 24)
*Source: `progress/progress_June_24.md`.*

### First, an honest negative about the previous phase's flagship

A "visual-stability rescue" (keep low-margin queries on the cheap model if the answer is invariant
across a resolution ladder) looked excellent on the frozen two-tier gate: 32B calls
**60% → 20%**, compute **69.5% → 43.3%** at parity, guardrail-clean. Integrated into the better
three-tier cascade, **a control experiment killed it**: at matched skip rate, latency is
none 1.50 s, resolution-stability 0.95 s, **random 0.65 s**, big-model-direct-confidence 0.58 s,
inverse-stability 0.90 s. Resolution-stability is the *worst real signal*; stable items benefit
*more* from reasoning (+0.058 vs +0.039), so the rescue skips the highest-value reasoning calls.
Supporting fact: running the 7B at full resolution fixes only **12.6%** of its cap320 errors versus
the 32B's **45.2%** — errors are capacity-bound, not resolution-bound.

Also settled here: **more gate calibration data does not help.** Eval behaviour is flat from 2,000
to 3,000 calibration rows (0.6503 → 0.6502, threshold variance → 0). The limit is the calibration
set's *perception-only distribution*, not its size. And **adaptive reasoning-length is dead**: the
32B commits its answer at the very *end* of the trace (answer-marker at median 0.99 of trace
length over 8,220 traces).

### Turning point 3 — multiple-choice → open-ended, and the routing ceiling breaks

*Reasoning for the move:* single-letter answers make every routing signal degenerate.

- **No within-family gap exists first:** the MCQ-trained models generalize uniformly poorly to
  free text (3B 0.457 ≈ 7B 0.419 ≈ 32B-direct 0.498 ≈ 32B-reasoning 0.453) — nothing to cascade to.
- **Cross-family, with a strong open-text model:** routing AUROC (predicting cheap-model-wrong)
  **0.866** for Lingshu-7B → Lingshu-32B, recoverability **0.804**; 0.735 cross-family — versus
  the ~0.6 multiple-choice ceiling.
- **The cause is 4-option discreteness, not answer length.** Open answers have median **1–2
  tokens** yet AUROC ~0.87.
- **Robustness:** holds under an LLM judge (0.860/0.784), on a third dataset (0.797), pooled
  3-dataset **0.846, 95% CI [0.830, 0.862]**, and on a fourth imaging modality (0.749).
- **The crucial honest split:** **detection** (is the cheap answer wrong?) improved from
  0.66 → 0.85. **Cascade gain** (oracle minus cheap) did not: only **+0.02 (MCQ) / +0.06 (open)**.
  One dataset makes this vivid: 7B 0.302 ≈ 32B 0.301 — *zero cascade headroom* — yet detection
  AUROC 0.749.
- **Theory backing:** the strong model breaks cheap-correct answers only 22% (MCQ) / 14% (open) of
  the time, i.e. it is close to a **uniform improver**; by Jitkrittum et al. (NeurIPS 2023),
  confidence is then near-Bayes-optimal. "The gate is saturated" became a *theory-predicted*
  result rather than a failure.
- **Exhaustive gate hunt on open text:** confidence 0.866 > exact self-consistency 0.845 >
  mean-F1 0.844 > semantic entropy 0.807 > semantic self-consistency 0.806 > self-verification
  0.755, and an honest 20-seed cross-validated fusion of all six **ties at 0.866 (+0.000)**.

### The user's scope decision, and its consequence

A training-free safe-abstention mechanism was built and validated (auto-answer 54% of one dataset
at ≤5% error, detection AUROC 0.89). **The user rejected the entire direction** — "not interested
in anything to do with abstention or referral" — and on 2026-07-07 made it a **permanent
project-wide prohibition**. The method must always answer. This is the only case in the whole arc
where working code and a genuine positive were discarded for scope rather than because they failed.

With the gate saturated and abstention forbidden, only two axes remained: **action** (do something
different) and **selection** (produce several candidates and pick).

- **Action:** cheap same-model repairs recover **14% of the errors the 5×-larger model misses**
  (11–17% per benchmark — a genuinely novel observation) but are **unharvestable**: a 4-rung repair
  ladder *loses* at parity (43% vs 39% compute). The big model's advantage is capacity-bound.
- **Selection:** greedy 0.730 → oracle-of-8 **0.879**, but self-consistency fails through a
  **majority trap** — the correct answer is a *minority* vote in **74–90%** of recoverable cases
  (mean ~1.5 of 8). Every training-free selector sits at the random-pick floor of 0.720
  (self-verification **0.715 — below random**; big-model listwise re-ranking 0.758), and **none
  beats simply running the big model once (0.819)**.

---

## 2.4 Phase 4 — The trained verifier (June 25–26)
*Source: `progress/progress_June_25-26.md` — the "detailed narrative" section is the best single
document in the June record.*

### The luck floor mapped across every remaining axis

Seven mechanisms were checked: action, selection, synthesis, knowledge augmentation, cross-family
agreement, few-shot in-context learning, structured grounding.

- **Retrieval ruled out with zero GPU time:** the 32B fixes genuinely-unknown errors **equally**
  across knowledge-type and perception-type questions (**38% vs 36%**) ⇒ the deficit is capacity,
  not retrievable knowledge.
- **Few-shot in-context learning hurts:** PathVQA 0.343 → **0.203**.
- **The structured-output escape is an artifact of incompetence:** a *weak* bounding-box grounder
  shows spatial-consistency AUROC 0.82; a *competent* grounder collapses to chance (**0.557**) and
  medoid selection ties greedy. **The luck floor generalizes to verifiable outputs.**
- **A self-refutation, documented rather than buried:** the hypothesis that PathVQA's difficulty is
  a caption-extraction artifact (formed from 14 eyeballed cases) was **refuted by the researcher's
  own systematic audit** — well-formed questions turned out to be the *hardest*, and
  "artifact"-labelled questions had *higher* accuracy (0.369 vs 0.144).

### Turning point 4 — training breaks the luck floor

A small LoRA verifier was fine-tuned to score P(correct | image, question, candidate answer) using
per-sample LLM-judge labels, and used to select the best of 8 samples.

| setting | greedy | best training-free | **trained verifier** | oracle | gap captured |
|---|---|---|---|---|---|
| free text, pooled-4, n=1,064 | 0.413 | 0.411 (self-consistency) | **0.501** | 0.592 | **49%** |
| organ bounding boxes, n=487 (IoU≥0.3) | 0.197 | 0.164 | **0.255 / 0.257** | 0.343 | 40% |
| chest X-ray boxes, n=435 | 0.041 | 0.053 | **0.232 / 0.230** | 0.285 | **78% / 77%** |

Sub-results that make it a principle rather than a fluke:
- **Training, not size, is the active ingredient:** trained 7B verifier **0.501** > zero-shot 32B
  verifier **0.357**.
- **Transfers zero-shot** to a held-out fifth dataset (+0.024).
- **Image-grounded:** blanking the image costs −0.047, refuting the "lazy verifier" failure mode.
- **A genuine test-time-scaling curve:** 0.385 → 0.501 over K = 1 → 8, while random selection
  stays flat.
- **Test-time compute beats parameters:** 7B best-of-8 0.501 > 32B single pass 0.444 (pooled).
- Bootstrap CIs: free text **+0.116 [+0.092, +0.139]**; chest X-ray boxes **+0.191 [+0.152,
  +0.232]**.

**A bug that flipped a sign.** An earlier chest-X-ray pass read as a *negative* (IoU 0.022). It was
a **coordinate-space bug** — the grounder emits boxes in smart-resized pixel space and chest X-rays
are large. Rescaling turned the false negative into the strongest positive in the project.

---

## 2.5 Phase 5 — Consolidation and the numbers audit (June 27–28)
*Sources: `progress/progress_June_27-28.md` (reconstructed 2026-07-02), `INCONSISTENCIES.md`.*

Not an experiment window — a **rigor** window, and it matters for this document's honesty.

The three-tier cascade's efficiency headline existed in **three conflicting forms** across the
paper draft, README, CLAUDE.md and the method spec. A three-way audit resolved everything to
`artifacts/master_data.csv` and froze it. Ten numeric corrections were logged (X1–X10; see §10).
A policy was set: forward-facing documents get corrected; dated diaries are left as written with a
pointer, so history is preserved rather than rewritten.

The thesis was refocused to *test-time compute for medical VLMs — what actually helps*, with a
two-positive arc: **the compute-configuration cascade (efficiency) + the trained verifier
(accuracy), unified by the luck floor**.

---

## 2.6 Phase 6 — Open text: the method beats the strong model (June 29–30)
*Source: `progress/progress_June_29-30.md`.*

- **The headline accuracy win:** a verifier-augmented cascade **beats the 32B**: 0.517 vs 0.462 at
  35% escalation, per dataset. It generalizes: held-out out-of-distribution 7B+verifier 0.353 vs
  32B 0.289; works cross-family; **all 12 family × dataset cells plus 3 held-out cells**
  (`docs/current/OPENTEXT_MASTER_TABLE.md`).
- **The gate question is settled:** **verifier confidence is the best gate** (0.518 at 34%
  escalation). Trained gates targeting recoverability do not beat it across three families, both
  regimes, and the full feature set. Input-perturbation stability is weak (0.60) and adds
  **nothing** on top of verifier confidence (0.852 → 0.853). The **recoverability wall (~0.6
  AUROC)** is the binding limit. *(See §7 hole 15 — one later artifact disagrees with this
  conclusion and the disagreement is unresolved.)*
- **A quiet but decisive finding:** with a held-out threshold, the gate is redundant at N=8 —
  **selection is the real lever, not escalation**.
- **The verifier's own ceiling diagnosed:** per-answer AUROC is 0.90 but **selection efficiency is
  only 74–82%** of oracle. A ranking/contrastive trainer lifts per-answer AUROC 0.90 → 0.93 **but
  not selection**. The higher-value levers are therefore **better candidates**, not more verifier
  training. *This diagnosis sets the entire July agenda.*

---

## 2.7 Phase 7 — Turning point 5: the faithful protocol (July 1–2)
*Source: `progress/progress_July_01-02.md`.*

**The move that anchored the project to reality.** The internal evaluation harness is not faithful
to the published Lingshu numbers; **MedEvalKit** is. A recipe was locked (a dedicated venv,
vLLM 0.9.0.1, the Qwen2_5_VL wrapper, `datasets_path=hf`, `TORCHDYNAMO_DISABLE`). Anchor:
**Lingshu-32B on MMMU-Medical = 0.633 vs the paper's 62.3 — exact.** MedXpert, PMC-VQA, SLAKE and
VQA-RAD also reproduce.

- **Faithful two-tier MCQ cascade at iso-strong accuracy:** PMC-VQA **−69% compute / −33%
  latency** (Lingshu, n=33k — `test_2.csv`), SLAKE −56% / −22%; cross-family PMC −49%
  (`test_2.csv`), VQA-RAD −41%. Win magnitude
  ≈ the (32B − 7B) accuracy gap. Gate-signal ordering: **margin > confidence > cumulative
  log-probability**.
- **The confidence gate cannot be improved by substitution:** the state-of-the-art post-hoc
  deferral rule and all simple signals are worse; trained gates only tie.
- **MCQ is saturated** (recoverability wall ~0.578) ⇒ *"beat the strong model" is an open-text
  phenomenon, not a multiple-choice one.*
- **A 32B zero-shot verifier is worse than the trained 7B (0.355 vs 0.403)** ⇒ task-training beats
  size, confirmed a second time.
- **Candidate quality is the real limit:** a cross-model candidate pool raises oracle-of-N by
  **+0.11 to +0.15**.
- **Claude-as-judge validated** (Sonnet-5 subagent, no API key; 100% exact-match anchor), which
  unblocked the open-ended halves of SLAKE and VQA-RAD.
- **A retraction inside 24 hours:** July 1 asserted "Lingshu has no promptable reasoning mode."
  **Wrong** — with the right prompt, generated tokens go 3 → 174, and 267 with real `<think>` tags.
  The cause was a weak-prompt artifact: the harness's `--reasoning True` flag only appends
  *"put the letter in \boxed{}"*.

---

## 2.8 Phase 8 — The three-family matrix and the gate-proxy correction (July 3–4)
*Sources: `progress/progress_July_03.md`, `progress/progress_July_04.md`.*

**Efficiency generalizes across architectures** (InternVL3 is not Qwen-based). Compute savings at
iso-strong accuracy: MMMU-Med Lingshu keep-cheap / MedVLThinker −14% / InternVL3 **−62%**;
PMC-VQA (`test_2.csv`) Lingshu **−69%** / MVT −49% / IV3 −16%; VQA-RAD −17 / −41 / **−67%**;
PathVQA −31 / **−68** / −20%.

**Reasoning is warranted only where reasoning has headroom.** MMMU-Medical gains
**+0.027 (Lingshu) / +0.100 (MedVLThinker) / +0.120 (InternVL3-38B)** — computed directly from
`MedEvalKit/eval_results_*/{}/MMMU-Medical-val/*/parsed_output.json`, and matching
`METHOD_FINAL_2026-07.md` L267. *(The July-3 diary records +0.034 / +0.107 / +0.120; those diary
endpoints do not exist on disk. Use the values above — see §10.)* MedXpert gains ~0 at the floor
except MedVLThinker +0.045. All three families reason on a *generic* prompt (generated tokens
3 → 275 / 561 / 368); MedVLThinker does **not** need its native `<think>` tag (0 tags emitted,
still +0.100), fully retiring the July-1 claim.

**July 4 — three workstreams, the intellectual core of the window:**

- **Evaluation validity: sound.** The "massive" VQA-RAD/PathVQA gaps versus the Lingshu paper were
  a **metric mismatch** (closed-only vs an open+closed blend), not a bug. Metric rule locked: pure
  exact-match for the fully-multiple-choice benchmarks; judge-scored open+closed blend for
  SLAKE/VQA-RAD/PathVQA. One genuine anomaly survives — **Lingshu-7B on MMMU is +26 points over
  its own published number** — and is excluded from claims. (This became the contamination saga;
  see §2.12.)
- **Turning point 6 — the gate bake-off had been ranked by the WRONG PROXY.** Cascade quality
  tracks **recoverability-AUROC, not detection-AUROC**. Evidence: the gate with the *top* detection
  AUROC (0.693) has cascade quality *below* the plain margin gate (Δ −0.0015); the cross-gate
  correlation is **r ≈ +0.65 for recoverability versus r ≈ −0.21 for detection**. Consequence: an
  earlier "stability beats agreement" claim is downgraded to a regime-dependent tie. A new gate
  wins in one regime but is beaten there by a literal implementation of the published baseline and
  loses in another ⇒ not a headline. Unifying insight: the gate that helps in *both* regimes is
  trained self-verification.
- **Architecture — the deterministic format-aware router works.** It detects multiple-choice
  versus open-ended **from the prompt text, never from the gold answer**, and gives a tested Pareto
  win over always-strong on all three families: parity accuracy (Δ −0.0001 / −0.0003 / +0.00004) at
  **0.38–0.77× compute and 0.68–0.88× latency**. The recommended no-router alternative — a single
  unified generative verifier — is an *accuracy* win only; it never escalates and spends
  **1.3–1.6× always-strong compute**.

---

## 2.9 Phase 9 — Infrastructure wall and the idea backlog (July 5)
*Source: `progress/progress_July_05.md`.*

- **A 7th benchmark (OmniMedVQA) cheap leg reproduces the paper:** Lingshu-7B **0.8274** on the
  full **88,996** questions versus the paper's 0.829 — a 0.2-point gap, validating the pipeline
  after a parser fix.
- **The 32B/38B strong leg hits a deterministic two-GPU NCCL hang.** Every mitigation was tried and
  ruled out (chunked runs with retry; disabling the heartbeat monitor, which makes it hang
  *forever*; larger eval batch; 3-hour timeouts). Single-GPU is impossible: 64 GB of weights plus
  multimodal activations do not fit an 80 GB card. **Keep-cheap fallback taken**, and it is
  defensible: in the published paper the two models are tied on this benchmark (82.9 vs 83.4), so a
  cascade keeps the cheap leg and the missing cell changes no conclusion. **No fabricated metrics
  file was written.** Net reproduction: 6 of 7 fully faithful plus one cheap-faithful.
- **The offline pivot that shaped the rest of the project:** with the GPUs blocked, a systematic
  cross-field sweep (economics of information, portfolio theory, crowdsourcing truth inference,
  coding theory, social choice, sequential analysis, bandits, computer architecture) seeded
  `results/cascade_methods/METHOD_IDEAS_BACKLOG.md` — 68 ideas, each mapped to a concrete testable
  variation and judged against four stated binding limits: (1) candidate quality / oracle-of-N;
  (2) verifier selection efficiency 74–82%; (3) the recoverability wall; (4) the cost tension
  (best-of-N break-even is N ≤ 2 on compute, but at batch-1 the 32B is only ~1.9× the 7B's latency).

---

## 2.10 Phase 10 — The execution marathon (July 6)
*Source: `progress/progress_July_06.md`, 422 lines, ~15 experiments.*

**Wins:**
- **An optimal-search adaptive controller** (Weitzman 1979, "Pandora's box") unifying adaptive
  sample count *and* the escalation gate into one rule. At iso-best-of-8 accuracy, **held out**
  (5-fold cross-fit calibration, while the *baselines'* thresholds are swept on full data):
  **11.74 compute-units (−27%) and 409.8 J (−28%)** versus fixed best-of-8, beating even the
  optimistically-tuned adaptive-N baseline (−19%). Honest caveats: it covers 9 of 11
  configurations, and it is **sequential**, so its serial latency is 2,951 ms against a batched
  best-of-8's 522 ms.
- **Cross-model candidate pooling:** held-out oracle-of-8 **+0.080** over the single best model
  (+0.105 at B=16) — but the clever portfolio allocation is **≈ naive uniform** (Δ +0.002, and
  *negative* on one dataset). *The win is diversity, not allocation.*
- **Diverse generation (GPU):** raises the oracle ceiling **+0.027 at matched budget / +0.064 at
  M=15** (both CIs exclude 0) → **+0.025 significant verifier accuracy**. But on the largest cell
  the biggest oracle lift (+0.110) **cannot be converted**: selection efficiency falls
  0.574 → 0.496 and the confident-distractor rate rises 0.426 → 0.504. **Diverse generation shifts
  the binding limit from coverage to selection.**
- **A real pairwise verifier overturns its own simulation.** A simulation of pairwise verdicts
  derived from pointwise scores had found parity (Δ −0.003, ceiling +0.000). A **real** A-vs-B
  forward pass beats pointwise argmax: selection accuracy **0.374 → 0.410 [+0.016, +0.055]**,
  efficiency 0.783 → 0.859, **+0.050 on the near-ties**; a knockout tournament captures ~87% of it
  in ~7 comparisons. The lesson is stated explicitly in the artifact: *you cannot manufacture
  comparative signal from pointwise scores.*

**Negatives:** the unified generative verifier applied to multiple choice is a **resolved
negative** — content mode (options hidden) craters accuracy (0.132 vs 0.534) and letter-mode gains
are label-sensitive and flip sign under the as-run parse ⇒ **the router stays the better method;
the generative verifier's home is open text.** Also negative: bandit allocation (Δ +0.002) and
unsupervised Dawid–Skene aggregation (≈ majority, −0.013).

**The five-experiment selectability battery — the best-of-N program characterized, not deployed:**
- **No compounding:** the diverse-generation lever (+0.0303, significant) and the pairwise lever
  (+0.0205, significant) do not stack — pairwise-over-diverse is **worse** than
  pointwise-over-diverse (−0.0117) and the both-levers gain is not significant.
- **No pre-filter beats plain diverse+pointwise** — the best filter is not significant and flips
  sign per dataset, because the *correct new* answers diverse generation adds are themselves rare.
- **Capacity does not break the wall:** a 32B zero-shot verifier scores **0.480 vs the trained 7B's
  0.475, Δ +0.005 [−0.023, +0.032], not significant** (n=600,
  `artifacts/verifier_32b_gpu.json`) — though 32B-zero-shot does beat 7B-zero-shot by +0.067.
  The selectability ceiling is substantially fundamental.
- **Best-of-N is Pareto-dominated on compute** (always-32B 4.57 units vs iid-best-of-8 16,
  diverse-best-of-15 30). The envelope is greedy → 7B+controller → always-32B.
- **But latency-alive:** parallel best-of-N base = 347.1 ms generation + 175.5 ms verification =
  **522.6 ms = 0.79×** the 665 ms 32B forward — *yet still does not beat always-32B* (accuracy
  ceiling ~0.549 vs 0.673). **This latency number is asserted, not measured — see §7 hole 8.**

**Conclusion of the day:** the deployable lever is the **router**, not best-of-N. **And the log
flags its own load-bearing assumption in a box:** everything above assumes the strong leg is the
665 ms *direct-mode* 32B, whereas the paper's motivating cost is the ~11 s *reasoning* 32B.

---

## 2.11 Phase 11 — Turning point 7: the re-grounding (July 7)
*Source: `progress/progress_July_07.md`, 485 lines — the biggest single day.*

**The pivot:** *"we have been comparing against the wrong 'strong' model."* July 6's entire verdict
benchmarked the cheap leg against the cheapest, fastest version of the strong model. The honest
baseline is the model people actually pay for when they want its accuracy.

**The two blocked cells, now measured (batch-1, NVML power logging):**
- **32B with reasoning, open text: 10,521.6 ms / 2,001.9 J** versus direct mode
  **665.0 ms / 126.9 J** ⇒ **15.8× latency, 15.8× energy** — *and reasoning is LESS accurate on
  open text*: pooled **0.387 vs 0.537 (−0.150)**.
  (`artifacts/opentext_32b_think.json`. **n = 15** — see §7 hole 9.)
- **InternVL3-38B: direct 1,409.3 ms / reasoning 6,220.0 ms** (4.4× latency, 5.5× energy).

**The reframe result:** re-scored against always-32B-with-reasoning, the method **Pareto-dominates
on all 5 families** — matching or beating accuracy at 9–68% of its compute, 8–99% lower latency,
33–99.7% lower energy. On **3 of 5 families the big reasoning model is actually LESS accurate**.
Reasoning:direct latency ratios: MedVLThinker 49×, MedGemma 45×, QoQ 43×, Chiron 15×,
**Lingshu 1.2×**.

> **⚠️ Annotated 2026-07-29.** The `always-32B-with-reasoning` column of this 5-family table is built
> from the same **withdrawn** native-think arms as the published Finding-1 table (C22/C23). For
> **Lingshu** in particular the "reasoning" baseline generated **3.0 tokens**, so its 1.2× ratio and its
> near-tie accuracy are properties of an answer-format prompt, not of reasoning; with a genuinely
> reasoning arm Lingshu's perception accuracy is **worse** (pooled −0.0866), which if anything widens the
> method's margin. The per-family cascade rows have **not** been recomputed against the repaired arms —
> that is an open item, not a claim. See §5.1 and §10.1 C22.

**Two corrections banked here:**
1. **On real Lingshu multiple choice, margin is the best gate** (AUROC 0.7254). The
   resolution-stability gate is **inert** — the 7B is 98.95% stable between cap320 and full
   resolution, so it collapses to margin — and agreement is the *worst* ranker (0.6565). The
   MedVLThinker-era premise "stability/agreement beat margin" is **false for Lingshu** ⇒
   **gate choice is model-specific**.
2. **A router is required.** A single unified gate is beaten by margin on multiple choice and by
   verifier confidence on open text.

**Three pushes:** a confidence-advantage fusion on PMC-VQA (+0.0135 [0.0100, 0.0169], n=33,430
held out — **`test_2.csv`** — a broad-slice beat, but it runs both legs, +22% compute); a **parallel prefill
prefetch** giving −12.1% latency at zero accuracy change; and best-of-8+verifier on SLAKE-open
(+0.0496) cutting escalation ~4× (53% → 13%).

**Evening — the unified pipeline, and both Pareto modes go compute-negative:**

| mode | accuracy | Δ vs 32B-reasoning | escalation | compute (rel. to one 32B pass) | parallel latency |
|---|---|---|---|---|---|
| compute-lean | 0.5754 | +0.0123 | 16.7% | **0.49×** | 468 ms |
| accuracy-max | 0.5844 | +0.0212 | 64.0% | **0.93×** | 729 ms |

*(These are the pre-measurement numbers; the measured ones are in §4.)* The flip to
compute-negative came from replacing the fusion lever with a **cross-fit conformal "certified
veto"**: keep the cheap answer only where its Wilson lower-bound precision ≥ the strong model's
accuracy on the training fold, so the strong model is never even run there. Full-suite compute
5.695 → 4.246 (1.25× → 0.93×) while retaining **70.4%** of the fusion's accuracy gain. A
learning-to-defer rule repaired the two residual open-text losses.

**Four more wall confirmations closed the day:** logit-level fusion certifies **exactly one cell
and it is the contaminated MMMU** (contrastive decoding certifies zero); automatic slice discovery
finds **no genuinely-new slice** (raw count 1.62 per split, *below* the permutation null of 5.61)
while reliably re-finding the known cells unprompted, validating the hand-built router; credibility
shrinkage confirms thin-slice overfit is real (7.5 violations/split) but the existing
CI-lower-bound guardrail is the better fix (→0.25); and a kNN gate loses to the scalar margin on
**0 of 5** datasets.

**§15 of that log made the abstention prohibition permanent** and excised the corresponding backlog
idea. A distinction was recorded: the certified veto *keeps the cheap answer*, so it is an
answer-producing gate, not abstention.

---

## 2.12 Phase 12 — The paper and the rigor day (July 8, the last logged day)
*Source: `progress/progress_July_08.md`.*

**Two final honest negatives:**
- **Test-time adaptation of the cheap leg:** entropy minimisation / SHOT **collapses** it (−0.159);
  even the *label-informed oracle* prior ceiling is **<1% and mixed-sign**; temperature scaling is
  **exactly 0** by construction (monotone, argmax-preserving).
- **A neuro-symbolic gate:** strict logical constraints fire on **~1 sample** (laterality
  contradiction coverage 0.0002); the shared confident-wrong errors (18.5% of items) are
  **perceptual, not logical** — symbolic constraints catch 16 of 1,118 (1.43%).

**The user's paper critique forced the final reframing.** The manuscript was codename-heavy,
chronological rather than argued, and estimate-laden. New thesis: *the accuracy–cost tension is not
a law but a consequence of spending test-time compute uniformly.* Four contributions; every
codename stripped; related work explicitly concedes that the multiple-choice gate is not novel.

**The honest-baseline experiment** added an **oracle-mode 32B** — per-benchmark best of
reasoning/direct, a non-deployable upper bound. Compute-lean **matches** it
(+0.0015 [−0.0025, +0.0055], not significant) at 0.49× its compute, 469 vs 894 ms, 83.5 vs 171 J;
accuracy-max **beats** it (+0.0136 [+0.0108, +0.0165], significant). **Only {always-7B,
compute-lean, accuracy-max} are non-dominated** — all three 32B strategies, including the oracle,
are dominated.

**Cross-family generalization audit.** Reasoning-hurts-perception holds in **15 of 20 perception
cells strictly, 19 of 20 within ±0.02**; **VQA-RAD is negative in all 5 families**; the only
genuine perception reasoning-win is MedGemma on PathVQA (+0.040). Honest framing recorded:
**Lingshu — the headline family — is the *weakest* case for the reasoning half** (latency ratio
1.2×). The verifier finding was **downgraded**: "competitive with / matches the 32B", not "beats"
unconditionally (+0.039 on seed 0 but **ties on seed 1**).

> **⚠️ Superseded 2026-07-29 (this paragraph is the July-8 record).** The arms behind these numbers
> were prompt-unmatched. Re-derived: **17 of 20** strictly negative (14/20 CI-significant, pooled
> **−0.0401 [−0.0456, −0.0347]**, n = 30,250); all 7 Lingshu cells and QoQ's reasoning cells are
> **withdrawn**; the reasoning half is **model-dependent, not universal**; MedGemma's PathVQA win is
> real and survives full matching at **+0.0413 [+0.0220, +0.0607]**; and Lingshu's "1.2× latency ratio"
> is a property of the withdrawn 3-token arm, not of reasoning. See §5.1 and §10.1 C20–C24.
> Artifacts: `artifacts/finding1_corrected_2026-07-29.json`,
> `artifacts/finding1_prompt_matching_audit.json`.

**The MMMU contamination saga (three passes).** Lingshu-7B scores **0.80** on MMMU-Medical versus
its own published 54.0, and beats its own 32B (0.633).
- Not sampling noise (McNemar 34 vs 9 discordant, **p = 1.7 × 10⁻⁴**).
- Not a position artifact (full cyclic-shift debiasing still gives 0.7708 vs 0.6321; position
  explains ~4 of ~18 points).
- Then the adversarial audit the user demanded: **model identity PASS** (8.29 B params, correct
  architecture, correct snapshot); **image ablation DECISIVE** (0.8267 with the real image → 0.62
  blank → 0.5933 text-only); **control model DECISIVE** (an untuned non-medical Qwen2.5-VL-7B
  scores **0.5667** through the identical harness); independent rescore 0.82.
- **Verdict: genuine Lingshu-7B weights, consistent with train-set contamination outside our
  control.**
- Handling: **MMMU excluded entirely** ("Variant B", 5 benchmarks / 8 cells / n = 42,224). The
  sample-weighted headline moves only **−0.0005** (MMMU is 0.35% of the pool) but the **macro
  average must be corrected** (+0.0777 → +0.0621).

**The last estimate became measured.** The 32B-with-reasoning open-text accuracy had been an
estimate from an n=200 subsample. Extending the reasoning dumps to the full open sets and judging
them with the method's own grader: **SLAKE-open 0.6791, VQA-RAD-open 0.5450, PathVQA-open 0.1087**
(the estimate had been ~0.246 — PathVQA-open reasoning **collapses**, −0.267 versus direct mode).
This *raised* the vs-reasoning headline.

**Deliverable:** `paper/adaptive-cascade-medvqa_ieee_2026-07-08.{tex,pdf}`, 9 pages, IEEEtran, via
a locally-installed tectonic (there is no system LaTeX on this VM). The final headline CI was
computed the next day (2026-07-09) by `src/cascade_methods/f8_mode_vsthink_ci.py`.

## 2.13 After July 8

**There is no progress diary for 2026-07-09 or for 2026-07-10 → 2026-07-27.** Two artifacts were
produced in that window: `artifacts/f8_mode_vsthink_ci.json` (2026-07-09 19:44 — the current
headline CI) and `meetings/progress_report_professor_2026-07-27.html` (the most current
source-cited summary in the repo). Whether the paper was submitted, and whether the MMMU exclusion
was finally ratified by the user, is **not recorded anywhere**.

**2026-07-29 — the prompt-matching audit and the Finding-1 re-derivation.** Triggered by
`artifacts/pathvqa_judge_audit.json` (key `prompt_confound`), which found that the **open-text**
reasoning/direct comparison had used different system prompts. The audit
(`artifacts/finding1_prompt_matching_audit.json`) traced every Finding-1 cell back to its checkpoint
and recovered both arms' **verbatim** prompts — which are *not* persisted in the JSONL rows and exist
only as shell variables in `runners/*.sh` and module constants. Verdict: the confound is real and
pervasive in the prompts, but **bounded on multiple choice** (single-letter gold has no style/length
grading channel; the residual extraction channel is 0–3.4%), and correcting it makes the perception
half **stronger**. The re-derivation (`artifacts/finding1_corrected_2026-07-29.json`,
`src/cascade_methods/finding1_corrected.py`) gives **17/20** rather than 15/20, withdraws all 7 Lingshu
cells and QoQ's reasoning cells, and downgrades the reasoning half to model-dependent. The open-text
comparison is **not** repairable offline and needs a matched-prompt re-run (in flight). One action
item it produced that has nothing to do with these numbers: **persist the prompt in every future
checkpoint row.** Full detail in §5.1; corrections in §10.1 C20–C24.

---

## 2.14 The eight genuine turning points, at a glance

| # | date | trigger | what changed |
|---|---|---|---|
| 1 | Jun 17 | Labelling the 32B in *all* modes: direct ≥ reasoning on perception (SLAKE +0.077, VQA-RAD +0.117) | Search moved from the **gate** to the **compute configuration** |
| 2 | Jun 17 | The deferral router's guardrail reversal (91.6% vs 64.3%) and its novel claim failing (ΔAUROC −0.003) | The "better gate" program was formally abandoned; the recoverability wall became a *result* |
| 3 | Jun 24 | Moving to open-ended answers: routing AUROC 0.6 → 0.87, cause = 4-option discreteness | "The gate is saturated" reframed as a **benchmark artifact**; the open-text half was born |
| 4 | Jun 25 | User forbids abstention; gate already saturated | Forced the **action** and **selection** axes → the systematic luck-floor map |
| 5 | Jun 25–26 | Training breaks the luck floor (49% / 40% / 78% of oracle gap) — after a coordinate bug had made it look negative | The second positive; "training is the universal active ingredient" |
| 6 | Jul 1 | Adopting MedEvalKit; Lingshu-32B MMMU 0.633 = paper 62.3 exactly | Every number anchored to a published, reproducible baseline; primary family switched |
| 7 | Jul 4 | The bake-off had ranked gates by **detection**-AUROC when quality tracks **recoverability**-AUROC (r +0.65 vs −0.21) | Invalidated a prior gate ranking; re-pointed at one unified method with efficiency first-class |
| 8 | Jul 7 | Measuring the 32B-with-reasoning open-text cost (10.5 s / 2 kJ, and −0.150 accuracy) | Reversed July 6's verdict; produced the final Pareto-dominance headline |

---

# 3. THE METHOD AS IT STANDS

**Canonical spec:** `results/cascade_methods/docs/current/METHOD_FINAL_2026-07.md` (numbers stale;
mechanism correct).
**Canonical code:** `src/cascade_methods/method_final.py`, plus
`method_final_mmmu_corrected.py`, `opentext_32b_think_full.py`, `f8_mode_vsthink_ci.py`.
**Canonical prose:** `paper/adaptive-cascade-medvqa_ieee_2026-07-08.tex`.

## 3.0 Models

- **Cheap leg:** Lingshu-7B.
- **Strong leg:** Lingshu-32B **in direct (no-reasoning) mode**.
- The 32B's *reasoning* mode is only the **baseline**, never a deployed tier in the Lingshu
  configuration — on Lingshu the reasoning gain is ~0, so a reserved reasoning tier would fire ~0%
  of the time. (On MedVLThinker and InternVL3 it would fire; see §5.1.)

## 3.1 Top level — the format router

Detect, **from the prompt text and never from the gold answer**, whether the question is
multiple-choice/closed or open-ended, and dispatch to the corresponding arm.

*Why a router rather than one unified gate:* the multiple-choice margin gate has no open-text
analogue, and the trained verifier is open-text-specific. A single unified 7B sequence-log-prob
gate loses to margin on multiple choice and to verifier confidence on open text
(`artifacts/integrated_method_vs_think.json:verdict.router_vs_unified`).

## 3.2 Multiple-choice arm

### compute-lean (default): `7B-direct → margin gate → 32B-direct`

- **Signal:** `margin` = P(top-1 option) − P(top-2 option), from the 7B's option log-probabilities.
  (`src/cascade_methods/integrated_method.py:115`.)
- **Rule:** escalate if and only if `margin < τ`.
  Code: `cascade_acc()` = `np.where(gate < tau, ok_strong, ok_cheap)`.
- **How τ is chosen:** `pick_tau_isocost()` returns the **minimum-escalation τ whose
  training-fold cascade accuracy is at least the strong leg's training-fold accuracy**, inside a
  deterministic modulo-5 cross-fit (`heldout(K=5)`).
- **Important:** τ is refit inside every fold. **No single global τ is stored anywhere.** Unlike
  the earlier MedVLThinker work — which froze one τ = 0.426 on a PMC-train sample and transferred
  it unchanged — this method has no materialized deployable threshold. See §7 hole 7.
- **Held-out escalation rates** (`artifacts/method_final.json`): PMC-VQA 8.5% (`test_2.csv`,
  n = 33,430), SLAKE-closed 20.5%, VQA-RAD-closed 57.0%, PathVQA-closed 45.7%, MedXpert 89.6%,
  MMMU 0%. Pooled 16.16%.

### accuracy-max: a guardrailed per-benchmark policy router

For each benchmark, pick the held-out **paired-bootstrap-certified** winner among
{always-32B-direct, keep-7B, fusion/veto}. The default is always-32B-direct; a deviation is
admitted only if its 95% paired-bootstrap **lower** bound exceeds 0, with a cost-aware tiebreak
(`src/cascade_methods/beat32b_fusion.py:f1_router_choice`, lines 246–258).
**Certified non-32B cells: PMC-VQA and MMMU only.** *(Every PMC-VQA number in this section and in §4
is the MedEvalKit track — v2 **`test_2.csv`**, n = 33,430. See §9.3.)*

- **Version 1 lever — confidence-advantage fusion on PMC-VQA**
  (`beat32b_fusion.py:confadv_fuse`): run BOTH legs; on the ~33% of items where they disagree, take
  the leg with the higher **isotonically calibrated** P(correct), cross-fit per fold; on agreement
  take the shared answer. This is a two-detector Chair–Varshney fuser. The classic
  *per-slice*-reliability version collapses to exactly always-32B (Δ = 0.0) — **the beat requires
  per-sample confidence.**
- **Version 2 lever — certified weak veto on PMC-VQA** (`beat32b_more.py:f8_veto`): cross-fit;
  partition PMC by training-fold 7B-confidence quintiles; **certify** a bin if the one-sided Wilson
  lower bound (z = 1.645, n ≥ 30) on the 7B's precision is at least the 32B's point accuracy in
  that bin. At test time, inside a certified bin **keep the 7B and never run the 32B**; otherwise
  take the 32B. Veto fires on **40.0%** of PMC, so the 32B runs on 60%. *(The "never worse by
  construction" language attached to this rule is not correct as stated — see §7 hole 11.)*
- **MMMU → keep-7B** in both shipped modes. This cell is contaminated and the paper excludes it.

## 3.3 Open-text arm — `7B best-of-N + trained verifier → 32B-direct`

- **Generator:** Lingshu-7B, N samples (fixed 8 in the reference implementation; adaptive N in the
  deployed point).
- **Selector:** the trained LoRA outcome verifier at `ckpts/train/lora_verifier_pooled4`. One
  forward scores all candidates; s = P("Yes") at the final token; pick the argmax. Trained with
  binary cross-entropy on judge-derived correctness labels. Per-answer AUROC **0.924** (n = 8,512).
- **Adaptive N** (`src/cascade_methods/integrated_pandora.py:pandora_open_arm`): Weitzman's
  optimal-search rule. Each "box" is either *draw one more 7B sample* (cost 2.0 compute-units =
  generation + verification; reward = isotonically calibrated verifier P(correct)) or *escalate to
  the 32B* (cost 4.57; reward = calibrated P(strong correct)). One λ knob yields both a
  stop-drawing reservation value and an escalation value. Fully nested 5-fold: calibration and λ
  are picked on the training fold as the minimum-compute λ that holds the training accuracy
  target, then frozen and applied to the held-out fold.
- **Escalation gate:** version 1 uses a verifier-confidence threshold (minimum escalation at
  parity). Version 2 uses a **team-objective learning-to-defer** rule
  (`beat32b_more.py:f10_l2d`): a cross-fit logistic regression over cheap-side features
  [verifier score max / range / mean / std, number of unique predictions, self-consistency,
  sequence log-probability] predicting P(7B best-of-N correct), with a threshold tuned on the
  training fold to maximise *team* accuracy. **Its improvement comes from the objective, not from a
  better signal** — its recoverability AUROC is not higher than the single verifier-confidence
  gate's.
- **Deployed adaptive draw counts** (`artifacts/method_final.json`): mean N = 3.45 (SLAKE-open),
  3.91 (VQA-RAD-open), 5.48 (PathVQA-open); escalation 15.8% / 12.5% / 35.7%.

## 3.4 A documented latency lever, not folded into the headline

**Parallel prefill prefetch**: run the 32B's image prefill concurrently with the 7B pass. The
measured prefill fraction is φ = 0.586, so the 32B prefill is 390 ms against the 7B's 347 ms — the
whole 7B pass hides inside it. Pooled latency **461.1 → 405.2 ms (−12.1%)** at identical accuracy.
**Cost caveat:** unconditional prefetch pays the 32B prefill on every query, pushing pooled compute
2.337 → 4.575 (≈ always-32B). A slice-gated variant (prefetch only where escalation ≥ 0.40) gives
429.8 ms at 2.492. (`artifacts/escalation_levers.json:G8_prefill_prefetch`.)

## 3.5 The cost model

One set of constants is used across the whole codebase (`integrated_method.py:53-57`,
`pandora_controller.py:49-50`, `paper_baselines.py:70-75` — three copies, values agree).

| symbol | latency | energy | compute-units |
|---|---|---|---|
| 7B direct greedy | 347 ms | 45.8 J | 1.00 |
| verifier forward (scores all N) | 175 ms | 25.3 J | 1.00 |
| best-of-8 (8 parallel generations + 1 verify) | 522 ms | 71.1 J | **16.0** |
| 32B direct — the strong leg *and* the honest "always-32B" | 665 ms | 127.0 J | 4.57 |
| 32B with reasoning — the naive baseline | 10,521.6 ms | 2,001.9 J | 4.57 † |
| both legs, co-resident | 665 ms parallel / 1,012 ms sequential | 172.8 J | 5.57 |

† The reasoning row's compute is deliberately set equal to the direct row's as a conservative
**lower bound** (a reasoning forward decodes 100–300 more tokens), so the reported figures
*understate* the baseline's cost.

- Compute-units = multiply-accumulate operations `F = 2·Θ·(P + G)` normalized to one 7B forward.
  **The 32B/7B ratio 4.57 appears in the code only as a hard-coded literal; no file derives it from
  parameter counts.** An older math document uses 7.6e9/33.0e9 = 4.34. See §7 hole 14.
- Expected per-query cost: `C = c₀ + e₀·c₁ + e₁·c₂` where the `e` are escalation rates.
- **Two latency accountings** are reported everywhere: *sequential* (single stream) and *parallel*
  (two co-resident GPUs — best-of-N batched, fusion legs concurrent).
- **A live inconsistency:** best-of-8 latency treats verification as ONE forward for all N
  (522 = 347 + 175) while its compute treats it as N forwards (16 = 8 × 2). Both cannot describe
  the same implementation. See §7 hole 14.

## 3.6 Scoring protocol

- Multiple-choice/closed → MedEvalKit exact-match; MMMU → judge-parsed output.
- Open text → LLM judge (`src/labeling/run_judge.py`, a neutral Qwen2.5-32B-based grader), because
  open-text exact match is known-broken in the harness (a gold of `"CT"` against a response of
  `"CT."` scores incorrect while ROUGE-1 ≈ 1.0).
- All thresholds are 5-fold cross-fit. All CIs are 10,000-resample **paired question-level
  bootstraps**. Calibration variance is *not* in any CI.

---

# 4. THE RESULTS

**Reporting convention.** The paper's suite is "Variant B": **MMMU excluded, 5 benchmarks, 8 cells,
n = 42,224.** Full-suite numbers (n = 42,374, MMMU included) are given where they exist.

## 4.1 Baselines (all measured)

Sources: `artifacts/method_final_mmmu_corrected.json`, `artifacts/paper_baselines.json`,
`artifacts/opentext_32b_think_full.json`, `artifacts/f8_mode_vsthink_ci.json`.

| system | accuracy | compute | latency (parallel) | energy |
|---|---|---|---|---|
| always-7B (cheap floor) | 0.5549 | 1.00 | 347 ms | 45.8 J |
| **always-32B with reasoning** (naive baseline) | **0.5591** | 4.57 | 10,521.6 ms | 2,001.9 J |
| always-32B direct | 0.5729 | 4.57 | 665 ms | 127 J |
| **oracle mode-select 32B** (per-benchmark best of the two — fairest strong baseline, not deployable) | 0.5730 | 4.57 | 860 ms | 164 J |

The oracle picks *reasoning* on only SLAKE-closed and MMMU; *direct* everywhere else.
**Note:** the measured 0.5591 exists only in `opentext_32b_think_full.json` and
`f8_mode_vsthink_ci.json`. `method_final_mmmu_corrected.json` still carries the superseded
estimate 0.5628 flagged `acc_is_estimated: true`, and the paper's Figure 1 is built from that file
— see §7 hole 14.

## 4.2 The headline

| operating point | accuracy | compute (rel.) | latency par. | energy | Δ vs 32B-reasoning (95% CI) | Δ vs oracle-mode (95% CI) |
|---|---|---|---|---|---|---|
| **compute-lean** | 0.5741 | 2.248 (**0.49×**) | **469 ms** | 83.6 J | **+0.0150 [+0.0107, +0.0192] SIG** | +0.0010 [−0.003, +0.005] n.s. |
| **accuracy-max (veto + learning-to-defer)** | 0.5836 | 4.257 (**0.93×**) | 731 ms | 136.7 J | **+0.0245 [+0.0216, +0.0274] SIG** | **+0.0106 [+0.0085, +0.0126] SIG** |
| accuracy-max⁺ (fusion variant) | 0.5862 | 5.712 (1.25×) | 668 ms | 177.1 J | **+0.0271 [+0.0237, +0.0305] SIG** | +0.0131 [+0.010, +0.016] SIG |

n = 42,224 throughout. Sources: `artifacts/f8_mode_vsthink_ci.json` (rows 1–2),
`artifacts/opentext_32b_think_full.json` (row 3).

Full-suite equivalents (n = 42,374): compute-lean +0.0154 [+0.0112, +0.0195]; accuracy-max
+0.0249 [+0.0219, +0.0278]; accuracy-max⁺ +0.0275 [+0.0241, +0.0308].

**Versus the fairer baselines** (`method_final_mmmu_corrected.json`, Variant B):
compute-lean vs always-32B-direct **+0.0011 [−0.0028, +0.0051] not significant**;
accuracy-max **+0.0107 [+0.0086, +0.0127] significant**.
Versus always-7B: compute-lean +0.0191 [+0.0167, +0.0214]; accuracy-max +0.0311 [+0.0276, +0.0346].

**Pool splits** (Variant B, accuracy-max, `f8_mode_vsthink_ci.json`):
multiple-choice only, n = 39,879, Δ vs reasoning **+0.0101 [+0.0073, +0.0128] SIG**;
open only, n = 2,345, Δ **+0.2699 [+0.2490, +0.2908] SIG**.
Compute-lean, multiple-choice only: **+0.0006 [−0.0037, +0.0048] not significant.**

## 4.3 Per-benchmark (compute-lean; Δ versus the oracle-mode-32B baseline)

Source: `artifacts/paper_baselines.json:paired_bootstrap_ci`.

| cell | n | 7B | 32B-direct | 32B-reasoning | method | policy (escalation) | Δ vs oracle (CI) | compute | latency seq/par |
|---|---|---|---|---|---|---|---|---|---|
| PMC-VQA *(`test_2.csv`, v2, unverified)* | 33,430 | 0.5427 | 0.5518 | 0.5494 | 0.5508 | margin cascade (8.5%) | −0.0010 [−0.0058, +0.0039] | 1.386 | 403/403 |
| SLAKE-closed | 836 | 0.8254 | 0.8589 | 0.8636 | 0.8517 | cascade (20.5%) | −0.0120 [−0.0299, +0.0060] | 1.935 | 483/483 |
| VQA-RAD-closed | 251 | 0.7809 | 0.8526 | 0.8406 | 0.8327 | cascade (57.0%) | −0.0199 [−0.0398, 0.0] | 3.604 | 726/726 |
| PathVQA-closed | 3,362 | 0.8409 | 0.8891 | 0.8891 ‡ | 0.8882 | cascade (45.7%) | −0.0009 [−0.0042, +0.0024] | 3.089 | 651/651 |
| MedXpert-MM | 2,000 | 0.2615 | 0.3065 | 0.3040 | 0.3005 | cascade (89.6%) | **−0.0060 [−0.0120, −0.0005] SIG BELOW** | 5.095 | 943/943 |
| MMMU-Medical | 150 | 0.8000 | 0.6333 | 0.6600 | 0.8000 | keep-7B | +0.1400 [+0.0533, +0.2267] SIG *(contaminated)* | 1.000 | 347/347 |
| SLAKE-open | 645 | 0.7364 greedy | 0.8186 | **0.6791** | 0.8093 | adaptive-N + verifier (15.8%, N̄ 3.45) | −0.0093 [−0.0372, +0.0171] | 7.622 | 1906/627 |
| VQA-RAD-open | 200 | 0.4650 | 0.6000 | **0.5450** | 0.5950 | adaptive-N (12.5%, N̄ 3.91) | −0.0050 [−0.060, +0.050] | 8.391 | 2124/605 |
| PathVQA-open | 1,500 | 0.3240 | 0.3760 | **0.1087** | 0.4520 | adaptive-N (35.7%, N̄ 5.48) | **+0.0760 [+0.0553, +0.0960] SIG** | 12.598 | 3100/759 |

‡ PathVQA-closed has **no 32B-reasoning dump**; the reasoning column is set equal to direct.

**accuracy-max changes only:** PMC (`test_2.csv`, n = 33,430) → fusion 0.5653 (Δ vs 32B-direct
**+0.0135 [+0.0100, +0.0169]
SIG**, compute 5.57) in version 1, or veto 0.5613 (Δ **+0.0095 [+0.0071, +0.0118] SIG**, compute
3.741, −32.8% versus fusion) in version 2; the other closed cells snap to always-32B-direct. The
learning-to-defer rule moves the open cells versus 32B-direct: SLAKE-open −0.0093 → **+0.0016**,
VQA-RAD-open −0.0050 → **+0.0050**, PathVQA-open +0.0760 → **+0.0860 [+0.064, +0.106]**. **Both
repaired cells' CIs still span zero** at n = 200–645; the only CI-certified open beat is
PathVQA-open.

## 4.4 The measured reasoning-baseline correction (2026-07-08)

`artifacts/opentext_32b_think_full.json`:

| set | n | measured reasoning | direct | Δ | prior estimate | estimate error |
|---|---|---|---|---|---|---|
| SLAKE-open | 645 | 0.6791 | 0.8186 | −0.1395 | 0.6236 | +0.0555 |
| VQA-RAD-open | 200 | 0.5450 | 0.6000 | −0.0550 | 0.4800 | +0.0650 |
| PathVQA-open | 1,500 | **0.1087** | 0.3760 | **−0.2673** | 0.2460 | −0.1373 |

Replacing the estimate with the measurement *raised* the compute-lean headline from +0.0117 to
+0.0154 (full suite).

## 4.5 What is load-bearing versus marginal

**Load-bearing 1 — the open-text verifier arm.** The 2,345 open items are 5.55% of the pool but
carry Δ +0.2699. `[derived]` 0.0555 × 0.2699 = **+0.0150 of accuracy-max's +0.0245**, and
0.0555 × 0.2597 = **+0.0144 of compute-lean's +0.0150 (Variant B)** — the multiple-choice half
contributes +0.0006 × 0.945 ≈ +0.0006. **Remove the open arm and compute-lean's entire
vs-reasoning win disappears** (multiple-choice-only Δ +0.0006, CI spans 0). This is the accuracy
engine.

**Load-bearing 2 — the PMC fusion/veto (accuracy-max only).** PMC (`test_2.csv`, n = 33,430) is
78.9% of the pool. Fusion lifts it 0.5508 → 0.5653, contributing **+0.0114 pooled**; the four other
closed cells snapping to always-32B add ≈ +0.0006. Remove it and accuracy-max collapses to
compute-lean. *(The 78.9% weight is a property of the MedEvalKit `test_2` pool only; in the
MedVLThinker-Eval pool the PMC cell — `test_clean.csv` — is 24.3%. §9.3.)*

**Load-bearing 3 — the margin gate and the direct-mode strong leg (the compute story).** The
multiple-choice arm's pooled compute is **1.739 versus 4.57** (0.38×) at 454 ms versus 10,522 ms.
Remove the gate (always escalate) and compute is 4.57, killing the 0.49× headline. Replace the
strong leg with reasoning mode and compute goes to 105% at 29.8 s.

**Load-bearing 4 — the baseline framing.** The vs-reasoning claim depends on the reasoning model
sabotaging itself (PathVQA-open 0.1087 versus 0.3760). Against 32B-direct or the oracle mode,
compute-lean is a **tie**, not a win.

**Marginal or near-free:**
- **Adaptive N:** pooled compute 2.538 → **2.244** at iso-accuracy (open arm 16.181 → 10.871,
  −33%). Removing it costs ~13% pooled compute and no accuracy.
- **Learning-to-defer:** ~+0.0006 pooled; its real value is qualitative (it flips two open cells
  from below-32B to above).
- **Veto versus fusion:** −0.0026 accuracy for −1.449 compute-units (−25%) and +63 ms parallel
  latency. A pure Pareto reshuffle.
- **MMMU keep-7B:** +0.140 on the cell but 0.35% of the pool → **±0.0005** on the sample-weighted
  headline. **But dominant in macro:** full-suite macro Δ 0.0777 → 0.0621, and multiple-choice-only
  macro collapses +0.0270 → +0.0036 if MMMU is escalated.
- **Prefill prefetch:** −12.1% latency at zero accuracy cost, not in the headline.
- Quantized strong leg: zero compute change. Image-token pruning: an unimplemented projection.

## 4.6 Reproduction path

| number | script → artifact |
|---|---|
| both knob settings, reconciliation, v1 + v2 levers | `src/cascade_methods/method_final.py` → `artifacts/method_final.json`, `method_final_v2.json` |
| MMMU-corrected Variants A/B × 3 modes, oracle-mode baseline, Pareto frontier | `src/cascade_methods/method_final_mmmu_corrected.py` → `artifacts/method_final_mmmu_corrected.json` |
| main baseline table, all paired CIs | `src/cascade_methods/paper_baselines.py` → `artifacts/paper_baselines.json` |
| measured 32B-reasoning open-text accuracy + re-derived CIs | `src/cascade_methods/opentext_32b_think_full.py` → `artifacts/opentext_32b_think_full.json` |
| **the current headline CI** | `src/cascade_methods/f8_mode_vsthink_ci.py` → `artifacts/f8_mode_vsthink_ci.json` |
| per-lever references | `integrated_method.py`, `beat32b_fusion.py`, `beat32b_more.py`, `integrated_pandora.py`, `pandora_controller.py`, `escalation_levers.py` |

**Does it run?** Verified this session, read-only: all 13 headline modules import cleanly;
`paper_baselines.build_cells()` completes in **41.6 s, CPU only**, returning all 9 cells at the
expected sizes (33,430 / 836 / 251 / 3,362 / 2,000 / 150 / 645 / 200 / 1,500 = 42,374). Every
declared input dump exists on disk.

**Launch discipline.** Modules use bare sibling imports (`import integrated_method as IM`), which
only resolve when a script is run as `python3 src/cascade_methods/<x>.py` **from the repository
root**. `ROOT` is hard-coded as `os.path.expanduser("~/medvlthinker-imgdiff-compute")`.

**A caveat that matters.** `method_final.py` is a **CPU re-costing of saved per-sample dumps**.
Nothing in the final method has ever been executed end-to-end as a live pipeline: escalation is
`np.where(margin < τ, ok_32B, ok_7B)` over recorded per-sample correctness, and latency/energy come
from per-leg batch-1 constants plugged into the expected-cost formula. `METHOD_MATH.md` calls this
"calibrated wall-clock", not a measured deployment. The one genuine live cascade in the repository
(`ckpts/rt_cascade_cap320.jsonl`) belongs to the older MedVLThinker work.

---

# 5. WHAT WE LEARNED THAT GENERALISES

These are the findings that should survive outside this project.

## 5.1 Reasoning hurts perception; on reasoning-heavy benchmarks it helps *some* families

> **Re-derived 2026-07-29 after a prompt-matching audit.** The arms behind the published 15/20 count
> were **prompt-unmatched** (and, for MedVLThinker, **resolution**-unmatched). This section now states
> the re-derivation from the best-matched arms already on disk. Sources:
> **`artifacts/finding1_corrected_2026-07-29.json`** (`src/cascade_methods/finding1_corrected.py`) and
> the audit that found the defect, **`artifacts/finding1_prompt_matching_audit.json`**.
> `artifacts/GENERALIZATION.md` carries the superseded version and is banner-marked. See §10.1 C20–C24.

Chain-of-thought is not free accuracy. **On perception-type medical VQA, generating a reasoning trace
*costs* accuracy relative to answering directly, in every family but one.** On reasoning-type
benchmarks it helps *some* families and not others — that half is **model-dependent, not universal**.

**The defensible statement.** Chain-of-thought reasoning does not pay for itself on perception-style
medical visual QA: on prompt- and resolution-matched arms, thinking is strictly worse than answering
directly in **17 of 20** (family × benchmark) perception cells across 5 medical VLM families —
**14/20** with 95% CIs excluding zero, pooled **−0.0401 [−0.0456, −0.0347]** over **30,250** paired
samples, **19/20** no better than +0.02 — and it reproduces at the same strength on arms that differ by
nothing but the reasoning instruction. On reasoning-heavy benchmarks CoT helps some families
(MedVLThinker-32B, MedGemma-27B, InternVL3-38B) but not others (Lingshu-32B, QoQ-Med-VL-32B).

**The corrected cross-family table** (best-matched arms on disk; Δ = think − no-think; **bold** = 95%
paired-bootstrap CI excludes zero):

*(Internal-harness cells: the **PMC** column is the human-verified v1 **`test_clean.csv`**,
n = 2,000 per cell — verified 2,000/2,000 against `test_clean.Answer_label`, **not** the
MedEvalKit `test_2.csv`. See §9.3.)*

| family | PMC | SLAKE | VQA-RAD | PathV | MMMU | MX-R | MX-U |
|---|---:|---:|---:|---:|---:|---:|---:|
| MedVLThinker-32B | −0.0075 | **−0.1274** | **−0.0846** | +0.0012 | **+0.0882** | **+0.0491** | **+0.0884** |
| Lingshu-32B | **−0.0425** | **−0.0649** | **−0.0919** | **−0.1017** | +0.0059 | +0.0000 | +0.0235 |
| QoQ-Med-VL-32B | **−0.0585** | −0.0144 | **−0.0662** | **−0.0523** | +0.0118 | −0.0131 | **−0.0433** |
| Chiron-o1-8B | **−0.0680** | **−0.1010** | **−0.1103** | **−0.0654** | +0.0294 | +0.0021 | +0.0273 |
| MedGemma-27B | −0.0135 | +0.0144 | **−0.0735** | **+0.0413** | +0.0353 | +0.0263 | **+0.0830** |

Per-family pooled perception Δ (sample-weighted, n = 6,050 each): MedVLThinker **−0.0144
[−0.0261, −0.0030]**, Lingshu **−0.0792 [−0.0902, −0.0681]**, QoQ **−0.0524 [−0.0640, −0.0407]**,
Chiron **−0.0707 [−0.0840, −0.0577]**, MedGemma **+0.0162 [+0.0028, +0.0298]**.

**Robustness.** The perception half does not depend on which correction policy you pick: three
independent policies (best-matched arm; strict resolution *and* format matching; strict with
MedVLThinker matched at full resolution instead of cap320) **all** give 17/20 strictly negative and
19/20 within +0.02, with pooled Δ between −0.0401 and −0.0405. **The as-published 15/20 is the
outlier — every better-matched pairing is stronger.** On the *fully-matched-only* subset, where nothing
is left to correct (Chiron + MedGemma with the shared peer think instruction: same absent system
message, same image budget, answer-format constraint in both arms): **6/8** strictly negative, pooled
**−0.0273 [−0.0367, −0.0176]** (n = 12,100). The two non-medical peer architectures were already fully
matched: **7/8** strictly negative (InternVL2.5-8B pooled −0.0076 [−0.0208, +0.0056]; Phi-3.5-Vision
**−0.0187 [−0.0336, −0.0036]**); their dumps are in
`ckpts/acc_gen/{internvl25_8b_think,phi35v_think}/`, 4 files / 6,050 rows each, perception only.

**Two cells flipped sign** versus the published table: MedVLThinker PMC-VQA +0.0055 → **−0.0075** and
Lingshu PMC-VQA +0.0115 → **−0.0425** (both on `test_clean.csv`, n = 2,000).

**The one real exception.** MedGemma-27B on PathVQA is a genuine, significant perception win for
reasoning: **+0.0413 [+0.0220, +0.0607]**, p = 0.0000, on a **fully matched** pair. It survived the
removal of the think-only persona that produced the published +0.0399, so it is a real exception, not a
prompt artifact — and it is the **only** one in the corrected table.

**The reasoning half, honestly.** 12 of 15 reasoning cells are point-positive, but only **4/15** have a
95% CI excluding zero and **1/15 is significantly negative**. Per family: MedVLThinker-32B **survives
and improves** (3/3 significant; MMMU +0.0647 → **+0.0882**, MX-R +0.0491, MX-U +0.0884); MedGemma-27B
**partially survives** (3/3 positive, 1/3 significant; its MMMU cell flips −0.0118 → **+0.0353** once
the persona is removed from the think arm); Chiron-o1-8B is directionally positive 3/3 but **no cell
reaches significance**; **Lingshu-32B and QoQ-Med-VL-32B do not support it at all** (see the
withdrawals). Corroborated on a second, independent harness (MedEvalKit) by MedVLThinker-32B (MMMU
+0.100 [+0.027, +0.173]; MX-R +0.046 [+0.020, +0.073]; MX-U +0.042 [+0.004, +0.079]) and InternVL3-38B
(MMMU +0.120 [+0.047, +0.193]; MX-R +0.035 [+0.012, +0.057]; MX-U +0.020, n.s.).

**Two withdrawals.**
- **All 7 Lingshu-32B published cells, both directions.** The "native think" instruction
  (`runners/run_native_think.sh:7`) is purely an answer-**format** string with no reasoning trigger;
  measured generated tokens are **3.0 vs 3.0–3.3**, i.e. the model never produced a chain of thought.
  Its repaired genuinely-reasoning arm (`ckpts/acc_gen/lingshu32b/think_fullres`, 150–259 tokens) says:
  perception — reasoning **hurts**, 4/4 strictly negative, all CIs excluding zero, pooled **−0.0866
  [−0.0972, −0.0757]** under the resolution-matched pairing; reasoning — **nothing** (MMMU +0.0000,
  MX-R +0.0048, MX-U +0.0271, none significant). This is a *conservative* reading: the replacement think
  arm carries an expert persona the direct arm lacks, an asymmetry that favours thinking, and thinking
  still loses. **Lingshu-32B must not be cited as evidence that reasoning helps reasoning-heavy VQA.**
- **QoQ-Med-VL-32B as reasoning-side evidence.** Its headline MMMU gain is a prompt artifact:
  +0.0706 → **+0.0118** (CI [−0.0588, +0.0824], n = 170) matched, and +0.0000 fully matched. It never had
  a significant MedXpert gain, and MedXpert-Understanding is significantly **negative** (−0.0433,
  p = 0.022).

**Costs.** For MedVLThinker, turning reasoning off on perception questions is nearly a **50×** latency
win at *better* accuracy; the measured reasoning:direct batch-1 ratios are MedVLThinker 49×, MedGemma
45×, QoQ 43×, Chiron 15×. **Lingshu's often-quoted 1.2× is not a reasoning:direct ratio** — it was
measured on the withdrawn native-think arm that generated 3.0 tokens, i.e. it is the cost ratio of two
answer-format prompts. No matched Lingshu reasoning:direct latency ratio has been measured on the MCQ
harness. (Lingshu *does* reason when the prompt actually triggers it — see C7 — and on open text a real
Lingshu-32B think pass costs 10.5 s vs 0.665 s, ≈16×.)

**Two precision caveats that must travel with the count.** (1) 17/20 is a **count of signs**, not a
measurement: per-cell n runs from 170 (MMMU) to 3,362 (PathVQA), so at n = 170 a 95% CI is roughly
±0.07 and near-zero cells could flip on resampling alone. Always report the count together with the
pooled Δ and the CI-significant subcount (14/20). (2) On MCQ the only channel an unmatched prompt has
is answer-**extraction** failure — gold is a single letter graded by exact equality — and that channel
was measured at 0–3.4% per cell; an adversarial correction that charges every extraction failure
against the finding changes **no** count.

**Still broken, and not repairable offline.** The **open-text** think-vs-direct comparison
(`src/labeling/run_openvqa.py:26/27`) compares a persona + "short, specific phrase / Do not explain"
direct prompt against a `<think>`-trace prompt that drops both. On free text that is a live
**style/length grading channel**, not a bounded extraction channel. A matched-prompt re-run is in
flight; until it lands, **the open-text half of Finding 1 is provisional** — including the
`Δ = −0.154` Lingshu-32B open-perception figure and the per-set 0.679 / 0.545 / 0.109 values.

**One further datum that should be better known.** On the *only* cell where genuine 32B
multiple-choice reasoning was measured on the faithful harness, 100× the generated tokens bought
nothing: MedXpert reasoning 0.3040 versus direct 0.3065 at 320 versus 3 generated tokens
(`MedEvalKit/eval_results_lingshu32b_reason` vs `..._full`; the post-edit `_reason` dump, which does
reason but is format-unmatched — §10.5. Per split: MX-R −0.0035 [−0.0284, +0.0208], MX-U +0.0000
[−0.0397, +0.0415]).

## 5.2 Answer format determines whether routing signals work at all

The same model, the same images, the same questions — switching from 4-option multiple choice to
free text moves routing AUROC from ~0.6 to ~0.87.

- Multiple choice: cheap-wrong detection AUROC ~0.66–0.73; recoverability ~0.578–0.6.
- Open text: cheap-wrong detection **0.866**, recoverability **0.804**; pooled 3-dataset
  **0.846 [0.830, 0.862]**; holds under an LLM judge and on a 4th imaging modality.
- **The cause is 4-option discreteness, not answer length** — open answers have median 1–2 tokens.

**Practical implication:** if your uncertainty signals look saturated, check whether your *benchmark
format* is the reason before concluding your model is uninformative. And do not transfer a
"confidence is saturated" conclusion from a multiple-choice benchmark to a generative deployment.

**Corollary that must travel with it:** **detection ≠ cascade gain.** Detection AUROC rose
0.66 → 0.85 while the oracle-minus-cheap headroom moved only +0.02 (MCQ) → +0.06 (open). One
dataset has 7B 0.302 ≈ 32B 0.301 (zero headroom) at detection AUROC 0.749. Knowing an answer is
wrong is not the same as having somewhere better to send it.

## 5.3 Sampling helps only where the answer space is open, and only with a trained selector

- On free text, best-of-8 with a **trained** verifier captures **49%** of the oracle gap
  (0.413 → 0.501 against oracle 0.592). On bounding boxes, 40% (organs) and **78%** (chest X-ray).
- On multiple choice, generative verification **craters** (content mode 0.132 vs 0.534) — the
  signal lives in the option set, and there is nothing left to verify once you remove it.
- Every **training-free** selector sits at the random-pick floor: on one open set, random 0.720,
  self-verification 0.715 (*below* random), self-consistency majority 0.736, big-model listwise
  0.758 — and running the big model once scores 0.819.
- The mechanism is the **majority trap**: on open text the correct answer is a *minority* vote in
  **74–90%** of recoverable questions (mean ~1.5 of 8). Majority voting is structurally the wrong
  aggregator here.

## 5.4 Training, not size, is the active ingredient in verification

Confirmed three independent times:
- trained 7B verifier **0.501** vs zero-shot 32B verifier **0.357** (June 26);
- trained 7B **0.403** vs zero-shot 32B **0.355** (July 1);
- 32B zero-shot **0.480** vs trained 7B **0.475**, Δ +0.005 [−0.023, +0.032] not significant, n=600
  (July 6, `artifacts/verifier_32b_gpu.json`) — a 7× larger verifier merely *ties* a small trained
  one, while beating an equally-untrained 7B by +0.067.

## 5.5 The two limits, and how many independent methods confirmed each

**Limit A — the recoverability wall.** "Will the strong model fix *this* error?" is ~0.5–0.6 AUROC
from anything cheap. **Sixteen independent mechanisms hit it**: hidden-state probes, kNN
neighbourhoods, self-verification, gradient-boosted gates, rich fused features, the published
post-hoc deferral rule, the verification-augmented deferral router, cross-family and CLIP-style
image routers, a full LoRA fine-tune, trained open-text gates, logit-level fusion, decision-level
fusion, super-learner ensembling, learned slice discovery, credibility shrinkage, and the
open-ended cascade.

**Limit B — the selection wall.** A verifier converts only **74–82%** of oracle-of-N. **Thirteen
independent attempts** hit it, killed three orthogonal ways: *capacity* (a 7× bigger verifier
ties), *compounding* (diverse generation × pairwise comparison do not stack, −0.0117), and
*pre-filtering* (no filter beats both baselines). A stratified check killed the "compound answers"
excuse: selection efficiency is 79% on short answers (the bulk), 90% medium, 80% long.

**They hand work to each other.** Raising oracle coverage (+0.110 from diverse generation) does not
raise accuracy (+0.015 converted) — it *relocates* the residual from the coverage limit into the
selection limit. Only attack coverage if you have already broken selection.

**A measurement worth carrying forward** `[derived from ckpts/train/lora_verifier_pooled4/perq_sc8.json]`:
of 1,064 held-out questions, **434 (40.8%) have no correct answer anywhere in the 8-sample pool**,
while the entire selection gap is 97 questions (0.0912). The coverage wall is **4.5× larger** than
the selection wall. Any future effort should be sized accordingly.

## 5.6 The right proxy for gate quality is recoverability, not detection

Optimising a gate for "is the cheap model wrong" optimises the wrong objective. The gate with the
top detection AUROC (0.693) had cascade quality *below* the plain margin (Δ −0.0015); across gates,
correlation with cascade quality is **r ≈ +0.65 for recoverability and r ≈ −0.21 for detection**.
This single mis-specification invalidated a prior gate ranking in this project and is easy to
repeat elsewhere.

## 5.7 Pooled metrics manufacture wins; per-benchmark guardrails kill them

The deferral router beat the confidence gate by 7 points of escalation *pooled*, and lost by 27
points once a "never worse than always-cheap on any benchmark" guardrail was imposed (91.6% vs
64.3%). Confidence gates are intrinsically per-benchmark-safe because they escalate on uncertainty.
Later, a simple CI-lower-bound guardrail beat a purpose-built actuarial shrinkage estimator at
controlling thin-slice overfit (0.25 versus 6.62 violations per split).

## 5.8 State which cost axis you are winning on

Several apparent wins in this project existed on exactly one of {compute, latency, energy, VRAM}
and were losses on another:
- best-of-N is compute-dominated (16 versus 4.57) yet nominally latency-cheaper at batch 1;
- INT4 quantization moves VRAM and energy but **not** the multiply-accumulate count, and only ~12%
  of latency, because the direct-mode strong leg is *prefill*-bound;
- prefill prefetch trades compute for latency roughly 1:1.

## 5.9 Report the degenerate cells

Where the cheap model already matches or beats the strong one, the entire method space collapses:
on three of five families every method — including *random* — produces identical numbers at 0%
escalation. Where both models are near chance (MedXpert), the mirror image happens: the gate
escalates ~everything and the cascade costs 143–151% of always-strong. A comparison table in which
half the cells are identical is telling you the premise is not met, not that your gate is good.

## 5.10 Simulation cannot manufacture signal that the simulated component lacks

A simulated pairwise verifier — with preferences derived as σ(logit sᵢ − logit sⱼ) from the
pointwise scores — produced a confident negative (−0.003, ceiling +0.000). A **real** A-vs-B
forward pass overturned it the same day (+0.036 [+0.016, +0.055]). This is a live warning about
every other CPU-only re-simulation in the July work.

---

# 6. THE NEGATIVE-RESULTS CATALOG

~90 distinct failed attempts, grouped by the **principle** that killed them. Each entry has its
killing number. Cross-references: `docs/archive_mcq/FINDINGS.md`, `artifacts/*.json`,
`progress/progress_*.md`.

**How to use this.** Before proposing anything, find the principle it would have to beat (§6.1–6.9
headers), then check whether your idea is already in the list. If it is, the burden is to say
what is different about your version, quantitatively.

## 6.1 Killed by the recoverability wall (~0.5–0.6 AUROC)

| # | attempt | killing number |
|---|---|---|
| N1 | Maximum softmax probability / Chow's rule as gate | parity at 61.9% compute but **breaks the per-benchmark guardrail** |
| N2 | Predictive entropy | 30.7% compute but **misses parity AND breaks the guardrail** |
| N3 | Gini / DOCTOR score | 59.1%, guardrail break |
| N4 | Conformal CP-Router (prediction-set size) | a faithful implementation **over-escalates 69–80%** and loses to plain confidence |
| N5 | Cross-resolution agreement ensemble as primary gate | agreement alone **AUROC 0.566**; fused all-signal gate 60.3% (worse than single margin on the 5-benchmark pool) |
| N6 | Hidden-state probe (layer-14, PCA-128) | correctness 0.60 vs log-probs 0.68; **recoverability 0.53 vs 0.63** — worse than the log-probs |
| N7 | Compute-elasticity / resolution-trajectory signals | no gain (family 8 of 9 in the exhausted search) |
| N8 | Vision-sensitivity (blank-image counterfactual) | 56.9% of answers unchanged, but insensitive items are **no less accurate** (0.620 vs 0.625) |
| N9 | Visual-stability gate on Lingshu | **inert**: the 7B is 98.95% cap320-vs-full stable |
| N10 | Cross-model agreement as an MCQ ranker | **worst ranker, AUROC 0.6565** vs margin 0.7254 — and it needs the 32B forward |
| N11 | Two-signal AND-gate | no gain; at matched escalation −0.0024 / −0.0080 |
| N12 | kNN neighbourhood-recovery gate | `knn_beats_margin = false` on **0 of 5** datasets |
| N13 | Learned gradient-boosted gate on log-prob features | highest detection AUROC 0.681, **lowest cascade quality 0.6407** |
| N14 | Learned rich-feature gate | top detection **0.693**, cascade quality 0.6430 < margin's 0.6445 |
| N15 | Published post-hoc deferral (Jitkrittum Diff-01/Diff-Prob) on MCQ | signal ceiling 45–57% compute, worse than raw margin's 38.9%; the in-distribution gain did not transfer |
| N16 | Verification-augmented deferral router | **guardrail reversal 91.6% vs 64.3%**; token volume 0.98×; energy +20% worse; its one novel claim moves AUROC −0.003 |
| N17 | Cross-family complementarity router | oracle union 0.801, learned router **0.621 ≈ always-7B**; image-based recoverability **AUROC 0.50 = chance** |
| N18 | LoRA-trained stability router | **AUROC 0.7226 < logistic-on-scalars 0.7328** |
| N19 | Trained gates in the open-text cascade | gradient-boosted 0.861 vs verifier-confidence 0.853 — no improvement; cheap-only trained gates 0.69–0.73 |
| N20 | Trained recoverability gates (open text) | ranking AUROC 0.604 → 0.649 but **cascade accuracy goes DOWN** (0.421 → 0.413) |
| N21 | A new expected-gain gate (EG-RC) | wins one regime (0.232 vs 0.198) but is beaten there by the literal published baseline (0.271) and loses on another dataset |
| N22 | Learned error-slice discovery (Domino/Spotlight) | genuinely-new slices **1.62/split, BELOW the permutation null of 5.61** |
| N23 | Credibility (Bühlmann–Straub) shrinkage of thin slices | 7.5 → 6.62 violations, versus the deployed CI-lower-bound rule's **0.25** |

## 6.2 Killed by the selection wall (74–82% conversion)

| # | attempt | killing number |
|---|---|---|
| N24 | More verifier training data / epochs | selection 0.5009 → 0.5056 (noise) |
| N25 | Within-question ranking / Bradley–Terry loss | per-answer AUROC 0.903 → **0.933** but selection **FLAT** (0.5009 / 0.4991) |
| N26 | A bigger (32B zero-shot) verifier | 0.355 vs trained-7B 0.403; later 0.480 vs 0.475, Δ +0.005 n.s. |
| N27 | Selection-rule tweaks (weighted mean, frequency blending) | argmax ≈ weighted mean; **adding answer frequency HURTS** |
| N28 | Unified generative verifier on multiple choice | content mode **0.132 vs 0.534**; letter-mode gain sign-flips under the as-run parse |
| N29 | Simulated active pairwise comparison | mean −0.003; round-robin ceiling **+0.000** *(later overturned by real pairwise — see §10)* |
| N30 | Compounding pairwise over diverse generation | **−0.0117**; both-levers gain not significant |
| N31 | Distractor pre-filtering (8 filters) | best +0.0108 n.s., sign-flips per dataset |
| N32 | Zero-shot self-verification as selector | below greedy (0.319 vs 0.343) and **below random** (0.715 vs 0.720) |
| N33 | Spatial self-consistency (medoid) for boxes | at/below greedy: 0.164 vs 0.197; 0.053 vs 0.041 |
| N34 | Zero-shot box verifier | 0.177 ≤ greedy 0.197 |

## 6.3 Killed by the luck floor (the oracle gap is sampling luck, not latent knowledge)

| # | attempt | killing number |
|---|---|---|
| N35 | **Single-model routing** (the ancestor) | oracle sat **~29σ BELOW the random-allocation floor** |
| N36 | Self-consistency / majority voting on open-ended medical VQA | flat on one set (0.736 vs 0.730), **hurts** on another (0.324 vs 0.343); the correct answer is a minority vote in **74–90%** of recoverable cases |
| N37 | Big-model listwise re-ranking of cheap candidates | 0.758 — captures 24% of the gap above random, far below the big model's own single pass 0.819 |
| N38 | Candidate-conditioned synthesis | **0.774 vs 0.819 free-generation (−0.045)** — the majority trap drags the strong model down |
| N39 | Unsupervised Dawid–Skene aggregation | ≈ plain majority (−0.013); reliability tracks **self-agreement 0.52, not accuracy 0.29** |
| N40 | Cross-family agreement as an accuracy selector | collapses to "trust the stronger model"; the weak model is right when it dissents only 28.9% of the time |
| N41 | Structured grounding as the escape | a *competent* grounder gives **AUROC 0.557 (chance)**; the 0.816 result was an artifact of an incompetent grounder |

## 6.4 Killed because the errors are capacity-bound

| # | attempt | killing number |
|---|---|---|
| N42 | Distilling big-model competence into the cheap leg | **+0.000 / +0.007**; redistributes (PathVQA +0.099, VQA-RAD −0.086) rather than lifting |
| N43 | Test-time adaptation of the cheap leg | entropy-min/SHOT **−0.1591 (collapse)**; label-informed oracle ceiling **<1% and mixed-sign**; temperature scaling exactly 0 |
| N44 | Multi-resolution majority-vote test-time augmentation | +0.0034 for a 4-pass cheap leg — real but not cost-effective |
| N45 | Few-shot in-context learning to align answer style | **0.343 → 0.203** |
| N46 | Resolution as an intermediate tier (7B at full res) | on the escalate set **0.5229 vs 0.5260** — no better; fixes 12.6% of errors vs the 32B's 45.2% |
| N47 | The multi-action repair ladder | 43% compute vs the 2-rung baseline's 39% at parity; every harvesting rule lands 0.608–0.626 vs the 32B's 0.645 |
| N48 | Knowledge augmentation / retrieval | the 32B fixes genuinely-unknown errors **equally** across knowledge (38%) and perception (36%) types ⇒ not a knowledge gap |
| N49 | The "big-gap" test with a 3B cheap leg | there is **no gap at any size**: 3B 0.457 ≈ 7B 0.419 ≈ 32B-direct 0.498 |

## 6.5 Killed by fusion having nothing new to certify

| # | attempt | killing number |
|---|---|---|
| N50 | Trained fusion head over both legs' logits | captures 4 / 14 / −8 / −8 / −2% of the union-oracle headroom; **collapses on leave-one-family-out** (0.242) |
| N51 | Logit-level / full-posterior fusion (4 combiners) | **every combiner certifies exactly ONE cell and it is the contaminated MMMU**; contrastive decoding certifies **zero** |
| N52 | Classic per-slice-reliability Chair–Varshney fusion | collapses to **exactly always-32B (Δ = 0.0)** |
| N53 | Super-learner + decision-level Bayesian model averaging + certified veto, as slice extenders | pooled positive but **new certified cells: none, for all three** |
| N54 | Double reading with the reasoning model as arbiter | the *free* confidence-advantage arbiter (0.4116) beats **32B-with-reasoning (0.3871)** |

## 6.6 Killed by the guardrail (pooled wins hiding per-benchmark damage)

Covered above as N1–N3 and N16, plus:

| # | attempt | killing number |
|---|---|---|
| N55 | Self-verification rescue, and doubly-robust rescue | both **violate the SLAKE guardrail** (0.7380 / 0.7476 vs always-cheap 0.762); the doubly-robust version **saves no compute** (Δ −0.6%) |
| N56 | Bidirectional fragility-escalate | only 105 of 2,409 items qualify; escalating them changes accuracy by **+0.0000** |
| N57 | Visual-stability rescue folded into the better cascade | the control kills it: at matched skip rate, **random 0.65 s beats resolution-stability 0.95 s**; stable items benefit *more* from reasoning |

## 6.7 Killed by cost accounting

| # | attempt | killing number |
|---|---|---|
| N58 | Diverse generation as a deployable lever | not on the compute-Pareto envelope: at accuracy 0.55, iid costs 3.3 and diverse 6.2 |
| N59 | Best-of-N as the deployable method | compute-dominated (16 / 30 versus always-32B's 4.57); does not beat always-32B on accuracy (~0.549 vs 0.673) |
| N60 | Cross-model candidate pooling with clever allocation | pooling lifts oracle +0.113…+0.150 but yields **no frontier gain**; Markowitz ≈ uniform (Δ +0.002, negative on one dataset) |
| N61 | Bandit / adaptive per-question allocation | best held-out **Δ = +0.002**; the de-biased oracle ceiling **inverts at B=8 (−0.026)** |
| N62 | Diversity *selection* (DPP/MMR) over a fixed candidate set | oracle-of-8 **identical (0.412)** across all orderings — by construction no ceiling lift |
| N63 | Correlated adaptive search at the best-of-8 target | coverage 9/11 → **7/11**, compute 11.74 → 12.31 |
| N64 | INT4 quantized strong leg | **not a compute lever** (precision-independent MAC count); latency only 665 → 582.7 ms (−12%) because the leg is prefill-bound |
| N65 | Recoverability suppressor | −0.0018 pooled accuracy to buy one cell's latency; net recovery per escalation is positive everywhere |
| N66 | Unconditional parallel prefill prefetch | latency −12.1% but compute **2.337 → 4.575 ≈ always-32B** |

## 6.8 Killed by degeneracy / regime

| # | attempt | killing number |
|---|---|---|
| N67 | Three-tier cascade on the 6-benchmark pool | parity at **75.7% compute — worse than the deployed 73.6%** |
| N68 | Cascading on MedXpert at all | **143% / 151% compute** of always-reasoning at matched accuracy |
| N69 | Cascading on three of five families | cheap leg already ≥ always-big-reasoning ⇒ every method escalates 0% and all numbers are identical |
| N70 | A selective reasoning tier for Lingshu | no accuracy headroom. *(Verdict unchanged but re-grounded 2026-07-29: the original justification also cited a "reasoning:direct latency ratio of only 1.2×", which was measured on an arm that never reasoned — C22. With a genuinely reasoning Lingshu arm the tier is still pointless, because the reasoning gain is ~0 everywhere (MMMU +0.0000, MX-R +0.0048, MX-U +0.0271, none significant) while the cost is now real, i.e. the case against the tier is **stronger**, not weaker.)* |
| N71 | Resolution-decoupled reasoning tier | per-tier saving is real (−28% prefill) but the cascade is unchanged because the tier fires ≤14% and is decode-bound |
| N72 | Adaptive reasoning-trace length / early exit | **dead**: the answer marker is at median **0.99** of trace length over 8,220 traces |

## 6.9 Blocked by infrastructure or missing data (not conceptual failures — keep separate)

| # | item | status |
|---|---|---|
| N73 | OmniMedVQA 32B/38B strong leg | deterministic two-GPU NCCL hang; ~2 days of mitigations all failed; single-GPU impossible (64 GB weights on an 80 GB card). Cheap leg reproduces the paper (0.8274 vs 0.829). No pooled 7-benchmark figure exists. |
| N74 | Two concurrent cheap legs | container cgroup OOM (~245 GB anonymous RSS) → strictly sequential driver |
| N75 | InternVL3 faithful evaluation | blocked by a harness wrapper bug; not pursued (confirmatory) |
| N76 | Semantic escalation cache | **data-absent**: no image hash in any dump; duplication ≈0 exactly where escalation is heavy |
| N77 | Generated-token early exit | **data-absent**: both legs emit ~3 generated tokens on every benchmark (prefill-bound); the real lever needs intermediate-layer logits nobody dumped |
| N78 | 32B image-token pruning | **deferred with a plan**: projection −26% compute at p=0.50, but no vLLM mid-network token drop and 3D positional re-indexing needed; the measured resolution-cap analogue is free on PMC (0.543 → 0.542, −0.001; `test_clean.csv`, n = 2,000) and **costs on radiology** (VQA-RAD −0.040) ⇒ a radiology guardrail is mandatory |
| N79 | INT4 latency/accuracy | **projection only** — a CDN outage stalled 2 of 6 quantized shards; the quantized checkpoint on disk is the non-medical base model, not Lingshu |
| N80 | Lossless speculative decoding | infeasible on vLLM 0.10 (draft-model speculative decoding rejected) |
| N81 | Datasets not adopted | Quilt-VQA (gated), Medical-Diff-VQA / MIMIC-CXR-VQA / EHRXQA (PhysioNet-gated), OmniMedVQA-as-open (it is multiple choice), PubMedVision / MedTrinity (training corpora), ROCOv2 / pmc_oa (captioning) |
| N82 | Metrics surveyed and rejected | GREEN, RaTEScore, RadGraph-F1, CheXbert-F1, RadCliQ — all chest-X-ray *report* metrics, not general medical-VQA metrics |
| N83 | PathVQA-open once dropped as unscoreable | exact match 0.058 — this was a **scorer bug**, recovered by the validated LLM judge |

## 6.10 Out of scope by standing rule

An abstention / clinician-referral direction was built and validated in June and is **permanently
forbidden** as of 2026-07-07. Its files still exist (`src/cascade_methods/selective_abstain.py`,
`abstain_calibration.py`, `triage_3way.py`, `deferral_curve.py`, `methods_deferral.py`,
`lingshu_deferral_apgr.py`; four artifacts; `docs/archive_mcq/SELECTIVE_ABSTENTION.md`;
`paper/figs/open/fig_triage.png`). They are **historical record only and must not be revived as a
direction.** The paper is clean on this ("The method always returns an answer"). The certified veto
is *not* abstention — it keeps the cheap model's answer.

## 6.11 Prior-art findings (novelty is not a result)

Adversarial prior-art checks retired several claimed contributions: the cross-model agreement gate
is Agreement-Based Cascading (arXiv 2407.02348); the deferral router's core is Jitkrittum et al.
(arXiv 2307.02764); invariance-as-confidence is Bahat & Shakhnarovich; the internally-named
"CASP" stability method collides with the protein-structure CASP and its real prior art is CCPS
(arXiv 2505.21772). One composition (a resolution-decoupled reasoning tier) was verified novel and
**showed no cascade-level improvement** — a novel composition that improves nothing is still a
negative result.

## 6.12 What survived

Exactly four things beat their baselines and held up:
1. **The compute-configuration axis** — turn reasoning off on perception; insert the big model's
   direct mode as a middle tier (reasoning escalation 86% → 19%, latency 29.8 s → 5.9 s at matched
   accuracy).
2. **The format-aware router** — the final method, §3–§4.
3. **The trained outcome verifier** for free text and bounding boxes — 49% / 40% / 78% of the
   oracle gap. The *only* thing that broke the luck floor.
4. **The optimal-search adaptive-N controller** for tight compute budgets (−27% compute held-out at
   iso-best-of-8).

Everything in §6.1–6.9 is what it cost to find those four.

---

# 7. HOLES AND WEAKNESSES

Ranked by severity. Every hole is stated as it is, not as it could be spun.

## Critical

**Hole 1 — The "always-32B-with-reasoning" baseline is a NO-reasoning run on ~90% of the pool,
charged at reasoning cost.**
The dump used as the reasoning baseline for PMC-VQA (`test_2.csv`, n = 33,430), SLAKE-closed and
VQA-RAD-closed
(`MedEvalKit/eval_results_lingshu32b_think`) has mean generated tokens of **3.09 / 4.18 / 4.26**,
and agrees with the plain direct-mode run on **92–94%** of predictions. It was produced by the
harness's `--reasoning True` flag under the *old* prompt, which only appends "put the letter in
\boxed{}". PathVQA-closed has **no reasoning dump at all** and is imputed as reasoning = direct —
yet the field is named `acc_32b_think_measured`. Genuine 32B reasoning exists only for MedXpert,
MMMU and the three open cells (4,345 of 42,224 = **10.3%** of the pool). Meanwhile
`paper_baselines.cost_fixed` charges 10,521.6 ms / 2,001.9 J uniformly to all nine cells.
`UNIFIED_METHOD_EXPERIMENTS.md:557-559` already declared those dumps invalid and "excluded" — and
`integrated_method.py:102-103` loads them anyway.
*Honest re-costing* `[derived]`: charging reasoning price only where a real reasoning run exists
gives a baseline of ~1,679 ms / ~320 J, so **−95.5% latency / −96% energy becomes ~−72% / ~−74%**.
*Fix:* either re-run the 32B with the working reasoning prompt on the four closed cells, or drop
the vs-reasoning framing entirely and headline versus always-32B-direct and the oracle mode, where
the numbers are already sound. Note that the one genuine multiple-choice reasoning measurement
(MedXpert, 0.3040 vs 0.3065 at 320 vs 3 tokens) argues the re-run would change little, which makes
the cheap fix the right one.
*Root cause identified 2026-07-29 (§10.5):* the "old prompt" was **upstream MedEvalKit**, whose reason
arm carried the direct arm's answer-format clause and **no reasoning trigger**. A local uncommitted edit
on 2026-07-02 added the trigger but **deleted** that clause, which is why the post-edit `*_reason` dumps
reason yet are format-unmatched. Repairing this properly means appending the trigger instead of
replacing the clause — a decision for the researcher, since `MedEvalKit/` is a protected dependency.

**Hole 2 — 89% of the headline delta comes from 2 of 8 cells; against the deployable baseline,
5 of 8 cells contribute exactly zero.**
`[derived from f8_mode_vsthink_ci.json]` Contributions to +0.0245: PathVQA-open +0.01255 (51%),
PMC-VQA +0.00942 (38%; `test_2.csv`, n = 33,430), SLAKE-open +0.00216 (9%), everything else
≤ +0.00028. Against
always-32B-direct the per-cell deltas are +0.0095 (PMC, `test_2.csv`), +0.0016 / +0.0050 / +0.0860 (open), and
**exactly 0.0000 for SLAKE-closed, VQA-RAD-closed, PathVQA-closed and MedXpert** — because the
method *is* always-32B-direct there. Pooled vs direct: +0.0106.
Additionally, and reported nowhere: `method_final_mmmu_corrected.json` Variant B records a
**significant macro loss** for compute-lean — multiple-choice-only macro versus oracle-mode
**−0.0080 [−0.0138, −0.0025], significant**, and versus always-32B-direct **−0.0070 [−0.0125,
−0.0017], significant**. "Pareto-dominates" is a *sample-weighted* statement only.
*Fix:* report the per-cell contribution table and both averaging schemes. State the honest claim
as "matches the strong model at ~half the compute, with a significant accuracy gain on two specific
cells."

**Hole 3 — The PathVQA-open reasoning collapse — half the headline — looks like an
answer-granularity artifact.**
The largest single contributor is the reasoning model scoring 0.1087 versus 0.376 direct on
PathVQA-open. Sampling the disagreements shows the reasoning model producing *more clinically
substantive* answers that do not match PathVQA's caption-fragment gold strings, and the judge
penalizing them. Examples read from `ckpts/openvqa/strong_lingshu_think` versus `strong_lingshu`:
gold "sectioned surface" → direct "Sectioned surface." (correct) versus reasoning "Chondrosarcoma."
(wrong); gold "fibrous plaques" → direct "Fibrous plaque." (correct) versus reasoning "Lipid
deposits, fibrous tissue, and calcifications." (wrong). Mean answer length 4.3 words versus 2.0.
Collapse magnitude tracks gold quality: PathVQA −0.267, SLAKE −0.140, VQA-RAD −0.055. And the
independent Claude-judge cross-validation covers **SLAKE and VQA-RAD only — not PathVQA**.
*Fix:* re-grade all three open sets, both modes, with the validated Claude-as-judge path (offline,
no GPU) under an explicit answer-granularity instruction, and report the delta both ways.
*Escalated 2026-07-29 — the granularity gap is written into the prompts, so re-grading alone cannot fix
it.* `src/labeling/run_openvqa.py:26` gives the **direct** arm *"You are an expert medical image
analyst. Answer the question with a short, specific phrase. Do not explain."*, while `:27` gives the
**think** arm *"You will solve a problem/request. You should provide your thoughts within `<think>`
`</think>` tags before providing the answer. After `</think>`, give only the short final answer."* — the
persona and both answer-style constraints are dropped. On free text that is a **live style/length
grading channel**, not a bounded extraction channel, so the two arms differ by more than reasoning.
**A matched-prompt re-run (identical system prompt in both arms, reasoning trigger appended rather than
substituted) is required, and is in flight; until it lands the open-text think-vs-direct delta is
provisional** (C24). Note the same dumps supply the **always-32B-with-reasoning open-text cells of the
headline baseline** (0.5591), so this hole and Hole 1 together bound the vs-reasoning framing —
the vs-direct and vs-oracle-mode comparisons are unaffected.

## Major

**Hole 4 — The open-text verifier scores items it was trained on; inflation is ~31% of the arm's
selection gain.**
The verifier is trained on a 70/30 grouped-by-question split of four open sets
(`run_lora_verifier_open.py:87-88`) and then scored over the **full** sets, so ~70% of every
reported open item was in its training data. The repo's existing leakage check compares raw
accuracies and reports "differs in both directions"; comparing the *selection gain* instead shows
one-directional inflation on all three cells `[derived]`: SLAKE full +0.0434 vs held-out +0.0238
(1.82×), PathVQA +0.1293 vs +0.0897 (1.44×), VQA-RAD +0.1100 vs +0.0926 (1.19×). n-weighted over
the 2,345 open items: full **+0.1040** versus held-out **+0.0718** — roughly **31% memorization**.
The open arm supplies ~93% of compute-lean's vs-reasoning delta.
*Fix:* retrain leave-one-dataset-out (4 runs; the pipeline already supports it), or report the open
arm on the verifier's own held-out 30% (n = 210 / 54 / 435). Kvasir and RadImageNet judge dumps
already exist for both legs and would serve as true held-out open cells.

**Hole 5 — The oracle-mode baseline's cost penalty is 85% a single noise-driven mode choice.**
The oracle picks reasoning on only SLAKE-closed and MMMU. `[derived]` SLAKE-closed alone accounts
for 194.6 ms of the baseline's 229.4 ms excess over 665 ms. That choice rests on a **+0.0047**
accuracy difference between two runs that agree on 94% of predictions and both emit ~4 generated
tokens — i.e. decoding noise buying a 10,521.6 ms price tag. Correcting it puts oracle-mode at
~700 ms / ~134 J, so compute-lean's cost advantage falls from −48% / −51% to ~−33% / −37%.
*Fix:* require the oracle's mode choice to be CI-certified rather than a point argmax.

**Hole 6 — The assembled cascade has never been run on a second model family, and the gate ranking
is known to be family-dependent.**
Every method number is Lingshu-7B → Lingshu-32B. `artifacts/GENERALIZATION.md` validates the three
underlying *findings* across families, explicitly not the assembled cascade. This matters more than
usual because the gate ranking **reverses** across families (on MedVLThinker, stability/agreement
were claimed to beat margin; on Lingshu, margin wins at 0.7254, agreement is worst at 0.6565, and
stability is inert), and because Lingshu is the family where the reasoning half of Finding 1 is
weakest. The MedEvalKit dumps and open-text judge dumps for MedVLThinker and InternVL3 already
exist on disk, so this is mostly an offline re-costing.

**Hole 7 — Thresholds and policies are fit per benchmark against that benchmark's own strong-leg
accuracy; this is not a deployable single policy.**
`heldout()` refits τ inside each cell with the target set to that cell's own 32B accuracy, and the
policy router picks a *different policy* per benchmark. Deployment would therefore require knowing
which benchmark a query came from **and** having labelled calibration data with strong-leg labels
for it. Escalation ranges 8.5% (PMC, `test_2.csv`) to 89.6% (MedXpert), so a single frozen threshold
would behave
very differently. **No τ is materialized in any artifact** — unlike the earlier MedVLThinker work,
which froze one τ = 0.426 and transferred it. The policy choice is also made using CIs computed on
the same held-out data that is then reported (6 cells × 3 policies = 18 tests, no multiplicity
control), and only 2 cells deviate — one contaminated, one carrying 79% of the MedEvalKit pool.
*Fix:* add a "single frozen policy" row and a Holm-corrected policy-selection row, and report how
much of the headline survives.

**Hole 8 — The best-of-N parallel latency is unmeasured and physically inconsistent with its own
energy number.**
Best-of-8 latency is asserted as 347 + 175 = 522 ms — 8 generations plus 8 verifier forwards in the
wall clock of one of each — while energy is correctly billed 8× at 568.8 J. `[derived]` That pairing
implies ~1,088 W of GPU draw, against ~132 W measured at batch-1 for the 7B and a 400 W card TDP.
No batch-8 measurement exists anywhere in the repo. An energy-consistent bound at 400 W puts
batched best-of-8 at ≥1.42 s, i.e. ~2.1× the 665 ms 32B forward rather than the claimed 0.79×.
Separately, the controller's own docstring says its batched-latency field is "only valid for FIXED
N", yet `paper_baselines.cost_pandora` uses it for the *adaptive* policy, which by construction
draws sequentially.
*Fix:* measure batch-8 wall clock and NVML energy for the 7B (a sub-30-minute run). Until then,
retract "best-of-N is latency-alive at 0.79×" and report the open cells on sequential latency.

**Hole 9 — The reasoning-cost constant is n = 15, skewed, and measured on the wrong workload.**
The denominator of every latency and energy claim is a mean over **15** batch-1 samples whose
**median is 12,896.2 ms** — a 23% mean/median divergence indicating a heavy tail — and the headline
uses the *smaller* of the two. It was measured on VQA-RAD cap320 images in the open-text setting at
98.3 generated tokens, then transferred unchanged to every multiple-choice cell, including MedXpert
whose real reasoning traces are 320 tokens and MMMU whose images are much larger. The direct-mode
reference is n = 25, from a gitignored log.
*Fix:* re-measure at n ≥ 100 stratified across all benchmarks' real prompts and images; report
per-benchmark, and report median alongside mean.

**Hole 10 — PMC-VQA carries 79% of the MedEvalKit pool and the only significant multiple-choice
beat, on an auto-generated split.**
The multiple-choice half of the win is entirely the PMC cell (+0.0095 versus direct mode, on
**`test_2.csv`**). It is evaluated on `test_2.csv` (v2, 33,430 items, hard-coded at
`MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39`), the automatically generated split, where every model
plateaus at 0.52–0.56 — consistent with a substantial noisy-label floor. Both legs are trained on
PMC-VQA's training split (v1 `train.csv`), so a rule that keeps the cheap answer where its
confidence is high may be exploiting shared label-noise structure.

PMC-VQA also ships a human-verified `test_clean` split (v1, 2,000 items — the authors' only manually
checked split). **Correction, 2026-07-30 (logged as X20): it *is* on disk, and it has been in use
all along.** `/data/dan/dataset/medevalkit/PMC-VQA/test_clean.csv` (418,686 bytes, 2,000 data rows,
mtime 2026-06-29 07:18) plus a byte-identical second copy at
`/data/dan/dataset/pmc_vqa_train/test_clean.csv`; and the MedVLThinker-era cascade track evaluates
exactly those 2,000 items — `MedVLThinker-Eval`'s `pmc_vqa` slice matches `test_clean.csv`
**2,000/2,000** positionally on question, `answer_label` and answer text, and
`ckpts/gate_7b_prune/cap320/ckpt_PMC-VQA_nothink_norag.jsonl`'s golds match
`test_clean.Answer_label[idx]` **2,000/2,000**.
*The withdrawn wording, kept for the record:* "PMC-VQA ships a human-verified `test_clean` split;
**it is not on disk** at `/data/dan/dataset/medevalkit/PMC-VQA/` (which holds `test.csv`, 50,001
lines, and `test_2.csv`, 34,824 lines) and has never been used anywhere in the repo." (Those two
line counts were raw `wc -l`, which over-counts: the files hold 50,000 and 33,430 *data* rows.)
Evidence: `docs/current/PMCVQA_PROVENANCE_2026-07-30.md` §3 and §3.1.
*One thing that was checked and is clean:* the beat is not a fold artifact — +0.0095 with
modulo-5 folds versus +0.0097 with figure-grouped folds, despite 26.4% of `test_2` PMC rows sharing
a figure.
*Fix (revised 2026-07-30):* the clean-split re-run is **already satisfied for the
cascade/MedVLThinker track** and is outstanding **only** for the MedEvalKit/Lingshu track, which
would need a one-line vendor patch at `PMC_VQA.py:39` (~20 min GPU) — or can be approached with zero
GPU from the clean-split dumps that already exist (`PMCVQA_PROVENANCE_2026-07-30.md` §4, Route B).
Note also that `test_clean` ∩ `test_2` = **6 items**, so the existing 33,430-row dumps cannot be
filtered down to the verified split; that line of inquiry is closed.

**Hole 14 — Reproducibility: the entire July chain is outside git, a superseded constant is still
live, and the paper's own figure and table disagree.**
Three compounding risks.
(a) **Preservation.** 44 untracked `.py` files under `src/` include *every* file in the live
headline chain; `results/` and `MedEvalKit/` are both gitignored, so all 107 numeric artifacts and
the primary evaluation dumps are untracked; the last commit (`8cdefef`, 2026-07-02) predates the
entire Lingshu era. **The paper's method, its inputs and its outputs exist on one disk.**
(b) **A stale constant propagating into a published figure.** `integrated_method.py:61` still
hard-codes the superseded n=200 open-text reasoning estimate
(−0.195 / −0.120 / −0.130) versus the measured −0.1395 / −0.055 / −0.2673, and
`paper_baselines.py:75` aliases it. So `method_final_mmmu_corrected.json` still carries
always-32B-reasoning = 0.5628 flagged estimated, while Table I of the paper reports the measured
0.5591 — and `paper/make_ieee_figs.py:67` builds Figure 1 from the *stale* file. The same delta is
**+0.0208 in one live artifact and +0.0245 in another.**
(c) **An underived constant defining the headline.** The 32B/7B compute ratio 4.57 appears only as
a hard-coded literal; no file derives it. An older document implies 4.34 from parameter counts. At
4.34, accuracy-max's ratio moves from ~0.93× to ~0.95× `[derived]` — the "compute-negative" claim
has a 7% margin and an underived denominator.
Also here: the professor deck (`paper/build_professor_html_2026-07-27.py`, the newest file in the
repo, 79 KB) makes **zero** JSON reads, contains 116 hand-typed four-decimal literals, and its
footer claims "All figures … were read from real artifacts … No number was estimated or invented."
The headline row transcribes correctly (spot-checked), but nothing enforces that, and the deck
labels **two different systems "accuracy-max"** on adjacent slides (5.70 compute-units = 1.25×, and
"0.93× … compute-negative"). The paper disambiguates them; the deck does not.

## Minor

**Hole 11 — The certified veto's "never worse by construction" claim is not a valid test.**
It compares a Wilson lower bound on one arm to a *point* estimate on the other, ignores the strong
model's own sampling error, ignores that the two are measured on the same items (a paired test is
correct), and runs 5 bins × 5 folds = 25 uncontrolled certifications. The project's own
thin-slice result already showed that this style of certification overfits.
*Fix:* one-sided paired test on the per-item difference within each bin, Holm-corrected.

**Hole 12 — No guard for the near-chance regime.** Where both legs are near chance the gate
escalates almost everything, so the cascade pays the cheap leg *on top of* the strong leg and
delivers slightly worse accuracy: MedXpert −0.0060 [−0.0120, −0.0005] versus oracle-mode, at 89.6%
escalation, 5.095 compute-units and 943 ms versus always-32B-direct's 4.57 / 665 ms. This is a
systematic failure mode with no detection or fallback.
*Fix:* a deployable degeneracy check on the calibration fold — if iso-accuracy escalation exceeds
~60%, fall back to always-strong and say so.

**Hole 13 — The open arm rests on 2,345 items, two of three cells have CIs spanning zero, and one
is a prefix slice.** SLAKE-open (+0.0016) and VQA-RAD-open (+0.0050) both span zero at n = 645 and
n = 200, so the only CI-certified open beat is PathVQA-open — the same cell flagged in Hole 3.
SLAKE-open uses 645 English-only items of the harness's 1,258; PathVQA-open uses a **prefix** of
1,500 of 3,357 (`run_openvqa.py:96` is `items[:n]`, not a random draw) taken "for speed", so it may
be a topically biased slice — a 10-minute offline check that has never been done.

**Hole 15 — An unlogged contradiction about whether verifier confidence is really the best open
gate.** Two runs of the same regime disagree on the published deferral baseline. One run
(2026-07-01) reports it at 0.3832 versus verifier confidence 0.3923 ⇒ "verifier confidence beats
the state of the art". The other (`artifacts/gate_unified_bakeoff.json`, 2026-07-04) reports
**0.3965 versus 0.3901, +0.0062 [+0.0040, +0.0086], winning on 100% of seeds**. Same underlying
pooled accuracies, opposite sign. `INCONSISTENCIES.md` does not mention it. **The claim "the gate
question is settled — verifier confidence is the best gate" is not safe until this is reconciled.**

**Hole 16 — Documentation is inverted.** The newest, most load-bearing code is the least
documented. `f8_mode_vsthink_ci.py`, which produces the paper's headline CI, appears in **no
markdown document anywhere in the repo**. `STRUCTURE.md` omits all four July-8/9 headline scripts.
Six abstention-direction scripts still sit in live `src/cascade_methods/` with artifacts and a
figure in `paper/figs/open/`, despite the permanent prohibition (the paper itself is clean).

---

# 8. WHERE THE PROJECT SHOULD GO NEXT

Sequenced so that integrity work comes before new science, and so that the cheapest tests that can
*invalidate* something come early.

## 8.0 Three measurements that reorder the list

Done during this retrospective's preparation, from live dumps:

**M1 — A peer-difficulty signal is far better at recoverability than the deployed margin.** Using
the response matrix across 6 aligned model configurations (44,694 items; MedEvalKit track, so the
PMC cell is **`test_2.csv`**, n = 33,430), recoverability AUROC
(peer-difficulty / margin): PMC 0.649 / **0.407**, SLAKE 0.583 / **0.236**, VQA-RAD 0.789 /
**0.472**, PathVQA 0.799 / **0.670**, MedXpert 0.774 / **0.450**. **The deployed margin is
anti-predictive of recoverability on 4 of 5 sets.**

**M2 — But that does not translate into a deployable win.** Net gain per escalated item is
P(strong right) − P(cheap right); margin wins the *detection* term, offsetting its
anti-correlation on the *recovery* term. At a fixed escalation budget of 20%, a cross-fit rule over
[margin, peer, margin×peer] versus margin alone: PMC **+0.0001** (`test_2.csv`; against oracle
headroom +0.1077),
SLAKE +0.0072, VQA-RAD +0.0022, PathVQA +0.0144, MedXpert +0.0175. A signal costing **four extra
model forwards** captures ~0% of net-gain headroom on the cell carrying 79% of the MedEvalKit pool. This is
the strongest evidence yet that the ~0.6 recoverability wall is close to a genuine ceiling.

**M3 — The coverage wall is 4.5× the selection wall** (see §5.5): 434 of 1,064 questions (40.8%)
have no correct answer in the pool at all, against a total selection gap of 0.0912. **Generator
ideas compete for +0.408; verifier ideas compete for at most +0.091.**

Consequence: further *gate* work is deprioritized; measuring the achievable ceiling is promoted;
generator work is promoted above verifier work.

## 8.1 Tier 1 — offline, on existing dumps. Do these first.

> **Item 1P — finish the prompt-matching repair** *(added 2026-07-29; closes C24/C25 and the offline half
> of hole 3; deliberately unnumbered so the item numbers §8.4 refers to still hold — but it belongs in the
> Week-1 integrity block).* Three parts, in order: **(a)** the **open-text matched-prompt re-run** —
> identical system prompt in both arms, the reasoning trigger *appended to* the answer-style constraints
> rather than substituted for them (the only part that needs a GPU; already in flight). **(b)** Decide the
> **`MedEvalKit` question** (§10.5): revert the dependency to upstream and pass the trigger from the
> caller, or vendor the change in this repo's own runner — then re-run MMMU/MedXpert so the reason arm is
> format-matched. **(c)** **Persist the prompt** (system message + user instruction + resolution cap) in
> every checkpoint row, and assert `mean_generated_tokens` on every think-arm run, so a think arm that
> never reasoned fails loudly instead of silently becoming a published cell. Part (c) is the cheapest item
> in this whole section and would have caught C22 in June.

1. **Honest re-costing and re-framing of the baselines** *(closes holes 1, 2, 5, 14b)*.
   Charge reasoning cost only where a genuine reasoning dump exists; require the oracle's mode
   choice to be CI-certified; emit the per-cell contribution table and both averaging schemes;
   headline versus always-32B-direct and the CI-certified oracle. *Expected:* the headline moves
   from −95.5% / −96% to roughly −72% / −74% on cost and from +0.0245 to +0.0106 on accuracy. This
   is a loss on paper and a large gain in survivability.
2. **Reproducibility lockdown** *(closes hole 14)*. Commit the 44 source files; force-add the ~10
   headline JSONs; copy the evaluation dumps out of the gitignored vendor directory. Replace the
   stale open-text reasoning constant with the measured per-sample vectors and regenerate the
   artifact that Figure 1 reads, so figure and table agree. Write down (or recompute) the 4.57
   derivation and add a sensitivity row.
3. **Item-difficulty calibration and a two-sided band gate.** Fit a 2-parameter item-response model
   on the four **non-Lingshu** configurations only (no leakage), recompute M1's AUROCs with proper
   item difficulty, then cross-fit a difficulty regressor from **test-time-available features
   only** and re-run compute-lean with a band gate. *Success criterion:* escalation-at-parity falls
   ≥15% on PathVQA-closed or VQA-RAD-closed with the guardrail intact. Explicitly **do not** set a
   criterion on PMC — M2 shows the headroom there is ~0. *Expected magnitude:* bounded above by
   M2's column, so +0.009 to +0.018 on the three cells that currently contribute exactly zero, and
   ~0 on PMC (`test_2.csv`). *(Closes hole 2 partially, hole 12 partially.)*
4. **Single frozen policy row** *(closes hole 7)*. One τ calibrated once on a held-out calibration
   set and applied unchanged everywhere; one globally-chosen policy; the policy selection
   Holm-corrected over 18 tests. Report honestly how much survives.
5. **Cross-family re-costing of the assembled cascade** *(closes hole 6)*. Run the assembled
   compute-lean pipeline on MedVLThinker and InternVL3 from the dumps already on disk. Claim a
   *recipe* (pick the gate per family), not a fixed gate.
6. **Repair the veto's certificate** *(closes hole 11)*. Paired one-sided test, Holm-corrected;
   report the beat under the corrected rule.
7. **Gauge R&R on the scorers** *(addresses holes 3 and 10)*. On a stratified PMC sample
   (`test_2.csv`, the split the +0.0135 was measured on) plus the
   three open sets, run every available scorer over the *same* outputs and decompose item / model /
   **scorer** variance. *Criterion:* scorer standard deviation < 1/3 of the claimed effect
   (< 0.0045 against +0.0135). Extend this to re-grading all three open sets with the validated
   Claude judge under an explicit granularity instruction — that is the offline half of hole 3.
8. **Queueing re-costing.** Re-evaluate the existing method under a load model rather than batch-1:
   a discrete-event simulation with service times drawn empirically from the latency logs, Poisson
   arrivals swept over λ, routing at measured escalation rates. Report p95/p99 and maximum
   sustainable arrival rate for all four systems. **Must be labelled a simulation from measured
   service times**, with the batch-1 numbers reported unchanged alongside.
9. **A decision-theoretic frame (desk work).** Instantiate a closed-form escalation condition per
   benchmark from the measured cost and correlation parameters and produce one table of
   predicted-winnable versus measured-winnable cells. This upgrades ~25 negative results from a
   list into a quantitative confirmation and gives reviewers a principled answer to "why didn't you
   try X?".

## 8.2 Tier 2 — small GPU experiments

10. **Verifier grounding audit — run this first (~30 min).** The image-ablation script already
    exists and **its result is nowhere on disk** (the three "scaling" run directories
    `lora_verifier_ds500`, `ds1500`, `pooled4_r32` are empty). Report real versus blank-image
    selection accuracy on the same split, both seeds. *Branch A:* Δ > 0.10 → the intrinsic-ceiling
    story is confirmed and sourced. *Branch B:* Δ < 0.05 → the project's central July conclusion is
    wrong and counterfactual-image training becomes a live lever. The decision value exceeds the
    accuracy value.
11. **Verifier data-scaling curve.** Train at 500 / 1,500 / 3,000 / 6,000 / 10,364 pairs × rank
    16/32 × 1/2 epochs on the identical split. The "verifier is at ceiling" verdict currently rests
    on **two points at the same end of the axis** with the intervening runs never executed. Either
    the curve is flat (the ceiling claim becomes defensible and two downstream items can be
    dropped) or it is still rising at 6k (data scaling is live).
12. **On-policy verifier retraining — highest-value verifier item.** Best-of-N is adversarial search
    against the verifier, and the deployed adapter has only ever scored *off-distribution* pools.
    Evidence it is live: selection efficiency falls monotonically with N (1.000 → 0.914 → 0.841 →
    0.770 at K=8 → **0.717 at K=16**) and 0.574 → 0.496 when moving to the diverse pool.
    Rebuild the training set from the diverse pool on train-half questions only and retrain.
    *Criterion:* efficiency on diverse@15 returns to ≥0.732 without regressing iid@8.
    *Expected:* diverse@15 lifts oracle +0.064 while the off-policy verifier converts only +0.025;
    recovering half the unconverted 0.039 is **+0.02 pooled at zero extra inference cost**.
13. **Listwise single-forward verifier — the only path to a deployable open arm.** Verification is
    currently O(N) forwards with the image prefill paid N times. Mean distinct candidates per
    question is **3.76**, so a listwise prompt is short. Train the adapter to emit the index;
    average over 2 shuffles for position debiasing; measure real batch-1 latency with the same NVML
    harness. *Criterion:* open-arm compute falls from 16 to ≤9 units at no selection loss, and a
    softmax-over-indices confidence retains gate AUROC ≥ 0.85. With the measured adaptive draw
    counts the open arm would land at ~4.5–6.5 units — comparable to one 32B forward.
14. **Option-order circular consistency as a gate signal.** Perturb the *text/label* channel rather
    than the image. It is emphatically not inert: 33% of 7B items and 27% of 32B items flip under
    cyclic option shifts, whereas the resolution-stability gate died precisely because the 7B is
    98.95% resolution-stable. *Cost discipline is mandatory:* K=3 on all traffic is 3 compute-units
    and eats the entire escalation saving, so it must fire only inside a borderline margin band
    (~20–30% of traffic), tuned held-out. *Caveat:* the pre-test data is on MMMU, the
    contamination-suspect benchmark, so replicate on PMC/SLAKE/VQA-RAD before any claim. Sell
    consistency as a *signal*; debiasing itself **lowers** accuracy for both legs.
15. **Noise-ceiling measurement — highest decision value per GPU-hour.** If the strong model's
    correctness is Bernoulli with probabilities concentrated mid-range, the Bayes-optimal AUROC is
    low regardless of features. Estimate per-item probabilities from replicates under nuisance
    perturbations (option-order shifts **plus** temperature-ε sampling — greedy decoding is exactly
    reproducible, so order alone is insufficient), then compute the achievable ceiling with a
    split-half estimator to remove finite-replicate optimism. *This is a program-level stop/go
    instrument.* If the ceiling lands near ~0.68, stop attacking the gate entirely and move
    everything to the strong leg's cost and to generation. If ~0.85, item 3 and the revived
    prefill probe become worth heavy investment.
16. **Batch-8 latency measurement (~30 min)** *(closes hole 8)*.
17. **Reasoning-latency re-measurement at n ≥ 100, per benchmark** *(closes hole 9)*.
18. **PMC-VQA clean-split check** *(closes hole 10)*. ~~Download the human-verified split~~ —
    **revised 2026-07-30 (X20): no download is needed, `test_clean.csv` is already on disk in two
    places and the whole MedVLThinker/internal-harness track already runs on it.** What is actually
    outstanding: (a) offline, zero GPU — replicate the PMC fusion/veto comparison on the clean 2,000
    from the existing `ckpts/gate_7b_prune/cap320/` and `ckpts/gate_32b_modes/nothink_fullres/`
    dumps (both have `opt_logprobs` on all 2,000 rows), pre-registering that n = 2,000 is
    *underpowered* against a +0.0135 effect (CI half-width ≈ 0.0141), so a null is the expected
    result, not a refutation; (b) ~20 min GPU — bring the **MedEvalKit/Lingshu** track onto the clean
    split via a one-line patch at `PMC_VQA.py:39`, reported as an extra column beside `test_2`, never
    as a replacement (comparability with the published Lingshu baseline depends on `test_2`).
    Details and the power table: `docs/current/PMCVQA_PROVENANCE_2026-07-30.md` §4.
19. **Open-arm leakage repair** *(closes hole 4)*: leave-one-dataset-out retraining, or the
    zero-GPU fallback of reporting on the verifier's held-out 30%, plus folding in the two existing
    truly-held-out open datasets.
20. **Open-set extension** *(closes hole 13)*: PathVQA-open 1,500 → 3,357 and SLAKE-open 645 →
    1,258. Minimum viable version: a 10-minute offline check that the 1,500 prefix is not sorted by
    topic or image — do this regardless.

## 8.3 Tier 3 — bigger bets

21. **A stronger cheap leg (e.g. Hulu-Med-7B).** *Proves:* whether the cascade still has a job when
    the cheap leg matches the strong one. **Do the zero-GPU version first:** recompute the operating
    point with the cheap-leg accuracy vector replaced by published numbers, cost model fixed. That
    afternoon of work says whether the cascade survives — and the honest counter-risk is that it
    does not, which would invalidate the headline rather than improve it. A full run requires
    reproducing **both** models under MedEvalKit (~3–5 GPU-days), because the published cross-model
    comparison does not use this harness and its own strong-model column disagrees with that
    model's paper.
22. **Attention-guided pruning of the escalated 32B prefill.** The only lever that *reduces the
    strong leg's cost* rather than avoiding it; the guidance signal is free because the 7B forward
    is already paid. Projected −26% on the escalation forward. *Decisive arm:* the chest X-ray box
    IoU set (n = 435, already on disk) as a radiology-safety probe — grounding collapse under
    pruning is a published failure mode and medical lesions are small. Measure damage **restricted
    to the escalated subset**.
23. **A trained generative / chain-of-thought verifier, open text only.** Gate it behind items 10
    and 11 — a rationale from a verifier that never looks at the image is confabulation, and if the
    scaling curve is flat this is not worth the GPU. Note the current training script masks all but
    the last token, which must change. Expect +0.02–0.04 pooled selection, and frame it as an
    accuracy-max-only ceiling result with measured verify latency so the compute verdict is
    computed, not assumed. Best deployed at N ≈ 2–4 under the adaptive controller.
24. **A generator-training ladder, strictly staged.** This outranks every remaining verifier idea
    because of M3 (0.408 target versus 0.091).
    *Stage 1 — self-distillation on judge-correct samples* (~1 GPU-day): transfers the measured
    +0.116 greedy→best-of-8 gap from test time to train time, realized at N=1 (1 compute-unit,
    347 ms) instead of best-of-8 (16 units). Target +0.03–0.06 greedy. **Monitor oracle-of-8 as a
    stopping criterion** — self-training narrows the output distribution, and shrinking oracle-of-8
    is the opposite of what the open arm needs; restart from base each iteration.
    *Stage 2 — best-of-N-aware training*: report oracle-of-8, verifier-best-of-8 and selection
    efficiency as three separate numbers; the whole claim is that they move together, unlike
    inference-time diversity which lifted oracle +0.064 while efficiency fell.
    *Stage 3 — reinforcement learning with verifiable rewards*: only if stages 1–2 pay. Judge
    throughput is the real bottleneck, so this must be an iterative loop, not online RL.
    All three stages invalidate cached cheap-leg checkpoints for that family.
25. **Lower priority, in order:** noisy-channel reverse-likelihood scoring (genuinely orthogonal to
    every ruled-out forward-direction signal; expect gains only on content-bearing answers);
    training-split verifier training (the only axis that also improves the deployed *gate*, and it
    frees the full test sets, halving every verifier CI — but do item 11 first); a multi-task
    answer+box verifier (the box verifier's +0.191 has never been used as training signal for the
    answer verifier; watch for negative transfer given base rates of 0.04 versus 0.41);
    image-space test-time augmentation (conservative augmentations only — no flips, no colour
    shifts on pathology — and note the visual prefill is not shareable, so it is strictly more
    expensive per candidate than temperature sampling); visual KV-cache eviction and speculative
    decoding, **both weakened** by the finding that the only genuine multiple-choice reasoning
    measurement shows reasoning buying nothing.

## 8.4 Recommended sequence

- **Week 1:** items 1 → 2 → 6 → 7 (the integrity block; nothing else is safe to build on). Run
  item 10's 30-minute image ablation in parallel — free, and it can invalidate a load-bearing
  conclusion.
- **Week 2:** items 3 → 4 → 5 → 8/9, then **item 15 (noise ceiling)** as the stop/go instrument.
  Fire items 16 / 17 / 18 opportunistically — each is a few GPU-hours and each closes a named hole.
- **Week 3+:** branch on item 15. Low ceiling (~0.68) → abandon gate work; put everything into
  item 13 (listwise verifier), item 12 (on-policy verifier), item 22 (strong-leg pruning) and
  item 24 stage 1 (generator training). High ceiling (~0.85) → item 3's band gate and a revived
  prefill probe become worth heavy investment.

## 8.5 The one framing change to make now, independent of any experiment

Stop describing the result as a suite-wide accuracy advantage. The honest claim is:

> *"Matches the strong model at roughly half the compute, with a significant accuracy gain on two
> specific cells — open-ended free text and PMC-VQA (`test_2.csv`) — and a measured
> characterization of why the remaining cells are unwinnable."*

The characterization (§5.5, M1–M3) is a stronger contribution than +0.0245 ever was, and unlike
+0.0245 it survives the honest re-costing in item 1.

---

# 9. PRACTICAL NOTES

## 9.1 Hardware, environments, data locations

- **Machine:** one VM, user `jamesyang`, **two A100 80 GB GPUs**, a shared `/data` mount.
- **Repository:** `~/medvlthinker-imgdiff-compute`. **Always launch scripts from the repository
  root** — several scripts resolve `ckpts/...` relative to the working directory, and the
  cascade_methods modules use bare sibling imports that only resolve when run as
  `python3 src/cascade_methods/<x>.py` from the root.
- **Weights:** under `/data/dan/weights/` and `/data/dan/hf_cache/hub/`. Runners hard-code
  **HuggingFace snapshot-hash paths** (e.g. `models--lingshu-medical-mllm--Lingshu-7B/snapshots/
  b98aecd4…`) — a cache refresh changes the hash and breaks them.
- **Evaluation data:** `/data/dan/dataset/` — MedEvalKit datasets under
  `/data/dan/dataset/medevalkit/`.
- **Two Python environments, only one documented.** The vLLM NGC container and HF transformers are
  documented in `CLAUDE.md` §8; a third, **`/data/dan/medeval_venv/bin/python`**, is used by 12
  runner invocations and is documented nowhere.
- **Long jobs use `nohup`, never `tmux`**, with checkpointed resumable runs and per-sample error
  guards.

## 9.2 The tensor-parallel hang, and every mitigation that failed

Running Lingshu-32B (or InternVL3-38B) under MedEvalKit with `tp=2` on OmniMedVQA produces a
**deterministic NCCL collective hang**, roughly 36 minutes of stall per chunk. Ruled out over
~2 days:
- chunked runs with retry ×3 — recurs on every chunk;
- `TORCH_NCCL_ENABLE_MONITORING=0` — **worse**: it hangs *forever* instead of aborting;
- `EVAL_BATCH_SIZE=256` — fixes a *different* problem (a container cgroup OOM), not the hang;
- 3-hour per-chunk timeouts with an aggregation backstop — re-hangs.
- **`tp=1` is impossible**: 64 GB of weights plus multimodal activations exceed an 80 GB card.

**The identified but never-executed way out:** an INT4 (AWQ) quantization of Lingshu-32B fits at
`tp=1` in ~20 GB, sidestepping the collective entirely. The quantized checkpoint currently on disk
is the **non-medical base model**, not Lingshu, so it cannot substitute.

Related gotchas worth knowing:
- Running two cheap legs concurrently causes a container cgroup OOM (~245 GB anonymous RSS) — use
  a strictly sequential driver.
- A 4-way concurrent model download silently **corrupted one 32B shard**. Always verify safetensors
  after downloading.
- `runners/run_full_matrix_medeval.sh` treats *any* JSON under the output path as "done", so a
  crashed partial run is skipped forever with no row-count validation.

## 9.3 Evaluation contexts — do not cross-multiply them

There are **three** separate evaluation contexts in this repo.

| context | harness | pool | n |
|---|---|---|---|
| **A. Faithful MedEvalKit** (the paper's suite) | `MedEvalKit/eval.py`, vLLM, seed 42 | 7 benchmarks | PMC 33,430 (`test_2.csv`) · SLAKE 2,094 · VQA-RAD 451 · PathVQA 6,719 · MMMU 150 · MedXpert 2,000 · OmniMed 88,996 |
| **B. Internal NGC harness** (the 5-family bake-off) | `src/labeling/*`, `ckpts/acc_gen/` | 6 benchmarks | 8,220 total, of which PMC 2,000 (`test_clean.csv`) |
| **C. Custom open-text pipeline** (LLM-judged) | `run_openvqa.py` + `run_judge.py` | 5 open sets | 645 · 200 · 1,500 · 1,200 · 2,000 |

> ### ⚠ LANDMINE — the project has **two different PMC-VQA splits**, one per track. Never mix them.
>
> | track | PMC-VQA file | n | share of its own pool | verified by the authors? |
> |---|---|---:|---:|---|
> | **A. MedEvalKit / Lingshu** (the paper's suite) | **`test_2.csv`** (v2) | 33,430 | **79%** of Variant B (n = 42,224) | **no** — zero published verification, zero published accuracy |
> | **B. Internal harness / MedVLThinker-Eval** (the June cascade) | **`test_clean.csv`** (v1) | 2,000 | **24.3%** of 8,220 | **yes** — the authors' only manually checked split |
>
> The split is *not* a setting we chose in track A: `test_2.csv` is hard-coded in unmodified vendor
> code at `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39`. **`test_clean` ∩ `test_2` = 6 items**, so the two
> are effectively disjoint populations — a number from one cannot be filtered into, compared against,
> or substituted for a number from the other, and the same models score several points apart on them.
> Any PMC-VQA number in this repository must be quoted with its file name and row count. Full
> provenance, including which dumps belong to which split:
> **`docs/current/PMCVQA_PROVENANCE_2026-07-30.md`**.

**The headline pool (n = 42,374) is a splice of A and C**: 40,029 multiple-choice cells from A plus
2,345 open cells from C. It is not a single harness run. The method spec acknowledges A-versus-B
must be kept separate but does not flag the A+C splice.

Other data facts a newcomer needs:
- **MedEvalKit's open-half exact match is broken** — gold `"CT"` versus response `"CT."` scores
  incorrect while ROUGE-1 ≈ 1.0. Only the `close` sub-metrics of SLAKE/VQA-RAD/PathVQA are usable
  from that harness; the open halves must be judged.
- Two model dumps in the tree have **unidentified provenance**: `MedEvalKit/eval_results_reason`
  (documented as "a DIFFERENT model, not Lingshu-32B") and the undated `eval_results` /
  `eval_results_32b`. They sit next to the canonical directories with no rename or README.
- Six independent Lingshu-32B MMMU runs exist at 0.6267 ×3, 0.6333 ×2 and 0.660 — a 5-question
  spread that is the effective noise floor for every MMMU claim. The method uses the 0.660 outlier
  as the reasoning baseline (the conservative choice).
- Two open-text results sit unused: Lingshu-32B is **worse** than Lingshu-7B on RadImageNet
  (0.289 versus 0.321) and tied on Kvasir (0.3008 versus 0.3017). Both judge dumps exist. This is
  the strongest existing evidence for the "the strong model is not uniformly strong" framing.

## 9.4 What a newcomer should read, in what order

The existing `READING_GUIDE.md` is a full era out of date and will mislead you — it stops at
June 26, points at a paper draft that has since been archived, and describes the removed abstention
work as part of the arc. Use this order instead:

0. **This document.**
1. `meetings/progress_report_professor_2026-07-27.html` — the best single artifact in the repo; the
   whole project in 13 source-cited sections. (Currently reachable from nothing, and untracked.)
2. `paper/adaptive-cascade-medvqa_ieee_2026-07-08.pdf` — the polished, numerically-current story.
3. `results/cascade_methods/docs/current/TECHNICAL_REPORT_2026-07.md` — technical walkthrough
   *(numbers pre-date the 2026-07-08 measurement; see §10)*.
4. `results/cascade_methods/docs/current/METHOD_FINAL_2026-07.md` — the full spec and ablations
   *(same caveat)*.
5. `progress/progress_July_04.md` → `July_08.md` — how the final method was actually found.
6. `progress/progress_June_17.md` → `June_25-26.md` — the compute-configuration and trained-verifier
   prehistory.
7. `results/cascade_methods/docs/current/PMCVQA_PROVENANCE_2026-07-30.md` — **read before quoting any
   PMC-VQA number**: how PMC-VQA is constructed (caption-only GPT-3.5 generation), how thinly it is
   validated (one undocumented 2,000-item human pass), which splits exist, and which of them each of
   this project's two evaluation tracks actually used.
8. `STRUCTURE.md` as the code index; `results/cascade_methods/docs/archive_mcq/` as deep-dive
   appendices; `results/cascade_methods/METHOD_IDEAS_BACKLOG.md` for the 68-idea backlog **(no
   per-idea status field — several entries in it have already been run and failed; cross-check
   §6 before proposing any of them)**.

## 9.5 Codename glossary (for reading the older documents)

| codename | plain meaning |
|---|---|
| ACC / ACC-v2/v3/v4 | the three-tier compute-configuration cascade (7B-direct → 32B-direct → 32B-reasoning) |
| VADR | verification-augmented deferral router (killed, §6.1 N16) |
| FALC | the deployed margin-gated two-tier cascade |
| CASP / CCPS | input-perturbation (resolution) stability as a confidence signal; renamed because "CASP" collides with the protein-structure competition |
| FLD | FastLeg-Distill — distilling the strong leg's direct-mode competence into the cheap leg (killed) |
| F1 | the per-benchmark guardrailed policy router |
| F3 | confidence-advantage fusion (accuracy-max v1 lever) |
| F5–F7, F11 | double reading, contrastive decoding, super-learner, Bayesian model averaging (all killed) |
| F8 | the certified weak veto (accuracy-max v2 lever) |
| F10 | team-objective learning-to-defer on the open arm |
| Pandora | the Weitzman optimal-search adaptive-N controller |
| G1–G8 | escalation-speed / systems levers (G4 pruning, G7 cache, G8 prefill prefetch, …) |
| H1–H9 | the pass-4 offline backlog items (H1 test-time adaptation, H2 kNN gate, H4 slice discovery, H8 shrinkage, H9 neuro-symbolic — all negative; H3 excised as abstention) |
| UGV | unified generative verifier |
| cap80/160/320/640 | image-resolution budgets (a cap on pixels); cap320 is the chosen cheap-leg operating point |
| ALL-6 / ALL-5 / competent-4 | benchmark pools on the internal harness |
| Variant A / Variant B | MMMU escalated / MMMU excluded |

## 9.6 Code health, in one paragraph

199 Python files under `src/`, 122 in `src/cascade_methods/`; **all 199 parse cleanly**; exactly one
broad `except` in the whole headline chain; hard assertions in the shared cell-builder that fail
loudly if the policy router's choice changes; every honesty caveat mirrored into the artifacts'
`data_gaps` fields. There are **two disjoint code universes**: the June multiple-choice era rooted
at `harness.py` (imported by 39 files, superseded as a source of headline numbers but still the
reproducibility anchor for the archived documents) and the July Lingshu chain (10 live files), which
imports `harness.py` **zero** times. A mechanical scan found **66 scripts** in `cascade_methods/`
that neither write an existing artifact nor are imported by anything. The failure mode in this
repository is not sloppiness — it is that corrections were made in *new* files instead of being
propagated back into the old ones. Consolidation targets: one cost-constants module (currently 3
copies), one loaders module (`auroc` is defined 12 times), one bootstrap function with an explicit
RNG argument and a single resample count (currently 2,000 in one file and 10,000 in others, and
CI bounds depend on call order because module-level RNG streams are shared and mutated across
modules).

---

# 10. CORRECTIONS LOG

Things this project believed at some point that turned out to be wrong, and how each was caught.
This list is an asset: the habit of publishing its own refutations is the strongest thing in the
record.

## 10.1 Scientific claims retracted or downgraded

| # | claim | correction | how it was caught |
|---|---|---|---|
| C1 | The three-tier cascade's agreement gate is "the one genuine improvement" | **Prior art** — Agreement-Based Cascading (arXiv 2407.02348). Retracted *in the same document*. | adversarial prior-art check |
| C2 | Stability/agreement gates beat the margin gate | **True-ish on MedVLThinker, false on Lingshu**: margin AUROC 0.7254; stability **inert** (98.95% stable); agreement **worst** (0.6565). Gate choice is model-specific. | re-running the bake-off on the new family |
| C3 | Self-consistency beats confidence on open text | **Conditional only** — self-consistency wins for a *miscalibrated* cheap model; for a calibrated one confidence wins (0.866 vs 0.845) | testing a second family |
| C4 | PathVQA's difficulty is a caption-extraction artifact | **Refuted by the researcher's own systematic audit** — well-formed questions are the *hardest*, "artifact"-labelled ones had *higher* accuracy (0.369 vs 0.144). The hypothesis had been formed from 14 eyeballed cases. | a systematic audit replacing eyeballing |
| C5 | The chest X-ray box verifier is a negative (IoU 0.022) | **A coordinate-space bug** — boxes are emitted in smart-resized pixel space. Corrected → 0.232, the strongest positive in the project. | the implausibility of the magnitude |
| C6 | The cascade's efficiency headline (three conflicting forms, e.g. 20.0 s → 5.7 s, compute 81 → 55%) | **Retired** → canonical 11.34 s → 2.27 s, compute 100 → 52%, energy 6,318.8 → 1,181.9 J. Root cause: the June-22 cost-methodology bug. | a three-way documentation audit |
| C7 | Lingshu has no promptable reasoning mode | **Retracted within 24 hours** — a weak-prompt artifact; generated tokens go 3 → 174 / 267 with the right prompt | probing the prompt directly |
| C8 | Simulated pairwise verification is a parity negative | **Overturned** by a real forward pass (+0.036 [+0.016, +0.055]). *You cannot manufacture comparative signal from pointwise scores.* | running the real thing |
| C9 | Best-of-N is not deployable; the router is the only lever | **Correctly derived but scoped to the direct-mode baseline only**; re-opened by the July-7 re-grounding | the log's own flagged assumption |
| C10 | Gates should be ranked by detection AUROC | **Wrong proxy** — cascade quality tracks recoverability (r +0.65) not detection (r −0.21); the top-detection gate has *below-margin* cascade quality | a controlled gate swap |
| C11 | The trained verifier *beats* the 32B | **Downgraded to "competitive with / matches"** — +0.039 on seed 0, **ties on seed 1** | running a second seed |
| C12 | MMMU keep-7B is a +0.140 beat-the-strong-model win | **Excluded entirely** after the contamination audit | an adversarial audit the user demanded |
| C13 | A conformal prediction-set router is equivalent to maximum softmax probability | **Wrong** — the equivalence holds only for ≤5 options; a faithful implementation over-escalates 69–80% | a post-audit re-implementation |
| C14 | Self-verification is a weak gate | **Wrong as stated** — it referred to predicting *cheap-model correctness*; as a *gate* with a faithful threshold it is the best on the competent-4 pool (20% escalation) and still collapses on the 6-benchmark pool (79%) | the same post-audit |
| C15 | Structured grounding escapes the luck floor (AUROC 0.816) | **An artifact of an incompetent grounder** — a competent one gives 0.557 (chance) | swapping in a competent grounder |
| C16 | "Reasoning hurts perception" at the June-17 magnitudes | **Real but inflated** by a foreign reasoning prompt applied to all families; native re-runs shift the numbers materially. **↺ Itself reversed on 2026-07-29 (C20):** the native-recipe "fix" *un*matched the two arms (persona, answer-format clause, resolution) and two native recipes had no reasoning trigger at all, so the corrected analysis returns to the *foreign*-think dumps and the effect comes out **larger**, not smaller. | recovering each model's native recipe → then the prompt-matching audit |
| C17 | Lingshu's always-reasoning is cheaper than always-direct | **A cost-methodology bug** — an `a·gen + b` fit measured at 70–407 generated tokens was extrapolated to gen = 3, giving a −16 J intercept | the user flagging a physical impossibility |
| C18 | A judged cross-modality result (0.84, "the 32B is worse") | **A racy partial-file read** — corrected to 0.749 after judging both legs fully | re-judging completely |
| C19 | The stability method was labelled "Ours" | **Relabelled** — it is a trained *baseline*, not this project's method | prior-art check |
| C20 | Finding 1's headline count, **15/20** perception cells strictly negative | **17/20** — and 19/20 within +0.02, 14/20 with 95% CIs excluding zero, pooled **−0.0401 [−0.0456, −0.0347]** on **30,250** paired samples. The published arms were **prompt-unmatched** (and resolution-unmatched for MedVLThinker); every better-matched pairing is *stronger*, so **15/20 was the outlier, not the ceiling**. Three independent correction policies all give 17/20. Two cells flip positive→negative: MedVLThinker PMC-VQA +0.0055 → **−0.0075**, Lingshu PMC-VQA +0.0115 → **−0.0425** (both `test_clean.csv`, n = 2,000). Fully-matched-only subset (nothing left to correct): 6/8 medical, 7/8 non-medical peers. | a prompt-matching audit of every Finding-1 cell (`artifacts/finding1_prompt_matching_audit.json` → `finding1_corrected_2026-07-29.json`) |
| C21 | "Reasoning helps reasoning-heavy benchmarks" **across 5 families** | **Downgraded to model-dependent, not universal.** 12/15 cells point-positive but only **4/15** CI-significant and **1/15 significantly negative**. It rests on MedVLThinker-32B (3/3 significant, and *improved* by matching: MMMU +0.0647 → **+0.0882**) plus MedGemma-27B (3/3 positive, 1/3 significant, on a fully matched pair), corroborated on MedEvalKit by MedVLThinker-32B and InternVL3-38B. Chiron-o1-8B is positive 3/3 but reaches significance nowhere. | same audit |
| C22 | **All 7 Lingshu-32B think-vs-no-think cells** (4 perception + 3 reasoning) | **WITHDRAWN, both directions.** The published "native think" prompt (`runners/run_native_think.sh:7`) is an answer-**format** string with no reasoning trigger — measured **3.0 vs 3.0–3.3** generated tokens, so no chain of thought ever occurred. Repaired with the genuinely-reasoning arm (150–259 tokens): perception **4/4 strictly negative, all CIs excluding zero, pooled −0.0866 [−0.0972, −0.0757]**; reasoning **nothing** (MMMU +0.0000, MX-R +0.0048, MX-U +0.0271, none significant). Consequence: **Lingshu-32B must not be cited as evidence that reasoning helps**, and its quoted **1.2× think:no-think latency ratio is not a reasoning ratio** — it is the ratio of two 3-token format prompts. | measuring generated tokens per arm |
| C23 | **QoQ-Med-VL-32B's reasoning gain** (MMMU +0.071) | **WITHDRAWN as reasoning-side evidence** — +0.0706 → **+0.0118** (CI [−0.0588, +0.0824]) under a matched prompt and +0.0000 fully matched; MedXpert-Understanding is significantly **negative** (−0.0433, p = 0.022). | same audit |
| C24 | The **open-text** think-vs-direct delta (Lingshu-32B, Δ = −0.154 pooled), and the pre-edit MedEvalKit `*_think` dumps | **Open text: PROVISIONAL, not withdrawn and not repairable offline.** `src/labeling/run_openvqa.py:26/27` compares a persona + "short, specific phrase / Do not explain" direct prompt against a `<think>` prompt that drops both — on free text that is a live **style/length grading channel**. A matched-prompt re-run is in flight. Separately, the pre-edit MedEvalKit `eval_results_*_think` dumps are **invalid as reasoning evidence** (2.6–3.2 generated tokens; the upstream "reason" prompt carried no reasoning trigger); the post-edit `*_reason` dumps do reason (275/561/368 tokens) but are format-unmatched. | `artifacts/pathvqa_judge_audit.json` key `prompt_confound`, then the audit |
| C25 | Prompts were assumed recoverable from the checkpoints | **They are not persisted anywhere in the JSONL rows** — recovering an arm's prompt requires tracing a `ckpts/` directory back to a shell variable in `runners/*.sh` or a module constant. This is what made the audit expensive and the defect invisible for three weeks. **Persist the prompt in every future checkpoint row.** | the audit's own method section |

## 10.2 Numeric corrections

| # | value | corrected to | source |
|---|---|---|---|
| X1 | cascade efficiency headline (three forms) | 11.34 s → 2.27 s; compute 100 → 52%; energy 6,318.8 → 1,181.9 J | `INCONSISTENCIES.md` X1 → `artifacts/master_data.csv` |
| X2 | chest X-ray box verifier 0.248 / 0.184 | **0.232 / 0.230** | `INCONSISTENCIES.md` X2 |
| X3 | "pooled-5" verifier set | **pooled-4, n = 1,064**, with the fifth dataset held out | `INCONSISTENCIES.md` |
| X4 | cascade energy saving "~4×" | **~5×** (5.3×) | `INCONSISTENCIES.md` |
| X5 | verifier gain +0.088 vs +0.116 | disambiguated by baseline (versus greedy, versus K=1) | `INCONSISTENCIES.md` |
| X6 | perception reasoning deltas at full resolution | the **operative** cap320 deltas are SLAKE +0.085, VQA-RAD +0.077 | `INCONSISTENCIES.md` X6 |
| X7 | MMMU reasoning gains +0.034 / +0.107 / +0.120 (July-3 diary) | **+0.027 / +0.100 / +0.120** — computed directly from the parsed-output files; the diary's endpoints do not exist on disk | `MedEvalKit/eval_results_*/{}/MMMU-Medical-val/`, matching `METHOD_FINAL_2026-07.md:267` |
| X8 | always-32B-reasoning accuracy 0.5632 (labelled "measured") | **0.5594 full suite / 0.5591 Variant B** — the 0.5632 version's open cells were *estimates* | `opentext_32b_think_full.json`, `f8_mode_vsthink_ci.json` |
| X9 | headline deltas +0.0117 / +0.0123 / +0.0212 / +0.0238 | **compute-lean +0.0150; accuracy-max +0.0245; accuracy-max⁺ +0.0271** (Variant B, measured) | `f8_mode_vsthink_ci.json`, `opentext_32b_think_full.json` |
| X10 | 32B capacity-verifier comparison quoted from prose | the artifact **exists**: `artifacts/verifier_32b_gpu.json`, n = **600** (not 1,064), exact-match labels, Δ +0.005 [−0.0233, +0.0317] n.s. | direct file read |
| X11 | `ckpts/acc_gen/{internvl25_8b_think, phi35v_think}` described as empty stubs | each holds **4 files / 6,050 rows** (the four perception benchmarks, flat layout) | direct listing |
| X12 | `harness.py` imported by 45 files | **39** (36 in `cascade_methods/`, 3 in `training_methods/`) | grep |
| X13 | 40 runner scripts | **38** | listing |
| X14 | PMC-VQA's human-verified `test_clean.csv` "exists on disk" | ~~**it does not** — `/data/dan/dataset/medevalkit/PMC-VQA/` holds only `test.csv` and `test_2.csv`; the clean split is listed in the dataset README and would need downloading~~ · **⚠ X14 IS ITSELF WITHDRAWN (2026-07-30) — see X20.** The claim it "corrected" was right: `test_clean.csv` *does* exist on disk. | listing — **wrong**, superseded by X20 |
| X15 | MedVLThinker PMC-VQA think−direct **+0.0055** (a perception "win") | **−0.0075** [−0.0275, +0.0120] once the think arm is resolution-matched (cap320 vs cap320). Sign flip. *(Split: `test_clean.csv`, n = 2,000.)* | `finding1_corrected_2026-07-29.json` |
| X16 | Lingshu PMC-VQA think−direct **+0.0115** (a perception "win") | **−0.0425 [−0.0625, −0.0220]**, p = 5.6e−5, with a genuinely-reasoning think arm. Sign flip, and now significant. *(Split: `test_clean.csv`, n = 2,000 — not the MedEvalKit `test_2` cell.)* | same |
| X17 | MedGemma PathVQA reasoning win **+0.0399** (from a persona-flattered think arm) | **+0.0413 [+0.0220, +0.0607]**, p = 0.0000, on a **fully matched** pair. The win is real; only its provenance changed. | same |
| X18 | Largest re-derivations, all Lingshu or MMMU: Lingshu PathVQA **−0.0170 → −0.1017**; Lingshu SLAKE −0.0096 → −0.0649; MedVLThinker SLAKE −0.0841 → −0.1274; MedGemma VQA-RAD −0.0184 → −0.0735; MedGemma MMMU −0.0118 → **+0.0353**; QoQ MMMU +0.0706 → **+0.0118**; QoQ SLAKE −0.0649 → −0.0144 | as shown | same |
| X19 | MMMU reasoning gains **+0.027 / +0.100 / +0.120** (Lingshu / MedVLThinker / InternVL3-38B) quoted without uncertainty | the values reproduce, but **only two of three are significant**: Lingshu **+0.0267 [−0.0467, +0.1000]** (n.s., n = 150), MedVLThinker **+0.100 [+0.027, +0.173]**, InternVL3-38B **+0.120 [+0.047, +0.193]**. Never quote the Lingshu figure as a gain. | `finding1_corrected_2026-07-29.json` → `medevalkit_external_corroboration.paired_with_ci` |
| **X20** *(added 2026-07-30 — a correction to **this** document)* | **this document asserted, on 2026-07-29, that PMC-VQA's human-verified `test_clean.csv` "is not on disk"** at `/data/dan/dataset/medevalkit/PMC-VQA/` and "has never been used anywhere in the repo" (§7 hole 10 and row X14 above) | **FALSIFIED 2026-07-30.** `test_clean.csv` is on disk in **two byte-identical copies** — `/data/dan/dataset/medevalkit/PMC-VQA/test_clean.csv` (418,686 bytes, **2,000** data rows, mtime 2026-06-29 07:18) and `/data/dan/dataset/pmc_vqa_train/test_clean.csv` (mtime 2026-06-08 15:55, md5 `6abfbcd088171c76a98911c5e7a8f5a0`) — and the **MedVLThinker-era cascade track has been evaluating exactly those 2,000 items all along**: `MedVLThinker-Eval`'s `pmc_vqa` slice matches `test_clean.csv` **2,000/2,000** on normalized question, `answer_label` and normalized answer text, and `ckpts/gate_7b_prune/cap320/ckpt_PMC-VQA_nothink_norag.jsonl` golds match `test_clean.Answer_label[idx]` **2,000/2,000** (every row also carries `opt_logprobs`). Consequence: hole 10's prescribed fix is already satisfied for the cascade track and is outstanding only for the MedEvalKit/Lingshu track. **Same statement, second error:** the `test_2.csv` hard-code is at `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:`**`39`**, not `:41`. **Root cause:** the 2026-07-29 pass inferred the directory contents instead of listing them, and no document recorded that the two evaluation tracks use *different* PMC-VQA splits (see §9.3). | `docs/current/PMCVQA_PROVENANCE_2026-07-30.md` §3, §3.1 **[measured 2026-07-30]**; independently re-verified 2026-07-30 by `ls -l` on both paths, `csv.reader` row counts (2,000 / 33,430), a gold-match against the cap320 checkpoint, and `sed -n '39p' MedEvalKit/utils/PMC_VQA/PMC_VQA.py` |

## 10.3 The `+0.02xx` number family, fully decoded

Five values circulate for the *same* operating point ("accuracy-max versus always-32B-with-
reasoning"). They differ on three orthogonal axes: which lever (fusion versus veto), which pool
(MMMU kept, escalated, or excluded), and whether the open-text reasoning cells were estimated or
measured.

| value | lever | pool | open reasoning cells | baseline | source |
|---|---|---|---|---|---|
| +0.0212 | veto + learning-to-defer (0.93×) | full suite, MMMU kept | estimated | 0.5632 | `method_final_v2.json` |
| +0.0207 | same | full suite, MMMU escalated (Variant A) | estimated | 0.5632 | `method_final_mmmu_corrected.json` |
| +0.0238 | fusion (1.25×) | full suite | estimated | 0.5632 | `method_final.json` |
| **+0.0245** | **veto + L2D (0.93×)** | **Variant B, n = 42,224** | **MEASURED** | **0.5591** | **`f8_mode_vsthink_ci.json` — CANONICAL** |
| +0.0271 | fusion (1.25×) | Variant B | measured | 0.5591 | `opentext_32b_think_full.json` |
| (+0.0275) | fusion | full suite | measured | 0.5594 | `progress_July_08.md` |

Companion compute-lean family: +0.0117 → +0.0123 (estimates) → **+0.0150 [+0.0107, +0.0192]
(Variant B, measured)** / +0.0154 (full suite, measured).

## 10.4 Documents that are numerically stale as of 2026-07-29

| document | what it gets wrong |
|---|---|
| `CLAUDE.md` (2026-07-02) | §0 and §1 present the MedVLThinker cascade (τ = 0.426, parity 0.572 at 74% compute) as the current headline and "ground truth". Zero occurrences of "Lingshu", "MedEvalKit", "method_final", "training_methods". §5's tree omits several directories. **The permanent abstention prohibition is absent from the file that carries the hard rules.** §6/§7/§8 (safe-cleanup procedure, landmines, environment) remain accurate and worth preserving. |
| `README.md`, `RESULTS.md` (2026-07-02) | Lead with the June MedVLThinker result. `RESULTS.md` L57 says the 7B reasoning dumps are "only PMC + MedXpert at n~500"; they are the full 8,220 rows (internal harness, so its PMC slice is `test_clean.csv`, n = 2,000). `README.md` says the 3-family matrix "is in progress"; it finished 2026-07-03/05. |
| `READING_GUIDE.md` | Six of 17 steps point at a paper draft now in `paper/archive/` with obsolete section numbering; describes the removed abstention section as part of the arc; presents the trained verifier as the latest result; nothing from July 1–8 appears. |
| `PROJECT_OVERVIEW.md` | Headline table is the pre-measurement estimate (+0.0123 / +0.0212); labels 0.5632 "(measured)"; names an archived file as "the paper" in three places; still banks the MMMU win. |
| `TECHNICAL_REPORT_2026-07.md` | Same "(measured) 0.5632" mislabel; pre-measurement headline table; open-text reasoning figures are the n=200 subsample (0.387 pooled versus the measured 0.3028); still banks MMMU; lists a completed experiment as "running". |
| `METHOD_FINAL_2026-07.md` | Baselines at 0.5632; an "open-text reasoning accuracy is estimated" footnote that is now obsolete; no Variant-B table. Mechanism descriptions are correct. |
| `RESEARCH_RESULTS_2026-07.md` | §7.7 still states +0.0123 / +0.0212 as the deployable headline; says the backlog holds 56 ideas when it holds 68. |
| `INCONSISTENCIES.md` (2026-06-27) | A whole era stale — covers only the MedVLThinker/ACC numbers, names two source-of-truth files that have since been moved, and covers none of the three live July conflicts. Two of its own stale-reference items remain unfixed. |
| `results/cascade_methods/README.md` | Its `docs/current/` index omits all four post-2026-07-02 documents; its "bottom line" is still the June story. |
| `STRUCTURE.md` | Accurate except the `paper/` section, which describes a layout reorganized four hours after that file was last refreshed; omits 45 `src/*.py` files including every headline script. |
| `artifacts/GENERALIZATION.md` + `generalization.json` | **Finding-1 section superseded** by the 2026-07-29 prompt-matching correction (15/20 → 17/20; Lingshu and QoQ withdrawals; reasoning half model-dependent). The `.md` now carries a banner and the corrected table; the `.json` is **not** annotated — anything reading `generalization.json` programmatically still gets the old cells. Findings 2 and 3 in it are unaffected. |
| `paper/build_professor_html_2026-07-27.py` + `meetings/progress_report_professor_2026-07-27.html` | Both **hard-code "15/20"** (script L37/L99/L108; HTML L216/L223) and present the reasoning half as universal. Left unedited as a dated deliverable — but **re-running that builder would republish the superseded count**. Fix the script before the next deck. |

**Two documents in `docs/current/` are not current** and should be marked or moved: `METHOD.md`
(2026-06-17) and `METHOD_deferral_router.md` (whose own first line reads "FINAL VERDICT: VADR is
NOT novel" and which calls itself superseded). `METHODS_MASTER.md` bills itself as the "single
source of truth for the paper's Method section" but predates the July-8 rewrite.

**One document in `docs/archive_mcq/` needs a banner:** `SELECTIVE_ABSTENTION.md` documents a
now-permanently-forbidden direction and says nothing to that effect.

## 10.5 An open dependency issue that needs a researcher decision (recorded 2026-07-29)

**`MedEvalKit/` carries two local, uncommitted edits** (both mtime **2026-07-02**) that changed the
reasoning arm's prompt:

- `MedEvalKit/utils/question_formats.py:11`
- `MedEvalKit/utils/MMMU/data_utils.py:158`

Upstream, the reason arm said *"Answer with the option's letter from the given choices and put the
letter in one `\boxed{}`"* — the **same answer-format contract as the direct arm** plus a `\boxed{}`
wrapper, and **no reasoning trigger at all**. The local edit **replaced** that string with *"First
reason step by step about the question and each option, then put the final answer letter from the given
choices in one `\boxed{}`"*, which adds a genuine reasoning trigger but **deletes** the
*"answer … directly"* format clause the direct arm still carries.

Two consequences, both already reflected above:

1. **Pre-edit `eval_results_*_think` dumps are invalid as reasoning evidence** — with the upstream
   prompt the models emitted **2.6–3.2** generated tokens; they never reasoned. Same failure mode as
   Lingshu's "native think" arm (C22).
2. **Post-edit `eval_results_*_reason` dumps do reason** (275 / 561 / 368 mean generated tokens) but are
   **format-unmatched** in the same clause-dropped way as the internal MedVLThinker think arm. Because
   MedEvalKit grades MCQ by letter equality with a `parse_response` that branches on the presence of
   `boxed`, the residual channel is **`\boxed{}` compliance, not answer style** — so these are reported
   as *corroboration*, not as matched evidence.

**`MedEvalKit/` is a protected dependency and was NOT modified** by the audit or by this correction
pass. The recommended repair is to **revert the dependency to upstream and pass the reasoning trigger
from the caller** (or vendor the change in this repo's own runner) so the trigger is *appended to* the
retained format clause rather than replacing it — then re-run. **That is a decision for the
researcher**; it is not made here. Verbatim diffs for both files are stored in
`artifacts/finding1_corrected_2026-07-29.json` under `medevalkit_dependency_problem`.

---

## CLOSING — the shape of the whole arc, in one paragraph

The project repeatedly discovered the same thing in different clothes: **oracle gaps are real and
large everywhere, but frozen-model signals cannot harvest them** — single-model routing at −29σ;
gate recoverability stuck at ~0.6 AUROC across sixteen mechanisms; open-text selection pinned at
the random-pick floor; structured grounding collapsing to chance once the grounder is competent;
six independent slice and fusion methods certifying zero new cells. Every genuine positive came
from *changing what is being routed rather than improving the router*: routing over **compute
configurations** instead of models, **training** a small verifier instead of reading a frozen one,
routing over **answer format** instead of using a single unified gate, and finally **re-pricing the
baseline** so the comparison is against the model a user would actually deploy. The negatives are
not filler — they are what makes the positives interpretable. And the single most valuable habit in
the record is the one that produced §10: publishing its own refutations, including the ones that
cost it a headline.

*— End of retrospective. Corrections to this document should be appended to §10 with their date and
the file that caught them.*
