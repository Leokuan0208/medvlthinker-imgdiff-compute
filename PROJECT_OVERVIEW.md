# Medical-VLM Efficiency & Accuracy — Project Overview (a 30-minute read)

*A plain-language tour of the whole project: what we set out to do, the method we ended with, and the honest map of
what's possible. Leads with where things stand **now**; the middle sections are the journey that got us there (kept as
the historical record). Every number is real — from the artifacts under `results/cascade_methods/artifacts/` and the
writeups in `results/cascade_methods/docs/current/`.*

> **Updated 2026-07-29** to match
> **[`results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md`](results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md)**
> — the definitive account of the project and the source of every number below (its §4 for results,
> §7 for the honest weaknesses, §10 for the corrections log). This file is the 30-minute version; the
> retrospective is the 3-hour version. **Where they disagree, the retrospective wins.**
>
> The two docs this used to hand off to — `docs/current/TECHNICAL_REPORT_2026-07.md` (technical
> walkthrough) and `docs/current/METHOD_FINAL_2026-07.md` (the method spec) — describe the **mechanism
> correctly** but were written on 2026-07-08 at 06:0x, hours before the measurement and the MMMU
> decision that changed the headline. Read them for *how it works*, not for *what it scores*.
>
> **⚠️ Late correction, also 2026-07-29 — the "thinking hurts perception" finding was re-derived.** The
> think-vs-direct arms behind it were **prompt-unmatched**. Correcting that makes the **perception** half
> *stronger* (**17 of 20** cells, not 15 of 20) and the **reasoning** half *weaker* (it helps some model
> families and not others). **Lingshu-32B's and QoQ-Med-VL-32B's reasoning evidence is withdrawn**, and
> the **open-ended** version of the comparison is **provisional** pending a re-run. §4 and §6 below carry
> the corrected wording; the numbers come verbatim from
> `results/cascade_methods/artifacts/finding1_corrected_2026-07-29.json` (audit:
> `…/finding1_prompt_matching_audit.json`). Full account: retrospective §5.1 and §10.1 C20–C25.

---

## 1. What this project is (in one paragraph)

A **medical vision–language model (VLM)** takes a medical **image** (a chest X-ray, a pathology slide) plus a
**question** and returns an **answer**. The strongest models are large "reasoning" models that think step by step —
**accurate but slow and power-hungry** (a 32-billion-parameter model with thinking on takes ~10 seconds and ~2
kilojoules per open-ended question). A small 7-billion model answers in ~0.3 seconds but is less accurate. **Our goal:
one method that is both *faster and more accurate* than always using the big model with thinking — across the whole
MedEvalKit medical-VQA benchmark suite, which contains both multiple-choice (MCQ) and open-ended (free-text)
questions.** The deliverable is a research paper: **`paper/adaptive-cascade-medvqa_ieee_2026-07-08.pdf`**
(9 pages, IEEEtran; source `.tex` alongside it). Earlier drafts — including the one this file used to point at,
`manuscript_final_2026-07.md` — are kept in `paper/archive/`; see `paper/README.md`.

## 2. Key terms you'll see (the only vocabulary you need)

| term | plain meaning |
|---|---|
| **Cascade / router** | Run the cheap 7B first; only call the expensive 32B on hard questions. A *router* also picks a strategy per **question format** (MCQ vs open-text). |
| **Gate** | The rule that decides "is this question hard enough to escalate to the 32B?" |
| **Confidence / margin** | How sure the model is (gap between its top-2 answer probabilities) — our MCQ gate. |
| **Think vs no-think** | The big model can emit a long reasoning trace ("think", slow) or answer directly ("no-think", fast). |
| **Verifier / best-of-N** | A small **trained** helper that scores "is this answer correct?"; sample N answers from the 7B, keep the one the verifier likes best — our open-text engine. |
| **Pandora adaptive-N** | Draw only as many samples as needed (an optimal-stopping rule), instead of a fixed N. |
| **Fusion** | On a slice where the two models are equally skilled, *combine* them to beat either alone. |
| **FLOPs / latency / energy** | Three costs: math operations (throughput) / wall-clock seconds / joules. A method can win one and lose another. |
| **FLOP-negative** | Uses **less** total compute than a single 32B forward. |
| **Recoverability / oracle gap** | Whether the 32B will *fix* a 7B error / whether a correct answer is *among* N samples. The two "walls" below are about these being real but hard to exploit. |

---

## 3. Where it stands NOW — the final method (the headline)

We built a **format-aware, regime-adaptive cascade** between Lingshu-7B and Lingshu-32B. It first detects **MCQ vs
open-text from the prompt** (never from the gold answer), then runs the right arm. It has a **Pareto knob** with two
settings, and both settings **match or beat always-Lingshu-32B-with-thinking while using less compute than a single 32B
forward.** Reproduce it all with one command: `python3 src/cascade_methods/method_final.py`.

**The reporting pool is "Variant B": 5 benchmarks / 8 cells / n = 42,224, with MMMU excluded** (see §3a). All
thresholds are 5-fold cross-fit; all CIs are 10,000-resample paired question-level bootstraps.

**Baselines.** always-32B-**think** = **0.5591** accuracy, 4.57 FLOP-eq, 10,521 ms, 2,002 J; always-32B-**no-think** =
0.5729, 4.57 FLOP-eq, 665 ms, 127 J; always-7B = 0.5549, 1.00 FLOP-eq, 347 ms. (FLOP-eq = multiples of one 7B forward;
"× a 32B call" divides by 4.57.) Sources: `artifacts/f8_mode_vsthink_ci.json`,
`artifacts/opentext_32b_think_full.json`, `artifacts/paper_baselines.json`.

| our method (held-out, n=42,224) | accuracy | Δacc vs 32B-**think** (95% CI) | FLOP-eq (× a 32B call) | latency (parallel) |
|---|---:|---:|---:|---:|
| **compute-lean** | 0.5741 | **+0.0150 [+0.0107, +0.0192]** | **2.25 (0.49×)** | **469 ms** |
| **accuracy-max** (certified veto + learn-to-defer) | 0.5836 | **+0.0245 [+0.0216, +0.0274]** | **4.26 (0.93×)** | 731 ms |
| accuracy-max⁺ (the fusion variant — *not* FLOP-negative) | 0.5862 | +0.0271 [+0.0237, +0.0305] | 5.71 (1.25×) | 668 ms |

> ⚠️ **Which number is "the" number.** **+0.0245 at 0.93×** is canonical — it is the only value that is
> simultaneously the FLOP-negative deployed configuration, MMMU-clean, *measured* (not estimated), and CI-certified
> (`artifacts/f8_mode_vsthink_ci.json`, 2026-07-09). Older docs, older artifacts and the 2026-07-27 slide deck circulate
> **+0.0212 / +0.0207 / +0.0238 / +0.0271 / +0.0275** for what looks like the same row; they differ by **lever**
> (certified-veto vs fusion), **pool** (MMMU kept / escalated / excluded) and whether the open-text think cells were
> **estimated or measured**. The full decode table is retrospective §10.3. *Two earlier versions of this file, and the
> deck, printed "Baselines (measured): always-32B-think = 0.5632" — that value's open-text cells were **estimates**.
> Corrected here to the measured 0.5591.*

**Where the win actually comes from — and it is concentrated.** 89% of the +0.0245 comes from **2 of the 8 cells**:
PathVQA-open (+0.01255, 51%) and PMC-VQA (+0.00942, 38%). Against the *deployable* baseline (always-32B-no-think, not
the thinking one), **four cells contribute exactly 0.0000** — because there the method simply *is* always-32B-no-think.
The honest one-line claim is therefore:

> *"Matches the strong model at roughly half the compute, with a significant accuracy gain on two specific cells —
> open-ended free text and PMC-VQA — plus a measured characterization of why the remaining cells are unwinnable."*

Per-benchmark: **PathVQA-open +0.076 → +0.086** (the trained verifier; the only CI-certified open beat), **PMC-VQA
+0.0135** (fusion) or **+0.0095** (certified veto) versus 32B-no-think, and the perception MCQ cells at **matched
accuracy** but a small fraction of the latency. SLAKE-open (+0.0016) and VQA-RAD-open (+0.0050) improve but their CIs
**span zero** at n = 645 / 200 — do not quote them as wins.

## 3a. Why MMMU is excluded, and what it used to claim

Earlier versions of this file banked **"MMMU +0.14 — the 7B beats the 32B there, so keep the 7B"** as a headline
per-benchmark win. **That has been retracted.** Lingshu-7B scores **0.80** on MMMU-Medical against its own *published*
54.0, and beats its own 32B (0.633). An adversarial audit (demanded by the user, 2026-07-08) checked it three ways:
model identity **PASS** (8.29 B params, correct architecture and snapshot); image ablation **DECISIVE** (0.827 with the
real image → 0.62 blank → 0.593 text-only); control model **DECISIVE** (an untuned non-medical Qwen2.5-VL-7B scores
0.567 through the identical harness). Verdict: **genuine Lingshu-7B weights, consistent with train-set contamination
outside our control.** So MMMU is dropped entirely from the reported suite. It costs the sample-weighted headline only
**−0.0005** (MMMU is 0.35% of the pool) but it materially changes the *macro* average (+0.0777 → +0.0621), which is why
it is stated rather than quietly dropped. Retrospective §2.12 and correction C12.

## 4. The one insight that made it work — the right baseline

Early on we compared our cheap method against the 32B in **no-think** mode (a single fast 665 ms forward) and wrongly
concluded "just call the 32B." The correct baseline is the 32B **with thinking on** — what you'd deploy for maximum
accuracy. We measured it, and two facts reorganized everything:

1. **Thinking *hurts* perception and is ~16× slower.** On open-ended perception, 32B-think costs **10,521 ms / 2,002 J**
   against no-think's **665 ms / 127 J**, and is *less accurate*. Per open set, **measured** on the full sets and judged
   (`artifacts/opentext_32b_think_full.json`, 2026-07-08): SLAKE-open **0.679 vs 0.819** (−0.140), VQA-RAD-open
   **0.545 vs 0.600** (−0.055), PathVQA-open **0.109 vs 0.376** (−0.267); pooled **0.3028 vs 0.537**. So for perception
   the method should *never* think. *(Earlier versions of this file quoted 0.387 pooled — that was an n=200/set
   estimate, superseded by the full measurement.)*
   > **⚠️ These open-ended accuracy numbers are PROVISIONAL (2026-07-29).** The two arms were given
   > **different prompts**: the direct arm alone gets *"You are an expert medical image analyst. Answer the
   > question with a short, specific phrase. Do not explain."* while the thinking arm gets a plain
   > `<think>`-trace instruction with neither the persona nor the answer-style constraints
   > (`src/labeling/run_openvqa.py:26/27`). On free text, that is a **live grading channel** — the direct
   > arm was told to answer in exactly the clipped style the gold answers are written in. So the
   > **−0.140 / −0.055 / −0.267 magnitudes are not yet trustworthy** (this is the same defect as
   > retrospective hole 3, and the same dumps supply the always-32B-with-reasoning baseline). A
   > matched-prompt re-run is in flight. **The cost side (16×) is unaffected** — that is measured
   > wall-clock and energy — and the *closed/MCQ* version of "thinking hurts perception" is unaffected and
   > in fact came out **stronger** (see §6).
2. **Thinking helps *only* reasoning, and only for some models** — MMMU-Medical think-gain on the faithful
   harness: **+0.100 [+0.027, +0.173] (MedVLThinker-32B)** and **+0.120 [+0.047, +0.193] (InternVL3-38B)**
   are real; **Lingshu-32B's +0.027 is NOT significant** ([−0.047, +0.100], n = 150) and must not be quoted
   as a gain. Those benchmarks are MCQ. *(Corrected 2026-07-29: the reasoning half of this finding is
   **model-dependent, not universal** — see §6.)*

So the method is **regime-adaptive**: never think on perception, reserve the slow think tier for the reasoning
residual — and it's scored against the slow always-think baseline, where a cheap method wins on both axes.

> **The honest caveat on this framing** (retrospective §7 hole 1, and it is the project's biggest open weakness): the
> dumps used as the *think* baseline for PMC-VQA, SLAKE-closed and VQA-RAD-closed average **3–4 generated tokens** and
> agree with the no-think run on 92–94% of predictions — they were produced by a harness flag that only appends *"put
> the letter in \boxed{}"*, which is not a reasoning prompt. Genuine 32B reasoning was measured on only **10.3%** of the
> pool, yet the 10,521 ms price is charged to all of it. Charging reasoning cost only where a real reasoning run exists
> turns the −95% latency claim into roughly **−72%**. **The vs-no-think and vs-oracle-mode comparisons (§3) do not have
> this problem** — against those, compute-lean is a tie and accuracy-max wins by +0.0106 [+0.0085, +0.0126].

## 5. How the method works — two engines

- **MCQ arm = the efficiency engine.** 7B answers; a **confidence-margin gate** (escalate iff `margin < τ`) sends only
  the low-confidence questions to the 32B in **no-think** mode. Pooled escalation is **16.2%**, but it varies enormously
  per benchmark — PMC-VQA 8.5%, SLAKE-closed 20.5%, PathVQA-closed 45.7%, VQA-RAD-closed 57.0%, MedXpert 89.6%. On
  **PMC-VQA** the accuracy-max setting adds a lever: either *fuse* the two models on the ~33% of items where they
  disagree (v1) or apply a **certified veto** — keep the 7B answer inside confidence bins where a Wilson lower bound on
  its precision beats the 32B, and never run the 32B there (v2, cheaper). A **prefill-prefetch** trick (run the 32B's
  image prefill concurrently with the 7B pass) buys **461 → 405 ms, −12.1%** at zero accuracy change — it is *documented
  but deliberately not folded into the headline*, because unconditional prefetch pays the 32B prefill on every query.
- **Open-text arm = the accuracy engine.** The 7B samples several answers and a **trained LoRA verifier**
  (`ckpts/train/lora_verifier_pooled4`, per-answer AUROC 0.924) picks the best. **Pandora adaptive-N** — Weitzman's
  optimal-search rule — draws only as many samples as needed (mean N = 3.45 / 3.91 / 5.48 per set, −33% open-arm
  compute at iso-accuracy), and a team-objective learning-to-defer rule decides when to escalate. This arm is
  **5.55% of the questions but carries essentially the entire vs-think win** (Δ +0.2699 on the open pool). Remove it
  and compute-lean's advantage disappears (MCQ-only Δ +0.0006, CI spans zero).

Two corrections we banked along the way: on **Lingshu**, the plain **margin** gate beats *agreement* and *CASP-stability*
(those won on the earlier MedVLThinker family — gate choice is model-specific); and a **two-arm router is required**
(no single gate works for both formats).

---

## 6. The journey, part 1 — ACC: the efficiency structure (kept as history)

The efficiency engine grew out of **ACC (Adaptive-Compute Cascade)**. The key discovery: the big 32B's *fast* no-think
mode is **as good as or better than** its slow think mode on perception — *thinking over-thinks it*. So we cascade over
*compute configurations*: `cheap 7B → big 32B (fast) → big 32B (slow think)`, with the expensive think pass firing only
on the reasoning residual (~15–18%). At equal accuracy vs always-big-think (MedVLThinker, 6-benchmark):

| | always big-think | **ACC** | change |
|---|---|---|---|
| accuracy | 0.572 | 0.569 | parity |
| latency | 11.34 s | **2.27 s** | **−80%** |
| energy | 6,319 J | **1,182 J** | **~5× less** |
| compute | 100% | **52%** | halved |

This regime-adaptive tiering is what the final method's MCQ arm inherits — though on Lingshu the slow think tier is
never deployed at all, because Lingshu's reasoning gain is ~0 so the tier would fire ~0% of the time. (Detail:
`docs/current/METHOD_ACC.md`; figure `paper/figs/fig1_latency_accuracy_frontier.png`, from the archived draft.)

> ### ⚠️ The cross-family version of this finding, corrected 2026-07-29
> This is the project's flagship *scientific* claim, so it got audited. The think and direct arms had been run
> with **different prompts** (and, for MedVLThinker, different image resolutions), so they differed by more
> than "did the model think?". Re-derived from the best-matched arms already on disk
> (`artifacts/finding1_corrected_2026-07-29.json`; audit `…/finding1_prompt_matching_audit.json`):
>
> **The defensible statement.** *Chain-of-thought reasoning does not pay for itself on perception-style
> medical visual QA: on prompt- and resolution-matched arms, thinking is strictly worse than answering
> directly in **17 of 20** (family × benchmark) perception cells across 5 medical VLM families — **14 of 20**
> with 95% confidence intervals excluding zero, pooled **−0.040 [−0.046, −0.035]** over **30,250** paired
> samples — and it reproduces at the same strength on arms that differ by nothing but the reasoning
> instruction. On reasoning-heavy benchmarks CoT helps some families (MedVLThinker-32B, MedGemma-27B,
> InternVL3-38B) but not others (Lingshu-32B, QoQ-Med-VL-32B): the reasoning-side gain is model-dependent,
> not universal.*
>
> In plain terms, six things changed:
> 1. **The perception half got stronger, not weaker** — 15/20 → **17/20** (and 19/20 no better than +0.02).
>    Three different ways of fixing the mismatch all give 17/20, so **the published 15/20 was the pessimistic
>    outlier**. Two cells even flipped from "thinking helped a little" to "thinking hurt".
> 2. **The reasoning half got downgraded.** 12 of 15 cells still point the right way, but only **4** have a
>    confidence interval that excludes zero, and **one is significantly negative**. It rests on
>    MedVLThinker-32B (3/3 significant) and MedGemma-27B (3/3 positive, 1/3 significant).
> 3. **Lingshu-32B's 7 cells are withdrawn entirely.** Its "native thinking" prompt turned out to be only an
>    *answer-format* instruction with no reasoning trigger — the model emitted **3 tokens** and never thought.
>    With a prompt that really makes it think, thinking **hurts** Lingshu's perception accuracy by
>    **−0.087 [−0.097, −0.076]** and gains it **nothing** on the reasoning benchmarks. So the line "Lingshu's
>    reasoning gain is ~0" above is still true — but for a better reason than we had.
> 4. **A knock-on:** Lingshu's often-quoted **"1.2× think:no-think" cost ratio is not a reasoning ratio** —
>    it is the ratio of two 3-token prompts. Do not present it as the price of Lingshu reasoning.
> 5. **QoQ-Med-VL-32B is withdrawn as reasoning evidence** (its MMMU gain +0.071 → +0.012, interval spans
>    zero; one MedXpert cell is significantly *negative*).
> 6. **One genuine exception survives:** **MedGemma-27B on PathVQA, +0.041 [+0.022, +0.061]**, on a fully
>    matched pair. It is the only perception cell where thinking really helps.
>
> Nothing here changes the **method** — if anything the perception effect it exploits is larger than we
> claimed. What changed is what we may *say*, and the process lesson: **prompts were never stored in the
> checkpoint files** (they lived only in shell variables in `runners/*.sh`), which is why a mismatch survived
> three weeks of review. Full account: retrospective §5.1, §2.2, §10.1 C20–C25, §10.5.
*Honest notes:* (a) the agreement rule ACC originally used is **not novel** — it is Agreement-Based Cascading
(arXiv 2407.02348), retracted in the same document that proposed it; the contribution is the **structure** (the fast
middle tier). (b) These are **internal-harness** numbers (evaluation context B, 6 benchmarks / 8,220 samples) and must
**never** be mixed with the MedEvalKit figures in §3 — see retrospective §9.3. (c) On Lingshu the simple margin gate is
what we deploy: gate choice turned out to be **model-specific** (margin AUROC 0.7254 vs agreement 0.6565, and
resolution-stability is *inert* because the 7B is 98.95% stable between cap320 and full resolution).

## 7. The journey, part 2 — the trained verifier (the accuracy engine, kept as history)

Frozen models hit a "luck floor": sample the 7B several times and a correct answer is often there, but you **can't pick
it better than random** without training. **Training a small verifier breaks that floor.** For free-text answers it
recovers **35–49%** of the oracle gap and transfers to unseen datasets:

| dataset | cheap 7B | **+ trained verifier** | best possible (oracle) |
|---|---|---|---|
| PathVQA | 0.352 | **0.441** | 0.513 |
| VQA-RAD | 0.519 | **0.611** | 0.722 |
| SLAKE | 0.738 | **0.762** | 0.895 |
| **average** | 0.413 | **0.501** | 0.592 |

It genuinely tells right from wrong (AUROC 0.924), uses the image (blanking it costs −0.047), transfers zero-shot to a
held-out fifth dataset (+0.024), and shows a real test-time-scaling curve (0.385 → 0.501 over K = 1 → 8 while random
selection stays flat). It also works on **structured** outputs: organ bounding boxes 0.197 → 0.255 (40% of the oracle
gap) and chest X-ray boxes on the real MS-CXR benchmark 0.041 → 0.232 (**78%**). Bootstrap CIs: free text
**+0.116 [+0.092, +0.139]**; chest X-ray boxes **+0.191 [+0.152, +0.232]**. Its best-of-N is **competitive with the
single 32B answer** on open text — this is the open-text arm. (Detail: `docs/archive_mcq/TRAINED_VERIFIER_RESULT.md`,
`docs/archive_mcq/BOX_VERIFIER_RESULT.md`.)

> **Two honest annotations added 2026-07-29.** (a) "The verifier **beats** the 32B" was **downgraded to "competitive
> with / matches"** on 2026-07-08: it is +0.039 on seed 0 but **ties on seed 1** (retrospective C11). (b) The verifier
> is trained on a 70/30 grouped split of these same four open sets and then scored over the *full* sets, so ~70% of
> every reported open item was in its training data; measured on **selection gain** the inflation is one-directional
> and about **31%** (full +0.1040 vs held-out +0.0718, n-weighted). Retrospective hole 4 — open, not fixed.

## 8. The journey, part 3 — the walls (the honest negatives — ~90 of them)

Half the contribution is a precise map of **what is impossible**, which makes the positives meaningful. Roughly **90
distinct attempts** failed, and the retrospective (§6) groups them by the principle that killed each one. Together they
bound the problem:

- **The recoverability wall.** To beat the 32B on an MCQ, you must predict *ex ante* which questions it will fix — a
  weak signal (**AUROC ≈ 0.5–0.6**, because the models fail on the same questions). **Sixteen independent mechanisms**
  hit it: hidden-state probes, kNN gates, self-verification, gradient-boosted gates, rich fused features, the published
  post-hoc deferral rule, cross-family and image-based routers, a full LoRA fine-tune, trained open-text gates,
  logit-level fusion, decision-level fusion, super-learner ensembling, learned slice discovery, credibility shrinkage,
  and the open-ended cascade. Result: the **only** closed-MCQ slice we can certifiably beat the 32B on is **PMC-VQA**.
  Automatic slice discovery, given the whole feature space, found **1.62 genuinely-new slices per split — *below* a
  permutation null of 5.61**, i.e. nothing.
- **The selection wall.** On open-text, a correct answer is often among N samples but the verifier converts only
  **74–82%** of that oracle gap. **Thirteen attempts** hit it, killed three orthogonal ways: *capacity* (a 32B zero-shot
  verifier merely ties the trained 7B, Δ +0.005 [−0.023, +0.032], n=600), *compounding* (diverse generation × pairwise
  comparison do not stack — pairwise-over-diverse is **−0.0117**), and *pre-filtering* (no filter beats plain
  diverse+pointwise). It is not an "answers are too long" artifact either — efficiency is 79% / 90% / 80% on short /
  medium / long answers.
- **The coverage wall is bigger than both** — and this is the newest finding. Of 1,064 held-out questions, **434 (40.8%)
  have no correct answer anywhere in the 8-sample pool**, while the *entire* selection gap is 97 questions (0.0912).
  **Generator ideas compete for +0.408; verifier ideas compete for at most +0.091.** Size future work accordingly.
- **The two walls hand work to each other.** Raising oracle coverage with diverse generation (+0.110) did **not** raise
  accuracy (+0.015 converted) — it *relocated* the residual from coverage into selection. Only attack coverage once
  selection is solved.
- **The MCQ-vs-open-ended twist.** The wall is worst on multiple-choice (a single A/B/C/D letter is a coarse signal;
  AUROC ~0.6) and much weaker on free-text (~0.87) — which is exactly where the verifier works. **Corollary that must
  travel with it: detection ≠ cascade gain.** Detection AUROC rose 0.66 → 0.85 while the oracle-minus-cheap headroom
  moved only +0.02 → +0.06. One dataset has 7B 0.302 ≈ 32B 0.301 (zero headroom) at detection AUROC 0.749. *Knowing an
  answer is wrong is not the same as having somewhere better to send it.*

Levers that *don't* help the way hoped (all documented): INT4 quantization (a VRAM/energy win, not a FLOP win under our
MAC-count unit); image-token pruning (real projected −26% but risky on radiology, deferred); test-time adaptation of the
cheap leg (entropy-min **collapses** it, −0.159) and neuro-symbolic constraint gates (strict constraints fire on ~1
sample; the shared confident-wrong errors are perceptual, not logical); retrieval/RAG (the 32B fixes genuinely-unknown
errors *equally* on knowledge and perception questions, 38% vs 36% ⇒ it is a capacity gap, not a knowledge gap);
few-shot in-context learning (PathVQA **0.343 → 0.203**); best-of-N as a deployable method (compute-dominated: 16–30
units against always-32B's 4.57). (Detail: retrospective §6; `RESEARCH_RESULTS_2026-07.md` §2, §6, §7.)

> **⛔ Note on scope:** *abstention / reject-option / deferring a question to a human is **permanently forbidden** in
> this project (made permanent 2026-07-07). It appears nowhere in the method, the paper or the backlog — **the method
> always produces an answer**. A training-free abstention mechanism was built and validated in June and was discarded
> for scope, not because it failed; its files survive as historical record only. The "certified veto" is **not**
> abstention — it keeps the cheap model's answer.*

---

## 9. Where the project stands

- **A reproducible method** (`src/cascade_methods/method_final.py`) that matches or beats always-Lingshu-32B on the
  MedEvalKit suite (both formats), with a Pareto knob whose **both** operating points use less compute than one 32B
  forward. *Caveat worth stating plainly:* `method_final.py` is a **CPU re-costing of saved per-sample dumps** —
  escalation is `np.where(margin < τ, ok_32B, ok_7B)` over recorded correctness, and latency/energy come from per-leg
  batch-1 constants. **The final method has never been executed end-to-end as a live pipeline.** (The one genuine live
  cascade in the repo, `ckpts/rt_cascade_cap320.jsonl`, belongs to the older MedVLThinker work.)
- **A faithful 3-family × 7-benchmark reproduction** underneath the baselines — anchored by Lingshu-32B on MMMU-Medical
  = 0.633 against the paper's 62.3, *exact*. 6 of 7 benchmarks fully faithful plus one cheap-faithful (Lingshu-7B
  OmniMed **0.8274** vs paper 0.829 on all 88,996 questions); the OmniMed strong leg is a documented infra fallback
  (a deterministic 2-GPU NCCL hang, `docs/current/OMNIMED_FALLBACK.md`) — **no fabricated metrics file was written**.
- **A map of the limits confirmed dozens of times over** (recoverability ×16, selection ×13, plus the coverage
  measurement) — a genuine negative-results contribution, not a caveat. The retrospective argues this
  characterization is a **stronger** contribution than the +0.0245 itself, because it survives the honest re-costing.
- **A stated list of the method's own weaknesses** — retrospective §7 ranks 16 holes, three of them critical. Anyone
  quoting this project's numbers should read it first.
- **A full, sourced audit trail:** the retrospective (`PROJECT_RETROSPECTIVE_2026-07-29.md`), the method spec
  (`METHOD_FINAL_2026-07.md`), the results ledger (`RESEARCH_RESULTS_2026-07.md`), the technical report
  (`TECHNICAL_REPORT_2026-07.md`), **13 daily diaries** (`progress/`, June 17 → July 8 — the most trustworthy layer in
  the tree), and the 68-idea cross-field backlog (`METHOD_IDEAS_BACKLOG.md`).
- **The paper:** `paper/adaptive-cascade-medvqa_ieee_2026-07-08.{tex,pdf}` — 9 pages, IEEEtran, and the *only* prose
  artifact besides the July-27 deck that carries the corrected numbers. Earlier drafts are in `paper/archive/`.
  **⚠️ As of 2026-07-29 the `.tex` carries the corrected Finding 1 and the `.pdf` does not** — the PDF and
  `paper/figs_final/fig_overthink.pdf` must be regenerated (`python3 paper/make_ieee_figs.py`, then
  `bash paper/build_ieee.sh paper/adaptive-cascade-medvqa_ieee_2026-07-08.tex`). The **July-27 deck still shows the
  superseded 15/20** and its builder `paper/build_professor_html_2026-07-27.py` hard-codes it — fix that script
  before producing another deck.

> **⚠️ Preservation warning.** The last git commit is `8cdefef` (2026-07-02). The IEEE paper, all nine July diaries,
> `paper/figs_final/`, the July-27 deck and every script in the live headline chain are **untracked**; `results/` and
> `MedEvalKit/` are gitignored. **Committing the working tree is the top-priority chore**, ahead of any new experiment.

**Where to read more (in order):** **`docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md`** (start here — the definitive
account) → this file → `meetings/progress_report_professor_2026-07-27.html` (13 source-cited sections) →
`paper/adaptive-cascade-medvqa_ieee_2026-07-08.pdf` → `docs/current/TECHNICAL_REPORT_2026-07.md` (mechanism; pre-seam
numbers) → `docs/current/METHOD_FINAL_2026-07.md` (the spec; same caveat) → `progress/progress_July_04..08.md` →
`READING_GUIDE.md` for the full ordering.

**What to do next** is sequenced in retrospective §8, integrity work first: (1) re-cost the baselines honestly and
commit the tree; (2) a 30-minute verifier image-ablation that could invalidate a load-bearing conclusion; (3) a
noise-ceiling measurement as the program-level stop/go instrument. **Open upside (small, non-headline):** a GPU confirm
of image-token pruning — the one remaining lever that *reduces* the strong leg's cost rather than avoiding it, pending
a radiology-safety check on the chest X-ray box set.
