# Medical-VLM Efficiency & Accuracy — Project Overview (a 30-minute read)

*A plain-language tour of the whole project: what we set out to do, the method we ended with, and the honest map of
what's possible. Leads with where things stand **now**; the middle sections are the journey that got us there (kept as
the historical record). Every number is real — from the artifacts under `results/cascade_methods/artifacts/` and the
writeups in `results/cascade_methods/docs/current/`.*

> **Updated 2026-07-30** (two settled corrections propagated — see the box below) to match
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

> ## ⚠️⚠️ TWO SETTLED CORRECTIONS, 2026-07-30 — READ BEFORE §3
>
> **1 — The accuracy metric is MACRO now, and the compute claim REVERSES.** The headline average is
> **equal weight per reporting cell (8 cells, 1/8 each)**, not sample-weighted, because sample-weighting
> let PMC-VQA (**79% of the items**) speak for the method. Reweighting: PMC **79.2% → 12.5%**; the
> open-text arm **5.6% of items → 37.5% of weight**; closed/MCQ **94.4% → 62.5%**.
> **Under equal weight NO operating point is compute-cheaper than always-32B-direct** — compute-lean
> **1.196×** (was 0.492×), accuracy-max **1.410×** (was 0.932×), fusion **1.435×** (was 1.250×).
> **"Matches the strong model at roughly half the compute" and "Pareto-dominates every fixed way of
> using the 32B" are both RETIRED.** §3 is rewritten. Source:
> `results/cascade_methods/artifacts/macro_average_headline_2026-07-30.json`; account: retrospective
> §4, §10.1 C26, §10.2 X21.
>
> **2 — Finding 1's reasoning half is an ANSWER-FORMAT effect (matched re-run complete, 6/6 cells).**
> With the answer format matched, the explicit reasoning *instruction* is worth ~nothing (**0/9**
> sub-cells CI-significant); the **`\boxed{}` request** is what carries the published gains (**3/9**
> significant). **Asking for `\boxed{}` is itself a reasoning trigger.** §4 and §6 carry the corrected
> wording. Source: `…/artifacts/medeval_matched_direct_2026-07-29.json`; account: retrospective §5.1,
> §10.1 C27, §10.2 X22.
>
> **⏳ And one thing left OPEN.** The **open-text accuracy claim is PROVISIONAL — a clean-verifier
> (disjoint-split) retrain is in progress** and will determine whether it is contaminated (the verifier
> was trained on ~70% of its own evaluation items). Marked as such everywhere below; the outcome is not
> pre-judged.

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
| **FLOP-negative** | Uses **less** total compute than a single 32B forward. *(A property of a weighting, not of the method: our operating points are FLOP-negative sample-weighted and FLOP-positive at equal weight per benchmark cell — §3.)* |
| **Macro vs sample-weighted** | **Macro** = average the 8 benchmark cells with equal weight (1/8 each) — the headline convention since 2026-07-30. **Sample-weighted** (a.k.a. pooled) = average the 42,224 items, which gives PMC-VQA 79.2% of the suite. They disagree, so every number must say which it is. |
| **Recoverability / oracle gap** | Whether the 32B will *fix* a 7B error / whether a correct answer is *among* N samples. The two "walls" below are about these being real but hard to exploit. |

---

## 3. Where it stands NOW — the final method (the headline)

We built a **format-aware, regime-adaptive cascade** between Lingshu-7B and Lingshu-32B. It first detects **MCQ vs
open-text from the prompt** (never from the gold answer), then runs the right arm. It has a **Pareto knob** with two
settings. Both settings **beat always-Lingshu-32B-with-thinking on accuracy at ~a tenth of its latency and energy** —
but, at equal weight per benchmark cell, **neither uses less compute than a single 32B forward**, which is a change
from what this file said before 2026-07-30. Reproduce it all with one command:
`python3 src/cascade_methods/method_final.py`; the macro re-basing with
`python3 src/cascade_methods/macro_average_headline.py`.

**The reporting pool is "Variant B": 5 benchmarks / 8 cells / n = 42,224, with MMMU excluded** (see §3a). All
thresholds are 5-fold cross-fit; all CIs are 10,000-resample paired question-level bootstraps.
**The headline average is MACRO — equal weight per cell, 1/8 each** (decided 2026-07-30). The 5-benchmark macro is a
secondary robustness check; the old **sample-weighted** convention is shown alongside for contrast, never mixed with a
macro number.

> **Which PMC-VQA split.** Every PMC-VQA number in §3–§9 of this file is the MedEvalKit/Lingshu track = **`test_2.csv`
> (v2, 33,430 items)** — hard-coded in vendor code at `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39`, and **79%** of this
> pool. The project's *other* track (the June MedVLThinker cascade, §6) used a **different** file: **`test_clean.csv`
> (v1, 2,000 items — the authors' only human-verified split)**, which is **24.3%** of its 8,220-item pool. The two
> overlap on **6 items**, so their numbers are not interchangeable. Details:
> [`results/cascade_methods/docs/current/PMCVQA_PROVENANCE_2026-07-30.md`](results/cascade_methods/docs/current/PMCVQA_PROVENANCE_2026-07-30.md).

**Baselines, macro (8 cells) with the sample-weighted value in brackets; costs are macro-weighted unless flat.**
always-32B-**think** = **0.5974** [0.5591], 4.57 FLOP-eq as charged (**5.70 honestly re-costed, macro**; 4.87
sample-weighted), 10,521 ms as charged (**6,291 ms honest macro**; 2,018 ms honest sample-weighted), 2,002 J
(**1,625 J honest macro**; 487 J honest sample-weighted); always-32B-**no-think** = **0.6567** [0.5729], 4.57
FLOP-eq, 665 ms, 127 J (flat, weighting-invariant); **oracle-mode-32B** = **0.6573** [0.5730], 4.57,
**1,897 ms / 361 J as charged macro** (860 ms / 164 J sample-weighted; 646 ms / 122 J honest macro); always-7B =
**0.5971** [0.5549], 1.00 FLOP-eq, 347 ms, 45.8 J (flat). (FLOP-eq = multiples of one 7B forward; "× a 32B call"
divides by 4.57.) Sources: `artifacts/macro_average_headline_2026-07-30.json`, `artifacts/f8_mode_vsthink_ci.json`,
`artifacts/opentext_32b_think_full.json`, `artifacts/paper_baselines.json`, `artifacts/honest_recosting_2026-07-29.json`.

**PRIMARY — macro over the 8 reporting cells (1/8 each):**

| our method (held-out, 8 cells / 42,224 items) | accuracy | Δacc vs 32B-**think** (95% CI) | Δacc vs 32B-**no-think** (95% CI) | FLOP-eq (× a 32B call) | latency par. / seq. |
|---|---:|---:|---:|---:|---:|
| **compute-lean** | 0.6600 | **+0.0626 [+0.0514, +0.0734]** | +0.0033 [−0.0054, +0.0121] n.s. | 5.47 (**1.196×**) | 650 / 1,292 ms |
| **accuracy-max** (certified veto + learn-to-defer) | **0.6694** | **+0.0720 [+0.0614, +0.0824]** | **+0.0128 [+0.0056, +0.0200]** | 6.44 (**1.410×**) | 691 / 1,334 ms |
| accuracy-max⁺ (the fusion variant) | 0.6661 | **+0.0686 [+0.0582, +0.0790]** | **+0.0094 [+0.0013, +0.0176]** | 6.56 (**1.435×**) | 665 / 1,350 ms |

**THE OLD SAMPLE-WEIGHTED CONVENTION, kept for contrast — never pair these costs with the accuracies above:**
compute-lean 0.5741, +0.0150 [+0.0107, +0.0192] vs think, **2.25 FLOP-eq (0.492×)**, 469 ms; accuracy-max 0.5836,
+0.0245 [+0.0216, +0.0274], **4.26 (0.932×)**, 731 ms; fusion 0.5862, +0.0271 [+0.0237, +0.0305], 5.71 (1.250×), 668 ms.
Secondary macro over the **5 benchmarks**: 0.6131 / 0.6223 / 0.6200 at 1.098× / 1.310× / 1.370×.

> ⚠️ **Which number is "the" number (changed 2026-07-30).** The canonicity rule is now **macro over 8 cells, Variant B,
> measured, veto lever ⇒ +0.0720 [+0.0614, +0.0824] versus the reasoning baseline** (and **+0.0128 [+0.0056, +0.0200]**
> versus the deployable always-32B-no-think). **+0.0245 at 0.93× is now "the sample-weighted equivalent"**, not the
> canonical value. Older docs, older artifacts and the 2026-07-27 slide deck circulate **+0.0212 / +0.0207 / +0.0238 /
> +0.0271 / +0.0275** for what looks like the same row; they differ by **lever** (certified-veto vs fusion), **pool**
> (MMMU kept / escalated / excluded), whether the open-text think cells were **estimated or measured**, and — the new
> fourth axis — **which weighting**. The full decode table is retrospective §10.3. *Two earlier versions of this file, and
> the deck, printed "Baselines (measured): always-32B-think = 0.5632" — that value's open-text cells were **estimates**.
> Corrected to the measured 0.5591 sample-weighted / 0.5974 macro.*

### The cost claim reversed — say this plainly

**Under equal weight per cell, no operating point is compute-cheaper than a single 32B forward.** Latency and energy
versus always-32B-no-think also flip or shrink: compute-lean is **−2.3% parallel latency but +94.3% sequential and
+48.0% energy**; accuracy-max **+4.0% / +100.6% / +62.4%**. **"Pareto-dominates every fixed way of using the 32B" is
retired** — the method points are still **non-dominated**, but because they are more *accurate*, not because they are
cheaper.

**Why.** Escalation is wildly heterogeneous — PMC-VQA **8.45%**, SLAKE-closed 20.45%, PathVQA-closed 45.72%,
VQA-RAD-closed 56.97%, MedXpert **89.60%**, SLAKE-open 15.81%, VQA-RAD-open 12.50%, PathVQA-open 35.67% — and PMC-VQA,
the **lowest-escalation** cell, carried **79.2%** of the sample-weighted average. Equal-weighting takes multiple-choice
escalation from **16.22% → 44.24%**. The three open cells cost the method **7.6–12.6 FLOP-eq** against the baseline's
flat 4.57, while PMC costs only **1.39**.

> **⚠️ The nuance that must travel with every cost number.** Macro-averaging **cost** answers a *different question*
> from sample-weighted cost. Cost is **additive per query**, so on traffic resembling this suite the **~0.49×** saving is
> what you would actually pay. The macro number instead tests whether the saving **generalises across task types** — and
> it does not: it is **concentrated on the low-escalation multiple-choice cells**. **Report accuracy on macro and BOTH
> cost numbers, each labelled for what it means.**
> **The defensible joint claim:** *large latency and energy savings against a reasoning baseline; compute savings that
> are real but concentrated on low-escalation multiple-choice traffic rather than uniform.*

**What survives, and what fails, at equal weight.**
- **Survives:** versus a 32B *actually made to reason*, accuracy-max is **+0.0720 [+0.0614, +0.0824]** at **−89%
  latency and −87% energy** (honestly re-costed) — though **not** fewer FLOP-eq (**1.41×** as charged, **1.13×**
  honestly re-costed). Versus always-32B-no-think, accuracy-max is **+0.0128 [+0.0056, +0.0200]**.
- **Fails:** compute-lean neither matches the strong model on the multiple-choice half (**−0.0070 [−0.0126, −0.0017] —
  a SIGNIFICANT LOSS**; versus oracle-mode **−0.0080 [−0.0137, −0.0024]**) nor stays cheap (**1.20×**).
- **The honest one-line claim** (verbatim from the artifact's `honest_headline`):
  > *"Under equal weight per benchmark cell (8 cells, 1/8 each), the accuracy-max setting beats always-32B-direct on
  > accuracy by +0.0128 [+0.0056, +0.0200] — but it now costs 1.41× that baseline's compute, not less, and the
  > compute-lean setting neither matches it on the multiple-choice half (−0.0070 [−0.0126, −0.0017], a significant loss)
  > nor stays cheap (1.20× compute, up from 0.49× sample-weighted); the one baseline the method still clearly beats at
  > equal weight is a 32B actually made to reason (+0.0720 [+0.0614, +0.0824] accuracy, −89% latency, −87% energy —
  > though not fewer FLOP-eq: 1.41× as charged, 1.13× honestly re-costed)."*

**Where the win comes from — concentration, measured honestly.** The old *"89% of the +0.0245 comes from 2 of the 8
cells"* phrasing is **retired**: under equal weight each cell contributes exactly 1/8 of its own delta, so the
contribution-share table collapses to the per-cell delta list. Concentration is now reported as a **leave-one-cell-out
range**: accuracy-max vs 32B-think **+0.0720, LOO [+0.0318, +0.0830]** (load-bearing cell **PathVQA-open**); accuracy-max
vs 32B-no-think **+0.0128, LOO [+0.0023, +0.0146]**; compute-lean vs 32B-no-think on MCQ **−0.0070, LOO
[−0.0085, −0.0038]** (load-bearing cell **PMC-VQA**). Note the macro CI is **2–4× wider** than the pooled one, because
the small cells now count as much as PMC-VQA — that is the point of the convention, not a defect.

Per-benchmark: **PathVQA-open +0.076 → +0.086** (the trained verifier; the only CI-certified open beat), **PMC-VQA
+0.0135** (fusion) or **+0.0095** (certified veto) versus 32B-no-think — both on **`test_2.csv`, n = 33,430** — and the
perception MCQ cells at **matched accuracy** but a small fraction of the latency. SLAKE-open (+0.0016) and VQA-RAD-open
(+0.0050) improve and their CIs **span zero** at n = 645 / 200 — but under equal weight they now carry **12.5% of the
headline each**, and their wide intervals are exactly what widens the macro CI, so **keep them in and report their own
CIs honestly** rather than dropping them as "too small to quote".

> **⏳ PROVISIONAL — clean-verifier retrain in progress.** Every open-text accuracy above (PathVQA-open, SLAKE-open,
> VQA-RAD-open, and the open-pool deltas) depends on a verifier that was trained on ~70% of its own evaluation items
> (retrospective §7 hole 4). A **disjoint-split retrain is in flight** (`artifacts/verifier_disjoint_split.json`) and
> will determine whether the open-text accuracy claim is contaminated. Because the open arm holds **37.5% of the macro
> weight** and is the load-bearing cell of every headline delta, this is the project's highest-leverage open item. Its
> outcome is **not** pre-judged here.

## 3a. Why MMMU is excluded, and what it used to claim

Earlier versions of this file banked **"MMMU +0.14 — the 7B beats the 32B there, so keep the 7B"** as a headline
per-benchmark win. **That has been retracted.** Lingshu-7B scores **0.80** on MMMU-Medical against its own *published*
54.0, and beats its own 32B (0.633). An adversarial audit (demanded by the user, 2026-07-08) checked it three ways:
model identity **PASS** (8.29 B params, correct architecture and snapshot); image ablation **DECISIVE** (0.827 with the
real image → 0.62 blank → 0.593 text-only); control model **DECISIVE** (an untuned non-medical Qwen2.5-VL-7B scores
0.567 through the identical harness). Verdict: **genuine Lingshu-7B weights, consistent with train-set contamination
outside our control.** So MMMU is dropped entirely from the reported suite.

> **⚠️ Updated 2026-07-30 — the rationale INVERTS now that the metric is macro.** The exclusion used to be defended as
> immaterial: it costs the *sample-weighted* headline only **−0.0005**, because MMMU is 0.35% of the items. Under
> **equal weight per cell MMMU would carry 1/9 = 11.1% of the headline**, so excluding the contaminated cell is a
> **large and consequential decision** and must be re-argued **on contamination grounds alone**. Its size, stated
> explicitly (macro-9 vs macro-8, versus always-32B-no-think): compute-lean **+0.0215 vs +0.0033**, accuracy-max
> **+0.0299 vs +0.0128**, fusion **+0.0269 vs +0.0094**. MMMU cell accuracies: 7B **0.800**, 32B-direct 0.6333,
> 32B-reasoning 0.660, method 0.800. Under macro the *danger* of including it is larger, not smaller.

Retrospective §2.12 and correction C12.

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
2. **Thinking helps *only* reasoning — and the lever turns out to be the ANSWER FORMAT, not the reasoning
   instruction** *(settled 2026-07-30)*. MMMU-Medical think-gain on the faithful harness:
   **+0.100 [+0.027, +0.173] (MedVLThinker-32B)** and **+0.120 [+0.047, +0.193] (InternVL3-38B)** are real
   *as magnitudes*; **Lingshu-32B's +0.027 is NOT significant** ([−0.047, +0.100], n = 150) and must not be
   quoted as a gain. Those benchmarks are MCQ.
   > **⚠️ The attribution changed.** A matched-prompt re-run (complete, **6/6 family × benchmark cells**,
   > 9 sub-cells, n = 145 / 1,446 / 554, paired on item id) decomposes each published gain into an
   > **answer-format** half (`\boxed{}` vs bare letter, both trigger-free) and an **explicit-trigger** half.
   > **0/9 trigger effects are CI-significant; 3/9 format effects are.** Per cell, published / format /
   > trigger: MVT-32B MMMU +0.103 / +0.062 / +0.041 n.s.; MVT MX-R +0.046 / **+0.046 SIG [+0.019, +0.072]** /
   > +0.001 n.s.; IV3-38B MMMU +0.124 / **+0.090 SIG** / +0.035 n.s.; Lingshu MMMU +0.028 / −0.014 /
   > +0.041 n.s. **Asking for the answer in `\boxed{}` is itself a reasoning trigger** — MedVLThinker emits
   > 431–580 tokens on 99–100% of items and InternVL3 193–289 on 94–95% with *no trigger present*; Lingshu
   > never does (3–4 tokens). `parse_ok ≥ 0.9986` in every new arm (the minimum over the 9 sub-cells; exactly 1.000 in 6 of them), so it is not an extraction artifact.
   > **So: drop "a reasoning instruction improves accuracy on reasoning-heavy benchmarks"; keep "getting a
   > reasoning-tuned model to emit a trace helps substantially, via the answer format".** The cascade's
   > gated-reasoning tier keeps its **full** value — the whole ladder (bare letter → boxed → boxed+trigger)
   > is what a think tier delivers; only the attribution changes. Ladder, MVT MMMU-MCQonly:
   > **0.634 @ 2 tok → 0.697 @ 431 → 0.738 @ 580**. Source:
   > `artifacts/medeval_matched_direct_2026-07-29.json`; retrospective §5.1, §10.1 C27.

So the method is **regime-adaptive**: never think on perception, reserve the slow think tier for the reasoning
residual — and it's scored against the slow always-think baseline, where a cheap method wins on both axes.

> **The honest caveat on this framing** (retrospective §7 hole 1, and it is the project's biggest open weakness): the
> dumps used as the *think* baseline for PMC-VQA (`test_2.csv`), SLAKE-closed and VQA-RAD-closed average **3–4 generated tokens** and
> agree with the no-think run on 92–94% of predictions — they were produced by a harness flag that only appends *"put
> the letter in \boxed{}"*, which is not a reasoning prompt. Genuine 32B reasoning was measured on only **10.3%** of the
> pool, yet the 10,521 ms price is charged to all of it. Charging reasoning cost only where a real reasoning run exists
> turns the −95% latency claim into roughly **−72%**. **The vs-no-think and vs-oracle-mode comparisons (§3) do not have
> this problem** — against those, compute-lean is a tie and accuracy-max wins by +0.0106 [+0.0085, +0.0126].

## 5. How the method works — two engines

- **MCQ arm = the efficiency engine.** 7B answers; a **confidence-margin gate** (escalate iff `margin < τ`) sends only
  the low-confidence questions to the 32B in **no-think** mode. **Escalation varies enormously per benchmark — PMC-VQA
  8.45%** (`test_2.csv`, n = 33,430), **SLAKE-closed 20.45%, PathVQA-closed 45.72%, VQA-RAD-closed 56.97%, MedXpert
  89.60%** (open cells: SLAKE 15.81%, VQA-RAD 12.50%, PathVQA 35.67%) — **so never quote a single suite escalation rate
  without saying which weighting it uses: 16.22% sample-weighted over the MCQ cells versus 44.24% at equal weight per
  cell** (all 8 cells: 16.89% vs 35.65%). That heterogeneity is the mechanism behind the 2026-07-30 cost reversal. On
  **PMC-VQA** the accuracy-max setting adds a lever: either *fuse* the two models on the ~33% of items where they
  disagree (v1) or apply a **certified veto** — keep the 7B answer inside confidence bins where a Wilson lower bound on
  its precision beats the 32B, and never run the 32B there (v2, cheaper). A **prefill-prefetch** trick (run the 32B's
  image prefill concurrently with the 7B pass) buys **461 → 405 ms, −12.1%** at zero accuracy change — it is *documented
  but deliberately not folded into the headline*, because unconditional prefetch pays the 32B prefill on every query.
- **Open-text arm = the accuracy engine.** The 7B samples several answers and a **trained LoRA verifier**
  (`ckpts/train/lora_verifier_pooled4`, per-answer AUROC 0.924) picks the best. **Pandora adaptive-N** — Weitzman's
  optimal-search rule — draws only as many samples as needed (mean N = 3.45 / 3.91 / 5.48 per set, −33% open-arm
  compute at iso-accuracy), and a team-objective learning-to-defer rule decides when to escalate. This arm is
  **5.55% of the questions but carries essentially the entire vs-think win** (Δ +0.2699 on the open pool,
  sample-weighted; **+0.1848 [+0.1583, +0.2110]** macro). Remove it and compute-lean's advantage disappears (MCQ-only
  Δ +0.0006, CI spans zero, sample-weighted; **−0.0046 [−0.0126, +0.0036]** macro).
  **⚠️ It is also the method's most EXPENSIVE arm — 7.6–12.6 FLOP-eq per cell versus the 32B baseline's flat 4.57 —
  and under equal weight it holds 37.5% of the reporting weight. That combination is what reverses the compute claim.**
  Its accuracy is **PROVISIONAL** pending the clean-verifier retrain (§3).

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
> 2. **The reasoning half got downgraded** — and on 2026-07-30 it was **re-attributed entirely**. 12 of 15
>    cells still point the right way, but only **4** have a confidence interval that excludes zero, and **one
>    is significantly negative**; the surviving cells rest on MedVLThinker-32B (3/3 significant) and
>    MedGemma-27B (3/3 positive, 1/3 significant). **⚠️ 2026-07-30: those "4 of 15" are
>    trigger-PLUS-format counts.** With the answer format matched, the reasoning *instruction* adds
>    ~nothing (**0 of 9** sub-cells CI-significant across Lingshu-32B / MedVLThinker-32B / InternVL3-38B),
>    while the **`\boxed{}` format** contrast is significant in **3 of 9**. **Asking for `\boxed{}` is itself
>    a reasoning trigger.** So the reasoning half is an **answer-format** finding, not a reasoning-instruction
>    finding — see §4 item 2 for the per-cell decomposition and the monotone ladder.
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
middle tier). (b) These are **internal-harness** numbers (evaluation context B, 6 benchmarks / 8,220 samples, whose
PMC-VQA cell is **`test_clean.csv`, n = 2,000 — 24.3% of the pool**, not the `test_2.csv` used in §3) and must
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
  and the open-ended cascade. Result: the **only** closed-MCQ slice we can certifiably beat the 32B on is **PMC-VQA**
  (on `test_2.csv`, n = 33,430).
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

- **A reproducible method** (`src/cascade_methods/method_final.py`) that beats always-Lingshu-32B-**with-thinking** on
  the MedEvalKit suite (both formats) on accuracy, latency and energy, and beats always-32B-**no-think** on accuracy at
  its accuracy-max setting. **⚠️ Corrected 2026-07-30: the claim that "both operating points use less compute than one
  32B forward" was a SAMPLE-WEIGHTED statement and is retired** — at equal weight per cell the operating points cost
  **1.196× / 1.410× / 1.435×** a single 32B forward, and compute-lean is a significant *loss* on the multiple-choice
  cells. See §3. *Caveat worth stating plainly:* `method_final.py` is a **CPU re-costing of saved per-sample dumps** —
  escalation is `np.where(margin < τ, ok_32B, ok_7B)` over recorded correctness, and latency/energy come from per-leg
  batch-1 constants. **The final method has never been executed end-to-end as a live pipeline.** (The one genuine live
  cascade in the repo, `ckpts/rt_cascade_cap320.jsonl`, belongs to the older MedVLThinker work.)
- **A faithful 3-family × 7-benchmark reproduction** underneath the baselines — anchored by Lingshu-32B on MMMU-Medical
  = 0.633 against the paper's 62.3, *exact*. 6 of 7 benchmarks fully faithful plus one cheap-faithful (Lingshu-7B
  OmniMed **0.8274** vs paper 0.829 on all 88,996 questions); the OmniMed strong leg is a documented infra fallback
  (a deterministic 2-GPU NCCL hang, `docs/current/OMNIMED_FALLBACK.md`) — **no fabricated metrics file was written**.
- **A map of the limits confirmed dozens of times over** (recoverability ×16, selection ×13, plus the coverage
  measurement) — a genuine negative-results contribution, not a caveat. The retrospective argues this
  characterization is a **stronger** contribution than the +0.0245 itself, because it survives the honest re-costing
  **and the 2026-07-30 macro re-basing** — which the efficiency headline did not.
- **A stated list of the method's own weaknesses** — retrospective §7 ranks 17 holes, three of them critical (the new
  **hole 17**: the thresholds were tuned against a *pooled* objective but the report is now *macro*, and a
  macro-objective refit has not been done). Anyone
  quoting this project's numbers should read it first.
- **A full, sourced audit trail:** the retrospective (`PROJECT_RETROSPECTIVE_2026-07-29.md`), the method spec
  (`METHOD_FINAL_2026-07.md`), the results ledger (`RESEARCH_RESULTS_2026-07.md`), the technical report
  (`TECHNICAL_REPORT_2026-07.md`), **13 daily diaries** (`progress/`, June 17 → July 8 — the most trustworthy layer in
  the tree), and the 68-idea cross-field backlog (`METHOD_IDEAS_BACKLOG.md`).
- **The paper:** `paper/adaptive-cascade-medvqa_ieee_2026-07-08.{tex,pdf}` — 9 pages, IEEEtran, and the *only* prose
  artifact besides the July-27 deck that carries the corrected numbers. Earlier drafts are in `paper/archive/`.
  **Rebuilt 2026-07-30 (12 pages):** `python3 paper/make_ieee_figs.py && bash paper/build_ieee.sh
  paper/adaptive-cascade-medvqa_ieee_2026-07-08.tex`. The `.tex`, the `.pdf` and all three figures now carry the
  corrected Finding 1 *and* the 8-cell macro re-basing; `fig_pareto.pdf` reads
  `macro_average_headline_2026-07-30.json:cost.pareto.honest_recost` (superseded copy:
  `fig_pareto_superseded_2026-07-08.pdf`) and `fig_overthink.pdf` hatches the format-confounded reasoning block. The **July-27 deck still shows the
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
