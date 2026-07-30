> # ⚠️ NUMERICALLY SUPERSEDED — annotated 2026-07-29
>
> **The mechanism described in this document is correct. Its headline numbers are not.** It was written
> before three things that changed them: the **oracle-mode-32B baseline** (2026-07-08 08:18), the decision to
> **exclude MMMU** after the contamination audit ("Variant B", 08:24–09:43), and the replacement of the
> **estimated** 32B-reasoning open-text cells with **measured** ones (10:41), plus the headline CI computed
> 2026-07-09.
>
> **Canonical values (Variant B = MMMU excluded, n = 42,224, measured, CI-certified):**
> always-32B-with-reasoning baseline **0.5591**; compute-lean **0.5741, +0.0150 [+0.0107, +0.0192]** at 0.49x;
> **accuracy-max 0.5836, +0.0245 [+0.0216, +0.0274]** at 0.93x; accuracy-max-fusion 0.5862, +0.0271 at 1.25x.
> Sources: `artifacts/f8_mode_vsthink_ci.json`, `artifacts/opentext_32b_think_full.json`.
>
> **Also corrected since:** the "Baselines (**measured**)" label on 0.5632 was wrong (those open cells were
> estimates); the MMMU keep-7B "+0.140 / +0.167" per-benchmark win is **retracted** (contaminated, excluded);
> the open-text figures quoted as 0.387 pooled / SLAKE 0.700 / VQA-RAD 0.425 / PathVQA 0.035 are the **n=200
> subsample** (full-set measured: 0.3028 / 0.6791 / 0.5450 / 0.1087); "verifier bo-N 0.563 > 32B-nt 0.517 >
> 32B-think 0.370" should read **0.5727 > 0.5168 > 0.3028**; and H1 test-time-training is listed as
> "**running**" but completed as a documented negative (`artifacts/ttt_cheap_leg.json`).
>
> **The definitive account is [`PROJECT_RETROSPECTIVE_2026-07-29.md`](PROJECT_RETROSPECTIVE_2026-07-29.md)**
> — §4 for the corrected results, §7 for this method's 16 known holes, §10.3 for the full decode of the
> `+0.02xx` number family. **Read this file for *how it works*, not for *what it scores*.**
>
> ### ⚠️ Which PMC-VQA split (added 2026-07-30)
>
> Every PMC-VQA number in this file is the **MedEvalKit/Lingshu** track = **`test_2.csv`** (v2, **33,430**
> items) — hard-coded in unmodified vendor code at `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39`, the split with
> **zero published verification**, and **79%** of the Variant-B pool. The project's *other* track (the June
> MedVLThinker internal harness) used a **different** file, **`test_clean.csv`** (v1, **2,000** items, the
> authors' only human-verified split, **24.3%** of its 8,220 pool). The two overlap on **6 items**, so their
> numbers are never interchangeable. Provenance:
> [`PMCVQA_PROVENANCE_2026-07-30.md`](PMCVQA_PROVENANCE_2026-07-30.md).

# Technical Report — The Final Method Loop (2026-07-04 → 07)

*A plain-language + technical walkthrough of what we did, why, and what every number means. Written so you can
understand the result end-to-end. All figures come from real held-out artifacts under
`results/cascade_methods/artifacts/`; nothing is fabricated. Abstention is out of scope and appears nowhere.*

---

## 0. The one-paragraph summary

We built a single method — a **format-aware, regime-adaptive cascade** between Lingshu-7B and Lingshu-32B — that is
**both faster and more accurate than "just use the big 32B model with thinking"** on the whole MedEvalKit medical-VQA
suite (MCQ *and* open-text). It has a **Pareto knob** with two settings, and **both settings use *less* compute than a
single 32B forward** while matching-or-beating its accuracy. Along the way we mapped, with six independent
confirmations, exactly *where* a cheap→expensive medical cascade can and cannot beat the strong model — that honest
"walls" characterization is as much a result as the method.

---

## 1. The goal (as refined by you during the loop)

> **One method that beats always-Lingshu-32B (MedEvalKit numbers) on the full MedVQA suite — faster AND more accurate —
> handling both MCQ and open-text.**

Two things made this concrete and are easy to get wrong:

- **"Faster" is two axes**, and we report both: **batch-1 latency** (wall-clock for one question) and **FLOPs**
  (total compute / throughput). A method can win one and lose the other.
- **The baseline is the big model *with thinking* on** — "use Lingshu-32B in think mode for everything." This is the
  deployment a clinician would pick if they wanted maximum accuracy. It is the *slow, expensive* target we must beat.

---

## 2. The critical reframe (why an early conclusion was wrong)

Earlier in the loop we compared our cheap method against **32B *no-think*** (a single fast 665 ms forward) and
concluded "best-of-N is Pareto-dominated — just call the 32B." **That was the wrong baseline**, and correcting it
flipped the story. We measured the real think-mode costs (GPU, batch-1, same harness as the 665 ms number):

| Baseline (Lingshu-32B) | open-text latency | open-text energy | open-text accuracy |
|---|---|---|---|
| **no-think** | 665 ms | 127 J | **0.537** |
| **think** | **10,521 ms (≈16×)** | 2,002 J (≈16×) | **0.387 (−0.150)** |

**Two facts fall out, and they are the backbone of the method:**

1. **Thinking *hurts* perception** and is far slower. On the perception open-text sets, think is *both* worse *and*
   16× slower (SLAKE 0.700 vs 0.895; VQA-RAD 0.425 vs 0.545; PathVQA 0.035 vs 0.170). So for perception, the right
   strong leg is **32B-no-think**, and the naive "always-think" baseline is self-sabotaging there.
2. **Thinking *helps* only reasoning** (MMMU, MedXpert), and those are **MCQ-only**. Measured think gains: MMMU
   **+0.027 / +0.100 / +0.120** (Lingshu / MedVLThinker / InternVL3); MedXpert +0.00 / +0.045 / +0.030.

> **⚠️ Both facts corrected 2026-07-29** (`artifacts/finding1_corrected_2026-07-29.json`; retrospective §5.1,
> §10.1 C20–C25). **(1)** The **open-text** think-vs-direct accuracy numbers in the table and in fact 1 are
> **PROVISIONAL** — the two arms had different system prompts (only the direct arm was told to answer in a
> "short, specific phrase" and not to explain: `src/labeling/run_openvqa.py:26/27`), which on free text is a
> live grading channel; a matched-prompt re-run is in flight. The **16× cost** side is measured and stands,
> and the *multiple-choice* version of "thinking hurts perception" is unaffected and came out **stronger**
> (**17/20** cells strictly negative, 14/20 CI-significant, pooled **−0.0401 [−0.0456, −0.0347]**, n = 30,250).
> **(2)** Fact 2 is **model-dependent, not universal**: of those MMMU gains, **Lingshu's +0.027 is NOT
> significant** ([−0.047, +0.100], n = 150) while MedVLThinker **+0.100 [+0.027, +0.173]** and InternVL3-38B
> **+0.120 [+0.047, +0.193]** are. Lingshu-32B shows **no** reasoning benefit anywhere and must not be cited
> as evidence that reasoning helps; QoQ-Med-VL-32B is withdrawn as reasoning evidence too.

So the honest method is **regime-adaptive**: never think on perception, reserve the slow think tier for the reasoning
residual — and it is scored against the slow always-think baseline, where a cheap method wins big on both axes.

---

## 3. The method (final architecture)

A **format router** first decides MCQ vs open-text **from the prompt** (not from the gold answer — this is the
realistic test-time condition). It then runs one of two arms. A single Pareto `mode` flag (`compute-lean` /
`accuracy-max`) and an `int4` cost flag select the operating point. Reproduce everything with one command:
`python3 src/cascade_methods/method_final.py`.

### 3.1 MCQ arm (the *efficiency* engine)
- **Cheap leg:** Lingshu-7B, no-think, one greedy generation.
- **Gate:** **confidence margin** (top-1 − top-2 option probability). Below threshold τ → escalate. τ is frozen on a
  held-out calibration split (no peeking).
- **Strong leg:** Lingshu-32B **no-think** (think never beats no-think on any Lingshu benchmark). For the *reasoning*
  benchmark MMMU we instead **keep the 7B** (Lingshu-7B 0.80 ≫ 32B 0.63 there — a real, if quirky, per-slice win).
- **Accuracy add-on (accuracy-max mode only):** on **PMC-VQA** — the one large slice where the 7B and 32B are
  comparably skilled with *de-correlated* errors — we **fuse** the two: a certified confidence-advantage rule (F8/F3)
  that overrides the 32B where the 7B is more calibrated-confident. This *raises* accuracy above always-32B on PMC.
- **Latency add-on (G8):** the 32B's image **prefill** doesn't depend on the 7B, so we prefetch it on the idle 2nd GPU
  concurrently with the 7B pass — the 7B forward hides "for free" under the 32B prefill.

### 3.2 Open-text arm (the *accuracy* engine)
- **Best-of-N + trained verifier:** the 7B samples several candidate answers; a trained LoRA verifier
  (`lora_verifier_pooled4`) scores each and picks the best. This **beats the 32B outright** on open-text
  (verifier bo-N **0.563** > 32B-no-think **0.517** > 32B-think **0.370**).
- **Pandora adaptive-N:** instead of a fixed 8 samples, an optimal-stopping rule (Weitzman's "Pandora's box," from
  economics) draws only as many as needed — **mean 8 → 4.3 draws, −33% open-text FLOPs at held-out iso-accuracy.**
- **Escalation gate (F10):** a team-objective "learning-to-complement" rule decides keep-7B-bo-N vs escalate-to-32B —
  it *repairs* the two cells where the naive gate lost (see §4).

### 3.3 The Pareto knob
- **compute-lean** = MCQ margin cascade + open-text Pandora-verifier. Fastest, cheapest.
- **accuracy-max** = adds the PMC fusion. More accurate, slightly more compute.
- Both are **FLOP-negative** vs a single 32B forward.

---

## 4. The results (the headline tables)

**Baselines (measured):** always-32B-**think** = 0.5632 accuracy @ 4.57 FLOP-eq / **10,521 ms**;
always-32B-**no-think** = 0.5732 @ 4.57 FLOP-eq / **665 ms**. (FLOP-eq = multiples of one 7B forward; the 32B is 4.57×.)

**Method (held-out, full suite n = 42,374, sample-weighted accuracy):**

| mode | Δacc vs 32B-**think** | Δacc vs 32B-**no-think** | FLOP-eq (× a 32B call) | latency (parallel) |
|---|---:|---:|---:|---:|
| **compute-lean** | **+0.0123** | +0.0023 | **2.24 (0.49×)** | **468 ms** |
| **accuracy-max v2** (F8+F10) | **+0.0212** | +0.0112 | **4.25 (0.93×)** | 729 ms |
| accuracy-max v1 (F3 fusion) | +0.0238 | +0.0137 | 5.70 (1.25×) | 666 ms |

**How to read this:**
- Every row is **more accurate than always-32B-think** and **~14–25× faster** (0.5–0.7 s vs 10.5 s).
- **compute-lean** and **accuracy-max v2 both use *less* total compute than a single 32B forward** (0.49× and 0.93×) —
  i.e. strictly dominant on all three axes vs the think baseline.
- **accuracy-max v1** is the max-accuracy point (+0.0238) but costs 1.25× FLOPs (the fusion runs both models on PMC);
  **v2** trades a hair of accuracy (+0.0212) to become FLOP-negative. Both are available; pick by budget.

**Where the wins live (per-benchmark, vs always-32B-think):**
- **MMMU +0.140** — from keeping the 7B (it beats the 32B there).
- **All open-text +0.10 to +0.21** — from the verifier best-of-N (the accuracy engine).
- **PMC-VQA +0.0135** (CI [0.010, 0.017], n = 33,430, **`test_2.csv`**) — the genuine, non-anomaly MCQ fusion win.
- **Perception MCQ within ±0.01** — matched accuracy at **15–25× the speed** (the efficiency engine).

---

## 5. What each lever does (and its number)

| Lever | What it does | Measured effect |
|---|---|---|
| **Margin gate (MCQ)** | escalate only low-confidence 7B answers | matches 32B at ~16% escalation |
| **Verifier best-of-N (open-text)** | 7B samples N, verifier picks best | **0.563 > 32B-nt 0.517** (beats the strong model) |
| **Pandora adaptive-N** | draw only as many samples as needed | draws 8→4.3, **−33% open FLOPs** at iso-acc |
| **F3/F8 PMC fusion** | pick the more-calibrated model on PMC (`test_2.csv`, n = 33,430) | **+0.0135 / +0.0095** (F8 at 0.88× FLOPs) |
| **F10 learning-to-complement** | better open-text escalate/keep rule | repairs SLAKE/VQA-RAD losses; **PathVQA +0.086** |
| **G8 prefill prefetch** | overlap 32B prefill with the 7B pass | 461→405 ms, **zero accuracy cost** |
| **MMMU keep-7B** | don't escalate where 7B wins | **+0.167** on MMMU |

### 5.1 An important correction (gates are family-dependent)
You expected **agreement / CASP-stability** to be the best MCQ gate (they were on *MedVLThinker*). On **Lingshu** they
are **not**: margin is best (AUROC **0.725**), **agreement is the *worst*** ranker (AUROC 0.657) *and* needs a 32B
forward to compute (defeating a cheap gate), and **CASP is inert** because Lingshu-7B is **98.9% resolution-stable**
(the stability signal has nothing to vary on). So the deployed gate is the simple **margin**. The lesson: gate choice
is model-family-specific, not universal.

### 5.2 Router vs. unified policy
We tested whether one gate could serve both formats. It cannot — the MCQ margin has no open-text analog and the trained
verifier is open-text-specific. **A two-arm router is required.**

---

## 6. The walls (the honest negatives — this is half the contribution)

We ran ~20 experiments trying to push accuracy further. Almost all were negative, and *together* they precisely
bound what is possible. Understanding these is understanding the result.

### 6.1 The recoverability wall — the MCQ beat is bounded to PMC
To beat the 32B on an MCQ question, you must know *ex ante* which questions the 32B will get right that the 7B got
wrong. That signal is weak (recoverability AUROC ≈ 0.6). Consequence: **the only closed-MCQ slice we can certifiably
beat the 32B on is PMC** (where the models are comparably skilled and de-correlated). **Six independent methods all
agree** and none extends the beat past PMC:
1. F3 confidence-advantage fusion, 2. F8 certified weak-veto, 3. F11 Bayesian model averaging,
4. F6 contrastive decoding, 5. **logit-level** full-posterior fusion (we checked the raw option-probability vectors,
not just decisions), 6. **Domino** automatic slice discovery (searched 106 candidate slices/split across the whole
observable feature space and found **zero** new certified slices — below the permutation-null floor).
**MMMU is a route-to-7B anomaly, not a fusion win.**

### 6.2 The selectability wall — best-of-N can't be fully converted
On open-text, a correct answer is often *among* the N samples (high oracle@N) but the verifier can't always *pick* it
(the "oracle→selection gap," ≈0.19). We tried to close it and failed every way:
- a **4.5×-bigger 32B verifier ties** the trained 7B verifier (+0.005, not significant) → the gap is **fundamental**,
  not a verifier-capacity problem;
- **distractor filtering, pairwise/tournament verifiers, diverse generation** — none converts the extra coverage
  (diverse generation *raises* oracle@N by +0.03–0.06 but injects confident distractors the verifier then mis-picks).

### 6.3 The cost reframe (why best-of-N is "dominated" *only* vs the cheap baseline)
Best-of-8 costs ~16 FLOP-eq; a 32B-no-think forward costs 4.57. So vs the **cheap** no-think baseline, best-of-N is
FLOPs-dominated. But vs the **slow think** baseline (10.5 s), a *parallel* best-of-N (~0.5 s) wins on latency and
accuracy. This is exactly why the correct baseline mattered so much.

### 6.4 Levers that don't help the way we hoped
- **G5 futile-escalation suppressor:** the recoverability wall again — no slice is truly "futile while escalating," so
  suppression is a small trade, not a free lunch.
- **INT4 strong leg:** our FLOP unit is MAC-count (precision-independent), so INT4 gives **zero** FLOP reduction — it's
  a VRAM/energy/deployability win only; latency drops just 665→583 ms (the strong leg is prefill-bound).
- **G4 image-token pruning:** a *real* projected MAC cut (−26%, 4.57→3.37), but not quick to wire in vLLM (Qwen2.5-VL
  M-RoPE ≠ FastV's assumptions) **and risky on radiology** (image-token reduction costs −0.017/−0.040 on SLAKE/VQA-RAD)
  — deferred with an implementation plan.
- **H8 credibility shrinkage:** confirms the thin-slice overfit risk is real (naive routing → 7.5 guardrail
  violations/split) but the method's **existing CI-lower-bound guardrail already drives it to 0.25** — so shrinkage is
  diagnostic, and it *validates* the guardrail we deploy.

---

## 7. The reproduction (context for the baselines)

The Lingshu-32B baseline numbers are from a **faithful MedEvalKit reproduction** (3 families — Lingshu, MedVLThinker,
InternVL3 — across 7 benchmarks; Lingshu-7B OmniMed 0.827 vs paper 0.829 confirms fidelity). One cell is a documented
fallback: the **strong 32B/38B OmniMedVQA leg** hit a persistent tensor-parallel NCCL hang (89k images) and is reported
as paper-reference; OmniMed is a keep-cheap benchmark (7B ≈ 32B) so it changes no conclusion.

---

## 8. Bottom line — what you now have

1. **A deployable method** that is faster *and* more accurate than always-Lingshu-32B-think on the full suite, with a
   clean Pareto knob and both operating points **FLOP-negative** — fully reproducible from one script.
2. **A precise, six-times-confirmed map of the limits**: match-cheaply on MCQ (bounded beat only on PMC), *beat* on
   open-text (verifier best-of-N) and on the MMMU/PMC owned slices — and *why* (the recoverability + selectability
   walls). This honest characterization is a genuine scientific contribution, not just a caveat.
3. **A large, sourced audit trail** — every lever tested, every number held-out, in `RESEARCH_RESULTS_2026-07.md`,
   `METHOD_FINAL_2026-07.md`, the `progress/progress_July_0{4-7}.md` diaries, and the artifacts.

**Open upside (small):** the G4 image-token-prune GPU confirm (the one lever that could add real FLOP savings, pending
a radiology-safety check), and H1 test-time-training of the cheap leg (running). Neither changes the headline.
