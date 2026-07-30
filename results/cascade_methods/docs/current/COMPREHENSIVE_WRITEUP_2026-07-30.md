# Adaptive Test-Time Compute for Medical Vision–Language Models

## A comprehensive write-up of the project, its corrections, and what survives

**Repository:** `medvlthinker-imgdiff-compute`
**Period covered:** 2026-06-17 → 2026-07-30
**Document written:** 2026-07-30
**Status:** this document supersedes every earlier summary in the repository. Where an earlier
document disagrees with this one, this one is correct and the earlier one is stale. A register of
known-stale documents is given in §11.3.

---

## 0. Executive summary

**What was done.** We built and evaluated a *cascade* for medical visual question answering (VQA): a
cheap 7-billion-parameter vision–language model answers every question, and an expensive
32-billion-parameter model is called only on the questions where the cheap model looks unreliable.
The goal was large-model accuracy at small-model cost. Over six weeks the project pivoted five
times — from visual-token pruning, to image-difficulty routing, to routing within a single model, to
a cross-model cascade, and finally to a **format-aware cascade** (Lingshu-7B → Lingshu-32B) that
detects from the prompt alone whether a question is multiple-choice or free-text and runs a
different policy for each. Evaluation is on the MedEvalKit suite plus a custom judged free-text
pipeline: 5 benchmarks, 8 reporting cells, 42,224 items.

On 2026-07-29/30 a systematic correction pass re-examined every load-bearing claim. Two corrections
landed together and cost the project its headline:

1. **Weighting.** The primary average was sample-weighted, and one benchmark (PMC-VQA, 33,430 items)
   held **79.2%** of the pool — and was also the cell where the gate escalated least (8.45%, against
   up to 89.60% elsewhere). The reported suite accuracy and suite cost were, approximately, that one
   benchmark's accuracy and cost. Re-basing on **equal weight per reporting cell** turned a claimed
   0.492× compute *saving* into a 1.196× compute *cost*.
2. **Verifier contamination.** The trained free-text answer verifier — the component that makes the
   open-text arm work — had been trained on **67–73%** of the items it was then scored on. A retrain
   on strictly disjoint data cut its selection gain by **2.90×**.

**The single most important finding** — and the one untouched by both corrections — is that
**chain-of-thought reasoning is a net accuracy loss on perception-style medical VQA**, and that the
apparent gains on reasoning-heavy benchmarks are an **answer-format** effect rather than a reasoning
effect:

- Reasoning is strictly worse than answering directly in **17 of 20** perception cells across 5
  medical VLM families (14/20 with 95% confidence intervals excluding zero), pooled
  **−0.0401 [−0.0456, −0.0347]** over 30,250 paired samples, replicated on two non-medical
  architectures.
- On free text, with a matched-prompt 2×2 design, the reasoning instruction costs
  **−0.2158 [−0.2354, −0.1962]** while the prompt-style confound it was suspected of being is worth
  **−0.0017 [−0.0111, +0.0077], not significant**.
- With the answer format held constant, the explicit reasoning *trigger* is worth nothing:
  **0 of 9** matched sub-cells reach significance, while **3 of 9** answer-*format* effects do.
  Asking for the answer in `\boxed{}` is itself a reasoning trigger — with no trigger present,
  MedVLThinker-32B emits 431–580 tokens on 99–100% of items.

**The current honest headline.** Averaged equally over the 8 reporting cells and with an
uncontaminated verifier:

> **The method ties a single 32B direct forward pass — accuracy-max is +0.0008 [−0.0022, +0.0037] —
> while costing 1.74× its FLOP-equivalents, +16.7% batch-1 latency and +101.0% energy. The cheaper
> setting is a significant loss (−0.0124 [−0.0188, −0.0060]) at 1.46× the compute.**

What survives is the comparison against the way a practitioner would *naively* deploy a
reasoning-capable medical VLM. Against **always-32B-with-reasoning**, accuracy-max is
**+0.0601 [+0.0498, +0.0703]** at **−87.7% latency** and **−84.3% energy** (though at 1.40× the
honestly re-costed FLOP-equivalents — the win is on wall-clock and joules, *not* on compute).

**The practical recommendation that outranks the entire method:** turn reasoning off.
Always-32B-direct scores **0.6567** macro against always-32B-with-reasoning's **0.5974** — a
+0.0593 gap — at 665 ms versus 6,291 ms and 127 J versus 1,625 J. That is a prompt change requiring
no second model, no gate and no verifier, and it is larger than every method delta in this project
combined.

**What a reader should take away.** The cascade is not the contribution. The contributions are
(a) the reasoning-versus-perception result and its re-attribution to answer format, (b) two
quantified limits that explain every negative result in the project — the difficulty of predicting
*whether the strong model would fix this particular error*, and the difficulty of *selecting* the
correct answer from a pool that contains it — with a measured budget showing that the missing-answer
problem is **4.5× larger** than the selection problem, and (c) a set of methodological failure modes,
each caught here only *after* it had already produced a published number.

---

## 1. Table of contents

| § | Section |
|---|---|
| 0 | Executive summary |
| 1 | Table of contents |
| 2 | How to verify any number in this document |
| 3 | The question, the arc, and what was under test |
| 3.1 | The original question and the measured cost gap |
| 3.2 | What "the suite" actually is |
| 3.3 | How the question changed — five pivots, each forced by a number |
| 3.4 | The five claims under test as of 2026-07-29 |
| 3.5 | Why each claim was worth doubting |
| 4 | Methodology — how each thing was tested |
| 4.1 | The shape every investigation was forced into |
| 4.2 | The honest re-costing |
| 4.3 | The matched-prompt experiments (free text; multiple choice) |
| 4.4 | The cross-family prompt-matching audit and re-derivation |
| 4.5 | The PMC-VQA item-level audit |
| 4.6 | The verifier validity work |
| 4.7 | The disjoint verifier retrain |
| 4.8 | The weighting change |
| 4.9 | Statistical practice |
| 4.10 | Verification gates, as a checklist |
| 5 | Results |
| 5.1 | Reasoning versus perception |
| 5.2 | Reasoning on reasoning-heavy benchmarks |
| 5.3 | The open-text matched-prompt result |
| 5.4 | PMC-VQA validity |
| 5.5 | Verifier contamination and the clean retrain |
| 5.6 | The headline under each accounting |
| 5.7 | Ancillary findings (MMMU contamination; the SLAKE path bug; the two PMC-VQA splits) |
| 5.8 | Summary of claim status |
| 6 | Corrections and retractions |
| 7 | Traps in this research area |
| 8 | What survives adversarial checking |
| 9 | What did not survive |
| 10 | The honest deployment recommendation, and what the paper can claim |
| 11 | Unverified, unrecorded, and stale — the full register |
| 12 | Open questions, ranked |

---

## 2. How to verify any number in this document

Every quantitative claim below is traceable to a file on disk. The conventions are:

**Artifact directory.** Unless another path is given, a cited `*.json` lives in
`results/cascade_methods/artifacts/`. Prose documents live in
`results/cascade_methods/docs/current/`. Code paths are given from the repository root.

**Citation form.** `file.json : key.path.to.value` — the JSON pointer is literal and can be read with

```bash
cd ~/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts
python3 -c "import json; d=json.load(open('FILE.json')); print(json.dumps(d['KEY'], indent=1))"
```

**Provenance tags.** Every figure carries exactly one:

| tag | meaning |
|---|---|
| **[M]** MEASURED | recomputed from a per-sample dump on disk — an `ok` / `judge_ok` / `correct` field, a token count, a hash, a file listing |
| **[D]** DERIVED | exact arithmetic on measured quantities (a pooled average, a corrected delta, a cost ratio) |
| **[Mo]** MODELLED | a cost produced by plugging measured per-leg batch-1 constants into an expected-cost formula |
| **[P]** PROJECTED | an extrapolation beyond what was run |
| **[U]** UNVERIFIED | documented in the repository but not traceable to an artifact this pass could read |

**A standing caution about [Mo].** *No number in any cost table in this document comes from executing
the assembled cascade end to end.* The method has only ever been re-costed offline, on CPU, from
saved per-sample dumps, using per-leg batch-1 latency/energy constants. The project's own
retrospective calls this "calibrated wall-clock", not a measured deployment. Where that distinction
changes the reading, it is flagged in place.

**Intervals.** Unless stated otherwise, every interval is a 95% **paired** bootstrap percentile
interval over 10,000 resamples, computed on the same items across the systems compared. "Significant"
means the interval excludes zero.

**Terminology.** Terms are defined at first use. Internal codenames used in earlier drafts have been
removed; where a mechanism had one, it is described by what it does.
---

# 3. The question, the arc, and what was under test

## 3.1 The original question, and the cost gap that motivates it

A **vision–language model (VLM)** takes an image plus a text question and produces a text answer. In
this project the images are medical (radiology, pathology, endoscopy, textbook figures) and the task
is **visual question answering (VQA)**. Larger VLMs answer more medical questions correctly; they
also cost far more per answer. The founding question was the standard efficiency question:

> **Can we get large-model accuracy at small-model cost by spending compute selectively — a lot on
> the questions that need it, almost none on the questions that do not?**

The vehicle was a **cascade**: run a cheap 7-billion-parameter model (the *cheap leg*) on every
question, and call an expensive 32-billion-parameter model (the *strong leg*) only where the cheap
model looks unreliable. The rule that decides "escalate or not" is the **gate**; the fraction of
questions it hands upward is the **escalation rate**; the confidence signal it thresholds is the
**margin** (the gap between the top-1 and top-2 answer probabilities), and the threshold is
**τ (tau)**.

**The cost gap is measured, not assumed.** All figures below are batch-1 measurements on one A100
80 GB with NVML power logging.

| system | latency | energy | FLOP-eq\* | provenance |
|---|---:|---:|---:|---|
| 7B, direct (no reasoning trace) | 347 ms | 45.8 J | 1.00 | `src/cascade_methods/paper_baselines.py:62` (`GEN7`) **[M, provenance is a code comment — see below]** |
| 32B, direct | 665 ms | 127.0 J | 4.57 | `paper_baselines.py:64` (`GEN32N`); n = 25 at 5.6 generated tokens (`honest_recosting_2026-07-29.json : provenance.anchor_direct`) **[M]** |
| 32B, **with reasoning**, open-text prompt | **10,521.6 ms** (median 12,896.2) | **2,001.9 J** | 4.57 (a deliberate lower bound) | `paper_baselines.py:65`; n = 15 at 98.3 generated tokens (`honest_recosting_2026-07-29.json : provenance.anchor_think`) **[M, weak — see §4.9]** |
| 32B, direct, cap320 images | 333.15 ms | 61.2 J | — | independent NVML sweep, n = 60, median, 2 generated tokens (`honest_recosting_2026-07-29.json : provenance.latency_32b_jsonl_medians.nothink@cap320`) **[M]** |
| 32B, **with reasoning**, full resolution | **22,983.25 ms** | **5,990.2 J** | — | same sweep, n = 60, median, 318 generated tokens **[M]** |
| one verifier forward (7B LoRA scorer) | 175 ms | 25.3 J | 1.00 | `src/cascade_methods/integrated_method.py:53-54` / `paper_baselines.py:63` (`VER7`) **[M, provenance is a code comment]** |

\*FLOP-eq = multiply–accumulate operations normalised to one 7B forward pass.

**Three provenance caveats that travel with this table.**

1. The **4.57** ratio is a hard-coded literal. `honest_recosting_2026-07-29.json :
   provenance.flop_ratio_derivation` reproduces it only as 32.0 B / 7.0 B = 4.571, and an older
   project document implies 4.34. **No file in the repository derives it.** Every compute ratio in
   this document inherits that undetermined denominator, a margin of roughly ±7% on the
   compute-negative claims. **[U]**
2. The 7B constants (347 ms / 45.8 J) and the verifier forward (175 ms / 25.3 J) are labelled
   "measured batch-1" in the code, but the raw NVML log backing them was not located in this pass —
   the 32B direct constant cites `logs/latency_opentext.jsonl`, which is gitignored. Their
   provenance is a code comment, not a file read. **[U on provenance; the values themselves are used
   consistently throughout]**
3. The reasoning constant is a **mean over n = 15** whose **median is 12,896.2 ms** — a 23%
   mean/median divergence indicating a heavy tail — measured on VQA-RAD cap320 open-text prompts at
   98.3 generated tokens and then transferred unchanged to every multiple-choice cell. The headline
   uses the *smaller* of mean and median.

**"Reasoning" here means the model emits a `<think>…</think>` trace before answering.** Turning it on
multiplies the 32B's cost by **15.8× in latency and 15.8× in energy** on the open-text workload, and
by ≈69× latency / ≈98× energy on the independent full-resolution sweep **[D: 22,983.25 / 333.15;
5,990.2 / 61.2]**.

**Why this matters clinically.** At 665 ms a single GPU answers roughly 90 questions per minute; at
10.5–23.0 s it answers 3–6. Energy per answered question moves from ~0.035 Wh to 0.56–1.66 Wh — the
reasoning 32B costs ≈44× the small model's 45.8 J per answer **[D: 2,001.9 / 45.8]**. A hospital
pre-reading a day's studies cares about questions per minute and about the electricity and cooling
bill; a clinician waiting on one hard case cares only about being right. A method that spends the
ten-second answer only where it changes the answer is therefore worth more than a fractional accuracy
point.

> **This paragraph is an argument, not a result.** The questions-per-minute and Wh-per-answer figures
> are **[D]** conversions of the measured latency and energy constants above. **No clinical
> workflow, utility, or patient-outcome measurement exists anywhere in this repository.** The
> clinical framing motivates the work; it is not evidence for it.

## 3.2 What "the suite" actually is — and never cross-multiply the tracks

There are **three separate evaluation contexts** in the repository, and the headline suite is a
**splice of two of them**.

| track | harness | scoring | what it is used for |
|---|---|---|---|
| **A — faithful MedEvalKit** | `MedEvalKit/eval.py`, vLLM 0.9.0.1, seed 42 | exact match on the option letter (MMMU: judge-parsed) | the paper's **multiple-choice / closed** cells; reproduces the published Lingshu numbers |
| **B — internal harness** | `src/labeling/*` → `ckpts/acc_gen/` | exact match | the 5-family, 6-benchmark bake-off (8,220 items per family) that carries the cross-family reasoning evidence |
| **C — custom open-text pipeline** | `src/labeling/run_openvqa.py` + `run_judge.py` | **LLM judge** (a Qwen2.5-32B-based grader) | the **free-text** cells |

Tracks A and C are the two evaluation tracks of the headline: **40,029** multiple-choice/closed items
from A plus **2,345** free-text items from C = **n = 42,374**. The paper's "Variant B" pool excludes
the contaminated MMMU cell, giving **5 benchmarks, 8 reporting cells, n = 42,224** (39,879 closed +
2,345 open) **[M]**. It is not one harness run. Track C exists because MedEvalKit's open-half exact
match is broken — a gold of `"CT"` against a response of `"CT."` scores incorrect while ROUGE-1 ≈ 1.0
— so the open halves must be judged.

**The 9 reporting cells [M]:**

| cell | n | format | scored by | type |
|---|---:|---|---|---|
| PMC-VQA (`test_2.csv`, v2) | 33,430 | 4-option multiple choice | exact match | perception |
| SLAKE-closed | 836 | closed-ended | exact match | perception |
| VQA-RAD-closed | 251 | closed-ended | exact match | perception |
| PathVQA-closed | 3,362 | closed-ended (largely yes/no) | exact match | perception |
| MedXpertQA-MM | 2,000 | multiple choice | exact match | **reasoning** |
| MMMU-Medical | 150 | multiple choice | judge-parsed | **reasoning** — *excluded (contamination, §5.7.1)* |
| SLAKE-open | 645 | free text | LLM judge | perception |
| VQA-RAD-open | 200 | free text | LLM judge | perception |
| PathVQA-open | 1,500 | free text | LLM judge | perception |

> **Unverified detail.** The description of SLAKE-closed and VQA-RAD-closed as "closed-ended (fixed
> answer set)" comes from `METHODS_MASTER.md` §14 ("open + closed"), which does **not** enumerate the
> closed answer space per benchmark; PathVQA-closed is described there as "many yes/no". The answer
> spaces were **not** verified against the raw data files in this pass. **[U]**

**Perception versus reasoning** is a fixed partition used throughout: perception = PMC-VQA, SLAKE,
VQA-RAD, PathVQA (read the image, name what is there); reasoning = MMMU-Medical and MedXpertQA-MM's
Reasoning and Understanding splits (multi-step exam-style inference) — `artifacts/GENERALIZATION.md:75`.
Domains: SLAKE is bilingual CT/MRI/X-ray, VQA-RAD is clinician-written radiology, PathVQA is
histology microscopy, PMC-VQA is PubMed Central article figures, MMMU is college medical exams,
MedXpert is expert-level exam material.

**Two size caveats that travel with every open-text number [M]:** SLAKE-open uses **645 English-only
items** of the harness's 1,258, and PathVQA-open is a **non-random prefix slice**, `items = items[:A.n]`
at `src/labeling/run_openvqa.py:132`, taking 1,500 of 3,357 items. That prefix over-samples a
degenerate taxonomy family (accuracy 0.632 within it against 0.562 outside), and **the check that it
is not topically biased has never been done.** PathVQA-open is the load-bearing cell of every
vs-reasoning claim in this document.

> **⚠️ Landmine — two different PMC-VQA splits.** Track A uses **`test_2.csv`** (v2, 33,430 items,
> **not** author-verified), hard-coded in unmodified vendor code at
> `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39`. Track B uses **`test_clean.csv`** (v1, 2,000 items, the
> authors' only manually verified split). **`test_clean` ∩ `test_2` = 6 items** — effectively
> disjoint populations. Every PMC-VQA number must be quoted with its filename and row count. Full
> account in §5.7.3.

Two averaging conventions are also in play, and the difference turned out to matter enormously:
**sample-weighted** (every item counts once, so PMC-VQA holds 79.2% of the weight) versus **macro**
(every reporting cell counts 1/8). As of 2026-07-30 the primary convention is macro
(`macro_average_headline_2026-07-30.json`); the claims in §3.4 were all formulated under
sample-weighting.

## 3.3 How the question changed — five pivots, each forced by a number

**Pre-history (before the daily logs).** Three directions were killed before the diary begins:

1. **Question-aware visual-token pruning** — drop unimportant image tokens. Killed: "no usable
   accuracy cliff."
2. **Image-difficulty-driven compute** — spend more on "hard-looking" images. Killed: correlations of
   the wrong sign or near zero.
3. **Single-model routing** — route within *one* model across its own configurations (reasoning
   on/off × retrieval on/off). Killed by a permutation control: with 2,000 shuffles preserving
   marginal accuracy but destroying per-question complementarity, the oracle sat **~29 standard
   deviations *below* the random-allocation floor**, i.e. one model's configurations are mutually
   redundant.

> **Unrecorded and unverified.** The **killing numbers for pivots 1 and 2 are not recorded anywhere**
> in the repository. Only the qualitative verdicts survive (`CLAUDE.md` §2). The raw CSVs in
> `archive/image-difficulty/` were never mined for a figure. **[U]**
>
> The **−29σ** result is quoted from documentation (`CLAUDE.md` §2;
> `docs/archive_mcq/FINDINGS.md:70`); the script's output artifact was **not located** in
> `results/cascade_methods/artifacts/`. Treat it as **documented but unverified [U]**.

Pivot 3 became the project's template failure: a large oracle gap that no frozen-model signal can
harvest.

**The anchor the project started from (2026-06-17).** Cross-model cascade, MedVLThinker-7B →
MedVLThinker-32B, frozen margin gate τ = 0.426 calibrated on a held-out PMC-VQA training sample:
**accuracy 0.5718 at 63.3% escalation and 73.6% of always-32B compute**, over 6 benchmarks / 8,220
samples **[M]**.

**Pivot 4 — from "which model?" to "which *configuration* of which model?" (2026-06-17).** Three
measurements ended the "build a better gate" programme on day one **[M]**: the oracle that escalates
only the recoverable errors costs **11.2%** of always-32B compute at **4.6%** escalation, against the
deployed gate's 73.6%; **62.8%** of the cheap model's errors are futile to escalate (the 32B is also
wrong); of the gate's escalations only **22.5%** are beneficial and **15.2%** actively harmful; and
recoverability rises only from **28% to 43%** across margin quintiles. But labelling the 32B in
*every* mode showed reasoning was **hurting** it on perception: SLAKE 0.764 → 0.841 direct
(**+0.077**), VQA-RAD 0.776 → 0.893 (**+0.117**), at ~2 generated tokens versus ~477. Retargeting
escalation from reasoning-mode to direct-mode moved compute **69.5% → 48.6%** while raising accuracy
**0.653 → 0.660**. The resulting three-tier compute-configuration cascade measured **11.34 s → 2.27 s
(−80%)**, **6,318.8 J → 1,181.9 J**, compute **100% → 52%** at parity **[M/Mo]**.

**Pivot 5 — from multiple choice to open-ended answers (2026-06-24).** Every routing signal was
degenerate on 4-option questions. On free text the same signals jumped: cheap-wrong **detection AUROC
0.66 → 0.866**, recoverability **→ 0.804**, pooled over 3 datasets **0.846 [0.830, 0.862]** **[M]**.
The cause is **4-option discreteness, not answer length** — open answers have a median of 1–2 tokens.
The honest split recorded at the same time: detection improved, but **cascade gain (oracle minus
cheap) moved only +0.02 (multiple choice) → +0.06 (open)**.

**Then: from reading a frozen model to training a small one (2026-06-25/26).** With the gate
saturated, and with a scope decision by the researcher making one whole family of mechanisms
permanently out of bounds (the method must always answer), only two axes remained: *action* and
*selection*. Every **training-free** selector sat at the random-pick floor: random 0.720,
self-verification **0.715 (below random)**, self-consistency majority 0.736, big-model listwise 0.758
— while simply running the big model once scored 0.819. A small **trained LoRA verifier** broke that
floor for the first time: free text 0.413 → **0.501** against an oracle of 0.592, i.e. **49% of the
gap**; organ bounding boxes 40%; chest X-ray boxes **78%** **[M]**.

**Then: from "what does the method cost?" to "what is the honest baseline?" (2026-07-01, decisively
2026-07-07).** First, the project abandoned its own harness for **MedEvalKit**, which reproduces the
published numbers of the models being used — anchor: Lingshu-32B on MMMU-Medical **0.633** against
the paper's 62.3. Second, it discovered it had been comparing itself against the *cheapest* way to
run the big model. Measuring the honest one: **32B-with-reasoning on open text is 10,521.6 ms /
2,001.9 J against 665.0 ms / 126.9 J direct — and it is *less accurate*.** Fully measured a day later:
SLAKE-open **0.6791** vs 0.8186 direct, VQA-RAD-open **0.5450** vs 0.6000, PathVQA-open **0.1087** vs
0.3760 **[M]** (`opentext_32b_think_full.json`).

**Finally: the format-aware cascade (2026-07-04 onward).** Because the multiple-choice margin gate has
no free-text analogue and the trained verifier is free-text-specific, the deliverable became a
**deterministic router that detects, from the prompt text alone and never from the gold answer,
whether a question is multiple-choice or open-ended**, then runs a different policy for each:
margin-gated `7B-direct → 32B-direct` on the closed arm; `7B best-of-N + trained verifier →
32B-direct` on the open arm. Tested on three families it gave parity accuracy
(Δ −0.0001 / −0.0003 / +0.00004) at **0.38–0.77× compute and 0.68–0.88× latency** **[M/Mo]** — all
figures under the sample-weighted convention that §5.6 retires.

## 3.4 The five claims under test as of 2026-07-29

These are the load-bearing propositions the correction pass set out to break. Each is stated **as it
stood then**, with the weighting convention it was formulated under. Their outcomes are in §5.8.

**Claim (a) — Chain-of-thought reasoning *hurts* perception-type medical VQA and *helps*
reasoning-type benchmarks.**
*As published (2026-07-08 record, superseded — see §5.1):* on the 5-family internal bake-off,
reasoning was strictly worse than answering directly in **15 of 20** perception cells, **19 of 20**
no worse than +0.02, VQA-RAD negative in all 5 families, with exactly one genuine perception win
(MedGemma-27B on PathVQA, +0.040); on reasoning benchmarks, reasoning helped (MMMU gains **+0.027
Lingshu / +0.100 MedVLThinker / +0.120 InternVL3-38B** on MedEvalKit). *Why it is load-bearing:* it
is the entire justification for making the 32B's *direct* mode the escalation target and reserving
reasoning for a gated third tier — and it is the only cross-family finding in the project.

> The "19/20" band is **one-sided**: `finding1_corrected_2026-07-29.json : _meta.noise_band` reads
> *"'within noise' = delta ≤ +0.02"*, and the audit key is literally `n_within_noise_le_plus_0_02`.
> It must never be written "±0.02" — under the primary corrected policy only **5 of 20** cells have
> |delta| ≤ 0.02 (`counts_by_policy.P1_audit_best_matched.perception.n_abs_delta_le_band = 5`).

**Claim (b) — The answer format, not the model, determines whether test-time compute (sampling,
routing) buys anything.**
*As published:* multiple choice — detection AUROC ~0.66–0.73, recoverability ~0.578–0.6; open text —
detection **0.866**, recoverability **0.804**, pooled **0.846 [0.830, 0.862]**, holding under an LLM
judge and on a fourth imaging modality; on multiple choice a generative verifier **craters** (content
mode 0.132 against 0.534) while on free text a trained one works. *Consequence:* "beat the strong
model" is an open-text phenomenon; multiple choice is saturated.

**Claim (c) — The cascade Pareto-dominates every fixed way of using the 32B.**
*As published (sample-weighted, Variant B, n = 42,224):* compute-lean **0.5741**,
**+0.0150 [+0.0107, +0.0192]** against always-32B-with-reasoning
(`opentext_32b_think_full.json : headline_variant_b_mmmu_excluded.compute_lean`), at **469.0 ms /
83.6 J / 2.248 FLOP-eq — 0.462× the baseline honestly re-costed, 0.492× as charged**
(`honest_recosting_2026-07-29.json : part3_corrected_headline.method_cost_variant_b.compute_lean`);
accuracy-max **0.5836**, **+0.0245 [+0.0216, +0.0274]** against reasoning at 0.932× compute
(`f8_mode_vsthink_ci.json : headline`). **Only always-7B, compute-lean and accuracy-max were
non-dominated** — all three 32B strategies, including a non-deployable per-benchmark oracle
mode-selector, were dominated.

> **Naming hazard.** `opentext_32b_think_full.json` labels **0.5862 / +0.0271** as `accuracy_max`;
> every later artifact calls that system **fusion** (decision-level fusion of both legs). The
> certified-veto accuracy-max row is **0.5836 / +0.0245**. Do not mix them.

**Claim (d) — A trained outcome verifier makes small-model best-of-N competitive with the 32B on open
text.**
*As published:* the LoRA verifier scores P(correct | image, question, candidate) with **per-answer
AUROC 0.924 (n = 8,512)**; selecting from 8 samples it gains **+0.1041 [+0.0891, +0.1190]** over
greedy decoding pooled across the three open cells
(`verifier_disjoint_retrain_2026-07-30.json : verdict.selection_stage_pooled_gain_over_greedy.contaminated`);
the assembled open arm reaches **0.5642** against always-32B-direct's **0.5168** on the same items;
and **7B best-of-8 (0.501) beats a 32B single pass (0.444)**, with training rather than size the
active ingredient.

> An independent bootstrap in `verifier_validity_2026-07-29.json` gives the same point estimate with
> the interval **[+0.0887, +0.1190]**. This document uses **[+0.0891, +0.1190]** throughout, from
> `verifier_disjoint_retrain_2026-07-30.json`, because that artifact computes the contaminated and
> clean arms on the same replicate stream. The two intervals differ by 0.0004 at the lower bound and
> are separate bootstrap runs of the same quantity.

**Claim (e) — Two limits explain every negative result in the project.**
*Limit A, the recoverability limit:* "will the strong model fix *this* error?" is ~0.5–0.6 AUROC from
anything cheap — **sixteen** independent mechanisms hit it (hidden-state probes, k-nearest-neighbour
retrieval, self-verification, gradient-boosted and rich-feature gates, the published post-hoc
deferral rule, cross-family and CLIP-style routers, a full LoRA fine-tune, logit- and decision-level
fusion, super-learner ensembling, learned slice discovery, credibility shrinkage, the open-ended
cascade). *Limit B, the selection limit:* a verifier converts only **74–82%** of the oracle-of-N gap,
and **thirteen** attempts hit it, killed three orthogonal ways (a 7× larger verifier merely ties;
diverse generation and pairwise comparison do not compound, −0.0117; no pre-filter beats plain
diverse-plus-pointwise). Behind both sits a **coverage** measurement: of 1,064 held-out questions,
**434 (40.8%) have no correct answer anywhere in the 8-sample pool**, against a total selection gap of
**0.0912** — the coverage limit is 4.5× the selection limit.

> The counts **sixteen** and **thirteen** are the project retrospective's own tallies (§5.5). They
> were **not** independently re-counted against the negative-results catalogue in this pass. **[U on
> the counts; the individual mechanisms are documented]**
>
> The 40.8% figure's denominator was confirmed: `ckpts/train/lora_verifier_pooled4/perq_sc8.json`
> holds exactly **1,064** entries **[M]**. It is corroborated by an independently measured pooled
> oracle-of-8 of **0.6260** against greedy 0.4495 on the 2,345-item open set — 37.4% of items
> unreachable by any selector (`verifier_disjoint_retrain_2026-07-30.json`) **[M]**.
>
> The selection limit is quoted two ways with two different denominators: **74–82% of oracle-of-N**
> (the retrospective's own efficiency measure, computed with the *contaminated* verifier) and
> **oracle conversion 0.589 → 0.203** (`(verifier − greedy) / (oracle − greedy)`, from the disjoint
> retrain). **No document reconciles the two definitions.** Both are reported here with their sources
> rather than merged.

## 3.5 Why each claim was worth doubting — the fragilities visible *before* testing

**(a) Reasoning-hurts-perception — the two arms were never verified to differ only by reasoning.**
Prompts are **not persisted in any checkpoint row**; recovering an arm's prompt requires tracing a
`ckpts/` directory back to a shell variable in `runners/*.sh`. Nothing asserted that a "think" arm
ever produced a trace. Worse, the June "fix" that switched every family to its *native* reasoning
recipe made the arms differ by persona, answer-format clause, and — for Lingshu and QoQ — **image
resolution**. Two symptoms were visible in advance: **Lingshu's "native think" arm generated 3.0 mean
tokens against 3.0–3.3 for its direct arm** — it never reasoned at all — and Lingshu is precisely the
family the final method is built on.

The baseline was equally suspect. The dumps used as *always-32B-with-reasoning* have mean generated
tokens of **3.09 (PMC-VQA) / 3.33 (SLAKE-closed) / 3.01 (VQA-RAD-closed)** and agree with the plain
direct run on **92.00% / 97.01% / 97.21%** of predictions **[M]**
(`honest_recosting_2026-07-29.json : part1_verify_the_premise.per_cell`); **PathVQA-closed has no
reasoning dump at all** (`reasoning_dump: null`, `verdict: "NO-DUMP"`, imputed as reasoning = direct);
and genuine reasoning exists for only **4,345 of 42,224 items = 10.3%** of the pool — yet all nine
cells were charged 10,521.6 ms / 2,001.9 J. An honest re-costing already implied the baseline is
**2,018.1 ms / 487.2 J** under the primary cost model, turning −95.5% / −95.8% into **−76.8% /
−82.8%** (§4.2).

**(b) Answer format — the open-text half rests on one judge, unmatched prompts, and small n.**
`src/labeling/run_openvqa.py:26` gives the **direct** arm a persona plus *"Answer the question with a
short, specific phrase. Do not explain"*, and `:27` gives the **think** arm a `<think>` instruction
that **drops both**. On free text that is a live **style/length grading channel**, not the bounded
extraction channel that makes the multiple-choice comparison safe. The grading itself is a single
point of failure: open-text exact match is broken in the harness, so everything routes through one
LLM judge, and the independent cross-validation of that judge covered **SLAKE and VQA-RAD only — not
PathVQA**, the load-bearing open cell. PathVQA's golds are caption fragments: **53.3% are a single
word, 75.6% are ≤2 words**, against a reasoning arm producing **4.28 words on average** versus the
direct arm's **1.95** — exactly the shape of an artifact, with the collapse magnitude tracking gold
quality (PathVQA −0.267, SLAKE −0.140, VQA-RAD −0.055) **[M]** (`pathvqa_judge_audit.json : descriptive`).

**(c) Pareto-dominance — one cell held 79% of the weight, and the cost model had three soft spots.**
PMC-VQA is **79.2%** of the sample-weighted pool and also the **lowest-escalation cell (8.45%)**,
while escalation ranges up to **89.60%** on MedXpert — so the pooled average was substantially
PMC-VQA's number, and the pooled *cost* substantially PMC-VQA's cost. Against always-32B-direct,
**4 of 8 cells contribute exactly 0.0000** for accuracy-max (SLAKE-closed, VQA-RAD-closed,
PathVQA-closed, MedXpertQA-MM) and **0 of 8** for compute-lean **[M]**
(`honest_recosting_2026-07-29.json : part4b_zero_contribution_cells`) — the method simply *is*
always-32B-direct in those cells. That same PMC cell is the auto-generated, **unverified** v2 split,
where every model plateaus at 0.52–0.56, and both legs are trained on PMC-VQA's training split.

The cost constants were thin in three places: the reasoning constant is **n = 15** with a 23%
mean/median divergence and the smaller value used, measured on VQA-RAD open-text images and
transferred to every multiple-choice cell; the best-of-8 latency of **522 ms** is **asserted, not
measured**, and pairs with an 8×-billed energy figure that implies **~1,088 W** against ~132 W
measured and a 400 W card TDP — an energy-consistent bound puts it at **≥1.42 s**; and the 4.57
compute ratio is an underived literal. Finally, the method has **no materialised τ**: thresholds are
refit inside every cross-validation fold against *that cell's own* strong-leg accuracy, and the
policy router picks a different policy per benchmark from intervals computed on the data it then
reports (18 uncontrolled tests) — so "deployable single policy" was never demonstrated.

**(d) The trained verifier — it graded a test it had largely already seen.**
The verifier was trained on a 70/30 grouped-by-question split of four open sets and then scored over
the **full** sets, so **67.4% (SLAKE-open), 71.0% (PathVQA-open) and 73.0% (VQA-RAD-open)** of every
reported open item was in its training data **[M]** (`verifier_validity_2026-07-29.json : A_overlap`).
A pre-test estimate of the damage already existed: comparing the *selection gain* rather than raw
accuracy showed one-directional inflation on all three cells (SLAKE 1.82×, PathVQA 1.44×, VQA-RAD
1.19×; n-weighted full **+0.1040** against held-out **+0.0718**, i.e. **~31% memorisation**). Two of
the three open cells have intervals spanning zero at n = 200 and n = 645, so pooled significance
rested on PathVQA-open — the same cell flagged under (b), and itself a non-random prefix slice. The
open arm holds **37.5% of the macro weight**.

**(e) The two limits — the most robust part of the project, and still not immune.**
First, a measurement made during the retrospective itself: a **peer-difficulty signal** (how many of
six aligned model configurations get the item right) reaches recoverability AUROC **0.649 / 0.583 /
0.789 / 0.799 / 0.774** on PMC / SLAKE / VQA-RAD / PathVQA / MedXpert, where **the deployed margin is
*anti*-predictive on 4 of 5 sets (0.407 / 0.236 / 0.472 / 0.670 / 0.450)** **[M]** — so "nothing
predicts recoverability" was, as literally stated, already false. Only the follow-up rescued it as a
*deployability* claim: a cross-fit rule over [margin, peer, margin×peer] at a 20% escalation budget
gains **+0.0001** on PMC against oracle headroom of +0.1077, and +0.0072 / +0.0022 / +0.0144 / +0.0175
elsewhere, at the price of four extra model forwards.

Second, the project has a documented instance of a CPU re-simulation manufacturing a confident wrong
negative: a **simulated** pairwise verifier reported parity (−0.003, ceiling +0.000) and a **real**
A-versus-B forward pass overturned it the same day (**+0.036 [+0.016, +0.055]**). Since the final
method's cost model is itself a CPU re-costing of saved dumps — nothing has ever been executed
end-to-end as a live pipeline — the same failure mode is live for everything downstream. Third, the
selection-limit verdict rests on **two training runs at the same end of the data-scaling axis**, with
the intervening runs never executed.

**One further reason to doubt everything at once.** The project's own correction log contains sixteen
prior retractions, several of which reversed a headline — a coordinate-space bug that turned the
strongest positive in the project into a false negative, a gate ranking built on the wrong proxy, a
"no promptable reasoning mode" claim retracted inside 24 hours, an MMMU cell that turned out to be
contaminated. A project with that record has earned the presumption that its *current* headline
contains a similar defect; the 2026-07-29/30 pass was the attempt to find it deliberately rather than
wait for a reviewer.
---

# 4. Methodology — how each thing was tested

This section is about experimental design, not results. For each investigation the 2026-07-29/30
correction pass rests on, it states the hypothesis, the design, the controls, what would have
falsified the claim, and the verification gates that had to pass before the number was believed.
Where an instrument is unusual — a monkeypatched dependency, a pattern-multinomial bootstrap, a
Wilson bound, a leave-one-cell-out range — it is explained from first principles, because the craft
is in the instrument.

## 4.1 The shape every investigation was forced into

Four rules recur, and they are why several headline claims could be retracted cleanly rather than
argued about.

1. **Everything is paired at the item level.** Two systems are only ever compared on the *same*
   questions, and the statistic is the per-item difference. Question difficulty is then common noise
   that cancels, instead of variance that hides the effect.
2. **Every threshold is cross-fit.** No τ, no adaptive-sampling parameter, no veto bin and no
   deferral rule is chosen on the data it is then scored on (`heldout(..., K=5)`,
   `src/cascade_methods/integrated_method.py:207`).
3. **Every experiment carries a pre-declared falsifier.** Before running, the design names the
   measurement that would kill the claim. Several did kill claims.
4. **Nothing is believed until a reproduction gate passes.** Each analysis must first re-derive a
   number that is already published, from the raw dumps. If the re-derivation drifts, the wiring is
   wrong and the new number is not a finding. This is the single most useful discipline in the whole
   record — the disjoint-verifier work compared **1,224 fields exactly** before it was allowed to
   report anything new (`macro_headline_clean_verifier_2026-07-30.json : validation.n_fields_compared_exactly`).

**Vocabulary.** A **cell** is one (benchmark × answer-format) reporting unit — `SLAKE_closed`,
`PathVQA_open`; there are 8 in the reported suite. An **arm** is one prompt/configuration of one
model. **Direct** means the model answers without producing a reasoning trace; **reasoning** means it
produces one.

## 4.2 The honest re-costing — measuring the baseline's true generation cost per cell

### Hypothesis

The baseline called *always-32B-with-reasoning* was billed one global constant — **10,521.6 ms /
2,001.9 J per query** — on every cell. That constant was measured on **n = 15** batch-1 samples of
open-text VQA-RAD at **98.3 generated tokens**. Hypothesis under test: *on most cells the "reasoning"
run never reasoned, so the baseline was being charged a price it did not pay.*

### Design — two stages, premise first

**Stage 1: verify the premise before repairing anything.** For each of the 8 cells, measure from the
dumps (a) the **mean generated tokens** in the reasoning run and (b) its **prediction agreement** with
the plain direct run. A threshold was declared *in advance*: a cell counts as `REASONED` only if the
reasoning arm emitted **≥ 50 mean generated tokens**
(`part1_verify_the_premise.threshold_mean_gen_tok`). All values **[M]**:

| cell | generated tokens, "reasoning" run | agreement with direct | verdict |
|---|---:|---:|---|
| PMC-VQA (n = 33,430) | 3.09 | 0.9200 | NOT-REASONED |
| SLAKE-closed (836) | 3.33 | 0.9701 | NOT-REASONED |
| VQA-RAD-closed (251) | 3.01 | 0.9721 | NOT-REASONED |
| PathVQA-closed (3,362) | — (**no dump exists**) | — | **NO-DUMP** (imputed = direct) |
| MedXpertQA-MM (2,000) | 320.33 | 0.229 | REASONED |
| SLAKE-open (645) | 122.41 | 0.4884 | REASONED |
| VQA-RAD-open (200) | 104.54 | 0.395 | REASONED |
| PathVQA-open (1,500) | 141.47 | 0.0527 | REASONED |

Only **4,345 of 42,224 items (10.3%)** come from cells where the reasoning mode genuinely fired. The
agreement column is the independent corroboration: a run that "reasons" and yet reproduces the direct
run's answer 92–97% of the time is a direct run under another name; the genuinely reasoning cells
agree only 5–23% of the time.

> **PathVQA-closed carries no reasoning measurement at all.** Its reasoning column is **imputed equal
> to direct** while still being charged the full reasoning latency and energy constant. Any
> "vs reasoning" number involving that cell is partly an imputation, not a measurement — and note
> that the field name in some artifacts (`acc_32b_think_measured`) is misleading on this point.

**Stage 2: rebuild the cost model from something measurable.** Batch-1 latency and energy for a
single forward are close to affine in generated length:

```
cost(query) = P + D · g          P = prefill intercept, D = per-generated-token rate, g = tokens generated
```

`D` was identified from a two-point contrast inside one measurement file (`artifacts/latency_32b.jsonl`,
n = 60 per configuration, medians), holding harness, resolution cap and images fixed and varying only
the mode **[M]**:

- `D_latency = (21,865.2 − 333.15) ms / (316 − 2) tok = **68.573 ms/token**`
- `D_energy = **18.261 J/token**`

`P` was then calibrated so the model reproduces the project's own direct anchor (665.0 ms / 126.9 J)
at that anchor's measured generation length (5.6 tokens) → `P = 280.99 ms / 24.638 J` **[D]**. Each
cell is then charged its own measured `g`.

> **A 0.1 J discrepancy worth naming once.** The measured direct-mode energy anchor is **126.9 J**
> (`honest_recosting_2026-07-29.json : provenance.anchor_direct.energy_j`); the constant used
> throughout the cost model is **127.0 J** (`paper_baselines.py:64`, `GEN32N`). 126.9 is the
> measurement; 127.0 is the rounded constant the cost code carries. They are the same measurement.

### Controls

- **Five cost models, not one** (`part2_corrected_cost_model`, M1–M5), reported as a sensitivity band
  rather than a single number: **M1 (primary)**; M2 using the repository's own literal prefill
  fraction φ = 0.586 → P = 389.7 ms; M3 using the quantization script's decomposition → P = 527.9 ms;
  M4 fitted to reproduce *both* published anchors exactly (forcing D = 106.3 ms/tok and an implausible
  ~70 ms prefill); M5 the crude binary "full reasoning price where a reasoning run exists, full direct
  price elsewhere". Deliberately including the repository's two *contradictory* internal constants is
  the point: it shows how much the conclusion depends on a disputed decomposition. It does not.
- **Two independent identification cross-checks.** The energy intercept derived from the 665 ms anchor
  (24.638 J) and from `latency_32b.jsonl` alone (24.678 J) agree to **99.8%** — the energy model is
  well identified. The *latency* intercepts do **not** agree (280.99 vs 196.0 ms) because the two runs
  used different images; this disagreement is recorded, and the anchoring choice is stated rather than
  hidden.

### What would have falsified it

If the per-cell generated-token counts had been large everywhere, the premise dies and the global
constant is fine. They were 3.0–3.3 on four of five multiple-choice cells.

### Why a single global constant was wrong — in *both* directions

- Where no reasoning happened, the constant **overstates** the baseline's latency and energy: charging
  10,521.6 ms to a cell that emitted 3.09 tokens (M1 cost 492.88 ms) inflates the method's apparent
  advantage.
- Where reasoning *did* happen, the convention **understates** the baseline's compute:
  FLOP-equivalents were charged a flat 4.57 regardless of decode length, while the genuinely reasoning
  cells actually cost **5.90–8.85** FLOP-eq.

So the same convention that inflated the latency win *deflated* the compute cost.

**The corrected baseline [D/Mo].** Pooled over Variant B, the baseline moves from
`10,521.6 ms / 2,001.9 J / 4.57 FLOP-eq` (as charged) to
**`2,018.135 ms / 487.24 J / 4.87 FLOP-eq`** under **M1, the primary model**
(`part3_corrected_headline.baseline_honestly_recosted.M1_primary`). Compute-lean's advantage falls
from **−95.5% parallel latency / −95.8% energy** to **−76.8% / −82.8%**
(`corrected_advantage.compute_lean.M1_primary`), with the five-model band spanning **−72.1%
(M5) to −83.0% (M4)** on parallel latency. The *direction* survives every model in the band; only the
magnitude moves.

> **Do not quote M5 as the primary.** The M5 sensitivity bound (1,679.279 ms / 319.844 J, giving
> −72.1% / −73.9%) appeared in an earlier draft as though it were the primary result. It is the
> crudest of the five models and sits at the optimistic end of the band. **M1 is primary.**

### Verification gates run in the same artifact

The re-costing script also re-checked two claims from the document that commissioned it, and
**contradicted one of them**: the "89% of the delta comes from two cells" claim was CONFIRMED
(PathVQA-open + PMC-VQA = **89.7%** of the pooled +0.0245), while "5 of 8 cells contribute exactly
zero" was **PARTIALLY REFUTED** — the true count is **4 of 8** for accuracy-max and **0 of 8** for
compute-lean (`part4b_zero_contribution_cells`). An audit that only ever confirms its commissioner is
not an audit.

## 4.3 The matched-prompt experiments

Two separate experiments, on two harnesses, testing the same defect: the reasoning and direct arms of
the project's flagship comparisons **differed by more than reasoning**.

### 4.3.1 Free text — why the naive matched prompt is unusable, and the fix from the other side

**Hypothesis.** "Reasoning hurts perception on open-text medical VQA" (Lingshu-32B: pooled 0.5168
direct against 0.3028 reasoning, Δ −0.2141) was measured with prompts differing in three ways at once.
The direct arm was *"You are an expert medical image analyst. Answer the question with a short,
specific phrase. Do not explain."*; the reasoning arm was *"You will solve a problem/request. You
should provide your thoughts within `<think></think>` tags before providing the answer. After
`</think>`, give only the short final answer."* — persona, style constraint and no-explanation clause
all dropped. On free text this is a live **grading channel**: an LLM judge marking a verbose answer
wrong is not evidence about reasoning.

**The naive design, and why it is unusable.** The obvious fix is to add the direct arm's wording to
the reasoning arm (arm A, `SYS_THINK_MATCHED`). It fails for a mechanical reason: this model family
only emits a `<think>` trace when the trigger sentences appear **verbatim**, and paraphrasing them —
or telling the model "Do not explain" — **suppresses the behaviour under test**. Measured
`reasoning_trace_rate` per dataset **[M]**
(`matched_prompt_reasoning_2026-07-29.json : per_dataset.*.diagnostics.reason_matched_{A,B}.reasoning_trace_rate`):

| arm | SLAKE-open | VQA-RAD-open | PathVQA-open |
|---|---:|---:|---:|
| arm A (`SYS_THINK_MATCHED`) | 0.6868 | 0.2550 | 0.6340 |
| arm B (`SYS_THINK_MATCHED2`) | 0.6698 | 0.3050 | 0.7107 |

> The measured range across both arms is **0.2550–0.7107**. That same artifact's *prose*
> (`interpretation.trap_1`) says "0.27–0.78 by dataset", which matches neither arm's measured values.
> The measured values are used here; the prose figure is an internal inconsistency in the artifact.

So arms A and B score **above** the unmatched reasoning arm (pooled 0.4235 / 0.4192 against 0.3028),
which naively reads as "most of the gap was the prompt". That reading is wrong, and the design proves
it wrong by **conditioning on whether the trace actually fired [M]**:

| arm B, PathVQA-open | n | arm B accuracy | direct accuracy, *same items* | Δ |
|---|---:|---:|---:|---:|
| trace fired | 1,066 | 0.1154 | 0.2730 | **−0.1576** |
| no trace | 434 | 0.6382 | 0.6290 | +0.0092 |

The entire deficit sits on the items where reasoning happened (same pattern on SLAKE: −0.1134 against
−0.0329; VQA-RAD: −0.0820 against −0.0288). Arms A and B are a *mixture* of reasoning and direct
answering, so their apparent recovery is **dilution, not repair**, and their implied "prompt share of
the gap" is an **upper bound**, not an estimate.

**The fix: match from the other side.** Instead of adding style to the reasoning arm (which kills
reasoning), *strip style from the direct arm*. A fifth arm, `direct_unstyled` — *"You will solve a
problem/request. Give only the short final answer."* — is the reasoning arm's wording **minus the
trigger**. This yields a clean 2×2 **[M]**:

| | reasoning **off** | reasoning **on** |
|---|---:|---:|
| **unstyled** convention | 0.5186 (`direct_unstyled`) | 0.3028 (`reason_unmatched`) |
| **styled** convention | 0.5168 (`direct`) | 0.4192 (arm B; trace fires on only part of the set) |

- **Reasoning effect at a fixed convention:** **−0.2158 [−0.2354, −0.1962]**, significant.
- **Convention effect with reasoning off:** **−0.0017 [−0.0111, +0.0077]**, **not significant**.
- **Attribution:** convention −0.8%, reasoning 100.8% of the original gap.

**What would have falsified it.** A large style effect with reasoning held off. It was ~0. The
confound is real *as a description of the two prompts* and contributes ~nothing to the measured gap.

**Verification gates.** An algebraic **identity check** — `original gap == reasoning-at-unstyled −
convention effect` — reproduces to a residual of **−0.0000** pooled (0.0001 on SLAKE), proving the
decomposition is arithmetic, not narrative. Everything else was held constant and listed explicitly:
same model snapshot, cap320 image budget, greedy temperature 0, `max_model_len` 4096, `max_tokens`
512, tensor parallel 2, identical answer extraction (text after the **last** `</think>`), identical
evaluated index sets, identical judge. Dumps were written with `--save_raw` so the trace rate is
re-verifiable offline.

**Honest residuals recorded with the result.** The symmetric test — reasoning on/off at the *styled*
convention — is not cleanly runnable on this family, because the styled wording suppresses the trace;
arms A and B bound it and both stay significantly below direct. VQA-RAD-open (n = 200) is
under-powered and inconclusive in every arm. Arm B moves the persona sentence's *position* (it must
follow the trigger, or the trace stops firing) — position, not content, is the residual difference.

**Propagation, reported rather than absorbed.** Substituting each arm's open-text vector into the
baseline moves the Variant-B headline from +0.0245 to **+0.0178 (arm A) / +0.0180 (arm B)**, a shift
of −0.0067 / −0.0065 **[D]**.

> **Scope limit.** This experiment is **Lingshu-32B only**. No matched open-text arm exists for any
> other family, so §5.3 does not generalise across families and is not cross-checked by a second
> architecture.

### 4.3.2 Multiple choice / MedEvalKit — an environment-gated monkeypatch, and an assertion that saved six GPU runs

**Hypothesis.** The "reasoning helps reasoning-heavy benchmarks" gains (MMMU +0.027 Lingshu / +0.100
MedVLThinker / +0.120 InternVL3-38B) come from a pair of prompts that differ **three ways at once**:
the reasoning trigger, the word *"directly"*, and `\boxed{}` versus a bare letter.

**The dependency constraint.** `MedEvalKit/` is a protected dependency, and an earlier uncommitted
local edit to it had already caused damage: it added a real reasoning trigger but **deleted** the
answer-format clause, which is exactly why the post-edit `*_reason` dumps reason yet are
format-unmatched. The design therefore had to change the prompt **without touching the dependency**.

**The instrument** (`src/labeling/medeval_matched_prompt.py`): an **environment-gated monkeypatch**.
With `MEDEVAL_MATCHED_PROMPT=1` it imports MedEvalKit's prompt builders
(`utils.question_formats.get_multiple_choice_prompt` and `utils.MMMU.data_utils.construct_prompt`),
wraps them, and replaces **only the instruction tail** — asserting the expected upstream tail is
present, so the question/option body cannot silently drift. Because MedEvalKit uses `from X import f`,
the patch also rebinds every already-imported reference. With the variable unset the wrapper is a
transparent pass-through, so an unpatched run is bit-for-bit an upstream run, and `MedEvalKit/` stays
byte-identical to upstream on disk.

**Three assertions run unconditionally at import:**

- **A — matched by construction:** the two installed tails must differ by *exactly* the clause
  `"First reason step by step about the question and each option, then "`, verified by rebuilding the
  reasoning tail from the direct tail and requiring string equality.
- **B — byte-identical to the pre-revert edit:** the reasoning string is AST-parsed out of backup
  copies of both patched files and required to be **byte-identical** to the string that produced the
  existing `eval_results_*_reason/` dumps. This is the load-bearing assertion: it means the six
  existing reasoning dumps remain a valid comparator and **only the direct arm had to be re-run**.
- **C — no double-patching:** MedEvalKit on disk must still carry upstream's tails and *not* the local
  edit; otherwise the run refuses to start.

**The decomposition.** Three arms on one shared item set: **rung 1** published-direct (bare letter),
**rung 2** matched-direct (`\boxed{}`, *no* trigger), **rung 3** reason (trigger + `\boxed{}`). Then
`format Δ = rung2 − rung1`, `trigger Δ = rung3 − rung2`, `published Δ = rung3 − rung1`. This splits
the published gain into an answer-**format** effect and an explicit-reasoning-**trigger** effect.

**Controls and verification gates [M]:**

- **Pairing:** `ids_identical_across_arms = true` for all 6 (family × benchmark) cells; 150 / 2,000
  rows per arm.
- **Token audit:** Lingshu's matched-direct arm emits 3.04–4.43 tokens with 0% trace rate while its
  reason arm is 100% at 284.7–321.7 tokens; MedVLThinker's *matched-direct* arm traces on **98.9–100%**
  of items at 416.5–580.1 tokens; InternVL3's on 93.5–95.2% at 193.1–288.9. That measurement *is* the
  mechanism: **asking for `\boxed{}` is itself a reasoning trigger.**
- **Extraction gate:** on the new `direct_matched` arm, `parse_ok` is **exactly 1.0000 in 8 of the 9
  primary sub-cells**, with a minimum of **0.9986** (InternVL3-38B, MedXpert-Reasoning, n = 1,446). If
  all three arms are counted per sub-cell, 7 of 9 are at exactly 1.0000 (Lingshu's MedXpert-Reasoning
  `reason` arm is 0.9993). **No delta can be an extraction artifact.**
- **Residual-scope audit:** only the multiple-choice prompt branch is matched; MMMU's 5 of 150 "open"
  items keep upstream's format-unmatched strings. Those five were checked and are **0/5 correct in all
  nine arms**, so they cannot move any delta — which is why the fully matched **MMMU-MCQonly
  (n = 145)** cell is the one quoted. *(If those items were ever scored non-zero, the MMMU deltas
  would move.)*
- **Settings matched to the reason arm's own runner:** seed 42, tensor parallel 2, `max_new_tokens`
  2048, temperature 0, `top_p` 1e-4, `datasets_path=hf`. **One known unmatched axis is declared:**
  `EVAL_BATCH_SIZE` 250 against 2000 (OOM safety at TP=2). This affects **MedXpert only**, under
  greedy temperature-0 decoding, so at most rare batch-composition tie-breaks — but it is a genuine
  unmatched axis.
- **A run incident treated as a matching problem, not a bug:** InternVL3-38B on MedXpert failed twice
  at `MAX_MODEL_LEN=16384` on a ~20.2k-token item. The `*_reason` arm had itself been re-run at 24000
  (`runners/run_clean_latency_reruns.sh:23`), so **24000 is the matched value**; 16384 was our error,
  and the cell was re-run at 24000.

**What would have falsified it.** If, with the format matched, the trigger still bought significant
accuracy, the published attribution stands. Result: **0/9 trigger effects reach significance** (8/9
point-positive; mean delta shift from matching −0.0276), while **3/9 format effects do**.

**The honest substitute for an unobtainable contrast.** A clean "reasoning versus no reasoning"
contrast does not exist in these families, because the only prompt that suppresses the trace also
changes the answer format. The design's answer is to report the **monotone ladder** instead —
MedVLThinker MMMU-MCQonly: **0.634 @ 2 tokens → 0.697 @ 431 → 0.738 @ 580**. The standing rule
produced: *any think-versus-direct pair must be format-matched **and** token-audited; a "direct" arm
that emits hundreds of tokens is not a direct arm.*

> **Two dependency caveats.** MedEvalKit carries two local uncommitted edits
> (`utils/question_formats.py:11` and `utils/MMMU/data_utils.py:158`, both modified 2026-07-02) that
> **replaced** rather than appended the reasoning trigger. Whether to revert the dependency and re-run
> is an open researcher decision, not settled here.

## 4.4 The cross-family prompt-matching audit and re-derivation

**Hypothesis.** Triggered by `pathvqa_judge_audit.json : prompt_confound`, the question was whether
the flagship cross-family table ("reasoning hurts perception in 15/20 cells") is a prompt artifact.

**Design — five steps, offline, no GPU, no new inference.**

1. Map every cell to the checkpoint that produced it.
2. **Reproduction gate:** recompute accuracy from those checkpoints and require it to reproduce
   `master_data.csv` / `generalization.json` **exactly** — all 35 cells did.
3. Recover **both arms' verbatim prompts**. These are *not persisted in the JSONL rows*; they exist
   only as shell variables in `runners/*.sh` and module constants, so each had to be traced to a
   `file:line`. (The action item this produced has nothing to do with the numbers: **persist the
   prompt in every future checkpoint row**.)
4. Measure the only channel an unmatched prompt has on multiple choice — `parse_ok`, unparsed
   fraction, mean generated tokens — per arm.
5. Recompute the finding from better-matched arms that already exist on disk.

**Classification of every comparison.** Each of the 8 comparison groups was labelled `MATCHED`,
`UNMATCHED-MILD`, `UNMATCHED-SEVERE (format-dropped)`, or `INVALID-AS-REASONING-EVIDENCE`, with a
written rationale:

- *MedVLThinker* — MILD: the reasoning arm's system prompt replaces the letter-only constraint, but
  extraction failure is **0.0000 in both arms on all 7 benchmarks**; also resolution-unmatched
  (cap320 against fullres).
- *QoQ* — MILD: **both** arms constrain the final answer to a letter (`\boxed{}` in the reasoning
  arm), so the format channel is preserved.
- *Chiron* — SEVERE: the reasoning instruction replaces the format constraint and **nothing** replaces
  it; highest measured reasoning-arm extraction failure (3.4% on SLAKE).
- *Lingshu* — **INVALID**: the "native think" instruction contains **no reasoning trigger at all** —
  measured generated tokens **3.0 against 3.0–3.3**. Those 7 cells compare two answer-format prompts
  and cannot count as evidence in *either* direction.

**Why the multiple-choice half is structurally immune.** This is a structural argument, not a
statistical one. The open-text defect acts through a **grading channel**: on free text, style and
length determine whether the judge marks an answer correct, so dropping "short, specific phrase" /
"Do not explain" moves the score independently of reasoning quality. On multiple choice the gold is a
single option letter graded by exact equality — `ok = int(g == p)` with `gold = answer_label[:1]`,
at `src/labeling/run_vlm_eval.py:172` and `:79`, and identically `run_peer_eval.py:212`
(`finding1_corrected_2026-07-29.json : _meta.grading_channel`). **There is no length or style freedom
to penalise**, so that channel does not exist. *(A companion argument in
`finding1_prompt_matching_audit.json : mcq_structural_immunity.argument[1]` cites the adjacent lines
`run_vlm_eval.py:80` and `run_peer_eval.py:81`; both citations describe the same exact-equality
grading.)*

The residual channel is answer **extraction**, and it is directly measurable — and it was measured:
`parse_ok ≥ 0.9663` in every reasoning arm across all 35 published cells, and exactly **1.0000 in both
arms for MedVLThinker**, the family carrying the largest drops **[M]**.

**The adversarial bound.** Rather than argue that 0–3.4% extraction failure is small, the audit charges
*every* unparsed reasoning-arm answer as **correct** and leaves the direct arm as measured. This
changes **no count** (15/20 stays 15/20; 17/20 stays 17/20).

**Three arm-selection policies, to test robustness to the analyst's own choice [M]:**

- **P1** — swap only the reasoning arm for the best-matched dump on disk (direct arm unchanged, so
  every delta is directly comparable to as-published).
- **P2** — strictest available: also move the *direct* arm where that improves matching (Lingshu and
  QoQ to fullres so the pair is fullres-against-fullres).
- **P3** — as P2, but match MedVLThinker at **fullres** instead of cap320.

| policy | perception strictly negative | ≤ +0.02 | CI-significant negative | pooled Δ (n = 30,250) |
|---|---:|---:|---:|---|
| P0 as published | 15/20 | 19/20 | 12/20 | −0.0252 [−0.0304, −0.0199] |
| **P1 best-matched (primary)** | **17/20** | 19/20 | **14/20** | **−0.0401 [−0.0456, −0.0347]** |
| P2 strict resolution + format | 17/20 | 19/20 | 13/20 | **−0.0408** [−0.0462, −0.0353] |
| P3 strict, MedVLThinker @fullres | 17/20 | 19/20 | 13/20 | −0.0405 [−0.0459, −0.0351] |

> The pooled perception delta across the three corrected policies therefore ranges
> **−0.0401 to −0.0408** (P1 −0.0401, P2 −0.0408, P3 −0.0405). An earlier draft and the retrospective
> quote "−0.0401 to −0.0405", which understates the P2 end.

**What would have falsified it.** If better matching had pulled the effect toward zero, the finding was
a prompt artifact. It got **stronger**, and the as-published pairing is the outlier — that asymmetry is
the signature of a real effect. The strongest evidence is the **fully-matched-only subset**, where
nothing is left to correct (same absent system message, same image budget, format constraint in both
arms): 6/8 medical cells strictly negative, pooled **−0.0273 [−0.0367, −0.0176]** (n = 12,100), plus
two non-medical peer architectures that were already fully matched (InternVL2.5-8B pooled
−0.0076 [−0.0208, +0.0056], not significant; Phi-3.5-Vision −0.0187 [−0.0336, −0.0036]) **[M]**
(`finding1_corrected_2026-07-29.json : fully_matched_subset.{medical_perception_pooled, peer_pooled_by_model}`).
*(A sibling key, `finding1_prompt_matching_audit.json : recount.fully_matched_subset_only`, holds the
counts and rounded points only; the intervals and sample sizes live in the corrected artifact.)*

**Per-cell statistics.** Exact two-sided **McNemar** on discordant pairs plus a 10,000-resample paired
bootstrap percentile interval, seed 20260729.

**Precision caveat carried with the count.** 17/20 is a **count of signs**, not a measurement: per-cell
n runs from 170 (MMMU) to 3,362 (PathVQA), so at n = 170 a 95% interval is roughly ±0.07 and near-zero
cells can flip on resampling alone. The rule adopted: never quote the count without the pooled Δ and
the CI-significant subcount (**14/20**).

**The reasoning half is *not* robust to the same treatment**, and the design surfaced that:
strictly-positive counts run 10–12/15 across policies with only **3–4/15** CI-significant, and two of
five families change verdict once matched (Lingshu withdrawn entirely; QoQ downgraded).
## 4.5 The PMC-VQA item-level audit — and why the control group is the whole experiment

**Hypothesis.** PMC-VQA carries 33,430 of 42,224 pooled items (79%) and supplies the only significant
multiple-choice accuracy beat (+0.0135 fusion / +0.0095 veto). A one-point win is only meaningful if
the gold labels are better than ~1 point accurate **on the decision-relevant items**. Hypothesis under
test: *the win is manufactured by label noise.*

**Stage 1 — isolate the decision-relevant items, with assertions.** The script re-implements the
fusion and veto policies **exactly** as the deployed code does (same 5-fold cross-fit isotonic
calibration, same deterministic `i % K == f` folds, no random number generator) and then:

- asserts the reproduced PMC deltas equal the published **+0.0135 / +0.0095**;
- asserts **row-for-row alignment** between `test_2.csv` and the dumps — all four choices and the gold
  letter compared per row, **0 misalignments over 33,430 rows** — before any index mapping is trusted;
- isolates `WIN` = fusion right ∧ always-32B wrong (**1,969**) and `LOSS` = fusion wrong ∧ always-32B
  right (**1,518**), and checks `(n_win − n_loss)/n` equals the delta **[M]**.

**Sampling and stratification.** Seeded (`SEED = 20260729`) draws of **100 wins / 50 losses / 50
agree-and-correct controls**. Each worksheet row carries the image, question, four options, gold
letter, both legs' raw responses **and the PMC-VQA source caption** — because PMC-VQA's gold was
auto-generated *from* the caption, the caption is the annotation's own provenance and is essential to
judging label quality.

**Rubric, precedence, and deliberate conservatism.** Five labels — `GENUINE`, `BAD-GOLD`,
`UNANSWERABLE`, `MULTI-CORRECT`, `UNCLEAR` — with an explicit precedence order
`BAD-GOLD > UNANSWERABLE > MULTI-CORRECT > GENUINE` so that an item with two problems is always
counted once, in the same way, by whoever grades it. Conservatism is written into the rubric and
biases *against* the audit's own thesis: hard-but-well-posed expert calls and awkward-but-answerable
wording count as **GENUINE**, and `UNCLEAR` is **never** counted as a defect.

> **The auditor.** The classifications were produced by **a single LLM auditor (Claude Opus 5) reading
> the images — not a clinician, and with no second rater and no inter-annotator agreement statistic.**
> The rubric explicitly limits the judgement to *label quality*, not diagnosis. The defect rates are
> therefore auditor-dependent in a way no confidence interval captures.

### Why the control group matters

**A defect rate is uninterpretable on its own.** "53% of the wins are defective" sounds fatal. But
PMC-VQA is noisy everywhere: the **agree-and-correct control stratum** — ordinary items both models
get right, drawn by the same seeded procedure and graded by the same rubric — is **28% defective**.
Without that number, 53% has no scale.

More importantly, the control fixes *what question is being asked*. The delta is a **difference**
between wins and losses, so it is destroyed by **asymmetric** error, not by error. The right question
is not "how defective are the wins?" but "are the wins *more* defective than the losses, in the
direction that manufactures the delta?"

**The bias test.** Three two-proportion contrasts, each with a normal-approximation z and a **Fisher
exact** p (Fisher because some counts are small), on Wilson-interval rates **[M]**:

| contrast | rates | difference | z | Fisher p | significant |
|---|---|---:|---:|---:|---|
| wins vs control | 0.53 vs 0.28 | +0.25 | 2.903 | 0.0051 | **yes** |
| losses vs control | 0.60 vs 0.28 | +0.32 | 3.223 | 0.0023 | **yes** |
| **wins vs losses** | 0.53 vs 0.60 | **−0.07** | −0.813 | **0.487** | **no** |

The first two say something real and uncomfortable: **decision-relevant items are much noisier than
typical items** — where the two models disagree, the benchmark is roughly twice as broken. The third
is the actual falsifier for "the win is manufactured by biased annotation error", and it **fails to
reject** — and the sign favours the losses. Mis-keying specifically is symmetric (BAD-GOLD 9% of wins
against 10% of losses).

**Four corrections, not one**, because which correction is "right" is a modelling choice and should be
exposed: **A** discount wins only (explicitly the *wrong* correction here, since the losses are at
least as defective); **B** symmetric drop-defective; **C** re-key BAD-GOLD only; **D** re-score on a
cleaned benchmark. Values in §5.4.3.

Uncertainty is propagated by a **200,000-draw Monte Carlo** with uniform Beta priors on each audited
proportion plus item-level noise of sd 0.00176 taken from the measured paired-bootstrap interval —
audit uncertainty and sampling uncertainty combined rather than reported separately.

**A noise ceiling, computed two ways** because "achievable accuracy" is ambiguous: defective items
scored wrong (`1 − d`) against defective items answered at 4-way chance (`(1−d) + 0.25d`), each under
two treatments of the unsampled strata → **0.63–0.77**. Every system in the project scores 0.54–0.57,
so the benchmark is not saturated — but a one-point margin is being measured where **31–37% of items
cannot support a correctness claim at all**.

**Verdict, and the distinction that matters.** The *arithmetic* survives; the *construct* does not.
46% of the wins sit on items where the gold is wrong or the answer is simply not in the shown image.
The recommendation is therefore not to discard the number but to **rename** it: report it as higher
agreement with PMC-VQA's caption-derived keys, with the 53% defect rate stated alongside, and stop
using PMC-VQA to carry an accuracy claim. Compute-lean is untouched (its PMC cell is −0.0010), so the
compute story is unaffected.

> `src/cascade_methods/finding1_corrected.py` and `pmc_label_noise_audit.py` were read only in part
> (headers, policy and statistics sections) in this pass. The artifacts they emit were verified
> against their documented method, but neither script was re-executed. **[U on re-execution]**

## 4.6 The verifier validity work

**Hypothesis.** The trained open-text verifier was trained on a 70/30 grouped split **of the
evaluation sets themselves**, so its selection gain may be memorisation.

**Measurement 1 — overlap, at two granularities [M].** Question-level: **67.4% (SLAKE) / 71.0%
(PathVQA) / 73.0% (VQA-RAD) / 69.6% (Kvasir)** of scored items were in the verifier's training data;
RadImageNet **0%**. Then a second, harder measurement that most leakage audits skip — **image-level
overlap of the supposedly clean 30%**: **100%** of SLAKE's "unseen" questions use an image that *was*
in training, PathVQA 94.5%, VQA-RAD 64.8%, Kvasir 18.9%. This is why the seen/unseen split can only
*bound* the inflation, not settle it, and why §4.7 was necessary.

**Measurement 2 — in-domain against unseen strata [M].** The same quantity (selection gain over
greedy, best-of-8) is computed on `full`, `seen`, `unseen`, plus a dataset the verifier never saw at
all (RadImageNet, 0% overlap) and one seen by a *different* verifier (Kvasir). Pooled over the 3
reported open cells: full **+0.1041 [+0.0887, +0.1190]** against unseen **+0.0701 [+0.0443, +0.0973]**;
`seen − unseen` **+0.0484 [+0.0160, +0.0804]**, significant → memorisation share **32.6%** (n-weighted
31.0%). Per dataset only PathVQA's seen-minus-unseen is significant.

**Measurement 3 — the image ablation, with six conditions.** The failure mode being tested is a "lazy
verifier" that scores answers from text priors alone. Six conditions separate three distinct
questions: does it use the image *at all* (`no_image`), does it merely respond to image statistics
(`blank_gray`, `blank_black`, `blank_matched` — mean-matched grey), and does it need **the right**
image (`mismatched` — a real but wrong image)?

- **Fidelity gate first:** re-scoring under `real` reproduces the saved dump (n = 2,880, mean absolute
  deviation 5.5e-5, 99.83% within 0.01). Any deviation elsewhere is therefore the condition, not the
  harness.
- **Result [M]** (pooled n = 360): `real` selection 0.6667, candidate AUROC 0.9325; **every** degraded
  condition is significantly below real (−0.042 to −0.058, all intervals excluding 0) with AUROC
  falling to 0.739–0.808. `mismatched` has the worst AUROC → the verifier uses the *specific* image,
  not just the presence of pixels.
- **The honest reading, also recorded:** a no-image verifier still reaches 0.6194 against greedy
  0.6028, so the verifier leans substantially on text priors. The ablation refutes "lazy verifier"; it
  does not establish "image-driven verifier".

**A provenance gate that could have invalidated the ablation.** A latent bug was found in
`src/training_methods/verifier_transfer_eval.py`'s `imgs_for()` — no `slake_open` branch, so the
`else` branch would have returned **PathVQA** images keyed by PathVQA row index. Four independent
checks established it never executed for SLAKE. Full account in §5.7.2. Published numbers affected:
**none**. The point of recording it: a bug that *could* have mattered is documented with the evidence
that it did not, rather than quietly fixed.

## 4.7 The disjoint verifier retrain

**Hypothesis.** If §4.6's inflation estimate is right, a verifier trained on **strictly disjoint**
data should show a materially smaller selection gain. This gates every open-text accuracy number,
because the open arm holds 37.5% of the macro weight.

### Split design, and the alternatives rejected *with numbers*

**Option A — hold out the eval images from the eval sets themselves — was rejected quantitatively, not
by taste.** Filtering the evaluation sets to items whose images are absent from training would discard
**71.2% (SLAKE) / 73.0% (VQA-RAD) / 71.9% (PathVQA)** of the reported eval items **[M]**. That does not
de-contaminate the experiment; it replaces it with a different, smaller benchmark, and destroys
comparability with every published cell.

**Option C (adopted) — hybrid:** train on the datasets' **official train splits** (image-disjoint by
construction from the test splits) plus two out-of-domain pools (Kvasir, RadImageNet), leaving all
2,345 reported eval items **100% intact**. Training pool: 16,621 items over 5,229 images at L1.

**Two strictness levels, and why L1 is the headline.**

- **L1 (headline)** — no eval **image**, no eval **item**; question *templates* may recur with other
  images. That is ordinary generalisation on templated medical VQA, not leakage.
- **L2 (lower bound)** — L1 **and** no eval question **text** at all. L2 was computed and reported, but
  explicitly *not* the estimate, for a measured reason: it starves the in-domain pools (PathVQA drops
  **7,306 of 9,903** training items; SLAKE 2,623 of 2,976), leaving a biased atypical-question
  subsample whose candidate AUROC falls **0.868 → 0.700**. L2 therefore conflates de-contamination
  with distribution shift and *understates* the verifier.

> **This is a judgement call, not a measurement**, and the artifact says so
> (`verifier_disjoint_retrain_2026-07-30.json : verdict.why_L1_is_the_headline`). **A reviewer could
> reasonably insist L2 is the honest headline, under which the open-text selection gain is null
> (−0.0119 [−0.0277, +0.0034]).** Both columns are reported throughout §5.5 and §5.6.

**Disjointness asserted, not assumed [M].** Images are identified by **md5 of the DECODED RGB pixels**
(width × height + raw bytes), not of the file — so a re-encoded or re-compressed copy of the same
image is still caught. Question items are identified as `(dataset family, normalised question text,
image pixel hash)`, because raw row ids are **not comparable across different split files**. Four
assertions run in the split builder and are re-asserted by the trainer: image-hash intersection **0**,
item intersection **0**, question-id intersection **0**, and for L2 question-text intersection **0**.
The artifact also records the diagnostic that shows why naive id-matching would have failed: **751**
raw numeric id coincidences between the PathVQA train and eval files that are *not* the same item.

**Composition matching, so provenance is the only variable [M].** The contaminated verifier's
**10,364** training examples were **894 SLAKE / 4,973 PathVQA / 522 VQA-RAD / 3,975 Kvasir**. The clean
trainer draws **the same per-family counts** from the disjoint pools (RadImageNet available as a
documented top-up pool; shortfall was **zero** at both levels). Without this, "clean against
contaminated" would be confounded with training-set size and domain mix.

**Architectural identity.** Everything except the data is held fixed and copied verbatim from the
contaminated trainer: base model Lingshu-7B, LoRA **r = 16 / α = 32 / dropout 0.05 / bias none** over
the same seven target modules, the same objective (next-token cross-entropy on the single "Yes"/"No"
continuation token), same optimiser (AdamW, lr 1e-4, gradient clip 1.0), batch size 2 × accumulation
8, 1 epoch, seed 0, same prompt and pixel budget, and the same **5,182** optimiser steps at both levels
(107.7 / 106.3 minutes).

> **A precision correction on this point.** Earlier drafts described an "architectural byte-identity
> check". What exists is (a) the trainer's documented verbatim copy of architecture, objective and
> hyperparameters from `run_lora_verifier_open.py`, with a comment noting `target_modules` matches
> the deployed adapter's `adapter_config.json`
> (`src/training_methods/run_lora_verifier_disjoint.py:10-21`), and (b) a check performed for this
> write-up: the two `adapter_config.json` files are **not byte-identical** (md5 `264fb9eb…` against
> `1d467262…`) but **agree on all 27 fields**, differing only in the serialisation *order* of
> `target_modules` (peft writes a set) **[M]**. The byte-identity assertion that *is* in code is a
> different one — the MedEvalKit reasoning prompt string (§4.3.2, assertion B).

**Measurement is exactly paired, by assertion [M].** Candidate answer lists, per-candidate judge
labels, the greedy label and the 32B strong-leg judge labels are asserted **identical item by item**
across arms; only the verifier scores differ (changed on **630/645**, **198/200**, **1,496/1,500**
items). Greedy, self-consistency and oracle are therefore identical by construction, and the run
aborts if they are not. The swap itself is an **exact-path redirect installed on `builtins.open`**, so
not one line of the scoring or aggregation machinery is duplicated or re-implemented.

**The null test.** Before reporting anything, the pipeline is run with `clean == contaminated` and must
reproduce the published cells **exactly**: the open-arm measure reproduces the published cells (SLAKE
0.8155 @ 12.6%, VQA-RAD 0.5850 @ 5.5%, PathVQA 0.4533 @ 0.1%, pooled 0.5642), and the macro re-run
reproduces `macro_average_headline_2026-07-30.json` across **1,224 fields** — accuracy levels, every
delta point estimate **and both interval bounds**, per-cell escalation, every cost ratio — with the
random-number stream replayed in the same order, so agreement is *exact*, not "within Monte-Carlo
noise" **[M]**.

**What would have falsified the contamination concern.** A clean verifier reproducing the contaminated
gain. It did not: pooled selection gain **+0.1041 → +0.0358 [+0.0213, +0.0503]** at L1 (2.90×
inflation), and **−0.0119 [−0.0277, +0.0034]** at L2.

**The diagnostic that explains the mechanism [M].** Contamination did *not* mainly buy ranking ability
— candidate AUROC falls only 0.9433 → 0.8856. What collapses is **oracle conversion**, the share of the
greedy→oracle headroom actually captured: **0.589 → 0.203 → −0.068**. And the efficiency claim is
damaged more than the accuracy claim: escalation needed to hold the same parity target goes
**3.97% → 26.9%** sample-weighted (**48.7%** macro).

> **A self-referential caveat.** The open arm's iso-accuracy *target* is itself derived from the
> best-of-8-plus-verifier arm, so a weaker verifier lowers the target as well as the delivered
> accuracy. The clean-arm accuracy numbers are partly self-referential.

## 4.8 The weighting change — sample-weighted against 8-cell macro

**What changed.** The primary average moved from **sample-weighted** (every item 1/N) to **macro over
the 8 reporting cells** (1/8 each), with a 5-benchmark macro (closed and open averaged into one number
first) as a secondary robustness check. Effect on the weights **[D]**: PMC-VQA **79.2% → 12.5%**; the
open-text arm **5.6% of items → 37.5% of weight**; the closed arm **94.4% → 62.5%**.

**What each average answers — and why they are not interchangeable.**

- **Sample-weighted** answers *"on traffic that looks like this suite, what do I get and what do I
  pay?"* Cost is **additive per query**, so the pooled cost is a genuine deployment quantity.
- **Macro** answers *"does this result generalise across task types, or is it one benchmark's result
  wearing the suite's name?"* Macro **cost** is therefore a generalisation test, **not** a bill.

Standing rule: **report accuracy on macro and BOTH cost numbers, each labelled — never pair a macro
accuracy with a sample-weighted cost.**

**The mechanism of the reversal, isolated [M].** Escalation is wildly heterogeneous — PMC-VQA
**8.45%**, SLAKE-closed 20.45%, PathVQA-closed 45.72%, VQA-RAD-closed 56.97%, MedXpert **89.60%**,
SLAKE-open 15.81%, VQA-RAD-open 12.50%, PathVQA-open 35.67% — and the cheapest cell held 79.2% of the
old average. Equal weighting raises the multiple-choice escalation rate **16.22% → 44.24%** (all 8
cells: 16.89% → 35.65%). Meanwhile the open cells cost the method **7.6–12.6 FLOP-eq** against the
baseline's flat 4.57, while PMC costs 1.386. That is the entire story.

**Controls on the new convention.** The strongest reviewer objection to the project's own primary
metric is written into the artifact
(`macro_average_headline_2026-07-30.json : caveats[1]`): **the 8 cells are not 8 independent
datasets** — SLAKE, VQA-RAD and PathVQA each contribute a closed *and* an open cell **from the same
images**, so equal-weighting triple-counts three source datasets and hands the open-text *format*
37.5% of the weight. That is precisely why the 5-benchmark version is computed. And the MMMU
exclusion's justification **inverts**: under sample weighting MMMU is 0.35% of items (headline moves
−0.0005), under macro it would carry **1/9 = 11.1%**, so the exclusion must be re-argued on
contamination grounds alone, with its size stated (§5.7.1).

**A disclosed mis-specification** (`caveats[8]`). τ, the adaptive-sampling parameter and the
veto/deferral thresholds were all calibrated to hold **pooled** accuracy at parity. Re-basing the
*report* on macro does not change the objective the method was *tuned* for, so §5.6's numbers are
"a pooled-tuned method scored on a macro metric". A macro-objective refit is a separate, CPU-only
experiment that **has not been done**. Every value is real; this is mis-specification, not
fabrication.

*(A companion caveat, `caveats[7]`, records that the open arm's parallel latency is an assumption and
gives the sequential range of 1.9–3.1 s.)*

## 4.9 Statistical practice throughout

### Paired bootstrap

Both systems answer the same items, so the estimator is the **mean per-item difference**, and the
resampling unit is the **item**, not the system. Implementation: **10,000 replicates**, percentile
interval at the 2.5th/97.5th percentiles, "significant" defined as the interval excluding 0, with
**common random numbers** across weightings so that the sample-weighted and macro numbers are computed
on the *same* replicate stream and their difference is not itself Monte-Carlo noise.

*Why paired:* medical VQA items differ enormously in difficulty. An unpaired comparison of two
systems' accuracies carries all of that item-to-item variance; the paired difference cancels it, which
is why deltas of +0.002 can be resolved on n = 33,430.

*One implementation detail worth knowing:* items are resampled **exactly** by drawing multinomial
counts over the unique per-item **outcome patterns** across the 7 systems. Items with the same pattern
are exchangeable, so this is distributionally identical to gathering resampled rows, and it makes a
33,430-item cell cost the same as a 200-item one (PMC-VQA has only **28** unique patterns). This
shortcut was **validated against literal item resampling** on the largest and one of the smallest
cells: PMC-VQA [−0.00586, +0.00389] against [−0.00577, +0.00383]; VQA-RAD-open [−0.06, +0.05] against
[−0.06, +0.05] **[M]**.

**McNemar's exact two-sided test** on discordant pairs is reported alongside the interval wherever
per-cell significance matters. It is the natural companion to a paired binary comparison: only the
items where the two systems disagree carry information.

### Cross-fit held-out calibration

Every free parameter is fit on data it is not scored on. Folds are deterministic (`i % K == f`,
K = 5). Inside each fold, τ is chosen by `pick_tau_isocost`
(`src/cascade_methods/integrated_method.py:195-205`) as the **minimum-escalation** τ whose
**training-fold** cascade accuracy is at least the strong leg's **training-fold** accuracy; it is then
applied unchanged to the held-out fold. The adaptive-sampling controller is **fully nested**: its
isotonic calibration *and* its threshold are picked on the training fold, frozen, and applied held
out. The certified veto and the deferral rule are cross-fit the same way.

Two consequences are stated rather than hidden. First, **calibration variance is not in any
interval** — the bootstrap resamples questions only. Second, because τ is refit inside every fold,
**no single deployable threshold is materialised anywhere**, so the reported numbers describe a
per-benchmark-calibrated procedure, not a shipped constant. (The earlier MedVLThinker work *did*
freeze one τ = 0.426 and transfer it.)

### Wilson intervals

For a proportion `k/n`, the textbook interval `p ± 1.96·√(p(1−p)/n)` misbehaves exactly where this
project needs it: with small `n`, or with `p` near 0 or 1, it can run outside [0, 1] and its real
coverage collapses (at `k = 0` it has zero width). The **Wilson** interval inverts the score test
instead and always lies inside [0, 1]. Used two ways:

- **Two-sided, z = 1.959964**, for every audit classification rate — `n` = 50–100 with counts as low as
  0 and 1 (e.g. control BAD-GOLD 1/50 → [0.0035, 0.105]; losses UNCLEAR 0/50 → [0, 0.0713]).
- **One-sided, z = 1.645, with `n ≥ 30` required**, as an actual *decision rule*
  (`src/cascade_methods/beat32b_more.py:54` and `:98-121`): the certified veto keeps the cheap model's
  answer inside a confidence bin only if the Wilson **lower bound** on the 7B's training-fold precision
  is at least the 32B's training-fold accuracy in that bin.

The criticism of that second use is recorded with it: it compares a **lower bound to a point
estimate**, ignores the strong model's own sampling error, ignores that both are measured **on the
same items** (a paired test is correct), and runs **5 bins × 5 folds = 25 uncontrolled
certifications** — so "never worse by construction" is not a valid claim as stated. The prescribed fix
(one-sided paired test per bin, Holm-corrected) **has not been run**.

### What the intervals do *not* capture, and the substitute

Every macro interval here resamples **items within cells**, treating the 8 cells as **fixed**. It
therefore covers within-dataset sampling noise and **not dataset-selection noise**. This is a
deliberate refusal: with 5–8 units, resampling *datasets* is hopelessly unstable — one bootstrap draw
can duplicate PathVQA-open three times.

The honest substitute is **leave-one-cell-out**: recompute the macro delta with each cell dropped in
turn and report the range **[M]**:

| comparison | macro Δ | leave-one-out range | load-bearing cell | cell holding it back |
|---|---:|---|---|---|
| accuracy-max vs 32B-reasoning (8 cells) | +0.0720 | [+0.0318, +0.0830] | PathVQA-open | SLAKE-closed |
| compute-lean vs 32B-reasoning (8 cells) | +0.0626 | [+0.0225, +0.0732] | PathVQA-open | SLAKE-closed |
| accuracy-max vs 32B-direct (8 cells) | +0.0128 | [+0.0023, +0.0146] | PathVQA-open | SLAKE-closed |
| compute-lean vs 32B-direct (multiple choice only) | −0.0070 | [−0.0085, −0.0038] | PMC-VQA | VQA-RAD-closed |

The macro interval is **2–4× wider** than the pooled one for the same comparison (accuracy-max against
32B-direct: [+0.0086, +0.0127] pooled against [+0.0056, +0.0200] macro). That is a feature of the
convention, not a defect — the small cells now count as much as the big one, and their genuine
uncertainty shows up instead of being averaged away.

### Multiplicity, and where it is not controlled

Stated plainly because a reviewer will find it: the accuracy-max policy router runs **6 cells × 3
policies = 18 tests with no multiplicity control**, using intervals computed on the same held-out data
that is then reported; only 2 cells deviate from the default, one of which is contaminated (MMMU) and
one of which carries 79% of the sample-weighted pool (PMC-VQA). The certified veto adds 25
uncontrolled certifications. **Neither a Holm-corrected policy-selection row nor a single-frozen-policy
row has been produced**, so this document cannot report what survives multiplicity correction.

### Measured against estimated against asserted

The project has already mislabelled an estimate as a measurement once and corrected it (§6.1.1), so the
distinction is tracked explicitly:

- **Measured** — per-sample correctness on all 42,224 items; the judged 32B-reasoning open-text vectors
  (which *replaced* an n = 200 estimate and moved the headline from +0.0117 to +0.0154); the decode
  rate and energy intercept; generated-token counts; trace rates; overlap and disjointness hashes.
- **Measured but weak, and labelled as such** — the reasoning latency/energy constant is a mean over
  **n = 15** whose **median is 12,896.2 ms** (a 23% divergence indicating a heavy tail), measured on one
  benchmark's images at 98.3 tokens and transferred to cells whose real traces are 320 tokens; the
  direct-mode reference is n = 25.
- **Asserted, not measured** — the best-of-N parallel latency (347 + 175 = 522 ms) is an assertion
  inconsistent with its own 8×-billed 568.8 J energy figure (implying ~1,088 W against ~132 W measured
  at batch 1 and a 400 W card); **no batch-8 measurement exists in the repository**, so every open-arm
  parallel-latency number in this document should be treated as unverified. The open arm's parallel
  latency likewise assumes overlappable draws, which was never measured, so sequential latency is
  reported alongside it everywhere.
- **Under-derived** — the 32B/7B compute ratio **4.57** appears only as a hard-coded literal; an older
  document implies 4.34, and no file derives it.

### Two judging caveats that bound everything free-text

The LLM-judge protocol (`src/labeling/run_judge.py`) was **not** independently re-verified in this
pass, nor was the separate Claude-as-judge validation referenced for SLAKE and VQA-RAD. The PMC audit
and the open-text scoring both rest on judges whose own validation is documented but was not re-run
here. Critically, **that judge cross-validation covers SLAKE and VQA-RAD only — not PathVQA**, which
is the load-bearing open cell of every vs-reasoning claim. **[U]**

## 4.10 The verification gates, as a checklist

Every result above had to clear some subset of these before it was believed. They are listed together
because they are the transferable part of the methodology.

| gate | what it catches | example |
|---|---|---|
| **Reproduce the published number first** | wiring errors masquerading as findings | all 35 cross-family cells reproduce `master_data.csv` exactly |
| **Null test (change nothing, expect nothing)** | silent pipeline drift | clean == contaminated reproduces 1,224 fields exactly |
| **Pairing / id-identity assertion** | comparing different item sets | `ids_identical_across_arms = true` in all 6 MedEvalKit cells |
| **Token audit of the reasoning arm** | an arm that never did the thing under test | Lingshu "native think" = 3.0 tokens → 7 cells withdrawn |
| **`parse_ok` / extraction measurement** | prompt effects leaking through the grader | ≥ 0.9663 everywhere; adversarial bound flips no count |
| **Fidelity check before an ablation** | attributing harness noise to the manipulation | ablation `real` reproduces the dump to 5.5e-5 |
| **Disjointness on decoded pixels** | re-encoded duplicate images | image-hash intersection = 0 over 5,229 against 528 images |
| **Composition matching** | confounding provenance with data volume/mix | identical 894/4,973/522/3,975 family quotas |
| **Algebraic identity check** | a decomposition that is narrative, not arithmetic | 2×2 residual −0.0000 pooled |
| **Sensitivity band, not a point** | a conclusion resting on one disputed constant | five cost models M1–M5 |
| **Adversarial bound** | small residual channels argued away rather than bounded | credit every unparsed answer as correct |
| **Control stratum** | an uninterpretable rate | 28% baseline defect rate against 53% on wins |
| **Leave-one-cell-out** | a macro average leaning on one dataset | +0.0720 → +0.0318 without PathVQA-open |
---

# 5. Results

Findings are grouped by *claim*, not by date. Each subsection states what was measured, the number
with its 95% interval and sample size, and the verdict on the original claim: **SURVIVES**,
**REDUCED**, or **RETRACTED**.

## 5.1 Reasoning versus perception

**What was measured.** For each of 5 medical VLM families × 4 perception benchmarks (PMC-VQA, SLAKE,
VQA-RAD, PathVQA) = 20 cells, accuracy with a chain-of-thought prompt minus accuracy answering
directly, on the same items. The originally published table paired arms that were **not
prompt-matched** (and, for MedVLThinker, not **resolution**-matched). `finding1_corrected.py` →
`finding1_corrected_2026-07-29.json` re-derived every cell from the best-matched dumps already on disk.
Multiple choice is graded by exact single-letter equality, so an unmatched prompt can only act through
answer-*extraction* failure — measured at 0–3.4% per cell.

### 5.1.1 The corrected cross-family table

Δ = reasoning − direct. **Bold** = 95% interval excludes zero. Policy P1 (best-matched arm on disk).
The PMC column is the human-verified `test_clean.csv`, n = 2,000 per cell — **not** the MedEvalKit
`test_2.csv`. All **[M]**.

| family | PMC-VQA (n = 2,000) | SLAKE (n = 416) | VQA-RAD (n = 272) | PathVQA (n = 3,362) |
|---|---:|---:|---:|---:|
| MedVLThinker-32B | −0.0075 [−0.0275, +0.0120] | **−0.1274** [−0.1659, −0.0913] | **−0.0846** [−0.1360, −0.0368] | +0.0012 [−0.0155, +0.0173] |
| Lingshu-32B | **−0.0425** [−0.0625, −0.0220] | **−0.0649** [−0.0986, −0.0312] | **−0.0919** [−0.1471, −0.0368] | **−0.1017** [−0.1169, −0.0872] |
| QoQ-Med-VL-32B | **−0.0585** [−0.0795, −0.0375] | −0.0144 [−0.0553, +0.0240] | **−0.0662** [−0.1176, −0.0184] | **−0.0523** [−0.0681, −0.0366] |
| Chiron-o1-8B | **−0.0680** [−0.0895, −0.0470] | **−0.1010** [−0.1466, −0.0577] | **−0.1103** [−0.1728, −0.0515] | **−0.0654** [−0.0842, −0.0467] |
| MedGemma-27B | −0.0135 [−0.0365, +0.0085] | +0.0144 [−0.0264, +0.0553] | **−0.0735** [−0.1250, −0.0221] | **+0.0413** [+0.0220, +0.0607] |

**Counts [M].** 17 of 20 cells strictly negative; 19 of 20 **no worse than +0.02** (a one-sided band —
see below); **14 of 20** with intervals excluding zero; 1 of 20 significantly positive. Pooled
perception Δ = **−0.0401 [−0.0456, −0.0347]** over **30,250** paired samples.

> **The band is one-sided.** `_meta.noise_band` reads *"'within noise' = delta ≤ +0.02"*. Written as
> "±0.02" it would assert something four times stronger than the data supports: under P1 only
> **5 of 20** cells have |delta| ≤ 0.02 (`n_abs_delta_le_band = 5`).

**Per-family pooled Δ (n = 6,050 each) [M]:** MedVLThinker **−0.0144** [−0.0261, −0.0030]; Lingshu
**−0.0792** [−0.0902, −0.0681]; QoQ **−0.0524** [−0.0640, −0.0407]; Chiron **−0.0707** [−0.0840,
−0.0577]; MedGemma **+0.0162** [+0.0028, +0.0298].

### 5.1.2 The three robustness policies

The count does not depend on which reasonable correction is applied **[M]**:

| policy | strictly negative | ≤ +0.02 | CI-significant negative | pooled Δ (n = 30,250) |
|---|---:|---:|---:|---|
| P0 — as published (the arms behind the 15/20 claim) | 15/20 | 19/20 | 12/20 | −0.0252 [−0.0304, −0.0199] |
| **P1 — best-matched arm on disk (primary)** | **17/20** | 19/20 | 14/20 | **−0.0401 [−0.0456, −0.0347]** |
| P2 — strict: resolution **and** answer-format matched | 17/20 | 19/20 | 13/20 | **−0.0408** [−0.0462, −0.0353] |
| P3 — as P2, MedVLThinker matched at fullres instead of cap320 | 17/20 | 19/20 | 13/20 | −0.0405 [−0.0459, −0.0351] |

**The as-published 15/20 is the outlier, not the ceiling** — every better-matched pairing makes the
effect *stronger*, and roughly *doubles* the pooled magnitude (−0.0252 → −0.0401 … −0.0408).

> **The July-8 record, explicitly superseded.** Everywhere else in this repository the phrasing "15/20
> strictly negative, 19/20 within noise, MedGemma PathVQA +0.040" appears, it is the **2026-07-08
> record and is superseded by this table.** The retrospective's §2.12 carries its own "Superseded
> 2026-07-29" notice. Any document still quoting 15/20 — including the 2026-07-27 professor deck — is
> stale (§11.3).

### 5.1.3 The fully-matched subset (nothing left to correct)

Cells whose two arms differ **only** by the reasoning instruction — same (absent) system message, same
image budget, answer-format constraint retained in both arms **[M]**:

| subset | cells | strictly negative | pooled Δ | n |
|---|---:|---:|---|---:|
| Medical (Chiron-o1-8B + MedGemma-27B, shared peer reasoning instruction) | 8 | **6/8** (5 CI-significant) | **−0.0273 [−0.0367, −0.0176]** | 12,100 |
| Non-medical peer architectures (InternVL2.5-8B + Phi-3.5-Vision) | 8 | **7/8** (4 CI-significant) | InternVL2.5-8B −0.0076 [−0.0208, +0.0056]; Phi-3.5-Vision **−0.0187 [−0.0336, −0.0036]** | 6,050 each |

Peer dumps: `ckpts/acc_gen/{internvl25_8b_think, phi35v_think}` — 4 files / 6,050 rows each, perception
only **[M]**.

### 5.1.4 The two cells that flipped sign

Both are PMC-VQA `test_clean.csv`, n = 2,000, and both flipped from a claimed perception *win* for
reasoning to a loss **[M]**:

| cell | published Δ | corrected Δ | cause of the flip |
|---|---:|---|---|
| MedVLThinker-32B PMC-VQA | +0.0055 | **−0.0075** [−0.0275, +0.0120], McNemar p = 0.481 | reasoning arm re-paired at cap320 against cap320 |
| Lingshu-32B PMC-VQA | +0.0115 | **−0.0425** [−0.0625, −0.0220], p = 5.6e−5 | published reasoning arm never reasoned (3.0 tokens); replaced by a genuinely reasoning arm (198.0 tokens) |

### 5.1.5 The one genuine exception

**MedGemma-27B on PathVQA: +0.0413 [+0.0220, +0.0607], p = 0.0000, n = 3,362, on a fully matched
pair [M].** It survived removing the reasoning-only persona that produced the published +0.0399, so it
is a real exception rather than a prompt artifact — and it is the **only** one in the corrected table.

### 5.1.6 Verdict

**SURVIVES, and strengthened.** "Chain-of-thought does not pay for itself on perception-style medical
VQA" holds at 17/20 with a pooled effect roughly twice the published size. Two precision caveats must
travel with it: (i) 17/20 is a **count of signs**, not a measurement — per-cell n runs 170–3,362, and
at n = 170 a 95% interval is roughly ±0.07, so near-zero cells could flip on resampling alone; always
quote the count with the pooled Δ and the 14/20 significant subcount. (ii) An adversarial correction
charging every extraction failure against the finding changes **no** count
(`n_strictly_negative_after_parse_adversarial` = 17 under P1/P2/P3) **[M]**.

## 5.2 Reasoning on reasoning-heavy benchmarks

**What was measured.** Two independent measurements. (a) Internal harness: 5 families × 3 reasoning
cells (MMMU, MedXpert-Reasoning, MedXpert-Understanding) = 15 cells, re-derived under the same four
policies. (b) The decisive experiment — a **matched-prompt re-run on MedEvalKit**
(`src/labeling/medeval_matched_prompt.py`; `MedEvalKit/` left byte-identical to upstream) producing
**6/6 (family × benchmark) matched-direct cells**, decomposed into 9 sub-cells, three arms each, paired
on item id:

- **rung 1 — published-direct:** "Answer with the option's letter … directly" (bare letter)
- **rung 2 — matched-direct:** "Put the final answer letter … in one `\boxed{}`" (**no reasoning
  trigger**)
- **rung 3 — reason:** "First reason step by step … then put the final answer letter … in one
  `\boxed{}`"

The arms differ by *exactly* the leading reasoning clause. Source:
`medeval_matched_direct_2026-07-29.json`.

### 5.2.1 The full 9-sub-cell decomposition: format against trigger

All **[M]**. "published Δ" = rung3 − rung1 (what was previously reported as a *reasoning* gain).
"format Δ" = rung2 − rung1 (both trigger-free). "trigger Δ" = rung3 − rung2 (the marginal value of the
explicit reasoning instruction).

| family | sub-cell | n | published Δ | **format** Δ (rung2 − rung1) | **trigger** Δ (rung3 − rung2) |
|---|---|---:|---:|---|---|
| Lingshu-32B | MMMU-MCQonly | 145 | +0.0276 | −0.0138 [−0.0483, +0.0207] | +0.0414 [−0.0345, +0.1172] n.s. |
| Lingshu-32B | MedXpert-Reasoning | 1,446 | −0.0035 | −0.0076 [−0.0180, +0.0021] | +0.0041 [−0.0207, +0.0290] n.s. |
| Lingshu-32B | MedXpert-Understanding | 554 | +0.0000 | −0.0018 [−0.0144, +0.0108] | +0.0018 [−0.0379, +0.0415] n.s. |
| MedVLThinker-32B | MMMU-MCQonly | 145 | +0.1034 | +0.0621 [−0.0071, +0.1310] | +0.0414 [−0.0138, +0.0966] n.s. |
| MedVLThinker-32B | MedXpert-Reasoning | 1,446 | +0.0463 | **+0.0456 [+0.0194, +0.0719] SIG** | +0.0007 [−0.0221, +0.0228] n.s. |
| MedVLThinker-32B | MedXpert-Understanding | 554 | +0.0415 | **+0.0433 [+0.0054, +0.0830] SIG** | −0.0018 [−0.0361, +0.0325] n.s. |
| InternVL3-38B | MMMU-MCQonly | 145 | +0.1241 | **+0.0897 [+0.0207, +0.1586] SIG** | +0.0345 [−0.0138, +0.0897] n.s. |
| InternVL3-38B | MedXpert-Reasoning | 1,446 | +0.0353 | +0.0221 [0.0000, +0.0436] | +0.0131 [−0.0090, +0.0353] n.s. |
| InternVL3-38B | MedXpert-Understanding | 554 | +0.0199 | +0.0090 [−0.0271, +0.0451] | +0.0108 [−0.0217, +0.0451] n.s. |

**Result: 0/9 explicit-reasoning-trigger effects are CI-significant** (8/9 point-positive; mean delta
shift from matching **−0.0276**). **3/9 answer-format effects are.** On the new `direct_matched` arm,
`parse_ok` is **exactly 1.0000 in 8 of the 9 primary sub-cells** with a minimum of **0.9986**
(InternVL3-38B, MedXpert-Reasoning) **[M]**, so none of this is an extraction artifact.

### 5.2.2 The mechanism: `\boxed{}` is itself a reasoning trigger

Mean generated tokens in the **matched-direct** arm, which contains **no reasoning instruction at
all**, over the 9 primary sub-cells (n = 145 / 1,446 / 554) **[M]**:

| family | matched-direct mean generated tokens | fraction of items producing a trace |
|---|---|---|
| MedVLThinker-32B | 430.8 – 580.1 | 98.9 – 100% |
| InternVL3-38B | 193.1 – 288.9 | 93.5 – 95.2% |
| Lingshu-32B | 3.0 – 4.4 | 0% |

Asking for the answer in `\boxed{}` alone makes the two reasoning-tuned families reason. For
MedVLThinker and InternVL3 the "matched-direct" arm is therefore itself **contaminated** as a
no-reasoning control: its trigger Δ measures the *marginal* value of an explicit instruction **on top
of** format-induced reasoning, not reasoning against none. A clean reasoning/no-reasoning contrast is
**not obtainable** in these families, because the only prompt that suppresses the trace also changes
the answer format.

### 5.2.3 The honest substitute: the monotone ladder

Since the clean contrast is unobtainable, the ladder is recorded instead **[M]**:

| family | MMMU-MCQonly (n = 145) rung1 → rung2 → rung3 | accuracy | mean generated tokens |
|---|---|---|---|
| MedVLThinker-32B | bare → `\boxed{}` → trigger + `\boxed{}` | 0.6345 → 0.6966 → **0.7379** | 2.03 → 430.81 → 580.34 |
| InternVL3-38B | same | 0.6345 → 0.7241 → **0.7586** | 2.00 → 288.86 → 377.63 |
| Lingshu-32B | same | 0.6552 → 0.6414 → 0.6828 | 2.61 → 4.43 → 284.65 |

The first two families are monotone in both accuracy and tokens; Lingshu is flat throughout.

### 5.2.4 The internal-harness reasoning half, corrected

Across policies **[M]**: 15 cells, 10–12 point-positive, **3–4 CI-significant**, 0–1 significantly
negative (P1: 12 positive, 4 significant, 1 significantly negative). Per family under P1:
MedVLThinker-32B 3/3 significant and *improved* by matching (MMMU +0.0647 → **+0.0882** [+0.0235,
+0.1529]; MedXpert-Reasoning **+0.0491**; MedXpert-Understanding **+0.0884**); MedGemma-27B 3/3
positive, 1/3 significant (MedXpert-Understanding **+0.0830** [+0.0397, +0.1264]; its MMMU cell flips
−0.0118 → **+0.0353** once the reasoning-only persona is removed); Chiron-o1-8B directionally positive
3/3 but significant nowhere.

**Note the internal "4/15 significant" is a trigger-plus-format count** and must not be read as
"reasoning helps in 4 cells".

### 5.2.5 Withdrawals — which families are no longer evidence, and why

| withdrawn | why | what replaces it |
|---|---|---|
| **All 7 Lingshu-32B published cells (4 perception + 3 reasoning), both directions** | The published "native think" prompt (`runners/run_native_think.sh:7`) is an answer-**format** string with no reasoning trigger; measured **3.0 against 3.0–3.3** generated tokens — no chain of thought ever occurred **[M]** | Repaired genuinely-reasoning arm (150–259 tokens): perception 4/4 strictly negative, all intervals excluding zero, pooled **−0.0866 [−0.0972, −0.0757]**; reasoning **nothing** (MMMU +0.0000, MedXpert-R +0.0048, MedXpert-U +0.0271, none significant) **[M]** |
| **QoQ-Med-VL-32B as reasoning-side evidence** | Its headline MMMU gain is a prompt artifact: **+0.0706 → +0.0118** [−0.0588, +0.0824] (n = 170) matched, **+0.0000** fully matched. It never had a significant MedXpert gain, and MedXpert-Understanding is significantly **negative** (−0.0433, p = 0.022) **[M]** | nothing — the family supports no reasoning claim |
| **The phrase "5 families" on the reasoning half** | Only 2 of 5 families (MedVLThinker, MedGemma) have any CI-significant reasoning gain on matched arms | the perception half keeps all 5 |
| **Pre-edit MedEvalKit `eval_results_*_think` dumps** | 2.6–3.2 generated tokens — the upstream "reason" prompt carried no reasoning trigger **[M]** | post-edit `*_reason` dumps (275/561/368 tokens), which reason but are format-unmatched |

A related standalone datum: on the fully format-matched Lingshu-32B MedXpert arm (n = 2,000), reasoning
0.3040 against matched-direct 0.3005, **Δ +0.0035 [−0.0185, +0.0250] n.s.**, at **320.33 against 3.05**
generated tokens **[M]** — 100× the tokens bought nothing, and that reading is now unconfounded (the
format channel here is worth −0.006).

### 5.2.6 Verdict

**RETRACTED as stated; a weaker form is kept.** "A reasoning *instruction* improves accuracy on
reasoning-heavy benchmarks" is dropped — 0/9 matched trigger effects reach significance. **Kept:**
getting a reasoning-tuned model to *emit a trace* raises accuracy substantially (MedVLThinker MMMU
+0.103, MedXpert-R +0.046; InternVL3 MMMU +0.124), **but the operative lever is the answer FORMAT, not
the reasoning instruction**. Lingshu-32B must not be cited as reasoning evidence at all. The cascade's
gated-reasoning tier keeps its full value — the rung1 → rung3 *total* is what such a tier delivers to a
user; only the attribution changes.

**Standing rule: any future reasoning-versus-direct pair must be format-matched AND token-audited. A
"direct" arm that emits hundreds of tokens is not a direct arm.**

## 5.3 The open-text matched-prompt result

**What was measured.** Five arms of Lingshu-32B over the same 2,345 open items, same judge, same greedy
decoding, same evaluated indices, forming a 2×2 in {output convention} × {reasoning}. Source:
`matched_prompt_reasoning_2026-07-29.json`.

> **This experiment is Lingshu-32B only.** No matched open-text arm exists for any other family, so
> nothing in §5.3 generalises across families or is cross-checked by a second architecture.

### 5.3.1 The 2×2 (pooled, n = 2,345) [M]

| arm | output convention | reasoning | accuracy |
|---|---|---|---:|
| `direct` (the headline's direct arm) | styled (persona + "short, specific phrase / Do not explain") | off | 0.5168 |
| `direct_unstyled` | unstyled | off | 0.5186 |
| `reason_unmatched` (the headline's reasoning arm) | unstyled | on | 0.3028 |
| `reason_matched_A` | styled | on (partial) | 0.4235 |
| `reason_matched_B` (decisive) | styled | on (partial) | 0.4192 |

| contrast | Δ | 95% interval | significant |
|---|---:|---|---|
| **Reasoning effect at fixed unstyled convention** — *the clean contrast* | **−0.2158** | [−0.2354, −0.1962] | **yes** |
| **Output-convention effect with reasoning off** — *the confound* | **−0.0017** | [−0.0111, +0.0077] | no |
| Original unmatched gap | −0.2141 | [−0.2341, −0.1945] | yes |
| Reasoning effect at styled convention (arm B − direct) | −0.0977 | [−0.1143, −0.0814] | yes |

Identity check: original gap = reasoning-at-unstyled − convention-effect; left-hand side −0.2141,
right-hand side −0.2141, residual **−0.0000** **[M]**.

### 5.3.2 Attribution

**Share of the original gap attributable to reasoning: 1.0079 (~100%). Share attributable to the
prompt confound: −0.0079 (~0%) [D].** The negative sign means the unstyled wording was, if anything,
marginally *better* for the direct arm. The confound is real as a description of the two arms and
contributes essentially nothing to the measured gap.

### 5.3.3 Per dataset [M]

| dataset | n | clean reasoning effect (unstyled) | 95% interval | significant | share attributable to reasoning | verdict |
|---|---:|---:|---|---|---:|---|
| SLAKE-open | 645 | −0.1349 | [−0.1674, −0.1039] | yes | 0.967 | SURVIVES |
| VQA-RAD-open | 200 | −0.0600 | [−0.1250, +0.0050] | **no** | 1.091 | COLLAPSES (under-powered) |
| PathVQA-open | 1,500 | −0.2713 | [−0.2973, −0.2460] | yes | 1.015 | SURVIVES PARTIALLY |

### 5.3.4 The dilution analysis — why the naive matched arms looked better

Arms A and B score far above the unmatched reasoning arm (0.4235 / 0.4192 against 0.3028), which
naively reads as "most of the gap was the prompt". **That reading is wrong.** Telling this model "Do
not explain" partly *suppresses the very behaviour under test*, so those arms are a mixture of
reasoning and direct answering. Splitting arm B by whether a `<think>` trace actually fired **[M]**:

| dataset | trace-fired share | accuracy (arm B, trace-fired) | accuracy (direct, same items) | Δ | accuracy (arm B, no trace) | accuracy (direct, same items) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SLAKE-open | 0.6698 (432/645) | 0.7616 | 0.8750 | **−0.1134** | 0.6714 | 0.7042 | −0.0329 |
| VQA-RAD-open | 0.3050 (61/200) | 0.7213 | 0.8033 | **−0.0820** | 0.4820 | 0.5108 | −0.0288 |
| PathVQA-open | 0.7107 (1,066/1,500) | 0.1154 | 0.2730 | **−0.1576** | 0.6382 | 0.6290 | +0.0092 |

**The entire deficit sits on the items where reasoning actually happened.** The apparent "prompt fix"
is dilution. Consequently the naive prompt-share statistics for arms A and B (pooled 0.544–0.564) are
an **upper bound** on the prompt's contribution, not an estimate; the clean 2×2 puts the real
contribution at ~0.

A second diagnostic confirms the mechanism is reasoning, not wording: on PathVQA's degenerate taxonomy
family the *unstyled* direct arm — which has none of the persona or style wording — emits
dataset-taxonomy tokens at 0.742 against the styled direct arm's 0.755 and scores the same **[M]**.
What collapses the taxonomy-token rate (to 0.151) is reasoning itself.

### 5.3.5 Headline consequence [D]

Replacing only the three open cells' always-32B-reasoning vector with the matched arm B:

| pool | n | 32B-reasoning (unmatched) | 32B-reasoning (matched B) | accuracy-max Δ vs reasoning: unmatched → matched | shift |
|---|---:|---:|---:|---|---:|
| Variant B (8 cells) | 42,224 | 0.5591 | 0.5656 | +0.0245 → **+0.0180** | −0.0065 |
| Full suite (9 cells) | 42,374 | 0.5594 | 0.5659 | +0.0249 → **+0.0185** | −0.0064 |
| Open only (3 cells) | 2,345 | 0.3028 | 0.4192 | +0.2699 → **+0.1535** | −0.1164 |

### 5.3.6 Verdict

**SURVIVES on the clean contrast; the magnitude against a matched reasoning arm is materially
smaller.** "Reasoning hurts perception on open-text medical VQA" survives matched prompts at the pooled
level (−0.2158 clean, ~100% attributable to reasoning), but the *headline's* vs-reasoning open-text
margin shrinks by −0.1164 when the baseline arm is style-matched.

Residual caveats: the clean contrast holds the convention fixed at the **unstyled** wording (the
symmetric test at the styled wording is not cleanly runnable because that wording suppresses the
trace); VQA-RAD-open at n = 200 is under-powered and inconclusive in every arm; arm B changes the
persona sentence's *position* (not content) relative to the direct prompt.

> **This experiment resolves what was previously an open item.** Earlier documents describe the
> open-text matched re-run as "in flight" or "outstanding". Both arms A and B are complete
> (`arms_missing = {}`), the artifact exists on disk, and this section reports its result. The
> retrospective's "in flight" language is stale.

## 5.4 PMC-VQA validity

**What was measured.** 200 PMC-VQA `test_2.csv` items opened image-by-image and classified against a
fixed rubric, with the question, four options, gold letter, both models' raw answers and the PMC-VQA
source caption in view. Three strata: **wins** (n = 100, items where fusion beats always-32B),
**losses** (n = 50), **control** (n = 50, both models agree and are correct). Source:
`pmc_label_noise_audit_2026-07-29.json`. **Auditor: a single LLM (Claude Opus 5), not a radiologist, no
second rater, no inter-annotator agreement statistic; label quality was judged, not diagnoses.**

### 5.4.1 Defect rates with intervals [M]

| stratum | n | GENUINE | BAD-GOLD | UNANSWERABLE | MULTI-CORRECT | UNCLEAR | **defective** | 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| wins | 100 | 45 | 9 | 37 | 7 | 2 | **53%** | [0.4329, 0.6249] |
| losses | 50 | 20 | 5 | 22 | 3 | 0 | **60%** | [0.4618, 0.7239] |
| control (agree-and-correct) | 50 | 36 | 1 | 11 | 2 | 0 | **28%** | [0.1747, 0.4167] |

Upper bounds counting UNCLEAR as defective: wins 55% [0.4524, 0.6439]; losses and control unchanged.

### 5.4.2 The three bias tests [M]

| test | difference | z | p (two-sided z) | Fisher p | significant |
|---|---:|---:|---:|---:|---|
| wins vs control | +0.25 | 2.903 | 0.00369 | 0.0051 | **yes** |
| **wins vs losses** (the decisive one) | **−0.07** | −0.813 | 0.41626 | **0.48701** | **no** |
| losses vs control | +0.32 | 3.223 | 0.00127 | 0.00233 | **yes** |

Decision-relevant disagreements are far more defective than the agreement control — but **defects are
not biased toward the wins**; the point difference actually favours the losses. Mis-keying is symmetric:
BAD-GOLD 9% of wins against 10% of losses.

### 5.4.3 Corrected deltas under each correction model

PMC-VQA fusion cell, measured Δ against always-32B-direct = **+0.0135 [+0.0100, +0.0169]**,
n = 33,430, W = 1,969 wins, L = 1,518 losses **[M]**. Intervals below are Monte Carlo (200,000 draws,
uniform Beta prior on each audit proportion, item noise sd 0.00176) **[D]**:

| correction model | formula | fusion Δ | 95% interval | veto transfer |
|---|---|---:|---|---:|
| **A** — discount the wins only (as briefed) | (1 − f_wins) × Δ | +0.0063 | [+0.0027, +0.0100] | +0.0045 |
| **B** — symmetric drop-defective (the *correct* one) | (W(1−f_w) − L(1−f_l)) / n | **+0.0094** | [+0.0004, +0.0183] | +0.0058 |
| **C** — re-key BAD-GOLD only (win ↔ loss) | (W − L − 2W·f_bg,w + 2L·f_bg,l) / n | +0.0124 | [+0.0018, +0.0236] | — |
| **D** — cleaned benchmark (denominator shrinks too) | (W(1−f_w) − L(1−f_l)) / (n(1−d)) | +0.0136 | [+0.0006, +0.0265] | — |

Correction A is what the brief asked for but is **not** the right model here: the losses are at least
as defective as the wins, so discounting only the wins overstates the damage. Under the correct
symmetric model **69.6%** of the measured delta survives **[D]**, and the interval nearly touches zero.

Propagation to the sample-weighted pooled headline (only the PMC cell is re-costed) **[D]**:

| claim | measured | corrected A | corrected B |
|---|---:|---:|---:|
| accuracy-max-veto vs 32B-reasoning | +0.0245 | +0.0205 | +0.0222 |
| accuracy-max + fusion vs 32B-reasoning | +0.0271 | +0.0214 | +0.0238 |
| accuracy-max-veto vs always-32B-direct | +0.0106 | +0.0066 | +0.0083 |
| **compute-lean vs 32B-reasoning** | +0.0150 | +0.0154 | +0.0152 |

compute-lean's PMC cell is **−0.0010**, so correcting it is immaterial — **compute-lean never depended
on a PMC win**.

### 5.4.4 The noise ceiling [D]

Stratum masses **[M]**: agree-and-correct 0.4296, agree-and-wrong 0.2407, fusion wins 0.0589, fusion
losses 0.0454, non-decisive disagreements 0.2254.

| ceiling variant | pool defect rate | ceiling if defective items scored wrong | ceiling if defective items answered at 4-way chance |
|---|---:|---:|---:|
| generous (unsampled strata charged the agree-correct rate) | 0.3093 | 0.6907 | 0.7681 |
| stratified (non-decisive disagreements charged the disagreement rate) | 0.3709 | 0.6291 | 0.7219 |

**Achievable PMC-VQA accuracy is bounded at roughly 0.63–0.77.** Every system in this project scores
**0.5427–0.5653** **[M]**, so the benchmark is not saturated — but a ~1-point margin is being measured
on a benchmark where **31–37%** of items cannot support a correctness claim at all.

### 5.4.5 The construct-validity finding

**46% of the wins sit on items where the gold is wrong or the answer is simply not in the shown
image [M]** (BAD-GOLD ∪ UNANSWERABLE). On those items the score is decided by which model's *language
prior* better matches a caption-derived key, not by which model reads the image better. Audited
examples **[M]**:

- `pmc-13058` — the question asks about a blue arrow on a head CT; the image is a **spleen ultrasound**.
- `pmc-24120` — the question asks about the femur; the image is a **chest CT**.
- `pmc-24810` — the question asks about blue labelling in photomicrographs; the image is a **photo of a
  cat's face**.
- `pmc-25510` — the 32B correctly described the panel that was actually shown, and was scored **wrong**.

### 5.4.6 Verdict

**Arithmetic SURVIVES; construct RETRACTED.** (1) The specific attack "the PMC delta is an artifact of
biased annotation error" **fails**, and fails for a measurable reason — defect symmetry across the
disagreement set (Fisher p = 0.487). (2) But the delta **cannot be described as an accuracy improvement
in the medical-visual sense**: about half of it is earned on items that are not visual questions.
Report it as **"higher agreement with PMC-VQA's caption-derived keys"**, with the 53% defect rate
stated alongside, and stop using PMC-VQA to carry an accuracy claim. (3) The audit does **not** bear on
the open-text cells or the compute/latency claims.

> **No external comparator exists.** A literature search for an independent quantitative audit of
> PMC-VQA label error found none; this is recorded as UNVERIFIED in `PMCVQA_PROVENANCE_2026-07-30.md`.
> The 53% / 60% / 28% defect rates therefore have no outside reference point. **[U]**

## 5.5 Verifier contamination and the clean retrain

**What was measured.** (a) An in-sample/out-of-sample stratification of the deployed adapter
(`verifier_validity_2026-07-29.json`); (b) a full **retrain on strictly disjoint data** and re-scoring
of the identical candidate pools (`verifier_disjoint_retrain_2026-07-30.json`,
`macro_headline_clean_verifier_2026-07-30.json`).

### 5.5.1 Overlap [M]

| eval set | n | seen in training | **% seen** | held-out 30% |
|---|---:|---:|---:|---:|
| SLAKE-open | 645 | 435 | **67.4%** | 210 |
| PathVQA-open | 1,500 | 1,065 | **71.0%** | 435 |
| VQA-RAD-open | 200 | 146 | **73.0%** | 54 |
| Kvasir-open (training pool member) | 1,200 | 835 | 69.6% | 365 |
| RadImageNet-open (out-of-domain) | 2,000 | 0 | 0% | — |

The first (stratified) estimate of inflation: pooled selection gain **+0.1040** full against **+0.0718**
n-weighted held-out → **~31% memorisation** **[D]**; per-cell inflation 1.82× (SLAKE), 1.44×
(PathVQA), 1.19× (VQA-RAD).

### 5.5.2 The clean retrain design [M]

A naive image-disjoint filter of the *evaluation* sets would discard **71.2% / 73.0% / 71.9%** of
SLAKE-open / VQA-RAD-open / PathVQA-open items, so the design instead trains on the datasets' official
train splits plus two out-of-domain pools, at two strictness levels: **L1 (headline)** — image-disjoint,
16,621 train questions → 10,364 training examples, composition matched; **L2 (lower bound only)** — L1
plus no eval question text at all, 6,490 train questions. Disjointness holds exactly: image pixel-hash
intersection **0**, question-item intersection **0**, question-id intersection **0**, L2 question-text
intersection **0** (5,229 train images against 528 eval images).

### 5.5.3 Clean selection gain over greedy [M]

| cell | n | contaminated gain | **clean L1 gain** | 95% interval (L1) | significant | inflation × (L1) | clean L2 gain |
|---|---:|---:|---:|---|---|---:|---:|
| SLAKE-open | 645 | +0.0434 | **+0.0109** | [−0.0171, +0.0388] | no | 4.00× | 0.0000 |
| VQA-RAD-open | 200 | +0.1100 | **+0.0150** | [−0.0350, +0.0650] | no | 7.33× | 0.0000 |
| PathVQA-open | 1,500 | +0.1293 | **+0.0493** | [+0.0320, +0.0667] | **yes** | 2.62× | −0.0187 |
| **POOLED** | 2,345 | **+0.1041 [+0.0891, +0.1190]** | **+0.0358** | **[+0.0213, +0.0503]** | **yes** | **2.90×** | −0.0119 [−0.0277, +0.0034] |

**Inflation factor: 2.90× at L1** — the earlier stratified estimate of 1.45× ("31% memorisation") was
itself an underestimate. Under the strictest reading (L2) the gain is **null**. L2 is reported as a
conservative lower bound only, for the reason given in §4.7 — but a reviewer could reasonably prefer
it, in which case the open-text selection gain is null.

### 5.5.4 The key mechanistic result: ranking held, oracle conversion collapsed [M]

| quantity | contaminated | clean L1 | clean L2 |
|---|---:|---:|---:|
| candidate-level ranking AUROC (pooled) | 0.9433 | **0.8856** | 0.7960 |
| **oracle conversion** (share of greedy→oracle-at-8 headroom captured) | **0.5894** | **0.2029** | **−0.0676** |

**Contamination did not buy ranking ability — it bought selection.** AUROC falls only 0.943 → 0.886,
while oracle conversion falls 0.589 → 0.203. Memorising the seen items is what turned a good *ranker*
into a good *selector*. (Greedy 0.4495, self-consistency 0.4469, oracle-at-8 0.6260, pooled
n = 2,345 **[M]**.)

### 5.5.5 The escalation-rate consequence — larger than the accuracy consequence [M/Mo]

Because τ is chosen to *reach* the strong leg's accuracy at minimum escalation, a weaker verifier is
paid for in escalation, not in accuracy:

| quantity | contaminated | clean L1 | clean L2 |
|---|---:|---:|---:|
| sample-weighted escalation to hold 32B-direct parity | **4.0%** | **26.9%** | 82.7% |
| open-arm best-of-8 parity escalation, macro | 0.0397 | **0.4868** | 0.8331 |
| open-arm batch-1 latency | 548.4 ms | 700.9 ms (sample-wtd) / 845.7 ms (macro) | 1,072.1 ms |
| open arm vs always-32B-**reasoning** | −94.8% | −93.3% / −92.0% | −89.8% |
| open arm vs always-32B-**direct** | **−17.5%** | **+5.4% / +27.2%** | +61.2% |

The published "−94.8% batch-1 latency" barely moves, because the reasoning baseline is 10.5 s. **The
figure that actually breaks is the one against a single 32B forward**: the arm goes from 17.5% faster
than always-32B-direct to **5.4–27.2% slower**, at 3.77× its FLOP-eq.

Open-arm accuracy **[M]**: contaminated 0.5642, **clean L1 0.5143**, always-32B-direct 0.5168 → the
open arm **beat** the 32B contaminated (+0.0473) and **does not** clean (−0.0025, i.e. parity).
First-order effect on the full-suite sample-weighted pooled accuracy: 0.5750 → **0.5722** (−0.0028)
**[D]** — small only because the open arm is 5.5% of the sample-weighted pool and the multiple-choice
arm (94.5%) never touches the verifier.

### 5.5.6 Verdict

**REDUCED to roughly one third, and only on one dataset.** The accuracy claim survives at L1 at
**+0.0358 [+0.0213, +0.0503]** against the published **+0.1041** (2.90× inflation), but pooled
significance rests **entirely on PathVQA-open** (64% of the open sample); SLAKE and VQA-RAD both have
intervals spanning zero. Under L2 the gain is null. The **efficiency** claim for the open arm is
damaged more than the accuracy claim.
## 5.6 The headline under each accounting

**What was measured.** The same 8 reporting cells (Variant B: MMMU excluded, 5 benchmarks,
n = 42,224) re-scored under three cumulative accounting decisions: **(A)** as published —
sample-weighted, contaminated verifier; **(B)** macro over cells (1/8 each), contaminated verifier;
**(C)** macro **plus** the clean L1 verifier. Column **D** (macro + L2) is a conservative lower bound.
Items are bootstrapped within cells (paired, exact pattern-multinomial), the macro average recomputed
per replicate; 10,000 replicates, seed 20260730. Source:
`macro_headline_clean_verifier_2026-07-30.json`, whose contaminated column reproduces
`macro_average_headline_2026-07-30.json` on **1,224 / 1,224** compared fields exactly.

Why the weighting matters: PMC-VQA goes from **79.17% → 12.5%** of the weight; the open-text arm from
**5.6% of items → 37.5%** of the weight. Escalation is wildly heterogeneous across cells (8.45%
PMC-VQA → 89.60% MedXpert), and PMC-VQA — the lowest-escalation cell — carried 79% of the pooled
average **[M]**.

### 5.6.1 Accuracy levels [M]

| system | **A** published (sample-wtd, contaminated) | **B** macro only (contaminated) | **C** macro + clean verifier (L1) | D macro + L2 (lower bound) |
|---|---:|---:|---:|---:|
| always-7B | 0.5549 | 0.5971 | 0.5971 | 0.5971 |
| always-32B-direct | 0.5729 | 0.6567 | 0.6567 | 0.6567 |
| always-32B-with-reasoning | 0.5591 | 0.5974 | 0.5974 | 0.5974 |
| oracle mode-select 32B (not deployable) | 0.5730 | 0.6573 | 0.6573 | 0.6573 |
| **method — compute-lean** | 0.5741 | 0.6600 | **0.6443** | 0.6409 |
| **method — accuracy-max (certified veto + learned deferral)** | 0.5836 | 0.6694 | **0.6575** | 0.6548 |
| method — accuracy-max with decision fusion | 0.5862 | 0.6661 | 0.6503 | 0.6470 |

Baselines are identical across columns by construction: de-contamination changes only the three
open-text cells, and the macro re-weighting moves the baselines but not their internals.

### 5.6.2 The headline comparisons, side by side [M]

| comparison | pool | **A** published | **B** macro only | **C** macro + clean L1 | D (L2) |
|---|---|---|---|---|---|
| **accuracy-max vs always-32B-direct** | all 8 | **+0.0107 [+0.0086, +0.0127] WIN** | **+0.0128 [+0.0056, +0.0200] WIN** | **+0.0008 [−0.0022, +0.0037] TIE** | −0.0019 [−0.0055, +0.0014] TIE |
| accuracy-max vs oracle-mode-32B | all 8 | +0.0106 WIN | +0.0122 WIN | **+0.0002 [−0.0029, +0.0033] TIE** | −0.0025 TIE |
| **compute-lean vs always-32B-direct** | all 8 | +0.0011 [−0.0028, +0.0051] TIE | +0.0033 [−0.0054, +0.0121] TIE | **−0.0124 [−0.0188, −0.0060] LOSS** | −0.0157 [−0.0220, −0.0096] LOSS |
| compute-lean vs always-32B-direct | 5 multiple-choice cells | −0.0015 [−0.0056, +0.0026] TIE | **−0.0070 [−0.0126, −0.0017] LOSS** | **−0.0070 [−0.0126, −0.0017] LOSS** | same |
| compute-lean vs oracle-mode-32B | 5 multiple-choice cells | −0.0016 TIE | **−0.0080 [−0.0137, −0.0024] LOSS** | **−0.0080 LOSS** | same |
| compute-lean vs always-32B-direct | 3 open cells | **+0.0456 [+0.0303, +0.0614] WIN** | +0.0206 [−0.0009, +0.0423] TIE | **−0.0214 [−0.0360, −0.0074] LOSS** | −0.0303 LOSS |
| accuracy-max vs always-32B-direct | 3 open cells | +0.0559 WIN | +0.0309 [+0.0116, +0.0502] WIN | **−0.0010 [−0.0090, +0.0067] TIE** | −0.0081 TIE |
| accuracy-max-fusion vs always-32B-direct | all 8 | +0.0132 WIN | +0.0094 [+0.0013, +0.0176] WIN | **−0.0063 [−0.0118, −0.0011] LOSS** | −0.0097 LOSS |
| **accuracy-max vs always-32B-with-reasoning** | all 8 | +0.0245 WIN | **+0.0720 [+0.0614, +0.0824] WIN** | **+0.0601 [+0.0498, +0.0703] WIN** | +0.0574 WIN |
| compute-lean vs always-32B-with-reasoning | all 8 | +0.0150 WIN | +0.0626 [+0.0514, +0.0734] WIN | **+0.0468 [+0.0353, +0.0583] WIN** | +0.0435 WIN |
| accuracy-max vs always-7B | all 8 | +0.0287 WIN | +0.0723 WIN | **+0.0604 [+0.0488, +0.0720] WIN** | +0.0577 WIN |

**Sign flips (11 recorded) [M].** compute-lean vs 32B-reasoning (multiple choice); compute-lean vs
32B-direct (all 8; open); compute-lean vs oracle (all 8; open); accuracy-max-veto vs 32B-direct (open);
accuracy-max-veto vs oracle (open); accuracy-max-fusion vs 32B-direct (all 8; open);
accuracy-max-fusion vs oracle (all 8; open).

**Significance changes A → C: 17 recorded [M].** Six WIN → LOSS or TIE → LOSS on compute-lean; six
WIN → TIE on accuracy-max-veto; five WIN → TIE/LOSS on fusion. The four significance losses
attributable to the **verifier alone** (B → C) are: accuracy-max-veto against 32B-direct (all 8, and
open), and against oracle-mode (all 8, and open).

**What became a significant loss [M].** Three things previously a win or a harmless tie:

1. **compute-lean vs always-32B-direct on the 5 multiple-choice cells: −0.0070 [−0.0126, −0.0017]** —
   created by the macro re-weighting alone, unchanged by the verifier (those cells never use it).
2. **compute-lean vs always-32B-direct over all 8 cells: −0.0124 [−0.0188, −0.0060]** — created by the
   clean verifier.
3. **accuracy-max-fusion vs always-32B-direct over all 8 cells: −0.0063 [−0.0118, −0.0011]**.

**Decomposition of the damage [D].** On the headline comparison (accuracy-max against
always-32B-direct, all 8 cells), macro re-weighting alone *helped* by **+0.0021**; the clean verifier
cost **−0.0120**; net **−0.0099**. The two corrections are separable.

**Concentration [M].** Leave-one-cell-out range for accuracy-max against 32B-direct at C:
**[−0.0004, +0.0024]**, load-bearing cell **PMC-VQA**. For the contaminated macro numbers: compute-lean
against reasoning +0.0626 with range [+0.0225, +0.0732], load-bearing cell **PathVQA-open**, held back
by SLAKE-closed. Macro intervals are 2–4× **wider** than pooled ones precisely because the small cells
now count as much as PMC-VQA.

### 5.6.3 The cost table

Cost constants are **measured** batch-1 quantities (decode 68.573 ms/tok and 18.261 J/tok from
`latency_32b.jsonl`, n = 60 per configuration, medians; prefill 280.99 ms / 24.638 J; 665 ms /
126.9 J measured direct anchor, carried in code as 127.0 J). The **FLOP-eq ratio 4.57** is an
**underived literal** reproducing 32.0 B / 7.0 B = 4.571; no file in the repository derives it, and an
older document implies 4.34 — every ratio below inherits that ~7% margin. The composition into a
per-query cost is **[Mo]**, and **no figure here comes from executing the assembled cascade**.

**Absolute macro costs [Mo]:**

| system | FLOP-eq (as charged) | FLOP-eq (honest re-cost) | latency, parallel | latency, sequential | energy |
|---|---:|---:|---:|---:|---:|
| always-7B | 1.00 | 1.00 | 347.0 ms | 347.0 ms | 45.8 J |
| always-32B-direct | 4.57 | 4.57 | 665.0 ms | 665.0 ms | 127.0 J |
| always-32B-with-reasoning | 4.57 | **5.697** | 10,521.6 ms → **6,291.2 ms** | same | 2,001.9 J → **1,625.2 J** |
| oracle-mode-32B | 4.57 | 4.57 | 1,897.1 ms → **645.5 ms** | same | 361.4 J → **121.8 J** |
| compute-lean (contaminated) | 5.465 | 5.465 | 649.7 ms | 1,292.0 ms | 188.0 J |
| **compute-lean (clean L1)** | **6.674** | 6.674 | 690.8 ms | 1,574.8 ms | 228.8 J |
| accuracy-max-veto (contaminated) | 6.444 | 6.444 | 691.4 ms | 1,333.7 ms | 206.2 J |
| **accuracy-max-veto (clean L1)** | **7.951** | 7.951 | 775.9 ms | 1,660.0 ms | 255.3 J |

**Ratios against always-32B-direct [D/Mo]:**

| operating point | A published (sample-wtd) | B macro only | C macro + clean L1 | D (L2) |
|---|---:|---:|---:|---:|
| compute-lean — FLOP-eq | **0.492×** | 1.196× | **1.46×** | 1.633× |
| compute-lean — parallel latency | −29.5% | −2.3% | **+3.9%** | +7.3% |
| compute-lean — sequential latency | −12.9% | +94.3% | **+136.8%** | +165.0% |
| compute-lean — energy | −34.2% | +48.0% | **+80.2%** | +101.3% |
| accuracy-max-veto — FLOP-eq | **0.932×** | 1.410× | **1.74×** | 1.897× |
| accuracy-max-veto — parallel latency | +9.9% | +4.0% | **+16.7%** | +18.6% |
| accuracy-max-veto — sequential latency | +26.4% | +100.6% | **+149.6%** | +176.3% |
| accuracy-max-veto — energy | +7.6% | +62.4% | **+101.0%** | +120.6% |
| fusion — FLOP-eq | 1.25× | 1.435× | 1.699× | 1.872× |

**Against a 32B actually made to reason, honestly re-costed [D/Mo]:** accuracy-max-veto at C is
**−87.7% parallel latency, −84.3% energy, 1.396× FLOP-eq** (1.74× as charged); compute-lean at C is
**−89.0% / −85.9% / 1.171×**.

**Escalation rates [M]:**

| quantity | A published (sample-wtd) | B macro only | C macro + clean L1 | D (L2) |
|---|---:|---:|---:|---:|
| compute-lean, all 8 cells | 0.1689 | 0.3565 | **0.4183** | 0.4525 |
| compute-lean, 3 open cells | 0.2823 | 0.2133 | **0.3780** | 0.4693 |
| open-arm best-of-8 parity escalation | 0.0397 | 0.0604 | **0.4868** | 0.8331 |

Per-cell escalation, contaminated **[M]**: PMC-VQA 8.45%, SLAKE-closed 20.45%, PathVQA-closed 45.72%,
VQA-RAD-closed 56.97%, MedXpert 89.60%, SLAKE-open 15.81%, VQA-RAD-open 12.50%, PathVQA-open 35.67%.
Equal-weighting raises the multiple-choice escalation rate from **16.22% → 44.24%**. That is the whole
mechanism of the cost reversal.

**A note on the "always-32B-with-reasoning" baseline.** It barely reasons on the multiple-choice cells:
measured mean generated tokens are PMC-VQA **3.09**, SLAKE-closed **3.33**, VQA-RAD-closed **3.01**,
**PathVQA-closed — no reasoning dump exists at all; the value is imputed equal to direct** — against
MedXpert **320.33**, SLAKE-open **122.41**, VQA-RAD-open **104.54**, PathVQA-open **141.47** **[M]**.
Genuine 32B reasoning exists on **4,345 of 42,224 items = 10.3%** of the pool **[M]**, yet the
as-charged convention bills 10,521.6 ms / 2,001.9 J to all cells. Hence the honest re-costing column.
**This baseline is the weakest input to the surviving vs-reasoning claim.**

### 5.6.4 Pareto status — computed, not asserted

Domination was computed axis by axis on the honestly re-costed macro numbers (a point dominates if it
is at least as accurate and no more expensive on FLOP-eq, parallel latency, sequential latency and
energy) **[D]**:

| operating point | vs always-32B-reasoning | vs always-32B-direct | vs oracle-mode-32B | on frontier (all 4 cost axes) |
|---|---|---|---|---|
| **B — macro, contaminated** | | | | |
| compute-lean | **dominates** | neither | neither | yes |
| accuracy-max-veto | neither (cheaper on latency/energy, dearer on FLOPs) | neither | neither | yes |
| accuracy-max-fusion | neither | neither | neither | FLOPs no, parallel latency yes |
| **C — macro + clean L1** | | | | |
| compute-lean | neither | **DOMINATED by always-32B-direct** | **DOMINATED by oracle-mode** | **no** |
| accuracy-max-veto | neither | neither (accuracy gap +0.0008) | neither (+0.0002) | **yes** |
| accuracy-max-fusion | neither | **DOMINATED** | **DOMINATED** | **no** |

**"Pareto-DOMINATES every fixed way of using the 32B" is RETRACTED** — it was a sample-weighted
artifact. Under macro plus a clean verifier, **compute-lean and accuracy-max-fusion are strictly
dominated by a single always-32B-direct forward pass**; only accuracy-max-veto remains non-dominated,
and only by a **+0.0008 accuracy edge that is not statistically distinguishable from zero**.

> The July-8 framing — "only always-7B, compute-lean and accuracy-max are non-dominated" — was
> sample-weighted and used the contaminated verifier. **It is superseded by this table**, not an
> alternative reading of it.

### 5.6.5 Verdict — the honest headline

> **8-cell macro, clean verifier: accuracy-max is +0.0008 [−0.0022, +0.0037] against
> always-32B-direct at 1.74× its compute — a tie bought with more compute, not a win.**

**What survives [M/D]:**

- Against a 32B **actually made to reason**, the accuracy margin *grows* under macro even after
  de-contamination: **+0.0601 [+0.0498, +0.0703]** at −87.7% latency and −84.3% energy (but 1.396×
  honestly re-costed FLOP-eq — *not* fewer FLOPs).
- "Reasoning mode is actively harmful on free-text medical VQA" is untouched by both corrections:
  PathVQA-open 0.1087 against 0.3760 direct; SLAKE-open 0.6791 against 0.8186; VQA-RAD-open 0.5450
  against 0.6000.
- On the 5 multiple-choice cells accuracy-max still beats always-32B-direct:
  **+0.0019 [+0.0014, +0.0024]** — that half never used the verifier.
- The verifier is still a real ranker after de-contamination (candidate AUROC 0.886).

**What does not survive:** "the method beats a single 32B forward pass"; "compute-lean matches the
strong model at half the compute"; "the open-text arm beats always-32B-direct"; "the method
Pareto-dominates the 32B baselines".

**Two structural caveats a reviewer will find.** (i) τ, the adaptive-sampling parameter and the
veto/deferral thresholds were all calibrated against a **pooled** iso-accuracy objective, but the
report is now on a **macro** metric; a macro-objective refit has **not** been done, so §5.6 is "a
pooled-tuned method scored on a macro metric" — a mis-specification, not a fabrication. (ii) The 8
cells are **not 8 independent datasets**: SLAKE, VQA-RAD and PathVQA each contribute a closed *and* an
open cell from the same images, so equal-weighting triple-counts three source datasets and gives the
open-text *format* 37.5% of the weight. That is why the 5-benchmark macro is computed as a robustness
check (compute-lean 0.6131, accuracy-max 0.6223; against reasoning +0.0498 and +0.0590) **[M]**.

## 5.7 Ancillary findings

### 5.7.1 The MMMU contamination audit

**What was measured.** Lingshu-7B scored **0.80** on the 150-item MMMU Health & Medicine validation set
against Lingshu-32B-direct's **0.6333** — +26 points over the Lingshu paper's own 7B number (54.0). Six
adversarial checks were run (`mmmu_verify.json`, `mmmu_fix.json`). All **[M]**:

| check | result |
|---|---|
| model identity | genuinely Lingshu-7B: 8.29 B parameters, Qwen2.5-VL-7B architecture (hidden 3584, 28 layers, visual tower present), snapshot `b98aecd…` |
| **image ablation (decisive)** | Lingshu-7B: real image **0.8267** → blank **0.62** → noise **0.62** → text-only **0.5933**. The image genuinely drives ~23 points |
| **control model (decisive)** | base Qwen2.5-VL-7B-Instruct (Lingshu's own base) scores **0.5667** with the real image through the identical harness |
| gold subset | 150 official validation items (5 subjects × 30), all unique ids, **0** gold-letter and **0** option-set mismatches against the official cached arrow |
| prompt leakage | full chat template dumped; the gold letter is never indicated |
| independent rescore | a non-MedEvalKit strict parser gives 123/150 = 0.82 against MedEvalKit's 120/150 = 0.80, 0 no-parse — MedEvalKit is if anything conservative |
| discordance | 7B-only-correct **34** against 32B-only-correct **9**; McNemar χ²(cc) 13.395, binomial two-sided **p = 1.70e−4** |
| position-bias test (full cyclic permutation of options) | 7B 0.8267 → **0.7708** debiased; 32B 0.6467 → **0.6321**; gap 0.18 → **0.1387**. Position bias explains ~4 of ~18 points |
| harness cross-check | the same 7B reproduces the Lingshu paper elsewhere: SLAKE 82.5 (83.1), PMC-VQA 54.3 (56.3), MedXpert-MM 26.2 (26.7); the 32B reproduces MMMU 63.3 (62.3) |

**Verdict: the 0.80 is a genuine Lingshu-7B output, not an our-end bug — and therefore most plausibly
train-set contamination outside our control.** MMMU is **excluded** from headline claims.

The exclusion's cost is asymmetric. Sample-weighted it moves the headline by **−0.0005** (MMMU is 0.35%
of the pool), but **under macro it would carry 1/9 = 11.1% of the weight**, and macro-9 against macro-8
deltas differ materially: compute-lean against direct **+0.0215 against +0.0033**, accuracy-max against
direct **+0.0299 against +0.0128**, fusion against direct **+0.0269 against +0.0094**. **The exclusion
must therefore be argued on contamination grounds alone, with its size stated.** Original claim "MMMU
keep-7B is a +0.140 beat-the-strong-model win": **RETRACTED**.

> The macro-9 against macro-8 figures in the paragraph above are quoted from `CLAUDE.md` §0, which
> attributes them to the 2026-07-30 macro pass. They were **not** re-derived directly from
> `macro_average_headline_2026-07-30.json` in this pass (that file is 260 KB and only its
> `accuracy_levels` and `verdict` blocks were read). **[U on re-derivation; the MMMU sensitivity
> figures in `mmmu_fix.json` were independently verified]**
>
> Separately: **whether the MMMU exclusion was ever ratified by the researcher is not recorded
> anywhere**, and neither is whether the paper was submitted. **[U]**

### 5.7.2 The SLAKE image-path bug — scoped and cleared

**What was measured.** `src/training_methods/verifier_transfer_eval.py`'s `imgs_for(ds)` had no
`slake_open` branch; its `else` branch selected PathVQA images keyed by PathVQA row index. Four
independent checks (`slake_image_path_bug_audit_2026-07-30.json`) **[M]**:

1. **Reachability** — SLAKE-open question ids run 11,934–12,991 while the PathVQA index space tops out
   at 6,717; the buggy path would have resolved **0** of them (the scorer skips indices not in the
   image map), so it could have produced **at most 0 rows**. The actual dump has **645**.
2. **Provenance** — the dump was produced by `src/cascade_methods/gen_slake_open_bestofN.py`, which
   defines its own correct SLAKE loader; 645 rows, **645/645 scores identical**, 0 differing.
3. **Image ablation** — the "real" condition reproduces the correct-image dump to mean absolute
   deviation **2.65e−06**, max **5.0e−06**, **100%** within 0.01 over n = 483, while
   blank-gray/blank-black/blank-matched/mismatched/no-image conditions all deviate by ~0.30–0.33. The
   ablation scored the true SLAKE images.
4. **Siblings** — 51 scripts scanned; only this one lacked a SLAKE loader, and git history confirms no
   `slake_open` branch ever existed — a latent gap, not a regression.

**Verdict: published numbers affected — NONE.** The conclusion that the verifier leans substantially on
text priors on SLAKE **stands as published**. Transfer results for VQA-RAD, PathVQA, Kvasir and
RadImageNet went through correct branches.

### 5.7.3 The two PMC-VQA splits

**What was measured.** Two evaluation tracks use **different PMC-VQA splits**, and no document recorded
it. `PMCVQA_PROVENANCE_2026-07-30.md`, all **[M]**:

| file | data rows | version | verified? | used by |
|---|---:|---|---|---|
| `test_clean.csv` | **2,000** | v1, 8-column | **yes** — the authors' only manually checked split | cascade / margin-gate track, via MedVLThinker-Eval (8,220 items) — PMC is **24.3%** of that pool |
| `test_2.csv` | **33,430** | v2 ("noncompound images"), undocumented | **no** — zero published verification | MedEvalKit / Lingshu track — PMC is **79.17%** of the 42,224-item pool |
| `test.csv` | 50,000 | v1 | no — the paper's "PMC-VQA-test-initial" | neither |

Key measurements: `test_clean` ⊂ `test.csv` at 2,000/2,000; **`test_clean` ∩ `test_2` = 6 items of
2,000** — effectively disjoint populations, so the 33,430-row dumps **cannot** be filtered down to the
verified split; `test_2` gold-letter distribution is skewed (C 12,636 / B 11,984 / A 4,423 / D 4,387 —
gold-A only 13.2%); non-clinical MDPI journal figures occur in ~3–5% of both splits. Gate calibration
is clean either way: the 3,000-item calibration sample shares **0** figures with `test_clean` and **0**
figure-plus-question duplicates; **0** of 176,948 v1-train rows collide with a `test_2` item.

> **That `test_clean.csv` IS the PMC-VQA paper's manually verified split is an INFERENCE**, not a
> verbatim-sourced fact. `PMCVQA_PROVENANCE_2026-07-30.md` §2.1 rates it high-confidence but explicitly
> not verbatim-sourced; the CSV carries no verification or provenance column, and the local README says
> only "metafile of test clean set". The inference rests on the paper's wording, the file name and the
> exact row count. **[U]**

**A self-correction is recorded here.** The 2026-07-29 retrospective asserted that `test_clean.csv` "is
not on disk" and "has never been used anywhere in the repo". **Both are false.** It exists in two
byte-identical copies (`/data/dan/dataset/medevalkit/PMC-VQA/test_clean.csv`, 418,686 bytes, 2,000 data
rows; and `/data/dan/dataset/pmc_vqa_train/test_clean.csv`, md5 `6abfbcd088171c76a98911c5e7a8f5a0`),
and the cascade track has been evaluating exactly those 2,000 items all along — MedVLThinker-Eval's
`pmc_vqa` slice matches `test_clean.csv` **2,000/2,000** on normalised question, `answer_label` and
answer text, and `ckpts/gate_7b_prune/cap320/ckpt_PMC-VQA_nothink_norag.jsonl` golds match
`test_clean.Answer_label[idx]` **2,000/2,000** **[M]**. (Second error in the same statement: the
`test_2` hard-code is at `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:39`, not `:41`.) Root cause: the
2026-07-29 pass *inferred* directory contents instead of listing them, and used `wc -l` as a CSV row
count, which over-counts because fields contain embedded newlines.

**Consequence for the claims.** The framing "the headline leans on PMC-VQA at 79% of the pool" is true
**only of the MedEvalKit pool**. The cross-family reasoning table's PMC column — including both sign
flips in §5.1.4 — is on the **verified** `test_clean.csv` (n = 2,000); the §5.4 label audit and the
fusion/veto win are on the **unverified** `test_2.csv` (n = 33,430). These must never be cross-quoted.
**Rule adopted: never write "PMC-VQA test" unqualified — always give the file name and the row count.**

## 5.8 Summary of claim status

| claim | status | corrected value |
|---|---|---|
| Reasoning hurts perception, 15/20 cells | **SURVIVES, strengthened** | 17/20; pooled −0.0401 [−0.0456, −0.0347], n = 30,250 |
| Reasoning helps reasoning-heavy benchmarks (5 families) | **RETRACTED as stated** | an answer-format effect; 0/9 matched trigger effects significant, 3/9 format effects are |
| Lingshu-32B as reasoning-versus-direct evidence (7 cells) | **WITHDRAWN, both directions** | published "think" arm generated 3.0–3.3 tokens |
| QoQ-Med-VL-32B reasoning gain (MMMU +0.071) | **WITHDRAWN** | +0.0118 [−0.0588, +0.0824] matched; +0.0000 fully matched |
| Reasoning hurts open-text perception (Δ = −0.214) | **SURVIVES; headline margin reduced** | clean contrast −0.2158 [−0.2354, −0.1962]; vs-reasoning open margin falls by −0.1164 |
| Answer format governs whether test-time compute buys anything | **SURVIVES** on multiple choice; the open-text half is **no longer provisional** — the matched re-run landed | detection AUROC ~0.66–0.73 (multiple choice) against 0.866 (open); open-text reasoning effect confirmed at −0.2158 with a null confound |
| PMC-VQA win is not an annotation artifact | **SURVIVES** | symmetric correction keeps +0.0094 [+0.0004, +0.0183] |
| PMC-VQA win is an accuracy improvement | **RETRACTED (construct)** | 46% of wins are bad-gold or unanswerable; report as key-agreement |
| Open-text verifier selection gain +0.1041 | **REDUCED ~3×** | clean L1 +0.0358 [+0.0213, +0.0503]; significant on PathVQA only; null at L2 |
| Open arm at 3.97% escalation / −94.8% latency | **RETRACTED (efficiency)** | clean L1 needs 26.9–48.7% escalation; +5.4% to +27.2% slower than one 32B forward |
| Method beats a single 32B forward pass | **RETRACTED** | accuracy-max +0.0008 [−0.0022, +0.0037] — a tie, at 1.74× the compute |
| Method Pareto-dominates every fixed way of using the 32B | **RETRACTED** | compute-lean and fusion strictly dominated; only accuracy-max-veto stays non-dominated, by a non-significant margin |
| Method beats a 32B made to reason | **SURVIVES, and grows under macro** | +0.0601 [+0.0498, +0.0703] at −87.7% latency, −84.3% energy — but computed against a prompt-unmatched reasoning arm (see §8.1) |
| The recoverability and selection limits | **SURVIVE** — the project's strongest contribution | see §8.5 |
| MMMU keep-7B is a +0.140 win | **RETRACTED** | genuine model output, but +26 over the published 7B number → contamination; excluded |
| SLAKE image-path bug invalidates the SLAKE verifier result | **REFUTED** | the buggy path was never executed; 0 published numbers affected |
---

# 6. Corrections and retractions

This project keeps a corrections log — not as an apology, as an instrument. Every entry records a
claim the project actually made, the evidence that falsified it, what replaced it, and the
generalisable lesson. The underlying log is `PROJECT_RETROSPECTIVE_2026-07-29.md` §10 (entries C1–C27,
X1–X22); this section restates the load-bearing ones and adds everything found in the 2026-07-29/30
pass. Entries duplicated with §5 are cross-referenced rather than repeated in full.

Two things set the standard for reading the rest:

1. **No number in this project was ever fabricated.** Every error below is a mis-specification, a
   mis-labelling, a mis-attribution, a contamination, or a propagation failure. The values were real;
   what they measured, or what they were compared against, was wrong.
2. **The 2026-07-30 pass cost the project its headline.** Two corrections landed the same day and
   together took the central claim from *"beats a single 32B forward pass"* to *"ties it, at 1.74× its
   compute."* That is written down here in the same detail as the wins.

## 6.1 Measurement hygiene

### 6.1.1 An estimate was labelled a measurement

**Believed.** Three documents and the 2026-07-27 slide deck printed **"Baselines (measured):
always-32B-think = 0.5632"**.

**Falsified by.** The artifact those documents drew from says otherwise in its own text. In
`method_final_mmmu_corrected.json` the three open-text cells carry `"acc_is_estimated": true` (16
occurrences of the flag are `true`, 98 are `false`) and the note *"32B-think open acc is a
measured-delta estimate; no per-sample CI"* — the pooled figure was a measured multiple-choice run
blended with an **estimated** open-text component, and the label "(measured)" was applied to the blend.

**Replaced by.** The fully measured value: **0.5594** on the full suite, **0.5591** on Variant B, from
`opentext_32b_think_full.json` and `f8_mode_vsthink_ci.json`.

**Downstream damage.** This single mis-label is one of four orthogonal axes along which the project's
"+0.02xx" headline family diverged (which mechanism, which pool, estimated-against-measured, and later
weighting). Five values — +0.0212 / +0.0207 / +0.0238 / +0.0245 / +0.0275 — circulated for what looked
like *the same row*. Decoding them took a dedicated table.

> **Lesson.** Provenance is a per-field property, not a per-file property. A pooled number inherits the
> *weakest* provenance of its components, so a pool containing one estimated cell is an estimate. Carry
> an explicit `estimated` / `measured` flag on every cell, propagate it into any aggregate, and never
> let a document assert a provenance the artifact does not.

### 6.1.2 A rounded `parse_ok`

**Believed.** In the first write-up of the answer-format result (§5.2), the extraction-integrity check
was stated as `parse_ok = 1.000` in every new arm — i.e. "no answer was ever unparseable, so the effect
cannot be an extraction artifact."

**Falsified by.** Reading the artifact rather than the rounded summary.
`medeval_matched_direct_2026-07-29.json : cells[*].parse_ok` shows the new `direct_matched` arms at
**exactly 1.0000 in 8 of the 9 primary sub-cells**, with a minimum of **0.9986** (InternVL3-38B,
MedXpert-Reasoning, n = 1,446). Counting all three arms per sub-cell, 7 of 9 are at exactly 1.0000
(Lingshu's MedXpert-Reasoning `reason` arm is 0.9993).

**Replaced by.** The phrasing now used: *"`parse_ok ≥ 0.9986` in every new arm; exactly 1.0000 in 8 of
the 9 primary sub-cells."*

**Why it mattered even though the conclusion did not change.** 0.9986 on n = 1,446 is two unparsed
items. The conclusion (not an extraction artifact) survives untouched. But "1.000" is a *claim of
exactness* the data does not support, and a reader who later found the 0.9986 would have no way to tell
an honest rounding from a covered-up defect.

> **A note on this correction's own provenance.** The pre-correction wording is **not preserved
> anywhere on disk.** Every copy of the write-up already carries corrected phrasing, and the
> repository's last commit predates the whole July chain, so git holds no earlier version. The
> corrected values *are* verified directly against the artifact; that the earlier draft said "1.000"
> is stated on the strength of the task record, not file evidence. **[U on the pre-correction
> wording]** A separate, third figure — "1.000 in 6 of 9" — appeared in an intermediate draft and is
> reproducible under no reading of the artifact; the correct counts are 8 of 9 (direct_matched arm) and
> 7 of 9 (all three arms per sub-cell).

> **Lesson.** Never round a diagnostic toward the answer you want. Report the **minimum** over sub-cells
> and the **count at the ceiling**, not a rounded mean. A sanity metric that reads exactly 1.000 should
> always be re-read at full precision before it is published.

### 6.1.3 Earlier cost-methodology errors

Three of the same kind, all caught before 2026-07-29 and all recorded:

| believed | falsified by | replaced by |
|---|---|---|
| Lingshu's always-reasoning mode is *cheaper* than always-direct | the researcher flagging a physical impossibility — the fitted cost model produced a **−16 J** intercept | an `a·gen + b` fit measured at 70–407 generated tokens must not be extrapolated to gen = 3 |
| the cascade efficiency headline, in three mutually inconsistent forms (e.g. 20.0 s → 5.7 s; compute 81 → 55%) | a three-way documentation audit | canonical **11.34 s → 2.27 s**, compute **100 → 52%**, energy **6,318.8 → 1,181.9 J** |
| the chest-X-ray box verifier is a negative, IoU **0.022** | the *implausibility of the magnitude* prompting a re-read: boxes were emitted in smart-resized pixel space | **0.232** — a coordinate-space bug had hidden the project's strongest positive |

> **Lesson.** Implausibility is evidence. A negative energy, a 10×-too-small IoU, or three numbers for
> one quantity are all falsifications available before any new experiment is run.

## 6.2 Contamination and provenance

### 6.2.1 MMMU-Medical — and the audit that proved it was *not* our bug

Full evidence table in §5.7.1. **Believed:** Lingshu-7B scores 0.80 on MMMU-Medical through our harness
while its own published number is 54.0, and it beats its own 32B there — banked as a **+0.140
beat-the-strong-model win** in three July-7 documents.

**The correct first suspicion was that we had broken something.** An adversarial audit was run and
concluded the *opposite* of the convenient answer: a genuine Lingshu-7B score, not an our-end bug,
*"consistent with train-set contamination outside our control."* A wiring or parsing bug could not
simultaneously make accuracy depend 23 points on image content *and* leave the untuned base model at
0.567.

**Replaced by.** MMMU-Medical is excluded from the headline; the +0.140 win is retracted. **And then
the exclusion itself was re-examined on 2026-07-30**: originally justified as immaterial (0.35% of
items, headline moves −0.0005), under equal weight per cell it would carry 11.1% of the headline. The
exclusion still stands — but must now be argued **on contamination grounds alone**, with its size
stated.

> **Lesson (two).** (a) When a number is anomalously *good*, the audit that clears your code is exactly
> as valuable as the one that finds a bug — run image-ablation × control-model, because together they
> separate "our pipeline is broken" from "this model has seen this data". (b) A decision justified by
> *"it doesn't matter much"* must be re-justified whenever the weighting changes, because "doesn't
> matter much" is a property of the aggregation, not of the decision.

### 6.2.2 The verifier was trained on 67–73% of what it scored

This is the most expensive correction in the project. Full numbers in §5.5.

**Believed.** The open-text arm's trained verifier delivers a **+0.1041 [+0.0891, +0.1190]** pooled
selection gain over greedy decoding, at **3.97%** escalation and **−94.8%** batch-1 latency.

**Falsified by, step 1 — the overlap audit.** The verifier was trained on a 70/30 grouped-by-question
split and then scored over the **full** sets: 67.4% (SLAKE-open), 71.0% (PathVQA-open), 73.0%
(VQA-RAD-open), 69.6% (Kvasir-open) seen in training. Worse, the "held-out 30%" was not clean either:
**100%** of SLAKE's unseen questions and **94.5%** of PathVQA's used an image the verifier had already
trained on.

**Falsified by, step 2 — the disjoint retrain.** Pooled selection gain
**+0.1041 → +0.0358 [+0.0213, +0.0503]** at L1 (2.90× inflation) and **−0.0119 [−0.0277, +0.0034]** at
L2; candidate AUROC 0.9433 → 0.8856 → 0.7960; **oracle conversion 0.589 → 0.203 → −0.068**; escalation
to hold parity 3.97% → 26.9% (sample-weighted) / 48.7% (macro) → 82.7%.

**What contamination actually bought.** *Not ranking ability.* What collapses is oracle conversion.
Memorising the seen items is what turned a good ranker into a good *selector*.

**Headline consequence.** accuracy-max against always-32B-direct goes **+0.0128 [+0.0056, +0.0200]
(WIN) → +0.0008 [−0.0022, +0.0037] (TIE)**, and compute-lean becomes **−0.0124 [−0.0188, −0.0060]
(significant LOSS)** at 1.46× the compute. The artifact separates the two 2026-07-30 corrections
cleanly: macro re-weighting alone *helped* this comparison (+0.0021); the clean verifier removed it
(−0.0120); net −0.0099.

> **Lesson.** "Grouped 70/30 split" is not a clean split if you then evaluate on the union. Three
> independent identity keys are needed — **item id, question text, and image pixel hash** — and the
> retrain must hold *everything else* (candidates, judge labels, random-number stream) byte-identical
> so the delta is attributable. And rank quality is not selection quality: report **oracle conversion**,
> not just AUROC, because contamination degrades the second far faster than the first.

### 6.2.3 "`test_clean.csv` is not on disk" — a correction to the corrections document

Full account in §5.7.3. **Believed** (asserted 2026-07-29, in the retrospective's own hole list and
correction row X14): PMC-VQA ships a human-verified 2,000-item `test_clean` split, but *"it is not on
disk"* and *"has never been used anywhere in the repo."* **Both are false**, and the same statement
cited the vendor hard-code at the wrong line (`:41` rather than `:39`).

**Root cause.** The 2026-07-29 pass **inferred** the directory contents instead of listing them, and
used `wc -l` as a CSV row count.

> **Lesson.** A corrections document is not exempt from the standard it enforces. Any claim of the form
> "X does not exist" is a *measurement*, and must be made with the command that measures it, quoted.
> And `wc -l` is not a row count for CSV.

### 6.2.4 Two PMC-VQA splits, used in two eras, recorded nowhere

Full table in §5.7.3. Two splits, one per evaluation track, **not comparable**: `test_2.csv` (33,430,
unverified, 79.2% of Variant B) against `test_clean.csv` (2,000, the authors' only manually checked
split, 24.3% of the internal pool), intersecting in **6 items**. Compounding it, the released file names
**invert the paper's naming**: in the PMC-VQA paper *"PMC-VQA-test"* IS the 2,000-item verified set and
the 50,000-item set is *"PMC-VQA-test-initial"* — but the shipped `test.csv` is the 50,000 unverified
set.

**The cross-split comparison that resulted.** The cross-family reasoning table's two PMC-VQA cells are
measured on `test_clean.csv`, n = 2,000, and were being read next to a headline PMC-VQA cell measured
on `test_2.csv`, n = 33,430, as though they were the same benchmark. Both cells also **flipped sign**
once the arms were properly matched (§5.1.4). Two independent defects — a split confusion and a prompt
confound — were sitting on the same two cells. Route A (filtering the 33,430-row dumps down to the
verified subset) is **permanently closed** — 6 items is not a subset.

> **Lesson.** When a project changes evaluation harnesses, it silently changes datasets. Record the
> **file name, row count and revision hash** of every split in the artifact itself, and treat a
> benchmark *name* as insufficient identification.

### 6.2.5 PMC-VQA's answer keys are derived from captions, not from images

Full account in §5.4. **Believed:** the one CI-certified multiple-choice win (+0.0135 fusion / +0.0095
veto) is an accuracy improvement. **Two findings, opposite in sign.** The specific attack fails,
measurably (53% wins against 60% losses, Fisher p = 0.487; the symmetric correction leaves
+0.0094 [+0.0004, +0.0183]). But the construct does not survive: **46% of the wins** sit on items where
the gold is wrong or the answer is **not in the shown image**.

**Replaced by.** Report any `test_2`-based PMC number as **"higher agreement with PMC-VQA's
caption-derived answer keys"**, with the 53% decision-relevant defect rate stated alongside. Reserve
the word *accuracy* for cells that survive an item-level validity check.

> **Lesson.** For any auto-generated VQA benchmark, ask **how the key was produced**. If questions and
> answers were generated from figure captions, a fraction of items are not visual questions at all, and
> a model can win them with a better language prior. The diagnostic is cheap and decisive: audit a
> stratified sample of the *decision-relevant* items with the source caption in view, and include an
> **agree-and-correct control stratum** so you can tell benchmark-wide noise (28%) from
> disagreement-set noise (53–60%).

## 6.3 Comparison design — the confound family

### 6.3.1 The prompt confound, and the fact that a previous *fix* introduced it

Full numbers in §5.1. **Believed:** "Reasoning hurts perception": **15/20** perception cells strictly
negative across five medical model families.

**How the defect was found.** Not by suspecting the cross-family table — by auditing something else.
The PathVQA judge audit noticed that the open-text arms used **different system prompts**
(`src/labeling/run_openvqa.py:26` against `:27`), a live style/length grading channel on free text. The
question then became: how far does this reach into the rest of the finding?

**Replaced by — and this is the surprise: correcting it made the finding *stronger*.** 17/20, not
15/20, under three independent correction policies; pooled −0.0401 [−0.0456, −0.0347] over 30,250
paired samples; two cells flip positive → negative. The multiple-choice half is **structurally immune**
to the open-text mechanism — a single-letter gold has no style or length grading channel — and that was
established, not assumed: the residual *extraction* channel was measured at 0–3.4%, and an adversarial
correction crediting every unparsed reasoning answer as correct changes no count.

**The part that is genuinely uncomfortable: a previous *fix* introduced the worst instance.** An
earlier correction had downgraded the finding on the grounds that a **foreign** reasoning prompt had
been applied to all families; the fix was to re-run each model with its **own native recipe**. That fix
**un-matched the arms** — the native recipes differ in persona, answer-format clause and image
resolution — and, decisively, **two of the native recipes contain no reasoning trigger at all**.
Lingshu's published "native think" instruction (`runners/run_native_think.sh:7`) is
`Answer with the option's letter from the given choices and put the letter in one "\boxed{}"` — a pure
answer-**format** string. Measured generated tokens: **3.0 against 3.0–3.3**. **No chain of thought ever
occurred.** All seven Lingshu-32B cells were **withdrawn in both directions**. Its quoted **1.2×
think:no-think latency ratio is not a reasoning ratio** — it is the ratio of two 3-token format prompts.

So one correction was reversed by a later one: the corrected analysis returns to the *foreign*-think
dumps, and the effect comes out **larger**, not smaller.

**Also corrected in the same pass:** QoQ-Med-VL-32B's MMMU reasoning gain **+0.0706 → +0.0118**
(interval [−0.0588, +0.0824]) under a matched prompt, with MedXpert-Understanding significantly
**negative** (−0.0433, p = 0.022) — withdrawn as reasoning evidence. Conversely, MedGemma-27B's PathVQA
reasoning win was **confirmed**: **+0.0399 → +0.0413 [+0.0220, +0.0607]**, p = 0.0000, on a **fully
matched** pair. The win is real; only its provenance changed. It remains the only perception cell where
chain-of-thought genuinely helps.

> **Lesson (three).** (a) A fix is an experiment and needs its own control — *"we switched to native
> prompts"* silently changed persona, format and resolution at once. (b) **Verify that the treatment
> occurred.** A "think" arm emitting 3.0 tokens is not a think arm; assert mean generated tokens on
> every reasoning arm before reading its accuracy. (c) Audit the *mechanism*, not the *label*: the
> open-text confound acted through a grading channel, so the right question was "does that channel
> exist here?" — which is what showed the multiple-choice half was safe.

### 6.3.2 The attribution error: the reasoning-side gains are an answer-**format** effect

Full 9-sub-cell table in §5.2.1. **Believed:** *a reasoning instruction improves accuracy on
reasoning-heavy benchmarks*, evidenced by MMMU +0.100 (MedVLThinker-32B) and +0.120 (InternVL3-38B),
MedXpert-Reasoning +0.046 / +0.035, MedXpert-Understanding +0.042 / +0.020.

**Falsified by** a matched-prompt re-run, complete across 6/6 (family × benchmark) cells and 9
sub-cells, paired on item id. Each published delta was decomposed into an **answer-format** component
and an explicit **reasoning-trigger** component: **0/9 trigger effects are CI-significant** (8/9
point-positive; mean shift from matching **−0.0276**); **3/9 format effects are**.

**Mechanism.** Asking for the answer in `\boxed{}` **is itself a reasoning trigger.** With **no trigger
present**, MedVLThinker-32B emits **431–580** generated tokens on **99–100%** of items and InternVL3-38B
**193–289** on **94–95%**; Lingshu never does (3–4 tokens). The published deltas conflated *"reasoning
against not"* with *"boxed against bare letter"*.

**Replaced by.** Drop *"a reasoning instruction improves accuracy on reasoning-heavy benchmarks."*
**Keep** the weaker, supported form — *getting a reasoning-tuned model to emit a trace helps
substantially* — with the **format** named as the operative lever. The honest substitute for the clean
contrast, unobtainable on these models because the format is what triggers the behaviour, is the
**monotone ladder** (§5.2.3).

**What did *not* change.** The cascade's gated-reasoning tier keeps its full value — the rung-1 →
rung-3 total is what such a tier actually delivers to a user. **Only the attribution changes.**

> **Lesson.** An output-format instruction is not a neutral control. Decompose any published delta into
> `format` + `trigger` before naming a cause, and state which one you measured. And a mechanism you
> cannot isolate should be reported as a **dose–response ladder**, not as a clean contrast you did not
> run.

### 6.3.3 The trap in the fix: a matched prompt can suppress the behaviour under test

Full account in §5.3. The open-text half was marked **provisional** pending a matched-prompt re-run. It
landed — and it contains the sharpest methodological lesson in the pass, because the *naive* reading of
the fix is wrong.

**The naive reading.** The matched arms score far above the unmatched reasoning arm (0.4235 / 0.4192
against 0.3028), so "most of the gap was the prompt." **That reading is wrong.** Telling this model *"Do
not explain"* **partly suppresses the very behaviour under test**: the matched arms emit a `<think>`
trace on only **25.5–71.1%** of items by dataset, against effectively all items for the unmatched
prompt. Their accuracy is a **mixture**, and their apparent gain is **dilution, not a prompt fix**.
Splitting arm B by whether it actually reasoned proves it — on trace-fired items it is far below the
direct arm on the *same* items (PathVQA **−0.1576**, SLAKE **−0.1134**, VQA-RAD **−0.0820**); on
no-trace items it matches (PathVQA **+0.0092**).

*(An internal inconsistency worth naming: the artifact's own prose says the matched arms trace on
"0.27–0.78 by dataset". Its measured per-arm rates are 0.2550–0.7107. The measured values are used
throughout this document.)*

**The clean contrast is the 2×2**, holding the output convention fixed: the convention effect with
reasoning off is **−0.0017 [−0.0111, +0.0077], not significant**; the reasoning effect at that fixed
convention is **−0.2158 [−0.2354, −0.1962], significant**. The identity check reconciles exactly
(residual −0.0000). The confound was **real as a description of the two arms and contributed ~nothing
to the measured gap**.

> **Lesson.** Before believing a matched-prompt result, check that the behaviour under test **still
> fires**, and report the effect **conditional on it firing**. A control that suppresses the treatment
> produces a diluted estimate that looks like a refutation. And prefer a **2×2** over a single matched
> pair: it lets you *measure* the confound's contribution instead of assuming it away.

## 6.4 Aggregation: the sample-weighting artifact behind "Pareto-dominates"

**Believed.** *"The method Pareto-dominates every fixed way of using the 32B."* This was the paper's
title, its second stated contribution, its main-result heading, and the one-line summary in
`README.md`, `PROJECT_OVERVIEW.md` and `READING_GUIDE.md`.

**Falsified by** re-basing the primary metric on **equal weight per reporting cell**
(`src/cascade_methods/macro_average_headline.py` → `macro_average_headline_2026-07-30.json`).
**No measured value changed.** Only which weighting it belongs to.

Full accuracy, cost and domination tables are in §5.6. The mechanism is that escalation is wildly
heterogeneous (8.45% to 89.60%) and **the lowest-escalation cell carried 79.2% of the pooled average**;
multiple-choice escalation goes 16.22% → 44.24%. Under equal weight **no operating point is
compute-cheaper than always-32B-direct**: compute-lean 0.492× → 1.196×, accuracy-max 0.932× → 1.410×,
fusion 1.250× → 1.435×. The old sample-weighted numbers are, approximately, PMC-VQA's accuracy and
PMC-VQA's cost.

**Replaced by.** *"Pareto-**optimal**" survives* — the points are non-dominated because they are more
**accurate**. *"Pareto-**dominates**" does not.* Restrict "dominates" to the always-32B-**with-reasoning**
baseline, and even there **not on FLOP-eq**. Compute-lean is now a **significant loss** on the five
multiple-choice cells: **−0.0070 [−0.0126, −0.0017]**.

**The nuance that must travel with it.** Macro **cost** answers a *different question* from
sample-weighted cost. Cost is **additive per query**, so on traffic resembling this suite the ~0.49×
saving is what you would actually pay; the macro number instead tests whether the saving
**generalises across task types** — and it does not, it is concentrated on the low-escalation cells.
**Report both, each labelled.**

**And a hole the re-basing opened.** τ, the adaptive-sampling parameter and the veto/deferral
thresholds were all calibrated against a **pooled** objective, which is no longer the reporting metric.
The §5.6 numbers are therefore *"a pooled-tuned method scored on a macro metric"*, not *"a
macro-optimal method"*. Every value is real — mis-specification, not fabrication — but it is stated,
not hidden.

> **Lesson.** Sample-weighting hands the microphone to your largest dataset. Decide the weighting
> **before** you read the result, report both, and label which question each answers. And note that
> fixing the aggregation can *invalidate the tuning*: a method optimised for one objective and reported
> on another is a third thing, and must be described as such.

## 6.5 Propagation: a retired claim survives in the artifact chain

Retracting a claim in prose does not retract it in the figures.

**The case.** `paper/make_ieee_figs.py` built the Pareto figure from `method_final_mmmu_corrected.json`
— Variant A, **sample-weighted**, and carrying an **estimated** always-32B-reasoning = 0.5628. After the
macro re-basing, the paper's main table carried **macro** accuracies and **macro** costs. The figure
plotted accuracy 0.5549–0.5862 at compute 1.00–5.71 while the table stated 0.5971–0.6694 at 1.00–6.56.
The figure and the table disagreed on **both axes**, and the figure still showed the method points
down-and-left of the 32B — *the retired dominance picture.*

**Caught by** the documentation audit. **Deliberately not regenerated in that pass** — the correct
artifact was a different file, and inventing the points was not an option. The instruction shipped
instead was: do not ship the PDF.

**FIXED on 2026-07-30, and this document records the fixed state as current.**
`paper/make_ieee_figs.py` (modified 2026-07-30 12:34) now sets `PARETO_ART =
"macro_average_headline_2026-07-30.json"` at **line 93** and reads `cost.pareto.honest_recost.*`; the
overthink figure now reads `finding1_corrected_2026-07-29.json` policy `P1_audit_best_matched`.
`paper/figs_final/fig_pareto.pdf` was regenerated at **2026-07-30 12:44**, with the old one preserved
as `fig_pareto_superseded_2026-07-08.pdf` **[M, verified this session]**.

> **The retrospective's own hole list and its stale-document register still describe this defect as
> live, and state "the PDF was deliberately NOT rebuilt". Both are stale as of 2026-07-30 12:44.**

**Two related propagation defects that remain open [M, verified this session]:**

- The 2026-07-27 professor deck and its builder **hard-code "15/20"**. In
  `paper/build_professor_html_2026-07-27.py` (modified 2026-07-29 23:09) the two live hard-codes are at
  **L118 and L127**; L37, L40 and L46 are comments, and L46 explicitly records the 17/20 correction.
  The rendered HTML `meetings/progress_report_professor_2026-07-27.html` carries them at **L216 and
  L223**. *(Earlier drafts cited L99/L108; those line numbers no longer hold those strings.)* The deck
  is left unedited as a **dated deliverable** — but **re-running that builder would republish a
  superseded count**. The script must be fixed before the next deck.
- That same builder makes **zero JSON reads** and contains **116 hand-typed four-decimal literals**,
  under a footer claiming *"All figures were read from real artifacts."* The headline row transcribes
  correctly on spot-check — but **nothing enforces that**.
- `artifacts/GENERALIZATION.md` was updated with the corrected cross-family table; the companion
  `generalization.json` was **not** annotated, so anything reading it programmatically still gets the
  old cells.

> **Lesson.** A claim is retired only when every artifact that *derives* from it is retired. Figures and
> decks read JSON; if the JSON is stale the figure is a live, publishable false claim. Keep a
> **stale-document register**, distinguish "frozen by design" from "not yet updated", and never
> hand-type a number into a deliverable that a script could read.

## 6.6 Process failures

The two below are recorded because compute time on a two-GPU box is the project's scarcest resource,
and both losses were self-inflicted. **The log evidence — timestamps, exit codes, the vLLM error text —
is exact. The operator's reasoning at the moment of each kill is not written down anywhere on disk, and
is not reconstructed here.** The narratives below are what the timestamps and surrounding log lines
imply; the researcher should confirm them before they are published as fact. **[U on intent]**

### 6.6.1 A healthy run replaced on a stale progress signal

`logs/verif_disjoint_master.log` records the disjoint-verifier retrain — the run that gated the entire
open-text claim — starting **twice, 105 seconds apart** (`00:45:01`, `00:46:46`). The first launcher
produced no output during that window because it was inside its silent GPU-wait loop.

The replacement then hit the *second* half of the same failure. Its readiness check was an
**instantaneous** free-memory reading:

```
00:52:47  GPUs free (26 MiB) after 360s
00:52:47  >> GEN slake_open_train (0/2976 done)
00:53:24  GEN slake_open_train rc=1 -> 0/2976 rows
00:53:24  ABORT: generation incomplete for slake_open_train
```

`logs/verif_disjoint_gen.log` gives the real state at that instant:
`ValueError: Free memory on device (46.74/79.14 GiB) on startup is less than desired GPU memory
utilization (0.88, 69.64 GiB)`. Another process still held ~32 GB. The "free" reading was **stale**.

**Cost.** The run did not restart until **09:56:52** — roughly **nine hours** of wall clock on the
project's highest-leverage open item.

**The fix, visible in the same log.** The successful launch replaced the point check with a
**sustained-free** check (`GPU idle streak reset (used=26 MiB, procs=2)` … `GPUs sustained-free
(26 MiB, 4 consecutive checks) after 5340s`), waited 89 minutes, and then ran cleanly end to end —
generation, judging, two LoRA trainings and six scoring passes — to
`16:01:15 === DISJOINT VERIFIER RETRAIN DONE ===`.

> **Lesson.** A health check must be **sustained and debounced**, never a single instantaneous reading;
> and *silence is not stalling*. A long-running job that legitimately waits looks identical to a hung
> one unless it heartbeats. Require **N consecutive** passing observations before acting on any
> resource signal.

### 6.6.2 A diagnosed-and-fixed run killed

The InternVL3-38B × MedXpert cell of the matched-prompt re-run failed twice on 2026-07-30
(`logs/medeval_direct_matched_master.log`, `01:47:08`–`01:56:18`, then `SKIPPED_AFTER_2_FAILURES`) with
vLLM's `The decoder prompt (length 20183) is longer than the maximum model length of 16384` — one
MedXpert item is ~20.2k tokens of image tiles.

The artifact's own incident note is blunt about the root cause: this was **not** the documented NCCL
hang, it was **deterministic**, so the retry reproduced it exactly — *"16384 was our error."* The fix
already existed on disk: the corresponding `*_reason` arm had hit the same limit and been re-run at
`MAX_MODEL_LEN=24000` (`runners/run_clean_latency_reruns.sh:23`). 24000 was the value **matched to the
reason arm** all along.

Then the fix was applied — and the fixed run was replaced too. At `09:58:08` a relaunch began, and the
log confirms it carried the fix (`max_seq_len=24000`, line 9380). It was loading a 38 B model across two
GPUs. At `10:03:29`, **5 minutes 21 seconds later**, a fresh master started and re-ran the same cell
from scratch, discarding the in-progress load. The cell finally completed at ~11:25.

**Two distinct failures in one incident:** (a) an automatic retry policy that re-attempted a
**deterministic** error identically, twice, instead of failing fast; (b) a **correctly diagnosed and
correctly fixed** run destroyed before it could produce anything, so the fix had to be paid for twice.

> **Lesson.** Classify failures before retrying: a deterministic error should **fail fast and
> escalate**, a transient one should retry. Propagate a per-cell configuration (context length,
> timeout) from whichever arm first needed it, so a matched comparison inherits the setting that made
> its counterpart work. And once a run is diagnosed and relaunched with the fix, **let it run** —
> model-loading silence is not failure.

### 6.6.3 A near-miss that cleared — recorded because it might not have

Full evidence in §5.7.2. `verifier_transfer_eval.py`'s `imgs_for(ds)` had **no `slake_open` branch**;
its `else` branch selected PathVQA images keyed by PathVQA row index. Discovering that in the file that
evaluates the verifier is alarming. Four checks established that **the bug was never executed** for
`slake_open`, and git history confirms no such branch existed in **any** revision: a **latent gap, not
a regression**. **Published numbers affected: NONE.** The branch was added anyway.

> **Lesson.** When you find a bug in an evaluation path, the question is not "is it wrong?" but **"was
> it ever executed, and by what?"** Prove it with reachability arithmetic and byte-level score
> comparison against the real producer — and publish the clean verdict with the same rigour as a
> damaging one.

## 6.7 What the correction record looks like in aggregate

| how the error was caught | count | examples |
|---|---:|---|
| running the real thing instead of a simulation / estimate | 4 | a simulated pairwise verifier overturned by a real forward pass; estimate-labelled-as-measurement; the disjoint retrain |
| testing a second family / second seed | 3 | the gate ranking reverses across families; a verifier "beats" → "ties" on seed 1 |
| a systematic audit replacing eyeballing | 3 | the PathVQA "artifact" hypothesis refuted by the researcher's own audit; §6.2.5; §6.3.1 |
| an adversarial audit the researcher demanded | 2 | MMMU (§6.2.1); the verifier (§6.2.2) |
| implausibility of a magnitude | 2 | IoU 0.022 → 0.232; a −16 J intercept |
| a documentation / three-way consistency audit | 3 | the efficiency headline; figure against table (§6.5); §6.2.3 |
| prior-art check | 2 | agreement gating is Agreement-Based Cascading (arXiv 2407.02348) |
| a control experiment | 2 | recoverability, not detection, predicts cascade quality |
| listing the disk instead of inferring it | 1 | §6.2.3 |

Two patterns stand out. **Corrections cluster where a comparison was assumed rather than constructed**
— prompts, splits, weightings, training pools. And **the project's own instruments caught most of
them**: the audits that produced the largest retractions were run *against* the project's headline, by
the project.

**One unresolved contradiction is recorded rather than corrected.** Whether verifier confidence is
really the best open-text gate is an **open inconsistency**: two runs of the same regime report
0.3832 against 0.3923 in one and 0.3965 against 0.3901 (+0.0062 [+0.0040, +0.0086]) in the other —
opposite in sign. No document reconciles them. This is not written up as a correction because it has
not been resolved. **[U]**

---

# 7. Traps in this research area

Generalised, so the next person can avoid them. Each is stated as the trap, the diagnostic that catches
it, and the reporting rule that prevents it recurring.

### T1. An output-format instruction can itself induce reasoning

**The trap.** Comparing a reasoning arm against a direct arm when the two arms *also* differ in how the
answer must be formatted. Asking for the answer in `\boxed{}` makes reasoning-tuned models emit hundreds
of tokens of chain of thought **with no reasoning trigger present** — MedVLThinker-32B **431–580**
tokens on 99–100% of items, InternVL3-38B **193–289** on 94–95%. The measured "reasoning gain" is then a
format gain. Here **0/9** trigger effects were significant while **3/9** format effects were.

**The mirror-image trap.** A "think" prompt with no actual trigger. Lingshu's published native-think
instruction was a pure format string; both arms emitted **3.0** tokens; seven cells were withdrawn.

**Diagnostic.** Log **mean and median generated tokens per arm, per cell**, and the fraction of items
that produced a trace. A direct arm emitting hundreds of tokens is not a direct arm; a reasoning arm
emitting three is not a reasoning arm.

> **Rule.** Every reasoning-versus-direct pair must be **format-matched AND token-audited**. Append a
> reasoning trigger to the retained format clause; never substitute one for the other. Decompose any
> published delta into `format` + `trigger` before naming a cause.

### T2. A matched prompt can suppress the behaviour under test

**The trap.** You correct a confound by matching the prompts — and the matched wording (`"Do not
explain"`) partly stops the model from reasoning at all. The matched arm's accuracy is then a
**mixture**, and its apparent improvement is **dilution**, not a corrected estimate. Here the matched
arms traced on only **25.5–71.1%** of items by dataset; conditioning on trace-fired restored the full
deficit (PathVQA −0.1576 on trace-fired against +0.0092 on no-trace).

**Diagnostic.** Measure the **treatment-fired rate** and report the effect **conditional on it firing**.

> **Rule.** Build a **2×2** (convention × treatment), not a single matched pair. It measures the
> confound's contribution instead of assuming it away, and it exposes suppression. Verify the identity
> `unmatched gap = treatment effect at fixed convention − convention effect` closes.

### T3. Benchmark answer keys derived from captions rather than images

**The trap.** Auto-generated VQA benchmarks generate questions *and* keys from figure captions. A
fraction of items are then not visual questions at all: the answer is in a different panel, a different
modality, a metadata field, or nowhere in the shown image. Models win those on **language prior**, not
perception. Here **46%** of the decision-relevant wins sat on such items, and a model that correctly
described the panel actually shown was scored **wrong**. Splits that "re-cut compound figures into
single panels and regenerate the QA" plausibly make this **worse**, not better.

**Diagnostic.** Hand-audit a stratified sample of the *decision-relevant* items with the source caption
in view, using an explicit rubric, and include an **agree-and-correct control stratum** — the contrast
between control-defect (28%) and win-defect (53%) is what makes the number interpretable. Then test
whether defects are **biased toward your wins** (here they were not: 53% against 60%, Fisher p = 0.487).

> **Rule.** Report a benchmark whose keys are caption-derived as **"agreement with the answer keys"**,
> not as accuracy, with the measured defect rate stated alongside. State the **achievable ceiling**
> (here 0.63–0.77) next to the measured score (0.54–0.57) so a one-point margin is read in context.

### T4. Verifier train/eval overlap

**The trap.** A grouped train/test split followed by evaluation on the **union** of both halves. Here
the verifier had seen **67–73%** of the items it scored, and the "held-out" 30% was not clean either:
**100%** of SLAKE's unseen questions used an image the verifier had trained on.

**What it inflates, and what it does not.** Contamination barely touched **ranking** (candidate AUROC
0.943 → 0.886) but destroyed **selection**: oracle conversion **0.589 → 0.203**, pooled gain
**+0.1041 → +0.0358** (2.9×). And it damaged the **efficiency** claim more than the accuracy claim —
escalation needed to hold parity went **3.97% → 26.9%**, turning "17.5% faster than one 32B forward"
into "5.4% slower."

**Diagnostic.** Three independent identity keys — **item id, question text, and image pixel hash** (md5
of decoded RGB, so re-encoded copies are caught). Report **oracle conversion**, not just AUROC.

> **Rule.** Retrain on a strictly disjoint pool, holding **candidates, judge labels and the
> random-number stream byte-identical**, so the only thing that differs is the verifier. Validate the
> swap harness by confirming `clean == contaminated` reproduces the published cells exactly. Report
> **both columns**, and treat a question-text-disjoint variant as a **lower bound** — on templated
> medical VQA it removes legitimate in-distribution coverage and conflates de-contamination with
> distribution shift.

### T5. Sample weighting lets one benchmark speak for the suite

**The trap.** Pooling items across benchmarks of wildly different sizes. Here **one** cell held
**79.2%** of the pool, and it was also the **lowest-escalation** cell (8.45% against up to 89.60%). The
reported suite accuracy and suite cost were, approximately, that one benchmark's accuracy and cost.
Equal weight per cell turned a 0.492× compute *saving* into a 1.196× compute *cost* and turned a tie
into a **significant loss** on the multiple-choice half.

**Diagnostic.** Report the **weight share** of every cell under each scheme, and a **leave-one-cell-out
range** for the headline. Under equal weight, "X% of the delta comes from N cells" becomes
arithmetically meaningless — every cell contributes exactly 1/k of its own delta — so concentration
must be re-expressed.

> **Rule.** Choose the weighting **before** reading the result, report macro **and** sample-weighted,
> and label what each answers: cost is **additive per query**, so sample-weighted cost is what you would
> pay on traffic like this suite, while macro cost tests whether the saving **generalises across task
> types**. **Never pair a macro accuracy with a sample-weighted cost.** And check whether your method's
> thresholds were *tuned* on the objective you are now *reporting* — if not, say so.

### T6. Checkpoints that do not record the prompt

**The trap.** Per-sample checkpoint rows store `idx, gold, pred, ok, parse_ok, logprobs, gen_tokens,
latency` — and **not the prompt**. Recovering which prompt produced an arm then requires tracing a
checkpoint directory back to a shell variable in a runner script. In this project that is precisely what
made the prompt confound **invisible for three weeks** and the audit expensive: every one of 35 cells
had to be mapped by hand.

> **Rule.** **Persist the prompt in every checkpoint row** — the full system and user strings, verbatim,
> plus the resolution cap, the sampling parameters and the harness revision. It costs bytes and saves
> weeks. Corollary: assert `mean_gen_tokens` in the checkpoint too, so T1 is detectable without
> re-reading raw outputs.

### T7. A fix is an experiment and needs its own control

Switching every family to its "native" reasoning recipe was a *correction*. It changed persona,
answer-format clause and image resolution simultaneously, and it introduced a worse confound than the
one it removed — two of the native recipes contained no reasoning trigger at all. The correction had to
be **reversed by a later correction**, and the original "foreign-prompt" measurement turned out to be
the more matched one.

> **Rule.** Change one thing. Diff the prompts before and after a "fix", token-audit the new arms, and
> re-run the affected comparison rather than assuming the fix strictly improved it.

### T8. An anomalously high score deserves an audit, not an assumption

The instinct on seeing a 7B beat its own 32B by 26 points was "we broke something." The audit that
cleared the pipeline was as valuable as one that found a bug — and only the pairing of **image
ablation** (0.827 → 0.593 without the image) with a **control model** (untuned base at 0.567 through the
identical harness) could distinguish "our wiring is broken" from "this model has seen this data".

> **Rule.** For any anomalous cell run: model-identity check, gold-subset check, prompt-leakage dump,
> independent re-score, **image ablation**, **control model**, and a cross-check that the same harness
> reproduces published numbers *elsewhere*. Then exclude on contamination grounds — and re-justify the
> exclusion whenever the weighting changes, because "it's only 0.35% of items" can become "it's 11.1% of
> the headline."

### T9. Provenance is per-field, and aggregates inherit the weakest

A pooled baseline containing three estimated cells was published as "(measured)" and propagated into
four documents, a slide deck, and a figure — and became one of four axes along which five different
values for the same row diverged.

> **Rule.** Flag `estimated` / `measured` **per cell**; propagate the flag into every aggregate; require
> any document quoting a number to quote its flag. Maintain a decode table whenever near-identical
> values circulate, listing every axis on which they differ.

### T10. Retire a claim in the artifact chain, not just in the prose

Figures and decks read JSON. A retracted claim survives in any figure whose input artifact still encodes
it — here a Pareto figure that disagreed with its own table on **both axes** and still showed the retired
dominance picture, and a slide builder with 116 hand-typed literals that would republish a superseded
count if re-run.

> **Rule.** When a claim is retired, grep for every artifact and script that derives from it, repoint or
> annotate them in the same pass, and keep a **stale-document register** distinguishing "frozen by
> design" (dated diaries) from "not yet updated". Preserve superseded figures under a
> `_superseded_<date>` name rather than deleting them. Never hand-type a number a script could read.

### T11. Health checks must be sustained; silence is not stalling

An instantaneous GPU-free reading fired while another process still held ~32 GB, aborting a gated run
and costing nine hours. A separate healthy run was replaced 105 seconds into its silent wait loop, and a
third — already correctly diagnosed and relaunched **with** the fix — was replaced 5 minutes into
loading a 38 B model.

> **Rule.** Require **N consecutive** passing observations before acting on any resource signal.
> Heartbeat from inside wait loops so waiting is distinguishable from hanging. Classify failures before
> retrying — deterministic errors should **fail fast and escalate**. And propagate a per-cell
> configuration from whichever arm first needed it.
---

# 8. What survives adversarial checking

## 8.1 The comparison against a reasoning-mode 32B

Against **always-32B-with-reasoning** — the naive way a practitioner would deploy a reasoning-capable
medical VLM — the method still wins clearly, and the margin *grows* under equal weighting even after
de-contamination **[M accuracy / Mo cost]**:

| operating point | Δ accuracy vs always-32B-reasoning (8-cell macro, clean verifier) | FLOP-eq (honestly re-costed) | batch-1 parallel latency | energy |
|---|---|---|---|---|
| compute-lean | **+0.0468 [+0.0353, +0.0583]** | 1.171× | **−89.0%** | **−85.9%** |
| accuracy-max (certified veto + learned deferral) | **+0.0601 [+0.0498, +0.0703]** | 1.396× | **−87.7%** | **−84.3%** |

The efficiency win is on **latency and energy, not FLOPs** — the method uses *more* multiply-accumulates
than a reasoning 32B (1.17–1.40×) while taking roughly a tenth of the wall-clock and energy, because the
reasoning baseline's cost is dominated by generating 100–320 tokens per query. **Never state this as a
compute saving.**

**Three caveats must travel with this claim.**

1. **The reasoning arm is prompt-unmatched.** With a matched reasoning prompt the same suite-level delta
   falls from +0.0245 to **+0.0180** sample-weighted, and the open-only delta from +0.2699 to **+0.1535**
   (§5.3.5). **The macro × clean-verifier × matched-reasoning combination has NOT been computed anywhere
   on disk.** The +0.0601 above should be read as an **upper bound** on the deployable version of this
   claim; the measured shifts are quoted rather than a macro number extrapolated. **[U on the combined
   figure]**
2. **The baseline barely reasons.** On ~90% of the pool the "always-32B-with-reasoning" arm generated
   3.0–3.3 tokens; PathVQA-closed has no reasoning dump at all and is imputed as reasoning = direct while
   still being charged the full reasoning cost. Genuine 32B reasoning exists on **4,345 of 42,224 items
   (10.3%)**. The honest re-costing partially compensates, but **this baseline remains the weakest input
   to the surviving claim** (§5.6.3).
3. **The load-bearing cell is PathVQA-open**, which is a non-random prefix slice of 1,500 of 3,357 items
   that over-samples a degenerate taxonomy family, is judged by an LLM whose cross-validation covered
   only SLAKE and VQA-RAD, and whose leave-one-out removal drops the macro delta from +0.0720 to +0.0318.

## 8.2 Reasoning hurts perception

This is the most robust result in the project, and the only one untouched by both the weighting change
and the verifier contamination. Full tables in §5.1–§5.3.

- **Multiple choice, 5 medical VLM families, prompt- and resolution-matched arms:** strictly worse in
  **17 of 20** perception cells, **14/20** with intervals excluding zero, pooled
  **−0.0401 [−0.0456, −0.0347]** over **30,250** paired samples; 19/20 no worse than +0.02. Three
  independent correction policies all give 17/20 — the previously published 15/20 was the *outlier*. Two
  non-medical peer architectures, already fully matched, give 7/8. The single genuine exception is
  MedGemma-27B on PathVQA, **+0.0413 [+0.0220, +0.0607]** on a fully matched pair.
- **Free text, decisive matched-prompt experiment** (Lingshu-32B only, same judge, same items,
  n = 2,345): holding the output convention fixed, the reasoning instruction costs
  **−0.2158 [−0.2354, −0.1962]**, while the prompt-style confound alone is worth
  **−0.0017 [−0.0111, +0.0077], not significant**. The entire deficit sits on trace-fired items. Per
  dataset the verdicts are SURVIVES (SLAKE-open), SURVIVES PARTIALLY (PathVQA-open), and **COLLAPSES**
  (VQA-RAD-open, n = 200, under-powered).
- **The corollary on reasoning-heavy benchmarks:** with the answer format matched, the explicit reasoning
  *trigger* is worth nothing — **0 of 9** sub-cells CI-significant, while **3 of 9** answer-*format*
  effects are. On the one cell where genuine 32B multiple-choice reasoning was measured against a fully
  format-matched direct arm, 100× the tokens bought nothing: MedXpert **0.3040 against 0.3005,
  Δ +0.0035 [−0.0185, +0.0250] n.s.** at 320.33 against 3.05 generated tokens.

## 8.3 The multiple-choice accuracy result

On the 5 multiple-choice cells — the half that never touches the trained verifier — **accuracy-max still
beats always-32B-direct: +0.0019 [+0.0014, +0.0024]** (8-cell macro convention) **[M]**. It is real, it
is CI-certified, and it is very small.

It is also almost entirely one cell, PMC-VQA, and that cell was audited item by item (§5.4). The audit's
verdict is **"the arithmetic survives, the construct does not"**: 53 of 100 decision-relevant wins sit on
defective items, but so do 30 of 50 losses (Fisher p = 0.487), so the defects are *symmetric* and the
delta survives correction (symmetric drop-defective **+0.0094 [+0.0004, +0.0183]** against a measured
+0.0135 [+0.0100, +0.0169]). What does not survive is the *description*: 46% of the wins are on items
where the gold key is wrong or the answer is not in the image, so the win is agreement with a
caption-derived key, not better image reading. Achievable accuracy on this benchmark is bounded at
roughly **0.63–0.77** and every system here scores 0.54–0.57.

## 8.4 The verifier as a ranker

De-contamination cost the verifier its role as a *selector* but not its role as a *ranker* **[M]**:

| quantity | contaminated (deployed) | clean L1 (no eval image/item) | clean L2 (lower bound) |
|---|---|---|---|
| candidate-level AUROC (pooled) | 0.9433 | **0.8856** | 0.7960 |
| selection gain over greedy (2,345 items) | +0.1041 [+0.0891, +0.1190] | **+0.0358 [+0.0213, +0.0503]** | −0.0119 [−0.0277, +0.0034] |
| share of oracle-of-8 headroom converted | 0.589 | **0.203** | −0.068 |

Ranking barely moved; **oracle conversion collapsed by ~3×**. The clean gain is significant **only on
PathVQA-open** (+0.0493 [+0.0320, +0.0667]); SLAKE-open (+0.0109) and VQA-RAD-open (+0.0150) both span
zero.

**The companion structured-output result.** A trained bounding-box verifier captured 40% (SLAKE organs)
and **77–78%** (MS-CXR chest X-ray pathology, n = 435, two seeds) of the oracle gap, against a
training-free spatial-consistency baseline sitting at or below greedy **[M]**. Its split is described as
"grouped by image", which would make it image-disjoint by the same definition used for L1 — **but it was
not re-audited by the 2026-07-30 disjoint retrain, and its contamination status is not independently
recorded.** **[U]**

## 8.5 The two documented limits

Both are characterisations, not methods, and both survive everything.

**The recoverability limit.** "Will the strong model fix *this* error?" is 0.5–0.6 AUROC from any cheap
signal. **Sixteen** independent mechanisms hit it (§3.4, claim (e)). A late peer-difficulty signal costing
four extra model forwards has much better recoverability AUROC (0.649 / 0.583 / 0.789 / 0.799 / 0.774
against the deployed margin's 0.407 / 0.236 / 0.472 / 0.670 / 0.450) yet buys **+0.0001** net gain at a
20% escalation budget on the cell carrying 79% of the sample-weighted pool.

**The selection limit.** A trained verifier converts only part of oracle-of-N — 74–82% by the project's
original efficiency measure (computed with the contaminated verifier), and only **20.3%** of the
greedy→oracle headroom once the verifier is de-contaminated. **Thirteen** independent attempts hit it,
killed three orthogonal ways: capacity (a 7× larger zero-shot verifier merely ties a small trained one,
Δ +0.005 [−0.023, +0.032], n = 600), compounding (diverse generation × pairwise comparison do not stack,
−0.0117), and pre-filtering (no filter beats both baselines).

**Behind both: the coverage limit.** Of 1,064 held-out open-text questions, **434 (40.8%) have no correct
answer anywhere in the 8-sample pool**, while the entire selection gap is 97 questions (0.0912). The
coverage limit is **4.5× larger** than the selection limit. Independently corroborated on the 2,345-item
open set: pooled oracle-of-8 is 0.6260 against greedy 0.4495, so 37.4% of items are unreachable by any
selector.

> Both mechanism counts (sixteen, thirteen) are the retrospective's own tallies and were not
> independently re-counted here. The two quotations of the selection limit use two different
> denominators and no document reconciles them (§3.4). **[U on the counts and their reconciliation; the
> individual measurements are documented]**

---

# 9. What did not survive

| claim as published | what replaced it |
|---|---|
| **"The method Pareto-dominates every fixed way of using the 32B"** (paper title, second contribution, README) | **Retired.** Under 8-cell macro + clean verifier, honestly re-costed, compute-lean and accuracy-max-fusion are *strictly dominated* by always-32B-direct on all four cost axes; only accuracy-max-veto stays on the frontier, and only by a **+0.0008** accuracy edge not distinguishable from zero. "Pareto-**optimal**" survives; "Pareto-**dominates**" does not. |
| **"The method beats a single 32B forward pass"** — +0.0107 [+0.0086, +0.0127] | **+0.0128 [+0.0056, +0.0200]** (macro only) → **+0.0008 [−0.0022, +0.0037] — a TIE** (macro + clean L1), at **1.74×** its FLOP-eq, +16.7% batch-1 latency and +101.0% energy. L2 lower bound: −0.0019 [−0.0055, +0.0014]. Decomposed: macro re-weighting alone *helped* (+0.0021); the clean verifier removed it (−0.0120). |
| **"Compute-lean matches the strong model at ~half its compute"** — 0.492× | **−0.0124 [−0.0188, −0.0060] — a significant LOSS** on all 8 cells, and **−0.0070 [−0.0126, −0.0017]** on the 5 multiple-choice cells, at **1.46×** the compute, +136.8% sequential latency and +80.2% energy. The compute ratio went 0.492× → 1.196× (macro) → 1.46× (macro + clean). |
| **Sample-weighted compute savings as a general claim** | The saving was real but **concentrated in the lowest-escalation cells**: escalation runs 8.45% to 89.60%, and PMC-VQA carried 79.2% of the sample-weighted average. At equal weight the multiple-choice escalation rate is **44.24%, not 16.22%**. Cost is additive per query, so the sample-weighted number is what you would pay on traffic resembling this suite — it is *not* evidence the saving generalises across task types, and it does not. **Report both, each labelled.** |
| **The open-text arm's magnitude** — selection gain +0.1041; arm accuracy 0.5642, "beats always-32B-direct"; 3.97% escalation; −94.8% batch-1 latency | Selection gain **+0.0358 [+0.0213, +0.0503]** (2.90× inflation); arm accuracy **0.5143** against always-32B-direct's **0.5168** — parity, not a beat; escalation to hold the same parity target rises **3.97% → 26.9%** sample-weighted / **48.7%** macro; latency against a single 32B forward goes **−17.5% → +5.4% / +27.2%**. Macro + clean, the open cells give accuracy-max **−0.0010 [−0.0090, +0.0067]** (tie) and compute-lean **−0.0214 [−0.0360, −0.0074]** (loss). **The efficiency claim was damaged more than the accuracy claim.** |
| **"A reasoning instruction improves accuracy on reasoning-heavy benchmarks"** | **Dropped — it is an answer-FORMAT effect.** 0/9 matched trigger effects significant; 3/9 format effects significant. MMMU gains decompose as MedVLThinker +0.103 = +0.062 format / +0.041 trigger (n.s.) and InternVL3 +0.124 = **+0.090 format (significant)** / +0.035 trigger (n.s.). **Lingshu-32B must not be cited as reasoning evidence at all**, and its quoted 1.2× reasoning:direct cost ratio is the ratio of two 3-token format prompts. |
| **MMMU +0.140 keep-7B win** | Excluded entirely after a contamination audit. Under macro this exclusion is consequential, not cosmetic: MMMU would carry 11.1% of the weight, and macro-9 against macro-8 would move accuracy-max against direct from +0.0128 to +0.0299. It must be defended on contamination grounds alone, with its size stated. |

**Two further honesty notes.** The method's thresholds were **calibrated against a pooled objective and
are now reported on a macro one** — the numbers are real, but they describe "a pooled-tuned method scored
on a macro metric". And **nothing in the final method has ever been executed end-to-end as a live
pipeline**: it is a CPU re-costing of saved per-sample dumps, with latency and energy from per-leg
batch-1 constants.

---

# 10. The honest deployment recommendation, and what the paper can claim

## 10.1 Deployment

**First, and worth more than the cascade: turn reasoning off.** Always-32B-direct scores **0.6567**
macro against always-32B-with-reasoning's **0.5974** — **[D] +0.0593** — at **665 ms against
6,291.2 ms** honestly re-costed and **127 J against 1,625.2 J**. That is a prompt/mode change, requires
no second model, no gate and no verifier, and it is larger than every method delta in this project
combined. Anyone with these two models should do this before anything else.

The exception is a genuinely reasoning-heavy multiple-choice workload with a reasoning-*tuned* model,
where getting the model to emit a trace does help (MedVLThinker-32B MMMU +0.103, MedXpert-Reasoning
+0.046) — but the operative lever is the answer format, so specify `\boxed{}` and audit
generated-token counts rather than trusting a "reason step by step" instruction.

**Second, use the cascade only inside its regime.** The cost model is transparent:
FLOP-eq = 1 + e × 4.57, sequential latency = 347 ms + e × 665 ms, energy = 45.8 J + e × 127 J, where *e*
is escalation-at-parity. **[D]** break-evens against always-32B-direct:

| escalation-at-parity *e* | verdict against a single 32B-direct call |
|---|---|
| **< ~48%** | cheaper on **every** axis — FLOPs, sequential latency, energy. Deploy the cascade. |
| ~48–64% | cheaper on FLOPs and energy, **slower** end to end. Deploy only if compute-billed. |
| ~64–78% | cheaper on FLOPs only. Marginal; probably not worth the operational complexity. |
| **> ~78%** | worse on everything. Do not deploy — call the 32B directly. |

> These three thresholds are **my arithmetic on the cited per-leg constants**, not a figure any artifact
> states. The underlying linear cost model was verified to reproduce the published per-cell FLOP and
> sequential-latency values for PMC-VQA, SLAKE-closed, VQA-RAD-closed and PathVQA-closed exactly; the
> break-even percentages themselves are **[D]**, and inherit the 4.57 ratio's ~7% uncertainty.

Measured escalation in this suite **[M]**: PMC-VQA 8.45%, SLAKE-closed 20.45%, PathVQA-closed 45.72%,
VQA-RAD-closed 56.97%, MedXpert-MM 89.60%. So the cascade is a clear win on the first two, marginal on
PathVQA-closed, and a loss on VQA-RAD-closed and MedXpert. **A deployment must measure its own
escalation-at-parity on a calibration fold and fall back to always-strong above ~50–60%** — this is the
missing guardrail. There is currently no degeneracy check, and MedXpert is the failure case:
**−0.0060 [−0.0120, −0.0005]** *below* the oracle-mode baseline at 5.095 compute units.

**Third, do not deploy the open-text best-of-N arm against a fast strong model.** With an honest verifier
it needs 26.9–48.7% escalation to reach parity, costs 2.8–3.8× a single 32B forward in FLOPs, and is
slower in wall-clock. If free-text answers are the workload and a 32B is available, call the 32B in
direct mode.

**Fourth, two practical caveats.** The per-benchmark policy router requires knowing which benchmark a
query came from *and* having labelled calibration data with strong-leg labels for it — no single frozen
threshold has ever been materialised for this cascade. And every method number is Lingshu-7B →
Lingshu-32B; **the gate ranking is known to reverse across families**, so the transferable artifact is a
**recipe** (pick the gate per family on a calibration fold), not a fixed gate.

## 10.2 What the paper can claim

**The defensible thesis, in one sentence:**

> *On medical visual question answering, chain-of-thought reasoning is a net accuracy loss on
> perception-style questions and a net cost disaster everywhere; a format-aware cascade over a 7B and a
> 32B model recovers most of a reasoning-mode 32B's deficit at roughly a tenth of its latency and
> energy — but it does not beat, and does not undercut, simply running the 32B once in direct mode.*

Supporting evidence, in order of strength:

1. **Reasoning hurts perception** — 17/20 cells, 5 families, pooled −0.0401 [−0.0456, −0.0347],
   n = 30,250; replicated on 2 non-medical architectures; confirmed on free text with a matched-prompt
   2×2 (−0.2158 [−0.2354, −0.1962] with a prompt confound of −0.0017, not significant).
2. **The apparent reasoning gain on reasoning-heavy benchmarks is an answer-format effect** — 0/9 trigger
   effects significant, 3/9 format effects significant, with generated-token audits showing the format
   alone induces 431–580 tokens.
3. **Against a reasoning-mode 32B** — +0.0468 to +0.0601 macro accuracy at −87.7% to −89.0% latency and
   −84.3% to −85.9% energy (1.17–1.40× FLOP-eq), stated as an upper bound pending the matched-reasoning
   recomputation.
4. **The two limits**, with 16 and 13 independent mechanisms respectively, plus the coverage budget.
5. **A small, CI-certified multiple-choice gain** (+0.0019 [+0.0014, +0.0024]) reported alongside its own
   construct audit.

**What the paper must not claim:**

- That the method Pareto-dominates any fixed use of the 32B, or beats a single 32B forward pass — it ties
  at 1.74× the cost.
- That it matches the strong model at half the compute — it is a significant loss at 1.46×.
- Any accuracy from one weighting paired with a cost from another, or any suite average without stating
  that PMC-VQA holds 79.2% of the sample-weighted mass.
- Any FLOP saving as the surviving efficiency claim. The surviving axes are latency and energy, against a
  reasoning baseline.
- That a reasoning *instruction* improves reasoning-heavy accuracy, or any Lingshu-32B
  reasoning-versus-direct number as reasoning evidence.
- That the PMC-VQA gain is a medical-visual accuracy improvement. Report it as agreement with a
  caption-derived key, with the 53% defect rate stated.
- That the open-text arm beats always-32B-direct, or that it escalates on ~4% of traffic.

## 10.3 What a reviewer should take from this

The positive result is now modest and honestly bounded: against the way a practitioner would naively
deploy a reasoning-capable medical VLM, a format-aware cascade delivers **+0.06 accuracy at roughly a
tenth of the latency and energy**; against the way they *should* deploy it — one 32B forward in direct
mode — the cascade **ties at 1.74× the cost**, and its cheaper setting loses. **A reviewer should not
accept the cascade as the contribution.**

**The negative and methodological results are the more valuable contribution.**

The scientific negatives are quantitative, replicated, and transferable. Two limits, each confirmed by 16
and 13 independent mechanisms, with a measured budget attached: the coverage limit is 4.5× the selection
limit, so anyone building test-time-compute systems for this domain should attack **generation before
verification**. A third result — that answer *format* determines whether routing signals work at all
(multiple-choice detection AUROC ~0.66–0.73 against open-text 0.866 on the same model, same images, same
questions, with open answers of median 1–2 tokens) — explains why the multiple-choice literature's
"confidence is saturated" conclusion should not be transferred to a generative deployment. And "reasoning
hurts perception" is, at 20 cells across 5 medical families plus 2 non-medical peers and a decisive
matched-prompt free-text experiment, one of the better-powered negative results available on medical VLM
test-time compute.

The methodological findings are, if anything, the most reusable output, because each was caught here
*after* it had already produced a published number:

- **Weighting.** A suite average in which one benchmark held 79.2% of the mass reported that cell's number
  as the suite's. Re-basing reversed a compute claim from 0.49× to 1.20× and turned a harmless tie into a
  significant loss. The diagnostic is heterogeneity in the gate's own firing rate (8.45% to 89.60%).
- **Verifier contamination.** Training on 67–73% of the scored items inflated the selection gain 2.9×,
  and — the non-obvious part — barely touched ranking (AUROC 0.943 → 0.886) while collapsing oracle
  conversion (0.589 → 0.203). *Report conversion, not AUROC, when you claim a selector works.*
- **Prompt matching and token audits.** Three separate published findings rested on reasoning arms that
  emitted 3 tokens and never reasoned, and a fourth on a direct arm that emitted 431–580 tokens and did.
  Prompts were not persisted in any checkpoint row, which made the defect invisible for three weeks.
- **Construct validity.** A CI-certified +0.0135 on a 33,430-item benchmark survived every statistical
  attack and still could not be described as an accuracy improvement, because 46% of its
  decision-relevant wins were on items whose answer is not in the image.
- **Simulation.** A confident simulated negative on pairwise verification was overturned the same day by
  one real forward pass (+0.036 [+0.016, +0.055]).

The most defensible thing in this record is the habit that produced §6 — publishing its own refutations,
including the ones that cost it a headline. Twenty-seven claims were retracted or downgraded, and the two
that mattered most were retracted *after* the paper was drafted around them. A reviewer should weigh that
as evidence about the reliability of what remains.

---

# 11. Unverified, unrecorded, and stale — the full register

Collected in one place so nothing in this document is read as better-supported than it is.

## 11.1 Not recorded anywhere in the repository

| item | status |
|---|---|
| The killing numbers for the two earliest pivots — question-aware visual-token pruning, and image-difficulty-driven compute | **Not recorded.** Only the qualitative verdicts survive (`CLAUDE.md` §2). The raw CSVs in `archive/image-difficulty/` were never mined for a figure. |
| Whether the paper was submitted | **Not recorded.** |
| Whether the MMMU exclusion was ever ratified by the researcher | **Not recorded.** |
| The operator's intent behind the two run kills in §6.6 | **Not recorded.** Timestamps, exit codes and error text are exact; the narrative is inferred and needs the researcher's confirmation before publication. |
| The pre-correction wording of the `parse_ok` claim (§6.1.2) | **Not preserved on disk**, and git holds no earlier version. |

## 11.2 Documented but not verifiable in this pass

| item | status |
|---|---|
| The **−29σ** luck-floor result that killed single-model routing | Quoted from `CLAUDE.md` §2 and `docs/archive_mcq/FINDINGS.md:70`; the script's output artifact was **not located** in the artifacts directory. |
| The 7B cost constants (347 ms / 45.8 J) and the verifier forward (175 ms / 25.3 J) | Labelled "measured batch-1" in code; the raw NVML log was not located (`logs/latency_opentext.jsonl` is gitignored). Provenance is a code comment. |
| The **4.57** FLOP-eq ratio | A hard-coded literal reproducing 32.0 B / 7.0 B; **no file derives it**, and an older document implies 4.34. ~7% margin on every compute-negative claim. |
| The best-of-N parallel latency **522 ms** (8 draws + 8 verifier forwards) | **Asserted, not measured**, and physically inconsistent with its own 568.8 J energy figure (implying ~1,088 W against ~132 W measured and a 400 W card TDP). No batch-8 measurement exists. Every open-arm parallel-latency number should be treated as unverified; an energy-consistent bound puts batched best-of-8 at ≥1.42 s. |
| The open arm's parallel latency generally | Assumes overlappable draws, never measured. Sequential latency is reported alongside everywhere. |
| The counts "sixteen mechanisms" (recoverability) and "thirteen attempts" (selection) | The retrospective's own tallies; not independently re-counted here. |
| The selection limit's two denominators ("74–82% of oracle-of-N" and "oracle conversion 0.589 → 0.203") | No document reconciles the two definitions. Both reported with sources. |
| The box-verifier result (MS-CXR 0.230–0.232, 77–78% of the oracle gap) | Reported on an image-grouped split, which would make it image-disjoint — but it was **not** re-audited by the 2026-07-30 disjoint retrain. Contamination status not independently recorded. |
| The macro-9 against macro-8 MMMU sensitivity figures | Quoted from `CLAUDE.md` §0; not re-derived directly from `macro_average_headline_2026-07-30.json` (only its `accuracy_levels` and `verdict` blocks were read). |
| The 40.8% coverage figure | Denominator confirmed (`perq_sc8.json` holds exactly 1,064 entries) and corroborated by an independent oracle-of-8 measurement; the per-question file itself was not re-analysed. |
| The LLM-judge protocol (`run_judge.py`) and the Claude-as-judge validation | Documented but **not re-run** in this pass. The cross-validation covers SLAKE and VQA-RAD only — **not PathVQA**, the load-bearing open cell. |
| Whether verifier confidence is the best open-text gate | An **unresolved contradiction** between two runs of the same regime (0.3832 against 0.3923 in one; 0.3965 against 0.3901, +0.0062 [+0.0040, +0.0086], in the other — opposite in sign). Not restated as settled anywhere in this document. |
| The macro × clean-verifier × matched-reasoning headline | **Not computed anywhere on disk.** The +0.0601 vs-reasoning figure is an upper bound. |
| That `test_clean.csv` IS the PMC-VQA paper's manually verified split | An **inference** from the paper's wording, the file name and the exact row count; the CSV carries no verification column. |
| An independent quantitative audit of PMC-VQA label error in the literature | **None exists** (searched; recorded as UNVERIFIED in the provenance document). The 53% / 60% / 28% defect rates have no external comparator. |
| The closed-answer spaces of SLAKE-closed and VQA-RAD-closed | Classified from `METHODS_MASTER.md` §14, which does not enumerate them; not verified against raw data. |
| `finding1_corrected.py` and `pmc_label_noise_audit.py` | Read only in part; the artifacts they emit were verified against their documented method, but neither script was re-executed. |

## 11.3 Known-stale documents, as of 2026-07-30

| document | how it is stale |
|---|---|
| `CLAUDE.md` (written 12:02) and `PROJECT_OVERVIEW.md` (12:27) | Both predate the clean-verifier artifacts (16:03 / 16:25). They still describe the disjoint retrain as "in flight" and still quote the **contaminated** macro headline (+0.0128 against always-32B-direct). **This document reports the retrain's result as the current state; those two entry documents need updating.** |
| `PROJECT_RETROSPECTIVE_2026-07-29.md` | Describes both the disjoint-verifier retrain and the open-text matched-prompt re-run as "in flight" in several places, although both artifacts now exist on disk. Its own §5.1 confirms the multiple-choice matched re-run landed; the open-text one is described as outstanding, but the artifact reports arms A and B complete with `arms_missing = {}`. **The artifacts are authoritative; the "in flight" language is stale.** |
| The same retrospective's hole list and stale-document register | Its "5 of 8 zero cells" heading contradicts its own body (the measured counts are 4 of 8 accuracy-max, 0 of 8 compute-lean); its Pareto-figure complaint and "the PDF was deliberately NOT rebuilt" statement were resolved at 2026-07-30 12:44; the register itself does not list the clean-verifier propagation and is therefore one correction behind. |
| `paper/build_professor_html_2026-07-27.py` (L118, L127) and `meetings/progress_report_professor_2026-07-27.html` (L216, L223) | Hard-code the superseded **15/20**. The rendered deck is a frozen dated deliverable; the builder must be fixed before the next deck. |
| `artifacts/generalization.json` | Not annotated with the corrected cross-family cells, though its companion `GENERALIZATION.md` was. |
| The dated progress diaries (`progress/progress_June_*.md`) | **Correctly frozen by design** — they are a historical record, not a current claim. |

## 11.4 Known unmatched axes and infrastructure risk

- **MedEvalKit local edits.** Two uncommitted edits (`utils/question_formats.py:11`,
  `utils/MMMU/data_utils.py:158`, both 2026-07-02) **replaced** rather than appended the reasoning
  trigger. Whether to revert the dependency and re-run is an open decision.
- **`EVAL_BATCH_SIZE` 250 against 2000** in the matched-direct run (OOM safety at TP=2). Affects MedXpert
  only, under greedy temperature-0 decoding — at most rare batch-composition tie-breaks, but a genuine
  unmatched axis.
- **MMMU's 5 of 150 "open" items** keep upstream's format-unmatched strings; they were audited as 0/5
  correct in all 9 arms, which is why the MMMU-MCQonly (n = 145) cell is quoted. Were those items ever
  scored non-zero, the MMMU deltas would move.
- **Multiplicity is uncontrolled** in two places: 18 policy-selection tests and 25 veto certifications.
  The prescribed Holm corrections **have not been run**, so this document cannot report what survives
  them.
- **Source control.** 44 untracked `.py` files include every file in the live headline chain, and
  `results/` and `MedEvalKit/` are both gitignored — **the paper's method, its inputs and its outputs
  exist on one disk.**

---

# 12. Open questions, ranked by value

**1. Generator work, not verifier work — the coverage bound.** Of 1,064 held-out open-text questions,
**434 (40.8%) have no correct answer anywhere in the 8-sample pool**, while the entire selection gap is
97 questions (0.0912). Independently confirmed on the 2,345-item open set: pooled oracle-of-8 is 0.6260
against greedy 0.4495, so 37.4% of items are unreachable by any selector. *Resolution:* a staged
generator ladder — self-distillation on judge-correct samples first (it moves the measured +0.116
greedy→best-of-8 gap from test time to train time at N = 1), with **oracle-of-8 monitored as the
stopping criterion**, since self-training narrows the output distribution. Generator ideas compete for
+0.408; verifier ideas compete for at most +0.091.

**2. The macro-objective refit.** The thresholds are tuned for a pooled objective and reported on a macro
one; the multiple-choice loss (−0.0070) may be an artifact of that mis-specification. *Resolution:*
CPU-only refit against an equal-weight iso-accuracy target over existing dumps. **Cheapest high-value
item in the list.**

**3. The noise ceiling — a program-level stop/go instrument.** If the strong model's per-item correctness
is Bernoulli with mid-range probabilities, the Bayes-optimal recoverability AUROC is low no matter what
features you build. *Resolution:* estimate per-item probabilities from replicates under nuisance
perturbations (option-order shifts *plus* temperature-ε sampling — greedy decoding is exactly
reproducible, so order alone is insufficient), with a split-half estimator. A ceiling near ~0.68 means
abandon gate work entirely; near ~0.85 means the band gate and the prefill probe are worth heavy
investment.

**4. The untested regime where best-of-N could still pay: an expensive or slow strong model.** Everything
measured here compares against a strong leg that answers in 665 ms for 4.57 FLOP-eq — the regime least
favourable to sampling. Against a strong leg that *is* slow, the same open arm is **−92.6% parallel
latency and −78.0% energy at 2.792× FLOP-eq**; against the fast one it is +16.3% parallel, +370.8%
sequential. Best-of-N pays exactly when one strong call costs more than N cheap calls *on the axis you
are billed on* — API pricing, a 70B-plus strong model, or a reasoning-mode strong leg. **This has never
been tested**, and is grounded here only in measured cost ratios already on disk. *Resolution, and it is
a prerequisite:* the batch-8 latency and NVML energy measurement that does not exist. The current
parallel-latency figure implies ~1,088 W of GPU draw against ~132 W measured at batch 1 and a 400 W card
TDP; an energy-consistent bound puts batched best-of-8 at ≥1.42 s, i.e. **~2.1× a 665 ms 32B forward
rather than the claimed 0.79×**. A ~30-minute run either rescues or kills the "best-of-N is latency-alive"
claim.

**5. Cross-family validation of the assembled cascade.** Every method number is one model family, and the
gate ranking is known to reverse across families. The MedEvalKit and open-text judge dumps for
MedVLThinker and InternVL3 are already on disk, so this is mostly offline re-costing.

**6. A single frozen policy.** One τ calibrated once and applied unchanged, one globally chosen policy,
Holm-corrected over the 18 policy-selection tests. Report honestly how much survives.

**7. PMC-VQA construct validity.** The clean, human-verified `test_clean.csv` (n = 2,000) is on disk and
the internal track already runs on it; the MedEvalKit track needs a one-line vendor patch. Pre-register
that n = 2,000 is **underpowered** against a +0.0135 effect (interval half-width ≈ 0.0141), so a null is
the expected outcome, not a refutation.

**8. Re-run the PathVQA-open prefix check.** PathVQA-open is a non-random prefix of 1,500 of 3,357 items
that over-samples a degenerate taxonomy family (0.632 against 0.562), and it is the load-bearing cell of
every vs-reasoning claim. The ten-minute check that this prefix is not topically biased has never been
done.

**9. Compute the macro × clean-verifier × matched-reasoning headline.** The three corrections have never
been combined; the surviving +0.0601 is an upper bound until they are.

**Blocked on infrastructure, kept separate from conceptual failures:** the OmniMedVQA 32B/38B strong leg
(deterministic two-GPU NCCL hang; ~2 days of mitigations failed; single-GPU impossible at 64 GB weights
on an 80 GB card — so no pooled 7-benchmark figure exists); InternVL3 faithful evaluation (harness
wrapper bug); INT4 latency/accuracy (a CDN outage stalled 2 of 6 quantized shards; the quantized
checkpoint on disk is the non-medical base model); lossless speculative decoding (rejected by vLLM 0.10);
two concurrent cheap legs (container cgroup OOM at ~245 GB RSS); and two data-absent items — a semantic
escalation cache (no image hash in any dump) and generated-token early exit (both legs emit ~3 tokens;
the real lever needs intermediate-layer logits nobody dumped).

---

*End of document.*
