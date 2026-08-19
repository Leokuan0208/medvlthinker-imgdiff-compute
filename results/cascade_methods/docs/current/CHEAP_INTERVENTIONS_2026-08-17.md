# CHEAP INTERVENTIONS ON LINGSHU-7B — 2026-08-17

> **VERDICT.** Across the 8-cell macro, cheap test-time intervention on Lingshu-7B is worth
> **+0.009926 [+0.004441, +0.015618] at 1.000× FLOP-eq against a baseline that was really costing
> 1.514×** — and **only +0.005242 of that is an actual improvement to the model.** The other
> +0.004683 is a **correction to the baseline**: three of its eight cells were never greedy, they were
> self-consistency@8. **This is a collection with one shared diagnostic, not a method.** Exactly one
> intervention survives a gold-balanced answer key (the PathVQA-closed prompt override), and it fails
> to replicate on the one other cell where it applies (VQA-RAD-closed, 0.0000).

**Verification artifact:** `results/cascade_methods/artifacts/cheap_interventions_verify_2026-08-17.json`
**Code:** `src/analysis/cheap_interventions_verify.py`, `cheap_verify_partB.py`, `..._partC.py`,
`..._partD.py`, `..._partE.py`, `..._partF.py`, `..._fitstab.py`, `..._consolidate.py`
**No GPU job ran.** CPU numpy over stored dumps. `MedEvalKit/` untouched. `freeze_selector.py` not run.
Nothing imported from the round's own analysis code — every per-item vector was rebuilt from raw dumps.

---

## 0. Null tests and controls, re-run independently

| control | result | verdict |
|---|---|---|
| **N1** frozen open-text metric (`src/training_methods/genframe_data.py`) | max abs dev **3.5967e-07**; sel_eff 0.775204, oracle@8 0.626013, greedy 0.449467, n=2345/1468 | **PASS** (matches the round's 3.6e-07) |
| **A1** PathVQA prompt fix re-derived from raw dumps with locally-written graders | judge **+0.041939** [+0.032124,+0.051755]; EM **+0.041642** [+0.031826,+0.051457] | **PASS**, reproduces to 6 dp |
| **Trap (b)** token audit on the prompt change | mean generated tokens **3.000297** (deployed) vs **3.000595** (open) | **CLEARED** — no length channel |
| **Trap (a)** PMC train/test leakage | `Figure_path` intersection **0**; (`Figure_path`,`Question`) intersection **0**; train n=152,603 / test n=33,430 | **CLEARED** as stated |
| **Trap (a)** cross-fit gap | transductive CV +0.021986 vs its leaky no-CV twin +0.019803 → gap **−0.001675** | no leakage inflation |
| **Trap (c)** permutation null on the per-cell selection | paired sign-flip, nperm=1000: observed **+0.010455**, null mean 0.000580, p95 0.004980, **max 0.011601**, **p = 0.002, z = 6.34** | **SURVIVES** (but see §5) |
| **Trap (d)** judge-vs-EM on every claimed arm | PathVQA prompt fix judge−EM = **+0.000297** | **CLEARED** — but see §1, the trap fired elsewhere |
| Open-cell baseline identity | `greedy_ok` == modal-of-8 on **2340/2345**, == slot-0 on 2150/2345 | **DEFECT CONFIRMED** |
| PMC slot exchangeability | per-slot acc spread **0.136** vs permutation null p95 **0.016**, **p = 0.000** (nperm 2000) | **DEFECT CONFIRMED** |
| PMC MCQ luck floor | vote@8 measured 0.5653 vs random-gold-from-same-marginal floor **0.3061**, clears by **+0.2593** | **PASS** |

*All from `cheap_interventions_verify_2026-08-17.json` §A–§G.*

---

## 1. ⛔ THE BRIEF'S OWN "ALREADY SETTLED" LIST CONTAINS THE TRAP IT ASKED ME TO HUNT

The round brief lists as settled: *"the free head on the 3 open cells (+0.0529 judge)"*. **That number
must not be used.** The repo refuted it the previous day (commit `43fe6fb`, 2026-08-16, and
`docs/current/CHEAP_VERIFIER_ON_7B_2026-08-16.md` §6, which titles the section
"⛔ THE REFUTED HEADLINE — do not ship head-only").

From `artifacts/central_table_2026-08-16.json` → `ARMS_vs_always_7B`, both currencies on identical picks:

| open-3 arm | judge | EM | macro-3 EM | open-q FLOP-eq |
|---|---|---|---|---:|
| **head-only, captured cap320 (FREE)** | +0.045203 [+0.029424,+0.060554] **WIN** | **−0.006823 [−0.022601,+0.008955]** | **−0.014256, 0 of 3 cells positive** | 2.370 |
| incumbent LoRA (deployed) | +0.035821 [+0.021322,+0.049893] WIN | +0.012367 [−0.001706,+0.026013] TIE | −0.001984 | 16.728 |
| prefix-shared LoRA | +0.035821 [+0.021748,+0.049893] WIN | +0.013646 [−0.000426,+0.027292] TIE | +0.001276 | 11.449 |
| **SHIPPED fusion (head + adapter)** | **+0.057996** [+0.043923,+0.072068] WIN | **+0.015352 [+0.001279,+0.029424] WIN** | +0.006023 | 4.556 |
| deployable free-head + prefix | +0.056290 [+0.042217,+0.070362] WIN | +0.017058 [+0.002985,+0.031130] WIN | +0.003729 | 4.556 |

Head-only's **EM selected accuracy is 0.448188, below the always-7B EM baseline 0.455011**
(`central_table_2026-08-16.json`, `free_head_2026-08-16.json:1889`). **This is trap (d) exactly: a
judge-only gain whose EM diagnostic points the other way.** The brief's +0.0529 matches none of the
rows above; the +0.0580 it attributes to the fusion is `SHIPPED_fusion`'s +0.057996, which is real and
does survive both currencies — **but at 4.556 FLOP-eq per open question, it is not a cheap
intervention** and does not belong in this round's table except as the incumbent.

**Second correction to the brief.** *"T=0.4 is the optimum and is free (+0.0094 judge on open)"* is a
**selection** gain, not a standalone one. Its artifact is
`decoding_ladder_cold_2026-08-14.json → PEAK.pairwise_between_rungs.T04_minus_T07r.judge.delta =
0.009382` — the T=0.4-minus-T=0.7 delta **on the best-of-8 + selector endpoint**. Sampling at T=0.4 and
keeping one answer is a **loss** versus greedy (macro −0.0062 at N=1,
`self_consistency_suite_2026-08-17.json` MACRO8). T=0.4 is a knob on the verifier arm; commit `7625d36`
further measured that only 2.4% of it survives the cascade's own escalation. **It is not a cheap
intervention on always-7B greedy and is excluded from the table below.**

---

## 2. THE TABLE THE PAPER NEEDS

One row per cell. Deltas are always against a control generated in the **same session**, never against
a published absolute. Judge = the primary currency; on the MCQ cells MedEvalKit's `judge_multi_choice`
reduces to exact letter match (verified fraction 1.000 over 33,430 items), and on the binary cells the
32B judge and exact match agree **5,204/5,204**, so "judge" and "EM" coincide there by measurement.

| cell (n) | always-7B greedy | prompt-side override | output-side answer prior | self-consistency@8 | delete baseline's own SC | **best SAFE pick** | judge Δ [95% CI] | EM diagnostic | FLOP-eq |
|---|---:|---|---|---|---|---|---|---|---:|
| **PMC_VQA** (33,430) | 0.5427 | n/a | **+0.023033** [+0.020162,+0.025845] *(vs readout, within-run)* | +0.0100 [+0.0048,+0.0152] *(n=6000)* | n/a | **none** ⚠️ | 0.0000 | — | **1.000** |
| **SLAKE_closed** (836) | 0.8254 | +0.003589 [−0.008373,+0.015550] TIE | TIE | +0.0036 TIE | n/a | none | 0.0000 | — | 1.000 |
| **VQA_RAD_closed** (251) | 0.7809 | **0.000000** [−0.039841,+0.039841] TIE | TIE | +0.0120 TIE | n/a | none | 0.0000 | −0.003984 | 1.000 |
| **PATH_VQA_closed** (3,362) | 0.8409 | **+0.041939** [+0.032124,+0.051755] **WIN** | +0.008031 *(oracle UPPER BOUND, not an arm)* | +0.0051 [+0.0003,+0.0098] WIN | n/a | **prompt override** | **+0.041939** [+0.032124,+0.051755] | **+0.041642**, gap +0.000297 | **1.000** |
| **MedXpertQA-MM** (2,000) | 0.2615 | n/a | **−0.012500** [−0.023,−0.002] **LOSS** | −0.0010 TIE | n/a | none | 0.0000 | — | 1.000 |
| **SLAKE_open** (645) | 0.7364 | n/a | not definable | +0.0031 TIE | −0.006202 [−0.023256,+0.010853] TIE | **true greedy** | −0.006202 | judge is the cell's currency | **1.000** (was 2.370) |
| **VQA_RAD_open** (200) | 0.4650 | n/a | not definable | −0.0100 TIE | +0.025000 [−0.010,+0.065] TIE | **true greedy** | +0.025000 | " | **1.000** (was 2.370) |
| **PATH_VQA_open** (1,500) | 0.3240 | n/a | not definable | −0.0038 TIE | **+0.018667** [+0.006,+0.032] **WIN** | **true greedy** | **+0.018667** | " | **1.000** (was 2.370) |

*Sources: prompt-side and self-consistency columns re-derived here (§A, §C, §D of the verification
artifact) and cross-checked against `closed_as_open_2026-08-16.json` and
`self_consistency_suite_2026-08-17.json`; output-side re-derived here (§B) against
`output_bias_correct_2026-08-17.json`; costs from `verifier_restructure_2026-08-16.json`
`Q1…per_config["count|default"].flopeq_rel_to_N1` (N=8 = 2.369969, MEASURED). The
PATH_VQA_closed output-side entry is the **oracle upper bound** from
`output_bias_audit_2026-08-17.json` `STEP1_bias_audit.ORACLE_upper_bounds.PATH_VQA_closed.
RECOVERABLE_scaled_to_the_whole_cell = 0.008031`, i.e. the most any re-ranking of that greedy string
could recover — **five times smaller than the prompt fix actually delivers.** You cannot get PathVQA's
win by rescoring; you have to change the prompt.*

**Settled entries carried in for completeness, none of which is cheap:** the shipped fusion verifier on
the 3 open cells, +0.057996 judge / +0.015352 EM at **4.556** FLOP-eq per open question
(`central_table_2026-08-16.json`); T=0.4 on that arm, +0.009382 judge
(`decoding_ladder_cold_2026-08-14.json`).

---

## 3. THE COMBINED RESULT

Interventions here act on **disjoint cells**, so they are **additive by construction** — there is no
sub-additivity to measure between them. (`hyperparameters_combined_2026-08-15.json` established the same
for its knobs, residual exactly 0.0.) The one **real** interaction is stated in §3.2.

### 3.1 The two honest framings

| policy | macro | Δ macro [95% CI] | macro-8 FLOP-eq | guardrail |
|---|---:|---|---:|---|
| **P0** baseline as published | 0.597100 | — | **1.513738** ⚠️ | — |
| **P1** PathVQA prompt override only | 0.602342 | **+0.005242** [+0.004015,+0.006469] | 1.513738 | clean, 0 losing cells |
| **P2 = THE RESULT** P1 + delete the baseline's own self-consistency on the 3 open cells | **0.607026** | **+0.009926** [+0.004441,+0.015618] | **1.000000** | clean, 0 losing cells |
| P3 = P2 + PMC answer prior **(NOT CLAIMED)** | 0.609905 | +0.012805 [+0.007424,+0.018464] | 1.000000 | clean on the natural key, **fails the balanced key** |

*`cheap_interventions_verify_2026-08-17.json` §E `POLICIES`. Macro CI by resampling items independently
within each moving cell, nboot=10000, seed 20260817.*

**Read it the other way and it is cleaner.** A *true* always-7B greedy decode on all eight cells scores
macro **0.601783** at **1.000** FLOP-eq (§C: +0.004683 over the published 0.5971, because the published
baseline's open cells are SC@8). Against **that** baseline, the only surviving cheap intervention is the
PathVQA prompt override: **0.601783 → 0.607026, +0.005242, at 1.000× exactly.** Both decompositions land
on the same endpoint, **macro 0.607026 at 1.000 FLOP-eq**, which is the number to quote.

**+0.005242 is 1.8× the +0.0029 bar. +0.009926 is 3.4×.**

### 3.2 The one real interaction, and it is negative

Correcting the open-cell baseline **shrinks the verifier's measured gain by exactly the correction**:
the fusion's +0.057996 judge was measured against `greedy_ok` = 0.449467 (SC@8); against the true greedy
0.461834 it becomes **+0.045629**. No published delta is wrong — every open-cell arm used the same
reference — but **the verifier's headline and this round's baseline correction cannot both be banked.**

### 3.3 Cross-fitting and the permutation null

- **Cross-fit.** `pm_train` is fitted on PMC-VQA `train_2.csv` only (intersection with `test_2.csv`
  measured **0** on both `Figure_path` and (`Figure_path`,`Question`)). Its nested 5-fold transductive
  twin gives +0.021986 against the leaky no-CV +0.019803, **gap −0.001675** — no leakage inflation.
  The prompt-side and baseline-correction interventions fit **nothing** and need no cross-fitting.
- **Permutation null** on "per cell adopt the largest-delta cheap correction whose paired 95% interval
  excludes 0": observed **+0.010455**, paired sign-flip null (nperm=1000) mean 0.000580, p95 0.004980,
  **p = 0.002, z = 6.34 → SURVIVES**. Stated honestly: the null's **maximum reaches +0.011601, above the
  observed value**, so this rule is not free of selection risk; it clears here, it would not clear on a
  smaller effect.

---

## 4. IS THIS A METHOD OR A COLLECTION?

**A collection — but one diagnostic separates the real fix from the artifacts, and it is the
contribution.** The unifying *hypothesis* the round was built on ("format-induced output biases are real
and correcting them is nearly free, on all eight cells") is **half right, and the half that is wrong is
the half that looked biggest.**

### 4.1 The diagnostic: does the gain survive a gold-balanced answer key?

Every intervention in this round moves the model's predicted marginal toward the gold marginal, so every
one is open to the same charge. Rescoring on a seeded subsample with an equal number of items per gold
class, **refitting nothing** (`cheap_interventions_verify_2026-08-17.json` §F):

**Output-side marginal matching, the two MCQ cells:**

| cell | answer-key skew (L1 from uniform) | natural-key Δ | **balanced-key Δ** |
|---|---:|---:|---|
| PMC_VQA (n=33,430) | **0.4729** | +0.020191 | **−0.000133 [−0.001938,+0.001653] TIE** |
| MedXpertQA-MM (n=2,000) | **0.0320** | −0.009500 | **−0.011047 [−0.013577,−0.008877] LOSS** |

**On a skewed key it gains everything on the natural key and exactly nothing on a balanced one; on a
near-uniform key it loses in both.** Marginal matching is not a bias correction at all — it is a readout
of the benchmark's answer key, and its size is a measurement of that key's skew.

**Prompt-side, the three binary cells:**

| cell | deployed prompt names the answer space? | yes-bias removed | natural-key Δ | **balanced-key Δ** |
|---|---|---:|---:|---|
| SLAKE_closed (355 y/n) | **no** | 0.0000 | 0.0000 | **0.0000** (proper no-op control) |
| VQA_RAD_closed (251) | yes | **0.0438** (+0.0478 → −0.0040) | −0.003984 | **−0.006970 [−0.016949,+0.004237] TIE** |
| PATH_VQA_closed (3,362) | yes | **0.0642** (+0.0693 → +0.0051) | +0.041642 | **+0.047101 [+0.045270,+0.049491] WIN** |

The prompt fix is the **only** intervention in the round whose balanced-key delta is **larger** than its
natural-key delta. By gold class on PathVQA: gold=no **+0.1151**, gold=yes **−0.0205** — it trades a
little sensitivity for a lot of specificity, which is a genuine balanced-accuracy gain.

### 4.2 What is shared, and what is not

- **Shared, and real:** the *diagnostic*. Prior-matching interventions and prompt-debiasing
  interventions look identical on a natural key and separate completely on a balanced one. This is a
  reusable test, and it is the round's most transferable output.
- **Shared, and surprising:** self-consistency is a **mode-seeking operator**, so it *amplifies* whatever
  prior the model already has — and the same operator has **opposite signs by format**. On PMC-VQA
  (skew 0.4729, model under-uses the key) vote@8 is **+0.0100 [+0.0048,+0.0152]** under the published
  grader and **+0.013167 [+0.008167,+0.018167]** under repaired letter EM; on the three open cells
  (3,919 distinct answers, no key to lean on) SC@8 is **−0.012367 [−0.022601,−0.002559]** relative to a
  true greedy decode. **The published baseline was paying 2.37× to make three of its cells worse.**
- **NOT shared:** there is no single mechanism. The PathVQA fix is a prompt-induced distortion; the PMC
  effect is a property of the benchmark, not the model; the open-cell result is a baseline mislabelling.
  Three different objects.

### 4.3 The mechanism, corrected

The round stated the yes-bias mechanism as *"induced by the answer space being given at all"*, citing
SLAKE as evidence that it is *not* solely instruction-induced. **That reading is wrong on the prompt
side.** SLAKE's deployed instruction is `"Answer the question using a single word or phrase."` and the
intervention replaces it with `"Please answer the question concisely."` — **neither names the answer
space, so on SLAKE the intervention is a pure no-op** (bias removed exactly 0.0000, Δ exactly 0.0000).
SLAKE never tested the prompt component; it only shows a yes-bias exists without an instruction.

The honest statement: **where the instruction names the answer space (PathVQA, VQA-RAD), removing it
removes 92%/92% of the yes-bias — 2 of 2. The accuracy gain follows on only 1 of 2.** VQA-RAD's Δ is
exactly 0.0000 with a CI half-width of 0.0398, i.e. **wide enough to have detected a PathVQA-sized
effect (+0.0419) and it did not.** That is a genuine non-replication, not merely low power, and it is
the single biggest threat to generalising this result.

---

## 5. TWO DEFECTS THAT MATTER MORE THAN THE EFFECTS

### 5.1 The prior-matching fit is not converged (NEW — found in this verification)

`output_bias_lib.fit_shift_marginal` runs a **constant** learning rate (lr=0.3) for a **fixed** 800
iterations with an `argmax` inside the objective. It is a stochastic-approximation iterate, not a fixed
point, and it never converges. Sweeping (target source × iters ∈ {200…6400} × lr ∈ {0.3,0.1,0.03}),
42 settings (`cheap_interventions_verify_2026-08-17.json` §G):

**pm_train accuracy spans 0.568950 – 0.576159, spread 0.0072; vs-readout spans +0.016781 to +0.023990.**
The reported +0.023033 is `iters=800, lr=0.3` — reproduced **bit-exactly** here (0.575202), and sitting
near the **top** of that range. The bootstrap CI [+0.020162,+0.025845] prices item sampling only and
does **not** contain this fit noise.

Sharpest consequence: I ran the **deliberate leakage control** — fit the shift on the **eval set's own
gold marginal** — and it scores **+0.021687 vs readout**, i.e. **inside the fit-noise band of the
legitimate train-fitted version.** Train and test key marginals differ by L1 = 0.011. **On this cell the
train/test discipline is a formality: the correction reads the same answer key either way.** It does not
change the verdict (both die on a balanced key), but no future `test_2.csv` number should be quoted from
this estimator without pinning `iters` and `lr` and reporting the sweep.

### 5.2 The published open-cell baseline is self-consistency@8 (CONFIRMED)

Independently reproduced. `greedy_ok` equals the modal-of-8 label on **2340/2345** and slot 0 on only
2150/2345. The repo's own code says so in two places, verbatim:

```
src/cascade_methods/gen_slake_open_bestofN.py:137-138
    modal_norm = Counter(norm(a) for a in preds).most_common(1)[0][0]
    greedy_ok  = int(aj[i].get(modal_norm, 0))
src/cascade_methods/verifier_n_scaling.py:173
    "greedy_repo": int(r["greedy_ok"]),   # NB: this is the MODAL-of-8 answer, i.e. SC@8
```

Against a matched true T=0 decode from the same June runner/engine/judge
(`ckpts/openvqa/cheap_lingshu7b/ckpt_<ds>_lingshu7b.judge.jsonl`): pooled **0.461834 vs 0.449467,
+0.012367 [+0.002559,+0.022601] WIN**; PATH_VQA_open **+0.018667 [+0.006,+0.032] WIN**.
**CLAUDE.md §0 and the round brief both need correcting:** the always-7B open cells are 0.730233 /
0.490000 / 0.342667 under a true greedy decode, macro **0.601783**, and the published baseline costs
**1.513738** FLOP-eq, not 1.0.

### 5.3 Carried forward, verified

- **PMC pools are not slot-exchangeable.** Per-slot accuracy spread **0.136** (slot 4 = 0.4453,
  slot 7 = 0.5813) vs null p95 0.016, **p = 0.000**; slot 4 emits letter **A** on **51.3%** of items
  against a 13.25% gold rate; corr(B+C mass, accuracy) = **0.9405**; mean generated tokens flat
  3.018–3.064. **Consequence: the N<8 points are not identified** — exact-subset vs first-N-prefix at
  N=1 gives **−0.015042 vs +0.022667, a sign flip of 0.038**. Only **N=8 is invariant** (+0.013167 both
  ways). Reproduced exactly.
- **The data-integrity incident** (`max_model_len=8192` → vLLM raising inside `LLM.generate`'s
  validation loop after earlier requests were queued → `zip(chunk, outputs)` pairing items with other
  items' answers) is recorded in `output_bias_correct_2026-08-17.json`; the pass was deleted and
  regenerated. Spot-checked: the regenerated PMC dump's N3 identity control sits at 0.000538 abs dev.

---

## 6. JUDGE vs EM — where they diverge, stated prominently

| arm | judge | EM | gap | reading |
|---|---|---|---:|---|
| **PathVQA prompt override** | +0.041939 | +0.041642 | **+0.000297** | **no divergence** — this is why it is credible |
| binary cells generally | — | — | — | 32B judge = exact match on **5,204/5,204** yes/no calls |
| MCQ cells | — | — | — | `judge_multi_choice` = letter EM, verified fraction **1.000** / 33,430 |
| **⚠️ head-only on the 3 open cells** | **+0.045203 WIN** | **−0.006823, 0/3 cells positive** | **0.052** | **SHARP — the arm is an artifact. Do not ship.** |
| self-consistency macro @ N=8 | +0.00237 TIE | +0.00513 WIN | 0.0028 | judge is the **conservative** currency here, so this is not paraphrase drift; take the TIE |
| T=0.4 inside the cascade | +0.00044 TIE | +0.00600 WIN | 0.0056 | judge taken per the user's ruling |

**Both known grader defects checked, neither is doing the work.** (1) `MedEvalKit/utils/utils.py:112`
fuzzy option-body fallback: 99.46% of deployed PMC responses are a bare letter, and the fallback **costs**
the deployed cell −0.009991 — it works *against* the intervention, and the debias claim is reported on
top of the readout arm anyway. (2) Judge leniency grows with length (`judge−EM` by word count on
SLAKE_open +0.0066 → +0.1154), but the greedy arm cannot harvest it: corr(pred length, judge) is
**negative** on all three open cells and the model is already shorter than gold. **No open-cell
intervention is proposed, so nothing here is exploited.**

---

## 7. WHAT IS LEFT, RANKED

1. **Fix the baseline in `CLAUDE.md` §0 and every downstream doc.** Always-7B open cells are
   0.730233 / 0.490000 / 0.342667 (true greedy), macro **0.601783**, cost **1.000**; the published
   0.5971 / "1.0 FLOP-eq" is SC@8 at 1.513738. **Zero GPU.** Highest value in the list, because every
   open-cell delta in the project is quoted against the wrong reference label.
2. **Replicate the answer-space prompt effect on a third and fourth binary cell, and on a second model
   family.** This is the only mechanism that survived a balanced key, and it is currently **1 of 2** on
   accuracy. VQA-RAD's flat 0.0000 at n=251 is the result to explain. Needs generation; cheap.
3. **Separate the two halves of MedEvalKit's instruction.** `MedEvalKit/utils/question_formats.py:29`
   confounds the yes/no answer space with the `\boxed{}` request, which this repo already knows is
   itself a reasoning trigger. A 2×2 (answer-space × boxed) on PathVQA-closed settles the mechanism.
   One generation pass, no training.
4. **Pin `fit_shift_marginal`** (add a convergence criterion or a decaying lr) before any `test_2.csv`
   number from it is quoted again, and publish the sweep alongside. **Zero GPU.**
5. **Report the balanced-key delta for every `test_2.csv` claim in the project**, starting with
   `armcombine_mcqonly_2026-08-11.json` (+0.0012 [+0.0009,+0.0015], 100% one cell). CLAUDE.md §0 says an
   answer-letter-bias audit is OWED there; §4.1 above discharges it, and **the answer is unfavourable**:
   a free +0.023 sits on that split for anything that reads the option posterior. **Zero GPU.**
6. **Do not** pursue more output-side rescoring on MCQ. Per-item contextual calibration loses
   (−0.047771 on PMC at a measured 2.459× prefill-inclusive cost); the global content-free variant wins
   on PMC only by overshooting the key and breaks the guardrail on MedXpertQA-MM (−0.0210
   [−0.0380,−0.0040]); uniform targets lose everywhere. The family is closed.
7. **Self-consistency: ship nowhere.** Macro TIE at N=8 for +137% compute; four CI-clean losses at N=2;
   and its only real per-cell win (PMC, +0.0100 published grader / +0.0132 letter EM) is on the split
   whose answer key §4.1 just showed is the thing being read.

---

## 8. Housekeeping

- New code: `src/analysis/cheap_interventions_verify.py`, `cheap_verify_partB.py`,
  `cheap_verify_partC.py`, `cheap_verify_partD.py`, `cheap_verify_partE.py`, `cheap_verify_partF.py`,
  `cheap_verify_fitstab.py`, `cheap_verify_consolidate.py`.
- New artifact: `results/cascade_methods/artifacts/cheap_interventions_verify_2026-08-17.json`
  (+ `_cheapverify/part{A,B,C,D,E,F}.json`, `fit_stability.json`).
- **No GPU job ran**; both A100s were idle at 13 MiB and nothing was launched. `MedEvalKit/` untouched.
  No visual LoRA scored under vLLM. `freeze_selector.py` not run.
- ⚠️ Standing: `ckpts/` and `logs/` have **zero tracked files**; a `git push` does not protect
  `ckpts/output_bias/`, `ckpts/closed_as_open*/` or the June T=0 dumps these numbers are read from.
- A bug found in **my own** first verification pass is recorded for the record:
  `sorted(glob("transfer_dump_<ds>*"))[0]` silently selected the **InternVL3-8B** and
  **MedVLThinker-7B** dumps instead of Lingshu-7B, inflating PATH_VQA_open to n=3357 and its delta to
  +0.0724. Fixed by pinning `_lingshu7b` and asserting the frozen item counts (645/200/1500).
