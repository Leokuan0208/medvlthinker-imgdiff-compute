# 2026-08-17 — Cheap interventions on always-7B: one real free win, one benchmark artifact, and a baseline that was never greedy

Three attacks landed overnight (output-bias audit + correction, a self-consistency sweep, a literature
sweep). This entry is the **independent adversarial verification pass** over all three, plus the
combined 8-cell policy the round was supposed to produce. No GPU job ran; both A100s sat idle at 13 MiB
and nothing was launched. Everything below is CPU numpy over stored dumps, with every per-item vector
rebuilt from scratch — nothing imported from the round's own analysis code.

Doc: `results/cascade_methods/docs/current/CHEAP_INTERVENTIONS_2026-08-17.md`
Artifact: `results/cascade_methods/artifacts/cheap_interventions_verify_2026-08-17.json`

---

## 1. The headline

**Macro 0.607026 at 1.000 FLOP-eq**, from a published 0.5971 that was really costing **1.513738**.

It decomposes two ways and both land on the same endpoint:

- **+0.005242** [+0.004015, +0.006469] — the PathVQA-closed **prompt override**. A genuine model
  improvement, free, guardrail-clean, 1.8× the +0.0029 bar.
- **+0.004683** — a **correction to the baseline**, not an improvement: three of the eight cells were
  never greedy.

Together **+0.009926 [+0.004441, +0.015618] at 1.000×**, i.e. **−34% baseline compute and +0.0099
accuracy at the same time.** Guardrail clean, 0 losing cells. Permutation null on the per-cell selection
rule (paired sign-flip, nperm=1000): observed +0.010455 against a null mean of 0.000580, **p = 0.002,
z = 6.34, SURVIVES** — though honestly, the null's *maximum* reaches +0.011601, so the rule is not free
of selection risk, it just clears here.

## 2. What reproduced exactly

The PathVQA prompt fix reproduced to six decimal places against my own graders and my own judge join:
**judge +0.041939 [+0.032124, +0.051755], EM +0.041642 [+0.031826, +0.051457], gap +0.000297**, with
generated tokens 3.000297 → 3.000595. Trap (b) — a prompt change that moves the grader rather than the
model — is cleared: there is no length channel to move it through. The frozen open-text metric re-ran at
max abs deviation 3.5967e-07.

The PMC output-side correction also reproduced **bit-exactly** (0.575202, +0.023033 vs readout), and the
train/test disjointness is real: `Figure_path` intersection **0**, (`Figure_path`, `Question`)
intersection **0**.

## 3. But the PMC gain is the answer key, and the balanced-key test proves it

Rescoring on a gold-balanced subsample with nothing refitted:

| | answer-key skew | natural key | **balanced key** |
|---|---:|---:|---|
| PMC_VQA, marginal matching | 0.4729 | +0.020191 | **−0.000133 TIE** |
| MedXpertQA-MM, marginal matching | 0.0320 | −0.009500 | **−0.011047 LOSS** |
| PATH_VQA_closed, prompt override | — | +0.041642 | **+0.047101 WIN — larger** |

On a skewed key the correction gains everything on the natural key and exactly nothing on a balanced
one; on a near-uniform key it loses both ways. **Marginal matching is not a bias correction, it is a
readout of the benchmark's answer key, and its size measures that key's skew.** The prompt fix is the
only intervention in the round whose balanced delta is *bigger* than its natural one (by gold class:
gold=no +0.1151, gold=yes −0.0205 — real balanced accuracy).

This discharges the answer-letter-bias audit CLAUDE.md §0 says is owed on `test_2.csv`, and the answer
is unfavourable: **a free +0.023 sits on that split for anything that reads the option posterior.**

## 4. Three things I had to correct in the incoming material

**(a) The brief's "already settled" list contained the trap it asked me to hunt.** It lists the free head
on the 3 open cells at "+0.0529 judge". The repo refuted head-only the previous day (`43fe6fb`,
`CHEAP_VERIFIER_ON_7B_2026-08-16.md` §6). From `central_table_2026-08-16.json`: head-only is judge
+0.045203 **WIN** but EM **−0.006823 with 0 of 3 cells positive**, and its EM accuracy 0.448188 sits
**below** the always-7B EM baseline 0.455011. That is trap (d) verbatim. The fusion (+0.057996 judge /
+0.015352 EM) is what survives dual currency — but at **4.556 FLOP-eq** per open question it is not a
cheap intervention and does not belong in this table.

**(b) T=0.4's +0.0094 is a selection gain, not a standalone one.** Its artifact is
`decoding_ladder_cold_2026-08-14.json → PEAK.pairwise_between_rungs.T04_minus_T07r.judge.delta` — the
delta on the best-of-8 + selector endpoint. Sampling at T=0.4 and keeping one answer is a *loss* versus
greedy (macro −0.0062 at N=1). Excluded.

**(c) SLAKE is not evidence that the yes-bias is prompt-independent.** The round read SLAKE's flat result
as showing the bias is "not solely instruction-induced". But SLAKE's deployed instruction is
`"Answer the question using a single word or phrase."` and the intervention replaces it with
`"Please answer the question concisely."` — **neither names the answer space**, so on SLAKE the
intervention is a pure no-op (bias removed exactly 0.0000, Δ exactly 0.0000). It never tested the prompt
component. The honest statement: where the instruction *does* name the answer space, removing it strips
~92% of the yes-bias on **2 of 2** cells; the accuracy gain follows on **1 of 2**. VQA-RAD's Δ is exactly
0.0000 with a CI half-width of 0.0398 — wide enough to have caught a PathVQA-sized effect, and it did
not. That non-replication is the biggest threat to generalising this result, and it should be attacked
next.

## 5. Two defects, both bigger than the effects

**The prior-matching fit never converges.** `fit_shift_marginal` runs a constant lr=0.3 for a fixed 800
iterations with an `argmax` inside the objective. Over 42 (target × iters × lr) settings, pm_train
accuracy spans **0.568950–0.576159, spread 0.0072**; vs-readout spans +0.016781 to +0.023990. The
reported +0.023033 is `iters=800, lr=0.3` and sits near the top. The bootstrap CI prices item sampling
only and does not contain this. Sharpest way to say it: I ran the deliberate leakage control — fit the
shift on the **eval set's own gold marginal** — and it scores **+0.021687, inside the fit-noise band of
the legitimate train-fitted version.** Train and test key marginals differ by L1 = 0.011, so on this
cell the train/test discipline is a formality; the correction reads the same key either way.

**The published open-cell baseline is self-consistency@8.** Confirmed independently: `greedy_ok` equals
the modal-of-8 label on **2340/2345** items and slot 0 on only 2150/2345, and the repo's own code says
so verbatim at `gen_slake_open_bestofN.py:137-138` and `verifier_n_scaling.py:173`. Against a matched
true T=0 decode from the same June runner/engine/judge: pooled **0.461834 vs 0.449467, +0.012367
[+0.002559, +0.022601] WIN**; PATH_VQA_open +0.018667 [+0.006, +0.032] WIN. **CLAUDE.md §0 needs
correcting** — the always-7B open cells are 0.730233 / 0.490000 / 0.342667, macro **0.601783**, and the
published baseline costs **1.513738** FLOP-eq, not 1.0.

There is one real interaction, and it is negative: correcting the baseline shrinks the verifier's
measured gain by exactly the correction (+0.057996 → **+0.045629**). No published delta is wrong — every
open-cell arm used the same reference — but the verifier's headline and this baseline correction cannot
both be banked.

## 6. Method or collection?

**A collection, with one shared diagnostic that is the actual contribution.** The three results are three
different objects: a prompt-induced distortion in the model, a property of a benchmark, and a
mislabelled baseline. What unifies them is the *test* — every one of them moves the predicted marginal
toward the gold marginal, they look identical on a natural key, and they separate completely on a
balanced one.

One genuinely surprising thing did fall out. Self-consistency is a mode-seeking operator, so it
*amplifies* whatever prior the model already has — and **the same operator has opposite signs by
format**. On PMC-VQA (skew 0.4729, model under-uses the key) vote@8 is +0.0100 published-grader /
+0.0132 letter EM. On the three open cells (3,919 distinct answers, nothing to lean on) SC@8 is
−0.012367 against a true greedy decode. The published baseline was paying 2.37× to make three of its
cells worse.

## 7. Housekeeping

- New code: `src/analysis/cheap_interventions_verify.py`, `cheap_verify_part{B,C,D,E,F}.py`,
  `cheap_verify_fitstab.py`, `cheap_verify_consolidate.py`.
- New artifacts: `cheap_interventions_verify_2026-08-17.json`, `_cheapverify/part{A..F}.json`,
  `_cheapverify/fit_stability.json`.
- `MedEvalKit/` untouched. No visual LoRA scored under vLLM. `freeze_selector.py` not run. No GPU job.
- ⚠️ `ckpts/` and `logs/` still have zero tracked files. A `git push` does not protect
  `ckpts/output_bias/`, `ckpts/closed_as_open*/`, or the June T=0 dumps these numbers are read from.
- For the record, a bug in **my own** first pass: `sorted(glob("transfer_dump_<ds>*"))[0]` silently
  picked the **InternVL3-8B** and **MedVLThinker-7B** dumps instead of Lingshu-7B, inflating
  PATH_VQA_open to n=3357 and its delta to +0.0724. Caught by the item-count assertion I should have
  written first; fixed by pinning `_lingshu7b` and asserting 645/200/1500.

## 8. What I would do next

1. Fix the baseline label and cost in CLAUDE.md §0 and downstream. Zero GPU, highest value — every
   open-cell delta in the project is quoted against a mislabelled reference.
2. Replicate the answer-space prompt effect on more binary cells and a second model family. It is the
   only mechanism that survived a balanced key and it is currently 1 of 2 on accuracy.
3. Separate the answer-space instruction from the `\boxed{}` request
   (`MedEvalKit/utils/question_formats.py:29` confounds them, and `\boxed{}` is a known reasoning
   trigger in this repo). A 2×2 on PathVQA-closed settles the mechanism in one generation pass.
4. Pin `fit_shift_marginal` before any `test_2.csv` number from it is quoted again.
5. Stop rescoring MCQ option posteriors. Per-item contextual calibration loses (−0.047771 at 2.459×),
   the global content-free variant breaks the guardrail on MedXpertQA-MM, uniform targets lose
   everywhere. The family is closed.
