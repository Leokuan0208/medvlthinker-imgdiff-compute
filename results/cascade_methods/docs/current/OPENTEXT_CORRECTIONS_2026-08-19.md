# Corrections to the open-text pivot (2026-08-18 → 2026-08-19)

**Written 2026-08-19 after a three-way audit** (dataset decisions / numerical claims / experimental
design) commissioned because a dataset had been rejected on inspection rather than measurement.
Every correction below was reproduced on this machine before being written down.

**Read this before quoting any number produced on 2026-08-18 or 2026-08-19.** Several claims that
were reported as results do not survive. CLAUDE.md §9.6: corrections must propagate into the files
that carry the stale numbers, not live in a new document — the propagation list is §7.

---

## 1. THE DONOR CURVE HEADLINE IS MOSTLY A LABEL-VOCABULARY EFFECT

**Claimed:** "~100–200 in-domain questions buy +0.10 to +0.15 selection efficiency, ~10× any
objective or architecture effect. The 0.78–0.81 field constant is at least partly a
domain-coverage artifact."

**The control that was never run:** score each eval candidate by `P(y=1 | normalised answer
string)`, counted only on the donor rows. No image, no hidden state, no head — a counter over
answer text. Measured here:

| domain | head 0→max donors | string prior 0→max | string covers |
|---|---|---|---:|
| **radimagenet** | 0.6573 → 0.8996 (+0.2423) | 0.6024 → 0.8333 | **72.6%** |
| **pathvqa_open_train** | 0.4323 → 0.6099 (+0.1776) | 0.3584 → 0.5439 | **62.8%** |
| slake_open_train | 0.6834 → 0.7826 (+0.0992) | 0.6308 → 0.6918 | 8.4% |
| kvasir_open | 0.6211 → 0.7956 (+0.1745) | 0.5430 → 0.5703 | −29.1% |

At RadImageNet frac=0.10 the string counter reaches **0.7992** against the head's 0.8079 — the head,
with the image and 3,584 hidden dimensions, adds **0.0087**. So "100 questions buy +0.151" is almost
entirely the head learning which answer strings tend to be correct in a 105-gold closed vocabulary.

`head_donor_hetero.py:10-20` predicted this failure mode in prose ("part of the jump is LEARNING THE
LABEL SET") and then did not test it. The two domains led with in reporting — radimagenet, and
pathvqa which was explicitly presented as "the realistic end" — are the two the control eats.

**What survives.** On kvasir the effect is real and large: the string prior gets to 0.5703 while the
head reaches 0.7956, i.e. the prior explains *none* of it. slake is mostly real. So in-domain data
does help where the answer space is open; where it is closed and small, most of the measured gain is
vocabulary memorisation that a `Counter` reproduces.

**Retract:** the "field constant is a domain-coverage artifact" claim. With *full* donor data three
of four domains saturate at or below 0.78–0.81 (kvasir 0.7956, slake 0.7826, pathvqa 0.6099); only
radimagenet exceeds it, and radimagenet is the inflated one. Our own eval cells sit at 0.80–0.84
*with* in-domain data.

**Required:** every donor/LODO number must be reported as *head minus string prior*, not *head minus
frac-0*. The control is ~30 lines and zero GPU.

---

## 2. THE BIG-TRAIN "WIN" IS A TIE

**Claimed:** "More data flipped BCE from a guardrail-dirty tie into a CI-clean, guardrail-clean win,
+0.0196 [+0.0090, +0.0299]."

That was a **2-seed** run. The 8-seed run — the deployed readout — had already completed:

| | 2 seeds (reported) | 8 seeds (correct) |
|---|---|---|
| control sel_eff | 0.8120 | **0.8283** |
| bce/h256 vs control | +0.0196 **WIN** | **+0.0072 [−0.0021, +0.0166] TIE** |
| guardrail | clean | **dirty** (pathvqa −0.0090) |

The entire "win" was the 2-seed control being under-estimated by 0.0163. This project's own standing
caveat says seed depth is load-bearing and a one-seed read reports wins that do not exist; the
warning was quoted repeatedly and then not applied. **Source of record:
`head_eval_bigtrain_2026-08-19.json`.**

---

## 3. "EVERY BCE CONFIG BEAT EVERY BT CONFIG" IS FALSE

| scope | true? |
|---|---|
| all 74 configs | **NO** — 3 BCE configs sit below 35 BT configs |
| restricted to hidden ≥ 128 | **NO** — the set-relative BCE variants score 0.667 |
| A-series only (matched representation), hidden > 0 | **yes** — BCE min 0.69029 > BT max 0.68047 |

`A_objbce_h0` (0.66426) loses to `A_objbt_h0` (0.66626). The companion line "the BCE median beats
the BT best" is true at **+0.00112** — about one fifth of the smallest per-config seed sd in the
sweep. It is decoration, not evidence. This claim is also asserted in `head_sweep.py:360`.

### 3b. And the objective comparison is confounded anyway

`head_sweep._groups(need_both=True)` drops every question lacking both a positive and a negative
candidate: **BT trains on 12,244 of 31,498 rows (38.9%) and 2,391 of 6,029 questions (39.7%)**, while
BCE trains on all of them. BT also gets **3.26× fewer optimiser steps** at equal `epochs`
(`epochs×NG/64` vs `epochs×n/256`). The project's own data curve says the last data doubling is worth
+0.024–0.036 sel_eff — so a 2.57× data deficit alone predicts the entire +0.017 gap. **The objective
claim is not established.** Either match the budget (BT at ~98 epochs, BCE restricted to the same
2,391 questions) or drop it. Given every BCE-vs-BT delta on eval is a TIE, dropping it is the honest
outcome.

---

## 4. TWO VERIFICATION FAILURES

**"The AdamW bug crashed but did not corrupt — verified."** `head_lodo_2026-08-18.json` and
`_head_lodo_PREFIX_buggyadam_2026-08-18.json` are **byte-identical** (`md5 8c67443f…`). The LODO
re-run never happened; the campaign was killed and restarted before reaching it. A file was compared
against a copy of itself, matched 12/12, and the result was reported as verification. **All LODO
numbers still come from the buggy path.** Re-run launched as `head_lodo_FIXEDADAM_2026-08-19.json`.
(The eval half of that comparison *was* real — the control genuinely moved −0.0034.)

**`--audit_overlap` in `build_omnimed_opentext.py` is vacuous.** It hashes **raw file bytes**
(`:58-63`) against `img_md5` values that `extract_generator_hidden.py:70-74` documents as **decoded
RGB pixels**. Two different hash spaces: the guard can never fire. The "0 collisions" printed in the
build log is not evidence of anything. Proven on `radimg_0.png`: file-bytes `ece6f9ad…`,
decoded-pixel `3d230e79…`, stored value `3d230e79…`.

---

## 5. OMNIMEDVQA: WRONG DATA, AND IT CONTAMINATES THE RADIMAGENET CELL

Already retracted for answerability (§ the `--qtypes` filter): 51.7% of the built file was Disease
Diagnosis with 224 golds, 42.9% of those questions mapping to more than one gold, and 17.3%
binary-in-disguise. The justification given was **answer length** — the exact test
`OPENTEXT_CELL_SURVEY_2026-08-18.md` §2 had disowned one day earlier when correcting the Quilt
rejection ("the disqualifying property is the KIND of answer, not the count").

**The audit found a second, unaddressed problem.** `RadImageNet.json` is one of OmniMedVQA's 42
sources. The old draw was **2,567/15,831 = 16.2%** RadImageNet; the new `--qtypes` filter raises it
to **23.3%**, because Modality Recognition and Anatomy Identification are exactly the types
RadImageNet dominates (ultrasound and MR are 100% RadImageNet). Hashing all 55,443 OmniMedVQA
RadImageNet images against the eval cell: **62 exact decoded-pixel collisions = 6.2% of the
1,000-image radimagenet cell**, plus task and taxonomy overlap the pixel hash cannot see.

You cannot keep both radimagenet-as-eval-cell and OmniMedVQA-with-RadImageNet-in-it. **Exclude
`RadImageNet.json` from the draw, or retire the cell.**

---

## 6. SMALLER NUMERICAL CORRECTIONS

| claimed | correct | where |
|---|---|---|
| "2.2× the pool" | **1.917×** (60,384/31,498; 2.2× is the ratio to 68,539) | `aece16d` |
| "37,041 unused judged rows" | **28,886** — 8,155 of them are radimagenet, now the eval cell | `aece16d` |
| "6–11× any effect in the sweep" | **2.3–4.5×** against the sweep's real spread (0.653→0.697) | `8228ec9` |
| "in-domain rises and out-of-domain falls monotonically" | true only of the 3-point mean; **no single dataset is monotone in OUT** | `8228ec9` |
| round-3 0.70393 vs "round-1 base 0.69742" | mixes 20 seeds against 5; the matched 20-seed base is **0.69423** | `aece16d` |
| Quilt "1.28 clauses, 20.9% ≥2" | that is a **sentence** count; clause-level gives **2.02–2.55 mean, 54.9–66.8% ≥2** | `c42a238` |
| donor headline "+0.10 to +0.15" | the 4th curve (slake) was on disk uncommitted at **+0.0992**, and **+0.0573 at 112 q** | `aece16d` |
| ProbMed "6,303 images" | **6,301** | survey doc |
| "9 modalities" (OmniMedVQA) | 8 real + a junk `'?'` bucket of n=1 | survey doc |
| donor doc cites "pathvqa 312 questions / 504 golds" | those are the **eval** cell's; the curve runs on `pathvqa_open_train` = **1,207 / 1,905** | donor artifacts |

**The Quilt retraction was as sloppy as the rejection.** The original objection ("multi-clause
explanations where partial correctness is the norm") was never refuted — a clause-level split
supports it. The empirical test (campaign4) is still the right way to settle it, but the retraction
should not have been written as though the objection had been disproved.

---

## 7. DESIGN DEFECTS TO FIX BEFORE ANY OF THIS IS QUOTED AGAIN

1. **`radimagenet_cell.py:73-74` does not use the deployed readout.** It takes a **seed-mean logit**;
   the deployed convention (`genframe_selector.head_rank`) is per-seed within-pool `rank_avg` then
   mean. Mean-of-logits is not scale-invariant across seeds. The "+0.0065 TIE" and "sel_eff 0.6396"
   headlines rest on it, and three downstream docstrings quote them.
2. **All CIs are too narrow.** `paired_boot` resamples *questions* i.i.d., but the eval pool is 2,345
   questions over 528 images (4.44 q/image, max 21). At ρ=0.2 every CI widens ×1.30. Resample images.
3. **The "one-shot" eval is the fourth read.** Four artifacts score the same 2,345 questions with the
   same three arms; `head_eval_bce.py:154` still says "eval was read for the first time here". The CV
   winner (h1024) loses to the runner-up (h256) in all four, and naming the runner-up is selection on
   eval. The refit control's own sel_eff spans 0.7936/0.8004/0.8038/0.8283 across thread counts —
   larger than every delta being claimed.
4. **`rank_avg` itself was selected on eval** (`genframe_data.py:635-645` compares two eval readings,
   0.806540 vs 0.798365). A +0.008 choice made on eval, now called "the frozen deployed convention".
5. **`stage_curve` seeds its subsample on the hidden width** (`head_eval_bce.py:236`,
   `default_rng(1000*f + hid)`), so the capacity axis is unmatched at every `frac<1`; and the
   "capacity spans 0.0041" null is one seed per cell, below the documented ~0.005 seed spread.
6. **`head_omnimed_curve.py` is an unedited copy** — its `design` string and docstring describe
   RadImageNet, so the artifact would misattribute its own provenance.
7. `head_lodo.py` weights per-fold sel_eff by all questions, not recoverable ones. `head_sweep.fit`
   silently returns an **untrained** net when `_groups` finds nothing usable. `head_sweep.auroc` is
   pooled across questions, the wrong AUROC for a selection task. `build_omnimed_opentext.py` writes
   no image key at all unless `--audit_overlap`, so downstream image-grouped CV cannot group its
   2,083 shared-image rows.
8. **The kvasir_x1 cell is a held-out-IMAGES cell, not an out-of-domain one** — same dataset, same
   generator, same taxonomy, and 5,199 of its question texts are LLM paraphrases of the same
   originals. Endoscopy frames from one procedure carry near-duplicates across the id boundary that
   id-matching cannot see. Frame it accordingly, and run a perceptual-hash scan.

---

## 8. WHAT THE AUDIT CLEARED

Image-grouped splitting is genuinely honoured everywhere (train ∩ eval = 0 of 3,457/528; bigtrain ∩
eval = 0 of 4,229/528; radimagenet ∩ both = 0 of 1,000). The standardiser is fitted train-only in
every script. `add_setrel`'s group mean cannot cross a fold. The refit control is the right control.
`radimagenet_cell_2026-08-18.json` reproduces exactly from the raw judge files, including "3 of 1,422
wrong answers contain the gold string". `head_curve_bce`, the Kvasir-x1 contamination arithmetic
(1,052 excluded, 10,703/3,006 surviving, intersection 0), the Quilt length distribution, the +0.0419
PathVQA prompt-bias delta, and ProbMed's rejection all verify. OmniMedVQA's `gt_answer` **is** always
one of the listed options (0/88,996 violations).

---

## 9. THE PATTERN

Every error here is the same one. A proxy was substituted for the measurement:

- answer **length** for answer **kind** (OmniMedVQA, and the Quilt retraction)
- **2 seeds** for the deployed 8 (the big-train "win")
- a file compared to **its own copy** for a re-run (the AdamW verification)
- **file bytes** for decoded pixels (the contamination audit)
- **frac-0** for a real null (the donor curves)
- a **repo name** for the repo's contents (GEMeX)
- "downloaded" for **inspected** (VQA-Med 2019)

The proxies were cheap and the measurements were also cheap — the string-prior control is thirty
lines and no GPU, and it moves the headline result by a factor of three.
