> # ⚠️ NUMERICALLY SUPERSEDED — annotated 2026-07-29
>
> **The mechanism described in this document is correct. Its headline numbers are not.** It was written
> before three things that changed them: the **oracle-mode-32B baseline** (2026-07-08 08:18), the decision to
> **exclude MMMU** after the contamination audit ("Variant B", 08:24–09:43), and the replacement of the
> **estimated** 32B-reasoning open-text cells with **measured** ones (10:41), plus the headline CI computed
> 2026-07-09.
>
> **Canonical values — MACRO, equal weight per reporting cell (Variant B = MMMU excluded, 8 cells, 1/8 each;
> re-based 2026-07-30):** baselines always-7B **0.5971**, always-32B-with-reasoning **0.5974**,
> always-32B-direct **0.6567**, oracle-mode **0.6573**. compute-lean **0.6600**, **+0.0626 [+0.0514, +0.0734]**
> vs reasoning / **+0.0033 [−0.0054, +0.0121] n.s.** vs direct, at **1.196x** a single 32B forward;
> **accuracy-max 0.6694, +0.0720 [+0.0614, +0.0824] / +0.0128 [+0.0056, +0.0200]**, at **1.410x**;
> accuracy-max-fusion **0.6661, +0.0686 / +0.0094 [+0.0013, +0.0176]**, at **1.435x**.
> Source: **`artifacts/macro_average_headline_2026-07-30.json`**.
>
> **Sample-weighted equivalents (previous convention; never mix with a macro number):**
> always-32B-with-reasoning baseline **0.5591**; compute-lean **0.5741, +0.0150 [+0.0107, +0.0192]** at 0.492x;
> **accuracy-max 0.5836, +0.0245 [+0.0216, +0.0274]** at 0.932x; accuracy-max-fusion 0.5862, +0.0271 at 1.250x.
> Sources: `artifacts/f8_mode_vsthink_ci.json`, `artifacts/opentext_32b_think_full.json`.
>
> ### ⚠️⚠️ TWO SETTLED CORRECTIONS, 2026-07-30
>
> **1 — "FLOP-NEGATIVE" IS RETIRED AS A SUITE-LEVEL CLAIM, and so is "Pareto-dominated".** This document's
> central efficiency result — that **F8 makes accuracy-max FLOP-negative (1.25x → 0.93x)** and that "**both
> Pareto modes are FLOP-negative (0.49x / 0.93x)**" — is a **sample-weighted** statement in which PMC-VQA held
> **79.2%** of the weight. **At equal weight per reporting cell no operating point is FLOP-negative:**
> compute-lean **1.196x**, accuracy-max **1.410x**, fusion **1.435x**. The **mechanism section is unaffected**
> — F8 really does cut **−1.449 FLOP-eq (−25%)** off the fusion variant, and that ordering survives every
> weighting (6.558 → 6.444 macro). What fails is the **sign**, i.e. the claim of being cheaper than one 32B
> forward. Also: compute-lean is a **significant LOSS on the 5 multiple-choice cells, −0.0070
> [−0.0126, −0.0017]** vs always-32B-direct (−0.0080 [−0.0137, −0.0024] vs oracle-mode).
> Mechanism of the reversal: escalation runs **8.45%** (PMC) to **89.60%** (MedXpert) — MCQ escalation
> **16.22% → 44.24%** at equal weight — and the three open cells cost the method **7.6–12.6 FLOP-eq** against
> the baseline's flat 4.57 while holding **37.5%** of the macro weight.
> **NUANCE:** macro *cost* answers a different question from sample-weighted cost. Cost is additive per query,
> so on traffic resembling this suite the **~0.49x** saving is what you would actually pay; macro tests whether
> the saving **generalises across task types** — and it does not. **Report accuracy on macro and BOTH cost
> numbers, each labelled.** *Defensible joint claim: large latency and energy savings against a reasoning
> baseline; compute savings that are real but concentrated on low-escalation multiple-choice traffic rather
> than uniform.* Retrospective §4, §10.1 C26, §10.2 X21.
>
> **2 — §4.x's "always-32B-THINK is Pareto-dominated" and the reasoning-side gains are re-attributed.** The
> matched-prompt re-run is complete (6/6 cells, 9 sub-cells, n = 145/1,446/554): **0/9** explicit-reasoning-
> trigger effects are CI-significant, **3/9** answer-**format** effects are. **Asking for the answer in
> `\boxed{}` is itself a reasoning trigger** (MedVLThinker 431–580 tokens on 99–100% of items, InternVL3
> 193–289 on 94–95%, with **no trigger present**; Lingshu never, 3–4 tokens). Drop "a reasoning instruction
> improves accuracy on reasoning-heavy benchmarks"; keep "getting a reasoning-tuned model to emit a trace helps
> substantially, **via the answer format**". **Lingshu-32B must not be cited as reasoning evidence at all.**
> **The gated-reasoning tier keeps its full value — only the attribution changes.** Source:
> `artifacts/medeval_matched_direct_2026-07-29.json`; retrospective §5.1, §10.1 C27, §10.2 X22.
>
> **⏳ OPEN — every open-text accuracy in this document is PROVISIONAL.** A **clean-verifier (disjoint-split)
> retrain is in progress** (`artifacts/verifier_disjoint_split.json`) and will determine whether the open-text
> accuracy claim is contaminated (~70% of evaluation items were in the verifier's training data; retrospective
> §7 hole 4). Not pre-judged.
>
> **Also corrected since:** the pooled baseline 0.5632 -> **0.5594 (full suite) / 0.5591 (Variant B)**; the
> footnote "open-text 32B-think accuracy is **estimated**" is **obsolete** (measured 2026-07-08); there is no
> Variant-B table in this file; and the MMMU keep-7B cell is **excluded**, not banked.
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

# METHOD_FINAL — the integrated format-aware router (2026-07)

> **What this is.** The project's **final, best, end-to-end method**, specified in enough detail to be the
> paper's Method + Ablations section. It is a **format router** (arm chosen from the *prompt*, never the gold
> answer) with an MCQ arm and an open-text arm, scored against the honest naive baseline **always-Lingshu-32B
> in THINK mode** ("just run the big *thinking* model on everything"). Everything below is a CPU re-costing of
> **saved per-sample dumps** — **no GPU, no new inference** — and every figure is copied from an artifact/code
> file that was read. Provenance is cited inline.
>
> **No-fabricated-numbers rule applies.** If a number here is not traceable to one of the sources below, it is a
> bug — report it.
>
> **Primary sources** (all under `results/cascade_methods/artifacts/` ← `src/cascade_methods/`):
> - **`method_final.json` ← `method_final.py` — ★ THE unified pipeline: both knob settings + INT4 + reconciliation,
>   all recomputed live (see the ★ section below). This is the canonical reproduction; the four below are its levers.**
> - **`method_final_v2.json` ← `method_final.py::run_v2()` — ★★ THE V2 pipeline: the same knob with two new levers
>   folded in — F8 (certified weak-veto) on the accuracy-max PMC cell and F10 (team-objective L2D) on the shared
>   open arm. Makes accuracy-max FLOP-NEGATIVE *(sample-weighted only — see the 2026-07-30 banner; at equal weight
>   per cell it is 1.410×)* and repairs the open-text losses (see the ★★ section below).**
> - `beat32b_more.json` ← `beat32b_more.py` — the offline source of the F8 and F10 mechanics (held-out 5-fold).
> - `integrated_method_vs_think.json` ← `integrated_method.py` — the compute-lean base router (fixed best-of-8).
> - `beat32b_fusion.json` ← `beat32b_fusion.py` — the **accuracy-max** operating point (PMC fusion).
> - `integrated_pandora_opentext.json` ← `integrated_pandora.py` — the Pandora adaptive-N open-text arm (FLOP-lean).
> - `quantized_strong_leg.json` ← `quantized_strong_leg.py` — the G3 INT4 strong-leg cost-mode (projected).
> - `escalation_levers.json` ← `escalation_levers.py` — the **G8 latency lever** (+ G5/G6 knobs).
> - `slake_open_bestofN.json` ← `best_method_lingshu.py`/verifier scoring — fills the SLAKE-open cheap leg.
> - `opentext_32b_think.json` — the **measured** always-32B-THINK open-text batch-1 cost + accuracy delta.
> - `iv3_38b_latency.json` — measured IV3-38B batch-1 latency/energy (peer strong leg).
> - `reframe_vs_bigthink.json` ← the ACC 5-family bake-off vs always-32B-THINK (regime framing + M1/M2/M3).
> - Prior FALC spec: `best_method_lingshu.py` → `best_method_lingshu_medeval.json` (the 32B-**no-think** baseline version).

---

## 0. TL;DR — one paragraph

> **⚠️ Read the 2026-07-30 banner at the top of this file first.** The "~16 % escalation", "~half its FLOPs" and
> "FLOP-negative" phrases below are **sample-weighted**; at equal weight per reporting cell MCQ escalation is
> **44.24 %** and neither operating point is FLOP-negative (**1.196× / 1.410×**). The mechanism is unchanged.

Detect the answer **format from the prompt**. **MCQ / closed** → run the cheap **Lingshu-7B no-think** leg, gate
on its **top1−top2 probability margin**, and **escalate the low-margin ~16 %** *(sample-weighted; 44.24 % at equal
weight per cell — the per-cell rates are 8.45 / 20.45 / 45.72 / 56.97 / 89.60 %)* to **Lingshu-32B *no-think*** (think
never helps Lingshu on this suite); **keep 7B on MMMU** (Lingshu-7B 0.80 > 32B-think 0.66, an anomaly the router
simply exploits); on the largest slice **PMC-VQA** an optional **slice-gated calibrated-confidence fusion** of the
two legs *beats* always-32B (+0.0135, held-out). **Open-text** → run **7B best-of-8 + a trained outcome verifier**
that picks the best candidate, gate on **verifier confidence**, escalate the low-confidence residual to 32B-no-think.
A **G8 parallel-prefill-prefetch** hides the 32B image-prefill under the 7B pass so no cell is slower than
always-32B-no-think. **Result (pooled, ALL-6 + 3 open-text, n=42 374): the method matches-or-beats always-32B-THINK
accuracy (0.575 vs 0.563, +0.012; up to +0.024 with PMC fusion) at 96 % lower batch-1 latency (460 ms vs 10 522 ms)
and ~half its FLOPs** on the MCQ arm. **V2 (§★★, 2026-07)** folds in two held-out levers that make it *cheaper and
stronger*: **F8** (certified weak-veto) turns the accuracy-max arm **FLOP-negative** (5.70 → **4.25 FLOP-eq**, 1.25× →
**0.93×** always-32B, retaining 70 % of the PMC beat) and **F10** (team-objective L2D on the shared open arm) **repairs
the two residual open-text losses** (SLAKE-open −0.008→+0.002, VQA-RAD-open −0.015→+0.005) and lifts PathVQA-open to
+0.086 — so ~~**both** Pareto modes are now FLOP-negative (0.49× / 0.93×)~~ **[STRUCK 2026-07-30: FLOP-negativity is
sample-weighted only. At equal weight per cell the modes cost 1.196× / 1.410× / 1.435× a single 32B forward. F8's
−1.449 FLOP-eq (−25 %) cut is real and survives every weighting; the *sign* of the ratio does not.]**

---

## ★ THE UNIFIED PIPELINE — `method_final.py` (one file, one knob) — reproduce everything

> **New (2026-07).** The four validated levers — base format router, PMC slice-fusion, Pandora adaptive-N
> open-text, and the INT4 strong-leg cost-mode — are now merged into **one reproducible file** with a single
> Pareto **`mode`** knob. It recomputes **every** number below **live** from the saved per-sample dumps +
> measured cost constants + held-out (5-fold cross-fit) calibration — it does **not** read any sibling
> `.json`. This supersedes running `integrated_method.py` + `beat32b_fusion.py` + `integrated_pandora.py`
> separately (they remain as the per-lever references).

```bash
cd ~/medvlthinker-imgdiff-compute
python3 src/cascade_methods/method_final.py      # → artifacts/method_final.json  (both modes + INT4 + reconciliation)
```

**The knob.** Both modes share the open-text arm (**Pandora adaptive-N + trained verifier → 32B-no-think**)
and **MMMU keep-7B**; they differ **only** in the MCQ perception arm:

- **`mode='compute-lean'`** — MCQ = `7B-nt + margin gate → 32B-no-think` cascade. FLOP-saving; PMC *matches* 32B.
- **`mode='accuracy-max'`** — MCQ = **F1 guardrailed slice router**: per benchmark route to the held-out
  paired-bootstrap-**certified** winner among {always-32B-nt, keep-7B, calibrated **confidence-advantage
  fusion**}. On **PMC-VQA** it certifies **fusion** (beats 32B); radiology/pathology-closed + MedXpert stay
  **always-32B** (guardrail-safe); MMMU keep-7B. Open text has no 32B option-confidence so it is **never
  fused** → Pandora on all open cells.
- **`int4=True`** (cost-mode, either knob) — re-cost the 32B strong leg with a projected AWQ/GPTQ-INT4 forward
  (see §6.4).

**Both headlines reproduce live as the two knob settings** (full suite, n=42 374, sample-weighted):

| knob | vs always-32B-**THINK** | vs always-32B-**no-think** | FLOP-eq | batch-1 latency (seq / parallel) |
|---|---:|---:|---:|---:|
| **compute-lean** (Pandora open) | **+0.0117** | +0.0017 | **2.244** (0.49×) | 578 / **469 ms** (−94 % / −96 %) |
| &nbsp;&nbsp;↳ *fixed-bo8 reference* | *+0.0118* | *+0.0018* | *2.538* | *460 / 460 ms* |
| **accuracy-max** (PMC fusion) | **+0.0238** | +0.0137 | 5.695 (1.25×) | 1050 / **666 ms** (−90 % / −94 %) |

The compute-lean **fixed-bo8 reference row is the literal +0.0118**; the deployed compute-lean point swaps the
fixed best-of-8 open arm for **Pandora adaptive-N**, which **holds that accuracy iso** (+0.0117, within the
5-fold held-out band) while cutting FLOPs 2.54→2.24. Both `always-32B` baselines: FLOP-eq 4.57; latency
THINK **10 521.6 ms** / no-think **665 ms**; pooled acc **0.5632** (think) / **0.5732** (no-think).

### ★.1 Final per-benchmark tables (both modes, all axes)

`d_think` = method − always-32B-think; `d_nt` = method − always-32B-no-think. Latency `seq / par` = batch-1
sequential (single-stream: adaptive draws one-at-a-time, fusion legs serial) / parallel (2-GPU co-resident:
best-of-N batched, fusion legs concurrent).

**compute-lean** (MCQ margin cascade + Pandora open + MMMU keep-7B):

| benchmark | n | 7B | 32B-nt | 32B-thk | policy | **method** | **d_think** | d_nt | FLOPs | lat seq / par |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| PMC-VQA *(`test_2.csv`)* | 33 430 | 0.5427 | 0.5518 | 0.5494 | margin-cascade | 0.5508 | +0.0014 | −0.0010 | 1.386 | 403 / 403 |
| SLAKE-closed | 836 | 0.8254 | 0.8589 | 0.8636 | margin-cascade | 0.8516 | −0.0120 | −0.0072 | 1.935 | 483 / 483 |
| VQA-RAD-closed | 251 | 0.7809 | 0.8526 | 0.8406 | margin-cascade | 0.8328 | −0.0079 | −0.0198 | 3.603 | 726 / 726 |
| PathVQA-closed | 3 362 | 0.8409 | 0.8891 | 0.8891‡ | margin-cascade | 0.8882 | −0.0009 | −0.0009 | 3.089 | 651 / 651 |
| MedXpert-MM | 2 000 | 0.2615 | 0.3065 | 0.3040 | margin-cascade | 0.3005 | −0.0035 | −0.0060 | 5.095 | 943 / 943 |
| MMMU-Medical | 150 | 0.8000 | 0.6333 | 0.6600 | keep-7B | 0.8000 | +0.1400 | +0.1667 | 1.000 | 347 / 347 |
| SLAKE-open | 645 | 0.7364 | 0.8186 | 0.6236§ | pandora-N + verifier | 0.8093 | +0.1857 | −0.0093 | 7.622 | 1906 / 627 |
| VQA-RAD-open | 200 | 0.4650 | 0.6000 | 0.4800§ | pandora-N + verifier | 0.5950 | +0.1150 | −0.0050 | 8.391 | 2124 / 605 |
| PathVQA-open | 1 500 | 0.3240 | 0.3760 | 0.2460§ | pandora-N + verifier | 0.4520 | +0.2060 | +0.0760 | 12.598 | 3100 / 759 |

**accuracy-max** (MCQ F1 slice router + Pandora open + MMMU keep-7B); open cells identical to above:

| benchmark | n | policy (F1 certified) | **method** | **d_think** | d_nt | FLOPs | lat seq / par |
|---|---:|---|---:|---:|---:|---:|---:|
| PMC-VQA *(`test_2.csv`)* | 33 430 | **fusion (F3 conf-adv)** | **0.5653** | **+0.0159** | **+0.0135** | 5.570 | 1012 / 665 |
| SLAKE-closed | 836 | always-32B-nt | 0.8589 | −0.0047 | −0.0000 | 4.570 | 665 / 665 |
| VQA-RAD-closed | 251 | always-32B-nt | 0.8526 | +0.0120 | −0.0000 | 4.570 | 665 / 665 |
| PathVQA-closed | 3 362 | always-32B-nt | 0.8891 | −0.0000 | −0.0000 | 4.570 | 665 / 665 |
| MedXpert-MM | 2 000 | always-32B-nt | 0.3065 | +0.0025 | +0.0000 | 4.570 | 665 / 665 |
| MMMU-Medical | 150 | keep-7B | 0.8000 | +0.1400 | +0.1667 | 1.000 | 347 / 347 |
| SLAKE / VQA-RAD / PathVQA-open | 2 345 | pandora-N + verifier | *(as above)* | | | | |

‡ PathVQA-closed has no 32B-think dump → think = no-think. § open-text 32B-think acc is **estimated**
(judged 32B-no-think + measured modal think-delta −0.195/−0.120/−0.130).

### ★.2 Final pooled tables — {compute-lean, accuracy-max} × {vs think, vs no-think} × {seq, par, FLOPs}

| mode | pool | n | method | **d_think** | **d_nt** | FLOP-eq (×32B) | lat seq (−vs think) | lat par (−vs think) | macro d_think |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **compute-lean** | full suite | 42 374 | 0.5749 | **+0.0117** | +0.0017 | **2.244** (0.49×) | 578 ms (−94 %) | **469 ms** (−96 %) | +0.0693 |
| | MCQ only | 40 029 | 0.5756 | +0.0011 | −0.0008 | 1.739 (0.38×) | 454 ms (−96 %) | 454 ms (−96 %) | +0.0195 |
| | open only | 2 345 | 0.5625 | +0.1927 | +0.0456 | 10.871 (2.38×) | 2688 ms (−74 %) | 710 ms (−93 %) | +0.1689 |
| **accuracy-max** | full suite | 42 374 | 0.5869 | **+0.0238** | **+0.0137** | 5.695 (1.25×) | 1050 ms (−90 %) | **666 ms** (−94 %) | +0.0747 |
| | MCQ only | 40 029 | 0.5883 | +0.0139 | +0.0119 | 5.392 (1.18×) | 954 ms (−91 %) | 664 ms (−94 %) | +0.0276 |
| | open only | 2 345 | 0.5625 | +0.1927 | +0.0456 | 10.871 (2.38×) | 2688 ms (−74 %) | 710 ms (−93 %) | +0.1689 |

> **⚠️ Do NOT read the `macro d_think` column here as the 2026-07-30 canonical macro.** These are the *old*
> macro values: **full suite (MMMU kept, 9 cells)** and with the open-text 32B-think cells **estimated**, not
> measured. The canonical macro (Variant B, 8 cells, measured) is **compute-lean +0.0626 [+0.0514, +0.0734]**
> and **accuracy-max +0.0720 [+0.0614, +0.0824]** (`artifacts/macro_average_headline_2026-07-30.json`). Note
> also that this table's own `macro d_think` column already showed the pooled/macro divergence (+0.0117 vs
> +0.0693) more than three weeks before the reporting convention was changed — nobody acted on it.
> **The FLOP columns in this table are sample-weighted throughout.** Macro FLOP-eq: compute-lean **5.465
> (1.196×)**, accuracy-max **6.444 (1.410×)**, fusion **6.558 (1.435×)**.

Baselines pooled: **always-32B-THINK** 0.5632 @ 4.57 FLOP-eq / 10 521.6 ms; **always-32B-no-think** 0.5732 @
4.57 FLOP-eq / 665 ms. Both modes are **faster than always-32B-no-think on parallel latency** and **≥ 90 %
faster than always-32B-THINK** on both latency accountings.

### ★.3 Reconciliation — the +0.0118 and +0.0238 are ONE method at two knob settings

The `integrated_pandora.py` agent flagged that the task quoted **+0.0238 / +0.0138** while its live pipeline
reproduced only **+0.0118 / +0.0018**. **Not a contradiction:** +0.0118 is *compute-lean* (base margin
router) and +0.0238 is *accuracy-max* (PMC slice-fusion). The gap was structural — **fusion lived in a
separate script** (`beat32b_fusion.py`) that the Pandora pipeline never included. `method_final.py` merges
them; the reconciliation is now computed **live** (`method_final.json:reconciliation`):

- **Only the MCQ arm differs** between the two knobs; the open (Pandora) arm is byte-identical. Of the MCQ
  cells the F1 guardrail deviates from always-32B on exactly **PMC** (→ fusion) and **MMMU** (keep-7B, but
  MMMU is keep-7B in compute-lean too, so it does **not** move the delta).
- **PMC is the load-bearing cell.** Fusion lifts PMC acc **+0.0145** (0.5508→0.5653) on **`test_2.csv`**. Because PMC is
  **n=33 430 = 78.9 %** of the pooled 42 374 samples, that one cell adds **0.789 × 0.0145 = +0.01142** to the
  pooled sample-weighted accuracy. The four non-PMC closed cells snapping margin-cascade → always-32B add a
  further **≈ +0.0006**. Together: pooled d_think **+0.0117 → +0.0238**.
- **Why sample-weighted moves but macro does not.** Macro weights all 9 benchmarks equally, so PMC's +0.0145
  contributes only /9 (+0.0016); macro was already dominated by MMMU (+0.14) and open text (+0.19), so it
  barely moves (**+0.0693 → +0.0747**). Sample-weighted is PMC-dominated, so fusing that cell ~**doubles**
  the vs-think delta and ~**8×**es the vs-no-think delta.
- **Bonus: the SLAKE-open inconsistency is gone.** `beat32b_fusion.py` scored SLAKE-open with the
  greedy+seqlogprob FALC *fallback*; the unified method uses **bo8+verifier (Pandora)** for SLAKE-open in
  **both** modes. That is why unified accuracy-max FLOPs are **5.695** (Pandora open) rather than the
  separate-script **5.751** (fixed-bo8 open) at the same accuracy — a cleaner, cheaper method.

---

## ★★ V2 (2026-07) — fold in F8 + F10 to make the method *cheaper* AND *stronger*

> **New (`method_final.py::run_v2` → `method_final_v2.json`).** Two levers from `beat32b_more.py` are folded into
> the *same* unified pipeline **without changing the v1 output** (`run()` → `method_final.json` is still
> byte-identical; `run_v2()` writes the separate v2 artifact). Both new mechanics stay **held-out (5-fold
> cross-fit)**. One command now writes **both** artifacts:
> `python3 src/cascade_methods/method_final.py` → `method_final.json` (v1) **and** `method_final_v2.json` (v2).
>
> - **F8 — certified high-precision weak-veto** **replaces the F3 confidence-advantage fusion** in the
>   **accuracy-max MCQ/PMC** cell. F8 runs the 7B on every item, then in each calibration-CERTIFIED
>   (dataset × 7B-conf-bin) cell where the 7B's **Wilson-lower-bound precision ≥ the 32B accuracy** it **vetoes the
>   32B** (keeps 7B → 7B-only → cheaper); elsewhere it takes the 32B. One-sided ⇒ **never-worse guardrail by
>   construction**. This flips accuracy-max from **FLOP-POSITIVE (1.25×)** to **FLOP-NEGATIVE (0.93×)**
>   *(sample-weighted; at equal weight per cell it moves 1.435× → 1.410×, i.e. a real cut but still FLOP-positive
>   — 2026-07-30)*.
> - **F10 — learning-to-complement (team-objective L2D rejector)** **replaces the open-text arm's
>   parity-targeting τ escalation gate** (shared by **both** modes). A learned rejector over 7B-side open-text
>   features (verifier-score max/range/mean/std, #unique preds, self-consistency, seqlogprob), tuned to the
>   **TEAM-accuracy** objective, decides escalate-to-32B vs keep-7B-best-of-N per item; **Pandora adaptive-N still
>   sets the cheap draw count** (so the FLOP model is unchanged, only the escalation set + delivered accuracy
>   come from F10). It **repairs the two residual open losses** and **lifts** the PathVQA-open win.

### ★★.1 Headline — V2 vs V1 (full suite, n=42 374, **sample-weighted — pre-seam**)

| knob | vs 32B-**THINK** | vs 32B-**no-think** | FLOP-eq (×32B) | lat seq / par | FLOP-neg? |
|---|---:|---:|---:|---:|:--:|
| compute-lean **v1** (Pandora-τ open) | +0.0117 | +0.0017 | 2.244 (0.49×) | 578 / 469 ms | ✅ |
| **compute-lean v2** (F10 open) | **+0.0123** | **+0.0023** | **2.238** (0.49×) | 577 / **468 ms** | ✅ |
| accuracy-max **v1** (F3 fusion, Pandora-τ open) | +0.0238 | +0.0137 | 5.695 (**1.25×**) | 1050 / 666 ms | ❌ |
| **accuracy-max v2** (F8 veto + F10 open) | **+0.0212** | **+0.0112** | **4.246** (**0.93×**) | 839 / 729 ms | ✅ |

> **⚠️ THE SAME ROWS UNDER THE NOW-PRIMARY MACRO WEIGHTING (8 cells, 1/8 each, Variant B, measured;
> `artifacts/macro_average_headline_2026-07-30.json`).** Note the **FLOP-neg column becomes ❌ for every row.**
>
> | knob | vs 32B-**reasoning** | vs 32B-**direct** | FLOP-eq (×32B) | lat par / seq | FLOP-neg? |
> |---|---:|---:|---:|---:|:--:|
> | compute-lean | **+0.0626** [+0.0514, +0.0734] | +0.0033 [−0.0054, +0.0121] n.s. | 5.465 (**1.196×**) | 650 / 1,292 ms | ❌ |
> | accuracy-max (F8 veto + F10 open) | **+0.0720** [+0.0614, +0.0824] | **+0.0128** [+0.0056, +0.0200] | 6.444 (**1.410×**) | 691 / 1,334 ms | ❌ |
> | accuracy-max v1 (F3 fusion) | **+0.0686** [+0.0582, +0.0790] | **+0.0094** [+0.0013, +0.0176] | 6.558 (**1.435×**) | 665 / 1,350 ms | ❌ |
>
> And on the **multiple-choice half alone**, compute-lean vs 32B-direct is **−0.0070 [−0.0126, −0.0017] — a
> significant LOSS** (vs oracle-mode −0.0080 [−0.0137, −0.0024]).

~~**Both modes are now FLOP-negative** (< always-32B's 4.57)~~ **[STRUCK 2026-07-30 — sample-weighted only.]**
Sample-weighted: compute-lean **0.492×** (unchanged — it always was), accuracy-max **newly 0.932×** (was 1.250×).
**What survives every weighting is the F8 FLOP CUT, not the sign:** **−1.449 FLOP-eq (−25 %)** sample-weighted,
and macro **6.558 → 6.444** — for a **−0.0026** vs-think accuracy give-back sample-weighted (F8 captures **70 %**
of F3's PMC beat; F10 gives some back on the open cells), still **CI-certified above 32B on PMC**. But it is
**not** "a strict Pareto move into the compute-negative half-plane" at equal weight, because at equal weight there
is no compute-negative half-plane to move into. Honest trade: accuracy-max parallel latency rises **666 → 729 ms**
sample-weighted because F8 is a *sequential* cascade on PMC (7B → maybe 32B, no parallel-fusion overlap) — still
**−93 %** vs 32B-think as charged, **−89 %** honestly re-costed and macro-weighted.

### ★★.2 F8 on the accuracy-max PMC cell (vs the prior F3 fusion)

| policy | acc | d vs 32B-nt | veto% | esc→32B% | FLOP-eq | lat seq / par | note |
|---|---:|---:|---:|---:|---:|---:|---|
| F3 confidence-advantage fusion (v1) | 0.5653 | +0.0135 | — | 100 % | 5.570 | 1012 / 665 ms | both legs on 100 % of PMC |
| **F8 certified weak-veto (v2)** | 0.5613 | **+0.0095** | 40.0 % | 60.0 % | **3.741** | 746 / 746 ms | 7B on all + 32B on non-veto |

F8 **captures 70.4 %** of F3's PMC beat over 32B at **−32.8 % PMC FLOPs**; still guardrail-safe (CI-certified
above 32B, `beat32b_more.json`: +0.0095, CI [0.0071, 0.0118]).

### ★★.3 F10 on the shared open-text arm (vs the prior parity-τ gate) — d vs always-32B-no-think

| open cell | n | prior gate (Pandora-τ) | **F10 L2D** | **gain** | repaired? | F10 acc | esc% | FLOP-eq |
|---|---:|---:|---:|---:|:--:|---:|---:|---:|
| SLAKE-open | 645 | −0.0093 | **+0.0016** | +0.0109 | ✅ | 0.8202 | 20.6 % | 7.842 |
| VQA-RAD-open | 200 | −0.0050 | **+0.0050** | +0.0100 | ✅ | 0.6050 | 37.0 % | 9.511 |
| PathVQA-open | 1 500 | +0.0760 | **+0.0860** | +0.0100 | (already won) | 0.4620 | 26.5 % | 12.178 |

**All three cells lift**; the **two residual open losses (SLAKE-open, VQA-RAD-open) are repaired** from below-32B
to above-32B, and PathVQA-open improves further. Pooled open-only: acc **0.5625 → 0.5727**, d-vs-32B-nt
**+0.0456 → +0.0559**, d-vs-think **+0.1927 → +0.2029**, at slightly **lower** open FLOPs (10.871 → 10.758). The
prior τ-gate targeted iso-32B *by design* (parity at min escalation), so it sat at/below 32B on those cells; F10
optimizes the right (team-accuracy) objective. *Note:* the multi-feature learned score does **not** have higher
recoverability AUROC than the single gate — the gain is from the objective, not a better signal.

### ★★.4 V2 confirmations (the two asks)

1. **accuracy-max with F8 is FLOP-NEGATIVE *sample-weighted*** — pooled full-suite FLOP-eq **4.246 < 4.57**
   (**0.932×** always-32B), down from **5.695 (1.250×)** with F3, at the same/better beat structure (retains 70 %
   of the PMC beat, still CI-certified above 32B; d-vs-think **+0.0212**). ✅ **⚠️ NOT at equal weight per cell:
   6.444 vs 4.57 = 1.410× (F3: 6.558 = 1.435×). The −25 % cut survives; the FLOP-negative sign does not.**
2. **F10 raises the open-text cells in both modes** (the open arm is shared): all three open cells gain, the two
   open losses are repaired. ✅ *(PROVISIONAL — clean-verifier retrain in progress.)*
3. ~~**Both modes are FLOP-negative** — compute-lean **0.49×**, accuracy-max **0.93×**.~~ ❌ **RETRACTED
   2026-07-30 as a suite-level claim** — true sample-weighted, false at equal weight per cell (1.196× / 1.410×).

### ★★.5 Honest caveats specific to V2

- **F8 captures MOST, not all, of F3's PMC beat** (+0.0135 → +0.0095 on the PMC cell). The accuracy-max vs-think
  headline drops **+0.0238 → +0.0212** sample-weighted; the trade buys a cheaper arm and a never-worse one-sided
  guarantee. **⚠️ 2026-07-30: PMC-VQA is 79.2 % of the sample-weighted pool but only 12.5 % of the macro weight,
  so at equal weight the F8/F3 choice can no longer carry the headline — macro vs-reasoning is +0.0720 (F8) vs
  +0.0686 (F3), i.e. F8 is now the *more* accurate of the two as well as the cheaper.**
- **F10 open-arm cost is billed at Pandora's adaptive `meanN` (<8) draws while the *kept*-leg accuracy is scored on
  the best-of-8 verifier pick** — mildly optimistic vs a strict best-of-`meanN`. The escalation set and delivered
  accuracy come from the held-out F10 rejector; only the cheap-leg FLOP count is Pandora's.
- **F10 routes on 7B-side features only** (open-text dumps have no 32B prediction *text*, only judge_ok) → no
  cross-model-agreement feature. Still fully OFFLINE.
- Small-n on VQA-RAD-open (n=200): the +0.0050 F10 point-beat's CI still spans 0; SLAKE-open (n=645) likewise. The
  robust CI-certified open beat remains PathVQA-open. (`beat32b_more.json:F10_l2d_open`.)

---

## 1. Baseline — always-Lingshu-32B-THINK (and why it is the honest one)

The deliverable is about the cost of the model you would otherwise deploy when you want a 32B reasoning model's
accuracy: the **THINK** model. Its **measured batch-1** cost (`opentext_32b_think.json`, HF batch-1, single GPU,
cap320 real VQA-RAD images, per-GPU NVML energy, n=15 after 3 warmups):

| 32B mode | latency (mean / median) | energy | gen tokens | source |
|---|---|---|---|---|
| **THINK** (native `<think>` prompt) | **10 521.6 ms** / 12 896.2 ms | 2 001.9 J | 98.3 | `opentext_32b_think.json` |
| no-think (reference strong leg) | 665.0 ms / 696.4 ms | 126.9 J | 5.6 | idem, `latency_energy_nothink_reference` |
| **ratio think : no-think** | **15.8×** | **15.8×** | — | idem |

Crucially, on open-text perception THINK is also **less accurate** than no-think (measured, n=200 paired per set,
modal scorer): SLAKE-open **0.700 vs 0.895 (−0.195)**, VQA-RAD-open **0.425 vs 0.545 (−0.120)**, PathVQA-open
**0.035 vs 0.170 (−0.135)**; pooled n=600 **0.387 vs 0.537 (−0.150)**. So **always-32B-THINK is dominated *by
always-32B-no-think*** (a model-vs-model statement, which still holds and is **not** the retired suite-level
"the method Pareto-dominates" claim — §10.1 C26): it is ~16× slower/costlier *and* less accurate than 32B-no-think
on perception. Think appears to help **only on reasoning**
(faithful MedEvalKit, `reframe_vs_bigthink.json`): MMMU-150 Lingshu +0.027 / MVT +0.100 / IV3 +0.120; MedXpert-2000
Lingshu −0.003 / MVT +0.045 / IV3 +0.031. (Peer strong leg IV3-38B batch-1 for reference, `iv3_38b_latency.json`:
no-think 1 409 ms / 598 J, think 6 220 ms / 3 276 J → 4.4× slower, 5.5× more energy.)

> **⚠️ RE-ATTRIBUTED 2026-07-30 — those reasoning-side gains are ANSWER-FORMAT effects, not
> reasoning-instruction effects.** Every one of the MMMU / MedXpert numbers in the paragraph above is a
> **published-direct (bare letter) → reason (trigger + `\boxed{}`)** contrast. Decomposed against a
> format-matched (`\boxed{}`-only, no trigger) direct arm — **published / format / trigger** — MVT MMMU
> **+0.103 / +0.062 / +0.041 n.s.**, MVT MX-R **+0.046 / +0.046 SIG [+0.019, +0.072] / +0.001 n.s.**, IV3 MMMU
> **+0.124 / +0.090 SIG / +0.035 n.s.**, Lingshu MMMU **+0.028 / −0.014 / +0.041 n.s.**, Lingshu MX-R
> **−0.004 / −0.008 / +0.004 n.s.** **0/9 trigger effects are CI-significant; 3/9 format effects are.**
> Mechanism: **asking for `\boxed{}` is itself a reasoning trigger** — MVT emits 431–580 tokens on 99–100 % of
> items, IV3 193–289 on 94–95 %, with **no trigger**; Lingshu never (3–4 tokens). `parse_ok ≥ 0.9986` (min over the 9 sub-cells; 1.000 in 6 of them) in every
> new arm. **The design consequence below is UNCHANGED** (the strong leg is 32B-no-think; the think tier is
> reserved for the reasoning residual, and the full ladder is what a think tier delivers) — **only the
> attribution changes.** Source: `artifacts/medeval_matched_direct_2026-07-29.json`; retrospective §5.1, C27.

**Consequence for the method:** the strong leg is **32B *no-think*** everywhere (it dominates 32B-think on cost and
on perception accuracy, and ties it on Lingshu reasoning), and the slow think tier is *reserved* for the reasoning
residual only. On Lingshu specifically the reasoning think-gain is ~0 (its "think" run ≈ no-think, latency ratio 1.2×),
so the Lingshu deployment fires think **~0 %** and the whole method reduces to `7B-nt → 32B-nt` (+ open-text arm).

> **⚠️ Re-grounded 2026-07-29 (the conclusion holds; the reason was wrong).** Lingshu's "think ≈ no-think,
> latency ratio 1.2×" was an artifact of a prompt with **no reasoning trigger** — that arm emitted **3.0
> generated tokens** (`runners/run_native_think.sh:7`), so all 7 published Lingshu think-vs-direct cells are
> **withdrawn** (retrospective §10.1 C22) and **1.2× is not a reasoning:direct cost ratio**. Re-measured
> against a genuinely reasoning Lingshu arm (150–259 tokens): the reasoning gain really is ~0 (MMMU +0.0000,
> MedXpert-R +0.0048, MedXpert-U +0.0271, **none significant**) *and* reasoning actively **hurts** Lingshu
> perception (pooled **−0.0866 [−0.0972, −0.0757]**). So "strong leg = 32B-no-think, think tier fires ~0%"
> is **more** justified than this paragraph claimed, not less — the cost of the tier is real, and its benefit
> is still zero. Also put a CI on the MMMU line above: **Lingshu +0.027 is NOT significant**
> ([−0.047, +0.100], n = 150); MVT +0.100 [+0.027, +0.173] and IV3 +0.120 [+0.047, +0.193] are.
> Source: `artifacts/finding1_corrected_2026-07-29.json`.

---

## 2. Cost model (batch-1, measured in this repo — one set of constants across the codebase)

| symbol | what | latency | FLOP-eq | source |
|---|---|---|---|---|
| `GEN7`   | one 7B no-think greedy gen (MCQ + SLAKE-open cheap leg) | 347 ms | 1.0 | measured; `integrated_method.py` |
| `VER7`   | one verifier forward (scores all 8 candidates in one batch) | 175 ms | 1.0 | measured |
| `BO8`    | open-text best-of-8 cheap leg: **8 gens in PARALLEL + 1 verify** | 522 ms | 16.0 | latency = 1 gen + 1 verify; FLOPs = 16 cheap forwards |
| `GEN32N` | 32B **no-think** (the escalation target *and* honest "always-32B") | 665 ms | 4.57 | measured |
| `GEN32T` | 32B **think** (the naive baseline) | 10 521.6 ms | 4.57† | measured latency; FLOP-eq is a **lower bound** (long decode) |
| `FUSE`   | a decision-fusion cell runs **both** legs, co-resident/parallel | max(347,665)=**665 ms** | **5.57** | 1.0 + 4.57; latency ≈ 32B-nt |

† `GEN32T` FLOP-eq is set equal to `GEN32N` (4.57) as a conservative lower bound; a think forward emits a long
`<think>` trace so its true decode FLOPs exceed this — the reported FLOPs *understate* the baseline's cost.

**Held-out protocol (all thresholds).** Every escalation threshold τ (and every G5 suppress decision) is chosen by
**5-fold cross-fit**: on 4/5 pick the **minimum-escalation** τ such that cascade accuracy ≥ the strong-leg's accuracy
(iso-accuracy target), evaluate on the held-out 1/5, average the folds. No peeking; reported as deployable.

**Scoring (faithful, as elsewhere in the repo).** MCQ/closed → MedEvalKit exact-match `correct` (MMMU → `parsed_output`
judge == `Correct`). Open-text → LLM/Claude-judge `judge_ok` (open-text exact-match is known-broken, e.g. `"CT." ≠ "CT"`).

---

## 3. The format router (top level)

```
                          ┌──────────────── detect format from the PROMPT (never the gold) ─────────────────┐
question + image ─────────┤                                                                                  │
                          │  MCQ / closed  ─────────────►  §4  MCQ ARM                                        │
                          │  open-ended    ─────────────►  §5  OPEN-TEXT ARM                                  │
                          └──────────────────────────────────────────────────────────────────────────────────┘
```

**Why a router and not one unified gate (Correction #2, `integrated_method.json:verdict.router_vs_unified`).** The
MCQ margin gate has **no open-text analog** (open answers have no single-letter logprob margin), and the trained
verifier is **open-text-specific**. A weak unified proxy (7B sequence-logprob) works passably as the open-text gate
(the SLAKE-open fallback) but is **beaten by margin on MCQ and by verifier-confidence on open** → a single unified
policy underperforms. **Keep the 2-arm router.**

---

## 4. MCQ arm — `7B-nt + margin gate → 32B-no-think`

**Mechanism.** Run Lingshu-7B no-think (one greedy gen). Compute the **margin** = P(top-1 option) − P(top-2 option)
from the option logprobs. **Escalate** (re-answer with 32B no-think) iff `margin < τ_mcq`. τ_mcq is the held-out
min-escalation threshold that reaches 32B-no-think parity. Two per-benchmark policy overrides:

- **MMMU → keep 7B** (no escalation). Lingshu-7B scores **0.80** on MMMU-Medical-val (an anomaly), vs 32B-no-think
  0.633 and 32B-think 0.660, so the router keeps 7B and **beats always-32B-think by +0.140** at 1.0 FLOP / 347 ms.
- **PMC-VQA → optional fusion** (the accuracy-max knob, §6). Default is the margin cascade (matches 32B-nt); the
  fusion variant *beats* 32B-nt.

**Per-benchmark results — default margin cascade** (`integrated_method_vs_think.json:per_benchmark`; held-out 5-fold,
batch-1 measured costs). `d_think` = method − always-32B-think; `d_nt` = method − always-32B-no-think.

| benchmark | n | 7B | 32B-nt | 32B-think | **method** | **d_think** | d_nt | esc% | lat (ms) | FLOPs | lat saved vs think |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PMC-VQA *(`test_2.csv`)* | 33 430 | 0.5427 | 0.5518 | 0.5494 | **0.5508** | **+0.0014** | −0.0010 | 8.5 % | 403.2 | 1.386 | 96.2 % |
| SLAKE-closed | 836 | 0.8254 | 0.8589 | 0.8636 | 0.8516 | −0.0120 | −0.0072 | 20.5 % | 483.0 | 1.935 | 95.4 % |
| VQA-RAD-closed | 251 | 0.7809 | 0.8526 | 0.8406 | 0.8328 | −0.0079 | −0.0198 | 57.0 % | 725.8 | 3.603 | 93.1 % |
| PathVQA-closed | 3 362 | 0.8409 | 0.8891 | 0.8891‡ | 0.8882 | −0.0009 | −0.0009 | 45.7 % | 651.0 | 3.089 | 93.8 % |
| MedXpert-MM | 2 000 | 0.2615 | 0.3065 | 0.3040 | 0.3005 | −0.0035 | −0.0060 | 89.6 % | 942.8 | 5.095 | 91.0 % |
| MMMU-Medical (keep-7B) | 150 | 0.8000 | 0.6333 | 0.6600 | **0.8000** | **+0.1400** | +0.1667 | 0 % | 347.0 | 1.000 | — |

‡ PathVQA-closed has no 32B-think dump → think acc set = no-think (think ≈ no-think on perception; honest gap).

**MCQ-arm pooled** (`pooled.mcq_only`, n=40 029): method **0.5756** vs 32B-think **0.5745** (**d_think +0.0011**) vs
32B-nt **0.5765** (d_nt −0.0009); esc **16.16 %**; latency **454.5 ms** vs 10 521.6 ms (**−95.7 %**); FLOPs **1.738**
vs 4.57 (**~2.6× cheaper**). Macro-avg method 0.704 vs 32B-think 0.6845 (d +0.0195). The MCQ arm **saves FLOPs**:
`1.0 + esc·4.57 ≪ 4.57`.

### 4.1 Gate bake-off — **Correction #1: margin > agreement > CASP on Lingshu**

The premise (backlog §F) was that CASP-stability (cap320-vs-full agreement) or cross-model agreement would beat
FALC's margin gate. **It does not.** Pooled perception-closed MCQ (`integrated_method.json:gate_comparison_mcq`,
n=37 879; cheap-7B-nt 0.5769, strong-32B-nt 0.5905). KEEP-signal = "trust 7B"; escalate the lowest-scoring:

| gate (KEEP signal) | detection AUROC | **min esc to reach 32B-nt parity** | deployable? |
|---|---:|---:|:--|
| **margin (7B, deployed)** | 0.7254 | **15.62 %** | **yes — cheap, continuous, best deployable** |
| conf / MSP (7B) | 0.7318 | 20.26 % | cheap but needs more escalation |
| CASP-stability (7B cap320-vs-full) | 0.7241 | 15.50 % | **INERT** (see below) |
| agreement (7B-nt vs 32B-nt) | 0.6565 | 19.96 % | needs the 32B run → not a cheap gate |

- **CASP is inert.** Lingshu-7B is **98.95 % cap320-vs-full stable** (`casp_frac_stable`), so the CASP signal carries
  almost no information and **collapses to the margin tiebreak** — its AUROC/min-esc ≈ margin *only because it is
  margin* for the 98.9 % stable majority (+ escalating the 1.1 % unstable). It is not a genuinely independent gate.
- **Agreement is informative but not a cheap gate.** P(7B correct | agree) = **0.6868** vs P(7B correct | disagree) =
  **0.3262** (agree-rate 0.6953) — a real binary trust signal — but as an escalation *ranker* it is the **worst**
  (AUROC 0.657), and being a committee signal it **requires running the 32B**, defeating the purpose of a cheap gate.
- **conf/MSP** has a hair-higher raw AUROC (0.7318) but reaches parity only at **20.3 %** escalation vs margin's
  **15.6 %**, so **margin dominates on the deployable metric**.

**Verdict:** FALC's margin choice was correct; **no cheap composite beats margin on MCQ** (matches the repo's standing
finding: no trained gate beats margin on MCQ). The "deployability" ranking is **margin > agreement > CASP** (margin is
the best cheap continuous ranker; agreement is a real-but-committee signal that needs the 32B; CASP is inert).

### 4.2 Escalation-speed lever — G8 parallel prefill prefetch (the load-bearing knob)

vs the *cheaper* baseline always-32B-**no-think** (665 ms) the MCQ arm's heavy-escalation cells are actually
**slower**: VQA-RAD-closed 725.9 ms (esc 57 %), MedXpert 942.8 ms (esc 90 %), SLAKE-open 698.6 ms (esc 53 %). G8 fixes
this at **zero accuracy cost** (`escalation_levers.json:G8_prefill_prefetch`).

- **Idea:** the 32B *image-prefill* does not depend on the 7B output, so run it **concurrently** with the 7B pass. An
  escalated query then pays `max(cheap_leg, prefill32) + decode32` instead of `cheap_leg + (prefill32 + decode32)`.
- **Prefill fraction** φ = **0.586** (measured, `latency_32b.jsonl`: no-think@cap320 prefill 195 ms of 333 ms) →
  prefill32 = 0.586·665 = **389.7 ms**, decode32 = **275.3 ms**. Since prefill32 (390 ms) > MCQ cheap leg (347 ms),
  **the entire 7B pass hides under the 32B prefill on every MCQ escalation** (robust for any φ ≥ 347/665 = 0.522).
- **Effect (pooled):** batch-1 latency **461.1 → 405.2 ms (−12.1 %)** at **identical accuracy 0.5749**; the slower
  cells `{VQA-RAD-closed, MedXpert, SLAKE-open}` → **none** (all flip under always-32B-nt). φ-insensitive (405 ms at
  φ ∈ [0.586, 0.80]).
- **FLOPs caveat (honest):** *unconditional* prefetch pays the 32B prefill on every query → pooled FLOPs **2.337 →
  4.575** (≈ always-32B). G8 **trades FLOPs/energy for latency** and is only "free" when the 2nd GPU is idle. The
  **slice-gated** deployable variant (prefetch only where base esc ≥ 0.40: VQA-RAD-closed / PathVQA-closed / MedXpert
  / SLAKE-open) recovers nearly all the latency win at a fraction of the FLOP cost: pooled **429.8 ms (−6.8 %)**,
  FLOPs **2.492** (vs base 2.337), acc unchanged.

**G5 (recoverability suppressor) and G6 (2-of-2 gate) are knobs, not free lunches** (`escalation_levers.json`):
- **G5:** no slice is *truly* futile. The named-futile MedXpert has the worst recovery (P(32B fixes 7B error)=0.225,
  P(32B breaks 7B correct)=0.479) and is grossly inefficient (+0.039 acc for +596 ms) but escalation is still net
  positive; suppressing it is a **trade** (−0.039 on MedXpert = **−0.0018 pooled**, flips MedXpert 943→347 ms).
- **G6:** no gain — there is no orthogonal 2nd cheap signal on MCQ (CASP is 98.9 % inert), so `AND(margin, casp)` ≤
  margin at matched escalation. Confirms "no cheap composite beats margin on MCQ."
- **Combined best (G8 slice-gated + G5 ε\*=0.06):** pooled acc **0.5731** (d −0.0018), latency **416.4 ms** (−9.7 %
  vs 461 ms base, **−96.0 % vs 32B-think**), FLOPs 2.285; every previously-slower cell ≤ always-32B-no-think.

---

## 5. Open-text arm — `7B best-of-8 + trained verifier (verifier-conf gate) → 32B-no-think`

**Mechanism.** Run **8 samples** of Lingshu-7B (in parallel), score each candidate with the **trained outcome
verifier** (`ckpts/train/lora_verifier_pooled4`, one forward scores all 8), and **pick** the argmax-P(correct)
candidate. **Gate** on the verifier's max score; **escalate** the low-confidence residual to 32B-no-think. SLAKE-open
originally had no verifier dump → **FALC fallback** = 7B greedy + free 7B sequence-logprob gate; that dump is now
**FILLED** (§6.3), so the final SLAKE-open cell uses bo8+verifier.

**Per-benchmark results** (`integrated_method_vs_think.json`; cheap leg = best-of-8, cost `BO8`; 32B-think acc is
**estimated** = judged 32B-no-think + the measured modal think-delta, flagged):

| benchmark | n | greedy 7B | bo8+verifier | 32B-nt | 32B-think (est) | **method** | **d_think** | d_nt | esc% | lat (ms) | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SLAKE-open | 645 | 0.7364 | 0.7798 | 0.8186 | 0.6236 = 0.8186−0.195 | **0.8155** | **+0.1919** | −0.0031 | 12.6 % | 605.5 | 16.574 |
| VQA-RAD-open | 200 | 0.4650 | 0.5750 | 0.6000 | 0.4800 = 0.6000−0.120 | 0.5850 | +0.1050 | −0.0150 | 5.5 % | 558.6 | 16.251 |
| PathVQA-open | 1 500 | 0.3240 | 0.4533 | 0.3760 | 0.2460 = 0.3760−0.130 | 0.4533 | +0.2073 | +0.0773 | 0.1 % | 522.4 | 16.003 |

**Open-arm pooled** (`pooled.open_only`, n=2 345): method **0.5642** vs 32B-think **0.3698** (**d_think +0.1943**) vs
32B-nt **0.5168** (d_nt +0.0473); esc **3.97 %**; latency **548.3 ms** vs 10 521.6 ms (**−94.8 %**); FLOPs **16.181**.
Macro-avg method 0.6179 vs 32B-think 0.4499 (d +0.1681).

**This is the accuracy engine.** The 7B best-of-8 ensemble *beats the 32B* on the OOD open-text sets, which in turn
beats 32B-THINK by +0.12…+0.21 (think over-thinks perception). The verifier-confidence gate escalates almost nothing
(≤ 13 %). **FLOPs cost is honest:** best-of-8 = 16 cheap forwards, so the open arm **costs more FLOPs** than a single
32B forward (break-even vs one 32B is N ≤ 2). It buys the latency win (parallel bo-N ≈ 522 ms ≪ 665 ms no-think ≪
10 522 ms think) **and** the accuracy win — but it is **not** a FLOP-saving lever on the open arm.

---

## 6. Two operating points — the Pareto knob (cascade ↔ fusion)

The method exposes **one accuracy↔compute knob on the PMC-VQA slice**: run the cheap margin cascade (compute-lean,
matches 32B) *or* run the slice-gated fusion (accuracy-max, beats 32B). Because PMC is ~79 % of the pooled samples,
this knob moves the whole pooled headline.

### 6.1 Default (compute-lean) — the integrated margin cascade
`integrated_method_vs_think.json:pooled.full_suite`, n=42 374:

| pool | method | 32B-think | **d_think** | 32B-nt | d_nt | esc% | latency | lat saved | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **full suite** (9 benchmarks) | **0.5750** | 0.5631 | **+0.0118** | 0.5732 | +0.0018 | 15.5 % | **459.6 ms** | **−95.6 %** | **2.538** |
| macro-avg | 0.6753 | 0.6063 | +0.0690 | — | — | — | — | — | — |

### 6.2 Accuracy-max — swap PMC margin-cascade → **slice-gated confidence-advantage FUSION**
`beat32b_fusion.json`. On the largest slice, 7B and 32B are **comparably skilled with de-correlated errors**, so
fusing their *calibrated* per-sample confidences beats either alone. The router picks the fusion policy on PMC
**only because a held-out paired-bootstrap guardrail certifies it** (95 % lower-CI > 0); everywhere else the
guardrail keeps always-32B (fusion *hurts* the small radiology/pathology sets where the 32B is clearly better).

- **PMC fusion (F3 confidence-advantage, held-out; `test_2.csv`, n = 33 430):** acc **0.5653** vs 32B-nt 0.5518 → **d_nt +0.0135, 95 % CI
  [0.0100, 0.0169]**, n=33 430; d_think +0.0159. Equivalent to a 2-detector **Chair-Varshney** fuser under equal
  option-count (reconciles: F2-cv3 +0.0134 CI [0.0109, 0.0159]). **Classic per-*slice*-reliability C-V collapses to
  exactly always-32B (d=0.0)** — the beat *requires per-sample confidence*, not slice reliability.
- **F1 router choices** (guardrailed): PMC → `F3_confadv`; SLAKE-cl / VQA-RAD-cl / PathVQA-cl / MedXpert →
  `always_32b_nt`; MMMU → `always_7b`. **Certified non-32B slices = {PMC-VQA, MMMU}** only. No finer non-MMMU
  pure-routing 7B-owned slice exists (MedXpert subject cells are within noise); the new broad win is **fusion**, not
  keep-7B.
- **F5 double-reading (why think is a poor arbiter):** on PMC, 7B & 32B agree 67.0 % (acc-on-agree 0.641, free);
  on the 33.0 % disagreement set the **free** calibrated conf-advantage arbiter scores **0.4116**, beating both the
  32B-nt arbiter (=always-32B, 0.3707) **and** the expensive 32B-think arbiter (0.3871). Think ≤ no-think as an
  arbiter on perception disagreements. (Oracle-UB on the disagreement set 0.689 → recoverability-AUROC 0.634: the
  recoverability wall bounds the beat.)

**Accuracy-max pooled** (`beat32b_fusion.json:pooled.full_suite`, n=42 374):

| pool | method | 32B-nt | d_nt | 32B-think | **d_think** | FLOPs (×always-32B) | latency | lat saved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **full suite** | **0.5869** | 0.5732 | **+0.0138** | 0.5631 | **+0.0238** | 5.751 (**1.26×**) | 653.3 ms | −93.8 % |
| macro-avg | 0.6802 | 0.6541 | +0.0261 | 0.6063 | +0.0739 | — | — | — |

**Cost of the knob (honest).** The PMC fusion cell must run **both legs** (FLOP 5.57 vs 4.57, **+22 %**), so on that
slice it **trades the margin-cascade's PMC FLOP saving** (which ran the 32B on only ~8 % of PMC) **for running the 32B
on 100 % of PMC**. Batch-1 latency stays ≈ parity (parallel, 665 ms) and ~16× faster than 32B-think. It is an
**accuracy (beat-32B) lever, not a compute-saving one**. You cannot cascade-away the 32B on a fusion slice — the fuser
needs the 32B decision.

**The knob in one line:** *compute-lean* = 0.575 @ FLOPs 2.54 (d_think +0.012); *accuracy-max* = 0.587 @ FLOPs 5.75
(d_think +0.024). PMC fusion ~2× the vs-think delta and ~8× the vs-no-think delta, at ~2.3× the pooled FLOPs.

### 6.3 The SLAKE-open best-of-8 fill (`slake_open_bestofN.json`)
The pooled4 verifier was scored **pointwise over the existing K=8 SC candidates** (n=645): **bo8+verifier 0.7798** vs
greedy_t0 0.7302 (**+0.0496**) vs self-consistency-modal 0.7364 (+0.0434); oracle@8 0.8791; 32B-no-think 0.8186
(bo8 − 32B-nt = −0.0388). With the verifier-confidence gate the SLAKE-open cell reaches 32B-nt parity at **~13 %
escalation** vs **~53 %** for the old greedy+seqlogprob fallback — a **~4× escalation cut**. **Caveat:** SLAKE-open is
**in-domain** for pooled4 (verifier trained on slake+pathvqa+kvasir+vqa_rad), the same situation as the VQA-RAD-open /
PathVQA-open cells that already used it.

### 6.4 INT4 strong-leg cost-mode (`int4=True`) — G3, PROJECTED

An optional cost-mode re-costs the 32B strong leg with an AWQ/GPTQ-INT4 forward (`method_final.json:
int4_projection`, recomputed live from `latency_32b.jsonl`; latency **projected**, accuracy **projected**):

- **Latency (projected, composition-grounded).** AWQ-INT4 accelerates only memory-bound **decode**; the
  no-think strong leg emits ~2 tokens so it is **prefill-bound** — decode is ~137 ms of the 665 ms leg
  (measured decode 68.6 ms/tok). At a literature 2.5× decode speedup the strong leg drops **665 → 583 ms**
  (ratio 0.876), **not** the ~2.5× a decode-heavy op would see. Pooled effect: compute-lean parallel latency
  469 → **455 ms**; accuracy-max parallel 666 → **588 ms**.
- **FLOPs — UNCHANGED.** The repo's FLOP unit is **MAC-count** (weight-precision independent): a 32B-INT4
  forward does the same MACs as FP16 = **4.57**. So method FLOPs are literally unchanged (2.244 / 5.695). A
  throughput-effective accounting would credit ~0.876× on this prefill-bound leg (reported, not headline).
- **Accuracy (projected bracket).** INT4 medical loss ≤ 1 % (literature W4A16); it touches only the
  quantized-32B share of the delivered answers, so method-acc erodes by `strong_share × δ`, δ∈[0.5 %, 1.0 %].
  The vs-**think** win is robustly held: compute-lean d_think **[+0.010, +0.011]**; accuracy-max d_think
  **[+0.014, +0.019]**. Both stay positive; the razor-thin compute-lean vs-**no-think** margin can erode to
  ≈ 0 (bracket [0.000, +0.001]).
- **What it buys.** VRAM/energy (real INT4 win) and lets the escalation model fit **tp=1 on one GPU**. It is
  a **deployability/energy lever**, not a FLOP-saver and not a material latency lever (the method already
  beats the think baseline ~16–23×).

---

## 7. Why the structure (ablations that justify each piece)

1. **Strong leg = no-think, not think** (`reframe_vs_bigthink.json`, `opentext_32b_think.json`). 32B-think ≤
   32B-no-think on every MedEvalKit perception benchmark and is 15.8× slower/costlier. Removing think as the strong
   target also removes the dependency on the (previously blocked) 32B-think latency measurement.
2. **The 32B-*no-think* MIDDLE tier is the load-bearing structural win** (`reframe_vs_bigthink.json:middle_tier`,
   `METHOD_ACC.md`, ALL-6, measured): inserting it turns an escalate-everything-to-think cascade into the ACC —

   | variant (ALL-6) | acc | think-esc | FLOPs% | latency | energy |
   |---|---:|---:|---:|---:|---:|
   | M2 escalate-to-think, no nt-middle (7B-think→32B-think) | 0.5725 | 86 % | 105 % | 29.8 s | 7 049 J |
   | M3 7B-think middle (7B-nt→7B-think→32B-think) | 0.5697 | 65 % | 89 % | 23.2 s | 5 499 J |
   | **M1 ACC — 32B-nt middle restored** | 0.5694 | **19 %** | **55 %** | **5.9 s** | **1 505 J** |
   | M1b ACC + agreement gate | 0.5710 | 14 % | 54 % | 4.86 s | 1 220 J |

   Δ(M1 vs M2): think-esc 86 %→19 %, FLOPs 105 %→55 %, latency **29.8→5.9 s (−80 %)**, energy −79 %, at matched acc.
3. **MMMU keep-7B** — the router simply exploits the Lingshu-7B MMMU anomaly (0.80 > 32B-think 0.66) for +0.140 at
   1 FLOP.
4. **Router, not unified gate** — Correction #2 (§3).
5. **Margin, not CASP/agreement** — Correction #1 (§4.1).
6. **Fusion only where certified** — the guardrail keeps 32B on the radiology/pathology sets where fusion hurts (§6.2).

---

## 8. Reproduction (CPU only; launch from repo root)

**One command reproduces the whole method** (both knob settings, INT4 cost-mode, the final tables, and the
reconciliation — all live from the dumps):

```bash
cd ~/medvlthinker-imgdiff-compute
python3 src/cascade_methods/method_final.py        # → artifacts/method_final.json   ★ THE unified pipeline
```

Per-lever reference scripts (each a subset of the above, kept for provenance):

```bash
python3 src/cascade_methods/integrated_method.py   # → integrated_method_vs_think.json  (compute-lean base router, fixed bo8)
python3 src/cascade_methods/beat32b_fusion.py      # → beat32b_fusion.json              (accuracy-max PMC fusion, separate script)
python3 src/cascade_methods/integrated_pandora.py  # → integrated_pandora_opentext.json (Pandora adaptive-N open arm)
python3 src/cascade_methods/quantized_strong_leg.py# → quantized_strong_leg.json         (G3 INT4 strong-leg re-costing)
python3 src/cascade_methods/escalation_levers.py   # → escalation_levers.json           (G8 latency lever, G5/G6)
```

Inputs (saved per-sample dumps): `MedEvalKit/eval_results_{lingshu7b_full,lingshu7b_cap320,lingshu32b_full,
lingshu32b_think,lingshu32b_reason}/…/results.json` (+ MMMU `parsed_output.json`); open-text verifier dumps
`ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_lingshu7b.json`; judge jsonl under `ckpts/openvqa/{cheap_lingshu7b,
strong_lingshu}/`. Measured cost constants: `opentext_32b_think.json`, `iv3_38b_latency.json`, `latency_32b.jsonl`.

---

## 9. Honest caveats / data gaps

- **FLOPs cost of fusion / best-of-N.** The PMC fusion cell runs both legs (+22 % FLOPs vs always-32B-nt on that
  slice); the open-text best-of-8 arm is **16 cheap forwards** (break-even vs one 32B forward is N ≤ 2). These are
  **latency + accuracy** levers, not FLOP-savers. The *MCQ margin cascade* is the FLOP-saving part (1.74 pooled MCQ
  FLOPs vs 4.57).
- **32B-THINK open-text accuracy is ESTIMATED** for the open cells (judged 32B-no-think + the measured *modal-space*
  think-delta −0.195/−0.120/−0.130). A judged 32B-think open-text dump would remove the estimate.
- **PathVQA-closed has no 32B-think dump** → its think acc is set = no-think (think ≈ no-think on perception).
- **always-32B-THINK batch-1 latency (10 521 ms)** was measured on open-text cap320; MCQ reasoning traces
  (MMMU/MedXpert ~275–320 gen tokens) would be ≥ this, so it is representative-to-conservative for the think baseline.
- **G8 φ=0.586** is measured on MVT-32B no-think@cap320 and transferred as a *fraction* to the Lingshu 665 ms constant
  (conservative). A direct Lingshu-32B prefill/decode split would remove the transfer. G8's unconditional-prefetch
  FLOPs assume an idle 2nd GPU; the prefetched-prefill *energy* is real even when latency is hidden.
- **SLAKE-open (and VQA-RAD-open / PathVQA-open) are IN-DOMAIN** for the pooled4 verifier — the open-text pooled-4
  numbers are in-domain, not held-out-domain. Flagged consistently.
- **OmniMed-32B is BLOCKED** (deterministic tp=2 NCCL hang; tp=1 has no gpu-mem window on 1×80 GB) → excluded (7B-only).
  Conclusion unchanged: OmniMed is keep-cheap (cheap Lingshu 0.827 ≈ paper strong 0.834), so only the pooled ALL-7
  figure is missing, no verdict.
- **Two eval contexts, kept separate (do not cross-multiply):** the faithful MedEvalKit eval (accuracy + reasoning
  think-deltas) vs the 5-family NGC ACC bake-off in `reframe_vs_bigthink.json` (measured batch-1 latency/energy).
- **A minor within-repo inconsistency to be aware of:** `escalation_levers.py` (and `beat32b_fusion.py`) score
  SLAKE-open with the greedy+seqlogprob **FALC fallback** (cheap = `GEN7`, esc ~53 %, 698.6 ms — a "slower cell"),
  whereas `integrated_method.py` uses the **bo8+verifier** SLAKE-open (605.5 ms, esc 12.6 %). The final method is the
  `integrated_method.py` treatment; the levers artifact operates on the earlier fallback and its pooled baseline
  (acc 0.5749, FLOPs 2.337) differs slightly from the integrated pooled (acc 0.5750, FLOPs 2.538) for this reason.
