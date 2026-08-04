# Progress — August 3, 2026 (the grounding pass — two silent constants, and the headline halves)

> **Follows `progress_July_30.md` after a three-day gap (see §0).** July 30's write-up closed with a
> register of things the project asserted but had never derived or measured. Today closed three of them,
> and **all three closed against the project**. The arc: (1) the **best-of-8 latency and energy
> measurement** — the 522 ms constant *was never measured*, and the truth is 1,305.3 ms; (2) the **FLOP
> ratio derivation** — the 4.57 that priced every compute claim for six weeks is the name-plate ratio
> 32/7, and the derived value is **R32 = 3.816**; (3) the **three-way headline recompute** — macro ×
> clean verifier × matched reasoning baseline, combined for the first time, taking the one surviving
> positive claim from **+0.0601 to +0.0325**; (4) the **comprehensive write-up** rebuilt to 4,149 lines
> with 7 figures and a `.docx`; and (5) an evening **verifier N-scaling analysis** which established
> that *there is no crossover at any N* — the two curves work against each other. **Both constants were
> caught by asking where a number came from, not by any experiment.** Almost everything today is
> CPU-only recomputation; the only GPU work is the batch-8 latency/energy measurement (§2). Every number
> is sourced to a named artifact; nothing is fabricated. **Abstention remains permanently out of scope**
> and appears nowhere.

---

## 0. The gap: 2026-07-31 → 2026-08-02

**Three days with no activity of any kind.** A `find` over the whole tree between 2026-07-30 20:00 and
2026-08-03 02:00, excluding `.git`, `__pycache__` and the two dependency repos, returns **nothing** — no
commit, no artifact, no log, no checkpoint. The first thing that moves is
`macro_average_headline_2026-07-30.json`, re-emitted unchanged at **02:56:16** as the session warmed up.

---

## 1. Why today's targets were chosen

They were not chosen today. July 30's write-up (§11.2, "documented but not verifiable in this pass")
listed them, and the register is what commissioned the work:

| register entry, as written 2026-07-30 | status then |
|---|---|
| the 32B/7B FLOP-eq ratio **4.57** | *"a hard-coded literal at 12 sites, reproducing only 32.0 B / 7.0 B = 4.571; an older document implies 4.34; **no file derives it**; ~7% margin claimed"* |
| the best-of-N batched latency **522 ms** | *"asserted, not measured"*; `GEN7 347.1 + VER7 175.5` |
| the macro × clean-verifier × matched-reasoning headline | *"Not computed anywhere on disk. The +0.0601 vs-reasoning figure is an upper bound."* |

The first two are the same species of defect and they are the day's real lesson: **a plausible-looking
constant that no file derives will silently price everything you claim.** Neither was caught by an
experiment. Both were caught by asking where a number came from and finding no answer.

---

## 2. The best-of-8 latency and energy — 522 ms was never measured (02:59)

`src/cascade_methods/bestofn_measure_batch8.py` → `bestofn_latency_energy_2026-08-03.json` and
`..._rep2.json`. Today's only GPU work.

### 2.1 Where 522 ms came from, and why it could not be right

Recorded verbatim in the artifact: `claimed_latency_ms 522.0`,
`claimed_latency_construction: "GEN7 347.1 + VER7 175.5 = 522.6 ms (rounded to 522)"`, status
**"MODELLED / ASSERTED — no batch-8 wall-clock was ever measured"**. The reasoning behind it was that
batching makes N drop out of latency, so best-of-8 costs one generation plus one verification.

Its companion energy figure was built on the *opposite* assumption: `claimed_energy_j 568.6`
= `8 × (GEN7_J 45.8 + VER7_J 25.28)`, i.e. **each of the 8 generations and 8 verifications costs its
full batch-1 energy**. The two together are physically impossible, and the artifact says so:

> *"568.6 J delivered in 0.5220 s requires 1089 W on the ONE A100 80GB PCIe the constants were measured
> on, whose NVML-enforced limit is 300 W (3.6x the limit) and which actually drew 132 W during GEN7.
> Physically impossible; the two figures cannot both be right."*

**The cost model had applied perfect parallelism to latency and zero parallelism to energy.**

### 2.2 The measurement

Protocol, verbatim: *"HF batch-1 request, cap320 (max_pixels=1280\*28\*28//4), real vqa_rad open items,
NVML integrated over visible GPUs, flash_attention_2, bf16; same recipe as
`src/cascade_methods/open_measure_latency_energy.py`"*. Lingshu-7B + adapter
`ckpts/train/lora_verifier_pooled4`; k = 8; **one** A100 80GB PCIe at a **300.0 W** enforced limit;
3 warm-up passes; `max_new` 32; temperature 0.7. **Two replicates**, n = 20 and n = 25.

| stage | pooled mean (n = 45) | rep1 / rep2 means |
|---|---:|---|
| gen ×1 | **350.0 ms** / 57.02 J | 360.4 / 341.7 ms |
| gen ×8 (batched) | **689.2 ms** / 158.86 J | 716.8 / 667.1 ms |
| verify ×1 | 275.4 ms (median 205.2) / 40.15 J | 294.2 / 260.3 ms |
| verify ×8 (batched) | **616.1 ms** / 157.84 J | 608.9 / 621.8 ms |
| **best-of-8 round trip** | **1,305.3 ms (median 1,290.7) / 316.7 J** | **1,325.7 / 1,289.0 ms; 322.72 / 311.89 J** |

Replicate agreement: latency **2.8%** apart, energy **3.4%** apart, and each replicate's mean lies inside
the other's p10–p90 (911–1,762 and 946–1,657 ms). **Harness validation:** the same harness measures the
canonical single-generation constant at **350.0 ms against 347.1 ms — a 0.8% difference** — so this is a
like-for-like replacement, not a different rig.

### 2.3 The verdicts

- **Latency: 522.0 → 1,305.3 ms, `factor_wrong: 2.5`.** Verbatim: *"WRONG — understated by ~2.5x.
  Batching 8 does NOT make N drop out."* Measured batch-8 speedup over 8 sequential calls is **~4.1×**
  for generation and **~3.6×** for verification — **not 8×**. Sequential-8 measures 4,441.6 ms.
- **Energy: 568.6 → 316.7 J, `factor_wrong: 0.56`.** *"ALSO WRONG, in the other direction — overstated
  by ~1.8x. Batching does save energy (shared prefill/weight reads), just not latency."* The measured
  pair implies **242.6 W** against the card's 300 W limit — physically sane. *(The "400 W TDP" in
  earlier drafts was also wrong, and the ≥1.42 s energy-consistent bound had been computed from the
  overstated energy, which is why the true 1.305 s sits below it.)*
- **The figure that actually breaks** is not the one against the reasoning baseline — that barely moves
  in sign, because the reasoning baseline is 10.5 s. It is the one against a **single 32B forward**: the
  open best-of-8 arm goes from **17.5% faster than always-32B-direct to 2.0× slower**
  (`flips_from_faster_to_slower_than_32b: true`). Verbatim: *"Under ANY serving assumption (sequential
  4442 ms or batched 1305 ms) the open-text best-of-8 arm is SLOWER than simply calling the 32B once."*

> **A naming correction that travels with this.** The published sentence *"+0.0601 at −87.7% latency"*
> took −87.7% from the **batched** axis, not batch-1. The batch-1 figure for the identical cell is
> **−73.6%**. Batch-1 latency is unaffected by the refuted assumption, so batch-1 is now the primary
> latency axis and the batched axis is labelled wherever it appears.

> **A new, unresolved flag raised by the same harness.** It measures **57.0 J** for one greedy 7B
> generation (replicates 55.9 / 57.9) against the **45.8 J** the cost model charges — **+24.5%** — while
> reproducing latency to +0.8%. Likely candidates are a different idle-power treatment (idle with model
> resident measured at 83.8–86.3 W) or a different integration window; **neither was confirmed.** If
> 57.0 J is right the correction again runs *against* the method, which runs more 7B forwards than the
> baselines do.

---

## 3. The FLOP ratio: 4.57 is rejected, R32 = 3.816 (03:13 – 03:19)

`src/cascade_methods/flop_ratio_derivation.py` → `flop_ratio_derivation_2026-08-03.json`, then
`flop_ratio_impact.py` → `flop_ratio_impact_2026-08-03.json`. CPU-only; safetensors **headers** read, no
weights loaded.

### 3.1 The provenance problem

The literal is defined at `src/cascade_methods/lingshu_medeval_cascade.py:21` (`R7=1.0; R32=4.57`) and
reused at **12 sites**, all agreeing except `honest_recosting.py:144` which carries `4.571`. Its
`only_stated_derivation`, verbatim: *"32.0B / 7.0B = 4.571 … name-plate sizes, neither of which is
either model's true parameter count."* A **different and incompatible** constant `4.34 (= 33.0e9/7.6e9)`
survives in `open_bestofN_adaptive.py:14` and `src/analysis/cascade/cascade_cost_prefill_flops.py:33`.

**A falsifier was declared in advance**, and it is what makes this a test rather than a rationalisation:
*"If the derived ratio had landed inside [4.4, 4.7], 4.57 was fine and only its provenance was missing.
It landed at 3.816, outside every plausible reading."*

### 3.2 The derivation

**Parameter counts [measured, safetensors headers]:**

| | Lingshu-7B | Lingshu-32B |
|---|---:|---:|
| **total** | **8,292,166,656** | **33,452,718,336** |
| lm_body | 6,525,621,760 | 31,206,740,992 |
| vision tower + merger | 676,550,144 | 688,841,984 |
| embed_tokens | 544,997,376 | 778,567,680 |
| lm_head | 544,997,376 | 778,567,680 |

Naive total-parameter ratio **4.034**; lm_body-only ratio **4.782**. The vision towers **differ by only
1.8%**. The analytic lm_body count from `config.json` matches safetensors exactly.

**Token geometry [measured, n = 25 VQA-RAD non-yes/no items, cap320]:** prompt **326.68** tokens, of
which **280.48 are image**; 46.2 text; 1,121.92 patches; generated 5.64 (7B) / 5.60 (32B). The
reconstruction reproduces the prefill recorded in `logs/latency_opentext.jsonl` **exactly to two
decimals**.

**The component arithmetic** (GFLOP at T = 326.68, M = 280.48, P = 1,121.9):

| component | 7B | 32B | share of 7B | share of 32B |
|---|---:|---:|---:|---:|
| vision tower (dense + attn + merger) | 1,479.12 | 1,493.47 | 25.4% | 6.7% |
| **lm prefill (dense + attn)** | **4,285.00** | **20,459.18** | **73.5%** | **91.9%** |
| lm decode (dense + attn) | 61.17 | 289.08 | 1.05% | 1.30% |
| lm_head | 6.15 | 8.72 | 0.11% | 0.04% |
| **TOTAL** | **5,831.45** | **22,250.45** | | |

> **`"Ratio = 22250 / 5831 = 3.816."`**

**Sensitivity across every reading**: same G both legs 3.816 · no causal halving 3.814 · prefill only
3.808 · **MCQ cap320 3.859** · **MCQ fullres 3.734** · image-dominated limit 3.716 · **pure-decode limit
4.524** · total-param 4.034 · lm_body-only 4.782 · name-plate 4.571.

**Why 4.57 is wrong, verbatim (abridged):** *"…it applies a parameter ratio to a quantity — a whole VLM
forward pass — that is not proportional to total parameters, because (i) the ~0.68 B vision tower is
shared and nearly identical in both models, (ii) the 0.545 B / 0.779 B embedding table costs 0 FLOPs,
and (iii) the lm_head is applied to O(G)=~6 positions, not to the whole prompt. Coincidentally 4.57 is
close to the pure-DECODE ratio 4.524 — it is roughly the right constant for a decode-only workload and
the wrong one for this prefill-dominated one."* **These arms emit 2–6 tokens.**

**Physical cross-check** (`logs/latency_opentext.jsonl`, n = 25, batch-1): measured latency ratio
**1.916**, energy ratio **2.77** (Lingshu) and **3.808** (MedVLThinker) — *"BOTH BELOW the derived 3.82
and far below the charged 4.57. That ordering is the physically expected one: at batch 1 the small model
is more severely under-utilised (implied MFU 5.4% vs 10.7%) … Nothing in the measurements supports 4.57
over 3.82, and the MedVLThinker energy ratio 3.81 lands almost exactly on the derived 3.82."*

> **⚠ A loose label, flagged rather than repeated uncritically.** The artifact recommends
> **"R32 = 3.82 ± 0.15, band [3.734, 3.859]"**, and the write-up repeats that phrasing in nine places.
> The two are not arithmetically consistent: `[3.734, 3.859]` is a **width** of 0.125, i.e. **±0.063**
> around 3.796. The ±0.15 only works as a *total width* over the wider sensitivity span 3.716→3.859.
> **The derivation is sound; only the ± label is loose**, and it should be restated as a band, not a
> symmetric interval.

### 3.3 The impact: no claim changes status, and every one gets worse

`flop_ratio_impact.py` re-executes `macro_average_headline.run()` **unmodified** with R32 monkeypatched
into six consumers; latency and energy are untouched. Two gates pass: at 4.57 every published FLOP figure
reproduces exactly, and the cost function is linear in R32 to `max_abs_error 0.000889`.

| weighting | claim | published × | **corrected ×** | change |
|---|---|---:|---:|---:|
| sample-weighted | compute-lean | 0.492 | **0.556** | +13.0% |
| sample-weighted | accuracy-max-veto | 0.932 | **0.989** | +6.1% |
| macro (8 cells) | compute-lean | 1.196 | **1.362** | +13.9% |
| macro | accuracy-max-veto | 1.410 | **1.554** | +10.2% |
| macro | accuracy-max-fusion | 1.435 | **1.579** | +10.0% |

**Why lowering the ratio makes the method look worse**, which is counter-intuitive and is stated in the
artifact's caveats: *"the method's cheap leg is charged in 7B units, so lowering R32 makes the 32B
baseline cheaper faster than it makes the method cheaper."*

**Verdict, verbatim:** *"NO COMPUTE CLAIM CHANGES STATUS; every one gets modestly WORSE … Both were
already >1x, i.e. already NOT compute-negative, and remain so. Sample-weighted, compute-lean 0.492x ->
0.556x: still <1x, so the one surviving compute-negative statement (sample-weighted only) survives, with
a thinner margin. The published '~7% margin of error' on the constant was an UNDERESTIMATE in magnitude
(4.57 -> 3.816 is -16.5%) but correct in direction of harmlessness."*

**One documented table changes:** the FLOPs break-even escalation rate moves **0.7812 → 0.7379** —
*"'e > ~78% -> worse on everything' becomes 'e > ~74%'"*. Latency (0.4782) and energy (0.6394)
break-evens are **measured**, not R32-dependent, and are unchanged. No cell changes verdict: MedXpert at
89.60% escalation was already past the threshold under both constants.

---

## 4. The three-way headline recompute (04:14; committed 04:17)

`src/cascade_methods/headline_three_way.py` → `headline_three_way_2026-08-03.json`. 10,000 replicates,
seed 20260730, runtime 499.9 s. **Three corrections that had never been applied together**, on one
bootstrap stream, with the grounded constants added last.

**Three validation gates, all exact, run before anything new is computed:** the
contaminated-and-unmatched configuration at R32 = 4.57 reproduces `macro_average_headline_2026-07-30.json`
on **1,224 fields exactly**; the clean-L1 configuration reproduces
`macro_headline_clean_verifier_2026-07-30.json` column C exactly; and the matched judge files reproduce
`matched_prompt_reasoning_2026-07-29.json`'s per-dataset accuracies exactly.

**The progression** — accuracy-max-veto against always-32B-with-reasoning:

| column | correction added | method | baseline | **Δ** | 95% CI | Δ change | FLOP × | batch-1 lat | energy |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | *(published)* sample-weighted, contaminated, unmatched | 0.5836 | 0.5591 | **+0.0245** | [+0.0217, +0.0274] | — | 0.874× | −58.4% | −71.9% |
| 2 | + **macro** (1/8 per cell) | 0.6694 | 0.5974 | **+0.0720** | [+0.0614, +0.0824] | **+0.0475** | 1.131× | −78.8% | −87.3% |
| 3 | + **clean L1 verifier** | 0.6575 | 0.5974 | **+0.0601** | [+0.0498, +0.0703] | **−0.0119** | 1.396× | −73.6% | −84.3% |
| 4 | + **matched reasoning baseline (arm B)** | 0.6575 | **0.6250** | **+0.0325** | [+0.0237, +0.0412] | **−0.0276** | 1.453× | −67.7% | −80.6% |
| 4a | *(arm A instead of B)* | 0.6575 | 0.6250 | +0.0325 | [+0.0240, +0.0411] | 0.0000 | 1.461× | −66.7% | −80.0% |
| **5** | + **grounded R32 = 3.816** | 0.6575 | 0.6250 | **+0.0325** | **[+0.0237, +0.0412]** | **0.0000** | **1.608×** | **−67.7%** | **−80.6%** |

**The claim SURVIVES, remains significant, and is 45.9% smaller.** The shrinkage is *entirely*
correction 3 — the R32 correction changes accuracy by exactly **0.0000** ("every system's macro accuracy
is identical to four decimal places"), it only raises the price.

**What the shrinkage actually is, and it is not the method getting worse.** It is **the baseline getting
its persona and answer-style clause back.** The published reasoning arm had dropped both, and on free
text that is a live grading channel. The same 32B, told to reason *and* to answer in a short specific
phrase, scores:

| open cell | n | unmatched | matched A | **matched B (primary)** | mean gen tokens (unmatched → B) |
|---|---:|---:|---:|---:|---|
| SLAKE-open | 645 | 0.6791 | 0.7194 | **0.7318** | 122.41 → 86.43 |
| VQA-RAD-open | 200 | 0.5450 | 0.5550 | **0.5550** | 104.54 → 45.80 |
| PathVQA-open | 1,500 | 0.1087 | 0.2787 | **0.2667** | 141.47 → 101.49 |

**PathVQA-open alone moves +0.158.**

**The multiple-choice half was handled explicitly, not reused** (a new methodology section, §4.11.3).
On three of the five multiple-choice cells the "reasoning" dump emits 3.01–3.33 tokens and agrees with
the direct run 92–97% of the time — *there is no reasoning behaviour to prompt-match*; PathVQA-closed has
no reasoning dump at all; and MedXpert's reasoning arm was re-tested directly (matched trigger effect
**+0.0035 [−0.0185, +0.0250], not significant, n = 2,000**). So the matched protocol changes the
multiple-choice half by **exactly zero**. **Genuine 32B reasoning exists on 4,345 of 42,224 items =
10.3% of the pool.** The strictest reading of the protocol gives **+0.0313**.

**What did not change:** accuracy-max still **ties** always-32B-direct (**+0.0008 [−0.0022, +0.0036]**),
compute-lean still **loses** (**−0.0124 [−0.0189, −0.0061]**), fusion still loses (−0.0063). Across the
36 tracked claims, **23 are unchanged and 13 flip** (6 to LOSS for compute-lean, 5 WIN→TIE, 4 WIN→LOSS
for fusion).

**The final cost table** (8-cell macro, clean L1, matched arm B, R32 = 3.816):

| operating point | FLOP-eq | vs 32B-reasoning | vs 32B-direct |
|---|---:|---|---|
| **accuracy-max-veto** | **7.342** | **1.608×**, −67.7% batch-1 latency, −80.6% energy | **1.924×**, +149.6% latency, +101.0% energy |
| compute-lean | 6.358 | 1.392×, −69.3%, −82.6% | 1.666×, +136.8%, +80.2% |
| always-32B-direct | 3.816 | — | — |
| always-32B-with-reasoning (honestly re-costed) | 4.567 | — | — |

**Concentration:** leave-one-cell-out on the final +0.0325 gives a range of **[+0.0203, +0.0378]**; the
claim is **carried by PathVQA-open** (dropping it → +0.0203) and held back by SLAKE-closed. Band
sensitivity: at R32 = 3.734 the claim costs 1.628×; at 3.859, 1.597×; the accuracy delta is unchanged
everywhere in the band.

**The honest headline, verbatim:**

> **"Against a 32B made to reason with a prompt-matched instruction, the accuracy-max cascade is
> +0.0325 [+0.0237, +0.0412] on an 8-cell macro average, at −67.7% batch-1 latency, −80.6% energy — and
> 1.608× the honestly re-costed FLOP-equivalents. It ties a single direct 32B call (+0.0008 [−0.0022,
> +0.0036]) at 1.924× its compute. The cheaper setting is a significant loss."**

**Four caveats travel with it, none of them new:** the baseline arm is a **mixture** that fires a trace
on only 30.5–71.1% of items, so **+0.0325 is a lower bound** against a fully reasoning 32B; genuine
reasoning exists on 10.3% of the pool; the claim is carried by PathVQA-open; and **the comparison is
against a baseline nobody should deploy** — turning reasoning off is worth **+0.0317** on its own
(always-32B-direct 0.6567 against matched-reasoning 0.6250, at 665 ms vs 5,137 ms and 127 J vs 1,318 J).
*That prompt change requires no second model, no gate and no verifier, and it is still larger than every
method delta in this project combined.*

---

## 5. The write-up, its figures, and the `.docx` (05:53 – 06:08)

`COMPREHENSIVE_WRITEUP_2026-08-03.md` — **4,149 lines / 45,344 words**, up from 3,294 / 34,756
(**+855 lines, +10,588 words**). The 07-30 file was retained with a 28-line SUPERSEDED banner prepended
rather than deleted. **15 headings added, 0 removed, 1 reworded** (§11.3's date). No section was deleted.

New material: **§4.11** (where the cost constants come from — the derivation, the measurement, the
multiple-choice handling, and the arm-choice bias direction), **§4.12** (where the two silent constants
came from), **§5.9.1–5.9.5** (the grounding pass and the final headline, with §5.9.4's cost table
replacing §5.6.3 for all forward use), **§6.1.4 / §6.1.5** (both constants as corrections-log entries),
**§6.8** (the +0.0601 → +0.0325 account), **§11.0** (a four-row *closed since 2026-07-30* register giving
the old value against the true one), and a new transferable lesson **T12 — "A plausible constant that no
file derives will price everything you claim."**

**Seven figures**, `paper/make_writeup_figs.py` (731 lines), 200 dpi, every canvas exactly 6.5 in wide.
Every data literal is hard-coded with a comment naming the artifact and JSON key path; the module banner
reads *"NO FABRICATED NUMBERS. … Nothing is recomputed, smoothed, rounded-for-looks, or filled in."*

| figure | source | what it shows |
|---|---|---|
| `fig_correction_cascade` | `headline_three_way` | the five-column waterfall +0.0245 → +0.0720 → +0.0601 → +0.0325 → +0.0325, CIs on each level |
| `fig_finding1_crossfamily` | `finding1_corrected_2026-07-29` | 5 families × 7 benchmarks, Δ(reasoning − direct), P1 arms; 17/20 perception cells negative; reasoning cells hatched as *not answer-format-controlled* |
| `fig_format_vs_trigger` | `medeval_matched_direct_2026-07-29` | the 9 cells decomposed into format vs trigger; 3/9 format significant, 0/9 trigger |
| `fig_accuracy_cost` | `headline_three_way` | macro accuracy against FLOP-eq for 7 systems, R32 line + band |
| `fig_verifier_contamination` | `verifier_disjoint_retrain_2026-07-30` | selection gain per arm, and the mechanism: AUROC 0.9433/0.8856/0.7960 against conversion 0.5894/0.2029/−0.0676 |
| `fig_pmc_defects` | `pmc_label_noise_audit_2026-07-29` | 53% / 60% / 28% with Wilson intervals and the three Fisher p-values |
| `fig_latency_correction` | `bestofn_latency_energy_2026-08-03` | 522 ms asserted against both replicates' p10–p90, pooled 1,305.3, one 32B forward at 665 ms |

> **⚠ One caption over-describes its figure.** §5.9.4's caption under `fig_accuracy_cost` promises
> *"batch-1 latency and energy as companion panels"* and *"ghosted markers … at the superseded
> R32 = 4.57"*. The code draws **one** axes and **no** ghosted markers; 4.57 appears only as the text
> string *"DERIVED, not name-plate (4.57 rejected)"*. The caption should be corrected or the panels
> added — this is exactly the figure-against-text drift §6.5 of the write-up warns about.

**The `.docx` was built with pandoc, installed rather than hand-rolled.** The repo already contained a
python-docx converter (`paper/archive/scripts/md2docx.py`) that built the June documents, but a 69-table,
7-figure, 156-heading document with a live table of contents was the wrong job for it. `sudo apt-get
install pandoc` was unavailable, so a standalone binary was fetched into the same repo-local `tools/`
directory that already holds `tectonic`:

```bash
cd tools
VER=3.1.11
curl -fsSL --max-time 180 -o pandoc.tar.gz \
  "https://github.com/jgm/pandoc/releases/download/${VER}/pandoc-${VER}-linux-amd64.tar.gz" \
  && tar xzf pandoc.tar.gz && cp pandoc-${VER}/bin/pandoc . && chmod +x pandoc \
  && rm -rf pandoc-${VER} pandoc.tar.gz
```

then

```bash
./tools/pandoc "$D/COMPREHENSIVE_WRITEUP_2026-08-03.md" \
  --from=gfm --to=docx --toc --toc-depth=3 --number-sections \
  --reference-doc=/tmp/ref_styled.docx --resource-path="$D" \
  -o "$D/COMPREHENSIVE_WRITEUP_2026-08-03.docx"
```

The **only** hand-rolled part is the reference document: pandoc's own
`--print-default-data-file=reference.docx` restyled by a short python-docx snippet (`Compact` → 8 pt so
the ten-column tables fit portrait; `Table Caption` → 8 pt; `Normal` / `Body Text` → 10.5 pt;
`Source Code` and `Verbatim Char` → 8 pt Consolas).

**Structural verification, recorded in the commit:** *"69 tables in the docx against 69 in the markdown,
7 of 7 images embedded at 5.83in… 156 headings, zero degenerate tables, live TOC field present, widest
table 7x10"* — with the honest gap stated in the same breath: *"NOT verified visually: neither
LibreOffice nor Word is available on this machine."*

> **A reproducibility caveat worth carrying forward.** The `.docx` is a committed 1.37 MB binary whose
> toolchain is **not reproducible from the repo**: `tools/` is gitignored (`.gitignore:51`), no build
> script exists, `/tmp/ref_styled.docx` is volatile, and `grep -r pandoc` over tracked files returns
> nothing. The build command should be committed as a runner.

---

## 6. Verifier N-scaling: there is no crossover (12:05; committed 12:07)

`src/cascade_methods/verifier_n_scaling.py` → `verifier_n_scaling_2026-08-03.json`. Fully offline, no
GPU, no new inference. The question, verbatim: *"Does the trained verifier's benefit keep growing with
N, and can 7B + verifier at higher N match/beat a 32B without ever calling the 32B at test time?"* —
i.e. is there a **crossover N** at which sampling alone replaces the strong model.

**Method.** The **clean L1** verifier (`ckpts/train/lora_verifier_disjoint`) and the same
`run_judge.py` (MedVLThinker-32B) grader as the headline; oracle@N, verifier@N and self-consistency@N are
**exact expectations over all C(8,N) subsets** (255 per question, enumerated — no Monte-Carlo); ties are
resolved as uniform random tie-breaks; CIs are a question-level bootstrap, 4,000 resamples. Coverage is
extrapolated by a zero-inflated beta-binomial **validated out of sample** against an **independent
16-sample generation** on the same items. Harness validation: recomputation matches the published cell to
`max_abs_diff 7.4e−05`; switching the tie-break from argmax-first to uniform moves verifier@8 from
0.4853 to 0.4841, *"does not touch any conclusion."*

**The measured curve, pooled (n = 2,345):**

| N | oracle@N | verifier@N (clean) | **selection efficiency** |
|--:|--:|--:|--:|
| 1 | 0.4233 | 0.4233 | 1.0000 |
| 2 | 0.4936 | 0.4554 | 0.9226 |
| 4 | 0.5604 | 0.4731 | 0.8443 |
| 5 | 0.5816 | 0.4769 | 0.8200 |
| 6 | 0.5990 | 0.4797 | 0.8009 |
| 8 | **0.6260** | **0.4841** | **0.7733** |

Baseline: **always-32B-direct 0.5168**. The measured anchor is a *measurement*, not an extrapolation:
verifier@8 − 32B = **−0.0328 [−0.0512, −0.0152]**.

**The two curves, and the reason there is no crossover:**

- **Selection efficiency FALLS `0.07612` per doubling of N, 95% CI [0.0687, 0.0832]** — from 1.000 at
  N = 1 to 0.7733 at N = 8. Verbatim: *"NOT constant and NOT rising -- FALLING."* Marginal conversion of
  each doubling collapses: 0.457 (1→2), 0.265 (2→4), **0.167 (4→8)**.
- **Coverage GROWS sub-logarithmically.** Measured per doubling: +0.070 (1→2), +0.067 (2→4), +0.066
  (4→8), **+0.052 (8→16)** — the last measured on the **independent** 16-sample pool, oracle@16 = 0.6695
  against a model prediction of 0.6644 (relative error −0.76%, a real out-of-sample test).

**0.052 of coverage cannot pay for 0.076 of lost efficiency.** Multiplying the two, `acc(N) =
coverage(N) × sel_eff(N)`, the projected accuracy **peaks at N = 5 (0.4794) and declines from there** —
0.4779 at N = 8, 0.4615 at 16, 0.4350 at 32, 0.4002 at 64, 0.3107 at 256 — and `beats_32B_central_case`
is **false at every N**.

> **A precision note, because the shorthand is easy to over-read.** The *measured* curve over the
> existing 8-sample pool is still monotonically increasing through N = 8 (0.4233 → 0.4841); it has simply
> gone **flat** — the last three samples buy +0.002 each. The N = 5 peak lives in
> `projection_vs_32B/…/best_achievable_over_all_N`, computed from the *extrapolated* coverage model. The
> honest statement is the artifact's own: *"verifier-selected accuracy is already flat by N~5-8 and
> declines beyond it."*

**Crossover N: `null`** under the measured trend **and everywhere in its 95% CI**. Verbatim: *"there is
NO crossover under the measured trend or anywhere in its 95% CI. The only scenario that crosses (N=18)
requires the selection-efficiency decline to stop dead at N=8, which the data rejects. What the N=3
perfect-selector figure says is that the SAMPLES are already good enough -- the SELECTOR is not."* Even
that counterfactual N = 18 would cost **19.20 7B-forward-equivalents = 5.03× a single 32B forward** and
2.16–2.61 s of wall clock against 665 ms.

**Independently corroborated by the memorising verifier.** The *contaminated* adapter measured to K = 16
(n = 1,621) shows the same shape: Δoracle +0.0506 but Δverifier only +0.0074, marginal conversion 0.146,
Δsel_eff **−0.0533**. *"Even the MEMORISING verifier saturates."*

**And the decomposition says which wall to attack.** oracle@8 = 0.6260 **already exceeds**
always-32B-direct 0.5168 — *a perfect selector over the existing 8-sample pool would beat the 32B by
+0.1092.* Of the questions the 32B gets right and 7B+verifier@8 misses, **47% are coverage-limited and
53% are conversion-limited**. Doubling the pool 8→16 removes only 0.0435 of the coverage hole, while the
selection hole at N = 8 is 0.1419 **and grows with N**. Verbatim: *"more samples is the wrong lever. The
lever is a selector that does not degrade with pool size (the confident-distractor problem), and that is
exactly the selection wall this project has already hit thirteen independent ways."*

---

## 7. Standing state (end of 2026-08-03) and open questions

**The headline, final for now:** against a prompt-matched reasoning 32B, accuracy-max is **+0.0325
[+0.0237, +0.0412]** at −67.7% batch-1 latency and −80.6% energy, costing **1.608×** the honestly
re-costed FLOP-equivalents. Against a single **direct** 32B forward it is a **tie** (+0.0008) at
**1.924×** the compute, and compute-lean is a **significant loss**. **The win is on wall-clock and
joules, never on compute.**

**The practical recommendation that outranks the entire method** is unchanged and now better grounded:
**turn reasoning off** — worth +0.0317 for a prompt change.

**What today added to the corrections log:** §6.1.4 (4.57 rejected), §6.1.5 (522 ms rejected and its
energy wrong the other way), §6.8 (the three corrections combined), and T12. Three entries left the
unverified register.

**Open questions, in the artifacts' own ranking:**

1. **Generator work, not verifier work.** 434 of 1,064 held-out open questions (**40.8%**) have no
   correct answer in the pool at all, against a total selection gap of 0.0912. **Generator ideas compete
   for +0.408; verifier ideas compete for at most +0.091.** Today's N-scaling result sharpens this: the
   *selector*, not the sample count, is the binding constraint.
2. **Nothing has been repointed to the derived constants.** `R32 = 3.816` and the measured best-of-8 cost
   exist **only** in `headline_three_way.py` and today's artifacts. Twelve-plus modules still carry
   `4.57`/`4.571`, two carry an incompatible `4.34`, and `method_final*.json` /
   `integrated_pandora_opentext.json` still carry the 522 ms family. **Any figure or deck regenerated
   today republishes a rejected constant.** CPU-only, no new inference — *"the cheapest high-value item
   in the list, and it is a correctness liability, not a nicety."*
3. **The 7B energy constant is unreconciled** — 57.0 J measured against 45.8 J charged (+24.5%), with
   latency reproducing to +0.8%. One instrumented run with the idle baseline logged explicitly.
4. **PathVQA-open still carries the claim** (dropping it takes +0.0325 → +0.0203), and it is still a
   non-random prefix of 1,500 of 3,357 items over-sampling a degenerate taxonomy family (0.632 against
   0.562), judged by a judge validated on SLAKE and VQA-RAD but **not** on PathVQA. The ten-minute check
   that this prefix is not topically biased has still never been done.
5. **The `± 0.15` on R32 should be restated as a band** (§3.2), and **`fig_accuracy_cost`'s caption
   should be reconciled with what the code actually draws** (§5).
6. **The `.docx` toolchain is not reproducible from the repo** (§5).
7. **The macro-objective refit** — the thresholds are tuned for a pooled objective and reported on a
   macro one, so the multiple-choice loss (−0.0070) may be an artifact of that mis-specification.
   CPU-only refit over existing dumps.
8. **Multiplicity is uncontrolled** in two places — 18 policy-selection tests and 25 veto certifications.
   The prescribed Holm corrections have not been run.
