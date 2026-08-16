# A cheap verifier on a 7B medical VLM — the central table

**2026-08-16 · baseline = ALWAYS-7B · every number names its artifact**

---

## VERDICT (first line)

**A small trained verifier improves Lingshu-7B on 2 of 8 benchmark cells under the 32B judge (1 of 8
under exact match), by +0.0462 macro-3 / +0.0173 macro-8 judge (+0.0037 / +0.0014 EM), at 4.56× the 7B's
compute on the three cells it touches and 2.33× macro-8 — down from 16.73× / 6.90× at the start of the
round.** Y is **not** ~1, but it is no longer ~11–17×: the two structural fixes are real, they are a TIE
in both currencies, and they cut the open arm 3.67×. The honest summary is that **the cost problem is
now mostly solved and the accuracy claim is thin** — under normalised exact match the method is worth
+0.0014 on the 8-cell macro, which is not a paper on its own.

**The round's most tempting headline is REFUTED.** "Drop the LoRA, keep the free head, ship at 1.20
FLOP-eq" is a **judge-only artifact**: head-only beats always-7B by +0.0452 under the judge and is
**negative on all three cells under exact match** (pooled −0.0068 [−0.0226, +0.0090]). It must not be
claimed. See §6.

---

## 1. What was verified adversarially, and what broke

Everything below was recomputed from the raw caches by
`src/cascade_methods/central_table_verify.py` → `artifacts/central_table_2026-08-16.json`. The three
build artifacts were **cross-checked, not trusted**.

| check | result | verdict |
|---|---|---|
| Frozen null test (`src/training_methods/genframe_data.py`) | max abs deviation **3.5967e-07**; sel_eff 0.775204, oracle@8 0.626013, greedy 0.449467, n=2345/1468 | PASS |
| Identity `selected = oracle@8 × sel_eff` | ≤ **1.11e-16** on every arm | PASS |
| Free-head capture alignment to the frozen loader | 0 missing, 0 unmapped of 8943 rows | PASS |
| Verifier FLOP re-derivation vs BUILD 2 | deployed 7.4653992685576815, prefix 2.186187155120831 — **deviation 0.0** | PASS |
| Prefix-sharing endpoint vs in-session deployed control | judge +0.000426 [−0.003838, +0.005117] TIE; EM +0.002132 [−0.002559, +0.006823] TIE; 134/2345 picks changed | PASS (trap b cleared) |
| Permutation null (frozen head fed row-permuted features) | 0.45471 ± 0.00382 (10 seeds) vs real 0.50576 → **13.35 σ** above null; random-pick floor 0.42335 | PASS (trap d cleared) |
| **Trap (a): are the captured states equivalent?** | **NO — and it does not matter.** Not bit-identical (max abs dev 1.125 on features peaking at 105; bf16 reduction order, not a definitional difference). Endpoint effect is a TIE. | CLEARED, with the deviation stated |
| **Trap (c): is the open-form gain a grading artifact?** | **NO for the PathVQA prompt fix** (judge +0.038370 vs EM +0.038073 — the two currencies agree to 0.0003). **NO for PMC self-consistency** (win present in both graders; the *defective* grader gives the *smaller* delta). | CLEARED |
| **The trap nobody flagged: head-only is judge-only** | **FOUND.** `verifier_restructure_2026-08-16.json` reports `vs_greedy7b_judge` for head-only with **no EM counterpart**, and pairs its judge-only accuracy (0.606173) with the cheapest cost row (1.076 macro-8 FLOP-eq). Under EM head-only is negative on 3/3 cells. | **CAUGHT — see §6** |

---

## 2. THE CENTRAL TABLE — always-7B baseline

**Method:** best-of-8 sampling from Lingshu-7B, picked by the frozen rank-fusion selector = trained LoRA
verifier (`ckpts/train/lora_verifier_disjoint`, prefix-shared) **+** generator-frame head
(`ckpts/train/genframe_head_ens8`, captured free during generation). N=8, T=0.7 — the pools the accuracy
is measured on.

Source: `artifacts/central_table_2026-08-16.json` → `ARMS_vs_always_7B.DEPLOYABLE_freehead_plus_prefix`.
Paired item bootstrap, nboot=10000, seed 20260816.

| cell | n | 7B greedy | 7B + verifier | Δ judge (95% CI) | Δ EM (95% CI) | fwd passes | FLOP-eq vs 7B | guardrail |
|---|---:|---:|---:|---|---|---|---:|---|
| **SLAKE_open** | 645 | 0.736434 (j) / 0.762791 (em) | 0.772093 / 0.773643 | **+0.035659 [+0.009302, +0.060465] WIN** | +0.010853 [−0.015504, +0.035659] TIE | 8 gen + 1 prefill + 3.82 tails | **4.556** | judge ✅ / EM ✅ |
| **VQA_RAD_open** | 200 | 0.465000 / 0.425000 | 0.500000 / 0.400000 | +0.035000 [−0.010000, +0.085000] TIE | −0.025000 [−0.070000, +0.020000] TIE | " | **4.556** | judge ✅ / EM ⚠️ **negative** |
| **PATH_VQA_open** | 1500 | 0.324000 / 0.326667 | 0.392000 / 0.352000 | **+0.068000 [+0.050000, +0.085333] WIN** | **+0.025333 [+0.007333, +0.043333] WIN** | " | **4.556** | judge ✅ / EM ✅ |
| pooled (3 open cells) | 2345 | 0.449467 / 0.455011 | 0.505757 / 0.472068 | **+0.056290 [+0.042217, +0.070362] WIN** | **+0.017058 [+0.002985, +0.031130] WIN** | | 4.556 | |
| — | | | | | | | | |
| PMC_VQA | 33430 | 0.542656 | *not applicable* | — | — | 1 | 1.000 | intrinsically MCQ |
| MedXpertQA-MM | 2000 | 0.261500 | *not applicable* | — | — | 1 | 1.000 | intrinsically MCQ |
| SLAKE_closed | 2094 | 0.825359 | *not applicable* | — | — | 1 | 1.000 | REFUTED — see §5 |
| VQA_RAD_closed | 451 | 0.780876 | *not applicable* | — | — | 1 | 1.000 | REFUTED — see §5 |
| PATH_VQA_closed | 6719 | 0.840869 | *not applicable* | — | — | 1 | 1.000 | REFUTED — see §5 |
| **MACRO-8** | | **0.5970868** | **0.6144191** | **+0.017332 (judge)** | **+0.001398 (EM)** | | **2.334** | |

**Cells the method applies to: 3 of 8.** **Cells with a CI-clean improvement: 2 of 8 (judge), 1 of 8
(EM), 1 of 8 in BOTH currencies simultaneously (PATH_VQA_open).**

**Latency is reported per component and NOT combined**, because nothing in this project has been run end
to end (standing caveat, CLAUDE.md §0). Measured: generation N=8 vs N=1 wall clock **1.790×**
(`verifier_restructure_2026-08-16.json` Q1 `time|default`); verifier per question on an idle A100,
batch 1, HF, 5 reps — deployed 0.9049 s → prefix-shared 0.5620 s (**1.610×**) → batched tails 0.2990 s
(**3.027×**) (`shared_prefix_verifier_2026-08-16.json` VRAM_AND_CLEAN_LATENCY, max_pixels 1003520).
VRAM (convention (d), process footprint, idle card): 17.0279 GiB mean → **16.8544 GiB** — prefix sharing
is *lower* on all four conventions because it never holds N full prompts' activations.

### Why the two currencies disagree so much

`selected` under the judge is 0.505757 but 0.472068 under EM, against baselines of 0.449467 and 0.455011.
The judge is more generous to the *sampled* answers than to the *greedy* one. This is the project's known
paraphrase-drift exposure (CLAUDE.md §0), and it is why **both currencies are reported for every cell**.
The EM column is the conservative reading and it is the one that should govern the paper's claim.

---

## 3. COST ACCOUNTING — where Y actually went

**Unit: 1.0 FLOP-eq = one Lingshu-7B cap320 open-text forward+generate = 5831.42 GFLOPs**
(`cost_decomposition_2026-08-12.json:N2`). always-7B = 1.0/question.
Source: `artifacts/central_table_2026-08-16.json` → `COST_TABLE`, `VERIFIER_FLOP_REDERIVATION`.

| configuration | generation | head | verifier | open q | macro-8 |
|---|---:|---:|---:|---:|---:|
| always-7B baseline | 1.000 | 0 | 0 | **1.000** | **1.000** |
| deployed at round start (N=8, T=0.7) | 2.370 | 6.893 | 7.465 | **16.728** | **6.898** |
| + free head only (BUILD 1) | 2.370 | 0.00002 | 7.465 | 9.835 | 4.313 |
| + prefix-shared verifier only (BUILD 2) | 2.370 | 6.893 | 2.186 | 11.449 | 4.918 |
| **THIS ROUND — both (N=8, T=0.7)** | 2.370 | **0.00002** | **2.186** | **4.556** | **2.334** |
| + T=0.4 at N=4 (parity; cost PROJECTED) | 1.665 | 0.00002 | 2.117 | 3.781 | 2.043 |
| + vision-embeds fix (accuracy NOT re-measured) | 1.203 | 0.00002 | 2.186 | 3.389 | 1.896 |
| *control: prefix caching OFF* | 8.244 | 6.893 | 7.465 | 22.603 | 9.101 |

**Open-question cost 16.728 → 4.556 = 3.67× cheaper. Macro-8 6.898 → 2.334 = 2.96× cheaper.**
And this costs **nothing in accuracy**: the full refactor vs the shipped fusion is
**judge −0.001706 [−0.007676, +0.004264] TIE and EM +0.001706 [−0.004691, +0.008102] TIE**, 216/2345
picks changed (`central_table_2026-08-16.json:DEPLOYABLE_minus_SHIPPED`).

### Three corrections to the project's cost model, all in this round

1. **Generation was over-charged 3.38×.** vLLM shares the LM prefill across `SamplingParams(n=8)`:
   measured **2.370** FLOP-eq, not the as-charged 8.0. The prefix-caching-OFF control lands at 8.244,
   which is exactly the as-charged convention — that is what validates the instrument.
   (`verifier_restructure_2026-08-16.json` Q1)
2. **The head was under-charged.** As-charged 1.0/pass × 3.8136 passes = 3.814; at its *measured*
   geometry (it ran at fullres while generation ran at cap320) it is **6.893**. Capturing it during
   generation takes it to **2.07e-05** — five orders of magnitude below everything else.
3. **The verifier was under-charged.** Convention A said 3.823; measured at its own 1,003,520 px
   geometry it is **7.465**. Prefix-sharing takes it to **2.186**.

So the brief's "~11×" was wrong in **both** directions at once. The honest starting point was 16.73×
per open question.

### The remaining cost is now one number

Re-derived here (matches BUILD 2 to 0.0):

```
verifier FLOP-eq(n_distinct) = 1.9993  +  0.04887 × n_distinct
                               ^^^^^^     ^^^^^^^
                               fixed      per candidate
```

**91.5% of the verifier's cost is now the one shared prefill, and it does not shrink with N.** After
prefix-sharing, sampling more candidates is nearly free on the verifier side; the entire remaining cost
is *one image+question prefill at 1,003,520 px* — **4× the generator's own 250,880 px**, because no
verifier script ever had a resolution flag. That single term is **43.9% of the whole open arm** (the
verifier as a whole is 48.0%).

**Dropping it to the generator's resolution does NOT work** and I confirmed this independently: at
250,880 px the verifier is judge **−0.010235** / EM **−0.010661** vs 1,003,520 px, negative on all three
cells in both currencies (`shared_prefix_verifier_2026-08-16.json` RESOLUTION; my recomputation:
prefix-arm selected judge 0.475480 @250,880 vs 0.485288 @1,003,520). It is also confounded with a
train/inference mismatch — the adapter's `train_config.json` records max_pixels 1,003,520. **501,760 px
is a TIE in both currencies** (36 picks changed) and is the untested free rung.

---

## 4. FOLDING IN THE SETTLED HYPERPARAMETERS

### T = 0.4 (`decoding_ladder_cold_2026-08-14.json`)

Confirmed from the artifact's own ladder. T=0.4 is the peak in **both** currencies
(`PEAK.argmax_judge` = `argmax_em` = T04), and the currency audit shows picked-slot length matched
(5.158 tokens at T=0.4 vs 5.141 at T=0.5, 5.133 at T=0.3), so it is not verbosity harvesting.

| arm | SELECTED judge | SELECTED EM | vs deployed T=0.7 N=8 (judge) | vs deployed (EM) | FLOPs |
|---|---:|---:|---|---|---:|
| T=0.7, N=8 (deployed) | 0.48045 | 0.46382 | — | — | 1.000 |
| **T=0.4, N=4** | 0.48205 | 0.47207 | +0.00159 [−0.0051, +0.0083] TIE | **+0.00825 [+0.0015, +0.0151] WIN** | **0.500** |
| **T=0.4, N=8** | 0.48984 | 0.47733 | **+0.00938 [+0.0034, +0.0154] WIN** | **+0.01350 [+0.0077, +0.0193] WIN** | 1.000 |

**T=0.4 at N=4 is parity-or-better in both currencies at half the generation cost.** No retraining is
needed (co-adaptation refuted at 10 seeds). This is the single best cost lever left.

⚠️ **NOT MEASURED: T=0.4 combined with the free head and the prefix-shared verifier.** The captured
layer-21 features and the prefix-shared scores exist only for the deployed T=0.7 pools. The T=0.4 row
above is the **incumbent LoRA alone**. Combining them is arithmetic on paper, not a measurement, and the
`+ T=0.4 at N=4` cost row is labelled PROJECTED for exactly this reason. The verifier cost there uses
distinct-answer count 2.398 measured on **T=0.7** pools — conservative, since T=0.4 pools are less
diverse.

### Coarser veto binning (`veto_binning_2026-08-15.json`)

Recorded, **out of scope for a 7B-only claim**: the certified veto decides whether to call the **32B**,
so it cannot appear in a claim whose baseline is always-7B. Retained for the paper's cascade section:
n_bins 2–3 (shipped was 5) raises the PMC veto rate 0.400 → 0.667 at Δ +0.0090 to +0.0105, and is
cheaper. Small bins made the Wilson intervals too wide to certify anything, which is why 4 of 5 MCQ
cells sat at veto rate 0.0000.

### The best combined 7B-side configuration

Two **non-verifier** results from BUILD 3 are free or nearly free and belong in the same system. They are
reported separately because **neither is a verifier result and neither may be folded into the verifier
claim.**

| cell | change | Δ (both currencies) | cost |
|---|---|---|---:|
| PATH_VQA_closed | drop MedEvalKit's `"Please output 'yes' or 'no'"` instruction | **+0.038370 judge / +0.038073 EM** vs matched in-session fullres control | **1.0** (and 0.60× the deployed prefill) |
| PMC_VQA | self-consistency, majority vote over 8 T=0.4 samples | **+0.0100 harness / +0.0132 letter-EM** | 2.370 |

The PathVQA fix is mechanistically explained, not just observed: the instruction induces a **+0.069
yes-bias** (predicted-yes 0.6095 vs a 0.5402 gold base rate); removing it takes bias to +0.005 and
specificity 0.7503 → 0.8655 for 0.021 of sensitivity. Judge and EM agree to 0.0003, so it is not a
grading artifact.

**Best combined 7B-side system, macro-8** (judge for open cells, harness for MCQ — the project's
convention):

| cell | value |
|---|---:|
| PMC_VQA (self-consistency) | 0.552656 |
| SLAKE_closed | 0.825359 |
| VQA_RAD_closed | 0.780876 |
| PATH_VQA_closed (prompt fix) | 0.879239 |
| MedXpertQA-MM | 0.261500 |
| SLAKE_open (verifier) | 0.772093 |
| VQA_RAD_open (verifier) | 0.500000 |
| PATH_VQA_open (verifier) | 0.392000 |
| **MACRO-8** | **0.620465** (**+0.023379** vs always-7B 0.597087) |
| **MACRO-8 FLOP-eq** | **2.505** = (3 × 4.556 open + 2.370 PMC + 4 × 1.0)/8 |

Under EM for the open cells the combined macro-8 is 0.603518 vs a 0.595715 baseline = **+0.007803**.
⚠️ The PMC self-consistency number is measured on a **6,000-item subset** of the 33,430-item cell
(greedy on that subset is 0.552167 letter-EM / 0.540667 harness vs the published full-cell 0.542656), so
extending it to the whole cell is an extrapolation, not a measurement.

---

## 5. Which cells the method does NOT apply to, and why

- **PMC_VQA, MedXpertQA-MM** — intrinsically multiple-choice; the question frequently cannot be answered
  without the options. Expected to stay outside the claim, and saying so is honest.
- **SLAKE_closed, VQA_RAD_closed, PATH_VQA_closed** — the pre-registered attempt to re-ask these as open
  text **REFUTED at 0 of 3 cells** (`closed_as_open_2026-08-16.json`). The mechanism is the finding: the
  reformat **never changed the candidate set**. Distinct answer strings across a whole cell were VQA-RAD
  **5** (over 2,008 slots), PathVQA **9** (over 26,896), SLAKE 54 — against 3,919 on the three open
  cells. On PathVQA the open prompt actually *narrows* the pool. The model answers yes/no whether or not
  the prompt says so: **the answer space is intrinsic to the question, and the prompt does not control
  candidate diversity.** The generalizable statement is that **candidate diversity, not candidate
  provenance, gates whether a verifier can do anything.**

That build also introduced the right metric for this: `sel_eff` is **not comparable across regimes**,
because a near-unanimous pool makes a *random* pick score ~0.93 of oracle. SKILL = (SELECTED −
random_pick_floor)/(oracle@8 − random_pick_floor) is **+0.3056** on the open cells the verifier was built
for, versus SLAKE +0.099, VQA-RAD −0.079, PathVQA −0.018 on the reformatted closed cells.

---

## 6. ⛔ THE REFUTED HEADLINE — do not ship head-only

`verifier_restructure_2026-08-16.json` THE_COST_TABLE contains a row labelled **"THE_PRIZE"**:
head-only, head captured free, vision shared via image embeds — **1.2027 FLOP-eq per open question,
1.0760 macro-8**, at macro-8 accuracy **0.606173**. That row is the most attractive number produced this
round and **it must not be used**, for three independent reasons:

1. **The accuracy is judge-only.** The artifact reports `vs_greedy7b_judge` and **no EM counterpart**. I
   computed it: head-only captured at cap320 is **judge +0.045203 [+0.029424, +0.060554] WIN** but
   **EM −0.006823 [−0.022601, +0.008955]** with **all three cells negative** (SLAKE −0.00310, VQA-RAD
   −0.03500, PathVQA −0.00467). Its EM `selected` is 0.448188, **below the always-7B EM baseline of
   0.455011.** This is precisely the paraphrase-drift exposure the project already documented; the
   single-currency read would have shipped a method that is worse than greedy under exact match.
2. **The 0.7956 that motivated it was a single-seed fit.** The frozen 8-seed ensemble head alone is
   0.801090 (per-seed range 0.788147–0.800409, sd 0.004456).
3. **The cost row's accuracy was never re-measured through the path it prices.** The image_embeds fix
   casts bf16 → fp16 for vLLM's multimodal serialisation; the artifact states the accuracy is
   "asserted-unchanged, not re-measured."

**The fusion is the only selector that survives dual currency**, and that is why making its head half
free is the right lever rather than deleting its LoRA half.

---

## 7. Secondary — relation to the 32B (context, not the claim)

Clearly subordinate. The system above does **not** target always-32B-direct and does not reach it.

| arm | macro-8 | vs always-32B-direct | compute |
|---|---:|---|---|
| always-7B | 0.5971 | −0.0596 | 1.0 FLOP-eq |
| **7B + cheap verifier (this round)** | **0.6144** | **−0.0423** | 2.33 FLOP-eq |
| **best combined 7B-side** | **0.6205** | **−0.0362** | 2.50 FLOP-eq |
| always-32B-reasoning (unmatched) | 0.5974 | −0.0593 | — |
| always-32B-reasoning (prompt-matched) | 0.6250 | −0.0317 | — |
| **always-32B-direct** | **0.6567** | **THE BAR** | — |

The combined 7B-side system lands **0.0045 below a prompt-matched reasoning 32B** and **0.0362 below
always-32B-direct**. Converting to the 32B's unit with the project's own 7B:32B ratios
(`verifier_restructure_2026-08-16.json`: R32_as_charged 4.57, R32_derived 3.816), 2.505 FLOP-eq is
**0.548× (as-charged) to 0.656× (honestly re-costed)** of one always-32B-direct pass. ⚠️ This crosses
costing conventions (the FLOP-eq unit here is a cap320 open-text forward; the 0.219× in CLAUDE.md §0
comes from the cascade accounting) and is an orientation figure, not a headline. The 32B numbers are
quoted from CLAUDE.md §0 / `cascade_selector_rerun_2026-08-05.json` and were **not** re-measured here.

---

## 8. Controls that back the table

- **Frozen null test** 3.5967e-07; identity `selected = oracle@8 × sel_eff` ≤ 1.11e-16 on all arms.
- **Permutation null**: frozen head fed row-permuted features → 0.45471 ± 0.00382 (10 seeds) vs the real
  0.50576 = **13.35 σ**. Random-pick floor 0.42335 (sel_eff 0.67626).
- **In-session matched controls** for every fresh-generation comparison (BUILD 2's deployed arm, BUILD
  3's `closedD_g_full`), satisfying the ±0.008 open-text reproducibility caveat. The central table
  itself re-scores **stored** pools, so it incurs no ±0.008 exposure.
- **Both currencies on identical picks** for every open-text endpoint, via `free_head_lib.endpoint`,
  which raises if the pool's surface answers disagree with the dump's.
- **Guardrail per cell**, reported: judge-clean on 3/3; **EM negative on VQA_RAD_open (−0.025)**, inside
  its CI but a real per-cell negative that must be printed.
- **Arm multiplicity**, stated: several near-equivalent feature/verifier sources exist and one of them
  (fusion on captured cap320 features with the *deployed* LoRA) is CI-clean on 3/3 judge cells. The
  deployable stack scored here is CI-clean on 2/3. **The 3/3 number is not claimed** — picking the arm
  that happens to clear the CI is exactly the selection effect this project has already been burned by
  (a per-cell pick-the-best rule earned +0.0109 macro from shuffled labels).
- **HF transformers** for everything adapter-scored; no visual LoRA was scored under vLLM.

---

## 9. What remains, ranked

1. **Re-run the free head + prefix-shared verifier on the T=0.4 pools.** The single highest-value
   missing measurement. It is the difference between a projected 2.043 macro-8 FLOP-eq and a measured
   one, and T=0.4/N=4 is the biggest remaining cost lever (halves generation at parity).
2. **Verifier prefill at 501,760 px, combined with prefix sharing.** The prefix term is 43.9% of the
   whole open arm and 91.5% of the verifier. 501,760 px is already a measured TIE in both currencies (36
   picks changed); combining the two is **untested**. Scaling by the measured full-forward ratio
   (1.6953/1.8793 = 0.902, `verifier_restructure_2026-08-16.json` `ver_flopeq_by_max_pixels`) suggests a
   prefix-shared verifier near 1.97 and an open arm near 4.34 FLOP-eq — **an extrapolation, not a
   measurement**; the geometry rows for that rung were never written. Cheapest remaining win.
3. **Capture layer-21 states from vLLM.** "Free" is proven for the HF `generate()` path only; vLLM does
   not expose per-step hidden states through its standard API. Until this exists, a deployed vLLM system
   pays the head's 6.893 FLOP-eq. **NOT MEASURED.**
4. **Re-score the pool through the image_embeds path.** The vision fix (2.370 → 1.203) is measured on
   cost and *asserted* on accuracy, and it casts bf16 → fp16. One re-scoring run closes it.
5. **Refit the head on cap320 features.** It was trained at fullres, so a free capture necessarily feeds
   it out-of-distribution features. Costless at the fusion level today, but it is the obvious repair and
   would likely recover the ~0.0017 the deployable stack gives up under the judge.
6. **Answer-letter-bias audit of PMC-VQA `test_2.csv`** before the self-consistency result is used, and
   extension of that result from the 6,000-item subset to the full 33,430-item cell.
7. **Widen the claim, or accept 3 cells.** Candidate diversity — not prompt format — is the gate. The
   closed cells are out; the honest options are more open-ended benchmarks or a generator that produces
   diverse candidates on binary questions.

---

## 10. The honest bottom line for the paper

The **cost** story is strong, mechanistic and fully measured: two structural defects (a recomputed
hidden-state pass, an N-times-repeated image prefill) were removed for a **3.67× reduction in the open
arm at a TIE in both currencies**, and a third (over-charged generation) was a costing error worth 3.38×.
Cost is now dominated by a single, clearly identified, still-reducible term.

The **accuracy** story is thinner than the judge suggests. The defensible claim is **+0.0171 pooled EM
[+0.0030, +0.0311] over 2,345 open questions, CI-clean on 1 of 8 cells**, or **+0.0563 judge**
[+0.0422, +0.0704] **CI-clean on 2 of 8**. The macro-8 gain under EM is **+0.0014**. A paper built on
"a small verifier improves a 7B" should lead with the cost mechanism and report the accuracy in both
currencies, or it will not survive review.

---

**Artifacts:** `central_table_2026-08-16.json` (this doc's re-verification, written by
`src/cascade_methods/central_table_verify.py`) · `free_head_2026-08-16.json` ·
`shared_prefix_verifier_2026-08-16.json` · `closed_as_open_2026-08-16.json` ·
`verifier_restructure_2026-08-16.json` · `decoding_ladder_cold_2026-08-14.json` ·
`coadapt_verifier_T04_2026-08-14.json` · `veto_binning_2026-08-15.json` ·
`sevenb_only_frontier_2026-08-12.json` · `cost_decomposition_2026-08-12.json`.
