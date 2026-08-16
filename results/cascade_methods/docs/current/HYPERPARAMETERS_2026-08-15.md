# Four untuned hyper-parameters — verification, combination, and the headline

**Date:** 2026-08-15 (verification pass 2026-08-16) · **Artifact:**
`results/cascade_methods/artifacts/hyperparameters_combined_2026-08-15.json` ·
**Code:** `src/cascade_methods/hyperparameters_verify.py`

---

## VERDICT

**Essentially nothing here moves the headline.** Three of the four knobs are worth exactly or
approximately **zero accuracy**; the fourth — the certified veto's binning — is real and survives its own
permutation null at 6.3 σ, and is worth **+0.0012 macro-8**. Combined with the free T=0.4 temperature the
best accuracy-max point is **0.6589 at 1.718×**, i.e. **+0.0022 [−0.0016, +0.0058] vs always-32B-direct —
still a TIE**, 0.76× the +0.0029 a significant macro delta needs. The one CI-clean arm in the round
(0.65895 at 0.915×) wins by **switching the open-text machinery off**, and its edge is
**answer-prior-sensitive**: it ranges from +0.0001 to +0.0030 depending on an arbitrary stratum threshold.

Two things this pass adds that the four rounds did not have:

1. **T=0.4 is not free on the shipped arm.** The temperature ladder measured it on the SELECTED endpoint
   (before escalation) and knob 4 measured it on the plain-Weitzman arm. On the **shipped accuracy-max
   open policy** — best-of-8 + the F10-L2D rejector, which escalates 55–86 % of items to the 32B — the
   open-3 macro moves **+0.00044 [−0.00354, +0.00412] judge, a TIE**. Only **2.4 %** of the SELECTED gain
   survives. Under exact match it survives (+0.00600 [+0.00189, +0.01026]); **the two currencies disagree
   about the verdict, and the published macro is denominated in the judge.**
2. **The veto round's SLAKE-closed counter-control is not robust.** Its answer-balanced delta swings from
   **−0.00293** (all 48 gold-answer strata) to **+0.02041** (the round's ≥30-item cut). The claim "item-level
   competence, not an answer prior" should be softened to "the control does not settle it".

---

## 1. Null tests first

Max abs deviation over six checks: **4.57e-05**, and that is the stored artifact's own 4-dp rounding.

| check | source | result |
|---|---|---|
| N1 frozen metric | `src/training_methods/genframe_data.null_test()` | max abs dev **3.5967e-07**, PASS |
| N2 **exact identity** `selected = oracle@8 × sel_eff` | same | 0.6260127931769722 × 0.7752043596730245 = 0.4852878464818763 vs selected 0.48528784648187634 → **residual 5.55e-17**. The **forbidden additive form** `greedy + sel_eff·(oracle−greedy)` = 0.586326, **over-predicts by +0.101038**, and is never used |
| N3 published PMC veto, re-implemented from scratch | vs `pmc_label_noise_audit_2026-07-29.json` | acc 0.5613 / delta 0.0095 / rate 0.4002 — **max abs dev 0.0** |
| N4 macro reconstruction | `_selector_rerun_parts/vec_disjoint.npz` | direct **0.656672**, accuracy-max **0.657505**, compute-lean **0.644277** — **dev 0.0** |
| N5 shipped open policy re-implemented (best-of-8 + F10-L2D) | `beat32b_more.open_features` | 0.81705 / 0.59000 / 0.38467 vs published 0.8171 / 0.5900 / 0.3847 — dev **4.57e-05** |
| N6 item alignment, stored dumps vs in-session pool build | 32B-direct judge vectors | **dev 0.0** |

Numerics pinned: `OMP/OPENBLAS/MKL_NUM_THREADS=1` (8 for the F10 rebuild), `PYTHONHASHSEED=0`, CPU only
(no torch in the verification path, so no TF32 exposure), frozen canonical row order, **argmax over raw
scores with a strict `>` (first-index argmax), never `rank_avg`**, paired item bootstrap nboot=10000 seed
20260815.

---

## 2. The frontier

MACRO, equal weight per cell, 8 cells at 1/8, Variant B, CLEAN disjoint verifier. Cost = as-charged
FLOP-eq from `_selector_rerun_parts/macro_disjoint.json`. All rows from
`hyperparameters_combined_2026-08-15.json` § `S5_THE_COMBINED_OPERATING_POINT.table`.

| arm | macro | vs direct 0.6567 | vs shipped a-max 0.6575 | vs cost pt 0.6569 | FLOP-eq | ×direct |
|---|---:|---|---|---|---:|---:|
| always-32B-direct | 0.656672 | — THE BAR | −0.0008 [−0.0037,+0.0022] | −0.0002 [−0.0023,+0.0019] | 4.570 | 1.000 |
| **shipped accuracy-max** | 0.657505 | +0.0008 [−0.0021,+0.0037] TIE | — | +0.0006 [−0.0030,+0.0042] | 7.951 | 1.740 |
| **shipped best cost point** | 0.656857 | +0.0002 [−0.0019,+0.0023] TIE | −0.0006 [−0.0042,+0.0031] | — | 3.952 | 0.865 |
| **A** = + tuned veto (knob 2) | 0.658702 | +0.0020 [−0.0011,+0.0051] TIE | **+0.0012 [+0.0003,+0.0023] WIN** | +0.0018 [−0.0016,+0.0053] | 7.853 | 1.718 |
| **B** = + T=0.4 (knob 4's free lever) | 0.657670 | +0.0010 [−0.0025,+0.0045] TIE | +0.0002 [−0.0013,+0.0016] n.s. | +0.0008 [−0.0033,+0.0049] | 7.951 | 1.740 |
| **C = COMBINED (accuracy-max)** | **0.658867** | **+0.0022 [−0.0016,+0.0058] TIE** | +0.0014 [−0.0004,+0.0031] n.s. | +0.0020 [−0.0021,+0.0058] | 7.853 | **1.718** |
| **D = COMBINED on the cost point** | **0.658951** | **+0.0023 [+0.0012,+0.0034] WIN\*** | +0.0014 [−0.0015,+0.0048] TIE | **+0.0021 [+0.0004,+0.0038] WIN\*** | 4.183 | **0.915** |

\* **read § 4 before quoting D.**

Per-cell for the two combined arms (accuracy · always-32B-direct in brackets):

| cell | n | C (accuracy-max) | D (cost point) | direct |
|---|---:|---:|---:|---:|
| PMC_VQA | 33,430 | 0.56229 | 0.56229 | 0.5518 |
| SLAKE_closed | 836 | 0.86746 | 0.86746 | 0.8589 |
| VQA_RAD_closed | 251 | 0.85259 | 0.85259 | 0.8526 |
| PATH_VQA_closed | 3,362 | 0.88905 | 0.88816 | 0.8891 |
| MedXpertQA-MM | 2,000 | 0.30650 | 0.30650 | 0.3065 |
| SLAKE_open | 645 | 0.82171 | 0.81860 | 0.8186 |
| VQA_RAD_open | 200 | 0.58333 | 0.60000 | 0.6000 |
| PATH_VQA_open | 1,500 | 0.38800 | 0.37600 | 0.3760 |

---

## 3. Knob by knob, and the permutation nulls

**The mandatory test.** This project has measured a per-cell "pick the best" rule earning **+0.0109 macro
from shuffled labels alone** against +0.0090 on real data (p = 0.67). Every claimed gain below is priced
against its own null.

### Knob 1 — refit τ / λ / the open escalation threshold against MACRO (`hole17_macro_refit_2026-08-15.json`)

**NO GAIN — the observed effect IS the null mean.** Nested permutation null, 200 replicates,
macro-iso-cost anchor: real **+0.0009162** vs null **mean +0.0007302, sd 0.0009434, p95 +0.0023811**,
**p(null ≥ real) = 0.375**; best-of-4-anchors p = 0.425. Fold-seed stability over 10 seeds: mean +0.00013,
sd 0.00233, range [−0.00325, +0.00348] — **the sign is not stable**. The measured **ceiling of the entire
threshold family is macro 0.6471** (at μ=0, pure accuracy maximisation), **below the 0.656672 bar**, so no
re-scalarisation can reach it. On the shipped accuracy-max arm the accuracy is **exactly 0.0000 by
construction**. Its cost knock-on (7.951 → 7.660, 1.740× → 1.676×) is **not taken into the combined
point**, because it lowers the Pandora `meanN` while the accuracy vector still comes from a fixed
best-of-8 pick — it spends the accounting inconsistency the same round flags as its own side finding
(`method_final_mmmu_corrected.py:160`). The round's structural correction stands: the hole-17 premise is
wrong, the thresholds were never fit on a pooled objective.

### Knob 2 — the certified veto's `n_bins` × `alpha_z` (`veto_binning_2026-08-15.json`)

**REAL, INDEPENDENTLY REPLICATED, AND OUTSIDE ITS NULL.** A fresh nested-CV implementation (own Wilson
bound, own binning, outer 5 × inner 5, 10 fold seeds) reproduces the shipped arm at **0.657514** (round:
0.65751) and rule R4 at **0.658702** (round: 0.65845), a tuning gain of **+0.001188** against the round's
+0.00094. The +0.00025 difference is the MedXpert guardrail leak: the round's folds admitted MedXpert on
6 % of decisions (−0.00095 on that cell); mine admitted it on **0 of 50** — fold-assignment noise, not a
property of the rule. Mechanism confirmed and **opposite to the round's stated prior: coarser, not
looser.** Selected settings cluster at `n_bins` 2–4, `alpha_z ∈ [0, 0.5]`; the productive move switches
**SLAKE-closed on for the first time** (veto rate 0.0000 → 0.298).

**Independent permutation null** (`_hpv_veto_null_indep.jsonl`; (ok7, ok32) permuted jointly within each
cell so both marginals and their correlation survive and only the confidence→outcome link dies; the whole
nested-CV pipeline re-run; **200 replicates, 0 errors**, 2 fold seeds each, fresh RNG stream):

| statistic | null mean | null sd | null p95 | null max | observed | z | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| R4 (guardrail-gated global) | −0.0001996 | 0.0002181 | +0.0000541 | +0.0005981 | **+0.0023797** | **11.83** | 0.00498 (floor) |
| **R4 − R0 = the tuning gain** | −0.0001990 | 0.0002176 | +0.0000541 | +0.0005981 | **+0.0011757** | **6.32** | 0.00498 (floor) |
| R0 (fixed setting, no selection) | −0.0000005 | 0.0000054 | 0 | 0 | +0.0012040 | — | sanity anchor |

The round's own null agrees (R4 observed +0.00206, null −0.00018 ± 0.00020, z = 11.27). **Unlike the
project's +0.0109 precedent these nulls are NEGATIVE** — vetoing at random replaces a 32B answer with a
worse 7B one, so noise-driven certification *loses*. The guardrail is what keeps the null tight: the
round measures the unguarded rule's null sd at 0.00057 against the guarded rule's 0.00020.

### Knob 3 — the verifier's scoring `max_pixels` (`verifier_hparams_2026-08-15.json`)

**COST-ONLY; no accuracy claim is licensed.** The round's own leakage control is decisive: "pick the best
rung" earns **null mean +0.004319 in sample (p95 +0.010899) from shuffled labels alone**, while the real
in-sample best rung gains **exactly 0.000000** → **p = 1.0**; nested CV is *worse* than the control
(global −0.002044, p = 0.6385; per-cell −0.005450, p = 0.845). The pre-registered prize (scoring at the
generator's cap320) is **refuted**: −0.0177 judge / −0.0142 EM, guardrail-dirty in both currencies.
`max_pixels 501,760` is a true tie (d_sel_eff judge **exactly 0.000000**) worth **−6.39 % open-arm
FLOP-eq, −0.52 GiB, −23 % batch-1 verifier latency**, and **1.0003× — nothing — at the macro**. Under the
as-charged currency this frontier uses it is **exactly 0**, because `pandora_controller.py:50-52` charges
one verifier forward at 1.0 regardless of resolution.

### Knob 4 — the Weitzman λ (`weitzman_T04_2026-08-15.json`)

**REFITTING IS A LOSS.** All five of the round's null tests reproduce (frozen metric 3.60e-07, identity
5.55e-17, ζ closed form 3.44e-15, policy bit-exact 0.0, shipped open cells 4.81e-05). Read from the ARMS
block: A_deployed_T07r open-3 judge **0.579712**, B_stale_on_T04 **0.588334** (+0.008622), C_refit_T04
**0.584719** — the refit costs **−0.003615 [−0.00553, −0.00172] judge / −0.00399 EM**, permutation
p = 0.005. Both currencies agree. **Not taken.** The round's structural finding stands and is the real
content: **0 of 1,247,602 escalated decisions had 0 < N < 8**, so an escalated open item costs
8×2.0 + 4.57 = **20.57 FLOP-eq** against the bar's 4.57, and one λ cannot trade "draw fewer" against
"escalate more". Its S1 warning also stands: **shuffled labels reach the bar in 90.5 % / 100 % of
replicates** because a flat isotonic map collapses the controller into always-32B-direct, so "the
controller reaches parity at some compute" is **not a claim**.

### Free lever — T = 0.4 (`decoding_ladder_cold_2026-08-14.json`)

Pre-registered before any rung existed, matched in-session control, CI-clean in **both** currencies
(+0.009382 judge / +0.013504 EM on SELECTED), declared plateau {T03r, T04}, co-adaptation refuted at 10
seeds so no retraining is needed. **No selection is priced here.** What this pass adds is § 4.1.

---

## 4. What the combination actually buys

### 4.1 T = 0.4 is absorbed by the shipped arm's own escalation policy

Measured here for the first time: the shipped accuracy-max open cells rebuilt on the in-session T=0.4 and
T=0.7 pools with the identical F10-L2D code (3 generation seeds, per-item vectors seed-averaged, rejector
refit per fold on each pool's own features, paired item bootstrap).

| cell | SELECTED Δ (judge) | escalation T07r → T04 | after F10 Δ (judge) | retained |
|---|---:|---:|---:|---:|
| SLAKE_open | +0.00827 | 0.548 → 0.555 | +0.00465 | 56 % |
| VQA_RAD_open | +0.04000 | 0.860 → 0.785 | **−0.00667** | — |
| PATH_VQA_open | +0.00578 | 0.498 → 0.377 | +0.00333 | 58 % |
| **open-3 macro** | **+0.01802** | — | **+0.00044 [−0.00354,+0.00412] TIE** | **2.4 %** |

The SELECTED column reproduces `decoding_ladder_cold_2026-08-14.json` exactly. Under **normalised exact
match** the same contrast is **+0.00600 [+0.00189, +0.01026] — a WIN**; the currencies disagree about the
verdict and the published macro is the judge. **Mechanism:** escalated items take the 32B's answer, which
is identical at both temperatures, so the temperature can only act on the retained fraction — and F10
preferentially retains the items the 7B already gets right, exactly the ones the temperature cannot fix.

⚠️ **F10's escalation rate is not identified.** On PATH_VQA_open it spans **0.326 – 0.762 across three
generation seeds at the same temperature** (T07r s0/s1/s2 = 0.406 / 0.326 / 0.762). No open-half cost
claim is made from its movement, and the open cost is held at the shipped values (knob 4 measures `meanN`
essentially unchanged under the stale policy at T=0.4: 5.919 → 5.934).

### 4.2 The combination is exactly additive — and that is the wrong place to look

`veto alone +0.001197` + `T04 alone +0.000165` = `+0.001362` = `combined +0.001362`, **residual 0.0**.
Knob 2 moves only multiple-choice cells and T=0.4 moves only open cells; the macro is an equal-weight
average over **disjoint** cells, so the two are additive **by construction**, and knobs 1 and 3 are exactly
0.0000 on accuracy so there is nothing for them to be sub-additive with. **The sub-additivity is inside
the open half:** T=0.4 is worth +0.0180 on the open-3 SELECTED macro and +0.00044 after the shipped arm's
own escalation. **The method's escalation policy is what eats the generator improvement**, not another
knob.

### 4.3 Arm D — the one CI-clean win, and why it should not be shipped

D = the pre-specified best cost point with the R4 tuned veto substituted on the two cells R4's own inner
guardrail admits (PMC, SLAKE-closed). It is **0.658951 at 0.915×**, **+0.0023 [+0.0012, +0.0034] vs
always-32B-direct**, guardrail-clean on all 8 cells against both always-7B and always-32B-direct, and it
**dominates the shipped accuracy-max arm on both axes** (0.65895 vs 0.65750 at 0.915× vs 1.740×).

Four reasons to hold it:

1. **It has the shape the project already distrusts.** 5 of 8 cells are byte-identical to the bar; it
   wins by running always-32B-direct on all three open cells, i.e. **by switching the open-text machinery
   off** — the same shape as `armcombine_mcqonly_2026-08-11.json`. The narrow CI (±0.0011 vs ±0.0034 for
   the accuracy-max arms) is a *consequence* of those zero-variance cells, not of a stronger effect.
2. **It is answer-prior-sensitive.** Raw +0.002279; with letter-balanced PMC and answer-balanced
   SLAKE-closed it is **+0.000113** (all 48 strata) or **+0.003030** (the round's ≥30-item cut). **The
   verdict flips on an arbitrary threshold.**
3. **It is post-hoc.** The cost point's cell assignment is pre-specified (`cost_floor_2026-08-10`
   cross-fit) and the veto setting comes from R4's nested CV under its own null — but the *decision to
   combine them* was taken after seeing both results.
4. **It does not dominate the best cost point**, which is 5.8 % cheaper at a statistically
   indistinguishable accuracy.

Better than `armcombine` in one respect: **leave-one-cell-out is robust.** Dropping any single cell leaves
+0.0011 to +0.0027 (vs armcombine's exact 0.000 on dropping PMC). Two cells carry it, not one.

---

## 5. Guardrails and the answer-prior controls

**Standing guardrail (never worse than always-7B on any cell):** 0 of 8 flagged for every arm in the table.

**Against always-32B-direct (the stricter bar):** arms B and C flag **VQA_RAD_open** (−0.01667 with the
T04 substitution vs −0.01000 shipped, n = 200, inside generation-seed noise). Arms A and D flag nothing.
The round's R4 flags **MedXpertQA-MM** at −0.00095 [−0.00185, −0.00010] from admitting the cell on ~6 % of
folds; this pass's independent replication admits it on **0 of 50** fold decisions and the cell sits at
exactly 0.0000. **The leak is fold-assignment noise — but if the rule is ever shipped it should carry an
explicit never-deploy-on-MedXpert clause.**

### PMC-VQA answer-letter control — independently recomputed, CONFIRMED

Gold marginal on `test_2.csv`: A 13.2 % / B 35.9 % / C 37.8 % / D 13.1 %.

| arm | raw Δ | letter-balanced Δ | gold-A Δ |
|---|---|---|---|
| R0 shipped | **+0.00962** [+0.00726,+0.01196] | **+0.00536** [+0.00286,+0.00786] | −0.01185 [−0.01809,−0.00558] |
| R4 tuned | **+0.01051** [+0.00770,+0.01332] | **+0.00472** [+0.00163,+0.00785] | **−0.02166** [−0.02941,−0.01402] |

Tuning **raises** the raw delta and **lowers** the letter-balanced one, and nearly **doubles** the gold-A
damage. **The PMC half of the veto tuning gain is answer-prior.** (Round values, from different fold
draws: +0.00955→+0.01025 raw, +0.00530→+0.00442 balanced, −0.0118→−0.02243 gold-A. Same conclusion.)

### SLAKE-closed counter-control — NOT ROBUST (the main correction in this pass)

The round reports SLAKE-closed's gain *growing* when answer-balanced (+0.00778 → +0.01792) and concludes
item-level competence. That depends on the arbitrary minimum-stratum-size cut:

| min stratum n | strata | item coverage | answer-balanced Δ | verdict |
|---:|---:|---:|---|---|
| 1 | 48 | 100 % | **−0.00293** [−0.00871,+0.00207] | TIE (point negative) |
| 5 | 19 | 93.1 % | −0.00213 [−0.01435,+0.00762] | TIE |
| 10 | 13 | 88.2 % | +0.00304 [−0.01354,+0.01469] | TIE |
| 20 | 7 | 79.1 % | +0.01749 [+0.00677,+0.02943] | WIN |
| **30 (the round's choice)** | 6 | 76.6 % | **+0.02041** [+0.00808,+0.03433] | WIN |

Semantic decomposition: affirmative answers **+0.02688** (n = 346), negative **−0.00219** (n = 365),
open-vocabulary organ/modality **−0.01040** (n = 125); AFF/NEG-balanced +0.01234 [+0.00408, +0.02187].
**The gain lives entirely on affirmative answers — the shape an answer prior has.** The sentence "so it is
item-level competence, not an answer prior" should be softened to **"the control does not settle it"**.

---

## 6. Translation to the headline

The published headline is **+0.0615 [+0.0514, +0.0715] vs a reasoning 32B, a TIE vs always-32B-direct, at
1.74×**. After this round:

- **vs always-32B-reasoning: unchanged.** No knob touches the reasoning baseline.
- **vs always-32B-direct: still a TIE.** Best accuracy-max point +0.0022 [−0.0016, +0.0058] against a
  significance bar of +0.0029. The bar needs a summed per-cell gain of ≈ +0.0235; this round delivers
  +0.0182 summed (D) or +0.0109 summed (C), of which the answer-prior-corrected part is +0.0009 to +0.0242.
- **vs the shipped accuracy-max arm: a small, real, cheap improvement.** C is +0.0014 [−0.0004, +0.0031]
  at 1.718× vs 1.740×; A alone is +0.0012 [+0.0003, +0.0023] at the same compute reduction.
- **vs the best cost point (0.6569 @ 0.865×): no Pareto improvement.** D is +0.0021 accuracy but 5.8 %
  more expensive; C is +0.0020 at nearly **twice** the compute.

**Nothing here changes a published verdict.**

---

## 7. What is left, ranked

1. **The PMC-VQA `test_2.csv` answer-letter-bias audit.** Already owed; now load-bearing twice over. CPU only. **BLOCKING.**
2. **A canonical answer-balanced currency for open-vocabulary closed cells.** SLAKE-closed's verdict is
   currently a choice, not a measurement. CPU only. **BLOCKING.**
3. **Pre-register arm D and re-run it on a held-out draw.** It is the only CI-clean point and was
   assembled post hoc.
4. **Repair the accuracy/cost inconsistency at `method_final_mmmu_corrected.py:160` (repair (ii)).** Every
   cost claim on the accuracy-max arm — including knob 1's 1.740× → 1.676× — currently spends it. Costs
   −0.0001 judge / −0.0009 EM macro and preserves the published 1.740×.
5. **Pre-register `n_bins=12, alpha_z ∈ {1.96, 2.326}`** — the only 2 of 135 settings that beat the
   shipped one *letter-balanced*, and they move the **opposite** way from what the sweep selects (tighter
   bound, lower veto rate 0.333 vs 0.400). Found in a currency chosen after the fact; pre-registration
   candidate only.
6. **Pin F10's escalation rate.** 0.326–0.762 across generation seeds at one temperature.
7. **Retire the Weitzman controller for a cross-fit fixed N=1 + confidence gate** (knob 4's addendum: more
   accurate in both currencies, 2.7× cheaper, 3.3× faster sequentially) — and re-examine the
   "11.74 vs 16.0 FLOP-eq (−27 %)" survivor claim at the same time, since it depends on that machinery.
8. **Ship `max_pixels 501,760` for the verifier as cost hygiene.** −6.39 % open-arm FLOP-eq at d_sel_eff
   judge exactly 0.000000. No accuracy claim attaches.
9. **Do not sweep τ, λ or the open escalation threshold again.** Knob 1 measured that family's ceiling at
   **0.6471**, below the bar, at any exchange rate.

---

## 8. Caveats of record

- **Nothing ran end to end.** Every operating point is a CPU re-costing of saved per-sample dumps with
  measured batch-1 constants. VRAM is not measured in this pass.
- **Arms B and C mix stored open cells with an in-session-measured delta.** The delta is
  nuisance-cancelled (matched T07r control generated in the same session), but the absolute open cells are
  the stored ones, so those rows inherit the **±0.008 open-text reproducibility caveat** on the open
  third. Arms A and D touch only the MCQ half and are free of that mix.
- **Both currencies are reported for every open-half endpoint**, on identical picks. § 4.1 is the case
  where they disagree about the verdict.
- vqa_rad cells are n = 200/251; their movements are reported with the seed spread attached.
- Two A100s were shared with concurrent sessions throughout. **No GPU job ran in this pass** — it is CPU
  numpy over stored caches — and nothing was killed. `MedEvalKit/` untouched. `freeze_selector.py` **not**
  run. No visual LoRA scored under vLLM.
