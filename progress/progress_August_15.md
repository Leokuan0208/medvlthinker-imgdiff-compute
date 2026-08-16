# Progress — 2026-08-15

**The hyper-parameter round.** After the temperature ladder turned in a real but small win (+0.0094 judge
/ +0.0135 EM on open SELECTED), the question was "any other hyper-parameters we can tune?". Four
genuinely untuned knobs were identified by inspection and swept. This entry records the four rounds, the
adversarial re-derivation of each, the combined operating point they were never evaluated at together,
and the verdict.

Doc: `results/cascade_methods/docs/current/HYPERPARAMETERS_2026-08-15.md`.
Artifact: `results/cascade_methods/artifacts/hyperparameters_combined_2026-08-15.json`.
Code: `src/cascade_methods/hyperparameters_verify.py`.

---

## 1. Verdict

**Essentially nothing moves the headline.** Three knobs are worth zero accuracy; one is worth **+0.0012
macro-8** and survives its own permutation null at **6.3 σ**. The best combined accuracy-max point is
**0.6589 at 1.718×** — **+0.0022 [−0.0016, +0.0058] vs always-32B-direct, still a TIE**, 0.76× the
+0.0029 a significant macro delta needs. `vs always-32B-reasoning` is untouched.

---

## 2. The four knobs

| knob | what | accuracy | verdict |
|---|---|---|---|
| 1 | refit τ / λ / the open escalation threshold against MACRO instead of the pooled objective | **+0.0009, p = 0.375 against its own null** — the observed effect *is* the null mean; **exactly 0.0000** on the shipped accuracy-max arm | NO GAIN; hole 17 was mis-stated and is closed as a diagnosis |
| 2 | the certified veto's `n_bins` × `alpha_z` (135 settings × 5 MCQ cells) | **+0.0012 macro-8**, z = 6.3 vs its own permutation null | REAL — the only one |
| 3 | the verifier's scoring `max_pixels` (7 rungs) | **0.000000**; the pre-registered "score at the generator's cap320" prize is **REFUTED** (−0.0177 judge / −0.0142 EM) | COST HYGIENE ONLY (−6.4 % open-arm FLOPs, 1.0003× at the macro) |
| 4 | the Weitzman λ, refit on T=0.4 pools | **−0.0036 [−0.0055, −0.0017] judge**, p = 0.005 | A LOSS — do not refit; the temperature was the lever, not λ |

Knob 1's ceiling is the useful negative: **the whole τ/λ threshold family tops out at macro 0.6471** (at
μ = 0, pure accuracy maximisation) — **below the 0.6567 bar**. No re-scalarisation of those thresholds can
reach it, so that direction is closed, not merely unproductive.

---

## 3. Two things the adversarial pass found that the rounds did not have

### 3.1 T = 0.4 is NOT free on the shipped arm — its own escalation policy eats it

The temperature ladder measured T=0.4 on the **SELECTED** endpoint (the best-of-8 pick, *before*
escalation) and knob 4 measured it on the plain-Weitzman arm. Neither is the shipped accuracy-max open
policy, which is best-of-8 **plus the F10-L2D rejector**. Rebuilt there for the first time, with an
in-session T=0.7 control so the ±0.008 nuisance cancels:

| cell | SELECTED Δ judge | escalation | after F10 Δ judge | retained |
|---|---:|---:|---:|---:|
| SLAKE_open | +0.00827 | 0.548 → 0.555 | +0.00465 | 56 % |
| VQA_RAD_open | +0.04000 | 0.860 → 0.785 | **−0.00667** | — |
| PATH_VQA_open | +0.00578 | 0.498 → 0.377 | +0.00333 | 58 % |
| **open-3 macro** | **+0.01802** | | **+0.00044 [−0.00354,+0.00412] TIE** | **2.4 %** |

Under **normalised exact match** the same contrast is **+0.00600 [+0.00189, +0.01026], a WIN** — the two
currencies **disagree about the verdict**, and the published macro is denominated in the judge. Mechanism:
escalated items take the 32B's answer, which is identical at both temperatures, and F10 preferentially
retains the items the 7B already gets right — exactly the ones the temperature cannot fix.

⚠️ Side finding: **F10's escalation rate is not identified.** On PATH_VQA_open it spans **0.326 – 0.762
across three generation seeds at the SAME temperature**. No open-half cost claim is made from its movement.

### 3.2 The veto round's SLAKE-closed counter-control is not robust

The round argued that the newly switched-on cell is competence rather than an answer prior because its
answer-balanced delta *grows* (+0.00778 → +0.01792). That depends on an arbitrary minimum stratum size:

| min stratum n | strata | coverage | answer-balanced Δ |
|---:|---:|---:|---|
| 1 | 48 | 100 % | **−0.00293** [−0.00871, +0.00207] |
| 10 | 13 | 88 % | +0.00304 [−0.01354, +0.01469] |
| 20 | 7 | 79 % | +0.01749 [+0.00677, +0.02943] |
| **30 (the round's cut)** | 6 | 77 % | **+0.02041** [+0.00808, +0.03433] |

The gain lives entirely on **affirmative** answers (+0.02688, n = 346) while negative (−0.00219) and
open-vocabulary organ/modality strata (−0.01040) go the other way. **The control does not settle it.** The
PMC half of the same knob is confirmed answer-prior: tuning raises the raw delta (+0.00962 → +0.01051) and
**lowers** the letter-balanced one (+0.00536 → +0.00472) while nearly doubling the gold-A damage
(−0.01185 → −0.02166).

---

## 4. The combined frontier

| arm | macro | vs direct 0.6567 | FLOP-eq | ×direct |
|---|---:|---|---:|---:|
| always-32B-direct | 0.656672 | THE BAR | 4.570 | 1.000 |
| shipped accuracy-max | 0.657505 | +0.0008 [−0.0021,+0.0037] TIE | 7.951 | 1.740 |
| shipped best cost point | 0.656857 | +0.0002 [−0.0019,+0.0023] TIE | 3.952 | 0.865 |
| + tuned veto | 0.658702 | +0.0020 [−0.0011,+0.0051] TIE | 7.853 | 1.718 |
| + T=0.4 | 0.657670 | +0.0010 [−0.0025,+0.0045] TIE | 7.951 | 1.740 |
| **COMBINED (accuracy-max)** | **0.658867** | **+0.0022 [−0.0016,+0.0058] TIE** | 7.853 | 1.718 |
| **COMBINED on the cost point** | **0.658951** | **+0.0023 [+0.0012,+0.0034] WIN\*** | 4.183 | **0.915** |

**The combination is exactly additive** (residual 0.0): knob 2 moves only multiple-choice cells, T=0.4
moves only open cells, and the macro averages disjoint cells. Knobs 1 and 3 are exactly 0.0000 on
accuracy. **The sub-additivity is inside the open half** — +0.0180 on SELECTED becomes +0.00044 after the
method's own escalation.

\* The one CI-clean arm is **not shippable**: 5 of 8 cells are byte-identical to the bar (it wins by
switching the open-text machinery off, the `armcombine_mcqonly` shape), its edge swings from **+0.0001 to
+0.0030** under the answer-prior correction depending on that stratum threshold, and the decision to
combine those two pre-specified pieces was taken **after** seeing both results. It is better than
`armcombine` in one respect: leave-one-cell-out is robust (+0.0011 to +0.0027 rather than collapsing to
exactly 0.000 on dropping PMC).

---

## 5. Null tests and permutation nulls

Max abs deviation over six null tests: **4.57e-05** (the stored artifact's own 4-dp rounding). Frozen
metric 3.5967e-07. The exact identity was asserted **multiplicatively** — `selected = oracle@8 × sel_eff`,
residual **5.55e-17**; the forbidden additive form over-predicts by **+0.101038** and is never used. The
published PMC veto and the shipped open cells were both re-implemented from scratch and reproduce at
**0.0** and **4.57e-05**.

Independent permutation null for the one real gain, fresh RNG stream, 200 replicates, 0 errors
(`_hpv_veto_null_indep.jsonl`):

| statistic | null mean ± sd | null max | observed | z |
|---|---|---:|---:|---:|
| R4 (guardrail-gated global) | −0.000200 ± 0.000218 | +0.000598 | **+0.002380** | **11.8** |
| **R4 − R0 = the tuning gain** | −0.000199 ± 0.000218 | +0.000598 | **+0.001176** | **6.3** |
| R0 (fixed setting, no selection) | −0.0000005 ± 0.0000054 | 0 | +0.001204 | sanity anchor |

**These nulls are NEGATIVE** — vetoing at random replaces a 32B answer with a worse 7B one, so
noise-driven certification *loses*, unlike the project's +0.0109-from-shuffled-labels precedent. The
guardrail is what keeps the null tight (unguarded sd 0.00057 vs guarded 0.00020). Knob 1's null:
p = 0.375, **the observed gain is the null mean**. Knob 3's null: the selection procedure earns **strictly
less than chance** (real in-sample best-rung gain exactly 0.000000, p = 1.0). Knob 4's null: the
refit-vs-stale and the temperature contrasts both survive at p = 0.005, but shuffled labels **reach the
bar** in 90.5 % / 100 % of replicates, so "the controller reaches parity at some compute" is not a claim.

---

## 6. Guardrails

0 of 8 cells below always-7B for every arm. Against always-32B-direct: the T=0.4 arms flag
**VQA_RAD_open** (−0.01667 vs −0.01000 shipped, n = 200, inside generation-seed noise); the veto rounds'
R4 flags **MedXpertQA-MM** at −0.00095 [−0.00185, −0.00010] from admitting it on ~6 % of folds — this
pass's independent replication admits it on **0 of 50** fold decisions, so the leak is fold-assignment
noise, but the rule needs an explicit never-deploy-on-MedXpert clause if it is ever shipped.

---

## 7. Housekeeping

- New code: `src/cascade_methods/hyperparameters_verify.py` (independent re-derivation + the combined
  frontier). Round code from the four sweeps is already on disk (`hole17_*.py`, `veto_binning_*.py`,
  `verifier_hparams_*.py`, `weitzman_T04*.py`).
- New artifacts: `hyperparameters_combined_2026-08-15.json`, `_hpv_veto_null_indep.jsonl`.
  Log: `logs/hpv_veto_null_indep.log`.
- **No GPU job ran in the verification pass** — CPU numpy over stored caches. Both A100s were shared with
  concurrent sessions; nothing was killed. `MedEvalKit/` untouched. `freeze_selector.py` **not** run. No
  visual LoRA scored under vLLM.
- ⚠️ Standing: `ckpts/`, `feats_hidden/` and `logs/` have **zero tracked files**. A `git push` does not
  protect the T=0.4 pools in `ckpts/openvqa/decoding_sweep/` that § 3.1 is measured from.

---

## 8. What this means for the direction

The tuning surface is exhausted. **Knob 1 measured a ceiling** — the whole escalation-threshold family
tops out at 0.6471, below the bar — and knobs 2–4 between them move the macro by less than half the
significance bar. The two live corrections are both about **currencies, not tuning**: the answer-prior
audit PMC-VQA `test_2.csv` still owes, and the open half's judge-vs-exact-match disagreement now decides
whether T=0.4 helps the shipped arm at all. **The binding limit is no longer a hyper-parameter; it is that
the method's own escalation policy discards 97.6 % of a genuine generator-side improvement.**
