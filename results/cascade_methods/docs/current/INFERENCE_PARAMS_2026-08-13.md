# Inference parameters of the 7B — can they improve the samples? (round of 2026-08-13/14)

> **Verdict, first line. NO — not for accuracy. No 7B inference-parameter change produces an accuracy
> gain that survives inference at the level of the claim being made. The best setting in the whole grid
> is temperature 0.5 at +0.005686 SELECTED, and its interval is [−0.00256, +0.01407] once
> generation-seed variance is included — it spans zero. Family-wise corrected over the seven settings
> tested, [−0.00583, +0.01748]. The measured ceiling of the lever is +0.0042 macro-8 for an
> eval-visible oracle choice of setting and +0.0207 macro-8 for an unattainable per-question oracle
> over settings; and pooling every candidate the whole grid produces (29.97 distinct per item, 24× the
> deployed budget, oracle@8 +0.1633) makes the frozen verifier's accuracy go **down** by 0.00498. The
> lever is closed.**
>
> **But the round is not empty, because the objective is cost.** Under corrected, dedup-aware costing
> **temperature 0.3 holds accuracy (+0.005402 [−0.00498, +0.01578], TIE, guardrail-clean) at 0.7952×
> the open arm's FLOPs — a 20.5% compute cut for free.** Colder sampling emits fewer *distinct*
> candidates (2.182 vs 3.665), and the verifier — half the arm — is only ever run on distinct ones.
> That is a genuine minimum-cost-at-parity win and it is the round's deliverable.
>
> **And the round's most important negative is structural: resolution cannot move the MCQ half at all.**
> Verified independently from the image files and the code path — every one of the five MCQ cells
> already runs uncapped, so 62.5% of the macro weight is closed to this lever by impossibility, not by
> a null.
>
> Every number below names its artifact. Abstention appears nowhere.

---

## 1. What was verified, and what I changed

Three sweeps were handed to this session: a single-variable decoding sweep, a resolution sweep, and a
vision-diversity sweep. I re-derived the decoding sweep from the raw dumps without importing any of its
analysis code, ran one new GPU experiment to settle a grading question, and re-measured the resolution
sweep's load-bearing geometry claim from the images themselves.

**Reproduced exactly** — `_infparams_verify_recompute.json`:

| check | result |
|---|---|
| Null test 1, frozen metric vs `genframe_data.PUBLISHED` | **PASS**, max abs deviation **3.5967302447481586e-07** — bit-identical to the value the sweep reports |
| Identity `selected = oracle@8 × sel_eff` | **1.11e-16** max abs error over all 24 pools |
| Per-seed `oracle@8` / `sel_eff` / `selected`, all 8 settings × 3 seeds | **exact match** to the sweep, every cell |
| Every pooled delta point estimate | **exact match** (CIs differ only in bootstrap RNG: my seed 20260814 vs theirs 20260813) |
| Missing judge / verifier slots after my own join | **0 / 0** across all 24 pools |
| Resolution sweep N2: 10 published MCQ cells | reproduced to **< 5e-5** (`_resolution_parts/null_tests.json`) — this is what pins the published arms to `max_pixels` 12,845,056 |

The arithmetic of all three sweeps is sound. What follows are the four things I changed.

---

## 2. Change 1 — the item bootstrap is the wrong interval, and it flips the headline

Every delta in the decoding sweep is reported with a **paired item bootstrap**, which resamples the
2,345 questions and holds the generation seed **fixed**. But the claim is about a sampling
*distribution* ("T=0.5 is better than T=0.7"), and each generation seed is one draw from that
distribution. Seed-to-seed variance is part of the uncertainty in the claim and the item bootstrap does
not contain it. For the control, the seed sd of SELECTED is **0.003962** — comparable to the entire
effect being claimed.

`_infparams_seedaware.json`, nboot 10,000, seed 20260814:

| setting | d SELECTED | item-only CI (as reported) | **seed-aware CI** | Welch t (3 seeds) | FWER-7 CI |
|---|---:|---|---|---|---|
| **T=0.5** | **+0.005686** | [+0.00100, +0.01038] **WIN** | **[−0.00256, +0.01407] TIE** | t=2.19, df=3.1, n.s. | [−0.00583, +0.01748] |
| T=0.3 | +0.005402 | [−0.00213, +0.01294] TIE | [−0.00498, +0.01578] TIE | t=1.69, df=4.0, n.s. | [−0.00924, +0.02018] |
| min_p 0.10 | +0.003980 | [−0.00100, +0.00896] TIE | [−0.00512, +0.01322] TIE | t=1.15, df=3.9, n.s. | [−0.00924, +0.01734] |
| rp 1.10 | +0.004833 | [+0.00085, +0.00896] **WIN** | **[−0.00554, +0.01535] TIE** | t=1.04, df=3.1, n.s. | [−0.00999, +0.02033] |
| rp 1.05 | +0.001279 | [−0.00185, +0.00441] TIE | [−0.00697, +0.00981] TIE | t=0.41, n.s. | [−0.01056, +0.01407] |
| T=1.0 | −0.016489 | [−0.02203, −0.01095] LOSS | [−0.02857, −0.00526] **LOSS** | t=−3.17, marginal | [−0.03359, −0.00038] |
| T=1.3 | −0.045629 | [−0.05387, −0.03781] LOSS | [−0.05899, −0.03227] **LOSS** | t=−8.27, **sig** | [−0.06458, −0.02654] |

**Both CI-clean wins in the round dissolve. The losses survive.** That asymmetry is not a coincidence:
the T≥1.0 effects are several times the seed sd, the candidate wins are not.

Two further points against the winners. The winner was **not pre-registered** — the sweep's own
pre-registration predicted *"no single-variable decoding change beats the deployed T=0.7 on SELECTED"*
(`_decoding_sweep_prereg.json`), so T=0.5 was identified by reading the eval endpoint, and the
family-wise rate over seven comparisons is the operating one. And the sweep's per-seed deltas pair seed
*k* with seed *k*, which is arbitrary — the seeds are independent draws. All nine cross-seed pairings of
T=0.5 against the control range **−0.00085 to +0.01109**, 8 of 9 positive: a consistent direction, an
underpowered test.

**Honest statement of what T=0.5 is:** a positive point estimate with a consistent sign across seeds and
cells, guardrail-clean, that this design cannot resolve from zero. Three generation seeds is not enough
to certify a +0.006 effect against a 0.004 seed sd. It is not a fabricated result and it is not a
demonstrated one.

---

## 3. Change 2 — I suspected the grader, and the grader is clean

Candidate labels came from two **disjoint** sources: a preload cache of judge labels harvested from
earlier runs, and a fresh judge pass in that session. The share drawn from the fresh judge is a
**monotone function of the treatment** — 0.0798 at T=0.3, 0.1275 at T=0.5, 0.2029 at T=0.7, 0.3483 at
T=1.0, 0.5004 at T=1.3 (`_infparams_ceiling.json`). Hotter sampling emits more novel strings. If the two
label sources disagreed at all systematically, that gradient would manufacture a temperature effect out
of nothing, in exactly the observed direction. The two sets share **zero keys**, so their agreement
could not be measured from the files.

So I measured it. **4,047 preload-labelled slots, stratified across the three cells, re-judged with the
same harness the sweep used for its fresh labels** (`src/labeling/run_judge.py`, MedVLThinker-32B,
text-only, greedy, `max_tokens` 2, Yes/No logprob comparison, tp=2 —
`runners/run_rejudge_confound.sh`, `_infparams_rejudge.json`):

- **agreement 1.000 — 4,047 / 4,047, zero disagreements**, per cell 1.000 / 1.000 / 1.000
- cached positive rate 0.2962688411168767, re-judged positive rate 0.2962688411168767 — identical
- patching every re-judged label back and recomputing: **every headline delta unchanged to 6 dp**

**The confound is refuted by measurement.** The judge is deterministic, the cache is sound, and label
reuse across settings is legitimate — it actually *removes* judge noise as a between-arm confound. This
is a positive verification result and the sweep deserves credit for it.

One related caution against over-reading a test of my own: restricting to the 836 items whose every
candidate in both arms was cache-labelled gives d = **exactly 0.0**, but that subset has pool Jaccard
0.9428 against 0.6617 on the full pool (`_infparams_currency.json`) — it retains only about a sixth of
the treatment contrast. Given the 100% agreement above, the source-matched null is attenuation, not
evidence. Reported so it is not mistaken for a finding.

---

## 4. Change 3 — the round's cost model overcharges the deployed arm by 1.57×

`resolution_greedy_vs_arm.py:125` charges the deployed open arm as
`arm_deployed = 8 * f_gen_cap320 + 8 * f_ver` — **eight generator forwards and eight verifier
forwards**. The verifier does not run eight times. Candidates are **deduplicated by normalized answer
string** before scoring. Evidence, three independent ways:

- `feats_hidden/generator_eval_s0of2.meta.json` n=4472 + `generator_eval_s1of2.meta.json` n=4471 =
  **8,943 rows over 2,345 questions = 3.8136 distinct candidates per question**
- the frozen transfer dumps hold 18,760 slots but only **8,943 distinct normalized strings** (0.4767)
- the sweep's own verifier score cache is keyed `(ds, idx, ans)`, not by slot

`_infparams_cost_correction.json`:

| | as the round charged it | dedup-aware | |
|---|---:|---:|---|
| deployed open arm, FLOPs/question | 1.473725e+14 | **9.408480e+13** | overcharge **1.5664×** |
| verifier share of the arm | 69.10% | **51.60%** | still the largest single term |
| one greedy native decode vs the arm | 0.075902× → **13.17×** | 0.118891× → **8.41×** | |

**The direction of every conclusion is unchanged and the saving is still large.** But "13.17× less
open-half compute" should be quoted as **8.41×**, and the striking line "the pipeline spends more
compute verifying than generating" is *true at the deployed operating point* (51.6% > 48.4%) but by a
much smaller margin than 69/31.

---

## 5. Change 4 — the deduplication makes temperature a COST lever, which nobody reported

The correction has a consequence the round missed. The distinct-candidate count **is a function of the
swept parameter** — 2.182 at T=0.3, 3.665 at T=0.7, 5.476 at T=1.3 — and the verifier only pays for
distinct candidates. So temperature moves the arm's cost, not just its accuracy, and generation cost is
identical across the whole grid (same N, same `max_tokens`, same image cap; generated tokens are 1.2% of
compute).

### The frontier — `_infparams_frontier.json`

3-seed means, judge currency, dedup-aware whole-arm FLOPs relative to the deployed T=0.7.

| setting | oracle@8 | sel_eff | SELECTED | d vs control (seed-aware) | macro-8 | distinct | **arm FLOPs** | verif. share | Pareto |
|---|---:|---:|---:|---|---:|---:|---:|---:|:--:|
| **T=0.3** | 0.558778 | 0.870262 | 0.486283 | +0.005402 [−0.00498, +0.01578] TIE | +0.004740 | 2.182 | **0.7952×** | 37.9% | **★** |
| **T=0.5** | 0.601279 | 0.809220 | **0.486567** | +0.005686 [−0.00256, +0.01407] TIE | +0.004151 | 2.955 | 0.9019× | 45.2% | **★** |
| min_p 0.10 | 0.605970 | 0.800141 | 0.484861 | +0.003980 [−0.00512, +0.01322] TIE | +0.002299 | 2.980 | 0.9054× | 45.4% | |
| rp 1.10 | 0.634399 | 0.765629 | 0.485714 | +0.004833 [−0.00554, +0.01535] TIE | +0.001228 | 3.877 | 1.0292× | 52.0% | |
| rp 1.05 | 0.630419 | 0.764825 | 0.482161 | +0.001279 [−0.00697, +0.00981] TIE | +0.000862 | 3.759 | 1.0129× | 51.2% | |
| **T=0.7 (deployed)** | **0.628571** | 0.765038 | 0.480881 | — | 0.000000 | 3.665 | **1.0000×** | 50.6% | |
| T=1.0 | 0.626155 | 0.741657 | 0.464392 | −0.016489 [−0.02857, −0.00526] LOSS | −0.004946 | 4.642 | 1.1348× | 56.5% | |
| T=1.3 | 0.576972 | 0.754373 | 0.435252 | −0.045629 [−0.05899, −0.03227] LOSS | −0.013834 | 5.476 | 1.2500× | 60.5% | |

**Non-dominated: {T=0.3, T=0.5}.** The deployed T=0.7 is dominated by both on point estimates.

**Latency: NOT MEASURED** — no wall-clock instrumentation in this round for any decoding setting.
**VRAM: NOT MEASURED** for any decoding setting. Same model, same N, same `max_tokens`, same image cap
means no mechanism for it to move, but that is an argument, not a measurement.

**Read the frontier correctly.** Every accuracy difference except the T≥1.0 losses has a seed-aware CI
spanning zero, so the accuracy axis is *not separated* — dominance here is on point estimates. The axis
that actually distinguishes these points is **cost**, and cost is arithmetic on measured token geometry,
not a noisy estimate.

### The cost-at-parity result

**T=0.3 is a TIE on accuracy (+0.005402 [−0.00498, +0.01578]), guardrail-clean, at −20.5% open-arm
compute.** Guardrails, seed-aware, all non-negative: slake_open +0.001034 [−0.01809, +0.02067],
vqa_rad_open +0.033333 [−0.00333, +0.07167], pathvqa_open +0.003556 [−0.00911, +0.01711]. T=0.5 gives
−9.8%, min_p 0.10 −9.5%.

This is a real win under the project's stated objective and it should not be buried because the accuracy
delta is ~0 — that is precisely the shape of a minimum-cost-at-parity result. Two honest limits: it is
worth **−20.5% of the open arm only**, which is 3 of 8 cells and a minority of total pipeline compute;
and it rests on the deployment actually deduplicating before scoring, which the frozen artifacts do and a
naive implementation might not.

---

## 6. The ceiling of the lever — so it can be closed rather than revisited

`_infparams_ceiling.json`. Macro-8 uses equal weight per reporting cell, 1/8 each; the open arm is 3 of 8
cells, and a pooled open delta enters at 3/8 weight only if the open arm is fully deployed with no
escalation — an upper bound.

| ceiling | open-pool value | macro-8 equivalent |
|---|---:|---:|
| best single setting, chosen **by reading the eval endpoint** | 0.486567 (T=0.5) | **+0.004151** |
| per-question **oracle over all 8 settings** (unattainable by any router) | 0.536034 | **+0.020682** |
| **union pool**, all 8 settings × 3 seeds, oracle coverage | oracle 0.791898 (+0.163326) | +0.061247 *(coverage only)* |
| **union pool, what the frozen verifier actually converts** | selected 0.475906 | **−0.005617** |

The last row is the cleanest closure statement this project has produced on the candidate-distribution
question. Give the frozen scorer **29.97 distinct candidates per item** — 24× the deployed sampling
budget, every candidate the entire decoding grid can produce at any temperature, any min_p, any
repetition penalty — and coverage rises by a huge **+0.163326**, while **SELECTED accuracy falls by
0.004975** and sel_eff collapses from 0.765038 to 0.600969.

**More candidates is not the problem and more candidates is not the fix.** This is the third independent
confirmation this week (decoding grid, resolution, vision views) and the sharpest, because it is not a
null — it is a large, significant, correctly-signed coverage gain that converts to a *negative*.

The brief's warning that the +0.0091 iid free-coverage bound is distribution-specific is confirmed and
should be propagated: it moved to 0.650070 → 0.672574 across the resolution ladder
(`resolution_sweep_2026-08-13.json`) and 0.680477 → 0.713874 across vision views
(`vision_diversity_2026-08-13.json`). **+0.0091 is a cap320, T=0.7 number.** It must not be quoted as
universal.

---

## 7. Task item 4 — did resolution move greedy MCQ accuracy? No, and it cannot

This was flagged in advance as the one result that would outrank everything else, because it would move
62.5% of the macro weight with no selection involved. **It is impossible, and I verified that
independently rather than accepting the sweep's table.**

**The code path** (`_infparams_mcq_geometry.json`): `MedEvalKit/models/Qwen2_5_VL/Qwen2_5_VL_vllm.py:51`
reads `_MP = int(os.environ.get("CAP_MAX_PIXELS","0"))` and line 54 applies it only `if _MP`. Unset means
no cap is passed and the processor keeps its own default. The Lingshu-7B `preprocessor_config.json`
default is **`max_pixels` 12,845,056**. The only two runners that set `CAP_MAX_PIXELS` are
`run_resolution_mcq_ladder.sh` and `run_resolution_mcq_pathvqa.sh` — both written for this round. **No
runner behind a published MCQ cell sets it.**

**The geometry**, re-measured from the image files with the harness's own `smart_resize`:

| cell | n images | max resized pixels | above the 12,845,056 default |
|---|---:|---:|---:|
| SLAKE_closed | 1,061 | 1,048,576 | **0.000** |
| VQA_RAD_closed | 451 | 1,345,536 | **0.000** |
| PATH_VQA_closed | 3,000 | 657,272 | **0.000** |
| MedXpertQA | 2,858 | 9,107,712 | **0.000** |
| PMC_VQA (`test_2.csv`) | 33,430 | 11,684,175 | **0.000** |

My SLAKE, VQA-RAD and MedXpert maxima match the sweep's `geometry_mcq.json` exactly. My PMC-VQA maximum
is 11,684,175 against the sweep's 8,870,964 — a discrepancy worth recording (I resolve `figures/` before
`images/`, following `MedEvalKit/utils/PMC_VQA/PMC_VQA.py:48,60`), but immaterial: both are below the
cap.

**Conclusion: the published MCQ arms already see every pixel the model would ever receive. Raising
`max_pixels` on the MCQ half is a no-op — not an untested hypothesis but an impossibility.** Cutting it
is not free either: the sweep measures +0.007606 macro-8 lost over four of five cells, 2.6× the project's
0.0029 threshold (`resolution_sweep_2026-08-13.json`).

**The structural finding stands and should be propagated to the paper.** One reported macro-8 is
evaluated at **three different resolutions** — MCQ legs and the always-32B-direct bar at 12,845,056, the
open generator and its 32B comparator at 250,880, the open LoRA verifier at 1,003,520: a **51.2× spread,
with the scorer seeing 4× more of the image than the generator whose candidates it ranks.** Each
comparison is internally matched so the per-cell deltas stand, but the macro is not one operating point
and per-item cost must be charged per cell.

---

## 8. Controls and guardrails

- **Null test** re-run independently: max abs deviation **3.5967302447481586e-07**; identity
  `selected = oracle@8 × sel_eff` max abs error **1.11e-16** over all 24 pools.
- **±0.008 caveat honoured.** The control was regenerated in-session in the same serving config; no
  delta in this document is taken against a stored number from another config. The resolution round
  re-measured the caveat itself on two cells with nothing changed: SLAKE −0.004785, PathVQA −0.003867.
- **Grader**: the project's existing `src/labeling/run_judge.py` (MedVLThinker-32B, text-only) — the same
  one used to build every open-text cell. Verified deterministic at 4,047/4,047.
- **Verifier scoring**: HF transformers only, bf16 + flash_attention_2, batch 1 — never vLLM (the
  192-`visual.*`-module landmine). The sweep's own null test 2 re-scored 1,440 stored slots at max abs
  deviation **0.0** and argmax agreement **1.0**.
- **Both currencies reported.** In exact match (no judge, no cache, seed-aware), T=0.3 is
  **+0.012793 [+0.00299, +0.02260] SIG** and min_p 0.10 **+0.009666 [+0.00171, +0.01777] SIG**, while
  rp 1.10 is a **−0.007676 [−0.01478, −0.00057] LOSS** — the currency conflict the sweep identified on
  rp 1.10 is real and survives seed-aware inference. But EM is **not** the deployed metric; the frozen
  endpoint is the judge, and in the judge currency nothing wins.
- **Guardrails** for the two frontier settings: no cell negative, all CIs spanning zero (§5).
- **Seeds**: 3 generation seeds per setting, 24 complete pools, 450,240 candidate slots, 0 missing judge
  labels and 0 missing verifier scores on my own independent join.
- **Numerics**: `OMP_NUM_THREADS=8`; all analysis CPU-side off frozen artifacts; the frozen metric
  `src/training_methods/genframe_data.py` imported, never reimplemented; `freeze_selector.py` **not**
  run; `MedEvalKit/` untouched; no process killed on either card.
- **Not measured, stated as such**: latency and VRAM for every decoding setting; `top_p`; `top_k`
  (`topk20` is 1,869 of 7,035 lines generated, `topk50` is empty, `minp005` is 2,253 of 7,035 — the round
  described these as "never generated", which is loosely stated: they were *started and abandoned*, and
  no endpoint was computed from them); energy anywhere in the round.

---

## 9. Ranked next steps

1. **Take the free 20.5%.** Move the open generator to T=0.3 and confirm the dedup assumption holds in
   the deployed path. Accuracy is a TIE with clean guardrails; the compute cut is arithmetic, not
   inference. Cheapest real win available. *Before shipping, run ≥10 generation seeds at T∈{0.3, 0.5,
   0.7} — three seeds cannot resolve the accuracy side and the whole frontier's accuracy axis currently
   rests on point estimates.*
2. **Fix the cost model everywhere it appears** (`resolution_greedy_vs_arm.py:125` and anything
   downstream of it). Charging the verifier per slot instead of per distinct candidate inflates every
   saving quoted against the deployed open arm by 1.566×. The `13.17×` figure must become `8.41×`
   wherever it has been written down. This is the propagate-the-correction failure mode the retrospective
   §9.6 warns about.
3. **Attack the verifier's cost, not the generator's.** It is 51.6% of the open arm and it runs at
   1,003,520 pixels — 4× the generator that produced the candidates it ranks. Nobody chose that; the
   verifier scripts never had a cap flag. A verifier-resolution sweep with a matched control on the full
   n=2,345 pool is the single highest-value unrun experiment on this half, and the n=600 pilot already
   found monotone degradation without resolving the cap320 rung.
4. **Close the candidate-distribution direction in writing.** The union-pool row (+0.163326 oracle,
   −0.004975 selected, sel_eff 0.765 → 0.601) belongs in the retrospective's negative-results section as
   the definitive statement. Three axes — decoding, resolution, vision views — now agree.
5. **Propagate the three-resolution structural finding** into the paper's cost accounting. The macro-8 is
   not a single operating point; R32 = 3.816 was derived at cap320 geometry and re-costs to 3.5908 at the
   resolution 62.5% of the macro actually runs at.
6. **Do NOT** revisit decoding parameters as an accuracy lever, scale N, or widen the candidate pool by
   any training-free means. All measured; ceiling reported above; the binding limit is the scorer.

---

## 10. Housekeeping

- Doc: `results/cascade_methods/docs/current/INFERENCE_PARAMS_2026-08-13.md` (this file).
- **New artifacts this session**: `_infparams_verify_recompute.json`, `_infparams_seedaware.json`,
  `_infparams_ceiling.json`, `_infparams_currency.json`, `_infparams_rejudge.json`,
  `_infparams_mcq_geometry.json`, `_infparams_frontier.json`, `_infparams_cost_correction.json`,
  `_infparams_verify_got.npz` — all under `results/cascade_methods/artifacts/`.
- **New code**: `src/cascade_methods/inference_params_{verify,seedaware,ceiling,currency,frontier,
  mcq_geometry,rejudge_build,rejudge_analyze}.py`; `runners/run_rejudge_confound.sh`;
  log `logs/rejudge_confound_2026-08-14.log`.
- **Round artifacts verified**: `decoding_sweep_2026-08-13.json`, `resolution_sweep_2026-08-13.json`,
  `vision_diversity_2026-08-13.json` and their `_*_parts/`.
- ⚠️ `ckpts/`, `feats_hidden/`, `logs/` still have **zero tracked files**, and `feats_hidden/` is what the
  dedup correction in §4 is measured from. A push does not protect it. Unchanged top-priority chore.
- ⚠️ Never run `freeze_selector.py`; never score a visual LoRA under vLLM.
