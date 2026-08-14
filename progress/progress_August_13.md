# Progress — August 13–14, 2026 (can the 7B's inference parameters improve the samples? — no, but they can make them cheaper)

> **Follows `progress_August_12.md`.** The question was Leo's: *"can changing the 7B's INFERENCE
> PARAMETERS improve the samples it generates?"* — a coverage attack, licensed by the project's own
> diagnosis that the binding limit is candidate provenance rather than the scorer. Three sweeps ran
> (decoding, resolution, vision views). **The answer is no on accuracy, and the lever is now closed with
> a measured ceiling rather than a null.** What the round actually produced is a **cost** result, a
> **cost-model correction**, and a structural finding about resolution that closes 62.5% of the macro by
> impossibility. Round doc:
> `results/cascade_methods/docs/current/INFERENCE_PARAMS_2026-08-13.md`. Every number names its
> artifact. Abstention appears nowhere.

---

## 1. The answer, first line

**No 7B inference-parameter change produces an accuracy gain that survives inference at the level of the
claim being made.** The best setting in the grid is temperature 0.5 at **+0.005686 SELECTED**, and once
generation-seed variance is in the interval it reads **[−0.00256, +0.01407]** — spanning zero. Corrected
family-wise over the seven settings tested against one control: **[−0.00583, +0.01748]**.

**The measured ceiling of the lever**, so it is closed rather than revisited (`_infparams_ceiling.json`):

- best single setting, chosen *by reading the eval endpoint*: **+0.004151 macro-8**
- per-question **oracle over all 8 settings**, unattainable by any router: **+0.020682 macro-8**
- **union pool** of all 8 settings × 3 seeds — 29.97 distinct candidates per item, 24× the deployed
  budget: oracle@8 rises **+0.163326**, and the frozen verifier's SELECTED **falls 0.004975**, sel_eff
  collapsing 0.765038 → 0.600969

That last line is the cleanest closure statement this project has produced on the
candidate-distribution question. It is not a null. It is a large, significant, correctly-signed coverage
gain that converts to a negative.

---

## 2. I re-derived the decoding sweep before believing any of it

From the raw dumps, without importing a line of its analysis code — my own judge join, my own verifier
join, my own bootstrap seed (20260814, deliberately different from theirs)
(`_infparams_verify_recompute.json`):

- Null test 1, frozen metric vs `genframe_data.PUBLISHED`: **PASS, max abs deviation
  3.5967302447481586e-07** — bit-identical to the value the sweep reports.
- Identity `selected = oracle@8 × sel_eff`: **1.11e-16** max abs error over all 24 pools.
- Every per-seed `oracle@8` / `sel_eff` / `selected`, all 8 settings × 3 seeds: **exact match**.
- Every pooled delta point estimate: **exact match** (CIs differ only in bootstrap RNG).
- Missing judge / verifier slots on my own join: **0 / 0**, all 24 pools.

The arithmetic is sound. Then I found four things to change.

---

## 3. Change 1 — the item bootstrap is the wrong interval, and it flips the headline

Every delta in the sweep uses a **paired item bootstrap**, which resamples the 2,345 questions and holds
the generation seed **fixed**. But the claim is about a sampling *distribution*, and each seed is one
draw from it. The control's seed sd of SELECTED is **0.003962** — comparable to the +0.005686 being
claimed. So I recomputed everything three ways (`_infparams_seedaware.json`):

| setting | d SELECTED | item-only (as reported) | **seed-aware** | Welch t |
|---|---:|---|---|---|
| T=0.5 | +0.005686 | [+0.00100, +0.01038] **WIN** | **[−0.00256, +0.01407] TIE** | 2.19, df 3.1 n.s. |
| rp 1.10 | +0.004833 | [+0.00085, +0.00896] **WIN** | **[−0.00554, +0.01535] TIE** | 1.04 n.s. |
| T=1.0 | −0.016489 | [−0.02203, −0.01095] LOSS | [−0.02857, −0.00526] **LOSS** | −3.17 |
| T=1.3 | −0.045629 | [−0.05387, −0.03781] LOSS | [−0.05899, −0.03227] **LOSS** | −8.27 **sig** |

**Both CI-clean wins dissolve; both losses survive.** Not a coincidence — the T≥1.0 effects are several
times the seed sd and the candidate wins are not. The sweep's own pre-registration predicted no setting
would win, so T=0.5 was found by looking, and the family-wise rate is the operating one. Also: the
sweep's per-seed pairing (seed *k* with seed *k*) is arbitrary; all nine cross-seed pairings of T=0.5
against the control run **−0.00085 to +0.01109**, 8 of 9 positive. A consistent direction and an
underpowered test — three seeds cannot certify +0.006 against a 0.004 seed sd.

---

## 4. Change 2 — I suspected the grader, spent the GPU, and the grader is clean

Labels came from two **disjoint** sources: a preload cache from earlier runs, and a fresh judge pass. The
fresh share is a **monotone function of the treatment** — 0.0798 at T=0.3 rising to 0.5004 at T=1.3,
because hotter sampling emits more novel strings. Any systematic disagreement between the sources would
manufacture a temperature effect in exactly the observed direction, and the two sets share zero keys, so
it could not be checked from the files.

So I re-judged **4,047 preload-labelled slots** with the same harness the sweep used for its fresh labels
(`runners/run_rejudge_confound.sh`, `_infparams_rejudge.json`):

- **agreement 1.000 — 4,047 / 4,047, zero disagreements**, 1.000 in each of the three cells
- cached positive rate 0.2962688411168767, re-judged 0.2962688411168767 — identical
- patching the re-judged labels back leaves **every headline delta unchanged to 6 dp**

**Refuted by measurement.** The judge is deterministic and label reuse across settings is legitimate — it
removes judge noise as a between-arm confound. Credit to the sweep. (A related caution against a test of
my own: the 836-item "same-source" subset gives d = exactly 0.0, but its pool Jaccard is 0.9428 against
0.6617 overall — it keeps about a sixth of the contrast, so that null is attenuation, not evidence.)

---

## 5. Change 3 — the round's cost model overcharges the deployed arm by 1.57×

`resolution_greedy_vs_arm.py:125` charges the deployed open arm `8 * f_gen_cap320 + 8 * f_ver` —
**eight verifier forwards**. The verifier does not run eight times; candidates are **deduplicated by
normalized answer string** first. Three independent confirmations: `feats_hidden/generator_eval_s0of2`
(4,472) + `s1of2` (4,471) = **8,943 rows over 2,345 questions = 3.8136 per question**; the frozen
transfer dumps hold 18,760 slots but 8,943 distinct strings; and the sweep's own score cache is keyed
`(ds, idx, ans)`.

`_infparams_cost_correction.json`:

| | as charged | dedup-aware |
|---|---:|---:|
| deployed open arm, FLOPs/question | 1.473725e+14 | **9.408480e+13** (overcharge **1.5664×**) |
| verifier share of the arm | 69.10% | **51.60%** |
| one greedy native decode vs the arm | 0.075902× → **13.17×** | 0.118891× → **8.41×** |

Every conclusion's direction is unchanged and the saving is still large, but **"13.17×" must become
"8.41×" wherever it has been written down.** This is exactly the propagate-the-correction failure mode
retrospective §9.6 warns about, so it goes in the doc as a ranked next step rather than a footnote.

---

## 6. Change 4 — and the correction turns temperature into a COST lever

The distinct-candidate count **is a function of the swept parameter** — 2.182 at T=0.3, 3.665 at T=0.7,
5.476 at T=1.3 — and the verifier only pays for distinct candidates. Generation cost is identical across
the whole grid (same N, same `max_tokens`, same image cap; generated tokens are 1.2% of compute). So
temperature moves the arm's cost, which nobody reported.

**The frontier** (`_infparams_frontier.json`), dedup-aware whole-arm FLOPs relative to the deployed T=0.7:

| setting | SELECTED | d vs control (seed-aware) | distinct | **arm FLOPs** | Pareto |
|---|---:|---|---:|---:|:--:|
| **T=0.3** | 0.486283 | +0.005402 [−0.00498, +0.01578] TIE | 2.182 | **0.7952×** | **★** |
| **T=0.5** | 0.486567 | +0.005686 [−0.00256, +0.01407] TIE | 2.955 | 0.9019× | **★** |
| min_p 0.10 | 0.484861 | +0.003980 [−0.00512, +0.01322] TIE | 2.980 | 0.9054× | |
| T=0.7 (deployed) | 0.480881 | — | 3.665 | 1.0000× | |
| T=1.0 | 0.464392 | −0.016489 LOSS | 4.642 | 1.1348× | |
| T=1.3 | 0.435252 | −0.045629 LOSS | 5.476 | 1.2500× | |

**T=0.3 holds accuracy — TIE, guardrail-clean on all three cells — at −20.5% open-arm compute.** That is
a real minimum-cost-at-parity win and the round's deliverable. Two honest limits: it is 20.5% of the
*open arm only* (3 of 8 cells), and it rests on the deployment actually deduplicating before scoring.
And the accuracy axis of this frontier is **not separated** — every difference except the T≥1.0 losses
has a CI spanning zero, so dominance is on point estimates; cost is the axis that actually distinguishes
them, and cost is arithmetic on measured token geometry.

**Latency and VRAM: NOT MEASURED** for any decoding setting. No mechanism for VRAM to move, but that is
an argument, not a measurement.

---

## 7. Resolution: it cannot touch the MCQ half, and I checked that myself

This was flagged in advance as the one result that would outrank everything else — higher resolution
raising greedy MCQ accuracy would move 62.5% of the macro with no selection involved. **It is
impossible**, and I verified it independently rather than trusting the sweep's table
(`_infparams_mcq_geometry.json`).

The code path: `Qwen2_5_VL_vllm.py:51` reads `CAP_MAX_PIXELS` with default `"0"` and line 54 applies it
only `if _MP`; unset means the processor keeps its own default, **12,845,056**. The only two runners that
set the variable were written for this round. Re-measured from the image files with the harness's own
`smart_resize`, max resized pixels: SLAKE 1,048,576 · VQA-RAD 1,345,536 · PathVQA 657,272 · MedXpert
9,107,712 · **PMC-VQA 11,684,175** — **0.000 above the cap in all five.** SLAKE, VQA-RAD and MedXpert
match the sweep exactly; my PMC max differs from its 8,870,964 (I resolve `figures/` before `images/`,
following `PMC_VQA.py:48,60`), immaterial since both are below the cap.

**The published MCQ arms already see every pixel the model would ever receive.** Cutting resolution is
not free either: +0.007606 macro-8 lost over four of five cells, 2.6× the 0.0029 threshold.

The structural finding stands and belongs in the paper: **one reported macro-8 is evaluated at three
different resolutions** — MCQ legs and the 32B-direct bar at 12,845,056, the open generator and its
comparator at 250,880, the open LoRA verifier at 1,003,520. A **51.2× spread, with the scorer seeing 4×
more of the image than the generator whose candidates it ranks.** Each comparison is internally matched
so the per-cell deltas stand, but the macro is not one operating point.

---

## 8. What this round means for the direction

The brief's premise was that the lever is the candidate **distribution**. Three axes tested it this week
and all three agree: **the distribution moves, and none of it converts.** Decoding (this round),
resolution (oracle@8 +0.0132 SIG → selected +0.0051 n.s., sel_eff −0.0220 SIG on the very cell that
gained), vision views (oracle@8 +0.0267 SIG on 3/3 draws → selected 0/12 significant, sel_eff −0.0238).
The union-pool row is the general statement: 24× the budget, +0.163 coverage, **−0.005 accuracy**.

The capture-recapture ceiling is confirmed distribution-specific as the brief warned — it moved
0.650070 → 0.672574 across the resolution ladder and 0.680477 → 0.713874 across vision views. **The
+0.0091 free-coverage bound is a cap320, T=0.7 number and must not be quoted as universal.**

**The remaining levers are the scorer's training distribution (no longer training-free) or the
generator's competence.** And on cost — which is the live objective — the verifier is now the target: it
is 51.6% of the open arm and runs at 4× the generator's resolution because no verifier script ever had a
cap flag.

---

## 9. Housekeeping

- Doc: `results/cascade_methods/docs/current/INFERENCE_PARAMS_2026-08-13.md`.
- Artifacts this session: `_infparams_verify_recompute.json`, `_infparams_seedaware.json`,
  `_infparams_ceiling.json`, `_infparams_currency.json`, `_infparams_rejudge.json`,
  `_infparams_mcq_geometry.json`, `_infparams_frontier.json`, `_infparams_cost_correction.json`,
  `_infparams_verify_got.npz`. Round artifacts verified: `decoding_sweep_2026-08-13.json`,
  `resolution_sweep_2026-08-13.json`, `vision_diversity_2026-08-13.json` and their `_*_parts/`.
- Code: `src/cascade_methods/inference_params_{verify,seedaware,ceiling,currency,frontier,mcq_geometry,
  rejudge_build,rejudge_analyze}.py`; `runners/run_rejudge_confound.sh`;
  log `logs/rejudge_confound_2026-08-14.log`.
- One GPU job ran (the re-judge, tp=2, both cards idle at launch, waits rather than evicts). Nothing was
  killed. `MedEvalKit/` untouched. `freeze_selector.py` **not** run. No visual LoRA scored under vLLM.
- Minor provenance note for the record: the decoding round describes `top_k` and `min_p=0.05` as "never
  generated". They were *started and abandoned* — `topk20` has 1,869 of 7,035 lines, `minp005` 2,253,
  `topk50` zero. No endpoint was computed from any of them, so nothing is affected, but "not generated"
  should read "not completed".
- ⚠️ `ckpts/`, `feats_hidden/`, `logs/` still have **zero tracked files**, and `feats_hidden/` is what
  the §5 dedup correction is measured from. A push does not protect it. Unchanged top-priority chore.
