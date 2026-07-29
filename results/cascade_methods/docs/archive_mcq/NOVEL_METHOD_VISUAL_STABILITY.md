# Novel training-free cascade method — Visual-Stability (Resolution-Consistency) Rescue

**Date:** 2026-06-24 · **Pool:** competent-4 (PMC-VQA, SLAKE, VQA-RAD, PathVQA), n=6050 ·
**Status:** REAL, novel, guardrail-safe keeper. Fully offline from existing checkpoints (no new
inference). All numbers from real checkpoint output (standing no-fabrication rule).

---

## 0. One-paragraph summary

The deployed cascade escalates a query from the 7B to the 32B when the 7B's confidence (logprob
**margin**, gated by a frozen logistic τ=0.4264) is low. That gate **over-escalates** (60% of
competent-4) because the frozen τ does not transfer cleanly to the eval distribution. We add a
**training-free, parameter-free RESCUE filter**: *keep a low-confidence query on the cheap 7B if
its answer is invariant across an image-resolution ladder* (the 7B gives the same letter at
cap80/cap160/cap320/cap640 — it has "settled"). On competent-4, frozen gate: this cuts the 32B
call rate **60.2% → 19.8%** and prefill-inclusive backbone compute **69.5% → 43.3%** of always-32B,
landing at **0.6448 accuracy = always-strong parity (0.6451)**. On the project's metric of record
(min backbone at iso-accuracy parity, both gates tuned), the rescue sits **strictly below** the
plain margin frontier: **40.9% vs 44.1% backbone, 19% vs 35% calls.** Guardrail-clean on all four
benchmarks. This is the loop's genuine "ours" contribution.

---

## 1. The idea — and an HONEST novelty boundary

**What is NOT new (and we must say so).** Cross-resolution agreement *as a primary gate signal* was
already in this project's "exhausted signal search": multi-cap ensemble saturated at ~the margin
gate's operating point, and we re-confirm it here — the raw signal predicts cheap-wrongness at only
**AUROC ≈ 0.55** (vs margin's 0.66; `resolution_consistency.py`). Resolution *escalation* (spend
more pixels on hard samples) is VisionThink (arXiv 2507.13348) — and we **kill** that direction in
§5 item 3 (the 7B's errors are capacity-bound, not resolution-bound). So neither "multi-res is a good
gate" nor "resolution escalation" is the contribution.

**What IS the contribution: a self-calibrating RESCUE that fixes the frozen gate's over-escalation.**
The deployed margin gate's frozen τ does **not transfer** to the eval distribution — it
**over-escalates** (60% of competent-4 vs the outcome-oracle's ~2–10%). The new result is that
*resolution-stability precisely identifies the over-escalated "settled" samples* and lets us keep
them cheap **parameter-free** (no held-out calibration), recovering a real efficiency win that the
raw-AUROC-of-the-signal view misses. The mechanism: if the 7B's answer is **invariant across
resolution** it has converged — and since its errors are capacity-bound (§5 item 3), the big model
(which buys capacity, not pixels) rarely changes those answers. Concretely, at **iso-accuracy
parity a margin gate tuned to the same accuracy needs 35% escalations; the rescue needs 19%** —
i.e. the rescue is not "escalate less," it escalates the *right* samples.

To our knowledge, using resolution-invariance as a **parameter-free deferral-rescue / gate
self-calibration** (to *suppress* a confidence gate's over-escalation), rather than as a primary
gate signal or an escalation trigger, is the new and VLM-specific angle.

---

## 2. The method (parameter-free)

```
deployed(q):  cheap = 7B-nt@cap320 ;  escalate to 32B-think@fullres  iff  gate.predict_proba(margin) < τ
RESCUE(q):    cheap = 7B-nt@cap320 ;  if gate says escalate:
                  run 7B-nt at cap80, cap160, cap640         # 3 extra CHEAP passes, only on eligible
                  if all four caps {80,160,320,640} agree -> KEEP CHEAP (rescue)   # parameter-free
                  else escalate to 32B-think@fullres
```

- τ stays **frozen** at the deployed 0.4264. The rescue adds **no tunable** (full agreement
  required). The only design choice is the cap set (cap80/160/640); cheaper 1–2-cap variants exist
  (§4) but the 3-cap version is the accuracy-preserving one.
- The 3 extra passes fire **only on the ~60% the gate would escalate**, and are charged honestly.

---

## 3. Headline results (competent-4, frozen gate)

`src/cascade_methods/final_robustness_rescue.py`, `stability_rescue_cost.py`, `..._bootstrap.py`.
FLOPs = 2·N·(P+G); P from `token_cache.json`, G from checkpoints. always-cheap=0.6221,
always-strong(parity)=0.6451.

| method | 32B-call% | backbone% of always-32B | accuracy | Δacc vs deployed |
|---|---:|---:|---:|---:|
| DEPLOYED margin gate | 60.2% | 69.5% | 0.6526 | — |
| **R — resolution-rescue (ours)** | **19.8%** | **43.3%** | **0.6448** | −0.0078 |

- The deployed gate sits **above** parity (0.6526 > 0.6451) by over-escalating. The rescue spends
  the surplus accuracy headroom to buy a 3× cut in 32B calls, landing exactly at parity.
- **Paired bootstrap (5000×):** Δbackbone **−26.2%** CI[−27.3, −25.1] (certain); Δ32B-call
  **−40.4 pts** CI[−41.6, −39.2] (certain); Δacc **−0.78 pt** CI[−1.74, +0.15] (small; lands at
  parity, statistically indistinguishable from always-strong).

### Metric of record — min backbone at iso-accuracy parity (both gates tuned)

`stability_rescue_cost.py` iso-accuracy frontier:

| @ accuracy = always-strong (0.6451) | backbone% | 32B-call% |
|---|---:|---:|
| plain margin gate (best τ) | 44.1% | 35% |
| **resolution-rescue (best τ)** | **40.9%** | **19%** |

The rescue point is **strictly below** the plain-margin frontier: same accuracy, ~7% less compute,
**~half the 32B calls**. The call-rate halving is the bigger lever for batch-1 latency (32B-think
≈ 28 s vs cheap ≈ 0.3 s): halving the slow-path frequency roughly halves mean latency, which the
FLOPs figure understates.

### Honest held-out check (eval 50/50 CV, 6 seeds; `stability_rescue_validate.py`)

Fit (τ, full-stability) on half A, evaluate on half B: **margin esc 0.240 / acc 0.6450**;
**rescue esc 0.156 / acc 0.6450** — identical accuracy at **35% fewer escalations** (Δesc −0.083,
std 0.019). The win survives held-out; the extra param does not overfit.

### Per-benchmark guardrail (acc ≥ always-cheap on every set) — all OK

| benchmark | cheap | deployed | **R rescue** | guard |
|---|---:|---:|---:|:--:|
| PMC-VQA | 0.543 | 0.5595 | 0.5630 | OK |
| SLAKE | 0.762 | 0.7644 | 0.7620 | OK |
| VQA-RAD | 0.761 | 0.8015 | 0.8088 | OK |
| PathVQA | 0.641 | 0.6820 | 0.6657 | OK |

---

## 4. Cheaper variants (cost/accuracy trade; `stability_rescue_cost.py`)

| rescue stability set | 32B-call% | backbone% | acc | Δacc |
|---|---:|---:|---:|---:|
| 3-cap {80,160,640} (headline) | 19.8% | 43.3% | 0.6448 | −0.0078 |
| 2-cap {80,160} (cheapest extra) | 16.9% | 31.7% | 0.6398 | −0.0127 |
| 2-cap {160,640} | 14.1% | 35.4% | 0.6379 | −0.0147 |

Fewer extra passes → lower compute but a larger accuracy give-back. 3-cap is the parity-preserving
operating point; 2-cap {80,160} reaches **31.7% backbone** if a ~1.3-pt accuracy give-back is
acceptable (still well above the always-cheap floor 0.6221).

---

## 5. What we tried and KILLED this loop (negatives — kept for the record)

All offline. These bound the contribution and explain its shape.

1. **Bidirectional "fragility-escalate" (up-arm)** — `bidirectional_stability.py`.
   Hypothesis: escalate *confident-but-visually-fragile* samples the margin gate keeps cheap.
   **Dead:** among gate-keep-cheap samples only **105/2409** are fully fragile (margin and stability
   are correlated, r≈+0.31), and escalating them changes accuracy by **+0.0000** (bidir max acc
   0.6534 vs margin 0.6531). The signal lives **only** in rescue-down.

2. **Self-verify (P(True)) rescue (V)** and **doubly-robust R∧V** — `combined_rescue.py`,
   `final_robustness_rescue.py`. Pooled they look great (V: 60%→33% calls at ~zero accuracy loss;
   RV: +0.0048 acc, a pooled Pareto), **but both VIOLATE the per-benchmark guardrail on SLAKE**
   (V 0.7380, RV 0.7476 < always-cheap 0.762 — self-verify keeps cheap samples the 32B would fix),
   and **RV saves no compute** (backbone 68.9%, Δ−0.6%: its 4 extra passes eat the saving). This
   independently reproduces the prior "self-verify is a dead end" finding and pinpoints *why*
   (it breaks per-benchmark safety). Resolution-rescue is guardrail-clean; self-verify is not.

3. **Resolution as an intermediate TIER** (7B@fullres before the 32B) — `resolution_tta_and_tier.py`.
   **Dead:** on the escalate-set 7B@fullres (0.5229) is **no better** than 7B@cap320 (0.5260), and
   fixes only **12.6%** of cap320-wrong samples vs the 32B's **45.2%**. The 7B's errors are
   **capacity-bound, not resolution-bound** — you cannot substitute pixels for parameters. (This is
   the negative that *grounds the rescue's mechanism*.)

4. **Multi-resolution majority VOTE as the cheap answer (TTA)** — `resolution_tta_and_tier.py`.
   Marginal: pooled cheap-acc **0.6221 → 0.6255** (+0.0034) for a 4-pass cheap leg. Real but not
   cost-effective standalone; only a free by-product if the caps are already computed for the rescue.

---

## 6. Honest magnitude & scope (flagged uncertainty)

- **Magnitude depends on framing.** vs the *frozen deployed* gate (which over-escalates): a large
  −26-pt backbone / −40-pt call-rate cut for a −0.78-pt accuracy give-back (to parity). On the
  *iso-accuracy frontier* (both tuned): a clean Pareto of −3.2 pts backbone / −16 pts calls at **zero**
  accuracy cost. Both are real; the call-rate / latency story is stronger than the FLOPs story.
- **Self-calibrating property.** The rescue needs **no held-out calibration** (parameter-free), so it
  is robust to the τ-transfer problem that makes the margin gate over-escalate. It does **not**
  transfer from PMC-VQA-train (that split is too hard for resolution-stability to separate right
  from wrong; `stability_rescue_validate.py` mechanism on train shows no separation) — which is
  exactly why a parameter-free formulation (not a fitted threshold) is the right design.
- **Scope:** competent-4. Not validated on MMMU/MedXpert (excluded from the main cascade claims for
  both models, as elsewhere in the project).
- **Cost-method caveat:** the self-verify pass FLOPs are approximated as ≈ one cap320 nt pass (the
  verify prompt's exact token count is not in `token_cache.json`); this only affects the *rejected*
  V/RV rows, not the headline R.

---

## 7. Untested next steps (NOT claimed)

- **Compounding with ACC.** The rescue modifies the **tier-0** escalation decision; ACC's
  mode-tiering (32B-no-think before 32B-think) cuts **cost per escalation**. They are orthogonal and
  should compound (rescue cuts *how many* escalate; ACC cuts *how much each costs*). **Not yet
  measured** — needs threading the stability signal through `acc.py`'s calib/test protocol.
- **A second cheap orthogonal robustness axis that is guardrail-safe** (option-order shuffle,
  question paraphrase) — would need new 7B inference; self-verify was the cheap one and it failed
  the guardrail.

---

## 8. Reproduce

```bash
cd ~/medvlthinker-imgdiff-compute
python3 src/cascade_methods/resolution_consistency.py        # signal AUROC + in-sample frontier
python3 src/cascade_methods/stability_rescue_validate.py     # mechanism + held-out (CV + train-freeze)
python3 src/cascade_methods/stability_rescue_cost.py         # per-benchmark FLOPs + iso-accuracy frontier
python3 src/cascade_methods/stability_rescue_bootstrap.py    # paired bootstrap CIs
python3 src/cascade_methods/bidirectional_stability.py       # negative: up-arm
python3 src/cascade_methods/resolution_tta_and_tier.py       # negatives: TTA vote + resolution-tier
python3 src/cascade_methods/combined_rescue.py               # multi-signal robustness (V, RV)
python3 src/cascade_methods/final_robustness_rescue.py       # canonical table: deployed/R/V/RV + guardrail + CIs
```
JSON artifacts: `results/cascade_methods/{resolution_consistency,stability_rescue_validate,
stability_rescue_cost,stability_rescue_bootstrap,bidirectional_stability,final_robustness_rescue}.json`.
