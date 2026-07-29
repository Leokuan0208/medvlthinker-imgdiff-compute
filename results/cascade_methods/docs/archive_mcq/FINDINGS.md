# Cascade-method research loop — findings log

All numbers come verbatim from `src/cascade_methods/harness.py` + `frontier.py` over the existing
checkpoints (7B cap320 no-think cheap leg, 32B think strong leg, PMC-train calibration). CPU-only,
no GPU re-runs. Cost = prefill-inclusive backbone% (identical accounting to
`src/sweep/grid_resolution_tau.py`). Metric of record: **min backbone% at iso-accuracy (parity =
always-32B acc)**, with "never worse than always-7B per benchmark" as a guardrail.

## Iteration 0 — foundation (baselines, oracle headroom, single-signal frontiers)

### Baselines (cap320)
| pool | always-7B | always-32B (parity) | gap | cheap-leg floor |
|---|---|---|---|---|
| ALL-6 (n=8220) | 0.5262 | 0.5723 | 0.0461 | 8.4% backbone |
| COMPETENT-4 (n=6050) | 0.6221 | 0.6451 | 0.0230 | 9.0% backbone |

`cheap-leg floor` = backbone% if we escalate nothing (just run the 7B on everything). A cascade
can never go below this. always-32B = 100% by definition.

### Oracle headroom (the prize)
| pool | outcome-oracle @ parity | margin-gate (deployed) | gap to close |
|---|---|---|---|
| ALL-6 | **11.2% backbone**, 4.6% esc | 73.6% backbone, 63% esc | 62 pts |
| COMPETENT-4 | **10.4% backbone**, 2.3% esc | 69.5% backbone, 60% esc | 59 pts |

Outcome-oracle = escalate only the cheapest "7B-wrong & 32B-right" samples until parity. Only
**17.6%** (ALL-6) / **15.7%** (COMPETENT-4) of samples are beneficial at all; you only need to
escalate ~5%/~2% of them (the cheap ones) to reach parity. This is unattainable (it knows
outcomes) but it is the ceiling: the prize is ~60 points of backbone%.

### Single confidence signals — best achievable (eval-oracle threshold @ parity)
The strongest single signal from the 7B's per-letter logprob distribution, with the threshold
chosen optimally on eval (an upper bound on any calibration):

| pool | best signal | esc% | backbone% |
|---|---|---|---|
| ALL-6 | maxlogprob / top1prob | ~60% | ~72.5% |
| COMPETENT-4 | prob_margin | ~29% | ~38.9% |

**Two structural facts established:**
1. **On ALL-6, single-model confidence is nearly useless for routing.** Even with a perfect
   threshold, every signal needs ~60% escalation to reach parity — because MMMU/MedXpert
   escalations seldom help (the 32B is also wrong), so "7B is unconfident" does not imply
   "escalating fixes it." The routable signal in single-model confidence is thin here.
2. **The deployed gate over-escalates.** On COMPETENT-4 the eval-oracle margin threshold reaches
   parity at ~39–41% backbone, but the calibrated gate sits at 69.5% because its threshold is
   pinned to the PMC-train error rate (~46%), far above what the easier eval benchmarks need.
   Threshold/calibration alone leaves ~30 points on the table on COMPETENT-4.

### Implications for the research loop
- The win has two separable components: **(a) better operating-point selection** (the gate
  escalates too much) and **(b) better signal** (predict not just "7B wrong" but "escalation will
  help", i.e. deferral/complementarity-aware — which single-model confidence cannot capture).
- maxlogprob/top1prob (Chow's rule / softmax response) marginally beats the raw-margin signal the
  deployed gate uses — a cheap, free improvement to check under honest calibration.
- Biggest expected gains (to be tested as the literature lands): multi-sample self-consistency,
  7B-think token-level uncertainty, and deferral-aware gating using 32B-on-calibration labels —
  all require a GPU re-run to obtain the needed signal.

Artifacts: `results/cascade_methods/frontier_{ALL-6,COMPETENT-4}_cap320.json`.

## Iteration 1 — literature, outcome structure, and the benefit-signal ceiling

Literature sweep (8 families, 84 methods; `results/cascade_methods/literature_raw.json`). It
independently arrived at the same direction the oracle analysis pointed to: **recoverability-aware
deferral** ("58% of escalations are futile"). It also confirmed three data facts: our stored
`opt_logprobs` are normalized log-softmax (sum-exp≈1) so **energy/max-logit degenerate to MSP**
(dropped); **semantic entropy/self-consistency collapse to our analytic option-entropy** for
single-letter MCQ (no GPU sampling needed); and single-model internal signal is not routable
(the −29σ luck-floor). Methods deemed N/A for our setup are logged in the raw catalog.

### Outcome structure (diagnostic, `diagnostics.py`)
| pool | beneficial(7w,32r) | futile(7w,32w) | harmful(7r,32w) | recoverability P(32r\|7w) |
|---|---|---|---|---|
| ALL-6 | 17.6% | 29.8% | 13.0% | 37.2% (→ 62.8% of errors futile) |
| COMPETENT-4 | 15.7% | 22.0% | 13.4% | 41.6% (→ 58.4% futile) |

Of the margin gate's escalations, only **22.5%** are beneficial, **15.2%** harmful. Recoverability
rises only weakly with 7B uncertainty (43%→28% across margin quintiles) and caps ~43-50% even for
the most uncertain — confidence alone can't separate recoverable from futile.

### Benefit-signal ceiling (in-distribution CV on eval, `ceiling.py`) — KEY, partly negative
Predicting escalation benefit from 7B features, *peeking at eval outcomes* (upper bound):
| pool | outcome-oracle | best confidence (eval-oracle) | benefit-predictor CV ceiling |
|---|---|---|---|
| ALL-6 | 11.2% | 73.7% (prob_margin) | **74.7%** (no gain over confidence) |
| COMPETENT-4 | 10.4% | 38.9% (prob_margin) | **33.3%** (benefit-z logistic; ~5pt gain) |

**On ALL-6, the 7B features carry NO routable benefit signal beyond confidence** — the futile mass
(MedXpert/MMMU near-chance) is unpredictable, which independently justifies excluding those
benchmarks and caps any 7B-feature method at ~73%. **On COMPETENT-4 there is a real but modest
~5pt signal gain** (logistic; GBM overfits → worse). So the recoverability *signal* is a minor lever.

### The dominant lever is CALIBRATION, not signal (`per-benchmark transfer diag`)
The honest margin gate over-escalates badly because the 7B's confidence doesn't track its
per-benchmark competence. At the PMC-train-calibrated threshold:
| benchmark | 7B acc | 32B acc | escalates | needs (eval-oracle) | backbone |
|---|---|---|---|---|---|
| SLAKE | 0.762 | 0.764 | 49% | **7%** | 55% |
| VQA-RAD | 0.761 | 0.776 | 58% | **3%** | 65% |
| PMC-VQA | 0.543 | 0.556 | 58% | 21% | 67% |
| PathVQA | 0.641 | 0.673 | 63% | 34% | 73% |

On SLAKE/VQA-RAD the 7B already ≈ the 32B, so almost no escalation is needed, yet the gate fires
~50-58%. A single raw-margin τ calibrated on (hard) PMC-train escalates ~55-63% everywhere. The
COMPETENT-4 honest gap (69.5% gate → 38.9% eval-oracle) is ~30pt of **calibration/transfer waste**,
6× larger than the ~5pt signal lever. **The novel method must primarily fix operating-point
transfer** (a benchmark-invariant or intrinsically-thresholded score), with benefit-awareness as a
secondary refinement. The deferral rule "escalate iff predicted Δacc>0" is attractive because its
operating point is intrinsic (no τ to transfer) — to be tested once 32B-on-PMC-train lands.

Zero-rerun honest leaderboard (`compare.py`, COMPETENT-4): maxlogprob/Chow (61.9%) and Gini (59.1%)
already beat the deployed margin gate (69.5%) at parity, but currently trip the never-worse-than-7B
guardrail on 1 benchmark — to be reconciled.

### Definitive honest leaderboard with 32B-on-calib (`evaluate.py`)
32B-on-PMC-train completed (3000 rows, 27 min): calib 7B=0.459, 32B=0.496, recoverability=0.266
(note: calib recoverability 0.266 << eval 0.42 — PMC-train is an unrepresentative, pessimistic
calibration set for the strong model). Results, COMPETENT-4, parity=0.6451:

- **Recoverability/benefit deferral does NOT transfer.** Trained on PMC-train, its *eval signal
  ceiling* (ORACLE thr) is 45-57% backbone — WORSE than raw confidence prob_margin (38.9%). The
  in-distribution CV gain (33.3%) did not survive PMC-train→eval transfer; p32(x) can't predict
  32B-correctness on benchmarks it never saw (esp. MMMU/MedXpert).
- **Honest, guardrail-clean, parity-hitting backbone is ~67-69% for every 7B-feature method.**
  Best deployable: prob_margin (ERR-RATE) 67.7%, vs deployed margin 69.5% — a real but ~2pt win.
  Cheaper signals (gini 59%, maxprob 62%, entropy 31%) all break the never-worse-than-7B guardrail.
- The CALIB-PARITY selector (calibrate to PMC-train cascade-parity) is *worse* than the error-rate
  rule here, because PMC-train's 32B barely beats its 7B (0.496 vs 0.459) → lax threshold.

**Conclusion (gate side): the gate is near-saturated.** Among training-free 7B-feature gates,
confidence (prob_margin) is ~optimal; the recoverability signal is real in-distribution but doesn't
transfer from a single-benchmark calibration set. This space is also crowded (2026 papers on
confidence-advantage routing). The remaining big lever is structural, not the gate.

## Iteration 2 — cheaper strong leg (the dominant cost lever)

The cheap-leg floor is ~8% backbone; **the escalated 32B call dominates** (~477 think-decode tokens
+ full-res prefill). Decode ≈ 36% and full-res prefill ≈ 36% of each 32B call, so a no-think and/or
cap320 strong leg could cut per-escalation cost ~36-72% *if accuracy holds*. New parameterized
runner `src/labeling/run_32b_modes_vllm.py`; comparison `src/cascade_methods/strong_leg.py`.
Early signal (32B-no-think@full-res, PMC-VQA, n=768): **acc=0.559 ≈ 32B-think (0.556)** at ~2 tokens
vs 477 — promising, but the decisive reasoning benchmarks (MMMU +14pt from think; MedXpert) are
pending. Novelty caveat: within-model "when to think" (ThinkSwitcher/SelfBudgeter/Adaptive Budget
Forcing) is crowded; the cascade-target mode/resolution framing must be checked for novelty.

### THE KEY FINDING: thinking HURTS perception VQA — escalate to 32B no-think (`run_32b_modes_vllm.py`)
32B no-think vs think, per benchmark (full eval, full-res):
| benchmark | 32B-think | 32B-**nothink** | Δ | (decode tok: 477→2) |
|---|---|---|---|---|
| SLAKE | 0.764 | **0.841** | +7.7 | thinking overthinks |
| VQA-RAD | 0.776 | **0.893** | +11.7 | thinking overthinks |
| PMC-VQA | 0.556 | 0.565 | +0.9 | |
| PathVQA | 0.673 | 0.672 | −0.1 | |
| MMMU | 0.688 | 0.629 | −5.9 | think needed |
| MedXpert-R | 0.326 | 0.288 | −3.8 | think needed |
| MedXpert-U | 0.384 | 0.301 | −8.3 | think needed |

On the 4 **competent** benchmarks, 32B-no-think (pooled **0.658**) ≥ 32B-think (0.645) while costing
~2 decode tokens instead of ~477. Think only helps on the excluded reasoning benchmarks. The strong
leg has been running in the wrong mode.

### Mode-adaptive cascade (`cheap_strong.py`, `multitier.py`) — denominator = always-32B-think
| | deployed →think (honest 7B-gate) | 2-tier →**nothink** | 3-tier (mode-adaptive) |
|---|---|---|---|
| COMPETENT-4 @ parity | 69.5% (frontier 40.3%) | **48.6% honest, acc 0.660** / frontier 27.4% | 28.7% (think never fires → = 2-tier) |
| ALL-6 @ parity | 73.6% (frontier 75.9%) | can't reach (caps 0.569) | **frontier 60.5%** (think on only 10%) |

- **COMPETENT-4: just switching the escalation target to 32B-no-think** drops the deployed cascade
  from 69.5%→**48.6%** backbone (compute saving 30%→51%) AND raises accuracy 0.653→0.660 — same
  margin gate, honest PMC-train calibration. Best gate threshold reaches parity at **27.4%**.
- **ALL-6: a 3-tier 7B→32B-nothink→32B-think** cascade reaches think-parity at **60.5%** (vs 75.9%)
  by routing only the ~10% reasoning-hard residual to think. 32B-no-think has full logprobs, so the
  tier-2→tier-3 gate is a confidence threshold like tier-1.

Pending: 32B {think,nothink}@cap320 (does reduced strong-leg resolution cut prefill further?) and
32B-nothink-on-PMC-train (honest tier-2 gate calibration for the 3-tier). The emerging novel method
is a **size×reasoning-mode cascade** that allocates thinking only to the reasoning residual.

### Head-to-head vs current SOTA cascade gates (`sota_comparison.py`, honest, guardrail-checked)
COMPETENT-4 @ think-parity (0.6451), backbone% (lower=better), honest PMC-train calibration:
| gate (training-free SOTA) | STRONG=think (standard cascade) | STRONG=**no-think (ours)** |
|---|---|---|
| margin (FrugalGPT-style) | 69.5% ✓ | 48.6% ✓ |
| prob_margin | **67.7% ✓** (best SOTA) | 47.4% ✓ |
| MSP / Chow's rule | 61.9% ✗guardrail | **43.5% ✓** (best ours) |
| entropy | 30.7% ✗parity+guardrail | 22.7% ✗ |
| Gini / DOCTOR Dα | 59.1% ✗guardrail | 41.6% ✓ |
| CP-Router (conformal LAC set-size) | ~~≡ MSP~~ **CORRECTED: over-escalates 69-80%, loses** | (see below) |
| post-hoc deferral (recoverability) | doesn't transfer (worse than confidence) | — |

> **POST-AUDIT CORRECTION (2026-06-19, see `baseline_audit.json` + `baseline_compare.py`/`.txt`):** two
> claims above/below are SUPERSEDED. (1) **CP-Router is NOT ≡ MSP** — that only holds when 1−q̂≥0.5 and for
> ≤5 options; our eval has 5-10 options and the rules diverge up to ~18% at CP-Router's αs. Implemented
> faithfully (LAC prediction set |C|≠1 + FBE α\*), CP-Router **over-escalates (69-80%)** and loses to the
> confidence gates. (2) **Self-verification (AutoMix) is NOT a weak gate** — with a faithful verify-threshold
> (meta-verifier variant), it reaches ~parity at **20% escalation / 28.7% FLOPs on COMPETENT-4** (best of all
> gates there) and 26%/35% on ALL-5, though it collapses on ALL-6 (79%). The "self-verify weak (AUC 0.65-0.70)"
> note below refers to *predicting 7B-correctness*, which is consistent, but the *gate* conclusion was wrong.
> Net: no single gate dominates across pools; this does not change ACC's structural win (the no-think tier).

**Best SOTA = prob_margin+think @ 67.7%. Best ours = MSP/Chow+no-think @ 43.5%** (acc 0.6557 >
parity, never-worse-than-7B on all 4 benchmarks: PMC 0.569, SLAKE 0.810, VQA-RAD 0.886, PathVQA
0.670). ≈24pt absolute / 36% relative compute reduction over SOTA, same honest protocol. The
no-think strong leg also *fixes* the guardrail violations that cheap gates (MSP/Gini) suffer on the
think leg (no-think doesn't overthink → fewer harmful escalations). On ALL-6, only the **3-tier
(60.5% frontier)** beats the SOTA gates' ~73%. Full table: `results/cascade_methods/sota_comparison.txt`.

### Think/no-think are COMPLEMENTARY (`diagnostic`) — the scientific core
2×2 of 32B-no-think × 32B-think outcomes: think *fixes* 11.5% of no-think's errors but *breaks*
11.1% (overthinking) — nearly canceling in aggregate (why pooled think≈no-think) yet large
per-question. **Per-question oracle mode selection → 0.683 (ALL-6) / 0.759 (COMPETENT-4)**, +10-11pt
over either mode alone. The choice is routable from 32B-no-think confidence: think-recoverability is
35-39% when no-think is unsure vs 10-13% when confident. This motivates the 3-tier's middle gate.
BUT a learned think-router ceiling test (CV on eval, 32B-no-think + 7B features → predict "think
helps") is a NEGATIVE result: routing lifts ALL-6 only to ~0.578 (vs 0.568 base, oracle 0.683) —
because think *breaks* no-think-correct answers in the SAME uncertain region where it *fixes*
others, so the net is ~flat. **The complementarity is real but NOT routable from confidence**
(same lesson as the futile-escalation result). So the mode complementarity cannot be used to exceed
parity; the deployable contribution is reaching parity *cheaply* by defaulting the strong leg to
no-think, with the 3-tier routing only the reasoning residual to think (a modest, compute-only gain
on ALL-6). Do NOT claim accuracy gains above the strong baseline.

## Iteration 3 — strong-leg RESOLUTION compounds the mode win → SOTA (`final_comparison.py`)
Full 2×2 of the 32B strong leg (mode × resolution), competent-4 pooled accuracy @ backbone% (vs
always-32B-think@fullres):
| strong-leg config | acc | backbone | note |
|---|---|---|---|
| think@fullres (SOTA default) | 0.6451 | 100% | the baseline to beat |
| nothink@fullres | 0.6582 | 66% | mode win |
| think@cap320 | 0.6319 | 72% | misses parity |
| **nothink@cap320 (ours)** | **0.6463** | **39%** | mode+resolution: > parity AND cheapest |

32B-no-think@cap320 retains competent accuracy (SLAKE 0.849, VQA-RAD 0.853, PMC 0.551, PathVQA 0.661)
at ~28% of a think call (prefill 375 vs 852, decode 2 vs 477). **THE SOTA RESULT (honest, COMPETENT-4):**
| | best **SOTA** gate→leg | **OURS** |
|---|---|---|
| config | prob_margin → 32B-think@fullres | MSP/Chow → 32B-**nothink@cap320** |
| backbone @ parity | **67.7%** | **29.3%** |
| accuracy | 0.6451 | 0.6481 (> parity) |
| never-worse-than-7B | ✓ | ✓ (PMC .559, SLAKE .815, VQA-RAD .853, PathVQA .664) |
| eval-oracle frontier | 39% | **21-23%** |

**≈2.3× less compute than the best SOTA gate** (compute saving 70.7% vs 32.3%), honest, guardrail-
clean, exceeding parity. margin→nothink@cap320 = 32.4% (acc 0.6522, larger cushion) is the robust
variant. Even *always*-32B-nothink@cap320 (0.646 @ 39%) beats always-32B-think. Full matrix:
`results/cascade_methods/final_comparison.txt`.

ALL-6 (honest, `final_3tier.py`): every single-tier config misses parity. The 3-tier
7B→32B-nothink@cap320→32B-think@fullres reaches parity (0.5723) but at **75.7% backbone honestly —
NOT a win** (deployed think cascade = 73.6%). Its eval-oracle ceiling is 53.4% (would beat SOTA),
but the honest tier-2 route-to-think gate over-escalates (42% vs the oracle's 19%) because
"think-helps" is not routable from confidence. Conclusion: on ALL-6 no training-free cascade beats
the deployed level — exactly why the project excludes MMMU/MedXpert. **The SOTA claim is COMPETENT-4.**

## Iteration 4 — REFRAME: minimize 32B escalation rate at iso-accuracy (ALL-6 & ALL-5)
Per user: the no-think/resolution trick is orthogonal (anyone can apply it); focus on the CASCADE
DECISION RULE. Metric = min **32B-think escalation rate** s.t. cascade acc ≥ always-32B-think, on
**ALL-6** and **ALL-5** (= 6 minus MedXpert; keeps recoverable MMMU). Strong leg held standard
(32B-think@fullres). `escalation_leaderboard.py`. Baselines to beat:
| | ALL-6 (parity 0.5723) | ALL-5 (parity 0.6463) |
|---|---|---|
| outcome-oracle escalation | 4.6% | 2.6% |
| best single-gate EVAL-ORACLE | ~60% (logistic 59.8) | ~35% (margin 34.9) |
| deployed margin (honest) | 63.3% (misses parity 0.5718) | 60.2% (0.6529 ✓) |

**Cheap-signal fusion is SATURATED** (`multi-resolution test`): CV-fused p7(7B-correct) from all
logprob signals + the free cap80/160/320/640 resolution-ensemble agreement reaches only 60.3% (ALL-6)
/ 42% (ALL-5, worse than single margin) — cross-cap agreement alone is useless (AUC 0.566). So any
function of the cheap model's forward pass(es) cannot escalate below ~60%/~35%. To beat SOTA we need
NEW information → testing AutoMix/P(True)-style **self-verification** (`run_7b_selfverify_vllm.py`,
one extra cheap pass) as a decorrelated signal, plus a focused workflow to pin the current SOTA
cascade baseline.

### SOTA baseline (workflow): calibrated prob_margin confidence gate (FrugalGPT/Chow family).
### Novel method: Verification-Augmented Deferral Router (meta-Δ)
Escalation score = predicted accuracy GAIN Δ(x) = P̂(32B-right|x) − P̂(7B-right|x), two small logistic
models on cheap features [7B logprob shape + free cross-resolution agreement + one-pass
self-verification P(True)], CV-pooled calibration. `metarouter_honest.py` (held-out calib/test,
30 seeds, model fit on calib half, escalation@parity read on test half).

**Self-verification ran** (`gate_7b_verify`): weak at predicting 7B-correctness (AUC 0.65-0.70 < margin
0.70-0.75, corr ~0.65) — but carries ORTHOGONAL signal for predicting 32B-RECOVERABILITY, which is
what Δ needs. Ablation: meta-Δ *without* verify is WORSE than SOTA; *with* verify it wins.

**Result under POOLED accuracy (standard cascade metric):** meta-Δ full significantly beats SOTA:
| pool | SOTA prob_margin | meta-Δ full | Δ (sig) |
|---|---|---|---|
| ALL-6 | 61.8±1.3% | **54.9±1.7%** | −7.0 *** |
| ALL-5 | 35.2±1.6% | **26.9±1.1%** | −8.3 *** |
| COMPETENT-4 | 29.5±1.5% | **22.3±1.0%** | −7.2 *** |

**BUT under the per-benchmark never-worse-than-7B GUARDRAIL the result REVERSES** (critical, honest):
| pool | SOTA (parity+guardrail) | meta-Δ (parity+guardrail) |
|---|---|---|
| ALL-6 | **64.3%** | 91.6% |
| ALL-5 | **40.9%** | 58.6% |
meta-Δ's pooled win comes from *sacrificing individual benchmarks* (worst-benchmark dip −3pt vs 7B).
Confidence escalates by uncertainty, which is intrinsically per-benchmark-safe; the deferral router
skips guardrail-critical items. The one guardrail-safe deferral mechanism (de-escalate only
confidently-futile items from the confidence set) gives **ZERO honest reduction** — recoverability is
too noisily predictable to safely remove any escalation (outcome-oracle headroom is 4.6%/2.6% but
unreachable).

**ROBUST CONCLUSION (≈20 experiments):** under the per-benchmark guardrail, the confidence gate is
near-optimal and NO training-free deferral/recoverability/verification method beats it. Under
pooled-accuracy-only, the Verification-Augmented Deferral Router beats SOTA by ~7-8pt (significant).
The metric choice (pooled vs per-benchmark guardrail) decides whether we have a SOTA win.

The novel method = **a (size × reasoning-mode × resolution) compute-configuration cascade**: the
discovered optimal strong-leg config is no-think@cap320 (not the default think@fullres), and think@
fullres is reserved as a 3rd tier for the reasoning residual, all gated by cheap-leg confidence.

