# Progress Log — Cascade-Method Research Loop (2026-06-17 → 06-18)

> Self-contained record of an autonomous research-loop session on the MedVLThinker 7B→32B
> compute-efficient medical-VQA cascade. Goal set by Leo: find/implement/compare training-free
> model-cascade methods, iterate toward a novel method that beats the deployed gate on compute
> while keeping accuracy, and verify novelty against the literature. **No fabricated numbers** —
> every figure here is from real checkpoint output or real GPU measurement. All code lives in
> `src/cascade_methods/` + `src/cascade/measure_config.py`; detailed docs in `results/cascade_methods/`.

---

## 0. Starting point (the deployed system)
- Two models: `MedVLThinker-7B-RL_m23k` (cheap), `MedVLThinker-32B-RL_m23k` (strong). Both reasoning
  ("think") VLMs; the 7B can also answer directly ("no-think").
- Deployed cascade: 7B-no-think @ image-resolution-cap "cap320" answers; a **frozen confidence-margin
  gate** (τ=0.426, calibrated on held-out PMC-VQA-train) escalates low-confidence cases to 32B-think@fullres.
- 6 benchmarks (8220 eval samples): **competent-4** = PMC-VQA(2000), SLAKE(416), VQA-RAD(272),
  PathVQA(3362); **excluded** = MMMU(170), MedXpert-Reasoning(1446)+Understanding(554) (both models near chance).
- Metric: prefill-inclusive **backbone%** = cascade FLOPs / always-32B-think FLOPs (1 run = 2·N·(P+G),
  N₇=7.6e9 N₃₂=33e9). Guardrail: never worse than always-7B per benchmark.
- Deployed headline (reproduced exactly this session): ALL-6 acc 0.5718, escalation 63.3%, backbone 73.6%.

## 1. Infrastructure built (iteration 0)
- `harness.py` — loads cheap-leg signals (full per-letter logprobs → margin/maxprob/entropy/gini/…),
  32B correctness + gen_tokens, token_cache prefill, PMC-train calibration; scores any escalation
  vector identically to the deployed accounting. **Reproduces the deployed anchor (0.572/63%/73.6%).**
- `frontier.py`, `diagnostics.py`, `methods.py`, `methods_deferral.py`, `ceiling.py`, `compare.py`,
  `evaluate.py` — frontiers, oracle headroom, signal/method registries, comparisons.
- Baselines (cap320): always-7B = 0.5262 (ALL-6)/0.6221(COMP)/0.6201(ALL-5);
  always-32B-think (parity) = 0.5723/0.6451/0.6463. Cheap-leg floor ≈ 8.4% backbone.
- **Oracle headroom @ parity** (escalate only beneficial 7w&32r, cheapest first): ALL-6 11.2% backbone
  (4.6% esc), COMPETENT 10.4% (2.3%) — the ceiling; ~60pt below the deployed gate.
- **Single-signal frontiers (eval-oracle threshold @ parity):** ALL-6 best confidence ~72.5% backbone
  (needs ~60% esc — single-model confidence is weak on the reasoning-heavy pool); COMPETENT prob_margin
  38.9% (29% esc). Established: signals are far from the oracle.

## 2. Iteration 1 — literature + the gate is signal-limited (key negative result)
- **Literature workflow:** 84 methods across 8 families catalogued (`literature_raw.json`). Confirmed
  data facts: stored logprobs are normalized log-softmax (energy/max-logit degenerate to MSP);
  semantic-entropy/self-consistency collapse to analytic option-entropy for single-letter MCQ (no GPU
  win); single-model internal signal not routable (the prior −29σ luck-floor result).
- **Outcome structure (`diagnostics.py`):** of 7B errors, **62.8% are FUTILE to escalate** (32B also
  wrong) on ALL-6; 58.4% on COMPETENT. Of the margin gate's escalations only 22.5% are beneficial,
  15.2% actively harmful. Recoverability rises only weakly with uncertainty (43%→28% across margin
  quintiles) and caps ~43-50%.
- **Benefit-signal ceiling (CV on eval, `ceiling.py`):** predicting escalation benefit from 7B
  features — ALL-6 CV ceiling ≈ confidence (NO gain; futility unpredictable); COMPETENT ~5pt gain.
- **Per-benchmark over-escalation:** SLAKE 7B(0.762)≈32B(0.764) needs ~7% escalation but the gate fires
  49%; VQA-RAD needs 3%, fires 58%. The single τ over-escalates easy benchmarks.
- **Conclusion:** among training-free 7B-feature gates, confidence (margin/prob_margin) is ~optimal;
  the gap to the oracle is unroutable from cheap signals.

## 3. Iteration 2 — the strong leg was running in the wrong mode (the real lever)
- GPU: labelled **32B on held-out PMC-train** (`gate_32b_pmctrain/`, 27 min) + **32B {nothink@fullres,
  think@cap320, nothink@cap320} on all eval** (`gate_32b_modes/`, runner `run_32b_modes_vllm.py`).
- **KEY EMPIRICAL FINDING — thinking overthinks perception VQA.** 32B no-think ≥ 32B-think on the
  competent benchmarks at ~2 decode tokens vs ~477:
  | bench | 32B-think | 32B-nothink | Δ |
  |---|---|---|---|
  | SLAKE | 0.764 | 0.841 | +7.7 |
  | VQA-RAD | 0.776 | 0.893 | +11.7 |
  | PMC-VQA | 0.556 | 0.565 | +0.9 |
  | PathVQA | 0.673 | 0.672 | −0.1 |
  | MMMU | 0.688 | 0.629 | −5.9 (think needed) |
  | MedXpert-R/U | 0.326/0.384 | 0.288/0.301 | think needed |
- Mode-adaptive cascade (`cheap_strong.py`, `multitier.py`): on COMPETENT-4, escalating to 32B-nothink
  instead of 32B-think dropped the cascade from 69.5%→48.6% backbone *and raised accuracy* 0.653→0.660.

## 4. Iteration 3 — resolution compounds it → first "SOTA" config
- `gate_32b_modes/{think_cap320,nothink_cap320}` measured. **32B-no-think@cap320** retains competent
  accuracy (pooled 0.646 ≈ think-parity 0.645) at ~28% of a think call (prefill 375 vs 852, decode 2 vs 477).
- COMPETENT-4 (`final_comparison.py`): MSP/Chow gate → 32B-nothink@cap320 = acc 0.648 at **29.3% backbone**
  (vs best confidence gate on think-leg 67.7%, vs deployed 69.5%), guardrail-clean, frontier 21%.
- Honest finding: ALL-6 single-tier can't reach parity (reasoning needs think); needs a 3-tier.

## 5. Iteration 4 — escalation-rate reframe; VADR; and a string of honest negative results
Leo redirected: focus on **escalation rate at iso pooled-accuracy on ALL-6 and ALL-5**, hold the strong
leg standard, want a cascade-DECISION-RULE contribution (no-think/cap is "a trick anyone can apply").
- **SOTA-baseline workflow** identified the closest method: Bi-directional Model Cascading with Proxy
  Confidence (arXiv 2504.19391) — predict the strong model's correctness without running it.
- **VADR (Verification-Augmented Deferral Router)** built: escalate by predicted Δacc = P̂(32B-right) −
  P̂(7B-right) from cheap features {logprob shape + cross-resolution agreement + one-pass self-verification
  P(True)}. GPU: ran self-verification (`run_7b_selfverify_vllm.py` → `gate_7b_verify*`).
- Pooled-metric result (held-out, 30 seeds, `metarouter_honest.py`): VADR beats the confidence gate —
  ALL-6 54.9% vs 61.8% escalation, ALL-5 26.9% vs 35.2% (significant). **But three findings killed it:**
  1. **Guardrail reversal:** under the per-benchmark never-worse-than-7B guardrail the result FLIPS —
     VADR needs 91.6% (ALL-6)/58.6%(ALL-5) vs the confidence gate's 64.3%/40.9%; its pooled win came from
     quietly sacrificing individual benchmarks. The one guardrail-safe deferral mechanism
     (de-escalate confidently-futile items) gave ZERO honest reduction.
  2. **Latency reality:** the escalation-count win is largely illusory — ALL-6 32B-think-token volume
     unchanged (0.98×; VADR escalates fewer but longer-reasoning calls), ALL-5 0.82×; energy +20% worse.
  3. **Novelty + the load-bearing claim FAIL.** Adversarial prior-art check: VADR's core (defer by
     Δ=P(strong)−P(weak) from cheap features) = Jitkrittum et al. NeurIPS 2023; LLM port Gupta 2024;
     bi-directional Warren&Dras 2025; self-verify = AutoMix; P(True)=Kadavath. Its only possibly-new
     piece — "P(True) predicts the STRONG model's recoverability" — **fails on our data** (adding P(True)
     changes strong-correctness AUROC by −0.003/+0.001; concurrent *Verification Mirage* 2026 finds
     self-verify is coupled to generator error).
- **Robust conclusion:** the cascade GATE is signal-limited (recoverability ~0.6 AUROC from any cheap
  signal); no training-free gate beats plain margin in a way that is novel + real-efficiency-positive +
  guardrail-safe. Big efficiency wins live in the structural strong-leg axis.

## 6. Iteration 5 — ACC: the genuine win (option 1)
Leo: reframe the strong-leg axis as a proper cascade method. Built the **Adaptive-Compute Cascade (ACC)**.
- **Method (3 tiers, stop at first confident):** Tier0 7B-nt@cap320 (≈0.18s) → Tier1 32B-**nt**@cap320
  (≈0.34s, has logprobs) → Tier2 32B-think@fullres (≈28s). Each tier gated by its own margin. The big
  model's FAST no-think mode is the intermediate workhorse; slow think fires only on the residual.
- **GPU: measured real batch-1 latency/energy** per config (`measure_config.py`): 7B-nt 0.18s/25J,
  32B-nt@cap320 0.34s/65J, 32B-think@fullres ≈28s/6994J (0.0716s & 18.17J per think token, R²=0.99).
- **Honest held-out eval (50/50, 20 seeds, `acc.py`/`acc_compare.py`), at parity accuracy:**
  | pool | metric | deployed/std | **ACC** |
  |---|---|---|---|
  | ALL-6 | latency | 20.0s | **5.7s (−72%)** |
  | ALL-6 | FLOPs | 81% | **55%** |
  | ALL-6 | energy | 7049J | **1505J** |
  | ALL-6 | guardrail-fails | 0.35 | **0.00** |
  | ALL-5 | latency | 9.1s | **0.28s (−97%)** |
  | COMP-4 | latency | 8.2s | **0.39s (−97%)** |

### 6a. 5-metric head-to-head vs standard reasoning cascades (`acc_compare.py`)
M1=ACC; M2 = standard 2-tier 7B-**think**→32B-think; M3 = standard 3-tier 7B-nt→7B-think→32B-think
(7B-think exists for all 6, 8220 rows, 100% logprobs; measured 7B-think latency 6.0s). All gated by
margin, calibrated to parity. ALL-6:
| method | acc | esc→think | FLOPs% | latency | energy | guard |
|---|---|---|---|---|---|---|
| M2 std 2-tier(think) | 0.5725 | 86% | 105% | 29.8s | 7049J | 0.35 |
| M3 std 3-tier | 0.5697 | 65% | 89% | 23.2s | 5499J | 0.25 |
| **M1 ACC (margin)** | 0.5694 | 19% | 54.7% | 5.93s | 1505J | 0.00 |
| **M1b ACC+agreement** | 0.5710 | 14% | 53.5% | 4.86s | 1220J | 0.00 |
ACC dominates all 5 at matched accuracy (M2 is >100% FLOPs because a 7B-think cheap leg is slow AND
less accurate on perception, so it still escalates 73-86% to think).

### 6b. Is the win the gate or the config? (`gate_compare.py`) — it's the CONFIG
Holding the 3-tier config fixed and swapping the gate (ALL-6): margin 5.93s/54.7%, MSP≡conformal
6.6s/57.8%, entropy/gini ~7.9s/62%, learned-correctness 7.6s/60.5%, learned-defer(VADR) 5.0s/51% (−0.002
acc), **random 23.4s/130%**. All real gates cluster; only random collapses ⇒ the cascade-method
contribution is the STRUCTURE; ACC's plain margin is already the best confidence gate.

### 6c. Best gate-side refinement — ACC-v2 (cross-model agreement) [NOTE: not novel — = ABC 2407.02348; see §7]
At Tier1 both 7B-nt and 32B-nt have run (free); escalate to think **only when they DISAGREE**
(query-by-committee), not on single-model margin. Strictly Pareto-improves on ALL-6: acc 0.5688→0.5702,
think 18%→14%, latency 5.73→4.83s (−18%), energy 1453→1214J (−19%), guardrail still 0; ties on ALL-5/COMP.
Canonical standalone impl `acc_v2.py` (freezes `ckpts/acc_v2_thresholds.json`). Note: PMC-train calibration
disables think (perception-only scope); reasoning-inclusive deployment needs mixed-set calibration.

## 7. Novelty assessments (honest) — every component is published; the method is a thin COMBINATION
Three adversarial prior-art checks were run (VADR, ACC structure, ACC-v2 agreement gate). Result:
**every building block of the method has prior art**, most of it genuinely prior (2023-2025):
| component | closest prior art | status |
|---|---|---|
| no-think ≈ think on medical VQA (premise) | Med-R1 / No-Thinking-Med-R1 (2503.13939) | prior |
| reasoning can hurt VLMs | "More Thought, Less Accuracy?" (2509.25848) | prior |
| large-model self-gated no-think→think (multimodal) | CAR (2505.15154); margin-gated CoT (2510.21007); HRBench Speculative (2605.28398) | prior/concurrent |
| resolution escalation as a compute axis | VisionThink (2507.13348) | prior |
| multi-tier confidence cascade | FrugalGPT; AutoMix; Cascaded LMs (2506.11887) | prior |
| **agreement-gated escalation (ACC-v2's think gate)** | **Agreement-Based Cascading / ABC (2407.02348, Jul 2024)**; semantic-agreement cascades (2509.21837); cross-model disagreement signal (2603.25450) | **prior** |
| deferral by predicted strong-model benefit (VADR) | Jitkrittum (2307.02764); Gupta (2404.10136); bi-directional proxy (2504.19391) | prior |

- **VADR:** NOT novel; its one new claim (P(True)→strong-recoverability) fails empirically. Abandoned.
- **ACC-v2 agreement gate:** NOT novel — it is Agreement-Based Cascading (ABC, 2024). Earlier I called it
  "the one genuine improvement"; that was wrong. It's a known technique applied in a cost-free place.
- **ACC overall:** INCREMENTAL — a novel *combination/instantiation*, not a new mechanism. The only
  thinly-unoccupied stack (per the 2026 routing survey 2603.04445, which names medical-VLM routing an open
  gap): large-model-no-think as the intermediate workhorse tier + medical perception VQA + agreement think-
  gate + resolution co-variation + measured wall-clock latency/energy. Honest positioning: a SYSTEMS /
  efficiency / application paper that COMBINES known parts (cite ABC, CAR, Med-R1, VisionThink, Jitkrittum),
  with the contribution being the medical-VLM instantiation + the measured efficiency wins under an honest
  protocol with a per-benchmark guarantee — NOT a new gate or a new cascade primitive.

## 7b. Signal search EXHAUSTED — 9 families tried, all capped at the recoverability ceiling (~0.6-0.69 AUROC)
Tested as cascade-gate signals (output AND internal); none beats the ceiling on this model+data:
(1) confidence margin/MSP/entropy/Gini/energy; (2) conformal/CP-Router; (3) learned correctness
(FrugalGPT); (4) recoverability/deferral (Jitkrittum/VADR); (5) self-verification P(True) (AutoMix);
(6) cross-model agreement (ABC); (7) multi-resolution ensemble; (8) compute-elasticity / resolution
trajectory; (9) **hidden-state probe** (7B layer-14 activations, PCA-128, feats_full): predicts 7B
correctness 0.60 vs confidence 0.68, recoverability 0.53 vs 0.63 — WORSE than logprobs (consistent
with the −29σ luck-floor). CONCLUSION: no cheap signal supports a novel high-impact gate here; the
ceiling is a property of THIS 7B+32B pair (the 32B's errors are unpredictable from 7B behavior). A
novel METHOD needs a different setting (model pair / task / signal source). Defensible contributions:
ACC (systems/efficiency) + the recoverability-ceiling characterization (a real negative-result finding).

## 8. Terminology correction (raised by Leo)
"Margin" (top1−top2) is OLD (Chow 1970 / margin sampling ~2009), NOT a recent method. We use it as the
baseline because it is the project's incumbent gate AND the empirically best gate in our bake-off. The
recent SOTA gates (conformal/learned/deferral/self-verification) WERE benchmarked and tie/lose to margin
here. Docs corrected to say "incumbent/best confidence gate", not "SOTA".

## 9. Artifacts produced this session
**Checkpoints (gitignored):** `ckpts/gate_32b_modes/{nothink_fullres,nothink_cap320,think_cap320}/`,
`ckpts/gate_32b_pmctrain/`, `ckpts/gate_32b_pmctrain_nothink_{cap320,fullres}/`,
`ckpts/gate_7b_verify/`, `ckpts/gate_7b_verify_cap80/`, `ckpts/gate_7b_pmctrain_verify*/`,
`ckpts/acc_v2_thresholds.json`, `results/cascade_methods/latency_{7b,7b_think,32b}.jsonl`.
**Code (`src/cascade_methods/`):** harness, frontier, diagnostics, methods, methods_deferral, ceiling,
compare, evaluate, frontier_compare, metarouter_honest, escalation_leaderboard, cheap_strong, multitier,
sota_comparison, final_comparison, final_3tier, latency_estimate, **acc, acc_compare, gate_compare, acc_v2**.
**Labelers/measurement:** `src/labeling/run_32b_modes_vllm.py`, `run_7b_selfverify_vllm.py`,
`src/cascade/measure_config.py`, edited `run_pmctrain_vllm.py` (+--cap).
**Docs (`results/cascade_methods/`):** README (index), FINDINGS (full narrative), METHOD_ACC (headline
method + novelty), METHOD_MATH (the math), METHOD_deferral_router (VADR, superseded), METHOD (early),
literature_raw.json, + saved `.txt` outputs (acc, acc_compare, gate_compare, frontier_compare, etc.).
Top-level CLAUDE.md/README.md/RESULTS.md updated with current status.

## 10. Bottom line & open directions
- **Result to report:** ACC / ACC-v2 — a confidence-gated 3-tier compute-configuration cascade that
  matches always-32B-think accuracy at **−72% latency / −75% energy / ~½ FLOPs on ALL-6, and ~40-97×
  lower latency on the competent/ALL-5 pools**, guardrail-clean. Scope claims to the competent-4
  perception benchmarks (MMMU/MedXpert excluded — both near chance). Latency/energy are calibrated
  wall-clock (measured batch-1 + rt_cascade, R²=0.99); FLOPs exact.
- **What's the contribution (honest):** a novel COMBINATION/instantiation, not a new mechanism — the
  STRUCTURE (large-model-no-think as the fast intermediate tier) on medical VLM, with an ABC-style
  cross-model-agreement think-gate (the gate itself is prior art, §7). The gate space is saturated
  (margin near-optimal); VADR/learned/recoverability gates don't beat it (a clean negative result).
- **Open / next:** (a) paper draft around "structure + agreement gate"; (b) mixed-set calibration of the
  think threshold for honest ALL-6 deployment; (c) the gate is at its ceiling — further gains would need
  a new signal beyond cheap-model confidence (none found that's free + helpful).

---

# Appendix A — Method comparison charts (vs FrugalGPT, CP-Router, AutoMix, …)

## A.1 Named SOTA cascade method → how we benchmarked it
| Named method | Mechanism | Benchmarked here as | Script |
|---|---|---|---|
| FrugalGPT (Chen 2023) | LLM cascade w/ a learned scorer; training-free form = confidence threshold | `margin` (threshold) + `learned-correct` (logistic P̂(correct)) | gate_compare, sota_comparison |
| Chow's rule (1970) / MSP (Hendrycks 2017) | threshold the max softmax prob | `MSP/Chow` | both |
| CP-Router / conformal cascade (2025) | conformal prediction-set-size deferral | `conformal` (≡ MSP for few-option MCQ — set-size monotone in top1-prob) | both |
| DOCTOR (NeurIPS 2021) | Gini-impurity reject detector | `gini` | both |
| AutoMix (NeurIPS 2024) | one-pass self-verification + meta-router | `self-verify P(True)` (metarouter) + `learned-defer` | metarouter_honest, gate_compare |
| Bi-directional proxy (Warren&Dras 2025) / Jitkrittum L2D (NeurIPS 2023) | predict strong-model correctness, defer by Δ=P(strong)−P(weak) | `learned-defer` (= our VADR signal) | gate_compare, metarouter_honest |
| margin sampling (~2009) — project incumbent | top1−top2 logprob | `margin` | both |
| **OURS** | ACC structure + cross-model agreement think-gate | `M1 / M1b / ACC-v2` | acc, acc_compare, acc_v2 |

## A.2 Headline: ACC vs standard reasoning cascades — 5 metrics (honest 50/50, 20 seeds, at parity)
M2 = standard 2-tier 7B-**think**→32B-think; M3 = standard 3-tier 7B-nt→7B-**think**→32B-think; all gated by margin.
Latency/energy from real batch-1 measurements; FLOPs exact. (acc_compare.py / acc_compare.txt)

**ALL-6 (parity acc 0.5723):**
| method | acc | esc→think | FLOPs% | lat mean | lat p90 | energy/q | guardrail-fails |
|---|---|---|---|---|---|---|---|
| M2 standard 2-tier (both think) | 0.5725 | 86% | 105.2% | 29.81s | 58.1s | 7049J | 0.35 |
| M3 standard 3-tier (reasoning-escalation) | 0.5697 | 65% | 88.7% | 23.25s | 53.9s | 5499J | 0.25 |
| M1 ACC (margin gate) | 0.5694 | 19% | 54.7% | 5.93s | 19.9s | 1505J | 0.00 |
| **M1b ACC-v2 (agreement gate, ours)** | **0.5710** | **14%** | **53.5%** | **4.86s** | **16.3s** | **1220J** | **0.00** |

**ALL-5 (excl MedXpert; parity 0.6463):**
| method | acc | esc→think | FLOPs% | lat mean | energy/q | guard |
|---|---|---|---|---|---|---|
| M2 standard 2-tier | 0.6437 | 73% | 91.2% | 20.25s | 4735J | 0.80 |
| M3 standard 3-tier | 0.6401 | 31% | 47.6% | 9.03s | 2090J | 1.15 |
| M1 ACC | 0.6457 | 1% | 27.1% | 0.49s | 88J | 0.10 |
| **M1b ACC-v2 (ours)** | 0.6460 | 0% | 27.1% | **0.44s** | **75J** | 0.10 |

**COMPETENT-4 (parity 0.6451):**
| method | acc | esc→think | FLOPs% | lat mean | energy/q | guard |
|---|---|---|---|---|---|---|
| M2 standard 2-tier | 0.6443 | 76% | 94.1% | 20.41s | 4798J | 0.65 |
| M3 standard 3-tier | 0.6414 | 30% | 46.7% | 8.63s | 2008J | 0.70 |
| M1 ACC | 0.6449 | 0% | 23.6% | 0.39s | 64J | 0.10 |
| **M1b ACC-v2 (ours)** | 0.6449 | 0% | 23.6% | **0.38s** | **61J** | 0.10 |

## A.3 SOTA-gate bake-off on the standard 2-tier cascade — backbone% (FLOPs) at iso-accuracy (honest, PMC-train calib)
Shows how the named gates compare on the deployed-style 7B→32B-think cascade, and what switching the
strong leg to no-think (our config lever) does. p=hits parity, x=misses, !1=fails per-benchmark guardrail
on 1 benchmark. (sota_comparison.py / sota_comparison.txt)

**COMPETENT-4 (parity 0.6451):**
| gate (named) | →32B-think (standard) | →32B-no-think (ours) |
|---|---|---|
| margin (FrugalGPT-confidence) | 69.5% ✓ | 48.6% ✓ |
| prob_margin | 67.7% ✓ (best standard) | 47.4% ✓ |
| MSP/Chow ≡ **CP-Router** | 61.9% ✗guardrail | 43.5% ✓ |
| Gini/DOCTOR | 59.1% ✗guardrail | 41.6% ✓ |
| entropy | 30.7% ✗parity | 22.7% ✗parity |
| **ACC 3-tier (ours, gate_compare honest)** | — | **23.6%** ✓ |

ALL-6: every gate on the 2-tier think cascade misses parity at the honest point (margin 73.6%, MSP 70.6%,
entropy 53.4% — all ✗); only the **ACC 3-tier reaches parity** (needs the think tier for the reasoning
residual). Takeaways: (1) among gates the spread is small and margin/prob_margin are best; (2) the big
lever is the strong-leg config (think→no-think nearly halves backbone); (3) CP-Router's conformal
set-size is monotone in top1-prob for ≤5-option MCQ, so it equals MSP/Chow.

## A.4 Gate bake-off under the FIXED ACC 3-tier config — proves the win is the STRUCTURE, not the gate
Same config (7B-nt@cap320 → 32B-nt@cap320 → 32B-think@fullres); only the escalation gate changes.
(gate_compare.py / gate_compare.txt) **ALL-6 (parity 0.5723):**
| gate (named) | acc | esc→think | FLOPs% | latency | energy | guard |
|---|---|---|---|---|---|---|
| margin (incumbent/Chow; ACC v1) | 0.5694 | 19% | 54.7% | 5.93s | 1505J | 0.00 |
| MSP/Chow ≡ CP-Router | 0.5704 | 19% | 57.8% | 6.60s | 1675J | 0.00 |
| entropy | 0.5695 | 21% | 62.2% | 7.99s | 2033J | 0.00 |
| Gini/DOCTOR | 0.5705 | 21% | 61.6% | 7.82s | 1991J | 0.00 |
| learned-correct (FrugalGPT scorer) | 0.5679 | 19% | 60.5% | 7.61s | 1934J | 0.10 |
| learned-defer (bi-dir/Jitkrittum/VADR) | 0.5673 | 14% | 51.0% | 5.01s | 1260J | 0.00 |
| **agreement (ACC-v2, ours)** | **0.5710** | **14%** | **53.5%** | **4.86s** | **1220J** | **0.00** |
| random (no signal) | 0.5689 | 86% | 130.2% | 23.44s | 6105J | 0.05 |

All real gates cluster (5.0–8.0s, 51–62% FLOPs); only **random collapses** (86%→think). ⇒ the
cascade-method contribution is the STRUCTURE; among gates, agreement (ours) and margin are best.
(ALL-5 / COMPETENT-4 show the same pattern with think firing ~0-1% — see gate_compare.txt.)

---

# Appendix B — The math / equations (full in `results/cascade_methods/METHOD_MATH.md`)

## B.1 Scores (gating signals)
One greedy decode → logprobs over candidate letters `{A: ℓ_A, …}` (natural log), sorted ℓ₍₁₎≥ℓ₍₂₎≥…
- **margin** (the ACC gate): `m = ℓ₍₁₎ − ℓ₍₂₎`. Escalate at a tier iff `m < τ`.
- softmax `pᵢ = e^{ℓᵢ}/Σⱼ e^{ℓⱼ}` → `top1prob=p₍₁₎`, `entropy=−Σ pᵢ ln pᵢ`, `gini=1−Σ pᵢ²`.
- **ACC-v2 think-gate (cross-model agreement):** `disagree = 1[pred_7B-nt ≠ pred_32B-nt]`;
  `s₁ = disagree + ε·(−m_32B-nt)`, ε=1e-6; escalate to think iff `s₁ > τ₁` (τ₁≈1 ⇒ the lowest-margin
  disagreements). Only fitted params: scalars τ₀, τ₁ (calibrated to parity at min latency).

## B.2 Routing & expected per-query cost (32B-no-think is GATED, not run every query)
```
run T0 (7B-nt@cap320), always ;  if m₀ ≥ τ₀ → STOP
else run T1 (32B-nt@cap320)   ;  if not think-gated → STOP
else run T2 (32B-think@fullres)
```
With e₀ = P(escalate past T0), e₁ = P(reach think), for any metric with per-tier cost c_T:
```
        C = c_T0 + e₀·c_T1 + e₁·c_T2
```
FLOPs of one run = `2·N·(P+G)`  (N₇=7.6e9, N₃₂=33e9; P=prompt incl. vision, G=generated).
backbone% = Σ(cascade FLOPs)/Σ(always-32B-think FLOPs). Measured per-tier costs (ALL-6):
| tier | meanP | meanG | FLOPs | latency | energy |
|---|---|---|---|---|---|
| T0 7B-nt@cap320 | 388 | 2 | 0.01e15 | 0.21s | 25J |
| T1 32B-nt@cap320 | 388 | 2 | 0.03e15 | 0.34s | 65J |
| T2 32B-think@fullres | 685 | 391 | 0.07e15 | 26.6s | 6994J |

## B.3 Worked numbers — why ACC saves (latency dominated by the think tier, T2≈80×T1)
- **Deployed 7B-nt→32B-think** (e₁=69%): latency = 0.21 + 0.69·26.6 = **19.4s**; energy = 25 + 0.69·6994 = **5048J**; FLOPs = **79%**.
- **ACC-v2** (e₀=84%, e₁=14%): latency = 0.21 + 0.84·0.34 + 0.14·26.6 = **5.05s**; energy = 25 + 0.84·65 + 0.14·6994 = **1268J**; FLOPs = **55%**.
The win is the e₁·c_T2 term collapsing (69%→14%) because 32B-no-think ≈ 32B-think on perception VQA, so
it absorbs escalations that don't need reasoning. FLOPs drops less than latency because FLOPs counts the
parallel vision PREFILL (paid on every escalation) while latency is dominated by the SERIAL think DECODE.

## B.4 Energy measurement
NVML power sampled every 25 ms during each batch-1 query (`measure_config.py` PowerSampler), integrated
by trapezoid: `E_query = Σ_k (P_k+P_{k+1})/2 · (t_{k+1}−t_k)` [J]. Per-config fit E(G)=α_e·G+β_e:
7B-nt 25J, 32B-nt 65J, 32B-think = 18.17·G − 107.5 ≈ 6994J at G≈391 (≈254 W × 27.5 s; 254 W matches the
rt_cascade NVML readings). Cascade energy uses the same C = c_T0 + e₀·c_T1 + e₁·c_T2.
**Caveat:** FLOPs exact; latency & energy are CALIBRATED wall-clock (fit on measured batch-1 + rt_cascade
think term, R²=0.99), applied via the expected-cost formula — not a single end-to-end pipeline timing.

---

# Appendix C — Autonomous loop v2 (2026-06-18): new models/data, cross-family, consolidation

User mandate: download models/datasets, find a NOVEL method, check novelty, keep researching. Constraints
honored: HF cache → /data/dan/hf_cache; main drive stayed at 72% (<80%); no dependency changes; Qwen2.5-VL
+ vLLM-native peers only. New code: `run_vlm_eval.py` (general Qwen2.5-VL runner, +`--blank`),
`run_peer_eval.py` (model-agnostic `llm.chat` runner), `embed_siglip.py`, `peer_premise.py`,
`peer_router.py`, `peer_router_img.py`, `vision_sensitivity.py`, `sd_test.py`. New downloads (on /data):
InternVL2.5-8B, Phi-3.5-Vision, SigLIP-so400m. New checkpoints: `ckpts/peer/{internvl25_8b,phi35v,blank7b}`,
`feats_peer/siglip_*.npz`. A 7-agent scout workflow ranked cross-family complementarity routing #1.

**Direction A — cross-family complementarity router: KILLED (clean negative, NEW finding).**
- Same-family wall is a nesting artifact: P(32Bw|7Bw)=0.584, φ=0.372 (verified on disk).
- Complementarity REAL & large: oracle union 7B|InternVL=0.753, 7B|InternVL|Phi=0.801 ≫ always-32B 0.645.
- UNEXPLOITABLE: learned router on confidence+agreement = 0.621 ≈ always-7B; SigLIP image+text
  recoverability AUROC = **0.50 (chance)** (refutes "image is the routing signal" here); best image router
  0.636 < parity; majority-vote fusion = 0.602 < 7B (ungated cross-family peers InternVL 0.581/Phi 0.516
  are weaker than the medical 7B 0.622 and drag the vote down). First cross-family medical-VLM map.

**Direction B — vision-sensitivity (blank-image counterfactual): KILLED.** 56.9% of 7B answers unchanged
when image blanked (language-prior reliance — striking diagnostic), but insensitive items are no less
accurate (0.620 vs 0.625) and add nothing to correctness/recoverability AUROC. Not a usable gate.

**Direction C — lossless speculative decoding: INFEASIBLE.** vLLM 0.10 rejects draft-model SD ("use
ngram/medusa/eagle"); a real lossless speedup needs a trained draft head (out of training-free scope).
Answer-level speculation reduces to ACC-v2 (66% 7B-think/32B-think agreement — not new).

**Verdict: 13 signal/mechanism families converge on the recoverability wall.** No novel high-impact routing
primitive exists for this 7B/32B medical-VLM pair. **User chose CONSOLIDATE.** Paper draft:
`paper/cvgip2026_draft.md` — ACC (systems win) + gate-saturation + cross-family map + language-prior
diagnostic, as an efficiency-systems + negative-results CVGIP-2026 contribution. No fabricated numbers.
