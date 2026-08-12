> # ⚠️ HISTORICAL (annotated 2026-07-29)
>
> Written **2026-06-17**, describing the MedVLThinker compute-configuration cascade. Preserved as the
> origin document of the compute-configuration idea. **It is not the current method** — see
> `METHOD_FINAL_2026-07.md` and [`PROJECT_RETROSPECTIVE_2026-07-29.md`](PROJECT_RETROSPECTIVE_2026-07-29.md).

# Final method — Compute-Configuration Cascade (mode × resolution × size)

> Result of the autonomous research loop. All numbers come verbatim from
> `src/cascade_methods/*` over real checkpoints; cost = prefill-inclusive backbone% with the
> denominator FIXED at always-32B-think@fullres (the strong baseline). No fabricated numbers.

## The discovery

The deployed cascade escalates uncertain 7B answers to **32B-think@full-resolution**. But the
escalated 32B call dominates cost (~477 reasoning tokens + ~852 prompt tokens of vision-heavy
prefill), and that mode/resolution is **the wrong configuration** for the competent benchmarks:

- **Thinking overthinks perception medical-VQA.** 32B-no-think ≥ 32B-think on all 4 competent
  benchmarks (SLAKE +7.7, VQA-RAD +11.7, PMC +0.9, PathVQA −0.1 pt) at ~2 decode tokens vs ~477.
- **The strong leg tolerates low resolution.** 32B-no-think@cap320 retains competent accuracy
  (pooled 0.646 ≈ 32B-think@fullres 0.645) at ~28% of a think call (prefill 375 vs 852).

So the strong leg should run **no-think @ cap320**, not think @ full-res. Think@full-res helps only
on the *excluded* reasoning benchmarks (MMMU, MedXpert), where it is reserved as a 3rd tier.

## The method

A confidence-gated cascade over **compute configurations** (size × reasoning-mode × resolution):

```
Tier 1   7B  no-think  cap320     -- answer if 7B margin >= tau1
Tier 2   32B no-think  cap320     -- answer if 32B-no-think margin >= tau2   (COMPETENT default)
Tier 3   32B think     fullres    -- only the reasoning-hard residual        (ALL-6 only)
```

Gates are the cheap per-tier confidence margin, calibrated on held-out PMC-VQA train (training-free,
no fine-tuning, both VLMs frozen). On the 4 competent benchmarks tier 3 never fires (a 2-tier
cascade); on the full 6 it routes the ~10% reasoning residual to think.

## Headline results (honest: PMC-train calibration, per-benchmark never-worse-than-7B guardrail)

**COMPETENT-4** (n=6050), parity = always-32B-think = 0.6451:
| method | accuracy | backbone% | guardrail |
|---|---|---|---|
| always-32B-think@fullres (strong baseline) | 0.6451 | 100% | — |
| deployed cascade (margin → 32B-think@fullres) | 0.6526 | 69.5% | ✓ |
| **best SOTA gate (prob_margin → think@fullres)** | 0.6451 | **67.7%** | ✓ |
| **OURS: MSP/Chow → 32B-nothink@cap320** | **0.6481** | **29.3%** | ✓ |
| OURS: margin → 32B-nothink@cap320 (robust, larger cushion) | 0.6522 | 32.4% | ✓ |
| OURS eval-oracle frontier (gate-quality ceiling) | 0.6451 | 21-23% | — |

**≈2.3× less compute than the best SOTA gate** (compute saving 70.7% vs 32.3%), at higher accuracy,
never worse than 7B on any benchmark (PMC 0.559, SLAKE 0.815, VQA-RAD 0.853, PathVQA 0.664).

**ALL-6** (n=8220), parity = 0.5723 — *not a win, reported honestly*: single-tier configs miss
parity (reasoning needs think). The 3-tier (7B→32B-nothink@cap320→32B-think@fullres) reaches parity
but only at **75.7% backbone honestly** (worse than the deployed 73.6%), because the tier-2
route-to-think gate over-escalates — "think helps" is not routable from confidence (verified). Its
eval-oracle ceiling is 53.4% (would beat SOTA) but honest calibration can't capture it. This is
consistent with the project's decision to EXCLUDE MMMU/MedXpert: on the near-chance reasoning
benchmarks, escalation futility is unpredictable and no training-free cascade helps. **The headline
claim is COMPETENT-4 only.**

## Why it beats the SOTA gates

The SOTA training-free gates (FrugalGPT-confidence, MSP/Chow, entropy, Gini/DOCTOR, CP-Router
conformal set-size, post-hoc/recoverability deferral) are all decision rules on the *cheap* model's
signal; in our bake-off they cluster at 60-77% backbone and the gate is near-saturated (recoverability
deferral does not even transfer). They all share the same expensive strong leg. Our win is orthogonal
and structural: **fix the escalation target's configuration**, which no gate can do. A cheaper strong
leg also *removes* the guardrail violations that cheap gates (MSP, Gini) suffer on the think leg —
no-think doesn't overthink, so it makes fewer harmful escalations.

## Honest novelty & limitations

- The *components* are individually known in 2025-26 literature: reasoning-can-hurt-VLMs
  (arXiv 2509.25848), confidence cascades (FrugalGPT; "Calibration/Cascading/Cleaning" 2026),
  3-tier cascades (NeurIPS 2025 2506.11887, base→large-think→human), resolution-efficient VLMs
  (FastVLM). The **combination** — escalating to large-model *no-think @ reduced-resolution* as a
  cheaper-AND-more-accurate cascade target, with think reserved as a reasoning-only 3rd tier, on
  medical VLM with prefill-inclusive FLOPs — is a new configuration with a large (≈2.3×) effect.
- think/no-think are complementary (per-question oracle 0.683) but the complementarity is NOT
  routable from confidence, so we do not claim accuracy *above* the strong baseline — only parity at
  far lower compute.
- Calibration set is PMC-VQA-train only; the honest operating point over-escalates (52% vs the
  oracle's ~31%), leaving the 29.3%→21% gap as calibration headroom.

## Reproduce
- Strong-leg ablation runner: `src/labeling/run_32b_modes_vllm.py` → `ckpts/gate_32b_modes/`
- Definitive leaderboard: `python3 src/cascade_methods/final_comparison.py`
- SOTA gate bake-off: `python3 src/cascade_methods/sota_comparison.py`
- Mechanism diagnostics: `diagnostics.py`, `multitier.py`, `ceiling.py`
