> # ⚠️ HISTORICAL / NEGATIVE RESULT (annotated 2026-07-29)
>
> Written **2026-06-19**. Its own first line is the verdict: the verification-augmented deferral router
> (VADR) is **not novel and not a real win** (guardrail reversal 91.6% vs 64.3%; its one novel claim moves
> AUROC −0.003). Preserved as the negative-result record — retrospective §6.1 N16.
> **Current method:** `METHOD_FINAL_2026-07.md`. **Definitive account:**
> [`PROJECT_RETROSPECTIVE_2026-07-29.md`](PROJECT_RETROSPECTIVE_2026-07-29.md).

# ⚠️ FINAL VERDICT (2026-06-18): VADR is NOT novel and NOT a real-efficiency win — see below.
> **AUDIT CAVEAT (2026-06-19):** the esc%@parity numbers in this (superseded) doc come from
> `metarouter_honest.py`, which selected its operating threshold on the TEST half (leakage), so absolute
> esc% here are optimistic for ALL methods compared in that file. The verdict (VADR not novel + illusory
> latency) is unaffected. For the faithful, leakage-free gate bake-off use `baseline_compare.py`.
> Adversarial prior-art check + decisive empirical test conclude:
> 1. **Not novel.** The core (defer by Δ=P̂(strong-right)−P̂(weak-right) from cheap features only) is
>    exactly Jitkrittum et al. NeurIPS 2023 (arXiv 2307.02764; post-hoc Diff-Prob/Diff-01); LLM port
>    Gupta 2024 (2404.10136); bi-directional "predict strong w/o running it" Warren&Dras 2025
>    (2504.19391); self-verification escalation = AutoMix (2310.12963); P(True) = Kadavath 2022.
> 2. **The one possibly-new piece fails.** "P(True) carries orthogonal signal for the STRONG model's
>    recoverability" — adding P(True) to logprob features changes strong-model-correctness AUROC by
>    −0.003/+0.001 (ALL-6/ALL-5) and recoverability AUROC by only +0.01 (noise). Claim does not hold
>    (consistent with concurrent *Verification Mirage* 2605.10850: self-verify coupled to generator
>    error). The earlier escalation "wins" were operating-point noise + the Jitkrittum Δ-reshuffle.
> 3. **Not a real-efficiency win** (latency ~16% ALL-5 / ~0 ALL-6; energy +20% worse; FLOPs neutral).
> CONCLUSION: on this model+data the cascade GATE is signal-limited (recoverability AUROC ceiling
> ~0.6 from any cheap signal) — no training-free gate beats the confidence margin gate in a way that
> is simultaneously novel, real-efficiency-positive, and guardrail-safe. Kept below for the record.

---

# (superseded) Verification-Augmented Deferral Router (VADR)

> Metric (user's binding constraint): minimize the **32B escalation rate** at iso **pooled** accuracy
> (cascade acc ≥ always-32B-think), pooled over **ALL-6** and **ALL-5** (excl. MedXpert). Strong leg
> held at the standard 32B-think@fullres, so this isolates the **cascade decision rule**. All numbers
> from held-out evaluation (model fit on a stratified calib half, escalation@parity read on the test
> half, 30 seeds, mean±SEM). Code in `src/cascade_methods/`.

## SOTA baseline (from focused literature workflow)
Calibrated **prob_margin confidence gate** (FrugalGPT/Chow family): escalate iff p_top1−p_top2 < τ.
On our escalation metric it is the strongest training-free, guardrail-clean gate (conformal CP-Router
collapses to it for few-option MCQ; recoverability/benefit gates calibrated on PMC-train-only do not
transfer). This is the number to beat.

## The method (VADR)
Escalate by the predicted **accuracy gain** of escalating:
  Δ(x) = P̂(32B-correct | x) − P̂(7B-correct | x)
Two small logistic models (calibration-only — both VLMs frozen) over cheap-model features:
  - 7B per-letter logprob shape: margin, max-prob, prob-margin, entropy, Gini, #options
  - free cross-resolution agreement (7B already decoded at cap80/160/320/640)
  - **one-pass self-verification P(True)**: re-prompt the 7B with its own answer, read normalized
    P("Yes") — an independent reasoning pass.
Calibration is **pooled** across benchmarks (the fix that makes the strong-model head transferable —
PMC-train-only calibration fails). Escalate the highest-Δ items until pooled parity.

**Why it beats confidence:** the confidence gate escalates by 7B *uncertainty*; it cannot tell a
*recoverable* error (32B will fix) from a *futile* one (32B also fails). VADR predicts the strong
model's recoverability and escalates only where escalation is expected to help. The key enabling
signal is self-verification: it is WEAK at predicting 7B-correctness (AUC 0.65-0.70 < margin 0.70-0.75,
corr ~0.65) yet carries ORTHOGONAL signal for predicting **32B-recoverability** — exactly what Δ needs
(ablation: meta-Δ *without* verify is no better than SOTA; *with* verify it wins).

## Result — escalation @ iso pooled-accuracy (held-out, 30 seeds)
| pool | SOTA prob_margin | **VADR (meta-Δ)** | reduction |
|---|---|---|---|
| ALL-6 | 61.8 ± 1.3 % | **54.9 ± 1.7 %** | −7.0 pt (11% rel) *** |
| ALL-5 (excl MedXpert) | 35.2 ± 1.6 % | **26.9 ± 1.1 %** | −8.3 pt (24% rel) *** |
| COMPETENT-4 | 29.5 ± 1.5 % | **22.3 ± 1.0 %** | −7.2 pt (24% rel) *** |

VADR also has **higher accuracy at every fixed escalation budget** (10–50%) on all pools — full
escalation-accuracy **frontier dominance** (`frontier_compare.py`). *** = mean gap > 2·SEM.

## Honest caveats (no over-claiming)
1. **Compute vs escalation-rate (quantified, cap80-verify run done).** VADR's headline win is on the
   **32B-call rate** (your metric): frontier-dominant, −7/−8pt at iso pooled-accuracy. But the signal
   costs extra **cheap** passes, so on TOTAL FLOPs there is no free lunch:
   | pool | SOTA gate (esc / bb) | VADR full (cap320-verify+xcap) | VADR compute-neutral (cap80-verify, no xcap) |
   |---|---|---|---|
   | ALL-6 | 62.4% / 73.4% | esc 54.9% (signal-only metric) | esc 58.0% / bb 76.3% (+3% FLOPs, −4.4pt calls) |
   | ALL-5 | 36.2% / 45.7% | esc 26.9% (signal-only metric) | esc 32.8% / bb 45.3% (−0.4% FLOPs, −3.4pt calls) |
   The big −7/−8pt escalation win needs cross-resolution passes that raise total FLOPs; at iso-total-
   compute (cap80-verify, no cross-cap) VADR still cuts 32B calls −3 to −4pt. A 32B-think call ≈ 15×
   a cheap 7B pass in FLOPs (and far more in latency/serving), so VADR is a clean win when the 32B is
   the bottleneck, and ~FLOPs-neutral otherwise. **Deep finding: the cascade gate on this model+data
   is near-saturated — escalation reductions cost roughly-proportional cheap compute.**
2. **Per-benchmark guardrail.** VADR optimizes pooled accuracy; under the stricter never-worse-than-7B
   per-benchmark guardrail it is NOT better than the confidence gate (it can dip a single benchmark).
   The user's chosen metric is pooled accuracy, under which VADR wins.
3. **Needs strong-model labels on a pooled calibration set** (to fit P̂(32B-correct)); the confidence
   gate does not. This is the standard bi-directional-proxy assumption and is training-free.

## REAL-TIME LATENCY/ENERGY VERDICT (from live measured data, `latency_estimate.py`)
Calibrated on `rt_cascade_cap320.jsonl` (real per-sample latencies): 7B leg ≈0.19s, 32B leg ≈28s
(lat32 ≈ 0.072s/think-token, R²=0.99 — latency is dominated by the 32B's serial think decode). The
verify pass ≈ one extra 0.19s cheap forward (negligible). Result:
| pool | SOTA mean lat | VADR mean lat | latency Δ | 32B think-token volume (VADR/SOTA) | energy |
|---|---|---|---|---|---|
| ALL-6 | 18.0s | 17.9s | **~0%** | 0.98× (unchanged!) | +21% worse |
| ALL-5 | 7.8s | 6.6s | **−16%** (tput +19%) | 0.82× | +20% worse |

**Critical: the escalation-COUNT win does NOT translate to latency/energy on ALL-6.** VADR escalates
10% fewer items but each averages LONGER 32B reasoning (416→454 tokens), so total 32B work is
unchanged (0.98×). A latency-aware variant (escalate by Δ / predicted-gen32) does NOT fix this
(ALL-6 1.00×) — the recoverable items ARE the long-reasoning ones; there is no cheap path to parity.
The think-token oracle (escalate only beneficial, cheapest-first) is 0.04×/0.05× (huge headroom) but
unreachable (recoverability unpredictable). **Net real-efficiency verdict: VADR gives a modest ~16%
latency win on ALL-5 only; ~0 on ALL-6; energy ~20% worse everywhere (verify overhead).** Use the
escalation-COUNT framing ONLY where literal 32B-call count is the cost (e.g. per-call API pricing /
rate limits), not for latency/energy/FLOPs.

## Reproduce
- self-verification signal: `src/labeling/run_7b_selfverify_vllm.py` → `ckpts/gate_7b_verify/`
- honest comparison + ablation: `python3 src/cascade_methods/metarouter_honest.py`
- frontier dominance: `python3 src/cascade_methods/frontier_compare.py`
