> # ⚠️ HISTORICAL (annotated 2026-07-29)
>
> The **MedVLThinker-era** 3-tier compute-configuration cascade. Its numbers (11.34 s → 2.27 s, compute
> 100 → 52%, energy ~5.3×) are real and canonical **for that era and that harness** (evaluation context B:
> internal NGC harness, 6 benchmarks, 8,220 samples) — never mix them with the MedEvalKit figures in the
> retrospective §4. ACC is **not** the current method: on Lingshu its slow reasoning tier would fire ~0% of
> the time. Its agreement gate is prior art (Agreement-Based Cascading, arXiv 2407.02348); what carried
> forward is the *structure*. Current: `METHOD_FINAL_2026-07.md`; definitive:
> [`PROJECT_RETROSPECTIVE_2026-07-29.md`](PROJECT_RETROSPECTIVE_2026-07-29.md).

# Adaptive-Compute Cascade (ACC) — the genuine improvement

> **⚠️ CANONICAL UPDATE (2026-06-27).** Latency/energy in this file use the *superseded co-resident rt_cascade* methodology (think ≈28s; ACC ALL-6 20.0s→5.7s). The **canonical batch-1-native** numbers are: think ≈11.3s; **ACC ALL-6 11.34s→2.27s (−80%), FLOPs 100→52%, energy 6319→1182J (~5×)**; ALL-5 8.88s→0.44s. Source: `master_data.csv` / `GROUND_TRUTH_NUMBERS.md`.



> A confidence-gated cascade that routes each query to the cheapest COMPUTE CONFIGURATION
> (model-size × reasoning-mode × resolution) that suffices, reserving the slow think leg for the
> reasoning residual. Honest eval: held-out 50/50 calib/test, thresholds chosen on calib (min
> latency s.t. calib acc ≥ always-32B-think parity), 20 seeds. Latency from REAL measured batch-1
> data. Code: `src/cascade_methods/acc.py`, `src/cascade/measure_config.py`.

## Terminology note (what "SOTA"/"baseline" means here)
"Margin" (top1−top2 logprob) is NOT a recent method — it is Chow's reject rule (1970) / margin
sampling (~2009). We use it as the **baseline gate** for two reasons: (1) it is the project's
**incumbent/deployed** gate (τ=0.426); (2) it is the **empirically strongest gate in our own
bake-off**. The genuinely recent SOTA cascade GATES — FrugalGPT learned scorer, AutoMix
self-verification, CP-Router conformal, bi-directional-proxy / Jitkrittum learned-deferral — were
benchmarked here (their training-free analogs in `gate_compare.py`: learned-correctness,
learned-deferral, conformal set-size; plus self-verification P(True) in `metarouter_honest.py`), and
on our data they TIE OR LOSE to plain margin (recoverability is ~0.6 AUROC, so learned/deferral gates
have little to exploit — consistent with Jitkrittum et al. NeurIPS 2023, who prove margin deferral is
near-optimal when the strong model's skill is ~input-independent). So below, read "SOTA cascade" as
"standard confidence-gated reasoning cascade (margin gate = the best/incumbent gate we found)", NOT
"newest published method". ACC's win is the STRUCTURE; ACC-v2's gate (cross-model agreement) is the
one signal that actually beats margin, and it is itself an old idea (query-by-committee) used in a
new, cost-free place.

## Tiers
```
Tier 0: 7B  no-think @ cap320     gate τ0 on 7B margin
Tier 1: 32B no-think @ cap320     gate τ1 on 32B-no-think margin   (this leg HAS logprobs; ~0.34s)
Tier 2: 32B think    @ fullres    terminal                          (~28s)
```
Baseline (SOTA confidence-gate cascade, the deployed design): 7B no-think@cap320 → 32B think@fullres.

## Measured per-config latency (batch-1, HF, real)
7B no-think@cap320 ≈ 0.18s | 32B no-think@cap320 ≈ **0.34s** | 32B no-think@fullres ≈ 0.50s |
32B think@fullres ≈ 0.072s/think-token ⇒ **~28s** (rt_cascade, R²=0.99). The think leg is ~80× the
no-think leg — so latency is dominated by how often the think tier fires.

## Result (held-out, 20 seeds) — accuracy matched at parity
| pool | policy | acc (parity tgt) | FLOPs% | latency mean | latency p90 | think-call% | guardrail fails |
|---|---|---|---|---|---|---|---|
| ALL-6 | SOTA | 0.5691 (0.5723) | 81.5% | 20.0s | 44s | 72% | 0.50 |
| ALL-6 | **ACC** | **0.5699** | **54.9%** | **5.7s** | **21s** | 18% | **0.00** |
| ALL-5 | SOTA | 0.6462 (0.6463) | 51.2% | 9.1s | 25s | 42% | 0.80 |
| ALL-5 | **ACC** | **0.6445** | **26.6%** | **0.28s** | 0.4s | 0% | **0.05** |
| COMP-4 | SOTA | 0.6462 | 47.4% | 8.2s | 24s | 38% | 0.75 |
| COMP-4 | **ACC** | **0.6458** | **24.7%** | **0.27s** | 0.4s | 0% | **0.00** |

**ACC vs SOTA: latency −72% (ALL-6) / −97% (ALL-5, COMP-4); FLOPs −22 to −27pt; think-calls −38 to
−53pt; accuracy matched (±0.002); guardrail strictly cleaner.** This is a real-efficiency win on
every axis at once — unlike the gate-side VADR (which only moved an illusory call-count metric).

## Head-to-head vs standard reasoning cascades (`acc_compare.py`, 5 metrics, honest 50/50, 20 seeds)
All three use the same confidence-margin gate (the best/incumbent gate — see Terminology note);
all calibrated to parity (always-32B-think); latency & energy from REAL measured batch-1 data. Methods:
- **M1 ACC (ours)** = 7B-nothink@cap320 → 32B-nothink@cap320 → 32B-think@fullres
- **M2 standard 2-tier (both think)** = 7B-THINK@fullres → 32B-think@fullres
- **M3 standard 3-tier (reasoning-escalation)** = 7B-nothink@cap320 → 7B-THINK@fullres → 32B-think@fullres

| pool | method | acc | esc→32B-think | FLOPs% | latency (mean) | energy/q | guardrail |
|---|---|---|---|---|---|---|---|
| ALL-6 | M2 2-tier(think) | 0.5725 | 86% | 105% | 29.8s | 7049J | 0.35 |
| ALL-6 | M3 3-tier(think) | 0.5697 | 65% | 89% | 23.2s | 5499J | 0.25 |
| ALL-6 | M1 ACC (margin) | 0.5694 | 19% | 55% | 5.9s | 1505J | 0.00 |
| ALL-6 | **M1b ACC+agreement** | **0.5710** | **14%** | **54%** | **4.86s** | **1220J** | **0.00** |
| ALL-5 | M2 2-tier(think) | 0.6437 | 73% | 91% | 20.3s | 4735J | 0.80 |
| ALL-5 | M3 3-tier(think) | 0.6401 | 31% | 48% | 9.0s | 2090J | 1.15 |
| ALL-5 | **M1 ACC** | 0.6457 | **1%** | **27%** | **0.49s** | **88J** | 0.10 |
| COMP-4 | M2 2-tier(think) | 0.6443 | 76% | 94% | 20.4s | 4798J | 0.65 |
| COMP-4 | M3 3-tier(think) | 0.6414 | 30% | 47% | 8.6s | 2008J | 0.70 |
| COMP-4 | **M1 ACC** | 0.6449 | **0%** | **24%** | **0.39s** | **64J** | 0.10 |

**ACC dominates both SOTA cascades on every metric at matched accuracy (±0.003):** 5× lower latency
(ALL-6) to ~40-50× (ALL-5/COMP); 4.7-75× lower energy; ~half the FLOPs (M2 is *>100%* — a 7B-think
cheap leg costs more than always-32B because it is slow AND less accurate on perception, so it
escalates 73-86% to think anyway); and the cleanest never-worse-than-7B guardrail. Root cause: the
SOTA "make every tier reason" assumption is backwards on perception VQA (thinking overthinks); ACC
reserves think for the 0-19% of queries that need it. Saved: `results/cascade_methods/acc_compare.txt`.

## Is the win the GATE or the CONFIG? (`gate_compare.py`) — it's the CONFIG
Holding the 3-tier config FIXED and swapping the gating method (ALL-6, parity 0.5723, latency in s):
| gate | acc | esc→think | FLOPs% | latency | energy |
|---|---|---|---|---|---|
| margin (ACC) | 0.5694 | 19% | 54.7% | 5.93 | 1505J |
| MSP/Chow ≡ conformal | 0.5704 | 19% | 57.8% | 6.60 | 1675J |
| entropy | 0.5695 | 21% | 62.2% | 7.99 | 2033J |
| gini | 0.5705 | 21% | 61.6% | 7.82 | 1991J |
| learned-correctness | 0.5679 | 19% | 60.5% | 7.61 | 1934J |
| learned-defer (VADR-style) | 0.5673 | 14% | 51.0% | 5.01 | 1260J |
| random | 0.5689 | 86% | 130% | 23.4 | 6105J |

All real gates cluster within ~15-30%; only random collapses. **The cascade-METHOD contribution is
therefore the STRUCTURE (which compute-configs to cascade over + confidence routing), not the gate** —
ACC's plain margin is already the lowest-latency confidence gate. Full table: `gate_compare.txt`.

**The best gate-side refinement: CROSS-MODEL AGREEMENT at the think tier** (NOT novel — see below). At
tier 1 both 7B-no-think and 32B-no-think have run, so gate the expensive think tier on whether they
DISAGREE (query-by-committee) instead of 32B-no-think's single-model margin. It strictly improves ACC on
ALL-6 (acc 0.5694→0.5710, think-calls 19%→14%, latency 5.9→4.86s −18%, energy 1505→1220J −19%,
guardrail still 0) and ties on ALL-5/COMP (think already ~0%). This is the recommended ACC gate.
**Prior-art caveat (do NOT claim this as novel):** agreement-gated escalation is exactly
Agreement-Based Cascading (ABC, arXiv 2407.02348, Jul 2024 — ensemble agreement→stop, disagreement→
escalate), and "cross-model disagreement as a correctness signal" is its own 2025-26 line (2509.21837,
2603.25450). It is a known technique applied here in a cost-free place (both votes already computed),
not a new mechanism.
Intuition: when two independent models agree, the answer is almost always right → don't spend the 28s
think pass; reserve think for genuine disagreements.

**Canonical implementation: `src/cascade_methods/acc_v2.py`** (standalone; `acc.py` is the v1/margin
reference, left untouched). It runs the honest v2-vs-v1 eval and freezes deployable thresholds to
`ckpts/acc_v2_thresholds.json`. Deployment note: thresholds frozen on PMC-train (perception-only)
DISABLE the think tier (correct for COMPETENT-4); reasoning-inclusive (ALL-6) deployment needs tau1
calibrated on a mixed set containing reasoning examples (the held-out pooled calibration fires think
~14% on ALL-6).

## Why it works (the contribution)
Standard cascades go small→large and escalate to the large model in its strongest (slow, think)
mode. The ACC inserts the large model in its FAST (no-think) mode as an intermediate tier: on
medical perception VQA the 32B in no-think mode is as accurate as think mode (thinking overthinks
perception) at ~80× lower latency, so it resolves most escalations; only the reasoning residual
(MMMU/MedXpert-style) reaches the slow think tier. Confidence gates at each tier (the 32B-no-think
leg exposes logprobs, so its own margin gates the think tier). The per-benchmark guardrail IMPROVES
because no-think doesn't break correct perception answers.

## Honest notes
- Accuracy is matched to the SOTA cascade (both ≈ parity within calibration-transfer noise, ±0.002);
  ACC ≥ SOTA acc on ALL-6, −0.0017 on ALL-5 (within noise). A small safety margin on τ guarantees ≥parity.
- FLOPs: the think tier re-prefills (think prompt ≠ no-think prompt), but think fires rarely, so total
  FLOPs still drop 22-27pt.
## Novelty (adversarial prior-art check, wqtj35fhz) — INCREMENTAL but defensible
ACC is a SYSTEMS/INTEGRATION contribution, not a new primitive. Every component is published:
- self-gated no-think→think on multimodal VQA = **CAR** (arXiv 2505.15154) — the most threatening hit
  (single model, perplexity-gated, token-not-latency, not medical);
- top1-top2 margin as the CoT gate = **2510.21007**;
- no-think ≈ think on MEDICAL VQA = **Med-R1 / No-Thinking-Med-R1** (2503.13939) — ACC's premise, already known;
- low-res→high-res compute axis = **VisionThink** (2507.13348);
- same-model fast→slow "Speculative" escalation = **HRBench** (2605.28398).

What is genuinely unoccupied (per the 2026 routing survey 2603.04445, which catalogs NO cascade whose
tiers are compute-configs of the same model and names medical-VLM routing an open gap): **the
three-tier-from-two-models structure that inserts the LARGE model's no-think mode as the intermediate
workhorse cascade tier (gated by its own logprob margin), so slow CoT fires only on the residual — on
medical perception VQA, with measured wall-clock latency cuts.**

HONEST FRAMING for a paper: efficiency-systems paper; lead with the structural delta; cite CAR &
2510.21007 as the source of the self-gate (do NOT claim it as new) and Med-R1 for the no-think==think
premise; cite the survey as evidence the combination is uncatalogued. Scope claims to the 4 competent
perception benchmarks (MMMU/MedXpert excluded — both near chance); call it "calibrated wall-clock
latency" (latency model fit on batch-1 measurements + rt_cascade). A tough reviewer will say "CAR +
a small front model + resolution + medical"; the defense is the unpublished 3-tier-from-2-models
structure + large measured latency cuts (72-97%) + the never-worse-than-7B guardrail.
