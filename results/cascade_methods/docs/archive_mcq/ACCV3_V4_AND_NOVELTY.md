# ACC-v3 (validated improvement) + ACC-v4 (novel composition) + gate-data-size + novelty

**Date:** 2026-06-24 · Baseline = ACC-v2 (cross-model agreement cascade, "Ours") · Honest 50/50
calib/test ×20 seeds, min-think@parity, measured batch-1 latency/energy. Families with a real firing
think tier: MedVLThinker (Lingshu = fast non-reasoning think; QoQ = degenerate). All real-output numbers.

---

## 1. ACC-v3 — confidence-tightened think gate (REAL, validated improvement; incremental novelty)

ACC-v2 fires the expensive think tier on EVERY small-nt/big-nt disagreement. ACC-v3 fires think only
when they disagree **AND** the big no-think model is itself unsure (`m1 < tau1`). Code:
`acc_v3_confgate.py`. Honest calib/test, min-think@parity:

| pool | method | acc | think% | FLOPs% | lat | energy | guard |
|---|---|---:|---:|---:|---:|---:|---:|
| **ALL-6** (reasoning matters) | ACC-v1 (confidence) | 0.5694 | 19% | 54.7% | 2.61s | 1371J | 0.00 |
| ALL-6 | ACC-v2 (agreement, baseline) | reaches parity 19/20 seeds | | | | | |
| ALL-6 | **ACC-v3 (agree+conf, ours)** | **0.5707** | **14%** | **52.6%** | **2.13s** | **1098J** | 0.00 |
| ALL-5 | ACC-v2 | 0.6486 | 16% | 39.7% | 1.76s | 907J | 0.00 |
| ALL-5 | **ACC-v3** | 0.6460 | **0%** | **27.3%** | **0.26s** | **71J** | 0.10 |
| COMPETENT-4 | ACC-v2 | 0.6469 | 15% | 36.2% | 1.55s | 790J | 0.00 |
| COMPETENT-4 | **ACC-v3** | 0.6455 | **0%** | **23.9%** | **0.23s** | **55J** | 0.10 |

- **ALL-6 (where think is genuinely needed): ACC-v3 beats BOTH baselines** — latency 2.61s→2.13s
  (−18% vs v1), energy −20%, +accuracy, and reaches parity 20/20 seeds (ACC-v2 only 19/20).
- **Perception pools: think is unnecessary** — ACC-v3's confidence gate drives think→0%, collapsing
  to ACC-v1 and crushing ACC-v2's over-thinking (1.55s→0.23s, energy 790→55J on COMPETENT-4).
- Cross-family: Lingshu ALL-6 ACC-v3 cuts FLOPs 77.8%→48.6% (think fast → no latency win); QoQ degenerate (no-op).
- Small caveat: guard 0.10 on perception pools (one benchmark occasionally dips below always-small).
- **Novelty: INCREMENTAL.** Confidence gating (CAR, arXiv 2505.15154) + agreement gating (ABC,
  2407.02348) are both prior art; ACC-v3 is their AND-combination. A real engineering improvement to
  the baseline, not a new primitive.

Detailed per-family / per-dataset tables: `DETAILED_TABLES.md`, `paper/figs/rescue/table_*.png`.

---

## 2. ACC-v4 — resolution-decoupled think tier (NOVEL composition; marginal cascade gain)

**Finding (offline, `acc_v4_lowres_think.py` premise):** medical-VLM REASONING is
resolution-insensitive; PERCEPTION is resolution-sensitive. Running the 32B think pass at cap320 vs
fullres:
- Reasoning benchmarks (the only ones reaching the think tier): MMMU **0.688→0.712 (+0.024)**,
  MedXpert ≈ unchanged.
- Perception benchmarks: SLAKE 0.764→0.721 (drops) — but perception is handled by the no-think tiers,
  so this never costs the cascade.

**Method (ACC-v4 = ACC-v3 + think@cap320):** run the expensive reasoning tier at low resolution
(−28% think prefill). Honest calib/test: **ACC-v4 ≈ ACC-v3** at the cascade level — because the think
tier fires rarely (≤14%) and is decode-bound (≈305 tokens), the ~28% prefill saving does not aggregate.
Per-tier the saving is real and never-worse; cascade-aggregate it is negligible.

**Novelty (independent check, IDs verified):**
- (i) the dissociation finding: **INCREMENTAL** — the pattern exists on the visual-token-count axis
  (Matryoshka M3 2405.17430, MQT 2405.19315; Inference-Optimal VLMs 2411.03312), general domain. New
  wrinkles: the **resolution** axis (max_pixels), the **medical** domain, and MMMU *improving* at low res.
- (ii) running the reasoning tier at low resolution as an efficiency lever: **NOVEL-to-incremental** —
  no defeater; VisionThink (2507.13348) does the opposite (escalates resolution within one model).
- (iii) inside a multi-tier medical-VLM cascade: **NOVEL** — no prior work composes (size cascade) ×
  (resolution-decoupled reasoning tier) × (medical).
- Most-threatening prior art: M3 (token-count dissociation), VisionThink (resolution axis, opposite direction).

**Honest status:** ACC-v4 is a *novel composition* but does NOT show a clear cascade-level improvement
over ACC-v3 in honest validation. The novel part (resolution-decoupled reasoning tier) is real per-tier
but marginal in aggregate because the think tier fires rarely and is decode-bound.

---

## 3. Gate-design data size (`gate_data_size.py`) — answered

Does using more of the 337k PMC-VQA-train set (vs the 3000 sample) for gate design change the result?
Subsample-convergence of the ACC-v3 thresholds + eval behaviour on competent-4 (20 subsamples/size):

| #calib | eval acc | eval esc0 | eval think | tau0 σ |
|---:|---:|---:|---:|---:|
| 250 | 0.6429 | 38% | 3% | 0.95 |
| 1000 | 0.6490 | 51% | 3% | 0.69 |
| 2000 | 0.6503 | 59% | 1% | 0.54 |
| 3000 | 0.6502 | 57% | 0% | 0.00 |

**No — more data would NOT change the result.** Eval accuracy is flat from 2000→3000 (0.6503→0.6502);
the gate has converged (a scalar threshold needs few samples). The real limitation is **distribution,
not size**: PMC-VQA-train is perception-only (one benchmark), so it cannot calibrate the think tier for
reasoning (τ1→0 disables think on PMC-train). A larger PMC sample cannot fix that — a *mixed* (reasoning-
containing) calibration set would.

---

## 4. Negatives this round
- **Adaptive think-LENGTH / early-exit reasoning: DEAD.** The 32B commits to its answer at the very
  end of the trace (answer-marker at median 0.99 of trace length, 8220 traces) — truncation cuts off
  before the answer. The reasoning is load-bearing to the end; no free decode saving.
- (Prior) Visual-stability rescue does not improve ACC-v2 (`RESCUE_INTO_ACCV2.md`).

---

## 5. Honest overall verdict
The genuine, validated improvement to the baseline is **ACC-v3** (incremental novelty). The most
**novel** idea (resolution-decoupled cascade, ACC-v4) is verified novel-in-composition but **does not
yield a clear cascade-level win**. A method that is simultaneously (novel) AND (a clear cascade
improvement) AND (validated across families) was not found — consistent with this project's repeatedly
documented conclusion that the medical-VLM cascade is at a genuine efficiency frontier and the gate is
signal-limited (recoverability ceiling ~0.6). A fundamentally novel winning method would require a
different setting (new task/data/model pair), not a new signal on this one.

Reproduce: `acc_v3_confgate.py --family {medvlthinker,lingshu,qoq}`, `acc_v4_lowres_think.py`,
`gate_data_size.py`, `make_detailed_table.py`. Artifacts in `results/cascade_methods/rescue_allfam/`
and `paper/figs/rescue/`.
