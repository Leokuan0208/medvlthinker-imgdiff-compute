# Integrating the Visual-Stability Rescue into ACC-v2 — cross-family result + honest verdict

**Date:** 2026-06-24 · **Method under test:** ACC-v2 (cross-model agreement cascade, "Ours") +
Visual-Stability (resolution-consistency) Rescue · **Families:** MedVLThinker, Lingshu, QoQ (the 3
Qwen2.5-VL families with a `max_pixels` resolution ladder) · Chiron / MedGemma: **N/A** (see §5).
All numbers from real checkpoint output, honest 50/50 calib/test ×20 seeds, parity = always-big-think.

---

## 0. Verdict (honest)

**The Visual-Stability Rescue does NOT improve ACC-v2.** Tested at both natural insertion points,
across all applicable families, with adversarial controls:

- **Rescue on Tier-0** (keep visually-stable low-margin queries on the small model): **fails** — it
  caps accuracy below parity (ceiling 0.641 < parity 0.645 on COMPETENT-4) and adds 3 cap passes to
  save only the *cheap* big-no-think escalation (0.34 s). Wrong tier: in ACC-v2 the expensive tier is
  gated by agreement, not by Tier-0 margin.
- **Rescue on the Think tier** (skip the 28 s think pass when the small answer is resolution-stable):
  *appears* to win (MedVLThinker COMPETENT-4 latency 1.55→1.01 s at parity) — **but the control
  proves the resolution signal is not responsible.** The win is generic *think→big-no-think
  rebalancing*, and resolution-stability is a **worse** think-skip selector than random or
  confidence (§3).

**Why:** ACC-v2's large-no-think intermediate tier already structurally captures the benefit the
rescue provided in the simpler 2-tier deployed cascade (where it corrected an over-escalating frozen
gate). The rescue is **redundant** with ACC-v2's structure. This is a clean negative — and a useful
one: it shows ACC-v2 is not improved by the orthogonal robustness signal.

**What the investigation DID surface** (a real, if incremental, lever): tightening ACC-v2's
think-gate with **big-no-think confidence** — think only when the two no-think models disagree *and*
the big no-think model is itself unsure — reaches parity at **0.60 s vs ACC-v2's 1.70 s (−65%)** on
ALL-5. That is standard confidence gating (overlaps ACC-v1's margin think-gate), not the novel
resolution signal; offered as a follow-up, not claimed as validated (in-sample frontier, §3).

---

## 1. How the rescue was integrated

3 compute tiers (stop at first accepted): **small-nt@cap320 → big-nt@cap320 → big-think@fullres.**
- ACC-v2 gate: Tier-0 escalate if small margin < τ0; Tier-1→think iff small-nt ≠ big-nt (agreement).
- **rescue@tier0:** Tier-0 escalate iff (margin<τ0) **and** small-nt answer is NOT resolution-stable
  (same letter across caps {80,160,320,640}); 3 extra small passes charged only on the would-escalate set.
- **rescue@think:** think iff (disagree) **and** small-nt is NOT resolution-stable; extra passes charged
  only on the think-candidate (disagree) set.
Cost: FLOPs=2N(P+G); latency/energy = measured batch-1 `a+b·gen` per tier. Code: `acc_rescue_allfam.py`.

---

## 2. Honest calib/test at big-think parity (per-dataset + ALL-5/6 + cost)

### MedVLThinker (the only family where think is both slow AND fires)

| pool | method | acc | esc0 | think | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| COMPETENT-4 | ACC-v2 | 0.6469 | 32% | 15% | 36.3% | 1.55 | 792 | 0.00 |
| COMPETENT-4 | + rescue@think | 0.6466 | 36% | 8% | 35.2% | **1.01** | **467** | 0.10 |
| ALL-5 | ACC-v2 | 0.6485 | 36% | 16% | 39.6% | 1.75 | 904 | 0.00 |
| ALL-5 | + rescue@think | 0.6468 | 42% | 8% | 38.5% | **1.11** | **513** | 0.05 |
| ALL-6 | ACC-v2 | reaches parity 19/20 seeds | | | | | | |
| ALL-6 | + rescue@think | **cannot reach parity** (MedXpert reasoning genuinely needs think) | | | | | | |

rescue@tier0 reaches parity in 0–3/20 seeds (caps accuracy) — excluded.

Per-benchmark accuracy @ parity (COMPETENT-4): ACC-v2 PMC .553 / SLAKE .791 / VQA-RAD .831 / PathVQA
.670; +rescue@think .553 / .795 / .833 / .669 (≈equal). Caveats: small guardrail cost (0.05–0.10 vs
0.00) and **ALL-6 unreachable**.

### Lingshu (think is FAST — gen≈3, doesn't truly reason)
ALL-6: ACC-v2 think24% FLOPs77.7% 0.38s 107J → +rescue@think think9% FLOPs65.4% **0.40s** 99J. FLOPs
& energy fall, **latency does not** (cheap think). guard 1.0 on ALL-6 inherited from ACC-v2.
COMPETENT-4: FLOPs 64.0→56.0%, energy 91.6→85.8J, latency 0.34→0.35s.

### QoQ (degenerate cascade)
always-small-nt ≥ big-think parity → ACC-v2 escalates **0%**; all variants identical (8.8% FLOPs,
0.12 s). Rescue correctly a **no-op**.

---

## 3. The control that settles it — resolution-stability is a poor think-skip signal

`control_think_signals.py`, MedVLThinker, min-LATENCY at big-think parity (in-sample). Each signal
skips the *same fraction* of think-candidates; the question is *which* to skip.

| think-skip signal | ALL-5 lat | COMPETENT-4 lat |
|---|---:|---:|
| none (ACC-v2) | 1.70 s | 1.50 s |
| **resolution-stability (ours)** | **1.07 s** | **0.95 s** |
| random (matched rate) | 0.72 s | 0.65 s |
| big-nt confidence | **0.60 s** | **0.58 s** |
| inverse-stability | 0.93 s | 0.90 s |

Resolution-stability is the **worst** real signal — random, confidence, and even *inverse*-stability
reach parity at lower latency. Mechanism (`think_rescue_mechanism.py`): among think-candidates,
resolution-**stable** ones benefit *more* from think (+0.058) than unstable ones (+0.039), so the
rescue skips the *higher*-value think calls. The apparent rescue@think gain is entirely the
think→big-nt rebalance, done better by any other selector. Chart: `paper/figs/rescue/control_think_signals.png`.

---

## 4. Cross-family summary

| family | think slow? | cascade non-degenerate? | rescue@think effect |
|---|---|---|---|
| MedVLThinker | yes (28 s) | yes | latency↓ at parity, **but signal is suboptimal** (control); guardrail cost; ALL-6 N/A |
| Lingshu | no (≈0.4 s) | partial (guardrail issues) | FLOPs/energy↓ modest, **no latency win** |
| QoQ | n/a | **no (0% escalation)** | no-op |

The effect tracks where think is slow *and* fires — never because of the resolution signal itself.

---

## 5. Chiron (InternVL3) and MedGemma (Gemma3) — why N/A

The rescue needs a **resolution ladder** on the small model. Both lack a `max_pixels` knob:
InternVL3 controls resolution by **dynamic tiling** (1–12 tiles), and Gemma3/MedGemma uses a **fixed**
896² input (pan-and-scan). There is no smooth cap80/160/320/640 ladder to perturb, so multi-resolution
answer-agreement is not defined for them, and their small-model checkpoints exist at one resolution
only. A **model-agnostic** generalization (perturb the *input image* directly — downscale the pixels,
independent of the model's internal handling) would apply to any VLM but needs new inference; given §3
(the signal is dominated even where it *is* defined), this was not pursued.

---

## 6. Novelty (independent literature check, 14 arXiv IDs verified)

- **Route-DOWN resolution-stability rescue** (suppress over-escalation): **NOVEL** in the abstract —
  no prior work uses multi-resolution answer-invariance to *keep a query cheap*. Opposite direction
  from **VisionThink (2507.13348)** which escalates resolution. But (this report) it does **not improve
  ACC-v2**, so the novelty is moot for *this* method.
- **Cross-model agreement think-gate** (ACC-v2 itself): **not novel** — Agreement-Based Cascading
  (**2407.02348**), and semantic-agreement LLM cascades (**2509.21837**).
- **Invariance-as-confidence**: **not novel** — Bahat & Shakhnarovich (**1804.00657 / 2006.16705**).
Closest threats to differentiate against: VisionThink (direction), Bahat & Shakhnarovich (principle).

---

## 7. Reproduce / artifacts

```bash
python3 src/cascade_methods/acc_rescue_allfam.py --family medvlthinker   # (also lingshu, qoq)
python3 src/cascade_methods/think_rescue_mechanism.py                    # why the think signal is wrong-way
python3 src/cascade_methods/control_think_signals.py                     # the decisive control
python3 src/cascade_methods/make_rescue_charts.py                        # 4 charts -> paper/figs/rescue/
```
JSON: `results/cascade_methods/rescue_allfam/{medvlthinker,lingshu,qoq,control_think_signals}.json`.
Charts: `paper/figs/rescue/{control_think_signals,medvlthinker_frontier,crossfamily_parity_cost,medvlthinker_perdataset}.png`.
