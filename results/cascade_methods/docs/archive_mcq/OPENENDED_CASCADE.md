# Open-ended medical VQA cascade (new setting) — self-consistency signal works; cascade does not

**Date:** 2026-06-24 · **Setting change** (user: "change the setting for real novelty; GPU free").
Moved off MCQ — where every routing signal saturates at ~0.6 AUROC because single-letter answers make
confidence/agreement/self-consistency degenerate — to **open-ended (generative) medical VQA**, where a
free-text answer makes **semantic self-consistency** a live signal. Datasets: SLAKE-open (645) +
VQA-RAD-open (200), short-answer, normalized exact-match + contains scoring. (PathVQA-open dropped:
long descriptive answers give exact-match acc 0.058 — unscoreable.) Models: MedVLThinker 7B/32B.
Inference: `run_openvqa.py` (7B confidence temp-0; 7B self-consistency temp-0.7 K=8; 32B-no-think &
32B-think strong). Analysis: `open_cascade_analyze.py`.

---

## 1. The genuinely novel POSITIVE: self-consistency breaks the MCQ error-detection ceiling

AUROC for predicting **the cheap 7B is WRONG** (pooled n=845):

| signal | AUROC (cheap-wrong) | note |
|---|---:|---|
| confidence (7B seq-logprob; Chow/Jitkrittum-style) | 0.735 | the validated baseline |
| **self-consistency (K=8 agreement)** | **0.781** | Wang et al. ICLR'23 signal |
| **answer-diversity (n_distinct over K)** | **0.788** | VQA-RAD up to 0.830 |

- **All are far above the ~0.6 AUROC ceiling that every signal hit in the MCQ setting** (where the
  project's prior gate work — margin/MSP/entropy/conformal/self-verify/agreement — saturated). The
  open-ended setting *does* break the signal wall: answer-set diversity is a strong error detector.
- **K-curve** (cost scales with K): self-consistency AUROC by #samples — K=2:0.68, K=3:0.72, **K=5:0.76**,
  K=8:0.78. K=5 already beats confidence (0.735), so the signal is obtainable at ~5× cheap-leg cost.
- This is a real, novel **selective-prediction / reliability** result for medical VLMs: *answer
  self-consistency is a substantially better "when to trust the small model" signal than sequence
  confidence, and is degenerate in the MCQ benchmarks where medical-VLM cascades are usually evaluated.*

---

## 2. The honest NEGATIVE: the cascade does not win (small, unroutable model-gap)

The cascade needs a strong model that is **reliably and predictably** better than the cheap one. On
open-ended medical VQA the MedVLThinker 32B is **barely** better than the 7B:

| | 7B-nt | 32B-no-think | 32B-think |
|---|---:|---:|---:|
| SLAKE-open acc | 0.419 | 0.498 | 0.453 |
| VQA-RAD-open acc | 0.370 | 0.475 | 0.375 |
| **pooled acc** | **0.407** | 0.492 | 0.434 |

- The 7B→32B gap is **+0.03 to +0.08** (think even *hurts* vs no-think — over-thinking persists in
  open-ended perception). recoverable (7B-wrong ∧ 32B-right) = only **0.11**.
- **Recoverability AUROC ≈ 0.49–0.53 (chance) for every signal**, including the strong
  self-consistency signal. The two models **disagree on 68% of SLAKE** yet have near-equal accuracy —
  so on disagreements *neither is reliably better*. Real complementarity exists (oracle cascade ceiling
  ≈ 0.516 vs always-32B 0.434) but it is **unroutable** — exactly the wall the MCQ work documented.
- **Cost frontier** (strong leg = 32B-think): at matched accuracy self-consistency gives only a small
  latency edge (midpoint 4.13 s vs confidence 4.57 s) at **~2.5× the FLOPs** (K=8 cheap passes:
  60 vs 25 PFLOP). Not a clean cascade win.

Scoring caveat: spot-checks show some synonym deflation ("thorax" vs "Chest"), but most 32B errors are
genuine ("Heart" vs "Lung"); an LLM-judge would raise absolute accuracies but not change the verdict
(the gap stays small and recoverability stays unroutable).

---

## 2b. The big-gap test (3B cheap leg) — there is NO gap at any size

To rule out "the 7B/32B are just both strong," we tried the legacy **MedVLThinker-3B** as a deliberately
weak cheap leg (`run_openvqa_3b.sh`). It is *not* weak: SLAKE-open **3B 0.457 ≈ 7B 0.419 ≈ 32B-think 0.453
≈ 32B-no-think 0.498**. **Token-F1 (partial-credit, rigor check ruling out exact-match compression)
confirms it:** 3B-F1 0.423 vs 32B-think-F1 0.390 (gap **−0.033**, think hurts) vs 32B-no-think-F1 0.457
(gap **+0.034**). So model size barely affects open-ended medical-VQA accuracy for this family, under
*both* scorers. recoverability AUROC stays ~0.54–0.59 (chance). The self-consistency error-detector stays
strong on the 3B too (n_distinct AUROC 0.810). There is simply no routable gap to build a cascade on.

## 2c. Cross-family big gap (Lingshu-32B strong leg) — the cascade WORKS, and the MCQ ceiling breaks

Pairing MedVLThinker-7B (cheap) with a genuinely stronger open-ended model, **Lingshu-32B**, finally
gives a real gap: 7B 0.407 → Lingshu-32B **0.775** (token-F1 0.376→0.789), recoverable **0.385**.
With a real gap, routing matters and the signals are strong. Two regimes:

| cascade | cheap | strong | conf AUROC cheap-wrong | conf AUROC recover | SC AUROC cheap-wrong | SC AUROC recover |
|---|---:|---:|---:|---:|---:|---:|
| **MedVLThinker-7B → Lingshu-32B** (miscalibrated cheap) | 0.407 | 0.775 | 0.735 | 0.575 | **0.781** | **0.618** |
| **Lingshu-7B → Lingshu-32B** (calibrated cheap, same-family) | 0.683 | 0.775 | **0.866** | **0.804** | 0.845 | 0.764 |

- **ROBUST finding: the routing-signal ceiling is an MCQ ARTIFACT.** In open-ended medical VQA, routing
  AUROC reaches **0.80–0.87** (Lingshu-7B confidence) vs the **~0.6 ceiling** every signal hit on MCQ
  (the project's "gate is saturated", §5.2). The discreteness — *not* answer length — is the cause:
  our open answers are median **1–2 tokens** (as short as a letter) yet routing AUROC is ~0.87, because
  the answer *space* is open, not 4 fixed options. So confidence-gated open-ended medical-VLM cascades
  genuinely work; the MCQ saturation does not transfer.
- **CONDITIONAL finding: self-consistency beats confidence ONLY for a miscalibrated cheap model**
  (MedVLThinker-7B, MCQ-RL-tuned → poorly calibrated on free-text: SC 0.781 > conf 0.735, recoverability
  +0.043 bootstrap CI [0.016,0.069]). For a natively open-ended, well-calibrated cheap model (Lingshu-7B),
  **confidence is the better gate** (0.866 vs 0.845). So self-consistency is a *rescue for miscalibration*,
  not a general improvement over confidence. (Honest correction: an earlier read overstated this as a
  general win.)

## 3. Verdict (revised)
**A genuinely-novel WINNING cascade method was not found in the open-ended setting either** — not because
of MCQ degeneracy this time, but because the **MedVLThinker 7B and 32B are nearly equivalent on
open-ended medical VQA** (small, unroutable gap). The cascade premise (a reliably-better strong model)
is not met by this model pair, in MCQ *or* open-ended.

**The robust, publishable finding (§2c):** the medical-VLM routing-signal **ceiling is an MCQ artifact**.
With a real gap (Lingshu-7B→Lingshu-32B), open-ended routing AUROC is **0.80–0.87** vs the **~0.6** MCQ
ceiling — because the MCQ degeneracy is from **4-option discreteness, not answer length** (open answers are
median 1–2 tokens, yet routing AUROC ~0.87). So **confidence-gated open-ended medical-VLM cascades work**,
reframing the project's "gate is saturated" result as a benchmark artifact. Novelty (independent check):
the agreement-as-gate idea and open-ended cascades are prior art (semantic-agreement cascade arXiv
2509.21837, EMNLP'25; ABC 2407.02348; Jitkrittum NeurIPS'23); the genuinely unoccupied cell is
**medical + VLM + open-ended cascade** + the MCQ-ceiling-is-discreteness point — publishable as an applied
contribution (CVGIP), not a new gate primitive.

**Honest negative on the novel-gate goal:** **self-consistency does NOT robustly beat confidence** — it
only wins for a *miscalibrated* cheap model (MedVLThinker-7B, MCQ-tuned on free-text); for a natively
calibrated cheap model (Lingshu-7B) plain confidence is the better gate (0.866 vs 0.845). So a new gate
that beats confidence was not found; the contribution is the *setting* (open-ended unlocks routing), with
the confidence gate (known) as the winner and self-consistency as a calibration rescue.

Reproduce: `run_openvqa_all.sh`, `run_openvqa_think.sh`, `run_openvqa_3b.sh`, `run_openvqa_lingshu.sh`,
`run_openvqa_lingshu7b.sh`, `open_cascade_analyze.py [--think|--cheap3b|--cheap_l7|--lingshu]`,
`make_open_chart.py`. Artifacts: `ckpts/openvqa/` (gitignored), `results/cascade_methods/open_cascade*.json`,
`paper/figs/open/`.
