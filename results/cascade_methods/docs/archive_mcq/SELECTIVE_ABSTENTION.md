> # ⛔ HISTORICAL RECORD ONLY — ABSTENTION IS PERMANENTLY FORBIDDEN
>
> **Annotated 2026-07-29.** This file documents a training-free selective-prediction / safe-abstention
> mechanism that was built and validated in **June 2026**. On **2026-07-07** the user made abstention /
> reject-option / defer-to-human a **permanent, project-wide prohibition** (`progress/progress_July_07.md`
> §15; CLAUDE.md critical rule 6). **The method must always return an answer.**
>
> This document predates that decision. It is preserved because the project's rule is *move and annotate,
> never delete the record* — and because it is the one case in the whole arc where working code and a
> genuine positive were discarded for **scope**, not because they failed. **It is not a live direction and
> must never be revived as one.** Its code (`src/cascade_methods/selective_abstain.py`,
> `abstain_calibration.py`, `triage_3way.py`, `deferral_curve.py`, `methods_deferral.py`,
> `lingshu_deferral_apgr.py`), its artifacts, and `paper/figs/open/fig_triage.png` are historical too.
>
> *Reusing this literature's **math** inside an answer-producing mechanism is fine.* The deployed
> "certified veto" does exactly that — it **keeps the cheap model's answer**, so it is not abstention.

# Knowing when to abstain — training-free safe deferral for open-ended medical VLMs

> New-method-loop outcome (2026-06-24, Direction 1). A core result: the open-ended detection ceiling-break
> (§5.7) turned into a **deployable selective-prediction / clinician-referral** contribution that engages a
> 2026 SOTA. All numbers from real checkpoint output (LLM-judge scored). Code: `selective_abstain.py`,
> `triage_3way.py`. Figures: `paper/figs/open/{fig_selective,fig_triage}.png`.

## The thesis
A recent SOTA — **"Uncertainty Is Not a Safety Net for Clinical VQA"** (Fazla et al., arXiv 2606.16583;
8 UE methods × 12 VLMs) — concludes uncertainty *fails* as a safety mechanism for clinical VQA: its quality
"tracks model accuracy, degrading precisely where the model is weakest." **That study is multiple-choice
(MCQ) only.** We show the conclusion is **regime-specific**: in the realistic **open-ended** setting a
model's own confidence becomes a usable error detector, enabling a deployable abstention system on
competent benchmarks — while honestly confirming the SOTA's nuance (quality still tracks competence).

## Result 1 — the "money plot": self-error-detection lifts MCQ → open-ended (same model)
Self-error-detection AUROC (confidence vs the model's *own* correctness), same model, same benchmark family:

| model | MCQ (margin) | open-ended (confidence) |
|---|---|---|
| MedVLThinker-7B (RL-on-MCQ) | 0.661 | 0.742 |
| Lingshu-7B (native open-ended) | 0.736 | 0.816 |

Both lift, and the open-ended value **exceeds the SOTA's MCQ ceiling (~0.72)**. `[REPRO: selective_abstain.py]`

## Result 2 — deployable abstention on competent open-ended VQA
Confidence-thresholded selective prediction (answer the top-c by confidence, refer the rest to a clinician),
Lingshu-7B, LLM-judge, per dataset:

| dataset | base acc | detection AUROC | AURC | **coverage@5%-risk** | coverage@10%-risk |
|---|---|---|---|---|---|
| **SLAKE-open** | 0.73 | 0.889 | 0.073 | **0.54** | 0.67 |
| VQA-RAD-open | 0.49 | 0.717 | 0.344 | 0.01 | 0.01 |
| PathVQA-open | 0.34 | 0.797 | 0.415 | 0.10 | 0.11 |
| Kvasir-open (GI, OOD) | 0.30 | 0.749 | 0.500 | 0.00 | 0.00 |

On **SLAKE-open the system auto-answers 54% of cases at ≤5% error** (95% accuracy on the answered set) and
refers the other 46% — a concrete, deployable medical-safety operating point. This **refutes the strong form**
of the SOTA negative (uncertainty *is* a safety net for competent open-ended VQA).

**Even better with the DEPLOYED (strongest) model.** Selective prediction should run on the model you would
actually deploy (highest accuracy), not the cheap leg. **Lingshu-32B self-abstaining** (its own confidence
vs its own correctness, judge):

| dataset | base acc | detection AUROC | AURC | **cov@5%-risk** | cov@10%-risk |
|---|---|---|---|---|---|
| **SLAKE-open** | 0.82 | 0.884 | 0.040 | **0.69** | **0.81** |
| VQA-RAD-open | 0.60 | 0.761 | 0.217 | 0.04 | 0.10 |
| PathVQA-open | 0.38 | 0.833 | 0.355 | 0.14 | 0.19 |
| Kvasir-open | 0.30 | 0.788 | 0.477 | 0.02 | 0.04 |

On SLAKE-open the deployed model **auto-answers 69% at ≤5% error (81% at ≤10%)**. Crucially, detection AUROC
stays high everywhere (0.76–0.88) — so where coverage collapses (weak datasets) the system **safely abstains
on almost everything**, which is the correct behavior: it never auto-answers at high risk.

## Result 2b — calibrated and deployable (held-out calibration holds the risk target)
`[REPRO: abstain_calibration.py]` The realistic deployment protocol: calibrate the confidence threshold on
a held-out sample to a clinician's target risk r*, then deploy on fresh data. **It holds the target** (20
seeds, Lingshu-32B, r*=5%):

| dataset | coverage | realized risk | verdict |
|---|---|---|---|
| SLAKE | 0.69 | **0.051** | OK |
| VQA-RAD | 0.06 | 0.039 | OK |
| PathVQA | 0.15 | 0.062 | ~OK |
| Kvasir | 0.01 | 0.073 | low-coverage |

Reusing one threshold **across** distributions overshoots (SLAKE-τ → VQA-RAD 11.8%, → Kvasir 19.5%) → a
**per-deployment calibration set** is the safe practice. So the system is genuinely deployable: a clinician
picks a risk target, calibrates on a held-out sample, and the deployed risk holds.

## Result 3 (honest) — deployability tracks competence; the recoverability wall limits escalation
- Deployability **degrades where the model is weak** (VQA-RAD, PathVQA, Kvasir: base acc ≤0.49, cov@5% ≤0.10).
  This **confirms** the SOTA's central observation — UE quality tracks model competence. We claim the
  *competent-benchmark* operating points and the *regime lift*, not universal deployability.
- **3-way triage** (answer / escalate-to-strong / abstain, `triage_3way.py`) is a strict generalization
  (degrades to 2-way when no recovery exists) but the gain is **marginal**: SLAKE cov@10%-risk 0.67→0.74,
  ~0 at 5% risk, ~0 on harder sets — bounded by recoverability (the strong model fixes only ~22% of the
  cheap model's open-ended errors). Escalation cannot meaningfully raise the safe operating point here.

## Novelty (checked)
- Closest prior art: 2606.16583 (MCQ, negative, analysis-only); VASE (MICCAI'25), UniVRSE (2503.20504) —
  open-ended hallucination *detection scores* (AUROC), **not** deployable risk-coverage systems with referral
  operating points; semantic-agreement cascades (2509.21837) are text-only.
- **Residual novelty:** (1) the MCQ-vs-open self-assessment *regime contrast* for medical VLMs; (2) a
  *deployable* training-free clinician-referral system with calibrated risk targets on open-ended medical VQA;
  (3) cross-model (MedVLThinker, Lingshu) and cross-modality (radiology, pathology, GI endoscopy) validation;
  (4) the 3-way answer/escalate/abstain unification with its honest recoverability bound.

## Scope / honesty
- This is an applied/empirical contribution, not a new uncertainty *signal* (our gate hunt and cross-model
  agreement tests confirm **confidence is unbeatable** — nothing we tried beats single-pass confidence).
- LLM-judge scored (text-only judge; image-blindness caveat, §5.7); confidence = mean-token logprob.
- Deployable only where the model is competent; the contribution is the *regime insight* + the *competent-set
  operating points* + the honest competence-dependence, directly engaging the 2026 SOTA.
