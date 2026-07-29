# New-direction phase (2026-06-25) — trained verifier + the structured-output pivot

> Per user mandate: consolidate (done, §5.9), try a TRAINED method, and hunt new datasets/tasks — each
> with a published HIGH-TIER baseline, all documented. This doc tracks the trained verifier and the
> structured-output pivot. Lit baselines verified by a research agent (real arXiv IDs).

## Why a pivot, and to what
Seven training-free mechanisms for an open-ended *free-text* accuracy win are luck/capacity-bound
([[openended-selection-luckfloor]], [[recovery-is-capacity-bound]]). The lit scan surfaced the escape:
**"Verification Mirage" (arXiv:2605.10850, 2026, VERIFIED)** shows self-verification fails on medical VQA
(false-acceptance 60–100%, "lazy verifier" ignores the image) **EXCEPT on quantitative/measurable tasks.**
That is exactly our luck-floor result — and it says the way out is **structured/verifiable outputs**
(bounding-box IoU, progression labels, measurements), where a verifier/selection signal has real grounding.

## Track B — trained verifier for open-ended best-of-N (running)
LoRA-fine-tune Lingshu-7B to score P(correct | image, question, free-text answer) on the 8,234 per-sample
LLM-judge labels; select best-of-8. Honest grouped split by question. `run_lora_verifier_open.py`.
**Published baselines to beat / cite (all VERIFIED):**
- Method class — **GenRM: Generative Verifiers** (arXiv:2408.15240, **ICLR 2025**): trained generative
  verifier for best-of-N; our LoRA verifier is an instance.
- Free baselines — **P(True)** (Kadavath, arXiv:2207.05221) and training-free **Self-Certainty BoN**
  (arXiv:2502.18581, **NeurIPS 2025**). Our zero-shot 32B verifier (0.758) ≈ the P(True) baseline.
- Closest medical artifact — **Med-RewardBench** (arXiv:2508.21430, 2025): first multimodal-medical
  reward-model/judge benchmark; shows current medical judges are weak (motivation). **Med-PRM**
  (arXiv:2506.11474, **EMNLP 2025 Oral**) is the strongest medical verifier but text-only + RAG.
- **Warning:** Verification Mirage predicts this will *fail* on free-text. Either result is publishable —
  a win beats the luck floor; a loss confirms the verifiability boundary with a *trained* (not just
  zero-shot) verifier. Target: beat zero-shot 32B-verify (0.758) and approach oracle@8 (0.879).

## Track C — the structured-output pivot: medical visual GROUNDING (the real shot)
**Hypothesis:** selection/self-consistency escapes the luck floor when the output is verifiable. Test on
medical grounding (bounding boxes, IoU). **Zero download:** SLAKE ships per-image `detection.json` gold
boxes (580/642 images; organs + abnormalities). `run_ground_slake.py` generates 1 greedy + 8 sampled boxes
from Lingshu-7B (which *can* ground — boxes organs roughly right, fails on tiny lesions; emits mixed
normalized/absolute coords, parseable). **Key experiment:** does *spatial self-consistency* (mean pairwise
IoU among the 8 boxes) predict box correctness (IoU vs gold)? If AUROC ≫ 0.6 and medoid-selection beats
greedy toward oracle, structured outputs escape the luck floor — a novel positive.
**Published baseline (VERIFIED):** **MedGround-R1** (arXiv:2507.02994, 2025) — SOTA medical phrase
grounding (MS-CXR mIoU 79.02). Standard benchmark MS-CXR is PhysioNet-gated; SLAKE-detection is our
zero-gate proxy. Training-free best-of-N bbox selection with an IoU/consistency verifier is **under-explored**
(MedGround-R1 is RL-trained), so a training-free selection result is a clean novelty cell. Adjacent
test-time work: CoTBox-TTT (arXiv:2511.12446).

### Grounding RESULT (n=1622 SLAKE targets, Lingshu-7B, `run_ground_slake.py` + `ground_analyze.py`)
**The structured-output DETECTION hypothesis is CONFIRMED** — spatial self-consistency (mean pairwise IoU
among 8 sampled boxes) predicts box correctness at **AUROC 0.816 (IoU≥0.5) / 0.774 (IoU≥0.3)**, far above
the free-text luck floor (~0.5–0.55). This is a clean, novel extension of the §5.7 detection result to a
**verifiable** modality, and an independent confirmation of Verification Mirage's "measurable tasks"
boundary. *Prompt format matters:* "Locate the X" beats a generic xyxy prompt 2× (the Qwen `<box>` format
emits resized-space coords); coords are mixed normalized/absolute (parsed by a ≤1.5→normalized heuristic).
**BUT no selection METHOD win:** Lingshu-7B's base grounding is near-floor (greedy acc@0.3=0.098, acc@0.5
=0.022; oracle@8 acc@0.3=0.219), so SC-medoid ≈ greedy — there is no accuracy to harvest because the
*oracle ceiling itself is low*. Same structural lesson as §5.7: **detection ≠ harvestable gain.** Here the
limiter is base *capability* (not recoverability): a selection win needs a competent grounder.

### Competent grounder (Qwen2.5-VL-7B) — the "escape" is an ARTIFACT; luck floor GENERALIZES to grounding
Downloaded base **Qwen2.5-VL-7B-Instruct** (strong grounding) and re-ran (parser fixed to read Qwen's
`{"bbox_2d":[...]}` JSON; an earlier near-zero was a parse bug grabbing the "2" in "bbox_2d"). Qwen DOES
ground SLAKE organs (greedy acc@0.3=0.254, **oracle@8=0.438** — a real +0.18 gap; abnormalities stay hard
~0.05). **But the structured-output "escape" VANISHES:** SC-medoid does NOT beat greedy (organ acc@0.3
0.246 vs 0.254) and **SC-agreement AUROC collapses to 0.557** (chance). The Lingshu AUROC 0.82 was an
**artifact of incompetence** — a weak grounder emits *diverse* boxes so rare agreements spuriously track
correctness; a *competent* grounder emits *consistent* boxes so agreement no longer discriminates, and the
real oracle gap is **luck-floored exactly like free-text**.
**GENERALIZED FINDING (stronger than the original hypothesis):** the generation→selection luck floor is
NOT specific to free-text — it holds for **verifiable/structured outputs (grounding) too, once the model is
competent**. Verification Mirage's "measurable tasks escape" is itself bounded by a tension: competence →
consistency → no discriminative selection signal. **Net: trained-verifier and grounding-selection are both
luck-floored; the strong novel finding is that the luck floor is general across output types, not the
free-text quirk it first appeared to be.** Code: `run_ground_slake.py`, `ground_analyze.py` (bbox_2d parse).

## New datasets identified (VERIFIED, for later)
- **RadImageNet-VQA** (`raidium/RadImageNet-VQA`, arXiv:2512.17396): large clean **open-ended**,
  anti-shortcut (near-random without image); SOTA VLMs struggle especially open-ended. Gated research-use.
- **Kvasir-VQA-x1** (`SimulaMet/Kvasir-VQA-x1`, arXiv:2506.09958, MICCAI): clean open-ended, complexity-
  stratified (a natural cascade-gating axis). (Confirm vs our existing Kvasir usage.)
- Multi-image / change detection — **BioViL-T** (arXiv:2301.04558, **CVPR 2023**) on MS-CXR-T (3-class
  progression); **MedFrameQA** (arXiv:2505.16964, MLLMs <50%). **No published cascade/BoN/selection work**
  in this lane — least crowded, strongest shot at novelty if grounding pans out.

## Avoid (already published)
Radiology report-generation Best-of-N with a frozen-model PRM is **already done**: *Process Reward Models
for Sentence-Level Verification of LVLM Radiology Reports* (arXiv:2510.23217) — report rejection +4.5%
F1-CheXbert, weighted BoN (N=128) +7.4%. Do not pivot there expecting novelty.

## "Try all" — data-access outcomes (2026-06-25)
- **MedFrameQA** (SuhaoYu1020/MedFrameQA, non-gated, downloaded): multi-image (2–5 frames) but **MCQ**
  (options A–E) → the MCQ-saturation regime (§5.7), not the open-ended setting where the verifier wins. ~3000 Q.
- **RadImageNet-VQA** (raidium/RadImageNet-VQA): gated=auto — file list is public but **download 401s**
  (needs an authenticated HF token with accepted terms; a user account action). BLOCKED autonomously.
- **MS-CXR / MS-CXR-T** (microsoft/ms-cxr, StanfordAIMI/ms-cxr-t): **401, PhysioNet-credentialed** — BLOCKED
  without the user's PhysioNet credentials. (Change-detection + grounding standard benchmarks both gated.)
- **Accessible high-value lane → GROUNDING BOX-VERIFIER (zero-download via SLAKE detection boxes):** the
  structured-output analog of the §5.10 answer-verifier positive — does *training* a box-verifier beat the
  SC-medoid luck floor (which §5.9 showed holds for structured outputs)? If yes, a second positive showing
  the trained-verifier approach generalizes from free-text to verifiable outputs. Pursuing this.

## GROUNDING BOX-VERIFIER — SECOND POSITIVE (2026-06-25): training breaks the structured-output luck floor too
`run_lora_box_verifier.py`, SLAKE grounding (Qwen2.5-VL 8 boxes, IoU≥0.3 vs detection.json gold), grouped
split, n=487 held-out. SC-medoid (training-free) **0.164 (below greedy 0.197 — luck-floored)** → **trained
box-verifier 0.255** (oracle@8 0.343) = **+0.058 over greedy / +0.091 over SC-medoid, captures 40% of the
oracle gap** (~3–5 s.e.). This is the structured-output analog of the §5.10 free-text verifier and captures
the SAME ~40% fraction. **UNIFIED FINDING:** training-free selection is luck-floored across BOTH free-text
and verifiable/structured outputs (§5.9), and a TRAINED verifier recovers ~40–50% of the oracle gap in BOTH
— so *training* is the universal active ingredient, not a free-text quirk. Baseline MedGround-R1 (2507.02994).
Added to paper §5.10.

## MS-CXR grounding (real benchmark) — CORRECTED: a 3rd box-verifier POSITIVE (2026-06-26)
> An earlier entry below called this "base-capability-bound negative" — that was a **coordinate-space bug**
> (Qwen2.5-VL emits bbox_2d in its smart-RESIZED space; MS-CXR chest X-rays are large, so resized≠original,
> and I'd compared to original-dim gold). FIXED via qwen_vl_utils.smart_resize scaling (no-op for small SLAKE
> images). Corrected: greedy meanIoU 0.022→0.098, oracle@8 acc@0.3 0.033→0.253 — a real harvestable gap.
> **Box-verifier on real MS-CXR phrase grounding (FULL 1448 boxes, n=435 held-out, IoU≥0.3):** SC-medoid 0.044 (below greedy,
> luck-floored) → greedy 0.074 → **trained box-verifier 0.232** (oracle@8 0.285) = **+0.11 over greedy,
> captures 78% of the oracle gap** (~3.3 s.e.). THIRD confirmation of the §5.10 principle (free-text 49%,
> SLAKE-organ 40%, MS-CXR-pathology 77%) — training breaks the luck floor across output types AND grounding
> domains, now on a real published benchmark. Honest scope: lifts SELECTION over a weak base grounder
> (Qwen2.5-VL greedy 0.074); does NOT beat MedGround-R1's grounding in absolute terms (a better base grounder,
> unreleased). Code: run_ground_mscxr.py, run_lora_box_verifier.py (smart_resize fix), ckpts/train/lora_box_verifier_mscxr.

## [SUPERSEDED by the above] MS-CXR first-pass — base-capability-bound (coord-bug artifact)
PhysioNet credentials work; downloaded MS-CXR 1.1.0 (1448 phrase-grounding boxes) + selectively pulled the
referenced MIMIC-CXR-JPG chest X-rays (PhysioNet throttling truncated many; validated 326+ clean → 451
grounding targets). **Qwen2.5-VL cannot ground chest-X-ray PATHOLOGY** (greedy IoU 0.022, acc@0.3 0.016,
oracle@8 0.033 — near-zero), vs SLAKE ORGAN grounding (Qwen oracle@0.3 0.44). Localizing subtle findings
("Bibasilar opacities", "Pleural Effusion") needs medical expertise a general VLM lacks — which is why the
MS-CXR SOTA **MedGround-R1** (arXiv:2507.02994, mIoU 79) is a *specially trained* grounder. **No oracle gap →
the box-verifier has nothing to harvest here.** So the box-verifier approach (validated on SLAKE organ
grounding, §5.10) needs a *competent base grounder*; for chest-X-ray pathology that requires a trained
grounder (MedGround-R1) as the base — the clear next step if pursued. Result is base-capability-bound,
unchanged by more images. Code: run_ground_mscxr.py.
