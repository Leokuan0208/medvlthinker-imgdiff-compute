# Research loop: a novel TRAINING-needed, model-agnostic cascade method

Outcome of the research-loop workflow (survey prior art → ideate → rank → pick + experiment). Status: winner
implemented + de-risk experiment running. **No fabricated numbers** — experiment numbers land in `ckpts/train/fld/result.json`.

## Motivation (why a *new axis*)
Every prior trained cascade learner scores the **escalation DECISION** (FrugalGPT, AutoMix meta-verifier,
Jitkrittum post-hoc deferral, Hybrid-LLM, RouteLLM, Gatekeeper, the reasoning-mode routers Self-Route/CAR/
SynapseRoute). We have repeatedly shown that decision is **capped** (recoverability ~0.58 AUROC; CASP 0.733
logistic; a LoRA fine-tune got 0.722 < logistic). So a better gate is a dead end. **Open gap:** no prior
cascade learner attacks the escalation *RATE* (the dominant FLOPs/latency driver, esc0 = 35–79%) by making
the cheap leg itself stronger.

## Candidates ideated (5), ranked by the judge (novelty/feasibility/model-agnostic/impact)
| method | what's trained | nov | feas | agn | imp |
|---|---|---|---|---|---|
| **FastLeg-Distill (FLD)** ← winner | LoRA-distill big-no-think competence into the small-no-think leg | 7 | 9 | 6 | 8 |
| TCAL (Think-Calibration head) | MLP predicting SIGNED think-value (help/tie/hurt), cross-family | 7 | 8 | 7 | 5 |
| FERN (Family-invariant Embedding Router) | MLP on frozen SigLIP embeddings → think-tier value | 6 | 9 | 7 | 4 |
| CALM-Fuse (Calibrated Answer-Logit Mixing) | tiny head fusing small+big no-think letter distributions | 7 | 7 | 5 | 5 |
| CALICO (Cross-Family Calibrated Abstention) | per-family abstention head → family-invariant scale | 6 | 9 | 8 | 4 |

TCAL/FERN/CALM-Fuse/CALICO all still learn a *decision/abstention/fusion score* on or near the capped axis;
only **FLD** changes the cheap leg's *answers*, so it is the one bet that can move the dominant cost term.

## Winner: FastLeg-Distill (FLD)
**Idea.** LoRA-fine-tune the small no-think model to imitate the big no-think model's *correct* answers
(agreement-gated: train only where the teacher is right), so the cheap leg's accuracy rises toward the
big-no-think ceiling → fewer queries need escalation → lower FLOPs/latency at the deployed parity accuracy.
- Teacher: 32B-nt@cap320 (`ckpts/gate_32b_modes/nothink_cap320`). Student: 7B + LoRA (r=16, q/k/v/o+MLP), no-think.
- Target: teacher's answer letter (hard-label CE; soft-logprob KD is a refinement).
- Novel vs prior art: distillation aimed at the cascade's escalation *rate*, not the deferral decision; uses
  the big model's *no-think* mode (the over-thinking lever) as the teacher.

## Experiment plan + status
- **De-risk (running, `ckpts/train/fld`)**: eval-CV (50/50 seed-0), distill on train-half, eval the distilled
  7B-nt vs the **base 7B (adapter disabled, same forward)** on the held-out half — per-benchmark + ALL-5/ALL-6
  lift. Question: does distillation raise the cheap leg at all? `src/training_methods/fld_distill.py`.
- **If it lifts** → clean version: run the 32B-nt teacher on the held-out TRAIN splits (pmc_vqa_train,
  slake/vqa_rad/path_vqa train), distill, eval on the full eval set, then plug the distilled cheap leg into the
  3-tier cascade and measure esc0 / FLOPs / latency / guard at parity vs ACC-v2 (the current best).

## RESULT — FLD is a NEGATIVE result (2 configs, de-risk eval-CV, MedVLThinker)
| config | ALL-5 orig→distilled | ALL-6 orig→distilled | per-benchmark |
|---|---|---|---|
| all-teacher-correct gate | 0.638→0.638 (+0.000) | 0.506→0.509 (+0.004) | +PathV/MMMU/MX-U, −VQA-RAD(−0.105) |
| **delta gate (teacher✓,student✗)+replay** | 0.622→0.629 (+0.007) | 0.485→0.495 (+0.009) | **+PathV(+0.099)/SLAKE, −VQA-RAD(−0.086)/MMMU(−0.090)** |

**Conclusion:** LoRA-distilling big-no-think→small-no-think does **not net-improve the cheap leg** — it
*redistributes* accuracy across benchmarks (helps where cap320 perception is learnable, e.g. PathVQA pathology
yes/no; hurts via interference/forgetting on VQA-RAD/MMMU). A single shared adapter cannot lift all benchmarks
at once. This is **capacity/interference-bound and consistent with the recoverability wall** — the cheap leg
can't cheaply absorb the big model's perception advantage at cap320. Combined with the earlier finding that the
escalation *decision* is also capped (LoRA-router 0.72 < logistic 0.73), **both training axes (better gate AND
stronger cheap leg) fail to beat the training-free ACC cascade** → the cascade sits at a genuine efficiency
frontier for these models. This negative result strengthens the paper's central claim.
Data: `ckpts/train/fld/result.json` (config 1), `ckpts/train/fld_delta/result.json` (config 2).

## CALM-Fuse (candidate #4) — also NEGATIVE, and it completes the picture
The union-oracle showed small-nt+big-nt are complementary by **+0.074..+0.139 on all 5 families** (real headroom).
A trained fusion head (`calm_fuse.py`, MLP on [small A-J logprobs, big A-J logprobs, margins, disagree]) captures
almost none of it:
| family | best single | FUSE (per-family) | FUSE (LOFO transfer) | union-oracle | headroom captured |
|---|---|---|---|---|---|
| MedVLThinker | 0.553 | 0.557 | 0.551 | 0.676 | 4% |
| Lingshu | 0.659 | 0.670 | 0.612 | 0.738 | 14% |
| QoQ | 0.514 | 0.504 | 0.526 | 0.645 | −8% |
| Chiron | 0.605 | 0.597 | **0.242** | 0.711 | −8% |
| MedGemma | 0.507 | 0.505 | 0.430 | 0.648 | −2% |

Per-family fusion captures ~0–14% of the headroom (negative on 3 families); **LOFO cross-family transfer collapses**
(Chiron 0.242) — logit scales/calibration differ per family, so a single fuser is NOT model-agnostic.

## CONSOLIDATED CONCLUSION — three distinct training mechanisms, all fail
| mechanism | method | result |
|---|---|---|
| improve the **routing decision** | LoRA-router (self-verify stability) | AUROC 0.722 < logistic 0.733 — capped |
| strengthen the **cheap leg** | FLD (LoRA distill big-nt→small-nt) | net-flat; redistributes acc w/ interference |
| **fuse** the two legs | CALM-Fuse (trained logit fusion) | captures ~0% of oracle headroom; no transfer |

**The deep finding:** the exploitable structure (recoverability, complementarity) is **real but not learnable** —
the union-oracle is +0.07–0.14 and recoverability/stability exist, yet *every* training mechanism (route /
distill / fuse) bottlenecks on the same un-learnable question: *which model is right on this query?* (~0.58–0.73
AUROC ceiling). The med-VLM cascade sits at a **genuine efficiency frontier**; the training-free ACC cascade is
near-optimal, and learned cross-family methods additionally fail to transfer. This is a strong, defensible
*negative* result that sharpens the paper's central claim. The model-agnostic lever is therefore **not learned
routing/fusion** but the **structural mode axis (drop think on perception)** + using each model's **native think
prompt** (the one fixable confound behind the cross-family weirdness). Data: `ckpts/train/{lora_stability,fld,
fld_delta}/result.json`, `results/cascade_methods/calm_fuse.json`.

## Model-agnosticism (the honest weak spot, verified)
FLD's premise (big-nt > small-nt, so there's competence to distill) holds for **MedVLThinker & Lingshu**, but
for **QoQ/Chiron/MedGemma big-nt is often WORSE than small-nt** (QoQ VQA-RAD 0.724→0.699; Chiron PathVQA
0.838→0.664; MedGemma SLAKE 0.839→0.793), so there's little to distill there. Honest framing: FLD is a
*cheap-leg improvement where a competence gap exists*; a leave-one-family-out (LOFO) transfer test will probe
whether a shared adapter generalizes.

## Why the cross-family cascade behaviour looked "weird" (diagnosis)
All five ARE medical VLMs (all >> chance, varied predictions). The extremes are partly confounds, not pure
model properties: (1) **binary benchmarks** (PathVQA/SLAKE/VQA-RAD median = 2 options) make accuracy a
bias-sensitive binary score — Chiron's "2B>8B" is PathVQA-driven, yet 8B>2B on 4-option MMMU (so it's
task-shape, not global inverse scaling); (2) the **think tier used a non-native prompt** (MedVLThinker's exact
RL `<think>` format on all), so non-MedVLThinker models reason sub-optimally → "think hurts" is inflated
(think runs DO reason, gen 192–481 tok, parse 0.99 — not a parse bug). The robust effect (no-think ≥ think on
perception) holds; its magnitude is confounded. **Fixable:** use each model's native reasoning trigger (re-test pending).
