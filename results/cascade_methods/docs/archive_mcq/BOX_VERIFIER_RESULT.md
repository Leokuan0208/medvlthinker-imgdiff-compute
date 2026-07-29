# Trained box-verifier — the structured-output positive (SLAKE organs + real MS-CXR)

> The structured-output half of the §5.10 trained-verifier result: training breaks the selection luck floor
> for verifiable bounding-box outputs, on organ grounding (SLAKE) AND a real chest-X-ray pathology benchmark
> (MS-CXR). Companion to the free-text answer-verifier ([[trained-verifier-result]]). All real checkpoint
> output. Code: `run_lora_box_verifier.py` (LoRA-Qwen2.5-VL judges "does this box localize the {target}?",
> IoU≥0.3=label, best-of-8 select, grouped split by image). Fig: `paper/figs/limits/fig_trained_verifier_unified.png`.

## Results (held-out, IoU≥0.3)
| grounding setting | greedy | SC-medoid (training-free) | **trained box-verifier** | oracle@8 | gap captured | n |
|---|---|---|---|---|---|---|
| SLAKE organs | 0.197 | 0.164 *(below greedy)* | **0.255** | 0.343 | **40%** | 487 |
| **MS-CXR pathology (real benchmark)** | 0.041 | 0.053 | **0.230** | 0.285 | **77%** (5.6× lift, ~9 s.e.) | 435 |

In both, the training-free **SC-medoid** (spatial self-consistency) is **luck-floored** (at/below greedy;
SC-agreement AUROC ≈ 0.48–0.56 on a competent grounder), and a **trained** box-verifier captures 40–77% of
the oracle gap — the same pattern as the free-text answer-verifier (49% pooled). Unified: *training* is the
active ingredient that breaks the luck floor across output **types** (free-text + boxes) and **domains**
(VQA, organ grounding, chest-X-ray pathology grounding), now including a real published benchmark.

## Key methods notes / gotchas
- **Coordinate space (critical):** Qwen2.5-VL emits `bbox_2d` in its **smart-resized** pixel space; for LARGE
  images (MS-CXR chest X-rays >1 MP) that differs from original dims. The fix (qwen_vl_utils.smart_resize
  scaling, no-op for small SLAKE images) turned an apparent MS-CXR "negative" (IoU 0.022) into the real
  result (meanIoU 0.098, oracle@0.3 0.253). ALWAYS scale Qwen boxes by W/w_bar, H/h_bar on large images.
- **The "escape is an incompetence artifact" subtlety (§5.9):** for a WEAK grounder, SC-agreement *appears*
  to predict correctness (diverse boxes → rare agreement spuriously correlates), but for a COMPETENT grounder
  it collapses to chance — so SC-medoid never gives a real selection win; only the trained verifier does.

## Published baselines / honest scope
- Baseline grounder: MS-CXR SOTA is **MedGround-R1** (arXiv:2507.02994, MICCAI'25, mIoU 79) — a *trained
  grounder*; its checkpoint is **unreleased** (github bio-mlhui/MedGround-R1). Our box-verifier lifts
  *selection over a weak base grounder* (Qwen2.5-VL greedy 0.041) — it does NOT beat MedGround-R1's grounding
  in absolute terms; the contribution is the selection-method result + the luck-floor characterization.
- Method class: GenRM (arXiv:2408.15240, ICLR'25) generative verifier for best-of-N — our box-verifier is the
  structured-output multimodal-medical instance. Training-free baseline beaten: SC-medoid (spatial consistency).

## Zero-shot baseline (training is the active ingredient — DONE)
Untrained Qwen judging the same boxes (run with --epochs 0, same split): SLAKE **0.177** (≤ greedy 0.197,
luck-floored) vs trained 0.255; MS-CXR **0.115** (30% of gap, modest signal) vs trained 0.230 (77%). The
trained box-verifier clearly beats zero-shot on both (+0.078 / +0.115); on SLAKE zero-shot is below greedy
(like free-text zero-shot P(True)), on MS-CXR it has modest signal (judging a drawn box is more concrete) —
but training is dominant either way (doubles the MS-CXR captured gap). Code: run_lora_box_verifier.py --epochs 0.

## Validation status — COMPLETE
- Robustness CONFIRMED across 2 seeds: SLAKE trained 0.255/0.257 (greedy 0.197/0.177, captures 40%/53%);
  MS-CXR trained 0.232/0.230 (greedy 0.041/0.039, captures 78%/77% — remarkably stable). Both box-verifiers
  robustly beat greedy/SC-medoid/zero-shot on every seed. (Free-text verifier already 2-seed.) Bug fixed:
  draw() now guards degenerate/off-image boxes (a rare seed-1 box crashed PIL rectangle).
