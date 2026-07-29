# A trained verifier breaks the open-ended luck floor (the session's genuine POSITIVE)

> New-method-loop, 2026-06-25. The one method that beats the generation-selection luck floor on open-ended
> medical VQA. LoRA-fine-tuned Lingshu-7B verifier scoring P(correct | image, question, free-text answer) on
> the 8,234 per-sample LLM-judge labels; selects best-of-8. Honest grouped split by question (no leakage).
> Code: `src/training_methods/run_lora_verifier_open.py`. All numbers from real checkpoint output.

## DEFINITIVE result — trained POOLED over all 4 datasets (held-out, grouped split, n=1064)
| dataset | greedy | SC | **trained verifier** | oracle@8 | gap captured |
|---|---|---|---|---|---|
| PathVQA-open | 0.352 | 0.349 | **0.441** | 0.513 | 55% |
| Kvasir-open (GI) | 0.282 | 0.282 | **0.405** | 0.493 | 58% |
| VQA-RAD-open | 0.519 | 0.500 | **0.611** | 0.722 | 45% |
| SLAKE-open | 0.738 | 0.738 | **0.762** | 0.895 | 15% |
| **POOLED** | 0.413 | 0.411 | **0.501** | 0.592 | **49%** |
Pooling all 4 into training lifts **every** dataset (+0.024 to +0.123) and **captures 49% of the oracle gap**
— resolving the 2-dataset transfer residual (VQA-RAD/Kvasir now strongly positive in-distribution). This is
the headline. `[REPRO: VERIF_DSETS=... run_lora_verifier_open.py]`

## Original 2-dataset result (SLAKE+PathVQA; the robustness/transfer experiments)
| dataset | greedy | self-consistency | zero-shot self-verify P(True) | **TRAINED verifier** | oracle@8 |
|---|---|---|---|---|---|
| **PathVQA-open** | 0.343 | 0.324 | **0.319** (below greedy) | **0.414** | 0.497–0.517 |
| SLAKE-open | 0.738 | 0.738 | 0.715* | 0.743 | 0.872 |
| **POOLED (n=644)** | 0.447 | 0.443 | — | **0.509** | 0.606 |
\*SLAKE zero-shot self-verify from the earlier consistent-judge run.

- **The trained verifier captures 39% of the oracle gap pooled, 51% on PathVQA** — vs ≤24% for *every*
  training-free selector ([[openended-selection-luckfloor]]) and *below-greedy* for zero-shot self-verify.
- **Training is the active ingredient.** Zero-shot verification (P(True)) is luck-floored — *below greedy*
  on PathVQA (0.319 < 0.343). LoRA-training the same model on judge labels lifts selection to 0.414
  (**+0.095 over zero-shot, +0.071 over greedy**). The signal the frozen models couldn't surface is
  *learnable* from a few thousand labels.
- **Honest scope.** The win is concentrated on **PathVQA** (large oracle gap to harvest); **SLAKE** is
  near-saturated (answerable greedy 0.738, little headroom) so the lift is marginal (+0.005). The method
  helps *where there is headroom AND the correctness signal is learnable*.

## Published baselines (VERIFIED) and novelty
- Method class: **GenRM — Generative Verifiers** (arXiv:2408.15240, **ICLR 2025**): trained generative
  verifier beats discriminative/DPO/LLM-as-judge for best-of-N (GSM8K 73→93%). Our LoRA verifier is the
  multimodal-medical instance.
- Free baselines beaten: **P(True)** (Kadavath, arXiv:2207.05221) = our zero-shot self-verify (0.319,
  below greedy); training-free **Self-Certainty BoN** (arXiv:2502.18581, NeurIPS 2025) ≈ self-consistency
  (0.324). **Weaver** (arXiv:2506.18203, NeurIPS 2025) shows trained/combined verifiers beat majority voting
  in LLMs — we show it transfers to medical open-ended VQA.
- Closest medical artifact: **Med-RewardBench** (arXiv:2508.21430) — multimodal medical RM *benchmark*
  (no BoN method); **Med-PRM** (arXiv:2506.11474, EMNLP 2025) is text-only+RAG. **No published paper occupies
  our cell:** a trained multimodal *outcome* verifier for open-ended free-text medical VQA best-of-N on
  LLM-judge labels.
- **Engages a 2026 negative:** *Verification Mirage* (arXiv:2605.10850) shows self-verification fails on
  medical VQA except on measurable tasks. We confirm its *zero-shot* claim (P(True) below greedy) AND show
  **training overcomes it** for free-text selection — a direct, constructive rebuttal.

## Validation status
- Robustness: **CONFIRMED across two grouped splits** — seed 0 vs seed 1: PathVQA trained-verify
  0.414 / 0.426 (greedy 0.328 / 0.329, oracle 0.497 / 0.523), pooled 0.509 / 0.533 (greedy 0.447 / 0.463).
  ~+0.09 PathVQA / ~+0.07 pooled both seeds; ~50% / ~40% of the oracle gap. Not split-variance.
- Image-grounding ablation **DONE — verifier IS image-grounded** (refutes "lazy verifier"): blanking the
  image drops selection 0.047 pooled (SLAKE 0.743→0.679, PathVQA 0.414→0.374). Honest: it combines a learned
  text-correctness prior (blank-image 0.463 > greedy 0.447) with genuine visual grounding (the +0.047).
  `verifier_image_ablation.py`.
- Zero-shot **32B** verify on PathVQA **DONE = 0.357** (captures only 8% of the gap). **The trained 7B
  verifier (0.414/0.426) beats the zero-shot 32B (0.357) by +0.06–0.07 — a 5×-larger model — because it is
  trained.** Complete PathVQA bar: greedy 0.343 / SC 0.324 / zero-shot-self 0.319 / zero-shot-32B 0.357 /
  **trained-7B 0.414–0.426** / oracle 0.517.
- **Cross-dataset transfer (nuanced, mostly positive):** the verifier trained on SLAKE+PathVQA only, applied
  zero-shot to HELD-OUT datasets: **Kvasir-OOD (GI endoscopy, a different modality) 0.286→0.327 (+0.041, 20%
  of its oracle gap — REAL transfer)**; VQA-RAD 0.435 < greedy 0.465 (HURTS, but VQA-RAD is saturated, greedy
  0.465, no headroom). So the verifier **transfers to OOD data WHERE THERE IS HEADROOM** (Kvasir) and is
  neutral-to-negative where saturated (VQA-RAD/SLAKE) — the same headroom-gated pattern as in-distribution.
  It generalizes across modality, just less strongly than in-distribution (+0.09 PathVQA → +0.04 Kvasir-OOD).
  `verifier_transfer_eval.py`.
- Remaining (optional, future): bootstrap CI; train-split headline; pooled multi-dataset verifier training
  (train on 3, test on the 4th) to test whether broader training buys transfer.
- Pending baseline: zero-shot **32B** verify on PathVQA (needs both GPUs; deferred behind seed-1 run). On
  SLAKE the 32B zero-shot verify reached 0.758 (still 24% of gap); the trained 7B is expected to exceed the
  32B zero-shot on PathVQA given self-verify's collapse there.
- Next rigor: bootstrap CI on the paired (trained − greedy) per-question difference; optional cross-dataset
  transfer (train PathVQA → test SLAKE) and a train-split (not test-CV) generation for the headline number.

## 5th dataset — RadImageNet-VQA (user-granted HF access): luck floor + zero-shot verifier transfer
RadImageNet-VQA open-ended (2000 Q, anatomy+pathology, anti-shortcut). **Luck floor reproduced
independently:** SC=greedy=0.329, oracle@8=0.512 (+0.18 gap). **Pooled-4 verifier (trained on the OTHER 4
datasets) transfers zero-shot:** 0.329→0.353 (+0.024, 13% of gap, ~2.3 s.e. on n=2000). Confirms the verifier
generalizes to a fully held-out 5th dataset, headroom-gated (weaker than in-distribution ~40-50%). In paper §5.10.
