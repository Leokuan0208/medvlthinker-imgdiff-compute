# Session progress log — June 25–26, 2026 (the trained-verifier program)

> Continues from `progress_June_24.md`. Records the 2026-06-25→26 work: after the user dropped abstention,
> a fresh new-method loop that (a) mapped the *luck floor* of training-free selection across every output
> type, then (b) found and exhaustively validated the genuine positive — a **trained outcome verifier** that
> breaks the luck floor for both free-text answers and structured grounding boxes, including a real
> PhysioNet benchmark. All numbers from real checkpoint output (standing no-fabrication rule); every headline
> figure audited against its `result.json`. Paper §5.9–§5.10 + abstract + intro contributions updated.
> Full detail in `results/cascade_methods/{OPENENDED_SELECTION_LUCKFLOOR, RECOVERABILITY_IS_CAPACITY_BOUND,
> KNOWLEDGE_AUGMENTATION_FEASIBILITY, NEW_DIRECTIONS_2026-06-25, TRAINED_VERIFIER_RESULT, BOX_VERIFIER_RESULT}.md`.

## Phase A — the luck floor (training-free is bounded everywhere; §5.9)
Probed every axis left after the gate: **action** (cheap same-model repairs recover 14% of errors the 32B
misses but are unharvestable — net-flat, ladder loses at parity → capacity-bound); **selection** (open-ended
oracle gap large, e.g. SLAKE 0.730→0.879, but a *majority trap* makes self-consistency fail and every
training-free selector sits at the random-pick floor: self-verify 0.715 < random 0.720, 32B listwise 0.758
< 32B single-pass 0.819); **synthesis** (priming the 32B with cheap candidates backfires, 0.774);
**knowledge augmentation / RAG** (genuinely-unknown errors are capacity-bound, not knowledge-retrievable;
PathVQA difficulty is genuine, a systematic audit *refuted* my own caption-artifact hypothesis);
**cross-family agreement** (real signal but collapses to "trust the stronger model"); **few-shot ICL**
(hurts). **Structured outputs too:** SLAKE/MS-CXR grounding SC-medoid is luck-floored, and the apparent
"escape" with a weak grounder is an artifact of incompetence (vanishes for a competent grounder). →
**Unified negative: training-free selection over a single model's samples is luck-bound regardless of output
type.** It's *luck, not latent knowledge* — the model doesn't know which sample is right.

## Phase B — THE POSITIVE: a trained outcome verifier breaks the luck floor (§5.10)
LoRA-fine-tune a verifier to score P(correct | image, question, output) on per-sample LLM-judge labels,
then select best-of-8. **Three confirmations, all 2-seed robust, all audited:**

| setting | greedy | training-free selector | **trained verifier** | oracle | gap captured |
|---|---|---|---|---|---|
| free-text answers (pooled-5 datasets, n=1064) | 0.413 | SC 0.411 / zero-shot-32B 0.357 | **0.501** | 0.592 | **49%** |
| SLAKE organ boxes (n=487, IoU≥0.3) | 0.197 | SC-medoid 0.164 / zero-shot 0.177 | **0.255/0.257** | 0.343 | 40–53% |
| **MS-CXR chest-X-ray pathology boxes (real benchmark, n=435)** | 0.041 | SC-medoid 0.053 / zero-shot 0.115 | **0.230/0.248** | 0.285 | **77/76%** |

Key sub-results:
- **Training is the active ingredient.** Zero-shot verification is luck-floored (free-text P(True) 0.319 <
  greedy; SLAKE box 0.177 ≤ greedy); training beats it on all (free-text 7B verifier beats the zero-shot
  **32B** verifier despite 5× smaller; MS-CXR training doubles the captured gap 30%→77%).
- **Generalizes:** the pooled-4 free-text verifier transfers zero-shot to a held-out **5th dataset**
  (RadImageNet-VQA, +0.024) and across modality (Kvasir-OOD).
- **Image-grounded** (blank-image ablation drops 0.047 — refutes Verification-Mirage's "lazy verifier").
- **A real test-time-scaling method:** best-of-K rises monotonically (0.385→0.501 at K=8; K=16 keeps rising
  to 0.424 on a fresh larger sample, with honest diminishing returns) while random stays flat.
- **Test-time compute beats parameters:** verifier-bo8 on the **7B (0.501)** beats the **32B single pass
  (0.444 pooled)** — when scaling params buys little (§5.7), test-time compute on the small model buys more.
- Published baselines: GenRM (ICLR'25), Weaver (NeurIPS'25), P(True), Med-RewardBench; constructive rebuttal
  of *Verification Mirage* (2606.10850); the novelty cell (trained multimodal outcome verifier for open-ended
  medical VQA + grounding best-of-N on judge labels) is unoccupied.

## Data / infra acquired (user granted access)
- **RadImageNet-VQA** (HF, auto-gated): 2000 open-ended Qs prepped (`prep_radimagenet.py`) — 5th dataset.
- **MS-CXR 1.1.0** (PhysioNet creds): 1448 phrase-grounding boxes + 1047 selectively-downloaded MIMIC-CXR-JPG
  chest X-rays (robust validated-retry beat the PhysioNet throttle). MedGround-R1 SOTA checkpoint is unreleased.
- **GOTCHA fixed:** Qwen2.5-VL emits boxes in SMART-RESIZED coords; for large images (chest X-rays) scale by
  W/w_bar,H/h_bar (no-op for small images). An earlier MS-CXR "negative" (IoU 0.022) was this coord bug.
- Reclaimed GPU 0 (68 GB leaked from a kill-9'd vLLM worker — killed the orphaned `VLLM::EngineCore`).

## Figures / paper
`fig_trained_verifier_unified.png` (3-positive bars), `fig_verifier_scaling.png` (best-of-K TTS curve),
`fig_selection_luckfloor.png` (§5.9). Abstract finding (1), intro contributions (v)/(vi)/(vii), §5.9, §5.10,
conclusion, and reproducibility index all updated and number-consistent.

## Bottom line
**Training-free selection is luck-floored across every output type (§5.9); a trained verifier is the
universal active ingredient that breaks it (§5.10) — 40–77% of the oracle gap for free-text answers AND
bounding boxes, on five VQA datasets and a real chest-X-ray benchmark, 2-seed robust, a genuine
test-time-scaling method, and accuracy-optimal vs simply using a 5× larger model.** The session's genuine,
exhaustively-validated novel positive. Open items (user's): rotate HF token + PhysioNet password; framing &
§5.8-abstention decisions; optional `.bib` conversion.
