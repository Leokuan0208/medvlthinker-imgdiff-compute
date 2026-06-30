# Open-text medical-VQA cascade — 3-family MASTER TABLE (2026-06-30)

All numbers regenerated from checkpoints via `src/cascade_methods/open_verifier_cascade_table.py` (judge-based
open-text accuracy; LLM judge = MedVLThinker-32B). Method = trained outcome-verifier **best-of-8 selection** +
**verifier-confidence escalation gate** → escalate to the strong model. "cascade-best" = best accuracy on the
acc-vs-escalation frontier (held-out-τ optimism is negligible, +0.00–0.002, see UNIFIED_METHOD_EXPERIMENTS.md).
RadImageNet = held-out OOD (verifier never trained on it). InternVL3 uses the **Lingshu-trained** verifier
(cross-architecture transfer — no InternVL3 verifier was trained).

## Accuracy: the method BEATS the strong model in every family, every dataset, incl. held-out OOD

| family | dataset | cheap(SC) | STRONG (32B/38B) | verifier-bo8 | **cascade-best @esc** | oracle@8 |
|---|---|---|---|---|---|---|
| **Lingshu** | VQA-RAD | 0.465 | 0.600 | 0.575 | **0.625 @24%** | 0.630 |
| (strong=Lingshu-32B) | PathVQA | 0.324 | 0.376 | 0.453 | **0.469 @33%** | 0.517 |
| | Kvasir | 0.286 | 0.301 | 0.439 | **0.448 @20%** | 0.491 |
| | RadImageNet-OOD | 0.329 | 0.289 | 0.353 | **0.353 @0%** | 0.512 |
| | **POOLED** | 0.322 | 0.331 | 0.414 | **0.421 @12%** | 0.513 |
| **MedVLThinker** | VQA-RAD | 0.420 | 0.525 | 0.490 | **0.555 @54%** | 0.600 |
| (strong=MVT-32B) | Kvasir | 0.343 | 0.361 | 0.477 | **0.483 @9%** | 0.550 |
| | RadImageNet-OOD | 0.204 | 0.202 | 0.241 | **0.243 @13%** | 0.317 |
| | **POOLED** | 0.266 | 0.277 | 0.339 | **0.344 @14%** | 0.416 |
| **InternVL3** | VQA-RAD | 0.445 | 0.415 | 0.570 | **0.580 @12%** | 0.620 |
| (strong=IV3-38B; | PathVQA | 0.081 | 0.096 | 0.116 | **0.125 @52%** | 0.192 |
| cross-family verifier) | Kvasir | 0.362 | 0.380 | 0.479 | **0.487 @17%** | 0.593 |
| | RadImageNet-OOD | 0.285 | 0.304 | 0.302 | **0.313 @52%** | 0.398 |
| | **POOLED** | 0.202 | 0.218 | 0.249 | **0.255 @36%** | 0.337 |

Every "cascade-best" row BEATS the strong model. The verifier transfers across **architectures** (Lingshu→InternVL3).

## The GATE: verifier-confidence is best; no trained gate beats it (the answer to "is confidence gate really best?")
AUROC for predicting pick-correctness, pooled: verifier-conf **0.853 / 0.885 / 0.875** (Lingshu/MVT/IV3). Best
trained gate (GBM/MLP/logit on verifier±cheap, incl. full margin/conf/seqlogprob on IV3) = 0.861 / 0.882 / 0.879
→ **Δ ≤ +0.008 (noise), identical cascade accuracy.** Cheap-only gates (self-consistency, n_distinct, even
cheap-conf) = 0.66–0.79, far below. The theoretically-optimal **recoverability** gate (Jitkrittum NeurIPS'23) nudges
recoverability-ranking AUROC (0.60→0.63–0.65) but gives **no better cascade accuracy** and overfits on small sets.
WHY: the **recoverability wall** — the strong model fixes only 6–10% of the verifier's errors (26% where the strong
is competitive), and *which* ones is near-unlearnable (AUROC ~0.4–0.6, far below the oracle ceiling). The verifier's
P(correct) is already a calibrated correctness estimator, so a stability/CASP gate is redundant, not orthogonal.
("CASP" is our coined name; cite **CCPS** arXiv:2505.21772. Verifier-score-as-gate precedents: Self-REF
arXiv:2410.13284, Kiyani 2026 arXiv:2602.17633 — neither uses a trained outcome verifier for a medical-VQA cascade.)

## COST (measured)
FLOPs (prefill-dominated, validated): verifier-bo2 (no gate) **beats the 32B on accuracy AND FLOPs (4.0 vs 4.57
7B-fwd-eq)** where the strong model is weak/OOD; where the 32B is genuinely strong (VQA-RAD) the method is a Pareto
trade (more FLOPs for more accuracy). Latency/energy (measured batch-1): 7B-gen 347ms/45.8J, 7B-verify 175ms/25.3J,
32B-gen 665ms/127J → 32B is only ~1.9× the 7B's latency at batch-1, so best-of-N latency is ~neutral-to-worse
sequentially and ~neutral when the N samples are batched.

## DEPLOYABLE RECOMMENDATION
**Trained outcome-verifier best-of-N selection (the dominant lever) + verifier-confidence escalation gate.**
N=2 is the cost knee (acc-positive at ≈parity FLOPs vs the 32B on weak/OOD domains); raise N to spend compute for
accuracy where the strong model is competent. The gate adds little at N≥8 (redundant with selection) and is mainly a
small-N cost-saver. CASP/CCPS-style trained stability gates belong in the paper as cited, **beaten** baselines.
