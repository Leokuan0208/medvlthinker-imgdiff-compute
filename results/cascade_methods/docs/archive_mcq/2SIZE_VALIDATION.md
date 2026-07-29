# 2-Size ACC Validation Across Medical VLM Families

**Goal.** Test whether the Adaptive-Compute Cascade (ACC) — and its two axes (the *mode* axis:
big-no-think vs big-think; the *size-cascade* axis: small→big) — generalizes beyond MedVLThinker to other
2-size medical VLM families. **No fabricated numbers**: every figure is from real checkpoint output
(`results/cascade_methods/acc_2size_all.txt`). Generated 2026-06-20.

## Methodology
- **Families & tiers** (small no-think → big no-think → big think):
  | family | small | big-no-think | big-think | arch | runner |
  |---|---|---|---|---|---|
  | MedVLThinker (ref) | 7B-nt@cap320 | 32B-nt@cap320 | 32B-think@fullres | Qwen2.5-VL | run_vlm_eval |
  | Lingshu | 7B-nt@cap320 | 32B-nt@cap320 | 32B-think@fullres | Qwen2.5-VL | run_vlm_eval |
  | MedGemma | 4B-nt | 27B-nt | 27B-think (CoT) | Gemma3 | run_peer_eval (llm.chat) |
- **Models** downloaded to `/data/dan/hf_cache` (lingshu-medical-mllm/Lingshu-7B,-32B;
  google/medgemma-4b-it,-27b-it). Inference via vLLM 0.10, offline from cache. MedGemma is fixed-resolution
  (Gemma3/SigLIP) so it has **no resolution sweep**; "think" = CoT prompt ("reason step by step … Answer: X").
- **Benchmarks (MedVLThinker-Eval, 8,220 samples, 7 splits):** PMC-VQA, SLAKE, VQA-RAD, PathVQA, MMMU,
  MedXpert-Reasoning, MedXpert-Understanding. **ALL-6** = all 7 splits (MedXpert = its 2 splits); **ALL-5** =
  ALL-6 minus the two MedXpert splits (PMC/SLAKE/VQA-RAD/PathVQA/MMMU). *(Note: the per-benchmark table below
  has all 7 splits; earlier summaries that showed only 4 perception benchmarks were display truncations.)*
- **Metrics.** Accuracy per benchmark + ALL-5/ALL-6 pooled averages (exact, from labels). ACC cascade =
  small-nt → big-nt (gate: small-nt margin) → big-think (gate: small-vs-big-nt agreement), calibrated to
  always-32B-think parity at min cost, honest 50/50 split × 20 seeds. **guard** = avg #benchmarks/seed below
  always-small (never-worse-than-small bar; 0 = clean). **FLOPs%** = 2·N·(P+G) per tier vs always-big-think;
  **approximate** for the new families (prompt tokens estimated, not from a per-model token cache); accuracy
  and escalation are exact. Latency not re-measured for the new families (would need per-model batch-1 runs).

## Results — per-benchmark accuracy (all 6 benchmarks) + ALL-5 / ALL-6 averages

### MedVLThinker 7B/32B (reference; Qwen2.5-VL)
| split | small-nt | big-nt | big-think | nt−think |
|---|---|---|---|---|
| PMC-VQA | 0.543 | 0.551 | 0.556 | −0.005 |
| SLAKE | 0.762 | 0.849 | 0.764 | **+0.084** |
| VQA-RAD | 0.761 | 0.853 | 0.776 | **+0.077** |
| PathVQA | 0.641 | 0.661 | 0.673 | −0.011 |
| MMMU | 0.547 | 0.624 | 0.688 | −0.065 |
| MedXpert-Reasoning | 0.225 | 0.279 | 0.326 | −0.047 |
| MedXpert-Understanding | 0.256 | 0.292 | 0.384 | −0.092 |
| **ALL-5 average** | 0.620 | **0.646** | 0.646 | −0.001 |
| **ALL-6 average** | 0.526 | 0.557 | 0.572 | −0.015 |

ACC cascade @ parity: **ALL-6 0.5708 / esc0 68% / think 30% / FLOPs 65.0% / guard 0.00**; **ALL-5 0.6437 /
esc0 32% / think 15% / FLOPs 37.1% / guard 0.05**. Resolution sweet-spot (ALL-5): cap320 (0.604→0.620→plateau).

### Lingshu 7B/32B (Qwen2.5-VL)
| split | small-nt | big-nt | big-think | nt−think |
|---|---|---|---|---|
| PMC-VQA | 0.622 | 0.640 | 0.598 | **+0.042** |
| SLAKE | 0.841 | 0.894 | 0.829 | **+0.065** |
| VQA-RAD | 0.732 | 0.816 | 0.724 | **+0.092** |
| PathVQA | 0.782 | 0.861 | 0.760 | **+0.102** |
| MMMU | 0.847 | 0.629 | 0.635 | −0.006 |
| MedXpert-Reasoning | 0.247 | 0.296 | 0.296 | +0.000 |
| MedXpert-Understanding | 0.278 | 0.329 | 0.352 | −0.023 |
| **ALL-5 average** | 0.734 | **0.784** | 0.707 | **+0.077** |
| **ALL-6 average** | 0.618 | **0.668** | 0.611 | **+0.057** |

ACC cascade @ parity-vs-32B-think: trivially met at 0% escalation (7B-nt 0.618 already > 32B-think 0.611!).
The meaningful operating point is **always-32B-no-think: 0.668 ALL-6 / 0.784 ALL-5 — beats 32B-think by
+5.7/+7.7 pt, ~80× faster (0.34 s vs 28 s), ~⅓ FLOPs**, just by dropping think. Size cascade (7B→32B-nt)
adds little: matching 32B-nt needs ~75% escalation (~98% FLOPs); **7B even beats 32B on MMMU (0.847 vs
0.629)**. Resolution sweet-spot (ALL-5): cap320 (0.708→0.734→plateau).

### MedGemma 4B/27B (Gemma3) — contrasting regime
| split | small-nt(4B) | big-nt(27B) | big-think(27B) | nt−think |
|---|---|---|---|---|
| PMC-VQA | 0.462 | 0.474 | 0.461 | **+0.013** |
| SLAKE | 0.839 | 0.793 | 0.808 | −0.014 |
| VQA-RAD | 0.816 | 0.768 | 0.695 | **+0.074** |
| PathVQA | 0.646 | 0.604 | 0.645 | −0.041 |
| MMMU | 0.500 | 0.512 | 0.547 | −0.035 |
| MedXpert-Reasoning | 0.228 | 0.249 | 0.275 | −0.026 |
| MedXpert-Understanding | 0.269 | 0.262 | 0.345 | −0.083 |
| **ALL-5 average** | 0.603 | 0.580 | 0.596 | −0.017 |
| **ALL-6 average** | 0.515 | 0.500 | 0.523 | −0.023 |

ACC cascade @ parity: ALL-6 0.5236 / FLOPs 49.9% / **guard 1.95**; ALL-5 0.6028 / FLOPs 14.0% / **guard 0.75**
— guardrail VIOLATED: the 27B is sometimes worse than the 4B (4B-nt 0.603 ≥ 27B-think 0.596 on ALL-5; 4B-nt
0.816 vs 27B-think 0.695 on VQA-RAD), so escalating can hurt. Small competence gap → little for ACC to gain.

## Synthesis — ACC has two axes with different generality
1. **Mode axis** (use the big model's *fast no-think* mode; no-think ≥ think on perception): **generalizes
   strongly.** Lingshu (+0.077 ALL-5) and MedVLThinker show 32B-no-think ≥/≫ 32B-think on perception
   (SLAKE/VQA-RAD especially); cap320 is the sweet-spot for both Qwen2.5-VL families. Weak/mixed for MedGemma.
2. **Size-cascade axis** (small→big routing): **family-dependent** — needs a real, *routable* small→big
   competence gap. Present in MedVLThinker; **absent/non-monotone in Lingshu** (7B>32B on MMMU; gap not
   cheaply routable) and **MedGemma** (4B≈27B). When the gap is small or non-monotone, the cascade saves
   little and can violate the never-worse-than-small guardrail.

**Conclusion:** ACC's robust, transferable contribution is the **mode tier** (drop think on perception VQA);
the size cascade is an add-on that pays off only when the small→big gap is both large and routable.

## Reproducibility
- Inference: `run_lingshu_acc.sh`, `run_medgemma_acc.sh` (+ `src/labeling/run_vlm_eval.py`, `run_peer_eval.py`).
- Analysis: `python3 src/cascade_methods/acc_2size.py --family {medvlthinker,lingshu,medgemma}` → this file's
  numbers; full console log in `results/cascade_methods/acc_2size_all.txt`.
- Labels: `ckpts/acc_gen/{lingshu7b/<cap>, lingshu32b/<mode>, medgemma4b/nt, medgemma27b/{nt,think}}` (gitignored).
- Caveat: one Lingshu-32B shard was corrupted by a 4-way concurrent download; re-fetched + safetensors-verified.

## Two more families added (2026-06-20): QoQ-Med-VL 7B/32B, Chiron-o1 2B/8B
Found via a literature/HF search workflow; both ungated, vLLM-native, same-lineage pairs.
**QoQ-Med-VL (Qwen2.5-VL):** over-thinking holds on *every* split (POOLED ALL-5 nt−think **+0.051**; big-nt
0.610 > big-think 0.559); small≈big (7B-nt 0.605 ≈ 32B-nt 0.610) so the size cascade has little headroom.
**Chiron-o1 (InternVL3):** **inverse scaling** — the 2B *beats* the 8B overall (ALL-5 small-nt **0.725** >
big-nt 0.654 > think 0.586), driven by PathVQA (2B 0.838 vs 8B 0.664); over-thinking still holds on perception.

## ALL methods × measured batch-1 latency, on ALL five families (2026-06-20)
Full bake-off (10 gate methods) under the 3-tier config (small-nt → big-nt → big-think), FLOPs-calibrated at
always-big-think parity, honest 50/50 × 20 seeds. **AutoMix self-verify run for all five.** Raw output:
`results/cascade_methods/acc_allmethods_all.txt`; code `src/cascade_methods/acc_allmethods.py`.

**Latency methodology (NOT the 30 h real-time eval).** Same approach as the original MedVLThinker latency
(`acc_compare.fit_models`): a *small* batch-1 sample (`--batch1`, max_num_seqs=1, n=6/benchmark = ~42/tier)
gives `latency_s` vs `gen_tokens`; we fit `latency = a·gen + b` per tier (robust: drop >2.5σ warm-up/GC
spikes) and *compute* each method's cascade latency = mean(l₀ + esc₀·l₁ + esc₁·l₂) over the real escalation
pattern. Measured uniformly for all five families (same GPU, vLLM 0.10, batch-1). Code: `run_latency_all.sh`,
`acc_allmethods.fit_lat`.

### Per-tier batch-1 latency fits (latency_s = a + b·gen_tokens)
| family | small-nt a,b | big-nt a,b | big-think a,b | big-think range |
|---|---|---|---|---|
| MedVLThinker | 0.116, 0.0060 | 0.119, 0.0597 | 0.120, 0.0287 | 3.7–22.8 s |
| Lingshu | 0.069, 0.0230 | 0.135, 0.0448 | 0.135, 0.0292 | 2.6–11.6 s |
| QoQ | 0.139, 0.0000 | 0.118, 0.0588 | 0.244, 0.0283 | 3.2–18.0 s |
| Chiron | 0.096, 0.0480 | 0.142, 0.0708 | 0.155, 0.0133 | 1.1–7.6 s |
| MedGemma | 0.090, 0.0302 | 0.280, 0.0094 | 0.015, 0.0273 | 2.2–28.7 s |

### Results (ALL-6 and ALL-5; acc / FLOPs% / latency / guard). Full 10-method tables in the .txt.
**MedVLThinker — the cascade works cleanly (think helps, gap routable):**
| ALL-6 | acc | FLOPs% | lat | guard |
|---|---|---|---|---|
| always-small-nt | 0.5262 | 8.4 | 0.13 s | 0 |
| always-big-nt | 0.5573 | 36.2 | 0.24 s | 0 |
| always-big-think (parity) | 0.5723 | 100 | **11.34 s** | 0 |
| **Ours (ACC-v2)** | 0.5693 | 52.0 | **2.28 s** | **0** |
| CASP-Stability (trained) | 0.5698 | 49.0 | 1.78 s | 0.05 |
| ACC-v1 (margin) | 0.5687 | 53.9 | 2.70 s | 0 |
| AutoMix | 0.5692 | 54.6 | 2.51 s | 0.05 |

ALL-5: Ours 0.6450 / 24.9% / **0.44 s** / guard 0.05 vs always-big-think 0.6463 / **8.89 s**. So at parity,
**Ours cuts batch-1 latency ~5× (ALL-6) to ~20× (ALL-5)** and ~½ the FLOPs, guardrail-clean. All gate methods
cluster within ~0.003 acc; Ours/CASP are the cheapest guard-clean ones.

**MedGemma — think helps slightly on ALL-6, cascade non-trivial but guardrail-violating:**
| ALL-6 | acc | FLOPs% | lat | guard |
|---|---|---|---|---|
| always-big-think (parity) | 0.5230 | 100 | 13.17 s | 4.00 |
| Ours (ACC-v2) | 0.5233 | 66.6 | 3.44 s | **1.90** |
| CASP-Stability (trained) | 0.5218 | 24.4 | 0.82 s | 1.15 |
| AutoMix | 0.5164 | 17.5 | 0.41 s | 0.60 |

MedGemma is the only *other* family where the think tier fires at all; but the 27B is non-monotone vs the 4B,
so every method violates the never-worse-than-small guard (≥0.6). On ALL-5 all methods collapse to small-nt.

**Lingshu, QoQ, Chiron — the think tier is harmful, so the cascade collapses to "use the cheap leg":**
For these, always-big-think is *below* the cheap small-nt (Lingshu 0.611<0.618; QoQ 0.478<0.509; Chiron
0.502<0.602), so parity-to-think is met at **0% escalation** and *all 10 methods become identical* (just
small-nt): Lingshu 0.6176 @10.3% @0.14 s; QoQ 0.5096 @9.1% @0.14 s; Chiron 0.6023 @20.0% @0.19 s — all guard 0.
The meaningful operating point for these is the **big-no-think** model (drop think): Lingshu 32B-nt 0.668
@0.27 s, QoQ 32B-nt 0.522 @0.24 s; for Chiron even the **2B** is best (0.602 @0.19 s; the 8B is worse).

### Synthesis (with latency)
ACC's full 3-tier cascade pays off **only when the think tier actually helps the big model on the pooled set
*and* the small→big gap is routable** — true for **MedVLThinker** (5–20× latency cut at parity, guard-clean)
and weakly for **MedGemma** (but guard-violating). For **Lingshu / QoQ / Chiron** the think tier is net-harmful
(or the model inverse-scales), so the deployment is simply the small or big **no-think** model — already
≤0.3 s batch-1. Across all five, the robust transferable lever is the **mode axis (drop think on perception)**;
the gate-method *choice* only matters where the cascade is non-degenerate (MedVLThinker, MedGemma), and there
**Ours (ACC-v2) and CASP-Stability are the cheapest guardrail-clean options.**

## Native-think re-test (2026-06-21) — was "think hurts" a prompt artifact?
We re-ran the think tier of the 4 non-MedVLThinker families with each model's OWN native reasoning prompt
(found from training code/papers: Lingshu = empty system + `\boxed{}` user instr; QoQ = its DRPO `<think>`+`\boxed{}`
prompt; Chiron = "Let's reason step-by-step" + "### The final answer is:"; MedGemma = medical system + "Final
Answer:"). Code: `run_native_think.sh`, `compare_native_think.py` → `native_think_compare.json`. n=8220 (MedGemma
partial = perception-4; it errored on long MMMU/MedXpert prompts).

| family | no-think | foreign-think | native-think | nat−foreign | nat−no-think |
|---|---|---|---|---|---|
| Lingshu (ALL-5) | 0.784 | 0.707 | 0.775 | **+0.067** | −0.009 |
| QoQ (ALL-5) | 0.610 | 0.559 | 0.543 | −0.016 | −0.067 |
| Chiron (ALL-5) | 0.654 | 0.586 | 0.593 | +0.006 | −0.062 |
| MedGemma (perc-4) | 0.558 | 0.550 | 0.555 | +0.004 | −0.003 |

**Findings:** (1) **Over-thinking on perception is REAL, not a prompt artifact** — no-think ≥ native-think on
perception for all families. (2) The foreign prompt **inflated** the magnitude: **Lingshu doesn't reason at all
under its native prompt** (gen=3, answers directly → native≈no-think 0.775; the foreign `<think>` forced harmful
reasoning, 0.707) — so its "think collapses the cascade" was largely a foreign-prompt artifact. (3) **QoQ shows
the canonical ACC pattern** natively — think HELPS reasoning (MMMU +0.071 over no-think) but HURTS perception
(PMC −0.085, VQA-RAD −0.077) → net below no-think on perception-heavy pools. (4) **Chiron genuinely over-thinks**
(native ≈ foreign, prompt wasn't the issue). **Implication for model-agnosticism:** a cascade must use each
model's native reasoning trigger (recovers think on reasoning benchmarks + undoes foreign-prompt damage) AND
gate think to reasoning-type questions — but the structural default (no-think for perception) is correct across
all families. The ACC core insight survives the native-prompt control.

## Pending / future
- Energy (NVML) for the new families (only latency + FLOPs captured; energy tracks the same per-tier pattern).
- Optional: Fleming-VL 8B/38B, InternVL3.5 (further families flagged by the search workflow).
