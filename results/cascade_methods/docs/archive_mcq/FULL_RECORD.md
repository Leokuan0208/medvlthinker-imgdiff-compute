# FULL RECORD — Adaptive-Compute Cascade: all methods × all models × all metrics

Auto-generated from real artifacts (`make_full_record.py`). **No fabricated numbers** — every value is read from `master_data.csv` / `allmethods_*.json` / NVML logs / LoRA `result.json`.

## 1. Scope

| family | small tier | big tier | arch | runner |
|---|---|---|---|---|
| MedVLThinker | 7B-nt@cap320 | 32B-nt@cap320 → 32B-think@fullres | Qwen2.5-VL | run_vlm_eval |
| Lingshu | 7B-nt@cap320 | 32B-nt@cap320 → 32B-think@fullres | Qwen2.5-VL | run_vlm_eval |
| QoQ-Med-VL | 7B-nt@cap320 | 32B-nt@cap320 → 32B-think@fullres | Qwen2.5-VL | run_vlm_eval |
| Chiron-o1 | 2B-nt | 8B-nt → 8B-think (CoT) | InternVL3 | run_peer_eval |
| MedGemma | 4B-nt | 27B-nt → 27B-think (CoT) | Gemma3 | run_peer_eval |

Benchmarks (MedVLThinker-Eval, 8,220): PMC-VQA, SLAKE, VQA-RAD, PathVQA, MMMU, MedXpert-Reasoning, MedXpert-Understanding. **ALL-6** = all 7 splits; **ALL-5** = drop the two MedXpert splits.

## 2. Method catalog (tier-0 stop signal, tier-1 think-gate signal)

| method | rule |
|---|---|
| **Ours (ACC-v2: agreement)** | tier0 = 7B margin; think-gate = cross-model **disagreement** (7B-nt≠big-nt), ε-tiebreak to low-margin |
| ACC-v1 (margin) | tier0 = 7B margin; think-gate = big-nt margin |
| CASP-Stability (trained) | tier0 = 1−P̂(answer stable) [logistic on 7B signals]; think-gate = agreement |
| MSP/Chow | top-1 probability |  | entropy | predictive entropy |  | Gini/DOCTOR | Gini impurity |
| AutoMix (self-verify) | tier0 = 1−P(self-verify Yes); think-gate = big-nt margin |
| FrugalGPT-style learned | logistic P(tier correct) on signals | | Jitkrittum L2D | P(next right)−P(this right) |
| random | random scores (control) |
| baselines | always-small-nt / always-big-nt / always-big-think (no routing) |

## 3. Measurement methodology

- **Accuracy / per-benchmark / FLOPs / guard**: exact, from the full 8,220-sample bulk labels. FLOPs = 2·N·(P+G) per tier vs always-big-think; **guard** = avg #benchmarks/seed below always-small (0=clean). Cascade calibrated to always-big-think parity at MIN FLOPs, honest 50/50 stratified split × 20 seeds.

- **Latency & energy**: real **batch-1** measurement (`--batch1`, max_num_seqs=1) on a small per-tier sample (6/benchmark), recording per-sample `latency_s` + NVML `energy_j` + peak VRAM; robust fit `metric = a·gen_tokens + b` per tier (drop >2.5σ warm-up/CUDA-graph spikes); per-method cost computed analytically over the real escalation pattern. (NOT a full real-time eval.) Uniform across all 5 families.

## 4. Per-tier cost models + peak VRAM

`latency_s = a·g+b` ; `energy_J = a·g+b` (g = generated tokens). Peak VRAM = vLLM reservation at the run's gpu_mem fraction (TP=2 sums both GPUs), an upper bound, not minimal footprint.

| family | tier | lat a,b | energy a,b | peak VRAM (GB) |
|---|---|---|---|---|
| medvlthinker | small_nt | 0.0000·g+0.129 | 0.000·g+19.88 | 71.45 |
| medvlthinker | big_nt | 0.0000·g+0.231 | 0.000·g+77.83 | 152.82 |
| medvlthinker | big_think | 0.0288·g+0.091 | 16.436·g+-104.54 | 154.08 |
| lingshu | small_nt | 0.0000·g+0.131 | 0.000·g+22.87 | 71.45 |
| lingshu | big_nt | 0.0000·g+0.270 | 0.000·g+88.14 | 152.82 |
| lingshu | big_think | 0.0000·g+0.322 | 0.000·g+113.18 | 154.08 |
| qoq | small_nt | 0.0000·g+0.121 | 0.000·g+17.60 | 71.45 |
| qoq | big_nt | 0.0000·g+0.227 | 0.000·g+68.84 | 152.82 |
| qoq | big_think | 0.0287·g+0.177 | 16.110·g+25.02 | 154.08 |
| chiron | small_nt | 0.0000·g+0.195 | 0.000·g+17.38 | 53.46 |
| chiron | big_nt | 0.0000·g+0.291 | 0.000·g+38.78 | 74.16 |
| chiron | big_think | 0.0131·g+0.207 | 3.785·g+11.04 | 74.17 |
| medgemma | small_nt | 0.0000·g+0.179 | 0.000·g+15.84 | 52.35 |
| medgemma | big_nt | 0.0000·g+0.282 | 0.000·g+63.87 | 155.52 |
| medgemma | big_think | 0.0276·g+0.008 | 14.397·g+-94.59 | 155.52 |

## 5.1. All methods — ALL-6 (acc / esc0% / think% / FLOPs% / latency / energy / guard)

### MedVLThinker 7B/32B (Qwen2.5-VL) — ALL-6 (parity acc = 0.5723)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.5262 | 0.0 | 0.0 | 8.4 | 0.129 | 19.9 | 0.0 |
| always-big-nt | 0.5573 | 100.0 | 0.0 | 36.2 | 0.231 | 77.8 | 0.0 |
| always-big-think [PARITY] | 0.5723 | 100.0 | 100.0 | 100.0 | 11.338 | 6318.8 | 0.0 |
| Ours (ACC-v2: agreement) | 0.5693 | 71.7 | 15.1 | 52.0 | 2.273 | 1181.9 | 0.0 |
| CASP-Stability (trained) | 0.5698 | 74.3 | 11.0 | 49.0 | 1.769 | 899.2 | 0.05 |
| ACC-v1 (margin) | 0.5687 | 66.0 | 19.5 | 53.9 | 2.69 | 1416.6 | 0.0 |
| MSP/Chow | 0.5697 | 69.8 | 19.3 | 57.4 | 2.959 | 1568.1 | 0.0 |
| entropy | 0.5691 | 71.3 | 21.1 | 62.0 | 3.481 | 1863.1 | 0.0 |
| Gini/DOCTOR | 0.5702 | 69.3 | 21.9 | 61.0 | 3.436 | 1837.1 | 0.0 |
| AutoMix (self-verify) | 0.5692 | 73.0 | 17.7 | 54.6 | 2.5 | 1307.0 | 0.05 |
| FrugalGPT-style learned | 0.5677 | 70.0 | 19.1 | 60.4 | 3.304 | 1765.5 | 0.1 |
| Jitkrittum L2D (Diff-Prob) | 0.5666 | 67.4 | 14.6 | 50.6 | 2.289 | 1194.5 | 0.0 |
| random | 0.5641 | 89.3 | 75.9 | 116.5 | 8.948 | 4889.5 | 0.05 |

### Lingshu 7B/32B (Qwen2.5-VL) — ALL-6 (parity acc = 0.6611)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.6175 | 0.0 | 0.0 | 13.1 | 0.131 | 22.9 | 0.0 |
| always-big-nt | 0.6675 | 100.0 | 0.0 | 56.8 | 0.27 | 88.1 | 1.0 |
| always-big-think [PARITY] | 0.6611 | 100.0 | 100.0 | 100.0 | 0.322 | 113.2 | 1.0 |
| Ours (ACC-v2: agreement) | 0.6614 | 60.7 | 0.0 | 48.6 | 0.295 | 76.3 | 1.0 |
| CASP-Stability (trained) | 0.6613 | 61.3 | 0.2 | 50.0 | 0.297 | 77.0 | 1.05 |
| ACC-v1 (margin) | 0.6614 | 60.7 | 0.0 | 48.6 | 0.295 | 76.3 | 1.0 |
| MSP/Chow | 0.6611 | 62.1 | 0.0 | 49.9 | 0.299 | 77.6 | 1.0 |
| entropy | 0.6613 | 64.2 | 0.0 | 51.3 | 0.304 | 79.5 | 1.0 |
| Gini/DOCTOR | 0.6615 | 63.6 | 0.0 | 50.8 | 0.303 | 78.9 | 1.05 |
| AutoMix (self-verify) | 0.6605 | 58.6 | 0.2 | 48.0 | 0.29 | 74.8 | 1.0 |
| FrugalGPT-style learned | 0.6614 | 68.2 | 0.0 | 54.1 | 0.315 | 83.0 | 1.0 |
| Jitkrittum L2D (Diff-Prob) | 0.6595 | 49.1 | 0.5 | 40.4 | 0.265 | 66.7 | 1.1 |
| random | 0.6608 | 91.4 | 0.0 | 65.0 | 0.378 | 103.5 | 1.0 |

### QoQ-Med-VL 7B/32B (Qwen2.5-VL) — ALL-6 (parity acc = 0.4689)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.5094 | 0.0 | 0.0 | 8.8 | 0.121 | 17.6 | 0.0 |
| always-big-nt | 0.522 | 100.0 | 0.0 | 38.3 | 0.227 | 68.8 | 1.0 |
| always-big-think [PARITY] | 0.4689 | 100.0 | 100.0 | 100.0 | 9.719 | 5381.6 | 4.0 |
| Ours (ACC-v2: agreement) | 0.5095 | 0.0 | 0.0 | 8.8 | 0.121 | 17.6 | 0.0 |
| CASP-Stability (trained) | 0.5095 | 0.0 | 0.0 | 8.9 | 0.123 | 18.8 | 0.05 |
| ACC-v1 (margin) | 0.5095 | 0.0 | 0.0 | 8.8 | 0.121 | 17.6 | 0.0 |
| MSP/Chow | 0.5094 | 0.0 | 0.0 | 8.9 | 0.124 | 19.0 | 0.15 |
| entropy | 0.5095 | 0.0 | 0.0 | 8.9 | 0.123 | 18.8 | 0.0 |
| Gini/DOCTOR | 0.5095 | 0.0 | 0.0 | 8.9 | 0.124 | 19.0 | 0.0 |
| AutoMix (self-verify) | 0.5095 | 0.0 | 0.0 | 8.9 | 0.123 | 18.7 | 0.0 |
| FrugalGPT-style learned | 0.5095 | 0.0 | 0.0 | 8.9 | 0.125 | 19.8 | 0.05 |
| Jitkrittum L2D (Diff-Prob) | 0.5095 | 0.0 | 0.0 | 8.9 | 0.124 | 19.0 | 0.0 |
| random | 0.5095 | 0.0 | 0.0 | 8.9 | 0.123 | 18.6 | 0.05 |

### Chiron-o1 2B/8B (InternVL3) — ALL-6 (parity acc = 0.5076)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.6024 | 0.0 | 0.0 | 19.3 | 0.195 | 17.4 | 0.0 |
| always-big-nt | 0.5512 | 100.0 | 0.0 | 77.0 | 0.291 | 38.8 | 3.0 |
| always-big-think [PARITY] | 0.5076 | 100.0 | 100.0 | 100.0 | 4.246 | 1175.6 | 4.0 |
| Ours (ACC-v2: agreement) | 0.6023 | 0.0 | 0.0 | 19.3 | 0.195 | 17.4 | 0.0 |
| CASP-Stability (trained) | 0.6023 | 0.0 | 0.0 | 19.3 | 0.197 | 17.8 | 0.0 |
| ACC-v1 (margin) | 0.6023 | 0.0 | 0.0 | 19.3 | 0.195 | 17.4 | 0.0 |
| MSP/Chow | 0.6023 | 0.0 | 0.0 | 19.3 | 0.197 | 17.7 | 0.0 |
| entropy | 0.6023 | 0.0 | 0.0 | 19.3 | 0.196 | 17.6 | 0.05 |
| Gini/DOCTOR | 0.6023 | 0.0 | 0.0 | 19.3 | 0.196 | 17.5 | 0.05 |
| AutoMix (self-verify) | 0.6023 | 0.0 | 0.0 | 19.3 | 0.197 | 17.7 | 0.2 |
| FrugalGPT-style learned | 0.6024 | 0.0 | 0.0 | 19.3 | 0.197 | 17.9 | 0.15 |
| Jitkrittum L2D (Diff-Prob) | 0.6023 | 0.0 | 0.0 | 19.3 | 0.196 | 17.4 | 0.0 |
| random | 0.6025 | 0.0 | 0.0 | 19.3 | 0.197 | 17.7 | 0.0 |

### MedGemma 4B/27B (Gemma3) — ALL-6 (parity acc = 0.5253)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.5146 | 0.0 | 0.0 | 10.6 | 0.179 | 15.8 | 0.0 |
| always-big-nt | 0.5001 | 100.0 | 0.0 | 66.3 | 0.282 | 63.9 | 4.0 |
| always-big-think [PARITY] | 0.5253 | 100.0 | 100.0 | 100.0 | 12.722 | 6534.5 | 3.0 |
| Ours (ACC-v2: agreement) | 0.5219 | 55.2 | 20.0 | 68.4 | 3.369 | 1613.5 | 1.15 |
| CASP-Stability (trained) | 0.5223 | 21.6 | 3.9 | 29.6 | 1.046 | 446.1 | 1.15 |
| ACC-v1 (margin) | 0.5207 | 62.1 | 41.4 | 93.8 | 5.872 | 2891.7 | 2.7 |
| MSP/Chow | 0.523 | 75.5 | 55.9 | 117.5 | 7.851 | 3897.9 | 2.95 |
| entropy | 0.5229 | 76.2 | 55.8 | 117.9 | 7.883 | 3914.0 | 3.1 |
| Gini/DOCTOR | 0.5229 | 75.7 | 56.0 | 117.7 | 7.864 | 3904.5 | 3.0 |
| AutoMix (self-verify) | 0.519 | 18.2 | 5.6 | 28.1 | 0.895 | 368.3 | 1.55 |
| FrugalGPT-style learned | 0.5233 | 18.4 | 4.4 | 28.1 | 1.149 | 501.6 | 1.3 |
| Jitkrittum L2D (Diff-Prob) | 0.5199 | 12.9 | 2.5 | 22.1 | 0.7 | 274.0 | 1.45 |
| random | 0.5199 | 79.2 | 71.2 | 134.4 | 9.498 | 4738.3 | 3.7 |

## 5.2. All methods — ALL-5 (acc / esc0% / think% / FLOPs% / latency / energy / guard)

### MedVLThinker 7B/32B (Qwen2.5-VL) — ALL-5 (parity acc = 0.6463)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.6201 | 0.0 | 0.0 | 8.9 | 0.129 | 19.9 | 0.0 |
| always-big-nt | 0.6457 | 100.0 | 0.0 | 38.7 | 0.231 | 77.8 | 0.0 |
| always-big-think [PARITY] | 0.6463 | 100.0 | 100.0 | 100.0 | 8.882 | 4915.9 | 0.0 |
| Ours (ACC-v2: agreement) | 0.645 | 35.1 | 2.3 | 24.9 | 0.436 | 172.8 | 0.05 |
| CASP-Stability (trained) | 0.6461 | 43.3 | 2.4 | 28.3 | 0.464 | 184.5 | 0.05 |
| ACC-v1 (margin) | 0.6435 | 31.8 | 3.4 | 24.8 | 0.525 | 223.7 | 0.05 |
| MSP/Chow | 0.6444 | 38.7 | 2.8 | 27.0 | 0.51 | 212.2 | 0.05 |
| entropy | 0.6459 | 49.9 | 2.5 | 31.2 | 0.517 | 211.0 | 0.05 |
| Gini/DOCTOR | 0.6461 | 44.1 | 2.0 | 28.4 | 0.455 | 179.4 | 0.1 |
| AutoMix (self-verify) | 0.6448 | 25.4 | 1.7 | 20.6 | 0.35 | 129.7 | 0.65 |
| FrugalGPT-style learned | 0.6449 | 47.6 | 2.9 | 30.8 | 0.559 | 235.2 | 0.2 |
| Jitkrittum L2D (Diff-Prob) | 0.6403 | 32.0 | 2.3 | 23.6 | 0.413 | 161.3 | 0.2 |
| random | 0.639 | 82.1 | 16.6 | 57.2 | 1.792 | 899.7 | 0.4 |

### Lingshu 7B/32B (Qwen2.5-VL) — ALL-5 (parity acc = 0.7746)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.7339 | 0.0 | 0.0 | 13.5 | 0.131 | 22.9 | 0.0 |
| always-big-nt | 0.7841 | 100.0 | 0.0 | 58.7 | 0.27 | 88.1 | 1.0 |
| always-big-think [PARITY] | 0.7746 | 100.0 | 100.0 | 100.0 | 0.322 | 113.2 | 1.0 |
| Ours (ACC-v2: agreement) | 0.7726 | 42.0 | 0.6 | 38.9 | 0.246 | 60.6 | 1.15 |
| CASP-Stability (trained) | 0.7726 | 47.0 | 0.2 | 41.6 | 0.259 | 64.5 | 1.15 |
| ACC-v1 (margin) | 0.7734 | 43.7 | 0.0 | 39.3 | 0.249 | 61.4 | 1.15 |
| MSP/Chow | 0.7726 | 47.4 | 0.2 | 41.8 | 0.26 | 64.9 | 1.1 |
| entropy | 0.7732 | 51.9 | 0.2 | 44.6 | 0.272 | 68.9 | 1.2 |
| Gini/DOCTOR | 0.7726 | 47.2 | 0.8 | 42.5 | 0.261 | 65.4 | 1.1 |
| AutoMix (self-verify) | 0.7733 | 34.3 | 0.2 | 34.2 | 0.224 | 53.4 | 1.0 |
| FrugalGPT-style learned | 0.774 | 48.3 | 0.0 | 42.2 | 0.261 | 65.4 | 1.1 |
| Jitkrittum L2D (Diff-Prob) | 0.7735 | 32.3 | 0.0 | 32.3 | 0.218 | 51.4 | 1.0 |
| random | 0.7742 | 85.0 | 0.0 | 63.5 | 0.36 | 97.8 | 1.0 |

### QoQ-Med-VL 7B/32B (Qwen2.5-VL) — ALL-5 (parity acc = 0.5432)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.605 | 0.0 | 0.0 | 9.1 | 0.121 | 17.6 | 0.0 |
| always-big-nt | 0.6101 | 100.0 | 0.0 | 39.4 | 0.227 | 68.8 | 1.0 |
| always-big-think [PARITY] | 0.5432 | 100.0 | 100.0 | 100.0 | 8.491 | 4692.0 | 4.0 |
| Ours (ACC-v2: agreement) | 0.6048 | 0.0 | 0.0 | 9.1 | 0.121 | 17.6 | 0.0 |
| CASP-Stability (trained) | 0.6049 | 0.0 | 0.0 | 9.2 | 0.126 | 20.3 | 0.0 |
| ACC-v1 (margin) | 0.6048 | 0.0 | 0.0 | 9.1 | 0.121 | 17.6 | 0.0 |
| MSP/Chow | 0.6047 | 0.0 | 0.0 | 9.1 | 0.126 | 20.2 | 0.3 |
| entropy | 0.6049 | 0.0 | 0.0 | 9.2 | 0.126 | 20.3 | 0.0 |
| Gini/DOCTOR | 0.6047 | 0.0 | 0.0 | 9.1 | 0.125 | 19.9 | 0.35 |
| AutoMix (self-verify) | 0.6048 | 0.0 | 0.0 | 9.1 | 0.123 | 18.7 | 0.05 |
| FrugalGPT-style learned | 0.6048 | 0.1 | 0.1 | 9.2 | 0.127 | 21.0 | 0.0 |
| Jitkrittum L2D (Diff-Prob) | 0.6047 | 0.0 | 0.0 | 9.1 | 0.125 | 19.9 | 0.35 |
| random | 0.6048 | 0.0 | 0.0 | 9.1 | 0.123 | 18.8 | 0.15 |

### Chiron-o1 2B/8B (InternVL3) — ALL-5 (parity acc = 0.5926)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.7252 | 0.0 | 0.0 | 19.9 | 0.195 | 17.4 | 0.0 |
| always-big-nt | 0.6543 | 100.0 | 0.0 | 79.7 | 0.291 | 38.8 | 2.0 |
| always-big-think [PARITY] | 0.5926 | 100.0 | 100.0 | 100.0 | 3.657 | 1005.7 | 4.0 |
| Ours (ACC-v2: agreement) | 0.7249 | 0.0 | 0.0 | 19.9 | 0.195 | 17.4 | 0.0 |
| CASP-Stability (trained) | 0.7248 | 0.0 | 0.0 | 20.0 | 0.197 | 17.7 | 0.1 |
| ACC-v1 (margin) | 0.7249 | 0.0 | 0.0 | 19.9 | 0.195 | 17.4 | 0.0 |
| MSP/Chow | 0.7248 | 0.0 | 0.0 | 20.0 | 0.198 | 18.0 | 0.1 |
| entropy | 0.7249 | 0.0 | 0.0 | 20.0 | 0.196 | 17.6 | 0.0 |
| Gini/DOCTOR | 0.7249 | 0.0 | 0.0 | 20.0 | 0.197 | 17.7 | 0.05 |
| AutoMix (self-verify) | 0.7248 | 0.0 | 0.0 | 20.0 | 0.196 | 17.6 | 0.15 |
| FrugalGPT-style learned | 0.725 | 0.0 | 0.0 | 20.0 | 0.197 | 17.8 | 0.0 |
| Jitkrittum L2D (Diff-Prob) | 0.7249 | 0.0 | 0.0 | 20.0 | 0.196 | 17.5 | 0.0 |
| random | 0.725 | 0.0 | 0.0 | 20.0 | 0.197 | 17.8 | 0.0 |

### MedGemma 4B/27B (Gemma3) — ALL-5 (parity acc = 0.5979)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.6031 | 0.0 | 0.0 | 11.5 | 0.179 | 15.8 | 0.0 |
| always-big-nt | 0.5797 | 100.0 | 0.0 | 72.0 | 0.282 | 63.9 | 3.0 |
| always-big-think [PARITY] | 0.5979 | 100.0 | 100.0 | 100.0 | 9.76 | 4990.1 | 3.0 |
| Ours (ACC-v2: agreement) | 0.6028 | 0.0 | 0.0 | 11.5 | 0.179 | 15.8 | 0.0 |
| CASP-Stability (trained) | 0.6029 | 0.0 | 0.0 | 11.5 | 0.185 | 18.5 | 0.1 |
| ACC-v1 (margin) | 0.6028 | 0.0 | 0.0 | 11.5 | 0.179 | 15.8 | 0.0 |
| MSP/Chow | 0.6028 | 0.0 | 0.0 | 11.5 | 0.179 | 15.8 | 0.0 |
| entropy | 0.6028 | 0.0 | 0.0 | 11.5 | 0.179 | 15.8 | 0.0 |
| Gini/DOCTOR | 0.6028 | 0.0 | 0.0 | 11.5 | 0.179 | 15.8 | 0.0 |
| AutoMix (self-verify) | 0.6027 | 0.0 | 0.0 | 11.5 | 0.181 | 16.7 | 0.1 |
| FrugalGPT-style learned | 0.6029 | 0.0 | 0.0 | 11.5 | 0.183 | 17.9 | 0.05 |
| Jitkrittum L2D (Diff-Prob) | 0.6028 | 0.0 | 0.0 | 11.5 | 0.181 | 16.6 | 0.05 |
| random | 0.6026 | 0.0 | 0.0 | 11.5 | 0.181 | 16.5 | 0.35 |

## 6. Per-benchmark accuracy (baselines + Ours), ALL-6

| family | config | PMC | SLAKE | VQARAD | PathV | MMMU | MX-R | MX-U |
|---|---|---|---|---|---|---|---|---|
| medvlthinker | small-nt | 0.543 | 0.762 | 0.761 | 0.6407 | 0.5471 | 0.2254 | 0.2563 |
| medvlthinker | big-nt | 0.551 | 0.8486 | 0.8529 | 0.6612 | 0.6235 | 0.2787 | 0.2924 |
| medvlthinker | big-think | 0.5565 | 0.7644 | 0.7757 | 0.6725 | 0.6882 | 0.3257 | 0.3845 |
| medvlthinker | Ours | 0.5613 | 0.8416 | 0.8609 | 0.6791 | 0.643 | 0.2821 | 0.3103 |
| lingshu | small-nt | 0.6215 | 0.8413 | 0.7316 | 0.782 | 0.8471 | 0.2469 | 0.278 |
| lingshu | big-nt | 0.64 | 0.8942 | 0.8162 | 0.8614 | 0.6294 | 0.296 | 0.3285 |
| lingshu | big-think | 0.6515 | 0.8846 | 0.7463 | 0.8444 | 0.6176 | 0.2967 | 0.3375 |
| lingshu | Ours | 0.6381 | 0.8631 | 0.8135 | 0.8539 | 0.7122 | 0.2852 | 0.3174 |
| qoq | small-nt | 0.5115 | 0.7212 | 0.7243 | 0.6404 | 0.5294 | 0.2061 | 0.2274 |
| qoq | big-nt | 0.52 | 0.7236 | 0.7426 | 0.6383 | 0.6235 | 0.2324 | 0.2888 |
| qoq | big-think | 0.4355 | 0.6587 | 0.6654 | 0.5756 | 0.6941 | 0.2324 | 0.2509 |
| qoq | Ours | 0.5115 | 0.7212 | 0.7226 | 0.6403 | 0.5287 | 0.2061 | 0.2302 |
| chiron | small-nt | 0.5395 | 0.8197 | 0.75 | 0.8379 | 0.4118 | 0.2042 | 0.2514 |
| chiron | big-nt | 0.596 | 0.8101 | 0.7721 | 0.6645 | 0.5706 | 0.2202 | 0.2477 |
| chiron | big-think | 0.5245 | 0.7019 | 0.6801 | 0.6136 | 0.5706 | 0.2258 | 0.2805 |
| chiron | Ours | 0.5395 | 0.8182 | 0.7518 | 0.8377 | 0.407 | 0.2047 | 0.2536 |
| medgemma | small-nt | 0.462 | 0.8389 | 0.8162 | 0.6457 | 0.5 | 0.2282 | 0.269 |
| medgemma | big-nt | 0.4745 | 0.7933 | 0.7684 | 0.6041 | 0.5118 | 0.249 | 0.2617 |
| medgemma | big-think | 0.4665 | 0.7981 | 0.75 | 0.644 | 0.5 | 0.2891 | 0.3267 |
| medgemma | Ours | 0.4678 | 0.8498 | 0.8243 | 0.642 | 0.5523 | 0.2557 | 0.2788 |

## 7. Over-thinking premise — big-no-think minus big-think accuracy (ALL-6 per benchmark)

Positive = no-think wins (thinking over-thinks perception).

| family | PMC | SLAKE | VQARAD | PathV | MMMU | MX-R | MX-U |
|---|---|---|---|---|---|---|---|
| medvlthinker | -0.005 | +0.084 | +0.077 | -0.011 | -0.065 | -0.047 | -0.092 |
| lingshu | -0.011 | +0.010 | +0.070 | +0.017 | +0.012 | -0.001 | -0.009 |
| qoq | +0.085 | +0.065 | +0.077 | +0.063 | -0.071 | +0.000 | +0.038 |
| chiron | +0.072 | +0.108 | +0.092 | +0.051 | +0.000 | -0.006 | -0.033 |
| medgemma | +0.008 | -0.005 | +0.018 | -0.040 | +0.012 | -0.040 | -0.065 |

## 8. Training-based methods

- **CASP-Stability (signal-trained)**: re-targets the routing label from un-learnable *recoverability* (~0.58 AUROC) to learnable *answer-stability under compute* (~0.71 AUROC); logistic ≈ MLP (capacity doesn't help). Reported as a baseline (not ours). See `casp_stability.txt`, §5.6 of the paper.

- **LoRA-router (needs peft; `lora_stability_router.py`)**: LoRA-fine-tune the 7B as a generative self-verifier of its own answer's stability, from raw image+text. Honest 50/50 (n_train=1429, n_test=1429, 2 epochs):

  | gate | AUROC |
  |---|---|
  | margin (training-free) | 0.6813 |
  | signal-CASP (logistic) | 0.7328 |
  | **LoRA self-verifier (trained)** | 0.7226 |

  **Finding:** LoRA fine-tuning on raw inputs **does NOT beat** the cheap logistic-on-signals gate (Δ -0.0102). The routing ceiling is feature-independent — extra model capacity + raw image/text access does not break it. (Smoke on 84 samples was noise.)

## 9. Synthesis

- ACC's full 3-tier cascade pays off **only when the think tier helps the big model on the pool AND the small→big gap is routable** — true for MedVLThinker (≈5× latency/energy cut at parity, guard-clean), weakly for MedGemma (guard-violating). For Lingshu/QoQ/Chiron think is net-harmful (Chiron inverse-scales 2B>8B) so every method collapses to the cheap leg.

- The robust, transferable lever is the **mode axis (drop think on perception)**. Where the cascade is non-degenerate, **Ours (ACC-v2) and CASP-Stability are the cheapest guardrail-clean gates**; the gate ceiling is fundamental (even LoRA fine-tuning doesn't beat it).

## 10. Artifact index

- `master_data.csv` — every (family×pool×method) row, all metrics + 7 per-benchmark accuracies.
- `MASTER_TABLES.md` — full markdown tables (incl. peak VRAM). `acc_allmethods_all.txt` — raw console.
- `allmethods_<family>.json` — per-family machine-readable (rows + lat/energy fits).
- `acc_2size_all.txt`, `2SIZE_VALIDATION.md` — premise + cascade + methodology narrative.
- `paper/figs/master/*.png` — 5 charts (per-benchmark heatmap; acc-vs-latency; acc-vs-energy; cost bars; ALL-5/ALL-6).
- `ckpts/train/lora_stability/` — LoRA adapter + result.json. `ckpts/acc_gen/<fam>/lat/<tier>/` — batch-1 lat+energy samples.
- Code: `src/cascade_methods/acc_allmethods.py`, `acc_2size.py`, `make_master_charts.py`; `src/training_methods/{casp_stability.py,lora_stability_router.py}`; `src/labeling/{run_vlm_eval,run_peer_eval,run_7b_selfverify_vllm,nvml_power}.py`.

