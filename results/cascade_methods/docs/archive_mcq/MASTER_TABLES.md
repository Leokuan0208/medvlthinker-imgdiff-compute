# Master tables — every data point (5 families × all methods)

Cost legend: FLOPs% vs always-big-think; latency = batch-1 (s); energy = batch-1 (J); guard = #benchmarks worse than always-small (0=clean). Source: `acc_allmethods.py` JSON dumps + NVML logs.

## Peak VRAM (GB, batch-1)

| family | small | big-nt | big-think |
|---|---|---|---|
| MedVLThinker 7B/32B | 71.45 | 152.82 | 154.08 |
| Lingshu 7B/32B | 71.45 | 152.82 | 154.08 |
| QoQ-Med 7B/32B | 71.45 | 152.82 | 154.08 |
| Chiron-o1 2B/8B | 53.46 | 74.16 | 74.17 |
| MedGemma 4B/27B | 52.35 | 155.52 | 155.52 |

## MedVLThinker 7B/32B — ALL-6 (parity=0.5723)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.5262 | 0 | 0 | 8.4 | 0.13 | 19.9 | 0.00 |
| always-big-nt | 0.5573 | 100 | 0 | 36.2 | 0.23 | 77.8 | 0.00 |
| always-big-think [PARITY] | 0.5723 | 100 | 100 | 100.0 | 11.34 | 6318.8 | 0.00 |
| Ours (ACC-v2: agreement) | 0.5693 | 72 | 15 | 52.0 | 2.27 | 1181.9 | 0.00 |
| CASP-Stability (trained) | 0.5698 | 74 | 11 | 49.0 | 1.77 | 899.2 | 0.05 |
| ACC-v1 (margin) | 0.5687 | 66 | 19 | 53.9 | 2.69 | 1416.6 | 0.00 |
| MSP/Chow | 0.5697 | 70 | 19 | 57.4 | 2.96 | 1568.1 | 0.00 |
| entropy | 0.5691 | 71 | 21 | 62.0 | 3.48 | 1863.1 | 0.00 |
| Gini/DOCTOR | 0.5702 | 69 | 22 | 61.0 | 3.44 | 1837.1 | 0.00 |
| AutoMix (self-verify) | 0.5692 | 73 | 18 | 54.6 | 2.50 | 1307.0 | 0.05 |
| FrugalGPT-style learned | 0.5677 | 70 | 19 | 60.4 | 3.30 | 1765.5 | 0.10 |
| Jitkrittum L2D (Diff-Prob) | 0.5666 | 67 | 15 | 50.6 | 2.29 | 1194.5 | 0.00 |
| random | 0.5641 | 89 | 76 | 116.5 | 8.95 | 4889.5 | 0.05 |

## MedVLThinker 7B/32B — ALL-5 (parity=0.6463)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.6201 | 0 | 0 | 8.9 | 0.13 | 19.9 | 0.00 |
| always-big-nt | 0.6457 | 100 | 0 | 38.7 | 0.23 | 77.8 | 0.00 |
| always-big-think [PARITY] | 0.6463 | 100 | 100 | 100.0 | 8.88 | 4915.9 | 0.00 |
| Ours (ACC-v2: agreement) | 0.6450 | 35 | 2 | 24.9 | 0.44 | 172.8 | 0.05 |
| CASP-Stability (trained) | 0.6461 | 43 | 2 | 28.3 | 0.46 | 184.5 | 0.05 |
| ACC-v1 (margin) | 0.6435 | 32 | 3 | 24.8 | 0.53 | 223.7 | 0.05 |
| MSP/Chow | 0.6444 | 39 | 3 | 27.0 | 0.51 | 212.2 | 0.05 |
| entropy | 0.6459 | 50 | 3 | 31.2 | 0.52 | 211.0 | 0.05 |
| Gini/DOCTOR | 0.6461 | 44 | 2 | 28.4 | 0.46 | 179.4 | 0.10 |
| AutoMix (self-verify) | 0.6448 | 25 | 2 | 20.6 | 0.35 | 129.7 | 0.65 |
| FrugalGPT-style learned | 0.6449 | 48 | 3 | 30.8 | 0.56 | 235.2 | 0.20 |
| Jitkrittum L2D (Diff-Prob) | 0.6403 | 32 | 2 | 23.6 | 0.41 | 161.3 | 0.20 |
| random | 0.6390 | 82 | 17 | 57.2 | 1.79 | 899.7 | 0.40 |

## Lingshu 7B/32B — ALL-6 (parity=0.6611)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.6175 | 0 | 0 | 13.1 | 0.13 | 22.9 | 0.00 |
| always-big-nt | 0.6675 | 100 | 0 | 56.8 | 0.27 | 88.1 | 1.00 |
| always-big-think [PARITY] | 0.6611 | 100 | 100 | 100.0 | 0.32 | 113.2 | 1.00 |
| Ours (ACC-v2: agreement) | 0.6614 | 61 | 0 | 48.6 | 0.29 | 76.3 | 1.00 |
| CASP-Stability (trained) | 0.6613 | 61 | 0 | 50.0 | 0.30 | 77.0 | 1.05 |
| ACC-v1 (margin) | 0.6614 | 61 | 0 | 48.6 | 0.29 | 76.3 | 1.00 |
| MSP/Chow | 0.6611 | 62 | 0 | 49.9 | 0.30 | 77.6 | 1.00 |
| entropy | 0.6613 | 64 | 0 | 51.3 | 0.30 | 79.5 | 1.00 |
| Gini/DOCTOR | 0.6615 | 64 | 0 | 50.8 | 0.30 | 78.9 | 1.05 |
| AutoMix (self-verify) | 0.6605 | 59 | 0 | 48.0 | 0.29 | 74.8 | 1.00 |
| FrugalGPT-style learned | 0.6614 | 68 | 0 | 54.1 | 0.32 | 83.0 | 1.00 |
| Jitkrittum L2D (Diff-Prob) | 0.6595 | 49 | 0 | 40.4 | 0.27 | 66.7 | 1.10 |
| random | 0.6608 | 91 | 0 | 65.0 | 0.38 | 103.5 | 1.00 |

## Lingshu 7B/32B — ALL-5 (parity=0.7746)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.7339 | 0 | 0 | 13.5 | 0.13 | 22.9 | 0.00 |
| always-big-nt | 0.7841 | 100 | 0 | 58.7 | 0.27 | 88.1 | 1.00 |
| always-big-think [PARITY] | 0.7746 | 100 | 100 | 100.0 | 0.32 | 113.2 | 1.00 |
| Ours (ACC-v2: agreement) | 0.7726 | 42 | 1 | 38.9 | 0.25 | 60.6 | 1.15 |
| CASP-Stability (trained) | 0.7726 | 47 | 0 | 41.6 | 0.26 | 64.5 | 1.15 |
| ACC-v1 (margin) | 0.7734 | 44 | 0 | 39.3 | 0.25 | 61.4 | 1.15 |
| MSP/Chow | 0.7726 | 47 | 0 | 41.8 | 0.26 | 64.9 | 1.10 |
| entropy | 0.7732 | 52 | 0 | 44.6 | 0.27 | 68.9 | 1.20 |
| Gini/DOCTOR | 0.7726 | 47 | 1 | 42.5 | 0.26 | 65.4 | 1.10 |
| AutoMix (self-verify) | 0.7733 | 34 | 0 | 34.2 | 0.22 | 53.4 | 1.00 |
| FrugalGPT-style learned | 0.7740 | 48 | 0 | 42.2 | 0.26 | 65.4 | 1.10 |
| Jitkrittum L2D (Diff-Prob) | 0.7735 | 32 | 0 | 32.3 | 0.22 | 51.4 | 1.00 |
| random | 0.7742 | 85 | 0 | 63.5 | 0.36 | 97.8 | 1.00 |

## QoQ-Med 7B/32B — ALL-6 (parity=0.4689)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.5094 | 0 | 0 | 8.8 | 0.12 | 17.6 | 0.00 |
| always-big-nt | 0.5220 | 100 | 0 | 38.3 | 0.23 | 68.8 | 1.00 |
| always-big-think [PARITY] | 0.4689 | 100 | 100 | 100.0 | 9.72 | 5381.6 | 4.00 |
| Ours (ACC-v2: agreement) | 0.5095 | 0 | 0 | 8.8 | 0.12 | 17.6 | 0.00 |
| CASP-Stability (trained) | 0.5095 | 0 | 0 | 8.9 | 0.12 | 18.8 | 0.05 |
| ACC-v1 (margin) | 0.5095 | 0 | 0 | 8.8 | 0.12 | 17.6 | 0.00 |
| MSP/Chow | 0.5094 | 0 | 0 | 8.9 | 0.12 | 19.0 | 0.15 |
| entropy | 0.5095 | 0 | 0 | 8.9 | 0.12 | 18.8 | 0.00 |
| Gini/DOCTOR | 0.5095 | 0 | 0 | 8.9 | 0.12 | 19.0 | 0.00 |
| AutoMix (self-verify) | 0.5095 | 0 | 0 | 8.9 | 0.12 | 18.7 | 0.00 |
| FrugalGPT-style learned | 0.5095 | 0 | 0 | 8.9 | 0.13 | 19.8 | 0.05 |
| Jitkrittum L2D (Diff-Prob) | 0.5095 | 0 | 0 | 8.9 | 0.12 | 19.0 | 0.00 |
| random | 0.5095 | 0 | 0 | 8.9 | 0.12 | 18.6 | 0.05 |

## QoQ-Med 7B/32B — ALL-5 (parity=0.5432)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.6050 | 0 | 0 | 9.1 | 0.12 | 17.6 | 0.00 |
| always-big-nt | 0.6101 | 100 | 0 | 39.4 | 0.23 | 68.8 | 1.00 |
| always-big-think [PARITY] | 0.5432 | 100 | 100 | 100.0 | 8.49 | 4692.0 | 4.00 |
| Ours (ACC-v2: agreement) | 0.6048 | 0 | 0 | 9.1 | 0.12 | 17.6 | 0.00 |
| CASP-Stability (trained) | 0.6049 | 0 | 0 | 9.2 | 0.13 | 20.3 | 0.00 |
| ACC-v1 (margin) | 0.6048 | 0 | 0 | 9.1 | 0.12 | 17.6 | 0.00 |
| MSP/Chow | 0.6047 | 0 | 0 | 9.1 | 0.13 | 20.2 | 0.30 |
| entropy | 0.6049 | 0 | 0 | 9.2 | 0.13 | 20.3 | 0.00 |
| Gini/DOCTOR | 0.6047 | 0 | 0 | 9.1 | 0.13 | 19.9 | 0.35 |
| AutoMix (self-verify) | 0.6048 | 0 | 0 | 9.1 | 0.12 | 18.7 | 0.05 |
| FrugalGPT-style learned | 0.6048 | 0 | 0 | 9.2 | 0.13 | 21.0 | 0.00 |
| Jitkrittum L2D (Diff-Prob) | 0.6047 | 0 | 0 | 9.1 | 0.13 | 19.9 | 0.35 |
| random | 0.6048 | 0 | 0 | 9.1 | 0.12 | 18.8 | 0.15 |

## Chiron-o1 2B/8B — ALL-6 (parity=0.5076)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.6024 | 0 | 0 | 19.3 | 0.20 | 17.4 | 0.00 |
| always-big-nt | 0.5512 | 100 | 0 | 77.0 | 0.29 | 38.8 | 3.00 |
| always-big-think [PARITY] | 0.5076 | 100 | 100 | 100.0 | 4.25 | 1175.6 | 4.00 |
| Ours (ACC-v2: agreement) | 0.6023 | 0 | 0 | 19.3 | 0.20 | 17.4 | 0.00 |
| CASP-Stability (trained) | 0.6023 | 0 | 0 | 19.3 | 0.20 | 17.8 | 0.00 |
| ACC-v1 (margin) | 0.6023 | 0 | 0 | 19.3 | 0.20 | 17.4 | 0.00 |
| MSP/Chow | 0.6023 | 0 | 0 | 19.3 | 0.20 | 17.7 | 0.00 |
| entropy | 0.6023 | 0 | 0 | 19.3 | 0.20 | 17.6 | 0.05 |
| Gini/DOCTOR | 0.6023 | 0 | 0 | 19.3 | 0.20 | 17.5 | 0.05 |
| AutoMix (self-verify) | 0.6023 | 0 | 0 | 19.3 | 0.20 | 17.7 | 0.20 |
| FrugalGPT-style learned | 0.6024 | 0 | 0 | 19.3 | 0.20 | 17.9 | 0.15 |
| Jitkrittum L2D (Diff-Prob) | 0.6023 | 0 | 0 | 19.3 | 0.20 | 17.4 | 0.00 |
| random | 0.6025 | 0 | 0 | 19.3 | 0.20 | 17.7 | 0.00 |

## Chiron-o1 2B/8B — ALL-5 (parity=0.5926)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.7252 | 0 | 0 | 19.9 | 0.20 | 17.4 | 0.00 |
| always-big-nt | 0.6543 | 100 | 0 | 79.7 | 0.29 | 38.8 | 2.00 |
| always-big-think [PARITY] | 0.5926 | 100 | 100 | 100.0 | 3.66 | 1005.7 | 4.00 |
| Ours (ACC-v2: agreement) | 0.7249 | 0 | 0 | 19.9 | 0.20 | 17.4 | 0.00 |
| CASP-Stability (trained) | 0.7248 | 0 | 0 | 20.0 | 0.20 | 17.7 | 0.10 |
| ACC-v1 (margin) | 0.7249 | 0 | 0 | 19.9 | 0.20 | 17.4 | 0.00 |
| MSP/Chow | 0.7248 | 0 | 0 | 20.0 | 0.20 | 18.0 | 0.10 |
| entropy | 0.7249 | 0 | 0 | 20.0 | 0.20 | 17.6 | 0.00 |
| Gini/DOCTOR | 0.7249 | 0 | 0 | 20.0 | 0.20 | 17.7 | 0.05 |
| AutoMix (self-verify) | 0.7248 | 0 | 0 | 20.0 | 0.20 | 17.6 | 0.15 |
| FrugalGPT-style learned | 0.7250 | 0 | 0 | 20.0 | 0.20 | 17.8 | 0.00 |
| Jitkrittum L2D (Diff-Prob) | 0.7249 | 0 | 0 | 20.0 | 0.20 | 17.5 | 0.00 |
| random | 0.7250 | 0 | 0 | 20.0 | 0.20 | 17.8 | 0.00 |

## MedGemma 4B/27B — ALL-6 (parity=0.5253)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.5146 | 0 | 0 | 10.6 | 0.18 | 15.8 | 0.00 |
| always-big-nt | 0.5001 | 100 | 0 | 66.3 | 0.28 | 63.9 | 4.00 |
| always-big-think [PARITY] | 0.5253 | 100 | 100 | 100.0 | 12.72 | 6534.5 | 3.00 |
| Ours (ACC-v2: agreement) | 0.5219 | 55 | 20 | 68.4 | 3.37 | 1613.5 | 1.15 |
| CASP-Stability (trained) | 0.5223 | 22 | 4 | 29.6 | 1.05 | 446.1 | 1.15 |
| ACC-v1 (margin) | 0.5207 | 62 | 41 | 93.8 | 5.87 | 2891.7 | 2.70 |
| MSP/Chow | 0.5230 | 76 | 56 | 117.5 | 7.85 | 3897.9 | 2.95 |
| entropy | 0.5229 | 76 | 56 | 117.9 | 7.88 | 3914.0 | 3.10 |
| Gini/DOCTOR | 0.5229 | 76 | 56 | 117.7 | 7.86 | 3904.5 | 3.00 |
| AutoMix (self-verify) | 0.5190 | 18 | 6 | 28.1 | 0.90 | 368.3 | 1.55 |
| FrugalGPT-style learned | 0.5233 | 18 | 4 | 28.1 | 1.15 | 501.6 | 1.30 |
| Jitkrittum L2D (Diff-Prob) | 0.5199 | 13 | 3 | 22.1 | 0.70 | 274.0 | 1.45 |
| random | 0.5199 | 79 | 71 | 134.4 | 9.50 | 4738.3 | 3.70 |

## MedGemma 4B/27B — ALL-5 (parity=0.5979)

| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |
|---|---|---|---|---|---|---|---|
| always-small-nt (cheap) | 0.6031 | 0 | 0 | 11.5 | 0.18 | 15.8 | 0.00 |
| always-big-nt | 0.5797 | 100 | 0 | 72.0 | 0.28 | 63.9 | 3.00 |
| always-big-think [PARITY] | 0.5979 | 100 | 100 | 100.0 | 9.76 | 4990.1 | 3.00 |
| Ours (ACC-v2: agreement) | 0.6028 | 0 | 0 | 11.5 | 0.18 | 15.8 | 0.00 |
| CASP-Stability (trained) | 0.6029 | 0 | 0 | 11.5 | 0.18 | 18.5 | 0.10 |
| ACC-v1 (margin) | 0.6028 | 0 | 0 | 11.5 | 0.18 | 15.8 | 0.00 |
| MSP/Chow | 0.6028 | 0 | 0 | 11.5 | 0.18 | 15.8 | 0.00 |
| entropy | 0.6028 | 0 | 0 | 11.5 | 0.18 | 15.8 | 0.00 |
| Gini/DOCTOR | 0.6028 | 0 | 0 | 11.5 | 0.18 | 15.8 | 0.00 |
| AutoMix (self-verify) | 0.6027 | 0 | 0 | 11.5 | 0.18 | 16.7 | 0.10 |
| FrugalGPT-style learned | 0.6029 | 0 | 0 | 11.5 | 0.18 | 17.9 | 0.05 |
| Jitkrittum L2D (Diff-Prob) | 0.6028 | 0 | 0 | 11.5 | 0.18 | 16.6 | 0.05 |
| random | 0.6026 | 0 | 0 | 11.5 | 0.18 | 16.5 | 0.35 |

## Per-benchmark accuracy (ALL-6): baselines + Ours

| family | config | PMC | SLAKE | VQARAD | PathV | MMMU | MX-R | MX-U |
|---|---|---|---|---|---|---|---|---|
| medvlthinker | always-small-nt | 0.543 | 0.762 | 0.761 | 0.641 | 0.547 | 0.225 | 0.256 |
| medvlthinker | always-big-nt | 0.551 | 0.849 | 0.853 | 0.661 | 0.624 | 0.279 | 0.292 |
| medvlthinker | always-big-think | 0.556 | 0.764 | 0.776 | 0.673 | 0.688 | 0.326 | 0.384 |
| medvlthinker | Ours | 0.561 | 0.842 | 0.861 | 0.679 | 0.643 | 0.282 | 0.310 |
| lingshu | always-small-nt | 0.622 | 0.841 | 0.732 | 0.782 | 0.847 | 0.247 | 0.278 |
| lingshu | always-big-nt | 0.640 | 0.894 | 0.816 | 0.861 | 0.629 | 0.296 | 0.329 |
| lingshu | always-big-think | 0.651 | 0.885 | 0.746 | 0.844 | 0.618 | 0.297 | 0.338 |
| lingshu | Ours | 0.638 | 0.863 | 0.814 | 0.854 | 0.712 | 0.285 | 0.317 |
| qoq | always-small-nt | 0.511 | 0.721 | 0.724 | 0.640 | 0.529 | 0.206 | 0.227 |
| qoq | always-big-nt | 0.520 | 0.724 | 0.743 | 0.638 | 0.624 | 0.232 | 0.289 |
| qoq | always-big-think | 0.435 | 0.659 | 0.665 | 0.576 | 0.694 | 0.232 | 0.251 |
| qoq | Ours | 0.511 | 0.721 | 0.723 | 0.640 | 0.529 | 0.206 | 0.230 |
| chiron | always-small-nt | 0.539 | 0.820 | 0.750 | 0.838 | 0.412 | 0.204 | 0.251 |
| chiron | always-big-nt | 0.596 | 0.810 | 0.772 | 0.664 | 0.571 | 0.220 | 0.248 |
| chiron | always-big-think | 0.524 | 0.702 | 0.680 | 0.614 | 0.571 | 0.226 | 0.281 |
| chiron | Ours | 0.539 | 0.818 | 0.752 | 0.838 | 0.407 | 0.205 | 0.254 |
| medgemma | always-small-nt | 0.462 | 0.839 | 0.816 | 0.646 | 0.500 | 0.228 | 0.269 |
| medgemma | always-big-nt | 0.474 | 0.793 | 0.768 | 0.604 | 0.512 | 0.249 | 0.262 |
| medgemma | always-big-think | 0.467 | 0.798 | 0.750 | 0.644 | 0.500 | 0.289 | 0.327 |
| medgemma | Ours | 0.468 | 0.850 | 0.824 | 0.642 | 0.552 | 0.256 | 0.279 |

## Open-ended routing (§5.7) — Lingshu-7B → Lingshu-32B, LLM-judge

Routing AUROC (predict the cheap model is wrong). MCQ ceiling ≈ 0.6 (§5.2).

| dataset | confidence cheap-wrong | confidence recover |
|---|---|---|
| SLAKE | 0.889 | 0.847 |
| VQA-RAD | 0.717 | 0.579 |
| PathVQA | 0.797 | 0.517 |
| pooled | 0.846 | 0.591 |

### Gate hunt (Lingshu-7B->Lingshu-32B, n=845) — no signal beats confidence

| signal | cheap-wrong AUROC |
|---|---|
| confidence | 0.866 |
| exact-SC | 0.845 |
| semantic-SC | 0.806 |
| sem-entropy | 0.807 |
| mean-F1 | 0.844 |
| P(True) | 0.755 |
| fusion(all) | 0.866 |

### Open-ended model accuracy — exact-match vs **LLM-judge** column (neutral MedVLThinker-32B grader)

| dataset | n | Lingshu-7B (cheap) EM | Lingshu-7B JUDGE | Lingshu-32B (strong) EM | Lingshu-32B JUDGE |
|---|---|---|---|---|---|
| SLAKE | 645 | 0.763 | 0.730 | 0.847 | 0.819 |
| VQA-RAD | 200 | 0.425 | 0.490 | 0.545 | 0.600 |
| PathVQA | 1500 | 0.327 | 0.343 | 0.344 | 0.376 |
| Kvasir(GI) | 1200 | 0.273 | 0.302 | 0.230 | 0.301 |

*(PathVQA/Kvasir exact-match is uninformative on long descriptive answers; the LLM-judge column is the meaningful score. EM≈JUDGE on short-answer SLAKE/VQA-RAD. On Kvasir (GI endoscopy, OOD) the 32B is no better than the 7B (0.301 vs 0.302) — yet the cheap model's confidence still detects its own errors at AUROC 0.75 (§5.7): detection ≠ cascade gain.)*
