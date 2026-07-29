# Detailed cascade results — ACC-v1/v2/v3 across families

Honest 50/50 calib/test, 20 seeds, min-think@parity. parity = always-big-think.
Measured batch-1 latency/energy. guard = #benchmarks below always-small.

## medvlthinker

### ALL-6  (parity=0.5723)

| method | acc | esc0 | think | FLOPs% | latency | energy | guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.5262 | 0% | 0% | 8% | 0.13s | 20J | 0.00 |
| always big-nt | 0.5573 | 100% | 0% | 45% | 0.36s | 98J | 0.00 |
| always big-think [parity] | 0.5723 | 100% | 100% | 145% | 11.70s | 6417J | 0.00 |
| ACC-v1 confidence | 0.5694 | 71% | 19% | 55% | 2.61s | 1371J | 0.00 |
| ACC-v2 agreement (baseline) | — | — | — | — | — | — | — |
| ACC-v3 agree+conf (ours) | 0.5707 | 78% | 14% | 53% | 2.13s | 1098J | 0.00 |

| method | PMC | SLAKE | VQARAD | PathV | MMMU | MedX-R | MedX-U |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.543 | 0.762 | 0.761 | 0.641 | 0.547 | 0.225 | 0.256 |
| always big-nt | 0.551 | 0.849 | 0.853 | 0.661 | 0.624 | 0.279 | 0.292 |
| always big-think [parity] | 0.556 | 0.764 | 0.776 | 0.673 | 0.688 | 0.326 | 0.384 |
| ACC-v1 confidence | 0.564 | 0.830 | 0.856 | 0.676 | 0.648 | 0.292 | 0.308 |
| ACC-v2 agreement (baseline) | — | — | — | — | — | — | — |
| ACC-v3 agree+conf (ours) | 0.561 | 0.848 | 0.861 | 0.680 | 0.644 | 0.285 | 0.310 |

### ALL-5  (parity=0.6463)

| method | acc | esc0 | think | FLOPs% | latency | energy | guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.6201 | 0% | 0% | 9% | 0.13s | 20J | 0.00 |
| always big-nt | 0.6457 | 100% | 0% | 48% | 0.36s | 98J | 0.00 |
| always big-think [parity] | 0.6463 | 100% | 100% | 148% | 9.24s | 5014J | 0.00 |
| ACC-v1 confidence | 0.6459 | 46% | 1% | 27% | 0.28s | 83J | 0.10 |
| ACC-v2 agreement (baseline) | 0.6486 | 36% | 16% | 40% | 1.76s | 907J | 0.00 |
| ACC-v3 agree+conf (ours) | 0.6460 | 47% | 0% | 27% | 0.26s | 71J | 0.10 |

| method | PMC | SLAKE | VQARAD | PathV | MMMU | MedX-R | MedX-U |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.543 | 0.762 | 0.761 | 0.641 | 0.547 | — | — |
| always big-nt | 0.551 | 0.849 | 0.853 | 0.661 | 0.624 | — | — |
| always big-think [parity] | 0.556 | 0.764 | 0.776 | 0.673 | 0.688 | — | — |
| ACC-v1 confidence | 0.556 | 0.813 | 0.850 | 0.664 | 0.605 | — | — |
| ACC-v2 agreement (baseline) | 0.555 | 0.792 | 0.842 | 0.672 | 0.620 | — | — |
| ACC-v3 agree+conf (ours) | 0.556 | 0.814 | 0.851 | 0.664 | 0.605 | — | — |

### COMPETENT-4  (parity=0.6451)

| method | acc | esc0 | think | FLOPs% | latency | energy | guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.6221 | 0% | 0% | 9% | 0.13s | 20J | 0.00 |
| always big-nt | 0.6463 | 100% | 0% | 48% | 0.36s | 98J | 0.00 |
| always big-think [parity] | 0.6451 | 100% | 100% | 148% | 9.11s | 4936J | 0.00 |
| ACC-v1 confidence | 0.6449 | 37% | 0% | 24% | 0.24s | 62J | 0.10 |
| ACC-v2 agreement (baseline) | 0.6469 | 32% | 15% | 36% | 1.55s | 790J | 0.00 |
| ACC-v3 agree+conf (ours) | 0.6455 | 38% | 0% | 24% | 0.23s | 55J | 0.10 |

| method | PMC | SLAKE | VQARAD | PathV | MMMU | MedX-R | MedX-U |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.543 | 0.762 | 0.761 | 0.641 | — | — | — |
| always big-nt | 0.551 | 0.849 | 0.853 | 0.661 | — | — | — |
| always big-think [parity] | 0.556 | 0.764 | 0.776 | 0.673 | — | — | — |
| ACC-v1 confidence | 0.553 | 0.808 | 0.840 | 0.663 | — | — | — |
| ACC-v2 agreement (baseline) | 0.553 | 0.791 | 0.831 | 0.670 | — | — | — |
| ACC-v3 agree+conf (ours) | 0.554 | 0.809 | 0.842 | 0.664 | — | — | — |

## lingshu

### ALL-6  (parity=0.6611)

| method | acc | esc0 | think | FLOPs% | latency | energy | guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.6175 | 0% | 0% | 13% | 0.13s | 23J | 0.00 |
| always big-nt | 0.6675 | 100% | 0% | 70% | 0.40s | 111J | 1.00 |
| always big-think [parity] | 0.6611 | 100% | 100% | 170% | 0.72s | 224J | 1.00 |
| ACC-v1 confidence | 0.6614 | 61% | 0% | 49% | 0.29s | 76J | 1.00 |
| ACC-v2 agreement (baseline) | 0.6618 | 64% | 24% | 78% | 0.38s | 107J | 1.00 |
| ACC-v3 agree+conf (ours) | 0.6614 | 61% | 0% | 49% | 0.29s | 76J | 1.00 |

| method | PMC | SLAKE | VQARAD | PathV | MMMU | MedX-R | MedX-U |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.622 | 0.841 | 0.732 | 0.782 | 0.847 | 0.247 | 0.278 |
| always big-nt | 0.640 | 0.894 | 0.816 | 0.861 | 0.629 | 0.296 | 0.329 |
| always big-think [parity] | 0.651 | 0.885 | 0.746 | 0.844 | 0.618 | 0.297 | 0.338 |
| ACC-v1 confidence | 0.638 | 0.863 | 0.814 | 0.854 | 0.712 | 0.285 | 0.317 |
| ACC-v2 agreement (baseline) | 0.650 | 0.858 | 0.797 | 0.848 | 0.711 | 0.290 | 0.314 |
| ACC-v3 agree+conf (ours) | 0.638 | 0.863 | 0.814 | 0.854 | 0.712 | 0.285 | 0.317 |

### ALL-5  (parity=0.7746)

| method | acc | esc0 | think | FLOPs% | latency | energy | guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.7339 | 0% | 0% | 14% | 0.13s | 23J | 0.00 |
| always big-nt | 0.7841 | 100% | 0% | 72% | 0.40s | 111J | 1.00 |
| always big-think [parity] | 0.7746 | 100% | 100% | 172% | 0.72s | 224J | 1.00 |
| ACC-v1 confidence | 0.7734 | 44% | 0% | 39% | 0.25s | 61J | 1.15 |
| ACC-v2 agreement (baseline) | 0.7720 | 45% | 16% | 56% | 0.30s | 80J | 1.30 |
| ACC-v3 agree+conf (ours) | 0.7734 | 44% | 0% | 39% | 0.25s | 61J | 1.15 |

| method | PMC | SLAKE | VQARAD | PathV | MMMU | MedX-R | MedX-U |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.622 | 0.841 | 0.732 | 0.782 | 0.847 | — | — |
| always big-nt | 0.640 | 0.894 | 0.816 | 0.861 | 0.629 | — | — |
| always big-think [parity] | 0.651 | 0.885 | 0.746 | 0.844 | 0.618 | — | — |
| ACC-v1 confidence | 0.636 | 0.852 | 0.795 | 0.844 | 0.759 | — | — |
| ACC-v2 agreement (baseline) | 0.644 | 0.844 | 0.787 | 0.839 | 0.759 | — | — |
| ACC-v3 agree+conf (ours) | 0.636 | 0.852 | 0.795 | 0.844 | 0.759 | — | — |

### COMPETENT-4  (parity=0.7790)

| method | acc | esc0 | think | FLOPs% | latency | energy | guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.7307 | 0% | 0% | 13% | 0.13s | 23J | 0.00 |
| always big-nt | 0.7884 | 100% | 0% | 72% | 0.40s | 111J | 0.00 |
| always big-think [parity] | 0.7790 | 100% | 100% | 172% | 0.72s | 224J | 0.00 |
| ACC-v1 confidence | 0.7778 | 51% | 0% | 43% | 0.27s | 68J | 0.10 |
| ACC-v2 agreement (baseline) | 0.7779 | 55% | 18% | 64% | 0.34s | 92J | 0.10 |
| ACC-v3 agree+conf (ours) | 0.7778 | 51% | 0% | 43% | 0.27s | 68J | 0.10 |

| method | PMC | SLAKE | VQARAD | PathV | MMMU | MedX-R | MedX-U |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.622 | 0.841 | 0.732 | 0.782 | — | — | — |
| always big-nt | 0.640 | 0.894 | 0.816 | 0.861 | — | — | — |
| always big-think [parity] | 0.651 | 0.885 | 0.746 | 0.844 | — | — | — |
| ACC-v1 confidence | 0.637 | 0.852 | 0.800 | 0.851 | — | — | — |
| ACC-v2 agreement (baseline) | 0.649 | 0.848 | 0.788 | 0.845 | — | — | — |
| ACC-v3 agree+conf (ours) | 0.637 | 0.852 | 0.800 | 0.851 | — | — | — |

## qoq

### ALL-6  (parity=0.4689)

| method | acc | esc0 | think | FLOPs% | latency | energy | guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.5094 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |
| always big-nt | 0.5220 | 100% | 0% | 47% | 0.35s | 86J | 1.00 |
| always big-think [parity] | 0.4689 | 100% | 100% | 147% | 10.07s | 5468J | 4.00 |
| ACC-v1 confidence | 0.5095 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |
| ACC-v2 agreement (baseline) | 0.5095 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |
| ACC-v3 agree+conf (ours) | 0.5095 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |

| method | PMC | SLAKE | VQARAD | PathV | MMMU | MedX-R | MedX-U |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.511 | 0.721 | 0.724 | 0.640 | 0.529 | 0.206 | 0.227 |
| always big-nt | 0.520 | 0.724 | 0.743 | 0.638 | 0.624 | 0.232 | 0.289 |
| always big-think [parity] | 0.435 | 0.659 | 0.665 | 0.576 | 0.694 | 0.232 | 0.251 |
| ACC-v1 confidence | 0.511 | 0.721 | 0.723 | 0.640 | 0.529 | 0.206 | 0.230 |
| ACC-v2 agreement (baseline) | 0.511 | 0.721 | 0.723 | 0.640 | 0.529 | 0.206 | 0.230 |
| ACC-v3 agree+conf (ours) | 0.511 | 0.721 | 0.723 | 0.640 | 0.529 | 0.206 | 0.230 |

### ALL-5  (parity=0.5432)

| method | acc | esc0 | think | FLOPs% | latency | energy | guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.6050 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |
| always big-nt | 0.6101 | 100% | 0% | 48% | 0.35s | 86J | 1.00 |
| always big-think [parity] | 0.5432 | 100% | 100% | 148% | 8.84s | 4778J | 4.00 |
| ACC-v1 confidence | 0.6048 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |
| ACC-v2 agreement (baseline) | 0.6048 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |
| ACC-v3 agree+conf (ours) | 0.6048 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |

| method | PMC | SLAKE | VQARAD | PathV | MMMU | MedX-R | MedX-U |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.511 | 0.721 | 0.724 | 0.640 | 0.529 | — | — |
| always big-nt | 0.520 | 0.724 | 0.743 | 0.638 | 0.624 | — | — |
| always big-think [parity] | 0.435 | 0.659 | 0.665 | 0.576 | 0.694 | — | — |
| ACC-v1 confidence | 0.511 | 0.721 | 0.723 | 0.640 | 0.529 | — | — |
| ACC-v2 agreement (baseline) | 0.511 | 0.721 | 0.723 | 0.640 | 0.529 | — | — |
| ACC-v3 agree+conf (ours) | 0.511 | 0.721 | 0.723 | 0.640 | 0.529 | — | — |

### COMPETENT-4  (parity=0.5390)

| method | acc | esc0 | think | FLOPs% | latency | energy | guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.6071 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |
| always big-nt | 0.6098 | 100% | 0% | 49% | 0.35s | 86J | 1.00 |
| always big-think [parity] | 0.5390 | 100% | 100% | 149% | 8.77s | 4743J | 4.00 |
| ACC-v1 confidence | 0.6070 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |
| ACC-v2 agreement (baseline) | 0.6070 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |
| ACC-v3 agree+conf (ours) | 0.6070 | 0% | 0% | 9% | 0.12s | 18J | 0.00 |

| method | PMC | SLAKE | VQARAD | PathV | MMMU | MedX-R | MedX-U |
|---|---:|---:|---:|---:|---:|---:|---:|
| always small-nt | 0.511 | 0.721 | 0.724 | 0.640 | — | — | — |
| always big-nt | 0.520 | 0.724 | 0.743 | 0.638 | — | — | — |
| always big-think [parity] | 0.435 | 0.659 | 0.665 | 0.576 | — | — | — |
| ACC-v1 confidence | 0.511 | 0.721 | 0.723 | 0.640 | — | — | — |
| ACC-v2 agreement (baseline) | 0.511 | 0.721 | 0.723 | 0.640 | — | — | — |
| ACC-v3 agree+conf (ours) | 0.511 | 0.721 | 0.723 | 0.640 | — | — | — |
