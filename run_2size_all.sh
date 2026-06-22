#!/bin/bash
# Sequential 2-size medical ACC validation: Lingshu (Qwen2.5-VL, resolution sweep) then MedGemma (Gemma3).
export HF_HUB_OFFLINE=1
cd ~/medvlthinker-imgdiff-compute
echo "===== LINGSHU CAMPAIGN START $(date +%H:%M) ====="
bash run_lingshu_acc.sh
echo "===== MEDGEMMA CAMPAIGN START $(date +%H:%M) ====="
bash run_medgemma_acc.sh
echo "===== ALL_2SIZE_DONE $(date +%H:%M) ====="
