#!/bin/bash
# ACC-generalization campaign on Lingshu 7B/32B (Qwen2.5-VL medical pair).
# Phase 1 (parallel, both GPUs): 7B no-think resolution sweep cap80/160/320/640/fullres -> sweet spot.
# Phase 2 (TP=2): 32B no-think@cap320, no-think@fullres, think@fullres.
# Competent-4 + MMMU. Resumable (run_vlm_eval skips done idx). Launch from repo root.
set -u
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache
M7=lingshu-medical-mllm/Lingshu-7B
M32=lingshu-medical-mllm/Lingshu-32B
DS="PMC-VQA SLAKE VQA-RAD PathVQA MMMU MedXpert-Reasoning MedXpert-Understanding"
TS="You are an expert medical AI. Reason step by step about the image and the question, then end your response with a line 'Answer: X' where X is the correct option letter."

echo "=== PHASE 1: Lingshu-7B no-think resolution sweep (both GPUs) ==="
( for cap in cap80 cap320 fullres; do
    CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_vlm_eval.py --model_path $M7 --tag lingshu7b \
      --arm nothink --cap $cap --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/lingshu7b/$cap --tp 1 --gpu_mem 0.85
  done; echo "GPU0_7B_DONE" ) > logs/lingshu_7b_a.log 2>&1 &
A=$!
( for cap in cap160 cap640; do
    CUDA_VISIBLE_DEVICES=1 python3 src/labeling/run_vlm_eval.py --model_path $M7 --tag lingshu7b \
      --arm nothink --cap $cap --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/lingshu7b/$cap --tp 1 --gpu_mem 0.85
  done; echo "GPU1_7B_DONE" ) > logs/lingshu_7b_b.log 2>&1 &
B=$!
wait $A $B
echo "=== PHASE1_DONE 7B sweep ==="

echo "=== PHASE 2: Lingshu-32B (TP=2) ==="
python3 src/labeling/run_vlm_eval.py --model_path $M32 --tag lingshu32b --arm nothink --cap cap320 \
  --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/lingshu32b/nothink_cap320 --tp 2 --gpu_mem 0.90 > logs/lingshu_32b_ntcap320.log 2>&1
python3 src/labeling/run_vlm_eval.py --model_path $M32 --tag lingshu32b --arm nothink --cap fullres \
  --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/lingshu32b/nothink_fullres --tp 2 --gpu_mem 0.90 > logs/lingshu_32b_ntfull.log 2>&1
python3 src/labeling/run_vlm_eval.py --model_path $M32 --tag lingshu32b --arm think --system "$TS" --cap fullres \
  --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/lingshu32b/think_fullres --tp 2 --gpu_mem 0.90 --max_tokens 1536 > logs/lingshu_32b_think.log 2>&1
echo "=== ALL_DONE Lingshu ACC campaign ==="
