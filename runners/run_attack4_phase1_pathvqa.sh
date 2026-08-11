#!/bin/bash
# ATTACK 4 (OPEN-DIVERSE) Phase 1 -- extend the M=15 portfolio pool from the unrepresentative 178-item
# PathVQA-open slice to the FULL 1500-item reporting cell.
#
# WHY: ckpts/openvqa/diverse/ckpt_pathvqa_open_lingshu7b_div.jsonl covers 178/1500 items, and that slice
# is far harder than the cell (measured on the incumbent pool: greedy 0.1348 vs 0.3240, oracle@8 0.3371
# vs 0.5167, sel_eff 0.4500 vs 0.7226). No PathVQA-open conclusion can be drawn from it.
#
# NOTE: --verifier_lora is deliberately NOT passed. diversity_generate_gpu.py scores with vLLM
# LoRARequest, and vLLM 0.9.0.1 silently drops all 192 visual.* LoRA modules. Candidates are scored
# afterwards with src/cascade_methods/open_diverse_score.py under HF transformers.
#
# Resumable: phase_generate skips idx already present in the ckpt, so the existing 178 are not redone.
set -e
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
L7="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"

DIVERSITY_GPU_OK=1 CUDA_VISIBLE_DEVICES=${GPU:-0} python3 src/cascade_methods/diversity_generate_gpu.py \
  --phase generate --model_path "$L7" --tag lingshu7b_div --dataset pathvqa_open \
  --ckpt_dir ckpts/openvqa/diverse --per_combo 1 --cap cap320 --tp 1 --max_model_len 4096 \
  --restrict_dump ckpts/openvqa/diverse/restrict_pathvqa_open_full1500.txt

echo ATTACK4_PHASE1_GEN_DONE
