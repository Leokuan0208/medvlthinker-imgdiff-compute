#!/bin/bash
# Extend Lingshu-32B-THINK open-text dumps to the FULL evaluated idx sets (SLAKE 645, PathVQA 1500;
# VQA-RAD already complete at 200). Per-chunk timeout -s KILL 3600 + one retry; run_openvqa.py
# checkpoints every 64 items and resumes, so a killed chunk restarts where it left off (tp=2 hang-safe).
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
LP="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/snapshots/36b98277cacb60db86f34b75ce0540b1ea35183c/"
IDXD=ckpts/openvqa/strong_lingshu_think/idxfiles
mkdir -p logs

run_chunk () {  # $1=dataset $2=idx_file $3=logtag
  local ds="$1" idxf="$2" tag="$3"
  for attempt in 1 2; do
    echo "=== $tag attempt $attempt ($(date)) ==="
    timeout -s KILL 3600 env CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_openvqa.py \
      --model_path "$LP" --tag lingshu32b_think --dataset "$ds" --idx_file "$idxf" \
      --n_samples 1 --temp 0 --think --ckpt_dir ckpts/openvqa/strong_lingshu_think --tp 2 \
      --gpu_mem 0.90 --max_model_len 4096 --max_tokens 512 > "logs/${tag}.log" 2>&1
    rc=$?
    if grep -q "^DONE " "logs/${tag}.log"; then echo "$tag OK (rc=$rc)"; return 0; fi
    echo "$tag attempt $attempt did not finish (rc=$rc); retrying"
  done
  echo "$tag FAILED after 2 attempts"; return 1
}

# SLAKE: 445 remaining (1 chunk)
run_chunk slake_open   "$IDXD/slake_open_need_chunk0.json"   think_ext_slake_c0
# PathVQA: 1300 remaining (3 chunks)
run_chunk pathvqa_open "$IDXD/pathvqa_open_need_chunk0.json" think_ext_pathvqa_c0
run_chunk pathvqa_open "$IDXD/pathvqa_open_need_chunk1.json" think_ext_pathvqa_c1
run_chunk pathvqa_open "$IDXD/pathvqa_open_need_chunk2.json" think_ext_pathvqa_c2

echo "THINK_EXTEND_ALL_DONE"
