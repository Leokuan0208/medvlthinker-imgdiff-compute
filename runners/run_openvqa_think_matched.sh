#!/bin/bash
# THE DECISIVE EXPERIMENT -- matched-prompt reasoning arm.
# Re-run Lingshu-32B in REASONING mode with the MATCHED system prompt (run_openvqa.py
# SYS_THINK_MATCHED, --think_matched): identical to the direct prompt in persona ("You are an expert
# medical image analyst") and answer-style constraints ("short, specific phrase", "Do not explain"),
# differing ONLY by the reason-first <think></think> instruction. Everything else (model, cap320,
# greedy n=1, max_tokens 512, tp=2, extraction after the last </think>) is IDENTICAL to the UNMATCHED
# think run in ckpts/openvqa/strong_lingshu_think, so the only moving part is the prompt.
#
# Evaluated idx sets = exactly the ones the headline uses (integrated_pandora.load_open_rows):
#   SLAKE-open 645, VQA-RAD-open 200, PathVQA-open 1500  -> ckpts/.../idxfiles/<ds>_chunk*.json
#
# GPU guardrails (tp=2 has a documented intermittent NCCL hang): PER-DATASET AND CHUNKED, per-chunk
# `timeout -s KILL 3600` + one retry; run_openvqa.py checkpoints every 64 items and resumes from the
# done-set, so a killed chunk restarts where it left off. If a chunk fails all attempts it is RECORDED
# in logs/think_matched_failed_chunks.txt and the runner CONTINUES. Never loops forever.
cd ~/medvlthinker-imgdiff-compute || exit 1
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
LP="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/snapshots/36b98277cacb60db86f34b75ce0540b1ea35183c/"
CK=ckpts/openvqa/strong_lingshu_think_matched
IDXD=$CK/idxfiles
FAIL=logs/think_matched_failed_chunks.txt
mkdir -p logs "$CK"; : > "$FAIL"

run_chunk () {  # $1=dataset $2=idx_file $3=logtag
  local ds="$1" idxf="$2" tag="$3"
  for attempt in 1 2 3; do
    echo "=== $tag attempt $attempt ($(date)) ==="
    timeout -s KILL 3600 env CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_openvqa.py \
      --model_path "$LP" --tag lingshu32b_think_matched --dataset "$ds" --idx_file "$idxf" \
      --n_samples 1 --temp 0 --think_matched --save_raw --ckpt_dir "$CK" --tp 2 \
      --gpu_mem 0.90 --max_model_len 4096 --max_tokens 512 > "logs/${tag}.log" 2>&1
    rc=$?
    if grep -q "^DONE " "logs/${tag}.log"; then echo "$tag OK (rc=$rc)"; return 0; fi
    echo "$tag attempt $attempt did not finish (rc=$rc); retrying"
    sleep 20
  done
  echo "$tag FAILED after 3 attempts"; echo "$ds $idxf $tag" >> "$FAIL"; return 1
}

# VQA-RAD first (smallest -> earliest sanity signal), then SLAKE, then PathVQA.
run_chunk vqa_rad_open "$IDXD/vqa_rad_open_chunk0.json" matched_vqarad_c0
run_chunk slake_open   "$IDXD/slake_open_chunk0.json"   matched_slake_c0
run_chunk slake_open   "$IDXD/slake_open_chunk1.json"   matched_slake_c1
run_chunk pathvqa_open "$IDXD/pathvqa_open_chunk0.json" matched_pathvqa_c0
run_chunk pathvqa_open "$IDXD/pathvqa_open_chunk1.json" matched_pathvqa_c1
run_chunk pathvqa_open "$IDXD/pathvqa_open_chunk2.json" matched_pathvqa_c2

echo "MATCHED_GEN_ALL_DONE ($(date))"
wc -l $CK/*.jsonl 2>/dev/null
echo "failed chunks:"; cat "$FAIL"
