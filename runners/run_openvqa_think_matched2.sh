#!/bin/bash
# MATCHED variant 2 -- the arm where reasoning is actually ON.
# WHY THIS EXISTS: SYS_THINK_MATCHED (--think_matched, runners/run_openvqa_think_matched.sh) paraphrases the
# reasoning trigger, and Lingshu-32B then SKIPS the <think> trace ~75% of the time (measured on VQA-RAD:
# trace rate 0.255, mean gen 39.8 tok vs 141.5 for the unmatched prompt) -- i.e. it matches the prompts but
# silently turns reasoning OFF, so it cannot attribute the effect. SYS_THINK_MATCHED2 (--think_matched2) keeps
# SYS_THINK's trigger sentences VERBATIM (the documented requirement for the trace to fire) and replaces only
# its answer-style clause with the direct prompt's persona + "short, specific phrase" + "Do not explain".
# Reasoning stays ON; the only difference from the direct arm is the reason-first instruction.
#
# Same evaluated idx sets, same everything else, same guardrails as the matched-A runner (per-chunk
# `timeout -s KILL 3600` + retries, per-64 checkpoint/resume, failed chunks recorded and skipped).
cd ~/medvlthinker-imgdiff-compute || exit 1
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
LP="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/snapshots/36b98277cacb60db86f34b75ce0540b1ea35183c/"
CK=ckpts/openvqa/strong_lingshu_think_matched2
IDXD=ckpts/openvqa/strong_lingshu_think_matched/idxfiles   # SAME frozen allowlists as matched-A
FAIL=logs/think_matched2_failed_chunks.txt
mkdir -p logs "$CK"; : > "$FAIL"

run_chunk () {  # $1=dataset $2=idx_file $3=logtag
  local ds="$1" idxf="$2" tag="$3"
  for attempt in 1 2 3; do
    echo "=== $tag attempt $attempt ($(date)) ==="
    timeout -s KILL 3600 env CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_openvqa.py \
      --model_path "$LP" --tag lingshu32b_think_matched2 --dataset "$ds" --idx_file "$idxf" \
      --n_samples 1 --temp 0 --think_matched2 --save_raw --ckpt_dir "$CK" --tp 2 \
      --gpu_mem 0.90 --max_model_len 4096 --max_tokens 512 > "logs/${tag}.log" 2>&1
    rc=$?
    if grep -q "^DONE " "logs/${tag}.log"; then echo "$tag OK (rc=$rc)"; return 0; fi
    echo "$tag attempt $attempt did not finish (rc=$rc); retrying"
    sleep 20
  done
  echo "$tag FAILED after 3 attempts"; echo "$ds $idxf $tag" >> "$FAIL"; return 1
}

run_chunk vqa_rad_open "$IDXD/vqa_rad_open_chunk0.json" matched2_vqarad_c0
run_chunk slake_open   "$IDXD/slake_open_chunk0.json"   matched2_slake_c0
run_chunk slake_open   "$IDXD/slake_open_chunk1.json"   matched2_slake_c1
run_chunk pathvqa_open "$IDXD/pathvqa_open_chunk0.json" matched2_pathvqa_c0
run_chunk pathvqa_open "$IDXD/pathvqa_open_chunk1.json" matched2_pathvqa_c1
run_chunk pathvqa_open "$IDXD/pathvqa_open_chunk2.json" matched2_pathvqa_c2

echo "MATCHED2_GEN_ALL_DONE ($(date))"
wc -l $CK/*.jsonl 2>/dev/null
echo "failed chunks:"; cat "$FAIL"
