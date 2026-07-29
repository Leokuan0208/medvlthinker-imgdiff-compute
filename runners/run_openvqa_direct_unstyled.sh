#!/bin/bash
# UNSTYLED DIRECT arm -- the fourth cell of the style x reasoning 2x2, and the one that makes the reasoning
# contrast CLEAN.
# WHY: matching the reasoning arm's prompt TO the direct arm cannot work on this model family -- keeping
# "Do not explain" (or paraphrasing the trigger) suppresses the <think> trace, so a "matched reasoning" arm
# silently stops reasoning (measured trace rates: arm A 0.26-0.69, arm B similar on VQA-RAD). Matching the
# OTHER way has no such failure mode: SYS_DIRECT_UNSTYLED (--direct_unstyled) is SYS_THINK with the reasoning
# instruction removed and nothing else changed, and a direct prompt cannot accidentally start reasoning.
# reason_unmatched (SYS_THINK) minus direct_unstyled therefore isolates REASONING at a fixed output
# convention; direct (SYS) minus direct_unstyled isolates the OUTPUT CONVENTION with reasoning off.
#
# Same evaluated idx sets / chunks / guardrails as the other arms (per-chunk timeout+retry, 64-item
# checkpoint+resume). Direct generation is short (max_tokens 64, ~5 tokens/answer) so chunks finish in
# seconds; the per-chunk cost is dominated by the model load.
cd ~/medvlthinker-imgdiff-compute || exit 1
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
LP="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/snapshots/36b98277cacb60db86f34b75ce0540b1ea35183c/"
CK=ckpts/openvqa/strong_lingshu_direct_unstyled
IDXD=ckpts/openvqa/strong_lingshu_think_matched/idxfiles   # SAME frozen allowlists as every other arm
FAIL=logs/direct_unstyled_failed_chunks.txt
mkdir -p logs "$CK"; : > "$FAIL"

run_ds () {  # $1=dataset $2=idx_file $3=logtag  (direct mode: whole dataset in one invocation)
  local ds="$1" idxf="$2" tag="$3"
  for attempt in 1 2 3; do
    echo "=== $tag attempt $attempt ($(date)) ==="
    timeout -s KILL 3600 env CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_openvqa.py \
      --model_path "$LP" --tag lingshu32b_direct_unstyled --dataset "$ds" --idx_file "$idxf" \
      --n_samples 1 --temp 0 --direct_unstyled --save_raw --ckpt_dir "$CK" --tp 2 \
      --gpu_mem 0.90 --max_model_len 4096 > "logs/${tag}.log" 2>&1
    rc=$?
    if grep -q "^DONE " "logs/${tag}.log"; then echo "$tag OK (rc=$rc)"; return 0; fi
    echo "$tag attempt $attempt did not finish (rc=$rc); retrying"
    sleep 20
  done
  echo "$tag FAILED after 3 attempts"; echo "$ds $idxf $tag" >> "$FAIL"; return 1
}

run_ds vqa_rad_open "$IDXD/vqa_rad_open_evaluated.json" unstyled_vqarad
run_ds slake_open   "$IDXD/slake_open_evaluated.json"   unstyled_slake
run_ds pathvqa_open "$IDXD/pathvqa_open_evaluated.json" unstyled_pathvqa

echo "DIRECT_UNSTYLED_GEN_ALL_DONE ($(date))"
wc -l $CK/*.jsonl 2>/dev/null
echo "failed chunks:"; cat "$FAIL"
