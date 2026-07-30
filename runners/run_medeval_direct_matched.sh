#!/bin/bash
# Matched-prompt DIRECT arm re-run on the reasoning-heavy MedEvalKit benchmarks.
#
# WHY: the existing eval_results_*_reason dumps were produced with a local MedEvalKit prompt
# edit that gave the reasoning arm a real reasoning trigger, but left the direct arm as
# upstream's "Answer with the option's letter ... directly." -- so the two arms differed in
# THREE ways (reasoning trigger, the word "directly", \boxed{} vs bare letter). This re-runs
# ONLY the direct arm under a prompt matched to the reasoning arm (same \boxed{} format, same
# wording, reasoning clause removed), so reasoning-vs-direct becomes a matched experiment.
#
# The prompt comes from src/labeling/medeval_matched_prompt.py, which monkeypatches MedEvalKit
# at import time -- MedEvalKit itself stays byte-identical to upstream on disk.
#
# Every sampling/runtime setting below is copied from runners/run_full_matrix_medeval.sh (the
# runner that produced the *_reason dumps) so the ONLY difference between the arms is the prompt:
#   --seed 42 --tensor_parallel_size 2 --use_vllm True --max_new_tokens 2048
#   --max_image_num 6 --temperature 0 --top_p 0.0001 --repetition_penalty 1 --datasets_path hf
# IV3-38B additionally needs MAX_MODEL_LEN=16384 (65536 does not fit the KV cache).
#
# Robustness (documented intermittent NCCL hang on TP=2): one job per (model,benchmark), each
# under `timeout -s KILL 3600` with ONE retry, small EVAL_BATCH_SIZE, skip-if-done, and
# skip-and-record after a second failure. Never loops forever.
#
# Launch detached:  setsid nohup bash runners/run_medeval_direct_matched.sh >/dev/null 2>&1 &
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export MEDEVAL_MATCHED_PROMPT=1          # <-- the whole point; without it the patch is inert
export EVAL_BATCH_SIZE=250               # keep decoded-image memory bounded (temp=0 => same outputs)

PY=/data/dan/medeval_venv/bin/python
L=$REPO/logs/medeval_direct_matched.log
M=$REPO/logs/medeval_direct_matched_master.log
mkdir -p "$REPO/logs"
echo "=== DIRECT-MATCHED RUN START $(date) ===" >> "$M"

# wait for the GPUs to be free (never loop forever: 6h cap)
waited=0
while true; do
  # pure-shell sum: `bc` is not installed on this VM
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  [ "${used:-99999}" -lt 2000 ] && break
  if [ "$waited" -ge 21600 ]; then echo "ABORT: GPUs still busy after 6h ($used MiB) $(date)" >> "$M"; exit 1; fi
  sleep 60; waited=$((waited+60))
done
echo "GPUs free (${used} MiB) after ${waited}s; starting $(date)" >> "$M"

# runjob: MODEL_NAME MODEL_PATH OUTDIR DATASET [MAX_MODEL_LEN]
# NOTE the "/{}" in --output_path: kept identical to run_full_matrix_medeval.sh so the new dumps
# have the same directory shape as the *_reason dumps they are compared against.
runjob(){
  local MN="$1" MP="$2" OUT="$3" DS="$4" MML="${5:-}"
  local DEST="$REPO/MedEvalKit/$OUT"
  if find "$DEST" -path "*/$DS/*" -name "*.json" 2>/dev/null | grep -q .; then
    echo "SKIP(done) $OUT $DS $(date)" >> "$M"; return 0; fi

  local attempt
  for attempt in 1 2; do
    echo ">> RUN $OUT $DS attempt=$attempt $(date)" >> "$M"
    timeout -s KILL 3600 env CUDA_VISIBLE_DEVICES=0,1 tensor_parallel_size=2 \
      MEDEVAL_PROMPT_LOG="$REPO/logs/medeval_direct_matched_prompts.jsonl" \
      ${MML:+MAX_MODEL_LEN=$MML} \
      "$PY" "$REPO/src/labeling/medeval_matched_prompt.py" -- \
        --eval_datasets "$DS" --datasets_path hf --output_path "$OUT/{}" \
        --model_name "$MN" --model_path "$MP" \
        --seed 42 --cuda_visible_devices "0,1" --tensor_parallel_size 2 --use_vllm True \
        --max_new_tokens 2048 --max_image_num 6 --temperature 0 --top_p 0.0001 \
        --repetition_penalty 1 --reasoning False --use_llm_judge False \
        --judge_model_type openai --judge_model None --api_key None --base_url None \
        --test_times 1 >> "$L" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && find "$DEST" -path "*/$DS/*" -name "*.json" 2>/dev/null | grep -q .; then
      echo "OK $OUT $DS attempt=$attempt $(date)" >> "$M"; return 0
    fi
    echo "FAIL $OUT $DS attempt=$attempt rc=$rc $(date)" >> "$M"
    pkill -9 -f VLLM::EngineCore 2>/dev/null; sleep 45
  done
  echo "SKIPPED_AFTER_2_FAILURES $OUT $DS $(date)" >> "$M"
  return 1
}

# The three families that have *_reason dumps to compare against.
for DS in MMMU-Medical-val MedXpertQA-MM; do
  runjob Qwen2.5-VL lingshu-medical-mllm/Lingshu-32B          eval_results_lingshu32b_direct_matched "$DS"
  runjob Qwen2.5-VL /data/dan/weights/MedVLThinker-32B-RL_m23k eval_results_mvt32b_direct_matched    "$DS"
  runjob InternVL   OpenGVLab/InternVL3-38B                   eval_results_iv3_38b_direct_matched    "$DS" 16384
done

echo "=== DIRECT_MATCHED_ALL_DONE $(date) ===" >> "$M"
