#!/usr/bin/env bash
# run_verifier_32b.sh -- GUARDED, BOUNDED tp=2 32B-verifier experiment.
#
# Tests whether a stronger (zero-shot) Lingshu-32B verifier converts oracle->selection better
# than the trained Lingshu-7B pointwise verifier (the selectability-wall attack).
#
# GUARDRAILS (do NOT repeat the multi-day OmniMed NCCL saga):
#   * ~200 questions/dataset x 3 datasets, deduped to distinct answers.
#   * timeout -s KILL 1800 (30 min) per dataset attempt, AT MOST 3 attempts (1 + 2 retries).
#   * If a dataset fails all 3 attempts -> mark tp=2-BLOCKED, skip it.
#   * If the FIRST dataset fails all attempts -> tp=2 won't run at all -> STOP, report what we have.
#   * Kill orphan vLLM workers + wait for GPU memory to drain between attempts.
# Self-cd's to repo root. Launch:  bash runners/run_verifier_32b.sh
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
LOG=logs/gpu_experiments.log
export HF_HOME=/data/dan/hf_cache
export VERIFIER_GPU_OK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
TS(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(TS)] [verifier32b] $*" | tee -a "$LOG"; }

rows_of(){ [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

kill_orphans(){
  pkill -9 -f verifier_32b_gpu 2>/dev/null
  pkill -9 -f 'EngineCore|VLLM::|multiprocessing.spawn|multiprocessing.resource_tracker' 2>/dev/null
  # wait up to 60s for GPU memory to drain below 3 GiB on both GPUs
  for _ in $(seq 1 12); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
    [ -z "$m" ] && break
    [ "$m" -lt 3000 ] && break
    sleep 5
  done
}

DATASETS=(vqa_rad_open slake_open pmc_content)
TARGET=200
M32=lingshu-medical-mllm/Lingshu-32B
M7=lingshu-medical-mllm/Lingshu-7B
declare -a OK32=() BLOCKED=()

say "=== START (32B tp=2 verifier, ${#DATASETS[@]} datasets x ~${TARGET} q) ==="
first=1
for ds in "${DATASETS[@]}"; do
  ck="ckpts/openvqa/verifier32b/ckpt_${ds}_lingshu32b.jsonl"
  done_flag=0
  for attempt in 1 2 3; do
    have=$(rows_of "$ck")
    if [ "$have" -ge "$TARGET" ]; then done_flag=1; say "$ds: already complete ($have rows)"; break; fi
    say "$ds: 32B attempt $attempt/3 (have $have/$TARGET) ..."
    CUDA_VISIBLE_DEVICES=0,1 timeout -s KILL 1800 \
      python3 src/cascade_methods/verifier_32b_gpu.py --dataset "$ds" \
        --model_path "$M32" --tag lingshu32b --tp 2 --n "$TARGET" >>"$LOG" 2>&1
    rc=$?
    have=$(rows_of "$ck")
    say "$ds: attempt $attempt rc=$rc rows=$have/$TARGET"
    if [ "$have" -ge "$TARGET" ]; then done_flag=1; break; fi
    say "$ds: attempt $attempt incomplete -> killing orphans + draining GPU"
    kill_orphans
  done
  if [ "$done_flag" -eq 1 ]; then
    OK32+=("$ds"); say "$ds: 32B DONE"
  else
    BLOCKED+=("$ds"); say "$ds: 32B tp=2-BLOCKED (3 attempts failed)"
    if [ "$first" -eq 1 ]; then
      say "FIRST dataset failed all attempts -> tp=2 will not run. STOPPING."
      say "RESULT: 32B tp=2-BLOCKED on all attempted. OK=[] BLOCKED=[${BLOCKED[*]}]"
      exit 2
    fi
  fi
  first=0
  kill_orphans
done

say "32B pass complete. OK=[${OK32[*]:-}] BLOCKED=[${BLOCKED[*]:-}]"

# ---- cheap + SAFE 7B zero-shot control (tp=1), only where 32B succeeded (same questions) ----
for ds in "${OK32[@]:-}"; do
  [ -z "$ds" ] && continue
  ck="ckpts/openvqa/verifier32b/ckpt_${ds}_lingshu7b_zs.jsonl"
  have=$(rows_of "$ck")
  if [ "$have" -ge "$TARGET" ]; then say "$ds: 7B-zs already complete"; continue; fi
  say "$ds: 7B zero-shot control (tp=1) ..."
  CUDA_VISIBLE_DEVICES=0 timeout -s KILL 1200 \
    python3 src/cascade_methods/verifier_32b_gpu.py --dataset "$ds" \
      --model_path "$M7" --tag lingshu7b_zs --tp 1 --n "$TARGET" >>"$LOG" 2>&1
  say "$ds: 7B-zs rc=$? rows=$(rows_of "$ck")/$TARGET"
  kill_orphans
done

say "Running CPU measure ..."
python3 src/cascade_methods/verifier_32b_measure.py >>"$LOG" 2>&1
say "=== COMPLETE. OK=[${OK32[*]:-}] BLOCKED=[${BLOCKED[*]:-}] ==="
