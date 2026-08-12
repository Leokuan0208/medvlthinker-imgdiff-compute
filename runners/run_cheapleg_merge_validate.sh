#!/bin/bash
# Attack B -- RE-RUN of the merge/serving validation that failed on 2026-08-12 01:09.
#
# WHY IT FAILED, and why the fix is not a change to the measurement.  vLLM 0.9.0.1's V1 engine
# starts its EngineCore in a SPAWNED subprocess, which re-imports the parent module; because
# cheapleg_merge_validate.py runs at module level (no `if __name__ == "__main__":` guard) Python's
# spawn guard raised "An attempt has been made to start a new process before the current process has
# finished its bootstrapping phase" and arm C never loaded.  VLLM_ENABLE_V1_MULTIPROCESSING=0 runs
# the same engine IN-PROCESS instead of spawning it.  Nothing about the model, the checkpoint, the
# prompts, the sampling params or the comparison changes -- only where the engine's event loop lives.
#
#   $1 = gpu id (default 1)
set -u
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
GPU="${1:-1}"
L=logs/cheapleg_merge_validate_rerun.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

# arm A and B are HF (~17 GB each, sequential); arm C is vLLM at gpu_mem 0.30 (~25 GB).
say "waiting for >=32000 MB free on gpu $GPU ..."
t0=$SECONDS
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  [ $((81920-used)) -ge 32000 ] && break
  [ $((SECONDS-t0)) -gt 43200 ] && { say "WAIT_MEM_TIMEOUT gpu$GPU"; exit 1; }
  sleep 60
done
say "gpu $GPU free enough; running A/B/C validation"

CUDA_VISIBLE_DEVICES=$GPU python3 src/training_methods/cheapleg_merge_validate.py \
  --adapter ckpts/train/lora_cheapleg_s0 --merged ckpts/train/merged_cheapleg_s0 \
  --n 60 --gpu_mem 0.30 \
  --out results/cascade_methods/artifacts/_cheapleg_merge_validate_s0.json >> "$L" 2>&1
say "merge validation rc=$?"
