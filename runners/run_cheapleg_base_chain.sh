#!/bin/bash
# Attack B -- everything the BASE (frozen Lingshu-7B) control arm needs, in order, on one GPU.
# Waits for the open-text generation already in flight, then: verifier-scoring null test ->
# explode/judge/score -> MedEvalKit MCQ suite.  Every stage is resumable and skipped when done.
set -u
cd ~/medvlthinker-imgdiff-compute
GPU="${1:-0}"
L=logs/cheapleg_base_chain.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

say "waiting for base open-text generation to finish ..."
until grep -q "CHEAPLEG_OPEN_GEN_DONE" logs/cheapleg_open_base.log 2>/dev/null; do sleep 60; done
say "open gen done"

# --- NULL TEST of the verifier-scoring harness against the stored incumbent dump ----------------
if [ ! -s ckpts/cheapleg/_nulltest/nulltest.json ]; then
  say ">> verifier scoring NULL TEST"
  HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    python3 src/training_methods/cheapleg_score_open.py --gen_dir ckpts/openvqa/cheap_lingshu7b \
      --tag lingshu7b --out_dir ckpts/cheapleg/_nulltest --nulltest 40 >> "$L" 2>&1
  say "null test rc=$?"
fi

bash runners/run_cheapleg_arm_finish.sh base7b "$GPU"
bash runners/run_cheapleg_mcq.sh \
  /data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/ \
  cheapleg_base7b "$GPU"
say "CHEAPLEG_BASE_CHAIN_DONE"
