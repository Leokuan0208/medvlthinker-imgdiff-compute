#!/bin/bash
# ATTACK 2 -- REPLICATION SEED for the round's only positive result.
#
# Arm B0 seed 0 beat always-7B on PATH_VQA_closed by +0.0598 [+0.0470, +0.0723], which FALSIFIES the
# prediction amendment 2 recorded in advance.  This project has retracted a "win" before that turned
# out to be the top of its own 10-seed range, so the win gets a second seed before it is believed.
# Seed 1 is trained with the SAME script, SAME hyperparameters and SAME eval-image ban list, and is
# scored on the two cells that decide the question: PATH_VQA_closed (the win) and VQA_RAD_closed (the
# other 2-option cell, where seed 0 tied).
#
#   setsid nohup bash runners/run_unified_b0_seed1.sh >/dev/null 2>&1 &
#
# Waits for the seed-0 pipeline to release a card; never kills anything.  nohup, never tmux.
set -u
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
cd "$REPO"
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M="$REPO/logs/unified_b0_seed1.log"
mkdir -p "$REPO/logs"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$M"; }
OUT=ckpts/train/lora_verifier_optiononly_s1

say "=== seed-1 replication armed; waiting for the gpu0 branch of the seed-0 pipeline ==="
for i in $(seq 1 720); do
  grep -q "=== gpu0 DONE ===" "$REPO/logs/unified_b0_score_2gpu.log" 2>/dev/null && break
  sleep 30
done
say "gpu0 released (or wait expired); starting seed 1"

if [ ! -s "$REPO/$OUT/adapter_model.safetensors" ]; then
  say ">> TRAIN $OUT (option-only, seed 1)"
  CUDA_VISIBLE_DEVICES=0 timeout -s KILL 25200 python3 \
    src/training_methods/train_unified_verifier.py --seed 1 --out_dir "$OUT" \
    --max_open 0 --max_option 10364 --deadline_s 21600 \
    >> "$REPO/logs/unified_train_lora_verifier_optiononly_s1.log" 2>&1
  say "   TRAIN rc=$?"
fi
[ -s "$REPO/$OUT/adapter_model.safetensors" ] || { say "ABORT: no seed-1 adapter"; exit 1; }
if [ -s "$REPO/$OUT/early_stop.json" ]; then say "WARNING: seed 1 hit its deadline and is EARLY-STOPPED"; fi

say ">> SCORE optiononly_s1 [PATH_VQA_closed,VQA_RAD_closed]"
CUDA_VISIBLE_DEVICES=0 timeout -s KILL 14400 python3 \
  src/cascade_methods/unified_pipeline_score.py --adapter "$OUT" --tag optiononly_s1 \
  --cells "PATH_VQA_closed,VQA_RAD_closed" --deadline_s 10800 --wait_s 2400 --min_free_gib 26 \
  >> "$REPO/logs/unified_score_optiononly_s1.log" 2>&1
say "   SCORE rc=$?"
say "=== SEED 1 DONE ==="
