#!/bin/bash
# ATTACK 2 arm B -- train ONE verifier on BOTH branches' candidate sets, then score with it.
#
# GPU ETIQUETTE: three other rounds share these two A100s.  Every stage waits for a SUSTAINED free
# block on its target GPU (N consecutive readings, not one lucky sample) and never kills anything.
# nohup/setsid, never tmux.  Launch from the repo root:
#   setsid nohup bash runners/run_unified_pipeline_armB.sh 0 >/dev/null 2>&1 &
# ($1 = seed, default 0)
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8 PYTHONHASHSEED=0
SEED="${1:-0}"
OUTD="ckpts/train/lora_verifier_unified_s${SEED}"
M="$REPO/logs/unified_armB_s${SEED}_master.log"
mkdir -p "$REPO/logs"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$M"; }
say "=== ARM B seed ${SEED} START ==="

# wait_gpu <gpu_index> <need_gib> <max_wait_s>  -> echoes 0/1 ok
wait_gpu(){
  local G="$1" NEED="$2" MAXW="$3" waited=0 streak=0
  while [ "$waited" -lt "$MAXW" ]; do
    free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$G" \
           | awk -F', ' '{printf "%d", ($1-$2)/1024}')
    if [ "${free:-0}" -ge "$NEED" ]; then
      streak=$((streak+1)); [ "$streak" -ge 3 ] && return 0
    else
      streak=0
    fi
    sleep 60; waited=$((waited+60))
  done
  return 1
}

# ---- 1. train -----------------------------------------------------------------------------------
if [ -s "$REPO/$OUTD/adapter_model.safetensors" ]; then
  say "SKIP(train done) $OUTD"
else
  # three other rounds share these cards; take whichever frees a sustained 34 GiB block first
  GPU=""
  waited=0
  while [ "$waited" -lt 21600 ]; do
    for g in 1 0; do
      if wait_gpu "$g" 34 180; then GPU="$g"; break; fi
    done
    [ -n "$GPU" ] && break
    waited=$((waited+360))
  done
  [ -n "$GPU" ] || { say "ABORT: no GPU with a sustained 34 GiB block in 6h"; exit 1; }
  say ">> TRAIN seed $SEED on GPU $GPU"
  timeout -s KILL 28800 env CUDA_VISIBLE_DEVICES="$GPU" python3 \
    src/training_methods/train_unified_verifier.py --seed "$SEED" --out_dir "$OUTD" \
    --deadline_s 25200 >> "$REPO/logs/unified_armB_train_s${SEED}.log" 2>&1
  say "TRAIN rc=$?"
  [ -s "$REPO/$OUTD/adapter_model.safetensors" ] || { say "ABORT: no adapter"; exit 1; }
fi

# ---- 2. score the four option cells (cheap cells first, so a deadline still buys the decisive ones)
score_cells(){
  local CELLS="$1" GPU="$2" DL="$3"
  say ">> SCORE [$CELLS] on GPU $GPU"
  timeout -s KILL $((DL+1800)) env CUDA_VISIBLE_DEVICES="$GPU" python3 \
    src/cascade_methods/unified_pipeline_score.py --adapter "$OUTD" --tag "unified_s${SEED}" \
    --cells "$CELLS" --deadline_s "$DL" --wait_s 7200 --min_free_gib 26 \
    >> "$REPO/logs/unified_armB_score_s${SEED}.log" 2>&1
  say "SCORE [$CELLS] rc=$?"
}
score_cells "VQA_RAD_closed,PATH_VQA_closed,MedXpertQA-MM" 1 14400
score_cells "PMC_VQA" 1 14400

# ---- 3. score the open-text pool with the SAME adapter (does unifying COST anything?) ------------
for DS in slake_open vqa_rad_open pathvqa_open; do
  if [ -s "$REPO/$OUTD/transfer_dump_${DS}_lingshu7b.json" ]; then say "SKIP(open done) $DS"; continue; fi
  say ">> SCORE open $DS"
  timeout -s KILL 10800 env CUDA_VISIBLE_DEVICES=1 python3 \
    src/training_methods/verifier_transfer_eval.py --adapter "$OUTD" --datasets "$DS" \
    >> "$REPO/logs/unified_armB_open_s${SEED}.log" 2>&1
  say "SCORE open $DS rc=$?"
done

say "=== ARM B seed ${SEED} DONE ==="
