#!/bin/bash
# ATTACK 2 arm B0 -- the OPTION-ONLY verifier (amendment 2).  Upper bound on what the unified scorer
# can do on the option branch, and the cheap decisive probe.  Then arm B (the unified verifier).
#
# GPU ETIQUETTE: three other rounds share these two A100s.  Every stage waits for a SUSTAINED free
# block (3 consecutive readings) and never kills anything.  nohup/setsid, never tmux.
#   setsid nohup bash runners/run_unified_pipeline_armB0.sh >/dev/null 2>&1 &
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M="$REPO/logs/unified_armB0_master.log"
mkdir -p "$REPO/logs"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$M"; }
say "=== ARM B0 (option-only) + ARM B (unified) START ==="

wait_gpu(){   # <gpu> <need_gib> <max_wait_s>
  local G="$1" NEED="$2" MAXW="$3" waited=0 streak=0 free=0
  while [ "$waited" -lt "$MAXW" ]; do
    free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$G" \
           | awk -F', ' '{printf "%d", ($1-$2)/1024}')
    if [ "${free:-0}" -ge "$NEED" ]; then
      streak=$((streak+1)); [ "$streak" -ge 3 ] && return 0
    else streak=0; fi
    sleep 45; waited=$((waited+45))
  done
  return 1
}
pick_gpu(){   # <need_gib> <max_wait_s>  -> echoes the gpu index, or nothing
  local NEED="$1" MAXW="$2" waited=0
  while [ "$waited" -lt "$MAXW" ]; do
    for g in 0 1; do
      if wait_gpu "$g" "$NEED" 135; then echo "$g"; return 0; fi
    done
    waited=$((waited+270))
  done
  return 1
}

train_one(){  # <out_dir> <max_open> <max_option> <seed>
  local OUTD="$1" MO="$2" MOPT="$3" SEED="$4"
  if [ -s "$REPO/$OUTD/adapter_model.safetensors" ]; then say "SKIP(train done) $OUTD"; return 0; fi
  # the CPU data build takes ~10 min, so the binding wait is INSIDE the python (right before the
  # weights load) and it picks whichever card has room THEN.  No CUDA_VISIBLE_DEVICES pin here:
  # pinning at launch time picked a card that was full 10 minutes later, twice.
  say ">> TRAIN $OUTD (open=$MO option=$MOPT seed=$SEED), device chosen at load time"
  timeout -s KILL 25200 env python3 \
    src/training_methods/train_unified_verifier.py --seed "$SEED" --out_dir "$OUTD" \
    --max_open "$MO" --max_option "$MOPT" --deadline_s 21600 \
    >> "$REPO/logs/unified_train_$(basename "$OUTD").log" 2>&1
  say "TRAIN $OUTD rc=$?"
  [ -s "$REPO/$OUTD/adapter_model.safetensors" ] || { say "ABORT: no adapter for $OUTD"; return 1; }
}

score_opts(){ # <out_dir> <tag> <cells> <deadline_s>
  local OUTD="$1" TAG="$2" CELLS="$3" DL="$4"
  local G; G=$(pick_gpu 27 14400) || { say "ABORT: no GPU for scoring $TAG [$CELLS]"; return 1; }
  say ">> SCORE $TAG [$CELLS] on GPU $G"
  timeout -s KILL $((DL+1800)) env CUDA_VISIBLE_DEVICES="$G" python3 \
    src/cascade_methods/unified_pipeline_score.py --adapter "$OUTD" --tag "$TAG" \
    --cells "$CELLS" --deadline_s "$DL" \
    >> "$REPO/logs/unified_score_${TAG}.log" 2>&1
  say "SCORE $TAG [$CELLS] rc=$?"
}

# ================= arm B0: option-only verifier =================================================
train_one ckpts/train/lora_verifier_optiononly_s0 0 10364 0 || exit 1
score_opts ckpts/train/lora_verifier_optiononly_s0 optiononly_s0 "PATH_VQA_closed,VQA_RAD_closed" 7200
score_opts ckpts/train/lora_verifier_optiononly_s0 optiononly_s0 "MedXpertQA-MM" 7200
score_opts ckpts/train/lora_verifier_optiononly_s0 optiononly_s0 "PMC_VQA" 14400

# ================= arm B: the UNIFIED verifier ==================================================
train_one ckpts/train/lora_verifier_unified_s0 10364 10364 0 || exit 1
score_opts ckpts/train/lora_verifier_unified_s0 unified_s0 "PATH_VQA_closed,VQA_RAD_closed" 7200
score_opts ckpts/train/lora_verifier_unified_s0 unified_s0 "MedXpertQA-MM" 7200
score_opts ckpts/train/lora_verifier_unified_s0 unified_s0 "PMC_VQA" 14400

# the open-text half of arm B: does unifying COST anything on the format the verifier was built for?
for DS in slake_open vqa_rad_open pathvqa_open; do
  D="ckpts/train/lora_verifier_unified_s0"
  [ -s "$REPO/$D/transfer_dump_${DS}_lingshu7b.json" ] && { say "SKIP(open done) $DS"; continue; }
  G=$(pick_gpu 27 14400) || { say "ABORT: no GPU for open $DS"; break; }
  say ">> SCORE open $DS on GPU $G"
  timeout -s KILL 10800 env CUDA_VISIBLE_DEVICES="$G" python3 \
    src/training_methods/verifier_transfer_eval.py --adapter "$D" --datasets "$DS" \
    >> "$REPO/logs/unified_armB_open.log" 2>&1
  say "SCORE open $DS rc=$?"
done

say "=== ARM B0 + ARM B DONE ==="
