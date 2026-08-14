#!/bin/bash
# SWEEP 2 -- open-text GENERATOR resolution sweep: generation stage.
# 6 caps x (3 sampling seeds @ T=0.7 n=8  +  1 greedy arm @ T=0 n=1), all 3 open cells.
# The cap320 arm is the DEPLOYED control and is generated HERE, in this session, so no cap-vs-cap
# delta is ever taken against a stored number from another serving config (the +-0.008 caveat).
# Waits for a GPU with enough free VRAM; never kills another process. Resumable per item.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 TORCHDYNAMO_DISABLE=1
L=logs/resolution_open_gen_2026-08-13.log
NEED_MIB=${NEED_MIB:-42000}

# PIN_GPU pins this job to one card so it cannot collide with the MCQ ladder running in the same
# session (they did collide once: an engine core was OOM-killed, logs/resolution_open_gen_*.log
# 12:32:27). It still WAITS for that card rather than oversubscribing, and never kills a tenant.
PIN_GPU=${PIN_GPU:-any}
pick_gpu(){   # echoes a card with >= NEED_MIB free (PIN_GPU=any) or the pinned card, else nothing
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
   | awk -v n="$NEED_MIB" -v g="$PIN_GPU" -F', *' \
     '{ if ((g=="any" || $1==g) && $2-$3 >= n) { print $1; exit } }'
}

say(){ echo "[$(date +%F\ %T)] $*" >> "$L"; }
say "RESOLUTION OPEN GEN START"

# cap_name:max_pixels  -- cap320 (250880) is the deployed generator resolution;
# 12845056 is the qwen_vl_utils default that the MedEvalKit MCQ arms run at.
# ORDER MATTERS: the decisive pair is the deployed control (cap320) against the top of the ladder
# (native = the qwen_vl_utils default the MCQ arms already run at), so those two are generated
# FIRST and the intermediate rungs fill in afterwards. A truncated run then still answers the
# question instead of leaving a ladder with no control.
CAPS=${CAPS:-"native:12845056 cap80:62720 cap320:250880 fullres:1003520 cap640:501760 cap160:125440"}

# RETRY per cap. Two distinct transient failures were observed on this shared box:
#   * the engine core OOM-killed when a co-tenant grew during load (12:32:27), and
#   * vLLM's own profiling assertion "Initial free memory 34.07 GiB, current free memory 36.65 GiB
#     ... other processes sharing the same container release GPU memory while vLLM is profiling"
#     (13:29:04) -- a co-tenant SHRINKING mid-load, which is not an OOM at all.
# Generation is per-item resumable, so a retry costs only the model load.
for spec in $CAPS; do
  NAME=${spec%%:*}; PX=${spec##*:}
  for ATTEMPT in 1 2 3 4 5 6 7 8; do
    G=""
    while [ -z "$G" ]; do G=$(pick_gpu); [ -z "$G" ] && { say "waiting for $NEED_MIB MiB free..."; sleep 120; }; done
    say ">> cap=$NAME max_pixels=$PX on GPU $G (attempt $ATTEMPT)"
    CUDA_VISIBLE_DEVICES=$G python3 src/cascade_methods/resolution_open_generate.py \
        --cap_name "$NAME" --max_pixels "$PX" --seeds 0 1 2 --greedy \
        --n_samples 8 --temp 0.7 --tp 1 --gpu_mem ${GPU_MEM:-0.30} >> "$L" 2>&1
    RC=$?
    say "<< cap=$NAME attempt=$ATTEMPT rc=$RC"
    [ "$RC" -eq 0 ] && break
    sleep 60
  done
done
say "RESOLUTION OPEN GEN DONE"
