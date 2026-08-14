#!/bin/bash
# run_coadapt_T04_trainpools.sh -- generate the TRAIN-split candidate pools at T = 0.4.
#
# PRE-REGISTRATION: results/cascade_methods/artifacts/_coadapt_verifier_prereg_2026-08-14.json
#
# WHAT THIS IS.  The incumbent verifier (ckpts/train/lora_verifier_disjoint) was trained on candidate
# pools generated at temperature 0.7 (runners/run_verifier_disjoint_retrain.sh stage 1,
# src/training_methods/build_disjoint_verifier_split.py:110).  The cold ladder crowned T = 0.4 as the
# peak at INFERENCE time.  This produces the T = 0.4 equivalent of those training pools, on the SAME
# train questions, with EVERY other generation parameter held byte-identical to the incumbent's:
#
#     src/labeling/run_openvqa.py --n_samples 8 --cap cap320 --max_model_len 4096 --max_tokens 64
#     (defaults), same idx allowlists from data/disjoint_split/, same prompt (SYS).
#     THE ONLY CHANGE IS  --temp 0.7  ->  --temp 0.4 .
#
# Output goes to a NEW directory so ckpts/openvqa/cheap_lingshu7b -- the artifact of record for every
# published verifier number -- is never written to.
#
# Resumable: run_openvqa.py resumes from the last completed line of each ckpt file.
#   setsid nohup bash runners/run_coadapt_T04_trainpools.sh >/dev/null 2>&1 &
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
L7="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"
OUTD="$REPO/ckpts/openvqa/cheap_lingshu7b_T04"
IDX="$REPO/data/disjoint_split"
M="$REPO/logs/coadapt_T04_master.log"
mkdir -p "$OUTD" "$REPO/logs"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$M"; }
say "=== T04 TRAIN POOL GENERATION START ==="

# ---- wait for a GPU to be sustained-free (never kill a co-tenant; queue instead) ---------------
wait_gpu(){   # $1 = gpu id, $2 = GiB needed
  local g="$1" need="$2" streak=0 waited=0
  while true; do
    local free=$(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g") / 1024 ))
    if [ "$free" -ge "$need" ]; then streak=$((streak+1)); [ "$streak" -ge 3 ] && return 0
    else streak=0; fi
    if [ "$waited" -ge 21600 ]; then say "ABORT: gpu$g never freed ${need}GiB in 6h"; return 1; fi
    sleep 20; waited=$((waited+20))
  done
}

gen(){        # $1 = gpu, $2 = dataset
  local g="$1" ds="$2"
  local out="$OUTD/ckpt_${ds}_lingshu7bT04_sc8.jsonl"
  local want=$(python3 -c "import json;print(len(json.load(open('$IDX/idx_${ds}.json'))))")
  local have=0; [ -f "$out" ] && have=$(wc -l < "$out")
  if [ "$have" -ge "$want" ]; then say "SKIP(gen done $have/$want) $ds"; return 0; fi
  for attempt in 1 2 3 4 5; do
    wait_gpu "$g" 40 || return 1
    say ">> GEN $ds attempt $attempt ($have/$want) on gpu$g"
    timeout -s KILL 7200 env CUDA_VISIBLE_DEVICES="$g" python3 src/labeling/run_openvqa.py \
      --model_path "$L7" --tag lingshu7bT04_sc8 --dataset "$ds" \
      --n_samples 8 --temp 0.4 --cap cap320 --ckpt_dir "$OUTD" --tp 1 --max_model_len 4096 \
      --idx_file "$IDX/idx_${ds}.json" >> "$REPO/logs/coadapt_T04_gen_gpu${g}.log" 2>&1
    have=0; [ -f "$out" ] && have=$(wc -l < "$out")
    say "GEN $ds -> $have/$want rows"
    [ "$have" -ge "$want" ] && return 0
    pkill -9 -f VLLM::EngineCore 2>/dev/null; sleep 20
  done
  say "ABORT: generation incomplete for $ds ($have/$want)"; return 1
}

( gen 0 pathvqa_open_train ) &
P0=$!
( gen 1 slake_open_train && gen 1 vqa_rad_open_train && gen 1 kvasir_open && gen 1 radimagenet_open ) &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
say "generation finished rc0=$R0 rc1=$R1"
say "=== T04 TRAIN POOL GENERATION DONE ==="
echo "COADAPT_T04_GEN_DONE"
