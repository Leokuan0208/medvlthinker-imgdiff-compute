#!/bin/bash
# run_coadapt_T04_judge.sh -- explode + preload + judge the T=0.4 TRAIN pools.
#
# PRE-REGISTRATION: results/cascade_methods/artifacts/_coadapt_verifier_prereg_2026-08-14.json
#
# The judge is the project's own (src/labeling/run_judge.py, MedVLThinker-32B, tp=2, temperature 0) --
# the SAME judge that labelled the incumbent's T=0.7 training pools, so the co-adapted verifier's
# labels come from an identical grader.
#
# Candidate strings the judge has already labelled for the SAME train question at T=0.7 are preloaded
# rather than re-judged (see src/training_methods/coadapt_T04_build_pools.py). A seeded holdout is left
# unpreloaded on purpose so the preload is null-tested against fresh labels in THIS round.
#
#   setsid nohup bash runners/run_coadapt_T04_judge.sh >/dev/null 2>&1 &
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
OUTD="$REPO/ckpts/openvqa/cheap_lingshu7b_T04"
IDX="$REPO/data/disjoint_split"
M="$REPO/logs/coadapt_T04_master.log"
say(){ echo "$(date -u +%H:%M:%S) $*" >> "$M"; }
say "=== T04 JUDGE STAGE START ==="

# ---- 0. refuse to proceed on an incomplete pool ------------------------------------------------
for DS in slake_open_train vqa_rad_open_train pathvqa_open_train kvasir_open radimagenet_open; do
  W=$(python3 -c "import json;print(len(json.load(open('$IDX/idx_${DS}.json'))))")
  H=0; [ -f "$OUTD/ckpt_${DS}_lingshu7bT04_sc8.jsonl" ] && H=$(wc -l < "$OUTD/ckpt_${DS}_lingshu7bT04_sc8.jsonl")
  [ "$H" -ge "$W" ] || { say "ABORT(judge): $DS pool is $H/$W"; exit 1; }
done
say "all five T04 train pools complete"

# ---- 1. explode + preload ----------------------------------------------------------------------
python3 src/training_methods/coadapt_T04_build_pools.py --stage explode  >> "$M" 2>&1 || { say "explode FAILED"; exit 1; }
python3 src/training_methods/coadapt_T04_build_pools.py --stage preload  >> "$M" 2>&1 || { say "preload FAILED"; exit 1; }

# ---- 2. judge whatever the preload could not cover ---------------------------------------------
NEED=""
for DS in slake_open_train vqa_rad_open_train pathvqa_open_train kvasir_open radimagenet_open; do
  EXP="$OUTD/ckpt_${DS}_lingshu7bT04_sc8_scexploded.jsonl"; JUD="${EXP%.jsonl}.judge.jsonl"
  E=$(wc -l < "$EXP"); Jn=0; [ -f "$JUD" ] && Jn=$(wc -l < "$JUD")
  if [ "$Jn" -lt "$E" ]; then NEED="$NEED $EXP"; else say "SKIP(judge done $Jn/$E) $DS"; fi
done
if [ -n "$NEED" ]; then
  for attempt in 1 2 3; do
    # BOTH GPUs must be free: the judge is a 32B at tp=2. Wait, never kill a co-tenant.
    streak=0; waited=0
    while true; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
      if [ "${used:-99999}" -lt 2000 ]; then streak=$((streak+1)); [ "$streak" -ge 3 ] && break
      else streak=0; fi
      if [ "$waited" -ge 21600 ]; then say "ABORT(judge): GPUs busy 6h (${used} MiB)"; exit 1; fi
      sleep 20; waited=$((waited+20))
    done
    say ">> JUDGE attempt $attempt:$NEED"
    timeout -s KILL 21600 env CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py --tp 2 \
      --preds $NEED >> "$REPO/logs/coadapt_T04_judge.log" 2>&1
    say "JUDGE rc=$?"
    pkill -9 -f VLLM::EngineCore 2>/dev/null; sleep 30
    LEFT=""
    for EXP in $NEED; do
      JUD="${EXP%.jsonl}.judge.jsonl"; E=$(wc -l < "$EXP"); Jn=0; [ -f "$JUD" ] && Jn=$(wc -l < "$JUD")
      [ "$Jn" -lt "$E" ] && LEFT="$LEFT $EXP"
    done
    NEED="$LEFT"
    [ -z "$NEED" ] && break
  done
  [ -z "$NEED" ] || { say "ABORT(judge): still incomplete:$NEED"; exit 1; }
fi

# ---- 3. the preload null test ------------------------------------------------------------------
python3 src/training_methods/coadapt_T04_build_pools.py --stage validate >> "$M" 2>&1 || { say "validate FAILED"; exit 1; }
say "=== T04 JUDGE STAGE DONE ==="
echo "COADAPT_T04_JUDGE_DONE"
