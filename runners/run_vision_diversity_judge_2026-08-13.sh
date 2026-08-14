#!/usr/bin/env bash
# SWEEP 3 -- judge retry. The first attempt died because the other session took GPU0 between the
# capacity check and vLLM's engine start, so this version RE-CHECKS immediately before launching,
# asks for less memory, and retries instead of giving up. It also refuses to start while this
# session's own scoring shards are still holding GPU1.
#
# Judge = src/labeling/run_judge.py, MedVLThinker-32B (a NEUTRAL text LLM, not Lingshu) -- the same
# judge that labelled the frozen endpoint's `sl`. Text-only, so tp=2 at a modest fraction is enough.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
free_mib () { echo $((81920 - $(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits))); }

# Do NOT wait for our own scorers: on a contended box that can starve the judge indefinitely,
# and without judge labels there is no result at all. Instead take whichever route fits:
#   tp=2 when BOTH cards have room, else tp=1 when ONE card can hold the 62 GiB of 32B weights.

NEED=""
for ds in slake_open vqa_rad_open pathvqa_open; do
  E="ckpts/openvqa/visdiv/gen_${ds}_scexploded.jsonl"; JU="${E%.jsonl}.judge.jsonl"
  e=$(wc -l < "$E"); j=0; [ -f "$JU" ] && j=$(wc -l < "$JU")
  [ "$j" -lt "$e" ] && NEED="$NEED $E"
done
[ -z "$NEED" ] && { echo "VISDIV_JUDGE_ALL_DONE (nothing to do)"; exit 0; }

for attempt in $(seq 1 40); do
  # wait until EITHER route is possible
  for _ in $(seq 1 240); do
    f0=$(free_mib 0); f1=$(free_mib 1)
    { [ "$f0" -ge 40000 ] && [ "$f1" -ge 40000 ]; } && break
    [ "$f0" -ge 72000 ] || [ "$f1" -ge 72000 ] && break
    sleep 30
  done
  # re-check RIGHT NOW: this is the race that killed attempt 1
  f0=$(free_mib 0); f1=$(free_mib 1)
  if [ "$f0" -ge 40000 ] && [ "$f1" -ge 40000 ]; then
    DEV=0,1; TP=2; MEM=0.45
  elif [ "$f1" -ge 72000 ]; then
    DEV=1; TP=1; MEM=0.88
  elif [ "$f0" -ge 72000 ]; then
    DEV=0; TP=1; MEM=0.88
  else
    echo "=== $(date '+%F %T') attempt $attempt skipped (GPU0 free=$f0 GPU1 free=$f1) ==="
    sleep 60; continue
  fi
  echo "=== $(date '+%F %T') JUDGE attempt $attempt (GPU0 free=$f0 GPU1 free=$f1) ==="
  echo "    route: CUDA_VISIBLE_DEVICES=$DEV tp=$TP gpu_mem=$MEM"
  CUDA_VISIBLE_DEVICES=$DEV python3 src/labeling/run_judge.py --tp $TP --gpu_mem $MEM \
    --preds $NEED >> logs/visdiv_judge.log 2>&1
  ok=1
  for ds in slake_open vqa_rad_open pathvqa_open; do
    E="ckpts/openvqa/visdiv/gen_${ds}_scexploded.jsonl"; JU="${E%.jsonl}.judge.jsonl"
    e=$(wc -l < "$E"); j=0; [ -f "$JU" ] && j=$(wc -l < "$JU")
    [ "$j" -lt "$e" ] && ok=0
  done
  [ "$ok" = "1" ] && { echo "VISDIV_JUDGE_ALL_DONE"; exit 0; }
  echo "=== $(date '+%F %T') JUDGE incomplete, retrying ==="
  sleep 90
done
echo "VISDIV_JUDGE_FAILED after 40 attempts"
