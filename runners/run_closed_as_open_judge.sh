#!/usr/bin/env bash
# run_closed_as_open_judge.sh -- BUILD 3: label every distinct (item, answer) with the project's
# 32B judge (MedVLThinker-32B, src/labeling/run_judge.py, unchanged).
#
# tp=2 needs BOTH cards, so this WAITS until each card is genuinely free enough rather than
# oversubscribing another user's job.  Never kills anything.
#
#   nohup bash runners/run_closed_as_open_judge.sh > logs/closed_as_open_judge_2026-08-16.log 2>&1 &
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 TORCHDYNAMO_DISABLE=1

NEED=58000        # MiB free PER CARD: 32 GB weights/card + activations + KV at util 0.80
UTIL=0.80
MAXWAIT=${MAXWAIT:-14400}

free_mib () { nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits; }

t=0
while :; do
  f0=$(free_mib 0); f1=$(free_mib 1)
  if [ "$f0" -ge "$NEED" ] && [ "$f1" -ge "$NEED" ]; then
    echo "both cards free enough (gpu0=${f0} gpu1=${f1} MiB) $(date -Is)"; break
  fi
  if [ "$t" -ge "$MAXWAIT" ]; then
    echo "!! gave up waiting for a free GPU pair after ${MAXWAIT}s (gpu0=${f0} gpu1=${f1})"; exit 9
  fi
  echo "waiting for a free GPU pair: gpu0=${f0} gpu1=${f1} MiB (need ${NEED} each) $(date -Is)"
  sleep 60; t=$((t+60))
done

PREDS=$(ls ckpts/closed_as_open/judge_*.jsonl 2>/dev/null | grep -v '\.judge\.jsonl$')
if [ -z "$PREDS" ]; then echo "!! no judge worklists -- run closed_as_open_explode.py first"; exit 1; fi
echo "judging: $PREDS"
wc -l $PREDS

for attempt in 1 2 3; do
  python3 src/labeling/run_judge.py --preds $PREDS --tp 2 --gpu_mem $UTIL --max_model_len 2048
  rc=$?
  echo "=== judge attempt $attempt rc=$rc $(date -Is) ==="
  [ $rc -eq 0 ] && break
  sleep 120
done
echo "CLOSED_AS_OPEN_JUDGE_STAGE_EXIT $(date -Is)"
