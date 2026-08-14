#!/bin/bash
# run_rejudge_confound.sh -- THE DECISIVE CONFOUND TEST for the 2026-08-13 decoding sweep.
#
# The sweep labelled candidate slots from two DISJOINT sources: a preload cache of judge
# labels harvested from earlier runs, and a fresh judge pass in that session. The share
# drawn from the fresh judge rises monotonically with sampling temperature (0.080 at
# T=0.3 -> 0.500 at T=1.3), so any systematic disagreement between the two label sources
# loads directly onto the temperature axis and could manufacture the reported "colder is
# better" effect. The two sets share zero keys, so agreement is unmeasurable from the
# files on disk -- it has to be measured by re-judging.
#
# This re-judges a stratified sample of 4,047 PRELOAD-labelled slots with the SAME judge
# harness the sweep used for its fresh labels (src/labeling/run_judge.py, MedVLThinker-32B,
# text-only, greedy, max_tokens 2, logprob Yes/No comparison). Agreement between the
# cached label and the re-judged label bounds the confound.
#
# Resumable by idx (run_judge.py appends and skips already-judged rows). Waits for GPU
# rather than evicting a co-tenant. Same memory arithmetic as run_judge_sweep_tight.sh:
# 32 GB of tp=2 weights + <2 GB KV for a max_tokens=2 text-only judge.
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
NEED_GIB=33
TOTAL_GIB=79
for attempt in $(seq 1 200); do
  for w in $(seq 1 240); do
    F0=$(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0) / 1024 ))
    F1=$(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1) / 1024 ))
    MIN=$F0; [ "$F1" -lt "$MIN" ] && MIN=$F1
    if [ "$MIN" -ge "$NEED_GIB" ]; then break; fi
    echo "rejudge waiting: free gpu0=${F0} gpu1=${F1} GiB, need ${NEED_GIB} on both ($(date +%H:%M:%S))"
    sleep 25
  done
  USE=$((MIN - 1)); [ "$USE" -gt 60 ] && USE=60
  MEM=$(python3 -c "print(round($USE/$TOTAL_GIB, 3))")
  echo "=== rejudge attempt $attempt mem=$MEM (min free ${MIN}GiB) $(date) ==="
  CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_judge.py \
      --preds ckpts/openvqa/decoding_sweep/rejudge_confound_in.jsonl \
      --tp 2 --gpu_mem "$MEM" --max_model_len 2048
  rc=$?
  if [ $rc -eq 0 ]; then echo "REJUDGE_DONE"; break; fi
  echo "--- rejudge exit $rc, waiting 25s then resuming ---"
  sleep 25
done
wc -l ckpts/openvqa/decoding_sweep/rejudge_confound_in.judge.jsonl
