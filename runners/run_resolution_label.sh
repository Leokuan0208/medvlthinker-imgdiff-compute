#!/bin/bash
# SWEEP 2 -- labelling stage for the open-text generator resolution sweep.
#   1. verify  : score every NEW (item, RAW candidate string) with the deployed clean disjoint
#                LoRA verifier at its own deployed max_pixels (1,003,520), HF transformers only
#                (vLLM 0.9.0.1 drops all 192 visual.* LoRA modules).
#   2. judge   : label every NEW (item, answer text) with the project's existing judge
#                (src/labeling/run_judge.py, MedVLThinker-32B, text-only). The cache means the
#                judge only ever sees answer strings it has not seen, so the control arm and the
#                swept arms carry byte-identical labels wherever their text coincides.
#
# ORDER: verifier FIRST. It is the long pole (tens of thousands of batch-1 forwards) and it needs
# only ~26 GiB, while the 32B judge needs a nearly-empty card. On this shared box the verifier is
# far more likely to get a slot, so running it first means a card that frees up late still buys
# the judge pass, whereas the reverse order would risk losing both.
# Both stages are cache-backed and resumable; both wait for free VRAM and never kill a tenant.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 TORCHDYNAMO_DISABLE=1
L=logs/resolution_label_2026-08-13.log
say(){ echo "[$(date +%F\ %T)] $*" >> "$L"; }
pick_gpu(){ nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
   | awk -v n="$1" -F', *' '{ if ($2-$3 >= n) { print $1; exit } }'; }
# BOUNDED wait: on this box both cards carry another tenant's jobs, so an unbounded wait would
# stall the whole stage. Times out and returns empty; the caller then skips that stage and the
# artifact records what was not labelled rather than hanging.
wait_gpu(){ local G="" t=0 lim="${2:-1800}"
  while [ -z "$G" ] && [ "$t" -lt "$lim" ]; do G=$(pick_gpu "$1"); [ -z "$G" ] && { sleep 60; t=$((t+60)); }; done
  echo "$G"; }

SW=ckpts/openvqa/resolution_sweep

say "LABEL STAGE START"
# ---- 1. verifier ------------------------------------------------------------------------------
for VA in 1 2 3 4 5 6; do
  G=$(wait_gpu "${VERIF_MIB:-26000}" "${VERIF_WAIT:-1800}")
  [ -z "$G" ] && { say "verifier attempt=$VA: no card with ${VERIF_MIB:-26000} MiB in ${VERIF_WAIT:-1800}s"; continue; }
  say ">> verifier scoring on GPU $G (attempt $VA)"
  CUDA_VISIBLE_DEVICES=$G python3 src/cascade_methods/resolution_verifier_score.py \
      ${NULLTEST:+--null_test $NULLTEST} ${VBATCH:+--batch $VBATCH} ${VSUB:+--subsample $VSUB} \
      --run >> "$L" 2>&1 \
    && { say "verifier OK"; break; } || { say "verifier FAIL attempt=$VA"; sleep 120; }
done

# ---- 2. judge ---------------------------------------------------------------------------------
python3 src/cascade_methods/resolution_judge_cache.py build >> "$L" 2>&1
N=$(wc -l < "$SW/judge_todo.jsonl")
say "judge_todo = $N rows"
if [ "$N" -gt 0 ]; then
  for JA in 1 2 3; do
    G=$(wait_gpu "${JUDGE_MIB:-68000}" "${JUDGE_WAIT:-900}")
    [ -z "$G" ] && { say "judge attempt=$JA: no card with ${JUDGE_MIB:-68000} MiB in ${JUDGE_WAIT:-900}s"; continue; }
    say ">> judge on GPU $G (MedVLThinker-32B, tp=1, attempt $JA)"
    CUDA_VISIBLE_DEVICES=$G python3 src/labeling/run_judge.py --tp 1 \
        --gpu_mem "${JUDGE_MEM:-0.82}" \
        --max_model_len 2048 --preds "$SW/judge_todo.jsonl" >> "$L" 2>&1 \
      && { say "judge OK"; break; } || { say "judge FAIL attempt=$JA"; sleep 180; }
  done
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 20
fi
python3 src/cascade_methods/resolution_judge_cache.py merge >> "$L" 2>&1
say "judge merged"
say "LABEL STAGE DONE"
