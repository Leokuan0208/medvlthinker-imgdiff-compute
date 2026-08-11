#!/bin/bash
# ATTACK 2 (MCQ-TTA) generation launcher.
#
# GPU ETIQUETTE.  Lingshu-32B needs BOTH A100s (tp=1 is impossible: 66.9 GB weights on an 80 GB
# card -- retrospective 9.2), and the round's plan gives the tp=2 slot to Attack 1 first.  A naive
# "is there free memory right now" check fires during Attack 1's inter-job gaps and collides with
# its next vLLM launch (this happened once at 13:21 and this launcher was killed to protect it).
# So the claim condition is: Attack 1's whole queue is gone, no other 32B tp=2 generator is alive,
# AND both cards have enough free memory -- held stable for STABLE consecutive checks.
#
# Cell order front-loads PMC_VQA: it is the only MCQ cell where the deployed method is not
# literally always-32B-direct, it is permutable, and it carries by far the largest measured
# answer-position error mode (right only 27.6% when it answers "A"). If the slot window is
# short, that is the cell worth having.
# nohup, never tmux.  Resumable per-item JSONL with a per-batch error guard.  Repo root only.
set -u
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HUB_OFFLINE=1 HF_HOME=/data/dan/hf_cache HF_ENDPOINT=https://hf-mirror.com
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
L=logs/mcq_tta_gen.log
NEED_FREE=${NEED_FREE:-58000}      # MiB free required on EACH card
GPU_MEM=${GPU_MEM:-0.62}           # vLLM fraction of TOTAL memory (49.6 GB of 80 GB)
STABLE=${STABLE:-3}                # consecutive 60 s checks the condition must hold
MAXWAIT=${MAXWAIT:-36000}          # seconds

echo "MCQ_TTA_LAUNCHER START $(date) (NEED_FREE=$NEED_FREE STABLE=$STABLE)" >> "$L"
t0=$(date +%s); hits=0
while :; do
  ok=1
  pgrep -f "run_openstrong_queue.sh" >/dev/null 2>&1 && ok=0     # Attack 1's queue still alive
  pgrep -f "openstrong_gen.py"       >/dev/null 2>&1 && ok=0     # its generator still alive
  pgrep -f "open_diverse_gen"        >/dev/null 2>&1 && ok=0     # Attack 4's generator, if any
  if [ "$ok" -eq 1 ]; then
    while read -r used total; do
      free=$((total-used)); [ "$free" -lt "$NEED_FREE" ] && ok=0
    done < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
  fi
  if [ "$ok" -eq 1 ]; then hits=$((hits+1)); else hits=0; fi
  [ "$hits" -ge "$STABLE" ] && break
  now=$(date +%s); el=$((now-t0))
  if [ "$el" -gt "$MAXWAIT" ]; then echo "MCQ_TTA_LAUNCHER TIMEOUT after ${el}s $(date)" >> "$L"; exit 3; fi
  sleep 60
done
echo "SLOT CLAIMED after $(( $(date +%s)-t0 ))s $(date)" >> "$L"
nvidia-smi --query-gpu=index,memory.used --format=csv >> "$L"

/data/dan/medeval_venv/bin/python src/cascade_methods/mcq_tta_generate.py \
  --stage A --tp 2 --gpu_mem "$GPU_MEM" --batch 500 \
  --cells PMC_VQA,VQA_RAD_closed,SLAKE_closed,MedXpertQA-MM,PATH_VQA_closed >> "$L" 2>&1 \
  && echo "MCQ_TTA_GEN_OK $(date)" >> "$L" || echo "MCQ_TTA_GEN_FAIL $(date)" >> "$L"

# measured batch-1 cost endpoint, in the SAME slot (the model is already the only tenant)
/data/dan/medeval_venv/bin/python src/cascade_methods/mcq_tta_cost.py --n 20 --reps 2 \
  --tp 2 --gpu_mem "$GPU_MEM" >> logs/mcq_tta_cost.log 2>&1 \
  && echo "MCQ_TTA_COST_OK $(date)" >> "$L" || echo "MCQ_TTA_COST_FAIL $(date)" >> "$L"
