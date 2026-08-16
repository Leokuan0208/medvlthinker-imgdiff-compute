#!/bin/bash
# BUILD 3 SECONDARY -- plain SELF-CONSISTENCY (majority vote over 8 T=0.4 samples) vs 7B greedy on the
# two INTRINSICALLY multiple-choice cells, PMC_VQA (6,000-item pre-registered subsample) and
# MedXpertQA-MM (2,000).  Training-free, generation cost only, NO verifier and NO 32B.
#
# These two cells are expected to stay OUTSIDE the verifier claim.  If majority vote beats greedy the
# paper can still say something about them; if not, the claim is scoped and that is said plainly.
#
# Lingshu-7B, tp=1, gpu_mem 0.55 so it fits beside a co-tenant.  Waits for free VRAM rather than
# evicting anything.  Resumable: closed_as_open_mcq_sc.py appends per item and skips completed i.
# vLLM is safe here -- no adapter is loaded (the visual.* LoRA landmine applies to adapter scoring).
cd /home/jamesyang/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0 TORCHDYNAMO_DISABLE=1
NEED_GIB=48
for attempt in $(seq 1 60); do
  GPU=""
  for w in $(seq 1 480); do
    for g in 0 1; do
      F=$(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g) / 1024 ))
      if [ "$F" -ge "$NEED_GIB" ]; then GPU=$g; break; fi
    done
    [ -n "$GPU" ] && break
    echo "mcq_sc waiting for ${NEED_GIB}GiB on one GPU ($(date +%H:%M:%S))"
    sleep 25
  done
  [ -z "$GPU" ] && { echo "mcq_sc: never got a GPU"; exit 1; }
  echo "=== mcq_sc gen attempt $attempt on gpu$GPU $(date) ==="
  CUDA_VISIBLE_DEVICES=$GPU python3 src/cascade_methods/closed_as_open_mcq_sc.py --stage gen
  rc=$?
  if [ $rc -eq 0 ]; then echo "MCQ_SC_STAGE_OK"; break; fi
  echo "--- mcq_sc gen exit $rc, waiting 30s then resuming ---"
  sleep 30
done
echo "MCQ_SC_RUNNER_EXIT $(date)"
