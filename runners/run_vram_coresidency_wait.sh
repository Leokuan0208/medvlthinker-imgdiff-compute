#!/usr/bin/env bash
# run_vram_coresidency_wait.sh -- ATTACK 4: wait (do not preempt) for a card with enough free VRAM to
# attempt the DIRECT 7B+32B co-load, then run it once.  Never kills another process; if no card ever
# frees, the artifact records "NOT ATTEMPTED -- insufficient free VRAM", which is the honest outcome.
# The 7B-bf16 + 32B-nf4 variant needs ~55 GiB; the bf16+bf16 variant needs ~82 GiB and is expected to
# be skipped on an 80 GB card (that is the point -- it does not fit).
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=/data/dan/hf_cache
export PYTHONPATH=/home/jamesyang/.pylibs_vram
need=${NEED_GIB:-55}
for i in $(seq 1 540); do            # up to ~3 h of polling at 20 s
  while read -r idx free; do
    if [ "$(( free / 1024 ))" -ge "$need" ]; then
      echo "[cores] gpu $idx has $(( free / 1024 )) GiB free at $(date +%H:%M:%S) -- launching"
      CUDA_VISIBLE_DEVICES=$idx python3 src/cascade/vram_levers.py --part cores \
          >> logs/vram_levers_cores_2026-08-12.log 2>&1
      echo "[cores] exit=$? at $(date +%H:%M:%S)"
      exit 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',')
  sleep 20
done
echo "[cores] no card ever had ${need} GiB free -- NOT ATTEMPTED"
