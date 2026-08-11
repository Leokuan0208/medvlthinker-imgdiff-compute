#!/bin/bash
# ATTACK 1 (OPEN-STRONG) -- GPU-polite serial queue, tp=2.
#
# Another agent (Attack 4 / OPEN-DIVERSE) is sharing this machine and holds ~18 GB of HF verifier
# on EACH card, so a tp=1 Lingshu-32B load (62.3 GiB of weights on one card) cannot fit anywhere.
# tp=2 splits the weights to ~31.2 GiB per card, which fits alongside the neighbour with room to
# spare, so this runner uses tp=2 at a DELIBERATELY LOW --gpu_mem so it never pre-empts the other
# agent's job.  It polls until both cards have enough free memory, then launches, and resumes
# per-item on retry.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
CK=ckpts/openvqa/strong_lingshu_bo
GMEM=0.70               # 55.4 GiB per card requested; weights need ~31.2 GiB per card
NEED_MIB=58368          # 57 GiB free needed on EACH card before we try

wait_for_gpus() {
  while true; do
    f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
    if [ "$f0" -ge "$NEED_MIB" ] && [ "$f1" -ge "$NEED_MIB" ]; then return; fi
    sleep 30
  done
}

run_one() {   # ds tag nsamples temp seed nlimit
  local ds=$1 tag=$2 ns=$3 tp=$4 sd=$5 nl=$6
  local out=$CK/ckpt_${ds}_${tag}.jsonl
  for attempt in $(seq 1 20); do
    if [ -f "$out" ]; then
      local have=$(wc -l < "$out")
      if [ "$have" -ge "$nl" ]; then echo "SKIP $ds $tag (have $have >= $nl)"; return 0; fi
    fi
    wait_for_gpus
    echo "=== launch $ds $tag N=$ns T=$tp seed=$sd tp=2 (attempt $attempt) ==="
    CUDA_VISIBLE_DEVICES=0,1 python3 src/cascade_methods/openstrong_gen.py \
      --dataset $ds --ckpt_dir $CK --tp 2 --gpu_mem $GMEM --n $nl \
      --n_samples $ns --temp $tp --seed $sd --tag $tag && return 0
    echo "  attempt $attempt failed; waiting 60s"
    sleep 60
  done
  echo "GIVE UP $ds $tag"
  return 1
}

# --- N3 identity control first: every open cell at N=1, T=0 --------------------------------
run_one slake_open    l32_n1 1 0   0 645
run_one vqa_rad_open  l32_n1 1 0   0 200
run_one pathvqa_open  l32_n1 1 0   0 1500

# --- A1: the N=8 pools, 3 independent generation seeds ---------------------------------------
for sd in 0 1 2; do
  run_one slake_open    l32_bo8_s$sd 8 0.7 $sd 645
  run_one vqa_rad_open  l32_bo8_s$sd 8 0.7 $sd 200
  run_one pathvqa_open  l32_bo8_s$sd 8 0.7 $sd 1500
done
echo "OPENSTRONG_QUEUE_DONE"
