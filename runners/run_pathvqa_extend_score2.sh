#!/bin/bash
# ATTACK D / PART 2 -- second scorer process, to finish the arms that matter first.
# The primary scorer walks the candidate universe in item order, so the sc8+greedy arms (E0/E1 --
# the PRE-REGISTERED cheapest intervention) only complete when the whole sc16 pool is also done.
# This process scores ONLY the sc8+greedy candidates. The cache is shared and keyed by
# (idx, normalized answer), so the union of the two runs is identical to one unrestricted run.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
until grep -q "=== judge done" logs/pathvqa_confirm_finish.log 2>/dev/null; do sleep 30; done
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  [ "$f" -ge 30000 ] && break
  sleep 30
done
CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/pathvqa_extend_score.py --only sc8greedy
echo PATHVQA_EXTEND_SCORE2_DONE
