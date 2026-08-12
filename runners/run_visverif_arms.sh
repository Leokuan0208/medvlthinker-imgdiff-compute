#!/usr/bin/env bash
# Fit the remaining vision-aware-verifier arms, one at a time, from the repo root.
# Resumable at the (arm, seed) level: vision_verifier_fit.py skips a seed whose part .npz exists.
# CPU-only; every arm is pinned to the SAME 8 threads because the trainer's batch permutation is
# thread-count sensitive (ckpts/train/genframe_head_ens8/recipe.json).
set -u
cd ~/medvlthinker-imgdiff-compute || exit 1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

SEEDS="0 1 2 3 4 5 6 7 8 9"
ARMS=${ARMS:-"Vmean L_maxsim L_prod_sim xattn"}

for arm in $ARMS; do
  echo "[run] arm=$arm at $(date -Is)"
  python3 -u src/training_methods/vision_verifier_fit.py \
      --arm "$arm" --seeds $SEEDS --threads 8 >> "logs/visverif_${arm}.log" 2>&1
  echo "[done] arm=$arm rc=$? at $(date -Is)"
done
echo "[ALL DONE] $(date -Is)"
