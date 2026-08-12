#!/usr/bin/env bash
# Mandatory image ablations for the CV-selected vision arm (pre-registration:
# artifacts/vision_verifier_2026-08-12_preregistration.json:mandatory_image_ablation).
#
#   blank / noise : the EVAL vision cache re-extracted from mid-grey / uniform-noise images of the
#                   SAME pixel size, so the merged patch grid and every sequence length are
#                   unchanged and only image CONTENT is destroyed.
#   perm          : permutation null -- every question keeps a REAL image vector, but another
#                   question's (deranged, no fixed points). Any gain surviving this is not coming
#                   from image content.
#
# Resumable per (arm, tag, seed). Pinned to the same 8 threads as every other arm.
#   ARM=L_prod bash runners/run_visverif_ablations.sh
set -u
cd ~/medvlthinker-imgdiff-compute || exit 1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

ARM=${ARM:?set ARM to the CV-selected vision arm}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

for spec in "blank --ablate_eval blank" "noise --ablate_eval noise" "perm --perm_vision 1"; do
  tag=${spec%% *}; args=${spec#* }
  echo "[run] arm=$ARM tag=$tag at $(date -Is)"
  python3 -u src/training_methods/vision_verifier_fit.py \
      --arm "$ARM" --seeds $SEEDS --threads 8 --tag "$tag" $args \
      >> "logs/visverif_${ARM}_${tag}.log" 2>&1
  echo "[done] arm=$ARM tag=$tag rc=$? at $(date -Is)"
done
echo "[ALL DONE] $(date -Is)"
