#!/bin/bash
# THE ONE-SHOT EVAL, run against ROUND 1's winner rather than waiting for rounds 2-3.
#
# This is methodologically clean: round 1 was a COMPLETE 74-config CV search with eval untouched,
# so taking its argmax and reading eval once is the pre-specified protocol.  Rounds 2-3 refine WHICH
# bce config wins; if they change the answer that is a SECOND eval read and will be declared as one.
# Waiting for them first would cost hours on 28,672-dim h1024 configs that round 1 already showed
# buy nothing at h256 -- and those are the configs that keep segfaulting.
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
python3 src/training_methods/head_eval_bce.py --stage eval --threads 16 --seeds 8 \
  --out results/cascade_methods/artifacts/head_eval_bce_2026-08-18.json
echo "HEAD_EVAL_NOW_DONE $(date -u +%H:%M:%S)"
python3 src/training_methods/head_eval_bce.py --stage curve --threads 16 \
  --out results/cascade_methods/artifacts/head_curve_bce_2026-08-18.json
echo "HEAD_CURVE_NOW_DONE $(date -u +%H:%M:%S)"
