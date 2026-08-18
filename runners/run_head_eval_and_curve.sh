#!/bin/bash
# Answers Leo's three questions, in order, chained behind the CPU experiments already queued.
#   1. retrain with bce and check eval  -> --stage eval  (ONE-SHOT, 3 declared arms)
#   2. does more training data help?    -> --stage curve (data axis)
#   3. or do we need a bigger head?     -> --stage curve (capacity axis + train-vs-CV gap)
cd ~/medvlthinker-imgdiff-compute
echo "waiting for HEAD_DOMAIN_CURVE_DONE ..."
until grep -q "HEAD_DOMAIN_CURVE_DONE" logs/head_domain_curve.log 2>/dev/null; do sleep 60; done
echo "prior chain done at $(date -u +%H:%M:%S)"

python3 src/training_methods/head_eval_bce.py --stage eval  --threads 24 --seeds 8
echo "HEAD_EVAL_BCE_DONE"
python3 src/training_methods/head_eval_bce.py --stage curve --threads 24
echo "HEAD_CURVE_DONE"
