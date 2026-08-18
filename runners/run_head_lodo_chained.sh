#!/bin/bash
# Leave-one-dataset-out transfer test. Chained behind the full sweep chain so it gets the whole
# machine -- it fits 3 configs x 4 datasets x (1 out-of-domain + 5 CV folds) x 3 seeds.
cd ~/medvlthinker-imgdiff-compute
echo "waiting for HEAD_SWEEP_CHAIN_DONE ..."
until grep -q "HEAD_SWEEP_CHAIN_DONE" logs/head_sweep_chain.log 2>/dev/null; do sleep 60; done
echo "sweep chain done at $(date -u +%H:%M:%S); starting LODO"
python3 src/training_methods/head_lodo.py --threads 24 --seeds 3
echo "HEAD_LODO_DONE"
