#!/bin/bash
# The donor curve: how much in-domain data the head needs before it stops being a TIE.
# Chained behind LODO so both transfer experiments get the whole machine in turn.
cd ~/medvlthinker-imgdiff-compute
echo "waiting for HEAD_LODO_DONE ..."
until grep -q "HEAD_LODO_DONE" logs/head_lodo.log 2>/dev/null; do sleep 60; done
echo "LODO done at $(date -u +%H:%M:%S); starting the donor curve"
python3 src/training_methods/head_domain_curve.py --threads 24 --seeds 3
echo "HEAD_DOMAIN_CURVE_DONE"
