#!/bin/bash
# Everything, re-run on the fixed AdamW path.
#
# ROOT CAUSE of every SIGSEGV on 2026-08-18: torch.optim.AdamW's default multi-tensor (_foreach)
# implementation faults on this CPU build once the parameter tensors get large.  faulthandler traced
# it to torch/optim/adam.py:953.  Not thread count (my second guess), not concurrency (my third).
# foreach=False fixes it: the 28,672-dim x h1024 config that crashed most now fits in 45s at 16
# threads, having previously taken the process down.
#
# The eval and LODO that already completed ran on the buggy path.  They finished rather than
# crashed, so their arithmetic should be sound -- but a memory bug that sometimes corrupts instead
# of faulting cannot be ruled out by argument alone, so both are RE-RUN here and compared against
# the copies kept under _head_*_PREFIX_buggyadam_*.json.
set -u
cd ~/medvlthinker-imgdiff-compute
export HEAD_SWEEP_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16

echo "=== [1/5] one-shot eval, 8 seeds (RE-RUN on the fixed path) ==="
python3 src/training_methods/head_eval_bce.py --stage eval --threads 16 --seeds 8 \
  --out results/cascade_methods/artifacts/head_eval_bce_2026-08-18.json && echo EVAL_OK

echo "=== [2/5] leave-one-dataset-out (RE-RUN) ==="
python3 src/training_methods/head_lodo.py --threads 16 --seeds 3 && echo LODO_OK

echo "=== [3/5] donor curve ==="
python3 src/training_methods/head_domain_curve.py --threads 16 --seeds 3 && echo DONOR_OK

echo "=== [4/5] data x capacity ==="
python3 src/training_methods/head_eval_bce.py --stage curve --threads 16 \
  --out results/cascade_methods/artifacts/head_curve_bce_2026-08-18.json && echo CURVE_OK

echo "=== [5/5] sweep rounds 2 and 3 ==="
for i in 0 1 2; do
  python3 src/training_methods/head_sweep.py --stage main2 --threads 5 --shard $i --nshard 3 \
    > logs/head_sweep_main2_f_s${i}.log 2>&1 &
done
wait
for i in 0 1 2; do
  python3 src/training_methods/head_sweep.py --stage final --threads 5 --shard $i --nshard 3 \
    --seeds 4 > logs/head_sweep_final_f_s${i}.log 2>&1 &
done
wait
echo "HEAD_ALL_STABLE_DONE $(date -u +%H:%M:%S)"
