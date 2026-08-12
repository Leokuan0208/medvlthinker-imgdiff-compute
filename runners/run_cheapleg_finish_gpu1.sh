#!/bin/bash
# Attack B -- run the ADAPTED arm's finish stage (explode -> judge -> frozen-verifier score) on GPU 1.
#
# WHY THIS EXISTS.  runners/run_cheapleg_adapt_chain.sh pins every stage to the GPU it was started on
# (GPU 0).  The judge is MedVLThinker-32B at tp=1 and needs ~75 GB FREE on that single device, i.e.
# the device must be almost empty.  Sibling research agents are running their own jobs in this same
# repo and are holding ~11 GB on GPU 0, so the chain's judge would sit in wait_mem until its 3-hour
# timeout and then abort.  GPU 1 came free at 01:17.
#
# SAFETY.  run_cheapleg_arm_finish.sh is idempotent and resumable -- it skips any dataset whose judge
# file is already complete and skips scoring entirely if the transfer dumps exist.  A flock ensures
# the GPU-0 copy and this GPU-1 copy can never both start a 32B judge; whichever gets the lock does
# the work, the other then finds it done and skips.  Nothing is recomputed and nothing is overwritten.
set -u
cd ~/medvlthinker-imgdiff-compute
L=logs/cheapleg_finish_gpu1.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

say "waiting for adapted open-text generation to finish ..."
until grep -q "CHEAPLEG_OPEN_GEN_DONE" logs/cheapleg_open_adapt7b_s0.log 2>/dev/null; do sleep 30; done
say "open gen done; taking the arm-finish lock"

exec 9>/tmp/cheapleg_arm_finish.lock
flock 9
say "lock acquired; running arm finish for adapt7b_s0 on gpu 1"
bash runners/run_cheapleg_arm_finish.sh adapt7b_s0 1
say "arm finish rc=$? (lock released on exit)"
