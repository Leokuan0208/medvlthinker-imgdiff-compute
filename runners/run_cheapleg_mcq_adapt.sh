#!/bin/bash
# Attack B -- MedEvalKit MCQ/closed suite for the ADAPTED arm, driven independently of the chain.
#
# WHY SEPARATE.  run_cheapleg_adapt_chain.sh pinned every stage to GPU 0 and ran them strictly in
# series.  The judge (MedVLThinker-32B, tp=1) needs ~75 GB FREE on ONE device and sibling research
# agents are holding memory on both GPUs, so serialising the MCQ suite behind the judge could cost
# hours for no reason -- the two stages share no inputs.  The chain shell was retired (its open-text
# generation child was left running, reparented to init); the judge+score stage runs on GPU 1 under
# runners/run_cheapleg_finish_gpu1.sh, and this runs the MCQ suite on GPU 0 in parallel.
#
# The MCQ command itself is UNCHANGED -- this only waits for the open-text generation to release its
# vLLM engine and for the device to have room, then calls run_cheapleg_mcq.sh, which is the same
# byte-for-byte harness invocation that produced the matched base control arm.
set -u
cd ~/medvlthinker-imgdiff-compute
L=logs/cheapleg_mcq_adapt_driver.log
GPU="${1:-0}"
NEED="${2:-46000}"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

say "waiting for adapted open-text generation to finish (releases its vLLM engine on gpu 0) ..."
until grep -q "CHEAPLEG_OPEN_GEN_DONE" logs/cheapleg_open_adapt7b_s0.log 2>/dev/null; do sleep 30; done
say "open gen done"

say "waiting for >=${NEED} MB free on gpu $GPU ..."
t0=$SECONDS
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  [ $((81920-used)) -ge "$NEED" ] && break
  [ $((SECONDS-t0)) -gt 36000 ] && { say "WAIT_MEM_TIMEOUT gpu$GPU"; exit 1; }
  sleep 60
done
say "running MedEvalKit MCQ suite for the adapted arm on gpu $GPU"

bash runners/run_cheapleg_mcq.sh \
  "$PWD/ckpts/train/merged_cheapleg_s0" cheapleg_adapt7b_s0 "$GPU"
say "CHEAPLEG_MCQ_ADAPT_DRIVER_DONE"
