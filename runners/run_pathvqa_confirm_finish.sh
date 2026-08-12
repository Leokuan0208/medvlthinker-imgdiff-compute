#!/bin/bash
# ATTACK D / PART 1 -- everything after config-2 generation: explode -> judge -> verifier-score.
#
# JUDGE DESIGN.  The judge is itself served by vLLM, so a judge run at a different tensor-parallel
# size is a second, uncontrolled config shift sitting on top of the one being measured.  To remove
# it, this runner judges BOTH configurations in ONE judge load at tp=1:
#   * the three config-2 pools + the config-2 greedy control            (new labels)
#   * COPIES of the three config-1 pools + the config-1 greedy control  (re-judged, into *.rj1.*)
# The re-judged config-1 files are also a NULL TEST against round 1's stored tp=2 labels: if they
# agree, judge serving config is not a confound and round 1's labels stand; if they do not, both
# configurations are still compared on the ONE judge run produced here.
# Nothing round 1 wrote is overwritten -- the re-judge lands on new *_rj1.jsonl paths.
#
# The 7B PART-2 verifier scorer keeps GPU 1 for itself throughout; everything here runs on GPU 0.
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
CK=ckpts/openvqa/strong_lingshu_bo
NEED=76800

until grep -qE "PATHVQA_CONFIRM_GEN_DONE|GIVE UP" logs/pathvqa_confirm_gen.log; do sleep 30; done
echo "=== generation finished; exploding ==="

FILES=()
for sd in 0 1 2; do
  SRC=$CK/ckpt_pathvqa_open_c2_bo8_s${sd}.jsonl
  [ -f "$SRC" ] || continue
  python3 src/cascade_methods/explode_sc_for_judge.py "$SRC"
  FILES+=("${SRC%.jsonl}_scexploded.jsonl")
done
FILES+=("$CK/ckpt_pathvqa_open_c2_n1.jsonl")

# re-judge copies of the config-1 files in the SAME judge load (null test + one judge scale)
for sd in 0 1 2; do
  cp -n $CK/ckpt_pathvqa_open_l32_bo8_s${sd}_scexploded.jsonl \
        $CK/ckpt_pathvqa_open_l32_bo8_s${sd}_scexploded_rj1.jsonl
  FILES+=("$CK/ckpt_pathvqa_open_l32_bo8_s${sd}_scexploded_rj1.jsonl")
done
cp -n $CK/ckpt_pathvqa_open_l32_n1.jsonl $CK/ckpt_pathvqa_open_l32_n1_rj1.jsonl
FILES+=("$CK/ckpt_pathvqa_open_l32_n1_rj1.jsonl")

echo "=== judging ${#FILES[@]} files, tp=1 on GPU0 ==="
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  [ "$f" -ge "$NEED" ] && break
  sleep 30
done
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_judge.py --tp 1 --gpu_mem 0.95 --preds "${FILES[@]}"
echo "=== judge done; verifier scoring config 2 (HF, never vLLM) ==="

for sd in 0 1 2; do
  [ -f "$CK/ckpt_pathvqa_open_c2_bo8_s${sd}.jsonl" ] || continue
  while true; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    [ "$f" -ge 25600 ] && break
    sleep 30
  done
  CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/openstrong_score.py \
      --tag c2_bo8_s${sd} --greedy_tag c2_n1 --datasets pathvqa_open
done
echo "PATHVQA_CONFIRM_FINISH_DONE"
