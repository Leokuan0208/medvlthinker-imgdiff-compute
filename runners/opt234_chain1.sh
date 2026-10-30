#!/usr/bin/env bash
# Options 2+3 GPU chain, part 1 (2026-10-22). Starts each GPU's work the moment THAT GPU's deep-layer extraction
# queue exits. Every step logs "produced or failed" and moves on -- a failed producer never blocks the chain.
set -u
M=/home/jamesyang/medvlthinker-imgdiff-compute; WT=$M/.claude/worktrees/exp-2026-10-15; L=$M/logs
cd $M
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
SUP0=3864318; SUP1=3864319                                  # deep_1021b_gpu{0,1} supervisors
SNAP32=/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/snapshots/36b98277cacb60db86f34b75ce0540b1ea35183c/
MEDG=$(ls -d /data/dan/hf_cache/hub/models--google--medgemma-27b-it/snapshots/*/ | head -1)
G32=$M/ckpts/openvqa/strong_lingshu32b_c320; mkdir -p $G32
OPEN8="pathvqa_open slake_open vqa_rad_open radimagenet_open kvasir_x1_open omnimed_open vqamed_open gemex_open"
say(){ echo "[$(date '+%F %T')] $*" >> $L/opt234_chain1.log; }
waitp(){ while kill -0 "$1" 2>/dev/null; do sleep 60; done; }
on(){ g=$1; shift; flock -w 172800 $L/.gpu$g.lock env CUDA_VISIBLE_DEVICES=$g "$@"; }
say "chain1 start; waiting for the extraction supervisors $SUP0 / $SUP1"

( waitp $SUP0; say "GPU0 free"
  timeout 2h on 0 python3 -u $WT/src/exp_2026_10/predict_modality.py > $L/opt234_modality.log 2>&1; say "modality rc=$?"
  timeout 5h on 0 python3 -u $WT/src/exp_2026_10/usc_select.py --policy T07 > $L/opt234_usc_T07.log 2>&1; say "USC T07 rc=$?" ) &
A=$!
( waitp $SUP1; say "GPU1 free"
  timeout 5h on 1 python3 -u $WT/src/exp_2026_10/usc_select.py --policy Tmap > $L/opt234_usc_Tmap.log 2>&1; say "USC Tmap rc=$?" ) &
B=$!
wait $A $B

# MedGemma-27B on the temperature policies' unlabelled picks, one half per GPU
if [ -s /data/dan/exp_2026_10/xjudge_missing_temp_labelfree.jsonl ]; then
  python3 $WT/src/exp_2026_10/build_xjudge_inputs.py missing >> $L/opt234_chain1.log 2>&1
  for g in 0 1; do timeout 8h on $g python3 -u $M/src/labeling/run_judge.py --judge_model "$MEDG" --tp 1 --gpu_mem 0.90 \
      --preds /data/dan/exp_2026_10/xjudge_1022/lf_part$g.jsonl > $L/opt234_xjudge_lf_gpu$g.log 2>&1 & done; wait
  say "MedGemma on temp-policy picks: $(cat /data/dan/exp_2026_10/xjudge_1022/lf_part*.judge.jsonl 2>/dev/null | wc -l) labels"
else say "no missing-label file from temp_labelfree; MedGemma step skipped"; fi

# Lingshu-32B greedy, same prompt / cap / token limit as the 7B greedy (tp=2: both GPUs). Hard timeout per cell
# because Lingshu-32B with tp=2 hangs deterministically in an NCCL collective under MedEvalKit (CLAUDE.md s8).
for c in $OPEN8; do
  timeout 3h flock -w 172800 $L/.gpu0.lock flock -w 172800 $L/.gpu1.lock env CUDA_VISIBLE_DEVICES=0,1 \
    python3 -u $M/src/labeling/run_openvqa.py --model_path $SNAP32 --ckpt_dir $G32 --tp 2 --max_model_len 4096 \
    --dataset $c --tag lingshu32b --n_samples 1 --temp 0.0 --cap cap320 > $L/opt234_g32_$c.log 2>&1
  say "32B greedy $c rc=$? rows=$(wc -l < $G32/ckpt_${c}_lingshu32b.jsonl 2>/dev/null)"
done
# judge of record (MedVLThinker-32B, tp=2) on the 32B answers
P=$(for c in $OPEN8; do [ -s $G32/ckpt_${c}_lingshu32b.jsonl ] && echo $G32/ckpt_${c}_lingshu32b.jsonl; done)
timeout 6h flock -w 172800 $L/.gpu0.lock flock -w 172800 $L/.gpu1.lock env CUDA_VISIBLE_DEVICES=0,1 \
  python3 -u $M/src/labeling/run_judge.py --tp 2 --preds $P > $L/opt234_g32_judge.log 2>&1; say "32B judge-of-record rc=$?"
# cross-family judge (MedGemma-27B) on the 32B answers
python3 $WT/src/exp_2026_10/build_xjudge_inputs.py g32 >> $L/opt234_chain1.log 2>&1
for g in 0 1; do timeout 6h on $g python3 -u $M/src/labeling/run_judge.py --judge_model "$MEDG" --tp 1 --gpu_mem 0.90 \
    --preds /data/dan/exp_2026_10/xjudge_1022/g32_part$g.jsonl > $L/opt234_xjudge_g32_gpu$g.log 2>&1 & done; wait
python3 $WT/src/exp_2026_10/build_xjudge_inputs.py merge >> $L/opt234_chain1.log 2>&1
say "chain1 DONE"
