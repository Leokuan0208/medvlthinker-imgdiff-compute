#!/usr/bin/env bash
# Options 3 GPU chain, part 2 (2026-10-22): LoRA verifier (pooled, cross-fitted) -> generator SFT (pooled,
# cross-fitted, merged, greedy) -> WED as published -> both judges on every new answer. Starts when chain 1 exits.
set -u
M=/home/jamesyang/medvlthinker-imgdiff-compute; WT=$M/.claude/worktrees/exp-2026-10-15; L=$M/logs; D=$M/data/sft_crossfit_2026-10-22
cd $M
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
MEDG=$(ls -d /data/dan/hf_cache/hub/models--google--medgemma-27b-it/snapshots/*/ | head -1)
OPEN8="pathvqa_open slake_open vqa_rad_open radimagenet_open kvasir_x1_open omnimed_open vqamed_open gemex_open"
say(){ echo "[$(date '+%F %T')] $*" >> $L/opt234_chain2.log; }
on(){ g=$1; shift; flock -w 172800 $L/.gpu$g.lock env CUDA_VISIBLE_DEVICES=$g "$@"; }
while kill -0 3893800 2>/dev/null; do sleep 60; done
say "chain2 start (chain1 exited)"

# A. LoRA verifier on the probe's pooled data, one fold per GPU
for f in 0 1; do ( timeout 10h on $f python3 -u $WT/src/exp_2026_10/lora_verifier_crossfit.py --fold $f > $L/opt234_loraver_f$f.log 2>&1
                   say "LoRA verifier fold $f rc=$?" ) & done; wait

# B. generator SFT on gold answers (same data universe), merge, greedy on the fold's held-out questions
for f in 0 1; do (
  timeout 8h on $f python3 -u $WT/src/training_methods/train_cheap_leg.py --manifest data/sft_crossfit_2026-10-22/fold$f.json       --quota '{"open_generic": 1000000}' --out_dir ckpts/train/sft_crossfit_1022_f$f --seed 0 > $L/opt234_sft_train_f$f.log 2>&1
  say "SFT train fold $f rc=$?"
  timeout 2h on $f python3 -u $WT/src/training_methods/merge_cheapleg_lora.py --adapter $M/ckpts/train/sft_crossfit_1022_f$f       --out $M/ckpts/train/sft_crossfit_1022_f${f}_merged > $L/opt234_sft_merge_f$f.log 2>&1
  say "SFT merge fold $f rc=$?"
  for c in $OPEN8; do
    timeout 3h on $f python3 -u $M/src/labeling/run_openvqa.py --model_path $M/ckpts/train/sft_crossfit_1022_f${f}_merged         --ckpt_dir $M/ckpts/openvqa/sft_crossfit_1022 --tp 1 --max_model_len 4096 --dataset $c --tag sft1022f$f         --n_samples 1 --temp 0.0 --cap cap320 --idx_file $D/heldout_${c}_fold$f.json > $L/opt234_sft_gen_${c}_f$f.log 2>&1
    say "SFT greedy $c fold $f rc=$?"
  done ) & done; wait

# C. WED as published on PathVQA (2 shards) + SLAKE + VQA-RAD
CK=$M/ckpts/openvqa/cheap_lingshu7b; WO=/data/dan/exp_2026_10/wed_published; mkdir -p $WO
( timeout 12h on 0 python3 -u $WT/src/exp_2026_10/wed/wed_select.py --pool $CK/ckpt_pathvqa_open_lingshu7b_sc8.jsonl --out $WO/wed_pub_pathvqa_open_s0.jsonl --gpu --cap cap320 --own_sampling --shard 0 --nshard 2 > $L/opt234_wedpub_pathvqa_s0.log 2>&1; say "WED-pub pathvqa s0 rc=$?"
  timeout 6h on 0 python3 -u $WT/src/exp_2026_10/wed/wed_select.py --pool $CK/ckpt_vqa_rad_open_lingshu7b_sc8.jsonl --out $WO/wed_pub_vqa_rad_open.jsonl --gpu --cap cap320 --own_sampling > $L/opt234_wedpub_vqa_rad.log 2>&1; say "WED-pub vqa_rad rc=$?" ) &
( timeout 12h on 1 python3 -u $WT/src/exp_2026_10/wed/wed_select.py --pool $CK/ckpt_pathvqa_open_lingshu7b_sc8.jsonl --out $WO/wed_pub_pathvqa_open_s1.jsonl --gpu --cap cap320 --own_sampling --shard 1 --nshard 2 > $L/opt234_wedpub_pathvqa_s1.log 2>&1; say "WED-pub pathvqa s1 rc=$?"
  timeout 6h on 1 python3 -u $WT/src/exp_2026_10/wed/wed_select.py --pool $CK/ckpt_slake_open_lingshu7b_sc8.jsonl --out $WO/wed_pub_slake_open.jsonl --gpu --cap cap320 --own_sampling > $L/opt234_wedpub_slake.log 2>&1; say "WED-pub slake rc=$?" ) &
wait

# D. both judges on every new answer (SFT greedy + WED-published picks)
python3 $WT/src/exp_2026_10/build_xjudge_inputs.py new >> $L/opt234_chain2.log 2>&1
timeout 6h flock -w 172800 $L/.gpu0.lock flock -w 172800 $L/.gpu1.lock env CUDA_VISIBLE_DEVICES=0,1   python3 -u $M/src/labeling/run_judge.py --tp 2 --preds /data/dan/exp_2026_10/xjudge_1022/new_all.jsonl > $L/opt234_new_judge.log 2>&1
say "judge of record on new answers rc=$?"
for g in 0 1; do timeout 6h on $g python3 -u $M/src/labeling/run_judge.py --judge_model "$MEDG" --tp 1 --gpu_mem 0.90     --preds /data/dan/exp_2026_10/xjudge_1022/new_part$g.jsonl > $L/opt234_xjudge_new_gpu$g.log 2>&1 & done; wait
python3 $WT/src/exp_2026_10/build_xjudge_inputs.py merge >> $L/opt234_chain2.log 2>&1
say "chain2 DONE"
