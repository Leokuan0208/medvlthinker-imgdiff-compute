#!/usr/bin/env bash
# End-to-end smoke of every chain-2 GPU script (2026-10-22), each into /data/dan/exp_2026_10/smoke/ only.
# Run as: runners/gpu_run.sh G runners/smoke_1022.sh   (one GPU; ~20 min)
set -u
M=/home/jamesyang/medvlthinker-imgdiff-compute; WT=$M/.claude/worktrees/exp-2026-10-15; S=/data/dan/exp_2026_10/smoke
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
mkdir -p $S; cd $M
say(){ echo "[$(date '+%F %T')] SMOKE $*"; }
python3 -c "import json;json.dump(json.load(open('$M/data/sft_crossfit_2026-10-22/heldout_vqa_rad_open_fold0.json'))[:6],open('$S/idx6.json','w'))"
python3 -u $WT/src/exp_2026_10/lora_verifier_crossfit.py --fold 0 --smoke 32 > $S/loraver.log 2>&1; say "loraver rc=$?"
python3 -u $WT/src/training_methods/train_cheap_leg.py --manifest data/sft_crossfit_2026-10-22/fold0.json \
  --quota '{"open_generic": 1000000}' --out_dir $S/sft_f0 --seed 0 --debug_n 3 > $S/sft_debug.log 2>&1; say "sft debug_n rc=$?"
python3 -u $WT/src/training_methods/train_cheap_leg.py --manifest data/sft_crossfit_2026-10-22/fold0.json \
  --quota '{"open_generic": 1000000}' --out_dir $S/sft_f0 --seed 0 --deadline_s 300 > $S/sft_train.log 2>&1; say "sft train rc=$?"
python3 -u $WT/src/training_methods/merge_cheapleg_lora.py --adapter $S/sft_f0 --out $S/sft_f0_merged > $S/sft_merge.log 2>&1; say "sft merge rc=$?"
python3 -u $M/src/labeling/run_openvqa.py --model_path $S/sft_f0_merged --ckpt_dir $S/sft_gen --tp 1 --max_model_len 4096 \
  --dataset vqa_rad_open --tag sftsmoke --n_samples 1 --temp 0.0 --cap cap320 --idx_file $S/idx6.json > $S/sft_gen.log 2>&1; say "sft greedy rc=$?"
python3 -u $WT/src/exp_2026_10/wed/wed_select.py --pool $M/ckpts/openvqa/cheap_lingshu7b/ckpt_vqa_rad_open_lingshu7b_sc8.jsonl \
  --out $S/wed_pub_vqa_rad.jsonl --gpu --cap cap320 --own_sampling --limit 3 > $S/wed.log 2>&1; say "wed own_sampling rc=$?"
say DONE
