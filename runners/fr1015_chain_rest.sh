#!/usr/bin/env bash
# fr1015_chain_rest.sh -- replaces chains 3-6 so GPU 0 does not sit idle while GPU 1 finishes stage 2.
#   stage 3 GPU 0 starts NOW (its inputs -- the stage-1 pools/labels -- are complete);
#   stage 3 GPU 1 starts when the stage-1/2 chain (pid $1) EXITS;
#   stages 4, 5, 6 then run in order exactly as fr1015_chain4/5/6.sh did.
# Every wait is on process EXIT (success or failure), never on an artifact.
# Usage: nohup bash fr1015_chain_rest.sh <pid_of_fr1015_chain.sh> > logs/fr1015_chain_rest.log 2>&1 &
cd /home/jamesyang/medvlthinker-imgdiff-compute || exit 1
WT=/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15
CK=/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b
SV="python3 src/reporting/supervisor.py --queue"
echo "[$(date +%T)] stage 3 GPU 0 starting"
$SV $WT/runners/fr1015_stage3_gpu0.json > logs/fr1015_stage3_supervisor_gpu0.log 2>&1 &
S30=$!
while kill -0 "$1" 2>/dev/null; do sleep 120; done
echo "[$(date +%T)] stage 1+2 chain exited; stage 3 GPU 1 starting"
$SV $WT/runners/fr1015_stage3_gpu1.json > logs/fr1015_stage3_supervisor_gpu1.log 2>&1 &
S31=$!
wait $S30 $S31
echo "[$(date +%T)] stage 3 done; stage 4 starting"
for g in 0 1; do $SV $WT/runners/fr1015_stage4_gpu$g.json > logs/fr1015_stage4_supervisor_gpu$g.log 2>&1 & done
wait
echo "[$(date +%T)] stage 4 done; stage 5 (MedGemma-27B cross-family judge) starting"
python3 $WT/src/exp_2026_10/build_xjudge_fr.py
SNAP=$(ls -d /data/dan/hf_cache/hub/models--google--medgemma-27b-it/snapshots/*/ | head -1)
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
for g in 0 1; do
  flock -w 172800 logs/.gpu$g.lock env CUDA_VISIBLE_DEVICES=$g python3 -u src/labeling/run_judge.py \
    --judge_model "$SNAP" --tp 1 --gpu_mem 0.90 \
    --preds /data/dan/exp_2026_10/xjudge/fr_medgemma27b_part$g.jsonl > logs/fr1015_xjudge_gpu$g.log 2>&1 &
done
wait
cat /data/dan/exp_2026_10/xjudge/fr_medgemma27b_part0.judge.jsonl /data/dan/exp_2026_10/xjudge/fr_medgemma27b_part1.judge.jsonl \
  > /data/dan/exp_2026_10/xjudge/fr_medgemma27b.judge.jsonl
echo "[$(date +%T)] stage 5 done: $(wc -l < /data/dan/exp_2026_10/xjudge/fr_medgemma27b.judge.jsonl) labels; stage 6 (WED) starting"
O=/data/dan/exp_2026_10/wed; mkdir -p $O
( flock -w 172800 logs/.gpu0.lock env CUDA_VISIBLE_DEVICES=0 python3 -u $WT/src/exp_2026_10/wed/wed_select.py --gpu \
    --pool $CK/ckpt_pathvqa_open_lingshu7bFR_sc8.jsonl --out $O/wed_pathvqa_open_FR.jsonl > logs/fr1015_wed_pathvqa.log 2>&1 ) &
( for c in vqa_rad_open slake_open; do
    flock -w 172800 logs/.gpu1.lock env CUDA_VISIBLE_DEVICES=1 python3 -u $WT/src/exp_2026_10/wed/wed_select.py --gpu \
      --pool $CK/ckpt_${c}_lingshu7bFR_sc8.jsonl --out $O/wed_${c}_FR.jsonl > logs/fr1015_wed_${c}.log 2>&1
  done ) &
wait
echo "[$(date +%T)] stage 6 (WED) done"; wc -l $O/*.jsonl
