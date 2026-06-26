#!/bin/bash
# After QoQ-verify + Chiron campaigns: run the two missing self-verify passes (Lingshu, MedGemma -> AutoMix
# for all families) then the full batch-1 latency campaign. Waits on the last outputs of the running jobs.
set -u; cd ~/medvlthinker-imgdiff-compute; export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1
DS="PMC-VQA SLAKE VQA-RAD PathVQA MMMU MedXpert-Reasoning MedXpert-Understanding"
rc(){ cat ckpts/acc_gen/$1/*.jsonl 2>/dev/null | wc -l; }
echo "=== waiting for QoQ-verify + Chiron to finish $(date +%H:%M) ==="
for i in $(seq 1 300); do
  [ "$(rc qoq7b/verify)" -ge 8000 ] && [ "$(rc chiron8b/think)" -ge 8000 ] && break
  sleep 60
done
sleep 30
echo "=== Lingshu-7B self-verify (AutoMix) $(date +%H:%M) ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_7b_selfverify_vllm.py --model_path lingshu-medical-mllm/Lingshu-7B --source eval --cap cap320 --datasets $DS --n 4000 --pred_dir_eval ckpts/acc_gen/lingshu7b/cap320 --ckpt_dir ckpts/acc_gen/lingshu7b/verify --tp 1 --gpu_mem 0.85 > logs/lingshu_verify.log 2>&1
echo "=== MedGemma-4B self-verify (AutoMix) $(date +%H:%M) ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path google/medgemma-4b-it --tag medgemma4b_verify --verify --pred_dir ckpts/acc_gen/medgemma4b/nt --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/medgemma4b/verify --tp 1 --gpu_mem 0.85 --max_side 896 --max_images 4 > logs/medgemma_verify.log 2>&1
echo "=== LATENCY campaign (all 5 families x 3 tiers) $(date +%H:%M) ==="
bash run_latency_all.sh
echo "=== FINALIZE_DONE $(date +%H:%M) ==="
for fam in medvlthinker lingshu qoq chiron medgemma; do for t in small_nt big_nt big_think; do echo "  $fam/lat/$t: $(rc $fam/lat/$t) rows"; done; done
