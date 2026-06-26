#!/bin/bash
# Batch-1 latency measurement for ALL families x 3 tiers (small-nt@cap320, big-nt@cap320, big-think@fullres).
# --batch1 = one request at a time (max_num_seqs=1), records real per-sample latency_s. n per benchmark is
# small (latency(gen) fit only). Qwen families via run_vlm_eval; InternVL/Gemma via run_peer_eval. From repo root.
set -u; cd ~/medvlthinker-imgdiff-compute; export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1
DS="PMC-VQA SLAKE VQA-RAD PathVQA MMMU MedXpert-Reasoning MedXpert-Understanding"
N=6
qwen_lat () {  # $1=fam  $2=small_path  $3=big_path
  echo "=== [$1] latency small-nt@cap320 $(date +%H:%M) ==="
  CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_vlm_eval.py --model_path "$2" --tag lat --arm nothink --cap cap320 --batch1 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/$1/lat/small_nt --tp 1 --gpu_mem 0.85 > logs/lat_${1}_s.log 2>&1
  echo "=== [$1] latency big-nt@cap320 (TP=2) ==="
  python3 src/labeling/run_vlm_eval.py --model_path "$3" --tag lat --arm nothink --cap cap320 --batch1 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/$1/lat/big_nt --tp 2 --gpu_mem 0.90 > logs/lat_${1}_bnt.log 2>&1
  echo "=== [$1] latency big-think@fullres (TP=2) ==="
  python3 src/labeling/run_vlm_eval.py --model_path "$3" --tag lat --arm think --cap fullres --batch1 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/$1/lat/big_think --tp 2 --gpu_mem 0.90 --max_tokens 1536 > logs/lat_${1}_bt.log 2>&1
}
qwen_lat medvlthinker /data/dan/weights/MedVLThinker-7B-RL_m23k /data/dan/weights/MedVLThinker-32B-RL_m23k
qwen_lat lingshu lingshu-medical-mllm/Lingshu-7B lingshu-medical-mllm/Lingshu-32B
qwen_lat qoq ddvd233/QoQ-Med-VL-7B ddvd233/QoQ-Med-VL-32B

# ---- Chiron (InternVL3) via run_peer_eval ----
echo "=== [chiron] latency $(date +%H:%M) ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path manglu3935/Chiron-o1-2B --tag lat --batch1 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/chiron/lat/small_nt --tp 1 --gpu_mem 0.60 --max_images 4 --max_model_len 12288 > logs/lat_chiron_s.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path manglu3935/Chiron-o1-8B --tag lat --batch1 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/chiron/lat/big_nt --tp 1 --gpu_mem 0.85 --max_images 4 --max_model_len 12288 > logs/lat_chiron_bnt.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path manglu3935/Chiron-o1-8B --tag lat --think --batch1 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/chiron/lat/big_think --tp 1 --gpu_mem 0.85 --max_images 4 --max_tokens 1024 --max_model_len 12288 > logs/lat_chiron_bt.log 2>&1

# ---- MedGemma (Gemma3) via run_peer_eval ----
echo "=== [medgemma] latency $(date +%H:%M) ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path google/medgemma-4b-it --tag lat --batch1 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/medgemma/lat/small_nt --tp 1 --gpu_mem 0.60 --max_side 896 --max_images 4 > logs/lat_medgemma_s.log 2>&1
python3 src/labeling/run_peer_eval.py --model_path google/medgemma-27b-it --tag lat --batch1 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/medgemma/lat/big_nt --tp 2 --gpu_mem 0.90 --max_side 896 --max_images 4 > logs/lat_medgemma_bnt.log 2>&1
python3 src/labeling/run_peer_eval.py --model_path google/medgemma-27b-it --tag lat --think --batch1 --datasets $DS --n $N --ckpt_dir ckpts/acc_gen/medgemma/lat/big_think --tp 2 --gpu_mem 0.90 --max_side 896 --max_images 4 --max_tokens 1024 > logs/lat_medgemma_bt.log 2>&1
echo "=== ALL_LATENCY_DONE $(date +%H:%M) ==="
