#!/bin/bash
# Chiron-o1 2B/8B (InternVL3 medical) — all-methods pipeline (run_peer_eval llm.chat; InternVL fixed-tiling).
# max_model_len bumped to 12288 (InternVL tiling is token-heavy; smoke hit 8318>8192).
set -u; cd ~/medvlthinker-imgdiff-compute; export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1
M2=manglu3935/Chiron-o1-2B; M8=manglu3935/Chiron-o1-8B
DS="PMC-VQA SLAKE VQA-RAD PathVQA MMMU MedXpert-Reasoning MedXpert-Understanding"
MML=12288
echo "=== Chiron 2B no-think (small) $(date +%H:%M) ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path $M2 --tag chiron2b_nt --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/chiron2b/nt --tp 1 --gpu_mem 0.55 --max_images 4 --max_model_len $MML > logs/chiron_2b_nt.log 2>&1
echo "=== Chiron 2B self-verify (AutoMix) ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path $M2 --tag chiron2b_verify --verify --pred_dir ckpts/acc_gen/chiron2b/nt --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/chiron2b/verify --tp 1 --gpu_mem 0.55 --max_images 4 --max_model_len $MML > logs/chiron_2b_verify.log 2>&1
echo "=== Chiron 8B no-think + think ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path $M8 --tag chiron8b_nt --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/chiron8b/nt --tp 1 --gpu_mem 0.85 --max_images 4 --max_model_len $MML > logs/chiron_8b_nt.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path $M8 --tag chiron8b_think --think --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/chiron8b/think --tp 1 --gpu_mem 0.85 --max_images 4 --max_tokens 1024 --max_model_len $MML > logs/chiron_8b_think.log 2>&1
echo "=== CHIRON_DONE $(date +%H:%M) ==="
