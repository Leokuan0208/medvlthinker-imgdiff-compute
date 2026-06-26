#!/bin/bash
# ACC-generalization campaign on MedGemma 4B/27B (Gemma3 medical pair; fixed-resolution vision encoder,
# so NO resolution sweep -- tests ACC's core size x mode cascade). Uses run_peer_eval (model-agnostic
# llm.chat) since Gemma3 != Qwen2.5-VL. Offline from cache. Competent-4 + MMMU + MedXpert (= ALL-6 keys).
# Sequential (27B on TP=2 to avoid OOM); 4B first on GPU0. Launch from repo root.
set -u
cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1
M4=google/medgemma-4b-it; M27=google/medgemma-27b-it
DS="PMC-VQA SLAKE VQA-RAD PathVQA MMMU MedXpert-Reasoning MedXpert-Understanding"

echo "=== MedGemma-4B no-think (small leg) ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py --model_path $M4 --tag medgemma4b_nt \
  --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/medgemma4b/nt --tp 1 --gpu_mem 0.85 --max_side 896 --max_images 4 > logs/medgemma_4b_nt.log 2>&1
echo "=== MedGemma-27B no-think (workhorse) TP=2 ==="
python3 src/labeling/run_peer_eval.py --model_path $M27 --tag medgemma27b_nt \
  --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/medgemma27b/nt --tp 2 --gpu_mem 0.90 --max_side 896 --max_images 4 > logs/medgemma_27b_nt.log 2>&1
echo "=== MedGemma-27B think (reasoning tier) TP=2 ==="
python3 src/labeling/run_peer_eval.py --model_path $M27 --tag medgemma27b_think --think \
  --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/medgemma27b/think --tp 2 --gpu_mem 0.90 --max_side 896 --max_images 4 --max_tokens 1024 > logs/medgemma_27b_think.log 2>&1
echo "=== ALL_DONE MedGemma ACC campaign ==="
