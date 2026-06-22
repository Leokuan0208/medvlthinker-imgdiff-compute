#!/bin/bash
# QoQ-Med-VL 7B/32B (Qwen2.5-VL medical) — full ACC + all-methods pipeline (run_vlm_eval, like Lingshu).
set -u; cd ~/medvlthinker-imgdiff-compute; export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1
M7=ddvd233/QoQ-Med-VL-7B; M32=ddvd233/QoQ-Med-VL-32B
DS="PMC-VQA SLAKE VQA-RAD PathVQA MMMU MedXpert-Reasoning MedXpert-Understanding"
TS="You are an expert medical AI. Reason step by step about the image and the question, then end your response with a line 'Answer: X' where X is the correct option letter."
echo "=== QoQ 7B no-think resolution sweep (both GPUs) $(date +%H:%M) ==="
( for cap in cap80 cap320 fullres; do CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_vlm_eval.py --model_path $M7 --tag qoq7b --arm nothink --cap $cap --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/qoq7b/$cap --tp 1 --gpu_mem 0.85; done ) > logs/qoq_7b_a.log 2>&1 &
( for cap in cap160 cap640; do CUDA_VISIBLE_DEVICES=1 python3 src/labeling/run_vlm_eval.py --model_path $M7 --tag qoq7b --arm nothink --cap $cap --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/qoq7b/$cap --tp 1 --gpu_mem 0.85; done ) > logs/qoq_7b_b.log 2>&1 &
wait
echo "=== QoQ 7B self-verify @cap320 (AutoMix) ==="
CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_7b_selfverify_vllm.py --model_path $M7 --source eval --cap cap320 --datasets $DS --n 4000 --pred_dir_eval ckpts/acc_gen/qoq7b/cap320 --ckpt_dir ckpts/acc_gen/qoq7b/verify --tp 1 --gpu_mem 0.85 > logs/qoq_verify.log 2>&1
echo "=== QoQ 32B no-think cap320 + fullres + think (TP=2) ==="
python3 src/labeling/run_vlm_eval.py --model_path $M32 --tag qoq32b --arm nothink --cap cap320 --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/qoq32b/nothink_cap320 --tp 2 --gpu_mem 0.90 > logs/qoq_32b_ntcap320.log 2>&1
python3 src/labeling/run_vlm_eval.py --model_path $M32 --tag qoq32b --arm nothink --cap fullres --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/qoq32b/nothink_fullres --tp 2 --gpu_mem 0.90 > logs/qoq_32b_ntfull.log 2>&1
python3 src/labeling/run_vlm_eval.py --model_path $M32 --tag qoq32b --arm think --system "$TS" --cap fullres --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/qoq32b/think_fullres --tp 2 --gpu_mem 0.90 --max_tokens 1536 > logs/qoq_32b_think.log 2>&1
echo "=== QOQ_DONE $(date +%H:%M) ==="
