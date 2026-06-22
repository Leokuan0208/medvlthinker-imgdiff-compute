#!/bin/bash
set -u; cd ~/medvlthinker-imgdiff-compute
export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1
M32=lingshu-medical-mllm/Lingshu-32B
DS="PMC-VQA SLAKE VQA-RAD PathVQA MMMU MedXpert-Reasoning MedXpert-Understanding"
TS="You are an expert medical AI. Reason step by step about the image and the question, then end your response with a line 'Answer: X' where X is the correct option letter."
python3 src/labeling/run_vlm_eval.py --model_path $M32 --tag lingshu32b --arm nothink --cap cap320 --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/lingshu32b/nothink_cap320 --tp 2 --gpu_mem 0.90
python3 src/labeling/run_vlm_eval.py --model_path $M32 --tag lingshu32b --arm nothink --cap fullres --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/lingshu32b/nothink_fullres --tp 2 --gpu_mem 0.90
python3 src/labeling/run_vlm_eval.py --model_path $M32 --tag lingshu32b --arm think --system "$TS" --cap fullres --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/lingshu32b/think_fullres --tp 2 --gpu_mem 0.90 --max_tokens 1536
echo "LINGSHU_32B_PHASE2_DONE"
