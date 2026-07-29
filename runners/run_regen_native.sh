#!/bin/bash
# Regenerate ALL results with the FIXED (native) think prompts: complete MedGemma native-think for ALL-6,
# then rerun every analysis + charts + tables + record off the native think tier. Launch from repo root.
set -u; cd ~/medvlthinker-imgdiff-compute; export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1
DS="PMC-VQA SLAKE VQA-RAD PathVQA MMMU MedXpert-Reasoning MedXpert-Understanding"
MEDGEMMA_INSTR="Reason step by step, then state your final answer as 'Final Answer: X' where X is the correct option letter."
echo "=== 1) complete MedGemma-27B native-think ALL-6 (resume; larger ctx for long MMMU/MedXpert prompts) $(date +%H:%M) ==="
python3 src/labeling/run_peer_eval.py --model_path google/medgemma-27b-it --tag medgemma27b_think --think \
  --system "You are a helpful medical assistant." --think_instr "$MEDGEMMA_INSTR" --max_tokens 1024 \
  --datasets $DS --n 4000 --ckpt_dir ckpts/acc_gen/medgemma27b/think_native --tp 2 --gpu_mem 0.92 \
  --max_side 768 --max_images 2 --max_model_len 16384 > logs/nt_medgemma_complete.log 2>&1
echo "  medgemma native rows now: $(cat ckpts/acc_gen/medgemma27b/think_native/*.jsonl 2>/dev/null | wc -l)"
echo "=== 2) native-think comparison $(date +%H:%M) ==="
python3 src/cascade_methods/compare_native_think.py > results/cascade_methods/artifacts/native_think_compare.txt 2>&1
echo "=== 3) acc_allmethods (all 5, NATIVE think) ==="
{ for fam in medvlthinker lingshu qoq chiron medgemma; do python3 src/cascade_methods/acc_allmethods.py --family $fam 2>/dev/null; done; } > results/cascade_methods/artifacts/acc_allmethods_all.txt 2>&1
echo "=== 4) acc_2size (all 5, NATIVE think) ==="
{ for fam in medvlthinker lingshu qoq chiron medgemma; do python3 src/cascade_methods/acc_2size.py --family $fam 2>/dev/null; done; } > results/cascade_methods/artifacts/acc_2size_all.txt 2>&1
echo "=== 5) regenerate master charts + tables + full record ==="
python3 src/cascade_methods/make_master_charts.py > logs/regen_charts.log 2>&1
python3 src/cascade_methods/make_full_record.py >> logs/regen_charts.log 2>&1
echo "=== REGEN_NATIVE_DONE $(date +%H:%M) ==="
for f in master_data.csv native_think_compare.txt; do echo "  $f: $(wc -l < results/cascade_methods/artifacts/$f 2>/dev/null) lines"; done
for f in MASTER_TABLES.md FULL_RECORD.md; do echo "  $f: $(wc -l < results/cascade_methods/docs/archive_mcq/$f 2>/dev/null) lines"; done
ls paper/figs/master/*.png 2>/dev/null | wc -l | xargs echo "  master charts:"
