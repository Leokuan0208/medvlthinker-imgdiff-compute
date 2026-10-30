#!/usr/bin/env python3
"""make_pool_1022.py -- writes runners/pool_1022.json: every GPU job owed for options 2+3 (2026-10-22), re-armed
after the `timeout ... on G` bug (a bash function cannot be exec'd by timeout -> rc 127) broke chains 1 and 2.
Order = priority; "deps" = jobs that must have FINISHED (produced or failed) first. Run by runners/pool_worker.py.
"""
import json, os
M = "/home/jamesyang/medvlthinker-imgdiff-compute"; WT = f"{M}/.claude/worktrees/exp-2026-10-15"
X = "/data/dan/exp_2026_10/xjudge_1022"; D = f"{M}/data/sft_crossfit_2026-10-22"
MEDG = sorted(os.listdir("/data/dan/hf_cache/hub/models--google--medgemma-27b-it/snapshots"))[0]
MEDG = f"/data/dan/hf_cache/hub/models--google--medgemma-27b-it/snapshots/{MEDG}/"
OPEN8 = "pathvqa_open slake_open vqa_rad_open radimagenet_open kvasir_x1_open omnimed_open vqamed_open gemex_open"
CK = f"{M}/ckpts/openvqa/cheap_lingshu7b"; WO = "/data/dan/exp_2026_10/wed_published"
WED_SHARDS = int(os.environ.get("WED_SHARDS", "4"))
medg = lambda preds: (f"python3 -u {M}/src/labeling/run_judge.py --judge_model {MEDG} --tp 1 --gpu_mem 0.90 --preds {preds}")
J = []
add = lambda name, cmd, timeout, deps=(), gpus=1: J.append({"name": name, "cmd": cmd, "timeout": timeout, "deps": list(deps), "gpus": gpus})

# option 2: label-free temperature -- predicted modality, MedGemma on the temperature policies' picks
add("modality", f"python3 -u {WT}/src/exp_2026_10/predict_modality.py", "3h")
for p in (0, 1):
    add(f"xj_lf{p}", medg(f"{X}/lf_part{p}.jsonl"), "8h")
# option 3: USC on both pools; MedGemma on the 32B greedy answers
for pol in ("T07", "Tmap"):
    add(f"usc_{pol}", f"python3 -u {WT}/src/exp_2026_10/usc_select.py --policy {pol}", "6h")
for p in (0, 1):
    add(f"xj_g32_{p}", medg(f"{X}/g32_part{p}.jsonl"), "8h")
# LoRA verifier retrained on the probe's pooled data, cross-fitted
for f in (0, 1):
    add(f"loraver_f{f}", f"python3 -u {WT}/src/exp_2026_10/lora_verifier_crossfit.py --fold {f}", "16h")
# generator SFT on gold answers, cross-fitted: train (clean stop at 10 h) -> merge (on /data) -> greedy per cell
for f in (0, 1):
    mg = f"/data/dan/exp_2026_10/sft_crossfit_1022_f{f}_merged"
    gen = " ; ".join(f"python3 -u {M}/src/labeling/run_openvqa.py --model_path {mg} --ckpt_dir {M}/ckpts/openvqa/sft_crossfit_1022 "
                     f"--tp 1 --max_model_len 4096 --dataset {c} --tag sft1022f{f} --n_samples 1 --temp 0.0 --cap cap320 "
                     f"--idx_file {D}/heldout_{c}_fold{f}.json" for c in OPEN8.split())
    add(f"sft_f{f}", f"python3 -u {WT}/src/training_methods/train_cheap_leg.py --manifest {D}/fold{f}.json "
        f"--quota '{{\"open_generic\": 1000000}}' --out_dir ckpts/train/sft_crossfit_1022_f{f} --seed 0 --deadline_s 36000 && "
        f"python3 -u {WT}/src/training_methods/merge_cheapleg_lora.py --adapter {M}/ckpts/train/sft_crossfit_1022_f{f} --out {mg} && "
        f"{{ {gen} ; }}", "16h")
# WED as published (authors' own sampling) on the three cells its code supports
wed = lambda c, out, extra="": (f"mkdir -p {WO} && python3 -u {WT}/src/exp_2026_10/wed/wed_select.py --pool {CK}/ckpt_{c}_lingshu7b_sc8.jsonl "
                                f"--out {WO}/{out} --gpu --cap cap320 --own_sampling {extra}")
for s in range(WED_SHARDS):
    add(f"wed_pathvqa_s{s}", wed("pathvqa_open", f"wed_pub_pathvqa_open_s{s}.jsonl", f"--shard {s} --nshard {WED_SHARDS}"), "16h")
add("wed_slake", wed("slake_open", "wed_pub_slake_open.jsonl"), "16h")
add("wed_vqa_rad", wed("vqa_rad_open", "wed_pub_vqa_rad_open.jsonl"), "8h")
# MedGemma on USC / LoRA-verifier picks not yet labelled (built after the lf + g32 labels are merged)
add("build_picks", f"python3 {WT}/src/exp_2026_10/build_xjudge_inputs.py merge && python3 {WT}/src/exp_2026_10/build_xjudge_inputs.py picks",
    "10m", deps=["xj_lf0", "xj_lf1", "xj_g32_0", "xj_g32_1", "usc_T07", "usc_Tmap", "loraver_f0", "loraver_f1"])
for p in (0, 1):
    add(f"xj_picks{p}", medg(f"{X}/picks_part{p}.jsonl"), "8h", deps=["build_picks"])
# both judges on every NEW answer (SFT greedy, WED-published picks): judge of record needs both cards
gen_jobs = ["sft_f0", "sft_f1", "wed_slake", "wed_vqa_rad"] + [f"wed_pathvqa_s{s}" for s in range(WED_SHARDS)]
add("judge_new", f"python3 {WT}/src/exp_2026_10/build_xjudge_inputs.py new && python3 -u {M}/src/labeling/run_judge.py --tp 2 "
    f"--preds {X}/new_all.jsonl", "8h", deps=gen_jobs, gpus=2)
for p in (0, 1):
    add(f"xj_new{p}", medg(f"{X}/new_part{p}.jsonl"), "8h", deps=["judge_new"])
ORDER = ["modality", "xj_lf0", "xj_lf1", "loraver_f0", "loraver_f1", "sft_f0", "sft_f1"]   # option 2 first, then longest first
J = sorted(J, key=lambda j: ORDER.index(j["name"]) if j["name"] in ORDER else len(ORDER))   # stable: the rest keep list order
json.dump(J, open(f"{WT}/runners/pool_1022.json", "w"), indent=1)
print(len(J), "jobs ->", f"{WT}/runners/pool_1022.json")
