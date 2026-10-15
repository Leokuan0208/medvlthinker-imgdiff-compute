#!/usr/bin/env python3
"""make_fullres_stage2.py -- stage 2 of the full-resolution regeneration: hidden-state extraction from
the judged lingshu7bFR pools, layers 18/20/22 (the shipped ensemble), written to
feats_hidden/generator_{eval,train}_FR*.npz. Balanced on the measured cap320-era extraction minutes
(gemex 165, kvasir_x1 126, omnimed 89, train 72, vqamed 60, pathvqa 39, radimagenet 15, slake+vqa_rad ~10).
The train cache uses the historical default draw (max_train 10364 + full groups), as the shipped
probe's did, so the full-resolution refit differs from the shipped one in resolution only.
"""
import json

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
WT = ROOT + "/.claude/worktrees/exp-2026-10-15"
L7 = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
      "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
ENV = "export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1"
EXT = (f"python3 src/training_methods/extract_generator_hidden.py --model_path {L7} --mode generator "
       f"--layers 18 20 22 --gen_tag lingshu7bFR --out feats_hidden")
SPLIT = {0: ["gemex_open", "omnimed_open", "slake_open", "vqa_rad_open", "radimagenet_open"],
         1: ["kvasir_x1_open", "vqamed_open", "pathvqa_open", "__train__"]}

for g, cells in SPLIT.items():
    q = []
    for c in cells:
        if c == "__train__":
            inner, stem, name = f"CUDA_VISIBLE_DEVICES={g} {EXT} --split train --stem_tag FR", "generator_train_FR", "train"
        else:
            inner = f"CUDA_VISIBLE_DEVICES={g} {EXT} --split eval --eval_ds {c} --stem_tag FR_{c}"
            stem, name = f"generator_eval_FR_{c}", c
        q.append({"name": f"fr1015_extract_{name}",
                  "cmd": f"flock -w 172800 logs/.gpu{g}.lock bash -c '{ENV}; {inner}'",
                  "log": f"logs/sv_fr1015_extract_{name}.log", "timeout_s": 8 * 3600, "stall_s": 3600,
                  "expect": f"{ROOT}/feats_hidden/{stem}.npz"})
    p = f"{WT}/runners/fr1015_stage2_gpu{g}.json"
    json.dump(q, open(p, "w"), indent=1)
    print(p, [j["name"] for j in q])
