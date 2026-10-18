#!/usr/bin/env python3
"""make_c320feat_queue.py -- the cap320-feature arm (2026-10-18). Every hidden-state cache the final method
reads was extracted at max_pixels 1,003,520 while the pools were generated at cap320 (250,880): AUDIT_2026-09-18
section 4. This re-extracts the train cache and every N=8 pool the Tmap policy scores (T=0.7 pools of all 12 cells
+ the frozen-temperature pools) at cap320, layers 18/20/22, into feats_hidden/generator_{train,eval}_C320*.npz,
read by final_method_eval.py --feat C320. Balanced on the measured full-resolution extraction minutes.
"""
import json

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
WT = ROOT + "/.claude/worktrees/exp-2026-10-15"
L7 = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
      "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
ENV = "export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1"
EXT = (f"python3 {WT}/src/training_methods/extract_generator_hidden.py --model_path {L7} --mode generator "
       f"--layers 18 20 22 --out feats_hidden --cap cap320")
# (cell, temperature tag or None for the T=0.7 pool)
SPLIT = {0: [("gemex_open", None), ("gemex_open", "T10"), ("omnimed_open", "T10"), ("slake_open", None),
             ("vqa_rad_open", None), ("vqa_rad_open", "T02"), ("radimagenet_open", None),
             ("radimagenet_open", "T10"), ("pathvqa_closed", None), ("slake_closed", None),
             ("vqa_rad_closed", None), ("vqa_rad_closed", "T02")],
         1: [("__train__", None), ("kvasir_x1_open", None), ("omnimed_open", None), ("vqamed_open", None),
             ("vqamed_open", "T04"), ("pathvqa_open", None), ("vqamed2019_test", None),
             ("vqamed2019_test", "T04")]}

for g, items in SPLIT.items():
    q = []
    for c, t in items:
        if c == "__train__":
            inner, stem, name = f"{EXT} --split train --stem_tag C320", "generator_train_C320", "train"
        else:
            tagarg = f" --gen_tag lingshu7b{t}" if t else ""
            st = f"C320_{t}_{c}" if t else f"C320_{c}"
            inner, stem, name = f"{EXT} --split eval --eval_ds {c}{tagarg} --stem_tag {st}", f"generator_eval_{st}", st
        q.append({"name": f"c320f_{name}",
                  "cmd": f"flock -w 172800 logs/.gpu{g}.lock bash -c '{ENV}; CUDA_VISIBLE_DEVICES={g} {inner}'",
                  "log": f"logs/sv_c320f_{name}.log", "timeout_s": 8 * 3600, "stall_s": 3600,
                  "expect": f"{ROOT}/feats_hidden/{stem}.npz"})
    p = f"{WT}/runners/c320feat_1018_gpu{g}.json"
    json.dump(q, open(p, "w"), indent=1)
    print(p, [j["name"] for j in q])
