#!/usr/bin/env python3
"""make_tmap16_queue.py -- N=16 pools at the frozen per-benchmark temperature (cap320), for the benchmarks
whose frozen T is not 0.7 (the T=0.7 N=16 pools and features already exist for all eight). With them the
final method can be scored at N=16 as well as N=8 -- the 8->16 step was +0.0167 at T=0.7
(coverage_sc16_ci_ALL_2026-09-16.json) but was never combined with the temperature map.
Pools: ckpt_<cell>_lingshu7bTxx_sc16.jsonl; features: generator_eval_ens_Txx_sc16_<cell>.npz (layers 18/20/22).
"""
import json

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
WT = ROOT + "/.claude/worktrees/exp-2026-10-15"
CK = ROOT + "/ckpts/openvqa/cheap_lingshu7b"
L7 = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
      "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
ENV = "export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1"
GEN = f"python3 {WT}/src/labeling/run_openvqa.py --model_path {L7} --ckpt_dir {CK} --tp 1 --max_model_len 4096"
EXT = (f"python3 {WT}/src/training_methods/extract_generator_hidden.py --model_path {L7} --mode generator "
       f"--layers 18 20 22 --out feats_hidden")
T = {"gemex_open": 1.0, "omnimed_open": 1.0, "radimagenet_open": 1.0, "vqa_rad_open": 0.2, "vqamed_open": 0.4}
SPLIT = {0: ["gemex_open", "vqa_rad_open"], 1: ["omnimed_open", "vqamed_open", "radimagenet_open"]}

for g, cells in SPLIT.items():
    q = []
    for c in cells:
        tag = f"lingshu7bT{int(round(T[c] * 10)):02d}"
        pool = f"{CK}/ckpt_{c}_{tag}_sc16.jsonl"
        stem = f"ens_{tag.replace('lingshu7b', '')}_sc16_{c}"
        for name, inner, expect in [
                (f"{c}_gen", f"CUDA_VISIBLE_DEVICES={g} {GEN} --dataset {c} --tag {tag}_sc16 --n_samples 16 "
                             f"--temp {T[c]} --cap cap320", pool),
                (f"{c}_judge", f"python3 src/cascade_methods/explode_sc_for_judge.py {pool} && CUDA_VISIBLE_DEVICES={g} "
                               f"python3 src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 --preds "
                               f"{pool.replace('.jsonl', '_scexploded.jsonl')}",
                 pool.replace(".jsonl", "_scexploded.judge.jsonl")),
                (f"{c}_extract", f"CUDA_VISIBLE_DEVICES={g} {EXT} --split eval --eval_ds {c} --gen_tag {tag} "
                                 f"--pool_tag _sc16 --stem_tag {stem}", f"{ROOT}/feats_hidden/generator_eval_{stem}.npz")]:
            q.append({"name": f"t16_1018_{name}", "cmd": f"flock -w 172800 logs/.gpu{g}.lock bash -c '{ENV}; {inner}'",
                      "log": f"logs/sv_t16_1018_{name}.log", "timeout_s": 10 * 3600, "stall_s": 3600, "expect": expect})
    p = f"{WT}/runners/t16_1018_gpu{g}.json"
    json.dump(q, open(p, "w"), indent=1)
    print(p, [j["name"] for j in q])
