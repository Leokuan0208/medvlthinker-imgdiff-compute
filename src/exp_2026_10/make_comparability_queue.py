#!/usr/bin/env python3
"""make_comparability_queue.py -- stage 4: pools, judge labels and features for the comparability cells
(build_comparability_cells.py), at cap320 (see REVISED below), so the final method can be scored on the FULL
official test splits (closed + open) and the official VQA-Med 2019 test.

REVISED 2026-10-16 to cap320 (before any stage-4 job ran): the full-resolution pools did not improve the
method (select_policy_cap320_vs_fullres_T07.json), so the comparability cells are generated at cap320 like the
final method's pools, under the project's cap320 tag/stem names (greedy lingshu7b, pool lingshu7b_sc8,
lingshu7bT02/T04_sc8; features generator_eval_finelayer_<c> / generator_eval_ens_T0x_<c>).
Pre-specified 2026-10-15, before any of these pools exists: every cell gets an N=8 T=0.7 pool and a
greedy pool; vqa_rad_closed and vqamed2019_test ALSO get an N=8 pool at their family's frozen
temperature (frozen_temperature_map_2026-10-15.json: vqa_rad 0.2, vqamed 0.4), so whichever
temperature policy the main experiment adopts can be applied here without regenerating.
Generation and extraction run the WORKTREE copies (they know these cells); judging runs main's.
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
EXTRA_T = {"vqa_rad_closed": 0.2, "vqamed2019_test": 0.4}
SPLIT = {0: ["pathvqa_closed"], 1: ["vqamed2019_test", "slake_closed", "vqa_rad_closed"]}


def job(g, name, inner, expect, t=6 * 3600):
    return {"name": f"fr1015_s4c_{name}", "cmd": f"flock -w 172800 logs/.gpu{g}.lock bash -c '{ENV}; {inner}'",
            "log": f"logs/sv_fr1015_s4c_{name}.log", "timeout_s": t, "stall_s": 3600, "expect": expect}


for g, cells in SPLIT.items():
    q = []
    for c in cells:
        cv = f"CUDA_VISIBLE_DEVICES={g}"
        pools = [("lingshu7b", 0.7)] + ([(f"lingshu7bT{int(round(EXTRA_T[c] * 10)):02d}", EXTRA_T[c])]
                                          if c in EXTRA_T else [])
        greedy = f"{CK}/ckpt_{c}_lingshu7b.jsonl"
        q.append(job(g, f"{c}_greedy", f"{cv} {GEN} --dataset {c} --tag lingshu7b --n_samples 1 --temp 0.0 "
                                       f"--cap cap320", greedy))
        for base, T in pools:
            sc8 = f"{CK}/ckpt_{c}_{base}_sc8.jsonl"
            q.append(job(g, f"{c}_{base}_gen", f"{cv} {GEN} --dataset {c} --tag {base}_sc8 --n_samples 8 "
                                               f"--temp {T} --cap cap320", sc8))
            targets = [sc8.replace(".jsonl", "_scexploded.jsonl")] + ([greedy] if base == "lingshu7b" else [])
            q.append(job(g, f"{c}_{base}_judge",
                         f"python3 src/cascade_methods/explode_sc_for_judge.py {sc8} && {cv} python3 "
                         f"src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 --preds " + " ".join(targets),
                         targets[0].replace(".jsonl", ".judge.jsonl")))
            stem = f"finelayer_{c}" if base == "lingshu7b" else f"ens_{base.replace('lingshu7b', '')}_{c}"
            q.append(job(g, f"{c}_{base}_extract",
                         f"{cv} {EXT} --split eval --eval_ds {c} --gen_tag {base} --stem_tag {stem}",
                         f"{ROOT}/feats_hidden/generator_eval_{stem}.npz", t=8 * 3600))
    p = f"{WT}/runners/fr1015_stage4c_gpu{g}.json"
    json.dump(q, open(p, "w"), indent=1)
    print(p, len(q), "jobs")
