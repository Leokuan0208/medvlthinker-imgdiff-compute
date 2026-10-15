#!/usr/bin/env python3
"""make_fullres_queues.py -- write the supervisor queues for backlog item 1.0 (DECIDED 2026-09-17):
regenerate the Lingshu-7B open-text pools at full resolution (max_pixels 1,003,520 = the resolution
extract_generator_hidden.py already extracts at), and judge them.

Per eval cell (7 of 8; radimagenet_open is excluded because none of its images exceeds even the
cap320 ceiling -- image_geometry_all8_2026-10-15.json -- so its existing pool is already full-res):
  * <cell>_FR_sc8   : N=8, T=0.7, --cap fullres            tag lingshu7bFR_sc8
  * <cell>_FR_greedy: greedy, --cap fullres                tag lingshu7bFR
  * <cell>_C320     : greedy, --cap cap320, SAME SESSION   tag lingshu7bC320
    (the matched control: regenerating under a new serving config moves cells by ~+-0.008, so
     fullres must be compared against a cap320 arm generated alongside it, not the stored one)
  * judge: explode the sc8 pool, then MedVLThinker-32B (run_judge.py default) on the exploded pool
    and both greedy files, in one model load.
Per train pool (the four-domain train data the pooled probe also learns from): N=8 fullres + judge.

Generation runs the worktree copy of run_openvqa.py (which also stores every sample's
log-probability); judging and exploding run main's unchanged copies. Every job runs from the repo
root, because ckpt/feature paths are resolved against the launch directory (CLAUDE.md §7).
"""
import json, os

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
WT = ROOT + "/.claude/worktrees/exp-2026-10-15"
CK = ROOT + "/ckpts/openvqa/cheap_lingshu7b"
L7 = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
      "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
ENV = "export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1"
GEN = f"python3 {WT}/src/labeling/run_openvqa.py --model_path {L7} --ckpt_dir {CK} --tp 1 --max_model_len 4096"

SPLIT = {  # balanced by measured cap320 runtimes x the fullres token ratio
    0: {"eval": ["gemex_open", "omnimed_open", "slake_open", "vqa_rad_open"],
        "train": ["slake_open_train", "vqa_rad_open_train"]},
    1: {"eval": ["kvasir_x1_open", "vqamed_open", "pathvqa_open"],
        "train": ["pathvqa_open_train", "kvasir_open"]},
}
IDX_FILE = {d: f"data/disjoint_split/idx_{d}.json"
            for d in ("slake_open_train", "vqa_rad_open_train", "pathvqa_open_train")}


def job(g, name, inner, expect, timeout_s=6 * 3600, stall_s=3600):
    return {"name": f"fr1015_{name}",
            "cmd": f"flock -w 172800 logs/.gpu{g}.lock bash -c '{ENV}; {inner}'",
            "log": f"logs/sv_fr1015_{name}.log", "timeout_s": timeout_s, "stall_s": stall_s,
            "expect": expect}


def queue(g):
    q, cv = [], f"CUDA_VISIBLE_DEVICES={g}"
    for ds in SPLIT[g]["eval"] + SPLIT[g]["train"]:
        is_eval = ds in SPLIT[g]["eval"]
        idx = f" --idx_file {IDX_FILE[ds]}" if ds in IDX_FILE else ""
        sc8 = f"{CK}/ckpt_{ds}_lingshu7bFR_sc8.jsonl"
        q.append(job(g, f"{ds}_FR_sc8",
                     f"{cv} {GEN} --dataset {ds} --tag lingshu7bFR_sc8 --n_samples 8 --temp 0.7 --cap fullres{idx}",
                     sc8))
        judge_targets = [sc8.replace(".jsonl", "_scexploded.jsonl")]
        if is_eval:
            for tag, cap in (("lingshu7bFR", "fullres"), ("lingshu7bC320", "cap320")):
                q.append(job(g, f"{ds}_{tag}_greedy",
                             f"{cv} {GEN} --dataset {ds} --tag {tag} --n_samples 1 --temp 0.0 --cap {cap}",
                             f"{CK}/ckpt_{ds}_{tag}.jsonl"))
                judge_targets.append(f"{CK}/ckpt_{ds}_{tag}.jsonl")
        q.append(job(g, f"{ds}_judge",
                     f"python3 src/cascade_methods/explode_sc_for_judge.py {sc8} && "
                     f"{cv} python3 src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 --preds " + " ".join(judge_targets),
                     judge_targets[0].replace(".jsonl", ".judge.jsonl"), timeout_s=8 * 3600))
    return q


for g in (0, 1):
    p = f"{WT}/runners/fr1015_gpu{g}.json"
    json.dump(queue(g), open(p, "w"), indent=1)
    print(p, len(queue(g)), "jobs")
