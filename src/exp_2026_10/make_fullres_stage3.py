#!/usr/bin/env python3
"""make_fullres_stage3.py -- out-of-sample test of PER-BENCHMARK SAMPLING TEMPERATURE.

The temperatures are FROZEN here from temp_crossfit_2026-10-15.json (cross-fitted on the cap320 pools),
before any full-resolution pool at these temperatures exists:
  rule: keep a benchmark's T where both folds agree; where they split between 0.7 and another T keep
  0.7; where both folds leave 0.7 but disagree, take the T with the larger in-sample gain in
  head_temp_ensemble_2026-08-30.json.
  gemex_open 1.0 · omnimed_open 1.0 · radimagenet_open 1.0 (existing T=1.0 pool hard-linked: images
  below every cap) · vqa_rad_open 0.2 · vqamed_open 0.4 (folds chose 0.4 / 0.2; 0.4 had +0.0149 vs
  +0.0138) · pathvqa_open, kvasir_x1_open, slake_open 0.7 (slake folds split 1.0 / 0.7).
The fresh full-resolution pools at these temperatures are new samples the choice never saw.
"""
import json

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
WT = ROOT + "/.claude/worktrees/exp-2026-10-15"
CK = ROOT + "/ckpts/openvqa/cheap_lingshu7b"
L7 = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
      "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
ENV = "export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1"
GEN = f"python3 {WT}/src/labeling/run_openvqa.py --model_path {L7} --ckpt_dir {CK} --tp 1 --max_model_len 4096"
EXT = (f"python3 src/training_methods/extract_generator_hidden.py --model_path {L7} --mode generator "
       f"--layers 18 20 22 --out feats_hidden")
FROZEN_T = {"gemex_open": 1.0, "omnimed_open": 1.0, "vqa_rad_open": 0.2, "vqamed_open": 0.4}
SPLIT = {0: ["gemex_open"], 1: ["omnimed_open", "vqamed_open", "vqa_rad_open"]}

for g, cells in SPLIT.items():
    q = []
    for c in cells:
        T = FROZEN_T[c]
        base = f"lingshu7bFRT{int(round(T * 10)):02d}"
        sc8 = f"{CK}/ckpt_{c}_{base}_sc8.jsonl"
        steps = [
            (f"{c}_{base}_gen", f"CUDA_VISIBLE_DEVICES={g} {GEN} --dataset {c} --tag {base}_sc8 --n_samples 8 "
                                f"--temp {T} --cap fullres", sc8),
            (f"{c}_{base}_judge", f"python3 src/cascade_methods/explode_sc_for_judge.py {sc8} && "
                                  f"CUDA_VISIBLE_DEVICES={g} python3 src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 "
                                  f"--preds {sc8.replace('.jsonl', '_scexploded.jsonl')}",
             sc8.replace(".jsonl", "_scexploded.judge.jsonl")),
            (f"{c}_{base}_extract", f"CUDA_VISIBLE_DEVICES={g} {EXT} --split eval --eval_ds {c} --gen_tag {base} "
                                    f"--stem_tag {base.replace('lingshu7b', '')}_{c}",
             f"{ROOT}/feats_hidden/generator_eval_{base.replace('lingshu7b', '')}_{c}.npz")]
        for name, inner, expect in steps:
            q.append({"name": f"fr1015_s3_{name}", "cmd": f"flock -w 172800 logs/.gpu{g}.lock bash -c '{ENV}; {inner}'",
                      "log": f"logs/sv_fr1015_s3_{name}.log", "timeout_s": 8 * 3600, "stall_s": 3600, "expect": expect})
    p = f"{WT}/runners/fr1015_stage3_gpu{g}.json"
    json.dump(q, open(p, "w"), indent=1)
    print(p, [j["name"] for j in q])
json.dump({"frozen_T": {**FROZEN_T, "radimagenet_open": 1.0, "pathvqa_open": 0.7, "kvasir_x1_open": 0.7,
                        "slake_open": 0.7},
           "source": "temp_crossfit_2026-10-15.json; rule in make_fullres_stage3.py docstring",
           "frozen_at": "2026-10-15, before any full-resolution pool at these temperatures was generated"},
          open(f"{WT}/results/cascade_methods/artifacts/frozen_temperature_map_2026-10-15.json", "w"), indent=1)
