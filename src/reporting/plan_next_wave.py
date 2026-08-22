#!/usr/bin/env python3
"""plan_next_wave.py -- refill the campaign lanes from what is actually on disk.

WHY THIS EXISTS.  The first campaign was three fixed queues, and GPU 0 drained its 12 jobs and sat
idle at 16 MiB while GPU 1 still had hours of work.  A fixed queue cannot keep hardware busy,
because the set of runnable jobs GROWS as earlier jobs land: every new judge file makes an
extraction runnable, and every new feature cache makes a measurement runnable.

So this reads the tree and emits, per lane, the jobs whose INPUTS EXIST and whose OUTPUTS DO NOT.
It is idempotent -- running it twice with nothing finished in between emits the same queue -- and it
emits nothing when there is nothing to do, which is how the driver knows the campaign is over.

Emits runners/auto_<lane>_wave<N>.json and prints the job count per lane.

  python3 src/reporting/plan_next_wave.py --wave 2
"""
import argparse, glob, json, os, re

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
FEATS = os.path.join(ROOT, "feats_hidden")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
L7 = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
      "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
E = "export HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1; "
T = "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HEAD_SWEEP_THREADS=4; "
TAGS = {"lingshu7b": "", "lingshu7bT02": "_T02", "lingshu7bT04": "_T04", "lingshu7bT10": "_T10"}
TEMPV = {"lingshu7b": 0.7, "lingshu7bT02": 0.2, "lingshu7bT04": 0.4, "lingshu7bT10": 1.0}
# cells that are real reporting cells with a source json the generator can load
CELLS = ["vqa_rad_open", "slake_open", "pathvqa_open", "radimagenet_open",
         "vqamed_open", "gemex_open", "omnimed_open", "kvasir_x1_open"]
# rough question counts, used only to run the cheap cells first so results land early
SIZE = {"vqa_rad_open": 200, "slake_open": 645, "pathvqa_open": 1500, "radimagenet_open": 2000,
        "vqamed_open": 3663, "gemex_open": 8000, "omnimed_open": 8883, "kvasir_x1_open": 10121}


def nonempty(p, mb=0):
    return os.path.exists(p) and os.path.getsize(p) > mb * 1_000_000


def gpulock(g, cmd):
    """Serialise everything that touches GPU g behind a lock file.

    vLLM reserves a fixed FRACTION of the card at startup (0.88), so two jobs on one GPU do not
    share -- the second dies with "Free memory on device (56.29/79.14 GiB) ... is less than desired
    GPU memory utilization". That is exactly what killed slake_open_lingshu7bT04_gen twice when a
    leftover extraction was still holding 22.8 GB. A lock is more robust than tuning the fraction,
    because it also protects against a stray process from an earlier driver, and it costs nothing
    when the GPU is genuinely free. Commands contain no single quotes, so this quoting is safe.
    """
    return f"flock -w 172800 logs/.gpu{g}.lock bash -c '{cmd}'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, required=True)
    A = ap.parse_args()
    gpu = {0: [], 1: []}
    cpu = []

    # ---- 1. generation+judge for any (cell, temperature) not yet sampled -------------------
    want = [(c, t) for c in CELLS for t in ("lingshu7bT02", "lingshu7bT04", "lingshu7bT10")]
    todo_gen = [(c, t) for c, t in want if not nonempty(f"{CK}/ckpt_{c}_{t}_sc8.jsonl")]
    todo_gen.sort(key=lambda ct: SIZE.get(ct[0], 99999))
    for i, (c, t) in enumerate(todo_gen):
        g = i % 2
        gpu[g].append({"name": f"{c}_{t}_gen",
            "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 src/labeling/run_openvqa.py "
                   f"--model_path {L7} --tag {t}_sc8 --dataset {c} --n_samples 8 "
                   f"--temp {TEMPV[t]} --ckpt_dir {CK} --tp 1 --max_model_len 4096",
            "log": f"logs/sv_{c}_{t}_gen.log", "timeout_s": 36000, "stall_s": 2700, "_gpu": g,
            "expect": f"{CK}/ckpt_{c}_{t}_sc8.jsonl"})
    # judge for anything sampled but unjudged (including what the gens above will produce)
    for i, (c, t) in enumerate([(c, t) for c, t in want
                                if not nonempty(f"{CK}/ckpt_{c}_{t}_sc8_scexploded.judge.jsonl")]):
        g = i % 2
        gpu[g].append({"name": f"{c}_{t}_judge",
            "cmd": E + f"python3 src/cascade_methods/explode_sc_for_judge.py "
                   f"{CK}/ckpt_{c}_{t}_sc8.jsonl && CUDA_VISIBLE_DEVICES={g} python3 "
                   f"src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 --preds "
                   f"{CK}/ckpt_{c}_{t}_sc8_scexploded.jsonl",
            "log": f"logs/sv_{c}_{t}_judge.log", "timeout_s": 54000, "stall_s": 3600, "_gpu": g,
            "expect": f"{CK}/ckpt_{c}_{t}_sc8_scexploded.judge.jsonl"})

    # ---- 2. hidden-state extraction wherever a judged pool has no feature cache ------------
    todo_ex = []
    for c in CELLS:
        for tag, suf in TAGS.items():
            if not nonempty(f"{CK}/ckpt_{c}_{tag}_sc8_scexploded.judge.jsonl"):
                continue
            if nonempty(f"{FEATS}/generator_eval_{c}{suf}.npz", mb=10):
                continue
            todo_ex.append((c, tag, suf))
    todo_ex.sort(key=lambda x: SIZE.get(x[0], 99999))
    for i, (c, tag, suf) in enumerate(todo_ex):
        g = i % 2
        gt = f"--gen_tag {tag} " if tag != "lingshu7b" else ""
        gpu[g].append({"name": f"{c}{suf}_extract",
            "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 "
                   f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                   f"--mode generator --split eval --eval_ds {c} {gt}--stem_tag {c}{suf} "
                   f"--out feats_hidden",
            "log": f"logs/sv_{c}{suf}_extract.log", "timeout_s": 86400, "stall_s": 5400, "_gpu": g,
            "expect": f"{FEATS}/generator_eval_{c}{suf}.npz", "expect_min_bytes": 10_000_000})

    # ---- 2b. fine-layer (8-layer) extraction for the new cells ----------------------------
    # Layer 21 was chosen on in-domain CV, the metric blind to the transfer failure. Scoring layers
    # on transfer needs these caches on the cells the head actually fails.
    FL = ["kvasir_x1_open", "vqamed_open", "omnimed_open", "gemex_open"]
    todo_fl = [c for c in FL
               if nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc8_scexploded.judge.jsonl")
               and not nonempty(f"{FEATS}/generator_eval_finelayer_{c}.npz", mb=10)]
    todo_fl.sort(key=lambda c: SIZE.get(c, 99999))
    for i, c in enumerate(todo_fl):
        g = i % 2
        gpu[g].append({"name": f"finelayer_{c}_extract",
            "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 "
                   f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                   f"--mode generator --split eval --eval_ds {c} --layers 10 12 16 18 20 22 24 26 "
                   f"--stem_tag finelayer_{c} --out feats_hidden",
            "log": f"logs/sv_finelayer_{c}_extract.log", "timeout_s": 86400, "stall_s": 5400,
            "expect": f"{FEATS}/generator_eval_finelayer_{c}.npz",
            "expect_min_bytes": 10_000_000, "_gpu": g})

    # ---- 3. analyses that are cheap and should be refreshed whenever inputs changed --------
    ANALYSES = [
        ("temperature_report", "src/cascade_methods/temperature_report.py",
         f"{ART}/temperature_curve_2026-08-22.json"),
        ("head_temperature_sweep", "src/cascade_methods/head_temperature_sweep.py",
         f"{ART}/head_temperature_2026-08-22.json"),
    ]
    for name, script, out in ANALYSES:
        newer = [p for p in glob.glob(f"{CK}/*judge.jsonl") + glob.glob(f"{FEATS}/*.npz")
                 if not os.path.exists(out) or os.path.getmtime(p) > os.path.getmtime(out)]
        if newer:
            cpu.append({"name": f"{name}_w{A.wave}", "cmd": T + f"python3 -u {script}",
                        "log": f"logs/sv_{name}_w{A.wave}.log",
                        "timeout_s": 36000, "stall_s": 3600, "expect": out})

    # ---- 4. price curves for cells that have features but no curve ------------------------
    for c in CELLS:
        out = f"{ART}/head_newdomain_{c}_2026-08-21.json"
        if nonempty(f"{FEATS}/generator_eval_{c}.npz", mb=10) and not os.path.exists(out):
            cpu.append({"name": f"newdomain_{c}_w{A.wave}",
                "cmd": T + f"python3 -X faulthandler -u src/training_methods/head_newdomain_curve.py "
                       f"--cell {c} --threads 4 --seeds 3",
                "log": f"logs/sv_newdomain_{c}_w{A.wave}.log",
                "timeout_s": 86400, "stall_s": 7200, "expect": out})

    # ---- 5. the fine-layer sweep shards (CPU) ---------------------------------------------
    for i, ls in enumerate([[10, 12], [16, 18], [20, 22], [24, 26]], 1):
        out = f"{ART}/head_finelayer_shard{i}.json"
        if not os.path.exists(out):
            cpu.append({"name": f"finelayer_shard{i}_w{A.wave}",
                "cmd": T + f"python3 -X faulthandler -u src/training_methods/head_finelayer.py "
                       f"--threads 4 --seeds 3 --layers {' '.join(map(str, ls))} --out {out}",
                "log": f"logs/sv_finelayer_shard{i}_w{A.wave}.log",
                "timeout_s": 86400, "stall_s": 7200, "expect": out})

    # every GPU job goes behind its card's lock; drop the marker so the queue stays clean JSON
    for g in (0, 1):
        for j in gpu[g]:
            j["cmd"] = gpulock(j.pop("_gpu", g), j["cmd"])

    n = 0
    for lane, jobs in (("gpu0", gpu[0]), ("gpu1", gpu[1]), ("cpu", cpu)):
        p = os.path.join(ROOT, f"runners/auto_{lane}_wave{A.wave}.json")
        if jobs:
            json.dump(jobs, open(p, "w"), indent=1)
        elif os.path.exists(p):
            os.remove(p)
        print(f"  {lane}: {len(jobs)} jobs")
        for j in jobs:
            print(f"     {j['name']}")
        n += len(jobs)
    print(f"TOTAL {n}")
    return 0 if n else 3


if __name__ == "__main__":
    raise SystemExit(main())
