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

    # ---- 4b. fine-layer extraction for the one cell still missing it -----------------------
    for c in ["radimagenet_open"]:
        if (nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc8_scexploded.judge.jsonl")
                and not nonempty(f"{FEATS}/generator_eval_finelayer_{c}.npz", mb=10)):
            gpu[0].append({"name": f"finelayer_{c}_extract",
                "cmd": E + f"CUDA_VISIBLE_DEVICES=0 python3 "
                       f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                       f"--mode generator --split eval --eval_ds {c} "
                       f"--layers 10 12 16 18 20 22 24 26 --stem_tag finelayer_{c} "
                       f"--out feats_hidden",
                "log": f"logs/sv_finelayer_{c}_extract.log", "timeout_s": 86400, "stall_s": 5400,
                "expect": f"{FEATS}/generator_eval_finelayer_{c}.npz",
                "expect_min_bytes": 10_000_000, "_gpu": 0})

    # ---- 4c. COLD POOLS FOR THE TRAINING SPLIT ---------------------------------------------
    # The head is trained on T=0.7 pools and then deployed over pools of whatever temperature the
    # generator ran at. head_temperature_2026-08-22.json shows the best deployment temperature is
    # 0.2 on three cells and 0.4 on one, so on those cells the head is scoring a distribution it
    # was never fitted to. These jobs build T=0.2 training pools so a temperature-MATCHED head can
    # be fitted and the mismatch measured rather than assumed.
    TRAIN_DS = ["vqa_rad_open_train", "slake_open_train", "kvasir_open", "pathvqa_open_train"]
    for i, c in enumerate(TRAIN_DS):
        g = i % 2
        if not nonempty(f"{CK}/ckpt_{c}_lingshu7bT02_sc8.jsonl"):
            gpu[g].append({"name": f"{c}_T02_gen",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 src/labeling/run_openvqa.py "
                       f"--model_path {L7} --tag lingshu7bT02_sc8 --dataset {c} --n_samples 8 "
                       f"--temp 0.2 --ckpt_dir {CK} --tp 1 --max_model_len 4096",
                "log": f"logs/sv_{c}_T02_gen.log", "timeout_s": 36000, "stall_s": 2700,
                "expect": f"{CK}/ckpt_{c}_lingshu7bT02_sc8.jsonl", "_gpu": g})
        if not nonempty(f"{CK}/ckpt_{c}_lingshu7bT02_sc8_scexploded.judge.jsonl"):
            gpu[g].append({"name": f"{c}_T02_judge",
                "cmd": E + f"python3 src/cascade_methods/explode_sc_for_judge.py "
                       f"{CK}/ckpt_{c}_lingshu7bT02_sc8.jsonl && CUDA_VISIBLE_DEVICES={g} python3 "
                       f"src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 --preds "
                       f"{CK}/ckpt_{c}_lingshu7bT02_sc8_scexploded.jsonl",
                "log": f"logs/sv_{c}_T02_judge.log", "timeout_s": 54000, "stall_s": 3600,
                "expect": f"{CK}/ckpt_{c}_lingshu7bT02_sc8_scexploded.judge.jsonl", "_gpu": g})
    if all(nonempty(f"{CK}/ckpt_{c}_lingshu7bT02_sc8_scexploded.judge.jsonl") for c in TRAIN_DS) \
            and not nonempty(f"{FEATS}/generator_train_T02.npz", mb=10):
        gpu[1].append({"name": "train_T02_extract",
            "cmd": E + "CUDA_VISIBLE_DEVICES=1 python3 "
                   f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                   f"--mode generator --split train --gen_tag lingshu7bT02 --stem_tag T02 "
                   f"--out feats_hidden",
            "log": "logs/sv_train_T02_extract.log", "timeout_s": 86400, "stall_s": 5400,
            "expect": f"{FEATS}/generator_train_T02.npz", "expect_min_bytes": 10_000_000,
            "_gpu": 1})

    # ---- 4c2. ODD-LAYER extraction (19, 21) ------------------------------------------------
    # The fine-layer grid is even (10..26), so the DEPLOYED layer 21 could not be trained under the
    # same recipe as 18 and 20 -- head_layer_eval kept failing on "layer 21 not extracted" and burned
    # ~30 waves before the lookup was fixed. 19 comes along because the transfer peak sits at 18.
    ODD = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
           "kvasir_x1_open", "vqamed_open", "omnimed_open", "gemex_open"]
    if not nonempty(f"{FEATS}/generator_train_oddlayer.npz", mb=10):
        gpu[0].append({"name": "train_oddlayer_extract",
            "cmd": E + f"CUDA_VISIBLE_DEVICES=0 python3 "
                   f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                   f"--mode generator --split train --layers 19 21 --stem_tag oddlayer "
                   f"--out feats_hidden",
            "log": "logs/sv_train_oddlayer_extract.log", "timeout_s": 86400, "stall_s": 5400,
            "expect": f"{FEATS}/generator_train_oddlayer.npz",
            "expect_min_bytes": 10_000_000, "_gpu": 0})
    todo_odd = [c for c in ODD
                if nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc8_scexploded.judge.jsonl")
                and not nonempty(f"{FEATS}/generator_eval_oddlayer_{c}.npz", mb=5)]
    todo_odd.sort(key=lambda c: SIZE.get(c, 99999))
    for i, c in enumerate(todo_odd):
        g = i % 2
        gpu[g].append({"name": f"oddlayer_{c}_extract",
            "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 "
                   f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                   f"--mode generator --split eval --eval_ds {c} --layers 19 21 "
                   f"--stem_tag oddlayer_{c} --out feats_hidden",
            "log": f"logs/sv_oddlayer_{c}_extract.log", "timeout_s": 86400, "stall_s": 5400,
            "expect": f"{FEATS}/generator_eval_oddlayer_{c}.npz",
            "expect_min_bytes": 5_000_000, "_gpu": g})

    # ---- 4c3. BEST-OF-16 on the two benchmarks the verifier still loses --------------------
    # vqa_rad and vqamed are coverage-limited, not selection-limited (vqamed oracle@8 0.2102 vs
    # greedy 0.0947). More samples raise oracle@N; whether a trained verifier can convert that is
    # the open question -- self-consistency provably cannot (flat at the floor as N grows).
    for i, c in enumerate(["vqa_rad_open", "vqamed_open"]):
        g = i % 2
        if not nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc16.jsonl"):
            gpu[g].append({"name": f"{c}_sc16_gen",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 src/labeling/run_openvqa.py "
                       f"--model_path {L7} --tag lingshu7b_sc16 --dataset {c} --n_samples 16 "
                       f"--temp 0.7 --ckpt_dir {CK} --tp 1 --max_model_len 4096",
                "log": f"logs/sv_{c}_sc16_gen.log", "timeout_s": 43200, "stall_s": 2700,
                "expect": f"{CK}/ckpt_{c}_lingshu7b_sc16.jsonl", "_gpu": g})
        if not nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc16_scexploded.judge.jsonl"):
            gpu[g].append({"name": f"{c}_sc16_judge",
                "cmd": E + f"python3 src/cascade_methods/explode_sc_for_judge.py "
                       f"{CK}/ckpt_{c}_lingshu7b_sc16.jsonl && CUDA_VISIBLE_DEVICES={g} python3 "
                       f"src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 --preds "
                       f"{CK}/ckpt_{c}_lingshu7b_sc16_scexploded.jsonl",
                "log": f"logs/sv_{c}_sc16_judge.log", "timeout_s": 54000, "stall_s": 3600,
                "expect": f"{CK}/ckpt_{c}_lingshu7b_sc16_scexploded.judge.jsonl", "_gpu": g})
        if (nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc16_scexploded.judge.jsonl")
                and not nonempty(f"{FEATS}/generator_eval_{c}_sc16.npz", mb=5)):
            gpu[g].append({"name": f"{c}_sc16_extract",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 "
                       f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                       f"--mode generator --split eval --eval_ds {c} --pool_tag _sc16 "
                       f"--layers 18 19 20 21 22 --stem_tag {c}_sc16 --out feats_hidden",
                "log": f"logs/sv_{c}_sc16_extract.log", "timeout_s": 86400, "stall_s": 5400,
                "expect": f"{FEATS}/generator_eval_{c}_sc16.npz",
                "expect_min_bytes": 5_000_000, "_gpu": g})

    # ---- 4c4. ensemble-layer features for the TEMPERATURE pools ----------------------------
    # The T=0.2/0.4/1.0 caches hold layers [7,14,21,28] only, so every temperature result so far was
    # measured with a SINGLE-LAYER layer-21 probe. The shipped verifier rank-ensembles 18/20/22 and
    # has never been evaluated across temperature at all -- and pooling was shown to flatten the
    # layer-choice effect, so the temperature preference may well differ for it.
    TCELLS = ["vqa_rad_open", "slake_open", "pathvqa_open", "radimagenet_open",
              "vqamed_open", "gemex_open", "omnimed_open", "kvasir_x1_open"]
    TT = {"lingshu7bT02": "_T02", "lingshu7bT04": "_T04", "lingshu7bT10": "_T10"}
    todo_t = [(c, tag, suf) for c in TCELLS for tag, suf in TT.items()
              if nonempty(f"{CK}/ckpt_{c}_{tag}_sc8_scexploded.judge.jsonl")
              and not nonempty(f"{FEATS}/generator_eval_ens{suf}_{c}.npz", mb=5)]
    todo_t.sort(key=lambda x: SIZE.get(x[0], 99999))
    for i, (c, tag, suf) in enumerate(todo_t):
        g = i % 2
        gpu[g].append({"name": f"ens{suf}_{c}_extract",
            "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 "
                   f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                   f"--mode generator --split eval --eval_ds {c} --gen_tag {tag} "
                   f"--layers 18 20 22 --stem_tag ens{suf}_{c} --out feats_hidden",
            "log": f"logs/sv_ens{suf}_{c}_extract.log", "timeout_s": 86400, "stall_s": 5400,
            "expect": f"{FEATS}/generator_eval_ens{suf}_{c}.npz",
            "expect_min_bytes": 5_000_000, "_gpu": g})

    # ---- 4c5. 16-SAMPLE POOLS FOR EVERY BENCHMARK ------------------------------------------
    # The matched-budget 8-vs-16 comparison exists for two benchmarks and is a TIE on both. Two
    # points cannot separate "sampling more never helps" from "it helps where coverage binds", and
    # the coverage-limited pair is exactly the pair least able to answer that. Run all eight.
    SC16 = ["slake_open", "pathvqa_open", "radimagenet_open", "gemex_open",
            "omnimed_open", "kvasir_x1_open"]
    for i, c in enumerate(SC16):
        g = i % 2
        if not nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc16.jsonl"):
            gpu[g].append({"name": f"{c}_sc16_gen",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 src/labeling/run_openvqa.py "
                       f"--model_path {L7} --tag lingshu7b_sc16 --dataset {c} --n_samples 16 "
                       f"--temp 0.7 --ckpt_dir {CK} --tp 1 --max_model_len 4096",
                "log": f"logs/sv_{c}_sc16_gen.log", "timeout_s": 86400, "stall_s": 3600,
                "expect": f"{CK}/ckpt_{c}_lingshu7b_sc16.jsonl", "_gpu": g})
        if not nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc16_scexploded.judge.jsonl"):
            gpu[g].append({"name": f"{c}_sc16_judge",
                "cmd": E + f"python3 src/cascade_methods/explode_sc_for_judge.py "
                       f"{CK}/ckpt_{c}_lingshu7b_sc16.jsonl && CUDA_VISIBLE_DEVICES={g} python3 "
                       f"src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 --preds "
                       f"{CK}/ckpt_{c}_lingshu7b_sc16_scexploded.jsonl",
                "log": f"logs/sv_{c}_sc16_judge.log", "timeout_s": 86400, "stall_s": 4200,
                "expect": f"{CK}/ckpt_{c}_lingshu7b_sc16_scexploded.judge.jsonl", "_gpu": g})
        if (nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc16_scexploded.judge.jsonl")
                and not nonempty(f"{FEATS}/generator_eval_{c}_sc16.npz", mb=5)):
            gpu[g].append({"name": f"{c}_sc16_extract",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 "
                       f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                       f"--mode generator --split eval --eval_ds {c} --pool_tag _sc16 "
                       f"--layers 18 20 22 --stem_tag {c}_sc16 --out feats_hidden",
                "log": f"logs/sv_{c}_sc16_extract.log", "timeout_s": 86400, "stall_s": 5400,
                "expect": f"{FEATS}/generator_eval_{c}_sc16.npz",
                "expect_min_bytes": 5_000_000, "_gpu": g})

    # ---- 4c6. cold/hot pools for the TRAINING domains --------------------------------------
    # A temperature-matched POOLED verifier needs training pools at each temperature. T=0.2 exists;
    # T=0.4 and T=1.0 do not, so the 2x2 could only ever be run at one cold point.
    for i, c in enumerate(["vqa_rad_open_train", "slake_open_train", "kvasir_open",
                           "pathvqa_open_train"]):
        for tag, tv in (("lingshu7bT04", 0.4), ("lingshu7bT10", 1.0)):
            g = (i + (0 if tv < 1 else 1)) % 2
            if not nonempty(f"{CK}/ckpt_{c}_{tag}_sc8.jsonl"):
                gpu[g].append({"name": f"{c}_{tag}_gen",
                    "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 src/labeling/run_openvqa.py "
                           f"--model_path {L7} --tag {tag}_sc8 --dataset {c} --n_samples 8 "
                           f"--temp {tv} --ckpt_dir {CK} --tp 1 --max_model_len 4096",
                    "log": f"logs/sv_{c}_{tag}_gen.log", "timeout_s": 43200, "stall_s": 3600,
                    "expect": f"{CK}/ckpt_{c}_{tag}_sc8.jsonl", "_gpu": g})
            if not nonempty(f"{CK}/ckpt_{c}_{tag}_sc8_scexploded.judge.jsonl"):
                gpu[g].append({"name": f"{c}_{tag}_judge",
                    "cmd": E + f"python3 src/cascade_methods/explode_sc_for_judge.py "
                           f"{CK}/ckpt_{c}_{tag}_sc8.jsonl && CUDA_VISIBLE_DEVICES={g} python3 "
                           f"src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 --preds "
                           f"{CK}/ckpt_{c}_{tag}_sc8_scexploded.jsonl",
                    "log": f"logs/sv_{c}_{tag}_judge.log", "timeout_s": 86400, "stall_s": 4200,
                    "expect": f"{CK}/ckpt_{c}_{tag}_sc8_scexploded.judge.jsonl", "_gpu": g})

    # ---- 4c7. training features at T=0.4 and T=1.0 -----------------------------------------
    # The 2x2 temperature-mismatch test could only ever run at ONE cold point because only T=0.2
    # training features existed. The pools for 0.4 and 1.0 are now judged, so the full 4x4 is
    # reachable -- and it matters more now that head_temp_ensemble showed the SHIPPED verifier has
    # a 0.0416 macro spread across temperature, i.e. pooling did NOT flatten that effect.
    for i, (tag, suf) in enumerate((("lingshu7bT04", "T04"), ("lingshu7bT10", "T10"))):
        if not nonempty(f"{FEATS}/generator_train_{suf}.npz", mb=10):
            gpu[i % 2].append({"name": f"train_{suf}_extract",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={i % 2} python3 "
                       f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                       f"--mode generator --split train --gen_tag {tag} --layers 18 20 22 "
                       f"--stem_tag {suf} --out feats_hidden",
                "log": f"logs/sv_train_{suf}_extract.log", "timeout_s": 86400, "stall_s": 5400,
                "expect": f"{FEATS}/generator_train_{suf}.npz",
                "expect_min_bytes": 10_000_000, "_gpu": i % 2})

    # ---- 4c8. 32-sample pools, to extend the matched-budget curve beyond 8 vs 16 -----------
    # 8 vs 16 was a TIE on both coverage-limited benchmarks. One more doubling says whether that is
    # a plateau or just too small a step to resolve.
    for i, c in enumerate(["vqamed_open", "vqa_rad_open"]):
        g = i % 2
        if not nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc32.jsonl"):
            gpu[g].append({"name": f"{c}_sc32_gen",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 src/labeling/run_openvqa.py "
                       f"--model_path {L7} --tag lingshu7b_sc32 --dataset {c} --n_samples 32 "
                       f"--temp 0.7 --ckpt_dir {CK} --tp 1 --max_model_len 4096",
                "log": f"logs/sv_{c}_sc32_gen.log", "timeout_s": 86400, "stall_s": 3600,
                "expect": f"{CK}/ckpt_{c}_lingshu7b_sc32.jsonl", "_gpu": g})
        if not nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc32_scexploded.judge.jsonl"):
            gpu[g].append({"name": f"{c}_sc32_judge",
                "cmd": E + f"python3 src/cascade_methods/explode_sc_for_judge.py "
                       f"{CK}/ckpt_{c}_lingshu7b_sc32.jsonl && CUDA_VISIBLE_DEVICES={g} python3 "
                       f"src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 --preds "
                       f"{CK}/ckpt_{c}_lingshu7b_sc32_scexploded.jsonl",
                "log": f"logs/sv_{c}_sc32_judge.log", "timeout_s": 86400, "stall_s": 4200,
                "expect": f"{CK}/ckpt_{c}_lingshu7b_sc32_scexploded.judge.jsonl", "_gpu": g})
        if (nonempty(f"{CK}/ckpt_{c}_lingshu7b_sc32_scexploded.judge.jsonl")
                and not nonempty(f"{FEATS}/generator_eval_{c}_sc32.npz", mb=5)):
            gpu[g].append({"name": f"{c}_sc32_extract",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 "
                       f"src/training_methods/extract_generator_hidden.py --model_path {L7} "
                       f"--mode generator --split eval --eval_ds {c} --pool_tag _sc32 "
                       f"--layers 18 20 22 --stem_tag {c}_sc32 --out feats_hidden",
                "log": f"logs/sv_{c}_sc32_extract.log", "timeout_s": 86400, "stall_s": 5400,
                "expect": f"{FEATS}/generator_eval_{c}_sc32.npz",
                "expect_min_bytes": 5_000_000, "_gpu": g})

    # ---- 4c9. A SECOND GENERATOR: Qwen2.5-VL-7B --------------------------------------------
    # Every result in this project is on Lingshu-7B, so nothing yet distinguishes "this method
    # works" from "this method works on Lingshu". Qwen2.5-VL-7B-Instruct is the base model Lingshu
    # was finetuned from: same architecture, so extract_generator_hidden runs unchanged (InternVL3
    # was tried first and is INCOMPATIBLE -- AutoProcessor returns a bare tokenizer with no
    # image_processor, and AutoModelForImageTextToText rejects InternVLChatConfig), and it asks the
    # sharper question: does the method need the MEDICAL finetuning, or just a VLM?
    QW = "/data/dan/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/"
    QTAG = "qwen25vl7b"
    QTRAIN = ["kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"]
    QEVAL = ["pathvqa_open", "slake_open", "vqa_rad_open", "kvasir_x1_open"]
    for qi, c in enumerate(QTRAIN + QEVAL):
        g = qi % 2
        if not nonempty(f"{CK}/ckpt_{c}_{QTAG}_sc8.jsonl"):
            gpu[g].append({"name": f"q_{c}_gen",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 src/labeling/run_openvqa.py "
                       f"--model_path {QW} --tag {QTAG}_sc8 --dataset {c} --n_samples 8 "
                       f"--temp 0.7 --ckpt_dir {CK} --tp 1 --max_model_len 4096",
                "log": f"logs/sv_q_{c}_gen.log", "timeout_s": 86400, "stall_s": 3600,
                "expect": f"{CK}/ckpt_{c}_{QTAG}_sc8.jsonl", "_gpu": g})
        if c in QEVAL and not nonempty(f"{CK}/ckpt_{c}_{QTAG}.jsonl"):
            gpu[g].append({"name": f"q_{c}_greedy",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 src/labeling/run_openvqa.py "
                       f"--model_path {QW} --tag {QTAG} --dataset {c} --n_samples 1 "
                       f"--temp 0.0 --ckpt_dir {CK} --tp 1 --max_model_len 4096",
                "log": f"logs/sv_q_{c}_greedy.log", "timeout_s": 43200, "stall_s": 3600,
                "expect": f"{CK}/ckpt_{c}_{QTAG}.jsonl", "_gpu": g})
        if (nonempty(f"{CK}/ckpt_{c}_{QTAG}_sc8.jsonl")
                and not nonempty(f"{CK}/ckpt_{c}_{QTAG}_sc8_scexploded.judge.jsonl")):
            extra = f" {CK}/ckpt_{c}_{QTAG}.jsonl" if c in QEVAL else ""
            gpu[g].append({"name": f"q_{c}_judge",
                "cmd": E + f"python3 src/cascade_methods/explode_sc_for_judge.py "
                       f"{CK}/ckpt_{c}_{QTAG}_sc8.jsonl && CUDA_VISIBLE_DEVICES={g} python3 "
                       f"src/labeling/run_judge.py --tp 1 --gpu_mem 0.92 --preds "
                       f"{CK}/ckpt_{c}_{QTAG}_sc8_scexploded.jsonl" + extra,
                "log": f"logs/sv_q_{c}_judge.log", "timeout_s": 86400, "stall_s": 4200,
                "expect": f"{CK}/ckpt_{c}_{QTAG}_sc8_scexploded.judge.jsonl", "_gpu": g})
        split = "train" if c in QTRAIN else "eval"
        stem = f"qwen_{c}"
        if (nonempty(f"{CK}/ckpt_{c}_{QTAG}_sc8_scexploded.judge.jsonl")
                and not nonempty(f"{FEATS}/generator_{split}_{stem}.npz", mb=5)):
            eds = f"--eval_ds {c} " if split == "eval" else ""
            gpu[g].append({"name": f"q_{c}_extract",
                "cmd": E + f"CUDA_VISIBLE_DEVICES={g} python3 "
                       f"src/training_methods/extract_generator_hidden.py --model_path {QW} "
                       f"--mode generator --split {split} --gen_tag {QTAG} --layers 18 20 22 "
                       + eds + f"--stem_tag {stem} --out feats_hidden",
                "log": f"logs/sv_q_{c}_extract.log", "timeout_s": 86400, "stall_s": 5400,
                "expect": f"{FEATS}/generator_{split}_{stem}.npz",
                "expect_min_bytes": 5_000_000, "_gpu": g})

    # ---- 4d. CPU analyses unlocked by what already exists ----------------------------------
    if not os.path.exists(f"{ART}/pool_pruning_2026-08-24.json"):
        cpu.append({"name": f"pool_pruning_w{A.wave}",
            "cmd": T + "python3 -u src/cascade_methods/pool_pruning.py",
            "log": f"logs/sv_pool_pruning_w{A.wave}.log", "timeout_s": 36000, "stall_s": 3600,
            "expect": f"{ART}/pool_pruning_2026-08-24.json"})
    # zero-GPU analyses that new caches unlock
    EXTRA = [
        ("head_temp_matched", "src/training_methods/head_temp_matched.py --threads 4 --seeds 5",
         f"{ART}/head_temp_matched_2026-08-24.json",
         lambda: nonempty(f"{FEATS}/generator_train_T02.npz", mb=10)),
        ("head_representation", "src/training_methods/head_representation.py --threads 4 --seeds 3",
         f"{ART}/head_representation_2026-08-24.json",
         lambda: nonempty(f"{FEATS}/generator_train_finelayer.npz", mb=10)),
        ("coverage_scaling_all",
         "src/cascade_methods/coverage_scaling.py --all",
         f"{ART}/coverage_scaling_ALL_2026-09-01.json",
         lambda: all(nonempty(f"{FEATS}/generator_eval_{c}_sc16.npz", mb=5) for c in CELLS)),
        ("head_price_from_lobo",
         "src/training_methods/head_price_from_lobo.py --threads 4 --seeds 3",
         f"{ART}/head_price_from_lobo_2026-08-30.json",
         lambda: nonempty(f"{FEATS}/generator_train_finelayer.npz", mb=10)),
        ("head_temp_ensemble",
         "src/cascade_methods/head_temp_ensemble.py",
         f"{ART}/head_temp_ensemble_2026-08-30.json",
         lambda: len(glob.glob(f"{FEATS}/generator_eval_ens_T*.npz")) >= 20),
        ("head_layer_ensemble_width",
         "src/training_methods/head_ens_width.py --threads 4 --seeds 5",
         f"{ART}/head_ens_width_2026-08-25.json",
         lambda: nonempty(f"{FEATS}/generator_train_oddlayer.npz", mb=10)),
        ("head_input_augmentation",
         "src/training_methods/head_input_augmentation.py --threads 4 --seeds 3",
         f"{ART}/head_input_augmentation_2026-08-24.json",
         lambda: nonempty(f"{FEATS}/generator_train_s0of2.npz", mb=10)),
        ("decomposition_report", "src/cascade_methods/decomposition_report.py",
         f"{ART}/decomposition_2026-08-24.json", lambda: True),
    ]
    for name, cmd, out, ready in EXTRA:
        if ready() and not os.path.exists(out):
            cpu.append({"name": f"{name}_w{A.wave}",
                "cmd": T + f"python3 -X faulthandler -u {cmd}",
                "log": f"logs/sv_{name}_w{A.wave}.log", "timeout_s": 86400, "stall_s": 7200,
                "expect": out})

    for lay in (18, 19, 20, 21, 22):
        out = f"{ART}/head_layer{lay}_eval_2026-08-24.json"
        have = (nonempty(f"{FEATS}/generator_train_finelayer.npz", mb=10) if lay % 2 == 0
                else nonempty(f"{FEATS}/generator_train_oddlayer.npz", mb=10))
        if have and not os.path.exists(out):
            cpu.append({"name": f"head_layer{lay}_eval_w{A.wave}",
                "cmd": T + f"python3 -X faulthandler -u src/training_methods/head_layer_eval.py "
                       f"--layer {lay} --threads 4 --seeds 8",
                "log": f"logs/sv_head_layer{lay}_w{A.wave}.log",
                "timeout_s": 86400, "stall_s": 7200, "expect": out})

    # ---- 5. the fine-layer sweep shards (CPU) ---------------------------------------------
    for i, ls in enumerate([[10, 12], [16, 18], [20, 22], [24, 26]], 1):
        out = f"{ART}/head_finelayer_shard{i}.json"
        if not os.path.exists(out):
            cpu.append({"name": f"finelayer_shard{i}_w{A.wave}",
                "cmd": T + f"python3 -X faulthandler -u src/training_methods/head_finelayer.py "
                       f"--threads 4 --seeds 3 --layers {' '.join(map(str, ls))} --out {out}",
                "log": f"logs/sv_finelayer_shard{i}_w{A.wave}.log",
                "timeout_s": 86400, "stall_s": 7200, "expect": out,
                # a 126-byte stub from a failed json encode counted as success and made the
                # planner skip the job forever; require a real artifact
                "expect_min_bytes": 400})

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
