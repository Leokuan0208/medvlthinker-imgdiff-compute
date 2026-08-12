#!/usr/bin/env python3
"""train_unified_verifier.py -- ATTACK 2 arm B.  Train ONE verifier on BOTH branches' candidate sets.

The scorer, its prompt, its objective and its hyperparameters are IDENTICAL to the clean open-text
verifier (src/training_methods/run_lora_verifier_disjoint.py -> ckpts/train/lora_verifier_disjoint).
THE ONLY CHANGE IS THE DATA: the training set is the union of

  OPEN branch    the incumbent's own 10,364 examples, rebuilt from the SAME disjoint pools with the
                 SAME seed-0 composition quota, so this half is example-for-example the incumbent's.
  OPTION branch  (question, ONE OF THE GIVEN OPTIONS) pairs drawn from TRAIN splits of the
                 option-branch families.  Labels are FREE: label = (option is the gold option).
                 No judge, no generation, no sampling luck -- the candidate set is the prompt's own
                 complete answer space, exactly as at test time.
                   PMC-VQA   /data/dan/dataset/pmc_vqa_train/train_2.csv  (v2 train, 4 options)
                   PathVQA   path_vqa train parquet, answer in {yes,no}   (2 options)
                   VQA-RAD   flaviagiammarino/vqa-rad train, answer in {yes,no} (2 options)

DISJOINTNESS IS ENFORCED, NOT ASSUMED.  Every option-branch training image is dropped if its
DECODED-RGB md5 appears among the eval images of ANY of the eight reporting cells.  VQA-RAD in
particular recycles images across its own splits (103/120 open-eval images reappear in its train
split), so that filter is mandatory there.  The surviving counts and the number dropped are written
to the artifact.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 \
     src/training_methods/train_unified_verifier.py --seed 0 \
     --out_dir ckpts/train/lora_verifier_unified_s0

Resumable only at the file level (an existing --out_dir/adapter_model.safetensors is left alone
unless --force).  nohup, never tmux.  Launch from the repo root.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_open", type=int, default=10364, help="open-branch examples (incumbent = 10364)")
ap.add_argument("--max_option", type=int, default=10364, help="option-branch examples")
ap.add_argument("--out_dir", default="ckpts/train/lora_verifier_unified_s0")
ap.add_argument("--split", default="results/cascade_methods/artifacts/verifier_disjoint_split.json")
ap.add_argument("--idx_dir", default="data/disjoint_split")
ap.add_argument("--deadline_s", type=float, default=0.0)
ap.add_argument("--force", type=int, default=0)
ap.add_argument("--dry", type=int, default=0, help="build the data, write the manifest, do not train")
A = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

OUTD = os.path.join(ROOT, A.out_dir)
os.makedirs(OUTD, exist_ok=True)
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
MAXPX = HIGH_PX
DEV = "cuda"
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether "
       "the proposed answer is correct. Respond with only 'Yes' or 'No'.")
CK = os.path.join(ROOT, os.environ.get("VERIF_CK", "ckpts/openvqa/cheap_lingshu7b"))
TAG = os.environ.get("VERIF_TAG", "lingshu7b")
CONTAMINATED_QUOTA = {"slake": 894, "pathvqa": 4973, "vqa_rad": 522, "kvasir": 3975}
FAMILY = {"slake_open_train": "slake", "pathvqa_open_train": "pathvqa",
          "vqa_rad_open_train": "vqa_rad", "kvasir_open": "kvasir", "radimagenet_open": "radimagenet"}
T0 = time.time()


def loadj(p):
    return ({r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())}
            if os.path.exists(p) else {})


def norm(s):
    return str(s).strip().lower()


def pixhash(img):
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{img.size[0]}x{img.size[1]}|".encode())
    h.update(img.tobytes())
    return h.hexdigest()


# ==================================================================================================
# 1. the OPEN branch -- rebuilt exactly as run_lora_verifier_disjoint.py builds it
# ==================================================================================================
def slake_imgs(split):
    m = {}
    for x in json.load(open(f"/data/dan/dataset/slake/{split}.json")):
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip):
                m[x["qid"]] = (x["question"], ip)
    return m


def parquet_imgs(base, split, want):
    import pandas as pd
    m = {}
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/{split}-*.parquet"))],
                   ignore_index=True)
    for i, r in df.iterrows():
        if int(i) not in want:
            continue
        q, a = r.get("question"), r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]
            q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
        if str(a).strip().lower() in ("yes", "no"):
            continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m


def json_imgs(jp):
    return {r["idx"]: (r["question"], r["img_path"]) for r in json.load(open(jp))
            if os.path.exists(r["img_path"])}


def build_open_examples(rng):
    if A.max_open <= 0:                       # option-only arm B0: skip the whole open build
        return [], {"per_source": {}, "quota_shortfall": {}, "skipped": "max_open == 0"}
    SPL = json.load(open(os.path.join(ROOT, A.split)))
    assert SPL["disjointness_assertion"]["image_pixel_hash_intersection"] == 0
    DSETS = list(SPL["train"].keys())
    ALLOW = {ds: set(json.load(open(os.path.join(ROOT, A.idx_dir, f"idx_{ds}.json"))))
             for ds in DSETS}
    IMG = {}
    for ds in DSETS:
        if ds == "slake_open_train":
            IMG[ds] = slake_imgs("train")
        elif ds == "vqa_rad_open_train":
            IMG[ds] = parquet_imgs("/data/dan/dataset/vqa_rad/data", "train", ALLOW[ds])
        elif ds == "pathvqa_open_train":
            IMG[ds] = parquet_imgs("/data/dan/dataset/path_vqa/data", "train", ALLOW[ds])
        elif ds == "kvasir_open":
            IMG[ds] = json_imgs("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json")
        elif ds == "radimagenet_open":
            IMG[ds] = json_imgs("/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")
        IMG[ds] = {k: v for k, v in IMG[ds].items() if k in ALLOW[ds]}
    QREC = {}
    for ds in DSETS:
        sc = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8.jsonl")
        exp = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.jsonl")
        jud = {k: v["judge_ok"] for k, v in loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.judge.jsonl").items()}
        if not sc or not jud:
            continue
        aj = defaultdict(dict)
        for cid, r in exp.items():
            if cid in jud:
                oi = cid.split("#")[0]
                oi = int(oi) if oi.lstrip("-").isdigit() else oi
                aj[oi][norm(r["modal_pred"])] = jud[cid]
        for i in sc:
            if i not in ALLOW[ds] or i not in IMG[ds] or i not in aj:
                continue
            q, img = IMG[ds][i]
            QREC[(ds, i)] = {"q": q, "img": img, "preds": sc[i]["preds"], "slabels": aj[i]}
    by_src = defaultdict(list)
    for k in sorted(QREC, key=lambda t: (t[0], str(t[1]))):
        r = QREC[k]
        for na, lab in r["slabels"].items():
            surf = next((a for a in r["preds"] if norm(a) == na), na)
            by_src[k[0]].append(("open", k[0], r["q"], r["img"], surf, int(lab)))
    ex, taken, short = [], {}, {}
    for ds in DSETS:
        fam = FAMILY[ds]
        if fam == "radimagenet":
            continue
        want = CONTAMINATED_QUOTA[fam]
        pool = list(by_src.get(ds, [])); rng.shuffle(pool)
        take = pool[:want]; ex += take; taken[ds] = len(take)
        if len(take) < want:
            short[ds] = want - len(take)
    deficit = A.max_open - len(ex)
    if deficit > 0:
        pool = list(by_src.get("radimagenet_open", [])); rng.shuffle(pool)
        take = pool[:deficit]; ex += take; taken["radimagenet_open"] = len(take)
    return ex[:A.max_open], {"per_source": taken, "quota_shortfall": short}


# ==================================================================================================
# 2. the OPTION branch -- (question, one given option), label = (option is gold).  FREE labels.
# ==================================================================================================
def eval_image_hashes_all():
    """Union of the DECODED-RGB md5 of every image that can appear in ANY of the eight reporting
    cells.  Deliberately STRICTER than the eight cells themselves: the WHOLE test split of every
    family is banned (PMC-VQA test_2, MedXpertQA-MM, SLAKE test, VQA-RAD test, PathVQA test), so a
    training image cannot leak into a cell through a row we did not happen to score."""
    cache = os.path.join(ROOT, "data/unified_pipeline/eval_imghash_all.json")
    if os.path.exists(cache):
        return set(json.load(open(cache)))
    import csv
    import glob as _g
    import pandas as pd
    from datasets import load_dataset
    hs, per = set(), {}

    base = "/data/dan/dataset/medevalkit/PMC-VQA"
    figs = sorted({r[1] for r in list(csv.reader(open(os.path.join(base, "test_2.csv"),
                                                      encoding="utf-8")))[1:]})
    n0 = len(hs)
    for f in figs:
        p = os.path.join(base, "figures", f)
        try:
            hs.add(pixhash(p))
        except Exception:
            pass
    per["PMC_VQA_test_2"] = len(hs) - n0

    dp = os.path.join(ROOT, "MedEvalKit/datas/MedXpertQA")
    n0 = len(hs)
    for f in _g.glob(os.path.join(dp, "images", "*")):
        try:
            hs.add(pixhash(f))
        except Exception:
            pass
    per["MedXpertQA_images"] = len(hs) - n0

    n0 = len(hs)
    for x in json.load(open(os.path.join(ROOT, "MedEvalKit/datas/SLAKE/test.json"))):
        p = os.path.join(ROOT, "MedEvalKit/datas/SLAKE/imgs", x["img_name"])
        try:
            hs.add(pixhash(p))
        except Exception:
            pass
    per["SLAKE_test"] = len(hs) - n0

    n0 = len(hs)
    for s in load_dataset("flaviagiammarino/vqa-rad", split="test"):
        try:
            hs.add(pixhash(s["image"]))
        except Exception:
            pass
    per["VQA_RAD_test"] = len(hs) - n0

    n0 = len(hs)
    for sp in sorted(_g.glob("/data/dan/dataset/path_vqa/data/test-*.parquet")):
        df = pd.read_parquet(sp)
        for _, r in df.iterrows():
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                try:
                    hs.add(pixhash(Image.open(io.BytesIO(img["bytes"]))))
                except Exception:
                    pass
    per["PATH_VQA_test"] = len(hs) - n0

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(sorted(hs), open(cache, "w"))
    json.dump(per, open(cache.replace(".json", "_breakdown.json"), "w"), indent=1)
    print(f"[ban] eval image hashes per source (new-unique): {per}", flush=True)
    return hs


def build_option_examples(rng, banned):
    """Balanced over the three option-branch families; every image checked against `banned`."""
    import csv
    import pandas as pd
    pools = {}
    drop = {}

    # ---- PMC-VQA train (4 options) ----------------------------------------------------------
    # LANDMINE (CLAUDE.md sec.0): there are TWO PMC-VQA splits.  The EVAL cell is v2 `test_2.csv`,
    # whose Figure_path values are SUB-FIGURE CROPS ("PMC8253797_Fig4_11.jpg").  The images actually
    # on disk under pmc_vqa_train/images are the v1 full figures, so v2 `train_2.csv` resolves ZERO
    # images (verified: 0 hits in the first 2,000 rows; the first dry build produced 0 PMC examples
    # and would have trained an "MCQ" verifier that never saw a lettered 4-option item).  Train on
    # the v1 `train.csv`, which resolves 2000/2000.
    # Because a v2 test CROP cannot pixel-md5-match a v1 full FIGURE, the md5 ban list alone would
    # not catch "this training figure contains that test crop".  So a STRICTLY STRONGER filter is
    # applied here as well: drop any training row whose PMC ARTICLE ID appears anywhere in the v2
    # test_2 figure list.
    base = "/data/dan/dataset/pmc_vqa_train"
    ev = "/data/dan/dataset/medevalkit/PMC-VQA"
    ban_pmcid = {str(r[1]).split("_")[0] for r in
                 list(csv.reader(open(os.path.join(ev, "test_2.csv"), encoding="utf-8")))[1:]}
    rows = list(csv.reader(open(os.path.join(base, "train.csv"), encoding="utf-8")))[1:]
    rng.shuffle(rows)
    pool, dropped, seen_img, dropped_id = [], 0, 0, 0
    for r in rows:
        if len(pool) >= A.max_option:            # PMC alone can fill the quota; cap the scan
            break
        fig, q, _ans_text, cA, cB, cC, cD, ans = r[:8]
        if str(fig).split("_")[0] in ban_pmcid:
            dropped_id += 1
            continue
        ip = os.path.join(base, "images", fig)
        if not os.path.exists(ip):
            continue
        try:
            h = pixhash(ip)
        except Exception:
            continue
        seen_img += 1
        if h in banned:
            dropped += 1
            continue
        bodies, letters = [], []
        ok = True
        for c in (cA, cB, cC, cD):
            m = __import__("re").match(r"^\s*([A-Za-z])\s*[:.)]\s*(.*)$", str(c), __import__("re").S)
            if not m:
                ok = False; break
            letters.append(m.group(1).upper()); bodies.append(m.group(2).strip())
        gold = str(ans).strip().upper()
        if not ok or gold not in letters:
            continue
        gi = letters.index(gold)
        for j, b in enumerate(bodies):
            pool.append(("option", "pmc_vqa_train", q.strip(), ip, b, int(j == gi)))
    pools["pmc_vqa_train"] = pool
    drop["pmc_vqa_train"] = {"source_csv": "pmc_vqa_train/train.csv (v1)",
                             "images_checked": seen_img, "images_dropped_as_eval": dropped,
                             "rows_dropped_by_PMC_article_id_shared_with_test_2": dropped_id,
                             "n_banned_pmc_article_ids": len(ban_pmcid)}

    # ---- PathVQA train yes/no ---------------------------------------------------------------
    pool, dropped, seen_img = [], 0, 0
    df = pd.concat([pd.read_parquet(f) for f in
                    sorted(glob.glob("/data/dan/dataset/path_vqa/data/train-*.parquet"))],
                   ignore_index=True)
    order = list(range(len(df))); rng.shuffle(order)
    for i in order:
        if len(pool) >= A.max_option:
            break
        r = df.iloc[i]
        a = str(r.get("answer", "")).strip().lower()
        if a not in ("yes", "no"):
            continue
        img = r["image"]
        if not (isinstance(img, dict) and "bytes" in img):
            continue
        try:
            im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            h = pixhash(im)
        except Exception:
            continue
        seen_img += 1
        if h in banned:
            dropped += 1
            continue
        for c in ("yes", "no"):
            pool.append(("option", "pathvqa_closed_train", str(r["question"]), im, c, int(c == a)))
    pools["pathvqa_closed_train"] = pool
    drop["pathvqa_closed_train"] = {"images_checked": seen_img, "images_dropped_as_eval": dropped}

    # ---- VQA-RAD train yes/no (images RECYCLED across splits -> the filter matters most here) --
    pool, dropped, seen_img = [], 0, 0
    from datasets import load_dataset
    dsr = load_dataset("flaviagiammarino/vqa-rad", split="train")
    order = list(range(len(dsr))); rng.shuffle(order)
    for i in order:
        if len(pool) >= A.max_option:
            break
        s = dsr[int(i)]
        a = str(s["answer"]).strip().lower()
        if a not in ("yes", "no"):
            continue
        try:
            im = s["image"].convert("RGB")
            h = pixhash(im)
        except Exception:
            continue
        seen_img += 1
        if h in banned:
            dropped += 1
            continue
        for c in ("yes", "no"):
            pool.append(("option", "vqa_rad_closed_train", str(s["question"]), im, c, int(c == a)))
    pools["vqa_rad_closed_train"] = pool
    drop["vqa_rad_closed_train"] = {"images_checked": seen_img, "images_dropped_as_eval": dropped}

    # ---- equal-ish mix: 1/3 each, topped up from PMC ----------------------------------------
    per = A.max_option // 3
    ex, taken = [], {}
    for k in ["pathvqa_closed_train", "vqa_rad_closed_train", "pmc_vqa_train"]:
        p = pools[k][:per]
        ex += p; taken[k] = len(p)
    if len(ex) < A.max_option:
        extra = pools["pmc_vqa_train"][taken["pmc_vqa_train"]:
                                       taken["pmc_vqa_train"] + (A.max_option - len(ex))]
        ex += extra; taken["pmc_vqa_train"] += len(extra)
    return ex[:A.max_option], {"per_source": taken, "image_filter": drop,
                               "available": {k: len(v) for k, v in pools.items()}}


# ==================================================================================================
# 3. train
# ==================================================================================================
def main():
    rng = random.Random(A.seed)
    np.random.seed(A.seed); torch.manual_seed(A.seed)

    print("[data] option-branch eval-image ban list ...", flush=True)
    banned = eval_image_hashes_all()
    print(f"[data] banned eval images = {len(banned)}", flush=True)

    print("[data] open branch ...", flush=True)
    open_ex, open_meta = build_open_examples(rng)
    print(f"[data] open examples = {len(open_ex)}  {open_meta}", flush=True)

    print("[data] option branch ...", flush=True)
    opt_ex, opt_meta = build_option_examples(rng, banned)
    print(f"[data] option examples = {len(opt_ex)}  {opt_meta['per_source']}", flush=True)
    print(f"[data] option image filter = {opt_meta['image_filter']}", flush=True)

    train_ex = open_ex + opt_ex
    rng.shuffle(train_ex)
    manifest = {
        "seed": A.seed, "n_open": len(open_ex), "n_option": len(opt_ex), "n_total": len(train_ex),
        "pos_rate_open": float(np.mean([e[5] for e in open_ex])) if open_ex else None,
        "pos_rate_option": float(np.mean([e[5] for e in opt_ex])) if opt_ex else None,
        "open_meta": open_meta, "option_meta": {k: v for k, v in opt_meta.items()},
        "hp": {"lora_r": A.lora_r, "lr": A.lr, "bs": A.bs, "accum": A.accum, "epochs": A.epochs,
               "max_pixels": MAXPX, "min_pixels": MIN_PX,
               "identical_to": "src/training_methods/run_lora_verifier_disjoint.py"},
        "n_banned_eval_images": len(banned),
    }
    json.dump(manifest, open(os.path.join(OUTD, "unified_manifest.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in manifest.items() if k != "option_meta"}, indent=1,
                     default=str), flush=True)
    if A.dry:
        return
    if os.path.exists(os.path.join(OUTD, "adapter_model.safetensors")) and not A.force:
        print("[skip] adapter already exists; pass --force 1 to retrain", flush=True)
        return

    # ---- GPU etiquette: three other rounds share these cards, and the CPU data build above takes
    # ~10 min, which is long enough for a card that was free when the runner picked it to fill up
    # (it did, once: OOM at load with four foreign processes holding 79.07 of 79.14 GiB).  So the
    # wait for room happens HERE, immediately before the weights are touched, and it never kills
    # anything -- it only waits.
    import subprocess
    global DEV

    def free_per_visible():
        """(local_index, free GiB) for every VISIBLE device, in torch's own ordering."""
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        rows = [[int(x) for x in ln.split(",")] for ln in q.stdout.strip().splitlines()]
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        order = [int(x) for x in vis.split(",")] if vis.strip() else list(range(len(rows)))
        return [(i, (rows[p][0] - rows[p][1]) / 1024.0) for i, p in enumerate(order)]

    need = float(os.environ.get("TRAIN_MIN_FREE_GIB", "26"))

    def wait_room(tag=""):
        """Wait for ANY visible device to hold a sustained `need` GiB, then bind to it."""
        streak, waited, chosen = 0, 0, None
        while waited < 21600:
            cand = max(free_per_visible(), key=lambda t: t[1])
            if cand[1] >= need and (chosen is None or cand[0] == chosen):
                chosen = cand[0]
                streak += 1
                if streak >= 3:
                    DEV = f"cuda:{chosen}"
                    torch.cuda.set_device(chosen)
                    print(f"[gpu]{tag} bound {DEV} with {cand[1]:.1f} GiB free after {waited}s",
                          flush=True)
                    return True
            else:
                streak, chosen = 0, None
                if waited % 300 == 0:
                    print(f"  [gpu-wait]{tag} best={cand} need {need}, waited {waited}s", flush=True)
            time.sleep(30); waited += 30
        return False

    if not wait_room():
        print("[gpu] ABORT: no room in 6h", flush=True)
        return

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    from peft import LoraConfig, get_peft_model

    proc = AutoProcessor.from_pretrained(A.model_path)
    YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]

    def encode(q, img, proposed, label):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": [
                    {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX},
                    {"type": "text", "text": f"Question: {q}\nProposed answer: {proposed}\n"
                                             "Is the proposed answer correct? Answer Yes or No."}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = text + ("Yes" if label == 1 else "No")
        imgs, vids = process_vision_info(msgs)
        return proc(text=[text], images=imgs, videos=vids, return_tensors="pt", padding=True)

    model = None
    for attempt in range(40):                    # a foreign process can grab the card mid-load
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                A.model_path, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2").to(DEV)
            break
        except torch.OutOfMemoryError as e:
            print(f"  [load-oom] attempt {attempt}: {str(e)[:100]}", flush=True)
            model = None
            torch.cuda.empty_cache()
            time.sleep(120)
            wait_room(f" retry{attempt}")
    if model is None:
        print("[gpu] ABORT: could not load the base model without OOM", flush=True)
        return
    model = get_peft_model(model, LoraConfig(
        r=A.lora_r, lora_alpha=2 * A.lora_r, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)

    nsteps = len(train_ex) * A.epochs
    step = 0
    opt.zero_grad()
    for ep in range(A.epochs):
        for e in train_ex:
            _branch, _src, q, img, ans, lab = e
            try:
                enc = encode(q, img, ans, lab).to(DEV)
                labels = enc["input_ids"].clone()
                labels[:, :-1] = -100
                out = model(**enc, labels=labels)
                (out.loss / A.accum).backward()
            except Exception as ex:
                print(f"  !! skip example: {str(ex)[:120]}", flush=True)
                opt.zero_grad()
                torch.cuda.empty_cache()
                step += 1
                continue
            step += 1
            if step % A.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
            if step % 400 == 0:
                print(f"  ep{ep} step{step}/{nsteps} {(time.time()-T0)/60:.1f}min", flush=True)
            if A.deadline_s > 0 and time.time() - T0 > A.deadline_s:
                print("[deadline] stopping cleanly", flush=True)
                model.save_pretrained(OUTD)
                json.dump({"stopped_early_at_step": step, "of": nsteps},
                          open(os.path.join(OUTD, "early_stop.json"), "w"))
                return
    model.save_pretrained(OUTD)
    print(f"saved -> {A.out_dir} in {(time.time()-T0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
