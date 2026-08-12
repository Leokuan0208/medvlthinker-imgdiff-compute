#!/usr/bin/env python3
"""build_cheapleg_train_split.py -- design + PROVE a strictly image-disjoint TRAIN split for
GENERATOR (cheap-leg) adaptation of Lingshu-7B.

WHY.  Every ceiling in this project (p10, oracle@8, greedy, sel_eff) is a property of a FROZEN
Lingshu-7B.  The verifier has been trained repeatedly; the generator never has.  Attack B trains the
cheap leg.  For that to be measurable at all, the training items must share NO IMAGE with any of the
eight reporting cells.

DISCIPLINE, copied verbatim from src/training_methods/build_disjoint_verifier_split.py (which proved
the clean verifier's split): images are compared by md5 of the DECODED RGB pixels (WxH header + raw
bytes), so a re-encoded / re-compressed copy of the same image is still caught; item identity is
(dataset family, normalized question text, image pixel hash) because raw row ids are not comparable
across different split files.  Build FAILS (assert) on any leak.

EVAL SIDE -- every image behind the 8 Variant-B reporting cells:
  PMC_VQA          MedEvalKit /data/dan/dataset/medevalkit/PMC-VQA/test_2.csv  (v2, 33,430 rows)
                   *** THE TWO-SPLIT LANDMINE (CLAUDE.md sec 0): this is test_2.csv, NOT test_clean.csv.
                   The training pool is therefore train_2.csv, the v2 train split. ***
  SLAKE cl/open    /data/dan/dataset/medevalkit/SLAKE/test.json  (all langs hashed; en is what is scored)
  VQA_RAD cl/open  /data/dan/dataset/vqa_rad/data/test-*.parquet
  PATH_VQA cl/open /data/dan/dataset/path_vqa/data/test-*.parquet
  MedXpertQA-MM    /data/dan/dataset/medevalkit/MedXpertQA/images  (no train source draws from it;
                   hashed anyway so the assertion covers all eight cells)

TRAIN SIDE -- official TRAIN splits only:
  pmc_vqa_train_mcq  PMC-VQA train_2.csv        -> MedEvalKit multiple-choice frame, target = letter
  slake_train_*      SLAKE train.json (en)      -> CLOSED and OPEN
  vqa_rad_train_*    VQA-RAD train parquet      -> yes/no (closed) and open
  pathvqa_train_*    PathVQA train parquet      -> yes/no (closed) and open

TWO PROMPT FRAMES are emitted, because the two arms of the cascade are evaluated under two different
prompts and two different image resolutions, and a resolution/prompt mismatch is exactly the defect
that forced this project's Finding-1 correction:
  frame "medeval" : MedEvalKit's own prompt, FULL resolution (CAP_MAX_PIXELS unset at eval)
                    -> serves PMC_VQA, SLAKE_closed, VQA_RAD_closed, PATH_VQA_closed, MedXpertQA-MM
  frame "openvqa" : run_openvqa.py's SYS prompt, cap320 (1280*28*28//4 px)
                    -> serves SLAKE_open, VQA_RAD_open, PATH_VQA_open

  python3 src/training_methods/build_cheapleg_train_split.py
  -> results/cascade_methods/artifacts/cheapleg_train_split_2026-08-11.json
  -> data/cheapleg_split/train_manifest.json     (the eligible training items, per source)
"""
import argparse, csv, glob, hashlib, io, json, os, random, re, string, sys
from collections import defaultdict
from multiprocessing import Pool

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
MEK_PMC = "/data/dan/dataset/medevalkit/PMC-VQA"
MEK_SLAKE = "/data/dan/dataset/medevalkit/SLAKE"
MEK_MXP = "/data/dan/dataset/medevalkit/MedXpertQA/images"
PATHVQA = "/data/dan/dataset/path_vqa/data"
VQARAD = "/data/dan/dataset/vqa_rad/data"

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pmc_candidates", type=int, default=14000,
                help="how many PMC-VQA train_2 rows to hash as candidates (hashing all 152,603 is "
                     "wasteful; we need <=6k examples)")
ap.add_argument("--procs", type=int, default=40)
ap.add_argument("--out", default="results/cascade_methods/artifacts/cheapleg_train_split_2026-08-11.json")
ap.add_argument("--manifest_dir", default="data/cheapleg_split")
A = ap.parse_args()


def qnorm(s):
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def _pix(img):
    img = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{img.size[0]}x{img.size[1]}|".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def hash_path(p):
    try:
        return _pix(Image.open(p))
    except Exception as e:                                    # per-item error guard
        return "ERR:" + str(e)[:40]


def hash_bytes(b):
    try:
        return _pix(Image.open(io.BytesIO(b)))
    except Exception as e:
        return "ERR:" + str(e)[:40]


def pmap(fn, xs, procs, tag):
    if not xs:
        return []
    with Pool(procs) as pool:
        out = []
        for i, h in enumerate(pool.imap(fn, xs, chunksize=32)):
            out.append(h)
            if (i + 1) % 5000 == 0:
                print(f"    {tag} {i+1}/{len(xs)}", flush=True)
    return out


# =================================================================================== EVAL SIDE
print("[eval] hashing every image behind the 8 Variant-B reporting cells ...", flush=True)
EVAL = {}          # name -> {"img": [hash], "item": set((fam,qnorm,hash))}
t_report = {}

# --- PMC-VQA (MedEvalKit test_2.csv, the v2 split; landmine CLAUDE.md sec 0) ---
rows = list(csv.reader(open(f"{MEK_PMC}/test_2.csv", encoding="utf-8")))
pmc_eval_rows = rows[1:]
pmc_eval_figs = sorted(set(r[1] for r in pmc_eval_rows))
paths = [os.path.join(MEK_PMC, "figures", f) for f in pmc_eval_figs]
hs = pmap(hash_path, paths, A.procs, "pmc-eval")
pmc_eval_h = dict(zip(pmc_eval_figs, hs))
bad = [f for f, h in pmc_eval_h.items() if h.startswith("ERR:")]
EVAL["PMC_VQA"] = {"img": set(h for h in hs if not h.startswith("ERR:")),
                   "item": set(("pmc_vqa", qnorm(r[3]), pmc_eval_h[r[1]]) for r in pmc_eval_rows
                               if not pmc_eval_h[r[1]].startswith("ERR:"))}
t_report["PMC_VQA"] = dict(file="/data/dan/dataset/medevalkit/PMC-VQA/test_2.csv", n_rows=len(pmc_eval_rows),
                           n_unique_figures=len(pmc_eval_figs), n_unhashable=len(bad))
print(f"  PMC_VQA        rows={len(pmc_eval_rows)} unique figures={len(pmc_eval_figs)} "
      f"distinct pixel hashes={len(EVAL['PMC_VQA']['img'])} unhashable={len(bad)}", flush=True)

# --- SLAKE test (imgs on disk) ---
slake_test = json.load(open(f"{MEK_SLAKE}/test.json"))
sl_names = sorted(set(x["img_name"] for x in slake_test))
hs = pmap(hash_path, [os.path.join(MEK_SLAKE, "imgs", n) for n in sl_names], A.procs, "slake-eval")
sl_h = dict(zip(sl_names, hs))
EVAL["SLAKE"] = {"img": set(h for h in hs if not h.startswith("ERR:")),
                 "item": set(("slake", qnorm(x["question"]), sl_h[x["img_name"]]) for x in slake_test
                             if not sl_h[x["img_name"]].startswith("ERR:"))}
t_report["SLAKE"] = dict(file=f"{MEK_SLAKE}/test.json", n_rows=len(slake_test),
                         n_rows_en=len([x for x in slake_test if x.get("q_lang") == "en"]),
                         n_unique_images=len(sl_names))
print(f"  SLAKE          rows={len(slake_test)} images={len(sl_names)} "
      f"hashes={len(EVAL['SLAKE']['img'])}", flush=True)


def load_parquet(base, split):
    """(row_idx, question, answer, image_bytes). row_idx = index in the concatenated sorted parquets
    -- run_openvqa.py's idx convention, so the open-cell eval item ids line up."""
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/{split}-*.parquet"))],
                   ignore_index=True)
    out = []
    for i, r in df.iterrows():
        q, a = r.get("question"), r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]
            q = conv[0]["value"].replace("<image>", "").strip()
            a = conv[1]["value"]
        img = r["image"]
        if not (isinstance(img, dict) and "bytes" in img):
            continue
        out.append((int(i), str(q), str(a).strip(), img["bytes"]))
    return out


for name, base in (("VQA_RAD", VQARAD), ("PATH_VQA", PATHVQA)):
    pool = load_parquet(base, "test")
    hs = pmap(hash_bytes, [p[3] for p in pool], A.procs, f"{name}-eval")
    fam = name.lower()
    EVAL[name] = {"img": set(h for h in hs if not h.startswith("ERR:")),
                  "item": set((fam, qnorm(p[1]), h) for p, h in zip(pool, hs) if not h.startswith("ERR:"))}
    t_report[name] = dict(file=f"{base}/test-*.parquet", n_rows=len(pool),
                          n_unique_images=len(EVAL[name]["img"]))
    print(f"  {name:14s} rows={len(pool)} distinct pixel hashes={len(EVAL[name]['img'])}", flush=True)

# --- MedXpertQA-MM images (no training source draws from it; hashed so the assertion covers all 8 cells)
mxp = sorted(glob.glob(f"{MEK_MXP}/*"))
hs = pmap(hash_path, mxp, A.procs, "mxp-eval")
EVAL["MedXpertQA-MM"] = {"img": set(h for h in hs if not h.startswith("ERR:")), "item": set()}
t_report["MedXpertQA-MM"] = dict(dir=MEK_MXP, n_files=len(mxp), n_unique_images=len(EVAL["MedXpertQA-MM"]["img"]))
print(f"  MedXpertQA-MM  files={len(mxp)} hashes={len(EVAL['MedXpertQA-MM']['img'])}", flush=True)

EVAL_IMG = set().union(*[v["img"] for v in EVAL.values()])
EVAL_ITEM = set().union(*[v["item"] for v in EVAL.values()])
print(f"  EVAL TOTAL: {len(EVAL_IMG)} distinct eval images / {len(EVAL_ITEM)} (family,question,image) items",
      flush=True)

# =================================================================================== TRAIN SIDE
print("\n[train] hashing candidate TRAIN-split items ...", flush=True)
rng = random.Random(A.seed)
TRAIN = {}          # source -> list of dicts (kept, disjoint)
dropped = {}

# --- PMC-VQA train_2.csv (v2 train split) -> MCQ frame -----------------------------------------
rows = list(csv.reader(open(f"{MEK_PMC}/train_2.csv", encoding="utf-8")))
pmc_tr = rows[1:]
rng.shuffle(pmc_tr)
cand = pmc_tr[:A.pmc_candidates]
figs = sorted(set(r[1] for r in cand))
hs = pmap(hash_path, [os.path.join(MEK_PMC, "figures", f) for f in figs], A.procs, "pmc-train")
fh = dict(zip(figs, hs))
keep, drop_img, drop_item, drop_err = [], 0, 0, 0
for r in cand:
    h = fh[r[1]]
    if h.startswith("ERR:"):
        drop_err += 1; continue
    if h in EVAL_IMG:
        drop_img += 1; continue
    if ("pmc_vqa", qnorm(r[3]), h) in EVAL_ITEM:
        drop_item += 1; continue
    # row: index, Figure_path, Caption, Question, A, B, C, D, Answer, split
    keep.append(dict(source="pmc_vqa_train_mcq", frame="medeval", idx=r[0], fig=r[1], imghash=h,
                     question=r[3], choices=[r[4], r[5], r[6], r[7]], answer=r[8]))
TRAIN["pmc_vqa_train_mcq"] = keep
dropped["pmc_vqa_train_mcq"] = dict(eval_image=drop_img, eval_item=drop_item, unhashable=drop_err,
                                    candidates=len(cand), total_pool=len(pmc_tr))
print(f"  pmc_vqa_train_mcq   candidates={len(cand)} -eval_image={drop_img} -eval_item={drop_item} "
      f"-> kept={len(keep)}", flush=True)

# --- SLAKE train.json (en) -> closed + open ----------------------------------------------------
# BOTH LANGUAGES: the SLAKE_closed reporting cell is n=836 = 416 en + 420 zh (MedEvalKit scores
# every row of test.json), so an en-only training pool would leave half the cell untouched.
sl_tr = list(json.load(open(f"{MEK_SLAKE}/train.json")))
names = sorted(set(x["img_name"] for x in sl_tr))
hs = pmap(hash_path, [os.path.join(MEK_SLAKE, "imgs", n) for n in names], A.procs, "slake-train")
nh = dict(zip(names, hs))
for kind in ("closed", "open"):
    keep, drop_img, drop_item = [], 0, 0
    for x in sl_tr:
        if (x.get("answer_type") == "CLOSED") != (kind == "closed"):
            continue
        h = nh[x["img_name"]]
        if h.startswith("ERR:"):
            continue
        if h in EVAL_IMG:
            drop_img += 1; continue
        if ("slake", qnorm(x["question"]), h) in EVAL_ITEM:
            drop_item += 1; continue
        if kind == "open" and x.get("q_lang") != "en":
            continue          # the SLAKE_open cell is en-only (run_openvqa.py filters q_lang == "en")
        keep.append(dict(source=f"slake_train_{kind}", frame="medeval" if kind == "closed" else "openvqa",
                         idx=x["qid"], lang=x.get("q_lang"),
                         img=os.path.join(MEK_SLAKE, "imgs", x["img_name"]), imghash=h,
                         question=x["question"], answer=str(x["answer"])))
    TRAIN[f"slake_train_{kind}"] = keep
    dropped[f"slake_train_{kind}"] = dict(eval_image=drop_img, eval_item=drop_item)
    print(f"  slake_train_{kind:7s} -eval_image={drop_img} -eval_item={drop_item} -> kept={len(keep)}",
          flush=True)

# --- VQA-RAD / PathVQA train parquet -> closed(yes/no) + open ----------------------------------
for fam, base in (("vqa_rad", VQARAD), ("pathvqa", PATHVQA)):
    pool = load_parquet(base, "train")
    hs = pmap(hash_bytes, [p[3] for p in pool], A.procs, f"{fam}-train")
    for kind in ("closed", "open"):
        keep, drop_img, drop_item = [], 0, 0
        for (i, q, a, b), h in zip(pool, hs):
            yn = str(a).strip().lower() in ("yes", "no")
            if yn != (kind == "closed"):
                continue
            if h.startswith("ERR:"):
                continue
            efam = "vqa_rad" if fam == "vqa_rad" else "path_vqa"
            if h in EVAL_IMG:
                drop_img += 1; continue
            if (efam, qnorm(q), h) in EVAL_ITEM:
                drop_item += 1; continue
            keep.append(dict(source=f"{fam}_train_{kind}",
                             frame="medeval" if kind == "closed" else "openvqa",
                             idx=i, parquet=f"{base}/train", imghash=h, question=q, answer=a))
        TRAIN[f"{fam}_train_{kind}"] = keep
        dropped[f"{fam}_train_{kind}"] = dict(eval_image=drop_img, eval_item=drop_item)
        print(f"  {fam}_train_{kind:7s} -eval_image={drop_img} -eval_item={drop_item} -> kept={len(keep)}",
              flush=True)

# =================================================================================== THE ASSERTION
print("\n[assert] proving train n eval = empty ...", flush=True)
TRAIN_IMG = set(r["imghash"] for v in TRAIN.values() for r in v)
inter = TRAIN_IMG & EVAL_IMG
assert not inter, f"IMAGE LEAK: {len(inter)} decoded-pixel hashes in both train and eval"
print(f"  images: |train|={len(TRAIN_IMG)} |eval|={len(EVAL_IMG)} INTERSECTION={len(inter)}  OK", flush=True)

FAM_OF = {"pmc_vqa_train_mcq": "pmc_vqa", "slake_train_closed": "slake", "slake_train_open": "slake",
          "vqa_rad_train_closed": "vqa_rad", "vqa_rad_train_open": "vqa_rad",
          "pathvqa_train_closed": "path_vqa", "pathvqa_train_open": "path_vqa"}
TRAIN_ITEM = set((FAM_OF[s], qnorm(r["question"]), r["imghash"]) for s, v in TRAIN.items() for r in v)
inter_i = TRAIN_ITEM & EVAL_ITEM
assert not inter_i, f"ITEM LEAK: {len(inter_i)} (family,question,image) triples in both"
print(f"  items:  |train|={len(TRAIN_ITEM)} |eval|={len(EVAL_ITEM)} INTERSECTION={len(inter_i)}  OK",
      flush=True)

# =================================================================================== write
os.makedirs(J(A.manifest_dir), exist_ok=True)
json.dump(TRAIN, open(J(f"{A.manifest_dir}/train_manifest.json"), "w"))
out = dict(
    title="Image-disjoint TRAIN split for GENERATOR (cheap-leg) adaptation of Lingshu-7B (Attack B).",
    date="2026-08-11", seed=A.seed,
    reproduce="python3 src/training_methods/build_cheapleg_train_split.py",
    discipline="md5 of DECODED RGB pixels (WxH header + raw bytes); item identity = "
               "(dataset family, normalized question text, image pixel hash). Same method as "
               "src/training_methods/build_disjoint_verifier_split.py, which proved the clean "
               "verifier's split. The build ASSERTS both intersections are empty and fails otherwise.",
    pmc_vqa_split_landmine="EVAL = /data/dan/dataset/medevalkit/PMC-VQA/test_2.csv (v2, 33,430 rows, "
                           "the MedEvalKit/Lingshu track). TRAIN = train_2.csv (v2, 152,603 rows) from "
                           "the same directory. test_clean.csv (v1, 2,000 rows) is NOT involved.",
    eval_sets=t_report,
    eval_total_distinct_images=len(EVAL_IMG), eval_total_items=len(EVAL_ITEM),
    train_sources={s: dict(kept=len(v), frame=v[0]["frame"] if v else None, dropped=dropped.get(s, {}))
                   for s, v in TRAIN.items()},
    train_total_items=sum(len(v) for v in TRAIN.values()),
    train_total_distinct_images=len(TRAIN_IMG),
    disjointness_assertion=dict(image_pixel_hash_intersection=len(inter),
                                question_item_intersection=len(inter_i),
                                asserted_in_code=["assert not inter", "assert not inter_i"]),
    prompt_frames={"medeval": "MedEvalKit's own prompt, FULL resolution (CAP_MAX_PIXELS unset at eval) "
                              "-- serves PMC_VQA, SLAKE_closed, VQA_RAD_closed, PATH_VQA_closed, MedXpertQA-MM",
                   "openvqa": "run_openvqa.py SYS prompt, cap320 (1280*28*28//4 px) -- serves the 3 open cells"},
    manifest=f"{A.manifest_dir}/train_manifest.json")
os.makedirs(os.path.dirname(J(A.out)), exist_ok=True)
json.dump(out, open(J(A.out), "w"), indent=1)
print(f"\nTRAIN {out['train_total_items']} items / {len(TRAIN_IMG)} images")
print(f"wrote -> {A.out}  and manifest -> {A.manifest_dir}/train_manifest.json")
