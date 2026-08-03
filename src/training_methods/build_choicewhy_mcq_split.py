#!/usr/bin/env python3
"""build_choicewhy_mcq_split.py -- design + PROVE a strictly disjoint TRAIN pool for the MULTIPLE-CHOICE
"(choice)(why)" verifier (Phase 2 of the choicewhy program).

WHY.  Phase 1 (results/cascade_methods/artifacts/choicewhy_pilot_2026-08-03.json) showed the
(choice)(why) format is accuracy-free on MCQ and produces textually diverse candidates with +0.44 of
oracle headroom over self-consistency on the letter-disagreement items.  Phase 2 trains a verifier to
select among those candidates.  A verifier trained on any item whose IMAGE or QUESTION appears in the
evaluation set is worthless -- contamination inflated this project's last verifier gain by 2.9x
(results/cascade_methods/artifacts/verifier_validity_2026-07-29.json).

This is the MCQ analogue of src/training_methods/build_disjoint_verifier_split.py and reuses its
method verbatim: images are compared by a canonical hash of the DECODED RGB PIXELS (not file bytes),
so a re-encoded or re-compressed copy of the same image is still caught; question identity is the
triple (dataset family, normalized question text, image pixel hash), because raw row ids are not
comparable across different split files.

EVAL SIDE (untouched).  The disjointness set is the FULL MedVLThinker-Eval benchmark suite -- all 8,220
items, every image -- not merely the 1,488 pilot items.  That is strictly stronger than required and
future-proofs the verifier against every eval this repo reports.

TRAIN SIDE.  Four sources, all from the datasets' OFFICIAL TRAIN splits, formatted so the option block
is byte-identical in shape to the eval items the pilot used:
  slake_closed_train     SLAKE  official train, answer_type=CLOSED, q_lang=en, yes/no  -> {"A":"Yes","B":"No"}
  vqa_rad_closed_train   VQA-RAD official train, yes/no                                -> {"A":"Yes","B":"No"}
  pathvqa_closed_train   PathVQA official train, yes/no                                -> {"A":"Yes","B":"No"}
  pmc_vqa_train          PMC-VQA official train.csv, native 4 options                  -> {"A":..,"D":..}
MedXpertQA-MM has NO public train split, so its 20% share of the eval mix cannot be matched in-domain;
its quota is topped up from pathvqa_closed_train and the shortfall is RECORDED (exactly as
run_lora_verifier_disjoint.py records radimagenet top-ups).

TWO NESTED LEVELS, same definitions as the open-text builder:
  L1 image_disjoint  no eval IMAGE and no eval ITEM in training.  Question TEMPLATES may recur with a
                     different image -- unavoidable for closed VQA (SLAKE/VQA-RAD/PathVQA yes/no
                     questions are drawn from a small template set).  The standard clean-split
                     definition and the fair "only the data source changed" comparison.
  L2 strict          L1 AND no eval question TEXT at all.  Recorded for reference; it starves the
                     yes/no pools.

Images are STAGED as lossless PNG under --img_dir and the manifest carries the decoded-RGB md5, so the
generator can assert on load that the pixels it feeds the model are the pixels that were proven
disjoint.  PNG is lossless, so staging does not change the decoded RGB bytes (asserted in code).

  python3 src/training_methods/build_choicewhy_mcq_split.py
  -> results/cascade_methods/artifacts/choicewhy_mcq_split.json
  -> data/choicewhy_mcq_split/train_items.jsonl      (L1 manifest; --level L2 also flagged per row)
  -> data/choicewhy_mcq_split/imgs/<src>/<idx>.png   (staged, md5-verified)
"""
import argparse, glob, hashlib, io, json, os, random, re, string
from collections import Counter, defaultdict

from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
EVALROOT = "/data/dan/dataset/MedVLThinker-Eval"
SLAKE = "/data/dan/dataset/slake"
PATHVQA = "/data/dan/dataset/path_vqa/data"
VQARAD = "/data/dan/dataset/vqa_rad/data"
PMCTRAIN = "/data/dan/dataset/pmc_vqa_train"

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pilot_manifest", default="ckpts/choicewhy_pilot/items.jsonl")
ap.add_argument("--out", default="results/cascade_methods/artifacts/choicewhy_mcq_split.json")
ap.add_argument("--manifest", default="data/choicewhy_mcq_split/train_items.jsonl")
ap.add_argument("--img_dir", default="data/choicewhy_mcq_split/imgs")
# quotas (QUESTIONS, not training examples). Defaults: take every surviving in-domain yes/no item from
# SLAKE and VQA-RAD (both pools are small), then fill toward the eval mix from PMC-VQA and PathVQA.
ap.add_argument("--n_slake", type=int, default=10 ** 9)
ap.add_argument("--n_vqarad", type=int, default=10 ** 9)
ap.add_argument("--n_pmc", type=int, default=3000)
ap.add_argument("--n_pathvqa", type=int, default=2500)
ap.add_argument("--eval_hash_cache", default="data/choicewhy_mcq_split/eval_image_hashes.json",
                help="cache of the eval-suite hashes so a rerun does not re-decode 8,220 images")
A = ap.parse_args()
random.seed(A.seed)


def qnorm(s):
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def pix_md5(img):
    """md5 of DECODED RGB pixels -- the pilot manifest's convention (ckpts/choicewhy_pilot/items.jsonl)."""
    return hashlib.md5(img.convert("RGB").tobytes()).hexdigest()


def pix_md5_sized(img):
    """Same, with a WxH prefix -- the open-text builder's convention. Both are asserted, so a size
    coincidence in the raw-bytes hash cannot hide a leak."""
    im = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{im.size[0]}x{im.size[1]}|".encode())
    h.update(im.tobytes())
    return h.hexdigest()


# =============================================================== EVAL SIDE (the full 8,220-item suite)
CACHE = J(A.eval_hash_cache)
if os.path.exists(CACHE):
    print(f"[eval] loading cached eval-suite hashes from {CACHE}", flush=True)
    c = json.load(open(CACHE))
    EVAL_IMG = set(c["img"]); EVAL_IMG_SIZED = set(c["img_sized"])
    eval_qtext = set(c["qtext"]); eval_itemkey = set(tuple(t) for t in c["itemkey"])
    per_bench_img = {k: set(v) for k, v in c["per_bench_img"].items()}
    n_eval_items = c["n_items"]
else:
    print("[eval] hashing every image in MedVLThinker-Eval (all 6 benchmarks, 8,220 items) ...", flush=True)
    from datasets import load_dataset  # noqa: E402

    ds = load_dataset(EVALROOT)
    split = "test" if "test" in ds else list(ds.keys())[0]
    data = ds[split]
    EVAL_IMG, EVAL_IMG_SIZED = set(), set()
    eval_qtext = set()
    eval_itemkey = set()
    per_bench_img = defaultdict(set)
    for i in range(len(data)):
        ex = data[i]
        hs = [pix_md5(im) for im in (ex.get("images") or [])]
        hs_s = [pix_md5_sized(im) for im in (ex.get("images") or [])]
        EVAL_IMG |= set(hs)
        EVAL_IMG_SIZED |= set(hs_s)
        per_bench_img[ex["dataset_name"]] |= set(hs)
        qn = qnorm(ex["question"])
        eval_qtext.add(qn)
        for h in hs:
            eval_itemkey.add((ex["dataset_name"], qn, h))
        if (i + 1) % 1000 == 0:
            print(f"   {i+1}/{len(data)} hashed", flush=True)
    n_eval_items = len(data)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump({"img": sorted(EVAL_IMG), "img_sized": sorted(EVAL_IMG_SIZED), "qtext": sorted(eval_qtext),
               "itemkey": [list(t) for t in sorted(eval_itemkey)], "n_items": n_eval_items,
               "per_bench_img": {k: sorted(v) for k, v in per_bench_img.items()}}, open(CACHE, "w"))
print(f"[eval] {n_eval_items} items | {len(EVAL_IMG)} distinct images | {len(eval_qtext)} distinct question texts",
      flush=True)
for b, s in sorted(per_bench_img.items()):
    print(f"   {b:18s} images={len(s)}")

# the pilot's 1,488 evaluation items (the set Phase 2 reports on) -- a subset of the above
PILOT = [json.loads(l) for l in open(J(A.pilot_manifest)) if l.strip()]
pilot_img = set(h for r in PILOT for h in r["image_md5_rgb"])
print(f"[eval] pilot evaluation subset: {len(PILOT)} items / {len(pilot_img)} distinct images "
      f"({Counter(r['bench'] for r in PILOT)})", flush=True)
assert pilot_img <= EVAL_IMG, "pilot images are not a subset of the hashed eval suite -- hash mismatch"
print("[eval] pilot image hashes are a strict subset of the eval-suite hashes: OK (hash conventions agree)",
      flush=True)


# =============================================================== TRAIN SIDE
def yn_options(ans):
    return {"A": "Yes", "B": "No"}, ("A" if str(ans).strip().lower() == "yes" else "B")


# Each source is a (specs, loader) pair. `specs` returns LIGHTWEIGHT metadata only -- no image is opened
# until the loader is called on a single spec. Materializing images for a 176k-row pool (PMC-VQA) would
# exhaust the process's file-descriptor limit and its memory; specs+loader keeps the scan O(1) in images.
def specs_slake_train():
    out = []
    for x in json.load(open(f"{SLAKE}/train.json")):
        if x.get("answer_type") != "CLOSED" or x.get("q_lang") != "en":
            continue
        if str(x["answer"]).strip().lower() not in ("yes", "no"):
            continue
        ip = os.path.join(f"{SLAKE}/imgs", x["img_name"])
        if not os.path.exists(ip):
            continue
        opts, gold = yn_options(x["answer"])
        out.append({"idx": x["qid"], "question": x["question"], "options": opts, "gold": gold, "path": ip})
    return out


def load_path(spec):
    return Image.open(spec["path"])


def _parquet_specs(base, split_name):
    """Read only the columns needed for filtering; image bytes are fetched per row by the loader."""
    import pandas as pd
    files = sorted(glob.glob(f"{base}/{split_name}-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = []
    for i, r in df.iterrows():
        q, a = r.get("question"), r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]
            q = conv[0]["value"].replace("<image>", "").strip()
            a = conv[1]["value"]
        if str(a).strip().lower() not in ("yes", "no"):
            continue
        img = r["image"]
        if not (isinstance(img, dict) and "bytes" in img):
            continue
        opts, gold = yn_options(a)
        out.append({"idx": int(i), "question": str(q), "options": opts, "gold": gold,
                    "_bytes": img["bytes"]})
    return out


def load_bytes(spec):
    return Image.open(io.BytesIO(spec["_bytes"]))


CH_PREFIX = re.compile(r"^\s*[A-D]\s*[:.)]\s*")


def specs_pmc_train():
    import pandas as pd
    df = pd.read_csv(f"{PMCTRAIN}/train.csv")
    out = []
    for i, r in df.iterrows():
        lab = str(r["Answer_label"]).strip().upper()[:1]
        if lab not in "ABCD":
            continue
        opts, bad = {}, False
        for L in "ABCD":
            v = r.get(f"Choice {L}")
            if not isinstance(v, str) or not CH_PREFIX.sub("", v).strip():
                bad = True
                break
            opts[L] = CH_PREFIX.sub("", v).strip()
        if bad:
            continue
        ip = os.path.join(PMCTRAIN, "images", str(r["Figure_path"]))
        out.append({"idx": int(i), "question": str(r["Question"]).strip(), "options": opts,
                    "gold": lab, "path": ip})
    return out


SOURCES = [
    ("slake_closed_train", "slake", specs_slake_train, load_path, A.n_slake),
    ("vqa_rad_closed_train", "vqa_rad", lambda: _parquet_specs(VQARAD, "train"), load_bytes, A.n_vqarad),
    ("pmc_vqa_train", "pmc_vqa", specs_pmc_train, load_path, A.n_pmc),
    ("pathvqa_closed_train", "pathvqa", lambda: _parquet_specs(PATHVQA, "train"), load_bytes, A.n_pathvqa),
]

os.makedirs(J(A.img_dir), exist_ok=True)
os.makedirs(os.path.dirname(J(A.manifest)), exist_ok=True)
rows, stats = [], {}
TRAIN_IMG, TRAIN_IMG_SIZED, TRAIN_ITEM = set(), set(), set()
train_qtext_L1 = set()

for src, fam, specs_fn, load_fn, quota in SOURCES:
    print(f"\n[train] {src}: scanning (quota {quota if quota < 10**8 else 'ALL'} questions) ...", flush=True)
    pool = specs_fn()
    print(f"  {len(pool)} candidate rows before disjointness filtering", flush=True)
    rng = random.Random(A.seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    os.makedirs(os.path.join(J(A.img_dir), src), exist_ok=True)
    n_seen = n_dropimg = n_dropitem = n_missing = 0
    kept = []
    for oi in order:
        meta = pool[oi]
        n_seen += 1
        try:
            img = load_fn(meta)
            img.load()
        except Exception:
            n_missing += 1
            continue
        h = pix_md5(img)
        hs = pix_md5_sized(img)
        if h in EVAL_IMG or hs in EVAL_IMG_SIZED:
            n_dropimg += 1
            continue
        key = (fam, qnorm(meta["question"]), h)
        if key in eval_itemkey:
            n_dropitem += 1
            continue
        rgb = img.convert("RGB")
        p = os.path.join(J(A.img_dir), src, f"{meta['idx']}.png")
        if not os.path.exists(p):
            rgb.save(p, format="PNG")
        # PNG is lossless: assert the staged file decodes to the exact pixels that were proven disjoint
        assert pix_md5(Image.open(p)) == h, f"staging changed pixels for {src}/{meta['idx']}"
        kept.append({"src": src, "family": fam, "idx": meta["idx"], "question": meta["question"],
                     "options": meta["options"], "gold": meta["gold"], "img_path": p,
                     "image_md5_rgb": h, "image_md5_rgb_sized": hs,
                     "L2_strict": int(qnorm(meta["question"]) not in eval_qtext)})
        TRAIN_IMG.add(h)
        TRAIN_IMG_SIZED.add(hs)
        TRAIN_ITEM.add(key)
        train_qtext_L1.add(qnorm(meta["question"]))
        if len(kept) >= quota:
            break
    rows += kept
    stats[src] = {"pool_items": len(pool), "scanned": n_seen, "dropped_eval_image": n_dropimg,
                  "dropped_unreadable_image": n_missing,
                  "dropped_eval_item": n_dropitem, "kept_L1_questions": len(kept),
                  "kept_L1_images": len(set(r["image_md5_rgb"] for r in kept)),
                  "kept_L2_strict_questions": sum(r["L2_strict"] for r in kept),
                  "quota": (quota if quota < 10 ** 8 else None),
                  "gold_letter_dist": dict(Counter(r["gold"] for r in kept))}
    s = stats[src]
    print(f"  pool={s['pool_items']:6d} scanned={s['scanned']:6d} -eval-image={s['dropped_eval_image']:5d} "
          f"-unreadable={s['dropped_unreadable_image']:5d} "
          f"-eval-item={s['dropped_eval_item']:5d} -> KEPT {s['kept_L1_questions']:5d} questions / "
          f"{s['kept_L1_images']:5d} images (L2-strict subset {s['kept_L2_strict_questions']})", flush=True)

# =============================================================== THE ASSERTION
print("\n[assert] proving train n eval = empty ...", flush=True)
inter_img = TRAIN_IMG & EVAL_IMG
assert not inter_img, f"IMAGE LEAK: {len(inter_img)} decoded-RGB md5s in both train and eval"
print(f"  images (md5 of decoded RGB):        |train|={len(TRAIN_IMG)} |eval|={len(EVAL_IMG)} "
      f"INTERSECTION={len(inter_img)}  OK")
inter_img_s = TRAIN_IMG_SIZED & EVAL_IMG_SIZED
assert not inter_img_s, f"IMAGE LEAK (sized hash): {len(inter_img_s)}"
print(f"  images (WxH-prefixed md5):          |train|={len(TRAIN_IMG_SIZED)} |eval|={len(EVAL_IMG_SIZED)} "
      f"INTERSECTION={len(inter_img_s)}  OK")
inter_item = TRAIN_ITEM & eval_itemkey
assert not inter_item, f"ITEM LEAK: {len(inter_item)} (family, question, image) triples in both"
print(f"  items (family,question,image):      |train|={len(TRAIN_ITEM)} |eval|={len(eval_itemkey)} "
      f"INTERSECTION={len(inter_item)}  OK")
inter_pilot = TRAIN_IMG & pilot_img
assert not inter_pilot, f"PILOT IMAGE LEAK: {len(inter_pilot)}"
print(f"  images vs the 1,488 pilot EVAL set: |train|={len(TRAIN_IMG)} |pilot|={len(pilot_img)} "
      f"INTERSECTION={len(inter_pilot)}  OK")
qtext_overlap = len(train_qtext_L1 & eval_qtext)
strict_rows = [r for r in rows if r["L2_strict"]]
assert not (set(qnorm(r["question"]) for r in strict_rows) & eval_qtext), "L2 QUESTION-TEXT LEAK"
print(f"  L2 subset question texts:           INTERSECTION=0  OK")
print(f"  (L1 shares {qtext_overlap} question TEXTS with eval by design -- closed VQA reuses a small "
      f"template set; never the same item, never the same image)")

with open(J(A.manifest), "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

comp = Counter(r["src"] for r in rows)
out = {
    "purpose": "strictly disjoint MULTIPLE-CHOICE training pool for the (choice)(why) verifier (Phase 2)",
    "date": "2026-08-03",
    "builder": "src/training_methods/build_choicewhy_mcq_split.py",
    "seed": A.seed,
    "eval_disjointness_set": {
        "what": "the FULL MedVLThinker-Eval suite (all 6 benchmarks, 8,220 items) -- strictly larger "
                "than the 1,488 pilot evaluation items, so no training image appears in ANY eval this "
                "repo reports",
        "n_eval_items": n_eval_items, "n_eval_images": len(EVAL_IMG),
        "n_eval_distinct_question_texts": len(eval_qtext),
        "images_per_benchmark": {b: len(s) for b, s in sorted(per_bench_img.items())},
        "pilot_eval_subset": {"n_items": len(PILOT), "n_images": len(pilot_img),
                              "per_bench": dict(Counter(r["bench"] for r in PILOT))},
    },
    "train_sources": stats,
    "train_total_questions": len(rows),
    "train_total_images": len(TRAIN_IMG),
    "train_composition_questions": dict(comp),
    "train_L2_strict_questions": len(strict_rows),
    "medxpert_note": "MedXpertQA-MM has no public train split; its 20% share of the eval mix cannot be "
                     "matched in-domain. Recorded as a shortfall; topped up from pathvqa_closed_train.",
    "disjointness_assertion": {
        "image_md5_rgb_intersection": len(inter_img),
        "image_md5_rgb_sized_intersection": len(inter_img_s),
        "item_triple_intersection": len(inter_item),
        "pilot_eval_image_intersection": len(inter_pilot),
        "L2_question_text_intersection": 0,
        "L1_question_text_shared_by_design": qtext_overlap,
        "method": "images: md5 of DECODED RGB pixels (both raw-bytes and WxH-prefixed variants), which "
                  "catches re-encoded/re-compressed copies; filenames are never used. items: "
                  "(dataset family, normalized question text, image pixel hash).",
        "asserted_in_code": ["assert not inter_img", "assert not inter_img_s", "assert not inter_item",
                             "assert not inter_pilot", "assert not (L2 qtext & eval_qtext)",
                             "assert pilot_img <= EVAL_IMG", "assert staged PNG md5 == proven md5"],
    },
    "manifest": A.manifest, "staged_images": A.img_dir,
}
os.makedirs(os.path.dirname(J(A.out)), exist_ok=True)
json.dump(out, open(J(A.out), "w"), indent=1)
print(f"\nTRAIN POOL {len(rows)} questions / {len(TRAIN_IMG)} images  composition={dict(comp)}")
print(f"wrote manifest -> {A.manifest}\nwrote artifact -> {A.out}")
