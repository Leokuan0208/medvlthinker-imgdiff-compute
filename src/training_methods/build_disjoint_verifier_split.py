#!/usr/bin/env python3
"""build_disjoint_verifier_split.py -- design + PROVE a strictly disjoint train/eval split for the
open-text answer verifier.

WHY.  The deployed verifier (ckpts/train/lora_verifier_pooled4) was trained by a *grouped 70/30 split
of the evaluation sets themselves* (run_lora_verifier_open.py:87-88) and then scored over the FULL
sets, so 67-73% of every reported open-text eval item was seen in training
(results/cascade_methods/artifacts/verifier_validity_2026-07-29.json).  Its reported gain is inflated
~1.8x.  To get an uncontaminated number we need a verifier whose training data shares neither a
QUESTION nor an IMAGE with the three reported eval sets.

DESIGN (option c -- hybrid, chosen; options a/b are quantified in the output artifact).
  eval  (untouched, FULL sets, exactly the items the paper reports):
      slake_open   645 items   (SLAKE test, answer_type=OPEN, q_lang=en)
      vqa_rad_open 200 items   (VQA-RAD test, non-yes/no)
      pathvqa_open 1500 items  (PathVQA test, non-yes/no, the generated sample)
  train (five sources, all provably disjoint from the above):
      slake_open_train    SLAKE *official train* split, OPEN/en       -- 0 eval images
      pathvqa_open_train  PathVQA *official train* split, non-yes/no  -- 0 eval images
      vqa_rad_open_train  VQA-RAD *official train* split, non-yes/no, MINUS every item whose image
                          is an eval image (VQA-RAD recycles images across its own splits: 103/120
                          eval-open images reappear in its train split, so this filter is mandatory)
      kvasir_open, radimagenet_open   already-generated out-of-domain pools (0 overlap by construction)

TWO NESTED DISJOINTNESS LEVELS are emitted, because they answer different questions:
  L1 "image_disjoint"  no eval IMAGE and no eval ITEM in training. Question TEMPLATES may recur with a
                       different image (SLAKE has only 169 distinct question texts for 645 eval items,
                       so template recurrence is unavoidable in-domain). This is the standard clean-split
                       definition and the fair "only the data source changed" comparison.
  L2 "strict"          L1 AND no eval question TEXT at all. Bulletproof against a question->answer-prior
                       shortcut, but it starves the in-domain pool (SLAKE loses ~88% of its train split).
L2 is a subset of L1, so ONE generation pass over L1 serves both.

DISJOINTNESS IS PROVEN, NOT ASSUMED.  Images are compared by a canonical hash of the *decoded RGB
pixels* (not file bytes), so a re-encoded copy of the same image is still caught.

  python3 src/training_methods/build_disjoint_verifier_split.py
  -> results/cascade_methods/artifacts/verifier_disjoint_split.json
  -> data/disjoint_split/idx_{ds}.json         (L1 allowlist -> run_openvqa.py --idx_file)
  -> data/disjoint_split/strict_idx_{ds}.json  (L2 allowlist, subset of L1)
"""
import argparse, glob, hashlib, io, json, os, random, re, string
from collections import defaultdict

from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
SLAKE = "/data/dan/dataset/slake"
PATHVQA = "/data/dan/dataset/path_vqa/data"
VQARAD = "/data/dan/dataset/vqa_rad/data"

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/cascade_methods/artifacts/verifier_disjoint_split.json")
ap.add_argument("--idx_dir", default="data/disjoint_split")
A = ap.parse_args()


def qnorm(s):
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def pixhash(img):
    """Canonical hash of DECODED RGB pixels -> catches re-encoded / re-compressed duplicates."""
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{img.size[0]}x{img.size[1]}|".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def load_slake(split):
    out = []
    for x in json.load(open(f"{SLAKE}/{split}.json")):
        if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en":
            continue
        ip = os.path.join(f"{SLAKE}/imgs", x["img_name"])
        if os.path.exists(ip):
            out.append((x["qid"], x["question"], str(x["answer"]), ip))
    return out


def load_parquet(base, split):
    """row_idx = index in the concatenated sorted split parquets (run_openvqa.py's idx convention)."""
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
        if str(a).strip().lower() in ("yes", "no"):
            continue
        img = r["image"]
        if not (isinstance(img, dict) and "bytes" in img):
            continue
        out.append((int(i), str(q), str(a).strip(), Image.open(io.BytesIO(img["bytes"]))))
    return out


def sc8_idx(ds):
    p = J(f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl")
    return [json.loads(l)["idx"] for l in open(p) if l.strip()]


# --------------------------------------------------------------------------- eval side
print("[eval] hashing the three reported eval sets ...", flush=True)
raw_eval = {"slake_open": load_slake("test"),
            "vqa_rad_open": load_parquet(VQARAD, "test"),
            "pathvqa_open": load_parquet(PATHVQA, "test")}
eval_items, eval_imghash, eval_qtext, eval_qid = {}, {}, set(), set()
eval_itemkey = set()      # (dataset family, normalized question, image pixel hash) = ITEM identity
for ds, pool in raw_eval.items():
    want = set(sc8_idx(ds))
    got = [it for it in pool if it[0] in want]
    assert len(got) == len(want), f"{ds}: sc8 has {len(want)} idx but only {len(got)} resolve"
    eval_items[ds] = got
    eval_imghash[ds] = set(pixhash(it[3]) for it in got)
    eval_qtext |= set(qnorm(it[1]) for it in got)
    eval_qid |= set((ds, it[0]) for it in got)
    eval_itemkey |= set((ds.replace("_open", ""), qnorm(it[1]), pixhash(it[3])) for it in got)
    print(f"  {ds:14s} items={len(got):5d}  images={len(eval_imghash[ds]):5d}  "
          f"distinct question texts={len(set(qnorm(it[1]) for it in got)):5d}", flush=True)
EVAL_IMG = set().union(*eval_imghash.values())
print(f"  EVAL total {sum(len(v) for v in eval_items.values())} items / {len(EVAL_IMG)} images / "
      f"{len(eval_qtext)} question texts", flush=True)
for a in eval_imghash:
    for b in eval_imghash:
        if a < b and eval_imghash[a] & eval_imghash[b]:
            print(f"  NOTE eval-vs-eval image overlap {a}&{b}: {len(eval_imghash[a] & eval_imghash[b])}")

# --------------------------------------------------------------------------- option (a) feasibility
print("\n[option a] image-disjoint re-partition WITHIN each eval set (70/30 by image) -- REJECTED:", flush=True)
opt_a = {}
for ds in eval_items:
    by_img = defaultdict(list)
    for it in eval_items[ds]:
        by_img[pixhash(it[3])].append(it[0])
    imgs = sorted(by_img)
    random.Random(A.seed).shuffle(imgs)
    ntr = int(0.7 * len(imgs))
    n_tr = sum(len(by_img[i]) for i in imgs[:ntr])
    n_te = sum(len(by_img[i]) for i in imgs[ntr:])
    opt_a[ds] = {"n_images": len(imgs), "n_train_images": ntr, "n_eval_images": len(imgs) - ntr,
                 "n_train_items": n_tr, "n_surviving_eval_items": n_te,
                 "pct_eval_lost": round(100.0 * (1 - n_te / len(eval_items[ds])), 1)}
    print(f"  {ds:14s} {len(imgs):4d} imgs -> train {ntr:4d}/{n_tr:5d} items | EVAL SURVIVES "
          f"{len(imgs)-ntr:4d} imgs/{n_te:5d} items ({opt_a[ds]['pct_eval_lost']}% of eval lost)", flush=True)

# --------------------------------------------------------------------------- train side
print("\n[train] hashing candidate training pools (official TRAIN splits) ...", flush=True)
pools = {"slake_open_train": load_slake("train"),
         "vqa_rad_open_train": load_parquet(VQARAD, "train"),
         "pathvqa_open_train": load_parquet(PATHVQA, "train")}
split = {}
for name, pool in pools.items():
    hs = [(it, pixhash(it[3])) for it in pool]
    l1 = [(it, h) for it, h in hs if h not in EVAL_IMG]
    l2 = [(it, h) for it, h in l1 if qnorm(it[1]) not in eval_qtext]
    fam = name.replace("_open_train", "")
    split[name] = {"pool_items": len(hs), "dropped_eval_image": len(hs) - len(l1),
                   "L1_image_disjoint_items": len(l1), "L1_images": len(set(h for _, h in l1)),
                   "L2_strict_items": len(l2), "L2_images": len(set(h for _, h in l2)),
                   "dropped_eval_question_text": len(l1) - len(l2),
                   "idx_L1": [it[0] for it, _ in l1], "idx_L2": [it[0] for it, _ in l2],
                   "imghash": set(h for _, h in l1),
                   "itemkey": set((fam, qnorm(it[1]), h) for it, h in l1),
                   "qid": set((name, it[0]) for it, _ in l1)}
    s = split[name]
    print(f"  {name:20s} pool={s['pool_items']:6d} -eval-image={s['dropped_eval_image']:5d} -> "
          f"L1={s['L1_image_disjoint_items']:6d} ({s['L1_images']} imgs); "
          f"-eval-qtext={s['dropped_eval_question_text']:5d} -> L2={s['L2_strict_items']:6d} "
          f"({s['L2_images']} imgs)", flush=True)

for name, jp in [("kvasir_open", "/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"),
                 ("radimagenet_open", "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")]:
    rows = [r for r in json.load(open(jp)) if os.path.exists(r["img_path"])]
    hs = {r["idx"]: pixhash(r["img_path"]) for r in rows}
    l1 = [r for r in rows if hs[r["idx"]] not in EVAL_IMG]
    l2 = [r for r in l1 if qnorm(r["question"]) not in eval_qtext]
    split[name] = {"pool_items": len(rows), "dropped_eval_image": len(rows) - len(l1),
                   "L1_image_disjoint_items": len(l1), "L1_images": len(set(hs[r["idx"]] for r in l1)),
                   "L2_strict_items": len(l2), "L2_images": len(set(hs[r["idx"]] for r in l2)),
                   "dropped_eval_question_text": len(l1) - len(l2),
                   "idx_L1": [r["idx"] for r in l1], "idx_L2": [r["idx"] for r in l2],
                   "imghash": set(hs[r["idx"]] for r in l1), "already_generated": True,
                   "itemkey": set((name, qnorm(r["question"]), hs[r["idx"]]) for r in l1),
                   "qid": set((name, r["idx"]) for r in l1)}
    s = split[name]
    print(f"  {name:20s} pool={s['pool_items']:6d} -eval-image={s['dropped_eval_image']:5d} -> "
          f"L1={s['L1_image_disjoint_items']:6d}; -eval-qtext={s['dropped_eval_question_text']:5d} -> "
          f"L2={s['L2_strict_items']:6d}", flush=True)

# --------------------------------------------------------------------------- THE ASSERTION
print("\n[assert] proving train n eval = empty ...", flush=True)
TRAIN_IMG = set().union(*[v["imghash"] for v in split.values()])
inter_img = TRAIN_IMG & EVAL_IMG
assert not inter_img, f"IMAGE LEAK: {len(inter_img)} decoded-pixel hashes in both train and eval"
print(f"  L1 images: |train|={len(TRAIN_IMG)} |eval|={len(EVAL_IMG)} INTERSECTION={len(inter_img)}  OK")

# QUESTION identity. Raw ids are not comparable across split files (PathVQA train row 0 and test row 0
# are different items), so the binding question-level assertion is on ITEM identity:
# (dataset family, normalized question text, image pixel hash).
TRAIN_ITEM = set().union(*[v["itemkey"] for v in split.values()])
inter_item = TRAIN_ITEM & eval_itemkey
assert not inter_item, f"ITEM LEAK: {len(inter_item)} (family, question, image) triples in both"
print(f"  L1 question items (family,question,image): |train|={len(TRAIN_ITEM)} |eval|={len(eval_itemkey)} "
      f"INTERSECTION={len(inter_item)}  OK")

TRAIN_QID = set().union(*[v["qid"] for v in split.values()])
inter_qid = TRAIN_QID & eval_qid
assert not inter_qid, f"QID LEAK: {len(inter_qid)} (source, question-id) pairs in both"
print(f"  L1 question ids (source,id): |train|={len(TRAIN_QID)} |eval|={len(eval_qid)} "
      f"INTERSECTION={len(inter_qid)}  OK")
raw_id_collisions = {name: len({i for _, i in v["qid"]} & {i for ds, i in eval_qid
                                                          if ds.replace("_open", "") == name.replace("_open_train", "")})
                     for name, v in split.items()}
print(f"  (raw numeric id coincidences within the same dataset family, DIFFERENT split files so NOT the "
      f"same item -- image+item assertions above are the binding proof): {raw_id_collisions})")

q_L1, q_L2 = set(), set()
for name, pool in pools.items():
    s1, s2 = set(split[name]["idx_L1"]), set(split[name]["idx_L2"])
    q_L1 |= set(qnorm(it[1]) for it in pool if it[0] in s1)
    q_L2 |= set(qnorm(it[1]) for it in pool if it[0] in s2)
for name, jp in [("kvasir_open", "/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"),
                 ("radimagenet_open", "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")]:
    s1, s2 = set(split[name]["idx_L1"]), set(split[name]["idx_L2"])
    rows = json.load(open(jp))
    q_L1 |= set(qnorm(r["question"]) for r in rows if r["idx"] in s1)
    q_L2 |= set(qnorm(r["question"]) for r in rows if r["idx"] in s2)
assert not (q_L2 & eval_qtext), f"L2 QUESTION-TEXT LEAK: {len(q_L2 & eval_qtext)}"
print(f"  L2 question texts: |train|={len(q_L2)} |eval|={len(eval_qtext)} INTERSECTION=0  OK")
print(f"  (L1 shares {len(q_L1 & eval_qtext)} question TEXTS with eval by design -- different images, "
      f"never the same item; L2 removes even those)")

# --------------------------------------------------------------------------- write
os.makedirs(J(A.idx_dir), exist_ok=True)
for name, v in split.items():
    # BOTH levels are written for EVERY source: the trainer allowlists every source, while
    # `needs_generation` (below) is what decides which ones still need candidate generation.
    json.dump(sorted(v["idx_L1"], key=str), open(J(f"{A.idx_dir}/idx_{name}.json"), "w"))
    json.dump(sorted(v["idx_L2"], key=str), open(J(f"{A.idx_dir}/strict_idx_{name}.json"), "w"))

out = {
    "design": "option_c_hybrid: in-domain OFFICIAL TRAIN splits (image-disjoint) + 2 out-of-domain pools",
    "levels": {"L1_image_disjoint": "no eval image, no eval item; question TEMPLATES may recur",
               "L2_strict": "L1 and no eval question TEXT at all (subset of L1)"},
    "seed": A.seed,
    "eval": {ds: {"n_items": len(eval_items[ds]), "n_images": len(eval_imghash[ds]),
                  "n_distinct_question_texts": len(set(qnorm(it[1]) for it in eval_items[ds]))}
             for ds in eval_items},
    "eval_total_items": sum(len(v) for v in eval_items.values()),
    "eval_total_images": len(EVAL_IMG),
    "option_a_feasibility_rejected": opt_a,
    "train": {k: {kk: vv for kk, vv in v.items()
                  if kk not in ("idx_L1", "idx_L2", "imghash", "itemkey", "qid")}
              for k, v in split.items()},
    "train_total_L1_items": sum(v["L1_image_disjoint_items"] for v in split.values()),
    "train_total_L2_items": sum(v["L2_strict_items"] for v in split.values()),
    "train_total_L1_images": len(TRAIN_IMG),
    "disjointness_assertion": {
        "image_pixel_hash_intersection": len(inter_img),
        "question_item_intersection": len(inter_item),
        "question_id_intersection": len(inter_qid),
        "L2_question_text_intersection": 0,
        "L1_question_text_shared_by_design": len(q_L1 & eval_qtext),
        "raw_numeric_id_coincidences_different_split_files_not_same_item": raw_id_collisions,
        "n_train_images": len(TRAIN_IMG), "n_eval_images": len(EVAL_IMG),
        "n_train_question_items": len(TRAIN_ITEM), "n_eval_question_items": len(eval_itemkey),
        "method": "images: md5 of decoded RGB pixels (WxH + raw bytes), catches re-encoded copies. "
                  "questions: item identity = (dataset family, normalized question text, image pixel hash), "
                  "because raw row ids are not comparable across different split files.",
        "asserted_in_code": ["assert not inter_img", "assert not inter_item", "assert not inter_qid",
                             "assert not (q_L2 & eval_qtext)"],
    },
    "needs_generation": [k for k, v in split.items() if not v.get("already_generated")],
}
os.makedirs(os.path.dirname(J(A.out)), exist_ok=True)
json.dump(out, open(J(A.out), "w"), indent=1)
print(f"\nTRAIN L1 {out['train_total_L1_items']} items / {out['train_total_L1_images']} images | "
      f"L2 {out['train_total_L2_items']} items")
print(f"EVAL     {out['eval_total_items']} items / {out['eval_total_images']} images (FULL, nothing lost)")
print(f"wrote -> {A.out} and allowlists -> {A.idx_dir}/")
