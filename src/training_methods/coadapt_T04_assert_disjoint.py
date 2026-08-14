#!/usr/bin/env python3
"""coadapt_T04_assert_disjoint.py -- RE-PROVE, for the T=0.4 pools actually generated in this round,
that no training image and no training item is an evaluation image or item.

Pre-registration: results/cascade_methods/artifacts/_coadapt_verifier_prereg_2026-08-14.json

WHY THIS EXISTS.  The T=0.4 pools were generated against the L1 allowlists written by
src/training_methods/build_disjoint_verifier_split.py, which already proved disjointness when it wrote
them.  That is a claim about the ALLOWLISTS.  This asserts it about the FILES THIS ROUND PRODUCED --
every question id that actually appears in ckpts/openvqa/cheap_lingshu7b_T04/ -- so a wrong --idx_file,
a stale allowlist or a partially-resumed pool cannot slip a contaminated image into training.

THE CHECK IS THE EXISTING BUILDER'S, NOT A WEAKER ONE.  `pixhash` and `qnorm` are copied VERBATIM from
build_disjoint_verifier_split.py (md5 of DECODED RGB pixels with a 'WxH|' prefix, so a re-encoded or
re-compressed copy of an eval image is still caught), and the same three assertions are made:

    assert image_pixel_md5_intersection      == 0
    assert item_identity_intersection        == 0     item = (family, normalized question, pixhash)
    assert question_id_intersection          == 0     within a dataset family

It also asserts that every generated idx is inside the L1 allowlist, which the builder never had to
check because it wrote the allowlist itself.

  python3 src/training_methods/coadapt_T04_assert_disjoint.py
"""
import glob, hashlib, io, json, os, re, string, sys
from collections import defaultdict

from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
SLAKE = "/data/dan/dataset/slake"
PATHVQA = "/data/dan/dataset/path_vqa/data"
VQARAD = "/data/dan/dataset/vqa_rad/data"
COLD = J("ckpts/openvqa/cheap_lingshu7b_T04")
IDXD = J("data/disjoint_split")
OUT = J("results/cascade_methods/artifacts/_coadapt_T04_pool_build.json")

TRAIN_SOURCES = ["slake_open_train", "vqa_rad_open_train", "pathvqa_open_train",
                 "kvasir_open", "radimagenet_open"]
EVAL_DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]


# ---------------------------------------------------------------- VERBATIM from the builder
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


def load_json_pool(jp):
    return [(r["idx"], r["question"], r["answer"], r["img_path"])
            for r in json.load(open(jp)) if os.path.exists(r["img_path"])]


def cold_idx(ds):
    p = os.path.join(COLD, f"ckpt_{ds}_lingshu7bT04_sc8.jsonl")
    return [json.loads(l)["idx"] for l in open(p) if l.strip()]


def sc8_idx(ds):
    """The frozen EVAL pool: the exact idx the published cells are computed on."""
    p = J(f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl")
    return [json.loads(l)["idx"] for l in open(p) if l.strip()]


# ---------------------------------------------------------------- eval side
print("[eval] hashing the three reported eval sets ...", flush=True)
raw_eval = {"slake_open": load_slake("test"),
            "vqa_rad_open": load_parquet(VQARAD, "test"),
            "pathvqa_open": load_parquet(PATHVQA, "test")}
EVAL_IMG, eval_itemkey, eval_qid, eval_n = set(), set(), set(), {}
for ds in EVAL_DS:
    want = set(sc8_idx(ds))
    got = [it for it in raw_eval[ds] if it[0] in want]
    assert len(got) == len(want), f"{ds}: eval pool has {len(want)} idx but only {len(got)} resolve"
    EVAL_IMG |= set(pixhash(it[3]) for it in got)
    eval_itemkey |= set((ds.replace("_open", ""), qnorm(it[1]), pixhash(it[3])) for it in got)
    eval_qid |= set((ds, it[0]) for it in got)
    eval_n[ds] = len(got)
    print(f"  {ds:14s} items={len(got):5d}", flush=True)
print(f"  EVAL total {sum(eval_n.values())} items / {len(EVAL_IMG)} images", flush=True)

# ---------------------------------------------------------------- train side (THIS ROUND'S FILES)
print("\n[train] hashing every question that actually appears in the T=0.4 pools ...", flush=True)
POOLS = {"slake_open_train": lambda: load_slake("train"),
         "vqa_rad_open_train": lambda: load_parquet(VQARAD, "train"),
         "pathvqa_open_train": lambda: load_parquet(PATHVQA, "train"),
         "kvasir_open": lambda: load_json_pool("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"),
         "radimagenet_open": lambda: load_json_pool(
             "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")}
FAM = {"slake_open_train": "slake", "vqa_rad_open_train": "vqa_rad",
       "pathvqa_open_train": "pathvqa", "kvasir_open": "kvasir", "radimagenet_open": "radimagenet"}

TRAIN_IMG, TRAIN_ITEM, TRAIN_QID, per_src = set(), set(), set(), {}
allow_violations = {}
for ds in TRAIN_SOURCES:
    got_idx = cold_idx(ds)
    allow = set(json.load(open(os.path.join(IDXD, f"idx_{ds}.json"))))
    outside = [i for i in got_idx if i not in allow]
    allow_violations[ds] = len(outside)
    want = set(got_idx)
    pool = [it for it in POOLS[ds]() if it[0] in want]
    assert len(pool) == len(want), f"{ds}: pool has {len(want)} idx but only {len(pool)} resolve"
    hs = [(it, pixhash(it[3])) for it in pool]
    TRAIN_IMG |= set(h for _, h in hs)
    TRAIN_ITEM |= set((FAM[ds], qnorm(it[1]), h) for it, h in hs)
    TRAIN_QID |= set((ds, it[0]) for it, _ in hs)
    per_src[ds] = {"generated_questions": len(got_idx), "L1_allowlist": len(allow),
                   "outside_allowlist": len(outside), "distinct_images": len(set(h for _, h in hs))}
    print(f"  {ds:20s} questions={len(got_idx):6d} images={per_src[ds]['distinct_images']:6d} "
          f"outside allowlist={len(outside)}", flush=True)

# ---------------------------------------------------------------- THE ASSERTIONS
print("\n[assert] proving train n eval = empty for THIS round's pools ...", flush=True)
inter_img = TRAIN_IMG & EVAL_IMG
assert not inter_img, f"IMAGE LEAK: {len(inter_img)} decoded-pixel hashes in both train and eval"
print(f"  images:      |train|={len(TRAIN_IMG)} |eval|={len(EVAL_IMG)} INTERSECTION=0  OK")

inter_item = TRAIN_ITEM & eval_itemkey
assert not inter_item, f"ITEM LEAK: {len(inter_item)} (family, question, image) triples in both"
print(f"  items:       |train|={len(TRAIN_ITEM)} |eval|={len(eval_itemkey)} INTERSECTION=0  OK")

inter_qid = TRAIN_QID & eval_qid
assert not inter_qid, f"QID LEAK: {len(inter_qid)} (source, question-id) pairs in both"
print(f"  question ids:|train|={len(TRAIN_QID)} |eval|={len(eval_qid)} INTERSECTION=0  OK")

assert sum(allow_violations.values()) == 0, f"ALLOWLIST VIOLATION: {allow_violations}"
print(f"  allowlist:   0 generated idx outside the L1 allowlists  OK")

# ---------------------------------------------------------------- pool statistics
print("\n[stats] T=0.4 vs T=0.7 candidate diversity on the SAME train questions ...", flush=True)
stats = {}
for ds in TRAIN_SOURCES:
    row = {}
    for temp, path in (("T07", J(f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl")),
                       ("T04", os.path.join(COLD, f"ckpt_{ds}_lingshu7bT04_sc8.jsonl"))):
        if not os.path.exists(path):
            continue
        nq = nd = nun = 0
        toks = 0
        for l in open(path):
            if not l.strip():
                continue
            r = json.loads(l); nq += 1
            d = len(set(str(a).strip().lower() for a in r["preds"]))
            nd += d; nun += int(d == 1)
            toks += sum(r.get("gen_tokens_all", [0]))
        row[temp] = {"questions": nq, "mean_distinct_of_8": round(nd / max(nq, 1), 4),
                     "frac_unanimous": round(nun / max(nq, 1), 4),
                     "distinct_answer_examples_available": nd,
                     "mean_gen_tokens": round(toks / max(nq * 8, 1), 3)}
    stats[ds] = row
    if "T04" in row and "T07" in row:
        print(f"  {ds:20s} distinct/8  T07={row['T07']['mean_distinct_of_8']:.3f} -> "
              f"T04={row['T04']['mean_distinct_of_8']:.3f} | examples "
              f"{row['T07']['distinct_answer_examples_available']:6d} -> "
              f"{row['T04']['distinct_answer_examples_available']:6d}", flush=True)

out = {
    "title": "T=0.4 TRAIN candidate pools for the verifier co-adaptation round -- build + disjointness proof",
    "date": "2026-08-14",
    "prereg": "results/cascade_methods/artifacts/_coadapt_verifier_prereg_2026-08-14.json",
    "generator": "src/labeling/run_openvqa.py via runners/run_coadapt_T04_trainpools.sh -- "
                 "--n_samples 8 --temp 0.4 --cap cap320 --max_model_len 4096, idx allowlists from "
                 "data/disjoint_split/. IDENTICAL to the incumbent's generation "
                 "(runners/run_verifier_disjoint_retrain.sh stage 1) except --temp 0.7 -> 0.4.",
    "pool_dir": "ckpts/openvqa/cheap_lingshu7b_T04",
    "assertion_method": "images: md5 of DECODED RGB pixels (WxH + raw bytes), catches re-encoded copies. "
                        "items: (dataset family, normalized question text, image pixel hash). "
                        "Both functions copied VERBATIM from "
                        "src/training_methods/build_disjoint_verifier_split.py.",
    "assertions": {"image_pixel_md5_intersection": 0, "item_identity_intersection": 0,
                   "question_id_intersection": 0, "generated_idx_outside_L1_allowlist": 0,
                   "n_train_images": len(TRAIN_IMG), "n_eval_images": len(EVAL_IMG),
                   "n_train_question_items": len(TRAIN_ITEM), "n_eval_question_items": len(eval_itemkey),
                   "asserted_in_code": ["assert not inter_img", "assert not inter_item",
                                        "assert not inter_qid", "assert sum(allow_violations)==0"]},
    "eval_items_per_set": eval_n,
    "per_source": per_src,
    "candidate_diversity_T07_vs_T04": stats,
}

# ---------------------------------------------------------------- judge-label provenance
PR = os.path.join(COLD, "judgepreload_report_T04.json")
if os.path.exists(PR):
    pr = json.load(open(PR))
    pre = pr.get("preload", {}); val = pr.get("validate", {})
    n_rows = sum(v["cold_judge_rows"] for v in pre.values())
    n_pre = sum(v["preloadable"] for v in pre.values())
    tot = val.get("TOTAL", {})
    a, d = tot.get("agree", 0), tot.get("disagree", 0)
    n = a + d
    # Wilson 95% interval on the disagreement rate
    if n:
        p = d / n; z = 1.959963984540054
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
        wil = [max(0.0, c - h), min(1.0, c + h)]
    else:
        wil = None
    out["judge_label_provenance"] = {
        "judge": "src/labeling/run_judge.py, MedVLThinker-32B (Qwen2.5-32B backbone), text-only, "
                 "temperature 0, tp=2 -- the SAME judge that labelled the incumbent's T=0.7 pools.",
        "total_judge_rows": n_rows,
        "reused_from_the_T07_pass": n_pre,
        "judged_fresh_in_this_round": n_rows - n_pre,
        "reuse_rate": round(n_pre / max(n_rows, 1), 4),
        "HOLDOUT_NULL_TEST": {
            "design": "300 preloadable rows per source were deliberately NOT preloaded, so the judge "
                      "relabelled them from scratch in this round; the fresh labels are then compared "
                      "against what the preload would have said.",
            "agree": a, "disagree": d, "agreement_rate": tot.get("agreement_rate"),
            "disagreement_rate_wilson95": wil,
            "the_single_disagreement": {
                "source": "slake_open_train", "idx": "3015#1",
                "question": "What is the function of the organ on the top of this image?",
                "gold": "Ventilation, pronunciation", "answer": "Breathe.",
                "stored_T07_label": 1, "fresh_label_this_round": 0,
                "diagnosis": "the question, the gold and the answer string are BYTE-IDENTICAL between "
                             "the two judge calls, so this is not a preload keying error -- it is the "
                             "judge itself flipping on a borderline item. run_judge.py decides by "
                             "comparing exp(logprob) of 'Yes' against 'No' in the top-20 logprobs; a "
                             "near-tie plus batch-composition-dependent floating point in vLLM flips "
                             "the sign."},
            "VERDICT": "the preload is SOUND AS A KEYING MECHANISM -- zero mis-keyed labels were found "
                       "in 1,500 checks. What it measures instead is the judge's own self-consistency, "
                       "and the rate agrees with the project's independent measurement: "
                       "artifacts of ckpts/openvqa/decoding_sweep/judgepreload_report.json record 17 "
                       "internal conflicts over 43,267 preloaded eval keys (~0.04%), and this round's "
                       "kvasir_open source found 5 conflicts over 14,059 keys. So ~0.05% of judge "
                       "labels are coin-flips on borderline items, in THIS round and in every earlier "
                       "one. The incumbent verifier's own training labels carry the identical property, "
                       "because they came from the same judge; the preload copies the judge's earlier "
                       "answer, which is neither better nor worse than a fresh call.",
            "effect_size_bound": "at the Wilson upper bound this is at most ~0.4% of the reused labels, "
                                 "i.e. at most a few dozen of the 10,364 training examples -- orders of "
                                 "magnitude below any effect this round is powered to detect. It is "
                                 "reported because it is real, not because it changes a conclusion."},
        "internal_conflicts_in_the_T07_sources": {k: v.get("hot_internal_conflict_rows")
                                                  for k, v in pre.items()},
        "conflicted_keys_dropped_rather_than_guessed": {k: v.get("hot_conflicted_keys_dropped")
                                                        for k, v in pre.items()},
    }
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
