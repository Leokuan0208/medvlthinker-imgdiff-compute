#!/usr/bin/env python3
"""Build the cross-family re-judge input for the probe's TRAINING rows (Lingshu-7B candidates).

Same row set head_final_stack.py / freeze_pooled_selector.py train on: the four train-domain rows of
generator_train_finelayer + the by-image TRAIN halves (md5("nd"+img_md5)%2==1) of the eight benchmarks, minus
rows whose image appears in a held-out half. Only the metas are read (no feature arrays). Question and gold
come from the sc8 dump of the same dataset; the answer is the RAW string (meta 'ans'), which is what judge A saw.
idx = "<ds>|<question idx>|<row number within this file>" so labels can be joined back by (ds, idx, ans)."""
import hashlib, json, os, sys
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
FE, CK = f"{ROOT}/feats_hidden", f"{ROOT}/ckpts/openvqa/cheap_lingshu7b"
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
SHARED = {"slake_open", "vqa_rad_open"}


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def meta(stem):
    return [r for r in json.load(open(f"{FE}/{stem}.meta.json"))["rows"] if r.get("n_tok", -1) > 0]


rows, evimgs = [], set()
for r in meta("generator_train_finelayer"):
    if r.get("ds") in TRAIN_DOMAINS:
        rows.append((r["ds"], r))
for cell in BENCH:
    stem = "generator_eval_finelayer" if cell in SHARED else f"generator_eval_finelayer_{cell}"
    for r in meta(stem):
        if cell in SHARED and r.get("ds") != cell:
            continue
        if half(r["img_md5"]) == 1:
            rows.append((cell, r))
        else:
            evimgs.add(r["img_md5"])
rows = [(d, r) for d, r in rows if r["img_md5"] not in evimgs]
Q = {}
for ds in sorted({d for d, _ in rows}):
    p = f"{CK}/ckpt_{ds}_lingshu7b_sc8.jsonl"
    for l in open(p):
        if l.strip():
            x = json.loads(l); Q[(ds, x["idx"])] = (x["question"], x["gold"])
n, miss, noans = 0, 0, 0
with open(sys.argv[1], "w") as fh:
    for k, (ds, r) in enumerate(rows):
        qg = Q.get((ds, r["idx"]))
        if qg is None:
            miss += 1; continue
        a = r.get("ans")
        if a is None:
            noans += 1; a = r["na"]
        fh.write(json.dumps({"idx": f"{ds}|{r['idx']}|{k}", "question": qg[0], "gold": qg[1], "modal_pred": a,
                             "na": r["na"], "y_A": int(r["y"])}) + "\n")
        n += 1
print(f"training rows {len(rows)}  written {n}  no question in dump {miss}  rows without raw 'ans' {noans}")
