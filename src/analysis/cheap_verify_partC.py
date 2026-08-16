#!/usr/bin/env python3
"""PART C -- independent verification of (i) the OPEN-CELL BASELINE DEFECT (published 'always-7B
greedy' on the 3 open cells is really self-consistency@8) and (ii) PMC self-consistency@8.
Written from scratch against the frozen dumps."""
from __future__ import annotations
import json, os, re, sys
from collections import Counter
import numpy as np
sys.path.insert(0, "/home/jamesyang/medvlthinker-imgdiff-compute/src")
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_cheapverify")
NBOOT, SEED = 10000, 20260817

def boot(a, b, nboot=NBOOT, seed=SEED):
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)],
                sign=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"), n=int(len(d)))

def jl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def norm(s):
    s = str(s).strip().lower().strip("\"'“”‘’ \t\n").rstrip("。.")
    return re.sub(r"\s+", " ", s).strip()

R = {}
# ------------------------------------------------------------------ the frozen loader's own view
from training_methods import genframe_data as G
items = G.load_items()
print("loaded frozen items:", len(items), file=sys.stderr)

DSMAP = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open", "pathvqa_open": "PATH_VQA_open"}
T0 = {"slake_open": "ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b.judge.jsonl",
      "vqa_rad_open": "ckpts/openvqa/cheap_lingshu7b/ckpt_vqa_rad_open_lingshu7b.judge.jsonl",
      "pathvqa_open": "ckpts/openvqa/cheap_lingshu7b/ckpt_pathvqa_open_lingshu7b.judge.jsonl"}
TRANSFER = {ds: f"ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_lingshu7b.json" for ds in DSMAP}

cells = {}
pooled_t0, pooled_pub = [], []
for ds, cell in DSMAP.items():
    p = os.path.join(ROOT, TRANSFER[ds])
    if not os.path.exists(p):
        cands = [q for q in os.listdir(os.path.join(ROOT, "ckpts/train/lora_verifier_pooled4"))
                 if q.startswith(f"transfer_dump_{ds}")]
        p = os.path.join(ROOT, "ckpts/train/lora_verifier_pooled4", cands[0])
    dump = json.load(open(p))
    t0 = {r["idx"]: int(r["judge_ok"]) for r in jl(os.path.join(ROOT, T0[ds]))}
    n_modal, n_slot0, n = 0, 0, 0
    gpub, gt0 = [], []
    for r in dump:
        sl = r["sl"]; preds = r["preds"]
        if all(x in (None, -1) for x in sl): continue
        # reconstruct the modal-of-8 label from the judge labels attached to each slot
        lab = {}
        for a, s in zip(preds, sl):
            if s not in (None, -1): lab[norm(a)] = int(s)
        modal = Counter(norm(a) for a in preds).most_common(1)[0][0]
        modal_ok = int(lab.get(modal, 0))
        slot0_ok = 0 if sl[0] in (None, -1) else int(sl[0])
        g = int(r["greedy_ok"])
        n += 1
        n_modal += int(g == modal_ok); n_slot0 += int(g == slot0_ok)
        gpub.append(g); gt0.append(t0.get(r["idx"], 0))
    cells[cell] = dict(
        n=n,
        published_greedy_ok_mean=float(np.mean(gpub)),
        greedy_ok_equals_modal_of_8=f"{n_modal}/{n} = {n_modal/n:.4f}",
        greedy_ok_equals_slot0=f"{n_slot0}/{n} = {n_slot0/n:.4f}",
        true_T0_greedy_acc=float(np.mean(gt0)),
        true_T0_minus_published=boot(gt0, gpub))
    pooled_t0 += gt0; pooled_pub += gpub

R["C1_open_cell_baseline_defect"] = dict(
    cells=cells,
    pooled=dict(n=len(pooled_pub),
                published_greedy_ok=float(np.mean(pooled_pub)),
                true_T0_greedy=float(np.mean(pooled_t0)),
                delta=boot(pooled_t0, pooled_pub)),
    macro8_shift=float(sum(cells[c]["true_T0_minus_published"]["delta"] for c in cells) / 8.0))
json.dump(R, open(os.path.join(OUT, "partC.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
