#!/usr/bin/env python3
"""s11: length control.  Does a PICK-THE-LONGEST selector reproduce the judge gain?  And what do the
MedGemma sc8 candidates look like next to its greedy answer?  READ-ONLY."""
import os, sys, json, time
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, "/data/dan/audit_2026-09-18/tmp/em-rescore")
from em_rescore_pooled_probe import em_norm, em_score, strict_em, token_f1, row_norm, nwords
MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
CK = os.path.join(MAIN, "ckpts/openvqa/cheap_lingshu7b")
OUT = "/data/dan/audit_2026-09-18/tmp/replication"
B = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
     "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
MT = {g: json.load(open(f"{OUT}/refit/{g}_L{L}.meta.json"))
      for g, L in (("qwen", 20), ("medgemma", 24))}
TAG = {"qwen": "qwen25vl7b", "medgemma": "medgemma4b", "lingshu": "lingshu7b"}
RES = {}
for gen in ("qwen", "medgemma", "lingshu"):
    tag = TAG[gen]
    cells = {}
    for cell in B:
        gj, gd = {}, {}
        for l in open(f"{CK}/ckpt_{cell}_{tag}.judge.jsonl"):
            if l.strip():
                d = json.loads(l); gj[d["idx"]] = int(d["judge_ok"])
        for l in open(f"{CK}/ckpt_{cell}_{tag}.jsonl"):
            if l.strip():
                d = json.loads(l); gd[d["idx"]] = d
        sc = {}
        for l in open(f"{CK}/ckpt_{cell}_{tag}_sc8.jsonl"):
            if l.strip():
                d = json.loads(l); sc[d["idx"]] = d
        cj = {}
        for l in open(f"{CK}/ckpt_{cell}_{tag}_sc8_scexploded.judge.jsonl"):
            if l.strip():
                d = json.loads(l); q, s = str(d["idx"]).split("#"); cj[(int(q), int(s))] = int(d["judge_ok"])
        if gen in MT:
            qs = sorted({r["idx"] for r in MT[gen]["heldout_rows"][cell]} & set(gj))
        else:
            qs = sorted(gj)      # lingshu: no refit meta here -> full benchmark, flagged below
        gw, vw, rw, lw, lj, lem, lst, lf1, gjv, rj = [], [], [], [], [], [], [], [], [], []
        for q in qs:
            s8, g = sc.get(q), gd.get(q)
            if s8 is None or g is None:
                continue
            preds = list(s8["preds"]); gold = s8["gold"]
            pn = [row_norm(a) for a in preds]
            first = {}
            for i, a in enumerate(pn):
                first.setdefault(a, i)
            if any((q, i) not in cj for i in first.values()):
                continue
            lab = [cj[(q, first[a])] for a in pn]
            k = int(np.argmax([nwords(a) for a in preds]))     # pick the LONGEST, first-index ties
            gw.append(nwords(g["preds"][0])); vw.append(nwords(preds[k]))
            rw.append(float(np.mean([nwords(a) for a in preds])))
            lj.append(lab[k]); lem.append(em_score(preds[k], gold))
            lst.append(strict_em(preds[k], gold)); lf1.append(token_f1(preds[k], gold))
            gjv.append(gj[q]); rj.append(float(np.mean(lab)))
        cells[cell] = {"n": len(gjv), "greedy_judge": float(np.mean(gjv)),
                       "longest_judge": float(np.mean(lj)),
                       "longest_minus_greedy_judge": float(np.mean(lj) - np.mean(gjv)),
                       "longest_em": float(np.mean(lem)), "longest_strict_em": float(np.mean(lst)),
                       "longest_token_f1": float(np.mean(lf1)),
                       "mean_words_greedy": float(np.mean(gw)),
                       "mean_words_longest": float(np.mean(vw)),
                       "mean_words_random_candidate": float(np.mean(rw)),
                       "random_pick_judge": float(np.mean(rj))}
        print("%-9s %-17s n%6d greedy_J %.4f longest_J %.4f d %+.4f  words g %.2f rand %.2f long %.2f"
              % (gen, cell, cells[cell]["n"], cells[cell]["greedy_judge"], cells[cell]["longest_judge"],
                 cells[cell]["longest_minus_greedy_judge"], cells[cell]["mean_words_greedy"],
                 cells[cell]["mean_words_random_candidate"], cells[cell]["mean_words_longest"]), flush=True)
    m = {k: float(np.mean([cells[c][k] for c in B])) for k in cells[B[0]] if k != "n"}
    m["n_positive_longest"] = int(sum(1 for c in B if cells[c]["longest_minus_greedy_judge"] > 0))
    m["heldout_only"] = gen in MT
    RES[gen] = {"cells": cells, "macro": m}
    print("  MACRO", gen, json.dumps({k: round(v, 4) for k, v in m.items() if isinstance(v, float)}), flush=True)
json.dump(RES, open(f"{OUT}/s11_length.json", "w"), indent=1)

# a few MedGemma examples
print("\n### MedGemma: greedy answer vs its 8 sampled candidates (held-out, kvasir/gemex/omnimed)")
for cell in ("kvasir_x1_open", "gemex_open", "omnimed_open"):
    sc = {}
    for l in open(f"{CK}/ckpt_{cell}_medgemma4b_sc8.jsonl"):
        if l.strip():
            d = json.loads(l); sc[d["idx"]] = d
    gd = {}
    for l in open(f"{CK}/ckpt_{cell}_medgemma4b.jsonl"):
        if l.strip():
            d = json.loads(l); gd[d["idx"]] = d
    ks = sorted(sc)[:3]
    for q in ks:
        print(f"-- {cell} idx {q}  GOLD: {sc[q]['gold'][:120]!r}")
        print(f"   greedy   ({nwords(gd[q]['preds'][0]):3}w): {gd[q]['preds'][0][:200]!r}")
        for j, a in enumerate(sc[q]["preds"][:3]):
            print(f"   cand[{j}]  ({nwords(a):3}w): {a[:200]!r}")
