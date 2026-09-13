#!/usr/bin/env python3
"""head_temp_ensemble.py -- the SHIPPED verifier's temperature preference, which was never measured.

Every temperature result in this project was produced with a single-layer layer-21 probe, because
the T=0.2/0.4/1.0 feature caches only held layers [7,14,21,28].  The shipped artifact rank-ensembles
layers 18/20/22 and has never been evaluated across temperature at all.

That gap matters because head_ens_width_2026-08-25.json found the pooled probe's single-layer spread
collapses to 0.003 against 0.013 for the four-domain probe -- pooling flattens the layer effect.  If
pooling also flattens the TEMPERATURE effect, then the per-benchmark best-T finding (worth +0.0117,
and in-sample at that) is an artifact of a weaker verifier rather than a property of the candidate
sets, and the deployed system can simply fix one temperature.

Scored with the frozen artifact on held-out image halves only.

  python3 src/cascade_methods/head_temp_ensemble.py
"""
import hashlib, json, os, sys
import numpy as np
from collections import defaultdict

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/head_temp_ensemble_2026-08-30.json")
ENS = [18, 20, 22]
TT = {0.2: ("_T02", "lingshu7bT02"), 0.4: ("_T04", "lingshu7bT04"),
      0.7: ("", "lingshu7b"), 1.0: ("_T10", "lingshu7bT10")}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
# 2026-09-13: pathvqa_open REMOVED from the shared set. It lived in the combined
# generator_eval_finelayer cache alongside slake and vqa_rad, and that cache covers only
# the truncated 1,500-question pathvqa. It now has its own complete 3,357-question cache at
# generator_eval_finelayer_pathvqa_open, so it is read per-benchmark like every other cell.
SHARED = {"slake_open", "vqa_rad_open"}


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def main():
    from pooled_selector import PooledSelector
    S = PooledSelector.load()
    art = {"title": "Temperature preference of the SHIPPED ensemble verifier", "date": "2026-08-30",
           "no_fabricated_numbers": True, "verifier": "genframe_head_pooled_ens",
           "eval": "held-out image halves", "cells": {}}
    for cell in BENCH:
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not os.path.exists(gjp):
            continue
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        rec = {}
        for T, (suf, tag) in sorted(TT.items()):
            stem = (f"{FEATS}/generator_eval_ens{suf}_{cell}" if suf else
                    (f"{FEATS}/generator_eval_finelayer" if cell in SHARED
                     else f"{FEATS}/generator_eval_finelayer_{cell}"))
            if not os.path.exists(stem + ".npz"):
                continue
            z = np.load(stem + ".npz"); m = json.load(open(stem + ".meta.json"))
            lay = [int(x) for x in z["layers"]]
            if not all(L in lay for L in ENS):
                continue
            dsf = cell if (not suf and cell in SHARED) else None
            keep = [i for i, r in enumerate(m["rows"])
                    if r.get("n_tok", -1) > 0 and half(r["img_md5"]) == 0
                    and (dsf is None or r.get("ds") == dsf)]
            if not keep:
                continue
            rr = [m["rows"][i] for i in keep]
            X = {L: z["h_span"][keep, lay.index(L)].astype(np.float32) for L in ENS}
            y = np.array([r["y"] for r in rr], dtype=int)
            byq = defaultdict(list)
            for i, r in enumerate(rr):
                byq[r["idx"]].append(i)
            qs = [q for q in byq if q in gok]
            if len(qs) < 40:
                continue
            acc, orc = [], []
            for q in qs:
                ii = np.array(byq[q])
                sc = S.scores({L: X[L][ii] for L in ENS})
                acc.append(int(y[ii][int(np.argmax(sc))])); orc.append(int(y[ii].max()))
            g = float(np.mean([gok[q] for q in qs]))
            rec[str(T)] = {"n": len(qs), "verifier": float(np.mean(acc)), "greedy": g,
                           "minus_greedy": float(np.mean(acc) - g),
                           "oracle": float(np.mean(orc)),
                           "mean_distinct": float(np.mean([len(byq[q]) for q in qs]))}
        if len(rec) < 2:
            continue
        art["cells"][cell] = rec
        Ts = sorted(float(t) for t in rec)
        best = max(Ts, key=lambda t: rec[str(t)]["minus_greedy"])
        art["cells"][cell]["best_T"] = best
        print(f"  {cell:17} " + "  ".join(f"T{t} {rec[str(t)]['minus_greedy']:+.4f}" for t in Ts)
              + f"   best {best}", flush=True)
        json.dump(art, open(OUT, "w"), indent=1)

    cs = art["cells"]
    if cs:
        Ts = [0.2, 0.4, 0.7, 1.0]
        mac = {t: float(np.mean([v[str(t)]["minus_greedy"] for v in cs.values() if str(t) in v]))
               for t in Ts if any(str(t) in v for v in cs.values())}
        art["macro_by_temperature"] = mac
        bt = max(mac, key=mac.get)
        spread = max(mac.values()) - min(mac.values())
        percell = float(np.mean([max(v[str(t)]["minus_greedy"] for t in Ts if str(t) in v)
                                 for v in cs.values()]))
        art["best_fixed_T"] = bt
        art["macro_at_best_fixed_T"] = mac[bt]
        art["macro_with_per_benchmark_T"] = percell
        art["value_of_tuning_T_per_benchmark"] = percell - mac[bt]
        art["macro_spread_across_T"] = spread
        for t, v in sorted(mac.items()):
            print(f"  MACRO T={t}  {v:+.4f}")
        art["VERDICT"] = (
            f"one fixed temperature T={bt} gives {mac[bt]:+.4f}; tuning T per benchmark (in-sample) "
            f"gives {percell:+.4f}, worth {percell-mac[bt]:+.4f}. Macro spread across temperatures "
            f"is {spread:.4f}. " +
            ("Temperature still matters for the shipped verifier and a per-benchmark choice is "
             "worth having." if percell - mac[bt] > 0.01 else
             "Pooling flattens the temperature effect as it flattened the layer effect: a single "
             "fixed temperature is within noise of per-benchmark tuning, so the deployed system "
             "does not need to choose one per dataset."))
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
