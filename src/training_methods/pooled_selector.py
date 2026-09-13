#!/usr/bin/env python3
"""pooled_selector.py -- load the retrained probe verifier from disk and score with it.

The point of freezing an artifact is that using it is a FORWARD PASS, not a retraining run: a
retrained probe is a draw from a seed distribution, not a fixed object. This module never fits
anything. It loads the 24 heads and the three per-layer standardizers written by
freeze_pooled_selector.py and reproduces the measured number from disk.

    from pooled_selector import PooledSelector
    S = PooledSelector.load()
    pick = S.select({18: X18, 20: X20, 22: X22})     # feature matrices for one candidate set

    python3 src/training_methods/pooled_selector.py    # verify() against the held-out halves
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np
import torch

D_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D_)
import head_sweep as HS

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CKDIR = os.path.join(ROOT, "ckpts/train/genframe_head_pooled_ens")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
ENS = [18, 20, 22]
# 2026-09-13: pathvqa_open REMOVED from the shared set. It lived in the combined
# generator_eval_finelayer cache alongside slake and vqa_rad, and that cache covers only
# the truncated 1,500-question pathvqa. It now has its own complete 3,357-question cache at
# generator_eval_finelayer_pathvqa_open, so it is read per-benchmark like every other cell.
SHARED = {"slake_open", "vqa_rad_open"}


class PooledSelector:
    def __init__(self, heads, stats, recipe):
        self.heads, self.stats, self.recipe = heads, stats, recipe

    @classmethod
    def load(cls, d=CKDIR):
        recipe = json.load(open(os.path.join(d, "recipe.json")))
        heads, stats = {}, {}
        for L in ENS:
            z = np.load(os.path.join(d, f"standardizer_L{L}.npz"))
            stats[L] = (z["mu"], z["sd"])
            hs = []
            for s in range(recipe["seeds_per_layer"]):
                m = HS.MLP(len(z["mu"]), hidden=256)
                m.load_state_dict(torch.load(os.path.join(d, f"head_L{L}_seed{s}.pt"),
                                             map_location="cpu"))
                m.eval(); hs.append(m)
            heads[L] = hs
        return cls(heads, stats, recipe)

    def scores(self, X_by_layer):
        """Mean within-candidate-set rank over all heads. X_by_layer: {layer: (n_slots, 3584)}."""
        from genframe_data import rank_avg
        rk = []
        for L, hs in self.heads.items():
            mu, sd = self.stats[L]
            Z = torch.tensor((X_by_layer[L] - mu) / sd, dtype=torch.float32)
            with torch.no_grad():
                for m in hs:
                    # HS.MLP.forward already squeezes the trailing dim, so squeezing again turns a
                    # single-candidate set into a 0-d tensor and rank_avg dies on len().
                    rk.append(rank_avg(np.atleast_1d(m(Z).numpy())))
        return np.mean(rk, axis=0)

    def select(self, X_by_layer):
        return int(np.argmax(self.scores(X_by_layer)))


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def verify():
    from collections import defaultdict
    S = PooledSelector.load()
    print(f"loaded {sum(len(v) for v in S.heads.values())} heads from {CKDIR}")
    print(f"recipe claims macro {S.recipe['measured_on_held_out_halves']['macro_verifier_minus_greedy']:+.4f}")
    deltas = {}
    for cell in ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
                 "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]:
        stem = "generator_eval_finelayer" if cell in SHARED else f"generator_eval_finelayer_{cell}"
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not (os.path.exists(f"{HS.FEATS}/{stem}.npz") and os.path.exists(gjp)):
            continue
        z = np.load(f"{HS.FEATS}/{stem}.npz")
        m = json.load(open(f"{HS.FEATS}/{stem}.meta.json"))
        lay = [int(x) for x in z["layers"]]
        dsf = {cell} if cell in SHARED else None
        keep = [i for i, r in enumerate(m["rows"])
                if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)
                and half(r["img_md5"]) == 0]
        if not keep:
            continue
        rr = [m["rows"][i] for i in keep]
        X = {L: z["h_span"][keep, lay.index(L)].astype(np.float32) for L in ENS}
        y = np.array([r["y"] for r in rr], dtype=int)
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        acc, gr = [], []
        for q, ii in byq.items():
            if q not in gok:
                continue
            ii = np.array(ii)
            p = S.scores({L: X[L][ii] for L in ENS})
            acc.append(int(y[ii][int(np.argmax(p))])); gr.append(gok[q])
        if len(acc) < 40:
            continue
        d = float(np.mean(acc) - np.mean(gr))
        deltas[cell] = d
        print(f"  {cell:17} n{len(acc):6}  verifier {np.mean(acc):.4f}  greedy {np.mean(gr):.4f}"
              f"  delta {d:+.4f}")
    macro = float(np.mean(list(deltas.values())))
    claim = S.recipe["measured_on_held_out_halves"]["macro_verifier_minus_greedy"]
    print(f"\n  MACRO from disk {macro:+.4f} | recipe claims {claim:+.4f} | "
          f"deviation {macro-claim:+.5f}")
    print(f"  benchmarks beaten {sum(1 for v in deltas.values() if v>0)}/{len(deltas)}")
    ok = abs(macro - claim) < 0.005
    print(f"\n  {'RELOAD VERIFIED' if ok else 'MISMATCH -- the artifact does not reproduce its recipe'}")
    return macro


if __name__ == "__main__":
    verify()
