#!/usr/bin/env python3
"""head_input_augmentation.py -- give the probe MORE THAN the hidden state, at the input.

WHY THIS AND NOT MORE FUSION.  We already tested combining the probe with self-consistency at the
SCORE level: free_signal_bakeoff_2026-08-21.json fuses the two rank vectors 50/50 and it loses to
the probe alone on the benchmarks the probe wins, while winning where it loses -- a wash. But score
-level fusion forces a fixed, question-independent trade-off. Feature-level fusion does not: the
probe can learn that self-consistency is worth attending to only when the hidden state is ambiguous.
That has never been tried, and it costs no GPU because self-consistency is recoverable from the raw
candidate-set dumps we already have for the training domains.

WHAT IS ADDED to the 3584-d hidden state, all computable at inference without labels:
    sc     the candidate's sampling frequency in its own candidate set
    len    answer length in tokens, z-scored within the candidate set (verbosity is a known
           confound here, so it is given to the probe explicitly rather than left latent)
    nd     number of distinct answers in the candidate set (a per-question difficulty proxy)

ALSO TESTED: TRAINING-SET AUGMENTATION over temperature. We now hold T=0.2 training pools as well as
T=0.7. Fitting on their union shows the probe candidate sets of both shapes; if the cold-pool effect
is about the probe never having seen low-diversity candidate sets, this is the cheap fix.

Endpoint is leave-one-training-domain-out transfer minus the answer-prior null -- the same one the
architecture, layer and representation sweeps used, so the numbers are comparable across all four.

  python3 src/training_methods/head_input_augmentation.py --threads 4 --seeds 3
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict, Counter

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import head_sweep as HS
from head_domain_scaling import pick_sel_eff, string_prior, norm

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
LAY = 21


def extras_for(rows, tag="lingshu7b"):
    """(sc, len_z, n_distinct) per row, from the raw pre-dedup candidate-set dumps."""
    raw = {}
    for ds in sorted({r["ds"] for r in rows}):
        p = f"{CK}/ckpt_{ds}_{tag}_sc8.jsonl"
        if not os.path.exists(p):
            continue
        for l in open(p):
            if l.strip():
                d = json.loads(l); raw[(ds, d["idx"])] = d
    sc = np.zeros(len(rows), np.float32)
    ln = np.zeros(len(rows), np.float32)
    nd = np.zeros(len(rows), np.float32)
    for i, r in enumerate(rows):
        d = raw.get((r["ds"], r["idx"]))
        a = norm(r["na"])
        if d:
            c = Counter(norm(p) for p in d["preds"])
            sc[i] = c.get(a, 0) / max(len(d["preds"]), 1)
            nd[i] = len(c) / max(len(d["preds"]), 1)
        ln[i] = len(str(r.get("ans", a)).split())
    # z-score length WITHIN each candidate set, so it is a relative not an absolute cue
    byq = defaultdict(list)
    for i, r in enumerate(rows):
        byq[(r["ds"], r["idx"])].append(i)
    for _, ii in byq.items():
        ii = np.array(ii)
        m, s = ln[ii].mean(), ln[ii].std() + 1e-6
        ln[ii] = (ln[ii] - m) / s
    return sc, ln, nd


def load(stems, layer=LAY):
    Xs, rows = [], []
    for st in stems:
        p = f"{HS.FEATS}/{st}"
        if not os.path.exists(p + ".npz"):
            continue
        z = np.load(p + ".npz"); m = json.load(open(p + ".meta.json"))
        lay = [int(x) for x in z["layers"]]
        if layer not in lay:
            continue
        keep = [i for i, r in enumerate(m["rows"])
                if r.get("n_tok", -1) > 0 and r["ds"] in
                {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}]
        Xs.append(z["h_span"][keep, lay.index(layer)].astype(np.float32))
        rows += [m["rows"][i] for i in keep]
    return np.concatenate(Xs), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR,
                                                  "head_input_augmentation_2026-08-24.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)

    X7, rows7 = load([f"generator_train_s{sh}of2" for sh in (0, 1)])
    sc7, ln7, nd7 = extras_for(rows7, "lingshu7b")
    print(f"T=0.7 pool: {len(rows7)} rows, mean self-consistency {sc7.mean():.3f}", flush=True)
    have_cold = os.path.exists(f"{HS.FEATS}/generator_train_T02.npz")
    if have_cold:
        X2, rows2 = load(["generator_train_T02"])
        sc2, ln2, nd2 = extras_for(rows2, "lingshu7bT02")
        print(f"T=0.2 pool: {len(rows2)} rows, mean self-consistency {sc2.mean():.3f}", flush=True)

    def pack(X, sc, ln, nd, which):
        cols = [X]
        if "sc" in which: cols.append(sc[:, None])
        if "len" in which: cols.append(ln[:, None])
        if "nd" in which: cols.append(nd[:, None])
        return np.concatenate(cols, 1).astype(np.float32)

    art = {"title": "Feature-level augmentation of the probe's input", "date": "2026-08-24",
           "no_fabricated_numbers": True, "layer": LAY,
           "endpoint": "leave-one-training-domain-out sel_eff MINUS the answer-prior null",
           "results": {}}

    def run(name, Xtr, rows, tag_extra=""):
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        qid = np.array([f"{r['ds']}|{r['idx']}" for r in rows])
        ds = np.array([r["ds"] for r in rows])
        na = np.array([norm(r["na"]) for r in rows])
        hp = []
        for d in sorted(set(ds)):
            isD = ds == d; tr = ~isD
            sp = pick_sel_eff(string_prior(y, na, tr), y, qid, isD)
            vals = []
            for s in range(A.seeds):
                mu, sg = Xtr[tr].mean(0), Xtr[tr].std(0) + 1e-6
                m = HS.fit((Xtr[tr] - mu) / sg, y[tr], qid[tr], None, objective="bce",
                           hidden=256, wd=1e-2, epochs=30, seed=s)
                sc_ = np.empty(len(Xtr), np.float32)
                for b0 in range(0, len(Xtr), 4096):
                    b1 = min(b0 + 4096, len(Xtr))
                    sc_[b0:b1] = HS.predict(m, (Xtr[b0:b1] - mu) / sg)
                vals.append(pick_sel_eff(sc_, y, qid, isD))
            hp.append(float(np.mean(vals)) - sp)
        art["results"][name] = {"per_domain": hp, "mean": float(np.mean(hp)),
                                "n_features": int(Xtr.shape[1]), "n_rows": int(len(y))}
        print(f"  {name:34} mean head-minus-prior {np.mean(hp):+.5f}  "
              f"({Xtr.shape[1]}d, {len(y)} rows)", flush=True)
        json.dump(art, open(A.out, "w"), indent=1)

    for which in ([], ["sc"], ["len"], ["sc", "len"], ["sc", "len", "nd"]):
        nm = "h_span" + ("+" + "+".join(which) if which else " (baseline)")
        run(nm, pack(X7, sc7, ln7, nd7, which), rows7)
    if have_cold:
        Xu = np.concatenate([pack(X7, sc7, ln7, nd7, []), pack(X2, sc2, ln2, nd2, [])])
        run("h_span, trained on T0.7+T0.2", Xu, rows7 + rows2)
        Xu2 = np.concatenate([pack(X7, sc7, ln7, nd7, ["sc", "len"]),
                              pack(X2, sc2, ln2, nd2, ["sc", "len"])])
        run("h_span+sc+len, trained on T0.7+T0.2", Xu2, rows7 + rows2)

    base = art["results"]["h_span (baseline)"]["mean"]
    best = max(art["results"], key=lambda k: art["results"][k]["mean"])
    art["baseline"] = base
    art["best"] = {"variant": best, "mean": art["results"][best]["mean"],
                   "gain": art["results"][best]["mean"] - base}
    art["VERDICT"] = (
        f"best is {best} at {art['results'][best]['mean']:+.5f}, {art['best']['gain']:+.5f} over the "
        f"plain hidden state. " +
        ("Feature-level fusion helps where score-level fusion did not."
         if art["best"]["gain"] > 0.005 else
         "Feature-level fusion does NOT help either: self-consistency and length carry nothing the "
         "hidden state does not already encode, which is why fusing them at the score level was "
         "also a wash."))
    print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
