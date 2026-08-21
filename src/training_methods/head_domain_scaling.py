#!/usr/bin/env python3
"""head_domain_scaling.py -- does DOMAIN BREADTH fix the transfer wall? The k-domain curve.

THE QUESTION THIS ANSWERS.  The frozen head, trained on four domains, loses to greedy on two of
three genuinely new cells and sits below the random-pick floor on one.  Two explanations survive:

  (a) the head needs to have SEEN a domain to help on it -- so nothing generalises and the method
      only ever works where you have training data;
  (b) the head has seen too FEW domains to have learned anything domain-general, and transfer
      improves as breadth grows.

They are distinguishable, and the distinction decides the whole roadmap.  (a) means the method is a
per-domain tool; (b) means it is a verifier that has been starved.  With seven domains now on disk
we can draw the curve: hold one out, train on k of the remaining six for k = 1..6, and watch
`head - string prior` on the held-out domain as k rises.

  rising in k   -> (b), breadth is the fix, and the slope says how many domains are needed
  flat in k     -> (a), the method is per-domain and should be presented as one

Reported as head-minus-string-prior throughout, because on a closed-vocabulary domain a counter over
answer strings reproduces up to 101% of a head's apparent gain (2026-08-19 audit).

  python3 src/training_methods/head_domain_scaling.py --threads 10 --seeds 3
"""
import argparse, itertools, json, os, sys
import numpy as np
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import head_sweep as HS

# every pool with generator-frame features on disk
CELLS = {
    "kvasir_open": None, "pathvqa_open_train": None,        # from the frozen train cache
    "slake_open_train": None, "vqa_rad_open_train": None,
    "omnimed_open": "generator_eval_omnimed_open",          # new cells, own caches
    "vqamed_open": "generator_eval_vqamed_open",
    "kvasir_x1_open": "generator_eval_kvasir_x1_open",
}


def norm(s):
    return str(s).strip().lower().rstrip(".")


def load_all():
    """Concatenate the frozen train pool with every new cell into one domain-labelled matrix."""
    H, y, qid, img, ds = HS.load_train()
    X = HS.assemble(H, HS.BASE)
    rows = []
    for sh in (0, 1):
        rows += json.load(open(os.path.join(HS.FEATS, f"generator_train_s{sh}of2.meta.json")))["rows"]
    rows = [r for r in rows if r.get("n_tok", -1) > 0]
    na = np.array([norm(r["na"]) for r in rows])
    Xs, ys, qs, ns, dsl = [X], [y], [qid], [na], [ds]
    for cell, stem in CELLS.items():
        if stem is None:
            continue
        p = os.path.join(HS.FEATS, stem + ".npz")
        if not os.path.exists(p):
            print(f"  [skip] {cell}: no feature cache", flush=True)
            continue
        z = np.load(p)
        m = json.load(open(os.path.join(HS.FEATS, stem + ".meta.json")))
        keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0]
        rr = [m["rows"][i] for i in keep]
        Xs.append(z["h_span"][keep, HS.LAYERS.index(21)].astype(np.float32))
        ys.append(np.array([r["y"] for r in rr], dtype=np.float32))
        qs.append(np.array([f"{cell}|{r['idx']}" for r in rr]))
        ns.append(np.array([norm(r["na"]) for r in rr]))
        dsl.append(np.array([cell] * len(rr)))
        print(f"  [load] {cell}: {len(rr)} rows", flush=True)
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(qs),
            np.concatenate(ns), np.concatenate(dsl))


def pick_sel_eff(sc, y, qid, mask):
    byq = defaultdict(list)
    for i in np.where(mask)[0]:
        byq[qid[i]].append(i)
    got, rec = [], []
    for q, ii in byq.items():
        ii = np.array(ii)
        got.append(int(y[ii][int(np.argmax(sc[ii]))]))
        rec.append(int(y[ii].max()))
    got, rec = np.array(got), np.array(rec)
    return float(got[rec == 1].mean()) if rec.sum() else float("nan")


def string_prior(y, na, train_mask):
    pos, tot = defaultdict(int), defaultdict(int)
    idx = np.where(train_mask)[0]
    for i in idx:
        tot[na[i]] += 1
        pos[na[i]] += int(y[i])
    gp = float(y[idx].mean()) if len(idx) else 0.5
    return np.array([(pos[a] + gp * 2) / (tot[a] + 2) if tot[a] else gp for a in na])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--draws", type=int, default=3, help="random domain subsets per k")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--domains", nargs="*", default=None,
                    help="held-out domains this process is responsible for. The curve for one "
                         "held-out domain is independent of every other, so sharding over them is "
                         "exact -- 7 domains x 6 k-values x 9 fits on 142k rows is ~28h serial.")
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR, "head_domain_scaling_2026-08-19.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)

    X, y, qid, na, ds = load_all()
    doms = sorted(set(ds))
    print(f"\n{len(y)} rows over {len(doms)} domains: {doms}\n", flush=True)

    art = {"title": "Does domain breadth fix the transfer wall? k-domain scaling curve",
           "date": "2026-08-19", "no_fabricated_numbers": True,
           "endpoint": "held-out-domain sel_eff MINUS the string prior fitted on the same training "
                       "rows, as a function of how many OTHER domains the head saw",
           "domains": doms, "n_rows": int(len(y)), "results": {}}
    rng = np.random.default_rng(0)

    todo = [d for d in doms if (A.domains is None or d in A.domains)]
    print(f"this shard holds out: {todo}\n", flush=True)
    for held in todo:
        isH = ds == held
        others = [d for d in doms if d != held]
        art["results"][held] = {}
        for k in range(1, len(others) + 1):
            subsets = ([tuple(others)] if k == len(others) else
                       [tuple(rng.choice(others, k, replace=False)) for _ in range(A.draws)])
            vals, priors = [], []
            for sub in subsets:
                tr = np.isin(ds, list(sub))
                if tr.sum() < 200:
                    continue
                priors.append(pick_sel_eff(string_prior(y, na, tr), y, qid, isH))
                for s in range(A.seeds):
                    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
                    m = HS.fit((X[tr] - mu) / sg, y[tr], qid[tr], None, objective="bce",
                               hidden=A.hidden, wd=1e-2, epochs=30, seed=s)
                    sc = np.empty(len(X), dtype=np.float32)
                    for b0 in range(0, len(X), 4096):
                        b1 = min(b0 + 4096, len(X))
                        sc[b0:b1] = HS.predict(m, (X[b0:b1] - mu) / sg)
                    vals.append(pick_sel_eff(sc, y, qid, isH))
            if not vals:
                continue
            hp = float(np.mean(vals) - np.mean(priors))
            art["results"][held][str(k)] = {
                "n_train_domains": k, "n_subsets": len(subsets),
                "out_sel_eff": float(np.mean(vals)), "sd": float(np.std(vals)),
                "string_prior": float(np.mean(priors)), "head_minus_prior": hp}
            print(f"  [{held:20}] k={k}  sel_eff {np.mean(vals):.4f}  prior {np.mean(priors):.4f}"
                  f"  head-prior {hp:+.4f}", flush=True)
            json.dump(art, open(A.out, "w"), indent=1)
        # slope of head-minus-prior in k -- the whole point of the curve
        ks = sorted(int(x) for x in art["results"][held])
        if len(ks) >= 2:
            hp = [art["results"][held][str(x)]["head_minus_prior"] for x in ks]
            sl = float(np.polyfit(ks, hp, 1)[0])
            art["results"][held]["slope_head_minus_prior_per_domain"] = sl
            print(f"  [{held:20}] SLOPE {sl:+.5f} per added domain\n", flush=True)
            json.dump(art, open(A.out, "w"), indent=1)
    sls = [v["slope_head_minus_prior_per_domain"] for v in art["results"].values()
           if "slope_head_minus_prior_per_domain" in v]
    if sls:
        art["MEAN_SLOPE"] = float(np.mean(sls))
        art["VERDICT"] = ("breadth helps -- the head is a starved verifier, not a per-domain tool"
                          if np.mean(sls) > 0.005 else
                          "breadth does NOT help -- the method is per-domain and must be presented "
                          "as one")
        print(f"\nMEAN SLOPE {np.mean(sls):+.5f}  =>  {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
