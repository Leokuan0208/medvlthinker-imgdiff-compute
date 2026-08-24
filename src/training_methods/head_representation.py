#!/usr/bin/env python3
"""head_representation.py -- WHICH hidden state should the probe read? h_span, h_last, or both?

A FREE LEVER WE NEVER PULLED.  extract_generator_hidden.py has always saved TWO representations per
layer and the probe has only ever used one:

    h_span[L]   mean hidden state over the candidate answer's tokens   <- what the probe reads
    h_last[L]   hidden state at the readout position                   <- saved, never used

Every cache on disk already holds both, so this costs no GPU.  The 2026-08-19 architecture sweep
varied how h_span was NORMALISED and COMPRESSED (pool-norm, PCA, ranks, domain-adversarial) and all
six alternatives transferred worse than plain h_span -- but it never questioned the choice of h_span
itself.  A mean over answer tokens and the state at the readout position encode different things:
the first is a summary of what was said, the second is the model's state at the moment it commits.

ALSO TESTED, and also free: concatenating ADJACENT LAYERS.  The end-to-end layer comparison puts
18, 20 and 22 within 0.006 of each other (+0.0289 / +0.0257 / +0.0316 macro), which is the signature
of correlated-but-not-identical signals -- exactly when a concatenation or an ensemble can beat any
single choice.  Both are tried: concat feeds one probe a wider vector, ensemble averages the
within-candidate-set ranks of separately fitted probes.

Selection is on OUT-OF-DOMAIN transfer (leave-one-training-domain-out, scored as
sel_eff minus the answer-prior null), not in-domain CV, because CV picked layer 20 while transfer
picked 18 and the end-to-end winner was 22 -- CV is not the metric that tracks what we care about.

  python3 src/training_methods/head_representation.py --threads 4 --seeds 3
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import head_sweep as HS
from head_domain_scaling import pick_sel_eff, string_prior, norm

VARIANTS = ["h_span", "h_last", "h_span+h_last", "layers_18_20_22_concat", "layers_18_20_22_ens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR, "head_representation_2026-08-24.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)

    p = f"{HS.FEATS}/generator_train_finelayer"
    z = np.load(p + ".npz")
    rows = json.load(open(p + ".meta.json"))["rows"]
    keep = [i for i, r in enumerate(rows) if r.get("n_tok", -1) > 0]
    rows = [rows[i] for i in keep]
    layers = [int(x) for x in z["layers"]]
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{r['ds']}|{r['idx']}" for r in rows])
    ds = np.array([r["ds"] for r in rows])
    na = np.array([norm(r["na"]) for r in rows])
    li = layers.index(A.layer)
    span = z["h_span"][keep, li].astype(np.float32)
    last = z["h_last"][keep, li].astype(np.float32)
    tri = {L: z["h_span"][keep, layers.index(L)].astype(np.float32) for L in (18, 20, 22)}
    print(f"{len(y)} rows | layers {layers} | using layer {A.layer} for the h_span/h_last contrast",
          flush=True)

    art = {"title": "Which hidden state should the probe read?", "date": "2026-08-24",
           "no_fabricated_numbers": True, "layer": A.layer, "variants": VARIANTS,
           "endpoint": "leave-one-training-domain-out sel_eff MINUS the answer-prior null",
           "results": {}}

    def transfer(getX, ens=False):
        out = []
        for d in sorted(set(ds)):
            isD = ds == d; tr = ~isD
            sp = pick_sel_eff(string_prior(y, na, tr), y, qid, isD)
            vals = []
            for s in range(A.seeds):
                if ens:
                    scores = []
                    for Xp in getX():
                        mu, sg = Xp[tr].mean(0), Xp[tr].std(0) + 1e-6
                        m = HS.fit((Xp[tr] - mu) / sg, y[tr], qid[tr], None, objective="bce",
                                   hidden=256, wd=1e-2, epochs=30, seed=s)
                        sc = np.empty(len(Xp), np.float32)
                        for b0 in range(0, len(Xp), 4096):
                            b1 = min(b0 + 4096, len(Xp))
                            sc[b0:b1] = HS.predict(m, (Xp[b0:b1] - mu) / sg)
                        scores.append(sc)
                    # average the WITHIN-CANDIDATE-SET ranks, not the raw scores, because the
                    # separately fitted probes are not on a common scale
                    from genframe_data import rank_avg
                    byq = defaultdict(list)
                    for i, q in enumerate(qid):
                        byq[q].append(i)
                    sc = np.zeros(len(y), np.float32)
                    for q, ii in byq.items():
                        ii = np.array(ii)
                        sc[ii] = np.mean([rank_avg(s_[ii]) for s_ in scores], axis=0)
                else:
                    Xp = getX()
                    mu, sg = Xp[tr].mean(0), Xp[tr].std(0) + 1e-6
                    m = HS.fit((Xp[tr] - mu) / sg, y[tr], qid[tr], None, objective="bce",
                               hidden=256, wd=1e-2, epochs=30, seed=s)
                    sc = np.empty(len(Xp), np.float32)
                    for b0 in range(0, len(Xp), 4096):
                        b1 = min(b0 + 4096, len(Xp))
                        sc[b0:b1] = HS.predict(m, (Xp[b0:b1] - mu) / sg)
                vals.append(pick_sel_eff(sc, y, qid, isD))
            out.append(float(np.mean(vals)) - sp)
        return out

    GET = {"h_span": lambda: span, "h_last": lambda: last,
           "h_span+h_last": lambda: np.concatenate([span, last], 1),
           "layers_18_20_22_concat": lambda: np.concatenate([tri[18], tri[20], tri[22]], 1),
           "layers_18_20_22_ens": lambda: [tri[18], tri[20], tri[22]]}
    for v in VARIANTS:
        hp = transfer(GET[v], ens=v.endswith("_ens"))
        art["results"][v] = {"per_domain_head_minus_prior": hp, "mean": float(np.mean(hp)),
                             "n_features": (int(GET[v]().shape[1]) if not v.endswith("_ens")
                                            else int(tri[20].shape[1]) * 3)}
        print(f"  {v:26} mean head-minus-prior {np.mean(hp):+.5f}   per-domain "
              f"{[round(x,4) for x in hp]}", flush=True)
        json.dump(art, open(A.out, "w"), indent=1)

    base = art["results"]["h_span"]["mean"]
    best = max(art["results"], key=lambda k: art["results"][k]["mean"])
    art["baseline_h_span"] = base
    art["best"] = {"variant": best, "mean": art["results"][best]["mean"],
                   "gain_over_h_span": art["results"][best]["mean"] - base}
    art["VERDICT"] = (
        f"best representation is {best} at {art['results'][best]['mean']:+.5f}, "
        f"{art['best']['gain_over_h_span']:+.5f} against the deployed h_span. " +
        ("Worth switching." if art["best"]["gain_over_h_span"] > 0.005 else
         "No representation beats plain h_span by enough to matter -- the choice of hidden state "
         "is not a lever, which is consistent with normalisation, compression and architecture "
         "all having failed on the same endpoint."))
    print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
