#!/usr/bin/env python3
"""head_arch_transfer.py -- architectures scored on OUT-OF-DOMAIN TRANSFER, not in-domain CV.

THE WALL.  Measured 2026-08-19 on three newly built cells, frozen head, deployed readout:
    kvasir_x1 (IN-domain)   head 0.3629 vs greedy 0.2849   +0.0781 WIN
    vqamed C4 (new domain)  head 0.0688 vs greedy 0.0947   -0.0259 LOSS
    omnimed   (new domain)  head 0.3396 vs greedy 0.3885   -0.0489 LOSS, sel_eff 0.5897 BELOW the
                                                            0.5933 random-pick floor
The head is a domain-specific component, not a verifier.  Every architecture change tried so far was
selected on IN-DOMAIN CV, which is exactly the metric that cannot see this.  This script selects on
transfer instead: train on three of the four training domains, score on the fourth, and rank by
`head - string prior` so vocabulary memorisation is never counted as skill.

WHAT IS TRIED, and why each is motivated by the failure rather than by taste:

  raw            the deployed baseline: global z-score, 3584-d, one hidden layer.
  poolnorm       standardise each candidate AGAINST ITS OWN QUESTION'S POOL (subtract the pool mean,
                 divide by the pool sd) instead of against global train statistics.  A domain shift
                 is largely a constant offset in activation space; a within-pool statistic cancels
                 it exactly, and the quantity the head needs -- "is this candidate better than its
                 siblings" -- is preserved.  This is the most targeted fix available.
  poolnorm+raw   both views concatenated, so absolute evidence is still available if it helps.
  pca128 / 32    compress before the head.  The probe-transfer literature reports that "under
                 distribution shift, structured and compressed features are more robust", and our
                 own regularisation sweep found a 32-unit head matches a 1024-unit one -- the signal
                 is low-dimensional, so the extra dimensions can only carry domain identity.
  rankonly       replace every feature with its WITHIN-POOL RANK.  Discards all absolute scale, so
                 nothing domain-specific survives by construction.  The strongest form of the idea;
                 if it wins, the head never needed the raw activations.
  domadv         domain-adversarial: a second head predicts which dataset a row came from through a
                 gradient-reversal layer, so the shared trunk is pushed to drop domain identity.

  python3 src/training_methods/head_arch_transfer.py --threads 8 --seeds 3
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import head_sweep as HS
import torch
import torch.nn as nn

ARCHS = ["raw", "poolnorm", "poolnorm+raw", "pca128", "pca32", "rankonly", "domadv"]


def norm(s):
    return str(s).strip().lower().rstrip(".")


def pool_stats(X, qid):
    """Per-question mean and sd over the candidates of that question."""
    byq = defaultdict(list)
    for i, q in enumerate(qid):
        byq[q].append(i)
    M = np.empty_like(X); S = np.empty_like(X)
    for q, ii in byq.items():
        ii = np.array(ii)
        sub = X[ii]
        M[ii] = sub.mean(0, keepdims=True)
        S[ii] = sub.std(0, keepdims=True) + 1e-6
    return M, S


def within_pool_rank(X, qid):
    """Each feature replaced by its rank inside the question's pool, scaled to [0,1]."""
    byq = defaultdict(list)
    for i, q in enumerate(qid):
        byq[q].append(i)
    R = np.empty_like(X)
    for q, ii in byq.items():
        ii = np.array(ii)
        sub = X[ii]
        if len(ii) == 1:
            R[ii] = 0.5
            continue
        order = np.argsort(sub, axis=0)
        rk = np.empty_like(sub)
        ar = np.arange(len(ii), dtype=np.float32)[:, None]
        np.put_along_axis(rk, order, np.repeat(ar, sub.shape[1], axis=1), axis=0)
        R[ii] = rk / (len(ii) - 1)
    return R


def build(arch, X, qid, pca=None):
    if arch == "raw":
        return X, None
    if arch in ("poolnorm", "poolnorm+raw"):
        M, S = pool_stats(X, qid)
        Z = (X - M) / S
        return (np.concatenate([X, Z], 1) if arch == "poolnorm+raw" else Z), None
    if arch.startswith("pca"):
        k = int(arch[3:])
        if pca is None:
            Xc = X - X.mean(0, keepdims=True)
            # randomised SVD is plenty here and avoids a 3584x3584 covariance
            g = np.random.default_rng(0).standard_normal((X.shape[1], k + 16)).astype(np.float32)
            Q, _ = np.linalg.qr(Xc @ g)
            B = Q.T @ Xc
            _, _, Vt = np.linalg.svd(B, full_matrices=False)
            pca = Vt[:k].T.astype(np.float32)
        return (X - X.mean(0, keepdims=True)) @ pca, pca
    if arch == "rankonly":
        return within_pool_rank(X, qid), None
    if arch == "domadv":
        return X, None
    raise ValueError(arch)


class DomAdv(nn.Module):
    """Shared trunk -> score head, plus a domain head behind a gradient-reversal layer."""

    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lam * g, None

    def __init__(self, d, hidden, ndom, lam=0.3):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.GELU())
        self.score = nn.Linear(hidden, 1)
        self.dom = nn.Linear(hidden, ndom)
        self.lam = lam

    def forward(self, x, with_dom=False):
        h = self.trunk(x)
        s = self.score(h).squeeze(-1)
        if not with_dom:
            return s
        return s, self.dom(DomAdv.GRL.apply(h, self.lam))


def fit_domadv(X, y, dom, hidden=256, epochs=30, lr=1e-3, wd=1e-2, bs=256, seed=0):
    torch.manual_seed(seed)
    ndom = int(dom.max()) + 1
    m = DomAdv(X.shape[1], hidden, ndom)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd, foreach=False)
    Xt, yt, dt = torch.tensor(X), torch.tensor(y), torch.tensor(dom, dtype=torch.long)
    pos = float(y.mean())
    lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6)))
    ce = nn.CrossEntropyLoss()
    n = len(yt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            s, dlog = m(Xt[j], with_dom=True)
            loss = lf(s, yt[j]) + ce(dlog, dt[j])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


def pick_stats(sc, y, qid, mask):
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
        tot[na[i]] += 1; pos[na[i]] += int(y[i])
    gp = float(y[idx].mean()) if len(idx) else 0.5
    return np.array([(pos[a] + gp * 2) / (tot[a] + 2) if tot[a] else gp for a in na])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--archs", nargs="*", default=ARCHS)
    ap.add_argument("--out", default=os.path.join(
        HS.OUTDIR, "head_arch_transfer_2026-08-19.json"))
    A = ap.parse_args()
    torch.set_num_threads(A.threads)

    H, y, qid, img, ds = HS.load_train()
    rows = []
    for sh in (0, 1):
        rows += json.load(open(os.path.join(HS.FEATS, f"generator_train_s{sh}of2.meta.json")))["rows"]
    rows = [r for r in rows if r.get("n_tok", -1) > 0]
    na = np.array([norm(r["na"]) for r in rows])
    X0 = HS.assemble(H, HS.BASE)
    sets = sorted(set(ds))
    dcode = {d: i for i, d in enumerate(sets)}
    dom = np.array([dcode[d] for d in ds])
    print(f"rows {len(y)} | domains {sets}", flush=True)

    art = {"title": "Architectures ranked by OUT-OF-DOMAIN transfer, not in-domain CV",
           "date": "2026-08-19", "no_fabricated_numbers": True,
           "endpoint": "leave-one-domain-out sel_eff on the held-out domain, MINUS the string-prior "
                       "null fitted on the same training rows. Selecting on in-domain CV is what "
                       "produced a head that loses to greedy on two of three new cells.",
           "results": {}}

    for arch in A.archs:
        art["results"][arch] = {}
        pca = None
        for d in sets:
            isD = ds == d
            tr = ~isD
            Xa, pca = build(arch, X0, qid, pca if arch.startswith("pca") else None)
            sp = pick_stats(string_prior(y, na, tr), y, qid, isD)
            vals = []
            for s in range(A.seeds):
                mu, sg = Xa[tr].mean(0), Xa[tr].std(0) + 1e-6
                Xs = (Xa - mu) / sg
                if arch == "domadv":
                    m = fit_domadv(Xs[tr], y[tr], dom[tr], hidden=A.hidden, seed=s)
                    with torch.no_grad():
                        sc = m(torch.tensor(Xs)).numpy()
                else:
                    m = HS.fit(Xs[tr], y[tr], qid[tr], None, objective="bce", hidden=A.hidden,
                               wd=1e-2, epochs=30, seed=s)
                    sc = np.empty(len(Xs), dtype=np.float32)
                    for b0 in range(0, len(Xs), 4096):
                        b1 = min(b0 + 4096, len(Xs))
                        sc[b0:b1] = HS.predict(m, Xs[b0:b1])
                vals.append(pick_stats(sc, y, qid, isD))
            out = float(np.mean(vals))
            art["results"][arch][d] = {"out_domain_sel_eff": out, "sd": float(np.std(vals)),
                                       "string_prior": sp, "head_minus_prior": out - sp,
                                       "n_features": int(Xa.shape[1])}
            print(f"  [{arch:13}] {d:22} out {out:.4f} (sd {np.std(vals):.4f}) "
                  f"prior {sp:.4f}  head-prior {out - sp:+.4f}", flush=True)
            json.dump(art, open(A.out, "w"), indent=1)
        hmp = [v["head_minus_prior"] for v in art["results"][arch].values()]
        art["results"][arch]["MEAN_head_minus_prior"] = float(np.mean(hmp))
        print(f"  [{arch:13}] MEAN head-minus-prior over 4 held-out domains: {np.mean(hmp):+.4f}\n",
              flush=True)
        json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
