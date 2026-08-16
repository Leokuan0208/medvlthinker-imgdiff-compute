#!/usr/bin/env python3
"""hole17_run.py -- HOLE 17: refit every escalation threshold against the MACRO objective.

WHAT IS FIT, AND WHERE IT LIVES IN THE SHIPPED CODE
  tau_k   (5 MCQ cells)  paper_baselines.cascade_persample -> integrated_method.pick_tau_isocost
  lam_k   (3 open cells) paper_baselines.pandora_persample  -> pandora_controller (zeta_cheap/zeta_strong).
          lam yields BOTH the draw-another reservation and the escalate reservation, so the
          "open-text verifier-confidence escalation threshold" is not a separate scalar in the
          shipped method -- it is z_strong = q_train - lam*4.57.  A DECOUPLED (z_c, z_s) family is
          swept separately as an extension.

PROTOCOL
  outer folds  : i % 5 == f, per cell -- the repo's own convention, so the incumbent arm reproduces
                 the published per-sample vectors byte-for-byte (asserted).
  HONEST number: NESTED CV.  mu is chosen on an INNER 4-fold inside each outer-train split, the
                 per-cell thresholds are refit on the full outer-train at that mu, and the policy is
                 applied to the untouched outer-test fold.
  DIAGNOSTIC   : mu chosen directly on the outer-held-out macro curve (eval-visible).  Reported
                 separately and never as the headline.
  NULL         : the identical machinery run on data whose gate<->outcome association has been
                 destroyed by a within-cell row permutation.  This is the manufacturing check the
                 arm-combination round demanded (+0.0109 macro from shuffled labels alone there).

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_run.py
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)

import integrated_method as IM
import integrated_pandora as IP
import hole17_data as HD
import hole17_engine as EN

ROOT = IM.ROOT
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
MCQ_B, OPEN_B, ORDER_B = HD.MCQ_B, HD.OPEN_B, HD.ORDER_B

K_OUTER, K_INNER = 5, 4
MUS = np.concatenate([[0.0], np.geomspace(1e-5, 2.0, 240)])
DIRECT_FLOPS = 4.57


# ===================================================================================================
# data holder
# ===================================================================================================
class Data:
    def __init__(self, verifier="ckpts/train/lora_verifier_disjoint"):
        IP.ADAPTER = verifier
        IM.OPEN_VERIFIER_DIR = verifier
        self.verifier = verifier
        self.mcq = HD.load_mcq()
        self.open = HD.load_open()
        self.n = {k: self.mcq[k]["n"] for k in MCQ_B}
        self.n.update({k: self.open[k]["n"] for k in OPEN_B})
        # the incumbent open-arm iso-accuracy TARGET: the fixed-bo8 arm's own held-out accuracy
        self.open_target = {}
        for k in OPEN_B:
            d = IM.open_bestof8(HD.OPEN_KEY[k])
            a_fix, _ = IM.heldout(d["ok7"], d["ok32"], d["gate"])
            self.open_target[k] = a_fix

    def permuted(self, seed):
        """Null data: destroy the gate<->outcome association inside every cell, keep all marginals."""
        rng = np.random.default_rng(seed)
        d = Data.__new__(Data)
        d.verifier = self.verifier; d.n = dict(self.n); d.open_target = dict(self.open_target)
        d.mcq = {}
        for k, c in self.mcq.items():
            p = rng.permutation(c["n"])
            d.mcq[k] = dict(ok7=c["ok7"][p], ok32=c["ok32"][p], margin=c["margin"], n=c["n"])
        d.open = {}
        for k, c in self.open.items():
            p = rng.permutation(c["n"])
            d.open[k] = dict(raw=c["raw"], sl=c["sl"][p], strong=c["strong"][p],
                             greedy=c["greedy"][p], n=c["n"], Nmax=c["Nmax"])
        return d


def folds(n, k, seed=None):
    if seed is None:
        return np.arange(n) % k                              # the repo's own modulo folding
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % k)


# ===================================================================================================
# one cell, one train/test split -> everything the objective needs
# ===================================================================================================
def mcq_split(c, tr, te):
    taus, a_tr, e_tr = EN.mcq_curves(c["ok7"][tr], c["ok32"][tr], c["margin"][tr])
    return dict(taus=taus, acc_tr=a_tr, cost_tr=EN.mcq_cost(e_tr), esc_tr=e_tr,
                tr=tr, te=te, target=float(c["ok32"][tr].mean()))


def mcq_test(c, te, tau):
    ok, e = EN.mcq_apply(c["ok7"][te], c["ok32"][te], c["margin"][te], tau)
    return ok, e


def open_split(c, tr, te):
    iso = EN.open_fit_calibrator(c["raw"][tr], c["sl"][tr])
    cal_tr_pool = iso.predict(c["raw"][tr].ravel())
    q_tr = float(c["strong"][tr].mean())
    cal_tr = iso.predict(c["raw"][tr].ravel()).reshape(c["raw"][tr].shape)
    cal_te = iso.predict(c["raw"][te].ravel()).reshape(c["raw"][te].shape)
    _, a_tr, ct_tr, mN_tr, e_tr = EN.open_curves(cal_tr_pool, q_tr, cal_tr, c["raw"][tr],
                                                 c["sl"][tr], c["strong"][tr])
    # test curve for every lambda, precomputed once (lets any mu be scored for free)
    A = np.empty(len(EN.LAMS)); C = np.empty(len(EN.LAMS))
    MN = np.empty(len(EN.LAMS)); E = np.empty(len(EN.LAMS))
    OKV, ESV, NV = [], [], []
    for j, lam in enumerate(EN.LAMS):
        z_c = EN.zeta_cheap_exact(cal_tr_pool, lam)
        from pandora_controller import zeta_strong
        z_s = zeta_strong(q_tr, lam)
        N, e, ok = HD.pandora_vec(cal_te, c["raw"][te], c["sl"][te], c["strong"][te], z_c, z_s)
        A[j] = ok.mean(); MN[j] = N.mean(); E[j] = e.mean()
        C[j] = MN[j] * HD.C_CHEAP + E[j] * HD.C_STRONG
        OKV.append(ok); ESV.append(e); NV.append(N)
    return dict(acc_tr=a_tr, cost_tr=ct_tr, esc_tr=e_tr, meanN_tr=mN_tr,
                acc_te=A, cost_te=C, esc_te=E, meanN_te=MN,
                ok_te=OKV, esc_vec_te=ESV, N_vec_te=NV, tr=tr, te=te)


# ===================================================================================================
# policy selection given a split
# ===================================================================================================
def mcq_pick_incumbent(sp):
    ok = sp["acc_tr"] >= sp["target"] - 1e-12
    if ok.any():
        idx = np.where(ok)[0]; return int(idx[np.argmin(sp["esc_tr"][idx])])
    return int(np.argmax(sp["acc_tr"]))


def open_pick_incumbent(sp, target):
    ok = sp["acc_tr"] >= target - EN.ISO_TOL
    if ok.any():
        idx = np.where(ok)[0]; return int(idx[np.argmin(sp["cost_tr"][idx])])
    return int(np.argmax(sp["acc_tr"]))


def pick_lagrange(acc_tr, cost_tr, mu):
    return int(np.argmax(acc_tr - mu * cost_tr))


# ===================================================================================================
# a full held-out pass over one fold assignment
# ===================================================================================================
def heldout_pass(data, fa, mus=MUS, want_items=False):
    """Returns per-cell held-out (acc, cost) for the INCUMBENT rule and for every mu in `mus`,
    plus (optionally) the per-item delivered vectors for each."""
    res = {"incumbent": {k: dict(ok=np.zeros(data.n[k]), esc=np.zeros(data.n[k]),
                                 N=np.zeros(data.n[k]), pol=[]) for k in ORDER_B},
           "mu": {k: dict(acc=np.zeros(len(mus)), cost=np.zeros(len(mus)),
                          esc=np.zeros(len(mus)), pol=np.zeros((K_OUTER, len(mus)), int))
                  for k in ORDER_B}}
    if want_items:
        for k in ORDER_B:
            res["mu"][k]["ok_items"] = np.zeros((len(mus), data.n[k]))
            res["mu"][k]["esc_items"] = np.zeros((len(mus), data.n[k]))
            res["mu"][k]["N_items"] = np.zeros((len(mus), data.n[k]))

    for k in ORDER_B:
        c = data.mcq[k] if k in MCQ_B else data.open[k]
        f_of = fa[k]
        for f in range(K_OUTER):
            te = f_of == f; tr = ~te
            if tr.sum() < 2 or te.sum() < 1: continue
            if k in MCQ_B:
                sp = mcq_split(c, tr, te)
                j0 = mcq_pick_incumbent(sp)
                ok, e = mcq_test(c, te, sp["taus"][j0])
                res["incumbent"][k]["ok"][te] = ok; res["incumbent"][k]["esc"][te] = e
                res["incumbent"][k]["pol"].append(float(sp["taus"][j0]))
                js = np.array([pick_lagrange(sp["acc_tr"], sp["cost_tr"], m) for m in mus])
                res["mu"][k]["pol"][f] = js
                for j in np.unique(js):
                    okj, ej = mcq_test(c, te, sp["taus"][j])
                    sel = js == j
                    res["mu"][k]["acc"][sel] += okj.sum(); res["mu"][k]["esc"][sel] += ej.sum()
                    res["mu"][k]["cost"][sel] += (EN.mcq_cost(ej)).sum()
                    if want_items:
                        res["mu"][k]["ok_items"][np.ix_(np.where(sel)[0], np.where(te)[0])] = okj
                        res["mu"][k]["esc_items"][np.ix_(np.where(sel)[0], np.where(te)[0])] = ej
            else:
                sp = open_split(c, tr, te)
                j0 = open_pick_incumbent(sp, data.open_target[k])
                res["incumbent"][k]["ok"][te] = sp["ok_te"][j0]
                res["incumbent"][k]["esc"][te] = sp["esc_vec_te"][j0]
                res["incumbent"][k]["N"][te] = sp["N_vec_te"][j0]
                res["incumbent"][k]["pol"].append(float(EN.LAMS[j0]))
                js = np.array([pick_lagrange(sp["acc_tr"], sp["cost_tr"], m) for m in mus])
                res["mu"][k]["pol"][f] = js
                for j in np.unique(js):
                    sel = js == j
                    res["mu"][k]["acc"][sel] += sp["ok_te"][j].sum()
                    res["mu"][k]["esc"][sel] += sp["esc_vec_te"][j].sum()
                    res["mu"][k]["cost"][sel] += (sp["N_vec_te"][j] * HD.C_CHEAP +
                                                  sp["esc_vec_te"][j] * HD.C_STRONG).sum()
                    if want_items:
                        ii = np.ix_(np.where(sel)[0], np.where(te)[0])
                        res["mu"][k]["ok_items"][ii] = sp["ok_te"][j]
                        res["mu"][k]["esc_items"][ii] = sp["esc_vec_te"][j]
                        res["mu"][k]["N_items"][ii] = sp["N_vec_te"][j]
        nk = data.n[k]
        res["mu"][k]["acc"] /= nk; res["mu"][k]["cost"] /= nk; res["mu"][k]["esc"] /= nk
        inc = res["incumbent"][k]
        inc["acc"] = float(inc["ok"].mean()); inc["esc_rate"] = float(inc["esc"].mean())
        inc["cost"] = float((EN.mcq_cost(inc["esc"])).mean() if k in MCQ_B
                            else (inc["N"] * HD.C_CHEAP + inc["esc"] * HD.C_STRONG).mean())
    return res


def aggregate(per_cell_acc, per_cell_cost, n, w="macro"):
    if w == "macro":
        wt = {k: 1.0 / len(ORDER_B) for k in ORDER_B}
    else:
        N = sum(n[k] for k in ORDER_B); wt = {k: n[k] / N for k in ORDER_B}
    return (sum(per_cell_acc[k] * wt[k] for k in ORDER_B),
            sum(per_cell_cost[k] * wt[k] for k in ORDER_B))


def anchor_mu(res, n, mus, weighting, mode):
    """mode 'iso_acc': min aggregate cost s.t. aggregate acc >= incumbent's.
       mode 'iso_cost': max aggregate acc s.t. aggregate cost <= incumbent's."""
    inc_a, inc_c = aggregate({k: res["incumbent"][k]["acc"] for k in ORDER_B},
                             {k: res["incumbent"][k]["cost"] for k in ORDER_B}, n, weighting)
    A = np.zeros(len(mus)); C = np.zeros(len(mus))
    wt = ({k: 1.0 / len(ORDER_B) for k in ORDER_B} if weighting == "macro"
          else {k: n[k] / sum(n[j] for j in ORDER_B) for k in ORDER_B})
    for k in ORDER_B:
        A += res["mu"][k]["acc"] * wt[k]; C += res["mu"][k]["cost"] * wt[k]
    if mode == "iso_acc":
        ok = A >= inc_a - 1e-12
        j = int(np.where(ok)[0][np.argmin(C[ok])]) if ok.any() else int(np.argmax(A))
    else:
        ok = C <= inc_c + 1e-12
        j = int(np.where(ok)[0][np.argmax(A[ok])]) if ok.any() else int(np.argmin(C))
    return j, float(mus[j]), dict(inc_acc=inc_a, inc_cost=inc_c, curve_acc=A, curve_cost=C)


# ===================================================================================================
# NULL TEST 2 -- the re-implemented incumbent must reproduce the PUBLISHED per-sample vectors
# ===================================================================================================
def harness_null_test(data):
    fa = {k: folds(data.n[k], K_OUTER) for k in ORDER_B}
    res = heldout_pass(data, fa, mus=np.array([0.0]))
    vec = np.load(os.path.join(ART, "_selector_rerun_parts", "vec_disjoint.npz"))
    stored_esc = json.load(open(os.path.join(ART, "_selector_rerun_parts",
                                             "summary_disjoint.json")))["escalation"]["per_cell"]
    out = {}
    for k in ORDER_B:
        live = res["incumbent"][k]["ok"]
        ref = np.asarray(vec[f"{k}|method_compute_lean"], float)
        out[k] = dict(n=int(len(live)), item_mismatches=int(np.abs(live - ref).sum()),
                      acc_live=float(live.mean()), acc_published=float(ref.mean()),
                      esc_live=round(res["incumbent"][k]["esc_rate"], 6),
                      esc_published=stored_esc[k],
                      esc_absdev=abs(res["incumbent"][k]["esc_rate"] - stored_esc[k]))
    out["_max_abs_acc_dev"] = max(abs(out[k]["acc_live"] - out[k]["acc_published"]) for k in ORDER_B)
    out["_max_abs_esc_dev"] = max(out[k]["esc_absdev"] for k in ORDER_B)
    out["_total_item_mismatches"] = sum(out[k]["item_mismatches"] for k in ORDER_B)
    return out, res




# ===================================================================================================
# NESTED CV -- mu chosen on an INNER split, never on the fold it is scored on
# ===================================================================================================
def _subset(data, keep):
    d = Data.__new__(Data)
    d.verifier = data.verifier; d.open_target = dict(data.open_target)
    d.mcq = {k: dict(ok7=data.mcq[k]["ok7"][keep[k]], ok32=data.mcq[k]["ok32"][keep[k]],
                     margin=data.mcq[k]["margin"][keep[k]], n=int(keep[k].sum())) for k in MCQ_B}
    d.open = {k: dict(raw=data.open[k]["raw"][keep[k]], sl=data.open[k]["sl"][keep[k]],
                      strong=data.open[k]["strong"][keep[k]], greedy=data.open[k]["greedy"][keep[k]],
                      n=int(keep[k].sum()), Nmax=data.open[k]["Nmax"]) for k in OPEN_B}
    d.n = {k: d.mcq[k]["n"] if k in MCQ_B else d.open[k]["n"] for k in ORDER_B}
    return d


def nested_pass(data, fa_outer, anchors, mus=MUS, inner_seed=None):
    """anchors: list of (label, weighting, mode).  Returns per-anchor held-out per-item vectors."""
    out = {lab: {k: dict(ok=np.zeros(data.n[k]), esc=np.zeros(data.n[k]), N=np.zeros(data.n[k]))
                 for k in ORDER_B} for lab, _, _ in anchors}
    chosen = {lab: [] for lab, _, _ in anchors}
    for f in range(K_OUTER):
        keep = {k: fa_outer[k] != f for k in ORDER_B}
        sub = _subset(data, keep)
        fa_in = {k: folds(sub.n[k], K_INNER, None if inner_seed is None else inner_seed * 97 + f)
                 for k in ORDER_B}
        rin = heldout_pass(sub, fa_in, mus=mus)
        for lab, wgt, mode in anchors:
            j, mu, _ = anchor_mu(rin, sub.n, mus, wgt, mode)
            chosen[lab].append(mu)
            for k in ORDER_B:
                c = data.mcq[k] if k in MCQ_B else data.open[k]
                tr = keep[k]; te = ~tr
                if k in MCQ_B:
                    sp = mcq_split(c, tr, te)
                    jj = pick_lagrange(sp["acc_tr"], sp["cost_tr"], mu)
                    ok, e = mcq_test(c, te, sp["taus"][jj])
                    out[lab][k]["ok"][te] = ok; out[lab][k]["esc"][te] = e
                else:
                    sp = open_split(c, tr, te)
                    jj = pick_lagrange(sp["acc_tr"], sp["cost_tr"], mu)
                    out[lab][k]["ok"][te] = sp["ok_te"][jj]
                    out[lab][k]["esc"][te] = sp["esc_vec_te"][jj]
                    out[lab][k]["N"][te] = sp["N_vec_te"][jj]
    for lab in out:
        for k in ORDER_B:
            r = out[lab][k]
            r["acc"] = float(r["ok"].mean()); r["esc_rate"] = float(r["esc"].mean())
            r["cost"] = float((EN.mcq_cost(r["esc"])).mean() if k in MCQ_B
                              else (r["N"] * HD.C_CHEAP + r["esc"] * HD.C_STRONG).mean())
        out[lab]["_mu_per_outer_fold"] = chosen[lab]
    return out


def diagnostic_pass(data, fa_outer, anchors, mus=MUS, res=None):
    """EVAL-VISIBLE: mu chosen on the outer-held-out curve itself.  Upper bound, never the headline."""
    if res is None:
        res = heldout_pass(data, fa_outer, mus=mus, want_items=True)
    out = {}
    for lab, wgt, mode in anchors:
        j, mu, aux = anchor_mu(res, data.n, mus, wgt, mode)
        rec = {}
        for k in ORDER_B:
            ok = res["mu"][k]["ok_items"][j]; e = res["mu"][k]["esc_items"][j]
            N = res["mu"][k]["N_items"][j] if k in OPEN_B else None
            rec[k] = dict(ok=ok, esc=e, N=N, acc=float(res["mu"][k]["acc"][j]),
                          cost=float(res["mu"][k]["cost"][j]),
                          esc_rate=float(res["mu"][k]["esc"][j]))
        rec["_mu"] = mu
        out[lab] = rec
    return out, res


# ===================================================================================================
# aggregation / stats
# ===================================================================================================
def summarise(cellrec, n):
    a = {k: cellrec[k]["acc"] for k in ORDER_B}
    c = {k: cellrec[k]["cost"] for k in ORDER_B}
    ma, mc = aggregate(a, c, n, "macro"); pa, pc = aggregate(a, c, n, "pooled")
    return dict(macro_acc=ma, macro_cost=mc, pooled_acc=pa, pooled_cost=pc,
                macro_x_direct=mc / DIRECT_FLOPS, pooled_x_direct=pc / DIRECT_FLOPS,
                per_cell_acc=a, per_cell_cost=c,
                per_cell_esc={k: cellrec[k]["esc_rate"] for k in ORDER_B})


def boot_macro_delta(vecA, vecB, n, nboot=10000, seed=20260815):
    """Paired ITEM bootstrap of the macro (1/8 per cell) accuracy delta.  Multinomial over unique
    per-item outcome patterns within each cell -- MAH.cell_boot_means, verbatim protocol."""
    rng = np.random.default_rng(seed)
    dist = np.zeros(nboot); pt = 0.0
    for k in ORDER_B:
        mat = np.column_stack([vecA[k], vecB[k]])
        pats, cnt = np.unique(mat, axis=0, return_counts=True)
        nn = mat.shape[0]
        m = (rng.multinomial(nn, cnt / nn, size=nboot) @ pats) / nn
        dist += (m[:, 0] - m[:, 1]) / len(ORDER_B)
        pt += (vecA[k].mean() - vecB[k].mean()) / len(ORDER_B)
    lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
    return dict(delta=float(pt), ci95=[lo, hi], sig=bool(lo > 0 or hi < 0),
                verdict="WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE")
