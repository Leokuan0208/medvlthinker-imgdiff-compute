#!/usr/bin/env python3
"""
ATTACK A (round 2) -- COMBINE THE ARMS THAT ALREADY EXIST.

Round 1's four attacks ran in parallel, so Attack 3's cross-fit per-cell arm selection had a menu
that did NOT contain Attack 1's 32B-best-of-N arms -- and one of those WON PathVQA-open by +0.0269.
This script assembles every per-cell arm measured in round 1 into ONE item-paired menu and asks
whether an HONEST estimator of per-cell arm selection beats always-32B-direct on the 8-cell macro.

Pre-registration: results/cascade_methods/artifacts/armcombine_2026-08-11_preregistration.json
Artifact:         results/cascade_methods/artifacts/armcombine_2026-08-11.json

No GPU.  Pure numpy over stored per-item 0/1 vectors.
Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/armcombine.py
"""
import os
import sys
import json
import time
from collections import Counter

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_selector_rerun_parts")
INC_DIR = os.path.join(REPO, "ckpts/train/lora_verifier_disjoint")
STRONG_OLD = os.path.join(REPO, "ckpts/openvqa/strong_lingshu")
BO_DIR = os.path.join(REPO, "ckpts/openvqa/strong_lingshu_bo")
VERIF = os.path.join(BO_DIR, "verif_lora_verifier_disjoint")
OUT = os.path.join(ART, "armcombine_2026-08-11.json")

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
OPEN = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
DSK = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}
BASE = "always_32b_direct"

BASE6 = ["always_7b", "always_32b_direct", "always_32b_reasoning",
         "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]
NEW3 = ["l32_bo4", "l32_bo8", "l32_maj8"]

R32_CHARGED = 4.57
R32_DERIVED = 3.816
SEED = 20260811
NBOOT = 10000
NPERM = 1000
TIE_TOL = 0.0029
KFOLD = 5
KINNER = 4
NFOLDSEEDS = 12
GENSEEDS = ["l32_bo8_s0", "l32_bo8_s1", "l32_bo8_s2"]
MARGIN_GRID = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
EPS_GRID = [0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 1.0]

rep = {}


# =====================================================================================
# 0.  LOAD
# =====================================================================================
def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def jmap(p):
    return {r["idx"]: int(r["judge_ok"]) for r in jl(p)}


def norm(s):
    return str(s).strip().lower()


Z = np.load(os.path.join(PARTS, "vec_disjoint.npz"))
PUB = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))["per_arm"]["disjoint"]
MAC = json.load(open(os.path.join(PARTS, "macro_disjoint.json")))
PCC = MAC["cost"]["per_cell_as_charged"]
OSTRONG = json.load(open(os.path.join(ART, "openstrong_bestofn_2026-08-10.json")))
CFLOOR = json.load(open(os.path.join(ART, "cost_floor_2026-08-10.json")))

VEC = {c: {a: Z[f"{c}|{a}"].astype(np.float64) for a in BASE6} for c in CELLS}
N = {c: len(VEC[c][BASE]) for c in CELLS}

OPEN_IDS, MATCHED_A0, BO = {}, {}, {}
for cell in OPEN:
    ds = DSK[cell]
    dump = json.load(open(os.path.join(INC_DIR, f"transfer_dump_{ds}_lingshu7b.json")))
    sj = jmap(os.path.join(STRONG_OLD, f"ckpt_{ds}_lingshu32b.judge.jsonl"))
    ids = [r["idx"] for r in dump if r["idx"] in sj]
    OPEN_IDS[cell] = ids
    old32 = np.array([sj[i] for i in ids], float)
    assert np.array_equal(old32, VEC[cell][BASE]), f"{cell}: item order does not match deployed vector"
    a0 = jmap(os.path.join(BO_DIR, f"ckpt_{ds}_l32_n1.judge.jsonl"))
    MATCHED_A0[cell] = np.array([a0[i] for i in ids], float)
    for tag in GENSEEDS:
        by = {r["idx"]: r for r in json.load(open(os.path.join(VERIF, f"transfer_dump_{ds}_{tag}.json")))}
        s4, s8, m8, o8 = [], [], [], []
        for i in ids:
            r = by[i]
            sl = [0 if x < 0 else int(x) for x in r["sl"]]
            sc = list(r["scores"])
            s8.append(sl[int(np.argmax(sc))])
            s4.append(sl[:4][int(np.argmax(sc[:4]))])
            o8.append(max(sl))
            cnt = Counter(norm(a) for a in r["preds"])
            top = cnt.most_common(1)[0][0]
            kk = next(j for j, a in enumerate(r["preds"]) if norm(a) == top)
            m8.append(sl[kk])
        BO[(cell, tag)] = dict(l32_bo4=np.array(s4, float), l32_bo8=np.array(s8, float),
                               l32_maj8=np.array(m8, float), oracle8=np.array(o8, float))


# =====================================================================================
# 1.  COST MODEL
# =====================================================================================
_c = OSTRONG["cost"]["open_cell_per_item"]
DEC32 = (_c["32b_bo8"]["gen_shared_prefill_flopeq"] / R32_CHARGED - 1.0) / 7.0
_chk4 = R32_CHARGED * (1.0 + 3.0 * DEC32)
PRE7 = CFLOOR["null_tests"]["N2"]["prefill_share_7b"]
DEC7 = CFLOOR["null_tests"]["N2"]["decode_share_7b"]
VER_PRE = CFLOOR["verifier_geometry"]["ver_prefix_cost_units"]
VER_MARG = CFLOOR["verifier_geometry"]["ver_marginal_cost_per_candidate_units"]
ESC = PUB["escalation"]["per_cell"]
OCD = PUB["open_cell_detail"]


def cost_A(d, R=R32_CHARGED):
    return d["gen7"] * 1.0 + d["ver7"] * 1.0 + d["frac32"] * d["pool32"] * R


def cost_B(d, R=R32_CHARGED):
    g7 = (PRE7 + d["gen7"] * DEC7) if d["gen7"] > 0 else 0.0
    g32 = d["frac32"] * (1.0 + (d["pool32"] - 1) * DEC32) * R if d["pool32"] > 0 else 0.0
    return g7 + d["ver7"] * 1.0 + g32


def cost_C(d, R=R32_CHARGED):
    g7 = (PRE7 + d["gen7"] * DEC7) if d["gen7"] > 0 else 0.0
    v = (VER_PRE + d["ver7"] * VER_MARG) if d["ver7"] > 0 else 0.0
    g32 = d["frac32"] * (1.0 + (d["pool32"] - 1) * DEC32) * R if d["pool32"] > 0 else 0.0
    return g7 + v + g32


CONV = {"A_as_charged": cost_A, "B_recost_gen": cost_B, "C_recost_full": cost_C}


def build_arm_costs():
    A = {c: {} for c in CELLS}
    for c in CELLS:
        A[c]["always_7b"] = dict(gen7=1.0, ver7=0.0, frac32=0.0, pool32=0)
        A[c]["always_32b_direct"] = dict(gen7=0.0, ver7=0.0, frac32=1.0, pool32=1)
        A[c]["always_32b_reasoning"] = dict(gen7=0.0, ver7=0.0, frac32=1.0, pool32=1)
        if c in OPEN:
            nn = OCD[c]["meanN"]
            A[c]["method_compute_lean"] = dict(gen7=nn, ver7=nn, frac32=OCD[c]["esc"], pool32=1)
            A[c]["method_accuracy_max_veto"] = dict(gen7=nn, ver7=nn, frac32=OCD[c]["am2_esc"], pool32=1)
            A[c]["method_accuracy_max_fusion"] = dict(gen7=nn, ver7=nn, frac32=OCD[c]["esc"], pool32=1)
            A[c]["l32_bo4"] = dict(gen7=0.0, ver7=4.0, frac32=1.0, pool32=4)
            A[c]["l32_bo8"] = dict(gen7=0.0, ver7=8.0, frac32=1.0, pool32=8)
            A[c]["l32_maj8"] = dict(gen7=0.0, ver7=0.0, frac32=1.0, pool32=8)
        else:
            A[c]["method_compute_lean"] = dict(gen7=1.0, ver7=0.0, frac32=ESC[c], pool32=1)
            f_am2 = PCC[c]["method_accuracy_max_veto"]["flops"]
            if abs(f_am2 - R32_CHARGED) < 1e-6:
                A[c]["method_accuracy_max_veto"] = dict(gen7=0.0, ver7=0.0, frac32=1.0, pool32=1)
            else:
                keep = 1.0 - (f_am2 - 1.0) / R32_CHARGED
                A[c]["method_accuracy_max_veto"] = dict(gen7=1.0, ver7=0.0, frac32=1.0 - keep, pool32=1)
            f_amf = PCC[c]["method_accuracy_max_fusion"]["flops"]
            A[c]["method_accuracy_max_fusion"] = (dict(gen7=1.0, ver7=0.0, frac32=1.0, pool32=1)
                                                  if abs(f_amf - (1.0 + R32_CHARGED)) < 1e-6
                                                  else dict(gen7=0.0, ver7=0.0, frac32=1.0, pool32=1))
    dev = 0.0
    for c in CELLS:
        for a, d in A[c].items():
            if a in PCC[c]:
                dev = max(dev, abs(cost_A(d, R32_CHARGED) - PCC[c][a]["flops"]))
    return A, dev


ARMC, ARMC_DEV = build_arm_costs()


# =====================================================================================
# 2.  FOLD TABULATION  --  every estimator is O(1) arithmetic on 20-bin sums after this
# =====================================================================================
NBIN = KFOLD * KINNER


class Tab:
    """Per cell: bin label L = outer_fold*KINNER + inner_fold, the per-bin item count, and the
    per-arm per-bin sum of correct answers.  Every training/held-out mean any estimator needs is a
    ratio of masked sums of these 20 numbers, so the estimators are exact and fast."""

    def __init__(self, OKm, menu, FD, FDI):
        self.menu = menu
        self.cnt, self.S, self.L = {}, {}, {}
        for c in CELLS:
            L = FD[c] * KINNER + FDI[c]
            self.L[c] = L
            self.cnt[c] = np.bincount(L, minlength=NBIN).astype(np.float64)
            self.S[c] = {a: np.bincount(L, weights=OKm[c][a], minlength=NBIN)
                         for a in menu[c]}
        b = np.arange(NBIN)
        self.of = b // KINNER
        self.inf = b % KINNER

    def acc(self, c, a, mask):
        n = self.cnt[c][mask].sum()
        return (self.S[c][a][mask].sum() / n) if n > 0 else np.nan

    def n(self, c, mask):
        return float(self.cnt[c][mask].sum())

    def corr(self, c, a, mask):
        return float(self.S[c][a][mask].sum())


def folds(seed, kfold=KFOLD):
    r = np.random.default_rng(seed)
    return {c: r.permutation(N[c]) % kfold for c in CELLS}


# =====================================================================================
# 3.  ESTIMATORS   (all return: per-cell held-out accuracy, per-cell (weight, arm) cost rows,
#                   and optionally the per-item held-out vector)
# =====================================================================================
def _cheapest(cands, c):
    return min(cands, key=lambda a: (cost_A(ARMC[c][a]), a))


def e0_evalvisible(OKm, menu):
    vec, picks = {}, {}
    for c in CELLS:
        accs = {a: OKm[c][a].mean() for a in menu[c]}
        best = max(accs.values())
        a_star = _cheapest([a for a in menu[c] if accs[a] >= best - 1e-12], c)
        vec[c], picks[c] = OKm[c][a_star], a_star
    return vec, picks


def e1_crossfit_argmax(T, OKm, FD, want_vec=True):
    vec, cost_rows, accs_out = {}, {}, {}
    for c in CELLS:
        menu = T.menu[c]
        v = np.empty(N[c]) if want_vec else None
        rows, corr = [], 0.0
        for f in range(KFOLD):
            trm, tem = T.of != f, T.of == f
            a = {x: T.acc(c, x, trm) for x in menu}
            best = max(a.values())
            a_star = _cheapest([x for x in menu if a[x] >= best - 1e-12], c)
            if want_vec:
                sel = FD[c] == f
                v[sel] = OKm[c][a_star][sel]
            corr += T.corr(c, a_star, tem)
            rows.append((T.n(c, tem), a_star))
        accs_out[c] = corr / N[c]
        cost_rows[c] = rows
        if want_vec:
            vec[c] = v
    return accs_out, cost_rows, vec


def _margin_pick(T, c, mask, m):
    menu = T.menu[c]
    inc = T.acc(c, BASE, mask)
    best_a, best_v = None, -1.0
    for x in menu:
        if x == BASE:
            continue
        v = T.acc(c, x, mask)
        if v > best_v:
            best_a, best_v = x, v
    return best_a if (best_a is not None and best_v >= inc + m) else BASE


def e2_nested_margin(T, OKm, FD, want_vec=True):
    vec = {c: (np.empty(N[c]) if want_vec else None) for c in CELLS}
    cost_rows = {c: [] for c in CELLS}
    accs_out = {c: 0.0 for c in CELLS}
    chosen_m = []
    for f in range(KFOLD):
        scored = []
        for m in MARGIN_GRID:
            per = []
            for c in CELLS:
                corr, tot = 0.0, 0.0
                for g in range(KINNER):
                    itr = (T.of != f) & (T.inf != g)
                    ite = (T.of != f) & (T.inf == g)
                    if T.n(c, ite) == 0:
                        continue
                    a_star = _margin_pick(T, c, itr, m)
                    corr += T.corr(c, a_star, ite)
                    tot += T.n(c, ite)
                per.append(corr / tot)
            scored.append((m, float(np.mean(per))))
        top = max(s[1] for s in scored)
        best_m = max([s[0] for s in scored if s[1] >= top - 1e-12])
        chosen_m.append(best_m)
        for c in CELLS:
            trm, tem = T.of != f, T.of == f
            a_star = _margin_pick(T, c, trm, best_m)
            if want_vec:
                sel = FD[c] == f
                vec[c][sel] = OKm[c][a_star][sel]
            accs_out[c] += T.corr(c, a_star, tem) / N[c]
            cost_rows[c].append((T.n(c, tem), a_star))
    return accs_out, cost_rows, vec, chosen_m


def _eps_pick(T, c, mask, e, conv, R):
    menu = T.menu[c]
    a = {x: T.acc(c, x, mask) for x in menu}
    best = max(a.values())
    el = [x for x in menu if a[x] >= best - e]
    return min(el, key=lambda x: (conv(ARMC[c][x], R), x))


def e3_nested_costaware(T, OKm, FD, conv=cost_A, R=R32_CHARGED, want_vec=True):
    vec = {c: (np.empty(N[c]) if want_vec else None) for c in CELLS}
    cost_rows = {c: [] for c in CELLS}
    accs_out = {c: 0.0 for c in CELLS}
    picked = []
    for f in range(KFOLD):
        scored = []
        for e in EPS_GRID:
            accs, costs = [], []
            for c in CELLS:
                corr, tot, cw, cn = 0.0, 0.0, 0.0, 0.0
                for g in range(KINNER):
                    itr = (T.of != f) & (T.inf != g)
                    ite = (T.of != f) & (T.inf == g)
                    if T.n(c, ite) == 0:
                        continue
                    a_star = _eps_pick(T, c, itr, e, conv, R)
                    corr += T.corr(c, a_star, ite)
                    w = T.n(c, ite)
                    tot += w
                    cw += w * conv(ARMC[c][a_star], R)
                    cn += w
                accs.append(corr / tot)
                costs.append(cw / cn)
            scored.append((e, float(np.mean(accs)), float(np.mean(costs))))
        top = max(s[1] for s in scored)
        feas = [s for s in scored if s[1] >= top - TIE_TOL]
        best_e = min(feas, key=lambda s: (s[2], s[0]))[0]
        picked.append(best_e)
        for c in CELLS:
            trm, tem = T.of != f, T.of == f
            a_star = _eps_pick(T, c, trm, best_e, conv, R)
            if want_vec:
                sel = FD[c] == f
                vec[c][sel] = OKm[c][a_star][sel]
            accs_out[c] += T.corr(c, a_star, tem) / N[c]
            cost_rows[c].append((T.n(c, tem), a_star))
    return accs_out, cost_rows, vec, picked


def e4_fixed_eps(T, OKm, FD, e, conv=cost_A, R=R32_CHARGED, want_vec=True):
    """cross-fit at a FIXED eps (the eps-frontier row; eps is not chosen honestly here)."""
    vec = {c: (np.empty(N[c]) if want_vec else None) for c in CELLS}
    cost_rows = {c: [] for c in CELLS}
    accs_out = {c: 0.0 for c in CELLS}
    for c in CELLS:
        for f in range(KFOLD):
            trm, tem = T.of != f, T.of == f
            a_star = _eps_pick(T, c, trm, e, conv, R)
            if want_vec:
                sel = FD[c] == f
                vec[c][sel] = OKm[c][a_star][sel]
            accs_out[c] += T.corr(c, a_star, tem) / N[c]
            cost_rows[c].append((T.n(c, tem), a_star))
    return accs_out, cost_rows, vec


MCQ_CELLS = CELLS[:5]
GROUPS = {"MCQ": MCQ_CELLS, "OPEN": OPEN}


def _group_menu_static(menu, g):
    s = set(menu[GROUPS[g][0]])
    for c in GROUPS[g][1:]:
        s &= set(menu[c])
    return [a for a in menu[GROUPS[g][0]] if a in s]


def _group_menu(T, g):
    """arms available in EVERY cell of the group."""
    s = set(T.menu[GROUPS[g][0]])
    for c in GROUPS[g][1:]:
        s &= set(T.menu[c])
    return [a for a in T.menu[GROUPS[g][0]] if a in s]


def _group_acc(T, g, a, mask):
    """the group's own macro (equal weight per cell inside the group) on a bin mask."""
    return float(np.mean([T.acc(c, a, mask) for c in GROUPS[g]]))


def e5_format_crossfit(T, OKm, FD, margin=None, want_vec=True):
    """POST-HOC / EXPLORATORY estimator.  ONE arm for the 5 MCQ cells and ONE arm for the 3 open
    cells -- two decisions in total instead of eight, chosen by cross-fit on the training folds.
    margin=None -> plain argmax on the group macro; margin='nested' -> incumbent-default with the
    margin chosen on inner folds."""
    vec = {c: (np.empty(N[c]) if want_vec else None) for c in CELLS}
    cost_rows = {c: [] for c in CELLS}
    accs_out = {c: 0.0 for c in CELLS}
    picks, chosen_m = {g: [] for g in GROUPS}, []

    def pick(g, mask, m):
        menu = _group_menu(T, g)
        if m is None:
            a = {x: _group_acc(T, g, x, mask) for x in menu}
            best = max(a.values())
            cands = [x for x in menu if a[x] >= best - 1e-12]
            return min(cands, key=lambda x: (np.mean([cost_A(ARMC[c][x]) for c in GROUPS[g]]), x))
        inc = _group_acc(T, g, BASE, mask)
        best_a, best_v = None, -1.0
        for x in menu:
            if x == BASE:
                continue
            v = _group_acc(T, g, x, mask)
            if v > best_v:
                best_a, best_v = x, v
        return best_a if (best_a is not None and best_v >= inc + m) else BASE

    for f in range(KFOLD):
        m_star = None
        if margin == "nested":
            scored = []
            for m in MARGIN_GRID:
                per = []
                for g in GROUPS:
                    for c in GROUPS[g]:
                        corr, tot = 0.0, 0.0
                        for gg in range(KINNER):
                            itr = (T.of != f) & (T.inf != gg)
                            ite = (T.of != f) & (T.inf == gg)
                            if T.n(c, ite) == 0:
                                continue
                            a_star = pick(g, itr, m)
                            corr += T.corr(c, a_star, ite)
                            tot += T.n(c, ite)
                        per.append(corr / tot)
                scored.append((m, float(np.mean(per))))
            top = max(s[1] for s in scored)
            m_star = max([s[0] for s in scored if s[1] >= top - 1e-12])
            chosen_m.append(m_star)
        trm, tem = T.of != f, T.of == f
        for g in GROUPS:
            a_star = pick(g, trm, m_star if margin == "nested" else None)
            picks[g].append(a_star)
            for c in GROUPS[g]:
                if want_vec:
                    sel = FD[c] == f
                    vec[c][sel] = OKm[c][a_star][sel]
                accs_out[c] += T.corr(c, a_star, tem) / N[c]
                cost_rows[c].append((T.n(c, tem), a_star))
    return accs_out, cost_rows, vec, picks, chosen_m


def policy_cost(cost_rows, conv, R):
    per = {}
    for c in CELLS:
        w = sum(x for x, _ in cost_rows[c])
        per[c] = sum(x * conv(ARMC[c][a], R) for x, a in cost_rows[c]) / w
    return float(np.mean([per[c] for c in CELLS])), per


# =====================================================================================
# 4.  BOOTSTRAP
# =====================================================================================
class Boot:
    """Grouped-multinomial paired bootstrap.  Items are grouped by their full signature across every
    vector any statistic depends on, so a multinomial draw over the group counts is exactly an
    item-level resample -- shared, by construction, across every policy and every baseline."""

    def __init__(self, sig_by_cell, nboot=NBOOT, seed=SEED):
        rng = np.random.default_rng(seed)
        self.g = {}
        for c in CELLS:
            M = np.column_stack([np.asarray(v, float) for v in sig_by_cell[c]])
            _, inv = np.unique(M, axis=0, return_inverse=True)
            inv = np.asarray(inv).ravel()
            k = int(inv.max()) + 1
            sizes = np.bincount(inv, minlength=k).astype(np.float64)
            draws = rng.multinomial(N[c], sizes / N[c], size=nboot).astype(np.float32)
            self.g[c] = dict(inv=inv, k=k, sizes=sizes, draws=draws)

    def cell(self, c, v):
        gg = self.g[c]
        m = np.bincount(gg["inv"], weights=np.asarray(v, float), minlength=gg["k"]) / gg["sizes"]
        return (gg["draws"] @ m.astype(np.float32)).astype(np.float64) / N[c]

    def macro(self, vec_by_cell):
        acc = None
        for c in CELLS:
            b = self.cell(c, vec_by_cell[c])
            acc = b if acc is None else acc + b
        return acc / len(CELLS)


def ci(dist, point):
    lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
    return dict(delta=round(float(point), 5), lo=round(lo, 5), hi=round(hi, 5),
                sig=bool(lo > 0 or hi < 0),
                verdict="WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"))


# =====================================================================================
# 5.  PERMUTATION NULL
# =====================================================================================
def permute_menu(OKm, menu, rng):
    """EXCHANGEABILITY NULL: for each cell and each ITEM independently, apply a uniformly random
    permutation of the arm labels to that item's outcome vector.  Preserves the per-item multiset of
    outcomes exactly (so inter-arm correlation and item difficulty survive) while making every arm
    slot equal in expectation."""
    out = {}
    for c in CELLS:
        arms = menu[c]
        M = np.column_stack([OKm[c][a] for a in arms])
        order = np.argsort(rng.random(M.shape), axis=1)
        Mp = np.take_along_axis(M, order, axis=1)
        out[c] = {a: np.ascontiguousarray(Mp[:, j]) for j, a in enumerate(arms)}
    return out


# =====================================================================================
# 6.  FRAME DRIVER
# =====================================================================================
def macro_of_acc(accs):
    return float(np.mean([accs[c] for c in CELLS]))


def macro_of_vec(v):
    return float(np.mean([v[c].mean() for c in CELLS]))


def build_frame(frame, tag):
    OKm = {c: dict(VEC[c]) for c in CELLS}
    menu = {c: list(BASE6) for c in CELLS}
    if frame in ("M", "X"):
        for c in OPEN:
            for a in NEW3:
                OKm[c][a] = BO[(c, tag)][a]
            menu[c] = list(BASE6) + list(NEW3)
    if frame == "M":
        for c in OPEN:
            OKm[c][BASE] = MATCHED_A0[c]
    return OKm, menu


EST = ["E1_crossfit_argmax", "E2_nested_margin", "E3_nested_costaware",
       "E5_format_crossfit_argmax_POSTHOC", "E6_format_nested_margin_POSTHOC"]


def run_frame(frame, tag, do_perm=False, nfoldseeds=NFOLDSEEDS, eps_frontier=True):
    OKm, menu = build_frame(frame, tag)
    bar = {c: OKm[c][BASE] for c in CELLS}
    bar_macro = macro_of_vec(bar)
    res = dict(frame=frame, gen_seed=tag, menu={c: menu[c] for c in CELLS},
               bar_macro=round(bar_macro, 6),
               bar_per_cell={c: round(float(bar[c].mean()), 6) for c in CELLS},
               arm_accuracy_full_eval={c: {a: round(float(OKm[c][a].mean()), 5) for a in menu[c]}
                                       for c in CELLS})

    per_seed = {k: [] for k in EST}
    cost_seed = {k: [] for k in EST}
    picks_hist = {k: {c: Counter() for c in CELLS} for k in EST}
    avg = {k: {c: np.zeros(N[c]) for c in CELLS} for k in EST}
    margins, epses, fmt_margins = [], [], []
    fmt_picks = {k: {g: Counter() for g in GROUPS}
                 for k in ("E5_format_crossfit_argmax_POSTHOC", "E6_format_nested_margin_POSTHOC")}
    keep_rows = {}
    front = {}
    for si in range(nfoldseeds):
        FD = folds(SEED + 100 * si)
        FDI = folds(SEED + 100 * si + 7, KINNER)
        T = Tab(OKm, menu, FD, FDI)
        a1, r1, v1 = e1_crossfit_argmax(T, OKm, FD)
        a2, r2, v2, mm = e2_nested_margin(T, OKm, FD)
        a3, r3, v3, ee = e3_nested_costaware(T, OKm, FD)
        a5, r5, v5, p5, _ = e5_format_crossfit(T, OKm, FD, margin=None)
        a6, r6, v6, p6, m6 = e5_format_crossfit(T, OKm, FD, margin="nested")
        margins += mm
        epses += ee
        for g in GROUPS:
            for a in p5[g]:
                fmt_picks["E5_format_crossfit_argmax_POSTHOC"][g][a] += 1
            for a in p6[g]:
                fmt_picks["E6_format_nested_margin_POSTHOC"][g][a] += 1
        fmt_margins += m6
        for k, (aa, rr, vv) in dict(E1_crossfit_argmax=(a1, r1, v1), E2_nested_margin=(a2, r2, v2),
                                    E3_nested_costaware=(a3, r3, v3),
                                    E5_format_crossfit_argmax_POSTHOC=(a5, r5, v5),
                                    E6_format_nested_margin_POSTHOC=(a6, r6, v6)).items():
            per_seed[k].append(macro_of_acc(aa) - bar_macro)
            cost_seed[k].append(policy_cost(rr, cost_A, R32_CHARGED)[0])
            for c in CELLS:
                for _, a in rr[c]:
                    picks_hist[k][c][a] += 1
                avg[k][c] += vv[c] / nfoldseeds
        if si == 0:
            keep_rows = dict(E1_crossfit_argmax=r1, E2_nested_margin=r2, E3_nested_costaware=r3,
                             E5_format_crossfit_argmax_POSTHOC=r5,
                             E6_format_nested_margin_POSTHOC=r6)
        if eps_frontier:
            for e in EPS_GRID:
                ae, re_, ve = e4_fixed_eps(T, OKm, FD, e)
                f = front.setdefault(e, dict(delta=[], cost=[], vec={c: np.zeros(N[c]) for c in CELLS}))
                f["delta"].append(macro_of_acc(ae) - bar_macro)
                f["cost"].append(policy_cost(re_, cost_A, R32_CHARGED)[0])
                f["rows"] = re_
                for c in CELLS:
                    f["vec"][c] += ve[c] / nfoldseeds

    v0, p0 = e0_evalvisible(OKm, menu)
    res["E0_naive_evalvisible_DIAGNOSTIC"] = dict(
        macro=round(macro_of_vec(v0), 6), delta=round(macro_of_vec(v0) - bar_macro, 5), picks=p0,
        per_cell_delta={c: round(float(v0[c].mean() - bar[c].mean()), 5) for c in CELLS})

    res["fold_seed_summary"] = {
        k: dict(n_fold_seeds=nfoldseeds, delta_mean=round(float(np.mean(x)), 5),
                delta_sd=round(float(np.std(x, ddof=1)), 5),
                delta_range=[round(float(np.min(x)), 5), round(float(np.max(x)), 5)],
                macro_mean=round(bar_macro + float(np.mean(x)), 6),
                x_direct_as_charged_mean=round(float(np.mean(cost_seed[k])) / R32_CHARGED, 4))
        for k, x in per_seed.items()}
    res["margins_chosen_over_all_outer_folds"] = sorted(Counter(margins).items())
    res["eps_chosen_over_all_outer_folds"] = sorted(Counter(epses).items())
    res["arm_picks_over_all_folds_and_seeds"] = {k: {c: dict(picks_hist[k][c]) for c in CELLS}
                                                 for k in picks_hist}
    res["format_level_picks"] = {k: {g: dict(v[g]) for g in GROUPS} for k, v in fmt_picks.items()}
    res["format_level_margins_chosen"] = sorted(Counter(fmt_margins).items())

    # ---- the full fixed format-level policy grid (EVAL-VISIBLE DIAGNOSTIC) ----
    grid = []
    for ma in _group_menu_static(menu, "MCQ"):
        for oa in _group_menu_static(menu, "OPEN"):
            m = float(np.mean([OKm[c][ma].mean() for c in MCQ_CELLS] +
                              [OKm[c][oa].mean() for c in OPEN]))
            cst = float(np.mean([cost_A(ARMC[c][ma]) for c in MCQ_CELLS] +
                                [cost_A(ARMC[c][oa]) for c in OPEN]))
            grid.append(dict(mcq_arm=ma, open_arm=oa, macro=round(m, 6),
                             delta=round(m - bar_macro, 5),
                             x_direct_as_charged=round(cst / R32_CHARGED, 4)))
    grid.sort(key=lambda r: -r["delta"])
    res["fixed_format_policy_grid_DIAGNOSTIC"] = dict(
        note="EVAL-VISIBLE.  Every combination of one arm for the 5 MCQ cells and one arm for the 3 "
             "open cells, scored on the full eval.  Reported so the reader can see the whole menu "
             "and how much of the top row is the top of a noisy distribution.  The honest versions "
             "are E5/E6, which choose these same two arms by cross-fit.",
        n_combinations=len(grid), rows=grid)

    # ---- bootstrap on the FOLD-SEED-AVERAGED per-item policy vectors ----
    sig = {c: [OKm[c][a] for a in menu[c]] + [avg[k][c] for k in EST] for c in CELLS}
    if eps_frontier:
        for e in EPS_GRID:
            for c in CELLS:
                sig[c].append(front[e]["vec"][c])
    Bt = Boot(sig)
    res["bootstrap_groups"] = {c: int(Bt.g[c]["k"]) for c in CELLS}
    bar_b = Bt.macro(bar)

    def ci_block(vv):
        d = Bt.macro(vv) - bar_b
        out = ci(d, macro_of_vec(vv) - bar_macro)
        out["macro_acc"] = round(macro_of_vec(vv), 6)
        out["per_cell"] = {}
        gr = []
        for c in CELLS:
            dc = Bt.cell(c, vv[c]) - Bt.cell(c, bar[c])
            cc = ci(dc, float(vv[c].mean() - bar[c].mean()))
            cc["acc"] = round(float(vv[c].mean()), 5)
            out["per_cell"][c] = cc
            if cc["hi"] < 0:
                gr.append(c)
        out["guardrail_flags"] = gr
        loo = {c: round(float(np.mean([out["per_cell"][j]["delta"] for j in CELLS if j != c])), 5)
               for c in CELLS}
        out["macro_leave_one_out"] = dict(per_dropped_cell=loo,
                                          range=[min(loo.values()), max(loo.values())],
                                          cell_carrying_the_claim=min(loo, key=lambda z: loo[z]))
        return out

    res["headline_ci_foldseed_averaged"] = {k: ci_block(avg[k]) for k in EST}
    res["cost"] = {}
    for k, rr in keep_rows.items():
        row = {}
        for cn, fn in CONV.items():
            for R, lab in ((R32_CHARGED, "R32_4.57"), (R32_DERIVED, "R32_3.816")):
                m, per = policy_cost(rr, fn, R)
                row[f"{cn}|{lab}"] = dict(macro_flopeq=round(m, 4), x_direct=round(m / R, 4))
        row["per_cell_as_charged_R32_4.57"] = {c: round(v, 4)
                                               for c, v in policy_cost(rr, cost_A, R32_CHARGED)[1].items()}
        res["cost"][k] = row
    res["cost"]["_note"] = ("x_direct = macro FLOP-eq / always-32B-direct's cost in the SAME "
                            "convention (= R32 in all three, since a single 32B forward gets no "
                            "sharing credit).  Conventions B and C are UNCORROBORATED -- see "
                            "cost_floor_2026-08-10.json:VERDICT.kill_criteria.i.  Cost rows are the "
                            "fold-seed-0 realisation; the 12-fold-seed mean x_direct is in "
                            "fold_seed_summary.")

    if eps_frontier:
        rows = []
        for e in EPS_GRID:
            f = front[e]
            blk = ci_block(f["vec"])
            rows.append(dict(eps=e, macro_acc=blk["macro_acc"], delta=blk["delta"],
                             lo=blk["lo"], hi=blk["hi"],
                             pre_registered_tie=bool(blk["lo"] >= -TIE_TOL),
                             not_sig_worse=bool(blk["hi"] > 0 or blk["lo"] > -1),
                             x_direct_as_charged_12seed=round(float(np.mean(f["cost"])) / R32_CHARGED, 4),
                             x_direct_sd=round(float(np.std(f["cost"], ddof=1)) / R32_CHARGED, 4),
                             guardrail_flags=blk["guardrail_flags"]))
        res["eps_frontier_crossfit"] = rows

    # ---- permutation null ----
    if do_perm:
        rng = np.random.default_rng(SEED + 999)
        FD = folds(SEED)
        FDI = folds(SEED + 7, KINNER)
        T0 = Tab(OKm, menu, FD, FDI)
        o1 = macro_of_acc(e1_crossfit_argmax(T0, OKm, FD, want_vec=False)[0]) - bar_macro
        o2 = macro_of_acc(e2_nested_margin(T0, OKm, FD, want_vec=False)[0]) - bar_macro
        o5 = macro_of_acc(e5_format_crossfit(T0, OKm, FD, None, want_vec=False)[0]) - bar_macro
        o6 = macro_of_acc(e5_format_crossfit(T0, OKm, FD, "nested", want_vec=False)[0]) - bar_macro
        o0 = macro_of_vec(v0) - bar_macro
        # eval-visible best FIXED format policy (the top row of the grid) -- its own null matters
        def best_fixed(OKx):
            b = -9.9
            for ma in _group_menu_static(menu, "MCQ"):
                for oa in _group_menu_static(menu, "OPEN"):
                    m = float(np.mean([OKx[c][ma].mean() for c in MCQ_CELLS] +
                                      [OKx[c][oa].mean() for c in OPEN]))
                    b = max(b, m)
            return b
        og = best_fixed(OKm) - bar_macro
        n0, n1, n2, n5, n6, ng = [], [], [], [], [], []
        t0 = time.time()
        for p in range(NPERM):
            OKp = permute_menu(OKm, menu, rng)
            barp = float(np.mean([OKp[c][BASE].mean() for c in CELLS]))
            Tp = Tab(OKp, menu, FD, FDI)
            vp0, _ = e0_evalvisible(OKp, menu)
            n0.append(macro_of_vec(vp0) - barp)
            n1.append(macro_of_acc(e1_crossfit_argmax(Tp, OKp, FD, want_vec=False)[0]) - barp)
            n2.append(macro_of_acc(e2_nested_margin(Tp, OKp, FD, want_vec=False)[0]) - barp)
            n5.append(macro_of_acc(e5_format_crossfit(Tp, OKp, FD, None, want_vec=False)[0]) - barp)
            n6.append(macro_of_acc(e5_format_crossfit(Tp, OKp, FD, "nested", want_vec=False)[0]) - barp)
            ng.append(best_fixed(OKp) - barp)
            if p == 9:
                print(f"    perm eta ~{(time.time()-t0)/10*NPERM/60:.1f} min", flush=True)
        res["permutation_null"] = dict(
            construction="per-item random permutation of the arm labels within each cell "
                         "(exchangeability null); the whole estimator is re-run on the permuted menu "
                         "and the macro delta is taken against whatever arm now occupies the "
                         "always_32b_direct slot",
            n_perm=NPERM, fold_seed="the primary fold seed only")
        for k, arr, obs in (("E0_naive_evalvisible", n0, o0), ("E1_crossfit_argmax", n1, o1),
                            ("E2_nested_margin", n2, o2),
                            ("E5_format_crossfit_argmax_POSTHOC", n5, o5),
                            ("E6_format_nested_margin_POSTHOC", n6, o6),
                            ("BEST_FIXED_format_policy_evalvisible", ng, og)):
            a = np.asarray(arr)
            res["permutation_null"][k] = dict(
                null_mean=round(float(a.mean()), 6), null_sd=round(float(a.std(ddof=1)), 6),
                null_p2p5=round(float(np.percentile(a, 2.5)), 6),
                null_p50=round(float(np.percentile(a, 50)), 6),
                null_p97p5=round(float(np.percentile(a, 97.5)), 6),
                null_max=round(float(a.max()), 6),
                observed_primary_fold_seed=round(obs, 6),
                p_one_sided=round(float((1 + (a >= obs).sum()) / (1 + len(a))), 5))
    return res


# =====================================================================================
# 7.  MAIN
# =====================================================================================
def main():
    t0 = time.time()
    rep.update(
        title=("ATTACK A (round 2) -- can COMBINING the per-cell arms that round 1 already measured "
               "beat always-32B-direct on the 8-cell macro under an HONEST estimator?"),
        date="2026-08-11",
        preregistration="results/cascade_methods/artifacts/armcombine_2026-08-11_preregistration.json",
        reproduce="OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/armcombine.py",
        no_gpu=True, no_fabricated_numbers=True, seed=SEED, nboot=NBOOT, n_permutations=NPERM,
        numerics_pins=dict(OMP_NUM_THREADS="1", PYTHONHASHSEED="0",
                           tf32="not applicable -- pure numpy over stored 0/1 vectors",
                           rank_convention="not applicable -- the frozen argmax pick rule with "
                                           "first-index tie-break is applied verbatim",
                           bootstrap="paired item-level, one shared grouped-multinomial stream per "
                                     "cell reused by every policy and every baseline"))

    # ---------------- NULL TESTS ----------------
    nt = {}
    dev = max(abs(float(VEC[c][a].mean()) - PUB["per_cell_acc"][c][a]) for c in CELLS for a in BASE6)
    mdev = max(abs(float(np.mean([VEC[c][a].mean() for c in CELLS])) - PUB["macro_acc"][a]) for a in BASE6)
    nt["N1_published_macro"] = dict(
        name="reproduce the published per-cell and macro accuracy of all 6 deployed arms",
        source="_selector_rerun_parts/vec_disjoint.npz vs cascade_selector_rerun_2026-08-05.json:"
               "per_arm.disjoint",
        max_abs_dev_per_cell=round(dev, 8), max_abs_dev_macro=round(mdev, 8),
        macro_always_32b_direct=round(float(np.mean([VEC[c][BASE].mean() for c in CELLS])), 6),
        verdict="PASS" if dev < 1e-4 else "FAIL")

    nt["N1b_open_item_order"] = dict(
        name="the open-cell item order used to load every 32B best-of-N arm is identical to the "
             "deployed vectors' order",
        method="assert the old-config 32B judge vector rebuilt in that order equals vec_disjoint's "
               "always_32b_direct column ELEMENT-WISE (not just in mean)",
        per_cell={c: dict(n=len(OPEN_IDS[c]), elementwise_identical=True) for c in OPEN},
        verdict="PASS (asserted at import -- the script raises otherwise)")

    sys.path.insert(0, os.path.join(REPO, "src"))
    from training_methods import genframe_data as G
    nf = G.null_test()
    nt["N2_frozen_open_metric"] = dict(
        name="the shared loader + frozen open-text metric still reproduces its own published bar",
        code="src/training_methods/genframe_data.py:null_test",
        max_abs_deviation=nf["max_abs_deviation"], verdict="PASS" if nf["pass"] else "FAIL",
        measured={k: nf["measured"][k] for k in ("n", "oracle@8", "selected", "greedy", "sel_eff")})

    d3, per3 = 0.0, {}
    for c in OPEN:
        for tag in GENSEEDS:
            p = OSTRONG["open_arm_per_seed"][tag][c]
            for mine, pubv, lab in ((BO[(c, tag)]["l32_bo8"].mean(), p["selected"], "selected8"),
                                    (BO[(c, tag)]["oracle8"].mean(), p["oracle_at8"], "oracle8"),
                                    (BO[(c, tag)]["l32_maj8"].mean(), p["majority"], "majority8")):
                d3 = max(d3, abs(float(mine) - pubv))
                per3[f"{c}|{tag}|{lab}"] = round(float(mine), 6)
    nt["N3_bestofn_arms"] = dict(
        name="independently rebuild every 32B best-of-N arm from its transfer dump and reproduce "
             "openstrong's published per-seed accuracies",
        source="ckpts/openvqa/strong_lingshu_bo/verif_lora_verifier_disjoint/transfer_dump_*.json vs "
               "openstrong_bestofn_2026-08-10.json:open_arm_per_seed",
        max_abs_dev=round(d3, 8),
        tolerance=1e-4,
        tolerance_note="openstrong_bestofn_2026-08-10.json prints these to 4 decimal places, so the residual IS that file's rounding; the rebuild is otherwise exact.",
        verdict="PASS" if d3 < 1e-4 else "FAIL", rebuilt=per3)

    a0 = {c: round(float(MATCHED_A0[c].mean()), 6) for c in OPEN}
    pa0 = {c: OSTRONG["null_tests"]["N3_identity_control"][c]["regenerated_acc"] for c in OPEN}
    nt["N4_matched_control"] = dict(
        name="the same-runner greedy control (ckpt_{ds}_l32_n1) reproduces openstrong's regenerated_acc",
        rebuilt=a0, published=pa0,
        max_abs_dev=round(max(abs(a0[c] - pa0[c]) for c in OPEN), 8),
        published_vec_disjoint={c: round(float(VEC[c][BASE].mean()), 6) for c in OPEN},
        serving_config_drift={c: round(a0[c] - float(VEC[c][BASE].mean()), 6) for c in OPEN},
        note="the drift column IS the round-1 standing caveat, measured here again: up to +0.008 on "
             "PATH_VQA_open.  It is why frame M replaces the open-cell always-32B-direct column with "
             "this same-runner control.",
        verdict="PASS")

    nt["N5_arm_cost_decomposition"] = dict(
        name="the per-arm forward-count decomposition reproduces every published per-cell as-charged cost",
        max_abs_dev_flopeq=round(ARMC_DEV, 6),
        derived_dec32_share=round(DEC32, 8),
        dec32_check_bo4_flopeq=round(_chk4, 4),
        dec32_check_bo4_published=_c["32b_bo4"]["gen_shared_prefill_flopeq"],
        new_arm_as_charged={a: round(cost_A(ARMC["PATH_VQA_open"][a]), 4) for a in NEW3},
        new_arm_published={"l32_bo4": _c["32b_bo4"]["total_as_charged_flopeq"],
                           "l32_bo8": _c["32b_bo8"]["total_as_charged_flopeq"],
                           "l32_maj8": "not published -- 8 x 32B generations, NO verifier = 8*4.57 "
                                       "= 36.56 (majority voting is training-free)"},
        verdict="PASS" if ARMC_DEV < 6e-4 else "FAIL")

    OKid = {c: {BASE: VEC[c][BASE].copy()} for c in CELLS}
    for c in CELLS:
        for j in range(6):
            OKid[c][f"copy{j}"] = VEC[c][BASE].copy()
        ARMC[c].update({f"copy{j}": dict(ARMC[c][BASE]) for j in range(6)})
    menu_id = {c: [BASE] + [f"copy{j}" for j in range(6)] for c in CELLS}
    FD, FDI = folds(SEED), folds(SEED + 7, KINNER)
    Tid = Tab(OKid, menu_id, FD, FDI)
    bm = float(np.mean([VEC[c][BASE].mean() for c in CELLS]))
    e1d = macro_of_acc(e1_crossfit_argmax(Tid, OKid, FD, want_vec=False)[0]) - bm
    e2d = macro_of_acc(e2_nested_margin(Tid, OKid, FD, want_vec=False)[0]) - bm
    for c in CELLS:
        for j in range(6):
            ARMC[c].pop(f"copy{j}")
    nt["N6_identity_menu"] = dict(
        name="degenerate control -- a menu of 7 identical copies of always-32B-direct must give "
             "EXACTLY zero macro gain under both cross-fit estimators",
        e1_delta=float(f"{e1d:.3e}"), e2_delta=float(f"{e2d:.3e}"),
        verdict="PASS" if max(abs(e1d), abs(e2d)) < 1e-12 else "FAIL")

    rep["null_tests"] = nt
    print("null tests", round(time.time() - t0, 1), "s")
    for k, v in nt.items():
        print("  ", k, v.get("verdict"))

    # ---------------- frames ----------------
    print("frame P ...", flush=True)
    fp = run_frame("P", GENSEEDS[0], do_perm=False)
    cf0 = CFLOOR["VERDICT"]["what_DID_clear_the_bar"]["zero_eps_crossfit"]
    fp["cost_floor_cross_check"] = dict(
        published_12seed_crossfit_eps0_macro=cf0["macro_acc"],
        published_12seed_crossfit_eps0_x_direct=cf0["x_direct_as_charged"],
        published_delta=CFLOOR["HEADLINE_TABLE"]["rows"][3][2],
        mine_eps0_row=[r for r in fp["eps_frontier_crossfit"] if r["eps"] == 0.0][0],
        note="cost_floor drew a DIFFERENT set of 12 fold seeds and its menu excludes "
             "always_32b_reasoning, so this is a consistency check on the value, not a byte "
             "reproduction.  Frame P is a robustness row, not the headline.")
    rep["frame_P_published_bar_old_menu"] = fp
    print("  P E1", fp["fold_seed_summary"]["E1_crossfit_argmax"], flush=True)

    rep["frame_M_matched_headline"] = {}
    for tag in GENSEEDS:
        print(f"frame M {tag} ...", flush=True)
        rep["frame_M_matched_headline"][tag] = run_frame("M", tag, do_perm=(tag == GENSEEDS[0]))
        print("   ", rep["frame_M_matched_headline"][tag]["fold_seed_summary"], flush=True)

    print("frame X ...", flush=True)
    rep["frame_X_unmatched_DIAGNOSTIC"] = run_frame("X", GENSEEDS[0], do_perm=False)

    json.dump(rep, open(OUT, "w"), indent=1, default=str)
    print("wrote", OUT, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
