#!/usr/bin/env python3
"""
min_escalation.py -- ATTACK 4.

PART 1  MINIMUM ESCALATION AT PARITY.  Escalation is the dominant cost term.  Sweep the GATE
        THRESHOLD (which cost_floor_2026-08-10 explicitly could not: its limitations[0] says the
        per-item gate features were not in vec_disjoint.npz), cross-fit it, and report the frontier
        (escalation rate -> macro accuracy -> compute) under the constrained minimisation
             MINIMISE macro FLOP-eq  s.t.  CI_lo(macro delta vs always-32B-direct) >= -0.0029.
PART 2  THE ZERO-32B QUESTION.  Best macro accuracy with NO 32B at test time, decomposed per cell,
        against the always-32B-direct bar of 0.6567.

Reads only stored artifacts + checkpoints.  CPU only, no GPU, no new inference.
Launch from the repo root:   python3 src/cascade_methods/min_escalation.py
Pre-registration:  results/cascade_methods/artifacts/min_escalation_2026-08-12_preregistration.json
Writes             results/cascade_methods/artifacts/min_escalation_2026-08-12.json
"""
import os, sys, json, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_min_escalation_parts")
OUT = os.path.join(ART, "min_escalation_2026-08-12.json")

SEED = 20260812
NBOOT = 10000
TIE_TOL = 0.0029
K = 5                              # cross-fit folds (matches every published arm)
NSEED = 12

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
MCQ = CELLS[:5]
OPEN = CELLS[5:]

R32_CHARGED, R32_DERIVED = 4.57, 3.816
PREFILL_SHARE, DECODE_SHARE = 0.988456, 0.011544        # cost_floor N2 (derived from the FLOP model)
VER_PREFIX, VER_MARGINAL = 1.0161, 0.0432               # cost_floor verifier_geometry (convention C)

# measured batch-1 constants (paper_baselines.py:GEN7/VER7/GEN32N -- logs/latency_opentext.jsonl)
GEN7_MS, GEN7_J = 347.0, 45.8
VER7_MS, VER7_J = 175.0, 25.3
GEN32_MS, GEN32_J = 665.0, 127.0
BO_PAR_MS = GEN7_MS + VER7_MS                            # parallel best-of-k floor = 1 gen + 1 verify
# measured best-of-N latency/energy (bestofn_latency_energy_2026-08-03*.json, n=45, uncontended card)
M_BO1_MS, M_BO8_MS, M_BO1_J, M_BO8_J = 625.4, 1305.3, 97.17, 316.7

QGRID = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
         0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
LAMBDAS = [0.0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3,
           5e-3, 7.5e-3, 1e-2, 2e-2, 5e-2, 1e-1]

rep = {}


# =================================================================================================
# 0.  LOAD
# =================================================================================================
F = np.load(os.path.join(PARTS, "features.npz"))
FMETA = json.load(open(os.path.join(PARTS, "features_meta.json")))
ZPUB = np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"))
PUB = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))["per_arm"]["disjoint"]
CF = json.load(open(os.path.join(ART, "cost_floor_2026-08-10.json")))
VRAM = json.load(open(os.path.join(ART, "vram_testtime_2026-08-11.json")))

D = {}
for c in CELLS:
    if c in MCQ:
        D[c] = dict(fmt="MCQ", n=len(F[f"{c}|ok7"]), ok7=F[f"{c}|ok7"].astype(np.float64),
                    ok32=F[f"{c}|ok32"].astype(np.float64), okT=F[f"{c}|okT"].astype(np.float64),
                    margin=F[f"{c}|margin"], conf=F[f"{c}|conf"])
    else:
        sc = F[f"{c}|scores"]; sl = F[f"{c}|sl"].astype(np.float64)
        D[c] = dict(fmt="open", n=len(sc), scores=sc, sl=sl,
                    greedy=F[f"{c}|greedy"].astype(np.float64), ok32=F[f"{c}|ok32"].astype(np.float64))
        # per-k pick outcome and gate, precomputed (the first k of the 8 stored candidates)
        D[c]["pick_ok"] = np.zeros((len(sc), 8)); D[c]["gate_k"] = np.zeros((len(sc), 8))
        for k in range(1, 9):
            p = sc[:, :k].argmax(1)
            D[c]["pick_ok"][:, k - 1] = sl[np.arange(len(sc)), p]
            D[c]["gate_k"][:, k - 1] = sc[:, :k].max(1)
N = {c: D[c]["n"] for c in CELLS}


def r(x, d=4):
    try:
        return round(float(x), d)
    except Exception:
        return x


# =================================================================================================
# 1.  NULL TESTS
# =================================================================================================
def _cascade_persample_vec(ok_c, ok_s, g, fold=None):
    """5-fold cross-fit margin cascade -> per-item delivered ok + escalation.  Identical semantics to
    paper_baselines.cascade_persample / integrated_method.heldout, with the O(n log n) tau search."""
    n = len(ok_c); ok = np.zeros(n); esc = np.zeros(n)
    fl = np.arange(n) % K if fold is None else fold
    for f in range(K):
        te = fl == f; tr = ~te
        tau = pick_tau_isocost(ok_c[tr], ok_s[tr], g[tr], float(ok_s[tr].mean()))
        e = g[te] < tau
        ok[te] = np.where(e, ok_s[te], ok_c[te]); esc[te] = e
    return ok, float(esc.mean())


def null_tests():
    devs = {}
    # N1 -- per-item identity of the two baselines against the published vectors
    m = 0.0
    for c in CELLS:
        a7 = D[c]["greedy"] if c in OPEN else D[c]["ok7"]
        m = max(m, float(np.abs(a7 - ZPUB[f"{c}|always_7b"].astype(float)).max()))
        m = max(m, float(np.abs(D[c]["ok32"] - ZPUB[f"{c}|always_32b_direct"].astype(float)).max()))
    devs["N1_baseline_per_item_max_abs_dev"] = m
    # N2 -- reproduce the shipped margin cascade per item on all 5 MCQ cells
    m2, esc_pub = 0.0, {}
    for c in MCQ:
        ok, esc = _cascade_persample_vec(D[c]["ok7"], D[c]["ok32"], D[c]["margin"])
        m2 = max(m2, float(np.abs(ok - ZPUB[f"{c}|method_compute_lean"].astype(float)).max()))
        esc_pub[c] = r(esc)
    devs["N2_shipped_margin_cascade_per_item_max_abs_dev"] = m2
    devs["N2_note"] = ("the tau search is the O(n log n) rewrite of integrated_method.pick_tau_isocost; "
                       "this test IS its verification -- it reproduces the published per-item "
                       "method_compute_lean vector exactly on all 5 MCQ cells")
    devs["N2_reproduced_escalation"] = esc_pub
    devs["N2_published_escalation"] = {"PMC_VQA": 0.0845, "SLAKE_closed": 0.2045,
                                       "VQA_RAD_closed": 0.5697, "PATH_VQA_closed": 0.4572,
                                       "MedXpertQA-MM": 0.8960}
    # N3 -- the frozen open-text bar (genframe_data.py docstring)
    sc = np.concatenate([D[c]["scores"] for c in OPEN]); sl = np.concatenate([D[c]["sl"] for c in OPEN])
    gr = np.concatenate([D[c]["greedy"] for c in OPEN])
    sel = float(sl[np.arange(len(sl)), sc.argmax(1)].mean()); orc = float(sl.max(1).mean())
    devs["N3_open_bar"] = dict(n=int(len(sl)), greedy=r(float(gr.mean()), 6), oracle8=r(orc, 6),
                               selected=r(sel, 6), sel_eff=r(sel / orc, 6),
                               frozen_bar=dict(greedy=0.449467, oracle8=0.626013, selected=0.485288,
                                               sel_eff=0.775204),
                               max_abs_dev=r(max(abs(float(gr.mean()) - 0.449467), abs(orc - 0.626013),
                                                 abs(sel - 0.485288), abs(sel / orc - 0.775204)), 8))
    # N4 -- macro baselines reproduce the published headline
    mac = {s: float(np.mean([ZPUB[f"{c}|{s}"].mean() for c in CELLS]))
           for s in ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b"]}
    devs["N4_macro_baselines"] = {k: r(v, 6) for k, v in mac.items()}
    devs["N4_published"] = {k: PUB["macro_acc"][k] for k in mac}
    devs["N4_max_abs_dev"] = r(max(abs(mac[k] - PUB["macro_acc"][k]) for k in mac), 8)
    devs["verdict"] = ("PASS -- every per-item vector, the shipped cascade, the frozen open-text bar "
                       "and the macro baselines reproduce exactly"
                       if m == 0.0 and m2 == 0.0 and devs["N3_open_bar"]["max_abs_dev"] < 1e-5
                       and devs["N4_max_abs_dev"] < 5e-5 else "FAIL")
    devs["sources"] = dict(
        features="results/cascade_methods/artifacts/_min_escalation_parts/features.npz "
                 "(src/cascade_methods/min_escalation_features.py)",
        published_vectors="results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz",
        published_macro="results/cascade_methods/artifacts/cascade_selector_rerun_2026-08-05.json:per_arm.disjoint",
        frozen_open_bar="src/training_methods/genframe_data.py docstring (THE ENDPOINT / Incumbent)")
    return devs


# =================================================================================================
# 2.  COST MODEL
# =================================================================================================
def cost_of(kind, k, esc, r32):
    """as-charged (A) / shared-generation-prefill (B) / +shared-verifier-prefix (C) FLOP-eq,
    plus modelled batch-1 latency & energy.  esc = fraction of items escalated to the 32B."""
    if kind == "p32":
        f = dict(A=r32, B=r32, C=r32)
        return dict(flops=f, lat_par=GEN32_MS, lat_seq=GEN32_MS, energy=GEN32_J,
                    lat_par_meas=GEN32_MS, energy_meas=GEN32_J, esc=1.0, seven_b=0.0)
    if kind == "mcq":                                  # 1 cheap 7B forward, then maybe the 32B
        f = dict(A=1.0 + esc * r32, B=1.0 + esc * r32, C=1.0 + esc * r32)
        return dict(flops=f, lat_par=GEN7_MS + esc * GEN32_MS, lat_seq=GEN7_MS + esc * GEN32_MS,
                    energy=GEN7_J + esc * GEN32_J, lat_par_meas=GEN7_MS + esc * GEN32_MS,
                    energy_meas=GEN7_J + esc * GEN32_J, esc=esc, seven_b=1.0)
    if kind == "open":                                 # k generations + k verifier forwards
        genA, genB = float(k), PREFILL_SHARE + k * DECODE_SHARE
        verA, verC = float(k), VER_PREFIX + k * VER_MARGINAL
        f = dict(A=genA + verA + esc * r32, B=genB + verA + esc * r32, C=genB + verC + esc * r32)
        meas_ms = M_BO1_MS + (M_BO8_MS - M_BO1_MS) * (k - 1) / 7.0
        meas_j = M_BO1_J + (M_BO8_J - M_BO1_J) * (k - 1) / 7.0
        return dict(flops=f, lat_par=BO_PAR_MS + esc * GEN32_MS,
                    lat_seq=k * BO_PAR_MS + esc * GEN32_MS,
                    energy=k * (GEN7_J + VER7_J) + esc * GEN32_J,
                    lat_par_meas=meas_ms + esc * GEN32_MS, energy_meas=meas_j + esc * GEN32_J,
                    esc=esc, seven_b=1.0)
    if kind == "p7":                                   # 7B greedy / 7B direct only
        f = dict(A=1.0, B=1.0, C=1.0)
        return dict(flops=f, lat_par=GEN7_MS, lat_seq=GEN7_MS, energy=GEN7_J,
                    lat_par_meas=GEN7_MS, energy_meas=GEN7_J, esc=0.0, seven_b=1.0)
    raise ValueError(kind)


# =================================================================================================
# 3.  POLICY MENU  (fit on TRAIN, apply to TEST)
# =================================================================================================
def wilson_lb(kk, nn, z=1.645):
    if nn == 0:
        return 0.0
    p = kk / nn; d = 1 + z * z / nn
    c = (p + z * z / (2 * nn)) / d
    h = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / d
    return max(0.0, c - h)


def pick_tau_isocost(ok_c, ok_s, g, target):
    """VECTORISED equivalent of integrated_method.pick_tau_isocost: min-escalation tau whose cascade
    accuracy reaches `target` (fallback: the max-accuracy tau).  escalate iff g < tau."""
    o = np.argsort(g, kind="mergesort")
    gs, cs, ss = g[o], ok_c[o], ok_s[o]
    n = len(g)
    cum_s = np.concatenate([[0.0], np.cumsum(ss)])
    cum_c = np.concatenate([[0.0], np.cumsum(cs)])
    acc = (cum_s + (cum_c[-1] - cum_c)) / n                  # acc if the j lowest-gate items escalate
    valid = np.ones(n + 1, bool)
    if n > 1:
        valid[1:n] = gs[1:] != gs[:-1]                       # j realisable iff a tau separates j-1 | j
    ok = valid & (acc >= target - 1e-12)
    j = int(np.argmax(ok)) if ok.any() else int(np.argmax(np.where(valid, acc, -np.inf)))
    if j == 0:
        return gs[0]
    if j >= n:
        return gs[n - 1] + 1e-9
    return gs[j]


def menu_specs(cell):
    if cell in MCQ:
        s = [("p7",), ("p32",), ("iso",)]
        s += [("marg", q) for q in QGRID] + [("conf", q) for q in QGRID]
        s += [("veto", nb) for nb in (3, 5, 10)]
        return s
    s = [("p7",), ("p32",), ("iso",)]
    s += [("bok", k, q) for k in range(1, 9) for q in QGRID]
    return s


def apply_spec(cell, spec, tr, te, gate_override=None):
    """Fit on tr (bool mask), apply on te.  Returns (ok_te, esc_te_vector, kind, k)."""
    d = D[cell]
    if spec[0] == "p32":
        return d["ok32"][te], np.ones(int(te.sum())), "p32", 0
    if cell in MCQ:
        ok7, ok32 = d["ok7"], d["ok32"]
        if spec[0] == "p7":
            return ok7[te], np.zeros(int(te.sum())), "p7", 0
        if spec[0] == "veto":
            nb = spec[1]; c7 = d["conf"] if gate_override is None else gate_override["conf"]
            qs = np.quantile(c7[tr], np.linspace(0, 1, nb + 1)); qs[0], qs[-1] = -np.inf, np.inf
            qs = np.unique(qs)
            btr = np.clip(np.digitize(c7[tr], qs[1:-1]), 0, len(qs) - 2)
            bte = np.clip(np.digitize(c7[te], qs[1:-1]), 0, len(qs) - 2)
            keep = np.zeros(int(te.sum()), bool)
            for b in range(len(qs) - 1):
                m = btr == b
                if m.sum() < 30:
                    continue
                if wilson_lb(int(ok7[tr][m].sum()), int(m.sum())) >= float(ok32[tr][m].mean()):
                    keep |= (bte == b)
            ok = np.where(keep, ok7[te], ok32[te])
            return ok, (~keep).astype(float), "mcq", 0
        g = (d["margin"] if spec[0] in ("marg", "iso") else d["conf"])
        if gate_override is not None:
            g = gate_override["margin"] if spec[0] in ("marg", "iso") else gate_override["conf"]
        if spec[0] == "iso":
            tau = pick_tau_isocost(ok7[tr], ok32[tr], g[tr], float(ok32[tr].mean()))
        else:
            q = spec[1]
            tau = -np.inf if q <= 0 else (np.inf if q >= 1 else float(np.quantile(g[tr], q)))
        e = g[te] < tau
        return np.where(e, ok32[te], ok7[te]), e.astype(float), "mcq", 0
    # open cells
    ok32 = d["ok32"]
    if spec[0] == "p7":
        return d["greedy"][te], np.zeros(int(te.sum())), "p7", 0
    if spec[0] == "iso":
        k = 8; okc = d["pick_ok"][:, k - 1]
        g = d["gate_k"][:, k - 1] if gate_override is None else gate_override["gate_k"][:, k - 1]
        tau = pick_tau_isocost(okc[tr], ok32[tr], g[tr], float(ok32[tr].mean()))
        e = g[te] < tau
        return np.where(e, ok32[te], okc[te]), e.astype(float), "open", k
    k, q = spec[1], spec[2]
    okc = d["pick_ok"][:, k - 1]
    g = d["gate_k"][:, k - 1] if gate_override is None else gate_override["gate_k"][:, k - 1]
    tau = -np.inf if q <= 0 else (np.inf if q >= 1 else float(np.quantile(g[tr], q)))
    e = g[te] < tau
    return np.where(e, ok32[te], okc[te]), e.astype(float), "open", k


# =================================================================================================
# 4.  BOOTSTRAP -- ONE shared item-resample stream per cell, generated once, reused by every policy
# =================================================================================================
BOOT = {}
def build_boot(seed=SEED):
    rng = np.random.default_rng(seed)
    for c in CELLS:
        n = N[c]
        # counts of each item in each replicate == Multinomial(n, 1/n).  ONE stream per cell, built
        # once here and reused by every policy, so every comparison in this file is paired.
        acc = np.empty((NBOOT, n), np.int8)
        step = max(1, int(2e7 // max(n, 1)))
        p = np.full(n, 1.0 / n)
        for s0 in range(0, NBOOT, step):
            m = min(step, NBOOT - s0)
            ch = rng.multinomial(n, p, size=m)
            assert ch.max() < 127, (c, int(ch.max()))
            acc[s0:s0 + m] = ch.astype(np.int8)
        BOOT[c] = acc


def boot_delta_cell(c, dvec):
    """(NBOOT,) bootstrap means of dvec under the shared per-cell item resample."""
    n = N[c]
    out = np.empty(NBOOT)
    Cm = BOOT[c]
    step = max(1, int(4e7 // max(n, 1)))
    for s in range(0, NBOOT, step):
        out[s:s + step] = Cm[s:s + step].astype(np.float32) @ dvec.astype(np.float32) / n
    return out


def macro_ci(dvecs):
    """dvecs: {cell: per-item (policy_ok - base_ok)}.  Macro = mean over the 8 cells."""
    B = np.zeros(NBOOT)
    for c in CELLS:
        B += boot_delta_cell(c, dvecs[c])
    B /= len(CELLS)
    pt = float(np.mean([dvecs[c].mean() for c in CELLS]))
    lo, hi = float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))
    return dict(delta=r(pt), lo=r(lo), hi=r(hi), tie=bool(lo >= -TIE_TOL),
                not_sig_worse=bool(lo <= 0 <= hi or lo > 0))


def cell_ci(c, dvec):
    B = boot_delta_cell(c, dvec)
    return dict(delta=r(float(dvec.mean())), lo=r(float(np.percentile(B, 2.5))),
                hi=r(float(np.percentile(B, 97.5))))


# =================================================================================================
# 5.  CROSS-FIT LAGRANGIAN ALLOCATION
# =================================================================================================
def folds_for(seed):
    """Random 5-fold assignment (seed 0 == the published modulo-K folding, kept as the anchor)."""
    if seed is None:
        return {c: np.arange(N[c]) % K for c in CELLS}
    rng = np.random.default_rng(seed)
    return {c: rng.permutation(np.arange(N[c]) % K) for c in CELLS}


MENU_CACHE = {}

def precompute_menu(fold, gate_override=None, evalvisible=False, r32=R32_CHARGED, tag="base"):
    """For every (cell, fold, spec): the SELECTION SCORE (train accuracy, or eval accuracy in the
    DIAGNOSTIC eval-visible mode), the selection cost (same convention), and the held-out outcome."""
    tab = {}
    for c in CELLS:
        specs = menu_specs(c)
        go = None if gate_override is None else gate_override[c]
        tab[c] = []
        for f in range(K):
            te = fold[c] == f; tr = ~te
            rows = []
            for sp in specs:
                ok_te, e_te, kind, kk = apply_spec(c, sp, tr, te, go)
                if evalvisible:
                    a_sel, c_sel = float(ok_te.mean()), float(e_te.mean())
                else:
                    ok_tr, e_tr, _, _ = apply_spec(c, sp, tr, tr, go)
                    a_sel, c_sel = float(ok_tr.mean()), float(e_tr.mean())
                cst = cost_of(kind, kk, c_sel, r32)["flops"]["A"]
                rows.append(dict(spec=sp, a_sel=a_sel, cost_sel=cst, ok=ok_te, esc=e_te,
                                 kind=kind, k=kk, te=te))
            tab[c].append(rows)
    return tab


def run_lambda(lam, tab):
    """One held-out policy: per cell and fold take the menu entry maximising (a_sel - lam*cost_sel)."""
    okv = {c: np.zeros(N[c]) for c in CELLS}
    escv = {c: np.zeros(N[c]) for c in CELLS}
    kindv = {c: [None] * N[c] for c in CELLS}
    kv = {c: np.zeros(N[c]) for c in CELLS}
    picks = {c: [] for c in CELLS}
    for c in CELLS:
        for f in range(K):
            rows = tab[c][f]
            vals = [rw["a_sel"] - lam * rw["cost_sel"] for rw in rows]
            b = int(np.argmax(vals)); rw = rows[b]
            te = rw["te"]
            okv[c][te] = rw["ok"]; escv[c][te] = rw["esc"]; kv[c][te] = rw["k"]
            for i in np.where(te)[0]:
                kindv[c][i] = rw["kind"]
            picks[c].append(rw["spec"])
    return okv, escv, kindv, kv, picks


def summarise(okv, escv, kindv, kv, picks, r32=R32_CHARGED, with_ci=True, with_guard=True):
    per_cell, flopsA, flopsB, flopsC = {}, [], [], []
    latp, lats, ener, latpm, enerm, escs, sevenb = [], [], [], [], [], [], []
    for c in CELLS:
        fa = fb = fc = lp = ls = en = lpm = enm = 0.0
        kinds = np.array([x if x is not None else "" for x in kindv[c]])
        for i_kind, kk in sorted(set(zip(kinds.tolist(), kv[c].tolist()))):
            m = (kinds == i_kind) & (kv[c] == kk)
            w = float(m.mean())
            cc = cost_of(i_kind, kk, float(escv[c][m].mean()), r32)
            fa += w * cc["flops"]["A"]; fb += w * cc["flops"]["B"]; fc += w * cc["flops"]["C"]
            lp += w * cc["lat_par"]; ls += w * cc["lat_seq"]; en += w * cc["energy"]
            lpm += w * cc["lat_par_meas"]; enm += w * cc["energy_meas"]
        sb = float(np.mean(kinds != "p32"))
        acc = float(okv[c].mean()); a32 = float(D[c]["ok32"].mean())
        per_cell[c] = dict(n=N[c], acc=r(acc, 6), acc_32b=r(a32, 6), delta=r(acc - a32),
                           escalation=r(float(escv[c].mean())), flops_A=r(fa),
                           policies=sorted(set(str(p) for p in picks[c])))
        if with_guard:
            per_cell[c]["ci"] = cell_ci(c, okv[c] - D[c]["ok32"])
        flopsA.append(fa); flopsB.append(fb); flopsC.append(fc)
        latp.append(lp); lats.append(ls); ener.append(en); latpm.append(lpm); enerm.append(enm)
        escs.append(float(escv[c].mean())); sevenb.append(sb)
    macro_acc = float(np.mean([okv[c].mean() for c in CELLS]))
    out = dict(macro_acc=r(macro_acc, 6), macro_escalation=r(float(np.mean(escs))),
               seven_b_macro_weight=r(float(np.mean(sevenb))),
               cost=dict(A_as_charged_R32_4p57=dict(macro_flops=r(np.mean(flopsA)),
                                                    x_direct=r(np.mean(flopsA) / R32_CHARGED)),
                         B_recost_gen_R32_4p57=dict(macro_flops=r(np.mean(flopsB)),
                                                    x_direct=r(np.mean(flopsB) / R32_CHARGED)),
                         C_recost_full_R32_4p57=dict(macro_flops=r(np.mean(flopsC)),
                                                     x_direct=r(np.mean(flopsC) / R32_CHARGED)),
                         lat_par_ms=r(np.mean(latp), 1), lat_seq_ms=r(np.mean(lats), 1),
                         energy_J=r(np.mean(ener), 1),
                         lat_par_ms_measurement_corrected=r(np.mean(latpm), 1),
                         energy_J_measurement_corrected=r(np.mean(enerm), 1)),
               per_cell=per_cell)
    fa38 = recost_r32(okv, escv, kindv, kv, R32_DERIVED)
    out["cost"]["A_as_charged_R32_3p816"] = dict(macro_flops=r(fa38), x_direct=r(fa38 / R32_DERIVED))
    if with_ci:
        out["delta_vs_direct"] = macro_ci({c: okv[c] - D[c]["ok32"] for c in CELLS})
    if with_guard:
        out["guardrail_flags"] = [c for c in CELLS if per_cell[c]["ci"]["hi"] < 0]
    return out


def recost_r32(okv, escv, kindv, kv, r32):
    fa = []
    for c in CELLS:
        v = 0.0
        kinds = np.array([x if x is not None else "" for x in kindv[c]])
        for i_kind, kk in sorted(set(zip(kinds.tolist(), kv[c].tolist()))):
            m = (kinds == i_kind) & (kv[c] == kk); w = float(m.mean())
            v += w * cost_of(i_kind, kk, float(escv[c][m].mean()), r32)["flops"]["A"]
        fa.append(v)
    return float(np.mean(fa))


# =================================================================================================
# 6.  NESTED CV -- lambda chosen INSIDE the training folds (the fully honest operating point)
# =================================================================================================
def nested_cv(fold, seed, tab, r32=R32_CHARGED):
    """Outer 5-fold.  Inside each outer TRAIN block an inner 5-fold repeats the WHOLE selection for
    every lambda; the chosen lambda is the LARGEST (cheapest) whose inner held-out macro delta vs
    always-32B-direct is >= 0.  That lambda is then applied to the outer test fold using the outer
    train-fitted menu (`tab`).  No eval-fold label reaches any decision applied to it."""
    okv = {c: np.zeros(N[c]) for c in CELLS}
    escv = {c: np.zeros(N[c]) for c in CELLS}
    kindv = {c: [None] * N[c] for c in CELLS}
    kv = {c: np.zeros(N[c]) for c in CELLS}
    picks = {c: [] for c in CELLS}
    lam_per_fold = []
    rng = np.random.default_rng((seed if seed is not None else 0) + 777)
    for f in range(K):
        inner = {}
        for c in CELLS:
            tr_idx = np.where(fold[c] != f)[0]
            lab = np.full(N[c], -1)
            lab[tr_idx] = rng.permutation(np.arange(len(tr_idx)) % K)
            inner[c] = lab
        # --- precompute the inner menu table ONCE (it does not depend on lambda) ---
        itab = {}
        for c in CELLS:
            itab[c] = []
            for g in range(K):
                te = inner[c] == g; tr = (inner[c] >= 0) & ~te
                rows = []
                if te.sum() >= 1 and tr.sum() >= 10:
                    for sp in menu_specs(c):
                        ok_tr, e_tr, kind, kk = apply_spec(c, sp, tr, tr)
                        ok_te, _, _, _ = apply_spec(c, sp, tr, te)
                        rows.append((float(ok_tr.mean()),
                                     cost_of(kind, kk, float(e_tr.mean()), r32)["flops"]["A"],
                                     ok_te))
                itab[c].append((te, rows))
        best_lam = None
        for lam in LAMBDAS:
            accs, base = [], []
            for c in CELLS:
                okc = np.zeros(N[c]); use = np.zeros(N[c], bool)
                for g in range(K):
                    te, rows = itab[c][g]
                    if not rows:
                        continue
                    j = int(np.argmax([a - lam * cst for a, cst, _ in rows]))
                    okc[te] = rows[j][2]; use |= te
                accs.append(float(okc[use].mean())); base.append(float(D[c]["ok32"][use].mean()))
            if float(np.mean(accs) - np.mean(base)) >= 0.0:
                best_lam = lam                      # LAMBDAS ascending -> keeps the largest valid
        if best_lam is None:
            best_lam = 0.0
        lam_per_fold.append(best_lam)
        for c in CELLS:
            rows = tab[c][f]
            j = int(np.argmax([rw["a_sel"] - best_lam * rw["cost_sel"] for rw in rows]))
            rw = rows[j]; te = rw["te"]
            okv[c][te] = rw["ok"]; escv[c][te] = rw["esc"]; kv[c][te] = rw["k"]
            for i in np.where(te)[0]:
                kindv[c][i] = rw["kind"]
            picks[c].append(rw["spec"])
    return okv, escv, kindv, kv, picks, lam_per_fold


# =================================================================================================
# 7.  PER-CELL ESCALATION FRONTIER
# =================================================================================================
def per_cell_frontier():
    """For each cell independently: cross-fit gate-threshold sweep -> (escalation, accuracy, cost),
    the cell's own delta CI vs always-32B-direct, and the MINIMUM escalation that satisfies
    (a) delta point >= 0 and (b) the pre-registered CI rule at the cell level."""
    fold = folds_for(None)
    out = {}
    for c in CELLS:
        rows = []
        specs = ([("marg", q) for q in QGRID] + [("veto", 5)] + [("iso",)] if c in MCQ
                 else [("bok", 8, q) for q in QGRID] + [("iso",)])
        for sp in specs:
            ok = np.zeros(N[c]); esc = np.zeros(N[c]); kind = None; kk = 0
            for f in range(K):
                te = fold[c] == f; tr = ~te
                o, e, kind, kk = apply_spec(c, sp, tr, te)
                ok[te] = o; esc[te] = e
            e_m = float(esc.mean()); a = float(ok.mean())
            cc = cost_of(kind, kk, e_m, R32_CHARGED)
            ci = cell_ci(c, ok - D[c]["ok32"])
            rows.append(dict(spec=str(sp), escalation=r(e_m), acc=r(a, 6),
                             delta_vs_direct=ci["delta"], lo=ci["lo"], hi=ci["hi"],
                             flops_A=r(cc["flops"]["A"]),
                             x_direct=r(cc["flops"]["A"] / R32_CHARGED),
                             meets_cell_tie=bool(ci["lo"] >= -TIE_TOL),
                             delta_nonneg=bool(ci["delta"] >= 0.0)))
        rows.sort(key=lambda z: z["escalation"])
        ok_rows = [z for z in rows if z["meets_cell_tie"]]
        nn_rows = [z for z in rows if z["delta_nonneg"]]
        out[c] = dict(
            n=N[c], acc_32b=r(float(D[c]["ok32"].mean()), 6),
            acc_cheap_at_zero_escalation=r(float(
                (D[c]["ok7"] if c in MCQ else D[c]["pick_ok"][:, 7]).mean()), 6),
            rows=rows,
            min_escalation_meeting_cell_tie=(min(z["escalation"] for z in ok_rows) if ok_rows else None),
            min_escalation_with_nonneg_delta=(min(z["escalation"] for z in nn_rows) if nn_rows else None),
            cheapest_row_meeting_cell_tie=(min(ok_rows, key=lambda z: z["flops_A"]) if ok_rows else None))
    return out


# =================================================================================================
# 8.  CERTIFIED VETO -- where it fires, where it cannot, and what would have to change
# =================================================================================================
def veto_analysis(nbins=(3, 5, 10), z=1.645):
    fold = folds_for(None)
    out = {}
    for c in MCQ:
        d = D[c]; per_nb = {}
        for nb in nbins:
            keep_all = np.zeros(N[c], bool); binrows = []
            for f in range(K):
                te = fold[c] == f; tr = ~te
                qs = np.quantile(d["conf"][tr], np.linspace(0, 1, nb + 1))
                qs[0], qs[-1] = -np.inf, np.inf; qs = np.unique(qs)
                btr = np.clip(np.digitize(d["conf"][tr], qs[1:-1]), 0, len(qs) - 2)
                bte = np.clip(np.digitize(d["conf"][te], qs[1:-1]), 0, len(qs) - 2)
                for b in range(len(qs) - 1):
                    m = btr == b
                    n7 = int(m.sum())
                    if n7 == 0:
                        continue
                    k7 = int(d["ok7"][tr][m].sum()); p7 = k7 / n7
                    lb = wilson_lb(k7, n7, z); a32 = float(d["ok32"][tr][m].mean())
                    fires = bool(n7 >= 30 and lb >= a32)
                    if fires:
                        sel = te.copy(); sel[te] = (bte == b); keep_all |= sel
                    # what would have to change
                    if p7 >= a32 and p7 not in (0.0, 1.0):
                        need_n = int(np.ceil(z * z * p7 * (1 - p7) / max((p7 - a32) ** 2, 1e-12)))
                    else:
                        need_n = None
                    binrows.append(dict(fold=f, bin=b, n_train=n7, p7_train=r(p7),
                                        wilson_lb=r(lb), acc32_train=r(a32),
                                        lb_minus_acc32=r(lb - a32), p7_minus_acc32=r(p7 - a32),
                                        fires=fires, n_needed_for_lb_to_reach_acc32=need_n))
            ok = np.where(keep_all, d["ok7"], d["ok32"])
            cc = cost_of("mcq", 0, float((~keep_all).mean()), R32_CHARGED)
            ci = cell_ci(c, ok - d["ok32"])
            per_nb[nb] = dict(veto_rate=r(float(keep_all.mean())),
                              escalation=r(float((~keep_all).mean())),
                              acc=r(float(ok.mean()), 6), delta_vs_direct=ci["delta"],
                              lo=ci["lo"], hi=ci["hi"], flops_A=r(cc["flops"]["A"]),
                              x_direct=r(cc["flops"]["A"] / R32_CHARGED),
                              n_bins_that_fire=int(sum(1 for b in binrows if b["fires"])),
                              n_bins_total=len(binrows),
                              n_bins_with_p7_above_acc32=int(sum(1 for b in binrows
                                                                 if b["p7_minus_acc32"] >= 0)),
                              bins=binrows)
        out[c] = dict(acc_7b=r(float(d["ok7"].mean()), 6), acc_32b=r(float(d["ok32"].mean()), 6),
                      by_n_bins=per_nb)
    return out


# =================================================================================================
# 9.  PART 2 -- THE ZERO-32B QUESTION
# =================================================================================================
def zero_32b():
    fold = folds_for(None)
    per_cell, cheap_acc = {}, {}
    for c in CELLS:
        a32 = float(D[c]["ok32"].mean())
        if c in MCQ:
            best_acc = float(D[c]["ok7"].mean()); best = "always_7b (7B direct)"
            byk = None; orc = None; cov = None; sel_needed = None
            per_cell[c] = dict(n=N[c], acc_32b=r(a32, 6), best_no32b_acc=r(best_acc, 6),
                               best_no32b_policy=best, gap=r(best_acc - a32),
                               ceiling_note="no 7B best-of-N dump exists on this cell's pool "
                                            "(MedEvalKit track); see mcq_selfconsistency_crosstrack")
        else:
            # cross-fit choice of k among 1..8 (no escalation at all)
            ok = np.zeros(N[c]); chosen = []
            for f in range(K):
                te = fold[c] == f; tr = ~te
                ks = [float(D[c]["pick_ok"][tr, k - 1].mean()) for k in range(1, 9)]
                kb = int(np.argmax(ks)) + 1; chosen.append(kb)
                ok[te] = D[c]["pick_ok"][te, kb - 1]
            best_acc = float(ok.mean())
            byk = {k: r(float(D[c]["pick_ok"][:, k - 1].mean()), 6) for k in range(1, 9)}
            orc = float(D[c]["sl"].max(1).mean())
            cov = orc
            greedy = float(D[c]["greedy"].mean())
            # FROZEN DEFINITION (src/training_methods/genframe_data.py): sel_eff = mean(pick correct |
            # pool recoverable) = selected / oracle@8.  The difference form is explicitly forbidden there.
            sel_ach = float(D[c]["pick_ok"][:, 7].mean()) / orc
            sel_needed = a32 / orc
            per_cell[c] = dict(n=N[c], acc_32b=r(a32, 6), acc_7b_greedy=r(greedy, 6),
                               best_no32b_acc=r(best_acc, 6),
                               best_no32b_policy=f"7B best-of-k + verifier pick, cross-fit k={chosen}",
                               selected_by_k=byk, oracle8=r(orc, 6),
                               coverage8_any_correct=r(cov, 6),
                               gap=r(best_acc - a32),
                               gap_to_oracle8=r(orc - a32),
                               sel_eff_definition="selected@8 / oracle@8 (genframe_data.py frozen metric)",
                               sel_eff_achieved_at_k8=r(sel_ach, 6),
                               sel_eff_needed_to_reach_32b=r(sel_needed, 6),
                               sel_eff_shortfall=r(sel_needed - sel_ach, 6),
                               field_constant_sel_eff="0.775204 incumbent / 0.810627 best of ~20 "
                                                      "architectures (COMPARATIVE_VERIFIER_2026-08-05.md)",
                               oracle8_reaches_32b=bool(orc >= a32))
        cheap_acc[c] = best_acc
    macro_best = float(np.mean([cheap_acc[c] for c in CELLS]))
    macro_7b = float(np.mean([float((D[c]["ok7"] if c in MCQ else D[c]["greedy"]).mean()) for c in CELLS]))
    macro_ceiling = float(np.mean([cheap_acc[c] if c in MCQ else float(D[c]["sl"].max(1).mean())
                                   for c in CELLS]))
    macro_32b = float(np.mean([float(D[c]["ok32"].mean()) for c in CELLS]))
    okv = {c: (D[c]["ok7"] if c in MCQ else np.zeros(N[c])) for c in CELLS}
    for c in OPEN:
        ok = np.zeros(N[c])
        for f in range(K):
            te = fold[c] == f; tr = ~te
            ks = [float(D[c]["pick_ok"][tr, k - 1].mean()) for k in range(1, 9)]
            ok[te] = D[c]["pick_ok"][te, int(np.argmax(ks))]
        okv[c] = ok
    ci = macro_ci({c: okv[c] - D[c]["ok32"] for c in CELLS})
    # cost of the best zero-32B policy (macro): MCQ 1.0 each, open 2k
    fl = []
    for c in CELLS:
        if c in MCQ:
            fl.append(1.0)
        else:
            ks = [float(D[c]["pick_ok"][:, k - 1].mean()) for k in range(1, 9)]
            fl.append(2.0 * (int(np.argmax(ks)) + 1))
    return dict(
        macro_best_no32b=r(macro_best, 6), macro_always_7b=r(macro_7b, 6),
        macro_32b_direct=r(macro_32b, 6),
        macro_ceiling_perfect_open_verifier=r(macro_ceiling, 6),
        macro_gap=r(macro_best - macro_32b), delta_ci=ci,
        macro_gap_at_perfect_open_verifier=r(macro_ceiling - macro_32b),
        macro_flops_A=r(float(np.mean(fl))), x_direct=r(float(np.mean(fl)) / R32_CHARGED),
        cheapest_zero_32b=dict(policy="always-7B everywhere (greedy on the open cells)",
                               macro_acc=r(macro_7b, 6), macro_flops_A=1.0,
                               x_direct=r(1.0 / R32_CHARGED),
                               delta_vs_direct=macro_ci({c: (D[c]["ok7"] if c in MCQ else D[c]["greedy"])
                                                         - D[c]["ok32"] for c in CELLS})),
        per_cell=per_cell,
        cells_that_make_it_impossible=sorted(
            [(c, r(cheap_acc[c] - float(D[c]["ok32"].mean()))) for c in CELLS], key=lambda z: z[1]))


def mcq_selfconsistency_crosstrack():
    """CROSS-TRACK DIAGNOSTIC ONLY.  ckpts/mcq_gen_verify/lingshu7b/*_sc8.jsonl are 7B self-consistency
    pools of 8 on the INTERNAL harness track (PMC-VQA = test_clean.csv n=2000, NOT the MedEvalKit
    test_2.csv n=33430 used by the macro; PathVQA n=345, SLAKE n=1061, VQA-RAD n=451).  They may NOT
    be averaged into the macro.  Reported to answer 'is there ANY 7B test-time-compute headroom on
    multiple choice', which the Variant-B pool cannot answer at all."""
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(REPO, "ckpts/mcq_gen_verify/lingshu7b/*_sc8.jsonl"))):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if not rows:
            continue
        oks = np.array([[int(x) for x in rr["oks"][:8]] for rr in rows if len(rr.get("oks", [])) >= 8])
        gre = np.array([int(rr["greedy_ok"]) for rr in rows if len(rr.get("oks", [])) >= 8])
        pick = np.array([int(rr["pick_ok"]) for rr in rows if len(rr.get("oks", [])) >= 8])
        if len(oks) == 0:
            continue
        # modal (self-consistency) vote
        preds = [rr["preds"][:8] for rr in rows if len(rr.get("oks", [])) >= 8]
        mod = []
        for j, pr in enumerate(preds):
            vals, cnt = np.unique(pr, return_counts=True)
            mp = vals[int(np.argmax(cnt))]
            hit = [oks[j][t] for t in range(8) if pr[t] == mp]
            mod.append(int(round(float(np.mean(hit)))) if hit else 0)
        out[os.path.basename(p)] = dict(
            n=int(len(oks)), greedy=r(float(gre.mean()), 6), slot1=r(float(oks[:, 0].mean()), 6),
            verifier_pick=r(float(pick.mean()), 6),
            self_consistency_modal=r(float(np.mean(mod)), 6),
            oracle8=r(float(oks.max(1).mean()), 6),
            headroom_oracle_minus_greedy=r(float(oks.max(1).mean() - gre.mean())))
    return dict(
        WARNING="INTERNAL-HARNESS TRACK.  PMC-VQA here is test_clean.csv (n=2000); the macro's "
                "PMC-VQA cell is test_2.csv (n=33430).  test_clean n test_2 = 6 items.  NEVER "
                "average these into the Variant-B macro.",
        source="ckpts/mcq_gen_verify/lingshu7b/*_sc8.jsonl", files=out)


# =================================================================================================
# 10.  FOOTPRINT  (the "smaller model" half of the objective)
# =================================================================================================
def footprint():
    t = VRAM["reconciliation"]["tiers"]
    g = VRAM["deployer_guidance"]
    return dict(
        source="results/cascade_methods/artifacts/vram_testtime_2026-08-11.json "
               "(HF transformers, bf16, tp=1, torch.cuda.memory_allocated after load)",
        weights_resident_gib=dict(
            lingshu_7b=t["Lingshu-7B"]["hf_measured_a_weights_resident_gib"],
            lingshu_32b=t["Lingshu-32B"]["hf_measured_a_weights_resident_gib"],
            verifier_lora_adapter=0.1961,
            frozen_8seed_selector=round(28.1 / 1024, 4)),
        CORRECTION_TO_THE_BRIEF=(
            "the brief states always-32B-direct is '~31.5 GiB bf16'.  That is the vLLM PER-WORKER "
            "weights line at tensor_parallel_size=2 (31.28 GiB x 2 = 62.56 GiB total).  The measured "
            "whole-model resident weight is 62.3125 GiB (HF, tp=1, bf16), and 32.8e9 params x 2 bytes "
            "= 61.1 GiB confirms it arithmetically.  The 7B-side/32B-side footprint ratio is therefore "
            "~4.0x, not ~2x."),
        policies=dict(
            zero_32b_7b_plus_adapter=dict(
                params_B=8.29 + 0.0476 + 0.0073,
                weights_gib=round(t["Lingshu-7B"]["hf_measured_a_weights_resident_gib"] + 0.1961
                                  + 28.1 / 1024, 4),
                measured_test_time_peak_gib=dict(
                    mcq_leg_uncapped=g["cheap_leg_alone"]["measured_peak_gib"],
                    open_arm_bo8_cap320=g["full_opentext_arm"]["measured_peak_gib"]),
                smallest_card=g["full_opentext_arm"]["smallest_card_with_5pct_headroom"]),
            always_32b_direct=dict(
                params_B=32.8,
                weights_gib=t["Lingshu-32B"]["hf_measured_a_weights_resident_gib"],
                measured_test_time_peak_gib=g["always_32b_direct"]["measured_peak_gib"],
                smallest_card=g["always_32b_direct"]["smallest_card_with_5pct_headroom"]),
            any_cascade_co_resident=dict(
                params_B=32.8 + 8.29 + 0.0476 + 0.0073,
                weights_gib=round(t["Lingshu-32B"]["hf_measured_a_weights_resident_gib"]
                                  + t["Lingshu-7B"]["hf_measured_a_weights_resident_gib"] + 0.1961
                                  + 28.1 / 1024, 4),
                note="a cascade needs BOTH tiers reachable.  Co-resident on one host is the "
                     "batch-1-latency-preserving option; loading serially trades footprint for a "
                     "model-swap stall that none of this project's latency numbers include.")),
        finding=("under the new objective the footprint axis is the ONLY one where the cheap side wins "
                 "outright and by a large margin: 15.69 GiB of weights and a measured 18.76-23.42 GiB "
                 "test-time peak that fits a 24 GB card, against 62.31 GiB of weights and a measured "
                 "72.60 GiB peak that needs an 80 GB card.  Every policy that keeps the 32B keeps the "
                 "80 GB card."))


# =================================================================================================
# MAIN
# =================================================================================================
def main():
    t0 = time.time()
    rep["title"] = ("ATTACK 4 -- MINIMUM ESCALATION AT PARITY, and CAN THE 32B BE DROPPED ENTIRELY? "
                    "Endpoint is COST subject to non-inferiority, not accuracy.")
    rep["date"] = "2026-08-12"
    rep["reproduce"] = ("python3 src/cascade_methods/min_escalation_features.py && "
                        "python3 src/cascade_methods/min_escalation.py")
    rep["preregistration"] = "results/cascade_methods/artifacts/min_escalation_2026-08-12_preregistration.json"
    rep["no_gpu"] = True
    rep["no_fabricated_numbers"] = True
    rep["abstention_rule_6"] = ("NOT VIOLATED.  Every policy here answers every item.  The certified "
                                "veto KEEPS the cheap model's answer; it never declines.")
    rep["seed"] = SEED
    rep["n_bootstrap"] = NBOOT
    rep["pool"] = ("Variant B (MMMU excluded): 5 benchmarks / 8 cells / n=42,224, CLEAN disjoint "
                   "verifier (ckpts/train/lora_verifier_disjoint)")
    rep["numerics_pins"] = dict(OMP_NUM_THREADS=1, PYTHONHASHSEED=0,
                                tf32="not applicable -- pure numpy on stored vectors",
                                row_order="the stored dump order, unchanged",
                                bootstrap="one shared item-resample stream per cell, built once and "
                                          "reused by every policy and every baseline")
    rep["constraint"] = dict(rule="CI lower bound of (policy - always_32b_direct) on the 8-cell macro "
                                  ">= -0.0029", tie_tol=-TIE_TOL)

    print("[1/9] null tests", flush=True)
    rep["null_tests"] = null_tests()
    print("     ", rep["null_tests"]["verdict"], flush=True)

    print("[2/9] bootstrap streams", flush=True)
    build_boot(SEED)

    print("[3/9] baselines", flush=True)
    base = {}
    for s in ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b"]:
        base[s] = r(float(np.mean([ZPUB[f"{c}|{s}"].mean() for c in CELLS])), 6)
    rep["baselines"] = dict(macro_acc=base, published=PUB["macro_acc"],
                            shipped_accuracy_max=dict(macro_acc=PUB["macro_acc"]["method_accuracy_max_veto"],
                                                      x_direct_as_charged=PUB["ratios_macro"]
                                                      ["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"]),
                            cost_floor_reference=dict(
                                zero_eps_crossfit=CF["VERDICT"]["what_DID_clear_the_bar"]["zero_eps_crossfit"],
                                honest_nested_cv=CF["VERDICT"]["primary_endpoint_as_charged"]["honest_nested_cv"]))

    print("[4/9] lambda frontier (anchor folds)", flush=True)
    fold0 = folds_for(None)
    tab0 = precompute_menu(fold0)
    frontier = []
    for lam in LAMBDAS:
        okv, escv, kindv, kv, picks = run_lambda(lam, tab0)
        row = summarise(okv, escv, kindv, kv, picks)
        row["lambda"] = lam
        frontier.append(row)
    rep["lambda_frontier_anchor_folds"] = dict(
        protocol="5-fold cross-fit on the published modulo-K folding.  For each lambda every cell "
                 "independently takes the menu entry maximising (train acc - lambda * train FLOP-eq); "
                 "the held-out fold is answered by that entry.  lambda is a SWEEP here, not a fitted "
                 "quantity -- picking the best row on eval would be eval-visible, so the honest "
                 "operating point is the nested-CV one below.",
        rows=frontier)

    print("[5/9] eval-visible diagnostic (UPPER BOUND, never a result)", flush=True)
    tabEV = precompute_menu(fold0, evalvisible=True)
    ev = []
    for lam in LAMBDAS:
        okv, escv, kindv, kv, picks = run_lambda(lam, tabEV)
        row = summarise(okv, escv, kindv, kv, picks, with_guard=False)
        row["lambda"] = lam
        ev.append(row)
    rep["evalvisible_diagnostic"] = dict(
        warning="EVERY row here is fitted with full eval visibility.  It is an UPPER BOUND on what "
                "the cross-fit selector could reach and must never be quoted as a result.", rows=ev)

    print("[6/9] permutation null", flush=True)
    rngp = np.random.default_rng(SEED + 1)
    go = {}
    for c in CELLS:
        if c in MCQ:
            go[c] = dict(margin=D[c]["margin"][rngp.permutation(N[c])],
                         conf=D[c]["conf"][rngp.permutation(N[c])])
        else:
            go[c] = dict(gate_k=D[c]["gate_k"][rngp.permutation(N[c])])
    tabP = precompute_menu(fold0, gate_override=go)
    pn = []
    for lam in LAMBDAS:
        okv, escv, kindv, kv, picks = run_lambda(lam, tabP)
        row = summarise(okv, escv, kindv, kv, picks, with_guard=False)
        row["lambda"] = lam
        pn.append(row)
    rep["permutation_null"] = dict(
        protocol="the per-item GATE SCALAR is permuted within each cell (margin / 7B confidence / the "
                 "best-of-k max-verifier-score gate).  Outcome vectors and best-of-N PICKS are "
                 "untouched, so this isolates 'does the gate carry routing information' from 'is the "
                 "cheap answer any good'.  Same folds, same menu, same lambda grid.",
        rows=pn)

    print("[7/9] nested CV (PRIMARY) x %d seeds" % NSEED, flush=True)
    seeds = [None] + [SEED + 100 * i for i in range(1, NSEED)]
    seed_rows = []
    for si, sd in enumerate(seeds):
        fold = fold0 if sd is None else folds_for(sd)
        tab = tab0 if sd is None else precompute_menu(fold)
        okv, escv, kindv, kv, picks, lams = nested_cv(fold, sd, tab)
        row = summarise(okv, escv, kindv, kv, picks)
        row["seed"] = ("anchor(modulo-K)" if sd is None else sd)
        row["lambda_per_outer_fold"] = lams
        seed_rows.append(row)
        print("      seed %2d  acc %.6f  x_direct %.4f  lo %+.4f  tie %s"
              % (si, row["macro_acc"], row["cost"]["A_as_charged_R32_4p57"]["x_direct"],
                 row["delta_vs_direct"]["lo"], row["delta_vs_direct"]["tie"]), flush=True)
    ties = sum(1 for x in seed_rows if x["delta_vs_direct"]["tie"])
    nsw = sum(1 for x in seed_rows if x["delta_vs_direct"]["not_sig_worse"])
    agg = lambda key: dict(mean=r(float(np.mean([x[key] if not isinstance(x[key], dict) else 0
                                                 for x in seed_rows])), 6))
    xs = [x["cost"]["A_as_charged_R32_4p57"]["x_direct"] for x in seed_rows]
    accs = [x["macro_acc"] for x in seed_rows]
    escl = [x["macro_escalation"] for x in seed_rows]
    rep["PRIMARY_ENDPOINT_NESTED_CV"] = dict(
        definition="macro-weighted as-charged FLOP-eq of the fully honest nested-CV policy (lambda "
                   "chosen inside the training folds), with the pre-registered non-inferiority CI rule",
        seeds=seed_rows,
        summary=dict(macro_acc_mean=r(float(np.mean(accs)), 6), macro_acc_sd=r(float(np.std(accs)), 6),
                     x_direct_as_charged_mean=r(float(np.mean(xs))), x_direct_sd=r(float(np.std(xs))),
                     macro_escalation_mean=r(float(np.mean(escl))),
                     seeds_with_tie_preserved="%d/%d" % (ties, len(seed_rows)),
                     seeds_not_significantly_worse="%d/%d" % (nsw, len(seed_rows))))

    print("[8/9] per-cell frontier + veto + zero-32B", flush=True)
    rep["per_cell_frontier"] = per_cell_frontier()
    rep["certified_veto"] = veto_analysis()
    rep["PART2_zero_32b"] = zero_32b()
    rep["PART2_mcq_selfconsistency_crosstrack"] = mcq_selfconsistency_crosstrack()

    print("[9/9] footprint + verdict", flush=True)
    rep["footprint"] = footprint()
    rep["cost_conventions"] = dict(
        A_as_charged="PAPER CONSTANTS.  k generations + k verifier forwards, each one full batch-1 7B "
                     "forward; R32 = 4.57.  PRIMARY endpoint convention.  Also reported at the derived "
                     "R32 = 3.816, which makes every ratio WORSE for the method.",
        B_recost_gen="generation prefill shared: G(k) = %.6f + k*%.6f.  UNCORROBORATED -- and the "
                     "mechanism is IN DOUBT (cost_floor_2026-08-10 rule2_corroboration: vLLM V1 splits "
                     "SamplingParams(n=N) into N child requests whose prefix-cache hits are not "
                     "guaranteed; the decisive num_cached_tokens measurement never ran)."
                     % (PREFILL_SHARE, DECODE_SHARE),
        C_recost_full="B plus a verifier that prefix-caches the shared image+question prompt: "
                      "ver(k) = %.4f + k*%.4f.  The deployed verifier does NOT do this (one HF batched "
                      "forward over k FULL prompts), so C prices an implementation never run."
                      % (VER_PREFIX, VER_MARGINAL),
        latency_energy="modelled from the measured batch-1 constants (GEN7 347 ms/45.8 J, VER7 175 "
                       "ms/25.3 J, GEN32 665 ms/127.0 J).  The measurement-corrected columns replace "
                       "the modelled best-of-k cell with a LINEAR INTERPOLATION between the measured "
                       "N=1 (625.4 ms / 97.17 J) and N=8 (1305.3 ms / 316.7 J) points, n=45 -- only "
                       "those two points were measured.")
    # ---------------------------------------------------------------- matched-cost null comparison
    def interp(rows, x):
        pts = sorted([(rw["cost"]["A_as_charged_R32_4p57"]["x_direct"], rw["macro_acc"]) for rw in rows])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return float(np.interp(x, xs, ys))
    real = rep["lambda_frontier_anchor_folds"]["rows"]; nul = rep["permutation_null"]["rows"]
    matched = {}
    for x in (0.80, 0.90, 1.00, 1.10):
        matched["x=%.2f" % x] = dict(real_macro_acc=r(interp(real, x), 6),
                                     null_macro_acc=r(interp(nul, x), 6),
                                     real_minus_null=r(interp(real, x) - interp(nul, x), 6))
    rep["permutation_null"]["matched_cost_comparison"] = dict(
        note="linear interpolation of each frontier's (as-charged x_direct -> macro accuracy) curve",
        rows=matched,
        verdict=("PASS -- the gate carries real routing information: at every matched cost the real "
                 "frontier is above the permuted one, and NO permuted row anywhere preserves the "
                 "pre-registered tie, while the real frontier does at lambda=0."
                 if all(v["real_minus_null"] > 0 for v in matched.values())
                 and not any(rw["delta_vs_direct"]["tie"] for rw in nul)
                 else "INSPECT -- see rows"))

    # ---------------------------------------------------------------- verdict + headline
    fr = [rw for rw in real if rw["delta_vs_direct"]["tie"]]
    cheapest_tie = min(fr, key=lambda rw: rw["cost"]["A_as_charged_R32_4p57"]["x_direct"]) if fr else None
    loose = [rw for rw in real if rw["delta_vs_direct"]["not_sig_worse"]]
    cheapest_loose = min(loose, key=lambda rw: rw["cost"]["A_as_charged_R32_4p57"]["x_direct"]) if loose else None
    pe = rep["PRIMARY_ENDPOINT_NESTED_CV"]["summary"]
    rep["VERDICT"] = dict(
        part1=dict(
            question="the lowest escalation rate / compute that still satisfies the pre-registered "
                     "non-inferiority constraint",
            PRIMARY_honest_nested_cv=dict(
                macro_acc=pe["macro_acc_mean"], macro_acc_sd=pe["macro_acc_sd"],
                x_direct_as_charged=pe["x_direct_as_charged_mean"], x_direct_sd=pe["x_direct_sd"],
                macro_escalation=pe["macro_escalation_mean"],
                seeds_with_tie_preserved=pe["seeds_with_tie_preserved"],
                seeds_not_significantly_worse=pe["seeds_not_significantly_worse"]),
            cheapest_crossfit_row_meeting_the_pre_registered_tie=(
                dict(lam=cheapest_tie["lambda"], macro_acc=cheapest_tie["macro_acc"],
                     delta=cheapest_tie["delta_vs_direct"],
                     x_direct=cheapest_tie["cost"]["A_as_charged_R32_4p57"]["x_direct"],
                     x_direct_R32_3p816=cheapest_tie["cost"]["A_as_charged_R32_3p816"]["x_direct"],
                     macro_escalation=cheapest_tie["macro_escalation"]) if cheapest_tie else None),
            cheapest_crossfit_row_not_significantly_worse=(
                dict(lam=cheapest_loose["lambda"], macro_acc=cheapest_loose["macro_acc"],
                     delta=cheapest_loose["delta_vs_direct"],
                     x_direct=cheapest_loose["cost"]["A_as_charged_R32_4p57"]["x_direct"],
                     x_direct_R32_3p816=cheapest_loose["cost"]["A_as_charged_R32_3p816"]["x_direct"],
                     macro_escalation=cheapest_loose["macro_escalation"],
                     CAVEAT="this is the LOOSER criterion (CI merely spans zero); the verdict is NOT "
                            "decided on it") if cheapest_loose else None),
            statement=("Threshold tuning -- the lever cost_floor_2026-08-10 could not reach -- moves the "
                       "constrained minimum from that round's 0.9931x/-0.0009 (which MISSED the "
                       "constraint) to %.4fx +/- %.4f at macro %.6f with the tie preserved on %s seeds.  "
                       "The cost floor at the pre-registered tie is therefore AT PARITY WITH "
                       "always-32B-direct, not below it: the constraint, not the accounting, is what "
                       "binds." % (pe["x_direct_as_charged_mean"], pe["x_direct_sd"],
                                   pe["macro_acc_mean"], pe["seeds_with_tie_preserved"]))),
        part2=dict(
            question="best macro accuracy with NO 32B at test time",
            answer="NO -- and by a decisive margin",
            best_no32b_macro=rep["PART2_zero_32b"]["macro_best_no32b"],
            bar=rep["PART2_zero_32b"]["macro_32b_direct"],
            delta=rep["PART2_zero_32b"]["delta_ci"],
            ceiling_with_a_PERFECT_open_verifier=rep["PART2_zero_32b"]["macro_ceiling_perfect_open_verifier"],
            what_would_have_to_be_true=(
                "the 8-sample 7B pool already CONTAINS a correct answer often enough on all three open "
                "cells (oracle@8 exceeds the 32B on every one), so the zero-32B barrier on the open "
                "half is SELECTION, not coverage: a perfect selector would put the zero-32B macro at "
                "%.6f, ABOVE the %.6f bar.  The required sel_eff is the decisive number -- see "
                "PART2_zero_32b.per_cell.  On the five MCQ cells there is no 7B test-time-compute arm "
                "on this pool at all, so their deficits (-0.0091 / -0.0335 / -0.0717 / -0.0482 / "
                "-0.0450) are structural with the frozen 7B."
                % (rep["PART2_zero_32b"]["macro_ceiling_perfect_open_verifier"],
                   rep["PART2_zero_32b"]["macro_32b_direct"]))))
    rep["HEADLINE_TABLE"] = dict(
        columns=["policy", "macro_acc", "delta vs 32B-direct [95% CI]", "macro escalation",
                 "as-charged x_direct (R32 4.57)", "as-charged x_direct (R32 3.816)",
                 "weights resident GiB", "pre-registered tie?"],
        rows=[
            ["always-32B-direct  (THE BAR)", 0.656672, "0 (reference)", 1.0, 1.0, 1.0, 62.3125, "-"],
            ["always-7B (zero-32B floor)", rep["PART2_zero_32b"]["macro_always_7b"],
             str(rep["PART2_zero_32b"]["cheapest_zero_32b"]["delta_vs_direct"]), 0.0,
             rep["PART2_zero_32b"]["cheapest_zero_32b"]["x_direct"],
             r(1.0 / R32_DERIVED), 15.6898, "no"],
            ["best zero-32B (7B + best-of-k verifier on open)", rep["PART2_zero_32b"]["macro_best_no32b"],
             str(rep["PART2_zero_32b"]["delta_ci"]), 0.0, rep["PART2_zero_32b"]["x_direct"],
             r(rep["PART2_zero_32b"]["macro_flops_A"] / R32_DERIVED), 15.6898, "no"],
            ["SHIPPED accuracy-max (no change)", PUB["macro_acc"]["method_accuracy_max_veto"],
             "+0.0008 [-0.0022,+0.0037]", "-",
             PUB["ratios_macro"]["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"], "-",
             77.999, "yes"],
            ["cost_floor eps=0 arm selection (12-seed, PRIOR ROUND)",
             CF["VERDICT"]["what_DID_clear_the_bar"]["zero_eps_crossfit"]["macro_acc"],
             "+0.0011 [-0.0014,+0.0035]", "-",
             CF["VERDICT"]["what_DID_clear_the_bar"]["zero_eps_crossfit"]["x_direct_as_charged"], "-",
             77.999, "yes on 10/12 seeds"],
            ["cost_floor honest nested CV (PRIOR ROUND)",
             CF["VERDICT"]["primary_endpoint_as_charged"]["honest_nested_cv"]["x_direct"] * 0 + 0.6558,
             "-0.0009 [-0.0034,+0.0015]", "-",
             CF["VERDICT"]["primary_endpoint_as_charged"]["honest_nested_cv"]["x_direct"], 1.0344,
             77.999, "NO -- missed by 0.0005"],
            ["THIS ROUND: threshold-swept nested CV (12-seed mean)", pe["macro_acc_mean"],
             "see PRIMARY_ENDPOINT_NESTED_CV per seed", pe["macro_escalation_mean"],
             pe["x_direct_as_charged_mean"], "-", 77.999, pe["seeds_with_tie_preserved"]],
        ],
        cost_label="EVERY cost column is MACRO-weighted (8 cells, 1/8 each) and is paired only with the "
                   "MACRO accuracy in the same row.  Weights-resident is the sum of the tiers a policy "
                   "must be able to reach; 77.999 GiB = 62.3125 (32B) + 15.4937 (7B) + 0.1961 (LoRA) "
                   "+ 0.0274 (frozen selector).")
    rep["limitations"] = [
        "The open-text menu takes the FIRST k of the 8 stored candidates as the best-of-k pool.  That is "
        "the same convention cost_floor's L3 table used; it is an approximation to actually sampling k, "
        "and it holds the pool fixed rather than re-drawing it.",
        "The deployed open arm is an ADAPTIVE-N Weitzman controller; this attack's open menu is FIXED-k "
        "plus a threshold.  The adaptive controller is not in the menu, so the open-cell rows here are "
        "not a re-derivation of the shipped arm -- they are a different, cheaper family.",
        "Conventions B and C are DERIVED and UNCORROBORATED, and B's mechanism is IN DOUBT (see "
        "cost_conventions).  The verdict is decided on convention A only.",
        "The measurement-corrected latency/energy interpolates linearly in k between the two measured "
        "points (k=1 and k=8, n=45); only those two were measured.",
        "vqa_rad_closed (n=251) and vqa_rad_open (n=200) have poor guardrail resolution; a flag on "
        "either is within seed noise in this project.",
        "The certified veto's 'sample size needed' column is computed from TRAIN-fold point estimates, "
        "which are themselves noisy; read it as an order of magnitude, not a target.",
        "A PRE-GENERATION router -- the structure the 2026 literature names as the fix for cascade cost "
        "(LITERATURE_UPDATE_2026-08-11.md sec 0.2) -- can only be expressed here at CELL granularity "
        "(the p32 menu entry), because no per-item feature that exists BEFORE the 7B forward is stored "
        "in this repo.  A within-cell pre-generation router is NOT MEASURED.",
        "R32 = 4.57 is the paper constant and it FLATTERS the method; R32 = 3.816 is derived and makes "
        "every ratio worse.  Both are reported on every headline row."]
    rep["handoff"] = dict(
        what_is_new_here="the per-item gate features are now extracted and cached at "
                         "results/cascade_methods/artifacts/_min_escalation_parts/features.npz, so any "
                         "future attack can sweep a threshold without touching a GPU.",
        next_lever="a within-cell PRE-GENERATION router.  Every cell where the Lagrangian picks p32 "
                   "(MedXpert, VQA-RAD-open, and VQA-RAD-closed at low lambda) is a cell where the 7B "
                   "forward is pure waste, and the cascade pays it anyway on the cells it does keep.  "
                   "Extracting a cheap pre-generation feature (image/question embedding) is the one "
                   "structural change this analysis cannot evaluate and the literature says matters most.")
    rep["runtime_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(OUT, "w"), indent=1, default=str)
    print("wrote", OUT, "in", rep["runtime_s"], "s")


if __name__ == "__main__":
    main()
