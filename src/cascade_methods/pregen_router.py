#!/usr/bin/env python3
"""
pregen_router.py -- ATTACK 2: THE PRE-GENERATION ROUTER.

THE OBJECTIVE (changed by the researcher on 2026-08-12):
    MINIMISE macro-weighted FLOP-eq  SUBJECT TO  the paired-bootstrap CI lower bound of
    (policy - always_32B_direct) on the 8-cell macro being >= -0.0029.
A policy that is CHEAPER AND TIES is a WIN.

THE ATTACK.  A cascade pays the cheap model on EVERY question before it can decide to escalate.
A PRE-GENERATION ROUTER decides FROM THE PROMPT (and image) ALONE which single model answers, and
pays only that one.  Published structural theorem, Bouchard arXiv:2605.06350 via
docs/current/LITERATURE_UPDATE_2026-08-11.md:
    "A lightweight pre-generation router exceeds the best cascade policy on four of five datasets,
     mainly because it avoids the cheap model's generation cost on queries sent directly to a larger
     model rather than because of a stronger routing signal.  ...  cascade performance is limited
     primarily by structural cost, since cascades pay the cheap model before any escalation decision."
Our escalation rates run 8.45% (PMC) to 89.60% (MedXpert), and on MedXpert the shipped cascade already
costs 5.095 FLOP-eq against always-32B-direct's 4.570 -- strictly worse than just running the strong
model.  Nobody in this project has ever evaluated a pre-generation router.

WHAT THE ROUTER MAY SEE.  Only features available BEFORE any generation, built by
src/cascade_methods/pregen_router_features.py: question-text statistics, option strings, the
prompt-only format flag (unified_router.detect_format), hashed word n-grams of the question, and
statistics of the DECODED image.  It may NOT see the 7B's response, margin, conf, cum_logprob,
gen_toks or latency -- that would make it a cascade, not a pre-router.  Enforced by null test N3.

DATASET IDENTITY -- three labelled variants, because "route on which benchmark it is" is close to
cheating and must never be sold as a prompt-only result:
    R-A  within-cell      -- trained and applied inside ONE cell; carries no cross-dataset information.
    R-B  pooled           -- one global model over all 8 cells, NO dataset feature.  Question text and
                             image style still leak dataset identity IMPLICITLY.  LABELLED.
    R-C  pooled + ds-1hot -- explicit dataset one-hot.  This IS dataset-identity routing.  LABELLED.

METHOD.  A single regressor estimates the ADVANTAGE h(x) = E[ok32 - ok7 | x].  The macro objective is
sum_i w_i (acc_i - mu * cost_i) with w_i = 1/(8 n_c), and macro cost uses the SAME weights, so the
Lagrangian optimum is a SINGLE GLOBAL THRESHOLD on h(x): send to the 32B iff h(x) > lambda.  Sweeping
lambda traces the frontier.  THE DELIVERABLE IS THE FRONTIER.

INTEGRITY.  5-fold cross-fit x 10 seeds (no item is ever scored by a model that saw it); a NESTED
outer CV that selects the router configuration AND the operating point inside training folds only;
a permutation null; a per-cell guardrail; and a paired item bootstrap nboot=10000 on ONE shared
resample stream per cell reused by every policy and every baseline.  Costs are reported under BOTH
R32 conventions.  Macro cost is never paired with sample-weighted accuracy.

No GPU.  Launch from the repo root:
    python3 src/cascade_methods/pregen_router.py
Writes results/cascade_methods/artifacts/pregen_router_2026-08-12.json
"""
import os, sys, json, time, hashlib
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse as sp

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_pregen_router_parts")
OUT = os.path.join(ART, "pregen_router_2026-08-12.json")

SEED = 20260812
NBOOT = 10000
NSEED = 10
KFOLD = 5
TIE_TOL = 0.0029
BLOCK = 200
NLAM = 61

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
R32_CHARGED = 4.57       # paper constant (lingshu_medeval_cascade.py:21)
R32_DERIVED = 3.816      # flop_ratio_derivation_2026-08-03.json:derived_ratio.R32_derived
LAT7, LAT32 = 347.0, 665.0        # measured batch-1 ms  (macro_disjoint per_cell_as_charged)
EN7, EN32 = 45.8, 127.0           # measured batch-1 J

rep = {}
T0 = time.time()


# =================================================================================================
# 0.  LOAD
# =================================================================================================
Z = np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"))
MAC = json.load(open(os.path.join(ART, "_selector_rerun_parts/macro_disjoint.json")))
PUB = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))["per_arm"]["disjoint"]
PCC = MAC["cost"]["per_cell_as_charged"]
FEATMAN = json.load(open(os.path.join(PARTS, "manifest.json")))

OK = {}
for c in CELLS:
    for s in PUB["per_cell_acc"][c]:
        if f"{c}|{s}" in Z:
            OK[(c, s)] = Z[f"{c}|{s}"].astype(np.float64)
N = {c: len(OK[(c, "always_32b_direct")]) for c in CELLS}
NTOT = sum(N.values())

FE, TEXTS = {}, {}
for c in CELLS:
    d = np.load(os.path.join(PARTS, f"features_{c}.npz"), allow_pickle=True)
    FE[c] = dict(X=d["X"].astype(np.float64), names=[str(x) for x in d["names"]],
                 ok7=d["ok7"].astype(np.float64), ok32=d["ok32"].astype(np.float64))
    TEXTS[c] = json.load(open(os.path.join(PARTS, f"texts_{c}.json")))
FNAMES = FE[CELLS[0]]["names"]
IMGCOLS = [i for i, n in enumerate(FNAMES) if n.startswith("img_")]
TXTCOLS = [i for i, n in enumerate(FNAMES) if not n.startswith("img_")]

ADV = {c: FE[c]["ok32"] - FE[c]["ok7"] for c in CELLS}
MACROW = {c: 1.0 / (len(CELLS) * N[c]) for c in CELLS}


def hash_cell(c):
    return int(hashlib.md5(c.encode()).hexdigest()[:8], 16) % 100000


def ci(d):
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# =================================================================================================
# 1.  NULL TESTS
# =================================================================================================
def null_n1():
    dev_a, dev_c, rows = [], [], {}
    for s in PUB["macro_acc"]:
        m = float(np.mean([OK[(c, s)].mean() for c in CELLS]))
        d = abs(m - PUB["macro_acc"][s]); dev_a.append(d)
        f = float(np.mean([PCC[c][s]["flops"] for c in CELLS]))
        dev_c.append(abs(f - PUB["cost_macro"][s]["flops"]))
        rows[s] = dict(acc_recomputed=round(m, 6), acc_published=PUB["macro_acc"][s],
                       acc_abs_dev=round(d, 8), cost_recomputed=round(f, 6),
                       cost_published=PUB["cost_macro"][s]["flops"],
                       cost_ratio_vs_direct=round(f / R32_CHARGED, 4))
    return dict(name="N1 -- reproduce the published 8-cell macro accuracy and macro as-charged cost",
                source_vectors="results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz",
                source_published="results/cascade_methods/artifacts/cascade_selector_rerun_2026-08-05"
                                 ".json:per_arm.disjoint",
                per_system=rows, max_abs_dev_accuracy=round(max(dev_a), 8),
                max_abs_dev_cost_flops=round(max(dev_c), 8),
                verdict=("PASS -- accuracy deviates only by the published file's 4-dp rounding; cost "
                         "reproduces to 4 dp" if max(dev_a) < 5e-5 and max(dev_c) < 5e-4 else "FAIL"))


def null_n2():
    rows, dev = {}, []
    for c in CELLS:
        d7 = float(np.abs(FE[c]["ok7"] - OK[(c, "always_7b")]).max())
        d32 = float(np.abs(FE[c]["ok32"] - OK[(c, "always_32b_direct")]).max())
        dev += [d7, d32]
        rows[c] = dict(n=N[c], n_from_features=len(FE[c]["ok7"]), abs_dev_ok7=d7, abs_dev_ok32=d32,
                       images_missing=FEATMAN[c]["images_missing"], text_md5=FEATMAN[c]["text_md5"])
    return dict(name="N2 -- the pre-generation feature cache is item-aligned to the published vectors",
                per_cell=rows, max_abs_dev=float(max(dev)),
                verdict=("PASS -- every cell reproduces ok7 and ok32 exactly (max abs dev 0.0)"
                         if max(dev) == 0.0 else "FAIL"))


BANNED = ["margin", "conf", "logprob", "response", "gen_tok", "latency", "seqlogprob",
          "verifier", "self_consistency", "n_distinct", "escalat", "greedy", "pick"]


def null_n3():
    bad = [n for n in FNAMES for b in BANNED if b in n.lower()]
    return dict(name="N3 -- no post-generation quantity entered the router's feature set",
                n_features=len(FNAMES), feature_names=FNAMES, banned_substrings=BANNED,
                banned_hits=bad,
                feature_provenance="src/cascade_methods/pregen_router_features.py builds every column "
                                   "from the question string, the option strings, "
                                   "unified_router.detect_format (prompt-only) and the DECODED image. "
                                   "results.json's margin / conf / cum_logprob / response / gen_toks / "
                                   "latency_s are never read.",
                verdict="PASS -- no banned substring appears in any feature name" if not bad else "FAIL")


# =================================================================================================
# 2.  DESIGN MATRICES
# =================================================================================================
HV = HashingVectorizer(n_features=2 ** 12, ngram_range=(1, 2), alternate_sign=False,
                       norm="l2", lowercase=True, analyzer="word")
HASH = {c: HV.transform(TEXTS[c]) for c in CELLS}
DS1HOT = {c: np.eye(len(CELLS))[CELLS.index(c)][None, :].repeat(N[c], axis=0) for c in CELLS}


def design(c, mode):
    X = FE[c]["X"]
    if mode == "dense":
        return X
    if mode == "txtstats":
        return X[:, TXTCOLS]
    if mode == "imgonly":
        return X[:, IMGCOLS]
    if mode == "dense_ds":
        return np.hstack([X, DS1HOT[c]])
    if mode == "full":
        return sp.hstack([sp.csr_matrix(X), HASH[c]], format="csr")
    if mode == "full_ds":
        return sp.hstack([sp.csr_matrix(X), HASH[c], sp.csr_matrix(DS1HOT[c])], format="csr")
    raise ValueError(mode)


def scale_fit(Xtr):
    """Scale-only (no mean centring) so a sparse design stays sparse; Ridge carries an unpenalised
    intercept.  Computed on TRAINING rows only.  Columns with (near-)zero train variance are DROPPED,
    never amplified."""
    if sp.issparse(Xtr):
        mu = np.asarray(Xtr.mean(axis=0)).ravel()
        X2 = Xtr.copy(); X2.data = X2.data ** 2
        sd = np.sqrt(np.maximum(np.asarray(X2.mean(axis=0)).ravel() - mu ** 2, 0.0))
    else:
        sd = Xtr.std(axis=0)
    inv = np.zeros_like(sd); g = sd > 1e-8; inv[g] = 1.0 / sd[g]
    return inv


def apply_scale(X, inv):
    if sp.issparse(X):
        return X.multiply(sp.csr_matrix(inv.reshape(1, -1))).tocsr()
    return X * inv


def ridge_normal(Xtr, ytr, wtr, Xte, alpha):
    """Weighted ridge by the PRIMAL normal equations (d <= 4,272 << n = 42,224, so this is exact and
    far cheaper than an iterative solver).  A ones column is appended and left UNPENALISED = the
    intercept.  Fully deterministic: no random state, no iteration count, no tolerance."""
    d = Xtr.shape[1]
    sw = np.sqrt(wtr)
    if sp.issparse(Xtr):
        Xa = sp.hstack([Xtr, sp.csr_matrix(np.ones((Xtr.shape[0], 1)))], format="csr")
        Xw = Xa.multiply(sp.csr_matrix(sw.reshape(-1, 1))).tocsr()
        G = np.asarray((Xw.T @ Xw).todense())
        b = np.asarray(Xw.T @ (sw * ytr)).ravel()
    else:
        Xa = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
        Xw = Xa * sw[:, None]
        G = Xw.T @ Xw
        b = Xw.T @ (sw * ytr)
    reg = np.full(d + 1, float(alpha)); reg[-1] = 0.0
    G = G + np.diag(reg)
    w = np.linalg.solve(G, b)
    return np.asarray(Xte @ w[:d]).ravel() + w[d]


def fit_predict(Xtr, ytr, wtr, Xte, model, seed):
    if model == "ridge":
        inv = scale_fit(Xtr)
        return ridge_normal(apply_scale(Xtr, inv), ytr, wtr, apply_scale(Xte, inv), 50.0)
    if model == "gbm":
        A = np.asarray(Xtr.todense()) if sp.issparse(Xtr) else Xtr
        B = np.asarray(Xte.todense()) if sp.issparse(Xte) else Xte
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_depth=4,
                                          min_samples_leaf=60, l2_regularization=1.0,
                                          random_state=seed, early_stopping=False)
        m.fit(A, ytr, sample_weight=wtr)
        return m.predict(B)
    raise ValueError(model)


def folds_for(c, seed):
    return np.random.default_rng(seed * 7919 + hash_cell(c)).permutation(N[c]) % KFOLD


def rowslice(X, m):
    return X[m] if not sp.issparse(X) else X[m]


def vstack(mats):
    return sp.vstack(mats, format="csr") if sp.issparse(mats[0]) else np.vstack(mats)


def crossfit_scores(variant, model, mode, seed, permute=False):
    """{cell: per-item HELD-OUT advantage score}.  variant: 'within' | 'pooled'."""
    fold = {c: folds_for(c, seed) for c in CELLS}
    y = ADV if not permute else {
        c: np.random.default_rng(seed * 104729 + hash_cell(c)).permutation(ADV[c]) for c in CELLS}
    X = {c: design(c, mode) for c in CELLS}
    out = {c: np.zeros(N[c]) for c in CELLS}
    if variant == "within":
        for c in CELLS:
            for f in range(KFOLD):
                tr, te = fold[c] != f, fold[c] == f
                if te.sum() == 0 or tr.sum() < 20:
                    continue
                out[c][te] = fit_predict(rowslice(X[c], tr), y[c][tr], np.ones(int(tr.sum())),
                                         rowslice(X[c], te), model, seed)
        return out
    for f in range(KFOLD):
        Xtr = vstack([rowslice(X[c], fold[c] != f) for c in CELLS])
        ytr = np.concatenate([y[c][fold[c] != f] for c in CELLS])
        wtr = np.concatenate([np.full(int((fold[c] != f).sum()), MACROW[c]) for c in CELLS])
        wtr = wtr / wtr.mean()
        Xte = vstack([rowslice(X[c], fold[c] == f) for c in CELLS])
        pr = fit_predict(Xtr, ytr, wtr, Xte, model, seed)
        o = 0
        for c in CELLS:
            m = int((fold[c] == f).sum())
            out[c][fold[c] == f] = pr[o:o + m]; o += m
    return out


def ensemble(list_of_scores):
    """Average the per-seed HELD-OUT scores after standardising each SEED on the POOLED (all-cell)
    mean and sd.  Standardising per CELL would destroy the between-cell level differences, and those
    levels are exactly what a single global threshold uses -- so the pooling is deliberate.  Every
    seed's score for item i still comes from a model that never saw item i, so this stays cross-fit;
    the averaging is a labelled variance reduction."""
    out = {c: np.zeros(N[c]) for c in CELLS}
    for sc in list_of_scores:
        allv = np.concatenate([sc[c] for c in CELLS])
        mu, sd = allv.mean(), max(allv.std(), 1e-12)
        for c in CELLS:
            out[c] += (sc[c] - mu) / sd
    return {c: out[c] / len(list_of_scores) for c in CELLS}


def between_only(score):
    """DIAGNOSTIC: replace every item's score by its CELL MEAN.  A global threshold on this is pure
    PER-CELL ALLOCATION -- it routes whole datasets, nothing within them."""
    return {c: np.full(N[c], float(score[c].mean())) for c in CELLS}


def within_only(score):
    """DIAGNOSTIC: per-cell centre and scale, removing all between-cell level information.  A global
    threshold on this routes roughly the same fraction out of every cell and can only exploit
    WITHIN-cell ranking."""
    return {c: (score[c] - score[c].mean()) / max(score[c].std(), 1e-12) for c in CELLS}


# =================================================================================================
# 3.  POLICY ARITHMETIC
# =================================================================================================
def macro_lam_grid(score, npts=NLAM):
    """lambda grid chosen at MACRO-weighted quantiles of the pooled score, so the frontier points are
    evenly spaced in macro 32B-fraction rather than in item count."""
    s = np.concatenate([score[c] for c in CELLS])
    w = np.concatenate([np.full(N[c], MACROW[c]) for c in CELLS])
    o = np.argsort(-s)
    cw = np.cumsum(w[o]) / w.sum()
    qs = np.linspace(0.0, 1.0, npts)
    lam = []
    for q in qs:
        k = int(np.searchsorted(cw, q))
        lam.append(s[o][min(k, len(o) - 1)] + (1e-12 if q == 0.0 else 0.0))
    lam = [s.max() + 1.0] + lam + [s.min() - 1.0]
    return np.array(sorted(set(lam), reverse=True))


def route_eval(score, lam):
    """Point estimates for the threshold policy 'send to 32B iff score > lam'."""
    val, frac = {}, {}
    for c in CELLS:
        r = score[c] > lam
        val[c] = np.where(r, OK[(c, "always_32b_direct")], OK[(c, "always_7b")])
        frac[c] = float(r.mean())
    return val, frac


def cost_of_fracs(frac, R):
    return float(np.mean([(1 - frac[c]) * 1.0 + frac[c] * R for c in CELLS]))


def lat_energy_of_fracs(frac):
    return (float(np.mean([(1 - frac[c]) * LAT7 + frac[c] * LAT32 for c in CELLS])),
            float(np.mean([(1 - frac[c]) * EN7 + frac[c] * EN32 for c in CELLS])))


# =================================================================================================
# 4.  ONE SHARED BOOTSTRAP FOR EVERYTHING
#     Fixed policies go through a matmul; threshold policies exploit the fact that
#         value(lambda) = ok7 + 1{score > lambda} * (ok32 - ok7),
#     so under a resample with counts c the extra term is a PREFIX SUM of c*adv in score order --
#     one cumsum per cell per replicate serves EVERY lambda at once.
# =================================================================================================
def shared_bootstrap(fixed, routers, seed=SEED, nboot=NBOOT, per_cell=False):
    """fixed:   {name: {cell: 0/1 vector}}
       routers: {cfg: (score_by_cell, lam_array)}
       returns (points, boots) keyed by policy name; boots are MACRO unless per_cell is a cell name."""
    fixed_names = list(fixed)
    rnames, rlams = [], {}
    for cfg, (sc, lams) in routers.items():
        for i in range(len(lams)):
            rnames.append(f"{cfg}@{i}")
        rlams[cfg] = lams
    names = fixed_names + rnames
    idx = {n: i for i, n in enumerate(names)}
    P = len(names)
    cells = CELLS if per_cell is False else [per_cell]
    wcell = 1.0 / len(CELLS) if per_cell is False else 1.0

    pts = np.zeros(P)
    for c in cells:
        for n in fixed_names:
            pts[idx[n]] += fixed[n][c].mean() * wcell
        for cfg, (sc, lams) in routers.items():
            o = np.argsort(-sc[c])
            advs = ADV[c][o]
            cum = np.concatenate([[0.0], np.cumsum(advs)])
            k = np.searchsorted(-sc[c][o], -lams, side="left")
            base = OK[(c, "always_7b")].mean()
            for i in range(len(lams)):
                pts[idx[f"{cfg}@{i}"]] += (base + cum[k[i]] / N[c]) * wcell

    boots = np.zeros((nboot, P))
    for c in cells:
        n = N[c]
        D = np.stack([fixed[nm][c] for nm in fixed_names], axis=1) if fixed_names else None
        orders, advsort, ks = {}, {}, {}
        for cfg, (sc, lams) in routers.items():
            o = np.argsort(-sc[c])
            orders[cfg] = o
            advsort[cfg] = ADV[c][o]
            ks[cfg] = np.searchsorted(-sc[c][o], -lams, side="left")
        ok7c = OK[(c, "always_7b")]
        rng = np.random.default_rng(seed + hash_cell(c))
        for s0 in range(0, nboot, BLOCK):
            b = min(BLOCK, nboot - s0)
            cnt = np.empty((b, n))
            for j in range(b):
                cnt[j] = np.bincount(rng.integers(0, n, n), minlength=n)
            if D is not None:
                boots[s0:s0 + b, :len(fixed_names)] += (cnt @ D) / n * wcell
            base = (cnt @ ok7c) / n
            for cfg in routers:
                U = cnt[:, orders[cfg]] * advsort[cfg]
                CU = np.cumsum(U, axis=1)
                CU = np.concatenate([np.zeros((b, 1)), CU], axis=1)
                kk = ks[cfg]
                cols = CU[:, kk] / n
                for i in range(len(kk)):
                    boots[s0:s0 + b, idx[f"{cfg}@{i}"]] += (base + cols[:, i]) * wcell
    return names, pts, boots


# =================================================================================================
# 5.  MAIN
# =================================================================================================
#: (name, variant, model, mode, description).  READ THE DATASET-IDENTITY LABELS.
#: A single GLOBAL threshold uses the LEVEL of the score, not just its within-cell ranking, so any
#: router fitted SEPARATELY PER CELL carries that cell's mean advantage in its own intercept and needs
#: the dataset label twice over (to pick which model to run, and to make its levels comparable).
#: R-B is therefore the only genuinely DEPLOYABLE prompt-only router in this table.
CONFIGS = [
    ("R-A_percell_gbm_dense", "within", "gbm", "dense",
     "PER-CELL gradient boosting on text statistics + format flag + image statistics -- a separate "
     "router fitted inside each cell.  USES DATASET IDENTITY: at deployment you must know which "
     "benchmark the question came from in order to pick which router to run, and its score LEVEL is "
     "that cell's own mean advantage.  Reported as an upper reference, not as a prompt-only result."),
    ("R-A_percell_ridge_full", "within", "ridge", "full",
     "PER-CELL ridge on dense features + hashed question n-grams.  USES DATASET IDENTITY, same reason."),
    ("R-B_pooled_gbm_dense", "pooled", "gbm", "dense",
     "POOLED gradient boosting over all 8 cells with NO dataset feature -- ONE model, one threshold, "
     "no dataset label needed at deployment.  THE DEPLOYABLE PROMPT-ONLY ROUTER.  Caveat kept in "
     "front: question text and image style still identify the benchmark IMPLICITLY, so part of what "
     "it learns is 'PMC-VQA-shaped questions do not need the 32B'."),
    ("R-B_pooled_ridge_full", "pooled", "ridge", "full",
     "POOLED ridge on dense + hashed n-grams, NO dataset feature; deployable, implicit dataset identity."),
    ("R-B_pooled_gbm_txtstats", "pooled", "gbm", "txtstats",
     "ABLATION: pooled gradient boosting on TEXT statistics + format flag only, image dropped."),
    ("R-B_pooled_gbm_imgonly", "pooled", "gbm", "imgonly",
     "ABLATION: pooled gradient boosting on IMAGE statistics only, all text dropped."),
    ("R-C_pooled_gbm_ds1hot", "pooled", "gbm", "dense_ds",
     "EXPLICIT dataset one-hot appended.  This IS dataset-identity routing and is reported for "
     "contrast only; it is not a deployable prompt-only result."),
]


def main():
    rep["title"] = ("ATTACK 2 -- THE PRE-GENERATION ROUTER.  Decide from the prompt and the image "
                    "ALONE which single model answers, and pay only that one.  Endpoint is the "
                    "COST-vs-MACRO-ACCURACY FRONTIER under the pre-registered non-inferiority "
                    "constraint CI_lower(policy - always_32B_direct) >= -0.0029.")
    rep["date"] = "2026-08-12"
    rep["objective"] = ("MINIMISE macro-weighted FLOP-eq SUBJECT TO macro accuracy being "
                        "statistically non-inferior to always-32B-direct.  A policy that is CHEAPER "
                        "AND TIES is a WIN.")
    rep["reproduce"] = ["python3 src/cascade_methods/pregen_router_features.py",
                        "python3 src/cascade_methods/pregen_router.py"]
    rep["no_gpu"] = True
    rep["no_fabricated_numbers"] = True
    rep["seed"] = SEED
    rep["n_seeds"] = NSEED
    rep["n_bootstrap"] = NBOOT
    rep["kfold"] = KFOLD
    rep["tie_tolerance"] = TIE_TOL
    rep["pool"] = f"Variant B (MMMU excluded): 5 benchmarks / 8 cells / n={NTOT}, CLEAN disjoint verifier"
    rep["literature_motivation"] = dict(
        source="results/cascade_methods/docs/current/LITERATURE_UPDATE_2026-08-11.md sections 0.2 and S2",
        quote="A lightweight pre-generation router exceeds the best cascade policy on four of five "
              "datasets, mainly because it avoids the cheap model's generation cost on queries sent "
              "directly to a larger model rather than because of a stronger routing signal.  These "
              "results suggest that cascade performance is limited primarily by structural cost, "
              "since cascades pay the cheap model before any escalation decision.",
        citation="Bouchard, arXiv:2605.06350 (2026-05-07), as recorded in LITERATURE_UPDATE_2026-08-11.md",
        shortlist_prediction="S2 predicted 'expect a LOSS -- query-only routing is the weakest point "
                             "on the routing plateau ... The value is a frontier point and a reviewer "
                             "answer, not a win.'  That prediction is tested here and reported "
                             "whichever way it lands.")
    rep["numerics_pins"] = dict(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="0",
        tf32="not applicable -- CPU only, no torch matmul anywhere in this script",
        python=sys.version.split()[0], numpy=np.__version__,
        sklearn=__import__("sklearn").__version__,
        ridge="primal normal equations, alpha=50, unpenalised intercept, numpy.linalg.solve -- "
              "deterministic, no iteration count and no solver tolerance",
        gbm="HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_depth=4, "
            "min_samples_leaf=60, l2_regularization=1.0, early_stopping=False), random_state = the "
            "seed",
        hashing="HashingVectorizer(n_features=2**12, ngram_range=(1,2), alternate_sign=False, "
                "norm='l2', analyzer='word') -- stateless, so no fit-on-test leakage is possible",
        scaler="scale-only, fitted on TRAINING rows of each fold; zero-variance columns dropped",
        bootstrap="paired item-level, ONE shared resample stream per cell (seeded per cell) reused by "
                  "every policy and every baseline; drawn as bincount(rng.integers(0,n,n)); threshold "
                  "policies evaluated by prefix cumsum so every lambda shares the identical replicate")

    print("null tests ...", flush=True)
    rep["null_tests"] = dict(N1=null_n1(), N2=null_n2(), N3=null_n3())
    for k, v in rep["null_tests"].items():
        print(f"  {k}: {v['verdict']}", flush=True)
        assert v["verdict"].startswith("PASS"), k

    # ---------------- structure ---------------------------------------------------------------
    a7m = float(np.mean([OK[(c, "always_7b")].mean() for c in CELLS]))
    a32m = float(np.mean([OK[(c, "always_32b_direct")].mean() for c in CELLS]))
    struct = {}
    for c in CELLS:
        o7, o32 = OK[(c, "always_7b")], OK[(c, "always_32b_direct")]
        struct[c] = dict(n=N[c], acc_7b=round(float(o7.mean()), 6), acc_32b=round(float(o32.mean()), 6),
                         advantage_32b_minus_7b=round(float((o32 - o7).mean()), 6),
                         p_7b_only_right=round(float(((o7 == 1) & (o32 == 0)).mean()), 6),
                         p_32b_only_right=round(float(((o7 == 0) & (o32 == 1)).mean()), 6),
                         oracle_item_acc=round(float(np.maximum(o7, o32).mean()), 6),
                         cascade_escalation=MAC["escalation"]["per_cell"][c],
                         cascade_flops_compute_lean=PCC[c]["method_compute_lean"]["flops"],
                         always32_flops=PCC[c]["always_32b_direct"]["flops"],
                         cascade_more_expensive_than_always32=bool(
                             PCC[c]["method_compute_lean"]["flops"] > PCC[c]["always_32b_direct"]["flops"]))
    orc_val = {c: np.maximum(OK[(c, "always_7b")], OK[(c, "always_32b_direct")]) for c in CELLS}
    orc_frac = {c: float(((OK[(c, "always_7b")] == 0) & (OK[(c, "always_32b_direct")] == 1)).mean())
                for c in CELLS}
    rep["structure"] = dict(
        per_cell=struct, macro_acc_7b=round(a7m, 6), macro_acc_32b=round(a32m, 6),
        macro_advantage=round(a32m - a7m, 6),
        ORACLE_pregen_router=dict(
            macro_acc=round(float(np.mean([orc_val[c].mean() for c in CELLS])), 6),
            macro_flops_as_charged=round(cost_of_fracs(orc_frac, R32_CHARGED), 4),
            ratio_vs_direct=round(cost_of_fracs(orc_frac, R32_CHARGED) / R32_CHARGED, 4),
            per_cell_32b_fraction={c: round(orc_frac[c], 4) for c in CELLS},
            note="ITEM-LEVEL ORACLE that knows per item which model is right.  UNREACHABLE; quoted "
                 "only to size the headroom.  This is the recoverability wall restated -- 16 prior "
                 "mechanisms failed to harvest it from the 7B's OWN confidence, and a pre-generation "
                 "router has strictly LESS information than those had."),
        no_skill_line=dict(
            formula="acc(f) = (1-f)*%.6f + f*%.6f ; flops(f) = (1-f)*1 + f*R32" % (a7m, a32m),
            f_for_point_delta_minus_0p0029=round(1.0 - TIE_TOL / (a32m - a7m), 4),
            ratio_at_that_f=round((1 - (1.0 - TIE_TOL / (a32m - a7m))
                                   + (1.0 - TIE_TOL / (a32m - a7m)) * R32_CHARGED) / R32_CHARGED, 4),
            note="a router with NO skill traces the straight line between always-7B and "
                 "always-32B-direct; every frontier below is compared against it at MATCHED cost."))

    # ---------------- train ---------------------------------------------------------------------
    print("training routers ...", flush=True)
    SCORES = {}
    for name, variant, model, mode, _ in CONFIGS:
        t = time.time()
        SCORES[name] = [crossfit_scores(variant, model, mode, SEED + s) for s in range(NSEED)]
        print(f"  {name}: {NSEED} seeds in {time.time()-t:.0f}s  (total {time.time()-T0:.0f}s)", flush=True)
    PERM = [crossfit_scores("pooled", "gbm", "dense", SEED + s, permute=True) for s in range(NSEED)]
    PERM_W = [crossfit_scores("within", "gbm", "dense", SEED + s, permute=True) for s in range(NSEED)]
    print(f"  permutation nulls done ({time.time()-T0:.0f}s)", flush=True)

    ENS = {k: ensemble(v) for k, v in SCORES.items()}
    ENS["DIAG_R-B_BETWEEN_cell_only"] = between_only(ENS["R-B_pooled_gbm_dense"])
    ENS["DIAG_R-B_WITHIN_cell_only"] = within_only(ENS["R-B_pooled_gbm_dense"])
    ENS["PERMUTATION_NULL_pooled"] = ensemble(PERM)
    ENS["PERMUTATION_NULL_percell"] = ensemble(PERM_W)

    # ---------------- discrimination -------------------------------------------------------------
    def auroc(score, y):
        score = np.asarray(score, float); y = np.asarray(y, int)
        P, Q = score[y == 1], score[y == 0]
        if len(P) == 0 or len(Q) == 0:
            return float("nan")
        a = np.concatenate([P, Q]); o = a.argsort(); rk = np.empty(len(a)); rk[o] = np.arange(1, len(a) + 1)
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        ss = np.zeros(len(cnt)); np.add.at(ss, inv, rk); rk = (ss / cnt)[inv]
        return float((rk[:len(P)].sum() - len(P) * (len(P) + 1) / 2) / (len(P) * len(Q)))

    disc = {}
    for name, sc in ENS.items():
        per = {}
        for c in CELLS:
            m = ADV[c] != 0
            per[c] = dict(auroc_advantage=round(auroc(sc[c][m], (ADV[c][m] > 0).astype(int)), 4),
                          mean_score=round(float(sc[c].mean()), 4),
                          true_mean_advantage=round(float(ADV[c].mean()), 6))
        s = np.concatenate([sc[c] for c in CELLS]); a = np.concatenate([ADV[c] for c in CELLS])
        m = a != 0
        disc[name] = dict(per_cell=per,
                          pooled_auroc_advantage=round(auroc(s[m], (a[m] > 0).astype(int)), 4),
                          pooled_corr_with_advantage=round(float(np.corrcoef(s, a)[0, 1]), 4),
                          between_cell_rank_corr=round(float(np.corrcoef(
                              [sc[c].mean() for c in CELLS], [ADV[c].mean() for c in CELLS])[0, 1]), 4))
    rep["router_discrimination"] = dict(
        metric="AUROC of the held-out advantage score at separating items where the 32B is right and "
               "the 7B wrong (positive) from items where the 7B is right and the 32B wrong (negative); "
               "items where both agree are excluded.  0.5 = no skill.  'between_cell_rank_corr' is the "
               "correlation across the 8 CELLS of mean predicted advantage against true mean "
               "advantage -- it is what a cost-saving global threshold actually needs.",
        seed_ensembled=disc)

    # ---------------- the frontier ---------------------------------------------------------------
    print("frontier + bootstrap ...", flush=True)
    fixed = {s: {c: OK[(c, s)] for c in CELLS} for s in
             ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b",
              "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]}
    fixed["ORACLE_pregen_item"] = orc_val
    routers = {k: (sc, macro_lam_grid(sc)) for k, sc in ENS.items()}
    names, pts, boots = shared_bootstrap(fixed, routers)
    IDX = {n: i for i, n in enumerate(names)}
    b32 = boots[:, IDX["always_32b_direct"]]
    p32 = pts[IDX["always_32b_direct"]]
    print(f"  bootstrap done ({time.time()-T0:.0f}s, {len(names)} policies)", flush=True)

    FIXCOST = {}
    for s in fixed:
        if s == "ORACLE_pregen_item":
            lp, en = lat_energy_of_fracs(orc_frac)
            FIXCOST[s] = dict(flops_charged=cost_of_fracs(orc_frac, R32_CHARGED),
                              flops_derived=cost_of_fracs(orc_frac, R32_DERIVED),
                              lat_par=lp, lat_seq=lp, energy=en,
                              macro_32b_fraction=float(np.mean([orc_frac[c] for c in CELLS])))
        else:
            FIXCOST[s] = dict(
                flops_charged=float(np.mean([PCC[c][s]["flops"] for c in CELLS])),
                flops_derived=None,
                lat_par=float(np.mean([PCC[c][s]["lat_par_ms"] for c in CELLS])),
                lat_seq=float(np.mean([PCC[c][s]["lat_seq_ms"] for c in CELLS])),
                energy=float(np.mean([PCC[c][s]["energy_j"] for c in CELLS])),
                macro_32b_fraction=None)

    def summarize(nm, cost):
        i = IDX[nm]
        d = boots[:, i] - b32
        lo, hi = ci(d)
        return dict(macro_acc=round(float(pts[i]), 6),
                    delta_vs_32b_direct=round(float(pts[i] - p32), 6),
                    ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                    meets_constraint=bool(lo >= -TIE_TOL),
                    flops_as_charged=round(cost["flops_charged"], 4),
                    ratio_as_charged=round(cost["flops_charged"] / R32_CHARGED, 4),
                    flops_R32_3p816=round(cost["flops_derived"], 4) if cost["flops_derived"] else None,
                    ratio_R32_3p816=round(cost["flops_derived"] / R32_DERIVED, 4)
                    if cost["flops_derived"] else None,
                    lat_par_ms=round(cost["lat_par"], 1), lat_seq_ms=round(cost["lat_seq"], 1),
                    energy_j=round(cost["energy"], 1),
                    macro_32b_fraction=round(cost["macro_32b_fraction"], 4)
                    if cost["macro_32b_fraction"] is not None else None)

    rep["baselines"] = {s: summarize(s, FIXCOST[s]) for s in fixed}

    rep["frontier"] = {}
    ROUTER_ROWS = {}
    N4DEV = []
    for cfg, (sc, lams) in routers.items():
        rows = []
        for i, lam in enumerate(lams):
            _, frac = route_eval(sc, lam)
            cost = dict(flops_charged=cost_of_fracs(frac, R32_CHARGED),
                        flops_derived=cost_of_fracs(frac, R32_DERIVED),
                        macro_32b_fraction=float(np.mean([frac[c] for c in CELLS])))
            cost["lat_par"], cost["energy"] = lat_energy_of_fracs(frac)
            cost["lat_seq"] = cost["lat_par"]
            r = summarize(f"{cfg}@{i}", cost)
            r["lam"] = round(float(lam), 8)
            r["policy_id"] = f"{cfg}@{i}"
            r["per_cell_32b_fraction"] = {c: round(frac[c], 4) for c in CELLS}
            # N4: the prefix-cumsum bootstrap point estimate must equal the literal re-evaluation
            lit = float(np.mean([np.where(sc[c] > lam, OK[(c, "always_32b_direct")],
                                          OK[(c, "always_7b")]).mean() for c in CELLS]))
            N4DEV.append(abs(lit - r["macro_acc"]))
            r["no_skill_line_acc_at_same_fraction"] = round(
                (1 - r["macro_32b_fraction"]) * a7m + r["macro_32b_fraction"] * a32m, 6)
            r["gain_over_no_skill_line"] = round(r["macro_acc"] - r["no_skill_line_acc_at_same_fraction"], 6)
            rows.append(r)
        ROUTER_ROWS[cfg] = rows
        ok_rows = [r for r in rows if r["meets_constraint"]]
        best = min(ok_rows, key=lambda r: r["flops_as_charged"]) if ok_rows else None
        DIAGDESC = {
            "DIAG_R-B_BETWEEN_cell_only":
                "DIAGNOSTIC decomposition of R-B_pooled_gbm_dense: every item's score replaced by its "
                "CELL MEAN, so a global threshold routes WHOLE DATASETS and nothing within them.  This "
                "isolates how much of the router's saving is 'recognise which benchmark this is'.",
            "DIAG_R-B_WITHIN_cell_only":
                "DIAGNOSTIC decomposition of R-B_pooled_gbm_dense: scores centred and scaled inside "
                "each cell, removing every between-cell level difference, so a global threshold can "
                "only exploit WITHIN-cell item ranking.",
            "PERMUTATION_NULL_pooled":
                "PERMUTATION NULL -- labels y = ok32 - ok7 shuffled WITHIN each cell before training "
                "the pooled GBM; folds, features, seeds and ensembling identical.",
            "PERMUTATION_NULL_percell":
                "PERMUTATION NULL for the per-cell GBM -- labels shuffled within each cell."}
        rep["frontier"][cfg] = dict(
            description=next((d for nm, _, _, _, d in CONFIGS if nm == cfg), DIAGDESC.get(cfg, "")),
            n_points=len(rows), n_points_meeting_constraint=len(ok_rows),
            cheapest_point_meeting_constraint=best, points=rows)
        if best:
            print(f"  {cfg}: cheapest feasible {best['ratio_as_charged']}x acc {best['macro_acc']:.4f} "
                  f"d {best['delta_vs_32b_direct']:+.4f} [{best['ci_lo']:+.4f},{best['ci_hi']:+.4f}]",
                  flush=True)
        else:
            print(f"  {cfg}: NO frontier point meets the constraint", flush=True)

    rep["null_tests"]["N4"] = dict(
        name="N4 -- the prefix-cumsum bootstrap estimator reproduces the literal policy evaluation",
        detail="every threshold policy's macro accuracy is computed twice: once via the cumulative-sum "
               "shortcut that makes the shared bootstrap affordable, and once by literally applying "
               "np.where(score > lambda, ok32, ok7) per cell.  They must agree exactly.",
        n_points_checked=len(N4DEV), max_abs_dev=float(max(N4DEV)),
        verdict="PASS -- max abs deviation %.2e" % max(N4DEV) if max(N4DEV) < 1e-12 else "FAIL")
    print(f"  N4: {rep['null_tests']['N4']['verdict']}", flush=True)
    assert rep["null_tests"]["N4"]["verdict"].startswith("PASS")

    # ---------------- permutation null verdict ---------------------------------------------------
    def interp_at(cfg, key, target):
        rows = sorted(ROUTER_ROWS[cfg], key=lambda r: r["macro_32b_fraction"])
        return float(np.interp(target, [r["macro_32b_fraction"] for r in rows], [r[key] for r in rows]))

    perm_rows = []
    for t in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        line = (1 - t) * a7m + t * a32m
        perm_rows.append(dict(
            macro_32b_fraction=t,
            R_B_pooled_gbm_dense_acc=round(interp_at("R-B_pooled_gbm_dense", "macro_acc", t), 6),
            R_A_percell_gbm_dense_acc=round(interp_at("R-A_percell_gbm_dense", "macro_acc", t), 6),
            DIAG_between_cell_only_acc=round(interp_at("DIAG_R-B_BETWEEN_cell_only", "macro_acc", t), 6),
            DIAG_within_cell_only_acc=round(interp_at("DIAG_R-B_WITHIN_cell_only", "macro_acc", t), 6),
            permutation_null_pooled_acc=round(interp_at("PERMUTATION_NULL_pooled", "macro_acc", t), 6),
            permutation_null_percell_acc=round(interp_at("PERMUTATION_NULL_percell", "macro_acc", t), 6),
            no_skill_line_acc=round(line, 6),
            real_minus_null_pooled=round(interp_at("R-B_pooled_gbm_dense", "macro_acc", t)
                                         - interp_at("PERMUTATION_NULL_pooled", "macro_acc", t), 6),
            null_minus_line=round(interp_at("PERMUTATION_NULL_pooled", "macro_acc", t) - line, 6)))
    mx = max(abs(r["null_minus_line"]) for r in perm_rows)
    rep["permutation_null"] = dict(
        design="labels y = ok32 - ok7 shuffled WITHIN each cell before training; identical folds, "
               "features, 10 seeds and ensembling.  A shuffled-label router must land ON the no-skill "
               "line at every matched 32B-fraction.",
        rows=perm_rows,
        max_abs_null_minus_no_skill_line=round(mx, 6),
        best_permutation_frontier_point=rep["frontier"]["PERMUTATION_NULL_pooled"]
        ["cheapest_point_meeting_constraint"],
        verdict=("PASS -- the shuffled-label router tracks the no-skill line to within %.4f macro "
                 "accuracy at every matched cost" % mx) if mx < 0.01 else
                ("SUSPECT -- the shuffled-label router deviates from the no-skill line by up to %.4f" % mx))
    print(f"  permutation null: {rep['permutation_null']['verdict']}", flush=True)

    # ---------------- NESTED selection of (config, lambda) ---------------------------------------
    # Selecting the best config AND the best lambda by looking at the eval macro is exactly how a fake
    # win gets manufactured here.  The honest headline chooses both INSIDE training folds.
    print("nested (config, lambda) selection ...", flush=True)
    DEPLOYABLE = [c[0] for c in CONFIGS]

    def nested_select(seed, kfold=KFOLD):
        fold = {c: np.random.default_rng(seed * 13 + hash_cell(c) + 3).permutation(N[c]) % kfold
                for c in CELLS}
        val = {c: np.empty(N[c]) for c in CELLS}
        cost_acc, lat_acc, en_acc, picked = [], [], [], []
        for f in range(kfold):
            best = None
            for cfg in DEPLOYABLE:
                sc = ENS[cfg]
                lams = macro_lam_grid(sc)
                for lam in lams:
                    acc, cst, wsum = 0.0, 0.0, 0.0
                    for c in CELLS:
                        tr = fold[c] != f
                        r = sc[c][tr] > lam
                        v = np.where(r, OK[(c, "always_32b_direct")][tr], OK[(c, "always_7b")][tr])
                        acc += v.mean() / len(CELLS)
                        cst += ((1 - r.mean()) + r.mean() * R32_CHARGED) / len(CELLS)
                        wsum += 1
                    base = np.mean([OK[(c, "always_32b_direct")][fold[c] != f].mean() for c in CELLS])
                    if acc - base >= -TIE_TOL / 2.0:          # train-side non-inferiority screen
                        if best is None or cst < best[2]:
                            best = (cfg, lam, cst, acc - base)
            if best is None:
                best = ("always_32b_direct", None, R32_CHARGED, 0.0)
            picked.append(dict(fold=f, config=best[0], lam=None if best[1] is None else float(best[1]),
                               train_cost=best[2], train_delta=best[3]))
            for c in CELLS:
                te = fold[c] == f
                if best[1] is None:
                    val[c][te] = OK[(c, "always_32b_direct")][te]
                    fr = 1.0
                else:
                    r = ENS[best[0]][c][te] > best[1]
                    val[c][te] = np.where(r, OK[(c, "always_32b_direct")][te], OK[(c, "always_7b")][te])
                    fr = float(r.mean())
                w = te.sum() / N[c] / len(CELLS)
                cost_acc.append((w, (1 - fr) + fr * R32_CHARGED, (1 - fr) + fr * R32_DERIVED))
                lat_acc.append((w, (1 - fr) * LAT7 + fr * LAT32))
                en_acc.append((w, (1 - fr) * EN7 + fr * EN32))
        return (val,
                sum(w * x for w, x, _ in cost_acc), sum(w * y for w, _, y in cost_acc),
                sum(w * x for w, x in lat_acc), sum(w * x for w, x in en_acc), picked)

    nested = {}
    nested_vals = {}
    for s in range(NSEED):
        v, fc, fh, lt, en, pk = nested_select(SEED + s)
        nested_vals[f"NESTED_seed{s}"] = v
        nested[f"NESTED_seed{s}"] = dict(flops_charged=fc, flops_derived=fh, lat_par=lt, lat_seq=lt,
                                         energy=en, macro_32b_fraction=(fc - 1.0) / (R32_CHARGED - 1.0),
                                         picks=pk)
    nn, npts_, nboots = shared_bootstrap(dict(nested_vals, always_32b_direct=fixed["always_32b_direct"]),
                                         {}, seed=SEED)
    nIDX = {n: i for i, n in enumerate(nn)}
    nb32 = nboots[:, nIDX["always_32b_direct"]]
    np32 = npts_[nIDX["always_32b_direct"]]
    nrows = {}
    for k in nested_vals:
        i = nIDX[k]
        d = nboots[:, i] - nb32
        lo, hi = ci(d)
        m = nested[k]
        nrows[k] = dict(macro_acc=round(float(npts_[i]), 6),
                        delta_vs_32b_direct=round(float(npts_[i] - np32), 6),
                        ci_lo=round(lo, 6), ci_hi=round(hi, 6), meets_constraint=bool(lo >= -TIE_TOL),
                        flops_as_charged=round(m["flops_charged"], 4),
                        ratio_as_charged=round(m["flops_charged"] / R32_CHARGED, 4),
                        flops_R32_3p816=round(m["flops_derived"], 4),
                        ratio_R32_3p816=round(m["flops_derived"] / R32_DERIVED, 4),
                        lat_par_ms=round(m["lat_par"], 1), energy_j=round(m["energy"], 1),
                        macro_32b_fraction=round(m["macro_32b_fraction"], 4),
                        picks=[dict(fold=p["fold"], config=p["config"], lam=p["lam"]) for p in m["picks"]])
    ratios = [nrows[k]["ratio_as_charged"] for k in nrows]
    deltas = [nrows[k]["delta_vs_32b_direct"] for k in nrows]
    nmeet = sum(1 for k in nrows if nrows[k]["meets_constraint"])
    rep["NESTED_HONEST_PRIMARY"] = dict(
        design="outer 5-fold CV.  Inside each outer TRAINING split the router configuration AND the "
               "threshold lambda are chosen to minimise train-side macro cost subject to a train-side "
               "non-inferiority screen (train delta >= -TIE_TOL/2); the choice is then applied to the "
               "held-out outer fold.  No eval-fold label ever reaches the choice applied to it.  "
               "CAVEAT, stated plainly: the ROUTER ITSELF was cross-fit on a different fold "
               "partition, so its training rows overlap this selector's training rows.  That is the "
               "standard construction and it is not eval leakage, but it is not a fully disjoint "
               "nested loop either.",
        per_seed=nrows,
        n_seeds=NSEED,
        seeds_meeting_constraint=f"{nmeet}/{NSEED}",
        ratio_as_charged_mean=round(float(np.mean(ratios)), 4),
        ratio_as_charged_sd=round(float(np.std(ratios)), 4),
        ratio_as_charged_min=round(float(np.min(ratios)), 4),
        ratio_as_charged_max=round(float(np.max(ratios)), 4),
        delta_mean=round(float(np.mean(deltas)), 6),
        delta_min=round(float(np.min(deltas)), 6), delta_max=round(float(np.max(deltas)), 6))
    print(f"  nested: {nmeet}/{NSEED} seeds meet the constraint, "
          f"ratio {np.mean(ratios):.4f} +- {np.std(ratios):.4f}", flush=True)

    # ---------------- per-seed robustness at matched cost ----------------------------------------
    tgt = np.linspace(0.05, 0.95, 19)
    seed_rows = {}
    for name in SCORES:
        acc_at = np.zeros((NSEED, len(tgt)))
        for s in range(NSEED):
            sc = SCORES[name][s]
            allv = np.concatenate([sc[c] for c in CELLS])
            w = np.concatenate([np.full(N[c], MACROW[c]) for c in CELLS])
            o = np.argsort(-allv); cw = np.cumsum(w[o]) / w.sum()
            for j, f in enumerate(tgt):
                k = min(int(np.searchsorted(cw, f)), len(o) - 1)
                lam = allv[o][k]
                val, _ = route_eval(sc, lam)
                acc_at[s, j] = float(np.mean([val[c].mean() for c in CELLS]))
        seed_rows[name] = dict(macro_32b_fraction_grid=[round(float(x), 3) for x in tgt],
                               acc_mean=[round(float(x), 6) for x in acc_at.mean(axis=0)],
                               acc_sd=[round(float(x), 6) for x in acc_at.std(axis=0)],
                               acc_min=[round(float(x), 6) for x in acc_at.min(axis=0)],
                               acc_max=[round(float(x), 6) for x in acc_at.max(axis=0)],
                               max_seed_sd=round(float(acc_at.std(axis=0).max()), 6))
    rep["seed_robustness"] = dict(
        n_seeds=NSEED,
        note="each seed is an independent 5-fold cross-fit (fold permutation + model random_state); "
             "accuracy is compared at MATCHED macro 32B-fraction so cost is held constant across "
             "seeds.  These are the PER-SEED routers, not the 10-seed ensemble.",
        per_config=seed_rows)

    # ---------------- per-cell guardrail ---------------------------------------------------------
    print("guardrail ...", flush=True)
    gpol = {"always_7b": fixed["always_7b"],
            "method_compute_lean": fixed["method_compute_lean"],
            "method_accuracy_max_veto": fixed["method_accuracy_max_veto"],
            "always_32b_direct": fixed["always_32b_direct"]}
    chosen_pts = {}
    for cfg in ["R-A_percell_gbm_dense", "R-B_pooled_gbm_dense", "R-C_pooled_gbm_ds1hot"]:
        b = rep["frontier"][cfg]["cheapest_point_meeting_constraint"]
        if b:
            sc = ENS[cfg]
            v, fr = route_eval(sc, b["lam"])
            gpol[f"{cfg}_cheapest_feasible"] = v
            chosen_pts[cfg] = b["policy_id"]
    gpol["NESTED_seed0"] = nested_vals["NESTED_seed0"]
    guard = {}
    for c in CELLS:
        nmz, p, bt = shared_bootstrap({k: v for k, v in gpol.items()}, {}, seed=SEED + 555,
                                      per_cell=c)
        bi = nmz.index("always_32b_direct")
        guard[c] = {}
        for j, nm in enumerate(nmz):
            if nm == "always_32b_direct":
                continue
            d = bt[:, j] - bt[:, bi]
            lo, hi = ci(d)
            guard[c][nm] = dict(delta=round(float(p[j] - p[bi]), 6), lo=round(lo, 6), hi=round(hi, 6),
                                flag_significant_loss=bool(hi < 0))
    rep["guardrail_per_cell"] = dict(
        baseline="always_32b_direct",
        policies_evaluated=chosen_pts,
        note="VQA_RAD cells are n=251 (closed) and n=200 (open); a flag there is frequently within "
             "seed noise and is reported, not treated as decisive.",
        per_cell=guard)

    # ---------------- HYBRID: per-cell arm selection including router arms ------------------------
    print("hybrid ...", flush=True)
    ARMS = {}
    for c in CELLS:
        ARMS[c] = {}
        for s in ["always_7b", "always_32b_direct", "method_compute_lean", "method_accuracy_max_veto"]:
            ARMS[c][s] = (OK[(c, s)], PCC[c][s]["flops"], PCC[c][s]["lat_par_ms"], PCC[c][s]["energy_j"])
    sc_h = ENS["R-B_pooled_gbm_dense"]
    hgrid = macro_lam_grid(sc_h, npts=21)
    for i, lam in enumerate(hgrid):
        val, frac = route_eval(sc_h, lam)
        for c in CELLS:
            ARMS[c][f"router@{i}"] = (val[c], (1 - frac[c]) + frac[c] * R32_CHARGED,
                                      (1 - frac[c]) * LAT7 + frac[c] * LAT32,
                                      (1 - frac[c]) * EN7 + frac[c] * EN32)
    ARMNAMES = list(ARMS[CELLS[0]])

    def hybrid_crossfit(eps, seed):
        fold = {c: folds_for(c, seed) for c in CELLS}
        out, agg, picks = {}, [], {}
        for c in CELLS:
            v = np.empty(N[c]); pk = []
            for f in range(KFOLD):
                tr, te = fold[c] != f, fold[c] == f
                if te.sum() == 0:
                    continue
                accs = {a: ARMS[c][a][0][tr].mean() for a in ARMNAMES}
                best = max(accs.values())
                elig = [a for a in ARMNAMES if accs[a] >= best - eps]
                a_star = min(elig, key=lambda a: (ARMS[c][a][1], a))
                v[te] = ARMS[c][a_star][0][te]
                w = te.sum() / N[c] / len(CELLS)
                agg.append((w, ARMS[c][a_star][1], ARMS[c][a_star][2], ARMS[c][a_star][3]))
                pk.append(a_star)
            out[c] = v; picks[c] = pk
        return out, (sum(w * x for w, x, _, _ in agg), sum(w * y for w, _, y, _ in agg),
                     sum(w * z for w, _, _, z in agg)), picks

    EPS_GRID = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
    hyb_vals, hyb_meta = {}, {}
    for e in EPS_GRID:
        v, (fc, lt, en), pk = hybrid_crossfit(e, SEED)
        nm = f"HYBRID_crossfit_eps{e}"
        hyb_vals[nm] = v
        hyb_meta[nm] = dict(eps=e, flops=fc, lat=lt, energy=en, picks=pk)

    def hybrid_nested(seed, inner=4):
        fold = {c: np.random.default_rng(seed * 31 + hash_cell(c) + 1).permutation(N[c]) % KFOLD
                for c in CELLS}
        out = {c: np.empty(N[c]) for c in CELLS}
        agg, chosen = [], []
        for f in range(KFOLD):
            scored = []
            for e in EPS_GRID:
                accs, csts = [], []
                for c in CELLS:
                    idx = np.flatnonzero(fold[c] != f)
                    ifold = (np.arange(len(idx)) + hash_cell(c)) % inner
                    v = np.empty(len(idx)); cc = []
                    for g in range(inner):
                        itr, ite = ifold != g, ifold == g
                        if ite.sum() == 0:
                            continue
                        a_ = {a: ARMS[c][a][0][idx[itr]].mean() for a in ARMNAMES}
                        bb = max(a_.values())
                        el = [a for a in ARMNAMES if a_[a] >= bb - e]
                        a_star = min(el, key=lambda a: (ARMS[c][a][1], a))
                        v[ite] = ARMS[c][a_star][0][idx[ite]]
                        cc.append((ite.sum(), ARMS[c][a_star][1]))
                    accs.append(v.mean()); csts.append(sum(w * x for w, x in cc) / sum(w for w, _ in cc))
                scored.append((e, float(np.mean(accs)), float(np.mean(csts))))
            top = max(s[1] for s in scored)
            feas = [s for s in scored if s[1] >= top - TIE_TOL]
            e_star = min(feas, key=lambda s: (s[2], s[0]))[0]
            chosen.append(e_star)
            for c in CELLS:
                tr, te = fold[c] != f, fold[c] == f
                a_ = {a: ARMS[c][a][0][tr].mean() for a in ARMNAMES}
                bb = max(a_.values())
                el = [a for a in ARMNAMES if a_[a] >= bb - e_star]
                a_star = min(el, key=lambda a: (ARMS[c][a][1], a))
                out[c][te] = ARMS[c][a_star][0][te]
                w = te.sum() / N[c] / len(CELLS)
                agg.append((w, ARMS[c][a_star][1], ARMS[c][a_star][2], ARMS[c][a_star][3]))
        return out, (sum(w * x for w, x, _, _ in agg), sum(w * y for w, _, y, _ in agg),
                     sum(w * z for w, _, _, z in agg)), chosen

    hv, (hfc, hlt, hen), hch = hybrid_nested(SEED)
    hyb_vals["HYBRID_nested_cv"] = hv
    hyb_meta["HYBRID_nested_cv"] = dict(eps="chosen inside the training folds", flops=hfc, lat=hlt,
                                        energy=hen, eps_per_fold=hch)

    hn, hp, hb = shared_bootstrap(dict(hyb_vals, always_32b_direct=fixed["always_32b_direct"]), {},
                                  seed=SEED)
    hI = {n: i for i, n in enumerate(hn)}
    hbase = hb[:, hI["always_32b_direct"]]; hpb = hp[hI["always_32b_direct"]]
    hrows = {}
    for k in hyb_vals:
        i = hI[k]
        d = hb[:, i] - hbase
        lo, hi = ci(d)
        m = hyb_meta[k]
        hrows[k] = dict(macro_acc=round(float(hp[i]), 6),
                        delta_vs_32b_direct=round(float(hp[i] - hpb), 6),
                        ci_lo=round(lo, 6), ci_hi=round(hi, 6), meets_constraint=bool(lo >= -TIE_TOL),
                        flops_as_charged=round(m["flops"], 4),
                        ratio_as_charged=round(m["flops"] / R32_CHARGED, 4),
                        lat_par_ms=round(m["lat"], 1), energy_j=round(m["energy"], 1),
                        eps=m.get("eps"), eps_per_fold=m.get("eps_per_fold"),
                        arms_picked={c: sorted(set(v)) for c, v in m.get("picks", {}).items()})
    rep["hybrid_percell_arm_selection"] = dict(
        design="per-cell 5-fold CROSS-FIT arm selection over the menu {always_7b, always_32b_direct, "
               "cascade compute-lean, cascade accuracy-max-veto, pre-generation router at 21 operating "
               "points}: on the other folds take the most accurate arm, deploy the CHEAPEST arm within "
               "eps of it, answer the held-out fold with that arm.  THIS USES DATASET IDENTITY -- it "
               "picks a different arm per cell and therefore needs to know which benchmark a question "
               "came from.  LABELLED.  This is the literature's 'pre-route the obviously-hard cells "
               "straight to the 32B and cascade only where escalation is genuinely uncertain'.",
        arm_menu=ARMNAMES, rows=hrows,
        comparator="results/cascade_methods/artifacts/cost_floor_2026-08-10.json ran the same "
                   "construction WITHOUT any router arm and got 1.1648x at +0.0011 [-0.0014,+0.0035] "
                   "(eps=0, 12-seed mean, tie on 10/12 seeds).")

    # ---------------- footprint --------------------------------------------------------------------
    router_params = dict(
        gbm_dense="HistGradientBoostingRegressor, 150 trees x max_depth 4 over 84 float features; "
                  "the fitted model pickles to a few hundred kB and its forward pass is ~150 depth-4 "
                  "comparisons per item -- negligible beside a 7B forward.  NOT separately weighed "
                  "on disk: not measured.",
        ridge_full="4,273 float64 coefficients = 34.2 kB.")
    rep["footprint"] = dict(
        source="measured 2026-08-11 (safetensors index total_size and adapter parameter counts), "
               "quoted from the round brief; the 32B figure is the project's standing constant",
        components=dict(lingshu_7b=dict(params=8.29e9, gib=15.45),
                        lingshu_32b=dict(params=32.8e9, gib=31.5),
                        verifier_lora=dict(params=47589376, mib=181.6),
                        frozen_selector=dict(params=7344136, mib=28.1)),
        router_head=router_params,
        systems=dict(
            always_32b_direct=dict(params=32.8e9, gib=31.5, note="ONE model resident. The footprint bar."),
            always_7b=dict(params=8.29e9, gib=15.45),
            shipped_cascade_accuracy_max=dict(params=8.29e9 + 32.8e9 + 47589376 + 7344136,
                                              gib=round(15.45 + 31.5 + 0.1773 + 0.0274, 4)),
            pregen_router=dict(params=8.29e9 + 32.8e9, gib=round(15.45 + 31.5, 4))),
        HONEST_STATEMENT=(
            "A pre-generation router that can send a question straight to the 32B still needs the 32B "
            "RESIDENT, and it needs the 7B resident as well.  IT SAVES COMPUTE, LATENCY AND ENERGY; "
            "IT DOES NOT SAVE MEMORY -- it costs ~15.45 GiB MORE than always-32B-direct, a 49% "
            "increase in resident weights.  What it does drop relative to the shipped cascade is the "
            "verifier LoRA and the frozen selector (209.7 MiB) and the best-of-8 KV-cache burst.  "
            "'A smaller model' is achieved only on the fraction of traffic the router sends to the 7B; "
            "the deployment still holds both models.  The only way this becomes a memory win is to "
            "drop the 32B entirely, and no policy here does that without breaking the accuracy "
            "constraint -- always-7B is -0.0596 macro, ~20x the tie tolerance."))

    # ---------------- headline -----------------------------------------------------------------------
    head = []
    for s, lab in [("always_32b_direct", "always-32B-direct (THE BAR, 1.000x by definition)"),
                   ("always_7b", "always-7B"),
                   ("method_accuracy_max_veto", "SHIPPED cascade, accuracy-max (clean disjoint verifier)"),
                   ("method_compute_lean", "SHIPPED cascade, compute-lean"),
                   ("ORACLE_pregen_item", "ITEM-LEVEL ORACLE pre-gen router (UNREACHABLE ceiling)")]:
        r = dict(rep["baselines"][s]); r["label"] = lab; r["kind"] = "reference"
        head.append(r)
    for cfg in [c[0] for c in CONFIGS]:
        b = rep["frontier"][cfg]["cheapest_point_meeting_constraint"]
        if b:
            r = dict(b); r.pop("per_cell_32b_fraction", None)
            r["label"] = f"PRE-GEN ROUTER {cfg} -- cheapest feasible point (EVAL-SELECTED lambda)"
            r["kind"] = "pregen_eval_selected"
            head.append(r)
    r = dict(rep["NESTED_HONEST_PRIMARY"]["per_seed"]["NESTED_seed0"])
    r.pop("picks", None)
    r["label"] = "PRE-GEN ROUTER, NESTED honest selection of (config, lambda), seed 0"
    r["kind"] = "pregen_nested"
    head.append(r)
    for k, v in hrows.items():
        r = dict(v); r.pop("arms_picked", None)
        r["label"] = k; r["kind"] = "hybrid_dataset_identity"
        head.append(r)
    rep["HEADLINE_TABLE"] = dict(
        cost_label="EVERY cost column is MACRO-weighted (8 cells, 1/8 each) and is paired only with "
                   "the MACRO accuracy in the same row.  Sample-weighted cost answers a different "
                   "question and is not printed here.",
        rows=head)
    rep["external_comparators"] = dict(
        cost_floor_crossfit_eps0=dict(
            macro_acc=0.6578, delta=0.0011, ci=[-0.0014, 0.0035], ratio_as_charged=1.1648,
            robustness="tie on 10/12 fold seeds",
            source="results/cascade_methods/artifacts/cost_floor_2026-08-10.json:"
                   "what_DID_clear_the_bar.zero_eps_crossfit"),
        cost_floor_nested_cv=dict(
            ratio_as_charged=0.9931, delta=-0.0009, ci=[-0.0034, 0.0015], tie_preserved=False,
            source="results/cascade_methods/artifacts/cost_floor_2026-08-10.json:"
                   "PRIMARY_ENDPOINT_HONEST_NESTED_CV",
            note="MISSES the pre-registered constraint by 0.0005 on the CI lower bound"),
        cost_floor_verdict="SUCCESS THRESHOLD NOT MET on the as-charged convention; across 12 fold "
                           "seeds the pre-registered tie survives on 10/12 only up to 1.118x.",
        mcq_only_arm=dict(ratio_as_charged=0.9773, delta=0.0012, ci=[0.0009, 0.0015],
                          source="results/cascade_methods/artifacts/armcombine_mcqonly_2026-08-11.json",
                          note="byte-identical to the baseline on 7 of 8 cells by construction and "
                               "gated behind an OWED answer-letter-bias audit of test_2.csv"))
    rep["runtime_seconds"] = round(time.time() - T0, 1)
    json.dump(rep, open(OUT, "w"), indent=1, default=float)
    print(f"\nwrote {OUT}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
