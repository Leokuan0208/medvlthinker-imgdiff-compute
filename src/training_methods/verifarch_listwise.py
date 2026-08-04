#!/usr/bin/env python3
"""verifarch_listwise.py -- TIER-1 T1.B: does the LEARNING OBJECTIVE explain the selector gap?

QUESTION.  The deployed verifier is trained POINTWISE ("is this answer correct?", BCE on a Yes/No
token logit) and then argmaxed over the 8 candidates.  The learning-to-rank literature says this is a
mismatch with the deployed objective, which is "pick the best of N".  Does swapping the objective for a
LISTWISE (softmax-over-the-8) or PAIRWISE (Bradley-Terry within question) one convert more of the
oracle gap?

DESIGN -- the objective is the ONLY variable.  Every arm is fit on the SAME feature matrix, the SAME
image-disjoint folds, the SAME architecture, the SAME optimiser, the SAME epoch-selection criterion
(inner-validation selection efficiency, identical for all arms so no arm is handicapped by being scored
on a criterion it was not trained for), and the SAME 5-init ensembling.  Only `loss` differs.

  pointwise_bce   BCE over all candidates                          (the incumbent's objective)
  listnet         listwise softmax CE, target = uniform over the correct candidates (ListNet / PL top-1)
  anypos          listwise -log SUM_{i correct} softmax(s)_i       (the exact sel_eff surrogate)
  ranknet_bt      pairwise -log sigmoid(s_pos - s_neg) over all within-question pos x neg pairs
                                                                   (Bradley-Terry, the N25 formulation)

Each is run over two feature sets:
  V     = the incumbent verifier score alone (1 feature)  -> monotone-invariance sanity arm
  FULL  = incumbent score + set-relative + pool-structure + text features (identical across arms)
  NOV   = FULL minus every feature derived from the incumbent score (deployable-in-principle arm)

WHY THIS IS A DIAGNOSTIC, NOT A DEPLOYABLE NUMBER.  The incumbent LoRA adapter has never been run over
the 16,621 image-disjoint TRAIN items, so any arm using its score as a feature can only be fit by
cross-fitting *inside* the 2345 eval items.  Cross-fitting makes the comparison BETWEEN arms internally
valid (identical folds, identical leakage exposure) but the absolute sel_eff of a V/FULL arm is an
upper bound, not a shippable result.  The NOV arms carry no such caveat.

NULL TEST (asserted in code, run aborts otherwise): the harness pointed at the incumbent's own scores
must reproduce verifier_disjoint_retrain_2026-07-30.json to 1e-9 -- sel_eff 0.775204, sel_acc 0.485288,
oracle@8 0.626013, greedy 0.449467, candidate AUROC 0.885592, per-set sel_eff .8501/.7619/.7226.

  python3 src/training_methods/verifarch_listwise.py
  -> results/cascade_methods/artifacts/verifarch_listwise_2026-08-04.json
"""
import argparse, json, os, re, string, sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
N = 8

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--inits", type=int, default=5)
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--nboot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="results/cascade_methods/artifacts/verifarch_listwise_2026-08-04.json")
A = ap.parse_args()
torch.set_num_threads(8)

# ---------------------------------------------------------------- 1. load ground truth
dumps = []
for ds in DS:
    dumps += json.load(open(J(f"ckpts/train/lora_verifier_disjoint/transfer_dump_{ds}_lingshu7b.json")))
pool = {}
for ds in DS:
    for line in open(J(f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl")):
        if line.strip():
            r = json.loads(line)
            pool[(ds, r["idx"])] = r
imghash = json.load(open(J("data/verifarch/eval_imghash.json")))

rows = []
for r in dumps:
    p = pool[(r["ds"], r["idx"])]
    assert p["preds"] == r["preds"], "candidate lists disagree between dump and sc8 pool"
    # NOTE: the sc8 pool's `oks` is EXACT-MATCH; the dump's `sl` is the 32B judge_ok used by the
    # incumbent artifact (they differ on 444/2345 items). `sl` is authoritative here -- it is what
    # reproduces oracle@8 0.626013.  Only label-free pool fields are read below.
    r = dict(r)
    r["question"] = p["question"]
    r["n_distinct"] = p["n_distinct"]
    r["self_consistency"] = p["self_consistency"]
    r["gen_tokens"] = p["gen_tokens"]
    r["img"] = imghash[r["ds"]][str(r["idx"])]
    rows.append(r)
n_q = len(rows)
assert n_q == 2345, n_q

sl = np.array([r["sl"] for r in rows], dtype=np.float64)           # (Q,8) judge labels
V = np.array([r["scores"] for r in rows], dtype=np.float64)        # (Q,8) incumbent verifier score
ds_of = np.array([r["ds"] for r in rows])
greedy_ok = np.array([r["greedy_ok"] for r in rows], dtype=np.float64)
has = sl.max(1) > 0                                                # oracle@8 hit
groups = np.array([r["img"] for r in rows])
n_distinct = np.array([r["n_distinct"] for r in rows])
contested = n_distinct >= 2                        # label-FREE contested stratum (deployment-visible)
mixed = (sl.sum(1) > 0) & (sl.sum(1) < N)          # label-MIXED stratum: the only questions where any
                                                   # ranker can gain or lose (all-correct lists are 100%
                                                   # by construction; all-wrong lists are 0% regardless)

# ---------------------------------------------------------------- 2. metrics + NULL TEST
def sel_from_scores(S):
    """argmax with first-index tie-break -- exactly the incumbent's convention."""
    return S.argmax(1)


def metrics(S, pick=None):
    p = sel_from_scores(S) if pick is None else pick
    ok = sl[np.arange(n_q), p]
    out = {"sel_acc": float(ok.mean()), "sel_eff": float(ok[has].mean()),
           "sel_eff_contested_ndistinct2": float(ok[has & contested].mean()),
           "sel_eff_mixed_label": float(ok[mixed].mean()),
           "auroc": float(roc_auc_score(sl.ravel(), S.ravel()))}
    for d in DS:
        m = ds_of == d
        out[f"sel_acc_{d}"] = float(ok[m].mean())
        out[f"sel_eff_{d}"] = float(ok[m & has].mean())
    return out, ok


NULL, _ = metrics(V)
REF = {"sel_eff": 0.7752043596730245, "sel_acc": 0.48528784648187634, "auroc": 0.8855921901711237,
       "sel_eff_slake_open": 0.8500881834215167, "sel_eff_vqa_rad_open": 0.7619047619047619,
       "sel_eff_pathvqa_open": 0.7225806451612903}
PUBLISHED_4DP = {"sel_eff": 0.7752, "sel_acc": 0.4853, "auroc": 0.8856,
                 "sel_eff_slake_open": 0.8501, "sel_eff_vqa_rad_open": 0.7619,
                 "sel_eff_pathvqa_open": 0.7226}
null_report = {"oracle_at_8": float(has.mean()), "greedy": float(greedy_ok.mean()),
               "self_consistency": None, "incumbent": NULL,
               "reference_verifier_disjoint_retrain_2026-07-30": REF,
               "published_rounded_4dp_from_the_brief": PUBLISHED_4DP, "max_abs_dev": None}
dev = max(abs(NULL[k] - v) for k, v in REF.items())
null_report["max_abs_dev"] = float(dev)
assert dev < 1e-12, f"NULL TEST FAILED, max deviation {dev}"
for k, v in PUBLISHED_4DP.items():
    assert round(NULL[k], 4) == v, f"NULL TEST FAILED at 4dp: {k} {NULL[k]} vs published {v}"
assert abs(has.mean() - 0.6260127931769722) < 1e-12
assert abs(greedy_ok.mean() - 0.4494669509594883) < 1e-12
print(f"[null] harness reproduces the incumbent to {dev:.2e}; oracle@8 {has.mean():.6f} "
      f"greedy {greedy_ok.mean():.6f} sel_eff {NULL['sel_eff']:.6f}", flush=True)

# controls -------------------------------------------------------------------
rng0 = np.random.default_rng(A.seed)
rand_pick = rng0.integers(0, N, size=n_q)
CTRL = {}
CTRL["random_pick"], _ = metrics(V, pick=rand_pick)
# self-consistency: plurality vote over exact normalised strings, first-index tie-break
def norm(s):
    s = str(s).lower().strip().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


sc_pick = np.zeros(n_q, dtype=int)
for i, r in enumerate(rows):
    c = Counter(norm(x) for x in r["preds"])
    best = max(c.values())
    sc_pick[i] = next(j for j, x in enumerate(r["preds"]) if c[norm(x)] == best)
CTRL["self_consistency"], sc_ok = metrics(V, pick=sc_pick)
null_report["self_consistency"] = CTRL["self_consistency"]["sel_acc"]
CTRL["greedy"] = {"sel_acc": float(greedy_ok.mean())}
CTRL["oracle_at_8"] = {"sel_acc": float(has.mean()), "sel_eff": 1.0}

# ---------------------------------------------------------------- 3. features
def tokset(s):
    return set(norm(s).split())


def tokf1(a, b):
    A_, B_ = tokset(a), tokset(b)
    if not A_ or not B_:
        return 0.0
    i = len(A_ & B_)
    return 2 * i / (len(A_) + len(B_))


FEAT_NAMES, X = [], np.zeros((n_q, N, 0))
cols, names = [], []


def add(name, arr):
    names.append(name)
    cols.append(np.asarray(arr, dtype=np.float64).reshape(n_q, N))


v = V
v_max_other = np.zeros_like(v)
for i in range(N):
    o = np.delete(v, i, axis=1)
    v_max_other[:, i] = o.max(1)
v_mean = v.mean(1, keepdims=True)
v_std = v.std(1, keepdims=True) + 1e-9
v_rank = np.argsort(np.argsort(-v, axis=1), axis=1) / (N - 1.0)

add("v", v)                                              # incumbent score  [V-derived]
add("v_z", (v - v_mean) / v_std)                         #                  [V-derived]
add("v_minus_maxother", v - v_max_other)                 #                  [V-derived]
add("v_minus_poolmean", v - v_mean)                      #                  [V-derived]
add("v_rank", v_rank)                                    #                  [V-derived]
add("v_pool_std", np.repeat(v_std, N, axis=1))           #                  [V-derived]
add("v_pool_mean", np.repeat(v_mean, N, axis=1))         #                  [V-derived]
V_DERIVED = set(names)

dupfrac = np.zeros((n_q, N)); ismodal = np.zeros((n_q, N))
nchars = np.zeros((n_q, N)); nwords = np.zeros((n_q, N))
simmean = np.zeros((n_q, N)); simmax = np.zeros((n_q, N))
qoverlap = np.zeros((n_q, N)); isyesno = np.zeros((n_q, N))
for i, r in enumerate(rows):
    pr = r["preds"]; nm = [norm(x) for x in pr]
    c = Counter(nm); best = max(c.values())
    qt = tokset(r["question"])
    for j in range(N):
        dupfrac[i, j] = c[nm[j]] / N
        ismodal[i, j] = 1.0 * (c[nm[j]] == best)
        nchars[i, j] = len(pr[j])
        nwords[i, j] = len(nm[j].split())
        s = [tokf1(pr[j], pr[k]) for k in range(N) if k != j]
        simmean[i, j] = float(np.mean(s)); simmax[i, j] = float(np.max(s))
        t = tokset(pr[j])
        qoverlap[i, j] = (len(t & qt) / len(t)) if t else 0.0
        isyesno[i, j] = 1.0 * (nm[j] in ("yes", "no"))

add("dupfrac", dupfrac)
add("is_modal", ismodal)
add("n_chars", nchars)
add("n_words", nwords)
add("n_words_dev", nwords - nwords.mean(1, keepdims=True))
add("sim_mean_to_others", simmean)
add("sim_max_to_others", simmax)
add("q_overlap", qoverlap)
add("is_yesno", isyesno)
add("pool_n_distinct", np.repeat(n_distinct.reshape(-1, 1), N, axis=1))
add("pool_self_consistency", np.repeat(np.array([r["self_consistency"] for r in rows]).reshape(-1, 1), N, axis=1))
add("pool_gen_tokens", np.repeat(np.array([r["gen_tokens"] for r in rows]).reshape(-1, 1), N, axis=1))
add("q_n_words", np.repeat(np.array([len(norm(r["question"]).split()) for r in rows]).reshape(-1, 1), N, axis=1))
for d in DS:
    add(f"is_{d}", np.repeat((ds_of == d).reshape(-1, 1).astype(float), N, axis=1))

X = np.stack(cols, axis=2)   # (Q, 8, F)
FEAT_NAMES = names
FSETS = {"V": [FEAT_NAMES.index("v")],
         "FULL": list(range(len(FEAT_NAMES))),
         "NOV": [i for i, nme in enumerate(FEAT_NAMES) if nme not in V_DERIVED]}
print(f"[feat] {len(FEAT_NAMES)} features; NOV={len(FSETS['NOV'])} (no incumbent-score-derived feature)",
      flush=True)

# ---------------------------------------------------------------- 4. objectives
def loss_pointwise_bce(s, y):
    return nn.functional.binary_cross_entropy_with_logits(s, y, reduction="mean")


def loss_listnet(s, y):
    """listwise softmax CE, target = uniform over the correct candidates (ListNet / Plackett-Luce top-1)."""
    t = y / y.sum(1, keepdim=True)
    return -(t * torch.log_softmax(s, dim=1)).sum(1).mean()


def loss_anypos(s, y):
    """the exact selection-efficiency surrogate: -log SUM_{i correct} softmax(s)_i."""
    lp = torch.log_softmax(s, dim=1)
    return -torch.logsumexp(lp.masked_fill(y < 0.5, -1e30), dim=1).mean()


def loss_ranknet_bt(s, y):
    """pairwise Bradley-Terry over every within-question positive x negative pair."""
    d = s.unsqueeze(2) - s.unsqueeze(1)                       # (B,8,8) s_i - s_j
    w = (y.unsqueeze(2) * (1 - y).unsqueeze(1))               # i correct, j wrong
    return -(nn.functional.logsigmoid(d) * w).sum() / w.sum().clamp(min=1.0)


# `nondeg`: a list whose 8 labels are all-correct or all-wrong carries NO within-question ordering
# information, so every listwise / pairwise objective is degenerate on it (its gradient is zero or,
# for ListNet, actively pushes the 8 scores together).  Dropping those lists is the standard LTR
# formulation -- and `pointwise_bce_nondeg` is the control that separates "the objective changed"
# from "the training set shrank", because it is pointwise BCE trained on exactly the same subset.
OBJ = {"pointwise_bce": (loss_pointwise_bce, False),
       "pointwise_bce_nondeg": (loss_pointwise_bce, True),
       "listnet": (loss_listnet, True),
       "anypos": (loss_anypos, True),
       "ranknet_bt": (loss_ranknet_bt, True)}
OBJ_LIST = ["pointwise_bce", "pointwise_bce_nondeg", "listnet", "anypos", "ranknet_bt"]
nondeg_mask = mixed
print(f"[lists] {int(nondeg_mask.sum())}/{n_q} questions are non-degenerate "
      f"({nondeg_mask.mean():.3f}); the remaining {int((~nondeg_mask).sum())} are all-correct "
      f"({int((sl.sum(1) == N).sum())}) or all-wrong ({int((sl.sum(1) == 0).sum())}) and are invisible "
      f"to every listwise/pairwise objective", flush=True)


class MLP(nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):                                     # (B,8,F) -> (B,8)
        return self.net(x).squeeze(-1)


class Linear(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.net = nn.Linear(f, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------- 5. cross-fitted training
gkf = GroupKFold(n_splits=A.folds)
FOLDS = list(gkf.split(np.arange(n_q), groups=groups))
for tr, te in FOLDS:                                  # image-disjointness of the folds, asserted
    assert not (set(groups[tr]) & set(groups[te]))
print(f"[folds] {A.folds} image-disjoint folds, sizes {[len(t) for _, t in FOLDS]}", flush=True)


def fit_predict_nn(fs, obj, arch="mlp", seed_base=0):
    """Cross-fitted out-of-fold scores.  Epoch is selected on an inner image-disjoint validation split
    by SELECTION EFFICIENCY -- the same criterion for every arm."""
    idx = FSETS[fs]
    lossfn, nondeg = OBJ[obj]
    oof = np.zeros((n_q, N))
    chosen = []
    for fi, (tr, te) in enumerate(FOLDS):
        inner = GroupKFold(n_splits=4).split(np.arange(len(tr)), groups=groups[tr])
        itr, iva = next(iter(inner))
        tr_fit = tr[itr]
        if nondeg:
            tr_fit = tr_fit[nondeg_mask[tr_fit]]
        mu = X[tr][:, :, idx].reshape(-1, len(idx)).mean(0)
        sd = X[tr][:, :, idx].reshape(-1, len(idx)).std(0) + 1e-9
        def prep(sel):
            return torch.tensor((X[sel][:, :, idx] - mu) / sd, dtype=torch.float32, device=A.device)
        Xtr, Xva, Xte = prep(tr_fit), prep(tr[iva]), prep(te)
        ytr = torch.tensor(sl[tr_fit], dtype=torch.float32, device=A.device)
        va_sl, va_has = sl[tr[iva]], has[tr[iva]]
        acc = np.zeros((len(te), N))
        for k in range(A.inits):
            torch.manual_seed(seed_base + 1000 * fi + k)
            net = (MLP(len(idx)) if arch == "mlp" else Linear(len(idx))).to(A.device)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            best, best_ep, best_scores = -1.0, 0, None
            for ep in range(1, A.epochs + 1):
                net.train(); opt.zero_grad()
                lossfn(net(Xtr), ytr).backward(); opt.step()
                if ep % 10 == 0:
                    net.eval()
                    with torch.no_grad():
                        sv = net(Xva).cpu().numpy()
                    e = va_sl[np.arange(len(sv)), sv.argmax(1)][va_has].mean()
                    if e > best:
                        with torch.no_grad():
                            best, best_ep, best_scores = e, ep, net(Xte).cpu().numpy()
            chosen.append(best_ep)
            # ensemble in RAW score space, each init globally z-scored on its own test fold (an affine,
            # order-preserving rescale) so inits are commensurate AND cross-question comparability --
            # which pooled candidate AUROC needs -- is preserved.
            acc += (best_scores - best_scores.mean()) / (best_scores.std() + 1e-9)
        oof[te] = acc / A.inits
    return oof, chosen


def fit_predict_gbm(fs, seed_base=0):
    idx = FSETS[fs]
    oof = np.zeros((n_q, N))
    for tr, te in FOLDS:
        g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=seed_base)
        g.fit(X[tr][:, :, idx].reshape(-1, len(idx)), sl[tr].reshape(-1))
        oof[te] = g.predict_proba(X[te][:, :, idx].reshape(-1, len(idx)))[:, 1].reshape(len(te), N)
    return oof


# ---------------------------------------------------------------- 6. paired bootstrap
def paired_boot(ok_a, ok_b, mask, nboot, seed, cluster=None):
    """Paired bootstrap of mean(ok_a)-mean(ok_b) over the masked questions.
    cluster=None -> resample questions (the incumbent artifact's convention);
    cluster=array -> resample CLUSTERS (image groups) for a correlation-robust check."""
    rng = np.random.default_rng(seed)
    a, b = ok_a[mask], ok_b[mask]
    if cluster is None:
        n = len(a)
        d = np.empty(nboot)
        for i in range(nboot):
            s = rng.integers(0, n, n)
            d[i] = a[s].mean() - b[s].mean()
    else:
        cl = cluster[mask]
        uc = np.unique(cl)
        members = [np.flatnonzero(cl == c) for c in uc]
        d = np.empty(nboot)
        for i in range(nboot):
            s = rng.integers(0, len(uc), len(uc))
            take = np.concatenate([members[j] for j in s])
            d[i] = a[take].mean() - b[take].mean()
    return float(a.mean() - b.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


_, inc_ok = metrics(V)
RESULTS = {}
ARMS = []
for fs in ["V", "FULL", "NOV"]:
    for obj in OBJ_LIST:
        ARMS.append((f"mlp_{fs}_{obj}", fs, obj, "mlp"))
for fs in ["FULL", "NOV"]:
    for obj in OBJ_LIST:
        ARMS.append((f"linear_{fs}_{obj}", fs, obj, "linear"))

for name, fs, obj, arch in ARMS:
    S, eps = fit_predict_nn(fs, obj, arch, seed_base=A.seed)
    m, ok = metrics(S)
    m["mean_selected_epoch"] = float(np.mean(eps))
    m["delta_sel_eff_vs_incumbent"], m["ci_sel_eff_vs_incumbent"] = paired_boot(ok, inc_ok, has, A.nboot, A.seed)
    m["delta_sel_acc_vs_incumbent"], m["ci_sel_acc_vs_incumbent"] = paired_boot(
        ok, inc_ok, np.ones(n_q, bool), A.nboot, A.seed)
    m["delta_sel_eff_contested"], m["ci_sel_eff_contested"] = paired_boot(
        ok, inc_ok, has & contested, A.nboot, A.seed)
    m["delta_sel_eff_mixed"], m["ci_sel_eff_mixed"] = paired_boot(ok, inc_ok, mixed, A.nboot, A.seed)
    m["picks"] = sel_from_scores(S).tolist()
    RESULTS[name] = m
    print(f"[arm] {name:34s} sel_eff {m['sel_eff']:.4f} ({m['delta_sel_eff_vs_incumbent']:+.4f} "
          f"[{m['ci_sel_eff_vs_incumbent'][0]:+.4f},{m['ci_sel_eff_vs_incumbent'][1]:+.4f}])  "
          f"acc {m['sel_acc']:.4f}  auroc {m['auroc']:.4f}", flush=True)
    RESULTS[name]["_ok"] = ok.tolist()

for fs in ["FULL", "NOV"]:
    S = fit_predict_gbm(fs, seed_base=A.seed)
    m, ok = metrics(S)
    m["delta_sel_eff_vs_incumbent"], m["ci_sel_eff_vs_incumbent"] = paired_boot(ok, inc_ok, has, A.nboot, A.seed)
    m["delta_sel_acc_vs_incumbent"], m["ci_sel_acc_vs_incumbent"] = paired_boot(
        ok, inc_ok, np.ones(n_q, bool), A.nboot, A.seed)
    m["delta_sel_eff_contested"], m["ci_sel_eff_contested"] = paired_boot(
        ok, inc_ok, has & contested, A.nboot, A.seed)
    m["delta_sel_eff_mixed"], m["ci_sel_eff_mixed"] = paired_boot(ok, inc_ok, mixed, A.nboot, A.seed)
    m["picks"] = sel_from_scores(S).tolist()
    m["_ok"] = ok.tolist()
    RESULTS[f"gbm_{fs}_pointwise_bce"] = m
    print(f"[arm] gbm_{fs}_pointwise_bce  sel_eff {m['sel_eff']:.4f} "
          f"({m['delta_sel_eff_vs_incumbent']:+.4f})", flush=True)

# ---------------------------------------------------------------- 7. objective contrasts (the endpoint)
CONTRASTS = {}
for arch in ["mlp", "linear"]:
  for base_obj in ["pointwise_bce", "pointwise_bce_nondeg"]:
    for fs in ["V", "FULL", "NOV"]:
        base = f"{arch}_{fs}_{base_obj}"
        if base not in RESULTS:
            continue
        bok = np.array(RESULTS[base]["_ok"])
        for obj in ["listnet", "anypos", "ranknet_bt"] + (
                ["pointwise_bce_nondeg"] if base_obj == "pointwise_bce" else []):
            k = f"{arch}_{fs}_{obj}"
            aok = np.array(RESULTS[k]["_ok"])
            p, ci = paired_boot(aok, bok, has, A.nboot, A.seed)
            pc, cic = paired_boot(aok, bok, has & contested, A.nboot, A.seed)
            CONTRASTS[f"{k}__minus__{base}"] = {
                "delta_sel_eff": p, "ci": ci, "significant": ci[0] > 0 or ci[1] < 0,
                "delta_sel_eff_contested": pc, "ci_contested": cic}
            print(f"[obj] {k} - {base}: sel_eff {p:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]", flush=True)

# best arm, image-clustered robustness bootstrap + per-set guardrail
best = max(RESULTS, key=lambda k: RESULTS[k]["sel_eff"])
bok = np.array(RESULTS[best]["_ok"])
pc, cic = paired_boot(bok, inc_ok, has, A.nboot, A.seed, cluster=groups)
GUARD = {}
for d in DS:
    m = ds_of == d
    p, ci = paired_boot(bok, inc_ok, has & m, A.nboot, A.seed)
    GUARD[d] = {"delta_sel_eff": p, "ci": ci}
BEST = {"arm": best, "sel_eff": RESULTS[best]["sel_eff"],
        "delta_vs_incumbent_question_bootstrap": RESULTS[best]["delta_sel_eff_vs_incumbent"],
        "ci_question_bootstrap": RESULTS[best]["ci_sel_eff_vs_incumbent"],
        "delta_vs_incumbent_image_clustered_bootstrap": pc, "ci_image_clustered": cic,
        "per_dataset_guardrail": GUARD,
        "guardrail_clean": all(v["ci"][1] > 0 or v["delta_sel_eff"] >= 0 for v in GUARD.values())}

for k in RESULTS:
    RESULTS[k].pop("_ok", None)

OUT = {
    "what": "T1.B -- LISTWISE / learning-to-rank vs POINTWISE on IDENTICAL features, folds, architecture "
            "and model-selection criterion. Tests whether the incumbent's pointwise-then-argmax objective "
            "mismatch explains the best-of-8 selector gap.",
    "date": "2026-08-04",
    "code": "src/training_methods/verifarch_listwise.py (+ verifarch_eval_imghash.py for fold groups)",
    "endpoint": ("selection efficiency P(pick correct | correct present) at N=8, pooled and per dataset, "
                 f"paired bootstrap 95% CI, nboot={A.nboot}, seed={A.seed}"),
    "incumbent": "ckpts/train/lora_verifier_disjoint (clean L1 image-disjoint LoRA verifier), sel_eff 0.775204",
    "null_test": null_report,
    "controls": CTRL,
    "n_questions": n_q, "n_images": int(len(set(groups))),
    "n_contested_questions": int(contested.sum()),
    "n_contested_recoverable": int((contested & has).sum()),
    "n_mixed_label_questions": int(mixed.sum()),
    "incumbent_picks": sel_from_scores(V).tolist(),
    "judge_labels_sl": sl.astype(int).tolist(),
    "dataset_of_question": ds_of.tolist(),
    "features": FEAT_NAMES,
    "feature_sets": {k: [FEAT_NAMES[i] for i in v] for k, v in FSETS.items()},
    "n_nondegenerate_lists": int(nondeg_mask.sum()),
    "n_all_correct_lists": int((sl.sum(1) == N).sum()),
    "n_all_wrong_lists": int((sl.sum(1) == 0).sum()),
    "protocol": {
        "folds": f"{A.folds}-fold GroupKFold, groups = md5 of decoded RGB pixels of the eval image "
                 "(image-disjointness asserted in code)",
        "auroc_caveat": "pooled candidate AUROC for fitted arms is computed on out-of-fold scores that "
                        "were standardised per fold, so it is approximate; sel_eff/sel_acc are exact "
                        "(argmax is within-question and invariant to any per-fold affine rescale).",
        "epoch_selection": "inner 4-fold GroupKFold on the training folds; the epoch maximising INNER "
                           "VALIDATION SELECTION EFFICIENCY is chosen -- identical criterion for every arm",
        "ensembling": f"{A.inits} inits per fold, averaged in globally z-scored raw-score space, "
                      "identical for every arm",
        "degenerate_lists": "listwise/pairwise objectives (and the pointwise_bce_nondeg control) are "
                            "trained only on lists that contain at least one correct and one wrong "
                            "candidate; plain pointwise_bce sees every list.",
        "architecture": "2x64 ReLU MLP scoring each candidate independently (and a linear variant); "
                        "the loss is the ONLY thing that differs between arms",
        "honesty": "arms whose feature set is V or FULL use the incumbent verifier's score, which exists "
                   "only on the 2345 EVAL items (the adapter was never run over the 16,621 disjoint train "
                   "items). They are cross-fitted DIAGNOSTICS / upper bounds, not deployable numbers. The "
                   "NOV arms use no incumbent-derived feature and carry no such caveat.",
    },
    "arms": RESULTS,
    "objective_contrasts_same_features": CONTRASTS,
    "best_arm": BEST,
}
os.makedirs(os.path.dirname(J(A.out)), exist_ok=True)
json.dump(OUT, open(J(A.out), "w"), indent=1)
print("\nwrote", J(A.out))
