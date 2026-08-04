#!/usr/bin/env python3
"""verifarch_features.py -- TIER-1 FEATURE-BASED DISCRIMINATIVE SELECTOR for best-of-8 open-text
medical VQA.  A categorically different computation from every verifier tried so far in this project:
no language model produces an opinion.  Gradient-boosted trees / a small MLP score each candidate from
ENGINEERED, ZERO-INFERENCE features (pool duplicate structure, string-similarity structure, answer shape,
question relation, train-derived answer-string priors) -- everything already on disk.

WHY THIS AND NOT A JUDGE.  Fifteen generative-verifier variants have failed (see the brief).  The one
computation nobody has run here is a discriminative head over hand-made features.  It costs CPU-minutes,
so its value is not only "does it win" but "WHICH SIGNALS CARRY SELECTION INFORMATION" -- reported here as
drop-one-group ablations and within-pool permutation importance on the selection endpoint itself.

WHAT IS MEASURED.  Endpoint = selection efficiency at N=8, sel_eff = P(pick correct | correct present),
pooled and per dataset, paired bootstrap 95% CI vs the incumbent trained LoRA verifier (0.7752).
Also reported: selected accuracy, candidate-level AUROC, and the controls (greedy, random, self-
consistency, oracle@8, the generator's own zero-shot P(Yes)).

TWO ARMS, because the incumbent's score is available on eval but NOT on the 16,621 disjoint train items
(the adapter was never run over them):
  ARM A  DEPLOYABLE.  Features only (no incumbent score).  Fitted on the L1 image-disjoint TRAIN pools
         (slake/vqa_rad/pathvqa official train + kvasir + radimagenet, 16,621 questions), frozen, applied
         once to the 2,345 eval items.  No eval question and no eval image is in training -- re-proven at
         md5-of-decoded-RGB-pixels level by src/training_methods/verifarch_assert_disjoint.py.
  ARM B  DIAGNOSTIC ONLY (never a deployable number).  Features + the incumbent's per-candidate score,
         cross-fitted on the eval set with GroupKFold over IMAGE PIXEL HASH (no fold shares an image),
         because there is no other way to obtain the incumbent's score on training items.

FOUR OBJECTIVES ON IDENTICAL FEATURES (this is the T1.B "is the constraint the objective or the
information?" test): HGB pointwise BCE, MLP pointwise BCE, MLP listwise softmax, MLP within-question
Bradley-Terry.

  python3 src/cascade_methods/verifarch_features.py
  -> results/cascade_methods/artifacts/verifarch_features_2026-08-04.json
"""
import argparse, difflib, json, math, os, re, string
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
CK = J("ckpts/openvqa/cheap_lingshu7b")
DUMP = J("ckpts/train/lora_verifier_disjoint")
XF = J("ckpts/openvqa/crossfam_verifier")

ap = argparse.ArgumentParser()
ap.add_argument("--nboot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/cascade_methods/artifacts/verifarch_features_2026-08-04.json")
A = ap.parse_args()
RNG = np.random.default_rng(A.seed)

EVAL_SETS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
TRAIN_SETS = ["slake_open_train", "vqa_rad_open_train", "pathvqa_open_train",
              "kvasir_open", "radimagenet_open"]
INCUMBENT_SEL_EFF = 0.7752043596730246   # reproduced by the null test below


# ----------------------------------------------------------------------------- io
def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def norm(s):
    return str(s).strip().lower()


def qnorm(s):
    s = str(s).lower().strip().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def toks(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def load_eval():
    """items: dict with ds, idx, question, preds(8), labels(8), incumbent(8), greedy_ok."""
    items = []
    for ds in EVAL_SETS:
        sc = {r["idx"]: r for r in jl(f"{CK}/ckpt_{ds}_lingshu7b_sc8.jsonl")}
        fam = ds.split("_open")[0]
        for r in json.load(open(f"{DUMP}/transfer_dump_{fam}_open_lingshu7b.json")):
            s = sc[r["idx"]]
            assert s["preds"] == r["preds"], "pool mismatch between sc8 and transfer dump"
            items.append(dict(ds=ds, idx=r["idx"], question=s["question"], gold=s["gold"],
                              preds=r["preds"], labels=[int(x) for x in r["sl"]],
                              incumbent=[float(x) for x in r["scores"]],
                              greedy_ok=int(r["greedy_ok"])))
    return items


def load_train():
    items = []
    for ds in TRAIN_SETS:
        sc = jl(f"{CK}/ckpt_{ds}_lingshu7b_sc8.jsonl")
        jud = {r["idx"]: r["judge_ok"] for r in jl(f"{CK}/ckpt_{ds}_lingshu7b_sc8_scexploded.judge.jsonl")}
        exp = {r["idx"]: r for r in jl(f"{CK}/ckpt_{ds}_lingshu7b_sc8_scexploded.jsonl")}
        allow = set(json.load(open(J(f"data/disjoint_split/idx_{ds}.json"))))
        by_item = defaultdict(dict)
        for cid, r in exp.items():
            if cid in jud:
                oi = cid.split("#")[0]
                oi = int(oi) if oi.lstrip("-").isdigit() else oi
                by_item[oi][norm(r["modal_pred"])] = jud[cid]
        n_drop = 0
        for s in sc:
            if s["idx"] not in allow:
                n_drop += 1
                continue
            m = by_item.get(s["idx"])
            if not m:
                continue
            labs = [m.get(norm(p)) for p in s["preds"]]
            if any(x is None for x in labs):
                continue
            items.append(dict(ds=ds, idx=s["idx"], question=s["question"], gold=s.get("gold", ""),
                              preds=s["preds"], labels=[int(x) for x in labs],
                              incumbent=None, greedy_ok=None))
        print(f"  train {ds:20s} pool={len(sc):6d} not_in_L1_allowlist={n_drop:5d} usable={sum(1 for i in items if i['ds']==ds):6d}",
              flush=True)
    return items


# ----------------------------------------------------------------------------- features
LAT = re.compile(r"\b(left|right|bilateral|both|lateral|medial|superior|inferior|anterior|posterior|upper|lower)\b")
NEG = re.compile(r"\b(no|not|none|absent|without|normal|negative|nothing|unremarkable)\b")
WH = ["what", "where", "which", "how", "why", "when", "who", "is", "are", "does", "do"]


def pool_features(item, pget):
    """Return (8, F) float array + the feature-name list. ZERO inference: only the candidate strings,
    the pool's duplicate/similarity structure, the question text, and TRAIN-derived string priors."""
    preds = item["preds"]
    n = len(preds)
    npred = [norm(p) for p in preds]
    cnt = Counter(npred)
    maxc = max(cnt.values())
    order = {}                                    # first appearance index of each distinct string
    for i, p in enumerate(npred):
        order.setdefault(p, i)
    distinct = list(cnt)
    # ranked distinct strings by count desc, first-appearance asc
    ranked = sorted(distinct, key=lambda s: (-cnt[s], order[s]))
    rank_of = {s: i for i, s in enumerate(ranked)}
    ent = -sum((c / n) * math.log(c / n) for c in cnt.values())
    qt = toks(item["question"])
    qn = qnorm(item["question"])
    first_word = qn.split()[0] if qn else ""
    wl = [len(re.findall(r"[a-z0-9]+", s)) for s in npred]
    med_w = float(np.median(wl)) if wl else 1.0
    # pairwise string similarity between DISTINCT strings (cheap; <=8x8 short strings)
    sim = {}
    for a in distinct:
        for b in distinct:
            if a < b:
                sim[(a, b)] = difflib.SequenceMatcher(None, a, b).ratio()
    getsim = lambda a, b: 1.0 if a == b else sim[(min(a, b), max(a, b))]
    jac = {}
    tk = {s: toks(s) for s in distinct}
    for a in distinct:
        for b in distinct:
            if a < b:
                u = tk[a] | tk[b]
                jac[(a, b)] = (len(tk[a] & tk[b]) / len(u)) if u else 0.0
    getjac = lambda a, b: 1.0 if a == b else jac[(min(a, b), max(a, b))]

    rows = []
    for i, p in enumerate(npred):
        c = cnt[p]
        others = [(q, cnt[q]) for q in distinct if q != p]
        wsum = sum(k for _, k in others)
        mean_sim = (sum(getsim(p, q) * k for q, k in others) / wsum) if wsum else 0.0
        max_sim = max([getsim(p, q) for q, _ in others], default=0.0)
        mean_jac = (sum(getjac(p, q) * k for q, k in others) / wsum) if wsum else 0.0
        soft_vote = (c - 1 + sum(getsim(p, q) * k for q, k in others)) / (n - 1)
        n_contains = sum(1 for q, _ in others if p in q)
        n_contained = sum(1 for q, _ in others if q in p)
        pw = len(re.findall(r"[a-z0-9]+", p))
        at = tk[p]
        pc, pp = pget(p)
        rows.append([
            # --- pool duplicate structure ---
            c, c / n, float(c == maxc), float(rank_of[p]), float(order[p]), float(order[p] == 0),
            float(len(distinct)), maxc / n, ent,
            # --- similarity / agreement structure ---
            mean_sim, max_sim, mean_jac, soft_vote, float(n_contains), float(n_contained),
            # --- answer shape ---
            float(len(p)), float(pw), math.log((pw + 1) / (med_w + 1)),
            float(bool(re.search(r"\d", p))), float(p.count(",")),
            float(bool(LAT.search(p))), float(bool(NEG.search(p))),
            # --- question relation ---
            (len(at & qt) / len(at | qt)) if (at | qt) else 0.0,
            float(len(qt)), float(len(at & qt)),
        ] + [float(first_word == w) for w in WH] + [
            # --- TRAIN-derived answer-string priors (leave-one-question-out on train) ---
            math.log1p(pc),
            (pp / pc) if pc > 0 else 0.2,
            float(pc > 0),
        ])
    return np.asarray(rows, dtype=np.float64)


FEAT_NAMES = ([
    "dup_count", "dup_frac", "is_modal", "dup_rank", "first_pos", "is_first_sample",
    "n_distinct", "self_consistency", "pool_entropy",
    "mean_sim_others", "max_sim_other", "mean_jaccard_others", "soft_vote", "n_others_containing",
    "n_others_contained",
    "n_chars", "n_words", "log_len_ratio", "has_digit", "n_commas", "has_laterality", "has_negation",
    "q_answer_jaccard", "q_n_tokens", "q_answer_overlap",
] + [f"q_starts_{w}" for w in WH] + ["train_str_logcount", "train_str_poscorrect_rate", "train_str_seen"])

GROUPS = {
    "pool_duplicate_structure": ["dup_count", "dup_frac", "is_modal", "dup_rank", "first_pos",
                                 "is_first_sample", "n_distinct", "self_consistency", "pool_entropy"],
    "similarity_structure": ["mean_sim_others", "max_sim_other", "mean_jaccard_others", "soft_vote",
                             "n_others_containing", "n_others_contained"],
    "answer_shape": ["n_chars", "n_words", "log_len_ratio", "has_digit", "n_commas",
                     "has_laterality", "has_negation"],
    "question_relation": ["q_answer_jaccard", "q_n_tokens", "q_answer_overlap"] + [f"q_starts_{w}" for w in WH],
    "train_answer_prior": ["train_str_logcount", "train_str_poscorrect_rate", "train_str_seen"],
}


def build_prior(train_items, exclude_item=None):
    """normalized answer string -> (n_occurrences_in_train_pools, n_judged_correct). TRAIN ONLY."""
    pr = {}
    for it in train_items:
        seen = {}
        for p, l in zip(it["preds"], it["labels"]):
            seen[norm(p)] = l
        for s, l in seen.items():
            a, b = pr.get(s, (0, 0))
            pr[s] = (a + 1, b + l)
    return pr


def make_pget(prior, item=None):
    """Return s -> (count, n_correct) from the TRAIN prior. If `item` is given, its own contribution is
    subtracted (leave-one-question-out) so a training item never sees its own label through the prior."""
    if item is None:
        return lambda s: prior.get(s, (0, 0))
    own = {}
    for p, l in zip(item["preds"], item["labels"]):
        own[norm(p)] = l
    def pget(s):
        a, b = prior.get(s, (0, 0))
        if s in own:
            a, b = a - 1, b - own[s]
        return (max(a, 0), max(b, 0))
    return pget


def featurize(items, prior, loo=False):
    X, y = [], []
    for it in items:
        X.append(pool_features(it, make_pget(prior, it if loo else None)))
        y.append(np.asarray(it["labels"], dtype=np.float64))
    return np.stack(X), np.stack(y), np.arange(len(items))


# ----------------------------------------------------------------------------- endpoint
def sel_stats(scores, labels):
    """scores,labels: (n,8). Returns dict with sel_acc, sel_eff, oracle, and per-item pick vector."""
    pick = np.argmax(scores, axis=1)
    ok = labels[np.arange(len(labels)), pick]
    orc = labels.max(axis=1)
    return dict(sel_acc=float(ok.mean()), oracle=float(orc.mean()),
                sel_eff=float(ok[orc == 1].mean()), n=len(ok)), ok, orc


def paired_boot(ok_a, orc, ok_b, nboot, rng):
    """paired bootstrap over ITEMS of (sel_eff_a - sel_eff_b) and (sel_acc_a - sel_acc_b)."""
    n = len(ok_a)
    d_eff, d_acc = [], []
    for _ in range(nboot):
        r = rng.integers(0, n, n)
        m = orc[r] == 1
        if m.sum() == 0:
            continue
        d_eff.append(ok_a[r][m].mean() - ok_b[r][m].mean())
        d_acc.append(ok_a[r].mean() - ok_b[r].mean())
    f = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(d_sel_eff=float(ok_a[orc == 1].mean() - ok_b[orc == 1].mean()), ci_sel_eff=f(d_eff),
                d_sel_acc=float(ok_a.mean() - ok_b.mean()), ci_sel_acc=f(d_acc))


def boot_ci(ok, orc, nboot, rng):
    n = len(ok)
    v = []
    for _ in range(nboot):
        r = rng.integers(0, n, n)
        m = orc[r] == 1
        if m.sum():
            v.append(ok[r][m].mean())
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ----------------------------------------------------------------------------- models
def fit_hgb(Xtr, ytr, seed=0):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                       min_samples_leaf=50, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.15,
                                       random_state=seed)
    m.fit(Xtr.reshape(-1, Xtr.shape[-1]), ytr.reshape(-1))
    return lambda X: m.predict_proba(X.reshape(-1, X.shape[-1]))[:, 1].reshape(X.shape[:2]), m


def fit_mlp(Xtr, ytr, loss="listwise", seed=0, epochs=300, hid=64, lr=3e-3, wd=1e-3, dev="cpu"):
    """Xtr (n,8,F) ytr (n,8). loss in {pointwise, listwise, bt}. dev defaults to cpu, which is what the
    reported ARM A numbers were produced with; dev='cuda' is used only by the follow-up sweep."""
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    Xz = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    Y = torch.tensor(ytr, dtype=torch.float32, device=dev)
    F = Xz.shape[-1]
    net = torch.nn.Sequential(torch.nn.Linear(F, hid), torch.nn.Tanh(),
                              torch.nn.Linear(hid, hid // 2), torch.nn.Tanh(),
                              torch.nn.Linear(hid // 2, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    has_pos = (Y.sum(1) > 0)
    has_mix = has_pos & (Y.sum(1) < Y.shape[1])
    for ep in range(epochs):
        opt.zero_grad()
        s = net(Xz).squeeze(-1)                                  # (n,8)
        if loss == "pointwise":
            L = torch.nn.functional.binary_cross_entropy_with_logits(s, Y)
        elif loss == "listwise":
            lp = torch.log_softmax(s, dim=1)
            sel = torch.where(Y[has_pos] > 0, lp[has_pos], torch.full_like(lp[has_pos], -1e9))
            L = -torch.logsumexp(sel, dim=1).mean()
        elif loss == "bt":
            sm, Ym = s[has_mix], Y[has_mix]
            d = sm.unsqueeze(2) - sm.unsqueeze(1)                # (m,8,8) s_i - s_j
            w = (Ym.unsqueeze(2) > Ym.unsqueeze(1)).float()      # i correct, j not
            L = -(torch.nn.functional.logsigmoid(d) * w).sum() / w.sum().clamp(min=1)
        else:
            raise ValueError(loss)
        L.backward()
        opt.step()
    net.eval()

    def score(X):
        import torch as T
        with T.no_grad():
            return net(T.tensor((X - mu) / sd, dtype=T.float32, device=dev)).squeeze(-1).cpu().numpy()
    return score, net


# ----------------------------------------------------------------------------- main
def main():
    out = {
        "what": ("TIER-1 feature-based DISCRIMINATIVE selector for best-of-8 open-text medical VQA: "
                 "gradient-boosted trees / small MLPs over engineered zero-inference per-candidate "
                 "features, replacing the generative judge entirely. Reports which signals carry "
                 "selection information."),
        "date": "2026-08-04",
        "generator": "Lingshu-7B (temp 0.7, n=8, cap320) -- pools UNCHANGED from the published open-text arm",
        "judge": "src/labeling/run_judge.py (MedVLThinker-32B, judge_ok) -- the SAME judge as the headline",
        "incumbent": "ckpts/train/lora_verifier_disjoint (trained same-family LoRA, clean L1 image-disjoint)",
        "endpoint": "selection efficiency at N=8 = P(pick correct | correct present); paired bootstrap over items",
        "nboot": A.nboot, "seed": A.seed,
        "code": "src/cascade_methods/verifarch_features.py",
    }

    print("[load] eval ...", flush=True)
    ev = load_eval()
    print(f"  eval items {len(ev)}", flush=True)
    print("[load] train (L1 image-disjoint pools) ...", flush=True)
    tr = load_train()
    print(f"  train items {len(tr)}", flush=True)
    out["data"] = {"n_eval_items": len(ev), "n_eval_candidates": 8 * len(ev),
                   "n_train_items": len(tr), "n_train_candidates": 8 * len(tr),
                   "train_sources": {ds: sum(1 for i in tr if i["ds"] == ds) for ds in TRAIN_SETS}}

    # ---------------- disjointness (re-proven by the companion script) ----------------
    ap_path = J("results/cascade_methods/artifacts/verifarch_disjoint_assert.json")
    assert os.path.exists(ap_path), "run src/training_methods/verifarch_assert_disjoint.py first"
    dj = json.load(open(ap_path))
    assert dj["total_image_pixel_hash_intersection"] == 0 and dj["total_item_intersection"] == 0
    out["disjointness"] = {"artifact": "results/cascade_methods/artifacts/verifarch_disjoint_assert.json",
                           "image_pixel_hash_intersection": dj["total_image_pixel_hash_intersection"],
                           "item_intersection": dj["total_item_intersection"],
                           "method": dj["method"]}
    imghash = json.load(open(J("data/verifarch/eval_imghash.json")))

    # ---------------- NULL TEST ----------------
    Yev = np.stack([np.asarray(i["labels"], float) for i in ev])
    Sinc = np.stack([np.asarray(i["incumbent"], float) for i in ev])
    st_inc, ok_inc, orc = sel_stats(Sinc, Yev)
    auroc_inc = roc_auc_score(Yev.reshape(-1), Sinc.reshape(-1))
    greedy = float(np.mean([i["greedy_ok"] for i in ev]))
    pub = {"sel_eff": 0.7752, "oracle_at_8": 0.6260127931769722, "greedy": 0.4494669509594883,
           "selected": 0.48528784648187634, "auroc": 0.8855921901711237}
    null = {"published": pub,
            "reproduced": {"sel_eff": st_inc["sel_eff"], "oracle_at_8": st_inc["oracle"],
                           "greedy": greedy, "selected": st_inc["sel_acc"], "auroc": float(auroc_inc)},
            "max_abs_deviation": max(abs(st_inc["oracle"] - pub["oracle_at_8"]),
                                     abs(greedy - pub["greedy"]),
                                     abs(st_inc["sel_acc"] - pub["selected"]),
                                     abs(float(auroc_inc) - pub["auroc"]))}
    null["tolerance_achieved"] = f"exact to {null['max_abs_deviation']:.2e} on every cell"
    print(f"[NULL TEST] sel_eff={st_inc['sel_eff']:.6f} oracle={st_inc['oracle']:.6f} "
          f"greedy={greedy:.6f} selected={st_inc['sel_acc']:.6f} auroc={auroc_inc:.6f} "
          f"maxdev={null['max_abs_deviation']:.2e}", flush=True)
    assert null["max_abs_deviation"] < 1e-9, "NULL TEST FAILED -- harness does not reproduce the incumbent"
    out["null_test"] = null

    # ---------------- controls ----------------
    n = len(ev)
    # random pick: exact expectation = mean over items of (#correct/8) restricted to recoverable
    rnd_ok = np.array([np.mean(i["labels"]) for i in ev])
    # self-consistency (normalized-string mode, first-appearance tiebreak) as a SCORE = dup count
    Sdup = np.stack([[Counter(norm(p) for p in i["preds"])[norm(c)] for c in i["preds"]] for i in ev], dtype=float)
    st_sc, ok_sc, _ = sel_stats(Sdup, Yev)
    # generator's own zero-shot P(Yes)
    zs = {}
    for ds in EVAL_SETS:
        for r in jl(f"{XF}/ckpt_{ds}_lingshu7b_zs.jsonl"):
            zs[(ds, r["idx"])] = r["scores_by_answer"]
    Szs, have_zs = [], True
    for i in ev:
        m = zs.get((i["ds"], i["idx"]))
        if m is None:
            have_zs = False
            break
        Szs.append([m.get(norm(p), 0.5) for p in i["preds"]])
    ctrl = {"greedy_published_modal_raw": greedy,
            "first_sample": float(np.mean([i["labels"][0] for i in ev])),
            "random_pick_expected": float(rnd_ok[orc == 1].mean()),
            "self_consistency": st_sc,
            "oracle_at_8": st_inc["oracle"],
            "incumbent_trained_lora": st_inc}
    ok_zs = None
    if have_zs:
        st_zs, ok_zs, _ = sel_stats(np.asarray(Szs), Yev)
        ctrl["generator_zero_shot_pyes"] = st_zs
    out["controls"] = ctrl
    print(f"[controls] SC eff={st_sc['sel_eff']:.4f} random eff={ctrl['random_pick_expected']:.4f} "
          f"zs eff={ctrl.get('generator_zero_shot_pyes',{}).get('sel_eff','na')}", flush=True)

    # ---------------- features ----------------
    print("[feat] building train-derived answer-string prior + features ...", flush=True)
    prior = build_prior(tr)
    Xtr, Ytr, _ = featurize(tr, prior, loo=True)      # leave-one-question-out prior inside train
    Xev, Yev2, _ = featurize(ev, prior, loo=False)
    assert np.array_equal(Yev2, Yev)
    assert Xtr.shape[-1] == len(FEAT_NAMES), (Xtr.shape, len(FEAT_NAMES))
    out["features"] = {"n_features": len(FEAT_NAMES), "names": FEAT_NAMES, "groups": GROUPS,
                       "inference_cost": "zero -- candidate strings + question text + train-derived priors only",
                       "train_prior_protocol": "leave-one-question-out within train; eval uses the full train prior"}
    print(f"  X_train {Xtr.shape}  X_eval {Xev.shape}  pos_rate_train={Ytr.mean():.4f}", flush=True)

    # ---------------- ARM A: deployable, train-fitted, frozen ----------------
    print("[ARM A] fitting on disjoint train pools, frozen -> eval ...", flush=True)
    armA = {}
    scorers = {}
    for name, fitter in [("hgb_pointwise_bce", lambda: fit_hgb(Xtr, Ytr, A.seed)),
                         ("mlp_pointwise_bce", lambda: fit_mlp(Xtr, Ytr, "pointwise", A.seed)),
                         ("mlp_listwise_softmax", lambda: fit_mlp(Xtr, Ytr, "listwise", A.seed)),
                         ("mlp_pairwise_bt", lambda: fit_mlp(Xtr, Ytr, "bt", A.seed))]:
        sc_fn, mdl = fitter()
        S = sc_fn(Xev)
        st, ok, _ = sel_stats(S, Yev)
        st["auroc"] = float(roc_auc_score(Yev.reshape(-1), S.reshape(-1)))
        st["ci_sel_eff"] = boot_ci(ok, orc, A.nboot, np.random.default_rng(A.seed))
        st["vs_incumbent"] = paired_boot(ok, orc, ok_inc, A.nboot, np.random.default_rng(A.seed))
        st["vs_self_consistency"] = paired_boot(ok, orc, ok_sc, A.nboot, np.random.default_rng(A.seed))
        st["per_dataset"] = {}
        for ds in EVAL_SETS:
            m = np.array([i["ds"] == ds for i in ev])
            s2, o2, r2 = sel_stats(S[m], Yev[m])
            s2["vs_incumbent"] = paired_boot(o2, r2, ok_inc[m], A.nboot, np.random.default_rng(A.seed))
            st["per_dataset"][ds] = s2
        armA[name] = st
        scorers[name] = (sc_fn, S, ok)
        print(f"  {name:22s} sel_eff={st['sel_eff']:.4f} {st['ci_sel_eff']} sel_acc={st['sel_acc']:.4f} "
              f"auroc={st['auroc']:.4f} d_vs_inc={st['vs_incumbent']['d_sel_eff']:+.4f} "
              f"{st['vs_incumbent']['ci_sel_eff']}", flush=True)
    out["arm_A_deployable"] = {
        "protocol": ("fitted on the 5 L1 image-disjoint TRAIN pools, frozen, applied once to the 2345 eval "
                     "items; no eval image or item in training (pixel-md5 asserted). NO incumbent score used."),
        "results": armA}

    # ---------------- feature importance (ARM A, best model) ----------------
    best = max(armA, key=lambda k: armA[k]["sel_eff"])
    sc_fn = scorers[best][0]
    print(f"[importance] permutation on the SELECTION endpoint, model={best} ...", flush=True)
    rng = np.random.default_rng(A.seed)
    base_eff = armA[best]["sel_eff"]
    perm = {}
    for fi, fname in enumerate(FEAT_NAMES):
        vals = []
        for rep in range(10):
            Xp = Xev.copy()
            order = rng.permutation(len(Xp))            # permute the feature ACROSS pools, keeping the
            Xp[:, :, fi] = Xp[order, :, fi]             # within-pool pattern intact
            s, o, _ = sel_stats(sc_fn(Xp), Yev)
            vals.append(base_eff - s["sel_eff"])
        perm[fname] = {"drop_sel_eff_mean": float(np.mean(vals)), "sd": float(np.std(vals))}
    grp = {}
    for gname, members in GROUPS.items():
        keep = [i for i, f in enumerate(FEAT_NAMES) if f not in members]
        LOSS = {"mlp_pointwise_bce": "pointwise", "mlp_listwise_softmax": "listwise",
                "mlp_pairwise_bt": "bt"}
        s_fn2, _ = (fit_hgb(Xtr[:, :, keep], Ytr, A.seed) if best.startswith("hgb")
                    else fit_mlp(Xtr[:, :, keep], Ytr, LOSS[best], A.seed))
        st2, _, _ = sel_stats(s_fn2(Xev[:, :, keep]), Yev)
        grp[gname] = {"sel_eff_without_group": st2["sel_eff"], "drop": base_eff - st2["sel_eff"],
                      "n_features_removed": len(members)}
        print(f"  drop-group {gname:26s} -> {st2['sel_eff']:.4f} (drop {base_eff-st2['sel_eff']:+.4f})", flush=True)
    # single-feature-only models (what does ONE signal buy on its own)
    solo = {}
    for gname, members in GROUPS.items():
        keep = [i for i, f in enumerate(FEAT_NAMES) if f in members]
        s_fn2, _ = fit_hgb(Xtr[:, :, keep], Ytr, A.seed)
        st2, _, _ = sel_stats(s_fn2(Xev[:, :, keep]), Yev)
        solo[gname] = {"sel_eff_group_alone": st2["sel_eff"]}
    out["feature_importance"] = {
        "model": best, "base_sel_eff": base_eff,
        "protocol": ("across-pool permutation of one feature column at a time (10 reps), scored by the loss "
                     "in SELECTION EFFICIENCY, not AUROC; plus retrain-without-group and group-alone models"),
        "permutation": dict(sorted(perm.items(), key=lambda kv: -kv[1]["drop_sel_eff_mean"])),
        "drop_one_group": grp, "group_alone": solo}

    # ---------------- ARM B: + incumbent score, cross-fitted on eval by IMAGE ----------------
    print("[ARM B] cross-fitted on eval (GroupKFold over image pixel hash), features + incumbent score ...",
          flush=True)
    inc_feats = []
    for i, it in enumerate(ev):
        s = np.asarray(it["incumbent"], float)
        z = (s - s.mean()) / (s.std() + 1e-8)
        rk = np.argsort(np.argsort(-s)).astype(float)
        inc_feats.append(np.stack([s, s - s.max(), z, rk], axis=1))
    Xb = np.concatenate([Xev, np.stack(inc_feats)], axis=2)
    NAMES_B = FEAT_NAMES + ["incumbent_score", "incumbent_margin_to_max", "incumbent_z_in_pool",
                            "incumbent_rank_in_pool"]
    groups = np.array([imghash[i["ds"]][str(i["idx"])] for i in ev])
    uniq = {h: k for k, h in enumerate(sorted(set(groups)))}
    gidx = np.array([uniq[h] for h in groups])
    armB = {}
    for tag, Xuse, names in [("features_only", Xev, FEAT_NAMES), ("features_plus_incumbent", Xb, NAMES_B)]:
        S = np.zeros_like(Yev)
        gkf = GroupKFold(n_splits=5)
        for tr_i, te_i in gkf.split(Xuse, groups=gidx):
            f, _ = fit_hgb(Xuse[tr_i], Yev[tr_i], A.seed)
            S[te_i] = f(Xuse[te_i])
        st, ok, _ = sel_stats(S, Yev)
        st["auroc"] = float(roc_auc_score(Yev.reshape(-1), S.reshape(-1)))
        st["ci_sel_eff"] = boot_ci(ok, orc, A.nboot, np.random.default_rng(A.seed))
        st["vs_incumbent"] = paired_boot(ok, orc, ok_inc, A.nboot, np.random.default_rng(A.seed))
        st["per_dataset"] = {}
        for ds in EVAL_SETS:
            m = np.array([i["ds"] == ds for i in ev])
            s2, o2, r2 = sel_stats(S[m], Yev[m])
            s2["vs_incumbent"] = paired_boot(o2, r2, ok_inc[m], A.nboot, np.random.default_rng(A.seed))
            st["per_dataset"][ds] = s2
        armB[tag] = st
        print(f"  {tag:24s} sel_eff={st['sel_eff']:.4f} {st['ci_sel_eff']} "
              f"d_vs_inc={st['vs_incumbent']['d_sel_eff']:+.4f} {st['vs_incumbent']['ci_sel_eff']}", flush=True)
    out["arm_B_crossfit_diagnostic"] = {
        "protocol": ("5-fold GroupKFold on the EVAL set, grouped by IMAGE PIXEL HASH (no fold shares an "
                     "image). Necessary because the incumbent adapter was never run over the 16,621 "
                     "disjoint train items. NOT A DEPLOYABLE NUMBER -- a cross-fitted diagnostic upper bound."),
        "results": armB}

    # ---------------- pair-oracle headroom between incumbent and the feature model ----------------
    okA = scorers[best][2]
    pair = np.maximum(okA, ok_inc)
    out["pair_oracle"] = {
        "definition": "per item, credit if EITHER the incumbent or the feature model picks a correct candidate",
        "sel_eff": float(pair[orc == 1].mean()),
        "headroom_over_incumbent": float(pair[orc == 1].mean() - ok_inc[orc == 1].mean()),
        "agreement_rate_on_pick": float(np.mean([np.argmax(Sinc[i]) == np.argmax(scorers[best][1][i])
                                                 for i in range(n)])),
        "both_correct": float(np.mean((okA == 1) & (ok_inc == 1))),
        "feature_only_correct": float(np.mean((okA == 1) & (ok_inc == 0))),
        "incumbent_only_correct": float(np.mean((okA == 0) & (ok_inc == 1))),
    }
    # contested stratum (>=2 distinct candidates) -- the sensitive endpoint
    contested = np.array([len(set(norm(p) for p in i["preds"])) >= 2 for i in ev])
    out["contested_stratum"] = {"definition": "items with >=2 distinct normalized candidates",
                                "n": int(contested.sum()), "frac": float(contested.mean())}
    for nm, okv in [("incumbent", ok_inc), (best, okA), ("self_consistency", ok_sc)] + \
                   ([("generator_zero_shot_pyes", ok_zs)] if ok_zs is not None else []):
        m = contested & (orc == 1)
        out["contested_stratum"][nm] = float(okv[m].mean())

    json.dump(out, open(J(A.out), "w"), indent=1)
    print(f"\nwrote {A.out}", flush=True)


if __name__ == "__main__":
    main()
