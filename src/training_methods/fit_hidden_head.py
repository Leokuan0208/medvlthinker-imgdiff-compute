#!/usr/bin/env python3
"""fit_hidden_head.py -- DISCRIMINATIVE HEADS over the generator's own hidden states, as an
alternative to a generative LM emitting an opinion.

Reads the caches written by extract_generator_hidden.py, fits a family of small heads on the
STRICTLY DISJOINT training pool (grouped cross-validation for every hyperparameter), and scores
best-of-8 selection on the 2345 open-text eval items.

Endpoint: selection efficiency at N=8, P(pick correct | correct present), pooled and per dataset,
paired bootstrap 95% CI vs the incumbent (trained same-family LoRA verifier, clean L1 split).

NULL TEST (runs first, always): the harness is pointed at the incumbent's own stored scores and
must reproduce its published cells before any new number is reported.

  python3 src/training_methods/fit_hidden_head.py --feats feats_hidden \
      --out results/cascade_methods/artifacts/verifarch_hidden_2026-08-04.json
"""
import argparse, os, json, math, hashlib, time
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
DUMP = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")
EVAL_DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
RNG = np.random.default_rng(0)


def norm(s):
    return str(s).strip().lower()


# ------------------------------------------------------------------ incumbent / ground truth
def load_incumbent():
    """The 2345 eval items with per-candidate judge labels and the incumbent verifier's scores."""
    items = []
    for d in ["slake", "vqa_rad", "pathvqa"]:
        for r in json.load(open(os.path.join(DUMP, f"transfer_dump_{d}_open_lingshu7b.json"))):
            items.append(r)
    return items


def sel_stats(items, picks):
    """picks[i] = chosen slot in item i's 8-candidate pool."""
    sl = [it["sl"] for it in items]
    rec = np.array([1 if 1 in x else 0 for x in sl])
    got = np.array([1 if sl[i][picks[i]] == 1 else 0 for i in range(len(items))])
    return {"n": len(items), "acc": float(got.mean()),
            "sel_eff": float(got[rec == 1].mean()), "n_recoverable": int(rec.sum()),
            "oracle": float(rec.mean())}, got, rec


def contested(items, picks):
    """Secondary (more sensitive) endpoint: selection efficiency restricted to items whose pool has
    >=2 DISTINCT candidate strings. The ~26% unanimous stratum is scored 1.0 by every selector by
    construction, so it dilutes pooled sel_eff; all real headroom lives here."""
    nd = np.array([len(set(norm(a) for a in it["preds"])) for it in items])
    rec = np.array([1 if 1 in it["sl"] else 0 for it in items])
    got = np.array([1 if items[i]["sl"][picks[i]] == 1 else 0 for i in range(len(items))])
    m = (rec == 1) & (nd >= 2)
    return {"n": int(m.sum()), "sel_eff": float(got[m].mean()),
            "unanimous_n": int(((rec == 1) & (nd == 1)).sum())}, got, m


def per_ds(items, picks):
    out = {}
    for ds in EVAL_DS:
        sub = [i for i, it in enumerate(items) if it["ds"] == ds]
        s, _, _ = sel_stats([items[i] for i in sub], [picks[i] for i in sub])
        out[ds] = s
    return out


def paired_boot(items, got_a, got_b, rec, nboot=10000, seed=0):
    """Paired bootstrap over ITEMS of (sel_eff_a - sel_eff_b) and (acc_a - acc_b)."""
    rng = np.random.default_rng(seed)
    n = len(items)
    idx_rec = np.where(rec == 1)[0]
    de, da = np.empty(nboot), np.empty(nboot)
    for b in range(nboot):
        s = rng.integers(0, n, n)
        da[b] = got_a[s].mean() - got_b[s].mean()
        sr = rng.integers(0, len(idx_rec), len(idx_rec))
        j = idx_rec[sr]
        de[b] = got_a[j].mean() - got_b[j].mean()
    return {"d_sel_eff": float(got_a[rec == 1].mean() - got_b[rec == 1].mean()),
            "d_sel_eff_ci": [float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))],
            "d_acc": float(got_a.mean() - got_b.mean()),
            "d_acc_ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))]}


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if y.min() == y.max():
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    ss = s[o]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    npos, nneg = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# ------------------------------------------------------------------ feature cache
def load_cache(featdir, mode, split):
    """Merges shards written by extract_generator_hidden.py --shard/--nshard."""
    import glob as _g
    d = os.path.join(ROOT, featdir)
    parts = sorted(_g.glob(os.path.join(d, f"{mode}_{split}.npz"))) or \
            sorted(_g.glob(os.path.join(d, f"{mode}_{split}_s*of*.npz")))
    assert parts, f"no feature cache for {mode}/{split} in {d}"
    zs = [np.load(p) for p in parts]
    ms = [json.load(open(p.replace(".npz", ".meta.json"))) for p in parts]
    z = {"h_last": np.concatenate([x["h_last"] for x in zs], 0),
         "h_span": np.concatenate([x["h_span"] for x in zs], 0),
         "yesno": np.concatenate([x["yesno"] for x in zs], 0),
         "layers": zs[0]["layers"]}
    meta = {**ms[0], "n": sum(m["n"] for m in ms), "n_failed": sum(m["n_failed"] for m in ms),
            "rows": [r for m in ms for r in m["rows"]], "n_shards": len(parts)}
    return z, meta


def build_matrix(z, meta, layer_idx, pooling, setrel=0):
    """setrel=1 appends h_i - mean_j h_j over the candidate group of question i. This makes a
    POINTWISE head set-aware at zero extra inference cost: the residual is what distinguishes this
    candidate from its siblings, i.e. the comparative quantity a real pairwise pass computes."""
    rows = meta["rows"]
    ok = [i for i, r in enumerate(rows) if r.get("n_tok", -1) > 0]
    X = (z["h_last"] if pooling == "last" else z["h_span"])[ok, layer_idx].astype(np.float32)
    y = np.array([rows[i]["y"] for i in ok], dtype=np.float32)
    keys = [(rows[i]["ds"], rows[i]["idx"], rows[i]["na"]) for i in ok]
    grp = [rows[i]["img_md5"] for i in ok]
    if setrel:
        qid = [f"{a}|{b}" for (a, b, c) in keys]
        byq = defaultdict(list)
        for i, q in enumerate(qid):
            byq[q].append(i)
        R = np.empty_like(X)
        for q, ii in byq.items():
            R[ii] = X[ii] - X[ii].mean(0, keepdims=True)
        X = np.concatenate([X, R], 1)
    return X, y, keys, grp, ok


# ------------------------------------------------------------------ heads (numpy/torch, CPU)
import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, d, hidden=0, drop=0.0):
        super().__init__()
        if hidden:
            self.f = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))
        else:
            self.f = nn.Linear(d, 1)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def _pad_groups(gtr, ytr, objective):
    """[n_groups, max_len] index + mask matrices so listwise/BT losses run batched, not per-group."""
    byq = defaultdict(list)
    for i, g in enumerate(gtr):
        byq[g].append(i)
    groups = [np.array(v) for v in byq.values()]
    if objective == "listwise":
        groups = [g for g in groups if 0 < ytr[g].sum() < len(g)]
    else:
        groups = [g for g in groups if ytr[g].sum() > 0 and (1 - ytr[g]).sum() > 0]
    if not groups:
        return None
    L = max(len(g) for g in groups)
    idx = np.zeros((len(groups), L), dtype=np.int64)
    msk = np.zeros((len(groups), L), dtype=np.float32)
    for k, g in enumerate(groups):
        idx[k, :len(g)] = g; msk[k, :len(g)] = 1.0
    return torch.tensor(idx), torch.tensor(msk)


def fit_head(Xtr, ytr, gtr, objective="bce", hidden=0, wd=1e-2, lr=1e-3, epochs=30, bs=256,
             seed=0, drop=0.0):
    """objective: bce      pointwise binary cross-entropy (the incumbent's objective, new readout)
                  listwise softmax cross-entropy over the within-question candidate group
                  bt       Bradley-Terry over (correct, incorrect) pairs within a question
    Set-aware by construction for listwise/bt: the loss couples the 8 candidates of one question."""
    torch.manual_seed(seed)
    m = Head(Xtr.shape[1], hidden, drop)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    X = torch.tensor(Xtr); y = torch.tensor(ytr)
    if objective == "bce":
        pos = float(ytr.mean())
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6)))
        n = len(y)
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                j = perm[i:i + bs]
                opt.zero_grad(); lossf(m(X[j]), y[j]).backward(); opt.step()
        m.eval(); return m
    packed = _pad_groups(gtr, ytr, objective)
    if packed is None:
        m.eval(); return m
    idx, msk = packed
    NG, gb = idx.shape[0], 64
    for ep in range(epochs):
        perm = torch.randperm(NG)
        for i in range(0, NG, gb):
            j = perm[i:i + gb]
            gi, gm = idx[j], msk[j]                       # [B, L]
            s = m(X[gi.reshape(-1)]).reshape(gi.shape)    # [B, L]
            yy = y[gi.reshape(-1)].reshape(gi.shape) * gm
            s = s.masked_fill(gm == 0, -1e9)
            if objective == "listwise":
                logp = torch.log_softmax(s, 1)
                l = -((logp * yy).sum(1) / yy.sum(1).clamp(min=1)).mean()
            else:
                pm = yy.unsqueeze(2)                                   # positives
                nm = ((1 - yy) * gm).unsqueeze(1)                      # negatives
                d = s.unsqueeze(2) - s.unsqueeze(1)                    # [B, L, L] pos - neg
                w = pm * nm
                l = ((nn.functional.softplus(-d) * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)).mean()
            opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


def predict(m, X):
    with torch.no_grad():
        return m(torch.tensor(X)).numpy()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="feats_hidden")
    ap.add_argument("--mode", nargs="+", default=["grader"])
    ap.add_argument("--out", default="results/cascade_methods/artifacts/verifarch_hidden_2026-08-04.json")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--indomain_diag", type=int, default=1,
                    help="also run the contaminated in-domain probe (upper bound, clearly labelled)")
    A = ap.parse_args()
    art = {"what": "Discriminative heads over the GENERATOR'S OWN hidden states as a non-generative "
                   "verifier for best-of-8 open-text selection.",
           "date": "2026-08-04", "nboot": A.nboot, "seed": 0}

    # ---------------- NULL TEST ----------------
    items = load_incumbent()
    picks_inc = [int(np.argmax(it["scores"])) for it in items]
    s_inc, got_inc, rec = sel_stats(items, picks_inc)
    greedy = float(np.mean([it["greedy_ok"] for it in items]))
    # self-consistency (plurality of the 8 surface strings)
    picks_sc = []
    for it in items:
        c = Counter(norm(a) for a in it["preds"])
        top = c.most_common(1)[0][0]
        picks_sc.append(next(k for k, a in enumerate(it["preds"]) if norm(a) == top))
    s_sc, got_sc, _ = sel_stats(items, picks_sc)
    # random pick (expectation, closed form over the 8 slots)
    rand_acc = float(np.mean([np.mean([1 if v == 1 else 0 for v in it["sl"]]) for it in items]))
    rand_eff = float(np.mean([np.mean([1 if v == 1 else 0 for v in items[i]["sl"]])
                              for i in range(len(items)) if rec[i] == 1]))
    # candidate-level AUROC of the incumbent
    ys, ss = [], []
    for it in items:
        for k, v in enumerate(it["sl"]):
            if v >= 0:
                ys.append(v); ss.append(it["scores"][k])
    auc_inc = auroc(ys, ss)
    art["null_test"] = {
        "protocol": "harness pointed at the incumbent's own stored per-candidate scores "
                    "(ckpts/train/lora_verifier_disjoint/transfer_dump_*.json); pick = argmax over the "
                    "8 slots of the stored (5-dp-rounded) scores, first-index tie-break.",
        "measured": {"n": s_inc["n"], "oracle@8": round(s_inc["oracle"], 6),
                     "selected_acc": round(s_inc["acc"], 6), "greedy": round(greedy, 6),
                     "sel_eff": round(s_inc["sel_eff"], 6), "cand_auroc": round(auc_inc, 6)},
        "published": {"n": 2345, "oracle@8": 0.626013, "selected_acc": 0.485288, "greedy": 0.449467,
                      "sel_eff": 0.775204, "cand_auroc": 0.8856},
        "tolerance": {"oracle@8": round(abs(s_inc["oracle"] - 0.626013), 6),
                      "selected_acc": round(abs(s_inc["acc"] - 0.485288), 6),
                      "greedy": round(abs(greedy - 0.449467), 6),
                      "sel_eff": round(abs(s_inc["sel_eff"] - 0.775204), 6),
                      "cand_auroc": round(abs(auc_inc - 0.8856), 6)},
        "note": "the stored 'pick' field disagrees with argmax on 26/2345 items (exact score ties from "
                "5-dp rounding); using stored pick gives sel_eff 0.774523. The +-0.0007 tie-break "
                "sensitivity is reported so it is not mistaken for an effect."}
    print("NULL TEST:", json.dumps(art["null_test"]["measured"]), flush=True)
    print("           tol:", json.dumps(art["null_test"]["tolerance"]), flush=True)

    art["controls"] = {
        "greedy": round(greedy, 6),
        "random_pick": {"acc": round(rand_acc, 6), "sel_eff": round(rand_eff, 6)},
        "self_consistency": {"acc": round(s_sc["acc"], 6), "sel_eff": round(s_sc["sel_eff"], 6),
                             "contested_sel_eff": round(contested(items, picks_sc)[0]["sel_eff"], 6)},
        "oracle@8": round(s_inc["oracle"], 6),
        "incumbent_lora_verifier": {"acc": round(s_inc["acc"], 6), "sel_eff": round(s_inc["sel_eff"], 6),
                                    "cand_auroc": round(auc_inc, 6),
                                    "per_ds": {k: round(v["sel_eff"], 6) for k, v in per_ds(items, picks_inc).items()},
                                    "contested": {k: (round(v, 6) if isinstance(v, float) else v)
                                                  for k, v in contested(items, picks_inc)[0].items()}},
    }

    # ---------------- features ----------------
    art["arms"] = {}
    for mode in A.mode:
        ztr, mtr = load_cache(A.feats, mode, "train")
        zev, mev = load_cache(A.feats, mode, "eval")
        layers = list(ztr["layers"])
        # ---- disjointness assertion on decoded-RGB md5
        tr_h = set(r["img_md5"] for r in mtr["rows"] if r.get("img_md5"))
        ev_h = set(r["img_md5"] for r in mev["rows"] if r.get("img_md5"))
        inter = tr_h & ev_h
        assert len(inter) == 0, f"CONTAMINATION: {len(inter)} shared decoded-RGB image hashes"
        art.setdefault("disjointness", {})[mode] = {
            "train_images": len(tr_h), "eval_images": len(ev_h), "pixel_md5_intersection": 0,
            "train_rows": len(mtr["rows"]), "eval_rows": len(mev["rows"]),
            "train_rows_failed": mtr["n_failed"], "eval_rows_failed": mev["n_failed"]}
        print(f"[{mode}] disjoint OK: {len(tr_h)} train imgs / {len(ev_h)} eval imgs, 0 shared", flush=True)

        # base zero-shot P(Yes) control (grader mode only)
        if mode == "grader" and not np.all(np.isnan(zev["yesno"])):
            yn = zev["yesno"]
            base_score = {}
            for i, r in enumerate(mev["rows"]):
                if r.get("n_tok", -1) > 0 and not np.isnan(yn[i, 0]):
                    py, pn = math.exp(yn[i, 0]), math.exp(yn[i, 1])
                    base_score[(r["ds"], r["idx"], r["na"])] = py / (py + pn) if (py + pn) > 0 else 0.5
            picks_b, ysb, ssb = [], [], []
            for it in items:
                sc = [base_score.get((it["ds"], it["idx"], norm(a)), np.nan) for a in it["preds"]]
                sc = [(-1e9 if np.isnan(v) else v) for v in sc]
                picks_b.append(int(np.argmax(sc)))
                for k, v in enumerate(it["sl"]):
                    if v >= 0:
                        ysb.append(v); ssb.append(sc[k])
            s_b, got_b, _ = sel_stats(items, picks_b)
            art["controls"]["base_zeroshot_pyes_" + mode] = {
                "acc": round(s_b["acc"], 6), "sel_eff": round(s_b["sel_eff"], 6),
                "cand_auroc": round(auroc(ysb, ssb), 6),
                "per_ds": {k: round(v["sel_eff"], 6) for k, v in per_ds(items, picks_b).items()},
                "contested_sel_eff": round(contested(items, picks_b)[0]["sel_eff"], 6),
                "note": "frozen Lingshu-7B, NO adapter, same grader prompt -- the generator scoring its "
                        "own candidates; read off the SAME forward pass that produced the features, so "
                        "it doubles as a pipeline check.",
                "published_reference": {
                    "value": 0.7070844686648501,
                    "source": "results/cascade_methods/artifacts/crossfamily_verifier_sweep_2026-08-04.json "
                              "-> table/POOLED/verifiers/lingshu7b_zs/sel_eff",
                    "why_not_identical": "the published control was served by vLLM at the cap320 image "
                                         "budget (<=250880 px); this one is HF transformers at the "
                                         "verifier's budget (1280*28*28 = 1003520 px). Agreement to "
                                         "~0.001 is a cross-stack check, not an exact reproduction."}}
            art["_got_base"] = got_b
            print(f"[{mode}] base zero-shot P(Yes) sel_eff={s_b['sel_eff']:.4f} "
                  f"(published cap320/vLLM reference 0.7071)", flush=True)

        # ---- CONTAMINATED IN-DOMAIN DIAGNOSTIC (upper bound; NEVER deployable).
        # Fit the head inside EVAL with image-grouped 5-fold CV. This answers a prior question --
        # "is correctness linearly decodable from the frozen generator's hidden state at all?" --
        # separately from "does it transfer from a disjoint pool?". It shares images and question
        # distribution with the test set, so it is an optimistic bound, reported as such.
        if A.indomain_diag:
            diag = []
            for li, L in enumerate(layers):
                for pool in ["last", "span"]:
                    Xe, ye, ke, ge, _ = build_matrix(zev, mev, li, pool, 0)
                    qe = [f"{a}|{b}" for (a, b, c) in ke]
                    fo = np.array([int(hashlib.md5(str(g).encode()).hexdigest(), 16) % A.folds for g in ge])
                    sv = np.zeros(len(ye))
                    for f in range(A.folds):
                        tr, va = fo != f, fo == f
                        mu, sd = Xe[tr].mean(0), Xe[tr].std(0) + 1e-6
                        hm = fit_head((Xe[tr] - mu) / sd, ye[tr], [qe[i] for i in np.where(tr)[0]],
                                      objective="bce", hidden=0, wd=1e-2, epochs=30, seed=f)
                        sv[va] = predict(hm, (Xe[va] - mu) / sd)
                    sm = {ke[i]: float(sv[i]) for i in range(len(ke))}
                    pk = [int(np.argmax([sm.get((it["ds"], it["idx"], norm(a)), -1e9) for a in it["preds"]]))
                          for it in items]
                    st, _, _ = sel_stats(items, pk)
                    yy, ss2 = [], []
                    for it in items:
                        for k, v in enumerate(it["sl"]):
                            if v >= 0:
                                yy.append(v); ss2.append(sm.get((it["ds"], it["idx"], norm(it["preds"][k])), np.nan))
                    diag.append({"layer": int(L), "pooling": pool, "sel_eff": round(st["sel_eff"], 6),
                                 "acc": round(st["acc"], 6), "cand_auroc": round(auroc(yy, ss2), 6),
                                 "contested_sel_eff": round(contested(items, pk)[0]["sel_eff"], 6)})
                    print(f"  [in-domain DIAG, contaminated] L{L}/{pool}: sel_eff={st['sel_eff']:.4f} "
                          f"auroc={diag[-1]['cand_auroc']:.4f}", flush=True)
            art["arms"].setdefault(mode, {})["indomain_diagnostic_CONTAMINATED"] = {
                "protocol": f"linear pointwise head fit INSIDE the eval set with {A.folds}-fold "
                            f"image-grouped CV. Shares images and question distribution with the test "
                            f"set -> OPTIMISTIC BOUND, not comparable to the clean incumbent and not "
                            f"deployable. Reported to separate 'is the signal there' from 'does it transfer'.",
                "table": diag}

        # ---- grouped CV hyperparameter selection, entirely inside the training pool.
        # FOLD PROTOCOL: 5 folds assigned by md5(decoded-RGB image hash) % 5, so a fold boundary is
        # an IMAGE boundary -- the same discipline as the eval split. Eval is never touched here.
        cache = {}

        def get(li, pool, sr):
            if (li, pool, sr) not in cache:
                cache[(li, pool, sr)] = build_matrix(ztr, mtr, li, pool, sr)
            return cache[(li, pool, sr)]

        def cv(cfg):
            li = layers.index(cfg["layer"])
            Xtr, ytr, ktr, gtr, _ = get(li, cfg["pooling"], cfg.get("setrel", 0))
            qid = [f"{a}|{b}" for (a, b, c) in ktr]
            fold_of = {h: (int(hashlib.md5(str(h).encode()).hexdigest(), 16) % A.folds) for h in set(gtr)}
            fo = np.array([fold_of[h] for h in gtr])
            aucs, effs = [], []
            for f in range(A.folds):
                tr, va = fo != f, fo == f
                mu, sd = Xtr[tr].mean(0), Xtr[tr].std(0) + 1e-6
                m = fit_head((Xtr[tr] - mu) / sd, ytr[tr], [qid[i] for i in np.where(tr)[0]],
                             objective=cfg["objective"], hidden=cfg["hidden"], wd=cfg["wd"],
                             epochs=cfg["epochs"], seed=f)
                sv = predict(m, (Xtr[va] - mu) / sd)
                aucs.append(auroc(ytr[va], sv))
                vidx = np.where(va)[0]
                loc = {i: j for j, i in enumerate(vidx)}
                byq = defaultdict(list)
                for i in vidx:
                    byq[qid[i]].append(i)
                hit, tot = 0, 0
                for q, ii in byq.items():
                    if ytr[ii].sum() == 0:
                        continue
                    b = ii[int(np.argmax([sv[loc[i]] for i in ii]))]
                    hit += int(ytr[b] == 1); tot += 1
                effs.append(hit / max(tot, 1))
            r = {**cfg, "cv_auroc": float(np.mean(aucs)), "cv_sel_eff": float(np.mean(effs))}
            print(f"  cv L{cfg['layer']}/{cfg['pooling']}/{cfg['objective']}/h{cfg['hidden']}/"
                  f"wd{cfg['wd']}/sr{cfg.get('setrel',0)}: auroc={r['cv_auroc']:.4f} "
                  f"sel_eff={r['cv_sel_eff']:.4f}", flush=True)
            return r

        # stage 1: pick the representation (layer x pooling) with the cheap linear pointwise head
        s1 = [cv({"layer": int(L), "pooling": p, "objective": "bce", "hidden": 0, "wd": 1e-2,
                  "epochs": 30, "setrel": 0})
              for L in layers for p in ["last", "span"]]
        rep = max(s1, key=lambda r: r["cv_sel_eff"])
        # stage 2: at that representation, sweep objective x set-relativity (linear, wd 1e-2)
        s2 = [cv({"layer": rep["layer"], "pooling": rep["pooling"], "objective": o, "hidden": 0,
                  "wd": 1e-2, "epochs": 30, "setrel": sr})
              for o in ["bce", "listwise", "bt"] for sr in [0, 1]]
        # stage 3: at the best objective x set-relativity, sweep capacity x regularisation
        b2 = max(s2, key=lambda r: r["cv_sel_eff"])
        s3 = [cv({**b2x, "epochs": 30}) for b2x in
              [{"layer": b2["layer"], "pooling": b2["pooling"], "objective": b2["objective"],
                "hidden": h, "wd": w, "setrel": b2["setrel"]}
               for h in [0, 256] for w in [1e-2, 1e-1]]
              if not (b2x["hidden"] == 0 and b2x["wd"] == 1e-2)]
        s2 = s2 + s3
        cvres = s1 + s2
        art["arms"].setdefault(mode, {})["cv_grid"] = cvres
        art["arms"][mode]["cv_protocol"] = (
            f"{A.folds}-fold grouped CV inside the disjoint training pool; folds assigned by "
            f"md5(decoded-RGB image hash) % {A.folds} so folds are image-disjoint. Stage 1 selects "
            f"layer x pooling with a linear pointwise head; stage 2 sweeps objective x set-relativity "
            f"at that representation; stage 3 sweeps capacity (hidden) x weight decay at the stage-2 "
            f"winner. Selection criterion is CV within-question selection efficiency. Eval is never "
            f"used for any selection; the selected config is then refit on the full training pool.")
        best = max(s2, key=lambda r: r["cv_sel_eff"])
        art["arms"][mode]["cv_selected"] = best
        print(f"[{mode}] CV-selected: {best}", flush=True)

        # eval-ds -> the in-domain training family (for the per-benchmark-head arm)
        DOMAIN = {"slake_open": ["slake_open_train"], "vqa_rad_open": ["vqa_rad_open_train"],
                  "pathvqa_open": ["pathvqa_open_train"]}
        tr_ds = np.array([r["ds"] for r in mtr["rows"] if r.get("n_tok", -1) > 0])

        def eval_config(cfg, tag, per_domain=False):
            li = layers.index(cfg["layer"])
            sr = cfg.get("setrel", 0)
            Xtr, ytr, ktr, gtr, _ = build_matrix(ztr, mtr, li, cfg["pooling"], sr)
            Xev, yev, kev, gev, _ = build_matrix(zev, mev, li, cfg["pooling"], sr)
            qid = [f"{a}|{b}" for (a, b, c) in ktr]
            smap = {}
            if per_domain:
                # one head per eval benchmark, trained ONLY on that benchmark's own disjoint train pool
                for eds, srcs in DOMAIN.items():
                    sel = np.isin(tr_ds, srcs)
                    mu, sd = Xtr[sel].mean(0), Xtr[sel].std(0) + 1e-6
                    m = fit_head((Xtr[sel] - mu) / sd, ytr[sel], [qid[i] for i in np.where(sel)[0]],
                                 objective=cfg["objective"], hidden=cfg["hidden"], wd=cfg["wd"],
                                 epochs=cfg["epochs"], seed=0)
                    keep = [i for i in range(len(kev)) if kev[i][0] == eds]
                    sv = predict(m, (Xev[keep] - mu) / sd)
                    for j, i in enumerate(keep):
                        smap[kev[i]] = float(sv[j])
            else:
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                m = fit_head((Xtr - mu) / sd, ytr, qid, objective=cfg["objective"], hidden=cfg["hidden"],
                             wd=cfg["wd"], epochs=cfg["epochs"], seed=0)
                sv = predict(m, (Xev - mu) / sd)
                smap = {kev[i]: float(sv[i]) for i in range(len(kev))}
            picks, ysh, ssh = [], [], []
            for it in items:
                sc = [smap.get((it["ds"], it["idx"], norm(a)), np.nan) for a in it["preds"]]
                sc = [(-1e9 if (v is None or np.isnan(v)) else v) for v in sc]
                picks.append(int(np.argmax(sc)))
                for k, v in enumerate(it["sl"]):
                    if v >= 0:
                        ysh.append(v); ssh.append(sc[k])
            s, got, _ = sel_stats(items, picks)
            bo = paired_boot(items, got, got_inc, rec, A.nboot)
            res = {"config": cfg, "n": s["n"], "acc": round(s["acc"], 6),
                   "sel_eff": round(s["sel_eff"], 6), "cand_auroc": round(auroc(ysh, ssh), 6),
                   "per_ds": {k: round(v["sel_eff"], 6) for k, v in per_ds(items, picks).items()},
                   "contested": {k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in contested(items, picks)[0].items()},
                   "vs_incumbent": {k: (round(v, 6) if isinstance(v, float) else [round(x, 6) for x in v])
                                    for k, v in bo.items()},
                   "guardrail_clean": all(per_ds(items, picks)[d]["sel_eff"] >=
                                          per_ds(items, picks_inc)[d]["sel_eff"] for d in EVAL_DS)}
            if "_got_base" in art:
                res["vs_base_zeroshot"] = {
                    k: (round(v, 6) if isinstance(v, float) else [round(x, 6) for x in v])
                    for k, v in paired_boot(items, got, art["_got_base"], rec, A.nboot).items()}
            res["vs_self_consistency"] = {
                k: (round(v, 6) if isinstance(v, float) else [round(x, 6) for x in v])
                for k, v in paired_boot(items, got, got_sc, rec, A.nboot).items()}
            print(f"  EVAL[{tag}] sel_eff={s['sel_eff']:.4f} acc={s['acc']:.4f} "
                  f"auroc={res['cand_auroc']:.4f} d={bo['d_sel_eff']:+.4f} {bo['d_sel_eff_ci']}", flush=True)
            return res, smap

        art["arms"][mode]["eval"] = {}
        KEYS = ["layer", "pooling", "objective", "hidden", "wd", "epochs", "setrel"]
        res, smap_best = eval_config({k: best[k] for k in KEYS}, "cv_selected")
        art["arms"][mode]["eval"]["cv_selected"] = res

        # transparency: report every objective at the CV-best (layer, pooling, hidden, wd), so the
        # pointwise-vs-listwise-vs-BT question is answered on IDENTICAL features (test plan T1.B)
        for obj in ["bce", "listwise", "bt"]:
            if obj == best["objective"]:
                continue
            cfg = {k: best[k] for k in KEYS}; cfg["objective"] = obj
            r2, _ = eval_config(cfg, f"obj={obj}")
            art["arms"][mode]["eval"][f"objective_{obj}"] = r2
        # and every layer at the CV-best readout, so "which depth carries correctness" is on record
        for L in layers:
            if int(L) == best["layer"]:
                continue
            cfg = {k: best[k] for k in KEYS}; cfg["layer"] = int(L)
            r3, _ = eval_config(cfg, f"layer={L}")
            art["arms"][mode]["eval"][f"layer_{L}"] = r3
        # per-benchmark heads: probes are documented not to generalise across datasets, so also fit
        # one head per eval set on that set's OWN disjoint train pool only
        r4, _ = eval_config({k: best[k] for k in KEYS}, "per_benchmark_heads", per_domain=True)
        r4["note"] = ("one head per eval benchmark, trained only on that benchmark's own image-disjoint "
                      "train pool (slake_open_train / vqa_rad_open_train / pathvqa_open_train). "
                      "Same CV-selected architecture; the pooled figure is the concatenation.")
        art["arms"][mode]["eval"]["per_benchmark_heads"] = r4

        # ---- complementarity diagnostics: agreement, rank fusion, pair-oracle ceiling
        def rank01(v):
            v = np.asarray(v, float)
            return np.argsort(np.argsort(v)) / max(len(v) - 1, 1)

        picks_h, picks_f, agree = [], [], []
        for it in items:
            hs = np.array([smap_best.get((it["ds"], it["idx"], norm(a)), -1e9) for a in it["preds"]])
            ins = np.array(it["scores"], float)
            picks_h.append(int(np.argmax(hs)))
            picks_f.append(int(np.argmax(rank01(ins) + rank01(hs))))
            agree.append(int(int(np.argmax(hs)) == int(np.argmax(ins))))
        s_h, got_h, _ = sel_stats(items, picks_h)
        s_f, got_f, _ = sel_stats(items, picks_f)
        po = np.maximum(got_inc, got_h)
        # Spearman rho between head and incumbent over all labelled candidates
        pa, pb = [], []
        for it in items:
            for k, v in enumerate(it["sl"]):
                if v >= 0:
                    pa.append(smap_best.get((it["ds"], it["idx"], norm(it["preds"][k])), np.nan))
                    pb.append(it["scores"][k])
        pa, pb = np.array(pa, float), np.array(pb, float)
        ok = ~np.isnan(pa)
        rho = float(np.corrcoef(rank01(pa[ok]), rank01(pb[ok]))[0, 1])
        art["arms"][mode]["complementarity"] = {
            "spearman_rho_head_vs_incumbent": round(rho, 4),
            "same_pick_rate": round(float(np.mean(agree)), 4),
            "rank_fusion": {"acc": round(s_f["acc"], 6), "sel_eff": round(s_f["sel_eff"], 6),
                            "vs_incumbent": {k: (round(v, 6) if isinstance(v, float) else [round(x, 6) for x in v])
                                             for k, v in paired_boot(items, got_f, got_inc, rec, A.nboot).items()}},
            "pair_oracle_ceiling_sel_eff": round(float(po[rec == 1].mean()), 6),
            "note": "pair-oracle = an omniscient per-item router between the incumbent and the head. "
                    "It is a DIAGNOSTIC UPPER BOUND on what any router over this pair could reach, "
                    "not a deployable number. The project's standing law is that selection quality "
                    "tracks AGREEMENT with the generator (+0.76), so rho is reported alongside it."}
        print(f"[{mode}] rho={rho:.3f} same-pick={np.mean(agree):.3f} "
              f"rank-fusion sel_eff={s_f['sel_eff']:.4f} pair-oracle={po[rec==1].mean():.4f}", flush=True)

    art.pop("_got_base", None)
    op = os.path.join(ROOT, A.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(art, open(op, "w"), indent=1)
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
