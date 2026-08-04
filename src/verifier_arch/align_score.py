#!/usr/bin/env python3
"""align_score.py -- the CONTRASTIVE IMAGE-TEXT ALIGNMENT verifier for best-of-8 selection, plus the
control battery that distinguishes *surface relevance* from *correctness*.

Selectors evaluated on the 2345 eval items (slake_open 645 / vqa_rad_open 200 / pathvqa_open 1500):
  zero-shot alignment   s = cos(img, text)                          [raw and PMI-normalised]
  trained alignment head  MLP on frozen [v, t, v*t, |v-t|], BCE on the image-DISJOINT train pools
  text-only head        identical MLP with the image half zeroed    <- the relevance/correctness control
  fusion with the incumbent verifier (cross-fitted on eval -> DIAGNOSTIC ONLY, never deployable)

Control battery (reported separately from the headline, per T1.D):
  (a) on-topic correctness AUROC/pairwise: correct vs incorrect candidates from the SAME pool
      (same question, same image, same generator => equally on-topic and equally fluent)
  (b) off-topic relevance pairwise: this item's candidate vs a candidate stolen from another item
  (c) image-permutation null: rank correlation + metric drop when the image is swapped for a random one
  (d) laterality / negation slices; (e) surface confounds (length, pool frequency)

  python3 src/verifier_arch/align_score.py --encoder siglip
  -> results/cascade_methods/artifacts/verifarch_alignment_2026-08-04.json (written by --all)
Run from the repo root.
"""
import os, json, math, argparse, hashlib, re
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CACHE = os.path.join(ROOT, "data/align_cache")
RNG = np.random.default_rng(0)

ap = argparse.ArgumentParser()
ap.add_argument("--encoders", nargs="+", default=["siglip", "pubmedclip", "biomedclip"])
ap.add_argument("--out", default="results/cascade_methods/artifacts/verifarch_alignment_2026-08-04.json")
ap.add_argument("--boot", type=int, default=2000)
A = ap.parse_args()

man = json.load(open(os.path.join(CACHE, "manifest.json")))
EVAL, TRAIN = man["eval"], man["train"]
TEMPLATES = {"ans": lambda q, a: a, "qa": lambda q, a: f"{q} {a}",
             "decl": lambda q, a: f"medical image. {q} answer: {a}"}


def norm(s):
    return str(s).strip().lower()


# ---------------------------------------------------------------- metrics
def sel_eff(rows, scores):
    """P(pick correct | correct present). scores: list of per-candidate score arrays."""
    keep = [(r, s) for r, s in zip(rows, scores) if max(r["sl"]) == 1]
    return float(np.mean([r["sl"][int(np.argmax(s))] == 1 for r, s in keep])), len(keep)


def sel_acc(rows, scores):
    return float(np.mean([max(r["sl"][int(np.argmax(s))], 0) for r, s in zip(rows, scores)]))


def cand_auroc(rows, scores):
    y, s = [], []
    for r, sc in zip(rows, scores):
        for lab, v in zip(r["sl"], sc):
            if lab >= 0:
                y.append(lab); s.append(v)
    y, s = np.array(y), np.array(s)
    if len(set(y.tolist())) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def ontopic_pairwise(rows, scores):
    """P(score(correct) > score(incorrect)) over DISTINCT candidate strings within the same pool.
    Ties count 0.5.  This is the 'equally on-topic, equally fluent' separation."""
    num = den = 0.0
    for r, sc in zip(rows, scores):
        seen = {}
        for a, lab, v in zip(r["preds"], r["sl"], sc):
            if lab >= 0:
                seen[norm(a)] = (lab, v)
        pos = [v for lab, v in seen.values() if lab == 1]
        neg = [v for lab, v in seen.values() if lab == 0]
        for p in pos:
            for n in neg:
                num += 1.0 if p > n else (0.5 if p == n else 0.0)
                den += 1
    return float(num / den) if den else float("nan"), int(den)


def boot_paired(rows, sA, sB, B=2000, seed=0):
    """paired bootstrap over ITEMS of (sel_eff(A) - sel_eff(B)) restricted to recoverable items."""
    idx = [i for i, r in enumerate(rows) if max(r["sl"]) == 1]
    a = np.array([rows[i]["sl"][int(np.argmax(sA[i]))] == 1 for i in idx], float)
    b = np.array([rows[i]["sl"][int(np.argmax(sB[i]))] == 1 for i in idx], float)
    rng = np.random.default_rng(seed); n = len(idx); d = []
    for _ in range(B):
        k = rng.integers(0, n, n); d.append(a[k].mean() - b[k].mean())
    d = np.array(d)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_ci(rows, s, B=2000, seed=0):
    idx = [i for i, r in enumerate(rows) if max(r["sl"]) == 1]
    a = np.array([rows[i]["sl"][int(np.argmax(s[i]))] == 1 for i in idx], float)
    rng = np.random.default_rng(seed); n = len(idx); d = []
    for _ in range(B):
        d.append(a[rng.integers(0, n, n)].mean())
    return float(a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- embedding access
class Emb:
    def __init__(self, name):
        z = np.load(os.path.join(CACHE, f"emb_{name}.npz"))
        self.ih = {h: i for i, h in enumerate(z["img_hash"])}
        self.tk = {k: i for i, k in enumerate(z["txt_key"])}
        V = z["img_emb"].astype(np.float64); T = z["txt_emb"].astype(np.float64)
        self.V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        self.T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        self.dim = self.V.shape[1]

    def v(self, h):
        return self.V[self.ih[h]]

    def t(self, tmpl, q, a):
        return self.T[self.tk[tmpl + "\x00" + TEMPLATES[tmpl](q, a)[:400]]]


def item_scores(E, rows, tmpl, pmi_bank=None, img_override=None):
    """cos(img, text) per candidate; if pmi_bank given, subtract the text's mean alignment to a bank
    of random images (PMI normalisation, T1.D step d)."""
    out = []
    for n, r in enumerate(rows):
        h = r["img"] if img_override is None else img_override[n]
        v = E.v(h)
        Tm = np.stack([E.t(tmpl, r["q"], a) for a in r["preds"]])
        s = Tm @ v
        if pmi_bank is not None:
            s = s - (Tm @ pmi_bank[r["ds"]].T).mean(1)
        out.append(s)
    return out


# ---------------------------------------------------------------- trained head
def feats(E, rows, tmpl, blind=False, img_override=None):
    X, Y, G, ITEM = [], [], [], []
    for n, r in enumerate(rows):
        h = r["img"] if img_override is None else img_override[n]
        v = E.v(h) * (0.0 if blind else 1.0)
        for a, lab in zip(r["preds"], r["sl"]):
            t = E.t(tmpl, r["q"], a)
            X.append(np.concatenate([v, t, v * t, np.abs(v - t)]))
            Y.append(lab); G.append(r["img"]); ITEM.append(n)
    return np.asarray(X, np.float32), np.asarray(Y), np.asarray(G), np.asarray(ITEM)


def train_mlp(X, Y, hid=256, epochs=12, lr=1e-3, wd=1e-4, seed=0, Xv=None, Yv=None, verbose=False):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = nn.Sequential(nn.Linear(X.shape[1], hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, 1)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, device=dev); Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    n = len(Xt); bs = 512
    best, bstate = -1, None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            k = perm[i:i + bs]
            opt.zero_grad(); l = lossf(m(Xt[k]).squeeze(-1), Yt[k]); l.backward(); opt.step()
        if Xv is not None:
            from sklearn.metrics import roc_auc_score
            p = predict(m, Xv)
            au = roc_auc_score(Yv, p)
            if verbose: print(f"    ep{ep} val_auroc={au:.4f}", flush=True)
            if au > best:
                best = au; bstate = {k: v.detach().clone() for k, v in m.state_dict().items()}
    if bstate is not None:
        m.load_state_dict(bstate)
    return m, best


def predict(m, X):
    import torch
    dev = next(m.parameters()).device
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            out.append(torch.sigmoid(m(torch.tensor(X[i:i + 4096], device=dev)).squeeze(-1)).cpu().numpy())
    return np.concatenate(out)


def to_item_scores(p, ITEM, nitems):
    out = [[] for _ in range(nitems)]
    for v, i in zip(p, ITEM):
        out[i].append(v)
    return [np.array(x) for x in out]


# ---------------------------------------------------------------- go
res = {"meta": {"date": "2026-08-04", "n_eval": len(EVAL), "n_train": len(TRAIN),
                "eval_sets": dict(Counter(r["ds"] for r in EVAL)),
                "train_sets": dict(Counter(r["ds"] for r in TRAIN))}}

# ---- NULL TEST: reproduce the incumbent's published cells from its own dumps
inc = [np.array(r["incumbent"], float) for r in EVAL]
e, ne = sel_eff(EVAL, inc)
res["null_test_incumbent"] = {
    "n": len(EVAL), "n_recoverable": ne,
    "oracle@8": float(np.mean([max(r["sl"]) for r in EVAL])),
    "selected_acc": sel_acc(EVAL, inc),
    "greedy": float(np.mean([r["greedy_ok"] for r in EVAL])),
    "sel_eff": e, "cand_auroc": cand_auroc(EVAL, inc),
    "published": {"oracle@8": 0.6260, "selected_acc": 0.485288, "greedy": 0.449467,
                  "sel_eff": 0.7752, "cand_auroc": 0.8856},
}
per = {}
for ds in sorted(set(r["ds"] for r in EVAL)):
    rs = [r for r in EVAL if r["ds"] == ds]; ss = [s for r, s in zip(EVAL, inc) if r["ds"] == ds]
    per[ds] = sel_eff(rs, ss)[0]
res["null_test_incumbent"]["sel_eff_per_set"] = per
print(json.dumps(res["null_test_incumbent"], indent=1), flush=True)

# ---- trivial controls
rand_eff = float(np.mean([sum(1 for x in r["sl"] if x == 1) / len(r["sl"]) for r in EVAL if max(r["sl"]) == 1]))
sc_eff = []
for r in EVAL:
    if max(r["sl"]) != 1: continue
    c = Counter(norm(a) for a in r["preds"]); top = c.most_common(1)[0][0]
    lab = {norm(a): l for a, l in zip(r["preds"], r["sl"])}[top]
    sc_eff.append(lab == 1)
res["controls"] = {"random_pick_sel_eff": rand_eff, "self_consistency_sel_eff": float(np.mean(sc_eff)),
                   "greedy_acc": float(np.mean([r["greedy_ok"] for r in EVAL])),
                   "oracle@8": float(np.mean([max(r["sl"]) for r in EVAL])),
                   "generator_zeroshot_pyes_sel_eff_published": 0.7071}

# generator's own zero-shot P(Yes), per item, from ckpts/openvqa/crossfam_verifier (same 2345 items)
ZS = {}
for short, ds in zip(["slake", "vqa_rad", "pathvqa"], ["slake_open", "vqa_rad_open", "pathvqa_open"]):
    p = os.path.join(ROOT, f"ckpts/openvqa/crossfam_verifier/ckpt_{ds}_lingshu7b_zs.jsonl")
    for l in open(p):
        r = json.loads(l); ZS[(ds, r["idx"])] = r["scores_by_answer"]
zs_scores, zs_ok = [], True
for r in EVAL:
    d = ZS.get((r["ds"], r["idx"]))
    if d is None:
        zs_ok = False; zs_scores.append(np.zeros(len(r["preds"]))); continue
    zs_scores.append(np.array([d.get(norm(a), 0.5) for a in r["preds"]]))
res["controls"]["generator_zeroshot_pyes_sel_eff_measured"] = sel_eff(EVAL, zs_scores)[0]
res["controls"]["generator_zeroshot_all_items_found"] = zs_ok

# contested stratum (n_distinct >= 2): the sensitive endpoint -- the unanimous stratum is scored 1.0
# by every selector by construction and only dilutes the pooled number.
CONT = [i for i, r in enumerate(EVAL) if (r["n_distinct"] or 1) >= 2]
res["controls"]["contested_stratum"] = {
    "n": len(CONT),
    "incumbent_sel_eff": sel_eff([EVAL[i] for i in CONT], [inc[i] for i in CONT])[0],
    "self_consistency_sel_eff": float(np.mean(
        [(lambda r: {norm(a): l for a, l in zip(r["preds"], r["sl"])}[
            Counter(norm(a) for a in r["preds"]).most_common(1)[0][0]] == 1)(EVAL[i])
         for i in CONT if max(EVAL[i]["sl"]) == 1])),
    "generator_zeroshot_sel_eff": sel_eff([EVAL[i] for i in CONT], [zs_scores[i] for i in CONT])[0],
}

res["encoders"] = {}
for name in A.encoders:
    p = os.path.join(CACHE, f"emb_{name}.npz")
    if not os.path.exists(p):
        print(f"!! no embeddings for {name}, skipping", flush=True); continue
    print(f"\n===== {name} =====", flush=True)
    E = Emb(name)
    R = {"dim": E.dim}

    # PMI bank: 20 random images per dataset, drawn from the EVAL pool of that dataset
    bank = {}
    for ds in set(r["ds"] for r in EVAL):
        hs = sorted({r["img"] for r in EVAL if r["ds"] == ds})
        pick = list(np.random.default_rng(1).choice(len(hs), size=min(20, len(hs)), replace=False))
        bank[ds] = np.stack([E.v(hs[i]) for i in pick])

    # ---------------- template choice on TRAIN ONLY
    tr_sel = {}
    for tmpl in TEMPLATES:
        s = item_scores(E, TRAIN, tmpl)
        tr_sel[tmpl] = sel_eff(TRAIN, s)[0]
    best_tmpl = max(tr_sel, key=tr_sel.get)
    R["template_selection_on_train"] = {k: round(v, 4) for k, v in tr_sel.items()}
    R["template_used"] = best_tmpl
    print(f"  template (chosen on TRAIN): {best_tmpl}  {R['template_selection_on_train']}", flush=True)

    # ---------------- zero-shot alignment, raw + PMI
    R["zero_shot"] = {}
    for tag, pb in (("raw", None), ("pmi", bank)):
        s = item_scores(E, EVAL, best_tmpl, pmi_bank=pb)
        eff, lo, hi = boot_ci(EVAL, s, A.boot)
        d, dlo, dhi = boot_paired(EVAL, s, inc, A.boot)
        ot, notp = ontopic_pairwise(EVAL, s)
        R["zero_shot"][tag] = {
            "sel_eff": eff, "sel_eff_ci95": [lo, hi], "sel_acc": sel_acc(EVAL, s),
            "cand_auroc": cand_auroc(EVAL, s),
            "delta_vs_incumbent": d, "delta_ci95": [dlo, dhi],
            "ontopic_pairwise_acc": ot, "n_ontopic_pairs": notp,
            "sel_eff_per_set": {ds: sel_eff([r for r in EVAL if r["ds"] == ds],
                                            [x for r, x in zip(EVAL, s) if r["ds"] == ds])[0]
                                for ds in sorted(set(r["ds"] for r in EVAL))},
        }
        print(f"  zero-shot {tag}: sel_eff={eff:.4f} [{lo:.4f},{hi:.4f}] "
              f"auroc={R['zero_shot'][tag]['cand_auroc']:.4f} ontopic_pw={ot:.4f}", flush=True)

    # ---------------- CONTROL (b): off-topic relevance vs on-topic correctness
    # positive = a candidate from this item's own pool; negative = a candidate stolen from a DIFFERENT
    # item of the same dataset (different image, different question).
    rng = np.random.default_rng(7)
    byds = defaultdict(list)
    for i, r in enumerate(EVAL):
        byds[r["ds"]].append(i)
    off = {"raw": [], "pmi": []}
    for i, r in enumerate(EVAL):
        pool = byds[r["ds"]]
        j = i
        while j == i:
            j = pool[rng.integers(0, len(pool))]
        o = EVAL[j]
        a_own = r["preds"][rng.integers(0, len(r["preds"]))]
        a_oth = o["preds"][rng.integers(0, len(o["preds"]))]
        v = E.v(r["img"])
        t1 = E.t(best_tmpl, r["q"], a_own); t2 = E.t(best_tmpl, o["q"], a_oth)
        for tag, B in (("raw", None), ("pmi", bank[r["ds"]])):
            s1, s2 = float(t1 @ v), float(t2 @ v)
            if B is not None:
                s1 -= float((B @ t1).mean()); s2 -= float((B @ t2).mean())
            off[tag].append(1.0 if s1 > s2 else (0.5 if s1 == s2 else 0.0))
    # a matched variant: keep the QUESTION fixed (so the template contributes identically) and steal
    # only the ANSWER string from another item -> isolates answer-content relevance
    off_ans = {"raw": [], "pmi": []}
    for i, r in enumerate(EVAL):
        pool = byds[r["ds"]]
        j = i
        while j == i:
            j = pool[rng.integers(0, len(pool))]
        o = EVAL[j]
        a_own = r["preds"][rng.integers(0, len(r["preds"]))]
        a_oth = o["preds"][rng.integers(0, len(o["preds"]))]
        if norm(a_own) == norm(a_oth):
            continue
        v = E.v(r["img"])
        try:
            t1 = E.t(best_tmpl, r["q"], a_own)
            t2 = E.t(best_tmpl, o["q"], a_oth) if best_tmpl == "ans" else None
        except KeyError:
            continue
        if best_tmpl != "ans":
            # need "<own question> <other answer>" which is not in the cache -> use the 'ans' template
            t1 = E.t("ans", r["q"], a_own); t2 = E.t("ans", o["q"], a_oth)
        for tag, B in (("raw", None), ("pmi", bank[r["ds"]])):
            s1, s2 = float(t1 @ v), float(t2 @ v)
            if B is not None:
                s1 -= float((B @ t1).mean()); s2 -= float((B @ t2).mean())
            off_ans[tag].append(1.0 if s1 > s2 else (0.5 if s1 == s2 else 0.0))
    R["control_relevance_vs_correctness"] = {
        "offtopic_pairwise_acc": {k: float(np.mean(v)) for k, v in off.items()},
        "offtopic_answer_only_pairwise_acc": {k: float(np.mean(v)) for k, v in off_ans.items()},
        "n_offtopic": len(off["raw"]), "n_offtopic_ans": len(off_ans["raw"]),
        "ontopic_pairwise_acc": {k: R["zero_shot"][k]["ontopic_pairwise_acc"] for k in ("raw", "pmi")},
        "note": ("on-topic = correct vs incorrect candidate for the SAME image+question (equally relevant, "
                 "equally fluent); off-topic = own candidate vs a candidate for a different item."),
    }
    print(f"  CONTROL off-topic pw={R['control_relevance_vs_correctness']['offtopic_pairwise_acc']} "
          f"vs on-topic pw={R['control_relevance_vs_correctness']['ontopic_pairwise_acc']}", flush=True)

    # ---------------- CONTROL (c): image-permutation null (zero-shot)
    rng = np.random.default_rng(11)
    perm = []
    for i, r in enumerate(EVAL):
        pool = [k for k in byds[r["ds"]]]
        j = i
        while EVAL[j]["img"] == r["img"]:
            j = pool[rng.integers(0, len(pool))]
        perm.append(EVAL[j]["img"])
    from scipy.stats import kendalltau
    sperm = item_scores(E, EVAL, best_tmpl, img_override=perm)
    strue = item_scores(E, EVAL, best_tmpl)
    taus = [kendalltau(a, b).correlation for a, b in zip(strue, sperm) if len(set(a.tolist())) > 1]
    R["control_image_permutation_zeroshot"] = {
        "kendall_tau_mean": float(np.nanmean(taus)), "n": int(len(taus)),
        "sel_eff_true_image": sel_eff(EVAL, strue)[0], "sel_eff_permuted_image": sel_eff(EVAL, sperm)[0],
        "ontopic_pairwise_true": ontopic_pairwise(EVAL, strue)[0],
        "ontopic_pairwise_permuted": ontopic_pairwise(EVAL, sperm)[0],
    }
    print(f"  CONTROL image-permutation: tau={R['control_image_permutation_zeroshot']['kendall_tau_mean']:.3f} "
          f"sel_eff {R['control_image_permutation_zeroshot']['sel_eff_true_image']:.4f}->"
          f"{R['control_image_permutation_zeroshot']['sel_eff_permuted_image']:.4f}", flush=True)

    # ---------------- trained alignment head (image-disjoint train pools)
    # inner split of TRAIN by IMAGE for model selection; nothing from EVAL is used.
    imgs = sorted({r["img"] for r in TRAIN})
    rs = np.random.default_rng(3); rs.shuffle(imgs)
    va_img = set(imgs[:max(1, len(imgs) // 5)])
    tr_rows = [r for r in TRAIN if r["img"] not in va_img]
    va_rows = [r for r in TRAIN if r["img"] in va_img]
    R["trained_head"] = {"n_train_items": len(tr_rows), "n_innerval_items": len(va_rows)}
    heads = {}
    for blind in (False, True):
        Xtr, Ytr, _, _ = feats(E, tr_rows, best_tmpl, blind=blind)
        Xva, Yva, _, _ = feats(E, va_rows, best_tmpl, blind=blind)
        ok = Ytr >= 0; Xtr, Ytr = Xtr[ok], Ytr[ok]
        ok = Yva >= 0; Xva, Yva = Xva[ok], Yva[ok]
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        m, vauc = train_mlp((Xtr - mu) / sd, Ytr, Xv=(Xva - mu) / sd, Yv=Yva)
        heads["textonly" if blind else "image_text"] = (m, mu, sd, vauc)
        print(f"  head blind={blind}: inner-val cand AUROC={vauc:.4f} (n_train_ex={len(Xtr)})", flush=True)

    # ---- seed robustness: 5 seeds, identical protocol, report the spread of the delta
    Xtr, Ytr, _, _ = feats(E, tr_rows, best_tmpl); ok = Ytr >= 0; Xtr, Ytr = Xtr[ok], Ytr[ok]
    Xva, Yva, _, _ = feats(E, va_rows, best_tmpl); ok = Yva >= 0; Xva, Yva = Xva[ok], Yva[ok]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xe_, _, _, ITEM_ = feats(E, EVAL, best_tmpl)
    seeds = []
    for sd_i in range(5):
        ms, _ = train_mlp((Xtr - mu) / sd, Ytr, Xv=(Xva - mu) / sd, Yv=Yva, seed=sd_i)
        ss = to_item_scores(predict(ms, (Xe_ - mu) / sd), ITEM_, len(EVAL))
        seeds.append({"seed": sd_i, "sel_eff": sel_eff(EVAL, ss)[0],
                      "delta_vs_incumbent": sel_eff(EVAL, ss)[0] - sel_eff(EVAL, inc)[0],
                      "sel_eff_per_set": {ds: sel_eff([r for r in EVAL if r["ds"] == ds],
                                                      [x for r, x in zip(EVAL, ss) if r["ds"] == ds])[0]
                                          for ds in sorted(set(r["ds"] for r in EVAL))}})
        del ms
    R["trained_head"]["seed_robustness"] = {
        "seeds": seeds,
        "sel_eff_mean": float(np.mean([s["sel_eff"] for s in seeds])),
        "sel_eff_sd": float(np.std([s["sel_eff"] for s in seeds])),
        "n_seeds_beating_incumbent": int(sum(s["delta_vs_incumbent"] > 0 for s in seeds)),
        "n_seeds_guardrail_clean": int(sum(all(s["sel_eff_per_set"][k] >= v for k, v in
                                              res["null_test_incumbent"]["sel_eff_per_set"].items())
                                          for s in seeds)),
    }
    print(f"  seed robustness: sel_eff {R['trained_head']['seed_robustness']['sel_eff_mean']:.4f}"
          f" +- {R['trained_head']['seed_robustness']['sel_eff_sd']:.4f}, "
          f"{R['trained_head']['seed_robustness']['n_seeds_beating_incumbent']}/5 beat incumbent, "
          f"{R['trained_head']['seed_robustness']['n_seeds_guardrail_clean']}/5 guardrail-clean", flush=True)

    for tag, (m, mu, sd, vauc) in heads.items():
        Xe, Ye, _, ITEM = feats(E, EVAL, best_tmpl, blind=(tag == "textonly"))
        p = predict(m, (Xe - mu) / sd)
        s = to_item_scores(p, ITEM, len(EVAL))
        eff, lo, hi = boot_ci(EVAL, s, A.boot)
        d, dlo, dhi = boot_paired(EVAL, s, inc, A.boot)
        ot, _ = ontopic_pairwise(EVAL, s)
        ent = {"inner_val_auroc": vauc, "sel_eff": eff, "sel_eff_ci95": [lo, hi],
               "sel_acc": sel_acc(EVAL, s), "cand_auroc": cand_auroc(EVAL, s),
               "delta_vs_incumbent": d, "delta_ci95": [dlo, dhi], "ontopic_pairwise_acc": ot,
               "sel_eff_per_set": {ds: sel_eff([r for r in EVAL if r["ds"] == ds],
                                               [x for r, x in zip(EVAL, s) if r["ds"] == ds])[0]
                                   for ds in sorted(set(r["ds"] for r in EVAL))},
               "contested_sel_eff": sel_eff([EVAL[i] for i in CONT], [s[i] for i in CONT])[0]}
        if tag == "image_text":
            Xp, _, _, ITEMp = feats(E, EVAL, best_tmpl, img_override=perm)
            sp = to_item_scores(predict(m, (Xp - mu) / sd), ITEMp, len(EVAL))
            ent["image_permutation_null"] = {
                "sel_eff_permuted": sel_eff(EVAL, sp)[0], "cand_auroc_permuted": cand_auroc(EVAL, sp),
                "ontopic_pairwise_permuted": ontopic_pairwise(EVAL, sp)[0],
                "kendall_tau_mean": float(np.nanmean([kendalltau(a, b).correlation for a, b in zip(s, sp)
                                                      if len(set(a.tolist())) > 1]))}
            R["trained_head_scores_corr_with_incumbent"] = float(np.corrcoef(
                np.concatenate(s), np.concatenate(inc))[0, 1])
        R["trained_head"][tag] = ent
        print(f"  trained head [{tag}]: sel_eff={eff:.4f} [{lo:.4f},{hi:.4f}] auroc={ent['cand_auroc']:.4f} "
              f"d_vs_inc={d:+.4f} [{dlo:+.4f},{dhi:+.4f}] ontopic_pw={ot:.4f}", flush=True)

    # ---- L2 (strict: no eval QUESTION TEXT in training either) retrain of the same head
    strict = {}
    for ds in ("slake_open_train", "vqa_rad_open_train", "pathvqa_open_train"):
        p = os.path.join(ROOT, f"data/disjoint_split/strict_idx_{ds}.json")
        strict[ds] = set(json.load(open(p)))
    TR2 = [r for r in TRAIN if r["idx"] in strict[r["ds"]]]
    imgs2 = sorted({r["img"] for r in TR2}); np.random.default_rng(3).shuffle(imgs2)
    va2 = set(imgs2[:max(1, len(imgs2) // 5)])
    a_rows = [r for r in TR2 if r["img"] not in va2]; b_rows = [r for r in TR2 if r["img"] in va2]
    Xa, Ya, _, _ = feats(E, a_rows, best_tmpl); ok = Ya >= 0; Xa, Ya = Xa[ok], Ya[ok]
    Xb, Yb, _, _ = feats(E, b_rows, best_tmpl); ok = Yb >= 0; Xb, Yb = Xb[ok], Yb[ok]
    mu2, sd2 = Xa.mean(0, keepdims=True), Xa.std(0, keepdims=True) + 1e-6
    m2, v2 = train_mlp((Xa - mu2) / sd2, Ya, Xv=(Xb - mu2) / sd2, Yv=Yb)
    s2 = to_item_scores(predict(m2, (Xe_ - mu2) / sd2), ITEM_, len(EVAL))
    d2, d2lo, d2hi = boot_paired(EVAL, s2, inc, A.boot)
    R["trained_head"]["L2_strict"] = {
        "n_train_items": len(a_rows), "inner_val_auroc": v2, "sel_eff": sel_eff(EVAL, s2)[0],
        "cand_auroc": cand_auroc(EVAL, s2), "delta_vs_incumbent": d2, "delta_ci95": [d2lo, d2hi],
        "sel_eff_per_set": {ds: sel_eff([r for r in EVAL if r["ds"] == ds],
                                        [x for r, x in zip(EVAL, s2) if r["ds"] == ds])[0]
                            for ds in sorted(set(r["ds"] for r in EVAL))},
        "note": "L2 = L1 + no eval question TEXT anywhere in training (bulletproof against a "
                "question->answer-prior shortcut); starves the in-domain pool.",
    }
    print(f"  L2-strict head: n_train={len(a_rows)} sel_eff={R['trained_head']['L2_strict']['sel_eff']:.4f} "
          f"d={d2:+.4f} [{d2lo:+.4f},{d2hi:+.4f}]", flush=True)
    del m2

    # ---- relevance-vs-correctness for the TRAINED head (same shape as the zero-shot control):
    # own candidate vs a candidate belonging to a different item, both scored against the OWN image.
    m, mu, sd, _ = heads["image_text"]
    rng = np.random.default_rng(7)
    Xo, Xf = [], []
    for i, r in enumerate(EVAL):
        pool = byds[r["ds"]]; j = i
        while j == i:
            j = pool[rng.integers(0, len(pool))]
        o = EVAL[j]
        v = E.v(r["img"])
        t1 = E.t(best_tmpl, r["q"], r["preds"][rng.integers(0, len(r["preds"]))])
        t2 = E.t(best_tmpl, o["q"], o["preds"][rng.integers(0, len(o["preds"]))])
        Xo.append(np.concatenate([v, t1, v * t1, np.abs(v - t1)]))
        Xf.append(np.concatenate([v, t2, v * t2, np.abs(v - t2)]))
    po = predict(m, (np.asarray(Xo, np.float32) - mu) / sd)
    pf_ = predict(m, (np.asarray(Xf, np.float32) - mu) / sd)
    R["control_relevance_vs_correctness"]["trained_head"] = {
        "offtopic_pairwise_acc": float(np.mean((po > pf_) + 0.5 * (po == pf_))),
        "ontopic_pairwise_acc": R["trained_head"]["image_text"]["ontopic_pairwise_acc"],
        "incumbent_ontopic_pairwise_acc": ontopic_pairwise(EVAL, inc)[0],
    }
    print(f"  CONTROL trained head: off-topic pw="
          f"{R['control_relevance_vs_correctness']['trained_head']['offtopic_pairwise_acc']:.4f} "
          f"on-topic pw={R['trained_head']['image_text']['ontopic_pairwise_acc']:.4f} "
          f"(incumbent on-topic {R['control_relevance_vs_correctness']['trained_head']['incumbent_ontopic_pairwise_acc']:.4f})",
          flush=True)

    # ---------------- slices (d) + surface confounds (e), on the best zero-shot and the trained head
    LAT = re.compile(r"\b(left|right|bilateral)\b", re.I)
    NEG = re.compile(r"\b(no|not|without|absent|negative|none)\b", re.I)
    m, mu, sd, _ = heads["image_text"]
    Xe, Ye, _, ITEM = feats(E, EVAL, best_tmpl)
    strained = to_item_scores(predict(m, (Xe - mu) / sd), ITEM, len(EVAL))
    slices = {}
    for sname, rx in (("laterality", LAT), ("negation", NEG)):
        keep = [i for i, r in enumerate(EVAL) if any(rx.search(a) for a in r["preds"]) or rx.search(r["q"])]
        rs_ = [EVAL[i] for i in keep]
        slices[sname] = {
            "n": len(keep),
            "incumbent_sel_eff": sel_eff(rs_, [inc[i] for i in keep])[0],
            "zeroshot_sel_eff": sel_eff(rs_, [strue[i] for i in keep])[0],
            "trained_sel_eff": sel_eff(rs_, [strained[i] for i in keep])[0],
            "trained_ontopic_pairwise": ontopic_pairwise(rs_, [strained[i] for i in keep])[0],
        }
    R["slices"] = slices
    lens = np.array([len(a) for r in EVAL for a in r["preds"]], float)
    freq = np.array([Counter(norm(x) for x in r["preds"])[norm(a)] for r in EVAL for a in r["preds"]], float)
    zs = np.concatenate(strue); ts = np.concatenate(strained)
    R["surface_confounds"] = {
        "zeroshot_corr_len": float(np.corrcoef(zs, lens)[0, 1]),
        "zeroshot_corr_poolfreq": float(np.corrcoef(zs, freq)[0, 1]),
        "trained_corr_len": float(np.corrcoef(ts, lens)[0, 1]),
        "trained_corr_poolfreq": float(np.corrcoef(ts, freq)[0, 1]),
        "incumbent_corr_poolfreq": float(np.corrcoef(np.concatenate(inc), freq)[0, 1]),
    }

    # ---------------- DIAGNOSTIC fusion with the incumbent (cross-fitted on eval; NOT deployable)
    from sklearn.linear_model import LogisticRegression
    Z = np.stack([np.concatenate(inc), ts, zs], 1)
    Yc = np.concatenate([r["sl"] for r in EVAL]); ITEMc = np.concatenate(
        [[i] * len(r["preds"]) for i, r in enumerate(EVAL)])
    folds = np.array([int(EVAL[i]["img"][:8], 16) % 5 for i in ITEMc])   # fold by IMAGE (md5), stable
    pf = np.zeros(len(Z))
    for f in range(5):
        tr = folds != f; te = ~tr
        lr = LogisticRegression(max_iter=2000).fit(Z[tr], Yc[tr])
        pf[te] = lr.predict_proba(Z[te])[:, 1]
    sf = to_item_scores(pf, ITEMc, len(EVAL))
    d, dlo, dhi = boot_paired(EVAL, sf, inc, A.boot)
    R["fusion_with_incumbent_DIAGNOSTIC_ONLY"] = {
        "sel_eff": sel_eff(EVAL, sf)[0], "delta_vs_incumbent": d, "delta_ci95": [dlo, dhi],
        "protocol": "5-fold cross-fit by image hash ON THE EVAL SET (the incumbent adapter was never "
                    "run over the disjoint train pool) -> upper bound / diagnostic, never deployable.",
    }
    print(f"  fusion (DIAGNOSTIC): sel_eff={R['fusion_with_incumbent_DIAGNOSTIC_ONLY']['sel_eff']:.4f} "
          f"d={d:+.4f} [{dlo:+.4f},{dhi:+.4f}]", flush=True)
    res["encoders"][name] = R

outp = os.path.join(ROOT, A.out)
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump(res, open(outp, "w"), indent=1)
print("\nwrote", outp)
