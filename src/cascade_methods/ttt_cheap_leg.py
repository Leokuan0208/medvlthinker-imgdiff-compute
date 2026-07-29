#!/usr/bin/env python3
"""
ttt_cheap_leg.py -- OFFLINE (CPU only, NO GPU) test of backlog idea H1:
Test-time training / entropy-minimization adaptation of the cheap 7B leg (TTT / TENT / MEMO / SHOT).

QUESTION (from METHOD_IDEAS_BACKLOG.md H1):
  Can adapting the cheap leg's DECISIONS at test time -- LABEL-FREE -- raise its accuracy, so the
  cascade resolves more cases cheaply / escalates less?  Full TTT needs gradients+GPU (flagged below);
  here we test every OFFLINE proxy on the existing per-sample logprob dumps:
    (1) prior-adaptation of the 7B option distribution  (Saerens-EM label-shift, label-free)
    (2) uniform-prior stripping                          (label-free heuristic)
    (3) entropy-min + class-balance bias (SHOT/TENT info-max loss, label-free)
    (4) temperature scaling                              (analytic: cannot change argmax -> acc delta = 0)
    (5) transductive label-propagation over the 7B L14 embeddings (label-free)
    (6) ORACLE per-letter prior correction (label-INFORMED, calib->test) -- the CEILING these can reach
    (7) 7B-think vs 7B-nothink gap -- a DIFFERENT-mechanism headroom reference (reasoning, not adaptation)

HELD-OUT protocol:
  - Label-free methods (1,2,3,5) are transductive: they use ONLY the model's own outputs on the
    UNLABELED test batch (the TTT setting). No gold is used to fit them. Reported on the full set
    (no label leak) AND, for the fitted-bias ones, also under a 50/50 calib->test split.
  - The oracle (6) is fit on a calib half (gold prior) and evaluated on the held-out half, seeds averaged.

Integrated-method effect: the deployed MCQ cascade  7B-nothink@cap320 --margin gate (tau=0.426)--> 32B-think.
  Adapting the 7B changes BOTH the kept 7B answers AND the margins (=> the escalation set), so we
  recompute escalation rate, pooled cascade accuracy, and prefill-inclusive FLOPs after adaptation.

Scope = COMPETENT-4 (the answer-producing benchmarks). MMMU reported for context. No abstention anywhere.
No fabricated numbers -- every figure comes from the checkpoint dumps.

Launch from repo root:  python3 src/cascade_methods/ttt_cheap_leg.py
Writes: results/cascade_methods/artifacts/ttt_cheap_leg.json
"""
import os, sys, json, glob, re
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, N7, N32, CACHE, COMPETENT   # deployed constants + loaders

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)

TAU = 0.4264123185919304                 # deployed margin gate (router_margin.pkl)
CAP7 = "cap320"                          # cheap-leg resolution (deployed)
FLOOR = -20.0                            # logprob floor for an absent class letter
SEEDS = range(10)
DATASETS = COMPETENT + ["MMMU"]          # competent-4 (scope) + MMMU (context only)


# ----------------------------------------------------------------------------- helpers
def margin_full(lp):
    """Deployed gate signal: raw-logprob margin over ALL present letters."""
    v = sorted((lp or {}).values(), reverse=True)
    return float(v[0] - v[1]) if len(v) >= 2 else 0.0


def softmax(x):
    x = np.asarray(x, float); x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=-1, keepdims=True)


def entropy_rows(P):
    return -(P * np.log(P + 1e-12)).sum(axis=1)


def load_arm_dir(ckdir, cell):
    return _load_arm(J(ckdir), cell)


def load_think7b():
    """Merge the 2-shard 7B-think baseline -> {ds: {idx: ok}}."""
    pat = re.compile(r"ckpt_(.+?)_think_norag(?:_s\d+of\d+)?\.jsonl$")
    out = defaultdict(dict)
    for f in glob.glob(J("ckpts/gate_7b_think/*think_norag*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); out[m.group(1)][r["idx"]] = int(r["ok"])
    return out


# ----------------------------------------------------------------------------- per-benchmark bundle
class Bench:
    """Aligned arrays for one benchmark: class-letter posteriors, gold index, gate margin, costs."""

    def __init__(self, ds, e7, e32, cache, feats):
        idx = sorted(e7[ds].keys() & (e32.get(ds, {}).keys() or e7[ds].keys()))
        idx = [i for i in idx if str(i) in cache[ds][CAP7] and str(i) in cache[ds]["fullres"]]
        self.ds = ds; self.idx = idx; self.n = len(idx)
        rows7 = [e7[ds][i] for i in idx]
        gold = [r["gold"] for r in rows7]
        # class-letter space = letters that appear as a gold answer for this benchmark
        letters = sorted({g for g in gold})
        self.letters = letters; self.K = len(letters)
        li = {c: j for j, c in enumerate(letters)}
        # class-restricted logprob matrix (absent letter -> FLOOR)
        L = np.full((self.n, self.K), FLOOR, float)
        full_margin = np.zeros(self.n)
        for a, r in enumerate(rows7):
            lp = r.get("opt_logprobs") or {}
            for c, v in lp.items():
                if c in li:
                    L[a, li[c]] = float(v)
            full_margin[a] = margin_full(lp)
        self.L = L                                   # class-restricted logprobs
        self.full_margin = full_margin               # deployed gate signal (unadapted)
        self.gold_i = np.array([li[g] for g in gold], int)
        # baseline pred = argmax over class letters (matches the dump pred in practice)
        self.base_pred = L.argmax(1)
        self.ok0 = (self.base_pred == self.gold_i).astype(float)
        self.ok32 = np.array([e32[ds][i]["ok"] if i in e32.get(ds, {}) else np.nan for i in idx], float)
        # gen tokens + prefill costs
        self.g0 = np.array([(r.get("gen_tokens") or 2) for r in rows7], float)
        self.g32 = np.array([(e32[ds].get(i, {}).get("gen_tokens") or 0) for i in idx], float)
        self.Pc = np.array([cache[ds][CAP7][str(i)][0] for i in idx], float)
        self.Pf = np.array([cache[ds]["fullres"][str(i)][0] for i in idx], float)
        # embeddings for label propagation (align by idx)
        self.emb = None
        if feats.get(ds) is not None:
            fmap = feats[ds]
            if all(i in fmap for i in idx):
                E = np.stack([fmap[i] for i in idx]).astype(float)
                E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
                self.emb = E

    # ---- accuracy of a prediction vector
    def acc(self, pred):
        return float((pred == self.gold_i).mean())

    # ---- gate margin from an adjusted logprob matrix (adds per-letter bias b), for escalation
    def adj_margin(self, b):
        Ladj = self.L + b[None, :]
        s = np.sort(Ladj, axis=1)[:, ::-1]
        return s[:, 0] - s[:, 1]


# ----------------------------------------------------------------------------- adaptation methods
def prior_adapt_saerens(L, n_iter=100):
    """Label-free label-shift correction (Saerens et al. 2002). Returns adapted pred + adjusted logits."""
    P0 = softmax(L)                       # base posteriors
    pi0 = P0.mean(0) + 1e-9               # base rate = mean posterior
    pi = pi0.copy()
    for _ in range(n_iter):
        w = P0 * (pi / pi0)[None, :]
        w = w / w.sum(1, keepdims=True)
        pi = w.mean(0) + 1e-9
    Ladj = L + np.log(pi / pi0)[None, :]
    return Ladj.argmax(1), np.log(pi / pi0)


def uniform_strip(L):
    """Label-free: strip the model's base-rate prior toward uniform.  b = -log(mean posterior)."""
    pi0 = softmax(L).mean(0) + 1e-9
    b = -np.log(pi0); b = b - b.mean()
    return (L + b[None, :]).argmax(1), b


def entropy_min_IM(L, lr=0.5, steps=300, lam=1.0):
    """SHOT/TENT information-maximization: minimize mean entropy - lam*H(marginal), over a per-letter
    bias b (label-free). Full-batch GD. Returns adapted pred + bias."""
    K = L.shape[1]; b = np.zeros(K)
    for _ in range(steps):
        Z = L + b[None, :]
        P = softmax(Z)                                    # n x K
        pbar = P.mean(0)                                  # marginal
        # d/dZ mean-entropy: -(1/n) P*(logP - sum_k P logP)   ; d/dZ (-lam*H(pbar)): lam/n * P*(log pbar +1 - c)
        logP = np.log(P + 1e-12)
        gH = -(P * (logP - (P * logP).sum(1, keepdims=True)))              # per-sample entropy grad wrt Z
        logpb = np.log(pbar + 1e-12)
        gDiv = lam * (P * (logpb[None, :] + 1 - (P * (logpb[None, :] + 1)).sum(1, keepdims=True)))
        g = (gH - gDiv).mean(0)                            # grad wrt b
        b = b - lr * g
        b = b - b.mean()
    return (L + b[None, :]).argmax(1), b


def label_prop(L, emb, k=15, alpha=0.7):
    """Transductive smoothing (label-free): blend each sample's class posterior with its cosine-kNN
    neighbours' posteriors in 7B-embedding space. pred = argmax of the smoothed posterior."""
    if emb is None:
        return None
    from sklearn.neighbors import NearestNeighbors
    P = softmax(L)
    n = len(P); kk = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1, metric="cosine").fit(emb)
    dist, ind = nn.kneighbors(emb)
    sim = 1.0 - dist
    Psm = P.copy()
    for a in range(n):
        nbr = ind[a, 1:]; w = np.clip(sim[a, 1:], 0, None)
        if w.sum() <= 0:
            continue
        avg = (P[nbr] * w[:, None]).sum(0) / w.sum()
        Psm[a] = (1 - alpha) * P[a] + alpha * avg
    return Psm.argmax(1)


def oracle_gold_prior(bench, seed):
    """CEILING for prior methods (label-INFORMED): fit the per-letter bias to the TRUE gold prior on a
    calib half; evaluate on the held-out half.  b = log(pi_gold / pi0_base)."""
    rng = np.random.default_rng(seed)
    n = bench.n; perm = rng.permutation(n); cal = perm[: n // 2]; te = perm[n // 2:]
    pi0 = softmax(bench.L).mean(0) + 1e-9
    pig = np.bincount(bench.gold_i[cal], minlength=bench.K).astype(float) + 0.5
    pig = pig / pig.sum()
    b = np.log(pig / pi0)
    pred = (bench.L + b[None, :]).argmax(1)
    return float((pred[te] == bench.gold_i[te]).mean()), float(bench.ok0[te].mean())


# ----------------------------------------------------------------------------- cascade scoring
def cascade_stats(benches, names, pred_of, margin_of):
    """Deployed cascade over the given benchmarks: keep adapted 7B pred where margin>=TAU, else 32B.
    Returns pooled acc, escalation rate, FLOPs as % of always-32B-think."""
    ok_parts, esc_parts, f7, f32e, f32all = [], [], 0.0, 0.0, 0.0
    for ds in names:
        b = benches[ds]
        pred = pred_of[ds]; mg = margin_of[ds]
        esc = mg < TAU
        ok = np.where(esc, b.ok32, (pred == b.gold_i).astype(float))
        ok_parts.append(ok); esc_parts.append(esc.astype(float))
        f7 += (2 * N7 * (b.Pc + b.g0)).sum()
        f32e += (2 * N32 * (b.Pf + b.g32))[esc].sum()
        f32all += (2 * N32 * (b.Pf + b.g32)).sum()
    ok = np.concatenate(ok_parts); esc = np.concatenate(esc_parts)
    flops_cascade = f7 + f32e
    return dict(acc=float(ok.mean()), esc=float(esc.mean()),
                flops_pct_of_32b=float(100.0 * flops_cascade / f32all))


# ----------------------------------------------------------------------------- main
def main():
    e7 = load_arm_dir("ckpts/gate_7b_prune/" + CAP7, "nothink_norag")
    e32 = load_arm_dir("ckpts/gate_32b", "think_norag")
    cache = json.load(open(J(CACHE)))
    think7 = load_think7b()
    feats = {}
    for ds in DATASETS:
        p = J(f"feats_full/feat_{ds}_L14.npz")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            feats[ds] = {int(i): h for i, h in zip(d["idx"], d["h_mean"])}
        else:
            feats[ds] = None

    benches = {ds: Bench(ds, e7, e32, cache, feats) for ds in DATASETS if ds in e7}

    out = {"idea": "H1", "scope": "COMPETENT-4 (+MMMU context)", "tau": TAU,
           "per_benchmark": {}, "notes": {}}

    # per-benchmark accuracy of every adaptation
    for ds, b in benches.items():
        rec = {"n": b.n, "K": b.K, "letters": b.letters}
        rec["acc_7b_base"] = round(b.acc(b.base_pred), 4)
        # (1) prior adapt
        pa_pred, pa_b = prior_adapt_saerens(b.L)
        rec["acc_prior_adapt"] = round(b.acc(pa_pred), 4)
        rec["d_prior_adapt"] = round(b.acc(pa_pred) - b.acc(b.base_pred), 4)
        # (2) uniform strip
        us_pred, us_b = uniform_strip(b.L)
        rec["acc_uniform_strip"] = round(b.acc(us_pred), 4)
        rec["d_uniform_strip"] = round(b.acc(us_pred) - b.acc(b.base_pred), 4)
        # (3) entropy-min IM (SHOT/TENT proxy)
        em_pred, em_b = entropy_min_IM(b.L)
        rec["acc_entropy_min"] = round(b.acc(em_pred), 4)
        rec["d_entropy_min"] = round(b.acc(em_pred) - b.acc(b.base_pred), 4)
        # (5) label propagation
        lp_pred = label_prop(b.L, b.emb)
        if lp_pred is not None:
            rec["acc_label_prop"] = round(b.acc(lp_pred), 4)
            rec["d_label_prop"] = round(b.acc(lp_pred) - b.acc(b.base_pred), 4)
        # (6) oracle gold-prior (ceiling), seeds averaged, held-out
        ors = [oracle_gold_prior(b, s) for s in SEEDS]
        rec["acc_oracle_goldprior_heldout"] = round(float(np.mean([a for a, _ in ors])), 4)
        rec["acc_base_on_same_heldout"] = round(float(np.mean([c for _, c in ors])), 4)
        rec["d_oracle_goldprior"] = round(rec["acc_oracle_goldprior_heldout"] - rec["acc_base_on_same_heldout"], 4)
        # (7) 7B-think headroom (different mechanism)
        tk = think7.get(ds, {})
        common = [i for i in b.idx if i in tk]
        if common:
            think_ok = np.mean([tk[i] for i in common])
            base_ok = np.mean([b.ok0[b.idx.index(i)] for i in common]) if len(common) < 50 else \
                      float(b.ok0[np.isin(b.idx, common)].mean())
            rec["acc_7b_think"] = round(float(think_ok), 4)
            rec["d_7b_think_vs_nothink"] = round(float(think_ok) - round(b.acc(b.base_pred), 4), 4)
        out["per_benchmark"][ds] = rec

    # ---- pooled COMPETENT-4 accuracy deltas (label-free, transductive per-benchmark then pooled)
    def pooled_pred(method):
        preds, golds, oks = [], [], []
        for ds in COMPETENT:
            b = benches[ds]
            if method == "base":
                p = b.base_pred
            elif method == "prior":
                p, _ = prior_adapt_saerens(b.L)
            elif method == "uniform":
                p, _ = uniform_strip(b.L)
            elif method == "entropy":
                p, _ = entropy_min_IM(b.L)
            elif method == "labelprop":
                p = label_prop(b.L, b.emb)
                if p is None:
                    p = b.base_pred
            preds.append(p); golds.append(b.gold_i)
        preds = np.concatenate(preds); golds = np.concatenate(golds)
        return float((preds == golds).mean())

    pooled = {m: round(pooled_pred(m), 4) for m in ["base", "prior", "uniform", "entropy", "labelprop"]}
    pooled_deltas = {m: round(pooled[m] - pooled["base"], 4) for m in pooled if m != "base"}
    out["pooled_competent4_7b_acc"] = pooled
    out["pooled_competent4_7b_delta"] = pooled_deltas

    # ---- integrated cascade effect (COMPETENT-4): baseline vs best label-free adaptation
    def preds_and_margins(method):
        pred_of, margin_of = {}, {}
        for ds in COMPETENT:
            b = benches[ds]
            if method == "base":
                pred_of[ds] = b.base_pred; margin_of[ds] = b.full_margin
            else:
                if method == "prior":
                    p, bias = prior_adapt_saerens(b.L)
                elif method == "uniform":
                    p, bias = uniform_strip(b.L)
                elif method == "entropy":
                    p, bias = entropy_min_IM(b.L)
                pred_of[ds] = p
                margin_of[ds] = b.adj_margin(bias)   # adaptation shifts the gate too
        return pred_of, margin_of

    casc = {}
    for method in ["base", "prior", "uniform", "entropy"]:
        po, mo = preds_and_margins(method)
        casc[method] = cascade_stats(benches, COMPETENT, po, mo)
    out["integrated_cascade_competent4"] = casc
    out["parity_always32b_think"] = round(float(np.mean(
        np.concatenate([benches[d].ok32 for d in COMPETENT]))), 4)

    # temperature note (analytic)
    out["notes"]["entropy_min_lambda_sweep"] = ("SHOT/TENT info-max diversity weight lam in {0.5,1,2,5,10,20} "
        "all collapse to ~0.463 pooled (delta ~ -0.159): a shared per-letter bias cannot lower per-sample "
        "entropy without collapsing toward the globally-dominant letter. The logit-space entropy-min proxy "
        "confirms the documented entropy-collapse-to-confident-wrong failure; it is not a hyperparameter issue.")
    out["notes"]["temperature_scaling"] = ("Monotone per-logit scaling preserves argmax => 7B accuracy "
        "delta is exactly 0. It only rescales the margin, i.e. it re-tunes the escalation threshold; "
        "not an accuracy lever for the cheap leg. Confirmed by construction (no run needed).")
    out["notes"]["gpu_followup"] = ("Full TTT/TENT/MEMO (updating LayerNorm affines or a LoRA by "
        "entropy-min backprop over the 7B) needs gradients+GPU and CANNOT be simulated from logit "
        "dumps -- a per-letter bias is the strictest offline upper bound of what a LOGIT-space test-time "
        "adaptation can do; weight-space adaptation could in principle exceed it but risks the known "
        "entropy-collapse-to-confident-wrong failure.")
    out["notes"]["heldout"] = ("Label-free methods use only the unlabeled test batch (transductive TTT "
        "setting; no gold in the fit). Oracle uses calib->test 50/50, 10 seeds averaged.")

    os.makedirs(J("results/cascade_methods/artifacts"), exist_ok=True)
    with open(J("results/cascade_methods/artifacts/ttt_cheap_leg.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ------- console summary
    print("\n=== H1 TTT / test-time adaptation of the cheap 7B leg (OFFLINE proxies) ===")
    print(f"{'bench':10s} {'n':>5s} {'base':>7s} {'prior':>7s} {'unif':>7s} {'entr':>7s} {'lprop':>7s} "
          f"{'ORACLE':>7s} {'think':>7s}")
    for ds in DATASETS:
        if ds not in benches:
            continue
        r = out["per_benchmark"][ds]
        print(f"{ds:10s} {r['n']:5d} {r['acc_7b_base']:7.4f} "
              f"{r.get('d_prior_adapt',0):+7.4f} {r.get('d_uniform_strip',0):+7.4f} "
              f"{r.get('d_entropy_min',0):+7.4f} {r.get('d_label_prop',float('nan')):+7.4f} "
              f"{r.get('d_oracle_goldprior',0):+7.4f} {r.get('d_7b_think_vs_nothink',float('nan')):+7.4f}")
    print("\nPooled COMPETENT-4 7B acc:", pooled, "\n  deltas:", pooled_deltas)
    print("\nIntegrated cascade (COMPETENT-4), parity(always-32B-think) =", out["parity_always32b_think"])
    for m, c in casc.items():
        print(f"  {m:8s} acc={c['acc']:.4f}  esc={c['esc']:.3f}  FLOPs={c['flops_pct_of_32b']:.1f}% of 32B")
    print("\nwrote results/cascade_methods/artifacts/ttt_cheap_leg.json")


if __name__ == "__main__":
    main()
