#!/usr/bin/env python3
"""
harness.py - unified OFFLINE evaluation harness for two-model cascade gating methods.

A cascade gate is, abstractly, a per-sample ESCALATION SCORE computed from the cheap 7B's
signals; we escalate the high-score (low-confidence) questions to the 32B. This module loads
all the data once and scores ANY escalation decision the same way the live pipeline does, so
many training-free cascade methods can be compared apples-to-apples WITHOUT any GPU run.

What it loads (default cap = cap320, the chosen operating point):
  - cheap 7B leg  : ckpts/gate_7b_prune/<cap>/   (no-think, capped res; full per-letter logprobs)
  - strong 32B leg: ckpts/gate_32b/              (think; correctness + gen_tokens; NO logprobs)
  - prompt tokens : ckpts/token_cache.json       (P at each cap, and fullres for the 32B)
  - calibration   : ckpts/gate_7b_pmctrain_prune/<cap>/ckpt_nothink.jsonl  (PMC-train, eval held out)

Cost model (IDENTICAL to src/sweep/grid_resolution_tau.py and cascade_cost_prefill_flops.py):
  one model run on q   = 2 * N_backbone * (P + G)         [P=prompt incl. vision, G=generated]
  cascade(q)           = run_7B(q) + [escalated] run_32B(q)
  always-32B(q)        = run_32B(q)
  backbone%            = sum(cascade FLOPs) / sum(always-32B FLOPs)
  cheap 7B leg priced at cap-Y prompt tokens; 32B leg priced at full-res prompt tokens.

Run directly to print baselines + the frozen-margin-gate anchor (must reproduce ~74% backbone,
0.572 acc, 63% escalation on the all-6 pool). CPU only. Launch from the repo root.
"""
import json, glob, os, re
import numpy as np
from collections import defaultdict

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
N7, N32, NVIT, MERGE = 7.6e9, 33.0e9, 0.675e9, 2          # backbones; shared ViT (used only for +ViT view)

COMPETENT = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]    # where the method is claimed to work
EXCLUDED  = ["MMMU", "MedXpert-Reasoning", "MedXpert-Understanding"]
ALL6      = COMPETENT + EXCLUDED                           # MedXpert R+U = one benchmark -> "6 benchmarks", 8220 rows
ALL5      = COMPETENT + ["MMMU"]                           # 5 benchmarks excl. MedXpert (MedX-M); incl. recoverable MMMU
CAPS      = ["fullres", "cap640", "cap320", "cap160", "cap80"]

EVAL_DIR = dict({"fullres": "ckpts/gate_7b_vllm"}, **{c: f"ckpts/gate_7b_prune/{c}" for c in CAPS[1:]})
PMCT_DIR = dict({"fullres": "ckpts/gate_7b_pmctrain"}, **{c: f"ckpts/gate_7b_pmctrain_prune/{c}" for c in CAPS[1:]})
DIR_32B  = "ckpts/gate_32b"
CACHE    = "ckpts/token_cache.json"


# ----------------------------------------------------------------------------- loaders
def _load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{re.escape(cell)}(?:_s\d+of\d+)?\.jsonl$")
    d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        for l in open(f):
            if l.strip():
                try:
                    r = json.loads(l); d[m.group(1)][r["idx"]] = r
                except Exception:
                    pass
    return d


def _load_calib(ckdir):
    rows = []
    for f in glob.glob(os.path.join(ckdir, "ckpt_nothink*.jsonl")):
        for l in open(f):
            if l.strip():
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass
    return rows


# ------------------------------------------------------------------- per-sample signals
def signals_from_logprobs(lp):
    """All confidence/uncertainty signals derivable from a 7B per-letter logprob dict.
    Returns a dict. Convention used by methods: build an ESCALATION SCORE where HIGHER == more
    likely to escalate (i.e. less confident). These are the raw ingredients; methods combine them.
    """
    lp = lp or {}
    vals = np.array(sorted(lp.values(), reverse=True), dtype=np.float64)   # logprobs, desc
    if vals.size == 0:
        return dict(margin=0.0, maxlogprob=-20.0, top1prob=0.0, prob_margin=0.0,
                    entropy=4.0, neg_energy=20.0, gini=1.0, top2mass=0.0, n_opts=0,
                    margin_top2=0.0, entropy2=1.0)
    # raw-logprob signals (what the deployed margin gate uses)
    margin     = float(vals[0] - vals[1]) if vals.size >= 2 else float(vals[0] - (-20.0))
    maxlogprob = float(vals[0])
    neg_energy = -float(np.log(np.exp(vals - vals.max()).sum()) + vals.max())   # = -logsumexp(lp)
    # softmax over all present letters -> proper distribution
    p = np.exp(vals - vals.max()); p = p / p.sum()
    top1prob   = float(p[0])
    prob_margin = float(p[0] - p[1]) if p.size >= 2 else float(p[0])
    entropy    = float(-(p * np.log(p + 1e-12)).sum())
    gini       = float(1.0 - (p * p).sum())
    top2mass   = float(p[0] + p[1]) if p.size >= 2 else float(p[0])
    # softmax restricted to top-2 letters (the actual competition for a MC answer)
    if vals.size >= 2:
        q = np.exp(vals[:2] - vals[:2].max()); q = q / q.sum()
        margin_top2 = float(q[0] - q[1])
        entropy2    = float(-(q * np.log(q + 1e-12)).sum())
    else:
        margin_top2, entropy2 = 1.0, 0.0
    return dict(margin=margin, maxlogprob=maxlogprob, top1prob=top1prob, prob_margin=prob_margin,
                entropy=entropy, neg_energy=neg_energy, gini=gini, top2mass=top2mass,
                n_opts=int(vals.size), margin_top2=margin_top2, entropy2=entropy2)


class CascadeData:
    """Holds aligned per-sample arrays for the eval set and the calibration set."""

    def __init__(self, cap="cap320", repo=REPO, strong_dir=DIR_32B, strong_cell="think_norag",
                 strong_cap="fullres"):
        """cap = cheap-leg (7B) resolution. strong_* select the escalation TARGET (32B) source so a
        cheaper strong leg can be scored: strong_dir/strong_cell pick the label files, strong_cap
        prices its prefill from token_cache. Defaults = deployed 32B-think@fullres."""
        self.cap = cap; self.strong_cap = strong_cap
        J = lambda p: os.path.join(repo, p)
        cache = json.load(open(J(CACHE)))
        ev = _load_arm(J(EVAL_DIR[cap]), "nothink_norag")
        r32 = _load_arm(J(strong_dir), strong_cell)

        self.ds_names = []
        self.per_ds = {}     # ds -> dict of aligned np arrays
        for ds in ALL6:
            if ds not in ev or ds not in r32:
                continue
            cY = cache.get(ds, {}).get(cap, {}); cF = cache.get(ds, {}).get(strong_cap, {})
            idx = sorted(set(ev[ds]) & set(r32[ds]) & {int(k) for k in cY} & {int(k) for k in cF})
            if not idx:
                continue
            sig = [signals_from_logprobs(ev[ds][i].get("opt_logprobs")) for i in idx]
            self.per_ds[ds] = dict(
                idx=np.array(idx),
                a7=np.array([ev[ds][i]["ok"] for i in idx], float),
                a32=np.array([r32[ds][i]["ok"] for i in idx], float),
                g7=np.array([float(ev[ds][i].get("gen_tokens") or 0) for i in idx], float),
                g32=np.array([float(r32[ds][i].get("gen_tokens") or 0) for i in idx], float),
                PY=np.array([cY[str(i)][0] for i in idx], float),       # cheap-leg prompt tokens (cap-Y)
                PF=np.array([cF[str(i)][0] for i in idx], float),       # 32B prompt tokens (full res)
                visY=np.array([cY[str(i)][1] for i in idx], float),
                visF=np.array([cF[str(i)][1] for i in idx], float),
                sig={k: np.array([s[k] for s in sig], float) for k in sig[0]},
            )
            self.ds_names.append(ds)

        # calibration: PMC-train at the same cap (cheap 7B only; its OWN correctness; eval held out)
        calib_rows = _load_calib(J(PMCT_DIR[cap]))
        self._calib_idx = [r["idx"] for r in calib_rows]
        csig = [signals_from_logprobs(r.get("opt_logprobs")) for r in calib_rows]
        self.calib = dict(
            ok=np.array([r["ok"] for r in calib_rows], float),
            sig={k: np.array([s[k] for s in csig], float) for k in csig[0]} if csig else {},
        )
        self._add_cross_cap_signal(repo)
        self._add_strong_calib(repo)

    def _add_strong_calib(self, repo):
        """Load 32B-on-PMC-train correctness + gen_tokens aligned to the calibration set, if the
        run exists. Unlocks deferral/recoverability-aware methods and an honest calib-to-parity
        threshold (no eval peeking). Sets self.calib['a32'], self.calib['g32'], self.has_strong."""
        self.has_strong = False
        path = os.path.join(repo, "ckpts/gate_32b_pmctrain/ckpt_think.jsonl")
        if not os.path.exists(path):
            return
        m = {}
        for l in open(path):
            if l.strip():
                try:
                    r = json.loads(l); m[r["idx"]] = r
                except Exception:
                    pass
        if len(m) < len(self._calib_idx):
            self._calib_strong_partial = (len(m), len(self._calib_idx))   # note partial; still usable
        idx = [i for i in self._calib_idx if i in m]
        if not idx:
            return
        # restrict calib arrays to rows where the 32B label exists (keeps everything aligned)
        keep = np.array([i in m for i in self._calib_idx], bool)
        self.calib["ok"] = self.calib["ok"][keep]
        self.calib["sig"] = {k: v[keep] for k, v in self.calib["sig"].items()}
        self.calib["a32"] = np.array([m[i]["ok"] for i in self._calib_idx if i in m], float)
        self.calib["g32"] = np.array([float(m[i].get("gen_tokens") or 0) for i in self._calib_idx if i in m], float)
        self._calib_idx = idx
        self.has_strong = True

    def _add_cross_cap_signal(self, repo):
        """Free decorrelated ensemble: the 7B is already decoded at cap80/160/320/640. Disagreement
        of the other caps with the operating cap = a resolution-perturbation uncertainty signal,
        partially independent of the logprob-shape signals. Adds 'cap_disagree' (frac of other caps
        whose pred differs) and 'cap_nuniq' (# distinct preds across caps, 1..4) to every sig dict.
        Calibrated on the matching PMC-train caps. Silently no-ops if cap files are missing."""
        J = lambda p: os.path.join(repo, p)
        other = [c for c in ["cap80", "cap160", "cap320", "cap640"]]
        try:
            ev = {c: _load_arm(J(EVAL_DIR[c]), "nothink_norag") for c in other}
            for ds in self.ds_names:
                idx = self.per_ds[ds]["idx"]
                preds = {c: ev[c].get(ds, {}) for c in other}
                dis, nun = [], []
                base = preds[self.cap]
                for i in idx:
                    ps = [preds[c][i]["pred"] for c in other if i in preds[c]]
                    bp = base.get(i, {}).get("pred")
                    dis.append(np.mean([p != bp for p in ps]) if ps and bp is not None else 0.0)
                    nun.append(len(set(ps)) if ps else 1)
                self.per_ds[ds]["sig"]["cap_disagree"] = np.array(dis, float)
                self.per_ds[ds]["sig"]["cap_nuniq"] = np.array(nun, float)
            # calibration (PMC-train caps)
            cal = {c: _load_calib(J(PMCT_DIR[c])) for c in other}
            calmap = {c: {r["idx"]: r.get("pred") for r in cal[c]} for c in other}
            base = calmap[self.cap]
            dis, nun = [], []
            for i in self._calib_idx:
                ps = [calmap[c][i] for c in other if i in calmap[c]]
                bp = base.get(i)
                dis.append(np.mean([p != bp for p in ps]) if ps and bp is not None else 0.0)
                nun.append(len(set(ps)) if ps else 1)
            self.calib["sig"]["cap_disagree"] = np.array(dis, float)
            self.calib["sig"]["cap_nuniq"] = np.array(nun, float)
        except Exception as e:
            print(f"  [cross-cap signal unavailable: {e}]")

    # ---- subset helpers ----
    def pool(self, names):
        names = [d for d in names if d in self.per_ds]
        cat = lambda key: np.concatenate([self.per_ds[d][key] for d in names])
        out = {k: cat(k) for k in ("a7", "a32", "g7", "g32", "PY", "PF", "visY", "visF")}
        out["sig"] = {k: np.concatenate([self.per_ds[d]["sig"][k] for d in names])
                      for k in self.per_ds[names[0]]["sig"]}
        out["names"] = names
        out["ds_of"] = np.concatenate([[d] * len(self.per_ds[d]["a7"]) for d in names])
        return out

    # ---- cost / accuracy scoring for an escalation boolean vector ----
    @staticmethod
    def _flops(P7, g7, PF, g32, include_vit=False, vis7=None, visF=None):
        run7 = 2 * N7 * (P7 + g7)
        run32 = 2 * N32 * (PF + g32)
        if include_vit:
            run7 = run7 + 2 * NVIT * vis7
            run32 = run32 + 2 * NVIT * visF
        return run7, run32

    def score(self, pool, esc, include_vit=False):
        """Given a pooled subset dict and a boolean escalation vector, return metrics."""
        esc = np.asarray(esc, bool)
        a7, a32 = pool["a7"], pool["a32"]
        run7, run32 = self._flops(pool["PY"], pool["g7"], pool["PF"], pool["g32"],
                                  include_vit, pool["visY"], pool["visF"])
        final = np.where(esc, a32, a7)
        backbone = (run7.sum() + run32[esc].sum()) / run32.sum()
        return dict(
            acc=float(final.mean()),
            esc=float(esc.mean()),
            backbone=float(backbone),
            n=int(len(a7)),
            acc7=float(a7.mean()),
            acc32=float(a32.mean()),
        )

    def per_benchmark(self, esc_by_ds, include_vit=False):
        """esc_by_ds: dict ds-> boolean vector aligned to self.per_ds[ds]. Returns per-ds metrics."""
        out = {}
        for ds in self.ds_names:
            d = self.per_ds[ds]
            esc = np.asarray(esc_by_ds[ds], bool)
            run7, run32 = self._flops(d["PY"], d["g7"], d["PF"], d["g32"], include_vit, d["visY"], d["visF"])
            final = np.where(esc, d["a32"], d["a7"])
            out[ds] = dict(acc=float(final.mean()), esc=float(esc.mean()),
                           backbone=float((run7.sum() + run32[esc].sum()) / run32.sum()),
                           acc7=float(d["a7"].mean()), acc32=float(d["a32"].mean()), n=len(d["a7"]))
        return out


# ------------------------------------------------------------------ baselines + anchor
def main():
    print("Loading cap320 cascade data ...", flush=True)
    D = CascadeData("cap320")
    print("datasets:", D.ds_names)
    for subset_name, names in [("ALL-6", ALL6), ("COMPETENT-4", COMPETENT)]:
        P = D.pool(names)
        n = len(P["a7"])
        print(f"\n===== {subset_name}  (n={n}) =====")
        print(f"  always-7B  acc={P['a7'].mean():.4f}")
        print(f"  always-32B acc={P['a32'].mean():.4f}   (parity reference)")
        # always-7B and always-32B backbone
        none = np.zeros(n, bool); allesc = np.ones(n, bool)
        print(f"  always-7B  backbone={D.score(P, none)['backbone']*100:.1f}%   "
              f"always-32B backbone={D.score(P, allesc)['backbone']*100:.1f}%")
        # union (oracle upper bound on accuracy)
        union = np.maximum(P["a7"], P["a32"]).mean()
        comp = float(((P["a7"] == 0) & (P["a32"] == 1)).mean())   # 7B wrong, 32B right (beneficial)
        hurt = float(((P["a7"] == 1) & (P["a32"] == 0)).mean())   # 7B right, 32B wrong (escalation hurts)
        print(f"  oracle-union acc (7B OR 32B right) = {union:.4f}   "
              f"beneficial(7w,32r)={comp:.3f}  harmful(7r,32w)={hurt:.3f}")

    # ---- anchor: reproduce the frozen margin gate (logistic on margin, tau = quantile@error) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cm = D.calib["sig"]["margin"].reshape(-1, 1); cy = D.calib["ok"]
    g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(cm, cy)
    tau = float(np.quantile(g.predict_proba(cm)[:, 1], 1.0 - cy.mean()))
    P6 = D.pool(ALL6)
    pe = g.predict_proba(P6["sig"]["margin"].reshape(-1, 1))[:, 1]
    esc = pe < tau
    m = D.score(P6, esc)
    print(f"\n===== ANCHOR: frozen margin gate (tau={tau:.3f}) on ALL-6 =====")
    print(f"  acc={m['acc']:.4f}  esc={m['esc']*100:.1f}%  backbone={m['backbone']*100:.1f}%")
    print(f"  (must reproduce grid_resolution_tau cap320,cap320: acc~0.572 esc~63% backbone~74%)")


if __name__ == "__main__":
    main()
