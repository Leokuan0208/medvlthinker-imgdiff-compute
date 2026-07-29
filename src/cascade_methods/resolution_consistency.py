#!/usr/bin/env python3
"""
resolution_consistency.py - NOVEL training-free cascade signal: VISUAL STABILITY.

Idea (this project's own, not in FrugalGPT/AutoMix/CAR/CP-Router): for a VLM the routing
question is "does the cheap model's answer depend on visual detail it can't resolve?" We already
have the 7B no-think answer at FIVE resolution caps (cap80/160/320/640/fullres). If the answer is
the SAME across caps, it is visually STABLE -> not bottlenecked on resolution -> escalating to the
big model (which mainly buys capacity + fullres) helps less. If it FLIPS across caps, it is
visually FRAGILE -> exactly where the big model helps. This is a DIFFERENT surface than the
single-pass margin (which is one forward pass' confidence), so it can be orthogonal.

Fully OFFLINE from existing ckpts. No new inference. Operating cheap leg = 7B-nt@cap320.
  cheap caps : ckpts/gate_7b_prune/{cap80,cap160,cap320,cap640}/ckpt_<DS>_nothink_norag.jsonl
               ckpts/gate_7b_vllm/ckpt_<DS>_nothink_norag.jsonl   (fullres)
  strong     : ckpts/gate_32b/ckpt_<DS>_think_norag.jsonl
margin def matches the deployed gate: logprob_top1 - logprob_top2 (tau=0.426).
"""
import os, re, json, glob
import numpy as np

PRUNE = "ckpts/gate_7b_prune"          # cap80/160/320/640 subdirs
FULLRES = "ckpts/gate_7b_vllm"         # fullres
STRONG = "ckpts/gate_32b"              # 32B think
TAU = 0.4264123185919304
COMPETENT4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
ALL5 = COMPETENT4 + ["MMMU"]
ALL6 = ALL5 + ["MedXpert-Reasoning", "MedXpert-Understanding"]
CAPS = ["cap80", "cap160", "cap320", "cap640", "fullres"]

def load_jsonl(path):
    m = {}
    if not os.path.exists(path): return m
    for l in open(path):
        if l.strip():
            r = json.loads(l); m[r["idx"]] = r
    return m

def cap_path(cap, ds):
    if cap == "fullres": return os.path.join(FULLRES, f"ckpt_{ds}_nothink_norag.jsonl")
    return os.path.join(PRUNE, cap, f"ckpt_{ds}_nothink_norag.jsonl")

def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0] - v[1]) if len(v) >= 2 else 0.0

def build(ds):
    """Per-sample aligned record over caps + 32B. Returns list of dicts."""
    caps = {c: load_jsonl(cap_path(c, ds)) for c in CAPS}
    strong = load_jsonl(os.path.join(STRONG, f"ckpt_{ds}_think_norag.jsonl"))
    idx = set(caps["cap320"])
    for c in CAPS: idx &= set(caps[c])
    idx &= set(strong)
    out = []
    for i in sorted(idx):
        r320 = caps["cap320"][i]
        preds = {c: caps[c][i]["pred"] for c in CAPS}
        p320 = preds["cap320"]
        # ladder = the 4 OTHER caps; how many agree with cap320's answer
        others = [c for c in CAPS if c != "cap320"]
        n_agree = sum(int(preds[c] == p320) for c in others)          # 0..4
        # neighbor agreement (cheapest two-view: cap160 + cap640 around cap320)
        nbr = int(preds["cap160"] == p320) + int(preds["cap640"] == p320)  # 0..2
        # full 5-way vote: fraction of caps giving the modal answer
        votes = {}
        for c in CAPS: votes[preds[c]] = votes.get(preds[c], 0) + 1
        vote_frac = max(votes.values()) / len(CAPS)
        out.append(dict(idx=i, ds=ds, pred320=p320, ok320=r320["ok"],
                        margin=margin(r320), n_agree=n_agree, nbr=nbr, vote_frac=vote_frac,
                        ok32=strong[i]["ok"], preds=preds))
    return out

def auroc(score, y):
    """Mann-Whitney AUROC. score oriented so higher => label 1 more likely."""
    score = np.asarray(score, float); y = np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks); avg = sums / cnt
    ranks = avg[inv]
    rsum_pos = ranks[:len(pos)].sum()
    return (rsum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

# ---------- build all ----------
REC = {ds: build(ds) for ds in ALL6}
def pool(names): return [r for ds in names for r in REC[ds]]

def report_auroc(rows, label):
    wrong = [1 - r["ok320"] for r in rows]                       # cheap leg WRONG
    recov = [int((r["ok320"] == 0) and (r["ok32"] == 1)) for r in rows]  # escalation helps
    # escalate-scores: higher => more likely wrong/recoverable
    s_margin = [-r["margin"] for r in rows]
    s_nagree = [-r["n_agree"] for r in rows]
    s_nbr    = [-r["nbr"] for r in rows]
    s_vote   = [-r["vote_frac"] for r in rows]
    print(f"\n[{label}]  n={len(rows)}  cheap-wrong={np.mean(wrong):.3f}  recoverable={np.mean(recov):.3f}")
    print(f"  AUROC predicting CHEAP-WRONG:  margin={auroc(s_margin,wrong):.3f}  "
          f"n_agree(5cap)={auroc(s_nagree,wrong):.3f}  nbr(160+640)={auroc(s_nbr,wrong):.3f}  "
          f"vote_frac={auroc(s_vote,wrong):.3f}")
    print(f"  AUROC predicting RECOVERABLE:  margin={auroc(s_margin,recov):.3f}  "
          f"n_agree(5cap)={auroc(s_nagree,recov):.3f}  nbr(160+640)={auroc(s_nbr,recov):.3f}  "
          f"vote_frac={auroc(s_vote,recov):.3f}")
    # orthogonality: among LOW-margin (would-escalate) samples, does stability still separate?
    lowm = [r for r in rows if r["margin"] < TAU]
    if lowm:
        w = [1 - r["ok320"] for r in lowm]
        print(f"  within margin<tau (n={len(lowm)}): n_agree AUROC for cheap-wrong = "
              f"{auroc([-r['n_agree'] for r in lowm], w):.3f}  (orthogonal lift if >0.5)")
    # correlation margin vs n_agree
    m = np.array([r["margin"] for r in rows]); na = np.array([r["n_agree"] for r in rows])
    if m.std() > 0 and na.std() > 0:
        print(f"  corr(margin, n_agree) = {np.corrcoef(m, na)[0,1]:+.3f}")

def cascade_point(rows, escalate_fn):
    """Given a per-row escalate predicate, return (esc_rate, cascade_acc)."""
    esc = np.array([escalate_fn(r) for r in rows], bool)
    final_ok = np.array([(r["ok32"] if e else r["ok320"]) for r, e in zip(rows, esc)])
    return esc.mean(), final_ok.mean()

def frontier_margin(rows, taus):
    return [(t, *cascade_point(rows, lambda r, t=t: r["margin"] < t)) for t in taus]

def main():
    print("=" * 78)
    print("VISUAL-STABILITY (cross-resolution) cascade signal — offline audit")
    print("=" * 78)
    for label, names in [("competent-4", COMPETENT4), ("ALL-5", ALL5), ("ALL-6", ALL6)]:
        report_auroc(pool(names), label)

    # ---- frontier on competent-4: does stability beat / augment margin? ----
    rows = pool(COMPETENT4)
    print("\n" + "=" * 78)
    print("FRONTIER on competent-4 (esc_rate, cascade_acc). Lower esc at >= acc = win.")
    print("=" * 78)
    base_esc, base_acc = cascade_point(rows, lambda r: r["margin"] < TAU)
    allcheap = np.mean([r["ok320"] for r in rows]); allstrong = np.mean([r["ok32"] for r in rows])
    print(f"  always-cheap acc={allcheap:.4f}   always-strong acc={allstrong:.4f}")
    print(f"  DEPLOYED margin@tau=0.426 : esc={base_esc:.3f}  acc={base_acc:.4f}")

    # pure margin frontier (sweep)
    taus = np.quantile([r["margin"] for r in rows], np.linspace(0.02, 0.98, 49))
    fm = frontier_margin(rows, taus)
    # helper: best esc-rate achieving >= target acc
    def best_esc_at_acc(frontier, target):
        ok = [(e, a) for (_, e, a) in frontier if a >= target - 1e-9]
        return min(ok, default=(None, None), key=lambda x: x[0])
    me, ma = best_esc_at_acc([(None, e, a) for (_, e, a) in fm], base_acc)
    print(f"  margin frontier @ acc>={base_acc:.4f}: esc={me:.3f}" if me is not None else "  margin: n/a")

    # candidate stability-augmented gates -------------------------------------
    print("\n  -- stability-augmented rules (sweep) --")
    cand = {}
    # A) UNION: escalate if margin<t OR fragile (n_agree <= k)
    for k in [0, 1, 2, 3]:
        for t in taus:
            e, a = cascade_point(rows, lambda r, t=t, k=k: (r["margin"] < t) or (r["n_agree"] <= k))
            cand.setdefault(f"UNION n_agree<= {k}", []).append((t, e, a))
    # B) RESCUE-DOWN: escalate if margin<t AND fragile (keep cheap when visually stable)
    for k in [4, 3, 2]:  # require near-full stability to KEEP cheap
        for t in taus:
            e, a = cascade_point(rows, lambda r, t=t, k=k: (r["margin"] < t) and (r["n_agree"] < k))
            cand.setdefault(f"RESCUE keep if n_agree>= {k}", []).append((t, e, a))
    # C) pure stability gate (no margin)
    for k in [0, 1, 2, 3]:
        e, a = cascade_point(rows, lambda r, k=k: r["n_agree"] <= k)
        cand.setdefault("PURE n_agree", []).append((k, e, a))

    print(f"\n  At PARITY with deployed acc ({base_acc:.4f}), min escalation rate by rule:")
    print(f"    {'rule':<28}{'esc@acc':>10}{'Δ vs margin':>14}")
    print(f"    {'DEPLOYED margin gate':<28}{me:>10.3f}{0.0:>+14.3f}")
    for name, fr in cand.items():
        e, a = best_esc_at_acc([(None, e, a) for (_, e, a) in fr], base_acc)
        if e is not None:
            print(f"    {name:<28}{e:>10.3f}{e-me:>+14.3f}")
        else:
            print(f"    {name:<28}{'—':>10}{'(cannot reach acc)':>14}")

    # also: can we reach always-strong parity cheaper?
    print(f"\n  At PARITY with always-strong acc ({allstrong:.4f}):")
    me2, _ = best_esc_at_acc([(None, e, a) for (_, e, a) in fm], allstrong)
    print(f"    {'margin gate':<28}{(me2 if me2 is not None else float('nan')):>10.3f}")
    for name, fr in cand.items():
        e, a = best_esc_at_acc([(None, e, a) for (_, e, a) in fr], allstrong)
        s = f"{e:.3f}" if e is not None else "—"
        print(f"    {name:<28}{s:>10}")

    # dump raw frontiers for charting
    out = dict(deployed=dict(esc=base_esc, acc=base_acc),
               always_cheap=allcheap, always_strong=allstrong,
               margin_frontier=[(float(t), float(e), float(a)) for (t, e, a) in fm],
               candidates={k: [(float(x), float(e), float(a)) for (x, e, a) in v] for k, v in cand.items()})
    os.makedirs("results/cascade_methods/artifacts", exist_ok=True)
    json.dump(out, open("results/cascade_methods/artifacts/resolution_consistency.json", "w"), indent=1)
    print("\n  -> results/cascade_methods/artifacts/resolution_consistency.json")

if __name__ == "__main__":
    main()
