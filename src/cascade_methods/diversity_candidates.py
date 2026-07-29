#!/usr/bin/env python3
"""
diversity_candidates.py - OFFLINE test of idea A1 "Diversity-maximized candidate set" (DPP/MMR).

BACKLOG: results/cascade_methods/METHOD_IDEAS_BACKLOG.md  A1 (TOP-5 #2).
The project's #1 binding limit is CANDIDATE QUALITY: oracle@N (does a correct answer exist among the
N samples) is the accuracy ceiling, and iid temperature sampling produces REDUNDANT candidates.
Idea A1: select/measure a DIVERSE subset of candidates (DPP/MAP ~ greedy MMR / farthest-first over
answer-string similarity) instead of iid, to reach the oracle ceiling in FEWER samples (raise the RATE
of oracle@k growth) and to quantify how much of the 8 iid samples are wasted duplicates.

WHAT THIS CAN AND CANNOT TEST OFFLINE (be honest):
  * CANNOT raise oracle@8 over the SAME 8 candidates -- reordering a fixed set never adds a new answer.
    FULL diversity (a HIGHER oracle@8) needs diverse GENERATION (different prompts/temps/models) = a GPU
    job. That script is written & queued separately: src/cascade_methods/diversity_generate_gpu.py.
  * CAN test whether diversity-aware SELECTION over the existing 8 iid candidates reaches the correct
    answer in fewer samples than iid draw-order or a random subset (i.e. cuts the effective N needed to
    hit the oracle ceiling), and CAN quantify the redundancy headroom (distinct-answer count => the
    max sample reduction a perfectly-diverse generator could buy at the SAME oracle@8 coverage).

Reads ONLY existing artifacts (transfer_dump_{ds}_{tag}.json: per-q preds[8], sl[8]=per-sample judge_ok,
scores[8]=verifier). CPU only. No inference. No fabricated numbers.
Launch from repo root:  python3 src/cascade_methods/diversity_candidates.py
Writes: results/cascade_methods/artifacts/diversity_candidates.json
"""
import os, json, re, math
import numpy as np
from itertools import combinations

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
ADAPTER = "ckpts/train/lora_verifier_pooled4"
# family -> (vtag used in transfer_dump filename)
FAM = {"Lingshu": "lingshu7b", "MedVLThinker": "7b", "InternVL3": "iv3_8b"}
DSETS = ["vqa_rad_open", "slake_open", "pathvqa_open", "kvasir_open", "radimagenet_open"]
JAC_MERGE = 0.5   # token-Jaccard >= this merges two answers into one near-duplicate cluster

# ---------------------------------------------------------------- string similarity (label-blind)
_PUNCT = re.compile(r"[^a-z0-9 ]+")
def norm(s):
    s = (s or "").strip().lower()
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def toks(s):
    return set(norm(s).split())

def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)

def dist_exact(pn):
    """distance matrix: 0 if identical normalized string else 1."""
    n = len(pn); D = np.ones((n, n))
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            d = 0.0 if pn[i] == pn[j] else 1.0
            D[i, j] = D[j, i] = d
    return D

def dist_jac(pt):
    """distance matrix: 1 - token-Jaccard."""
    n = len(pt); D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - jaccard(pt[i], pt[j])
            D[i, j] = D[j, i] = d
    return D

# ---------------------------------------------------------------- clustering / redundancy
def n_distinct_exact(pn):
    return len(set(pn))

def n_clusters_jac(pt, thr=JAC_MERGE):
    """union-find merge of answers with token-Jaccard >= thr; returns #clusters."""
    n = len(pt); parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(pt[i], pt[j]) >= thr:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})

# ---------------------------------------------------------------- diversity ordering (MMR / DPP-MAP)
def medoid(D):
    """label-blind anchor = the most central answer (min total distance) ~ the modal/consensus answer."""
    return int(np.argmin(D.sum(axis=1)))

def farthest_first(D, seed):
    """Greedy farthest-first traversal (MMR with lambda=0 relevance == DPP-MAP proxy):
    start at `seed`, then repeatedly add the remaining item with the MAX min-distance to the selected
    set (defers near-duplicates). Ties broken by original draw order (lowest index first)."""
    n = D.shape[0]
    order = [seed]; remaining = [i for i in range(n) if i != seed]
    while remaining:
        best, bestval = None, -1.0
        for i in remaining:
            mind = min(D[i, j] for j in order)
            if mind > bestval + 1e-12:      # strict -> ties keep draw order (lower index seen first)
                bestval, best = mind, i
        order.append(best); remaining.remove(best)
    return order

# ---------------------------------------------------------------- oracle@k for an ordering
def cover_curve(sl, order):
    """oracle indicator at each k=1..n for a given ordering: 1 if any of first k picks is correct."""
    n = len(order); out = []; hit = 0
    for k in range(n):
        if sl[order[k]] == 1: hit = 1
        out.append(hit)
    return out  # length n, out[k-1] = oracle@k

def rand_cover_curve(c, n):
    """closed-form E[oracle@k] over a uniformly random k-subset: 1 - C(n-c,k)/C(n,k)."""
    out = []
    for k in range(1, n + 1):
        if c <= 0: out.append(0.0)
        elif n - c < k: out.append(1.0)
        else: out.append(1.0 - math.comb(n - c, k) / math.comb(n, k))
    return out

def kreach_order(sl, order):
    """1-based position of first correct answer in the ordering; None if none correct."""
    for k, i in enumerate(order):
        if sl[i] == 1: return k + 1
    return None

def kreach_random_expected(c, n):
    """E[position of first success] over a random permutation with c successes = (n+1)/(c+1)."""
    return (n + 1) / (c + 1) if c > 0 else None

# ---------------------------------------------------------------- load
def load(vtag, ds):
    p = J(os.path.join(ADAPTER, f"transfer_dump_{ds}_{vtag}.json"))
    if not os.path.exists(p): return None
    rows = []
    for r in json.load(open(p)):
        sl = [0 if x is None or x == -1 else int(x) for x in r["sl"]]
        preds = list(r["preds"]); scores = list(r.get("scores", []))
        m = min(len(sl), len(preds))
        if m < 2: continue
        rows.append(dict(sl=sl[:m], preds=preds[:m], scores=scores[:m]))
    return rows

# ---------------------------------------------------------------- per-dataset analysis
def analyze(rows):
    N = 8
    rows = [r for r in rows if len(r["sl"]) == N]      # keep the clean 8-sample questions
    n = len(rows)
    # redundancy
    dist_ex, clus_jac, dup_frac, pair_sim_ex, pair_sim_jac = [], [], [], [], []
    for r in rows:
        pn = [norm(p) for p in r["preds"]]; pt = [toks(p) for p in r["preds"]]
        dist_ex.append(n_distinct_exact(pn))
        clus_jac.append(n_clusters_jac(pt))
        # largest exact-duplicate group fraction (mode share)
        from collections import Counter
        dup_frac.append(max(Counter(pn).values()) / N)
        se = [1.0 if pn[i] == pn[j] else 0.0 for i, j in combinations(range(N), 2)]
        sj = [jaccard(pt[i], pt[j]) for i, j in combinations(range(N), 2)]
        pair_sim_ex.append(np.mean(se)); pair_sim_jac.append(np.mean(sj))

    # oracle@k curves for the four orderings
    curves = {k: np.zeros(N) for k in ("draw", "random", "mmr_exact", "mmr_jac")}
    # k_reach on oracle@8=1 questions
    kr = {k: [] for k in ("draw", "random", "mmr_exact", "mmr_jac")}
    conc_c = []          # #correct among the 8, for oracle@8=1 questions
    for r in rows:
        sl = r["sl"]; c = sum(sl)
        pn = [norm(p) for p in r["preds"]]; pt = [toks(p) for p in r["preds"]]
        De = dist_exact(pn); Dj = dist_jac(pt)
        o_draw = list(range(N))
        o_me = farthest_first(De, medoid(De))
        o_mj = farthest_first(Dj, medoid(Dj))
        curves["draw"]      += np.array(cover_curve(sl, o_draw))
        curves["mmr_exact"] += np.array(cover_curve(sl, o_me))
        curves["mmr_jac"]   += np.array(cover_curve(sl, o_mj))
        curves["random"]    += np.array(rand_cover_curve(c, N))
        if c > 0:            # oracle@8 == 1
            conc_c.append(c)
            kr["draw"].append(kreach_order(sl, o_draw))
            kr["mmr_exact"].append(kreach_order(sl, o_me))
            kr["mmr_jac"].append(kreach_order(sl, o_mj))
            kr["random"].append(kreach_random_expected(c, N))
    for k in curves: curves[k] = (curves[k] / n).tolist()

    def aulc(cv):   # area under oracle@k curve, k=1..N, normalized to [0,1]
        return float(np.mean(cv))
    oracle8 = curves["draw"][-1]     # identical across orderings by construction
    n_solvable = len(conc_c)
    md = float(np.mean(dist_ex))
    return dict(
        n=n, oracle_at_8=oracle8, n_oracle_hit=n_solvable,
        redundancy=dict(
            mean_distinct_exact=md, mean_clusters_jac=float(np.mean(clus_jac)),
            mean_mode_share=float(np.mean(dup_frac)),
            mean_pairwise_sim_exact=float(np.mean(pair_sim_ex)),
            mean_pairwise_sim_jaccard=float(np.mean(pair_sim_jac)),
            pct_redundant_samples=float(1 - md / 8),      # share of the 8 that are exact duplicates
        ),
        oracle_at_k=curves,
        aulc=dict((k, aulc(v)) for k, v in curves.items()),
        kreach=dict((k, float(np.mean(kr[k]))) for k in kr if kr[k]),
        concentration=dict(
            mean_correct_of_8=float(np.mean(conc_c)) if conc_c else None,
            pct_singleton_correct=float(np.mean([c == 1 for c in conc_c])) if conc_c else None,
            pct_minority_correct=float(np.mean([c <= 3 for c in conc_c])) if conc_c else None,
        ),
        # idealized headroom: a perfectly-diverse generator reaches the SAME oracle@8 (all distinct
        # answers incl. the correct one) with mean_distinct samples => max sample reduction at fixed
        # coverage. UPPER BOUND; realizing it needs diverse GENERATION (GPU) which may add distractors.
        headroom_diverse_generation=dict(
            mean_samples_needed_for_same_oracle=md,
            max_pct_fewer_samples_at_same_oracle=float(1 - md / 8),
        ),
    )

def pool(list_of_rows):
    return analyze([r for rows in list_of_rows for r in rows])

def fmt_pct(x): return f"{x*100:5.1f}%"

def main():
    print("=" * 108)
    print("DIVERSITY-MAXIMIZED CANDIDATE SET (A1) - OFFLINE on the existing 8 iid candidates. CPU only.")
    print("Q: over the SAME 8, does diversity-aware SELECTION (MMR/DPP-MAP) reach the oracle in fewer samples")
    print("   than iid draw-order / a random subset? And how redundant are the 8 (=> headroom for diverse GEN)?")
    print("=" * 108)
    out = {"meta": dict(
        n_candidates=8, adapter=ADAPTER,
        orderings=dict(
            draw="iid draw order (first-k) = what you get by stopping sampling at k",
            random="expected oracle over a uniformly random k-subset (closed form, hypergeometric)",
            mmr_exact="greedy farthest-first (DPP-MAP proxy) over exact-string distance, seeded at the medoid",
            mmr_jac="greedy farthest-first over (1 - token-Jaccard) distance, seeded at the medoid"),
        note="Reordering a FIXED set cannot raise oracle@8; it can only change the RATE oracle@k reaches "
             "that ceiling. Higher oracle@8 needs diverse GENERATION (GPU): diversity_generate_gpu.py."),
        "families": {}}

    for fam, vtag in FAM.items():
        per_ds = {}; fam_rows = []
        for ds in DSETS:
            rows = load(vtag, ds)
            if not rows: continue
            per_ds[ds] = analyze(rows); fam_rows.append(rows)
        if not fam_rows: continue
        pooled = pool(fam_rows)
        out["families"][fam] = dict(per_dataset=per_ds, pooled=pooled)
        print(f"\n############  {fam}  ({vtag})  ############")
        hdr = f"  {'dataset':<16}{'n':>6}{'oracle@8':>10}{'distinct/8':>11}{'redund%':>9}"
        hdr += f"{'or@2 draw':>10}{'rnd':>7}{'mmrEx':>7}{'mmrJac':>7} | {'kreach draw':>12}{'rnd':>6}{'mmrEx':>7}{'mmrJac':>7}"
        print(hdr)
        for ds, a in list(per_ds.items()) + [("POOLED", pooled)]:
            ok = a["oracle_at_k"]; kr = a["kreach"]
            print(f"  {ds:<16}{a['n']:>6}{a['oracle_at_8']:>10.3f}"
                  f"{a['redundancy']['mean_distinct_exact']:>11.2f}"
                  f"{a['redundancy']['pct_redundant_samples']*100:>8.0f}%"
                  f"{ok['draw'][1]:>10.3f}{ok['random'][1]:>7.3f}{ok['mmr_exact'][1]:>7.3f}{ok['mmr_jac'][1]:>7.3f} | "
                  f"{kr.get('draw',float('nan')):>12.2f}{kr.get('random',float('nan')):>6.2f}"
                  f"{kr.get('mmr_exact',float('nan')):>7.2f}{kr.get('mmr_jac',float('nan')):>7.2f}")

    # -------- grand pooled across all families/datasets --------
    all_rows = []
    for fam, vtag in FAM.items():
        for ds in DSETS:
            r = load(vtag, ds)
            if r: all_rows.append(r)
    grand = pool(all_rows)
    out["pooled_all"] = grand

    print("\n" + "=" * 108)
    print("POOLED ACROSS ALL FAMILIES/DATASETS")
    print("=" * 108)
    r = grand["redundancy"]; ok = grand["oracle_at_k"]; kr = grand["kreach"]; cc = grand["concentration"]
    print(f"  n={grand['n']}  oracle@8={grand['oracle_at_8']:.3f}  (n_solvable={grand['n_oracle_hit']})")
    print(f"  REDUNDANCY: mean distinct answers among 8 = {r['mean_distinct_exact']:.2f}  "
          f"({fmt_pct(r['pct_redundant_samples'])} of the 8 iid samples are exact duplicates); "
          f"jaccard-clusters = {r['mean_clusters_jac']:.2f}; mean pairwise jaccard-sim = {r['mean_pairwise_sim_jaccard']:.3f}")
    print(f"  CONCENTRATION (on oracle@8=1): mean #correct/8 = {cc['mean_correct_of_8']:.2f}; "
          f"singleton-correct (exactly 1 of 8 right) = {fmt_pct(cc['pct_singleton_correct'])}; "
          f"minority-correct (<=3 of 8) = {fmt_pct(cc['pct_minority_correct'])}")
    print("\n  oracle@k  (fraction of questions whose first-k selected candidates contain a correct answer):")
    print(f"    {'k':>3}{'draw(iid)':>11}{'random':>9}{'mmr_exact':>11}{'mmr_jac':>10}")
    for k in range(8):
        print(f"    {k+1:>3}{ok['draw'][k]:>11.3f}{ok['random'][k]:>9.3f}{ok['mmr_exact'][k]:>11.3f}{ok['mmr_jac'][k]:>10.3f}")
    a = grand["aulc"]
    print(f"  AULC (mean oracle@k over k=1..8; higher=reaches ceiling faster): "
          f"draw={a['draw']:.3f} random={a['random']:.3f} mmr_exact={a['mmr_exact']:.3f} mmr_jac={a['mmr_jac']:.3f}")
    print(f"  k_reach (mean #samples to FIRST correct, on oracle@8=1 Qs; lower=better): "
          f"draw={kr['draw']:.2f}  random={kr['random']:.2f}  mmr_exact={kr['mmr_exact']:.2f}  mmr_jac={kr['mmr_jac']:.2f}")
    h = grand["headroom_diverse_generation"]
    print(f"  HEADROOM for diverse GENERATION (idealized upper bound): a perfectly-diverse generator would "
          f"reach the SAME oracle@8 with ~{h['mean_samples_needed_for_same_oracle']:.1f} samples instead of 8 "
          f"=> up to {fmt_pct(h['max_pct_fewer_samples_at_same_oracle'])} fewer samples at fixed coverage.")

    os.makedirs(J("results/cascade_methods/artifacts"), exist_ok=True)
    outp = J("results/cascade_methods/artifacts/diversity_candidates.json")
    json.dump(out, open(outp, "w"), indent=1)
    print(f"\n-> {outp}")

if __name__ == "__main__":
    main()
