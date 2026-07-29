#!/usr/bin/env python3
"""
generator_portfolio.py - OFFLINE test of backlog idea A2: "Generator portfolio via
error-correlation (Markowitz)".

Mechanism (finance -> our cascade). Treat each cheap GENERATOR (model x sampling-temperature)
as an ASSET with a RETURN (its marginal correctness contribution = per-sample accuracy) and a
COVARIANCE (how correlated its ERRORS are with the other generators). Markowitz says: to
maximise return for a fixed budget, combine LOW / NEGATIVELY-correlated assets. Our binding
limit is oracle@N (does a correct answer appear among the drawn candidates?); a cross-model pool
raises oracle precisely BECAUSE the models fail on DIFFERENT questions (low error correlation).
So: given a fixed sample budget B, ALLOCATE B samples across the generators to MAXIMISE oracle@B.

Assets available OFFLINE (no temperature variants exist on disk, so asset == model):
  Lingshu-7B ("lingshu7b"), MedVLThinker-7B ("7b"), InternVL3-8B ("iv3_8b").
Each generator dumped 8 i.i.d. best-of-N samples per question with per-candidate correctness.

Coverage model (a "pass@k"-style with-replacement estimator, applied IDENTICALLY to every
method so the comparison is fair): from the 8 stored labels of generator m on question q, its
per-draw correctness rate is p_mq = mean(sl). Drawing k_m i.i.d. samples, the probability the
answer is NOT covered by generator m is (1-p_mq)^{k_m}; across the pooled generators the
question is covered iff ANY drawn sample is correct:
      oracle_q(alloc) = 1 - prod_m (1 - p_mq)^{k_m},        oracle@B = mean_q oracle_q(alloc).
This is monotone-submodular in the sample multiset, so a GREEDY per-sample allocation (add the
next sample to the generator with the largest marginal coverage gain) is the natural optimiser;
with only 3 assets and B<=16 we ALSO solve it EXACTLY by enumerating every integer allocation
and assert greedy == exact.

Methods compared at each budget B in {2,4,8,16}, per-dataset and pooled:
  * single-best      : the single generator with the highest oracle@B (all B samples on it).
  * uniform          : split B as evenly as possible across the generators (avg over tie-perms).
  * portfolio (ours) : the oracle@B-maximising allocation.
HONESTY: the portfolio allocation is fit on a TRAIN split and scored on a HELD-OUT split
(5-fold CV) so the reported win is not in-sample-optimistic; single-best is likewise fit on
train (best generator chosen on train). We report both the CV-held-out and the in-sample number.

Both-axes reading: B samples ~ B cheap forward passes ~ fixed compute, so a HIGHER oracle@B at
the SAME B is a HIGHER accuracy CEILING at the same compute. A trained verifier then realises
~74-82% of that ceiling (project's measured selection efficiency; not recomputed here).

Markowitz inputs are reported for transparency: per-generator RETURN (per-sample & greedy acc)
and the ERROR-CORRELATION (phi) matrix on the greedy-correct indicator.

Reads ONLY existing artifacts. CPU-only. NO GPU / NO inference. Launch from repo root:
    python3 src/cascade_methods/generator_portfolio.py
"""
import os, json, itertools
import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
DUMP = "ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_{tag}.json"
NAME = {"lingshu7b": "Lingshu-7B", "7b": "MedVLThinker-7B", "iv3_8b": "InternVL3-8B"}
BUDGETS = [2, 4, 8, 16]
KFOLDS = 5
SEED = 0

# datasets where all THREE generators are present (apples-to-apples portfolio)
DS3 = ["kvasir_open", "radimagenet_open", "vqa_rad_open"]
TAGS3 = ["lingshu7b", "7b", "iv3_8b"]
# supplementary 2-generator case (MedVLThinker-7B never ran pathvqa)
DS2 = ["pathvqa_open"]
TAGS2 = ["lingshu7b", "iv3_8b"]


# ---------------------------------------------------------------- loading
def load_dataset(ds, tags):
    """Return (idx_list, {tag: {"p": per-sample-rate array, "g": greedy-ok array}}) aligned on
    the idx common to ALL requested generators."""
    per = {}
    for t in tags:
        p = J(DUMP.format(ds=ds, tag=t))
        if not os.path.exists(p):
            return None, None
        per[t] = {r["idx"]: r for r in json.load(open(p))}
    idx = sorted(set.intersection(*[set(d.keys()) for d in per.values()]))
    if not idx:
        return None, None
    out = {}
    for t in tags:
        prate, greedy = [], []
        for i in idx:
            r = per[t][i]
            sl = [0 if x in (None, -1) else int(x) for x in r["sl"]]
            prate.append(float(np.mean(sl)) if sl else 0.0)
            greedy.append(int(r["greedy_ok"]))
        out[t] = dict(p=np.array(prate, float), g=np.array(greedy, int))
    return idx, out


# ---------------------------------------------------------------- coverage / oracle
def oracle_at(P, alloc):
    """P: list of per-sample-rate arrays (one per generator, same length n). alloc: tuple k_m.
    Returns mean-over-questions oracle coverage = mean(1 - prod_m (1-p_m)^{k_m})."""
    surv = np.ones_like(P[0])
    for p, k in zip(P, alloc):
        if k:
            surv = surv * np.power(1.0 - p, k)
    return float(np.mean(1.0 - surv))


def compositions(B, M):
    """all integer tuples length M, nonneg, summing to B."""
    if M == 1:
        yield (B,); return
    for first in range(B + 1):
        for rest in compositions(B - first, M - 1):
            yield (first,) + rest


def best_alloc(P, B):
    """exact oracle@B-maximising allocation via enumeration (M small, B<=16)."""
    best, ba = -1.0, None
    for a in compositions(B, len(P)):
        v = oracle_at(P, a)
        if v > best:
            best, ba = v, a
    return ba, best


def greedy_alloc(P, B):
    """greedy submodular: add each of B samples to the generator with the max marginal gain."""
    M = len(P); k = [0] * M
    surv = np.ones_like(P[0])                       # prod so far of (1-p)^{k}
    for _ in range(B):
        gains = [float(np.mean(surv * P[m])) for m in range(M)]   # coverage added by 1 more from m
        m = int(np.argmax(gains)); k[m] += 1; surv = surv * (1.0 - P[m])
    return tuple(k), oracle_at(P, k)


def uniform_alloc_variants(B, M):
    """balanced integer allocations: floor(B/M) each, remainder spread one-each over any subset;
    return every tie-permutation so we can average (canonical 'uniform pool')."""
    base = B // M; rem = B % M
    seen = set()
    for extra in itertools.combinations(range(M), rem):
        a = [base] * M
        for e in extra:
            a[e] += 1
        seen.add(tuple(a))
    return list(seen)


def uniform_oracle(P, B):
    variants = uniform_alloc_variants(B, len(P))
    return float(np.mean([oracle_at(P, a) for a in variants])), variants[0]


# ---------------------------------------------------------------- Markowitz inputs
def phi_matrix(G):
    """G: dict tag-> greedy-ok int array (aligned). phi coeff between ERROR indicators e=1-g."""
    tags = list(G); M = len(tags); mat = np.eye(M)
    for a in range(M):
        for b in range(a + 1, M):
            ea = 1 - G[tags[a]]; eb = 1 - G[tags[b]]
            n11 = int(np.sum((ea == 1) & (eb == 1))); n00 = int(np.sum((ea == 0) & (eb == 0)))
            n10 = int(np.sum((ea == 1) & (eb == 0))); n01 = int(np.sum((ea == 0) & (eb == 1)))
            den = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
            phi = (n11 * n00 - n10 * n01) / den if den > 0 else 0.0
            mat[a, b] = mat[b, a] = phi
    return tags, mat


# ---------------------------------------------------------------- CV held-out evaluation
def cv_eval(P, tags, B, rng):
    """5-fold: fit alloc (portfolio) / pick generator (single-best) on TRAIN folds, score on the
    held-out fold. uniform needs no fit. Returns dict of held-out oracle@B per method."""
    n = len(P[0]); idx = rng.permutation(n); folds = np.array_split(idx, KFOLDS)
    acc = {"portfolio": [], "single_best": [], "uniform": []}
    for f in range(KFOLDS):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(KFOLDS) if g != f])
        Ptr = [p[tr] for p in P]; Pte = [p[te] for p in P]
        a_star, _ = best_alloc(Ptr, B)                         # fit portfolio on train
        acc["portfolio"].append(oracle_at(Pte, a_star))
        # single-best generator chosen on train (all B on the train-best generator)
        sb = int(np.argmax([oracle_at([Ptr[m]], (B,)) for m in range(len(P))]))
        onehot = tuple(B if m == sb else 0 for m in range(len(P)))
        acc["single_best"].append(oracle_at(Pte, onehot))
        u_val, _ = uniform_oracle(Pte, B)                      # uniform: no fit
        acc["uniform"].append(u_val)
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------- per-configuration report
def analyse(label, P, tags, G):
    """P: list of rate arrays; tags: generator tags aligned to P; G: dict tag->greedy array."""
    n = len(P[0]); rng = np.random.default_rng(SEED)
    print(f"\n{'='*100}\n{label}   (n={n}, generators: {', '.join(NAME[t] for t in tags)})\n{'='*100}")

    # --- Markowitz inputs
    ret_ps = {t: float(np.mean(P[i])) for i, t in enumerate(tags)}
    ret_g = {t: float(np.mean(G[t])) for t in tags}
    print("  RETURN per asset (generator):")
    for t in tags:
        print(f"    {NAME[t]:<16} per-sample-acc={ret_ps[t]:.3f}   greedy-acc={ret_g[t]:.3f}")
    ptags, phi = phi_matrix({t: G[t] for t in tags})
    print("  ERROR-CORRELATION (phi) matrix  [lower/negative = more de-correlated = better to pool]:")
    print("            " + "".join(f"{NAME[t][:10]:>12}" for t in ptags))
    for a, t in enumerate(ptags):
        print(f"    {NAME[t][:10]:<10}" + "".join(f"{phi[a, b]:>12.3f}" for b in range(len(ptags))))
    mean_offdiag = float(np.mean([phi[a, b] for a in range(len(ptags)) for b in range(len(ptags)) if a < b]))
    print(f"    mean off-diagonal error-phi = {mean_offdiag:.3f}")

    # absolute ceilings for scale
    full_pool = oracle_at(P, tuple([8] * len(P)))               # every generator's full 8 samples
    print(f"  reference ceilings:  best single generator oracle@8="
          f"{max(oracle_at([P[m]], (8,)) for m in range(len(P))):.3f}   "
          f"FULL pool (8 each, {8*len(P)} samples) oracle={full_pool:.3f}")

    # --- budget sweep
    print(f"\n  {'B':>3}  {'single-best':>26}{'uniform':>12}{'portfolio (ours)':>28}   "
          f"{'held-out oracle@B (5-fold)':>34}")
    print(f"  {'':>3}  {'gen  in-sample':>26}{'in-sample':>12}{'alloc  in-sample':>28}   "
          f"{'single  uniform  portf   Δ vs unif':>34}")
    print("  " + "-" * 108)
    results = {}
    for B in BUDGETS:
        # in-sample optima
        sb_vals = [oracle_at([P[m]], (B,)) for m in range(len(P))]
        sb = int(np.argmax(sb_vals)); sb_in = sb_vals[sb]
        u_in, _ = uniform_oracle(P, B)
        a_star, pf_in = best_alloc(P, B)
        a_greedy, pf_g = greedy_alloc(P, B)
        assert abs(pf_in - pf_g) < 1e-9, f"greedy!=exact @B={B}: {a_greedy}{pf_g} vs {a_star}{pf_in}"
        # held-out
        ho = cv_eval(P, tags, B, np.random.default_rng(SEED))
        alloc_str = "/".join(f"{NAME[tags[m]][:3]}{a_star[m]}" for m in range(len(P)))
        d_unif = ho["portfolio"] - ho["uniform"]
        print(f"  {B:>3}  {NAME[tags[sb]][:6]:>8} {sb_in:>7.3f}   {u_in:>9.3f}   "
              f"{alloc_str:>16} {pf_in:>7.3f}   "
              f"{ho['single_best']:>6.3f}  {ho['uniform']:>6.3f}  {ho['portfolio']:>6.3f}  {d_unif:>+7.3f}")
        results[B] = dict(
            single_best=dict(gen=NAME[tags[sb]], insample=sb_in, heldout=ho["single_best"]),
            uniform=dict(insample=u_in, heldout=ho["uniform"]),
            portfolio=dict(alloc={NAME[tags[m]]: a_star[m] for m in range(len(P))},
                           insample=pf_in, heldout=ho["portfolio"], greedy_alloc=a_greedy),
            delta_portfolio_vs_uniform_heldout=d_unif,
            delta_portfolio_vs_single_heldout=ho["portfolio"] - ho["single_best"],
        )
    return dict(n=n, generators=[NAME[t] for t in tags],
                return_persample={NAME[t]: ret_ps[t] for t in tags},
                return_greedy={NAME[t]: ret_g[t] for t in tags},
                error_phi={NAME[ptags[a]]: {NAME[ptags[b]]: float(phi[a, b]) for b in range(len(ptags))}
                           for a in range(len(ptags))},
                mean_offdiag_error_phi=mean_offdiag,
                ceiling_full_pool=full_pool, budgets=results)


def build(ds_list, tags):
    """concatenate per-sample-rate arrays + greedy arrays across a list of datasets (aligned)."""
    P = [[] for _ in tags]; G = {t: [] for t in tags}; perds = {}
    for ds in ds_list:
        idx, d = load_dataset(ds, tags)
        if idx is None:
            continue
        perds[ds] = (idx, d)
        for i, t in enumerate(tags):
            P[i].append(d[t]["p"]); G[t].append(d[t]["g"])
    if not perds:
        return None, None, None
    P = [np.concatenate(x) for x in P]
    G = {t: np.concatenate(G[t]) for t in tags}
    return P, G, perds


def main():
    print("#" * 100)
    print("GENERATOR PORTFOLIO via ERROR-CORRELATION (Markowitz)  -  OFFLINE, CPU, no inference")
    print("  assets = cheap generators; budget B = total samples; objective = oracle@B (answer coverage)")
    print("#" * 100)
    OUT = {"method": "generator_portfolio_markowitz",
           "coverage_model": "oracle@B = mean_q(1 - prod_m (1-p_mq)^{k_m}); p_mq=mean(sl_mq); "
                             "with-replacement pass@k estimator applied identically to all methods",
           "note": "no temperature variants on disk -> assets are the 3 models; "
                   "portfolio allocation fit on train fold, scored held-out (5-fold CV)",
           "budgets": BUDGETS, "configs": {}}

    # --- per-dataset (all-3-generator datasets)
    for ds in DS3:
        P, G, _ = build([ds], TAGS3)
        if P is None:
            continue
        OUT["configs"][ds] = analyse(f"PER-DATASET: {ds}", P, TAGS3, G)

    # --- pooled over the 3 all-generator datasets
    P, G, _ = build(DS3, TAGS3)
    OUT["configs"]["POOLED_3ds_3gen"] = analyse("POOLED (kvasir+radimagenet+vqa_rad, 3 generators)",
                                                P, TAGS3, G)

    # --- supplementary: pathvqa, 2 generators (no MedVLThinker-7B run)
    P2, G2, _ = build(DS2, TAGS2)
    if P2 is not None:
        OUT["configs"]["pathvqa_open_2gen"] = analyse("SUPPLEMENTARY: pathvqa_open (2 generators)",
                                                      P2, TAGS2, G2)

    os.makedirs(J("results/cascade_methods/artifacts"), exist_ok=True)
    outp = J("results/cascade_methods/artifacts/generator_portfolio.json")
    json.dump(OUT, open(outp, "w"), indent=1)

    # ---- headline
    print("\n" + "#" * 100)
    print("HEADLINE  (held-out oracle@B = accuracy ceiling at fixed compute B)")
    print("#" * 100)
    for cfg in ["POOLED_3ds_3gen"] + DS3 + (["pathvqa_open_2gen"] if P2 is not None else []):
        r = OUT["configs"].get(cfg)
        if not r:
            continue
        print(f"\n  {cfg}  (n={r['n']}):")
        for B in BUDGETS:
            b = r["budgets"][B]
            al = ", ".join(f"{g.split('-')[0]}:{k}" for g, k in b["portfolio"]["alloc"].items() if k)
            print(f"    B={B:>2}: single-best({b['single_best']['gen'].split('-')[0]})={b['single_best']['heldout']:.3f}"
                  f"  uniform={b['uniform']['heldout']:.3f}  portfolio[{al}]={b['portfolio']['heldout']:.3f}"
                  f"  (Δvs-uniform={b['delta_portfolio_vs_uniform_heldout']:+.3f},"
                  f" Δvs-single={b['delta_portfolio_vs_single_heldout']:+.3f})")
    print(f"\n[dump] {outp}")


if __name__ == "__main__":
    main()
