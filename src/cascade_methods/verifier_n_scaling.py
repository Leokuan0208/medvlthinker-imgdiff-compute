#!/usr/bin/env python3
"""verifier_n_scaling.py -- OFFLINE (no GPU, no new inference) answer to the design question:

  "Does the trained verifier's benefit keep growing with the number of samples N, and could a
   7B + verifier at higher N MATCH OR BEAT a 32B without ever calling the 32B at test time?"

Everything here is recomputed from checkpoints already on disk. The verifier used is the
CLEAN L1 image-disjoint verifier (`ckpts/train/lora_verifier_disjoint`, named by
`results/cascade_methods/artifacts/verifier_disjoint_retrain_2026-07-30.json`), NEVER the
contaminated `lora_verifier_pooled4` one -- the contaminated verifier memorised 67-73% of the
items it is scored on, so its scaling curve would be meaningless. The contaminated verifier is
carried alongside for contrast ONLY, always labelled.

INPUTS (all pre-existing)
  ckpts/train/lora_verifier_disjoint/transfer_dump_{ds}_lingshu7b.json   clean L1 scores  [8 cands]
  ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_lingshu7b.json    contaminated scores (contrast)
  ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8*.jsonl           8-sample pool + judge labels
  ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc16*.jsonl          16-sample pool + judge labels
  ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.judge.jsonl          TRUE greedy (temp 0) labels
  ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.judge.jsonl          always-32B-direct labels
  results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json     R32 = 3.82
  results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03.json    measured batch-8 bo8

METHOD
  * The 8 (resp. 16) samples are i.i.d. draws from ONE distribution (vLLM SamplingParams
    n=8/n=16, temperature 0.7 -- see runners/run_openvqa_*.sh), so they are exchangeable and a
    uniformly-random N-subset is an unbiased estimator of "what N samples would have given".
  * oracle@N and verifier@N are computed by EXACT combinatorics over all C(pool,N) subsets
    (no Monte-Carlo noise); score ties are broken by averaging over T random tie-break orders.
    self-consistency@N needs the actual subset multiset, so it uses Monte-Carlo.
  * CIs are non-parametric bootstrap over QUESTIONS (the permutation/tie randomness is already
    integrated out analytically).

  python3 src/cascade_methods/verifier_n_scaling.py
  -> results/cascade_methods/artifacts/verifier_n_scaling_2026-08-03.json
"""
import argparse, json, math, os
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
DSETS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
TAG = "lingshu7b"
CK = "ckpts/openvqa/cheap_lingshu7b"
STRONG = "ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.judge.jsonl"
norm = lambda s: str(s).strip().lower()

ap = argparse.ArgumentParser()
ap.add_argument("--clean", default="ckpts/train/lora_verifier_disjoint",
                help="CLEAN L1 image-disjoint verifier (the one this analysis is about)")
ap.add_argument("--contaminated", default="ckpts/train/lora_verifier_pooled4",
                help="contrast only -- never the basis of any conclusion")
ap.add_argument("--nboot", type=int, default=4000)
ap.add_argument("--ntie", type=int, default=128, help="random tie-break orders averaged over")
ap.add_argument("--nmc_sc", type=int, default=2000, help="MC draws for self-consistency@N")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/cascade_methods/artifacts/verifier_n_scaling_2026-08-03.json")
A = ap.parse_args()
RNG = np.random.default_rng(A.seed)

# ----------------------------------------------------------------------------- loaders
def loadj(p):
    p = J(p)
    if not os.path.exists(p):
        return {}
    return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


def judge_map(p):
    return {k: int(v["judge_ok"]) for k, v in loadj(p).items()}


def pool_labels(ds, sc_tag):
    """-> {idx: [judge label per SLOT]} for the sc8 / sc16 i.i.d. pool."""
    sc = loadj(f"{CK}/ckpt_{ds}_{TAG}_{sc_tag}.jsonl")
    exp = loadj(f"{CK}/ckpt_{ds}_{TAG}_{sc_tag}_scexploded.jsonl")
    jud = judge_map(f"{CK}/ckpt_{ds}_{TAG}_{sc_tag}_scexploded.judge.jsonl")
    ans = defaultdict(dict)
    for cid, r in exp.items():
        if cid in jud:
            oi = cid.split("#")[0]
            oi = int(oi) if oi.lstrip("-").isdigit() else oi
            ans[oi][norm(r["modal_pred"])] = jud[cid]
    out = {}
    for i, r in sc.items():
        if i not in ans:
            continue
        lab = [ans[i].get(norm(a)) for a in r["preds"]]
        if any(x is None for x in lab):
            continue
        out[i] = lab
    return out


# ----------------------------------------------------------------- exact subset combinatorics
def _logC(n, k):
    if k < 0 or k > n or n < 0:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def oracle_at_N(labels, N):
    """E[any correct among a uniform N-subset]  -- exact. labels: list of 0/1 per pool slot."""
    M = len(labels); k = int(sum(labels))
    if N > M:
        return None
    # P(no correct) = C(M-k, N)/C(M, N)
    lp = _logC(M - k, N) - _logC(M, N)
    return float(1.0 - (math.exp(lp) if np.isfinite(lp) else 0.0))


from itertools import combinations

_SUBSETS8 = {N: list(combinations(range(8), N)) for N in range(1, 9)}


def exact_curves_8(labels, scores_clean, scores_cont, preds):
    """EXACT expectations over ALL C(8,N) subsets for N=1..8 (only 255 subsets in total), for
    oracle@N, verifier@N (clean + contaminated) and self-consistency@N.

    Ties are handled as the uniform random tie-break they are:
      * verifier -- among the argmax-score slots of the subset, each is equally likely to be the
        one `np.argmax` lands on after a random draw order, so the expectation is the MEAN label
        over the tied top-scoring slots;
      * self-consistency -- among answer strings tied on vote count, all have the same multiplicity,
        so under exchangeability each is equally likely to occur first (= run_openvqa.py's
        modal_pred rule); expectation is the MEAN label over the tied top strings.
    No Monte-Carlo, no sampling noise.
    """
    lab = np.asarray(labels, float)
    sc = np.asarray(scores_clean, float)
    kc = np.asarray(scores_cont, float)
    keys = [norm(p) for p in preds]
    orc, vcl, vct, svt = {}, {}, {}, {}
    for N, subs in _SUBSETS8.items():
        o = v1 = v2 = s = 0.0
        for idx in subs:
            li = lab[list(idx)]
            o += 1.0 if li.max() > 0 else 0.0
            si = sc[list(idx)]
            v1 += li[si == si.max()].mean()
            ki = kc[list(idx)]
            v2 += li[ki == ki.max()].mean()
            c = Counter(keys[j] for j in idx)
            top = max(c.values())
            tied = [k for k, n_ in c.items() if n_ == top]
            s += float(np.mean([lab[next(j for j in idx if keys[j] == k)] for k in tied]))
        m = len(subs)
        orc[N], vcl[N], vct[N], svt[N] = o / m, v1 / m, v2 / m, s / m
    return orc, vcl, vct, svt


# =============================================================== 1. load everything, per dataset
DATA = {}
for ds in DSETS:
    clean = {r["idx"]: r for r in json.load(open(J(f"{A.clean}/transfer_dump_{ds}_{TAG}.json")))}
    cont = {r["idx"]: r for r in json.load(open(J(f"{A.contaminated}/transfer_dump_{ds}_{TAG}.json")))}
    greedy_t0 = judge_map(f"{CK}/ckpt_{ds}_{TAG}.judge.jsonl")
    strong32 = judge_map(STRONG.format(ds=ds))
    p16 = pool_labels(ds, "sc16")
    rows = []
    for i, r in clean.items():
        assert r["preds"] == cont[i]["preds"], f"{ds}/{i}: candidate lists differ"
        assert r["sl"] == cont[i]["sl"], f"{ds}/{i}: judge labels differ"
        assert all(x != -1 for x in r["sl"]), f"{ds}/{i}: unlabelled candidate"
        rows.append({
            "idx": i, "preds": r["preds"], "sl": [int(x) for x in r["sl"]],
            "sc_clean": list(map(float, r["scores"])), "sc_cont": list(map(float, cont[i]["scores"])),
            "greedy_repo": int(r["greedy_ok"]),           # NB: this is the MODAL-of-8 answer, i.e. SC@8
            "greedy_t0": greedy_t0.get(i),                 # true temperature-0 greedy decode
            "strong32": strong32.get(i),
            "sl16": p16.get(i),
        })
    DATA[ds] = rows
    n16 = sum(1 for r in rows if r["sl16"] is not None)
    print(f"{ds:14s} n={len(rows):5d}  with sc16 pool={n16:5d}  32B labels="
          f"{sum(1 for r in rows if r['strong32'] is not None)}", flush=True)

POOLED = [r for ds in DSETS for r in DATA[ds]]
GROUPS = {ds: DATA[ds] for ds in DSETS}
GROUPS["POOLED"] = POOLED

# =============================================================== 2. per-question curves (exact)
NS8 = list(range(1, 9))
NS16 = list(range(1, 17))
for r in POOLED:
    o, v1, v2, s = exact_curves_8(r["sl"], r["sc_clean"], r["sc_cont"], r["preds"])
    r["oracle"], r["ver_clean"], r["ver_cont"], r["sc_vote"] = o, v1, v2, s
    r["sample1"] = float(np.mean(r["sl"]))       # E[label of one random temp-0.7 sample]
    # deterministic first-max tie-break -- the convention verifier_disjoint_measure.py used, kept
    # only so the published N=8 cell reproduces to the digit
    r["ver_clean_argmaxfirst"] = float(r["sl"][int(np.argmax(r["sc_clean"]))])
    if r["sl16"] is not None:
        r["oracle16"] = {N: oracle_at_N(r["sl16"], N) for N in NS16}
print("per-question exact curves done", flush=True)


def mean(rows, key, N=None, sub=None):
    vals = [(r[key][N] if N is not None else r[key]) for r in rows if (sub is None or r[sub] is not None)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def boot_ci(rows, fn, nboot=None, seed=0):
    nboot = nboot or A.nboot
    rng = np.random.default_rng(seed)
    n = len(rows); arr = np.array([fn(r) for r in rows], float)
    ok = ~np.isnan(arr)
    arr = arr[ok]; n = len(arr)
    out = np.empty(nboot)
    for b in range(nboot):
        out[b] = arr[rng.integers(0, n, n)].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


# =============================================================== 3. the two curves
CURVES = {}
for g, rows in GROUPS.items():
    n = len(rows)
    base_repo = mean(rows, "greedy_repo")
    base_t0 = mean(rows, "greedy_t0", sub="greedy_t0")
    s32 = mean(rows, "strong32", sub="strong32")
    rows16 = [r for r in rows if r["sl16"] is not None]
    tab = {}
    for N in NS8:
        orc = mean(rows, "oracle", N)
        vc = mean(rows, "ver_clean", N)
        vk = mean(rows, "ver_cont", N)
        sv = mean(rows, "sc_vote", N)
        row = {
            "oracle_at_N": orc,
            "verifier_clean_at_N": vc,
            "verifier_contaminated_at_N": vk,
            "self_consistency_at_N": sv,
            # conversion, repo convention: baseline = the dump's `greedy_ok` (= modal-of-8)
            "conversion_clean_vs_repo_greedy": (vc - base_repo) / (orc - base_repo) if orc > base_repo else None,
            "conversion_contaminated_vs_repo_greedy": (vk - base_repo) / (orc - base_repo) if orc > base_repo else None,
            # conversion against the TRUE temp-0 greedy decode (an N-independent baseline)
            "conversion_clean_vs_true_greedy": (vc - base_t0) / (orc - base_t0) if (base_t0 is not None and orc > base_t0) else None,
            # selection efficiency = P(pick a correct answer | a correct answer is in the N-subset)
            "sel_eff_clean": vc / orc if orc else None,
            "sel_eff_contaminated": vk / orc if orc else None,
        }
        if N in (1, 2, 4, 8):
            row["ci_oracle"] = boot_ci(rows, lambda r: r["oracle"][N], seed=100 + N)
            row["ci_verifier_clean"] = boot_ci(rows, lambda r: r["ver_clean"][N], seed=200 + N)
        tab[N] = row
    o16 = {N: mean(rows16, "oracle16", N) for N in NS16} if rows16 else None
    CURVES[g] = {
        "n_questions": n,
        "n_with_sc16": len(rows16),
        "baselines": {
            "greedy_repo_dump_field_is_modal_of_8": base_repo,
            "true_greedy_temp0": base_t0,
            "one_random_sample_temp07": mean(rows, "sample1"),
            "always_32B_direct": s32,
        },
        "curve_from_sc8_pool": tab,
        "measured_oracle_from_sc16_pool": o16,
    }

# ------- harness validation: reproduce the published clean-L1 N=8 cells exactly
V = CURVES["POOLED"]["curve_from_sc8_pool"][8]
PUB = {"verifier_clean": 0.4853, "oracle_at_8": 0.6260, "greedy": 0.4495, "conversion": 0.2029,
       "verifier_contaminated": 0.5535}
GBASE = CURVES["POOLED"]["baselines"]["greedy_repo_dump_field_is_modal_of_8"]
VAF = float(np.mean([r["ver_clean_argmaxfirst"] for r in POOLED]))
VALID = {
    "published_source": "results/cascade_methods/artifacts/verifier_disjoint_retrain_2026-07-30.json"
                        " /levels/L1_image_disjoint/selection_stage/POOLED",
    "verifier_clean_at_8_argmax_first_tiebreak": {"recomputed": VAF, "published": PUB["verifier_clean"]},
    "conversion_at_8_argmax_first_tiebreak": {"recomputed": (VAF - GBASE) / (V["oracle_at_N"] - GBASE),
                                              "published": PUB["conversion"]},
    "oracle_at_8": {"recomputed": V["oracle_at_N"], "published": PUB["oracle_at_8"]},
    "greedy": {"recomputed": GBASE, "published": PUB["greedy"]},
    "verifier_contaminated_at_8": {"recomputed": V["verifier_contaminated_at_N"],
                                   "published": PUB["verifier_contaminated"]},
    "tie_break_note": (
        "the published cell used np.argmax, which on a tie silently takes the FIRST candidate; the "
        "clean verifier's dumped scores are rounded to 5 dp and many candidates are literally the "
        "same answer string, so exact ties are common. This analysis resolves ties as the uniform "
        "random choice they actually are, which is why its N=8 clean verifier reads "
        f"{V['verifier_clean_at_N']:.4f} (conversion {V['conversion_clean_vs_repo_greedy']:.4f}) "
        f"instead of {VAF:.4f} (conversion {(VAF - GBASE) / (V['oracle_at_N'] - GBASE):.4f}). The "
        "difference is 0.0012 of accuracy and does not touch any conclusion."),
    "unbiased_tiebreak_values_used_throughout": {
        "verifier_clean_at_8": V["verifier_clean_at_N"],
        "conversion_at_8": V["conversion_clean_vs_repo_greedy"],
    },
}
VALID["max_abs_diff"] = max(abs(v["recomputed"] - v["published"]) for v in VALID.values()
                            if isinstance(v, dict) and "recomputed" in v)
print(f"\nharness validation vs published clean-L1 N=8 cells (matched tie convention): "
      f"max |diff| = {VALID['max_abs_diff']:.5f}", flush=True)

# =============================================================== 4. coverage extrapolation
# Zero-inflated beta-binomial: question i is either UNREACHABLE (p_i = 0, weight pi0) or has a
# latent per-sample success rate p_i ~ Beta(a,b). k_i correct slots out of M ~ Binomial(M, p_i).
#   coverage(N) = (1 - pi0) * (1 - E_Beta[(1-p)^N]) = (1 - pi0) * (1 - B(a, b+N)/B(a,b))
def zibb_nll(theta, k, M):
    la, lb, lz = theta
    a, b = math.exp(la), math.exp(lb)
    pi0 = 1.0 / (1.0 + math.exp(-lz))
    lc = gammaln(M + 1) - gammaln(k + 1) - gammaln(M - k + 1)
    lbb = lc + betaln(a + k, b + M - k) - betaln(a, b)
    ll = np.where(k == 0, np.logaddexp(math.log(pi0 + 1e-300), math.log(1 - pi0 + 1e-300) + lbb),
                  math.log(1 - pi0 + 1e-300) + lbb)
    return -float(np.sum(ll))


def fit_zibb(k, M, zero_inflated=True):
    best = None
    for a0 in (-1.5, -0.5, 0.5):
        for b0 in (-0.5, 0.5, 1.5):
            for z0 in ((-2.0, 0.0) if zero_inflated else (-30.0,)):
                x0 = np.array([a0, b0, z0], float)
                if zero_inflated:
                    r = minimize(zibb_nll, x0, args=(k, M), method="Nelder-Mead",
                                 options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
                else:
                    r = minimize(lambda t: zibb_nll(np.array([t[0], t[1], -30.0]), k, M), x0[:2],
                                 method="Nelder-Mead", options={"maxiter": 4000})
                    r = type("R", (), {"x": np.array([r.x[0], r.x[1], -30.0]), "fun": r.fun})
                if best is None or r.fun < best.fun:
                    best = r
    a, b = math.exp(best.x[0]), math.exp(best.x[1])
    pi0 = 1.0 / (1.0 + math.exp(-best.x[2]))
    return a, b, pi0, float(best.fun)


def zibb_coverage(a, b, pi0, N):
    return float((1 - pi0) * (1 - math.exp(betaln(a, b + N) - betaln(a, b))))


def plugin_coverage(k, M, N):
    """Empirical plug-in with p_hat = k/M -- a hard LOWER bound: it forces every question with
    0 correct slots to stay unreachable forever, so it saturates at the observed coverage@M."""
    p = np.asarray(k, float) / M
    return float(np.mean(1 - (1 - p) ** N))


COV = {}
for g, rows in GROUPS.items():
    rows16 = [r for r in rows if r["sl16"] is not None]
    k8 = np.array([int(sum(r["sl"])) for r in rows], int)
    k16 = np.array([int(sum(r["sl16"])) for r in rows16], int) if rows16 else None
    entry = {"n8": len(k8), "n16": len(k16) if k16 is not None else 0}

    # (a) fit on the 8-sample pool, PREDICT 16, check against the measured 16-sample pool
    a8, b8, z8, _ = fit_zibb(k8, 8)
    pred16_from8 = zibb_coverage(a8, b8, z8, 16)
    meas16 = mean(rows16, "oracle16", 16) if rows16 else None
    entry["validation_fit8_predict16"] = {
        "fit": {"alpha": a8, "beta": b8, "pi0_unreachable": z8},
        "predicted_oracle_at_16": pred16_from8,
        "measured_oracle_at_16_sc16_pool": meas16,
        "abs_error": (pred16_from8 - meas16) if meas16 is not None else None,
        "rel_error": ((pred16_from8 - meas16) / meas16) if meas16 else None,
        "plugin_lower_predicted_at_16": plugin_coverage(k8, 8, 16),
        "note": "the sc16 pool is an INDEPENDENT 16-sample generation on the same items, so this is a "
                "real out-of-sample test of the extrapolation model, not a self-check",
    }

    # (b) fit on the 16-sample pool and extrapolate
    if k16 is not None and len(k16) > 30:
        a16, b16, z16, _ = fit_zibb(k16, 16)
        aN, bN, _z, _ = fit_zibb(k16, 16, zero_inflated=False)
        entry["fit_on_sc16"] = {"alpha": a16, "beta": b16, "pi0_unreachable": z16}
        entry["fit_on_sc16_no_zero_inflation"] = {"alpha": aN, "beta": bN}
        ext = {}
        for N in [8, 16, 24, 32, 48, 64, 128, 256, 1024, 10 ** 6]:
            ext[N] = {
                "central_zibb": zibb_coverage(a16, b16, z16, N),
                "upper_no_zero_inflation": zibb_coverage(aN, bN, 0.0, N),
                "lower_empirical_plugin": plugin_coverage(k16, 16, N),
            }
        entry["extrapolated_coverage"] = ext
        entry["asymptote_central"] = 1 - z16
        entry["_k16"] = k16
        entry["band_reading"] = (
            "lower = empirical plug-in (p_hat = k/16): a HARD floor, it can never exceed the measured "
            "coverage@16 because every question with 0 correct slots in 16 is frozen as unreachable. "
            "central = zero-inflated beta-binomial; its N->inf asymptote is not credible (the fit puts "
            "pi0 ~ 0, i.e. it believes infinite sampling eventually answers everything, which ignores "
            "wrong golds and genuinely-absent knowledge). Trust the band at N<=64; past that read the "
            "lower bound.")
    COV[g] = entry

# =============================================================== 5. conversion / sel_eff trend
def loglin_fit(Ns, ys):
    """least squares y = c0 + c1*log2(N) over the finite points; slope is per DOUBLING of N."""
    x = np.log2(np.asarray(Ns, float)); y = np.asarray(ys, float)
    m = np.isfinite(y)
    x, y = x[m], y[m]
    Amat = np.vstack([np.ones_like(x), x]).T
    c, *_ = np.linalg.lstsq(Amat, y, rcond=None)
    return float(c[0]), float(c[1])


TREND = {}
NSC = [2, 3, 4, 5, 6, 7, 8]     # conversion is undefined at N=1 (oracle@1 == the single sample)
for g, rows in GROUPS.items():
    tab = CURVES[g]["curve_from_sc8_pool"]
    Ns = NS8
    conv = [tab[N]["conversion_clean_vs_repo_greedy"] for N in NSC]
    seff = [tab[N]["sel_eff_clean"] for N in Ns]
    seffk = [tab[N]["sel_eff_contaminated"] for N in Ns]
    c0, c1 = loglin_fit(NSC, conv)
    s0, s1 = loglin_fit(Ns, seff)
    kc0, kc1 = loglin_fit(Ns, seffk)
    # bootstrap the slopes over questions
    rng = np.random.default_rng(7)
    n = len(rows)
    V = np.array([[r["ver_clean"][N] for N in Ns] for r in rows])
    O = np.array([[r["oracle"][N] for N in Ns] for r in rows])
    G = np.array([r["greedy_repo"] for r in rows], float)
    slopes, conv_slopes, d816 = [], [], []
    for _ in range(1000):
        ix = rng.integers(0, n, n)
        vb, ob, gg = V[ix].mean(0), O[ix].mean(0), G[ix].mean()
        slopes.append(loglin_fit(Ns, vb / ob)[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(ob - gg > 1e-9, (vb - gg) / (ob - gg), np.nan)
        conv_slopes.append(loglin_fit(NSC, cv[1:])[1])
    # measured increments: what does DOUBLING N actually buy, at each doubling we can see?
    dbl = {}
    for a, b in ((1, 2), (2, 4), (4, 8)):
        dbl[f"{a}->{b}"] = {
            "d_oracle": tab[b]["oracle_at_N"] - tab[a]["oracle_at_N"],
            "d_verifier_clean": tab[b]["verifier_clean_at_N"] - tab[a]["verifier_clean_at_N"],
            "d_sel_eff": tab[b]["sel_eff_clean"] - tab[a]["sel_eff_clean"],
            "marginal_conversion": ((tab[b]["verifier_clean_at_N"] - tab[a]["verifier_clean_at_N"]) /
                                    (tab[b]["oracle_at_N"] - tab[a]["oracle_at_N"])),
        }
    o16 = CURVES[g]["measured_oracle_from_sc16_pool"]
    if o16:
        dbl["8->16 (oracle MEASURED on the independent sc16 pool)"] = {
            "d_oracle": o16[16] - o16[8],
            "d_verifier_clean": "NOT MEASURED -- the clean verifier scored only the 8-sample pool",
        }
    TREND[g] = {
        "conversion_clean_by_N": {N: tab[N]["conversion_clean_vs_repo_greedy"] for N in Ns},
        "sel_eff_clean_by_N": {N: tab[N]["sel_eff_clean"] for N in Ns},
        "sel_eff_contaminated_by_N": {N: tab[N]["sel_eff_contaminated"] for N in Ns},
        "conversion_loglin_N2to8": {"intercept": c0, "slope_per_doubling": c1,
                                    "slope_ci95": [float(np.percentile(conv_slopes, 2.5)),
                                                   float(np.percentile(conv_slopes, 97.5))]},
        "sel_eff_loglin": {"intercept": s0, "slope_per_doubling": s1,
                           "slope_ci95": [float(np.percentile(slopes, 2.5)),
                                          float(np.percentile(slopes, 97.5))]},
        "sel_eff_contaminated_loglin": {"intercept": kc0, "slope_per_doubling": kc1},
        "measured_per_doubling": dbl,
    }

# =============================================================== 6. extrapolate verifier accuracy
SELEFF_MODES = {
    "A_measured_trend": "CENTRAL. sel_eff is the MEASURED value up to N=8, then continues past the "
                        "N=8 anchor at the log-linear decline measured over N=1..8 (slope "
                        "significantly < 0). Corroborated in direction and magnitude by the "
                        "contaminated verifier's real N=16 measurement (scaling_curve16.json).",
    "B_trend_at_ci_upper": "OPTIMISTIC. same anchor, decline as SHALLOW as the 95% CI allows.",
    "C_frozen_at_8": "COUNTERFACTUAL. the decline simply stops at N=8 and sel_eff never falls "
                     "again. Contradicted by the data (slope CI excludes 0) and by the measured "
                     "8->16 sel_eff drop of the contaminated verifier; shown only to price the "
                     "best case for the verifier.",
    "D_perfect_selector": "UPPER BOUND, assumption-free. sel_eff = 1: a PERFECT selector. "
                          "verifier accuracy can never exceed coverage(N), whatever verifier you build.",
}


def sel_eff_at(g, N, mode):
    """sel_eff is MEASURED for N<=8; beyond that the measured N=8 value is carried forward with a
    slope. Anchoring at N=8 (rather than using the raw regression intercept) keeps every projected
    curve continuous with the measurement it starts from."""
    t = TREND[g]["sel_eff_loglin"]
    tab = CURVES[g]["curve_from_sc8_pool"]
    if N <= 8 and float(N).is_integer() and mode != "D_perfect_selector":
        return tab[int(N)]["sel_eff_clean"]
    anchor = tab[8]["sel_eff_clean"]
    d = math.log2(N / 8.0)
    if mode == "A_measured_trend":
        return max(0.0, anchor + t["slope_per_doubling"] * d)
    if mode == "B_trend_at_ci_upper":
        return max(0.0, anchor + t["slope_ci95"][1] * d)
    if mode == "C_frozen_at_8":
        return anchor
    if mode == "D_perfect_selector":
        return 1.0
    raise ValueError(mode)


def coverage_at(g, N, band):
    a16 = COV[g].get("fit_on_sc16"); an = COV[g].get("fit_on_sc16_no_zero_inflation")
    k16 = COV[g].get("_k16")
    if band == "central":
        return zibb_coverage(a16["alpha"], a16["beta"], a16["pi0_unreachable"], N)
    if band == "upper":
        return zibb_coverage(an["alpha"], an["beta"], 0.0, N)
    if band == "lower":
        return plugin_coverage(k16, 16, N)
    raise ValueError(band)


PROJ = {}
for g in GROUPS:
    if "extrapolated_coverage" not in COV[g]:
        continue
    tab = CURVES[g]["curve_from_sc8_pool"]
    s32 = CURVES[g]["baselines"]["always_32B_direct"]
    rowsN = {}
    for N in COV[g]["extrapolated_coverage"]:
        cell = {"coverage_central": coverage_at(g, N, "central"),
                "coverage_lower": coverage_at(g, N, "lower"),
                "coverage_upper": coverage_at(g, N, "upper")}
        for smode in SELEFF_MODES:
            cell[f"sel_eff__{smode}"] = sel_eff_at(g, N, smode)
            for band in ("central", "lower", "upper"):
                cell[f"acc__{band}_coverage__{smode}"] = coverage_at(g, N, band) * sel_eff_at(g, N, smode)
        cell["beats_32B_central_case"] = bool(cell["acc__central_coverage__A_measured_trend"] > s32)
        cell["beats_32B_even_with_a_PERFECT_selector"] = bool(
            cell["acc__central_coverage__D_perfect_selector"] > s32)
        rowsN[N] = cell
    # crossover search + where the central curve peaks
    cross, peak = {}, {}
    for smode in SELEFF_MODES:
        for band in ("central", "upper", "lower"):
            found = None
            for N in range(1, 8193):
                if coverage_at(g, N, band) * sel_eff_at(g, N, smode) >= s32:
                    found = N; break
            cross[f"{band}_coverage__{smode}"] = found
        vals = [(N, coverage_at(g, N, "central") * sel_eff_at(g, N, smode)) for N in range(1, 8193)]
        bn, bv = max(vals, key=lambda t: t[1])
        peak[smode] = {"argmax_N": bn, "max_accuracy": bv, "vs_always_32B": bv - s32}
    rs = [r for r in GROUPS[g] if r["strong32"] is not None]
    dif = boot_ci(rs, lambda r: r["ver_clean"][8] - r["strong32"], seed=911)
    PROJ[g] = {
        "always_32B_direct": s32,
        "measured_verifier_clean_at_8": tab[8]["verifier_clean_at_N"],
        "measured_best_over_N_le_8": {"N": max(NS8, key=lambda N: tab[N]["verifier_clean_at_N"]),
                                      "accuracy": max(tab[N]["verifier_clean_at_N"] for N in NS8)},
        "MEASURED_gap_at_N8_verifier_minus_32B": {
            "point": tab[8]["verifier_clean_at_N"] - s32, "ci95_paired_bootstrap": dif,
            "reading": "this one IS a measurement, not an extrapolation"},
        "sel_eff_scenarios": SELEFF_MODES,
        "projection": rowsN,
        "crossover_N_to_reach_always_32B_direct": cross,
        "best_achievable_over_all_N": peak,
        "note": "acc(N) = coverage(N) x sel_eff(N) is an identity, not a model: sel_eff(N) is DEFINED "
                "as verifier_acc(N)/oracle(N). The only extrapolated pieces are coverage(N) beyond "
                "N=16 and sel_eff(N) beyond N=8.",
        # what would have to be TRUE for the crossover to happen at each N
        "requirement_to_match_32B": {
            N: {"coverage_available_central": coverage_at(g, N, "central"),
                "sel_eff_REQUIRED": s32 / coverage_at(g, N, "central"),
                "sel_eff_projected_A_trend": sel_eff_at(g, N, "A_measured_trend"),
                "sel_eff_shortfall": s32 / coverage_at(g, N, "central") - sel_eff_at(g, N, "A_measured_trend"),
                "sel_eff_measured_at_N8_for_reference": tab[8]["sel_eff_clean"],
                "coverage_REQUIRED_if_sel_eff_frozen_at_its_N8_value": s32 / tab[8]["sel_eff_clean"],
                }
            for N in (8, 16, 32, 64, 128, 256)},
    }

# =============================================================== 7. coverage vs conversion split
SPLIT = {}
for g, rows in GROUPS.items():
    rs = [r for r in rows if r["strong32"] is not None]
    n = len(rs)
    s32 = np.array([r["strong32"] for r in rs], float)
    o8 = np.array([r["oracle"][8] for r in rs], float)          # 0/1: is a correct answer in the pool
    v8 = np.array([r["ver_clean"][8] for r in rs], float)       # P(verifier picks a correct answer)
    rs16 = [r for r in rs if r["sl16"] is not None]
    o16 = np.array([r["oracle16"][16] for r in rs16], float) if rs16 else None
    s32_16 = np.array([r["strong32"] for r in rs16], float) if rs16 else None
    gap = float(s32.mean() - v8.mean())
    cov_lim = float(np.mean(s32 * (1 - o8)))                    # 32B right, no correct answer in pool
    conv_lim = float(np.mean(s32 * o8 * (1 - v8)))              # right answer present, not picked
    cheap_win = float(np.mean((1 - s32) * v8))                  # 7B+verifier right, 32B wrong
    SPLIT[g] = {
        "n": n,
        "always_32B_direct": float(s32.mean()),
        "verifier_clean_at_8": float(v8.mean()),
        "oracle_at_8": float(o8.mean()),
        "no_correct_answer_in_8_pool_rate": float(np.mean(o8 == 0)),
        "gap_to_32B": gap,
        "decomposition_of_32B_wins_the_7B_misses": {
            "coverage_limited": cov_lim,
            "conversion_limited": conv_lim,
            "coverage_share_of_losses": cov_lim / (cov_lim + conv_lim) if (cov_lim + conv_lim) else None,
            "conversion_share_of_losses": conv_lim / (cov_lim + conv_lim) if (cov_lim + conv_lim) else None,
        },
        "questions_the_7B_wins_and_32B_loses": cheap_win,
        "identity_check_gap": cov_lim + conv_lim - cheap_win,
        "headroom_if_conversion_were_perfect_at_N8": float(o8.mean() - v8.mean()),
        "headroom_if_coverage_were_perfect_but_conversion_unchanged": None,
        "with_16_sample_pool": ({
            "n": len(rs16),
            "oracle_at_16": float(o16.mean()),
            "no_correct_answer_in_16_pool_rate": float(np.mean(o16 == 0)),
            "still_coverage_limited": float(np.mean(s32_16 * (1 - o16))),
        } if rs16 else None),
    }
    tab = CURVES[g]["curve_from_sc8_pool"]
    SPLIT[g]["headroom_if_coverage_were_perfect_but_conversion_unchanged"] = float(
        1.0 * tab[8]["sel_eff_clean"] - v8.mean())

# ============================== 7b. independent corroboration of the two shapes (existing files)
CORROB = {}

# (i) the CONTAMINATED verifier was actually RUN out to K=16 on the sc16 pools. It cannot be used
#     for a level, but its SHAPE is a real measurement, and because contamination inflates
#     selection it is a generous UPPER BOUND on how well the clean verifier converts extra samples.
sc16f = J(f"{A.contaminated}/scaling_curve16.json")
if os.path.exists(sc16f):
    d = json.load(open(sc16f))
    ks = sorted(int(k) for k in d if k.isdigit())
    rows = {k: {"verifier": d[str(k)]["verifier"], "oracle": d[str(k)]["oracle"],
                "random": d[str(k)]["random"],
                "sel_eff": d[str(k)]["verifier"] / d[str(k)]["oracle"]} for k in ks}
    CORROB["contaminated_verifier_measured_to_K16"] = {
        "source": f"{A.contaminated}/scaling_curve16.json  (src/training_methods/verifier_scaling_curve.py, "
                  "SC_TAG=sc16, n=1621 held-out questions)",
        "status": "MEASURED, but with the CONTAMINATED verifier -- used ONLY for the shape, and it is "
                  "an upper bound on the clean verifier's behaviour",
        "by_K": rows,
        "K8_to_K16": {
            "d_oracle": rows[16]["oracle"] - rows[8]["oracle"],
            "d_verifier": rows[16]["verifier"] - rows[8]["verifier"],
            "marginal_conversion": ((rows[16]["verifier"] - rows[8]["verifier"]) /
                                    (rows[16]["oracle"] - rows[8]["oracle"])),
            "d_sel_eff": rows[16]["sel_eff"] - rows[8]["sel_eff"],
        },
        "reading": "doubling 8->16 buys +0.0506 of oracle coverage and only +0.0074 of accuracy: the "
                   "marginal conversion of extra samples is ~15%, and selection efficiency keeps "
                   "falling (0.770 -> 0.717). Even the MEMORISING verifier saturates.",
    }

# (ii) the one pre-existing datapoint beyond N=8 that the brief names
dg = J("results/cascade_methods/artifacts/diverse_generation_gpu.json")
if os.path.exists(dg):
    d = json.load(open(dg))["pooled"]
    CORROB["diverse_generation_M15_crosscheck"] = {
        "source": "results/cascade_methods/artifacts/diverse_generation_gpu.json /pooled (n=1623 matched)",
        "oracle_iid_at_8": d["oracle"]["iid@8"],
        "oracle_diverse_portfolio_at_15": d["oracle"]["diverse_full@M"],
        "lift": d["oracle_lift"]["fullM_minus_iid8"],
        "this_analysis_iid_oracle_at_8": CURVES["POOLED"]["measured_oracle_from_sc16_pool"][8],
        "this_analysis_iid_oracle_at_16": CURVES["POOLED"]["measured_oracle_from_sc16_pool"][16],
        "this_analysis_lift_8_to_16": (CURVES["POOLED"]["measured_oracle_from_sc16_pool"][16] -
                                       CURVES["POOLED"]["measured_oracle_from_sc16_pool"][8]),
        "reading": "the 5-prompt x 3-temperature DIVERSE portfolio at M=15 lifted oracle by +0.0635 over "
                   "iid@8; plain iid sampling from 8 to 16 lifts it by a comparable amount on the same "
                   "kind of budget. Coverage responds to sample COUNT; the portfolio is not buying much "
                   "beyond what more samples buy.",
        "and_it_did_not_convert": {
            "d_oracle": d["oracle"]["diverse_full@M"] - d["oracle"]["iid@8"],
            "d_verifier_selected_accuracy": d["verifier_bo_n"]["delta_fullM_minus_iid"]["sel_acc"],
            "marginal_conversion": (d["verifier_bo_n"]["delta_fullM_minus_iid"]["sel_acc"] /
                                    (d["oracle"]["diverse_full@M"] - d["oracle"]["iid@8"])),
            "confident_distractor_rate_iid8_vs_diverse15": [
                d["verifier_bo_n"]["iid@8"]["confident_distractor_rate"],
                d["verifier_bo_n"]["diverse_full@M"]["confident_distractor_rate"]],
            "note": "measured with the CONTAMINATED verifier -- again a generous bound. +0.0635 of "
                    "coverage converted to +0.0246 of accuracy (39%), and the confident-distractor "
                    "rate ROSE 0.268 -> 0.301 as the pool grew. This is the selection-efficiency "
                    "decline showing up as a mechanism, not just a curve.",
        },
    }

# =============================================================== 8. cost model
FR = json.load(open(J("results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json")))
LAT = json.load(open(J("results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03.json")))
G7 = FR["flop_model"]["lingshu_7b_gflops"]; G32 = FR["flop_model"]["lingshu_32b_gflops"]
F7 = G7["TOTAL"]; F32 = G32["TOTAL"]
R32 = FR["derived_ratio"]["recommended"]["value"]
marginal_decode = G7["lm_decode_dense"] + G7["lm_decode_attn"] + G7["lm_head"]   # one extra sequence


def cost_flopeq(N, share_prefill=True):
    """cost of best-of-N + verify-all-N, in units of ONE 7B forward (= 1.0)."""
    if share_prefill:
        gen = (F7 + (N - 1) * marginal_decode) / F7      # one prefill, N sampled continuations
    else:
        gen = float(N)
    ver = float(N)                                       # one verifier forward per candidate
    return {"generate": gen, "verify": ver, "total": gen + ver}


REP = LAT["reconciliation"]["measurement"]["pooled"] if "pooled" in LAT["reconciliation"]["measurement"] else None
BO8_MS = 1305.3; BO8_J = 316.7                            # n=45 over 2 replicates, measured
GEN8_MS = 689.2; VER8_MS = 616.1
GEN1_MS = 350.0
GEN32_MS = FR["measurement_crosscheck"]["lingshu"]["lat_ms"][1]
GEN32_J = FR["measurement_crosscheck"]["lingshu"]["energy_j"][1]

COST = {
    "flop_constants": {
        "one_7b_forward_gflop": F7, "one_32b_forward_gflop": F32,
        "R32_used": R32,
        "source": "results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json",
        "marginal_gflop_per_extra_sampled_sequence_shared_prefill": marginal_decode,
        "marginal_as_fraction_of_one_7b_forward": marginal_decode / F7,
    },
    "flop_eq_by_N": {N: {"shared_prefill_generation": cost_flopeq(N, True),
                         "independent_generation": cost_flopeq(N, False),
                         "x_of_one_32B_forward_shared": cost_flopeq(N, True)["total"] / R32,
                         "x_of_one_32B_forward_independent": cost_flopeq(N, False)["total"] / R32}
                     for N in [1, 2, 4, 8, 16, 32, 64, 128, 256]},
    "measured_wallclock": {
        "source": "results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03.json "
                  "(n=45 over 2 replicates, HF batch-1 request, batch-8 internally, A100-80GB, NVML)",
        "bo8_total_ms": BO8_MS, "bo8_total_j": BO8_J,
        "gen8_ms": GEN8_MS, "verify8_ms": VER8_MS, "gen1_ms": GEN1_MS,
        "one_32B_nothink_forward_ms": GEN32_MS, "one_32B_nothink_forward_j": GEN32_J,
        "bo8_x_of_one_32B_forward_latency": BO8_MS / GEN32_MS,
        "bo8_x_of_one_32B_forward_energy": BO8_J / GEN32_J,
    },
    "wallclock_projection": {
        "caveat": "NOT MEASURED beyond N=8. Two bracketing models, both anchored on the measured "
                  "batch-8 point; the truth is between them.",
        "lower_affine_in_N": "a + b*N fitted on the two measured points per stage "
                             "(gen: 350 ms @1, 689.2 ms @8; verify: 205.2 ms median @1, 616.1 ms @8)",
        "upper_batches_of_8": "ceil(N/8) * 1305.3 ms -- one A100 can only hold so many concurrent "
                              "sequences; beyond a batch it serialises",
    },
}
gen_a = (689.2 - 350.0) / 7.0; gen_b = 350.0 - gen_a
ver_a = (616.1 - 205.2) / 7.0; ver_b = 205.2 - ver_a
for N in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    COST["wallclock_projection"].setdefault("ms_by_N", {})[N] = {
        "lower_affine": (gen_b + gen_a * N) + (ver_b + ver_a * N),
        "upper_batches_of_8": math.ceil(N / 8) * BO8_MS,
        "measured": BO8_MS if N == 8 else None,
    }

# =============================================================== 9. write out
def clean_keys(o):
    if isinstance(o, dict):
        return {str(k): clean_keys(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [clean_keys(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


OUT = {
    "question": "Does the trained verifier's benefit keep growing with N, and can 7B + verifier at "
                "higher N match/beat a 32B without ever calling the 32B at test time?",
    "date": "2026-08-03",
    "offline": True, "no_gpu": True, "no_new_inference": True,
    "verifier": {
        "used": A.clean,
        "level": "L1 image-disjoint (no eval image, no eval item in verifier training)",
        "named_by": "results/cascade_methods/artifacts/verifier_disjoint_retrain_2026-07-30.json"
                    " /levels/L1_image_disjoint/adapter",
        "contaminated_shown_for_contrast_only": A.contaminated,
        "warning": "every headline number here is the CLEAN L1 verifier; the contaminated verifier "
                   "memorised 67-73% of the items it scores and its scaling curve is meaningless",
    },
    "judge": "src/labeling/run_judge.py (MedVLThinker-32B, judge_ok) -- the same judge as the headline",
    "candidate_pools": {
        "sc8": "ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl -- vLLM n=8, temperature 0.7",
        "sc16": "ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc16.jsonl -- INDEPENDENT n=16, "
                "temperature 0.7 generation on the same items; used for the measured coverage curve to N=16",
        "exchangeability": "i.i.d. samples from one distribution at a fixed temperature, so a uniformly "
                           "random N-subset is an unbiased estimate of an N-sample run",
    },
    "method": {
        "oracle_verifier_and_self_consistency": "EXACT expectation over ALL C(8,N) subsets of the "
            "8-sample pool (255 subsets per question, enumerated); score/vote ties are resolved as the "
            "uniform random tie-break they are (mean label over the tied winners). No Monte-Carlo.",
        "oracle_from_sc16": "closed form 1 - C(16-k,N)/C(16,N) over the independent 16-sample pool",
        "ci": f"non-parametric bootstrap over questions, {A.nboot} resamples",
        "coverage_extrapolation": "zero-inflated beta-binomial on the per-question count of correct "
                                  "slots; validated out-of-sample by fitting on the 8-sample pool and "
                                  "predicting the independently measured 16-sample pool",
    },
    "harness_validation": VALID,
    "curves": CURVES,
    "coverage_model": COV,
    "conversion_and_selection_efficiency": TREND,
    "independent_corroboration": CORROB,
    "projection_vs_32B": PROJ,
    "coverage_vs_conversion_decomposition": SPLIT,
    "cost": COST,
}

P, C, T, S = PROJ["POOLED"], CURVES["POOLED"], TREND["POOLED"], SPLIT["POOLED"]
tab = C["curve_from_sc8_pool"]
OUT["verdict"] = {
    "headline": "OUT OF REACH on this evidence. 7B + the CLEAN verifier does not reach always-32B-direct "
                "at any N. The two curves work against each other: coverage grows sub-logarithmically "
                "while selection efficiency falls at a rate that cancels it, so verifier-selected "
                "accuracy is already flat by N~5-8 and declines beyond it.",
    "curve_1_coverage": {
        "shape": "grows, flattening. MEASURED to N=16 on an independent 16-sample generation.",
        "measured": {N: C["measured_oracle_from_sc16_pool"][N] for N in (1, 2, 4, 8, 12, 16)},
        "per_doubling_measured": "+0.070 (1->2), +0.067 (2->4), +0.066 (4->8), +0.052 (8->16)",
        "extrapolated_central": {N: COV["POOLED"]["extrapolated_coverage"][N]["central_zibb"]
                                 for N in (32, 64, 256)},
        "extrapolated_lower_bound": {N: COV["POOLED"]["extrapolated_coverage"][N]["lower_empirical_plugin"]
                                     for N in (32, 64, 256)},
        "model_validated_out_of_sample": COV["POOLED"]["validation_fit8_predict16"],
    },
    "curve_2_conversion": {
        "answer": "NOT constant and NOT rising -- FALLING. Selection efficiency "
                  "P(pick a correct answer | one is present) falls "
                  f"{-T['sel_eff_loglin']['slope_per_doubling']:.4f} per doubling of N "
                  f"(95% CI [{-T['sel_eff_loglin']['slope_ci95'][1]:.4f}, "
                  f"{-T['sel_eff_loglin']['slope_ci95'][0]:.4f}]), from 1.000 at N=1 to "
                  f"{tab[8]['sel_eff_clean']:.4f} at N=8.",
        "sel_eff_by_N": {N: tab[N]["sel_eff_clean"] for N in NS8},
        "marginal_conversion_of_each_doubling": "0.457 (1->2), 0.265 (2->4), 0.167 (4->8) -- each "
                                                "doubling converts LESS of the coverage it adds",
        "ratio_style_conversion_vs_greedy": {N: tab[N]["conversion_clean_vs_repo_greedy"] for N in NS8},
        "why_the_ratio_conversion_looks_flat": "conversion = (ver-greedy)/(oracle-greedy) hovers near "
            "0.20-0.21 for N>=3 only because BOTH its numerator and denominator are growing; it hides "
            "the fact that the marginal conversion of each extra sample is collapsing. sel_eff is the "
            "diagnostic to read.",
        "corroborated_at_N16": CORROB.get("contaminated_verifier_measured_to_K16", {}).get("K8_to_K16"),
    },
    "crossover_N": {
        "central_measured_trend": P["crossover_N_to_reach_always_32B_direct"]["central_coverage__A_measured_trend"],
        "optimistic_shallowest_decline_in_CI": P["crossover_N_to_reach_always_32B_direct"]["central_coverage__B_trend_at_ci_upper"],
        "counterfactual_if_the_decline_simply_stopped_at_N8": P["crossover_N_to_reach_always_32B_direct"]["central_coverage__C_frozen_at_8"],
        "with_a_PERFECT_selector_over_the_same_pool": P["crossover_N_to_reach_always_32B_direct"]["central_coverage__D_perfect_selector"],
        "reading": "there is NO crossover under the measured trend or anywhere in its 95% CI. The only "
                   "scenario that crosses (N=18) requires the selection-efficiency decline to stop dead "
                   "at N=8, which the data rejects. What the N=3 perfect-selector figure says is that the "
                   "SAMPLES are already good enough -- the SELECTOR is not.",
        "status": "EXTRAPOLATION, not measurement",
    },
    "cost_at_the_counterfactual_crossover_N18": {
        "flop_eq_7b_forwards": cost_flopeq(18, True)["total"],
        "x_of_one_32B_forward": cost_flopeq(18, True)["total"] / R32,
        "wallclock_ms_range": [COST["wallclock_projection"]["ms_by_N"][16]["lower_affine"],
                               COST["wallclock_projection"]["ms_by_N"][16]["upper_batches_of_8"]],
        "one_32B_forward_ms": GEN32_MS,
        "reading": "even the counterfactual best case costs ~5x a single 32B forward in FLOPs and "
                   "3-4x its wall clock. This was never going to be an efficiency claim; the only "
                   "claim available was hardware accessibility, and the accuracy does not arrive.",
    },
    "coverage_vs_conversion_decomposition": {
        "aggregate": f"coverage is ALREADY not the binding constraint in aggregate: oracle@8 = "
                     f"{tab[8]['oracle_at_N']:.4f} > always-32B-direct {P['always_32B_direct']:.4f}. "
                     "A perfect selector over the EXISTING 8-sample pool would beat the 32B by "
                     f"{tab[8]['oracle_at_N'] - P['always_32B_direct']:+.4f}.",
        "per_question": (f"of the questions the 32B gets right and 7B+verifier@8 misses, "
                         f"{100 * S['decomposition_of_32B_wins_the_7B_misses']['coverage_share_of_losses']:.0f}% "
                         f"are coverage-limited (no correct answer anywhere in the pool: "
                         f"{S['decomposition_of_32B_wins_the_7B_misses']['coverage_limited']:.4f} of all "
                         f"questions) and "
                         f"{100 * S['decomposition_of_32B_wins_the_7B_misses']['conversion_share_of_losses']:.0f}% "
                         f"are conversion-limited (the right answer is in the pool and the verifier does "
                         f"not pick it: {S['decomposition_of_32B_wins_the_7B_misses']['conversion_limited']:.4f})."),
        "no_correct_answer_in_pool": {"at_N8": S["no_correct_answer_in_8_pool_rate"],
                                      "at_N16_measured": S["with_16_sample_pool"]["no_correct_answer_in_16_pool_rate"]},
        "which_to_attack": "CONVERSION. Doubling the pool 8->16 removes only "
                           f"{S['no_correct_answer_in_8_pool_rate'] - S['with_16_sample_pool']['no_correct_answer_in_16_pool_rate']:.4f} "
                           "of the coverage hole, while the selection hole at N=8 is "
                           f"{tab[8]['oracle_at_N'] - tab[8]['verifier_clean_at_N']:.4f} and GROWS with N.",
    },
    "what_would_have_to_change": {
        "selection_efficiency_needed_at_N16": P["requirement_to_match_32B"][16]["sel_eff_REQUIRED"],
        "selection_efficiency_measured_at_N8": tab[8]["sel_eff_clean"],
        "selection_efficiency_projected_at_N16": P["requirement_to_match_32B"][16]["sel_eff_projected_A_trend"],
        "shortfall_at_N16": P["requirement_to_match_32B"][16]["sel_eff_shortfall"],
        "shortfall_at_N64": P["requirement_to_match_32B"][64]["sel_eff_shortfall"],
        "statement": "to match the 32B at N=16 the verifier would have to select correctly on "
                     f"{P['requirement_to_match_32B'][16]['sel_eff_REQUIRED']:.1%} of the questions "
                     "whose pool contains a correct answer -- i.e. it would have to get BETTER as the "
                     "pool grows, when it measurably gets worse. Even at N=64, where coverage has risen "
                     f"to {P['requirement_to_match_32B'][64]['coverage_available_central']:.3f}, the "
                     f"requirement is {P['requirement_to_match_32B'][64]['sel_eff_REQUIRED']:.1%} against "
                     f"a projected {P['requirement_to_match_32B'][64]['sel_eff_projected_A_trend']:.1%}.",
        "so": "more samples is the wrong lever. The lever is a selector that does not degrade with pool "
              "size (the confident-distractor problem), and that is exactly the selection wall this "
              "project has already hit thirteen independent ways.",
    },
    "measured_anchor_do_not_lose": {
        "verifier_clean_at_8": tab[8]["verifier_clean_at_N"],
        "always_32B_direct": P["always_32B_direct"],
        "gap": P["MEASURED_gap_at_N8_verifier_minus_32B"],
    },
}
os.makedirs(os.path.dirname(J(A.out)), exist_ok=True)
json.dump(clean_keys(OUT), open(J(A.out), "w"), indent=1)
print(f"\nwrote -> {A.out}")

# =============================================================== 10. console report
def pct(x):
    return "  n/a " if x is None else f"{x:6.4f}"


for g in ["POOLED"] + DSETS:
    c = CURVES[g]; t = c["curve_from_sc8_pool"]; b = c["baselines"]
    print(f"\n=================== {g}  n={c['n_questions']}")
    print(f"  greedy(temp0)={pct(b['true_greedy_temp0'])}  modal-of-8={pct(b['greedy_repo_dump_field_is_modal_of_8'])}"
          f"  1 sample={pct(b['one_random_sample_temp07'])}  always-32B={pct(b['always_32B_direct'])}")
    print(f"  {'N':>3} {'oracle@N':>9} {'ver_clean':>10} {'SC@N':>8} {'conv':>7} {'sel_eff':>8} {'ver_cont':>9}")
    for N in NS8:
        r = t[N]
        print(f"  {N:>3} {pct(r['oracle_at_N']):>9} {pct(r['verifier_clean_at_N']):>10} "
              f"{pct(r['self_consistency_at_N']):>8} {pct(r['conversion_clean_vs_repo_greedy']):>7} "
              f"{pct(r['sel_eff_clean']):>8} {pct(r['verifier_contaminated_at_N']):>9}")
    o16 = c["measured_oracle_from_sc16_pool"]
    if o16:
        print("  measured oracle from the INDEPENDENT sc16 pool: " +
              " ".join(f"@{N}={o16[N]:.4f}" for N in (1, 2, 4, 8, 12, 16)))
    if g in COV and "extrapolated_coverage" in COV[g]:
        v = COV[g]["validation_fit8_predict16"]
        print(f"  extrapolation check: fit on 8 -> predict oracle@16 = {v['predicted_oracle_at_16']:.4f} "
              f"vs MEASURED {v['measured_oracle_at_16_sc16_pool']:.4f} "
              f"(err {v['abs_error']:+.4f}, {100*v['rel_error']:+.1f}%)")
        e = COV[g]["extrapolated_coverage"]
        for N in (16, 32, 64):
            print(f"    coverage@{N:<4d} central {e[N]['central_zibb']:.4f}  "
                  f"[lower {e[N]['lower_empirical_plugin']:.4f}, upper {e[N]['upper_no_zero_inflation']:.4f}]")
        print(f"    coverage ceiling (N->inf, central) = {COV[g]['asymptote_central']:.4f}")
    tr = TREND[g]
    print(f"  sel_eff slope per doubling of N: {tr['sel_eff_loglin']['slope_per_doubling']:+.4f} "
          f"CI95 [{tr['sel_eff_loglin']['slope_ci95'][0]:+.4f}, {tr['sel_eff_loglin']['slope_ci95'][1]:+.4f}]"
          f"   conversion slope (N=2..8): {tr['conversion_loglin_N2to8']['slope_per_doubling']:+.4f} "
          f"CI95 [{tr['conversion_loglin_N2to8']['slope_ci95'][0]:+.4f}, "
          f"{tr['conversion_loglin_N2to8']['slope_ci95'][1]:+.4f}]")
    print("  what each measured DOUBLING of N bought:")
    for k, v in tr["measured_per_doubling"].items():
        if isinstance(v.get("d_verifier_clean"), str):
            print(f"    N {k:<48s} d_oracle {v['d_oracle']:+.4f}   d_verifier {v['d_verifier_clean']}")
        else:
            print(f"    N {k:<48s} d_oracle {v['d_oracle']:+.4f}   d_verifier {v['d_verifier_clean']:+.4f}"
                  f"   marginal conversion {v['marginal_conversion']:.3f}   d_sel_eff {v['d_sel_eff']:+.4f}")
    if g in PROJ:
        p = PROJ[g]
        print(f"  PROJECTED verifier-selected accuracy vs always-32B-direct {p['always_32B_direct']:.4f} "
              f"[EXTRAPOLATION, not measurement]:")
        print(f"    {'N':>5} {'coverage':>9} {'A trend':>9} {'B ci-hi':>9} {'C frozen@8':>11} {'D perfect':>10}")
        for N in (8, 16, 32, 64, 256):
            c = p["projection"][N]
            print(f"    {N:>5} {c['coverage_central']:>9.4f} "
                  f"{c['acc__central_coverage__A_measured_trend']:>9.4f} "
                  f"{c['acc__central_coverage__B_trend_at_ci_upper']:>9.4f} "
                  f"{c['acc__central_coverage__C_frozen_at_8']:>11.4f} "
                  f"{c['acc__central_coverage__D_perfect_selector']:>10.4f}")
        for k, v in p["crossover_N_to_reach_always_32B_direct"].items():
            if k.startswith("central"):
                print(f"    crossover N ({k}): {v}")
        for k, v in p["best_achievable_over_all_N"].items():
            print(f"    best achievable, {k:<22s}: {v['max_accuracy']:.4f} at N={v['argmax_N']} "
                  f"({v['vs_always_32B']:+.4f} vs always-32B)")
        m = p["MEASURED_gap_at_N8_verifier_minus_32B"]
        print(f"    MEASURED at N=8: verifier {p['measured_verifier_clean_at_8']:.4f} - 32B "
              f"{p['always_32B_direct']:.4f} = {m['point']:+.4f} "
              f"95% CI [{m['ci95_paired_bootstrap'][0]:+.4f}, {m['ci95_paired_bootstrap'][1]:+.4f}]")
    s = SPLIT[g]
    d = s["decomposition_of_32B_wins_the_7B_misses"]
    print(f"  gap to 32B = {s['gap_to_32B']:+.4f}; of the questions the 32B wins and the 7B+verifier "
          f"loses: coverage-limited {d['coverage_limited']:.4f} ({100*d['coverage_share_of_losses']:.1f}%), "
          f"conversion-limited {d['conversion_limited']:.4f} ({100*d['conversion_share_of_losses']:.1f}%)")

print("\n---- cost ----")
for N in (1, 8, 16, 32, 64, 256):
    c = COST["flop_eq_by_N"][N]
    w = COST["wallclock_projection"]["ms_by_N"][N]
    print(f"  N={N:<4d} FLOP-eq {c['shared_prefill_generation']['total']:8.2f} 7B-forwards "
          f"= {c['x_of_one_32B_forward_shared']:6.2f}x one 32B forward | "
          f"latency {w['lower_affine']:7.0f}-{w['upper_batches_of_8']:7.0f} ms "
          f"(32B forward = {GEN32_MS:.0f} ms)")
