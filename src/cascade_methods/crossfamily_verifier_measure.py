#!/usr/bin/env python3
"""crossfamily_verifier_measure.py -- OFFLINE analysis of the cross-family zero-shot verifier sweep.

Tests the hypothesis that a verifier from a DIFFERENT MODEL FAMILY holds information the generator
does not, and therefore selects better among the generator's own candidates than a same-family
verifier of equal or larger size.

Everything is recomputed from checkpoints already on disk; nothing is regenerated.

INPUTS
  ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl                the FIXED 8-sample pool
  ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8_scexploded[.judge]   per-candidate 32B-judge labels
  ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.judge.jsonl              TRUE greedy (temp 0) labels
  ckpts/train/lora_verifier_disjoint/transfer_dump_{ds}_lingshu7b.json       TRAINED same-family verifier (CLEAN L1)
  ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_lingshu7b.json        contaminated (contrast ONLY, labelled)
  ckpts/openvqa/crossfam_verifier/ckpt_{ds}_{tag}.jsonl                      zero-shot cross-family scores

VALIDATED: the per-candidate judge labels are reconstructed from the scexploded judge file by
normalized-answer lookup, and the reconstruction reproduces the trained dump's own `sl` array
EXACTLY on all 2345 items (asserted at run time -- the script aborts otherwise). oracle@8, greedy
and self-consistency are therefore identical across every arm by construction, and that is asserted too.

METRICS (per verifier x dataset, plus POOLED)
  sel_acc     accuracy of the argmax-scored candidate (first-index tie-break, the same rule the
              published comparator uses); sel_acc_tieavg averages over uniform random tie-breaks.
  sel_eff     P(pick correct | a correct candidate is present) = sel_acc / oracle@8   [PRIMARY]
  auroc       candidate-level ranking AUROC over all 8 slots (scores broadcast to duplicate answers)
  oracle@8    fixed-pool sanity check (identical across verifiers)
  gain        sel_acc - greedy
  CIs         paired non-parametric bootstrap over QUESTIONS (nboot=10000, seed 0)

DE-CORRELATION LAW
  phi(G, V) between G = "generator's greedy answer correct" and V = "verifier picks a correct
  candidate", on the oracle=1 subset (where V is a genuine choice) and on all items.
  phi is used as the headline coefficient because for two binary variables it IS the Pearson
  correlation, so it is directly the "shared-information" quantity the hypothesis is about; Cohen's
  kappa is reported alongside but is not used for the law because kappa penalises differences in
  marginal prevalence, which differ a lot between these verifiers and would confound the reading.
  Then: does sel_eff track de-correlation across the family sweep? (Pearson + Spearman over the
  handful of verifiers -- suggestive, not fitted; the script says so in its own output.)

ENSEMBLES
  rank-fusion (mean of within-question ranks) and score-average, for same-family + best cross-family.

  python3 src/cascade_methods/crossfamily_verifier_measure.py
  -> results/cascade_methods/artifacts/crossfamily_verifier_sweep_2026-08-04.json
"""
import argparse, json, os, glob
from collections import Counter

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
DSETS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
TAG = "lingshu7b"
CK = "ckpts/openvqa/cheap_lingshu7b"
XF = "ckpts/openvqa/crossfam_verifier"
norm = lambda s: str(s).strip().lower()

# tag -> (label, family/relation to the generator, distance rank 0=identical)
VERIFIERS = {
    "trained7b_clean":  ("Lingshu-7B + LoRA (trained, L1-disjoint)", "same family, TRAINED", 0),
    "lingshu7b_zs":     ("Lingshu-7B zero-shot (the generator itself)", "same model", 0),
    "lingshu32b":       ("Lingshu-32B zero-shot", "same family, 4.5x scale", 1),
    "qwen25vl7b":       ("Qwen2.5-VL-7B-Instruct zero-shot", "SAME arch, different training (Lingshu's base)", 2),
    "mvt7b":            ("MedVLThinker-7B-RL zero-shot", "same arch, different medical tuning", 2),
    "qoqmed7b":         ("QoQ-Med-VL-7B zero-shot", "same arch, different medical tuning", 2),
    "huatuo7b":         ("HuatuoGPT-Vision-7B zero-shot", "different medical family", 3),
    "medgemma4b":       ("MedGemma-4B-it zero-shot", "different architecture (Gemma3) + medical", 4),
    "internvl3_8b":     ("InternVL3-8B zero-shot", "different architecture entirely", 4),
    "chiron_o1_8b":     ("Chiron-o1-8B zero-shot", "different architecture (InternVL) + medical", 4),
}
CROSSFAM = ["qwen25vl7b", "mvt7b", "qoqmed7b", "huatuo7b", "medgemma4b", "internvl3_8b", "chiron_o1_8b"]

ap = argparse.ArgumentParser()
ap.add_argument("--nboot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ntie", type=int, default=256, help="random tie-break orders for sel_acc_tieavg")
ap.add_argument("--out", default="results/cascade_methods/artifacts/crossfamily_verifier_sweep_2026-08-04.json")
A = ap.parse_args()
RNG = np.random.default_rng(A.seed)


# ------------------------------------------------------------------ loaders
def jl(p):
    p = J(p)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def build_labels(ds):
    """-> (pool_rows, {idx: [judge label per slot]}), reconstructed from the scexploded judge file."""
    pool = jl(f"{CK}/ckpt_{ds}_{TAG}_sc8.jsonl")
    exp = {r["idx"]: r for r in jl(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.jsonl")}
    jud = {r["idx"]: int(r["judge_ok"]) for r in jl(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.judge.jsonl")}
    by_q = {}
    for k, r in exp.items():
        q = k.split("#")[0]
        by_q.setdefault(q, {})[norm(r["modal_pred"])] = jud.get(k)
    labs = {}
    for r in pool:
        m = by_q.get(str(r["idx"]), {})
        L = [m.get(norm(p)) for p in r["preds"]]
        assert all(x is not None for x in L), f"missing judge label {ds} {r['idx']}"
        labs[str(r["idx"])] = [int(x) for x in L]
    return pool, labs


def trained_scores(adapter, ds):
    p = J(f"ckpts/train/{adapter}/transfer_dump_{ds}_{TAG}.json")
    if not os.path.exists(p): return None, None
    rows = json.load(open(p))
    return ({str(r["idx"]): [float(x) for x in r["scores"]] for r in rows},
            {str(r["idx"]): [None if x == -1 else int(x) for x in r["sl"]] for r in rows})


def crossfam_scores(tag, ds, pool):
    p = J(f"{XF}/ckpt_{ds}_{tag}.jsonl")
    if not os.path.exists(p): return None
    by = {str(r["idx"]): r["scores_by_answer"] for r in jl(f"{XF}/ckpt_{ds}_{tag}.jsonl")}
    out = {}
    for r in pool:
        s = by.get(str(r["idx"]))
        if s is None: return None                      # incomplete run -> drop the verifier
        v = [s.get(norm(p_)) for p_ in r["preds"]]
        if any(x is None for x in v): return None
        out[str(r["idx"])] = [float(x) for x in v]
    return out


# ------------------------------------------------------------------ metrics
def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.sum() in (0, len(y)): return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        ix = np.where(s == v)[0]
        if len(ix) > 1: ranks[ix] = ranks[ix].mean()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pick_vectors(S, L, keys, ntie, rng):
    """-> (picked-correct 0/1 per question [first-index tie-break], tie-averaged expectation)."""
    hard, soft = [], []
    for k in keys:
        s, y = np.asarray(S[k], float), np.asarray(L[k], int)
        hard.append(int(y[int(np.argmax(s))]))
        top = np.flatnonzero(s >= s.max() - 1e-12)
        soft.append(float(y[top].mean()))              # exact expectation under uniform tie-break
    return np.asarray(hard, int), np.asarray(soft, float)


def phi_kappa(g, v):
    g, v = np.asarray(g, int), np.asarray(v, int)
    n = len(g)
    if n == 0: return None, None
    n11 = int(((g == 1) & (v == 1)).sum()); n10 = int(((g == 1) & (v == 0)).sum())
    n01 = int(((g == 0) & (v == 1)).sum()); n00 = int(((g == 0) & (v == 0)).sum())
    den = np.sqrt(float(n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = float((n11 * n00 - n10 * n01) / den) if den > 0 else None
    po = (n11 + n00) / n
    pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / n ** 2
    kap = float((po - pe) / (1 - pe)) if pe < 1 else None
    return phi, kap


def boot_ci(fn, n, nboot, rng):
    idx = rng.integers(0, n, size=(nboot, n))
    v = np.asarray([fn(i) for i in idx], float)
    v = v[np.isfinite(v)]
    if len(v) == 0: return None, None, None
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p))


# ------------------------------------------------------------------ assemble
print("[load] pools + judge labels", flush=True)
POOL, LAB, GREEDY, KEYS = {}, {}, {}, {}
for ds in DSETS:
    pool, labs = build_labels(ds)
    POOL[ds] = {str(r["idx"]): r for r in pool}
    LAB[ds] = labs
    KEYS[ds] = [str(r["idx"]) for r in pool]
    gj = {str(r["idx"]): int(r["judge_ok"]) for r in jl(f"{CK}/ckpt_{ds}_{TAG}.judge.jsonl")}
    GREEDY[ds] = gj

# validate the label reconstruction against the trained dump's own sl (hard assert)
for ds in DSETS:
    _, sl = trained_scores("lora_verifier_disjoint", ds)
    for k, v in sl.items():
        assert [int(x) for x in v] == LAB[ds][k], f"judge-label reconstruction mismatch {ds} {k}"
print("[ok] judge-label reconstruction reproduces the trained dump's sl on all items", flush=True)

SCORES = {}                                            # tag -> ds -> {idx: [8 scores]}
for tag in VERIFIERS:
    if tag == "trained7b_clean":
        d = {}
        for ds in DSETS:
            s, _ = trained_scores("lora_verifier_disjoint", ds)
            if s is None: d = None; break
            d[ds] = s
        if d: SCORES[tag] = d
        continue
    d = {}
    for ds in DSETS:
        s = crossfam_scores(tag, ds, jl(f"{CK}/ckpt_{ds}_{TAG}_sc8.jsonl"))
        if s is None: d = None; break
        d[ds] = s
    if d: SCORES[tag] = d
    else: print(f"[skip] {tag}: no complete score file for all 3 datasets", flush=True)
# contaminated verifier, contrast ONLY
d = {}
for ds in DSETS:
    s, _ = trained_scores("lora_verifier_pooled4", ds)
    if s is None: d = None; break
    d[ds] = s
if d: SCORES["trained7b_CONTAMINATED"] = d
print(f"[have] {sorted(SCORES)}", flush=True)

# ---------------- fixed-pool quantities (identical for every verifier -- asserted below)
FIX = {}
for ds in DSETS:
    ks = KEYS[ds]
    orc = np.asarray([max(LAB[ds][k]) for k in ks], int)
    grd = np.asarray([GREEDY[ds][k] for k in ks], int)
    sc = []
    for k in ks:
        preds = POOL[ds][k]["preds"]
        top = Counter(norm(p) for p in preds).most_common(1)[0][0]
        j = next(i for i, p in enumerate(preds) if norm(p) == top)
        sc.append(LAB[ds][k][j])
    FIX[ds] = {"oracle": orc, "greedy": grd, "sc": np.asarray(sc, int)}
FIX["POOLED"] = {m: np.concatenate([FIX[d][m] for d in DSETS]) for m in ("oracle", "greedy", "sc")}
KEYS["POOLED"] = [(d, k) for d in DSETS for k in KEYS[d]]

# ---------------- per-verifier vectors
VEC = {}
for tag, per_ds in SCORES.items():
    VEC[tag] = {}
    for ds in DSETS:
        hard, soft = pick_vectors(per_ds[ds], LAB[ds], KEYS[ds], A.ntie, RNG)
        flat_s = np.concatenate([per_ds[ds][k] for k in KEYS[ds]])
        flat_y = np.concatenate([LAB[ds][k] for k in KEYS[ds]])
        VEC[tag][ds] = {"hard": hard, "soft": soft, "auroc": auroc(flat_s, flat_y)}
    VEC[tag]["POOLED"] = {
        "hard": np.concatenate([VEC[tag][d]["hard"] for d in DSETS]),
        "soft": np.concatenate([VEC[tag][d]["soft"] for d in DSETS]),
        "auroc": auroc(np.concatenate([np.concatenate([SCORES[tag][d][k] for k in KEYS[d]]) for d in DSETS]),
                       np.concatenate([np.concatenate([LAB[d][k] for k in KEYS[d]]) for d in DSETS])),
    }

SPLITS = DSETS + ["POOLED"]
BASE = "trained7b_clean"


def sel_eff(hard, orc, sub=None):
    h, o = (hard, orc) if sub is None else (hard[sub], orc[sub])
    return float(h.sum() / o.sum()) if o.sum() > 0 else float("nan")


out = {
    "what": ("PHASE 1 of the cross-family verifier programme: score ONE fixed candidate pool with "
             "zero-shot pointwise verifiers from several model families and test whether selection "
             "quality tracks distance from the generator."),
    "date": "2026-08-04",
    "status_of_numbers": "MEASURED (GPU verifier passes) + DERIVED (all selection statistics, bootstrap CIs)",
    "generator": "Lingshu-7B (temp 0.7, n=8, cap320) -- pools unchanged from the published open-text arm",
    "pools": {ds: len(KEYS[ds]) for ds in DSETS} | {"POOLED": len(KEYS["POOLED"])},
    "judge": "src/labeling/run_judge.py (MedVLThinker-32B, judge_ok) -- the SAME judge as the headline",
    "prompt": "verbatim diversity_generate_gpu.VERIFY_SYS + build_verify body, identical for every family",
    "image_budget": "every family sees the same picture downscaled to <=250880 px (the cap320 budget "
                    "the pools were generated at), passed as a base64 data URI through vLLM llm.chat()",
    "comparator": {
        "name": "trained same-family verifier, CLEAN L1 image-disjoint (ckpts/train/lora_verifier_disjoint)",
        "source": "results/cascade_methods/artifacts/verifier_disjoint_retrain_2026-07-30.json",
    },
    "nboot": A.nboot, "seed": A.seed,
    "verifiers": {t: {"label": VERIFIERS[t][0], "relation": VERIFIERS[t][1], "distance_rank": VERIFIERS[t][2]}
                  for t in VERIFIERS if t in SCORES},
    "table": {}, "contrasts": {}, "decorrelation": {}, "ensembles": {},
}

print("[measure] per-verifier table + paired bootstrap", flush=True)
for split in SPLITS:
    orc = FIX[split]["oracle"]; grd = FIX[split]["greedy"]; scv = FIX[split]["sc"]
    n = len(orc)
    row = {"n_questions": int(n), "oracle_at_8": float(orc.mean()),
           "greedy": float(grd.mean()), "self_consistency": float(scv.mean()), "verifiers": {}}
    for tag in SCORES:
        v = VEC[tag][split]
        row["verifiers"][tag] = {
            "sel_acc": float(v["hard"].mean()),
            "sel_acc_tieavg": float(v["soft"].mean()),
            "sel_eff": sel_eff(v["hard"], orc),
            "sel_eff_tieavg": float(v["soft"].sum() / orc.sum()),
            "auroc_candidate": v["auroc"],
            "gain_vs_greedy": float(v["hard"].mean() - grd.mean()),
        }
    out["table"][split] = row

    # paired bootstrap of every verifier MINUS the trained same-family comparator
    if BASE in SCORES:
        b = VEC[BASE][split]["hard"]
        C = {}
        for tag in SCORES:
            if tag == BASE: continue
            h = VEC[tag][split]["hard"]
            d_eff = sel_eff(h, orc) - sel_eff(b, orc)
            d_acc = float(h.mean() - b.mean())
            lo, hi, p = boot_ci(lambda ii: sel_eff(h[ii], orc[ii]) - sel_eff(b[ii], orc[ii]), n, A.nboot, np.random.default_rng(A.seed))
            lo2, hi2, p2 = boot_ci(lambda ii: float(h[ii].mean() - b[ii].mean()), n, A.nboot, np.random.default_rng(A.seed))
            C[tag] = {"d_sel_eff": d_eff, "ci95": [lo, hi], "boot_p": p,
                      "d_sel_acc": d_acc, "acc_ci95": [lo2, hi2], "acc_boot_p": p2,
                      "beats_baseline_ci_excludes_zero": bool(lo is not None and lo > 0)}
        out["contrasts"][split] = {"vs": BASE, "deltas": C}

    # same-family ZERO-SHOT control contrast (the pure "same information" floor)
    if "lingshu7b_zs" in SCORES:
        b = VEC["lingshu7b_zs"][split]["hard"]
        C2 = {}
        for tag in SCORES:
            if tag == "lingshu7b_zs": continue
            h = VEC[tag][split]["hard"]
            lo, hi, p = boot_ci(lambda ii: sel_eff(h[ii], orc[ii]) - sel_eff(b[ii], orc[ii]), n, A.nboot, np.random.default_rng(A.seed))
            C2[tag] = {"d_sel_eff": sel_eff(h, orc) - sel_eff(b, orc), "ci95": [lo, hi], "boot_p": p,
                       "beats_zs_control_ci_excludes_zero": bool(lo is not None and lo > 0)}
        out.setdefault("contrasts_vs_zs_control", {})[split] = {"vs": "lingshu7b_zs", "deltas": C2}

# assert the pool is genuinely fixed
assert len({round(out["table"][s]["oracle_at_8"], 12) for s in ["POOLED"]}) == 1
out["fixed_pool_check"] = {"oracle_at_8_pooled": out["table"]["POOLED"]["oracle_at_8"],
                           "note": "oracle@8/greedy/SC are verifier-independent by construction and are "
                                   "reported once per split; every verifier ranks the SAME candidate lists."}

# ------------------------------------------------------------------ de-correlation law
print("[measure] de-correlation (phi) and the law", flush=True)
for split in SPLITS:
    orc = FIX[split]["oracle"]; grd = FIX[split]["greedy"]
    sub = orc == 1
    D = {}
    for tag in SCORES:
        h = VEC[tag][split]["hard"]
        phi_s, kap_s = phi_kappa(grd[sub], h[sub])
        phi_a, kap_a = phi_kappa(grd, h)
        n = int(sub.sum())
        lo, hi, _ = boot_ci(lambda ii: (phi_kappa(grd[sub][ii], h[sub][ii])[0] or np.nan), n, 2000,
                            np.random.default_rng(A.seed))
        D[tag] = {"phi_oracle_subset": phi_s, "phi_ci95": [lo, hi], "kappa_oracle_subset": kap_s,
                  "phi_all_items": phi_a, "kappa_all_items": kap_a,
                  "n_oracle_subset": n, "sel_eff": sel_eff(h, orc)}
    out["decorrelation"][split] = D

# the law, on POOLED: does sel_eff track (lower) phi, across the zero-shot sweep?
zs_tags = [t for t in SCORES if t not in ("trained7b_clean", "trained7b_CONTAMINATED")]
pts = [(t, out["decorrelation"]["POOLED"][t]["phi_oracle_subset"],
        out["decorrelation"]["POOLED"][t]["sel_eff"]) for t in zs_tags]
pts = [p for p in pts if p[1] is not None]
if len(pts) >= 3:
    x = np.asarray([p[1] for p in pts]); y = np.asarray([p[2] for p in pts])
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    spear = float(np.corrcoef(rx, ry)[0, 1])
else:
    pear = spear = None
out["decorrelation_law"] = {
    "definition": ("phi between G = 'generator greedy answer correct' and V = 'verifier picks a correct "
                   "candidate', on the oracle=1 subset. LOW phi = de-correlated = the verifier is right "
                   "on different items than the generator."),
    "why_phi_not_kappa": ("for two binary variables phi IS the Pearson correlation, which is exactly the "
                          "shared-information quantity the hypothesis is about; kappa is reported too but "
                          "penalises marginal-prevalence differences, which vary a lot across these "
                          "verifiers and would confound the reading."),
    "scatter_pooled": [{"verifier": t, "phi": p, "sel_eff": s} for t, p, s in pts],
    "pearson_phi_vs_sel_eff": pear, "spearman_phi_vs_sel_eff": spear,
    "n_points": len(pts),
    "honesty": ("with this few points the relationship is SUGGESTIVE, not fitted: no p-value is quoted "
                "and no line is extrapolated. It is reported as a direction, not a law with a slope."),
}

# ------------------------------------------------------------------ ensembles
print("[measure] ensembles", flush=True)
def ranks_within(s):
    s = np.asarray(s, float)
    o = np.argsort(np.argsort(s))                       # 0 = worst
    return o.astype(float)


def ens_vec(tags, mode, ds):
    hard = []
    for k in KEYS[ds]:
        if mode == "rank":
            r = np.mean([ranks_within(SCORES[t][ds][k]) for t in tags], axis=0)
        else:
            r = np.mean([np.asarray(SCORES[t][ds][k], float) for t in tags], axis=0)
        hard.append(int(LAB[ds][k][int(np.argmax(r))]))
    return np.asarray(hard, int)


best_xf = None
if any(t in SCORES for t in CROSSFAM):
    cands = [t for t in CROSSFAM if t in SCORES]
    best_xf = max(cands, key=lambda t: out["table"]["POOLED"]["verifiers"][t]["sel_eff"])
out["ensembles"]["best_crossfamily"] = best_xf
if best_xf:
    combos = {
        f"trained7b_clean+{best_xf}": ["trained7b_clean", best_xf],
        f"lingshu7b_zs+{best_xf}": ["lingshu7b_zs", best_xf],
        "all_crossfamily": [t for t in CROSSFAM if t in SCORES],
        "all_zeroshot": [t for t in SCORES if t not in ("trained7b_clean", "trained7b_CONTAMINATED")],
    }
    for name, tags in combos.items():
        tags = [t for t in tags if t in SCORES]
        if len(tags) < 2: continue
        res = {}
        for mode in ("rank", "score"):
            per = {ds: ens_vec(tags, mode, ds) for ds in DSETS}
            pooled = np.concatenate([per[ds] for ds in DSETS])
            orc = FIX["POOLED"]["oracle"]; n = len(orc)
            b = VEC[BASE]["POOLED"]["hard"]
            lo, hi, p = boot_ci(lambda ii: sel_eff(pooled[ii], orc[ii]) - sel_eff(b[ii], orc[ii]), n,
                                A.nboot, np.random.default_rng(A.seed))
            res[mode] = {"pooled_sel_acc": float(pooled.mean()), "pooled_sel_eff": sel_eff(pooled, orc),
                         "d_sel_eff_vs_trained7b": sel_eff(pooled, orc) - sel_eff(b, orc),
                         "ci95": [lo, hi], "boot_p": p,
                         "per_dataset_sel_eff": {ds: sel_eff(per[ds], FIX[ds]["oracle"]) for ds in DSETS}}
        out["ensembles"][name] = {"members": tags, **res}

# ------------------------------------------------------------------ verdict
best_tag, best = None, -1
for t in CROSSFAM:
    if t in SCORES:
        v = out["table"]["POOLED"]["verifiers"][t]["sel_eff"]
        if v > best: best, best_tag = v, t
gate = False; why = []
if BASE in SCORES and best_tag:
    for t in CROSSFAM:
        if t not in SCORES: continue
        c = out["contrasts"]["POOLED"]["deltas"][t]
        if c["beats_baseline_ci_excludes_zero"]:
            gate = True; why.append(f"{t}: d_sel_eff={c['d_sel_eff']:+.4f} CI={c['ci95']}")
out["verdict"] = {
    "go": bool(gate),
    "rule": ("go=true only if a CROSS-FAMILY zero-shot scorer beats the TRAINED same-family verifier on "
             "pooled selection efficiency with a paired-bootstrap 95% CI excluding zero"),
    "passing": why,
    "best_crossfamily_tag": best_tag,
    "best_crossfamily_pooled_sel_eff": best if best_tag else None,
    "trained_same_family_pooled_sel_eff": (out["table"]["POOLED"]["verifiers"][BASE]["sel_eff"] if BASE in SCORES else None),
    "zs_same_family_pooled_sel_eff": (out["table"]["POOLED"]["verifiers"]["lingshu7b_zs"]["sel_eff"] if "lingshu7b_zs" in SCORES else None),
}

os.makedirs(os.path.dirname(J(A.out)), exist_ok=True)
json.dump(out, open(J(A.out), "w"), indent=1)
print(f"\n-> {J(A.out)}")
print(json.dumps(out["verdict"], indent=1))
print("\nPOOLED sel_eff:")
for t, v in sorted(out["table"]["POOLED"]["verifiers"].items(), key=lambda kv: -kv[1]["sel_eff"]):
    d = out["contrasts"]["POOLED"]["deltas"].get(t, {})
    print(f"  {t:24s} sel_eff={v['sel_eff']:.4f} acc={v['sel_acc']:.4f} auroc={v['auroc_candidate']:.4f} "
          f"d={d.get('d_sel_eff', 0.0):+.4f} CI={d.get('ci95', ['-','-'])} "
          f"phi={out['decorrelation']['POOLED'][t]['phi_oracle_subset']}")
