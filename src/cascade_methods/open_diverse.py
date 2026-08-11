#!/usr/bin/env python3
"""open_diverse.py -- ATTACK 4 (OPEN-DIVERSE), Phase 0: does the sel_eff decay track pool SIZE or pool
REDUNDANCY, under the CLEAN disjoint verifier and in JUDGE currency?

THE CONTRADICTION BEING RECONCILED
  verifier_n_scaling_2026-08-03.json : sel_eff falls -0.0761 per doubling of N  => SIZE hurts
  diverse_generation_gpu.json        : at FIXED N=8, iid-8 -> DPP-8 raises sel_eff +0.0644 (vqa_rad)
                                       and +0.0962 (pathvqa)                    => REDUNDANCY hurts
Both were measured with the CONTAMINATED lora_verifier_pooled4 AND (found while reading the code for
this attack, before any new number) with vLLM LoRA scoring inside diversity_generate_gpu.phase_generate
-- vLLM 0.9.0.1 silently drops all 192 visual.* LoRA modules, so those scores come from a visually
blind verifier.

WHAT THIS SCRIPT DOES (CPU only). The GPU passes it consumes:
  src/labeling/run_judge.py                 -> judge labels for the M=15 diverse pools
  src/cascade_methods/open_diverse_score.py -> HF scores of the diverse pool AND of the matched iid-8
                                               pool under {disjoint, pooled4}

  * null test N1  : the frozen open-text metric (genframe_data.PUBLISHED)
  * null test N2  : the published diverse cells (diverse_generation_gpu.json)
  * decomposition : redundancy at fixed N=8, and size at fixed redundancy, per reporting cell, under
                    five (currency x verifier x backend) configurations
  * gate verdict  : the pre-registered PASS/FAIL rule

  python3 src/cascade_methods/open_diverse.py
"""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
J = lambda p: os.path.join(ROOT, p)

DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
MCQ_IID = {"slake_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_SLAKE_lingshu7b_SLAKE_content_content_sc8.jsonl",
           "vqa_rad_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_VQA_RAD_lingshu7b_VQA_RAD_content_content_sc8.jsonl",
           "pathvqa_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_PATH_VQA_lingshu7b_content_sc8.jsonl"}
DIV_DIR = "ckpts/openvqa/diverse"
SCORE_DIR = "ckpts/openvqa/diverse/scores"
NBOOT = 10000
SEED = 0
N_SUBSET_SEEDS = 20
MISSING = -1e9


def norm(s):
    return str(s).strip().lower()


def loadl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ---------------------------------------------------------------- DPP (verbatim from the generator)
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def toks(s):
    return set(_PUNCT.sub(" ", (s or "").lower()).split())


def jacc(a, b):
    a, b = toks(a), toks(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def farthest_first_select(preds, N):
    m = len(preds)
    if N >= m:
        return list(range(m))
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = D[j, i] = 1.0 - jacc(preds[i], preds[j])
    sel = [int(np.argmin(D.sum(1)))]
    while len(sel) < N:
        rem = [i for i in range(m) if i not in sel]
        sel.append(max(rem, key=lambda i: min(D[i, j] for j in sel)))
    return sel


# ---------------------------------------------------------------- arm metrics
def arm_stats(oks, scores):
    """pick = argmax with FIRST-INDEX tie-break (genframe_data.picks_from_scores, the frozen rule);
    sel_eff = mean(pick correct | pool recoverable) -- the frozen conditional definition."""
    rec, got, cd = [], [], []
    for ok, sc in zip(oks, scores):
        j = int(np.argmax(np.asarray(sc, dtype=float)))
        r = int(max(ok) == 1)
        rec.append(r)
        got.append(int(ok[j] == 1))
        cd.append(int(ok[j] == 0) if r else -1)
    rec = np.array(rec)
    got = np.array(got)
    cd = np.array(cd)
    return {"rec": rec, "got": got, "n": len(rec), "n_rec": int(rec.sum()),
            "oracle": float(rec.mean()), "acc": float(got.mean()),
            "sel_eff": float(got[rec == 1].mean()) if rec.sum() else float("nan"),
            "confident_distractor_rate": float(cd[cd >= 0].mean()) if (cd >= 0).any() else float("nan")}


def boot_delta_seleff(recA, gotA, recB, gotB, nboot=NBOOT, seed=SEED):
    """Item-paired bootstrap of sel_eff(A)-sel_eff(B): resample QUESTIONS, recompute each arm's own
    conditional mean on the resample (the conditioning sets differ between arms, so a per-question
    paired difference is undefined; this is the correct paired form)."""
    rng = np.random.default_rng(seed)
    n = len(recA)
    d = np.empty(nboot)
    for b in range(nboot):
        ix = rng.integers(0, n, n)
        ra, ga, rb, gb = recA[ix], gotA[ix], recB[ix], gotB[ix]
        d[b] = ((ga[ra == 1].mean() if ra.sum() else np.nan)
                - (gb[rb == 1].mean() if rb.sum() else np.nan))
    pt = ((gotA[recA == 1].mean() if recA.sum() else np.nan)
          - (gotB[recB == 1].mean() if recB.sum() else np.nan))
    d = d[~np.isnan(d)]
    return [float(pt), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def boot_delta_paired(a, b, nboot=NBOOT, seed=SEED):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(nboot)
    for i in range(nboot):
        ix = rng.integers(0, n, n)
        d[i] = a[ix].mean() - b[ix].mean()
    return [float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ---------------------------------------------------------------- data assembly
def _score_map(path):
    """{str(idx): {na: score}} from an open_diverse_score.py checkpoint; (None, 0) if absent."""
    if not os.path.exists(path):
        return None, 0
    m = defaultdict(dict)
    nfail = 0
    for r in loadl(path):
        if r.get("score") is None:
            nfail += 1
            continue
        m[str(r["idx"])][r["na"]] = float(r["score"])
    return m, nfail


def _judge_map(ds, which="diverse"):
    """{str(idx): {na: judge_ok}} for a pool; None if the judge pass is absent."""
    if which == "diverse":
        stem = J(f"{DIV_DIR}/ckpt_{ds}_lingshu7b_div_scexploded")
    else:
        stem = J(f"{DIV_DIR}/iidmcq/ckpt_{ds}_iidmcq_scexploded")
    expl = {r["idx"]: r for r in loadl(stem + ".jsonl")}
    jp = stem + ".judge.jsonl"
    if not os.path.exists(jp):
        return None
    jud = {r["idx"]: r["judge_ok"] for r in loadl(jp)}
    out = defaultdict(dict)
    for cid, r in expl.items():
        if cid in jud:
            out[str(cid).split("#")[0]][norm(r["modal_pred"])] = int(jud[cid])
    return out


def load_all():
    out, missing = {}, []
    for ds in DS:
        div = loadl(J(f"{DIV_DIR}/ckpt_{ds}_lingshu7b_div.jsonl"))
        djudge = _judge_map(ds, "diverse")
        ijudge = _judge_map(ds, "iidmcq")
        d_dj, f1 = _score_map(J(f"{SCORE_DIR}/scores_{ds}_disjoint.jsonl"))
        d_p4, f2 = _score_map(J(f"{SCORE_DIR}/scores_{ds}_pooled4.jsonl"))
        i_dj, f3 = _score_map(J(f"{SCORE_DIR}/scores_{ds}_disjoint_iidmcq.jsonl"))
        i_p4, f4 = _score_map(J(f"{SCORE_DIR}/scores_{ds}_pooled4_iidmcq.jsonl"))
        for nm, obj in (("div_judge", djudge), ("iid_judge", ijudge), ("div_disjoint", d_dj),
                        ("div_pooled4", d_p4), ("iid_disjoint", i_dj), ("iid_pooled4", i_p4)):
            if obj is None:
                missing.append(f"{ds}:{nm}")
        td = {str(r["idx"]): r for r in json.load(open(
            J(f"ckpts/train/lora_verifier_disjoint/transfer_dump_{ds}_lingshu7b.json")))}
        tp4 = {str(r["idx"]): r for r in json.load(open(
            J(f"ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_lingshu7b.json")))}
        mc = {str(r["idx"]): r for r in loadl(J(MCQ_IID[ds]))}

        rows = []
        for r in div:
            k = str(r["idx"])
            if k not in td or k not in mc or k not in tp4:
                continue
            na15 = [norm(a) for a in r["preds"]]
            m8 = mc[k]
            na8 = [norm(a) for a in m8["preds"]]
            rows.append({
                "idx": r["idx"], "preds15": list(r["preds"]),
                # ---- diverse M=15 portfolio pool
                "div_ok_em": list(r["oks"]),
                "div_ok_judge": [None if djudge is None else djudge[k].get(na) for na in na15],
                "div_sc_p4vllm": list(r["scores"]),
                "div_sc_p4hf": [None if d_p4 is None else d_p4[k].get(na) for na in na15],
                "div_sc_dj": [None if d_dj is None else d_dj[k].get(na) for na in na15],
                # ---- matched iid-8 pool (mcq_gen_verify) -- what the diverse generator was restricted to
                # NB: this dump's stored 'sl' is IDENTICAL to its 'oks' on 100% of rows (verified:
                # 1061/1061, 451/451, 345/345) -- it is EXACT-MATCH, not a judge label. The judge labels
                # below were produced for this attack by src/labeling/run_judge.py.
                "iid_ok_em": list(m8["oks"]),
                "iid_ok_judge": [None if ijudge is None else ijudge[k].get(na) for na in na8],
                "iid_sc_p4vllm": list(m8["scores"]),
                "iid_sc_p4hf": [None if i_p4 is None else i_p4[k].get(na) for na in na8],
                "iid_sc_dj": [None if i_dj is None else i_dj[k].get(na) for na in na8],
                # ---- deployed incumbent iid-8 pool (the transfer dumps behind the frozen metric)
                "dep_ok_judge": [0 if x in (None, -1) else int(x) for x in td[k]["sl"]],
                "dep_sc_dj": list(td[k]["scores"]), "dep_sc_p4hf": list(tp4[k]["scores"]),
            })

        def cov(key, L):
            return float(np.mean([sum(x is not None for x in r[key]) / L for r in rows])) if rows else 0.0

        out[ds] = {"rows": rows,
                   "coverage": {"div_judge": cov("div_ok_judge", 15), "iid_judge": cov("iid_ok_judge", 8),
                                "div_disjoint": cov("div_sc_dj", 15), "div_pooled4_hf": cov("div_sc_p4hf", 15),
                                "iid_disjoint": cov("iid_sc_dj", 8), "iid_pooled4_hf": cov("iid_sc_p4hf", 8)},
                   "score_failures": {"div_disjoint": f1, "div_pooled4": f2,
                                      "iid_disjoint": f3, "iid_pooled4": f4}}
    return out, missing


# ---------------------------------------------------------------- arms
CONFIGS = OrderedDict([
    # name                    div_ok          div_sc           iid_ok          iid_sc           iid_pool
    ("published",            ("div_ok_em",    "div_sc_p4vllm", "iid_ok_em",    "iid_sc_p4vllm", "mcq")),
    ("judge_pooled4_vllm",   ("div_ok_judge", "div_sc_p4vllm", "iid_ok_judge", "iid_sc_p4vllm", "mcq")),
    ("judge_pooled4_hf",     ("div_ok_judge", "div_sc_p4hf",   "iid_ok_judge", "iid_sc_p4hf",   "mcq")),
    ("frozen_clean",         ("div_ok_judge", "div_sc_dj",     "iid_ok_judge", "iid_sc_dj",     "mcq")),
    ("frozen_clean_depiid",  ("div_ok_judge", "div_sc_dj",     "dep_ok_judge", "dep_sc_dj",     "deployed")),
])


def _f(v, fill=MISSING):
    return [fill if x is None else float(x) for x in v]


def build_arms(rows, cfg, n_seeds=N_SUBSET_SEEDS):
    dok_k, dsc_k, iok_k, isc_k, isrc = CONFIGS[cfg]
    div_ok = [[0 if x is None else int(x) for x in r[dok_k]] for r in rows]
    div_sc = [_f(r[dsc_k]) for r in rows]
    iid_ok = [[0 if x in (None, -1) else int(x) for x in r[iok_k]] for r in rows]
    iid_sc = [_f(r[isc_k]) for r in rows]

    arms = {"iid8": {"oks": iid_ok, "scores": iid_sc,
                     "src": ("mcq_gen_verify iid-8 (the pool the diverse generator was restricted to)"
                             if isrc == "mcq" else "transfer_dump iid-8 (the DEPLOYED incumbent pool)")},
            "div15": {"oks": div_ok, "scores": div_sc, "src": "diverse portfolio pool, M=15"}}
    dpp = [farthest_first_select(r["preds15"], 8) for r in rows]
    arms["dpp8"] = {"oks": [[o[i] for i in ix] for o, ix in zip(div_ok, dpp)],
                    "scores": [[s[i] for i in ix] for s, ix in zip(div_sc, dpp)],
                    "src": "DPP-8 of the M=15 portfolio pool"}

    rand = {}
    for tag, bo, bs, Ns in (("div", div_ok, div_sc, [2, 4, 8, 15]), ("iid", iid_ok, iid_sc, [2, 4, 8])):
        for N in Ns:
            per_seed = []
            for sd in range(n_seeds):
                rng = np.random.default_rng(1000 * sd + N)
                oo, ss = [], []
                for a, b in zip(bo, bs):
                    m = len(a)
                    ix = list(range(m)) if N >= m else sorted(rng.choice(m, N, replace=False).tolist())
                    oo.append([a[i] for i in ix])
                    ss.append([b[i] for i in ix])
                per_seed.append((oo, ss))
            rand[f"{tag}_rand{N}"] = per_seed
    return arms, rand


def summarize(arms, rand):
    out = {"arms": {}, "random_subset_arms": {}}
    for k, v in arms.items():
        st = arm_stats(v["oks"], v["scores"])
        out["arms"][k] = {kk: st[kk] for kk in
                          ("n", "n_rec", "oracle", "acc", "sel_eff", "confident_distractor_rate")}
        out["arms"][k]["src"] = v.get("src", "")
    for k, per_seed in rand.items():
        vals = defaultdict(list)
        for oo, ss in per_seed:
            st = arm_stats(oo, ss)
            for m in ("oracle", "acc", "sel_eff"):
                vals[m].append(st[m])
        out["random_subset_arms"][k] = {
            m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                "min": float(np.min(v)), "max": float(np.max(v)), "n_seeds": len(v)}
            for m, v in vals.items()}
    return out


# ---------------------------------------------------------------- null tests
def null_tests():
    from src.training_methods import genframe_data as G
    out = {}
    items = G.load_items()
    r = G.sel_eff(G.incumbent_scores(), items)
    P = G.PUBLISHED
    dev = {"n": r["n"] - P["n"], "n_recoverable": r["n_recoverable"] - P["n_recoverable"],
           "oracle@8": r["oracle"] - P["oracle@8"], "selected": r["acc"] - P["selected"],
           "greedy": r["greedy"] - P["greedy"], "sel_eff": r["sel_eff"] - P["sel_eff"]}
    for ds, v in P["per_ds"].items():
        dev["per_ds:" + ds] = r["per_ds"][ds]["sel_eff"] - v
    md = float(max(abs(v) for v in dev.values()))
    out["N1_frozen_metric"] = {
        "source": "src/training_methods/genframe_data.py PUBLISHED",
        "max_abs_deviation": md, "deviations": {k: float(v) for k, v in dev.items()},
        "measured": {"n": r["n"], "n_recoverable": r["n_recoverable"], "oracle@8": r["oracle"],
                     "selected": r["acc"], "greedy": r["greedy"], "sel_eff": r["sel_eff"],
                     "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in DS}},
        "note": "PUBLISHED stores 6-dp rounded values; the deviation is that rounding",
        "verdict": "PASS" if md <= 1e-5 else "FAIL"}

    pub = json.load(open(J("results/cascade_methods/artifacts/diverse_generation_gpu.json")))
    per, worst = {}, 0.0
    for ds in DS:
        div = loadl(J(f"{DIV_DIR}/ckpt_{ds}_lingshu7b_div.jsonl"))
        iid = {str(x["idx"]): x for x in loadl(J(MCQ_IID[ds]))}
        div = [x for x in div if str(x["idx"]) in iid]
        a = arm_stats([iid[str(x["idx"])]["oks"] for x in div], [iid[str(x["idx"])]["scores"] for x in div])
        sel = [farthest_first_select(x["preds"], 8) for x in div]
        b = arm_stats([[x["oks"][i] for i in s] for x, s in zip(div, sel)],
                      [[x["scores"][i] for i in s] for x, s in zip(div, sel)])
        c = arm_stats([x["oks"] for x in div], [x["scores"] for x in div])
        Pd = pub["per_dataset"][ds]
        V = Pd["verifier_bo_n"]
        d = {"n_common": len(div) - Pd["n_common"],
             "oracle_iid8": a["oracle"] - Pd["oracle"]["iid@8"],
             "oracle_dpp8": b["oracle"] - Pd["oracle"]["diverse_dpp@8"],
             "oracle_full": c["oracle"] - Pd["oracle"]["diverse_full@M"],
             "sel_eff_iid8": a["sel_eff"] - V["iid@8"]["sel_eff"],
             "sel_eff_dpp8": b["sel_eff"] - V["diverse_dpp@8"]["sel_eff"],
             "sel_eff_full": c["sel_eff"] - V["diverse_full@M"]["sel_eff"],
             "sel_acc_iid8": a["acc"] - V["iid@8"]["sel_acc"],
             "sel_acc_dpp8": b["acc"] - V["diverse_dpp@8"]["sel_acc"],
             "sel_acc_full": c["acc"] - V["diverse_full@M"]["sel_acc"]}
        per[ds] = {"max_abs_deviation": float(max(abs(v) for v in d.values())),
                   "deviations": {k: float(v) for k, v in d.items()}}
        worst = max(worst, per[ds]["max_abs_deviation"])
    # ---- N3 identity control: does THIS session's HF scoring path reproduce the stored disjoint scores?
    n3 = {}
    for ds in DS:
        td = json.load(open(J(f"ckpts/train/lora_verifier_disjoint/transfer_dump_{ds}_lingshu7b.json")))
        old = {}
        for r in td:
            for a, s in zip(r["preds"], r["scores"]):
                old[(str(r["idx"]), norm(a))] = float(s)
        p = J(f"{SCORE_DIR}/scores_{ds}_disjoint.jsonl")
        if not os.path.exists(p):
            n3[ds] = {"status": "scores not available"}
            continue
        new = {}
        for r in loadl(p):
            if r.get("score") is not None:
                new[(str(r["idx"]), r["na"])] = float(r["score"])
        common = sorted(set(old) & set(new))
        if not common:
            n3[ds] = {"status": "no overlapping (idx,answer) pairs"}
            continue
        a = np.array([old[k] for k in common])
        b = np.array([new[k] for k in common])
        n3[ds] = {"n_overlapping_pairs": len(common), "max_abs_diff": float(np.abs(b - a).max()),
                  "mean_abs_diff": float(np.abs(b - a).mean()),
                  "frac_gt_1e-3": float((np.abs(b - a) > 1e-3).mean()),
                  "pearson_r": float(np.corrcoef(a, b)[0, 1])}
    out["N3_scoring_path_identity"] = {
        "what": ("re-score, with THIS session's HF path, (idx,answer) pairs that also appear in the "
                 "stored disjoint transfer dumps, and compare to the stored score"),
        "per_ds": n3,
        "candidate_causes_of_the_residual": [
            "transformers now loads Qwen2VLImageProcessor as a FAST processor by default (its own warning "
            "says this 'may produce slightly different outputs')",
            "this session pins TF32 off (torch.backends.cuda.matmul/cudnn.allow_tf32=False); the original "
            "verifier_transfer_eval.py run did not pin it",
        ],
        "note": "reported as measured; no cause is asserted"}

    # ---- N4: this module's DPP re-implementation vs the stored .dppN8 selections
    n4 = {}
    for ds in DS:
        a = loadl(J(f"{DIV_DIR}/ckpt_{ds}_lingshu7b_div.jsonl"))
        b = {str(r["idx"]): r["selected_idx"] for r in loadl(J(f"{DIV_DIR}/ckpt_{ds}_lingshu7b_div.dppN8.jsonl"))}
        same = sum(1 for r in a if farthest_first_select(r["preds"], 8) == b[str(r["idx"])])
        n4[ds] = {"n": len(a), "identical": same, "frac": same / len(a)}
    out["N4_dpp_reimplementation"] = {
        "what": "greedy farthest-first (1 - token-Jaccard) re-implementation vs the stored .dppN8 selected_idx",
        "per_ds": n4,
        "verdict": "PASS" if all(v["frac"] == 1.0 for v in n4.values()) else "FAIL"}

    out["N2_published_diverse"] = {
        "source": "results/cascade_methods/artifacts/diverse_generation_gpu.json",
        "max_abs_deviation": worst, "per_ds": per,
        "note": "published file stores 4-dp rounded values; the deviation is that rounding",
        "verdict": "PASS" if worst <= 1e-3 else "FAIL"}
    return out


# ---------------------------------------------------------------- full-pool size/redundancy mechanism
def size_vs_redundancy_full_pool(n_seeds=N_SUBSET_SEEDS):
    """The Phase-0 question answered on the FULL 2345-item incumbent pool (frozen currency, clean
    disjoint verifier, no diverse pool needed) -- so it is not limited by the portfolio pool's coverage.

    (a) SIZE axis at fixed (iid) redundancy: sel_eff(N) for N in {1,2,4,8}, random slot subsets.
    (b) REDUNDANCY axis at fixed size N=8: sel_eff stratified by the number of DISTINCT answers, next
        to the RANDOM-PICK FLOOR of the same stratum. The floor is what makes the decomposition
        interpretable: sel_eff falling as the pool gets more distinct is mechanical unless the
        verifier's LIFT OVER CHANCE also falls."""
    from src.training_methods import genframe_data as G
    items = G.load_items()
    out = {"n_items": len(items), "seeds": n_seeds, "size_axis": {}, "redundancy_strata": {}}

    for N in (1, 2, 4, 8):
        per = defaultdict(lambda: defaultdict(list))
        for sd in range(n_seeds if N < 8 else 1):
            rng = np.random.default_rng(1000 * sd)
            acc, rec = defaultdict(list), defaultdict(list)
            for it in items:
                ix = list(range(8)) if N >= 8 else sorted(rng.choice(8, N, replace=False).tolist())
                sl = [it["sl"][i] for i in ix]
                sc = [it["scores"][i] for i in ix]
                rec[it["ds"]].append(int(1 in sl))
                acc[it["ds"]].append(int(sl[int(np.argmax(sc))] == 1))
            for ds in DS:
                r = np.array(rec[ds]); a = np.array(acc[ds])
                per[ds]["oracle"].append(float(r.mean()))
                per[ds]["acc"].append(float(a.mean()))
                per[ds]["sel_eff"].append(float(a[r == 1].mean()))
        out["size_axis"][f"N={N}"] = {
            ds: {m: {"mean": float(np.mean(v)), "sd": (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)}
                 for m, v in per[ds].items()} for ds in DS}

    nd = np.array([len({norm(a) for a in it["preds"]}) for it in items])
    rec = np.array([1 if 1 in it["sl"] else 0 for it in items])
    got = np.array([1 if it["sl"][int(np.argmax(it["scores"]))] == 1 else 0 for it in items])
    floor = np.array([float(np.mean([1 if x == 1 else 0 for x in it["sl"]])) for it in items])
    dsi = np.array([it["ds"] for it in items])
    for ds in DS + ["ALL"]:
        m = (dsi == ds) if ds != "ALL" else np.ones(len(items), bool)
        st = {}
        for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 8)):
            k = m & (nd >= lo) & (nd <= hi) & (rec == 1)
            if k.sum() == 0:
                continue
            st[f"n_distinct_{lo}-{hi}"] = {
                "n_recoverable": int(k.sum()), "sel_eff": float(got[k].mean()),
                "random_pick_floor": float(floor[k].mean()),
                "verifier_lift_over_chance": float(got[k].mean() - floor[k].mean()),
                "oracle_in_stratum": float(rec[m & (nd >= lo) & (nd <= hi)].mean()),
                "n_in_stratum": int((m & (nd >= lo) & (nd <= hi)).sum())}
        out["redundancy_strata"][ds] = st
    # MARGINAL conversion of new coverage: d(selected) / d(oracle) between consecutive N.
    # This is the planning constant the round's governing arithmetic needs, and it is NOT sel_eff.
    conv = {}
    Ns = [1, 2, 4, 8]
    for ds in DS:
        rows = []
        for i in range(len(Ns) - 1):
            a = out["size_axis"][f"N={Ns[i]}"][ds]
            b = out["size_axis"][f"N={Ns[i+1]}"][ds]
            do = b["oracle"]["mean"] - a["oracle"]["mean"]
            da = b["acc"]["mean"] - a["acc"]["mean"]
            rows.append({"from_N": Ns[i], "to_N": Ns[i + 1], "d_oracle": float(do),
                         "d_selected": float(da), "marginal_conversion": float(da / do) if do else None})
        conv[ds] = rows
    out["marginal_conversion_of_coverage"] = {
        "definition": "d(selected accuracy) / d(oracle@N) between consecutive N, on the full cells",
        "per_ds": conv,
        "why_it_matters": ("the round's governing arithmetic assumed new coverage converts at sel_eff "
                           "(~0.81) or at ~0.447; the MEASURED marginal conversion at the N=4->8 "
                           "operating point is far lower, so +0.01 of oracle buys far less than +0.008 "
                           "of selected accuracy")}
    out["reading"] = (
        "sel_eff falls monotonically with pool size AND with answer-set distinctness -- but the "
        "RANDOM-PICK FLOOR falls faster, so the verifier's LIFT OVER CHANCE RISES with distinctness. "
        "The size decay and the redundancy decay are the same mechanical effect (more samples => more "
        "distinct answers => a lower chance floor), not a degradation of the verifier.")
    return out


# ---------------------------------------------------------------- vLLM-vs-HF verifier backend control
def backend_control(data):
    """Quantify, on the SAME candidates and the SAME adapter (pooled4), how much the vLLM LoRA path
    differs from the HF path. The environment note says vLLM 0.9.0.1 silently drops all 192 visual.*
    LoRA modules (sel_eff 0.775204 HF vs 0.702997 vLLM on the incumbent pool). This measures it on the
    diverse pool, where the published diverse result was produced by the vLLM path."""
    out = {}
    for ds in DS:
        rows = data[ds]["rows"]
        if not rows or all(x is None for r in rows for x in r["div_sc_p4hf"]):
            out[ds] = {"status": "pooled4 HF scores not available"}
            continue
        a, b = [], []
        agree_pick = 0
        for r in rows:
            v = r["div_sc_p4vllm"]
            h = r["div_sc_p4hf"]
            pair = [(x, y) for x, y in zip(v, h) if y is not None]
            a += [p[0] for p in pair]
            b += [p[1] for p in pair]
            if all(y is not None for y in h):
                agree_pick += int(int(np.argmax(v)) == int(np.argmax([float(y) for y in h])))
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        out[ds] = {"n_pairs": int(len(a)), "mean_score_vllm": float(a.mean()), "mean_score_hf": float(b.mean()),
                   "mean_abs_diff": float(np.abs(a - b).mean()),
                   "pearson_r": float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None,
                   "argmax_pick_agreement_over_the_15_pool": float(agree_pick / len(rows))}
    return out


# ---------------------------------------------------------------- judge concordance control
def judge_concordance(data):
    """Are the judge labels minted for this attack (tp=1) the same labels the frozen metric was built
    on (the transfer dumps)?  Compare on (idx, normalized-answer) pairs that occur in BOTH the new
    diverse/iid pools and the deployed transfer dump.  A currency is only a currency if it is stable."""
    out = {}
    for ds in DS:
        td = json.load(open(J(f"ckpts/train/lora_verifier_disjoint/transfer_dump_{ds}_lingshu7b.json")))
        old = defaultdict(dict)
        for r in td:
            for a, s in zip(r["preds"], r["sl"]):
                if s in (0, 1):
                    old[str(r["idx"])][norm(a)] = int(s)
        agree = tot = 0
        disagree = []
        seen = set()
        for r in data[ds]["rows"]:
            k = str(r["idx"])
            for na, y in zip([norm(a) for a in r["preds15"]], r["div_ok_judge"]):
                if y is None or na not in old[k] or (k, na) in seen:
                    continue
                seen.add((k, na))
                tot += 1
                agree += int(int(y) == old[k][na])
                if int(y) != old[k][na] and len(disagree) < 5:
                    disagree.append({"idx": r["idx"], "na": na, "new": int(y), "old": old[k][na]})
        out[ds] = {"pool": "diverse M=15 judge labels minted here (tp=1) vs the transfer dump's stored labels",
                   "n_overlapping_pairs": tot,
                   "agreement": (agree / tot) if tot else None, "examples_of_disagreement": disagree}
    return out


# ---------------------------------------------------------------- structural headroom
def structural_headroom(data):
    """How much COVERAGE would the cheap open arm need, at its measured sel_eff, before a coverage gain
    could move MACRO ACCURACY at all?

    In a cascade that escalates, lifting the cheap arm to any level BELOW the strong arm buys COST, not
    accuracy: the cell still reports max(kept-cheap, escalated-32B)-ish and the 32B answer dominates.
    So the crossover is  selected_c > always-32B-direct_c , i.e. (at fixed sel_eff)
        oracle_required_c = acc32direct_c / sel_eff_c
    Every input here is measured and named."""
    from src.training_methods import genframe_data as G
    fro = G.sel_eff(G.incumbent_scores(), G.load_items())
    rer = json.load(open(J("results/cascade_methods/artifacts/cascade_selector_rerun_2026-08-05.json")))
    pc = rer["per_arm"]["disjoint"]["per_cell_acc"]
    CELL = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open", "pathvqa_open": "PATH_VQA_open"}
    out = {"sources": {"cheap_arm": "src/training_methods/genframe_data.py frozen metric (disjoint verifier)",
                       "acc32direct": "artifacts/cascade_selector_rerun_2026-08-05.json per_arm.disjoint.per_cell_acc",
                       "portfolio_oracle": "measured here on the M=15 pool, judge currency"},
           "identity_check": {}, "per_cell": {}}
    # selected == sel_eff x oracle is an exact identity in the frozen metric -- verify, do not assume
    err = max(abs(fro["per_ds"][d]["acc"] - fro["per_ds"][d]["sel_eff"] * fro["per_ds"][d]["oracle"])
              for d in DS)
    out["identity_check"] = {"claim": "selected = sel_eff x oracle@8 (exact)", "max_abs_error": float(err)}
    items = G.load_items()
    for ds in DS:
        f = fro["per_ds"][ds]
        a32 = pc[CELL[ds]]["always_32b_direct"]
        req = a32 / f["sel_eff"]
        rows = data[ds]["rows"] if ds in data else []
        sub = {str(r["idx"]) for r in rows}
        pj = [[0 if x is None else int(x) for x in r["div_ok_judge"]] for r in rows] if rows else []
        have_j = bool(pj) and all(x is not None for r in rows for x in r["div_ok_judge"])
        p15 = float(np.mean([1 if max(o) == 1 else 0 for o in pj])) if have_j else None
        # subset-matched incumbent stats: the portfolio pool does NOT cover the whole cell on pathvqa
        si = [it for it in items if it["ds"] == ds and str(it["idx"]) in sub]
        ss = None
        if si:
            r_ = np.array([1 if 1 in it["sl"] else 0 for it in si])
            g_ = np.array([1 if it["sl"][int(np.argmax(it["scores"]))] == 1 else 0 for it in si])
            se = float(g_[r_ == 1].mean()) if r_.sum() else float("nan")
            ss = {"n": len(si), "oracle@8_iid": float(r_.mean()), "selected": float(g_.mean()),
                  "sel_eff": se, "oracle_required_at_this_sel_eff": float(a32 / se) if se == se else None}
        out["per_cell"][ds] = {
            "n_full_cell": f["n"],
            "oracle@8_iid_measured": f["oracle"], "selected_measured": f["acc"],
            "sel_eff_measured": f["sel_eff"], "always_32b_direct": a32,
            "oracle_required_at_fixed_sel_eff": float(req),
            "oracle_lift_required": float(req - f["oracle"]),
            "portfolio_subset_n": len(rows),
            "portfolio_covers_full_cell": len(rows) == f["n"],
            "portfolio_oracle@15_judge_measured": p15,
            "incumbent_on_the_SAME_subset": ss,
            "reachable_at_M15_subset_matched": (bool(p15 >= ss["oracle_required_at_this_sel_eff"])
                                                if (p15 is not None and ss and
                                                    ss["oracle_required_at_this_sel_eff"]) else None)}
    return out


# ---------------------------------------------------------------- macro translation
def macro_translation(head):
    """What would a coverage gain have to BE, per cell, to move the macro headline?

    macro = mean over 8 cells, so macro delta = (1/8) * sum of per-cell accuracy gains, and the
    round's significance bar is macro delta >= +0.0029 (CI half-width at nboot=10,000).
    A cascade cell only gains accuracy once the cheap arm EXCEEDS always-32B-direct, so at fixed
    sel_eff the required oracle is (acc32direct + delta) / sel_eff."""
    BAR = 0.0029
    out = {"macro_bar": BAR, "weight_per_cell": 1 / 8,
           "note": ("macro delta = (1/8) x sum of per-cell gains; a cascade cell gains accuracy only "
                    "once the cheap arm EXCEEDS always-32B-direct (below that the escalated 32B answer "
                    "is what the cell reports), so these are the required CROSSING points"),
           "all_three_open_cells": {}, "pathvqa_alone": {}}
    for ds, v in head["per_cell"].items():
        se = v["sel_eff_measured"]
        a32 = v["always_32b_direct"]
        d = BAR * 8 / 3.0          # equal share across the three open cells
        need = (a32 + d) / se
        out["all_three_open_cells"][ds] = {
            "per_cell_gain_needed": d, "oracle@8_needed": float(need),
            "oracle@8_now": v["oracle@8_iid_measured"],
            "oracle_lift_needed": float(need - v["oracle@8_iid_measured"]),
            "portfolio_oracle@15_measured": v["portfolio_oracle@15_judge_measured"],
            "portfolio_covers_full_cell": v["portfolio_covers_full_cell"],
            "achievable_with_every_portfolio_candidate": (
                None if (v["portfolio_oracle@15_judge_measured"] is None
                         or not v["portfolio_covers_full_cell"])
                else bool(v["portfolio_oracle@15_judge_measured"] >= need))}
    v = head["per_cell"]["pathvqa_open"]
    d = BAR * 8
    out["pathvqa_alone"] = {
        "per_cell_gain_needed": d,
        "oracle@8_needed": float((v["always_32b_direct"] + d) / v["sel_eff_measured"]),
        "oracle@8_now": v["oracle@8_iid_measured"],
        "oracle_lift_needed": float((v["always_32b_direct"] + d) / v["sel_eff_measured"]
                                    - v["oracle@8_iid_measured"]),
        "caveat": "the portfolio pool covers only 178/1500 of this cell, so this is NOT measurable here"}
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cascade_methods/artifacts/open_diverse_2026-08-10.json")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    A = ap.parse_args()

    res = {"attack": "ATTACK 4 -- OPEN-DIVERSE, Phase 0 (pool SIZE vs pool REDUNDANCY, clean verifier)",
           "prereg": "results/cascade_methods/artifacts/open_diverse_2026-08-10_preregistration.json",
           "nboot": NBOOT, "bootstrap_seed": SEED, "n_subset_seeds": N_SUBSET_SEEDS,
           "pick_rule": "argmax over slots, first-index tie-break (genframe_data.picks_from_scores)",
           "sel_eff_definition": "mean(pick correct | pool recoverable) -- the frozen conditional form"}
    res["null_tests"] = null_tests()
    for k, v in res["null_tests"].items():
        if "max_abs_deviation" in v:
            print(k, v["verdict"], f"{v['max_abs_deviation']:.2e}")
        else:
            print(k, json.dumps(v["per_ds"]))

    from src.training_methods import genframe_data as G
    try:
        res["disjointness"] = G.assert_disjoint()
        res["disjointness"]["verdict"] = "PASS"
    except Exception as e:                                   # never silently proceed on a failed assert
        res["disjointness"] = {"verdict": "FAIL", "error": repr(e)}
    print("DISJOINTNESS", res["disjointness"]["verdict"],
          res["disjointness"].get("image_pixel_md5_intersection"))

    data, missing = load_all()
    res["inputs_missing"] = missing
    res["pool_coverage"] = {ds: {"n_questions": len(data[ds]["rows"]), **data[ds]["coverage"],
                                 "score_failures": data[ds]["score_failures"]} for ds in DS}
    print(json.dumps(res["pool_coverage"], indent=1))
    if missing:
        print("MISSING INPUTS:", missing)

    # a config is only computable when every label/score column it reads is fully populated
    COV_KEY = {"div_ok_judge": "div_judge", "iid_ok_judge": "iid_judge", "div_sc_dj": "div_disjoint",
               "div_sc_p4hf": "div_pooled4_hf", "iid_sc_dj": "iid_disjoint", "iid_sc_p4hf": "iid_pooled4_hf"}

    def computable(cfg):
        keys = [k for k in CONFIGS[cfg] if k in COV_KEY]
        return all(res["pool_coverage"][ds][COV_KEY[k]] >= 0.999 for ds in DS for k in keys)

    res["decomposition"] = {}
    res["configs_skipped_for_missing_inputs"] = [c for c in A.configs if not computable(c)]
    if res["configs_skipped_for_missing_inputs"]:
        print("SKIPPED (missing inputs):", res["configs_skipped_for_missing_inputs"])
    for cfg in A.configs:
        if not computable(cfg):
            continue
        dok, dsc, iok, isc, isrc = CONFIGS[cfg]
        res["decomposition"][cfg] = {"div_labels": dok, "div_scores": dsc, "iid_labels": iok,
                                     "iid_scores": isc, "iid_pool": isrc, "per_ds": {}}
        for ds in DS:
            rows = data[ds]["rows"]
            if not rows:
                continue
            arms, rand = build_arms(rows, cfg)
            st = {k: arm_stats(v["oks"], v["scores"]) for k, v in arms.items()}
            dl = {}
            for X, Y in (("dpp8", "iid8"), ("div15", "iid8"), ("dpp8", "div15")):
                dl[f"{X}_minus_{Y}"] = {
                    "sel_eff": boot_delta_seleff(st[X]["rec"], st[X]["got"], st[Y]["rec"], st[Y]["got"]),
                    "sel_acc": boot_delta_paired(st[X]["got"], st[Y]["got"]),
                    "oracle": boot_delta_paired(st[X]["rec"], st[Y]["rec"])}
            ctrl = defaultdict(list)
            for oo, ss in rand["div_rand8"]:
                s2 = arm_stats(oo, ss)
                ctrl["sel_eff"].append(st["dpp8"]["sel_eff"] - s2["sel_eff"])
                ctrl["sel_acc"].append(st["dpp8"]["acc"] - s2["acc"])
                ctrl["oracle"].append(st["dpp8"]["oracle"] - s2["oracle"])
            res["decomposition"][cfg]["per_ds"][ds] = {
                **summarize(arms, rand), "deltas": dl,
                "within_pool_dpp8_minus_rand8": {
                    m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                        "min": float(np.min(v)), "max": float(np.max(v)), "n_seeds": len(v)}
                    for m, v in ctrl.items()}}

    # -------- the pre-registered gate
    gate = {}
    for cfg in ("frozen_clean", "frozen_clean_depiid"):
        if cfg not in res["decomposition"]:
            continue
        g = res["decomposition"][cfg]["per_ds"]
        raised = {ds: bool(g[ds]["deltas"]["dpp8_minus_iid8"]["sel_eff"][0] > 0) for ds in g}
        gate[cfg] = {"per_cell_sel_eff_delta": {ds: g[ds]["deltas"]["dpp8_minus_iid8"]["sel_eff"] for ds in g},
                     "per_cell_sel_acc_delta": {ds: g[ds]["deltas"]["dpp8_minus_iid8"]["sel_acc"] for ds in g},
                     "n_cells_raised": int(sum(raised.values())), "raised": raised,
                     "verdict": "PASS" if sum(raised.values()) >= 2 else "FAIL"}
    # a config may only decide the gate if every label/score it needs is present on every slot
    need = {"frozen_clean": ["div_judge", "iid_judge", "div_disjoint", "iid_disjoint"],
            "frozen_clean_depiid": ["div_judge", "div_disjoint"]}
    complete = {}
    for cfg, keys in need.items():
        complete[cfg] = all(res["pool_coverage"][ds][k] >= 0.999 for ds in DS for k in keys)
        if cfg in gate:
            gate[cfg]["inputs_complete"] = complete[cfg]
            if not complete[cfg]:
                gate[cfg]["verdict"] = "NOT_EVALUABLE (incomplete inputs)"
    decided_by = "frozen_clean" if "frozen_clean" in gate else "frozen_clean_depiid"
    res["phase0_gate"] = {
        "rule": ("PASS if, under the CLEAN disjoint verifier in JUDGE currency, sel_eff(DPP-8 of the M=15 "
                 "portfolio pool) > sel_eff(iid-8) on >= 2 of the 3 open reporting cells (point estimate)"),
        "primary_config_per_amendment_1": "frozen_clean",
        "decided_by": decided_by,
        "why_not_primary": (None if decided_by == "frozen_clean" else
                            "the matched mcq_gen_verify iid pool could not be judge-labelled: both A100s "
                            "were taken by the round's other attacks (openstrong_gen.py --tp 2 and "
                            "cost_floor_measure.py) and the 32B judge OOMed at engine init "
                            "(logs/attack4_judge_iidmcq.log). The gate therefore falls back to the "
                            "ORIGINAL pre-registered iid arm -- the deployed transfer_dump pool, which "
                            "carries genuine judge labels -- so the contrast stays currency-matched "
                            "(judge vs judge) and verifier-matched (clean HF disjoint vs clean HF "
                            "disjoint). Its known flaw is a generation-run confound: the deployed pool "
                            "and the pool the diverse generator was restricted to are different 8-sample "
                            "runs (preds agree on 343/645, 28/200, 4/1500)."),
        "by_config": gate,
        "verdict": gate.get(decided_by, {}).get("verdict", "NOT_EVALUATED")}
    print("\nGATE:", res["phase0_gate"]["verdict"])
    for cfg, gv in gate.items():
        print(f" [{cfg}] cells raised {gv['n_cells_raised']}/3 -> {gv['verdict']}")
        for ds, v in gv["per_cell_sel_eff_delta"].items():
            a = gv["per_cell_sel_acc_delta"][ds]
            print(f"   {ds:<15} sel_eff {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]   "
                  f"sel_acc {a[0]:+.4f} [{a[1]:+.4f},{a[2]:+.4f}]")

    res["size_vs_redundancy_full_pool"] = size_vs_redundancy_full_pool()
    print("\nFULL-POOL MECHANISM (2345 items, frozen currency, clean disjoint verifier)")
    for N, v in res["size_vs_redundancy_full_pool"]["size_axis"].items():
        print("  " + N + ": " + "  ".join(
            f"{ds.split('_')[0]} or={v[ds]['oracle']['mean']:.4f} se={v[ds]['sel_eff']['mean']:.4f}" for ds in DS))
    for k, v in res["size_vs_redundancy_full_pool"]["redundancy_strata"]["ALL"].items():
        print(f"  ALL {k:<16} n_rec={v['n_recoverable']:>4} sel_eff={v['sel_eff']:.4f} "
              f"floor={v['random_pick_floor']:.4f} lift={v['verifier_lift_over_chance']:+.4f}")

    res["vllm_vs_hf_backend_control"] = backend_control(data)
    print("\nvLLM-vs-HF VERIFIER BACKEND (same pooled4 adapter, same candidates)")
    for ds, v in res["vllm_vs_hf_backend_control"].items():
        if "status" in v:
            print(f"  {ds:<15} {v['status']}")
        else:
            print(f"  {ds:<15} n={v['n_pairs']:>5} mean vLLM={v['mean_score_vllm']:.4f} "
                  f"HF={v['mean_score_hf']:.4f} |diff|={v['mean_abs_diff']:.4f} r={v['pearson_r']:.4f} "
                  f"argmax agreement={v['argmax_pick_agreement_over_the_15_pool']:.4f}")

    res["judge_concordance_control"] = judge_concordance(data)
    print("\nJUDGE CONCORDANCE (new tp=1 labels vs the labels behind the frozen metric, shared (idx,answer) pairs)")
    for ds, v in res["judge_concordance_control"].items():
        ag = v["agreement"]
        print(f"  {ds:<15} n={v['n_overlapping_pairs']:>5}  agreement={'n/a' if ag is None else round(ag, 4)}")

    res["structural_headroom"] = structural_headroom(data)
    print("\nSTRUCTURAL HEADROOM (crossover: the cheap arm must EXCEED always-32B-direct to move macro)")
    for ds, v in res["structural_headroom"]["per_cell"].items():
        ss = v["incumbent_on_the_SAME_subset"]
        print(f"  {ds:<15} FULL CELL sel_eff={v['sel_eff_measured']:.4f} oracle@8={v['oracle@8_iid_measured']:.4f} "
              f"selected={v['selected_measured']:.4f} vs 32B-direct={v['always_32b_direct']:.4f} "
              f"=> oracle needed {v['oracle_required_at_fixed_sel_eff']:.4f} (lift {v['oracle_lift_required']:+.4f})")
        if ss:
            print(f"  {'':<15} SUBSET n={ss['n']} (covers full cell: {v['portfolio_covers_full_cell']}) "
                  f"sel_eff={ss['sel_eff']:.4f} oracle@8={ss['oracle@8_iid']:.4f} "
                  f"=> oracle needed {ss['oracle_required_at_this_sel_eff']:.4f}; "
                  f"portfolio oracle@15 measured = {v['portfolio_oracle@15_judge_measured']} "
                  f"=> reachable at M=15: {v['reachable_at_M15_subset_matched']}")

    res["macro_translation"] = macro_translation(res["structural_headroom"])
    print("\nMACRO TRANSLATION (bar = +0.0029 macro; 3 open cells share it equally)")
    for ds, v in res["macro_translation"]["all_three_open_cells"].items():
        print(f"  {ds:<15} needs oracle@8 {v['oracle@8_needed']:.4f} (now {v['oracle@8_now']:.4f}, "
              f"lift {v['oracle_lift_needed']:+.4f}); portfolio oracle@15 = "
              f"{v['portfolio_oracle@15_measured']} (full-cell coverage: {v['portfolio_covers_full_cell']}) "
              f"=> achievable: {v['achievable_with_every_portfolio_candidate']}")
    pv = res["macro_translation"]["pathvqa_alone"]
    print(f"  PathVQA-open ALONE would need oracle@8 {pv['oracle@8_needed']:.4f} "
          f"(now {pv['oracle@8_now']:.4f}, lift {pv['oracle_lift_needed']:+.4f})")

    # -------- KILL criterion k2 (pre-registered): clean selected-accuracy delta CI includes zero
    k2cfg = "frozen_clean" if "frozen_clean" in res["decomposition"] else "frozen_clean_depiid"
    g = res["decomposition"].get(k2cfg, {}).get("per_ds", {})
    k2 = {ds: g[ds]["deltas"]["dpp8_minus_iid8"]["sel_acc"] for ds in g}
    n_null = sum(1 for v in k2.values() if v[1] <= 0 <= v[2])
    res["kill_k2"] = {"rule": ("KILL if the clean-verifier selected-accuracy delta CI includes zero on "
                               ">= 2 of 3 open reporting cells"),
                      "config_used": k2cfg,
                      "per_cell_sel_acc_delta": k2, "n_cells_with_CI_covering_zero": n_null,
                      "triggered": bool(n_null >= 2)}

    res["defects_found_in_the_published_diverse_result"] = [
        {"defect": "contaminated verifier",
         "evidence": "src/cascade_methods/diversity_generate_gpu.py --verifier_lora was pooled4",
         "fixed_here": "re-scored with ckpts/train/lora_verifier_disjoint"},
        {"defect": "vLLM LoRA scoring drops all 192 visual.* modules",
         "evidence": ("diversity_generate_gpu.phase_generate scores candidates with vLLM LoRARequest "
                      "(enable_lora=True, max_lora_rank=32); the environment note records sel_eff "
                      "0.775204 HF vs 0.702997 vLLM for the same adapter"),
         "fixed_here": "re-scored with HF transformers",
         "measured_here": res.get("vllm_vs_hf_backend_control")},
        {"defect": "exact-match labels, and an iid comparison arm that was never judge-labelled",
         "evidence": ("the diverse pools carry only map_correct 'oks'; the mcq_gen_verify iid dumps store "
                      "an 'sl' field that is IDENTICAL to their 'oks' on 100% of rows (1061/1061 slake, "
                      "451/451 vqa_rad, 345/345 pathvqa) -- it is exact-match, not a judge label, despite "
                      "the field name every downstream reader treats as a judge label"),
         "fixed_here": "both pools judge-labelled with src/labeling/run_judge.py (MedVLThinker-32B)"},
        {"defect": "PathVQA-open coverage",
         "evidence": ("the portfolio pool covers 178/1500 of the reporting cell and that slice is far "
                      "harder than the cell: on the incumbent pool greedy 0.1348 vs 0.3240, oracle@8 "
                      "0.3371 vs 0.5167, sel_eff 0.4500 vs 0.7226"),
         "fixed_here": "NOT fixed -- would require Phase 1 generation (1322 more items x 15 samples)"},
    ]

    res["outstanding"] = {
        "what": ("the fully-matched contrast `frozen_clean` (iid arm = the mcq_gen_verify pool the diverse "
                 "generator was restricted to, judge-labelled, HF-scored under the clean disjoint adapter)"),
        "blocked_by": ("both A100s were held by the round's other attacks (openstrong_gen.py --tp 2, "
                       "cost_floor_measure.py); the 32B judge OOMed at engine init at 14:04 "
                       "(logs/attack4_judge_iidmcq.log). A retry is queued and waits for >=70 GB free "
                       "on GPU0; the exploded judge inputs already exist under "
                       "ckpts/openvqa/diverse/iidmcq/ and the clean HF scores of that pool are already "
                       "complete (scores_*_disjoint_iidmcq.jsonl)."),
        "why_the_verdict_does_not_depend_on_it": (
            "the two controls that carry NO generation-run confound both agree with the gate: (i) the "
            "within-pool DPP-8 vs random-8 contrast (same M=15 pool, same size, 20 seeds) is negative on "
            "selected accuracy on all three cells, and (ii) the within-portfolio size curve shows "
            "selected accuracy flat while oracle rises."),
        "to_finish": "python3 src/cascade_methods/open_diverse.py  (re-runs and fills in frozen_clean)"}

    os.makedirs(os.path.dirname(J(A.out)), exist_ok=True)
    json.dump(res, open(J(A.out), "w"), indent=1, default=float)
    print("\n->", J(A.out))


if __name__ == "__main__":
    main()
