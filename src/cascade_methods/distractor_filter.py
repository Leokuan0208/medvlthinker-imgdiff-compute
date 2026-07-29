#!/usr/bin/env python3
"""
distractor_filter.py -- OFFLINE (CPU only, no GPU) test of DISTRACTOR-FILTER rules on the diverse pool.

CONTEXT (the negative result we attack):
  Diverse generation (5-prompt x 3-temp portfolio, M~15) RAISES oracle@N (+0.027..+0.110) but the gain
  does NOT convert to selection accuracy: diverse prompts inject CONFIDENT SEMANTIC DISTRACTORS --
  plausible-but-wrong specialized answers that the pointwise (and pairwise) verifier mis-picks.
  Unfiltered-diverse-pointwise = 0.5494 pooled-3ds (barely > iid@8-pointwise 0.5191); PMC oracle +0.110
  but pointwise selection only +0.015.  Source: combine_diverse_pairwise.json / diverse_generation_gpu.json.

HYPOTHESIS: if we FILTER the confident distractors BEFORE selection, the diverse-gen coverage gain converts.

Every filter uses ONLY test-time-available signals -- candidate text, pointwise verifier score, and
cross-candidate AGREEMENT (how many pool candidates share a normalized answer). NONE uses correctness `oks`.
`oks` is used ONLY to score the final pick (identical scorer across every cell -> apples-to-apples).

FILTER RULES tested (each -> a single selected slot per question; ok = oks[slot]):
  (1) drop_lone_confident[q] : drop candidates that are HIGH verifier-score (>= pool q-quantile) AND
                               LOW agreement (singleton, count==1); then pointwise-argmax the survivors.
  (2) consensus              : keep only candidates whose answer appears >=2x in the pool, then argmax.
                               (fallback = full-pool argmax when nothing survives.)
  (3) rarity_downweight      : argmax( score * log(1+count) )   (down-weight rare answers).
                               (+ literal score*log(count) variant.)
  (4) topk_agreed[k]         : restrict pool to the k most-agreed distinct answers, then argmax.

For each: per-dataset + pooled (3ds excl-PMC to match the quoted baselines, and 4ds incl-PMC), with
PAIRED bootstrap 95% CIs vs (a) unfiltered-diverse-pointwise and (b) iid@8-pointwise. Bootstrap keys on a
GLOBALLY-UNIQUE uid=f"{ds}:{idx}" (raw idx collides across datasets -- a prior analyzer had that bug).
Also reports FILTERED-ORACLE (does the correct answer survive the filter?) to separate "genuine net gain"
from "traded coverage back for selectability (net zero)".  No fabricated numbers.

Launch from repo root:  python3 src/cascade_methods/distractor_filter.py
Writes: results/cascade_methods/artifacts/distractor_filter.json
"""
import os, json, math
from collections import Counter
import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)

DIVERSE_DUMP = {
    "vqa_rad_open": "ckpts/openvqa/diverse/ckpt_vqa_rad_open_lingshu7b_div.jsonl",
    "slake_open":   "ckpts/openvqa/diverse/ckpt_slake_open_lingshu7b_div.jsonl",
    "pathvqa_open": "ckpts/openvqa/diverse/ckpt_pathvqa_open_lingshu7b_div.jsonl",
    "pmc_content":  "ckpts/openvqa/diverse/ckpt_pmc_content_lingshu7b_div.jsonl",
}
IID_DUMP = {
    "vqa_rad_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_VQA_RAD_lingshu7b_VQA_RAD_content_content_sc8.jsonl",
    "slake_open":   "ckpts/mcq_gen_verify/lingshu7b/ckpt_SLAKE_lingshu7b_SLAKE_content_content_sc8.jsonl",
    "pathvqa_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_PATH_VQA_lingshu7b_content_sc8.jsonl",
    "pmc_content":  "ckpts/mcq_gen_verify/lingshu7b/ckpt_PMC_VQA_lingshu7b_content_sc8.jsonl",
}
DATASETS = ["vqa_rad_open", "slake_open", "pathvqa_open", "pmc_content"]
THREE = ["vqa_rad_open", "slake_open", "pathvqa_open"]   # excl PMC -> matches quoted baselines 0.549/0.519
B_BOOT = 3000
SEED = 20260706

# the two reference baselines every filter must beat
BASE_A = "diverse_pointwise"   # unfiltered-diverse-pointwise (~0.549 pooled-3ds)
BASE_B = "iid_pointwise"       # iid@8-pointwise               (~0.519 pooled-3ds)


def norm(s):
    return str(s).strip().lower()


def load_dump(path, cap=None):
    """idx -> (preds, oks, scores). Drops rows with any None score (matches prior analyzers)."""
    out = {}
    for l in open(J(path)):
        if not l.strip():
            continue
        r = json.loads(l)
        sc = r.get("scores")
        if sc is None or any(s is None for s in sc):
            continue
        preds = list(r["preds"]); oks = [int(x) for x in r["oks"]]; sc = [float(s) for s in sc]
        n = min(len(preds), len(oks), len(sc))
        if cap:
            n = min(n, cap)
        if n < 1:
            continue
        out[str(r["idx"])] = (preds[:n], oks[:n], sc[:n])
    return out


# ---------------------------------------------------------------- selection primitives
def argmax_sub(scores, keep):
    """slot with max score restricted to `keep` (first-max tie-break, matching np.argmax convention)."""
    best = keep[0]
    for i in keep:
        if scores[i] > scores[best]:
            best = i
    return best


def select_slot(method, preds, oks, scores):
    """Return the selected slot index for a given method. Uses ONLY preds/scores/agreement (never oks)."""
    n = len(preds)
    keys = [norm(p) for p in preds]
    cnt = Counter(keys)
    counts = [cnt[k] for k in keys]
    allslots = list(range(n))

    if method == "pointwise":                          # unfiltered diverse pointwise (BASE_A)
        return int(np.argmax(scores))

    if method == "modal_vote":                         # self-consistency (majority vote) reference
        # most frequent answer; tie-break by summed score of that answer's candidates
        best_key = max(cnt, key=lambda k: (cnt[k], sum(scores[i] for i in allslots if keys[i] == k)))
        keep = [i for i in allslots if keys[i] == best_key]
        return argmax_sub(scores, keep)

    if method.startswith("drop_lone_confident"):       # (1)
        q = float(method.split("@q")[1]) if "@q" in method else 0.5
        thresh = float(np.quantile(scores, q))
        keep = [i for i in allslots if not (counts[i] == 1 and scores[i] >= thresh)]
        if not keep:
            keep = allslots
        return argmax_sub(scores, keep)

    if method == "consensus":                          # (2)
        keep = [i for i in allslots if counts[i] >= 2]
        if not keep:
            keep = allslots                            # fallback: nothing repeats -> full-pool argmax
        return argmax_sub(scores, keep)

    if method == "consensus_modalfallback":            # (2) variant: fallback to majority vote
        keep = [i for i in allslots if counts[i] >= 2]
        if not keep:
            return select_slot("modal_vote", preds, oks, scores)
        return argmax_sub(scores, keep)

    if method == "rarity_log1p":                       # (3) primary: score*log(1+count)
        w = [scores[i] * math.log1p(counts[i]) for i in allslots]
        return int(np.argmax(w))

    if method == "rarity_logliteral":                  # (3) literal: score*log(count) (singleton->0)
        w = [scores[i] * math.log(counts[i]) if counts[i] > 0 else 0.0 for i in allslots]
        if max(w) <= 0.0:                              # all singletons -> fallback to pointwise
            return int(np.argmax(scores))
        return int(np.argmax(w))

    if method.startswith("topk_agreed"):               # (4)
        k = int(method.split("@k")[1])
        ranked = sorted(cnt.keys(),
                        key=lambda kk: (cnt[kk], max(scores[i] for i in allslots if keys[i] == kk)),
                        reverse=True)
        top_keys = set(ranked[:k])
        keep = [i for i in allslots if keys[i] in top_keys]
        if not keep:
            keep = allslots
        return argmax_sub(scores, keep)

    raise ValueError(method)


def filtered_keep(method, preds, scores):
    """Return the surviving slot set for a filter (to compute filtered-oracle / coverage retention)."""
    n = len(preds); keys = [norm(p) for p in preds]; cnt = Counter(keys); counts = [cnt[k] for k in keys]
    allslots = list(range(n))
    if method.startswith("drop_lone_confident"):
        q = float(method.split("@q")[1]) if "@q" in method else 0.5
        thresh = float(np.quantile(scores, q))
        keep = [i for i in allslots if not (counts[i] == 1 and scores[i] >= thresh)]
        return keep or allslots
    if method in ("consensus", "consensus_modalfallback"):
        keep = [i for i in allslots if counts[i] >= 2]
        return keep or allslots
    if method.startswith("topk_agreed"):
        k = int(method.split("@k")[1])
        ranked = sorted(cnt.keys(),
                        key=lambda kk: (cnt[kk], max(scores[i] for i in allslots if keys[i] == kk)),
                        reverse=True)
        top_keys = set(ranked[:k])
        keep = [i for i in allslots if keys[i] in top_keys]
        return keep or allslots
    return allslots  # rarity_* / pointwise / modal keep the whole pool (re-weight only)


# candidate methods (filters + references)
FILTERS = [
    "drop_lone_confident@q0.5", "drop_lone_confident@q0.75",
    "consensus", "consensus_modalfallback",
    "rarity_log1p", "rarity_logliteral",
    "topk_agreed@k2", "topk_agreed@k3",
]
REFS = ["pointwise", "modal_vote"]


# ---------------------------------------------------------------- build per-question records
def build_perq():
    perq = {ds: [] for ds in DATASETS}
    for ds in DATASETS:
        div = load_dump(DIVERSE_DUMP[ds])
        iid = load_dump(IID_DUMP[ds], cap=8)
        common = [k for k in div if k in iid]
        for k in common:
            dp, do, ds_ = div[k]
            ip, io, is_ = iid[k]
            rec = {"ds": ds, "idx": k, "uid": f"{ds}:{k}",
                   "iid_pointwise": io[int(np.argmax(is_))],
                   "oracle_iid8": max(io),
                   "oracle_diverse": max(do),
                   "n_uniq_div": len(set(norm(p) for p in dp))}
            # diverse selection methods
            for m in REFS + FILTERS:
                j = select_slot(m, dp, do, ds_)
                rec[m] = do[j]
            # filtered-oracle (coverage retained by each filter)
            for m in FILTERS:
                keep = filtered_keep(m, dp, ds_)
                rec["ORC_" + m] = max(do[i] for i in keep)
            perq[ds].append(rec)
    return perq


# ---------------------------------------------------------------- metrics + bootstrap
def mean_key(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def boot_ci(rows, key, rng, B=B_BOOT):
    if not rows:
        return (float("nan"), float("nan"), float("nan"))
    a = np.asarray([r[key] for r in rows], float); n = len(a); vals = np.empty(B)
    for b in range(B):
        vals[b] = a[rng.integers(0, n, n)].mean()
    return float(a.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_delta(rows, key_a, key_b, rng, B=B_BOOT):
    """Paired 95% CI of mean(key_a)-mean(key_b) over the SAME rows (resample questions jointly)."""
    if not rows:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    a = np.asarray([r[key_a] for r in rows], float); b = np.asarray([r[key_b] for r in rows], float)
    n = len(a); base = float(a.mean() - b.mean()); vals = np.empty(B)
    for k in range(B):
        s = rng.integers(0, n, n); vals[k] = a[s].mean() - b[s].mean()
    return {"delta": round(base, 4), "lo": round(float(np.percentile(vals, 2.5)), 4),
            "hi": round(float(np.percentile(vals, 97.5)), 4), "n": n}


def block(rows, rng):
    """sel_acc + CI for every method, filtered-oracle, and paired deltas vs both baselines."""
    out = {"n": len(rows)}
    methods = REFS + FILTERS
    out["sel_acc"] = {}
    for m in methods:
        mean, lo, hi = boot_ci(rows, m, rng)
        out["sel_acc"][m] = {"acc": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    out["oracle"] = {"iid8": round(mean_key(rows, "oracle_iid8"), 4),
                     "diverse": round(mean_key(rows, "oracle_diverse"), 4)}
    out["filtered_oracle"] = {m: round(mean_key(rows, "ORC_" + m), 4) for m in FILTERS}
    # deltas vs baselines
    out["delta_vs_diverse_pointwise"] = {m: boot_delta(rows, m, "pointwise", rng) for m in FILTERS}
    out["delta_vs_iid_pointwise"] = {m: boot_delta(rows, m, "iid_pointwise", rng) for m in FILTERS}
    return out


def beats(delta):
    return delta["lo"] > 0            # CI excludes 0 on the positive side


def main():
    perq = build_perq()
    for ds in DATASETS:
        print(f"[{ds:13s}] matched={len(perq[ds]):4d}  mean_uniq_div={np.mean([r['n_uniq_div'] for r in perq[ds]]):.1f}")

    pooled3 = [r for ds in THREE for r in perq[ds]]
    pooled4 = [r for ds in DATASETS for r in perq[ds]]

    rng = np.random.default_rng(SEED)
    RESULT = {
        "note": ("Distractor-filter test on diverse pool (Lingshu-7B, cap320, pooled4 pointwise verifier). "
                 "Filters use only candidate text + verifier score + cross-candidate agreement (never oks). "
                 "Correctness = oks (identical scorer all cells). CIs = 3000-sample PAIRED question bootstrap "
                 "keyed on uid=ds:idx. NOTE: pmc_content `oks` is loose option-letter matching (see findings) "
                 "-- interpret PMC with caution; the 3 open sets use exact-match/judge oks."),
        "methods": {"references": REFS, "filters": FILTERS,
                    "baselines": {"a_unfiltered_diverse_pointwise": BASE_A, "b_iid8_pointwise": BASE_B}},
        "per_dataset": {}, "pooled_3ds_exclPMC": {}, "pooled_4ds_inclPMC": {}, "pmc_deepdive": {}, "verdict": {},
    }

    for ds in DATASETS:
        RESULT["per_dataset"][ds] = block(perq[ds], np.random.default_rng(SEED))
    RESULT["pooled_3ds_exclPMC"] = {"datasets": THREE, **block(pooled3, np.random.default_rng(SEED))}
    RESULT["pooled_4ds_inclPMC"] = {"datasets": DATASETS, **block(pooled4, np.random.default_rng(SEED))}

    # ---- PMC deep dive: does any filter CONVERT the +0.11 oracle lift? ----
    pmc = perq["pmc_content"]; pmc_blk = block(pmc, np.random.default_rng(SEED))
    iid_pt_pmc = mean_key(pmc, "iid_pointwise")
    div_pt_pmc = pmc_blk["sel_acc"]["pointwise"]["acc"]
    RESULT["pmc_deepdive"] = {
        "n": len(pmc),
        "oracle_iid8": pmc_blk["oracle"]["iid8"], "oracle_diverse": pmc_blk["oracle"]["diverse"],
        "oracle_lift": round(pmc_blk["oracle"]["diverse"] - pmc_blk["oracle"]["iid8"], 4),
        "iid_pointwise": round(iid_pt_pmc, 4), "diverse_pointwise": div_pt_pmc,
        "filter_sel_acc": {m: pmc_blk["sel_acc"][m]["acc"] for m in FILTERS},
        "filtered_oracle": pmc_blk["filtered_oracle"],
        "delta_vs_diverse_pointwise": pmc_blk["delta_vs_diverse_pointwise"],
        "delta_vs_iid_pointwise": pmc_blk["delta_vs_iid_pointwise"],
        "converted_fraction_of_oracle_lift": {
            m: (round((pmc_blk["sel_acc"][m]["acc"] - iid_pt_pmc) /
                      (pmc_blk["oracle"]["diverse"] - pmc_blk["oracle"]["iid8"]), 3)
                if (pmc_blk["oracle"]["diverse"] - pmc_blk["oracle"]["iid8"]) else None)
            for m in FILTERS},
    }

    # ---- pick the best filter on the pooled-3ds (matches the quoted baselines) ----
    p3 = RESULT["pooled_3ds_exclPMC"]
    ranked = sorted(FILTERS, key=lambda m: p3["sel_acc"][m]["acc"], reverse=True)
    best = ranked[0]
    dA = p3["delta_vs_diverse_pointwise"][best]     # vs unfiltered diverse pointwise
    dB = p3["delta_vs_iid_pointwise"][best]         # vs iid@8 pointwise
    base_a_acc = p3["sel_acc"]["pointwise"]["acc"]
    base_b_acc = round(mean_key(pooled3, "iid_pointwise"), 4)

    # honesty: did the best filter keep coverage or trade it back?
    forc_best = p3["filtered_oracle"][best]
    coverage_note = (f"filtered-oracle {forc_best:.4f} vs diverse-oracle {p3['oracle']['diverse']:.4f} "
                     f"(iid-oracle {p3['oracle']['iid8']:.4f})")

    verdict_lines = [
        f"POOLED-3ds (excl PMC, n={p3['n']}) selection accuracy [oks], 95% CI:",
        f"  BASELINE a  unfiltered-diverse-pointwise = {base_a_acc:.4f}",
        f"  BASELINE b  iid@8-pointwise              = {base_b_acc:.4f}",
        f"  diverse-oracle ceiling = {p3['oracle']['diverse']:.4f}  (iid@8-oracle {p3['oracle']['iid8']:.4f})",
        "",
        "  filter sel_acc (pooled-3ds):",
    ]
    for m in ranked:
        s = p3["sel_acc"][m]; da = p3["delta_vs_diverse_pointwise"][m]; db = p3["delta_vs_iid_pointwise"][m]
        verdict_lines.append(
            f"    {m:26s} {s['acc']:.4f} {s['ci95']}  vsDiv {da['delta']:+.4f}[{da['lo']:+.3f},{da['hi']:+.3f}]"
            f"{'*' if beats(da) else ' '}  vsIID {db['delta']:+.4f}[{db['lo']:+.3f},{db['hi']:+.3f}]{'*' if beats(db) else ' '}")
    verdict_lines += [
        "",
        f"BEST filter (pooled-3ds) = {best}  acc={p3['sel_acc'][best]['acc']:.4f}",
        f"  vs (a) unfiltered-diverse-pointwise: {dA['delta']:+.4f} CI[{dA['lo']:+.4f},{dA['hi']:+.4f}]"
        f"  -> beats a: {'YES' if beats(dA) else 'NO (CI incl 0)'}",
        f"  vs (b) iid@8-pointwise:              {dB['delta']:+.4f} CI[{dB['lo']:+.4f},{dB['hi']:+.4f}]"
        f"  -> beats b: {'YES' if beats(dB) else 'NO (CI incl 0)'}",
        f"  coverage check: {coverage_note}",
        "",
        f"PMC conversion (n={RESULT['pmc_deepdive']['n']}): oracle lift +{RESULT['pmc_deepdive']['oracle_lift']:.3f}; "
        f"iid-pointwise {RESULT['pmc_deepdive']['iid_pointwise']:.3f} -> best-filter "
        f"{RESULT['pmc_deepdive']['filter_sel_acc'][best]:.3f} "
        f"(converted {RESULT['pmc_deepdive']['converted_fraction_of_oracle_lift'][best]} of the lift).",
    ]
    any_beats_both = any(beats(p3["delta_vs_diverse_pointwise"][m]) and beats(p3["delta_vs_iid_pointwise"][m])
                         for m in FILTERS)
    beats_a = beats(dA); beats_b = beats(dB)
    if beats_a and beats_b:
        concl = ("=> POSITIVE: the best filter genuinely nets out ahead of BOTH baselines (both CIs exclude 0).")
    elif beats_b and not beats_a:
        concl = ("=> PARTIAL: best filter beats iid@8 but is statistically tied with unfiltered-diverse-pointwise "
                 "(filtering does not add over plain diverse pointwise).")
    elif not beats_b:
        concl = ("=> NEGATIVE: best filter does not robustly beat iid@8-pointwise; the diverse-gen coverage "
                 "gain still does not convert -- filtering trades coverage back for selectability (net ~zero).")
    else:
        concl = "=> mixed."
    verdict_lines += ["", concl,
                      f"(any filter beating BOTH baselines w/ CI-excludes-0: {'YES' if any_beats_both else 'NO'})"]
    RESULT["verdict"] = {"best_filter": best,
                         "best_delta_vs_a_diverse_pointwise": dA,
                         "best_delta_vs_b_iid_pointwise": dB,
                         "beats_a": bool(beats_a), "beats_b": bool(beats_b),
                         "any_filter_beats_both": bool(any_beats_both),
                         "conclusion": concl, "lines": verdict_lines}

    for l in verdict_lines:
        print(l)

    outp = J("results/cascade_methods/artifacts/distractor_filter.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(RESULT, open(outp, "w"), indent=1)
    print(f"\n[dump] {outp}")


if __name__ == "__main__":
    main()
