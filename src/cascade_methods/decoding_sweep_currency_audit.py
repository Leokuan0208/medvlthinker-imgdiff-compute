#!/usr/bin/env python3
"""decoding_sweep_currency_audit.py -- WHY does a setting win under the LLM judge and lose under
exact match?

The 2026-08-13 sweep produced a sign flip for repetition_penalty=1.10:
    judge-currency oracle@8   rp11 - control = +0.0090  (CI-clean WIN)
    EM-currency    oracle@8   rp11 - control = -0.0132  (CI-clean LOSS)
and rp11 generates longer answers (6.33 vs 5.26 mean tokens). Either rp11 genuinely produces better
answers that EM's crude normaliser misses, or the judge is more permissive on longer strings and the
"win" is a GRADING effect rather than a GENERATION effect.

The judge is a deterministic function of (question, gold, candidate string) and every label is cached
by (ds, idx, norm(answer)), so there is NO per-setting judge bias for the SAME string. Any difference
therefore comes from the DISTRIBUTION of strings a setting emits. This script measures that:

  1. per-setting slot-level confusion between EM and the judge, and the judge RESCUE rate P(judge=1|EM=0)
  2. the same rescue rate stratified by generated-token length -- separates a COMPOSITIONAL shift
     (rp11 emits longer answers, longer answers get rescued more) from a per-length judge shift
  3. a length-MATCHED reweighting of the rescue rate: what rp11's rescue rate would be if its
     token-length histogram were the control's
  4. the item-level decomposition of the oracle@8 delta into EM-covered and judge-only-covered items
  5. a printed sample of the actual judge-only rescues, so the strings can be read rather than assumed

Outputs results/cascade_methods/artifacts/_decoding_sweep_currency_audit.json
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G                      # noqa: E402
from src.cascade_methods.decoding_sweep_analyse import (                 # noqa: E402
    load_judge, load_pool, DS, SWEEP)

ap = argparse.ArgumentParser()
ap.add_argument("--control", default="T07")
ap.add_argument("--settings", default="T03,T07,T13,minp01,rp11,rp105,minp005,topk20,topk50,"
                                      "topp09,topp095,T05,T10,T13minp010")
ap.add_argument("--n_examples", type=int, default=40)
ap.add_argument("--out", default="results/cascade_methods/artifacts/_decoding_sweep_currency_audit.json")
A = ap.parse_args()

lab = load_judge()
ref = G.load_items()
BINS = [(0, 2), (3, 4), (5, 6), (7, 9), (10, 14), (15, 999)]


def binof(t):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= t <= hi:
            return i
    return len(BINS) - 1


def seeds_of(setting):
    out = []
    for s in range(8):
        tag = f"{setting}_s{s}"
        if os.path.exists(os.path.join(SWEEP, f"ckpt_{DS[0]}_{tag}.jsonl")):
            p = load_pool(tag, strict=False)
            if p is not None:
                out.append((tag, p))
    return out


def slot_stats(pool):
    """Slot-level EM vs judge confusion, plus per-length-bin rescue counts."""
    n = em1j1 = em1j0 = em0j1 = em0j0 = 0
    bin_tot = np.zeros(len(BINS)); bin_resc = np.zeros(len(BINS))
    bin_em0 = np.zeros(len(BINS))
    toks = []
    for it in ref:
        r = pool[(it["ds"], it["idx"])]
        gt = r.get("gen_tokens_all") or [0] * len(r["preds"])
        for a, e, t in zip(r["preds"], r["oks_em"], gt):
            y = lab.get((it["ds"], it["idx"], G.norm(a)))
            if y is None:
                continue
            n += 1; toks.append(t); b = binof(t); bin_tot[b] += 1
            if e and y:
                em1j1 += 1
            elif e and not y:
                em1j0 += 1
            elif (not e) and y:
                em0j1 += 1; bin_resc[b] += 1; bin_em0[b] += 1
            else:
                em0j0 += 1; bin_em0[b] += 1
    return dict(n_slots=n, em1_judge1=em1j1, em1_judge0=em1j0, em0_judge1=em0j1, em0_judge0=em0j0,
                em_rate=(em1j1 + em1j0) / max(n, 1), judge_rate=(em1j1 + em0j1) / max(n, 1),
                judge_rescue_rate_given_EM0=em0j1 / max(em1j0 + em0j0 + em0j1 - em1j0, 1) if (em0j1 + em0j0) else 0.0,
                mean_tokens=float(np.mean(toks)) if toks else 0.0,
                bin_tot=bin_tot.tolist(), bin_em0=bin_em0.tolist(), bin_rescued=bin_resc.tolist())


def merge_bins(sts):
    tot = np.sum([s["bin_tot"] for s in sts], axis=0)
    em0 = np.sum([s["bin_em0"] for s in sts], axis=0)
    res = np.sum([s["bin_rescued"] for s in sts], axis=0)
    return tot, em0, res


settings = [s for s in A.settings.split(",")]
res = {"method": __doc__.strip().split("\n\n")[0],
       "judge": "src/labeling/run_judge.py -- MedVLThinker-32B, text-only, temp 0 (the project's own judge)",
       "em": "score_em() verbatim from src/labeling/run_openvqa.py / decoding_sweep_gen.py",
       "length_bins_tokens": [f"{lo}-{hi}" for lo, hi in BINS],
       "per_setting": {}}

pools = {}
for st in settings:
    sd = seeds_of(st)
    if not sd:
        continue
    pools[st] = sd
    sts = [slot_stats(p) for _, p in sd]
    tot, em0, resc = merge_bins(sts)
    res["per_setting"][st] = {
        "n_seeds": len(sd), "seeds": [t for t, _ in sd],
        "n_slots": int(sum(s["n_slots"] for s in sts)),
        "mean_tokens": float(np.mean([s["mean_tokens"] for s in sts])),
        "slot_EM_rate": float(np.mean([s["em_rate"] for s in sts])),
        "slot_JUDGE_rate": float(np.mean([s["judge_rate"] for s in sts])),
        "slot_confusion_summed": {
            "EM1_judge1": int(sum(s["em1_judge1"] for s in sts)),
            "EM1_judge0": int(sum(s["em1_judge0"] for s in sts)),
            "EM0_judge1": int(sum(s["em0_judge1"] for s in sts)),
            "EM0_judge0": int(sum(s["em0_judge0"] for s in sts))},
        "judge_rescue_rate_P_judge1_given_EM0": float(resc.sum() / max(em0.sum(), 1)),
        "by_token_bin": {"slots": tot.tolist(), "EM0_slots": em0.tolist(),
                         "rescued": resc.tolist(),
                         "rescue_rate": (resc / np.maximum(em0, 1)).tolist(),
                         "share_of_slots": (tot / max(tot.sum(), 1)).tolist()},
    }

# ---- (3) length-matched reweighting of the rescue rate onto the control's length histogram ----
C = A.control
if C in res["per_setting"]:
    ctot = np.array(res["per_setting"][C]["by_token_bin"]["slots"], float)
    cw = ctot / max(ctot.sum(), 1)
    for st, blk in res["per_setting"].items():
        r = np.array(blk["by_token_bin"]["rescue_rate"], float)
        em0 = np.array(blk["by_token_bin"]["EM0_slots"], float)
        # reweight this setting's PER-LENGTH rescue rates by the CONTROL's EM0 length mix
        cem0 = np.array(res["per_setting"][C]["by_token_bin"]["EM0_slots"], float)
        w = cem0 / max(cem0.sum(), 1)
        ok = em0 > 0
        blk["rescue_rate_LENGTH_MATCHED_to_control"] = float((r[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-12))
        blk["rescue_rate_RAW"] = blk["judge_rescue_rate_P_judge1_given_EM0"]

# ---- (4) item-level decomposition of the oracle@8 delta ----
def oracle_vecs(pool):
    em, ju = [], []
    for it in ref:
        r = pool[(it["ds"], it["idx"])]
        em.append(int(any(r["oks_em"])))
        ju.append(int(any(lab.get((it["ds"], it["idx"], G.norm(a)), 0) for a in r["preds"])))
    return np.array(em), np.array(ju)


if C in pools:
    cem = np.mean([oracle_vecs(p)[0] for _, p in pools[C]], axis=0)
    cju = np.mean([oracle_vecs(p)[1] for _, p in pools[C]], axis=0)
    res["oracle_decomposition_vs_control"] = {}
    for st, sd in pools.items():
        if st == C:
            continue
        sem = np.mean([oracle_vecs(p)[0] for _, p in sd], axis=0)
        sju = np.mean([oracle_vecs(p)[1] for _, p in sd], axis=0)
        res["oracle_decomposition_vs_control"][st] = {
            "em_oracle_delta": float(sem.mean() - cem.mean()),
            "judge_oracle_delta": float(sju.mean() - cju.mean()),
            "judge_only_coverage_share_setting": float(np.mean(sju - sem)),
            "judge_only_coverage_share_control": float(np.mean(cju - cem)),
            "delta_in_judge_only_coverage": float(np.mean(sju - sem) - np.mean(cju - cem)),
            "note": "judge_only_coverage = items the judge calls covered that EM calls uncovered. "
                    "If delta_in_judge_only_coverage accounts for the whole judge_oracle_delta while "
                    "em_oracle_delta is negative, the gain lives entirely in judge-minus-EM disagreement."}

# ---- (5) readable examples: slots the judge rescues that EM rejects ----
ex = {}
for st in ([C] + [s for s in ["rp11", "rp105"] if s in pools]):
    if st not in pools:
        continue
    tag, pool = pools[st][0]
    rows = []
    for it in ref:
        r = pool[(it["ds"], it["idx"])]
        gt = r.get("gen_tokens_all") or [0] * len(r["preds"])
        for a, e, t in zip(r["preds"], r["oks_em"], gt):
            y = lab.get((it["ds"], it["idx"], G.norm(a)))
            if y == 1 and e == 0:
                rows.append({"ds": it["ds"], "idx": it["idx"], "gold": r.get("gold", ""),
                             "pred": a, "tokens": t})
    rng = np.random.default_rng(20260814)
    pick = rng.choice(len(rows), size=min(A.n_examples, len(rows)), replace=False) if rows else []
    ex[st] = {"tag": tag, "n_judge_only_rescued_slots": len(rows),
              "sample": [rows[i] for i in sorted(pick)]}
res["judge_only_rescue_examples"] = ex

os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, A.out), "w"), indent=1)
print(f"wrote {A.out}")
for st, b in res["per_setting"].items():
    print(f"{st:12s} seeds {b['n_seeds']}  tok {b['mean_tokens']:.2f}  EM {b['slot_EM_rate']:.4f}  "
          f"JUDGE {b['slot_JUDGE_rate']:.4f}  rescue(raw) {b.get('rescue_rate_RAW', 0):.4f}  "
          f"rescue(len-matched) {b.get('rescue_rate_LENGTH_MATCHED_to_control', float('nan')):.4f}")
