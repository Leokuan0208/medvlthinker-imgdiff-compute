#!/usr/bin/env python3
"""
choicewhy_pilot_bestofn.py -- the decisive Phase-1 measurement.

The programme's premise is that on MULTIPLE CHOICE best-of-N degenerates because every candidate is
a single letter, so a verifier has nothing to discriminate.  This measures, on the SAME items, with
N=8 samples at temperature 0.7:

    arm A  letter-only            vs   arm B2  (choice)(why) with a forced justification

  * coverage@8 (oracle)   -- is ANY of the 8 correct?  This is the HEADROOM a selector could reach.
  * self-consistency@8    -- majority vote over the letters: the training-free selector baseline.
  * n_distinct_letters    -- how many different options the 8 samples actually propose.
  * n_distinct_strings    -- how many textually different candidates there are (the thing a text
                             verifier can discriminate); for letter-only this is ~n_distinct_letters.
  * degenerate rate       -- fraction of items where all 8 samples give the SAME letter, i.e. where
                             best-of-N has literally nothing to choose between.

The key comparison is NOT accuracy: it is whether (choice)(why) changes the headroom
(coverage@8 - self-consistency@8) and the fraction of items on which a selector can act at all.

Appends to results/cascade_methods/artifacts/choicewhy_pilot_2026-08-03.json under "best_of_n_probe".
Run from repo root.
"""
import json, os, re
import numpy as np

CK = "ckpts/choicewhy_pilot"
OUT = "results/cascade_methods/artifacts/choicewhy_pilot_2026-08-03.json"
ARMS = ["A_letter_only", "B2_answer_first_forced"]
BENCHES = ["SLAKE", "PMC-VQA"]
RNG = np.random.default_rng(20260803)

LET = re.compile(r"\b([A-J])\b")
LEAD = re.compile(r"^\s*[*\"'(\[]*\s*([A-J])\s*(?=[).:,;\-—\]]|$|\n)")


def letter_of(t, arm):
    t = (t or "").strip()
    if "answer_first" in arm:
        m = LEAD.match(t)
        if m:
            return m.group(1)
    m = LET.search(t)
    return m.group(1) if m else "?"


def boot_ci(vals, n=10000):
    v = np.asarray(vals, float)
    bs = np.array([v[RNG.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


res = {}
for b in BENCHES:
    res[b] = {}
    for arm in ARMS:
        p = os.path.join(CK, f"ckpt_{b}_{arm}_sc8.jsonl")
        rows = [json.loads(l) for l in open(p) if l.strip()]
        cov, sc, ndl, nds, degen, acc1 = [], [], [], [], [], []
        for r in rows:
            g = r["gold"]
            outs = r["raw_outputs"]
            ls = [letter_of(o, arm) for o in outs]
            cov.append(int(any(x == g for x in ls)))
            vals, cnts = np.unique(ls, return_counts=True)
            top = vals[cnts == cnts.max()]
            sc.append(float(np.mean([x == g for x in top])))   # unbiased random tie-break
            ndl.append(len(set(ls)))
            nds.append(len(set(o.strip() for o in outs)))
            degen.append(int(len(set(ls)) == 1))
            acc1.append(int(ls[0] == g))
        res[b][arm] = {
            "n_items": len(rows), "n_samples": rows[0]["n_samples"], "temp": rows[0]["temp"],
            "one_sample_acc": float(np.mean(acc1)),
            "self_consistency_at_8": float(np.mean(sc)), "sc_ci95": boot_ci(sc),
            "coverage_at_8_oracle": float(np.mean(cov)), "cov_ci95": boot_ci(cov),
            "headroom_oracle_minus_sc": float(np.mean(cov) - np.mean(sc)),
            "mean_distinct_letters": float(np.mean(ndl)),
            "mean_distinct_strings": float(np.mean(nds)),
            "all_8_same_letter_rate": float(np.mean(degen)),
            "mean_gen_tokens": float(np.mean([np.mean(r["gen_tokens_all"]) for r in rows])),
        }

# pooled
pool = {}
for arm in ARMS:
    agg = {k: [] for k in ("cov", "sc", "ndl", "nds", "degen")}
    for b in BENCHES:
        p = os.path.join(CK, f"ckpt_{b}_{arm}_sc8.jsonl")
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l); g = r["gold"]
            ls = [letter_of(o, arm) for o in r["raw_outputs"]]
            agg["cov"].append(int(any(x == g for x in ls)))
            vals, cnts = np.unique(ls, return_counts=True)
            top = vals[cnts == cnts.max()]
            agg["sc"].append(float(np.mean([x == g for x in top])))
            agg["ndl"].append(len(set(ls))); agg["nds"].append(len(set(o.strip() for o in r["raw_outputs"])))
            agg["degen"].append(int(len(set(ls)) == 1))
    pool[arm] = {"n_items": len(agg["cov"]),
                 "self_consistency_at_8": float(np.mean(agg["sc"])), "sc_ci95": boot_ci(agg["sc"]),
                 "coverage_at_8_oracle": float(np.mean(agg["cov"])), "cov_ci95": boot_ci(agg["cov"]),
                 "headroom_oracle_minus_sc": float(np.mean(agg["cov"]) - np.mean(agg["sc"])),
                 "mean_distinct_letters": float(np.mean(agg["ndl"])),
                 "mean_distinct_strings": float(np.mean(agg["nds"])),
                 "all_8_same_letter_rate": float(np.mean(agg["degen"]))}
res["POOLED-SLAKE+PMCVQA"] = pool

d = json.load(open(OUT))
d["best_of_n_probe"] = {
    "question": "does the (choice)(why) format change the best-of-N picture on MCQ, i.e. is there "
                "headroom for a verifier to select into, and are the candidates distinguishable?",
    "setup": "N=8, temperature 0.7, seed 1234, same items and same prompts as the greedy arms; "
             "SLAKE-closed (416) + PMC-VQA (500) only",
    "dumps": "ckpts/choicewhy_pilot/ckpt_{bench}_{arm}_sc8.jsonl",
    "generator": "src/labeling/run_choicewhy_pilot.py --n_samples 8 --temp 0.7",
    "results": res,
}
json.dump(d, open(OUT, "w"), indent=2)
print(f"appended best_of_n_probe -> {OUT}\n")

hdr = f"{'group':<22}{'arm':<24}{'1samp':>7}{'SC@8':>7}{'orc@8':>7}{'head':>7}{'dLet':>6}{'dStr':>6}{'allsame':>9}"
print(hdr)
for g in list(BENCHES) + ["POOLED-SLAKE+PMCVQA"]:
    for arm in ARMS:
        r = res[g][arm]
        one = f"{r['one_sample_acc']:.3f}" if "one_sample_acc" in r else "  -  "
        print(f"{g:<22}{arm:<24}{one:>7}{r['self_consistency_at_8']:>7.3f}{r['coverage_at_8_oracle']:>7.3f}"
              f"{r['headroom_oracle_minus_sc']:>7.3f}{r['mean_distinct_letters']:>6.2f}"
              f"{r['mean_distinct_strings']:>6.2f}{r['all_8_same_letter_rate']:>9.3f}")
