#!/usr/bin/env python3
"""resolution_open_em.py -- SWEEP 2: the judge-free view of the open generator sweep.

run_openvqa.py's own scorer (normalized exact match + a short-answer 'contains' fallback) is
already applied to every sample at generation time and stored in each row's `oks`.  That gives a
per-cap table -- modal accuracy, oracle@8, greedy@T=0 -- that needs NO judge pass and NO GPU, and
it is computed identically for every arm.

It is a SECONDARY endpoint, not the endpoint.  The project's primary label is the LLM judge
(src/labeling/run_judge.py), because exact match under-counts synonyms; the published open cells
are judge-labelled.  This table exists so the resolution comparison has a matched, fully paired
answer even if the judge/verifier stage cannot get a card, and so the judge-labelled result can be
checked against a completely independent labelling rule.

    python3 src/cascade_methods/resolution_open_em.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
SWEEP = os.path.join(ROOT, "ckpts/openvqa/resolution_sweep")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
os.makedirs(OUT, exist_ok=True)
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
NEXP = {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500}
CONTROL = "cap320"
NBOOT, BSEED = 10000, 20260813


def arm(cap, tag):
    out = {}
    for ds in DS:
        p = os.path.join(SWEEP, f"ckpt_{ds}_{cap}_{tag}.jsonl")
        if not os.path.exists(p):
            return None
        d = {}
        for l in open(p):
            if l.strip():
                try:
                    r = json.loads(l)
                    d[r["idx"]] = r
                except Exception:
                    pass
        if len(d) < NEXP[ds]:
            return None
        out[ds] = d
    return out


def boot(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(BSEED)
    idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
    s = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    names = set()
    for f in glob.glob(os.path.join(SWEEP, "ckpt_*.jsonl")):
        b = os.path.basename(f)[:-6]
        for ds in DS:
            if b.startswith(f"ckpt_{ds}_"):
                names.add(b[len(f"ckpt_{ds}_"):])
    caps = sorted({n.rsplit("_", 1)[0] for n in names})
    arms = {}
    for cap in caps:
        for tag in ["s0", "s1", "s2", "t0"]:
            A = arm(cap, tag)
            if A is not None:
                arms[(cap, tag)] = A

    # canonical item order: the control arm if it exists, else any complete arm
    ref = arms.get((CONTROL, "s0")) or (list(arms.values())[0] if arms else None)
    if ref is None:
        print("no complete arms yet")
        return
    order = [(ds, i) for ds in DS for i in sorted(ref[ds].keys(), key=lambda x: str(x))]

    def vecs(A):
        modal, orc = [], []
        for ds, i in order:
            r = A[ds].get(i)
            if r is None:
                modal.append(0); orc.append(0)
                continue
            modal.append(int(r["modal_ok"]))
            orc.append(int(max(r["oks"]) == 1))
        return np.array(modal), np.array(orc)

    dsmask = {ds: np.array([d == ds for d, _ in order]) for ds in DS}
    res = {"_metric": "run_openvqa.py's normalized exact match + short-answer contains fallback, "
                      "applied at generation time and stored per sample as `oks`. NOT the judge.",
           "_n_items": len(order), "by_cap": {}, "vs_control": {}}
    for cap in caps:
        seeds = [t for t in ["s0", "s1", "s2"] if (cap, t) in arms]
        if not seeds and (cap, "t0") not in arms:
            continue
        row = {"seeds": seeds, "per_seed": {}}
        for t in seeds:
            m, o = vecs(arms[(cap, t)])
            row["per_seed"][t] = {"modal_em": round(float(m.mean()), 6),
                                  "oracle8_em": round(float(o.mean()), 6),
                                  "per_cell_oracle8_em": {ds: round(float(o[dsmask[ds]].mean()), 6)
                                                          for ds in DS},
                                  "per_cell_modal_em": {ds: round(float(m[dsmask[ds]].mean()), 6)
                                                        for ds in DS}}
        for q in ["modal_em", "oracle8_em"]:
            if not seeds:
                row[q] = {"mean": None, "sd": None, "_note": "no sampled seed complete at this cap"}
                continue
            v = np.array([row["per_seed"][t][q] for t in seeds])
            row[q] = {"mean": round(float(v.mean()), 6),
                      "sd": round(float(v.std(ddof=1)), 6) if len(v) > 1 else None}
        if (cap, "t0") in arms:
            m, _ = vecs(arms[(cap, "t0")])
            row["greedy_t0_em"] = {"all": round(float(m.mean()), 6),
                                   **{ds: round(float(m[dsmask[ds]].mean()), 6) for ds in DS}}
        res["by_cap"][cap] = row
    for cap in caps:
        if cap == CONTROL or (CONTROL, "s0") not in arms:
            continue
        seeds = [t for t in ["s0", "s1", "s2"] if (cap, t) in arms and (CONTROL, t) in arms]
        if not seeds and not ((cap, "t0") in arms and (CONTROL, "t0") in arms):
            continue
        blk = {"seeds_paired": seeds}
        for q, j in [("modal_em", 0), ("oracle8_em", 1)]:
            if not seeds:
                continue
            ds_, cis = [], []
            for t in seeds:
                a = vecs(arms[(CONTROL, t)])[j]
                b = vecs(arms[(cap, t)])[j]
                d, ci = boot(a, b)
                ds_.append(d); cis.append(ci)
            blk[q] = {"delta_mean_over_seeds": round(float(np.mean(ds_)), 6),
                      "delta_per_seed": [round(x, 6) for x in ds_],
                      "ci95_per_seed": [[round(c[0], 6), round(c[1], 6)] for c in cis],
                      "all_seeds_ci_exclude_zero": bool(all(c[0] > 0 or c[1] < 0 for c in cis))}
        if (cap, "t0") in arms and (CONTROL, "t0") in arms:
            a = vecs(arms[(CONTROL, "t0")])[0]
            b = vecs(arms[(cap, "t0")])[0]
            d, ci = boot(a, b)
            blk["greedy_t0_em"] = {"delta": round(d, 6), "ci95": [round(ci[0], 6), round(ci[1], 6)],
                                   "significant": bool(ci[0] > 0 or ci[1] < 0),
                                   "per_cell": {}}
            for ds in DS:
                m = dsmask[ds]
                dd, cc = boot(a[m], b[m])
                blk["greedy_t0_em"]["per_cell"][ds] = {
                    "n": int(m.sum()), "acc_control": round(float(a[m].mean()), 6),
                    "acc_at_cap": round(float(b[m].mean()), 6), "delta": round(dd, 6),
                    "ci95": [round(cc[0], 6), round(cc[1], 6)],
                    "significant": bool(cc[0] > 0 or cc[1] < 0)}
            # what a greedy-only change of this size is worth on the 8-cell macro, if it carried
            # through to the open cells' reported arm (it is NOT the reported arm -- see _read).
            blk["greedy_t0_em"]["macro8_equivalent_if_it_carried"] = {
                "value": round(d * 3.0 / 8.0, 6),
                "ci95": [round(ci[0] * 3.0 / 8.0, 6), round(ci[1] * 3.0 / 8.0, 6)],
                "project_significance_threshold": 0.0029,
                "_read": "ARITHMETIC, not a measurement of the deployed arm: the open cells carry "
                         "3/8 of the macro-8 but they are reported as best-of-8 + verifier, not as "
                         "a greedy decode. This says what the greedy shift would be worth IF it "
                         "carried through unchanged, and it is the reason the selected-accuracy "
                         "measurement is the one that decides."}
        res["vs_control"][cap] = blk
    json.dump(res, open(os.path.join(OUT, "open_em.json"), "w"), indent=1)
    for cap, r in res["by_cap"].items():
        print(f"{cap:9s} seeds={r['seeds']} modal_em={r['modal_em']} oracle8_em={r['oracle8_em']} "
              f"greedy_t0_em={r.get('greedy_t0_em', {}).get('all')}")
    for cap, b in res["vs_control"].items():
        if "oracle8_em" in b:
            print(f"  vs {CONTROL}: {cap:9s} d_oracle8={b['oracle8_em']['delta_mean_over_seeds']} "
                  f"allCIexcl0={b['oracle8_em']['all_seeds_ci_exclude_zero']}")
        if "greedy_t0_em" in b:
            g = b["greedy_t0_em"]
            print(f"  vs {CONTROL}: {cap:9s} d_greedy_t0={g['delta']:+.6f} {g['ci95']} "
                  f"{'SIG' if g['significant'] else 'n.s.'}")
    print("wrote", os.path.join(OUT, "open_em.json"))


if __name__ == "__main__":
    main()
