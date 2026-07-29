#!/usr/bin/env python3
"""
opentext_32b_think_full.py -- replace the ESTIMATED open-text always-32B-THINK accuracy with a fully
MEASURED one and RECOMPUTE the method-vs-always-32B-THINK comparison end-to-end.

WHAT WAS ESTIMATED (and is now measured).  method_final.py / paper_baselines.py had no per-sample judged
32B-THINK dump on the open-text sets, so they ESTIMATED the open-text 32B-think accuracy as
    aT_open = judged_32B-no-think_acc + THINK_DELTA_OPEN   (modal-space delta measured on n=200)
    THINK_DELTA_OPEN = slake -0.195 / vqa_rad -0.120 / pathvqa -0.130
and therefore DROPPED the open cells from the vs-THINK paired-bootstrap CI (MCQ-pool CI only).

WHAT THIS DOES.  Lingshu-32B was run in THINK mode over the FULL evaluated open idx sets the method is
scored on (SLAKE 645, VQA-RAD 200, PathVQA 1500 -- the exact `load_open_rows` idx), and its `modal_pred`
was judged through the IDENTICAL judge the method uses for its open-text accuracy (src/labeling/run_judge.py,
neutral MedVLThinker/Qwen2.5-32B grader -> judge_ok).  We load that MEASURED per-sample judged think
vector, align it to the method's open cells, and recompute the vs-always-32B-THINK comparison:
  * per benchmark + pooled, BOTH method modes (compute-lean, accuracy-max),
  * Variant B (MMMU-excluded) and the full 9-cell suite,
  * paired-bootstrap 95% CIs -- now FULLY REAL per-sample (MCQ think real + open think measured), so the
    full-suite / Variant-B vs-THINK delta is no longer a point estimate.

REUSE.  All per-sample method/baseline vectors come from paper_baselines.build_cells() (held-out 5-fold,
same as the paper); only the open-text 32B-THINK vector is swapped estimate -> measured.  No GPU.
Launch from repo root:  python3 src/cascade_methods/opentext_32b_think_full.py
"""
import os, json
import numpy as np

import paper_baselines as PB
import integrated_pandora as IP
import integrated_method as IM

ROOT = PB.ROOT
OUT  = os.path.join(ROOT, "results/cascade_methods/artifacts/opentext_32b_think_full.json")
THINK_DIR = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu_think")

OPEN_KEY = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}
MCQ_ORDER, OPEN_ORDER = PB.MCQ_ORDER, PB.OPEN_ORDER
ORDER = MCQ_ORDER + OPEN_ORDER


def load_think_judge(dskey):
    """Measured judged 32B-THINK ok per idx: read run_judge.py output on the think dump (judge_ok),
    dedup last (mirrors load_judge_jsonl / PC.load_judge)."""
    p = os.path.join(THINK_DIR, f"ckpt_{dskey}_lingshu32b_think.judge.jsonl")
    m = {}
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    for l in open(p):
        if l.strip():
            r = json.loads(l); m[r["idx"]] = int(r["judge_ok"])
    return m


def measured_open_think():
    """Per open benchmark: aligned MEASURED judged-think per-sample vector (rows order = the method's
    open cells) + the scalar measured/estimated accuracies for reporting."""
    out = {}
    for name in OPEN_ORDER:
        dskey = OPEN_KEY[name]
        rows = IP.load_open_rows(dskey)                       # SAME idx/order the method open cells use
        tj = load_think_judge(dskey)
        idxs = [r["idx"] for r in rows]
        missing = [i for i in idxs if i not in tj]
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} evaluated idx missing a judged-think label "
                               f"(e.g. {missing[:5]}) -- generation/judging incomplete")
        okT = np.array([tj[i] for i in idxs], float)
        a32nt = float(np.array([r["strong"] for r in rows], float).mean())
        est = a32nt + IM.THINK_DELTA_OPEN[dskey]
        out[name] = dict(okT=okT, n=len(idxs),
                         measured_think_acc=float(okT.mean()),
                         prior_estimate=round(est, 4),
                         a32_nothink=round(a32nt, 4),
                         think_delta_measured=round(float(okT.mean()) - a32nt, 4),
                         think_delta_prior=IM.THINK_DELTA_OPEN[dskey],
                         judged_n=len(tj))
    return out


def build():
    cells = PB.build_cells()                                 # held-out per-sample method + baseline vectors
    meas = measured_open_think()

    # ---- swap the open-text 32B-THINK estimate for the MEASURED per-sample judged vector ----
    for name in OPEN_ORDER:
        c = cells[name]; mk = meas[name]
        assert c["n"] == mk["n"], (name, c["n"], mk["n"])
        c["okT"] = mk["okT"]                                  # was None (estimated); now real per-sample
        c["think_real"] = True
        c["think_acc_measured"] = mk["measured_think_acc"]

    # ---- method per-sample vectors per mode ----
    method_vec = {"compute_lean": lambda c: c["cl_ok"], "accuracy_max": lambda c: c["am_ok"]}

    def acc(v):    return float(np.asarray(v, float).mean())
    def think_vec(c): return c["okT"]                        # real for every cell now

    per_bench = {}
    for k in ORDER:
        c = cells[k]
        row = dict(format=c["format"], n=int(c["n"]),
                   acc_32b_think_measured=round(acc(think_vec(c)), 4),
                   acc_32b_nothink=round(acc(c["ok32"]), 4))
        if c["format"] == "open":
            row["acc_32b_think_prior_estimate"] = meas[k]["prior_estimate"]
            row["think_est_error"] = round(row["acc_32b_think_measured"] - meas[k]["prior_estimate"], 4)
        for mode, mv in method_vec.items():
            mvec = mv(c)
            row[mode] = dict(method_acc=round(acc(mvec), 4),
                             d_vs_think=round(acc(mvec) - acc(think_vec(c)), 4),
                             ci=PB.paired_ci(mvec, think_vec(c)))
        per_bench[k] = row

    # ---- pooled (sample-weighted point delta + paired-bootstrap CI over concatenated real vectors) ----
    POOLS = {"full_suite": ORDER,
             "variant_b_mmmu_excluded": [k for k in ORDER if k != "MMMU-Medical-val"],
             "mcq_only": MCQ_ORDER, "open_only": OPEN_ORDER,
             "mcq_only_mmmu_excluded": [k for k in MCQ_ORDER if k != "MMMU-Medical-val"]}
    pooled = {}
    for lab, keys in POOLS.items():
        n = sum(cells[k]["n"] for k in keys)
        aT = sum(acc(think_vec(cells[k])) * cells[k]["n"] for k in keys) / n
        aNT = sum(acc(cells[k]["ok32"]) * cells[k]["n"] for k in keys) / n
        block = dict(benchmarks=keys, n=int(n),
                     acc_32b_think_measured=round(aT, 4), acc_32b_nothink=round(aNT, 4))
        for mode, mv in method_vec.items():
            am = sum(acc(mv(cells[k])) * cells[k]["n"] for k in keys) / n
            A = np.concatenate([mv(cells[k]) for k in keys])
            T = np.concatenate([think_vec(cells[k]) for k in keys])
            block[mode] = dict(method_acc=round(am, 4),
                               d_vs_think=round(am - aT, 4),
                               ci=PB.paired_ci(A, T))
        pooled[lab] = block

    # ---- headline comparison to the PRIOR (estimate-based) numbers ----
    prior = dict(
        note="prior = the estimate-based vs-always-32B-THINK numbers from method_final.json full_suite "
             "(open-text 32B-think was estimated; open cells dropped from the CI).",
        open_think_estimate={k: meas[k]["prior_estimate"] for k in OPEN_ORDER},
        pooled_full_suite=dict(compute_lean=dict(d_vs_think=0.0117, acc_32b_think=0.5632),
                               accuracy_max=dict(d_vs_think=0.0238, acc_32b_think=0.5631)))

    def headline(pool_lab):
        p = pooled[pool_lab]
        return {mode: dict(method_acc=p[mode]["method_acc"],
                           acc_32b_think_measured=p["acc_32b_think_measured"],
                           d_vs_think=p[mode]["d_vs_think"],
                           ci95=[p[mode]["ci"]["lo"], p[mode]["ci"]["hi"]],
                           sig=p[mode]["ci"]["sig"], n=p["n"])
                for mode in ("compute_lean", "accuracy_max")}

    out = dict(
        title="MEASURED always-32B-THINK open-text accuracy + recomputed vs-always-32B-THINK comparison "
              "(no more estimate). Lingshu-32B-THINK over the FULL evaluated open idx sets, judged through "
              "the method's own run_judge.py (judge_ok); paired-bootstrap 95% CIs now fully real (MCQ think "
              "real + open think measured).",
        reproduce="python3 src/cascade_methods/opentext_32b_think_full.py",
        no_gpu=True, no_fabricated_numbers=True, n_bootstrap=PB.NBOOT,
        judge="src/labeling/run_judge.py (neutral MedVLThinker/Qwen2.5-32B grader, judge_ok on modal_pred "
              "vs gold) -- IDENTICAL to the judge producing the method's 32B-no-think open-text accuracy",
        measured_open_think_accuracy={
            k: dict(n=meas[k]["n"],
                    measured_judged_think_acc=round(meas[k]["measured_think_acc"], 4),
                    prior_estimate=meas[k]["prior_estimate"],
                    estimate_error=round(round(meas[k]["measured_think_acc"], 4) - meas[k]["prior_estimate"], 4),
                    acc_32b_nothink=meas[k]["a32_nothink"],
                    measured_think_delta=meas[k]["think_delta_measured"],
                    prior_think_delta=meas[k]["think_delta_prior"])
            for k in OPEN_ORDER},
        per_benchmark=per_bench,
        pooled=pooled,
        headline_full_suite=headline("full_suite"),
        headline_variant_b_mmmu_excluded=headline("variant_b_mmmu_excluded"),
        prior_estimate_based=prior,
        oracle_open_mode_unchanged={
            k: dict(measured_think_acc=round(meas[k]["measured_think_acc"], 4), a32_nothink=meas[k]["a32_nothink"],
                    oracle_still_picks_no_think=bool(meas[k]["measured_think_acc"] < meas[k]["a32_nothink"]))
            for k in OPEN_ORDER},
        data_gaps=[
            "PATH_VQA_closed (MCQ) still has no 32B-think dump -> its okT = 32B-no-think (think~=no-think on "
            "perception closed); this is the MCQ closed cell, unrelated to the now-measured OPEN cells.",
            "SLAKE/VQA-RAD/PathVQA-open are IN-DOMAIN for the pooled4 verifier (unchanged caveat).",
            "MMMU/MedXpert MCQ think vectors are the pre-existing real 32B-reason dumps (unchanged).",
        ])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    console(out)
    print(f"\nwrote {OUT}")
    return out


def console(out):
    print("=" * 118)
    print("MEASURED open-text always-32B-THINK accuracy (judged; vs the prior estimate)")
    print("=" * 118)
    print(f"  {'dataset':<14}{'n':>6}{'32Bnt':>8}{'think_MEAS':>12}{'think_EST':>11}{'est_err':>9}"
          f"{'dlt_meas':>10}{'dlt_prior':>10}")
    for k in OPEN_ORDER:
        m = out["measured_open_think_accuracy"][k]
        print(f"  {k:<14}{m['n']:>6}{m['acc_32b_nothink']:>8.3f}{m['measured_judged_think_acc']:>12.3f}"
              f"{m['prior_estimate']:>11.3f}{m['estimate_error']:>+9.3f}{m['measured_think_delta']:>+10.3f}"
              f"{m['prior_think_delta']:>+10.3f}")
    for tag, lab in [("FULL SUITE (9 cells)", "headline_full_suite"),
                     ("VARIANT B (MMMU-excluded)", "headline_variant_b_mmmu_excluded")]:
        print("\n" + "=" * 118)
        print(f"vs always-32B-THINK -- {tag}  (paired-bootstrap 95% CI, now FULLY MEASURED)")
        print("=" * 118)
        h = out[lab]
        for mode in ("compute_lean", "accuracy_max"):
            d = h[mode]
            print(f"  {mode:<14} method {d['method_acc']:.4f}  vs 32B-think(meas) {d['acc_32b_think_measured']:.4f}"
                  f"  dTHINK {d['d_vs_think']:+.4f}  CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]  "
                  f"{'SIG' if d['sig'] else 'ns'}   (n={d['n']})")
    print("\n  prior (estimate-based) full-suite dTHINK: compute-lean +0.0117 / accuracy-max +0.0238")
    print("  per-benchmark open-cell dTHINK (measured):")
    for k in OPEN_ORDER:
        r = out["per_benchmark"][k]
        cl, am = r["compute_lean"], r["accuracy_max"]
        print(f"    {k:<14} think_meas={r['acc_32b_think_measured']:.3f} (est {r['acc_32b_think_prior_estimate']:.3f}, "
              f"err {r['think_est_error']:+.3f})  CL dTHINK {cl['d_vs_think']:+.4f} "
              f"CI[{cl['ci']['lo']:+.4f},{cl['ci']['hi']:+.4f}]")


if __name__ == "__main__":
    build()
