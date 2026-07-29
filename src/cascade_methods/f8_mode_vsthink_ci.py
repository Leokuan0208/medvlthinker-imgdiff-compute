#!/usr/bin/env python3
"""
f8_mode_vsthink_ci.py -- attach a paired-bootstrap 95% CI to the FLOP-NEGATIVE accuracy-max mode's
vs-always-32B-THINK delta, using the MEASURED per-sample 32B-THINK vectors (no estimate).

WHY.  The paper reports TWO accuracy-max operating points:
  * accuracy-max (F8 certified-veto on PMC + F10 team-objective L2D on open) -- 0.93x FLOP-eq,
    acc 0.5836 (Variant B / MMMU-excluded).  Its vs-always-32B-THINK delta was a POINT estimate
    (+0.0245) with NO CI.
  * accuracy-max+ (F3 confidence-advantage fusion on PMC) -- 1.25x FLOP-eq, acc 0.5862, whose
    vs-THINK CI was already computed in opentext_32b_think_full.py (that file's `accuracy_max`
    == paper_baselines' `am_ok` == the F3-fusion variant).
So only the F8+F10 (FLOP-negative) row lacked a measured-think CI.  This file closes that gap so
BOTH headline modes are CI-backed against always-32B-THINK.

WHAT.  Reconstruct the F8+F10 method per-sample delivered-ok vector EXACTLY as
method_final_mmmu_corrected.py does (paper_baselines.build_cells() + add_v2_vectors -> cells[k]["am2_ok"];
F8 certified weak-veto on PMC via beat32b_more.f8_veto, F10 team-objective consistent-L2D on the open
cells), and pair it against the always-32B-THINK per-sample vector:
  * MCQ think cells  -> real per-sample okT from paper_baselines.build_cells() (PATH_VQA_closed has no
    think dump -> okT = 32B-no-think, unchanged repo convention).
  * OPEN think cells -> MEASURED judged per-sample vector loaded via opentext_32b_think_full.measured_open_think()
    (ckpts/openvqa/strong_lingshu_think/ckpt_*_lingshu32b_think.judge.jsonl; the exact idx the method is
    scored on; judged through the IDENTICAL run_judge.py grader).
Paired bootstrap NBOOT=10000 over questions (paper_baselines.paired_ci), 5-fold cross-fit vectors,
Variant B (MMMU excluded) as the headline, with the full 9-cell suite + MCQ/open pools for context.
NO GPU.  Launch from repo root:  python3 src/cascade_methods/f8_mode_vsthink_ci.py
"""
import os, json
import numpy as np

import paper_baselines as PB
import method_final_mmmu_corrected as MFC     # add_v2_vectors -> cells[k]["am2_ok"] (F8 PMC + F10 open)
import opentext_32b_think_full as OTF         # measured_open_think() -> per-sample judged 32B-THINK okT

ROOT = PB.ROOT
OUT  = os.path.join(ROOT, "results/cascade_methods/artifacts/f8_mode_vsthink_ci.json")

MCQ_ORDER, OPEN_ORDER = PB.MCQ_ORDER, PB.OPEN_ORDER
ORDER = MCQ_ORDER + OPEN_ORDER
MMMU  = "MMMU-Medical-val"


def build():
    # ---- per-sample vectors: F8+F10 accuracy-max (am2) + every baseline (held-out 5-fold cross-fit) ----
    cells = PB.build_cells()
    MFC.add_v2_vectors(cells)                  # cells[k]["am2_ok"] = F8 veto (PMC) | F10 L2D (open) | 32B-nt else

    # ---- swap the open-text 32B-THINK estimate for the MEASURED per-sample judged vector ----
    meas = OTF.measured_open_think()
    for name in OPEN_ORDER:
        c = cells[name]; mk = meas[name]
        assert c["n"] == mk["n"], (name, c["n"], mk["n"])
        c["okT"] = mk["okT"]                    # was None (estimated); now real per-sample judged think
        c["think_real"] = True

    def am2(c):  return np.asarray(c["am2_ok"], float)   # F8+F10 accuracy-max method vector
    def thk(c):  return np.asarray(c["okT"], float)      # always-32B-THINK (MCQ real + open measured)
    def acc(v):  return float(np.asarray(v, float).mean())

    # ---- per-benchmark (all 9 cells reported; F8+F10 method vs measured 32B-THINK) ----
    per_bench = {}
    for k in ORDER:
        c = cells[k]
        per_bench[k] = dict(
            format=c["format"], n=int(c["n"]),
            method_acc=round(acc(am2(c)), 4),
            acc_32b_think_measured=round(acc(thk(c)), 4),
            acc_32b_nothink=round(acc(c["ok32"]), 4),
            policy=c.get("am2_choice"),
            d_vs_think=round(acc(am2(c)) - acc(thk(c)), 4),
            ci=PB.paired_ci(am2(c), thk(c)))

    # ---- pooled: Variant B (headline) + full 9-cell suite + MCQ/open pools ----
    POOLS = {
        "variant_b_mmmu_excluded": [k for k in ORDER if k != MMMU],
        "full_suite": ORDER,
        "mcq_only_mmmu_excluded": [k for k in MCQ_ORDER if k != MMMU],
        "open_only": OPEN_ORDER,
    }
    pooled = {}
    for lab, keys in POOLS.items():
        n = sum(cells[k]["n"] for k in keys)
        am = sum(acc(am2(cells[k])) * cells[k]["n"] for k in keys) / n
        aT = sum(acc(thk(cells[k])) * cells[k]["n"] for k in keys) / n
        aNT = sum(acc(cells[k]["ok32"]) * cells[k]["n"] for k in keys) / n
        A = np.concatenate([am2(cells[k]) for k in keys])
        T = np.concatenate([thk(cells[k]) for k in keys])
        pooled[lab] = dict(benchmarks=keys, n=int(n),
                           method_acc=round(am, 4),
                           acc_32b_think_measured=round(aT, 4),
                           acc_32b_nothink=round(aNT, 4),
                           d_vs_think=round(am - aT, 4),
                           ci=PB.paired_ci(A, T))

    hb = pooled["variant_b_mmmu_excluded"]
    headline = dict(
        mode="accuracy-max (F8 certified-veto on PMC + F10 team-objective L2D on open); FLOP-negative 0.93x",
        variant="B (MMMU excluded)",
        method_acc=hb["method_acc"],
        acc_32b_think_measured=hb["acc_32b_think_measured"],
        d_vs_think=hb["d_vs_think"],
        ci95=[hb["ci"]["lo"], hb["ci"]["hi"]],
        sig=hb["ci"]["sig"], n=hb["n"],
        note="Paper's accuracy-max row: acc 0.5836, 0.93x FLOP-eq. This CI replaces the prior POINT estimate "
             "(+0.0245, no CI) with a paired-bootstrap 95% CI against the MEASURED per-sample 32B-THINK.")

    out = dict(
        title="FLOP-negative accuracy-max mode (F8 certified-veto + F10 L2D) vs always-32B-THINK: "
              "paired-bootstrap 95% CI using the MEASURED per-sample 32B-THINK vectors (MCQ real + open "
              "measured-judged). Closes the one headline row that lacked a vs-THINK CI.",
        reproduce="python3 src/cascade_methods/f8_mode_vsthink_ci.py",
        no_gpu=True, no_fabricated_numbers=True, n_bootstrap=PB.NBOOT,
        method_vector="F8+F10 accuracy-max (method_final_mmmu_corrected accuracy_max_v2_F8F10), Variant B; "
                      "per-sample delivered-ok reconstructed held-out 5-fold cross-fit.",
        think_vector="always-32B-THINK per-sample: MCQ real okT (paper_baselines.build_cells; PATH_VQA_closed "
                     "has no think dump -> okT=32B-no-think), OPEN measured judged-think "
                     "(ckpts/openvqa/strong_lingshu_think/ckpt_*_lingshu32b_think.judge.jsonl, "
                     "opentext_32b_think_full.measured_open_think()).",
        headline=headline,
        per_benchmark=per_bench,
        pooled=pooled,
        data_gaps=[
            "PATH_VQA_closed (MCQ) still has no 32B-think dump -> okT = 32B-no-think (think~=no-think on "
            "perception closed); unrelated to the now-measured OPEN cells.",
            "SLAKE/VQA-RAD/PathVQA-open are IN-DOMAIN for the pooled4 verifier (unchanged caveat).",
            "paired bootstrap resamples QUESTIONS; held-out tau/lambda/veto thresholds are fixed per item, "
            "so calibration variance is not in the CI.",
        ])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    console(out)
    print(f"\nwrote {OUT}")
    return out


def console(out):
    print("=" * 110)
    print("FLOP-NEGATIVE accuracy-max (F8 veto + F10 L2D) vs always-32B-THINK -- MEASURED per-sample, paired bootstrap")
    print("=" * 110)
    h = out["headline"]
    print(f"  HEADLINE (Variant B, MMMU excluded, n={h['n']}):")
    print(f"    method {h['method_acc']:.4f}  vs 32B-think(measured) {h['acc_32b_think_measured']:.4f}"
          f"  d_vs_think {h['d_vs_think']:+.4f}  CI[{h['ci95'][0]:+.4f},{h['ci95'][1]:+.4f}]  "
          f"{'SIG' if h['sig'] else 'ns'}")
    print("\n  per-benchmark (F8+F10 method vs measured 32B-THINK):")
    print(f"    {'benchmark':<18}{'fmt':<5}{'n':>6}{'method':>9}{'32Bthk':>9}{'dTHINK':>9}{'CI':>22}")
    for k, r in out["per_benchmark"].items():
        ci = r["ci"]
        print(f"    {k:<18}{r['format']:<5}{r['n']:>6}{r['method_acc']:>9.4f}{r['acc_32b_think_measured']:>9.4f}"
              f"{r['d_vs_think']:>+9.4f}{('[%+.4f,%+.4f]' % (ci['lo'], ci['hi'])):>22}")
    print("\n  pooled:")
    for lab, p in out["pooled"].items():
        ci = p["ci"]
        print(f"    {lab:<26} n={p['n']:>6}  method {p['method_acc']:.4f}  dTHINK {p['d_vs_think']:+.4f}"
              f"  CI[{ci['lo']:+.4f},{ci['hi']:+.4f}]  {'SIG' if ci['sig'] else 'ns'}")


if __name__ == "__main__":
    build()
