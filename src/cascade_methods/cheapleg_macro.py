#!/usr/bin/env python3
"""cheapleg_macro.py -- run the FULL published cascade end-to-end with an ADAPTED cheap leg and
re-report the canonical MACRO headline (Variant B, 8 cells, 1/8 each).

WHAT VARIES BETWEEN ARMS.  Only the 7B generator.  Every mechanic -- the margin cascade, the F1 slice
router, F8's certified veto, the Pandora Weitzman draw policy, the F10 team-objective L2D rejector,
the 5-fold cross-fitting, the cost constants, the macro weighting, the bootstrap -- is the existing
code, called unmodified, exactly as src/cascade_methods/cascade_selector_rerun.py does it for the
selector.  Two redirections do the work:

  (1) MULTIPLE-CHOICE.  integrated_method.MEK / beat32b_fusion.MEK are pointed at a SHADOW directory
      whose `eval_results_lingshu7b_full` is a symlink to the arm's own MedEvalKit output and whose
      every other tag symlinks to the real MedEvalKit results.  No loader is patched; the harness
      simply reads a different 7B.
  (2) OPEN TEXT.  integrated_method.OPEN_VERIFIER_DIR / integrated_pandora.ADAPTER /
      beat32b_more.OPEN_VERIFIER_DIR are pointed at the arm's transfer dumps, produced by the FROZEN
      incumbent verifier (ckpts/train/lora_verifier_disjoint) scoring the arm's own 8-sample pools.

THE ONE UNAVOIDABLE PATCH, and why.  paper_baselines.build_cells contains four asserts that the F1
slice router still picks the PUBLISHED policy per benchmark ("F3_confadv" on PMC-VQA, "always_32b_nt"
elsewhere).  Those asserts are drift guards for the published 7B.  On a DIFFERENT 7B the router is
supposed to re-pick -- that is what the router is for, and if the adapted 7B ever beat the 32B on a
cell the router picking "always_7b" there would BE the result.  So this script loads paper_baselines
from source with exactly those four asserts replaced by a recorder, asserts that exactly four
substitutions were made, and REPORTS every choice the router made.  Nothing else is altered; the
patched module is verified to reproduce the published macro numbers on the base arm (--nulltest).

    python3 src/cascade_methods/cheapleg_macro.py --nulltest
    python3 src/cascade_methods/cheapleg_macro.py --arm base7b  --mek_tag cheapleg_base7b  \
        --open_dir ckpts/cheapleg/scores_base7b
    python3 src/cascade_methods/cheapleg_macro.py --arm adapt7b_s0 --mek_tag cheapleg_adapt7b_s0 \
        --open_dir ckpts/cheapleg/scores_adapt7b_s0
    python3 src/cascade_methods/cheapleg_macro.py --combine
"""
import argparse, importlib.util, json, os, re, sys, types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import integrated_method as IM          # noqa: E402
import integrated_pandora as IP         # noqa: E402
import beat32b_more as BB               # noqa: E402
import beat32b_fusion as BF             # noqa: E402

ROOT = IM.ROOT
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_cheapleg_macro_parts")
OUT = os.path.join(ART, "train_cheap_leg_2026-08-11_macro.json")
os.makedirs(PARTS, exist_ok=True)
NBOOT = 10000
SEED = 20260811

# every eval_results_* tag the loaders can ask for, and where the REAL one lives
REAL_MEK = IM.MEK
TAGS = ["lingshu7b_full", "lingshu7b_cap320", "lingshu32b_full", "lingshu32b_think",
        "lingshu32b_reason", "lingshu32b_reason_mmmu"]

ROUTER_CHOICES = []


def load_patched_paper_baselines():
    """paper_baselines with its four F1-router drift asserts replaced by a recorder.  Everything else
    is byte-identical; the substitution count is asserted."""
    src_path = os.path.join(_HERE, "paper_baselines.py")
    src = open(src_path).read()
    pat_pmc = 'assert choice == "F3_confadv", f"PMC accuracy-max choice changed: {choice}"'
    pat_oth = 'assert ch == "always_32b_nt", f"{name} F1 choice changed: {ch}"'
    assert src.count(pat_pmc) == 1 and src.count(pat_oth) == 1, "paper_baselines.py drifted"
    src = src.replace(pat_pmc, '_ROUTER_REC(("PMC_VQA", choice))')
    src = src.replace(pat_oth, '_ROUTER_REC((name, ch))')
    mod = types.ModuleType("paper_baselines_cheapleg")
    mod.__file__ = src_path
    mod.__dict__["_ROUTER_REC"] = ROUTER_CHOICES.append
    exec(compile(src, src_path, "exec"), mod.__dict__)
    sys.modules["paper_baselines_cheapleg"] = mod
    return mod


def build_shadow(mek_tag):
    """A directory of symlinks where only eval_results_lingshu7b_full points at the arm."""
    if mek_tag is None:
        return REAL_MEK
    shadow = os.path.join(ROOT, "ckpts/cheapleg/_mek_shadow", mek_tag)
    os.makedirs(shadow, exist_ok=True)
    for t in TAGS:
        dst = os.path.join(shadow, f"eval_results_{t}")
        src = os.path.join(REAL_MEK, f"eval_results_{mek_tag}" if t == "lingshu7b_full"
                           else f"eval_results_{t}")
        if os.path.islink(dst):
            os.unlink(dst)
        if os.path.exists(src):
            os.symlink(src, dst)
    assert os.path.exists(os.path.join(shadow, "eval_results_lingshu7b_full")), \
        f"missing MedEvalKit output eval_results_{mek_tag}"
    return shadow


def set_arm(mek_tag, open_dir):
    shadow = build_shadow(mek_tag)
    IM.MEK = shadow
    BF.MEK = shadow
    if open_dir:
        for ds in ("slake_open", "vqa_rad_open", "pathvqa_open"):
            p = os.path.join(ROOT, open_dir, f"transfer_dump_{ds}_lingshu7b.json")
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        IM.OPEN_VERIFIER_DIR = open_dir
        BB.OPEN_VERIFIER_DIR = open_dir
        IP.ADAPTER = open_dir
    return shadow


def run_arm(name, mek_tag, open_dir, nboot=NBOOT):
    ROUTER_CHOICES.clear()
    shadow = set_arm(mek_tag, open_dir)
    PB = load_patched_paper_baselines()
    import macro_average_headline as MAH
    import importlib
    importlib.reload(MAH)                      # MAH imports PB at module level -> rebind
    MAH.PB = PB
    MAH.OUT = os.path.join(PARTS, f"macro_{name}.json")
    MAH.NBOOT = nboot
    out = MAH.run()

    cells = MAH.build()
    vec = {}
    for k in MAH.ORDER_B:
        for s in MAH.SYSTEMS:
            vec[f"{k}|{s}"] = np.asarray(cells[k][MAH.SYS_KEY[s]], np.int8)
    np.savez_compressed(os.path.join(PARTS, f"vec_{name}.npz"), **vec)

    small = dict(
        arm=name, mek_tag=mek_tag, open_dir=open_dir, shadow=shadow,
        f1_router_choices=[list(x) for x in ROUTER_CHOICES],
        macro_acc=out["accuracy_levels"]["full_pool"]["macro_cells"],
        sw_acc=out["accuracy_levels"]["full_pool"]["sample_weighted"],
        open_only=out["accuracy_levels"]["subpools"]["open_only"],
        mcq_only=out["accuracy_levels"]["subpools"]["mcq_only"],
        per_cell_acc={k: out["per_cell_accuracy"][k]["accuracy"] for k in MAH.ORDER_B},
        per_cell_n={k: out["per_cell_accuracy"][k]["n"] for k in MAH.ORDER_B},
        deltas={m: {b: out["deltas"][m][b]["all8"]["macro_cells"] for b in MAH.HEADLINE_BASELINES}
                for m in MAH.METHODS},
        deltas_open={m: {b: out["deltas"][m][b]["open_only"]["macro_cells"]
                         for b in MAH.HEADLINE_BASELINES} for m in MAH.METHODS},
        deltas_mcq={m: {b: out["deltas"][m][b]["mcq_only"]["macro_cells"]
                        for b in MAH.HEADLINE_BASELINES} for m in MAH.METHODS},
        loo={m: {b: out["deltas"][m][b]["all8"]["macro_cells_leave_one_out"]
                 for b in MAH.HEADLINE_BASELINES} for m in MAH.METHODS},
        ratios_macro=out["cost"]["method_vs_baseline_ratios"]["as_charged"]["macro_cells"],
        ratios_macro_honest=out["cost"]["method_vs_baseline_ratios"]["honest_recost"]["macro_cells"],
        escalation=out["escalation"])
    json.dump(small, open(os.path.join(PARTS, f"summary_{name}.json"), "w"), indent=1, default=float)
    print(f"\nwrote {PARTS}/summary_{name}.json")
    print(f"  F1 router choices: {ROUTER_CHOICES}")
    return small


# ===================================================================================================
def boot_means(mat, nboot, rng):
    pats, cnt = np.unique(mat, axis=0, return_counts=True)
    n = mat.shape[0]
    return (rng.multinomial(n, cnt / n, size=nboot) @ pats) / n


def combine(names, nboot=NBOOT):
    import macro_average_headline as MAH
    S = {n: json.load(open(os.path.join(PARTS, f"summary_{n}.json"))) for n in names}
    V = {n: np.load(os.path.join(PARTS, f"vec_{n}.npz")) for n in names}
    order = MAH.ORDER_B
    open_b = MAH.OPEN_B
    mcq_b = [k for k in order if k not in open_b]

    rng = np.random.default_rng(SEED)
    colidx, per_cell_boot = {}, {}
    for k in order:
        cols = []
        for n in names:
            for s in MAH.SYSTEMS:
                colidx[(k, n, s)] = len(cols)
                cols.append(np.asarray(V[n][f"{k}|{s}"], float))
        per_cell_boot[k] = boot_means(np.column_stack(cols), nboot, rng)

    def macro_dist(n, s, keys):
        w = 1.0 / len(keys)
        return sum(per_cell_boot[k][:, colidx[(k, n, s)]] * w for k in keys)

    def point(n, s, keys):
        return float(np.mean([S[n]["per_cell_acc"][k][s] for k in keys]))

    def ci(dist, pt):
        lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
        return dict(delta=round(pt, 4), ci95=[round(lo, 4), round(hi, 4)],
                    sig=bool(lo > 0 or hi < 0),
                    verdict="WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE")

    pools = (("all8_macro", order), ("open_only_macro", open_b), ("mcq_only_macro", mcq_b))
    method_vs_base = {}
    for n in names:
        method_vs_base[n] = {}
        for m in MAH.METHODS:
            method_vs_base[n][m] = {}
            for bl in MAH.HEADLINE_BASELINES:
                method_vs_base[n][m][bl] = {
                    lab: ci(macro_dist(n, m, keys) - macro_dist(n, bl, keys),
                            point(n, m, keys) - point(n, bl, keys)) for lab, keys in pools}
    arm_vs_arm = {}
    pairs = [(a, b) for i, b in enumerate(names) for a in names[i + 1:]]
    for a, b in pairs:
        arm_vs_arm[f"{a} - {b}"] = {}
        for s in MAH.SYSTEMS:
            arm_vs_arm[f"{a} - {b}"][s] = {
                lab: ci(macro_dist(a, s, keys) - macro_dist(b, s, keys),
                        point(a, s, keys) - point(b, s, keys)) for lab, keys in pools}
    # per-cell arm-vs-arm on every system (the guardrail view)
    per_cell = {}
    for a, b in pairs:
        per_cell[f"{a} - {b}"] = {}
        for k in order:
            per_cell[f"{a} - {b}"][k] = {}
            for s in MAH.SYSTEMS:
                d = per_cell_boot[k][:, colidx[(k, a, s)]] - per_cell_boot[k][:, colidx[(k, b, s)]]
                pt = S[a]["per_cell_acc"][k][s] - S[b]["per_cell_acc"][k][s]
                per_cell[f"{a} - {b}"][k][s] = ci(d, pt)
    return S, method_vs_base, arm_vs_arm, per_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--mek_tag", default=None)
    ap.add_argument("--open_dir", default=None)
    ap.add_argument("--nulltest", action="store_true",
                    help="run the PUBLISHED arm through the patched pipeline and check it reproduces "
                         "cascade_selector_rerun_2026-08-05.json's `disjoint` arm exactly")
    ap.add_argument("--combine", nargs="*", default=None)
    ap.add_argument("--nboot", type=int, default=NBOOT)
    A = ap.parse_args()

    if A.nulltest:
        s = run_arm("_nulltest_published", None, "ckpts/train/lora_verifier_disjoint", A.nboot)
        ref = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))
        r = ref["per_arm"]["disjoint"]
        dev = {}
        for k in r["macro_acc"]:
            dev[f"macro_acc.{k}"] = abs(s["macro_acc"][k] - r["macro_acc"][k])
        for m in s["deltas"]:
            for b in s["deltas"][m]:
                dev[f"delta.{m}.{b}"] = abs(s["deltas"][m][b]["delta"] - r["deltas"][m][b]["delta"])
        mx = max(dev.values())
        print("\nNULL TEST vs cascade_selector_rerun_2026-08-05.json arm 'disjoint':")
        for k, v in sorted(dev.items(), key=lambda x: -x[1])[:8]:
            print(f"    {v:.3e}  {k}")
        print(f"  MAX ABS DEVIATION = {mx:.3e}  ({len(dev)} fields)")
        print(f"  F1 router choices on the published arm: {s['f1_router_choices']}")
        json.dump({"max_abs_deviation": mx, "n_fields": len(dev), "per_field": dev,
                   "router_choices": s["f1_router_choices"]},
                  open(os.path.join(PARTS, "nulltest.json"), "w"), indent=1)
        return

    if A.arm:
        run_arm(A.arm, A.mek_tag, A.open_dir, A.nboot)

    if A.combine is not None:
        names = A.combine or [f[len("summary_"):-len(".json")]
                              for f in sorted(os.listdir(PARTS))
                              if f.startswith("summary_") and not f.startswith("summary__")]
        S, mvb, ava, pc = combine(names, A.nboot)
        json.dump(dict(title="Attack B -- macro headline with an ADAPTED cheap leg (Variant B, "
                             "8 cells, 1/8 each).",
                       date="2026-08-11", arms=names, n_bootstrap=A.nboot, seed=SEED,
                       per_arm=S, method_vs_baseline_macro=mvb, arm_vs_arm=ava,
                       arm_vs_arm_per_cell=pc),
                  open(OUT, "w"), indent=1, default=float)
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
