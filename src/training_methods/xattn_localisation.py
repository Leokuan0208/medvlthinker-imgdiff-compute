#!/usr/bin/env python3
"""xattn_localisation.py -- ATTACK 1(b)'s mechanism check: does the learned cross-attention
actually LOOK somewhere sensible, or is it a uniform blur that adds capacity and no grounding?

The xattn arm scores a candidate by querying the candidate's language-side vector against the 6x6
pooled patch grid.  vision_verifier_fit.py --save_attn 1 stores the resulting (n_rows, 36)
attention distribution.  A verifier that fixed the laterality failure would have to put its mass on
DIFFERENT parts of the image for a candidate that says "left" than for one that says "right".

Measured, on the laterality items only:
  x_com     horizontal centre of mass of the attention, in [0, 1] (0 = image left edge)
  y_com     vertical centre of mass
  entropy   of the 36-way distribution, in nats; log(36) = 3.5835 is perfectly uniform
  peak_frac mass on the single most-attended patch (1/36 = 0.0278 is uniform)

The decisive statistic is PAIRED and WITHIN ITEM: for items whose candidate pool contains both a
"left"-bearing and a "right"-bearing candidate, x_com(right) - x_com(left).  Under real grounding
this is systematically non-zero; under a blur it is zero.  A sign convention note that matters
clinically: in radiology "left" usually denotes the PATIENT's left, which appears on the VIEWER's
right, so a genuinely grounded head could show either sign -- what matters is that it is not zero
and that it is consistent.

Significance by a within-item sign-flip permutation test (10,000 draws), which needs no
distributional assumption.
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import genframe_data as G   # noqa: E402
import visverif_lib as V    # noqa: E402

PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_visverif_parts")
LEFT = re.compile(r"\bleft\b", re.I)
RIGHT = re.compile(r"\bright\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="xattn")
    ap.add_argument("--P", type=int, default=6)
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(PARTS, "xattn_localisation.json"))
    A = ap.parse_args()

    files = sorted(glob.glob(os.path.join(PARTS, f"{A.arm}_seed*.npz")))
    files = [f for f in files if "attn" in np.load(f).files]
    if not files:
        print(f"no {A.arm} part files carrying an 'attn' array yet"); return
    print(f"[load] {len(files)} seeds with attention maps")

    _, rows, _ = G.load_cache("generator", "eval", layers=[], pooling=())
    P = A.P
    xs = (np.arange(P * P) % P) / (P - 1.0)
    ys = (np.arange(P * P) // P) / (P - 1.0)

    per_seed = []
    for f in files:
        att = np.load(f)["attn"].astype(np.float64)          # (n_rows, P*P)
        att = att / att.sum(1, keepdims=True)
        xcom = att @ xs
        ycom = att @ ys
        ent = -(att * np.log(att + 1e-12)).sum(1)
        peak = att.max(1)

        # rows -> item, and the candidate text of each row
        byitem = defaultdict(list)
        for i, r in enumerate(rows):
            byitem[(r["ds"], r["idx"])].append(i)

        d, nitems = [], 0
        for k, ii in byitem.items():
            l = [i for i in ii if LEFT.search(str(rows[i]["ans"]))]
            rr = [i for i in ii if RIGHT.search(str(rows[i]["ans"]))]
            if l and rr:
                d.append(float(np.mean(xcom[rr]) - np.mean(xcom[l])))
                nitems += 1
        d = np.asarray(d)
        rec = {"seed": int(f.split("_seed")[-1].split(".")[0]),
               "n_rows": int(att.shape[0]),
               "entropy_mean": float(ent.mean()), "entropy_uniform": float(np.log(P * P)),
               "entropy_frac_of_uniform": float(ent.mean() / np.log(P * P)),
               "peak_mass_mean": float(peak.mean()), "peak_mass_uniform": 1.0 / (P * P),
               "xcom_mean": float(xcom.mean()), "xcom_sd": float(xcom.std()),
               "ycom_sd": float(ycom.std()),
               "n_items_with_left_and_right_candidates": nitems}
        if nitems:
            rng = np.random.default_rng(0)
            obs = float(d.mean())
            perm = np.array([float((d * rng.choice([-1.0, 1.0], len(d))).mean())
                             for _ in range(A.nperm)])
            rec["paired_xcom_right_minus_left"] = {
                "mean": obs, "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
                "n_items": int(nitems),
                "perm_p_two_sided": float((np.abs(perm) >= abs(obs)).mean()),
                "ci95_from_perm_null": [float(np.percentile(perm, 2.5)),
                                        float(np.percentile(perm, 97.5))]}
        per_seed.append(rec)
        print(f"  seed {rec['seed']}: entropy {rec['entropy_mean']:.4f}/{rec['entropy_uniform']:.4f} "
              f"({100*rec['entropy_frac_of_uniform']:.1f}% of uniform), peak {rec['peak_mass_mean']:.4f} "
              f"(uniform {rec['peak_mass_uniform']:.4f}), "
              f"dx {rec.get('paired_xcom_right_minus_left', {}).get('mean')}", flush=True)

    ents = [r["entropy_frac_of_uniform"] for r in per_seed]
    dxs = [r["paired_xcom_right_minus_left"]["mean"] for r in per_seed
           if "paired_xcom_right_minus_left" in r]
    ps = [r["paired_xcom_right_minus_left"]["perm_p_two_sided"] for r in per_seed
          if "paired_xcom_right_minus_left" in r]
    rep = {
        "what": "does the xattn head's learned attention localise, and does it move between "
                "'left'- and 'right'-bearing candidates on the same image?",
        "date": "2026-08-12", "code": "src/training_methods/xattn_localisation.py",
        "grid": f"{P}x{P} adaptively pooled patch grid",
        "sign_convention_caveat": "in radiology 'left' usually means the PATIENT's left, which is "
                                  "on the VIEWER's right. A grounded head may show either sign; the "
                                  "test is whether the shift is non-zero and consistent.",
        "per_seed": per_seed,
        "summary": {
            "n_seeds": len(per_seed),
            "entropy_frac_of_uniform_mean": float(np.mean(ents)),
            "paired_dx_mean_over_seeds": (float(np.mean(dxs)) if dxs else None),
            "paired_dx_range": ([float(min(dxs)), float(max(dxs))] if dxs else None),
            "n_seeds_with_perm_p_below_0.05": int(sum(1 for p in ps if p < 0.05)),
            "n_seeds_tested": len(ps)},
    }
    e = rep["summary"]["entropy_frac_of_uniform_mean"]
    rep["verdict"] = {
        "attention_is_essentially_uniform": bool(e > 0.98),
        "reading": ("the learned attention is within {:.1f}% of a uniform distribution over the 36 "
                    "patches, i.e. it does not localise at all -- the 'cross-attention' is acting as "
                    "extra capacity, not as spatial grounding, so the arm cannot be credited with "
                    "'looking where the evidence is'.").format(100 * (1 - e)) if e > 0.98 else
                   ("the attention does concentrate (entropy {:.1f}% of uniform). Whether it "
                    "concentrates MEANINGFULLY is decided by the paired left/right shift, not by "
                    "concentration alone.").format(100 * e)}
    json.dump(rep, open(A.out, "w"), indent=1)
    print(json.dumps(rep["summary"], indent=1))
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
