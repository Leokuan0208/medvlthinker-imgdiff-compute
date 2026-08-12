#!/usr/bin/env python3
"""
ATTACK C / PART 2 -- CLUSTER-RESAMPLING RE-CERTIFICATION.

WHY.  LITERATURE_UPDATE_2026-08-11.md cites RouteGuard (arXiv:2608.07583), whose reported routing
gains VANISHED when the bootstrap resampled GROUPS (images / question templates / source studies)
instead of individual items.  Every CI this project has published is an item-level paired bootstrap.
Medical VQA is exactly the setting where that matters: SLAKE asks ~6.7 questions per image,
PathVQA ~4, VQA-RAD ~1.9.  If items inside an image are correlated, the item bootstrap understates
the variance and our CIs are too narrow.

WHAT.  Re-runs the headline macro deltas under three resampling schemes:
   item     -- resample items within each cell  (reproduces the published CIs; this is the NULL TEST)
   image    -- resample IMAGE CLUSTERS within each cell, taking every question on a drawn image
   cell     -- resample the 8 REPORTING CELLS themselves with replacement (the "would this
               generalise to a new benchmark" question; 8 units, so deliberately brutal)
Clusters come from results/cascade_methods/artifacts/_stats_recert/meta_*.json, built by
stats_recert_meta.py; image identity is md5 of DECODED RGB pixels for the parquet-backed cells and
the dataset's own image name for SLAKE / PMC-VQA.

NULL TEST.  scheme='item' must reproduce the published point estimates EXACTLY (deterministic --
they are means of frozen vectors) and the published CI bounds to Monte-Carlo error.

Launch from the repo root:  python3 src/cascade_methods/stats_recert_p2_cluster.py
Writes results/cascade_methods/artifacts/_stats_recert/part2_cluster.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_recert_common import (ART, ARMS, CELLS, CELLS_MCQ, CELLS_OPEN, META, NBOOT, PARTS,
                                 SEED, ci, cell_boot, cluster_ids, jdump, load_vec)

W = 1.0 / len(CELLS)

# The published values these claims must be checked against, verbatim from
# results/cascade_methods/artifacts/_selector_rerun_parts/summary_{disjoint,ens8_scaled}.json
# and CLAUDE.md section 0's RESOLVED box.
PUBLISHED = {
    ("disjoint", "method_accuracy_max_veto", "always_32b_direct"): (0.0008, -0.0022, 0.0037),
    ("disjoint", "method_accuracy_max_veto", "always_32b_reasoning"): (0.0601, 0.0499, 0.0700),
    ("disjoint", "method_compute_lean", "always_32b_direct"): (-0.0124, None, None),
    ("ens8_scaled", "method_accuracy_max_veto", "always_32b_direct"): (0.0023, -0.0010, 0.0054),
    ("ens8_scaled", "method_accuracy_max_veto", "always_32b_reasoning"): (0.0615, 0.0514, 0.0715),
    ("ens8_scaled", "method_compute_lean", "always_32b_direct"): (-0.0091, -0.0153, -0.0031),
}

CLAIMS = [("method_accuracy_max_veto", "always_32b_direct"),
          ("method_accuracy_max_veto", "always_32b_reasoning"),
          ("method_compute_lean", "always_32b_direct"),
          ("method_accuracy_max_fusion", "always_32b_direct"),
          ("always_32b_direct", "always_7b")]


def macro_point(vec, keys):
    return {a: float(np.mean([vec[c][a].mean() for c in keys])) for a in ARMS}


def run_source(source, groups, nboot):
    vec = load_vec(source)
    mats = {c: np.column_stack([vec[c][a] for a in ARMS]) for c in CELLS}
    ai = {a: i for i, a in enumerate(ARMS)}
    out = {"source": source, "n_bootstrap": nboot, "seed": SEED, "point": {}, "schemes": {}}

    for lab, keys in (("all8", CELLS), ("open_only", CELLS_OPEN), ("mcq_only", CELLS_MCQ)):
        out["point"][lab] = {a: round(v, 6) for a, v in macro_point(vec, keys).items()}

    # ---- item and image (=cluster) schemes: independent draw per cell, PAIRED across arms -----
    for scheme in ("item", "cluster"):
        rng = np.random.default_rng(SEED)
        per_cell = {c: cell_boot(mats[c], groups[c], nboot, rng, scheme) for c in CELLS}
        blk = {}
        for lab, keys in (("all8", CELLS), ("open_only", CELLS_OPEN), ("mcq_only", CELLS_MCQ)):
            w = 1.0 / len(keys)
            md = sum(per_cell[c] * w for c in keys)          # (nboot, n_arms) macro distribution
            pt = macro_point(vec, keys)
            rows = {}
            for m, b in CLAIMS:
                rows[f"{m} - {b}"] = ci(md[:, ai[m]] - md[:, ai[b]], pt[m] - pt[b])
            blk[lab] = rows
        out["schemes"]["image" if scheme == "cluster" else scheme] = blk

    # ---- cell scheme: the 8 reporting cells ARE the resampling units -------------------------
    rng = np.random.default_rng(SEED + 1)
    cell_means = np.array([[vec[c][a].mean() for a in ARMS] for c in CELLS])   # (8, n_arms)
    idx = rng.integers(0, len(CELLS), size=(nboot, len(CELLS)))
    md = cell_means[idx].mean(axis=1)                                          # (nboot, n_arms)
    pt = macro_point(vec, CELLS)
    out["schemes"]["cell"] = {"all8": {f"{m} - {b}": ci(md[:, ai[m]] - md[:, ai[b]], pt[m] - pt[b])
                                       for m, b in CLAIMS}}
    out["per_cell_delta"] = {f"{m} - {b}": {c: round(float(vec[c][m].mean() - vec[c][b].mean()), 4)
                                            for c in CELLS} for m, b in CLAIMS}
    return out


def main():
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else NBOOT
    groups = {c: cluster_ids(c) for c in CELLS}
    cl_info = {c: dict(n=int(len(groups[c])), n_clusters=int(groups[c].max() + 1),
                       items_per_cluster=round(len(groups[c]) / (groups[c].max() + 1), 3),
                       max_cluster=int(np.bincount(groups[c]).max())) for c in CELLS}
    for c in CELLS:
        print(f"  {c:16s} n={cl_info[c]['n']:6d} clusters={cl_info[c]['n_clusters']:6d} "
              f"items/cluster={cl_info[c]['items_per_cluster']}")

    res = {"what": "ATTACK C part 2 -- cluster-resampling re-certification of the headline macro "
                   "deltas (RouteGuard arXiv:2608.07583 correction)",
           "date": "2026-08-11",
           "cluster_definition": {
               "PMC_VQA": "test_2.csv Figure_path",
               "SLAKE_closed": "SLAKE img_name", "SLAKE_open": "SLAKE img_name (via qid)",
               "VQA_RAD_closed": "md5 of DECODED RGB pixels", "VQA_RAD_open": "md5 decoded RGB",
               "PATH_VQA_closed": "md5 of DECODED RGB pixels", "PATH_VQA_open": "md5 decoded RGB",
               "MedXpertQA-MM": "tuple of image file names (all singletons in practice)"},
           "cluster_stats": cl_info,
           "published_for_comparison": {f"{s}|{m}|{b}": dict(delta=d, lo=lo, hi=hi)
                                        for (s, m, b), (d, lo, hi) in PUBLISHED.items()},
           "results": {}}
    for source in ("disjoint", "ens8_scaled"):
        print(f"[part2] {source}", flush=True)
        res["results"][source] = run_source(source, groups, nboot)

    # ---------------- NULL TEST: item scheme must reproduce the published numbers -------------
    dev = {}
    for (s, m, b), (d, lo, hi) in PUBLISHED.items():
        got = res["results"][s]["schemes"]["item"]["all8"][f"{m} - {b}"]
        dev[f"{s}|{m}|{b}"] = dict(point_dev=round(abs(got["delta"] - d), 6),
                                   lo_dev=(None if lo is None else round(abs(got["lo"] - lo), 6)),
                                   hi_dev=(None if hi is None else round(abs(got["hi"] - hi), 6)),
                                   got=got)
    flat = [v for r in dev.values() for k, v in r.items() if k.endswith("_dev") and v is not None]
    res["null_test"] = dict(
        what="scheme='item' reimplemented from the frozen vectors must reproduce the published "
             "point estimates exactly and the published CI bounds to Monte-Carlo error.",
        per_claim=dev, max_abs_deviation_point=round(max(v["point_dev"] for v in dev.values()), 6),
        max_abs_deviation_ci=round(max(flat), 6),
        passed=bool(max(v["point_dev"] for v in dev.values()) <= 1e-4 and max(flat) <= 2e-3),
        tolerance="point <= 1e-4 (rounding of published 4-dp values); CI <= 2e-3 (Monte-Carlo, "
                  "nboot=%d, independent RNG stream from the published run)" % nboot)
    jdump(res, os.path.join(META, "part2_cluster.json"))
    n = res["null_test"]
    print(f"[null] point dev {n['max_abs_deviation_point']}  ci dev {n['max_abs_deviation_ci']}  "
          f"passed={n['passed']}")
    for s in ("disjoint", "ens8_scaled"):
        for m, b in CLAIMS[:3]:
            k = f"{m} - {b}"
            r = {sc: res["results"][s]["schemes"][sc]["all8"][k] for sc in ("item", "image", "cell")}
            print(f"{s:12s} {k:48s} item {r['item']['delta']:+.4f} [{r['item']['lo']:+.4f},{r['item']['hi']:+.4f}] {r['item']['verdict']:4s}"
                  f" | image [{r['image']['lo']:+.4f},{r['image']['hi']:+.4f}] {r['image']['verdict']:4s}"
                  f" | cell [{r['cell']['lo']:+.4f},{r['cell']['hi']:+.4f}] {r['cell']['verdict']}")


if __name__ == "__main__":
    main()
