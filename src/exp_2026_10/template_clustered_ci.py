#!/usr/bin/env python3
"""template_clustered_ci.py -- do the headline intervals survive clustering by QUESTION TEMPLATE, not just image?

Owed since the 2026-10-19 deck ("evaluation owed: template-clustered CIs"). Every interval in the project is a
paired bootstrap clustered by image. But 55-100% of held-out questions on five benchmarks reuse a question
string that also occurs on other images (template_stratified_gain_2026-09-20.json), so questions sharing a
template are not independent either, and an image-only bootstrap can be too narrow.

Same per-question picks as the shipped evaluation (/data/dan/exp_2026_10/final_eval_FINAL_cap320_<pol>_records.json),
same quantity (macro over the 8 open benchmarks of mean(pick - greedy), both folds), four cluster schemes:
  image        what every artifact reports (recomputed here with this script's own RNG)
  template     one-way: resample question templates
  two-way      image x template, Owen's pigeonhole bootstrap: each question weighted by w_image * w_template,
               with independent multinomial weights for images and templates (Owen 2007, Ann. Appl. Stat.)
  components   connected components of the image-template graph (the most conservative scheme); reported
               with its cluster count, because it can collapse a benchmark into a handful of clusters
Template = the normalised question string (lower-case, whitespace collapsed, trailing ?.! stripped) -- the
definition template_stratified_gain.py used.

  python3 src/exp_2026_10/template_clustered_ci.py     (CPU, a few minutes)
"""
import json, sys
from collections import defaultdict
import numpy as np

CK = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b"
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
CUR = ("judge", "xjudge", "em")
NB, SEED = 10000, 20261022
OUT = ("/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/"
       "artifacts/template_clustered_ci_2026-10-22.json")
nq = lambda s: " ".join(str(s).lower().split()).rstrip("?.! ")


def questions(cell):
    q = {}
    for l in open(f"{CK}/ckpt_{cell}_lingshu7b.jsonl"):
        if l.strip():
            d = json.loads(l); q[d["idx"]] = nq(d["question"])
    return q


def components(imgs, tmpls):
    par = {}
    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in zip(imgs, tmpls):
        ra, rb = find(("i", a)), find(("t", b))
        if ra != rb: par[ra] = rb
    return ["%s|%s" % find(("i", a)) for a in imgs]     # strings: np.unique would flatten tuples


def weights(rng, labels):
    keys, inv = np.unique(labels, return_inverse=True)
    w = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=NB).astype(float)
    return w[:, inv], len(keys)                     # NB x n_questions


out = {"title": "Headline CIs clustered by image, template, image x template, and connected components",
       "date": "2026-10-22", "no_fabricated_numbers": True, "script": __file__, "bootstrap": {"n": NB, "seed": SEED},
       "template_definition": "normalised question string", "policies": {}}
rng = np.random.default_rng(SEED)
for pol in ("T07", "Tmap"):
    R = json.load(open(f"/data/dan/exp_2026_10/final_eval_FINAL_cap320_{pol}_records.json"))
    reps = {s: {cu: [] for cu in CUR} for s in ("image", "template", "two-way", "components")}
    pts, ncl = {cu: [] for cu in CUR}, defaultdict(dict)
    for c in OPEN8:
        qs = questions(c); recs = R[c]
        img = [r["img"] for r in recs]; tpl = [qs[r["idx"]] for r in recs]; comp = components(img, tpl)
        Wi, ni = weights(rng, img); Wt, nt = weights(rng, tpl); Wc, nc = weights(rng, comp)
        ncl[c] = {"questions": len(recs), "images": ni, "templates": nt, "components": nc,
                  "largest_component_share": float(np.max(np.unique(comp, return_counts=True)[1]) / len(recs))}
        W = {"image": Wi, "template": Wt, "two-way": Wi * Wt, "components": Wc}
        for cu in CUR:
            m = np.array([r["pick"].get(cu) is not None and r["greedy"].get(cu) is not None for r in recs])
            d = np.array([(r["pick"][cu] - r["greedy"][cu]) if mm else 0.0 for r, mm in zip(recs, m)])
            pts[cu].append(float(d[m].mean()))
            for s, w in W.items():
                reps[s][cu].append((w @ d) / np.maximum(w @ m.astype(float), 1e-12))
        print(f"  {pol:4} {c:17} q {len(recs):5}  images {ni:5}  templates {nt:5}  components {nc:5}", flush=True)
    res = {"clusters": ncl}
    for cu in CUR:
        res[cu] = {"macro": float(np.mean(pts[cu]))}
        for s in reps:
            Rm = np.stack(reps[s][cu], 1).mean(1)
            lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
            res[cu][s] = {"ci": [lo, hi], "half_width": (hi - lo) / 2,
                          "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")}
        print(f"  {pol:4} {cu:6} macro {res[cu]['macro']:+.4f}  " + "  ".join(
            f"{s} [{res[cu][s]['ci'][0]:+.4f},{res[cu][s]['ci'][1]:+.4f}] {res[cu][s]['verdict']}" for s in reps), flush=True)
    out["policies"][pol] = res
out["COMPLETE"] = True
json.dump(out, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
