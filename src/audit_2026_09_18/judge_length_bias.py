#!/usr/bin/env python3
"""AUDIT 2026-09-18/20 -- is the judge-currency gain of the probe verifier a LENGTH / leniency artefact?

Input: the per-question cache written by em_rescore_pooled_probe.py (frozen 24 probes, held-out halves,
identical picks). Every candidate carries y (Lingshu-32B judge), em (project's normalised lenient match),
st (strict EM), f1 (token F1), w (words), votes; each question carries pick (verifier) and greedy.

Diagnostics, per benchmark and macro, image-clustered bootstrap (10,000):
  D1  length-only selectors: pick the LONGEST / SHORTEST distinct candidate (ties -> more votes -> first).
      If 'longest' alone beats greedy under the judge, the judge rewards length.
  D2  judge-positive rate of candidates by word-count bucket, overall and among em==0 candidates
      (leniency as a function of length, holding lenient-EM at 0).
  D3  length-matched verifier gain: restrict to questions where words(pick) <= words(greedy).
  D4  where does the verifier's judge gain come from? decompose delta_judge into questions where the
      pick is em-correct too ("both currencies agree it is right") vs judge-only.
  D5  logistic regression of judge label on [em, f1, log words] over all held-out candidates:
      coefficient on log-words = judge's length preference at fixed string overlap.
"""
import json, sys
import numpy as np

P = '/data/dan/audit_2026-09-18/tmp/em-rescore/em_rescore_per_question_cache.json'
OUT = sys.argv[1]
R = json.load(open(P))['records']
rng = np.random.default_rng(20260920)
B = 10000


def boot(imgs, vals):
    """image-clustered bootstrap CI of the mean of vals (one value per question)."""
    keys = {}
    for im, v in zip(imgs, vals):
        keys.setdefault(im, []).append(v)
    s = np.array([sum(v) for v in keys.values()], float)
    n = np.array([len(v) for v in keys.values()], float)
    idx = rng.integers(0, len(s), size=(B, len(s)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def sel(cands, key):
    best = 0
    for i, c in enumerate(cands):
        if key(c) > key(cands[best]):
            best = i
    return best


out = {"title": "Is the probe verifier's judge-currency gain a length / leniency artefact?",
       "date": "2026-09-20", "no_fabricated_numbers": True, "source_cache": P,
       "currency_notes": "y = Lingshu-32B judge; em = project's normalised lenient match; st = strict EM; "
                         "f1 = token F1. Held-out halves, frozen 24-probe ensemble, identical picks.",
       "bootstrap": f"{B} resamples, clustered by img_md5 within benchmark", "cells": {}}
allX, ally = [], []
for cell, recs in R.items():
    imgs = [r['img'] for r in recs]
    c = {"n": len(recs)}
    g = {k: np.array([r['greedy'][k] for r in recs], float) for k in ('y', 'em', 'st', 'f1', 'w')}
    pk = [r['cands'][r['pick']] for r in recs]
    p = {k: np.array([x[k] for x in pk], float) for k in ('y', 'em', 'st', 'f1', 'w')}
    # D1
    for nm, key in (('longest', lambda x: (x['w'], x['votes'])), ('shortest', lambda x: (-x['w'], x['votes'])),
                    ('most_votes', lambda x: (x['votes'],))):
        ch = [r['cands'][sel(r['cands'], key)] for r in recs]
        for cur in ('y', 'em', 'f1'):
            d = np.array([x[cur] for x in ch], float) - g[cur]
            c[f"D1_{nm}_minus_greedy_{cur}"], c[f"D1_{nm}_minus_greedy_{cur}_ci"] = boot(imgs, d)
    for cur in ('y', 'em', 'st', 'f1'):
        c[f"verifier_minus_greedy_{cur}"], c[f"verifier_minus_greedy_{cur}_ci"] = boot(imgs, p[cur] - g[cur])
    c["words_greedy"], c["words_pick"] = float(g['w'].mean()), float(p['w'].mean())
    c["frac_pick_longer"], c["frac_pick_shorter"] = float((p['w'] > g['w']).mean()), float((p['w'] < g['w']).mean())
    # D2
    cand = [x for r in recs for x in r['cands']]
    W = np.array([x['w'] for x in cand]); Y = np.array([x['y'] for x in cand]); E = np.array([x['em'] for x in cand])
    F = np.array([x['f1'] for x in cand])
    qs = np.unique(np.quantile(W, [0, .25, .5, .75, 1.0]))
    buckets = {}
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (W >= lo) & ((W < hi) if hi < qs[-1] else (W <= hi))
        m0 = m & (E == 0)
        buckets[f"{lo:g}-{hi:g}w"] = {"n": int(m.sum()), "judge_pos": float(Y[m].mean()) if m.any() else None,
                                      "em_pos": float(E[m].mean()) if m.any() else None,
                                      "judge_pos_given_em0": float(Y[m0].mean()) if m0.any() else None,
                                      "n_em0": int(m0.sum())}
    c["D2_judge_rate_by_length"] = buckets
    # D3
    m = p['w'] <= g['w']
    c["D3_n_pick_not_longer"] = int(m.sum())
    if m.sum() > 30:
        ii = [im for im, mm in zip(imgs, m) if mm]
        c["D3_delta_judge_pick_not_longer"], c["D3_ci"] = boot(ii, (p['y'] - g['y'])[m])
        c["D3_delta_em_pick_not_longer"], c["D3_em_ci"] = boot(ii, (p['em'] - g['em'])[m])
    m2 = p['w'] > g['w']
    if m2.sum() > 30:
        ii = [im for im, mm in zip(imgs, m2) if mm]
        c["D3b_delta_judge_pick_longer"], c["D3b_ci"] = boot(ii, (p['y'] - g['y'])[m2])
        c["D3b_delta_em_pick_longer"], c["D3b_em_ci"] = boot(ii, (p['em'] - g['em'])[m2])
    # D4: flips
    up = (p['y'] == 1) & (g['y'] == 0); dn = (p['y'] == 0) & (g['y'] == 1)
    c["D4_flips_up"], c["D4_flips_down"] = int(up.sum()), int(dn.sum())
    c["D4_flips_up_em_confirms"] = int((up & (p['em'] == 1)).sum())
    c["D4_flips_up_f1_ge_half"] = int((up & (p['f1'] >= 0.5)).sum())
    c["D4_flips_up_f1_zero"] = int((up & (p['f1'] == 0)).sum())
    c["D4_flips_down_em_confirms"] = int((dn & (g['em'] == 1)).sum())
    c["D4_flips_down_f1_zero_greedy"] = int((dn & (g['f1'] == 0)).sum())
    out["cells"][cell] = c
    allX.append(np.stack([E, F, np.log1p(W)], 1)); ally.append(Y)
    print(f"{cell:17} ver-J {c['verifier_minus_greedy_y']:+.4f} ver-EM {c['verifier_minus_greedy_em']:+.4f} | "
          f"LONGEST J {c['D1_longest_minus_greedy_y']:+.4f} EM {c['D1_longest_minus_greedy_em']:+.4f} | "
          f"SHORTEST J {c['D1_shortest_minus_greedy_y']:+.4f} | votes J {c['D1_most_votes_minus_greedy_y']:+.4f} | "
          f"pick longer {c['frac_pick_longer']:.2f} | not-longer dJ {c.get('D3_delta_judge_pick_not_longer', float('nan')):+.4f} "
          f"(n {c['D3_n_pick_not_longer']}) | up {c['D4_flips_up']} (em-confirmed {c['D4_flips_up_em_confirms']}, "
          f"f1=0 {c['D4_flips_up_f1_zero']}) down {c['D4_flips_down']}", flush=True)

# D5 pooled logistic regression (numpy IRLS, no sklearn dependence)
X = np.concatenate(allX); y = np.concatenate(ally).astype(float)
X = np.concatenate([np.ones((len(X), 1)), X], 1)
w = np.zeros(X.shape[1])
for _ in range(50):
    z = X @ w; p_ = 1 / (1 + np.exp(-z)); Wt = p_ * (1 - p_) + 1e-9
    w_new = np.linalg.solve((X * Wt[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]), (X * Wt[:, None]).T @ (z + (y - p_) / Wt))
    if np.abs(w_new - w).max() < 1e-8:
        w = w_new; break
    w = w_new
cov = np.linalg.inv((X * Wt[:, None]).T @ X)
out["D5_logit_judge_on_em_f1_logwords"] = {"n_candidates": int(len(y)),
                                           "coef": dict(zip(["intercept", "em", "f1", "log1p_words"], map(float, w))),
                                           "se": dict(zip(["intercept", "em", "f1", "log1p_words"], map(float, np.sqrt(np.diag(cov))))),
                                           "note": "naive (non-clustered) SEs; read the sign and size, not the p-value"}
keys = [k for k in out["cells"]["pathvqa_open"] if isinstance(out["cells"]["pathvqa_open"][k], float)]
out["macro"] = {k: float(np.mean([v[k] for v in out["cells"].values() if k in v])) for k in keys}
print(json.dumps({k: round(v, 4) for k, v in out["macro"].items() if k.startswith(("D1", "verifier", "D3", "frac"))}, indent=1))
print(out["D5_logit_judge_on_em_f1_logwords"])
json.dump(out, open(OUT, "w"), indent=1)
