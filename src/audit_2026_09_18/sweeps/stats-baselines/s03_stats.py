#!/usr/bin/env python3
"""Hostile-reviewer statistics on the pooled-probe headline, computed from the per-question cache.
All arms share IDENTICAL questions. Currencies: Lingshu-32B judge, lenient EM, strict EM, token-F1."""
import json, os, re, string, math
import numpy as np
from collections import Counter

CACHE = '/data/dan/audit_2026-09-18/tmp/em-rescore/em_rescore_per_question_cache.json'
CK = '/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa'
OUT = '/data/dan/audit_2026-09-18/tmp/stats-baselines/s03_stats.json'
rng = np.random.default_rng(20260920)
B = 10000
CUR = ('y', 'em', 'st', 'f1')
CURNAME = {'y': 'judge', 'em': 'lenientEM', 'st': 'strictEM', 'f1': 'tokenF1'}


def em_norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def em_score(pred, gold):
    p, g = em_norm(pred), em_norm(gold)
    if not p:
        return 0
    if p == g:
        return 1
    if g and (g in p.split() or p in g.split() or g in p or p in g):
        return 1
    return 0


def strict_em(pred, gold):
    p, g = em_norm(pred), em_norm(gold)
    return int(bool(p) and p == g)


def token_f1(pred, gold):
    p, g = set(em_norm(pred).split()), set(em_norm(gold).split())
    if not p or not g:
        return 0.0
    ov = len(p.intersection(g))
    if ov == 0:
        return 0.0
    pr, rc = ov / len(p), ov / len(g)
    return 2 * pr * rc / (pr + rc)


def boot_ci(imgs, vals, nb=B):
    keys = {}
    for im, v in zip(imgs, vals):
        keys.setdefault(im, []).append(v)
    s = np.array([sum(v) for v in keys.values()], float)
    n = np.array([len(v) for v in keys.values()], float)
    idx = rng.integers(0, len(s), size=(nb, len(s)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def sign_test(k, n):
    from math import comb
    tail = sum(comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2 * tail / (2 ** n))


def t_interval(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1) / math.sqrt(n)
    tcrit = {7: 2.364624, 6: 2.446912, 5: 2.570582, 3: 3.182446}[n - 1]
    return float(m), [float(m - tcrit * s), float(m + tcrit * s)]


D = json.load(open(CACHE))
R = D['records']
BENCH = list(R.keys())
res = {"title": "Hostile-reviewer statistics on the pooled probe headline",
       "date": "2026-09-20", "no_fabricated_numbers": True, "source_cache": CACHE,
       "bootstrap": str(B) + " resamples clustered by img_md5 within benchmark",
       "currencies": CURNAME, "per_benchmark": {}}


def arm_vals(recs, chooser, cur):
    return np.array([recs[i]['cands'][chooser[i]][cur] for i in range(len(recs))], float)


allrows = {}
for cell, recs in R.items():
    imgs = [r['img'] for r in recs]
    n = len(recs)
    ch_probe = [r['pick'] for r in recs]
    ch_vote = [max(range(len(r['cands'])), key=lambda i, rr=r: (rr['cands'][i]['votes'], -i)) for r in recs]
    ch_long = [max(range(len(r['cands'])), key=lambda i, rr=r: (rr['cands'][i]['w'], rr['cands'][i]['votes'], -i)) for r in recs]
    ch_first = [0 for r in recs]
    # answer-prior control: P(y=1 . normalised answer string), counted LEAVE-ONE-QUESTION-OUT on the
    # held-out half itself. OPTIMISTIC for the prior (it sees held-out labels) -> a hostile upper
    # bound on what a pure answer-string counter can achieve.
    pos, tot = Counter(), Counter()
    for r in recs:
        for c in r['cands']:
            k = em_norm(c['ans'])
            tot[k] += 1
            pos[k] += c['y']
    ch_prior = []
    for r in recs:
        best, bs = 0, -1e9
        for i, c in enumerate(r['cands']):
            k = em_norm(c['ans'])
            t = tot[k] - 1
            p = pos[k] - c['y']
            s = (p + 0.5) / (t + 1.0)
            if s > bs:
                bs, best = s, i
        ch_prior.append(best)

    cellout = {"n": n, "n_images": len(set(imgs))}
    for cur in CUR:
        g = np.array([r['greedy'][cur] for r in recs], float)
        arms = {
            "greedy": g,
            "probe": arm_vals(recs, ch_probe, cur),
            "majority_vote": arm_vals(recs, ch_vote, cur),
            "random_pick": np.array([sum(c['votes'] * c[cur] for c in r['cands']) / 8.0 for r in recs], float),
            "first_sample": arm_vals(recs, ch_first, cur),
            "longest": arm_vals(recs, ch_long, cur),
            "answer_prior_LOO_optimistic": arm_vals(recs, ch_prior, cur),
            "oracle8": np.array([max(c[cur] for c in r['cands']) for r in recs], float),
            "worst8": np.array([min(c[cur] for c in r['cands']) for r in recs], float),
        }
        c = {k: float(v.mean()) for k, v in arms.items()}
        for k in ("probe", "majority_vote", "random_pick", "answer_prior_LOO_optimistic", "longest", "oracle8"):
            d, ciq = boot_ci(imgs, arms[k] - g)
            c[k + "_minus_greedy"] = d
            c[k + "_minus_greedy_ci"] = ciq
        d, ciq = boot_ci(imgs, arms["probe"] - arms["majority_vote"])
        c["probe_minus_vote"], c["probe_minus_vote_ci"] = d, ciq
        d, ciq = boot_ci(imgs, arms["probe"] - arms["answer_prior_LOO_optimistic"])
        c["probe_minus_prior"], c["probe_minus_prior_ci"] = d, ciq
        cellout[CURNAME[cur]] = c
        allrows.setdefault(cur, {})[cell] = {k: float(v.mean()) for k, v in arms.items()}
    res["per_benchmark"][cell] = cellout
    print("[A] %-17s n %5d judge g %.4f probe %.4f vote %.4f rand %.4f priorLOO %.4f" % (
        cell, n, cellout['judge']['greedy'], cellout['judge']['probe'],
        cellout['judge']['majority_vote'], cellout['judge']['random_pick'],
        cellout['judge']['answer_prior_LOO_optimistic']), flush=True)

sizes = {c: len(R[c]) for c in BENCH}
NQ = sum(sizes.values())
summary = {}
for cur in CUR:
    cn = CURNAME[cur]
    S = {}
    for arm in ("probe", "majority_vote", "random_pick", "answer_prior_LOO_optimistic", "longest", "oracle8"):
        per = np.array([allrows[cur][c][arm] - allrows[cur][c]["greedy"] for c in BENCH])
        macro = float(per.mean())
        micro = float(sum((allrows[cur][c][arm] - allrows[cur][c]["greedy"]) * sizes[c] for c in BENCH) / NQ)
        lobo = {BENCH[i]: float(np.delete(per, i).mean()) for i in range(len(BENCH))}
        npos = int((per > 0).sum())
        nneg = int((per < 0).sum())
        m, ti = t_interval(per)
        S[arm] = {"macro": macro, "micro_question_weighted": micro, "macro_minus_micro": macro - micro,
                  "per_benchmark": {c: float(per[i]) for i, c in enumerate(BENCH)},
                  "LOBO_macro": lobo, "LOBO_range": [min(lobo.values()), max(lobo.values())],
                  "n_positive": npos, "n_negative": nneg,
                  "sign_test_two_sided_p": sign_test(npos, len(BENCH)),
                  "benchmark_t_interval_df7": ti}
    summary[cn] = S
res["summary_by_currency"] = summary

print("")
print("=== MACRO vs MICRO vs LOBO, arm minus greedy ===")
for cn, S in summary.items():
    for arm in ("probe", "majority_vote", "random_pick", "answer_prior_LOO_optimistic"):
        s = S[arm]
        print(" %-10s %-28s macro %+.4f micro %+.4f (d %+.4f) LOBO [%+.4f,%+.4f] %d/8 signp %.3f t7 [%+.4f,%+.4f]" % (
            cn, arm, s['macro'], s['micro_question_weighted'], s['macro_minus_micro'],
            s['LOBO_range'][0], s['LOBO_range'][1], s['n_positive'], s['sign_test_two_sided_p'],
            s['benchmark_t_interval_df7'][0], s['benchmark_t_interval_df7'][1]))

S32 = {"note": "Lingshu-32B GREEDY dumps on disk joined by idx to the SAME held-out questions. "
               "judge_ok is the Lingshu-32B judge scoring the 32B's own output (self-judging); "
               "EM/strictEM/F1 recomputed here with the identical normaliser.", "cells": {}}
FILES = {
    'slake_open': CK + '/strong_lingshu/ckpt_slake_open_lingshu32b.jsonl',
    'vqa_rad_open': CK + '/strong_lingshu/ckpt_vqa_rad_open_lingshu32b.jsonl',
    'pathvqa_open': CK + '/strong_lingshu/ckpt_pathvqa_open_lingshu32b.jsonl',
    'radimagenet_open': CK + '/strong_lingshu/ckpt_radimagenet_open_lingshu32b_t0.jsonl',
}
for cell, fp in FILES.items():
    jf = fp.replace('.jsonl', '.judge.jsonl')
    if not (os.path.exists(fp) and os.path.exists(jf)):
        S32["cells"][cell] = {"status": "MISSING", "path": fp}
        print("[32B] %-17s MISSING %s" % (cell, fp), flush=True)
        continue
    gen, jud = {}, {}
    for l in open(fp):
        l = l.strip()
        if l:
            r = json.loads(l)
            gen[r['idx']] = r
    for l in open(jf):
        l = l.strip()
        if l:
            r = json.loads(l)
            jud[r['idx']] = r.get('judge_ok')
    recs = R[cell]
    matched = [r for r in recs if r['idx'] in gen and r['idx'] in jud]
    if len(matched) < 30:
        S32["cells"][cell] = {"status": "TOO_FEW_MATCHED", "n_heldout": len(recs),
                              "n_matched": len(matched), "n_in_32b_dump": len(gen)}
        print("[32B] %-17s TOO FEW MATCHED %d of %d (dump has %d)" % (cell, len(matched), len(recs), len(gen)), flush=True)
        continue
    imgs = [r['img'] for r in matched]
    a32 = {c: [] for c in CUR}
    a7g = {c: [] for c in CUR}
    a7p = {c: [] for c in CUR}
    for r in matched:
        p32 = gen[r['idx']]['modal_pred']
        gold = r['gold']
        a32['y'].append(float(jud[r['idx']]))
        a32['em'].append(float(em_score(p32, gold)))
        a32['st'].append(float(strict_em(p32, gold)))
        a32['f1'].append(float(token_f1(p32, gold)))
        for c in CUR:
            a7g[c].append(float(r['greedy'][c]))
            a7p[c].append(float(r['cands'][r['pick']][c]))
    cc = {"n_heldout": len(recs), "n_matched": len(matched), "n_in_32b_dump": len(gen),
          "n_images": len(set(imgs))}
    for c in CUR:
        v32, v7g, v7p = np.array(a32[c]), np.array(a7g[c]), np.array(a7p[c])
        cc[CURNAME[c]] = {"lingshu32b_greedy": float(v32.mean()),
                          "lingshu7b_greedy": float(v7g.mean()),
                          "lingshu7b_probe_bo8": float(v7p.mean())}
        d, ciq = boot_ci(imgs, v7p - v32)
        cc[CURNAME[c]]["probe_minus_32b"], cc[CURNAME[c]]["probe_minus_32b_ci"] = d, ciq
        d, ciq = boot_ci(imgs, v32 - v7g)
        cc[CURNAME[c]]["32b_minus_7b_greedy"], cc[CURNAME[c]]["32b_minus_7b_greedy_ci"] = d, ciq
    S32["cells"][cell] = cc
    print("[32B] %-17s matched %5d of %5d :: J 32B %.4f 7Bg %.4f 7B+probe %.4f probe-32B %+.4f [%+.4f,%+.4f] :: EM 32B %.4f 7B+probe %.4f probe-32B %+.4f [%+.4f,%+.4f]" % (
        cell, len(matched), len(recs), cc['judge']['lingshu32b_greedy'], cc['judge']['lingshu7b_greedy'],
        cc['judge']['lingshu7b_probe_bo8'], cc['judge']['probe_minus_32b'],
        cc['judge']['probe_minus_32b_ci'][0], cc['judge']['probe_minus_32b_ci'][1],
        cc['lenientEM']['lingshu32b_greedy'], cc['lenientEM']['lingshu7b_probe_bo8'],
        cc['lenientEM']['probe_minus_32b'], cc['lenientEM']['probe_minus_32b_ci'][0],
        cc['lenientEM']['probe_minus_32b_ci'][1]), flush=True)
res["bigger_model_reference_32B"] = S32
json.dump(res, open(OUT, 'w'), indent=1)
print("")
print("wrote " + OUT)
