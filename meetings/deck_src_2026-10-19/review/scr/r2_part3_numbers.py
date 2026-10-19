#!/usr/bin/env python3
"""part3_numbers.py — statistics for deck Part V–VII that are not in a repo artifact (read-only).

Inputs: /data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json (per-question records behind
final_eval_FINAL_cap320_T07.json), WED picks /data/dan/exp_2026_10/wed/wed_<cell>_c320.jsonl, and the cheap_lingshu7b
sample/judge checkpoints. Bootstrap: 10,000 resamples of images within benchmark (paired), seed 12345.
Output: numbers_review_part3.json as {key: {value, computed_by, inputs}}.
"""
import json, sys
from collections import defaultdict
import numpy as np

MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
REC = "/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/review/scr/r2_part3_out.json"
BY = "deck/part3_numbers.py"
R = json.load(open(REC))
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
NB = 10000
rng = np.random.default_rng(12345)
T975_DF7 = 2.364624  # Student t 0.975 quantile, 7 df
out = {}


def put(k, v, inputs=REC):
    out[k] = {"value": v, "computed_by": BY, "inputs": inputs}


def boot(pairs_by_img):
    """pairs: list of (img, diff); returns point, draws of the mean diff (image-clustered)."""
    imgs, inv = np.unique([p[0] for p in pairs_by_img], return_inverse=True)
    d = np.array([p[1] for p in pairs_by_img], float)
    W = rng.multinomial(len(imgs), np.full(len(imgs), 1 / len(imgs)), size=NB).astype(float)
    s = W @ np.bincount(inv, weights=d, minlength=len(imgs))
    n = W @ np.bincount(inv, minlength=len(imgs)).astype(float)
    return float(d.mean()), s / n, s, n, len(d)


def r4(x):
    return round(float(x), 4)


def pct(a, lo=2.5, hi=97.5):
    return [r4(v) for v in np.percentile(a, [lo, hi])]


for cu in ["judge", "xjudge", "em"]:
    pts, draws, S, N, ns = {}, {}, {}, {}, {}
    for c in OPEN8:
        prs = [(r["img"], r["pick"][cu] - r["greedy"][cu]) for r in R[c]
               if r["pick"].get(cu) is not None and r["greedy"].get(cu) is not None]
        pts[c], draws[c], S[c], N[c], ns[c] = boot(prs)
    mac = np.mean([draws[c] for c in OPEN8], 0)
    put(f"{cu}.macro", {"pt": r4(np.mean(list(pts.values()))), "ci": pct(mac)})
    mic = sum(S[c] for c in OPEN8) / sum(N[c] for c in OPEN8)
    put(f"{cu}.micro_question_weighted", {"pt": r4(sum(pts[c] * ns[c] for c in OPEN8) / sum(ns.values())), "ci": pct(mic)})
    # benchmarks as the sampling unit: t interval over the 8 per-benchmark deltas
    v = np.array([pts[c] for c in OPEN8])
    hw = T975_DF7 * v.std(ddof=1) / np.sqrt(8)
    put(f"{cu}.benchmark_as_unit_t_interval", {"pt": r4(v.mean()), "ci": [r4(v.mean() - hw), r4(v.mean() + hw)], "df": 7})
    # leave one benchmark out (macro of the other 7, image-clustered CI)
    put(f"{cu}.leave_one_benchmark_out", {c: {"pt": r4(np.mean([pts[k] for k in OPEN8 if k != c])),
                                               "ci": pct(np.mean([draws[k] for k in OPEN8 if k != c], 0))} for c in OPEN8})
    # Bonferroni per-benchmark intervals (8 benchmarks; 40 = 8 benchmarks x 5 currencies)
    put(f"{cu}.per_benchmark_bonferroni", {c: {"pt": r4(pts[c]), "ci95": pct(draws[c]),
                                                "ci_bonf8": pct(draws[c], 2.5 / 8, 100 - 2.5 / 8),
                                                "ci_bonf40": pct(draws[c], 2.5 / 40, 100 - 2.5 / 40)} for c in OPEN8})
    for f in (0, 1):
        put(f"{cu}.fold{f}_macro_pt", r4(np.mean([np.mean([r["pick"][cu] - r["greedy"][cu] for r in R[c]
                                                            if r["fold"] == f and r["pick"].get(cu) is not None and r["greedy"].get(cu) is not None])
                                                   for c in OPEN8])))

# headroom = oracle@8 - greedy; captured share = (pick - greedy) / headroom (judge)
hr, cap, g_, h_, o_ = {}, {}, [], [], []
for c in OPEN8:
    gr = np.mean([r["greedy"]["judge"] for r in R[c]]); pk = np.mean([r["pick"]["judge"] for r in R[c]])
    oc = np.mean([r["oracle_judge"] for r in R[c]])
    hr[c], cap[c] = r4(oc - gr), round(float((pk - gr) / (oc - gr)), 3)
    g_.append(pk - gr); h_.append(oc - gr); o_.append(oc)
put("judge.headroom_oracle_minus_greedy", hr)
put("judge.headroom_captured", {**cap, "macro": round(float(np.mean(list(cap.values()))), 3)})
put("judge.corr_gain_vs_headroom_8_benchmarks", round(float(np.corrcoef(g_, h_)[0, 1]), 3))
put("judge.corr_gain_vs_oracle_8_benchmarks", round(float(np.corrcoef(g_, o_)[0, 1]), 3))
put("pick_differs_from_greedy_string", {c: round(float(np.mean([r["pick_ans"].strip().lower() != r["greedy_ans"].strip().lower()
                                                                  for r in R[c]])), 3) for c in OPEN8})

# WED: questions where WED had >= 2 candidates (its game runs) vs <= 1; probe - WED judge on the game-active subset
sys.path.insert(0, MAIN + "/.claude/worktrees/exp-2026-10-15/src/audit_2026_09_18")
import em_rescore_pooled_probe as ER  # noqa: E402
CK = MAIN + "/ckpts/openvqa/cheap_lingshu7b"
wed_out = {}
for c in ["pathvqa_open", "slake_open", "vqa_rad_open"]:
    wed = {d["idx"]: d for d in map(json.loads, open(f"/data/dan/exp_2026_10/wed/wed_{c}_c320.jsonl"))}
    exp = ER.loadj(f"{CK}/ckpt_{c}_lingshu7b_sc8_scexploded.jsonl")
    jud = ER.loadj(f"{CK}/ckpt_{c}_lingshu7b_sc8_scexploded.judge.jsonl")
    jmap = defaultdict(dict)
    for cid, r in exp.items():
        if cid in jud:
            q = cid.split("#")[0]; q = int(q) if q.lstrip("-").isdigit() else q
            jmap[q][ER.row_norm(r["modal_pred"])] = int(jud[cid]["judge_ok"])
    act, n_all, n_le1, miss = [], 0, 0, 0
    for r in R[c]:
        w = wed.get(r["idx"])
        if w is None:
            continue
        n_all += 1
        if w.get("n_cand", 0) <= 1:
            n_le1 += 1
            continue
        wj = jmap[r["idx"]].get(ER.row_norm(w["pick_text"]))
        if wj is None:
            miss += 1
            continue
        act.append((r["img"], r["pick"]["judge"], wj))
    pt, dr, _, _, n = boot([(a, p - wj) for a, p, wj in act])
    wed_out[c] = {"questions": n_all, "wed_le1_candidate": n_le1, "game_active_scored": n, "unscored_wed_pick": miss,
                  "probe_judge_active": r4(np.mean([p for _, p, _ in act])), "wed_judge_active": r4(np.mean([w for _, _, w in act])),
                  "probe_minus_wed_judge_active": {"pt": r4(pt), "ci": pct(dr)}}
put("wed.game_active_subset", wed_out, REC + "; /data/dan/exp_2026_10/wed/wed_<cell>_c320.jsonl; ckpts/openvqa/cheap_lingshu7b/*_scexploded{,.judge}.jsonl")
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
