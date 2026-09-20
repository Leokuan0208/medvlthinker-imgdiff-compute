#!/usr/bin/env python3
"""s07_currency.py -- READ-ONLY.  The core of the replication audit.

For each generator (qwen / medgemma) with a locally refitted SINGLE-LAYER pooled probe
(refit/{gen}_L{L}.npz, produced by s03_refit_worker.py, which mirrors head_final_stack.py's
pooled_singlelayer arm line by line), compute on the SAME held-out halves:

  greedy / random-pick / majority-vote / probe-pick / oracle@8   in four currencies
  (32B-judge, project normalised lenient EM, strict EM, token F1)

with 10,000-resample IMAGE-CLUSTERED bootstrap CIs per benchmark, an image-clustered macro CI and
a benchmark-level (n=8) macro CI.  Plus an ANSWER-PRIOR control fitted on the by-image train half.

Scorers imported verbatim from the audit's em-rescore script.  Readout (rank_avg, mean over seeds,
argmax with first-index tie-break) imported from the project's genframe_data.
"""
import os, sys, json, time, hashlib
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, "/data/dan/audit_2026-09-18/tmp/em-rescore")
from em_rescore_pooled_probe import em_norm, em_score, strict_em, token_f1, row_norm, half, nwords
MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(MAIN, "src/training_methods"))
from genframe_data import rank_avg

CK = os.path.join(MAIN, "ckpts/openvqa/cheap_lingshu7b")
FEATS = os.path.join(MAIN, "feats_hidden")
ART = os.path.join(MAIN, "results/cascade_methods/artifacts")
OUT = "/data/dan/audit_2026-09-18/tmp/replication"
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
NBOOT, BOOT_SEED = 10000, 20260920
GENS = {
    "qwen": {"tag": "qwen25vl7b", "L": 20, "stem": lambda c: f"generator_eval_qwen_{c}",
             "art": "head_final_stack_qwen_2026-09-13.json"},
    "medgemma": {"tag": "medgemma4b", "L": 24, "stem": lambda c: f"generator_eval_medgemma_{c}",
                 "art": "head_final_stack_medgemma_ALL8_2026-09-16.json"},
}


def loadj(p):
    out = {}
    with open(p) as f:
        for l in f:
            if l.strip():
                d = json.loads(l)
                out[d["idx"]] = d
    return out


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def verdict(lo, hi):
    return "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")


def cluster_boot(M, clusters, nboot, rng, chunk=500):
    keys, inv = np.unique(np.asarray(clusters), return_inverse=True)
    ncl = len(keys)
    S = np.zeros((ncl, M.shape[1]))
    np.add.at(S, inv, M)
    cnt = np.bincount(inv, minlength=ncl).astype(float)
    out = np.empty((nboot, M.shape[1]))
    p = np.full(ncl, 1.0 / ncl)
    for a in range(0, nboot, chunk):
        W = rng.multinomial(ncl, p, size=min(chunk, nboot - a)).astype(float)
        out[a:a + W.shape[0]] = (W @ S) / (W @ cnt)[:, None]
    return out, ncl


COLS = ["g_j", "g_em", "g_st", "g_f1",
        "v_j", "v_em", "v_st", "v_f1",
        "r_j", "r_em", "r_st", "r_f1",
        "m_j", "m_em", "m_st", "m_f1",
        "o_j", "o_em", "o_st", "o_f1",
        "p_j", "p_em", "p_st", "p_f1"]        # p_* = answer-prior control
CI_ = {c: i for i, c in enumerate(COLS)}


def main():
    t0 = time.time()
    RES = {"_meta": {"date": "2026-09-20", "script": os.path.abspath(__file__),
                     "nboot": NBOOT, "boot_seed": BOOT_SEED, "no_fabricated_numbers": True,
                     "readout": "mean over seeds of rank_avg(score) within the candidate set, "
                                "argmax first-index tie-break (head_final_stack.py:322-326)"}}
    for gen, G in GENS.items():
        rp = f"{OUT}/refit/{gen}_L{G['L']}.npz"
        rm = f"{OUT}/refit/{gen}_L{G['L']}.meta.json"
        if not (os.path.exists(rp) and os.path.exists(rm)):
            print(f"!! refit missing for {gen}: {rp}"); RES[gen] = {"ERROR": "refit missing"}; continue
        Z = np.load(rp)
        MT = json.load(open(rm))
        nseeds = Z[BENCH[0]].shape[0]
        print(f"=== {gen}  L{G['L']}  seeds={nseeds}  refit_minutes={MT['minutes']:.1f} "
              f"pooled_rows={MT['pooled_rows']}", flush=True)
        art = json.load(open(os.path.join(ART, G["art"])))
        # ---------------- answer prior: counts on the by-image TRAIN half, pooled over benchmarks
        prior = defaultdict(lambda: [0, 0])
        ntr = 0
        for cell in BENCH:
            meta = json.load(open(f"{FEATS}/{G['stem'](cell)}.meta.json"))
            for r in meta["rows"]:
                if r.get("n_tok", -1) > 0 and r.get("ds") == cell and half(r["img_md5"]) == 1:
                    k = em_norm(r["ans"])
                    prior[k][0] += int(r["y"]); prior[k][1] += 1
                    ntr += 1
            del meta
            print(f"  prior cache {cell}: cum train rows {ntr}  [{time.time()-t0:.0f}s]", flush=True)
        gp_c = sum(v[0] for v in prior.values()); gp_n = sum(v[1] for v in prior.values())
        gprior = gp_c / max(gp_n, 1)
        print(f"  answer prior: {len(prior):,} distinct normalised strings over {ntr:,} train rows, "
              f"global P(correct)={gprior:.4f}", flush=True)

        per, reps, point, macro_rows = {}, {}, {}, {}
        ss = np.random.SeedSequence(BOOT_SEED)
        child = ss.spawn(len(BENCH) + 1)
        for bi, cell in enumerate(BENCH):
            rows = MT["heldout_rows"][cell]
            S = Z[cell]                                    # (seeds, n_rows)
            greedy = loadj(f"{CK}/ckpt_{cell}_{G['tag']}.jsonl")
            gjudge = loadj(f"{CK}/ckpt_{cell}_{G['tag']}.judge.jsonl")
            sc8 = loadj(f"{CK}/ckpt_{cell}_{G['tag']}_sc8.jsonl")
            cj = {}
            with open(f"{CK}/ckpt_{cell}_{G['tag']}_sc8_scexploded.judge.jsonl") as f:
                for l in f:
                    if l.strip():
                        d = json.loads(l)
                        q, s = str(d["idx"]).split("#")
                        cj[(int(q), int(s))] = int(d["judge_ok"])
            byq = defaultdict(list)
            for i, r in enumerate(rows):
                byq[r["idx"]].append(i)
            qs = sorted(q for q in byq if q in gjudge)
            recs, drop = [], Counter()
            for q in qs:
                ii = np.array(byq[q])
                s8 = sc8.get(q); g = greedy.get(q)
                if s8 is None or g is None:
                    drop["no_sc8_or_greedy"] += 1; continue
                gold = s8["gold"]
                preds = list(s8["preds"])
                pn = [row_norm(a) for a in preds]
                first = {}
                for i2, a in enumerate(pn):
                    first.setdefault(a, i2)
                if any((q, i2) not in cj for i2 in first.values()):
                    drop["slot_label_absent"] += 1; continue
                lab = {a: cj[(q, i2)] for a, i2 in first.items()}
                ys8 = [lab[a] for a in pn]
                ems8 = [em_score(a, gold) for a in preds]
                sts8 = [strict_em(a, gold) for a in preds]
                f1s8 = [token_f1(a, gold) for a in preds]
                gv = Counter(em_norm(a) for a in preds)
                order = []
                for a in preds:
                    k = em_norm(a)
                    if k not in order:
                        order.append(k)
                key = max(order, key=lambda k: gv[k])
                ms = next(i2 for i2, a in enumerate(preds) if em_norm(a) == key)
                # ---- probe pick (row space)
                hr = np.mean([rank_avg(S[s][ii]) for s in range(nseeds)], axis=0)
                pick = int(np.argmax(hr))
                # ---- per-seed picks (spread)
                picks_seed = [int(np.argmax(rank_avg(S[s][ii]))) for s in range(nseeds)]
                # ---- answer-prior pick (row space), ties -> more votes, then first index
                vote = Counter(pn)
                best, bk = None, None
                for j, i in enumerate(ii):
                    r = rows[i]
                    c, n = prior.get(em_norm(r["ans"]), (0, 0))
                    p = (c + 1.0) / (n + 2.0) if n > 0 else gprior
                    k = (p, vote.get(row_norm(r["ans"]), 0))
                    if bk is None or k > bk:
                        bk, best = k, j
                gpred = g["preds"][0]
                rec = {"idx": q, "img": rows[ii[0]]["img_md5"], "gold": gold,
                       "g": {"j": int(gjudge[q]["judge_ok"]), "em": em_score(gpred, gold),
                             "st": strict_em(gpred, gold), "f1": token_f1(gpred, gold),
                             "w": nwords(gpred)},
                       "pick": pick, "picks_seed": picks_seed, "prior_pick": best,
                       "cands": [{"ans": rows[i]["ans"], "j": int(rows[i]["y"]),
                                  "em": em_score(rows[i]["ans"], gold),
                                  "st": strict_em(rows[i]["ans"], gold),
                                  "f1": token_f1(rows[i]["ans"], gold),
                                  "w": nwords(rows[i]["ans"])} for i in ii],
                       "pool": {"j": ys8, "em": ems8, "st": sts8, "f1": f1s8, "mslot": ms}}
                recs.append(rec)
            M = np.zeros((len(recs), len(COLS)))
            for n, r in enumerate(recs):
                c = r["cands"]; v = c[r["pick"]]; pp = c[r["prior_pick"]]
                for cur in ("j", "em", "st", "f1"):
                    M[n, CI_[f"g_{cur}"]] = r["g"][cur]
                    M[n, CI_[f"v_{cur}"]] = v[cur]
                    M[n, CI_[f"p_{cur}"]] = pp[cur]
                    M[n, CI_[f"r_{cur}"]] = float(np.mean(r["pool"][cur]))
                    M[n, CI_[f"m_{cur}"]] = r["pool"][cur][r["pool"]["mslot"]]
                    M[n, CI_[f"o_{cur}"]] = max(r["pool"][cur])
            imgs = [r["img"] for r in recs]
            R, ncl = cluster_boot(M, imgs, NBOOT, np.random.default_rng(child[bi]))
            mean = M.mean(0)
            o = {"n_questions": len(recs), "n_images": ncl, "dropped": dict(drop),
                 "artifact_n_questions": art["cells"].get(cell, {}).get("n_questions")}
            for cur, nm in (("j", "judge"), ("em", "em"), ("st", "strict_em"), ("f1", "token_f1")):
                for a, an in (("g", "greedy"), ("r", "random_pick"), ("m", "majority"),
                              ("v", "probe"), ("o", "oracle8"), ("p", "answer_prior")):
                    o[f"{an}_{nm}"] = float(mean[CI_[f"{a}_{cur}"]])
                orc, rnd = mean[CI_[f"o_{cur}"]], mean[CI_[f"r_{cur}"]]
                o[f"sel_eff_over_oracle_{nm}"] = float(mean[CI_[f"v_{cur}"]] / orc) if orc > 0 else None
                o[f"sel_eff_gap_normalised_{nm}"] = float((mean[CI_[f"v_{cur}"]] - rnd) / (orc - rnd)) \
                    if orc > rnd else None
            CON = {"probe_minus_greedy_judge": ("v_j", "g_j"),
                   "probe_minus_greedy_em": ("v_em", "g_em"),
                   "probe_minus_greedy_strict_em": ("v_st", "g_st"),
                   "probe_minus_greedy_token_f1": ("v_f1", "g_f1"),
                   "probe_minus_random_judge": ("v_j", "r_j"),
                   "probe_minus_majority_judge": ("v_j", "m_j"),
                   "probe_minus_majority_em": ("v_em", "m_em"),
                   "probe_minus_answerprior_judge": ("v_j", "p_j"),
                   "probe_minus_answerprior_em": ("v_em", "p_em"),
                   "answerprior_minus_greedy_judge": ("p_j", "g_j"),
                   "majority_minus_greedy_judge": ("m_j", "g_j"),
                   "currency_gap_judge_minus_em": None}
            reps[cell], point[cell] = {}, {}
            for nm, ab in CON.items():
                if ab is None:
                    rv = (R[:, CI_["v_j"]] - R[:, CI_["g_j"]]) - (R[:, CI_["v_em"]] - R[:, CI_["g_em"]])
                    d = float((mean[CI_["v_j"]] - mean[CI_["g_j"]]) - (mean[CI_["v_em"]] - mean[CI_["g_em"]]))
                else:
                    a, b = ab
                    rv = R[:, CI_[a]] - R[:, CI_[b]]
                    d = float(mean[CI_[a]] - mean[CI_[b]])
                lo, hi = ci(rv)
                o[nm] = d; o["ci_" + nm] = [lo, hi]; o["verdict_" + nm] = verdict(lo, hi)
                reps[cell][nm], point[cell][nm] = rv, d
            # per-seed spread of the judge delta
            sd_ = []
            for s in range(nseeds):
                sd_.append(float(np.mean([r["cands"][r["picks_seed"][s]]["j"] for r in recs])
                                 - np.mean([r["g"]["j"] for r in recs])))
            o["per_seed_probe_minus_greedy_judge"] = sd_
            # diagnostics
            ys = np.array([y for r in recs for y in r["pool"]["j"]])
            es = np.array([e for r in recs for e in r["pool"]["em"]])
            fs = np.array([f for r in recs for f in r["pool"]["f1"]])
            gy = np.array([r["g"]["j"] for r in recs]); ge = np.array([r["g"]["em"] for r in recs])
            gf = np.array([r["g"]["f1"] for r in recs])
            vy = M[:, CI_["v_j"]]; ve = M[:, CI_["v_em"]]
            vf = M[:, CI_["v_f1"]]
            o["diagnostics"] = {
                "pool_P_judge1_given_em0": float(ys[es == 0].mean()) if (es == 0).any() else None,
                "pool_P_judge1_given_f1zero": float(ys[fs == 0].mean()) if (fs == 0).any() else None,
                "pool_frac_f1zero": float((fs == 0).mean()),
                "greedy_P_judge1_given_em0": float(gy[ge == 0].mean()) if (ge == 0).any() else None,
                "greedy_P_judge1_given_f1zero": float(gy[gf == 0].mean()) if (gf == 0).any() else None,
                "probe_P_judge1_given_em0": float(vy[ve == 0].mean()) if (ve == 0).any() else None,
                "probe_P_judge1_given_f1zero": float(vy[vf == 0].mean()) if (vf == 0).any() else None,
                "probe_judgeYes_emNo": float(((vy == 1) & (ve == 0)).mean()),
                "greedy_judgeYes_emNo": float(((gy == 1) & (ge == 0)).mean()),
                "mean_words_greedy": float(np.mean([r["g"]["w"] for r in recs])),
                "mean_words_probe_pick": float(np.mean([r["cands"][r["pick"]]["w"] for r in recs])),
                "frac_pick_same_string_as_greedy": float(np.mean(
                    [row_norm(r["cands"][r["pick"]]["ans"]) == row_norm(r["g"].get("ans", "\x00"))
                     for r in recs])) if recs and "ans" in recs[0]["g"] else None,
                "mean_candidate_rows_per_question": float(np.mean([len(r["cands"]) for r in recs])),
            }
            per[cell] = o
            print(f"  {cell:17} n{len(recs):6} img{ncl:6} greedy_J {o['greedy_judge']:.4f} "
                  f"probe_J {o['probe_judge']:.4f} dJ {o['probe_minus_greedy_judge']:+.4f} "
                  f"dEM {o['probe_minus_greedy_em']:+.4f} prior_J {o['answer_prior_judge']:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        # -------- macro
        macro = {"n_benchmarks": len(BENCH),
                 "n_questions": int(sum(per[c]["n_questions"] for c in BENCH))}
        for nm in ("greedy", "random_pick", "majority", "probe", "oracle8", "answer_prior"):
            for cur in ("judge", "em", "strict_em", "token_f1"):
                macro[f"{nm}_{cur}"] = float(np.mean([per[c][f"{nm}_{cur}"] for c in BENCH]))
        rng8 = np.random.default_rng(child[-1])
        bidx = rng8.integers(0, len(BENCH), size=(NBOOT, len(BENCH)))
        for nm in reps[BENCH[0]]:
            pts = np.array([point[c][nm] for c in BENCH])
            RR = np.stack([reps[c][nm] for c in BENCH], axis=1)
            lo1, hi1 = ci(RR.mean(1))
            bm = pts[bidx].mean(1); lo2, hi2 = ci(bm)
            sd = pts.std(ddof=1)
            macro[nm] = {"macro": float(pts.mean()),
                         "per_benchmark": {c: point[c][nm] for c in BENCH},
                         "n_positive": int((pts > 0).sum()),
                         "n_significant_positive_image_clustered":
                             int(sum(1 for c in BENCH if per[c]["ci_" + nm][0] > 0)),
                         "n_significant_negative_image_clustered":
                             int(sum(1 for c in BENCH if per[c]["ci_" + nm][1] < 0)),
                         "ci_image_clustered": [lo1, hi1], "verdict_image_clustered": verdict(lo1, hi1),
                         "ci_benchmark_level_n8": [lo2, hi2], "verdict_benchmark_level": verdict(lo2, hi2),
                         "sd_across_benchmarks": float(sd),
                         "t_interval_df7": [float(pts.mean() - 2.3646 * sd / np.sqrt(8)),
                                            float(pts.mean() + 2.3646 * sd / np.sqrt(8))]}
        # headroom split
        lo_cells = [c for c in BENCH if per[c]["greedy_judge"] < 0.10]
        hi_cells = [c for c in BENCH if per[c]["greedy_judge"] >= 0.20]
        d = {c: per[c]["probe_minus_greedy_judge"] for c in BENCH}
        macro["headroom"] = {
            "cells_greedy_lt_0.10": lo_cells,
            "contribution_of_lt0.10_cells_to_macro": float(sum(d[c] for c in lo_cells) / 8),
            "share_of_macro_from_lt0.10_cells": float(sum(d[c] for c in lo_cells) / sum(d.values()))
                if sum(d.values()) != 0 else None,
            "cells_greedy_ge_0.20": hi_cells,
            "macro_restricted_to_ge0.20": float(np.mean([d[c] for c in hi_cells])) if hi_cells else None,
            "macro_all8": float(np.mean(list(d.values())))}
        RES[gen] = {"tag": G["tag"], "layer": G["L"], "seeds": int(nseeds),
                    "refit_meta": {k: v for k, v in MT.items() if k != "heldout_rows"},
                    "artifact_compared": G["art"],
                    "artifact_macro": art.get("macro"), "artifact_beats": art.get("beats_greedy"),
                    "answer_prior": {"n_train_rows": ntr, "n_distinct_strings": len(prior),
                                     "global_P_correct": gprior},
                    "cells": per, "macro": macro}
        print(json.dumps({k: v for k, v in macro.items() if not isinstance(v, dict)}, indent=1), flush=True)
        for nm in ("probe_minus_greedy_judge", "probe_minus_greedy_em", "probe_minus_greedy_strict_em",
                   "probe_minus_greedy_token_f1", "probe_minus_answerprior_judge",
                   "answerprior_minus_greedy_judge", "majority_minus_greedy_judge"):
            m = macro[nm]
            print(f"  MACRO {nm:36} {m['macro']:+.4f} imgCI {m['ci_image_clustered']} "
                  f"{m['verdict_image_clustered']}  bench-n8 {m['ci_benchmark_level_n8']} "
                  f"{m['verdict_benchmark_level']}  pos {m['n_positive']}/8  "
                  f"sig+ {m['n_significant_positive_image_clustered']}/8", flush=True)
        print("  headroom:", json.dumps(macro["headroom"]), flush=True)
        json.dump(RES, open(f"{OUT}/replication_currency_2026-09-20.json", "w"), indent=1)
    json.dump(RES, open(f"{OUT}/replication_currency_2026-09-20.json", "w"), indent=1)
    print("wrote replication_currency_2026-09-20.json  [%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main()
