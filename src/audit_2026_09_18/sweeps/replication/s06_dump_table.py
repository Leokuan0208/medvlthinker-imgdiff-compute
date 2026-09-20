#!/usr/bin/env python3
"""s06_dump_table.py -- READ-ONLY.  Per-benchmark, per-generator table straight from the dumps:
greedy / random-pick / majority-vote / oracle@8 in JUDGE currency, plus EM currencies and the
judge-vs-EM leniency diagnostics.  The verifier column is read from the head_final_stack artifacts.

Scorer definitions are IMPORTED VERBATIM from the audit's em-rescore script so both agents use one
definition of "normalised exact match".
"""
import os, sys, json, hashlib, time
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, "/data/dan/audit_2026-09-18/tmp/em-rescore")
from em_rescore_pooled_probe import em_norm, em_score, strict_em, token_f1, row_norm, half, nwords

MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
CK = os.path.join(MAIN, "ckpts/openvqa/cheap_lingshu7b")
ART = os.path.join(MAIN, "results/cascade_methods/artifacts")
OUT = "/data/dan/audit_2026-09-18/tmp/replication"
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
GENS = {"lingshu": ("lingshu7b", "head_final_stack_PVFIXED_2026-09-13.json"),
        "qwen": ("qwen25vl7b", "head_final_stack_qwen_2026-09-13.json"),
        "medgemma": ("medgemma4b", "head_final_stack_medgemma_ALL8_2026-09-16.json")}


def loadj(p, key="idx"):
    out = {}
    with open(p) as f:
        for l in f:
            if l.strip():
                d = json.loads(l)
                out[d[key]] = d
    return out


def build(tag):
    """per-benchmark per-question records from the dumps alone."""
    cells = {}
    for cell in BENCH:
        gp = f"{CK}/ckpt_{cell}_{tag}.jsonl"
        gj = f"{CK}/ckpt_{cell}_{tag}.judge.jsonl"
        sp = f"{CK}/ckpt_{cell}_{tag}_sc8.jsonl"
        sj = f"{CK}/ckpt_{cell}_{tag}_sc8_scexploded.judge.jsonl"
        if not all(os.path.exists(p) for p in (gp, gj, sp, sj)):
            print("  MISSING dumps for", tag, cell, flush=True); continue
        greedy, gjudge, sc8 = loadj(gp), loadj(gj), loadj(sp)
        cj = {}
        with open(sj) as f:
            for l in f:
                if l.strip():
                    d = json.loads(l)
                    q, s = str(d["idx"]).split("#")
                    cj[(int(q), int(s))] = int(d["judge_ok"])
        recs, miss = [], Counter()
        for q, gjd in gjudge.items():
            g, s8 = greedy.get(q), sc8.get(q)
            if g is None or s8 is None:
                miss["no_greedy_or_sc8"] += 1; continue
            gold = s8["gold"]
            preds = list(s8["preds"])
            pn = [row_norm(a) for a in preds]
            first = {}
            for i, a in enumerate(pn):
                first.setdefault(a, i)
            lab = {}
            ok = True
            for a, i in first.items():
                if (q, i) not in cj:
                    ok = False; break
                lab[a] = cj[(q, i)]
            if not ok:
                miss["slot_label_absent"] += 1; continue
            ys = [lab[a] for a in pn]
            ems = [em_score(a, gold) for a in preds]
            sts = [strict_em(a, gold) for a in preds]
            f1s = [token_f1(a, gold) for a in preds]
            # majority vote, em_norm grouping, first-max tie-break (run_openvqa.py:223 convention)
            gv = Counter(em_norm(a) for a in preds)
            order = []
            for a in preds:
                k = em_norm(a)
                if k not in order:
                    order.append(k)
            key = max(order, key=lambda k: gv[k])
            mslot = next(i for i, a in enumerate(preds) if em_norm(a) == key)
            gpred = g["preds"][0]
            recs.append({"idx": q, "gold": gold,
                         "g_j": int(gjd["judge_ok"]), "g_em": em_score(gpred, gold),
                         "g_st": strict_em(gpred, gold), "g_f1": token_f1(gpred, gold),
                         "g_w": nwords(gpred), "g_ans": gpred,
                         "ys": ys, "ems": ems, "sts": sts, "f1s": f1s,
                         "mslot": mslot, "npool": len(preds),
                         "n_uniq": len(first)})
        cells[cell] = {"recs": recs, "miss": dict(miss), "n_judged": len(gjudge)}
        print(f"  [{tag}] {cell:17} judged_q {len(gjudge):6} usable {len(recs):6} {dict(miss)}", flush=True)
    return cells


def summarise(cells, art):
    out = {}
    for cell, d in cells.items():
        R = d["recs"]
        if not R:
            continue
        o = {"n_questions": len(R), "n_judged_in_greedy_judge_file": d["n_judged"], "dropped": d["miss"]}
        for cur, gk, ck in (("judge", "g_j", "ys"), ("em", "g_em", "ems"),
                            ("strict_em", "g_st", "sts"), ("token_f1", "g_f1", "f1s")):
            o[f"greedy_{cur}"] = float(np.mean([r[gk] for r in R]))
            o[f"random_pick_{cur}"] = float(np.mean([np.mean(r[ck]) for r in R]))
            o[f"majority_{cur}"] = float(np.mean([r[ck][r["mslot"]] for r in R]))
            o[f"oracle8_{cur}"] = float(np.mean([max(r[ck]) for r in R]))
        a = art["cells"].get(cell, {})
        for arm in ("pooled_ens", "pooled_singlelayer", "pooled_ens_sc", "deployed_4dom_L21ish"):
            if arm in a:
                o[f"verifier_{arm}_judge"] = a[arm]
                o[f"verifier_{arm}_minus_greedy_judge"] = a[arm + "_minus_greedy"]
        o["artifact_greedy_judge"] = a.get("greedy")
        o["artifact_n_questions"] = a.get("n_questions")
        # selection efficiency, judge currency, for whichever arms exist
        for arm in ("pooled_ens", "pooled_singlelayer"):
            v = a.get(arm)
            if v is None:
                continue
            rp, orc = o["random_pick_judge"], o["oracle8_judge"]
            o[f"sel_eff_over_oracle_{arm}"] = float(v / orc) if orc > 0 else None
            o[f"sel_eff_gap_normalised_{arm}"] = float((v - rp) / (orc - rp)) if orc > rp else None
        # judge leniency / self-preference diagnostics on the candidate pool
        ys = np.array([y for r in R for y in r["ys"]])
        ems = np.array([e for r in R for e in r["ems"]])
        f1s = np.array([f for r in R for f in r["f1s"]])
        o["pool_diagnostics"] = {
            "n_candidate_slots": int(len(ys)),
            "P_judge1": float(ys.mean()), "P_em1": float(ems.mean()),
            "P_judge1_given_em0": float(ys[ems == 0].mean()) if (ems == 0).any() else None,
            "P_judge0_given_em1": float(1 - ys[ems == 1].mean()) if (ems == 1).any() else None,
            "P_judge1_given_f1_zero": float(ys[f1s == 0].mean()) if (f1s == 0).any() else None,
            "frac_f1_zero": float((f1s == 0).mean()),
            "mean_uniq_per_pool": float(np.mean([r["n_uniq"] for r in R])),
            "mean_pool_size": float(np.mean([r["npool"] for r in R]))}
        gy = np.array([r["g_j"] for r in R]); ge = np.array([r["g_em"] for r in R])
        gf = np.array([r["g_f1"] for r in R])
        o["greedy_diagnostics"] = {
            "P_judge1_given_em0": float(gy[ge == 0].mean()) if (ge == 0).any() else None,
            "P_judge1_given_f1_zero": float(gy[gf == 0].mean()) if (gf == 0).any() else None,
            "mean_words": float(np.mean([r["g_w"] for r in R]))}
        out[cell] = o
    return out


def main():
    t0 = time.time()
    res = {"_meta": {"date": "2026-09-20", "script": os.path.abspath(__file__),
                     "no_fabricated_numbers": True}}
    for gen, (tag, af) in GENS.items():
        print("===", gen, tag, flush=True)
        art = json.load(open(os.path.join(ART, af)))
        cells = build(tag)
        res[gen] = {"tag": tag, "artifact": af, "artifact_macro": art.get("macro"),
                    "artifact_beats_greedy": art.get("beats_greedy"),
                    "artifact_VERDICT": art.get("VERDICT"), "cells": summarise(cells, art)}
        # macro rows
        C = res[gen]["cells"]
        m = {}
        for k in ("greedy", "random_pick", "majority", "oracle8"):
            for cur in ("judge", "em", "strict_em", "token_f1"):
                m[f"{k}_{cur}"] = float(np.mean([C[c][f"{k}_{cur}"] for c in BENCH if c in C]))
        for arm in ("pooled_ens", "pooled_singlelayer"):
            vals = [C[c].get(f"verifier_{arm}_judge") for c in BENCH if c in C]
            if all(v is not None for v in vals) and vals:
                m[f"verifier_{arm}_judge"] = float(np.mean(vals))
                m[f"verifier_{arm}_minus_greedy_judge"] = float(
                    np.mean([C[c][f"verifier_{arm}_minus_greedy_judge"] for c in BENCH if c in C]))
        m["n_cells"] = len([c for c in BENCH if c in C])
        res[gen]["macro_from_dumps"] = m
        print(json.dumps(m, indent=1), flush=True)
    json.dump(res, open(os.path.join(OUT, "s06_dump_table.json"), "w"), indent=1)
    print("wrote s06_dump_table.json in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
