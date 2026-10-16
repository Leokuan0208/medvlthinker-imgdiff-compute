#!/usr/bin/env python3
"""wed_compare.py -- our probe verifier vs Wasserstein Equilibrium Decoding (WED, arXiv 2605.18313) on IDENTICAL
candidate pools and questions (cap320 T=0.7 Lingshu-7B pools; WED's picks from src/exp_2026_10/wed/wed_select.py).

Probe picks come from the FINAL evaluation records (final_eval_FINAL_cap320_T07_records.json), so every question
is scored by a probe that never saw its image. Currencies, all on the same questions:
  judge     MedVLThinker-32B label of the picked string (pool exploded judge file; greedy file for greedy)
  xjudge    MedGemma-27B-it label (xjudge_c320/merged)
  em, f1    run_openvqa lenient EM / project token-F1 (em_rescore_pooled_probe)
  llava_recall  LLaVA-Med v1.0.0 open recall (verbatim)
  wed_em, wed_f1  WED's OWN metrics as its paper defines them: EM = lowercased, stripped string equality;
            F1 over token SETS (lowercased whitespace tokens)
Paired image-clustered bootstrap (10,000) of probe - WED and of each - greedy, per benchmark.
  python3 wed_compare.py
"""
import json, os, sys
from collections import defaultdict
import numpy as np

MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, MAIN + "/.claude/worktrees/exp-2026-10-15/src/audit_2026_09_18")
sys.path.insert(0, MAIN + "/.claude/worktrees/exp-2026-10-15/src/exp_2026_10")
sys.path.insert(0, "/data/dan/repos/llava_med_v1.0.0_eval")
import em_rescore_pooled_probe as ER      # noqa: E402
from final_method_eval import llava        # noqa: E402

CK = MAIN + "/ckpts/openvqa/cheap_lingshu7b"
REC = "/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json"
WED = "/data/dan/exp_2026_10/wed"
XJ = "/data/dan/exp_2026_10/xjudge_c320/merged.jsonl"
OUT = MAIN + "/results/cascade_methods/artifacts/wed_vs_probe_2026-10-16.json"
CELLS = ["pathvqa_open", "vqa_rad_open", "slake_open"]


_CANON = None


def canon(p):
    """WED's own canonicalize_answer (repo utils.py:193), imported, applied to every PREDICTION before its
    metrics -- in WED's main.py the scored answers are canonicalised candidates; the gold is not."""
    global _CANON
    if _CANON is None:
        hf = os.environ.get("HF_HOME")
        sys.path.insert(0, "/data/dan/repos/Wasserstein-BDG-medical-VQA-main")
        from utils import canonicalize_answer   # noqa: E402  (authors' code, unlicensed: imported, not copied)
        if hf is not None:
            os.environ["HF_HOME"] = hf            # the repo's config.py overwrites HF_HOME at import
        _CANON = canonicalize_answer
    return _CANON(str(p))


def wed_em(p, g):
    """WED utils.evaluate_answer 'exact': canonicalised pred, both stripped + lowercased, equality."""
    return float(canon(p).strip().lower() == str(g).strip().lower())


def wed_f1(p, g):
    """WED utils.evaluate_answer 'f1' over token sets (equality fallback when either side is empty)."""
    a, b = set(canon(p).strip().lower().split()), set(str(g).strip().lower().split())
    if not a or not b:
        return float(canon(p).strip().lower() == str(g).strip().lower())
    if not a or not b or not (a & b):
        return 0.0
    pr, rc = len(a & b) / len(a), len(a & b) / len(b)
    return 2 * pr * rc / (pr + rc)


def main():
    R = json.load(open(REC))
    xl = {}
    lab = {}
    for l in open(XJ.replace(".jsonl", ".judge.jsonl")):
        d = json.loads(l); lab.setdefault(d["idx"], d["judge_ok"])
    for l in open(XJ):
        d = json.loads(l); c, q, _ = d["idx"].split("|"); xl[(c, q, d["modal_pred"])] = lab.get(d["idx"])
    out = {"title": "Probe verifier vs WED on identical pools", "no_fabricated_numbers": True,
           "probe_records": REC, "wed_dir": WED, "cells": {}}
    rng = np.random.default_rng(20261019)
    for c in CELLS:
        wp = f"{WED}/wed_{c}_c320.jsonl"
        if not os.path.exists(wp):
            print("missing", wp); continue
        wed = {d["idx"]: d for d in (json.loads(l) for l in open(wp) if l.strip())}
        sc8 = ER.loadj(f"{CK}/ckpt_{c}_lingshu7b_sc8.jsonl")
        exp = ER.loadj(f"{CK}/ckpt_{c}_lingshu7b_sc8_scexploded.jsonl")
        jud = ER.loadj(f"{CK}/ckpt_{c}_lingshu7b_sc8_scexploded.judge.jsonl")
        jmap = defaultdict(dict)
        for cid, r in exp.items():
            if cid in jud:
                q = cid.split("#")[0]; q = int(q) if q.lstrip("-").isdigit() else q
                jmap[q][ER.row_norm(r["modal_pred"])] = int(jud[cid]["judge_ok"])
        recs = {r["idx"]: r for r in R[c]}
        rows, imgs = [], []
        for q, r in recs.items():
            w = wed.get(q)
            if w is None or q not in sc8:
                continue
            gold = sc8[q]["gold"]
            wa = w["pick_text"]

            def cur(ans, judge):
                return {"judge": judge, "xjudge": xl.get((c, str(q), ans)), "em": ER.em_score(ans, gold),
                        "f1": ER.token_f1(ans, gold), "llava_recall": llava(ans, gold)[0],
                        "wed_em": wed_em(ans, gold), "wed_f1": wed_f1(ans, gold)}
            rows.append({"probe": cur(r["pick_ans"], r["pick"]["judge"]),
                         "wed": cur(wa, jmap[q].get(ER.row_norm(wa))),
                         "greedy": cur(r["greedy_ans"], r["greedy"]["judge"])})
            imgs.append(r["img"])
        keys, inv = np.unique(imgs, return_inverse=True)
        W = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=10000).astype(float)
        o = {"n": len(rows), "n_wed_picks_on_file": len(wed)}
        for cu in rows[0]["probe"]:
            for a in ("probe", "wed", "greedy"):
                v = [x[a][cu] for x in rows if x[a][cu] is not None]
                o[f"{a}_{cu}"] = float(np.mean(v)) if v else None
            for a, b in (("probe", "wed"), ("probe", "greedy"), ("wed", "greedy")):
                m = np.array([x[a][cu] is not None and x[b][cu] is not None for x in rows])
                d = np.array([(x[a][cu] - x[b][cu]) if mm else 0.0 for x, mm in zip(rows, m)])
                Rs = (W @ np.bincount(inv, weights=d, minlength=len(keys))) / np.maximum(
                    W @ np.bincount(inv, weights=m.astype(float), minlength=len(keys)), 1e-12)
                lo, hi = float(np.percentile(Rs, 2.5)), float(np.percentile(Rs, 97.5))
                o[f"{a}_minus_{b}_{cu}"] = {"delta": float(d[m].mean()), "ci": [lo, hi],
                                            "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
                                            "n": int(m.sum())}
        out["cells"][c] = o
        print(c, o["n"], {cu: (round(o[f"probe_minus_wed_{cu}"]["delta"], 4), o[f"probe_minus_wed_{cu}"]["verdict"])
                          for cu in rows[0]["probe"]}, flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
