#!/usr/bin/env python3
"""s12: quantify Gemma chat-template leakage / degeneration in the sc8 candidate pools, and re-run
the probe-vs-greedy judge delta on the CLEAN subset (questions with no degenerate candidate).
READ-ONLY."""
import os, sys, json, re
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, "/data/dan/audit_2026-09-18/tmp/em-rescore")
from em_rescore_pooled_probe import em_norm, em_score, strict_em, token_f1, row_norm, nwords
MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(MAIN, "src/training_methods"))
from genframe_data import rank_avg
CK = os.path.join(MAIN, "ckpts/openvqa/cheap_lingshu7b")
OUT = "/data/dan/audit_2026-09-18/tmp/replication"
B = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
     "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
TAG = {"qwen": "qwen25vl7b", "medgemma": "medgemma4b", "lingshu": "lingshu7b"}
ROLE = re.compile(r"(^|\n)\s*(model|user|image|output|response|label|answer|argument|display|"
                  r"display_name|display comment|tool_code|input image)\b", re.I)


def degenerate(s):
    """chat-template role marker on its own line, a code fence, or a sentence repeated >=3x."""
    if "```" in s or "<end_of_turn>" in s or "<start_of_turn>" in s:
        return True
    if ROLE.search(s):
        return True
    parts = [p.strip().lower() for p in re.split(r"[\n.]+", s) if p.strip()]
    if parts and max(Counter(parts).values()) >= 3:
        return True
    return False


RES = {}
for gen in ("qwen", "medgemma", "lingshu"):
    tag = TAG[gen]
    cells = {}
    for cell in B:
        n_c = n_deg = n_q = n_qdeg = 0
        wl_clean, wl_deg = [], []
        for l in open(f"{CK}/ckpt_{cell}_{tag}_sc8.jsonl"):
            if not l.strip():
                continue
            d = json.loads(l)
            n_q += 1
            flags = [degenerate(a) for a in d["preds"]]
            n_c += len(flags); n_deg += sum(flags)
            if any(flags):
                n_qdeg += 1
            for a, f in zip(d["preds"], flags):
                (wl_deg if f else wl_clean).append(nwords(a))
        gdeg = gn = 0
        for l in open(f"{CK}/ckpt_{cell}_{tag}.jsonl"):
            if l.strip():
                d = json.loads(l); gn += 1; gdeg += degenerate(d["preds"][0])
        cells[cell] = {"n_questions": n_q, "n_candidates": n_c,
                       "frac_candidates_degenerate": n_deg / max(n_c, 1),
                       "frac_questions_with_any_degenerate": n_qdeg / max(n_q, 1),
                       "frac_greedy_degenerate": gdeg / max(gn, 1),
                       "mean_words_clean_candidates": float(np.mean(wl_clean)) if wl_clean else None,
                       "mean_words_degenerate_candidates": float(np.mean(wl_deg)) if wl_deg else None}
        print("%-9s %-17s cand_deg %.3f  q_any_deg %.3f  greedy_deg %.3f  w_clean %s w_deg %s" % (
            gen, cell, cells[cell]["frac_candidates_degenerate"],
            cells[cell]["frac_questions_with_any_degenerate"], cells[cell]["frac_greedy_degenerate"],
            ("%.1f" % cells[cell]["mean_words_clean_candidates"]) if wl_clean else "-",
            ("%.1f" % cells[cell]["mean_words_degenerate_candidates"]) if wl_deg else "-"), flush=True)
    RES[gen] = {"cells": cells,
                "macro": {k: float(np.mean([cells[c][k] for c in B if cells[c][k] is not None]))
                          for k in cells[B[0]] if k not in ("n_questions", "n_candidates")}}
    print("  MACRO", gen, json.dumps({k: round(v, 4) for k, v in RES[gen]["macro"].items()}), flush=True)

# ---- probe delta on the CLEAN subset (questions whose whole pool is clean)
print("\n### probe - greedy (judge) restricted to questions with NO degenerate candidate")
for gen, L in (("qwen", 20), ("medgemma", 24)):
    tag = TAG[gen]
    MT = json.load(open(f"{OUT}/refit/{gen}_L{L}.meta.json"))
    Z = np.load(f"{OUT}/refit/{gen}_L{L}.npz")
    ns = Z[B[0]].shape[0]
    out = {}
    for cell in B:
        rows = MT["heldout_rows"][cell]; S = Z[cell]
        gj = {}
        for l in open(f"{CK}/ckpt_{cell}_{tag}.judge.jsonl"):
            if l.strip():
                d = json.loads(l); gj[d["idx"]] = int(d["judge_ok"])
        sc = {}
        for l in open(f"{CK}/ckpt_{cell}_{tag}_sc8.jsonl"):
            if l.strip():
                d = json.loads(l); sc[d["idx"]] = d
        byq = defaultdict(list)
        for i, r in enumerate(rows):
            byq[r["idx"]].append(i)
        a_all, a_cl, g_all, g_cl = [], [], [], []
        for q in sorted(byq):
            if q not in gj or q not in sc:
                continue
            ii = np.array(byq[q])
            hr = np.mean([rank_avg(S[s][ii]) for s in range(ns)], axis=0)
            v = int(rows[ii[int(np.argmax(hr))]]["y"])
            a_all.append(v); g_all.append(gj[q])
            if not any(degenerate(a) for a in sc[q]["preds"]):
                a_cl.append(v); g_cl.append(gj[q])
        out[cell] = {"n_all": len(a_all), "n_clean": len(a_cl),
                     "frac_clean": len(a_cl) / max(len(a_all), 1),
                     "delta_all": float(np.mean(a_all) - np.mean(g_all)),
                     "delta_clean": float(np.mean(a_cl) - np.mean(g_cl)) if a_cl else None,
                     "greedy_clean": float(np.mean(g_cl)) if a_cl else None}
        print("%-9s %-17s n %5d clean %5d (%.2f)  d_all %+.4f  d_clean %s" % (
            gen, cell, out[cell]["n_all"], out[cell]["n_clean"], out[cell]["frac_clean"],
            out[cell]["delta_all"],
            ("%+.4f" % out[cell]["delta_clean"]) if a_cl else "n/a"), flush=True)
    ok = [c for c in B if out[c]["delta_clean"] is not None and out[c]["n_clean"] >= 40]
    print("  MACRO %s  d_all %+.4f | d_clean %+.4f over %d cells with >=40 clean questions (%s)" % (
        gen, float(np.mean([out[c]["delta_all"] for c in B])),
        float(np.mean([out[c]["delta_clean"] for c in ok])) if ok else float("nan"), len(ok),
        ",".join(ok)), flush=True)
    RES[gen]["clean_subset"] = out
json.dump(RES, open(f"{OUT}/s12_degenerate.json", "w"), indent=1)
