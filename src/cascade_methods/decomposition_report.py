#!/usr/bin/env python3
"""decomposition_report.py -- selection skill vs sampling penalty, in ONE currency.

    head - greedy = (head - one sample) - (greedy - one sample) = selection skill - sampling penalty

CURRENCY BUG THIS FIXES (found 2026-08-24, before the numbers reached a deck).  The first version of
this decomposition read `oks` out of the raw sc8 dump for the one-sample term.  That field is
EXACT MATCH, computed at generation time, while `greedy` and `head` are 32B-JUDGE labels: on
gemex_open the same 117 questions score 0.0288 by exact match and 0.3718 by judge.  Mixing them
inflated the sampling penalty 12.4x (gemex read penalty +0.2288 instead of the correct +0.018484,
this file's own artifact) and made the skill term meaningless.  Every arm here is judge currency.
The "roughly eightfold / +0.0285" this docstring used to claim was itself never read off an artifact
-- corrected 2026-09-13 against decomposition_2026-08-24.json cells.gemex_open["0.7"].

MULTIPLICITY MATTERS TOO.  explode_sc_for_judge.py dedups identical (question, answer) pairs before
judging -- the judge is text-only, so identical strings tie -- which means the judge file has one
row per DISTINCT answer, not per sample.  Averaging its rows gives the mean over distinct answers,
not the expected accuracy of one draw.  This maps each of the 8 raw samples back through its
normalised string to that string's judge label, so a mode sampled five times counts five times.

  python3 src/cascade_methods/decomposition_report.py
"""
import json, os
import numpy as np
from collections import defaultdict

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
OUT = os.path.join(ART, "decomposition_2026-08-24.json")
TAGS = {"lingshu7b": 0.7, "lingshu7bT02": 0.2, "lingshu7bT04": 0.4, "lingshu7bT10": 1.0}
CELLS = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]


def norm(s):
    return str(s).strip().lower().rstrip(".")


def main():
    bake = json.load(open(os.path.join(ART, "free_signal_bakeoff_2026-08-21.json")))["cells"]
    art = {"title": "Selection skill vs sampling penalty, all arms in judge currency",
           "date": "2026-08-24", "no_fabricated_numbers": True,
           "identity": "head - greedy = (head - one sample) - (greedy - one sample)",
           "head_is": "FROZEN incumbent verifier, read from free_signal_bakeoff arms_judge.frozen_head. head_temperature_sweep.py reports a head REFITTED per temperature under the same name -- the two agree exactly on 7 of 8 benchmarks and differ by 2 questions on slake_open",
           "cells": defaultdict(dict)}
    for cell in CELLS:
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not os.path.exists(gjp):
            continue
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        for tag, T in sorted(TAGS.items(), key=lambda kv: kv[1]):
            raw_p = f"{CK}/ckpt_{cell}_{tag}_sc8.jsonl"
            exp_p = f"{CK}/ckpt_{cell}_{tag}_sc8_scexploded.jsonl"
            jud_p = f"{CK}/ckpt_{cell}_{tag}_sc8_scexploded.judge.jsonl"
            if not all(os.path.exists(p) and os.path.getsize(p) for p in (raw_p, exp_p, jud_p)):
                continue
            jud = {}
            for l in open(jud_p):
                if l.strip():
                    d = json.loads(l); jud[d["idx"]] = int(d["judge_ok"])
            # composite idx 'q#k' -> the answer string that row judged
            lab = {}
            for l in open(exp_p):
                if l.strip():
                    d = json.loads(l)
                    if d["idx"] in jud:
                        q = str(d["idx"]).rsplit("#", 1)[0]
                        lab.setdefault(q, {})[norm(d["modal_pred"])] = jud[d["idx"]]
            one, orc, gr, n = [], [], [], 0
            for l in open(raw_p):
                if not l.strip():
                    continue
                r = json.loads(l)
                q = str(r["idx"])
                key = r["idx"] if r["idx"] in gok else (int(q) if q.isdigit() and int(q) in gok
                                                        else None)
                if key is None or q not in lab:
                    continue
                m = lab[q]
                ys = [m.get(norm(p)) for p in r["preds"]]
                ys = [v for v in ys if v is not None]
                if not ys:
                    continue
                one.append(float(np.mean(ys)))          # multiplicity-weighted: one draw
                orc.append(int(max(ys)))
                gr.append(gok[key]); n += 1
            if n < 50:
                continue
            one_m, orc_m, gr_m = float(np.mean(one)), float(np.mean(orc)), float(np.mean(gr))
            head_m = (bake.get(cell, {}).get("arms_judge", {}).get("frozen_head")
                      if T == 0.7 else None)
            rec = {"n_questions": n, "greedy": gr_m, "one_sample": one_m, "oracle_at_8": orc_m,
                   "sampling_penalty": gr_m - one_m, "headroom_above_greedy": orc_m - gr_m}
            if head_m is not None:
                rec.update({"head": head_m, "selection_skill": head_m - one_m,
                            "head_minus_greedy": head_m - gr_m})
            art["cells"][cell][str(T)] = rec
            extra = (f"  head {head_m:.4f} skill {head_m-one_m:+.4f} h-g {head_m-gr_m:+.4f}"
                     if head_m is not None else "")
            print(f"  {cell:17} T={T:<4} n{n:6} greedy {gr_m:.4f} 1samp {one_m:.4f} "
                  f"penalty {gr_m-one_m:+.4f} oracle {orc_m:.4f}{extra}", flush=True)
    art["cells"] = dict(art["cells"])
    at7 = {c: v["0.7"] for c, v in art["cells"].items() if "0.7" in v and "selection_skill" in v["0.7"]}
    if at7:
        sk = [v["selection_skill"] for v in at7.values()]
        art["summary_T07"] = {
            "n_cells": len(at7),
            "cells_with_positive_selection_skill":
                f"{sum(1 for x in sk if x > 0)}/{len(sk)}",
            "mean_selection_skill": float(np.mean(sk)),
            "mean_sampling_penalty": float(np.mean([v["sampling_penalty"] for v in at7.values()])),
            "mean_head_minus_greedy": float(np.mean([v["head_minus_greedy"] for v in at7.values()]))}
        print(f"\n  T=0.7: selection skill positive on "
              f"{art['summary_T07']['cells_with_positive_selection_skill']} cells; "
              f"mean skill {art['summary_T07']['mean_selection_skill']:+.4f}, "
              f"mean penalty {art['summary_T07']['mean_sampling_penalty']:+.4f}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
