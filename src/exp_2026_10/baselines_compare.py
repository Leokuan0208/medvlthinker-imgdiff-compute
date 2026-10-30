#!/usr/bin/env python3
"""baselines_compare.py -- every owed baseline from the 2026-10-19 deck, against the shipped probe, on identical questions.

Probe = the deck's headline method: FINAL run, T=0.7 pools, frozen final probes, rank_vote, both folds
(/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json). Baselines (each scored only where it produced an
answer; questions with no answer are counted and reported, never silently dropped from one arm only):
  usc_T07 / usc_Tmap   Universal Self-Consistency over the same pools (usc_select.py)
  lingshu32b_greedy    the 4.6x bigger model, greedy, same prompt / cap / token limit (chain 1)
  lora_verifier_pooled the July LoRA verifier retrained on the probe's exact data universe (lora_verifier_crossfit.py)
  sft_greedy           Lingshu-7B LoRA-SFT on the gold answers of that same universe, merged, greedy (chain 2)
  wed_published        WED with its own sampling + selection, on PathVQA / SLAKE / VQA-RAD (chain 2)
  wed_port             WED's selection over our fixed pools (FINAL_METHOD sec 5.3), same three benchmarks
Currencies on identical answers: judge of record (MedVLThinker-32B), cross-family MedGemma-27B-it, lenient EM.
For each: baseline - greedy, and probe - baseline; paired, image-clustered (10,000), macro over the benchmarks the
baseline covers. Freshly generated answer sets (32B, SFT, WED-published) also get a stop-token audit (mean tokens,
share at the 64-token cap) -- the check that invalidated MedGemma-4B.

  python3 src/exp_2026_10/baselines_compare.py
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_method_eval as FME                     # noqa: E402

ER, OPEN8 = FME.ER, FME.OPEN8
M = FME.MAIN; CK = FME.CK
OUT = os.path.join(FME.WT, "results/cascade_methods/artifacts/baselines_compare_2026-10-22.json")
NB, SEED = 10000, 20261022
loadjl = lambda p: [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

# ---- labels
xlab, xj = {}, {}
XJS = ["/data/dan/exp_2026_10/xjudge_c320/merged.jsonl", "/data/dan/exp_2026_10/xjudge_1022/merged.jsonl"]
for X in XJS:
    for d in loadjl(X.replace(".jsonl", ".judge.jsonl")):
        xj.setdefault(d["idx"], int(d["judge_ok"]))
for X in XJS:   # (cell, q, answer) -> its first LABELLED idx: one answer string can sit in several work lists
    for d in loadjl(X):
        if d["idx"] in xj:
            xlab.setdefault(tuple(d["idx"].split("|")[:2]) + (d["modal_pred"],), d["idx"])
xl = lambda c, q, a: (lambda k: xj.get(k) if k is not None else None)(xlab.get((c, str(q), a)))
NEWJ = {d["idx"]: int(d["judge_ok"]) for d in loadjl("/data/dan/exp_2026_10/xjudge_1022/new_all.judge.jsonl")}

# ---- probe + greedy (FINAL T07 records), gold, and the T07 pool rows (answer -> judge label)
REC = json.load(open("/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json"))
base = {(c, r["idx"]): r for c in OPEN8 for r in REC[c]}
RECM = json.load(open("/data/dan/exp_2026_10/final_eval_FINAL_cap320_Tmap_records.json"))   # FINAL_METHOD's policy
baseM = {(c, r["idx"]): r for c in OPEN8 for r in RECM[c]}
gold = {(c, k): v["gold"] for c in OPEN8 for k, v in FME.loadj(f"{CK}/ckpt_{c}_lingshu7b_sc8.jsonl").items()}
ROWS = {}
for c in OPEN8:
    for pol, tag in (("T07", "lingshu7b_sc8"), ("Tmap", FME.pool_tag(c, "Tmap", True))):
        stem, dsf = FME.eval_stem(c, tag, True)
        for r in json.load(open(f"{FME.FEATS}/{stem}.meta.json"))["rows"]:
            if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf):
                ROWS.setdefault((pol, c, r["idx"], r["na"]), r)


def from_row(pol, c, q, na):
    r = ROWS.get((pol, c, q, na))
    if r is None:
        return None
    return {"judge": float(r["y"]), "em": float(ER.em_score(r["ans"], gold[(c, q)])),
            "xjudge": (lambda x: np.nan if x is None else float(x))(xl(c, q, r["ans"]))}


def fresh(c, q, ans, judge):
    return {"judge": np.nan if judge is None else float(judge), "em": float(ER.em_score(ans, gold[(c, q)])),
            "xjudge": (lambda x: np.nan if x is None else float(x))(xl(c, q, ans))}


B = defaultdict(dict); audit = {}
for pol in ("T07", "Tmap"):
    for d in loadjl(f"/data/dan/exp_2026_10/usc_{pol}.jsonl"):
        v = from_row(pol, d["cell"], d["idx"], ER.row_norm(d["ans"]))
        if v: B[f"usc_{pol}"][(d["cell"], d["idx"])] = dict(v, fallback=d["fallback"])
for f in (0, 1):
    for d in loadjl(f"/data/dan/exp_2026_10/lora_verifier_crossfit_fold{f}.jsonl"):
        v = from_row("T07", d["cell"], d["idx"], d["pick_na"])
        if v: B["lora_verifier_pooled"][(d["cell"], d["idx"])] = v
G32 = f"{M}/ckpts/openvqa/strong_lingshu32b_c320"
for c in OPEN8:
    J = {d["idx"]: d["judge_ok"] for d in loadjl(f"{G32}/ckpt_{c}_lingshu32b.judge.jsonl")}
    rows = loadjl(f"{G32}/ckpt_{c}_lingshu32b.jsonl")
    for d in rows:
        B["lingshu32b_greedy"][(c, d["idx"])] = fresh(c, d["idx"], d["modal_pred"], J.get(d["idx"]))
    if rows:
        t = [d.get("gen_tokens", 0) for d in rows]; audit[f"lingshu32b_greedy|{c}"] = {"mean_tokens": float(np.mean(t)), "at_cap_pct": 100 * float(np.mean(np.array(t) >= 64))}
for c in OPEN8:
    for f in (0, 1):
        rows = loadjl(f"{M}/ckpts/openvqa/sft_crossfit_1022/ckpt_{c}_sft1022f{f}.jsonl")
        for d in rows:
            B["sft_greedy"][(c, d["idx"])] = fresh(c, d["idx"], d["modal_pred"], NEWJ.get(f"{c}|{d['idx']}|sft{f}"))
        if rows:
            t = [d.get("gen_tokens", 0) for d in rows]; audit[f"sft_greedy|{c}|f{f}"] = {"mean_tokens": float(np.mean(t)), "at_cap_pct": 100 * float(np.mean(np.array(t) >= 64))}
for fp in glob.glob("/data/dan/exp_2026_10/wed_published/wed_pub_*.jsonl"):
    b = os.path.basename(fp)[len("wed_pub_"):-len(".jsonl")]; c = b.rsplit("_s", 1)[0] if b[-3:-1] == "_s" else b
    for d in loadjl(fp):
        B["wed_published"][(c, d["idx"])] = dict(fresh(c, d["idx"], d["pick_text"], NEWJ.get(f"{c}|{d['idx']}|wedpub")),
                                                 fallback=bool(d.get("fallback")))
for fp in glob.glob("/data/dan/exp_2026_10/wed/wed_*_c320.jsonl"):
    c = os.path.basename(fp)[len("wed_"):-len("_c320.jsonl")]
    if c == "pathvqa": c = "pathvqa_open"
    for d in loadjl(fp):
        v = from_row("T07", c, d["idx"], ER.row_norm(d["pick_text"]))
        if v: B["wed_port"][(c, d["idx"])] = v


def paired(keys, fa, fb, cu):
    """macro over the cells in keys of mean(fa - fb), image-clustered bootstrap"""
    rng = np.random.default_rng(SEED); reps, pts = [], []
    for c in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == c]
        a = np.array([fa(k)[cu] for k in ks], float); b = np.array([fb(k)[cu] for k in ks], float)
        d = a - b; m = ~np.isnan(d)
        if m.sum() == 0: continue
        img = [base[k]["img"] for k in ks]; keys_, inv = np.unique(img, return_inverse=True)
        w = rng.multinomial(len(keys_), np.full(len(keys_), 1 / len(keys_)), size=NB).astype(float); d = np.where(m, d, 0.0)
        reps.append((w @ np.bincount(inv, weights=d, minlength=len(keys_))) / np.maximum(w @ np.bincount(inv, weights=m.astype(float), minlength=len(keys_)), 1e-12))
        pts.append(d[m].mean())
    if not pts: return None
    Rm = np.stack(reps, 1).mean(1); lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
    return {"macro": float(np.mean(pts)), "ci": [lo, hi], "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"), "n_cells": len(pts)}


probe = lambda k: base[k]["pick"]; greedy = lambda k: base[k]["greedy"]
probeM = lambda k: baseM[k]["pick"]                   # a baseline run on the Tmap pools is compared with the Tmap probe
out = {"title": "Owed baselines vs the shipped probe (deck headline config), identical questions", "date": "2026-10-22",
       "no_fabricated_numbers": True, "script": os.path.abspath(__file__), "baselines": {}, "stop_token_audit": audit}
for name in ["usc_T07", "usc_Tmap", "lingshu32b_greedy", "lora_verifier_pooled", "sft_greedy", "wed_published", "wed_port"]:
    got = B.get(name, {})
    if not got:
        out["baselines"][name] = {"status": "not yet run (no inputs on disk)"}; print(f"{name:22} not yet run"); continue
    keys = [k for k in got if k in base]
    e = {"n_questions": len(keys), "cells": sorted({k[0] for k in keys}),
         "fallback_rate": float(np.mean([got[k].get("fallback", False) for k in keys]))}
    bl = lambda k: got[k]; pr = probeM if name.endswith("_Tmap") else probe
    e["probe"] = "Tmap records (final_eval_FINAL_cap320_Tmap)" if name.endswith("_Tmap") else "T07 records (final_eval_FINAL_cap320_T07)"
    for cu in ("judge", "xjudge", "em"):
        e[cu] = {"baseline_minus_greedy": paired(keys, bl, greedy, cu), "probe_minus_baseline": paired(keys, pr, bl, cu),
                 "probe_minus_greedy_same_questions": paired(keys, pr, greedy, cu),
                 "coverage": float(np.mean([not np.isnan(bl(k)[cu]) for k in keys]))}
    out["baselines"][name] = e
    j = e["judge"]
    if j["probe_minus_baseline"] is None:              # fresh answers whose judge labels are not in yet (e.g. SFT before judge_new)
        print(f"{name:22} n={len(keys):6} cells={len(e['cells'])}  judge: no labels yet | EM probe-baseline "
              f"{e['em']['probe_minus_baseline']['macro']:+.4f} {e['em']['probe_minus_baseline']['verdict']}"); continue
    print(f"{name:22} n={len(keys):6} cells={len(e['cells'])}  judge: baseline-greedy {j['baseline_minus_greedy']['macro']:+.4f} | "
          f"probe-baseline {j['probe_minus_baseline']['macro']:+.4f} {j['probe_minus_baseline']['verdict']}  "
          f"(MedGemma cov {e['xjudge']['coverage']:.2f})")
json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT)
