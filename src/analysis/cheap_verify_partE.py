#!/usr/bin/env python3
"""PART E -- the COMBINED cheap-intervention policy across all 8 cells, its cross-fit / balanced-key
filter, and a PAIRED SIGN-FLIP PERMUTATION NULL over the per-cell selection rule.
All per-item ok vectors are rebuilt here from the raw dumps."""
from __future__ import annotations
import csv, json, os, re, sys
from collections import Counter
import numpy as np
sys.path.insert(0, "/home/jamesyang/medvlthinker-imgdiff-compute/src")
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_cheapverify")
NBOOT, NPERM, SEED = 10000, 1000, 20260817
csv.field_size_limit(10**9)

def jl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def load_ob(stem):
    rows = []
    for s in ("_s0of2", "_s1of2", ""):
        p = os.path.join(ROOT, "ckpts/output_bias", stem + s + ".jsonl")
        if os.path.exists(p): rows += jl(p)
    return rows
def boot(d, nboot=NBOOT, seed=SEED):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)],
                sign=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"), n=int(len(d)))
def norm_text(s):
    s = str(s).strip().lower().strip("\"'“”‘’ \t\n").rstrip("。.")
    return re.sub(r"\s+", " ", s).strip()
AFF = {"yes","yeah","yep","true","correct","y"}; NEG = {"no","nope","false","incorrect","n","not"}
def polarity(r):
    for t in [x for x in re.split(r"[^a-z0-9]+", norm_text(r)) if x]:
        if t in AFF: return "yes"
        if t in NEG: return "no"
    return None
def strip_marker(t):
    t = str(t); return t[1:] if (t.startswith("Ġ") or t.startswith(" ")) else t
def letter_logits(row, K):
    lp = row.get("first_logprobs") or {}; best = {}
    for t, v in lp.items():
        s = strip_marker(t)
        if len(s) == 1 and "A" <= s <= "Z": best[s] = max(float(v), best.get(s, -1e9))
    floor = min([float(v) for v in lp.values()], default=-30.0) if lp else -30.0
    return np.array([best.get(chr(65 + i), floor) for i in range(K)])
def fit_shift(logits, target, iters=800, lr=0.3):
    n, K = logits.shape; w = np.zeros(K)
    for _ in range(iters):
        cur = np.bincount((logits - w).argmax(1), minlength=K) / n
        w = w + lr * np.log(np.maximum(cur, 1e-4) / np.maximum(target, 1e-4)); w -= w.mean()
    return w

# =============================================================== per-cell candidate delta vectors
D = {}   # cell -> {candidate_name: per-item paired delta vector}
META = {}

# ---- 1. PMC_VQA: output-side family, WITHIN-RUN vs the readout arm -----------------------------
ev = load_ob("gen_PMC_VQA_id"); ev.sort(key=lambda r: r["i"])
trd = load_ob("gen_PMC_TRAIN_train"); trd.sort(key=lambda r: r["i"])
X = np.array([letter_logits(r, 4) for r in ev])
gold = np.array([ord(str(r["answer"]).strip()[0]) - 65 for r in ev])
Xtr = np.array([letter_logits(r, 4) for r in trd])
tr = list(csv.DictReader(open("/data/dan/dataset/medevalkit/PMC-VQA/train_2.csv")))
tgt = np.bincount([ord(r["Answer"].strip()[0]) - 65 for r in tr], minlength=4) / len(tr)
ro = (X.argmax(1) == gold).astype(float)
pm = ((X - fit_shift(Xtr, tgt)).argmax(1) == gold).astype(float)
f = np.arange(len(X)) % 5; cvp = np.zeros(len(X), int)
for k in range(5):
    m = f != k; cvp[~m] = (X[~m] - fit_shift(X[m], tgt)).argmax(1)
cv = (cvp == gold).astype(float)
D["PMC_VQA"] = {"output_pm_train": pm - ro, "output_pm_transductive_cv": cv - ro}
META["PMC_VQA"] = dict(n=len(ev), gold_marginal=[float(x) for x in np.bincount(gold, minlength=4)/len(gold)])

# ---- 2. MedXpertQA-MM: same family ------------------------------------------------------------
mx = load_ob("gen_MedXpertQA-MM_id"); mx.sort(key=lambda r: r["i"])
Km = max(int(r.get("n_choices") or 0) for r in mx)
Xm = np.zeros((len(mx), Km)); gm = np.zeros(len(mx), int)
for k, r in enumerate(mx):
    n = int(r.get("n_choices") or Km)
    v = letter_logits(r, Km); v[n:] = -1e9; Xm[k] = v
    gm[k] = ord(str(r["answer"]).strip()[0]) - 65
rom = (Xm.argmax(1) == gm).astype(float)
fm = np.arange(len(Xm)) % 5; cvm = np.zeros(len(Xm), int)
tgt_m = np.full(Km, 1.0/Km)
for k in range(5):
    m = fm != k; cvm[~m] = (Xm[~m] - fit_shift(Xm[m], tgt_m)).argmax(1)
D["MedXpertQA-MM"] = {"output_pm_uniform_cv": (cvm == gm).astype(float) - rom}
META["MedXpertQA-MM"] = dict(n=len(mx), n_options=int(Km))

# ---- 3. three binary cells: PROMPT-SIDE, judge currency ---------------------------------------
CAO = os.path.join(ROOT, "ckpts/closed_as_open")
for cell in ("SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed"):
    ctl = {r["i"]: r for r in jl(f"{CAO}/gen_{cell}_closedD_g_full.jsonl")}
    itv = {r["i"]: r for r in jl(f"{CAO}/gen_{cell}_openMEK_g_full.jsonl")}
    meta = {r["idx"]: r for r in jl(f"{CAO}/judge_{cell}.jsonl")}
    lab = {r["idx"]: r["judge_ok"] for r in jl(f"{CAO}/judge_{cell}.judge.jsonl")}
    jm = {(meta[k]["i"], meta[k]["na"]): v for k, v in lab.items() if k in meta}
    ids = sorted(set(ctl) & set(itv))
    d = [jm[(i, norm_text(itv[i]["preds"][0]))] - jm[(i, norm_text(ctl[i]["preds"][0]))]
         for i in ids if (i, norm_text(itv[i]["preds"][0])) in jm and (i, norm_text(ctl[i]["preds"][0])) in jm]
    D[cell] = {"prompt_open_instruction": np.array(d, float)}
    META[cell] = dict(n=len(d))

# ---- 4. three open cells: DELETE the baseline's own self-consistency ---------------------------
from training_methods import genframe_data as G
_ = G.load_items()
DSMAP = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open", "pathvqa_open": "PATH_VQA_open"}
T0 = {k: f"ckpts/openvqa/cheap_lingshu7b/ckpt_{k}_lingshu7b.judge.jsonl" for k in DSMAP}
VD = os.path.join(ROOT, "ckpts/train/lora_verifier_pooled4")
for ds, cell in DSMAP.items():
    # MUST be the Lingshu-7B dump: this directory also holds MedVLThinker-7B (_7b) and
    # InternVL3-8B (_iv3_8b) dumps of the same datasets, with different item counts.
    fp = os.path.join(VD, f"transfer_dump_{ds}_lingshu7b.json")
    dump = json.load(open(fp))
    t0 = {r["idx"]: int(r["judge_ok"]) for r in jl(os.path.join(ROOT, T0[ds]))}
    d = [t0.get(r["idx"], 0) - int(r["greedy_ok"]) for r in dump
         if not all(x in (None, -1) for x in r["sl"])]
    EXPECT = {"SLAKE_open": 645, "VQA_RAD_open": 200, "PATH_VQA_open": 1500}
    assert len(d) == EXPECT[cell], f"{cell}: got {len(d)}, expected {EXPECT[cell]} -- wrong dump"
    D[cell] = {"delete_baseline_self_consistency": np.array(d, float)}
    META[cell] = dict(n=len(d), source=os.path.basename(fp))

R = {"META": META, "PER_CELL_CANDIDATES": {}}
for cell, cs in D.items():
    R["PER_CELL_CANDIDATES"][cell] = {k: boot(v) for k, v in cs.items()}

# =============================================================== the selection rule + its null
CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
def select(vecs):
    """per cell adopt the largest-delta candidate whose paired normal 95% interval excludes 0."""
    tot, chosen = 0.0, {}
    for cell in CELLS:
        best, bd = None, 0.0
        for k, v in vecs.get(cell, {}).items():
            m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            if se > 0 and abs(m) > 1.96 * se and m > bd:
                best, bd = k, m
        if best: chosen[cell] = (best, float(bd)); tot += bd
    return tot / 8.0, chosen

obs, chosen = select(D)
rng = np.random.default_rng(SEED)
null = []
for _ in range(NPERM):
    perm = {c: {k: v * rng.choice([-1.0, 1.0], size=len(v)) for k, v in cs.items()} for c, cs in D.items()}
    null.append(select(perm)[0])
null = np.array(null)
R["SELECTION_RULE"] = dict(
    rule="per cell adopt the largest-delta cheap correction whose paired 95% interval excludes 0, else nothing",
    observed_macro_gain=float(obs),
    chosen={k: {"candidate": v[0], "delta": v[1]} for k, v in chosen.items()},
    permutation_null=dict(kind="paired sign-flip of the per-item delta, within cell", nperm=NPERM,
                          mean=float(null.mean()), p95=float(np.percentile(null, 95)),
                          max=float(null.max()),
                          p_value=float((null >= obs).mean()),
                          z=float((obs - null.mean()) / null.std(ddof=1))),
    verdict=("SURVIVES" if float((null >= obs).mean()) < 0.05 else "DOES NOT SURVIVE"))

# =============================================================== the honest combined policies
pub = {"PMC_VQA":0.5427,"SLAKE_closed":0.8254,"VQA_RAD_closed":0.7809,"PATH_VQA_closed":0.8409,
       "MedXpertQA-MM":0.2615,"SLAKE_open":0.7364,"VQA_RAD_open":0.4650,"PATH_VQA_open":0.3240}
def macro_policy(name, picks, cost_per_cell):
    per, dv = {}, []
    for c in CELLS:
        if c in picks:
            v = D[c][picks[c]]; b = boot(v); per[c] = dict(intervention=picks[c], **b,
                                                           new_cell=pub[c] + b["delta"])
            dv.append(v)
        else:
            per[c] = dict(intervention="none", delta=0.0, ci=[0.0, 0.0], sign="TIE", n=0,
                          new_cell=pub[c])
    tot = sum(per[c]["delta"] for c in CELLS) / 8.0
    # macro bootstrap: resample items inside each moving cell independently
    rg = np.random.default_rng(SEED); bs = np.zeros(NBOOT)
    for v in dv:
        bs += v[rg.integers(0, len(v), size=(NBOOT, len(v)))].mean(1) / 8.0
    return dict(name=name, per_cell=per, macro_before=float(np.mean(list(pub.values()))),
                macro_after=float(np.mean(list(pub.values())) + tot), macro_delta=float(tot),
                macro_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n_cells_moved=len(picks),
                guardrail_losing_cells=[c for c in CELLS if per[c]["sign"] == "LOSS"],
                macro8_flopeq=float(np.mean([cost_per_cell.get(c, 1.0) for c in CELLS])))

BASE_COST = {c: 1.0 for c in CELLS}
BASE_COST.update({"SLAKE_open": 2.369969, "VQA_RAD_open": 2.369969, "PATH_VQA_open": 2.369969})
R["POLICIES"] = {
 "P0_baseline_as_published": dict(macro=0.5971, macro8_flopeq=float(np.mean(list(BASE_COST.values()))),
      note="the three open cells of this baseline are self-consistency@8, not greedy (verified PART C)"),
 "P1_prompt_fix_only": macro_policy("PathVQA prompt override only",
      {"PATH_VQA_closed": "prompt_open_instruction"}, BASE_COST),
 "P2_prompt_fix_plus_true_greedy_open": macro_policy(
      "PathVQA prompt override + delete the baseline's own self-consistency on the 3 open cells",
      {"PATH_VQA_closed": "prompt_open_instruction",
       "SLAKE_open": "delete_baseline_self_consistency",
       "VQA_RAD_open": "delete_baseline_self_consistency",
       "PATH_VQA_open": "delete_baseline_self_consistency"}, {c: 1.0 for c in CELLS}),
 "P3_add_the_PMC_answer_prior_NOT_CLAIMED": macro_policy(
      "P2 + PMC output-side pm_train (fails the balanced key -- reported, not claimed)",
      {"PATH_VQA_closed": "prompt_open_instruction", "PMC_VQA": "output_pm_train",
       "SLAKE_open": "delete_baseline_self_consistency",
       "VQA_RAD_open": "delete_baseline_self_consistency",
       "PATH_VQA_open": "delete_baseline_self_consistency"}, {c: 1.0 for c in CELLS})}
json.dump(R, open(os.path.join(OUT, "partE.json"), "w"), indent=1)
print(json.dumps({k: R[k] for k in ("PER_CELL_CANDIDATES", "SELECTION_RULE")}, indent=1))
for k, v in R["POLICIES"].items():
    if "macro_delta" in v:
        print(f"{k:48s} macro {v['macro_before']:.6f} -> {v['macro_after']:.6f}  d={v['macro_delta']:+.6f} "
              f"[{v['macro_ci'][0]:+.6f},{v['macro_ci'][1]:+.6f}]  cost={v['macro8_flopeq']:.4f}  "
              f"losing={v['guardrail_losing_cells']}")
    else:
        print(f"{k:48s} macro {v['macro']:.6f} cost={v['macro8_flopeq']:.4f}")
