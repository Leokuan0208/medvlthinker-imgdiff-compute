#!/usr/bin/env python3
"""
twosided_veto.py -- ATTACK C: POST-HOC TWO-SIDED ARBITRATION between the 7B and the 32B.

THE QUESTION.  Sigma p10 = P(7B right AND 32B wrong) = 0.529 over the 8 Variant-B cells, i.e. a
perfectly-identified keep-7B rule would add +0.0661 to the 8-cell macro; the shipped cascade converts
1.3% of it.  The shipped certified veto decides from the 7B's confidence ALONE, BEFORE the 32B runs.
This file asks whether deciding AFTER both models have answered -- using BOTH answers and BOTH
confidence profiles -- converts more.

⚠️ PRIOR ART / REDISCOVERY.  This is substantially a re-run of `beat32b_fusion.py` (F3
"confidence-advantage arbitration" and F5 "double reading"), which already ran the two-sided arbiter
on the 5 MCQ cells and is the source of the shipped `accuracy-max+` fusion lever.  This file (a)
reproduces that prior result exactly as a null test, (b) asks the question the prior work did NOT:
how much of the remaining ceiling is reachable at all from cheap two-sided features, with a HONEST
nested-CV protocol and a richer feature set, and (c) extends arbitration to the 3 open-text cells,
which the prior work never did.

⛔ NOT ABSTENTION (CRITICAL RULE 6).  Every policy here emits one of the two models' answers on every
item.  Nothing defers, rejects, or returns "unknown".

WHAT IS MEASURED, in order:
  PART 0  NULL TESTS: (i) all 8 cells x 7 systems reproduced from vec_disjoint.npz vs
          cascade_selector_rerun_2026-08-05.json; (ii) the frozen open-text bar (sel_eff 0.775204,
          oracle@8 0.626013, greedy 0.449467, per-set 0.8501/0.7619/0.7226); (iii) the PRIOR ART --
          beat32b_fusion.json's F5 double-reading table, reproduced with the prior file's OWN
          normaliser, to prove this is the same experiment before it is extended.
  PART 1  CEILING.  Per cell: the 2x2 contingency, the disagreement set, the DECISIVE disagreement
          set, the base rate pi = p10/(p10+p01), and the exact oracle-arbitration gain (= p10).
          Plus the ceiling restricted to the items the deployed compute-lean gate already escalates
          (where arbitration is FREE), for the MCQ cells.
  PART 2  IDENTIFIABILITY.  On the decisive disagreement set, AUROC of every cheap signal at ranking
          "the 7B is the right one".  One-sided (7B-only) vs two-sided (7B+32B).  Bootstrap CIs.
  PART 3  HONEST REALISED GAIN.  Nested 5x5 cross-validation (inner folds pick the arm, outer folds
          score it), 10 fold seeds, per cell and macro, paired item bootstrap nboot=10000.  Arms:
          take-32B (null), the prior art's calibrated confidence-advantage, 1-sided logistic,
          2-sided logistic, 2-sided gradient boosting.  Plus a LEAVE-ONE-CELL-OUT transfer arm with
          zero within-cell fitting, and the in-sample (oracle-threshold, oracle-arm) upper bound.
  PART 4  A MEASUREMENT DEFECT found while doing PART 1, which any PMC-VQA claim inherits, and the
          decomposition of every arbiter gain into "the two models emitted the same answer" (a
          scoring artifact) versus "the two models genuinely disagreed".
  PART 5  Cost accounting and the verdict against always-32B-direct on the 8-cell macro.

NUMERICS (landmine list, item 8).  Pure numpy float64 + scikit-learn on CPU.  OMP/MKL/OpenBLAS
threads are pinned to 1 BEFORE numpy is imported; no torch, so TF32 does not apply.  Feature rows are
built in the cells' own dump order and never re-sorted.  Every estimator has a fixed random_state.

Launch from the repo root:   python3 src/cascade_methods/twosided_veto.py
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import integrated_method as IM      # noqa: E402
import integrated_pandora as IP     # noqa: E402

ROOT = IM.ROOT
MEK = IM.MEK
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
VEC = os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz")
RERUN = os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")
FUSION = os.path.join(ART, "beat32b_fusion.json")
OUT = os.path.join(ART, "twosided_veto_2026-08-11.json")

DISJOINT = "ckpts/train/lora_verifier_disjoint"

ORDER = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
MCQ, OPEN = ORDER[:5], ORDER[5:]
OPEN_KEY = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}
SYSTEMS = ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b",
           "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]

NBOOT = 10000
NBOOT_AUROC = 2000
SEED = 20260811
NSEED = 10
KOUT = KIN = 5
GEN7_F, GEN32_F, BO8_F, VER_F = 1.0, 4.57, 16.0, 1.0   # paper_baselines / integrated_method constants
GEN7_MS, GEN32_MS, BO8_MS = 347.0, 665.0, 522.0

R = lambda x, k=4: (None if x is None else round(float(x), k))
_t0 = time.time()


def log(*a):
    print(f"[{time.time() - _t0:7.1f}s]", *a, flush=True)


# =====================================================================================================
# helpers
# =====================================================================================================
def as_ok(r):
    v = r.get("correct")
    return int(v is True or str(v).strip().lower() in ("true", "1"))


def as_float(x, d=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def npred_legacy(r):
    """The normaliser `beat32b_fusion.py`/`integrated_method.py` use.  Kept ONLY to reproduce the prior
    art bit-for-bit.  It strips every non-[a-z0-9] character, so a CJK response collapses to the empty
    string and every Chinese SLAKE item is scored as an AGREEMENT -- see PART 4 defect D2."""
    return re.sub(r"[^a-z0-9]", "", str(r.get("response", "")).strip().lower())


def npred_u(s):
    """Unicode-safe answer normaliser: keep word characters (letters incl. CJK, digits), drop
    punctuation and whitespace."""
    return re.sub(r"\W", "", str(s).strip().lower(), flags=re.UNICODE)


def auroc(score, y):
    """Rank AUROC with tie handling (identical implementation to IM.auroc / headroom_percell.auroc)."""
    score = np.asarray(score, float)
    y = np.asarray(y, int)
    P, N = score[y == 1], score[y == 0]
    if len(P) == 0 or len(N) == 0:
        return None
    a = np.concatenate([P, N])
    o = a.argsort()
    rk = np.empty(len(a))
    rk[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    ss = np.zeros(len(c))
    np.add.at(ss, inv, rk)
    rk = (ss / c)[inv]
    return float((rk[:len(P)].sum() - len(P) * (len(P) + 1) / 2) / (len(P) * len(N)))


def boot_mean_ci(d, rng, nboot=NBOOT, chunk=200):
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return dict(mean=None, lo=None, hi=None, sig=None, n=0)
    b = np.empty(nboot)
    for s in range(0, nboot, chunk):
        m = min(chunk, nboot - s)
        b[s:s + m] = d[rng.integers(0, n, size=(m, n))].mean(axis=1)
    lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    return dict(mean=R(d.mean()), lo=R(lo), hi=R(hi), sig=bool(lo > 0 or hi < 0), n=int(n))


def boot_macro_ci(per_cell_gain, rng, nboot=NBOOT, chunk=200, cells=None):
    """Paired item-level bootstrap WITHIN each cell, macro = equal weight over cells."""
    cells = cells or ORDER
    acc = np.zeros(nboot)
    for c in cells:
        d = np.asarray(per_cell_gain[c], float)
        n = len(d)
        b = np.empty(nboot)
        for s in range(0, nboot, chunk):
            m = min(chunk, nboot - s)
            b[s:s + m] = d[rng.integers(0, n, size=(m, n))].mean(axis=1)
        acc += b
    acc /= len(cells)
    lo, hi = float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))
    mean = float(np.mean([np.mean(per_cell_gain[c]) for c in cells]))
    return dict(mean=R(mean), lo=R(lo), hi=R(hi), sig=bool(lo > 0 or hi < 0))


def boot_auroc_ci(score, y, rng, nboot=NBOOT_AUROC):
    score, y = np.asarray(score, float), np.asarray(y, int)
    n = len(y)
    if n < 20 or y.sum() in (0, n):
        return dict(auroc=auroc(score, y), lo=None, hi=None, n=int(n), note="too few items for a CI")
    vals = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, len(i)):
            continue
        vals.append(auroc(score[i], y[i]))
    vals = np.asarray(vals, float)
    return dict(auroc=R(auroc(score, y)), lo=R(np.percentile(vals, 2.5)),
                hi=R(np.percentile(vals, 97.5)), n=int(n),
                excludes_chance=bool(np.percentile(vals, 2.5) > 0.5 or np.percentile(vals, 97.5) < 0.5))


def stable_hash(s):
    """Deterministic across processes (Python's builtin str hash is salted by PYTHONHASHSEED)."""
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


# =====================================================================================================
# PART 0a.  build the 8 cells: ok vectors, answer strings, cheap two-sided features
# =====================================================================================================
def mcq_cell(cell, ds, closed):
    r7 = IM.load_raw("lingshu7b_full", ds)
    r32 = IM.load_raw("lingshu32b_full", ds)
    r7c = IM.load_raw("lingshu7b_cap320", ds)
    n = min(len(r7), len(r32))
    if closed == "SLAKE":
        idx = [i for i in range(n) if r7[i].get("answer_type") == "CLOSED"]
    elif closed == "YESNO":
        idx = [i for i in range(n) if str(r7[i].get("answer", "")).strip().lower() in ("yes", "no")]
    else:
        idx = list(range(n))
    g = lambda rs, k: np.array([as_float(rs[i].get(k)) for i in idx], float)
    ok7 = np.array([as_ok(r7[i]) for i in idx], float)
    ok32 = np.array([as_ok(r32[i]) for i in idx], float)
    p7 = np.array([npred_u(r7[i].get("response", "")) for i in idx], object)
    p32 = np.array([npred_u(r32[i].get("response", "")) for i in idx], object)
    p7_legacy = np.array([npred_legacy(r7[i]) for i in idx], object)
    p32_legacy = np.array([npred_legacy(r32[i]) for i in idx], object)

    c7, m7, c32, m32 = g(r7, "conf"), g(r7, "margin"), g(r32, "conf"), g(r32, "margin")
    F = {
        "7B|conf": c7, "7B|margin": m7, "7B|logit_conf": logit(c7),
        "7B|cum_logprob": g(r7, "cum_logprob"), "7B|gen_toks": g(r7, "gen_toks"),
        "32B|conf": c32, "32B|margin": m32, "32B|logit_conf": logit(c32),
        "32B|cum_logprob": g(r32, "cum_logprob"), "32B|gen_toks": g(r32, "gen_toks"),
        "X|d_conf": c7 - c32, "X|d_margin": m7 - m32, "X|d_logit_conf": logit(c7) - logit(c32),
        "X|d_cum_logprob": g(r7, "cum_logprob") - g(r32, "cum_logprob"),
    }
    if r7c is not None and len(r7c) >= n:
        F["7B|selfconsistency_cap320"] = np.array(
            [1.0 if npred_u(r7[i].get("response", "")) == npred_u(r7c[i].get("response", "")) else 0.0
             for i in idx], float)
    return dict(ok7=ok7, ok32=ok32, p7=p7, p32=p32, p7_legacy=p7_legacy, p32_legacy=p32_legacy,
                feats=F, gate7=m7, raw7=[r7[i] for i in idx], raw32=[r32[i] for i in idx],
                fmt="MCQ/closed")


def open_cell(cell, variant):
    """variant='greedy'  -> the 7B side is ONE greedy pass (cost 1.0).  This is what the canonical macro
                            table calls `always_7b` (0.7364 / 0.4650 / 0.3240) and what the stated p10
                            ceiling (0.0465 / 0.0500 / 0.0760) was computed against.  PRIMARY.
       variant='bo8'     -> the 7B side is best-of-8 + the clean disjoint verifier's pick (cost 16.0),
                            i.e. the shipped cheap arm (0.7473 / 0.4800 / 0.3733).  SECONDARY: a higher
                            ceiling bought with 16x the cheap-leg compute."""
    key = OPEN_KEY[cell]
    IM.OPEN_VERIFIER_DIR = DISJOINT
    IP.ADAPTER = DISJOINT
    rows = IP.load_open_rows(key)
    dp = json.load(open(os.path.join(ROOT, DISJOINT, f"transfer_dump_{key}_lingshu7b.json")))
    by_idx = {r["idx"]: r for r in dp}

    def load_gen(path):
        m = {}
        for l in open(os.path.join(ROOT, path)):
            if l.strip():
                r = json.loads(l)
                m[int(r["idx"])] = r
        return m

    strong = load_gen(f"ckpts/openvqa/strong_lingshu/ckpt_{key}_lingshu32b.jsonl")
    cheap = load_gen(f"ckpts/openvqa/cheap_lingshu7b/ckpt_{key}_lingshu7b.jsonl")

    sc = np.array([r["scores"][:8] for r in rows], float)
    sl = np.array([r["sl"][:8] for r in rows], float)
    pick = sc.argmax(1)
    ok32 = np.array([r["strong"] for r in rows], float)
    greedy = np.array([r["greedy"] for r in rows], float)
    ok7 = sl[np.arange(len(rows)), pick] if variant == "bo8" else greedy

    p7, p32, sc32, gt32, len7, len32, agree8, scg, gtg = [], [], [], [], [], [], [], [], []
    for j, r in enumerate(rows):
        d = by_idx[r["idx"]]
        preds = [str(x) for x in d.get("preds", [])[:8]]
        s = strong.get(r["idx"], {})
        cg = cheap.get(r["idx"], {})
        mp = str(s.get("modal_pred", ""))
        a7 = (preds[pick[j]] if len(preds) == 8 else "") if variant == "bo8" else str(cg.get("modal_pred", ""))
        p7.append(npred_u(a7))
        p32.append(npred_u(mp))
        sc32.append(as_float(s.get("seqlogprob"), -1.0))
        gt32.append(as_float(s.get("gen_tokens"), 0.0))
        scg.append(as_float(cg.get("seqlogprob"), -1.0))
        gtg.append(as_float(cg.get("gen_tokens"), 0.0))
        len7.append(float(len(a7)))
        len32.append(float(len(mp)))
        nrm = [npred_u(x) for x in preds]
        agree8.append(float(np.unique(nrm, return_counts=True)[1].max()) / 8.0 if len(nrm) == 8 else np.nan)

    len7, len32 = np.asarray(len7, float), np.asarray(len32, float)
    sc32, gt32 = np.asarray(sc32, float), np.asarray(gt32, float)
    scg, gtg = np.asarray(scg, float), np.asarray(gtg, float)
    F = {"32B|seqlogprob": sc32, "32B|gen_tokens": gt32, "32B|answer_len": len32,
         "X|d_len": len7 - len32}
    if variant == "bo8":
        srt = np.sort(sc, axis=1)
        F.update({"7B|verifier_max": sc.max(1), "7B|verifier_mean": sc.mean(1),
                  "7B|verifier_std": sc.std(1), "7B|verifier_gap12": srt[:, -1] - srt[:, -2],
                  "7B|pool_selfconsistency": np.asarray(agree8, float), "7B|pick_len": len7,
                  "X|d_conf_proxy": sc.max(1) - np.exp(np.clip(sc32, -20, 0))})
        gate = sc.max(1)
    else:
        F.update({"7B|seqlogprob": scg, "7B|gen_tokens": gtg, "7B|answer_len": len7,
                  "X|d_seqlogprob": scg - sc32,
                  "X|d_conf_proxy": np.exp(np.clip(scg, -20, 0)) - np.exp(np.clip(sc32, -20, 0))})
        gate = scg
    return dict(ok7=ok7, ok32=ok32, p7=np.array(p7, object), p32=np.array(p32, object),
                p7_legacy=np.array(p7, object), p32_legacy=np.array(p32, object),
                feats=F, gate7=gate, raw7=None, raw32=None, fmt="open-text", variant=variant,
                greedy=greedy, oracle8=sl.max(1), selected=sl[np.arange(len(rows)), pick])


def build_cells():
    C = {}
    for cell, ds, cl in [("PMC_VQA", "PMC_VQA", None), ("SLAKE_closed", "SLAKE", "SLAKE"),
                         ("VQA_RAD_closed", "VQA_RAD", "YESNO"), ("PATH_VQA_closed", "PATH_VQA", "YESNO")]:
        C[cell] = mcq_cell(cell, ds, cl)
    r7 = IM.load_raw("lingshu7b_full", "MedXpertQA-MM")
    r32 = IM.load_raw("lingshu32b_full", "MedXpertQA-MM")
    n = min(len(r7), len(r32))
    idx = list(range(n))
    g = lambda rs, k: np.array([as_float(rs[i].get(k)) for i in idx], float)
    c7, c32 = g(r7, "conf"), g(r32, "conf")
    C["MedXpertQA-MM"] = dict(
        ok7=np.array([as_ok(r7[i]) for i in idx], float),
        ok32=np.array([as_ok(r32[i]) for i in idx], float),
        p7=np.array([npred_u(r7[i].get("response", "")) for i in idx], object),
        p32=np.array([npred_u(r32[i].get("response", "")) for i in idx], object),
        p7_legacy=np.array([npred_legacy(r7[i]) for i in idx], object),
        p32_legacy=np.array([npred_legacy(r32[i]) for i in idx], object),
        feats={"7B|conf": c7, "7B|margin": g(r7, "margin"), "7B|logit_conf": logit(c7),
               "7B|cum_logprob": g(r7, "cum_logprob"), "7B|gen_toks": g(r7, "gen_toks"),
               "32B|conf": c32, "32B|margin": g(r32, "margin"), "32B|logit_conf": logit(c32),
               "32B|cum_logprob": g(r32, "cum_logprob"), "32B|gen_toks": g(r32, "gen_toks"),
               "X|d_conf": c7 - c32, "X|d_margin": g(r7, "margin") - g(r32, "margin"),
               "X|d_logit_conf": logit(c7) - logit(c32),
               "X|d_cum_logprob": g(r7, "cum_logprob") - g(r32, "cum_logprob")},
        gate7=g(r7, "margin"), raw7=[r7[i] for i in idx], raw32=[r32[i] for i in idx], fmt="MCQ/closed")
    for cell in OPEN:
        C[cell] = open_cell(cell, "greedy")
    return C


# =====================================================================================================
# PART 0b.  null tests
# =====================================================================================================
def null_tests(C):
    z = np.load(VEC)
    V = {c: {s: np.asarray(z[f"{c}|{s}"], float) for s in SYSTEMS} for c in ORDER}
    art = json.load(open(RERUN))["per_arm"]["disjoint"]

    dev, ncmp = 0.0, 0
    for c in ORDER:
        for s, v in art["per_cell_acc"][c].items():
            dev = max(dev, abs(round(float(V[c][s].mean()), 4) - v)); ncmp += 1
    for s in SYSTEMS:
        m = float(np.mean([V[c][s].mean() for c in ORDER]))
        dev = max(dev, abs(round(m, 4) - art["macro_acc"][s])); ncmp += 1
    nt1 = dict(source=os.path.relpath(RERUN, ROOT), arm="disjoint", fields_compared=ncmp,
               max_abs_deviation=R(dev, 10), passed=bool(dev == 0.0))

    # my freshly-built cells must equal the stored per-arm vectors item for item
    dev2, ncmp2 = 0.0, 0
    for c in ORDER:
        assert len(C[c]["ok7"]) == len(V[c]["always_7b"]), c
        dev2 = max(dev2, float(np.abs(C[c]["ok7"] - V[c]["always_7b"]).max()))
        dev2 = max(dev2, float(np.abs(C[c]["ok32"] - V[c]["always_32b_direct"]).max()))
        ncmp2 += 2 * len(C[c]["ok7"])
    nt2 = dict(what="this file's freshly-rebuilt ok7 / ok32 vectors vs vec_disjoint.npz, item by item",
               items_compared=int(ncmp2), max_abs_deviation=R(dev2, 10), passed=bool(dev2 == 0.0))

    G = np.concatenate([C[c]["greedy"] for c in OPEN])
    O = np.concatenate([C[c]["oracle8"] for c in OPEN])
    S = np.concatenate([C[c]["selected"] for c in OPEN])
    frozen = dict(oracle8=0.626013, greedy=0.449467, selected=0.485288, sel_eff=0.775204,
                  per_ds={"slake_open": 0.850088, "vqa_rad_open": 0.761905, "pathvqa_open": 0.722581})
    got = dict(oracle8=float(O.mean()), greedy=float(G.mean()), selected=float(S.mean()),
               sel_eff=float(S[O == 1].mean()))
    per_ds = {OPEN_KEY[c]: float(C[c]["selected"][C[c]["oracle8"] == 1].mean()) for c in OPEN}
    d3 = max([abs(got[k] - frozen[k]) for k in got] + [abs(per_ds[k] - frozen["per_ds"][k]) for k in per_ds])
    nt3 = dict(source="src/training_methods/genframe_data.py FROZEN block (incumbent, clean disjoint)",
               n=int(len(G)), measured={k: R(v, 6) for k, v in got.items()},
               measured_per_ds_sel_eff={k: R(v, 6) for k, v in per_ds.items()},
               frozen=frozen, max_abs_deviation=R(d3, 9), passed=bool(d3 < 1e-6))

    # ---- PRIOR ART: reproduce beat32b_fusion.json's F5 double-reading table with ITS OWN normaliser
    fu = json.load(open(FUSION))["idea_findings"]["F5_double_reading"]
    rep, dev4, n4 = {}, 0.0, 0
    for cell in MCQ:
        pub = fu.get(cell)
        if pub is None:
            continue
        d = C[cell]
        dis = np.array([a != b for a, b in zip(d["p7_legacy"], d["p32_legacy"])])
        ok7, ok32 = d["ok7"], d["ok32"]
        c7 = d["feats"]["7B|conf"]; c32 = d["feats"]["32B|conf"]
        # the prior file's in-sample isotonic calibration for the AUROC row
        pr7 = np.clip(IsotonicRegression(out_of_bounds="clip").fit(c7, ok7).predict(c7), 1e-6, 1 - 1e-6)
        pr32 = np.clip(IsotonicRegression(out_of_bounds="clip").fit(c32, ok32).predict(c32), 1e-6, 1 - 1e-6)
        res = dis & (ok7 != ok32)
        got_ = dict(agree_rate=float((~dis).mean()),
                    acc_on_agree=float(ok32[~dis].mean()),
                    disagree_rate=float(dis.mean()),
                    oracle_UB_on_disagree=float(np.maximum(ok7, ok32)[dis].mean()),
                    take_32b_nt_on_disagree=float(ok32[dis].mean()),
                    recoverability_auroc_confadv=(auroc((pr7 - pr32)[res], ok7[res]) if res.sum() > 5 else None))
        pubv = dict(agree_rate=pub["agree_rate"], acc_on_agree=pub["acc_on_agree"],
                    disagree_rate=pub["disagree_rate"],
                    oracle_UB_on_disagree=pub["arbiter_acc_on_disagree"]["oracle_UB"],
                    take_32b_nt_on_disagree=pub["arbiter_acc_on_disagree"]["take_32b_nt"],
                    recoverability_auroc_confadv=pub.get("recoverability_auroc_confadv"))
        for k in got_:
            if got_[k] is not None and pubv[k] is not None:
                dev4 = max(dev4, abs(round(got_[k], 4) - pubv[k])); n4 += 1
        rep[cell] = dict(reproduced={k: R(v) for k, v in got_.items()}, published=pubv)
    nt4 = dict(source=os.path.relpath(FUSION, ROOT),
               what="PRIOR ART reproduction: beat32b_fusion.py F5 double-reading, using that file's own "
                    "[^a-z0-9] normaliser, to establish that ATTACK C is the same experiment before it "
                    "is extended",
               fields_compared=n4, max_abs_deviation=R(dev4, 6), passed=bool(dev4 <= 1e-4), per_cell=rep)
    return dict(cascade_rerun=nt1, rebuilt_vectors=nt2, open_text_bar=nt3, prior_art_F5=nt4), V


# =====================================================================================================
# PART 1.  the ceiling of post-hoc arbitration
# =====================================================================================================
def crossfit_escalation_mask(ok7, ok32, gate, K=5):
    """The deployed compute-lean MCQ gate, per item (copy of paper_baselines.cascade_persample's rule,
    returning the MASK instead of only its rate).  tau fit on 4/5 to reach the strong leg's accuracy at
    minimum escalation, applied to the held-out 1/5."""
    n = len(ok7)
    esc = np.zeros(n, bool)
    for f in range(K):
        te = np.arange(n) % K == f
        tr = ~te
        if tr.sum() < 2 or te.sum() < 1:
            continue
        tau = IM.pick_tau_isocost(ok7[tr], ok32[tr], gate[tr], ok32[tr].mean())
        esc[te] = gate[te] < tau
    return esc


def part1_ceiling(C, rng, cells=None):
    cells = cells or ORDER
    out = {}
    for c in cells:
        d = C[c]
        ok7, ok32 = d["ok7"], d["ok32"]
        n = len(ok7)
        dis = np.array([a != b for a, b in zip(d["p7"], d["p32"])])
        dec = ok7 != ok32
        p11 = float(((ok7 == 1) & (ok32 == 1)).mean())
        p10 = float(((ok7 == 1) & (ok32 == 0)).mean())
        p01 = float(((ok7 == 0) & (ok32 == 1)).mean())
        p00 = float(((ok7 == 0) & (ok32 == 0)).mean())
        row = dict(
            n=int(n), format=d["fmt"],
            acc=dict(always_7b=R(ok7.mean()), always_32b_direct=R(ok32.mean())),
            contingency=dict(both_right=R(p11), only_7B_right=R(p10), only_32B_right=R(p01),
                             both_wrong=R(p00)),
            answer_disagreement_rate=R(float(dis.mean())),
            decisive_rate=R(float(dec.mean())),
            n_decisive=int(dec.sum()),
            base_rate_pi_7B_is_the_right_one=R(p10 / (p10 + p01)) if (p10 + p01) > 0 else None,
            oracle_arbitration_gain=boot_mean_ci(np.where(dec, ok7 - ok32, 0.0).clip(0, 1), rng),
            consistency=dict(
                decisive_but_answers_string_identical=R(float((dec & ~dis).mean())),
                n_decisive_but_string_identical=int((dec & ~dis).sum()),
                note="items scored differently although both models emitted the SAME normalised answer "
                     "string; a gain harvested there is a SCORING artifact, not accuracy -- see PART 4"),
        )
        if c in MCQ:
            esc = crossfit_escalation_mask(ok7, ok32, d["gate7"])
            row["free_on_deployed_escalated_set"] = dict(
                escalation_rate=R(float(esc.mean())),
                ceiling_gain_if_arbitration_only_where_both_already_ran=R(
                    float(((ok7 == 1) & (ok32 == 0) & esc).mean())),
                note="the compute-lean gate's own cross-fit escalation mask; on these items both models "
                     "have already run, so arbitration is FREE.  On the non-escalated items the cascade "
                     "already keeps the 7B, so no arbitration is possible or needed there.")
        out[c] = row
    macro_ceiling = float(np.mean([out[c]["contingency"]["only_7B_right"] for c in cells]))
    return out, macro_ceiling


# =====================================================================================================
# PART 2.  identifiability on the decisive disagreement set
# =====================================================================================================
def part2_identifiability(C, rng, cells=None):
    cells = cells or ORDER
    out = {}
    for c in cells:
        d = C[c]
        ok7, ok32 = d["ok7"], d["ok32"]
        dec = ok7 != ok32
        y = ok7[dec].astype(int)
        if dec.sum() < 20:
            out[c] = dict(n_decisive=int(dec.sum()), note="too few decisive items to estimate AUROC")
            continue
        rows = {}
        for name, v in sorted(d["feats"].items()):
            s = v[dec]
            if float(np.nanstd(s)) == 0.0 or not np.isfinite(s).all():
                continue
            rows[name] = boot_auroc_ci(s, y, np.random.default_rng(SEED + stable_hash(c + name) % 10000))
        # the prior art's signal: cross-fit isotonic calibrated P(correct) advantage
        if c in MCQ:
            c7, c32 = d["feats"]["7B|conf"], d["feats"]["32B|conf"]
            adv = np.zeros(len(ok7))
            n = len(ok7)
            for f in range(KOUT):
                te = np.arange(n) % KOUT == f
                tr = ~te
                a = np.clip(IsotonicRegression(out_of_bounds="clip").fit(c7[tr], ok7[tr]).predict(c7[te]), 1e-6, 1 - 1e-6)
                b = np.clip(IsotonicRegression(out_of_bounds="clip").fit(c32[tr], ok32[tr]).predict(c32[te]), 1e-6, 1 - 1e-6)
                adv[te] = a - b
            rows["PRIOR_ART|calibrated_conf_advantage"] = boot_auroc_ci(
                adv[dec], y, np.random.default_rng(SEED + 7))
        best1 = max([(v["auroc"], k) for k, v in rows.items()
                     if v["auroc"] is not None and k.startswith("7B|")], default=(None, None))
        best2 = max([(v["auroc"], k) for k, v in rows.items()
                     if v["auroc"] is not None and (k.startswith("32B|") or k.startswith("X|")
                                                    or k.startswith("PRIOR"))], default=(None, None))
        out[c] = dict(n_decisive=int(dec.sum()), base_rate_pi=R(float(y.mean())),
                      auroc_per_signal={k: rows[k] for k in sorted(rows)},
                      best_one_sided_7B_only=dict(signal=best1[1], auroc=R(best1[0])),
                      best_signal_using_the_32B_side=dict(signal=best2[1], auroc=R(best2[0])),
                      two_sided_advantage_auroc=R((best2[0] - best1[0]) if (best1[0] and best2[0]) else None))
    return out


# =====================================================================================================
# PART 3.  honest realised gain -- nested cross-validation over arbiter arms
# =====================================================================================================
def _mat(d, names):
    return np.column_stack([d["feats"][n] for n in names])


def _feat_names(d, sided):
    ks = sorted(d["feats"])
    if sided == "1sided":
        return [k for k in ks if k.startswith("7B|")]
    return ks


class Arm:
    """An arbiter arm: fit on (X, y) over DECISIVE items, decide take-7B on ALL disagreement items."""

    def __init__(self, name, kind, sided, thr_mode):
        self.name, self.kind, self.sided, self.thr_mode = name, kind, sided, thr_mode

    def fit_predict(self, d, tr, te, dec, dis):
        """tr/te are boolean item masks.  Returns a boolean take-7B decision on the te items."""
        take = np.zeros(int(te.sum()), bool)
        if self.kind == "null":
            return take
        names = _feat_names(d, self.sided)
        X = _mat(d, names)
        ok7, ok32 = d["ok7"], d["ok32"]
        ftr = tr & dec
        if ftr.sum() < 20 or len(np.unique(ok7[ftr])) < 2:
            return take
        mu, sd = X[ftr].mean(0), X[ftr].std(0)
        sd[sd == 0] = 1.0
        Xtr, Xte = (X[ftr] - mu) / sd, (X[te] - mu) / sd
        ytr = ok7[ftr].astype(int)
        if self.kind == "logistic":
            m = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=0)
        elif self.kind == "gbm":
            m = HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.1,
                                               early_stopping=False, random_state=0)
        elif self.kind == "confadv":
            if "32B|conf" not in d["feats"]:
                return take
            c7, c32 = d["feats"]["7B|conf"], d["feats"]["32B|conf"]
            a = np.clip(IsotonicRegression(out_of_bounds="clip").fit(c7[tr], ok7[tr]).predict(c7[te]), 1e-6, 1 - 1e-6)
            b = np.clip(IsotonicRegression(out_of_bounds="clip").fit(c32[tr], ok32[tr]).predict(c32[te]), 1e-6, 1 - 1e-6)
            return (a > b) & dis[te]
        elif self.kind == "verifadv":
            # open-text analogue of the prior art: the 7B side's own confidence (the verifier's max
            # score for the best-of-8 arm, the greedy sequence log-prob for the greedy arm), isotonically
            # calibrated, against the 32B's isotonically calibrated sequence log-prob
            v7 = d["feats"].get("7B|verifier_max", d["feats"].get("7B|seqlogprob"))
            v32 = d["feats"]["32B|seqlogprob"]
            if v7 is None:
                return take
            a = np.clip(IsotonicRegression(out_of_bounds="clip").fit(v7[tr], ok7[tr]).predict(v7[te]), 1e-6, 1 - 1e-6)
            b = np.clip(IsotonicRegression(out_of_bounds="clip").fit(v32[tr], ok32[tr]).predict(v32[te]), 1e-6, 1 - 1e-6)
            return (a > b) & dis[te]
        else:
            raise ValueError(self.kind)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        if self.thr_mode == "bayes":
            thr = 0.5
        else:
            ptr = m.predict_proba(Xtr)[:, 1]
            dtr = (ok7 - ok32)[ftr]
            o = np.argsort(-ptr, kind="stable")
            cs = np.cumsum(dtr[o])
            k = int(np.argmax(cs))
            thr = ptr[o][k] if cs[k] > 0 else 2.0
        return (p >= thr) & dis[te]


ARMS = [Arm("take_32B_always", "null", None, None),
        Arm("prior_art_calibrated_conf_advantage", "confadv", None, None),
        Arm("logistic_1sided_7B_only", "logistic", "1sided", "bayes"),
        Arm("logistic_2sided", "logistic", "2sided", "bayes"),
        Arm("logistic_2sided_tuned_threshold", "logistic", "2sided", "tuned"),
        Arm("gbm_2sided", "gbm", "2sided", "bayes"),
        Arm("gbm_2sided_tuned_threshold", "gbm", "2sided", "tuned")]
OPEN_ARMS = [Arm("take_32B_always", "null", None, None),
             Arm("prior_art_calibrated_conf_advantage", "verifadv", None, None),
             Arm("logistic_1sided_7B_only", "logistic", "1sided", "bayes"),
             Arm("logistic_2sided", "logistic", "2sided", "bayes"),
             Arm("logistic_2sided_tuned_threshold", "logistic", "2sided", "tuned"),
             Arm("gbm_2sided", "gbm", "2sided", "bayes"),
             Arm("gbm_2sided_tuned_threshold", "gbm", "2sided", "tuned")]


def gain_vector(d, take, mask):
    """Per-item gain over always-32B-direct, for the items in `mask` given take-7B decisions."""
    g = np.zeros(len(d["ok7"]))
    idx = np.where(mask)[0]
    g[idx] = np.where(take, d["ok7"][idx] - d["ok32"][idx], 0.0)
    return g


def nested_cv_cell(d, arms, seed):
    """Outer 5-fold scores; inner 5-fold (inside each outer TRAIN) picks the arm.  Returns the honest
    per-item gain vector and the per-outer-fold arm choices."""
    n = len(d["ok7"])
    ok7, ok32 = d["ok7"], d["ok32"]
    dec = ok7 != ok32
    dis = np.array([a != b for a, b in zip(d["p7"], d["p32"])])
    perm = np.random.default_rng(seed).permutation(n)
    fold = np.empty(n, int)
    fold[perm] = np.arange(n) % KOUT
    g = np.zeros(n)
    picks, single = [], {a.name: np.zeros(n) for a in arms}
    for f in range(KOUT):
        te = fold == f
        tr = ~te
        # -- inner CV on the outer-train, to pick the arm
        itr_idx = np.where(tr)[0]
        iperm = np.random.default_rng(seed * 131 + f).permutation(len(itr_idx))
        ifold = np.empty(len(itr_idx), int)
        ifold[iperm] = np.arange(len(itr_idx)) % KIN
        scores = {}
        for a in arms:
            tot = 0.0
            for h in range(KIN):
                ite = np.zeros(n, bool); ite[itr_idx[ifold == h]] = True
                itr = np.zeros(n, bool); itr[itr_idx[ifold != h]] = True
                take = a.fit_predict(d, itr, ite, dec, dis)
                tot += float(np.where(take, (ok7 - ok32)[ite], 0.0).sum())
            scores[a.name] = tot / max(1, tr.sum())
        best = max(arms, key=lambda a: (scores[a.name], a.name == "take_32B_always"))
        if scores[best.name] <= 0:
            best = arms[0]
        picks.append(best.name)
        take = best.fit_predict(d, tr, te, dec, dis)
        g += gain_vector(d, take, te)
        for a in arms:                                   # per-arm (non-nested) reference
            single[a.name] += gain_vector(d, a.fit_predict(d, tr, te, dec, dis), te)
    return g, picks, single


def insample_upper_bound(d, arms):
    """OPTIMISTIC upper bound: every arm fit AND scored on the whole cell, best arm kept.  This is the
    most any of these cheap features could give if the fit did not have to generalise.  DIAGNOSTIC."""
    n = len(d["ok7"])
    dec = d["ok7"] != d["ok32"]
    dis = np.array([a != b for a, b in zip(d["p7"], d["p32"])])
    allm = np.ones(n, bool)
    best = ("take_32B_always", 0.0)
    for a in arms:
        take = a.fit_predict(d, allm, allm, dec, dis)
        v = float(gain_vector(d, take, allm).mean())
        if v > best[1]:
            best = (a.name, v)
    return dict(arm=best[0], gain=R(best[1]))


def loco_transfer(C, cells, arms_kind="2sided"):
    """LEAVE-ONE-CELL-OUT: train one arbiter on the OTHER cells of the same format and apply it to the
    held-out cell with ZERO within-cell fitting.  The strictest honesty check available here."""
    out = {}
    common = None
    for c in cells:
        ks = set(k for k in C[c]["feats"] if not k.startswith("7B|selfconsistency"))
        common = ks if common is None else (common & ks)
    common = sorted(common)
    for c in cells:
        others = [o for o in cells if o != c]
        Xtr = np.vstack([np.column_stack([C[o]["feats"][k] for k in common])[C[o]["ok7"] != C[o]["ok32"]]
                         for o in others])
        ytr = np.concatenate([C[o]["ok7"][C[o]["ok7"] != C[o]["ok32"]] for o in others]).astype(int)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        m = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0).fit((Xtr - mu) / sd, ytr)
        d = C[c]
        Xte = (np.column_stack([d["feats"][k] for k in common]) - mu) / sd
        dis = np.array([a != b for a, b in zip(d["p7"], d["p32"])])
        take = (m.predict_proba(Xte)[:, 1] >= 0.5) & dis
        g = np.where(take, d["ok7"] - d["ok32"], 0.0)
        out[c] = dict(gain=R(float(g.mean())), take7_rate=R(float(take.mean())),
                      n_train_items=int(len(ytr)), features=common, per_item=g)
    return out


def part3(C, rng, cells=None):
    cells = cells or ORDER
    per_seed_cell = {c: [] for c in cells}
    per_seed_macro, picks_all, singles = [], {c: {} for c in cells}, {c: {} for c in cells}
    gbar = {c: np.zeros(len(C[c]["ok7"])) for c in cells}
    per_seed_vectors = []
    for s in range(NSEED):
        seed = SEED + 1000 * s
        mac = 0.0
        seed_vec = {}
        for c in cells:
            arms = OPEN_ARMS if c in OPEN else ARMS
            g, picks, single = nested_cv_cell(C[c], arms, seed)
            seed_vec[c] = g
            gbar[c] += g / NSEED
            per_seed_cell[c].append(float(g.mean()))
            mac += float(g.mean())
            for p in picks:
                picks_all[c][p] = picks_all[c].get(p, 0) + 1
            for k, v in single.items():
                singles[c].setdefault(k, []).append(float(v.mean()))
        per_seed_macro.append(mac / len(cells))
        per_seed_vectors.append(seed_vec)
        log(f"  seed {s + 1}/{NSEED} done, macro nested gain {per_seed_macro[-1]:+.5f}")

    per_cell = {}
    for c in cells:
        d = C[c]
        dis = np.array([a != b for a, b in zip(d["p7"], d["p32"])])
        dec = d["ok7"] != d["ok32"]
        gv = gbar[c]
        per_cell[c] = dict(
            nested_gain_seed_mean=R(float(np.mean(per_seed_cell[c])), 5),
            nested_gain_seed_sd=R(float(np.std(per_seed_cell[c])), 5),
            nested_gain_seed_range=[R(float(np.min(per_seed_cell[c])), 5), R(float(np.max(per_seed_cell[c])), 5)],
            ci_on_seed_averaged_per_item_gain=boot_mean_ci(gv, rng),
            arm_chosen_by_inner_cv=dict(sorted(picks_all[c].items(), key=lambda kv: -kv[1])),
            per_arm_crossfit_gain_seed_mean={k: R(float(np.mean(v)), 5) for k, v in sorted(singles[c].items())},
            insample_upper_bound=insample_upper_bound(d, OPEN_ARMS if c in OPEN else ARMS),
            oracle_gain=R(float(((d["ok7"] == 1) & (d["ok32"] == 0)).mean())),
            gain_decomposition=dict(
                from_items_where_answers_genuinely_differ=R(float((gv * dis).mean()), 5),
                from_items_where_answers_are_string_identical=R(float((gv * (~dis)).mean()), 5),
                note="the second column is a SCORING artifact, not accuracy (PART 4)"),
            n_decisive=int(dec.sum()))
    macro = boot_macro_ci(gbar, rng, cells=cells)
    # A DEPLOYED system is ONE fit, not the average of ten.  So also give each individual seed its own
    # paired item bootstrap and count how many of the ten would have been declared a win on their own.
    per_seed_ci = [boot_macro_ci(v, np.random.default_rng(SEED + 31 * i), nboot=2000, cells=cells)
                   for i, v in enumerate(per_seed_vectors)]
    n_sig_pos = sum(1 for x in per_seed_ci if x["lo"] is not None and x["lo"] > 0)
    n_pos = sum(1 for x in per_seed_macro if x > 0)
    return dict(per_cell=per_cell,
                single_seed_reality_check=dict(
                    per_seed_macro_ci=per_seed_ci,
                    seeds_with_a_positive_point_estimate=f"{n_pos}/{NSEED}",
                    seeds_whose_own_95pct_lower_bound_exceeds_zero=f"{n_sig_pos}/{NSEED}",
                    note="the headline CI is a bootstrap over ITEMS of the SEED-AVERAGED per-item gain, "
                         "which removes fit-to-fit variance that a deployed system actually carries; "
                         "this row restores it"), macro=macro, per_seed_macro=[R(x, 5) for x in per_seed_macro],
                per_seed_macro_sd=R(float(np.std(per_seed_macro)), 5)), gbar


# =====================================================================================================
# PART 4.  the measurement defects found on the way
# =====================================================================================================
def part4_defects(C):
    r7 = IM.load_raw("lingshu7b_full", "PMC_VQA")
    r32 = IM.load_raw("lingshu32b_full", "PMC_VQA")

    def lead(x):
        s = str(x.get("response", "")).strip()
        return s[0].upper() if s and s[0].isalpha() else None

    def suffix_colon(x):
        s = str(x.get("response", "")).strip()
        return 1.0 if s[1:2] == ":" else 0.0

    D = {}
    for tag, rs in (("lingshu7b_full", r7), ("lingshu32b_full", r32)):
        mek = np.array([as_ok(x) for x in rs], float)
        strict = np.array([1.0 if lead(x) == x["answer"] else 0.0 for x in rs])
        col = np.array([suffix_colon(x) for x in rs])
        D[tag] = dict(
            n=int(len(rs)),
            rows_whose_first_char_is_not_a_valid_option_letter=int(sum(
                1 for x in rs if not (str(x.get("response", "")).strip()[:1].isalpha()
                                      and str(x.get("response", "")).strip()[:1].upper() in "ABCD"))),
            acc_medevalkit_judge_multi_choice=R(mek.mean()),
            acc_strict_leading_option_letter=R(strict.mean()),
            delta_scorer=R(float(mek.mean() - strict.mean())),
            frac_responses_with_colon_after_the_letter=R(col.mean()),
            acc_on_colon_rows=dict(medevalkit=R(mek[col == 1].mean()), strict=R(strict[col == 1].mean()),
                                   n=int(col.sum())),
            acc_on_dot_rows=dict(medevalkit=R(mek[col == 0].mean()), strict=R(strict[col == 0].mean()),
                                 n=int((col == 0).sum())))
    same_letter = np.array([1.0 if lead(a) == lead(b) else 0.0 for a, b in zip(r7, r32)])
    ok7 = np.array([as_ok(x) for x in r7], float)
    ok32 = np.array([as_ok(x) for x in r32], float)
    art = (same_letter == 1) & (ok7 != ok32)
    f7 = int(((ok7 == 1) & art).sum())
    f32 = int(((ok32 == 1) & art).sum())

    d1 = dict(
        name="D1 -- MedEvalKit judge_multi_choice mis-scores option letters followed by a colon",
        where="MedEvalKit/utils/utils.py:111-112 (READ ONLY, NOT MODIFIED)",
        mechanism="`split_response = response.split('.')[0]` then `split_response.split(':')[-1]`.  For "
                  "'c.' this yields 'c' and matches the gold; for 'c:' it yields the EMPTY STRING, the "
                  "letter branch is missed, and the item falls through to a fuzzy "
                  "find_most_similar_index() over the choice texts.",
        affected_cells=["PMC_VQA"],
        per_model=D,
        items_where_both_models_emit_the_SAME_option_letter_but_are_scored_differently=dict(
            n=int(art.sum()), frac_of_cell=R(float(art.mean())),
            favouring_the_7B=f7, favouring_the_32B=f32,
            net_gift_to_the_7B_on_this_cell=R((f7 - f32) / len(r7), 5)),
        consequence="Every PMC-VQA number in this project inherits this.  PMC-VQA is the ONE cell that "
                    "carries the whole vs-always-32B-direct claim (leave-one-cell-out range "
                    "[-0.0004,+0.0024]).  Under a punctuation-robust scorer the 32B-direct/7B gap on the "
                    "cell WIDENS from +0.0091 to +0.0167, i.e. the canonical scorer hands the 7B "
                    "0.0076 of the cell = +0.00095 of the 8-cell macro -- larger than the entire "
                    "published +0.0008 vs-direct delta.",
        status="REPORTED, NOT ACTED ON.  MedEvalKit is a protected dependency and the paper's protocol "
               "is its scorer; this is a sensitivity that must be stated, not a re-scoring of the paper.")

    d = C["SLAKE_closed"]
    dis_legacy = np.array([a != b for a, b in zip(d["p7_legacy"], d["p32_legacy"])])
    dis_u = np.array([a != b for a, b in zip(d["p7"], d["p32"])])
    d2 = dict(
        name="D2 -- the project's answer normaliser deletes CJK text, so every Chinese SLAKE item is "
             "counted as a model AGREEMENT",
        where="src/cascade_methods/integrated_method.py:npred and beat32b_fusion.py:npred "
              "(regex [^a-z0-9])",
        mechanism="SLAKE is bilingual; a Chinese response normalises to the empty string, so any two "
                  "Chinese responses compare equal.",
        measured_on="SLAKE_closed (n=836)",
        agreement_rate_legacy_normaliser=R(float((~dis_legacy).mean())),
        agreement_rate_unicode_safe_normaliser=R(float((~dis_u).mean())),
        published_value_it_invalidates="beat32b_fusion.json idea_findings.F5_double_reading."
                                       "SLAKE_closed.agree_rate = 0.9486",
        consequence="beat32b_fusion.json's F5 double-reading row for SLAKE_closed (agree_rate, "
                    "acc_on_agree, disagree_rate, arbiter accuracies, oracle_UB=1.0) is computed on the "
                    "wrong disagreement set and must not be quoted.  The cell's headline accuracies are "
                    "unaffected -- `correct` never passes through this normaliser.",
        status="THIS FILE USES THE UNICODE-SAFE NORMALISER EVERYWHERE except the prior-art null test.")
    return dict(D1_medevalkit_colon_scoring=d1, D2_cjk_normaliser=d2)


# =====================================================================================================
# PART 5.  cost + verdict
# =====================================================================================================
def part5_cost(C, macro_gain):
    art = json.load(open(RERUN))["per_arm"]["disjoint"]
    base_macro = art["macro_acc"]["always_32b_direct"]
    per_cell_cost_mcq = GEN7_F + GEN32_F                       # both legs on every item
    cost_open_bo8 = BO8_F + GEN32_F
    cost_open_greedy = GEN7_F + GEN32_F
    macro_cost_bo8 = float(np.mean([per_cell_cost_mcq] * 5 + [cost_open_bo8] * 3))
    macro_cost_greedy = float(np.mean([per_cell_cost_mcq] * 5 + [cost_open_greedy] * 3))
    return dict(
        policy="run BOTH legs on every item, then arbitrate post hoc; the arbiter itself is arithmetic "
               "on numbers both forward passes already produced, so it costs nothing",
        flopeq_constants=dict(one_7B_pass=GEN7_F, one_32B_pass=GEN32_F, best_of_8_plus_verify=BO8_F,
                              source="paper as-charged constants (R32=4.57); the honest derived "
                                     "R32=3.816 makes every ratio here WORSE for the method"),
        macro_cost_vs_always_32B_direct=dict(
            open_side_is_best_of_8=R(macro_cost_bo8 / GEN32_F, 3),
            open_side_is_7B_greedy=R(macro_cost_greedy / GEN32_F, 3),
            weighting="MACRO (equal weight per cell) -- NEVER to be paired with a sample-weighted accuracy"),
        latency_ms_parallel=R(max(GEN7_MS, GEN32_MS), 1),
        latency_note="both legs run co-resident on the two A100s, so batch-1 parallel latency is "
                     "max(347, 665) = 665 ms, equal to always-32B-direct; SEQUENTIAL it is 1012 ms",
        macro_accuracy=dict(always_32b_direct=base_macro,
                            with_arbitration=R(base_macro + (macro_gain["mean"] or 0.0)),
                            delta=macro_gain))


# =====================================================================================================
def main():
    log("building cells ...")
    C = build_cells()
    for c in ORDER:
        log(f"  {c:16s} n={len(C[c]['ok7']):6d}  feats={len(C[c]['feats'])}")
    log("null tests ...")
    NT, V = null_tests(C)
    for k, v in NT.items():
        log(f"  {k:18s} passed={v['passed']}  max_abs_dev={v['max_abs_deviation']}")
    assert NT["cascade_rerun"]["passed"] and NT["rebuilt_vectors"]["passed"] and NT["open_text_bar"]["passed"]

    rng = np.random.default_rng(SEED)
    log("PART 1  ceiling ...")
    P1, macro_ceiling = part1_ceiling(C, rng)
    log(f"  macro oracle-arbitration ceiling = {macro_ceiling:+.4f}")
    log("PART 2  identifiability ...")
    P2 = part2_identifiability(C, rng)
    for c in ORDER:
        if "best_one_sided_7B_only" in P2[c]:
            log(f"  {c:16s} pi={P2[c]['base_rate_pi']}  1-sided {P2[c]['best_one_sided_7B_only']['auroc']}"
                f"  2-sided {P2[c]['best_signal_using_the_32B_side']['auroc']}")
    log("PART 3  nested CV, 10 seeds ...")
    P3, gbar = part3(C, rng)
    log(f"  macro nested gain {P3['macro']}")
    log("PART 3b leave-one-cell-out transfer ...")
    loco_mcq = loco_transfer(C, MCQ)
    loco_open = loco_transfer(C, OPEN)
    loco = {}
    for c in ORDER:
        src = loco_mcq if c in MCQ else loco_open
        loco[c] = {k: v for k, v in src[c].items() if k != "per_item"}
    loco_macro = boot_macro_ci({c: (loco_mcq if c in MCQ else loco_open)[c]["per_item"] for c in ORDER}, rng)
    log(f"  LOCO macro {loco_macro}")
    log("PART 3c  SECONDARY: open cells with the best-of-8 + verifier 7B side ...")
    C8 = {c: open_cell(c, "bo8") for c in OPEN}
    frozen_seleff = {"SLAKE_open": 0.850088, "VQA_RAD_open": 0.761905, "PATH_VQA_open": 0.722581}
    sel_dev = 0.0
    for c in OPEN:
        got = float(C8[c]["ok7"][C8[c]["oracle8"] == 1].mean())
        sel_dev = max(sel_dev, abs(got - frozen_seleff[c]))
    P1_8, mac_ceil_8 = part1_ceiling(C8, rng, cells=OPEN)
    P2_8 = part2_identifiability(C8, rng, cells=OPEN)
    P3_8, gbar8 = part3(C8, rng, cells=OPEN)
    open_secondary = dict(
        what="the 7B side of the open cells is best-of-8 + the clean disjoint verifier's pick "
             "(the shipped cheap arm, cost 16.0 FLOP-eq) instead of one greedy pass (cost 1.0)",
        null_test=dict(compared_to="the FROZEN per-set selection efficiencies in "
                                   "src/training_methods/genframe_data.py (0.850088 / 0.761905 / 0.722581)",
                       max_abs_deviation=R(sel_dev, 9), passed=bool(sel_dev < 1e-6)),
        ceiling=P1_8, identifiability=P2_8, honest_realised_gain=P3_8,
        mean_oracle_ceiling_over_the_3_open_cells=R(mac_ceil_8),
        macro_gain_if_this_open_side_is_used=R(
            (sum(P3["per_cell"][c]["nested_gain_seed_mean"] for c in MCQ)
             + sum(P3_8["per_cell"][c]["nested_gain_seed_mean"] for c in OPEN)) / 8.0, 5),
        cost_note="this arm costs 16.0 + 4.57 = 20.57 FLOP-eq per open item, 4.50x always-32B-direct")

    log("PART 4  defects ...")
    P4 = part4_defects(C)
    log("PART 5  cost ...")
    P5 = part5_cost(C, P3["macro"])

    art = json.load(open(RERUN))["per_arm"]["disjoint"]
    bar = art["macro_acc"]["always_32b_direct"]
    win = bool(P3["macro"]["lo"] is not None and P3["macro"]["lo"] > 0)
    fires = [c for c in ORDER if abs(P3["per_cell"][c]["nested_gain_seed_mean"] or 0) > 1e-6]
    covers_zero = [c for c in fires
                   if not P3["per_cell"][c]["ci_on_seed_averaged_per_item_gain"]["sig"]]

    out = dict(
        title="ATTACK C -- two-sided post-hoc arbitration between Lingshu-7B and Lingshu-32B-direct",
        date="2026-08-11",
        reproduce="python3 src/cascade_methods/twosided_veto.py",
        no_gpu=True, no_new_inference=True, no_fabricated_numbers=True,
        not_abstention="every policy in this file emits one of the two models' answers on every item "
                       "(CRITICAL RULE 6)",
        convention="MACRO, equal weight per reporting cell, Variant B (MMMU excluded), 8 cells, 1/8 "
                   "each, n=42,224; CLEAN disjoint open-text verifier (ckpts/train/lora_verifier_disjoint)",
        numerics=dict(threads="OMP/MKL/OpenBLAS pinned to 1 before numpy import",
                      dtype="float64", torch_used=False, tf32_applicable=False,
                      feature_row_order="each cell's own dump order, never re-sorted",
                      estimators="LogisticRegression(lbfgs, C=1, max_iter=2000, random_state=0); "
                                 "HistGradientBoostingClassifier(max_iter=100, max_depth=3, lr=0.1, "
                                 "early_stopping=False, random_state=0)"),
        n_bootstrap=NBOOT, n_bootstrap_auroc=NBOOT_AUROC, seed=SEED, n_fold_seeds=NSEED,
        PRIOR_ART=dict(
            verdict="ATTACK C IS SUBSTANTIALLY A REDISCOVERY",
            what="beat32b_fusion.py already implements two-sided post-hoc arbitration: F3 "
                 "'calibrated confidence-advantage' (on 7B/32B disagreements take the leg with the "
                 "higher isotonic-calibrated P(correct), 5-fold cross-fit) and F5 'double reading' "
                 "(agree -> shared answer; disagree -> arbiter), with an oracle upper bound and a "
                 "recoverability AUROC per cell.  Its PMC-VQA fusion lever is the shipped "
                 "accuracy-max+ variant (CLAUDE.md §0 table, row 'accuracy-max+ (fusion variant)').",
            published_outcome="wins on PMC-VQA only (+0.0135 [+0.0100,+0.0169] vs always-32B-direct); "
                              "LOSES on SLAKE_closed 0.8457 vs 0.8589, VQA_RAD_closed 0.8406 vs 0.8526, "
                              "PATH_VQA_closed 0.8843 vs 0.8891, MedXpert 0.2965 vs 0.3065.  The "
                              "guardrailed router therefore certifies it on PMC-VQA only.",
            what_this_file_adds=[
                "an honest NESTED-CV protocol (the prior art fit its arm choice with eval visibility)",
                "a much richer two-sided feature set (14 features incl. both margins, both "
                "cum_logprobs, both token counts, and their differences) instead of 2 calibrated "
                "confidences, so the negative cannot be blamed on feature poverty",
                "10 fold seeds with mean/sd/range instead of one modulo-5 split",
                "arbitration on the 3 OPEN-TEXT cells, which the prior art never attempted",
                "a leave-one-cell-out transfer arm with zero within-cell fitting",
                "the macro (8-cell equal weight) accounting, which post-dates that artifact",
                "two measurement defects that invalidate part of the prior art's own table (PART 4)"],
            recommendation="do not spend further GPU time on post-hoc arbitration; the ceiling "
                           "measured here is the reason (see HEADLINE)"),
        sources=dict(
            mcq_cells="MedEvalKit/eval_results_lingshu{7b,32b}_full/{}/<DS>/results.json "
                      "(margin, conf, cum_logprob, gen_toks, response, correct)",
            open_cells="ckpts/train/lora_verifier_disjoint/transfer_dump_*_lingshu7b.json + "
                       "ckpts/openvqa/strong_lingshu/ckpt_*_lingshu32b.jsonl(.judge.jsonl)",
            per_arm_vectors=os.path.relpath(VEC, ROOT),
            canonical_macro=os.path.relpath(RERUN, ROOT),
            prior_art=os.path.relpath(FUSION, ROOT)),
        null_tests=NT,
        HEADLINE=dict(
            bar_always_32B_direct_macro=bar,
            macro_gain_needed_for_a_significant_win="+0.0029 (the published CI half-width), i.e. "
                                                    "+0.0235 summed over the 8 cells",
            oracle_arbitration_ceiling_macro=R(macro_ceiling),
            honest_nested_macro_gain=P3["macro"],
            leave_one_cell_out_macro_gain=loco_macro,
            macro_accuracy_with_arbitration=R(bar + (P3["macro"]["mean"] or 0.0)),
            beats_always_32B_direct=win,
            cost=P5["macro_cost_vs_always_32B_direct"],
            kill_criterion="cross-fit arbiter gain CI covers zero on >=2 of the cells where it fires",
            cells_where_the_arbiter_fires=fires,
            cells_where_it_fires_and_its_CI_covers_zero=covers_zero,
            kill_criterion_met=bool(len(covers_zero) >= 2)),
        part1_ceiling=P1,
        part1_macro_oracle_ceiling=R(macro_ceiling),
        part2_identifiability=P2,
        part3_honest_realised_gain=P3,
        part3b_leave_one_cell_out=dict(per_cell=loco, macro=loco_macro,
                                       protocol="one logistic arbiter trained on the DECISIVE "
                                                "disagreements of the other cells of the same format, "
                                                "applied to the held-out cell with zero within-cell "
                                                "fitting"),
        part3c_open_secondary_bestof8_7B_side=open_secondary,
        part4_measurement_defects=P4,
        part5_cost=P5,
        caveats=[
            "The arbiter can only ever act on items where the two models DISAGREE.  On SLAKE_closed "
            "(n=836), VQA_RAD_closed (n=251) and PATH_VQA_closed (n=3362) the decisive-disagreement "
            "counts are small; per-cell CIs there are wide and single-cell flags are within noise "
            "(the standing vqa_rad caveat).",
            "The 32B-side features available at zero cost are thin: for MCQ they are the same four "
            "scalars the 7B exposes (conf, margin, cum_logprob, gen_toks); for open text the 32B dump "
            "exposes only seqlogprob, gen_tokens and the answer string (its self_consistency and "
            "n_distinct are constant because the strong open arm is a single greedy sample, so those "
            "two features carry no information and are dropped by the zero-variance guard).",
            "No new inference was run, so the Round-1 standing caveat about tensor-parallel "
            "reproducibility of the open-text arm does not apply: every open-text number here comes "
            "from the SAME stored dumps as the published arms, and the null test confirms it.",
            "PART 4 D1 means every PMC-VQA figure in this file (and in the paper) is scorer-dependent "
            "at the ~0.01 level per model.  The gain decomposition in part3 separates the artifact-"
            "eligible items, but the underlying labels are still MedEvalKit's.",
            "The in-sample upper bound is an in-sample quantity by construction and is labelled "
            "DIAGNOSTIC; it is not a deployable number."])

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    log(f"WROTE {OUT}")
    print(json.dumps(out["HEADLINE"], indent=1))


if __name__ == "__main__":
    main()
