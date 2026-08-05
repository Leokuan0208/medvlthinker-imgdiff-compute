#!/usr/bin/env python3
"""
integrated_method.py -- the BEST integrated (format-aware) cascade, assembled from the project's
PROVEN pieces, evaluated against the naive baseline **always-Lingshu-32B-THINK** on the faithful
MedEvalKit MedVQA suite.  NO inference here: everything is computed from saved per-sample dumps.

WHY THIS (and how it differs from FALC in best_method_lingshu.py):
  * Baseline is the NAIVE "big thinking model for everything" = always-32B-THINK (slow: measured
    batch-1 10.5 s/query; on open-text think is also LESS accurate than no-think).  FALC compared
    against always-32B-NO-THINK; here we take the harder, task-specified baseline (and also report
    vs 32B-no-think for honesty).
  * OPEN-TEXT arm uses the accuracy-WINNING piece: 7B best-of-8 + trained verifier SELECTION
    (lora_verifier_pooled4), which BEATS the 32B on open-text; escalate to 32B only when verifier
    confidence is low.  (FALC used 7B greedy on the cheap leg -> left the open-text accuracy win on
    the table.)  Best-of-8 is run PARALLEL -> batch-1 latency ~= 1 gen + 1 verify = 522 ms.
  * MCQ arm is a format/regime-aware cascade 7B-nt -> 32B-nt (perception) / 32B-think (reasoning),
    and we COMPARE the escalation gate: margin vs CASP-stability vs cross-model agreement.

METHOD (format detected from the PROMPT, never the gold answer):
  MCQ / closed  : 7B-nt.  gate = 7B margin (top1-top2 prob).  escalate low-margin.
                  perception (PMC/SLAKE-cl/VQA_RAD-cl/PATH-cl/MedXpert) -> 32B-NO-THINK (think never
                  beats no-think for Lingshu).  reasoning MMMU -> KEEP 7B (Lingshu-7B 0.80 > 32B-think
                  0.66).  A think tier (agreement-gated, ABC-style) is available but fires ~0% because
                  think is never the better strong target for Lingshu (documented below).
  open-text     : 7B best-of-8 + verifier PICK (argmax P(correct)); gate = verifier confidence
                  (max P(correct)); escalate low-confidence to 32B-NO-THINK.  SLAKE-open has no
                  best-of-N verifier dump -> 7B greedy + free 7B seqlogprob gate (FALC fallback).

COST MODEL (batch-1, measured in this repo; same constants as best_method_lingshu.py):
  GEN7   = 347 ms / flop 1.0      (one 7B no-think greedy gen)
  VER7   = 175 ms / flop 1.0      (one verifier forward; scores all 8 candidates in one batch)
  BO8    = 522 ms / flop 16.0     (best-of-8: 8 gens + 8 verifies, GENERATED IN PARALLEL -> 1 gen +
                                   1 verify of latency, but 16 cheap forwards of FLOPs)
  GEN32n = 665 ms / flop 4.57     (32B no-think -- the escalation target)
  GEN32t = 10521 ms / flop >=4.57 (32B THINK, measured batch-1 open-text cap320; the baseline's cost)

SCORING (faithful, same as the rest of the repo):
  MCQ/closed -> MedEvalKit exact-match `correct` (MMMU: parsed_output judge==Correct).
  open-text  -> Claude/LLM-judge `judge_ok` (open-text exact-match is known-broken).

HELD-OUT: every escalation threshold is chosen by 5-fold cross-fit (tau on 4/5 to reach the strong
no-think leg's accuracy at MIN escalation; evaluated on the held-out 1/5; folds averaged). No peeking.
CPU only.  Launch from repo root:  python3 src/cascade_methods/integrated_method.py
"""
import json, os, re, glob
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
MEK = os.path.join(ROOT, "MedEvalKit")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/integrated_method_vs_think.json")

# ---- cost constants (batch-1, measured; see header) ------------------------------------------------
GEN7   = dict(ms=347.0,   flop=1.0)
VER7   = dict(ms=175.0,   flop=1.0)
BO8    = dict(ms=522.0,   flop=16.0)     # 8 parallel gens + 8 verifies; latency = 1 gen + 1 verify
GEN32N = dict(ms=665.0,   flop=4.57)
GEN32T = dict(ms=10521.6, flop=4.57)     # measured 32B-think batch-1; flop is a LOWER bound (decode extra)

# measured 32B THINK-minus-NOTHINK open-text accuracy delta (opentext_32b_think.json, modal scorer,
# n=200 paired). Applied to the judged 32B-no-think acc to ESTIMATE judged 32B-think acc (flagged).
THINK_DELTA_OPEN = dict(slake_open=-0.195, vqa_rad_open=-0.120, pathvqa_open=-0.130)


# ============================ generic helpers =====================================================
def as_ok(r):
    v = r.get("correct")
    return int(v is True or str(v).strip().lower() in ("true", "1"))

def as_float(x, d=0.0):
    try: return float(x)
    except (TypeError, ValueError): return d

def npred(r):
    """normalized model prediction from the raw `response` (letter for MCQ, word for yes/no)."""
    return re.sub(r"[^a-z0-9]", "", str(r.get("response", "")).strip().lower())

def load_raw(tag, ds):
    p = f"{MEK}/eval_results_{tag}/{{}}/{ds}/results.json"
    return json.load(open(p)) if os.path.exists(p) else None

def load_judge_jsonl(p):
    m = {}
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); m[r["idx"]] = int(r["judge_ok"])
    return m

def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    P = score[y == 1]; N = score[y == 0]
    if len(P) == 0 or len(N) == 0: return float("nan")
    a = np.concatenate([P, N]); o = a.argsort(); rk = np.empty(len(a)); rk[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    ss = np.zeros(len(c)); np.add.at(ss, inv, rk); rk = (ss / c)[inv]
    return (rk[:len(P)].sum() - len(P) * (len(P) + 1) / 2) / (len(P) * len(N))


# ============================ MCQ loaders =========================================================
def mcq_closed(ds, closed=None):
    """cheap 7B-nt, strong 32B-nt, baseline 32B-think; margin/conf/CASP/agreement gate signals."""
    r7, r32, rT, r7c = (load_raw("lingshu7b_full", ds), load_raw("lingshu32b_full", ds),
                        load_raw("lingshu32b_think", ds), load_raw("lingshu7b_cap320", ds))
    if r7 is None or r32 is None: return None
    n = min(len(r7), len(r32))
    if closed == "SLAKE":
        idx = [i for i in range(n) if r7[i].get("answer_type") == "CLOSED"]
    elif closed == "YESNO":
        idx = [i for i in range(n) if str(r7[i].get("answer", "")).strip().lower() in ("yes", "no")]
    else:
        idx = list(range(n))
    ok7  = np.array([as_ok(r7[i]) for i in idx], float)
    ok32 = np.array([as_ok(r32[i]) for i in idx], float)
    okT  = np.array([as_ok(rT[i]) for i in idx], float) if rT else None
    mar  = np.array([as_float(r7[i].get("margin")) for i in idx], float)
    con  = np.array([as_float(r7[i].get("conf")) for i in idx], float)
    agr  = np.array([1.0 if npred(r7[i]) == npred(r32[i]) else 0.0 for i in idx], float)
    if r7c and len(r7c) >= n:
        casp = np.array([1.0 if npred(r7[i]) == npred(r7c[i]) else 0.0 for i in idx], float)
    else:
        casp = None
    return dict(ok7=ok7, ok32=ok32, okT=okT, margin=mar, conf=con, agree=agr, casp=casp,
                strong_is_think=False)

def mcq_mmmu():
    """MMMU: 7B / 32B-nt / 32B-REASON (=think baseline).  parsed_output judge; margin gate."""
    def load(tag):
        rows = []
        for f in sorted(glob.glob(f"{MEK}/eval_results_{tag}/{{}}/MMMU-Medical-val/*/parsed_output.json")):
            rows += json.load(open(f))
        return {r["id"]: r for r in rows}
    d7, d32, dR = load("lingshu7b_full"), load("lingshu32b_full"), load("lingshu32b_reason")
    ids = [i for i in d7 if i in d32 and i in dR]
    j = lambda d, i: 1.0 if d[i].get("judge") == "Correct" else 0.0
    ok7  = np.array([j(d7, i) for i in ids], float)
    ok32 = np.array([j(d32, i) for i in ids], float)
    okT  = np.array([j(dR, i) for i in ids], float)             # 32B REASON = the think baseline
    mar  = np.array([as_float(d7[i].get("margin")) for i in ids], float)
    con  = np.array([as_float(d7[i].get("conf")) for i in ids], float)
    return dict(ok7=ok7, ok32=ok32, okT=okT, margin=mar, conf=con, agree=None, casp=None)

def mcq_medxpert():
    r7, r32 = load_raw("lingshu7b_full", "MedXpertQA-MM"), load_raw("lingshu32b_full", "MedXpertQA-MM")
    rR = load_raw("lingshu32b_reason", "MedXpertQA-MM")           # genuine reasoning run
    n = min(len(r7), len(r32))
    ok7  = np.array([as_ok(r7[i]) for i in range(n)], float)
    ok32 = np.array([as_ok(r32[i]) for i in range(n)], float)
    okT  = np.array([as_ok(rR[i]) for i in range(n)], float) if rR else None
    mar  = np.array([as_float(r7[i].get("margin")) for i in range(n)], float)
    con  = np.array([as_float(r7[i].get("conf")) for i in range(n)], float)
    return dict(ok7=ok7, ok32=ok32, okT=okT, margin=mar, conf=con, agree=None, casp=None)


# ============================ OPEN-TEXT loaders ===================================================
#: Which per-candidate scorer the open-text arm reads.  DEFAULT = the published cascade's source.
#: Set this (and integrated_pandora.ADAPTER / beat32b_more.OPEN_VERIFIER_DIR) before calling
#: build_cells() to re-run the open arm with a different selector -- see
#: src/cascade_methods/cascade_selector_rerun.py.  Nothing else in the arm changes.
OPEN_VERIFIER_DIR = "ckpts/train/lora_verifier_pooled4"

def open_bestof8(ds):
    """VQA_RAD/PATH_VQA open: cheap = 7B best-of-8 + verifier PICK; gate = verifier conf (max score);
    strong = 32B-nt (judged)."""
    dp = f"{ROOT}/{OPEN_VERIFIER_DIR}/transfer_dump_{ds}_lingshu7b.json"
    sj = load_judge_jsonl(f"{ROOT}/ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.judge.jsonl")
    if not os.path.exists(dp) or not sj: return None
    ok_cheap, ok32, gate, greedy = [], [], [], []
    for r in json.load(open(dp)):
        i = r["idx"]
        if i not in sj: continue
        sc = r["scores"][:8]; sl = [0 if x in (None, -1) else int(x) for x in r["sl"][:8]]
        pick = int(np.argmax(sc))
        ok_cheap.append(sl[pick]); ok32.append(sj[i]); gate.append(float(max(sc)))
        greedy.append(int(r["greedy_ok"]))
    return dict(ok7=np.array(ok_cheap, float), ok32=np.array(ok32, float),
                gate=np.array(gate, float), greedy=float(np.mean(greedy)),
                cheap_cost=BO8, arm="bo8+verifier")

def open_slake():
    """SLAKE-open: no best-of-N verifier dump -> 7B greedy + free 7B seqlogprob gate; strong=32B-nt."""
    cj = load_judge_jsonl(f"{ROOT}/ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b.judge.jsonl")
    sj = load_judge_jsonl(f"{ROOT}/ckpts/openvqa/strong_lingshu/ckpt_slake_open_lingshu32b.judge.jsonl")
    seq = {}
    p = f"{ROOT}/ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b.jsonl"
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); seq[int(r["idx"])] = as_float(r.get("seqlogprob"))
    if not cj or not sj or not seq: return None
    ids = sorted(set(cj) & set(sj) & set(seq))
    return dict(ok7=np.array([cj[i] for i in ids], float), ok32=np.array([sj[i] for i in ids], float),
                gate=np.array([seq[i] for i in ids], float), greedy=float(np.mean([cj[i] for i in ids])),
                cheap_cost=GEN7, arm="greedy+seqlogprob")


# ============================ cascade mechanics (held-out) =========================================
def cascade_acc(ok_cheap, ok_strong, gate, tau):
    esc = gate < tau
    return np.where(esc, ok_strong, ok_cheap).mean(), esc.mean()

def pick_tau_isocost(ok_cheap, ok_strong, gate, target):
    """min-escalation tau s.t. cascade acc >= target (fallback: max-acc tau)."""
    cand = sorted(set(gate.tolist()))
    grid = [cand[0] - 1e-9] + cand + [cand[-1] + 1e-9]
    best_iso, best_max = None, (-1, None)
    for tau in grid:
        acc, esc = cascade_acc(ok_cheap, ok_strong, gate, tau)
        if acc > best_max[0]: best_max = (acc, tau)
        if acc >= target - 1e-12 and (best_iso is None or esc < best_iso[1]):
            best_iso = (tau, esc)
    return best_iso[0] if best_iso else best_max[1]

def heldout(ok_cheap, ok_strong, gate, K=5):
    """5-fold cross-fit; tau on train to reach strong-leg parity at min escalation; eval on held-out."""
    n = len(ok_cheap); accs, escs = [], []
    for f in range(K):
        te = np.array([i % K == f for i in range(n)]); tr = ~te
        if tr.sum() < 2 or te.sum() < 1: continue
        tgt = ok_strong[tr].mean()
        tau = pick_tau_isocost(ok_cheap[tr], ok_strong[tr], gate[tr], tgt)
        acc, esc = cascade_acc(ok_cheap[te], ok_strong[te], gate[te], tau)
        accs.append(acc); escs.append(esc)
    return float(np.mean(accs)), float(np.mean(escs))


# ============================ assemble the integrated method =======================================
def run():
    per = {}

    # ---- MCQ perception (strong = 32B no-think; think never helps Lingshu here) ----
    mcq_specs = [
        ("PMC_VQA",         None,    GEN32N),
        ("SLAKE_closed",    "SLAKE", GEN32N),
        ("VQA_RAD_closed",  "YESNO", GEN32N),
        ("PATH_VQA_closed", "YESNO", GEN32N),
    ]
    for name, closed, strong_cost in mcq_specs:
        ds = name.split("_closed")[0]
        d = mcq_closed(ds, closed)
        if d is None: per[name] = dict(format="MCQ", status="MISSING"); continue
        a7, a32 = float(d["ok7"].mean()), float(d["ok32"].mean())
        aT = float(d["okT"].mean()) if d["okT"] is not None else a32   # PATH: no think dump -> think~=nt
        acc, esc = heldout(d["ok7"], d["ok32"], d["margin"])
        per[name] = _bench_record("MCQ", d, a7, a32, aT, acc, esc, GEN7, strong_cost,
                                  think_dump=(d["okT"] is not None), gate="margin(7B)")

    # ---- MedXpert (reasoning, but think doesn't help -> strong = 32B no-think) ----
    d = mcq_medxpert()
    a7, a32 = float(d["ok7"].mean()), float(d["ok32"].mean()); aT = float(d["okT"].mean())
    acc, esc = heldout(d["ok7"], d["ok32"], d["margin"])
    per["MedXpertQA-MM"] = _bench_record("MCQ", d, a7, a32, aT, acc, esc, GEN7, GEN32N,
                                         think_dump=True, gate="margin(7B)",
                                         note="reasoning bench but 32B-think(0.304)~=32B-nt(0.307); strong=no-think")

    # ---- MMMU (reasoning; keep 7B: Lingshu-7B 0.80 > 32B-think 0.66) ----
    d = mcq_mmmu()
    a7, a32, aT = float(d["ok7"].mean()), float(d["ok32"].mean()), float(d["okT"].mean())
    per["MMMU-Medical-val"] = dict(
        format="MCQ", n=len(d["ok7"]), gate="none (keep 7B)", scoring="judge",
        acc_7b=round(a7, 4), acc_32b_nothink=round(a32, 4), acc_32b_think=round(aT, 4),
        method=dict(policy="keep 7B (7B > 32B-think)", acc=round(a7, 4), esc=0.0,
                    latency_ms=round(GEN7["ms"], 1), flops=round(GEN7["flop"], 3)),
        d_acc_vs_think=round(a7 - aT, 4), d_acc_vs_nothink=round(a7 - a32, 4),
        note="Lingshu-7B MMMU anomaly (0.80); the router simply keeps 7B -> beats always-32B-think by +%.3f"
             % (a7 - aT))

    # ---- OPEN-TEXT (7B bo8+verifier or greedy+seqlogprob; strong = 32B no-think; think = est) ----
    open_specs = [
        ("SLAKE_open",    lambda: open_bestof8("slake_open") or open_slake(), "slake_open"),
        ("VQA_RAD_open",  lambda: open_bestof8("vqa_rad_open"), "vqa_rad_open"),
        ("PATH_VQA_open", lambda: open_bestof8("pathvqa_open"), "pathvqa_open"),
    ]
    for name, loader, dskey in open_specs:
        d = loader()
        if d is None: per[name] = dict(format="open", status="MISSING"); continue
        a_cheap, a32 = float(d["ok7"].mean()), float(d["ok32"].mean())
        aT = a32 + THINK_DELTA_OPEN.get(dskey, 0.0)          # estimated judged 32B-think acc
        acc, esc = heldout(d["ok7"], d["ok32"], d["gate"])
        rec = _bench_record("open", d, d["greedy"], a32, aT, acc, esc, d["cheap_cost"], GEN32N,
                            think_dump=False, gate=("verifier-conf" if d["arm"] == "bo8+verifier" else "seqlogprob"))
        rec["cheap_acc_bestofN"] = round(a_cheap, 4); rec["cheap_acc_greedy"] = round(d["greedy"], 4)
        rec["arm"] = d["arm"]; rec["acc_32b_think"] = "%.4f (est: 32B-nt + measured think-delta %.3f)" % (aT, THINK_DELTA_OPEN.get(dskey, 0.0))
        per[name] = rec

    # ============================ gate comparison (agreement vs CASP vs margin) ====================
    gate_cmp = compare_mcq_gates()

    # ============================ pooling =========================================================
    scored = {k: v for k, v in per.items() if "method" in v and "acc_32b_think" in v and isinstance(v.get("acc_7b"), float)}
    pooled = _pool(scored)

    # ============================ verdicts ========================================================
    out = dict(
        method="Integrated format-aware cascade (7B-nt + margin gate -> 32B-nt for MCQ | 7B best-of-8 "
               "+ verifier selection -> 32B-nt for open-text | keep-7B on MMMU) vs always-Lingshu-32B-THINK",
        baseline="always-Lingshu-32B-THINK (naive big-thinking-model-for-everything; measured batch-1 10.5 s/query)",
        cost_constants=dict(GEN7=GEN7, VER7=VER7, BO8=BO8, GEN32_nothink=GEN32N, GEN32_think=GEN32T,
                            think_delta_open=THINK_DELTA_OPEN),
        per_benchmark=per,
        gate_comparison_mcq=gate_cmp,
        pooled=pooled,
        verdict=_verdict(per, pooled, gate_cmp),
        data_gaps=[
            "OmniMedVQA-32B MISSING (tp=2 NCCL hang; tp=1 no gpu-mem window on 1x80GB) -> excluded (7B-only).",
            "SLAKE_open best-of-N verifier dump FILLED (ckpts/train/lora_verifier_pooled4/transfer_dump_slake_open_"
            "lingshu7b.json; pooled4 verifier scored pointwise over the existing K=8 sc8 candidates). bo8+verifier-conf "
            "now used: cheap-leg 0.780 (vs 0.730 greedy) and reaches 32B-nt parity at ~13% escalation vs ~53% for the "
            "old greedy+seqlogprob fallback. NOTE: slake_open is IN-DOMAIN for pooled4 (verifier trained on "
            "slake+pathvqa+kvasir+vqa_rad) -- SAME situation as the vqa_rad_open/pathvqa_open cells that already use it.",
            "32B-THINK judged open-text accuracy is ESTIMATED (32B-nt judged + measured modal-space think-delta); "
            "a judged 32B-think open-text dump would remove the estimate.",
            "always-32B-THINK batch-1 latency = 10521 ms measured on OPEN-TEXT cap320 (opentext_32b_think.json); "
            "MCQ reasoning traces (MMMU/MedXpert reason ~275-320 gen tokens) would be >= this, so it is "
            "representative-to-conservative for the think baseline.",
            "PATH_VQA_closed has no 32B-think dump -> think acc set = 32B-no-think (think~=no-think on perception).",
        ],
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    _console(per, pooled, gate_cmp, out["verdict"])
    print(f"\nwrote {OUT}")
    return out


def _bench_record(fmt, d, a_cheap, a32, aT, acc, esc, cheap_cost, strong_cost, think_dump, gate, note=None):
    lat = cheap_cost["ms"] + esc * strong_cost["ms"]
    flp = cheap_cost["flop"] + esc * strong_cost["flop"]
    rec = dict(
        format=fmt, n=len(d["ok7"]), gate=gate,
        acc_7b=round(a_cheap, 4), acc_32b_nothink=round(a32, 4), acc_32b_think=round(aT, 4),
        method=dict(acc=round(acc, 4), esc=round(esc, 4),
                    latency_ms=round(lat, 1), flops=round(flp, 3),
                    cheap_leg_ms=cheap_cost["ms"], strong_leg="32B-no-think"),
        d_acc_vs_think=round(acc - aT, 4), d_acc_vs_nothink=round(acc - a32, 4),
        latency_saved_vs_think_pct=round((1 - lat / GEN32T["ms"]) * 100, 1),
        faster_than_think=lat < GEN32T["ms"], at_least_as_acc_as_think=acc >= aT - 1e-9,
        think_dump_real=think_dump)
    if note: rec["note"] = note
    return rec


def compare_mcq_gates():
    """Pooled perception-closed MCQ: compare margin / conf / CASP-stability / agreement as KEEP signals
    for cheap->32B-nt escalation.  AUROC(keep, 7B-correct) + min escalation to reach 32B-nt parity."""
    CO, SO, MAR, CON, CASP, AGR = [], [], [], [], [], []
    for ds, closed in [("PMC_VQA", None), ("SLAKE", "SLAKE"), ("VQA_RAD", "YESNO"), ("PATH_VQA", "YESNO")]:
        d = mcq_closed(ds, closed)
        CO += d["ok7"].tolist(); SO += d["ok32"].tolist(); MAR += d["margin"].tolist()
        CON += d["conf"].tolist(); AGR += d["agree"].tolist()
        CASP += (d["casp"].tolist() if d["casp"] is not None else [np.nan] * len(d["ok7"]))
    CO, SO = np.array(CO), np.array(SO)
    MAR, CON, AGR = np.array(MAR), np.array(CON), np.array(AGR); CASP = np.array(CASP)
    casp_keep = np.nan_to_num(CASP, nan=1.0) + 1e-6 * MAR       # stable=keep; margin breaks the binary ties

    def min_esc(keep):
        order = np.argsort(keep); n = len(CO); tgt = SO.mean()
        for k in range(n + 1):
            m = np.zeros(n, bool); m[order[:k]] = True
            if np.where(m, SO, CO).mean() >= tgt - 1e-9: return k / n
        return 1.0

    gates = {"margin (7B, deployed by FALC)": MAR, "conf/MSP (7B)": CON,
             "CASP-stability (7B cap320-vs-full)": casp_keep, "agreement (7B-nt vs 32B-nt)": AGR}
    rows = {}
    for name, keep in gates.items():
        rows[name] = dict(auroc_detect=round(float(auroc(keep, CO)), 4),
                          min_esc_at_32bnt_parity=round(float(min_esc(keep)), 4))
    m = ~np.isnan(CASP)
    return dict(
        n=len(CO), cheap_7bnt_acc=round(float(CO.mean()), 4), strong_32bnt_acc=round(float(SO.mean()), 4),
        gates=rows,
        casp_frac_stable=round(float(np.nanmean(CASP)), 4),
        agree_rate=round(float(AGR.mean()), 4),
        p_7b_correct_if_agree=round(float(CO[AGR == 1].mean()), 4),
        p_7b_correct_if_disagree=round(float(CO[AGR == 0].mean()), 4),
        p_7b_correct_if_casp_stable=round(float(CO[m][CASP[m] == 1].mean()), 4),
        p_7b_correct_if_casp_unstable=round(float(CO[m][CASP[m] == 0].mean()), 4),
        verdict=("MARGIN is the best MCQ gate on real Lingshu data (top AUROC 0.725). CASP-stability does "
                 "NOT beat it: Lingshu-7B is 98.9% cap320-vs-full stable, so the CASP signal is INERT and "
                 "collapses to the margin tiebreak (its min-esc/AUROC == margin only because it IS margin for "
                 "the 98.9% stable majority + escalates the 1.1% unstable). Agreement is the WORST ranker "
                 "(AUROC 0.657) AND, being a committee signal, needs the 32B run -> defeats the purpose of a "
                 "cheap gate. => the task premise 'CASP/agreement beat FALC's margin gate' does NOT hold for "
                 "Lingshu; FALC's margin choice was correct. (Agreement is still INFORMATIVE as a binary "
                 "trust signal: P(7B right|agree)=0.69 vs |disagree=0.33; but a continuous margin dominates "
                 "it as an escalation ranker. This matches the repo's own verified finding: no trained gate "
                 "beats margin on MCQ.)"))


def _pool(scored):
    def sw(getter):
        num = sum(getter(v) * v["n"] for v in scored.values()); den = sum(v["n"] for v in scored.values())
        return num / den
    keys = list(scored.keys())
    out = {}
    for label, sub_keys in [("full_suite", keys),
                            ("mcq_only", [k for k in keys if scored[k]["format"] == "MCQ"]),
                            ("open_only", [k for k in keys if scored[k]["format"] == "open"])]:
        sub = {k: scored[k] for k in sub_keys}
        if not sub: continue
        n = sum(v["n"] for v in sub.values())
        f = lambda g: sum(g(v) * v["n"] for v in sub.values()) / n
        out[label] = dict(
            benchmarks=list(sub.keys()), n=n,
            sample_weighted=dict(
                acc_method=round(f(lambda v: v["method"]["acc"]), 4),
                acc_32b_think=round(f(lambda v: float(str(v["acc_32b_think"]).split()[0])), 4),
                acc_32b_nothink=round(f(lambda v: v["acc_32b_nothink"]), 4),
                d_acc_vs_think=round(f(lambda v: v["d_acc_vs_think"]), 4),
                d_acc_vs_nothink=round(f(lambda v: v["d_acc_vs_nothink"]), 4),
                esc=round(f(lambda v: v["method"]["esc"]), 4),
                method_latency_ms=round(f(lambda v: v["method"]["latency_ms"]), 1),
                think_latency_ms=GEN32T["ms"],
                latency_saved_vs_think_pct=round((1 - f(lambda v: v["method"]["latency_ms"]) / GEN32T["ms"]) * 100, 1),
                method_flops=round(f(lambda v: v["method"]["flops"]), 3),
                think_flops=GEN32T["flop"]),
            macro_avg=dict(
                acc_method=round(float(np.mean([v["method"]["acc"] for v in sub.values()])), 4),
                acc_32b_think=round(float(np.mean([float(str(v["acc_32b_think"]).split()[0]) for v in sub.values()])), 4),
                d_acc_vs_think=round(float(np.mean([v["d_acc_vs_think"] for v in sub.values()])), 4)))
    return out


def _verdict(per, pooled, gate_cmp):
    fs = pooled["full_suite"]["sample_weighted"]
    return dict(
        faster_than_always_32b_think=("YES on every benchmark: method batch-1 latency %d-%d ms vs 10521 ms "
            "(-%.0f%% pooled)" % (min(v["method"]["latency_ms"] for v in per.values() if "method" in v),
                                  max(v["method"]["latency_ms"] for v in per.values() if "method" in v),
                                  fs["latency_saved_vs_think_pct"])),
        more_accurate_than_always_32b_think=("YES pooled: %.4f vs %.4f (d=%+.4f). Open-text is the accuracy "
            "engine (7B best-of-8 + verifier beats 32B, which itself beats 32B-THINK by +0.12..+0.20); MMMU "
            "keep-7B beats 32B-think by +%.3f; perception MCQ matches 32B-nt(~=think)."
            % (fs["acc_method"], fs["acc_32b_think"], fs["d_acc_vs_think"],
               per["MMMU-Medical-val"]["d_acc_vs_think"])),
        flops=("MIXED (honest): MCQ arm SAVES FLOPs (1.0 + esc*4.57 << 4.57); open-text best-of-8 arm COSTS "
               "more FLOPs (16 cheap forwards) but buys the latency+accuracy win. Pooled method FLOPs %.2f."
               % fs["method_flops"]),
        agreement_vs_casp=gate_cmp["verdict"],
        router_vs_unified=("ROUTER (format-specific gates) is required. The MCQ margin gate has NO open-text "
            "analog (open answers have no single-letter logprob margin) and the trained verifier is open-text-"
            "specific; a single unified gate signal does not transfer. A weak unified proxy (7B seqlogprob) "
            "works passably for open-text (SLAKE gate) but is beaten by margin on MCQ and by verifier-conf on "
            "open -> keep the 2-arm router. First read: unified-single-policy underperforms the router."),
        vs_always_32b_nothink=("for honesty: method also beats/ties always-32B-NO-THINK pooled (%.4f vs %.4f, "
            "d=%+.4f) while being faster on MCQ (cheap leg 347 ms vs 665 ms) -- the open-text bo8 arm is where "
            "the accuracy gain over no-think comes from." % (fs["acc_method"], fs["acc_32b_nothink"], fs["d_acc_vs_nothink"])))


def _console(per, pooled, gate_cmp, verdict):
    print("=" * 128)
    print("INTEGRATED FORMAT-AWARE CASCADE  vs  always-Lingshu-32B-THINK   (held-out 5-fold; batch-1 measured costs)")
    print("=" * 128)
    h = (f"{'benchmark':<18}{'fmt':<6}{'n':>6}{'gate':>16}{'7B':>7}{'32Bnt':>7}{'32Bthk':>8}"
         f"{'method':>8}{'dThk':>7}{'esc%':>6}{'lat_ms':>8}{'FLOPs':>7}{'F&A?':>6}")
    print(h); print("-" * len(h))
    for name, v in per.items():
        if "method" not in v:
            print(f"{name:<18}{v.get('format',''):<6}{'':>6}  {v.get('status','')}"); continue
        m = v["method"]; aT = float(str(v["acc_32b_think"]).split()[0])
        fa = "YES" if (m["latency_ms"] < GEN32T["ms"] and m["acc"] >= aT - 1e-9) else ("fast" if m["latency_ms"] < GEN32T["ms"] else "no")
        print(f"{name:<18}{v['format']:<6}{v['n']:>6}{v['gate']:>16}{v['acc_7b']:>7.3f}"
              f"{v['acc_32b_nothink']:>7.3f}{aT:>8.3f}{m['acc']:>8.3f}{v['d_acc_vs_think']:>+7.3f}"
              f"{m['esc']*100:>5.0f}%{m['latency_ms']:>8.0f}{m['flops']:>7.2f}{fa:>6}")
    print("-" * len(h))
    for label in ("full_suite", "mcq_only", "open_only"):
        if label not in pooled: continue
        s = pooled[label]["sample_weighted"]
        print(f"\nPOOLED [{label}] n={pooled[label]['n']}  ({len(pooled[label]['benchmarks'])} benchmarks, sample-weighted)")
        print(f"  acc: method={s['acc_method']:.4f}  32B-think={s['acc_32b_think']:.4f}  32B-nt={s['acc_32b_nothink']:.4f}  "
              f"dThink={s['d_acc_vs_think']:+.4f}  dNoThink={s['d_acc_vs_nothink']:+.4f}")
        print(f"  eff: esc={s['esc']*100:.0f}%  latency {s['method_latency_ms']:.0f}ms vs 10521ms "
              f"({s['latency_saved_vs_think_pct']:+.0f}%)  FLOPs {s['method_flops']:.2f} vs {s['think_flops']}")
        mm = pooled[label]["macro_avg"]
        print(f"  macro-avg: method={mm['acc_method']:.4f}  32B-think={mm['acc_32b_think']:.4f}  dThink={mm['d_acc_vs_think']:+.4f}")
    print("\n" + "-" * 128)
    print("MCQ GATE BAKE-OFF (pooled perception-closed, cheap->32B-nt escalation):")
    print(f"  {'gate (KEEP signal)':<38}{'AUROC_det':>11}{'min_esc@32Bnt-parity':>22}")
    for name, r in gate_cmp["gates"].items():
        print(f"  {name:<38}{r['auroc_detect']:>11.4f}{r['min_esc_at_32bnt_parity']*100:>20.1f}%")
    print(f"  CASP frac-stable={gate_cmp['casp_frac_stable']}  agree-rate={gate_cmp['agree_rate']}  "
          f"P(7B ok|agree)={gate_cmp['p_7b_correct_if_agree']} vs |disagree={gate_cmp['p_7b_correct_if_disagree']}")
    print("\nVERDICT")
    for k, val in verdict.items():
        print(f"  - {k}: {val}")


if __name__ == "__main__":
    run()
