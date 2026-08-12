#!/usr/bin/env python3
"""
cost_decomposition.py -- ATTACK 1 under the NEW OBJECTIVE.

    MINIMISE compute  SUBJECT TO  macro accuracy non-inferior to always-32B-direct
    (paired-bootstrap CI lower bound of the 8-cell macro delta >= -0.0029).

WHAT THIS FILE ANSWERS (each with numbers, each traced to a named artifact):

  Q0  A complete (cell x stage) FLOP decomposition of the SHIPPED accuracy-max arm at its 1.740x,
      into: cheap-leg vision / cheap-leg LM-prefill / cheap-leg decode, the N open-arm generations,
      verifier vision / prefill / decode, the selector head, and the 32B escalation (vision /
      prefill / decode).  Which stages dominate, on which cells.

  Q1  What FRACTION of total compute is spent on questions the cascade escalates ANYWAY.  That is
      pure waste under a cascade structure and is exactly what the published pre-generation-router
      theorem attacks.  Computed PER ITEM (not esc_rate x meanN), because the adaptive-N controller
      draws MORE samples on the questions it later escalates, so the factorised product understates
      the waste.

  Q2  Is the open-text machinery worth running at all under the new objective?  Per-cell accuracy
      and cost of {always-7B greedy, 7B best-of-N + verifier select (no escalation), the deployed
      open arm, always-32B-direct}, plus the macro consequence of switching the open arm OFF.

  Q3  Theoretical minimum compute at parity: a per-cell knapsack over the measured deployable arms,
      minimising macro FLOP-eq subject to the non-inferiority constraint.  Reported three ways:
      EVAL-VISIBLE BOUND (never achievable), HONEST NESTED-CV (the number to quote), and two
      PERMUTATION / synthetic NULLS.

  Q4  MODEL FOOTPRINT for every frontier point: which weights must be resident, measured from the
      safetensors indices on disk and from the 2026-08-11 test-time VRAM measurement.

METHODOLOGY (non-negotiables, all enforced below)
  * NULL TESTS FIRST.  N1 reproduces the published 8-cell macro accuracy + the published per-cell
    as-charged cost + the 1.740x ratio from the frozen eval vectors.  N2 rebuilds the FLOP model
    component-by-component and reproduces R32_derived.  N3 re-derives the per-item escalation / draw
    vectors and asserts they reproduce the PUBLISHED aggregate escalation rates and meanN.  N4
    asserts the stage decomposition re-sums to the published per-cell cost.
  * Paired item bootstrap, nboot=10000, ONE shared resample stream per cell reused by every policy
    and every baseline.
  * NESTED CV wherever an arm or a threshold is selected; >= 10 fold seeds; guardrail per cell.
  * Numerics pinned: OMP_NUM_THREADS=1, PYTHONHASHSEED=0, pure numpy/sklearn on CPU (no TF32 path).
  * No GPU.  Reads only committed artifacts + checkpoints.

Launch from the repo root:   python3 src/cascade_methods/cost_decomposition.py
Writes results/cascade_methods/artifacts/cost_decomposition_2026-08-12.json
"""
import json
import os
import struct
import sys
import zlib

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
OUT = os.path.join(ART, "cost_decomposition_2026-08-12.json")
_HERE = os.path.join(REPO, "src/cascade_methods")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SEED = 20260812
NBOOT = 10000
TIE_TOL = 0.0029          # pre-registered non-inferiority margin (CI lower bound must be >= -TIE_TOL)
NFOLD = 5
N_FOLD_SEEDS = 12         # >= 10 seeds, as required
N_NULL = 200

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
MCQ = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"]
OPEN = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
OPEN_KEY = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}
BASE = "always_32b_direct"
# the deployable menu (oracle_mode_32b is an ORACLE and is excluded, as in cost_floor.py)
DEPLOYABLE = ["always_7b", "always_32b_direct", "always_32b_reasoning",
              "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]

R32_CHARGED = 4.57        # the paper constant (lingshu_medeval_cascade.py:21)
R32_DERIVED = 3.816       # flop_ratio_derivation_2026-08-03.json

rep = {}


# =================================================================================================
# 0.  LOAD FROZEN ARTIFACTS
# =================================================================================================
Z = np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"))
MACD = json.load(open(os.path.join(ART, "_selector_rerun_parts/macro_disjoint.json")))
PUB = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))["per_arm"]["disjoint"]
FLOPD = json.load(open(os.path.join(ART, "flop_ratio_derivation_2026-08-03.json")))
CF = json.load(open(os.path.join(ART, "cost_floor_2026-08-10.json")))
VRAM = json.load(open(os.path.join(ART, "vram_testtime_2026-08-11.json")))
PCC = MACD["cost"]["per_cell_as_charged"]

OK = {(c, s): Z[f"{c}|{s}"].astype(np.float64) for c in CELLS for s in DEPLOYABLE}
NITEM = {c: len(Z[f"{c}|{BASE}"]) for c in CELLS}
ARMDEC = CF["arm_decomposition"]["table"]


# =================================================================================================
# 1.  NULL TESTS
# =================================================================================================
def null_n1():
    """Reproduce published macro accuracy + per-cell as-charged cost + the 1.740x ratio."""
    dev_a, dev_c, rows = [], [], {}
    for s in DEPLOYABLE:
        m = float(np.mean([OK[(c, s)].mean() for c in CELLS]))
        d = abs(m - PUB["macro_acc"][s])
        dev_a.append(d)
        f = float(np.mean([PCC[c][s]["flops"] for c in CELLS]))
        dc = abs(f - PUB["cost_macro"][s]["flops"])
        dev_c.append(dc)
        rows[s] = dict(macro_recomputed=round(m, 6), macro_published=PUB["macro_acc"][s],
                       abs_dev_acc=round(d, 8), cost_recomputed=round(f, 6),
                       cost_published=PUB["cost_macro"][s]["flops"], abs_dev_cost=round(dc, 8))
    r = float(np.mean([PCC[c]["method_accuracy_max_veto"]["flops"] for c in CELLS])) / R32_CHARGED
    ok = max(dev_a) < 5e-5 and max(dev_c) < 5e-4
    return dict(
        name="N1 -- reproduce the published 8-cell macro accuracy, per-cell as-charged cost, and 1.740x",
        source_vectors="results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz",
        source_published="results/cascade_methods/artifacts/cascade_selector_rerun_2026-08-05.json"
                         ":per_arm.disjoint",
        per_system=rows,
        max_abs_dev_accuracy=round(max(dev_a), 8),
        max_abs_dev_cost_flops=round(max(dev_c), 8),
        macro_ratio_accmax_vs_direct=round(r, 4),
        published_ratio=PUB["ratios_macro"]["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"],
        verdict="PASS" if ok else "FAIL")


def null_n2():
    """Rebuild the FLOP model from MEASURED parameter counts + token geometry; reproduce R32_derived
    AND extract the vision / LM-prefill / decode split each stage of the decomposition needs."""
    P7 = FLOPD["parameter_counts"]["lingshu_7b"]["by_role"]
    P32 = FLOPD["parameter_counts"]["lingshu_32b"]["by_role"]
    g = FLOPD["flop_model"]["operating_point"]
    T, M, PP = g["T"], g["M"], g["P_patches"]
    VITL, VITD, VITFULL, WIN = 32, 1280, 4, 64
    vit_attn = (VITFULL * 4 * PP ** 2 * VITD +
                (VITL - VITFULL) * (PP / WIN) * 4 * WIN ** 2 * VITD) / 1e9
    comp, tot = {}, {}
    for nm, cfg in [("lingshu_7b", dict(L=28, d=3584, G=g["G_7b"], role=P7)),
                    ("lingshu_32b", dict(L=64, d=5120, G=g["G_32b"], role=P32))]:
        L, d, G, role = cfg["L"], cfg["d"], cfg["G"], cfg["role"]
        c = dict(
            vision_tower_dense=2 * role["vision_tower"] * PP / 1e9,
            vision_tower_attn=vit_attn,
            vision_merger=2 * role["vision_merger"] * M / 1e9,
            lm_prefill_dense=2 * role["lm_body"] * T / 1e9,
            lm_prefill_attn=2 * L * T ** 2 * d / 1e9,
            lm_decode_dense=2 * role["lm_body"] * (G - 1) / 1e9,
            # decode attention is NOT causally halved -- each decoded token attends to the whole
            # context, and both QK^T and AV are full matmuls.  The context grows during decode, so
            # the mean context is T + (G-1)/2.  Matched to cost_floor_2026-08-10.json:N2.
            lm_decode_attn=4 * L * (G - 1) * (T + (G - 1) / 2) * d / 1e9,
            lm_head=2 * role["lm_head"] * G / 1e9)
        comp[nm] = {k: round(v, 2) for k, v in c.items()}
        tot[nm] = round(sum(c.values()), 2)
    ref = CF["null_tests"]["N2"]["rebuilt_component_gflops"]
    dev = max(abs(comp[m][k] - ref[m][k]) for m in comp for k in comp[m])
    devt = max(abs(tot[m] - CF["null_tests"]["N2"]["rebuilt_total_gflops"][m]) for m in tot)
    stages = {}
    for nm in comp:
        c = comp[nm]
        vis = c["vision_tower_dense"] + c["vision_tower_attn"] + c["vision_merger"]
        pre = c["lm_prefill_dense"] + c["lm_prefill_attn"]
        dec = c["lm_decode_dense"] + c["lm_decode_attn"] + c["lm_head"]
        stages[nm] = dict(vision_gflops=round(vis, 2), lm_prefill_gflops=round(pre, 2),
                          decode_gflops=round(dec, 2), total_gflops=round(vis + pre + dec, 2),
                          vision_share=round(vis / tot[nm], 6),
                          lm_prefill_share=round(pre / tot[nm], 6),
                          decode_share=round(dec / tot[nm], 6))
    ratio = tot["lingshu_32b"] / tot["lingshu_7b"]
    return dict(
        name="N2 -- rebuild the FLOP model from measured parameter counts; reproduce R32_derived and "
             "extract the per-stage shares",
        source="results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json",
        operating_point=dict(prompt_tok=T, image_tok_or_patches=dict(M=M, P_patches=PP),
                             gen_tok_7b=g["G_7b"], gen_tok_32b=g["G_32b"],
                             note="cap320 open-text geometry, the repo's single FLOP operating point"),
        rebuilt_component_gflops=comp,
        rebuilt_total_gflops=tot,
        published_total_gflops=FLOPD["flop_model"]["lingshu_7b_gflops"] if isinstance(
            FLOPD["flop_model"].get("lingshu_7b_gflops"), (int, float)) else None,
        reference_totals_cost_floor_N2=CF["null_tests"]["N2"]["rebuilt_total_gflops"],
        max_abs_dev_component_gflops=round(dev, 4),
        max_abs_dev_total_gflops=round(devt, 4),
        rebuilt_ratio=round(ratio, 4), published_ratio=R32_DERIVED,
        stage_shares=stages,
        formula_note="vision_tower_dense uses P_patches (1121.9) and decode attention is uncalibrated by the causal factor -- both matched to cost_floor_2026-08-10.json:N2 so the two rebuilds are bit-comparable",
        verdict="PASS" if (abs(ratio - R32_DERIVED) < 5e-3 and dev < 0.01) else "FAIL")


N1 = null_n1()
N2 = null_n2()
SH7 = N2["stage_shares"]["lingshu_7b"]
SH32 = N2["stage_shares"]["lingshu_32b"]
print(f"N1 {N1['verdict']}  max|dev| acc {N1['max_abs_dev_accuracy']:.2e} "
      f"cost {N1['max_abs_dev_cost_flops']:.2e}  ratio {N1['macro_ratio_accmax_vs_direct']}")
print(f"N2 {N2['verdict']}  R32_derived rebuilt {N2['rebuilt_ratio']}  "
      f"7B stage shares vis/pre/dec = {SH7['vision_share']:.4f}/{SH7['lm_prefill_share']:.4f}/"
      f"{SH7['decode_share']:.4f}  32B = {SH32['vision_share']:.4f}/{SH32['lm_prefill_share']:.4f}/"
      f"{SH32['decode_share']:.4f}")


# =================================================================================================
# 2.  PER-ITEM RE-DERIVATION (needed for Q1: the joint distribution of draws x escalation)
# =================================================================================================
print("\nre-deriving per-item draw counts and escalation flags (CPU, ~2 min) ...")
import integrated_method as IM          # noqa: E402
import integrated_pandora as IP         # noqa: E402
import beat32b_more as BB               # noqa: E402
import beat32b_fusion as BF             # noqa: E402
import pandora_controller as PC         # noqa: E402
import paper_baselines as PB            # noqa: E402
from sklearn.isotonic import IsotonicRegression      # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

DISJOINT = "ckpts/train/lora_verifier_disjoint"
IM.OPEN_VERIFIER_DIR = DISJOINT
BB.OPEN_VERIFIER_DIR = DISJOINT
IP.ADAPTER = DISJOINT


def cascade_peritem(ok_cheap, ok_strong, gate, K=NFOLD):
    """Literal copy of paper_baselines.cascade_persample, returning the per-ITEM escalate flag."""
    n = len(ok_cheap)
    ok, esc = np.zeros(n), np.zeros(n, bool)
    for f in range(K):
        te = np.array([i % K == f for i in range(n)])
        tr = ~te
        if tr.sum() < 2 or te.sum() < 1:
            continue
        tau = IM.pick_tau_isocost(ok_cheap[tr], ok_strong[tr], gate[tr], ok_strong[tr].mean())
        e = gate[te] < tau
        ok[te] = np.where(e, ok_strong[te], ok_cheap[te])
        esc[te] = e
    return ok, esc


def pandora_peritem(rows, target_acc, K=NFOLD):
    """Literal copy of paper_baselines.pandora_persample, returning per-ITEM N and escalate flag."""
    n = len(rows)
    raw = [r["scores"] for r in rows]
    sl = [r["sl"] for r in rows]
    strong = np.array([r["strong"] for r in rows], float)
    Nmax = min(len(s) for s in raw)
    N_out, esc_out, ok_out = np.zeros(n), np.zeros(n), np.zeros(n)
    for f in range(K):
        te = np.array([i % K == f for i in range(n)])
        tr = ~te
        tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
        if len(tr_idx) < 2 or len(te_idx) < 1:
            continue
        xs = np.concatenate([raw[i][:Nmax] for i in tr_idx])
        ys = np.concatenate([np.array(sl[i][:Nmax], float) for i in tr_idx])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(xs, ys)
        train_pool = iso.predict(xs)
        q = float(strong[tr_idx].mean())
        tr_cal = {i: iso.predict(np.asarray(raw[i][:Nmax], float)) for i in tr_idx}
        best_iso, best_max = None, (-1.0, None, None, None)
        for lam in PC.LAMS:
            zc = PC.zeta_cheap(train_pool, lam)
            zs = PC.zeta_strong(q, lam)
            Ns = es = oks = 0.0
            for i in tr_idx:
                Nk, e, ok = PC.run_pandora(raw[i][:Nmax], tr_cal[i], sl[i][:Nmax], strong[i], zc, zs)
                Ns += Nk; es += e; oks += ok
            m = len(tr_idx)
            acc = oks / m
            flops = (Ns / m) * PC.C_CHEAP_F + (es / m) * PC.C_STRONG_F
            if acc > best_max[0]:
                best_max = (acc, lam, zc, zs)
            if acc >= target_acc - IP.ISO_TOL and (best_iso is None or flops < best_iso[0]):
                best_iso = (flops, lam, zc, zs)
        _, lam, zc, zs = best_iso if best_iso is not None else best_max
        for i in te_idx:
            cal = iso.predict(np.asarray(raw[i][:Nmax], float))
            Nk, e, ok = PC.run_pandora(raw[i][:Nmax], cal, sl[i][:Nmax], strong[i], zc, zs)
            N_out[i], esc_out[i], ok_out[i] = Nk, e, ok
    return ok_out, N_out, esc_out.astype(bool)


def f10_peritem(dskey, K=BB.K):
    """Literal copy of method_final_mmmu_corrected.f10_persample, returning per-ITEM take-7B flag."""
    d = BB.open_features(dskey)
    ok7, ok32, X = d["ok7"], d["ok32"], d["X"]
    n = len(ok7)
    ii = np.arange(n)
    team = np.zeros(n)
    took7 = np.zeros(n, bool)
    for f in range(K):
        te = ii % K == f
        tr = ~te
        mdl = LogisticRegression(max_iter=500, C=1.0).fit(X[tr], ok7[tr])
        g_tr = mdl.predict_proba(X[tr])[:, 1]
        g_te = mdl.predict_proba(X[te])[:, 1]
        grid = np.unique(np.quantile(g_tr, np.linspace(0, 1, 41)))
        best_t, best_a = 1.0, -1.0
        for t in grid:
            keep = g_tr >= t
            a = np.where(keep, ok7[tr], ok32[tr]).mean()
            if a > best_a:
                best_a, best_t = a, t
        keep_te = g_te >= best_t
        team[te] = np.where(keep_te, ok7[te], ok32[te])
        took7[te] = keep_te
    return team, took7


PERITEM = {}

# ---- MCQ: the compute-lean margin cascade (per item) + the F8 certified veto on PMC ----
# EXACTLY the loaders paper_baselines.build_cells uses, so the per-item vectors are the deployed ones.
for c, closed in [("PMC_VQA", None), ("SLAKE_closed", "SLAKE"),
                  ("VQA_RAD_closed", "YESNO"), ("PATH_VQA_closed", "YESNO")]:
    d = IM.mcq_closed(c.split("_closed")[0], closed)
    _ok, esc = cascade_peritem(d["ok7"], d["ok32"], d["margin"])
    PERITEM[c] = dict(cl_esc=esc, cl_N=np.ones(len(esc)))
d = IM.mcq_medxpert()
_ok, esc = cascade_peritem(d["ok7"], d["ok32"], d["margin"])
PERITEM["MedXpertQA-MM"] = dict(cl_esc=esc, cl_N=np.ones(len(esc)))
BF.RNG = np.random.default_rng(0)
ok_f8, veto = BB.f8_veto(BF.mcq("PMC_VQA"))
PERITEM["PMC_VQA"]["am2_esc"] = ~veto
PERITEM["PMC_VQA"]["am2_N"] = np.ones(len(veto))
for c in MCQ[1:]:
    # accuracy-max on the other four MCQ cells IS always-32B-direct: no cheap leg at all
    n_ = len(PERITEM[c]["cl_esc"])
    PERITEM[c]["am2_esc"] = np.ones(n_, bool)
    PERITEM[c]["am2_N"] = np.zeros(n_)

# ---- OPEN: Pandora per-item draw count + its own escalation, and the F10 escalation set ----
for c in OPEN:
    dskey = OPEN_KEY[c]
    rows = IP.load_open_rows(dskey)
    fixed = IM.open_bestof8(dskey)
    a_fix, _esc_fix = IM.heldout(fixed["ok7"], fixed["ok32"], fixed["gate"])
    _ok, Nv, escv = pandora_peritem(rows, a_fix)
    _team, took7 = f10_peritem(dskey)
    assert len(took7) == len(Nv) == len(fixed["ok7"]), (c, len(took7), len(Nv))
    PERITEM[c] = dict(cl_esc=escv, cl_N=Nv, am2_esc=~took7, am2_N=Nv,
                      sel_ok=np.asarray(fixed["ok7"], float),     # deployed best-of-8 verifier PICK
                      greedy_ok=np.array([r["greedy"] for r in rows], float),
                      oracle8_ok=np.array([1.0 if max(r["sl"]) else 0.0 for r in rows], float),
                      strong_ok=np.array([r["strong"] for r in rows], float))


def null_n3():
    """Assert the per-item re-derivation reproduces the PUBLISHED aggregate escalation rates,
    meanN, and the F8 veto rate."""
    rows, dev = {}, []
    pub_esc = PUB["escalation"]["per_cell"]
    lens = {c: dict(peritem=int(len(PERITEM[c]["cl_esc"])), eval_vector=int(NITEM[c]))
            for c in CELLS}
    len_ok = all(v["peritem"] == v["eval_vector"] for v in lens.values())
    for c in CELLS:
        r = dict(cl_esc_recomputed=round(float(PERITEM[c]["cl_esc"].mean()), 6),
                 cl_esc_published=pub_esc[c])
        dev.append(abs(r["cl_esc_recomputed"] - r["cl_esc_published"]))
        if c in OPEN:
            r["meanN_recomputed"] = round(float(PERITEM[c]["cl_N"].mean()), 6)
            r["meanN_published"] = PUB["open_cell_detail"][c]["meanN"]
            dev.append(abs(r["meanN_recomputed"] - r["meanN_published"]))
            r["am2_esc_recomputed"] = round(float(PERITEM[c]["am2_esc"].mean()), 6)
            r["am2_esc_published"] = PUB["open_cell_detail"][c]["am2_esc"]
            dev.append(abs(r["am2_esc_recomputed"] - r["am2_esc_published"]))
        r["am2_esc_from_arm_decomposition"] = ARMDEC[c]["method_accuracy_max_veto"]["n_32b"]
        dev.append(abs(float(PERITEM[c]["am2_esc"].mean()) -
                       ARMDEC[c]["method_accuracy_max_veto"]["n_32b"]))
        if c in OPEN:
            dev.append(abs(float(PERITEM[c]["am2_N"].mean()) -
                           ARMDEC[c]["method_accuracy_max_veto"]["n_gen7"]))
        rows[c] = r
    m = max(dev)
    return dict(
        name="N3 -- per-item draw counts and escalation flags reproduce the published aggregates",
        method="literal re-implementations of paper_baselines.cascade_persample / .pandora_persample "
               "and method_final_mmmu_corrected.f10_persample that additionally return the per-item "
               "vectors; beat32b_more.f8_veto already returns its per-item veto flag",
        verifier_dir=DISJOINT,
        per_cell=rows,
        item_counts=lens,
        item_counts_match_eval_vectors=bool(len_ok),
        max_abs_dev=round(m, 6),
        verdict="PASS" if (m < 1e-3 and len_ok) else "FAIL",
        note="tolerance 1e-3 because the published values are rounded to 4 dp")


N3 = null_n3()
print(f"N3 {N3['verdict']}  max|dev| {N3['max_abs_dev']:.2e}")


# =================================================================================================
# 3.  Q0 -- THE (CELL x STAGE) COST DECOMPOSITION
# =================================================================================================
SEL_HEAD_PARAMS = 918017 * 8            # measured: ckpts/train/genframe_head_ens8/head_seed*.pt
LORA_PARAMS = 47589376                  # measured: adapter_model.safetensors header
G7_TOTAL_GFLOPS = N2["rebuilt_total_gflops"]["lingshu_7b"]
SEL_HEAD_GFLOPS = 2 * SEL_HEAD_PARAMS / 1e9     # one forward per candidate, dense MLP on a pooled vector

STAGES = ["cheap_gen_vision", "cheap_gen_lm_prefill", "cheap_gen_decode",
          "verifier_vision", "verifier_lm_prefill", "verifier_decode",
          "selector_head", "strong_vision", "strong_lm_prefill", "strong_decode"]


def stage_split(n_gen7, n_ver7, n_32b, r32=R32_CHARGED, sel_forwards=0.0):
    """Split a (n_gen7, n_ver7, n_32b) arm into per-stage FLOP-eq, in units of ONE 7B forward.
    The per-stage SHARES are MEASURED (N2, from parameter counts + token geometry); the per-forward
    TOTALS are the costing convention's constants (1.0 for a 7B forward, r32 for a 32B forward)."""
    return {
        "cheap_gen_vision":     n_gen7 * SH7["vision_share"],
        "cheap_gen_lm_prefill": n_gen7 * SH7["lm_prefill_share"],
        "cheap_gen_decode":     n_gen7 * SH7["decode_share"],
        "verifier_vision":      n_ver7 * SH7["vision_share"],
        "verifier_lm_prefill":  n_ver7 * SH7["lm_prefill_share"],
        "verifier_decode":      n_ver7 * SH7["decode_share"],
        "selector_head":        sel_forwards * SEL_HEAD_GFLOPS / G7_TOTAL_GFLOPS,
        "strong_vision":        n_32b * r32 * SH32["vision_share"],
        "strong_lm_prefill":    n_32b * r32 * SH32["lm_prefill_share"],
        "strong_decode":        n_32b * r32 * SH32["decode_share"],
    }


def decompose(arm, r32=R32_CHARGED):
    per_cell, dev = {}, []
    for c in CELLS:
        a = ARMDEC[c][arm]
        s = stage_split(a["n_gen7"], a["n_ver7"], a["n_32b"], r32)
        tot = sum(s.values())
        if abs(r32 - R32_CHARGED) < 1e-9:
            dev.append(abs(tot - PCC[c][arm]["flops"]))
        per_cell[c] = dict(n=NITEM[c], n_gen7=a["n_gen7"], n_ver7=a["n_ver7"], n_32b=a["n_32b"],
                           stages={k: round(v, 5) for k, v in s.items()},
                           total_flopeq=round(tot, 5),
                           x_direct=round(tot / r32, 5),
                           published_total_flopeq=PCC[c][arm]["flops"])
    macro = {k: float(np.mean([per_cell[c]["stages"][k] for c in CELLS])) for k in STAGES}
    mt = sum(macro.values())
    return dict(per_cell=per_cell,
                macro_stages={k: round(v, 5) for k, v in macro.items()},
                macro_stage_pct={k: round(100 * v / mt, 3) for k, v in macro.items()},
                macro_total_flopeq=round(mt, 5),
                macro_x_direct=round(mt / r32, 5),
                max_abs_dev_vs_published_per_cell=round(max(dev), 6) if dev else None)


DEC_AM = decompose("method_accuracy_max_veto")
DEC_CL = decompose("method_compute_lean")
DEC_AM_D = decompose("method_accuracy_max_veto", R32_DERIVED)

N4 = dict(name="N4 -- the stage decomposition re-sums to the published per-cell as-charged cost",
          max_abs_dev_vs_published_per_cell_flops=DEC_AM["max_abs_dev_vs_published_per_cell"],
          macro_x_direct_rebuilt=DEC_AM["macro_x_direct"],
          macro_x_direct_published=PUB["ratios_macro"]["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"],
          verdict="PASS" if DEC_AM["max_abs_dev_vs_published_per_cell"] < 1e-3 else "FAIL")
print(f"N4 {N4['verdict']}  stage re-sum max|dev| {N4['max_abs_dev_vs_published_per_cell_flops']:.2e}"
      f"  macro {DEC_AM['macro_x_direct']}x")

# grouped view: which stage families dominate
def grouped(dec):
    m = dec["macro_stages"]
    tot = dec["macro_total_flopeq"]
    grp = {
        "cheap_leg_7B_generation": m["cheap_gen_vision"] + m["cheap_gen_lm_prefill"] + m["cheap_gen_decode"],
        "verifier_7B_scoring": m["verifier_vision"] + m["verifier_lm_prefill"] + m["verifier_decode"],
        "selector_head": m["selector_head"],
        "escalation_32B": m["strong_vision"] + m["strong_lm_prefill"] + m["strong_decode"],
    }
    sub = {
        "ALL_vision_towers": m["cheap_gen_vision"] + m["verifier_vision"] + m["strong_vision"],
        "ALL_lm_prefill": m["cheap_gen_lm_prefill"] + m["verifier_lm_prefill"] + m["strong_lm_prefill"],
        "ALL_decode": m["cheap_gen_decode"] + m["verifier_decode"] + m["strong_decode"],
    }
    return dict(by_leg={k: dict(flopeq=round(v, 5), pct=round(100 * v / tot, 2)) for k, v in grp.items()},
                by_phase={k: dict(flopeq=round(v, 5), pct=round(100 * v / tot, 2)) for k, v in sub.items()},
                total_flopeq=round(tot, 5))


# =================================================================================================
# 4.  Q1 -- COMPUTE SPENT ON QUESTIONS THAT ESCALATE ANYWAY
# =================================================================================================
def wasted(arm_tag):
    """Per item: the 7B-side (cheap-leg + verifier) FLOP-eq that is spent and then THROWN AWAY
    because the item escalates to the 32B.  Computed on the JOINT per-item (N, escalate), not on
    meanN x esc_rate."""
    per_cell = {}
    for c in CELLS:
        Nv = PERITEM[c][f"{arm_tag}_N"].astype(float)
        ev = PERITEM[c][f"{arm_tag}_esc"].astype(bool)
        # 7B-side cost per item, as-charged: N generations + (open only) N verifier forwards
        ver = Nv if c in OPEN else np.zeros_like(Nv)
        cheap_i = Nv * 1.0 + ver * 1.0
        strong_i = ev.astype(float) * R32_CHARGED
        tot_i = cheap_i + strong_i
        w = float((cheap_i * ev).mean())
        t = float(tot_i.mean())
        fac = float(cheap_i.mean()) * float(ev.mean())     # the factorised (naive) estimate
        per_cell[c] = dict(
            n=NITEM[c],
            esc_rate=round(float(ev.mean()), 6),
            meanN=round(float(Nv.mean()), 6),
            meanN_given_escalate=round(float(Nv[ev].mean()) if ev.any() else 0.0, 6),
            meanN_given_keep=round(float(Nv[~ev].mean()) if (~ev).any() else 0.0, 6),
            cheap_side_flopeq=round(float(cheap_i.mean()), 6),
            total_flopeq=round(t, 6),
            wasted_flopeq=round(w, 6),
            wasted_pct_of_cell=round(100 * w / t, 3) if t > 0 else 0.0,
            wasted_factorised_flopeq=round(fac, 6),
            factorisation_understates_by=round(w - fac, 6))
    W = float(np.mean([per_cell[c]["wasted_flopeq"] for c in CELLS]))
    T = float(np.mean([per_cell[c]["total_flopeq"] for c in CELLS]))
    Wsw = float(sum(per_cell[c]["wasted_flopeq"] * NITEM[c] for c in CELLS) / sum(NITEM.values()))
    Tsw = float(sum(per_cell[c]["total_flopeq"] * NITEM[c] for c in CELLS) / sum(NITEM.values()))
    return dict(per_cell=per_cell,
                macro_wasted_flopeq=round(W, 5), macro_total_flopeq=round(T, 5),
                macro_wasted_pct=round(100 * W / T, 3),
                macro_wasted_as_x_direct=round(W / R32_CHARGED, 5),
                sample_weighted_wasted_pct=round(100 * Wsw / Tsw, 3),
                open_only_wasted_pct=round(
                    100 * float(np.mean([per_cell[c]["wasted_flopeq"] for c in OPEN])) /
                    float(np.mean([per_cell[c]["total_flopeq"] for c in OPEN])), 3),
                mcq_only_wasted_pct=round(
                    100 * float(np.mean([per_cell[c]["wasted_flopeq"] for c in MCQ])) /
                    float(np.mean([per_cell[c]["total_flopeq"] for c in MCQ])), 3))


WASTE_AM = wasted("am2")
WASTE_CL = wasted("cl")
print(f"\nQ1 accuracy-max: {WASTE_AM['macro_wasted_pct']:.1f}% of macro FLOP-eq is 7B-side work on "
      f"items that escalate anyway ({WASTE_AM['macro_wasted_flopeq']:.3f} FLOP-eq = "
      f"{WASTE_AM['macro_wasted_as_x_direct']:.3f}x one 32B-direct pass)")
print(f"Q1 compute-lean: {WASTE_CL['macro_wasted_pct']:.1f}%")


# =================================================================================================
# 5.  BOOTSTRAP MACHINERY (one shared resample stream per cell)
# =================================================================================================
def cell_pattern_boot(mat, nboot, rng):
    """Paired item bootstrap via the exact multinomial-over-patterns shortcut (identical in
    distribution to gathering n resampled rows).  mat: (n_items, n_systems)."""
    pats, cnt = np.unique(mat, axis=0, return_counts=True)
    n = mat.shape[0]
    draws = rng.multinomial(n, cnt / n, size=nboot)
    return (draws @ pats) / n


rng = np.random.default_rng(SEED)
ARM_IDX = {a: i for i, a in enumerate(DEPLOYABLE)}
BOOT = {}          # cell -> (nboot, n_arms) resampled accuracies, PAIRED within the cell
ACC = {}           # cell -> arm -> point accuracy
for c in CELLS:
    mat = np.stack([OK[(c, a)] for a in DEPLOYABLE], axis=1)
    BOOT[c] = cell_pattern_boot(mat, NBOOT, rng)
    ACC[c] = {a: float(OK[(c, a)].mean()) for a in DEPLOYABLE}

BASE_I = ARM_IDX[BASE]
DBOOT = {c: BOOT[c] - BOOT[c][:, [BASE_I]] for c in CELLS}     # per-cell delta vs always-32B-direct


def macro_ci(assign):
    """assign: cell -> arm.  Returns point delta vs always-32B-direct + 95% paired-bootstrap CI."""
    d = np.mean([DBOOT[c][:, ARM_IDX[assign[c]]] for c in CELLS], axis=0)
    pt = float(np.mean([ACC[c][assign[c]] - ACC[c][BASE] for c in CELLS]))
    return pt, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def macro_cost(assign, r32=R32_CHARGED, key="flops"):
    if key == "flops":
        return float(np.mean([sum(stage_split(ARMDEC[c][assign[c]]["n_gen7"],
                                              ARMDEC[c][assign[c]]["n_ver7"],
                                              ARMDEC[c][assign[c]]["n_32b"], r32).values())
                              for c in CELLS]))
    return float(np.mean([PCC[c][assign[c]][key] for c in CELLS]))


def guardrail(assign):
    out, flags = {}, []
    for c in CELLS:
        d = DBOOT[c][:, ARM_IDX[assign[c]]]
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        pt = ACC[c][assign[c]] - ACC[c][BASE]
        out[c] = dict(delta=round(pt, 4), lo=round(lo, 4), hi=round(hi, 4), worse_sig=bool(hi < 0))
        if hi < 0:
            flags.append(c)
    return out, flags


def point(name, assign, kind):
    pt, lo, hi = macro_ci(assign)
    acc = float(np.mean([ACC[c][assign[c]] for c in CELLS]))
    g, fl = guardrail(assign)
    touches32 = any(ARMDEC[c][assign[c]]["n_32b"] > 0 for c in CELLS)
    uses7 = any(ARMDEC[c][assign[c]]["n_gen7"] > 0 for c in CELLS)
    usesver = any(ARMDEC[c][assign[c]]["n_ver7"] > 0 for c in CELLS)
    return dict(
        name=name, kind=kind, assignment=dict(assign),
        macro_acc=round(acc, 4), delta_vs_direct=round(pt, 4), lo=round(lo, 4), hi=round(hi, 4),
        meets_constraint=bool(lo >= -TIE_TOL),
        not_significantly_worse=bool(hi > 0 or lo > 0 or (lo < 0 < hi)),
        cost=dict(
            as_charged_R32_4p57=dict(flopeq=round(macro_cost(assign), 4),
                                     x_direct=round(macro_cost(assign) / R32_CHARGED, 4)),
            derived_R32_3p816=dict(flopeq=round(macro_cost(assign, R32_DERIVED), 4),
                                   x_direct=round(macro_cost(assign, R32_DERIVED) / R32_DERIVED, 4)),
            lat_par_ms=round(macro_cost(assign, key="lat_par_ms"), 1),
            lat_seq_ms=round(macro_cost(assign, key="lat_seq_ms"), 1),
            energy_j=round(macro_cost(assign, key="energy_j"), 1)),
        footprint=footprint(touches32, uses7, usesver),
        guardrail=g, guardrail_flags=fl)


# =================================================================================================
# 6.  Q4 -- MODEL FOOTPRINT (measured on disk + measured VRAM)
# =================================================================================================
def _safetensors_index(path):
    d = json.load(open(path))
    return d["metadata"]["total_size"], len(set(d["weight_map"].values()))


HUB = "/data/dan/hf_cache/hub"
SNAP7 = os.path.join(HUB, "models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
                          "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")
SNAP32 = os.path.join(HUB, "models--lingshu-medical-mllm--Lingshu-32B/snapshots/"
                           "36b98277cacb60db86f34b75ce0540b1ea35183c")
B7, S7 = _safetensors_index(os.path.join(SNAP7, "model.safetensors.index.json"))
B32, S32 = _safetensors_index(os.path.join(SNAP32, "model.safetensors.index.json"))
with open(os.path.join(REPO, "ckpts/train/lora_verifier_disjoint/adapter_model.safetensors"), "rb") as f:
    _n = struct.unpack("<Q", f.read(8))[0]
    _h = json.loads(f.read(_n))
LORA_BYTES = os.path.getsize(os.path.join(REPO, "ckpts/train/lora_verifier_disjoint/"
                                                "adapter_model.safetensors"))
LORA_PARAMS = sum(int(np.prod(v["shape"])) for k, v in _h.items() if k != "__metadata__")

P7 = FLOPD["parameter_counts"]["lingshu_7b"]["total"]
P32 = FLOPD["parameter_counts"]["lingshu_32b"]["total"]
assert B7 == FLOPD["parameter_counts"]["lingshu_7b"]["index_total_size_bytes"]
assert B32 == FLOPD["parameter_counts"]["lingshu_32b"]["index_total_size_bytes"]

VRAM_S1 = VRAM["scenarios"]["S1_lingshu7b_direct_mcq"]
VRAM_S2 = VRAM["scenarios"]["S2_lingshu32b_direct_mcq"]
VRAM_S3 = VRAM["scenarios"]["S3_lingshu7b_plus_lora_verifier"]
VRAM_S4 = VRAM["scenarios"]["S4_opentext_bestof8_full_arm"]


def footprint(touches32, uses7, usesver):
    par = 0
    byt = 0
    parts = []
    if uses7 or usesver:
        par += P7; byt += B7; parts.append("Lingshu-7B base")
    if usesver:
        par += LORA_PARAMS; byt += LORA_BYTES; parts.append("verifier LoRA (same base, adapter only)")
    if touches32:
        par += P32; byt += B32; parts.append("Lingshu-32B")
    return dict(resident=parts,
                params=int(par), params_B=round(par / 1e9, 4),
                weight_bytes=int(byt), weight_GiB=round(byt / 2 ** 30, 4),
                needs_32B=bool(touches32),
                fits_one_A100_80GB_weights_only=bool(byt / 2 ** 30 < 80.0))


# =================================================================================================
# 7.  Q2 -- IS THE OPEN-TEXT MACHINERY WORTH RUNNING AT ALL?
# =================================================================================================
def q2():
    per_cell = {}
    for c in OPEN:
        p = PERITEM[c]
        mN = float(p["cl_N"].mean())
        a = ARMDEC[c]["method_accuracy_max_veto"]
        rows_ = dict(
            n=NITEM[c],
            always_7b_greedy=dict(acc=round(float(p["greedy_ok"].mean()), 4),
                                  flopeq=1.0, x_direct=round(1.0 / R32_CHARGED, 4)),
            bestofN_verifier_select_NO_escalation=dict(
                acc=round(float(p["sel_ok"].mean()), 4),
                meanN=round(mN, 4),
                flopeq=round(2.0 * mN, 4), x_direct=round(2.0 * mN / R32_CHARGED, 4),
                note="the open arm's own selected answer, escalation switched off"),
            bestof8_fixed_verifier_select=dict(
                acc=round(float(p["sel_ok"].mean()), 4), meanN=8.0, flopeq=16.0,
                x_direct=round(16.0 / R32_CHARGED, 4),
                note="the paper's fixed BO8 charge; the SELECTED answer is the same pick"),
            oracle_of_8=dict(acc=round(float(p["oracle8_ok"].mean()), 4),
                             note="free upper bound on any selector over this pool"),
            deployed_open_arm_accuracy_max=dict(
                acc=round(ACC[c]["method_accuracy_max_veto"], 4),
                meanN=round(a["n_gen7"], 4), esc=round(a["n_32b"], 4),
                flopeq=round(sum(stage_split(a["n_gen7"], a["n_ver7"], a["n_32b"]).values()), 4),
                x_direct=round(sum(stage_split(a["n_gen7"], a["n_ver7"], a["n_32b"]).values())
                               / R32_CHARGED, 4)),
            always_32b_direct=dict(acc=round(ACC[c][BASE], 4), flopeq=R32_CHARGED, x_direct=1.0))
        # paired deltas of every open policy vs always-32B-direct, on this cell
        i7 = ARM_IDX["always_7b"]
        rows_["deltas_vs_direct"] = {}
        for lab, vec in [("always_7b_greedy", p["greedy_ok"]),
                         ("bestofN_verifier_select_NO_escalation", p["sel_ok"]),
                         ("deployed_open_arm_accuracy_max", OK[(c, "method_accuracy_max_veto")])]:
            mat = np.stack([vec, OK[(c, BASE)]], axis=1)
            b = cell_pattern_boot(mat, NBOOT, np.random.default_rng(SEED + zlib.crc32((c + lab).encode()) % 10000))
            d = b[:, 0] - b[:, 1]
            rows_["deltas_vs_direct"][lab] = dict(
                delta=round(float(vec.mean() - OK[(c, BASE)].mean()), 4),
                lo=round(float(np.percentile(d, 2.5)), 4),
                hi=round(float(np.percentile(d, 97.5)), 4))
        per_cell[c] = rows_
    # macro consequence: switch the open arm OFF
    shipped = {c: "method_accuracy_max_veto" for c in CELLS}
    mcq_only = dict(shipped)
    for c in OPEN:
        mcq_only[c] = BASE
    open_off_lean = {c: ("method_compute_lean" if c in MCQ else BASE) for c in CELLS}
    pts = [point("SHIPPED accuracy-max (all 8 cells)", shipped, "reference -- the shipped arm"),
           point("accuracy-max on MCQ, always-32B-direct on OPEN (open machinery OFF)", mcq_only,
                 "PRE-SPECIFIED arm swap, no eval-side selection"),
           point("compute-lean on MCQ, always-32B-direct on OPEN", open_off_lean,
                 "PRE-SPECIFIED arm swap, no eval-side selection")]
    # shipped vs open-off, paired
    d = np.mean([DBOOT[c][:, ARM_IDX[mcq_only[c]]] - DBOOT[c][:, ARM_IDX[shipped[c]]]
                 for c in CELLS], axis=0)
    swap = dict(delta_open_off_minus_shipped=round(
        float(np.mean([ACC[c][mcq_only[c]] - ACC[c][shipped[c]] for c in CELLS])), 4),
        lo=round(float(np.percentile(d, 2.5)), 4), hi=round(float(np.percentile(d, 97.5)), 4))
    macro_open_arm_flopeq = float(np.mean([sum(stage_split(
        ARMDEC[c]["method_accuracy_max_veto"]["n_gen7"],
        ARMDEC[c]["method_accuracy_max_veto"]["n_ver7"],
        ARMDEC[c]["method_accuracy_max_veto"]["n_32b"]).values()) for c in OPEN]))
    return dict(per_cell=per_cell, operating_points=pts, shipped_vs_open_off=swap,
                open_arm_macro_flopeq_over_3_open_cells=round(macro_open_arm_flopeq, 4),
                open_arm_x_direct=round(macro_open_arm_flopeq / R32_CHARGED, 4),
                open_cells_carry_pct_of_macro_cost=round(
                    100 * 3 * macro_open_arm_flopeq / (8 * DEC_AM["macro_total_flopeq"]), 2))


Q2 = q2()
print(f"\nQ2 open arm over the 3 open cells: {Q2['open_arm_x_direct']:.3f}x one 32B-direct pass; "
      f"open cells carry {Q2['open_cells_carry_pct_of_macro_cost']:.1f}% of the shipped arm's macro cost")
print(f"Q2 switching the open machinery OFF: macro delta {Q2['shipped_vs_open_off']}")


# =================================================================================================
# 8.  Q3 -- THE KNAPSACK
# =================================================================================================
GRID = np.array(np.meshgrid(*[np.arange(6)] * 8, indexing="ij")).reshape(8, -1).T
COSTV = {c: np.array([sum(stage_split(ARMDEC[c][a]["n_gen7"], ARMDEC[c][a]["n_ver7"],
                                      ARMDEC[c][a]["n_32b"]).values()) for a in DEPLOYABLE])
         for c in CELLS}
ACCV = {c: np.array([ACC[c][a] for a in DEPLOYABLE]) for c in CELLS}
NA = len(DEPLOYABLE)
GRID_COST = sum(COSTV[c][GRID[:, i]] for i, c in enumerate(CELLS)) / 8.0


def enumerate_frontier(dboot, costv, accv, tie_tol=TIE_TOL, top=None):
    """Exhaustive 6^8 = 1,679,616 assignment search.  Screen by a normal approximation on the macro
    delta (mean - 1.96 sd), then verify the survivors EXACTLY with the shared bootstrap stream, in
    ascending cost order, and return the cheapest that meets the constraint."""
    mu = {c: dboot[c].mean(axis=0) for c in CELLS}
    sd = {c: dboot[c].std(axis=0) for c in CELLS}
    grid = GRID
    cost = GRID_COST if costv is COSTV else sum(costv[c][grid[:, i]] for i, c in enumerate(CELLS)) / 8.0
    mmu = sum(mu[c][grid[:, i]] for i, c in enumerate(CELLS)) / 8.0
    msd = np.sqrt(sum(sd[c][grid[:, i]] ** 2 for i, c in enumerate(CELLS))) / 8.0
    approx_lo = mmu - 1.96 * msd
    cand = np.where(approx_lo >= -tie_tol - 0.004)[0]            # generous screen margin
    cand = cand[np.argsort(cost[cand])]
    best = None
    checked = 0
    for j in cand:
        a = {c: DEPLOYABLE[grid[j, i]] for i, c in enumerate(CELLS)}
        d = np.mean([dboot[c][:, grid[j, i]] for i, c in enumerate(CELLS)], axis=0)
        lo = float(np.percentile(d, 2.5))
        checked += 1
        if lo >= -tie_tol:
            best = (a, float(cost[j]), lo)
            break
        if checked > 40000:
            break
    return best, checked, len(cand)


print("\nQ3 exhaustive knapsack (eval-visible bound) ...")
BEST_EV, NCHK, NCAND = enumerate_frontier(DBOOT, COSTV, ACCV)
EV_POINT = point("EVAL-VISIBLE minimum-cost assignment", BEST_EV[0],
                 "DIAGNOSTIC UPPER BOUND -- fitted with full eval visibility, NOT achievable")
print(f"   eval-visible bound: {EV_POINT['cost']['as_charged_R32_4p57']['x_direct']}x, "
      f"delta {EV_POINT['delta_vs_direct']} [{EV_POINT['lo']},{EV_POINT['hi']}]")


def nested_cv_knapsack(fold_seed, cells_ok=None, tie_tol=TIE_TOL):
    """HONEST: for each outer fold, solve the min-cost knapsack on the OTHER folds only (using a
    train-fold bootstrap for the constraint), then apply the resulting per-cell arm to the held-out
    fold.  Concatenate held-out outcomes -> one honest per-item delivered vector per cell."""
    r = np.random.default_rng(fold_seed)
    src = cells_ok if cells_ok is not None else {(c, a): OK[(c, a)] for c in CELLS for a in DEPLOYABLE}
    folds = {c: r.integers(0, NFOLD, size=NITEM[c]) for c in CELLS}
    deliv = {c: np.zeros(NITEM[c]) for c in CELLS}
    picks = {c: [] for c in CELLS}
    rb = np.random.default_rng(fold_seed + 7)
    for f in range(NFOLD):
        dboot_tr, ok_tr = {}, {}
        for c in CELLS:
            tr = folds[c] != f
            mat = np.stack([src[(c, a)][tr] for a in DEPLOYABLE], axis=1)
            b = cell_pattern_boot(mat, 2000, rb)
            dboot_tr[c] = b - b[:, [BASE_I]]
            ok_tr[c] = mat
        best, _, _ = enumerate_frontier(dboot_tr, COSTV, None, tie_tol)
        if best is None:
            best = ({c: BASE for c in CELLS}, macro_cost({c: BASE for c in CELLS}), 0.0)
        for c in CELLS:
            te = folds[c] == f
            deliv[c][te] = src[(c, best[0][c])][te]
            picks[c].append(best[0][c])
    return deliv, picks, folds


print("Q3 honest nested-CV knapsack over %d fold seeds ..." % N_FOLD_SEEDS)
nested_rows = []
for s in range(N_FOLD_SEEDS):
    deliv, picks, _ = nested_cv_knapsack(SEED + 1000 * s)
    rboot = np.random.default_rng(SEED + 50000 + s)
    dv = []
    accs = []
    for c in CELLS:
        mat = np.stack([deliv[c], OK[(c, BASE)]], axis=1)
        b = cell_pattern_boot(mat, NBOOT, rboot)
        dv.append(b[:, 0] - b[:, 1])
        accs.append(float(deliv[c].mean()))
    d = np.mean(dv, axis=0)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    pt = float(np.mean([accs[i] - ACC[c][BASE] for i, c in enumerate(CELLS)]))
    # cost = the fold-weighted mixture of the arms actually applied
    cst = float(np.mean([np.mean([COSTV[c][ARM_IDX[a]] for a in picks[c]]) for c in CELLS]))
    cst_d = float(np.mean([np.mean([sum(stage_split(ARMDEC[c][a]["n_gen7"], ARMDEC[c][a]["n_ver7"],
                                                    ARMDEC[c][a]["n_32b"], R32_DERIVED).values())
                                    for a in picks[c]]) for c in CELLS]))
    touches32 = any(ARMDEC[c][a]["n_32b"] > 0 for c in CELLS for a in picks[c])
    uses7 = any(ARMDEC[c][a]["n_gen7"] > 0 for c in CELLS for a in picks[c])
    usesver = any(ARMDEC[c][a]["n_ver7"] > 0 for c in CELLS for a in picks[c])
    nested_rows.append(dict(
        seed=SEED + 1000 * s,
        macro_acc=round(float(np.mean(accs)), 4), delta_vs_direct=round(pt, 4),
        lo=round(lo, 4), hi=round(hi, 4), meets_constraint=bool(lo >= -TIE_TOL),
        x_direct_as_charged=round(cst / R32_CHARGED, 4),
        x_direct_derived=round(cst_d / R32_DERIVED, 4),
        footprint=footprint(touches32, uses7, usesver),
        picks={c: sorted(set(picks[c])) for c in CELLS}))
    print(f"   seed {s}: {nested_rows[-1]['x_direct_as_charged']}x  d={nested_rows[-1]['delta_vs_direct']} "
          f"[{lo:.4f},{hi:.4f}] tie={nested_rows[-1]['meets_constraint']}")

nested_x = np.array([r["x_direct_as_charged"] for r in nested_rows])
nested_d = np.array([r["delta_vs_direct"] for r in nested_rows])
nested_lo = np.array([r["lo"] for r in nested_rows])
NESTED = dict(
    kind="FULLY HONEST -- the per-cell arm is chosen inside the training folds only; this is the "
         "number to quote",
    n_fold_seeds=N_FOLD_SEEDS,
    x_direct_as_charged_mean=round(float(nested_x.mean()), 4),
    x_direct_as_charged_sd=round(float(nested_x.std(ddof=1)), 4),
    x_direct_as_charged_min=round(float(nested_x.min()), 4),
    x_direct_as_charged_max=round(float(nested_x.max()), 4),
    delta_mean=round(float(nested_d.mean()), 4),
    lo_mean=round(float(nested_lo.mean()), 4),
    seeds_meeting_constraint=int(sum(r["meets_constraint"] for r in nested_rows)),
    per_seed=nested_rows)


# ---- permutation / synthetic nulls --------------------------------------------------------------
def null_random_arm(n_draws=N_NULL):
    """P1: replace the train-fold KNAPSACK PICK with a uniformly random arm per cell, everything
    else identical.  Shows what 'no selection skill' produces on the same machinery."""
    out = []
    for k in range(n_draws):
        r = np.random.default_rng(SEED + 900000 + k)
        folds = {c: r.integers(0, NFOLD, size=NITEM[c]) for c in CELLS}
        deliv = {c: np.zeros(NITEM[c]) for c in CELLS}
        picks = {c: [] for c in CELLS}
        for f in range(NFOLD):
            for c in CELLS:
                a = DEPLOYABLE[int(r.integers(0, NA))]
                te = folds[c] == f
                deliv[c][te] = OK[(c, a)][te]
                picks[c].append(a)
        rb = np.random.default_rng(SEED + 950000 + k)
        dv = []
        for c in CELLS:
            mat = np.stack([deliv[c], OK[(c, BASE)]], axis=1)
            b = cell_pattern_boot(mat, 1000, rb)
            dv.append(b[:, 0] - b[:, 1])
        d = np.mean(dv, axis=0)
        lo = float(np.percentile(d, 2.5))
        cst = float(np.mean([np.mean([COSTV[c][ARM_IDX[a]] for a in picks[c]]) for c in CELLS]))
        out.append((float(np.mean([deliv[c].mean() - ACC[c][BASE] for c in CELLS])), lo,
                    cst / R32_CHARGED))
    a = np.array(out)
    return dict(
        name="P1 -- random-arm control (no selection skill), same nested machinery",
        n_draws=n_draws,
        delta_mean=round(float(a[:, 0].mean()), 4),
        delta_p2p5=round(float(np.percentile(a[:, 0], 2.5)), 4),
        delta_p97p5=round(float(np.percentile(a[:, 0], 97.5)), 4),
        pass_rate_of_the_preregistered_constraint=round(float((a[:, 1] >= -TIE_TOL).mean()), 4),
        x_direct_mean=round(float(a[:, 2].mean()), 4),
        reading="if the honest nested-CV point is inside this distribution, the per-cell selection "
                "carries no transferable signal")


def null_equal_accuracy(n_draws=N_NULL):
    """P2: synthesise EVERY arm in a cell as i.i.d. Bernoulli(p_direct) with the cell's real n, so
    every arm truly ties, then run the IDENTICAL honest nested-CV knapsack.  Measures the false-pass
    rate of the pre-registered constraint and the cost the pipeline 'achieves' for free."""
    out = []
    for k in range(n_draws):
        r = np.random.default_rng(SEED + 800000 + k)
        src = {}
        for c in CELLS:
            p = ACC[c][BASE]
            for a in DEPLOYABLE:
                src[(c, a)] = (r.random(NITEM[c]) < p).astype(float)
        deliv, picks, _ = nested_cv_knapsack(SEED + 810000 + k, cells_ok=src)
        rb = np.random.default_rng(SEED + 820000 + k)
        dv = []
        for c in CELLS:
            mat = np.stack([deliv[c], src[(c, BASE)]], axis=1)
            b = cell_pattern_boot(mat, 1000, rb)
            dv.append(b[:, 0] - b[:, 1])
        d = np.mean(dv, axis=0)
        lo = float(np.percentile(d, 2.5))
        cst = float(np.mean([np.mean([COSTV[c][ARM_IDX[a]] for a in picks[c]]) for c in CELLS]))
        out.append((float(np.mean([deliv[c].mean() - src[(c, BASE)].mean() for c in CELLS])), lo,
                    cst / R32_CHARGED))
    a = np.array(out)
    return dict(
        name="P2 -- equal-accuracy synthetic null (every arm i.i.d. Bernoulli(p_direct), same n)",
        n_draws=n_draws,
        delta_mean=round(float(a[:, 0].mean()), 4),
        delta_p2p5=round(float(np.percentile(a[:, 0], 2.5)), 4),
        delta_p97p5=round(float(np.percentile(a[:, 0], 97.5)), 4),
        pass_rate_of_the_preregistered_constraint=round(float((a[:, 1] >= -TIE_TOL).mean()), 4),
        x_direct_mean=round(float(a[:, 2].mean()), 4),
        x_direct_p2p5=round(float(np.percentile(a[:, 2], 2.5)), 4),
        reading="under a TRUE tie the cheapest arm is always feasible, so this measures the FLOOR "
                "the constraint can be pushed to by chance; a real result must be dearer than this "
                "floor OR carry a delta the null cannot produce")


print("\nQ3 nulls ...")
P1 = null_random_arm()
P2 = null_equal_accuracy(60)
print(f"   P1 random-arm: delta {P1['delta_mean']} [{P1['delta_p2p5']},{P1['delta_p97p5']}], "
      f"pass rate {P1['pass_rate_of_the_preregistered_constraint']}, {P1['x_direct_mean']}x")
print(f"   P2 equal-acc null: pass rate {P2['pass_rate_of_the_preregistered_constraint']}, "
      f"floor {P2['x_direct_mean']}x")


# =================================================================================================
# 9.  PRE-SPECIFIED OPERATING POINTS (no eval-side selection at all)
# =================================================================================================
PRESPEC = []
_shipped = {c: "method_accuracy_max_veto" for c in CELLS}
PRESPEC.append(point("always-32B-direct (THE BAR)", {c: BASE for c in CELLS}, "baseline"))
PRESPEC.append(point("always-7B", {c: "always_7b" for c in CELLS}, "cheap floor"))
PRESPEC.append(point("SHIPPED accuracy-max", _shipped, "the shipped arm"))
PRESPEC.append(point("SHIPPED compute-lean", {c: "method_compute_lean" for c in CELLS},
                     "the shipped cheap knob"))
_a = dict(_shipped)
for c in OPEN:
    _a[c] = BASE
PRESPEC.append(point("accuracy-max MCQ + always-32B-direct OPEN", _a,
                     "PRE-SPECIFIED: the open machinery switched off"))
_b = {c: ("method_compute_lean" if c in MCQ else BASE) for c in CELLS}
PRESPEC.append(point("compute-lean MCQ + always-32B-direct OPEN", _b, "PRE-SPECIFIED"))
_c = {c: ("method_accuracy_max_veto" if c == "PMC_VQA" else
          "method_compute_lean" if c in ("SLAKE_closed", "PATH_VQA_closed") else BASE)
      for c in CELLS}
PRESPEC.append(point("veto on PMC + compute-lean on SLAKE-cl/PathVQA-cl + direct elsewhere", _c,
                     "PRE-SPECIFIED from cost_floor_2026-08-10.json's cross-fit picks (an "
                     "ARTIFACT-DERIVED prior, not fitted here)"))

for p in PRESPEC:
    print(f"   {p['name'][:58]:58s} {p['cost']['as_charged_R32_4p57']['x_direct']:>7.4f}x  "
          f"acc {p['macro_acc']:.4f}  d {p['delta_vs_direct']:+.4f} [{p['lo']:+.4f},{p['hi']:+.4f}]  "
          f"tie={p['meets_constraint']}  {p['footprint']['weight_GiB']:.1f} GiB")


# =================================================================================================
# 10.  ASSEMBLE
# =================================================================================================
rep = dict(
    title="ATTACK 1 -- where does every FLOP go, and what is the floor?  A complete (cell x stage) "
          "cost decomposition of the shipped accuracy-max arm, and the minimum-compute frontier "
          "under the new objective (minimise compute subject to macro non-inferiority vs "
          "always-32B-direct).",
    date="2026-08-12",
    reproduce="python3 src/cascade_methods/cost_decomposition.py",
    no_gpu=True,
    no_fabricated_numbers=True,
    objective=dict(
        primary="MINIMISE macro-weighted FLOP-eq as a ratio of one always-32B-direct forward pass",
        secondary="MINIMISE model footprint -- parameters and GiB of weights that must be resident",
        tertiary="parallel latency and energy (measured currencies)",
        constraint=f"paired-bootstrap 95% CI lower bound of (policy - always-32B-direct) on the "
                   f"8-cell macro must be >= {-TIE_TOL}",
        pre_registered=True),
    seed=SEED, n_bootstrap=NBOOT, n_fold_seeds=N_FOLD_SEEDS,
    pool="Variant B (MMMU excluded): 5 benchmarks / 8 cells / n=42,224; CLEAN disjoint verifier",
    numerics_pins=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONHASHSEED="0",
                       tf32="not applicable -- pure numpy/sklearn on CPU, no GPU matmul",
                       bootstrap="paired item-level, one shared resample stream per cell reused by "
                                 "every policy and every baseline",
                       rank_convention="not applicable -- no ranking step"),
    null_tests=dict(N1=N1, N2=N2, N3=N3, N4=N4),
    Q0_cost_decomposition=dict(
        convention="PRIMARY = as-charged (R32 = 4.57, every 7B generation and every verifier forward "
                   "charged one full batch-1 7B pass).  The per-stage SHARES are MEASURED from "
                   "parameter counts + token geometry (N2); the per-forward TOTALS are the "
                   "convention's constants.  The derived-R32 (3.816) view is given alongside and "
                   "makes every ratio WORSE for the method.",
        stage_definitions=dict(
            cheap_gen_vision="Lingshu-7B ViT + patch merger, once per generated sample",
            cheap_gen_lm_prefill="Lingshu-7B LM prefill over the prompt, once per generated sample",
            cheap_gen_decode="Lingshu-7B LM decode + lm_head for the generated tokens",
            verifier_vision="the verifier is a LoRA on the SAME 7B base; its forward re-runs the ViT",
            verifier_lm_prefill="verifier LM prefill over image+question+candidate",
            verifier_decode="verifier scoring head / decode",
            selector_head=f"the frozen 8-seed genframe head, {SEL_HEAD_PARAMS:,} params measured "
                          f"from ckpts/train/genframe_head_ens8/head_seed*.pt.  ZERO in the "
                          f"canonical `disjoint` arm, which uses the LoRA verifier as its selector.",
            strong_vision="Lingshu-32B ViT + merger on escalation",
            strong_lm_prefill="Lingshu-32B LM prefill on escalation",
            strong_decode="Lingshu-32B LM decode on escalation"),
        shipped_accuracy_max_as_charged=DEC_AM,
        shipped_accuracy_max_derived_R32=DEC_AM_D,
        shipped_compute_lean_as_charged=DEC_CL,
        grouped_accuracy_max=grouped(DEC_AM),
        grouped_compute_lean=grouped(DEC_CL),
        lora_adapter_flop_note=dict(
            lora_params=int(LORA_PARAMS), base_params=int(P7),
            lora_fraction_of_base=round(LORA_PARAMS / P7, 6),
            statement="the as-charged convention charges the verifier a FULL 7B forward, which "
                      "already over-covers the adapter's own arithmetic; the LoRA adds "
                      f"{100 * LORA_PARAMS / P7:.2f}% more parameters to the matmul path."),
        caveat="the FLOP model has ONE operating point (cap320 open-text geometry, 326.68 prompt "
               "tokens, 5.6 generated tokens).  The MCQ arm actually runs at MedEvalKit's full "
               "resolution, where the vision share is LARGER -- vram_testtime_2026-08-11.json "
               "measures up to 46,816 vision tokens on MedXpert.  The stage SHARES on MCQ cells are "
               "therefore a lower bound on the vision fraction.  NOT MEASURED: a per-cell FLOP model."),
    Q1_waste_on_escalated_questions=dict(
        question="what fraction of total compute is spent on questions the cascade escalates anyway?",
        method="per ITEM, on the joint (draw count, escalate) distribution -- NOT esc_rate x meanN.  "
               "The adaptive-N controller draws MORE samples on questions it later escalates, so the "
               "factorised product understates the waste; both are reported.",
        shipped_accuracy_max=WASTE_AM,
        shipped_compute_lean=WASTE_CL),
    Q2_is_the_open_arm_worth_running=Q2,
    Q3_minimum_compute_at_parity=dict(
        menu=DEPLOYABLE,
        menu_note="oracle_mode_32b is EXCLUDED -- it is an oracle, not deployable.  6 arms x 8 cells "
                  "= 1,679,616 assignments, enumerated exhaustively.",
        eval_visible_bound=EV_POINT,
        eval_visible_bound_warning="NOT ACHIEVABLE.  Fitted with full eval visibility; selecting the "
                                   "best arm per cell on eval is exactly how a fake win has already "
                                   "been manufactured in this project.  Quote the nested-CV row.",
        honest_nested_cv=NESTED,
        nulls=dict(P1=P1, P2=P2)),
    Q4_footprint=dict(
        measured_on_disk=dict(
            lingshu_7b=dict(params=int(P7), index_total_size_bytes=int(B7), shards=S7,
                            GiB=round(B7 / 2 ** 30, 4), GB=round(B7 / 1e9, 4),
                            path=SNAP7),
            lingshu_32b=dict(params=int(P32), index_total_size_bytes=int(B32), shards=S32,
                             GiB=round(B32 / 2 ** 30, 4), GB=round(B32 / 1e9, 4),
                             path=SNAP32),
            verifier_lora_disjoint=dict(params=int(LORA_PARAMS), bytes=int(LORA_BYTES),
                                        MiB=round(LORA_BYTES / 2 ** 20, 2), dtype="F32",
                                        r=16, alpha=32,
                                        base="lingshu-medical-mllm/Lingshu-7B",
                                        note="an ADAPTER on the generator's own base -- generator "
                                             "and verifier are ONE resident model plus this."),
            frozen_selector_head=dict(params=int(SEL_HEAD_PARAMS), per_seed=918017, seeds=8,
                                      note="not used by the canonical `disjoint` arm")),
        correction_to_the_brief=dict(
            brief_said="always-32B-direct weights ~32.8B params / ~31.5 GiB bf16",
            measured=f"{P32:,} params / {B32 / 2 ** 30:.2f} GiB "
                     f"({B32:,} bytes, {S32} shards, all bf16)",
            explanation="32.8B is the 32B's parameter count EXCLUDING its 636,401,664-param vision "
                        "tower (33,452,718,336 - 636,401,664 = 32,816,316,672).  31.5 GiB is not a "
                        "bf16 weight size at all -- it is params/2^30, i.e. ONE byte per parameter.  "
                        "The bf16 weights are 62.31 GiB.  Source: the safetensors index on disk, "
                        "read this run; cross-checked against "
                        "flop_ratio_derivation_2026-08-03.json:parameter_counts."),
        co_residency=dict(
            seven_b_only=dict(params=int(P7 + LORA_PARAMS),
                              GiB=round((B7 + LORA_BYTES) / 2 ** 30, 4)),
            both_models=dict(params=int(P7 + LORA_PARAMS + P32),
                             GiB=round((B7 + LORA_BYTES + B32) / 2 ** 30, 4)),
            ratio=round((B7 + LORA_BYTES + B32) / (B7 + LORA_BYTES), 3),
            statement="a policy that NEVER touches the 32B needs "
                      f"{(B7 + LORA_BYTES) / 2 ** 30:.2f} GiB of weights resident; one that does "
                      f"needs {(B7 + LORA_BYTES + B32) / 2 ** 30:.2f} GiB co-resident, or a model "
                      "swap on the critical path."),
        measured_test_time_vram=dict(
            source="results/cascade_methods/artifacts/vram_testtime_2026-08-11.json",
            S1_lingshu7b_direct_mcq={k: VRAM_S1.get(k) for k in
                                     ("n", "b_peak_allocated_gib", "c_peak_reserved_gib",
                                      "d_board_used_gib")},
            S2_lingshu32b_direct_mcq={k: VRAM_S2.get(k) for k in
                                      ("n", "b_peak_allocated_gib", "c_peak_reserved_gib",
                                       "d_board_used_gib")},
            S3_lingshu7b_plus_lora_verifier={k: VRAM_S3.get(k) for k in
                                             ("n", "b_peak_allocated_gib", "c_peak_reserved_gib",
                                              "d_board_used_gib")},
            S4_opentext_bestof8_full_arm={k: VRAM_S4.get(k) for k in
                                          ("n", "b_peak_allocated_gib", "c_peak_reserved_gib",
                                           "d_board_used_gib")},
            key_finding=VRAM["key_findings"].get("3_the_verifier_is_nearly_free_and_the_open_arm_"
                                                 "costs_ONE_7B", {}).get("claim"))),
    operating_points_prespecified=PRESPEC,
)
os.makedirs(ART, exist_ok=True)
json.dump(rep, open(OUT, "w"), indent=1, default=float)
print(f"\nwrote {OUT}")
