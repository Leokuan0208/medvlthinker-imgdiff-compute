#!/usr/bin/env python3
"""
cost_floor.py -- ATTACK 3 (COST-FLOOR).  Pre-registered endpoint:

    "Macro-weighted AS-CHARGED FLOP-eq cost of the best policy whose macro accuracy delta vs
     always-32B-direct has a 95% CI whose LOWER bound is >= -0.0029 (a genuine tie, not a
     disguised loss)."

SUCCESS = macro as-charged cost <= 1.00x always-32B-direct with that tie, AND the re-costing
corroborated by measurement, AND the latency/energy advantage vs the reasoning baseline
RE-DERIVED for the new policy.  STRETCH = <= 0.85x.

Four levers (brief section 3.1):
  L1  the best-of-N charge ignores shared prefill (BO8 = 16.0 = 8 gens + 8 verifies each at
      full batch-1 cost, integrated_method.py:55).  run_openvqa.py:154 really generates with
      vLLM SamplingParams(n=N), i.e. the deployed path SHARES the prefill.
  L2  cross-fit per-cell arm selection (never eval-visible).
  L3  N reduction (selected accuracy is flat by N~5-8).
  L4  skip the cheap leg where it is pure waste (a special case of L2).

INTEGRITY CONSTRAINTS (brief section 3.2), all enforced in code:
  1. SYMMETRY -- every re-costing rule is applied to the baselines too, on the same code path.
     always-32B-direct has n_gen7 = n_ver7 = 0 and n_32b = 1, so shared-prefill credit cannot
     move it; always-7B has n_gen7 = 1 and G(1) == 1.0 exactly.  Asserted below.
  2. MEASUREMENT, NOT MODELLING -- the re-costed FLOP model must be corroborated by a fresh
     batch-1 wall-clock + NVML measurement of the actual serving path
     (src/cascade_methods/cost_floor_measure.py).  If measured BoN@8 energy exceeds the
     re-costed model by > 30%, the re-costing is REJECTED.
  3. THREE CURRENCIES, each labelled; macro cost never paired with sample-weighted accuracy.
  4. as-charged and re-costed published SIDE BY SIDE, always.

No GPU.  Reads only committed artifacts + checkpoints.  Launch from the repo root:
    python3 src/cascade_methods/cost_floor.py
Writes results/cascade_methods/artifacts/cost_floor_2026-08-10.json
"""
import json, os, sys, hashlib
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
OUT = os.path.join(ART, "cost_floor_2026-08-10.json")
SEED = 20260810
NBOOT = 10000
TIE_TOL = 0.0029          # pre-registered: lower CI bound must be >= -TIE_TOL

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
OPEN = {"SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"}
BASE = "always_32b_direct"
REAS = "always_32b_reasoning"
DEPLOYABLE = ["always_7b", "always_32b_direct", "always_32b_reasoning",
              "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]

R32_CHARGED = 4.57        # the paper constant (lingshu_medeval_cascade.py:21)
R32_DERIVED = 3.816       # flop_ratio_derivation_2026-08-03.json:derived_ratio.R32_derived

rep = {}                  # the report


# =================================================================================================
# 0.  LOAD
# =================================================================================================
Z = np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"))
MAC = json.load(open(os.path.join(ART, "_selector_rerun_parts/macro_disjoint.json")))
PUB = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))["per_arm"]["disjoint"]
FLOPD = json.load(open(os.path.join(ART, "flop_ratio_derivation_2026-08-03.json")))
BO8A = json.load(open(os.path.join(ART, "bestofn_latency_energy_2026-08-03.json")))
BO8B = json.load(open(os.path.join(ART, "bestofn_latency_energy_2026-08-03_rep2.json")))
PCC = MAC["cost"]["per_cell_as_charged"]

OK = {(c, s): Z[f"{c}|{s}"].astype(np.float64) for c in CELLS
      for s in PUB["per_cell_acc"][c].keys() if f"{c}|{s}" in Z}
N = {c: len(Z[f"{c}|{BASE}"]) for c in CELLS}


# =================================================================================================
# 1.  NULL TESTS
# =================================================================================================
def null_n1():
    """Reproduce the 8-cell macro accuracy AND the 1.739x macro as-charged cost."""
    dev_a, rows = [], {}
    for s in PUB["macro_acc"]:
        m = float(np.mean([OK[(c, s)].mean() for c in CELLS]))
        d = abs(m - PUB["macro_acc"][s]); dev_a.append(d)
        rows[s] = dict(recomputed=round(m, 6), published=PUB["macro_acc"][s], abs_dev=round(d, 8))
    dev_c = []
    for s in PUB["cost_macro"]:
        f = float(np.mean([PCC[c][s]["flops"] for c in CELLS]))
        dev_c.append(abs(f - PUB["cost_macro"][s]["flops"]))
        rows[s]["cost_recomputed"] = round(f, 6)
        rows[s]["cost_published"] = PUB["cost_macro"][s]["flops"]
    r_am = float(np.mean([PCC[c]["method_accuracy_max_veto"]["flops"] for c in CELLS])) / R32_CHARGED
    return dict(
        name="N1 -- reproduce the published 8-cell macro accuracy and macro as-charged cost",
        source_vectors="results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz",
        source_published="results/cascade_methods/artifacts/cascade_selector_rerun_2026-08-05.json:per_arm.disjoint",
        per_system=rows,
        max_abs_dev_accuracy=round(max(dev_a), 8),
        max_abs_dev_cost_flops=round(max(dev_c), 8),
        macro_ratio_accmax_vs_direct=round(r_am, 4),
        published_ratio=PUB["ratios_macro"]["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"],
        verdict=("PASS -- accuracy deviates only by the published file's 4-dp rounding; cost reproduces "
                 "to 4 dp" if max(dev_a) < 5e-5 and max(dev_c) < 5e-4 else "FAIL"))


def null_n2():
    """Rebuild the FLOP model from the artifact's MEASURED parameter counts + token geometry and
    reproduce R32_derived = 3.816.  This also yields the prefill/decode split that lever L1 needs."""
    P7 = FLOPD["parameter_counts"]["lingshu_7b"]["by_role"]
    P32 = FLOPD["parameter_counts"]["lingshu_32b"]["by_role"]
    g = FLOPD["flop_model"]["operating_point"]
    T, M, PP = g["T"], g["M"], g["P_patches"]
    cfg = dict(lingshu_7b=dict(L=28, d=3584, G=g["G_7b"], role=P7),
               lingshu_32b=dict(L=64, d=5120, G=g["G_32b"], role=P32))
    # ViT: depth 32, hidden 1280, 4 full-attention layers, 28 windowed over 64 patches
    VITL, VITD, VITFULL, WIN = 32, 1280, 4, 64
    vit_attn = (VITFULL * 4 * PP ** 2 * VITD + (VITL - VITFULL) * (PP / WIN) * 4 * WIN ** 2 * VITD) / 1e9
    comp, tot = {}, {}
    for k, c in cfg.items():
        r, L, d, G = c["role"], c["L"], c["d"], c["G"]
        e = dict(
            vision_tower_dense=2 * PP * r["vision_tower"] / 1e9,
            vision_tower_attn=vit_attn,
            vision_merger=2 * M * r["vision_merger"] / 1e9,
            lm_prefill_dense=2 * T * r["lm_body"] / 1e9,
            lm_prefill_attn=2 * L * T ** 2 * d / 1e9,            # 4*L*T^2*d halved for causal
            lm_decode_dense=2 * (G - 1) * r["lm_body"] / 1e9,
            lm_decode_attn=2 * (G - 1) * L * T * d / 1e9 / 1000,  # negligible; artifact carries 0.61/1.98
            lm_head=2 * G * r["lm_head"] / 1e9)
        e["lm_decode_attn"] = FLOPD["flop_model"][f"{k}_gflops"]["lm_decode_attn"]
        comp[k] = {a: round(v, 2) for a, v in e.items()}
        tot[k] = sum(e.values())
    ratio = tot["lingshu_32b"] / tot["lingshu_7b"]
    pubc = {k: FLOPD["flop_model"][f"{k}_gflops"] for k in cfg}
    dev = max(abs(comp[k][a] - pubc[k][a]) for k in cfg for a in comp[k])
    PRE = ["vision_tower_dense", "vision_tower_attn", "vision_merger", "lm_prefill_dense", "lm_prefill_attn"]
    DEC = ["lm_decode_dense", "lm_decode_attn", "lm_head"]
    pre7 = sum(comp["lingshu_7b"][a] for a in PRE) / tot["lingshu_7b"]
    dec7 = sum(comp["lingshu_7b"][a] for a in DEC) / tot["lingshu_7b"]
    return dict(
        name="N2 -- rebuild the FLOP model from measured parameter counts and reproduce R32_derived",
        source="results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json",
        rebuilt_component_gflops=comp,
        rebuilt_total_gflops={k: round(v, 2) for k, v in tot.items()},
        published_total_gflops={k: pubc[k]["TOTAL"] for k in cfg},
        max_abs_dev_component_gflops=round(dev, 3),
        rebuilt_ratio=round(ratio, 4),
        published_ratio=FLOPD["derived_ratio"]["R32_derived"],
        prefill_share_7b=round(pre7, 6), decode_share_7b=round(dec7, 6),
        note=("prefill_share + decode_share = 1 by construction. G(N) = prefill_share + N*decode_share "
              "is the shared-prefill cost of N samples in units of ONE 7B forward; G(1) == 1.0 exactly, "
              "which is what makes the re-costing symmetric for every N=1 baseline."),
        verdict=("PASS" if abs(ratio - FLOPD["derived_ratio"]["R32_derived"]) < 0.001 and dev < 0.05
                 else "FAIL"))


def null_n3():
    """Reproduce the measured gen1/gen8/verify1/verify8 constants and report their spread."""
    legs = ["gen1", "gen8", "verify1", "verify8", "bo8_total"]
    rows = {}
    for lg in legs:
        a, b = BO8A["measured"][lg], BO8B["measured"][lg]
        pooled = BO8A["reconciliation"]["measurement"]["pooled"][lg]
        rows[lg] = dict(
            rep1=dict(n=a["n"], lat_ms=a["lat_ms_mean"], energy_j=a["energy_j_mean"]),
            rep2=dict(n=b["n"], lat_ms=b["lat_ms_mean"], energy_j=b["energy_j_mean"]),
            pooled_published=dict(n=pooled["n"], lat_ms=pooled["lat_ms_mean"], energy_j=pooled["energy_j_mean"]),
            recomputed_pooled_lat_ms=round((a["lat_ms_mean"] * a["n"] + b["lat_ms_mean"] * b["n"]) / (a["n"] + b["n"]), 1),
            recomputed_pooled_energy_j=round((a["energy_j_mean"] * a["n"] + b["energy_j_mean"] * b["n"]) / (a["n"] + b["n"]), 2),
            replicate_spread_lat_pct=round(abs(a["lat_ms_mean"] - b["lat_ms_mean"])
                                           / ((a["lat_ms_mean"] + b["lat_ms_mean"]) / 2) * 100, 1),
            replicate_spread_energy_pct=round(abs(a["energy_j_mean"] - b["energy_j_mean"])
                                              / ((a["energy_j_mean"] + b["energy_j_mean"]) / 2) * 100, 1))
    dev = max(max(abs(rows[l]["recomputed_pooled_lat_ms"] - rows[l]["pooled_published"]["lat_ms"]),
                  abs(rows[l]["recomputed_pooled_energy_j"] - rows[l]["pooled_published"]["energy_j"]))
              for l in legs)
    return dict(
        name="N3 -- reproduce the measured best-of-8 latency/energy constants and their replicate spread",
        source=["results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03.json",
                "results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03_rep2.json"],
        legs=rows, max_abs_dev_vs_published_pooled=round(dev, 3),
        gpu=BO8A["gpu_names"], gpu_power_limit_w=BO8A["gpu_power_limit_w"],
        harness_note=("HF transformers, num_return_sequences=8 -- this path expands the inputs BEFORE the "
                      "prefill forward, so it does NOT share the prefill.  It is therefore a measurement of "
                      "the NON-shared-prefill implementation and is used here only as a control."),
        verdict="PASS" if dev < 0.15 else "FAIL")


def null_n4_recosting_scope():
    """Confirm honest_recosting_2026-07-29 did NOT already touch the BO8 charge (brief section 3.3)."""
    s = json.dumps(json.load(open(os.path.join(ART, "honest_recosting_2026-07-29.json"))))
    return dict(
        name="N4 -- confirm the BO8 = 16.0 charge was never re-costed before (lever L1 is not a rediscovery)",
        honest_recosting_mentions_BO8=("BO8" in s), honest_recosting_mentions_best_of=("best-of" in s.lower()),
        honest_recosting_scope=("re-costs ONLY the always-32B-with-reasoning baseline, charging each cell its "
                                "own measured generation length instead of one global reasoning constant"),
        bo8_charge_site="src/cascade_methods/integrated_method.py:55  BO8 = dict(ms=522.0, flop=16.0)",
        verdict="PASS -- L1 is a live, never-propagated discrepancy" if "BO8" not in s else "FAIL")


# =================================================================================================
# 2.  ARM COST DECOMPOSITION  (and the assertion that it reproduces the published per-cell cost)
# =================================================================================================
ESC = PUB["escalation"]["per_cell"]
OCD = PUB["open_cell_detail"]
PMC_VETO_KEEP = None  # solved from the published PMC accuracy-max cost


def build_arms():
    """cell -> arm -> (n_gen7, n_ver7, n_32b).  Reconstructed from the published per-cell costs and
    escalation rates, then ASSERTED to reproduce PCC[cell][arm]['flops'] under the as-charged model."""
    global PMC_VETO_KEEP
    A = {c: {} for c in CELLS}
    for c in CELLS:
        A[c]["always_7b"] = (1.0, 0.0, 0.0)
        A[c]["always_32b_direct"] = (0.0, 0.0, 1.0)
        A[c]["always_32b_reasoning"] = (0.0, 0.0, 1.0)      # same forward count; reasoning cost is a
                                                            # latency/energy axis, handled separately
    for c in CELLS:
        if c in OPEN:
            nn = OCD[c]["meanN"]
            A[c]["method_compute_lean"] = (nn, nn, OCD[c]["esc"])
            A[c]["method_accuracy_max_veto"] = (nn, nn, OCD[c]["am2_esc"])
            A[c]["method_accuracy_max_fusion"] = (nn, nn, OCD[c]["esc"])
        else:
            A[c]["method_compute_lean"] = (1.0, 0.0, ESC[c])
            f_am2 = PCC[c]["method_accuracy_max_veto"]["flops"]
            if abs(f_am2 - R32_CHARGED) < 1e-6:
                A[c]["method_accuracy_max_veto"] = (0.0, 0.0, 1.0)   # router picked always-32B outright
            else:
                keep = 1.0 - (f_am2 - 1.0) / R32_CHARGED            # certified-veto keep rate
                if c == "PMC_VQA":
                    PMC_VETO_KEEP = keep
                A[c]["method_accuracy_max_veto"] = (1.0, 0.0, 1.0 - keep)
            f_amf = PCC[c]["method_accuracy_max_fusion"]["flops"]
            A[c]["method_accuracy_max_fusion"] = ((1.0, 0.0, 1.0) if abs(f_amf - (1.0 + R32_CHARGED)) < 1e-6
                                                  else (0.0, 0.0, 1.0))
    # ---- assertion: the decomposition reproduces every published per-cell as-charged cost ----
    dev = 0.0
    for c in CELLS:
        for a, (ng, nv, n32) in A[c].items():
            if a in PCC[c]:
                dev = max(dev, abs((ng + nv + n32 * R32_CHARGED) - PCC[c][a]["flops"]))
    assert dev < 6e-4, f"arm decomposition does not reproduce published per-cell cost (max dev {dev})"
    return A, dev


# =================================================================================================
# 3.  THE COSTING CONVENTIONS
# =================================================================================================
def costing(pre7, dec7, ver_pre, ver_marg):
    """Return {name: (fn(ng,nv,n32) -> flop-eq, description)}.  Applied to EVERY system identically."""
    def A(ng, nv, n32, R=R32_CHARGED):      # as-charged (paper constants)
        return ng * 1.0 + nv * 1.0 + n32 * R

    def B(ng, nv, n32, R=R32_CHARGED):      # generation prefill shared (what vLLM n=N actually does)
        g = (pre7 + ng * dec7) if ng > 0 else 0.0
        return g + nv * 1.0 + n32 * R

    def C(ng, nv, n32, R=R32_CHARGED):      # + verifier prefix sharing (needs an implementation change)
        g = (pre7 + ng * dec7) if ng > 0 else 0.0
        v = (ver_pre + nv * ver_marg) if nv > 0 else 0.0
        return g + v + n32 * R
    return {
        "A_as_charged": (A, "PAPER CONSTANTS. N generations + N verifier forwards, each charged one full "
                            "batch-1 7B forward (BO8 = 16.0); R32 = 4.57.  PRIMARY endpoint convention."),
        "B_recost_gen": (B, "GENERATION PREFILL SHARED. G(N) = prefill_share + N*decode_share, derived from "
                            "the repo's own FLOP model.  This is what vLLM SamplingParams(n=N) does and it "
                            "is the path run_openvqa.py:154 actually used.  Verifier left at N x 1.0.  "
                            "CONSERVATIVE re-costing."),
        "C_recost_full": (C, "GENERATION PREFILL SHARED + VERIFIER PREFIX SHARED.  The verifier's shared "
                             "image+question+'Proposed answer: ' prefix is prefilled once and only the "
                             "~21-token candidate suffix is recomputed per candidate.  REQUIRES AN "
                             "IMPLEMENTATION CHANGE the repo never ran -- OPTIMISTIC, flagged as such."),
    }


# =================================================================================================
# 4.  BOOTSTRAP
# =================================================================================================
GRP = {}     # cell -> (signature id per item, group sizes, multinomial count matrix)


def boot_streams(sig_extra, nboot=NBOOT, seed=SEED):
    """ONE shared paired bootstrap stream per cell, reused by every policy and every baseline.

    Implemented with the repo's own multinomial-pattern shortcut (macro_average_headline.py
    documents it and verifies it against a literal gather): items are grouped by their FULL
    correctness signature across every arm plus every fold label any policy uses, so any policy's
    per-item value is constant within a group; a multinomial draw over the group counts is then
    exactly the distribution of an item-level resample, and it is shared by construction."""
    rng = np.random.default_rng(seed)
    for c in CELLS:
        cols = [OK[(c, s)] for s in sorted({s for (cc, s) in OK if cc == c})] + \
               [np.asarray(x, float) for x in sig_extra[c]]
        key = np.zeros(N[c], dtype=np.int64)
        for col in cols:
            u = np.unique(col)
            key = key * len(u) + np.searchsorted(u, col)
        uk, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uk)).astype(np.float64)
        GRP[c] = dict(inv=inv, k=len(uk), sizes=cnt,
                      draws=rng.multinomial(N[c], cnt / N[c], size=nboot).astype(np.float64))
    return GRP


def macro_boot(vec_by_cell, streams=None):
    """vec_by_cell: cell -> per-item 0/1.  Returns (point, nboot-vector of macro means)."""
    per = []
    for c in CELLS:
        v = vec_by_cell[c]
        g = np.bincount(GRP[c]["inv"], weights=v, minlength=GRP[c]["k"]) / GRP[c]["sizes"]
        per.append(GRP[c]["draws"] @ g / N[c])
    return float(np.mean([vec_by_cell[c].mean() for c in CELLS])), np.mean(per, axis=0)


def cell_boot(c, v):
    g = np.bincount(GRP[c]["inv"], weights=v, minlength=GRP[c]["k"]) / GRP[c]["sizes"]
    return GRP[c]["draws"] @ g / N[c]


def ci_of(d):
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return lo, hi


# =================================================================================================
# 5.  POLICIES
# =================================================================================================
def policy_vectors(assign):
    return {c: OK[(c, assign[c])] for c in CELLS}


def policy_cost(assign, ARMS, fn, R=R32_CHARGED):
    return float(np.mean([fn(*ARMS[c][assign[c]], R) for c in CELLS]))


KFOLD = 5
FOLD_CF = {}      # cross-fit fold label per item  (fixed before any bootstrap is drawn)
FOLD_NS = {}      # nested-CV outer fold label per item


def make_folds(seed=SEED, kfold=KFOLD):
    r1 = np.random.default_rng(seed)
    r2 = np.random.default_rng(seed + 1)
    for c in CELLS:
        FOLD_CF[c] = r1.permutation(N[c]) % kfold
        FOLD_NS[c] = r2.permutation(N[c]) % kfold


# =================================================================================================
# 5b.  LATENCY / ENERGY  (as-charged, and corrected by the 2026-08-03 batch-8 MEASUREMENT)
# =================================================================================================
def latency_energy_constants():
    """Invert the published per-cell open-cell latency/energy to recover the as-charged constants,
    then pair them with the MEASURED batch-8 constants so the correction is a clean substitution."""
    rowsN, y_e, y_lp, y_ls = [], [], [], []
    for c in ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]:
        nn = OCD[c]["meanN"]
        for arm, esc in (("method_compute_lean", OCD[c]["esc"]), ("method_accuracy_max_veto", OCD[c]["am2_esc"])):
            rowsN.append((nn, esc)); p = PCC[c][arm]
            y_e.append(p["energy_j"]); y_lp.append(p["lat_par_ms"]); y_ls.append(p["lat_seq_ms"])
    Xn = np.array([[n, e] for n, e in rowsN])
    Xc = np.array([[1.0, e] for _, e in rowsN])
    e_par, r_e, *_ = np.linalg.lstsq(Xn, np.array(y_e), rcond=None)
    lp, r_lp, *_ = np.linalg.lstsq(Xc, np.array(y_lp), rcond=None)
    ls, r_ls, *_ = np.linalg.lstsq(Xn, np.array(y_ls), rcond=None)
    resid = max(float(np.abs(Xn @ e_par - y_e).max()), float(np.abs(Xc @ lp - y_lp).max()),
                float(np.abs(Xn @ ls - y_ls).max()))
    pooled = BO8A["reconciliation"]["measurement"]["pooled"]
    m1_lat = pooled["gen1"]["lat_ms_mean"] + pooled["verify1"]["lat_ms_mean"]
    m8_lat = pooled["bo8_total"]["lat_ms_mean"]
    m1_e = pooled["gen1"]["energy_j_mean"] + pooled["verify1"]["energy_j_mean"]
    m8_e = pooled["bo8_total"]["energy_j_mean"]
    return dict(as_charged=dict(energy_per_sample_j=float(e_par[0]), energy_32b_j=float(e_par[1]),
                                lat_par_bon_ms=float(lp[0]), lat_32b_ms=float(lp[1]),
                                lat_seq_per_sample_ms=float(ls[0])),
                inversion_max_residual=round(resid, 4),
                measured=dict(bo1_lat_ms=round(m1_lat, 1), bo8_lat_ms=round(m8_lat, 1),
                              bo1_energy_j=round(m1_e, 2), bo8_energy_j=round(m8_e, 2),
                              n=pooled["bo8_total"]["n"], gpu=BO8A["gpu_names"],
                              power_limit_w=BO8A["gpu_power_limit_w"]),
                interpolation="linear in N between the measured N=1 and N=8 points [INTERPOLATED]")


def cell_arm_lat_energy(c, arm, K, ARMS):
    """(lat_par_ms, lat_seq_ms, energy_j) under as-charged and under the measured correction."""
    ac = PCC[c][arm]
    base = (ac["lat_par_ms"], ac["lat_seq_ms"], ac["energy_j"])
    ng = ARMS[c][arm][0]
    if ng <= 1.0:                      # no best-of-N leg -> the correction cannot touch it
        return base, base
    K_ = K["as_charged"]; M = K["measured"]
    esc = ARMS[c][arm][2]
    ac_par = K_["lat_par_bon_ms"]; ac_seq = ng * K_["lat_seq_per_sample_ms"]; ac_e = ng * K_["energy_per_sample_j"]
    m_par = M["bo1_lat_ms"] + (ng - 1) * (M["bo8_lat_ms"] - M["bo1_lat_ms"]) / 7.0
    m_seq = ng * M["bo1_lat_ms"]
    m_e = M["bo1_energy_j"] + (ng - 1) * (M["bo8_energy_j"] - M["bo1_energy_j"]) / 7.0
    corr = (base[0] - ac_par + m_par, base[1] - ac_seq + m_seq, base[2] - ac_e + m_e)
    return base, corr


def crossfit_assign(eps, arms_avail, ARMS, cost_fn, kfold=KFOLD):
    """Per cell, per fold: on the OTHER folds only, take the most accurate arm; deploy the CHEAPEST
    arm within eps of it.  Held-out fold gets that arm's answer.  No eval-fold label is ever seen
    by the choice that is applied to it."""
    out_vec, out_cost, chosen = {}, {}, {}
    for c in CELLS:
        n = N[c]
        fold = FOLD_CF[c]
        v = np.empty(n); costs = []; picks = []
        for f in range(kfold):
            tr, te = fold != f, fold == f
            if te.sum() == 0:
                continue
            accs = {a: OK[(c, a)][tr].mean() for a in arms_avail}
            best = max(accs.values())
            elig = [a for a in arms_avail if accs[a] >= best - eps]
            a_star = min(elig, key=lambda a: (cost_fn(*ARMS[c][a]), a))
            v[te] = OK[(c, a_star)][te]
            costs.append((te.sum(), cost_fn(*ARMS[c][a_star])))
            picks.append(a_star)
        out_vec[c] = v
        out_cost[c] = float(sum(w * x for w, x in costs) / sum(w for w, _ in costs))
        chosen[c] = picks
    return out_vec, out_cost, chosen


def evalvisible_assign(eps, arms_avail, ARMS, cost_fn):
    """DIAGNOSTIC upper bound: the same rule with full eval visibility."""
    assign = {}
    for c in CELLS:
        accs = {a: OK[(c, a)].mean() for a in arms_avail}
        best = max(accs.values())
        elig = [a for a in arms_avail if accs[a] >= best - eps]
        assign[c] = min(elig, key=lambda a: (cost_fn(*ARMS[c][a]), a))
    return assign


def nested_eps(arms_avail, ARMS, cost_fn, eps_grid, kfold=KFOLD, inner=4):
    """FULLY HONEST primary: eps itself is chosen inside the training folds, never on eval."""
    out_vec, out_cost, picked_eps = {}, {}, []
    fold_of = FOLD_NS
    for f in range(kfold):
        # ---- inner CV on the training folds only ----
        best_e, best_cost = eps_grid[0], None
        scored = []
        for e in eps_grid:
            accs, costs = [], []
            for c in CELLS:
                tr = fold_of[c] != f
                idx = np.flatnonzero(tr)
                ifold = (np.arange(len(idx)) + hash(c) % inner) % inner
                v = np.empty(len(idx)); cc = []
                for g in range(inner):
                    itr, ite = ifold != g, ifold == g
                    if ite.sum() == 0:
                        continue
                    a_ = {a: OK[(c, a)][idx[itr]].mean() for a in arms_avail}
                    bb = max(a_.values())
                    el = [a for a in arms_avail if a_[a] >= bb - e]
                    a_star = min(el, key=lambda a: (cost_fn(*ARMS[c][a]), a))
                    v[ite] = OK[(c, a_star)][idx[ite]]
                    cc.append((ite.sum(), cost_fn(*ARMS[c][a_star])))
                accs.append(v.mean()); costs.append(sum(w * x for w, x in cc) / sum(w for w, _ in cc))
            scored.append((e, float(np.mean(accs)), float(np.mean(costs))))
        top = max(s[1] for s in scored)
        feas = [s for s in scored if s[1] >= top - TIE_TOL]
        best_e = min(feas, key=lambda s: (s[2], s[0]))[0]
        picked_eps.append(best_e)
        # ---- apply to the held-out outer fold ----
        for c in CELLS:
            tr, te = fold_of[c] != f, fold_of[c] == f
            a_ = {a: OK[(c, a)][tr].mean() for a in arms_avail}
            bb = max(a_.values())
            el = [a for a in arms_avail if a_[a] >= bb - best_e]
            a_star = min(el, key=lambda a: (cost_fn(*ARMS[c][a]), a))
            out_vec.setdefault(c, np.empty(N[c]))[te] = OK[(c, a_star)][te]
            out_cost.setdefault(c, []).append((te.sum(), cost_fn(*ARMS[c][a_star]), a_star))
    cost = {c: float(sum(w * x for w, x, _ in out_cost[c]) / sum(w for w, _, _ in out_cost[c])) for c in CELLS}
    picks = {c: [a for _, _, a in out_cost[c]] for c in CELLS}
    return out_vec, cost, picks, picked_eps


# =================================================================================================
# 6.  MAIN
# =================================================================================================
def main():
    rep["title"] = ("ATTACK 3 -- COST-FLOOR.  Can the method reach <= 1.00x always-32B-direct's macro "
                    "compute while holding a genuine accuracy tie?  Endpoint is COST, not accuracy.")
    rep["date"] = "2026-08-10"
    rep["reproduce"] = "python3 src/cascade_methods/cost_floor.py"
    rep["no_gpu"] = True
    rep["no_fabricated_numbers"] = True
    rep["seed"] = SEED
    rep["n_bootstrap"] = NBOOT
    rep["pool"] = "Variant B (MMMU excluded): 5 benchmarks / 8 cells / n=42224, CLEAN disjoint verifier"
    rep["numerics_pins"] = dict(OMP_NUM_THREADS="1", PYTHONHASHSEED="0",
                               tf32="not applicable -- this script is pure numpy on stored 0/1 vectors",
                               rank_convention="not applicable -- no ranking step in this attack",
                               bootstrap="paired item-level, one shared resample stream per cell reused by "
                                         "every policy and every baseline")

    # ---------------- null tests ----------------
    rep["null_tests"] = dict(N1=null_n1(), N2=null_n2(), N3=null_n3(), N4=null_n4_recosting_scope())
    for k, v in rep["null_tests"].items():
        print(f"[{k}] {v['verdict']}")
        if not v["verdict"].startswith("PASS"):
            print("NULL TEST FAILED -- aborting per pre-registration"); sys.exit(1)

    pre7 = rep["null_tests"]["N2"]["prefill_share_7b"]
    dec7 = rep["null_tests"]["N2"]["decode_share_7b"]

    # ---------------- verifier prompt geometry (for convention C) ----------------
    geo = json.load(open(os.path.join(ART, "_cost_floor_verifier_geometry.json")))
    T_pre, s_marg = geo["shared_prefix_tok"], geo["per_candidate_suffix_tok"]
    P7 = FLOPD["parameter_counts"]["lingshu_7b"]["by_role"]
    tot7 = FLOPD["flop_model"]["lingshu_7b_gflops"]["TOTAL"]
    vis = (FLOPD["flop_model"]["lingshu_7b_gflops"]["vision_tower_dense"]
           + FLOPD["flop_model"]["lingshu_7b_gflops"]["vision_tower_attn"]
           + FLOPD["flop_model"]["lingshu_7b_gflops"]["vision_merger"])
    lm_per_tok = 2 * P7["lm_body"] / 1e9
    head1 = 2 * 1 * P7["lm_head"] / 1e9
    ver_pre = (vis + lm_per_tok * T_pre) / tot7
    ver_marg = (lm_per_tok * s_marg + head1) / tot7
    rep["verifier_geometry"] = dict(geo, ver_prefix_cost_units=round(ver_pre, 4),
                                    ver_marginal_cost_per_candidate_units=round(ver_marg, 4),
                                    ver_1_candidate_units=round(ver_pre + ver_marg, 4),
                                    as_charged_ver_1_candidate_units=1.0,
                                    cost_units_provenance="derived from the N2 FLOP model + the measured "
                                                          "token geometry above")

    ARMS, arm_dev = build_arms()
    rep["arm_decomposition"] = dict(
        provenance="derived -- reconstructed from the published per-cell as-charged costs and escalation "
                   "rates, then ASSERTED to reproduce every published per-cell cost",
        max_abs_dev_vs_published_per_cell_flops=round(arm_dev, 8),
        pmc_certified_veto_keep_rate=round(PMC_VETO_KEEP, 4) if PMC_VETO_KEEP else None,
        table={c: {a: dict(n_gen7=round(v[0], 4), n_ver7=round(v[1], 4), n_32b=round(v[2], 4))
                   for a, v in ARMS[c].items()} for c in CELLS})

    CONV = costing(pre7, dec7, ver_pre, ver_marg)
    # ---- integrity constraint 1: SYMMETRY, asserted ----
    sym = {}
    for name, (fn, _) in CONV.items():
        sym[name] = dict(always_7b=round(fn(1.0, 0.0, 0.0), 6),
                         always_32b_direct=round(fn(0.0, 0.0, 1.0), 6),
                         always_32b_reasoning=round(fn(0.0, 0.0, 1.0), 6))
        assert abs(sym[name]["always_7b"] - 1.0) < 1e-9
        assert abs(sym[name]["always_32b_direct"] - R32_CHARGED) < 1e-9
    rep["symmetry_check"] = dict(
        statement=("Every convention is applied to the baselines on the SAME code path.  always-32B-direct "
                   "has n_gen7 = n_ver7 = 0 and n_32b = 1, so no shared-prefill credit can reach it: it is "
                   "4.57 in all three conventions.  always-7B has n_gen7 = 1 and G(1) = prefill+decode = "
                   "1.0 exactly.  The re-costing therefore cannot move a baseline, only the method."),
        baseline_costs_by_convention=sym)

    rep["costing_conventions"] = {k: v[1] for k, v in CONV.items()}
    rep["R32"] = dict(as_charged=R32_CHARGED, derived=R32_DERIVED,
                      band=FLOPD["derived_ratio"]["recommended"]["band"],
                      direction_note=("The method's ratio to the baseline is (7B work)/R32 + escalation, so a "
                                      "LARGER R32 FLATTERS the method.  R32 = 4.57 (the paper constant) is "
                                      "therefore the value that helps us and R32 = 3.816 (the derived one) "
                                      "is the one that hurts us.  Both are reported for every headline."))

    # ---------------- the deployed operating points, re-costed ----------------
    make_folds()
    streams = boot_streams({c: [FOLD_CF[c], FOLD_NS[c]] for c in CELLS})
    rep["bootstrap_groups"] = {c: dict(n=N[c], n_signature_groups=GRP[c]["k"]) for c in CELLS}
    base_vec = {c: OK[(c, BASE)] for c in CELLS}
    base_pt, base_boot = macro_boot(base_vec, streams)

    def evaluate(name, vec_by_cell, cost_by_cell_or_assign, kind, extra=None):
        pt, bt = macro_boot(vec_by_cell, streams)
        d = bt - base_boot
        lo, hi = ci_of(d)
        delta = pt - base_pt
        if all(("|" in k) for k in cost_by_cell_or_assign):        # already a {conv|R: {...}} table
            costs = cost_by_cell_or_assign
        else:
            costs = {}
            for conv, (fn, _) in CONV.items():
                for Rlab, R in (("R32_4.57", R32_CHARGED), ("R32_3.816", R32_DERIVED)):
                    cc = float(np.mean([cost_by_cell_or_assign[conv][Rlab][c] for c in CELLS]))
                    costs[f"{conv}|{Rlab}"] = dict(macro_flops=round(cc, 4), x_direct=round(cc / R, 4))
        per_cell = {c: dict(acc=round(float(vec_by_cell[c].mean()), 4),
                            delta_vs_direct=round(float(vec_by_cell[c].mean() - OK[(c, BASE)].mean()), 4))
                    for c in CELLS}
        # guardrail: per-cell paired bootstrap vs always-32B-direct
        gr = {}
        for c in CELLS:
            dd = cell_boot(c, vec_by_cell[c] - OK[(c, BASE)])
            l, h = ci_of(dd)
            gr[c] = dict(delta=round(float(vec_by_cell[c].mean() - OK[(c, BASE)].mean()), 4),
                         lo=round(l, 4), hi=round(h, 4), worse_sig=bool(h < 0))
        return dict(name=name, kind=kind, macro_acc=round(pt, 4),
                    delta_vs_direct=round(delta, 4), lo=round(lo, 4), hi=round(hi, 4),
                    tie_preserved=bool(lo >= -TIE_TOL),
                    verdict=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"),
                    cost=costs, per_cell=per_cell, guardrail=gr, **(extra or {}))

    def cost_grid(assign_or_costfn):
        """cell costs under every (convention, R32) pair, from a per-cell arm assignment."""
        g = {}
        for conv, (fn, _) in CONV.items():
            g[conv] = {}
            for Rlab, R in (("R32_4.57", R32_CHARGED), ("R32_3.816", R32_DERIVED)):
                g[conv][Rlab] = {c: fn(*ARMS[c][assign_or_costfn[c]], R) for c in CELLS}
        return g

    shipped = {}
    for lab, arm in (("deployed_compute_lean", "method_compute_lean"),
                     ("deployed_accuracy_max", "method_accuracy_max_veto"),
                     ("deployed_accuracy_max_fusion", "method_accuracy_max_fusion"),
                     ("baseline_always_7b", "always_7b"),
                     ("baseline_always_32b_reasoning", "always_32b_reasoning")):
        assign = {c: arm for c in CELLS}
        shipped[lab] = evaluate(lab, policy_vectors(assign), cost_grid(assign), "shipped operating point",
                                dict(assignment=assign,
                                     seven_b_macro_weight=round(float(np.mean(
                                         [1.0 if ARMS[c][arm][0] > 0 else 0.0 for c in CELLS])), 4)))
    rep["shipped_operating_points_recosted"] = shipped

    # ---------------- L1 headline: what the re-costing alone does ----------------
    am = shipped["deployed_accuracy_max"]["cost"]
    rep["L1_recosting_effect"] = dict(
        statement=("Lever L1 alone -- no policy change, no re-routing, the SHIPPED accuracy-max arm, "
                   "only the best-of-N charge corrected for the shared prefill the deployed vLLM path "
                   "actually performs."),
        accuracy_unchanged=shipped["deployed_accuracy_max"]["macro_acc"],
        x_direct_as_charged=am["A_as_charged|R32_4.57"]["x_direct"],
        x_direct_recost_gen=am["B_recost_gen|R32_4.57"]["x_direct"],
        x_direct_recost_full=am["C_recost_full|R32_4.57"]["x_direct"],
        x_direct_as_charged_R32derived=am["A_as_charged|R32_3.816"]["x_direct"],
        x_direct_recost_gen_R32derived=am["B_recost_gen|R32_3.816"]["x_direct"],
        x_direct_recost_full_R32derived=am["C_recost_full|R32_3.816"]["x_direct"])

    # ---------------- L2/L4: the cross-fit arm selector, swept over eps ----------------
    arms_avail = ["always_7b", "always_32b_direct", "method_compute_lean",
                  "method_accuracy_max_veto", "method_accuracy_max_fusion"]
    EPS = [0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 1.0]
    frontier = {}
    for conv, (fn, _) in CONV.items():
        rows = []
        for e in EPS:
            vec, ccost, picks = crossfit_assign(e, arms_avail, ARMS, lambda *a: fn(*a, R32_CHARGED))
            pt, bt = macro_boot(vec, streams)
            d = bt - base_boot; lo, hi = ci_of(d)
            # cost of THIS cross-fit policy under every convention/R pair
            cg = {}
            for conv2, (fn2, _) in CONV.items():
                for Rlab, R in (("R32_4.57", R32_CHARGED), ("R32_3.816", R32_DERIVED)):
                    cc = []
                    for c in CELLS:
                        pk = picks[c]
                        cc.append(float(np.mean([fn2(*ARMS[c][a], R) for a in pk])))
                    cg[f"{conv2}|{Rlab}"] = dict(macro_flops=round(float(np.mean(cc)), 4),
                                                 x_direct=round(float(np.mean(cc)) / R, 4))
            sevenb = float(np.mean([np.mean([1.0 if ARMS[c][a][0] > 0 else 0.0 for a in picks[c]])
                                    for c in CELLS]))
            gr = {}
            for c in CELLS:
                dd = cell_boot(c, vec[c] - OK[(c, BASE)])
                l2, h2 = ci_of(dd)
                gr[c] = dict(delta=round(float(vec[c].mean() - OK[(c, BASE)].mean()), 4),
                             lo=round(l2, 4), hi=round(h2, 4), worse_sig=bool(h2 < 0))
            rows.append(dict(eps=e, macro_acc=round(pt, 4), delta_vs_direct=round(pt - base_pt, 4),
                             lo=round(lo, 4), hi=round(hi, 4), tie_preserved=bool(lo >= -TIE_TOL),
                             not_significantly_worse=bool(hi > 0),
                             cost=cg, seven_b_macro_weight=round(sevenb, 4),
                             guardrail=gr,
                             guardrail_flags=[c for c in CELLS if gr[c]["worse_sig"]],
                             picks={c: sorted(set(picks[c])) for c in CELLS}))
        frontier[conv] = rows
    rep["crossfit_frontier"] = dict(
        protocol=("For every cell and every 5-fold split: the arm menu is scored on the FOUR TRAINING "
                  "folds only; the deployed arm is the CHEAPEST arm within eps of the most accurate one; "
                  "the held-out fold is answered by that arm.  No eval-fold label ever reaches the choice "
                  "applied to it.  Each row is a genuine held-out operating point.  The selector is fitted "
                  "under the same convention it is being priced in (one selector per convention)."),
        arm_menu=arms_avail, eps_grid=EPS, rows=frontier)

    # ---------------- the pre-registered PRIMARY endpoint ----------------
    def cheapest_tie(rows, key):
        feas = [r for r in rows if r["tie_preserved"]]
        return min(feas, key=lambda r: r["cost"][key]["x_direct"]) if feas else None

    prim = cheapest_tie(frontier["A_as_charged"], "A_as_charged|R32_4.57")
    rep["PRIMARY_ENDPOINT"] = dict(
        definition=("macro-weighted AS-CHARGED FLOP-eq cost of the cheapest cross-fit policy whose macro "
                    "accuracy delta vs always-32B-direct has a 95%% CI lower bound >= -%.4f" % TIE_TOL),
        eps_selected_by_the_eval_constraint=prim["eps"] if prim else None,
        eps_selection_status=("DIAGNOSTIC on the eps scalar -- eps is picked here by applying the endpoint's "
                              "own eval-side constraint.  The fully honest nested-CV variant, in which eps is "
                              "chosen inside the training folds, is reported immediately below and is the "
                              "number to quote."),
        result=prim)

    vecN, costN, picksN, epsN = nested_eps(arms_avail, ARMS,
                                           lambda *a: CONV["A_as_charged"][0](*a, R32_CHARGED), EPS)
    cgN = {}
    for conv2, (fn2, _) in CONV.items():
        for Rlab, R in (("R32_4.57", R32_CHARGED), ("R32_3.816", R32_DERIVED)):
            cc = [float(np.mean([fn2(*ARMS[c][a], R) for a in picksN[c]])) for c in CELLS]
            cgN[f"{conv2}|{Rlab}"] = dict(macro_flops=round(float(np.mean(cc)), 4),
                                          x_direct=round(float(np.mean(cc)) / R, 4))
    nested = evaluate("nested_cv_costfloor", vecN, cgN, "FULLY HONEST -- eps chosen inside the training folds",
                      dict(eps_per_outer_fold=epsN,
                           picks={c: sorted(set(picksN[c])) for c in CELLS},
                           seven_b_macro_weight=round(float(np.mean(
                               [np.mean([1.0 if ARMS[c][a][0] > 0 else 0.0 for a in picksN[c]])
                                for c in CELLS])), 4)))
    nested["cost"] = cgN
    rep["PRIMARY_ENDPOINT_HONEST_NESTED_CV"] = nested

    # ---------------- eval-visible upper bound (DIAGNOSTIC) ----------------
    diag = {}
    for e in EPS:
        assign = evalvisible_assign(e, arms_avail, ARMS, lambda *a: CONV["A_as_charged"][0](*a, R32_CHARGED))
        r = evaluate(f"evalvisible_eps{e}", policy_vectors(assign), cost_grid(assign),
                     "DIAGNOSTIC -- eval-visible, an UPPER BOUND, not deployable",
                     dict(assignment=assign))
        diag[str(e)] = r
    rep["evalvisible_diagnostic"] = dict(
        warning="EVERY row here is fitted with full eval visibility.  It is an upper bound on what the "
                "cross-fit selector could reach, and must never be quoted as a result.",
        rows=diag)

    # ---------------- L3: is N reducible? (offline, transfer dumps) ----------------
    rep["L3_n_reduction"] = l3_analysis()

    # ---------------- latency / energy of the selected policies (RE-DERIVED, not inherited) -------
    KLE = latency_energy_constants()
    rep["latency_energy_constants"] = KLE

    def lat_energy_of(picks_or_assign):
        """picks_or_assign: cell -> arm name, or cell -> list of per-fold arm names."""
        ac, cr = [], []
        for c in CELLS:
            arms = picks_or_assign[c]
            arms = [arms] if isinstance(arms, str) else arms
            a_, c_ = [], []
            for a in arms:
                b, m = cell_arm_lat_energy(c, a, KLE, ARMS)
                a_.append(b); c_.append(m)
            ac.append(np.mean(a_, axis=0)); cr.append(np.mean(c_, axis=0))
        return np.mean(ac, axis=0), np.mean(cr, axis=0)

    REF = {
        "always_32b_direct": dict(as_charged=[665.0, 665.0, 127.0], label="the bar"),
        "always_32b_reasoning_as_charged": dict(
            as_charged=[PUB["cost_macro"][REAS]["lat_par_ms"], PUB["cost_macro"][REAS]["lat_seq_ms"],
                        PUB["cost_macro"][REAS]["energy_j"]],
            label="flat reasoning constant (the paper's charge)"),
        "always_32b_reasoning_honest_recost": dict(
            as_charged=[PUB["cost_macro_honest"][REAS]["lat_par_ms"], PUB["cost_macro_honest"][REAS]["lat_seq_ms"],
                        PUB["cost_macro_honest"][REAS]["energy_j"]],
            label="honest_recosting_2026-07-29: each cell charged its OWN measured generation length"),
    }
    le = {}
    for lab, pk in (("deployed_accuracy_max", {c: "method_accuracy_max_veto" for c in CELLS}),
                    ("crossfit_eps0_MOST_ROBUST",
                     crossfit_assign(0.0, arms_avail, ARMS,
                                     lambda *a: CONV["A_as_charged"][0](*a, R32_CHARGED))[2]),
                    ("crossfit_primary_eps%s" % (prim["eps"] if prim else "na"),
                     crossfit_assign(prim["eps"] if prim else 0.0, arms_avail, ARMS,
                                     lambda *a: CONV["A_as_charged"][0](*a, R32_CHARGED))[2]),
                    ("nested_cv", picksN)):
        a_, c_ = lat_energy_of(pk)
        row = dict(as_charged=dict(lat_par_ms=round(a_[0], 1), lat_seq_ms=round(a_[1], 1), energy_j=round(a_[2], 1)),
                   measurement_corrected=dict(lat_par_ms=round(c_[0], 1), lat_seq_ms=round(c_[1], 1),
                                              energy_j=round(c_[2], 1)))
        for rlab, r in REF.items():
            v = r["as_charged"]
            row[f"vs_{rlab}"] = dict(
                as_charged=dict(lat_par_pct=round((a_[0] / v[0] - 1) * 100, 1),
                                lat_seq_pct=round((a_[1] / v[1] - 1) * 100, 1),
                                energy_pct=round((a_[2] / v[2] - 1) * 100, 1)),
                measurement_corrected=dict(lat_par_pct=round((c_[0] / v[0] - 1) * 100, 1),
                                           lat_seq_pct=round((c_[1] / v[1] - 1) * 100, 1),
                                           energy_pct=round((c_[2] / v[2] - 1) * 100, 1)),
                reference=r["label"])
        le[lab] = row
    rep["latency_energy_rederived"] = dict(
        warning=("RE-DERIVED for each policy.  The published -87.9%% latency / -84.3%% energy belong to the "
                 "SHIPPED accuracy-max arm and must not be carried over to any policy below.  Latency and "
                 "energy are MEASURED currencies; FLOP-eq is a MODEL.  Never mix them."),
        reference_points={k: dict(v) for k, v in REF.items()}, policies=le)

    # ---------------- sample-weighted cost (labelled, never paired with the macro accuracy) -------
    tot = sum(N[c] for c in CELLS)
    sw = {}
    for lab, pk in (("deployed_accuracy_max", {c: ["method_accuracy_max_veto"] for c in CELLS}),
                    ("nested_cv", picksN)):
        for conv, (fn, _) in CONV.items():
            v = sum(N[c] / tot * float(np.mean([fn(*ARMS[c][a], R32_CHARGED)
                                                for a in (pk[c] if isinstance(pk[c], list) else [pk[c]])]))
                    for c in CELLS)
            sw[f"{lab}|{conv}"] = dict(sample_weighted_flops=round(v, 4), x_direct=round(v / R32_CHARGED, 4))
    rep["sample_weighted_cost"] = dict(
        warning=("SAMPLE-WEIGHTED cost answers a DIFFERENT question from the macro cost and MUST NEVER be "
                 "paired with the macro accuracy above.  On traffic resembling this suite this is what you "
                 "would pay; the macro number instead tests whether the saving generalises across task types."),
        rows=sw)

    # ---------------- Pareto: non-dominated set in (macro accuracy, as-charged macro cost) --------
    pts = []
    for lab, r in shipped.items():
        pts.append(dict(label=lab, acc=r["macro_acc"], x_direct=r["cost"]["A_as_charged|R32_4.57"]["x_direct"],
                        kind="shipped/baseline"))
    for r in frontier["A_as_charged"]:
        pts.append(dict(label=f"crossfit_eps{r['eps']}", acc=r["macro_acc"],
                        x_direct=r["cost"]["A_as_charged|R32_4.57"]["x_direct"], kind="cross-fit (held out)"))
    pts.append(dict(label="always_32b_direct", acc=round(base_pt, 4), x_direct=1.0, kind="THE BAR"))
    for p in pts:
        p["non_dominated"] = not any((q["acc"] >= p["acc"] and q["x_direct"] <= p["x_direct"]
                                      and (q["acc"] > p["acc"] or q["x_direct"] < p["x_direct"])) for q in pts)
    rep["pareto_as_charged"] = dict(
        axes="macro accuracy (8 cells, 1/8 each) vs macro AS-CHARGED FLOP-eq, x always-32B-direct",
        note=("'Pareto-DOMINATES' is retired (retrospective section 10.1 C26) and is not revived here.  "
              "These points are marked non-dominated within THIS enumerated set only."),
        SEED_WARNING=("Every cross-fit point here is a SINGLE fold seed (%d).  The 12-seed sweep in "
                      "seed_robustness shows the accuracy of these points moves by ~0.001 sd and the cost by "
                      "~0.02-0.03 sd, which is enough to change which points are non-dominated and enough to "
                      "move points across the tie boundary.  Read this frontier as a shape, not as a set of "
                      "operating points, and take every quotable number from seed_robustness instead." % SEED),
        points=sorted(pts, key=lambda p: p["x_direct"]))

    # ---------------- >=10 fold seeds (protocol rule 4: the selector IS fitted) --------------------
    rep["seed_robustness"] = seed_sweep(arms_avail, ARMS, EPS, nseeds=12)

    # ---------------- rule 2: does MEASUREMENT corroborate the re-costing? -------------------------
    rep["rule2_corroboration"] = corroborate(rep["null_tests"]["N2"]["prefill_share_7b"],
                                             rep["null_tests"]["N2"]["decode_share_7b"])

    # ---------------- L3 quantified ---------------------------------------------------------------
    l3 = rep["L3_n_reduction"]
    save = 0.0
    for c in ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]:
        k = l3[c]["k_within_0p005_of_k8"]
        nn = OCD[c]["meanN"]
        save += 2.0 * max(0.0, nn - k)          # 1 generation + 1 verifier forward per dropped sample
    rep["L3_n_reduction"]["quantified"] = dict(
        macro_flops_saved_upper_bound=round(save / 8.0, 4),
        x_direct_saved_upper_bound=round(save / 8.0 / R32_CHARGED, 4),
        finding=("L3 IS NOT A LEVER HERE.  The deployed open arm already runs an ADAPTIVE N (Weitzman) at "
                 "meanN = %.2f / %.2f / %.2f, i.e. BELOW the k at which selected accuracy is within 0.005 "
                 "of k=8 (%d / %d / %d).  Capping N at the plateau therefore removes almost nothing, and "
                 "the figure above is an UPPER BOUND because it holds escalation fixed."
                 % (OCD["SLAKE_open"]["meanN"], OCD["VQA_RAD_open"]["meanN"], OCD["PATH_VQA_open"]["meanN"],
                    l3["SLAKE_open"]["k_within_0p005_of_k8"], l3["VQA_RAD_open"]["k_within_0p005_of_k8"],
                    l3["PATH_VQA_open"]["k_within_0p005_of_k8"])))

    # ---------------- VERDICT against the pre-registered success / kill criteria -------------------
    SR = rep["seed_robustness"]["per_eps"]
    robust = [(float(e), r) for e, r in SR.items()
              if int(r["seeds_with_tie_preserved"].split("/")[0]) >= 10]
    major = [(float(e), r) for e, r in SR.items()
             if int(r["seeds_with_tie_preserved"].split("/")[0]) >= 7]
    cheap_rob = min(robust, key=lambda t: t[1]["x_direct_as_charged"]["mean"]) if robust else None
    cheap_maj = min(major, key=lambda t: t[1]["x_direct_as_charged"]["mean"]) if major else None
    rep["VERDICT"] = dict(
        primary_endpoint_as_charged=dict(
            single_seed_cheapest_tie=dict(eps=prim["eps"], x_direct=prim["cost"]["A_as_charged|R32_4.57"]["x_direct"],
                                          delta=prim["delta_vs_direct"], lo=prim["lo"], hi=prim["hi"]),
            honest_nested_cv=dict(x_direct=nested["cost"]["A_as_charged|R32_4.57"]["x_direct"],
                                  delta=nested["delta_vs_direct"], lo=nested["lo"], hi=nested["hi"],
                                  tie_preserved=nested["tie_preserved"]),
            twelve_seed_cheapest_tie_10of12=dict(
                eps=cheap_rob[0], x_direct=cheap_rob[1]["x_direct_as_charged"]["mean"],
                sd=cheap_rob[1]["x_direct_as_charged"]["sd"]) if cheap_rob else None,
            twelve_seed_cheapest_tie_7of12=dict(
                eps=cheap_maj[0], x_direct=cheap_maj[1]["x_direct_as_charged"]["mean"],
                sd=cheap_maj[1]["x_direct_as_charged"]["sd"]) if cheap_maj else None,
            verdict=("SUCCESS THRESHOLD NOT MET on the as-charged convention.  A SINGLE fold seed reaches "
                     "%.3fx, but across 12 fold seeds the PRE-REGISTERED tie (lower CI bound >= -0.0029) "
                     "survives on 10/12 seeds only up to eps where the cost is %.3fx, and on a bare 7/12 "
                     "majority at %.3fx -- both ABOVE 1.00x.  The sub-1.0x single-seed number is FITTING "
                     "NOISE and must not be quoted."
                     % (prim["cost"]["A_as_charged|R32_4.57"]["x_direct"],
                        cheap_rob[1]["x_direct_as_charged"]["mean"] if cheap_rob else float("nan"),
                        cheap_maj[1]["x_direct_as_charged"]["mean"] if cheap_maj else float("nan"))),
            secondary_standard_non_inferiority=dict(
                criterion="the 95% CI merely spans zero -- 'not SIGNIFICANTLY worse than always-32B-direct'.  "
                          "This is LOOSER than the pre-registered rule and the verdict is NOT decided on it.",
                eps=0.01, x_direct_as_charged=SR["0.01"]["x_direct_as_charged"]["mean"],
                macro_acc=SR["0.01"]["macro_acc"]["mean"],
                delta_point=round(SR["0.01"]["macro_acc"]["mean"] - 0.656672, 4),
                delta_ci=[SR["0.01"]["delta_lo"]["mean"], SR["0.01"]["delta_hi"]["mean"]],
                seeds=SR["0.01"]["seeds_not_significantly_worse"],
                reading=("Under the ORDINARY reading of 'ties always-32B-direct', 0.934x as-charged IS "
                         "reachable, on 11/12 seeds, at a point-estimate accuracy loss of -0.0016 macro.  "
                         "The pre-registered criterion was deliberately stricter than that and it is the "
                         "one the verdict is decided on.  Both are printed so no later reader can pick "
                         "whichever suits them without seeing the other."))),
        what_DID_clear_the_bar=dict(
            zero_eps_crossfit=dict(
                description=("eps = 0 cross-fit per-cell arm selection -- 'deploy the arm the training folds "
                             "say is most accurate' -- is not a cost-motivated choice at all, and it is the "
                             "single most robust row in the table (tie on 10/12 seeds)."),
                macro_acc=SR["0.0"]["macro_acc"]["mean"], macro_acc_sd=SR["0.0"]["macro_acc"]["sd"],
                x_direct_as_charged=SR["0.0"]["x_direct_as_charged"]["mean"],
                x_direct_recost_gen=SR["0.0"]["x_direct_recost_gen"]["mean"],
                x_direct_recost_full=SR["0.0"]["x_direct_recost_full"]["mean"],
                vs_shipped_accuracy_max=("shipped accuracy-max is 0.6575 at 1.740x as-charged; this is "
                                         "%.4f at %.3fx -- equal accuracy for a 1.49x cost reduction, with "
                                         "no re-costing argument involved at all."
                                         % (SR["0.0"]["macro_acc"]["mean"], SR["0.0"]["x_direct_as_charged"]["mean"]))),
            latency_and_energy=dict(
                description=("RE-DERIVED per policy.  eps=0 reaches LATENCY parity with always-32B-direct "
                             "(+0.6% parallel) but still pays +25.5% energy; the cheaper eps=0.01 policy "
                             "reaches parity on BOTH (-1.6% latency, -3.1% energy).  The shipped arm reaches "
                             "parity on neither (+16.7% latency, +101% energy as-charged; +47.2% / +55.7% "
                             "once the 2026-08-03 batch-8 measurement replaces the modelled BO8 constants).  "
                             "Against a 32B actually made to reason, honestly re-costed, eps=0 is -89.4% "
                             "latency and -90.2% energy."),
                crossfit_eps0_vs_direct=le["crossfit_eps0_MOST_ROBUST"]["vs_always_32b_direct"],
                crossfit_eps0p01_vs_direct=le[[k for k in le if k.startswith("crossfit_primary")][0]]["vs_always_32b_direct"],
                crossfit_eps0_vs_reasoning_honest=le["crossfit_eps0_MOST_ROBUST"]["vs_always_32b_reasoning_honest_recost"],
                shipped_vs_direct=le["deployed_accuracy_max"]["vs_always_32b_direct"])),
        kill_criteria=dict(
            i_measurement_rejects_recosting=rep["rule2_corroboration"]["verdict"],
            ii_cheapest_tie_ge_1p2x=dict(
                fired=bool(cheap_maj and cheap_maj[1]["x_direct_as_charged"]["mean"] >= 1.2),
                value=cheap_maj[1]["x_direct_as_charged"]["mean"] if cheap_maj else None,
                note="kill (ii) as worded requires >= 1.2x; the measured floor is below that, so this kill "
                     "does NOT fire -- but neither does SUCCESS.  The honest report is the in-between one."),
            iii_degenerate=dict(
                seven_b_macro_weight_at_the_cheapest_robust_tie=(cheap_rob[1]["seven_b_macro_weight"]["mean"]
                                                                 if cheap_rob else None),
                degenerate=bool(cheap_rob and cheap_rob[1]["seven_b_macro_weight"]["mean"] < 0.20),
                note="pre-committed: < 0.20 macro weight on the 7B is reported as degenerate, not as a win")),
        the_uncomfortable_composition=(
            "Every cheap tie-preserving policy reaches its cost by ROUTING THE OPEN-TEXT CELLS TO "
            "always-32B-direct.  SLAKE_open and VQA_RAD_open go to 32B-direct at EVERY eps and EVERY seed, "
            "because 32B-direct is BOTH more accurate AND cheaper than the 7B best-of-N arm on those two "
            "cells (0.8186 vs 0.8171 at 4.57 vs 13.97 FLOP-eq; 0.6000 vs 0.5900 at 4.57 vs 17.30).  The "
            "open-text best-of-N machinery survives only on PATH_VQA_open, and only at some eps/seeds.  "
            "That is a cost result about the METHOD, not about the accounting, and it must be reported "
            "with the cost number."))

    # ---------------- degeneracy check (pre-committed) ----------------
    for lab, r in (("PRIMARY_ENDPOINT", prim), ("NESTED_CV", nested)):
        if r is None:
            continue
        w = r.get("seven_b_macro_weight")
        rep.setdefault("degeneracy_check", {})[lab] = dict(
            seven_b_macro_weight=w,
            degenerate=bool(w is not None and w < 0.20),
            pre_committed_rule="a policy that runs the 7B on <20% of macro weight is reported as a "
                               "DEGENERATE SOLUTION, not as a win")
    SRr = rep["seed_robustness"]["per_eps"]
    rep["HEADLINE_TABLE"] = dict(
        columns=["policy", "macro_acc", "delta_vs_32B_direct [95% CI]", "as-charged xdirect",
                 "recost-gen xdirect (UNCORROBORATED)", "recost-full xdirect (UNCORROBORATED, needs an "
                 "implementation change)", "lat_par ms", "energy J", "pre-registered tie?"],
        rows=[
            ["always-32B-direct  (THE BAR)", round(base_pt, 4), "0 (reference)", 1.0, 1.0, 1.0, 665.0, 127.0, "-"],
            ["always-32B-reasoning", shipped["baseline_always_32b_reasoning"]["macro_acc"],
             "%+.4f [%+.4f,%+.4f]" % (shipped["baseline_always_32b_reasoning"]["delta_vs_direct"],
                                      shipped["baseline_always_32b_reasoning"]["lo"],
                                      shipped["baseline_always_32b_reasoning"]["hi"]),
             1.0, 1.0, 1.0, PUB["cost_macro"][REAS]["lat_par_ms"], PUB["cost_macro"][REAS]["energy_j"], "-"],
            ["SHIPPED accuracy-max (no change)", shipped["deployed_accuracy_max"]["macro_acc"],
             "%+.4f [%+.4f,%+.4f]" % (shipped["deployed_accuracy_max"]["delta_vs_direct"],
                                      shipped["deployed_accuracy_max"]["lo"], shipped["deployed_accuracy_max"]["hi"]),
             1.740, 1.373, 1.024, le["deployed_accuracy_max"]["as_charged"]["lat_par_ms"],
             le["deployed_accuracy_max"]["as_charged"]["energy_j"], "yes"],
            ["cross-fit arm selection, eps=0 (12-seed mean)", SRr["0.0"]["macro_acc"]["mean"],
             "%+.4f [%+.4f,%+.4f]" % (SRr["0.0"]["macro_acc"]["mean"] - base_pt,
                                      SRr["0.0"]["delta_lo"]["mean"], SRr["0.0"]["delta_hi"]["mean"]),
             SRr["0.0"]["x_direct_as_charged"]["mean"], SRr["0.0"]["x_direct_recost_gen"]["mean"],
             SRr["0.0"]["x_direct_recost_full"]["mean"],
             le["crossfit_eps0_MOST_ROBUST"]["as_charged"]["lat_par_ms"],
             le["crossfit_eps0_MOST_ROBUST"]["as_charged"]["energy_j"], "yes on 10/12 seeds"],
            ["cross-fit, eps=0.005 (12-seed mean)", SRr["0.005"]["macro_acc"]["mean"],
             "%+.4f [%+.4f,%+.4f]" % (SRr["0.005"]["macro_acc"]["mean"] - base_pt,
                                      SRr["0.005"]["delta_lo"]["mean"], SRr["0.005"]["delta_hi"]["mean"]),
             SRr["0.005"]["x_direct_as_charged"]["mean"], SRr["0.005"]["x_direct_recost_gen"]["mean"],
             SRr["0.005"]["x_direct_recost_full"]["mean"], "-", "-", "yes on 7/12 seeds only"],
            ["cross-fit, eps=0.01 (12-seed mean)", SRr["0.01"]["macro_acc"]["mean"],
             "%+.4f [%+.4f,%+.4f]" % (SRr["0.01"]["macro_acc"]["mean"] - base_pt,
                                      SRr["0.01"]["delta_lo"]["mean"], SRr["0.01"]["delta_hi"]["mean"]),
             SRr["0.01"]["x_direct_as_charged"]["mean"], SRr["0.01"]["x_direct_recost_gen"]["mean"],
             SRr["0.01"]["x_direct_recost_full"]["mean"],
             le[[k for k in le if k.startswith("crossfit_primary")][0]]["as_charged"]["lat_par_ms"],
             le[[k for k in le if k.startswith("crossfit_primary")][0]]["as_charged"]["energy_j"],
             "NO -- 1/12 seeds (but not significantly worse on 11/12)"],
        ],
        cost_label=("EVERY cost column is MACRO-weighted (8 cells, 1/8 each) and is paired only with the "
                    "MACRO accuracy in the same row.  The sample-weighted cost is reported separately and "
                    "answers a different question."))
    rep["handoff"] = dict(
        pending_measurement=("runners/run_cost_floor_measure.sh is left running with nohup.  It polls for a "
                             "genuinely CLEAN GPU (util < 8%, > 30 GiB free, sustained over three 60 s polls) "
                             "and then runs src/cascade_methods/cost_floor_measure.py.  RE-RUNNING "
                             "`python3 src/cascade_methods/cost_floor.py` after it lands fills in "
                             "rule2_corroboration automatically from "
                             "results/cascade_methods/artifacts/_cost_floor_measure/*.jsonl."),
        what_it_will_settle=("(a) whether vLLM V1's SamplingParams(n=N) actually shares the prefill for this "
                             "multimodal prompt -- read off RequestOutput.num_cached_tokens, an EXACT counter "
                             "immune to GPU contention; (b) the NVML energy and wall-clock of the "
                             "shared-prefill path, which needs an uncontended card to be valid."),
        why_it_did_not_run=("Both A100s were held by the sibling attacks of this round for the whole session, "
                            "oscillating between 19 and 75 GiB of the 80 GiB card.  Two launch attempts were "
                            "aborted rather than race them, because OOM-ing another agent's multi-hour "
                            "generation run is a worse outcome than a pending measurement."))
    rep["limitations"] = [
        "The policy family is limited to the SHIPPED operating points, because vec_disjoint.npz stores only "
        "per-item CORRECTNESS, not the per-item gate features.  Re-tuning an escalation threshold inside an "
        "arm is therefore out of reach of this attack; the eps sweep spans the arms, not the thresholds.",
        "Conventions B and C are DERIVED and, in this session, UNCORROBORATED: both A100s were held by the "
        "sibling attacks and NVML energy on a contended card is invalid.  They must not be used as a headline "
        "until src/cascade_methods/cost_floor_measure.py has run on a clean card.",
        "Convention C additionally assumes a verifier that prefix-caches the shared image+question prompt.  "
        "The deployed verifier does NOT do this (it runs one HF batched forward over 8 full prompts), so C "
        "prices an implementation the project has never run.",
        "The measurement-corrected latency/energy uses a LINEAR interpolation in N between the measured N=1 "
        "and N=8 points; only those two points were measured.",
        "vqa_rad_open is n=200 and its guardrail resolution is poor (clean-seed counts have run 0/10 to 7/10 "
        "across seeds elsewhere in this project); a guardrail flag on that cell is within seed noise.",
        "R32 = 4.57 is the paper constant and it FLATTERS the method; the derived R32 = 3.816 makes every "
        "ratio here worse and is reported alongside every headline.",
    ]
    json.dump(rep, open(OUT, "w"), indent=1)
    print("\n-> " + OUT)


def corroborate(pre7, dec7):
    """INTEGRITY RULE 2.  Compare the MEASURED best-of-N cost ratio (energy and wall-clock, batch-1
    request, NVML) against what the re-costed FLOP model predicts.  If the measurement exceeds the
    model by more than 30%, the re-costing is REJECTED."""
    d = os.path.join(ART, "_cost_floor_measure")
    out = dict(rule="if measured BoN@8 energy exceeds the re-costed model by > 30%, REJECT the re-costing",
               MECHANISM_AUDIT_THAT_UNDERCUTS_L1=dict(
                   status="CODE INSPECTION, not a measurement -- but it contradicts the brief's premise",
                   the_briefs_premise=("'run_openvqa.py:154 really generates with vLLM SamplingParams(n=N), "
                                       "i.e. the deployed generation path SHARES the prefill.'"),
                   what_the_installed_vllm_actually_does=(
                       "In vLLM V1, SamplingParams(n=N) is NOT a post-prefill sequence fork.  "
                       "vllm/v1/engine/parallel_sampling.py:ParentRequest splits the request into N CHILD "
                       "REQUESTS, each carrying the SAME full prompt and n=1, and prefill sharing then "
                       "depends entirely on AUTOMATIC PREFIX CACHING.  V1 is the default engine in the "
                       "vLLM 0.9.0.1 the deployed run_openvqa.py used "
                       "(vllm/engine/arg_utils.py:991, 'if VLLM_USE_V1 is unset, we enable V1 for supported "
                       "features'), and Qwen2.5-VL is a supported model."),
                   why_that_may_void_the_credit=(
                       "The N siblings arrive simultaneously and 8 x ~327 = ~2.6k tokens is well inside the "
                       "V1 scheduler's token budget, so all N are eligible for the SAME scheduling step.  A "
                       "V1 prefix-cache block is only committed after the request that computed it returns, "
                       "so N siblings scheduled together can each compute the full prefill and each MISS the "
                       "cache.  If that is what happens, the deployed generation really did cost N prefills, "
                       "the as-charged N x 1.0 generation charge is approximately RIGHT, and lever L1's "
                       "generation credit is VOID."),
                   the_verifier_half_is_definitely_not_shared=(
                       "Independently of the above, the deployed verifier scores the 8 candidates as ONE HF "
                       "batched forward over 8 FULL prompts (bestofn_measure_batch8.py verify8: 'ONE batched "
                       "verifier forward over 8 candidates').  A batch does not reduce FLOPs.  So the "
                       "verifier half of BO8 = 16.0 is correctly charged at N x 1.0 as-charged, and only "
                       "convention C -- which prices a prefix-caching verifier the project has NEVER built "
                       "-- moves it."),
                   the_two_minute_experiment_that_settles_it=(
                       "src/cascade_methods/cost_floor_measure.py --phase vllm records "
                       "RequestOutput.num_cached_tokens.  If sum(cached_tok)/sum(prompt_tok) ~ 7/8 for "
                       "n=8, the prefill IS shared and convention B is correct; if it is ~0, it is not and "
                       "L1 is dead.  That counter is EXACT and immune to GPU contention.  It did not run "
                       "because both A100s were saturated by the sibling attacks for the whole session."),
                   consequence_for_this_attack=(
                       "The PRIMARY endpoint is reported on the AS-CHARGED convention and is therefore "
                       "UNAFFECTED.  Conventions B and C are downgraded from 'corrections' to HYPOTHETICAL "
                       "re-costings whose premise is unverified and, on this reading, doubtful.")),
               model_predicted_gen8_over_gen1_flop_ratio=round(pre7 + 8 * dec7, 4),
               control_hf_nonshared_prefill=dict(
                   source="results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03.json",
                   gen8_over_gen1_energy=round(BO8A["reconciliation"]["measurement"]["pooled"]["gen8"]["energy_j_mean"]
                                               / BO8A["reconciliation"]["measurement"]["pooled"]["gen1"]["energy_j_mean"], 3),
                   gen8_over_gen1_latency=round(BO8A["reconciliation"]["measurement"]["pooled"]["gen8"]["lat_ms_mean"]
                                                / BO8A["reconciliation"]["measurement"]["pooled"]["gen1"]["lat_ms_mean"], 3),
                   why_it_is_only_a_control=("HF generate(num_return_sequences=8) expands the inputs BEFORE the "
                                             "prefill forward, so this path really does recompute the prefill 8 "
                                             "times.  It measures a DIFFERENT implementation from the one the "
                                             "re-costing describes and CANNOT be used to accept or reject it.")),
               idle_floor_caveat=dict(
                   idle_w_model_resident=BO8A["idle_w_model_resident"],
                   gen1_measured_lat_ms=BO8A["reconciliation"]["measurement"]["pooled"]["gen1"]["lat_ms_mean"],
                   gen1_measured_energy_j=BO8A["reconciliation"]["measurement"]["pooled"]["gen1"]["energy_j_mean"],
                   idle_share_of_gen1_energy_pct=round(
                       BO8A["idle_w_model_resident"]
                       * BO8A["reconciliation"]["measurement"]["pooled"]["gen1"]["lat_ms_mean"] / 1000.0
                       / BO8A["reconciliation"]["measurement"]["pooled"]["gen1"]["energy_j_mean"] * 100, 1),
                   warning=("About half of a batch-1 forward's measured joules is the model-resident IDLE "
                            "floor (83.8 W over a ~350 ms call), not compute.  An energy RATIO therefore "
                            "cannot equal a FLOP ratio at batch 1 even under a perfect model: the floor "
                            "alone pushes any measured ratio above a small modelled one.  Rule 2's '30%' "
                            "test must be read with that in mind, and the pending measurement is reported "
                            "with the idle floor stated so the comparison is interpretable rather than "
                            "mechanical.")))
    rows = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".jsonl") and f.startswith("vllm"):
                for line in open(os.path.join(d, f)):
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if not r.get("warm", True) and "error" not in r:
                        rows.append(r)
    if not rows:
        out["measured_shared_prefill_path"] = None
        out["verdict"] = ("NOT CORROBORATED, AND THE MECHANISM IS NOW IN DOUBT.  The shared-prefill "
                          "measurement never obtained a clean, uncontended GPU window (both A100s were held "
                          "by the sibling attacks all session; NVML energy on a contended card is invalid), "
                          "AND the code audit above shows vLLM V1 implements SamplingParams(n=N) as N child "
                          "requests relying on prefix caching rather than as a post-prefill fork -- so the "
                          "brief's premise that the deployed path shares the prefill is itself unverified.  "
                          "Per integrity rule 2, conventions B and C are HYPOTHETICAL and must not be used "
                          "as a headline.  The AS-CHARGED convention -- on which the primary endpoint is "
                          "reported -- is unaffected, and on the balance of the code evidence the as-charged "
                          "BO8 = 16.0 charge may simply be RIGHT, which is pre-registered kill (i).")
        return out
    agg = {}
    for r in rows:
        k = (r["pc"], r["leg"])
        agg.setdefault(k, []).append(r)
    tab = {}
    for (pc, leg), rs in sorted(agg.items()):
        tab[f"{pc}|{leg}"] = dict(n=len(rs),
                                  lat_ms=round(float(np.mean([x["lat_ms"] for x in rs])), 1),
                                  energy_j=round(float(np.mean([x["energy_j"] for x in rs])), 2),
                                  cached_tok=round(float(np.mean([x.get("cached_tok", 0) for x in rs])), 1),
                                  prompt_tok=round(float(np.mean([x.get("prompt_tok", 0) for x in rs])), 1))
    out["measured_shared_prefill_path"] = tab
    try:
        e1 = tab["on|gen_n1"]["energy_j"]; e8 = tab["on|gen_n8"]["energy_j"]
        ratio = e8 / e1
        model = pre7 + 8 * dec7
        out["measured_gen8_over_gen1_energy"] = round(ratio, 3)
        out["excess_over_model_pct"] = round((ratio / model - 1) * 100, 1)
        out["prefill_sharing_proven"] = bool(tab["on|gen_n8"]["cached_tok"] > 0.5 * tab["on|gen_n8"]["prompt_tok"])
        out["verdict"] = ("CORROBORATED" if ratio <= 1.30 * model else
                          "REJECTED -- measured BoN@8 energy exceeds the re-costed model by more than 30%")
    except KeyError:
        out["verdict"] = "PARTIAL -- some legs missing; see the table"
    return out


def seed_sweep(arms_avail, ARMS, eps_grid, nseeds=12):
    """Protocol rule 4: the arm selector is FITTED, so a single fold seed is not a result.  Re-draws
    the 5-fold split (and its own paired bootstrap stream) nseeds times and reports mean/sd/range."""
    acc = {e: [] for e in eps_grid}; dlo = {e: [] for e in eps_grid}; dhi = {e: [] for e in eps_grid}
    nsw = {e: [] for e in eps_grid}
    cA = {e: [] for e in eps_grid}; cB = {e: [] for e in eps_grid}; cC = {e: [] for e in eps_grid}
    w7 = {e: [] for e in eps_grid}; tie = {e: [] for e in eps_grid}
    CONV = costing(rep["null_tests"]["N2"]["prefill_share_7b"], rep["null_tests"]["N2"]["decode_share_7b"],
                   rep["verifier_geometry"]["ver_prefix_cost_units"],
                   rep["verifier_geometry"]["ver_marginal_cost_per_candidate_units"])
    for s in range(nseeds):
        sd = SEED + 1000 * (s + 1)
        make_folds(sd)
        GRP.clear()
        boot_streams({c: [FOLD_CF[c]] for c in CELLS}, seed=sd)
        _, bb = macro_boot({c: OK[(c, BASE)] for c in CELLS})
        for e in eps_grid:
            vec, _, picks = crossfit_assign(e, arms_avail, ARMS,
                                            lambda *a: CONV["A_as_charged"][0](*a, R32_CHARGED))
            pt, bt = macro_boot(vec)
            lo, hi = ci_of(bt - bb)
            acc[e].append(pt); dlo[e].append(lo); dhi[e].append(hi)
            tie[e].append(bool(lo >= -TIE_TOL)); nsw[e].append(bool(hi > 0))
            for store, conv in ((cA, "A_as_charged"), (cB, "B_recost_gen"), (cC, "C_recost_full")):
                fn = CONV[conv][0]
                store[e].append(float(np.mean([np.mean([fn(*ARMS[c][a], R32_CHARGED) for a in picks[c]])
                                               for c in CELLS])) / R32_CHARGED)
            w7[e].append(float(np.mean([np.mean([1.0 if ARMS[c][a][0] > 0 else 0.0 for a in picks[c]])
                                        for c in CELLS])))

    def ms(v):
        return dict(mean=round(float(np.mean(v)), 4), sd=round(float(np.std(v, ddof=1)), 4),
                    min=round(float(np.min(v)), 4), max=round(float(np.max(v)), 4))
    rows = {str(e): dict(macro_acc=ms(acc[e]), delta_lo=ms(dlo[e]), delta_hi=ms(dhi[e]),
                         seeds_not_significantly_worse=f"{sum(nsw[e])}/{nseeds}",
                         x_direct_as_charged=ms(cA[e]), x_direct_recost_gen=ms(cB[e]),
                         x_direct_recost_full=ms(cC[e]),
                         seven_b_macro_weight=ms(w7[e]),
                         seeds_with_tie_preserved=f"{sum(tie[e])}/{nseeds}") for e in eps_grid}
    make_folds()            # restore the primary fold assignment
    GRP.clear()
    boot_streams({c: [FOLD_CF[c], FOLD_NS[c]] for c in CELLS})
    return dict(n_seeds=nseeds, seeds=[SEED + 1000 * (s + 1) for s in range(nseeds)],
                note=("Each seed re-draws BOTH the 5-fold split and its own paired bootstrap stream, so the "
                      "spread below is the honest fitting noise of the selector.  The deployable number is "
                      "the seed MEAN, and a row is only reportable as a tie if it preserves the tie on a "
                      "clear majority of seeds."),
                two_criteria=dict(
                    PRE_REGISTERED_PRIMARY="seeds_with_tie_preserved: 95% CI lower bound >= -0.0029.  This "
                                           "is the criterion the verdict is decided on.",
                    SECONDARY_STANDARD="seeds_not_significantly_worse: the 95% CI merely spans zero, i.e. "
                                       "the policy is not SIGNIFICANTLY worse than always-32B-direct.  This "
                                       "is the ordinary non-inferiority reading and it is LOOSER.  It is "
                                       "reported for transparency and is NOT the pre-registered endpoint; "
                                       "the verdict above is not decided on it."),
                per_eps=rows)


def l3_analysis():
    """Selected / oracle accuracy as a function of pool size k, per open reporting cell, from the CLEAN
    disjoint verifier's transfer dumps.  Escalation is held FIXED, which makes the implied saving an
    UPPER BOUND (a smaller pool would in general raise escalation)."""
    out = {}
    for cell, fn in (("SLAKE_open", "slake"), ("VQA_RAD_open", "vqa_rad"), ("PATH_VQA_open", "pathvqa")):
        d = json.load(open(os.path.join(REPO, f"ckpts/train/lora_verifier_disjoint/"
                                              f"transfer_dump_{fn}_open_lingshu7b.json")))
        sl = np.array([r["sl"] for r in d], float)
        sc = np.array([r["scores"] for r in d], float)
        rows = {}
        for k in range(1, 9):
            pick = sc[:, :k].argmax(axis=1)
            rows[k] = dict(selected=round(float(sl[np.arange(len(sl)), pick].mean()), 4),
                           oracle=round(float((sl[:, :k].max(axis=1)).mean()), 4))
        sel8 = rows[8]["selected"]
        plateau = min([k for k in rows if rows[k]["selected"] >= sel8 - 0.005] or [8])
        out[cell] = dict(n=len(d), by_k=rows, k_within_0p005_of_k8=plateau,
                         note="escalation held fixed; a smaller pool would in general raise escalation, so "
                              "any cost saving read off this table is an UPPER BOUND")
    return out


if __name__ == "__main__":
    main()
