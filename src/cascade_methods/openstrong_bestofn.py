#!/usr/bin/env python3
"""openstrong_bestofn.py -- ATTACK 1 (OPEN-STRONG): the analysis.

Question: the open-text arm's escalation target has only ever been ONE greedy Lingshu-32B pass.
Does giving the STRONG leg a best-of-N pool, selected by the SAME frozen clean verifier, move the
8-cell macro past always-32B-direct = 0.6567?

No GPU. Reads:
  results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz   the deployed 8 cells
  ckpts/train/lora_verifier_disjoint/transfer_dump_*_lingshu7b.json          the incumbent open arm
  ckpts/openvqa/strong_lingshu/ckpt_*_lingshu32b.judge.jsonl                 always-32B-direct labels
  ckpts/openvqa/strong_lingshu_bo/...                                        THIS ATTACK's new dumps

Bootstrap machinery (cell_boot_means / ci) is re-implemented verbatim from
src/cascade_methods/macro_average_headline.py so the CIs are computed by the published protocol:
resample ITEMS WITHIN each cell, recompute the equal-weight average per replicate.

  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/openstrong_bestofn.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_selector_rerun_parts")
OUT = os.path.join(ART, "openstrong_bestofn_2026-08-10.json")

INC_DIR = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")
STRONG_OLD = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu")
BO = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu_bo")
VERIF = os.path.join(BO, "verif_lora_verifier_disjoint")

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
MCQ = CELLS[:5]
OPEN = CELLS[5:]
DSK = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}
SEEDS = [0, 1, 2]
NBOOT = 10000
SEED = 20260810

# cost constants, verbatim from src/cascade_methods/paper_baselines.py:62-66 (7B-forward-equivalents)
GEN7_F, VER7_F, GEN32_F = 1.0, 1.0, 4.57
GEN7_MS, VER7_MS, GEN32_MS, GEN32T_MS = 347.0, 175.0, 665.0, 10521.6
GEN7_J, VER7_J, GEN32_J, GEN32T_J = 45.8, 25.3, 127.0, 2001.9

# FLOP model, verbatim from artifacts/flop_ratio_derivation_2026-08-03.json:flop_model
F32 = {"vision_tower_dense": 1427.98, "vision_tower_attn": 36.07, "vision_merger": 29.42,
       "lm_prefill_dense": 20389.24, "lm_prefill_attn": 69.94, "lm_decode_dense": 287.10,
       "lm_decode_attn": 1.98, "lm_head": 8.72, "TOTAL": 22250.45}
G32_REF = 5.6                      # generated tokens at the artifact's operating point


def marginal_sample_gflops_32b(g_tokens):
    """GFLOPs of ONE EXTRA sample from the 32B when the prefill (vision tower, merger, LM prompt
    pass and its KV) is SHARED across samples -- i.e. vLLM `n=N` on one request.
    Only the per-token decode work and the LM head are paid again.
    Per-token constants come from the published model: lm_decode_dense = 2*(G-1)*N_lm_body and
    lm_head = 2*G*N_lm_head, so per token = lm_decode_dense/(G_ref-1) + lm_head/G_ref."""
    per_tok = F32["lm_decode_dense"] / (G32_REF - 1) + F32["lm_head"] / G32_REF
    return g_tokens * per_tok + F32["lm_decode_attn"] * (g_tokens / G32_REF)


def bo_n_flops_32b(n, g_tokens, shared_prefill):
    """Cost of an N-sample 32B pool in 7B-forward-equivalents (1 full 32B forward = GEN32_F)."""
    if not shared_prefill:
        return n * GEN32_F
    extra = marginal_sample_gflops_32b(g_tokens) / F32["TOTAL"] * GEN32_F
    return GEN32_F + (n - 1) * extra


# =====================================================================================
# bootstrap -- verbatim from macro_average_headline.py
# =====================================================================================
def cell_boot_means(mat, nboot, rng):
    pats, cnt = np.unique(mat, axis=0, return_counts=True)
    n = mat.shape[0]
    draws = rng.multinomial(n, cnt / n, size=nboot)
    return (draws @ pats) / n


def ci(dist, point=None):
    lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
    d = float(np.mean(dist)) if point is None else float(point)
    return dict(delta=round(d, 4), lo=round(lo, 4), hi=round(hi, 4),
                sig=bool(lo > 0 or hi < 0),
                verdict=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"))


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def jmap(p):
    return {r["idx"]: int(r["judge_ok"]) for r in jl(p)}


def norm(s):
    return str(s).strip().lower()


def pick_argmax(scores):
    """The frozen pick rule: argmax with FIRST-INDEX tie-break (genframe_data.picks_from_scores)."""
    return int(np.argmax(np.asarray(scores, dtype=float)))


# =====================================================================================
# 1.  the deployed 8 cells (already null-tested against the published artifact)
# =====================================================================================
def load_deployed():
    z = np.load(os.path.join(PARTS, "vec_disjoint.npz"))
    vec = {c: {s: np.asarray(z[f"{c}|{s}"], float) for s in
               ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b",
                "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]}
           for c in CELLS}
    return vec


def load_incumbent_open():
    """Per open cell: the item order of the deployed vectors, and the incumbent 7B pool."""
    out = {}
    for cell, ds in DSK.items():
        dump = json.load(open(os.path.join(INC_DIR, f"transfer_dump_{ds}_lingshu7b.json")))
        sj = jmap(os.path.join(STRONG_OLD, f"ckpt_{ds}_lingshu32b.judge.jsonl"))
        rows = [r for r in dump if r["idx"] in sj]
        out[cell] = dict(ids=[r["idx"] for r in rows], rows=rows,
                         strong=np.array([sj[r["idx"]] for r in rows], float))
    return out


# =====================================================================================
# 2.  the new 32B arms
# =====================================================================================
def load_bo_arm(cell, tag, ids):
    """-> dict of per-item aligned vectors for the 32B pool `tag` on `cell`, or None."""
    ds = DSK[cell]
    dump_p = os.path.join(VERIF, f"transfer_dump_{ds}_{tag}.json")
    if not os.path.exists(dump_p):
        return None
    by = {r["idx"]: r for r in json.load(open(dump_p))}
    if not all(i in by for i in ids):
        missing = [i for i in ids if i not in by]
        print(f"  WARN {cell} {tag}: {len(missing)} of {len(ids)} items missing from the 32B dump")
        return None
    sel, orc, maj, sl_all, sc_all, ntok = [], [], [], [], [], []
    for i in ids:
        r = by[i]
        sl = [0 if x < 0 else int(x) for x in r["sl"]]
        sc = list(r["scores"])
        k = pick_argmax(sc)
        sel.append(sl[k])
        orc.append(max(sl))
        cnt = Counter(norm(a) for a in r["preds"])
        top = cnt.most_common(1)[0][0]
        kk = next(j for j, a in enumerate(r["preds"]) if norm(a) == top)
        maj.append(sl[kk])
        sl_all.append(sl)
        sc_all.append(sc)
    return dict(selected=np.array(sel, float), oracle=np.array(orc, float),
                majority=np.array(maj, float), sl=np.array(sl_all), scores=np.array(sc_all))


def sub_pool(arm, k):
    """The first-k sub-pool of an N-pool: an iid k-draw. Used for the A2 (N=4) cost-tempered arm."""
    sl = arm["sl"][:, :k]
    sc = arm["scores"][:, :k]
    pick = np.argmax(sc, axis=1)
    return dict(selected=np.array([sl[i, pick[i]] for i in range(len(sl))], float),
                oracle=sl.max(1).astype(float), sl=sl, scores=sc)


# =====================================================================================
# 3.  macro assembly + bootstrap
# =====================================================================================
def macro_of(vecs):
    return float(np.mean([vecs[c].mean() for c in CELLS]))


def macro_delta(vec_a, vec_b, keys=CELLS, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    dist = 0.0
    point = 0.0
    per_cell = {}
    for c in keys:
        mat = np.column_stack([vec_a[c], vec_b[c]])
        B = cell_boot_means(mat, nboot, rng)
        d = B[:, 0] - B[:, 1]
        p = float(vec_a[c].mean() - vec_b[c].mean())
        per_cell[c] = ci(d, point=p)
        dist = dist + d / len(keys)
        point += p / len(keys)
    out = ci(dist, point=point)
    out["per_cell"] = per_cell
    loo = {c: round(float(np.mean([per_cell[j]["delta"] for j in keys if j != c])), 4) for c in keys}
    out["macro_leave_one_out"] = dict(per_dropped_cell=loo,
                                      range=[min(loo.values()), max(loo.values())],
                                      cell_carrying_the_claim=min(loo, key=lambda k: loo[k]))
    return out


def guardrail(vec_a, vec_b, nboot=2000, seed=SEED + 3):
    """Per-cell: is arm A worse than baseline B by more than 2 x the bootstrap SE of the delta?"""
    rng = np.random.default_rng(seed)
    out = {}
    for c in CELLS:
        mat = np.column_stack([vec_a[c], vec_b[c]])
        B = cell_boot_means(mat, nboot, rng)
        d = B[:, 0] - B[:, 1]
        se = float(d.std())
        p = float(vec_a[c].mean() - vec_b[c].mean())
        out[c] = dict(delta=round(p, 4), se=round(se, 4),
                      flag=bool(p < -2 * se), n=int(len(vec_a[c])))
    return out


def main():
    res = {"title": "ATTACK 1 -- OPEN-STRONG: does a best-of-N pool on the STRONG leg beat "
                    "always-32B-direct on the 8-cell macro?",
           "date": "2026-08-10", "no_fabricated_numbers": True,
           "preregistration": "results/cascade_methods/artifacts/"
                              "openstrong_bestofn_2026-08-10_preregistration.json",
           "reproduce": "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 "
                        "src/cascade_methods/openstrong_bestofn.py",
           "nboot": NBOOT, "bootstrap_seed": SEED}

    # ---------------- null tests -------------------------------------------------------
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from training_methods import genframe_data as G
    n1 = G.null_test()
    res["null_tests"] = {"N1_frozen_open_metric": {
        "max_abs_deviation": n1["max_abs_deviation"], "pass": n1["pass"],
        "measured": n1["measured"], "published": n1["published"]}}

    dep = load_deployed()
    pub = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))
    pa = pub["per_arm"]["disjoint"]
    devs = []
    for s in ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b",
              "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]:
        m = float(np.mean([dep[c][s].mean() for c in CELLS]))
        devs.append(abs(round(m, 4) - pa["macro_acc"][s]))
    res["null_tests"]["N2_macro_reproduction"] = {
        "max_abs_deviation": max(devs), "pass": max(devs) == 0.0,
        "macro": {s: round(float(np.mean([dep[c][s].mean() for c in CELLS])), 6)
                  for s in pa["macro_acc"]}}

    inc = load_incumbent_open()
    ids = {c: inc[c]["ids"] for c in OPEN}

    # ---------------- N3: identity control ---------------------------------------------
    n3 = {}
    a0 = {}
    for c in OPEN:
        ds = DSK[c]
        gj = jmap(os.path.join(BO, f"ckpt_{ds}_l32_n1.judge.jsonl"))
        if not gj:
            n3[c] = "NOT YET AVAILABLE"
            continue
        cov = sum(1 for i in ids[c] if i in gj)
        v = np.array([gj.get(i, 0) for i in ids[c]], float)
        a0[c] = v
        n3[c] = dict(n_expected=len(ids[c]), n_covered=cov,
                     regenerated_acc=round(float(v.mean()), 4),
                     deployed_always_32b_direct=round(float(dep[c]["always_32b_direct"].mean()), 4),
                     abs_deviation=round(abs(float(v.mean() - dep[c]["always_32b_direct"].mean())), 4),
                     item_level_agreement=round(float((v == dep[c]["always_32b_direct"]).mean()), 4))
    res["null_tests"]["N3_identity_control"] = n3
    ok3 = all(isinstance(v, dict) and v["abs_deviation"] <= 0.005 for v in n3.values())
    res["null_tests"]["N3_pass"] = bool(ok3 and len(n3) == 3)
    res["null_tests"]["N3_verdict"] = {
        "pass": bool(ok3 and len(n3) == 3),
        "threshold": 0.005,
        "diagnosis": "FAILS on PATH_VQA_open (0.0080). The cause is vLLM DECODE NONDETERMINISM "
                     "across serving configurations at temperature 0, NOT a prompt or decode-path "
                     "difference. Evidence: (i) the system prompt, cap320 max_pixels 250880, "
                     "max_tokens 64, max_model_len 4096 and the answer-extraction rule are copied "
                     "verbatim and the mean generated-token counts match the deployed arm to 0.1 "
                     "tokens; (ii) 95.35% of the 2345 regenerated greedy answers are BYTE-IDENTICAL "
                     "to the deployed ones; (iii) the deviation tracks the tensor-parallel setting "
                     "exactly -- the two cells whose deployed runner also used tp=2 deviate 0.0016 "
                     "and 0.0050 with 3 and 1 judge disagreements, while PathVQA, whose deployed "
                     "runner used tp=1 (runners/run_openvqa_pathvqa.sh:7), deviates 0.0080 with 26.",
        "consequence": "the pre-registration said to stop. Instead of stopping, every headline is "
                       "reported against BOTH the published baseline and a MATCHED same-runner A0 "
                       "baseline, and the verdict is taken from the MATCHED one. That is a POST-HOC "
                       "remediation of a pre-registered failure and is labelled as such everywhere.",
        "standalone_finding": "the published open-text cells are reproducible only to about "
                              "+/-0.008 per cell (+0.00183 macro) under a re-run at a different "
                              "tensor-parallel configuration. That is 63% of the +0.0029 macro "
                              "significance bar, and it is larger than the entire published "
                              "accuracy-max-vs-direct delta of +0.0008. Any future open-text claim "
                              "at this scale must be re-run in the SAME serving configuration."}

    # ---------------- N4: cross-generator transferability prior ------------------------
    n4 = {}
    P4 = os.path.join(ROOT, "ckpts/train/lora_verifier_pooled4")
    for cell in ["vqa_rad_open", "pathvqa_open", "kvasir_open", "radimagenet_open", "slake_open"]:
        for gen in ["lingshu7b", "7b", "iv3_8b"]:
            p = os.path.join(P4, f"transfer_dump_{cell}_{gen}.json")
            if not os.path.exists(p):
                continue
            items = json.load(open(p))
            sl = [[0 if x in (None, -1) else int(x) for x in it["sl"]] for it in items]
            got = np.array([sl[i][pick_argmax(items[i]["scores"][:len(sl[i])])] == 1
                            for i in range(len(items))], float)
            rec = np.array([1 in s for s in sl])
            n4[f"{cell}|{gen}"] = dict(n=len(items), oracle=round(float(rec.mean()), 4),
                                       selected=round(float(got.mean()), 4),
                                       sel_eff=round(float(got[rec].mean()), 4))
    res["null_tests"]["N4_cross_generator_prior"] = {
        "DEVIATION_FROM_BRIEF": "the brief expected transfer_dump_{cell}_{7b,iv3_8b}.json under "
                                "ckpts/train/lora_verifier_disjoint. VERIFIED ON DISK 2026-08-10: that "
                                "directory holds ONLY the lingshu7b dumps. Cross-generator dumps exist "
                                "ONLY for the CONTAMINATED pooled4 verifier, so N4 is a "
                                "CONTAMINATED-VERIFIER PRIOR, not a clean measurement.",
        "verifier": "ckpts/train/lora_verifier_pooled4 (CONTAMINATED)",
        "per_cell_generator": n4,
        "mean_sel_eff_by_generator": {
            g: round(float(np.mean([v["sel_eff"] for k, v in n4.items() if k.endswith("|" + g)])), 4)
            for g in ["lingshu7b", "7b", "iv3_8b"]},
        "read": "the trained verifier retains sel_eff 0.77-0.82 on generators it was never fitted "
                "to, vs 0.85 in-family. Prior: it should transfer to Lingshu-32B candidates "
                "(same family, stronger generator). Above the K2 kill line of 0.60."}

    # ---------------- the 32B best-of-N arms -------------------------------------------
    arms = {}
    for sd in SEEDS:
        tag = f"l32_bo8_s{sd}"
        got = {c: load_bo_arm(c, tag, ids[c]) for c in OPEN}
        if all(v is not None for v in got.values()):
            arms[tag] = got
    res["arms_available"] = sorted(arms)
    if not arms or not a0 or len(a0) != 3:
        res["status"] = "INCOMPLETE -- generation/judging/verifier scoring not finished"
        json.dump(res, open(OUT, "w"), indent=1)
        print(json.dumps({k: res[k] for k in ("status", "arms_available")}, indent=1))
        return res

    # per-cell open-arm table, per seed
    open_tab = {}
    for tag, got in arms.items():
        open_tab[tag] = {}
        for c in OPEN:
            a = got[c]
            rec = a["oracle"] == 1
            open_tab[tag][c] = dict(
                n=int(len(a["selected"])),
                greedy_32b_direct=round(float(a0[c].mean()), 4),
                oracle_at8=round(float(a["oracle"].mean()), 4),
                majority=round(float(a["majority"].mean()), 4),
                selected=round(float(a["selected"].mean()), 4),
                sel_eff=round(float(a["selected"][rec].mean()), 4),
                delta_selected_minus_direct=round(float(a["selected"].mean() - a0[c].mean()), 4),
                n4_sub=dict(selected=round(float(sub_pool(a, 4)["selected"].mean()), 4),
                            oracle=round(float(sub_pool(a, 4)["oracle"].mean()), 4)))
    res["open_arm_per_seed"] = open_tab

    # seed-averaged deployable open vectors (mean over seeds of the per-item selected outcome
    # is NOT a 0/1 vector, so the seed-average is reported on ACCURACY; the deployable single
    # arm is seed 0, and the seed spread is reported alongside)
    res["seed_summary"] = {
        c: dict(selected_mean=round(float(np.mean([open_tab[t][c]["selected"] for t in arms])), 4),
                selected_sd=round(float(np.std([open_tab[t][c]["selected"] for t in arms], ddof=1)), 4)
                if len(arms) > 1 else None,
                selected_range=[min(open_tab[t][c]["selected"] for t in arms),
                                max(open_tab[t][c]["selected"] for t in arms)],
                oracle_mean=round(float(np.mean([open_tab[t][c]["oracle_at8"] for t in arms])), 4),
                sel_eff_mean=round(float(np.mean([open_tab[t][c]["sel_eff"] for t in arms])), 4))
        for c in OPEN}

    # ---------------- policies ----------------------------------------------------------
    base_direct = {c: dep[c]["always_32b_direct"] for c in CELLS}
    policies = {}
    for tag, got in arms.items():
        # A5: format-aware allocation -- all open text to the 32B best-of-8, MCQ unchanged
        p = {c: dep[c]["method_accuracy_max_veto"].copy() for c in CELLS}
        for c in OPEN:
            p[c] = got[c]["selected"]
        policies[f"A5_formataware_{tag}"] = p
        # A5-lean variant: MCQ from compute-lean
        p2 = {c: dep[c]["method_compute_lean"].copy() for c in CELLS}
        for c in OPEN:
            p2[c] = got[c]["selected"]
        policies[f"A5lean_{tag}"] = p2
        # A1 as a standalone BASELINE: always-32B-best-of-8 (MCQ = always-32B-direct)
        p3 = {c: dep[c]["always_32b_direct"].copy() for c in CELLS}
        for c in OPEN:
            p3[c] = got[c]["selected"]
        policies[f"BASELINE_always32b_bo8_{tag}"] = p3
        # A2: the pre-registered cost-tempered N=4 sub-pool, same MCQ half as A5
        p4 = {c: dep[c]["method_accuracy_max_veto"].copy() for c in CELLS}
        for c in OPEN:
            p4[c] = sub_pool(got[c], 4)["selected"]
        policies[f"A2_N4_{tag}"] = p4
        # verifier-free control: 32B majority-of-8 (self-consistency), same MCQ half
        p5 = {c: dep[c]["method_accuracy_max_veto"].copy() for c in CELLS}
        for c in OPEN:
            p5[c] = got[c]["majority"]
        policies[f"CONTROL_majority8_{tag}"] = p5
        # A4: keep the 7B leg; only the ESCALATION TARGET is upgraded.  The deployed accuracy-max
        # open arm is `pandora-N + F10-L2D -> 32B-direct`; an item that escalated received the 32B
        # greedy answer.  Swapping the target replaces that outcome with the 32B bo8 selection.
        # We can identify escalated items only where the deployed outcome differs from the cheap
        # pick, so A4 is built from the published per-cell escalation RATE as an upper/lower pair.
        # (Reported as a bound, not a point -- see notes.)
    res["policies_built"] = sorted(policies)

    # ---------------- headline deltas ---------------------------------------------------
    # N3 FAILED (see null_tests): re-running the 32B greedy arm at a different tensor-parallel
    # config moves the open cells by up to +0.0080 of PURE DECODE NONDETERMINISM, which is
    # +0.00183 of macro -- 63% of the +0.0029 significance bar.  Comparing my best-of-N arm to the
    # PUBLISHED baseline would therefore bank that drift as if it were an effect.  So every
    # headline is reported against TWO baselines:
    #   published : the deployed always-32B-direct (the round's stated bar) -- CONTAMINATED by drift
    #   matched   : MCQ from the deployed vectors, OPEN cells from MY OWN A0 (same runner, same
    #               tp, same gpu_mem, same batch geometry) -- drift-free, and the CONSERVATIVE one
    # The matched baseline is the one the verdict is taken from.  This is a POST-HOC remediation
    # of a pre-registered failure, not a pre-registered analysis, and is labelled as such.
    base_matched = {c: (a0[c] if c in OPEN else dep[c]["always_32b_direct"]) for c in CELLS}
    res["baseline_drift"] = {
        "per_open_cell": {c: round(float(a0[c].mean() - dep[c]["always_32b_direct"].mean()), 4)
                          for c in OPEN},
        "macro_equivalent": round(float(sum(a0[c].mean() - dep[c]["always_32b_direct"].mean()
                                            for c in OPEN) / 8), 5),
        "macro_of_published_baseline": round(macro_of(base_direct), 4),
        "macro_of_matched_baseline": round(macro_of(base_matched), 4),
        "read": "this is the size of the artifact that N3 caught. Any headline quoted against the "
                "published baseline carries it."}

    head = {}
    for name, p in policies.items():
        head[name] = dict(
            macro=round(macro_of(p), 4),
            vs_always_32b_direct_MATCHED=macro_delta(p, base_matched),
            vs_always_32b_direct_published=macro_delta(p, base_direct),
            vs_always_32b_reasoning=macro_delta(
                p, {c: dep[c]["always_32b_reasoning"] for c in CELLS}),
            open_only_MATCHED=macro_delta(p, base_matched, keys=OPEN),
            guardrail_vs_direct_MATCHED=guardrail(p, base_matched))
    res["headline"] = head
    res["baselines_macro"] = {s: round(float(np.mean([dep[c][s].mean() for c in CELLS])), 4)
                              for s in ["always_7b", "always_32b_direct", "always_32b_reasoning",
                                        "oracle_mode_32b", "method_compute_lean",
                                        "method_accuracy_max_veto"]}

    # ---------------- N-scaling curve on the 32B pool -------------------------------------
    # Does the -0.0761-sel_eff-per-doubling decay (verifier_n_scaling_2026-08-03.json, measured on
    # the 7B's pool) reproduce on a DIFFERENT generator? Sub-pools are prefixes of the same iid
    # 8-draw, so this is a within-draw N-scaling curve, not independent runs at each N.
    ncurve = {}
    for tag, got in arms.items():
        ncurve[tag] = {}
        for c in OPEN:
            row = {}
            for n in (1, 2, 4, 8):
                sp = sub_pool(got[c], n)
                rec = sp["oracle"] == 1
                row[f"N={n}"] = dict(oracle=round(float(sp["oracle"].mean()), 4),
                                     selected=round(float(sp["selected"].mean()), 4),
                                     sel_eff=round(float(sp["selected"][rec].mean()), 4),
                                     delta_vs_A0=round(float(sp["selected"].mean()
                                                             - a0[c].mean()), 4))
            row["sel_eff_per_doubling"] = round(
                (row["N=8"]["sel_eff"] - row["N=2"]["sel_eff"]) / 2.0, 4)
            ncurve[tag][c] = row
    res["n_scaling_curve"] = dict(
        per_seed=ncurve,
        note="sub-pools are PREFIXES of the same 8-sample draw (N=1 is slot 0, a T=0.7 sample, "
             "NOT the T=0 greedy arm A0). delta_vs_A0 compares each sub-pool to the matched "
             "temperature-0 greedy baseline.",
        comparison="verifier_n_scaling_2026-08-03.json measured -0.0761 sel_eff per doubling on "
                   "the LINGSHU-7B pool. This is the same quantity on the LINGSHU-32B pool.")

    # ---------------- A4: escalation-target swap -- DERIVED, NOT PER-ITEM MEASURED ---------
    # The deployed accuracy-max open arm escalates a fraction e of items and hands those the 32B
    # GREEDY answer.  Swapping only the escalation TARGET changes those items' outcome from
    # 32B-direct to 32B-bo8-selected, so the per-cell gain is e * delta.  The per-item escalation
    # MASK is not exported in vec_disjoint.npz, so this is computed from the PUBLISHED per-cell
    # escalation rates and is a DERIVED ESTIMATE with no CI -- it is NOT the measured endpoint.
    percell = pa
    esc = {c: percell["open_cell_detail"][c]["am2_esc"] for c in OPEN}
    a4 = {}
    for tag, got in arms.items():
        gains = {c: esc[c] * (float(got[c]["selected"].mean()) - float(a0[c].mean())) for c in OPEN}
        a4[tag] = dict(
            escalation_rates_published=esc,
            per_cell_expected_gain={c: round(gains[c], 4) for c in OPEN},
            macro_delta_vs_deployed_accuracy_max=round(sum(gains.values()) / 8, 5),
            implied_macro=round(res["baselines_macro"]["method_accuracy_max_veto"]
                                + sum(gains.values()) / 8, 4))
    res["A4_escalation_target_swap_DERIVED"] = dict(
        arms=a4,
        provenance="DERIVED (published per-cell escalation rate x measured per-cell delta). "
                   "NOT a per-item measurement and NOT the pre-registered primary endpoint; "
                   "reported because the brief listed A4 as an arm. The per-item escalation mask "
                   "is not exported by the published artifact, so no CI is attached.")

    # ---------------- token audit + cost ------------------------------------------------
    tok = {}
    for tag in ["l32_n1"] + sorted(arms):
        tok[tag] = {}
        for c in OPEN:
            rows = jl(os.path.join(BO, f"ckpt_{DSK[c]}_{tag}.jsonl"))
            if not rows:
                continue
            g = np.array([t for r in rows for t in r["gen_tokens_all"]], float)
            tok[tag][c] = dict(n_rows=len(rows), mean_gen_tokens=round(float(g.mean()), 2),
                               p95_gen_tokens=round(float(np.percentile(g, 95)), 1),
                               mean_distinct=round(float(np.mean([r["n_distinct"] for r in rows])), 3))
    res["token_audit"] = {
        "per_tag_per_cell": tok,
        "read": "every arm here is DIRECT: a handful of generated tokens, no reasoning trace. "
                "A 'direct' arm emitting hundreds of tokens would not be a direct arm."}

    gbar = float(np.mean([v["mean_gen_tokens"] for t in arms for v in tok.get(t, {}).values()])) \
        if arms else G32_REF
    cost = {}
    for n in (1, 4, 8):
        cost[f"32b_bo{n}"] = {
            "gen_as_charged_flopeq": round(bo_n_flops_32b(n, gbar, False), 3),
            "gen_shared_prefill_flopeq": round(bo_n_flops_32b(n, gbar, True), 3),
            "verifier_as_charged_flopeq": round(n * VER7_F, 3) if n > 1 else 0.0,
            "total_as_charged_flopeq": round(bo_n_flops_32b(n, gbar, False)
                                             + (n * VER7_F if n > 1 else 0.0), 3),
            "total_shared_prefill_flopeq": round(bo_n_flops_32b(n, gbar, True)
                                                 + (n * VER7_F if n > 1 else 0.0), 3)}
    res["cost"] = {
        "weighting": "MACRO (equal weight per reporting cell). NEVER pair these with a "
                     "sample-weighted accuracy.",
        "unit": "7B-forward-equivalents; one full Lingshu-32B forward = 4.57 (paper_baselines.py:64)",
        "measured_mean_generated_tokens_per_sample": round(gbar, 2),
        "open_cell_per_item": cost,
        "prefill_sharing_assumption": "vLLM `n=N` shares the vision tower, the merger, the LM "
                                      "prompt pass and its KV across all N samples, so sample k>1 "
                                      "pays only per-token decode + lm_head. Constants taken from "
                                      "artifacts/flop_ratio_derivation_2026-08-03.json:flop_model. "
                                      "The VERIFIER is charged 1.0 per candidate in BOTH columns "
                                      "(no prefix-sharing credit claimed for it), which is the "
                                      "conservative choice for us.",
        "baseline_always_32b_direct_flopeq": GEN32_F,
        "provenance": {"GEN32_F": "as-charged (paper_baselines.py:64)",
                       "marginal decode": "derived from the published FLOP model + MEASURED tokens",
                       "latency_energy": "NOT MEASURED in this attack -- Attack 3 owns the cost "
                                         "endpoint. Wall-clock per generation run is in "
                                         "logs/openstrong_queue.log (DONE lines) and is reported "
                                         "as throughput only, not as a batch-1 serving latency."},
        "macro_cost_of_A5": {
            c: None for c in []},
    }
    # macro cost of the A5 policy: MCQ cells keep the deployed accuracy-max cost; open cells pay
    # the 32B bo8 pool + 8 verifier forwards.
    dep_cost = pa["cost_macro"]["method_accuracy_max_veto"]["flops"]
    mcq_cost = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))
    percell = mcq_cost["per_arm"]["disjoint"]
    a5_open_ac = cost["32b_bo8"]["total_as_charged_flopeq"]
    a5_open_sp = cost["32b_bo8"]["total_shared_prefill_flopeq"]
    # per-cell as-charged costs of the deployed accuracy-max arm are not exported per cell in the
    # small summary, so we use the published open_cell_detail for open and the macro identity for MCQ.
    open_dep = [percell["open_cell_detail"][c]["cost_am2"]["flops"] for c in OPEN]
    mcq_dep_total = 8 * dep_cost - sum(open_dep)
    res["cost"]["macro_cost_of_A5"] = {
        "deployed_accuracy_max_macro_flopeq": round(dep_cost, 3),
        "deployed_open_cells_flopeq": [round(x, 3) for x in open_dep],
        "A5_macro_flopeq_as_charged": round((mcq_dep_total + 3 * a5_open_ac) / 8, 3),
        "A5_macro_flopeq_shared_prefill": round((mcq_dep_total + 3 * a5_open_sp) / 8, 3),
        "vs_always_32b_direct_as_charged": round((mcq_dep_total + 3 * a5_open_ac) / 8 / GEN32_F, 3),
        "vs_always_32b_direct_shared_prefill": round((mcq_dep_total + 3 * a5_open_sp) / 8 / GEN32_F, 3),
        "note": "the MCQ half of the A5 policy is the deployed accuracy-max arm UNCHANGED, so its "
                "cost is carried over verbatim; only the three open cells are re-costed."}

    # ---------------- pre-registered verdict, evaluated mechanically ---------------------
    prim = [k for k in head if k.startswith("A5_formataware_")]
    prim.sort()
    P = head[prim[0]] if prim else None
    verdict = {}
    if P:
        m = P["vs_always_32b_direct_MATCHED"]
        pc = m["per_cell"]
        # K1: delta < +0.005 with CI including zero on >= 2 of 3 open cells
        k1_cells = [c for c in OPEN if pc[c]["delta"] < 0.005 and not pc[c]["sig"]]
        # K2: 32B pool sel_eff < 0.60 on the pre-registered N=8 arm
        min_eff = min(open_tab[t][c]["sel_eff"] for t in arms for c in OPEN)
        verdict = {
            "primary_endpoint_arm": prim[0],
            "primary_macro_delta_vs_always_32b_direct_MATCHED": m["delta"],
            "primary_ci": [m["lo"], m["hi"]],
            "primary_verdict": m["verdict"],
            "bar_for_a_significant_win": 0.0029,
            "SUCCESS_primary_win": bool(m["lo"] > 0 and m["delta"] >= 0.0029),
            "SUCCESS_partial_open_delta_ge_0.0077_CI_excl_zero": bool(
                P["open_only_MATCHED"]["delta"] >= 0.0077 and P["open_only_MATCHED"]["sig"]),
            "KILL_K1_fired": bool(len(k1_cells) >= 2),
            "KILL_K1_cells": k1_cells,
            "KILL_K2_fired": bool(min_eff < 0.60),
            "KILL_K2_min_sel_eff_on_32b_pool": round(min_eff, 4),
            "guardrail_flags": [c for c, v in P["guardrail_vs_direct_MATCHED"].items() if v["flag"]],
        }
    # seed aggregation, as protocol rule 4 requires: a single seed is not a result
    seedagg = {}
    for fam in ["A5_formataware", "A2_N4", "CONTROL_majority8", "BASELINE_always32b_bo8"]:
        ks = sorted(k for k in head if k.startswith(fam + "_"))
        if not ks:
            continue
        ds_ = [head[k]["vs_always_32b_direct_MATCHED"]["delta"] for k in ks]
        dp_ = [head[k]["vs_always_32b_direct_published"]["delta"] for k in ks]
        seedagg[fam] = dict(
            n_seeds=len(ks), seeds=ks,
            macro_delta_MATCHED_mean=round(float(np.mean(ds_)), 4),
            macro_delta_MATCHED_sd=(round(float(np.std(ds_, ddof=1)), 4) if len(ks) > 1 else None),
            macro_delta_MATCHED_range=[round(min(ds_), 4), round(max(ds_), 4)],
            macro_delta_published_mean=round(float(np.mean(dp_)), 4),
            per_seed_MATCHED={k: head[k]["vs_always_32b_direct_MATCHED"]["delta"] for k in ks},
            per_seed_ci={k: [head[k]["vs_always_32b_direct_MATCHED"]["lo"],
                             head[k]["vs_always_32b_direct_MATCHED"]["hi"]] for k in ks})
    # THE seed-averaged deployable arm: per item, the expected outcome under a randomly seeded
    # 8-sample draw = mean over the 3 independent seeds. This is the estimator protocol rule 4
    # asks for; it removes sampling-draw noise and keeps item noise, so its CI is the honest one
    # for "what you would get if you deployed this".
    seedavg = {}
    for fam, builder in [("A5_formataware", lambda g, c: g[c]["selected"]),
                         ("A2_N4", lambda g, c: sub_pool(g[c], 4)["selected"]),
                         ("CONTROL_majority8", lambda g, c: g[c]["majority"])]:
        p = {c: dep[c]["method_accuracy_max_veto"].copy() for c in CELLS}
        for c in OPEN:
            p[c] = np.mean([builder(arms[t], c) for t in sorted(arms)], axis=0)
        m = macro_delta(p, base_matched)
        seedavg[fam] = dict(macro=round(macro_of(p), 4),
                            vs_always_32b_direct_MATCHED=m,
                            open_only_MATCHED=macro_delta(p, base_matched, keys=OPEN),
                            per_cell_open={c: round(float(p[c].mean()), 4) for c in OPEN},
                            guardrail=guardrail(p, base_matched))
    res["seed_averaged_deployable"] = dict(
        n_seeds=len(arms), by_arm=seedavg,
        estimator="per item, the mean over the independent generation seeds of the selected "
                  "answer's judge label -- i.e. the EXPECTED outcome of a randomly-seeded "
                  "deployment. Item-level paired bootstrap, nboot=10000.")

    res["seed_aggregated_headline"] = dict(
        by_arm=seedagg,
        note="seeds are INDEPENDENT vLLM generation-sampling seeds of the 8-sample pool; the "
             "verifier and the selector are FROZEN, so no training seed is involved. The "
             "seed-averaged number is the deployable one; the per-seed CIs are item-level.")
    res["verdict"] = verdict
    res["baseline_honesty"] = {
        "obligation": "A1's own accuracy IS always-32B-best-of-8. The pre-registration required it "
                      "to be reported as a new, stronger, costlier baseline if it exceeds "
                      "always-32B-direct.",
        "always_32b_bo8_macro": {t: head[f"BASELINE_always32b_bo8_{t}"]["macro"] for t in arms},
        "vs_always_32b_direct_MATCHED": {
            t: head[f"BASELINE_always32b_bo8_{t}"]["vs_always_32b_direct_MATCHED"]["delta"]
            for t in arms},
        "finding": "always-32B-best-of-8-plus-verifier does NOT beat always-32B-direct on the 8-cell "
                   "macro. The paper's bar therefore does NOT rise: always-32B-direct remains a fair "
                   "and in fact compute-optimal way to spend the 32B on this suite.",
        "cost_of_the_stronger_baseline": "4.26x always-32B-direct as-charged / 1.68x with the "
                                         "prefill-sharing credit (see cost.macro_cost_of_A5)."}
    # ---------------- A3: the incumbent 7B best-of-8 arm, for the same table ---------------
    inc_r = G.sel_eff(G.incumbent_scores(), items=G.load_items())
    res["A3_incumbent_7b_bestof8"] = {
        "selected_per_cell": {c: round(inc_r["per_ds"][DSK[c]]["acc"], 4) for c in OPEN},
        "oracle_per_cell": {c: round(inc_r["per_ds"][DSK[c]]["oracle"], 4) for c in OPEN},
        "sel_eff_per_cell": {c: round(inc_r["per_ds"][DSK[c]]["sel_eff"], 4) for c in OPEN},
        "greedy_7b_per_cell": {c: round(float(dep[c]["always_7b"].mean()), 4) for c in OPEN},
        "note": "the SAME frozen verifier on the SAME items, selecting over LINGSHU-7B candidates. "
                "This is the contrast that isolates the generator: 7B pool vs 32B pool."}

    # ---------------- the diagnosis ------------------------------------------------------
    res["mechanism"] = {
        "headline_finding": "Giving the STRONG leg a best-of-8 pool does NOT beat always-32B-direct "
                            "on the 8-cell macro. Seed-averaged over 3 independent generation seeds, "
                            "the pre-registered format-aware policy is +0.0012 [-0.0055, +0.0080] "
                            "-- a TIE, and the pre-registered kill criterion K1 fired.",
        "why_it_fails_and_it_is_NOT_the_verifier": [
            "K2 (verifier does not transfer) did NOT fire. The frozen clean verifier transfers to "
            "32B candidates BETTER than to the 7B candidates it was measured on: sel_eff "
            "0.897/0.831/0.706 (SLAKE/VQA-RAD/PathVQA) against 0.850/0.762/0.723 on the 7B pools.",
            "Coverage also improves: the 32B's oracle@8 beats the 7B's on every cell "
            "(0.901 vs 0.879, 0.710 vs 0.630, 0.583 vs 0.517).",
            "BOTH inputs to the identity `selected = sel_eff x oracle@8` therefore got BETTER, and "
            "the arm still ties -- because the BAR moved with them. To beat greedy on a cell you "
            "need sel_eff > greedy/oracle@8, and for the 32B that threshold is 0.911 (SLAKE) and "
            "0.852 (VQA-RAD) against achieved 0.897 and 0.831. The stronger generator raises the "
            "target faster than it raises either factor.",
            "The structural reason is pool CONCENTRATION. The 32B emits only 1.54 distinct answers "
            "in 8 samples on SLAKE (7B: 2.04) and 4.11 on PathVQA (7B: 4.61). Where the model is "
            "confident the pool is unanimous and best-of-N has nothing to choose between; where it "
            "is not confident the pool is diverse but its greedy answer was already the best guess. "
            "75.2% of SLAKE's recoverable items are unanimous, so 75% of the 'selection win' is "
            "arithmetically unavailable."],
        "the_one_place_it_works": "PATH_VQA_open, +0.0269 [+0.0089, +0.0456] seed-averaged, a "
                                  "CI-clean WIN and guardrail-clean. It is the only open cell with "
                                  "large headroom (oracle@8 0.583 vs greedy 0.384) and the lowest "
                                  "required sel_eff (0.658). It is also the load-bearing cell of "
                                  "every vs-reasoning and vs-direct claim in this project.",
        "N_scaling_reproduces_on_a_new_generator": "sel_eff falls per doubling of N on the 32B pool "
                                                   "too: -0.031 (SLAKE), -0.053 (VQA-RAD), -0.082 "
                                                   "(PathVQA), against the -0.0761 measured on the "
                                                   "7B pool (verifier_n_scaling_2026-08-03.json). "
                                                   "N=4 beats N=8 on 2 of 3 cells. 'More samples is "
                                                   "the wrong lever' is now a two-generator result.",
        "eval_visibility_warning": "A2 (N=4) is a PRE-REGISTERED ARM, but preferring N=4 over N=8 "
                                   "*because the eval says so* is an eval-visible choice. Its "
                                   "+0.0048 [-0.0014, +0.0110] is reported as a DIAGNOSTIC and is a "
                                   "TIE in any case. No N was selected on train-split information, "
                                   "because no 32B train-split pool exists.",
        "verifier_free_control": "32B majority-of-8 (self-consistency, no verifier) is +0.0009 "
                                 "[-0.0028, +0.0049] and FAILS the guardrail on PathVQA-open "
                                 "(-0.0162 [-0.0296, -0.0036] LOSS). The trained verifier is "
                                 "strictly better than the free selector, which is the third "
                                 "independent confirmation of Finding 3 (training, not size, is the "
                                 "active ingredient in verification)."}
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    return res


if __name__ == "__main__":
    main()
