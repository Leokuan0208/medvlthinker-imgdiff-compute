#!/usr/bin/env python3
"""vrestruct_lib.py -- shared loader + COST MODEL for the 2026-08-16 verifier-restructuring round.

THE OBJECTIVE CHANGED.  The baseline is now ALWAYS-7B (macro 0.5971, 1.0 FLOP-eq per question),
not always-32B-direct.  The claim shape is "a small verifier improves a 7B medical VLM by +X at Y x
the 7B's own compute", so Y is an endpoint, not a footnote.  This module supplies:

  1. the frozen selection pool + the three selector structures (LoRA-only / head-only / fused),
  2. BOTH accuracy currencies (32B judge and normalised exact match) on IDENTICAL picks,
  3. a per-pass FLOP model that is honest about RESOLUTION -- which the project's 1.0-per-pass
     charge is not.

THE RESOLUTION LANDMINE (measured, not assumed).  The generator runs at cap320 (max_pixels
250,880).  The incumbent LoRA verifier runs at max_pixels 1,003,520 and so does the teacher-forced
pass that reads the generator-frame head's layer-21 state (feats_hidden/*.meta.json:max_pixels).
A pass at 1,003,520 is NOT one FLOP-eq: measured geometry gives 12.729 TFLOP against the cap320
generator's 5.693 TFLOP, i.e. 2.236 FLOP-eq per verification pass.  Everything downstream of this
module therefore charges verification passes at their MEASURED resolution.

Sources, all verbatim:
  pool + metric   src/training_methods/genframe_data.py           (null test: 3.60e-07)
  selector        ckpts/train/genframe_head_ens8/                 (frozen 8-seed head + fusion)
  EM currency     artifacts/_verifier_hparams_parts/em_slots.npz  (built 2026-08-16, re-verified here)
  FLOP model      src/cascade_methods/flop_ratio_derivation.forward_flops (safetensors headers)
  geometry        artifacts/resolution_sweep_2026-08-13.json  cost.open_half_per_candidate
  shared prefill  artifacts/cost_floor_2026-08-10.json  null_tests.N2  (prefill/decode shares)

Nothing here touches a GPU or writes a file.  Import is side-effect free.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
for p in (ROOT, os.path.join(ROOT, "src/training_methods"), os.path.join(ROOT, "src/cascade_methods")):
    if p not in sys.path:
        sys.path.insert(0, p)

import genframe_data as G          # noqa: E402
import genframe_selector as GS     # noqa: E402

ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_vrestruct_parts")
EM_NPZ = os.path.join(ART, "_verifier_hparams_parts/em_slots.npz")
RESSWEEP = os.path.join(ART, "resolution_sweep_2026-08-13.json")
COSTFLOOR = os.path.join(ART, "cost_floor_2026-08-10.json")
HEADDIR = os.path.join(ROOT, "ckpts/train/genframe_head_ens8")

BOOT_SEED = 20260816
NBOOT = 10000


# ======================================================================================
# 1.  the pool, both currencies
# ======================================================================================
def load_pool(layer=21, pooling="span"):
    """items, judge(n,8), em(n,8), incumbent(n,8), head logits per seed, slot->row map."""
    items = G.load_items()
    n = len(items)
    judge = np.array([it["sl"] for it in items], dtype=int)
    inc = np.array([it["scores"] for it in items], dtype=float)
    z = np.load(EM_NPZ)
    em, jz = z["em"], z["judge"]
    if em.shape != judge.shape or not np.array_equal(jz, judge):
        raise ValueError("em_slots.npz judge column does not match the frozen transfer dumps")
    ds_index = np.array([G.EVAL_DS.index(it["ds"]) for it in items], dtype=int)
    greedy_ok = np.array([it["greedy_ok"] for it in items], dtype=int)

    ev = G.load_candidates("eval", "generator", layers=[layer], pooling=(pooling,), order="concat")
    slot_rows = np.array([q.slot_rows for q in ev.questions], dtype=int)   # (n, 8)
    if slot_rows.shape != (n, 8):
        raise ValueError(f"slot_rows {slot_rows.shape}")
    X = ev.matrix(pooling, layer)                                          # (n_rows, 3584)
    return dict(items=items, n=n, judge=judge, em=em, inc=inc, ds_index=ds_index,
                greedy_ok=greedy_ok, slot_rows=slot_rows, X=X, ev=ev)


_SEL = None


def selector():
    global _SEL
    if _SEL is None:
        _SEL = GS.FrozenSelector.load(HEADDIR)
    return _SEL


def head_logits(P):
    """(8 seeds, n_rows) raw logits of the FROZEN heads on the loaded feature matrix."""
    S = selector()
    return S.head_logits(P["X"])          # standardises with the frozen train mu/sd


def head_rank_slots(P, L, seeds):
    """(n, 8) the mean within-pool rank_avg of the chosen seeds' logits, per slot."""
    n = P["n"]
    sr = P["slot_rows"]
    out = np.empty((n, 8), float)
    sub = L[list(seeds)]
    for i in range(n):
        rows = sr[i]
        out[i] = np.mean([G.rank_avg(sub[s][rows]) for s in range(sub.shape[0])], axis=0)
    return out


def rank_rows(M):
    """rank_avg applied row-wise to an (n, 8) matrix."""
    return np.stack([G.rank_avg(M[i]) for i in range(M.shape[0])])


def picks_of(S):
    """argmax over slots, FIRST-INDEX tie-break -- the frozen pick rule."""
    return np.argmax(np.asarray(S, float), axis=1).astype(int)


# ======================================================================================
# 2.  the metric, both currencies, on IDENTICAL picks
# ======================================================================================
def evaluate(P, picks, label=""):
    """sel_eff + selected accuracy in BOTH currencies + per-cell guardrail rows."""
    n = P["n"]
    rows = np.arange(n)
    out = {"label": label, "n": n, "picks": picks}
    for cur, M in (("judge", P["judge"]), ("em", P["em"])):
        got = M[rows, picks].astype(int)
        rec = (M.max(1) == 1).astype(int)
        per = {}
        for j, ds in enumerate(G.EVAL_DS):
            m = P["ds_index"] == j
            per[ds] = dict(n=int(m.sum()), n_recoverable=int(rec[m].sum()),
                           oracle=float(rec[m].mean()), acc=float(got[m].mean()),
                           sel_eff=float(got[m & (rec == 1)].mean()))
        out[cur] = dict(acc=float(got.mean()), oracle=float(rec.mean()),
                        sel_eff=float(got[rec == 1].mean()),
                        macro_cells=float(np.mean([per[d]["acc"] for d in G.EVAL_DS])),
                        per_ds=per, got=got, rec=rec)
    # identity check, judge currency: selected == oracle@8 * sel_eff
    out["identity_dev_judge"] = abs(out["judge"]["acc"] - out["judge"]["oracle"] * out["judge"]["sel_eff"])
    return out


def paired_boot(a, b, mask=None, nboot=NBOOT, seed=BOOT_SEED):
    """Paired ITEM bootstrap of mean(a)-mean(b); mask restricts to a stratum (e.g. recoverable)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        a, b = a[m], b[m]
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()), lo=float(lo), hi=float(hi),
                p_two_sided=float(2 * min((bs <= 0).mean(), (bs >= 0).mean())),
                significant=bool(lo > 0 or hi < 0), n=int(n), nboot=int(nboot))


def perm_null(P, score_fn, nperm=1000, seed=BOOT_SEED, currency="judge"):
    """Permutation null: shuffle the SLOT ORDER within each question, recompute the endpoint.

    Destroys any real association between the score vector and which slot is correct while
    preserving the pool composition exactly.
    """
    rng = np.random.default_rng(seed)
    n = P["n"]
    M = P[currency]
    rec = (M.max(1) == 1)
    vals = []
    S = score_fn()
    for _ in range(nperm):
        perm = np.argsort(rng.random((n, 8)), axis=1)
        Sp = np.take_along_axis(S, perm, axis=1)
        pk = picks_of(Sp)
        got = M[np.arange(n), np.take_along_axis(perm, pk[:, None], 1).ravel()]
        vals.append(float(got[rec].mean()))
    return np.array(vals)


# ======================================================================================
# 3.  THE COST MODEL -- per-pass FLOPs at the pass's OWN resolution
# ======================================================================================
_PC = None


def _param_counts():
    global _PC
    if _PC is None:
        import flop_ratio_derivation as F
        _PC = F.param_counts("Lingshu-7B")
    return _PC


def fwd_tflops(M_img_tok, T_prompt_tok, G_gen_tok):
    """One Lingshu-7B forward+generate, in TFLOP, at the given token geometry."""
    import flop_ratio_derivation as F
    return F.forward_flops(_param_counts(), M_img_tok, T_prompt_tok, G_gen_tok)["TOTAL"] / 1e12


VHP = os.path.join(ART, "_verifier_hparams_parts")


def cost_constants():
    """Every per-pass constant this round charges, each with its measured geometry [MEASURED].

    The verifier's per-pass cost is taken from the CONCURRENT resolution round, which measured the
    prompt geometry on all 8,965 scored triples per rung (verifier_hparams_score.py) -- a far
    larger sample than resolution_sweep_2026-08-13's n=120 verifier probe.  Both are reported.
    """
    rs = json.load(open(RESSWEEP))["cost"]
    cap320 = rs["open_half_per_candidate"]["cap320"]
    unit = cap320["flops_per_candidate"] / 1e12                    # 1 FLOP-eq, by definition

    cf = json.load(open(COSTFLOOR))
    n2 = cf["null_tests"]["N2"]
    vg = cf["verifier_geometry"]

    # ---- verifier ladder, measured on 8,965 forwards per rung [MEASURED] --------------------
    ver_by_px, ver_geo_by_px = {}, {}
    p = os.path.join(VHP, "recost.json")
    if os.path.exists(p):
        rc = json.load(open(p))["by_max_pixels"]
        cst = json.load(open(os.path.join(VHP, "cost.json")))["by_max_pixels"]
        for px, r in rc.items():
            ver_by_px[int(px)] = float(r["verifier_forward_in_generator_equivalents"])
            ver_geo_by_px[int(px)] = dict(
                vision=cst[px]["measured_mean_vision_tokens"],
                prompt=cst[px]["measured_mean_prompt_tokens"],
                tflop=cst[px]["flops_per_verifier_forward"] / 1e12,
                n_forwards_measured=cst[px]["n_forwards_measured"])
    ver_dep = ver_by_px.get(1003520)
    ver_640 = ver_by_px.get(501760)
    ver_320 = ver_by_px.get(250880)

    # secondary estimate of the same quantity, from the n=120 probe
    ver_tflop_probe = cap320["flops_verifier_per_candidate_at_1003520"] / 1e12

    # ---- the head's teacher-forced pass ------------------------------------------------------
    # GENERATOR prompt, but rendered at 1,003,520 like the feature cache
    # (feats_hidden/generator_eval_s0of2.meta.json: max_pixels 1003520), G=1 (teacher forced).
    gen_text_tok = cap320["measured_mean_prompt_tokens"] - cap320["measured_mean_vision_tokens"]
    vis_dep = ver_geo_by_px.get(1003520, {}).get("vision", 520.5)
    head_prompt = vis_dep + gen_text_tok + vg["candidate_answer_tok_mean"]
    head_tflop = fwd_tflops(vis_dep, head_prompt, 1.0)
    head_tflop_cap320 = fwd_tflops(cap320["measured_mean_vision_tokens"],
                                   cap320["measured_mean_prompt_tokens"]
                                   + vg["candidate_answer_tok_mean"], 1.0)

    return dict(
        unit_tflop=unit,
        unit_definition="1.0 FLOP-eq = one Lingshu-7B cap320 open-text forward+generate = "
                        f"{unit:.4f} TFLOP (resolution_sweep_2026-08-13.json "
                        "cost.open_half_per_candidate.cap320.flops_per_candidate)",
        gen_cap320_flopeq=1.0,
        gen_geometry=dict(vision=cap320["measured_mean_vision_tokens"],
                          prompt=cap320["measured_mean_prompt_tokens"],
                          gen=cap320["measured_mean_gen_tokens"]),
        ver_flopeq_by_max_pixels=ver_by_px,
        ver_geometry_by_max_pixels=ver_geo_by_px,
        ver_1003520_flopeq=ver_dep, ver_501760_flopeq=ver_640, ver_250880_flopeq=ver_320,
        ver_1003520_flopeq_secondary_n120_probe=ver_tflop_probe / unit,
        head_1003520_tflop=head_tflop, head_1003520_flopeq=head_tflop / unit,
        head_cap320_tflop=head_tflop_cap320, head_cap320_flopeq=head_tflop_cap320 / unit,
        prefill_share_7b=n2["prefill_share_7b"], decode_share_7b=n2["decode_share_7b"],
        ver_prefix_cost_units=vg["ver_prefix_cost_units"],
        ver_marginal_cost_per_candidate_units=vg["ver_marginal_cost_per_candidate_units"],
        verifier_geometry_tokens=vg,
        R32_as_charged=4.57, R32_derived=3.816,
        RESOLUTION_MISMATCH_WARNING=(
            "The generator runs at max_pixels 250,880 (cap320) but BOTH scorers run at 1,003,520: "
            "the LoRA verifier by its runner's default and the generator-frame head because "
            "feats_hidden/*.meta.json was extracted at 1,003,520. The 'generator-frame' head "
            "therefore does NOT run in the generator's frame. Capturing the layer-21 state DURING "
            "generation would read it at 250,880 -- a resolution the frozen heads have never seen. "
            "That transfer must be measured before the free-head saving can be claimed."),
        provenance=dict(
            unit="artifacts/resolution_sweep_2026-08-13.json",
            verifier_pass="artifacts/_verifier_hparams_parts/{recost,cost}.json by_max_pixels "
                          "-- MEASURED on 8,965 verifier forwards per rung (concurrent round "
                          "verifier_hparams_2026-08-15)",
            verifier_pass_secondary="artifacts/resolution_sweep_2026-08-13.json n=120 probe",
            head_pass="computed with flop_ratio_derivation.forward_flops on the feature cache's OWN "
                      "geometry: feats_hidden/generator_eval_s0of2.meta.json max_pixels=1003520, "
                      "vision tokens from the concurrent round's 1,003,520 rung, generator text "
                      "prompt + candidate answer, G=1 (teacher forced)",
            shares="artifacts/cost_floor_2026-08-10.json null_tests.N2",
            verifier_prefix="artifacts/cost_floor_2026-08-10.json verifier_geometry"))


def pass_counts(P):
    """The MEASURED per-question pass counts of the current structure."""
    items = P["items"]
    nd = np.array([len(set(G.norm(a) for a in it["preds"])) for it in items], dtype=float)
    return dict(mean_distinct_answers=float(nd.mean()),
                mean_pool_size=8.0,
                per_question_distinct=nd)


def G_of_N(N, c):
    """Shared-prefill cost of N sampled generations, in FLOP-eq (convention B)."""
    return c["prefill_share_7b"] + N * c["decode_share_7b"]
