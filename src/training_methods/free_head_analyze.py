#!/usr/bin/env python3
"""free_head_analyze.py -- BUILD 1, part D (CPU only): the equivalence test and the endpoint.

Consumes free_head_capture.py's outputs and answers, for the frozen 8-seed generator-frame head:

  1. NULL TESTS.  genframe_data.null_test (the incumbent's published cells), genframe_selector.verify
     (the frozen head reloaded from disk), and -- new here -- does THIS capture harness's
     teacher-forced pass at FULLRES reproduce the frozen feature cache feats_hidden/ that the head
     was trained and evaluated on?  Max abs deviation is reported per array.

  2. EQUIVALENCE.  For each captured feature source, against the DEPLOYED features:
       max abs deviation, relative L2, cosine, per-row;
       head logit deviation; head_rank deviation;
       ARGMAX PICKS CHANGED -- the question that actually matters;
       sel_eff / SELECTED in BOTH currencies, per set, with the guardrail and a paired bootstrap.

  3. Feature sources.
       deployed          feats_hidden, FULLRES, teacher-forced   (what ships today, 3.813646 passes)
       fullres_tf        this harness, FULLRES, teacher-forced   (harness null test)
       fullres_ar        this harness, FULLRES, captured during generation
       cap320_tf         this harness, cap320,  teacher-forced   (isolates the resolution shift)
       cap320_ar         this harness, cap320,  captured during generation  <-- THE FREE HEAD, since
                         the deployed pools were GENERATED at cap320
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G                      # noqa: E402
import genframe_selector as GS                 # noqa: E402
import free_head_lib as F                      # noqa: E402
import free_head_cost as C                     # noqa: E402
from genframe_selector import FrozenSelector   # noqa: E402

OUTDIR = os.path.join(G.ROOT, "results/cascade_methods/artifacts/_free_head_parts")
FEATFREE = os.path.join(G.ROOT, "feats_free")
os.makedirs(OUTDIR, exist_ok=True)


# ---------------------------------------------------------------------------- loaders
def load_free(cap, layer=21):
    """(keys, {kind: (n,3584) float32}) for one capture run; complete rows only."""
    stem = os.path.join(FEATFREE, f"free_{cap}_L{layer}")
    rows = [json.loads(l) for l in open(stem + ".rows.jsonl") if l.strip()]
    seen, keep = set(), []
    for r in rows:                                     # last write wins; skip errored rows
        if "err" in r or r["row"] in seen:
            continue
        seen.add(r["row"]); keep.append(r)
    keep.sort(key=lambda r: r["row"])
    idx = np.array([r["row"] for r in keep], int)
    out = {}
    for kind in ("h_span_tf", "h_span_ar", "h_last_tf", "h_last_ar"):
        out[kind] = np.load(stem + f".{kind}.npy", mmap_mode="r")[idx].astype(np.float32)
    keys = [(r["ds"], r["idx"], r["na"]) for r in keep]
    return keys, out, keep


def slot_matrix(keys, H, questions):
    """(n_questions, 8) row index into H for every pool slot, or -1 when the row is missing."""
    k2r = {k: i for i, k in enumerate(keys)}
    S = np.full((len(questions), 8), -1, int)
    for i, q in enumerate(questions):
        for s, a in enumerate(q.preds):
            S[i, s] = k2r.get((q.ds, q.idx, G.norm(a)), -1)
    return S


def head_scores_from(H, S, sel, items):
    """{(ds, idx): 8 slot scores} for the HEAD ALONE, from an arbitrary feature matrix."""
    L = sel.head_logits(H)                                  # (8 seeds, n_rows)
    out = {}
    for i, it in enumerate(items):
        rows = S[i]
        if (rows < 0).any():
            out[(it["ds"], it["idx"])] = [G.MISSING_SCORE] * 8
            continue
        out[(it["ds"], it["idx"])] = list(np.mean([G.rank_avg(L[s][rows]) for s in range(L.shape[0])],
                                                  axis=0))
    return out, L


def dev_stats(A, B):
    d = np.abs(A - B)
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    cos = (A * B).sum(1) / np.maximum(na * nb, 1e-12)
    rel = np.linalg.norm(A - B, axis=1) / np.maximum(na, 1e-12)
    return {"n_rows": int(A.shape[0]),
            "max_abs_deviation": float(d.max()), "mean_abs_deviation": float(d.mean()),
            "max_abs_value": float(np.abs(A).max()),
            "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
            "rel_l2_max": float(rel.max()), "rel_l2_mean": float(rel.mean()),
            "n_rows_bitidentical_fp16": int((A == B).all(1).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="cap320,fullres")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "equivalence.json"))
    A = ap.parse_args()

    items = G.load_items()
    em = F.em_slot_labels(items)
    sel = FrozenSelector.load(F.SELDIR)
    ev = G.load_candidates("eval", "generator", layers=[21], pooling=("span",), order="concat")
    Hdep = ev.matrix("span", 21)                                   # DEPLOYED features
    keys_dep = [(r["ds"], r["idx"], r["na"]) for r in ev.rows]
    Sdep = slot_matrix(keys_dep, Hdep, ev.questions)
    inc = G.incumbent_scores()

    sources = {"deployed": (keys_dep, Hdep)}
    row_meta = {}
    for cap in [c for c in A.caps.split(",") if c]:
        p = os.path.join(FEATFREE, f"free_{cap}_L21.rows.jsonl")
        if not os.path.exists(p):
            print(f"  [skip] no capture for {cap}")
            continue
        keys, Hs, keep = load_free(cap)
        row_meta[cap] = keep
        sources[f"{cap}_tf"] = (keys, Hs["h_span_tf"])
        sources[f"{cap}_ar"] = (keys, Hs["h_span_ar"])
        print(f"  [load] {cap}: {len(keys)}/8943 rows")

    # ------------------------------------------------------------------ arms
    arms, head_logits, slots = {}, {}, {}
    for nm, (keys, H) in sources.items():
        S = Sdep if nm == "deployed" else slot_matrix(keys, H, ev.questions)
        slots[nm] = S
        hs, L = head_scores_from(H, S, sel, items)
        head_logits[nm] = (keys, L)
        fu = {}
        for i, it in enumerate(items):
            k = (it["ds"], it["idx"])
            fu[k] = list(G.rank_avg(np.asarray(inc[k], float)) + G.rank_avg(np.asarray(hs[k], float)))
        cov = float((S >= 0).all(1).mean())
        arms[f"head_only::{nm}"] = F.endpoint(hs, items, em, f"head-only, features={nm}")
        arms[f"head_only::{nm}"]["slot_coverage"] = cov
        arms[f"fusion::{nm}"] = F.endpoint(fu, items, em, f"fusion(inc + head), features={nm}")
        arms[f"fusion::{nm}"]["slot_coverage"] = cov
    arms["incumbent"] = F.endpoint(inc, items, em, "incumbent LoRA verifier alone (no head)")
    arms["incumbent"]["slot_coverage"] = 1.0

    # ------------------------------------------------------------------ equivalence
    eq = {}
    for cap in [c for c in A.caps.split(",") if c]:
        if f"{cap}_tf" not in sources:
            continue
        keys, _ = sources[f"{cap}_tf"]
        k2d = {k: i for i, k in enumerate(keys_dep)}
        di = np.array([k2d[k] for k in keys], int)
        pairs = {
            f"{cap}_ar_vs_{cap}_tf": (sources[f"{cap}_ar"][1], sources[f"{cap}_tf"][1], None),
            f"{cap}_tf_vs_deployed": (sources[f"{cap}_tf"][1], Hdep[di], None),
            f"{cap}_ar_vs_deployed": (sources[f"{cap}_ar"][1], Hdep[di], None),
        }
        for nm, (Aa, Bb, _) in pairs.items():
            eq[nm] = dev_stats(Aa, Bb)
        # head-logit level, on the SAME rows
        kd = {k: i for i, k in enumerate(head_logits["deployed"][0])}
        jd = np.array([kd[k] for k in keys], int)
        Ldep = head_logits["deployed"][1][:, jd]
        for kind in ("tf", "ar"):
            Lk = head_logits[f"{cap}_{kind}"][1]
            eq[f"head_logits_{cap}_{kind}_vs_deployed"] = {
                "max_abs_deviation": float(np.abs(Lk - Ldep).max()),
                "mean_abs_deviation": float(np.abs(Lk - Ldep).mean()),
                "logit_sd_deployed": float(Ldep.std()),
                "pearson_r": float(np.corrcoef(Lk.ravel(), Ldep.ravel())[0, 1])}
        m = row_meta[cap]
        eq[f"{cap}_row_diagnostics"] = {
            "n_rows": len(m),
            "n_errors": sum(1 for r in m if "err" in r),
            "tokenisation_boundary_ok_frac": float(np.mean([r.get("boundary_ok", False) for r in m])),
            "span_start_matches_prompt_end_frac": float(np.mean([r.get("span_ok", False) for r in m])),
            "forced_tokens_emitted_frac": float(np.mean([r.get("forced_ok", False) for r in m])),
            "n_ans_tok_mean": float(np.mean([r["n_ans_tok"] for r in m])),
            "n_img_tok_mean": float(np.mean([r["n_img_tok"] for r in m])),
            "n_tok_mean": float(np.mean([r["n_tok"] for r in m])),
            "n_prompt_tok_mean": float(np.mean([r["n_prompt_tok"] for r in m])),
            "note_boundary": "boundary_ok: tokenising prompt+answer gives prompt_ids + "
                             "tokenise(answer) exactly, so the span the head reads is exactly the "
                             "candidate's tokens and the AR path can force the identical ids."}

    # ------------------------------------------------------------------ comparisons
    cmp = {}
    base = "fusion::deployed"
    for nm in arms:
        if nm == base:
            continue
        cmp[f"{nm}__vs__{base}"] = F.compare(arms[nm], arms[base], nboot=A.nboot)
    for cap in [c for c in A.caps.split(",") if c]:
        for kind in ("head_only", "fusion"):
            a, b = f"{kind}::{cap}_ar", f"{kind}::{cap}_tf"
            if a in arms and b in arms:
                cmp[f"{a}__vs__{b}"] = F.compare(arms[a], arms[b], nboot=A.nboot)
        if f"head_only::{cap}_ar" in arms:
            cmp[f"head_only::{cap}_ar__vs__head_only::deployed"] = F.compare(
                arms[f"head_only::{cap}_ar"], arms["head_only::deployed"], nboot=A.nboot)
            cmp[f"head_only::{cap}_ar__vs__incumbent"] = F.compare(
                arms[f"head_only::{cap}_ar"], arms["incumbent"], nboot=A.nboot)

    # ------------------------------------------------------------------ candidate-level AUROC
    ylab = {(r["ds"], r["idx"], r["na"]): int(r["y"]) for r in ev.rows}
    auroc = {}
    for nm, (keys, L) in head_logits.items():
        yl = np.array([ylab[k] for k in keys], float)
        auroc[nm] = {"head_mean_logit_auroc": G.auroc(yl, L.mean(0)),
                     "n_rows": int(len(keys)),
                     "note": "row-level AUROC of the 8-seed MEAN RAW head logit against the judge "
                             "label of the distinct candidate (not the 18760-slot convention)"}

    # ------------------------------------------------------------------ shuffled-feature null
    # A frozen head fed features belonging to OTHER rows must fall to the random-pick floor.
    rngfloor = G.random_pick(items)
    shuf = {}
    for nm, (keys, H) in sources.items():
        S = slots[nm]
        vals = []
        for seed in range(10):
            rng = np.random.default_rng(1000 + seed)
            perm = rng.permutation(H.shape[0])
            hs, _ = head_scores_from(H[perm], S, sel, items)
            vals.append(G.sel_eff(hs, items)["sel_eff"])
        shuf[nm] = {"sel_eff_mean": float(np.mean(vals)), "sel_eff_sd": float(np.std(vals, ddof=1)),
                    "sel_eff_min": float(np.min(vals)), "sel_eff_max": float(np.max(vals)),
                    "random_pick_floor": rngfloor["sel_eff"], "n_seeds": 10}

    # ------------------------------------------------------------------ vs the NEW baseline
    v7 = {nm: F.vs_always_7b(a, items, nboot=A.nboot) for nm, a in arms.items()}

    # ------------------------------------------------------------------ cost
    cost = None
    if "cap320" in row_meta:
        good = [r for r in row_meta["cap320"] if "err" not in r]
        if len(good) >= 8000:
            cost = C.cost_table(good, n_q=len(items))
        else:
            cost = {"status": "cap320 capture incomplete (%d/8943 rows) -- cost table not computed"
                              % len(good)}

    out = {
        "title": "BUILD 1 -- MAKE THE HEAD FREE: capture the generator-frame features during "
                 "generation instead of recomputing them",
        "date": "2026-08-16",
        "reproduce": ["python3 src/training_methods/free_head_selectors.py",
                      "CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/free_head_capture.py --cap cap320 --out feats_free",
                      "CUDA_VISIBLE_DEVICES=1 python3 src/training_methods/free_head_capture.py --cap fullres --out feats_free",
                      "python3 src/training_methods/free_head_analyze.py"],
        "null_tests": {
            "genframe_metric": G.null_test(),
            "frozen_selector_reload": GS.verify(F.SELDIR),
            "flop_model_rebuild": C.null_test(),
        },
        "nboot": A.nboot,
        "sources": {k: ("feats_hidden (fullres, teacher-forced) -- what ships today"
                        if k == "deployed" else f"feats_free/free_{k.replace('_tf','').replace('_ar','')}_L21")
                    for k in sources},
        "equivalence": eq,
        "candidate_auroc": auroc,
        "shuffled_feature_null": shuf,
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in arms.items()},
        "comparisons": cmp,
        "vs_always_7b": v7,
        "cost": cost,
    }
    json.dump(out, open(A.out, "w"), indent=1, default=float)

    # ------------------------------------------------------------------ print
    print("\nNULL TESTS")
    for k, v in out["null_tests"].items():
        print(f"  {k:<24} pass={v['pass']}  maxdev="
              f"{v.get('max_abs_deviation', v.get('max_abs_deviation_gflops')):.3e}")
    print("\nEQUIVALENCE (h_span, layer 21)")
    for k, v in eq.items():
        if "row_diagnostics" in k or "head_logits" in k:
            continue
        print(f"  {k:<30} maxabs {v['max_abs_deviation']:.4f}  cos_min {v['cosine_min']:.6f}  "
              f"relL2_mean {v['rel_l2_mean']:.5f}  bit-identical {v['n_rows_bitidentical_fp16']}/{v['n_rows']}")
    print("\nARMS")
    hdr = f"{'arm':<26}{'cov':>6}{'selE_j':>9}{'sel_j':>9}{'selE_em':>9}{'sel_em':>9}"
    print(hdr)
    for k, a in arms.items():
        print(f"{k:<26}{a['slot_coverage']:>6.3f}{a['sel_eff_judge']:>9.6f}{a['selected_judge']:>9.6f}"
              f"{a['sel_eff_em']:>9.6f}{a['selected_em']:>9.6f}")
    print("\nCOMPARISONS")
    for k, c in cmp.items():
        print(f"  {k}  picks_changed={c['picks_changed']}")
        for cur in ("judge", "em"):
            d = c[cur]
            print(f"     {cur:<6} dSEL {d['d_selected']:+.6f} [{d['d_selected_ci'][0]:+.6f},"
                  f"{d['d_selected_ci'][1]:+.6f}] {d['verdict']:<5} guardrail={d['guardrail_clean_sel_eff']}")
    print("\nvs ALWAYS-7B (greedy)  -- per-cell CI-clean wins out of the 3 open cells")
    for nm in ("incumbent", "head_only::deployed", "fusion::deployed",
               "head_only::cap320_ar", "fusion::cap320_ar",
               "head_only::fullres_ar", "fusion::fullres_ar"):
        if nm not in v7:
            continue
        r = v7[nm]
        print(f"  {nm:<26} judge {r['judge']['pooled_delta']:+.5f} "
              f"[{r['judge']['pooled_ci'][0]:+.5f},{r['judge']['pooled_ci'][1]:+.5f}] "
              f"{r['judge']['n_cells_CI_clean_win']}/3 cells | "
              f"em {r['em']['pooled_delta']:+.5f} "
              f"[{r['em']['pooled_ci'][0]:+.5f},{r['em']['pooled_ci'][1]:+.5f}] "
              f"{r['em']['n_cells_CI_clean_win']}/3 cells")
    if cost and "arms_fixed_bestof8" in cost:
        print("\nCOST  (1 FLOP-eq = one always-7B answer)")
        print(f"  {'arm':<38}{'passes':>9}{'as-charged':>12}{'measured':>11}")
        for k, v in cost["arms_fixed_bestof8"].items():
            print(f"  {k:<38}{v['total_passes']:>9.3f}{v['flopeq_as_charged']:>12.3f}"
                  f"{v['flopeq_measured_geometry']:>11.3f}")
    print("\nwrote", A.out)


if __name__ == "__main__":
    main()
