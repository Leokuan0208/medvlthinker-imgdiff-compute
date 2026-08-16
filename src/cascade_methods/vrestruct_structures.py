#!/usr/bin/env python3
"""vrestruct_structures.py -- QUESTION 1 (+5): should the LoRA verifier be dropped entirely?

The head ALONE (0.7956) already beats the LoRA verifier ALONE (0.7752).  The LoRA's only
contribution is the fusion increment (0.8065 fused), and it costs 3.823 forward passes per
question -- at max_pixels 1,003,520, i.e. 2.236 FLOP-eq each, not the 1.0 the project charges.
Under a COST objective that trade has to be priced, not assumed.

This script measures, on the frozen 2,345-question eval pool, with NOTHING refit:
  * lora_only / head_only(k seeds) / fused(k seeds), k = 1..8
  * controls: random-slot (closed form), self-consistency, oracle@8, greedy-7B
  * sel_eff AND selected accuracy in BOTH currencies (32B judge and normalised exact match) on
    IDENTICAL picks, per cell, with the guardrail and paired item bootstraps
  * the 8-cell macro under the NEW baseline framing (7B only, no 32B at test time: the 5 MCQ cells
    stay at greedy-7B because no 7B-only MCQ mechanism has ever measured positive on this pool --
    sevenb_only_frontier_2026-08-12.json PART1.menu_note)
  * a permutation null for every structure
  * measured cost of each structure in forward passes AND FLOP-eq against always-7B = 1.0
  * QUESTION 5 side-quests: seed-count sufficiency, early-stop scoring, dedup normalisation

CPU only.  No GPU, no refit, no new generation.  Every score comes off disk.

    OMP_NUM_THREADS=4 python3 src/cascade_methods/vrestruct_structures.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))

import genframe_data as G          # noqa: E402
import genframe_selector as GS     # noqa: E402
import vrestruct_lib as V          # noqa: E402

PARTS = V.PARTS
MCQ_CELLS = {"PMC_VQA": 0.542656, "SLAKE_closed": 0.825359, "VQA_RAD_closed": 0.780876,
             "PATH_VQA_closed": 0.840869, "MedXpertQA-MM": 0.2615}
MCQ_SRC = ("artifacts/sevenb_only_frontier_2026-08-12.json "
           "PART1_7B_only_frontier.menu_per_cell_accuracy_EVAL_VISIBLE.*.greedy_7b")
OPEN_CELL_NAME = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open",
                  "pathvqa_open": "PATH_VQA_open"}


# ======================================================================================
def null_tests(P, L):
    """Everything must reproduce from disk BEFORE anything new is reported."""
    nt = {}
    nt["NT1_frozen_metric"] = G.null_test()

    # NT2: the frozen 8-seed selector reproduces its published endpoint through THIS harness
    S_head8 = V.head_rank_slots(P, L, range(8))
    S_fused8 = V.rank_rows(P["inc"]) + V.rank_rows(S_head8)
    r = V.evaluate(P, V.picks_of(S_fused8), "fused_8seed")
    pub = GS.PUBLISHED
    dev = {"sel_eff": abs(r["judge"]["sel_eff"] - pub["sel_eff"]),
           "acc": abs(r["judge"]["acc"] - pub["acc"])}
    for d in G.EVAL_DS:
        dev[f"per_ds.{d}"] = abs(r["judge"]["per_ds"][d]["sel_eff"] - pub["per_ds"][d])
    nt["NT2_frozen_selector_through_this_harness"] = {
        "measured": {"sel_eff": r["judge"]["sel_eff"], "acc": r["judge"]["acc"],
                     "per_ds": {d: r["judge"]["per_ds"][d]["sel_eff"] for d in G.EVAL_DS}},
        "published": pub, "abs_deviation": dev, "max_abs_deviation": max(dev.values()),
        "pass": bool(max(dev.values()) <= 1e-6)}

    # NT3: the incumbent alone reproduces the bar
    ri = V.evaluate(P, V.picks_of(P["inc"]), "lora_only")
    nt["NT3_incumbent_bar"] = {
        "measured_sel_eff": ri["judge"]["sel_eff"], "published_sel_eff": G.PUBLISHED["sel_eff"],
        "abs_deviation": abs(ri["judge"]["sel_eff"] - G.PUBLISHED["sel_eff"]),
        "pass": bool(abs(ri["judge"]["sel_eff"] - G.PUBLISHED["sel_eff"]) <= 1e-6)}

    # NT4: the EXACT identity  selected = oracle@8 x sel_eff
    idn = max(r["identity_dev_judge"], ri["identity_dev_judge"])
    nt["NT4_identity_selected_eq_oracle_times_seleff"] = {
        "max_abs_deviation": float(idn), "pass": bool(idn < 1e-12),
        "note": "asserted for every structure this script reports, not just these two"}

    # NT5: the per-cell open accuracies reproduce the 7B-only frontier's menu
    fro = json.load(open(os.path.join(V.ART, "sevenb_only_frontier_2026-08-12.json")))
    menu = fro["PART1_7B_only_frontier"]["menu_per_cell_accuracy_EVAL_VISIBLE"]
    d5 = {}
    for ds, name in OPEN_CELL_NAME.items():
        d5[name] = abs(r["judge"]["per_ds"][ds]["acc"] - menu[name]["bo8_frozen_ens8_selector"])
    nt["NT5_reproduces_7B_only_frontier_open_cells"] = {
        "abs_deviation": d5, "max_abs_deviation": max(d5.values()),
        "pass": bool(max(d5.values()) <= 1e-5), "source": MCQ_SRC}

    nt["all_pass"] = bool(nt["NT1_frozen_metric"]["pass"] and
                          nt["NT2_frozen_selector_through_this_harness"]["pass"] and
                          nt["NT3_incumbent_bar"]["pass"] and
                          nt["NT4_identity_selected_eq_oracle_times_seleff"]["pass"] and
                          nt["NT5_reproduces_7B_only_frontier_open_cells"]["pass"])
    return nt, S_head8, S_fused8


# ======================================================================================
def macro8(res):
    """8-cell macro under the 7B-ONLY framing: 5 MCQ cells at greedy-7B, 3 open at this structure."""
    cells = dict(MCQ_CELLS)
    for ds, name in OPEN_CELL_NAME.items():
        cells[name] = res["judge"]["per_ds"][ds]["acc"]
    return float(np.mean(list(cells.values()))), cells


def macro8_vector(P, res):
    """Per-item 0/1 vector over the 8-cell macro, for a paired bootstrap of the macro delta.

    Only the 3 open cells vary between structures; the 5 MCQ cells are byte-identical constants,
    so the macro delta between two structures is exactly (1/8) * sum over the 3 open cells of the
    per-cell accuracy delta.  We bootstrap the open cells item-wise and scale.
    """
    return res["judge"]["got"]


def cell_delta_boot(P, a, b, nboot=V.NBOOT, seed=V.BOOT_SEED):
    """Per-open-cell paired bootstrap of the macro contribution (1/8 per cell), judge currency."""
    rng = np.random.default_rng(seed)
    tot = np.zeros(nboot)
    per = {}
    for j, ds in enumerate(G.EVAL_DS):
        m = P["ds_index"] == j
        d = (a["judge"]["got"][m] - b["judge"]["got"][m]).astype(float)
        idx = rng.integers(0, len(d), size=(nboot, len(d)))
        bs = d[idx].mean(1)
        per[OPEN_CELL_NAME[ds]] = dict(delta=float(d.mean()),
                                       lo=float(np.percentile(bs, 2.5)),
                                       hi=float(np.percentile(bs, 97.5)))
        tot += bs / 8.0
    md = sum(per[OPEN_CELL_NAME[ds]]["delta"] for ds in G.EVAL_DS) / 8.0
    return dict(macro_delta=float(md), lo=float(np.percentile(tot, 2.5)),
                hi=float(np.percentile(tot, 97.5)),
                significant=bool(np.percentile(tot, 2.5) > 0 or np.percentile(tot, 97.5) < 0),
                per_cell=per)


# ======================================================================================
def main():
    os.makedirs(PARTS, exist_ok=True)
    np.random.seed(V.BOOT_SEED)
    print("loading pool ...", flush=True)
    P = V.load_pool()
    L = V.head_logits(P)
    print(f"  n={P['n']} rows={P['X'].shape}", flush=True)

    nt, S_head8, S_fused8 = null_tests(P, L)
    print("NULL TESTS pass =", nt["all_pass"], flush=True)
    for k, v in nt.items():
        if isinstance(v, dict) and "max_abs_deviation" in v:
            print(f"   {k}: max_abs_dev {v['max_abs_deviation']:.3e} pass={v.get('pass')}",
                  flush=True)
    if not nt["all_pass"]:
        json.dump(nt, open(os.path.join(PARTS, "nulls.json"), "w"), indent=1, default=float)
        raise SystemExit("NULL TEST FAILED -- refusing to report new numbers")
    json.dump(nt, open(os.path.join(PARTS, "nulls.json"), "w"), indent=1, default=float)

    c = V.cost_constants()
    pc = V.pass_counts(P)
    ndist = pc["per_question_distinct"]

    # ---------------------------------------------------------------- the structures
    structures = {}
    scores = {}
    scores["lora_only"] = P["inc"].copy()
    for k in range(1, 9):
        scores[f"head_only_{k}seed"] = V.head_rank_slots(P, L, range(k))
        scores[f"fused_{k}seed"] = (V.rank_rows(P["inc"])
                                    + V.rank_rows(scores[f"head_only_{k}seed"]))
    # self-consistency control
    sc = np.zeros((P["n"], 8))
    for i, it in enumerate(P["items"]):
        from collections import Counter
        cn = Counter(G.norm(a) for a in it["preds"])
        sc[i] = [cn[G.norm(a)] for a in it["preds"]]
    scores["self_consistency"] = sc

    for name, S in scores.items():
        r = V.evaluate(P, V.picks_of(S), name)
        if r["identity_dev_judge"] > 1e-12:
            raise ValueError(f"identity broken for {name}: {r['identity_dev_judge']}")
        structures[name] = r

    base = structures["lora_only"]
    head8 = structures["head_only_8seed"]
    fused8 = structures["fused_8seed"]

    # ---------------------------------------------------------------- cost per structure
    def cost_open(gen_passes, ver_passes, ver_flopeq, head_passes, head_flopeq,
                  gen_flopeq_total=None):
        """Per-open-question FLOP-eq. gen_flopeq_total lets us swap the generation convention."""
        gen = gen_passes * 1.0 if gen_flopeq_total is None else gen_flopeq_total
        return dict(gen_passes=float(gen_passes), ver_passes=float(ver_passes),
                    head_passes=float(head_passes),
                    total_passes=float(gen_passes + ver_passes + head_passes),
                    gen_flopeq=float(gen), ver_flopeq=float(ver_passes * ver_flopeq),
                    head_flopeq=float(head_passes * head_flopeq),
                    total_flopeq=float(gen + ver_passes * ver_flopeq
                                       + head_passes * head_flopeq))

    D = float(ndist.mean())
    costs = {}
    costs["baseline_always_7b"] = cost_open(1, 0, 0, 0, 0)
    costs["current_deployed_fused"] = cost_open(8, D, c["ver_1003520_flopeq"],
                                                D, c["head_1003520_flopeq"])
    costs["lora_only_asdeployed"] = cost_open(8, D, c["ver_1003520_flopeq"], 0, 0)
    costs["head_only_teacherforced_1003520"] = cost_open(8, 0, 0, D, c["head_1003520_flopeq"])
    costs["head_only_teacherforced_cap320"] = cost_open(8, 0, 0, D, c["head_cap320_flopeq"])
    costs["head_only_free_capture"] = cost_open(8, 0, 0, 0, 0)
    costs["fused_verifier_at_cap320"] = cost_open(8, D, c["ver_cap320_flopeq"],
                                                  D, c["head_cap320_flopeq"])
    costs["fused_free_head_plus_prefixshared_verifier"] = cost_open(
        8, 0, 0, 0, 0)
    costs["fused_free_head_plus_prefixshared_verifier"].update(
        ver_passes=float(D),
        ver_flopeq=float(c["ver_prefix_cost_units"]
                         + c["ver_marginal_cost_per_candidate_units"] * D),
        total_passes=8.0 + D,
        total_flopeq=8.0 + c["ver_prefix_cost_units"]
        + c["ver_marginal_cost_per_candidate_units"] * D)

    # every structure also priced with the SHARED-PREFILL generation convention (question 2)
    gshared = V.G_of_N(8, c)
    for k in list(costs):
        cc = dict(costs[k])
        n_gen = cc["gen_passes"]
        cc_shared = dict(cc)
        cc_shared["gen_flopeq"] = V.G_of_N(n_gen, c) if n_gen >= 1 else 0.0
        cc_shared["total_flopeq"] = (cc_shared["gen_flopeq"] + cc["ver_flopeq"]
                                     + cc["head_flopeq"])
        costs[k] = {"gen_charged_per_sample": cc, "gen_shared_prefill": cc_shared}

    # ---------------------------------------------------------------- comparisons
    cmp_ = {}
    for a_name, b_name in (("head_only_8seed", "lora_only"),
                           ("fused_8seed", "head_only_8seed"),
                           ("fused_8seed", "lora_only"),
                           ("head_only_8seed", "self_consistency"),
                           ("lora_only", "self_consistency")):
        A_, B_ = structures[a_name], structures[b_name]
        e = {}
        for cur in ("judge", "em"):
            e[cur] = dict(
                pooled_acc=V.paired_boot(A_[cur]["got"], B_[cur]["got"]),
                sel_eff_on_recoverable=V.paired_boot(A_[cur]["got"], B_[cur]["got"],
                                                     mask=A_[cur]["rec"] == 1),
                per_cell={OPEN_CELL_NAME[ds]: V.paired_boot(
                    A_[cur]["got"][P["ds_index"] == j], B_[cur]["got"][P["ds_index"] == j])
                    for j, ds in enumerate(G.EVAL_DS)})
            e[cur]["guardrail_clean"] = bool(all(
                e[cur]["per_cell"][OPEN_CELL_NAME[ds]]["delta"] >= 0 for ds in G.EVAL_DS))
        e["macro8_judge"] = cell_delta_boot(P, A_, B_)
        cmp_[f"{a_name}__vs__{b_name}"] = e

    # vs the always-7B greedy baseline (judge currency only -- see limitation note)
    g = P["greedy_ok"].astype(float)
    vs7b = {}
    for name in ("lora_only", "head_only_8seed", "fused_8seed", "self_consistency"):
        r = structures[name]
        vs7b[name] = dict(
            judge_pooled=V.paired_boot(r["judge"]["got"], g),
            per_cell={OPEN_CELL_NAME[ds]: V.paired_boot(
                r["judge"]["got"][P["ds_index"] == j], g[P["ds_index"] == j])
                for j, ds in enumerate(G.EVAL_DS)})
        vs7b[name]["guardrail_clean"] = bool(all(
            vs7b[name]["per_cell"][OPEN_CELL_NAME[ds]]["delta"] >= 0 for ds in G.EVAL_DS))
        m, cells = macro8(r)
        vs7b[name]["macro8"] = m
        vs7b[name]["macro8_cells"] = cells
        gr = V.evaluate(P, np.zeros(P["n"], int), "greedy_placeholder")
        gr["judge"]["got"] = g.astype(int)
        for j, ds in enumerate(G.EVAL_DS):
            mm = P["ds_index"] == j
            gr["judge"]["per_ds"][ds]["acc"] = float(g[mm].mean())
        vs7b[name]["macro8_delta_vs_always7b"] = cell_delta_boot(P, r, gr)

    # ---------------------------------------------------------------- permutation nulls
    perm = {}
    for name in ("lora_only", "head_only_8seed", "fused_8seed"):
        d = V.perm_null(P, (lambda s=scores[name]: s), nperm=1000)
        obs = structures[name]["judge"]["sel_eff"]
        perm[name] = dict(observed_sel_eff=obs, null_mean=float(d.mean()),
                          null_sd=float(d.std()), null_p95=float(np.percentile(d, 95)),
                          p_value=float((d >= obs).mean()), nperm=1000)

    # ---------------------------------------------------------------- seed sufficiency
    seedtab = {}
    for k in range(1, 9):
        subs = list(itertools.combinations(range(8), k))
        if len(subs) > 40:
            rng = np.random.default_rng(V.BOOT_SEED + k)
            subs = [tuple(sorted(rng.choice(8, k, replace=False))) for _ in range(40)]
            subs = sorted(set(subs))
        ho, fu = [], []
        for s in subs:
            Sh = V.head_rank_slots(P, L, s)
            ho.append(V.evaluate(P, V.picks_of(Sh))["judge"]["sel_eff"])
            fu.append(V.evaluate(P, V.picks_of(V.rank_rows(P["inc"]) + V.rank_rows(Sh)))
                      ["judge"]["sel_eff"])
        seedtab[k] = dict(n_subsets=len(subs),
                          head_only=dict(mean=float(np.mean(ho)), sd=float(np.std(ho, ddof=1)) if len(ho) > 1 else 0.0,
                                         min=float(np.min(ho)), max=float(np.max(ho)),
                                         first_k=float(structures[f"head_only_{k}seed"]["judge"]["sel_eff"])),
                          fused=dict(mean=float(np.mean(fu)), sd=float(np.std(fu, ddof=1)) if len(fu) > 1 else 0.0,
                                     min=float(np.min(fu)), max=float(np.max(fu)),
                                     first_k=float(structures[f"fused_{k}seed"]["judge"]["sel_eff"])))
        print(f"  seeds k={k}: head {np.mean(ho):.6f}  fused {np.mean(fu):.6f}", flush=True)

    # ---------------------------------------------------------------- early stop (LoRA scoring)
    # score distinct candidates in RECORDED slot order, stop once a score >= tau; pick argmax so far.
    early = {}
    inc = P["inc"]
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01):
        passes, picks = np.zeros(P["n"]), np.zeros(P["n"], int)
        for i, it in enumerate(P["items"]):
            seen, order, best, bi, np_ = {}, [], -np.inf, 0, 0
            for s, a in enumerate(it["preds"]):
                na = G.norm(a)
                if na not in seen:
                    seen[na] = inc[i, s]
                    np_ += 1
                v = seen[na]
                if v > best:
                    best, bi = v, s
                if v >= tau:
                    break
            passes[i], picks[i] = np_, bi
        r = V.evaluate(P, picks, f"lora_earlystop_tau{tau}")
        early[str(tau)] = dict(mean_ver_passes=float(passes.mean()),
                               judge_sel_eff=r["judge"]["sel_eff"], judge_acc=r["judge"]["acc"],
                               em_acc=r["em"]["acc"],
                               per_cell_acc={OPEN_CELL_NAME[ds]: r["judge"]["per_ds"][ds]["acc"]
                                             for ds in G.EVAL_DS})

    # ---------------------------------------------------------------- dedup normalisation
    import re
    import string
    ART_RE = re.compile(r"^(the|a|an)\s+")

    def norm2(s):
        t = G.norm(s)
        t = t.translate(str.maketrans("", "", string.punctuation))
        t = ART_RE.sub("", t).strip()
        t = re.sub(r"\s+", " ", t)
        return t

    def norm3(s):
        t = norm2(s)
        toks = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in t.split()]
        return " ".join(sorted(toks))

    dedup = {}
    for nm, fn in (("current_G.norm", G.norm), ("plus_punct_articles", norm2),
                   ("plus_plural_bagofwords", norm3)):
        nd, lossy, lossy_slots = [], 0, 0
        for i, it in enumerate(P["items"]):
            grp = {}
            for s, a in enumerate(it["preds"]):
                grp.setdefault(fn(a), []).append(s)
            nd.append(len(grp))
            for k2, ss in grp.items():
                labs = set(int(P["judge"][i, s]) for s in ss)
                if len(labs) > 1:
                    lossy += 1
                    lossy_slots += len(ss)
        dedup[nm] = dict(mean_distinct=float(np.mean(nd)),
                         n_groups_with_mixed_judge_labels=int(lossy),
                         n_slots_in_those_groups=int(lossy_slots),
                         note="a group with mixed labels is a MERGE THAT LOSES INFORMATION: two "
                              "answers that normalise the same but are not both correct")

    out = dict(
        title="Verifier restructuring, Q1+Q5: is the LoRA verifier worth its passes?",
        date="2026-08-16", cpu_only=True, no_refit=True, no_new_generation=True,
        null_tests=nt, cost_constants=c, pass_counts={"mean_distinct_answers": D},
        structures={k: {cur: {kk: vv for kk, vv in r[cur].items()
                              if kk not in ("got", "rec")}
                        for cur in ("judge", "em")} | {"identity_dev_judge": r["identity_dev_judge"]}
                    for k, r in structures.items()},
        costs=costs, comparisons=cmp_, vs_always_7b=vs7b, permutation_nulls=perm,
        seed_sufficiency=seedtab, early_stop_lora=early, dedup_normalisation=dedup,
        mcq_cells_held_at_greedy_7b=dict(cells=MCQ_CELLS, source=MCQ_SRC))
    json.dump(out, open(os.path.join(PARTS, "structures.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(PARTS, "structures.json"), flush=True)

    print("\n=== sel_eff (judge) ===")
    for k in ("lora_only", "head_only_8seed", "fused_8seed", "self_consistency"):
        r = structures[k]
        print(f"  {k:22s} sel_eff {r['judge']['sel_eff']:.6f}  acc {r['judge']['acc']:.6f}  "
              f"acc_em {r['em']['acc']:.6f}  macro8 {macro8(r)[0]:.6f}")


if __name__ == "__main__":
    main()
