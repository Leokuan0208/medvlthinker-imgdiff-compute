#!/usr/bin/env python3
"""decoding_ladder_cold.py -- the COLD extension of the temperature ladder (2026-08-14).

Pre-registration: results/cascade_methods/artifacts/_decoding_ladder_cold_prereg.json (written
2026-08-14T07:14:20Z, BEFORE any candidate string of any new rung existed).

What this measures, per temperature rung, on the frozen 2,345-item open pool:
  * distinct-candidate count and the UNANIMOUS fraction (pools with exactly 1 distinct answer)
  * oracle@8, sel_eff, SELECTED -- in BOTH grading currencies, on IDENTICAL verifier picks
  * the COLLAPSE POINT: SELECTED minus a single draw from the SAME pool (the verifier's contribution)
  * the N-interaction: SELECTED at N = 1, 2, 4, 8 by EXACT subsampling of the existing 8-pools

THE N-SUBSAMPLING IS EXACT, NOT MONTE CARLO.  Verifier scores are fixed, so given a subset of N of the
M drawn slots the pick is deterministic.  Rank the slots by (-score, index) -- the frozen first-index
tie-break -- and the slot of rank r is picked iff it is in the subset and none of the r better-ranked
slots are, with probability C(M-1-r, N-1) / C(M, N).  Summing over r gives 1 (hockey-stick identity),
which the script asserts.  E[correct] is then the label-weighted sum.  At N=1 this reduces to the plain
per-item mean of the slot labels, i.e. a single random draw from the same pool -- the baseline the
collapse point is defined against.

T = 0.0 is a special rung: vLLM 0.10.1.1 refuses n>1 with greedy sampling ("n must be 1 when using
greedy sampling, got 8."), so it is generated at n=1, three times with three seeds.  The frozen metric
handles a 1-slot pool correctly on its own (unfilled slots hold MISSING_SCORE, argmax lands on slot 0),
giving oracle = modal = SELECTED and sel_eff = 1.0 by construction.  Nothing is replicated to fake 8
draws; the 3-seed byte-identity rate is reported instead as the determinism measurement.

  python3 src/cascade_methods/decoding_ladder_cold.py
"""
import argparse, json, os, sys
from collections import defaultdict
from math import comb
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G                       # noqa: E402
from src.cascade_methods.decoding_sweep_analyse import (                   # noqa: E402
    load_judge, load_vscores, load_pool, boot, DS, SWEEP)

NBOOT, BSEED = 10000, 20260814

# tag -> (temperature, provenance).  Order is COLD -> HOT.
LADDER = [("T00", 0.0, "in_session_NEW (n=1, greedy)"),
          ("T005", 0.05, "in_session_NEW"),
          ("T01", 0.1, "in_session_NEW"),
          ("T02", 0.2, "in_session_NEW"),
          ("T03r", 0.3, "in_session_REGENERATED"),
          ("T04", 0.4, "in_session_NEW"),
          ("T05r", 0.5, "in_session_REGENERATED"),
          ("T07r", 0.7, "in_session_CONTROL")]
# shown for curve continuity only -- generated in the 2026-08-13/14 round, NOT regenerated here.
PRIOR = [("T03", 0.3), ("T05", 0.5), ("T07", 0.7), ("T10", 1.0), ("T13", 1.3)]
CONTROL = "T07r"


# ------------------------------------------------------------------ per-pool measurement
def pick_probs(M, N):
    """P(slot of rank r is the argmax of a uniformly random N-subset of M slots), r = 0..M-1."""
    if N > M:
        return None
    den = comb(M, N)
    p = np.array([comb(M - 1 - r, N - 1) / den if M - 1 - r >= N - 1 else 0.0 for r in range(M)])
    assert abs(p.sum() - 1.0) < 1e-12, f"pick_probs({M},{N}) sums to {p.sum()}"
    return p


def one_seed(tag, lab, vsc, ref):
    """Per-item arrays for one generated pool, or a dict describing why it was refused."""
    pool = load_pool(tag, strict=False)
    if pool is None:
        return {"refused": "pool absent or short"}
    n = len(ref)
    miss_j = miss_v = 0
    Ms, sl_j, sl_e, sc, toks, modal_j, modal_e, nd, golds = [], [], [], [], [], [], [], [], []
    for it in ref:
        r = pool[(it["ds"], it["idx"])]
        preds = r["preds"]
        Ms.append(len(preds))
        lj, le, ss = [], list(r["oks_em"]), []
        for a in preds:
            y = lab.get((it["ds"], it["idx"], G.norm(a)))
            if y is None:
                miss_j += 1; y = 0
            lj.append(int(y))
            if (it["ds"], it["idx"], a) not in vsc:
                miss_v += 1
            ss.append(vsc.get((it["ds"], it["idx"], a), G.MISSING_SCORE))
        sl_j.append(lj); sl_e.append(le); sc.append(ss)
        toks.append(list(r.get("gen_tokens_all", [0] * len(preds))))
        mi = preds.index(r["modal_pred"]) if r["modal_pred"] in preds else 0
        modal_j.append(int(lab.get((it["ds"], it["idx"], G.norm(r["modal_pred"])), 0)))
        modal_e.append(int(le[mi]))
        nd.append(len(set(G.norm(a) for a in preds)))
        golds.append(r.get("gold", ""))
    if miss_j or miss_v:
        return {"refused": f"{miss_j} unjudged / {miss_v} unscored slots"}

    # exact E[correct | N] and the pick distribution, per item
    NS = [1, 2, 4, 8]
    got = {c: {N: np.zeros(n) for N in NS} for c in ("judge", "em")}
    orc = {c: {N: np.zeros(n) for N in NS} for c in ("judge", "em")}
    tok_pick = np.zeros(n); pick8 = np.zeros(n, int)
    for i in range(n):
        M = Ms[i]
        order = sorted(range(M), key=lambda k: (-sc[i][k], k))       # frozen first-index tie-break
        lj = np.array([sl_j[i][k] for k in order], float)
        le = np.array([sl_e[i][k] for k in order], float)
        tk = np.array([toks[i][k] for k in order], float)
        kj, ke = int(sum(sl_j[i])), int(sum(sl_e[i]))
        for N in NS:
            Ne = min(N, M)                       # a pool of M draws cannot be subsampled beyond M
            p = pick_probs(M, Ne)
            got["judge"][N][i] = float(p @ lj); got["em"][N][i] = float(p @ le)
            orc["judge"][N][i] = 1.0 - (comb(M - kj, Ne) / comb(M, Ne) if M - kj >= Ne else 0.0)
            orc["em"][N][i] = 1.0 - (comb(M - ke, Ne) / comb(M, Ne) if M - ke >= Ne else 0.0)
        pick8[i] = order[0]
        tok_pick[i] = tk[0]

    ds_index = np.array([DS.index(it["ds"]) for it in ref], int)
    rec_j = np.array([1 if 1 in x else 0 for x in sl_j], int)
    rec_e = np.array([1 if 1 in x else 0 for x in sl_e], int)
    out = {"got": got, "oracle_n": orc, "rec_judge": rec_j, "rec_em": rec_e,
           "modal_judge": np.array(modal_j, float), "modal_em": np.array(modal_e, float),
           "n_distinct": np.array(nd, float), "M": np.array(Ms, int),
           "tok_all_mean": float(np.mean([t for tt in toks for t in tt])),
           "tok_pick": tok_pick, "pick8": pick8, "ds_index": ds_index,
           "golds": golds, "preds_modal_slot": None}
    # cross-check against the frozen metric: N=8 expectation must equal the frozen argmax pick
    fr = G.sel_eff({(ref[i]["ds"], ref[i]["idx"]): sc[i] for i in range(n)},
                   items=[{"ds": ref[i]["ds"], "idx": ref[i]["idx"], "sl": sl_j[i],
                           "preds": pool[(ref[i]["ds"], ref[i]["idx"])]["preds"],
                           "greedy_ok": modal_j[i]} for i in range(n)])
    out["frozen"] = {"selected": fr["acc"], "oracle": fr["oracle"], "sel_eff": fr["sel_eff"],
                     "greedy": fr["greedy"], "contested_n": fr["contested"]["n"],
                     "contested_sel_eff": fr["contested"]["sel_eff"],
                     "per_ds": {d: dict(fr["per_ds"][d]) for d in DS},
                     "identity_residual": abs(fr["acc"] - fr["oracle"] * fr["sel_eff"])}
    out["max_abs_dev_vs_frozen_pick"] = float(np.max(np.abs(got["judge"][8] - fr["got"])))
    out["contested_mask"] = fr["contested_mask"]
    return out


# ------------------------------------------------------------------ main
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/cascade_methods/artifacts/decoding_ladder_cold_2026-08-14.json")
ap.add_argument("--control", default=CONTROL,
                help="the matched control. DEFAULT IS THE IN-SESSION REGENERATION T07r AND SHOULD STAY "
                     "THAT WAY for any reported number; the flag exists only so the code path can be "
                     "smoke-tested against the previous round's pools.")
A = ap.parse_args()
CONTROL = A.control

print("NULL TEST 1 ...", flush=True)
nt = G.null_test()
r0 = G.sel_eff(G.incumbent_scores())
print(f"  pass={nt['pass']} max_abs_deviation={nt['max_abs_deviation']:.3e} "
      f"identity_residual={abs(r0['acc']-r0['oracle']*r0['sel_eff']):.3e}", flush=True)

lab, vsc, ref = load_judge(), load_vscores(), G.load_items()
LAT = None      # built from the CONTROL pool's gold answers once the pools are loaded -- the frozen
                # transfer-dump items carry no gold field.

def seed_tags(setting):
    return [f"{setting}_s{s}" for s in range(3)
            if os.path.exists(os.path.join(SWEEP, f"ckpt_{DS[0]}_{setting}_s{s}.jsonl"))]

DATA, REFUSED = {}, {}
for tag, temp, prov in LADDER + [(t, v, "prior_round_NOT_regenerated") for t, v in PRIOR]:
    seeds = {}
    for st in seed_tags(tag):
        r = one_seed(st, lab, vsc, ref)
        if "refused" in r:
            REFUSED[st] = r["refused"]; print(f"  [refused] {st}: {r['refused']}", flush=True)
            continue
        seeds[st] = r
    if seeds:
        DATA[tag] = seeds
        print(f"  [ok] {tag}: {len(seeds)} seeds", flush=True)

if CONTROL not in DATA:
    sys.exit(f"CONTROL {CONTROL} not available -- refusing to report deltas against anything else")

# laterality stratum: the project's own regex on the GOLD answer, read from the control pool
try:
    from src.training_methods.visverif_lib import LATERAL
    _golds = DATA[CONTROL][sorted(DATA[CONTROL])[0]]["golds"]
    LAT = np.array([bool(LATERAL.search(str(g))) for g in _golds], bool)
    print(f"  laterality stratum: {int(LAT.sum())}/{len(LAT)} items", flush=True)
except Exception as e:
    print("  !! laterality mask unavailable:", e)

TEMP = dict([(t, v) for t, v, _ in LADDER] + PRIOR)
PROV = dict([(t, p) for t, _, p in LADDER] + [(t, "prior_round_NOT_regenerated") for t, _ in PRIOR])


def avg(seeds, key, sub=None, N=None):
    if sub is None:
        return np.mean([v[key] for v in seeds.values()], axis=0)
    return np.mean([v[key][sub][N] for v in seeds.values()], axis=0)


out = {"title": "COLD extension of the decoding temperature ladder -- where the peak is, "
                "where the verifier goes inert, and what that costs",
       "date": "2026-08-14",
       "prereg": "results/cascade_methods/artifacts/_decoding_ladder_cold_prereg.json "
                 "(written 2026-08-14T07:14:20Z, before any rung existed)",
       "generator": "src/cascade_methods/decoding_sweep_gen.py (unchanged) over "
                    "artifacts/_decoding_ladder_cold_settings{,_T00}.json",
       "analysis": "src/cascade_methods/decoding_ladder_cold.py",
       "endpoint": "SELECTED accuracy on the frozen 2,345-item open pool (slake_open 645 / "
                   "vqa_rad_open 200 / pathvqa_open 1500), frozen incumbent LoRA verifier "
                   "(ckpts/train/lora_verifier_disjoint) scored under HF transformers, batch 1.",
       "control": f"{CONTROL} = T 0.7, the deployed setting, REGENERATED IN THIS SESSION. Every delta "
                  "is against it, never against a stored number.",
       "held_fixed": {"prompt": "SYS verbatim from src/labeling/run_openvqa.py", "image_cap": "cap320 "
                      "(max_pixels 250880)", "N": 8, "max_tokens": 64, "top_p": 1.0, "top_k": -1,
                      "min_p": 0.0, "repetition_penalty": 1.0,
                      "generation_seeds": [20260813, 20261813, 20262813]},
       "NULL_TEST_1": {"what": "the frozen metric (src/training_methods/genframe_data.py) reproduces "
                               "every published incumbent cell",
                       "pass": bool(nt["pass"]), "max_abs_deviation": nt["max_abs_deviation"],
                       "measured": nt["measured"],
                       "identity_residual": abs(r0["acc"] - r0["oracle"] * r0["sel_eff"])},
       "NULL_TEST_2": {"what": "this round's cached HF verifier scores reproduce the FROZEN incumbent "
                               "transfer dumps, slot for slot",
                       "slots_compared": 15312, "max_abs_score_deviation": 0.0,
                       "argmax_agreement": "1273/1273 = 1.000000",
                       "note": "computed from ckpts/openvqa/decoding_sweep/vscore_cache_shard*.jsonl "
                               "against ckpts/train/lora_verifier_disjoint/transfer_dump_*.json; the "
                               "3,448 slots not compared are simply ones no sweep round ever queued."},
       "NULL_TEST_3": {"what": "THIS script, run with --control T07, reproduces the 2026-08-13 sweep's "
                               "published deltas from the same pools -- so the new analysis code is not "
                               "the thing that moved any number",
                       "artifact": "artifacts/_decoding_ladder_cold_nulltest3_prior_round.json",
                       "reproduced": "see that file; T03 judge +0.00540 / em +0.01279, T05 judge "
                                     "+0.00569 [+0.00100,+0.01038] / em +0.00796, T10 judge -0.01649, "
                                     "T13 judge -0.04563 -- identical to "
                                     "artifacts/decoding_sweep_2026-08-13.json"},
       "nboot": NBOOT, "bootstrap_seed": BSEED,
       "pools_refused": REFUSED,
       "LADDER": {}}

for tag, seeds in DATA.items():
    gj8 = avg(seeds, "got", "judge", 8); ge8 = avg(seeds, "got", "em", 8)
    gj1 = avg(seeds, "got", "judge", 1); ge1 = avg(seeds, "got", "em", 1)
    mj = avg(seeds, "modal_judge"); me = avg(seeds, "modal_em")
    rj = avg(seeds, "rec_judge"); re_ = avg(seeds, "rec_em")
    b = {"temperature": TEMP[tag], "provenance": PROV[tag], "n_seeds": len(seeds),
         "seeds": sorted(seeds),
         "pool_size_M_mean": float(np.mean([v["M"].mean() for v in seeds.values()])),
         "distinct_of_8": float(np.mean([v["n_distinct"].mean() for v in seeds.values()])),
         "frac_items_1_distinct_UNANIMOUS": float(np.mean([(v["n_distinct"] == 1).mean()
                                                           for v in seeds.values()])),
         "mean_gen_tokens_all_slots": float(np.mean([v["tok_all_mean"] for v in seeds.values()])),
         "mean_gen_tokens_PICKED_slot": float(np.mean([v["tok_pick"].mean() for v in seeds.values()])),
         "oracle@8_judge": float(np.mean([v["frozen"]["oracle"] for v in seeds.values()])),
         "oracle@8_em": float(np.mean([v["rec_em"].mean() for v in seeds.values()])),
         "sel_eff_judge": float(np.mean([v["frozen"]["sel_eff"] for v in seeds.values()])),
         "sel_eff_em": float(np.mean([v["got"]["em"][8].sum() / max(v["rec_em"].sum(), 1)
                                      for v in seeds.values()])),
         "SELECTED_judge": float(gj8.mean()), "SELECTED_em": float(ge8.mean()),
         "SELECTED_judge_seed_sd": float(np.std([v["got"]["judge"][8].mean() for v in seeds.values()],
                                                ddof=1)) if len(seeds) > 1 else 0.0,
         "SELECTED_em_seed_sd": float(np.std([v["got"]["em"][8].mean() for v in seeds.values()],
                                             ddof=1)) if len(seeds) > 1 else 0.0,
         "MODAL_VOTE_judge": float(mj.mean()), "MODAL_VOTE_em": float(me.mean()),
         "RANDOM_SLOT_judge": float(gj1.mean()), "RANDOM_SLOT_em": float(ge1.mean()),
         "identity_residual_selected_eq_oracle_x_sel_eff":
             float(np.mean([v["frozen"]["identity_residual"] for v in seeds.values()])),
         "max_abs_dev_exact_N8_expectation_vs_frozen_argmax_pick":
             float(np.max([v["max_abs_dev_vs_frozen_pick"] for v in seeds.values()])),
         "per_cell": {}, }
    for j, d in enumerate(DS):
        m = seeds[sorted(seeds)[0]]["ds_index"] == j
        b["per_cell"][d] = {"n": int(m.sum()),
                            "SELECTED_judge": float(gj8[m].mean()), "SELECTED_em": float(ge8[m].mean()),
                            "oracle@8_judge": float(rj[m].mean()),
                            "sel_eff_judge": float(np.mean([v["frozen"]["per_ds"][d]["sel_eff"]
                                                            for v in seeds.values()])),
                            "SELECTED_judge_seed_sd": float(np.std(
                                [v["got"]["judge"][8][m].mean() for v in seeds.values()], ddof=1))
                                if len(seeds) > 1 else 0.0}
    b["SELECTED_at_N"] = {c: {str(N): float(avg(seeds, "got", c, N).mean()) for N in (1, 2, 4, 8)}
                          for c in ("judge", "em")}
    b["oracle_at_N"] = {c: {str(N): float(avg(seeds, "oracle_n", c, N).mean()) for N in (1, 2, 4, 8)}
                        for c in ("judge", "em")}
    out["LADDER"][tag] = b

# ---------------------------------------------------------------- deltas vs the in-session control
C = DATA[CONTROL]
cj = avg(C, "got", "judge", 8); ce = avg(C, "got", "em", 8)
con_mask = np.mean([v["contested_mask"] for v in C.values()], axis=0) >= 0.5
out["contested_stratum_definition"] = ("recoverable under the CONTROL pool AND >=2 distinct candidate "
                                       "strings in the CONTROL pool -- fixed by the control so the "
                                       "same items are compared across rungs")
out["contested_n"] = int(con_mask.sum())
out["laterality_n"] = int(LAT.sum()) if LAT is not None else None
out["DELTAS_VS_CONTROL"] = {}
for tag, seeds in DATA.items():
    if tag == CONTROL:
        continue
    sj = avg(seeds, "got", "judge", 8); se = avg(seeds, "got", "em", 8)
    dj, de = boot(sj, cj, nboot=NBOOT, seed=BSEED), boot(se, ce, nboot=NBOOT, seed=BSEED)
    row = {"temperature": TEMP[tag], "provenance": PROV[tag],
           "judge": dj, "em": de,
           "CI_clean_in_BOTH": bool(dj["verdict"] == de["verdict"] and dj["sig"] and de["sig"]),
           "currency_conflict": bool((dj["verdict"] == "WIN" and de["verdict"] == "LOSS")
                                     or (dj["verdict"] == "LOSS" and de["verdict"] == "WIN")),
           "contested": {"judge": boot(sj, cj, mask=con_mask, nboot=NBOOT, seed=BSEED),
                         "em": boot(se, ce, mask=con_mask, nboot=NBOOT, seed=BSEED)},
           "per_cell_guardrail": {}}
    if LAT is not None and LAT.sum() > 0:
        row["laterality"] = {"judge": boot(sj, cj, mask=LAT, nboot=NBOOT, seed=BSEED),
                             "em": boot(se, ce, mask=LAT, nboot=NBOOT, seed=BSEED)}
    dsi = seeds[sorted(seeds)[0]]["ds_index"]
    for j, d in enumerate(DS):
        m = dsi == j
        row["per_cell_guardrail"][d] = {"judge": boot(sj, cj, mask=m, nboot=NBOOT, seed=BSEED),
                                        "em": boot(se, ce, mask=m, nboot=NBOOT, seed=BSEED)}
    row["guardrail_flags"] = [d for d in DS if row["per_cell_guardrail"][d]["judge"]["verdict"] == "LOSS"
                              or row["per_cell_guardrail"][d]["em"]["verdict"] == "LOSS"]
    out["DELTAS_VS_CONTROL"][tag] = row

# ---------------------------------------------------------------- the collapse point
out["COLLAPSE_POINT"] = {
    "definition_primary": "VC_single(T) = SELECTED(T) - RANDOM_SLOT(T): what best-of-8 + the frozen "
                          "verifier buys over ONE draw from the SAME pool at the SAME temperature "
                          "(exact expectation, not a sampled draw). Pre-registered as the primary "
                          "statistic before any rung was generated.",
    "definition_secondary": "VC_modal(T) = SELECTED(T) - MODAL_VOTE(T): the harder bar, against "
                            "majority vote over the same pool, which is itself a training-free selector.",
    "per_rung": {}}
for tag, seeds in DATA.items():
    sj8 = avg(seeds, "got", "judge", 8); se8 = avg(seeds, "got", "em", 8)
    sj1 = avg(seeds, "got", "judge", 1); se1 = avg(seeds, "got", "em", 1)
    mj = avg(seeds, "modal_judge"); me = avg(seeds, "modal_em")
    out["COLLAPSE_POINT"]["per_rung"][tag] = {
        "temperature": TEMP[tag],
        "distinct_of_8": out["LADDER"][tag]["distinct_of_8"],
        "frac_items_1_distinct": out["LADDER"][tag]["frac_items_1_distinct_UNANIMOUS"],
        "VC_single_judge": boot(sj8, sj1, nboot=NBOOT, seed=BSEED),
        "VC_single_em": boot(se8, se1, nboot=NBOOT, seed=BSEED),
        "VC_modal_judge": boot(sj8, mj, nboot=NBOOT, seed=BSEED),
        "VC_modal_em": boot(se8, me, nboot=NBOOT, seed=BSEED)}

# ---------------------------------------------------------------- N interaction vs the deployed arm
# FLOP model and per-candidate geometry are NOT re-derived here: they are read verbatim from
# artifacts/resolution_sweep_2026-08-13.json ["cost"]["open_half_per_candidate"]["cap320"], which
# measured them for this exact cap and this exact verifier configuration.
RES = os.path.join(ROOT, "results/cascade_methods/artifacts/resolution_sweep_2026-08-13.json")
COST = None
if os.path.exists(RES):
    c = json.load(open(RES))["cost"]["open_half_per_candidate"]["cap320"]
    dec = c["parts"]["lm_decode_dense"] + c["parts"]["lm_decode_attn"] + c["parts"]["lm_head"]
    COST = {"source": "artifacts/resolution_sweep_2026-08-13.json cost.open_half_per_candidate.cap320",
            "gen_fixed_per_candidate_flops": c["flops_per_candidate"] - dec,
            "gen_decode_flops_per_token": dec / c["measured_mean_gen_tokens"],
            "verifier_flops_per_candidate": c["flops_verifier_per_candidate_at_1003520"],
            "deployed_arm_8gen_plus_8verify_flops": c["flops_arm_8gen_plus_8verify"],
            "note": "decode is only %.2f%% of a generator candidate's FLOPs at cap320, so the cost of "
                    "an N-sample arm is N/8 of the deployed arm to within ~1%%; temperature moves only "
                    "the decode term." % (100.0 * dec / c["flops_per_candidate"])}


def arm_flops(tag, N):
    if COST is None:
        return None
    M = out["LADDER"][tag]["pool_size_M_mean"]
    Ne = min(N, M)
    tok = out["LADDER"][tag]["mean_gen_tokens_all_slots"]
    gen = COST["gen_fixed_per_candidate_flops"] + COST["gen_decode_flops_per_token"] * tok
    return Ne * (gen + COST["verifier_flops_per_candidate"])


cn8j, cn8e = float(cj.mean()), float(ce.mean())
DEPLOYED_FLOPS = arm_flops(CONTROL, 8)
out["N_INTERACTION"] = {
    "question": "does a COLD pool at N=4 (or 2) match the DEPLOYED hot pool at N=8? If so the arm cost "
                "falls by ~N/8 at parity, and the project's current objective is MINIMUM COST AT PARITY.",
    "method": "EXACT subsampling of the already-generated 8-sample pools: given N of the M slots, the "
              "verifier's pick is deterministic, so E[correct] is a closed-form weighted sum over the "
              "slot ranks (probability C(M-1-r, N-1)/C(M,N) for the slot of rank r). No new generation, "
              "no Monte Carlo. Both the generator AND the verifier are charged, because N samples need "
              "N verifier passes.",
    "cost_model": COST,
    "deployed_reference": {"setting": CONTROL, "N": 8, "SELECTED_judge": cn8j, "SELECTED_em": cn8e,
                           "flops": DEPLOYED_FLOPS},
    "rows": []}
for tag, seeds in DATA.items():
    for N in (1, 2, 4, 8):
        sj = avg(seeds, "got", "judge", N); se = avg(seeds, "got", "em", N)
        dj, de = boot(sj, cj, nboot=NBOOT, seed=BSEED), boot(se, ce, nboot=NBOOT, seed=BSEED)
        f = arm_flops(tag, N)
        out["N_INTERACTION"]["rows"].append({
            "setting": tag, "temperature": TEMP[tag], "N": N,
            "N_effective": float(min(N, out["LADDER"][tag]["pool_size_M_mean"])),
            "SELECTED_judge": float(sj.mean()), "SELECTED_em": float(se.mean()),
            "oracle_judge": float(avg(seeds, "oracle_n", "judge", N).mean()),
            "vs_deployed_N8_judge": dj, "vs_deployed_N8_em": de,
            "matches_deployed_in_BOTH": bool(dj["verdict"] != "LOSS" and de["verdict"] != "LOSS"),
            "PARITY_OR_BETTER_in_BOTH": bool(dj["verdict"] != "LOSS" and de["verdict"] != "LOSS"),
            "flops_eq_vs_deployed_N8": (f / DEPLOYED_FLOPS) if f and DEPLOYED_FLOPS else None,
            "gen_token_ratio_vs_deployed_N8": float(
                out["LADDER"][tag]["mean_gen_tokens_all_slots"] * min(N, out["LADDER"][tag]["pool_size_M_mean"])
                / (out["LADDER"][CONTROL]["mean_gen_tokens_all_slots"] * 8))})

# ---------------------------------------------------------------- reproducibility / determinism
def raw_preds(tag):
    p = load_pool(tag, strict=False)
    return None if p is None else {k: v["preds"] for k, v in p.items()}


def identity_rate(a, b):
    ks = sorted(set(a) & set(b))
    if not ks:
        return None
    same = sum(1 for k in ks if a[k] == b[k])
    slots = [(x, y) for k in ks for x, y in zip(a[k], b[k])]
    return {"n_items_compared": len(ks), "items_byte_identical": same,
            "item_identity_rate": same / len(ks),
            "slot_identity_rate": sum(1 for x, y in slots if x == y) / max(len(slots), 1)}


out["REPRODUCIBILITY"] = {}
g = {s: raw_preds(f"T00_s{s}") for s in range(3)}
if all(v is not None for v in g.values()):
    out["REPRODUCIBILITY"]["T00_greedy_determinism_across_seeds"] = {
        "why": "vLLM refuses n>1 at temperature 0, so T=0 is generated at n=1 three times with three "
               "different SamplingParams seeds. If the three agree byte-for-byte, argmax decoding is "
               "deterministic here and the 8-sample pool at T=0 provably holds exactly 1 distinct "
               "candidate -- which is what makes the verifier inert, and it is MEASURED, not assumed.",
        "s0_vs_s1": identity_rate(g[0], g[1]), "s0_vs_s2": identity_rate(g[0], g[2]),
        "s1_vs_s2": identity_rate(g[1], g[2])}
for a, b in (("T07", "T07r"), ("T03", "T03r"), ("T05", "T05r")):
    pa, pb = raw_preds(f"{a}_s0"), raw_preds(f"{b}_s0")
    if pa and pb:
        r = identity_rate(pa, pb)
        blk = {"why": f"{a} was generated in the 2026-08-13/14 round and {b} in this session, with the "
                      "SAME code, SAME sampling seed and SAME cap. Any difference is engine/serving "
                      "nondeterminism, which is exactly the +-0.008 open-text reproducibility caveat.",
               "string_identity_seed0": r}
        if a in DATA and b in DATA:
            sa = avg(DATA[a], "got", "judge", 8); sb = avg(DATA[b], "got", "judge", 8)
            ea = avg(DATA[a], "got", "em", 8); eb = avg(DATA[b], "got", "em", 8)
            blk["SELECTED_judge_old_minus_new"] = boot(sa, sb, nboot=NBOOT, seed=BSEED)
            blk["SELECTED_em_old_minus_new"] = boot(ea, eb, nboot=NBOOT, seed=BSEED)
        out["REPRODUCIBILITY"][f"{a}_vs_{b}"] = blk

# ---------------------------------------------------------------- currency mechanism audit
out["CURRENCY_AUDIT"] = {"rule": "pre-registered H6: if the two currencies disagree about WHICH rung is "
                                 "the peak, the tie-break is mechanistic -- a judge-favoured rung whose "
                                 "PICKED answers are systematically longer is harvesting judge leniency; "
                                 "if picked-slot length is matched within +-1 token the disagreement is "
                                 "declared noise and the peak is reported as an interval.",
                         "per_rung": {}}
for tag, seeds in DATA.items():
    sj = avg(seeds, "got", "judge", 8); se = avg(seeds, "got", "em", 8)
    out["CURRENCY_AUDIT"]["per_rung"][tag] = {
        "temperature": TEMP[tag],
        "mean_gen_tokens_PICKED_slot": out["LADDER"][tag]["mean_gen_tokens_PICKED_slot"],
        "judge_minus_EM_on_picked_slot": float(sj.mean() - se.mean()),
        "disagreement_rate_picked_slot": float(np.mean(np.abs(sj - se)))}

# ---------------------------------------------------------------- peak, plateau, collapse, verdict
IN_SESSION = [t for t, _, _ in LADDER if t in DATA]
if not IN_SESSION:                      # smoke-test path only (--control on a prior-round pool)
    IN_SESSION = sorted(DATA, key=lambda t: TEMP[t])
    print("  !! no in-session rung available; PEAK/COLLAPSE computed on prior-round pools "
          "(SMOKE TEST, not reportable)")
out["PEAK"] = {"in_session_rungs": IN_SESSION}
pj = max(IN_SESSION, key=lambda t: out["LADDER"][t]["SELECTED_judge"])
pe = max(IN_SESSION, key=lambda t: out["LADDER"][t]["SELECTED_em"])
out["PEAK"]["argmax_judge"] = {"rung": pj, "T": TEMP[pj], "SELECTED": out["LADDER"][pj]["SELECTED_judge"]}
out["PEAK"]["argmax_em"] = {"rung": pe, "T": TEMP[pe], "SELECTED": out["LADDER"][pe]["SELECTED_em"]}
out["PEAK"]["pairwise_between_rungs"] = {}
for i, a in enumerate(IN_SESSION):
    for b in IN_SESSION[i + 1:]:
        aj, bj = avg(DATA[a], "got", "judge", 8), avg(DATA[b], "got", "judge", 8)
        ae, be = avg(DATA[a], "got", "em", 8), avg(DATA[b], "got", "em", 8)
        out["PEAK"]["pairwise_between_rungs"][f"{a}_minus_{b}"] = {
            "judge": boot(aj, bj, nboot=NBOOT, seed=BSEED), "em": boot(ae, be, nboot=NBOOT, seed=BSEED)}
# the plateau = every rung not CI-cleanly worse than the best rung, in EITHER currency
best = pj
plateau = [best]
for t in IN_SESSION:
    if t == best:
        continue
    k = f"{best}_minus_{t}" if f"{best}_minus_{t}" in out["PEAK"]["pairwise_between_rungs"] \
        else f"{t}_minus_{best}"
    r = out["PEAK"]["pairwise_between_rungs"][k]
    if r["judge"]["verdict"] == "TIE" and r["em"]["verdict"] == "TIE":
        plateau.append(t)
out["PEAK"]["plateau_indistinguishable_from_best_in_BOTH_currencies"] = sorted(
    plateau, key=lambda t: TEMP[t])

cp = out["COLLAPSE_POINT"]["per_rung"]
sig = {t: (cp[t]["VC_single_judge"]["verdict"] == "WIN", cp[t]["VC_single_em"]["verdict"] == "WIN")
       for t in IN_SESSION}
cold_first = sorted(IN_SESSION, key=lambda t: TEMP[t])
def bracket(ci):
    dead = [t for t in cold_first if not sig[t][ci]]
    live = [t for t in cold_first if sig[t][ci]]
    return {"coldest_rung_where_the_verifier_still_beats_a_single_draw":
                (min(live, key=lambda t: TEMP[t]) if live else None),
            "coldest_T_still_CI_clean": (TEMP[min(live, key=lambda t: TEMP[t])] if live else None),
            "warmest_rung_where_it_does_NOT": (max(dead, key=lambda t: TEMP[t]) if dead else None),
            "warmest_T_not_CI_clean": (TEMP[max(dead, key=lambda t: TEMP[t])] if dead else None),
            "rungs_where_contribution_is_indistinguishable_from_zero": [t for t in cold_first
                                                                        if not sig[t][ci]]}
out["COLLAPSE_POINT"]["bracket_judge"] = bracket(0)
out["COLLAPSE_POINT"]["bracket_em"] = bracket(1)

os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, A.out), "w"), indent=1, default=float)
print(f"\nwrote {A.out}")

# ---------------------------------------------------------------- console table
print(f"\n{'rung':7s} {'T':>5s} {'sd':>4s} {'dist/8':>7s} {'unan':>6s} {'orc_j':>7s} {'seff_j':>7s} "
      f"{'SEL_j':>8s} {'SEL_em':>8s} {'modal_j':>8s} {'rand_j':>7s} {'tok':>6s}")
for tag in sorted(out["LADDER"], key=lambda t: (TEMP[t], t)):
    b = out["LADDER"][tag]
    print(f"{tag:7s} {b['temperature']:5.2f} {b['n_seeds']:4d} {b['distinct_of_8']:7.3f} "
          f"{b['frac_items_1_distinct_UNANIMOUS']:6.3f} {b['oracle@8_judge']:7.4f} "
          f"{b['sel_eff_judge']:7.4f} {b['SELECTED_judge']:8.5f} {b['SELECTED_em']:8.5f} "
          f"{b['MODAL_VOTE_judge']:8.5f} {b['RANDOM_SLOT_judge']:7.4f} "
          f"{b['mean_gen_tokens_all_slots']:6.2f}")
print(f"\nvs in-session control {CONTROL}:")
for tag in sorted(out["DELTAS_VS_CONTROL"], key=lambda t: (TEMP[t], t)):
    r = out["DELTAS_VS_CONTROL"][tag]
    j, e = r["judge"], r["em"]
    print(f"  {tag:7s} T={r['temperature']:4.2f}  judge {j['delta']:+.5f} [{j['lo']:+.5f},{j['hi']:+.5f}] "
          f"{j['verdict']:5s} | em {e['delta']:+.5f} [{e['lo']:+.5f},{e['hi']:+.5f}] {e['verdict']:5s}"
          + ("  <<< CURRENCY CONFLICT" if r["currency_conflict"] else "")
          + (f"  guardrail:{r['guardrail_flags']}" if r["guardrail_flags"] else ""))
print("\nCOLLAPSE (SELECTED - one draw from the same pool):")
for tag in sorted(out["COLLAPSE_POINT"]["per_rung"], key=lambda t: (TEMP[t], t)):
    r = out["COLLAPSE_POINT"]["per_rung"][tag]
    a, b_ = r["VC_single_judge"], r["VC_single_em"]
    print(f"  {tag:7s} T={r['temperature']:4.2f} dist={r['distinct_of_8']:5.2f} "
          f"judge {a['delta']:+.5f} [{a['lo']:+.5f},{a['hi']:+.5f}] {a['verdict']:5s} | "
          f"em {b_['delta']:+.5f} [{b_['lo']:+.5f},{b_['hi']:+.5f}] {b_['verdict']:5s}")
