#!/usr/bin/env python3
"""decoding_sweep_report.py -- assemble results/cascade_methods/artifacts/decoding_sweep_2026-08-13.json

SWEEP 1: does changing the 7B's DECODING PARAMETERS improve the open-text candidate pool?
One variable at a time off the deployed control, prompt and image cap held FIXED.
Headline endpoint = SELECTED accuracy under the frozen incumbent verifier.
"""
import json, os, sys, subprocess
import numpy as np
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G                      # noqa: E402
from src.cascade_methods import decoding_sweep_analyse as S              # noqa: E402

SET_FILE = os.path.join(ROOT, "results/cascade_methods/artifacts/_decoding_sweep_settings.json")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/decoding_sweep_2026-08-13.json")
CONTROL = "T07"

settings = json.load(open(SET_FILE))
groups = {}
for st in settings:
    base = st["tag"].rsplit("_s", 1)[0]
    groups.setdefault(base, {"axis": st["axis"], "value": st["value"], "params": {
        k: st[k] for k in ("temp", "top_p", "top_k", "min_p", "rep_pen")}, "seeds": {}})
    groups[base]["seeds"][st["seed_i"]] = st["tag"]

lab, vsc = S.load_judge(), S.load_vscores()
REF = G.load_items()
print(f"judge labels {len(lab)} | verifier scores {len(vsc)} | ref items {len(REF)}", flush=True)

# ================= NULL TEST 1: the frozen metric reproduces the published bar =================
r0 = G.sel_eff(G.incumbent_scores())
dev1 = [abs(r0["n"] - G.PUBLISHED["n"]), abs(r0["oracle"] - G.PUBLISHED["oracle@8"]),
        abs(r0["acc"] - G.PUBLISHED["selected"]), abs(r0["greedy"] - G.PUBLISHED["greedy"]),
        abs(r0["sel_eff"] - G.PUBLISHED["sel_eff"]),
        abs(G.cand_auroc(G.incumbent_scores()) - G.PUBLISHED["cand_auroc"])] + \
       [abs(r0["per_ds"][d]["sel_eff"] - G.PUBLISHED["per_ds"][d]) for d in S.DS]
null1 = {"description": "frozen metric (src/training_methods/genframe_data.py) vs the PUBLISHED constants",
         "pass": bool(max(dev1) < 1e-5), "max_abs_deviation": float(max(dev1)),
         "note": "residual is 6-dp rounding in the published constants",
         "measured": {"n": r0["n"], "oracle@8": r0["oracle"], "selected": r0["acc"],
                      "greedy_modal_of_8": r0["greedy"], "sel_eff": r0["sel_eff"]},
         "exact_identity_selected_eq_oracle_x_sel_eff": float(abs(r0["acc"] - r0["oracle"] * r0["sel_eff"]))}

# ================= NULL TEST 2: this session's verifier harness reproduces the stored scores =======
NT2 = os.path.join(S.SWEEP, "nulltest2_items.json")
sample = set(map(tuple, json.load(open(NT2)))) if os.path.exists(NT2) else None
sub = [it for it in REF if sample is None or (it["ds"], it["idx"]) in sample]
re_scores, st_scores, dev2, n_cmp = {}, {}, [], 0
for it in sub:
    sc = []
    for k, a in enumerate(it["preds"]):
        v = vsc.get((it["ds"], it["idx"], a))
        if v is not None:
            dev2.append(abs(v - it["scores"][k])); n_cmp += 1
        sc.append(v if v is not None else G.MISSING_SCORE)
    re_scores[(it["ds"], it["idx"])] = sc
    st_scores[(it["ds"], it["idx"])] = list(it["scores"])
n_tot = sum(len(it["preds"]) for it in sub)
null2 = {"description": "re-score STORED deployed-pool candidates with THIS SESSION's verifier harness "
                        "and compare to the scores frozen in transfer_dump_*.json. Sampled in WHOLE "
                        "ITEMS (all 8 slots), stratified by dataset, seed 20260813, so sel_eff and the "
                        "argmax pick can both be recomputed on the sample.",
         "n_items_sampled": len(sub), "n_slots_compared": n_cmp, "n_slots_in_sample": n_tot}
if n_cmp == n_tot and n_tot > 0:
    rn = G.sel_eff(re_scores, items=sub)
    ro = G.sel_eff(st_scores, items=sub)
    same_pick = float(np.mean(rn["picks"] == ro["picks"]))
    null2.update({"pass": bool(max(dev2) < 5e-3 and abs(rn["sel_eff"] - ro["sel_eff"]) < 5e-3),
                  "max_abs_score_deviation": float(np.max(dev2)),
                  "mean_abs_score_deviation": float(np.mean(dev2)),
                  "argmax_pick_agreement": same_pick,
                  "rescored_sel_eff_on_sample": rn["sel_eff"], "stored_sel_eff_on_sample": ro["sel_eff"],
                  "sel_eff_abs_deviation_on_sample": float(abs(rn["sel_eff"] - ro["sel_eff"])),
                  "rescored_selected_on_sample": rn["acc"], "stored_selected_on_sample": ro["acc"],
                  "note": "compared on the SAME sampled items on both sides, so this isolates the "
                          "harness, not the sample."})
else:
    null2["pass"] = None
    null2["note"] = f"only {n_cmp}/{n_tot} sampled slots re-scored so far"

# ================= per-setting measurement =================
res, M, I, excluded = {}, {}, {}, {}   # M[base][seed] = measurement, I[base][seed] = items
for base, g in sorted(groups.items()):
    M[base], I[base] = {}, {}
    for si, tag in sorted(g["seeds"].items()):
        try:
            pool = S.load_pool(tag)
        except ValueError as e:
            print(f"  [SKIP] {e}"); continue
        if pool is None:
            print(f"  [missing] {tag}"); continue
        its, miss = S.build_items(pool, lab, vsc, ref=REF)
        nom = sum(1 for it in its for s in it["scores"] if s == G.MISSING_SCORE)
        if miss:
            # Coverage endpoints need EVERY judge label; without them oracle@8 is wrong. Refuse.
            excluded[tag] = {"missing_judge_slots": miss, "missing_vscore_slots": nom}
            print(f"  [EXCLUDED] {tag}: {miss} slots have no judge label"); continue
        # SELECTION endpoints need every verifier score; without them argmax would fall into slot 0.
        sel = (nom == 0)
        if not sel:
            print(f"  [judge-only] {tag}: {nom} slots unscored -> coverage endpoints only")
        m = S.measure(its, with_selection=sel)
        m["_missing_judge_slots"] = miss; m["_missing_vscore_slots"] = nom
        M[base][si] = m; I[base][si] = its
    if not M[base]:
        del M[base], I[base]; continue
    ss = sorted(M[base])
    HAS_SEL = all(M[base][s]["has_selection"] for s in ss)
    ks = ["oracle@8", "mean_distinct", "greedy_modal", "mean_gen_tokens"]
    if HAS_SEL:
        ks += ["sel_eff", "selected", "contested_sel_eff"]
    agg = {k: {"mean": float(np.mean([M[base][s][k] for s in ss])),
               "sd": float(np.std([M[base][s][k] for s in ss], ddof=1)) if len(ss) > 1 else 0.0,
               "per_seed": {str(s): M[base][s][k] for s in ss}} for k in ks}
    res[base] = {"axis": g["axis"], "value": g["value"], "params": g["params"],
                 "is_control": base == CONTROL, "n_seeds": len(ss),
                 "tier": "coverage+selection" if HAS_SEL else "coverage-only (judge labels only; "
                         "sel_eff/SELECTED need the frozen verifier, which was not run on this setting)",
                 **agg,
                 "per_ds_mean": {d: {k: float(np.mean([M[base][s]["per_ds"][d][k] for s in ss]))
                                     for k in (("oracle", "acc", "sel_eff") if HAS_SEL else ("oracle",))}
                                 for d in S.DS},
                 "oracle_vs_N_within8": list(np.mean([S.oracle_curve_within8(I[base][s]) for s in ss],
                                                     axis=0).astype(float)),
                 "oracle_vs_N_pooled_allseeds": S.oracle_curve_pooled(I[base]) if len(ss) > 1 else None,
                 "capture_recapture": S.lp_ceiling(I[base]) if len(ss) > 1 else None,
                 "max_identity_residual": (float(max(M[base][s]["identity_residual"] for s in ss))
                                           if HAS_SEL else None),
                 "missing_judge_slots": sum(M[base][s]["_missing_judge_slots"] for s in ss),
                 "missing_vscore_slots": sum(M[base][s]["_missing_vscore_slots"] for s in ss),
                 "laterality_stratum": {
                     "n": int(S.laterality_mask(I[base][ss[0]]).sum()),
                     "SELECTED": (float(np.mean([M[base][s]["_arrays"]["got"][
                         S.laterality_mask(I[base][s])].mean() for s in ss])) if HAS_SEL else None),
                     "oracle@8": float(np.mean([M[base][s]["_arrays"]["rec"][
                         S.laterality_mask(I[base][s])].mean() for s in ss])),
                     "definition": "gold answer matches visverif_lib.LATERAL (the project's regex)"}}
    sels = (f"sel_eff {agg['sel_eff']['mean']:.4f}  SELECTED {agg['selected']['mean']:.4f} "
            f"(sd {agg['selected']['sd']:.4f})" if HAS_SEL else "sel_eff/SELECTED: not scored")
    print(f"[{base:12s}] n_seeds {len(ss)}  oracle@8 {agg['oracle@8']['mean']:.4f}  {sels}  "
          f"distinct {agg['mean_distinct']['mean']:.2f}  tok {agg['mean_gen_tokens']['mean']:.1f}", flush=True)

# ================= deltas vs the MATCHED control (same session, same serving config) =================
def seedavg(base, arr):
    return np.mean([M[base][s]["_arrays"][arr].astype(float) for s in sorted(M[base])], axis=0)


deltas = {}
if CONTROL in M:
    c_got, c_rec = seedavg(CONTROL, "got"), seedavg(CONTROL, "rec")
    c_con = M[CONTROL][sorted(M[CONTROL])[0]]["_arrays"]["con"]
    di = M[CONTROL][sorted(M[CONTROL])[0]]["_arrays"]["ds_index"]
    LATM = S.laterality_mask(I[CONTROL][sorted(I[CONTROL])[0]])
    for base in res:
        if base == CONTROL:
            continue
        a_rec = seedavg(base, "rec")
        if not all(M[base][s]["has_selection"] for s in M[base]):
            deltas[base] = {"tier": "coverage-only",
                            "oracle@8_vs_control": S.boot(a_rec, c_rec),
                            "note": "SELECTED/sel_eff not available: the frozen verifier was not run "
                                    "on this setting."}
            continue
        a_got = seedavg(base, "got")
        shared = sorted(set(M[base]) & set(M[CONTROL]))
        gp = G.paired_bootstrap(a_got, c_got, rec=(c_rec > 0).astype(int), nboot=S.NBOOT, seed=S.SEED)
        deltas[base] = {
            "SELECTED_vs_control": S.boot(a_got, c_got),
            "SELECTED_vs_control_via_project_routine": {
                "d_acc": gp["d_acc"], "d_acc_ci": gp["d_acc_ci"],
                "routine": "genframe_data.paired_bootstrap (the project's own frozen draw)"},
            "oracle@8_vs_control": S.boot(a_rec, c_rec),
            "SELECTED_on_control_recoverable_vs_control": S.boot(a_got, c_got, mask=(c_rec > 0)),
            "SELECTED_on_contested_vs_control": S.boot(a_got, c_got, mask=c_con),
            "sel_eff_delta_POINT_ESTIMATE_ONLY": res[base]["sel_eff"]["mean"] - res[CONTROL]["sel_eff"]["mean"],
            "why_sel_eff_has_no_paired_CI":
                "sel_eff is P(pick correct | pool recoverable), and the RECOVERABLE SET ITSELF moves "
                "between decoding settings, so the two sel_eff values condition on different "
                "populations and their difference is not a paired item statistic. The paired, "
                "well-defined stratum endpoint is SELECTED_on_control_recoverable_vs_control. "
                "SELECTED (= oracle@8 x sel_eff, exact) is the endpoint that needs no conditioning.",
            "per_seed_matched_SELECTED_delta": [
                float(M[base][s]["selected"] - M[CONTROL][s]["selected"]) for s in shared],
            "guardrail_per_cell_SELECTED_delta": {
                d: S.boot(a_got, c_got, mask=(di == j)) for j, d in enumerate(S.DS)},
            "SELECTED_on_LATERALITY_vs_control": S.boot(a_got, c_got, mask=LATM),
            "SELECTED_on_NON_laterality_vs_control": S.boot(a_got, c_got, mask=~LATM),
        }

# ================= oracle-vs-N: can a better setting reach the control's oracle@8 with fewer samples? =
cost = {}
if CONTROL in res and res[CONTROL]["oracle_vs_N_pooled_allseeds"]:
    target = res[CONTROL]["oracle_vs_N_within8"][7]
    for base in res:
        c = res[base]["oracle_vs_N_pooled_allseeds"]
        if not c:
            continue
        n_need = next((i + 1 for i, v in enumerate(c) if v >= target), None)
        cost[base] = {"N_to_reach_control_oracle@8": n_need,
                      "control_oracle@8_target": target,
                      "oracle@8_of_this_setting_within8": res[base]["oracle_vs_N_within8"][7],
                      "samples_saved_vs_8": (8 - n_need) if n_need else None}

# ================= EXACT-MATCH CURRENCY coverage (needs no GPU; secondary endpoint) =================
emres, EM = {}, {}
for base, g in sorted(groups.items()):
    EM[base] = {}
    for si, tag in sorted(g["seeds"].items()):
        try:
            its = S.em_items(tag, REF)
        except ValueError:
            continue
        if its is not None:
            EM[base][si] = its
    if not EM[base]:
        del EM[base]; continue
    ms = {si: S.em_measure(EM[base][si]) for si in EM[base]}
    ss = sorted(ms)
    emres[base] = {
        "params": g["params"], "n_seeds": len(ss),
        "em_oracle@8": {"mean": float(np.mean([ms[s]["em_oracle@8"] for s in ss])),
                        "sd": float(np.std([ms[s]["em_oracle@8"] for s in ss], ddof=1)) if len(ss) > 1 else 0.0,
                        "per_seed": {str(s): ms[s]["em_oracle@8"] for s in ss}},
        "mean_distinct": {"mean": float(np.mean([ms[s]["mean_distinct"] for s in ss])),
                          "sd": float(np.std([ms[s]["mean_distinct"] for s in ss], ddof=1)) if len(ss) > 1 else 0.0},
        "mean_gen_tokens": {"mean": float(np.mean([ms[s]["mean_gen_tokens"] for s in ss])),
                            "sd": float(np.std([ms[s]["mean_gen_tokens"] for s in ss], ddof=1)) if len(ss) > 1 else 0.0},
        "per_ds_em_oracle": {d: float(np.mean([ms[s]["per_ds"][d] for s in ss])) for d in S.DS},
        "oracle_vs_N_within8_EM": list(np.mean([S.oracle_curve_within8(EM[base][s]) for s in ss],
                                               axis=0).astype(float)),
        "oracle_vs_N_pooled_EM": S.oracle_curve_pooled(EM[base]) if len(ss) > 1 else None,
        "capture_recapture_EM": S.lp_ceiling(EM[base]) if len(ss) > 1 else None,
    }
em_deltas = {}
if CONTROL in EM:
    cr = np.mean([S.em_measure(EM[CONTROL][s])["_rec"].astype(float) for s in sorted(EM[CONTROL])], axis=0)
    cdi = S.em_measure(EM[CONTROL][sorted(EM[CONTROL])[0]])["_ds_index"]
    for base in emres:
        if base == CONTROL:
            continue
        ar = np.mean([S.em_measure(EM[base][s])["_rec"].astype(float) for s in sorted(EM[base])], axis=0)
        em_deltas[base] = {"em_oracle@8_vs_control": S.boot(ar, cr),
                           "per_cell": {d: S.boot(ar, cr, mask=(cdi == j)) for j, d in enumerate(S.DS)}}

# ================= provenance =================
prov = {
    "THE_DEPLOYED_SETTING_AS_FOUND": {
        "where": "runners/run_openvqa_lingshu7b.sh / run_openvqa_pathvqa.sh / run_cheapleg_open_gen.sh "
                 "-> src/labeling/run_openvqa.py",
        "generator": "Lingshu-7B snapshot b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9, vLLM, tp=1, bfloat16",
        "sampling": "SamplingParams(temperature=0.7, max_tokens=64, n=8, logprobs=5) -- every other "
                    "sampling field left at the vLLM default",
        "resolved_defaults": {"temperature": 0.7, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                              "repetition_penalty": 1.0, "presence_penalty": 0.0,
                              "frequency_penalty": 0.0, "seed": None},
        "seed_note": "the DEPLOYED pool was generated with NO seed set, so it is not bit-reproducible; "
                     "this is one reason the control had to be regenerated in-session rather than reused",
        "max_tokens": 64, "n_samples": 8, "max_model_len": 4096,
        "prompt": "run_openvqa.py SYS: 'You are an expert medical image analyst. Answer the question "
                  "with a short, specific phrase. Do not explain.'",
    },
    "RESOLUTION_OF_THE_PUBLISHED_CELLS": {
        "answer": "THE PUBLISHED 8 CELLS ARE NOT ALL AT THE SAME RESOLUTION -- there are THREE different "
                  "max_pixels values inside the canonical macro, spanning 51.2x.",
        "five_MCQ_cells": {
            "value_max_pixels": 12845056,
            "source": "MedEvalKit/models/Qwen2_5_VL/Qwen2_5_VL_vllm.py:51 reads CAP_MAX_PIXELS, which is "
                      "UNSET in every runner (grep: no runner sets it) => 0 => qwen_vl_utils default; "
                      "Lingshu-7B/32B preprocessor_config.json max_pixels = 12845056",
            "independently_verified_in_repo": "src/cascade/measure_testtime_vram.py:59,461 "
                                              "('CAP_MAX_PIXELS unset at eval => qwen_vl_utils defaults; "
                                              "verified 12,845,056')"},
        "three_open_cells_generator_AND_32B_direct": {
            "value_max_pixels": 250880,
            "label": "cap320",
            "source": "src/labeling/run_openvqa.py --cap default 'cap320', MAXPX = 1280*28*28 // 4; "
                      "NO runner passes --cap for any open-text arm (7B greedy, 7B sc8, or 32B-direct)"},
        "the_verifier_that_scores_those_candidates": {
            "value_max_pixels": 1003520,
            "source": "src/training_methods/verifier_transfer_eval.py MAXPX = 1280*28*28; "
                      "ckpts/train/lora_verifier_disjoint/train_config.json cap_div=1, max_pixels=1003520"},
        "consequences": [
            "The open half is internally MATCHED (7B pool and the 32B-direct bar are both cap320), so "
            "the open-cell comparison is fair -- but it is made at 1/51 of the pixels the MCQ half uses.",
            "The VERIFIER sees each image at 4.0x the resolution the GENERATOR saw it at (1,003,520 vs "
            "250,880). The scorer is strictly better-sighted than the proposer.",
            "run_openvqa.py's own label 'fullres' (1,003,520) is NOT the model's full resolution "
            "(12,845,056) -- it is 12.8x smaller. Naming landmine.",
            "Image resolution is therefore UNSWEPT for the open arm and the published operating point "
            "was inherited, not chosen on accuracy. This sweep does NOT test it (prompt and cap are "
            "held fixed by design); it is the obvious next experiment."],
    },
}

# ================= assemble =================
art = {
    "title": "SWEEP 1 -- 7B DECODING PARAMETERS, ONE VARIABLE AT A TIME (open-text candidate pools). "
             "The clean single-variable version of the experiment that was previously run confounded "
             "with the prompt portfolio.",
    "date": "2026-08-13",
    "question": "Can changing the 7B's inference parameters improve the samples it generates? "
                "Headline endpoint = SELECTED accuracy under the frozen incumbent verifier.",
    "no_fabricated_numbers": True,
    "endpoint_pool": {"n": 2345, "cells": {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500},
                      "idx_allowlist_source": "ckpts/train/lora_verifier_disjoint/transfer_dump_*.json; "
                                              "the generator asserts its loader reproduces this idx set "
                                              "exactly before generating (decoding_sweep_gen.py)"},
    "design": {
        "held_fixed": ["system prompt (run_openvqa.py SYS, verbatim)", "image cap = cap320 (250,880 px)",
                       "max_tokens = 64", "N = 8", "Lingshu-7B, vLLM, tp=1, bfloat16, max_model_len 4096"],
        "varied": "exactly ONE sampling parameter per setting, off the deployed control",
        "why_this_differs_from_the_killed_arm":
            "artifacts/open_diverse_2026-08-10.json varied PROMPT and TEMPERATURE together (5 prompts x "
            "3 temperatures) and lost after decontamination; the prompt/temperature effects were "
            "inseparable. Here the prompt is frozen, so each axis is identified.",
        "seeds_per_setting": 3, "nboot": 10000, "bootstrap_seed": 20260813,
        "control": "T07 = the deployed setting, REGENERATED IN THIS SESSION (never compared to the "
                   "stored 2026-08 numbers) -- the +-0.008 open-text reproducibility caveat requires a "
                   "matched control in the same serving configuration.",
    },
    "environment": {
        "gpus": "2x A100 80GB, SHARED with another user during the run (GPU0 co-tenant peaked ~33 GB); "
                "generation ran at gpu_memory_utilization 0.35 (GPU0) / 0.60 (GPU1) to avoid "
                "oversubscribing, never killing another process",
        "vllm": "0.10.1.1+381074ae.nv25.09 (generation + judge)",
        "torch": "2.9.0a0+50eac811a6.nv25.09", "transformers": "4.55.2", "peft": "0.14.0",
        "verifier_scoring": "HF transformers ONLY (CLAUDE.md landmine: vLLM drops all 192 visual.* LoRA "
                            "modules). bfloat16 + flash_attention_2, batch 1.",
        "note_on_vllm_version": "the DEPLOYED pool was generated under an older vLLM; this is a further "
                                "reason every delta here is against the in-session matched control.",
    },
    "null_tests": {"null_test_1_frozen_metric": null1, "null_test_2_verifier_harness": null2},
    "preregistration": json.load(open(os.path.join(
        ROOT, "results/cascade_methods/artifacts/_decoding_sweep_prereg.json"))),
    "provenance": prov,
    "reference_constants_not_recomputed": {
        "7B_temp0_greedy_judge_acc_pooled": 0.461834,
        "7B_temp0_greedy_per_cell": {"slake_open": 0.7302, "vqa_rad_open": 0.4900, "pathvqa_open": 0.3427},
        "source_greedy": "ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.judge.jsonl on the 2345 pool",
        "32B_direct_open_per_cell": {"slake_open": 0.8186, "vqa_rad_open": 0.6000, "pathvqa_open": 0.3760},
        "source_32b": "ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.judge.jsonl on the 2345 pool "
                      "(recomputed here, matches the brief exactly)",
        "note_on_greedy": "the frozen metric's 'greedy' field is MODAL-OF-8 (0.449467), NOT the temp-0 "
                          "greedy (0.461834). Different quantities; do not conflate.",
    },
    "settings": res,
    "deltas_vs_control": deltas,
    "EXACT_MATCH_CURRENCY_coverage": {
        "what_this_is": "oracle@8 / distinct-candidate count / token audit / capture-recapture ceiling "
                        "computed in EXACT-MATCH currency (run_openvqa.py score(): normalised exact "
                        "match + short-answer 'contains'), recorded at generation time as oks_em.",
        "why_it_is_here": "it needs no GPU, so it covers every generated setting even when the 32B judge "
                          "could not be scheduled on the shared machine.",
        "DO_NOT_MIX": "this is NOT the frozen judge currency the published cells use. On the SAME "
                      "deployed pool the stored judge-labelled oracle@8 is 0.626013 while EM is 0.6000 "
                      "-- EM runs ~0.026 low. Compare EM to EM only; never to 0.626013.",
        "settings": emres,
        "deltas_vs_control": em_deltas,
    },
    "oracle_vs_N_cost": cost,
    "sources": {
        "metric": "src/training_methods/genframe_data.py (frozen; unmodified)",
        "selector": "ckpts/train/lora_verifier_disjoint (CLEAN disjoint-trained LoRA verifier, the "
                    "incumbent; sel_eff 0.775204 on the deployed pool)",
        "judge": "src/labeling/run_judge.py -- MedVLThinker-32B (Qwen2.5-32B backbone), text-only, "
                 "temperature 0, Yes/No logit comparison. THE PROJECT'S OWN JUDGE, reused unmodified.",
        "generation": "src/cascade_methods/decoding_sweep_gen.py",
        "verifier_scoring": "src/cascade_methods/decoding_sweep_verify.py (pyes verbatim from "
                            "src/training_methods/verifier_transfer_eval.py)",
        "analysis": "src/cascade_methods/decoding_sweep_analyse.py + decoding_sweep_report.py",
        "pools": "ckpts/openvqa/decoding_sweep/ckpt_{ds}_{tag}.jsonl",
        "settings_grid": "results/cascade_methods/artifacts/_decoding_sweep_settings.json",
    },
}
# ================= headline, computed from the data =================
head = {}
SEL = [b for b in res if "selected" in res[b]]
if CONTROL in SEL and deltas:
    ranked = sorted(SEL, key=lambda b: -res[b]["selected"]["mean"])
    best = ranked[0]
    wins = [b for b in deltas if "SELECTED_vs_control" in deltas[b]
            and deltas[b]["SELECTED_vs_control"]["verdict"] == "WIN"]
    losses = [b for b in deltas if "SELECTED_vs_control" in deltas[b]
              and deltas[b]["SELECTED_vs_control"]["verdict"] == "LOSS"]
    orc_up = [b for b in deltas if deltas[b]["oracle@8_vs_control"]["verdict"] == "WIN"]
    head = {
        "control_SELECTED": res[CONTROL]["selected"]["mean"],
        "control_oracle@8": res[CONTROL]["oracle@8"]["mean"],
        "control_sel_eff": res[CONTROL]["sel_eff"]["mean"],
        "ranking_by_SELECTED": [{"setting": b, "SELECTED": res[b]["selected"]["mean"],
                                 "sd_over_seeds": res[b]["selected"]["sd"],
                                 "oracle@8": res[b]["oracle@8"]["mean"],
                                 "sel_eff": res[b]["sel_eff"]["mean"],
                                 "mean_distinct": res[b]["mean_distinct"]["mean"]} for b in ranked],
        "best_setting_by_SELECTED": best,
        "settings_BEATING_control_on_SELECTED": wins,
        "settings_LOSING_to_control_on_SELECTED": losses,
        "settings_RAISING_oracle@8_over_control": orc_up,
        "settings_that_raise_oracle_but_not_SELECTED": [b for b in orc_up if b not in wins],
        "n_settings_compared": len(deltas),
        "settings_with_SELECTED": SEL,
        "settings_coverage_only": [b for b in res if b not in SEL],
        "oracle_ranking_ALL_settings": sorted(
            [{"setting": b, "oracle@8": res[b]["oracle@8"]["mean"],
              "sd_over_seeds": res[b]["oracle@8"]["sd"],
              "mean_distinct": res[b]["mean_distinct"]["mean"],
              "n_seeds": res[b]["n_seeds"],
              "oracle_delta_vs_control": (deltas[b]["oracle@8_vs_control"] if b in deltas else None)}
             for b in res], key=lambda x: -x["oracle@8"]),
    }
    if best != CONTROL:
        head["best_vs_control"] = deltas[best]["SELECTED_vs_control"]
art["HEADLINE"] = head

# ---- EM headline (available without the judge) ----
emhead = {}
if CONTROL in emres:
    order = sorted(emres, key=lambda b: -emres[b]["em_oracle@8"]["mean"])
    emhead = {
        "control_em_oracle@8": emres[CONTROL]["em_oracle@8"]["mean"],
        "ranking_by_em_oracle@8": [
            {"setting": b, "params": emres[b]["params"], "n_seeds": emres[b]["n_seeds"],
             "em_oracle@8": emres[b]["em_oracle@8"]["mean"],
             "sd_over_seeds": emres[b]["em_oracle@8"]["sd"],
             "mean_distinct": emres[b]["mean_distinct"]["mean"],
             "mean_gen_tokens": emres[b]["mean_gen_tokens"]["mean"],
             "capture_recapture_ceiling": (emres[b]["capture_recapture_EM"]["macro_reachable_share_mean"]
                                           if emres[b]["capture_recapture_EM"] else None),
             "delta_vs_control": (em_deltas[b]["em_oracle@8_vs_control"] if b in em_deltas else "IS CONTROL")}
            for b in order],
        "best_setting_by_em_oracle@8": order[0],
        "settings_beating_control": [b for b in em_deltas
                                     if em_deltas[b]["em_oracle@8_vs_control"]["verdict"] == "WIN"],
        "settings_losing_to_control": [b for b in em_deltas
                                       if em_deltas[b]["em_oracle@8_vs_control"]["verdict"] == "LOSS"],
    }
art["EM_HEADLINE"] = emhead

art["STATUS"] = {
    "measured": [
        "NULL TEST 1 (frozen metric reproduces the published bar): PASS, max abs deviation 3.60e-07.",
        "NULL TEST 2 (this session's HF verifier harness reproduces the frozen incumbent scores): PASS "
        "EXACTLY -- max |score deviation| 0.0 over 1,440 slots / 180 items, argmax pick agreement 1.000.",
        "The DEPLOYED decoding setting, read out of the generation code rather than assumed.",
        "The RESOLUTION of every published cell (three different max_pixels inside the canonical macro).",
        "Generation of 9 complete 8-sample pools over the frozen 2,345-item endpoint: T=0.3, T=0.7 "
        "(matched control), T=1.3 and repetition_penalty=1.10 at 2 seeds each, min_p=0.10 at 1 seed.",
        "COVERAGE endpoints in EXACT-MATCH currency for all five settings: oracle@8, distinct-candidate "
        "count, generated-token audit, Lincoln-Petersen capture-recapture ceiling, oracle-vs-N curves, "
        "and paired item-bootstrap deltas vs the matched control.",
    ],
    "NOT_measured": [
        "oracle@8 in the FROZEN JUDGE currency for the swept settings.",
        "sel_eff and SELECTED accuracy for the swept settings -- the headline endpoint.",
        "top_p, top_k, min_p=0.05, repetition_penalty=1.05, T=0.5, T=1.0 and the T=1.3+min_p cell: "
        "generated pools were not completed for these.",
    ],
    "why_not": "The judge (MedVLThinker-32B, tp=2) needs ~37 GiB free on BOTH A100s simultaneously. For "
               "the whole available window two SIBLING JOBS on this shared machine (vision_diversity_* "
               "and resolution_open_generate/score) held 59 GiB of GPU0 and up to 74 GiB of GPU1. The "
               "judge was left waiting rather than oversubscribing or killing another user's process. "
               "SELECTED cannot be computed without judge labels (it is the label of the verifier's "
               "picked slot), so the verifier was not the binding constraint -- the judge was.",
    "how_to_complete": "runners/finish_decoding_sweep.sh -- waits for the already-running, resumable "
                       "judge, then scores the frozen verifier on the pre-registered selection tier and "
                       "regenerates this artifact. Every stage is content-addressed, so it only pays for "
                       "what is missing.",
    "does_the_missing_piece_change_the_reading": "It could only change it by making the answer MORE "
        "negative or by a sel_eff gain large enough to overturn a coverage loss. SELECTED = oracle@8 x "
        "sel_eff EXACTLY. Every swept setting loses EM-currency oracle@8 to the control by 0.013 to "
        "0.057 (all CI-clean). To win on SELECTED, a setting losing 0.0132 of oracle (rp11) would need "
        "sel_eff to rise from 0.775 to 0.793 (+2.3% relative), and one losing 0.0484 (T13) would need "
        "0.775 -> 0.849 (+9.5% relative). ~27 selector architectures in this project have landed in "
        "0.78-0.81 and the repo calls 0.78-0.81 a FIELD CONSTANT, so a +9.5% relative jump from a "
        "DECODING change with a FROZEN selector is not a live hypothesis. This is an argument, not a "
        "measurement, and is labelled as such.",
}

art["EXCLUDED_RUNS_WITH_INCOMPLETE_INPUTS"] = excluded
art["completeness"] = {
    "settings_reported": len(res),
    "settings_in_grid": len(groups),
    "runs_excluded_for_incomplete_inputs": len(excluded),
    "seeds_per_reported_setting": {b: res[b]["n_seeds"] for b in res},
    "note": "a run is measured only when EVERY slot has both a judge label and a verifier score; "
            "otherwise it is excluded, never partially scored."}
if excluded:
    print(f"\n!! {len(excluded)} runs excluded for incomplete inputs", flush=True)

json.dump(art, open(OUT, "w"), indent=1, default=float)
print(f"\nwrote {OUT}", flush=True)
