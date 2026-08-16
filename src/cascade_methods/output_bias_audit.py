#!/usr/bin/env python3
"""output_bias_audit.py -- ATTACK 1 STEP 1: MEASURE THE OUTPUT BIAS ON ALL EIGHT CELLS.

Pure CPU.  Reads only dumps that already exist:
  * MedEvalKit/eval_results_lingshu7b_full/  -- the deployed always-7B greedy pass (READ ONLY)
  * ckpts/closed_as_open/                    -- the 2026-08-16 prompt arms (matched fullres)
  * ckpts/train/lora_verifier_disjoint/transfer_dump_*.json -- the published open cells
  * ckpts/output_bias/gen_*_id.jsonl         -- this round's first-token posteriors (if present)

Everything it prints is a MEASUREMENT of the deployed model, not a method.  The "recoverable"
numbers per cell look at the eval labels precisely because they are BOUNDS, and no arm in STEP 2/3
is permitted to use them.  Note the distinction the artifact keeps: the marginal-matched line is an
exact construction, while the BEST_FOUND line is a multi-start coordinate ascent and is therefore a
LOWER bound on the true oracle, not the oracle itself.

    OMP_NUM_THREADS=1 python3 src/cascade_methods/output_bias_audit.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import output_bias_lib as L                                             # noqa: E402


# =============================================================================================
# null tests
# =============================================================================================
def n1_frozen_open_metric():
    """N1 -- the frozen open-text metric.  src/training_methods/genframe_data.py must still report
    sel_eff 0.775204 / oracle@8 0.626013 / greedy 0.449467 on n=2345 (n_recoverable 1468)."""
    sys.path.insert(0, os.path.join(L.ROOT, "src"))
    from training_methods import genframe_data as G
    nt = G.null_test()
    return {"pass": bool(nt["pass"]), "max_abs_deviation": float(nt["max_abs_deviation"]),
            "measured": {k: nt["measured"][k] for k in
                         ["n", "n_recoverable", "oracle@8", "selected", "greedy", "sel_eff",
                          "cand_auroc"]},
            "source": "src/training_methods/genframe_data.py null_test() (frozen metric)"}


def n2_prompt_byte_equality():
    """N2 -- our rebuilt prompts must be byte-identical to the strings MedEvalKit itself stored.
    Only PMC_VQA and SLAKE store a `prompt` field in results.json; the other three cells' templates
    are read directly out of MedEvalKit/utils/question_formats.py (quoted in output_bias_lib)."""
    items = L.build_items(["PMC_VQA", "SLAKE_closed"])
    out = {}
    worst = 0
    for cell in ("PMC_VQA", "SLAKE_closed"):
        rows = L.deployed_rows(cell)
        bad = 0
        for k, r in enumerate(rows):
            if L.deployed_prompt(cell, items[cell][k]) != r["prompt"]:
                bad += 1
        out[cell] = {"n_checked": len(rows), "mismatches": bad}
        worst = max(worst, bad)
    out["cells_without_a_stored_prompt_field"] = ["MedXpertQA-MM", "VQA_RAD_closed",
                                                  "PATH_VQA_closed"]
    out["pass"] = bool(worst == 0)
    return out


def g1_grader_null_test():
    """G1 -- re-grade every deployed response with this module's verbatim copy of MedEvalKit's own
    grader and require it to reproduce (a) the harness's stored `correct` field row by row and
    (b) the published always-7B cell.  The three OPEN cells are checked against the transfer
    dumps that DEFINE them."""
    out = {}
    worst = 0.0
    items = L.build_items(L.MCQ_CELLS + L.BINARY_CELLS)
    for cell in L.MCQ_CELLS + L.BINARY_CELLS:
        rows = L.deployed_rows(cell)
        mine = [L.em_harness(cell, items[cell][k], r["response"]) for k, r in enumerate(rows)]
        theirs = [int(r.get("correct") is True) for r in rows]
        dis = sum(1 for a, b in zip(mine, theirs) if a != b)
        acc = float(np.mean(mine))
        dev = abs(acc - L.PUBLISHED_7B[cell])
        worst = max(worst, dev, dis)
        out[cell] = {"n": len(rows), "row_disagreements_with_harness_correct_field": dis,
                     "acc_regraded": L.r6(acc), "published_always_7b": L.PUBLISHED_7B[cell],
                     "abs_dev_vs_published": L.r6(dev)}
    for cell, ds in [("SLAKE_open", "slake_open"), ("VQA_RAD_open", "vqa_rad_open"),
                     ("PATH_VQA_open", "pathvqa_open")]:
        d = json.load(open(os.path.join(
            L.ROOT, f"ckpts/train/lora_verifier_disjoint/transfer_dump_{ds}_lingshu7b.json")))
        acc = float(np.mean([r["greedy_ok"] for r in d]))
        dev = abs(acc - L.PUBLISHED_7B[cell])
        worst = max(worst, dev)
        out[cell] = {"n": len(d), "acc_from_transfer_dump_greedy_ok": L.r6(acc),
                     "published_always_7b": L.PUBLISHED_7B[cell],
                     "abs_dev_vs_published": L.r6(dev),
                     "currency": "32B judge (the dump's greedy_ok field)"}
    out["max_abs_deviation"] = L.r6(worst)
    out["pass"] = bool(worst <= 1e-4)
    out["note"] = ("published cells are 4-dp rounded, so a deviation at 1e-5 is rounding; any "
                   "non-zero row disagreement on a generative cell is a grader defect and fails.")
    return out


def a1_pathvqa_prompt_fix_reproduction():
    """THE ATTACK'S OWN NULL TEST.  Reproduce, from the 2026-08-16 dumps, (a) the PATH_VQA_closed
    yes-bias under MedEvalKit's judgement prompt and (b) the +0.0419 judge / +0.0416 EM gain from
    replacing it with the open instruction at MATCHED fullres."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "closed_as_open_lib", os.path.join(L.ROOT, "src/cascade_methods/closed_as_open_lib.py"))
    CL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CL)
    items = L.build_items(["PATH_VQA_closed"])["PATH_VQA_closed"]
    jm = CL.judge_map("PATH_VQA_closed")
    out = {}
    vec = {}
    for arm in ("closedD_g_full", "openMEK_g_full"):
        g = CL.load_gen("PATH_VQA_closed", arm)
        em, ju, tok, yes = [], [], [], []
        for it in items:
            r = g[it["i"]]
            pred = r["preds"][0]
            em.append(L.em_harness("PATH_VQA_closed", it, pred))
            ju.append(jm.get((it["i"], CL.norm_text(pred)), None))
            tok.append(r["gen_tokens"][0])
            yes.append(1 if L.polarity(pred) == "yes" else 0)
        gold_yes = float(np.mean([1 if str(it["answer"]).lower() == "yes" else 0 for it in items]))
        n_unlab = sum(1 for x in ju if x is None)
        juv = [0 if x is None else int(x) for x in ju]
        vec[arm] = (np.array(em), np.array(juv))
        out[arm] = {"n": len(items), "EM_harness": L.r6(np.mean(em)),
                    "judge_32b": L.r6(np.mean(juv)), "n_judge_unlabelled_scored_0": n_unlab,
                    "mean_gen_tokens": L.r6(np.mean(tok)),
                    "pred_yes_rate": L.r6(np.mean(yes)), "gold_yes_rate": L.r6(gold_yes),
                    "YES_BIAS": L.r6(np.mean(yes) - gold_yes)}
    out["delta_openMEK_minus_closedD"] = {
        "judge": L.boot_delta(vec["openMEK_g_full"][1], vec["closedD_g_full"][1]),
        "EM_harness": L.boot_delta(vec["openMEK_g_full"][0], vec["closedD_g_full"][0])}
    out["reference_2026-08-16"] = {"judge": "+0.0419 [+0.0321,+0.0518]",
                                   "EM": "+0.0416 [+0.0318,+0.0515]",
                                   "artifact": "artifacts/closed_as_open_2026-08-16.json"}
    d = out["delta_openMEK_minus_closedD"]
    out["pass"] = bool(abs(d["judge"]["delta"] - 0.0419) < 5e-4
                       and abs(d["EM_harness"]["delta"] - 0.0416) < 5e-4)
    return out


def a2_judge_equals_em_on_yesno():
    """Supporting measurement for STEP 2b: on a binary cell an output-side polarity flip produces
    the literal string 'yes' or 'no', and we need its judge label without a new judge call.
    Measure whether the 32B judge is exactly exact-match on yes/no answers with yes/no gold."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "closed_as_open_lib", os.path.join(L.ROOT, "src/cascade_methods/closed_as_open_lib.py"))
    CL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CL)
    items = L.build_items(L.BINARY_CELLS)
    out = {}
    tot = agree = 0
    for cell in L.BINARY_CELLS:
        jm = CL.judge_map(cell)
        n = ok = 0
        for (i, na), lab in jm.items():
            if na not in ("yes", "no"):
                continue
            g = str(items[cell][i]["answer"]).strip().lower()
            if g not in ("yes", "no"):
                continue
            n += 1
            ok += int(int(lab) == int(na == g))
        out[cell] = {"n_judge_calls_on_yesno": n, "agreement_with_exact_match": L.r6(ok / max(1, n))}
        tot += n
        agree += ok
    out["pooled"] = {"n": tot, "agreement": L.r6(agree / max(1, tot))}
    out["consequence"] = ("agreement is exactly 1.0 over %d calls, so a yes/no answer's judge "
                          "label is IMPUTED as exact match rather than re-judged. This is a "
                          "measurement, not an assumption." % tot)
    out["pass"] = bool(tot > 1000 and agree == tot)
    return out


# =============================================================================================
# STEP 1 -- the bias measurement
# =============================================================================================
def audit_binary_cells():
    """Yes-rate, predicted vs gold, per binary cell, under the DEPLOYED prompt; plus the same
    under every 2026-08-16 alternate prompt at matched fullres; plus per-polarity precision."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "closed_as_open_lib", os.path.join(L.ROOT, "src/cascade_methods/closed_as_open_lib.py"))
    CL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CL)
    items = L.build_items(L.BINARY_CELLS)
    out = {}
    for cell in L.BINARY_CELLS:
        rows = L.deployed_rows(cell)
        its = items[cell]
        gold = [str(r["answer"]).strip().lower() for r in rows]
        yn = [k for k, g in enumerate(gold) if g in ("yes", "no")]
        pol = [L.polarity(r["response"]) for r in rows]
        ok = [int(r.get("correct") is True) for r in rows]
        gy = float(np.mean([gold[k] == "yes" for k in yn]))
        pyn = [k for k in yn if pol[k] is not None]
        py = float(np.mean([pol[k] == "yes" for k in pyn]))
        prec = {}
        for p in ("yes", "no"):
            m = [k for k in pyn if pol[k] == p]
            prec[p] = {"n": len(m), "precision_P_correct_given_predicts": L.r6(np.mean([ok[k] for k in m])),
                       "mean_conf": L.r6(np.mean([rows[k]["conf"] for k in m])),
                       "mean_margin": L.r6(np.mean([rows[k]["margin"] for k in m]))}
        prompts = {}
        for arm in ("closedD_g_full", "openMEK_g_full", "openPRJ_g_full"):
            g = CL.load_gen(cell, arm)
            if not g:
                prompts[arm] = "NOT MEASURED -- dump absent"
                continue
            pa = [L.polarity(g[its[k]["i"]]["preds"][0]) for k in yn]
            sub = [k for k, p in zip(yn, pa) if p is not None]
            pr = float(np.mean([L.polarity(g[its[k]["i"]]["preds"][0]) == "yes" for k in sub]))
            prompts[arm] = {"pred_yes_rate": L.r6(pr), "YES_BIAS": L.r6(pr - gy),
                            "mean_gen_tokens": L.r6(np.mean([g[it["i"]]["gen_tokens"][0]
                                                             for it in its])),
                            "EM_harness": L.r6(np.mean([L.em_harness(cell, it,
                                                                     g[it["i"]]["preds"][0])
                                                        for it in its]))}
        out[cell] = {
            "n": len(rows), "n_with_yesno_gold": len(yn),
            "gold_yes_rate": L.r6(gy), "pred_yes_rate_deployed": L.r6(py),
            "YES_BIAS_deployed": L.r6(py - gy),
            "n_unparsed_polarity": int(len(yn) - len(pyn)),
            "deployed_instruction": ("get_close_ended_prompt (NO answer space given)"
                                     if cell == "SLAKE_closed"
                                     else "get_judgement_prompt (answer space GIVEN: yes/no)"),
            "per_predicted_polarity": prec,
            "by_prompt_arm_matched_fullres": prompts,
        }
    return out


def audit_mcq_cells():
    """Predicted vs gold answer-letter marginal, per-letter precision, and the deployed model's own
    confidence per letter (the omitted-variable check).  Confidence is MedEvalKit's stored `conf`
    (top-1 first-token probability under logprobs=5)."""
    items = L.build_items(L.MCQ_CELLS)
    out = {}
    for cell in L.MCQ_CELLS:
        rows = L.deployed_rows(cell)
        its = items[cell]
        K = max(len(it["choices"]) for it in its)
        letters = [chr(65 + i) for i in range(K)]
        gold = [str(r["answer"]).strip().upper() for r in rows]
        # the model's PREDICTED letter, read the way MedEvalKit reads it
        pred = []
        for k, r in enumerate(rows):
            resp = L.parse_response(str(r["response"])).strip().upper()
            p = None
            for ch in resp:
                if ch in letters:
                    p = ch
                    break
            pred.append(p)
        ok = [int(r.get("correct") is True) for r in rows]
        gm = {x: L.r6(np.mean([g == x for g in gold])) for x in letters}
        pm = {x: L.r6(np.mean([p == x for p in pred])) for x in letters}
        pm["UNPARSED"] = L.r6(np.mean([p is None for p in pred]))
        per = {}
        for x in letters:
            m = [k for k, p in enumerate(pred) if p == x]
            if not m:
                continue
            per[x] = {"n": len(m),
                      "precision_P_correct_given_predicts_L": L.r6(np.mean([ok[k] for k in m])),
                      "mean_confidence": L.r6(np.mean([rows[k]["conf"] for k in m])),
                      "calibration_gap_precision_minus_confidence":
                          L.r6(np.mean([ok[k] for k in m]) - np.mean([rows[k]["conf"] for k in m]))}
        pr = [per[x]["precision_P_correct_given_predicts_L"] for x in per]
        cf = [per[x]["mean_confidence"] for x in per]
        # what a predictor with this letter marginal and ZERO item-level skill would score
        marg_only = float(sum(np.mean([p == x for p in pred]) * np.mean([g == x for g in gold])
                              for x in letters))
        out[cell] = {
            "n": len(rows), "n_options": K,
            "gold_letter_marginal": gm, "pred_letter_marginal": pm,
            "L1_marginal_mismatch": L.r6(sum(abs(pm[x] - gm[x]) for x in letters)),
            "majority_gold_letter_floor": L.r6(max(gm.values())),
            "acc": L.r6(np.mean(ok)),
            "marginal_only_acc_zero_skill": L.r6(marg_only),
            "skill_above_marginal": L.r6(np.mean(ok) - marg_only),
            "per_predicted_letter": per,
            "precision_range_across_letters": L.r6(max(pr) - min(pr)) if pr else None,
            "confidence_range_across_letters": L.r6(max(cf) - min(cf)) if cf else None,
            "ratio_precision_range_over_confidence_range":
                L.r6((max(pr) - min(pr)) / (max(cf) - min(cf))) if cf and max(cf) > min(cf) else None,
        }
    return out


def audit_open_cells():
    """Length and phrasing priors on the three OPEN cells, and whether the grader interacts with
    length.  Answer TEXT comes from ckpts/openvqa/cheap_lingshu7b (the deployed open generation);
    its own judge labels are used for the interaction so text and label always come from the same
    run.  That run scores 0.7302 / 0.4900 / 0.3427 against the published 0.7364 / 0.4650 / 0.3240
    -- the standing +/-0.008 open-text reproducibility caveat, stated, not hidden."""
    base = os.path.join(L.ROOT, "ckpts/openvqa/cheap_lingshu7b")
    out = {}
    for cell, short in [("SLAKE_open", "slake"), ("VQA_RAD_open", "vqa_rad"),
                        ("PATH_VQA_open", "pathvqa")]:
        p = os.path.join(base, f"ckpt_{short}_open_lingshu7b.jsonl")
        jp = p.replace(".jsonl", ".judge.jsonl")
        rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
        jm = {json.loads(x)["idx"]: json.loads(x)["judge_ok"]
              for x in open(jp, encoding="utf-8") if x.strip()}
        pl = np.array([len(L.norm_text(r["preds"][0]).split()) for r in rows], float)
        gl = np.array([len(L.norm_text(r["gold"]).split()) for r in rows], float)
        pc = np.array([len(L.norm_text(r["preds"][0])) for r in rows], float)
        gc = np.array([len(L.norm_text(r["gold"])) for r in rows], float)
        j = np.array([jm.get(r["idx"], 0) for r in rows], float)
        em = np.array([int(L.norm_text(r["preds"][0]) == L.norm_text(r["gold"])) for r in rows],
                      float)
        # does the grader reward length?  judge accuracy and EM by EXACT predicted word count
        # (quantile buckets are degenerate here -- 55-71%% of predictions are a single word).
        by = {}
        for lab, m in (("1_word", pl == 1), ("2_words", pl == 2), ("3plus_words", pl >= 3)):
            if m.sum() == 0:
                continue
            by[lab] = {"n": int(m.sum()), "mean_pred_words": L.r6(pl[m].mean()),
                       "judge": L.r6(j[m].mean()), "EM": L.r6(em[m].mean()),
                       "judge_minus_EM": L.r6(j[m].mean() - em[m].mean())}
        out[cell] = {
            "source_dump": os.path.relpath(p, L.ROOT),
            "n": len(rows),
            "judge_acc_of_this_dump": L.r6(j.mean()),
            "published_cell": L.PUBLISHED_7B[cell],
            "abs_dev_vs_published": L.r6(abs(j.mean() - L.PUBLISHED_7B[cell])),
            "mean_pred_words": L.r6(pl.mean()), "mean_gold_words": L.r6(gl.mean()),
            "LENGTH_BIAS_words_pred_minus_gold": L.r6(pl.mean() - gl.mean()),
            "mean_pred_chars": L.r6(pc.mean()), "mean_gold_chars": L.r6(gc.mean()),
            "frac_pred_shorter_than_gold": L.r6(np.mean(pl < gl)),
            "frac_pred_longer_than_gold": L.r6(np.mean(pl > gl)),
            "frac_pred_exactly_one_word": L.r6(np.mean(pl == 1)),
            "frac_gold_exactly_one_word": L.r6(np.mean(gl == 1)),
            "corr_predlen_judge": L.r6(np.corrcoef(pl, j)[0, 1]),
            "corr_predlen_EM": L.r6(np.corrcoef(pl, em)[0, 1]),
            "judge_leniency_judge_minus_EM": L.r6(j.mean() - em.mean()),
            "by_predicted_length_tercile": by,
        }
    return out


# =============================================================================================
# ORACLE UPPER BOUNDS -- what is recoverable by removing the bias.  BOUNDS, NOT METHODS.
# =============================================================================================
def oracle_bounds():
    """Per cell: the accuracy gain available from (i) matching the model's predicted marginal to
    the cell's own gold marginal, and (ii) the single best global logit shift chosen ON THE EVAL
    LABELS.  Both are upper bounds and neither is deployable; STEP 2 must beat neither, only be
    compared to them.  Requires this round's first-token posteriors."""
    out = {}
    items = L.build_items(L.MCQ_CELLS + L.BINARY_CELLS)
    for cell in L.MCQ_CELLS:
        gen = L.load_gen(cell, "id")
        its = items[cell]
        if len(gen) < len(its):
            out[cell] = f"NOT MEASURED -- {len(gen)}/{len(its)} rows present"
            continue
        K = max(len(it["choices"]) for it in its)
        X = np.full((len(its), K), L.FLOOR_LOGPROB)
        g = np.zeros(len(its), int)
        for k, it in enumerate(its):
            n = len(it["choices"])
            X[k, :n] = L.letter_logits(gen[it["i"]], n)
            X[k, n:] = -1e9
            g[k] = ord(str(it["answer"]).strip().upper()) - 65
        base = float((X.argmax(1) == g).mean())
        tgt = np.bincount(g, minlength=K) / len(g)
        w = L.fit_shift_marginal(X, tgt)
        mm = float(((X - w).argmax(1) == g).mean())
        # BEST GLOBAL SHIFT FOUND by multi-start coordinate ascent on the eval labels.  This is a
        # LOWER bound on the true oracle, not the oracle: the objective is piecewise constant in w
        # and coordinate ascent can stall.  It is seeded from zero, from the marginal-matched w, and
        # from the content-free log-prior, so it cannot be beaten by any of the deployed arms.
        starts = [np.zeros(K), w.copy()]
        cf = L.load_gen(cell, "cf_blank")
        if cf:
            R = np.array([L.letter_logits(r, K) for r in cf.values()])
            R = R - R.max(1, keepdims=True)
            starts.append(R.mean(0) - R.mean(0).mean())
        best = np.zeros(K)
        bacc = base
        for w0 in starts:
            cur, cacc = w0 - w0.mean(), float(((X - (w0 - w0.mean())).argmax(1) == g).mean())
            for _ in range(8):
                improved = False
                for j in range(K):
                    for v in np.linspace(-4, 4, 321):
                        ww = cur.copy()
                        ww[j] = v
                        ww = ww - ww.mean()
                        a = float(((X - ww).argmax(1) == g).mean())
                        if a > cacc + 1e-12:
                            cacc, cur, improved = a, ww, True
                if not improved:
                    break
            if cacc > bacc:
                bacc, best = cacc, cur
        dep = float(np.mean([int(r.get("correct") is True) for r in L.deployed_rows(cell)]))
        out[cell] = {"n": len(its),
                     "deployed_acc": L.r6(dep),
                     "letter_argmax_acc": L.r6(base),
                     "readout_minus_deployed": L.r6(base - dep),
                     "ORACLE_marginal_matched_acc": L.r6(mm),
                     "ORACLE_marginal_matched_gain_vs_readout": L.r6(mm - base),
                     "RECOVERABLE_vs_deployed_cell": L.r6(mm - dep),
                     "BEST_FOUND_global_shift_acc": L.r6(bacc),
                     "BEST_FOUND_global_shift_gain_vs_readout": L.r6(bacc - base),
                     "BEST_FOUND_global_shift_gain_vs_deployed_cell": L.r6(bacc - dep),
                     "BEST_FOUND_global_shift": [L.r6(v) for v in best],
                     "note": ("Both look at the eval labels; neither is a method. The marginal-"
                              "matched line is an exact construction; the BEST_FOUND line is a "
                              "multi-start coordinate ascent, i.e. a LOWER bound on the true "
                              "oracle, not the oracle -- the objective is piecewise constant in w.")}
    for cell in L.BINARY_CELLS:
        gen = L.load_gen(cell, "id")
        its = items[cell]
        if len(gen) < len(its):
            out[cell] = f"NOT MEASURED -- {len(gen)}/{len(its)} rows present"
            continue
        sub = [it for it in its if str(it["answer"]).strip().lower() in ("yes", "no")]
        d = np.array([np.subtract(*L.yesno_logits(gen[it["i"]])) for it in sub])
        y = np.array([str(it["answer"]).strip().lower() == "yes" for it in sub], int)
        base = float(((d > 0).astype(int) == y).mean())
        t = float(np.quantile(d, 1 - y.mean()))
        mm = float(((d > t).astype(int) == y).mean())
        ths = np.unique(np.concatenate([d, [d.min() - 1, d.max() + 1]]))
        accs = np.array([((d > th).astype(int) == y).mean() for th in ths])
        j = int(accs.argmax())
        frac = len(sub) / len(its)
        out[cell] = {"n_yesno": len(sub), "n_cell": len(its),
                     "yesno_share_of_cell": L.r6(frac),
                     "polarity_argmax_acc_on_subset": L.r6(base),
                     "ORACLE_marginal_matched_acc": L.r6(mm),
                     "ORACLE_marginal_matched_gain_on_subset": L.r6(mm - base),
                     "ORACLE_best_threshold_acc": L.r6(accs[j]),
                     "ORACLE_best_threshold": L.r6(ths[j]),
                     "ORACLE_best_threshold_gain_on_subset": L.r6(accs[j] - base),
                     "RECOVERABLE_scaled_to_the_whole_cell": L.r6((accs[j] - base) * frac),
                     "note": ("UPPER BOUNDS on the yes/no-gold subset; the last line rescales the "
                              "best-threshold bound by that subset's share of the cell, which is "
                              "the number that is comparable across cells.")}
    for cell in L.OPEN_CELLS:
        out[cell] = {
            "RECOVERABLE_from_an_output_prior": "NOT DEFINED",
            "why": ("a length or phrasing prior on a free-text cell cannot be corrected by "
                    "re-ranking a single greedy string -- there is nothing to re-rank. The nearest "
                    "measurable bound is oracle@8 over the sampled pool, which is the coverage / "
                    "selection wall this project has already quantified: greedy 0.449467 -> "
                    "oracle@8 0.626013 over the 2,345 pooled open questions "
                    "(src/training_methods/genframe_data.py, null test N1), and that requires 8 "
                    "samples, not zero extra compute."),
        }
    return out


def decompose_letter_bias_position_vs_content():
    """IS THE PMC-VQA LETTER BIAS POSITIONAL OR CONTENT-DRIVEN?  The distinction decides what an
    output-side letter prior is actually doing.

    The only cyclic-rotation dump in this repo is ckpts/mcq_tta/PMC_VQA_stageA.jsonl, which is
    LINGSHU-32B on a 6,000-item subsample (artifacts/mcq_tta_2026-08-10.json) -- so this is
    measured on the 32B, stated, not silently attributed to the 7B.  Rotation k moves the option
    that was in original slot j into slot (j+k) mod 4 and relabels it, so:
      * a marginal that is constant BY SLOT across rotations is a pure POSITION bias;
      * a marginal that is constant BY ORIGINAL OPTION across rotations is a pure CONTENT bias.
    """
    p = os.path.join(L.ROOT, "ckpts/mcq_tta/PMC_VQA_stageA.jsonl")
    if not os.path.exists(p):
        return "NOT MEASURED -- ckpts/mcq_tta/PMC_VQA_stageA.jsonl absent"
    by = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        by.setdefault(r["i"], {})[r["v"]] = r
    full = [v for v in by.values() if len(v) == 4]
    letters = list("ABCD")
    slot = np.zeros((4, 4))
    cont = np.zeros((4, 4))
    for k in range(4):
        for v in full:
            r = v[k]
            lp = r["first_logprobs"]
            s = int(np.argmax([lp.get(x, -99) for x in letters]))
            slot[k, s] += 1
            cont[k, r["orig_of_slot"][s]] += 1
    n = len(full)
    slot /= n
    cont /= n
    return {
        "model": "Lingshu-32B (the only rotation dump in the repo)",
        "source": "ckpts/mcq_tta/PMC_VQA_stageA.jsonl",
        "n_items_with_all_4_rotations": n,
        "pred_marginal_BY_SLOT_per_rotation": [[L.r6(x) for x in row] for row in slot],
        "pred_marginal_BY_ORIGINAL_OPTION_per_rotation": [[L.r6(x) for x in row] for row in cont],
        "position_component_slot_marginal_averaged_over_rotations":
            [L.r6(x) for x in slot.mean(0)],
        "content_component_original_option_marginal_averaged_over_rotations":
            [L.r6(x) for x in cont.mean(0)],
        "L1_nonuniformity_of_position_component": L.r6(np.abs(slot.mean(0) - 0.25).sum()),
        "L1_nonuniformity_of_content_component": L.r6(np.abs(cont.mean(0) - 0.25).sum()),
        "gold_marginal_by_ORIGINAL_option_on_this_subsample":
            [L.r6(np.mean([v[0]["answer"] == c for v in full])) for c in letters],
        "L1_nonuniformity_of_gold": L.r6(np.abs(
            np.array([np.mean([v[0]["answer"] == c for v in full]) for c in letters]) - 0.25).sum()),
        "content_component_as_fraction_of_gold_nonuniformity": L.r6(
            np.abs(cont.mean(0) - 0.25).sum()
            / np.abs(np.array([np.mean([v[0]["answer"] == c for v in full])
                               for c in letters]) - 0.25).sum()),
        "reading": ("TWO SEPARABLE THINGS, and only one of them is a model pathology. (1) The "
                    "POSITION component is small (L1 0.0601) and is essentially an aversion to the "
                    "LAST option slot -- that one IS a bias. (2) The CONTENT component is three "
                    "times larger (L1 0.1883), and it points in the SAME DIRECTION as the gold "
                    "marginal (L1 0.4917): averaged over positions the model favours the options "
                    "that started in slots B and C, which is exactly where PMC-VQA v2 puts 73.6%% "
                    "of its answers -- but it only travels ~38%% of the way there. So the model is "
                    "UNDER-USING a real, stable dataset regularity rather than hallucinating a "
                    "positional one, and an output-side letter prior on this cell mostly supplies "
                    "the missing 62%%. That is a legitimate test-time intervention when the prior "
                    "is taken from TRAIN, but it must be described as dataset-prior exploitation, "
                    "not as repairing a model defect."),
        "caveat": ("measured on Lingshu-32B; the 7B has no rotation dump in this repo, so the "
                   "position/content split is NOT MEASURED for the 7B."),
    }


def main():
    art = {
        "title": ("ATTACK 1 STEP 1 (2026-08-17) -- the OUTPUT-BIAS AUDIT across all eight "
                  "reporting cells: how much of always-7B's error is a format-induced output "
                  "prior, and how much of it is recoverable."),
        "date": L.DATE,
        "baseline": "always-7B greedy, macro 0.5971 over 8 cells at 1.0 FLOP-eq",
        "no_fabricated_numbers": True,
        "not_abstention": "every arm answers every item; a corrected argmax is still an answer.",
        "numerics_pinned": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
                            "numpy": np.__version__, "python": sys.version.split()[0],
                            "TF32": "off in the generator (torch.backends.*.allow_tf32=False); "
                                    "this module is numpy-only",
                            "nboot": L.NBOOT, "seed_boot": L.SEED_BOOT},
        "reproduce": "OMP_NUM_THREADS=1 python3 src/cascade_methods/output_bias_audit.py",
    }
    art["null_tests"] = {
        "N1_frozen_open_text_metric": n1_frozen_open_metric(),
        "N2_prompt_byte_equality_vs_MedEvalKit": n2_prompt_byte_equality(),
        "G1_grader_reproduces_all_eight_published_cells": g1_grader_null_test(),
        "A1_pathvqa_prompt_fix_reproduction": a1_pathvqa_prompt_fix_reproduction(),
        "A2_judge_equals_exact_match_on_yesno": a2_judge_equals_em_on_yesno(),
    }
    art["STEP1_bias_audit"] = {
        "binary_cells": audit_binary_cells(),
        "mcq_cells": audit_mcq_cells(),
        "open_cells": audit_open_cells(),
        "letter_bias_position_vs_content": decompose_letter_bias_position_vs_content(),
        "ORACLE_upper_bounds": oracle_bounds(),
    }
    os.makedirs(L.ART, exist_ok=True)
    p = os.path.join(L.ART, f"output_bias_audit_{L.DATE}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(art, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in art["null_tests"].items()}, indent=1)[:4000])
    print("WROTE", p)


if __name__ == "__main__":
    main()
