#!/usr/bin/env python3
"""resolution_null_tests.py -- SWEEP 2: the null tests, run before any new number is trusted.

Four independent nulls, each with its max absolute deviation stated:

  N1  FROZEN METRIC.  Recompute the endpoint from the stored deployed transfer dumps with
      src/training_methods/genframe_data.py and compare to the published cells.  Deviation must
      be 0 -- this proves the metric this round uses is the metric the published number used.
  N2  PUBLISHED MCQ CELLS.  Recompute the 5 MCQ macro-8 cells from MedEvalKit's own per-item
      dumps at the default resolution and compare to the values the project reports.
  N3  GENERATION PATH.  This round's own temperature-0 arm at the deployed cap320 against the
      stored deployed greedy dump, item by item -- how much of a difference a fresh serving
      config makes with NO experimental variable changed (the +-0.008 caveat, measured).
  N4  VERIFIER RE-SCORE.  Written by resolution_verifier_score.py --null_test; folded in here.

    python3 src/cascade_methods/resolution_null_tests.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
P = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
SWEEP = os.path.join(ROOT, "ckpts/openvqa/resolution_sweep")
os.makedirs(P, exist_ok=True)

PUBLISHED_OPEN = {"oracle@8": 0.626013, "selected": 0.485288, "greedy": 0.449467,
                  "sel_eff": 0.775204, "n": 2345, "n_recoverable": 1468,
                  "per_ds": {"slake_open": 0.850088, "vqa_rad_open": 0.761905,
                             "pathvqa_open": 0.722581}}
# the 5 MCQ cells of the macro-8 as CLAUDE.md / the task brief report them for Lingshu-7B direct
PUBLISHED_MCQ_7B = {"PMC_VQA": 0.5427, "SLAKE_closed": 0.8254, "VQA_RAD_closed": 0.7809,
                    "PATH_VQA_closed": 0.8409, "MedXpertQA-MM": 0.2615}
PUBLISHED_MCQ_32B = {"PMC_VQA": 0.5518, "SLAKE_closed": 0.8589, "VQA_RAD_closed": 0.8526,
                     "PATH_VQA_closed": 0.8891, "MedXpertQA-MM": 0.3065}


def n1():
    from src.training_methods import genframe_data as G
    ev = G.load_candidates("eval", layers=[], pooling=())
    r = G.sel_eff({(q.ds, q.idx): q.inc_scores for q in ev.questions})
    got = {"oracle@8": r["oracle"], "selected": r["acc"], "greedy": r["greedy"],
           "sel_eff": r["sel_eff"], "n": r["n"], "n_recoverable": r["n_recoverable"],
           "per_ds": {k: v["sel_eff"] for k, v in r["per_ds"].items()}}
    dev = [abs(got[k] - PUBLISHED_OPEN[k]) for k in
           ["oracle@8", "selected", "greedy", "sel_eff"]]
    dev += [abs(got["per_ds"][k] - PUBLISHED_OPEN["per_ds"][k]) for k in PUBLISHED_OPEN["per_ds"]]
    dev += [abs(got["n"] - PUBLISHED_OPEN["n"]), abs(got["n_recoverable"] - PUBLISHED_OPEN["n_recoverable"])]
    return {"what": "frozen open-text endpoint recomputed from the stored deployed transfer dumps "
                    "with src/training_methods/genframe_data.py",
            "expected": PUBLISHED_OPEN, "measured": got,
            "max_abs_deviation": float(max(dev)),
            "verdict": "PASS" if max(dev) < 1e-6 else "FAIL"}


def n2():
    base = os.path.join(ROOT, "MedEvalKit", "eval_results_lingshu7b_full", "{}")
    b32 = os.path.join(ROOT, "MedEvalKit", "eval_results_lingshu32b_full", "{}")
    spec = [("PMC_VQA", "PMC_VQA", None), ("SLAKE_closed", "SLAKE", "slake"),
            ("VQA_RAD_closed", "VQA_RAD", "yn"), ("PATH_VQA_closed", "PATH_VQA", "yn"),
            ("MedXpertQA-MM", "MedXpertQA-MM", None)]
    out = {}
    for tag, root, pub in [("lingshu7b", base, PUBLISHED_MCQ_7B), ("lingshu32b", b32, PUBLISHED_MCQ_32B)]:
        got, dev = {}, []
        for cell, ds, mode in spec:
            p = os.path.join(root, ds, "results.json")
            if not os.path.exists(p):
                continue
            rs = json.load(open(p))
            if mode == "slake":
                rs = [r for r in rs if r.get("answer_type") == "CLOSED"]
            elif mode == "yn":
                rs = [r for r in rs if str(r.get("answer")).strip().lower() in ("yes", "no")]
            got[cell] = {"n": len(rs), "acc": round(float(np.mean([bool(r["correct"]) for r in rs])), 6)}
            dev.append(abs(got[cell]["acc"] - pub[cell]))
        out[tag] = {"expected_published": pub, "measured": got,
                    "max_abs_deviation": round(float(max(dev)), 6) if dev else None,
                    "verdict": "PASS" if dev and max(dev) < 5e-5 else "CHECK"}
    out["_what"] = ("the 5 MCQ macro-8 cells recomputed from MedEvalKit's own per-item dumps at "
                    "CAP_MAX_PIXELS unset (max_pixels 12,845,056). Reproducing the published cells "
                    "to <5e-5 is what PINS the published arms to that resolution.")
    return out


def n3():
    """this round's cap320 t0 arm vs the stored deployed greedy dump, item by item."""
    DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
    dep = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
    out = {"per_cell": {}}
    tot_n = tot_same = 0
    accs = []
    for ds in DS:
        f_new = os.path.join(SWEEP, f"ckpt_{ds}_cap320_t0.jsonl")
        f_old = os.path.join(dep, f"ckpt_{ds}_lingshu7b.jsonl")
        if not (os.path.exists(f_new) and os.path.exists(f_old)):
            out["per_cell"][ds] = "arm not generated yet"
            continue
        new = {json.loads(l)["idx"]: json.loads(l) for l in open(f_new) if l.strip()}
        old = {json.loads(l)["idx"]: json.loads(l) for l in open(f_old) if l.strip()}
        ks = [k for k in new if k in old]
        same = sum(1 for k in ks
                   if str(new[k]["modal_pred"]).strip().lower()
                   == str(old[k]["modal_pred"]).strip().lower())
        jf = os.path.join(dep, f"ckpt_{ds}_lingshu7b.judge.jsonl")
        lab = {r["idx"]: r["judge_ok"] for r in
               (json.loads(l) for l in open(jf) if l.strip())} if os.path.exists(jf) else {}
        acc_old = float(np.mean([lab[k] for k in ks if k in lab])) if lab else None
        out["per_cell"][ds] = {"n_compared": len(ks), "identical_answer_string": same,
                               "agreement_rate": round(same / max(1, len(ks)), 6),
                               "stored_deployed_greedy_judge_acc": acc_old}
        tot_n += len(ks)
        tot_same += same
        if acc_old is not None:
            accs.append(acc_old)
    out["pooled_agreement_rate"] = round(tot_same / max(1, tot_n), 6)
    out["_what"] = ("this session's temperature-0 arm at the DEPLOYED cap320 against the stored "
                    "deployed greedy dump. Both are run_openvqa.py's prompt, loaders and scorer at "
                    "max_pixels 250,880; they differ only in vLLM version and image-processor "
                    "backend (fast vs slow). The disagreement rate is the size of the serving-config "
                    "shift with NO experimental variable changed, and it is why every cap-vs-cap "
                    "delta in this round is taken against a control generated in THIS session.")
    return out


def n5():
    """What the published open pool's 'greedy' row actually is.

    verifier_transfer_eval.py:g takes `sc[i]["modal_pred"]` from the sc8 file -- the MODAL answer
    of the 8 T=0.7 samples -- so the 0.449467 / 0.7364 / 0.4650 / 0.3240 row is a modal-of-8
    number, not a temperature-0 decode. The temperature-0 decode exists on disk with its own judge
    labels (ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.jsonl + .judge.jsonl) and is a
    DIFFERENT number. Reported so this round never conflates the two.
    """
    DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
    dep = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
    dumps = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")
    nm = {"slake_open": "slake", "vqa_rad_open": "vqa_rad", "pathvqa_open": "pathvqa"}
    out = {"per_cell": {}}
    tot_t0, tot_modal, tot_n = 0.0, 0.0, 0
    for ds in DS:
        d = json.load(open(os.path.join(dumps, f"transfer_dump_{nm[ds]}_open_lingshu7b.json")))
        ids = [r["idx"] for r in d]
        modal = float(np.mean([r["greedy_ok"] for r in d]))
        jf = os.path.join(dep, f"ckpt_{ds}_lingshu7b.judge.jsonl")
        lab = {r["idx"]: r["judge_ok"] for r in
               (json.loads(l) for l in open(jf) if l.strip())}
        t0 = float(np.mean([lab.get(i, 0) for i in ids]))
        out["per_cell"][ds] = {"n": len(ids),
                               "published_row_called_greedy_is_modal_of_8": round(modal, 6),
                               "true_temperature_0_decode": round(t0, 6),
                               "difference": round(t0 - modal, 6)}
        tot_t0 += t0 * len(ids)
        tot_modal += modal * len(ids)
        tot_n += len(ids)
    out["pooled"] = {"n": tot_n,
                     "published_row_called_greedy_is_modal_of_8": round(tot_modal / tot_n, 6),
                     "true_temperature_0_decode": round(tot_t0 / tot_n, 6),
                     "difference": round((tot_t0 - tot_modal) / tot_n, 6)}
    out["_read"] = ("this is a NAMING fact, not an error in the endpoint: sel_eff is a conditional "
                    "mean over recoverable pools and does not use the greedy row at all, and "
                    "selected = oracle@8 x sel_eff is unaffected. But 'greedy' in the published "
                    "open cells is a best-of-8 quantity, so it must not be quoted as the model's "
                    "single-decode accuracy. This round reports BOTH, under separate names "
                    "(greedy_t0 and pool_modal).")
    return out


def main():
    res = {}
    try:
        res["N1_frozen_metric"] = n1()
    except Exception as e:
        res["N1_frozen_metric"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        res["N2_published_mcq_cells"] = n2()
    except Exception as e:
        res["N2_published_mcq_cells"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        res["N3_generation_path"] = n3()
    except Exception as e:
        res["N3_generation_path"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        res["N5_what_greedy_means_in_the_published_open_pool"] = n5()
    except Exception as e:
        res["N5_what_greedy_means_in_the_published_open_pool"] = {"error": f"{type(e).__name__}: {e}"}
    vn = os.path.join(SWEEP, "verifier_null_test.json")
    if os.path.exists(vn):
        res["N4_verifier_rescore"] = json.load(open(vn))
        res["N4_verifier_rescore"]["_what"] = (
            "randomly chosen stored (item, candidate) pairs re-scored with the deployed clean "
            "disjoint LoRA at its deployed max_pixels 1,003,520, compared to the score stored in "
            "the transfer dump.")
    else:
        res["N4_verifier_rescore"] = "not run yet"
    json.dump(res, open(os.path.join(P, "null_tests.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
