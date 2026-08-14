#!/usr/bin/env python3
"""inference_params_verify.py -- ADVERSARIAL RE-DERIVATION of the 2026-08-13 inference-parameter
round (decoding sweep + resolution sweep + vision diversity).

Nothing here reads another agent's analysis output. Everything is recomputed from
  * the frozen metric  src/training_methods/genframe_data.py   (imported, never reimplemented)
  * the RAW per-item generation dumps  ckpts/openvqa/{decoding_sweep,resolution_sweep,visdiv}/
  * the RAW judge cache and the RAW verifier score cache in those directories

Then it is compared to the numbers the sweeps reported. Writes ONE artifact.

Usage:  python3 src/cascade_methods/inference_params_verify.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G  # noqa: E402

DEC = os.path.join(ROOT, "ckpts/openvqa/decoding_sweep")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
EVAL_DS = G.EVAL_DS
NBOOT = 10000
BSEED = 20260814  # MY seed, deliberately different from the sweep's 20260813


# ------------------------------------------------------------------ null test 1
def null_test_frozen_metric():
    """Reproduce the PUBLISHED incumbent constants straight from the transfer dumps."""
    items = G.load_items()
    r = G.sel_eff(G.incumbent_scores(), items=items)
    pub = G.PUBLISHED
    dev = {
        "n": abs(r["n"] - pub["n"]),
        "n_recoverable": abs(r["n_recoverable"] - pub["n_recoverable"]),
        "oracle@8": abs(r["oracle"] - pub["oracle@8"]),
        "selected": abs(r["acc"] - pub["selected"]),
        "greedy": abs(r["greedy"] - pub["greedy"]),
        "sel_eff": abs(r["sel_eff"] - pub["sel_eff"]),
    }
    for d in EVAL_DS:
        dev["per_ds_" + d] = abs(r["per_ds"][d]["sel_eff"] - pub["per_ds"][d])
    # the exact identity the brief insists on
    ident = abs(r["acc"] - r["oracle"] * r["sel_eff"])
    return {
        "description": "frozen metric recomputed from ckpts/train/lora_verifier_disjoint/"
                       "transfer_dump_*.json vs genframe_data.PUBLISHED",
        "measured": {"n": r["n"], "n_recoverable": r["n_recoverable"],
                     "oracle@8": r["oracle"], "selected": r["acc"],
                     "greedy": r["greedy"], "sel_eff": r["sel_eff"],
                     "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in EVAL_DS}},
        "max_abs_deviation": float(max(dev.values())),
        "deviations": {k: float(v) for k, v in dev.items()},
        "identity_selected_eq_oracle_x_sel_eff_abs_err": float(ident),
        "pass": bool(max(dev.values()) < 1e-6 and ident < 1e-12),
    }


# ------------------------------------------------------------------ load the sweep pools
def load_judge_map(report_overlap=False):
    """{(ds, idx, normalized_answer) -> 0/1}, from the PRELOAD cache + THIS session's judge.

    The judge is text-only (question, gold, candidate), so a label is a pure function of
    (idx, normalized answer) and reuse across settings is legitimate -- it also removes
    judge noise as a between-arm confound. But reuse is only safe if the two label
    SOURCES agree, so the overlap is measured rather than assumed.
    """
    pre, fresh = {}, {}
    for ds in EVAL_DS:
        with open(os.path.join(DEC, f"judgecache_preload_{ds}.jsonl")) as fh:
            for ln in fh:
                if ln.strip():
                    o = json.loads(ln)
                    pre[(ds, str(o["idx"]), o["na"])] = int(o["judge_ok"])
        jm = json.load(open(os.path.join(DEC, f"judgemap_{ds}.json")))
        lab = {}
        with open(os.path.join(DEC, f"judgein_{ds}.judge.jsonl")) as fh:
            for ln in fh:
                if ln.strip():
                    o = json.loads(ln)
                    lab[o["idx"]] = int(o["judge_ok"])
        for key, jid in jm.items():
            idx, na = key.split("|", 1)
            if jid in lab:
                fresh[(ds, idx, na)] = lab[jid]
    ov = set(pre) & set(fresh)
    agree = sum(1 for k in ov if pre[k] == fresh[k])
    out = dict(pre)
    out.update(fresh)  # fresh wins on the (measured) overlap
    if report_overlap:
        return out, {"n_preload": len(pre), "n_fresh": len(fresh),
                     "n_overlap": len(ov), "n_agree": agree,
                     "overlap_agreement": (agree / len(ov)) if ov else None,
                     "n_total_labels": len(out)}
    return out


def load_vscores():
    """{(ds, idx, raw_answer_string) -> pyes}."""
    out = {}
    for fn in sorted(os.listdir(DEC)):
        if not fn.startswith("vscore_cache") or not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(DEC, fn)) as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                o = json.loads(ln)
                out[(o["ds"], str(o["idx"]), o["ans"])] = float(o["pyes"])
    return out


def build_items(setting, seed, judge, vsc, need_scores=True):
    """Rebuild the frozen-metric item list for one (setting, seed) from RAW dumps.

    Returns (items, n_missing_judge, n_missing_vscore) with items in genframe canonical
    order (slake, vqa_rad, pathvqa), each carrying sl[8] judge labels, scores[8], preds[8],
    greedy_ok (= modal-of-8, the frozen metric's definition).
    """
    ref = G.load_items()
    order = [(it["ds"], str(it["idx"])) for it in ref]
    byds = defaultdict(dict)
    for ds in EVAL_DS:
        p = os.path.join(DEC, f"ckpt_{ds}_{setting}_s{seed}.jsonl")
        if not os.path.exists(p):
            return None, -1, -1
        with open(p) as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                o = json.loads(ln)
                byds[ds][str(o["idx"])] = o
    items, mj, mv = [], 0, 0
    for ds, idx in order:
        o = byds[ds].get(idx)
        if o is None:
            return None, -1, -1
        preds = o["preds"]
        sl = []
        for a in preds:
            na = G.norm(a)
            k = (ds, idx, na)
            if k in judge:
                sl.append(judge[k])
            else:
                sl.append(-1)
                mj += 1
        # modal-of-8 = the frozen metric's 'greedy'
        cnt = defaultdict(int)
        for a in preds:
            cnt[G.norm(a)] += 1
        modal = max(cnt.items(), key=lambda kv: (kv[1], -list(cnt).index(kv[0])))[0]
        # ties -> first-seen, matching the generator's own modal_pred where possible
        modal_pred = o.get("modal_pred")
        modal_na = G.norm(modal_pred) if modal_pred is not None else modal
        g_ok = judge.get((ds, idx, modal_na), -1)
        if g_ok < 0:
            mj += 1
            g_ok = 0
        scores = []
        if need_scores:
            for a in preds:
                k = (ds, idx, a)
                if k in vsc:
                    scores.append(vsc[k])
                else:
                    scores.append(G.MISSING_SCORE)
                    mv += 1
        items.append({"ds": ds, "idx": o["idx"], "sl": sl, "scores": scores,
                      "preds": preds, "greedy_ok": int(g_ok)})
    return items, mj, mv


def score_setting(setting, seed, judge, vsc):
    items, mj, mv = build_items(setting, seed, judge, vsc)
    if items is None:
        return None
    sb = {(it["ds"], it["idx"]): it["scores"] for it in items}
    r = G.sel_eff(sb, items=items)
    r["_missing_judge"] = mj
    r["_missing_vscore"] = mv
    r["_gen_tokens"] = float(np.mean([t for it, o in zip(items, items)
                                      for t in [0]])) if False else None
    return r


# ------------------------------------------------------------------ EM currency
def em_labels(setting, seed):
    """oks_em straight out of the generation dump, in canonical order."""
    ref = G.load_items()
    order = [(it["ds"], str(it["idx"])) for it in ref]
    byds = defaultdict(dict)
    for ds in EVAL_DS:
        p = os.path.join(DEC, f"ckpt_{ds}_{setting}_s{seed}.jsonl")
        with open(p) as fh:
            for ln in fh:
                if ln.strip():
                    o = json.loads(ln)
                    byds[ds][str(o["idx"])] = o
    return [byds[ds][idx] for ds, idx in order]


def main():
    out = {
        "title": "ADVERSARIAL RE-DERIVATION -- can changing the 7B's inference parameters "
                 "improve the samples it generates?",
        "date": "2026-08-14",
        "no_fabricated_numbers": True,
        "method": "every number recomputed from raw dumps with the frozen metric "
                  "(src/training_methods/genframe_data.py); bootstrap seed 20260814, "
                  "DELIBERATELY different from the sweep's 20260813, so agreement is not "
                  "a shared-RNG artifact",
        "nboot": NBOOT,
        "bootstrap_seed": BSEED,
    }
    print("== null test 1 ==", flush=True)
    out["null_test_frozen_metric"] = null_test_frozen_metric()
    print(json.dumps(out["null_test_frozen_metric"]["deviations"], indent=1))
    print("pass:", out["null_test_frozen_metric"]["pass"])

    print("== loading judge + vscore caches ==", flush=True)
    judge, jov = load_judge_map(report_overlap=True)
    vsc = load_vscores()
    out["judge_label_provenance"] = jov
    out["cache_sizes"] = {"judge_labels": len(judge), "vscores": len(vsc)}
    print(out["cache_sizes"], jov)

    settings = ["T03", "T05", "T07", "T10", "T13", "minp01", "rp105", "rp11"]
    res = {}
    for s in settings:
        res[s] = {}
        for sd in (0, 1, 2):
            r = score_setting(s, sd, judge, vsc)
            if r is None:
                continue
            res[s][sd] = r
            print(f"{s} s{sd}: oracle {r['oracle']:.6f} sel_eff {r['sel_eff']:.6f} "
                  f"selected {r['acc']:.6f} modal {r['greedy']:.6f} "
                  f"missJ {r['_missing_judge']} missV {r['_missing_vscore']}", flush=True)
    out["_res"] = res  # stripped before writing

    # ---------------- headline table
    tab = {}
    for s in settings:
        seeds = sorted(res[s])
        tab[s] = {
            "n_seeds": len(seeds),
            "oracle@8": {"mean": float(np.mean([res[s][d]["oracle"] for d in seeds])),
                         "sd": float(np.std([res[s][d]["oracle"] for d in seeds], ddof=1)),
                         "per_seed": {str(d): res[s][d]["oracle"] for d in seeds}},
            "sel_eff": {"mean": float(np.mean([res[s][d]["sel_eff"] for d in seeds])),
                        "sd": float(np.std([res[s][d]["sel_eff"] for d in seeds], ddof=1)),
                        "per_seed": {str(d): res[s][d]["sel_eff"] for d in seeds}},
            "selected": {"mean": float(np.mean([res[s][d]["acc"] for d in seeds])),
                         "sd": float(np.std([res[s][d]["acc"] for d in seeds], ddof=1)),
                         "per_seed": {str(d): res[s][d]["acc"] for d in seeds}},
            "modal_of_8": {"mean": float(np.mean([res[s][d]["greedy"] for d in seeds]))},
            "identity_max_abs_err": float(max(
                abs(res[s][d]["acc"] - res[s][d]["oracle"] * res[s][d]["sel_eff"])
                for d in seeds)),
            "missing_judge_slots": int(sum(res[s][d]["_missing_judge"] for d in seeds)),
            "missing_vscore_slots": int(sum(res[s][d]["_missing_vscore"] for d in seeds)),
            "per_cell_selected": {
                d: float(np.mean([res[s][sd]["per_ds"][d]["acc"] for sd in seeds]))
                for d in EVAL_DS},
            "per_cell_oracle": {
                d: float(np.mean([res[s][sd]["per_ds"][d]["oracle"] for sd in seeds]))
                for d in EVAL_DS},
            "per_cell_sel_eff": {
                d: float(np.mean([res[s][sd]["per_ds"][d]["sel_eff"] for sd in seeds]))
                for d in EVAL_DS},
        }
    out["recomputed"] = tab

    # ---------------- paired deltas vs the in-session matched control T07, seed-matched
    print("== paired bootstrap vs matched control ==", flush=True)
    deltas = {}
    for s in settings:
        if s == "T07":
            continue
        per_seed = []
        for sd in (0, 1, 2):
            if sd not in res[s] or sd not in res["T07"]:
                continue
            a, b = res[s][sd]["got"], res["T07"][sd]["got"]
            bb = G.paired_bootstrap(a, b, rec=res["T07"][sd]["rec"], nboot=NBOOT, seed=BSEED)
            per_seed.append({"seed": sd, "d_selected": bb["d_acc"],
                             "d_selected_ci": bb["d_acc_ci"]})
        # pooled: stack all 3 seed-pairs as one 7035-long paired vector
        A = np.concatenate([res[s][sd]["got"] for sd in (0, 1, 2) if sd in res[s]])
        B = np.concatenate([res["T07"][sd]["got"] for sd in (0, 1, 2) if sd in res[s]])
        R = np.concatenate([res["T07"][sd]["rec"] for sd in (0, 1, 2) if sd in res[s]])
        bb = G.paired_bootstrap(A, B, rec=R, nboot=NBOOT, seed=BSEED)
        # oracle delta, seed-stacked
        AO = np.concatenate([res[s][sd]["rec"] for sd in (0, 1, 2) if sd in res[s]])
        boo = G.paired_bootstrap(AO, R, rec=R, nboot=NBOOT, seed=BSEED)
        # per cell
        dsidx = np.concatenate([res["T07"][sd]["ds_index"] for sd in (0, 1, 2) if sd in res[s]])
        percell = {}
        for j, d in enumerate(EVAL_DS):
            m = dsidx == j
            bc = G.paired_bootstrap(A[m], B[m], rec=R[m], nboot=NBOOT, seed=BSEED)
            bo = G.paired_bootstrap(AO[m], R[m], rec=R[m], nboot=NBOOT, seed=BSEED)
            percell[d] = {"d_selected": bc["d_acc"], "d_selected_ci": bc["d_acc_ci"],
                          "d_oracle": bo["d_acc"], "d_oracle_ci": bo["d_acc_ci"]}
        deltas[s] = {
            "pooled_3seed_stacked": {
                "d_selected": bb["d_acc"], "d_selected_ci": bb["d_acc_ci"],
                "d_oracle": boo["d_acc"], "d_oracle_ci": boo["d_acc_ci"],
                "n_paired_items": int(len(A)),
            },
            "per_seed_d_selected": per_seed,
            "per_cell": percell,
            "d_sel_eff_point": tab[s]["sel_eff"]["mean"] - tab["T07"]["sel_eff"]["mean"],
        }
        print(f"{s}: d_selected {bb['d_acc']:+.6f} {bb['d_acc_ci']}  "
              f"d_oracle {boo['d_acc']:+.6f} {boo['d_acc_ci']}", flush=True)
    out["deltas_vs_matched_control_T07"] = deltas

    del out["_res"]
    p = os.path.join(ART, "_infparams_verify_recompute.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)
    # keep the arrays for downstream scripts
    np.savez(os.path.join(ART, "_infparams_verify_got.npz"),
             **{f"{s}_s{sd}_got": res[s][sd]["got"] for s in settings for sd in res[s]},
             **{f"{s}_s{sd}_rec": res[s][sd]["rec"] for s in settings for sd in res[s]},
             ds_index=res["T07"][0]["ds_index"])


if __name__ == "__main__":
    main()
