#!/usr/bin/env python3
"""verifier_hparams_nulls.py -- KNOB 3 null tests + the max_tokens=64 truncation audit.

N1  the frozen metric reproduces from the stored deployed transfer dumps
N2  the EXACT identity  selected = oracle@8 x sel_eff   (never the additive form)
N3  the EM currency reproduces from the stored generation dumps, and the generation dumps'
    `preds` are byte-identical to the transfer dumps' `preds` (so judge and EM currencies are
    computed on the SAME pool in the SAME slot order)
N4  does the generator's max_tokens=64 (src/labeling/run_openvqa.py:64) ever bind?

CPU only.  python3 src/cascade_methods/verifier_hparams_nulls.py
"""
import json
import os
import re
import string
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G   # noqa: E402

GEN = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
GENFILE = {"slake_open": "ckpt_slake_open_lingshu7b_sc8.jsonl",
           "vqa_rad_open": "ckpt_vqa_rad_open_lingshu7b_sc8.jsonl",
           "pathvqa_open": "ckpt_pathvqa_open_lingshu7b_sc8.jsonl"}
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_verifier_hparams_parts")


# ---- run_openvqa.py's scorer, VERBATIM (lines 79-88) --------------------------------------
def em_norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def em_score(pred, gold):
    p, g = em_norm(pred), em_norm(gold)
    if not p:
        return 0
    if p == g:
        return 1
    if g and (g in p.split() or p in g.split() or g in p or p in g):
        return 1
    return 0


def load_gen():
    """{(ds, idx) -> generation record} from the DEPLOYED sc8 dumps."""
    out = {}
    for ds, f in GENFILE.items():
        p = os.path.join(GEN, f)
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                out[(ds, r["idx"])] = r
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {}
    items = G.load_items()

    # ---------------- N1 : the frozen metric -------------------------------------------
    inc = G.incumbent_scores()
    r = G.sel_eff(inc, items)
    exp = G.PUBLISHED
    meas = {"oracle@8": r["oracle"], "selected": r["acc"], "greedy": r["greedy"],
            "sel_eff": r["sel_eff"], "n": r["n"], "n_recoverable": r["n_recoverable"],
            "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in G.EVAL_DS}}
    dev = [abs(meas[k] - exp[k]) for k in ["oracle@8", "selected", "greedy", "sel_eff"]]
    dev += [abs(meas["per_ds"][d] - exp["per_ds"][d]) for d in G.EVAL_DS]
    dev += [abs(meas["n"] - exp["n"]), abs(meas["n_recoverable"] - exp["n_recoverable"])]
    rep["N1_frozen_metric"] = {
        "expected": {k: exp[k] for k in ["oracle@8", "selected", "greedy", "sel_eff", "n",
                                         "n_recoverable"]} | {"per_ds": exp["per_ds"]},
        "measured": meas, "max_abs_deviation": float(max(dev)),
        "verdict": "PASS" if max(dev) < 1e-5 else "FAIL",
        "_code": "src/training_methods/genframe_data.py sel_eff() on incumbent_scores()"}

    # ---------------- N2 : the EXACT identity ------------------------------------------
    ident = r["oracle"] * r["sel_eff"]
    add = r["greedy"] + r["sel_eff"] * (r["oracle"] - r["greedy"])
    rep["N2_identity"] = {
        "selected": r["acc"], "oracle@8_x_sel_eff": float(ident),
        "abs_deviation": float(abs(ident - r["acc"])),
        "verdict": "PASS" if abs(ident - r["acc"]) < 1e-12 else "FAIL",
        "the_FORBIDDEN_additive_form": float(add),
        "additive_form_error": float(add - r["acc"]),
        "_read": "selected = oracle@8 x sel_eff is EXACT. greedy + sel_eff*(oracle-greedy) "
                 "over-predicts and must never be used."}

    # ---------------- N3 : the EM currency is on the SAME pool -------------------------
    gen = load_gen()
    n_missing, n_slotmismatch, n_predmismatch = 0, 0, 0
    em_sl, gold_by = {}, {}
    for it in items:
        k = (it["ds"], it["idx"])
        g = gen.get(k)
        if g is None:
            n_missing += 1
            continue
        if len(g["preds"]) != len(it["preds"]):
            n_slotmismatch += 1
        if list(g["preds"]) != list(it["preds"]):
            n_predmismatch += 1
        gold_by[k] = g["gold"]
        em_sl[k] = [int(em_score(p, g["gold"])) for p in it["preds"]]
    rep["N3_em_currency_pairing"] = {
        "n_items": len(items), "n_missing_from_generation_dump": n_missing,
        "n_pool_length_mismatch": n_slotmismatch,
        "n_pred_string_mismatch_vs_transfer_dump": n_predmismatch,
        "verdict": "PASS" if (n_missing == 0 and n_slotmismatch == 0 and n_predmismatch == 0)
                   else "FAIL",
        "_what": "the EM currency is recomputed from the transfer dumps' OWN preds with "
                 "run_openvqa.py's scorer and the generation dump's gold, so judge-EM deltas "
                 "are on identical picks by construction.",
        "_source": "ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl"}

    # cross-check: our recomputed EM equals the stored `oks` slot-for-slot
    ndiff = 0
    for it in items:
        k = (it["ds"], it["idx"])
        if k in em_sl and list(em_sl[k]) != list(gen[k]["oks"]):
            ndiff += 1
    rep["N3_em_currency_pairing"]["n_items_where_recomputed_EM_differs_from_stored_oks"] = ndiff
    rep["N3_em_currency_pairing"]["verdict_vs_stored_oks"] = "PASS" if ndiff == 0 else "FAIL"

    # EM baselines on the frozen pool
    ems = np.array([em_sl[(it["ds"], it["idx"])] for it in items], dtype=int)
    judge = np.array([it["sl"] for it in items], dtype=int)
    rep["EM_vs_JUDGE_on_the_frozen_pool"] = {
        "n_items": len(items), "n_slots": int(ems.size),
        "judge_oracle@8": float((judge.max(axis=1) == 1).mean()),
        "em_oracle@8": float((ems.max(axis=1) == 1).mean()),
        "judge_slot_positive_rate": float(judge.mean()),
        "em_slot_positive_rate": float(ems.mean()),
        "slot_agreement": float((ems == judge).mean()),
        "judge_yes_em_no_slots": int(((judge == 1) & (ems == 0)).sum()),
        "em_yes_judge_no_slots": int(((ems == 1) & (judge == 0)).sum()),
        "_read": "the judge is more lenient than exact match by construction (its prompt "
                 "instructs leniency about phrasing and synonyms). BOTH currencies are "
                 "reported for every endpoint in this round."}
    np.savez_compressed(os.path.join(OUT, "em_slots.npz"), em=ems, judge=judge)

    # ---------------- N4 : does max_tokens=64 bind? ------------------------------------
    # (a) the stored per-item gen_tokens (slot 0 only in this dump version)
    gt = [gen[(it["ds"], it["idx"])].get("gen_tokens") for it in items]
    gt = [x for x in gt if x is not None]
    gta = [gen[(it["ds"], it["idx"])].get("gen_tokens_all") for it in items]
    gta_flat = [x for r_ in gta if r_ for x in r_]
    # (b) tokenizer-exact check over ALL 18,760 candidate strings
    tok_hist, n64, n_ge60, maxtok = {}, 0, 0, 0
    try:
        from transformers import AutoTokenizer
        os.environ.setdefault("HF_HOME", "/data/dan/hf_cache")
        tk = AutoTokenizer.from_pretrained("lingshu-medical-mllm/Lingshu-7B")
        per_ds = {}
        for it in items:
            for p in it["preds"]:
                n = len(tk.encode(p, add_special_tokens=False))
                maxtok = max(maxtok, n)
                tok_hist[n] = tok_hist.get(n, 0) + 1
                if n >= 64:
                    n64 += 1
                if n >= 60:
                    n_ge60 += 1
                per_ds.setdefault(it["ds"], []).append(n)
        lens = np.array([n for v in per_ds.values() for n in v], float)
        rep["N4_max_tokens_truncation_audit"] = {
            "_lever": "src/labeling/run_openvqa.py:64  ap.add_argument('--max_tokens', "
                      "type=int, default=64); the deployed open arm uses the default.",
            "_method": "every one of the 18,760 stored candidate strings re-tokenized with the "
                       "Lingshu-7B tokenizer (add_special_tokens=False). A generation stopped by "
                       "the 64-token budget must tokenize to ~64; run_openvqa.extract() only "
                       ".strip()s, so the count is exact up to trailing whitespace.",
            "n_candidate_strings": int(lens.size),
            "n_at_or_above_64_tokens": int(n64),
            "n_at_or_above_60_tokens": int(n_ge60),
            "max_tokens_observed": int(maxtok),
            "mean_tokens": float(lens.mean()), "median_tokens": float(np.median(lens)),
            "p99_tokens": float(np.percentile(lens, 99)),
            "per_ds_mean": {d: float(np.mean(v)) for d, v in per_ds.items()},
            "per_ds_max": {d: int(np.max(v)) for d, v in per_ds.items()},
            "truncation_rate": float(n64 / lens.size),
            "stored_gen_tokens_slot0": {"n": len(gt), "mean": float(np.mean(gt)),
                                        "max": int(np.max(gt)),
                                        "n_at_64": int(sum(1 for x in gt if x >= 64))},
            "stored_gen_tokens_all_slots": ({"n": len(gta_flat), "mean": float(np.mean(gta_flat)),
                                             "max": int(np.max(gta_flat)),
                                             "n_at_64": int(sum(1 for x in gta_flat if x >= 64))}
                                            if gta_flat else
                                            {"_note": "this dump version stores gen_tokens for "
                                                      "slot 0 only; the tokenizer pass above "
                                                      "covers all 8 slots"}),
        }
    except Exception as e:
        rep["N4_max_tokens_truncation_audit"] = {"_error": f"{type(e).__name__}: {e}"}

    json.dump(rep, open(os.path.join(OUT, "nulls.json"), "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
