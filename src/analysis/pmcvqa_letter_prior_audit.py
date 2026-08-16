"""Zero-GPU audit of the LIVE PMC-VQA cell's answer-letter prior and its calibration residual.

Reads only the existing MedEvalKit greedy dump. No model is loaded. Run from the repo root:
    python3 src/analysis/pmcvqa_letter_prior_audit.py
Writes results/cascade_methods/artifacts/pmcvqa_letter_prior_2026-08-17.json
"""
import json, re, collections, os

SRC = "MedEvalKit/eval_results_lingshu7b_full/{}/PMC_VQA/results.json"
OUT = "results/cascade_methods/artifacts/pmcvqa_letter_prior_2026-08-17.json"
PUBLISHED_ACC = 0.5427
L = "ABCD"


def parse_letter(resp):
    m = re.match(r"\s*([ABCD])\b", resp or "")
    return m.group(1) if m else None


def main():
    rows = json.load(open(SRC))
    n = len(rows)
    acc = sum(1 for r in rows if r["correct"]) / n
    gold = collections.Counter(r["answer"] for r in rows)
    pred = collections.Counter(parse_letter(r["response"]) for r in rows)

    per_letter = {}
    for x in L:
        sel = [r for r in rows if parse_letter(r["response"]) == x]
        tp = sum(1 for r in sel if r["answer"] == x)
        per_letter[x] = {
            "gold_rate": gold[x] / n,
            "pred_rate": len(sel) / n,
            "n_pred": len(sel),
            "precision": tp / len(sel) if sel else None,
            "recall": tp / gold[x] if gold[x] else None,
            "mean_conf_when_predicted": sum(r["conf"] for r in sel) / len(sel) if sel else None,
            "gold_given_pred": {
                y: sum(1 for r in sel if r["answer"] == y) / len(sel) for y in L
            } if sel else None,
        }

    precs = [per_letter[x]["precision"] for x in L]
    confs = [per_letter[x]["mean_conf_when_predicted"] for x in L]
    prec_range = max(precs) - min(precs)
    conf_range = max(confs) - min(confs)

    selA = per_letter["A"]["gold_given_pred"]
    crude_delta = (selA["C"] - selA["A"]) * per_letter["A"]["pred_rate"]

    out = {
        "title": "LIVE PMC-VQA (test_2.csv track) answer-letter prior and calibration residual",
        "no_fabricated_numbers": True,
        "source_dump": SRC,
        "gpu_used": False,
        "null_test": {
            "n": n,
            "acc_recomputed": acc,
            "published_always_7b": PUBLISHED_ACC,
            "abs_dev_vs_published": abs(acc - PUBLISHED_ACC),
            "unparsed_responses": pred[None],
            "passed": abs(acc - PUBLISHED_ACC) < 1e-3,
        },
        "per_letter": per_letter,
        "gold_B_plus_C": (gold["B"] + gold["C"]) / n,
        "pred_B_plus_C": (pred["B"] + pred["C"]) / n,
        "constant_guesser_acc": {x: gold[x] / n for x in L},
        "calibration_residual": {
            "precision_range_across_predicted_letters": prec_range,
            "mean_conf_range_across_predicted_letters": conf_range,
            "ratio": prec_range / conf_range,
            "reading": "confidence is not a sufficient statistic for correctness on this cell; "
                       "the residual is the predicted answer letter",
        },
        "label_level_prior_shift_is_dead": {
            "status": "DIAGNOSTIC, evaluated on eval gold; NOT cross-fit, NOT a claimed method",
            "gold_given_pred_A": selA,
            "reading": "gold is near-uniform given pred=A, so A-predictions are near-uninformative "
                       "rather than systematically mis-signed",
            "crude_rule_reroute_all_predA_to_C_cell_delta": crude_delta,
            "macro_contribution_one_eighth": crude_delta / 8,
        },
        "caveat_wrong_track_dump": {
            "file": "ckpts/gate_lingshu7b_mcq/ckpt_PMC-VQA_nothink_norag.jsonl",
            "note": "has opt_logprobs but is n=500, gold B+C=0.6100, acc=0.6040 -- NOT the live cell",
        },
        "live_dump_lacks_option_logprobs": {
            "keys_present": sorted(rows[0].keys()),
            "note": "conf (top-1 prob) and margin (top1-top2) only; any contextual/batch calibration "
                    "or PriDe test needs a re-scoring pass over all items",
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("wrote", OUT)
    print("null test passed:", out["null_test"]["passed"],
          "| abs dev %.2e" % out["null_test"]["abs_dev_vs_published"])
    print("precision/conf range ratio: %.2fx" % out["calibration_residual"]["ratio"])
    print("label-level prior shift macro contribution: %+.5f"
          % out["label_level_prior_shift_is_dead"]["macro_contribution_one_eighth"])


if __name__ == "__main__":
    main()
