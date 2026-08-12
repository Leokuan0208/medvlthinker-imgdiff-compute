#!/usr/bin/env python3
"""cheapleg_probe.py -- ATTACK B's primary measurement: what does training the GENERATOR move?

For each of the 8 Variant-B reporting cells, and for BOTH arms (frozen Lingshu-7B control vs the
LoRA-adapted cheap leg, generated in the SAME serving configuration), report

    greedy accuracy                      -- the cheap leg's own accuracy
    p10 = P(7B right AND 32B-direct wrong)-- the FREE UPPER BOUND on any 7B-vs-32B router
    p01 = P(7B wrong AND 32B-direct right)-- what escalation can still buy
    oracle@8, selected, sel_eff           -- open cells only, under the FROZEN incumbent verifier
                                             (ckpts/train/lora_verifier_disjoint), THE frozen metric
                                             src/training_methods/genframe_data.sel_eff

with a PAIRED item-level bootstrap (nboot=10000) on every adapted-minus-base delta, per cell, plus
the guardrail (never worse on any single cell).

The two arms are paired item-by-item: the MCQ cells are the same MedEvalKit rows in the same order,
and the open cells are the same question ids.  Items present in only one arm are dropped and counted.

    python3 src/cascade_methods/cheapleg_probe.py \
        --base_mek cheapleg_base7b   --base_open  ckpts/cheapleg/scores_base7b \
        --adapt_mek cheapleg_adapt7b_s0 --adapt_open ckpts/cheapleg/scores_adapt7b_s0
"""
import argparse, json, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.join(os.path.dirname(_HERE), "training_methods")):
    if p not in sys.path:
        sys.path.insert(0, p)

import genframe_data as G               # noqa: E402  THE frozen open-text metric

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
MEK = os.path.join(ROOT, "MedEvalKit")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
NBOOT = 10000
SEED = 20260811

MCQ_CELLS = [("PMC_VQA", "PMC_VQA", None), ("SLAKE_closed", "SLAKE", "SLAKE"),
             ("VQA_RAD_closed", "VQA_RAD", "YESNO"), ("PATH_VQA_closed", "PATH_VQA", "YESNO"),
             ("MedXpertQA-MM", "MedXpertQA-MM", None)]
OPEN_CELLS = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open",
              "PATH_VQA_open": "pathvqa_open"}


def as_ok(r):
    v = r.get("correct")
    return int(v is True or str(v).strip().lower() in ("true", "1"))


def load_mek(tag, ds):
    p = f"{MEK}/eval_results_{tag}/{{}}/{ds}/results.json"
    return json.load(open(p)) if os.path.exists(p) else None


def mcq_vectors(tag, ds, closed):
    """ok, margin, conf, parse-ability -- per item, in the harness's own row order."""
    r = load_mek(tag, ds)
    if r is None:
        return None
    if closed == "SLAKE":
        idx = [i for i in range(len(r)) if r[i].get("answer_type") == "CLOSED"]
    elif closed == "YESNO":
        idx = [i for i in range(len(r)) if str(r[i].get("answer", "")).strip().lower() in ("yes", "no")]
    else:
        idx = list(range(len(r)))
    return dict(idx=idx,
                ok=np.array([as_ok(r[i]) for i in idx], float),
                margin=np.array([float(r[i].get("margin") or 0.0) for i in idx]),
                conf=np.array([float(r[i].get("conf") or 0.0) for i in idx]),
                gen_toks=np.array([float(r[i].get("gen_toks") or 0.0) for i in idx]),
                resp=[str(r[i].get("response", "")) for i in idx])


def open_items(d):
    return json.load(open(os.path.join(ROOT, d, "transfer_dump_{}_lingshu7b.json")))


def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    P, N = s[y == 1], s[y == 0]
    if not len(P) or not len(N):
        return float("nan")
    a = np.concatenate([P, N]); o = a.argsort(); rk = np.empty(len(a)); rk[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    ss = np.zeros(len(c)); np.add.at(ss, inv, rk); rk = (ss / c)[inv]
    return float((rk[:len(P)].sum() - len(P) * (len(P) + 1) / 2) / (len(P) * len(N)))


def paired_ci(a, b, rng, nboot=NBOOT):
    """paired item-level bootstrap of mean(a)-mean(b); a,b aligned per item."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a)
    idx = rng.integers(0, n, size=(nboot, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return dict(delta=round(float(a.mean() - b.mean()), 5), ci95=[round(lo, 5), round(hi, 5)],
                sig=bool(lo > 0 or hi < 0),
                verdict="WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_mek", default=None)
    ap.add_argument("--adapt_mek", default=None)
    ap.add_argument("--base_open", default=None)
    ap.add_argument("--adapt_open", default=None)
    ap.add_argument("--strong_mek", default="lingshu32b_full")
    ap.add_argument("--out", default=os.path.join(ART, "train_cheap_leg_2026-08-11.json"))
    ap.add_argument("--nboot", type=int, default=NBOOT)
    A = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = {"cells": {}, "arms": {"base": {"mek": A.base_mek, "open": A.base_open},
                                 "adapted": {"mek": A.adapt_mek, "open": A.adapt_open}}}

    # ------------------------------------------------------------------ MCQ / closed cells
    if A.base_mek and A.adapt_mek:
        for cell, ds, closed in MCQ_CELLS:
            b = mcq_vectors(A.base_mek, ds, closed)
            a = mcq_vectors(A.adapt_mek, ds, closed)
            s = mcq_vectors(A.strong_mek, ds, closed)
            if b is None or a is None or s is None:
                print(f"  SKIP {cell} (missing arm)")
                continue
            n = min(len(b["ok"]), len(a["ok"]), len(s["ok"]))
            ok_b, ok_a, ok_s = b["ok"][:n], a["ok"][:n], s["ok"][:n]
            rec = dict(
                format="mcq", n=int(n),
                greedy={"base": float(ok_b.mean()), "adapted": float(ok_a.mean()),
                        "delta": paired_ci(ok_a, ok_b, rng, A.nboot)},
                strong_32b_direct=float(ok_s.mean()),
                p10={"base": float(((ok_b == 1) & (ok_s == 0)).mean()),
                     "adapted": float(((ok_a == 1) & (ok_s == 0)).mean()),
                     "delta": paired_ci(((ok_a == 1) & (ok_s == 0)).astype(float),
                                        ((ok_b == 1) & (ok_s == 0)).astype(float), rng, A.nboot)},
                p01={"base": float(((ok_b == 0) & (ok_s == 1)).mean()),
                     "adapted": float(((ok_a == 0) & (ok_s == 1)).mean())},
                cheap_beats_strong={"base": bool(ok_b.mean() > ok_s.mean()),
                                    "adapted": bool(ok_a.mean() > ok_s.mean())},
                vs_strong={"base": paired_ci(ok_b, ok_s, rng, A.nboot),
                           "adapted": paired_ci(ok_a, ok_s, rng, A.nboot)},
                keep7b_auroc={"base": auroc((ok_b == 1) & (ok_s == 0), b["margin"][:n]),
                              "adapted": auroc((ok_a == 1) & (ok_s == 0), a["margin"][:n])},
                mean_gen_tokens={"base": float(b["gen_toks"][:n].mean()),
                                 "adapted": float(a["gen_toks"][:n].mean())},
                mean_margin={"base": float(b["margin"][:n].mean()),
                             "adapted": float(a["margin"][:n].mean())},
                empty_response_rate={"base": float(np.mean([len(x.strip()) == 0 for x in b["resp"][:n]])),
                                     "adapted": float(np.mean([len(x.strip()) == 0 for x in a["resp"][:n]]))})
            # ---- ANSWER-PRIOR DIAGNOSTIC (PMC-VQA especially) -------------------------------
            # PMC-VQA v2's gold letters are strongly skewed (test_2: C 12636, B 11984, A 4423,
            # D 4387) and train_2 carries the SAME skew, so a model fine-tuned on train_2 can gain
            # accuracy by learning the label prior rather than the medicine. Report the predicted
            # letter histogram and accuracy stratified by GOLD letter: a prior-only gain shows up
            # as the predicted histogram moving toward the gold histogram while per-gold-letter
            # accuracy rises on the frequent letters and FALLS on the rare ones.
            r_raw = load_mek(A.base_mek, ds)
            golds = [str(r_raw[i].get("answer", "")).strip() for i in b["idx"]][:n]
            if golds and all(len(g) == 1 and g.isalpha() for g in golds[:50]):
                def hist(resp):
                    h = {}
                    for x in resp[:n]:
                        c = (x.strip()[:1] or "?").upper()
                        h[c] = h.get(c, 0) + 1
                    return {k: v / n for k, v in sorted(h.items())}
                gh = {}
                for g in golds:
                    gh[g] = gh.get(g, 0) + 1
                rec["answer_prior"] = dict(
                    gold_hist={k: v / n for k, v in sorted(gh.items())},
                    pred_hist={"base": hist(b["resp"]), "adapted": hist(a["resp"])},
                    acc_by_gold_letter={
                        lab: {g: float(v[np.array([x == g for x in golds])].mean())
                              for g in sorted(set(golds))
                              if np.array([x == g for x in golds]).sum() > 0}
                        for lab, v in (("base", ok_b), ("adapted", ok_a))},
                    n_by_gold_letter={g: int(sum(1 for x in golds if x == g)) for g in sorted(set(golds))},
                    balanced_acc={lab: float(np.mean([
                        v[np.array([x == g for x in golds])].mean() for g in sorted(set(golds))]))
                        for lab, v in (("base", ok_b), ("adapted", ok_a))})
            out["cells"][cell] = rec
            print(f"  {cell:<16} n={n:5d} greedy {rec['greedy']['base']:.4f} -> "
                  f"{rec['greedy']['adapted']:.4f}  ({rec['greedy']['delta']['delta']:+.4f} "
                  f"[{rec['greedy']['delta']['ci95'][0]:+.4f},{rec['greedy']['delta']['ci95'][1]:+.4f}] "
                  f"{rec['greedy']['delta']['verdict']})  32B={rec['strong_32b_direct']:.4f}  "
                  f"p10 {rec['p10']['base']:.4f}->{rec['p10']['adapted']:.4f}", flush=True)

    # ------------------------------------------------------------------ open-text cells
    if A.base_open and A.adapt_open:
        strong = {}
        for cell, dskey in OPEN_CELLS.items():
            p = f"{ROOT}/ckpts/openvqa/strong_lingshu/ckpt_{dskey}_lingshu32b.judge.jsonl"
            strong[cell] = {json.loads(l)["idx"]: int(json.loads(l)["judge_ok"])
                            for l in open(p) if l.strip()}
        arms = {}
        for lab, d in (("base", A.base_open), ("adapted", A.adapt_open)):
            items = []
            for dk in G.DUMP_ORDER:
                items.extend(json.load(open(os.path.join(
                    ROOT, d, f"transfer_dump_{dk}_open_lingshu7b.json"))))
            arms[lab] = {(it["ds"], it["idx"]): it for it in items}
        common = sorted(set(arms["base"]) & set(arms["adapted"]),
                        key=lambda k: (G.EVAL_DS.index(k[0]), str(k[1])))
        out["open_pairing"] = dict(n_base=len(arms["base"]), n_adapted=len(arms["adapted"]),
                                   n_paired=len(common))
        # THE frozen metric, applied to each arm's own items restricted to the paired set
        res = {}
        for lab in ("base", "adapted"):
            it = [arms[lab][k] for k in common]
            res[lab] = G.sel_eff({(x["ds"], x["idx"]): x["scores"] for x in it}, items=it)
        for cell, dskey in OPEN_CELLS.items():
            m = np.array([k[0] == dskey for k in common])
            ids = [k[1] for k, keep in zip(common, m) if keep]
            ok_s = np.array([strong[cell].get(i, 0) for i in ids], float)
            g = {lab: np.array([arms[lab][k]["greedy_ok"] for k, keep in zip(common, m) if keep], float)
                 for lab in ("base", "adapted")}
            sel = {lab: (res[lab]["got"][m]).astype(float) for lab in ("base", "adapted")}
            orc = {lab: (res[lab]["rec"][m]).astype(float) for lab in ("base", "adapted")}
            rec = dict(
                format="open", n=int(m.sum()),
                greedy={"base": float(g["base"].mean()), "adapted": float(g["adapted"].mean()),
                        "delta": paired_ci(g["adapted"], g["base"], rng, A.nboot)},
                oracle8={"base": float(orc["base"].mean()), "adapted": float(orc["adapted"].mean()),
                         "delta": paired_ci(orc["adapted"], orc["base"], rng, A.nboot)},
                selected={"base": float(sel["base"].mean()), "adapted": float(sel["adapted"].mean()),
                          "delta": paired_ci(sel["adapted"], sel["base"], rng, A.nboot)},
                sel_eff={lab: res[lab]["per_ds"][dskey]["sel_eff"] for lab in ("base", "adapted")},
                strong_32b_direct=float(ok_s.mean()),
                p10={"base": float(((g["base"] == 1) & (ok_s == 0)).mean()),
                     "adapted": float(((g["adapted"] == 1) & (ok_s == 0)).mean()),
                     "delta": paired_ci(((g["adapted"] == 1) & (ok_s == 0)).astype(float),
                                        ((g["base"] == 1) & (ok_s == 0)).astype(float), rng, A.nboot)},
                p10_selected={"base": float(((sel["base"] == 1) & (ok_s == 0)).mean()),
                              "adapted": float(((sel["adapted"] == 1) & (ok_s == 0)).mean())},
                n_distinct={lab: float(np.mean(res[lab]["n_distinct"][m])) for lab in ("base", "adapted")})
            out["cells"][cell] = rec
            print(f"  {cell:<16} n={rec['n']:5d} greedy {rec['greedy']['base']:.4f} -> "
                  f"{rec['greedy']['adapted']:.4f} ({rec['greedy']['delta']['verdict']})  "
                  f"oracle@8 {rec['oracle8']['base']:.4f} -> {rec['oracle8']['adapted']:.4f} "
                  f"({rec['oracle8']['delta']['verdict']})  sel_eff "
                  f"{rec['sel_eff']['base']:.4f} -> {rec['sel_eff']['adapted']:.4f}  "
                  f"32B={rec['strong_32b_direct']:.4f}", flush=True)
        out["open_pooled"] = {lab: {k: res[lab][k] for k in
                                    ("n", "n_recoverable", "oracle", "greedy", "acc", "sel_eff")}
                              for lab in ("base", "adapted")}

    # ------------------------------------------------------------------ headline arithmetic
    cells = out["cells"]
    if len(cells) == 8:
        gb = {k: v["greedy"]["base"] for k, v in cells.items()}
        ga = {k: v["greedy"]["adapted"] for k, v in cells.items()}
        out["macro_always7b"] = {"base": float(np.mean(list(gb.values()))),
                                 "adapted": float(np.mean(list(ga.values()))),
                                 "delta": float(np.mean(list(ga.values())) - np.mean(list(gb.values())))}
        out["macro_always32b_direct"] = float(np.mean([v["strong_32b_direct"] for v in cells.values()]))
        out["sum_p10"] = {lab: float(sum(v["p10"][lab] for v in cells.values()))
                          for lab in ("base", "adapted")}
        out["router_ceiling_macro"] = {lab: out["sum_p10"][lab] / 8.0 for lab in ("base", "adapted")}
        out["guardrail_greedy"] = {k: v["greedy"]["delta"]["verdict"] for k, v in cells.items()}
    json.dump(out, open(A.out, "w"), indent=1, default=float)
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
