#!/usr/bin/env python3
"""
ugv_mcq_verdict.py -- THE VERDICT for the Unified Generative Verifier (UGV) on the MCQ half.

Reads the per-sample dumps produced by src/labeling/run_mcq_generate_verify.py (in ckpts/mcq_gen_verify/)
and answers: does the SINGLE trained generative verifier (same head/prompt as open-text) pick correct
answers among N GENERATED MCQ answers -- enabling a no-router unified method?

For each (family, dataset, mode) it computes, over N sampled generations per item:
  greedy      = acc of the first sample                      (temp>0 draw, the "just generate one" baseline)
  verifier@N  = acc of the sample the trained verifier scores highest   (UGV best-of-N)
  oracle@N    = acc if ANY of the N samples is correct        (ceiling for best-of-N selection)
  selfcons@N  = acc of the majority-vote (self-consistency) answer
Reported under TWO correctness labels:
  as-run  = map_correct (nearest MCQ option by token overlap; letter-credit) -- the pipeline's own label
  strict  = normalized exact/containment match of the generated text to the gold answer text
            (robustness check: nearest-option mapping over-credits token overlap on MCQ content mode)
Also: verifier discrimination (AUROC of score vs correctness) and the oracle-gap closed by the verifier.
For MCQ datasets (PMC_VQA, MedXpertQA-MM) it contrasts CONTENT mode (options hidden -> generate) with
LETTER mode (options shown -> pick a letter; the deployed letter approach).

Writes results/cascade_methods/artifacts/ugv_mcq_verdict.json and prints the tables + verdict.
Run from repo root:  python3 src/cascade_methods/ugv_mcq_verdict.py
"""
import json, glob, os, re, string
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
BASE = os.path.join(ROOT, "ckpts/mcq_gen_verify")
OUT  = os.path.join(ROOT, "results/cascade_methods/artifacts/ugv_mcq_verdict.json")
DIRS = {"lingshu7b": os.path.join(BASE, "lingshu7b"), "mvt7b": os.path.join(BASE, "mvt7b")}
MCQ_DATASETS = {"PMC_VQA", "MedXpertQA-MM"}   # true A/B/C/D option sets (letter mode meaningful)

# (family, dataset, mode) tuples to report. content = generate-answer; letter = pick-a-letter.
REPORT = [
    ("lingshu7b", "PMC_VQA",       "content"),
    ("lingshu7b", "MedXpertQA-MM", "content"),
    ("lingshu7b", "PATH_VQA",      "content"),
    ("lingshu7b", "SLAKE",         "content"),
    ("lingshu7b", "VQA_RAD",       "content"),
    ("lingshu7b", "PMC_VQA",       "letter"),
    ("lingshu7b", "MedXpertQA-MM", "letter"),
    ("mvt7b",     "PMC_VQA",       "content"),
    ("mvt7b",     "PATH_VQA",      "content"),
    ("mvt7b",     "VQA_RAD",       "content"),
]

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()

def strict_ok(pred, gold):
    p, g = norm(pred), norm(gold)
    return int(bool(g) and (p == g or g in p or p in g))

def resolve(family, ds, mode):
    """Find the dump for (family, ds, mode); prefer the file with the most rows (fresh 2000-row > old partials)."""
    cand = glob.glob(os.path.join(DIRS[family], f"ckpt_{ds}_*_{mode}_sc8.jsonl"))
    cand = [c for c in cand if os.path.isfile(c)]
    if not cand:
        return None
    return max(cand, key=lambda c: sum(1 for _ in open(c)))

def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); start = cum - cnt
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def analyze(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    # as-run labels (map_correct, stored in oks)
    greedy   = np.mean([r["oks"][0] for r in rows])
    ver_boN  = np.mean([r["pick_ok"] for r in rows])
    oracle   = np.mean([max(r["oks"]) for r in rows])
    # self-consistency majority vote (as-run)
    def sc_ok(r):
        from collections import Counter
        groups = {}
        for p, o in zip(r["preds"], r["oks"]):
            groups.setdefault(norm(p), []).append(o)
        modal = max(groups.items(), key=lambda kv: len(kv[1]))[0]
        return groups[modal][0]
    selfcons = np.mean([sc_ok(r) for r in rows])
    # strict labels (recomputed from text vs gold)
    def soks(r): return [strict_ok(p, r["gold"]) for p in r["preds"]]
    S = [soks(r) for r in rows]
    greedy_s = np.mean([s[0] for s in S])
    ver_s    = np.mean([S[i][rows[i]["pick"]] for i in range(n)])
    oracle_s = np.mean([max(s) for s in S])
    # verifier discrimination: flatten all (score, ok) pairs, as-run and strict
    fs = [sc for r in rows for sc in r["scores"]]
    fo = [o for r in rows for o in r["oks"]]
    fss = [o for s in S for o in s]
    def gap_closed(g, v, o):
        return float((v - g) / (o - g)) if (o - g) > 1e-9 else None
    return {
        "file": os.path.relpath(path, ROOT), "n": n,
        "asrun": {"greedy": round(float(greedy), 4), "verifier_boN": round(float(ver_boN), 4),
                  "oracle_boN": round(float(oracle), 4), "selfcons": round(float(selfcons), 4),
                  "verifier_gain": round(float(ver_boN - greedy), 4),
                  "oracle_gap_closed": gap_closed(greedy, ver_boN, oracle),
                  "auroc_score_vs_ok": (round(a, 4) if (a := auroc(fs, fo)) is not None else None)},
        "strict": {"greedy": round(float(greedy_s), 4), "verifier_boN": round(float(ver_s), 4),
                   "oracle_boN": round(float(oracle_s), 4),
                   "verifier_gain": round(float(ver_s - greedy_s), 4),
                   "oracle_gap_closed": gap_closed(greedy_s, ver_s, oracle_s),
                   "auroc_score_vs_ok": (round(a, 4) if (a := auroc(fs, fss)) is not None else None)},
    }

def main():
    results, missing = {}, []
    for fam, ds, mode in REPORT:
        p = resolve(fam, ds, mode)
        key = f"{fam}|{ds}|{mode}"
        if p is None:
            missing.append(key); continue
        results[key] = analyze(p)

    # ---- print tables ----
    def line(k, r, lab):
        a = r[lab]
        print(f"  {k:34s} n={r['n']:<5d} greedy={a['greedy']:.3f}  ver@N={a['verifier_boN']:.3f}  "
              f"oracle={a['oracle_boN']:.3f}  gain={a['verifier_gain']:+.3f}  "
              f"gapClosed={a['oracle_gap_closed'] if a['oracle_gap_closed'] is None else round(a['oracle_gap_closed'],2)}  "
              f"AUROC={a['auroc_score_vs_ok']}")
    for lab in ("asrun", "strict"):
        print(f"\n===== {lab.upper()} correctness =====")
        for k in results:
            line(k, results[k], lab)
    if missing:
        print("\nMISSING (run not found yet):", missing)

    # ---- verdict logic (use STRICT labels for MCQ content; they are the honest signal) ----
    verdict = {"per_dataset": {}, "missing": missing}
    mcq_content_hold = []   # does content verifier beat content greedy, strictly?
    competitive = []        # is content verifier@N competitive with letter greedy (>= letter*0.95)?
    for ds in ("PMC_VQA", "MedXpertQA-MM"):
        entry = {}
        for fam in ("lingshu7b", "mvt7b"):
            ck, lk = f"{fam}|{ds}|content", f"{fam}|{ds}|letter"
            if ck not in results:
                continue
            c = results[ck]
            e = {"content_strict": c["strict"], "content_asrun": c["asrun"]}
            if lk in results:
                l = results[lk]
                e["letter_strict"] = l["strict"]; e["letter_asrun"] = l["asrun"]
                # competitiveness: content verifier@N (strict) vs letter greedy (strict)
                lg = l["strict"]["greedy"]; cv = c["strict"]["verifier_boN"]
                e["content_ver_vs_letter_greedy_strict"] = round(cv - lg, 4)
                competitive.append(cv >= 0.95 * lg if lg > 0 else None)
            entry[fam] = e
            mcq_content_hold.append(c["strict"]["verifier_gain"] > 0)
        verdict["per_dataset"][ds] = entry

    # aggregate the "does UGV pick correctly on generated MCQ" signal across MCQ content runs
    mcq_keys = [k for k in results if k.split("|")[1] in MCQ_DATASETS and k.endswith("content")]
    aurocs = [results[k]["strict"]["auroc_score_vs_ok"] for k in mcq_keys
              if results[k]["strict"]["auroc_score_vs_ok"] is not None]
    gains  = [results[k]["strict"]["verifier_gain"] for k in mcq_keys]
    verdict["mcq_content_mean_verifier_gain_strict"] = round(float(np.mean(gains)), 4) if gains else None
    verdict["mcq_content_mean_auroc_strict"] = round(float(np.mean(aurocs)), 4) if aurocs else None
    verdict["holds"] = bool(gains and np.mean(gains) > 0 and aurocs and np.mean(aurocs) > 0.55)
    verdict["all_runs"] = results

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(verdict, open(OUT, "w"), indent=2)
    print("\n=== VERDICT ===")
    print("MCQ content mean verifier gain (strict):", verdict["mcq_content_mean_verifier_gain_strict"])
    print("MCQ content mean verifier AUROC (strict):", verdict["mcq_content_mean_auroc_strict"])
    print("UGV-on-MCQ holds:", verdict["holds"])
    print("wrote", os.path.relpath(OUT, ROOT))

if __name__ == "__main__":
    main()
