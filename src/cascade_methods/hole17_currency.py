#!/usr/bin/env python3
"""hole17_currency.py -- the open-text endpoint in BOTH currencies, on IDENTICAL picks.

The project's frozen open-text label is the 32B judge.  Every verifier/open-text endpoint must also
be reported under normalised exact match, because a newly fitted policy gets a free ~0.006-0.009
under the judge from paraphrase drift (artifacts/coadapt_verifier_T04_2026-08-14.json).  Here the
POLICY IS NOT REFIT UNDER EM -- it is fit exactly as shipped (judge labels, judge-trained verifier),
and only the LABEL APPLIED TO THE DELIVERED SLOT changes.  Same picks, same escalations, two
currencies.

EM scorer = score_em() from decoding_sweep_gen.py, which is run_openvqa.py's own normalised exact
match plus its short-answer contains fallback -- imported, not re-implemented.

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_currency.py
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import integrated_method as IM
import integrated_pandora as IP
import hole17_data as HD


ROOT = IM.ROOT
OPEN_KEY = HD.OPEN_KEY

# ---- the EM scorer, lifted VERBATIM out of src/labeling/run_openvqa.py (the file that wrote every
# open-text label in this project) rather than re-implemented, so it cannot drift. -----------------
def _load_run_openvqa_scorer():
    import re as _re, string as _string
    src = open(os.path.join(ROOT, "src/labeling/run_openvqa.py")).read()
    m = _re.search(r"^def norm\(s\):\n(?:.*\n)*?^    return 0\n", src, _re.M)
    assert m, "could not locate norm()/score() in run_openvqa.py"
    ns = {"re": _re, "string": _string}
    exec(compile(m.group(0), "run_openvqa.py:norm+score", "exec"), ns)
    return ns["score"], m.group(0)

score_em, _SCORER_SRC = _load_run_openvqa_scorer()


def pandora_pick(cal, raw, z_c, z_s):
    """(N, esc, pick) -- the same policy as HD.pandora_vec, exposing the chosen slot."""
    n, Nmax = cal.shape
    if z_s > z_c:
        return np.zeros(n), np.ones(n, bool), np.zeros(n, int)
    hit = cal >= z_c
    first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, Nmax)
    drawn = np.arange(Nmax)[None, :] < first[:, None]
    best_cal = np.where(drawn, cal, -np.inf).max(axis=1)
    esc = (first == Nmax) & (best_cal < z_s)
    pick = np.where(drawn, raw, -np.inf).argmax(axis=1)
    return first.astype(float), esc, pick


def load_em(verifier="ckpts/train/lora_verifier_disjoint"):
    """{cell: dict(sl_em (n,8), strong_em (n,), greedy_em (n,), n)} aligned to IP.load_open_rows order."""
    IP.ADAPTER = verifier
    out = {}
    for name in HD.OPEN_B:
        ds = OPEN_KEY[name]
        dump = {r["idx"]: r for r in json.load(open(os.path.join(ROOT, verifier,
                                                                 f"transfer_dump_{ds}_lingshu7b.json")))}
        cheap = {}
        for l in open(os.path.join(ROOT, f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.jsonl")):
            r = json.loads(l); cheap[r["idx"]] = r
        strong = {}
        for l in open(os.path.join(ROOT, f"ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.jsonl")):
            r = json.loads(l); strong[r["idx"]] = r
        rows = IP.load_open_rows(ds)                    # the ORDER of record
        SL, ST, GR = [], [], []
        for r in rows:
            i = r["idx"] if "idx" in r else None
            assert i is not None, "load_open_rows must carry idx"
            gold = cheap[i]["gold"]
            preds = dump[i]["preds"][:8]
            SL.append([score_em(p, gold) for p in preds])
            ST.append(score_em(strong[i]["modal_pred"], gold))
            GR.append(score_em(cheap[i]["modal_pred"], gold))
        out[name] = dict(sl_em=np.array(SL, float), strong_em=np.array(ST, float),
                         greedy_em=np.array(GR, float), n=len(rows))
    return out


if __name__ == "__main__":
    IP.ADAPTER = "ckpts/train/lora_verifier_disjoint"
    openc = HD.load_open()
    em = load_em()
    # equivalence of pandora_pick with HD.pandora_vec on the judge currency
    rng = np.random.default_rng(0); worst = 0
    for name, d in openc.items():
        for _ in range(20):
            z_c = float(rng.choice(d["raw"].ravel())); z_s = float(rng.choice(d["raw"].ravel()))
            N, E, O = HD.pandora_vec(d["raw"], d["raw"], d["sl"], d["strong"], z_c, z_s)
            N2, E2, P2 = pandora_pick(d["raw"], d["raw"], z_c, z_s)
            O2 = np.where(E2, d["strong"], d["sl"][np.arange(d["n"]), P2])
            worst = max(worst, np.abs(N - N2).max(), np.abs(E - E2).max(), np.abs(O - O2).max())
    print("max |pandora_pick - pandora_vec| over 3 cells x 20 (z_c,z_s) = %g" % worst)
    assert worst == 0
    print(json.dumps({k: dict(n=em[k]["n"],
                              judge_greedy=round(float(openc[k]["greedy"].mean()), 4),
                              em_greedy=round(float(em[k]["greedy_em"].mean()), 4),
                              judge_strong=round(float(openc[k]["strong"].mean()), 4),
                              em_strong=round(float(em[k]["strong_em"].mean()), 4),
                              judge_oracle8=round(float((openc[k]["sl"].max(1) > 0).mean()), 4),
                              em_oracle8=round(float((em[k]["sl_em"].max(1) > 0).mean()), 4))
                      for k in em}, indent=1))
