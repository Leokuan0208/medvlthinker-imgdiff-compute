import json, numpy as np
R = json.load(open("/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json"))
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
NB = 10000
rng = np.random.default_rng(12345)
def boot_cell(recs, cu, arm="pick"):
    R2 = [r for r in recs if r[arm].get(cu) is not None and r["greedy"].get(cu) is not None]
    imgs, inv = np.unique([r["img"] for r in R2], return_inverse=True)
    d = np.array([r[arm][cu] - r["greedy"][cu] for r in R2], float)
    W = rng.multinomial(len(imgs), np.full(len(imgs), 1 / len(imgs)), size=NB).astype(float)
    s = W @ np.bincount(inv, weights=d, minlength=len(imgs)); n = W @ np.bincount(inv, minlength=len(imgs)).astype(float)
    return d.mean(), s / n, s, n, len(R2)
out = {}
for cu in ["judge", "xjudge", "em"]:
    pts, draws, S, N, ns = {}, {}, {}, {}, {}
    for c in OPEN8:
        pts[c], draws[c], S[c], N[c], ns[c] = boot_cell(R[c], cu)
    mac = np.mean([draws[c] for c in OPEN8], 0)
    mic_pt = sum(pts[c] * ns[c] for c in OPEN8) / sum(ns.values())
    mic = sum(S[c] for c in OPEN8) / sum(N[c] for c in OPEN8)
    o = {"macro": float(np.mean(list(pts.values()))), "macro_ci": np.percentile(mac, [2.5, 97.5]).round(4).tolist(),
         "micro": round(float(mic_pt), 4), "micro_ci": np.percentile(mic, [2.5, 97.5]).round(4).tolist()}
    # Bonferroni per-cell: 8 tests and 40 tests (8 cells x 5 currencies)
    o["cells"] = {}
    for c in OPEN8:
        o["cells"][c] = {"pt": round(float(pts[c]), 4),
                         "ci95": np.percentile(draws[c], [2.5, 97.5]).round(4).tolist(),
                         "ci_bonf8": np.percentile(draws[c], [100 * 0.025 / 8, 100 - 100 * 0.025 / 8]).round(4).tolist(),
                         "ci_bonf40": np.percentile(draws[c], [100 * 0.025 / 40, 100 - 100 * 0.025 / 40]).round(4).tolist()}
    # leave-one-benchmark-out macro point
    o["lobo_macro"] = {c: round(float(np.mean([pts[k] for k in OPEN8 if k != c])), 4) for c in OPEN8}
    # per fold macro point
    for f in (0, 1):
        o[f"fold{f}_macro_pt"] = round(float(np.mean([np.mean([r["pick"][cu] - r["greedy"][cu] for r in R[c] if r["fold"] == f and r["pick"].get(cu) is not None and r["greedy"].get(cu) is not None]) for c in OPEN8])), 4)
    out[cu] = o
# gain vs headroom / oracle correlation (judge)
g, h, orc, se = [], [], [], []
for c in OPEN8:
    rr = R[c]
    gr = np.mean([r["greedy"]["judge"] for r in rr]); pk = np.mean([r["pick"]["judge"] for r in rr]); oc = np.mean([r["oracle_judge"] for r in rr])
    g.append(pk - gr); h.append(oc - gr); orc.append(oc); se.append(pk / oc)
out["corr_gain_headroom"] = round(float(np.corrcoef(g, h)[0, 1]), 3)
out["corr_gain_oracle"] = round(float(np.corrcoef(g, orc)[0, 1]), 3)
out["headroom_by_cell"] = {c: round(float(x), 4) for c, x in zip(OPEN8, h)}
out["gain_over_headroom"] = {c: round(float(a / b), 3) for c, a, b in zip(OPEN8, g, h)}
# fraction of questions where pick answer string differs from greedy answer
out["pick_differs_from_greedy"] = {c: round(float(np.mean([r["pick_ans"].strip().lower() != r["greedy_ans"].strip().lower() for r in R[c]])), 3) for c in OPEN8}
# judge positive rate of greedy vs pick: how often xjudge missing
out["xjudge_missing_pick"] = {c: int(sum(r["pick"].get("xjudge") is None for r in R[c])) for c in OPEN8}
print(json.dumps(out, indent=1))
