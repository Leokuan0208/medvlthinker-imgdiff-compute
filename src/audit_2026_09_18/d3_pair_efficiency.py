#!/usr/bin/env python3
"""D3 (b) -- HOW FEW UNLABELLED PAIRS DOES THE LINEAGE MAP NEED?

pilot_lineage_alignment_2026-09-20.json fitted a ridge map Qwen2.5-VL-7B -> Lingshu-7B hidden space on
9,688 label-free pairs and recovered 41% of the zero-shot-to-native gap (+0.0451 -> +0.0603, native
+0.0820).  That says the method works.  It does not say whether it is PRACTICAL, and the practicality is
the whole pitch: "a verifier trained once on a base model serves every fine-tune of it" only matters if
adapting it to a new fine-tune is cheap.

So: refit the map on n in {100, 300, 1000, 3000, 9688} pairs and re-measure.  Pairs cost NO LABELS -- they
are questions where both generators happened to emit the same normalised answer, so both were
teacher-forced on identical text.  What they DO cost is running both generators over the same prompts,
which is the real budget this curve prices.

EVERYTHING ELSE IS HELD FIXED against the pilot: same pairing key, same TRAIN-half-only restriction, same
ridge grid, same frozen Lingshu probes, same judge-of-record labels, same image-clustered bootstrap.  The
n=9688 point must reproduce the pilot's +0.0603 or something is wrong with this script, and that is
checked explicitly at the end.

ONE DESIGN NOTE.  The ridge strength is re-chosen at every n on a 10% label-free split of THAT n, because
a map fitted on 100 pairs needs more regularisation than one fitted on 9,688; holding lambda at the
9,688-pair optimum would make small n look artificially bad and the curve would be an artefact of the
hyperparameter, not of the data.

  python3 src/audit_2026_09_18/d3_pair_efficiency.py <out.json>

Read-only, CPU, 4 threads.
"""
import hashlib, json, os, sys, time
import numpy as np
import torch

torch.set_num_threads(4)
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import head_sweep as HS                      # noqa: E402
from genframe_data import rank_avg           # noqa: E402
FE, CK = f"{ROOT}/feats_hidden", f"{ROOT}/ckpts/openvqa/cheap_lingshu7b"
PD = f"{ROOT}/ckpts/train/genframe_head_pooled_ens_v2"
ART = f"{ROOT}/results/cascade_methods/artifacts"
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
SHARED = {"slake_open", "vqa_rad_open"}
ENS = [18, 20, 22]
NS = [100, 300, 1000, 3000, 0]               # 0 = all available pairs (the pilot's 9,688)
SEEDS = [0, 1, 2]                            # subsample seeds, so small n carries its own spread
rng = np.random.default_rng(20260921); NB = 10000


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def load(stem, dsf=None):
    z = np.load(f"{FE}/{stem}.npz"); m = json.load(open(f"{FE}/{stem}.meta.json"))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
    H = z["h_span"]
    return {L: H[keep, lay.index(L)].astype(np.float32) for L in ENS}, [m["rows"][i] for i in keep]


def key(s):
    return " ".join(str(s).lower().split()).rstrip(".!?, ")


def boot(res, a, b):
    ks = {}
    for r in res:
        ks.setdefault(r[0], []).append(r[a] - r[b])
    s = np.array([sum(v) for v in ks.values()], float); n = np.array([len(v) for v in ks.values()], float)
    ix = rng.integers(0, len(s), size=(NB, len(s))); bb = s[ix].sum(1) / n[ix].sum(1)
    return float(s.sum() / n.sum()), [float(np.quantile(bb, .025)), float(np.quantile(bb, .975))]


t0 = time.time()
H = {}
for L in ENS:
    st = np.load(f"{PD}/standardizer_L{L}.npz"); ms = []
    for s in range(8):
        m = HS.MLP(3584, 256, 1, 0.0, False)
        m.load_state_dict(torch.load(f"{PD}/head_L{L}_seed{s}.pt", map_location="cpu")); m.eval(); ms.append(m)
    H[L] = (st["mu"].astype(np.float32), st["sd"].astype(np.float32), ms)

pairs = {L: ([], []) for L in ENS}; qtr = {L: [] for L in ENS}; held = {}
for cell in BENCH:
    Xl, rl = load("generator_eval_finelayer" if cell in SHARED else f"generator_eval_finelayer_{cell}",
                  {cell} if cell in SHARED else None)
    Xq, rq = load(f"generator_eval_qwen_{cell}")
    il = {(r["idx"], key(r["na"])): i for i, r in enumerate(rl) if half(r["img_md5"]) == 1}
    iq = [(i, il[(r["idx"], key(r["na"]))]) for i, r in enumerate(rq)
          if half(r["img_md5"]) == 1 and (r["idx"], key(r["na"])) in il]
    trq = np.array([half(r["img_md5"]) == 1 for r in rq])
    for L in ENS:
        if iq:
            pairs[L][0].append(Xq[L][[a for a, _ in iq]]); pairs[L][1].append(Xl[L][[b for _, b in iq]])
        qtr[L].append(Xq[L][trq])
    g = {}
    for l in open(f"{CK}/ckpt_{cell}_qwen25vl7b.judge.jsonl"):
        if l.strip():
            d = json.loads(l); g[d["idx"]] = int(d["judge_ok"])
    held[cell] = ({L: Xq[L][~trq] for L in ENS}, [rq[i] for i in np.where(~trq)[0]], g)
    print(f"{cell:17} paired rows {len(iq):6} ({time.time()-t0:.0f}s)", flush=True)
    del Xl, Xq

PAIRQ = {L: np.concatenate(pairs[L][0]) for L in ENS}
PAIRL = {L: np.concatenate(pairs[L][1]) for L in ENS}
STQ = {}
for L in ENS:
    allq = np.concatenate(qtr[L]); STQ[L] = (allq.mean(0), allq.std(0) + 1e-6)
NTOT = len(PAIRQ[ENS[0]])
print(f"\n{NTOT:,} label-free pairs available in total\n", flush=True)


def fit_map(L, idx, seed):
    """Ridge Z_qwen -> Z_lingshu on the given pair subset; lambda chosen label-free on a 10% split."""
    muq, sdq = STQ[L]; mul, sdl, _ = H[L]
    Zq = (PAIRQ[L][idx] - muq) / sdq
    Zl = (PAIRL[L][idx] - mul) / sdl
    r = np.random.default_rng(1000 + seed)
    perm = r.permutation(len(Zq)); k = max(1, len(Zq) // 10); va, tr = perm[:k], perm[k:]
    G = (Zq[tr].T @ Zq[tr]).astype(np.float64); C = (Zq[tr].T @ Zl[tr]).astype(np.float64)
    best = None
    for lam in (1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
        w = np.linalg.solve(G + lam * np.eye(G.shape[0]), C).astype(np.float32)
        err = float(((Zq[va] @ w - Zl[va]) ** 2).mean())
        if best is None or err < best[1]:
            best = (lam, err, w)
    return best


def evaluate(W):
    """Macro verifier-minus-greedy over the eight held-out halves, with the mapped Qwen states."""
    per = {}
    for cell in BENCH:
        X, rows, g = held[cell]
        y = np.array([r["y"] for r in rows], int)
        byq = {}
        for i, r in enumerate(rows):
            byq.setdefault(r["idx"], []).append(i)
        sc = []
        for L in ENS:
            muq, sdq = STQ[L]; _, _, ms = H[L]
            Z = ((X[L] - muq) / sdq) @ W[L]
            sc.append(np.stack([HS.predict(m, Z.astype(np.float32)) for m in ms]))
        res = []
        for q, ii in byq.items():
            if q not in g:
                continue
            ii = np.array(ii)
            hr = np.mean([rank_avg(s[k][ii]) for s in sc for k in range(s.shape[0])], axis=0)
            res.append([rows[ii[0]]["img_md5"], g[q], int(y[ii][int(np.argmax(hr))])])
        d, ci = boot(res, 2, 1)
        per[cell] = {"n": len(res), "delta": d, "ci": ci}
    per["_macro"] = float(np.mean([v["delta"] for k, v in per.items() if not k.startswith("_")]))
    return per


out = {"title": "D3(b): how many LABEL-FREE pairs does the Qwen->Lingshu lineage map need?",
       "date": "2026-09-21", "no_fabricated_numbers": True,
       "currency": "judge of record (MedVLThinker-32B)",
       "pairs_available": int(NTOT), "seeds": SEEDS, "bootstrap": f"{NB}, clustered by img_md5",
       "reference": {"zero_shot_A": 0.0451, "restandardised_B": 0.0490,
                     "ridge_all_pairs_pilot": 0.0603, "qwen_native": 0.0820,
                     "source": "pilot_lineage_alignment_2026-09-20.json"},
       "curve": {}, "COMPLETE": False}

for n in NS:
    lab = "all" if n == 0 else str(n)
    macros = []; lams = {}
    for seed in (SEEDS if n else [0]):          # the full-pair point has nothing to subsample
        r = np.random.default_rng(7000 + seed)
        idx = np.arange(NTOT) if n == 0 else r.choice(NTOT, size=min(n, NTOT), replace=False)
        W = {}
        for L in ENS:
            lam, err, w = fit_map(L, idx, seed)
            W[L] = w; lams.setdefault(str(L), []).append(lam)
        per = evaluate(W)
        macros.append(per["_macro"])
        if seed == (SEEDS[0] if n else 0):
            first = per
    out["curve"][lab] = {"n_pairs": int(NTOT if n == 0 else min(n, NTOT)),
                         "macro_mean": float(np.mean(macros)),
                         "macro_sd": float(np.std(macros, ddof=1)) if len(macros) > 1 else 0.0,
                         "macro_per_seed": [float(x) for x in macros],
                         "ridge_lambda": lams,
                         "per_benchmark_first_seed": {k: v for k, v in first.items() if not k.startswith("_")}}
    print(f"n_pairs {lab:>5}  macro {np.mean(macros):+.4f}"
          f"{'  sd %.4f' % np.std(macros, ddof=1) if len(macros) > 1 else ''}"
          f"   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)

full = out["curve"]["all"]["macro_mean"]
out["reproduces_pilot"] = {"this_run_all_pairs": full, "pilot": 0.0603,
                           "deviation": round(full - 0.0603, 5),
                           "ok": bool(abs(full - 0.0603) < 0.004)}
z = out["reference"]["zero_shot_A"]; nat = out["reference"]["qwen_native"]
out["VERDICT"] = (
    f"With all {NTOT:,} label-free pairs the map gives {full:+.4f} against {z:+.4f} zero-shot and "
    f"{nat:+.4f} native. Curve: " +
    ", ".join(f"{k}={v['macro_mean']:+.4f}" for k, v in out["curve"].items()) +
    ". The practical question D3 turns on is where this saturates: if a few hundred pairs already buy "
    "most of the gain, adapting a frozen verifier to a new fine-tune of the same base is nearly free "
    "and the onboarding cost changes from per-benchmark-per-model to per-benchmark-once-per-family.")
out["COMPLETE"] = True
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("\n" + out["VERDICT"])
print(f"reproduces pilot: {out['reproduces_pilot']}")
