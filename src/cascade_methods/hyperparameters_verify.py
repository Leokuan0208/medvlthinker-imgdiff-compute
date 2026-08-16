#!/usr/bin/env python3
"""hyperparameters_verify.py -- ADVERSARIAL re-derivation of the four hyper-parameter rounds of
2026-08-15, plus the COMBINED operating point they were never evaluated at together.

Rounds under test
  knob 1  hole17_macro_refit_2026-08-15.json      (macro-objective refit of tau / lambda / the open gate)
  knob 2  veto_binning_2026-08-15.json            (f8_veto n_bins x alpha_z)
  knob 3  verifier_hparams_2026-08-15.json        (the verifier's scoring max_pixels)
  knob 4  weitzman_T04_2026-08-15.json            (the Weitzman lambda, on T=0.4 pools)
  free    decoding_ladder_cold_2026-08-14.json    (T=0.4, pre-registered, established)

Everything here is written from first principles against the raw dumps; the round's own scripts are only
imported where the round's code IS the object under test (beat32b_fusion.mcq as the dump reader,
beat32b_more.open_features as the shipped feature builder).

Output: results/cascade_methods/artifacts/hyperparameters_combined_2026-08-15.json
        results/cascade_methods/artifacts/_hpv_veto_null_indep.jsonl   (permutation null, resumable)

Numerics pinned: OMP/OPENBLAS/MKL_NUM_THREADS=1 (8 for the F10 rebuild), PYTHONHASHSEED=0, CPU only
(no torch, so no TF32 exposure), frozen canonical row order, argmax over raw scores with a strict >
(first-index argmax), paired item bootstrap nboot=10000 seed 20260815.

Run from the repo root:
    python3 src/cascade_methods/hyperparameters_verify.py --stage nulltest
    python3 src/cascade_methods/hyperparameters_verify.py --stage seeds       # 10-seed nested CV, R0 + R4
    nohup python3 -u src/cascade_methods/hyperparameters_verify.py --stage null > logs/hpv_veto_null_indep.log 2>&1 &
    python3 src/cascade_methods/hyperparameters_verify.py --stage f10         # rebuild the open half on T04/T07r
    python3 src/cascade_methods/hyperparameters_verify.py --stage frontier    # the combined table
"""
import argparse, json, os, pickle, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import numpy as np

ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
CACHE = "/tmp/hpv"
os.makedirs(CACHE, exist_ok=True)

NB, BSEED = 10000, 20260815
MCQ = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"]
OPEN = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
CELLS8 = MCQ + OPEN
BAR = 0.656672
NBINS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64]
ALPHAZ = [0.0, 0.25, 0.5, 0.842, 1.0, 1.282, 1.645, 1.96, 2.326]
MINTR, KOUT, KIN = 30, 5, 5


# ------------------------------------------------------------------ inputs
def load_mcq():
    p = os.path.join(CACHE, "mcq_cache.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    import beat32b_fusion as B
    spec = [("PMC_VQA", lambda: B.mcq("PMC_VQA")),
            ("SLAKE_closed", lambda: B.mcq("SLAKE", "SLAKE")),
            ("VQA_RAD_closed", lambda: B.mcq("VQA_RAD", "YESNO")),
            ("PATH_VQA_closed", lambda: B.mcq("PATH_VQA", "YESNO")),
            ("MedXpertQA-MM", lambda: B.mcq("MedXpertQA-MM", None, think_tag="lingshu32b_reason"))]
    D = {nm: dict(ok7=np.asarray(fn()["ok7"], float)) for nm, fn in []}          # placeholder
    D = {}
    for nm, fn in spec:
        d = fn()
        D[nm] = dict(ok7=np.asarray(d["ok7"], float), ok32=np.asarray(d["ok32"], float),
                     c7=np.asarray(d["c7"], float))
    pickle.dump(D, open(p, "wb"))
    return D


def frozen_vectors():
    return np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"), allow_pickle=True)


# ------------------------------------------------------------------ the certified veto, re-implemented
def wilson_lb(k, n, z):
    """One-sided Wilson lower bound.  Independent of beat32b_more.wilson_lb."""
    if n == 0:
        return 0.0
    if z == 0.0:
        return k / n
    p = k / n
    den = 1.0 + z * z / n
    return max(0.0, (p + z * z / (2 * n)) / den
               - (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))


def _edges(c, nb):
    qs = np.quantile(c, np.linspace(0, 1, nb + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    return np.unique(qs)


def fit_cert(ok7, ok32, c7, idx_tr, nb):
    e = _edges(c7[idx_tr], nb)
    b = np.clip(np.digitize(c7[idx_tr], e[1:-1]), 0, len(e) - 2)
    stats = []
    for j in range(len(e) - 1):
        m = b == j
        n = int(m.sum())
        stats.append((int(ok7[idx_tr][m].sum()), n, float(ok32[idx_tr][m].mean()) if n else 0.0))
    return e, stats


def apply_cert(ok7, ok32, c7, idx_te, e, stats, az):
    b = np.clip(np.digitize(c7[idx_te], e[1:-1]), 0, len(e) - 2)
    ok = ok32[idx_te].copy()
    vt = np.zeros(len(idx_te), bool)
    for j, (k7, n7, a32) in enumerate(stats):
        if n7 < MINTR:
            continue
        if wilson_lb(k7, n7, az) >= a32:
            m = b == j
            ok[m] = ok7[idx_te][m]
            vt[m] = True
    return ok, vt


def veto_published_folds(d, nb=5, az=1.645):
    """The shipped setting on the published fold structure arange(n) % 5 -- the S0 null test."""
    n = len(d["ok7"])
    fold = np.arange(n) % KOUT
    out, vt = d["ok32"].copy(), np.zeros(n, bool)
    for f in range(KOUT):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        e, st = fit_cert(d["ok7"], d["ok32"], d["c7"], tr, nb)
        o, v = apply_cert(d["ok7"], d["ok32"], d["c7"], te, e, st, az)
        out[te], vt[te] = o, v
    return out, vt


def mcnemar_z(ok_arm, ok_ref):
    b = float(((ok_arm == 1) & (ok_ref == 0)).sum())
    c = float(((ok_arm == 0) & (ok_ref == 1)).sum())
    return 0.0 if b + c == 0 else (b - c) / np.sqrt(b + c)


def inner_eval(ok7, ok32, c7, tr_idx, rng):
    """Cross-fit every (n_bins, alpha_z) inside tr_idx.  alpha_z does not change the binning, so the
    per-bin counts are computed once per n_bins and all 9 alpha_z read off them."""
    n = len(tr_idx)
    perm = rng.permutation(n)
    ff = np.empty(n, int)
    ff[perm] = np.arange(n) % KIN
    res = {}
    for nb in NBINS:
        oks = {az: np.empty(n) for az in ALPHAZ}
        vts = {az: np.zeros(n, bool) for az in ALPHAZ}
        for f in range(KIN):
            te, tr = np.where(ff == f)[0], np.where(ff != f)[0]
            e, st = fit_cert(ok7, ok32, c7, tr_idx[tr], nb)
            for az in ALPHAZ:
                o, v = apply_cert(ok7, ok32, c7, tr_idx[te], e, st, az)
                oks[az][te], vts[az][te] = o, v
        for az in ALPHAZ:
            res[(nb, az)] = (oks[az], vts[az])
    return res


def run_seed(D, seed, rule="R4"):
    """Nested CV.  R0 = the shipped fixed setting, veto on PMC only.  R4 = one GLOBAL (n_bins, alpha_z)
    chosen on the inner cross-fit, deployed only on the cells an inner one-sided McNemar admits (z>1.645)."""
    rng = np.random.default_rng(seed)
    out = {c: np.empty(len(D[c]["ok7"])) for c in MCQ}
    vout = {c: np.zeros(len(D[c]["ok7"]), bool) for c in MCQ}
    folds = {}
    for c in MCQ:
        n = len(D[c]["ok7"])
        p = rng.permutation(n)
        f = np.empty(n, int)
        f[p] = np.arange(n) % KOUT
        folds[c] = f
    sel = []
    for of in range(KOUT):
        if rule == "R0":
            for c in MCQ:
                d = D[c]
                tr, te = np.where(folds[c] != of)[0], np.where(folds[c] == of)[0]
                if c == "PMC_VQA":
                    e, st = fit_cert(d["ok7"], d["ok32"], d["c7"], tr, 5)
                    o, v = apply_cert(d["ok7"], d["ok32"], d["c7"], te, e, st, 1.645)
                else:
                    o, v = d["ok32"][te], np.zeros(len(te), bool)
                out[c][te], vout[c][te] = o, v
            continue
        cand = {}
        for c in MCQ:
            tr = np.where(folds[c] != of)[0]
            cand[c] = inner_eval(D[c]["ok7"], D[c]["ok32"], D[c]["c7"], tr,
                                 np.random.default_rng(seed * 1000 + of * 7 + hash(c) % 997))
        best = None
        for nb in NBINS:
            for az in ALPHAZ:
                tot, adm = 0.0, []
                for c in MCQ:
                    tr = np.where(folds[c] != of)[0]
                    ok, _ = cand[c][(nb, az)]
                    ref = D[c]["ok32"][tr]
                    if mcnemar_z(ok, ref) > 1.645:
                        adm.append(c)
                        tot += ok.mean() - ref.mean()
                if best is None or tot > best[0]:
                    best = (tot, nb, az, tuple(adm))
        _, nb, az, adm = best
        sel.append((nb, az, adm))
        for c in MCQ:
            d = D[c]
            tr, te = np.where(folds[c] != of)[0], np.where(folds[c] == of)[0]
            if c in adm:
                e, st = fit_cert(d["ok7"], d["ok32"], d["c7"], tr, nb)
                o, v = apply_cert(d["ok7"], d["ok32"], d["c7"], te, e, st, az)
            else:
                o, v = d["ok32"][te], np.zeros(len(te), bool)
            out[c][te], vout[c][te] = o, v
    return out, vout, sel


# ------------------------------------------------------------------ the shipped open policy (F10-L2D)
def f10_persample(ok7, ok32, X, K=5):
    """best-of-8 verifier pick + learned consistent-L2D rejector -> 32B-direct.  Mirrors
    method_final_mmmu_corrected.f10_persample; cross-fit, threshold tuned on TRAIN team accuracy."""
    from sklearn.linear_model import LogisticRegression
    n = len(ok7)
    ii = np.arange(n)
    team = np.zeros(n)
    took = np.zeros(n, bool)
    for f in range(K):
        te, tr = ii % K == f, ~(ii % K == f)
        mdl = LogisticRegression(max_iter=500, C=1.0).fit(X[tr], ok7[tr])
        gtr, gte = mdl.predict_proba(X[tr])[:, 1], mdl.predict_proba(X[te])[:, 1]
        bt, ba = 1.0, -1.0
        for t in np.unique(np.quantile(gtr, np.linspace(0, 1, 41))):
            a = np.where(gtr >= t, ok7[tr], ok32[tr]).mean()
            if a > ba:
                ba, bt = a, t
        keep = gte >= bt
        team[te] = np.where(keep, ok7[te], ok32[te])
        took[te] = keep
    return team, float(took.mean())


def open_features_from_pool(tag, verifier_dir="ckpts/train/lora_verifier_disjoint"):
    """The SAME 7 features as beat32b_more.open_features, built from an in-session decoding-sweep pool.
    `seqlogprob` is an item-level column from the deployed cheap dump and is identical in every arm."""
    from src.training_methods import genframe_data as G
    import decoding_sweep_analyse as DSA
    DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
    CELL = dict(zip(DS, OPEN))
    lab, vsc, ref = DSA.load_judge(), DSA.load_vscores(), G.load_items()
    slp, strong = {}, {}
    for ds in DS:
        for l in open(f"{ROOT}/ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.jsonl"):
            if l.strip():
                r = json.loads(l)
                slp[(ds, int(r["idx"]))] = float(r.get("seqlogprob") or 0.0)
        j, e = {}, {}
        for l in open(f"{ROOT}/ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.judge.jsonl"):
            if l.strip():
                r = json.loads(l); j[int(r["idx"])] = int(r["judge_ok"])
        for l in open(f"{ROOT}/ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.jsonl"):
            if l.strip():
                r = json.loads(l); e[int(r["idx"])] = int(r["modal_ok"])
        strong[ds] = (j, e)
    pool = DSA.load_pool(tag, strict=False)
    if pool is None:
        return None
    out = {}
    for ds in DS:
        items = [it for it in ref if it["ds"] == ds]
        ok7j, ok7e, ok32j, ok32e, X = [], [], [], [], []
        for it in items:
            idx = it["idx"]; r = pool[(ds, idx)]
            preds = r["preds"][:8]
            sc = np.array([vsc.get((ds, idx, a), G.MISSING_SCORE) for a in preds], float)
            slj = np.array([int(lab.get((ds, idx, G.norm(a)), 0)) for a in preds], float)
            sle = np.array(r["oks_em"][:8], float)
            pick = int(np.argmax(sc)); pk = preds[pick]
            X.append([float(sc.max()), float(sc.max() - sc.min()), float(sc.mean()), float(sc.std()),
                      float(len(set(preds))), float(np.mean([p == pk for p in preds])), slp[(ds, idx)]])
            ok7j.append(slj[pick]); ok7e.append(sle[pick])
            ok32j.append(strong[ds][0][idx]); ok32e.append(strong[ds][1][idx])
        X = np.array(X)
        res = {}
        for cur, o7, o32 in (("judge", np.array(ok7j), np.array(ok32j)),
                             ("em", np.array(ok7e), np.array(ok32e))):
            team, took = f10_persample(o7, o32, X)
            res[cur] = dict(vec=team, ok32=o32, bo8=float(o7.mean()), esc=float(1 - took),
                            acc=float(team.mean()))
        out[CELL[ds]] = res
    return out


# ------------------------------------------------------------------ stages
def stage_nulltest():
    from src.training_methods import genframe_data as G
    D = load_mcq()
    z = frozen_vectors()
    r = G.null_test()
    ok, vt = veto_published_folds(D["PMC_VQA"])
    d = D["PMC_VQA"]
    print("N1 frozen metric        max_abs_dev =", r["max_abs_deviation"], "pass =", r["pass"])
    o, s = r["measured"]["oracle@8"], r["measured"]["sel_eff"]
    print("N2 identity selected = oracle@8 x sel_eff  residual =",
          abs(o * s - r["measured"]["selected"]),
          "| FORBIDDEN additive form over-predicts by",
          round(r["measured"]["greedy"] + s * (o - r["measured"]["greedy"]) - r["measured"]["selected"], 6))
    print("N3 published PMC veto   acc=%.4f delta=%.4f rate=%.4f  (published 0.5613 / 0.0095 / 0.4002)"
          % (ok.mean(), ok.mean() - d["ok32"].mean(), vt.mean()))
    print("N4 macro reconstruction direct=%.6f  accuracy_max=%.6f  compute_lean=%.6f"
          % tuple(float(np.mean([z[f"{c}|{a}"].mean() for c in CELLS8]))
                  for a in ("always_32b_direct", "method_accuracy_max_veto", "method_compute_lean")))


def stage_seeds(nseeds=10):
    D = load_mcq()
    z = frozen_vectors()
    open_amax = {c: float(z[f"{c}|method_accuracy_max_veto"].mean()) for c in OPEN}
    res = {}
    for rule in ("R0", "R4"):
        accs = {c: [] for c in MCQ}; vets = {c: [] for c in MCQ}
        vecs = {c: [] for c in MCQ}; sels = []
        for sd in range(nseeds):
            o, v, s = run_seed(D, sd, rule)
            for c in MCQ:
                accs[c].append(o[c].mean()); vets[c].append(v[c].mean()); vecs[c].append(o[c])
            sels += [(a, b, tuple(dd)) for a, b, dd in s]
        macro = [float((sum(accs[c][i] for c in MCQ) + sum(open_amax.values())) / 8) for i in range(nseeds)]
        res[rule] = dict(macro=float(np.mean(macro)), macro_sd=float(np.std(macro)),
                         macro_range=[float(min(macro)), float(max(macro))],
                         per={c: dict(acc=float(np.mean(accs[c])), veto=float(np.mean(vets[c]))) for c in MCQ},
                         deploy_freq={c: float(np.mean([c in dd for _, _, dd in sels])) for c in MCQ} if sels else {})
        np.savez(os.path.join(CACHE, f"vecs_{rule}.npz"), **{c: np.mean(vecs[c], axis=0) for c in MCQ})
        print(rule, json.dumps(res[rule], indent=1))
    json.dump(res, open(os.path.join(CACHE, "indep_seeds.json"), "w"), indent=1)


def stage_null(nperm=200, seeds=(101, 102)):
    """Permutation null: (ok7, ok32) permuted JOINTLY within each cell; the ENTIRE nested-CV pipeline
    re-run.  Resumable, one JSON row per replicate, per-replicate error guard.  rep = -1 is the real data."""
    D = load_mcq()
    out = os.path.join(ART, "_hpv_veto_null_indep.jsonl")
    direct = {c: float(D[c]["ok32"].mean()) for c in MCQ}
    stat = lambda o: sum(o[c].mean() - direct[c] for c in MCQ) / 8.0
    done = set()
    if os.path.exists(out):
        for ln in open(out):
            try:
                done.add(json.loads(ln)["rep"])
            except Exception:
                pass
    f = open(out, "a")
    for rep in range(-1, nperm):
        if rep in done:
            continue
        t0 = time.time()
        try:
            if rep < 0:
                Dp = D
            else:
                rg = np.random.default_rng(900000 + rep)
                Dp = {}
                for c in MCQ:
                    p = rg.permutation(len(D[c]["ok7"]))
                    Dp[c] = dict(ok7=D[c]["ok7"][p], ok32=D[c]["ok32"][p], c7=D[c]["c7"])
            row = {"rep": rep}
            for rule in ("R0", "R4"):
                row[rule] = float(np.mean([stat(run_seed(Dp, sd, rule)[0]) for sd in seeds]))
            row["R4_minus_R0"] = row["R4"] - row["R0"]
            row["secs"] = round(time.time() - t0, 1)
            f.write(json.dumps(row) + "\n"); f.flush()
            print(rep, row, flush=True)
        except Exception as e:                                     # per-replicate guard
            f.write(json.dumps({"rep": rep, "error": repr(e)}) + "\n"); f.flush()
            print("ERR", rep, repr(e), flush=True)
    f.close()


def stage_f10():
    arms = {}
    for tag in ["T07r_s0", "T07r_s1", "T07r_s2", "T04_s0", "T04_s1", "T04_s2"]:
        a = open_features_from_pool(tag)
        if a is None:
            print("missing", tag); continue
        arms[tag] = a
        print(tag, {c: round(a[c]["judge"]["acc"], 4) for c in OPEN}, flush=True)
    np.savez(os.path.join(CACHE, "f10_pools.npz"),
             **{f"{t}|{c}|{cur}": arms[t][c][cur]["vec"] for t in arms for c in arms[t] for cur in ("judge", "em")},
             **{f"{t}|{c}|{cur}|ok32": arms[t][c][cur]["ok32"] for t in arms for c in arms[t] for cur in ("judge", "em")})
    json.dump({t: {c: {cur: {k: v for k, v in arms[t][c][cur].items() if k not in ("vec", "ok32")}
                       for cur in ("judge", "em")} for c in arms[t]} for t in arms},
              open(os.path.join(CACHE, "f10_pools.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["nulltest", "seeds", "null", "f10"])
    ap.add_argument("--nperm", type=int, default=200)
    a = ap.parse_args()
    {"nulltest": stage_nulltest, "seeds": stage_seeds, "f10": stage_f10,
     "null": lambda: stage_null(a.nperm)}[a.stage]()
