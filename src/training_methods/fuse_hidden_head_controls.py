#!/usr/bin/env python3
"""fuse_hidden_head_controls.py -- robustness + falsification controls for the one positive result
in verifarch_hidden_2026-08-04.json: rank-fusing the incumbent LoRA verifier with a discriminative
head over the frozen generator's hidden states.

The question this script exists to answer is NOT "is the fusion better" (that is already measured)
but "is the gain attributable to the DISCRIMINATIVE HEAD, or would fusing the incumbent with any
second same-family score do the same?".  Controls, all on the same 2345 items:

  C1  fuse(incumbent, base zero-shot P(Yes))     -- a second GENERATIVE opinion from the same
                                                    frozen model, same prompt, no head at all.
                                                    If this also gains, the head is not the cause.
  C2  fuse(incumbent, self-consistency count)    -- a second score that is free and non-model.
  C3  fuse(incumbent, random score)              -- pure tie-breaking null; must be ~0.
  C4  tie handling: average-rank vs argsort-rank -- the incumbent's stored scores have exact ties,
                                                    so ranking convention must not carry the result.
  C5  z-scored mean instead of rank mean         -- the gain must not be a rank-transform artifact.
  C6  head variants (listwise / per-benchmark)   -- is the effect architecture-specific?

Everything is parameter-free at fusion time (no weight is fit on eval), and every head is trained
only on the image-disjoint pool.

  python3 src/training_methods/fuse_hidden_head_controls.py
"""
import os, sys, json, math, hashlib
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import importlib.util
spec = importlib.util.spec_from_file_location("fh", os.path.join(ROOT, "src/training_methods/fit_hidden_head.py"))
_argv = sys.argv; sys.argv = ["fit_hidden_head"]
fh = importlib.util.module_from_spec(spec); spec.loader.exec_module(fh); sys.argv = _argv

NBOOT = 10000
items = fh.load_incumbent()
picks_inc = [int(np.argmax(it["scores"])) for it in items]
s_inc, got_inc, rec = fh.sel_stats(items, picks_inc)


def rank_avg(v):
    """average ranks for ties, scaled to [0,1] -- the tie-safe ranking convention"""
    v = np.asarray(v, float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float); r[o] = np.arange(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r / max(len(v) - 1, 1)


def rank_argsort(v):
    return np.argsort(np.argsort(np.asarray(v, float))) / max(len(v) - 1, 1)


def zscore(v):
    v = np.asarray(v, float); sd = v.std()
    return (v - v.mean()) / (sd if sd > 1e-9 else 1.0)


def score_arm(fn, tag):
    picks = [int(np.argmax(fn(it))) for it in items]
    s, got, _ = fh.sel_stats(items, picks)
    bo = fh.paired_boot(items, got, got_inc, rec, NBOOT)
    out = {"tag": tag, "sel_eff": round(s["sel_eff"], 6), "acc": round(s["acc"], 6),
           "contested_sel_eff": round(fh.contested(items, picks)[0]["sel_eff"], 6),
           "per_ds": {k: round(v["sel_eff"], 6) for k, v in fh.per_ds(items, picks).items()},
           "d_sel_eff": round(bo["d_sel_eff"], 6), "d_sel_eff_ci": [round(x, 6) for x in bo["d_sel_eff_ci"]],
           "d_acc": round(bo["d_acc"], 6), "d_acc_ci": [round(x, 6) for x in bo["d_acc_ci"]],
           "guardrail_clean": all(fh.per_ds(items, picks)[d]["sel_eff"] >=
                                  fh.per_ds(items, picks_inc)[d]["sel_eff"] for d in fh.EVAL_DS)}
    print(f"  {tag:<46s} sel_eff={s['sel_eff']:.4f} d={bo['d_sel_eff']:+.4f} "
          f"[{bo['d_sel_eff_ci'][0]:+.4f},{bo['d_sel_eff_ci'][1]:+.4f}] guard={out['guardrail_clean']}",
          flush=True)
    return out


def train_head(cfg, per_domain=False, mode="grader"):
    """Refit a head on the FULL image-disjoint training pool and score every eval candidate."""
    ztr, mtr = fh.load_cache("feats_hidden", mode, "train")
    zev, mev = fh.load_cache("feats_hidden", mode, "eval")
    layers = list(ztr["layers"]); li = layers.index(cfg["layer"])
    Xtr, ytr, ktr, gtr, _ = fh.build_matrix(ztr, mtr, li, cfg["pooling"], cfg.get("setrel", 0))
    Xev, yev, kev, gev, _ = fh.build_matrix(zev, mev, li, cfg["pooling"], cfg.get("setrel", 0))
    qid = [f"{a}|{b}" for (a, b, c) in ktr]
    smap = {}
    if per_domain:
        DOMAIN = {"slake_open": ["slake_open_train"], "vqa_rad_open": ["vqa_rad_open_train"],
                  "pathvqa_open": ["pathvqa_open_train"]}
        tr_ds = np.array([r["ds"] for r in mtr["rows"] if r.get("n_tok", -1) > 0])
        for eds, srcs in DOMAIN.items():
            sel = np.isin(tr_ds, srcs)
            mu, sd = Xtr[sel].mean(0), Xtr[sel].std(0) + 1e-6
            m = fh.fit_head((Xtr[sel] - mu) / sd, ytr[sel], [qid[i] for i in np.where(sel)[0]],
                            objective=cfg["objective"], hidden=cfg["hidden"], wd=cfg["wd"],
                            epochs=cfg["epochs"], seed=0)
            keep = [i for i in range(len(kev)) if kev[i][0] == eds]
            sv = fh.predict(m, (Xev[keep] - mu) / sd)
            for j, i in enumerate(keep):
                smap[kev[i]] = float(sv[j])
    else:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        m = fh.fit_head((Xtr - mu) / sd, ytr, qid, objective=cfg["objective"], hidden=cfg["hidden"],
                        wd=cfg["wd"], epochs=cfg["epochs"], seed=0)
        sv = fh.predict(m, (Xev - mu) / sd)
        smap = {kev[i]: float(sv[i]) for i in range(len(kev))}
    return smap


def base_zeroshot_map():
    zev, mev = fh.load_cache("feats_hidden", "grader", "eval")
    yn = zev["yesno"]; sm = {}
    for i, r in enumerate(mev["rows"]):
        if r.get("n_tok", -1) > 0 and not np.isnan(yn[i, 0]):
            py, pn = math.exp(yn[i, 0]), math.exp(yn[i, 1])
            sm[(r["ds"], r["idx"], r["na"])] = py / (py + pn) if (py + pn) > 0 else 0.5
    return sm


def as_vec(smap, it, default=-1e9):
    return np.array([smap.get((it["ds"], it["idx"], fh.norm(a)), default) for a in it["preds"]], float)


if __name__ == "__main__":
    CV = {"layer": 21, "pooling": "last", "objective": "bce", "hidden": 256, "wd": 0.01,
          "epochs": 30, "setrel": 0}   # the CV-selected config, verbatim from the artifact
    print("refitting heads on the image-disjoint pool ...", flush=True)
    H_cv = train_head(CV)
    H_lw = train_head({**CV, "objective": "listwise"})
    H_pb = train_head(CV, per_domain=True)
    ZS = base_zeroshot_map()
    rng = np.random.default_rng(0)
    RND = {k: float(rng.random()) for k in H_cv}
    SC = {}
    for it in items:
        c = Counter(fh.norm(a) for a in it["preds"])
        for a in it["preds"]:
            SC[(it["ds"], it["idx"], fh.norm(a))] = c[fh.norm(a)]

    res = {"what": "robustness + falsification controls for the incumbent x hidden-head rank fusion",
           "date": "2026-08-04", "nboot": NBOOT,
           "incumbent": {"sel_eff": round(s_inc["sel_eff"], 6), "acc": round(s_inc["acc"], 6)},
           "arms": []}
    print("\nHEAD ALONE (no fusion):", flush=True)
    for tag, H in [("head_cv_selected(L21/last/bce/h256)", H_cv), ("head_listwise(post-hoc)", H_lw),
                   ("head_per_benchmark", H_pb), ("base_zeroshot_pyes", ZS)]:
        res["arms"].append(score_arm(lambda it, H=H: as_vec(H, it), f"ALONE {tag}"))

    print("\nFUSION with the incumbent (rank-average, tie-safe average ranks):", flush=True)
    for tag, H in [("head_cv_selected", H_cv), ("head_listwise(post-hoc)", H_lw),
                   ("head_per_benchmark", H_pb),
                   ("C1 base_zeroshot_pyes [KEY CONTROL]", ZS),
                   ("C2 self_consistency_count", SC), ("C3 random [null]", RND)]:
        res["arms"].append(score_arm(
            lambda it, H=H: rank_avg(np.array(it["scores"], float)) + rank_avg(as_vec(H, it)),
            f"FUSE_rankavg incumbent+{tag}"))

    print("\nTIE-CONVENTION and TRANSFORM robustness (CV-selected head):", flush=True)
    res["arms"].append(score_arm(
        lambda it: rank_argsort(np.array(it["scores"], float)) + rank_argsort(as_vec(H_cv, it)),
        "FUSE_rankargsort incumbent+head_cv_selected"))
    res["arms"].append(score_arm(
        lambda it: zscore(np.array(it["scores"], float)) + zscore(as_vec(H_cv, it, default=-5.0)),
        "FUSE_zmean incumbent+head_cv_selected"))

    # generator-prompt head, if its features exist (the "last answer token probe" representation)
    try:
        GCFG = json.load(open(os.path.join(
            ROOT, "results/cascade_methods/artifacts/verifarch_hidden_generatorprompt_2026-08-04.json"))
        )["arms"]["generator"]["cv_selected"]
        GCFG = {k: GCFG[k] for k in ["layer", "pooling", "objective", "hidden", "wd", "epochs", "setrel"]}
        H_gen = train_head(GCFG, mode="generator")
        print(f"\nGENERATOR-PROMPT head (CV-selected {GCFG}):", flush=True)
        res["generator_prompt_cv_selected_config"] = GCFG
        res["arms"].append(score_arm(lambda it: as_vec(H_gen, it), "ALONE head_generator_prompt"))
        res["arms"].append(score_arm(
            lambda it: rank_avg(np.array(it["scores"], float)) + rank_avg(as_vec(H_gen, it)),
            "FUSE_rankavg incumbent+head_generator_prompt"))
        res["arms"].append(score_arm(
            lambda it: rank_avg(np.array(it["scores"], float)) + rank_avg(as_vec(H_cv, it))
                       + rank_avg(as_vec(H_gen, it)),
            "FUSE_rankavg incumbent+head_grader+head_generator"))
    except Exception as e:
        print(f"\n(generator-prompt head skipped: {str(e)[:120]})", flush=True)

    op = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_hidden_fusion_controls_2026-08-04.json")
    json.dump(res, open(op, "w"), indent=1)
    print(f"\nwrote {op}", flush=True)
