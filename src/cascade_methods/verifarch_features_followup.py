#!/usr/bin/env python3
"""verifarch_features_followup.py -- three diagnostics the first pass demanded.  Every hyperparameter is
selected on a HELD-OUT SPLIT INSIDE THE DISJOINT TRAIN POOLS (grouped by normalized question text); the
2,345 eval items are scored once per selected config, never used for selection.

  (1) OBJECTIVE SWEEP.  Is "listwise softmax loses to pointwise BCE on identical features" a real property
      of the objective, or an under-trained MLP?  Each of {pointwise BCE, listwise softmax, within-question
      Bradley-Terry} gets the same (epochs) grid at fixed width/lr, selected by held-out train sel_eff.
  (2) MODEL-CLASS SWEEP.  The first pass had HGB at sel_eff 0.708 / AUROC 0.714 against the MLP's
      0.770 / 0.897 on identical features -- were the trees genuinely worse, or over-fitted to the
      PathVQA-dominated training mixture?
  (3) COMPLEMENTARITY with the incumbent LoRA verifier: within-pool rank correlation, pick agreement,
      pair-oracle, and the (known-losing) naive rank fusion control.

  CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/verifarch_features_followup.py
  -> results/cascade_methods/artifacts/verifarch_features_followup_2026-08-04.json
"""
import json, os, sys
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

torch.set_num_threads(4)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
sys.argv = [sys.argv[0]]
src = open(J("src/cascade_methods/verifarch_features.py")).read().split("if __name__ ==")[0]
G = {"__name__": "verifarch_features"}
exec(compile(src, "verifarch_features.py", "exec"), G)

NBOOT, SEED = 10000, 0
CACHE = J("data/verifarch/feat_cache.npz")
ev, tr = G["load_eval"](), G["load_train"]()
if os.path.exists(CACHE):
    z = np.load(CACHE)
    Xtr, Ytr, Xev, Yev = z["Xtr"], z["Ytr"], z["Xev"], z["Yev"]
else:
    prior = G["build_prior"](tr)
    Xtr, Ytr, _ = G["featurize"](tr, prior, loo=True)
    Xev, Yev, _ = G["featurize"](ev, prior, loo=False)
    np.savez_compressed(CACHE, Xtr=Xtr, Ytr=Ytr, Xev=Xev, Yev=Yev)
qgroups = np.array([G["qnorm"](i["question"]) for i in tr])
_, qg = np.unique(qgroups, return_inverse=True)
print(f"train {Xtr.shape} eval {Xev.shape} q-groups {qg.max()+1} dev={DEV}", flush=True)

Sinc = np.stack([np.asarray(i["incumbent"], float) for i in ev])
st_inc, ok_inc, orc = G["sel_stats"](Sinc, Yev)
# ONE held-out split inside train, grouped by normalized question text (model selection only)
tri, tei = next(iter(GroupKFold(n_splits=4).split(Xtr, groups=qg)))
print(f"model-selection split inside train: fit {len(tri)} / held-out {len(tei)} items", flush=True)


def eff(S, Y):
    st, ok, _ = G["sel_stats"](S, Y)
    return st["sel_eff"], st, ok


def report(S, tag):
    e, st, ok = eff(S, Yev)
    st["auroc"] = float(roc_auc_score(Yev.reshape(-1), S.reshape(-1)))
    st["ci_sel_eff"] = G["boot_ci"](ok, orc, NBOOT, np.random.default_rng(SEED))
    st["vs_incumbent"] = G["paired_boot"](ok, orc, ok_inc, NBOOT, np.random.default_rng(SEED))
    st["per_dataset"] = {}
    for ds in G["EVAL_SETS"]:
        m = np.array([i["ds"] == ds for i in ev])
        s2, o2, r2 = G["sel_stats"](S[m], Yev[m])
        s2["vs_incumbent"] = G["paired_boot"](o2, r2, ok_inc[m], NBOOT, np.random.default_rng(SEED))
        st["per_dataset"][ds] = s2
    print(f"  == {tag}: EVAL sel_eff={e:.4f} {['%.4f'%x for x in st['ci_sel_eff']]} auroc={st['auroc']:.4f} "
          f"d_vs_inc={st['vs_incumbent']['d_sel_eff']:+.4f} "
          f"{['%.4f'%x for x in st['vs_incumbent']['ci_sel_eff']]}", flush=True)
    return st, ok


res = {"what": "objective sweep + model-class sweep + incumbent-complementarity for the feature-based "
                "discriminative selector; hyperparameters selected on a held-out split INSIDE the disjoint "
                "train pools (GroupKFold over normalized question text), eval scored once per config",
       "date": "2026-08-04", "incumbent_sel_eff": st_inc["sel_eff"], "nboot": NBOOT, "seed": SEED,
       "device": DEV, "code": "src/cascade_methods/verifarch_features_followup.py",
       "model_selection_split": {"fit_items": int(len(tri)), "heldout_items": int(len(tei)),
                                 "grouped_by": "normalized question text, GroupKFold n_splits=4, first fold"}}

# ------------------------------------------------------------------ (1) objective sweep
EPOCHS = (300, 1500, 5000, 15000)
obj, ok_by_loss = {}, {}
for loss in ("pointwise", "listwise", "bt"):
    rows = []
    for ep in EPOCHS:
        f = G["fit_mlp"](Xtr[tri], Ytr[tri], loss, SEED, epochs=ep, hid=64, lr=3e-3, dev=DEV)[0]
        c = eff(f(Xtr[tei]), Ytr[tei])[0]
        rows.append({"epochs": ep, "heldout_train_sel_eff": c})
        print(f"  [{loss:9s}] ep={ep:6d} heldout_train_sel_eff={c:.4f}", flush=True)
    best = max(rows, key=lambda r: r["heldout_train_sel_eff"])
    f = G["fit_mlp"](Xtr, Ytr, loss, SEED, epochs=best["epochs"], hid=64, lr=3e-3, dev=DEV)[0]
    st, ok = report(f(Xev), f"objective={loss} (epochs={best['epochs']})")
    obj[loss] = {"selected_config": best, "grid": rows, "eval": st}
    ok_by_loss[loss] = ok
res["objective_sweep"] = obj
res["listwise_minus_pointwise_on_identical_features"] = G["paired_boot"](
    ok_by_loss["listwise"], orc, ok_by_loss["pointwise"], NBOOT, np.random.default_rng(SEED))
res["bt_minus_pointwise_on_identical_features"] = G["paired_boot"](
    ok_by_loss["bt"], orc, ok_by_loss["pointwise"], NBOOT, np.random.default_rng(SEED))
print("listwise - pointwise:", res["listwise_minus_pointwise_on_identical_features"]["d_sel_eff"],
      res["listwise_minus_pointwise_on_identical_features"]["ci_sel_eff"], flush=True)

# ------------------------------------------------------------------ (2) model-class sweep
HG = [(it, lv, ms, l2) for it in (60, 200) for lv in (7, 31) for ms in (50, 400) for l2 in (0.0, 10.0)]
rows = []
for it, lv, ms, l2 in HG:
    m = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.06, max_leaf_nodes=lv,
                                       min_samples_leaf=ms, l2_regularization=l2,
                                       early_stopping=False, random_state=SEED)
    m.fit(Xtr[tri].reshape(-1, Xtr.shape[-1]), Ytr[tri].reshape(-1))
    p = m.predict_proba(Xtr[tei].reshape(-1, Xtr.shape[-1]))[:, 1].reshape(Xtr[tei].shape[:2])
    c = eff(p, Ytr[tei])[0]
    rows.append({"max_iter": it, "max_leaf_nodes": lv, "min_samples_leaf": ms,
                 "l2_regularization": l2, "heldout_train_sel_eff": c})
    print(f"  [hgb] iter={it:4d} leaves={lv:3d} msl={ms:4d} l2={l2:<5g} heldout={c:.4f}", flush=True)
best = max(rows, key=lambda r: r["heldout_train_sel_eff"])
m = HistGradientBoostingClassifier(max_iter=best["max_iter"], learning_rate=0.06,
                                   max_leaf_nodes=best["max_leaf_nodes"],
                                   min_samples_leaf=best["min_samples_leaf"],
                                   l2_regularization=best["l2_regularization"],
                                   early_stopping=False, random_state=SEED)
m.fit(Xtr.reshape(-1, Xtr.shape[-1]), Ytr.reshape(-1))
S = m.predict_proba(Xev.reshape(-1, Xev.shape[-1]))[:, 1].reshape(Xev.shape[:2])
st, _ = report(S, f"hgb tuned {best}")
res["hgb_sweep"] = {"selected_config": best, "grid": rows, "eval": st}

# ------------------------------------------------------------------ (3) complementarity
from scipy.stats import spearmanr
fp = G["fit_mlp"](Xtr, Ytr, "pointwise", SEED,
                  epochs=obj["pointwise"]["selected_config"]["epochs"], hid=64, lr=3e-3, dev=DEV)[0]
best_S = fp(Xev)
_, _, okp = eff(best_S, Yev)
rho = [spearmanr(best_S[i], Sinc[i]).correlation for i in range(len(ev))
       if len(set(Sinc[i])) > 1 and len(set(best_S[i].tolist())) > 1]
comp = {"spearman_within_pool_mean": float(np.nanmean(rho)), "n_pools_with_variation": len(rho),
        "same_pick_rate": float(np.mean([np.argmax(best_S[i]) == np.argmax(Sinc[i]) for i in range(len(ev))])),
        "pair_oracle_sel_eff": float(np.maximum(okp, ok_inc)[orc == 1].mean()),
        "pair_oracle_headroom": float(np.maximum(okp, ok_inc)[orc == 1].mean() - ok_inc[orc == 1].mean()),
        "feature_only_correct": float(np.mean((okp == 1) & (ok_inc == 0))),
        "incumbent_only_correct": float(np.mean((okp == 0) & (ok_inc == 1)))}
rk = lambda S: np.argsort(np.argsort(S, axis=1), axis=1).astype(float)
for w in (0.25, 0.5, 0.75):
    e2, st2, ok2 = eff(w * rk(best_S) + (1 - w) * rk(Sinc), Yev)
    comp[f"rank_fusion_w{w}"] = {"sel_eff": e2,
                                 "vs_incumbent": G["paired_boot"](ok2, orc, ok_inc, NBOOT,
                                                                  np.random.default_rng(SEED))}
    print(f"  rank_fusion w={w}: sel_eff={e2:.4f} d={comp[f'rank_fusion_w{w}']['vs_incumbent']['d_sel_eff']:+.4f}",
          flush=True)
res["complementarity_with_incumbent"] = comp

json.dump(res, open(J("results/cascade_methods/artifacts/verifarch_features_followup_2026-08-04.json"), "w"),
          indent=1)
print("wrote results/cascade_methods/artifacts/verifarch_features_followup_2026-08-04.json", flush=True)
