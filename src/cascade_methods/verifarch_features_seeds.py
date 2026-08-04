#!/usr/bin/env python3
"""verifarch_features_seeds.py -- seed stability of the ARM A (deployable, frozen, zero-inference) feature
MLP.  The headline "statistically tied with the trained LoRA verifier" was measured at seed 0; this refits
the identical configuration at 10 seeds and reports the spread, plus the seed-averaged ensemble.

  CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/verifarch_features_seeds.py
  -> results/cascade_methods/artifacts/verifarch_features_seeds_2026-08-04.json
"""
import json, os, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

torch.set_num_threads(4)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
sys.argv = [sys.argv[0]]
src = open(J("src/cascade_methods/verifarch_features.py")).read().split("if __name__ ==")[0]
G = {"__name__": "verifarch_features"}
exec(compile(src, "verifarch_features.py", "exec"), G)

NBOOT = 10000
ev, tr = G["load_eval"](), G["load_train"]()
z = np.load(J("data/verifarch/feat_cache.npz"))
Xtr, Ytr, Xev, Yev = z["Xtr"], z["Ytr"], z["Xev"], z["Yev"]
Sinc = np.stack([np.asarray(i["incumbent"], float) for i in ev])
st_inc, ok_inc, orc = G["sel_stats"](Sinc, Yev)

rows, Ss = [], []
for s in range(10):
    f = G["fit_mlp"](Xtr, Ytr, "pointwise", s, epochs=300, hid=64, lr=3e-3, dev=DEV)[0]
    S = f(Xev)
    Ss.append((S - S.mean()) / (S.std() + 1e-9))
    st, ok, _ = G["sel_stats"](S, Yev)
    rows.append({"seed": s, "sel_eff": st["sel_eff"], "sel_acc": st["sel_acc"],
                 "auroc": float(roc_auc_score(Yev.reshape(-1), S.reshape(-1)))})
    print(f"  seed {s}: sel_eff={st['sel_eff']:.4f} sel_acc={st['sel_acc']:.4f} auroc={rows[-1]['auroc']:.4f}",
          flush=True)
Sens = np.mean(Ss, axis=0)
st_e, ok_e, _ = G["sel_stats"](Sens, Yev)
st_e["auroc"] = float(roc_auc_score(Yev.reshape(-1), Sens.reshape(-1)))
st_e["ci_sel_eff"] = G["boot_ci"](ok_e, orc, NBOOT, np.random.default_rng(0))
st_e["vs_incumbent"] = G["paired_boot"](ok_e, orc, ok_inc, NBOOT, np.random.default_rng(0))
st_e["per_dataset"] = {}
for ds in G["EVAL_SETS"]:
    m = np.array([i["ds"] == ds for i in ev])
    s2, o2, r2 = G["sel_stats"](Sens[m], Yev[m])
    s2["vs_incumbent"] = G["paired_boot"](o2, r2, ok_inc[m], NBOOT, np.random.default_rng(0))
    st_e["per_dataset"][ds] = s2
# ---- stratified by GOLD ANSWER LENGTH: the repo localises the failure to short one-token contrasts
import re as _re
gw = np.array([len(_re.findall(r"[A-Za-z0-9]+", str(i["gold"]))) for i in ev])
strata = {"short_le3w": gw <= 3, "medium_4to8w": (gw >= 4) & (gw <= 8), "long_gt8w": gw > 8}
lat = np.array([bool(G["LAT"].search(G["norm"](i["gold"]))) for i in ev])
strata["gold_has_laterality_or_position"] = lat
strat = {}
for nm, msk in strata.items():
    m = msk & (orc == 1)
    strat[nm] = {"n_items": int(msk.sum()), "n_recoverable": int(m.sum()),
                 "incumbent_sel_eff": float(ok_inc[m].mean()) if m.sum() else None,
                 "feature_ensemble_sel_eff": float(ok_e[m].mean()) if m.sum() else None,
                 "delta": float(ok_e[m].mean() - ok_inc[m].mean()) if m.sum() else None}
    print(f"  [{nm:30s}] n={int(msk.sum()):5d} inc={strat[nm]['incumbent_sel_eff']} "
          f"feat={strat[nm]['feature_ensemble_sel_eff']} d={strat[nm]['delta']}", flush=True)

v = [r["sel_eff"] for r in rows]
out = {"what": "seed stability of the ARM A frozen feature MLP (identical config, 10 seeds) + the "
                "seed-averaged ensemble (z-scored per seed, then averaged)",
       "date": "2026-08-04", "config": {"loss": "pointwise BCE", "epochs": 300, "hid": 64, "lr": 3e-3,
                                        "wd": 1e-3, "features": 39},
       "incumbent_sel_eff": st_inc["sel_eff"],
       "per_seed": rows,
       "sel_eff_mean": float(np.mean(v)), "sel_eff_sd": float(np.std(v)),
       "sel_eff_min": float(np.min(v)), "sel_eff_max": float(np.max(v)),
       "seed_ensemble": st_e,
       "stratified_by_gold_answer_length": strat,
       "code": "src/cascade_methods/verifarch_features_seeds.py"}
print(f"  mean={np.mean(v):.4f} sd={np.std(v):.4f} range=[{np.min(v):.4f},{np.max(v):.4f}]  "
      f"ensemble={st_e['sel_eff']:.4f} d_vs_inc={st_e['vs_incumbent']['d_sel_eff']:+.4f} "
      f"{['%.4f' % x for x in st_e['vs_incumbent']['ci_sel_eff']]}", flush=True)
json.dump(out, open(J("results/cascade_methods/artifacts/verifarch_features_seeds_2026-08-04.json"), "w"),
          indent=1)
print("wrote results/cascade_methods/artifacts/verifarch_features_seeds_2026-08-04.json", flush=True)
