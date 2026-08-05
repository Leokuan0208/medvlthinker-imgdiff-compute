#!/usr/bin/env python3
"""genframe_selector.py -- THE FROZEN, RELOADABLE best-of-N open-text selector.

This is the deployable artifact of the 2026-08-05 comparative-verification round
(results/cascade_methods/docs/current/COMPARATIVE_VERIFIER_2026-08-05.md sec.5).  Before this
file the recommended selector existed only as a *scoring procedure* inside
src/training_methods/integrate_eval.py -- i.e. it had to be re-TRAINED to be used, and a
re-trained head is a draw from a seed distribution, not an artifact.  Here the 8 fitted heads
and the train mu/sd are on disk and this module only ever does a forward pass.

THE RECIPE (frozen; every element below is load-bearing and was measured to matter)
-----------------------------------------------------------------------------------
  features   generator-frame hidden states, LAYER 21, SPAN pooling, 3584-d
             (feats_hidden/generator_{train,eval}_s*of2.npz, row order 'concat')
  standardise  x <- (x - mu_TRAIN) / sd_TRAIN, sd = train std + 1e-6.
             !! Omitting this step costs 0.0075 sel_eff -- larger than most effects in the
             round.  mu/sd are frozen in standardizer.npz, NOT recomputed at inference.
  head       Linear(3584,256) -> GELU -> Linear(256,1), 918,529 params, one per seed
  objective  Bradley-Terry over (correct, incorrect) pairs inside a question ('bt')
  training   AdamW lr 1e-3, wd 1e-2, 30 epochs, group-batch 64, seeds 0..7
  readout    per seed: rank_avg of that seed's logits over the N slots of the question;
             head_rank = MEAN of the 8 seed rank vectors
  fusion     final = rank_avg(incumbent verifier slot scores) + rank_avg(head_rank)
             ('rank_avg' = average ranks for ties -- rank_argsort gives 0.798365, NOT this)
  pick       np.argmax over the N slots, FIRST-INDEX tie-break

  Endpoint reproduced by verify():  sel_eff 0.810627 | acc 0.507463
  per set slake_open 0.885362 / vqa_rad_open 0.809524 / pathvqa_open 0.756129

COST.  Zero extra VLM forward passes over the already-deployed recipe: the same cached 3584-d
vector is scored by 8 tiny MLPs (~1.8 MFLOP each) instead of 1.  The incumbent LoRA verifier
forward (3.823 passes/question) and the generator-frame extraction (3.814 passes/question) are
unchanged at 7.636674 total.

WHAT IS BOUGHT is DETERMINISM, not a new mechanism: the 1-seed recipe that is currently
deployed is a draw from [0.802452, 0.814033] (16-seed range) and this is a fixed point at
0.810627, +0.004087 [-0.004087, +0.012262] over it -- NOT significant.

USE
---
    from genframe_selector import FrozenSelector
    S = FrozenSelector.load()                    # ckpts/train/genframe_head_ens8
    slot_scores = S.slot_scores(feat_by_slot, incumbent_slot_scores)   # (n_slots,)
    pick = S.select(feat_by_slot, incumbent_slot_scores)               # int

    python3 src/training_methods/genframe_selector.py       # runs verify(), prints deviation
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import genframe_data as G  # noqa: E402

DEFAULT_DIR = os.path.join(G.ROOT, "ckpts/train/genframe_head_ens8")

#: what verify() must reproduce (COMPARATIVE_VERIFIER_2026-08-05.md sec.5, [INT])
PUBLISHED = {"sel_eff": 0.810627, "acc": 0.507463,
             "per_ds": {"slake_open": 0.885362, "vqa_rad_open": 0.809524,
                        "pathvqa_open": 0.756129}}


class FrozenSelector:
    """8 frozen heads + the frozen train standardiser + the frozen fusion rule."""

    def __init__(self, W, mu, sd, recipe):
        self.W = W                      # list of dicts of numpy arrays, one per seed
        self.mu = np.asarray(mu, np.float32)
        self.sd = np.asarray(sd, np.float32)
        self.recipe = recipe

    # ---------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str = DEFAULT_DIR) -> "FrozenSelector":
        import torch
        recipe = json.load(open(os.path.join(path, "recipe.json")))
        z = np.load(os.path.join(path, "standardizer.npz"))
        W = []
        for s in recipe["seeds"]:
            m = _build_head(recipe["in_dim"], recipe["hidden"])
            m.load_state_dict(torch.load(os.path.join(path, f"head_seed{s}.pt"),
                                         map_location="cpu", weights_only=True))
            m.eval()
            W.append(m)
        return cls(W, z["mu"], z["sd"], recipe)

    # ---------------------------------------------------------------- forward
    def standardize(self, H) -> np.ndarray:
        """(n, 3584) raw layer-21 span-pooled hidden states -> standardised with the TRAIN mu/sd."""
        return (np.asarray(H, np.float32) - self.mu) / self.sd

    def head_logits(self, H, standardized: bool = False) -> np.ndarray:
        """(n_seeds, n) raw head logits for n candidate feature vectors."""
        import torch
        X = np.asarray(H, np.float32) if standardized else self.standardize(H)
        with torch.no_grad():
            t = torch.tensor(X)
            return np.stack([m(t).numpy().astype(np.float64) for m in self.W])

    # ---------------------------------------------------------------- the selector
    def head_rank(self, H, standardized: bool = False) -> np.ndarray:
        """(n,) the 8-seed MEAN within-pool rank_avg of the head logits -- the ensemble readout."""
        L = self.head_logits(H, standardized)
        return np.mean([G.rank_avg(L[s]) for s in range(L.shape[0])], axis=0)

    def slot_scores(self, H, incumbent_scores, standardized: bool = False) -> np.ndarray:
        """THE selector score, one per slot of ONE question.

        H                 (n_slots, 3584) generator-frame vector of the candidate in each slot
                          (duplicated answers legitimately share a vector)
        incumbent_scores  (n_slots,) the incumbent LoRA verifier's P(correct) per slot
        """
        hr = self.head_rank(H, standardized)
        inc = np.asarray(incumbent_scores, float)
        if len(inc) != len(hr):
            raise ValueError(f"slot count mismatch: {len(hr)} features vs {len(inc)} incumbent scores")
        return G.rank_avg(inc) + G.rank_avg(hr)

    def select(self, H, incumbent_scores, standardized: bool = False) -> int:
        """Index of the selected slot.  argmax, FIRST-INDEX tie-break (the frozen pick rule)."""
        return int(np.argmax(self.slot_scores(H, incumbent_scores, standardized)))


def _build_head(d: int, hidden: int):
    """The head architecture, byte-identical to fit_hidden_head.Head(d, hidden, drop=0.0).

    Reproduced here (rather than imported) so the deployable selector does not depend on the
    training module.  freeze_selector.py asserts the two are structurally identical.
    """
    import torch.nn as nn

    class _Head(nn.Module):
        def __init__(self, d, hidden):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.0),
                                   nn.Linear(hidden, 1))

        def forward(self, x):
            return self.f(x).squeeze(-1)

    return _Head(d, hidden)


# ======================================================================================
# verification -- the selector must reproduce the published endpoint from disk alone
# ======================================================================================
def score_eval_pool(sel: FrozenSelector, layer=None, pooling=None):
    """Run the frozen selector over the whole 2345-question eval pool via genframe_data.

    Returns (slot-score dict ready for G.sel_eff, per-seed row logits)."""
    layer = layer or sel.recipe["layer"]
    pooling = pooling or sel.recipe["pooling"]
    ev = G.load_candidates("eval", sel.recipe["mode"], layers=[layer], pooling=(pooling,),
                           order=sel.recipe["row_order"])
    X = sel.standardize(ev.matrix(pooling, layer))
    L = sel.head_logits(X, standardized=True)                  # (n_seeds, n_rows)
    out = {}
    for q in ev.questions:
        rows = np.asarray(q.slot_rows)                          # (8,) row per slot
        hr = np.mean([G.rank_avg(L[s][rows]) for s in range(L.shape[0])], axis=0)
        out[(q.ds, q.idx)] = list(G.rank_avg(np.asarray(q.inc_scores, float)) + G.rank_avg(hr))
    return out, L


def verify(path: str = DEFAULT_DIR, tol: float = 1e-6) -> dict:
    """Reload from disk and reproduce the published cells through genframe_data's frozen metric."""
    sel = FrozenSelector.load(path)
    items = G.load_items()
    smap, L = score_eval_pool(sel)
    r = G.sel_eff(smap, items)
    measured = {"sel_eff": r["sel_eff"], "acc": r["acc"],
                "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in G.EVAL_DS}}
    dev = {"sel_eff": abs(measured["sel_eff"] - PUBLISHED["sel_eff"]),
           "acc": abs(measured["acc"] - PUBLISHED["acc"])}
    dev.update({f"per_ds.{d}": abs(measured["per_ds"][d] - PUBLISHED["per_ds"][d]) for d in G.EVAL_DS})
    # additionally: do the reloaded heads reproduce the training run's stored eval logits?
    ref_p = os.path.join(G.ROOT, "results/cascade_methods/artifacts/_integrate_parts",
                         "eval_head_seed_scores_cpu.npy")
    logit_dev = None
    if os.path.exists(ref_p):
        S = np.load(ref_p)[:L.shape[0]]
        logit_dev = float(np.max(np.abs(S - L.astype(np.float64))))
    mx = max(dev.values())
    return {"path": path, "measured": measured, "published": PUBLISHED, "abs_deviation": dev,
            "max_abs_deviation": mx, "pass": bool(mx <= tol), "tolerance": tol,
            "reloaded_vs_training_run_max_abs_logit_deviation": logit_dev,
            "note": "published cells are 6-dp rounded; deviations at 1e-6 are rounding, not disagreement."}


if __name__ == "__main__":
    import pprint
    v = verify(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
    print("SELECTOR RELOAD VERIFY  pass =", v["pass"], " max abs deviation =", v["max_abs_deviation"])
    print("  head-logit deviation vs the training run:",
          v["reloaded_vs_training_run_max_abs_logit_deviation"])
    pprint.pprint(v["measured"])
