#!/usr/bin/env python3
"""
methods.py - registry of TRAINING-FREE cascade gating methods, each expressed as a function that
produces a per-sample ESCALATION SCORE (higher == escalate) on the calibration set and on an eval
pool, using ONLY the cheap 7B's signals (+ 7B self-correctness on calibration). No 32B labels on
calibration are used here, so every method in this file runs on the existing checkpoints with no
GPU. (Deferral-aware methods that need 32B-on-calibration live in methods_deferral.py once that
run lands.)

Each method: fit(D) -> object with .score_calib (array over calib) and .score_eval(pool) -> array.
Convention: higher score => more likely to escalate (less confident / more benefit).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# single-signal scores: (signal_key, sign) so that score = sign*signal is "escalation priority"
SINGLE = {
    "margin":      ("margin", -1.0),      # deployed baseline: low top1-top2 logprob gap -> escalate
    "maxlogprob":  ("maxlogprob", -1.0),  # Chow's rule / max-logit
    "top1prob":    ("top1prob", -1.0),    # softmax response (max softmax prob)
    "prob_margin": ("prob_margin", -1.0),
    "margin_top2": ("margin_top2", -1.0),
    "entropy":     ("entropy", +1.0),     # predictive entropy
    "entropy2":    ("entropy2", +1.0),    # top-2 restricted entropy
    "neg_energy":  ("neg_energy", +1.0),  # energy score = -logsumexp(logits)
    "gini":        ("gini", +1.0),
    "top2mass":    ("top2mass", -1.0),
}

COMBO_FEATS = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "gini", "neg_energy", "n_opts"]


class SingleSignal:
    def __init__(self, key, sign):
        self.key, self.sign = key, sign

    def fit(self, D):
        self.score_calib = self.sign * D.calib["sig"][self.key]
        return self

    def score_eval(self, pool):
        return self.sign * pool["sig"][self.key]


class LogisticCorrectness:
    """Predict P(7B correct) from a feature vector (calib-only, label = 7B self-correctness).
    Escalation score = -P(correct). This is the 'learned correctness gate' class (training-free:
    a tiny logistic fit on the held-out calibration split)."""
    def __init__(self, feats=COMBO_FEATS):
        self.feats = feats

    def _X(self, sig):
        return np.column_stack([sig[f] for f in self.feats])

    def fit(self, D):
        X = self._X(D.calib["sig"]); y = D.calib["ok"]
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(X, y)
        self.score_calib = -self.clf.predict_proba(X)[:, 1]
        return self

    def score_eval(self, pool):
        return -self.clf.predict_proba(self._X(pool["sig"]))[:, 1]


class GBMCorrectness:
    """HistGradientBoosting predict P(7B correct); escalation score = -P(correct)."""
    def __init__(self, feats=COMBO_FEATS):
        self.feats = feats

    def _X(self, sig):
        return np.column_stack([sig[f] for f in self.feats])

    def fit(self, D):
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = self._X(D.calib["sig"]); y = D.calib["ok"]
        self.clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.08,
                                                  l2_regularization=1.0, early_stopping=True,
                                                  random_state=42).fit(X, y)
        self.score_calib = -self.clf.predict_proba(X)[:, 1]
        return self

    def score_eval(self, pool):
        return -self.clf.predict_proba(self._X(pool["sig"]))[:, 1]


class Conformal:
    """Split-conformal max-prob deferral (CP-router): nonconformity = 1 - top1prob; escalate if
    eval score above the calib (1-alpha) quantile. With alpha set to the calib error rate this is
    a coverage-guaranteed max-prob threshold."""
    def fit(self, D):
        self.score_calib = 1.0 - D.calib["sig"]["top1prob"]
        return self

    def score_eval(self, pool):
        return 1.0 - pool["sig"]["top1prob"]


def registry():
    reg = {f"single:{n}": SingleSignal(k, s) for n, (k, s) in SINGLE.items()}
    reg["combo:logistic"] = LogisticCorrectness()
    reg["combo:gbm"] = GBMCorrectness()
    reg["conformal:maxprob"] = Conformal()
    return reg
