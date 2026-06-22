#!/usr/bin/env python3
"""
methods_deferral.py - RECOVERABILITY-AWARE deferral gates (the novel direction). These need
32B-on-PMC-train labels (ckpts/gate_32b_pmctrain/) so that, on the CALIBRATION set, we know not
just "is the 7B wrong" but "would escalating actually FIX it" (32B right) and "would it BREAK a
correct answer" (32B wrong while 7B right). All training is calibration-only (a tiny logistic/GBM
on the held-out PMC-train split) -> still training-free; no fine-tuning of either VLM, and the
32B is used only as a black-box correctness oracle on calibration (it exposes no logprobs).

Feature vector x = the 7B cheap-leg signals (logprob-shape + free cross-cap resolution ensemble).

Gates:
  RecoverabilityGate  : Delta(x) = P(32B right|x) - P(7B right|x)  -> escalate where escalation has
                        positive expected accuracy gain. The deferral-theoretic value of escalation.
  BenefitGate         : directly predict z = 1[7B wrong & 32B right]; escalate where benefit is high.
  CostAwareGate       : score = benefit(x) / expected_32B_FLOPs(x); prefers cheap, high-yield
                        escalations (mirrors the outcome-oracle's cheapest-beneficial-first logic).
                        expected_32B_FLOPs uses known full-res prefill + a calib-fit decode predictor.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

FEATS = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2",
         "gini", "n_opts", "cap_disagree", "cap_nuniq"]
N7, N32 = 7.6e9, 33.0e9


def _X(sig, feats):
    return np.column_stack([sig[f] for f in feats])


def _clf(kind):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    return HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.06,
                                          l2_regularization=1.0, early_stopping=True, random_state=42)


class RecoverabilityGate:
    """score = P(32B right|x) - P(7B right|x): expected accuracy gain from escalating."""
    def __init__(self, kind="logistic", feats=FEATS):
        self.kind, self.feats = kind, feats

    def fit(self, D):
        assert D.has_strong, "needs 32B-on-PMC-train (ckpts/gate_32b_pmctrain/)"
        Xc = _X(D.calib["sig"], self.feats)
        self.p7 = _clf(self.kind).fit(Xc, D.calib["ok"])
        self.p32 = _clf(self.kind).fit(Xc, D.calib["a32"])
        self.score_calib = self.p32.predict_proba(Xc)[:, 1] - self.p7.predict_proba(Xc)[:, 1]
        return self

    def score_eval(self, pool):
        X = _X(pool["sig"], self.feats)
        return self.p32.predict_proba(X)[:, 1] - self.p7.predict_proba(X)[:, 1]


class BenefitGate:
    """score = P(7B wrong & 32B right | x): probability escalation fixes the answer."""
    def __init__(self, kind="logistic", feats=FEATS):
        self.kind, self.feats = kind, feats

    def fit(self, D):
        assert D.has_strong
        Xc = _X(D.calib["sig"], self.feats)
        z = ((D.calib["ok"] == 0) & (D.calib["a32"] == 1)).astype(float)
        self.clf = _clf(self.kind).fit(Xc, z)
        self.score_calib = self.clf.predict_proba(Xc)[:, 1]
        return self

    def score_eval(self, pool):
        return self.clf.predict_proba(_X(pool["sig"], self.feats))[:, 1]


class CostAwareGate:
    """score = expected accuracy gain / expected 32B FLOPs. Escalates cheap, high-yield cases first,
    which is exactly the structure of the outcome-oracle's compute floor. expected 32B FLOPs uses the
    KNOWN full-res prefill (token_cache) plus a calib-fit decode-length predictor (32B gen_tokens)."""
    def __init__(self, kind="logistic", feats=FEATS):
        self.kind, self.feats = kind, feats

    def fit(self, D):
        assert D.has_strong
        Xc = _X(D.calib["sig"], self.feats)
        self.p7 = _clf(self.kind).fit(Xc, D.calib["ok"])
        self.p32 = _clf(self.kind).fit(Xc, D.calib["a32"])
        # decode-length predictor for the 32B (calib has g32); robust to noise via GBM regressor
        self.gpred = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.06,
                                                   random_state=42).fit(Xc, D.calib["g32"])
        self._gmean = float(D.calib["g32"].mean())
        delta = self.p32.predict_proba(Xc)[:, 1] - self.p7.predict_proba(Xc)[:, 1]
        # calib prefill unknown per-sample here (calib is PMC-train, single benchmark); use mean PF proxy
        self.score_calib = np.maximum(delta, 0) / (self._gmean + 1.0)
        return self

    def score_eval(self, pool):
        X = _X(pool["sig"], self.feats)
        delta = self.p32.predict_proba(X)[:, 1] - self.p7.predict_proba(X)[:, 1]
        ghat = np.clip(self.gpred.predict(X), 1.0, None)
        cost = 2 * N32 * (pool["PF"] + ghat)              # marginal 32B FLOPs if escalated
        return np.maximum(delta, 0) / cost * 1e12          # scale for readability; ranking-invariant


def deferral_registry():
    return {
        "rad:recoverability-logit": RecoverabilityGate("logistic"),
        "rad:recoverability-gbm": RecoverabilityGate("gbm"),
        "rad:benefit-logit": BenefitGate("logistic"),
        "rad:benefit-gbm": BenefitGate("gbm"),
        "rad:costaware-logit": CostAwareGate("logistic"),
        "rad:costaware-gbm": CostAwareGate("gbm"),
    }
