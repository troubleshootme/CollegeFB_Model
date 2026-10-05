"""Picklable model wrappers used by train.py and simulate.py."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_ridge(alpha: float = 20.0) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_hgb(**overrides) -> HistGradientBoostingRegressor:
    """Shallow, heavily regularised HGB.

    Walk-forward 2019-2025 margin MAE: depth 6 / 400 iters 13.23, this 12.87.
    Football margins are noisy (~13 pt residual sd), so deep trees just memorise.
    """
    params = dict(
        max_depth=3,
        learning_rate=0.05,
        max_iter=150,
        l2_regularization=0.1,
        min_samples_leaf=50,
        random_state=42,
    )
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


class BlendRegressor(RegressorMixin, BaseEstimator):
    """Weighted average of a ridge and a shallow HGB (walk-forward MAE 12.76 vs 12.79 / 12.87 alone)."""

    def __init__(self, ridge_weight: float = 0.5, ridge_alpha: float = 20.0):
        self.ridge_weight = ridge_weight
        self.ridge_alpha = ridge_alpha

    def fit(self, X, y):
        self.ridge_ = make_ridge(self.ridge_alpha).fit(X, y)
        self.hgb_ = make_hgb().fit(X, y)
        return self

    def predict(self, X):
        w = float(self.ridge_weight)
        return w * self.ridge_.predict(X) + (1.0 - w) * self.hgb_.predict(X)


class MarginWinClassifier:
    """P(home win) as a logistic function of a margin model's output.

    Fitting the calibrator on out-of-sample margin predictions fixed the tail
    over-confidence of the standalone HGB classifier (2025 Brier 0.186 vs 0.199).
    Exposes ``predict_proba`` so callers can treat it like a sklearn classifier.
    """

    def __init__(self, margin_model, calibrator: LogisticRegression):
        self.margin_model = margin_model
        self.calibrator = calibrator

    @classmethod
    def fit_calibrator(cls, oof_margin: np.ndarray, home_win: np.ndarray) -> LogisticRegression:
        mask = np.isfinite(oof_margin)
        return LogisticRegression(C=100.0).fit(np.asarray(oof_margin)[mask].reshape(-1, 1), np.asarray(home_win)[mask])

    def predict_proba(self, X) -> np.ndarray:
        margin = np.asarray(self.margin_model.predict(X), dtype=float).reshape(-1, 1)
        return self.calibrator.predict_proba(margin)
