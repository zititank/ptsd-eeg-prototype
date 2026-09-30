"""Feature transforms applied before the linear discriminant.

- v3 chain:  median imputation -> standardisation (mean / SD).
- C1 chain (used by the shipped model): a stateless per-subject compositional transform, then median
  imputation -> winsorisation at the training 1st/99th percentiles -> robust scaling (median / IQR).

Why compositional: the three relative powers (alpha, beta, rest of 1-40 Hz) sum to one, so they live on
a simplex. The centred log-ratio (CLR) removes that constraint; the IAF-band share is a single
proportion and gets a logit.

Fitting uses sklearn objects; `compile_prep` turns a fitted chain into plain arrays so that the saved
weights are numpy arrays and scoring never depends on sklearn internals.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

C1_FEATURES: list[str] = ["rest_IAF", "clr_alpha_O", "logit_alpha_IAF", "clr_beta_O"]
FLOOR = 1e-6                    # lower bound on proportions before log / logit
WINSOR_PCT = (1.0, 99.0)


def c1_subject_transform(X: np.ndarray) -> np.ndarray:
    """Set A [IAF, rel_alpha_O, rel_alpha_IAF, rel_beta_O] -> [IAF, clr_alpha_O, logit_alpha_IAF, clr_beta_O].

    With a = relative alpha, b = relative beta and r = 1 - a - b (the rest of 1-40 Hz), each floored at
    1e-6: clr_alpha = ln a - g, clr_beta = ln b - g, where g = (ln a + ln b + ln r) / 3.
    Stateless and per subject; NaN stays NaN (imputed later, inside the training fold).
    """
    X = np.asarray(X, dtype=float)
    iaf, a, p, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    r = np.maximum(1.0 - a - b, FLOOR)
    a, b = np.maximum(a, FLOOR), np.maximum(b, FLOOR)
    la, lb, lr = np.log(a), np.log(b), np.log(r)
    g = (la + lb + lr) / 3.0
    p = np.clip(p, FLOOR, 1.0 - FLOOR)
    return np.column_stack([iaf, la - g, np.log(p / (1.0 - p)), lb - g])


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip every feature to the training 1st/99th percentiles (fitted in-fold)."""

    def __init__(self, pct: tuple[float, float] = WINSOR_PCT):
        self.pct = pct

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.percentile(X, self.pct, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_prep(kind: str) -> Pipeline:
    """Unfitted preprocessing before LDA. 'v3': impute -> standardise. 'c1': impute -> winsorise -> robust scale."""
    impute = ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    if kind == "v3":
        return Pipeline([impute, ("scale", StandardScaler())])
    if kind == "c1":
        return Pipeline([impute, ("winsor", Winsorizer()), ("scale", RobustScaler())])
    raise ValueError(f"unknown preprocessing {kind!r}")


def compile_prep(prep: Pipeline) -> dict[str, np.ndarray]:
    """Plain-array form of a fitted `make_prep` chain: fill, lo, hi, center, scale (one value per feature)."""
    steps = dict(prep.steps)
    n = steps["impute"].statistics_.shape[0]
    out = dict(fill=np.asarray(steps["impute"].statistics_, float).copy(),
               lo=np.full(n, -np.inf), hi=np.full(n, np.inf))
    if "winsor" in steps:
        out["lo"], out["hi"] = steps["winsor"].lo_.copy(), steps["winsor"].hi_.copy()
    sc = steps["scale"]
    if isinstance(sc, StandardScaler):
        out["center"], out["scale"] = sc.mean_.copy(), sc.scale_.copy()
    else:
        out["center"], out["scale"] = sc.center_.copy(), sc.scale_.copy()
    return out


def apply_prep(p: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    """Apply a compiled chain: fill NaN, clip to [lo, hi], then (X - center) / scale."""
    X = np.asarray(X, dtype=float)
    X = np.where(np.isnan(X), p["fill"], X)
    X = np.clip(X, p["lo"], p["hi"])
    return (X - p["center"]) / p["scale"]
