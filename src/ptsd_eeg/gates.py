"""Leakage gates and the pre-registered decision rule of PROTOCOL_v4.

- G1a `infold_check`: perturbing the held-out row (features and label) must leave its fold model
  bit-identical, i.e. nothing learnt in fold i ever saw subject i.
- G1b `permutation_check`: one predeclared label permutation through the full nested procedure must
  give an AUC inside the recorded 95% null range of the v3 pipeline.
- `criteria` / `decide` (PROTOCOL_v4 §4): a candidate replaces v3 only if its proxy points beat v3 with
  a paired bootstrap 2.5th percentile above zero (criterion 1), its flip rate is at most v3's + 0.005
  (criterion 2), and it passes the safety gates (criterion 3). Among passing candidates the highest proxy
  estimate wins; exact ties go to the lower C-number.

The delivery gates of the hackathon submission (inference-contract stress test, container substitute) are not
part of this repository.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import model as M
from .bootstrap import EVAL_SEED
from .cohort import PRIMARY, training_frame
from .features import SET_A

FLIP_TOL = 0.005
PERMUTATION_NULL_95 = (0.14239267676767678, 0.5795770202020203)
"""95% range of 499 label permutations of the v3 LOSO pipeline (observed AUC 0.782, p = 0.002)."""


def arrays_equal(a, b) -> bool:
    """Deep, exact equality of nested dicts/lists/arrays (NaN equals NaN)."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(arrays_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(arrays_equal(x, z) for x, z in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
    return a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b))


def infold_check(name: str, cohort: pd.DataFrame, folds=(0, 30, 89)) -> dict:
    """G1a: the held-out row (features and label) cannot influence its own fold model."""
    recipe = M.RECIPES[name]
    tr = training_frame(cohort, recipe["soma"])
    X = M.input_matrix(recipe, tr[SET_A].to_numpy(float))
    y, st = tr.y.to_numpy(), tr.batch.to_numpy().astype(str)
    out = {}
    for i in folds:
        a = M.loso_fold(X, y, st, i, recipe)
        X2, y2 = X.copy(), y.copy()
        X2[i] = 1e6
        y2[i] = 1 - y2[i]
        b = M.loso_fold(X2, y2, st, i, recipe)
        same = arrays_equal(a, b)
        keep = np.arange(len(y)) != i
        med = np.nanmedian(X[keep], axis=0)
        fill_ok = bool(np.allclose(a["prep"]["fill"][0], med)) if not recipe["bags"] else True
        out[int(i)] = dict(identical_after_perturbing_heldout=same, imputer_equals_train_fold_median=fill_ok)
    return dict(passed=all(v["identical_after_perturbing_heldout"] and v["imputer_equals_train_fold_median"]
                           for v in out.values()), folds=out)


def permutation_check(name: str, cohort: pd.DataFrame, n_jobs: int = 4) -> dict:
    """G1b: one predeclared label permutation (seed 20260928) through the full nested procedure."""
    from .reproduce import run_model
    lo, hi = PERMUTATION_NULL_95
    tr = training_frame(cohort, M.RECIPES[name]["soma"])
    yp = np.random.default_rng(EVAL_SEED).permutation(tr.y.to_numpy())
    res = run_model(name, cohort, n_jobs=n_jobs, y_override=yp)
    oof = res["pred"].set_index("subject_id").loc[tr.subject_id].score.to_numpy()
    rows = tr.batch.isin(PRIMARY).to_numpy()
    auc = float(roc_auc_score(yp[rows], oof[rows]))
    return dict(passed=bool(lo <= auc <= hi), permuted_auc=auc, null_95=[lo, hi], seed=EVAL_SEED)


def criteria(metrics: pd.DataFrame) -> pd.DataFrame:
    """Criteria 1 and 2 for every candidate row of `metrics` (needs `delta_proxy_lo` and `flip_rate`)."""
    v3 = metrics.set_index("model").loc["v3"]
    rows = []
    for r in metrics.itertuples():
        if r.model == "v3":
            continue
        c1 = bool(r.delta_proxy > 0 and r.delta_proxy_lo > 0)
        c2 = bool(r.flip_rate <= v3.flip_rate + FLIP_TOL)
        rows.append(dict(model=r.model, proxy_points=r.proxy_points, v3_proxy_points=v3.proxy_points,
                         delta_proxy=r.delta_proxy, delta_proxy_p2_5=r.delta_proxy_lo, delta_proxy_p97_5=r.delta_proxy_hi,
                         criterion1=c1, flip_rate=r.flip_rate, v3_flip_rate=v3.flip_rate,
                         flip_limit=v3.flip_rate + FLIP_TOL, criterion2=c2, needs_gates=c1 and c2))
    return pd.DataFrame(rows)


def decide(crit: pd.DataFrame, gates_passed: dict[str, bool], metrics: pd.DataFrame) -> tuple[str, list[str]]:
    """PROTOCOL_v4 §4: candidates passing all three criteria; highest proxy estimate; ties -> lower C-number."""
    passing = [r.model for r in crit.itertuples()
               if r.criterion1 and r.criterion2 and gates_passed.get(r.model, False)]
    if not passing:
        return "v3", passing
    m = metrics.set_index("model")
    best = max(passing, key=lambda c: (m.loc[c, "proxy_points"], -int(c[1:])))
    return best, passing
