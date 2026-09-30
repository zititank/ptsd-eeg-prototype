"""Metric helpers against reference implementations."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ptsd_eeg.metrics import auc_rank, ci, proxy
from ptsd_eeg.transforms import c1_subject_transform


def test_auc_rank_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 60)
        s = np.round(rng.random(60), 1)
        assert auc_rank(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_proxy_points():
    assert proxy(0.5, 0.0) == 20
    assert proxy(1.0, 0.0) == 50
    assert proxy(0.3, 1.0) == 0
    assert proxy(0.8162878787878789, 0.1340206185567010) == pytest.approx(30 * 0.6325757575757578 + 20 * 0.865979381443299)


def test_percentile_interval():
    lo, hi = ci(np.arange(1001))
    assert (lo, hi) == (25.0, 975.0)


def test_clr_transform_is_centred():
    X = np.array([[10.0, 0.5, 0.4, 0.2], [np.nan, 0.3, 0.2, 0.3]])
    Z = c1_subject_transform(X)
    a, b = X[:, 1], X[:, 3]
    r = 1 - a - b
    g = (np.log(a) + np.log(b) + np.log(r)) / 3
    assert np.allclose(Z[:, 1], np.log(a) - g) and np.allclose(Z[:, 3], np.log(b) - g)
    assert np.allclose(Z[:, 2], np.log(X[:, 2] / (1 - X[:, 2])))
    assert np.isnan(Z[1, 0])
