"""Determinism: identical fits on repeated runs, serial equals parallel, the synthetic generator is seeded."""
import numpy as np
import pytest

from make_synthetic_edf import make_signals
from ptsd_eeg import gates as G
from ptsd_eeg import model as M
from ptsd_eeg.cohort import training_frame
from ptsd_eeg.features import SET_A


def _xy(cohort, name):
    recipe = M.RECIPES[name]
    tr = training_frame(cohort, recipe["soma"])
    return recipe, M.input_matrix(recipe, tr[SET_A].to_numpy(float)), tr.y.to_numpy(), tr.batch.to_numpy().astype(str)


@pytest.mark.parametrize("name", ["v3", "C1", "C2", "C3", "C4"])
def test_fit_member_twice_identical(name, synthetic_cohort):
    recipe, X, y, st = _xy(synthetic_cohort, name)
    assert G.arrays_equal(M.loso_fold(X, y, st, 3, recipe), M.loso_fold(X, y, st, 3, recipe))


def test_bootstrap_serial_equals_parallel(synthetic_cohort):
    recipe, X, y, st = _xy(synthetic_cohort, "C2")
    sc = synthetic_cohort[synthetic_cohort.scorable]
    Xs = M.input_matrix(recipe, sc[SET_A].to_numpy(float))
    a = M.bootstrap_refits(X, y, st, recipe, Xs, n_boot=3, n_jobs=1)
    b = M.bootstrap_refits(X, y, st, recipe, Xs, n_boot=3, n_jobs=2)
    assert np.array_equal(a[0], b[0])


def test_generator_is_seeded():
    a = make_signals(np.random.default_rng(5), "high_alpha", 10)
    b = make_signals(np.random.default_rng(5), "high_alpha", 10)
    assert np.array_equal(a, b)


def test_predict_twice_identical(demo_dir, shipped_model):
    s = demo_dir / "SYN-H01"
    assert M.predict(shipped_model, s) == M.predict(shipped_model, s)
