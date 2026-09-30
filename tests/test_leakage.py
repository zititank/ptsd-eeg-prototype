"""Leakage and validation logic on a synthetic cohort: every learnt object is fitted in-fold, the
training set has the right composition, batches follow the documented rules, bags are balanced."""
import numpy as np
import pytest

from ptsd_eeg import gates as G
from ptsd_eeg import model as M
from ptsd_eeg.cohort import PRIMARY, training_frame
from ptsd_eeg.features import SET_A


def test_batches_follow_rules(synthetic_cohort):
    c = synthetic_cohort.set_index("subject_id")
    assert (c.loc[c.index.str.startswith("SYN-P"), "batch"] == "PTSD").all()
    assert (c.loc[c.index.str.startswith("SYN-D"), "batch"] == "CTRL_B").all()
    assert (c.loc[c.index.str.startswith("SYN-C"), "batch"] == "CTRL_other").all()
    assert (c.loc[c.index.str.startswith("SYN-S"), "batch"] == "SOMA").all()
    assert (c.loc[c.index.str.startswith("SYN-O"), "batch"] == "CTRL_elderly").all()
    assert (c.loc[c.index.str.contains("-SYN"), "batch"] == "CTRL_C").all()


def test_cohort_order_is_canonical(synthetic_cohort):
    order = synthetic_cohort.batch.map({"CTRL_other": 0, "CTRL_B": 0, "PTSD": 1, "SOMA": 2,
                                        "CTRL_C": 3, "CTRL_elderly": 3})
    assert order.is_monotonic_increasing


@pytest.mark.parametrize("name", ["v3", "C4"])
def test_training_set_composition(name, synthetic_cohort):
    tr = training_frame(synthetic_cohort, M.RECIPES[name]["soma"])
    allowed = set(PRIMARY) | ({"SOMA"} if M.RECIPES[name]["soma"] else set())
    assert set(tr.batch) == allowed and tr.scorable.all()
    assert not tr.batch.isin(["CTRL_C", "CTRL_elderly"]).any()
    assert list(tr.y) == sorted(tr.y) or name == "C4"


def test_short_ptsd_folder_is_gated_not_dropped(synthetic_cohort):
    row = synthetic_cohort.set_index("subject_id").loc["SYN-P07"]
    assert row.min_data_fail and row.too_short


@pytest.mark.parametrize("name", ["v3", "C2"])
def test_heldout_row_cannot_influence_its_fold(name, synthetic_cohort):
    n = len(training_frame(synthetic_cohort, False))
    rec = G.infold_check(name, synthetic_cohort, folds=(0, n - 1))
    assert rec["passed"], rec


def test_gate_threshold_fitted_on_training_rows_only(synthetic_cohort):
    tr = training_frame(synthetic_cohort, soma=False)
    X = M.input_matrix(M.RECIPES["C3"], tr[SET_A].to_numpy(float))
    keep = np.arange(len(tr)) != 2
    g = M.fit_gate(X[keep])
    assert g["threshold"] == pytest.approx(np.percentile(M.mahalanobis2(g, X[keep]), 99), abs=0)


def test_bags_balance_negative_strata():
    y = np.r_[np.ones(10), np.zeros(40)].astype(int)
    strata = np.r_[["PTSD"] * 10, ["A"] * 35, ["B"] * 5]
    bags = M.bag_indices(y, strata, 400, seed=0)
    neg = np.concatenate([b[y[b] == 0] for b in bags])
    share_b = np.mean(strata[neg] == "B")
    assert abs(share_b - 0.5) < 0.02  # each negative stratum gets equal total probability
    assert all((y[b] == 1).sum() == 10 for b in bags)


def test_decision_rule():
    import pandas as pd
    metrics = pd.DataFrame(dict(model=["v3", "C1", "C2"], proxy_points=[30.0, 34.0, 36.0],
                                delta_proxy=[0, 4.0, 6.0], delta_proxy_lo=[0, 0.1, 1.8], delta_proxy_hi=[0, 8, 10],
                                flip_rate=[0.106, 0.1064, 0.089]))
    crit = G.criteria(metrics)
    assert crit.criterion1.all() and crit.criterion2.all()
    assert G.decide(crit, {"C1": True, "C2": True}, metrics) == ("C2", ["C1", "C2"])
    assert G.decide(crit, {"C1": True, "C2": False}, metrics) == ("C1", ["C1"])
    assert G.decide(crit, {}, metrics)[0] == "v3"
