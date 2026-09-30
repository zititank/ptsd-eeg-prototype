"""Data-gated tests (skipped unless PTSD_DATA points to the confidential data root).

They check the public code against the published aggregates only: no per-subject reference values
are stored in this repository.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import REPO
from ptsd_eeg import gates as G
from ptsd_eeg import model as M
from ptsd_eeg.bootstrap import bootstrap_arrays, bootstrap_index_sets
from ptsd_eeg.features import SET_A
from ptsd_eeg.metrics import batch_table, point_metrics
from ptsd_eeg.reproduce import compare, metrics_row, run_model

pytestmark = pytest.mark.data
TOL = 1e-9


@pytest.fixture(scope="module")
def runs(real_cohort):
    v3 = run_model("v3", real_cohort, n_jobs=4)
    c2 = run_model("C2", real_cohort, n_jobs=4)
    c2g = run_model("C2G", real_cohort, members=c2["members"])
    return dict(v3=v3, C2=c2, C2G=c2g)


def test_cohort_counts(real_cohort, real_layout):
    """Training set as published (90 scorable: 24 PTSD, 46 batch B, 20 other controls; 97 somatoform + 65+
    folders); exact folder counts per batch come from the private layout file when it lists them."""
    c = real_cohort
    tr = c[c.scorable & c.batch.isin(["PTSD", "CTRL_B", "CTRL_other"])].batch.value_counts().to_dict()
    assert tr == {"PTSD": 24, "CTRL_B": 46, "CTRL_other": 20}
    assert int(c.batch.isin(["SOMA", "CTRL_elderly"]).sum()) == 97
    exp = real_layout.expected_counts
    if not exp:
        pytest.skip("no expected_counts in the private layout file")
    assert len(c) == exp["folders"] and int(c.scorable.sum()) == exp["scorable"]
    assert c.batch.value_counts().to_dict() == exp["batches"]
    assert int((c.is_duplicate & (c.batch == "SOMA")).sum()) == exp["soma_duplicates"]


def test_metrics_reproduce_published_aggregates(runs, real_cohort):
    idx = bootstrap_index_sets(real_cohort)
    boots = {k: bootstrap_arrays(r["pred"], real_cohort, idx) for k, r in runs.items()}
    pms = {k: point_metrics(r["pred"]) for k, r in runs.items()}
    rows = pd.DataFrame([metrics_row(k, runs[k]["pred"], real_cohort, idx, boots, pms) for k in runs])
    cmp = compare(rows, pd.read_csv(REPO / "results" / "metrics.csv"))
    assert len(cmp) > 150
    worst = cmp.sort_values("abs_diff").iloc[-1]
    assert cmp.abs_diff.max() < TOL, worst.to_dict()


def test_batch_tables_reproduce(runs):
    ref = pd.read_csv(REPO / "results" / "batch_tables.csv")
    for name in runs:
        got = batch_table(runs[name]["pred"]).set_index("batch")
        exp = ref[ref.model == name].set_index("batch")
        num = [c for c in exp.columns if c not in ("model", "splits")]
        assert np.allclose(got.loc[exp.index, num].to_numpy(float), exp[num].to_numpy(float),
                           rtol=0, atol=TOL, equal_nan=True), name


def test_shipped_weights_equal_retrained_ensemble(runs, real_cohort, shipped_model):
    """External subjects are scored by the mean of all members, so the shipped weights must give the same
    scores as the ensemble retrained here from raw data."""
    c2g = runs["C2G"]["pred"].set_index("subject_id")
    ext = c2g[(c2g.split == "outer_ensemble") & ~c2g.min_data_gated]
    X = M.input_matrix(shipped_model["recipe_spec"],
                       real_cohort.set_index("subject_id").loc[ext.index, SET_A].to_numpy(float))
    shipped = M.score_matrix(shipped_model["members"], X)[0].mean(0)
    assert len(ext) > 100
    assert np.abs(shipped - ext.score.to_numpy()).max() < 1e-12


@pytest.mark.parametrize("name", ["v3", "C2"])
def test_heldout_row_cannot_influence_its_fold(name, real_cohort):
    assert G.infold_check(name, real_cohort, folds=(0, 89))["passed"]


@pytest.mark.slow
def test_v3_label_permutation_inside_recorded_null(real_cohort):
    rec = G.permutation_check("v3", real_cohort, n_jobs=4)
    assert rec["passed"], rec
