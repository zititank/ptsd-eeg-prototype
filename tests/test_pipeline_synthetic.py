"""End-to-end training and reproduction on the synthetic data root (no confidential data)."""
import json

import numpy as np
import pandas as pd
import pytest

from ptsd_eeg import cli
from ptsd_eeg import model as M
from ptsd_eeg.reproduce import reproduce
from ptsd_eeg.weights import load_weights


@pytest.fixture(scope="module")
def trained(synthetic_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("trained")
    assert cli.main(["train", "--data", str(synthetic_root / "train"), "--recipe", "C1",
                     "--out", str(out)]) == 0
    return out


def test_train_writes_loadable_weights(trained):
    m = load_weights(trained)
    meta = json.loads((trained / "model_meta.json").read_text())
    assert m["recipe"] == "C1" and meta["member_order"] == "outer LOSO fold order"
    assert meta["n_train"] == len(m["members"]) == meta["n_ptsd"] + meta["n_negative"]


def test_training_oof_equals_heldout_member(trained, synthetic_root):
    m = load_weights(trained)
    oof = pd.read_csv(trained / "training_oof.csv")
    train = M.training_records(synthetic_root / "train")
    train = train[train.scorable.astype(bool)].reset_index(drop=True)
    X = M.input_matrix(m["recipe_spec"], train[M.SET_A].to_numpy(float))
    heldout = np.array([M.member_proba(mb, X[[i]])[0] for i, mb in enumerate(m["members"])])
    assert np.abs(heldout - oof.score.to_numpy()).max() < 1e-12


def test_reproduce_runs_on_synthetic_root(synthetic_root, tmp_path):
    s = reproduce(synthetic_root, tmp_path, models=["v3", "C2", "C2G"], n_jobs=1,
                  reference=tmp_path / "no_reference.csv", log=lambda *_: None)
    assert s["models"] == ["v3", "C2", "C2G"] and "max_abs_diff" not in s
    met = pd.read_csv(tmp_path / "metrics_reproduced.csv").set_index("model")
    assert met.auc.between(0, 1).all()
    assert met.loc["C2G", "n_min_data_gated"] >= 1
    assert (met.loc["C2G", "sensitivity_with_fallback"] <= met.loc["C2", "sensitivity_with_fallback"])
    bt = pd.read_csv(tmp_path / "batch_tables.csv")
    assert set(bt.batch) == {"PTSD", "CTRL_B", "CTRL_other", "SOMA", "CTRL_C", "CTRL_elderly"}
    assert not (tmp_path / "per_subject_DO_NOT_COMMIT").exists()
