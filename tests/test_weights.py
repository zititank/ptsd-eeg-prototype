"""The shipped weights: checksums, pickle-free loading, no identifiers, byte-reproducible saving,
member-order invariance and the published coefficient table."""
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from conftest import REPO
from ptsd_eeg import model as M
from ptsd_eeg import weights as W

WD = REPO / "weights"


def test_checksums_match():
    res = W.verify_weights(WD)
    assert set(res) == {"c2g_members.npz", "model_meta.json"} and all(res.values())


def test_npz_has_no_pickled_objects():
    with np.load(WD / "c2g_members.npz", allow_pickle=False) as z:
        assert set(z.files) == {"shrinkage", "coef", "intercept", *[f"prep_{k}" for k in W.PREP_KEYS]}
        assert all(z[k].dtype != object for k in z.files)
    with zipfile.ZipFile(WD / "c2g_members.npz") as zf:
        assert all(n.endswith(".npy") for n in zf.namelist())


def test_metadata_is_aggregate_only():
    meta = json.loads((WD / "model_meta.json").read_text())
    forbidden = {"training_subject_ids", "inner_audit", "subject_id", "subject_ids"}
    assert not forbidden & set(meta)
    assert meta["n_train"] == 90 and meta["n_ptsd"] == 24 and meta["n_negative"] == 66
    assert meta["n_members"] == 90 and meta["n_bags"] == 100 and meta["recipe"] == "C2G"
    assert meta["fallback_score"] == 24 / 90 and meta["threshold"] == 0.5
    assert meta["member_order"].startswith("random permutation")


def test_loaded_model_structure(shipped_model):
    m = shipped_model
    assert len(m["members"]) == 90
    mb = m["members"][0]
    assert mb["coef"].shape == (100, 4) and mb["intercept"].shape == (100,)
    assert all(mb["prep"][k].shape == (100, 4) for k in W.PREP_KEYS)
    assert mb["clip"] == (0.02, 0.98) and mb["gate"] is None
    assert m["recipe_spec"]["min_data_gate"] is True


def test_save_is_byte_reproducible_and_lossless(shipped_model, tmp_path):
    W.save_weights(shipped_model, tmp_path / "a")
    W.save_weights(shipped_model, tmp_path / "b")
    assert W.sha256_file(tmp_path / "a" / "c2g_members.npz") == W.sha256_file(tmp_path / "b" / "c2g_members.npz")
    assert W.sha256_file(tmp_path / "a" / "c2g_members.npz") == W.sha256_file(WD / "c2g_members.npz")
    again = W.load_weights(tmp_path / "a")
    for x, y in zip(shipped_model["members"], again["members"]):
        assert np.array_equal(x["coef"], y["coef"]) and np.array_equal(x["intercept"], y["intercept"])


def test_tampered_weights_are_rejected(tmp_path):
    import shutil
    d = tmp_path / "w"
    shutil.copytree(WD, d)
    meta = json.loads((d / "model_meta.json").read_text())
    meta["threshold"] = 0.1
    (d / "model_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="checksum"):
        W.load_weights(d)


def test_member_order_does_not_change_scores(shipped_model):
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(8, 12, 50), rng.uniform(.1, .7, 50), rng.uniform(.1, .7, 50),
                         rng.uniform(.05, .3, 50)])
    X[::7, 0] = np.nan
    Xi = M.input_matrix(shipped_model["recipe_spec"], X)
    a = M.score_matrix(shipped_model["members"], Xi)[0].mean(0)
    perm = [shipped_model["members"][i] for i in rng.permutation(90)]
    b = M.score_matrix(perm, Xi)[0].mean(0)
    assert np.abs(a - b).max() < 1e-15


def test_published_coefficient_table(shipped_model):
    got = W.coefficient_table(shipped_model)
    ref = pd.read_csv(REPO / "results" / "c2g_coefficients.csv")
    assert list(got.feature) == ["rest_IAF", "clr_alpha_O", "logit_alpha_IAF", "clr_beta_O"]
    num = ref.select_dtypes("number").columns
    assert np.allclose(got[num].to_numpy(float), ref[num].to_numpy(float), rtol=0, atol=1e-12)
