"""Inference contract of `ptsd-eeg predict` on synthetic folders: one row per folder, scores in [0, 1],
the prior fallback for anything unscorable or gated, and no crash on garbage input."""
import numpy as np
import pandas as pd
import pytest

from ptsd_eeg import cli
from ptsd_eeg import model as M

FALLBACK = 24 / 90
DEFECTIVE = ["SYN-E01-short", "SYN-E02-truncated", "SYN-E03-flat", "SYN-E04-no-rest"]


@pytest.fixture(scope="module")
def demo_predictions(demo_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("pred") / "pred.csv"
    assert cli.main(["predict", str(demo_dir), str(out)]) == 0
    return pd.read_csv(out, float_precision="round_trip")


def test_output_shape_and_range(demo_predictions, demo_dir):
    assert list(demo_predictions.columns) == ["subject_id", "ptsd_probability"]
    assert list(demo_predictions.subject_id) == sorted(p.name for p in demo_dir.iterdir() if p.is_dir())
    assert demo_predictions.ptsd_probability.between(0, 1).all()


def test_defective_folders_get_fallback(demo_predictions):
    p = demo_predictions.set_index("subject_id").ptsd_probability
    for sid in DEFECTIVE:
        assert p[sid] == FALLBACK, sid


def test_unknown_record_count_is_scored(demo_predictions):
    p = demo_predictions.set_index("subject_id").ptsd_probability
    assert p["SYN-E05-unknown-records"] != FALLBACK


def test_scores_inside_clip_range(demo_predictions):
    scored = demo_predictions[~demo_predictions.subject_id.isin(DEFECTIVE)].ptsd_probability
    assert scored.between(0.02, 0.98).all()


def test_garbage_folders_never_crash(tmp_path, shipped_model):
    (tmp_path / "empty").mkdir()
    bad = tmp_path / "random_bytes"
    bad.mkdir()
    (bad / "rest.edf").write_bytes(np.random.default_rng(0).bytes(5000))
    two = tmp_path / "two_rest_files"
    two.mkdir()
    (two / "rest.edf").write_bytes(b"")
    (two / "S01_REST.edf").write_bytes(b"")
    for d in tmp_path.iterdir():
        with pytest.warns(UserWarning):
            assert M.predict(shipped_model, d) == FALLBACK
    out = tmp_path / "out.csv"
    assert cli.main(["predict", str(tmp_path), str(out)]) == 0
    assert len(pd.read_csv(out)) == 3


def test_verify_weights_command():
    assert cli.main(["verify-weights"]) == 0
