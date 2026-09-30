"""Golden output: Set A features, gate flags and shipped-model scores of the seeded synthetic demo set.

Regenerate `tests/golden/synthetic_demo.csv` only for an intentional change:
    PYTHONPATH=src python tests/test_golden.py --regenerate
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

GOLDEN = Path(__file__).resolve().parent / "golden" / "synthetic_demo.csv"
COLUMNS = ["rest_IAF", "rest_rel_alpha_O", "rest_rel_alpha_IAF", "rest_rel_beta_O", "n_kept", "duration_s",
           "truncated", "min_data_fail", "scorable", "score"]
TOL = 1e-9


def golden_table(demo_dir: Path, model: dict) -> pd.DataFrame:
    from ptsd_eeg import model as M
    from ptsd_eeg import qc
    rows = []
    for s in sorted(p for p in demo_dir.iterdir() if p.is_dir()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = qc.min_data_record(s)
            except (OSError, ValueError, IndexError):
                r = dict(scorable=False)
            score = M.predict(model, s)
        rows.append(dict(subject_id=s.name, **{c: r.get(c, np.nan) for c in COLUMNS[:-1]}, score=score))
    return pd.DataFrame(rows)


def test_golden_synthetic_demo(demo_dir, shipped_model):
    got = golden_table(demo_dir, shipped_model).set_index("subject_id")
    ref = pd.read_csv(GOLDEN, float_precision="round_trip").set_index("subject_id")
    assert list(got.index) == list(ref.index)
    for c in COLUMNS:
        a, b = got[c].astype(float).to_numpy(), ref[c].astype(float).to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b)), c
        ok = ~np.isnan(a)
        assert np.abs(a[ok] - b[ok]).max(initial=0) < TOL, (c, np.abs(a[ok] - b[ok]).max())


if __name__ == "__main__" and "--regenerate" in sys.argv:
    import tempfile
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo / "src"), str(repo / "examples")]
    from make_synthetic_edf import make_demo_set

    from ptsd_eeg.weights import load_weights
    with tempfile.TemporaryDirectory() as d:
        make_demo_set(d, seed=0)
        GOLDEN.parent.mkdir(exist_ok=True)
        golden_table(Path(d), load_weights(repo / "weights")).to_csv(GOLDEN, index=False)
    print(f"wrote {GOLDEN}")
