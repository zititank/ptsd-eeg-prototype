"""EDF reading on generated files: round trip, label-based channel selection, defective headers."""
import numpy as np
import pytest

from edf_writer import write_edf
from ptsd_eeg import io, qc

LABELS = ["Fp1", "Fp2", "T3", "T4", "O1", "O2", "Event"]


def _signals(n=125 * 40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 20, size=(7, n))


def test_round_trip_within_quantisation(tmp_path):
    x = _signals()
    p = write_edf(tmp_path / "rest.edf", x, 125.0, LABELS)
    data, fs, h = io.load_edf_with_header(p)
    assert fs == 125.0 and h.actual_records == 40 and h.header_records == 40
    order = [LABELS.index(c) for c in io.CHANNELS]
    assert data.shape == (6, 125 * 40)
    assert np.abs(data - x[order]).max() <= 0.051  # 0.1 uV quantisation step


def test_missing_channel_raises(tmp_path):
    p = write_edf(tmp_path / "a.edf", _signals()[:5], 125.0, LABELS[:5])
    with pytest.raises(ValueError, match="O2"):
        io.load_edf(p)


def test_truncated_file_uses_complete_records_and_flags(tmp_path):
    p = write_edf(tmp_path / "rest.edf", _signals(), 125.0, LABELS, truncate_records=10)
    with pytest.warns(UserWarning):
        h = io.read_edf_header(p)
    assert h.header_records == 40 and h.actual_records == 30
    f = qc.min_data_flags(h)
    assert f["truncated"] and not f["too_short"] and f["min_data_fail"] and f["duration_s"] == 30.0


def test_unknown_record_count_is_not_truncated(tmp_path):
    p = write_edf(tmp_path / "rest.edf", _signals(), 125.0, LABELS, header_records=-1)
    f = qc.min_data_flags(io.read_edf_header(p))
    assert not f["truncated"] and not f["min_data_fail"] and f["actual_records"] == 40


def test_short_recording_fails_gate(tmp_path):
    p = write_edf(tmp_path / "rest.edf", _signals(125 * 20), 125.0, LABELS)
    f = qc.min_data_flags(io.read_edf_header(p))
    assert f["too_short"] and f["min_data_fail"]


def test_not_an_edf_raises_oserror(tmp_path):
    p = tmp_path / "rest.edf"
    p.write_bytes(b"not an edf" * 10)
    with pytest.raises(OSError):
        io.read_edf_header(p)


@pytest.mark.parametrize("name,rest", [("rest.edf", True), ("S01_REST.EDF", True), ("s01-rest.edf", True),
                                       ("task1.edf", False), ("rest.txt", False)])
def test_rest_file_naming(name, rest):
    assert io.is_rest_name(name) is rest


def test_rest_path_requires_exactly_one(tmp_path):
    (tmp_path / "task1.edf").write_bytes(b"")
    with pytest.raises(ValueError):
        io.rest_path(tmp_path)
    (tmp_path / "rest.edf").write_bytes(b"")
    assert io.rest_path(tmp_path).name == "rest.edf"
    (tmp_path / "S01_REST.edf").write_bytes(b"")
    with pytest.raises(ValueError):
        io.rest_path(tmp_path)


def test_layout_file_overrides_names(tmp_path, monkeypatch):
    import json

    from ptsd_eeg import cohort
    from ptsd_eeg.layout import get_layout
    lay = tmp_path / "layout.json"
    lay.write_text(json.dumps(dict(rest_suffix="_EC", batch_b_start_date="01.01.00", ctrl_c_prefix="X",
                                   group_dirs=dict(control="hc", ptsd="pt", soma="sf"))))
    monkeypatch.setenv("PTSD_LAYOUT", str(lay))
    assert get_layout().group_dirs["control"] == "hc"
    assert io.is_rest_name("S01_ec.edf") and not io.is_rest_name("rest.edf")
    assert cohort.assign_batch("control", "train", "S01", "01.01.00") == "CTRL_B"
    assert cohort.assign_batch("control", "train", "S01", "02.01.00") == "CTRL_other"
    assert cohort.assign_batch("control", "extended", "X12", None) == "CTRL_C"
    assert cohort.assign_batch("control", "extended", "S12", None) == "CTRL_elderly"
