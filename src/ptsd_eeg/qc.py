"""Label-free signal QC and the minimum-data gate of the shipped recipe (C2G).

Two independent mechanisms:

- Coupling QC (used only by the C3 candidate): median |r| over the 15 channel pairs and r(Fp1, Fp2),
  both on 2-40 Hz filtered data. Physiological EEG shows clear inter-channel coupling; near-zero coupling
  flags a non-physiological recording.
- Minimum-data gate (PROTOCOL_v4_AMENDMENT_1, shipped in C2G): a rest recording shorter than 30 s of
  complete data, or truncated relative to its own header, is not scored and receives the prior fallback.

Thresholds were fixed in the data-integrity audit before any model comparison.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .features import OCCIPITAL, extract_array
from .io import CHANNELS, EDFHeader, load_edf_with_header, rest_path

MIN_DURATION_REST_S = 30.0          # seconds of complete EDF records
COUPLING_BAND_HZ = (2.0, 40.0)
QC_MEDIAN_ABS_R_MIN = 0.05
QC_FP1_FP2_R_MIN = 0.30


def coupling_qc(x: np.ndarray, fs: float) -> dict[str, float]:
    """Median |r| over the 15 channel pairs and r(Fp1, Fp2), 2-40 Hz; NaN when < 4 s of data."""
    x = x - np.median(x, axis=1, keepdims=True)
    out = dict(median_abs_r=np.nan, fp1fp2_r=np.nan)
    if x.shape[1] >= int(4 * fs):
        xf = sosfiltfilt(butter(4, COUPLING_BAND_HZ, btype="bandpass", fs=fs, output="sos"), x, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(xf)
        iu = np.triu_indices(6, 1)
        out["median_abs_r"] = float(np.nanmedian(np.abs(c[iu])))
        out["fp1fp2_r"] = float(c[CHANNELS.index("Fp1"), CHANNELS.index("Fp2")])
    return out


def qc_flags(duration_s: float, median_abs_r: float, fp1fp2_r: float) -> dict[str, bool]:
    """Coupling-QC and duration flags. A NaN coupling value never flags."""
    qc_fail = bool((median_abs_r < QC_MEDIAN_ABS_R_MIN) or (fp1fp2_r < QC_FP1_FP2_R_MIN))
    too_short = bool(duration_s < MIN_DURATION_REST_S)
    return dict(qc_fail=qc_fail, too_short=too_short)


def min_data_flags(header: EDFHeader) -> dict:
    """Minimum-data gate flags (PROTOCOL_v4_AMENDMENT_1 §2).

    Duration counts complete records only. `truncated` means the data section is shorter than the
    header's declared record count; a header record count of -1 (unknown) is never truncated.
    """
    duration = float(header.actual_records * header.record_duration_s)
    data_bytes = header.actual_records * header.bytes_per_record + header.trailing_data_bytes
    truncated = bool(header.header_records >= 0 and data_bytes < header.header_records * header.bytes_per_record)
    too_short = bool(duration < MIN_DURATION_REST_S)
    return dict(duration_s=duration, header_records=int(header.header_records),
                actual_records=int(header.actual_records), truncated=truncated, too_short=too_short,
                min_data_fail=bool(too_short or truncated))


def min_data_record(subject_dir: Path | str, occipital: Sequence[str] = OCCIPITAL) -> dict:
    """Set A features plus minimum-data flags from a single read of the rest file (C2G scoring path)."""
    x, fs, header = load_edf_with_header(rest_path(Path(subject_dir)))
    features, _, _ = extract_array(x, fs, occipital=occipital)
    return dict(**features, **min_data_flags(header))


def subject_record(subject_dir: Path | str, occipital: Sequence[str] = OCCIPITAL) -> dict:
    """Features, duration and coupling QC from one read of the rest file (C3 scoring path).

    Raises OSError/ValueError/IndexError if the file is unreadable.
    """
    x, fs, header = load_edf_with_header(rest_path(Path(subject_dir)))
    features, _, _ = extract_array(x, fs, occipital=occipital)
    duration = float(header.actual_records * header.record_duration_s)
    qc = coupling_qc(x, fs)
    return dict(**features, duration_s=duration, **qc, **qc_flags(duration, qc["median_abs_r"], qc["fp1fp2_r"]))
