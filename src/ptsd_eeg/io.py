"""EDF reading and dataset file naming.

A minimal reader for the standard 16-bit EDF/EDF+ files written by a 6-channel consumer EEG headband. It
depends only on numpy, so the pipeline has no EDF library dependency.

- Channels are selected by label (case and whitespace insensitive), never by position.
- The sampling rate is read from every file (the dataset is not uniform).
- `Number of data records = -1` and truncated files are handled from the file size: only complete data
  records are used, and a warning is emitted when the header disagrees with the file.
- Files are opened read-only and never modified.

Dataset layout: one folder per subject, containing short task recordings and exactly one resting-state
recording (`rest.edf` in the generic layout). Folder and file names come from `ptsd_eeg.layout`.
"""
from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .layout import get_layout

CHANNELS: list[str] = ["O1", "T3", "Fp1", "Fp2", "T4", "O2"]
"""The six EEG channels of the headband, in the order used everywhere downstream."""


@dataclass(frozen=True)
class EDFHeader:
    """Parsed EDF header plus the record count actually present in the file.

    `header_records` is the value written in the header (-1 = unknown); `actual_records` is the number
    of complete data records inferred from the file size, which is what the reader uses.
    """

    path: Path
    header_bytes: int
    n_signals: int
    record_duration_s: float
    header_records: int
    actual_records: int
    trailing_data_bytes: int
    labels: tuple[str, ...]
    physical_dimensions: tuple[str, ...]
    physical_mins: np.ndarray
    physical_maxs: np.ndarray
    digital_mins: np.ndarray
    digital_maxs: np.ndarray
    samples_per_record: np.ndarray
    bytes_per_record: int


def _decode_ascii(field: bytes) -> str:
    return field.decode("latin-1", errors="replace").strip()


def _parse_int(field: bytes, name: str) -> int:
    s = _decode_ascii(field)
    if not s:
        raise OSError(f"Missing EDF integer field: {name}")
    try:
        return int(s)
    except ValueError as exc:
        raise OSError(f"Invalid EDF integer field {name}: {s!r}") from exc


def _parse_float(field: bytes, name: str) -> float:
    s = _decode_ascii(field)
    if not s:
        raise OSError(f"Missing EDF float field: {name}")
    try:
        return float(s)
    except ValueError as exc:
        raise OSError(f"Invalid EDF float field {name}: {s!r}") from exc


def _read_signal_field(header: bytes, offset: int, width: int, n: int) -> tuple[list[str], int]:
    out = []
    for i in range(n):
        out.append(_decode_ascii(header[offset + i * width: offset + (i + 1) * width]))
    return out, offset + width * n


def read_edf_header(path: Path | str) -> EDFHeader:
    """Parse the EDF header of `path` and count the complete data records present in the file.

    Raises OSError for files that cannot be an EDF (too short, malformed fields, no complete record).
    """
    path = Path(path)
    size = path.stat().st_size
    if size < 256:
        raise OSError(f"{path}: file is shorter than the EDF fixed header")

    with path.open("rb") as f:
        fixed = f.read(256)
        version = _decode_ascii(fixed[0:8])
        if version not in {"0", "0       ".strip()}:
            if not version.startswith("0"):
                warnings.warn(f"{path}: unusual EDF version field {version!r}")

        header_bytes = _parse_int(fixed[184:192], "header_bytes")
        header_records = _parse_int(fixed[236:244], "number_of_datarecords")
        record_duration = _parse_float(fixed[244:252], "record_duration")
        n_signals = _parse_int(fixed[252:256], "number_of_signals")

        if n_signals <= 0:
            raise OSError(f"{path}: invalid number of signals: {n_signals}")
        expected_header = 256 + 256 * n_signals
        if header_bytes < expected_header:
            raise OSError(
                f"{path}: header says {header_bytes} bytes, but {n_signals} signals require at least {expected_header}"
            )
        if size < header_bytes:
            raise OSError(f"{path}: truncated EDF header")
        if record_duration <= 0:
            raise OSError(f"{path}: invalid record duration: {record_duration}")

        f.seek(0)
        header = f.read(header_bytes)

    off = 256
    labels, off = _read_signal_field(header, off, 16, n_signals)
    _, off = _read_signal_field(header, off, 80, n_signals)  # transducer
    dimensions, off = _read_signal_field(header, off, 8, n_signals)
    phys_min_s, off = _read_signal_field(header, off, 8, n_signals)
    phys_max_s, off = _read_signal_field(header, off, 8, n_signals)
    dig_min_s, off = _read_signal_field(header, off, 8, n_signals)
    dig_max_s, off = _read_signal_field(header, off, 8, n_signals)
    _, off = _read_signal_field(header, off, 80, n_signals)  # prefiltering
    spr_s, off = _read_signal_field(header, off, 8, n_signals)
    _, off = _read_signal_field(header, off, 32, n_signals)  # reserved

    try:
        phys_min = np.asarray([float(x) for x in phys_min_s], dtype=float)
        phys_max = np.asarray([float(x) for x in phys_max_s], dtype=float)
        dig_min = np.asarray([float(x) for x in dig_min_s], dtype=float)
        dig_max = np.asarray([float(x) for x in dig_max_s], dtype=float)
        samples_per_record = np.asarray([int(x) for x in spr_s], dtype=int)
    except ValueError as exc:
        raise OSError(f"{path}: invalid per-signal EDF header field") from exc

    if np.any(samples_per_record < 0):
        raise OSError(f"{path}: negative samples-per-record value")

    bytes_per_record = int(2 * samples_per_record.sum())
    if bytes_per_record <= 0:
        raise OSError(f"{path}: invalid EDF record size")

    data_bytes = size - header_bytes
    complete_records, trailing = divmod(data_bytes, bytes_per_record)
    if complete_records <= 0:
        raise OSError(f"{path}: no complete EDF data records")

    if trailing:
        warnings.warn(
            f"{path}: {trailing} trailing/incomplete data bytes ignored; using {complete_records} complete records"
        )

    if header_records >= 0 and header_records != complete_records:
        warnings.warn(
            f"{path}: header says {header_records} records, file contains {complete_records}; "
            "using complete records inferred from file size"
        )

    return EDFHeader(
        path=path,
        header_bytes=header_bytes,
        n_signals=n_signals,
        record_duration_s=float(record_duration),
        header_records=header_records,
        actual_records=int(complete_records),
        trailing_data_bytes=int(trailing),
        labels=tuple(labels),
        physical_dimensions=tuple(dimensions),
        physical_mins=phys_min,
        physical_maxs=phys_max,
        digital_mins=dig_min,
        digital_maxs=dig_max,
        samples_per_record=samples_per_record,
        bytes_per_record=bytes_per_record,
    )


def _normalized_label(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def find_channel_indices(header: EDFHeader, wanted: Sequence[str] = CHANNELS) -> list[int]:
    """Signal indices of the `wanted` channel labels; raises ValueError if one is missing."""
    mapping = {_normalized_label(label): i for i, label in enumerate(header.labels)}
    indices = []
    for name in wanted:
        key = _normalized_label(name)
        if key not in mapping:
            raise ValueError(
                f"{header.path}: channel {name!r} not found. Available: {list(header.labels)}"
            )
        indices.append(mapping[key])
    return indices


def load_edf_with_header(path: Path | str) -> tuple[np.ndarray, float, EDFHeader]:
    """The six EEG channels in EDF physical units (µV for this device), shape (6, n_samples).

    Returns `(data, sampling_rate_hz, header)`. The six channels must share one sampling rate.
    """
    header = read_edf_header(path)
    indices = find_channel_indices(header)

    eeg_spr = header.samples_per_record[indices]
    sfreqs = eeg_spr.astype(float) / header.record_duration_s
    if not np.allclose(sfreqs, sfreqs[0], rtol=0, atol=1e-9):
        raise ValueError(
            f"{header.path}: required EEG channels have different sampling frequencies: {sfreqs.tolist()}"
        )
    sfreq = float(sfreqs[0])
    if sfreq <= 0:
        raise ValueError(f"{header.path}: invalid EEG sampling rate {sfreq}")

    total_int16 = header.actual_records * (header.bytes_per_record // 2)
    with header.path.open("rb") as f:
        f.seek(header.header_bytes)
        raw = np.fromfile(f, dtype="<i2", count=total_int16)

    if raw.size != total_int16:
        raise OSError(
            f"{header.path}: expected {total_int16} int16 samples in complete EDF records, got {raw.size}"
        )

    signal_offsets = np.concatenate(([0], np.cumsum(header.samples_per_record)))
    records = raw.reshape(header.actual_records, -1)
    channels = []
    for idx in indices:
        a = int(signal_offsets[idx])
        b = int(signal_offsets[idx + 1])
        digital = records[:, a:b].reshape(-1).astype(np.float64)
        dmin = header.digital_mins[idx]
        dmax = header.digital_maxs[idx]
        pmin = header.physical_mins[idx]
        pmax = header.physical_maxs[idx]
        if not all(np.isfinite(v) for v in (dmin, dmax, pmin, pmax)) or dmax == dmin:
            raise OSError(f"{header.path}: invalid physical/digital calibration for channel {header.labels[idx]!r}")
        channels.append((digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin)

    lengths = {len(x) for x in channels}
    if len(lengths) != 1:
        raise ValueError(f"{header.path}: required EEG channels produced unequal lengths: {sorted(lengths)}")

    dims = [header.physical_dimensions[i].strip().lower() for i in indices]
    if len(set(dims)) > 1:
        warnings.warn(f"{header.path}: EEG channels have different physical dimensions: {dims}")

    return np.vstack(channels), sfreq, header


def load_edf(path: Path | str) -> tuple[np.ndarray, float]:
    """`(data, sampling_rate_hz)` of the six EEG channels; see `load_edf_with_header`."""
    x, fs, _ = load_edf_with_header(path)
    return x, fs


def edf_duration(path: Path | str) -> float:
    """Recording duration in seconds, based on complete EDF records."""
    h = read_edf_header(path)
    return float(h.actual_records * h.record_duration_s)


def is_rest_name(name: str) -> bool:
    """True for a resting-state file name: an `.edf` (any case) whose upper-cased stem ends with the layout's
    `rest_suffix` (`rest.edf`, `S01_REST.EDF`, ... in the generic layout)."""
    return (Path(name).suffix.lower() == ".edf"
            and Path(name).stem.upper().endswith(get_layout().rest_suffix.upper()))


def rest_path(subject_dir: Path | str) -> Path:
    """The single resting-state EDF of a subject folder; raises ValueError if there is not exactly one."""
    paths = [p for p in Path(subject_dir).glob("*") if is_rest_name(p.name)]
    if len(paths) != 1:
        raise ValueError(f"Expected one rest EDF, found {len(paths)} in {subject_dir}")
    return paths[0]


def header_start_date(path: Path | str) -> str:
    """EDF start-date field (`dd.mm.yy`).

    Used only to assign the recording batch (training bag strata and reporting), never as a model feature.
    """
    with Path(path).open("rb") as f:
        return _decode_ascii(f.read(176)[168:176])


def subject_dirs(group_dir: Path | str) -> list[Path]:
    """Subject folders of one group folder, sorted by name (this order defines the LOSO fold order)."""
    return sorted(s for s in Path(group_dir).iterdir() if s.is_dir())
