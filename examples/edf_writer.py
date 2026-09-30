"""Minimal 16-bit EDF writer (numpy only), for synthetic test data.

Writes plain EDF (not EDF+) with one data record per `record_duration_s` seconds. Signals are given in
physical units (µV) and quantised to int16 with a 0.1 µV step over +-3276.8 µV, like the headband's
export. Two options exist only to produce deliberately defective files for the minimum-data gate:

- `header_records`: the record count written in the header (default: the true count; -1 = unknown).
- `truncate_records`: drop this many complete records from the end of the data section after writing
  the header, which simulates an interrupted recording.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

PHYS_MIN, PHYS_MAX = -3276.8, 3276.7
DIG_MIN, DIG_MAX = -32768, 32767
SYNTHETIC_DATE = "01.01.00"
"""Placeholder start date (dd.mm.yy) written into synthetic headers."""


def _field(value, width: int) -> bytes:
    s = str(value)
    if len(s) > width:
        raise ValueError(f"EDF header field too long: {s!r} (max {width})")
    return s.ljust(width).encode("ascii")


def _num(x: float, width: int = 8) -> str:
    """Shortest decimal representation of x that fits an 8-character EDF field."""
    for fmt in ("{:.1f}", "{:.4g}", "{:.3g}"):
        s = fmt.format(x)
        if len(s) <= width:
            return s
    raise ValueError(f"cannot fit {x} into {width} characters")


def write_edf(path: Path | str, signals: np.ndarray, fs: float, labels: Sequence[str],
              start_date: str = SYNTHETIC_DATE, start_time: str = "10.00.00", record_duration_s: float = 1.0,
              patient: str = "X X X X", recording: str = "Startdate X X X synthetic",
              header_records: int | None = None, truncate_records: int = 0) -> Path:
    """Write `signals` (n_channels, n_samples) in µV as an EDF file; returns the path.

    The number of samples is cut to a whole number of records. `fs * record_duration_s` must be an integer.
    """
    path = Path(path)
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    n_ch = signals.shape[0]
    if len(labels) != n_ch:
        raise ValueError("one label per channel is required")
    spr = fs * record_duration_s
    if abs(spr - round(spr)) > 1e-9:
        raise ValueError("fs * record_duration_s must be an integer")
    spr = int(round(spr))
    n_rec = signals.shape[1] // spr
    if n_rec < 1:
        raise ValueError("signal shorter than one data record")
    x = signals[:, :n_rec * spr]
    scale = (DIG_MAX - DIG_MIN) / (PHYS_MAX - PHYS_MIN)
    dig = np.clip(np.round((x - PHYS_MIN) * scale + DIG_MIN), DIG_MIN, DIG_MAX).astype("<i2")

    hdr_bytes = 256 + 256 * n_ch
    declared = n_rec if header_records is None else int(header_records)
    head = b"".join([
        _field("0", 8), _field(patient, 80), _field(recording, 80), _field(start_date, 8),
        _field(start_time, 8), _field(hdr_bytes, 8), _field("", 44), _field(declared, 8),
        _field(_num(record_duration_s), 8), _field(n_ch, 4)])
    per_signal = [
        [_field(lab, 16) for lab in labels],
        [_field("AgAgCl electrode", 80)] * n_ch,
        [_field("uV", 8)] * n_ch,
        [_field(_num(PHYS_MIN), 8)] * n_ch,
        [_field(_num(PHYS_MAX), 8)] * n_ch,
        [_field(DIG_MIN, 8)] * n_ch,
        [_field(DIG_MAX, 8)] * n_ch,
        [_field("HP:0.5Hz LP:45Hz", 80)] * n_ch,
        [_field(spr, 8)] * n_ch,
        [_field("", 32)] * n_ch,
    ]
    head += b"".join(b"".join(col) for col in per_signal)
    assert len(head) == hdr_bytes

    # records: for each record, channel 0 samples, then channel 1, ...
    data = dig.reshape(n_ch, n_rec, spr).transpose(1, 0, 2)
    keep = n_rec - int(truncate_records)
    if keep < 1:
        raise ValueError("truncate_records removes every record")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(head)
        f.write(np.ascontiguousarray(data[:keep]).tobytes())
    return path
