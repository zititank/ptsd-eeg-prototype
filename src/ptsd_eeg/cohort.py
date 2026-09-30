"""The evaluation cohort, rebuilt from the raw data folders only.

Folder layout: see `ptsd_eeg.layout` (generic names by default, overridable with `$PTSD_LAYOUT`).

Batches (reporting and bag strata):

- `PTSD`, `SOMA`       the PTSD and somatoform folders.
- `CTRL_B`             control batch B: training controls whose rest file carries the layout's batch-B start
                       date (one recording session with non-physiological signals; see docs/data_integrity.md).
- `CTRL_other`         the remaining training controls (the physiologically valid controls).
- `CTRL_C`             extended controls whose folder name starts with the layout's `ctrl_c_prefix`.
- `CTRL_elderly`       the remaining extended controls (a 65+ cohort).

The row order (training controls, PTSD, somatoform, extended controls; each sorted by folder name)
matters: bootstrap index sets are drawn by position.
"""
from __future__ import annotations

import hashlib
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .features import SET_A, extract_array
from .io import load_edf_with_header, read_edf_header, rest_path, subject_dirs
from .layout import get_layout
from .qc import coupling_qc, min_data_flags, qc_flags

PRIMARY = ["PTSD", "CTRL_B", "CTRL_other"]
"""Batches of the 90-subject training / primary-evaluation set."""


def discover(data_root: Path | str) -> list[dict]:
    """Subject folders in canonical cohort order, with group key and source (train / extended)."""
    lay = get_layout()
    root = Path(data_root)
    train = root / lay.train_dir
    rows = []
    train_controls = {s.name for s in subject_dirs(train / lay.group_dirs["control"])}
    for key in ("control", "ptsd", "soma"):
        for s in subject_dirs(train / lay.group_dirs[key]):
            rows.append(dict(subject_id=s.name, group=key, source="train", path=str(s)))
    extended = root / lay.extended_dir
    if extended.is_dir():
        for s in subject_dirs(extended):
            if s.name not in train_controls:
                rows.append(dict(subject_id=s.name, group="control", source="extended", path=str(s)))
    return rows


def control_stratum(start_date: str | None) -> str:
    """`CTRL_B` for the batch-B start date of the layout, else `CTRL_other`."""
    b = get_layout().batch_b_start_date
    return "CTRL_B" if b is not None and start_date == b else "CTRL_other"


def assign_batch(group: str, source: str, subject_id: str, start_date: str | None) -> str:
    """Batch label of one subject (see module docstring); `group` is the key `control`, `ptsd` or `soma`."""
    if group == "ptsd":
        return "PTSD"
    if group == "soma":
        return "SOMA"
    if source == "extended":
        return "CTRL_C" if subject_id.startswith(get_layout().ctrl_c_prefix) else "CTRL_elderly"
    return control_stratum(start_date)


def raw_row(item: dict) -> dict:
    """Label-free measurements of one subject's rest recording.

    Set A (O1+O2 and the O1-only sensitivity variant), n_kept epochs, duration, coupling QC, minimum-data
    flags, the header start date (batch assignment only) and the SHA-256 of the rest file (duplicate
    detection). Unreadable files give `scorable=False`.
    """
    path = Path(item["path"])
    row = dict(subject_id=item["subject_id"])
    warnings.simplefilter("ignore")
    try:
        rp = rest_path(path)
        row["rest_sha256"] = hashlib.sha256(rp.read_bytes()).hexdigest()
        with rp.open("rb") as f:
            row["start_date"] = f.read(176)[168:176].decode("latin-1", errors="replace").strip()
        row.update(min_data_flags(read_edf_header(rp)))
    except (OSError, ValueError, IndexError):
        row.update(min_data_fail=False)
    try:
        x, fs, h = load_edf_with_header(rest_path(path))
    except (OSError, ValueError, IndexError) as exc:
        return dict(row, scorable=False, error=f"{type(exc).__name__}: {exc}")
    try:
        f, _, _ = extract_array(x, fs)
        row.update({k: f[k] for k in [*SET_A, "n_kept", "scorable"]})
    except (OSError, ValueError, IndexError) as exc:
        row.update(scorable=False, error=f"{type(exc).__name__}: {exc}")
    try:
        f1, _, _ = extract_array(x, fs, occipital=("O1",))
        row.update({f"O1_{k}": f1[k] for k in [*SET_A, "scorable"]})
    except (OSError, ValueError, IndexError):
        row["O1_scorable"] = False
    row["duration_s"] = float(h.actual_records * h.record_duration_s)
    try:
        row.update(coupling_qc(x, fs))
    except ValueError:
        row.update(median_abs_r=np.nan, fp1fp2_r=np.nan)
    row.update(qc_flags(row["duration_s"], row["median_abs_r"], row["fp1fp2_r"]))
    return row


def build_cohort(data_root: Path | str, n_jobs: int = 4) -> pd.DataFrame:
    """One row per subject folder with batch, Set A, QC and gate flags, in canonical order.

    `is_duplicate` marks rest files byte-identical to an earlier subject of the same batch (the first
    subject in sorted order is kept); it is used only for the "unique signals" somatoform FPR.
    """
    items = discover(data_root)
    if n_jobs == 1:
        raw = [raw_row(it) for it in items]
    else:
        with ProcessPoolExecutor(n_jobs) as ex:
            raw = list(ex.map(raw_row, items, chunksize=4))
    c = pd.DataFrame(items).merge(pd.DataFrame(raw), on="subject_id", how="left", validate="one_to_one")
    c["batch"] = [assign_batch(r.group, r.source, r.subject_id, r.start_date if isinstance(r.start_date, str) else None)
                  for r in c.itertuples()]
    for k in ["scorable", "qc_fail", "too_short", "O1_scorable", "min_data_fail"]:
        if k not in c:
            c[k] = False
        c[k] = c[k].astype(object).where(c[k].notna(), False).astype(bool)
    c["is_duplicate"] = c.duplicated(["batch", "rest_sha256"]) & c.rest_sha256.notna()
    return c


def training_frame(cohort: pd.DataFrame, soma: bool) -> pd.DataFrame:
    """The training rows in LOSO order: scorable primary subjects sorted by (label, id); C4 appends somatoform."""
    prim = cohort[cohort.scorable & cohort.batch.isin(PRIMARY)].copy()
    prim["g"] = (prim.batch == "PTSD").astype(int)
    tr = prim.sort_values(["g", "subject_id"]).drop(columns="g")
    if soma:
        tr = pd.concat([tr, cohort[cohort.scorable & (cohort.batch == "SOMA")].sort_values("subject_id")])
    tr = tr.reset_index(drop=True)
    tr["y"] = (tr.batch == "PTSD").astype(int)
    return tr
