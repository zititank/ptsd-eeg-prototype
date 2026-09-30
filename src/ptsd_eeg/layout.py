"""Folder and file naming of a data root.

The confidential dataset is not public, and neither are its folder names. The code only knows a generic
layout (the defaults below, also written by `examples/make_synthetic_edf.py`):

    DATA/
      train/control/<subject>/rest.edf   training controls
      train/ptsd/<subject>/...           PTSD
      train/soma/<subject>/...           somatoform disorder (never trained on by C2G)
      extended_controls/<subject>/...    extended control set; subjects already in the training
                                         controls are skipped

A data holder with different names points the environment variable `PTSD_LAYOUT` to a JSON file that
overrides any of the fields of `Layout` (keep that file outside the repository). Example:

    {"train_dir": "train", "group_dirs": {"control": "hc", "ptsd": "ptsd", "soma": "sfd"},
     "extended_dir": "more_controls", "rest_suffix": "_EC", "batch_b_start_date": "01.01.00",
     "ctrl_c_prefix": "X"}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

LAYOUT_ENV = "PTSD_LAYOUT"


@dataclass(frozen=True)
class Layout:
    train_dir: str = "train"
    """Folder (under the data root) that holds the three group folders of the training set."""
    group_dirs: dict[str, str] = field(
        default_factory=lambda: {"control": "control", "ptsd": "ptsd", "soma": "soma"})
    """Group folder names for the keys `control`, `ptsd` and `soma`."""
    extended_dir: str = "extended_controls"
    """Folder (under the data root) of the extended control set; optional."""
    rest_suffix: str = "REST"
    """The resting-state EDF is the one file whose upper-cased stem ends with this suffix."""
    batch_b_start_date: str | None = "02.01.00"
    """EDF start-date field (`dd.mm.yy`) that marks control batch B (one recording session)."""
    ctrl_c_prefix: str = "C-"
    """Extended controls whose folder name starts with this prefix form control set C; the rest are 65+."""
    expected_counts: dict | None = None
    """Optional cohort sizes for the data-gated tests (private layouts only)."""


_CACHE: dict[str, Layout] = {}


def get_layout() -> Layout:
    """The layout from `$PTSD_LAYOUT` (read at call time, cached per path), else the generic default."""
    p = os.environ.get(LAYOUT_ENV, "")
    if p not in _CACHE:
        _CACHE[p] = Layout(**json.loads(Path(p).read_text(encoding="utf-8"))) if p else Layout()
    return _CACHE[p]
