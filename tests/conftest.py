"""Shared fixtures.

Two kinds of tests:
- synthetic (default, run in CI): generated EDF files only, no confidential data needed;
- data-gated (marker `data`): need the confidential dataset; they are skipped unless the environment
  variable PTSD_DATA points to the data root (see DATA_ACCESS.md).
"""
from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
PRIVATE_LAYOUT = os.environ.pop("PTSD_LAYOUT", None)
"""Synthetic tests always run on the generic layout; the private one is restored only to read the real data."""
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "examples"))
sys.dont_write_bytecode = True

import pytest  # noqa: E402

import ptsd_eeg  # noqa: E402

if not Path(ptsd_eeg.__file__).resolve().is_relative_to(REPO):
    raise RuntimeError(f"tests must import the package from this repository, got {ptsd_eeg.__file__}")

warnings.filterwarnings("ignore")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PTSD_DATA"):
        return
    skip = pytest.mark.skip(reason="confidential data not available (set PTSD_DATA to the data root)")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory) -> Path:
    """The synthetic demo set (13 subjects, including the defective ones), seed 0."""
    from make_synthetic_edf import make_demo_set
    d = tmp_path_factory.mktemp("demo")
    make_demo_set(d, seed=0)
    return d


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """A synthetic data root in the generic layout, with every evaluation batch."""
    from make_synthetic_edf import make_data_root
    return make_data_root(tmp_path_factory.mktemp("root"), seed=1)


@pytest.fixture(scope="session")
def synthetic_cohort(synthetic_root):
    from ptsd_eeg.cohort import build_cohort
    return build_cohort(synthetic_root, n_jobs=1)


@pytest.fixture(scope="session")
def shipped_model():
    from ptsd_eeg.weights import load_weights
    return load_weights(REPO / "weights")


@contextmanager
def private_layout():
    """`$PTSD_LAYOUT` as given by the caller (e.g. `make test-full`), for reading the confidential data only."""
    if PRIVATE_LAYOUT:
        os.environ["PTSD_LAYOUT"] = PRIVATE_LAYOUT
    try:
        yield
    finally:
        os.environ.pop("PTSD_LAYOUT", None)


@pytest.fixture(scope="session")
def real_layout():
    from ptsd_eeg.layout import get_layout
    with private_layout():
        return get_layout()


@pytest.fixture(scope="session")
def real_root(real_layout) -> Path:
    root = Path(os.environ.get("PTSD_DATA", ""))
    if not os.environ.get("PTSD_DATA") or not (root / real_layout.train_dir).is_dir():
        pytest.skip("confidential data not available (set PTSD_DATA to the data root and PTSD_LAYOUT to its layout)")
    return root


@pytest.fixture(scope="session")
def real_cohort(real_root):
    from ptsd_eeg.cohort import build_cohort
    with private_layout():
        return build_cohort(real_root, n_jobs=4)
