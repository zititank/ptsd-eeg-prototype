"""Synthetic 6-channel headband subject folders for demos and tests (no real data involved).

    python examples/make_synthetic_edf.py demo OUT_DIR [--seed 0]        flat folder of subjects for `predict`
    python examples/make_synthetic_edf.py data-root OUT_DIR [--seed 0]   generic data-root layout for `train` / `reproduce`

Each subject folder holds `task1.edf ... task5.edf` (short task recordings) and one resting-state recording
`rest.edf`, six EEG channels (O1, T3, Fp1, Fp2, T4, O2) plus an event channel, at 125 Hz, 1 s data records.
The data-root layout is the default of `ptsd_eeg.layout.Layout`.

Signal model (a caricature, not a physiological simulation): per-channel 1/f background, a shared
component that gives realistic inter-channel coupling, an extra shared frontal component (Fp1/Fp2
correlation), an alpha rhythm at the subject's alpha frequency that is strongest over O1/O2, and
band-limited beta. `high_alpha` subjects get more occipital alpha and less beta than `low_alpha` ones,
which moves the shipped model's score up (see docs/science.md for why that direction is itself a caveat).
The synthetic groups carry no clinical meaning.

Edge cases in the demo set exercise the minimum-data gate and the fallback:
`short` (20 s), `truncated` (header declares more records than present), `flat` (all zeros, unscorable),
`no_rest` (no rest file), and `unknown_records` (header record count -1, still scored normally).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edf_writer import SYNTHETIC_DATE, write_edf  # noqa: E402

from ptsd_eeg.layout import Layout  # noqa: E402

FS = 125.0
CHANNELS = ["O1", "T3", "Fp1", "Fp2", "T4", "O2"]
FILE_LABELS = ["Fp1", "Fp2", "T3", "T4", "O1", "O2", "Event"]
"""Channel order inside the synthetic files (deliberately not the pipeline order: selection is by label)."""
LAYOUT = Layout()
"""Always the generic default, whatever $PTSD_LAYOUT says: synthetic roots never mimic a private layout."""
REST_NAME = "rest.edf"
TASK_NAMES = [f"task{i}.edf" for i in range(1, 6)]
ALPHA_GAIN = {"O1": 1.0, "O2": 1.0, "T3": 0.35, "T4": 0.35, "Fp1": 0.12, "Fp2": 0.12}


def _pink(rng: np.random.Generator, n: int, exponent: float = 1.0) -> np.ndarray:
    """Unit-variance noise with a 1/f^exponent power spectrum."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    x = np.fft.irfft(spec / f ** (exponent / 2), n)
    return x / x.std()


def _band_noise(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def make_signals(rng: np.random.Generator, kind: str, duration_s: float = 60.0) -> np.ndarray:
    """Six-channel synthetic rest EEG in µV, rows in CHANNELS order; `kind` is 'high_alpha' or 'low_alpha'."""
    n = int(round(duration_s * FS))
    t = np.arange(n) / FS
    alpha_amp = rng.uniform(10, 14) if kind == "high_alpha" else rng.uniform(4, 6)
    beta_amp = rng.uniform(2, 3) if kind == "high_alpha" else rng.uniform(4, 5)
    iaf = rng.uniform(9.0, 11.0)
    envelope = 1 + 0.4 * _band_noise(rng, n, 0.05, 0.5)
    phase = 2 * np.pi * np.cumsum(iaf + 0.3 * _band_noise(rng, n, 0.1, 1.0)) / FS
    alpha = alpha_amp * np.clip(envelope, 0.2, None) * np.sin(phase)
    common = _pink(rng, n)
    frontal = _pink(rng, n)
    out = np.empty((6, n))
    for i, ch in enumerate(CHANNELS):
        x = 12 * (0.8 * _pink(rng, n) + 0.6 * common)
        if ch.startswith("Fp"):
            x += 10 * frontal
        x += ALPHA_GAIN[ch] * alpha + beta_amp * _band_noise(rng, n, 13, 30)
        x += 0.5 * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 2 * np.pi))
        out[i] = x
    return out


def _file_matrix(signals: np.ndarray) -> np.ndarray:
    """Rows in FILE_LABELS order (the event channel is zero)."""
    rows = [signals[CHANNELS.index(lab)] if lab in CHANNELS else np.zeros(signals.shape[1]) for lab in FILE_LABELS]
    return np.vstack(rows)


def write_subject(folder: Path, rng: np.random.Generator, kind: str, edge: str | None = None,
                  start_date: str = SYNTHETIC_DATE, rest_name: str = REST_NAME) -> Path:
    """One synthetic subject folder; `edge` selects a defect (see module docstring)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in TASK_NAMES:
        write_edf(folder / name, _file_matrix(make_signals(rng, kind, 10.0)), FS, FILE_LABELS, start_date=start_date)
    if edge == "no_rest":
        return folder
    duration = 20.0 if edge == "short" else 60.0
    sig = make_signals(rng, kind, duration)
    if edge == "flat":
        sig = np.zeros_like(sig)
    kw = {}
    if edge == "truncated":
        kw = dict(truncate_records=15)
    elif edge == "unknown_records":
        kw = dict(header_records=-1)
    write_edf(folder / rest_name, _file_matrix(sig), FS, FILE_LABELS, start_date=start_date, **kw)
    return folder


DEMO_SUBJECTS = (
    [(f"SYN-H{i:02d}", "high_alpha", None) for i in range(1, 5)]
    + [(f"SYN-L{i:02d}", "low_alpha", None) for i in range(1, 5)]
    + [("SYN-E01-short", "high_alpha", "short"), ("SYN-E02-truncated", "high_alpha", "truncated"),
       ("SYN-E03-flat", "low_alpha", "flat"), ("SYN-E04-no-rest", "low_alpha", "no_rest"),
       ("SYN-E05-unknown-records", "high_alpha", "unknown_records")]
)


def make_demo_set(out_dir: Path | str, seed: int = 0) -> list[dict]:
    """A flat folder of synthetic subjects for `ptsd-eeg predict`; returns one dict per subject."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    rows = []
    for sid, kind, edge in DEMO_SUBJECTS:
        write_subject(out_dir / sid, rng, kind, edge)
        rows.append(dict(subject_id=sid, kind=kind, edge=edge or ""))
    return rows


def make_data_root(out_dir: Path | str, seed: int = 0, n_ptsd: int = 8, n_ctrl_other: int = 8,
                   n_ctrl_b: int = 6, n_soma: int = 6, n_ctrl_c: int = 4, n_elderly: int = 4) -> Path:
    """Generic data-root layout with every batch the evaluation expects (see `ptsd_eeg.cohort`).

    Synthetic "PTSD" subjects are `high_alpha`, controls mostly `low_alpha`; the control batch B subjects get
    the layout's batch-B start date. One PTSD folder has a short rest file so the gate and fallback appear in
    the cohort.
    """
    root = Path(out_dir)
    train = root / LAYOUT.train_dir
    dirs = LAYOUT.group_dirs
    extended = root / LAYOUT.extended_dir
    rng = np.random.default_rng(seed)
    for i in range(n_ptsd):
        write_subject(train / dirs["ptsd"] / f"SYN-P{i:02d}", rng, "high_alpha" if i % 4 else "low_alpha",
                      edge="short" if i == n_ptsd - 1 else None)
    for i in range(n_ctrl_other):
        write_subject(train / dirs["control"] / f"SYN-C{i:02d}", rng, "low_alpha" if i % 3 else "high_alpha")
    for i in range(n_ctrl_b):
        write_subject(train / dirs["control"] / f"SYN-D{i:02d}", rng, "low_alpha",
                      start_date=LAYOUT.batch_b_start_date)
    for i in range(n_soma):
        write_subject(train / dirs["soma"] / f"SYN-S{i:02d}", rng, "high_alpha" if i % 2 else "low_alpha")
    for i in range(n_ctrl_c):
        write_subject(extended / f"{LAYOUT.ctrl_c_prefix}SYN{i:02d}", rng, "low_alpha")
    for i in range(n_elderly):
        write_subject(extended / f"SYN-O{i:02d}", rng, "low_alpha")
    return root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("layout", choices=["demo", "data-root"])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.layout == "demo":
        rows = make_demo_set(a.out_dir, a.seed)
        print(f"wrote {len(rows)} synthetic subject folders to {a.out_dir}")
    else:
        make_data_root(a.out_dir, a.seed)
        print(f"wrote a synthetic data root to {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
