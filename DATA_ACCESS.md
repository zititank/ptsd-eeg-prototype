# Data access

**The data are confidential and are not available.** They are resting and task EEG recordings of patients and
volunteers, made available to hackathon participants under a data agreement for the task only. They are not
included in this repository, cannot be downloaded from it, and we cannot share or forward them. The hackathon,
the data provider and the recording details are withheld under the same agreement.

## What you can do without the data

- Run the demo, the synthetic tests and the CLI on synthetic EDF files (`make demo`, `make test`).
- Rebuild every figure and table from the aggregate CSVs in [`results/`](results/) (`make figures`).
- Rerun the external check on the public Mendeley dataset (du Bois et al., CC BY 4.0,
  doi:10.17632/gsxphk87mc.3), see [docs/external_check.md](docs/external_check.md).
- Train and evaluate the same pipeline on your own resting EEG, laid out as below.

A second public dataset that would allow a stronger external check (eyes-closed rest, about 30 trauma-exposed
adults) exists at Queensland University of Technology (doi:10.25912/RDF_1684466012814); access is by request
to its authors.

## Folder layout the code reads

The folder and file names below are the generic defaults (`src/ptsd_eeg/layout.py`). A dataset with other names
is read by pointing `PTSD_LAYOUT` (or `LAYOUT=` in the Makefile) to a small JSON file that overrides them; the
layout file for the original data is private.

```text
DATA/
├── train/
│   ├── control/               training controls; one folder per subject
│   │   └── <subject>/
│   │       ├── task1.edf … task5.edf    task recordings (not used by the model)
│   │       └── rest.edf                 eyes-closed rest (the only file the model reads)
│   ├── ptsd/                  PTSD; same structure
│   │   └── <subject>/ …
│   └── soma/                  somatoform; same structure (evaluation only, never trained on)
│       └── <subject>/ …
└── extended_controls/         expanded controls (control set C and 65+); evaluation only
    └── <subject>/ …           subjects already present in the training controls are skipped
```

Notes:

- The rest file is found by name: any `.edf` whose stem ends in the layout's rest suffix (`rest` by default),
  ignoring case.
- Channels are read by label (O1, T3, Fp1, Fp2, T4, O2), not by position. Sampling rate is read from each file.
- Batches are assigned from the folders and the layout file only (`src/ptsd_eeg/cohort.py`): training controls
  whose recording session matches the layout's batch-B session form `CTRL_B`, the other training controls
  `CTRL_other`; expanded controls whose folder name starts with the layout's control-set-C prefix form `CTRL_C`,
  the rest `CTRL_elderly`. The session is never a model input.

Then run:

```bash
make test-full DATA=/path/to/DATA LAYOUT=/path/to/layout.json   # adds the data-gated tests
make reproduce DATA=/path/to/DATA LAYOUT=/path/to/layout.json   # nested-LOSO retrain of v3, C2 and C2G vs results/metrics.csv
ptsd-eeg train --data /path/to/DATA/train --recipe C2G --out my_weights/
# without `pip install -e .`: PYTHONPATH=src python -m ptsd_eeg.cli train ... (same arguments)
```

The data-gated tests and `make reproduce` check against the original cohort, so on other data they report
differences by design. A full `make reproduce` takes about a minute on four cores; adding the 200-refit
flip-rate bootstrap (`--flip`) takes several minutes more (see
[docs/reproducibility.md](docs/reproducibility.md#runtime)).

`ptsd-eeg train` takes the folder that directly contains the three group folders and writes a weights folder that
`ptsd-eeg predict --weights my_weights/` can use; it also writes per-subject files (`training_oof.csv`,
`training_coverage.csv`) that must stay private. See [docs/reproducibility.md](docs/reproducibility.md).
