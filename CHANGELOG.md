# Changelog

All notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/) from 4.0.0 on.

Versions up to 4.1.0 were developed in a private repository together with the confidential data; this public
repository starts with a fresh git history (why: [docs/development_history.md](docs/development_history.md)).
4.1.0-public was prepared but not published on its own: the first public tag is `v4.1.1`, which contains it. The
model it ships is the private 4.1.0 model (C2G).

## [4.1.1] - 2026

First public release: documentation, credits and a final confidentiality pass. Same model, same weights
(checksums unchanged) and the same model metrics as 4.1.0-public.

### Added

- [`docs/story.md`](docs/story.md): the project story in the author's own words (how the artefact was found,
  lessons tied to the numbers, next steps) and [`docs/eli5.md`](docs/eli5.md), a plain-language explanation.
- README header with a banner, badges, highlights, a story hook and animations; a Team section and
  [`CONTRIBUTORS.md`](CONTRIBUTORS.md).
- [`media/`](media/): five animations (MP4, GIF preview, poster) and the banner, rebuilt from `results/` and
  synthetic signals by `tools/make_animations.py` (`make animations`); outputs carry no metadata.
- `CITATION.cff` and the README BibTeX list the author, Bogdan (zititank), and the hackathon teammates AdetyTy and
  Qwertyqwerty579.

### Changed

- Confidentiality: per-batch folder totals, fallback counts and whole-dataset counts were removed from the docs,
  `results/C2G_batch_table.csv`, `results/batch_tables.csv` and `results/weights_equivalence.json`; counts are kept
  only as denominators of reported metrics. The model-card subgroup table now reports rates among scorable
  recordings.
- Confidentiality: the recording protocol is described only as "an eyes-closed resting recording plus a timed
  attention task"; the behavioural variant is called the task-duration feature (internal model ids in `results/`
  are unchanged), and its figure is now `figures/task_duration_age_confound.png`. The exploratory bonus analysis
  is reduced to one line in [`docs/limitations.md`](docs/limitations.md) (item 7); its detailed discussion and
  references were removed. Remaining descriptions of individual data defects were generalised.
- The data-gated batch-table test compares the published columns only.
- [`docs/ai_assistance.md`](docs/ai_assistance.md): tools used and the author's review statement;
  [TRIPOD+AI](docs/tripod_ai_checklist.md) items 18a/18b (no funding, no conflicts); full title of reference [L1].
- `tools/privacy_scan.py` accepts `.mp4` media; `tools/check_links.py` also checks `media/` paths.

## [4.1.0-public] - 2026

Public release. Same model and same results as 4.1.0; everything that could identify a person, the hackathon, the
data provider or the recording device was removed.

### Confidentiality scrub (before first publication)

- Batch labels neutralised: the non-physiological control batch is "control batch B" (`CTRL_B`), the young expanded
  controls "control set C" (`CTRL_C`). The rename keeps the sort order, so every number is bit-identical.
- Dataset folder and file names moved out of the code into a private layout file (`PTSD_LAYOUT`,
  `src/ptsd_eeg/layout.py`); the public defaults and the synthetic generator use a generic layout.
- Removed the EEG-derived data descriptions (integrity tables, per-batch spectra, per-batch missing-IAF and age-band
  tables) and the two spectra figures; added a synthetic brain vs non-brain illustration. Model-output tables and
  figures are unchanged.
- Protocol texts replaced by [`protocols/README.md`](protocols/README.md): the original SHA-256 hashes and a summary
  of the decision rule.
- `tools/privacy_scan.py` gained a confidentiality denylist kept outside the repository and a check for non-Latin
  text.

### Added

- Pickle-free weights: `weights/c2g_members.npz` + `model_meta.json` + `SHA256SUMS`, loaded with
  `numpy.load(allow_pickle=False)`. Training subject codes and the inner-selection audit trail were removed and the
  member order was shuffled; scores match the delivered model within 1.7e-16 on every real subject folder, with no
  decision changed ([`results/weights_equivalence.json`](results/weights_equivalence.json)).
- A minimal EDF writer and a synthetic 6-channel headband generator (`examples/`), including short, truncated, flat and
  incomplete subjects that exercise the minimum-data gate; `make demo` and a golden-output test.
- A synthetic test suite that runs without data (CI) and data-gated tests that skip unless `PTSD_DATA` is set.
- `tools/privacy_scan.py` (subject-code patterns, a salted-hash list of the real codes kept outside the repository,
  dates, EDF headers, forbidden file types, home paths, group sizes below 5) and `tools/check_numbers.py` (every key
  number in the docs recomputed from `results/`) and `tools/check_links.py` (links, anchors, images, repository
  paths in code spans, no unembedded figures), all run in CI (`make check` runs the same set locally).
- English documentation: README, model card, ethics, data access, disclaimer, science core, pipeline, data
  integrity, model selection, external check, limitations, references with a citation-to-decision map,
  reproducibility, glossary, TRIPOD+AI checklist, AI-assistance disclosure, development history, review guide.
- Aggregate result tables with provenance ([`results/MANIFEST.csv`](results/MANIFEST.csv)); figures built from
  aggregates only; screenshots of local runs; notebooks 01 (synthetic quickstart) and 02 (results and figures).
- MIT license, `CITATION.cff`, GitHub Actions CI.

### Changed

- The code was reorganised into the `ptsd_eeg` package with docstrings and type hints; the evaluation module was
  split into `metrics`, `bootstrap`, `gates`, `cohort` and `reproduce`. Numerical behaviour is unchanged
  (the data-gated tests reproduce `results/metrics.csv`).
- The protocols are represented by their pre-registration hashes and a decision-rule summary
  ([`protocols/README.md`](protocols/README.md)).

### Removed

- Hackathon delivery tooling (submission packaging, report and notebook builders), per-subject tables, recording
  dates, notebooks with real-data outputs, and the pickled model.

## [4.1.0]

Recipe C2G: C2 plus a minimum-data gate, under Amendment 1 of the protocol
(SHA-256 `38ad28a931699b6e3c284fccff8e701df8724bc2f6475254a79ee1f6217c83d8`, see
[`protocols/README.md`](protocols/README.md)).

### Added

- Minimum-data gate: a rest recording shorter than 30 s of complete data, or truncated relative to its EDF header,
  gets the fallback score 24/90 instead of an ordinary score. Fold members identical to C2; the gate fired for one
  evaluation-only subject; no metric changed.
- External direction check on public data (Mendeley, du Bois et al., CC BY 4.0) under its own hashed protocol and
  amendment: null and underpowered.

## [4.0.0]

Recipe C2, selected by the decision rule of the frozen PROTOCOL_v4
(SHA-256 `88ef6934f896d83ac2b5fa3e1f7c5a7e3ee6ad1cfe70bc97cf5a6f956a5bb047`, see
[`protocols/README.md`](protocols/README.md)).

### Added

- Data-integrity audit (control batch B contains no physiological EEG) and instability audit of v3.
- Candidates C1 (compositional transforms, winsorising, robust scaling), C2 (C1 + stratum-balanced bagging,
  B = 100, output clipped to [0.02, 0.98]), C3 (C2 + out-of-distribution/QC gate) and C4 (C2 + somatoform
  negatives), evaluated with a paired bootstrap, a flip-rate criterion and safety gates G1-G4.
- Reporting of the AUC against physiologically valid controls next to every headline AUC.

### Changed

- C2 replaced v3: proxy points 36.30 vs 30.59, flip rate 0.089 vs 0.106. The gain comes from fewer somatoform
  false positives; the AUC against valid controls stayed at chance (0.435 vs 0.410).

## [3.0.0]

Baseline: the hackathon submission model ("frozen v3").

### Added

- Set A features (individual alpha frequency and three relative band powers of O1/O2, eyes-closed rest),
  nested-LOSO shrinkage LDA, a 90-member ensemble, one shared amplitude scale for all channels, and a fixed fallback
  score 24/90 for unscorable recordings; a task-style `predict`/`run` interface.
