# Review guide

A reading order through the code for a reviewer. The author used it for his own review pass before
publishing. Each stop has **What** (what the code does), **Why** (the reason for the design), **Where** (file and
function) and an `ELI5` line to fill in with your own one-sentence explanation. If you cannot write the ELI5 line,
that is the part to reread.

Background first: [science.md](science.md) (the method and the result) and [glossary.md](glossary.md) (terms).
The pipeline in one picture is in [pipeline.md](pipeline.md).

## How to review

- [ ] Set up the venv and run `make check` (= `make test`, `make privacy`, `make numbers`, `make links`,
      `make demo`; see [reproducibility.md](reproducibility.md)). All five must pass.
- [ ] Read [science.md](science.md) end to end; check that every number you read is also in `results/`.
- [ ] Walk the 17 stops below in order, with the code open next to this page. Fill in each ELI5 line.
- [ ] Read [limitations.md](limitations.md) and [ETHICS.md](../ETHICS.md); decide whether you agree with every item
      and whether anything is missing.
- [ ] Check the "Where AI wrote most" list at the end with extra care.
- [ ] Compare your ELI5 lines with [eli5.md](eli5.md) and the author's review statement in
      [ai_assistance.md](ai_assistance.md#what-the-author-reviewed).

## Part 1. From a file to four numbers

### 1. EDF reading and the subject-folder contract

- **What.** `read_edf_header` parses the EDF header and computes how many complete data records the file really
  contains (from its size), separately from what the header claims. `load_edf_with_header` picks the six channels
  O1, T3, Fp1, Fp2, T4, O2 by label, checks they share one sampling rate and converts digital values to µV with the
  per-channel calibration. `rest_path` finds the single rest file of a subject folder (stem ending in the layout's
  rest suffix, case-insensitive); more or fewer than one raises an error. Folder and file names come from
  `layout.get_layout()`: generic defaults, or a private JSON file named by `PTSD_LAYOUT`.
- **Why.** The source files are not uniform: sampling rates vary between files, record counts can be
  `-1` or wrong, files can be truncated, and channel order is not guaranteed. An own reader with no EDF dependency
  makes each of these explicit and lets the minimum-data gate see the header/file mismatch.
- **Where.** [`src/ptsd_eeg/io.py`](../src/ptsd_eeg/io.py): `read_edf_header`, `find_channel_indices`,
  `load_edf_with_header`, `is_rest_name`, `rest_path`, constant `CHANNELS`;
  [`src/ptsd_eeg/layout.py`](../src/ptsd_eeg/layout.py): `Layout`, `get_layout`. Tests:
  [`tests/test_io_synthetic.py`](../tests/test_io_synthetic.py).

> ELI5 (Bogdan): ...

### 2. Cohort and batch labels

- **What.** `discover` lists subject folders in a fixed order (training controls, PTSD, somatoform, then expanded
  controls). `assign_batch` labels each one: `PTSD`, `SOMA`, `CTRL_B` (training controls whose rest file header
  carries the layout's batch-B session start), `CTRL_other`, `CTRL_C` and `CTRL_elderly` (expanded controls, by folder
  prefix).
- **Why.** Batch is used for two things only: balancing the bags (stop 10) and reporting per batch. It is never a
  model input. The fixed order matters because the bootstrap draws subjects by position.
- **Where.** [`src/ptsd_eeg/cohort.py`](../src/ptsd_eeg/cohort.py): `discover`, `assign_batch`, `build_cohort`,
  `training_frame`; the packaged-training equivalent is `training_records` in
  [`src/ptsd_eeg/model.py`](../src/ptsd_eeg/model.py). Tests: `test_batches_follow_rules`,
  `test_cohort_order_is_canonical` in [`tests/test_leakage.py`](../tests/test_leakage.py).

> ELI5 (Bogdan): ...

### 3. Signal QC and the minimum-data gate

- **What.** Two separate mechanisms. `min_data_flags` (shipped, C2G) marks a rest recording as failing when it has
  less than 30 s of complete records or its data section is shorter than the header promises; a header record count
  of `-1` is never "truncated". `coupling_qc` / `qc_flags` compute the median absolute correlation over the 15
  channel pairs and the Fp1–Fp2 correlation on 2–40 Hz data and flag near-zero coupling; this was used only by
  candidate C3, which was not shipped.
- **Why.** C2 gave an ordinary score to a file cut to a third of its length. The gate is label-free, was
  predeclared in Amendment 1, and sends such files to the fallback. The coupling QC is what exposed control batch B
  (its frontal electrodes are uncorrelated, unlike every real recording), but as a model gate it removed too many
  subjects.
- **Where.** [`src/ptsd_eeg/qc.py`](../src/ptsd_eeg/qc.py): `min_data_flags`, `min_data_record`, `coupling_qc`,
  `qc_flags`, constants `MIN_DURATION_REST_S`, `QC_MEDIAN_ABS_R_MIN`, `QC_FP1_FP2_R_MIN`. Tests:
  `test_truncated_file_uses_complete_records_and_flags`, `test_short_recording_fails_gate` in
  [`tests/test_io_synthetic.py`](../tests/test_io_synthetic.py).

> ELI5 (Bogdan): ...

### 4. Preprocessing and epochs

- **What.** `preprocess_rest` subtracts each channel's median, applies a 50 Hz notch only when a clear 50 Hz spike
  exists, and a 1–40 Hz band-pass only when at least 5% of the power lies outside 1–40 Hz. `normalized_epochs`
  divides all channels by the mean O1/O2 interquartile range, cuts 2 s epochs with 50% overlap and drops epochs whose
  peak amplitude or variance has a robust z above 8 on any channel.
- **Why.** Some batches arrive already filtered; filtering again would distort the spectrum, hence the conditions.
  The shared scale makes epoch rejection independent of device gain. All thresholds were fixed before any
  validation and never tuned on labels.
- **Where.** [`src/ptsd_eeg/features.py`](../src/ptsd_eeg/features.py): `preprocess_rest`, `maybe_notch`,
  `bandpass_1_40`, `make_rest_epochs`, `normalized_epochs`, `robust_z`.

> ELI5 (Bogdan): ...

### 5. Welch PSD and relative band power

- **What.** `welch_epochs` computes a Welch power spectrum for every epoch and channel (2 s segments);
  `mean_channel_psd` averages over epochs, then over O1 and O2. `bandpower` integrates the spectrum over a band with
  the trapezoid rule; `relative_bandpower` divides by the 1–40 Hz power.
- **Why.** Relative power is dimensionless, so it is insensitive to amplitude calibration, which differs between
  batches. The headset's hardware band starts at 2 Hz, so the denominator is effectively 2–40 Hz.
- **Where.** [`src/ptsd_eeg/features.py`](../src/ptsd_eeg/features.py): `welch_epochs`, `mean_channel_psd`,
  `integrate`, `bandpower`, `relative_bandpower`.

> ELI5 (Bogdan): ...

### 6. Individual alpha frequency (IAF)

- **What.** `find_iaf` takes the highest point of the occipital spectrum between 7 and 14 Hz and keeps it only if it
  is at least 1.5 times the median of the 7–14 Hz spectrum outside ±1 Hz of the peak; otherwise IAF is missing (NaN).
- **Why.** Alpha frequency differs between people (and with age and drowsiness), so a fixed 8–13 Hz band can miss a
  person's alpha. Requiring a real peak avoids calling noise an alpha rhythm; missing IAF is imputed later, inside
  the training fold.
- **Where.** [`src/ptsd_eeg/features.py`](../src/ptsd_eeg/features.py): `find_iaf`.

> ELI5 (Bogdan): ...

### 7. Set A: the four features, and "scorable"

- **What.** `rest_spectral_features` returns `rest_IAF`, `rest_rel_alpha_O` (8–13 Hz / 1–40 Hz),
  `rest_rel_alpha_IAF` (IAF ± 2 Hz / 1–40 Hz, NaN without IAF) and `rest_rel_beta_O` (13–30 Hz / 1–40 Hz), plus three
  descriptive values the model never uses. `extract_array` marks a subject scorable when at least one epoch survives
  and relative alpha and beta are finite.
- **Why.** These are the frozen v3 definitions, following Kovacevic et al. [H1]; no feature was added or selected
  afterwards. Unscorable subjects are not imputed; they get the fallback (stop 12).
- **Where.** [`src/ptsd_eeg/features.py`](../src/ptsd_eeg/features.py): `SET_A`, `rest_spectral_features`,
  `extract_array`, `extract_subject`. Definitions table: [science.md §3](science.md#3-the-markers-set-a).

> ELI5 (Bogdan): ...

## Part 2. From four numbers to a score

### 8. CLR and logit transforms

- **What.** `c1_subject_transform` maps Set A to the model inputs. With a = relative alpha, b = relative beta and
  r = 1 − a − b (each floored at 1e-6): `clr_alpha_O = ln a − g` and `clr_beta_O = ln b − g` with
  g = (ln a + ln b + ln r) / 3; `logit_alpha_IAF = ln(p / (1 − p))`; `rest_IAF` unchanged. Stateless, per subject.
- **Why.** Relative powers are shares of one total, bounded in [0, 1] and dependent on each other. The centred
  log-ratio and the logit move them to an unbounded scale where a linear model is sensible.
- **Where.** [`src/ptsd_eeg/transforms.py`](../src/ptsd_eeg/transforms.py): `c1_subject_transform`, `C1_FEATURES`.
  Test: `test_clr_transform_is_centred` in [`tests/test_metrics.py`](../tests/test_metrics.py).

> ELI5 (Bogdan): ...

### 9. Imputation, winsorising 1/99 and robust scaling

- **What.** `make_prep("c1")` is median imputation, then `Winsorizer` (clip every feature to the training 1st and
  99th percentiles), then robust scaling (subtract the median, divide by the interquartile range). `compile_prep`
  turns the fitted chain into five plain arrays (fill, lo, hi, center, scale); `apply_prep` applies them.
- **Why.** With 89 training subjects per fold, a single extreme value can dominate a mean and a standard deviation;
  percentiles and medians are robust to that. Everything is fitted inside the training fold. Storing plain arrays
  means scoring never depends on scikit-learn internals or pickle.
- **Where.** [`src/ptsd_eeg/transforms.py`](../src/ptsd_eeg/transforms.py): `Winsorizer`, `make_prep`,
  `compile_prep`, `apply_prep`.

> ELI5 (Bogdan): ...

### 10. Shrinkage LDA, inner shrinkage selection, stratum-balanced bagging

- **What.** `make_model` is the preprocessing chain plus `LinearDiscriminantAnalysis(solver="lsqr", shrinkage=…)`.
  `choose_inner` runs a stratified 5-fold split inside the training fold (seed 20260924 + fold index), refits the
  preprocessing per split, and scores every value of the grid [auto, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90].
  Rule: among values with inner AUC ≥ 0.690, take the lowest control false-positive rate, then the highest AUC, then
  grid order. `bag_indices` draws 100 bootstrap bags: positives uniformly, negatives with weights so that each control
  stratum (batch B, other) has equal total probability. `fit_member` fits the chain on every bag with the chosen
  shrinkage; `member_proba` averages the 100 bag probabilities and clips to [0.02, 0.98].
- **Why.** Shrinkage stabilises the covariance estimate with few subjects. The inner selection keeps the choice out
  of the outer test subject's reach. Balanced bags stop the 46-subject control batch B from defining "control" on its
  own, and bagging reduces decision flips near 0.5. Clipping bounds the influence of any one member.
- **Where.** [`src/ptsd_eeg/model.py`](../src/ptsd_eeg/model.py): `GRID`, `INNER_AUC_MIN`, `make_model`,
  `choose_inner`, `bag_indices`, `fit_member`, `member_proba`, `RECIPES`. Test: `test_bags_balance_negative_strata`
  in [`tests/test_leakage.py`](../tests/test_leakage.py).

> ELI5 (Bogdan): ...

### 11. Nested leave-one-subject-out (`loso_fold`)

- **What.** `loso_fold(X, y, strata, i, recipe)` fits fold member i on every row except row i, with row i's index as
  the seed offset. `nested_loso` runs it for all 90 training subjects (optionally in parallel). Member i scores
  subject i to give the out-of-fold score used by every training metric.
- **Why.** The task required subject-level splits, and every learnt object (imputer, winsor bounds, scaler,
  shrinkage, bags) must be fitted without the scored subject. The deployed model is exactly these 90 members, so the
  validated models and the shipped model are the same objects.
- **Where.** [`src/ptsd_eeg/model.py`](../src/ptsd_eeg/model.py): `loso_fold`, `nested_loso`, `train_from_dir`.
  Tests: `test_heldout_row_cannot_influence_its_fold` in [`tests/test_leakage.py`](../tests/test_leakage.py),
  `test_training_oof_equals_heldout_member` in [`tests/test_pipeline_synthetic.py`](../tests/test_pipeline_synthetic.py).

> ELI5 (Bogdan): ...

### 12. Ensemble scoring, fallback 24/90 and the weights files

- **What.** `predict` reads a subject folder; any unreadable file returns the fallback. `score_record` returns the
  fallback 24/90 = 0.267 when the subject is unscorable or fails the minimum-data gate, otherwise the mean over the
  90 members of `member_proba`; a non-finite or out-of-range score raises instead of returning a number.
  `load_weights` verifies `SHA256SUMS`, then loads the `.npz` with `allow_pickle=False`. The CLI `predict` command
  writes one row per folder and never stops on one bad subject.
- **Why.** 24/90 is the PTSD share of the training set: an uninformative prior that is below the 0.5 threshold, so
  a recording that cannot be scored never produces a positive decision. Pickle-free weights cannot execute code.
- **Where.** [`src/ptsd_eeg/model.py`](../src/ptsd_eeg/model.py): `FALLBACK`, `score_record`, `subject_record`,
  `predict`; [`src/ptsd_eeg/weights.py`](../src/ptsd_eeg/weights.py): `load_weights`, `verify_weights`,
  `save_weights`; [`src/ptsd_eeg/cli.py`](../src/ptsd_eeg/cli.py): `_predict_dir`. Tests:
  [`tests/test_contract.py`](../tests/test_contract.py), [`tests/test_weights.py`](../tests/test_weights.py),
  [`tests/test_golden.py`](../tests/test_golden.py).

> ELI5 (Bogdan): ...

## Part 3. How the model was judged

### 13. Metrics: AUC, false-positive rates, proxy points

- **What.** `auc_rank` is the Mann–Whitney AUC with mid-ranks for ties. `point_metrics` computes the AUC of PTSD vs
  all training controls from out-of-fold scores, the AUC against each control batch separately, sensitivity, the
  false-positive rate over the 97 somatoform and 65+ folders (fallbacks included, counted as negatives), and
  `proxy` = 30 · max(0, 2·AUC − 1) + 20 · (1 − FPR). `batch_table` gives per-batch counts and positive rates.
- **Why.** Proxy points mirror the task's objective score. The per-batch AUCs are what exposes the artefact:
  the overall AUC mixes a trivial comparison (vs control batch B) with a hard one (vs valid controls).
- **Where.** [`src/ptsd_eeg/metrics.py`](../src/ptsd_eeg/metrics.py): `auc_rank`, `proxy`, `point_metrics`,
  `batch_table`, `balanced_accuracy`. Tests: [`tests/test_metrics.py`](../tests/test_metrics.py).

> ELI5 (Bogdan): ...

### 14. Paired bootstrap and flip rate

- **What.** `bootstrap_index_sets` draws 2000 resamples of subjects within each batch (seed 20260928); the same
  index sets are reused for every model, so model differences are paired. `bootstrap_arrays` recomputes AUCs, FPR,
  proxy points and balanced accuracy on each resample with the scores held fixed. The flip rate (`run_flip`,
  `bootstrap_refits`, `flip_rate`) refits the whole training procedure 200 times on class-stratified resamples and
  measures, per subject, the share of refits on the minority side of 0.5.
- **Why.** Pairing cancels the noise shared by two models scored on the same people, which makes the Δ proxy
  interval honest and narrow. The flip rate measures what the LOSO ensemble hides: how much a decision depends on
  which people happened to be in the training set.
- **Where.** [`src/ptsd_eeg/bootstrap.py`](../src/ptsd_eeg/bootstrap.py): `bootstrap_index_sets`,
  `bootstrap_arrays`, `run_flip`; [`src/ptsd_eeg/model.py`](../src/ptsd_eeg/model.py): `bootstrap_refits`,
  `flip_rate`. Test: `test_bootstrap_serial_equals_parallel` in [`tests/test_determinism.py`](../tests/test_determinism.py).

> ELI5 (Bogdan): ...

### 15. Gates G1 to G4 and the decision rule

- **What.** G1a `infold_check`: set the held-out row to 10⁶ and flip its label; the fold member must stay
  bit-identical. G1b `permutation_check`: one predeclared label permutation through the full nested procedure must
  give an AUC inside the recorded null range [0.142, 0.580]. `criteria` applies criterion 1 (Δ proxy vs v3 with a
  2.5th percentile above 0) and criterion 2 (flip rate ≤ v3 + 0.005); `decide` picks the highest proxy among
  candidates that pass both and the gates. G2 is the batch table (`metrics.batch_table`); G3, the inference contract,
  is covered here by [`tests/test_contract.py`](../tests/test_contract.py); G4, reproduction, is `reproduce.reproduce`
  / `compare` (retrain from raw data, compare with `results/metrics.csv` within 1e-9).
- **Why.** The rule was hashed before any candidate was scored, so the choice could not be steered by results.
  The gates catch leakage and irreproducibility, the two failures that make a small-sample result meaningless.
- **Where.** [`src/ptsd_eeg/gates.py`](../src/ptsd_eeg/gates.py): `infold_check`, `permutation_check`,
  `PERMUTATION_NULL_95`, `criteria`, `decide`; [`src/ptsd_eeg/reproduce.py`](../src/ptsd_eeg/reproduce.py):
  `reproduce`, `compare`. Outcomes: [`results/candidate_gates.csv`](../results/candidate_gates.csv),
  [`results/criteria.csv`](../results/criteria.csv). Test: `test_decision_rule` in
  [`tests/test_leakage.py`](../tests/test_leakage.py).

> ELI5 (Bogdan): ...

## Part 4. What keeps the repository safe and honest

### 16. The privacy scan

- **What.** `scan` walks every file (skipping `.git`, `.demo`, `.scratch` and other local folders) and fails on:
  subject-code patterns, tokens whose salted hash is in the private ID list (`PTSD_ID_HASHES`, kept outside the
  repository; skipped in CI), participant numbers of the public dataset, `dd.mm.yy` dates other than two allowlisted
  constants, EDF headers, forbidden file types (`.edf`, `.pkl`, …), files over 5 MB, home-directory paths, and group
  sizes of 1–4 in CSV size columns. Notebook sources and text outputs are scanned too. Upper-case tokens outside an
  acronym whitelist are printed as warnings.
- **Why.** The data are not ours to share, and identifiers leak through the most boring places: file names, notebook
  outputs, log lines, table cells. A scan in CI makes a leak fail loudly instead of being noticed after publishing.
  Findings are masked so the report itself leaks nothing.
- **Where.** [`tools/privacy_scan.py`](../tools/privacy_scan.py): `scan`, `scan_text`, `scan_csv`,
  `scan_notebook`, `HashList`, `ID_PATTERNS`, `ACRONYMS`. Tests: [`tests/test_privacy_scan.py`](../tests/test_privacy_scan.py).

> ELI5 (Bogdan): ...

### 17. The number checker

- **What.** Checks that the coefficient and shrinkage tables in `results/` match the public weights, recomputes more
  than 100 key numbers from `results/`, formats them exactly as the docs write them, and fails if one is missing from
  the doc that should contain it. Every other decimal number in the docs is matched against the shipped sources and
  reported if unmatched.
- **Why.** Prose drifts from the data, especially when much of it was drafted by AI. This ties the text to the
  aggregate tables mechanically.
- **Where.** [`tools/check_numbers.py`](../tools/check_numbers.py).

> ELI5 (Bogdan): ...

## Where AI wrote most, verify carefully

Most of the code and text was drafted by AI agents ([ai_assistance.md](ai_assistance.md)). These places matter most
and are the easiest to get subtly wrong:

- **`model.choose_inner` and `model.bag_indices`.** The inner selection rule must match
  PROTOCOL_v4 §1 (fixed ingredients) and the negative-stratum weights §2 (C2); the text is private, the summary is in
  [`protocols/README.md`](../protocols/README.md). Compare them line by line.
- **`bootstrap.bootstrap_index_sets` / `bootstrap_arrays`.** Which strata resample scorable subjects only and which
  resample every folder decides every confidence interval. Check against §3 (evaluation) of the protocol.
- **`qc.min_data_flags`.** The definition of "truncated" (data section vs header record count, `-1` never truncated)
  is the whole of Amendment 1. Check it against §2 of the amendment (summary in
  [`protocols/README.md`](../protocols/README.md)).
- **`gates.PERMUTATION_NULL_95`.** A hard-coded range from an earlier v3 run; the docstring says where it came from.
  You cannot re-derive it without the data; decide whether you are comfortable citing it.
- **`cohort.assign_batch`.** Batch labels drive the bag strata and every per-batch table. Check the batch-B session rule and
  the control-set-C prefix rule (both read from the layout file) against [data_integrity.md](data_integrity.md).
- **[references.md](references.md).** Every "Claim" and "Agreed?" line was summarised by an AI from the papers.
  Spot-check at least the H1, H4, H5, B1 and L1 entries against the original abstracts.
- **[limitations.md](limitations.md) item 8 (drowsiness) and [science.md §3](science.md#3-the-markers-set-a)
  (physiological meaning).** Interpretive prose without numbers is not covered by the number checker.
- **[ETHICS.md](../ETHICS.md) §1, the residual-risk paragraph about the weights.** Make sure you agree with the
  judgement, and confirm with the data owners.
- **`tools/privacy_scan.py`.** A scanner can only find what it looks for. Run it with both private lists (subject-ID
  hashes and the confidentiality denylist), and also grep the repository by hand for any real subject code, name,
  date or place you remember before pushing.
