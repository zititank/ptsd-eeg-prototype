# Glossary

Short definitions of terms used in the code and docs. Numbers here are protocol constants, not results.

## Evaluation

- **AUC (area under the ROC curve).** The probability that a random PTSD patient scores higher than a random control.
  0.5 is chance, 1.0 is perfect separation. Independent of the threshold.
- **ROC curve.** Sensitivity against false-positive rate over all thresholds.
- **Sensitivity / specificity.** Share of PTSD patients scored ≥ 0.5 / share of non-PTSD people scored < 0.5.
- **FPR (false-positive rate).** Share of non-PTSD people scored ≥ 0.5 (1 − specificity). The protocol's main
  specificity metric is the FPR over the 97 somatoform and 65+ folders, fallbacks included.
- **Balanced accuracy.** Mean of sensitivity and specificity; not inflated by unequal class sizes.
- **Proxy points.** Our estimate of the task's objective score: 30 · max(0, 2·AUC − 1) + 20 · (1 − FPR).
  An approximation; the closed test was never available to us.
- **Flip rate.** How unstable one subject's decision is. The whole training procedure is refitted on 200
  class-stratified bootstrap samples; per subject, `min(p, 1 − p)` where `p` is the share of refits scoring it
  ≥ 0.5. 0 means always the same decision, 0.5 means a coin flip. The protocol averages it over scorable subjects
  outside control batch B.
- **Bootstrap confidence interval.** Resample subjects with replacement, recompute the metric, take the 2.5th and
  97.5th percentiles. The protocol uses 2000 resamples stratified by batch.
- **Paired bootstrap Δ.** The difference between two models computed on the same resamples, so shared noise cancels.

## Validation and model

- **LOSO (leave-one-subject-out).** Train on all subjects but one, score that one, repeat for everyone.
- **Nested cross-validation.** Hyperparameters (here the LDA shrinkage) are chosen by an inner cross-validation
  inside each outer training fold, so the outer score never influences model choice.
- **Fold member.** The model fitted in one outer LOSO fold. The shipped model averages all 90.
- **Shrinkage LDA.** Linear discriminant analysis whose covariance matrix is pulled towards a diagonal target;
  stabilises estimates with few subjects. Grid: auto, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90.
- **CLR (centred log-ratio).** A transform for proportions that sum to one: `clr_alpha_O = ln a − g`, with
  `g = (ln a + ln b + ln r) / 3`. Removes the bound at [0, 1] and the forced dependence between shares.
- **Logit.** `ln(p / (1 − p))`; maps a proportion to the whole real line.
- **Winsorising.** Clipping each feature at the training fold's 1st and 99th percentiles; test values use the same bounds.
- **Robust scaling.** Subtract the training-fold median, divide by the interquartile range. Model coefficients are
  in these units.
- **Bagging.** Fit many models on bootstrap samples and average their predictions; reduces variance near the threshold.
- **Stratum-balanced sampling.** Each bag draws negatives so that every control stratum has the same total
  probability (per subject ∝ 1 / (number of strata × stratum size)), so a large batch cannot dominate.
- **Fallback.** A fixed score for recordings that cannot be scored: 24/90 = 0.267, the training PTSD prevalence,
  always below the threshold.
- **Minimum-data gate.** The predeclared v4.1 rule: rest shorter than 30 s or truncated relative to its header →
  fallback.
- **OOD gate (out-of-distribution).** A label-free rule giving the fallback when a recording looks unlike the
  training data. Tried in candidate C3, not shipped.

## Data and quality

- **Control batch B (`CTRL_B`).** The original training controls of one recording batch (46 scorable). Their files
  contain no physiological EEG. The other original controls (`CTRL_other`, 20 scorable) are valid.
- **Control set C (`CTRL_C`).** Young expanded controls, evaluation only.
- **65+ (`CTRL_elderly`).** Older expanded controls aged 65 and over, evaluation only.
- **Coupling QC.** A label-free skin-contact test: a recording fails if the median absolute correlation over the 15
  channel pairs is below 0.05 or the Fp1–Fp2 correlation is below 0.30 (2–40 Hz).
- **Data-section hash.** SHA-256 of the EDF bytes after the header (signal only). Proves signals are unchanged and
  finds duplicate recordings.
- **Task-duration feature.** How long a subject took for the timed attention task. Analysed as a descriptive variant only, never used by the shipped model.
- **Scorable.** At least one clean epoch and finite relative alpha and beta.

## EEG features and responses

- **IAF (individual alpha frequency).** Frequency of the eyes-closed occipital alpha peak, usually near 10 Hz;
  `rest_IAF`.
- **Relative alpha / beta.** Share of O1/O2 spectral power in 8–13 Hz / 13–30 Hz relative to 1–40 Hz;
  `rest_rel_alpha_O`, `rest_rel_beta_O`. `rest_rel_alpha_IAF` uses IAF ± 2 Hz. The four together are **Set A**.
- **Aperiodic exponent.** Slope of the 1/f background of the spectrum; flattens with age. Not used by the model.

## Clinical scales

- **PCL-5.** PTSD Checklist for DSM-5, a self-report severity questionnaire.
- **CAPS-5.** Clinician-Administered PTSD Scale for DSM-5, a structured interview. Neither is available for our
  patients; PCL-5 was available in the public external dataset.
