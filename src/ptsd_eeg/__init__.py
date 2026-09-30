"""Resting-state EEG PTSD research prototype (not a medical device).

Modules:
    io          minimal EDF reader and dataset file naming
    features    Set A spectral markers of the eyes-closed rest recording
    qc          label-free signal QC and the minimum-data gate
    transforms  compositional (CLR / logit) transform and in-fold robust preprocessing
    model       recipes, nested LOSO, bagged shrinkage LDA, scoring
    weights     pickle-free weights (.npz + JSON) and the ensemble coefficient table
    cohort      evaluation cohort and batches from the raw data folders
    metrics     AUC, false-positive rates, proxy points, batch tables
    bootstrap   paired evaluation bootstrap and flip-rate refits
    gates       leakage gates and the pre-registered decision rule
    reproduce   full retrain from raw data, compared with results/metrics.csv
    cli         the `ptsd-eeg` command
"""
__version__ = "4.1.1"
