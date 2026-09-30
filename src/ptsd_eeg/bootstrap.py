"""Uncertainty of PROTOCOL_v4: stratified bootstrap of the evaluation set and the flip-rate refits.

- Evaluation bootstrap (2000 replicates, seed 20260928): subjects are resampled with replacement within
  each batch (`BOOT_STRATA`), scores stay fixed. The same index sets are reused for every model, so
  differences between models are paired.
- Flip rate (200 refits, seed 20260927): the whole training procedure is refitted on class-stratified
  bootstrap samples; a subject's flip rate is the share of refits on the minority side of 0.5.
  The protocol's criterion 2 averages it over scorable subjects outside the `CTRL_B` batch.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import model as M
from .cohort import PRIMARY, training_frame
from .features import SET_A
from .metrics import BATCH_ORDER, auc_rank, proxy

EVAL_SEED = 20260928
N_BOOT = 2000
N_FLIP = 200
BOOT_STRATA = ["PTSD", "CTRL_B", "CTRL_other", "SOMA", "CTRL_elderly"]


def bootstrap_index_sets(cohort: pd.DataFrame, n_boot: int = N_BOOT, seed: int = EVAL_SEED) -> list[dict]:
    """Row positions (into `cohort`) per stratum, drawn in the order of BOOT_STRATA.

    Primary strata resample scorable subjects only (the AUC population); specificity strata resample
    every folder (unscorable folders count at the fallback score).
    """
    pools = {b: np.flatnonzero(((cohort.batch == b) & cohort.scorable).to_numpy()) if b in PRIMARY
             else np.flatnonzero((cohort.batch == b).to_numpy()) for b in BOOT_STRATA}
    rng = np.random.default_rng(seed)
    return [{b: rng.choice(pools[b], pools[b].size, replace=True) for b in BOOT_STRATA} for _ in range(n_boot)]


def bootstrap_arrays(pred: pd.DataFrame, cohort: pd.DataFrame, idx_sets: list[dict]) -> dict[str, np.ndarray]:
    """Per-replicate AUC (overall and per control batch), specificity FPR, proxy points, balanced accuracy
    and the somatoform / elderly FPR."""
    s = pred.set_index("subject_id").loc[cohort.subject_id].score.to_numpy()
    keys = ["auc", "fpr", "proxy", "auc_vs_CTRL_other", "auc_vs_CTRL_B", "ba", "fpr_soma", "fpr_elderly"]
    out = {k: np.empty(len(idx_sets)) for k in keys}
    thr = M.THRESHOLD
    for b, d in enumerate(idx_sets):
        pos, cb, oth = d["PTSD"], d["CTRL_B"], d["CTRL_other"]
        ii = np.concatenate([pos, cb, oth])
        yy = np.r_[np.ones(pos.size), np.zeros(cb.size + oth.size)].astype(int)
        out["auc"][b] = auc_rank(yy, s[ii])
        out["auc_vs_CTRL_other"][b] = auc_rank(np.r_[np.ones(pos.size), np.zeros(oth.size)].astype(int),
                                               s[np.concatenate([pos, oth])])
        out["auc_vs_CTRL_B"][b] = auc_rank(np.r_[np.ones(pos.size), np.zeros(cb.size)].astype(int),
                                           s[np.concatenate([pos, cb])])
        out["fpr"][b] = float(np.mean(s[np.concatenate([d["SOMA"], d["CTRL_elderly"]])] >= thr))
        out["proxy"][b] = proxy(out["auc"][b], out["fpr"][b])
        se = np.mean(s[pos] >= thr)
        sp = 1 - np.mean(s[np.concatenate([cb, oth])] >= thr)
        out["ba"][b] = (se + sp) / 2
        out["fpr_soma"][b] = np.mean(s[d["SOMA"]] >= thr)
        out["fpr_elderly"][b] = np.mean(s[d["CTRL_elderly"]] >= thr)
    return out


def run_flip(name: str, cohort: pd.DataFrame, n_jobs: int = 4, n_boot: int = N_FLIP,
             min_data_gate: bool = False) -> tuple[pd.DataFrame, dict]:
    """Flip-rate bootstrap of recipe `name` over every scorable subject.

    With `min_data_gate`, subjects failing the minimum-data gate are set to the fallback in every refit
    (this is how C2G's flip rate is derived from the C2 refits). Returns the per-subject frame (keep it
    private) and an aggregate summary dict.
    """
    recipe = M.RECIPES[name]
    tr = training_frame(cohort, recipe["soma"])
    X = M.input_matrix(recipe, tr[SET_A].to_numpy(float))
    sc = cohort[cohort.scorable].reset_index(drop=True)
    Xs = M.input_matrix(recipe, sc[SET_A].to_numpy(float))
    gate = (sc.qc_fail | sc.too_short).to_numpy() if recipe["gate"] else None
    t0 = time.time()
    B, G, shr, _ = M.bootstrap_refits(X, tr.y.to_numpy(), tr.batch.to_numpy(), recipe, Xs, gate,
                                      n_boot=n_boot, n_jobs=n_jobs)
    if min_data_gate:
        gi = sc.min_data_fail.to_numpy(bool)
        B[gi], G[gi] = M.FALLBACK, True
    secs = time.time() - t0
    subj = sc[["subject_id", "batch"]].copy()
    subj["boot_flip_rate"] = [M.flip_rate(r) for r in B]
    subj["boot_mean"], subj["boot_sd"] = B.mean(1), B.std(1, ddof=1)
    subj["boot_gated_share"] = G.mean(1)
    crit = subj[subj.batch != "CTRL_B"]
    summ = dict(model=name, flip_rate=float(crit.boot_flip_rate.mean()), n_subjects=len(crit),
                flip_rate_all_scorable=float(subj.boot_flip_rate.mean()), n_all=len(subj),
                boot_gated_share_mean=float(G.mean()), n_refits=n_boot, seconds=secs,
                shrinkage_counts=json.dumps(pd.Series(shr).value_counts().to_dict()),
                **{f"flip_{b}": float(subj[subj.batch == b].boot_flip_rate.mean()) for b in BATCH_ORDER})
    return subj, summ
