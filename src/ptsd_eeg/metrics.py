"""Point metrics of PROTOCOL_v4: AUC, false-positive rates, proxy points, per-batch tables.

All functions take a prediction frame with one row per subject folder and the columns
`subject_id, batch, scorable, split, score, positive, gate_fraction` (+ `is_duplicate`), as produced by
`reproduce.run_model`. Unscorable subjects carry the fallback score 24/90 and count as negatives at the
0.5 threshold.

Definitions:

- AUC: PTSD vs all training controls, out-of-fold (outer LOSO) scores of the 90 scorable training subjects.
- Specificity FPR: share of somatoform + elderly folders (external groups, never trained on) at score >= 0.5.
- Proxy points (the hackathon's scoring proxy, PROTOCOL_v4 §4): 30 * max(0, 2 AUC - 1) + 20 * (1 - FPR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .cohort import PRIMARY

THRESHOLD = 0.5
SPEC = ["SOMA", "CTRL_elderly"]
BATCH_ORDER = ["PTSD", "CTRL_B", "CTRL_other", "SOMA", "CTRL_C", "CTRL_elderly"]


def auc_rank(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with mid-ranks for ties (fast, used inside the bootstrap)."""
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def proxy(auc: float, fpr: float) -> float:
    """Proxy points = 30 * max(0, 2 AUC - 1) + 20 * (1 - FPR); range 0-50."""
    return 30 * max(0.0, 2 * auc - 1) + 20 * (1 - fpr)


def ci(a: np.ndarray) -> tuple[float, float]:
    """Percentile 95% interval."""
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def point_metrics(pred: pd.DataFrame) -> dict:
    """Every point metric of the protocol (discrimination, per-group FPR, proxy points, gate shares)."""
    p = pred.set_index("subject_id")
    prim = p[p.scorable & p.batch.isin(PRIMARY)]
    y = (prim.batch == "PTSD").astype(int).to_numpy()
    s = prim.score.to_numpy()
    spec = p[p.batch.isin(SPEC)]
    out = dict(auc=float(roc_auc_score(y, s)), n_ptsd=int(y.sum()), n_control=int((y == 0).sum()))
    for b in ["CTRL_other", "CTRL_B"]:
        q = prim[prim.batch.isin(["PTSD", b])]
        out[f"auc_vs_{b}"] = float(roc_auc_score((q.batch == "PTSD").astype(int), q.score))
    out["sensitivity"] = float(prim[prim.batch == "PTSD"].positive.mean())
    out["sensitivity_with_fallback"] = float(p[p.batch == "PTSD"].positive.mean())
    out["control_fpr"] = float(prim[prim.batch != "PTSD"].positive.mean())
    out["fpr_soma_elderly"] = float(spec.positive.mean())
    out["n_soma_elderly"] = int(len(spec))
    out["n_soma_elderly_positive"] = int(spec.positive.sum())
    out["proxy_points"] = proxy(out["auc"], out["fpr_soma_elderly"])
    out["proxy_auc_part"] = 30 * max(0.0, 2 * out["auc"] - 1)
    out["proxy_fpr_part"] = 20 * (1 - out["fpr_soma_elderly"])
    for b in BATCH_ORDER:
        q = p[p.batch == b]
        out[f"positive_rate_{b}"] = float(q.positive.mean())
    soma = p[p.batch == "SOMA"]
    out["fpr_soma"] = float(soma.positive.mean())
    out["fpr_soma_scorable"] = float(soma[soma.scorable].positive.mean())
    out["fpr_soma_unique"] = float(soma[~soma.is_duplicate.astype(bool)].positive.mean())
    out["fpr_elderly"] = float(p[p.batch == "CTRL_elderly"].positive.mean())
    cc = p[p.batch == "CTRL_C"]
    out["fpr_ctrl_c"] = float(cc.positive.mean())
    out["fpr_ctrl_c_scorable"] = float(cc[cc.scorable].positive.mean())
    sc = p[p.scorable]
    out["gated_share_scorable_full"] = float((sc.gate_fraction == 1).mean())
    out["gated_share_scorable_any"] = float((sc.gate_fraction > 0).mean())
    out["gated_share_training"] = float((sc[sc.split == "outer_loso"].gate_fraction == 1).mean())
    ext = sc[sc.split == "outer_ensemble"]
    out["gated_share_external_full"] = float((ext.gate_fraction == 1).mean()) if len(ext) else np.nan
    out["gated_share_external_any"] = float((ext.gate_fraction > 0).mean()) if len(ext) else np.nan
    out["fallback_share_unscorable"] = float((~p.scorable).mean())
    return out


def point_metrics_subset(pred: pd.DataFrame) -> dict:
    """AUC, specificity FPR and proxy on an arbitrary subset of rows (sensitivity analyses)."""
    prim = pred[pred.scorable & pred.batch.isin(PRIMARY)]
    y = (prim.batch == "PTSD").astype(int)
    spec = pred[pred.batch.isin(SPEC)]
    auc = float(roc_auc_score(y, prim.score))
    fpr = float(spec.positive.mean())
    return dict(auc=auc, n_primary=len(prim), fpr_soma_elderly=fpr, n_soma_elderly=len(spec),
                proxy_points=proxy(auc, fpr), fpr_ctrl_c=float(pred[pred.batch == "CTRL_C"].positive.mean()),
                n_ctrl_c=int((pred.batch == "CTRL_C").sum()))


def batch_table(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per batch: folder counts, fallbacks, gated subjects, positives and score summaries."""
    rows = []
    for b in BATCH_ORDER:
        q = pred[pred.batch == b]
        sc = q[q.scorable]
        rows.append(dict(model=pred.model.iloc[0], batch=b, n_folders=len(q), n_scorable=len(sc),
                         n_fallback_unscorable=int((~q.scorable).sum()),
                         n_gated_full=int((sc.gate_fraction == 1).sum()), n_gated_any=int((sc.gate_fraction > 0).sum()),
                         n_positive=int(q.positive.sum()), positive_rate_folders=float(q.positive.mean()),
                         positive_rate_scorable=float(sc.positive.mean()) if len(sc) else np.nan,
                         mean_score=float(q.score.mean()), median_score=float(q.score.median()),
                         splits=",".join(sorted(q.split.unique()))))
    return pd.DataFrame(rows)


def balanced_accuracy(pred: pd.DataFrame) -> dict:
    """Specificity and balanced accuracy of the primary out-of-fold scores at threshold 0.5."""
    p = pred.set_index("subject_id")
    prim = p[p.scorable & p.batch.isin(PRIMARY)]
    sens = float(prim[prim.batch == "PTSD"].positive.mean())
    spec = float(1 - prim[prim.batch != "PTSD"].positive.mean())
    return dict(specificity=spec, ba=(sens + spec) / 2)
