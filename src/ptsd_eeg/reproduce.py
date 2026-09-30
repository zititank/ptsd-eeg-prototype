"""Full retrain from the raw data and comparison with the published aggregate metrics.

    ptsd-eeg reproduce --data DATA_ROOT [--models v3 C2 C2G] [--flip] [--out DIR]

Steps: build the cohort from raw EDF (`cohort.build_cohort`), run the nested LOSO of every requested
recipe, score every subject (training subjects out-of-fold, external subjects by the ensemble mean),
compute the protocol metrics with the paired bootstrap, and compare them with `results/metrics.csv`.

Only aggregate files are written by default (`metrics_reproduced.csv`, `batch_tables.csv`,
`comparison.csv`). `--save-predictions` also writes per-subject predictions: those contain subject IDs
of the confidential dataset and must never be committed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as M
from .bootstrap import bootstrap_arrays, bootstrap_index_sets, run_flip
from .cohort import build_cohort, training_frame
from .features import SET_A
from .gates import criteria
from .metrics import balanced_accuracy, batch_table, ci, point_metrics

PROTOCOL_SHA256 = "88ef6934f896d83ac2b5fa3e1f7c5a7e3ee6ad1cfe70bc97cf5a6f956a5bb047"
"""SHA-256 of the original PROTOCOL_v4.md, fixed before any candidate metric was computed (protocols/README.md)."""
REPO = Path(__file__).resolve().parents[2]
REFERENCE_METRICS = REPO / "results" / "metrics.csv"
TOL = 1e-9


def run_model(name: str, cohort: pd.DataFrame, n_jobs: int = 4, y_override=None, members=None) -> dict:
    """Nested LOSO of recipe `name` and a score for every cohort row.

    Training subjects get their out-of-fold score (`split = outer_loso`), other scorable subjects the
    ensemble mean (`outer_ensemble`), unscorable folders the fallback. For recipes with the minimum-data
    gate, gated scorable subjects are set to the fallback (`gate_fraction = 1`). `members` reuses already
    fitted fold models (C2 and C2G share them).
    """
    recipe = M.RECIPES[name]
    tr = training_frame(cohort, recipe["soma"])
    y = tr.y.to_numpy() if y_override is None else np.asarray(y_override)
    X = M.input_matrix(recipe, tr[SET_A].to_numpy(float))
    if members is None:
        members = M.nested_loso(X, y, tr.batch.to_numpy(), recipe, n_jobs=n_jobs)
    sc = cohort[cohort.scorable]
    Xs = M.input_matrix(recipe, sc[SET_A].to_numpy(float))
    gate = (sc.qc_fail | sc.too_short).to_numpy() if recipe["gate"] else None
    S, G = M.score_matrix(members, Xs, gate)
    pos = {s: i for i, s in enumerate(tr.subject_id)}
    col = {s: j for j, s in enumerate(sc.subject_id)}
    rows = []
    for r in cohort.itertuples():
        d = dict(subject_id=r.subject_id, group=r.group, batch=r.batch, scorable=r.scorable, model=name,
                 is_duplicate=bool(r.is_duplicate), min_data_gated=bool(r.min_data_fail))
        if not r.scorable:
            d.update(split="fallback_unscorable", score=M.FALLBACK, gate_fraction=0.0)
        elif r.subject_id in pos:
            i, j = pos[r.subject_id], col[r.subject_id]
            d.update(split="outer_loso", score=float(S[i, j]), gate_fraction=float(G[i, j]))
        else:
            j = col[r.subject_id]
            d.update(split="outer_ensemble", score=float(S[:, j].mean()), gate_fraction=float(G[:, j].mean()))
        rows.append(d)
    pred = pd.DataFrame(rows)
    if recipe.get("min_data_gate", False):
        g = pred.min_data_gated & pred.scorable
        pred.loc[g, "score"] = M.FALLBACK
        pred.loc[g, "gate_fraction"] = 1.0
    pred["positive"] = pred.score >= M.THRESHOLD
    sel = pd.DataFrame([dict(model=name, fold=i, **a) for i, m in enumerate(members) for a in m["inner_audit"]])
    return dict(name=name, pred=pred, members=members, train=tr, y=y, sel=sel, S=S, G=G)


def metrics_row(name: str, pred: pd.DataFrame, cohort: pd.DataFrame, idx_sets, boots: dict, pms: dict) -> dict:
    """All metrics of one model with percentile CIs and paired deltas vs v3 (and vs C2 for C2G)."""
    pm = pms[name]
    b = boots[name]
    row = dict(model=name, protocol_sha256=PROTOCOL_SHA256, n_train=int((pred.split == "outer_loso").sum()), **pm)
    row["auc_lo"], row["auc_hi"] = ci(b["auc"])
    row["fpr_soma_elderly_lo"], row["fpr_soma_elderly_hi"] = ci(b["fpr"])
    row["proxy_lo"], row["proxy_hi"] = ci(b["proxy"])
    for k in ["auc_vs_CTRL_other", "auc_vs_CTRL_B"]:
        row[f"{k}_lo"], row[f"{k}_hi"] = ci(b[k])
    if "v3" in boots:
        ref, rpm = boots["v3"], pms["v3"]
        row["delta_proxy"] = pm["proxy_points"] - rpm["proxy_points"]
        row["delta_proxy_lo"], row["delta_proxy_hi"] = ci(b["proxy"] - ref["proxy"])
        row["delta_auc"] = pm["auc"] - rpm["auc"]
        row["delta_auc_lo"], row["delta_auc_hi"] = ci(b["auc"] - ref["auc"])
        row["delta_fpr"] = pm["fpr_soma_elderly"] - rpm["fpr_soma_elderly"]
        row["delta_fpr_lo"], row["delta_fpr_hi"] = ci(b["fpr"] - ref["fpr"])
    row.update(balanced_accuracy(pred))
    row["ba_lo"], row["ba_hi"] = ci(b["ba"])
    row["fpr_soma_lo"], row["fpr_soma_hi"] = ci(b["fpr_soma"])
    row["fpr_elderly_lo"], row["fpr_elderly_hi"] = ci(b["fpr_elderly"])
    if M.RECIPES[name].get("min_data_gate", False):
        row["n_min_data_gated"] = int(pred.min_data_gated.sum())
        if "C2" in boots:
            bc = boots["C2"]
            for k, kb in [("proxy_points", "proxy"), ("auc", "auc"), ("fpr_soma_elderly", "fpr")]:
                row[f"delta_{kb}_vs_C2"] = pm[k] - pms["C2"][k]
                row[f"delta_{kb}_vs_C2_lo"], row[f"delta_{kb}_vs_C2_hi"] = ci(b[kb] - bc[kb])
    return row


def compare(reproduced: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Absolute differences of every shared numeric column, per shared model."""
    rows = []
    ref = reference.set_index("model")
    for r in reproduced.to_dict("records"):
        if r["model"] not in ref.index:
            continue
        for k, v in r.items():
            if k == "model" or k not in ref.columns:
                continue
            rv = ref.loc[r["model"], k]
            try:
                a, b = float(v), float(rv)
            except (TypeError, ValueError):
                continue
            if np.isnan(a) and np.isnan(b):
                d = 0.0
            else:
                d = abs(a - b)
            rows.append(dict(model=r["model"], column=k, reference=b, reproduced=a, abs_diff=d))
    return pd.DataFrame(rows)


def reproduce(data_root: Path, out_dir: Path, models=("v3", "C2", "C2G"), flip: bool = False, n_jobs: int = 4,
              save_predictions: bool = False, reference: Path = REFERENCE_METRICS, log=print) -> dict:
    """Retrain every requested recipe from raw data and compare with the published metrics.

    Returns a summary dict with the maximum absolute difference over all compared cells.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(["v3", *models]))
    t0 = time.time()
    log(f"building cohort from {data_root}")
    cohort = build_cohort(data_root, n_jobs=n_jobs)
    log(f"cohort: {len(cohort)} folders, {int(cohort.scorable.sum())} scorable ({time.time() - t0:.0f}s)")
    idx_sets = bootstrap_index_sets(cohort)
    results, boots, pms, flips = {}, {}, {}, {}
    for name in models:
        t1 = time.time()
        shared = None
        if M.RECIPES[name].get("min_data_gate") and "C2" in results:
            shared = results["C2"]["members"]
        results[name] = run_model(name, cohort, n_jobs=n_jobs, members=shared)
        pred = results[name]["pred"]
        boots[name] = bootstrap_arrays(pred, cohort, idx_sets)
        pms[name] = point_metrics(pred)
        log(f"{name}: AUC {pms[name]['auc']:.4f}, proxy {pms[name]['proxy_points']:.2f} ({time.time() - t1:.0f}s)")
        if flip:
            _, flips[name] = run_flip(name, cohort, n_jobs=n_jobs,
                                      min_data_gate=bool(M.RECIPES[name].get("min_data_gate")))
            log(f"{name}: flip rate {flips[name]['flip_rate']:.6f} ({flips[name]['seconds']:.0f}s)")
    rows = []
    for name in models:
        row = metrics_row(name, results[name]["pred"], cohort, idx_sets, boots, pms)
        if flip:
            f = flips[name]
            row.update(flip_rate=f["flip_rate"], flip_rate_n=f["n_subjects"],
                       flip_rate_all_scorable=f["flip_rate_all_scorable"], flip_boot_gated_share=f["boot_gated_share_mean"])
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics_reproduced.csv", index=False)
    pd.concat([batch_table(r["pred"]) for r in results.values()]).to_csv(out_dir / "batch_tables.csv", index=False)
    if flip:
        pd.DataFrame(flips.values()).to_csv(out_dir / "flip_summary.csv", index=False)
        if len(models) > 1:
            criteria(metrics).to_csv(out_dir / "criteria.csv", index=False)
    if save_predictions:
        priv = out_dir / "per_subject_DO_NOT_COMMIT"
        priv.mkdir(exist_ok=True)
        for name, r in results.items():
            r["pred"].to_csv(priv / f"{name}_predictions.csv", index=False)
    summary = dict(models=models, n_folders=len(cohort), n_scorable=int(cohort.scorable.sum()),
                   flip=flip, seconds=time.time() - t0)
    if Path(reference).is_file():
        cmp = compare(metrics, pd.read_csv(reference))
        cmp.to_csv(out_dir / "comparison.csv", index=False)
        summary.update(n_compared=len(cmp), max_abs_diff=float(cmp.abs_diff.max()) if len(cmp) else float("nan"),
                       worst=cmp.sort_values("abs_diff").tail(1)[["model", "column", "abs_diff"]].to_dict("records"),
                       passed=bool(len(cmp) and cmp.abs_diff.max() < TOL), tolerance=TOL)
    (out_dir / "reproduce_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
