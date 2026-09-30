"""Regenerate figures/*.png from the aggregate tables in results/ (and public data only).

    python tools/make_figures.py [--only name ...] [--mendeley DIR]

Every number drawn on a figure is read from `results/*.csv` or from the public weights in `weights/`, or is
computed from synthetic signals (`brain_vs_nonbrain`, `eeg_trace_synthetic`). No EEG-derived value of the
confidential cohort is drawn. Two figures need the public Mendeley PTSD dataset
(du Bois et al., Mendeley Data v3, doi:10.17632/gsxphk87mc.3, CC BY 4.0): `external_scatter` and
`eeg_trace_public`. Point `--mendeley` (or `MENDELEY_CACHE`) at a folder with `mat/*.mat` (the EEG files)
and `workbooks/psych-data-scores.xlsx`; without it those two figures are skipped and the committed PNGs
are kept. Participant codes of the public dataset are never drawn.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "examples"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
RES = REPO / "results"
FIG = REPO / "figures"
warnings.simplefilter("ignore")

# Okabe-Ito colour-blind-safe palette
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#8C8C8C")
BATCHES = {
    "PTSD": ("PTSD", OI["vermillion"]),
    "CTRL_other": ("Original controls, valid", OI["blue"]),
    "CTRL_B": ("Control batch B", OI["orange"]),
    "SOMA": ("Somatoform", OI["purple"]),
    "CTRL_C": ("Control set C", OI["sky"]),
    "CTRL_elderly": ("Controls 65+", OI["green"]),
}
MODEL_COLOUR = {"v3": OI["grey"], "C2G": OI["blue"]}
MENDELEY_CREDIT = ("Data: du Bois et al., Mendeley Data v3, doi:10.17632/gsxphk87mc.3 (CC BY 4.0)")

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 150, "axes.grid": False,
})


# ------------------------------------------------------------------ helpers
def source_note(fig, text: str) -> None:
    fig.text(0.995, 0.008, text, ha="right", va="bottom", fontsize=9, color="0.4")


def save(fig, name: str) -> Path:
    FIG.mkdir(exist_ok=True)
    out = FIG / f"{name}.png"
    fig.savefig(out, dpi=150, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def wilson(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval of a proportion k/n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def metrics() -> pd.DataFrame:
    return pd.read_csv(RES / "metrics.csv").set_index("model")


# ------------------------------------------------------------------ synthetic illustration
def _nonbrain_signals(rng: np.random.Generator, n: int) -> np.ndarray:
    """Six independent channels of weakly coloured noise: no shared field, no alpha rhythm (µV)."""
    from make_synthetic_edf import _pink
    return np.vstack([6 * _pink(rng, n, exponent=0.4) for _ in range(6)])


def _rel_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (2 s windows) of the O1/O2 mean, divided by its 1-40 Hz power."""
    from scipy.signal import welch

    from ptsd_eeg.features import integrate
    from ptsd_eeg.io import CHANNELS
    f, pxx = welch(x[[CHANNELS.index("O1"), CHANNELS.index("O2")]], fs=fs, nperseg=int(2 * fs))
    p = pxx.mean(0)
    band = (f >= 1) & (f <= 40)
    return f[band], p[band] / integrate(p[band], f[band])


def fig_brain_vs_nonbrain() -> None:
    from make_synthetic_edf import FS, make_signals

    from ptsd_eeg.io import CHANNELS
    from ptsd_eeg.qc import QC_FP1_FP2_R_MIN, QC_MEDIAN_ABS_R_MIN, coupling_qc
    rng = np.random.default_rng(7)
    n = int(60 * FS)
    sig = {"brain": make_signals(rng, "high_alpha", 60.0), "nonbrain": _nonbrain_signals(rng, n)}
    style = {"brain": ("Brain-like EEG (synthetic)", OI["blue"]),
             "nonbrain": ("Non-brain signal (synthetic)", OI["orange"])}
    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.18)
    show = ["Fp1", "Fp2", "O1", "O2"]
    a, b = int(20 * FS), int(24 * FS)
    t = np.arange(a, b) / FS
    for row, key in enumerate(["brain", "nonbrain"]):
        ax = fig.add_subplot(gs[row, 0])
        lab, col = style[key]
        seg = sig[key][[CHANNELS.index(c) for c in show], a:b]
        _trace_panel(ax, t, seg, show, [col] * 4, spacing=60)
        q = coupling_qc(sig[key], FS)
        ok = q["median_abs_r"] >= QC_MEDIAN_ABS_R_MIN and q["fp1fp2_r"] >= QC_FP1_FP2_R_MIN
        ax.set_title(f"{lab}: median |r| {q['median_abs_r']:.2f}, Fp1–Fp2 r {q['fp1fp2_r']:.2f} "
                     f"→ QC {'pass' if ok else 'fail'}", fontsize=13, color=col)
        if row == 0:
            ax.set_xlabel("")
    bx = fig.add_subplot(gs[:, 1])
    bx.axvspan(8, 13, color="0.90", zorder=0)
    for key in ["brain", "nonbrain"]:
        f, p = _rel_psd(sig[key], FS)
        lab, col = style[key]
        bx.plot(f, p, color=col, lw=2.4, label=lab)
    bx.set_yscale("log")
    bx.set_xlim(1, 40)
    bx.set_xlabel("Frequency (Hz)")
    bx.set_ylabel("Relative PSD, O1/O2 mean (1/Hz)")
    bx.text(10.5, bx.get_ylim()[1] * 0.7, "alpha\n8–13 Hz", ha="center", va="top", fontsize=11, color="0.3")
    bx.set_title("Spectrum: alpha peak vs none")
    bx.legend(loc="lower left")
    fig.suptitle("SYNTHETIC ILLUSTRATION: what a brain signal and a non-brain signal look like to the pipeline",
                 fontsize=15, fontweight="bold", color=OI["vermillion"])
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.1)
    source_note(fig, "Synthetic signals (examples/make_synthetic_edf.py model, seed 7), not study data; "
                     "QC thresholds from ptsd_eeg.qc")
    save(fig, "brain_vs_nonbrain")


# ------------------------------------------------------------------ ROC and metrics
def fig_roc_ci() -> None:
    roc = pd.read_csv(RES / "roc_curves.csv")
    m = metrics().loc["C2G"]
    spec = [("all_controls", "PTSD vs all training controls", OI["black"], "auc"),
            ("CTRL_B", "PTSD vs control batch B", OI["orange"], "auc_vs_CTRL_B"),
            ("CTRL_other", "PTSD vs valid original controls", OI["blue"], "auc_vs_CTRL_other")]
    fig, ax = plt.subplots(figsize=(8.6, 8))
    ax.plot([0, 1], [0, 1], ls=":", color="0.5", lw=1.2, label="Chance (AUC 0.5)")
    for comp, lab, col, key in spec:
        d = roc[(roc.model == "C2G") & (roc.comparison == comp)].sort_values("fpr")
        ax.fill_between(d.fpr, d.tpr_lo, d.tpr_hi, color=col, alpha=0.15, lw=0, step="post")
        ax.step(d.fpr, d.tpr, where="post", color=col, lw=2.4,
                label=f"{lab} (n = {int(d.n_ptsd.iloc[0])} vs {int(d.n_control.iloc[0])})\n"
                      f"AUC {m[key]:.3f} [{m[key + '_lo']:.3f}, {m[key + '_hi']:.3f}]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False-positive rate (1 − specificity)")
    ax.set_ylabel("True-positive rate (sensitivity)")
    ax.set_title("Shipped model (C2G): out-of-fold ROC, 95% bootstrap band", fontsize=14)
    ax.legend(loc="lower right", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    source_note(fig, "Source: results/roc_curves.csv, results/metrics.csv (nested LOSO, 90 training subjects)")
    save(fig, "roc_ci")


def _bar_pair(ax, x, vals, los, his, labels, width=0.36):
    for i, (model, col) in enumerate(MODEL_COLOUR.items()):
        xs = x + (i - 0.5) * width
        v = np.array(vals[model], float)
        lo = np.array(los[model], float)
        hi = np.array(his[model], float)
        has = np.isfinite(lo)
        ax.bar(xs, v, width, color=col, label=model if model == "v3" else "C2G (shipped)", zorder=2)
        ax.errorbar(xs[has], v[has], yerr=[v[has] - lo[has], hi[has] - v[has]], fmt="none", ecolor="k",
                    capsize=4, lw=1.3, zorder=3)
        for xx, vv, tt, hh in zip(xs, v, labels[model], np.where(has, hi, v)):
            ax.text(xx, hh + 0.015 * ax.get_ylim()[1], tt, ha="center", va="bottom", fontsize=10.5)


def fig_v3_vs_c2g() -> None:
    m = metrics()
    fl = pd.read_csv(RES / "flip_summary.csv").set_index("model")
    npt = int(m.loc["C2G", "n_ptsd"])
    nse = int(m.loc["C2G", "n_soma_elderly"])
    vals, los, his, labs = {}, {}, {}, {}
    for model in ("v3", "C2G"):
        r = m.loc[model]
        k_sens = round(r.sensitivity * npt)
        k_fpr = int(r.n_soma_elderly_positive)
        s_lo, s_hi = wilson(k_sens, npt)
        flip = r.flip_rate if np.isfinite(r.flip_rate) else fl.loc[model, "flip_rate"]
        vals[model] = [r.auc, r.auc_vs_CTRL_other, r.sensitivity, r.fpr_soma_elderly, flip]
        los[model] = [r.auc_lo, r.auc_vs_CTRL_other_lo, s_lo, r.fpr_soma_elderly_lo, np.nan]
        his[model] = [r.auc_hi, r.auc_vs_CTRL_other_hi, s_hi, r.fpr_soma_elderly_hi, np.nan]
        labs[model] = [f"{r.auc:.3f}", f"{r.auc_vs_CTRL_other:.3f}", f"{k_sens}/{npt}", f"{k_fpr}/{nse}",
                       f"{flip:.3f}"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw=dict(width_ratios=[4.2, 1.2]))
    ax.set_ylim(0, 1.08)
    x = np.arange(5)
    _bar_pair(ax, x, vals, los, his, labs)
    ax.plot([-0.5, 1.5], [0.5, 0.5], color="0.5", ls=":", lw=1.2)
    ax.text(1.5, 0.5, " chance", va="center", fontsize=10, color="0.4")
    ax.set_xticks(x, ["AUC, all\ncontrols ↑", "AUC, valid\ncontrols ↑", "Sensitivity\nat 0.5 ↑",
                      "FPR, somatoform\n+ 65+ ↓", "Flip rate ↓"])
    ax.set_ylabel("Value (proportion or AUC)")
    ax.set_title("v3 vs shipped model: metrics on a 0–1 scale")
    ax.legend(loc="upper right")
    bx.set_ylim(0, 50)
    _bar_pair(bx, np.array([0]), {k: [m.loc[k, "proxy_points"]] for k in MODEL_COLOUR},
              {k: [m.loc[k, "proxy_lo"]] for k in MODEL_COLOUR}, {k: [m.loc[k, "proxy_hi"]] for k in MODEL_COLOUR},
              {k: [f"{m.loc[k, 'proxy_points']:.2f}"] for k in MODEL_COLOUR})
    bx.set_xlim(-0.75, 0.75)
    bx.set_xticks([0], ["Proxy points ↑"])
    bx.set_ylabel("Points (max 50)")
    bx.set_title("Decision score")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    source_note(fig, "Source: results/metrics.csv, results/flip_summary.csv. Whiskers: 95% bootstrap CI "
                     "(sensitivity: Wilson). ↑/↓ = better direction")
    save(fig, "v3_vs_c2g")


def fig_decision_rule() -> None:
    cr = pd.read_csv(RES / "criteria.csv").set_index("model")
    # Gate outcomes (G1-G4) as reported in docs/model_selection.md; gates were only run for candidates
    # that passed criteria 1 and 2.
    gates = {"C1": "fail (G1)", "C2": "pass", "C3": "not run", "C4": "not run"}
    models = ["C1", "C2", "C3", "C4"]
    fig, (ax, tb) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw=dict(width_ratios=[2.6, 1.2]))
    y = np.arange(len(models))[::-1]
    for yy, mdl in zip(y, models):
        r = cr.loc[mdl]
        full = bool(r.criterion1) and bool(r.criterion2) and gates[mdl] == "pass"
        col = OI["blue"] if full else (OI["grey"] if r.criterion1 else OI["vermillion"])
        if full:
            ax.axhspan(yy - 0.42, yy + 0.42, color="#e6f0f8", zorder=0)
        ax.errorbar(r.delta_proxy, yy, xerr=[[r.delta_proxy - r.delta_proxy_p2_5], [r.delta_proxy_p97_5 - r.delta_proxy]],
                    fmt="o", color=col, ms=10, capsize=6, lw=2.2, zorder=3)
        ax.text(r.delta_proxy, yy + 0.2, f"{r.delta_proxy:+.2f} [{r.delta_proxy_p2_5:+.2f}, {r.delta_proxy_p97_5:+.2f}]",
                ha="center", va="bottom", fontsize=10.5)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_yticks(y, [f"{m_}{'  (shipped)' if m_ == 'C2' else ''}" for m_ in models])
    ax.set_xlabel("Δ proxy points vs v3 (paired bootstrap, 95% CI)")
    ax.set_xlim(-19, 17)
    ax.set_ylim(-0.6, len(models) - 0.1)
    ax.set_title("Paired Δ proxy points vs v3")

    def mark(ok: bool | None) -> tuple[str, str]:
        if ok is None:
            return "–", "0.5"
        return ("✓", OI["green"]) if ok else ("✗", OI["vermillion"])

    cols = [("Crit. 1\nΔ CI > 0", lambda r, g: bool(r.criterion1)),
            ("Crit. 2\nflip ≤ limit", lambda r, g: bool(r.criterion2)),
            ("Gates\nG1–G4", lambda r, g: None if g == "not run" else g == "pass")]
    tb.set_xlim(0, 3)
    tb.axis("off")
    tb.set_title("Pre-registered criteria")
    for xpos, (head, fn) in zip([0.5, 1.5, 2.5], cols):
        tb.text(xpos, len(models) - 0.45, head, ha="center", va="center", fontsize=12, fontweight="bold")
        for yy, mdl in zip(y, models):
            s, c = mark(fn(cr.loc[mdl], gates[mdl]))
            tb.text(xpos, yy, s, ha="center", va="center", fontsize=24, color=c)
    tb.text(2.5, y[0] - 0.3, "permuted AUC below\nthe null range", ha="center", va="top", fontsize=9.5, color="0.35")
    fig.suptitle("Model selection: only C2 passed every pre-registered check", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    source_note(fig, "Source: results/criteria.csv (Δ, CI, criteria 1–2); gate outcomes: docs/model_selection.md")
    save(fig, "decision_rule")


def fig_per_batch() -> None:
    bt = pd.read_csv(RES / "C2G_batch_table.csv").set_index("batch")
    m = metrics().loc["C2G"]
    order = ["PTSD", "CTRL_other", "CTRL_B", "SOMA", "CTRL_C", "CTRL_elderly"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    y = np.arange(len(order))[::-1]
    for yy, b in zip(y, order):
        r = bt.loc[b]
        k, n = int(r.n_positive), int(r.n_scorable)
        lo, hi = wilson(k, n)
        lab, col = BATCHES[b]
        ax.barh(yy, k / n, color=col, height=0.62, zorder=2)
        ax.errorbar(k / n, yy, xerr=[[k / n - lo], [hi - k / n]], fmt="none", ecolor="k", capsize=4, lw=1.3, zorder=3)
        ax.text(hi + 0.02, yy, f"{k}/{n}", va="center", fontsize=11)
    ax.set_yticks(y, [BATCHES[b][0] + ("\n(= sensitivity)" if b == "PTSD" else "") for b in order])
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Share scored ≥ 0.5 (scorable folders)")
    ax.set_title("Positive rate per batch at threshold 0.5")

    comps = [("vs all training controls", "auc", OI["black"], int(m.n_control)),
             ("vs control batch B", "auc_vs_CTRL_B", OI["orange"], int(bt.loc["CTRL_B", "n_scorable"])),
             ("vs valid original controls", "auc_vs_CTRL_other", OI["blue"], int(bt.loc["CTRL_other", "n_scorable"]))]
    y2 = np.arange(len(comps))[::-1]
    for yy, (lab, key, col, n) in zip(y2, comps):
        v, lo, hi = m[key], m[key + "_lo"], m[key + "_hi"]
        bx.errorbar(v, yy, xerr=[[v - lo], [hi - v]], fmt="o", color=col, ms=11, capsize=6, lw=2.2)
        bx.text(v, yy + 0.18, f"{v:.3f} [{lo:.3f}, {hi:.3f}]", ha="center", va="bottom", fontsize=11)
    bx.axvline(0.5, color="0.5", ls=":", lw=1.2)
    bx.text(0.505, -0.55, "chance", color="0.4", fontsize=10)
    bx.set_yticks(y2, [f"PTSD ({int(m.n_ptsd)})\n{c[0]} ({c[3]})" for c in comps])
    bx.set_xlim(0.2, 1.02)
    bx.set_ylim(-0.6, len(comps) - 0.4)
    bx.set_xlabel("AUC (out of fold, 95% bootstrap CI)")
    bx.set_title("Discrimination depends on the control batch")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    source_note(fig, "Source: results/C2G_batch_table.csv (Wilson 95% CI), results/metrics.csv")
    save(fig, "per_batch_auc_fpr")


def fig_coefficient_spread() -> None:
    from ptsd_eeg.transforms import C1_FEATURES
    from ptsd_eeg.weights import load_weights
    model = load_weights(REPO / "weights")
    coef = np.stack([np.asarray(mb["coef"], float) for mb in model["members"]]).reshape(-1, len(C1_FEATURES))
    ref = pd.read_csv(RES / "c2g_coefficients.csv").set_index("feature")
    for j, f in enumerate(C1_FEATURES):
        assert abs(coef[:, j].mean() - ref.loc[f, "mean"]) < 1e-12, f
    order = ["clr_alpha_O", "logit_alpha_IAF", "rest_IAF", "clr_beta_O"]
    data = [coef[:, C1_FEATURES.index(f)] for f in order]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    parts = ax.violinplot(data, positions=range(4), widths=0.8, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(OI["sky"])
        pc.set_edgecolor(OI["blue"])
        pc.set_alpha(0.55)
    ax.boxplot(data, positions=range(4), widths=0.16, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="k"), medianprops=dict(color=OI["vermillion"], lw=2))
    ax.axhline(0, color="k", lw=1)
    top = float(np.concatenate(data).max())
    ax.set_ylim(float(np.concatenate(data).min()) - 0.2, top + 1.3)
    for i, f in enumerate(order):
        r = ref.loc[f]
        ax.text(i, top + 1.2, f"mean {r['mean']:+.2f}\n{100 * r.share_positive:.1f}% > 0", ha="center",
                va="top", fontsize=11.5, bbox=dict(fc="white", ec="0.75", boxstyle="round,pad=0.3"))
    ax.set_xticks(range(4), order)
    ax.set_ylabel("Standardized LDA coefficient\n(per robust-scaled unit; + raises the score)")
    ax.set_title(f"Ensemble weights: {coef.shape[0]} fits ({int(ref.n_members.iloc[0])} LOSO members × "
                 f"{int(ref.n_bags.iloc[0])} bags)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    source_note(fig, "Source: weights/c2g_members.npz (checked against results/c2g_coefficients.csv); "
                     "boxes: IQR, whiskers 1.5 IQR")
    save(fig, "coefficient_spread")


N_ELDERLY = 23
"""Denominator of `fpr_elderly` in results/metrics.csv (healthy 65+ folders, fallbacks counted as negatives)."""


def fig_task_duration() -> None:
    m = metrics()
    # model ids in results/ keep their internal names
    models = [("C2G", "C2G (shipped)", OI["blue"]),
              ("C2_plus_schulte", "C2 + task duration\n(not shipped)", OI["orange"]),
              ("schulte_duration_only", "Task duration\nonly (not shipped)", OI["purple"])]
    n_eld = N_ELDERLY
    for k, _, _ in models:
        v = float(m.loc[k, "fpr_elderly"]) * n_eld
        assert abs(v - round(v)) < 1e-9, (k, v)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.2))
    for i, (k, lab, col) in enumerate(models):
        v = m.loc[k, "fpr_elderly"]
        kk = round(v * n_eld)
        lo, hi = wilson(kk, n_eld)
        ax.bar(i, v, color=col, width=0.6, zorder=2)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="k", capsize=5, lw=1.3, zorder=3)
        ax.text(i, hi + 0.02, f"{v:.3f}\n({kk}/{n_eld})", ha="center", va="bottom", fontsize=11)
        a, alo, ahi = m.loc[k, "auc_vs_CTRL_other"], m.loc[k, "auc_vs_CTRL_other_lo"], m.loc[k, "auc_vs_CTRL_other_hi"]
        bx.bar(i, a, color=col, width=0.6, zorder=2)
        bx.errorbar(i, a, yerr=[[a - alo], [ahi - a]], fmt="none", ecolor="k", capsize=5, lw=1.3, zorder=3)
        bx.text(i, ahi + 0.02, f"{a:.3f}", ha="center", va="bottom", fontsize=11)
    for a_ in (ax, bx):
        a_.set_xticks(range(3), [x[1] for x in models])
        a_.set_ylim(0, 1.12)
    ax.set_ylabel("False-positive rate, healthy 65+")
    ax.set_title("Healthy controls 65+ flagged at 0.5")
    bx.axhline(0.5, color="0.5", ls=":", lw=1.2)
    bx.set_ylabel("AUC, PTSD vs valid original controls")
    bx.set_title("Apparent gain vs valid controls")
    fig.suptitle("Adding the task-duration feature flags most healthy 65+ controls (age confound)", fontsize=16,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    source_note(fig, "Source: results/metrics.csv (FPR: Wilson 95% CI; AUC: 95% bootstrap CI)")
    save(fig, "task_duration_age_confound")


# ------------------------------------------------------------------ diagrams
def _box(ax, x, y, w, h, text, fc, ec="k", fs=12, bold=False, round_=0.02):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0.01,rounding_size={round_}",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, a, b, text=None, col="k", rad=0.0, fs=10.5, offset=(0, 0)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, lw=1.6, color=col,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, fontsize=fs, color=col,
                ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1))


def fig_loso_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 8.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.text(7, 7.85, "Nested leave-one-subject-out (LOSO) validation", ha="center", fontsize=17, fontweight="bold")
    _box(ax, 7, 7.05, 7.4, 0.7, "90 scorable training subjects: 24 PTSD, 66 controls", "#f2f2f2", bold=True)
    outer = FancyBboxPatch((0.4, 1.7), 9.0, 4.75, boxstyle="round,pad=0.02,rounding_size=0.15", fc="none",
                           ec=OI["blue"], lw=2.2, ls="--")
    ax.add_patch(outer)
    ax.text(0.6, 6.2, "Outer loop: i = 1 … 90 (one fold per subject)", fontsize=13, color=OI["blue"], fontweight="bold")
    _box(ax, 2.2, 5.35, 3.2, 0.9, "Hold out subject i\n(never seen by its member)", "#fbe3d6")
    _box(ax, 6.6, 5.35, 4.6, 0.9, "Training fold: the other 89 subjects", "#dceaf7")
    inner = FancyBboxPatch((4.45, 2.0), 4.3, 2.75, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f7fbfe",
                           ec=OI["sky"], lw=1.8)
    ax.add_patch(inner)
    ax.text(6.6, 4.5, "Inside the training fold only", ha="center", fontsize=11.5, color="0.3", style="italic")
    _box(ax, 6.6, 3.85, 3.9, 0.7, "Inner loop: stratified 5-fold CV\npicks the LDA shrinkage", "white", fs=11.5)
    _box(ax, 6.6, 2.75, 3.9, 0.8, "B = 100 stratum-balanced bags:\nimpute, winsorise 1/99, robust scale, LDA", "white",
         fs=11.5)
    _arrow(ax, (6.6, 4.9), (6.6, 4.22))
    _arrow(ax, (6.6, 3.5), (6.6, 3.17))
    _box(ax, 2.2, 2.75, 2.6, 0.9, "Out-of-fold score\nfor subject i", "#fbe3d6", bold=True)
    _arrow(ax, (2.2, 4.9), (2.2, 3.2))
    _arrow(ax, (4.65, 2.6), (3.5, 2.6))
    ax.text(4.07, 2.85, "member i", ha="center", va="center", fontsize=10.5)
    _box(ax, 11.7, 2.75, 4.1, 0.9, "Fold member i\n(100 bagged LDAs)", "#e3f2ec")
    _arrow(ax, (8.55, 2.75), (9.65, 2.75))
    _box(ax, 2.2, 0.85, 3.6, 0.9, "Metrics from the 90 OOF scores:\nAUC, FPR, proxy points", "#fff6d6")
    _arrow(ax, (2.2, 2.3), (2.2, 1.3))
    _box(ax, 11.7, 4.35, 4.1, 1.0, "Deployed model:\nmean of the 90 members\n(no refit on all 90)", "#e3f2ec", bold=True)
    _arrow(ax, (11.7, 3.2), (11.7, 3.85))
    _box(ax, 11.7, 5.9, 4.1, 0.95, "Scores for new and external subjects\n(somatoform, control set C, 65+)", "#f2f2f2")
    _arrow(ax, (11.7, 4.85), (11.7, 5.42))
    _arrow(ax, (7, 6.7), (7, 6.45))
    fig.tight_layout()
    source_note(fig, "Same scheme as the mermaid diagram in docs/science.md; files of one subject never cross folds")
    save(fig, "loso_diagram")


def fig_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.text(7.5, 6.0, "From a raw EDF file to a score", ha="center", fontsize=17, fontweight="bold")
    top = [(1.5, "Rest EDF\n6 channels, ~125 Hz\n(eyes closed)", "#f2f2f2"),
           (4.6, "Minimum-data gate\n≥ 30 s, not truncated,\nreadable", "#fff6d6"),
           (7.8, "Clean + epoch\nmedian centring, cond.\nnotch / 1–40 Hz band-pass\n2 s epochs, robust-z reject",
            "#dceaf7"),
           (11.1, "Welch PSD\nmean over epochs,\nthen O1 and O2", "#dceaf7"),
           (13.9, "Set A (4 features)\nIAF, rel. alpha O,\nrel. alpha IAF,\nrel. beta O", "#dceaf7")]
    for x, t, c in top:
        _box(ax, x, 4.35, 2.55 if x < 13 else 2.1, 1.6, t, c, fs=11.5)
    for (x0, *_), (x1, *_) in zip(top[:-1], top[1:]):
        _arrow(ax, (x0 + 1.3, 4.35), (x1 - 1.3 if x1 < 13 else x1 - 1.08, 4.35))
    bottom = [(13.9, "C1 transforms\nCLR of band powers,\nlogit of IAF-band alpha", "#e3f2ec"),
              (10.2, "90 LOSO members × 100 bags\nimpute, winsorise 1/99,\nrobust scale, shrinkage LDA", "#e3f2ec"),
              (6.5, "Average\nmean over bags, clip\n[0.02, 0.98], mean over\n90 members", "#e3f2ec"),
              (3.0, "Score in 0–1\ndecision at 0.5\n(research output,\nnot a diagnosis)", "#fbe3d6")]
    for x, t, c in bottom:
        _box(ax, x, 1.55, {10.2: 2.9, 13.9: 2.1}.get(x, 2.5), 1.6, t, c, fs=11.5, bold=(x == 3.0))
    _arrow(ax, (13.9, 3.55), (13.9, 2.35))
    _arrow(ax, (12.85, 1.55), (11.65, 1.55))
    _arrow(ax, (8.75, 1.55), (7.75, 1.55))
    _arrow(ax, (5.25, 1.55), (4.25, 1.55))
    _box(ax, 4.6, 2.75, 2.3, 0.6, "Fallback 24/90 = 0.267", "white", ec=OI["vermillion"], fs=11)
    _arrow(ax, (4.6, 3.55), (4.6, 3.05), "fail", col=OI["vermillion"], offset=(0.35, 0))
    _arrow(ax, (3.45, 2.6), (3.0, 2.35), col=OI["vermillion"])
    ax.text(12.35, 3.0, "no usable spectrum\n→ fallback", fontsize=10, color=OI["vermillion"], ha="center")
    fig.tight_layout()
    source_note(fig, "Code: io.load_edf_with_header, qc.min_data_flags, features.extract_array, "
                     "transforms, model.member_proba (see docs/pipeline.md)")
    save(fig, "pipeline_overview")


# ------------------------------------------------------------------ EEG traces
def _trace_panel(ax, t, x, labels, colours, spacing=None):
    spacing = spacing or 4 * np.median([np.percentile(np.abs(r - np.median(r)), 95) for r in x])
    for i, (row, lab) in enumerate(zip(x, labels)):
        off = -i * spacing
        ax.plot(t, row - np.median(row) + off, color=colours[i], lw=0.8)
    ax.set_yticks([-i * spacing for i in range(len(labels))], labels)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return spacing


def _scale_bar(ax, t, spacing, n_rows, uv):
    x0 = t[-1] + 0.12 * (t[-1] - t[0]) / 10
    y0 = -(n_rows - 1) * spacing
    ax.plot([x0, x0], [y0, y0 + uv], color="k", lw=2.5, clip_on=False)
    ax.text(x0 + 0.08, y0 + uv / 2, f"{uv:g} µV", va="center", fontsize=11, clip_on=False)


def fig_eeg_trace_synthetic() -> None:
    from make_synthetic_edf import REST_NAME, write_subject

    from ptsd_eeg.io import CHANNELS, load_edf_with_header
    with tempfile.TemporaryDirectory() as tmp:
        folder = write_subject(Path(tmp) / "SYN-TRACE", np.random.default_rng(0), "high_alpha")
        x, fs, _ = load_edf_with_header(folder / REST_NAME)
    a, b = int(20 * fs), int(30 * fs)
    t = np.arange(a, b) / fs
    seg = x[:, a:b]
    col = [OI["blue"] if c.startswith("O") else OI["vermillion"] if c.startswith("Fp") else OI["black"] for c in CHANNELS]
    fig, ax = plt.subplots(figsize=(14, 7))
    sp = _trace_panel(ax, t, seg, CHANNELS, col)
    _scale_bar(ax, t, sp, len(CHANNELS), 50)
    ax.set_title("SYNTHETIC EEG: 10 s of a generated 'high-alpha' subject (6 channels, 125 Hz)",
                 color=OI["vermillion"])
    fig.tight_layout(rect=(0, 0.03, 0.96, 1))
    source_note(fig, "Source: examples/make_synthetic_edf.py (seed 0; pink noise + occipital alpha); "
                     "not real EEG, no clinical meaning")
    save(fig, "eeg_trace_synthetic")


# ------------------------------------------------------------------ public Mendeley data
GTEC = ["FP1", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
        "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "PZ", "P4", "P8", "PO7", "PO3", "PO4", "PO8", "OZ"]
FLEX = ["FC3", "FCZ", "FC4", "P3", "O1", "P7", "OZ", "PZ"]
EXT_MAP = {"gtec": ["PO3", "T7", "FP1", "FP2", "T8", "PO4"], "flex": ["O1", "O1", "O1", "OZ", "OZ", "OZ"]}
"""Device channels standing in for O1, T3, Fp1, Fp2, T4, O2 (protocol amendment of the external check)."""
SESSION1_REST = re.compile(r"_ss1__(?:EEG_)?relax1?\.mat$")


def _load_mat(path: Path):
    import h5py
    with h5py.File(path, "r") as h:
        key = [k for k in h.keys() if not k.startswith("#")][0]
        a = np.array(h[key], dtype=float)
    names = GTEC if a.shape[1] == 33 else FLEX if a.shape[1] == 9 else None
    if names is None:
        raise ValueError(f"unexpected matrix width {a.shape[1]}")
    return a[:, :-1].T, names, ("gtec" if a.shape[1] == 33 else "flex")


def _emulate(x: np.ndarray) -> np.ndarray:
    """Device emulation of the external check: 250 -> 125 Hz, 2-40 Hz band-pass."""
    from scipy.signal import butter, resample_poly, sosfiltfilt
    y = resample_poly(x, 1, 2, axis=-1)
    return sosfiltfilt(butter(2, [2.0, 40.0], btype="bandpass", fs=125.0, output="sos"), y, axis=-1)


def _participant_number(path: Path) -> int:
    return int(re.search(r"_sb(\d+)_", path.name).group(1))


def _primary_recordings(cache: Path) -> list[tuple[Path, np.ndarray, list[str], str]]:
    """Session-1 pre-training rest files; recordings whose signal matrix occurs twice are dropped."""
    recs = []
    for path in sorted(cache.glob("mat/*.mat"), key=_participant_number):
        if SESSION1_REST.search(path.name):
            raw, names, device = _load_mat(path)
            recs.append((path, raw, names, device, hashlib.sha256(raw.tobytes()).hexdigest()))
    counts = pd.Series([r[-1] for r in recs]).value_counts()
    return [r[:4] for r in recs if counts[r[-1]] == 1]


def _pcl5(cache: Path) -> dict[int, float]:
    wb = cache / "workbooks" / "psych-data-scores.xlsx"
    raw = pd.read_excel(wb, sheet_name="raw-scores", header=None)
    head = raw.iloc[1]
    c0 = [j for j in range(raw.shape[1]) if str(head[j]).startswith("PTSD Diagnostic")][0]
    items = {}
    for pid, it in zip(raw.iloc[3:, 0], raw.iloc[3:, c0:c0 + 20].itertuples(index=False)):
        if isinstance(pid, str) and pid.startswith("P"):
            v = pd.to_numeric(pd.Series(it), errors="coerce")
            items[int(pid[1:])] = float(v.sum()) if v.notna().all() else np.nan
    group = {}
    for g, lo in (("control", 3), ("MI", 16), ("NF", 30)):
        for i in range(lo, lo + 10):
            p = raw.iloc[i, 0]
            if isinstance(p, str) and p.startswith("P"):
                group[int(p[1:])] = g
    sheet = pd.read_excel(wb, sheet_name="PCL-5", header=None)
    col = pd.to_numeric(sheet.iloc[:, 3], errors="coerce")
    tot = {}
    for g, start, n in (("control", 1, 9), ("MI", 14, 10), ("NF", 28, 10)):
        ids = [k for k in items if group.get(k) == g]
        tot.update(dict(zip(ids, col.iloc[start:start + n].to_numpy())))
    return {k: (tot[k] if k in tot and np.isfinite(tot[k]) else items[k]) for k in items}


def _within_scores(v: np.ndarray, strata: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    out = np.empty(len(v))
    for s in np.unique(strata):
        mk = strata == s
        n = mk.sum()
        out[mk] = (rankdata(v[mk]) - (n + 1) / 2) / ((n - 1) / 2) if n > 1 else 0.0
    return out


def fig_external_scatter(cache: Path) -> None:
    from ptsd_eeg.features import extract_array
    pcl = _pcl5(cache)
    rows = []
    for path, raw, names, device in _primary_recordings(cache):
        x = _emulate(np.vstack([raw[names.index(c)] for c in EXT_MAP[device]]))
        f, _, _ = extract_array(x, 125.0)
        score = pcl.get(_participant_number(path), np.nan)
        if f["scorable"] and np.isfinite(score):
            rows.append(dict(device=device, alpha=f["rest_rel_alpha_O"], pcl5=score))
    d = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    res = pd.read_csv(RES / "external_results.csv")
    prim = res[(res.feature == "rest_rel_alpha_O") & (res.stat == "device-stratified Spearman")
               & res.analysis.str.startswith("primary")].iloc[0]
    a = _within_scores(d.alpha.to_numpy(), d.device.to_numpy())
    b = _within_scores(d.pcl5.to_numpy(), d.device.to_numpy())
    rho = float(np.corrcoef(a, b)[0, 1])
    assert len(d) == int(prim.n) and abs(rho - prim.rho) < 1e-9, (len(d), rho, prim.rho)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.4))
    style = {"gtec": ("o", OI["blue"], "g.tec 32-ch (PO3/PO4)"), "flex": ("s", OI["orange"], "FlexEEG 8-ch (O1/Oz)")}
    for dev, (mk, col, lab) in style.items():
        s = d.device == dev
        ax.scatter(d.pcl5[s], d.alpha[s], marker=mk, s=90, color=col, edgecolor="k", lw=0.7,
                   label=f"{lab}, n = {int(s.sum())}", zorder=3)
        bx.scatter(b[s], a[s], marker=mk, s=90, color=col, edgecolor="k", lw=0.7, zorder=3)
    ax.set_xlabel("PCL-5 total, pre-intervention (points)")
    ax.set_ylabel("Occipital relative alpha, 8–13 Hz\n(fraction of 1–40 Hz power)")
    ax.set_title("Raw values, eyes-open rest")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.035),
               fontsize=12)
    bx.axhline(0, color="0.8", lw=0.8)
    bx.axvline(0, color="0.8", lw=0.8)
    bx.set_xlabel("PCL-5, within-device rank score")
    bx.set_ylabel("Relative alpha, within-device rank score")
    bx.set_title("Device-stratified ranks")
    bx.text(0.03, 0.97, f"ρ = {prim.rho:+.2f} [{prim.ci_lo:+.2f}, {prim.ci_hi:+.2f}]\none-sided p = {prim.p:.2f}",
            transform=bx.transAxes, ha="left", va="top", fontsize=13,
            bbox=dict(fc="white", ec="0.7", boxstyle="round,pad=0.4"))
    fig.suptitle(f"External direction check on public data: relative alpha vs PTSD severity (n = {len(d)})",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    source_note(fig, MENDELEY_CREDIT + "; statistics: results/external_results.csv; points unlabeled")
    save(fig, "external_scatter")


def fig_eeg_trace_public(cache: Path) -> None:
    recs = [r for r in _primary_recordings(cache) if r[3] == "gtec"]
    _, raw, names, _ = recs[0]
    chans = ["FP1", "FP2", "T7", "T8", "PO3", "PO4"]
    x = _emulate(np.vstack([raw[names.index(c)] for c in chans]))
    fs = 125.0
    a, b = int(60 * fs), int(70 * fs)
    t = np.arange(b - a) / fs
    col = [OI["vermillion"]] * 2 + [OI["black"]] * 2 + [OI["blue"]] * 2
    fig, ax = plt.subplots(figsize=(14, 7))
    labels = ["Fp1", "Fp2", "T7", "T8", "PO3", "PO4"]
    sp = _trace_panel(ax, t, x[:, a:b], labels, col)
    _scale_bar(ax, t, sp, len(chans), 20)
    ax.set_title("Public PTSD EEG: 10 s of eyes-open rest (g.tec, 6 of 32 channels)")
    fig.tight_layout(rect=(0, 0.03, 0.96, 1))
    source_note(fig, MENDELEY_CREDIT + "; resampled to 125 Hz, band-pass 2–40 Hz")
    save(fig, "eeg_trace_public")


# ------------------------------------------------------------------ main
FIGURES = {
    "brain_vs_nonbrain": fig_brain_vs_nonbrain,
    "roc_ci": fig_roc_ci,
    "v3_vs_c2g": fig_v3_vs_c2g,
    "decision_rule": fig_decision_rule,
    "per_batch_auc_fpr": fig_per_batch,
    "coefficient_spread": fig_coefficient_spread,
    "loso_diagram": fig_loso_diagram,
    "pipeline_overview": fig_pipeline_overview,
    "task_duration_age_confound": fig_task_duration,
    "eeg_trace_synthetic": fig_eeg_trace_synthetic,
}
PUBLIC_DATA_FIGURES = {"external_scatter": fig_external_scatter, "eeg_trace_public": fig_eeg_trace_public}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=[*FIGURES, *PUBLIC_DATA_FIGURES])
    ap.add_argument("--mendeley", type=Path, default=os.environ.get("MENDELEY_CACHE") or None,
                    help="folder with the public Mendeley files (mat/, workbooks/)")
    a = ap.parse_args(argv)
    names = a.only or [*FIGURES, *PUBLIC_DATA_FIGURES]
    for n in names:
        if n in FIGURES:
            FIGURES[n]()
        elif a.mendeley and (Path(a.mendeley) / "mat").is_dir():
            PUBLIC_DATA_FIGURES[n](Path(a.mendeley))
        else:
            print(f"skipped {n}: public Mendeley data not given (--mendeley DIR); keeping figures/{n}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
