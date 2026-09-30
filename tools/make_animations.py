"""Render the README animations (MP4 + GIF preview + poster PNG) and the banner into media/.

    python tools/make_animations.py [--results DIR] [--out DIR] [--only NAME ...] [--no-gif] [--keep-frames]
    make animations                      (same, from the repository root)

Every number drawn is read from the aggregate tables in `results/` (roc_curves.csv, metrics.csv,
negative_controls.csv, candidate_gates.csv, shrinkage_selection.csv). Signals are synthetic and generated here
with a fixed seed; no recording of the restricted cohort is read. Frames go to `<out>/.frames/<name>/` and are
deleted after encoding. Needs matplotlib, numpy, pandas, Pillow and an ffmpeg with libx264
(`FFMPEG=/path/to/ffmpeg` overrides the lookup). Outputs carry no metadata (no paths, dates or tags).
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
W, H, DPI, FPS = 1280, 720, 100, 24
MP4_MAX, GIF_MAX = 2_500_000, 3_000_000
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

# Okabe-Ito colour-blind-safe palette, same as tools/make_figures.py
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#8C8C8C")
INK, MUTED, PANEL = "#1B1F24", "#5F6B7A", "#EEF2F7"
C_PTSD, C_CTRL, C_B, C_VALID = OI["vermillion"], OI["blue"], OI["orange"], OI["blue"]

# Data integrity audit (docs/data_integrity.md, section 1): alpha peak present in 0 of the 46 rest files.
BATCH_B_FILES, BATCH_B_WITH_ALPHA = 46, 0

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "text.color": INK,
})


# ------------------------------------------------------------------ helpers
def clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def lin(t: float, a: float, b: float) -> float:
    return clamp((t - a) / (b - a))


def ease(t: float, a: float, b: float) -> float:
    x = lin(t, a, b)
    return x * x * (3 - 2 * x)


def chrome(fig, title: str, subtitle: str | None = None, source: str | None = None,
           tag: str | None = None) -> None:
    fig.text(0.04, 0.935, title, fontsize=25, weight="bold", va="center")
    if subtitle:
        fig.text(0.04, 0.882, subtitle, fontsize=15, color=MUTED, va="center")
    fig.text(0.04, 0.028, "Research prototype · not a medical device", fontsize=11, color=MUTED)
    if source:
        fig.text(0.96, 0.028, source, fontsize=11, color=MUTED, ha="right")
    if tag:
        fig.text(0.96, 0.935, tag, ha="right", va="center", fontsize=13, weight="bold", color=OI["vermillion"],
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=OI["vermillion"], lw=1.6))


def box(fig, x, y, w, h, text, fc="white", ec=INK, lw=1.5, alpha=1.0, fontsize=14, color=INK, **kw):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                  transform=fig.transFigure, fc=fc, ec=ec, lw=lw, alpha=alpha))
    if text:
        fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color,
                 alpha=alpha, linespacing=1.3, **kw)


def arrow(fig, a, b, alpha=1.0, color=INK, lw=2.0, style="-|>", rad=0.0):
    fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle=style, mutation_scale=18,
                                   color=color, lw=lw, alpha=alpha, connectionstyle=f"arc3,rad={rad}"))


def fmt_ci(v: float, lo: float, hi: float) -> str:
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


@dataclass
class Scene:
    name: str
    duration: float
    poster_t: float
    draw: Callable[[plt.Figure, float], None]


@dataclass
class Data:
    roc: pd.DataFrame
    metrics: pd.DataFrame
    neg: pd.DataFrame
    gates: pd.DataFrame
    shrink: pd.DataFrame
    batch_b: str


def load(res: Path) -> Data:
    roc = pd.read_csv(res / "roc_curves.csv")
    # the flagged control batch is the one comparison that is neither pooled nor the valid controls
    flagged = [c for c in roc.comparison.unique() if c not in ("all_controls", "CTRL_other")]
    assert len(flagged) == 1, flagged
    return Data(roc=roc, metrics=pd.read_csv(res / "metrics.csv").set_index("model"),
                neg=pd.read_csv(res / "negative_controls.csv").set_index("name"),
                gates=pd.read_csv(res / "candidate_gates.csv").set_index("model"),
                shrink=pd.read_csv(res / "shrinkage_selection.csv", dtype={"shrinkage": str}),
                batch_b=flagged[0])


# ------------------------------------------------------------------ synthetic signals
FS = 125.0


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    f = np.fft.rfftfreq(x.size, 1 / FS)
    X = np.fft.rfft(x)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, x.size)


def _coloured(rng, n: int, exponent: float) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    X = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / f ** (exponent / 2)
    x = np.fft.irfft(X, n)
    return x / x.std()


def synthetic_rest(seed: int, seconds: float = 24.0) -> dict[str, np.ndarray]:
    """Brain-like eyes-closed rest (1/f + waxing 10 Hz alpha) and a flat, near-white non-brain signal."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    env = _bandpass(rng.normal(size=n), 0.05, 0.5)
    env = np.clip(1.0 + 0.6 * env / env.std(), 0.15, None)
    alpha = env * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
    brain = _bandpass(_coloured(rng, n, 1.6) + 0.7 * alpha, 1.0, 40.0)
    nonbrain = _bandpass(_coloured(rng, n, 0.1), 1.0, 40.0)
    return {"brain": brain / brain.std(), "nonbrain": nonbrain / nonbrain.std()}


def epoch_psd(x: np.ndarray, win: float = 2.0, step: float = 1.0):
    nw, ns = int(win * FS), int(step * FS)
    w = np.hanning(nw)
    P = np.array([np.abs(np.fft.rfft((x[s:s + nw] - x[s:s + nw].mean()) * w)) ** 2
                  for s in range(0, x.size - nw + 1, ns)])
    return np.fft.rfftfreq(nw, 1 / FS), P


def alpha_check(f: np.ndarray, p: np.ndarray):
    """Peak in 7-14 Hz at least 1.5x a power-law baseline fitted to the 3-7 and 14-30 Hz flanks."""
    flank = ((f >= 3) & (f < 7)) | ((f > 14) & (f <= 30))
    coef = np.polyfit(np.log(f[flank]), np.log(p[flank]), 1)
    band = (f >= 7) & (f <= 14)
    base = np.exp(np.polyval(coef, np.log(f[band])))
    ratio = p[band] / base
    k = int(np.argmax(ratio))
    return f[band], base, bool(ratio[k] >= 1.5), float(f[band][k]), float(ratio[k])


# ------------------------------------------------------------------ 1. brain vs non-brain
def scene_brain(d: Data) -> Scene:
    T_SCROLL, SPEED, WIN = 7.5, 2.2, 4.0
    sig_end = WIN + SPEED * T_SCROLL
    n_ep = int(sig_end - 2) + 1
    for seed in range(20260930, 20261030):
        sig = synthetic_rest(seed)
        spec = {k: epoch_psd(v) for k, v in sig.items()}
        chk = {k: alpha_check(f, P[:n_ep].mean(0)) for k, (f, P) in spec.items()}
        if chk["brain"][2] and chk["brain"][4] > 3 and chk["nonbrain"][4] < 1.3:
            break
    else:
        raise RuntimeError("no seed gives a clean synthetic contrast")
    f = spec["brain"][0]
    show = (f >= 2) & (f <= 30)
    norm = (f >= 1) & (f <= 40)
    cum = {k: np.cumsum(P, 0) / np.arange(1, len(P) + 1)[:, None] for k, (_, P) in spec.items()}
    cum = {k: v / v[:, norm].sum(1, keepdims=True) for k, v in cum.items()}
    finals = np.concatenate([cum[k][n_ep - 1, show] for k in cum])
    ylim = (finals.min() * 0.4, finals.max() * 6)
    rows = [("brain", "Brain-like: synthetic eyes-closed rest", OI["green"], 0.625),
            ("nonbrain", "Non-brain: synthetic flat noise", C_B, 0.335)]

    def draw(fig, t):
        chrome(fig, "Where is the alpha peak?",
               "Eyes-closed EEG shows a bump near 10 Hz. A recording without brain signal does not.",
               source="Synthetic signals. Audit count: docs/data_integrity.md", tag="SYNTHETIC ILLUSTRATION")
        sig_t = WIN + SPEED * min(t, T_SCROLL)
        k = min(int(sig_t - 2) + 1, n_ep)
        verdict = ease(t, 7.6, 8.2)
        for key, label, colour, y in rows:
            ax = fig.add_axes([0.05, y, 0.50, 0.19])
            i0, i1 = int((sig_t - WIN) * FS), int(sig_t * FS)
            tt = np.arange(i0, i1) / FS
            ax.plot(tt, sig[key][i0:i1], color=INK, lw=0.9)
            ax.set_xlim(sig_t - WIN, sig_t)
            ax.set_ylim(-4.2, 4.2)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="x", labelsize=11)
            if key == "nonbrain":
                ax.set_xlabel("time (s)", fontsize=12, labelpad=1)
            ax.text(0.0, 1.05, label, transform=ax.transAxes, fontsize=15, weight="bold", color=colour)

            sx = fig.add_axes([0.63, y, 0.33, 0.19])
            sx.axvspan(8, 13, color=OI["yellow"], alpha=0.35, lw=0)
            sx.semilogy(f[show], cum[key][k - 1, show], color=colour, lw=2.4)
            sx.set_xlim(2, 30)
            sx.set_ylim(*ylim)
            sx.set_yticks([])
            sx.minorticks_off()
            sx.set_xticks([5, 10, 15, 20, 25, 30])
            sx.tick_params(axis="x", labelsize=11)
            if key == "nonbrain":
                sx.set_xlabel("frequency (Hz)", fontsize=12, labelpad=1)
            sx.text(10.5, ylim[1] * 0.7, "alpha band", ha="center", va="top", fontsize=11, color=MUTED)
            sx.text(0.98, 0.06, f"{k} epochs averaged", transform=sx.transAxes, ha="right", fontsize=11,
                    color=MUTED)
            if verdict > 0:
                fb, base, found, peak_f, _ = chk[key]
                scale = cum[key][n_ep - 1, show].sum() / spec[key][1][:n_ep].mean(0)[show].sum()
                sx.plot(fb, 1.5 * base * scale, ls="--", color=INK, lw=1.4, alpha=verdict)
                txt = f"✓ alpha peak at {peak_f:.1f} Hz" if found else "✗ no alpha peak"
                sx.text(0.0, 1.05, txt, transform=sx.transAxes, ha="left", fontsize=15,
                        weight="bold", color=OI["green"] if found else OI["vermillion"], alpha=verdict)
                sx.text(0.98, 0.9, "- - 1.5 × baseline", transform=sx.transAxes, ha="right",
                        va="top", fontsize=11, color=MUTED, alpha=verdict)
            else:
                sx.text(0.0, 1.05, "power spectrum", transform=sx.transAxes, fontsize=15, color=MUTED)

        s = ease(t, 8.2, 8.6)
        if s > 0:
            n_fail = int(round(lin(t, 8.6, 10.8) * (BATCH_B_FILES - BATCH_B_WITH_ALPHA)))
            fig.text(0.04, 0.215, f"Control batch B, real audit result:  {n_fail}/{BATCH_B_FILES} rest files failed "
                     "the alpha-peak check", fontsize=16, weight="bold", alpha=s)
            ax = fig.add_axes([0.04, 0.085, 0.92, 0.085])
            ax.set_xlim(0, BATCH_B_FILES)
            ax.set_ylim(0, 1)
            ax.axis("off")
            for i in range(BATCH_B_FILES):
                failed = i < n_fail
                ax.add_patch(Rectangle((i + 0.1, 0.1), 0.8, 0.8, fc=OI["vermillion"] if failed else "white",
                                       ec=OI["vermillion"], lw=1.2, alpha=s))
                if failed:
                    ax.text(i + 0.5, 0.5, "✗", ha="center", va="center", fontsize=12, color="white",
                            weight="bold")

    return Scene("brain_vs_nonbrain", 12.0, 11.9, draw)


# ------------------------------------------------------------------ 2. ROC falls apart
def scene_roc(d: Data) -> Scene:
    roc = d.roc[d.roc.model == "C2G"]
    m = d.metrics.loc["C2G"]
    B = d.batch_b

    def curve(comp):
        c = roc[roc.comparison == comp].sort_values("fpr")
        return c.fpr.to_numpy(), c.tpr.to_numpy(), c.tpr_lo.to_numpy(), c.tpr_hi.to_numpy(), int(c.n_control.iloc[0])

    specs = []
    for comp, name, colour, col, a, b in (("all_controls", "All {n} controls", INK, "auc", 0.8, 3.3),
                                          (B, "Control batch B only ({n})", C_B, f"auc_vs_{B}", 4.9, 7.1),
                                          ("CTRL_other", "Valid controls only ({n})", C_VALID,
                                           "auc_vs_CTRL_other", 7.5, 9.7)):
        x, y, lo, hi, n = curve(comp)
        specs.append(dict(x=x, y=y, lo=lo, hi=hi, name=name.format(n=n), colour=colour,
                          auc=m[col], auc_lo=m[f"{col}_lo"], auc_hi=m[f"{col}_hi"], a=a, b=b))
    short = ["All controls", "Control batch B", "Valid controls"]

    def draw(fig, t):
        chrome(fig, "The headline ROC falls apart",
               "Same out-of-fold scores of the shipped model, split by which controls they are compared with.",
               source="results/roc_curves.csv, results/metrics.csv (model C2G, 95% CIs)")
        ax = fig.add_axes([0.07, 0.135, 0.39, 0.693])
        ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1.5)
        ax.text(0.62, 0.55, "chance", rotation=45, color=MUTED, fontsize=12, ha="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("false-positive rate")
        ax.set_ylabel("true-positive rate")
        handles = []
        for i, s in enumerate(specs):
            p = lin(t, s["a"], s["b"])
            if p <= 0:
                continue
            fade = 0.3 if (i == 0 and t > 4.6) else 1.0
            fade = 1 - (1 - fade) * ease(t, 4.4, 4.9) if i == 0 else fade
            band = ease(t, s["b"], s["b"] + 0.8)
            if band > 0:
                ax.fill_between(s["x"], s["lo"], s["hi"], color=s["colour"], alpha=0.14 * band * fade, lw=0,
                                step="post")
            keep = s["x"] <= p
            xs = np.append(s["x"][keep], p)
            ys = np.append(s["y"][keep], np.interp(p, s["x"], s["y"]))
            ax.step(xs, ys, where="post", color=s["colour"], lw=3, alpha=fade)
            if p < 1:
                ax.plot(xs[-1], ys[-1], "o", color=s["colour"], ms=9)
            handles.append(Line2D([], [], color=s["colour"], lw=3, alpha=fade, label=s["name"]))
        if handles:
            ax.legend(handles=handles, loc="lower right", fontsize=13)

        tx = 0.53
        s0 = specs[0]
        if t >= s0["a"]:
            v = s0["auc"] * ease(t, s0["a"], s0["b"]) + 0.5 * (1 - ease(t, s0["a"], s0["b"]))
            fig.text(tx, 0.78, f"Headline AUC  {v:.3f}", fontsize=24, weight="bold")
            if t >= s0["b"]:
                fig.text(tx, 0.735, f"95% CI [{s0['auc_lo']:.3f}, {s0['auc_hi']:.3f}], 24 PTSD vs 66 controls",
                         fontsize=14, color=MUTED)
        a = ease(t, 3.8, 4.4)
        if a > 0:
            fig.text(tx, 0.665, "Now split the controls by acquisition batch:", fontsize=17, alpha=a)

        bx = fig.add_axes([0.66, 0.29, 0.30, 0.32])
        bx.set_xlim(0, 1.0)
        bx.set_ylim(-0.6, 2.6)
        bx.axvline(0.5, ls=":", color=MUTED, lw=1.5)
        bx.text(0.51, -0.55, "chance", ha="left", va="bottom", fontsize=11, color=MUTED)
        bx.set_yticks([2, 1, 0])
        bx.set_yticklabels(short, fontsize=14)
        bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        bx.set_xlabel("AUC", fontsize=13)
        bx.spines["left"].set_visible(False)
        bx.tick_params(axis="y", length=0)
        for i, s in enumerate(specs):
            y = 2 - i
            p = ease(t, s["a"], s["b"])
            if p <= 0:
                continue
            v = 0.5 + (s["auc"] - 0.5) * p
            bx.barh(y, v, height=0.56, color=s["colour"], alpha=0.85)
            e = ease(t, s["b"], s["b"] + 0.6)
            if e > 0:
                bx.errorbar(s["auc"], y, xerr=[[s["auc"] - s["auc_lo"]], [s["auc_hi"] - s["auc"]]], fmt="none",
                            ecolor=INK, capsize=5, lw=1.5, alpha=e)
            bx.text(0.02, y + 0.36, f"{v:.3f}" + (f"  [{s['auc_lo']:.3f}, {s['auc_hi']:.3f}]" if e > 0 else ""),
                    fontsize=12, va="bottom", color=INK)
        v = ease(t, 10.3, 10.9)
        if v > 0:
            box(fig, tx, 0.085, 0.43, 0.13,
                "The headline is carried by control batch B,\nwhich the audit found holds no physiological EEG.\n"
                "Against valid controls the model is at chance.",
                fc="#FFF4E0", ec=C_B, alpha=v, fontsize=14.5)

    return Scene("roc_falls_apart", 13.0, 12.9, draw)


# ------------------------------------------------------------------ 3. nested LOSO
def scene_loso(d: Data) -> Scene:
    N, N_PTSD = 90, 24
    grid = ["auto", "0.05", "0.1", "0.2", "0.3", "0.5", "0.7", "0.9"]
    counts = {("auto" if s == "auto" else f"{float(s):g}"): int(c)
              for s, c in zip(d.shrink.shrinkage, d.shrink.members_selecting)}
    assert sum(counts.values()) == N, counts
    m, g = d.metrics.loc["C2G"], d.gates.loc["C2"]
    xs, ys = np.arange(N) % 10, 8 - np.arange(N) // 10
    base = np.array([C_PTSD] * N_PTSD + [C_CTRL] * (N - N_PTSD))
    inner_group = np.full(N, -1)
    slow = [(0.8, 3.6), (3.6, 5.6)]
    FAST_A, FAST_B = 5.6, 11.0

    def state(t):
        """(current fold, fraction of fold elapsed, inner fold k or -1, bag progress, n scored)"""
        if t < slow[0][0]:
            return -1, 0.0, -1, 0.0, 0
        for i, (a, b) in enumerate(slow):
            if t < b:
                u = (t - a) / (b - a)
                k = int(u / 0.55 * 5) if u < 0.55 else -1
                return i, u, k, lin(u, 0.55, 0.9), i + (u >= 0.9)
        if t < FAST_B:
            fcur = 2 + (t - FAST_A) / (FAST_B - FAST_A) * (N - 2)
            i = int(fcur)
            return i, fcur - i, -1, 1.0, i
        return N, 1.0, -1, 1.0, N

    def draw(fig, t):
        chrome(fig, "Nested leave-one-subject-out, fold by fold",
               "Every validation score comes from a model that never saw that subject.",
               source="results/metrics.csv, shrinkage_selection.csv, candidate_gates.csv")
        i, u, k, bags, scored = state(t)
        ax = fig.add_axes([0.04, 0.1, 0.46, 0.74])
        ax.set_xlim(-0.7, 9.7)
        ax.set_ylim(-0.7, 8.7)
        ax.axis("off")
        fade = ease(t, 0, 0.6)
        alphas = np.where(np.arange(N) < scored, 1.0, 0.28) * fade
        if 0 <= i < N:
            order = np.flatnonzero(np.arange(N) != i)
            inner_group[:] = -1
            for stratum in (order[order < N_PTSD], order[order >= N_PTSD]):
                perm = np.random.default_rng(1000 + i).permutation(stratum)
                inner_group[perm] = np.arange(perm.size) % 5
        ax.scatter(xs, ys, s=720, c=base, alpha=alphas, linewidths=0)
        if k >= 0:
            sel = inner_group == k
            ax.scatter(xs[sel], ys[sel], s=720, facecolors="none", edgecolors=INK, linewidths=2.4)
        if 0 <= i < N:
            ax.scatter([xs[i]], [ys[i]], s=1150, facecolors="none", edgecolors=INK, linewidths=3.5)
            ax.text(xs[i], ys[i] + 0.52, "held out", ha="center", va="bottom", fontsize=12, weight="bold")
        ax.text(-0.5, -0.62, "●", color=C_PTSD, fontsize=16, va="center")
        ax.text(-0.1, -0.62, "24 PTSD", fontsize=13, va="center")
        ax.text(1.8, -0.62, "●", color=C_CTRL, fontsize=16, va="center")
        ax.text(2.2, -0.62, "66 controls", fontsize=13, va="center")
        ax.text(5.0, -0.62, "faded = not yet scored", fontsize=13, va="center", color=MUTED)

        px = 0.56
        fold_label = f"{min(i + 1, N)} / {N}" if i >= 0 else f"– / {N}"
        fig.text(px, 0.79, f"Outer fold  {fold_label}", fontsize=24, weight="bold")
        fig.text(px, 0.745, "Hold out 1 subject, fit everything on the other 89.", fontsize=14, color=MUTED)
        fig.text(px, 0.67, "Inner loop: stratified 5-fold CV on the 89", fontsize=15)
        for j in range(5):
            on = (j == k)
            box(fig, px + j * 0.047, 0.60, 0.034, 0.045, f"{j + 1}", fc=INK if on else PANEL,
                ec=INK, lw=1.0, fontsize=13, color="white" if on else INK)
        fig.text(px + 0.245, 0.622, "→ picks LDA shrinkage", fontsize=14, va="center", color=MUTED)
        fig.text(px, 0.535, "100 stratum-balanced bags → fold member", fontsize=15)
        bx = fig.add_axes([px, 0.49, 0.25, 0.025])
        bx.axis("off")
        bx.set_xlim(0, 1)
        bx.add_patch(Rectangle((0, 0), 1, 1, fc=PANEL, ec="none"))
        bx.add_patch(Rectangle((0, 0), bags if i >= 0 else 0, 1, fc=OI["green"], ec="none"))
        fig.text(px + 0.265, 0.502, f"LDA fits so far: {scored * 100:,}", fontsize=14, va="center")

        tx = fig.add_axes([0.62, 0.12, 0.32, 0.27])
        prog = scored / N
        vals = [counts.get(s, 0) for s in grid]
        yy = np.arange(len(grid))[::-1]
        tx.barh(yy, [v * prog for v in vals], color=OI["sky"], height=0.7)
        tx.set_yticks(yy)
        tx.set_yticklabels(["auto"] + [f"{float(s):.2f}" for s in grid[1:]], fontsize=11)
        tx.set_xlim(0, 36)
        tx.tick_params(axis="x", labelsize=11)
        tx.tick_params(axis="y", length=0)
        tx.spines["left"].set_visible(False)
        tx.set_title("Shrinkage chosen by the 90 inner loops", fontsize=13, loc="left", pad=4)
        if prog >= 1:
            for y, v in zip(yy, vals):
                tx.text(v + 0.5, y, str(v), va="center", fontsize=11)

        a = ease(t, 11.2, 11.8)
        if a > 0:
            B = d.batch_b
            box(fig, 0.05, 0.15, 0.44, 0.60,
                f"{N} out-of-fold scores\n→ AUC {fmt_ci(m.auc, m.auc_lo, m.auc_hi)}\n"
                f"(by batch: {m[f'auc_vs_{B}']:.3f} vs control batch B,\n"
                f"{m.auc_vs_CTRL_other:.3f} vs valid controls)\n\n"
                f"Deployed model = mean of the {N} fold members\n({N} × 100 = {N * 100:,} LDA fits)\n\n"
                f"Label-permutation run: AUC {g.permuted_auc:.3f},\ninside the null range "
                f"[{g.null_95_lo:.3f}, {g.null_95_hi:.3f}]",
                fc="white", ec=INK, alpha=a * 0.96, fontsize=15)

    return Scene("loso_folds", 14.0, 13.9, draw)


# ------------------------------------------------------------------ 4. metadata leak
def scene_leak(d: Data) -> Scene:
    hdr = d.neg.loc["header_only"]
    rng = np.random.default_rng(7)
    field_w = rng.uniform(0.25, 0.85, 6)
    chans = [_coloured(rng, 600, 1.2) for _ in range(6)]
    T0, T1 = 3.0, 6.5

    def draw(fig, t):
        chrome(fig, "The metadata leak",
               "A negative control that no PTSD marker should pass. It passed, and that was the warning.",
               source="results/negative_controls.csv (row header_only, 95% CI)",
               tag="CONCEPTUAL DIAGRAM")
        ax = fig.add_axes([0.04, 0.13, 0.34, 0.69])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        a = ease(t, 0, 0.8)
        ax.add_patch(Rectangle((0.02, 0.80), 0.96, 0.15, fc=PANEL, ec=INK, lw=1.5, alpha=a))
        for j, wv in enumerate(field_w):
            ax.add_patch(Rectangle((0.06 + (j % 2) * 0.46, 0.905 - (j // 2) * 0.04), 0.4 * wv, 0.022,
                                   fc=INK, ec="none", alpha=0.45 * a))
        ax.text(0.02, 0.965, "metadata (file header)", fontsize=14, weight="bold", alpha=a)
        ax.add_patch(Rectangle((0.02, 0.03), 0.96, 0.72, fc="white", ec=INK, lw=1.5, alpha=a))
        ax.text(0.05, 0.71, "EEG signal, 6 channels (synthetic)", fontsize=13, weight="bold", alpha=a, va="top")
        xx = np.linspace(0.05, 0.95, 600)
        for j, c in enumerate(chans):
            ax.plot(xx, 0.58 - j * 0.095 + 0.016 * np.clip(c, -2.5, 2.5), color=INK, lw=0.8, alpha=a)

        mute_sig = ease(t, 1.5, 2.3)
        if mute_sig > 0:
            ax.add_patch(Rectangle((0.025, 0.035), 0.95, 0.71, fc="white", ec="none", alpha=0.85 * mute_sig,
                                   zorder=5))
            ax.text(0.5, 0.36, "not read", ha="center", va="center", fontsize=24, weight="bold",
                    color=MUTED, alpha=mute_sig, zorder=6)

        box(fig, 0.43, 0.44, 0.14, 0.12, "classifier", fc=PANEL, ec=INK, fontsize=16, alpha=a)
        a1 = ease(t, 2.0, 2.8)
        if a1 > 0:
            arrow(fig, (0.375, 0.735), (0.43, 0.53), alpha=a1, lw=2.5, rad=-0.2)
            arrow(fig, (0.57, 0.50), (0.62, 0.50), alpha=a1, lw=2.5)

        bx = fig.add_axes([0.64, 0.44, 0.32, 0.24])
        bx.set_xlim(0, 1.0)
        bx.set_ylim(-0.6, 1.6)
        bx.axvline(0.5, ls=":", color=MUTED, lw=1.5)
        bx.text(0.51, -0.55, "chance", ha="left", va="bottom", fontsize=11, color=MUTED)
        bx.set_yticks([])
        bx.spines["left"].set_visible(False)
        bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        bx.set_xlabel("AUC", fontsize=13)
        p = ease(t, T0, T1)
        if p > 0:
            v = 0.5 + (hdr.auc - 0.5) * p
            bx.barh(0, v, height=0.7, color=OI["purple"], alpha=0.85)
            bx.text(0.0, 0.45, "Metadata only, zero EEG samples:\nPTSD vs controls", fontsize=14, va="bottom",
                    linespacing=1.2)
            bx.text(0.03, 0, f"{v:.2f}", va="center", ha="left", fontsize=20, weight="bold", color="white")
            e = ease(t, T1, T1 + 0.5)
            if e > 0:
                bx.errorbar(hdr.auc, 0, xerr=[[hdr.auc - hdr.auc_lo], [hdr.auc_hi - hdr.auc]], fmt="none",
                            ecolor=INK, capsize=6, lw=1.8, alpha=e)

        k = ease(t, 7.3, 7.9)
        if k > 0:
            fig.text(0.80, 0.32, "✓ Our negative control caught it", ha="center", va="center", fontsize=18,
                     weight="bold", color=OI["green"], alpha=k,
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=OI["green"], lw=1.8, alpha=k))

        c = ease(t, 8.8, 9.4)
        if c > 0:
            box(fig, 0.43, 0.08, 0.53, 0.15,
                "Metadata carries no biology, yet it separated PTSD from controls.\n"
                "Acquisition batch was baked into the labels, so no AUC was trusted\n"
                "until batch had been ruled out.",
                fc="#F6EEF4", ec=OI["purple"], alpha=c, fontsize=14)

    return Scene("metadata_leak", 12.0, 11.9, draw)


# ------------------------------------------------------------------ 5. pipeline flow
def scene_pipeline(d: Data) -> Scene:
    stages = [
        ("Rest EDF", "eyes closed", "One eyes-closed rest recording per subject; channels picked by label."),
        ("Minimum-data gate", "≥ 30 s, not truncated",
         "Shorter than 30 s of complete data, or truncated → fixed fallback score 24/90 = 0.267."),
        ("Clean", "centre · notch · filter",
         "Median centring; 50 Hz notch only with a clear spike; 1–40 Hz band-pass only if needed."),
        ("Gain normalise", "÷ mean O1/O2 IQR",
         "Device gains differ between batches, so every channel is divided by the O1/O2 interquartile range."),
        ("Epochs", "2 s · 50% overlap", "Epochs with robust z > 8 in amplitude or variance are dropped."),
        ("Welch PSD", "mean over O1, O2", "Power spectrum per epoch, averaged over epochs, then over O1 and O2."),
        ("4 features", "IAF · α · α(IAF) · β",
         "Individual alpha frequency, relative alpha 8–13 Hz, alpha at IAF ± 2 Hz, relative beta 13–30 Hz."),
        ("Transforms", "CLR + logit", "Compositional (CLR) and logit transforms of the relative powers."),
        ("Ensemble", "90 × 100 bagged LDA",
         "90 nested-LOSO fold members × 100 bags of shrinkage-LDA; probabilities averaged."),
        ("Score", "decision at 0.5", "One number per subject; positive if the score is at least 0.5."),
    ]
    BW, BH = 0.165, 0.15
    cols = [0.125, 0.3125, 0.5, 0.6875, 0.875]
    centres = [(cols[j], 0.60) for j in range(5)] + [(cols[4 - j], 0.33) for j in range(5)]
    T0, STEP, MOVE = 0.8, 0.84, 0.3

    def pos(t):
        if t < T0:
            return centres[0], 0
        u = (t - T0) / STEP
        i = int(u)
        if i >= len(centres) - 1:
            return centres[-1], len(centres) - 1
        frac = ease(u - i, 0, MOVE / STEP)
        (x0, y0), (x1, y1) = centres[i], centres[i + 1]
        return (x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac), i + (frac >= 1)

    def draw(fig, t):
        chrome(fig, "From one rest file to one score",
               "The shipped pipeline. No age, no metadata, no task timing enter the model.",
               source="docs/pipeline.md")
        (tx, ty), reached = pos(t)
        a = ease(t, 0, 0.6)
        for i in range(len(centres) - 1):
            (x0, y0), (x1, y1) = centres[i], centres[i + 1]
            if y0 == y1:
                dx = BW / 2 + 0.004
                sgn = 1 if x1 > x0 else -1
                arrow(fig, (x0 + sgn * dx, y0), (x1 - sgn * dx, y1), alpha=a, color=MUTED)
            else:
                arrow(fig, (x0, y0 - BH / 2 - 0.004), (x1, y1 + BH / 2 + 0.004), alpha=a, color=MUTED)
        gx, gy = centres[1]
        hot = reached == 1 and t < T0 + 2 * STEP
        box(fig, gx - 0.07, 0.755, 0.14, 0.055, "fallback 0.267", fc="#FDE8DC" if hot else "white",
            ec=OI["vermillion"], alpha=a, fontsize=12.5, color=OI["vermillion"])
        arrow(fig, (gx, gy + BH / 2 + 0.004), (gx, 0.753), alpha=a, color=OI["vermillion"], lw=1.5)
        fig.text(gx + 0.008, gy + BH / 2 + 0.018, "no", fontsize=11, color=OI["vermillion"], alpha=a)
        for i, ((cx, cy), (name, sub, _)) in enumerate(zip(centres, stages)):
            lit = i <= reached and t >= T0 - 0.2
            fc = "#DCEEFB" if lit else "white"
            ec = OI["blue"] if lit else MUTED
            if i == reached and t < T0 + len(centres) * STEP:
                fc, ec = OI["blue"], OI["blue"]
            txt_c = "white" if fc == OI["blue"] else INK
            box(fig, cx - BW / 2, cy - BH / 2, BW, BH, "", fc=fc, ec=ec, lw=2, alpha=a)
            fig.text(cx, cy + 0.022, name, ha="center", va="center", fontsize=13.5, weight="bold", color=txt_c,
                     alpha=a)
            fig.text(cx, cy - 0.028, sub, ha="center", va="center", fontsize=11.5, color=txt_c, alpha=a)
            fig.text(cx - BW / 2 + 0.006, cy + BH / 2 - 0.006, str(i + 1), ha="left", va="top", fontsize=10,
                     color=txt_c, alpha=0.8 * a)
        if t >= T0:
            fig.add_artist(plt.Circle((tx, ty - BH / 2 - 0.022), 0.009, transform=fig.transFigure,
                                      color=OI["orange"], zorder=10))
        detail = stages[reached][2] if t >= T0 - 0.2 else ""
        fin = ease(t, T0 + len(centres) * STEP - 0.2, T0 + len(centres) * STEP + 0.4)
        if fin > 0:
            detail = "Same code path for training, validation and new subjects; retraining reproduces within 5e-16."
        fig.text(0.5, 0.15, detail, ha="center", va="center", fontsize=16, color=INK)

    return Scene("pipeline_flow", 10.5, 10.4, draw)


# ------------------------------------------------------------------ banner
def make_banner(out: Path) -> Path:
    bg = "#0E1726"
    fig = plt.figure(figsize=(12.8, 4.0), dpi=100, facecolor=bg)
    rng = np.random.default_rng(11)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="none")
    ax.axis("off")
    n = 1400
    tt = np.linspace(0, 1, n)
    for j, (col, amp) in enumerate(((OI["sky"], 0.9), (OI["green"], 0.7), (OI["orange"], 1.0),
                                    (OI["purple"], 0.6), (OI["sky"], 0.8))):
        x = _coloured(rng, n, 1.5) * 0.25 + amp * 0.35 * np.sin(2 * np.pi * 40 * tt + j) * \
            np.clip(np.sin(np.pi * (tt * 3.0 + j * 0.3)), 0, None)
        ax.plot(tt, 0.9 - j * 0.2 + 0.045 * x, color=col, lw=1.0, alpha=0.28)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    sx = fig.add_axes([0.79, 0.14, 0.18, 0.5], facecolor="none")
    f = np.linspace(2, 30, 300)
    p = 1 / f ** 1.3 + 0.3 * np.exp(-0.5 * ((f - 10) / 1.1) ** 2)
    sx.fill_between(f, p, color=OI["orange"], alpha=0.25, lw=0)
    sx.plot(f, p, color=OI["orange"], lw=3)
    sx.axis("off")
    sx.text(10, 0.46, "alpha peak", color=OI["orange"], ha="center", fontsize=13, weight="bold")
    sx.set_ylim(0, 0.6)
    fig.text(0.045, 0.72, "PTSD from EEG — a safety-net case study", fontsize=29, weight="bold",
             color="white", va="center")
    fig.text(0.045, 0.55, "Nested LOSO · negative controls · data-integrity audit", fontsize=16,
             color="#B8C4D6", va="center")
    fig.text(0.045, 0.30, "Research prototype — not a medical device", fontsize=15, weight="bold",
             color=OI["orange"], va="center",
             bbox=dict(boxstyle="round,pad=0.5", fc=bg, ec=OI["orange"], lw=1.6))
    path = out / "banner.png"
    fig.savefig(path, dpi=100, facecolor=bg, metadata={"Software": None})
    plt.close(fig)
    return path


# ------------------------------------------------------------------ rendering
def remove_frames(path: Path, frames_root: Path) -> None:
    path, frames_root = path.resolve(), frames_root.resolve()
    if frames_root.name != ".frames" or not (path == frames_root or frames_root in path.parents):
        raise RuntimeError(f"refusing to delete {path}: not inside {frames_root}")
    if path.exists():
        shutil.rmtree(path)


def ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


_JOB: tuple[Scene, Path] | None = None
_FIG = None


def _render_frame(i: int) -> None:
    global _FIG
    scene, frames = _JOB
    if _FIG is None:
        _FIG = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    _FIG.clear()
    scene.draw(_FIG, i / FPS)
    _FIG.canvas.draw()
    rgb = np.asarray(_FIG.canvas.buffer_rgba())[..., :3]
    assert rgb.shape == (H, W, 3), rgb.shape
    Image.fromarray(rgb).save(frames / f"f{i:04d}.png", compress_level=1)


def render(scene: Scene, out: Path, gif: bool, keep: bool, jobs: int) -> list[Path]:
    global _JOB, _FIG
    frames_root = out / ".frames"
    frames = frames_root / scene.name
    remove_frames(frames, frames_root)
    frames.mkdir(parents=True)
    n = int(round(scene.duration * FPS))
    _JOB, _FIG = (scene, frames), None
    if jobs > 1:
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            pool.map(_render_frame, range(n), chunksize=8)
    else:
        for i in range(n):
            _render_frame(i)
        plt.close(_FIG)

    pattern = str(frames / "f%04d.png")
    mp4 = out / f"{scene.name}.mp4"
    for crf in (24, 27, 30, 33, 36):
        ffmpeg("-framerate", str(FPS), "-i", pattern, "-map_metadata", "-1", "-c:v", "libx264", "-preset", "slow",
               "-tune", "animation", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4))
        if mp4.stat().st_size <= MP4_MAX:
            break
    outputs = [mp4]

    if gif:
        g = out / f"{scene.name}.gif"
        for fps, width, colours in ((10, 720, 128), (10, 640, 96), (8, 640, 64), (8, 560, 48), (6, 480, 32)):
            vf = (f"fps={fps},scale={width}:-2:flags=lanczos,split[a][b];"
                  f"[a]palettegen=max_colors={colours}:stats_mode=diff[p];"
                  f"[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
            ffmpeg("-framerate", str(FPS), "-i", pattern, "-map_metadata", "-1", "-vf", vf, "-loop", "0", str(g))
            if g.stat().st_size <= GIF_MAX:
                break
        outputs.append(g)

    poster = out / f"{scene.name}_poster.png"
    idx = min(n - 1, int(round(scene.poster_t * FPS)))
    Image.open(frames / f"f{idx:04d}.png").save(poster, optimize=True)
    outputs.append(poster)

    if not keep:
        remove_frames(frames, frames_root)
        if frames_root.exists() and not any(frames_root.iterdir()):
            remove_frames(frames_root, frames_root)
    return outputs


SCENES = {"brain_vs_nonbrain": scene_brain, "roc_falls_apart": scene_roc, "loso_folds": scene_loso,
          "metadata_leak": scene_leak, "pipeline_flow": scene_pipeline}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "media")
    ap.add_argument("--only", nargs="+", choices=[*SCENES, "banner"])
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = args.only or [*SCENES, "banner"]
    data = load(args.results)
    written: list[Path] = []
    for name in wanted:
        if name == "banner":
            written.append(make_banner(args.out))
        else:
            written += render(SCENES[name](data), args.out, gif=not args.no_gif, keep=args.keep_frames,
                              jobs=args.jobs)
    for p in written:
        print(f"wrote {p.name:32s} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
