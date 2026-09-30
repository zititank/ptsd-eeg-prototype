"""Model recipes, nested leave-one-subject-out (LOSO) training and scoring.

The shipped recipe, C2G, is:

1. Set A features -> C1 compositional transform (`transforms.c1_subject_transform`).
2. In-fold median imputation, winsorisation at the 1st/99th percentiles, robust scaling.
3. Shrinkage linear discriminant analysis (LDA, lsqr solver); the shrinkage is chosen by an inner
   5-fold selection on the training fold only (`choose_inner`).
4. 100 stratum-balanced bootstrap bags per fold; the fold output is the mean bag probability,
   clipped to [0.02, 0.98].
5. Outer LOSO over the 90 training subjects: fold model i never sees subject i. The deployed model is
   the ensemble of all 90 fold models; a new subject's score is their mean output.
6. Minimum-data gate at scoring: rest recordings under 30 s or truncated get the prior fallback 24/90.

Every learnt object (imputer, winsor bounds, scaler, shrinkage, LDA, bags, the C3 Mahalanobis gate) is
fitted inside the training fold it belongs to. A fitted model is a plain dict of numpy arrays and Python
scalars; `weights.py` stores it as `.npz` + JSON without pickle.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import qc as qcmod
from .features import SET_A, extract_subject
from .layout import get_layout
from .transforms import apply_prep, c1_subject_transform, compile_prep, make_prep

GRID: list = ["auto", .05, .10, .20, .30, .50, .70, .90]   # LDA shrinkage candidates ("auto" = Ledoit-Wolf)
SEED = 20260924            # inner StratifiedKFold seed base (v3)
BAG_SEED = 20260928        # bag seed base (PROTOCOL_v4 §2)
BOOT_SEED = 20260927       # flip-rate bootstrap seed
THRESHOLD = .5             # decision threshold on the ensemble probability
INNER_AUC_MIN = .690       # inner selection: shrinkages below this inner AUC are not eligible
FALLBACK = 24 / 90         # PTSD share of the scorable training cohort (prior), for unscorable input
FORMAT = "ptsd_eeg.model/1"

RECIPES: dict[str, dict] = {
    "v3": dict(prep="v3", subject_transform=None, bags=0, clip=None, gate=False, soma=False),
    "C1": dict(prep="c1", subject_transform="c1", bags=0, clip=None, gate=False, soma=False),
    "C2": dict(prep="c1", subject_transform="c1", bags=100, clip=(0.02, 0.98), gate=False, soma=False),
    "C3": dict(prep="c1", subject_transform="c1", bags=100, clip=(0.02, 0.98), gate=True, soma=False),
    "C4": dict(prep="c1", subject_transform="c1", bags=100, clip=(0.02, 0.98), gate=False, soma=True),
    # PROTOCOL_v4_AMENDMENT_1: the C2 members; the minimum-data gate acts only at scoring.
    "C2G": dict(prep="c1", subject_transform="c1", bags=100, clip=(0.02, 0.98), gate=False, soma=False,
                min_data_gate=True),
}
"""Candidate recipes of PROTOCOL_v4 (v3 reference, C1-C4) and the shipped C2G."""


def input_matrix(recipe: dict, X_setA: np.ndarray) -> np.ndarray:
    """The recipe's model input: Set A as is (v3) or C1-transformed."""
    X = np.asarray(X_setA, dtype=float)
    return c1_subject_transform(X) if recipe["subject_transform"] == "c1" else X


def make_model(prep_kind: str, shrinkage) -> Pipeline:
    """Unfitted preprocessing + shrinkage LDA pipeline."""
    return Pipeline(make_prep(prep_kind).steps + [
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage))])


def choose_inner(X: np.ndarray, y: np.ndarray, outer_index: int, prep_kind: str = "v3",
                 grid: Sequence = GRID) -> tuple[object, list[dict]]:
    """Inner shrinkage selection on one training fold (5-fold stratified CV, preprocessing refitted per split).

    Rule: among shrinkages with inner AUC >= 0.690, take the lowest inner control false-positive rate
    (ties: higher AUC, then grid order); if none is eligible, take the highest AUC. Returns the chosen
    shrinkage and an audit table (one dict per grid value).
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED + outer_index)
    scores = np.empty((len(grid), len(y)))
    for tr, va in cv.split(X, y):
        prefix = make_prep(prep_kind).fit(X[tr], y[tr])
        a, b = prefix.transform(X[tr]), prefix.transform(X[va])
        for j, shrink in enumerate(grid):
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink).fit(a, y[tr])
            scores[j, va] = clf.predict_proba(b)[:, 1]
    auc = [float(roc_auc_score(y, p)) for p in scores]
    fpr = [float(np.mean(p[y == 0] >= THRESHOLD)) for p in scores]
    eligible = [j for j in range(len(grid)) if auc[j] >= INNER_AUC_MIN]
    j = min(eligible, key=lambda j: (fpr[j], -auc[j], j)) if eligible else int(np.argmax(np.round(auc, 12)))
    audit = [dict(shrinkage=str(s), inner_auc=auc[k], inner_control_fpr=fpr[k],
                  eligible=auc[k] >= INNER_AUC_MIN, selected=k == j) for k, s in enumerate(grid)]
    return grid[j], audit


def _fit_one(X: np.ndarray, y: np.ndarray, prep_kind: str, shrinkage) -> dict:
    m = make_model(prep_kind, shrinkage).fit(X, y)
    lda = m.steps[-1][1]
    return dict(prep=compile_prep(Pipeline(m.steps[:-1])), coef=lda.coef_[0].copy(), intercept=float(lda.intercept_[0]))


def bag_indices(y: np.ndarray, strata: np.ndarray, n_bags: int, seed: int) -> list[np.ndarray]:
    """Stratum-balanced bootstrap bags.

    Positives are resampled uniformly. Negatives are resampled so that every negative stratum (e.g. the
    two control batches) has equal total probability, so no single batch dominates the control class.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    names, counts = np.unique(strata[neg], return_counts=True)
    w = {n: 1.0 / (len(names) * c) for n, c in zip(names, counts)}
    p = np.array([w[s] for s in strata[neg]])
    p = p / p.sum()
    return [np.concatenate([rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True, p=p)])
            for _ in range(n_bags)]


def fit_gate(X: np.ndarray) -> dict:
    """Mahalanobis out-of-distribution gate of C3 (Ledoit-Wolf covariance on the whole training fold)."""
    prep = compile_prep(make_prep("c1").fit(X))
    lw = LedoitWolf().fit(apply_prep(prep, X))
    g = dict(prep=prep, location=lw.location_.copy(), precision=lw.precision_.copy())
    g["threshold"] = float(np.percentile(mahalanobis2(g, X), 99))
    return g


def mahalanobis2(gate: dict, X: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of every row of X to the gate's training distribution."""
    d = apply_prep(gate["prep"], X) - gate["location"]
    return np.einsum("nf,fg,ng->n", d, gate["precision"], d)


def fit_member(X: np.ndarray, y: np.ndarray, strata: np.ndarray, outer_index: int, recipe: dict) -> dict:
    """Fit one fold model: inner shrinkage selection, (bagged) LDA fits, optional C3 gate.

    `outer_index` seeds the inner CV and the bags, which makes every fold reproducible.
    """
    X, y = np.asarray(X, float), np.asarray(y, int)
    strata = np.asarray(strata, dtype=object).astype(str)
    shrink, audit = choose_inner(X, y, outer_index, recipe["prep"])
    idxs = bag_indices(y, strata, recipe["bags"], BAG_SEED + outer_index) if recipe["bags"] else [np.arange(len(y))]
    fits = [_fit_one(X[i], y[i], recipe["prep"], shrink) for i in idxs]
    member = dict(shrinkage=str(shrink), inner_audit=audit,
                  prep={k: np.stack([f["prep"][k] for f in fits]) for k in fits[0]["prep"]},
                  coef=np.stack([f["coef"] for f in fits]), intercept=np.array([f["intercept"] for f in fits]),
                  clip=recipe["clip"], gate=fit_gate(X) if recipe["gate"] else None)
    return member


def member_proba(member: dict, X: np.ndarray) -> np.ndarray:
    """Fold-model probability: mean over bags of sigmoid(coef · z + intercept), clipped if the recipe clips.

    X is the recipe's input matrix (n_subjects, 4); z is X after the bag's compiled preprocessing.
    """
    X = np.asarray(X, float)
    p = member["prep"]
    Z = np.where(np.isnan(X)[None], p["fill"][:, None, :], X[None])
    Z = (np.clip(Z, p["lo"][:, None, :], p["hi"][:, None, :]) - p["center"][:, None, :]) / p["scale"][:, None, :]
    dec = np.einsum("bnf,bf->bn", Z, member["coef"]) + member["intercept"][:, None]
    prob = expit(dec).mean(axis=0)
    if member["clip"] is not None:
        prob = np.clip(prob, *member["clip"])
    return prob


def member_output(member: dict, X: np.ndarray, subject_gate=None) -> tuple[np.ndarray, np.ndarray]:
    """Fold-model output and gate mask. `subject_gate`: fold-independent duration/QC flags (C3 only)."""
    prob = member_proba(member, X)
    gated = np.zeros(len(prob), dtype=bool)
    if member["gate"] is not None:
        gated = mahalanobis2(member["gate"], X) > member["gate"]["threshold"]
        if subject_gate is not None:
            gated |= np.asarray(subject_gate, dtype=bool)
    return np.where(gated, FALLBACK, prob), gated


def score_matrix(members: Sequence[dict], X: np.ndarray, subject_gate=None) -> tuple[np.ndarray, np.ndarray]:
    """Outputs of every member for every row: (n_members, n_subjects) scores and gate masks."""
    out = [member_output(m, X, subject_gate) for m in members]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def loso_fold(X: np.ndarray, y: np.ndarray, strata: np.ndarray, i: int, recipe: dict) -> dict:
    """Fold model i of the outer LOSO: trained on every subject except row i."""
    keep = np.arange(len(y)) != i
    return fit_member(X[keep], y[keep], strata[keep], i, recipe)


def nested_loso(X: np.ndarray, y: np.ndarray, strata: np.ndarray, recipe: dict, n_jobs: int = 1) -> list[dict]:
    """Outer LOSO; member i never sees subject i. Row order defines the outer index (and inner seeds)."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    strata = np.asarray(strata, dtype=object).astype(str)
    if n_jobs == 1:
        return [loso_fold(X, y, strata, i, recipe) for i in range(len(y))]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(delayed(loso_fold)(X, y, strata, i, recipe) for i in range(len(y)))


def _boot_fold(X, y, strata, bi, b, recipe, X_score, gate_score):
    m = fit_member(X[bi], y[bi], strata[bi], b, recipe)
    out, gated = member_output(m, X_score, gate_score)
    return out, gated, m["shrinkage"]


def bootstrap_refits(X: np.ndarray, y: np.ndarray, strata: np.ndarray, recipe: dict, X_score: np.ndarray,
                     gate_score=None, n_boot: int = 200, seed: int = BOOT_SEED, n_jobs: int = 1):
    """Class-stratified bootstrap refits of the whole training procedure, used for the flip rate.

    Returns (scores (n_score, n_boot), gate masks, selected shrinkages, bootstrap index sets).
    """
    X, y = np.asarray(X, float), np.asarray(y, int)
    strata = np.asarray(strata, dtype=object).astype(str)
    rng = np.random.default_rng(seed)
    ip, ic = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    idx = [np.concatenate([rng.choice(ip, ip.size, replace=True), rng.choice(ic, ic.size, replace=True)])
           for _ in range(n_boot)]
    if n_jobs == 1:
        res = [_boot_fold(X, y, strata, bi, b, recipe, X_score, gate_score) for b, bi in enumerate(idx)]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(_boot_fold)(X, y, strata, bi, b, recipe, X_score, gate_score)
                                      for b, bi in enumerate(idx))
    return (np.stack([r[0] for r in res], axis=1), np.stack([r[1] for r in res], axis=1),
            [r[2] for r in res], idx)


def flip_rate(scores: np.ndarray) -> float:
    """Share of refits on the minority side of 0.5 (0 = always the same decision, 0.5 = coin flip)."""
    p = np.mean(np.asarray(scores) >= THRESHOLD)
    return float(min(p, 1 - p))


# ---------------------------------------------------------------------------------------------
# Packaged model: build, score from features, predict from a raw subject folder
# ---------------------------------------------------------------------------------------------
def build_model(recipe_name: str, members: Sequence[dict], y: np.ndarray, **extra) -> dict:
    """Packaged model dict. Stores aggregate training counts only, never subject identifiers."""
    recipe = RECIPES[recipe_name]
    y = np.asarray(y, int)
    return dict(format=FORMAT, version=f"ptsd_eeg_{recipe_name}", recipe=recipe_name, recipe_spec=dict(recipe),
                members=list(members), feature_names=list(SET_A), threshold=THRESHOLD, fallback_score=FALLBACK,
                qc=dict(min_duration_rest_s=qcmod.MIN_DURATION_REST_S, median_abs_r_min=qcmod.QC_MEDIAN_ABS_R_MIN,
                        fp1fp2_r_min=qcmod.QC_FP1_FP2_R_MIN, band_hz=qcmod.COUPLING_BAND_HZ),
                n_train=int(len(y)), n_ptsd=int(y.sum()), n_negative=int((y == 0).sum()), **extra)


def subject_gate_flags(record: dict) -> bool:
    """C3 fold-independent gate: coupling-QC failure or a too-short recording."""
    return bool(record.get("qc_fail", False) or record.get("too_short", False))


def score_record(model: dict, record: dict) -> float:
    """Ensemble score of one subject's feature record (fallback if unscorable or gated). Model failures raise."""
    if not record.get("scorable", False):
        return float(model["fallback_score"])
    spec = model["recipe_spec"]
    if spec.get("min_data_gate", False) and record["min_data_fail"]:
        return float(model["fallback_score"])
    X = input_matrix(spec, np.array([[record[c] for c in model["feature_names"]]], dtype=float))
    gate = np.array([subject_gate_flags(record)]) if spec["gate"] else None
    S, _ = score_matrix(model["members"], X, gate)
    p = float(S[:, 0].mean())
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Invalid model score")
    return p


def subject_record(model: dict, subject_dir: Path) -> dict:
    """Features (+ QC or minimum-data flags when the recipe gates) from the raw rest EDF."""
    if model["recipe_spec"]["gate"]:
        return qcmod.subject_record(subject_dir)
    if model["recipe_spec"].get("min_data_gate", False):
        return qcmod.min_data_record(subject_dir)
    return extract_subject(subject_dir)


def training_records(input_dir: Path, soma: bool = False):
    """Training-set group folders -> one row per subject (controls, PTSD, then somatoform for C4).

    `input_dir` holds the group folders named in `ptsd_eeg.layout`. The `stratum` column drives the bag
    balancing: control batch B (one recording session with non-physiological signals, see docs) forms its own
    stratum.
    """
    import pandas as pd

    from .cohort import control_stratum
    from .io import header_start_date, rest_path, subject_dirs
    dirs = get_layout().group_dirs
    rows = []
    groups = [(dirs["control"], 0), (dirs["ptsd"], 1)] + ([(dirs["soma"], 0)] if soma else [])
    for group, label in groups:
        for s in subject_dirs(Path(input_dir) / group):
            row = dict(subject_id=s.name, group=group, y=label)
            try:
                path = rest_path(s)
                row.update(extract_subject(s), rest_file=str(path.relative_to(input_dir)))
                row["start_date"] = header_start_date(path)
            except (OSError, ValueError, IndexError) as exc:
                row.update(scorable=False, error=f"{type(exc).__name__}: {exc}")
            if group == dirs["ptsd"]:
                row["stratum"] = "PTSD"
            elif group == dirs["soma"]:
                row["stratum"] = "SOMA"
            else:
                row["stratum"] = control_stratum(row.get("start_date"))
            rows.append(row)
    return pd.DataFrame(rows)


def train_from_dir(input_dir: Path, recipe_name: str = "C2G", output_dir: Path | None = None, n_jobs: int = 1,
                   shuffle_members: bool = False, **extra):
    """Nested-LOSO training from the raw training-set group folders (see `training_records`).

    Returns `(model, oof_frame, coverage_frame)`. With `output_dir`, writes the weights (`weights.save_weights`)
    plus `training_oof.csv`, `training_coverage.csv` and `training_manifest.json`. The last three are
    per-subject files of *your* data: keep them out of version control.
    """
    import pandas as pd
    recipe = RECIPES[recipe_name]
    frame = training_records(Path(input_dir), soma=recipe["soma"])
    train = frame[frame.scorable.astype(bool)].reset_index(drop=True)
    if not train.subject_id.is_unique:
        raise ValueError("Subject IDs must be unique across training groups")
    y = train.y.to_numpy(int)
    if min(np.bincount(y, minlength=2)) < 6:
        raise ValueError("Nested five-fold selection requires at least six subjects in each class")
    X = input_matrix(recipe, train[SET_A].to_numpy(float))
    members = nested_loso(X, y, train.stratum.to_numpy(), recipe, n_jobs=n_jobs)
    oof = np.array([member_proba(m, X[[i]])[0] for i, m in enumerate(members)])
    gated = np.zeros(len(y), dtype=bool)
    if recipe["gate"]:
        rec = [qcmod.subject_record(Path(input_dir) / f) for f in
               (train.group + "/" + train.subject_id)]
        flags = np.array([subject_gate_flags(r) for r in rec])
        for i, m in enumerate(members):
            out, g = member_output(m, X[[i]], flags[[i]])
            oof[i], gated[i] = out[0], g[0]
    min_data = None
    if recipe.get("min_data_gate", False):
        from .io import read_edf_header, rest_path
        min_data = pd.DataFrame([qcmod.min_data_flags(read_edf_header(rest_path(Path(input_dir) / g / s)))
                                 for g, s in zip(train.group, train.subject_id)])
        gated = min_data.min_data_fail.to_numpy(bool)
        oof = np.where(gated, FALLBACK, oof)
    strata_counts = {k: int(v) for k, v in train.stratum.value_counts().sort_index().items()}
    model = build_model(recipe_name, members, y, strata_counts=strata_counts, **extra)
    oof_df = pd.DataFrame(dict(subject_id=train.subject_id, group=train.group, stratum=train.stratum,
                               score=oof, gated=gated, selected_shrinkage=[m["shrinkage"] for m in members]))
    if min_data is not None:
        oof_df["duration_s"] = min_data.duration_s.to_numpy()
        oof_df["truncated"] = min_data.truncated.to_numpy()
    cov_cols = [c for c in ["subject_id", "group", "stratum", "scorable", "n_kept", "rest_file", "error"] if c in frame]
    coverage = frame[cov_cols]
    if output_dir is not None:
        import json
        import sys

        import sklearn

        from .weights import save_weights
        output_dir = Path(output_dir)
        save_weights(model, output_dir, shuffle=shuffle_members)
        oof_df.to_csv(output_dir / "training_oof.csv", index=False)
        coverage.to_csv(output_dir / "training_coverage.csv", index=False)
        (output_dir / "training_manifest.json").write_text(json.dumps(dict(
            recipe=recipe_name, python=sys.version, numpy=np.__version__, pandas=pd.__version__,
            sklearn=sklearn.__version__, n_train=len(y), n_ptsd=int(y.sum()), n_negative=int((y == 0).sum()),
            strata=strata_counts, fallback_score=FALLBACK, **extra), indent=2), encoding="utf-8")
    return model, oof_df, coverage


def predict(model: dict, subject_dir: Path) -> float:
    """PTSD-likeness score of one raw subject folder.

    An unreadable or unmeasurable rest recording returns the prior fallback (with a warning);
    genuine model failures raise.
    """
    try:
        record = subject_record(model, Path(subject_dir))
    except (OSError, ValueError, IndexError) as exc:
        warnings.warn(f"{Path(subject_dir).name}: unscorable rest; prior fallback: {exc}")
        return float(model["fallback_score"])
    if not record["scorable"]:
        warnings.warn(f"{Path(subject_dir).name}: no usable rest spectrum; prior fallback")
    return score_record(model, record)
