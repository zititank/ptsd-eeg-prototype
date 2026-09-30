"""Pickle-free model weights: one `.npz` of numpy arrays plus a JSON metadata file.

Layout of a weights folder (the shipped one is `weights/` at the repository root):

- `c2g_members.npz`   arrays stacked over the M ensemble members and B bags per member, F = 4 features:
                      `shrinkage` (M,) str, `coef` (M, B, F), `intercept` (M, B),
                      `prep_fill`, `prep_lo`, `prep_hi`, `prep_center`, `prep_scale` (M, B, F).
- `model_meta.json`   recipe, feature names, threshold, fallback score, QC constants, protocol hashes and
                      aggregate training counts. No subject identifiers.
- `SHA256SUMS`        checksums of the two files above (verified on load).

Loading uses `numpy.load(allow_pickle=False)` and `json`, so a weights folder cannot execute code.
The member order in the shipped file is a random permutation that was not recorded (see ETHICS.md).
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

WEIGHTS_FORMAT = "ptsd_eeg.weights/1"
META_FILE = "model_meta.json"
SUMS_FILE = "SHA256SUMS"
PREP_KEYS = ("fill", "lo", "hi", "center", "scale")
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
"""`weights/` of a source checkout (override with `--weights` or the PTSD_WEIGHTS environment variable)."""

_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: Path | str) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Byte-reproducible `.npz` (fixed zip timestamps, sorted entries, no pickled objects)."""
    with zipfile.ZipFile(path, "w") as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())


def model_to_arrays(model: dict) -> dict[str, np.ndarray]:
    """Stack the members of a model dict into the arrays stored in the `.npz`."""
    members = model["members"]
    if any(m.get("gate") is not None for m in members):
        raise NotImplementedError("members with a Mahalanobis gate (recipe C3) are not supported by this format")
    arrays = dict(shrinkage=np.array([str(m["shrinkage"]) for m in members]),
                  coef=np.stack([np.asarray(m["coef"], float) for m in members]),
                  intercept=np.stack([np.asarray(m["intercept"], float) for m in members]))
    for k in PREP_KEYS:
        arrays[f"prep_{k}"] = np.stack([np.asarray(m["prep"][k], float) for m in members])
    return arrays


def _jsonable(x):
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def save_weights(model: dict, out_dir: Path | str, shuffle: bool = False,
                 rng: np.random.Generator | None = None, extra_meta: dict | None = None) -> dict:
    """Write `<recipe>_members.npz`, `model_meta.json` and `SHA256SUMS` into `out_dir`.

    With `shuffle=True` the member order is permuted by `rng` (default: an unseeded generator) and the
    permutation is not stored anywhere. Returns the metadata dict that was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    members = list(model["members"])
    if shuffle:
        rng = rng if rng is not None else np.random.default_rng()
        members = [members[i] for i in rng.permutation(len(members))]
    arrays = model_to_arrays(dict(model, members=members))
    npz_name = f"{model['recipe'].lower()}_members.npz"
    _write_npz(out_dir / npz_name, arrays)

    keep = ["recipe", "version", "recipe_spec", "feature_names", "threshold", "fallback_score", "qc",
            "n_train", "n_ptsd", "n_negative", "strata_counts", "protocol_sha256", "amendment_sha256"]
    meta = dict(format=WEIGHTS_FORMAT, members_file=npz_name, n_members=len(members),
                n_bags=int(arrays["coef"].shape[1]), n_features=int(arrays["coef"].shape[2]),
                member_order="random permutation, not recorded" if shuffle else "outer LOSO fold order",
                **{k: _jsonable(model[k]) for k in keep if k in model})
    if extra_meta:
        meta.update(_jsonable(extra_meta))
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    sums = "".join(f"{sha256_file(out_dir / n)}  {n}\n" for n in (npz_name, META_FILE))
    (out_dir / SUMS_FILE).write_text(sums, encoding="utf-8")
    return meta


def verify_weights(weights_dir: Path | str) -> dict[str, bool]:
    """Check every file listed in `SHA256SUMS`; returns {file name: matches}."""
    weights_dir = Path(weights_dir)
    out = {}
    for line in (weights_dir / SUMS_FILE).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        out[name] = (weights_dir / name).is_file() and sha256_file(weights_dir / name) == digest
    return out


def load_weights(weights_dir: Path | str | None = None, verify: bool = True) -> dict:
    """Load a weights folder into the model dict used by `model.score_record` / `model.predict`.

    Raises ValueError if `verify` is set and a checksum does not match.
    """
    import os
    weights_dir = Path(weights_dir or os.environ.get("PTSD_WEIGHTS") or DEFAULT_WEIGHTS_DIR)
    if verify:
        bad = [n for n, ok in verify_weights(weights_dir).items() if not ok]
        if bad:
            raise ValueError(f"checksum mismatch in {weights_dir}: {bad}")
    meta = json.loads((weights_dir / META_FILE).read_text(encoding="utf-8"))
    if meta.get("format") != WEIGHTS_FORMAT:
        raise ValueError(f"unsupported weights format {meta.get('format')!r}")
    spec = dict(meta["recipe_spec"])
    clip = tuple(spec["clip"]) if spec.get("clip") is not None else None
    spec["clip"] = clip
    with np.load(weights_dir / meta["members_file"], allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    members = []
    for i in range(a["coef"].shape[0]):
        members.append(dict(shrinkage=str(a["shrinkage"][i]),
                            prep={k: a[f"prep_{k}"][i] for k in PREP_KEYS},
                            coef=a["coef"][i], intercept=a["intercept"][i], clip=clip, gate=None))
    qc = dict(meta.get("qc", {}))
    if "band_hz" in qc:
        qc["band_hz"] = tuple(qc["band_hz"])
    model = {k: v for k, v in meta.items() if k not in ("members_file", "recipe_spec", "qc")}
    model.update(recipe_spec=spec, qc=qc, members=members, weights_dir=str(weights_dir))
    return model


def coefficient_table(model: dict) -> "pd.DataFrame":  # noqa: F821
    """Aggregate view of the ensemble weights, per model input feature.

    Coefficients act on robust-scaled inputs, so they are comparable across features ("standardized").
    Returns, over all members x bags: mean, SD, 2.5/50/97.5 percentiles, share positive; plus the SD of
    the per-member mean (spread across LOSO members) and the mean within-member SD (spread across bags);
    then the 5th/95th percentiles and the range (min, max) of the per-member means.
    """
    import pandas as pd

    from .transforms import C1_FEATURES
    coef = np.stack([np.asarray(m["coef"], float) for m in model["members"]])  # (M, B, F)
    names = C1_FEATURES if model["recipe_spec"].get("subject_transform") == "c1" else list(model["feature_names"])
    flat = coef.reshape(-1, coef.shape[-1])
    rows = []
    for j, name in enumerate(names):
        v = flat[:, j]
        member_mean = coef[:, :, j].mean(axis=1)
        rows.append(dict(feature=name, mean=float(v.mean()), sd=float(v.std(ddof=1)),
                         p2_5=float(np.percentile(v, 2.5)), median=float(np.median(v)),
                         p97_5=float(np.percentile(v, 97.5)), share_positive=float((v > 0).mean()),
                         sd_across_members=float(member_mean.std(ddof=1)),
                         mean_sd_across_bags=float(coef[:, :, j].std(axis=1, ddof=1).mean()),
                         n_members=int(coef.shape[0]), n_bags=int(coef.shape[1]),
                         p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                         member_mean_min=float(member_mean.min()), member_mean_max=float(member_mean.max())))
    return pd.DataFrame(rows)
