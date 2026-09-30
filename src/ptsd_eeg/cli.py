"""Command line interface (`ptsd-eeg`, or `python -m ptsd_eeg.cli`).

    ptsd-eeg predict [--weights DIR] INPUT_DIR OUTPUT_CSV     score every subject folder of INPUT_DIR
    ptsd-eeg verify-weights [--weights DIR]                   check SHA256SUMS and print the metadata
    ptsd-eeg train --data DIR [--recipe C2G] --out DIR         nested-LOSO training -> a weights folder
    ptsd-eeg reproduce --data DATA_ROOT [--models ...] [--flip] [--out DIR]
                                                              retrain from raw data, compare with results/
    ptsd-eeg gates --data DATA_ROOT [--models v3]             leakage gates G1a/G1b

Research prototype: the output is a model score, not a diagnosis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _predict_dir(weights: Path | None, input_dir: Path, output_csv: Path) -> int:
    import pandas as pd

    from . import model as M
    from .weights import load_weights
    model = load_weights(weights)
    rows = []
    for s in sorted(p for p in Path(input_dir).iterdir() if p.is_dir()):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = M.predict(model, s)
        except Exception as exc:  # noqa: BLE001 - one subject must not stop the run
            print(f"{s.name}: fallback ({type(exc).__name__}: {exc})", file=sys.stderr)
            p = float(model["fallback_score"])
        rows.append(dict(subject_id=s.name, ptsd_probability=p))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["subject_id", "ptsd_probability"]).to_csv(output_csv, index=False)
    print(f"scored {len(rows)} subject folders -> {output_csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ptsd-eeg", description="Rest-EEG PTSD research prototype (not a medical device).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="score every subject folder of INPUT_DIR")
    p.add_argument("--weights", type=Path, default=None, help="weights folder (default: repository weights/)")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_csv", type=Path)
    v = sub.add_parser("verify-weights", help="check SHA256SUMS of a weights folder")
    v.add_argument("--weights", type=Path, default=None)
    t = sub.add_parser("train", help="nested-LOSO training from raw group folders (see ptsd_eeg.layout)")
    t.add_argument("--data", type=Path, required=True, help="folder containing the control and PTSD group folders")
    t.add_argument("--recipe", default="C2G", choices=["v3", "C1", "C2", "C4", "C2G"])
    t.add_argument("--out", type=Path, required=True, help="output weights folder")
    t.add_argument("--jobs", type=int, default=1)
    t.add_argument("--shuffle-members", action="store_true", help="store members in a random, unrecorded order")
    r = sub.add_parser("reproduce", help="retrain from raw data and compare with results/metrics.csv")
    r.add_argument("--data", type=Path, required=True, help="data root (see DATA_ACCESS.md)")
    r.add_argument("--models", nargs="+", default=["v3", "C2", "C2G"])
    r.add_argument("--out", type=Path, default=Path(".reproduce"))
    r.add_argument("--flip", action="store_true", help="also run the 200-refit flip-rate bootstrap (slow)")
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--save-predictions", action="store_true", help="write per-subject predictions (never commit)")
    g = sub.add_parser("gates", help="leakage gates G1a (in-fold) and G1b (label permutation)")
    g.add_argument("--data", type=Path, required=True)
    g.add_argument("--models", nargs="+", default=["v3"])
    g.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")

    if args.cmd == "predict":
        return _predict_dir(args.weights, args.input_dir, args.output_csv)
    if args.cmd == "verify-weights":
        from .weights import DEFAULT_WEIGHTS_DIR, verify_weights
        wd = args.weights or Path(os.environ.get("PTSD_WEIGHTS") or DEFAULT_WEIGHTS_DIR)
        res = verify_weights(wd)
        for name, ok in res.items():
            print(f"{'OK  ' if ok else 'FAIL'} {name}")
        meta = json.loads((wd / "model_meta.json").read_text(encoding="utf-8"))
        print(json.dumps({k: meta[k] for k in ["recipe", "n_members", "n_bags", "n_train", "n_ptsd", "n_negative",
                                               "threshold", "fallback_score", "member_order"] if k in meta}, indent=2))
        return 0 if all(res.values()) else 1
    if args.cmd == "train":
        from . import model as M
        from .reproduce import PROTOCOL_SHA256
        _, oof, _ = M.train_from_dir(args.data, args.recipe, args.out, n_jobs=args.jobs,
                                     shuffle_members=args.shuffle_members, protocol_sha256=PROTOCOL_SHA256)
        print(f"trained {args.recipe}: {len(oof)} training subjects -> {args.out}")
        return 0
    if args.cmd == "reproduce":
        from .reproduce import reproduce
        s = reproduce(args.data, args.out, models=args.models, flip=args.flip, n_jobs=args.jobs,
                      save_predictions=args.save_predictions)
        print(json.dumps(s, indent=2, default=str))
        return 0 if s.get("passed", True) else 1
    if args.cmd == "gates":
        from .cohort import build_cohort
        from .gates import infold_check, permutation_check
        cohort = build_cohort(args.data, n_jobs=args.jobs)
        ok = True
        for m in args.models:
            a = infold_check(m, cohort)
            b = permutation_check(m, cohort, n_jobs=args.jobs)
            print(m, dict(G1a_infold=a["passed"], G1b_permutation=b["passed"], permuted_auc=round(b["permuted_auc"], 4),
                          null_95=[round(x, 4) for x in b["null_95"]]))
            ok = ok and a["passed"] and b["passed"]
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
