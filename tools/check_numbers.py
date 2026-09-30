"""Check the numbers in the docs against the aggregate results shipped in this repository.

Three passes:
  1. consistency: the coefficient and shrinkage tables in results/ match the public weights in weights/;
  2. strict: each key number is recomputed from results/ (or weights/), formatted as written, and must appear in
     the named doc(s);
  3. pool: every decimal number in every top-level and docs/ Markdown file must match a value from results/
     (any rounding to 0-4 decimals), protocols/README.md, a reviewed literature constant below, or a protocol
     constant. Unmatched numbers are listed for manual review; they do not fail the run.

Needs only the repository (no confidential data). Prints aggregates only.
Run: python tools/check_numbers.py    (exit 1 on any consistency or strict failure)
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
sys.path.insert(0, str(REPO / "src"))


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


M = {r["model"]: r for r in rows(RES / "metrics.csv")}
CRITERIA = {r["model"]: r for r in rows(RES / "criteria.csv")}
BATCH_TABLE = {r["batch"]: r for r in rows(RES / "C2G_batch_table.csv")}
EXT = rows(RES / "external_results.csv")
EXT_META = json.loads((RES / "external_run_meta.json").read_text(encoding="utf-8"))
WEQ = json.loads((RES / "weights_equivalence.json").read_text(encoding="utf-8"))
assert WEQ["n_decision_changed_at_0_5"] == 0
NEG = {r["name"]: r for r in rows(RES / "negative_controls.csv")}
SENSITIVITY = {(r["model"], r["analysis"][:1]): r for r in rows(RES / "sensitivity_analyses.csv")}
GATE_ROWS = {r["model"]: r for r in rows(RES / "candidate_gates.csv")}
PER_BATCH = {r["source"]: r for r in rows(RES / "audit" / "per_batch_auc_fpr.csv")}
COEF_ROWS = {r["feature"]: r for r in rows(RES / "c2g_coefficients.csv")}
SHRINK = {r["shrinkage"]: r for r in rows(RES / "shrinkage_selection.csv")}
N_GATED = int(float(M["C2G"]["n_min_data_gated"]))

with np.load(REPO / "weights" / "c2g_members.npz", allow_pickle=False) as _z:
    NPZ_COEF = _z["coef"]                   # (members, bags, features)
    NPZ_INTERCEPT = _z["intercept"]         # (members, bags)
    NPZ_SHRINK = collections.Counter(str(s) for s in _z["shrinkage"])
INTERCEPT_MEAN = float(NPZ_INTERCEPT.mean())

# ---- pass 1: results tables vs public weights -----------------------------------------------------
consistency_fail = 0


def consistent(label: str, ok: bool, detail: str = "") -> None:
    global consistency_fail
    consistency_fail += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {label:44s} {detail}")


from ptsd_eeg.weights import coefficient_table, load_weights  # noqa: E402

_table = coefficient_table(load_weights(REPO / "weights"))
_worst = 0.0
for _, r in _table.iterrows():
    for col, v in r.items():
        if col != "feature" and col in COEF_ROWS[r.feature]:
            _worst = max(_worst, abs(float(COEF_ROWS[r.feature][col]) - float(v)))
consistent("results/c2g_coefficients.csv vs weights/", _worst < 1e-12, f"max abs diff {_worst:.1e}")
consistent("results/shrinkage_selection.csv vs weights/",
           {k: int(v["members_selecting"]) for k, v in SHRINK.items()} == dict(NPZ_SHRINK), str(dict(NPZ_SHRINK)))


def ext(analysis: str, feature: str, stat: str = "device-stratified Spearman") -> dict:
    return next(r for r in EXT if r["analysis"].startswith(analysis) and r["feature"] == feature and r["stat"] == stat)


def f(x, d=3, sign=False) -> str:
    v = float(x)
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "−")


def ci(lo, hi, d=3, sign=False) -> str:
    return f"[{f(lo, d, sign)}, {f(hi, d, sign)}]"


def frac(rate, n) -> str:
    k = round(float(rate) * n)
    assert abs(k / n - float(rate)) < 1e-9, (rate, n)
    return f"{k}/{n}"


def pct(x, d=1) -> str:
    return f"{100 * float(x):.{d}f}%"


c2g, v3, c2 = M["C2G"], M["v3"], M["C2"]
sch, sdo = M["C2_plus_schulte"], M["schulte_duration_only"]
age_all = M["age_ridge_all_controls"]
g1 = GATE_ROWS["C2"]  # C2G ships the C2 members; the gates were run on C2

README, CARD_DOC, ETH, DIS = "README.md", "MODEL_CARD.md", "ETHICS.md", "DISCLAIMER.md"
SCI, SEL, LIM, EXT_DOC = "docs/science.md", "docs/model_selection.md", "docs/limitations.md", "docs/external_check.md"
INT, REPRO_DOC, REF = "docs/data_integrity.md", "docs/reproducibility.md", "docs/references.md"
FEATURES = ["clr_alpha_O", "logit_alpha_IAF", "rest_IAF", "clr_beta_O"]

CHECKS: list[tuple[str, str, list[str]]] = [
    # headline metrics, shipped model
    ("AUC C2G", f(c2g["auc"]) + " " + ci(c2g["auc_lo"], c2g["auc_hi"]), [README, CARD_DOC, SCI, LIM]),
    ("AUC vs valid", f(c2g["auc_vs_CTRL_other"]) + " " + ci(c2g["auc_vs_CTRL_other_lo"], c2g["auc_vs_CTRL_other_hi"]),
     [README, CARD_DOC, SCI, LIM, ETH, DIS]),
    ("AUC vs batch B", f(c2g["auc_vs_CTRL_B"]) + " " + ci(c2g["auc_vs_CTRL_B_lo"], c2g["auc_vs_CTRL_B_hi"]),
     [README, CARD_DOC, SCI, LIM]),
    ("sensitivity C2G", frac(c2g["sensitivity"], 24), [README, CARD_DOC, SCI, SEL, LIM]),
    ("sensitivity v3", frac(v3["sensitivity"], 24), [README, SCI, SEL, LIM]),
    ("soma+elderly FPR C2G", frac(c2g["fpr_soma_elderly"], 97), [README, CARD_DOC, SCI, SEL, LIM]),
    ("soma+elderly FPR CI", ci(c2g["fpr_soma_elderly_lo"], c2g["fpr_soma_elderly_hi"]), [README, CARD_DOC, SCI, LIM]),
    ("soma+elderly FPR v3", frac(v3["fpr_soma_elderly"], 97), [README, SCI, SEL]),
    ("BA C2G", f(c2g["ba"]) + " " + ci(c2g["ba_lo"], c2g["ba_hi"]), [CARD_DOC, SCI]),
    ("BA v3", f(v3["ba"]) + " " + ci(v3["ba_lo"], v3["ba_hi"]), [SCI]),
    ("specificity C2G", f(c2g["specificity"]), [CARD_DOC]),
    ("proxy C2G", f(c2g["proxy_points"], 2) + " " + ci(c2g["proxy_lo"], c2g["proxy_hi"], 2), [CARD_DOC, SCI]),
    ("proxy v3", f(v3["proxy_points"], 2), [README, SCI, SEL]),
    ("delta proxy", f(c2g["delta_proxy"], 2, True) + " " + ci(c2g["delta_proxy_lo"], c2g["delta_proxy_hi"], 2, True),
     [README, SCI, SEL]),
    ("flip C2G", f(c2g["flip_rate"]), [README, CARD_DOC, SCI, SEL]),
    ("flip v3", f(v3["flip_rate"]), [README, SCI, SEL]),
    ("soma FPR C2G", f(c2g["fpr_soma"]), [SCI, ETH]),
    ("soma FPR v3", f(v3["fpr_soma"]), [SCI, REF]),
    ("v3 AUC", f(v3["auc"]) + " " + ci(v3["auc_lo"], v3["auc_hi"]), [README, SCI]),
    ("v3 AUC vs valid", f(v3["auc_vs_CTRL_other"]) + " " + ci(v3["auc_vs_CTRL_other_lo"], v3["auc_vs_CTRL_other_hi"]),
     [README, SCI, INT]),
    ("v3 AUC vs batch B", f(v3["auc_vs_CTRL_B"]) + " " + ci(v3["auc_vs_CTRL_B_lo"], v3["auc_vs_CTRL_B_hi"]),
     [README, SCI, INT]),
    # task-duration feature and age
    ("C2+task duration AUC", f(sch["auc"]) + " " + ci(sch["auc_lo"], sch["auc_hi"]), [SCI]),
    ("C2+task duration vs valid", f(sch["auc_vs_CTRL_other"]), [SCI]),
    ("task duration elderly FPR", f(sch["fpr_elderly"]), [README, SCI, LIM, ETH, REF]),
    ("duration-only elderly FPR", f(sdo["fpr_elderly"]), [SCI]),
    ("duration-only AUC", f(sdo["auc"]) + " " + ci(sdo["auc_lo"], sdo["auc_hi"]), [SCI]),
    ("duration-only AUC vs valid", f(sdo["auc_vs_CTRL_other"]), [SCI, REF]),
    ("rho score vs task duration", f"ρ {f(c2g['rho_score_vs_schulte'])}, p = {f(c2g['rho_score_vs_schulte_p'])}, "
                             f"n = {int(float(c2g['rho_score_vs_schulte_n']))}", [SCI]),
    ("age ridge all controls", f"R² {f(age_all['r2'])} {ci(age_all['r2_lo'], age_all['r2_hi'])}", [CARD_DOC, ETH, LIM]),
    # gates and reproducibility
    ("retrain diff", f"{float(c2g['final_gate_retrain_oof_max_abs_diff']):.1e}", [REPRO_DOC, SEL]),
    ("external inference diff", f"{float(c2g['final_gate_external_max_abs_diff']):.1e}", [REPRO_DOC, SEL]),
    ("external n", f"n = {int(float(c2g['final_gate_external_n']))}", [REPRO_DOC]),
    ("clean venv diff", f"{float(c2['final_gate_clean_venv_max_abs_diff']):.1f}", [README, REPRO_DOC]),
    ("contract", f"{int(float(c2g['final_gate_contract_n_pass']))}/14", [SEL, REPRO_DOC]),
    ("public weights vs pickle", f"{WEQ['max_abs_diff']:.1e}", [REPRO_DOC, ETH]),
    ("permuted AUC", f(g1["permuted_auc"]), [README, SCI, SEL, REF]),
    ("null range", ci(g1["null_95_lo"], g1["null_95_hi"]), [README, SCI, SEL, REF]),
    ("protocol sha", c2g["protocol_sha256"], [SEL]),
    ("amendment sha", c2g["amendment_sha256"], [SEL]),
    ("protocol sha short", c2g["protocol_sha256"][:8] + "…" + c2g["protocol_sha256"][-4:], [SCI, CARD_DOC]),
    ("amendment sha short", c2g["amendment_sha256"][:8] + "…" + c2g["amendment_sha256"][-4:], [SCI, CARD_DOC]),
    ("gate fired", f"{N_GATED} subject", [SCI, SEL]),
    ("header-only AUC", f(NEG["header_only"]["auc"]), [SCI, CARD_DOC, INT, LIM]),
    ("control-batch EEG AUC", f(NEG["control_batch_EEG"]["auc"], 2), [INT]),
    # batch table
    ("valid controls flagged", f"{BATCH_TABLE['CTRL_other']['n_positive']} of {BATCH_TABLE['CTRL_other']['n_scorable']}", [CARD_DOC, LIM]),
    ("PTSD flagged", f"{BATCH_TABLE['PTSD']['n_positive']} of {BATCH_TABLE['PTSD']['n_scorable']}", [CARD_DOC]),
    # candidates (criteria.csv)
    *[(f"{m} proxy", f(CRITERIA[m]["proxy_points"], 2), [SCI, SEL]) for m in ["C1", "C2", "C3", "C4"]],
    *[(f"{m} delta", f(CRITERIA[m]["delta_proxy"], 2, True) + " " + ci(CRITERIA[m]["delta_proxy_p2_5"], CRITERIA[m]["delta_proxy_p97_5"], 2, True),
       [SCI, SEL]) for m in ["C1", "C2", "C3", "C4"]],
    *[(f"{m} flip", f(CRITERIA[m]["flip_rate"], 4), [SCI, SEL]) for m in ["C1", "C2", "C3", "C4"]],
    ("v3 flip 4dp", f(CRITERIA["C2"]["v3_flip_rate"], 4), [SCI, SEL]),
    ("flip limit", f(CRITERIA["C2"]["flip_limit"], 4), [SEL]),
    ("C3 AUC", f(M["C3"]["auc"]), [SEL]),
    ("C3 gated share", f(M["C3"]["gated_share_scorable_full"]), [SEL]),
    # sensitivity analyses (C2)
    ("sens a FPR", f(SENSITIVITY[("C2", "a")]["fpr_soma_elderly"]), [SEL]),
    ("sens a proxy", f(SENSITIVITY[("C2", "a")]["proxy_points"], 2), [SEL]),
    ("sens b ctrl C O1", f(SENSITIVITY[("C2", "b")]["fpr_ctrl_c"]) + " (" + frac(SENSITIVITY[("C2", "b")]["fpr_ctrl_c"], 40) + ")", [SEL]),
    ("sens c soma", f(SENSITIVITY[("C2", "c")]["fpr_soma_unique"]), [SEL]),
    ("sens c soma+eld", f(SENSITIVITY[("C2", "c")]["fpr_soma_elderly"]), [SEL]),
    ("sens c proxy", f(SENSITIVITY[("C2", "c")]["proxy_points"], 2), [SEL]),
    # audit
    ("refit without batch B", f(PER_BATCH["metrics.csv:v3_without_CTRL_B"]["auc"]) + " "
     + ci(PER_BATCH["metrics.csv:v3_without_CTRL_B"]["auc_lo"], PER_BATCH["metrics.csv:v3_without_CTRL_B"]["auc_hi"]), [INT]),
    ("refit without other", f(PER_BATCH["metrics.csv:v3_without_other"]["auc"]), [INT]),
    # external check
    ("ext primary", f"{f(ext('primary', 'rest_rel_alpha_O')['rho'], 2)} "
                    f"{ci(ext('primary', 'rest_rel_alpha_O')['ci_lo'], ext('primary', 'rest_rel_alpha_O')['ci_hi'], 2)}",
     [README, CARD_DOC, LIM, REF]),
    ("ext primary table", f"{f(ext('primary', 'rest_rel_alpha_O')['rho'], 2)} | "
                          f"{ci(ext('primary', 'rest_rel_alpha_O')['ci_lo'], ext('primary', 'rest_rel_alpha_O')['ci_hi'], 2)}",
     [EXT_DOC]),
    ("ext primary p", f"p = {f(ext('primary', 'rest_rel_alpha_O')['p'], 2)}", [CARD_DOC, LIM, EXT_DOC]),
    ("ext n", f"n = {ext('primary', 'rest_rel_alpha_O')['n']}", [README, CARD_DOC, LIM, EXT_DOC]),
    ("ext C2 score", f(ext("C2 model score", "c2_score")["rho"], 3, True), [CARD_DOC, EXT_DOC]),
    ("ext C2 score p", f"p = {f(ext('C2 model score', 'c2_score')['p'], 2)}", [CARD_DOC, EXT_DOC]),
    ("ext beta", f(ext("secondary", "rest_rel_beta_O")["rho"], 2) + " | "
     + ci(ext("secondary", "rest_rel_beta_O")["ci_lo"], ext("secondary", "rest_rel_beta_O")["ci_hi"], 2), [EXT_DOC]),
    ("ext IAF", f(ext("secondary", "rest_IAF")["rho"], 2, True) + " | "
     + ci(ext("secondary", "rest_IAF")["ci_lo"], ext("secondary", "rest_IAF")["ci_hi"], 2), [EXT_DOC]),
    ("ext first 60s", f"ρ {f(ext('sensitivity: first 60 s', 'rest_rel_alpha_O')['rho'], 2)} "
                      f"(p = {f(ext('sensitivity: first 60 s', 'rest_rel_alpha_O')['p'], 2)})", [EXT_DOC]),
    ("ext PO7", f"ρ {f(ext('sensitivity: g.tec', 'rest_rel_alpha_O')['rho'], 2)} "
                f"(p = {f(ext('sensitivity: g.tec', 'rest_rel_alpha_O')['p'], 2)}", [EXT_DOC]),
    ("ext later", f"ρ {f(ext('sensitivity: + later', 'rest_rel_alpha_O')['rho'], 2)} "
                  f"(p = {f(ext('sensitivity: + later', 'rest_rel_alpha_O')['p'], 2)}, n = 18)", [EXT_DOC]),
    ("ext IAF 60s", f"ρ {f(ext('sensitivity: first 60 s', 'rest_IAF')['rho'], 2)}, "
                    f"p = {f(ext('sensitivity: first 60 s', 'rest_IAF')['p'], 2)}", [EXT_DOC]),
    ("ext MDE", f"about {f(EXT_META['mde_one_sided'], 2)}", [EXT_DOC, LIM]),
    ("ext boot/perm", f"{EXT_META['n_boot']} bootstrap and {EXT_META['n_perm']} permutation", [EXT_DOC]),
    # coefficients (results/c2g_coefficients.csv, itself checked against weights/ above)
    *[(f"coef {k}", f(COEF_ROWS[k]["mean"], 2, True), [README, SCI]) for k in FEATURES],
    *[(f"share {k}", pct(COEF_ROWS[k]["share_positive"]), [README, SCI]) for k in FEATURES],
    *[(f"sd {k}", f"| {f(COEF_ROWS[k]['sd'], 2)} |", [SCI]) for k in FEATURES],
    *[(f"p5-p95 {k}", f"{f(COEF_ROWS[k]['p5'], 2, True)} to {f(COEF_ROWS[k]['p95'], 2, True)}", [SCI]) for k in FEATURES],
    *[(f"member range {k}", f"{f(COEF_ROWS[k]['member_mean_min'], 2, True)} to {f(COEF_ROWS[k]['member_mean_max'], 2, True)}", [SCI])
      for k in FEATURES],
    ("intercept", f"intercept {f(INTERCEPT_MEAN, 2)}", [SCI]),
    ("shrinkage counts", "`auto` {} times, 0.1 {}, 0.05 {}, 0.2 {}, 0.3 {}, 0.5 {}".format(
        *(SHRINK[k]["members_selecting"] for k in ["auto", "0.1", "0.05", "0.2", "0.3", "0.5"])), [SCI]),
]

DOC_FILES = sorted([p for p in REPO.glob("*.md")] + [p for p in (REPO / "docs").glob("*.md")])
texts = {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8") for p in DOC_FILES}


def contains(doc: str, needle: str) -> bool:
    t = texts[doc]
    if " | " in needle:  # table-cell pair: both parts on one line
        a, b = needle.split(" | ")
        return any(a in line and b in line for line in t.splitlines())
    flat = re.sub(r"\s+", " ", t.replace("\n> ", " "))  # wrapped lines count as one
    return needle in t or needle in flat


# ---- pass 2: strict --------------------------------------------------------------------------------
print()
fail = 0
for label, needle, docs in CHECKS:
    missing = [d for d in docs if not contains(d, needle)]
    fail += bool(missing)
    status = "OK  " if not missing else "FAIL"
    print(f"{status} {label:28s} {needle!s:60.60s} " + (f"missing in {missing}" if missing else f"({len(docs)} docs)"))

# ---- pass 3: pool ----------------------------------------------------------------------------------
pool: set[str] = set()


def add_num(x) -> None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return
    for d in range(0, 5):
        pool.add(f"{abs(v):.{d}f}")
        pool.add(f"{abs(100 * v):.{d}f}")


for src in sorted(RES.rglob("*.csv")):
    if src.name == "MANIFEST.csv":
        continue
    for r in rows(src):
        for v in r.values():
            add_num(v)
for src in sorted(RES.rglob("*.json")):
    for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", src.read_text(encoding="utf-8")):
        add_num(v)
for v in [INTERCEPT_MEAN, NPZ_INTERCEPT.std(ddof=1), *np.percentile(NPZ_INTERCEPT, [5, 95])]:
    add_num(v)

lit = set()
for p in sorted((REPO / "protocols").rglob("*.md")):
    lit |= {m.replace(",", ".") for m in re.findall(r"\d+[.,]\d+", p.read_text(encoding="utf-8"))}

# Numbers quoted from the literature that no shipped file contains; each was matched against its source.
LIT = {"10.07", "9.24", "1.02"}  # Kovacevic et al. [H1]: eyes-closed IAF, PTSD vs controls (Hz), effect size
# protocol constants, versions, citation volume/page numbers and similar non-result numbers
CONSTANTS = {"0.5", "0.02", "0.98", "0.05", "0.10", "0.20", "0.30", "0.50", "0.70", "0.90", "0.690", "1.5", "0.1", "0.2",
         "0.3", "0.267", "0.2667", "0.8", "1.0", "4.1", "4.0", "4.1.0", "4.1.1", "3.12", "3.12.13", "1.26.4", "3.0.5", "1.17.1",
         "1.9.0", "0.0", "0.001", "0.25912", "10.17632", "0.85", "0.9", "0.6", "0.4", "12.5", "1.2.0", "3.11",
         "1.17"}

num_re = re.compile(r"(?<![\w./-])[−+-]?(\d+\.\d+)(?![\w/])")
unknown: dict[str, set[str]] = {}
n_seen = 0
for doc, t in texts.items():
    t = re.sub(r"doi:\S+|https?://\S+|`[0-9a-f]{8,}[^`]*`|\b[0-9a-f]{64}\b|PMID \d+|PMC\d+", " ", t)
    for m in num_re.finditer(t):
        n = m.group(1)
        n_seen += 1
        if n in pool or n in lit or n in LIT or n in CONSTANTS:
            continue
        unknown.setdefault(doc, set()).add(n)

print()
print(f"consistency checks: {2 - consistency_fail}/2 passed")
print(f"strict checks: {len(CHECKS) - fail}/{len(CHECKS)} passed")
n_unknown = sum(len(v) for v in unknown.values())
print(f"pool check: {n_seen} decimal numbers scanned in {len(texts)} docs; "
      f"{n_unknown} not found in any source (review manually)")
for doc, nums in sorted(unknown.items()):
    print(f"  {doc}: {' '.join(sorted(nums))}")
sys.exit(1 if (fail or consistency_fail) else 0)
