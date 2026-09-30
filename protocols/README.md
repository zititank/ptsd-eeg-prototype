# Pre-registration

The analysis followed four pre-registration documents. Each was written and hashed (SHA-256) before the step it
governs was run. The full texts describe the confidential dataset in detail, so they are not published; they are
kept privately by the author (see [DATA_ACCESS.md](../DATA_ACCESS.md)).
The hashes below let anyone who later sees the originals confirm that they were not edited.

| Document | Governs | Frozen before | SHA-256 of the original |
|---|---|---|---|
| `PROTOCOL_v4.md` | candidates C1–C4, evaluation, decision rule | any candidate metric was computed | `88ef6934f896d83ac2b5fa3e1f7c5a7e3ee6ad1cfe70bc97cf5a6f956a5bb047` |
| `PROTOCOL_v4_AMENDMENT_1.md` | minimum-data gate, recipe C2G (release v4.1) | the gate was implemented or any subject checked against it | `38ad28a931699b6e3c284fccff8e701df8724bc2f6475254a79ee1f6217c83d8` |
| `PROTOCOL_EXTERNAL.md` | external direction check on public data | any external dataset was downloaded | `44d869a0204dc9c9a7888814a0541e945234c9de1f2e14be41cf834a5f01c07a` |
| `PROTOCOL_EXTERNAL_AMENDMENT_1.md` | dataset facts found during access | any PCL-5 value was read | `bfa1a059a440a2f6f62d15d2f8c1b086eadb68f9d90e9e769c6c4c7bd56f79eb` |

The protocol hash is also stored in every row of [`results/metrics.csv`](../results/metrics.csv) (`protocol_sha256`)
and in [`weights/model_meta.json`](../weights/model_meta.json).

## PROTOCOL_v4 in short

- **Fixed ingredients.** The previous model (v3) cohort unchanged: 90 scorable training subjects (24 PTSD, 66
  controls). Set A features only (`rest_IAF`, `rest_rel_alpha_O`, `rest_rel_alpha_IAF`, `rest_rel_beta_O`); no
  feature selection, no new features, no header or duration feature, no batch correction (batch is confounded with
  the label). Outer loop: leave-one-subject-out (LOSO); inner loop: stratified 5-fold selection of the LDA shrinkage.
  Threshold 0.5; unscorable subjects get the fallback 24/90.
- **Candidates (at most four).** C1: robust transforms (centred log-ratio of occipital alpha/beta, logit of IAF-band
  alpha, winsorising, robust scaling). C2: C1 plus stratum-balanced bagging (B = 100) so that control batch B cannot
  dominate the weights. C3: C2 plus an out-of-distribution / QC gate at scoring time. C4: C2 with somatoform subjects
  added as negatives.
- **Evaluation.** Out-of-fold AUC on the 90 training subjects with a 2000-replicate subject bootstrap stratified by
  batch; false-positive rate on the somatoform and 65+ groups (never trained on); proxy points
  = 30 · max(0, 2·AUC − 1) + 20 · (1 − FPR); flip rate from 200 bootstrap refits of the whole procedure.

## Decision rule (fixed in advance)

A candidate replaces v3 only if **all** of these hold:

1. its proxy points beat v3, with the 2.5th percentile of the paired bootstrap difference above zero;
2. its flip rate is at most v3's flip rate plus 0.005 (Monte Carlo tolerance);
3. it passes the safety gates: every learnt object fitted in-fold, a label-permutation AUC inside the recorded null
   range, the per-batch table produced, the 14-case inference contract, and inference / retrain reproduction within
   1e-9.

If several pass, the highest proxy estimate wins (exact ties go to the lower C-number). If none passes, v3 ships.
No thresholds, grids, bag counts or strata may be re-tuned after results are seen. Whatever ships, the report must
state the AUC against the physiologically valid controls next to the headline AUC.

**Amendment 1** adds a label-free minimum-data gate (rest recording shorter than 30 s, or truncated relative to its
header, gets the fallback 24/90) as recipe C2G. Training is unchanged; the amendment is judged on safety acceptance
only, never on metrics.

**External protocol.** One-sided Spearman test that occipital relative alpha falls with PCL-5 severity on public
data, with the frozen feature code and device emulation. Results: [docs/external_check.md](../docs/external_check.md).

Outcomes of the rule: [docs/model_selection.md](../docs/model_selection.md).
