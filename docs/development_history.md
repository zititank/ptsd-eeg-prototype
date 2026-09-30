# Development history

How the project got from a hackathon submission to this repository, told as it happened, including the parts that
did not work. Release-level changes are in [CHANGELOG.md](../CHANGELOG.md).

## 1. The hackathon

The task, at an international medical AI hackathon (name withheld under the data agreement): from one-minute
recordings of a 6-channel consumer EEG headband, score how close a person is to a PTSD cohort, while giving low
scores to people with somatoform disorders and to older adults. Scoring combined discrimination (AUC) with the false-positive rate in those two groups on a closed test set
we never saw. Splits had to be by subject.

The hackathon team was Bogdan (zititank), who built the pipeline and presented it at the oral defense, with
AdetyTy and Qwertyqwerty579 (ML and data). Credits: [CONTRIBUTORS.md](../CONTRIBUTORS.md).

Over the hackathon days the model went through three frozen versions, each packaged and reproducible:

| Version | Recipe |
|---|---|
| baseline | Set A features, L2-regularised logistic regression |
| frozen v2 | Set A, nested-LOSO shrinkage LDA with a constrained shrinkage selector |
| frozen v3 (submission) | v2 plus one shared amplitude scale for all channels and a fixed fallback for unscorable recordings |

v3 reached AUC 0.749 on the 90 scorable training subjects. It was reproduced exactly (inference and retraining from
raw EDF) on a second operating system before submission.

## 2. Why v3 could not be trusted

Two audits were run on the frozen v3 before touching the model ([data_integrity.md](data_integrity.md)):

- **Data integrity.** 46 of the 66 training controls come from one recording batch ("control batch B") whose files
  contain no physiological EEG: none of the 46 rest files has an alpha peak.
  Against the 20 valid controls, v3's AUC was 0.410, i.e. chance. EDF header fields alone separated the groups
  (AUC 0.970). The headline AUC was an acquisition artefact.
- **Instability.** Refitting v3 on bootstrap resamples showed that PTSD, valid controls and somatoform patients sit
  in one overlapping cloud around the 0.5 threshold, so their decisions flip between refits (flip rate 0.106). The
  batch B never flips, because it is trivially "not EEG". The inner shrinkage choice was close to arbitrary.
  The 90-member LOSO ensemble hid this, because its members differ by one training subject each.

The audits did not change the conclusion that there is no PTSD signal against valid controls. They changed what
could honestly be done next: any "improvement" could only be judged by a rule fixed in advance.

## 3. Pre-registration and candidates C1 to C4

Before any candidate was fitted, the decision rule was written into the protocol (PROTOCOL_v4) and its SHA-256
was recorded. Candidates:

- **C1**: compositional transforms (CLR, logit) and robust in-fold scaling;
- **C2**: C1 plus stratum-balanced bagging (B = 100) and clipped outputs;
- **C3**: C2 plus an out-of-distribution / signal-QC gate;
- **C4**: C2 with somatoform patients as extra negatives.

A candidate could replace v3 only if its proxy points beat v3 with a paired bootstrap lower bound above zero, its
flip rate was no worse, and it passed safety gates G1 to G4 (leakage, batch table, inference contract,
reproduction). C1 failed the leakage gate on its label-permutation check, C3 and C4 failed the score criterion, and
**C2 passed everything**. It shipped as v4.0. It won by flagging fewer somatoform patients, not by finding PTSD
better; against valid controls it stays at chance (0.435). Details: [model_selection.md](model_selection.md).

## 4. C2G: the minimum-data gate (v4.1)

The inference-contract test showed that C2 gave an ordinary score to a rest file cut to a third of its length.
Amendment 1 to the protocol, hashed before the change was implemented, added a label-free rule: rest shorter than
30 s or truncated relative to its header gets the fallback 24/90. The fold members are identical to C2; the gate
fired for one evaluation-only subject and no metric changed. This is the shipped model, C2G.

## 5. External check

A separate protocol was hashed before downloading a public PTSD EEG dataset (Mendeley,
du Bois et al., CC BY 4.0), and an amendment was hashed after inspecting the file layout but before reading any
severity score. Result: no relation between occipital alpha and PTSD severity, and no relation for the model
score; with n = 15 and eyes-open recordings the check is underpowered ([external_check.md](external_check.md)).

## 6. Hashes as evidence

The order "rule first, results second" is documented by the SHA-256 hashes recorded when each protocol was frozen:

| Document | SHA-256 of the original | Frozen |
|---|---|---|
| PROTOCOL_v4 | `88ef6934f896d83ac2b5fa3e1f7c5a7e3ee6ad1cfe70bc97cf5a6f956a5bb047` | before any candidate was scored |
| PROTOCOL_v4, Amendment 1 | `38ad28a931699b6e3c284fccff8e701df8724bc2f6475254a79ee1f6217c83d8` | before the gate was implemented |
| External protocol | `44d869a0204dc9c9a7888814a0541e945234c9de1f2e14be41cf834a5f01c07a` | before the public dataset was downloaded |
| External protocol, Amendment 1 | `bfa1a059a440a2f6f62d15d2f8c1b086eadb68f9d90e9e769c6c4c7bd56f79eb` | before any severity score was read |

In the private repository, the v4.0 commit (C2 shipped) comes after the protocol freeze, and the v4.1 commits
after the amendment. Two caveats, stated plainly:

- This is self-registration: the hashes and freeze times were recorded by us, not by a public registry or a trusted
  timestamping service. They make silent changes detectable; they do not prove the time to a third party.
- The protocol texts describe the confidential data, so they are not published. The hashes and a summary of the
  decision rule are in [`protocols/README.md`](../protocols/README.md); a reviewer granted access under the data
  agreement can check the originals against them
  ([reproducibility.md](reproducibility.md#pre-registration-hashes)).

## 7. Why this repository starts with a fresh history

The private repository cannot be published, and neither can its git history:

- about a hundred tracked files contain subject codes of the confidential dataset;
- the notebooks carry outputs computed on real data;
- some tables carry recording dates;
- the pickled model stores the list of training subject codes;
- git keeps all of this in every past commit, so deleting files today would not remove it.

Rewriting that history reliably is harder and riskier than starting over. This repository was therefore built from
an allowlist export: only files that were explicitly selected, cleaned and scanned were copied, and the history
starts at the public release. The commits here are real and carry the times at which they were made. **No commit is
backdated** to imitate the development timeline: the timeline above is documented by the protocol hashes and by
this page, not by fabricated git dates. The model and every number are the same as in the private 4.1.0; what
changed is packaging, documentation, privacy and confidentiality ([CHANGELOG.md](../CHANGELOG.md)).

Development was heavily AI-assisted; see [ai_assistance.md](ai_assistance.md).
