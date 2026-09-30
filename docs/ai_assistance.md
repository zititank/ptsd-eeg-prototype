# AI assistance disclosure

**Development of this project was heavily AI-assisted.** Most code, audits, protocols, analyses and documents were
drafted by AI coding agents working inside the author's editor, under the author's direction. This page says what
the AI did, what safeguards do not depend on trusting it, and what the author reviewed personally.

## What the AI agents did

- Refactored the hackathon model (v3) into the `ptsd_eeg` package and wrote the tests.
- Wrote and ran the data-integrity and instability audits.
- Drafted the pre-registered protocols (PROTOCOL_v4, Amendment 1, the external-check protocol) for the author to
  approve before they were hashed.
- Implemented candidates C1–C4 and C2G, the evaluation, the bootstrap and the safety gates.
- Ran the external check and the exploratory analyses.
- Searched, verified and summarised the literature (DOIs resolved via Crossref, findings read in Europe PMC or PubMed).
- Translated and condensed the original project documents into these English docs, and drafted this repository's
  documentation, figures and privacy tooling.

Tools: Cursor IDE with LLM coding agents.

## Safeguards that do not rely on trusting the AI

- **Pre-registration by hash.** The decision rule and the gate were hashed before results existed; the hashes are in
  [`protocols/`](../protocols/). An agent could not quietly move the goalposts afterwards.
- **Numbers are checked by script.** Every key number in the docs is compared with the aggregate CSVs
  ([reproducibility.md](reproducibility.md)).
- **Bit-level reproduction.** Retrain within 5.0e-16, clean-environment difference 0.0, and a 14/14 inference
  contract.
- **Automated privacy and confidentiality scan** over files, notebook outputs and screenshot logs, with 0 hits before
  release.
- **Literature claims are traceable.** Every citation in [references.md](references.md) states the claim and how it
  was used; unverifiable statements were not used.

## What the author reviewed

I audited the pipeline while preparing for the oral defense, because I had to be able to explain and defend every
part of it. I went through the safety nets (subject-level splits, in-fold fitting, the leakage and permutation
tests, the minimum-data gate), checked that the metrics and their confidence intervals mean what the docs say, and
followed the nested LOSO from the outer fold down to the shipped ensemble. Before publishing I went through the
confidentiality rules of the data agreement and what this repository may and may not say. I did not re-derive
every line of the bootstrap and plotting code by hand; there I rely on the tests, the number checker and the
bit-level reproduction above.

— Bogdan (zititank)

## Known risks of AI-assisted research, and how they apply here

- **Plausible but wrong text.** Mitigated by the number checker and by linking every claim to a file. Residual risk
  remains in prose that contains no numbers.
- **Over-engineering.** Parts of the tooling (gates, amendments) are heavier than a hackathon needs; they are kept
  because they make the negative result credible.
- **Citation errors.** References were verified against Crossref and Europe PMC or PubMed; a few entries are marked
  in [references.md](references.md) where only partial records were available.
