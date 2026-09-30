# TRIPOD+AI checklist

TRIPOD+AI (Collins et al., *BMJ* 2024;385:e078378 [T1]) lists 27 reporting items for prediction models that use
machine learning. This is a research prototype, not a clinical prediction model, so some items apply only in part.
Status: **yes** = reported, **partly** = reported with a stated gap, **n/a** = not applicable, with the reason.

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Title: identifies a prediction-model study, the target population and the outcome | yes | [README](../README.md) header and TL;DR |
| 2 | Abstract | yes | [README](../README.md) TL;DR |
| 3 | Background: healthcare context, rationale, existing models | yes | [science.md §1](science.md), [references.md](references.md), [limitations.md](limitations.md#context-what-others-achieved) |
| 4 | Objectives, including development and/or evaluation | yes | [science.md §1](science.md) |
| 5 | Data sources, dates of collection | partly: data source, recording details and dates withheld under the data agreement | [DATA_ACCESS.md](../DATA_ACCESS.md), [data_integrity.md](data_integrity.md) |
| 6 | Participants: setting, eligibility, treatments | partly: groups and sizes given; medication and treatment unknown | [MODEL_CARD.md](../MODEL_CARD.md), [limitations.md](limitations.md) items 5 and 10 |
| 7 | Data preparation and quality checks | yes | [data_integrity.md](data_integrity.md), [pipeline.md](pipeline.md) |
| 8 | Outcome: definition, how and when assessed, blinding | partly: label = clinical group folder (verified diagnosis per the data provider); no severity scores; assessment not blinded to us | [MODEL_CARD.md](../MODEL_CARD.md) |
| 9 | Predictors: definition, measurement, rationale | yes | [science.md §3–4](science.md), `src/ptsd_eeg/features.py` |
| 10 | Sample size justification | partly: fixed by the task; power stated | [limitations.md](limitations.md) item 3 |
| 11 | Missing data handling | yes: missing IAF imputed in-fold; unscorable → fallback | [science.md §4–5](science.md), [pipeline.md](pipeline.md) |
| 12 | Analytical methods: model type, tuning, internal validation, performance measures, model comparison | yes | [science.md §5–8](science.md), [model_selection.md](model_selection.md) |
| 13 | Class imbalance handling | yes: stratum-balanced bagging; empirical fold priors | [science.md §5](science.md) |
| 14 | Fairness: approaches to assess and handle | partly: age and sex discussed and measured where possible; no patient ages | [ETHICS.md](../ETHICS.md) §3, [MODEL_CARD.md](../MODEL_CARD.md) |
| 15 | Model output and threshold | yes: score in [0, 1], threshold 0.5, fallback 24/90 | [MODEL_CARD.md](../MODEL_CARD.md) |
| 16 | Differences between development and evaluation data | yes: batch structure and evaluation-only groups | [data_integrity.md](data_integrity.md), [MODEL_CARD.md](../MODEL_CARD.md) |
| 17 | Ethical approval and consent | partly: data made available under the hackathon's data agreement; approval and consent are held by the data provider, whose identity is withheld under that agreement | [ETHICS.md](../ETHICS.md) |
| 18a | Funding | yes: no funding | this file |
| 18b | Conflicts of interest | yes: none declared | this file |
| 18c | Protocol availability | partly: SHA-256 hashes and a decision-rule summary; the texts describe the confidential data and are not published | [`protocols/README.md`](../protocols/README.md) |
| 18d | Registration | partly: self-registration by hash, not a public registry | [model_selection.md](model_selection.md) |
| 18e | Data sharing | yes: confidential and not available; aggregate model metrics published | [DATA_ACCESS.md](../DATA_ACCESS.md), [`results/`](../results/) |
| 18f | Code sharing | yes: MIT | this repository |
| 19 | Patient and public involvement | n/a: none in a hackathon setting | – |
| 20 | Participants: flow, characteristics, missing data per group | partly: scorable counts per group and the denominators of every metric; folder totals are withheld under the data agreement; no demographics per group | [MODEL_CARD.md](../MODEL_CARD.md) subgroup table |
| 21 | Model development: number of participants and outcome events per analysis | yes: 90 training subjects, 24 PTSD | [science.md §7](science.md) |
| 22 | Full model specification | yes: weights, transforms and code | [`weights/`](../weights/), [science.md §6](science.md) |
| 23 | Model performance with CIs, including subgroups | yes | [science.md §9](science.md), [MODEL_CARD.md](../MODEL_CARD.md) |
| 24 | Model updating | yes: v3 → v4.0 → v4.1 with predeclared rules | [model_selection.md](model_selection.md) |
| 25 | Interpretation, including fairness | yes | [science.md §6, §9](science.md), [ETHICS.md](../ETHICS.md) |
| 26 | Limitations | yes | [limitations.md](limitations.md) |
| 27 | Usability in context: intended use, how to use, future research | yes: research-only use; next steps | [MODEL_CARD.md](../MODEL_CARD.md), [DISCLAIMER.md](../DISCLAIMER.md), [limitations.md](limitations.md#next-steps) |

Note on item 16: external validation in the TRIPOD sense (a new sample for the same task) was not possible. The
public-data check in [external_check.md](external_check.md) is a direction check on a different population and
protocol, not a validation of the model.
