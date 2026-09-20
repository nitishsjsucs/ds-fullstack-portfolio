# Data Science Full Stack

**Eight end-to-end data science systems, on eight real public datasets, in one application.**

Every project follows the complete CRISP-DM cycle with evaluated phase gates, ships a
live inference endpoint, and publishes its own leakage audit. Every number on the site
comes from a model trained on genuine data — there is no synthetic result anywhere in
this repository.

<!-- VIDEO:START -->
## 📺 Video walkthrough

**▶ [Watch the full walkthrough on YouTube](https://youtu.be/4C5_ZU11Ues)**

A guided tour of all eight systems: the architecture, the CRISP-DM record, the live
inference playgrounds, and the three findings that make the portfolio worth reading.
<!-- VIDEO:END -->

---

## Verification first

Two commands reproduce every correctness claim in this repository.

```bash
pytest backend/tests -q      # 133 tests
python scripts/audit.py      # static AST leakage scan + phase-gate aggregation
```

`scripts/audit.py` parses the AST of all eight projects looking for the syntactic shapes
that cause leakage, then aggregates every trained artifact and regenerates
[`AUDIT_REPORT.md`](AUDIT_REPORT.md). Current output, reproducible from a clean checkout:

| Check | Result |
|---|---|
| Static leakage scan | **0 critical, 0 warnings** across 8 projects |
| Per-project audit checks | **83 / 83** passed |
| CRISP-DM phase gates | **47 / 48** passed |
| Open findings | **1** — `segments` fails a gate, and the report keeps it |
| Test suite | **133** tests: core guarantees, per-project leakage, API contract |

The one failing gate is the point. It is reported rather than tuned away, and the same
payload is served live at `/api/audit`.

What the scan does **not** establish: a static scan matches syntax, not semantics, and
passing every check means the known failure modes were checked for — not that the
analysis is correct.

---

## The systems

<!-- PROJECT-TABLE:START -->
| # | Project | Data | Task | Headline result |
|---|---|---|---|---|
| 01 | [NYC Taxi Fare & Duration](docs/papers/taxi/paper.md)<br/><sub>Spatio-temporal gradient boosting on 150,000 real January-2024 yellow-cab trips.</sub> | NYC TLC Yellow Taxi Trip Records, January 2024<br/><sub>150,000 rows</sub> | Supervised regression (dual target) | mae minutes **2.9284**, rmse minutes **4.6072** |
| 02 | [Telco Churn & Retention Economics](docs/papers/churn/paper.md)<br/><sub>Cost-sensitive classification with calibration and a fairness audit on 7,043 subscribers.</sub> | IBM Telco Customer Churn<br/><sub>7,043 rows</sub> | Supervised binary classification (imbalanced) | pr auc **0.6604**, roc auc **0.848** |
| 03 | [RFM Customer Segmentation](docs/papers/segments/paper.md)<br/><sub>Unsupervised segmentation of 4,339 real UK e-commerce customers.</sub> | UCI Online Retail<br/><sub>4,339 rows</sub> | Unsupervised clustering | k **4**, silhouette **0.2389** |
| 04 | [Market Basket Association Mining](docs/papers/basket/paper.md)<br/><sub>Apriori and FP-Growth over 20,136 real invoices and 3,925 SKUs.</sub> | UCI Online Retail<br/><sub>530,693 rows</sub> | Association rule mining | baskets **16,785**, items **480** |
| 05 | [Network Intrusion Anomaly Detection](docs/papers/anomaly/paper.md)<br/><sub>Unsupervised outlier detection on 80,000 KDD-99 connections with a 3.4% attack base rate.</sub> | KDD Cup 1999 network intrusion<br/><sub>80,000 rows</sub> | Anomaly / outlier detection | pr auc **0.4739**, roc auc **0.9659** |
| 06 | [Bike-Share Demand Forecasting](docs/papers/forecast/paper.md)<br/><sub>Multi-horizon hourly forecasting over 17,379 real Capital Bikeshare observations.</sub> | UCI Bike Sharing<br/><sub>17,379 rows</sub> | Time series forecasting | mase **0.3347**, mae **28.7969** |
| 07 | [AutoML Stacking Tournament](docs/papers/automl/paper.md)<br/><sub>Hill-climbing search and Caruana ensembling over 48,842 census records.</sub> | UCI Adult / Census Income<br/><sub>48,842 rows</sub> | AutoML / model selection | roc auc **0.9314**, pr auc **0.8363** |
| 08 | [NanoGPT: A Transformer From Scratch](docs/papers/nanollm/paper.md)<br/><sub>A decoder-only language model built layer by layer and trained on 1.1M characters of Shakespeare.</sub> | Tiny Shakespeare character corpus<br/><sub>1,115,394 rows</sub> | Self-supervised sequence modelling | parameters **812,288**, val bits per char **2.1714** |
<!-- PROJECT-TABLE:END -->

---

## What makes this different from a notebook dump

Three commitments run through every project.

**Real data, with provenance.** [`scripts/fetch_data.py`](scripts/fetch_data.py) downloads
from the original sources — the NYC TLC's own parquet archive, UCI, IBM's Telco sample,
the KDD Cup — and records a SHA-256, a licence and an exact curation rule for every
committed file in [`data/MANIFEST.json`](data/MANIFEST.json). Curation is limited to
column selection, dtype coercion, deterministic subsampling and stable sorting. No
imputation, no scaling, no target-aware filtering happens at that layer, because all of
those belong downstream of the train/test split.

**Leakage discipline, enforced in code.** Every fitted transform lives inside a
scikit-learn `Pipeline` that cross-validation clones per fold. Time series are split
chronologically and asserted, with cross-validation purged by the longest feature
look-back. The hold-out is scored exactly once per model family, after the search closes,
and the CV-to-hold-out gap is published — that gap is the cost of searching.
[`scripts/audit.py`](scripts/audit.py) walks the AST of every project looking for the
syntactic shapes that cause leakage, and the test suite asserts the guarantees directly.

**Honest reporting.** Baselines are published next to results. Negative findings are kept
rather than deleted. Metrics are chosen to suit the problem instead of to flatter the
model — at a 3.4% attack rate, accuracy appears nowhere as a headline. One CRISP-DM phase
gate in this portfolio **fails**, and the report says so rather than moving the threshold.

---

## Three results worth opening

**Three anomaly detectors are inverted.** On network telemetry, Local Outlier Factor,
one-class SVM and PCA reconstruction all score *below* 0.5 ROC-AUC (0.34, 0.28, 0.24) —
they rank intrusions as more normal than benign traffic. The dominant flood attacks are
enormously repetitive, so they form the densest region of the feature space, and every
density-based method reads them as the most ordinary traffic present. A fourth detector,
Elliptic Envelope, produced NaN scores outright: its covariance is singular on 52
largely-one-hot columns, so the Gaussian assumption is not fitting badly, it is
undefined. "Anomalous" and "malicious" are different properties.
→ [`docs/papers/anomaly/paper.md`](docs/papers/anomaly/paper.md)

**Ensembling bought nothing.** After a full hill-climbing tournament on census income,
Caruana greedy selection reaches 0.9314 hold-out ROC-AUC against 0.9313 for the best
single model, and the logistic meta-learner scores 0.9309 — below it. A difference in the
fourth decimal on a 9,769-row hold-out is indistinguishable from noise, which is exactly
what a 0.92 correlation between base learners predicts. Reported at face value rather
than dressed up as a win.
→ [`docs/papers/automl/paper.md`](docs/papers/automl/paper.md)

**An 80% interval that covered 67%.** Raw gradient-boosted quantile regression produced
bands that were confidently wrong. Conformalized quantile regression, calibrated on a
held-out block the quantile models never saw, moved empirical coverage from 67.1% to
79.2% — at the cost of a band that is 17% wider. An interval that lies about its own
coverage is worse than no interval.
→ [`docs/papers/forecast/paper.md`](docs/papers/forecast/paper.md)

---

## The console

Every screen below is a real page rendered against real trained artifacts. The capture
script asserts each page produced content before saving, so the
[**full 67-screen gallery**](docs/screenshots/) doubles as an end-to-end check.

### Portfolio

|  |  |
|---|---|
| **Landing page** — eight systems, their datasets and metrics | **Data provenance** — source, licence, curation rule and SHA-256 per file |
| [![Landing page](docs/screenshots/00_home.webp)](docs/screenshots/00_home.webp) | [![Data provenance](docs/screenshots/00_datasets.webp)](docs/screenshots/00_datasets.webp) |

### One project, all the way through

NYC taxi — the same eight tabs every project gets.

|  |  |
|---|---|
| **Data** — quality scorecard, distributions, exclusion rules | **CRISP-DM** — six phases, evaluated exit gates, findings with evidence |
| [![Taxi data](docs/screenshots/01_taxi_data.webp)](docs/screenshots/01_taxi_data.webp) | [![Taxi CRISP-DM](docs/screenshots/01_taxi_crispdm.webp)](docs/screenshots/01_taxi_crispdm.webp) |
| **Models** — hill-climbing leaderboard and the full trial trajectory | **Live inference** — real calls to the deployed pipeline, with SHAP |
| [![Taxi models](docs/screenshots/01_taxi_models.webp)](docs/screenshots/01_taxi_models.webp) | [![Taxi live](docs/screenshots/01_taxi_live.webp)](docs/screenshots/01_taxi_live.webp) |

### The findings

|  |  |
|---|---|
| **Three inverted detectors** — ROC-AUC below 0.5, plus one that returned NaN | **Conformal calibration** — an 80% band that covered 67%, repaired to 79% |
| [![Anomaly models](docs/screenshots/05_anomaly_models.webp)](docs/screenshots/05_anomaly_models.webp) | [![Forecast evaluation](docs/screenshots/06_forecast_evaluation.webp)](docs/screenshots/06_forecast_evaluation.webp) |
| **Fairness audit** — disparate impact despite excluding sex and race | **Attention from the trained weights** — the causal mask, visible |
| [![AutoML fairness](docs/screenshots/07_automl_live.webp)](docs/screenshots/07_automl_live.webp) | [![NanoGPT explain](docs/screenshots/08_nanollm_explain.webp)](docs/screenshots/08_nanollm_explain.webp) |

### The rest

|  |  |
|---|---|
| **Churn** — retention economics you can drag | **Segments** — PCA map, personas, assignment confidence |
| [![Churn live](docs/screenshots/02_churn_live.webp)](docs/screenshots/02_churn_live.webp) | [![Segments live](docs/screenshots/03_segments_live.webp)](docs/screenshots/03_segments_live.webp) |
| **Market basket** — rule network and basket completion | **Audit & model card** — every check with its evidence |
| [![Basket live](docs/screenshots/04_basket_live.webp)](docs/screenshots/04_basket_live.webp) | [![Taxi audit](docs/screenshots/01_taxi_audit.webp)](docs/screenshots/01_taxi_audit.webp) |

---

## Running it

Requires Python 3.11+ and Node 18+.

```bash
git clone <this-repo> && cd ds-fullstack-portfolio

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download the eight datasets and write data/MANIFEST.json with checksums.
python scripts/fetch_data.py

# 2. Train every project and persist its artifacts (~1 hour; --quick takes ~4 minutes).
python scripts/train_all.py

# 3. Serve the API and the console.
uvicorn app.main:app --app-dir backend --port 8000   # terminal 1
cd frontend && npm install && npm run dev            # terminal 2 → localhost:5173
```

Or use the Makefile: `make setup && make data && make train && make dev`.

Everything is seeded. Two runs on the same data produce the same leaderboard.

---

## Repository layout

```
backend/
  app/
    main.py              FastAPI app; generic artifact routes + per-project routers
    registry.py          the single source of truth for what the portfolio contains
    core/                shared machinery every project depends on
      splitting.py         leakage-safe splits, purged/embargoed CV, assertions
      evaluation.py        metrics and chart-ready payloads; PR-AUC over accuracy
      eda.py               the reusable profiler behind every Data tab
      autoresearch.py      hill-climbing search + Caruana greedy ensembling
      explain.py           permutation importance, TreeSHAP, partial dependence
      conformal.py         split-conformal interval calibration
      fairness.py          four group-fairness criteria, reported together
      crispdm.py           the evidence-bearing six-phase record
    projects/<slug>/     data.py · train.py · __init__.py (predict + router)
  artifacts/<slug>/      pinned JSON payloads + pickled pipelines
  tests/                 133 tests: core guarantees, per-project leakage, API contract

frontend/
  src/charts/            hand-drawn SVG chart kit (no charting dependency)
  src/views/tabs/        Overview · Data · CRISP-DM · Models · Evaluation · Explain · Audit
  src/views/live/        one inference playground per project

scripts/
  fetch_data.py          the only place bytes enter the repository
  train_all.py           trains every project
  audit.py               static AST leakage scan + artifact aggregation
  generate_docs.py       writes the papers from the artifacts
  capture_screenshots.py Playwright capture of every screen

docs/
  papers/<slug>/         paper.md · abstract.md · article.md, generated from artifacts
  screenshots/           every screen, captured
  VIDEO_SCRIPT.md        timestamped narration + demo run-order
data/
  curated/               the committed, checksummed samples
  MANIFEST.json          source, licence, curation rule and SHA-256 for each
```

---

## The data

Nine curated files, 838k rows, all genuinely public. Full provenance is served at
`/api/datasets` and recorded in [`data/MANIFEST.json`](data/MANIFEST.json).

| Dataset | Source | Licence | Used by |
|---|---|---|---|
| NYC TLC Yellow Taxi, Jan 2024 | NYC Taxi & Limousine Commission | Public domain | `taxi` |
| TLC Taxi Zone Lookup | NYC TLC | Public domain | `taxi` |
| IBM Telco Customer Churn | IBM sample data | Apache-2.0 | `churn` |
| UCI Online Retail | UCI ML Repository | CC BY 4.0 | `segments`, `basket` |
| KDD Cup 1999 (SA subset) | DARPA / MIT Lincoln Labs | Public domain | `anomaly` |
| UCI Bike Sharing | UCI ML Repository | CC BY 4.0 | `forecast` |
| UCI Adult / Census Income | UCI ML Repository | CC BY 4.0 | `automl` |
| Tiny Shakespeare | Public domain | Public domain | `nanollm` |

---

## Verification, in full

`make verify` runs everything a reviewer should run:

```bash
pytest backend/tests -q      # 133 tests
python scripts/audit.py      # static leakage scan + phase-gate aggregation
cd frontend && npm run build # typecheck + production build
```

The numbers these produce are tabulated under [Verification first](#verification-first),
along with the limits of what a static scan can establish.

**What the 133 tests actually assert.** Not that the code runs — that the leakage
guarantees hold. A sample: the forecast purge window covers the longest feature
look-back; the anomaly label never enters the feature matrix; the transformer cannot
attend forward; the language-model split is contiguous and unshuffled; contamination is
treated as an assumption rather than a measurement.

**What `scripts/audit.py` looks for.** It parses every `.py` file under each project and
applies five syntactic rules, then cross-checks the trained artifacts against their
declared CRISP-DM exit gates:

| Rule | Shape it catches | Severity |
|---|---|---|
| `R1-rolling-without-shift` | a rolling/expanding window with no preceding `.shift()` — the row predicts itself | critical |
| `R2-shuffled-temporal-split` | `train_test_split(shuffle=True)` on a time-ordered project | critical |
| `R3-fit-transform-on-full-frame` | `fit_transform` outside a `Pipeline`, so CV cannot re-fit per fold | warning |
| `R4-resample-before-split` | SMOTE/oversampling applied to the full frame | warning |
| `R5-target-in-feature-list` | the target name appearing in a declared feature list | critical |

Two details worth noting, because they are where this kind of tool usually goes wrong.
R1 follows one level of assignment before firing, so the idiomatic two-statement
`past = target.shift(1)` pattern is not reported — the code's own comment explains why:
an audit tool that cries wolf gets switched off. And exceptions are made explicit rather
than by loosening a rule: an `# audit: ok - <reason>` marker suppresses a single call
site. The entire portfolio uses **one** such marker, in
[`backend/app/projects/segments/data.py`](backend/app/projects/segments/data.py), with
its reasoning written out.

---

## Documentation

- [`PROMPTS.md`](PROMPTS.md) — the prompts used to build this, and how they evolved
- [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md) — the walkthrough script and demo order
- [`AUDIT_REPORT.md`](AUDIT_REPORT.md) — generated leakage and methodology audit
- [`docs/papers/`](docs/papers/) — a paper, abstract and article per project
- `/docs` on the running API — interactive OpenAPI reference

---

## Licence

Code is MIT. The datasets carry their own licences, recorded per dataset in
`data/MANIFEST.json`; all are freely redistributable for research and education.
