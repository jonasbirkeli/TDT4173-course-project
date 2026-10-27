# Tokke-Vinje unit commitment

Predict ON probabilities for 14 hydropower generators over a 168-hour planning
week. The two active predictor types are **standalone TCN** and **XGBoost**.
Jonas develops XGBoost; this branch maintains the TCN.

## Start here

| File | Purpose |
|---|---|
| [sequence_model.ipynb](sequence_model.ipynb) | Canonical standalone TCN; frozen `terminal_resolution_6h` recipe, user-reported public score **0.99261** |
| [xgboost.ipynb](xgboost.ipynb) | Pure-XGBoost pipeline; preserved for Jonas |
| [docs/experiments.md](docs/experiments.md) | Compact experiment history, measured comparisons, failed trials and sources |
| [docs/development_summary.md](docs/development_summary.md) | Short round-by-round story: features added, motivation, results and discarded branches |
| [docs/experiments_history.md](docs/experiments_history.md) | Complete original 5,246-line project chronology for the report |
| [docs/tcn_features.csv](docs/tcn_features.csv) | Exhaustive **913 hourly + 182 static** feature inventory, source, formula function and availability |
| [environments/tcn.txt](environments/tcn.txt) | Pinned measured CPU TCN environment |
| [environments/xgboost.txt](environments/xgboost.txt) | Pinned measured XGBoost environment |
| `tests/` | Non-fitting feature, leakage, mapping and pipeline checks |
| `test.ipynb` | Existing Kaggle download/setup notebook |

The TCN notebook explains the feature groups, topology routing, network and
parameter choices. It distinguishes measured choices from retained heuristics.
One shared network has 343,034 parameters and 14 outputs. Nine jointly learned
plant branches receive named local hydraulic inputs. No ensemble is used.

## Setup and reproduction

Supply the original course resources separately: `data/kernel/`, `data/extended/`,
`prediction_mapping.csv`, `sample_submission.csv` and `kaggle_metric.py`.
Data, local PDFs, `.venv/` and generated `outputs/` are ignored by Git. A fresh
checkout does not contain them. The optional local schedule viewer is preserved.

From the repository root, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r environments/tcn.txt
```

The pinned environment was measured with Python 3.14.7. Select that environment
as the notebook kernel. `requirements.txt` retains the broader development
package list; the environment files specify the measured model versions.
The user controls installation and training. Nothing is installed automatically.

To reproduce the TCN, open `sequence_model.ipynb`, change only `RUN_MODE` to
`'submit'`, and Run All from a fresh kernel. Runtime code and the frozen
configuration are inline. Supplied raw inputs are sufficient: no historical
output, feature cache, separate source helper or pretrained weights are read.
The feature inventory is regenerated as documentation under `outputs/`.

| Mode | What happens in this exact recipe |
|---|---|
| `check` (default) | Raw-input, feature, partition and frozen-configuration checks; no fitting |
| `smoke` | Small two-epoch trial; no selected submission |
| `validate` | Two purged chronological fits: 2020 scoring at 20 epochs; 2021 at 42 epochs |
| `freeze` | Retain the predeclared 22-epoch final budget; no new fitting |
| `holdout` | Optional frozen diagnostic on previously exposed 2022 |
| `submit` | Train on all 2,922 labeled 2015-2022 cases for 22 epochs; predict 725 cases |

The 20/42/22 durations were inherited from the earlier reference, not retuned
on this recipe's public score. `max_epochs=100` remains the cosine schedule
length even in a 22-epoch final fit. Earlier pipeline versions used inner
stopping; do not apply their mode descriptions to this fixed-budget recipe.

Known CSV: `outputs/tcn_candidates_20261027_resolution/terminal_resolution_6h_submission.csv`.
SHA256: `7afc5ebf0daffd0a482b668f18b19bbd70f7ffe151c49f4b2db3e3c2e2231ede`.
Two original raw-input refits reproduced that hash in 500.19 and 476.59 seconds,
about 9.11 GiB peak RAM, four CPU threads. These are historical measurements,
not a new training run during cleanup or a timing guarantee on other PCs.

## Checks and evidence

```powershell
.\.venv\Scripts\python.exe tests/check_sequence_pipeline.py
```

This check blocks optimizer updates and pretrained loading. It audits the actual
inline frozen configuration, timestamp alignment, future-storage/label invariance,
case isolation, scaler, feature catalog, gradients, official metric and output
mapping. Mode orchestration uses fabricated probabilities in temporary files.

Development micro AUC for the selected recipe is **0.9809209327 / 0.9892533563**
on purged 2020/2021 cases. These years have repeated selection exposure; 2022
has prior project exposure. Public scores are user reports, private performance
is unknown, and the public/private split must not be assumed to follow years.
See the experiment record for unsuccessful controls and uncertainty.

## Cleanup and remaining delivery work

Nineteen superseded files were archived and removed from the active tree:
13 portable TCN snapshots, the earlier CatBoost/LightGBM/ensemble notebooks,
their two obsolete checks and the old combined environment file. The original
long experiment log and previous main notebook are also backed up. The complete
log is additionally retained in Git as `docs/experiments_history.md`. The verified
local ZIP and manifest are identified by `outputs/cleanup/latest.json`.
**The archive is ignored and is not pushed to Git.** All datasets, submission
CSVs and native experiment evidence remain locally under their existing paths.

The official report, student/team metadata and two final Short notebooks remain
unfinished. Each submitted notebook must run offline from raw resources within
12 hours on up to four CPU cores and 32 GB RAM. No training or dependency
installation was performed during this cleanup.
