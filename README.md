# Tokke-Vinje unit commitment

Predict ON probabilities for 14 hydropower generators over a 168-hour planning
week. The two active predictor types are **standalone TCN** and **XGBoost**.
Jonas develops XGBoost; this branch maintains the TCN.

## Start here

| File | Purpose |
|---|---|
| [sequence_model.ipynb](sequence_model.ipynb) | Canonical standalone TCN; frozen `terminal_resolution_6h` recipe, user-reported public score **0.99261** |
| [analysis/tcn_candidate_20261028.ipynb](analysis/tcn_candidate_20261028.ipynb) | Global nonlinear input encoder; user-reported public score **0.99264** |
| [analysis/tcn_alternative_20261028.ipynb](analysis/tcn_alternative_20261028.ipynb) | Grouped semantic input encoders; parent reference, user-reported public score **0.99271** |
| [xgboost.ipynb](xgboost.ipynb) | Pure-XGBoost pipeline; preserved for Jonas |
| [docs/experiments.md](docs/experiments.md) | Compact experiment history, measured comparisons, failed trials and sources |
| [docs/development_summary.md](docs/development_summary.md) | Short round-by-round story: features added, motivation, results and discarded branches |
| [docs/validation_years.md](docs/validation_years.md) | Year-shift EDA, measured eight-fold LOYO scores and matched forward comparison |
| [docs/tcn_local_parameters_20261030.md](docs/tcn_local_parameters_20261030.md) | Dense local epoch/LR/batch study, decay controls and rejected candidates; existing .99334 recipe retained |
| [analysis/year_diagnostics.ipynb](analysis/year_diagnostics.ipynb) | Reproduce input/target EDA, saved-score plots and horizon-purged partition audits |
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

The October 28 candidates preserve all **913 hourly / 182 static** inputs and
add a nonlinear representation before the temporal blocks. The primary has
368,662 parameters; the alternative has 368,506. Each is one network, with its
own frozen configuration and all runtime code inline. The original scored
notebook remains unchanged. See [the matched comparisons](docs/experiments.md#october-28-test-how-the-network-represents-the-existing-features)
for selection exposure, failed controls and diagnostic limits. Either candidate
uses the same pinned environment and RUN_MODE workflow described below.
The grouped alternative has the more consistent locked diagnostics: gains
against v35 on both 2020/2021 seeds and on exposed 2022. The global candidate
loses the second-seed comparison. On October 29 the user reported public scores
of **.99271 grouped** and **.99264 global**, against the v35 reference's .99261.
These are small public gains; private performance remains unknown. The frozen
notebooks retain their original pre-feedback prose and exact reproduction hashes;
the experiment record contains the subsequent results.

## October 29 tuned TCN candidates

- [analysis/tcn_tuned_20261029.ipynb](analysis/tcn_tuned_20261029.ipynb): LR 0.00075, batch 16, 22 epochs; local CSV `outputs/tcn_tuned_20261029.csv`.
- [analysis/tcn_tuned_alternative_20261029.ipynb](analysis/tcn_tuned_alternative_20261029.ipynb): LR 0.0005, batch 8, 22 epochs; local CSV `outputs/tcn_tuned_alternative_20261029.csv`.

One standalone grouped network per CSV; the 913/182 input inventory and architecture remain fixed. Each notebook is self-contained and defaults to check. Its validate mode uses the selected epoch budget on both 2020/2021 folds; the 20/42 table below describes the unchanged canonical recipe. The user subsequently reported public AUC **.99333 main / .99334 alternative**; private performance remains unknown. See the experiment record for matched controls, parameter selection, failures and measured runtime. The frozen sources retain their pre-feedback prose and remain unchanged.

The October 30 local study tested 71 distinct training recipes, including every
epoch from 18 through 26 around the strongest LR/batch region and separate
learning-rate decay controls. The batch-4 proposal improved all three years at
seed 42 but lost the matched seed-17 check; faster decay did not improve the
three-year mean. The existing **LR .0005 / batch 8 / 22 epochs / cosine denominator
100** recipe and scored CSV remain unchanged. See the
[full local study](docs/tcn_local_parameters_20261030.md) for curves, the disclosed
protocol amendment and limits. No new submission was exported.

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

## October 28 candidate CSVs

- Grouped input encoders: `outputs/tcn_alternative_20261028.csv`; source [analysis/tcn_alternative_20261028.ipynb](analysis/tcn_alternative_20261028.ipynb).
- Global input encoder: `outputs/tcn_submission_20261028.csv`; source [analysis/tcn_candidate_20261028.ipynb](analysis/tcn_candidate_20261028.ipynb).

Each CSV was reproduced by two fresh raw-input fits with identical hashes,
four neural CPU threads and about 9.1 GiB peak main-process RSS. Exact
timings, hashes and diagnostic limitations are in the experiment record.
The CSVs remain ignored local outputs and must be supplied separately.

## Checks and evidence

```powershell
.\.venv\Scripts\python.exe tests/check_sequence_pipeline.py
```

Use `--notebook analysis/tcn_candidate_20261028.ipynb` (or the alternative's
path) to audit that notebook instead. The retained check supports the actual
frozen channel and parameter counts of each recipe.

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
