# Tokke-Vinje unit commitment

Predict hourly ON probabilities for 14 generators over a 168-hour planning week.
The current candidate is **XGBoost with topology and water-allocation features**.

## Files to use

| File | Purpose |
|---|---|
| `xgboost.ipynb` | Current pure-XGBoost pipeline and frozen configuration |
| `catboost.ipynb` | Earlier CatBoost experiments and reference |
| `lightgbm.ipynb` | LightGBM comparison |
| `sequence_model.ipynb` | Temporal convolution comparison |
| `hydro_ensemble.ipynb` | Original CatBoost/LightGBM/TCN ensemble |
| `test.ipynb` | Existing Kaggle download/setup notebook |
| `schedule_visualization.ipynb` | Existing schedule viewer, local course resource |
| [Experiment history](docs/experiments.md) | Results, failed trials, feature rationale and research sources for the report |
| `tests/` | Non-fitting feature and pipeline checks |
| `environments/` | Exact measured dependency versions |

Every model notebook contains its runtime code. No separate builder, runner,
feature helper, pretrained model or feature cache is needed to run it.

## Setup

From the project root in Windows PowerShell:

```powershell
uv venv --python 3.14
uv pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

Select `.venv` as the notebook kernel. For the measured XGBoost-only environment,
use `uv pip install -r environments/xgboost.txt` instead. The other model versions
are recorded in `environments/models.txt`. The user installs dependencies and runs
training; notebooks do not install packages or download external data.

The provided `data/`, `prediction_mapping.csv` and `sample_submission.csv` must be
present locally. They and the task PDFs, supplied metric and `outputs/` are ignored
by Git. Obtain them from the course dataset when setting up another checkout.

## Run the current model

Open `xgboost.ipynb`, restart the kernel and run all cells from the project root.
It defaults to `RUN_MODE = "check"`, which checks inputs without fitting models.

| Mode | Purpose |
|---|---|
| `check` | Validate data, features, time alignment and mapping |
| `smoke` | Small training test; not a final submission |
| `validate` | Train/evaluate on purged 2020/2021 development folds |
| `holdout` | Evaluate a frozen configuration on 2022 without early stopping |
| `submit` | Refit all 2922 labeled cases and export 725 test-case predictions |

The selected `topology_allocation` preset and its frozen dictionary are already
inline. Set `RUN_MODE = "submit"` to reproduce it; no configuration needs copying.
Changing the experiment requires its own validation and matching frozen dictionary.
CatBoost retains its earlier `submit` default; check its configuration before Run All.
The ensemble uses `ENSEMBLE_MODE` rather than `RUN_MODE` as its main setting.

Selected XGBoost development micro ROC-AUC: **0.982952 in 2020 / 0.987918 in 2021**.
These are selected validation scores, not Kaggle scores. The 19 allocation features
come from a coarse water-balance relaxation; it is not an exact SHOP simulation.
No observed future storage or UC labels enter those features. 2022 has prior project
exposure and is not a wholly untouched confirmation period.

The verified pure-XGBoost CSV is:

`outputs/xgboost/20261014T162856_740453Z_submit_topology_allocation/xgboost_submission.csv`

The final refit took 14 min 1 s on four threads with peak 2.246 GiB, reusing deterministic
preprocessing computed earlier in 229.51 s. This is not a measured fully uncached
end-to-end benchmark. The notebook recomputes that preprocessing when run normally.
No Kaggle score has been reported for this CSV.

## Checks without training

```powershell
.\.venv\Scripts\python.exe tests/check_xgboost_features.py
.\.venv\Scripts\python.exe tests/check_xgboost_allocation.py
```

Other retained checks cover CatBoost, LightGBM and the temporal model. Checks use
the supplied raw files and do not fit predictors. Existing notebooks and saved
submissions were preserved during cleanup; original development scripts and notes
are backed up under `outputs/cleanup/` (manifest in `outputs/cleanup/latest.json`).

The full course report, student/team metadata and two final standalone short
notebooks still need completing. See AGENTS.md for project constraints and current
context, and the experiment history for evidence supporting the report.
