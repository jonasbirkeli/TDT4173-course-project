## Setup

From the project root in Windows PowerShell:

```powershell
uv venv --python 3.14
uv pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

Select the project's `.venv` as the notebook kernel. Dependencies are installed by
the user; the notebooks do not install packages or download external data.

On Linux/macOS:

```bash
cp .env.example .env
uv venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements.txt
```

## CatBoost notebook

`catboost.ipynb` contains the complete raw-data-to-submission pipeline. It trains
either one shared binary classifier with categorical generator identity or one
classifier per generator, using operating inputs and weekly context. All required
code is inside the notebook.

1. Start with `RUN_MODE = "check"` and run all cells to audit data and feature alignment.
2. Set `RUN_MODE = "smoke"`, restart the kernel, and run all cells yourself for a small
   end-to-end training test. This is not a full-data score or a final submission.
3. Use `RUN_MODE = "validate"` for the two development folds (2020 and 2021).
4. Copy the configuration printed by validation into `SELECTED_CONFIG`. Use
   `RUN_MODE = "holdout"` to evaluate that frozen configuration on 2022.
5. Use `RUN_MODE = "submit"` to refit all labeled cases and create
   `outputs/catboost/<run>/catboost_submission.csv` in the template's exact order.

Each training run records parameters, input hashes, versions, predictions, metrics,
feature importance, and timing in a separate output directory. The user completed
the original validation run in 21.1 minutes (micro AUC 0.946865 on 2020 and 0.976947
on 2021) and the original submission run in 12.0 minutes. These are observations
for the original setup, not measurements for subsequent experiments. The user owns
installation and training.

### Improvement experiments

Use the `EXPERIMENT` setting to compare presets on the same development folds:

| Preset | Features | Early-stopping metric |
|---|---|---|
| `baseline_auc` | Original 193 features | Per-generator AUC |
| `probability_logloss` | Original 193 features | Logloss |
| `upstream_logloss` | 224 features including upstream reservoirs, initial head proxies, and price windows | Logloss |
| `shared_logloss` | Original 193 numeric features plus categorical generator identity; one shared model | Logloss |
| `decision_logloss` (default) | 216 features: original inputs plus economic periods and water-pressure proxies | Logloss |
| `economics_logloss` | 207 features: economics group only | Logloss |
| `water_balance_logloss` | 202 features: water-pressure group only | Logloss |

The user's completed `probability_logloss` run scored 0.957664 (2020) and 0.978575
(2021); `upstream_logloss` scored 0.957444 and 0.978637. The additional features did
not consistently improve both years. These are local validation scores.

For the new `decision_logloss` experiment, run `check`, then `smoke`, then `validate`
yourself, restarting the kernel between modes. It retains the 14 separate models,
seed, folds and model parameters from `probability_logloss`, adding 23 features.
The two optional ablations isolate each new group. New scores and training runtime
are unmeasured. See [the experiment record](experiments/2026-10-09-decision-features.md)
for definitions, saved-prediction diagnostics, limitations and measured references.

The shared experiment completed in 27.4 minutes with micro AUC 0.947842 (2020) and
0.976520 (2021), below the separate probability models on both years.

Full shared training expands to 6,872,544 rows. Its raw numeric array alone needs
about 4.94 GiB, with additional memory needed by preprocessing and CatBoost. The
notebook quantizes each partition separately using training-derived boundaries.
`SHARED_MAX_MATRIX_GIB` limits a raw allocation, not peak process memory. Check
available RAM and the smoke runtime before launching full validation.

`SELECTED_CONFIG` has been reset: paste the dictionary from the experiment you
choose after validation. `EXPERIMENT` must match that dictionary in `holdout` and
`submit` modes. The previous configuration remains available as `REFERENCE_CONFIG`;
use it with `EXPERIMENT = "baseline_auc"` to reproduce the original approach.
The validated probability configuration is preserved as `PROBABILITY_REFERENCE_CONFIG`;
use it only with `EXPERIMENT = "probability_logloss"`. Existing submissions and model
artifacts are preserved.

Run the notebook from this directory with the supplied `data/`,
`prediction_mapping.csv`, and `sample_submission.csv` present. These resources are
ignored by Git and must be obtained from the course dataset. The notebook's default
`check` mode never trains a model or writes a submission.

`sequence_model.ipynb` remains a development scaffold. For final course submission,
prepare two standalone short notebooks that reproduce the selected predictions
offline from raw data within the course's runtime and hardware limits.


### Checks without training

```powershell
.\.venv\Scripts\python.exe tests/check_catboost_features.py
.\.venv\Scripts\python.exe analyze_catboost_errors.py outputs/catboost/20260928T162829_635805Z_validate_probability_logloss --output outputs/catboost_diagnostics/probability_errors.csv
```

The diagnostic script reads completed development predictions, verifies case alignment
and reports generator, horizon-day, case-start-quarter and transition metrics. It
does not read holdout predictions, fit models or write a submission. These development
scripts are not runtime dependencies of the self-contained notebook.

### Saved-prediction blend audit

`analyze_catboost_blends.py` compares fixed blends of completed development runs
without fitting models or reading holdout predictions. The initial audit found that
75% separate probability predictions plus 25% shared predictions improved both
development years. See [the measured blend analysis](experiments/2026-10-09-blend-analysis.md).
This is an exploratory candidate requiring confirmation, not a measured Kaggle gain.
