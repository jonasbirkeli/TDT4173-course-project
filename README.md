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
one binary classifier per generator, sequentially, using operating inputs and
weekly context. All required code is inside the notebook.

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

Use the `EXPERIMENT` setting to compare three presets on the same development folds:

| Preset | Features | Early-stopping metric |
|---|---|---|
| `baseline_auc` | Original 193 features | Per-generator AUC |
| `probability_logloss` (default) | Original 193 features | Logloss |
| `upstream_logloss` | 224 features including upstream reservoirs, initial head proxies, and price windows | Logloss |

First run `check`, then run `validate` yourself for `probability_logloss`. Compare
both years against the original run in the notebook's final comparison table.
Then test `upstream_logloss` to isolate the effect of the extra features. Neither
experimental preset has been trained or proven to improve scores yet. Micro AUC
remains the selection objective; Logloss and Brier are additional diagnostics.

`SELECTED_CONFIG` has been reset: paste the dictionary from the experiment you
choose after validation. `EXPERIMENT` must match that dictionary in `holdout` and
`submit` modes. The previous configuration remains available as `REFERENCE_CONFIG`;
use it with `EXPERIMENT = "baseline_auc"` to reproduce the original approach.
Existing submissions and model artifacts are preserved.

Run the notebook from this directory with the supplied `data/`,
`prediction_mapping.csv`, and `sample_submission.csv` present. These resources are
ignored by Git and must be obtained from the course dataset. The notebook's default
`check` mode never trains a model or writes a submission.

`sequence_model.ipynb` remains a development scaffold. For final course submission,
prepare two standalone short notebooks that reproduce the selected predictions
offline from raw data within the course's runtime and hardware limits.

