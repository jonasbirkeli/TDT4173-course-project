# Project instructions and working context

## Communication and scope

- Discuss the project with the user in Norwegian unless they request otherwise.
- Write all code, comments, docstrings, notebook prose, error messages, and project documentation in English.
- Explain modeling choices and distinguish measured results from hypotheses.
- The user is aiming for an A and prioritizes predictive quality and rigorous experiments. Do not select a model solely because it is quick to implement, or assume a larger model is better.
- ROC-AUC above 0.995 is the user's aspirational target, not an official grade threshold or a demonstrated achievable result. Never promise this score.
- The user installs dependencies and runs training themselves. Add required packages to `requirements.txt`, but do not install them or start any model training unless the user explicitly changes this instruction.
- Implement and test data processing, code structure, validation, and output mapping without training. Clearly state that model fitting, runtime, and predictive performance remain unverified until the user runs them.

## Task and authoritative local references

This TDT4173 course project predicts unit commitment (ON/OFF) in the Tokke-Vinje hydropower system. Predictions are intended to assist the scheduling optimizer, not to replace the entire optimization problem.

Read the relevant reference before resolving uncertainty:

- `ML_Task_TDT4173.pdf`: task specification.
- `Dataset_definitions_and_explanation.pdf`: data meanings, units, and time alignment.
- `TDT4173_Project_Submission_and_Grading.pdf`: submission and grading rules.
- `kaggle_metric.py`: supplied evaluation implementation.
- `sample_submission.csv`: authoritative output column names and ordering.

There are 17 reservoirs, 9 power plants, and 14 generating units. Each scheduling case starts at midnight UTC and covers 168 hours, indexed `t0` through `t167`. A new case starts daily, so adjacent cases overlap.

Generators, in the observed target order:

1. `Hogga_G1`
2. `Tokke_G1`
3. `Tokke_G2`
4. `Tokke_G3`
5. `Tokke_G4`
6. `Lio_G1`
7. `Byrte_G1`
8. `Vinje_G1`
9. `Vinje_G2`
10. `Vinje_G3`
11. `Haukeli_G1`
12. `Songa_G1`
13. `Kjela_G1`
14. `Vesle Kjela_G1`

Each case requires 2352 probabilities (14 units x 168 hours), not hard classifications or production volumes. Target names look like `result_committed_Tokke_G2_t24`.

## Data and time alignment

### Training and test cases

- `data/kernel/Unit_commitment_decisions.csv`: 2922 labeled cases, start dates 2015-01-01 through 2022-12-31, Run No 1 through 2922.
- Training metadata: `Run No`, `starttime`, `total_calculation_time`. The remaining 2352 columns contain binary labels.
- `total_calculation_time` is optimizer output, not an available prediction-time input. Do not use it as a feature.
- `prediction_mapping.csv`: 725 test cases, start dates 2023-01-01 through 2024-12-25, Run No 2923 through 3647.
- The final test horizon ends at 2024-12-31 23:00 UTC.
- `sample_submission.csv`: 725 rows and 2353 columns including `Run No`. Its placeholder values are not test labels.
- Preserve submission IDs, column names, and ordering exactly. Validate finite probabilities in [0, 1].

### Inputs

- `Historical_day_ahead_price_2015_2025.csv`: electricity prices, EUR/MWh.
- `Historical_inflow_1958_2025.csv`: inflow to 17 reservoirs and one river, m3/s.
- `Historical_volume_2015_2024.csv`: daily initial reservoir storage, million m3.
- `Synthetic_water_value_2015_2024.csv`: daily synthetic marginal water values, EUR/MWh.
- These files are in `data/kernel/` and provide inputs for both training and test dates.
- `data/extended/Constraint_min_flow.csv`: hourly minimum flow for six river stretches, m3/s; zero means no minimum requirement.
- `data/extended/Constraint_min_volume.csv`: hourly minimum volume for Totak, Byrtevatn, and Staavatn, million m3; zero means no minimum requirement.
- `data/extended/Tokke_Vinje_topology.yaml`: system configuration and connections.
- `data/extended/Tokke_topology.pdf`: system map.

For a case starting at time S, use price and inflow inputs for [S, S + 168 hours), storage at S, and terminal water values at S + 168 hours. Use constraints for the same horizon. For example, January 1-7 uses January 1 initial volumes and January 8 water values.

- Join by timestamps and named objects, never by coincidental row positions.
- Keep timestamps consistently in UTC.
- Do not use observed future reservoir storage within the planning week as if known at the start.
- Verify timestamp coverage, uniqueness, frequency, missing values, and units. The price file includes quarter-hour timestamps in its later 2025 section; do not assume the entire file is hourly or reshape it blindly.
- Input history extends beyond the labeled cases; extra timestamps do not provide extra UC labels.

## Evaluation and leakage prevention

- Official metric: micro ROC-AUC across all case/generator/hour predictions. It measures ranking, not classification accuracy.
- Split cases chronologically before expanding them into generator-hour rows.
- Remove training cases whose horizons overlap the validation period. With exclusive horizon ends, require `training_start + 168 hours <= validation_start`.
- Fit preprocessing and learned features only on the training fold. Never use held-out UC labels in features.
- Use multiple chronological development folds and reserve a final untouched period for model selection confirmation.
- Compare candidates on the same folds and metric. Report per-generator performance and runtime as well as overall micro ROC-AUC.
- Do not optimize decisions solely against the public Kaggle leaderboard.
- Do not infer that the 6.9 million binary training labels are independent observations: they come from only 2922 overlapping cases.
- A preliminary audit found agreement of about 82-96% between overlapping decisions in consecutive plans, depending on the generator. The same absolute timestamp is not necessarily the same scheduling decision across cases.

## Grading and reproducibility requirements

The grading PDF, rather than a fixed AUC target, is authoritative:

- An A requires 89-100 project points. Base points depend on the number of instructor Virtual Teams beaten on the private leaderboard, followed by applicable deductions.
- Public leaderboard ranking does not determine the grade. The better of the two selected submissions on the private leaderboard is used.
- Try and document at least two predictor types to avoid a 3-point deduction. Both types do not have to appear in the final submissions, and two submissions do not necessarily mean two different model families.
- Include at least four of the listed EDA activities: domain knowledge, data plausibility, understanding data generation, individual-feature exploration, feature-group exploration, and feature cleaning.
- Document feature engineering and model interpretation. Missing EDA, feature engineering, or interpretation each incurs a 3-point deduction under the specified rules.
- Explain hyperparameter selection and report unsuccessful experiments as well as successful ones.
- No external data is allowed.
- Select two Kaggle predictions and submit two short notebooks, `Short notebook 1.ipynb` and `Short notebook 2.ipynb`, reproducing the respective predictions, plus a report (`Report.pdf` or `Report.ipynb`).
- Final notebooks must run offline from the provided raw data. All submitted files except the two short notebooks and an environment configuration are removed before rerunning; do not depend on separate source files or pretrained artifacts.
- Temporary files may be generated during execution. Specify dependency versions in the final configuration.
- Each short notebook has a maximum total runtime of 12 hours, including preprocessing, training, and prediction, on a normal PC (up to 4 CPU cores and 32 GB RAM).
- GPU use requires documenting the model and software; do not assume a high-end GPU. Multiple GPUs are not allowed.
- Twelve hours is a ceiling, not a training target. Measure runtime and leave a safety margin.
- Include full names, student IDs, and Kaggle team name at the beginning of notebooks and report. The Kaggle team name must include the group number.
- Deadline stated in the PDF: November 8, 2026, 22:00 Oslo time. Consult the document for late-submission penalties and other administrative rules.

## Agreed modeling plan

Investigate two stronger, distinct model families. Neither candidate has demonstrated superiority on this dataset yet. The user deleted `predict.py`; do not recreate it. A fold-matched seasonal reference is now inlined in the CatBoost notebook.

### Candidate 1: CatBoost

- Gradient-boosted decision trees using actual operating conditions and engineered weekly context.
- Candidate features: hourly prices and inflows, initial storage/filling fractions, terminal water values, horizon position, calendar variables, constraints, and topology-informed relationships.
- Include context such as price relative to the rest of the week and accumulated/remaining weekly inflow.
- Investigate one shared classifier with generator identity versus separate generator models using measured validation performance and resource costs.
- Use feature importance and/or SHAP with appropriate limitations for interpretation.
- LightGBM and XGBoost remain legitimate alternatives; CatBoost was not chosen because it has been proven better here.

### Candidate 2: Neural sequence model

- Start with a temporal convolutional sequence architecture inspired by TCN, using the 168-hour input sequence plus initial state and terminal water values.
- Predict 168 x 14 ON probabilities, with shared representations across time and generators.
- The architecture should respect the task's available inputs: supplied planning-horizon inputs may provide whole-week context, while future labels and unavailable realized states may not.
- Tune capacity and regularization empirically. A large Transformer is not automatically preferable and is not the agreed first neural candidate.

### Comparison and optional ensemble

- Benchmark representative training runs before choosing the full compute budget.
- Evaluate both models using identical temporal folds and micro ROC-AUC.
- Consider blending only if held-out predictions show complementary errors and reproducible improvements within the runtime limit.
- Preserve an experiment log with parameters, features, seeds, fold definitions, scores, runtime, and memory observations.

### Development files

- `predict.py`: deleted by the user; historical baseline only, not a current dependency.
- `catboost.ipynb`: self-contained implementation with 14 sequential generator-specific CatBoost classifiers, timestamp-aligned features, chronological validation, interpretation, and verified submission mapping. The user has trained the original setup; improvement presets are untrained.
- `sequence_model.ipynb`: neural-model scaffold with the same setup and split; sequence construction and modeling remain TODO.
- `report.ipynb`: English report outline covering EDA, experiments, comparisons, interpretation, and reproducibility; analysis remains TODO.
- Shared Python helpers are acceptable during development. Final short notebooks must inline all required code and be independently runnable.

## Current implementation status

- The deleted `predict.py` implemented a seasonal-frequency lookup baseline, averaging each generator/horizon-hour target by the case's start month. It did not use operating inputs.
- Historical baseline validation used 2551 training cases whose horizons ended before or at 2022-01-01 and 365 validation cases starting in 2022.
- Measured baseline validation micro ROC-AUC: 0.564993. This is not a Kaggle/private leaderboard score.
- Historical baseline output was `outputs/submission.csv`; do not treat it as output from the new CatBoost model.
- CatBoost evaluation uses the same scikit-learn micro ROC-AUC call as the supplied metric, inlined to avoid dependence on the ignored `kaggle_metric.py`.
- `schedule_visualization.ipynb` was corrected to read `data/kernel/Unit_commitment_decisions.csv` instead of the nonexistent `Generator_schedules_2015_2023.csv`.
- `requirements.txt` includes the original packages plus CatBoost, scikit-learn, and PyYAML. The user installed the dependencies. Subsequent environment inspection found CatBoost 1.2.10 and scikit-learn 1.9.1. Do not install or upgrade dependencies for the user.
- The user completed the original CatBoost validation and submission runs. The neural candidate remains a scaffold. New CatBoost improvement presets have not been trained; do not attribute the original run's scores or times to them.
- Original validation run: `outputs/catboost/20260928T151057_267694Z_validate`, completed in 1266.74 seconds. Micro ROC-AUC: 0.9468646363 (2020) and 0.9769469054 (2021).
- Original submission run: `outputs/catboost/20260928T153454_952823Z_submit`, completed in 721.99 seconds. Its `catboost_submission.csv` was verified to match the template with valid probabilities. No Kaggle score is known.
- A diagnostic re-evaluation of the existing validation predictions produced mean per-generator Logloss/Brier of 0.294538/0.093357 for 2020 and 0.195099/0.057779 for 2021. This involved no training. Diagnostic CSVs are under `outputs/catboost/baseline_fold_*_probability_diagnostics.csv`.
- The CatBoost notebook's `check` mode has run successfully without installing packages or training. Its default mode is `check`; other modes are `smoke`, `validate`, `holdout`, and `submit`.
- CatBoost has three controlled `EXPERIMENT` presets: `baseline_auc` uses the original 193 features and AUC stopping; `probability_logloss` uses the same 193 features with Logloss stopping; `upstream_logloss` uses 224 features with Logloss stopping. The current default is `probability_logloss`, in non-training `check` mode.
- The v2 features include all connected upstream reservoirs, capacity-weighted aggregates, initial levels from supplied volume-level curves, a gross-head proxy, and in-horizon price-window summaries. Haukeli's upstream set includes Langeidvatn as well as its immediate intake Vatjern. The aggregates are descriptors, not a hydraulic simulator or realized future storage.
- The rationale for Logloss stopping is that per-generator AUC ignores probability scales across generators, which affect global micro AUC. AUC-based stopping retained only 6 trees for Tokke G1 in the 2020 fold; its validation Logloss was substantially better later. This is evidence to test another stopping criterion, not proof that Logloss improves Kaggle performance.
- All three presets' data checks pass. Original v1 features were compared exactly against the previous implementation; v2 rolling-window boundaries, topology, and invariance to future realized storage/labels were checked. Evaluation matches the supplied Kaggle metric on saved predictions. No new models were fitted.
- CatBoost uses two default development folds (2020 and 2021), purging overlapping horizons at each boundary. 2022 is reserved for a frozen-configuration evaluation, without early stopping. The baseline has already been evaluated on 2022, so do not describe it as wholly unseen across project history.
- A successful `validate` run prints `SELECTED_CONFIG`, containing experiment name, parameters, feature version/signature, seed, and median best tree counts. The user copies this dictionary into the configuration cell before `holdout` or `submit`; the experiment must match the frozen configuration.
- The old selected dictionary is retained as `REFERENCE_CONFIG`. Current `SELECTED_CONFIG` is empty pending new validation. To reproduce the old approach use `EXPERIMENT = "baseline_auc"` and `SELECTED_CONFIG = REFERENCE_CONFIG`. A pre-edit notebook backup is stored under `outputs/catboost/reference_notebook_before_improvements.ipynb`; it is not a runtime dependency.
- Notebook section 8 compares completed development runs only; it does not choose presets using holdout/test scores. Existing outputs are preserved and new run directory names include the experiment name.
- Each training-mode run writes a new directory under `outputs/catboost/` with metadata, input hashes, package versions, models, metrics, interpretation, and timing. Only `submit` exports `catboost_submission.csv`, after refitting all labeled cases. No saved models are needed for reproduction.
- The sequence scaffold still has only the 2021 development fold; align it with the final CatBoost comparison protocol when implementing it.
- A `.venv` exists; Windows PowerShell can run it directly with `.\.venv\Scripts\python.exe`. `uv` has been found at `C:/Users/fredr/.local/bin/uv.exe`.
- `.gitignore` excludes local data, outputs, `.env`, and several downloaded task resources. Do not assume those resources will be available from a fresh Git checkout.

## Immediate next steps when the user proceeds

1. Let the user run `check` and `validate` for `probability_logloss`, then optionally `upstream_logloss`, in `catboost.ipynb`.
2. Inspect their results/errors and compare both development years against the original setup; do not run training for them or claim an improvement before measurement.
3. Use the user's validation results to investigate features, hyperparameters, and any benefit from LightGBM/XGBoost alternatives or blending.
4. Develop the neural sequence candidate and compare on the same folds when requested.
5. Record interpretation, ablations, and runtime; later prepare and verify the two standalone submission notebooks.
