# Project instructions and working context

## Communication and scope

- Discuss the project with the user in Norwegian unless they request otherwise.
- Write all code, comments, docstrings, notebook prose, error messages, and project documentation in English.
- Explain modeling choices and distinguish measured results from hypotheses.
- The user is aiming for an A and prioritizes predictive quality and rigorous experiments. Do not select a model solely because it is quick to implement, or assume a larger model is better.
- ROC-AUC above 0.995 is the user's aspirational target, not an official grade threshold or a demonstrated achievable result. Never promise this score.
- Do not launch long training runs without establishing the available compute and runtime budget. Those details are not yet settled.
- Current request: record this context before starting the new models. Do not treat this file as authorization to immediately start training them.

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

Retain the simple baseline and investigate two stronger, distinct model families. Neither candidate has demonstrated superiority on this dataset yet.

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

### Planned development files

- `predict.py`: retain the seasonal-frequency baseline.
- `catboost.ipynb`: planned CatBoost development notebook.
- `sequence_model.ipynb`: planned neural-model development notebook.
- `report.ipynb`: planned EDA, experiments, comparisons, and interpretation.
- Shared Python helpers are acceptable during development. Final short notebooks must inline all required code and be independently runnable.

## Current implementation status

- `predict.py` implements a seasonal-frequency lookup baseline. It averages each generator/horizon-hour target by the case's start month. All cases starting in the same month receive identical predictions.
- It does not yet use prices, inflows, volumes, water values, constraints, or topology.
- Baseline validation uses 2551 training cases whose horizons end before or at 2022-01-01 and 365 validation cases starting in 2022.
- Measured baseline validation micro ROC-AUC: 0.564993. This is not a Kaggle/private leaderboard score.
- It refits on all labeled cases and writes `outputs/submission.csv`; output shape, IDs, column order, and probability range have been checked.
- The baseline has a local rank-based micro-AUC implementation. Use the supplied metric or a verified equivalent when extending evaluation.
- `schedule_visualization.ipynb` was corrected to read `data/kernel/Unit_commitment_decisions.csv` instead of the nonexistent `Generator_schedules_2015_2023.csv`.
- `requirements.txt` includes kagglehub, python-dotenv, ipywidgets, pandas, matplotlib, plotly, numpy, ipykernel, and nbformat.
- An attempted installation of CatBoost, scikit-learn, and PyYAML was interrupted. Do not assume installation succeeded; inspect the environment first. These packages are not yet listed in requirements.
- Neither the CatBoost nor neural candidate has been implemented or benchmarked yet. Do not claim a training-time estimate or score for them.
- A `.venv` exists; Windows PowerShell can run it directly with `.\.venv\Scripts\python.exe`. `uv` has been found at `C:/Users/fredr/.local/bin/uv.exe`.
- `.gitignore` excludes local data, outputs, `.env`, and several downloaded task resources. Do not assume those resources will be available from a fresh Git checkout.

## Immediate next steps when the user proceeds

1. Confirm available hardware and experiment runtime budget; inspect installed dependencies.
2. Build and verify the shared input alignment and chronological validation pipeline.
3. Develop and benchmark CatBoost with actual input features while retaining the baseline.
4. Develop the neural sequence candidate and compare on the same folds.
5. Record interpretation, ablations, and runtime; later prepare and verify the two standalone submission notebooks.
