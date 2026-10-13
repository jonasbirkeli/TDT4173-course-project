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
- `catboost.ipynb`: self-contained implementation with 14 sequential generator-specific CatBoost classifiers, timestamp-aligned features, chronological validation, interpretation, and verified submission mapping. The user has trained the separate-model presets; the shared preset was also trained and was weaker on both development years. New decision-context presets are implemented but untrained.
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
- The user completed the original CatBoost validation and submission runs. The neural candidate remains a scaffold. The probability and upstream presets have also been trained by the user; the shared preset was also trained by the user; only the new decision-context presets remain untrained.
- Original validation run: `outputs/catboost/20260928T151057_267694Z_validate`, completed in 1266.74 seconds. Micro ROC-AUC: 0.9468646363 (2020) and 0.9769469054 (2021).
- Original submission run: `outputs/catboost/20260928T153454_952823Z_submit`, completed in 721.99 seconds. Its `catboost_submission.csv` was verified to match the template with valid probabilities. No Kaggle score is known.
- A diagnostic re-evaluation of the existing validation predictions produced mean per-generator Logloss/Brier of 0.294538/0.093357 for 2020 and 0.195099/0.057779 for 2021. This involved no training. Diagnostic CSVs are under `outputs/catboost/baseline_fold_*_probability_diagnostics.csv`.
- The CatBoost notebook's `check` mode has run successfully without installing packages or training. Its default mode is `check`; other modes are `smoke`, `validate`, `holdout`, and `submit`.
- CatBoost has seven controlled `EXPERIMENT` presets: `baseline_auc` uses the original 193 features and AUC stopping; `probability_logloss` uses the same 193 features with Logloss stopping; `upstream_logloss` uses 224 features with Logloss stopping. `shared_logloss` uses the original 193 numeric features plus categorical generator identity in one shared model. The current default is `decision_logloss`, in non-training `check` mode. It adds 23 features to v1 (216 total). Optional `economics_logloss` (207) and `water_balance_logloss` (202) isolate the groups.
- The v2 features include all connected upstream reservoirs, capacity-weighted aggregates, initial levels from supplied volume-level curves, a gross-head proxy, and in-horizon price-window summaries. Haukeli's upstream set includes Langeidvatn as well as its immediate intake Vatjern. The aggregates are descriptors, not a hydraulic simulator or realized future storage.
- The rationale for Logloss stopping is that per-generator AUC ignores probability scales across generators, which affect global micro AUC. AUC-based stopping retained only 6 trees for Tokke G1 in the 2020 fold; its validation Logloss was substantially better later. This is evidence to test another stopping criterion, not proof that Logloss improves Kaggle performance.
- All seven presets' data checks pass. Original v1 features were compared exactly against the previous implementation; v2 rolling-window boundaries, topology, and invariance to future realized storage/labels were checked. Evaluation matches the supplied Kaggle metric on saved predictions. No new models were fitted.
- CatBoost uses two default development folds (2020 and 2021), purging overlapping horizons at each boundary. 2022 is reserved for a frozen-configuration evaluation, without early stopping. The baseline has already been evaluated on 2022, so do not describe it as wholly unseen across project history.
- A successful `validate` run prints `SELECTED_CONFIG`, containing experiment name, parameters, feature version/signature, seed, and median best tree counts. The user copies this dictionary into the configuration cell before `holdout` or `submit`; the experiment must match the frozen configuration.
- The old selected dictionary is retained as `REFERENCE_CONFIG`. Current `SELECTED_CONFIG` is empty pending decision-feature validation. The completed probability configuration is retained as `PROBABILITY_REFERENCE_CONFIG`. To reproduce the old approach use `EXPERIMENT = "baseline_auc"` and `SELECTED_CONFIG = REFERENCE_CONFIG`. A pre-edit notebook backup is stored under `outputs/catboost/reference_notebook_before_improvements.ipynb`; it is not a runtime dependency.
- Notebook section 8 compares completed development runs only; it does not choose presets using holdout/test scores. Existing outputs are preserved and new run directory names include the experiment name.
- Each training-mode run writes a new directory under `outputs/catboost/` with metadata, input hashes, package versions, models, metrics, interpretation, and timing. Only `submit` exports `catboost_submission.csv`, after refitting all labeled cases. No saved models are needed for reproduction.
- The sequence scaffold still has only the 2021 development fold; align it with the final CatBoost comparison protocol when implementing it.
- A `.venv` exists; Windows PowerShell can run it directly with `.\.venv\Scripts\python.exe`. `uv` has been found at `C:/Users/fredr/.local/bin/uv.exe`.
- `.gitignore` excludes local data, outputs, `.env`, and several downloaded task resources. Do not assume those resources will be available from a fresh Git checkout.

## Immediate next steps when the user proceeds

1. Let the user run `check`, `smoke`, and then `validate` for `decision_logloss` in `catboost.ipynb`. Do not train on their behalf.
2. Inspect their results/errors and compare both development years against the original setup; do not run training for them or claim an improvement before measurement.
3. Use the user's validation results to investigate features, hyperparameters, and any benefit from LightGBM/XGBoost alternatives or blending.
4. Develop the neural sequence candidate and compare on the same folds when requested.
5. Record interpretation, ablations, and runtime; later prepare and verify the two standalone submission notebooks.

## Shared-model experiment update

- User-completed `probability_logloss` validation: 0.9576642204 (2020), 0.9785746981 (2021), 1387.98 seconds; run `20260928T162829_635805Z_validate_probability_logloss`.
- User-completed `upstream_logloss` validation: 0.9574442752 (2020), 0.9786369332 (2021), 1644.36 seconds; run `20260928T165421_240982Z_validate_upstream_logloss`. Extra features did not consistently improve both years.
- `shared_logloss` tests sharing across generators with the same numeric inputs as `probability_logloss`. Generator identity uses a categorical feature with one-hot handling. Validation trains two models, submit one. Frozen configurations now identify the model strategy and use a single `shared` tree count for this preset.
- Full shared training has 6,872,544 rows; the raw numeric matrix alone is about 4.94 GiB. Pools and preprocessing need additional RAM. Quantization boundaries are derived only from training and reused for evaluation; partitions are prepared separately. `SHARED_MAX_MATRIX_GIB` is an allocation guard, not a peak-memory guarantee.
- Shared notebook schema, non-training check mode, generator/target ordering, quantized CatBoost pools, unlabeled test inputs, and memory guard were verified without fitting a model. The user subsequently trained shared validation: 0.9478418857 (2020), 0.9765204360 (2021), 1643.01 seconds; run `20260928T173917_869380Z_validate_shared_logloss`. The user owns installation and all training.

## Decision-feature experiment update (2026-10-09)

- The user completed probability submission run `20260928T185105_840038Z_submit_probability_logloss` in 420.34 seconds and later reported Kaggle score 0.98417, worse than their earlier submission. This score is user-reported, not independently retrieved; the previous Kaggle score/partition is unknown. No completed CatBoost holdout run was present at inspection.
- `decision_logloss` adds economic-period context and water-pressure proxies to v1, preserving separate models, seed, folds, depth, learning rate and stopping. Its 216 features exclude v2 upstream additions. Two ablations isolate economics (207) and water balance (202). These are untrained hypotheses, not improvements.
- Economics features use the existing local price-minus-terminal-value descriptor. They do not claim to represent net profit or properly converted downstream marginal water cost. Water-pressure features use initial storage and direct intake natural inflow only, with no releases/transfers; they are not forecasts of realized storage or mandatory production.
- `analyze_catboost_errors.py` is a development-only, non-training audit of saved development predictions. It rejects holdout/test/smoke/incomplete runs, checks IDs and raw-label hash, reproduces micro AUC, and segments by generator, horizon day, case-start quarter and label transitions. Label-derived segments never enter features.
- Measured findings: Haukeli has the highest Brier error in both probability development folds. Its 2020 mean predicted ON probability is 0.531146 against an observed 0.244709, and the Logloss curve selects only 5 trees (38 in 2021). Further training worsened that fold's Logloss; more trees alone is not a proven fix. Transition hours have higher conditional error but are only 0.56%/1.78% of evaluated hours after t0. Errors do not consistently worsen late in the horizon.
- Details and evidence are in `experiments/2026-10-09-decision-features.md`; diagnostic output is `outputs/catboost_diagnostics/20261009_development_errors.csv`.
- `tests/check_catboost_features.py` passed non-training checks for boundary windows/streaks, water units and per-reservoir excess, case isolation, label and future-volume invariance, CatBoost input, all preset schemas, ablations and exact regression of previous features/signatures. Model fitting is blocked in the test runner. No packages were installed and no new models were trained.
- The new notebook default is `decision_logloss` / `check`; `SELECTED_CONFIG = None`. The user's previous notebook state is backed up in `outputs/catboost/notebook_before_decision_features.ipynb`, not used at notebook runtime. Existing submissions remain unchanged.

## Saved-prediction ensemble audit (2026-10-09)

- `analyze_catboost_blends.py` compared fixed candidate weights 0.25/0.50/0.75 for original, upstream and shared predictions blended with the separate probability reference. No training, holdout evaluation or submission generation occurred. Input manifests, fold definitions, raw labels and saved metrics were checked.
- Best tested combination on both development years: 75% `probability_logloss` + 25% `shared_logloss`, micro AUC 0.96031427 (2020) and 0.97941690 (2021), versus 0.95766422/0.97857470 for the reference. The shared model remains useful as a potential ensemble component despite its weaker standalone results.
- This is development selection, not independent confirmation or a Kaggle improvement. Do not repeatedly tune weights on 2022 or public scores. Final ensemble predictions require appropriately refitted separate and shared models, with both runtimes counted; the user still owns all training.
- Details: `experiments/2026-10-09-blend-analysis.md`; full results: `outputs/catboost_diagnostics/20261009_blend_comparison.csv`. The active notebook feature experiment remains `decision_logloss` in check mode.

## Decision-feature validation completed by the user

- Run `20261009T085100_521077Z_validate_decision_logloss` completed in 1988.97 seconds. Micro AUC: 0.9667133428 (2020), 0.9816361537 (2021), above the probability reference's 0.9576642204/0.9785746981. Input hashes, folds and model parameters match. New features now have measured development results. No Kaggle score for this preset is known.
- Per-generator AUC improved for 12/14 generators in 2020 and 11/14 in 2021. Haukeli ranking improved on both years; its 2020 Brier slightly worsened. Lio's 2021 AUC worsened from 0.955941 to 0.939166, and Vinje G1 worsened in both years. No retained model reached the 1500-tree cap.
- `selected_configuration.json` in that run freezes the new preset; the existing probability configuration is incompatible with the new feature signature. Do not start training on the user's behalf.
- The user expressed a preference to investigate CatBoost + LightGBM + XGBoost rather than a meta-model across different generators. Next model candidate: LightGBM using the same features and purged temporal folds, then evaluate complementary errors/blending. The neural candidate remains part of the distinct-model-family plan.

## Pure decision-feature submission prepared

- At the user's request to test a new pure CatBoost submission first, `catboost.ipynb` is now configured for `EXPERIMENT = "decision_logloss"`, `RUN_MODE = "submit"` and the exact frozen configuration from `20261009T085100_521077Z_validate_decision_logloss`. This overrides earlier current-mode notes.
- Configuration, feature signature, seed and 14 positive tree counts were verified. No training started and no new submission was generated by the assistant. The user should restart the kernel and run all cells. No further model changes are needed before this controlled submission.
- A 2022 holdout confirmation remains outstanding. The next Kaggle score is unknown; improved development results do not guarantee improved public performance.

## Pure decision-feature submission completed by the user

- Run `20261009T093004_209176Z_submit_decision_logloss` completed in 718.99 seconds (11 min 59 s). New CSV: `outputs/catboost/20261009T093004_209176Z_submit_decision_logloss/catboost_submission.csv`.
- Independent CSV verification passed: 725 rows, 2353 columns, exact template columns/order/IDs and finite probabilities in [0, 1]. This is a pure 14-model CatBoost submission with the 216 decision features. Kaggle score remains unknown until upload. The assistant did not run training.

## User-reported decision-feature Kaggle score

- The user reported Kaggle ROC-AUC 0.98679 for `20261009T093004_209176Z_submit_decision_logloss`, compared with 0.98417 for the earlier probability submission: +0.00262. Scores are user-reported, not independently retrieved; leaderboard partition is unspecified. Do not describe this as a private leaderboard score or a grade guarantee.
- New decision features improved both measured development folds and this reported Kaggle comparison. Preserve the exact frozen configuration and artifacts as the reference for LightGBM and later ensembles. No assistant training occurred.

## Jonas branch review before the next model implementation

- The user asked to inspect Jonas's XGBoost work before starting LightGBM. Reviewed local `origin/jonas` at `5fe43fe`. No checkout/merge, installation or fitting occurred. Details: `experiments/2026-10-09-jonas-branch-review.md`; read-only snapshots: `outputs/jonas_review/`.
- Branch records baseline `output.csv` score 0.98722 and volpath+cyclic `output2.csv` public score 0.98711. These are branch-recorded, not independently retrieved; do not infer a grade from the branch's asserted threshold.
- All 193 original baseline features were verified numerically identical to our v1 features after renaming, for all 14 generators across seven representative train/test cases. The baseline's added value is model/training choices rather than new inputs.
- Transferable ideas: isolated CatBoost `rsm=0.8` feature subsampling on the existing decision features; isolated four cyclic hour/day-of-week features; later predeclared fixed-tree-budget sensitivity or Depthwise growth. These are untrained hypotheses. Preserve the 0.98679 reference and test on matching purged folds.
- Protocol audit: stopping cases are 90 in-sample training cases; six training horizons cross the 2022 validation boundary. Its reported validation score is not directly comparable with our development folds. Its volpath reads observed volumes at S+1..S+7 days, contrary to our prediction-time feature rule; do not copy this feature group. Development early stopping is legitimate tuning, but final confirmation must remain separate.
- Haukeli/Lio whole-week errors are useful diagnostics, not proof of an unavailable outage variable; high in-sample AUC is not a generalization ceiling. No CatBoost code was changed during this review.

## Isolated CatBoost RSM experiment prepared

- `catboost.ipynb` now defaults to `EXPERIMENT = "decision_rsm_logloss"`, `RUN_MODE = "check"`, `SELECTED_CONFIG = None`. This overrides prior current-mode notes. It uses exactly the 216 decision features, same signature/folds/seed/parameters, with only `rsm=0.8` added. No new features or dependencies.
- The exact decision configuration that produced the reported 0.98679 is preserved inline as `DECISION_REFERENCE_CONFIG`, usable with `EXPERIMENT = "decision_logloss"`. The RSM preset requires its own validation and frozen configuration. Existing output artifacts are preserved.
- RSM samples candidate features at split selection, unlike XGBoost's per-tree column sampling. Its potential benefit is generalization; score and training runtime are unmeasured until the user trains. The assistant did not start training or install dependencies.

## RSM validation completed by the user

- `20261009T103934_880896Z_validate_decision_rsm_logloss` completed in 1592.96 seconds (26 min 33 s). Micro AUC: 0.9678485591 (2020), 0.9827461050 (2021), versus 0.9667133428/0.9816361537 for decision_logloss. Input hashes, folds and feature lists match; only rsm=0.8 differs in model parameters. Mean Logloss and Brier also improved on both folds. These remain development results; no Kaggle score is known for RSM.
- Notebook configured for decision_rsm_logloss / submit with the exact selected configuration from that completed validation. The user starts training. No assistant training occurred. The 0.98679 decision reference remains preserved inline and in its run artifacts.
