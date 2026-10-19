# Project instructions and working context

## Communication and scope

- Discuss the project with the user in Norwegian unless they request otherwise.
- Write all code, comments, docstrings, notebook prose, error messages, and project documentation in English.
- Explain modeling choices and distinguish measured results from hypotheses.
- The user is aiming for an A and prioritizes predictive quality and rigorous experiments. Do not select a model solely because it is quick to implement, or assume a larger model is better.
- ROC-AUC above 0.995 is the user's aspirational target, not an official grade threshold or a demonstrated achievable result. Never promise this score.
- The user installs dependencies and runs training themselves. Add required packages to `requirements.txt`, but do not install them or start any model training unless the user explicitly changes this instruction.
- Implement and check data processing, validation, and output mapping without fitting models. For untrained changes, state that fitting, runtime and predictive quality are unverified. Distinguish those hypotheses from completed measured runs.

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

## Current modeling direction and files

- The latest modeling direction is XGBoost-only experimentation with physically informed features. Do not switch families or launch training merely because older notes mention another plan.
- The user deleted `predict.py`; do not recreate it. The old seasonal reference is inlined in CatBoost. Its historical2022micro AUC was .564993; it is not a current submission.
- All previous bounded training/installation authorizations are completed. A new run requires a new explicit user instruction; no training is currently active. The user normally installs dependencies and trains.
- `xgboost.ipynb`: current pure-XGBoost candidate, self-contained, `check` mode, `topology_allocation`, frozen dictionary inline. Fourteen classifiers,298features, seed42, four CPU threads.
- `catboost.ipynb`: earlier CatBoost RSM pipeline and historical controlled presets. It retains the user's existing `submit` mode; do not execute its run cell during a check without overriding the mode in memory.
- `lightgbm.ipynb`: selected neutral-initialization classifier pipeline, check mode and frozen configuration inline.
- `sequence_model.ipynb`: fully implemented wide TCN-inspired model, check mode. It is no longer a scaffold. Symmetric convolutions legitimately use known whole-week inputs; do not call it a strictly causal TCN.
- `hydro_ensemble.ipynb`: independently runnable original56.25%CatBoost/18.75%LightGBM/25%TCN ensemble, check mode. These are not the later35/30/35weights.
- `test.ipynb`: existing Kaggle download/setup notebook. `schedule_visualization.ipynb`: existing schedule viewer, reading Unit_commitment_decisions.csv. Preserve both.
- No `report.ipynb` currently exists. `docs/experiments.md` consolidates experiment findings, unsuccessful trials, topology reasoning, sources and artifact paths for writing the report. Do not imply that the course report or final short notebooks are finished.
- Notebook runtime code is inlined. Do not recreate deleted notebook builders, batch runners, temporary cache generators or development helpers for routine notebook use.
- `tests/` contains the retained non-fitting feature/pipeline checks. Force check mode and block model fitting when inspecting notebooks. Tests may require the supplied ignored course files.
- `requirements.txt` is the general setup. `environments/models.txt` and `environments/xgboost.txt` preserve exact measured package versions; do not install or upgrade packages without authorization.
- `.venv` exists. PowerShell: `.\.venv\Scripts\python.exe`. uv was found at `C:/Users/fredr/.local/bin/uv.exe`.
- Data, outputs, local PDFs, metric and mapping/template files are ignored by Git. A fresh checkout needs the provided course resources separately.

## Frozen measured results

Development uses the same purged2020/2021case folds (1820/360 and2186/359 fitting/evaluation cases).2022has prior project exposure; never describe it as wholly unseen. Learned normalization uses fitting cases only. Compare feature pipelines as pipelines rather than claiming architecture superiority from different inputs.

| Candidate | 2020 micro AUC | 2021 micro AUC |
|---|---:|---:|
| CatBoost decision RSM | .9678485591 | .9827461050 |
| Neutral LightGBM | .9643583 | .9833711 |
| Wide temporal CNN | .95586145 | .96460585 |
| Original frozen three-family blend | .9731119557 | .9853425654 |
| Later35/30/35blend, development selection only | .9735356938 | .9856353219 |
| XGBoost decision reference216features | .9632372220 | .9827852977 |
| XGBoost topology descriptors279features | .9625373539 | .9823556500 |
| Selected XGBoost topology + allocation298features | .9829521212 | .9879182383 |

The user reports Kaggle .98877 for the later three_model_submission.csv; partition unspecified. No Kaggle score has been reported for the selected XGBoost CSV. Another team's .99655 is user-reported XGBoost with a mention of topology, without code; never promise replication.

The original blend's frozen2022confirmation was .9837118366 versus .9811846717 for its tree reference. This result does not belong to the later35/30/35weights or to XGBoost. Development selection and independent confirmation are different.

XGBoost allocation improved per-unit AUC for14/14 units in2020 and13/14 in2021; Byrte slightly declined in2021. History362features had an incomplete experiment and no complete twofold score; shape/physics/combined presets were checked but not fitted. Topology descriptors alone were weaker on both years. Record negative results honestly.

## Selected XGBoost details and artifacts

- Native validation: `outputs/xgboost/20261014T162530_176483Z_validate_topology_allocation`; exact frozen config and raw hashes are there and inline in the notebook. Signature: `d08f69d2d991d7e227d3f0447ac1a9b16736704ccad26fb9fd74f608aa86f4da`.
- CPU hist128bins, depth7, lr.05, min child weight5, L2=10, row/column sampling.8, base score.5, seed42. Development Logloss stopping80, ceiling1200; no selected model reached the ceiling. Final tree budgets are integer medians across the two development folds. Structural parameters were held fixed, not demonstrated optimal.
- Allocation features are a deterministic12hour linear-flow relaxation over the supplied reservoir/plant/river/tunnel graph. They use initial storage, known weekly price/inflow/constraints and terminal water values, never future observed storage or UC labels.
- Projected storage is calculated from available inputs. This is not exact SHOP: no full tunnel pressure balance, changing head, spill curves, flow penalties or binary startup physics. Initial-head efficiency and local-to-global terminal-value conversions are approximate. Do not overstate physical fidelity.
- All3647cases solved successfully in229.51s; balance, case isolation and future-volume invariance passed. Some development trials overlapped; do not treat their elapsed times as controlled speed comparisons.
- Full refit: `outputs/xgboost/20261014T162856_740453Z_submit_topology_allocation/xgboost_submission.csv`, SHA256 `99f6132d9befa4370dabbeb395f1c8d9a02b9c3943c61919c495b3924c1078de`.
-14models refitted on all2922cases in840.91s, peak2.246GiB, four threads. This bounded refit reused deterministic raw preprocessing calculated earlier.229.51s +840.91s is a sum of stage times, not a measured uncached end-to-end run. The standalone notebook recomputes features from raw inputs.
- Verification: exact725x2353schema/order/IDs, finite[0,1]probabilities, frozen budgets, and three fresh raw-input reconstructions within2.98e-08. Manifest: `outputs/xgboost_experiments/20261014T162851_099377Z_submit_topology_allocation/verification.json`; selection: `outputs/xgboost_development/frozen_selection.json`.
- Original ensemble CSV: `outputs/hydro_ensemble/20261014T002236_840638Z_submit/ensemble_submission.csv`; full raw reproduction1024.27s,29models, peak2.73GiB.
- Later user-scored CSV: `outputs/tcn_experiments/weight_audit_20261014/three_model_submission.csv`. Preserve all existing outputs; never overwrite a prior submission silently.

## Repository cleanup (2026-10-19)

The user requested removal of unnecessary files. Twenty-three one-off development Python scripts and the obsolete blend-helper test were removed. Fourteen experiment notes were consolidated into `docs/experiments.md`; pinned environments moved into `environments/`. Notebook contents, datasets and submissions were preserved. The old long AGENTS and removed sources have a verified ZIP backup under `outputs/cleanup/`; `outputs/cleanup/latest.json` identifies its manifest. Historical script names in the consolidated record refer to archived experiments, not current entry points.

## Remaining work

The user can upload the verified pure-XGBoost CSV and report the actual score. Further modeling needs a new instruction/budget. Keep model selection rigorous; do not automatically tune against public scores or reuse2022to repeatedly optimize.

Write the English course report using the consolidated evidence and complete names/student IDs/group team metadata. Prepare the two independently runnable final short notebooks and exact environment configuration. The working notebooks are development artifacts, not proof that final delivery packaging is finished.
