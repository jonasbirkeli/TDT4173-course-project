# Project instructions and working context

## Communication and authorization

- Discuss the project in Norwegian. Write all code, comments, docstrings,
  notebook prose, error messages and project documentation in English.
- Explain choices and distinguish measurements from hypotheses. The user aims
  for an A and prioritizes predictive quality and rigorous experiments. Neither
  model size nor implementation speed alone justifies a model choice.
- The user installs dependencies and runs training unless they explicitly
  authorize an exception. Do not install packages or fit models without a new
  authorization. All earlier bounded training exceptions are completed; no
  fitting remains active. Cleanup on 2026-10-27 authorized documentation,
  non-fitting checks and archival removal, not another training round.
- Implement/check processing, validation and mapping without fitting when no
  training authorization exists. State unverified runtime/quality for untrained
  model changes. Never promise a public/private score or an AUC-based grade.
- The user owns standalone TCN development; Jonas owns XGBoost. No ensembling
  in the user's TCN work. Independent generator models would fill disjoint
  columns, but the selected recipe is one jointly trained shared network.
- Preserve user edits and existing datasets/submissions/results. Archive
  superseded sources before removal. Do not silently overwrite submissions.
- Do not recreate deleted `predict.py`, notebook builders, batch runners or
  cache-generator helpers for routine use. Runtime code stays inline.

## Authoritative references and task

Read the relevant reference before resolving uncertainty:
`ML_Task_TDT4173.pdf`, `Dataset_definitions_and_explanation.pdf`,
`TDT4173_Project_Submission_and_Grading.pdf`, `kaggle_metric.py` and
`sample_submission.csv`. PDFs/metric/data are supplied ignored local resources.

Predict hourly unit commitment (ON/OFF probabilities) for Tokke-Vinje to assist
its scheduling optimizer. There are 17 reservoirs, 9 plants and 14 units.
Each case starts at midnight UTC and spans 168 hours, t0-t167. Starts occur
daily, so adjacent planning cases overlap. Output 2,352 probabilities per case,
not production volumes or hard classes. Labels resemble
`result_committed_Tokke_G2_t24`.

Observed generator order:
`Hogga_G1`, `Tokke_G1`, `Tokke_G2`, `Tokke_G3`, `Tokke_G4`, `Lio_G1`,
`Byrte_G1`, `Vinje_G1`, `Vinje_G2`, `Vinje_G3`, `Haukeli_G1`, `Songa_G1`,
`Kjela_G1`, `Vesle Kjela_G1`. The template remains authoritative for column order.

## Data and prediction-time availability

- `data/kernel/Unit_commitment_decisions.csv`: 2,922 labeled daily starts,
  2015-01-01 through 2022-12-31, Run No 1-2922. `Run No` and `starttime` are
  metadata. `total_calculation_time` is optimizer output, never an input.
- `prediction_mapping.csv`: 725 starts, 2023-01-01 through 2024-12-25,
  Run No 2923-3647. The last horizon ends 2024-12-31 23:00 UTC.
- `sample_submission.csv`: exact 725 x 2353 schema including Run No. Preserve
  IDs, column order/names; validate finite probabilities in [0,1]. Placeholder
  values are not test labels. Public/private partition is unknown; do not
  assign 2023 to public and 2024 to private without authoritative evidence.
- Kernel inputs: historical day-ahead prices (EUR/MWh), inflow (m3/s, 17
  reservoirs plus one river), initial volume (million m3), synthetic marginal
  water value (EUR/MWh). Extra input history does not provide extra UC labels.
- Extended inputs: hourly minimum river flow for six stretches (m3/s), minimum
  volume for Totak/Byrtevatn/Staavatn (million m3), topology YAML and map PDF.
  Zero constraint values mean no corresponding minimum requirement.
- For a case starting S: storage at S, price/inflow/constraints during
  [S,S+168h), terminal water values at S+168h. January 1-7 uses January 1
  storage and January 8 values. No observed future storage during the week.
- Join named objects and UTC timestamps, never coincidental row positions.
  Check uniqueness, coverage, frequency, missing values and units. The price
  file's later 2025 section contains quarter-hour timestamps; never blindly
  reshape the entire file as hourly.

## Evaluation and honest interpretation

- Official metric is micro ROC-AUC over all case/unit/hour entries, including
  cross-unit rankings. Not accuracy or an average of per-unit AUCs.
- Split cases chronologically before expansion. Purge overlapping horizons:
  training_start + 168 hours <= validation_start (exclusive horizon ends).
  Fit learned preprocessing/features only on fitting cases. No held-out UC
  labels in inputs. Scoring cases must remain within the scoring period.
- Use comparable chronological folds, per-unit results and measured resource
  costs. Millions of overlapping binary targets are not independent examples.
  Overlapping absolute hours can have different decisions in different plans.
- 2020/2021 have repeated selection exposure. 2022 has prior project exposure;
  never call it pristine confirmation. Earlier years used for stress checks
  also become exposed. Do not select solely by public leaderboard feedback.
- A public target above .99350 (earlier .995) is aspirational. Private scores
  determine the leaderboard contribution to the grade and remain unknown.
- Approximate LP/physics features are not exact SHOP or guaranteed feasible
  binary commitment. Input routing is not learned graph message passing;
  subset gradients are not causal/population feature importance. Same-seed
  deterministic refits are reproducibility checks, not seed stability.

## Canonical TCN: terminal_resolution_6h (v35)

`sequence_model.ipynb` now contains the recipe that the user reports scored
**0.99261 public**, replacing the old v32 default (.99240). A subsequent
curve-bound candidate scored .99254 and is not promoted. Scores are user
reports, not independently retrieved. No model was fitted during cleanup.

- One standalone shared network, 343,034 parameters, 913 hourly/182 static
  channels and 14 outputs. Width 64, two kernel-3 convolutions per residual
  block, dilations 1/2/4/8/16/32/64, receptive field 509. Symmetric padding;
  a TCN-inspired noncausal temporal CNN because all exogenous week inputs
  are supplied. Nine width-16 plant branches receive named local/adjacent
  hydraulic features; they learn jointly, no ensemble or reservoir embedding.
- Actual exhaustive inventory: `docs/tcn_features.csv`, regenerated from
  tensor names inside the notebook to `outputs/tcn_feature_catalog.csv`.
  Neither catalog is read by model fitting. Per-group definitions and
  parameter evidence/limitations are explained in the notebook.
- Includes price/calendar, source-time storage/terminal values, inflow,
  constraints, local opportunity/water context, LP allocation, projected
  head/loss/loading, asinh/regime coordinates, duals/opportunity windows,
  flow-deficit/bypass, river geometry/penalties and 54 binary DP scenarios.
  No observed future storage, ordinal-count, extra curve branch or history.
- LP has 28 six-hour blocks; all neural inputs/outputs retain 168 hours.
  Terminal conversion uses the supplied main-waterway route, 33-point
  initial-head best-point energy and receiving-reservoir value. The selected
  recipe retains its approximate four-iteration efficiency inversion.
- BCE/AdamW, lr .001, weight decay .0001, batch 32, dropout .1, gradient
  clip 1, fitting-only asinh scaling/normalization with clip 10. Cosine schedule
  denominator remains max_epochs=100 even for the final 22 epochs. Most
  values are inherited heuristics, not demonstrated globally optimal.
- Current protocol `purged-fixed-reference-resolution-v3`: validate fits fresh
  20/42-epoch models on the same purged 2020/2021 folds (1820/360 and
  2186/359 fitting/scoring cases). No new inner stopping. Freeze records
  inherited 22 final epochs without fitting. Submit uses all 2922 cases.
  Optional holdout fits 2551 cases and scores exposed 2022's 365 cases.
- The 20/42 outer and 22 final durations came from earlier v32; do not claim
  newly selected epochs for v35. Seed42/four neural threads; up to four LP
  workers with one solver thread each. Patience15 is inactive in fixed fits.
- Frozen implementation signature:
  `9e290ce6bdaab720fabbe37115bb5a7738daa78488bd33a880b06df4a0cb9dee`.
  Feature signature:
  `3760ef3d550f5b4eafaae3fb08360c7edfdd96c0f6206a0145ab072aed11e618`.
  The actual inline guard checks parameters, versions, raw hashes and functions.
  Never transplant configuration dictionaries across implementations.
- Known CSV:
  `outputs/tcn_candidates_20261027_resolution/terminal_resolution_6h_submission.csv`.
  SHA256 `7afc5ebf0daffd0a482b668f18b19bbd70f7ffe151c49f4b2db3e3c2e2231ede`.
  Development .980920932736332/.9892533562652732. Two original fresh raw
  refits 500.19/476.59 seconds, peak9.11GiB/four threads, identical hash.
  During cleanup raw/scaled fixture inputs and initial logits match the
  original portable source exactly; no new full fit/timing was performed.
- Reproduction: change only RUN_MODE from check to submit, fresh kernel,
  Run All with supplied raw files and `environments/tcn.txt`. No pretrained
  artifact, cache, external source helper or download is needed at runtime.

## Active files, cleanup and historical evidence

- `docs/experiments_history.md` retains the complete original 5,246-line
  chronology for the report. Preserve it; older settings and paths are historical.
- `docs/experiments.md` is the compact historical decision record, including
  successful/failed/partial/unfitted trials, limitations and research sources.
  The course report and official final notebooks are NOT finished.
- `xgboost.ipynb`, its two checks and `environments/xgboost.txt` are preserved
  for Jonas. Its retained reference uses 14 models and 298 topology/allocation
  features; .9829521212/.9879182383 development. No score is inferred for
  Jonas's newer work. Another team's .99655 is an unverified user report.
- `test.ipynb` and the ignored `schedule_visualization.ipynb` are preserved.
  `requirements.txt` retains the user's broad development list. Pinned
  environments capture exact measured versions; no automatic installation.
- Cleanup on 2026-10-27 removed 19 superseded sources from the active tree:
  13 portable TCN notebooks, CatBoost/LightGBM/ensemble notebooks, their two
  obsolete checks and `environments/models.txt`. Their exact current bytes,
  the old long experiment log, old instructions and previous main are in a
  verified ZIP identified by `outputs/cleanup/latest.json`. The ignored local
  archive is not pushed. All outputs/submissions/native evidence are retained.
  Do not recreate these removed entry points for routine TCN work.
- Current non-fitting check, from the repository root:
  `.\.venv\Scripts\python.exe tests/check_sequence_pipeline.py`.
  It blocks optimizer updates and pretrained loading, audits the actual frozen
  recipe/catalog, timestamps, case isolation, future-storage/label invariance,
  normalization, gradients, official metric, mapping and mocked orchestration.
- `.venv` exists; Windows Python is `.\.venv\Scripts\python.exe`.
  uv was previously found at `C:/Users/fredr/.local/bin/uv.exe`.
- Dataset/PDF/metric/mapping/template and outputs are ignored. A fresh checkout
  requires supplied course resources separately. Preserve these local files.

## Grading and final delivery

The grading PDF is authoritative. An A requires 89-100 project points, based
on instructor Virtual Teams beaten on the private leaderboard and deductions;
there is no official fixed AUC threshold. The better of two selected private
submissions is used. Try/document at least two predictor types (not necessarily
both in final submissions) to avoid the specified 3-point deduction.

Document at least four listed EDA activities: domain knowledge, plausibility,
data generation, individual-feature exploration, feature-group exploration,
cleaning. Explain feature engineering, model interpretation and hyperparameter
selection, including unsuccessful experiments; the PDF specifies deductions.
No external data is allowed.

Submit `Short notebook 1.ipynb`, `Short notebook 2.ipynb`, environment
configuration and `Report.pdf` or `Report.ipynb`. Other files are removed before
rerunning; both notebooks must be self-contained and fit from provided raw
resources offline, without separate source files/pretrained artifacts. Temporary
execution files are allowed. Specify exact dependency versions. Each notebook
has a total 12-hour limit on a normal PC up to four CPU cores/32 GB RAM. GPU
use needs documented model/software; no multiple GPUs or high-end assumptions.
Leave a measured runtime margin. Add full names, student IDs and Kaggle team
name (including group number) at the beginning of notebooks and report.

PDF deadline: November 8, 2026, 22:00 Oslo; consult the source for penalties and
administrative rules. Student metadata, report and official two-Short-notebook
packaging remain unfinished. Do not imply cleanup completes those deliverables.
