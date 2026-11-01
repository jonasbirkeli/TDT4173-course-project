# Project instructions and working context

## Communication and authorization

- Completed bounded exception (2026-10-30): user approved up to four hours, 00:07:42-04:07:42 UTC, fitting cutoff 03:59:42. Last diagnostic completed 03:17:06.328597 UTC. Twenty primary cold trajectories plus six diagnostics measured 71 distinct recipes and 158 saved checkpoint scores (small proofs excluded). Dense epochs18-26 and nearby LR/batch candidates did not establish a supported replacement: LR.00025/b4/e22 improved all3seed42 years but lost2021seed17 by .000531269; cosine40/LR.0005/b8/e22 lost three-year mean .000023409 and seed17 .000029945. Diagnostic eligibility for the already locked cosine40 proposal was transparently amended after screening; do not call this preregistered confirmation. Retain original .99334 source/CSV and LR.0005/b8/e22/cosine100; no new all-data fit/notebook/submission, install, ensemble, upload, commit or push. All sources/CSVs/history preserved. No fitting remains active; new fitting needs new authorization. Native evidence: `outputs/tcn_development/20261030T000742Z_local_parameters/`; report/curves: `docs/tcn_local_parameters_20261030.md`.

- Completed LOYO diagnostic (2026-10-29): user explicitly requested fixed-recipe CV on the .99334 alternative and resumed after pausing. Eight completed cold fits (2015-2022, LR .0005/batch 8/22 epochs/seed 42) finished 2026-10-29T12:47:46.068035+00:00, before resumed operational cap 2026-10-29T13:46:50.940027+00:00. 2022 was retained by hash, seven remaining years fitted on resume, and the partial 2020 fit archived without loading weights. Mean annual micro AUC 0.989387673; pooled OOF 0.989793683; worst year 2020 at 0.979700929. OOF 2874 cases, 48 cross-year starts excluded from scoring. 2022 predictions bitwise match the existing forward run. No tuning, installs, pretrained loading, ensemble, new submission, upload, commit or push. All sources/CSVs/history preserved. No fitting remains active; new fitting needs new authorization. Evidence: `outputs/tcn_loyo/20261029T103109Z_best_grouped/`; interpretation in `docs/validation_years.md`.

- Completed bounded exception (2026-10-29): authorized 00:05:28-04:05:28 UTC; last measured fit/export 2026-10-29T03:36:49.220969+00:00. Grouped-TCN parameter search and applicable diagnostics are complete. No fitting remains active; new fitting needs new authorization. No installs, ensemble, Kaggle upload, commit or push. Evidence: `outputs/tcn_development/20261029T000528Z_hyperparameters/`.

- Completed bounded exception (2026-10-28): authorized window 01:59:09-05:59:09 UTC; last fit/export completed 2026-10-28T05:14:09.636003+00:00. Seven twofold TCN trials, locked seed17/exposed-2022 diagnostics and two fresh raw reproductions per candidate are complete. No fitting remains active; new fitting needs new authorization. No installs, ensemble, Kaggle upload, commit or push. Evidence: `outputs/tcn_development/20261028T015909Z_regime_learning/`.

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

## October 28 TCN candidates (public feedback reported October 29)

- Canonical `sequence_model.ipynb` remains the exact .99261 v35 reference.
- New self-contained sources: `analysis/tcn_candidate_20261028.ipynb` (global
  width22 residual input encoder, 368662 parameters) and
  `analysis/tcn_alternative_20261028.ipynb` (seven semantic width16 input
  encoders, 368506 parameters). Each fits one shared standalone network.
- Both retain the exact 913/182 inventory, six-hour LP, original backbone,
  local branches and BCE/AdamW settings. No LP sensitivity channels retained.
  Final seed42/22epochs and cosine denominator100 remain fixed. Each source
  contains its own frozen dictionary and strict implementation guard.
- Initial seed42 local mean narrowly favored global (.981165623/.989772729
  against semantic .981433309/.989470418). Choices locked before new 2022
  results. Semantic has more consistent diagnostics: both-year seed17 gains
  (.980881975/.990207193 vs reference .980640964/.989788417) and exposed
  2022 .989619034 vs reference .989252559. Global loses both seed17 years,
  with 2022 .989334884. Names reflect the initial lock, not private superiority.
- Verified `outputs/tcn_submission_20261028.csv` SHA256 `ab2f6e8c54657a9f238e55b74f8ca9359c01ccc7a0379188d5e96c92ee014e4b`.
- Verified `outputs/tcn_alternative_20261028.csv` SHA256 `b0ac7be382412d661c461ef4a2944f16e32d9f83bebe5b02f0f68b74bb82b04a`.
- Two identical fresh raw refits per source; roughly eight-nine minutes,
  9.1GiB peak main-process RSS, four neural threads. Exact reports, raw hashes,
  source signatures and complete audits are in the round's `deliverables.json`
  and `final_integrity.json`. Original history and sources remain unchanged.
- The generic retained check supports `--notebook` for each source and passed
  all three notebooks. Check mode fits nothing; runtime code remains inline.
- User reported public .99271 for grouped/semantic and .99264 for global on
  2026-10-29, versus reference .99261: gains .00010/.00003. Grouped became the best
  reported TCN public result before the subsequent parameter round. Feedback is attached
  to the CSV hashes in `leaderboard_feedback_20261029.json` under the round.
  No source/notebook/configuration was changed or model fitted for this update;
  immutable notebook prose still describes the pre-feedback state.
- All years are exposed; only two seeds were checked. Private scores remain
  unknown. No final course report or Short packaging is completed.

## October 29 tuned grouped TCN candidates (subsequent public feedback)

- Features/architecture stay identical to the .99271 grouped parent: 913 hourly/182 static channels, 368506 parameters, one shared network. Only training parameters change. Coarse grid 12->6->3 at 22/44/66 epochs; conditional refinement and missing seed diagnostics were time-gated. All validation years are exposed.
- User subsequently reported public .99333 main / .99334 alternative, gains .00062/.00063 from .99271. Best reported TCN public score is now .99334; the .00001 difference does not establish private superiority. Scores were not independently retrieved. Sources/CSVs retain exact frozen bytes and pre-feedback prose; private scores remain unknown.
- 22 was an inherited first resource rung, not a physical constant or exhaustive optimum. Main measured 22/44/66; alternative 11/22/33. Nearby 20/21/23/24 were not compared for these LR/batch recipes. Later training loss fell while validation AUC worsened; final 22 epochs transfer this selected budget to all 2922 cases rather than establishing a separate all-data optimum.
- Main recipe narrowly wins equal-year mean; alternative wins 2020/2021 but loses 2022. Against fresh matched 22-epoch references, both improve micro AUC in all three years. This does not beat every historical 20/42 result. On 2020 only 4/14 per-unit AUCs improve; the positive micro change comes from cross-unit pair rankings, not uniform unit improvements.
- Seed17 2021 AUC: main .989657166, alternative .989836348, reference .988702190. The alternative diagnostic was measured after export freezing and did not alter parameters/ranking/CSVs. Only two seeds and exposed years are available. Fresh raw runtimes 560.26/631.82 seconds, peak process RSS 9.11 GiB including in-process LP worker threads; 4026/8052 actual AdamW updates.
- `analysis/tcn_tuned_20261029.ipynb`: LR 0.00075, batch 16, epochs 22; CSV `outputs/tcn_tuned_20261029.csv` SHA256 `c9b507693edc5865a792c638d40f1ab56ce14ef5023da0c54660ee6f81973795`.
- `analysis/tcn_tuned_alternative_20261029.ipynb`: LR 0.0005, batch 8, epochs 22; CSV `outputs/tcn_tuned_alternative_20261029.csv` SHA256 `84a83f1df1206acfbb69b32c9ec831e4b12fa7c6c64affd9ac1bcac88613c693`.
- New source protocol purged-parameter-racing-v4 uses selected equal epoch budgets on 2020/2021; exposed 2022 diagnostic is 359 starts through horizon-end 2023-01-01, not historical 365. Each source has its own strict frozen dictionary/signature; never transplant old configs. Cosine denominator 100/floor .1 and all other model settings remain fixed.
- Retained checks and final_integrity passed; one fresh raw full fit per new source, actual 2922 fitting/725 prediction IDs and AdamW counts audited, no pretrained/cache loading. This is not two full reproducibility runs. Native results, failed initial protocol fixture, corrected continuation, skipped stages and seed limitations are recorded in the round.
- Original v35, grouped/global parents, XGBoost, environments, all earlier CSVs and 5246-line history are preserved. Generic check derives epoch/holdout expectations from the actual selected recipe and still passes old canonical source. No report/official Short packaging completed.

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
