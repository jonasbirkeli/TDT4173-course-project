# Jonas branch review: lessons for CatBoost

Reviewed the locally available `origin/jonas` at commit
`5fe43feadcb4cb22acdf458917bca20ddfd72a87`. No checkout, merge, installation or model
training was performed. Read-only notebook snapshots are under `outputs/jonas_review/`.

## Recorded scores and attribution

- The branch records **0.98722** for the original XGBoost `output.csv`.
- It records **0.98711 public** for `output2.csv`, the volpath + cyclic variant.
- Our user-reported decision-feature CatBoost score is **0.98679**.
- These are recorded/reported Kaggle scores, not independently retrieved. Small
  leaderboard differences do not identify which feature or hyperparameter caused them.
  Do not import the branch's asserted grade threshold as an authoritative grading rule.

## Verified similarities

`notebooks/xgboost_baseline.ipynb` and the baseline section of
`notebooks/kaggle_submission.ipynb` use 14 separate generator classifiers. Their
193 original features match our v1 baseline numerically, after renaming columns.
This was checked for all 14 generators on seven representative training/test cases.
Both align initial storage at S, terminal water value at S+168h, and hourly inputs
inside the 168-hour horizon. Our measured decision preset adds 23 features to this
shared baseline. Jonas's baseline has no additional source of input information.

XGBoost parameters: 400 maximum trees, learning rate 0.05, depth 7, row subsampling
0.8, feature subsampling `colsample_bytree=0.8`, histogram training, seed 0 and
AUC-based stopping with patience 25. Most retained counts are near 400; Songa is 133
in the saved baseline validation output.

## Transferable experiments, in priority order

1. **Feature subsampling:** retain the 216 decision features and test CatBoost
   `rsm=0.8` against its existing default of 1.0. This transfers the idea of feature
   sampling, not the exact XGBoost algorithm: CatBoost `rsm` samples at split
   selection rather than being identical to `colsample_bytree`. Keep all other
   settings and both purged development folds fixed.
2. **Cyclic hour/week encodings:** add four features for sin/cos of hour-of-day
   (period 24) and day-of-week (period 7), as a separate ablation. Our day-of-year
   sin/cos already exists; these four do not. They use prediction-time-known data.
   Jonas's second submission combines these with volpath, so its score does not
   isolate cyclic encoding and is not evidence it improves CatBoost.
3. **Refit tree-budget sensitivity:** compare a small predeclared fixed-budget
   experiment (e.g. 400 trees) with the current generator-specific early-stopped
   budgets on the same development years. This tests whether the median refit rule
   is too conservative for some units; it does not justify blindly increasing all
   counts. No current CatBoost fold hit its 1500-tree maximum. Learned budgets for
   an outer evaluation must come only from earlier training/inner development data.
4. **Optional tree-growth experiment:** CatBoost Depthwise can test a different
   tree structure with explicit depth/regularization. Depth 7 does not imply equal
   capacity in XGBoost and symmetric CatBoost. This is a lower-priority hypothesis,
   not a claim that copying the XGBoost hyperparameters will reproduce its score.

References: [CatBoost common parameters](https://catboost.ai/docs/en/references/training-parameters/common)
documents `rsm`, `grow_policy`, `min_data_in_leaf` and stopping-related settings.

## Protocol differences that should not be copied as clean evaluation

**Training-tail stopping is in-sample.** The branch defines `stop_cases` as the
last 90 days of `train_cases`, then fits on all of `train_cases` and monitors
`stop_cases`. Confirmed: every one of the 90 stopping IDs is in the fitting pool.
This is a training diagnostic, not a held-out early-stopping estimate. It does not
read outer-validation labels, but should not be described as independent validation.
Our use of development validation for early stopping is allowed model selection;
it means those development scores are not a final untouched-test estimate. A properly
purged inner stopping split and an outer evaluation serve different purposes.

**Six training cases overlap the 2022 boundary.** The branch selects training
cases by start <2022-01-01, admitting December 26-31, 2021 cases whose 168-hour
horizons extend into 2022. Our protocol removes these overlaps. Its 2022 score
cannot be directly compared with our 2020/2021 results or used as proof of model
superiority without rerunning on matched, purged folds.

**Volpath uses observed future reservoir storage.** The extra-feature function
reads `vol_wide` at S+1 through S+7 days. Those values are not initial storage at S,
and conflict with our prediction-time availability rule. Do not port them into
CatBoost. Supplied in-horizon prices/inflows and terminal water values have a
different availability status; future storage being present in a CSV does not by
itself authorize using it. The cyclic features can be isolated without volpath.

**Error interpretation is not proof of missing information.** The branch's
Haukeli/Lio regime diagnostics are useful leads. Whole-week errors and a failed GRU
do not establish a maintenance schedule as the cause or prove the available inputs
cannot explain the labels. Its in-sample 0.9985 is a fit measurement, not a forecast
ceiling or proof of deterministic labels from the features.

## Next action

Before implementing LightGBM, report these findings to the user. The highest-priority
CatBoost experiment suggested by the review is isolated `rsm=0.8` on decision
features, followed by isolated cyclic encoding. Preserve the 0.98679 configuration.
Any new models are trained by the user; new score and runtime are unmeasured.
