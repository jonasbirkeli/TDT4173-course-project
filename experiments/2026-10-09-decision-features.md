# Decision-context CatBoost experiment

## Evidence before the new experiment

The user reported Kaggle ROC-AUC **0.98417** for the latest submission and said it
was worse than the earlier submission. This is user-reported, not independently
retrieved; the previous Kaggle score and leaderboard partition are not recorded.

Existing completed local development runs:

| Preset | 2020 micro AUC | 2021 micro AUC | Recorded time |
|---|---:|---:|---:|
| baseline_auc | 0.946865 | 0.976947 | 21.1 min |
| probability_logloss | 0.957664 | 0.978575 | 23.1 min |
| upstream_logloss | 0.957444 | 0.978637 | 27.4 min |
| shared_logloss | 0.947842 | 0.976520 | 27.4 min |

Separate probability models remain the local reference. Neither extra upstream
features nor the tested shared model consistently improved on that reference.
These results do not establish which strategy would win after further tuning.

## Saved-prediction error audit (no training)

`analyze_catboost_errors.py` recomputes micro AUC from saved predictions, verifies
label-file hashes, aligns cases by Run No, verifies purged development dates and
reports generator, horizon-day, case-start-quarter and transition metrics.
The script rejects holdout, test, smoke and incomplete runs.

Output: `outputs/catboost_diagnostics/20261009_development_errors.csv`.
The output covers the original and probability presets on the same development folds.

Key findings for `probability_logloss`:

- Haukeli has the highest per-generator Brier error in both years: 0.233399
  (2020) and 0.113202 (2021). In 2020 its observed ON fraction is 0.244709,
  while its mean predicted probability is 0.531146. Its AUC is 0.857806.
- Haukeli's saved Logloss curves choose 5 trees in 2020 and 38 in 2021. The
  2020 validation loss increases with further training. More trees alone are
  therefore not a demonstrated remedy. Its frozen refit budget is 21 trees.
- For hours with a label change from the preceding horizon hour, Brier is
  0.240180/0.219875 (2020/2021), versus 0.083875/0.054488 for stable hours.
  These transitions account for only 0.56%/1.78% of evaluated hours after t0.
  Their high conditional error does not imply they dominate global micro AUC.
  Labels define these diagnostic groups only; they are never features.
- Errors do not consistently increase with horizon day. 2021 day 7 has better
  micro AUC (0.982923) than day 1 (0.975994).
- Performance varies by case-start quarter: 2020 Q3 micro AUC is 0.929370,
  while 2021 Q4 is 0.957520. This motivates checking seasonal robustness,
  but does not prove a particular cause or justify fitting to held-out labels.

These are descriptive results from overlapping cases, not independent samples
or statistical significance tests. Per-generator AUC is not an additive breakdown
of micro AUC. 2022 has not been used for this audit.

## New experiment

`decision_logloss` adds 23 features to the existing 193 (216 total), keeping
the same 14 models, chronological folds, seed, hyperparameters and Logloss stopping.

**Economics: 14 features.** Whole-horizon patterns in price minus the existing
capacity-weighted intake terminal water value: positive-margin streak length and
summed margin, distance to positive hours, remaining opportunities, normalized
margin and preceding/following 6h/24h mean margins. All windows include the current
hour and truncate at case boundaries. Distance 168 denotes no matching hour.
These are descriptors, not net-profit estimates: no efficiency, startup cost or
downstream marginal-water-value conversion is assumed.

**Water balance: 9 features.** Natural inflow against initial headroom, hypothetical
excess under zero release/transfer, and margins against supplied minimum volumes.
Compute reservoir excesses individually before aggregation to prevent cancellation.
Ratios use a 1%-capacity headroom floor and log1p. Use supplied maximum capacities,
initial volumes, in-horizon inflow integrated to Mm3 and minimum-volume requirements.
No realized future storage or inferred upstream dispatch is used. These proxies
are not a hydraulic simulation and do not imply mandatory turbine production.

`economics_logloss` (207 features) and `water_balance_logloss` (202 features) provide
group ablations. Compare the combined preset against the probability reference on
both years first; ablations identify whether either group helps or hurts.

## Verification and next action

No dependencies were added or installed. No models were fitted. Check-mode execution
passed for all presets. Tests cover hand-calculated streaks/windows and water
quantities, reservoir-wise excess, case isolation, label/metadata invariance,
invariance to realized future storage, CatBoost Pool construction, old feature
values/signatures, ablation equality and the existing target-order roundtrip.

The user should run `check`, then `smoke`, then `validate` for `decision_logloss`,
restarting the kernel between modes. Save results even if performance worsens.
Do not reuse an older preset's selected configuration with these features.
New training runtime and predictive performance are unmeasured. Preserve 2022
for frozen-candidate confirmation rather than repeatedly tuning against it.

## User-completed validation result

Run `20261009T085100_521077Z_validate_decision_logloss` completed in 1988.97 seconds
(33 min 9 s). Its input hashes, fold definitions and model parameters match the
probability reference exactly; feature engineering is the changed component.

| Preset | 2020 micro AUC | 2021 micro AUC |
|---|---:|---:|
| probability_logloss | 0.95766422 | 0.97857470 |
| decision_logloss | **0.96671334** | **0.98163615** |
| Difference | +0.00904912 | +0.00306146 |

Per-generator AUC improved for 12/14 generators in 2020 and 11/14 in 2021.
Haukeli improved from 0.857806 to 0.909452 (2020) and from 0.927213 to 0.942968
(2021), although its 2020 Brier worsened slightly. Lio worsened in 2021 from
0.955941 to 0.939166, and Vinje G1 worsened in both years. Overall mean Logloss
and Brier improved in both folds. No model hit the 1500-tree limit (largest retained
counts: 620 in 2020, 881 in 2021).

This is a measured improvement on reused development years, not independent
confirmation or a demonstrated Kaggle gain. The new preset is the strongest
standalone CatBoost development candidate so far. Keep its frozen configuration
for later confirmation and use the same data/folds as the reference for LightGBM.
Do not automatically replace individual generator models using these diagnostics.

## User-reported Kaggle result

The user completed full-data submission run
`20261009T093004_209176Z_submit_decision_logloss` in 718.99 seconds and reported
Kaggle ROC-AUC **0.98679**, compared with **0.98417** for the preceding probability
submission (+0.00262). The CSV was independently checked for 725 rows, exact
2353-column template schema/IDs and finite probabilities in [0, 1]. Scores were
reported by the user; leaderboard partition is unspecified. This supports retaining
the decision-feature setup as a reference, but is not private-score confirmation.
Preserve its selected configuration while comparing LightGBM on the same input
features and development folds.
