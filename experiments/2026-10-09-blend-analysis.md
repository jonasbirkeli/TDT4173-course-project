# Saved-prediction blend analysis

Compared fixed probability blends on the two existing purged development folds,
without fitting any models. Reference: `probability_logloss`. Candidate weights
were fixed at 0.25, 0.50 and 0.75 before evaluation; the reference receives the
remaining weight. The same weight is used for all generators and both years.

The audit checks completed validation status, matching raw-input manifests and
fold definitions, the current raw-label hash, prediction IDs/columns/probabilities,
and exact reproduction of both models' recorded micro AUC (within 1e-8).
It rejects folds extending beyond the reserved 2022 boundary.

| Candidate | Candidate weight | 2020 micro AUC | 2021 micro AUC |
|---|---:|---:|---:|
| Reference alone | 0 | 0.95766422 | 0.97857470 |
| baseline_auc | 0.25 | 0.95700979 | 0.97841608 |
| baseline_auc | 0.50 | 0.95516636 | 0.97812416 |
| baseline_auc | 0.75 | 0.95190290 | 0.97766616 |
| upstream_logloss | 0.25 | 0.95898008 | 0.97902994 |
| upstream_logloss | 0.50 | 0.95940715 | 0.97914061 |
| upstream_logloss | 0.75 | 0.95895231 | 0.97901217 |
| shared_logloss | 0.25 | **0.96031427** | **0.97941690** |
| shared_logloss | 0.50 | 0.95986889 | 0.97939043 |
| shared_logloss | 0.75 | 0.95672911 | 0.97863334 |

Although the shared model was weaker alone, mixing 25% shared with 75% separate
probability predictions improved both development years. The gains were 0.00265005
and 0.00084220 over the reference. This supports investigating complementary models;
it does not demonstrate an improvement on the reserved holdout or Kaggle. Selection
from these nine blends reuses development labels and requires frozen-candidate
confirmation. Do not tune additional weights repeatedly on the public leaderboard.

The original AUC-stopped baseline did not improve the reference at the tested
weights. Additional upstream features were inconsistent as a standalone candidate,
but all three tested blends with them improved both reference folds.

Reproduce with `analyze_catboost_blends.py`. Full precision results are saved at
`outputs/catboost_diagnostics/20261009_blend_comparison.csv`. This development script
is not a runtime dependency of the notebook and does not create submissions.

Next: finish the controlled decision-feature experiment. Keep the 75/25 separate/
shared blend as another candidate. A final blend requires both model strategies
refitted on the appropriate training partition with frozen configurations; existing
development models cannot be used as substitutes for full-data fits. Include both
strategies' preprocessing, fitting and prediction in the final runtime budget.
