# AGENTS.md — project state log (TDT4173 course project)

This file is the working memory for AI-assisted sessions on this repo. Read it
before doing anything. Update it whenever a decision or a measured result lands.

## The task

Predict unit commitment for the Tokke-Vinje hydropower system: for every daily
case (a 168-hour week), output a continuous ON-probability (0-1) for each of
14 generators x 168 hours. Kaggle metric: **micro ROC-AUC** pooled over all
(case, hour, generator) points. Labeled data: 2922 cases, 2015-2022. Hidden
Kaggle test: 725 cases, 2023-01-01 to 2024-12-25 (Run No 2923-3647, see
`prediction_mapping.csv` / `sample_submission.csv` for the submission format).
**Measured on the hidden test: the full-train baseline submission scored
0.98722 micro AUC; the A-grade threshold is 0.98753 (gap 0.00031).**

The user is an ML beginner, runs all notebooks locally in Jupyter (this
assistant cannot execute notebooks), and pastes results back. Aspirational
goal: 0.995 (an A in the course) — never promise it; report measured numbers
only, never pre-state results in markdown before they are measured.

## Non-negotiable protocol decisions (do not regress on these)

- **Per-generator models**: 14 separate classifiers, one per generator, not
  one pooled model. Pooled XGBoost got 0.8586; per-generator got 0.9802.
- **Early stopping must never touch the validation set.** Stop on the last 90
  days of the *train* period ("train_tail" protocol). The old val-based early
  stopping was leaky and unstable; fixing it was worth +0.012 (0.9686 ->
  0.9802) and removed +/-0.02 noise from every feature comparison.
  Exception learned later: for a *recurrent net*, the stop days must also be
  held out of the training pool (see GRU v2 below) — a GRU that early-stops on
  data it trains on just memorizes it (stop AUC ~1.0, no stopping ever fires).
  Measured counterpoint for XGBoost (`notebooks/xgboost_stop_heldout.ipynb`):
  holding the 90-day stop window out of the train pool scores 0.9588 vs
  0.9802 — honest stopping on one season underfits (stops too early). For
  XGBoost, train_tail with the stop set inside train is the measured winner.
- **Feature search is CLOSED.** Five candidate groups (spill risk, gross head
  from vol_head curves, water-value spread, inflow anomaly vs 1958-2014
  climatology, trailing 30-day price) all measured within +/-0.0012 of the
  baseline under the clean protocol = noise. None added. Documented in
  `feature_experiments.ipynb`. Do not re-open without a new idea that is not
  a re-encoding of the same physics.
- **Baseline numbers to beat**: XGBoost micro AUC **0.9802** (val 2022),
  pooled LOYO OOF **0.9758** (all 2922 cases, the most stable comparison
  number). In-sample ceiling: 0.9985 (`insample.ipynb`) — the features nearly
  determine the labels; the gap is generalization, not information.

## Current pipeline (best so far)

14 per-generator `xgb.XGBClassifier`s on 193 features (186 shared + 7
`local_*` aggregates over the plant's intake reservoirs). Hyperparameters:
`n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
colsample_bytree=0.8, tree_method='hist', eval_metric='auc',
early_stopping_rounds=25, random_state=0`. Train 2015-2021, stop on last 90
days of train, score val 2022. Lives in `per_generator_xgboost.ipynb`.

Feature groups (all in `feature_matrix()` + `generator_features()`, copied
verbatim across notebooks): calendar (hour/dow/month/doy sin+cos, horizon
hour, hours remaining), price (level, week stats, rank, prefix/suffix means,
offsets, daily stats), per-reservoir (init vol/frac, terminal water value,
price minus value), per-inflow-source (flow, week mean, cumulative and
remaining Mm3), min-flow / min-volume constraints, and the 7 local columns
(init frac, terminal value, price minus value, inflow, cum/rem/available
fractions over the plant's feeder reservoirs).

## Measured results so far

| notebook | protocol | micro AUC |
|---|---|---|
| `xgboost.ipynb` | pooled single model, leaky early stop | 0.8586 |
| `per_generator_xgboost.ipynb` | per-gen, train_tail stop | **0.9802** (val 2022) |
| `insample.ipynb` | train=test (ceiling) | 0.9985 |
| `ensemble.ipynb` | XGB+LGBM+GRU, 19 feats, leaky | 0.9031 |
| `loyo_cv.ipynb` | leave-one-year-out, 8 folds | **0.9758** pooled OOF |
| `gru_sequence.ipynb` v1 | stop set inside train, 60-epoch cap | 0.9696 (val 2022) |
| `gru_sequence.ipynb` v2 | stop days held out, honest stopping | 0.9579 (val 2022) |

LOYO per-year micro AUC 2015-2022: 0.9834, 0.9872, 0.9872, 0.9781, 0.9774,
0.9018 (COVID outlier year), 0.9832, 0.9802. Saved `oof_predictions.npz`
(one OOF score per case-hour-generator) — reuse it as the baseline to beat
instead of retraining.

GRU v2 per-generator val 2022: Byrte 0.9743, Haukeli 0.8850, Hogga 0.9611,
Kjela 0.9575, Lio 0.9273, Songa 0.9846, Tokke_G1 0.9811, Tokke_G2 0.9845,
Tokke_G3 0.9801, Tokke_G4 0.9858, Vesle Kjela 0.9853, Vinje_G1 0.9794,
Vinje_G2 0.9752, Vinje_G3 0.9566. GRU val-2022 predictions saved to
`gru_val_preds.npz`.

## Error analysis findings (`error_analysis.ipynb`, val 2022)

- Haukeli_G1 (XGB 0.9352) and Lio_G1 (XGB 0.9225) are the hard units. Both
  fail via **whole-week all-or-nothing regimes** (maintenance outages /
  must-run weeks): Haukeli has fully-off weeks the model predicts on (its
  reservoir fill is constant 40-60% all 2022 — no signal in the data); Lio is
  FN-heavy (on 81% of hours at fill 0-0.2, must-run at low fill) plus fully-on
  weeks predicted off.
- Conclusion: the missing signal is a **regime variable (outage schedule)**
  that is not in the provided data. Do not chase it with more features.
- The GRU experiment was designed to test exactly this: v2 Haukeli got
  *worse* (0.8850, held-out stop AUC only 0.8939 — it cannot model Haukeli
  even on nearly-in-sample data). This supports the "missing variable"
  conclusion.

## Resolved question: GRU vs XGBoost ensemble (val 2022)

Measured in `notebooks/gru_xgb_ensemble.ipynb` (sanity gates passed: XGB
0.9802, GRU 0.9579): mean ensemble **0.9809** (+0.0007 = within the
+/-0.001 noise floor), rank ensemble **0.9556** (much worse — per-generator
percentile ranks destroy the between-generator level information that the
pooled micro AUC exploits; mean probabilities preserve it).

**Decision per the pre-registered rule: the GRU is dropped.** The submission
is XGBoost-only by default.

One real finding to keep: the complementary signal exists but is
concentrated on the two regime units — Lio_G1 ensemble 0.9570/0.9638 vs XGB
0.9225, Haukeli_G1 0.9463 vs 0.9352 — while the GRU slightly drags the
twelve easy units, netting ~0 at micro level. Optional flagged variant for
the submission notebook: XGB everywhere + mean-ensemble on Haukeli/Lio only
(naively ~+0.003 micro on val). Caveat: selected on val 2022, no GRU OOF
exists to validate honestly; val numbers for it would be optimistic.

## Next steps (in priority order)

1. DONE: GRU-vs-XGB ensemble comparison — GRU dropped (see resolved
   question above); optional Haukeli/Lio-only hybrid flagged for the
   submission notebook.
2. DONE: seed-averaged XGBoost, lr 0.03 at 1000/1500 trees, and recency
   weighting all screened in `notebooks/xgboost_experiments.ipynb` — every
   delta within the measured noise floor (seed range 0.0014). Nothing
   adopted; baseline config stands. Model-improvement search on this axis
   is exhausted.
3. DONE: `notebooks/kaggle_submission.ipynb` executed — trained the
   baseline per-generator XGBoost on ALL 2922 labeled cases, predicted the
   725 test cases, wrote `output.csv` (repo root). **Kaggle score:
   0.98722.** A-grade needs 0.98753 (gap 0.00031). Workflow confirmed
   correct: develop on train/val, freeze, retrain on all data, submit;
   do not iterate against the Kaggle score.
4. `notebooks/feature_screening.ipynb` (authored, not yet executed): a
   simple default-XGB screen for the report story — (1) univariate AUC per
   feature per generator (direction-corrected), (2) feature-group ablation
   with default `xgb.XGBClassifier` (100 trees, no early stopping),
   (3) top-10-features-only fit. Indicative only; any promising feature
   still needs a clean-protocol test vs 0.9802 before adoption. Runtime
   ~10-15 min.
5. README rewrite + `requirements.txt` update (missing: lightgbm, torch,
   pyyaml, nbformat/nbclient) — low priority, before delivery.
4. README rewrite + `requirements.txt` update (missing: lightgbm, torch,
   pyyaml, nbformat/nbclient) — low priority, before delivery.

## Environment and tooling gotchas

- venv `.venv` (Python 3.14). Notebooks run under Jupyter with the `.venv`
  kernel. **This assistant cannot execute notebooks or python** — author
  notebooks as raw JSON (`write_file` for new files, `sed`/`jq` for edits,
  validate with `jq empty`), the user runs them and pastes output.
- **torch/MPS and xgboost must not share a kernel** on this Mac: after heavy
  MPS use, xgboost fits in the same process hang for hours or crash the
  kernel (happened twice in `gru_sequence.ipynb`). Keep them in separate
  kernels; save predictions to `.npz` to pass results between kernels.
- Memory: the cached feature matrices are ~300 MB each; a notebook holding
  four of them plus per-generator copies can OOM. Delete caches when done
  (`del base_fit; gc.collect()`), save valuable predictions to disk early.
- `BCEWithLogitsLoss` requires float targets — `labels_for()` returns uint8;
  cast with `.astype(np.float32)` before `torch.from_numpy`.
- Data: price is hourly (15-min resolution only from Oct 2025, after the
  test window). Topology YAML `data/extended/Tokke_Vinje_topology.yaml` is
  authoritative for plant -> reservoir mapping (FEEDERS via
  `supplying_reservoirs`).

## Notebook inventory (repo root)

Cleanup done (user-approved): deleted `xgboost.ipynb` (old pooled 0.8586),
`ensemble.ipynb` (old weak 0.9031), `rolling_origin.ipynb`,
`resampling_eda.ipynb`, `schedule_visualization.ipynb`, `test.ipynb`,
`gru_sequence.ipynb` + `gru_val_preds.npz` (GRU dropped; the Haukeli/Lio
hybrid would need a full retrain anyway), and `__pycache__`. Their measured
results live in this file. Most were untracked in git — deletion was
permanent.

- `per_generator_xgboost.ipynb` — current best pipeline, executed, with
  experiment writeup and takeaways.
- `feature_experiments.ipynb` — executed; the closed feature search grid.
- `loyo_cv.ipynb` — executed; pooled OOF 0.9758; saves `oof_predictions.npz`.
- `error_analysis.ipynb` — executed; Haukeli/Lio regime-error breakdown.
- `insample.ipynb` — ceiling 0.9985.
- `kaggle_metric.py` — local metric implementation.
- `output.csv` — the scored Kaggle submission (0.98722).

## Notebook folder `notebooks/`

- `notebooks/eda_unit_commitment.ipynb` — label EDA (ON-rates per
  generator/year/hour/month, state stickiness and all-on/all-off weeks,
  ON-rate per price decile, solve-time stats, in-sample 2022 ceiling with
  the exact per-generator pipeline). Data paths are `../data/...`. Follows
  the markdown-documents / python-answers style.
- `notebooks/xgboost_baseline.ipynb` — the per-generator XGBoost baseline
  (same 193 features, same hyperparameters, train_tail stop, val 2022)
  reproduced self-contained in this folder as the clean starting point for
  the planned experiments (hyperparameters, seed averaging, recency
  weighting). Executed: reproduces 0.9802 exactly — every per-generator AUC
  and tree count matches `per_generator_xgboost.ipynb` to 4 decimals
  (pipeline fully deterministic at random_state=0). 13/14 generators sit
  at the 400-tree cap; the only real early stop is Songa at 133 trees
  (also the best unit, 0.9966) — confirms the stop set is effectively
  inert and n_estimators=400 is the operating point. Micro is dragged by
  Haukeli (0.9352) and Lio (0.9225); the other twelve sit at 0.97-0.996.
- `notebooks/xgboost_stop_heldout.ipynb` — protocol experiment: the stop
  window (last 90 days of 2021, same window as baseline) is removed from the
  training pool so the early-stopping signal is out-of-sample; full 2022
  val, everything else identical to the baseline. Comparison target:
  0.9802. If within noise, the train_tail protocol stands; if it wins, it
  becomes the protocol (and is the structure recurrent models require).
  Executed: **0.9588** vs baseline 0.9802 (-0.021) — clearly worse.
  Mechanism confirmed by tree counts: all models stopped far below the
  400 cap (Vinje_G1 5 trees, Tokke_G4 20, Vesle Kjela 21, Vinje_G3 37;
  only Byrte 216 / Hogga 244 kept substantial trees), and the losses
  concentrate on the earliest-stopped, hardest units: Lio 0.9225 ->
  0.8578, Haukeli 0.9352 -> 0.9055, Vinje_G1 0.9835 -> 0.9704. The
  out-of-sample stop signal on a single-season window peaks early and
  patience fires on noise; the baseline's in-sample stop set is
  effectively an inert "run to 400 trees" rule, and 400 is a good
  operating point.
  Decision: train_tail stands. Honest early stopping on a single-season
  90-day window selects a tree count that is wrong for the rest of the
  year. Note this does NOT invalidate the GRU exception: a recurrent net
  must still hold its stop days out of its own training pool (it
  memorizes them otherwise); for XGBoost the in-sample stop set is
  effectively inert and harmless.
- `notebooks/xgboost_experiments.ipynb` — cheap val-2022 screen of the three
  planned model experiments under the unchanged baseline protocol:
  (1) seed averaging, 5 seeds, mean-prob and mean-rank ensembles;
  (2) lr 0.03 with 1000 and 1500 trees; (3) recency weighting, per-case
  weights decay**(2021-year) for decay 0.9 and 0.75. Executed — all within
  noise, nothing adopted: single seeds 0.9790/0.9802/0.9804/0.9804/0.9790
  (noise floor: range 0.0014), seed mean-prob ensemble 0.9802, lr 0.03 at
  1000/1500 trees 0.9803/0.9804, recency decay 0.9/0.75 0.9804/0.9790.
  Baseline config stands (lr 0.05, 400 trees, seed 0, uniform weights).
  Lessons: seed averaging gains nothing (seeds too correlated; helped
  Haukeli 0.9352 -> 0.9404 individually, micro unmoved); rank ensembles are
  the wrong tool for the pooled metric, now measured twice (seed rank
  ensemble 0.9492, GRU rank ensemble 0.9556 — per-generator rank
  normalization destroys between-generator level information); strong
  recency weighting hurts (-0.0012), mild is neutral.
- `notebooks/gru_xgb_ensemble.ipynb` — the GRU-vs-XGB ensemble comparison,
  isolated from torch. Executed: mean 0.9809, rank 0.9556 vs XGB 0.9802;
  GRU dropped (see resolved question).
- `notebooks/feature_screening.ipynb` — feature-screening notebook for the
  report story. Executed (one bug fixed in the final cell: the per-generator
  top-10 table passed None as the column list; now returns the top-10
  directly — rerun just that cell). Measured (default XGB, train 2015-2021,
  val 2022, no early stopping): univariate top feature is
  `price_minus_val__Vinjevatn` (mean AUC 0.8258 across generators); the
  top-20 is dominated by `price_minus_val__*` / `term_val__*` columns.
  Group ablation (all features 0.9764): -reservoir 0.9355 (drop 0.0410 —
  the water-value group carries nearly everything), -price 0.9729
  (0.0035), -calendar 0.9757 (0.0007), -inflow 0.9767 (-0.0003),
  -constraints 0.9776 (-0.0012), -local 0.9781 (-0.0017; dropping
  inflow/constraints/local slightly *helps* the default model = noise /
  mild regularization, consistent with the closed feature search).
  Top-10-features-only: 0.9548 — signal is spread across many features,
  not concentrated in a few. Implication: new-feature ideas should target
  the reservoir/water-value axis (interactions), not inflow/constraints.
- `notebooks/kaggle_submission.ipynb` — full-train XGBoost, test-case
  predictions, writes `../output.csv` (see next steps). Executed: in-sample 2022
  fit is ~1.0 per generator (Vinje_G1 0.9999, rest 1.0) — at 61k rows the
  400-tree model fully memorizes, so this says nothing beyond "features are
  expressive"; the informative ceiling remains 0.9985 on all 2922 cases
  (`insample.ipynb`).
  Other measured EDA results (all 2922 cases): flips are rare (0.3-2.2% of
  hours, P(on|on) 0.96-0.997) and 30-80% of case-weeks are pure all-on or
  all-off regimes (Haukeli 81%, Byrte 74%, Lio 70%) — the label is mostly a
  week-level regime decision with hourly modulation. ON-rates drift strongly
  by year (Haukeli 0.87 in 2015 -> 0.14 in 2022; Songa 0.70 in 2020 -> 0.12
  in 2022). ON-rate is non-monotone across price deciles for nearly every
  unit (several most-ON at the *lowest* prices, spring flood) — water state
  dominates price level, corroborating the closed feature search.

## Working style for this repo

- KISS notebooks: markdown cells only document what is being tested, placed
  before the code cells that test it. All analysis and results live strictly
  in Python (code + printed outputs/plots); no results markdown, no
  interpretation prose in markdown cells.
- Every comparison states its protocol (which early-stopping set, which
  years); never compare numbers across protocols.
- One experiment per notebook; keep the notebook self-contained (data
  loading + features are copied in, not imported).
