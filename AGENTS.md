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

Cleanup history: first pass (user-approved) deleted `xgboost.ipynb` (old
pooled 0.8586), `ensemble.ipynb` (old weak 0.9031), `rolling_origin.ipynb`,
`resampling_eda.ipynb`, `schedule_visualization.ipynb`, `test.ipynb`,
`gru_sequence.ipynb` + `gru_val_preds.npz`, and `loyo_cv.ipynb` (its
measured result, pooled OOF 0.9758, is recorded here). Second pass moved
the superseded-but-recorded notebooks into `archive/` (see below).

- `error_analysis.ipynb` — executed; Haukeli/Lio regime-error breakdown
  (report material).
- `kaggle_metric.py` — local metric implementation.
- `output.csv` — scored Kaggle submission 1 (0.98722).
- `output2.csv` — Kaggle submission 2 (volpath+cyclic variant), scored
  **0.98711** public — slightly below output.csv (0.98722), as expected for
  an equal-within-noise variant. Both stay selected: the better of the two
  counts on the private leaderboard.

## Archive folder `archive/`

Superseded notebooks, kept for their executed outputs (the record of
negative results for the report). Their internal data paths are NOT fixed
for the new location — they are read-only records, not runnable from
`archive/`.

- `archive/per_generator_xgboost.ipynb` — the original best-pipeline
  notebook (duplicated by `notebooks/xgboost_baseline.ipynb`).
- `archive/feature_experiments.ipynb` — the first closed feature search.
- `archive/insample.ipynb` — in-sample ceiling 0.9985 (all 2922 cases).
- `archive/xgboost_experiments.ipynb` — seeds / lr-trees / recency screen
  (all noise, nothing adopted).
- `archive/xgboost_stop_heldout.ipynb` — held-out stop experiment
  (0.9588 vs 0.9802; train_tail stands).
- `archive/gru_xgb_ensemble.ipynb` — GRU-vs-XGB ensemble comparison
  (GRU dropped).
- `archive/oof_predictions.npz` — LOYO OOF predictions (pooled 0.9758).

## Notebook folder `notebooks/`

Archived (moved to `archive/`, see above): `xgboost_stop_heldout`,
`xgboost_experiments`, `gru_xgb_ensemble`.

- `notebooks/eda_unit_commitment.ipynb` — label EDA (ON-rates per
  generator/year/hour/month, state stickiness and all-on/all-off weeks,
  ON-rate per price decile, solve-time stats, in-sample 2022 ceiling with
  the exact per-generator pipeline). Data paths are `../data/...`. Follows
  the markdown-documents / python-answers style. Section 1b added (not yet
  executed): ON-rate by day of year (15-day smoothed) per generator — the
  annual cycle motivating the sin/cos day-of-year encoding (the daily
  cycle is already section 1's hour-of-day panel).
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
  Section 4 added (not yet executed): five new feature groups screened on
  top of the 193 baseline with the same default XGB — cyclic (hod/dow
  sin+cos; NEW, never tested before), head (per-reservoir vol_head curve
  head at init volume + local head + local gross head vs plant
  outlet_line), spread (terminal water-value spread/std), anomaly
  (inflow vs 1958-2014 doy climatology), trailing (30-day trailing price
  and price minus it). Groups 2-5 are re-screens of the closed
  feature-experiments groups (all noise under the clean protocol) —
  included for the report story; only the cyclic group is genuinely new.
  A sixth group added after user question: **volpath** — per-reservoir
  volume fraction at each of the 7 day boundaries of the case week
  (baseline only uses the case-start level) + local capacity-weighted
  fractions per day. Also genuinely new, and the most promising idea of
  the batch (the level trajectory is optimizer state). Any promising
  delta still needs a clean-protocol test vs 0.9802.
- `notebooks/feature_importance.ipynb` — model-interpretation notebook for
  the report, executed: baseline per-generator XGB kept
  in memory, then (1) built-in gain importance normalized per generator,
  (2) mean |treeSHAP| via `pred_contribs` (no extra package) with a
  Spearman gain-vs-SHAP rank comparison, (3) dependence plots: predicted
  vs actual ON-rate by 20 quantile bins of local price-minus-water-value
  and by hour of day, for Tokke_G1 / Vinje_G1 / Lio_G1. Closes the
  "no model interpretation (-3)" grading deduction. Runtime ~15 min.
  Executed: fit cell reproduces the baseline exactly (micro 0.9802,
  per-gen AUCs match). Gain top: local_price_minus_val 0.157 normalized,
  then price_minus_val__Vinjevatn 0.038. SHAP top:
  price_minus_val__Vinjevatn 0.138, local_price_minus_val 0.127,
  price_minus_val__Venemo 0.070. Spearman(gain, SHAP) mean rankings =
  0.341 — the two views disagree in detail (gain concentrates on the
  local aggregate, SHAP spreads across per-reservoir columns) but agree
  the price-minus-water-value family dominates. Notable: the constraint
  feature minflow__b_Byrtevatn reaches gain rank 8 (0.0159) despite the
  constraints group ablation being negative — real weight for Byrte.
- `notebooks/xgboost_volpath.ipynb` — clean-protocol adoption test of the
  screen winners. Executed: baseline 0.9802 (reproduces exactly);
  +volpath 0.9804 (+0.0002); +volpath+cyclic 0.9806 (+0.0004). Both
  deltas within the ~0.001 noise floor -> **volpath NOT adopted** per the
  pre-registered bar; the feature search stays closed. Screen-to-clean
  shrinkage confirmed again: +0.0017 on the default-XGB screen became
  +0.0002 under the clean protocol (same pattern as the original closed
  feature search). Per-generator: Haukeli +0.0048 with volpath (but
  -0.0017 with volpath+cyclic), Lio -0.0081 (the two hard units move in
  opposite directions and net out), Kjela +0.0034 (vol+cyc), rest noise.
  Flagged option: volpath+cyclic is a legitimate SECOND Kaggle selection
  (the format allows two; the better counts on the private leaderboard) —
  equal-within-noise val performance means a free second draw, not an
  improvement claim.
- `notebooks/kaggle_submission.ipynb` — full-train XGBoost, test-case
  predictions, writes `../output.csv` (scored 0.98722). Extended with a
  second-prediction section (executed): the volpath+cyclic variant trained
  on all 2922 cases, writing `../output2.csv` as the second Kaggle
  selection (equal-within-noise val 0.9806; diversification pick, scored
  0.98711 public vs 0.98722 for output.csv — the better of the two
  selected counts on the private leaderboard). The
  fit-predict cell no longer deletes the base matrices (the variant
  section reuses them). Executed: in-sample 2022
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
