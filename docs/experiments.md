# Experiment record

For the short round-by-round explanation of features, motivation and results,
read [Development journey](development_summary.md).

The complete original 5,246-line chronology is preserved in
[experiments_history.md](experiments_history.md). This file is the concise current
index; the history retains the original narrative, including obsolete entry points
and superseded decisions. Use this index and the active notebook for current settings.

## Current decision

Use `sequence_model.ipynb` for the **standalone TCN with 6-hour allocation
features** (`terminal_resolution_6h`, v35). It produces one probability for
each of 14 units and 168 planning hours. There is no ensemble.

- Best user-reported TCN public score: **0.99261**.
- Development micro ROC-AUC: **0.9809209327 (2020)** and **0.9892533563 (2021)**.
- One network, **343,034 parameters**, **913 hourly and 182 static channels**.
- Final fit: **22 epochs on all 2,922 labeled 2015–2022 cases**.
- Known submission: `outputs/tcn_candidates_20261027_resolution/terminal_resolution_6h_submission.csv`.
- Submission SHA256: `7afc5ebf0daffd0a482b668f18b19bbd70f7ffe151c49f4b2db3e3c2e2231ede`.

Public scores below were reported by the user, not independently retrieved.
Private performance is unknown. The target above 0.99350 is aspirational;
no fixed ROC-AUC threshold establishes an A grade.

## How to read the evidence

This is a retrospective summary, not a claim that the complete search was
preregistered. Earlier public feedback influenced subsequent starting references.
Within the later rounds, candidate selection used local chronological results;
reported public feedback arrived after the respective frozen exports.

The original detailed log, old notebooks and earlier instructions are preserved
in the verified local cleanup archive identified by `outputs/cleanup/latest.json`.
Native experiment evidence and all submissions remain under `outputs/`.
These ignored local files are not pushed to Git and are not runtime dependencies.
The summary here retains the decisions, numerical results and unsuccessful trials
needed to write the report.

## Data and evaluation protocol

A case starts at midnight UTC and covers 168 hours. Adjacent daily cases overlap.
The 2,922 labeled cases start in 2015–2022. The 725 test cases start on
2023-01-01 through 2024-12-25. The public/private partition is not established;
do not equate 2023 with public and 2024 with private.

For start S use storage at S, price/inflow/constraints during [S, S+168h), and
terminal water values at S+168h. Future observed storage and optimizer calculation
time are excluded. Timestamp joins use named objects and UTC.

The official score is micro ROC-AUC over all case/unit/hour entries, including
cross-unit ranking. It is not accuracy or the average of unit-specific AUCs.
Cases are split before expansion. Training horizons must end by the scoring
start: training_start + 168h <= scoring_start. Preprocessing learns only from
fitting cases. Millions of overlapping binary labels are not independent samples.

| Scoring year | Fitting cases | Scoring cases | Fixed outer epochs in current recipe |
|---|---:|---:|---:|
| 2020 | 1,820 | 360 | 20 |
| 2021 | 2,186 | 359 | 42 |

The current 6-hour recipe reuses earlier v32 duration choices: 20/42 outer
budgets and 22 final epochs. Its `validate` mode fits two models without new
inner stopping; `freeze` records the existing 22-epoch budget without fitting.
These fixed-budget comparisons do not establish each recipe's separately tuned
optimum. Earlier experiments used inner stopping on 2019/2020 and fresh outer
refits. The development years have repeated exposure. **2022 also has prior
project exposure** and is not an untouched confirmation set. It is included in
final submission training. No new 2022 scoring selected the 6-hour model or the
later curve-bound control.

## Main development path

| TCN stage | 2020 AUC | 2021 AUC | User-reported public AUC | Interpretation |
|---|---:|---:|---:|---|
| Dense river features | .9749072383 | .9841080911 | .98895 | Useful hydraulic/river reference |
| Binary transition-cost scenarios (v8) | .9763825881 | .9842951006 | .98966 | Improved both local years |
| Quantity/count descriptors | .9781826445 | .9853343721 | .98875 | Local gains did not transfer to reported public score |
| Energy/count descriptors | .9777192732 | .9857195687 | .98928 | Alternative annual tradeoff |
| Graph input routing | .9786063608 | .9876630481 | .99159 | Better local scores than local-only control |
| Local plant routing | .9782461670 | .9862995905 | .99176 | Public ordering reversed the local comparison |
| Local hydraulic context (v22) | .9788354325 | .9879521391 | .99197 | Adjacent reservoir/river/tunnel inputs routed locally |
| Initial-head best-point terminal conversion (v32) | .9806587999 | .9891596511 | .99240 | Large improvement on both development years |
| **6-hour allocation (v35)** | **.9809209327** | **.9892533563** | **.99261** | **Current reference** |
| Curve-bound operating descriptors | .9809822430 | .9892376461 | .99254 | Tiny mean gain, one-year regression; not promoted |

These are feature-pipeline comparisons. Inputs and sometimes frontend capacity
change, so the table does not isolate causal architectural effects. The
public ordering of graph/local and count variants is a reason to avoid choosing
models from public feedback alone.

### Why the 6-hour interval was kept

The original 12-hour LP interval was a computational approximation, not a
physical time constant. Retesting with the v32 terminal conversion gave:

| LP interval | 2020 AUC | 2021 AUC | Equal-year mean |
|---|---:|---:|---:|
| 12 hours | .9806587999 | .9891596511 | .9849092255 |
| **6 hours** | **.9809209327** | **.9892533563** | **.9850871445** |
| 1 hour | .9804206958 | .9891709601 | .9847958279 |

Network, input dimensions, seed and training budgets were held fixed. Six hours
won both years. All 2,547 required/probe LP cases solved. Hourly LPs nevertheless
needed up to 29.1601 million m3 of artificial storage slack; solver success does
not imply physical feasibility. Six-hour cases had zero artificial storage
slack. The TCN still consumes and predicts **all 168 individual hours**.

### Topology and physical features

The supplied network has 17 reservoirs, 9 plants, 14 units, 37 rivers and 5 tunnels.
A conservative-flow LP uses the named directed connections to couple releases,
reservoir storage and downstream opportunities. It supplies allocation, projected
storage and dual sensitivities. The model routes adjacent reservoir, tunnel,
river and economic descriptors into nine local plant encoders, alongside a
shared temporal encoder. The selected recipe has **no neural graph-message
passing and no learned reservoir-ID embedding table**.

Terminal local EUR/MWh values are converted using a 33-point best-operating-point
proxy at initial head and preferred generating/main-waterway routes. Receiving
reservoir values are added downstream. This was useful empirically but remains
an approximation to SHOP. Full tunnel-pressure balance, changing-head optimization,
spill behavior and binary commitment are not solved exactly.

The 54 binary DP features are 9 plants x 3 relative transition-cost scales x
2 initial-state assumptions. They describe persistence opportunities, not feasible
unit schedules or exact currency-converted startup profits.

## Other predictor types

XGBoost is Jonas's active tree-model direction; the user's active model is TCN.
Earlier CatBoost, LightGBM and blends are historical comparisons, not required
for running the selected TCN. Their notebooks are archived locally.

| Pipeline | 2020 micro AUC | 2021 micro AUC |
|---|---:|---:|
| CatBoost decision RSM | .9678485591 | .9827461050 |
| Neutral LightGBM | .9643583 | .9833711 |
| Earlier wide temporal CNN | .95586145 | .96460585 |
| Original frozen three-family blend | .9731119557 | .9853425654 |
| Later 35/30/35 development-selected blend | .9735356938 | .9856353219 |
| XGBoost decision reference, 216 features | .9632372220 | .9827852977 |
| XGBoost topology descriptors, 279 features | .9625373539 | .9823556500 |
| XGBoost topology + allocation, 298 features | .9829521212 | .9879182383 |

Topology descriptors alone lost both years in XGBoost; allocation features helped.
The user's .98877 score belongs to `three_model_submission.csv`, not to the
later selected pure XGBoost CSV. No public score is recorded for that pure CSV.
The original blend's .9837118366 confirmation on previously exposed 2022 belongs
to the original frozen blend, not the later 35/30/35 weights.

## Unsuccessful, partial and unfitted experiments

- Fourteen independent TCNs: .9745214599/.9815896016, below the contemporary shared
  river reference .9749072383/.9841080911. Independent models filled separate
  columns; they were not averaged.
- Wider/global/local-history, auxiliary count loss, pair loss, loading, water
  windows, pressure and buffer features were not consistently beneficial. The
  complete measured records appear in the table below.
- Best-point-only and local-temporal controls in the v32 round were checked but
  not fitted before its time gate. No negative scores are inferred for them.
- In the latest round, direct augmentation with 63 legal-range descriptors
  completed 2020 at .9799966525; its 2021 fit was abandoned. It has no completed
  two-year mean. The curve-limited LP prototype was checked, never fitted.
- A neutral curve-only correction preserved initial baseline parameters,
  outputs and RNG exactly, yet scored .9809258249/.9890665743. Mathematical
  consistency and controlled initialization did not guarantee better AUC.

The latest round produced no candidate improving both development years. A
separate **post-hoc exploratory export** selected the mean-positive curve-bound
control: +.0000613103 in 2020 and -.0000157101 in 2021, mean +.0000228001.
Its user-reported public score .99254 is .00007 below the .99261 reference.
The reference remains unchanged. Paired 128-replicate case-block intervals for
the mean difference include zero: 7-day [-.000217464, +.000268637],
28-day [-.000398131, +.000325543]. These exposed-year, fixed-prediction diagnostics
are not independent confirmation, training-seed uncertainty or private evidence.

## Completed twofold TCN trial inventory

Rows below come from completed local JSON reports. Scores are micro AUC for
2020/2021, not Kaggle scores. Repeated rows from different feature generations
are not interchangeable. Numerical precision supports matching source records,
not a claim that tiny differences are meaningful.

| Round | Recipe | 2020 | 2021 |
|---|---|---:|---:|
| 20261022T005535Z_specialization | `river_1h` | 0.9739198509 | 0.9824438151 |
| 20261022T005535Z_specialization | `river_6h` | 0.9751435010 | 0.9837498326 |
| 20261022T005535Z_specialization | `river_commitment_dp` | 0.9763825881 | 0.9842951006 |
| 20261022T005535Z_specialization | `river_dispatch` | 0.9765040808 | 0.9827344389 |
| 20261022T005535Z_specialization | `river_heads` | 0.9707177514 | 0.9766748976 |
| 20261022T005535Z_specialization | `river_multiscale6` | 0.9753724783 | 0.9840399946 |
| 20261022T230854Z_refinement | `commitment_auxcount` | 0.9768418007 | 0.9810702188 |
| 20261022T230854Z_refinement | `commitment_counts` | 0.9781826445 | 0.9853343721 |
| 20261022T230854Z_refinement | `commitment_energy` | 0.9777192732 | 0.9857195687 |
| 20261022T230854Z_refinement | `commitment_rank` | 0.9760776049 | 0.9840488446 |
| 20261022T230854Z_refinement | `commitment_sensitivity` | 0.9757406783 | 0.9847173124 |
| 20261022T230854Z_refinement | `commitment_wide` | 0.9758677242 | 0.9857489299 |
| 20261024T235757Z_direct_search | `river_global16` | 0.9774359998 | 0.9848958063 |
| 20261024T235757Z_direct_search | `river_graph16` | 0.9786063608 | 0.9876630481 |
| 20261024T235757Z_direct_search | `river_history_all` | 0.9757418391 | 0.9832377522 |
| 20261024T235757Z_direct_search | `river_history_storage` | 0.9748458141 | 0.9850477147 |
| 20261024T235757Z_direct_search | `river_local16` | 0.9782461670 | 0.9862995905 |
| 20261024T235757Z_direct_search | `river_local32` | 0.9785984593 | 0.9855973952 |
| 20261024T235757Z_direct_search | `river_local_history16` | 0.9785503695 | 0.9857547697 |
| 20261024T235757Z_direct_search | `river_local_temporal16` | 0.9775271625 | 0.9864086228 |
| 20261026T014506Z_local_physics | `graph_gated16` | 0.9783770799 | 0.9874224163 |
| 20261026T014506Z_local_physics | `local_hydraulic16` | 0.9788354325 | 0.9879521391 |
| 20261026T014506Z_local_physics | `local_loading16` | 0.9775500454 | 0.9878627863 |
| 20261026T014506Z_local_physics | `local_loading_only16` | 0.9772778671 | 0.9849239179 |
| 20261026T014506Z_local_physics | `local_water_windows16` | 0.9784014705 | 0.9875606194 |
| 20261027T001630Z_resolution | `local_buffers16` | 0.9791091494 | 0.9874526770 |
| 20261027T001630Z_resolution | `local_hydraulic6h` | 0.9785399123 | 0.9866797995 |
| 20261027T001630Z_resolution | `local_pressure16` | 0.9789281828 | 0.9870578905 |
| 20261027T001630Z_resolution | `local_terminal_bestpoint16` | 0.9806587999 | 0.9891596511 |
| 20261027T064625Z_time_resolution | `terminal_resolution_1h` | 0.9804206958 | 0.9891709601 |
| 20261027T064625Z_time_resolution | `terminal_resolution_6h` | 0.9809209327 | 0.9892533563 |
| 20261027T083654Z_dynamic_head | `bestpoint_economics6h` | 0.9809172333 | 0.9891923625 |
| 20261027T083654Z_dynamic_head | `curve_bounded_physics6h` | 0.9809822430 | 0.9892376461 |
| 20261027T083654Z_dynamic_head | `curve_neutral_residual6h` | 0.9809258249 | 0.9890665743 |
| 20261027T083654Z_dynamic_head | `dynamic_consistent6h` | 0.9804872566 | 0.9896390857 |
| 20261027T083654Z_dynamic_head | `dynamic_dispatch6h` | 0.9805214441 | 0.9896339250 |

## Reproduction and evidence

The selected 6-hour raw-data refits took **500.19 / 476.59 seconds**, with peaks
**9.11 / 9.11 GiB**, four CPU threads and identical CSV hashes. The rejected
curve-bound control also reproduced exactly: **579.98 / 770.99 seconds**,
**9.13 / 9.13 GiB**. These are fresh-process raw-input runs, not independent seeds;
OS caching and concurrent development work prevent controlled speed comparisons.

The v32 recipe had a locked, previously exposed 2022 comparison
(.9890916613 vs v22 .9872446952) and matched-budget seed17 gains on both development
years. Those results do **not** belong to the later 6-hour recipe. Earlier routing
checks on seed17/2019 similarly concern their own source versions, not proof of
private superiority of every descendant.

| Evidence | Local directory under `outputs/tcn_development/` |
|---|---|
| River features | `20261020T224435Z_feature_search/` |
| Shared/independent and binary DP | `20261022T005535Z_specialization/` |
| Count/energy/loss controls | `20261022T230854Z_refinement/` |
| Graph/local/history controls | `20261024T235757Z_direct_search/` |
| Local hydraulic routing | `20261026T014506Z_local_physics/` |
| Best-point terminal conversion | `20261027T001630Z_resolution/` |
| Selected 1/6/12-hour comparison | `20261027T064625Z_time_resolution/` |
| Later physics controls | `20261027T083654Z_dynamic_head/` |

Native reports preserve case partitions, raw hashes, per-unit/season/horizon
metrics, parameters, runtimes and selected configurations. Gradient sensitivity
summaries cover small input subsets; they are not causal or population-wide
feature importance. The raw-input, frozen-config and mapping checks fit no models.

The cleanup promotes the frozen .99261 recipe into `sequence_model.ipynb`,
clarifies its configuration and prose, and adds an exhaustive feature catalog.
No training is authorized during cleanup. Model-function fingerprints, frozen
parameters, feature arrays, scalers and initial predictions must match the
preserved recipe; new raw-refit timing or Kaggle quality is not remeasured here.

## References and report limits

- [Bai, Kolter and Koltun, 2018](https://arxiv.org/abs/1803.01271): motivation for
  dilated residual temporal convolutions. Our padding is symmetric/noncausal,
  because the complete planning-week exogenous inputs are supplied; this is a
  TCN-inspired temporal CNN, not an exact causal implementation from the paper.
- [SINTEF hydro UC study, EEM 2026](https://www.sintef.no/en/publications/publication/019f93eb7024-9db6a5de-cdc7-4ad4-9d28-d051824cd633/): domain motivation for convolutional
  models. Reported F1 cannot be converted into this competition's micro AUC.
- [SHOP water values](https://docs.shop.sintef.energy/examples/water_values/water_values.html),
  [generators](https://docs.shop.sintef.energy/objects/generator/generator.html),
  [tunnels](https://docs.shop.sintef.energy/objects/tunnel/tunnel.html) and
  [rivers](https://docs.shop.sintef.energy/objects/river/river.html): physical
  definitions, not proof that the simplified LP reconstructs all SHOP settings.
- [Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html):
  model-selection bias motivates reporting repeated-year exposure and failures.

At least two predictor types, four EDA activities, feature engineering,
interpretation and hyperparameter reasoning must be documented in the final
report. The separate report, student/team metadata and two official offline
Short notebooks remain unfinished. Each final notebook has a 12-hour ceiling
on a normal PC (up to four CPU cores and 32 GB RAM); it must fit from supplied
raw resources without separate source helpers or pretrained artifacts.
