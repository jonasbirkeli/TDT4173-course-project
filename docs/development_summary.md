# Development journey: what changed and what helped

This is the short narrative for writing the report. The
[complete chronology](experiments_history.md) preserves the original details;
the [experiment record](experiments.md) contains the numerical trial inventory.
All public scores here are user-reported; private performance is unknown.

Development was not a straight sequence of improvements. We compared branches,
discarded some additions, and returned to earlier references. The final model
does **not** contain every feature or architecture tried along the way.

## Progress at a glance

Dates identify experiment rounds, not necessarily the day a score was reported.
Public scores compare whole pipelines; they do not isolate individual features.

| Round | What changed | Reported public AUC | What happened next |
|---|---|---:|---|
| Early CatBoost | Raw inputs and basic calendar, price, intake and water summaries | .98417 | Starting probability reference |
| October 9: decision context | Add favorable-price streaks/windows and water/headroom pressure | .98679 | Retain decision-context features |
| October 13-14: other model families | LightGBM and temporal CNN; later CatBoost/LightGBM/CNN blend | .98877 for the blend | Historical comparison; not the final standalone TCN |
| October 14: XGBoost topology investigation | Add graph-coupled continuous flow/storage allocation features | No public score recorded for this exact export | Strong local evidence for physics-derived inputs; Jonas later owns XGBoost |
| October 20: standalone TCN physics | Projected head, losses, loading bands and reservoir/tunnel context | .98599 | Starting standalone TCN submission |
| October 21: robust encoding and rivers | Asinh scaling, dense economic/physical context, duals, flow-deficit and river descriptors | .98895 for dense rivers | Use dense river reference |
| October 22: binary persistence | Add 54 transition-cost scenario channels | .98966 | Keep these channels in subsequent references |
| October 23: unit-count alternatives | Add 84 quantity/count or energy/count channels | .98875 / .98928 | Local gains failed to transfer publicly; later search returned to the earlier binary-DP reference |
| October 25: local/graph input branches | Learn nonlinear plant-specific representations; also test graph messages | .99176 local / .99159 graph | Preserve both; explore local hydraulic routing next |
| October 26: local hydraulic routing | Route existing adjacent reservoir/river/tunnel context into plant branches | .99197 | Retain local routing without neural graph messages |
| October 27: terminal conversion | Change conversion of local water values using main-waterway routes and best-point energy | .99240 | Retain corrected representation of terminal opportunity cost |
| October 27: allocation resolution | Change the LP feature generator from 12-hour to 6-hour intervals | **.99261** | Current frozen reference |
| October 27: further physical controls | Test changing-head allocation and legal curve-bound operating descriptors | .99254 for the curve-bound export | Not promoted; keep .99261 recipe |

## 1. Start by describing the decision, not just the raw inputs

The early tree inputs included prices, inflow, initial storage, terminal water
values, constraints and basic summaries. A single hour's price does not describe
whether starting a generator is worthwhile over a continuous operating period.

The decision-context extension added **23 features per generator matrix**:
14 economic descriptors and nine water descriptors. Economic inputs described
positive price-minus-water-value streaks, distance to favorable hours, remaining
opportunities and preceding/following 6/24-hour margins. Water inputs described
natural inflow relative to headroom, hypothetical excess without releases and
initial margins against minimum-volume requirements.

CatBoost development AUC increased from .95766422/.97857470 to
.96671334/.98163615 on 2020/2021; reported public AUC rose from .98417 to .98679.
These were useful decision proxies, not exact dispatch profits or future storage.

## 2. Represent how water can move through the system

Static topology descriptors alone did not consistently help. The more useful
idea was to derive **case-dependent features from the connected water system**.
A continuous linear-programming (LP) relaxation allocated flows and storage
using initial volumes, known weekly inputs, terminal values and named connections.
It supplied planned plant-flow fractions, projected reservoir storage and
constraint-pressure summaries. Initially it used 12-hour intervals.

In the controlled XGBoost comparison, topology descriptors alone scored
.96253735/.98235565 locally, whereas topology plus allocation scored
.98295212/.98791824. This motivated reusing deterministic allocation inputs in
the standalone TCN. The LP is a feature generator, not a separately trained
predictor or an ensemble member.

The first physical TCN added **153 hourly channels**: 81 hydraulic, 36 economic
and 36 reservoir/tunnel-network descriptors. Examples included projected gross
and net head, quadratic loss scenarios, flow relative to unit operating bands,
cost-relative favorable runs, reservoir margins and tunnel-opening/pressure
proxies. The resulting 560-hourly/127-static pipeline scored .98599 publicly.

## 3. Preserve economic information across changing price regimes

An audit found that the original standardization/clipping removed substantial
variation in unusually large 2021 prices and water values. We tested fitting-only
**asinh encoding**, which compresses extremes while retaining their sign, along
with relative price/value coordinates and a retained absolute-scale descriptor.

Further dense inputs included LP dual sensitivities (approximate marginal water
costs and scarcity), opportunity windows, residual minimum-flow/bypass scenarios,
and river geometry and relative penalty descriptors. River features added
**141 hourly and 36 static channels**. The dense-river pipeline reached
**859 hourly and 182 static channels** and public AUC .98895.

This bundle should not be described as proof that every addition helped.
Asinh alone explained much of the local gain; ratios alone were weaker.
Dense physics scored .98844 publicly, and compact physics .98667. Dense rivers
had only a tiny local mean advantage over dense physics and lost on local 2021.

## 4. Give the TCN information about starting and stopping

Continuous LP flows do not encode whether it is worthwhile to stay ON through
a short unfavorable interval to avoid stopping and restarting. We added
**54 binary dynamic-programming features**: nine plants, three relative
transition-cost scales and two initial-state assumptions.

Each channel measures the approximate whole-week payoff advantage of being ON
rather than OFF at an hour. Initial OFF and free initial state are alternatives
because actual initial commitment was not supplied. The model learns from these
scenarios; they are not observed UC states or guaranteed feasible schedules.

Inputs increased from 859 to **913 hourly channels**, with 182 static channels.
Development AUC improved from .97490724/.98410809 to .97638259/.98429510;
reported public AUC rose from .98895 to **.98966**.

We then added 84 ordinal active-unit-count channels, using either approximate
water-flow bands or energy rewards. Both improved local development scores but
gave lower public scores (.98875/.98928). Those extra count channels are
**absent from the final model**.

## 5. Improve where existing features enter the network

The shared TCN compressed hundreds of channels through a width-64 temporal
path. Small **width-16 plant branches** offered a nonlinear representation of
local inputs before combining them with the shared temporal path. A graph
alternative exchanged learned messages between connected plants.

Both used the existing 913/182 input schema. The graph version won both local
development years, but local branches scored higher publicly (.99176 versus
.99159). This disagreement did not establish a private winner. Fourteen
independent generator TCNs had also been tested and lost both local years to
the contemporary shared reference.

The next local-routing revision supplied each plant branch with existing
adjacent reservoir, tunnel, river and normalized reservoir-economic inputs.
These had already been visible to the global network. This changed **access to
information**, not the number of observations or unique input channels.
It improved both local years against the earlier local model and reached
**.99197** publicly. The final architecture retains this routing, with
**no neural graph-message passing or reservoir-ID embedding table**.

## 6. Improve the meaning of terminal water values

The previous allocation proxy converted local terminal EUR/MWh values using
maximum-load energy factors and broad reachable downstream routes. The revision
used a **33-point best-operating-point grid at initial head**, preferred
generating/main-waterway routes and receiving-reservoir values below the plant.

This changed values inside the existing physical/economic representation,
not the 913/182 channel count or the neural architecture. Development AUC rose
from .97883543/.98795214 to **.98065880/.98915965**; reported public AUC reached
**.99240**. It was a conversion-and-routing bundle, not an isolated causal test
of either component, and still approximated the supplied simulator.

## 7. Test the LP time resolution instead of assuming finer is better

With that terminal conversion fixed, we compared 12-, 6- and 1-hour LP intervals
using the same neural inputs, architecture, seed and inherited fitting budgets.

| LP interval | 2020 micro AUC | 2021 micro AUC |
|---|---:|---:|
| 12 hours | .98065880 | .98915965 |
| **6 hours** | **.98092093** | **.98925336** |
| 1 hour | .98042070 | .98917096 |

Six hours improved both years and reached **.99261** publicly. It gives 28 LP
intervals, while the TCN still receives and predicts **all 168 individual hours**.
One-hour allocation did not improve the result and required artificial storage
slack in some cases. Finer numerical resolution was not automatically better.

Later changing-head, extra curve and operating-range controls did not improve
both development years. The exported curve-bound candidate scored .99254
publicly, so the 6-hour reference remains the canonical recipe.

## What the final recipe retains

The [feature inventory](tcn_features.csv) lists all **913 hourly + 182 static
channels**. The final model retains raw/calendar/price context, initial storage
and terminal values, inflows and constraints, local economic/water descriptors,
allocation/projected physics, regime/asinh representation, environmental and
river descriptors, dual opportunities and 54 binary persistence scenarios.
It uses one shared TCN with jointly learned local hydraulic branches, revised
terminal conversion and a 6-hour LP feature generator. Count extensions,
history features, neural graph messages and the later curve extensions are absent.

The main progression was from raw state, to approximate operating opportunities,
to network-coupled water/economic context, to better local representation and
terminal-value conversion. Reported standalone TCN AUC increased from .98599
to .99261, a difference of **.00662**, with unsuccessful branches documented.
This describes the observed journey, not proof that each component caused the gain.

Local comparisons used purged chronological 2020/2021 cases, with preprocessing
learned only from fitting cases. Early rounds used inner stopping; the final
resolution comparison reused fixed 20/42 outer epochs and 22 final epochs.
Repeated development-year selection and prior 2022 exposure limit independence.
Public scores cannot establish private performance. Final submission fitting
uses all 2,922 labeled 2015-2022 cases; no future observed storage or held-out UC
labels enter feature construction. This summary adds no training or new results.
