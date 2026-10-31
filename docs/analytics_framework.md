# Fraud Analytics Framework

An analytics layer over the fraud detection platform: a governed metric layer,
dimensional marts, data-quality gates, threshold economics, and BI extracts
sized for Spotfire / Power BI / Excel.

```bash
python scripts/build_analytics.py          # holdout window (default)
python scripts/build_analytics.py --scope all --false-decline-cost 25
```

---

## Why a separate layer

The platform was built to **make decisions**: a model bundle, a Redis key per
card, an append-only audit log, a 2.5 ms scoring path. None of that is
queryable, and nobody should point a dashboard at an authorization path.

Everything an analytics consumer needs was therefore missing: no marts, no
agreed KPI definitions, no quality gates, no BI hand-off, and no way to answer
*"should we move the threshold?"* with anything but a model metric. This layer
is that gap closed.

```
registered model ──► decision replay ──► data-quality GATE
                                               │ (blocks on failure)
                                               ▼
                                     DuckDB star schema
                                               │
             ┌─────────────────┬───────────────┼────────────────┐
             ▼                 ▼               ▼                ▼
        metric layer     threshold         operations       BI extracts
        (19 KPIs)        economics          KPIs            (CSV + Parquet)
```

---

## 1. Metric layer — one definition per KPI

`analytics/metrics.py`

The failure mode this exists to prevent has nothing to do with modelling: four
people compute "false positive rate" four ways and bring four numbers to the
same steering meeting. One divides false declines by all authorizations, one by
legitimate authorizations only, one by alerts raised, one excludes step-ups.
Every number is defensible alone, and the meeting becomes about arithmetic.

A KPI is therefore defined **once, as data** — name, plain-language definition,
SQL, unit, owner, direction, target. The marts, extracts, reports and tests all
read these definitions. Adding a KPI is a dictionary entry.

**19 KPIs across 6 domains:** volume (3), decisioning (4), detection (5),
financial (1), operations (2), model health (4).

Two conventions, because these are the ones usually fudged:

- **Rates name their denominator.** `decline_rate` is over all authorizations;
  `false_decline_rate` is over *legitimate* authorizations only. Enforced by a
  test that reads the definition text.
- **Label-dependent KPIs are flagged.** Chargebacks settle 30–90 days after the
  authorization, so recent periods have thin labels. Any KPI needing `is_fraud`
  carries `requires_label=True` and must be reported against a matured window —
  otherwise missing data reads as a sudden improvement in fraud performance.

The **KPI catalog** ships as an extract, so a Spotfire user reads the
definition next to the number instead of emailing to ask.

## 2. Dimensional marts

`analytics/marts.py` — DuckDB, in-process, no server, reads/writes Parquet.
Standard SQL, so the same models lift to Snowflake, Synapse or Athena by
swapping the connection. For a client engagement that means an analytics layer
demonstrable on day one rather than after a procurement cycle.

| Table | Grain | Rows (holdout) |
|---|---|---|
`fact_decision` | one authorization decision | 18,000 |
`fact_alert` | one queued alert | 497 |
`dim_date` | calendar day | 62 |
`dim_merchant` | merchant + risk tier + volume band | 1,500 |
`dim_card` | card + tenure band | 6,000 |
`dim_action` | approve / step_up / review / decline | 4 |
`dim_model_version` | model lineage | 1 |

Design choices that earn their keep:

- **The grain is stated.** Half of all disagreements about a fraud number are
  two people aggregating different grains — authorizations vs alerts vs cases —
  without realising.
- **Flags are materialised.** `is_declined` is a column, so a BI user cannot
  define "declined" as `risk_score > 0.8` and quietly disagree with the KPI.
- **Conformed dimensions.** Merchant, card, date and action mean the same thing
  in every report, which is what lets a strategy view and an operations view be
  compared at all.
- **`restricted_age_band`** is named to warn. It exists in `dim_card` for
  fairness monitoring and is deliberately excluded from the detail extract so
  nobody segments a performance report by it.

## 3. Data quality — a gate, not a report

`analytics/quality.py` — 17 checks across the six dimensions a governance
function names: **completeness, validity, uniqueness, timeliness, consistency,
accuracy.**

Fraud reporting is unusually exposed: the numbers are small, so a 2%
duplication rate or one day of missing feed moves a headline KPI further than
any genuine change in behaviour — and in a direction that looks like a story
somebody then explains in a meeting.

The operational distinction:

| | Behaviour | Example |
|---|---|---|
**FAIL + blocking** | refresh aborts, previous extracts retained | duplicate grain, flag/action mismatch, out-of-domain action |
**FAIL, non-blocking** | published with a caveat | stale feed, thin labels |
**WARN** | annotated | metric drifting toward its limit |

Stale-but-correct beats fresh-but-wrong when the output goes into a client pack.
Equally, a warning that silently blocks a refresh trains people to ignore
warnings.

## 4. Threshold economics — the actual recommendation

`analytics/thresholds.py`

Where to cut the score is **not** a modelling question. Treating it as one is
the most common reason an otherwise good fraud programme underperforms: someone
maximises F1, nobody can explain F1 to the business, and the number is never
revisited.

Three competing costs, priced from the client's own assumptions:

```
fraud allowed through     → loss, net of recovery
good customers declined   → customer-experience cost
alerts queued for review  → analyst time
```

Current output on the holdout window:

```
  point             thresh   detect    value   precis  false_dec  alerts/10k*  net benefit
  current             0.86    0.250    0.681    0.844          7         74.4       80,893
  unconstrained       0.11    0.836    0.984    0.289        313        244.4      113,861
  constrained         0.76    0.309    0.729    0.758         15         74.4       86,607
```

The **constrained** row is the deliverable. The unconstrained optimum assumes an
analyst team that does not exist and a false-decline rate retail banking will
not accept whatever the arithmetic says. Constraints are what make a
recommendation implementable rather than merely correct.

**Sensitivity analysis is what changes the conversation.** It answers *"how
wrong would our false-decline cost have to be before the answer moves?"* — far
more useful than the optimum itself, because that cost is never known precisely
and a recommendation surviving a 16× error can be adopted without first settling
an unanswerable costing debate.

> `alerts/10k*` is `alerts_per_10k_score_only` and will **not** match the
> `alerts_per_10k` KPI — 74 against 276 here. Not a defect: this module asks
> "what if we moved the score cut", so it can only reason about scores, while
> the KPI counts decisions as actually taken, including the ~2.4% that
> deterministic rules escalated from below the threshold. Two questions, two
> names. Giving them one name is how a meeting ends up arguing about which
> number is broken.

## 5. BI extracts

`analytics/exports.py` → `artifacts/analytics/extracts/`, 15 extracts as both
CSV and Parquet.

| Extract | Use |
|---|---|
`kpi_catalog` | definitions published alongside the numbers |
`kpi_overall`, `kpi_daily` | headline tiles and trend surface |
`segment_*` (5) | merchant category, country, channel, entry mode, card tenure |
`decision_detail` | wide flat table for drill-down |
`operations_summary`, `analyst_productivity` | queue and analyst views |
`threshold_frontier`, `threshold_sensitivity` | the recommendation and its robustness |
`data_quality`, `cost_assumptions` | the governance trail |

Shapes are chosen for the tool, not for elegance:

- **Long/tidy KPI series** (`metric_key`, `value`) not one column per KPI — a
  new metric then needs no data-model change, no column mapping and no
  republished template.
- **One wide denormalised detail table** — Spotfire is happiest with a flat
  table for drill-down; asking business users to join a star schema in the tool
  is how filters get applied to the wrong grain.
- **CSV *and* Parquet** — Parquet for anything programmatic, CSV because an
  analyst wants it in Excel within the hour, and a tool that refuses that does
  not get adopted.

### Suggested Spotfire pages

1. **Executive** — decline rate, detection rate, value detection rate, fraud
   loss bps, net benefit; trend + MoM delta; data-quality badge.
2. **Strategy** — segment heatmap (category × country) on fraud prevalence and
   false-decline rate; threshold frontier with the current point marked.
3. **Operations** — queue volume, handling-time distribution, SLA attainment,
   analyst throughput, case-conversion rate.
4. **Model health** — mean risk score, high-risk rate, cold-start rate, label
   coverage; the label-coverage tile gates the detection page.
5. **Case drill-down** — `decision_detail` filtered to alerts, with reason codes.

---

## Example of the insight this surfaces

Portfolio-level numbers tell a client nothing actionable. The segment cut does:

| merchant category | authorizations | fraud prevalence (bps) | detection rate | value detection |
|---|---|---|---|---|
| digital_goods | 1,912 | **125.5** | 0.208 | **0.063** |
| pharmacy | 1,828 | 103.9 | 0.158 | 0.688 |
| travel | 1,464 | 102.5 | 0.200 | 0.363 |
| money_transfer | 1,804 | 94.2 | 0.353 | 0.654 |
| electronics | 2,499 | 80.0 | 0.400 | 0.862 |

Digital goods has the **highest fraud prevalence and the worst value detection
in the portfolio** — 6% of fraud value stopped against 86% in electronics. That
is a specific, defensible policy recommendation (a category-specific threshold
or a velocity rule on low-value card-not-present) rather than "improve the
model".

---

## Known limitations

Stated plainly, because an analytics framework that oversells itself is worse
than one with a short scope.

- **`fact_alert` is simulated.** No case-management system is wired in, so
  handling times, analyst assignment and dispositions come from a plausible
  distribution. Included because operations KPIs are what a fraud-ops
  stakeholder steers on, and omitting them produces a layer only data
  scientists can use. Replace `build_alert_queue()` with a load from the real
  system; everything downstream is unchanged.
- **Underlying data is synthetic.** Every number describes the generator, not a
  real portfolio.
- **Default scope is the holdout window.** `--scope all` includes the training
  period, where detection reads 98% precision against 84% out-of-sample. It
  exists for volume analysis and prints a warning; those figures must never be
  published as performance.
- **No incremental refresh.** The mart rebuilds in full. Fine at 120k rows,
  wrong at production volume, where this becomes a partitioned incremental load.
- **No row-level security.** A real deployment scopes `dim_card` and
  `decision_detail` by entitlement; here every consumer sees everything.
