"""BI extracts - the hand-off to Spotfire, Power BI or Excel.

Every extract is generated from the metric layer rather than from hand-written
SQL in a report, so a dashboard tile and a committee pack cannot disagree. The
KPI catalog ships alongside the numbers, which means a consumer can read the
definition of a measure next to the measure - the single cheapest fix for
"what does this column actually mean" questions arriving by email.

Shapes are chosen for the tool, not for elegance:

* **Long/tidy KPI series** (``metric_key``, ``value``) rather than one column
  per KPI. Adding a metric then requires no change to the data model, no
  column mapping, and no republished Spotfire template - the new metric simply
  appears in the measure list.
* **One wide denormalised detail table.** Spotfire is happiest with a single
  flat table for drill-down; asking business users to join a star schema in
  the tool is how filters end up applied to the wrong grain.
* **CSV and Parquet.** Parquet for anything programmatic, CSV because an
  analyst will want it in Excel within the hour and a tool that refuses that
  does not get adopted.

Note the grain boundary: the KPI catalog is defined at the *decision* grain.
Operations KPIs live at the *alert* grain and are therefore built by explicit
queries here rather than forced into the shared catalog - mixing grains in one
measure list is exactly the confusion the metric layer exists to prevent.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fraudplat.analytics.metrics import (
    DEFAULT_COSTS,
    METRICS,
    CostModel,
    catalog_rows,
    select_clause,
)

log = logging.getLogger(__name__)


def kpi_overall(con, keys: list[str] | None = None) -> pd.DataFrame:
    """Headline KPIs for the whole window, in long form."""
    keys = keys or list(METRICS)
    df = con.execute(f"SELECT {select_clause(keys)} FROM fact_decision").fetchdf()
    long = df.T.reset_index()
    long.columns = ["metric_key", "value"]
    meta = pd.DataFrame(catalog_rows())
    return long.merge(meta, on="metric_key", how="left")


def kpi_daily(con, keys: list[str] | None = None) -> pd.DataFrame:
    """Daily KPI time series - the trend surface for every dashboard."""
    keys = keys or list(METRICS)
    wide = con.execute(f"""
        SELECT d.calendar_date, d.day_name, d.is_weekend,
               {select_clause(keys)}
        FROM fact_decision f JOIN dim_date d USING (date_key)
        GROUP BY d.calendar_date, d.day_name, d.is_weekend
        ORDER BY d.calendar_date
    """).fetchdf()
    return wide.melt(
        id_vars=["calendar_date", "day_name", "is_weekend"],
        var_name="metric_key",
        value_name="value",
    )


def kpi_by_segment(con, segment: str, keys: list[str] | None = None) -> pd.DataFrame:
    """KPIs cut by a business dimension - where the actionable insight is.

    A portfolio-level decline rate tells a client nothing they can act on.
    'Decline rate is 4x higher on digital goods in two countries' is a policy
    change they can make on Monday.
    """
    allowed = {
        "merchant_category": "m.merchant_category",
        "merchant_country": "m.merchant_country",
        "risk_tier": "m.risk_tier",
        "volume_band": "m.volume_band",
        "channel": "f.channel",
        "entry_mode": "f.entry_mode",
        "tenure_band": "c.tenure_band",
        "action": "f.action",
    }
    if segment not in allowed:
        raise ValueError(f"segment must be one of {sorted(allowed)}")
    keys = keys or list(METRICS)
    return con.execute(f"""
        SELECT {allowed[segment]} AS segment_value,
               '{segment}' AS segment_name,
               {select_clause(keys)}
        FROM fact_decision f
        JOIN dim_merchant m USING (merchant_id)
        JOIN dim_card c USING (card_id)
        GROUP BY 1 ORDER BY authorizations DESC
    """).fetchdf()


def decision_detail(con, limit: int | None = None) -> pd.DataFrame:
    """Wide denormalised detail table for drill-down.

    ``restricted_age_band`` is deliberately excluded. It exists in dim_card for
    fairness monitoring and must not reach a tool where a business user could
    segment a performance report by it.
    """
    lim = f"LIMIT {limit}" if limit else ""
    return con.execute(f"""
        SELECT f.decision_sk, f.transaction_id, f.event_time, d.calendar_date,
               d.iso_week, d.day_name, f.hour_of_day,
               f.card_id, c.tenure_band, c.lifetime_decisions,
               f.merchant_id, m.merchant_category, m.merchant_country,
               m.risk_tier AS merchant_risk_tier, m.volume_band AS merchant_volume_band,
               f.channel, f.entry_mode, f.amount,
               f.risk_score, f.supervised_score, f.anomaly_score, f.sequence_score,
               f.action, a.action_rank, a.creates_friction, a.is_adverse_action,
               f.triggered_rule_count, f.triggered_rules,
               f.is_fraud, f.is_declined, f.is_false_decline, f.is_caught_fraud,
               f.is_missed_fraud, f.fraud_value_stopped, f.fraud_value_lost,
               f.model_version
        FROM fact_decision f
        JOIN dim_date d USING (date_key)
        JOIN dim_merchant m USING (merchant_id)
        JOIN dim_card c USING (card_id)
        JOIN dim_action a USING (action)
        ORDER BY f.event_time {lim}
    """).fetchdf()


def operations_summary(con, costs: CostModel = DEFAULT_COSTS) -> pd.DataFrame:
    """Alert-grain operations KPIs - what a fraud-ops manager steers on."""
    return con.execute(f"""
        SELECT
            COUNT(*)                                                     AS alerts,
            COUNT(DISTINCT analyst_id)                                   AS analysts_active,
            AVG(handling_minutes)                                        AS avg_handling_minutes,
            MEDIAN(handling_minutes)                                     AS median_handling_minutes,
            QUANTILE_CONT(handling_minutes, 0.9)                         AS p90_handling_minutes,
            SUM(handling_minutes) / 60.0                                 AS total_analyst_hours,
            AVG(CASE WHEN sla_breach THEN 1.0 ELSE 0.0 END)              AS sla_breach_rate,
            AVG(CASE WHEN disposition = 'pending' THEN 1.0 ELSE 0.0 END) AS open_rate,
            SUM(CASE WHEN disposition = 'confirmed_fraud' THEN 1.0 ELSE 0.0 END)
                / NULLIF(COUNT(*), 0)                                    AS case_conversion_rate,
            AVG(CASE WHEN disposition <> 'pending'
                     AND ((disposition = 'confirmed_fraud') = (is_fraud = 1))
                     THEN 1.0 ELSE 0.0 END)                              AS analyst_agreement_rate,
            COUNT(*) * {costs.review_cost}                               AS review_cost_total,
            SUM(CASE WHEN disposition = 'confirmed_fraud' THEN amount ELSE 0 END)
                                                                         AS value_confirmed_fraud
        FROM fact_alert
    """).fetchdf()


def analyst_productivity(con) -> pd.DataFrame:
    return con.execute("""
        SELECT analyst_id,
               COUNT(*)                                        AS alerts_worked,
               AVG(handling_minutes)                           AS avg_handling_minutes,
               AVG(CASE WHEN sla_breach THEN 1.0 ELSE 0.0 END) AS sla_breach_rate,
               SUM(CASE WHEN disposition = 'confirmed_fraud' THEN 1 ELSE 0 END) AS fraud_confirmed
        FROM fact_alert GROUP BY analyst_id ORDER BY alerts_worked DESC
    """).fetchdf()


def write_extracts(
    con,
    out_dir: Path,
    frontier: pd.DataFrame | None = None,
    sensitivity_df: pd.DataFrame | None = None,
    quality_df: pd.DataFrame | None = None,
    costs: CostModel = DEFAULT_COSTS,
) -> dict[str, Path]:
    """Write every extract as CSV + Parquet. Returns {name: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    extracts = {
        "kpi_catalog": pd.DataFrame(catalog_rows()),
        "kpi_overall": kpi_overall(con),
        "kpi_daily": kpi_daily(con),
        "decision_detail": decision_detail(con),
        "operations_summary": operations_summary(con, costs),
        "analyst_productivity": analyst_productivity(con),
        "cost_assumptions": pd.DataFrame([asdict(costs)]),
    }
    for seg in ("merchant_category", "merchant_country", "channel", "entry_mode", "tenure_band"):
        extracts[f"segment_{seg}"] = kpi_by_segment(con, seg)
    if frontier is not None:
        extracts["threshold_frontier"] = frontier
    if sensitivity_df is not None:
        extracts["threshold_sensitivity"] = sensitivity_df
    if quality_df is not None:
        extracts["data_quality"] = quality_df

    for name, df in extracts.items():
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        written[name] = p
        log.info("extract %-24s rows=%-8d -> %s", name, len(df), p.name)
    return written
