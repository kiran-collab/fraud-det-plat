"""Dimensional marts - the layer BI tools and analysts actually query.

The serving platform stores what it needs to make a decision: a model bundle, a
Redis key per card, an append-only audit log. None of that is queryable by an
analyst, and nobody should be pointing a dashboard at the authorization path.

So decisions are projected into a star schema: one fact table at the decision
grain, conformed dimensions around it. The reasons are the usual ones and they
matter more in a client setting than in a notebook:

* **A fact row is a decision, and that grain is stated.** Half of all
  disagreements about a fraud number are two people aggregating different
  grains - authorizations vs alerts vs cases - and not realising it.
* **Flags are materialised, not re-derived.** ``is_declined`` lives in the
  table, so a Spotfire user cannot accidentally define "declined" as
  ``risk_score > 0.8`` and quietly disagree with the KPI.
* **Conformed dimensions.** Merchant, card, date and action mean the same thing
  in every report, which is what lets a fraud-strategy view and an operations
  view be compared at all.

DuckDB is the engine. It runs in-process with no server, reads and writes
Parquet directly, and speaks standard SQL - so the same models that back a
local analysis can be lifted to Snowflake, Synapse or Athena by swapping the
connection. For a client engagement that means an analytics layer demonstrable
on day one rather than after a platform procurement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DIM_TABLES = ["dim_date", "dim_merchant", "dim_card", "dim_action", "dim_model_version"]
FACT_TABLES = ["fact_decision", "fact_alert"]


@dataclass
class MartBuildReport:
    rows_scored: int = 0
    tables: dict[str, int] | None = None
    window: tuple[str, str] = ("", "")

    def summary(self) -> str:
        t = ", ".join(f"{k}={v:,}" for k, v in (self.tables or {}).items())
        return f"{self.rows_scored:,} decisions | {self.window[0]} -> {self.window[1]} | {t}"


# --------------------------------------------------------------------------
# 1. Decision log
# --------------------------------------------------------------------------
def build_decision_log(
    df: pd.DataFrame,
    bundle,
    batch_features: pd.DataFrame | None = None,
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
) -> pd.DataFrame:
    """Replay transactions through the model to produce the decision fact source.

    Scored in bulk rather than one call at a time through the API. The vector
    maths is identical - same features, same models, same blend - but a
    120k-row replay takes seconds instead of minutes, which is the difference
    between an analytics refresh that runs nightly and one that does not run.
    """
    from fraudplat.config import SETTINGS
    from fraudplat.features.transforms import FEATURE_NAMES, FeatureEngine, compute_batch
    from fraudplat.serving.decisioning import decide

    df = df.sort_values("event_time", ignore_index=True)
    feats = (
        batch_features
        if batch_features is not None
        else compute_batch(df, engine=FeatureEngine(bundle.merchant_profile))
    )
    x = feats[FEATURE_NAMES].to_numpy(np.float32)

    p_sup = bundle.supervised.predict_proba(x)
    s_if = bundle.iforest.score(x)
    s_seq = bundle.sequence.score(x, df["card_id"].to_numpy())
    risk = bundle.ensemble.blend(p_sup, s_if, s_seq)
    calibrated = bundle.ensemble.calibrate(p_sup)

    cfg = SETTINGS.decision
    actions, rule_counts, rules = [], [], []
    records = df.to_dict("records")
    feat_records = feats.to_dict("records")
    for i, (txn, frow) in enumerate(zip(records, feat_records, strict=True)):
        d = decide(float(risk[i]), txn, frow, cfg)
        actions.append(d.action.value)
        rule_counts.append(len(d.triggered_rules))
        rules.append("|".join(d.triggered_rules))

    out = pd.DataFrame(
        {
            "transaction_id": df["transaction_id"].values,
            "event_time": df["event_time"].values,
            "card_id": df["card_id"].values,
            "merchant_id": df["merchant_id"].values,
            "merchant_category": df["merchant_category"].values,
            "merchant_country": df["merchant_country"].values,
            "channel": df["channel"].values,
            "entry_mode": df["entry_mode"].values,
            "customer_age_band": df["customer_age_band"].values,
            "amount": df["amount"].values.astype(float),
            "risk_score": risk,
            "supervised_score": calibrated,
            "anomaly_score": s_if,
            "sequence_score": s_seq,
            "action": actions,
            "triggered_rule_count": rule_counts,
            "triggered_rules": rules,
            "model_version": bundle.version,
            "is_fraud": df["is_fraud"].values,
        }
    )

    # Materialised flags - see module docstring.
    out["is_declined"] = out.action == "decline"
    out["is_review"] = out.action == "review"
    out["is_step_up"] = out.action == "step_up"
    out["is_approved"] = out.action == "approve"
    out["creates_friction"] = out.action != "approve"
    out["is_false_decline"] = out.is_declined & (out.is_fraud == 0)
    out["is_caught_fraud"] = out.is_declined & (out.is_fraud == 1)
    out["is_missed_fraud"] = (~out.is_declined) & (out.is_fraud == 1)
    out["fraud_value_stopped"] = np.where(out.is_caught_fraud, out.amount, 0.0)
    out["fraud_value_lost"] = np.where(out.is_missed_fraud, out.amount, 0.0)

    # Which window the MODEL was fitted on. Without this the analytics layer
    # happily reports detection rate over the training period, where any
    # boosted model is near-perfect - measured here at 98% precision in-sample
    # versus 86% on held-out data. Publishing that to a client as fraud
    # performance would be indefensible, and nothing in a BI tool would catch
    # it, so the distinction is carried in the data rather than left to
    # whoever writes the filter.
    n = len(out)
    a, b = int(n * train_frac), int(n * (train_frac + valid_frac))
    split = np.full(n, "test", dtype=object)
    split[:a] = "train"
    split[a:b] = "valid"
    out["data_split"] = split
    out["is_out_of_sample"] = out.data_split == "test"
    return out


# --------------------------------------------------------------------------
# 2. Alert queue
# --------------------------------------------------------------------------
def build_alert_queue(decisions: pd.DataFrame, seed: int = 17) -> pd.DataFrame:
    """Analyst case activity for the review queue.

    **This is simulated.** There is no case-management system wired in, so
    handling times, analyst assignment and dispositions are generated from a
    plausible distribution rather than observed. It is included because
    operations KPIs - throughput, SLA attainment, alert-to-case conversion -
    are what a fraud-operations stakeholder actually steers on, and leaving
    them out produces an analytics layer that only data scientists can use.

    Replace this function with a load from the real case system; everything
    downstream reads ``fact_alert`` and needs no change.
    """
    rng = np.random.default_rng(seed)
    q = decisions[decisions.action.isin(["review", "decline"])].copy().reset_index(drop=True)
    n = len(q)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "alert_id",
                "transaction_id",
                "queued_at",
                "analyst_id",
                "handling_minutes",
                "disposition",
                "sla_breach",
                "priority",
            ]
        )

    # Confirmed fraud is worked faster: it is more obvious and escalated sooner.
    base = rng.lognormal(mean=2.3, sigma=0.55, size=n)
    handling = np.where(q.is_fraud == 1, base * 0.75, base).clip(1.5, 180.0)

    # Dispositions agree with the label most of the time; analysts are not oracles.
    agree = rng.random(n) < 0.88
    disposition = np.where(
        agree,
        np.where(q.is_fraud == 1, "confirmed_fraud", "cleared"),
        np.where(q.is_fraud == 1, "cleared", "confirmed_fraud"),
    )
    # A slice is still open, as a live queue always is.
    pending = rng.random(n) < 0.06
    disposition = np.where(pending, "pending", disposition)

    return pd.DataFrame(
        {
            "alert_id": [f"alrt_{i:09d}" for i in range(n)],
            "transaction_id": q.transaction_id.values,
            "queued_at": q.event_time.values,
            "analyst_id": [f"anl_{i:03d}" for i in rng.integers(0, 24, n)],
            "handling_minutes": np.round(handling, 1),
            "disposition": disposition,
            # 30-minute operational SLA for a queued alert.
            "sla_breach": handling > 30.0,
            "priority": np.where(q.is_declined, "P1", np.where(q.amount >= 1000, "P2", "P3")),
        }
    )


# --------------------------------------------------------------------------
# 3. Star schema
# --------------------------------------------------------------------------
def build_marts(decisions: pd.DataFrame, alerts: pd.DataFrame, db_path: Path, manifest: dict):
    """Create the star schema in DuckDB and return an open connection."""
    import duckdb

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))

    con.register("src_decisions", decisions)
    con.register("src_alerts", alerts)

    # -- dimensions ----------------------------------------------------
    con.execute("""
        CREATE TABLE dim_date AS
        SELECT DISTINCT
            CAST(strftime(event_time, '%Y%m%d') AS INTEGER) AS date_key,
            CAST(event_time AS DATE)                        AS calendar_date,
            YEAR(event_time)                                AS year,
            MONTH(event_time)                               AS month,
            WEEKOFYEAR(event_time)                          AS iso_week,
            DAYOFWEEK(event_time)                           AS day_of_week,
            strftime(event_time, '%a')                      AS day_name,
            DAYOFWEEK(event_time) IN (0, 6)                 AS is_weekend
        FROM src_decisions ORDER BY date_key
    """)

    # Risk/volume bands are assigned once here so every report segments
    # merchants identically.
    con.execute("""
        CREATE TABLE dim_merchant AS
        WITH agg AS (
            SELECT merchant_id,
                   ANY_VALUE(merchant_category) AS merchant_category,
                   ANY_VALUE(merchant_country)  AS merchant_country,
                   COUNT(*)                     AS decision_volume,
                   AVG(risk_score)              AS mean_risk_score,
                   SUM(amount)                  AS attempted_spend
            FROM src_decisions GROUP BY merchant_id
        )
        SELECT *,
               CASE WHEN mean_risk_score >= 0.20 THEN 'high'
                    WHEN mean_risk_score >= 0.05 THEN 'medium'
                    ELSE 'low' END AS risk_tier,
               CASE WHEN decision_volume >= 500 THEN 'large'
                    WHEN decision_volume >= 100 THEN 'mid'
                    ELSE 'small' END AS volume_band
        FROM agg
    """)

    # customer_age_band is carried as RESTRICTED: it exists for fairness
    # monitoring and must never be used as a segmentation filter in a
    # business report, which is why the column name says so.
    con.execute("""
        CREATE TABLE dim_card AS
        SELECT card_id,
               ANY_VALUE(customer_age_band) AS restricted_age_band,
               COUNT(*)                     AS lifetime_decisions,
               MIN(event_time)              AS first_seen,
               MAX(event_time)              AS last_seen,
               CASE WHEN COUNT(*) >= 30 THEN 'established'
                    WHEN COUNT(*) >= 10 THEN 'developing'
                    ELSE 'new' END          AS tenure_band
        FROM src_decisions GROUP BY card_id
    """)

    con.execute("""
        CREATE TABLE dim_action AS
        SELECT * FROM (VALUES
            ('approve',  0, FALSE, FALSE, 'Transaction allowed'),
            ('step_up',  1, TRUE,  FALSE, 'Cardholder challenged (3-D Secure / OTP)'),
            ('review',   2, TRUE,  FALSE, 'Queued for human analyst'),
            ('decline',  3, TRUE,  TRUE,  'Transaction blocked - adverse action')
        ) AS t(action, action_rank, creates_friction, is_adverse_action, description)
    """)

    ens = (manifest.get("metrics", {}) or {}).get("ensemble", {}) or {}
    con.execute(
        """
        CREATE TABLE dim_model_version AS
        SELECT ? AS model_version, ? AS created_at, ? AS git_sha,
               ? AS train_rows, ? AS pr_auc, ? AS roc_auc
    """,
        [
            manifest.get("version", "unknown"),
            str(manifest.get("created_at", "")),
            manifest.get("git_sha", ""),
            manifest.get("rows_train", 0),
            ens.get("pr_auc"),
            ens.get("roc_auc"),
        ],
    )

    # -- facts ---------------------------------------------------------
    con.execute("""
        CREATE TABLE fact_decision AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY event_time) AS decision_sk,
            transaction_id, event_time,
            CAST(strftime(event_time, '%Y%m%d') AS INTEGER) AS date_key,
            HOUR(event_time) AS hour_of_day,
            card_id, merchant_id, model_version,
            channel, entry_mode,
            amount, risk_score, supervised_score, anomaly_score, sequence_score,
            action, triggered_rule_count, triggered_rules,
            data_split, is_out_of_sample,
            is_fraud, is_declined, is_review, is_step_up, is_approved,
            creates_friction, is_false_decline, is_caught_fraud, is_missed_fraud,
            fraud_value_stopped, fraud_value_lost
        FROM src_decisions
    """)

    con.execute("""
        CREATE TABLE fact_alert AS
        SELECT a.*, f.decision_sk, f.date_key, f.amount, f.risk_score, f.is_fraud
        FROM src_alerts a JOIN fact_decision f USING (transaction_id)
    """)

    for t in ("fact_decision", "fact_alert"):
        con.execute(f"CREATE INDEX idx_{t}_date ON {t}(date_key)")

    report = MartBuildReport(
        rows_scored=len(decisions),
        tables={
            t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in DIM_TABLES + FACT_TABLES
        },
        window=(str(decisions.event_time.min())[:10], str(decisions.event_time.max())[:10]),
    )
    log.info("marts built: %s", report.summary())
    return con, report
