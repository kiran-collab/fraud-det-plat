#!/usr/bin/env python
"""Build the analytics layer: marts, KPIs, data quality, threshold economics, BI extracts.

    python scripts/build_analytics.py

Runs in the order a refresh must run, because each step gates the next:

    1. replay decisions through the registered model   (fact source)
    2. data quality checks                             (GATE - blocks publication)
    3. dimensional marts in DuckDB                     (star schema)
    4. KPI computation from the metric layer           (single definitions)
    5. threshold economics                             (the recommendation)
    6. BI extracts for Spotfire / Power BI / Excel

Step 2 is a gate, not a report. A blocking quality failure stops the refresh
and leaves the previous extracts in place, because stale-but-correct beats
fresh-but-wrong when the output goes into a client pack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.analytics import exports, marts, quality, thresholds  # noqa: E402
from fraudplat.analytics.metrics import DEFAULT_COSTS, CostModel  # noqa: E402
from fraudplat.config import SETTINGS  # noqa: E402
from fraudplat.models.registry import ModelRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=SETTINGS.paths.data / "transactions.parquet")
    ap.add_argument("--out", type=Path, default=SETTINGS.paths.root / "artifacts" / "analytics")
    ap.add_argument("--version", default=None, help="model version (default: current)")
    ap.add_argument("--max-false-decline-rate", type=float, default=0.0010)
    ap.add_argument("--max-alerts-per-10k", type=float, default=120.0)
    ap.add_argument("--false-decline-cost", type=float, default=DEFAULT_COSTS.false_decline_cost)
    ap.add_argument("--review-cost", type=float, default=DEFAULT_COSTS.review_cost)
    ap.add_argument(
        "--scope",
        choices=["holdout", "all"],
        default="holdout",
        help="holdout = report only the window the model never trained on (default)",
    )
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"no transaction data at {args.data}; run scripts/train.py first")
        return 1

    costs = CostModel(false_decline_cost=args.false_decline_cost, review_cost=args.review_cost)
    bundle = ModelRegistry().load(args.version)
    df = pd.read_parquet(args.data)
    print(f"model {bundle.version} | {len(df):,} transactions")

    # --- 1. decision replay ------------------------------------------
    print("\n[1/6] replaying decisions")
    decisions = marts.build_decision_log(df, bundle)
    mix = decisions.action.value_counts(normalize=True)
    print("      action mix: " + "  ".join(f"{k}={v:.2%}" for k, v in mix.items()))

    if args.scope == "holdout":
        full = len(decisions)
        decisions = decisions[decisions.is_out_of_sample].reset_index(drop=True)
        print(
            f"      scope=holdout: reporting {len(decisions):,} of {full:,} decisions "
            f"(in-sample rows excluded - see marts.build_decision_log)"
        )
    else:
        print(
            "      scope=all: WARNING - includes the training window; detection KPIs "
            "will be optimistic and must not be published as performance"
        )

    # --- 2. data quality GATE ----------------------------------------
    print("\n[2/6] data quality")
    qrep = quality.run_quality_checks(
        decisions, reference_time=pd.to_datetime(decisions.event_time.max(), utc=True)
    )
    print(quality.format_quality_report(qrep))
    if not qrep.publishable:
        print("\nREFRESH ABORTED - blocking data-quality failure. Previous extracts retained.")
        return 2

    # --- 3. marts ----------------------------------------------------
    print("\n[3/6] building marts")
    alerts = marts.build_alert_queue(decisions)
    con, mrep = marts.build_marts(
        decisions, alerts, args.out / "fraud_analytics.duckdb", bundle.manifest
    )
    print(f"      {mrep.summary()}")

    # --- 4. KPIs -----------------------------------------------------
    print("\n[4/6] KPIs")
    overall = exports.kpi_overall(con)
    for _, r in overall.iterrows():
        v = r["value"]
        if r["unit"] == "rate":
            shown = f"{v:.3%}"
        elif r["unit"] == "currency":
            shown = f"{v:,.0f}"
        elif r["unit"] in ("count",):
            shown = f"{v:,.0f}"
        else:
            shown = f"{v:,.2f}"
        flag = " (needs settled labels)" if r["requires_settled_label"] else ""
        print(f"      {r['metric_label']:32}{shown:>14}{flag}")

    # --- 5. threshold economics --------------------------------------
    print("\n[5/6] threshold economics")
    y = decisions.is_fraud.to_numpy()
    scores = decisions.risk_score.to_numpy()
    amounts = decisions.amount.to_numpy()
    frontier = thresholds.sweep(
        y, scores, amounts, review_at=SETTINGS.decision.review_at, costs=costs
    )
    constrained = thresholds.constrained_optimum(
        frontier,
        max_false_decline_rate=args.max_false_decline_rate,
        max_alerts_per_10k=args.max_alerts_per_10k,
    )
    print(thresholds.format_recommendation(frontier, SETTINGS.decision.decline_at, constrained))

    sens = None
    if not args.skip_sensitivity:
        sens = thresholds.sensitivity(y, scores, amounts, costs=costs)
        stable = sens[sens.assumption == "false_decline_cost"].optimal_threshold.nunique()
        print(
            f"\n      sensitivity: optimal threshold takes {stable} distinct value(s) "
            f"across a 16x range of false-decline cost"
        )

    # --- 6. extracts -------------------------------------------------
    print("\n[6/6] BI extracts")
    written = exports.write_extracts(
        con,
        args.out / "extracts",
        frontier=frontier,
        sensitivity_df=sens,
        quality_df=qrep.to_frame(),
        costs=costs,
    )
    print(f"      {len(written)} extracts -> {args.out / 'extracts'}")

    summary = {
        "model_version": bundle.version,
        "window": list(mrep.window),
        "rows": mrep.rows_scored,
        "tables": mrep.tables,
        "quality_score": qrep.score(),
        "quality_publishable": qrep.publishable,
        "kpis": {
            r["metric_key"]: (None if pd.isna(r["value"]) else float(r["value"]))
            for _, r in overall.iterrows()
        },
        "current_threshold": SETTINGS.decision.decline_at,
        "recommended_threshold": float(thresholds.optimal_point(frontier).decline_threshold),
        "constrained_threshold": (
            None if constrained is None else float(constrained.decline_threshold)
        ),
        "cost_assumptions": {k: float(v) for k, v in costs.__dict__.items()},
    }
    (args.out / "analytics_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {args.out / 'analytics_summary.json'}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
