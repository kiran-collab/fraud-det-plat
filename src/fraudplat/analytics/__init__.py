"""Analytics framework: metric layer, dimensional marts, data quality,
threshold economics and BI extracts."""

from fraudplat.analytics.metrics import METRICS, CostModel, Domain, Metric, Unit, metric
from fraudplat.analytics.quality import QualityReport, run_quality_checks
from fraudplat.analytics.thresholds import evaluate_threshold, optimal_point, sweep

__all__ = [
    "METRICS",
    "Metric",
    "Unit",
    "Domain",
    "CostModel",
    "metric",
    "run_quality_checks",
    "QualityReport",
    "sweep",
    "optimal_point",
    "evaluate_threshold",
]
