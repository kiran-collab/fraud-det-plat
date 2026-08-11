"""Offline evaluation.

Fraud is a ~0.85%-prevalence problem, so the headline metric is PR-AUC, not
ROC-AUC. Everything else here answers a question a risk manager actually asks:

  * how much fraud value do we stop at the false-positive rate operations can
    staff for?
  * what does the review queue cost per fraud caught?
  * what net loss reduction does this buy versus the incumbent rules?
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class OperatingPoint:
    threshold: float
    precision: float
    recall: float
    false_positive_rate: float
    alerts_per_10k: float
    value_recall: float  # share of *fraud dollars* caught, not just counts


@dataclass
class EvaluationReport:
    n: int
    n_fraud: int
    prevalence: float
    pr_auc: float
    roc_auc: float
    recall_at_1pct_fpr: float
    recall_at_0p5pct_fpr: float
    operating_points: list[OperatingPoint]
    value_detection_rate: float
    estimated_loss_reduction: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["operating_points"] = [asdict(op) for op in self.operating_points]
        return d


def _recall_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


def evaluate(
    y: np.ndarray,
    score: np.ndarray,
    amount: np.ndarray | None = None,
    thresholds: tuple[float, ...] = (0.35, 0.55, 0.70, 0.86),
    recovery_rate: float = 0.85,
    review_cost: float = 4.50,
    false_decline_cost: float = 12.0,
) -> EvaluationReport:
    """Score a model on a held-out window.

    ``recovery_rate`` is the share of a blocked fraudulent authorization that is
    actually avoided (some fraud is re-attempted successfully elsewhere).
    ``review_cost`` is fully-loaded analyst cost per queued alert;
    ``false_decline_cost`` prices the customer-experience damage of blocking a
    good transaction. All three are business inputs, not model outputs - they
    live here so the loss estimate is explicit and arguable rather than buried.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    amount = np.ones_like(score) if amount is None else np.asarray(amount, dtype=float)

    fraud_value = float(amount[y == 1].sum()) or 1.0
    points: list[OperatingPoint] = []
    for t in thresholds:
        flagged = score >= t
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        tn = int((~flagged & (y == 0)).sum())
        points.append(OperatingPoint(
            threshold=t,
            precision=tp / max(tp + fp, 1),
            recall=tp / max(tp + fn, 1),
            false_positive_rate=fp / max(fp + tn, 1),
            alerts_per_10k=10_000 * (tp + fp) / max(len(y), 1),
            value_recall=float(amount[flagged & (y == 1)].sum()) / fraud_value,
        ))

    # Net benefit at the decline threshold: value of fraud stopped, less the
    # cost of the alerts we generate and the good customers we inconvenience.
    decline_t = max(thresholds)
    declined = score >= decline_t
    stopped_value = float(amount[declined & (y == 1)].sum()) * recovery_rate
    false_declines = int((declined & (y == 0)).sum())
    review_band = (score >= min(thresholds)) & (score < decline_t)
    net = stopped_value - false_declines * false_decline_cost - int(review_band.sum()) * review_cost

    return EvaluationReport(
        n=len(y),
        n_fraud=int(y.sum()),
        prevalence=float(y.mean()),
        pr_auc=float(average_precision_score(y, score)),
        roc_auc=float(roc_auc_score(y, score)),
        recall_at_1pct_fpr=_recall_at_fpr(y, score, 0.01),
        recall_at_0p5pct_fpr=_recall_at_fpr(y, score, 0.005),
        operating_points=points,
        value_detection_rate=float(amount[declined & (y == 1)].sum()) / fraud_value,
        estimated_loss_reduction=net,
    )


def format_report(report: EvaluationReport, title: str = "Evaluation") -> str:
    lines = [
        f"== {title} ==",
        f"  rows={report.n:,}  fraud={report.n_fraud:,}  prevalence={report.prevalence:.4%}",
        f"  PR-AUC={report.pr_auc:.4f}   ROC-AUC={report.roc_auc:.4f}",
        f"  recall @ 1.0% FPR = {report.recall_at_1pct_fpr:.3f}",
        f"  recall @ 0.5% FPR = {report.recall_at_0p5pct_fpr:.3f}",
        f"  fraud value stopped at decline threshold = {report.value_detection_rate:.1%}",
        f"  estimated net loss reduction = ${report.estimated_loss_reduction:,.0f}",
        "  threshold   precision   recall   FPR      alerts/10k   value-recall",
    ]
    for op in report.operating_points:
        lines.append(
            f"    {op.threshold:<10.2f}{op.precision:<12.3f}{op.recall:<9.3f}"
            f"{op.false_positive_rate:<9.4f}{op.alerts_per_10k:<13.1f}{op.value_recall:.3f}"
        )
    return "\n".join(lines)
