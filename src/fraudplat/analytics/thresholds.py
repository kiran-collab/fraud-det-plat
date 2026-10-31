"""Threshold economics - turning a score distribution into a business decision.

The model produces a ranking. Where to cut it is not a modelling question, and
treating it as one is the most common way an otherwise good fraud programme
underperforms: a data scientist picks the threshold that maximises F1, nobody
can explain what F1 means to the business, and the number is never revisited.

The cut point is an economic choice with three competing costs:

    fraud allowed through      - loss, net of recovery
    good customers declined    - customer-experience cost
    alerts queued for review   - analyst time

This module sweeps every candidate threshold, prices each outcome using the
client's own cost assumptions, and reports the efficient frontier plus the
cost-optimal point. The deliverable is not "use 0.86" - it is a curve a Head
of Fraud can read, with the assumptions visible so they can argue with them.

**One naming note that matters.** The queue size here is
``alerts_per_10k_score_only`` and it will not match the ``alerts_per_10k`` KPI
in the metric layer - on the current holdout, 74 against 276. That is not a
defect: this module answers "what if we moved the score cut point", so it can
only reason about scores, whereas the KPI counts decisions as they were
actually taken, including the ~2.4% that deterministic rules escalated into
the queue from below the score threshold. Two different questions deserve two
different names; giving them the same name is how a steering meeting ends up
arguing about which number is broken.

The sensitivity analysis is the part that usually changes the conversation.
It answers "how wrong would our false-decline cost have to be before the
recommendation changes?", which is a far more useful question than "what is
the optimum?" - because the cost of a false decline is never known precisely,
and a recommendation that survives a 3x error in it can actually be adopted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fraudplat.analytics.metrics import DEFAULT_COSTS, CostModel


@dataclass
class OperatingPoint:
    decline_threshold: float
    review_threshold: float
    declines: int
    reviews: int
    caught_fraud: int
    missed_fraud: int
    false_declines: int
    detection_rate: float
    value_detection_rate: float
    precision: float
    false_decline_rate: float
    alerts_per_10k_score_only: float
    fraud_value_stopped: float
    fraud_value_lost: float
    gross_benefit: float
    false_decline_cost: float
    review_cost: float
    net_benefit: float


def evaluate_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    decline_at: float,
    review_at: float,
    costs: CostModel = DEFAULT_COSTS,
) -> OperatingPoint:
    """Price one candidate operating point."""
    declined = scores >= decline_at
    reviewed = (scores >= review_at) & ~declined

    caught = declined & (y == 1)
    missed = (~declined) & (y == 1)
    false_dec = declined & (y == 0)

    n_fraud = max(int((y == 1).sum()), 1)
    n_legit = max(int((y == 0).sum()), 1)
    fraud_value = max(float(amounts[y == 1].sum()), 1e-9)

    stopped = float(amounts[caught].sum())
    lost = float(amounts[missed].sum())

    # Only the recovered share is a real saving: some blocked fraud is simply
    # re-attempted successfully elsewhere.
    gross = stopped * costs.recovery_rate
    fd_cost = int(false_dec.sum()) * costs.false_decline_cost
    rv_cost = int(reviewed.sum()) * costs.review_cost

    return OperatingPoint(
        decline_threshold=round(decline_at, 4),
        review_threshold=round(review_at, 4),
        declines=int(declined.sum()),
        reviews=int(reviewed.sum()),
        caught_fraud=int(caught.sum()),
        missed_fraud=int(missed.sum()),
        false_declines=int(false_dec.sum()),
        detection_rate=int(caught.sum()) / n_fraud,
        value_detection_rate=stopped / fraud_value,
        precision=int(caught.sum()) / max(int(declined.sum()), 1),
        false_decline_rate=int(false_dec.sum()) / n_legit,
        alerts_per_10k_score_only=10_000 * (int(declined.sum()) + int(reviewed.sum())) / len(y),
        fraud_value_stopped=round(stopped, 2),
        fraud_value_lost=round(lost, 2),
        gross_benefit=round(gross, 2),
        false_decline_cost=round(fd_cost, 2),
        review_cost=round(rv_cost, 2),
        net_benefit=round(gross - fd_cost - rv_cost, 2),
    )


def sweep(
    y: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    review_at: float = 0.55,
    costs: CostModel = DEFAULT_COSTS,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Price every candidate decline threshold."""
    grid = np.arange(0.05, 0.991, 0.01) if grid is None else grid
    rows = [
        asdict(evaluate_threshold(y, scores, amounts, t, min(review_at, t), costs)) for t in grid
    ]
    return pd.DataFrame(rows)


def optimal_point(frontier: pd.DataFrame) -> pd.Series:
    return frontier.loc[frontier.net_benefit.idxmax()]


def constrained_optimum(
    frontier: pd.DataFrame,
    max_false_decline_rate: float | None = None,
    max_alerts_per_10k: float | None = None,
) -> pd.Series | None:
    """Best net benefit subject to operational limits.

    The unconstrained optimum is frequently unusable: it assumes an analyst
    team that does not exist, or a false-decline rate the retail bank will not
    accept whatever the arithmetic says. Constraints are how the recommendation
    becomes implementable rather than merely correct.
    """
    f = frontier
    if max_false_decline_rate is not None:
        f = f[f.false_decline_rate <= max_false_decline_rate]
    if max_alerts_per_10k is not None:
        f = f[f.alerts_per_10k_score_only <= max_alerts_per_10k]
    return None if f.empty else f.loc[f.net_benefit.idxmax()]


def sensitivity(
    y: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    costs: CostModel = DEFAULT_COSTS,
    multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> pd.DataFrame:
    """How far can each cost assumption be wrong before the answer moves?

    One assumption varied at a time, holding the others. If the recommended
    threshold is stable across a 16x range of false-decline cost, the
    recommendation is robust and can be adopted without first settling an
    unanswerable costing debate.
    """
    rows = []
    for field_name in ("false_decline_cost", "review_cost", "recovery_rate"):
        base = getattr(costs, field_name)
        for m in multipliers:
            value = min(base * m, 1.0) if field_name == "recovery_rate" else base * m
            variant = CostModel(**{**asdict(costs), field_name: value})
            best = optimal_point(sweep(y, scores, amounts, costs=variant))
            rows.append(
                {
                    "assumption": field_name,
                    "multiplier": m,
                    "value": round(value, 4),
                    "optimal_threshold": best.decline_threshold,
                    "net_benefit": best.net_benefit,
                    "detection_rate": round(best.detection_rate, 4),
                    "false_declines": int(best.false_declines),
                }
            )
    return pd.DataFrame(rows)


def format_recommendation(
    frontier: pd.DataFrame,
    current_threshold: float,
    constrained: pd.Series | None = None,
) -> str:
    best = optimal_point(frontier)
    cur = frontier.iloc[(frontier.decline_threshold - current_threshold).abs().argmin()]
    lines = [
        "== Threshold economics ==",
        f"  {'point':16}{'thresh':>8}{'detect':>9}{'value':>9}{'precis':>9}"
        f"{'false_dec':>11}{'alerts/10k*':>13}{'net benefit':>14}",
    ]

    def row(name: str, p) -> str:
        return (
            f"  {name:16}{p.decline_threshold:>8.2f}{p.detection_rate:>9.3f}"
            f"{p.value_detection_rate:>9.3f}{p.precision:>9.3f}"
            f"{int(p.false_declines):>11,}{p.alerts_per_10k_score_only:>13.1f}{p.net_benefit:>14,.0f}"
        )

    lines.append(row("current", cur))
    lines.append(row("unconstrained", best))
    if constrained is not None:
        lines.append(row("constrained", constrained))

    delta = best.net_benefit - cur.net_benefit
    lines.append(
        f"\n  moving {cur.decline_threshold:.2f} -> {best.decline_threshold:.2f} is worth "
        f"{delta:+,.0f} on this window "
        f"({int(best.caught_fraud - cur.caught_fraud):+d} fraud caught, "
        f"{int(best.false_declines - cur.false_declines):+d} false declines)"
    )
    return "\n".join(lines)
