"""Data quality controls for the analytics layer.

An analytics framework without data-quality gates is a confident way to be
wrong. Fraud reporting is worse than most: the numbers are small, so a
2% duplication rate or a day of missing feed moves a headline KPI more than
any genuine change in fraud behaviour, and it moves it in a direction that
looks like a story. Somebody then explains the story in a steering meeting.

Checks are grouped by the six dimensions a data-governance function will ask
for by name, so the output maps onto an existing control framework rather than
being a bespoke list:

    completeness  - is anything missing that should be there?
    validity      - are values inside their allowed domain?
    uniqueness    - is the grain actually unique?
    timeliness    - is the data recent enough to report on?
    consistency   - do derived fields agree with their source?
    accuracy      - are outcomes settled enough to be trusted?

Each check returns a measured value against a threshold, and a severity. The
distinction that matters operationally: **FAIL blocks publication, WARN
annotates it.** A warning that silently blocks a refresh trains people to
ignore warnings; a failure that merely annotates lets a broken number reach a
client deck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd


class Severity(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class Dimension(StrEnum):
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"


@dataclass
class CheckResult:
    check: str
    dimension: Dimension
    measured: float
    threshold: float
    severity: Severity
    detail: str
    blocks_publication: bool = False


@dataclass
class QualityReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity is Severity.FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity is Severity.WARN]

    @property
    def publishable(self) -> bool:
        """Only a blocking failure stops publication."""
        return not any(r.blocks_publication and r.severity is Severity.FAIL for r in self.results)

    def score(self) -> float:
        """Share of checks passing - the single number for a scorecard tile."""
        if not self.results:
            return 0.0
        return sum(r.severity is Severity.PASS for r in self.results) / len(self.results)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check": r.check,
                    "dimension": r.dimension.value,
                    "measured": round(r.measured, 6),
                    "threshold": r.threshold,
                    "severity": r.severity.value,
                    "blocks_publication": r.blocks_publication,
                    "detail": r.detail,
                }
                for r in self.results
            ]
        )


# Columns that must never be null for a decision to be reportable.
REQUIRED_COLUMNS = [
    "transaction_id",
    "event_time",
    "card_id",
    "merchant_id",
    "amount",
    "risk_score",
    "action",
]
VALID_ACTIONS = {"approve", "step_up", "review", "decline"}


def _judge(
    measured: float, threshold: float, higher_is_better: bool, warn_band: float = 0.5
) -> Severity:
    """PASS at threshold, WARN within a band of it, FAIL beyond.

    The warn band exists so a metric drifting toward its limit is visible
    before it breaches - a binary pass/fail only tells you once it is too late.
    """
    if higher_is_better:
        if measured >= threshold:
            return Severity.PASS
        return (
            Severity.WARN if measured >= threshold - abs(threshold) * warn_band else Severity.FAIL
        )
    if measured <= threshold:
        return Severity.PASS
    return (
        Severity.WARN
        if measured <= threshold + abs(threshold or 1e-6) * warn_band
        else Severity.FAIL
    )


def run_quality_checks(
    decisions: pd.DataFrame,
    label_coverage_target: float = 0.95,
    max_staleness_days: float = 2.0,
    reference_time: pd.Timestamp | None = None,
) -> QualityReport:
    rep = QualityReport()
    n = len(decisions)
    if n == 0:
        rep.results.append(
            CheckResult(
                "dataset_not_empty",
                Dimension.COMPLETENESS,
                0,
                1,
                Severity.FAIL,
                "no decisions to report on",
                blocks_publication=True,
            )
        )
        return rep

    # --- completeness -------------------------------------------------
    for col in REQUIRED_COLUMNS:
        if col not in decisions.columns:
            rep.results.append(
                CheckResult(
                    f"column_present[{col}]",
                    Dimension.COMPLETENESS,
                    0,
                    1,
                    Severity.FAIL,
                    f"required column '{col}' is absent",
                    blocks_publication=True,
                )
            )
            continue
        filled = float(decisions[col].notna().mean())
        rep.results.append(
            CheckResult(
                f"not_null[{col}]",
                Dimension.COMPLETENESS,
                filled,
                1.0,
                _judge(filled, 1.0, True, warn_band=0.001),
                f"{(1 - filled) * n:,.0f} null of {n:,}",
                blocks_publication=True,
            )
        )

    # --- validity -----------------------------------------------------
    bad_action = float((~decisions.action.isin(VALID_ACTIONS)).mean())
    rep.results.append(
        CheckResult(
            "action_in_domain",
            Dimension.VALIDITY,
            bad_action,
            0.0,
            _judge(bad_action, 0.0, False),
            f"allowed: {sorted(VALID_ACTIONS)}",
            blocks_publication=True,
        )
    )

    bad_score = float(((decisions.risk_score < 0) | (decisions.risk_score > 1)).mean())
    rep.results.append(
        CheckResult(
            "risk_score_in_0_1",
            Dimension.VALIDITY,
            bad_score,
            0.0,
            _judge(bad_score, 0.0, False),
            "risk_score must be a probability",
            blocks_publication=True,
        )
    )

    nonpos = float((decisions.amount <= 0).mean())
    rep.results.append(
        CheckResult(
            "amount_positive",
            Dimension.VALIDITY,
            nonpos,
            0.0,
            _judge(nonpos, 0.0, False),
            "non-positive amounts indicate a feed defect",
        )
    )

    # Extreme values are not invalid, but a spike in them usually means a
    # currency or scaling error upstream rather than genuine big-ticket spend.
    extreme = float((decisions.amount > 25_000).mean())
    rep.results.append(
        CheckResult(
            "amount_within_expected_range",
            Dimension.VALIDITY,
            extreme,
            0.001,
            _judge(extreme, 0.001, False),
            "share above the 25,000 scheme ceiling",
        )
    )

    # --- uniqueness ---------------------------------------------------
    dup = float(decisions.transaction_id.duplicated().mean())
    rep.results.append(
        CheckResult(
            "transaction_id_unique",
            Dimension.UNIQUENESS,
            dup,
            0.0,
            _judge(dup, 0.0, False),
            "duplicates double-count fraud and inflate every rate",
            blocks_publication=True,
        )
    )

    # --- timeliness ---------------------------------------------------
    latest = pd.to_datetime(decisions.event_time.max(), utc=True)
    ref = reference_time or pd.Timestamp.now(tz="UTC")
    staleness = max(0.0, (ref - latest).total_seconds() / 86_400)
    rep.results.append(
        CheckResult(
            "data_freshness_days",
            Dimension.TIMELINESS,
            staleness,
            max_staleness_days,
            _judge(staleness, max_staleness_days, False),
            f"most recent decision {latest:%Y-%m-%d}",
        )
    )

    # A gap in the daily series is a missing feed, which reads as a quiet
    # fraud day rather than as an outage.
    daily = (
        decisions.set_index(pd.to_datetime(decisions.event_time, utc=True)).resample("1D").size()
    )
    gaps = int((daily == 0).sum())
    rep.results.append(
        CheckResult(
            "no_missing_days",
            Dimension.TIMELINESS,
            float(gaps),
            0.0,
            _judge(float(gaps), 0.0, False),
            f"{gaps} day(s) with zero decisions in the window",
        )
    )

    # --- consistency --------------------------------------------------
    if "is_declined" in decisions.columns:
        mismatch = float((decisions.is_declined != (decisions.action == "decline")).mean())
        rep.results.append(
            CheckResult(
                "flag_matches_action",
                Dimension.CONSISTENCY,
                mismatch,
                0.0,
                _judge(mismatch, 0.0, False),
                "materialised is_declined must agree with action",
                blocks_publication=True,
            )
        )

    # Rules may only escalate. A decision below the step-up threshold with no
    # rule fired would mean the decision layer has drifted from its contract.
    if "triggered_rule_count" in decisions.columns:
        from fraudplat.config import SETTINGS

        impossible = float(
            (
                (decisions.action != "approve")
                & (decisions.risk_score < SETTINGS.decision.step_up_at)
                & (decisions.triggered_rule_count == 0)
            ).mean()
        )
        rep.results.append(
            CheckResult(
                "friction_is_explained",
                Dimension.CONSISTENCY,
                impossible,
                0.0,
                _judge(impossible, 0.0, False),
                "friction below threshold must be attributable to a rule",
            )
        )

    # --- accuracy -----------------------------------------------------
    coverage = float(decisions.is_fraud.notna().mean()) if "is_fraud" in decisions.columns else 0.0
    rep.results.append(
        CheckResult(
            "label_coverage",
            Dimension.ACCURACY,
            coverage,
            label_coverage_target,
            _judge(coverage, label_coverage_target, True),
            "below target, detection KPIs are not yet reportable for this window",
        )
    )

    return rep


def format_quality_report(rep: QualityReport) -> str:
    icon = {Severity.PASS: "PASS", Severity.WARN: "WARN", Severity.FAIL: "FAIL"}
    lines = [
        f"== Data quality: {rep.score():.0%} of {len(rep.results)} checks passing "
        f"({'PUBLISHABLE' if rep.publishable else 'BLOCKED'}) ==",
        f"  {'check':38}{'dim':14}{'measured':>12}{'limit':>12}  result",
    ]
    for r in sorted(rep.results, key=lambda r: (r.severity is Severity.PASS, r.dimension.value)):
        lines.append(
            f"  {r.check:38}{r.dimension.value:14}{r.measured:>12.6f}{r.threshold:>12.6f}  "
            f"{icon[r.severity]}{'  (blocking)' if r.blocks_publication and r.severity is Severity.FAIL else ''}"
        )
    if rep.failures:
        lines.append("  failures:")
        lines += [f"    - {r.check}: {r.detail}" for r in rep.failures]
    return "\n".join(lines)
