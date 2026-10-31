"""The metric layer - one definition per KPI, for every consumer.

This module exists because of a failure mode that has nothing to do with
modelling: four people compute "false positive rate" four ways and bring four
numbers to the same steering meeting. One divides false declines by all
authorizations, one by legitimate authorizations only, one by alerts raised,
one excludes step-ups. Every number is defensible in isolation and the meeting
is about arithmetic instead of the business.

So a KPI is defined exactly once, here, as data: a name, a plain-language
definition, the SQL that computes it, its unit, who owns it, and whether a
higher value is good. The mart build, the BI extracts, the daily report and
the tests all read these definitions rather than re-implementing them. Adding
a KPI is a dictionary entry, and every consumer picks it up.

Two conventions worth stating because they are the ones usually fudged:

* **Rates name their denominator.** ``decline_rate`` is over all
  authorizations; ``false_decline_rate`` is over *legitimate* authorizations
  only. The denominator is in the definition text, not folklore.
* **Label-dependent KPIs are flagged.** Chargebacks settle 30-90 days after
  the authorization, so recent periods have incomplete labels. Any KPI needing
  ``is_fraud`` carries ``requires_label=True`` and must be reported against a
  matured window, or it will read as a sudden improvement in fraud performance
  when it is only missing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Unit(StrEnum):
    COUNT = "count"
    CURRENCY = "currency"
    RATE = "rate"  # 0-1, rendered as a percentage
    BPS = "bps"  # basis points of spend
    RATIO = "ratio"  # unbounded, e.g. 3.2 false positives per catch
    MINUTES = "minutes"


class Domain(StrEnum):
    VOLUME = "volume"  # what came through the door
    DECISIONING = "decisioning"  # what the platform did about it
    DETECTION = "detection"  # how well it did it (needs labels)
    FINANCIAL = "financial"  # what it was worth
    OPERATIONS = "operations"  # what it cost the humans
    MODEL = "model"  # health of the model itself


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    definition: str
    sql: str
    unit: Unit
    domain: Domain
    owner: str
    higher_is_better: bool | None = None  # None = directionless/contextual
    requires_label: bool = False
    target: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def as_select(self) -> str:
        return f"{self.sql} AS {self.key}"


# --- business cost assumptions -------------------------------------------
# Deliberately explicit and overridable. These are *client* inputs, not model
# outputs: a bank with a high-touch customer base prices a false decline very
# differently from a prepaid issuer, and the optimal operating point moves with
# it. Burying them as literals inside a query is how a threshold recommendation
# becomes unarguable for the wrong reason.
@dataclass(frozen=True)
class CostModel:
    recovery_rate: float = 0.85  # share of a blocked fraud actually avoided
    review_cost: float = 4.50  # fully-loaded analyst cost per alert
    false_decline_cost: float = 12.00  # customer-experience cost of a wrong block
    step_up_cost: float = 0.35  # 3-D Secure / OTP cost per challenge
    chargeback_fee: float = 25.00  # scheme fee per fraud that gets through


DEFAULT_COSTS = CostModel()


# --- the catalog ----------------------------------------------------------
# Expressions assume the `fact_decision` grain: one row per authorization
# decision. Flags are materialised in the mart so BI tools never re-derive them.
METRICS: dict[str, Metric] = {
    # -- volume ----------------------------------------------------------
    "authorizations": Metric(
        key="authorizations",
        label="Authorizations",
        definition="Count of authorization decisions scored by the platform.",
        sql="COUNT(*)",
        unit=Unit.COUNT,
        domain=Domain.VOLUME,
        owner="Fraud Strategy",
    ),
    "authorized_spend": Metric(
        key="authorized_spend",
        label="Authorized spend",
        definition="Total value of transactions the platform approved or challenged (i.e. not blocked).",
        sql="SUM(CASE WHEN NOT is_declined THEN amount ELSE 0 END)",
        unit=Unit.CURRENCY,
        domain=Domain.VOLUME,
        owner="Fraud Strategy",
    ),
    "attempted_spend": Metric(
        key="attempted_spend",
        label="Attempted spend",
        definition="Total value of all authorization attempts, blocked or not.",
        sql="SUM(amount)",
        unit=Unit.CURRENCY,
        domain=Domain.VOLUME,
        owner="Fraud Strategy",
    ),
    # -- decisioning -----------------------------------------------------
    "decline_rate": Metric(
        key="decline_rate",
        label="Decline rate",
        definition="Declined decisions as a share of ALL authorizations.",
        sql="AVG(CASE WHEN is_declined THEN 1.0 ELSE 0.0 END)",
        unit=Unit.RATE,
        domain=Domain.DECISIONING,
        owner="Fraud Strategy",
        higher_is_better=None,
        target=0.004,
    ),
    "challenge_rate": Metric(
        key="challenge_rate",
        label="Step-up rate",
        definition="Share of ALL authorizations sent to a 3-D Secure / OTP challenge.",
        sql="AVG(CASE WHEN action = 'step_up' THEN 1.0 ELSE 0.0 END)",
        unit=Unit.RATE,
        domain=Domain.DECISIONING,
        owner="Fraud Strategy",
    ),
    "review_rate": Metric(
        key="review_rate",
        label="Review rate",
        definition="Share of ALL authorizations queued for human review.",
        sql="AVG(CASE WHEN action = 'review' THEN 1.0 ELSE 0.0 END)",
        unit=Unit.RATE,
        domain=Domain.DECISIONING,
        owner="Fraud Operations",
    ),
    "alerts_per_10k": Metric(
        key="alerts_per_10k",
        label="Alerts per 10k authorizations",
        definition="Decisions requiring human review per 10,000 authorizations. Drives analyst headcount.",
        sql="10000.0 * AVG(CASE WHEN action IN ('review','decline') THEN 1.0 ELSE 0.0 END)",
        unit=Unit.RATIO,
        domain=Domain.OPERATIONS,
        owner="Fraud Operations",
        higher_is_better=False,
    ),
    "rule_override_rate": Metric(
        key="rule_override_rate",
        label="Rule escalation rate",
        definition="Share of decisions where a deterministic rule escalated the model's action.",
        sql="AVG(CASE WHEN triggered_rule_count > 0 THEN 1.0 ELSE 0.0 END)",
        unit=Unit.RATE,
        domain=Domain.DECISIONING,
        owner="Fraud Strategy",
        tags=("governance",),
    ),
    # -- detection (label-dependent) -------------------------------------
    "fraud_prevalence_bps": Metric(
        key="fraud_prevalence_bps",
        label="Fraud prevalence",
        definition="Confirmed-fraud authorizations per 10,000 authorizations.",
        sql="10000.0 * AVG(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0.0 END)",
        unit=Unit.BPS,
        domain=Domain.DETECTION,
        owner="Fraud Strategy",
        requires_label=True,
    ),
    "detection_rate": Metric(
        key="detection_rate",
        label="Detection rate (recall)",
        definition="Share of confirmed-fraud authorizations that were declined. Denominator: confirmed fraud only.",
        sql=(
            "SUM(CASE WHEN is_fraud = 1 AND is_declined THEN 1.0 ELSE 0.0 END) "
            "/ NULLIF(SUM(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0.0 END), 0)"
        ),
        unit=Unit.RATE,
        domain=Domain.DETECTION,
        owner="Data Science",
        higher_is_better=True,
        requires_label=True,
        target=0.30,
    ),
    "value_detection_rate": Metric(
        key="value_detection_rate",
        label="Value detection rate",
        definition=(
            "Share of confirmed-fraud VALUE that was declined. Reported alongside "
            "detection rate because catching few large frauds beats catching many small ones."
        ),
        sql=(
            "SUM(CASE WHEN is_fraud = 1 AND is_declined THEN amount ELSE 0 END) "
            "/ NULLIF(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0)"
        ),
        unit=Unit.RATE,
        domain=Domain.DETECTION,
        owner="Data Science",
        higher_is_better=True,
        requires_label=True,
        target=0.60,
    ),
    "precision_at_decline": Metric(
        key="precision_at_decline",
        label="Decline precision",
        definition="Share of declined decisions that were confirmed fraud. Denominator: declines only.",
        sql=(
            "SUM(CASE WHEN is_declined AND is_fraud = 1 THEN 1.0 ELSE 0.0 END) "
            "/ NULLIF(SUM(CASE WHEN is_declined THEN 1.0 ELSE 0.0 END), 0)"
        ),
        unit=Unit.RATE,
        domain=Domain.DETECTION,
        owner="Data Science",
        higher_is_better=True,
        requires_label=True,
        target=0.70,
    ),
    "false_decline_rate": Metric(
        key="false_decline_rate",
        label="False decline rate",
        definition=(
            "Legitimate authorizations wrongly declined, as a share of LEGITIMATE "
            "authorizations only. The metric customers actually feel."
        ),
        sql=(
            "SUM(CASE WHEN is_declined AND is_fraud = 0 THEN 1.0 ELSE 0.0 END) "
            "/ NULLIF(SUM(CASE WHEN is_fraud = 0 THEN 1.0 ELSE 0.0 END), 0)"
        ),
        unit=Unit.RATE,
        domain=Domain.DETECTION,
        owner="Fraud Strategy",
        higher_is_better=False,
        requires_label=True,
        target=0.0005,
    ),
    "false_positive_ratio": Metric(
        key="false_positive_ratio",
        label="False positives per catch",
        definition="Legitimate transactions declined for every fraud caught. The operational framing of precision.",
        sql=(
            "SUM(CASE WHEN is_declined AND is_fraud = 0 THEN 1.0 ELSE 0.0 END) "
            "/ NULLIF(SUM(CASE WHEN is_declined AND is_fraud = 1 THEN 1.0 ELSE 0.0 END), 0)"
        ),
        unit=Unit.RATIO,
        domain=Domain.OPERATIONS,
        owner="Fraud Operations",
        higher_is_better=False,
        requires_label=True,
    ),
    "fraud_loss_bps": Metric(
        key="fraud_loss_bps",
        label="Fraud loss (bps of spend)",
        definition="Value of fraud that was NOT declined, per 10,000 units of attempted spend.",
        sql=(
            "10000.0 * SUM(CASE WHEN is_fraud = 1 AND NOT is_declined THEN amount ELSE 0 END) "
            "/ NULLIF(SUM(amount), 0)"
        ),
        unit=Unit.BPS,
        domain=Domain.FINANCIAL,
        owner="Fraud Strategy",
        higher_is_better=False,
        requires_label=True,
    ),
    # -- model health (no labels needed - available immediately) ----------
    "mean_risk_score": Metric(
        key="mean_risk_score",
        label="Mean risk score",
        definition="Average blended risk score. A leading drift indicator - moves before labels arrive.",
        sql="AVG(risk_score)",
        unit=Unit.RATE,
        domain=Domain.MODEL,
        owner="Data Science",
        tags=("drift",),
    ),
    "high_risk_rate": Metric(
        key="high_risk_rate",
        label="High-risk rate",
        definition="Share of authorizations scoring above 0.55. Monitored daily as an early drift signal.",
        sql="AVG(CASE WHEN risk_score >= 0.55 THEN 1.0 ELSE 0.0 END)",
        unit=Unit.RATE,
        domain=Domain.MODEL,
        owner="Data Science",
        tags=("drift",),
    ),
    "cold_start_rate": Metric(
        key="cold_start_rate",
        label="Cold-start rate",
        definition=(
            "Share of authorizations on cards with too little history for the sequence "
            "model, scored at the neutral 0.5. High values mean reduced model coverage."
        ),
        sql="AVG(CASE WHEN sequence_score = 0.5 THEN 1.0 ELSE 0.0 END)",
        unit=Unit.RATE,
        domain=Domain.MODEL,
        owner="Data Science",
        higher_is_better=False,
        tags=("coverage",),
    ),
    "label_coverage": Metric(
        key="label_coverage",
        label="Label coverage",
        definition=(
            "Share of authorizations with a settled fraud outcome. Below ~95% the "
            "detection KPIs for that period are not yet reportable."
        ),
        sql="AVG(CASE WHEN is_fraud IS NOT NULL THEN 1.0 ELSE 0.0 END)",
        unit=Unit.RATE,
        domain=Domain.MODEL,
        owner="Data Governance",
        higher_is_better=True,
        target=0.95,
        tags=("data_quality",),
    ),
}


def metric(key: str) -> Metric:
    if key not in METRICS:
        raise KeyError(f"unknown metric '{key}'. Add it to METRICS rather than inlining SQL.")
    return METRICS[key]


def select_clause(keys: list[str] | None = None) -> str:
    keys = keys or list(METRICS)
    return ",\n       ".join(metric(k).as_select() for k in keys)


def by_domain(domain: Domain) -> list[Metric]:
    return [m for m in METRICS.values() if m.domain == domain]


def label_dependent() -> list[str]:
    return [k for k, m in METRICS.items() if m.requires_label]


def catalog_rows() -> list[dict]:
    """The KPI catalog, as rows - published to BI so consumers can read the
    definition next to the number instead of guessing at it."""
    return [
        {
            "metric_key": m.key,
            "metric_label": m.label,
            "definition": m.definition,
            "unit": m.unit.value,
            "domain": m.domain.value,
            "owner": m.owner,
            "direction": (
                "higher is better"
                if m.higher_is_better
                else "lower is better"
                if m.higher_is_better is False
                else "contextual"
            ),
            "requires_settled_label": m.requires_label,
            "target": m.target,
            "tags": ",".join(m.tags),
        }
        for m in METRICS.values()
    ]
