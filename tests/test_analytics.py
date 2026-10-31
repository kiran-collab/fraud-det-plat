"""Analytics framework tests.

The metric-layer tests are the ones that matter most. A KPI whose SQL quietly
changes meaning is undetectable in a dashboard - the number simply becomes
different, and whoever notices assumes the business moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudplat.analytics import metrics as M
from fraudplat.analytics import quality, thresholds
from fraudplat.analytics.metrics import CostModel


# --------------------------------------------------------------------------
# Metric layer
# --------------------------------------------------------------------------
def test_every_metric_is_fully_specified():
    """A KPI without an owner or a definition cannot be governed."""
    for key, m in M.METRICS.items():
        assert m.key == key, f"{key} key mismatch"
        assert m.definition.strip() and m.definition.endswith("."), (
            f"{key} needs a prose definition"
        )
        assert m.owner, f"{key} has no accountable owner"
        assert m.sql.strip(), f"{key} has no SQL"


def test_rate_definitions_state_their_denominator():
    """Rates are the metrics people compute four different ways. The
    denominator must be in the text, not in folklore."""
    for m in M.METRICS.values():
        if m.unit is M.Unit.RATE and m.requires_label:
            text = m.definition.lower()
            assert any(w in text for w in ("denominator", "share of", "as a share")), (
                f"{m.key} is a rate but does not state its denominator"
            )


def test_label_dependent_metrics_are_flagged():
    """Chargebacks settle 30-90 days later, so these cannot be reported on a
    recent window without reading as an improvement."""
    for key in (
        "detection_rate",
        "value_detection_rate",
        "precision_at_decline",
        "false_decline_rate",
        "fraud_loss_bps",
    ):
        assert M.metric(key).requires_label, f"{key} needs labels but is not flagged"


def test_model_health_metrics_need_no_labels():
    """These exist precisely to monitor the model before labels arrive."""
    for key in ("mean_risk_score", "high_risk_rate", "cold_start_rate"):
        assert not M.metric(key).requires_label


def test_unknown_metric_fails_loudly():
    with pytest.raises(KeyError, match="rather than inlining SQL"):
        M.metric("made_up_kpi")


def test_catalog_is_publishable():
    rows = M.catalog_rows()
    assert len(rows) == len(M.METRICS)
    assert all(
        r["direction"] in {"higher is better", "lower is better", "contextual"} for r in rows
    )


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------
def _decisions(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-03-01", periods=n, freq="15min", tz="UTC")
    score = rng.beta(1, 30, n)
    action = np.where(score >= 0.86, "decline", np.where(score >= 0.55, "review", "approve"))
    return pd.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(n)],
            "event_time": t,
            "card_id": rng.choice([f"c{i}" for i in range(40)], n),
            "merchant_id": rng.choice([f"m{i}" for i in range(20)], n),
            "amount": rng.uniform(5, 500, n).round(2),
            "risk_score": score,
            "action": action,
            "is_declined": action == "decline",
            "triggered_rule_count": 0,
            "is_fraud": (rng.random(n) < 0.01).astype(int),
        }
    )


def test_clean_data_passes_and_publishes():
    d = _decisions()
    rep = quality.run_quality_checks(d, reference_time=d.event_time.max())
    assert rep.publishable
    assert rep.score() == 1.0, [
        r.check for r in rep.results if r.severity is not quality.Severity.PASS
    ]


def test_duplicate_grain_blocks_publication():
    """Duplicates double-count fraud and inflate every rate."""
    d = pd.concat([_decisions(100), _decisions(100)])
    rep = quality.run_quality_checks(d, reference_time=d.event_time.max())
    assert not rep.publishable
    assert any(r.check == "transaction_id_unique" for r in rep.failures)


def test_inconsistent_flag_blocks_publication():
    """A materialised flag disagreeing with its source means every BI tile
    built on that flag is wrong."""
    d = _decisions()
    d.loc[d.index[:50], "is_declined"] = True
    rep = quality.run_quality_checks(d, reference_time=d.event_time.max())
    assert not rep.publishable
    assert any(r.check == "flag_matches_action" for r in rep.failures)


def test_out_of_domain_values_are_caught():
    d = _decisions()
    d.loc[d.index[0], "action"] = "refer_to_issuer"
    d.loc[d.index[1], "risk_score"] = 1.7
    rep = quality.run_quality_checks(d, reference_time=d.event_time.max())
    checks = {r.check for r in rep.failures}
    assert {"action_in_domain", "risk_score_in_0_1"} <= checks


def test_stale_data_warns_but_does_not_block():
    """Late data is a reporting caveat, not a reason to withhold the refresh."""
    d = _decisions()
    rep = quality.run_quality_checks(d, reference_time=d.event_time.max() + pd.Timedelta(days=30))
    staleness = next(r for r in rep.results if r.check == "data_freshness_days")
    assert staleness.severity is quality.Severity.FAIL
    assert rep.publishable, "staleness must not block publication"


@pytest.mark.parametrize(
    "missing,expected",
    [
        (200, quality.Severity.WARN),  # 50% coverage - approaching the limit
        (380, quality.Severity.FAIL),
    ],  # 5% coverage - unreportable
)
def test_incomplete_labels_escalate_but_never_block(missing, expected):
    """Chargebacks settle late, so thin labels are the normal state of a
    recent window. It must downgrade the report, not withhold it."""
    d = _decisions()
    d["is_fraud"] = d.is_fraud.astype(float)
    d.loc[d.index[:missing], "is_fraud"] = np.nan
    rep = quality.run_quality_checks(d, reference_time=d.event_time.max())
    cov = next(r for r in rep.results if r.check == "label_coverage")
    assert cov.severity is expected
    assert rep.publishable, "label lag must not block publication"


def test_empty_dataset_blocks():
    rep = quality.run_quality_checks(_decisions().iloc[:0])
    assert not rep.publishable


# --------------------------------------------------------------------------
# Threshold economics
# --------------------------------------------------------------------------
def _labelled(n=6000, seed=1):
    """A deliberately OVERLAPPING score distribution.

    An earlier version drew legitimate scores from beta(1, 30) and simply
    added 0.55 for fraud, which is perfectly separable: zero false declines at
    every threshold and no fraud above 0.9. Three tests below passed vacuously
    against it - precision was 1.0 wherever there were any declines, and the
    cost model could not change the optimum because there were no false
    declines to price. Real score distributions overlap, so this one must too.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.02).astype(int)
    scores = np.where(y == 1, rng.beta(4, 3, n), rng.beta(1.2, 12, n)).clip(0, 1)
    amounts = rng.lognormal(4, 1, n) * (1 + 2 * y)  # fraud skews high-value
    return y, scores, amounts


def test_raising_the_threshold_trades_recall_for_precision():
    y, s, a = _labelled()
    low = thresholds.evaluate_threshold(y, s, a, 0.30, 0.30)
    high = thresholds.evaluate_threshold(y, s, a, 0.80, 0.55)
    assert low.detection_rate > high.detection_rate
    assert high.precision > low.precision
    assert low.false_declines > high.false_declines
    # Value detection must move with count detection, or the fixture has no
    # relationship between score and amount.
    assert low.value_detection_rate > high.value_detection_rate


def test_frontier_is_monotone_in_the_obvious_directions():
    y, s, a = _labelled()
    f = thresholds.sweep(y, s, a)
    assert f.detection_rate.is_monotonic_decreasing
    assert f.declines.is_monotonic_decreasing
    assert f.net_benefit.notna().all()


def test_cost_assumptions_actually_move_the_recommendation():
    """If the optimum ignored the cost model, the whole exercise would be
    theatre."""
    y, s, a = _labelled()
    cheap = thresholds.optimal_point(
        thresholds.sweep(y, s, a, costs=CostModel(false_decline_cost=1.0))
    )
    dear = thresholds.optimal_point(
        thresholds.sweep(y, s, a, costs=CostModel(false_decline_cost=400.0))
    )
    assert dear.decline_threshold > cheap.decline_threshold, (
        "expensive false declines must push the threshold up"
    )


def test_constraints_bind():
    y, s, a = _labelled()
    f = thresholds.sweep(y, s, a)
    free = thresholds.optimal_point(f)
    tight = thresholds.constrained_optimum(f, max_false_decline_rate=0.0002)
    assert tight is not None
    assert tight.false_decline_rate <= 0.0002
    assert tight.net_benefit <= free.net_benefit  # a constraint can only cost


def test_impossible_constraint_returns_none_rather_than_a_bad_answer():
    y, s, a = _labelled()
    assert thresholds.constrained_optimum(thresholds.sweep(y, s, a), max_alerts_per_10k=0.0) is None


def test_queue_metric_is_named_to_prevent_a_false_comparison():
    """The frontier cannot model rule escalations, so its queue figure must not
    share a name with the KPI that does."""
    y, s, a = _labelled()
    assert "alerts_per_10k_score_only" in thresholds.sweep(y, s, a).columns
    assert "alerts_per_10k" not in thresholds.sweep(y, s, a).columns


def test_sensitivity_covers_each_assumption():
    y, s, a = _labelled(n=1500)
    out = thresholds.sensitivity(y, s, a, multipliers=(0.5, 1.0, 2.0))
    assert set(out.assumption) == {"false_decline_cost", "review_cost", "recovery_rate"}
    assert (out.optimal_threshold.between(0, 1)).all()


def test_recovery_rate_is_never_above_one():
    """A 4x multiplier on an 0.85 recovery rate must clamp, not imply that
    blocking fraud recovers more than the fraud was worth."""
    y, s, a = _labelled(n=1200)
    out = thresholds.sensitivity(y, s, a, multipliers=(4.0,))
    assert (out[out.assumption == "recovery_rate"].value <= 1.0).all()
