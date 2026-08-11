"""Decision-layer tests."""

from __future__ import annotations

import pytest

from fraudplat.config import DecisionConfig
from fraudplat.serving.decisioning import Action, decide, top_reasons


@pytest.fixture
def cfg():
    return DecisionConfig(decline_at=0.86, review_at=0.55, step_up_at=0.35, high_value_amount=2500.0)


@pytest.fixture
def benign_txn():
    return {"amount": 42.0, "card_id": "c1", "merchant_id": "m1"}


@pytest.fixture
def benign_features():
    return {
        "card_txn_count_1h": 0.0,
        "card_distinct_countries_7d": 1.0,
        "is_new_device_for_card": 0.0,
        "entry_is_swipe": 0.0,
        "is_cross_border": 0.0,
    }


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.05, Action.APPROVE),
        (0.34, Action.APPROVE),
        (0.35, Action.STEP_UP),
        (0.54, Action.STEP_UP),
        (0.55, Action.REVIEW),
        (0.85, Action.REVIEW),
        (0.86, Action.DECLINE),
        (0.99, Action.DECLINE),
    ],
)
def test_threshold_bands(score, expected, benign_txn, benign_features, cfg):
    assert decide(score, benign_txn, benign_features, cfg).action is expected


def test_rules_only_escalate_never_downgrade(benign_txn, cfg):
    """A rule that could soften a model decline would be an unreviewable bypass
    of the model's authority. Every overlay must be monotone upward."""
    high_risk_features = {
        "card_txn_count_1h": 9.0,
        "card_distinct_countries_7d": 5.0,
        "is_new_device_for_card": 1.0,
        "entry_is_swipe": 1.0,
        "is_cross_border": 1.0,
    }
    decision = decide(0.95, {"amount": 3.0}, high_risk_features, cfg)
    assert decision.action is Action.DECLINE  # still declined despite REVIEW-level rules
    assert decision.triggered_rules  # and the rules were recorded


def test_card_testing_burst_escalates_a_low_score(cfg):
    """The signature is several tiny card-not-present auths in an hour. The
    model may score each one low in isolation; the pattern is the signal."""
    features = {"card_txn_count_1h": 6.0, "card_distinct_countries_7d": 1.0,
                "is_new_device_for_card": 0.0, "entry_is_swipe": 0.0, "is_cross_border": 0.0}
    decision = decide(0.10, {"amount": 3.50}, features, cfg)
    assert decision.action is Action.REVIEW
    assert "velocity.card_testing_burst" in decision.triggered_rules


def test_high_value_on_new_device_forces_step_up(benign_features, cfg):
    features = {**benign_features, "is_new_device_for_card": 1.0}
    decision = decide(0.05, {"amount": 4000.0}, features, cfg)
    assert decision.action is Action.STEP_UP
    assert "ato.high_value_new_device" in decision.triggered_rules


def test_cross_border_swipe_always_leaves_an_audit_record(benign_txn, cfg):
    """Separately reportable, so it must be recorded even when the model is
    unconcerned."""
    features = {"card_txn_count_1h": 0.0, "card_distinct_countries_7d": 1.0,
                "is_new_device_for_card": 0.0, "entry_is_swipe": 1.0, "is_cross_border": 1.0}
    decision = decide(0.01, benign_txn, features, cfg)
    assert "counterfeit.cross_border_swipe" in decision.triggered_rules
    assert decision.action is Action.REVIEW


def test_reason_codes_are_human_readable_and_ranked():
    contributions = {
        "merchant_risk_score": 2.4,
        "card_txn_count_1h": -0.1,
        "is_new_device_for_card": 1.1,
        "amount": 0.3,
    }
    reasons = top_reasons(contributions, k=2)
    assert len(reasons) == 2
    assert "elevated historical fraud rate" in reasons[0]
    # Ranked by magnitude, so the strongest driver comes first.
    assert "device not seen before" in reasons[1]


def test_reason_codes_rank_by_magnitude_not_sign():
    """A strongly *exculpatory* feature is still one of the top drivers of the
    score and belongs in the explanation."""
    reasons = top_reasons({"amount": 0.2, "merchant_risk_score": -3.0}, k=1)
    assert "merchant" in reasons[0]


def test_action_ranks_are_totally_ordered():
    ranks = [a.rank for a in (Action.APPROVE, Action.STEP_UP, Action.REVIEW, Action.DECLINE)]
    assert ranks == sorted(ranks) == [0, 1, 2, 3]
