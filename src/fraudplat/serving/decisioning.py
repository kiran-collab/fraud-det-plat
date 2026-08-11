"""Score -> action.

The model emits a probability; the business needs one of four actions. Keeping
that mapping in its own module means thresholds can be retuned without
retraining, and the deterministic overlay rules stay visible to Compliance
instead of being folded into a model artifact nobody can read.

Rules run *after* the model and only ever escalate, never de-escalate. A rule
that could downgrade a model decline would be an unreviewable bypass of the
model's authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fraudplat.config import DecisionConfig


class Action(StrEnum):
    APPROVE = "approve"
    STEP_UP = "step_up"      # 3-D Secure / OTP challenge
    REVIEW = "review"        # queue for a human analyst
    DECLINE = "decline"

    @property
    def rank(self) -> int:
        return {"approve": 0, "step_up": 1, "review": 2, "decline": 3}[self.value]


@dataclass
class Decision:
    action: Action
    risk_score: float
    reasons: list[str] = field(default_factory=list)
    triggered_rules: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "risk_score": round(self.risk_score, 6),
            "reasons": self.reasons,
            "triggered_rules": self.triggered_rules,
        }


def _base_action(score: float, cfg: DecisionConfig) -> Action:
    if score >= cfg.decline_at:
        return Action.DECLINE
    if score >= cfg.review_at:
        return Action.REVIEW
    if score >= cfg.step_up_at:
        return Action.STEP_UP
    return Action.APPROVE


def decide(
    score: float,
    txn: Mapping[str, Any],
    features: Mapping[str, float],
    cfg: DecisionConfig,
) -> Decision:
    action = _base_action(score, cfg)
    triggered: list[str] = []

    def escalate(to: Action, rule: str) -> None:
        nonlocal action
        triggered.append(rule)
        if to.rank > action.rank:
            action = to

    # --- deterministic overlays -----------------------------------------
    # Card testing: several auths in an hour, all tiny, card-not-present.
    if features.get("card_txn_count_1h", 0) >= 5 and float(txn.get("amount", 0)) < 20:
        escalate(Action.REVIEW, "velocity.card_testing_burst")

    # Impossible travel: two countries inside an hour on the same card.
    if features.get("card_distinct_countries_7d", 0) >= 3 and features.get("card_txn_count_1h", 0) >= 2:
        escalate(Action.REVIEW, "geo.multi_country_velocity")

    # High-value on a brand-new device is the canonical takeover signature.
    if (
        float(txn.get("amount", 0)) >= cfg.high_value_amount
        and features.get("is_new_device_for_card", 0) == 1.0
    ):
        escalate(Action.STEP_UP, "ato.high_value_new_device")

    # Regulatory: a swipe abroad on a chip-issued card is a counterfeit tell
    # and is separately reportable, so it must always leave an audit record.
    if features.get("entry_is_swipe", 0) == 1.0 and features.get("is_cross_border", 0) == 1.0:
        escalate(Action.REVIEW, "counterfeit.cross_border_swipe")

    return Decision(action=action, risk_score=score, triggered_rules=triggered)


def top_reasons(contributions: Mapping[str, float], k: int = 4) -> list[str]:
    """Human-readable reason codes from SHAP contributions.

    Regulation (and any chargeback dispute) requires an explanation of an
    adverse action, so this runs on the decline path, not just in the UI.
    """
    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:k]
    return [f"{REASON_TEXT.get(name, name)} ({value:+.3f})" for name, value in ranked]


REASON_TEXT: dict[str, str] = {
    "amount": "transaction amount",
    "log_amount": "transaction amount",
    "amount_zscore_card": "amount unusual for this card",
    "amount_ratio_card_mean": "amount far above this card's average",
    "card_txn_count_1h": "high transaction count in the last hour",
    "card_txn_count_24h": "high transaction count in the last 24 hours",
    "card_amount_sum_1h": "high spend in the last hour",
    "card_amount_sum_24h": "high spend in the last 24 hours",
    "card_velocity_ratio": "spend rate above this card's norm",
    "card_seconds_since_last": "unusually short gap since the previous transaction",
    "card_distinct_merchants_24h": "many distinct merchants in 24 hours",
    "card_distinct_countries_7d": "transactions across several countries",
    "card_distinct_devices_7d": "several devices used recently",
    "is_new_merchant_for_card": "first transaction with this merchant",
    "is_new_device_for_card": "device not seen before on this card",
    "is_new_country_for_card": "country not seen before on this card",
    "is_cross_border": "cross-border transaction",
    "merchant_risk_score": "elevated historical fraud rate at this merchant",
    "merchant_log_volume": "low-volume merchant",
    "mcc_risk_score": "high-risk merchant category",
    "entry_is_swipe": "magnetic-stripe entry",
    "entry_is_keyed": "manually keyed card number",
    "channel_is_ecom": "card-not-present transaction",
    "channel_is_atm": "ATM withdrawal",
    "is_night": "transaction outside typical hours",
    "hour_deviation_card": "time of day unusual for this card",
    "hour_of_day": "time of day",
    "day_of_week": "day of week",
    "is_weekend": "weekend transaction",
    "card_amount_max_24h": "large maximum transaction in 24 hours",
    "card_txn_count_7d": "transaction count over 7 days",
}
