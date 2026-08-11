"""Feature engineering - one implementation, two call sites.

The single most expensive bug class in a fraud platform is training/serving
skew: the batch job computes ``card_txn_count_1h`` one way, the streaming
consumer computes it another, and the model silently degrades in production
while every offline metric stays green.

This module removes that failure mode by construction. ``FeatureEngine`` is an
incremental, event-at-a-time engine. The streaming consumer feeds it one
transaction per Kafka message; the batch trainer feeds it the same
transactions in timestamp order. Both get byte-identical vectors, and
``tests/test_features.py`` asserts that equality rather than trusting it.

Every window is *strictly backward-looking*: a transaction never sees itself or
anything after it. That is what makes the offline metrics trustworthy.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from fraudplat.data.schema import PROTECTED_ATTRIBUTES

HOUR = 3_600.0
DAY = 86_400.0
WEEK = 7 * DAY

# How many past feature vectors to retain per card for the sequence model.
# Must be >= the sequence model's seq_len; kept here rather than imported to
# avoid a circular import between features and models.
SEQ_WINDOW = 8

# Prior strength for merchant target encoding. Higher = more shrinkage toward
# the portfolio base rate, which is what you want for thin-file merchants.
MERCHANT_PRIOR_WEIGHT = 50.0

MCC_RISK_PRIOR = {
    "grocery": 0.2, "fuel": 0.5, "restaurant": 0.3, "electronics": 1.6,
    "travel": 1.2, "apparel": 0.7, "digital_goods": 2.2, "pharmacy": 0.4,
    "gambling": 3.0, "money_transfer": 3.4,
}

FEATURE_NAMES: list[str] = [
    # --- transaction-intrinsic ---
    "amount",
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "is_night",
    "is_weekend",
    "channel_is_ecom",
    "channel_is_atm",
    "entry_is_swipe",
    "entry_is_keyed",
    "is_cross_border",
    # --- card behavioural baseline ---
    "amount_zscore_card",
    "amount_ratio_card_mean",
    "hour_deviation_card",
    # --- card velocity ---
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_txn_count_7d",
    "card_amount_sum_1h",
    "card_amount_sum_24h",
    "card_amount_max_24h",
    "card_seconds_since_last",
    "card_velocity_ratio",
    # --- card diversity ---
    "card_distinct_merchants_24h",
    "card_distinct_countries_7d",
    "card_distinct_devices_7d",
    "is_new_merchant_for_card",
    "is_new_device_for_card",
    "is_new_country_for_card",
    # --- counterparty risk ---
    "merchant_risk_score",
    "merchant_log_volume",
    "mcc_risk_score",
]


@dataclass
class MerchantProfile:
    """Smoothed historical fraud rate per merchant.

    Built from the training window only, then frozen and shipped alongside the
    model. Refreshing it requires a retrain, which is intentional: silently
    swapping the encoding under a fixed model is a classic drift source.
    """

    base_rate: float = 0.01
    fraud_counts: dict[str, float] = field(default_factory=dict)
    total_counts: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame, label_col: str = "is_fraud") -> MerchantProfile:
        base = float(df[label_col].mean()) if len(df) else 0.01
        grouped = df.groupby("merchant_id", observed=True)[label_col].agg(["sum", "count"])
        return cls(
            base_rate=base,
            fraud_counts={str(k): float(v) for k, v in grouped["sum"].items()},
            total_counts={str(k): float(v) for k, v in grouped["count"].items()},
        )

    def risk(self, merchant_id: str) -> float:
        total = self.total_counts.get(merchant_id, 0.0)
        fraud = self.fraud_counts.get(merchant_id, 0.0)
        return (fraud + MERCHANT_PRIOR_WEIGHT * self.base_rate) / (total + MERCHANT_PRIOR_WEIGHT)

    def volume(self, merchant_id: str) -> float:
        return self.total_counts.get(merchant_id, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_rate": self.base_rate,
            "fraud_counts": self.fraud_counts,
            "total_counts": self.total_counts,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MerchantProfile:
        return cls(
            base_rate=float(d["base_rate"]),
            fraud_counts=dict(d["fraud_counts"]),
            total_counts=dict(d["total_counts"]),
        )


@dataclass
class _CardState:
    """Rolling 7-day window for one card, plus running moments.

    Memory is bounded by the 7-day prune, which is what keeps the online store
    footprint predictable at 5M transactions/day.
    """

    events: deque = field(default_factory=deque)  # (ts, amount, merchant, country, device, hour)
    # Last few *feature vectors*, kept so the online path can rebuild the
    # sequence model's input window without recomputing history from scratch.
    recent_vectors: deque = field(default_factory=lambda: deque(maxlen=SEQ_WINDOW))
    n_seen: int = 0
    amount_sum: float = 0.0
    amount_sq_sum: float = 0.0
    hour_sin_sum: float = 0.0
    hour_cos_sum: float = 0.0
    merchants_ever: set[str] = field(default_factory=set)
    devices_ever: set[str] = field(default_factory=set)
    countries_ever: set[str] = field(default_factory=set)
    last_ts: float | None = None

    def prune(self, now: float) -> None:
        cutoff = now - WEEK
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    # -- serialization ---------------------------------------------------
    # The online store persists exactly this, so there is one state schema
    # rather than one for batch and a parallel one for streaming.
    def dump(self) -> dict[str, Any]:
        return {
            "events": [list(e) for e in self.events],
            "recent_vectors": [list(v) for v in self.recent_vectors],
            "n_seen": self.n_seen,
            "amount_sum": self.amount_sum,
            "amount_sq_sum": self.amount_sq_sum,
            "hour_sin_sum": self.hour_sin_sum,
            "hour_cos_sum": self.hour_cos_sum,
            # Bounded: an unbounded "ever seen" set on a high-volume card is an
            # unbounded Redis value. 512 merchants is far more than the novelty
            # check needs and caps the envelope at a few tens of KB.
            "merchants_ever": list(self.merchants_ever)[-512:],
            "devices_ever": list(self.devices_ever)[-128:],
            "countries_ever": list(self.countries_ever)[-64:],
            "last_ts": self.last_ts,
        }

    @classmethod
    def load(cls, d: Mapping[str, Any]) -> _CardState:
        return cls(
            events=deque(tuple(e) for e in d.get("events", [])),
            recent_vectors=deque(
                (list(v) for v in d.get("recent_vectors", [])), maxlen=SEQ_WINDOW
            ),
            n_seen=int(d.get("n_seen", 0)),
            amount_sum=float(d.get("amount_sum", 0.0)),
            amount_sq_sum=float(d.get("amount_sq_sum", 0.0)),
            hour_sin_sum=float(d.get("hour_sin_sum", 0.0)),
            hour_cos_sum=float(d.get("hour_cos_sum", 0.0)),
            merchants_ever=set(d.get("merchants_ever", [])),
            devices_ever=set(d.get("devices_ever", [])),
            countries_ever=set(d.get("countries_ever", [])),
            last_ts=d.get("last_ts"),
        )


class FeatureEngine:
    """Stateful, incremental feature builder.

    Not thread-safe by design - shard by ``card_id`` across workers, which is
    also how the Kafka partitions are keyed, so a card's events are always
    ordered within a single consumer.
    """

    def __init__(self, merchant_profile: MerchantProfile | None = None) -> None:
        self.merchant_profile = merchant_profile or MerchantProfile()
        self._cards: dict[str, _CardState] = defaultdict(_CardState)

    # -- state access ----------------------------------------------------
    def card_state(self, card_id: str) -> _CardState:
        return self._cards[card_id]

    def reset(self) -> None:
        self._cards.clear()

    def reset_card(self, card_id: str) -> None:
        """Evict one card. The streaming writer calls this after persisting, so
        a long-running consumer does not accumulate every card it has ever
        seen."""
        self._cards.pop(card_id, None)

    def dump_card(self, card_id: str) -> dict[str, Any]:
        """Serialise one card's state for the online store."""
        return self._cards[card_id].dump()

    def hydrate_card(self, card_id: str, payload: Mapping[str, Any] | None) -> None:
        """Load one card's state from the online store.

        The API process is stateless and horizontally scaled, so it hydrates
        per request rather than holding a resident map of every card.
        """
        self._cards[card_id] = _CardState.load(payload) if payload else _CardState()

    def sequence_window(self, card_id: str, seq_len: int) -> np.ndarray | None:
        """Recent feature vectors for the sequence model, oldest first.

        Returns ``None`` when the engine has no cached vectors for this card -
        the caller then falls back to the neutral sequence score rather than
        scoring against a fabricated history.
        """
        st = self._cards.get(card_id)
        if st is None or not st.recent_vectors:
            return None
        window = np.zeros((seq_len, len(FEATURE_NAMES)), dtype=np.float32)
        recent = list(st.recent_vectors)[-(seq_len - 1):]
        if recent:
            window[seq_len - 1 - len(recent):seq_len - 1] = np.asarray(recent, dtype=np.float32)
        return window

    # -- core ------------------------------------------------------------
    def transform(self, txn: Mapping[str, Any], *, update: bool = True) -> dict[str, float]:
        """Return the feature vector for ``txn`` given everything seen so far.

        ``update=False`` scores without mutating state - used by the API when a
        transaction has not yet been authorised, so a declined attempt does not
        pollute the cardholder's behavioural baseline.
        """
        for protected in PROTECTED_ATTRIBUTES:
            if protected in txn and protected in FEATURE_NAMES:  # pragma: no cover - guard
                raise ValueError(f"protected attribute {protected!r} must not be a feature")

        ts = _epoch(txn["event_time"])
        card_id = str(txn["card_id"])
        merchant_id = str(txn["merchant_id"])
        country = str(txn.get("merchant_country", "US"))
        device = str(txn.get("device_id") or "unknown")
        amount = float(txn["amount"])

        st = self._cards[card_id]
        st.prune(ts)

        dt = datetime.fromtimestamp(ts, tz=_UTC)
        hour, dow = dt.hour, dt.weekday()

        # --- windowed aggregates over the pruned 7d buffer ---------------
        c1h = c24h = 0
        s1h = s24h = 0.0
        max24h = 0.0
        merch_24h: set[str] = set()
        countries_7d: set[str] = set()
        devices_7d: set[str] = set()
        for e_ts, e_amt, e_mch, e_ctry, e_dev, _ in st.events:
            age = ts - e_ts
            countries_7d.add(e_ctry)
            devices_7d.add(e_dev)
            if age <= DAY:
                c24h += 1
                s24h += e_amt
                max24h = max(max24h, e_amt)
                merch_24h.add(e_mch)
                if age <= HOUR:
                    c1h += 1
                    s1h += e_amt

        # --- card amount baseline (running moments, not the window) ------
        if st.n_seen >= 2:
            mean = st.amount_sum / st.n_seen
            var = max(st.amount_sq_sum / st.n_seen - mean * mean, 1e-9)
            zscore = (amount - mean) / np.sqrt(var)
            ratio = amount / max(mean, 1e-6)
        else:
            mean, zscore, ratio = amount, 0.0, 1.0

        # Circular distance between this hour and the card's mean hour.
        if st.n_seen >= 3:
            mean_hour = np.arctan2(st.hour_sin_sum / st.n_seen, st.hour_cos_sum / st.n_seen)
            mean_hour = (np.degrees(mean_hour) % 360) / 15.0  # -> hours
            diff = abs(hour - mean_hour)
            hour_dev = min(diff, 24.0 - diff)
        else:
            hour_dev = 0.0

        seconds_since = (ts - st.last_ts) if st.last_ts is not None else 7 * DAY
        # 1h spend relative to the card's typical hourly run-rate.
        velocity_ratio = s1h / max(s24h / 24.0, 1e-6) if s24h > 0 else 0.0

        feats: dict[str, float] = {
            "amount": amount,
            "log_amount": float(np.log1p(amount)),
            "hour_of_day": float(hour),
            "day_of_week": float(dow),
            "is_night": float(hour < 6 or hour >= 23),
            "is_weekend": float(dow >= 5),
            "channel_is_ecom": float(txn.get("channel") == "ecom"),
            "channel_is_atm": float(txn.get("channel") == "atm"),
            "entry_is_swipe": float(txn.get("entry_mode") == "swipe"),
            "entry_is_keyed": float(txn.get("entry_mode") == "keyed"),
            "is_cross_border": float(country != str(txn.get("issuer_country", "US"))),
            "amount_zscore_card": float(np.clip(zscore, -20.0, 20.0)),
            "amount_ratio_card_mean": float(np.clip(ratio, 0.0, 500.0)),
            "hour_deviation_card": float(hour_dev),
            "card_txn_count_1h": float(c1h),
            "card_txn_count_24h": float(c24h),
            "card_txn_count_7d": float(len(st.events)),
            "card_amount_sum_1h": float(s1h),
            "card_amount_sum_24h": float(s24h),
            "card_amount_max_24h": float(max24h),
            "card_seconds_since_last": float(min(seconds_since, 7 * DAY)),
            "card_velocity_ratio": float(np.clip(velocity_ratio, 0.0, 500.0)),
            "card_distinct_merchants_24h": float(len(merch_24h)),
            "card_distinct_countries_7d": float(len(countries_7d)),
            "card_distinct_devices_7d": float(len(devices_7d)),
            "is_new_merchant_for_card": float(merchant_id not in st.merchants_ever),
            "is_new_device_for_card": float(device not in st.devices_ever),
            "is_new_country_for_card": float(country not in st.countries_ever),
            "merchant_risk_score": float(self.merchant_profile.risk(merchant_id)),
            "merchant_log_volume": float(np.log1p(self.merchant_profile.volume(merchant_id))),
            "mcc_risk_score": float(MCC_RISK_PRIOR.get(str(txn.get("merchant_category")), 1.0)),
        }

        if update:
            st.events.append((ts, amount, merchant_id, country, device, hour))
            st.recent_vectors.append([feats[name] for name in FEATURE_NAMES])
            st.n_seen += 1
            st.amount_sum += amount
            st.amount_sq_sum += amount * amount
            rad = np.radians(hour * 15.0)
            st.hour_sin_sum += float(np.sin(rad))
            st.hour_cos_sum += float(np.cos(rad))
            st.merchants_ever.add(merchant_id)
            st.devices_ever.add(device)
            st.countries_ever.add(country)
            st.last_ts = ts

        return feats

    def vector(self, txn: Mapping[str, Any], *, update: bool = True) -> np.ndarray:
        f = self.transform(txn, update=update)
        return np.array([f[name] for name in FEATURE_NAMES], dtype=np.float32)


_UTC = UTC


def _epoch(value: Any) -> float:
    if value is None:
        # Defaulting to "now" here would be worse than failing: a missing
        # timestamp would silently produce plausible-looking velocity features
        # computed against the wrong instant.
        raise ValueError("event_time is required to compute time-windowed features")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return float(value.timestamp())
    if isinstance(value, datetime):
        v = value if value.tzinfo else value.replace(tzinfo=_UTC)
        return float(v.timestamp())
    return float(pd.Timestamp(value).timestamp())


def compute_batch(
    df: pd.DataFrame,
    merchant_profile: MerchantProfile | None = None,
    engine: FeatureEngine | None = None,
) -> pd.DataFrame:
    """Replay ``df`` in timestamp order through :class:`FeatureEngine`.

    Returns a feature frame indexed identically to ``df``. Pass an existing
    ``engine`` to carry card state across splits - which you must do for the
    validation and test windows, or their velocity features start from zero and
    look nothing like production.
    """
    if not df["event_time"].is_monotonic_increasing:
        df = df.sort_values("event_time")

    eng = engine or FeatureEngine(merchant_profile)
    records: Iterable[dict[str, Any]] = df.to_dict("records")
    rows = [eng.vector(r) for r in records]
    return pd.DataFrame(np.vstack(rows), columns=FEATURE_NAMES, index=df.index)
