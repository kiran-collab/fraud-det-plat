"""Canonical transaction contract.

This is the single schema shared by the Kafka producer, the streaming feature
writer, the batch training job and the scoring API. Anything that reads or
writes a transaction goes through here so the online and offline paths cannot
drift apart.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Channel = Literal["pos", "ecom", "atm", "moto"]
EntryMode = Literal["chip", "contactless", "swipe", "keyed", "token"]

RAW_COLUMNS = [
    "transaction_id",
    "event_time",
    "card_id",
    "customer_id",
    "merchant_id",
    "merchant_category",
    "merchant_country",
    "amount",
    "currency",
    "channel",
    "entry_mode",
    "device_id",
    "ip_prefix",
    "issuer_country",
    "customer_age_band",
    "is_fraud",
]


class Transaction(BaseModel):
    """One authorization request."""

    transaction_id: str
    event_time: datetime
    card_id: str
    customer_id: str
    merchant_id: str
    merchant_category: str
    merchant_country: str
    amount: float = Field(gt=0)
    currency: str = "USD"
    channel: Channel = "pos"
    entry_mode: EntryMode = "chip"
    device_id: str | None = None
    ip_prefix: str | None = None
    issuer_country: str = "US"
    # Protected-attribute proxy retained ONLY for offline bias monitoring.
    # It is stripped before feature assembly - see features.transforms.
    customer_age_band: str | None = None
    is_fraud: int | None = None

    @field_validator("event_time")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=UTC)

    def to_record(self) -> dict[str, Any]:
        d = self.model_dump()
        d["event_time"] = self.event_time.isoformat()
        return d


# Columns that must never reach a model. Enforced in features.transforms and
# asserted in tests/test_features.py.
PROTECTED_ATTRIBUTES = frozenset({"customer_age_band"})

# Direct identifiers. High-cardinality and trivially memorised, so they are
# used to build aggregates but never passed to the estimator as raw values.
IDENTIFIER_COLUMNS = frozenset(
    {"transaction_id", "card_id", "customer_id", "merchant_id", "device_id", "ip_prefix"}
)
