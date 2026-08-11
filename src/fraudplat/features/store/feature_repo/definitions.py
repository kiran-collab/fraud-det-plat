"""Feast feature definitions.

Feast owns the **contract and the point-in-time correctness guarantee**; the
arithmetic still lives in ``features/transforms.py``. That division is
deliberate. Re-expressing the velocity logic as Feast on-demand transforms
would duplicate it into a second implementation and reintroduce exactly the
train/serve skew the shared engine exists to prevent.

So:

* the streaming consumer computes vectors with ``FeatureEngine`` and pushes
  them to Feast via the push source;
* training pulls the same rows through ``get_historical_features``, which
  applies the point-in-time join against the label timestamps;
* serving reads them back through ``get_online_features``.

``ttl`` on each view is a correctness control, not a storage one: a feature
older than its TTL is served as null rather than as a stale value, which fails
loudly instead of scoring a card against week-old velocity.
"""

from __future__ import annotations

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, PushSource, ValueType
from feast.types import Float32, Int64, String

# --- entities -----------------------------------------------------------
card = Entity(
    name="card",
    join_keys=["card_id"],
    value_type=ValueType.STRING,
    description="Payment card (PAN token). The primary velocity entity.",
)

merchant = Entity(
    name="merchant",
    join_keys=["merchant_id"],
    value_type=ValueType.STRING,
    description="Acquiring merchant.",
)

# --- sources ------------------------------------------------------------
# Batch source is the historical log used for point-in-time joins at training
# time. In EKS this is the Glue/Athena table over the S3 transaction lake.
card_batch_source = FileSource(
    name="card_features_batch",
    path="s3://fraudplat-features/card_features/",
    timestamp_field="event_time",
    created_timestamp_column="created_at",
)

# Push source is how the Kafka consumer lands features online with low latency
# while the same rows also flow to the offline store for the next retrain.
card_push_source = PushSource(name="card_features_push", batch_source=card_batch_source)

merchant_batch_source = FileSource(
    name="merchant_features_batch",
    path="s3://fraudplat-features/merchant_features/",
    timestamp_field="event_time",
    created_timestamp_column="created_at",
)

# --- views --------------------------------------------------------------
card_velocity_view = FeatureView(
    name="card_velocity",
    entities=[card],
    # 25h, not 24h: the longest window these features describe is 24 hours, and
    # an exact-24h TTL races the window boundary and nulls out live features.
    ttl=timedelta(hours=25),
    schema=[
        Field(name="card_txn_count_1h", dtype=Float32),
        Field(name="card_txn_count_24h", dtype=Float32),
        Field(name="card_txn_count_7d", dtype=Float32),
        Field(name="card_amount_sum_1h", dtype=Float32),
        Field(name="card_amount_sum_24h", dtype=Float32),
        Field(name="card_amount_max_24h", dtype=Float32),
        Field(name="card_seconds_since_last", dtype=Float32),
        Field(name="card_velocity_ratio", dtype=Float32),
        Field(name="card_distinct_merchants_24h", dtype=Float32),
        Field(name="card_distinct_countries_7d", dtype=Float32),
        Field(name="card_distinct_devices_7d", dtype=Float32),
    ],
    source=card_push_source,
    online=True,
    tags={"team": "fraud-ml", "tier": "realtime", "pii": "false"},
)

card_profile_view = FeatureView(
    name="card_profile",
    entities=[card],
    ttl=timedelta(days=8),
    schema=[
        Field(name="amount_zscore_card", dtype=Float32),
        Field(name="amount_ratio_card_mean", dtype=Float32),
        Field(name="hour_deviation_card", dtype=Float32),
        Field(name="is_new_merchant_for_card", dtype=Float32),
        Field(name="is_new_device_for_card", dtype=Float32),
        Field(name="is_new_country_for_card", dtype=Float32),
        Field(name="card_lifetime_txn_count", dtype=Int64),
    ],
    source=card_push_source,
    online=True,
    tags={"team": "fraud-ml", "tier": "realtime", "pii": "false"},
)

merchant_risk_view = FeatureView(
    name="merchant_risk",
    entities=[merchant],
    # Merchant risk is refreshed by the nightly batch job, so a longer TTL is
    # correct here - unlike the velocity views, a day-old value is still valid.
    ttl=timedelta(days=3),
    schema=[
        Field(name="merchant_risk_score", dtype=Float32),
        Field(name="merchant_log_volume", dtype=Float32),
        Field(name="mcc_risk_score", dtype=Float32),
        Field(name="merchant_category", dtype=String),
    ],
    source=merchant_batch_source,
    online=True,
    tags={"team": "fraud-ml", "tier": "batch", "pii": "false"},
)

ALL_VIEWS = [card_velocity_view, card_profile_view, merchant_risk_view]
