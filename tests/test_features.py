"""Feature engine tests.

The parity test is the important one. Everything else in the platform is built
on the claim that the batch trainer and the streaming scorer compute identical
features; if that claim ever stops holding, every offline metric becomes a
fiction and nothing else here would catch it.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudplat.data.schema import PROTECTED_ATTRIBUTES
from fraudplat.features.transforms import (
    FEATURE_NAMES,
    FeatureEngine,
    MerchantProfile,
    compute_batch,
)


def test_batch_and_stream_produce_identical_features(transactions, merchant_profile):
    """The central invariant: replaying event-at-a-time must equal the batch path."""
    batch = compute_batch(transactions, engine=FeatureEngine(merchant_profile))

    stream_engine = FeatureEngine(merchant_profile)
    stream_rows = [stream_engine.vector(row) for row in transactions.to_dict("records")]
    stream = np.vstack(stream_rows)

    np.testing.assert_allclose(batch.to_numpy(), stream, rtol=0, atol=0)


def test_streaming_survives_state_round_trip(transactions, merchant_profile):
    """Hydrating from the online store must not change a single feature value.

    This is the serving path: each request loads the card's state from Redis,
    scores, and writes it back. If serialization is lossy, production silently
    diverges from training even though the *code* is shared.
    """
    rows = transactions.head(800).to_dict("records")

    direct = FeatureEngine(merchant_profile)
    expected = np.vstack([direct.vector(r) for r in rows])

    store: dict[str, dict] = {}
    round_tripped = []
    for r in rows:
        engine = FeatureEngine(merchant_profile)  # fresh process each time
        card = str(r["card_id"])
        engine.hydrate_card(card, store.get(card))
        round_tripped.append(engine.vector(r))
        store[card] = engine.dump_card(card)

    np.testing.assert_allclose(expected, np.vstack(round_tripped), rtol=0, atol=0)


def test_features_are_strictly_backward_looking(engine, sample_txn):
    """A transaction must never see itself in its own aggregates."""
    first = engine.transform(sample_txn, update=True)
    assert first["card_txn_count_1h"] == 0
    assert first["card_txn_count_24h"] == 0
    assert first["is_new_merchant_for_card"] == 1.0

    second = engine.transform({**sample_txn, "transaction_id": "t2"}, update=True)
    assert second["card_txn_count_1h"] == 1  # sees the first, not itself
    assert second["is_new_merchant_for_card"] == 0.0


def test_update_false_does_not_mutate_state(engine, sample_txn):
    """A pre-auth quote must not advance the cardholder's baseline."""
    engine.transform(sample_txn, update=True)
    before = engine.dump_card(sample_txn["card_id"])
    engine.transform({**sample_txn, "amount": 9999.0}, update=False)
    assert engine.dump_card(sample_txn["card_id"]) == before


def test_no_protected_attribute_is_a_feature():
    assert PROTECTED_ATTRIBUTES.isdisjoint(FEATURE_NAMES)


def test_no_raw_identifier_is_a_feature():
    """Identifiers drive aggregates but must never be passed through raw -
    a model that memorises card IDs cannot generalise to a new card."""
    for name in FEATURE_NAMES:
        assert name not in {"card_id", "customer_id", "merchant_id", "device_id", "transaction_id"}


def test_velocity_windows_respect_boundaries(engine, sample_txn):
    base = "2025-02-01T12:00:00+00:00"
    engine.transform({**sample_txn, "event_time": base}, update=True)

    # 30 minutes later: inside the 1h window.
    f = engine.transform(
        {**sample_txn, "event_time": "2025-02-01T12:30:00+00:00"}, update=True
    )
    assert f["card_txn_count_1h"] == 1

    # 3 hours after the first: outside 1h, still inside 24h.
    f = engine.transform(
        {**sample_txn, "event_time": "2025-02-01T15:30:00+00:00"}, update=True
    )
    assert f["card_txn_count_1h"] == 0
    assert f["card_txn_count_24h"] == 2

    # 9 days later: everything has aged out of the 7d buffer.
    f = engine.transform(
        {**sample_txn, "event_time": "2025-02-10T15:30:00+00:00"}, update=True
    )
    assert f["card_txn_count_7d"] == 0


def test_merchant_profile_shrinks_thin_file_merchants():
    """A merchant with one transaction must not get a 100% fraud rate."""
    import pandas as pd

    # m_safe supplies the volume that sets a low portfolio base rate, so
    # m_busy is genuinely riskier than the portfolio rather than merely being
    # the portfolio (a single dominant merchant *is* the base rate, and its
    # shrunk estimate is then correctly pulled toward itself).
    df = pd.DataFrame({
        "merchant_id": ["m_busy"] * 500 + ["m_safe"] * 2000 + ["m_new"],
        "is_fraud": [1] * 100 + [0] * 400 + [1] * 20 + [0] * 1980 + [1],
    })
    profile = MerchantProfile.fit(df)
    assert profile.base_rate < 0.06
    # One transaction, one chargeback - must not become a 100% risk score.
    assert profile.risk("m_new") < 0.2, "thin-file merchant was not shrunk toward the prior"
    # A high-volume, genuinely risky merchant keeps its signal.
    assert profile.risk("m_busy") > 3 * profile.base_rate
    assert profile.risk("m_safe") < profile.base_rate
    # An entirely unseen merchant falls back to the portfolio base rate.
    assert profile.risk("m_unknown") == pytest.approx(profile.base_rate, abs=1e-9)


def test_cold_card_features_are_neutral_not_extreme(engine, sample_txn):
    """A card's first transaction must not look anomalous purely for being first."""
    f = engine.transform(sample_txn, update=True)
    assert f["amount_zscore_card"] == 0.0
    assert f["amount_ratio_card_mean"] == 1.0
    assert f["hour_deviation_card"] == 0.0


def test_feature_vector_matches_declared_order(engine, sample_txn):
    feats = engine.transform(sample_txn, update=False)
    vector = engine.vector(sample_txn, update=False)
    assert list(feats.keys()) == FEATURE_NAMES
    np.testing.assert_allclose(vector, [feats[n] for n in FEATURE_NAMES])


def test_all_features_are_finite(features):
    """NaN or inf reaching LightGBM is silent; reaching ONNX is a crash."""
    values = features.to_numpy()
    assert np.isfinite(values).all()
