"""Shared fixtures.

The dataset here is small and generated once per session. Tests assert on
*structural* properties (parity, monotonicity, ordering) rather than on metric
values, so they stay meaningful without pinning a model that legitimately
changes between runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.data.generator import generate  # noqa: E402
from fraudplat.features.transforms import (  # noqa: E402
    FeatureEngine,
    MerchantProfile,
    compute_batch,
)


@pytest.fixture(scope="session")
def transactions():
    return generate(n_transactions=6_000, seed=17)


@pytest.fixture(scope="session")
def merchant_profile(transactions):
    return MerchantProfile.fit(transactions)


@pytest.fixture(scope="session")
def features(transactions, merchant_profile):
    return compute_batch(transactions, engine=FeatureEngine(merchant_profile))


@pytest.fixture
def engine(merchant_profile):
    return FeatureEngine(merchant_profile)


@pytest.fixture
def sample_txn():
    return {
        "transaction_id": "txn_test_1",
        "event_time": "2025-02-01T12:00:00+00:00",
        "card_id": "card_0000001",
        "customer_id": "cust_0000001",
        "merchant_id": "mch_000001",
        "merchant_category": "electronics",
        "merchant_country": "US",
        "amount": 120.0,
        "currency": "USD",
        "channel": "ecom",
        "entry_mode": "keyed",
        "device_id": "dev_0000001",
        "issuer_country": "US",
    }


@pytest.fixture
def rng():
    return np.random.default_rng(0)
