"""Synthetic transaction generator.

Production reads from the card network feed; this generator exists so the whole
platform - training, streaming, serving, evaluation - is runnable end to end
without access to real cardholder data.

The fraud it injects is deliberately *patterned* rather than random, because a
model that can only separate uniform noise tells you nothing:

  * card-testing      - a burst of small ecom auths on a fresh merchant
  * account-takeover  - high-value auths from a new device after a quiet period
  * counterfeit       - swipe transactions in a country the card never visits
  * merchant-collusion- an inflated fraud rate concentrated on a few merchants
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudplat.data.schema import RAW_COLUMNS

MCC_POOL = [
    "grocery", "fuel", "restaurant", "electronics", "travel",
    "apparel", "digital_goods", "pharmacy", "gambling", "money_transfer",
]
# Base fraud propensity by category - gambling / money transfer / digital goods
# carry more risk, which is what the merchant-risk feature should learn.
MCC_RISK = {
    "grocery": 0.2, "fuel": 0.5, "restaurant": 0.3, "electronics": 1.6,
    "travel": 1.2, "apparel": 0.7, "digital_goods": 2.2, "pharmacy": 0.4,
    "gambling": 3.0, "money_transfer": 3.4,
}
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "MX", "BR", "IN", "NG", "RO"]
FOREIGN_WEIGHTS = np.array([0.62, 0.08, 0.06, 0.05, 0.04, 0.05, 0.04, 0.03, 0.02, 0.01])


def _amount_for(rng: np.random.Generator, mcc: str, n: int) -> np.ndarray:
    """Log-normal amounts with category-specific location/scale."""
    loc, scale = {
        "grocery": (3.6, 0.6), "fuel": (3.5, 0.5), "restaurant": (3.3, 0.7),
        "electronics": (5.2, 0.9), "travel": (5.6, 1.0), "apparel": (4.1, 0.8),
        "digital_goods": (2.8, 1.1), "pharmacy": (3.2, 0.7),
        "gambling": (4.4, 1.2), "money_transfer": (5.0, 1.3),
    }[mcc]
    return np.round(rng.lognormal(loc, scale, n), 2).clip(1.0, 25_000.0)


def generate(
    n_transactions: int = 250_000,
    n_cards: int | None = None,
    n_merchants: int | None = None,
    days: int = 60,
    fraud_rate: float = 0.0085,
    seed: int = 17,
) -> pd.DataFrame:
    """Return a time-sorted transaction frame with a labelled ``is_fraud`` column.

    ``fraud_rate`` is the target prevalence; the realised rate lands within a
    few basis points because the pattern injectors overlap slightly.

    ``n_cards`` and ``n_merchants`` default to ratios that hold the *per-entity*
    density roughly constant as volume changes: ~20 transactions per card and
    ~80 per merchant. That density is the thing that matters - velocity
    features, card baselines and the sequence model all need a card to have a
    history, and a merchant needs volume before its fraud rate means anything.
    Holding the entity counts fixed while shrinking the row count (the obvious
    thing to do for a fast smoke test) silently produces a dataset where none
    of those features carry signal, and every model looks broken for reasons
    that have nothing to do with the model.
    """
    rng = np.random.default_rng(seed)
    n = n_transactions
    n_cards = n_cards or max(200, n // 20)
    n_merchants = n_merchants or max(50, n // 80)

    card_ids = np.array([f"card_{i:07d}" for i in range(n_cards)])
    merchant_ids = np.array([f"mch_{i:06d}" for i in range(n_merchants)])
    merchant_mcc = rng.choice(MCC_POOL, n_merchants)
    merchant_country = rng.choice(COUNTRIES, n_merchants, p=FOREIGN_WEIGHTS / FOREIGN_WEIGHTS.sum())

    # Cards have unequal activity (Zipf-ish): a few very active, a long tail.
    card_weights = rng.pareto(1.6, n_cards) + 1.0
    card_weights /= card_weights.sum()
    card_idx = rng.choice(n_cards, n, p=card_weights)

    # Merchants likewise - a handful of large acquirers dominate volume.
    mch_weights = rng.pareto(1.3, n_merchants) + 1.0
    mch_weights /= mch_weights.sum()
    mch_idx = rng.choice(n_merchants, n, p=mch_weights)

    # Timestamps: uniform over the window then bent toward waking hours.
    start = pd.Timestamp("2025-01-01", tz="UTC")
    offsets = rng.uniform(0, days * 86_400, n)
    ts = start + pd.to_timedelta(offsets, unit="s")
    hour_shift = rng.normal(0, 2.5, n) * (rng.random(n) < 0.7)
    ts = ts + pd.to_timedelta(hour_shift, unit="h")

    mcc = merchant_mcc[mch_idx]
    amount = np.empty(n)
    for cat in MCC_POOL:
        mask = mcc == cat
        if mask.any():
            amount[mask] = _amount_for(rng, cat, int(mask.sum()))

    channel = rng.choice(["pos", "ecom", "atm", "moto"], n, p=[0.55, 0.36, 0.07, 0.02])
    entry_mode = np.where(
        channel == "ecom",
        rng.choice(["token", "keyed"], n, p=[0.7, 0.3]),
        rng.choice(["chip", "contactless", "swipe"], n, p=[0.55, 0.4, 0.05]),
    )

    # Each card has a home device; a minority of traffic comes from elsewhere.
    card_device = np.array([f"dev_{i:07d}" for i in range(n_cards)])
    device_id = np.where(rng.random(n) < 0.88, card_device[card_idx],
                         np.array([f"dev_{i:07d}" for i in rng.integers(0, n_cards * 2, n)]))
    ip_prefix = np.array([f"{a}.{b}.0.0" for a, b in
                          zip(rng.integers(1, 224, n), rng.integers(0, 256, n), strict=True)])

    df = pd.DataFrame({
        "transaction_id": [f"txn_{i:09d}" for i in range(n)],
        "event_time": ts,
        "card_id": card_ids[card_idx],
        "customer_id": np.char.replace(card_ids[card_idx].astype(str), "card_", "cust_"),
        "merchant_id": merchant_ids[mch_idx],
        "merchant_category": mcc,
        "merchant_country": merchant_country[mch_idx],
        "amount": amount,
        "currency": "USD",
        "channel": channel,
        "entry_mode": entry_mode,
        "device_id": device_id,
        "ip_prefix": ip_prefix,
        "issuer_country": "US",
        "customer_age_band": rng.choice(["18-25", "26-40", "41-60", "60+"], n,
                                        p=[0.18, 0.37, 0.31, 0.14]),
        "is_fraud": 0,
    })

    df = df.sort_values("event_time", ignore_index=True)
    _inject_fraud(df, rng, fraud_rate)
    # Episodes rewrite timestamps to compress a burst into minutes, so re-sort.
    return df.sort_values("event_time", ignore_index=True)[RAW_COLUMNS]


def _inject_fraud(df: pd.DataFrame, rng: np.random.Generator, fraud_rate: float) -> None:
    """Stamp fraud patterns in place. Mutates ``df``.

    Fraud is injected as *episodes* on a card rather than as independent rows.
    That is how it actually arrives - a compromised card is worked hard over
    minutes or hours until it is blocked - and it is the only way the velocity
    and sequence features have anything to detect. Injecting isolated rows
    produces a dataset where single-transaction features are the only signal,
    which flatters a plain classifier and makes the rest of the stack look
    worthless for reasons that are an artifact of the generator.
    """
    n = len(df)
    target = int(n * fraud_rate)

    label = df["is_fraud"].to_numpy(copy=True)
    amount = df["amount"].to_numpy(copy=True)
    channel = df["channel"].to_numpy(copy=True)
    entry = df["entry_mode"].to_numpy(copy=True)
    country = df["merchant_country"].to_numpy(copy=True)
    device = df["device_id"].to_numpy(copy=True)
    merchant = df["merchant_id"].to_numpy(copy=True)
    mcc = df["merchant_category"].to_numpy(copy=True)
    times = df["event_time"].to_numpy(copy=True)

    rows_by_card: dict[str, np.ndarray] = {
        card: np.asarray(idx, dtype=np.int64)
        for card, idx in df.groupby("card_id", sort=False).indices.items()
    }
    eligible = [c for c, idx in rows_by_card.items() if len(idx) >= 6]
    rng.shuffle(eligible)

    def pick_episode(idx: np.ndarray, k: int) -> tuple[np.ndarray, np.datetime64]:
        """Choose ``k`` consecutive rows of a card, plus the episode's anchor time.

        The anchor is drawn uniformly from *all* of the card's transactions, not
        from the chosen slice. Deriving it from the slice instead is a subtle
        trap: a slice of k consecutive rows can never be centred on a card's
        first or last transaction, so anchoring on it pulls every episode toward
        the middle of the timeline. The result is a dataset whose test window
        carries roughly half the true fraud prevalence - measured at 0.43%
        against an overall 0.85% before this was fixed - which quietly
        misstates every held-out metric.
        """
        anchor = int(rng.integers(0, len(idx)))
        start = int(np.clip(anchor - k // 2, 0, max(0, len(idx) - k)))
        return idx[start:start + k], times[idx[anchor]]

    def compress(hit: np.ndarray, anchor_time, minutes_lo: float, minutes_hi: float) -> None:
        """Pull an episode into a tight window centred on ``anchor_time``."""
        gaps = np.cumsum(rng.uniform(minutes_lo, minutes_hi, len(hit)))
        gaps -= gaps.mean()
        times[hit] = anchor_time + (gaps * 60_000_000_000).astype("timedelta64[ns]")

    # --- 1. merchant compromise (~20% of target) -------------------------
    # A handful of merchants leak card data. Concentrating fraud on specific
    # merchants is what gives merchant_risk_score real, persistent signal.
    risk = pd.Series(mcc).map(MCC_RISK).to_numpy(dtype=float)
    merchant_ids = df["merchant_id"].unique()
    n_bad = max(2, len(merchant_ids) // 60)
    mch_risk = pd.DataFrame({"m": merchant, "r": risk}).groupby("m")["r"].mean()
    bad_merchants = rng.choice(
        mch_risk.index.to_numpy(), n_bad, replace=False,
        p=(mch_risk.to_numpy() / mch_risk.to_numpy().sum()),
    )
    compromised = np.isin(merchant, bad_merchants)
    pool = np.flatnonzero(compromised)
    take = min(len(pool), int(target * 0.20))
    if take:
        hit = rng.choice(pool, take, replace=False)
        label[hit] = 1
        amount[hit] = np.round(amount[hit] * rng.uniform(1.4, 4.0, take), 2).clip(max=25_000)

    # --- 2. card testing (~20%): rapid tiny card-not-present auths --------
    placed, ci = int(label.sum()), 0
    while placed < target * 0.40 and ci < len(eligible):
        idx = rows_by_card[eligible[ci]]
        ci += 1
        hit, anchor_time = pick_episode(idx, int(rng.integers(4, 8)))
        label[hit] = 1
        amount[hit] = np.round(rng.uniform(0.9, 14.0, len(hit)), 2)
        channel[hit] = "ecom"
        entry[hit] = "keyed"
        # All probes land on one throwaway merchant, minutes apart.
        merchant[hit] = rng.choice(merchant_ids)
        mcc[hit] = "digital_goods"
        compress(hit, anchor_time, 1.0, 6.0)
        placed = int(label.sum())

    # --- 3. account takeover (~30%): new device, escalating amounts -------
    while placed < target * 0.70 and ci < len(eligible):
        idx = rows_by_card[eligible[ci]]
        ci += 1
        hit, anchor_time = pick_episode(idx, int(rng.integers(2, 5)))
        label[hit] = 1
        # Escalating: attackers test small, then drain.
        ramp = np.linspace(2.0, 12.0, len(hit)) * rng.uniform(0.7, 1.4)
        amount[hit] = np.round(amount[hit] * ramp, 2).clip(max=25_000)
        channel[hit] = "ecom"
        entry[hit] = "keyed"
        device[hit] = f"dev_ato_{ci:07d}"
        compress(hit, anchor_time, 8.0, 90.0)
        placed = int(label.sum())

    # --- 4. counterfeit (~30%): magstripe cloning abroad ------------------
    while placed < target and ci < len(eligible):
        idx = rows_by_card[eligible[ci]]
        ci += 1
        hit, anchor_time = pick_episode(idx, int(rng.integers(2, 5)))
        label[hit] = 1
        entry[hit] = "swipe"
        channel[hit] = "pos"
        country[hit] = rng.choice(["NG", "RO", "BR", "MX"])
        amount[hit] = np.round(amount[hit] * rng.uniform(1.8, 6.0, len(hit)), 2).clip(max=25_000)
        compress(hit, anchor_time, 20.0, 180.0)
        placed = int(label.sum())

    df["is_fraud"] = label.astype(int)
    df["amount"] = amount
    df["channel"] = channel
    df["entry_mode"] = entry
    df["merchant_country"] = country
    df["device_id"] = device
    df["merchant_id"] = merchant
    df["merchant_category"] = mcc
    df["event_time"] = times


def time_split(
    df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/valid/test split.

    Fraud models must never be split randomly: a random split leaks future
    merchant behaviour and card velocity into training and inflates every
    metric you care about.
    """
    df = df.sort_values("event_time", ignore_index=True)
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + valid_frac))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()
