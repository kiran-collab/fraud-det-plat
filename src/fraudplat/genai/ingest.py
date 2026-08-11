"""Build the case knowledge base the assistant retrieves from.

Cases are written as short structured narratives rather than raw feature rows.
Retrieval is over text similarity, so what gets indexed has to *read* like what
an analyst would ask about - "card-testing burst on a digital goods merchant"
retrieves well; a vector of 31 floats does not.

Every document is redacted before it is embedded. The index is long-lived, so
an unredacted PAN in it is a durable data-retention problem regardless of what
the model later does with it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import pandas as pd

from fraudplat.genai.guardrails import redact
from fraudplat.genai.vectorstore import CaseDocument

log = logging.getLogger(__name__)

DISPOSITIONS = {1: "confirmed_fraud", 0: "legitimate"}


def case_narrative(txn: dict[str, Any], features: dict[str, float] | None = None) -> str:
    """Render one transaction as the kind of text an analyst would write."""
    features = features or {}
    lines = [
        f"Transaction of {txn.get('currency', 'USD')} {float(txn.get('amount', 0)):,.2f} "
        f"at a {txn.get('merchant_category', 'unknown')} merchant "
        f"in {txn.get('merchant_country', 'unknown')}.",
        f"Channel {txn.get('channel')}, entry mode {txn.get('entry_mode')}.",
    ]

    signals: list[str] = []
    if features.get("is_cross_border"):
        signals.append("cross-border")
    if features.get("entry_is_swipe"):
        signals.append("magnetic stripe entry on a chip-capable card")
    if features.get("is_new_device_for_card"):
        signals.append("device not previously seen on this card")
    if features.get("is_new_country_for_card"):
        signals.append("country not previously seen on this card")
    if features.get("card_txn_count_1h", 0) >= 4:
        signals.append(f"{features['card_txn_count_1h']:.0f} transactions in the preceding hour")
    if features.get("amount_ratio_card_mean", 1) >= 4:
        signals.append(
            f"amount {features['amount_ratio_card_mean']:.1f}x the card's average"
        )
    if features.get("merchant_risk_score", 0) > 0.02:
        signals.append("merchant with an elevated historical fraud rate")
    if features.get("is_night"):
        signals.append("outside the cardholder's usual hours")

    if signals:
        lines.append("Observed signals: " + "; ".join(signals) + ".")

    label = txn.get("is_fraud")
    if label is not None:
        lines.append(f"Disposition: {DISPOSITIONS.get(int(label), 'unknown')}.")
    return " ".join(lines)


def build_documents(
    df: pd.DataFrame,
    features: pd.DataFrame | None = None,
    max_cases: int = 4000,
    fraud_share: float = 0.5,
    seed: int = 17,
) -> list[CaseDocument]:
    """Sample a balanced case library.

    Deliberately over-sampling fraud relative to its 0.85% natural rate. The
    index is a retrieval corpus, not a training set: at natural prevalence a
    similarity search returns almost exclusively legitimate cases and tells an
    analyst nothing about what a confirmed-fraud case of this shape looked
    like. Balance is what makes retrieval useful; it does not bias any model.
    """
    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).to_numpy()
    df = df.iloc[rng].reset_index(drop=True)
    if features is not None:
        features = features.iloc[rng].reset_index(drop=True)

    fraud_idx = df.index[df["is_fraud"] == 1][: int(max_cases * fraud_share)]
    legit_idx = df.index[df["is_fraud"] == 0][: max_cases - len(fraud_idx)]
    keep = sorted(list(fraud_idx) + list(legit_idx))

    docs: list[CaseDocument] = []
    for i in keep:
        txn = df.iloc[i].to_dict()
        feat = features.iloc[i].to_dict() if features is not None else {}
        text, _ = redact(case_narrative(txn, feat))
        docs.append(CaseDocument(
            case_id=str(txn["transaction_id"]),
            text=text,
            metadata={
                "disposition": DISPOSITIONS.get(int(txn.get("is_fraud", 0)), "unknown"),
                "merchant_category": str(txn.get("merchant_category")),
                "channel": str(txn.get("channel")),
                "merchant_country": str(txn.get("merchant_country")),
                "amount": float(txn.get("amount", 0)),
            },
        ))
    log.info("built %s case documents (%s fraud)", len(docs), len(fraud_idx))
    return docs


def ingest(index, docs: Iterable[CaseDocument], batch_size: int = 500) -> int:
    docs = list(docs)
    total = 0
    for i in range(0, len(docs), batch_size):
        total += index.upsert(docs[i:i + batch_size])
    return total
