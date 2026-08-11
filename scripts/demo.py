#!/usr/bin/env python
"""End-to-end walkthrough of the platform on synthetic traffic.

    python scripts/demo.py

Runs the whole path without any infrastructure: streams transactions through
the feature writer into the online store, scores them, shows the decisions and
reason codes for a set of hand-built attack scenarios, and (if an API key is
configured) drafts an investigation summary for one of them.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.features.store.online import OnlineFeatureStore  # noqa: E402
from fraudplat.features.streaming.consumer import FeatureStreamWriter  # noqa: E402
from fraudplat.models.registry import ModelRegistry  # noqa: E402
from fraudplat.serving.scorer import TransactionScorer  # noqa: E402

BASE = datetime(2025, 3, 1, 14, 0, tzinfo=UTC)


def txn(offset_min: float, **overrides) -> dict:
    base = {
        "transaction_id": f"demo_{int(offset_min * 60):06d}",
        "event_time": BASE + timedelta(minutes=offset_min),
        "card_id": "card_demo_001",
        "customer_id": "cust_demo_001",
        "merchant_id": "mch_000042",
        "merchant_category": "grocery",
        "merchant_country": "US",
        "amount": 54.20,
        "currency": "USD",
        "channel": "pos",
        "entry_mode": "chip",
        "device_id": "dev_demo_001",
        "issuer_country": "US",
    }
    return {**base, **overrides}


def show(title: str, result) -> None:
    print(f"\n  {title}")
    print(
        f"    -> {result.action.upper():<9} risk={result.risk_score:.4f}  "
        f"(supervised={result.supervised_score:.3f} anomaly={result.anomaly_score:.3f} "
        f"sequence={result.sequence_score:.3f})  {result.latency_ms:.1f}ms"
    )
    if result.triggered_rules:
        print(f"       rules:   {', '.join(result.triggered_rules)}")
    if result.reasons:
        for r in result.reasons[:3]:
            print(f"       reason:  {r}")


def main() -> int:
    try:
        bundle = ModelRegistry().load()
    except FileNotFoundError:
        print("No trained model found. Run:  python scripts/train.py --promote")
        return 1

    store = OnlineFeatureStore()
    print(f"model {bundle.version}   online store: {store.backend}")

    # --- 1. establish a normal spending history for the card --------------
    print("\n[1] Streaming 20 ordinary transactions through the feature writer")
    writer = FeatureStreamWriter(bundle.merchant_profile, store)
    history = [
        txn(-60 * 24 * (20 - i), amount=30 + (i * 7) % 60,
            merchant_id=f"mch_{i % 5:06d}", transaction_id=f"hist_{i:03d}")
        for i in range(20)
    ]
    writer.run(iter(history), log_every=0)
    print(f"    wrote {writer.stats.messages} events; card state now in the {store.backend} store")

    scorer = TransactionScorer(bundle=bundle, store=store)
    print(f"    scoring backend: {scorer.inference_backend}")

    # --- 2. score a range of scenarios ------------------------------------
    print("\n[2] Scoring scenarios against that history")

    show("Normal grocery purchase, familiar merchant and device",
         scorer.score(txn(0), explain=True, commit=False))

    show("Same amount, but a merchant category the card never uses",
         scorer.score(txn(1, merchant_category="gambling", merchant_id="mch_000777",
                          channel="ecom", entry_mode="keyed"),
                      explain=True, commit=False))

    show("High-value purchase from a device never seen on this card",
         scorer.score(txn(2, amount=3800.0, device_id="dev_unknown_9",
                          channel="ecom", entry_mode="keyed"),
                      explain=True, commit=False))

    show("Magnetic-stripe swipe in a country the card has never visited",
         scorer.score(txn(3, amount=610.0, merchant_country="RO",
                          entry_mode="swipe", channel="pos"),
                      explain=True, commit=False))

    # Card-testing needs real velocity, so these must be committed.
    print("\n  Card-testing burst (six tiny card-not-present auths, ~2 min apart)")
    for i in range(6):
        result = scorer.score(
            txn(10 + i * 2, amount=2.50 + i, merchant_id="mch_000999",
                merchant_category="digital_goods", channel="ecom", entry_mode="keyed",
                transaction_id=f"burst_{i}"),
            explain=(i == 5), commit=True,
        )
        print(f"    #{i + 1}  {result.action:<9} risk={result.risk_score:.4f}"
              f"  1h_count={result.features['card_txn_count_1h']:.0f}"
              + (f"   rules: {', '.join(result.triggered_rules)}" if result.triggered_rules else ""))

    # --- 3. investigation assistant ---------------------------------------
    print("\n[3] Investigation assistant (RAG over historical cases)")
    from fraudplat.genai.rag import InvestigationAssistant, InvestigationRequest
    from fraudplat.genai.vectorstore import build_index

    index = build_index()
    if len(getattr(index, "_docs", [])) == 0:
        print("    case index is empty - run: python scripts/build_case_index.py")
        return 0

    suspicious = txn(30, amount=3800.0, device_id="dev_unknown_9",
                     channel="ecom", entry_mode="keyed", merchant_category="electronics")
    scored = scorer.score(suspicious, explain=True, commit=False)
    request = InvestigationRequest(
        case_id="CASE-DEMO-001",
        transaction=suspicious,
        risk_score=scored.risk_score,
        decision=scored.action,
        reason_codes=scored.reasons,
        feature_snapshot=scored.features,
    )

    assistant = InvestigationAssistant(index=index)
    response = assistant.investigate(request)
    print(f"    retrieved {len(response.retrieved_case_ids)} comparable cases "
          f"from the {response.vector_backend} index")
    if response.blocked:
        print(f"    assistant unavailable: {response.summary}")
        print("    (set ANTHROPIC_API_KEY and install langchain-anthropic for the narrative)")
    else:
        print(f"\n    Summary: {response.summary}")
        for factor in response.risk_factors[:4]:
            print(f"      - {factor}")
        print(f"    Confidence: {response.confidence}")
        for check in response.recommended_checks[:3]:
            print(f"      next: {check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
