"""Kafka -> feature store writer.

This is the process that keeps the online store warm. It consumes the raw
authorization stream, advances each card's ``FeatureEngine`` state, and writes
the result back so the scoring API can read it in single-digit milliseconds.

Three properties that matter more than the code:

**Partition by card_id.** The engine's windows are per-card and order-dependent,
so all of a card's events must land on one partition and therefore one
consumer. Round-robin partitioning would interleave a card's events across
workers and corrupt every velocity feature. The producer enforces this by using
``card_id`` as the message key.

**Read-modify-write is safe only because of that partitioning.** There is no
lock around the get/update/put cycle. With one consumer per card that is fine;
if the topic is ever repartitioned mid-flight it is not, which is why the
consumer group is configured for a cooperative-sticky rebalance rather than the
default eager one.

**At-least-once, not exactly-once.** Offsets are committed after the store
write. A crash between write and commit replays the message and double-counts
one transaction in a velocity window - a bounded, self-healing error that
decays out of the window within 7 days. The alternative (commit first) drops
the event entirely and leaves a permanent hole, which is worse.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from fraudplat.config import SETTINGS, KafkaConfig
from fraudplat.features.store.online import OnlineFeatureStore
from fraudplat.features.transforms import FeatureEngine, MerchantProfile

log = logging.getLogger(__name__)


@dataclass
class ConsumerStats:
    messages: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def rate(self) -> float:
        elapsed = max(time.time() - self.started_at, 1e-6)
        return self.messages / elapsed


class FeatureStreamWriter:
    """Runs the consume -> transform -> persist loop."""

    def __init__(
        self,
        merchant_profile: MerchantProfile | None = None,
        store: OnlineFeatureStore | None = None,
        kafka: KafkaConfig | None = None,
    ) -> None:
        self.engine = FeatureEngine(merchant_profile)
        self.store = store or OnlineFeatureStore()
        self.kafka = kafka or SETTINGS.kafka
        self.stats = ConsumerStats()
        self._running = True

    # -- single message --------------------------------------------------
    def handle(self, txn: dict[str, Any]) -> dict[str, float]:
        """Advance one card's state and persist it. Returns the feature vector."""
        card_id = str(txn["card_id"])
        self.engine.hydrate_card(card_id, self.store.get_card_state(card_id))
        features = self.engine.transform(txn, update=True)
        self.store.put_card_state(card_id, self.engine.dump_card(card_id))
        # Drop the card from the local engine: this process handles millions of
        # cards and the online store is the source of truth, so holding state
        # in the worker is an unbounded memory leak with no benefit.
        self.engine.reset_card(card_id)
        self.stats.messages += 1
        return features

    # -- loop ------------------------------------------------------------
    def run(self, messages: Iterator[dict[str, Any]] | None = None, log_every: int = 10_000) -> ConsumerStats:
        """Consume until interrupted.

        ``messages`` lets tests and the local demo drive the same code path with
        an iterable instead of a broker.
        """
        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)

        source = messages if messages is not None else self._kafka_messages()
        for txn in source:
            if not self._running:
                break
            try:
                self.handle(txn)
            except Exception:
                self.stats.errors += 1
                # Never let one poisoned message stop the stream; the DLQ
                # decision belongs to the platform, not this loop.
                log.exception("failed to process transaction %s", txn.get("transaction_id"))
            if log_every and self.stats.messages % log_every == 0:
                log.info("processed %s messages (%.0f/s)", self.stats.messages, self.stats.rate)
        return self.stats

    def _stop(self, *_: Any) -> None:
        log.info("shutdown signal received; draining")
        self._running = False

    def _kafka_messages(self) -> Iterator[dict[str, Any]]:
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            self.kafka.transactions_topic,
            bootstrap_servers=self.kafka.bootstrap_servers.split(","),
            group_id=self.kafka.consumer_group,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            # Manual commit, after the store write - see module docstring.
            enable_auto_commit=False,
            auto_offset_reset="latest",
            partition_assignment_strategy=["cooperative-sticky"],
            max_poll_records=500,
        )
        try:
            for message in consumer:
                yield message.value
                consumer.commit()
        finally:
            consumer.close()
