"""Transaction producer.

Stands in for the card network feed. Used to replay a generated dataset onto
Kafka for load tests and the local demo.

``key=card_id`` is not cosmetic - it is what guarantees a card's events land on
one partition and are therefore processed in order by a single consumer. Every
velocity feature depends on it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from typing import Any

from fraudplat.config import SETTINGS, KafkaConfig

log = logging.getLogger(__name__)


class TransactionProducer:
    def __init__(self, kafka: KafkaConfig | None = None) -> None:
        self.kafka = kafka or SETTINGS.kafka
        self._producer = None

    def _connect(self):
        from kafka import KafkaProducer

        return KafkaProducer(
            bootstrap_servers=self.kafka.bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            linger_ms=5,          # small batching window; 5ms is well inside budget
            compression_type="lz4",
            acks=1,               # leader ack: durable enough for a replayable feed
        )

    def send(self, transactions: Iterable[dict[str, Any]], rate_per_sec: float | None = None) -> int:
        if self._producer is None:
            self._producer = self._connect()

        interval = 1.0 / rate_per_sec if rate_per_sec else 0.0
        count = 0
        for txn in transactions:
            self._producer.send(
                self.kafka.transactions_topic,
                key=txn["card_id"],  # partition by card - see module docstring
                value=txn,
            )
            count += 1
            if interval:
                time.sleep(interval)
        self._producer.flush()
        log.info("published %s transactions to %s", count, self.kafka.transactions_topic)
        return count

    def close(self) -> None:
        if self._producer is not None:
            self._producer.close()
            self._producer = None
