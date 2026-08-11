"""Decision audit log.

Every automated decline must be reconstructable years later: what the score
was, which model version produced it, which features it saw, and what reason
codes the customer was given. Chargeback disputes, regulatory examinations and
model-validation reviews all ask the same question - "why did you decline this
transaction on this date" - and the only defensible answer is a record written
at decision time.

Two properties that make it an audit log rather than application logging:

* **Append-only, hash-chained.** Each record carries the hash of its
  predecessor, so a deleted or edited row breaks the chain and
  ``verify_chain`` finds it. This does not prevent tampering; it makes silent
  tampering detectable, which is the achievable goal.
* **No cardholder data.** Card and customer identifiers are stored as salted
  hashes. The log answers "why was this decision made" without becoming a
  second copy of the card master under a weaker retention policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _salt() -> str:
    # In EKS this is injected from Secrets Manager. The default exists so the
    # platform runs locally; it is not a security boundary and says so.
    return os.environ.get("FP_AUDIT_SALT", "local-development-salt")


def pseudonymize(value: str) -> str:
    """Salted hash of an identifier. Stable across records so a card's history
    can be reconstructed, but not reversible to a PAN without the salt."""
    return hashlib.sha256(f"{_salt()}:{value}".encode()).hexdigest()[:32]


@dataclass
class AuditRecord:
    timestamp: str
    transaction_id: str
    card_hash: str
    model_version: str
    action: str
    risk_score: float
    supervised_score: float
    anomaly_score: float
    sequence_score: float
    reason_codes: list[str] = field(default_factory=list)
    triggered_rules: list[str] = field(default_factory=list)
    feature_snapshot: dict[str, float] = field(default_factory=dict)
    prev_hash: str = GENESIS
    record_hash: str = ""

    def compute_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "record_hash"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()


class AuditLog:
    """JSONL-backed hash chain.

    JSONL in production is shipped to S3 with object-lock and queried through
    Athena; the local file is the same format so nothing about the record
    schema differs between environments.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = None
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    last = line
        return json.loads(last)["record_hash"] if last else GENESIS

    def append(
        self,
        transaction_id: str,
        card_id: str,
        model_version: str,
        result: Any,
        include_features: bool = True,
    ) -> AuditRecord:
        """Write one decision. Called for every scored transaction.

        ``include_features`` is on by default: reconstructing a decision
        without the feature values it was made from is not actually possible,
        since the card's state at that instant is gone.
        """
        with self._lock:
            record = AuditRecord(
                timestamp=datetime.now(UTC).isoformat(),
                transaction_id=transaction_id,
                card_hash=pseudonymize(card_id),
                model_version=model_version,
                action=result.action,
                risk_score=round(float(result.risk_score), 6),
                supervised_score=round(float(result.supervised_score), 6),
                anomaly_score=round(float(result.anomaly_score), 6),
                sequence_score=round(float(result.sequence_score), 6),
                reason_codes=list(result.reasons),
                triggered_rules=list(result.triggered_rules),
                feature_snapshot=(
                    {k: round(float(v), 6) for k, v in result.features.items()}
                    if include_features else {}
                ),
                prev_hash=self._last_hash,
            )
            record.record_hash = record.compute_hash()
            with self.path.open("a") as fh:
                fh.write(json.dumps(asdict(record), default=str) + "\n")
            self._last_hash = record.record_hash
            return record

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def verify_chain(self) -> tuple[bool, str | None]:
        """Walk the chain. Returns ``(ok, first_broken_transaction_id)``."""
        prev = GENESIS
        for row in self.read():
            if row["prev_hash"] != prev:
                return False, row["transaction_id"]
            record = AuditRecord(**{k: v for k, v in row.items() if k != "record_hash"})
            if record.compute_hash() != row["record_hash"]:
                return False, row["transaction_id"]
            prev = row["record_hash"]
        return True, None
