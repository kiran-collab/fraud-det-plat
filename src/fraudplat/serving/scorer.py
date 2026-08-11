"""The authorization scoring path.

Everything here runs inside the per-transaction latency budget, so the ordering
of work is chosen to keep the common case short:

  1. read the card's state from the online store        (~1-2ms, one GET)
  2. build the feature vector                            (~0.05ms, pure Python)
  3. score supervised + isolation forest + sequence      (~1-3ms via ONNX)
  4. blend, apply overlay rules                          (negligible)
  5. explain - only when asked, or when declining        (~1-3ms)
  6. write the card state back                           (~1-2ms, one SETEX)

Step 5 is conditional because most transactions are approved and nobody reads
the reason codes for an approval. Declines always get them: an adverse action
without a recorded reason is a compliance problem, not just a UX gap.

Step 6 is skipped for ``commit=false`` quotes so a speculative score cannot
pollute a cardholder's behavioural baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from fraudplat.config import SETTINGS
from fraudplat.features.store.online import OnlineFeatureStore
from fraudplat.features.transforms import FEATURE_NAMES, FeatureEngine
from fraudplat.models.registry import ModelBundle, ModelRegistry
from fraudplat.serving.decisioning import Action, decide, top_reasons


@dataclass
class ScoringResult:
    action: str
    risk_score: float
    supervised_score: float
    anomaly_score: float
    sequence_score: float
    reasons: list[str]
    triggered_rules: list[str]
    features: dict[str, float]
    latency_ms: float


class TransactionScorer:
    def __init__(
        self,
        bundle: ModelBundle | None = None,
        store: OnlineFeatureStore | None = None,
        use_onnx: bool = True,
    ) -> None:
        self.registry = ModelRegistry()
        self.bundle = bundle or self.registry.load()
        self.store = store or OnlineFeatureStore()
        self.engine = FeatureEngine(self.bundle.merchant_profile)
        self.decision_cfg = SETTINGS.decision
        self._explainer = None
        self._onnx = None
        self.inference_backend = "python"
        if use_onnx:
            self._try_load_onnx()

    # -- setup -----------------------------------------------------------
    def _try_load_onnx(self) -> None:
        """Prefer the ONNX graphs when a build exists, fall back silently.

        Falling back rather than failing is intentional: an ONNX build is a
        performance optimisation, and a pod that cannot find one should still
        serve correct scores slightly slower rather than refuse traffic.
        """
        from fraudplat.inference.runtime import OnnxScorer

        d = Path(self.registry.current_dir()) / "onnx"
        sup = d / "supervised.onnx"
        if not sup.exists():
            return
        try:
            seq = d / "sequence.int8.onnx"
            self._onnx = OnnxScorer(
                sup,
                seq if seq.exists() else d / "sequence.onnx",
                d / "iforest.onnx",
            )
            self.inference_backend = "onnxruntime"
        except Exception:
            self._onnx = None

    @property
    def explainer(self):
        # Built lazily: TreeExplainer construction is slow and most pods in a
        # canary never receive an explain request.
        if self._explainer is None:
            from fraudplat.explain.shap_explainer import ShapExplainer

            self._explainer = ShapExplainer.from_model(self.bundle.supervised)
        return self._explainer

    @property
    def model_version(self) -> str:
        return self.bundle.version

    # -- scoring ---------------------------------------------------------
    def score(self, txn: dict[str, Any], explain: bool = False, commit: bool = True) -> ScoringResult:
        t0 = time.perf_counter()
        txn = dict(txn)
        # `event_time` is optional in the request schema, so it arrives either
        # absent *or* explicitly null. setdefault only covers the first case;
        # the second reached _epoch as None and raised on NaT.
        if txn.get("event_time") is None:
            txn["event_time"] = datetime.now(UTC)
        card_id = str(txn["card_id"])

        # 1. hydrate
        self.engine.hydrate_card(card_id, self.store.get_card_state(card_id))

        # 2. features - computed against pre-transaction state
        window = self.engine.sequence_window(card_id, self.bundle.sequence.seq_len)
        features = self.engine.transform(txn, update=commit)
        vector = np.array([features[n] for n in FEATURE_NAMES], dtype=np.float32)

        # 3. score
        if self._onnx is not None:
            p_sup = float(self._onnx.predict_proba(vector[None, :])[0])
        else:
            p_sup = float(self.bundle.supervised.predict_proba(vector[None, :])[0])

        if self._onnx is not None and self._onnx.has_iforest:
            # Same trees, same calibration constants - just a faster graph.
            s_if = float(self.bundle.iforest.calibrate_raw(self._onnx.iforest_raw(vector))[0])
        else:
            s_if = float(self.bundle.iforest.score(vector[None, :])[0])
        s_seq = self._sequence_score(window, vector)

        # 4. blend + rules
        risk = self.bundle.ensemble.score_one(p_sup, s_if, s_seq)
        decision = decide(risk.score, txn, features, self.decision_cfg)

        # 5. explain when asked, and always on an adverse action
        reasons: list[str] = []
        if explain or decision.action is Action.DECLINE:
            reasons = top_reasons(self.explainer.explain_row(vector))
        decision.reasons = reasons

        # 6. persist
        if commit:
            self.store.put_card_state(card_id, self.engine.dump_card(card_id))
        self.engine.reset_card(card_id)

        return ScoringResult(
            action=decision.action.value,
            risk_score=risk.score,
            supervised_score=risk.supervised,
            anomaly_score=risk.isolation_forest,
            sequence_score=risk.sequence,
            reasons=reasons,
            triggered_rules=decision.triggered_rules,
            features=features,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )

    def _sequence_score(self, window: np.ndarray | None, vector: np.ndarray) -> float:
        """Neutral 0.5 when the card has no usable history.

        A cold card genuinely has no rhythm to deviate from. Scoring it as
        highly anomalous would penalise every new cardholder's first
        transactions, which is both bad detection and a fair-lending problem.
        """
        if window is None:
            return 0.5
        window = window.copy()
        window[-1] = vector
        return float(self.bundle.sequence.score_window(window))
