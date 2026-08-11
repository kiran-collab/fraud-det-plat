"""Unsupervised outlier scorer (Isolation Forest).

Purpose is coverage, not accuracy. The supervised model can only recognise
fraud patterns that appeared in labelled history, and label feedback runs
30-90 days behind a new attack. The forest is fit on *legitimate traffic only*
so novel behaviour scores as anomalous on day one, before a single chargeback
has been filed.

Raw ``score_samples`` output is unbounded and not comparable across refits, so
it is min-max calibrated against the training distribution and clipped to
[0, 1] before it enters the ensemble.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class IsolationForestScorer:
    n_estimators: int = 300
    max_samples: int | str = 8192
    contamination: float = 0.01
    model: object | None = None
    _lo: float = 0.0
    _hi: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray | None = None, seed: int = 17) -> IsolationForestScorer:
        from sklearn.ensemble import IsolationForest

        # Fit on the negative class only when labels are available.
        x_fit = x[y == 0] if y is not None else x
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(x_fit)) if isinstance(self.max_samples, int) else self.max_samples,
            contamination=self.contamination,
            random_state=seed,
            n_jobs=-1,
        ).fit(x_fit)

        raw = -self.model.score_samples(x_fit)  # higher = more anomalous
        # 1st/99th percentile rather than min/max: a single extreme training
        # point would otherwise squash the entire usable range.
        self._lo, self._hi = float(np.percentile(raw, 1)), float(np.percentile(raw, 99))
        if self._hi <= self._lo:
            self._hi = self._lo + 1e-6
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsolationForestScorer.fit() must be called first")
        return self.calibrate_raw(-self.model.score_samples(x))

    def calibrate_raw(self, raw: np.ndarray) -> np.ndarray:
        """Map raw anomaly values onto [0, 1].

        Public so the ONNX path can reuse the exact same calibration constants
        rather than re-deriving them - the graph exports the tree ensemble, not
        the percentile bounds fitted around it.
        """
        return np.clip((np.asarray(raw) - self._lo) / (self._hi - self._lo), 0.0, 1.0)

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        path.with_suffix(".calib.json").write_text(json.dumps({"lo": self._lo, "hi": self._hi}))

    @classmethod
    def load(cls, path: Path) -> IsolationForestScorer:
        import joblib

        obj = cls()
        obj.model = joblib.load(path)
        calib = json.loads(path.with_suffix(".calib.json").read_text())
        obj._lo, obj._hi = float(calib["lo"]), float(calib["hi"])
        return obj
