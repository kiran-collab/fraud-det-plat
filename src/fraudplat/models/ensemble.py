"""Score blending.

Three scorers answer three different questions:

  supervised  - does this look like fraud we have seen and labelled?
  iforest     - does this look unlike normal traffic at all?
  sequence    - does this follow from what this card just did?

They are blended rather than stacked, because a stacker trained on labelled
history inherits exactly the blind spot the unsupervised components exist to
cover. Fixed, auditable coefficients also survive a model-risk review far more
easily than a second learned layer.

**Blending happens in logit space, not probability space.** This is the
important design decision. A linear blend - ``0.65*p + 0.15*iforest +
0.20*sequence`` - is the obvious approach and it is wrong here. The calibrated
supervised probability sits near zero for the 99.15% of traffic that is
legitimate, while the anomaly scores are spread across [0, 1] by construction.
Adding them linearly puts a floor under every legitimate transaction that any
detector finds mildly unusual, which wipes out precision in exactly the
high-threshold band where the decline decision lives. Measured on the test
window, linear blending dropped precision@0.86 from 1.00 to 0.00 while the
supervised model alone was fine.

In logit space the unsupervised scores act as bounded *evidence adjustments*
around their own median. A transaction that is unremarkable to both detectors
gets no adjustment and keeps the supervised model's calibrated probability
untouched; a genuinely strange one gets pushed up by a bounded number of
log-odds. Calibration at the low end survives, and the detectors can still
escalate novel attacks the supervised model has never seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fraudplat.config import EnsembleConfig

_EPS = 1e-6
# Cap on how far the unsupervised detectors may move the supervised model, in
# log-odds. Roughly one order of magnitude in either direction: enough for an
# anomaly detector to lift an unlabelled attack into the review queue, not
# enough for it to overturn a confident supervised judgement on its own.
MAX_LOGIT_ADJUSTMENT = 2.5


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class RiskScore:
    score: float
    supervised: float
    isolation_forest: float
    sequence: float

    def as_dict(self) -> dict[str, float]:
        return {
            "risk_score": round(self.score, 6),
            "supervised_score": round(self.supervised, 6),
            "anomaly_score": round(self.isolation_forest, 6),
            "sequence_score": round(self.sequence, 6),
        }


@dataclass
class EnsembleScorer:
    weights: EnsembleConfig = field(default_factory=EnsembleConfig)
    calibrator: object | None = None
    # Reference points: the median unsupervised score on legitimate validation
    # traffic. Anything at the reference contributes exactly zero.
    ref_iforest: float = 0.5
    ref_sequence: float = 0.5

    # -- calibration -----------------------------------------------------
    def fit(
        self,
        p_supervised: np.ndarray,
        y: np.ndarray,
        s_iforest: np.ndarray | None = None,
        s_sequence: np.ndarray | None = None,
    ) -> EnsembleScorer:
        """Fit on a held-out validation split.

        Isotonic rather than Platt for the supervised head: raw LightGBM
        probabilities under ``scale_pos_weight`` are badly miscalibrated in a
        way that is monotone but not sigmoid-shaped, and isotonic recovers the
        true rate without assuming a functional form.
        """
        from sklearn.isotonic import IsotonicRegression

        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(p_supervised, y)

        legit = y == 0
        if s_iforest is not None and legit.any():
            self.ref_iforest = float(np.median(np.asarray(s_iforest)[legit]))
        if s_sequence is not None and legit.any():
            self.ref_sequence = float(np.median(np.asarray(s_sequence)[legit]))
        return self

    # Kept as an alias: the training script and older pipelines call this name.
    fit_calibration = fit

    def calibrate(self, p: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return np.asarray(p, dtype=np.float64)
        return np.asarray(self.calibrator.predict(p), dtype=np.float64)

    # -- blending --------------------------------------------------------
    def blend(
        self,
        p_supervised: np.ndarray,
        s_iforest: np.ndarray,
        s_sequence: np.ndarray,
    ) -> np.ndarray:
        base = _logit(self.calibrate(np.asarray(p_supervised)))
        adjustment = (
            self.weights.iforest_logit_weight * (np.asarray(s_iforest) - self.ref_iforest)
            + self.weights.sequence_logit_weight * (np.asarray(s_sequence) - self.ref_sequence)
        )
        adjustment = np.clip(adjustment, -MAX_LOGIT_ADJUSTMENT, MAX_LOGIT_ADJUSTMENT)
        return _sigmoid(base + adjustment)

    def score_one(self, p_supervised: float, s_iforest: float, s_sequence: float) -> RiskScore:
        blended = float(
            self.blend(np.array([p_supervised]), np.array([s_iforest]), np.array([s_sequence]))[0]
        )
        return RiskScore(
            score=blended,
            supervised=float(self.calibrate(np.array([p_supervised]))[0]),
            isolation_forest=float(s_iforest),
            sequence=float(s_sequence),
        )

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.calibrator, path)
        path.with_suffix(".weights.json").write_text(json.dumps({
            "iforest_logit_weight": self.weights.iforest_logit_weight,
            "sequence_logit_weight": self.weights.sequence_logit_weight,
            "ref_iforest": self.ref_iforest,
            "ref_sequence": self.ref_sequence,
        }))

    @classmethod
    def load(cls, path: Path) -> EnsembleScorer:
        import joblib

        w = json.loads(path.with_suffix(".weights.json").read_text())
        obj = cls(
            weights=EnsembleConfig(
                iforest_logit_weight=w["iforest_logit_weight"],
                sequence_logit_weight=w["sequence_logit_weight"],
            ),
            ref_iforest=float(w.get("ref_iforest", 0.5)),
            ref_sequence=float(w.get("ref_sequence", 0.5)),
        )
        obj.calibrator = joblib.load(path)
        return obj
