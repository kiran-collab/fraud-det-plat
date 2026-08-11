"""Supervised fraud classifier (LightGBM).

This is the workhorse: it carries ~65% of the ensemble weight and produces the
SHAP values the investigation assistant and the model-governance pack consume.

Two choices worth calling out:

* ``scale_pos_weight`` rather than resampling. At ~0.85% prevalence, SMOTE-style
  oversampling manufactures cardholder behaviour that never happened and the
  velocity features go out of distribution. Re-weighting keeps the empirical
  joint intact.
* Early stopping on validation **PR-AUC**, not ROC-AUC. At this prevalence
  ROC-AUC is dominated by the negative class and will happily plateau while
  precision in the alerting band collapses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fraudplat.features.transforms import FEATURE_NAMES

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 120,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbosity": -1,
    "num_threads": 0,
}


@dataclass
class SupervisedModel:
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 800
    early_stopping_rounds: int = 60
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    booster: Any = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
        seed: int = 17,
    ) -> SupervisedModel:
        import lightgbm as lgb

        params = dict(self.params)
        params["seed"] = seed
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        # Re-weight rather than resample; see module docstring.
        params["scale_pos_weight"] = max(neg / max(pos, 1.0), 1.0)

        train_set = lgb.Dataset(x_train, label=y_train, feature_name=self.feature_names)
        valid_sets, callbacks = [], []
        if x_valid is not None and y_valid is not None:
            valid_sets = [lgb.Dataset(x_valid, label=y_valid, feature_name=self.feature_names)]
            callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]

        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets or None,
            callbacks=callbacks or None,
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("SupervisedModel.fit() must be called before predict_proba()")
        return np.asarray(self.booster.predict(x), dtype=np.float64).ravel()

    def feature_importance(self) -> dict[str, float]:
        if self.booster is None:
            return {}
        gains = self.booster.feature_importance(importance_type="gain")
        total = float(gains.sum()) or 1.0
        return {
            name: float(g) / total
            for name, g in sorted(zip(self.feature_names, gains, strict=True), key=lambda kv: -kv[1])
        }

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        if self.booster is None:
            raise RuntimeError("nothing to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> SupervisedModel:
        import lightgbm as lgb

        model = cls()
        model.booster = lgb.Booster(model_file=str(path))
        return model
