"""SHAP explanations for the supervised head.

Two consumers, two very different latency budgets:

* the **authorization path** needs a reason code inside the same sub-50ms
  budget as the score. ``TreeExplainer`` on a single row is ~1-3ms, which fits;
  a KernelExplainer would not, which is why the supervised model is the one
  that carries the explanation duty rather than the ensemble as a whole.
* **model governance** needs global attributions over a sample, refreshed each
  retrain and filed with the model documentation.

SHAP is used rather than raw LightGBM gain because gain is a property of the
*model*, while a chargeback dispute or an adverse-action notice is about a
specific transaction. Additivity also means the contributions reconcile to the
score, which is the property an auditor checks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from fraudplat.features.transforms import FEATURE_NAMES

# SHAP emits this on *every* call for LightGBM binary models, announcing that
# the return type is now a list of arrays. That is the shape handled below, so
# the notice carries no information - but left unfiltered it writes a log line
# per explained transaction, which at decline volume is millions of lines a day
# and drowns out the entries on-call actually needs.
warnings.filterwarnings(
    "ignore",
    message=".*LightGBM binary classifier with TreeExplainer.*",
    category=UserWarning,
)


@dataclass
class ShapExplainer:
    booster: Any
    feature_names: list[str]
    _explainer: Any = None

    @classmethod
    def from_model(cls, supervised: Any) -> ShapExplainer:
        import shap

        obj = cls(booster=supervised.booster, feature_names=list(FEATURE_NAMES))
        obj._explainer = shap.TreeExplainer(supervised.booster)
        return obj

    def explain_row(self, x: np.ndarray) -> dict[str, float]:
        """Per-feature log-odds contributions for one transaction."""
        row = np.asarray(x, dtype=np.float64).reshape(1, -1)
        values = self._explainer.shap_values(row)
        # LightGBM binary returns either (1, n_features) or a 2-element list.
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values).reshape(-1)[: len(self.feature_names)]
        return {name: float(v) for name, v in zip(self.feature_names, values, strict=False)}

    def global_importance(self, x: np.ndarray, sample: int = 5000, seed: int = 17) -> dict[str, float]:
        """Mean |SHAP| over a sample - the governance-pack view."""
        x = np.asarray(x, dtype=np.float64)
        if len(x) > sample:
            rng = np.random.default_rng(seed)
            x = x[rng.choice(len(x), sample, replace=False)]
        values = self._explainer.shap_values(x)
        if isinstance(values, list):
            values = values[-1]
        mean_abs = np.abs(np.asarray(values)).mean(axis=0)[: len(self.feature_names)]
        total = float(mean_abs.sum()) or 1.0
        return dict(
            sorted(
                ((n, float(v) / total) for n, v in zip(self.feature_names, mean_abs, strict=False)),
                key=lambda kv: -kv[1],
            )
        )
