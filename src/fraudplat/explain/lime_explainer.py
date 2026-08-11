"""LIME explanations - the analyst-facing second opinion.

SHAP carries the production explanation path. LIME is here for a specific
reason rather than for completeness: it answers a *different* question. SHAP
attributes the score to features given the model's global structure; LIME fits
a sparse local surrogate and answers "what would have had to change, near this
transaction, for the decision to flip".

That local counterfactual framing is what an analyst actually wants when
working a queued alert, and having two methods that disagree is itself
diagnostic - a case where SHAP and LIME point at different features is usually
a case sitting on a sharp decision boundary and worth a closer look.

It is deliberately kept off the authorization path: LIME perturbs and re-scores
thousands of samples per explanation, which is hundreds of milliseconds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from fraudplat.features.transforms import FEATURE_NAMES


@dataclass
class LimeExplainer:
    explainer: Any
    predict_fn: Callable[[np.ndarray], np.ndarray]

    @classmethod
    def from_model(
        cls,
        supervised: Any,
        x_background: np.ndarray,
        seed: int = 17,
    ) -> LimeExplainer:
        from lime.lime_tabular import LimeTabularExplainer

        explainer = LimeTabularExplainer(
            training_data=np.asarray(x_background, dtype=np.float64),
            feature_names=list(FEATURE_NAMES),
            class_names=["legitimate", "fraud"],
            discretize_continuous=True,
            random_state=seed,
            mode="classification",
        )

        def predict(rows: np.ndarray) -> np.ndarray:
            p = np.asarray(supervised.predict_proba(np.asarray(rows, dtype=np.float32)))
            return np.column_stack([1.0 - p, p])

        return cls(explainer=explainer, predict_fn=predict)

    def explain_row(self, x: np.ndarray, num_features: int = 8) -> list[tuple[str, float]]:
        exp = self.explainer.explain_instance(
            np.asarray(x, dtype=np.float64).reshape(-1),
            self.predict_fn,
            num_features=num_features,
            labels=(1,),
        )
        return [(str(cond), float(weight)) for cond, weight in exp.as_list(label=1)]
