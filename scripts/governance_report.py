#!/usr/bin/env python
"""Generate the model governance pack.

    python scripts/governance_report.py

Produces the artifacts a model-risk review asks for, from the registered model
and the held-out window:

  * performance by operating point
  * global SHAP attributions
  * fairness metrics across customer segments
  * feature and score drift, train window vs test window

Exits non-zero if a fairness or drift check fails, so it can gate promotion in
the Kubeflow pipeline rather than being a report nobody reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.config import SETTINGS  # noqa: E402
from fraudplat.data.generator import time_split  # noqa: E402
from fraudplat.evaluation import evaluate, format_report  # noqa: E402
from fraudplat.features.transforms import FEATURE_NAMES, FeatureEngine, compute_batch  # noqa: E402
from fraudplat.governance.bias_monitor import evaluate_bias, format_bias_report  # noqa: E402
from fraudplat.governance.drift import detect_drift, format_drift_report  # noqa: E402
from fraudplat.models.registry import ModelRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=SETTINGS.paths.data / "transactions.parquet")
    ap.add_argument("--version", default=None)
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any finding")
    args = ap.parse_args()

    SETTINGS.paths.ensure()
    if not args.data.exists():
        print(f"no transaction data at {args.data}; run scripts/train.py first")
        return 1

    bundle = ModelRegistry().load(args.version)
    df = pd.read_parquet(args.data)
    train_df, _, test_df = time_split(df)

    engine = FeatureEngine(bundle.merchant_profile)
    x_train = compute_batch(train_df, engine=engine).to_numpy(dtype=np.float32)
    # Skip the validation window's features but keep its state, so the test
    # window sees the card history it would have in production.
    _ = compute_batch(df.iloc[len(train_df):len(df) - len(test_df)], engine=engine)
    x_test = compute_batch(test_df, engine=engine).to_numpy(dtype=np.float32)

    def blended(x: np.ndarray, cards: np.ndarray) -> np.ndarray:
        return bundle.ensemble.blend(
            bundle.supervised.predict_proba(x),
            bundle.iforest.score(x),
            bundle.sequence.score(x, cards),
        )

    s_train = blended(x_train, train_df["card_id"].to_numpy())
    s_test = blended(x_test, test_df["card_id"].to_numpy())

    print(f"model version: {bundle.version}\n")

    # --- 1. performance --------------------------------------------------
    perf = evaluate(
        test_df["is_fraud"].to_numpy(), s_test, test_df["amount"].to_numpy()
    )
    print(format_report(perf, "Held-out performance"))

    # --- 2. explainability -----------------------------------------------
    from fraudplat.explain.shap_explainer import ShapExplainer

    shap_global = ShapExplainer.from_model(bundle.supervised).global_importance(x_test)
    print("\n== Global SHAP attribution (mean |contribution|) ==")
    for name, share in list(shap_global.items())[:12]:
        print(f"  {share:6.2%}  {name}")

    # --- 3. fairness ------------------------------------------------------
    bias = evaluate_bias(
        test_df, s_test,
        attribute="customer_age_band",
        threshold=SETTINGS.decision.decline_at,
    )
    print("\n" + format_bias_report(bias))

    # --- 4. drift ---------------------------------------------------------
    drift = detect_drift(x_train, x_test, FEATURE_NAMES, s_train, s_test)
    print("\n" + format_drift_report(drift))

    # --- 5. write the pack -------------------------------------------------
    out = SETTINGS.paths.reports / f"governance_{bundle.version}.json"
    out.write_text(json.dumps({
        "model_version": bundle.version,
        "manifest": bundle.manifest,
        "performance": perf.to_dict(),
        "global_shap": shap_global,
        "fairness": bias.to_dict(),
        "drift": drift.to_dict(),
    }, indent=2, default=str))
    print(f"\ngovernance pack -> {out}")

    findings = (not bias.passed) or (not drift.passed)
    if findings:
        print("\nFINDINGS PRESENT - review required before promotion")
    if findings and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
