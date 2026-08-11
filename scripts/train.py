#!/usr/bin/env python
"""Train the full model bundle and register a new version.

    python scripts/train.py --rows 250000 --promote

Steps, in order, and the order matters:

  1. chronological split - never random, or velocity features leak the future
  2. merchant profile fitted on the TRAIN window only
  3. features replayed through one FeatureEngine across all three splits, so
     validation and test start with realistic card state instead of cold
  4. LightGBM + Isolation Forest + sequence model
  5. isotonic calibration fitted on validation
  6. evaluation on test, then registry write
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.config import SETTINGS  # noqa: E402
from fraudplat.data.generator import generate, time_split  # noqa: E402
from fraudplat.evaluation import evaluate, format_report  # noqa: E402
from fraudplat.features.transforms import (  # noqa: E402
    FeatureEngine,
    MerchantProfile,
    compute_batch,
)
from fraudplat.models.ensemble import EnsembleScorer  # noqa: E402
from fraudplat.models.iforest import IsolationForestScorer  # noqa: E402
from fraudplat.models.lgbm import SupervisedModel  # noqa: E402
from fraudplat.models.registry import ModelBundle, ModelRegistry  # noqa: E402
from fraudplat.models.transformer import SequenceAnomalyModel  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_or_generate(rows: int, seed: int, data_path: Path | None) -> pd.DataFrame:
    if data_path and data_path.exists():
        log(f"loading transactions from {data_path}")
        return pd.read_parquet(data_path)
    log(f"generating {rows:,} synthetic transactions")
    df = generate(n_transactions=rows, seed=seed)
    if data_path:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)
        log(f"cached transactions -> {data_path}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=SETTINGS.random_seed)
    ap.add_argument("--data", type=Path, default=SETTINGS.paths.data / "transactions.parquet")
    ap.add_argument("--epochs", type=int, default=4, help="sequence-model epochs")
    ap.add_argument("--promote", action="store_true", help="repoint 'current' at this version")
    ap.add_argument("--version", default=None)
    args = ap.parse_args()

    SETTINGS.paths.ensure()
    t0 = time.time()

    df = load_or_generate(args.rows, args.seed, args.data)
    train_df, valid_df, test_df = time_split(df)
    log(
        f"split  train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}  "
        f"prevalence={df['is_fraud'].mean():.4%}"
    )

    # --- 2. merchant encoding from the training window only --------------
    profile = MerchantProfile.fit(train_df)
    log(f"merchant profile: {len(profile.total_counts):,} merchants, base rate {profile.base_rate:.4%}")

    # --- 3. one engine, replayed in time order across all splits ---------
    engine = FeatureEngine(profile)
    log("building features (train)")
    x_train = compute_batch(train_df, engine=engine)
    log("building features (valid)")
    x_valid = compute_batch(valid_df, engine=engine)
    log("building features (test)")
    x_test = compute_batch(test_df, engine=engine)

    y_train = train_df["is_fraud"].to_numpy()
    y_valid = valid_df["is_fraud"].to_numpy()
    y_test = test_df["is_fraud"].to_numpy()
    xt, xv, xs = (x.to_numpy(dtype=np.float32) for x in (x_train, x_valid, x_test))

    # --- 4. models -------------------------------------------------------
    log("training LightGBM")
    supervised = SupervisedModel().fit(xt, y_train, xv, y_valid, seed=args.seed)
    log(f"  best iteration: {supervised.booster.best_iteration}")

    log("training Isolation Forest (legitimate traffic only)")
    iforest = IsolationForestScorer().fit(xt, y_train, seed=args.seed)

    log(f"training sequence model ({args.epochs} epochs)")
    sequence = SequenceAnomalyModel(epochs=args.epochs).fit(
        xt, train_df["card_id"].to_numpy(), y_train, seed=args.seed
    )

    # --- 5. calibrate the supervised head on validation ------------------
    log("calibrating on validation split")
    p_valid = supervised.predict_proba(xv)
    ensemble = EnsembleScorer().fit(
        p_valid,
        y_valid,
        iforest.score(xv),
        sequence.score(xv, valid_df["card_id"].to_numpy()),
    )

    # --- 6. evaluate on the held-out test window -------------------------
    log("scoring test window")
    p_test = supervised.predict_proba(xs)
    s_if = iforest.score(xs)
    s_seq = sequence.score(xs, test_df["card_id"].to_numpy())
    blended = ensemble.blend(p_test, s_if, s_seq)

    amount = test_df["amount"].to_numpy()
    rep_ens = evaluate(y_test, blended, amount)
    rep_sup = evaluate(y_test, ensemble.calibrate(p_test), amount)
    rep_if = evaluate(y_test, s_if, amount)
    rep_seq = evaluate(y_test, s_seq, amount)

    print()
    print(format_report(rep_ens, "Ensemble (production scorer)"))
    print()
    print(format_report(rep_sup, "LightGBM only"))
    print()
    print(format_report(rep_if, "Isolation Forest only"))
    print()
    print(format_report(rep_seq, "Sequence model only"))
    print()

    importance = supervised.feature_importance()
    print("== Top features by gain ==")
    for name, share in list(importance.items())[:12]:
        print(f"  {share:6.2%}  {name}")
    print()

    # --- 7. register -----------------------------------------------------
    bundle = ModelBundle(
        supervised=supervised,
        iforest=iforest,
        sequence=sequence,
        ensemble=ensemble,
        merchant_profile=profile,
        manifest={
            "rows_total": len(df),
            "rows_train": len(train_df),
            "rows_valid": len(valid_df),
            "rows_test": len(test_df),
            "train_window": [str(train_df["event_time"].min()), str(train_df["event_time"].max())],
            "test_window": [str(test_df["event_time"].min()), str(test_df["event_time"].max())],
            "prevalence": float(df["is_fraud"].mean()),
            "seed": args.seed,
            "lgbm_best_iteration": int(supervised.booster.best_iteration or 0),
            "metrics": {
                "ensemble": rep_ens.to_dict(),
                "supervised_only": rep_sup.to_dict(),
                "iforest_only": rep_if.to_dict(),
                "sequence_only": rep_seq.to_dict(),
            },
            "feature_importance": importance,
            "training_seconds": round(time.time() - t0, 1),
        },
    )

    registry = ModelRegistry()
    out = registry.save(bundle, version=args.version, promote=args.promote)
    log(f"registered model at {out}" + ("  (promoted to current)" if args.promote else ""))

    report_path = SETTINGS.paths.reports / f"train_{out.name}.json"
    report_path.write_text(json.dumps(bundle.manifest, indent=2, default=str))
    log(f"training report -> {report_path}")
    log(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
