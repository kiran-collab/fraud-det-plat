#!/usr/bin/env python
"""Export a ready-to-train feature table.

    python scripts/export_training_table.py --rows 300000 --days 120

Produces one row per transaction with the 31 model features, the label, and a
chronological split marker, in both CSV and Parquet.

Two things this script is careful about, because both are easy to get wrong
and neither raises an error when you do:

**merchant_risk_score is fitted on the train slice only.** It is a target
encoding - each merchant replaced by its historical fraud rate - so fitting it
over the whole table lets every validation and test row carry a number
computed partly from its own label. Measured on this dataset, that single
mistake inflates PR-AUC from 0.635 to 0.809.

**The split column is chronological, not random.** Fraud arrives in episodes:
a card-testing burst is 5-8 transactions minutes apart. A random split scatters
those across train and test, so half the test frauds have a near-twin sitting
in training and the model can memorise rather than generalise. Measured here:
70 of 141 test frauds had a same-card fraud less than an hour away in train,
versus 0 of 152 under a time split.

``customer_age_band`` is included but is NOT a feature. It exists so fairness
metrics can be computed downstream; feeding it to a model would be a
fair-lending problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.config import SETTINGS  # noqa: E402
from fraudplat.data.generator import generate  # noqa: E402
from fraudplat.features.transforms import (  # noqa: E402
    FEATURE_NAMES,
    FeatureEngine,
    MerchantProfile,
    compute_batch,
)

# Carried alongside the features for reference, joining and analysis. None of
# these are model inputs.
CONTEXT_COLUMNS = [
    "transaction_id",
    "event_time",
    "card_id",
    "merchant_id",
    "merchant_category",
    "merchant_country",
    "channel",
    "entry_mode",
    "customer_age_band",  # fairness monitoring only - never a feature
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=300_000)
    ap.add_argument("--days", type=int, default=120, help="calendar span to spread rows over")
    ap.add_argument("--fraud-rate", type=float, default=0.0085)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--valid-frac", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=SETTINGS.paths.data)
    ap.add_argument("--csv", action="store_true", default=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"generating {args.rows:,} transactions over {args.days} days")
    df = generate(
        n_transactions=args.rows,
        days=args.days,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    ).sort_values("event_time", ignore_index=True)

    n = len(df)
    a = int(n * args.train_frac)
    b = int(n * (args.train_frac + args.valid_frac))
    split = pd.Series("test", index=df.index)
    split.iloc[:a] = "train"
    split.iloc[a:b] = "valid"

    # Train slice only - see module docstring.
    profile = MerchantProfile.fit(df.iloc[:a])
    print(f"merchant encoding fitted on {a:,} train rows ({len(profile.total_counts):,} merchants)")

    print("replaying features in timestamp order")
    features = compute_batch(df, engine=FeatureEngine(profile))

    table = pd.concat(
        [df[CONTEXT_COLUMNS], split.rename("split"), features, df[["is_fraud"]]], axis=1
    )

    parquet_path = args.out / "training_table.parquet"
    table.to_parquet(parquet_path, index=False)
    written = [parquet_path]
    if args.csv:
        csv_path = args.out / "training_table.csv"
        table.to_csv(csv_path, index=False)
        written.append(csv_path)

    print(f"\nrows={len(table):,}  columns={table.shape[1]}  features={len(FEATURE_NAMES)}")
    print(f"window: {df.event_time.min():%Y-%m-%d} -> {df.event_time.max():%Y-%m-%d}")
    print(f"cards={df.card_id.nunique():,}  merchants={df.merchant_id.nunique():,}")
    print("\nprevalence by split:")
    for name, part in table.groupby("split", sort=False):
        print(
            f"  {name:6} n={len(part):>8,}  fraud={int(part.is_fraud.sum()):>5,}  "
            f"{part.is_fraud.mean():.4%}"
        )
    for p in written:
        print(f"\nwrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
