#!/usr/bin/env python
"""Build the investigation assistant's case index.

    python scripts/build_case_index.py [--max-cases 4000]

Reads the transaction log, replays it through the same FeatureEngine the models
use (so narratives describe the same signals the scorecard saw), and upserts the
documents into the configured vector backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.config import SETTINGS  # noqa: E402
from fraudplat.features.transforms import FeatureEngine, compute_batch  # noqa: E402
from fraudplat.genai.ingest import build_documents, ingest  # noqa: E402
from fraudplat.genai.vectorstore import build_index  # noqa: E402
from fraudplat.models.registry import ModelRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=SETTINGS.paths.data / "transactions.parquet")
    ap.add_argument("--max-cases", type=int, default=4000)
    ap.add_argument("--backend", default=None, help="local | pinecone")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"no transaction data at {args.data}; run scripts/train.py first")
        return 1

    df = pd.read_parquet(args.data)
    print(f"loaded {len(df):,} transactions")

    # Reuse the trained merchant profile so narratives quote the same merchant
    # risk values the production scorer would have seen.
    try:
        profile = ModelRegistry().load().merchant_profile
    except FileNotFoundError:
        profile = None
        print("no trained model found; narratives will omit merchant risk context")

    print("replaying features")
    features = compute_batch(df, engine=FeatureEngine(profile))

    docs = build_documents(df, features, max_cases=args.max_cases)
    index = build_index(args.backend)
    print(f"upserting {len(docs):,} cases into the {index.backend} index")
    ingest(index, docs)

    sample = index.search("card testing burst small ecom digital goods keyed", top_k=3)
    print("\nsanity check - query: 'card testing burst small ecom digital goods keyed'")
    for r in sample:
        print(f"  [{r.score:.3f}] {r.case_id} ({r.metadata.get('disposition')}) {r.text[:90]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
