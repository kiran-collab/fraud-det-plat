#!/usr/bin/env python
"""Measure the inference paths against the latency budget.

    python scripts/benchmark_inference.py [--iterations 2000]

Reports p50/p95/p99 for each stage and for the full end-to-end scoring call,
comparing the native Python path against ONNX Runtime.

Latency is reported as percentiles, never as a mean. The mean of a latency
distribution is dominated by the fast path and hides exactly the tail that
causes authorization timeouts.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.features.transforms import FEATURE_NAMES  # noqa: E402
from fraudplat.models.registry import ModelRegistry  # noqa: E402
from fraudplat.serving.scorer import TransactionScorer  # noqa: E402

BUDGET_MS = 50.0


def percentiles(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {
        "p50": round(statistics.median(s), 3),
        "p95": round(s[int(len(s) * 0.95)], 3),
        "p99": round(s[int(len(s) * 0.99)], 3),
        "max": round(s[-1], 3),
    }


def time_it(fn, iterations: int, warmup: int = 50) -> dict[str, float]:
    for _ in range(warmup):  # let allocators and caches settle
        fn()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return percentiles(samples)


def row(name: str, stats: dict[str, float], budget: bool = False) -> str:
    flag = ""
    if budget:
        flag = "  OK" if stats["p99"] <= BUDGET_MS else "  OVER BUDGET"
    return (
        f"  {name:<34}p50={stats['p50']:>8.3f}  p95={stats['p95']:>8.3f}  "
        f"p99={stats['p99']:>8.3f}  max={stats['max']:>8.3f}{flag}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--version", default=None)
    args = ap.parse_args()

    registry = ModelRegistry()
    bundle = registry.load(args.version)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, len(FEATURE_NAMES))).astype(np.float32)
    window = rng.normal(size=(bundle.sequence.seq_len, bundle.sequence.d_in)).astype(np.float32)

    print(f"model {bundle.version}   iterations={args.iterations}   budget={BUDGET_MS}ms (p99)\n")
    print("-- component latency (single transaction, milliseconds) --")

    print(row("LightGBM (python)", time_it(lambda: bundle.supervised.predict_proba(x), args.iterations)))
    print(row("IsolationForest (python)", time_it(lambda: bundle.iforest.score(x), args.iterations)))
    print(row("Transformer (pytorch)", time_it(lambda: bundle.sequence.score_window(window), args.iterations)))

    onnx_dir = Path(registry.version_dir(bundle.version)) / "onnx"
    if (onnx_dir / "supervised.onnx").exists():
        from fraudplat.inference.runtime import OnnxScorer

        fp32 = OnnxScorer(onnx_dir / "supervised.onnx", onnx_dir / "sequence.onnx")
        print(row("LightGBM (onnx)", time_it(lambda: fp32.predict_proba(x), args.iterations)))
        print(row("Transformer (onnx fp32)", time_it(lambda: fp32.sequence_forward(window), args.iterations)))

        if (onnx_dir / "sequence.int8.onnx").exists():
            int8 = OnnxScorer(onnx_dir / "supervised.onnx", onnx_dir / "sequence.int8.onnx")
            print(row("Transformer (onnx int8)", time_it(lambda: int8.sequence_forward(window), args.iterations)))
    else:
        print("  (no ONNX build found - run scripts/export_onnx.py)")

    # --- end to end -----------------------------------------------------
    print("\n-- end-to-end /score (features + 3 models + blend + rules + store) --")
    txn = {
        "transaction_id": "bench", "card_id": "card_0000007", "merchant_id": "mch_000010",
        "merchant_category": "electronics", "merchant_country": "US", "amount": 240.0,
        "channel": "ecom", "entry_mode": "keyed", "device_id": "dev_0000007",
        "issuer_country": "US",
    }
    n = min(args.iterations, 500)  # end-to-end is the slow one; keep runtime sane
    headline: dict[str, dict[str, float]] = {}
    for backend, use_onnx in (("python", False), ("onnxruntime", True)):
        scorer = TransactionScorer(bundle=bundle, use_onnx=use_onnx)
        if use_onnx and scorer.inference_backend != "onnxruntime":
            print("  (onnx path unavailable; run scripts/export_onnx.py)")
            continue
        # Bind `scorer` as a default argument: a bare closure over the loop
        # variable would have every lambda call whichever scorer the loop
        # ended on, silently benchmarking the same backend twice.
        stats = time_it(lambda s=scorer: s.score(dict(txn), explain=False), n, warmup=20)
        headline[scorer.inference_backend] = stats
        print(row(f"scorer [{scorer.inference_backend}]", stats, budget=True))
        stats_x = time_it(lambda s=scorer: s.score(dict(txn), explain=True), n, warmup=20)
        print(row(f"scorer [{backend}] + SHAP", stats_x, budget=True))

    # --- headline numbers ------------------------------------------------
    print("\n-- summary --")
    for backend, stats in headline.items():
        per_core = 1000.0 / stats["p50"]
        print(
            f"  {backend:<12} p50={stats['p50']:.2f}ms  p99={stats['p99']:.2f}ms  "
            f"~{per_core:,.0f} txn/s per core  "
            f"({5_000_000 / 86_400 / per_core * 100:.1f}% of one core to carry 5M/day)"
        )
    if "python" in headline and "onnxruntime" in headline:
        speedup = headline["python"]["p50"] / headline["onnxruntime"]["p50"]
        print(f"  ONNX Runtime speedup: {speedup:.2f}x at p50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
