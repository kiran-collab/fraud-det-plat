#!/usr/bin/env python
"""Export the current model version to ONNX and quantize the transformer.

    python scripts/export_onnx.py [--version v20250101-120000]

Writes into ``<model_version>/onnx/`` so the artifacts travel with the version
they were built from. The scorer picks them up automatically on next start;
there is no separate "enable ONNX" switch to forget to flip.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.inference.onnx_export import (  # noqa: E402
    export_iforest,
    export_sequence,
    export_supervised,
    quantize,
)
from fraudplat.models.registry import ModelRegistry  # noqa: E402

# Above this, the exported graph is not numerically equivalent to the model it
# came from and must not be promoted - a silent score shift is far worse than a
# failed build.
MAX_ACCEPTABLE_ERROR = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=None)
    args = ap.parse_args()

    registry = ModelRegistry()
    bundle = registry.load(args.version)
    out_dir = Path(registry.version_dir(bundle.version)) / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"exporting model {bundle.version} -> {out_dir}")

    failures: list[str] = []

    sup = export_supervised(bundle.supervised, out_dir / "supervised.onnx")
    print(f"  supervised.onnx      {sup.size_bytes / 1024:8.1f} KB  max|Δ|={sup.max_abs_error:.3e}")
    if sup.max_abs_error > MAX_ACCEPTABLE_ERROR:
        failures.append(f"supervised parity {sup.max_abs_error:.3e}")

    iso = export_iforest(bundle.iforest, out_dir / "iforest.onnx")
    print(f"  iforest.onnx         {iso.size_bytes / 1024:8.1f} KB  max|Δ|={iso.max_abs_error:.3e}")
    if iso.max_abs_error > MAX_ACCEPTABLE_ERROR:
        failures.append(f"iforest parity {iso.max_abs_error:.3e}")

    seq = export_sequence(bundle.sequence, out_dir / "sequence.onnx")
    print(f"  sequence.onnx        {seq.size_bytes / 1024:8.1f} KB  max|Δ|={seq.max_abs_error:.3e}")
    if seq.max_abs_error > MAX_ACCEPTABLE_ERROR:
        failures.append(f"sequence parity {seq.max_abs_error:.3e}")

    q = quantize(out_dir / "sequence.onnx", out_dir / "sequence.int8.onnx")
    print(
        f"  sequence.int8.onnx   {q.size_bytes / 1024:8.1f} KB  "
        f"({100 * q.size_bytes / seq.size_bytes:.0f}% of fp32)"
    )

    if failures:
        print("\nFAILED numerical parity: " + "; ".join(failures))
        print("Refusing to promote an export whose scores differ from the trained model.")
        return 1

    print("\nexport complete; the scorer will use these on next start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
