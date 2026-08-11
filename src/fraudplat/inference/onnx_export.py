"""Export the scoring path to ONNX and quantize it.

Why bother when LightGBM and PyTorch both have perfectly good Python
inference: at 5M authorizations/day the p99 budget is the product requirement,
and the Python paths spend most of their time in per-call overhead rather than
arithmetic. A single ONNX Runtime session with a fixed graph removes the
Python-level dispatch, releases the GIL during execution (so one process can
actually use its cores), and gives a deployment artifact that does not depend
on the training library version.

Quantization is applied only to the transformer. Dynamic int8 quantization
targets ``MatMul``/``Gemm`` operators, which is exactly what a transformer is
and exactly what a gradient-boosted tree ensemble is not - quantizing the
LightGBM graph changes the comparison thresholds at every split, which shifts
decision boundaries for no throughput gain. The measured effect of quantizing
each is in ``scripts/benchmark_inference.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fraudplat.features.transforms import FEATURE_NAMES


@dataclass
class ExportResult:
    path: Path
    size_bytes: int
    max_abs_error: float
    note: str = ""


def export_supervised(supervised, out_path: Path, n_features: int | None = None) -> ExportResult:
    """LightGBM booster -> ONNX via onnxmltools."""
    # Must come from onnxmltools' own data_types, not onnxconverter_common:
    # the LightGBM shape calculator does an exact class check and rejects the
    # identically-named type from the other package.
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    n_features = n_features or len(FEATURE_NAMES)
    initial_types = [("input", FloatTensorType([None, n_features]))]
    # onnxmltools' LightGBM converter caps at opset 15 - it does not simply
    # ignore a higher request, it raises. The transformer export below uses 17;
    # the two graphs are separate files and need not share an opset.
    onnx_model = convert_lightgbm(
        supervised.booster,
        initial_types=initial_types,
        target_opset=15,
        zipmap=False,  # plain tensor output; ZipMap emits a list of dicts
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())

    probe = np.random.default_rng(0).normal(size=(64, n_features)).astype(np.float32)
    error = _compare(out_path, {"input": probe}, supervised.predict_proba(probe), output_index=1)
    return ExportResult(out_path, out_path.stat().st_size, error)


def export_iforest(iforest, out_path: Path, n_features: int | None = None) -> ExportResult:
    """Isolation Forest -> ONNX via skl2onnx.

    This one matters most. Benchmarking showed scikit-learn's ``score_samples``
    at ~4.9ms for a single row - roughly 85% of the entire end-to-end scoring
    call, and 100x the cost of the LightGBM predict it sits next to. The reason
    is per-call overhead, not arithmetic: sklearn's ensemble path validates
    input, allocates per-tree buffers and dispatches through joblib on every
    invocation, which is amortised over a large batch and catastrophic on a
    batch of one. Exporting the same 300 trees to a static graph removes all of
    it. Without this step, ONNX-ing the other two models buys almost nothing at
    the level anyone actually measures.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    n_features = n_features or len(FEATURE_NAMES)
    onnx_model = convert_sklearn(
        iforest.model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        # Both domains must be pinned: the tree ensemble lands in 'ai.onnx.ml',
        # whose v4 skl2onnx does not yet emit.
        target_opset={"": 15, "ai.onnx.ml": 3},
        options={id(iforest.model): {"score_samples": True}},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())

    probe = np.random.default_rng(0).normal(size=(64, n_features)).astype(np.float32)
    expected = iforest.model.score_samples(probe)
    error = _compare(out_path, {"input": probe}, expected, output_index=-1)
    return ExportResult(out_path, out_path.stat().st_size, error)


def export_sequence(sequence, out_path: Path) -> ExportResult:
    """Transformer -> ONNX via torch.onnx."""
    import torch

    seq_len, d_in = sequence.seq_len, sequence.d_in
    # Trace with batch > 1. Tracing at batch=1 lets the exporter fold the batch
    # dimension into a constant Reshape inside the attention block, and the
    # resulting graph then fails on any other batch size at runtime - despite
    # dynamic_axes claiming otherwise. The online path scores one transaction
    # at a time, so this would have shipped and only broken under batch replay.
    dummy = torch.zeros(4, seq_len, d_in, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        sequence.net,
        (dummy,),
        str(out_path),
        input_names=["window"],
        output_names=["prediction"],
        dynamic_axes={"window": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    probe = np.random.default_rng(0).normal(size=(16, seq_len, d_in)).astype(np.float32)
    with torch.no_grad():
        expected = sequence.net(torch.from_numpy(probe)).numpy()
    error = _compare(out_path, {"window": probe}, expected)
    return ExportResult(out_path, out_path.stat().st_size, error)


def quantize(src: Path, dst: Path) -> ExportResult:
    """Dynamic int8 quantization (weights only, activations computed at runtime).

    Dynamic rather than static because static quantization needs a calibration
    set to fix activation ranges, and the activation distribution here shifts
    with cardholder mix. Dynamic recomputes per batch and avoids that drift for
    a small throughput cost.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return ExportResult(dst, dst.stat().st_size, float("nan"), note="int8 dynamic")


def _compare(model_path: Path, feeds: dict[str, np.ndarray], expected: np.ndarray, output_index: int = 0) -> float:
    """Numerical parity check. An export that silently changes scores is worse
    than no export at all, so this runs on every conversion."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, feeds)
    got = np.asarray(outputs[min(output_index, len(outputs) - 1)])
    exp = np.asarray(expected)
    if got.ndim == 2 and exp.ndim == 1:
        got = got[:, -1] if got.shape[1] > 1 else got.ravel()
    return float(np.max(np.abs(got.ravel()[: exp.size] - exp.ravel())))
