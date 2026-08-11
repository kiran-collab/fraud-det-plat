"""ONNX Runtime scoring session.

Session construction is expensive (graph load, optimisation, arena
allocation); inference is cheap. So sessions are built once at process start
and reused for the life of the pod, which is what makes the per-call cost
tolerable.

Threading is pinned to 1 intra-op thread deliberately. The instinct is to give
each session all the cores, but this service runs many concurrent single-row
requests, not a few large batches. With multiple threads per session the
runtime spends more time in fork/join barriers than in arithmetic, and p99
latency gets *worse* under load. Concurrency comes from running several
workers, each with a single-threaded session.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _session(path: Path, intra_op_threads: int = 1):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


@dataclass
class OnnxScorer:
    """Wraps the two exported graphs behind the same call signature the
    Python models expose, so the API code is identical either way."""

    supervised_path: Path
    sequence_path: Path | None = None
    iforest_path: Path | None = None
    _sup: object = None
    _seq: object = None
    _if: object = None
    _sup_input: str = "input"
    _seq_input: str = "window"
    _if_input: str = "input"

    def __post_init__(self) -> None:
        self._sup = _session(self.supervised_path)
        self._sup_input = self._sup.get_inputs()[0].name
        if self.sequence_path and Path(self.sequence_path).exists():
            self._seq = _session(self.sequence_path)
            self._seq_input = self._seq.get_inputs()[0].name
        if self.iforest_path and Path(self.iforest_path).exists():
            self._if = _session(self.iforest_path)
            self._if_input = self._if.get_inputs()[0].name

    @property
    def has_sequence(self) -> bool:
        return self._seq is not None

    @property
    def has_iforest(self) -> bool:
        return self._if is not None

    def iforest_raw(self, x: np.ndarray) -> np.ndarray:
        """Raw anomaly values, sign-matched to ``IsolationForestScorer``.

        The graph emits sklearn's ``score_samples`` (higher = more normal);
        the scorer's convention is higher = more anomalous, hence the negation.
        Calibration onto [0, 1] is applied by the caller using the fitted
        percentile bounds, which are not part of the exported graph.
        """
        if self._if is None:
            raise RuntimeError("no isolation forest graph loaded")
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return -np.asarray(self._if.run(None, {self._if_input: x})[-1]).ravel()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        outputs = self._sup.run(None, {self._sup_input: x})
        # convert_lightgbm(zipmap=False) emits [label, probabilities].
        probs = np.asarray(outputs[-1])
        return probs[:, -1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()

    def sequence_forward(self, window: np.ndarray) -> np.ndarray:
        if self._seq is None:
            raise RuntimeError("no sequence graph loaded")
        w = np.ascontiguousarray(window, dtype=np.float32)
        if w.ndim == 2:
            w = w[None, :, :]
        return np.asarray(self._seq.run(None, {self._seq_input: w})[0])
