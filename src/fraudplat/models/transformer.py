"""Transformer sequence-anomaly model (PyTorch).

The gradient-boosted model sees one transaction at a time. A lot of card fraud
is only visible as a *change in rhythm* - the amount is unremarkable, the
merchant is unremarkable, but this transaction does not follow from the eight
that preceded it. That is what this model scores.

Formulation: next-event prediction. Given the card's previous ``seq_len - 1``
transactions, predict the standardised feature vector of the current one. The
anomaly score is the prediction error. Training uses legitimate transactions
only, so "surprising" means "unlike this cardholder's normal behaviour" rather
than "unlike known fraud" - which is what lets it fire on attack patterns that
have never been labelled.

Deliberately small (2 layers, d_model=64, ~120k parameters). It runs per
authorization inside a sub-50ms budget after ONNX export and int8 quantization;
a larger encoder buys a negligible amount of AUC for several times the latency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fraudplat.features.transforms import FEATURE_NAMES

SEQ_LEN = 8

# The model predicts only these features, not the whole vector.
#
# Roughly half of FEATURE_NAMES are rolling aggregates of the card's own
# history (card_txn_count_24h, card_amount_sum_1h, ...). Those are almost
# perfectly predictable from the input window by construction - they *are* the
# input window, summarised - so including them lets the network drive the loss
# down without learning anything about behaviour, and the reconstruction error
# stops discriminating. What is left is the part of a transaction that is
# genuinely a choice: what was bought, where, how, for how much, at what hour.
SEQ_TARGET_FEATURES: list[str] = [
    "log_amount",
    "amount_zscore_card",
    "amount_ratio_card_mean",
    "hour_of_day",
    "is_night",
    "channel_is_ecom",
    "channel_is_atm",
    "entry_is_swipe",
    "entry_is_keyed",
    "is_cross_border",
    "is_new_merchant_for_card",
    "is_new_device_for_card",
    "is_new_country_for_card",
    "merchant_risk_score",
    "mcc_risk_score",
]
SEQ_TARGET_IDX = np.array([FEATURE_NAMES.index(f) for f in SEQ_TARGET_FEATURES], dtype=np.int64)


def build_sequence_index(card_ids: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Map each row to the positions of its preceding transactions on the same card.

    Returns an ``(n, seq_len)`` int array. Position ``seq_len - 1`` is the row
    itself; earlier slots hold the previous rows for that card, or ``-1`` where
    the card has no history yet (padding). Input must already be time-sorted.

    Storing indices instead of materialising ``(n, seq_len, d)`` keeps this at
    ~8 MB for 250k rows instead of ~250 MB, which matters when the training job
    runs in a Kubeflow pod with a fixed memory request.
    """
    n = len(card_ids)
    index = np.full((n, seq_len), -1, dtype=np.int64)
    history: dict[str, list[int]] = {}
    for i, card in enumerate(card_ids):
        hist = history.setdefault(card, [])
        window = hist[-(seq_len - 1):]
        if window:
            index[i, seq_len - 1 - len(window):seq_len - 1] = window
        index[i, seq_len - 1] = i
        hist.append(i)
    return index


def _build_net(d_in: int, d_out: int, seq_len: int, d_model: int, nhead: int, nlayers: int):
    import torch
    from torch import nn

    class SequenceAnomalyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seq_len = seq_len
            self.input_proj = nn.Linear(d_in, d_model)
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            # enable_nested_tensor is incompatible with norm_first and would
            # otherwise emit a UserWarning on every model load.
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=nlayers, enable_nested_tensor=False
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_out),
            )

        def forward(self, x):  # (B, L, d_in) -> (B, d_out)
            # The caller zeroes the final timestep; the model reconstructs it
            # from the preceding ones. Padded history is also zeros, which the
            # positional embedding lets the encoder distinguish from a genuine
            # all-zero (i.e. mean) transaction.
            h = self.input_proj(x) + self.pos
            h = self.encoder(h)
            return self.head(h[:, -1, :])

    return SequenceAnomalyNet()


@dataclass
class SequenceAnomalyModel:
    seq_len: int = SEQ_LEN
    d_model: int = 64
    nhead: int = 4
    nlayers: int = 2
    epochs: int = 6
    batch_size: int = 512
    lr: float = 1e-3
    net: object | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    _lo: float = 0.0
    _hi: float = 1.0
    d_in: int = 0
    feature_weight_: np.ndarray | None = field(default=None, repr=False)

    # -- helpers ---------------------------------------------------------
    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def _gather(self, x_std: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Materialise one batch of ``(B, L, d)`` windows from row indices."""
        safe = np.where(index < 0, 0, index)
        seq = x_std[safe]
        seq[index < 0] = 0.0
        return seq.astype(np.float32)

    # -- training --------------------------------------------------------
    def fit(
        self,
        x: np.ndarray,
        card_ids: np.ndarray,
        y: np.ndarray | None = None,
        seed: int = 17,
    ) -> SequenceAnomalyModel:
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.d_in = x.shape[1]
        self.mean_ = x.mean(axis=0)
        self.std_ = np.where(x.std(axis=0) < 1e-6, 1.0, x.std(axis=0))
        x_std = self._standardize(x).astype(np.float32)

        index = build_sequence_index(np.asarray(card_ids), self.seq_len)
        # Train on legitimate traffic with at least a little history - a
        # sequence model cannot learn rhythm from a card's first transaction.
        has_history = (index[:, :-1] >= 0).sum(axis=1) >= 2
        train_rows = np.flatnonzero(has_history & ((y == 0) if y is not None else True))
        if len(train_rows) == 0:
            raise ValueError("no rows with sufficient card history to train on")

        net = _build_net(
            self.d_in, len(SEQ_TARGET_IDX), self.seq_len, self.d_model, self.nhead, self.nlayers
        )
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr,
            total_steps=max(1, self.epochs * (len(train_rows) // self.batch_size + 1)),
        )
        loss_fn = nn.SmoothL1Loss()  # robust to the heavy tail in amount features
        rng = np.random.default_rng(seed)

        net.train()
        for _ in range(self.epochs):
            order = rng.permutation(train_rows)
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start + self.batch_size]
                seq = self._gather(x_std, index[rows])
                target = torch.from_numpy(seq[:, -1, SEQ_TARGET_IDX].copy())
                seq[:, -1, :] = 0.0  # mask the position being predicted
                pred = net(torch.from_numpy(seq))
                loss = loss_fn(pred, target)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()

        net.eval()
        self.net = net

        # Calibrate the raw error onto [0, 1] against legitimate traffic.
        raw = self._raw_error(x_std, index, rows=train_rows)
        self._lo, self._hi = float(np.percentile(raw, 5)), float(np.percentile(raw, 99.5))
        if self._hi <= self._lo:
            self._hi = self._lo + 1e-6
        return self

    # -- scoring ---------------------------------------------------------
    def _raw_error(self, x_std: np.ndarray, index: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        import torch

        rows = np.arange(len(x_std)) if rows is None else rows
        out = np.zeros(len(rows), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(rows), 4096):
                chunk = rows[start:start + 4096]
                seq = self._gather(x_std, index[chunk])
                target = seq[:, -1, SEQ_TARGET_IDX].copy()
                seq[:, -1, :] = 0.0
                pred = self.net(torch.from_numpy(seq)).numpy()
                out[start:start + len(chunk)] = np.mean((pred - target) ** 2, axis=1)
        return out

    def score(self, x: np.ndarray, card_ids: np.ndarray) -> np.ndarray:
        """Anomaly score in [0, 1] for every row, aligned to ``x``."""
        if self.net is None:
            raise RuntimeError("SequenceAnomalyModel.fit() must be called first")
        x_std = self._standardize(x).astype(np.float32)
        index = build_sequence_index(np.asarray(card_ids), self.seq_len)
        raw = self._raw_error(x_std, index)
        score = np.clip((raw - self._lo) / (self._hi - self._lo), 0.0, 1.0)
        # A card with no history has nothing to be surprising relative to.
        # Emitting the neutral 0.5 rather than a fabricated extreme keeps the
        # ensemble from punishing brand-new cards.
        cold = (index[:, :-1] >= 0).sum(axis=1) < 2
        score[cold] = 0.5
        return score

    def score_window(self, window: np.ndarray) -> float:
        """Score a single ``(seq_len, d_in)`` window - the online path.

        ``window[-1]`` is the transaction being authorised; earlier rows are its
        card's recent history, zero-padded at the front.
        """
        import torch

        if self.net is None:
            raise RuntimeError("SequenceAnomalyModel.fit() must be called first")
        seq = self._standardize(window).astype(np.float32)[None, :, :]
        if (np.abs(seq[0, :-1]).sum(axis=1) > 0).sum() < 2:
            return 0.5
        target = seq[:, -1, SEQ_TARGET_IDX].copy()
        seq[:, -1, :] = 0.0
        with torch.no_grad():
            pred = self.net(torch.from_numpy(seq)).numpy()
        raw = float(np.mean((pred - target) ** 2))
        return float(np.clip((raw - self._lo) / (self._hi - self._lo), 0.0, 1.0))

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        path.with_suffix(".meta.json").write_text(json.dumps({
            "seq_len": self.seq_len, "d_model": self.d_model, "nhead": self.nhead,
            "nlayers": self.nlayers, "d_in": self.d_in,
            "mean": self.mean_.tolist(), "std": self.std_.tolist(),
            "lo": self._lo, "hi": self._hi,
        }))

    @classmethod
    def load(cls, path: Path) -> SequenceAnomalyModel:
        import torch

        meta = json.loads(path.with_suffix(".meta.json").read_text())
        obj = cls(
            seq_len=meta["seq_len"], d_model=meta["d_model"],
            nhead=meta["nhead"], nlayers=meta["nlayers"],
        )
        obj.d_in = meta["d_in"]
        obj.mean_ = np.array(meta["mean"])
        obj.std_ = np.array(meta["std"])
        obj._lo, obj._hi = float(meta["lo"]), float(meta["hi"])
        net = _build_net(
            obj.d_in, len(SEQ_TARGET_IDX), obj.seq_len, obj.d_model, obj.nhead, obj.nlayers
        )
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        obj.net = net
        return obj
