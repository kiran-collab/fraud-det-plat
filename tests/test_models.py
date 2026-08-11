"""Model and ensemble tests.

These assert on *properties* the blend must satisfy rather than on metric
values, so they keep catching regressions as the models legitimately change.
"""

from __future__ import annotations

import numpy as np

from fraudplat.config import EnsembleConfig
from fraudplat.models.ensemble import MAX_LOGIT_ADJUSTMENT, EnsembleScorer
from fraudplat.models.iforest import IsolationForestScorer
from fraudplat.models.transformer import SEQ_TARGET_IDX, build_sequence_index


# --- ensemble ------------------------------------------------------------
def _fitted_ensemble(seed: int = 0) -> EnsembleScorer:
    rng = np.random.default_rng(seed)
    n = 4000
    y = (rng.random(n) < 0.01).astype(int)
    p = np.clip(rng.beta(1, 40, n) + 0.5 * y, 0, 1)
    s_if = np.clip(rng.beta(2, 5, n) + 0.15 * y, 0, 1)
    s_seq = np.clip(rng.beta(2, 5, n) + 0.15 * y, 0, 1)
    return EnsembleScorer().fit(p, y, s_if, s_seq)


def test_neutral_unsupervised_scores_leave_supervised_untouched():
    """The whole point of logit blending: an unremarkable anomaly score must
    not move a calibrated probability at all."""
    ens = _fitted_ensemble()
    p = np.array([0.001, 0.02, 0.4, 0.95])
    blended = ens.blend(
        p,
        np.full_like(p, ens.ref_iforest),
        np.full_like(p, ens.ref_sequence),
    )
    # Tolerance is the logit epsilon clamp: isotonic calibration can output a
    # hard 0.0 or 1.0, and the round trip through log-odds maps those to
    # 1e-6 / 1-1e-6 rather than back to the saturated value. Harmless - it sits
    # ~5 orders of magnitude below the nearest decision threshold - but it does
    # mean the blend never returns an exactly-certain score.
    np.testing.assert_allclose(blended, ens.calibrate(p), atol=2e-6)


def test_blend_does_not_inflate_low_risk_traffic():
    """A linear blend put a floor under every legitimate transaction that any
    detector found mildly unusual, destroying precision at the decline
    threshold. Logit blending must not."""
    ens = _fitted_ensemble()
    p = np.full(500, 1e-4)                    # confidently legitimate
    mildly_odd = np.full(500, ens.ref_iforest + 0.25)
    blended = ens.blend(p, mildly_odd, np.full(500, ens.ref_sequence))
    assert blended.max() < 0.35, "mild anomaly lifted clean traffic into the review band"


def test_strong_anomaly_can_escalate_but_not_dominate():
    """Unsupervised detectors must be able to raise an unlabelled attack into
    review, without being able to overturn a confident supervised verdict."""
    ens = _fitted_ensemble()
    p = np.array([0.02])
    escalated = ens.blend(p, np.array([1.0]), np.array([1.0]))
    assert escalated[0] > ens.calibrate(p)[0], "anomaly detectors cannot escalate at all"

    confident_legit = np.array([1e-6])
    still_low = ens.blend(confident_legit, np.array([1.0]), np.array([1.0]))
    assert still_low[0] < 0.5, "anomaly detectors overrode a confident supervised score"


def test_logit_adjustment_is_bounded():
    ens = EnsembleScorer(weights=EnsembleConfig(iforest_logit_weight=99.0, sequence_logit_weight=99.0))
    p = np.array([0.5])
    high = ens.blend(p, np.array([1.0]), np.array([1.0]))[0]
    # sigmoid(0 + MAX) is the ceiling regardless of how large the weights are.
    assert high <= 1.0 / (1.0 + np.exp(-MAX_LOGIT_ADJUSTMENT)) + 1e-9


def test_blend_is_monotone_in_each_component():
    ens = _fitted_ensemble()
    grid = np.linspace(0.0, 1.0, 25)
    base = np.full_like(grid, 0.05)
    ref_seq = np.full_like(grid, ens.ref_sequence)
    by_iforest = ens.blend(base, grid, ref_seq)
    assert np.all(np.diff(by_iforest) >= -1e-12)
    by_sequence = ens.blend(base, np.full_like(grid, ens.ref_iforest), grid)
    assert np.all(np.diff(by_sequence) >= -1e-12)


def test_blend_output_is_a_probability():
    ens = _fitted_ensemble()
    rng = np.random.default_rng(3)
    out = ens.blend(rng.random(2000), rng.random(2000), rng.random(2000))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()


def test_calibration_improves_probability_accuracy():
    """Isotonic calibration should bring predicted rates closer to observed."""
    rng = np.random.default_rng(11)
    n = 20_000
    y = (rng.random(n) < 0.01).astype(int)
    # Systematically over-confident scores, as scale_pos_weight produces.
    raw = np.clip(rng.beta(1, 12, n) + 0.45 * y, 0, 1)
    ens = EnsembleScorer().fit(raw, y)
    assert abs(ens.calibrate(raw).mean() - y.mean()) < abs(raw.mean() - y.mean())


# --- isolation forest ----------------------------------------------------
def test_iforest_scores_are_bounded_and_calibrated(rng):
    x = rng.normal(size=(2000, 8))
    y = np.zeros(len(x), dtype=int)
    scorer = IsolationForestScorer(n_estimators=50).fit(x.astype(np.float32), y, seed=0)
    scores = scorer.score(x.astype(np.float32))
    assert scores.min() >= 0.0 and scores.max() <= 1.0

    # A far outlier must score above typical in-distribution traffic.
    outlier = np.full((1, 8), 25.0, dtype=np.float32)
    assert scorer.score(outlier)[0] > np.percentile(scores, 90)


def test_iforest_calibrate_raw_matches_score(rng):
    """The ONNX path calls calibrate_raw directly; it must agree with score()."""
    x = rng.normal(size=(500, 6)).astype(np.float32)
    scorer = IsolationForestScorer(n_estimators=40).fit(x, np.zeros(len(x), int), seed=0)
    np.testing.assert_allclose(
        scorer.score(x), scorer.calibrate_raw(-scorer.model.score_samples(x)), rtol=1e-12
    )


# --- sequence model ------------------------------------------------------
def test_sequence_index_is_causal():
    cards = np.array(["a", "b", "a", "a", "b"])
    idx = build_sequence_index(cards, seq_len=4)
    # Last slot is always the row itself.
    assert list(idx[:, -1]) == [0, 1, 2, 3, 4]
    # Row 3 (card 'a', third occurrence) sees rows 0 and 2, padded before.
    assert list(idx[3]) == [-1, 0, 2, 3]
    # A card's first transaction has no history at all.
    assert list(idx[0]) == [-1, -1, -1, 0]
    # No row may reference a future position.
    for i, row in enumerate(idx):
        assert all(p < i for p in row[:-1] if p >= 0)


def test_sequence_index_never_mixes_cards():
    rng = np.random.default_rng(5)
    cards = rng.choice(["a", "b", "c"], 300)
    idx = build_sequence_index(cards, seq_len=6)
    for i, row in enumerate(idx):
        for p in row:
            if p >= 0:
                assert cards[p] == cards[i]


def test_sequence_targets_exclude_self_derived_aggregates():
    """Targets must not include rolling aggregates of the input window - those
    are trivially predictable and would collapse the reconstruction error."""
    from fraudplat.features.transforms import FEATURE_NAMES

    targets = {FEATURE_NAMES[i] for i in SEQ_TARGET_IDX}
    assert not any(t.startswith("card_txn_count") for t in targets)
    assert not any(t.startswith("card_amount_sum") for t in targets)
    assert "log_amount" in targets and "is_cross_border" in targets
