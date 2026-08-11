"""Governance tests: drift statistics, fairness metrics, audit integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudplat.governance.audit import AuditLog, pseudonymize
from fraudplat.governance.bias_monitor import evaluate_bias
from fraudplat.governance.drift import detect_drift, population_stability_index


# --- PSI ------------------------------------------------------------------
def test_psi_is_near_zero_for_identical_distributions(rng):
    x = rng.normal(size=5000)
    y = rng.normal(size=5000)
    assert population_stability_index(x, y) < 0.05


def test_psi_grows_with_distribution_shift(rng):
    base = rng.normal(size=8000)
    small = population_stability_index(base, rng.normal(0.2, 1, 8000))
    large = population_stability_index(base, rng.normal(2.0, 1, 8000))
    assert small < large
    assert large > 0.25  # would be flagged for investigation


def test_psi_handles_skewed_features(rng):
    """Amounts and velocity counts are heavily skewed. Equal-width bins would
    put nearly all mass in one bucket and report no drift regardless of shift;
    quantile edges must not."""
    base = rng.lognormal(3, 1, 8000)
    shifted = rng.lognormal(4, 1, 8000)
    assert population_stability_index(base, shifted) > 0.25


def test_psi_is_finite_for_constant_features():
    """Rare binary flags are near-constant; PSI must degrade gracefully rather
    than divide by an empty bucket."""
    base = np.zeros(1000)
    current = np.concatenate([np.zeros(990), np.ones(10)])
    psi = population_stability_index(base, current)
    assert np.isfinite(psi)


def test_psi_does_not_explode_on_an_empty_bucket(rng):
    """Without Laplace smoothing a single unlucky empty bin sends PSI to inf."""
    base = rng.normal(size=2000)
    current = rng.normal(size=2000)
    current = current[current > -0.5]  # carve a hole in the left tail
    assert np.isfinite(population_stability_index(base, current))


def test_drift_report_flags_and_ranks(rng):
    names = ["a", "b", "c"]
    baseline = rng.normal(size=(4000, 3))
    current = baseline.copy()
    current[:, 1] += 3.0  # only 'b' drifts
    report = detect_drift(baseline, current, names)
    assert report.features[0].feature == "b"      # sorted by PSI descending
    assert report.features[0].status == "investigate"
    assert not report.passed
    assert any("b" in f for f in report.flagged)


# --- fairness -------------------------------------------------------------
def _bias_frame(rng, n=8000, biased=False):
    seg = rng.choice(["18-25", "26-40", "41-60", "60+"], n)
    y = (rng.random(n) < 0.01).astype(int)
    score = np.clip(rng.beta(1, 30, n) + 0.5 * y, 0, 1)
    if biased:
        # Inflate scores for one segment regardless of the label.
        score = np.where(seg == "60+", np.clip(score + 0.85, 0, 1), score)
    return pd.DataFrame({"customer_age_band": seg, "is_fraud": y}), score


def test_bias_monitor_passes_on_equitable_outcomes(rng):
    df, score = _bias_frame(rng)
    report = evaluate_bias(df, score, threshold=0.86, max_fpr_gap=0.02, max_tpr_gap=0.4)
    assert report.passed, report.flags


def test_bias_monitor_detects_a_segment_declined_far_more_often(rng):
    df, score = _bias_frame(rng, biased=True)
    report = evaluate_bias(df, score, threshold=0.86)
    assert not report.passed
    assert report.adverse_impact_ratio < 0.8
    assert any("adverse impact" in f for f in report.flags)


def test_small_segments_are_marked_unreliable(rng):
    df = pd.DataFrame({
        "customer_age_band": ["big"] * 900 + ["tiny"] * 5,
        "is_fraud": [0] * 905,
    })
    report = evaluate_bias(df, np.zeros(905), threshold=0.5)
    tiny = next(s for s in report.segments if s.segment == "tiny")
    assert not tiny.reliable


def test_tpr_is_withheld_when_positives_are_too_few(rng):
    """A TPR computed from three fraud cases is noise and must not be reported
    as a fairness finding."""
    df = pd.DataFrame({
        "customer_age_band": ["a"] * 500 + ["b"] * 500,
        "is_fraud": [1] * 3 + [0] * 497 + [1] * 3 + [0] * 497,
    })
    report = evaluate_bias(df, np.zeros(1000), threshold=0.5)
    assert all(s.true_positive_rate is None for s in report.segments)
    assert report.tpr_gap == 0.0


# --- audit log ------------------------------------------------------------
class _Result:
    action = "decline"
    risk_score = 0.93
    supervised_score = 0.91
    anomaly_score = 0.7
    sequence_score = 0.6
    reasons = ["merchant risk"]
    triggered_rules = ["counterfeit.cross_border_swipe"]
    features = {"amount": 100.0}


def test_audit_chain_verifies(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(20):
        log.append(f"txn_{i}", f"card_{i % 4}", "v1", _Result())
    ok, broken = log.verify_chain()
    assert ok and broken is None


def test_audit_chain_detects_a_modified_record(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(6):
        log.append(f"txn_{i}", "card_1", "v1", _Result())

    lines = path.read_text().splitlines()
    lines[2] = lines[2].replace('"action": "decline"', '"action": "approve"')
    path.write_text("\n".join(lines) + "\n")

    ok, broken = AuditLog(path).verify_chain()
    assert not ok and broken == "txn_2"


def test_audit_chain_detects_a_deleted_record(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(6):
        log.append(f"txn_{i}", "card_1", "v1", _Result())

    lines = path.read_text().splitlines()
    del lines[3]
    path.write_text("\n".join(lines) + "\n")

    ok, broken = AuditLog(path).verify_chain()
    assert not ok


def test_audit_log_stores_no_raw_card_id(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("txn_1", "card_0001234", "v1", _Result())
    content = (tmp_path / "audit.jsonl").read_text()
    assert "card_0001234" not in content
    assert pseudonymize("card_0001234") in content


def test_pseudonymization_is_stable_and_distinct():
    assert pseudonymize("card_a") == pseudonymize("card_a")
    assert pseudonymize("card_a") != pseudonymize("card_b")


def test_audit_log_resumes_the_chain_after_restart(tmp_path):
    """A pod restart must continue the existing chain, not start a new one."""
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append("txn_0", "card_1", "v1", _Result())
    AuditLog(path).append("txn_1", "card_1", "v1", _Result())  # fresh instance
    ok, _ = AuditLog(path).verify_chain()
    assert ok
    assert len(list(AuditLog(path).read())) == 2
