"""Drift detection on features and scores.

Fraud models decay faster than almost any other production model, for two
reasons that call for different responses:

* **Population drift** - the traffic mix changed (a new acquirer, a new
  geography, a holiday). The model is still valid; the thresholds may not be.
* **Adversarial drift** - attackers changed behaviour specifically to evade.
  Retraining on labelled history is the wrong reflex here, because labels lag
  the attack by 30-90 days; the unsupervised scorers are what should react.

Labels are not available for weeks, so drift is monitored on inputs and score
distributions, which are available immediately. PSI is the primary statistic
because it is the one model-risk functions already have thresholds for
(<0.1 stable, 0.1-0.25 monitor, >0.25 investigate), and being able to hand a
governance committee a number they already interpret is worth more than a
marginally better test they would have to be taught.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

PSI_STABLE = 0.10
PSI_INVESTIGATE = 0.25


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    baseline_mean: float
    current_mean: float
    mean_shift_pct: float
    status: str  # stable | monitor | investigate


@dataclass
class DriftReport:
    n_baseline: int
    n_current: int
    features: list[FeatureDrift] = field(default_factory=list)
    score_psi: float = 0.0
    score_status: str = "stable"
    flagged: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.flagged

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def population_stability_index(
    baseline: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between two samples, using baseline quantile edges.

    Quantile edges from the baseline, not equal-width bins: fraud features are
    heavily skewed (amounts, velocity counts), and equal-width bins put ~95% of
    the mass in one bucket where PSI cannot see anything.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]
    if len(baseline) < 2 or len(current) < 2:
        return 0.0

    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # Near-constant feature (e.g. a rare binary flag): PSI is meaningless,
        # so compare rates directly instead of manufacturing bins.
        b_rate, c_rate = float(baseline.mean()), float(current.mean())
        return float(abs(b_rate - c_rate) * np.log(max(c_rate, 1e-6) / max(b_rate, 1e-6)))

    edges[0], edges[-1] = -np.inf, np.inf
    b_counts, _ = np.histogram(baseline, bins=edges)
    c_counts, _ = np.histogram(current, bins=edges)

    # Laplace smoothing: an empty bucket in either sample would otherwise send
    # PSI to infinity on a single unlucky observation.
    b_pct = (b_counts + 0.5) / (b_counts.sum() + 0.5 * len(b_counts))
    c_pct = (c_counts + 0.5) / (c_counts.sum() + 0.5 * len(c_counts))
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def _status(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    return "monitor" if psi < PSI_INVESTIGATE else "investigate"


def detect_drift(
    baseline_features: np.ndarray,
    current_features: np.ndarray,
    feature_names: list[str],
    baseline_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
) -> DriftReport:
    report = DriftReport(n_baseline=len(baseline_features), n_current=len(current_features))

    for i, name in enumerate(feature_names):
        b, c = baseline_features[:, i], current_features[:, i]
        psi = population_stability_index(b, c)
        b_mean, c_mean = float(np.mean(b)), float(np.mean(c))
        status = _status(psi)
        report.features.append(FeatureDrift(
            feature=name,
            psi=psi,
            baseline_mean=b_mean,
            current_mean=c_mean,
            mean_shift_pct=100.0 * (c_mean - b_mean) / max(abs(b_mean), 1e-9),
            status=status,
        ))
        if status == "investigate":
            report.flagged.append(f"{name}: PSI {psi:.3f}")

    if baseline_scores is not None and current_scores is not None:
        report.score_psi = population_stability_index(baseline_scores, current_scores, bins=20)
        report.score_status = _status(report.score_psi)
        if report.score_status == "investigate":
            # Score drift is the more serious signal: it means the *decisions*
            # moved, so the thresholds no longer mean what they were tuned to.
            report.flagged.append(f"score distribution: PSI {report.score_psi:.3f}")

    report.features.sort(key=lambda f: -f.psi)
    return report


def format_drift_report(report: DriftReport, top_n: int = 12) -> str:
    lines = [
        f"== Drift review (baseline n={report.n_baseline:,}, current n={report.n_current:,}) ==",
        f"  score PSI: {report.score_psi:.4f} [{report.score_status}]",
        f"  {'feature':<32}{'PSI':>9}{'base mean':>13}{'curr mean':>13}{'shift':>10}  status",
    ]
    for f in report.features[:top_n]:
        lines.append(
            f"  {f.feature:<32}{f.psi:>9.4f}{f.baseline_mean:>13.4f}"
            f"{f.current_mean:>13.4f}{f.mean_shift_pct:>9.1f}%  {f.status}"
        )
    lines.append(f"  RESULT: {'PASS' if report.passed else 'INVESTIGATE'}")
    lines += [f"    - {f}" for f in report.flagged]
    return "\n".join(lines)
