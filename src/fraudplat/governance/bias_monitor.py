"""Fairness monitoring across customer segments.

A fraud model is not a credit decision, but a decline is still an adverse
action a customer experiences, and a model that declines one age band at twice
the rate of another needs an explanation - whether or not the disparity is
lawful, it is a finding the second-line risk function will raise.

The protected attribute is never a model input (``PROTECTED_ATTRIBUTES`` in
``data/schema.py``, asserted in the tests). Excluding it does not make the model
fair - proxies survive exclusion - which is precisely why measurement has to
happen downstream on outcomes rather than being assumed away at the input.

Three metrics, because they can disagree and the disagreement is informative:

* **Selection rate / demographic parity** - are we declining segments at
  different rates? Ignores whether the decline was correct.
* **Equal opportunity (TPR gap)** - among genuinely fraudulent transactions,
  do we catch them equally across segments? A gap means unequal protection.
* **False positive rate gap** - among legitimate transactions, do we wrongly
  decline some segments more? This is the one customers feel.

A segment with a high selection rate *and* a proportionally high fraud rate is
not evidence of bias; a segment with a high false-positive rate is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Below this the estimate is too noisy to act on. Reporting a 40-point TPR gap
# computed from six fraud cases invites exactly the wrong remediation.
MIN_SEGMENT_SIZE = 200
MIN_POSITIVES_FOR_TPR = 20
# Minimum transactions actually flagged in a segment before its selection rate
# is comparable. Without this, a threshold that flags almost nothing yields
# selection rates of 0.0004 vs 0.0000 and an "adverse impact ratio" of zero -
# a maximally severe finding manufactured from two or three transactions.
MIN_SELECTED_FOR_RATIO = 20

# Four-fifths rule, the conventional disparate-impact screen.
ADVERSE_IMPACT_RATIO_FLOOR = 0.8


@dataclass
class SegmentMetrics:
    segment: str
    n: int
    n_fraud: int
    selection_rate: float          # share flagged at the decision threshold
    true_positive_rate: float | None
    false_positive_rate: float
    precision: float | None
    mean_score: float
    reliable: bool                 # sample large enough to interpret


@dataclass
class BiasReport:
    attribute: str
    threshold: float
    segments: list[SegmentMetrics] = field(default_factory=list)
    selection_rate_disparity: float = 0.0   # max/min ratio across reliable segments
    tpr_gap: float = 0.0
    fpr_gap: float = 0.0
    adverse_impact_ratio: float = 1.0
    # findings - each one requires review before promotion
    flags: list[str] = field(default_factory=list)
    # caveats - metrics that could not be assessed. Recorded because "we did
    # not measure this" and "we measured this and it was fine" must never look
    # identical in a governance pack, but they are not findings.
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.flags

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def evaluate_bias(
    df: pd.DataFrame,
    scores: np.ndarray,
    attribute: str = "customer_age_band",
    label_col: str = "is_fraud",
    threshold: float = 0.86,
    max_fpr_gap: float = 0.01,
    max_tpr_gap: float = 0.15,
) -> BiasReport:
    """Measure outcome disparities across the values of ``attribute``."""
    scores = np.asarray(scores, dtype=float)
    y = df[label_col].to_numpy().astype(int)
    flagged = scores >= threshold
    report = BiasReport(attribute=attribute, threshold=threshold)

    for value, idx in df.groupby(attribute, observed=True).indices.items():
        idx = np.asarray(idx)
        seg_y, seg_flag, seg_score = y[idx], flagged[idx], scores[idx]
        n_fraud = int(seg_y.sum())

        tp = int((seg_flag & (seg_y == 1)).sum())
        fp = int((seg_flag & (seg_y == 0)).sum())
        n_legit = int((seg_y == 0).sum())

        report.segments.append(SegmentMetrics(
            segment=str(value),
            n=len(idx),
            n_fraud=n_fraud,
            selection_rate=float(seg_flag.mean()),
            true_positive_rate=(tp / n_fraud) if n_fraud >= MIN_POSITIVES_FOR_TPR else None,
            false_positive_rate=(fp / n_legit) if n_legit else 0.0,
            precision=(tp / (tp + fp)) if (tp + fp) else None,
            mean_score=float(seg_score.mean()),
            reliable=len(idx) >= MIN_SEGMENT_SIZE,
        ))

    reliable = [s for s in report.segments if s.reliable]
    if len(reliable) < 2:
        report.notes.append("insufficient data: fewer than two segments meet the minimum size")
        return report

    # The ratio is meaningful as soon as *any* segment is flagged often enough
    # to have a stable rate. Requiring every segment to clear the bar would
    # discard the most important case: one segment declined heavily while the
    # others are barely touched is a genuine 20:1 disparity, not missing data.
    # What is not meaningful is the case where nothing is flagged anywhere.
    sel = [s.selection_rate for s in reliable]
    if max(s.selection_rate * s.n for s in reliable) >= MIN_SELECTED_FOR_RATIO:
        report.selection_rate_disparity = max(sel) / max(min(sel), 1e-9)
        report.adverse_impact_ratio = min(sel) / max(max(sel), 1e-9)
    else:
        # Leave the ratio at its neutral 1.0 and say why, rather than emitting
        # a number a governance committee would read as a violation.
        report.notes.append(
            "selection-rate parity not assessed: no segment reaches "
            f"{MIN_SELECTED_FOR_RATIO} flagged transactions at this threshold"
        )

    fprs = [s.false_positive_rate for s in reliable]
    report.fpr_gap = max(fprs) - min(fprs)

    tprs = [s.true_positive_rate for s in reliable if s.true_positive_rate is not None]
    report.tpr_gap = (max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0

    # --- thresholds -----------------------------------------------------
    if report.adverse_impact_ratio < ADVERSE_IMPACT_RATIO_FLOOR:
        report.flags.append(
            f"adverse impact ratio {report.adverse_impact_ratio:.3f} below the "
            f"{ADVERSE_IMPACT_RATIO_FLOOR} four-fifths floor"
        )
    if report.fpr_gap > max_fpr_gap:
        report.flags.append(
            f"false positive rate gap {report.fpr_gap:.4f} exceeds {max_fpr_gap} "
            "- some segments are wrongly declined more often"
        )
    if report.tpr_gap > max_tpr_gap:
        report.flags.append(
            f"true positive rate gap {report.tpr_gap:.4f} exceeds {max_tpr_gap} "
            "- some segments receive less fraud protection"
        )
    return report


def format_bias_report(report: BiasReport) -> str:
    lines = [
        f"== Bias review: {report.attribute} @ threshold {report.threshold} ==",
        f"  {'segment':<12}{'n':>8}{'fraud':>7}{'select%':>10}{'TPR':>8}{'FPR%':>8}{'prec':>8}",
    ]
    for s in sorted(report.segments, key=lambda s: s.segment):
        tpr = f"{s.true_positive_rate:.3f}" if s.true_positive_rate is not None else "  n/a"
        prec = f"{s.precision:.3f}" if s.precision is not None else "  n/a"
        mark = "" if s.reliable else "  (small)"
        lines.append(
            f"  {s.segment:<12}{s.n:>8,}{s.n_fraud:>7,}{100 * s.selection_rate:>9.3f}%"
            f"{tpr:>8}{100 * s.false_positive_rate:>7.3f}%{prec:>8}{mark}"
        )
    lines += [
        f"  adverse impact ratio: {report.adverse_impact_ratio:.3f} "
        f"(floor {ADVERSE_IMPACT_RATIO_FLOOR})",
        f"  FPR gap: {report.fpr_gap:.4f}    TPR gap: {report.tpr_gap:.4f}",
        f"  RESULT: {'PASS' if report.passed else 'REVIEW REQUIRED'}",
    ]
    lines += [f"    ! {f}" for f in report.flags]
    lines += [f"    - not assessed: {n}" for n in report.notes]
    return "\n".join(lines)
