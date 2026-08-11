# Model Governance

What a second-line model-risk function needs to sign off a fraud scorecard, and
where each item is produced in this repository.

Generate the current pack:

```bash
python scripts/governance_report.py --strict
```

`--strict` exits non-zero on any finding, which is how the Kubeflow pipeline
gates promotion. A governance report that cannot block a release is a report
nobody reads.

---

## 1. Model inventory

Every registered version writes a manifest (`artifacts/models/<version>/manifest.json`)
recording the data window and row counts per split, realised prevalence, seed,
LightGBM best iteration, full metric set, feature list and count, git SHA, and
training wall-clock.

The feature list is checked on load: if the code's `FEATURE_NAMES` no longer
matches the manifest, the registry refuses to serve rather than silently
scoring a misaligned vector.

---

## 2. Performance

Primary metric is **PR-AUC**. At 0.85% prevalence, ROC-AUC is dominated by the
negative class — it stays near 0.99 while precision in the alerting band
collapses, so it is reported but never used as the decision metric.

Reported at each operating point: precision, recall, false-positive rate, alerts
per 10k transactions, and **value recall** (share of fraud *dollars* caught, not
just counts — a scorecard that catches many small frauds and misses the large
ones is worse than its count-based recall suggests).

`evaluation.py` also produces an estimated net loss reduction from three
explicit business inputs — recovery rate, analyst cost per reviewed alert, and
the cost of a false decline. They live in code so the estimate is arguable
rather than buried.

**Champion/challenger.** The pipeline compares a candidate against the incumbent
on the same held-out window and requires a ≥0.01 PR-AUC improvement. Retraining
noise alone is worth ~0.005 on this data, and every promotion costs a
cache-cold period and fresh operational risk.

---

## 3. Explainability

| Method | Where | Purpose |
|---|---|---|
| SHAP (TreeExplainer) | production, ~1-3ms/row | per-transaction reason codes |
| SHAP global | governance pack | mean\|contribution\| over the test window |
| LightGBM gain | manifest | cross-check against SHAP |
| LIME | analyst tooling | local counterfactual second opinion |

SHAP rather than gain for individual decisions: gain is a property of the
*model*, while a chargeback dispute is about a specific transaction. Additivity
means contributions reconcile to the score, which is the property an examiner
checks.

LIME is not redundant — it answers a different question ("what would have had to
change for this to flip"). Where SHAP and LIME disagree, the case is usually
sitting on a sharp decision boundary and worth a closer look.

Reason codes are mapped to plain language in `serving/decisioning.py` and
attached to **every decline**, because an adverse action without a recorded
reason is a compliance problem.

---

## 4. Fairness

Measured across `customer_age_band`, which is **excluded from the model** and
retained solely for this monitoring.

Excluding a protected attribute does not make a model fair — proxies survive
exclusion. That is exactly why measurement happens downstream on outcomes.

| Metric | Question | Default threshold |
|---|---|---|
| Adverse impact ratio | are segments declined at comparable rates? | ≥0.80 (four-fifths rule) |
| False positive rate gap | are some segments wrongly declined more? | ≤0.01 |
| True positive rate gap | do some segments get less protection? | ≤0.15 |

**Findings vs. caveats are separated.** A metric that could not be assessed —
too few transactions in a segment, too few positives for a stable TPR, nothing
flagged at all at this threshold — is recorded as a *note*, not a *flag*. "We
did not measure this" and "we measured this and it was fine" must never look
identical in a governance pack, but only the latter should gate a release.

Sample-size floors: 200 transactions per segment, 20 positives before a TPR is
reported, 20 flagged transactions before selection-rate parity is assessed.
Reporting a 40-point TPR gap computed from six fraud cases invites exactly the
wrong remediation.

A high selection rate with a proportionally high fraud rate is not evidence of
bias. A high **false positive rate** is.

---

## 5. Drift

Fraud models decay faster than almost any other production model, and for two
reasons that need different responses:

* **Population drift** — the traffic mix changed. The model is still valid; the
  thresholds may not be.
* **Adversarial drift** — attackers changed behaviour to evade. Retraining is
  the wrong reflex, because labels lag the attack; the unsupervised scorers are
  what should react.

Labels are unavailable for weeks, so monitoring runs on **inputs and score
distributions**, which are available immediately.

PSI is the primary statistic because model-risk functions already have
thresholds for it (<0.10 stable, 0.10–0.25 monitor, >0.25 investigate).
Implementation notes that matter: quantile bin edges from the baseline (fraud
features are heavily skewed; equal-width bins hide everything in one bucket),
Laplace smoothing (an empty bucket would otherwise send PSI to infinity), and a
rate-difference fallback for near-constant features.

Score PSI is treated as more serious than any individual feature's: it means
the *decisions* moved, so the thresholds no longer mean what they were tuned to.

---

## 6. Audit trail

Every scored transaction can produce a record: timestamp, transaction ID,
salted card hash, model version, action, all four scores, reason codes,
triggered rules, and the full feature snapshot.

The feature snapshot is not optional detail. Reconstructing a decision without
it is impossible after the fact — the card's state at that instant is gone.

Records are **hash-chained**: each carries its predecessor's hash, so a deleted
or edited row breaks the chain and `verify_chain()` locates it. This does not
prevent tampering; it makes silent tampering detectable, which is the
achievable goal. In AWS the log lands in an S3 bucket with Object Lock in
COMPLIANCE mode for seven years, and the service's IAM policy grants
`PutObject` without `DeleteObject`.

Card identifiers are salted hashes — stable enough to reconstruct a card's
decision history, not reversible to a PAN without the salt.

---

## 7. GenAI-specific controls

The investigation assistant is held to different controls than the scorecard,
because it has a different threat model.

**It does not decide.** The response schema has no disposition field. An LLM
that recommended approve/decline would become the de facto decision-maker
without any of the validation the scoring models are held to.

**Input controls.** PAN, CVV, SSN, email and phone are redacted before anything
is embedded or prompted. Redaction runs before embedding because the vector
index is long-lived — an unredacted PAN in it is a data-retention problem
independent of what the model does.

**Retrieval controls.** Merchant descriptors and device strings are
attacker-controllable and end up in prompts. Retrieved documents are fenced in
`<case>` tags with an explicit system-prompt contract that their content is
data, never instructions; closing-tag sequences are neutralised so a document
cannot break out of its fence. Pattern matching *flags* injection attempts for
the security team rather than being the only defence — sanitising by regex is a
losing game against unbounded phrasings.

**Output controls.** Responses are blocked if they contain anything matching a
card number (redaction failed upstream; do not compound it) or claim to have
taken an action (the assistant has no write path, and a false "I have blocked
the card" would leave an analyst believing a case is handled).

**Audit.** Every call logs case ID, redaction counts, injection flags, retrieved
case IDs and model — but never the question text, which may contain what the
redactor missed.

---

## 8. Promotion gates

```
train → evaluate → governance ─┐
                               ├─ both must pass ─→ export → promote
        champion comparison ───┘
```

Promotion is the only pipeline step that touches production, and it is
downstream of both gates. Rollback is the same `promote()` call with an earlier
version — atomic, and the reason nothing else in the codebase reads a version
string directly.

---

## Open items

Honest gaps a reviewer would raise:

- No production A/B or shadow-mode framework; champion/challenger is offline only.
- Fairness is monitored on one attribute. A real deployment needs the full
  protected-class set available to second line.
- Drift thresholds are conventional defaults, not calibrated to this
  portfolio's observed month-over-month variation.
- No automated retraining trigger — drift findings alert, they do not act.
- The GenAI assistant has guardrails but no red-team evaluation suite; the
  injection tests cover known patterns, not adaptive attackers.
