# Fraud Detection Platform

Real-time card-transaction fraud detection: streaming feature computation, a
three-model scoring ensemble, per-transaction explanations, a RAG investigation
assistant for analysts, and the governance controls a bank's second-line risk
function will ask for.

Everything runs end to end on a laptop against synthetic data — no Kafka, no
Redis, no cloud account, no API key required.

```bash
make venv           # create .venv and install
make all            # train -> export to ONNX -> build case index -> governance pack
make demo           # end-to-end walkthrough with attack scenarios
make serve          # scoring API on :8080
```

---

## What it does

A card authorization arrives. Within a few milliseconds the platform has to
decide: approve, challenge, queue for review, or decline.

```
                    Kafka (partitioned by card_id)
                              │
                     ┌────────▼─────────┐
                     │ Feature writer   │  FeatureEngine, event-at-a-time
                     └────────┬─────────┘
                              ▼
                     Redis online store          ◄── same state schema as batch
                              │
   auth request ─────►┌───────▼────────┐
                      │  Scoring API   │
                      │                │
                      │  LightGBM      │ known fraud patterns    (calibrated)
                      │  IsolationF.   │ unlike normal traffic   ─┐ logit
                      │  Transformer   │ unlike this cardholder  ─┘ uplift
                      │       ▼        │
                      │  blend + rules │
                      │       ▼        │
                      │  SHAP reasons  │ (declines always; others on request)
                      └───────┬────────┘
                              ▼
                approve / step-up / review / decline
                              │
                    ┌─────────▼──────────┐
                    │ hash-chained audit │  ◄── analyst opens a review case
                    └────────────────────┘         │
                                                   ▼
                                       RAG investigation assistant
                                       (Claude + case retrieval)
```

---

## Measured results

From `make train` on 120k synthetic transactions (0.85% fraud prevalence),
scored on a **chronologically held-out** test window:

| Scorer | PR-AUC | ROC-AUC | Recall @ 1% FPR | Precision @ 0.86 | Fraud value stopped |
|---|---|---|---|---|---|
| **Ensemble** | **0.618** | **0.988** | **0.757** | 0.860 | **68.0%** |
| LightGBM only | 0.585 | 0.987 | 0.737 | 1.000 | 50.7% |
| Isolation Forest only | 0.070 | 0.892 | 0.000 | 0.095 | 79.7% |
| Sequence model only | 0.035 | 0.689 | 0.132 | 0.196 | 48.4% |

PR-AUC is the headline, not ROC-AUC: at 0.85% prevalence ROC-AUC is dominated
by the negative class and stays near 0.99 while precision in the alerting band
collapses.

The two unsupervised scorers are weak alone — that is expected and is not what
they are for. Isolation Forest illustrates why the columns must be read
together: it stops 79.7% of fraud value, more than anything else in the table,
at 9.5% precision — it flags roughly one transaction in ten and would be
unusable as a decision rule.

Blended, they add **+0.033 PR-AUC** over the supervised model and lift fraud
*value* stopped at the decline threshold from 50.7% to 68.0%, by escalating the
high-value takeover cases the supervised model rates just under the line.

**This is a trade, not a free win.** Precision at the decline threshold drops
from 1.000 to 0.860. Concretely, on the 18,000-transaction test window
(152 fraud): the supervised model declines 30 transactions, all of them fraud;
the ensemble declines 43, of which 37 are fraud and **6 are false declines**.
Seven additional frauds caught — and a 17-point gain in fraud *value* stopped,
since the extra catches skew high-value — at the cost of six inconvenienced
customers.

The cost model in `evaluation.py` prices that as clearly favourable, but
whether it stays favourable is portfolio-specific, which is why the recovery
rate, review cost and false-decline cost are explicit parameters rather than
constants buried in the code.

### Latency (`make bench`, single core, M-series laptop)

| Path | p50 | p99 |
|---|---|---|
| End-to-end, native Python | 6.00 ms | 6.63 ms |
| **End-to-end, ONNX Runtime** | **2.55 ms** | **3.91 ms** |
| + SHAP reason codes | 4.43 ms | 5.85 ms |

**2.36× faster** end to end, against a 50 ms budget — ~393 txn/s per core, so
5M transactions/day is ~15% of one core of steady-state compute.
Per-component:

| Component | Python | ONNX | Speedup |
|---|---|---|---|
| LightGBM | 0.049 ms | 0.007 ms | 7× |
| Isolation Forest | 4.94 ms | 1.67 ms | 3× |
| Transformer | 0.397 ms | 0.057 ms (int8) | 7× |

The Isolation Forest was the finding. It looked like the cheapest model and was
actually **85% of total latency** — scikit-learn's per-call validation and
joblib dispatch overhead, amortised over a batch but catastrophic on a batch of
one. Exporting the other two models first produced almost no end-to-end
improvement; the benchmark is what surfaced why.

---

## Design decisions worth reading

### One feature implementation, not two

The most expensive bug class in a fraud platform is train/serve skew: the batch
job computes `card_txn_count_1h` one way, the streaming consumer another, and
the model degrades in production while every offline metric stays green.

[`features/transforms.py`](src/fraudplat/features/transforms.py) is a single
incremental engine. The Kafka consumer feeds it one transaction per message;
the trainer feeds it the same transactions in timestamp order. Both get
byte-identical vectors, and
[`test_features.py`](tests/test_features.py) asserts that equality — including
across a full serialize/hydrate round trip through the online store, which is
what the serving path actually does.

### Blending in logit space, not probability space

The obvious ensemble is `0.65·p + 0.15·iforest + 0.20·sequence`. It is wrong
here, and measurably so.

The calibrated supervised probability sits near zero for the 99.15% of traffic
that is legitimate, while anomaly scores are spread across [0,1] by
construction. Adding them linearly puts a floor under every legitimate
transaction any detector finds mildly odd. Measured: precision at the decline
threshold went from **1.00 → 0.00**.

[`models/ensemble.py`](src/fraudplat/models/ensemble.py) blends in log-odds
instead. The unsupervised scores act as *bounded evidence adjustments* around
their own median: a transaction unremarkable to both detectors keeps the
supervised model's calibrated probability exactly, while a genuinely strange
one is pushed up by at most 2.5 log-odds. Calibration survives at the low end;
novel attacks can still be escalated.

### The sequence model predicts choices, not aggregates

The transformer does next-event prediction: given a card's previous 7
transactions, predict the current one; the error is the anomaly score. But
roughly half the feature vector is rolling aggregates *of the input window* —
trivially predictable by construction. Including them let the network drive the
loss down while learning nothing, and the reconstruction error stopped
discriminating.

It now predicts only the 15 features that represent a genuine *choice*: what
was bought, where, how, for how much, at what hour.

### Cold starts are neutral, not anomalous

A card's first transaction has no rhythm to deviate from. Scoring it as highly
anomalous would penalise every new cardholder — bad detection and a fair-lending
problem. Cards without history get a neutral 0.5 sequence score, asserted in
tests.

### Rules escalate, never downgrade

Deterministic overlays (card-testing bursts, impossible travel, cross-border
magstripe) run *after* the model and can only raise the action. A rule that
could soften a model decline would be an unreviewable bypass of the model's
authority.

### Retrieved documents are fenced, not trusted

The investigation assistant reads merchant descriptors and device strings —
attacker-controllable text that ends up in a model prompt. That is an indirect
prompt-injection channel.

[`genai/guardrails.py`](src/fraudplat/genai/guardrails.py) fences retrieved
content in `<case>` tags with an explicit system-prompt contract that content
inside them is data, never instructions. Pattern-matching *flags* injection
attempts for the security team rather than being the only defence — sanitising
by regex is a losing game. Output validation blocks any response claiming to
have taken an action, because the assistant is advisory and has no write path.

---

## Repository layout

```
src/fraudplat/
  data/           transaction schema + synthetic generator with realistic attack patterns
  features/       the shared feature engine; Feast definitions; Kafka producer/consumer
  models/         LightGBM, Isolation Forest, PyTorch transformer, ensemble, registry
  explain/        SHAP (production path) and LIME (analyst second opinion)
  inference/      ONNX export, int8 quantization, runtime session management
  serving/        FastAPI service, scorer, decision layer
  genai/          RAG assistant, vector store, guardrails, case ingestion
  governance/     bias monitoring, drift detection, hash-chained audit log
scripts/          train, export, benchmark, governance, demo, index build
pipelines/        Kubeflow training + promotion pipeline with gates
infra/terraform/  EKS, MSK, ElastiCache, S3, KMS, IRSA
deploy/k8s/       deployments, HPA/KEDA autoscaling, PDB
docs/             architecture and model governance
```

---

## Running it

### Train

```bash
python scripts/train.py --rows 250000 --promote
```

Chronological split (never random — a random split leaks future card velocity
into training), merchant encoding fitted on the training window only, features
replayed through one engine across all three splits so validation and test
start with realistic card state.

### Export and benchmark

```bash
python scripts/export_onnx.py        # refuses to promote if parity > 1e-4
python scripts/benchmark_inference.py
```

### Serve

```bash
make serve
curl -s localhost:8080/score -H 'content-type: application/json' -d '{
  "transaction_id":"t1","card_id":"card_0000001","merchant_id":"mch_000777",
  "merchant_category":"gambling","merchant_country":"RO","amount":3800,
  "channel":"ecom","entry_mode":"keyed","device_id":"dev_new","explain":true}' | jq
```

Endpoints: `/score`, `/score/batch`, `/health`, `/ready`, `/metrics`, `/model`.

### Governance

```bash
python scripts/governance_report.py --strict   # non-zero exit gates promotion
```

Performance by operating point, global SHAP attributions, fairness across
customer segments (selection-rate parity, equal opportunity, FPR gap), and
feature/score drift (PSI).

### GenAI assistant

Needs `pip install -e ".[genai]"` and `ANTHROPIC_API_KEY`. Without them the
assistant degrades to retrieval-only rather than failing — retrieved case IDs
are still useful to an analyst.

---

## Testing

```bash
make test    # 91 tests
make lint
```

Tests assert on **properties** rather than metric values, so they keep catching
regressions as models legitimately change: batch/stream parity, causality of
sequence windows, monotonicity and boundedness of the blend, rule escalation
ordering, injection detection, PSI behaviour on skewed and constant features,
and audit-chain tamper detection.

---

## Limitations

Stated plainly, because these matter more than the headline numbers:

- **The data is synthetic.** The fraud patterns are modelled on real attack
  shapes (card testing, takeover, counterfeit, merchant compromise) but the
  metrics above describe this generator, not a production portfolio. Real
  card-fraud PR-AUC depends heavily on portfolio mix and label quality.
- **Feast and Kafka are wired but not exercised end to end** in the local demo,
  which uses the in-process store. The interfaces are the ones the streaming
  path uses; the integration is not covered by tests.
- **Terraform is not applied.** It validates and formats; it has never been run
  against a real account, so instance sizing is reasoned from the measured
  latency rather than from production load.
- **The sequence model is weak on this data** (ROC-AUC 0.69 standalone). It
  earns its place through the blend, not on its own.
- **Metrics move between retrains.** Value-based figures in particular are
  sensitive to which side of the decline threshold a handful of high-value
  transactions land on; treat single-run differences under a few points as
  noise rather than signal.
- **Bias monitoring measures outcomes, not causes.** A flagged disparity is the
  start of an investigation, not a conclusion.
