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
scored on a **chronologically held-out** test window.

> New to these metrics? **[Jump to the glossary](#glossary--every-term-above-defined)** —
> every term below is defined there against this exact run's numbers.

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

## Glossary — every term above, defined

Everything here is anchored to **one real example**: the held-out test window
from `make train`. Learn this table and every metric below becomes arithmetic
you can do in your head.

### The one example

18,000 transactions were scored. 152 were actually fraud. At the decline
threshold of 0.86, the model flagged 43 of them.

|  | **Actually fraud** | **Actually legitimate** | total |
|---|---|---|---|
| **Model said fraud** (flagged 43) | **37** ✓ caught | **6** ✗ wrongly blocked | 43 |
| **Model said fine** | **115** ✗ missed | **17,842** ✓ correctly allowed | 17,957 |
| total | 152 | 17,848 | 18,000 |

Those four numbers have names, and **every** metric in this README is a ratio
built from them.

| Term | Means | Here |
|---|---|---|
| **True Positive (TP)** | flagged, and it really was fraud — a catch | 37 |
| **False Positive (FP)** | flagged, but it was a real customer — a *false decline* | 6 |
| **False Negative (FN)** | not flagged, but it was fraud — a *miss*, money lost | 115 |
| **True Negative (TN)** | not flagged, and it was fine — the boring majority | 17,842 |
| **Prevalence** | how much of the data is actually fraud | 152/18,000 = **0.84%** |

> **Why prevalence dominates everything.** At 0.84%, a model that simply says
> "never fraud" is **99.16% accurate**. That's why the word *accuracy* appears
> nowhere in this project — it's a useless metric here, and any fraud system
> quoting it is hiding something.

### The two ratios everyone mixes up

Both are fractions of the "caught" number 37. They differ only in the
denominator, and that is the whole trick:

**Precision = TP / (TP + FP)** — *of everything I flagged, how much was really fraud?*
→ 37 / 43 = **0.86**. When we decline, we're right 86% of the time.
**Precision is about not annoying customers.**

**Recall = TP / (TP + FN)** — *of all the fraud that existed, how much did I catch?*
→ 37 / 152 = **0.24**. We caught about a quarter of it.
**Recall is about not losing money.**

> **The memory hook.** Point at the box, then ask:
> *Precision* looks at the **column you flagged** — "was I right?"
> *Recall* looks at the **row of real fraud** — "did I get them all?"
>
> Or in one sentence: **precision is how often you're right when you speak;
> recall is how often you speak when you should.**

**They trade against each other.** Lower the threshold and you flag more —
recall goes up, precision goes down. You can see the exact see-saw in this
project's own numbers:

| Threshold | Precision | Recall | Meaning |
|---|---|---|---|
| 0.35 | 0.449 | 0.724 | catch most fraud, but over half of flags are wrong |
| 0.55 | 0.612 | 0.539 | balanced |
| 0.70 | 0.733 | 0.362 | |
| 0.86 | 0.860 | 0.243 | rarely wrong, but misses three-quarters of fraud |

**There is no "best" row.** Which one you pick *is* the business decision — and
it's why these live in a ConfigMap, changeable without retraining.

### Rate terms

**False Positive Rate (FPR) = FP / (FP + TN)** — *of all the legitimate
customers, what fraction did I wrongly block?*
→ 6 / 17,848 = **0.034%**.

> **FPR vs precision — the confusion worth clearing up once.** Both involve
> false positives, but FPR divides by *all legitimate traffic* (17,848) while
> precision divides by *what you flagged* (43). Because legitimate traffic is
> enormous, FPR always looks tiny and reassuring. A 1% FPR sounds harmless — but
> here it means **178 wrongly blocked customers to catch 115 frauds**: precision
> collapses to 0.39, so most of your declines would be wrong.
> **FPR is the operations view; precision is the customer's view.** Always
> convert an FPR into a headcount before agreeing to it.

**Recall @ 1% FPR = 0.757** — *"if we accept wrongly blocking 1% of good
customers, what fraction of fraud can we catch?"* → 75.7%.

This is the single most useful number in the whole table, because it fixes the
cost and asks about the benefit. Comparing two models by recall alone is
meaningless (any model gets 100% recall by flagging everything); comparing them
at the *same* FPR is a fair fight. 1% is the level a fraud-operations team can
realistically staff.

**Alerts per 10,000 transactions** — how many cases land in the human review
queue. → 23.9 at threshold 0.86. This is a **staffing number**: multiply by
daily volume to get how many analysts you need to hire.

### Threshold and operating point

**Threshold** — the model outputs a score from 0 to 1. The threshold is the
line where you act. Above 0.86 → decline.

**Operating point** — one chosen threshold and the whole set of consequences
that follow (precision, recall, FPR, queue size). Moving the threshold slides
you along the trade-off; it does **not** make the model better. Improving the
model moves the whole curve.

### The two curves (single-number summaries)

A threshold gives you one point. Sweeping *every possible* threshold traces a
curve, and the **area under that curve (AUC)** compresses the model's whole
quality into one number — useful for comparing models without arguing about
thresholds first.

**ROC-AUC = 0.988.** Plots recall against FPR. Interpretation: *pick one random
fraud and one random legitimate transaction — how often does the model score the
fraud higher?* 98.8% of the time. 0.5 = coin flip, 1.0 = perfect.

**PR-AUC = 0.618.** Plots precision against recall. Roughly: *average precision
across all recall levels.* A coin flip scores ≈ prevalence (0.0084 here), not
0.5 — so **0.618 is ~73× better than random**.

> **Why this project reports PR-AUC as the headline and treats ROC-AUC as
> decoration.** Look at the gap: 0.988 vs 0.618, same model, same data.
>
> ROC-AUC's denominator is dominated by the 17,848 legitimate transactions. You
> can add hundreds of false positives and FPR barely moves, so ROC-AUC stays
> near 0.99 while precision in the alerting band quietly collapses. It flatters
> every rare-event model.
>
> PR-AUC ignores true negatives entirely — it only looks at what you flagged
> and what you caught. That's exactly where the pain is. **When prevalence is
> under a few percent, believe PR-AUC.**

### The money view

**Value recall (a.k.a. "fraud value stopped") = 68.0%** — of all the fraud
*dollars*, what fraction did we block? Compare to plain recall: **24.3%**.

> **The most important line in this README.** We catch a quarter of fraud
> *transactions* but two-thirds of fraud *money* — because the model
> preferentially catches the big ones. A count-based metric would have called
> this model mediocre. Thieves steal dollars, not row counts, so value recall
> is what the loss line actually responds to.

**Estimated net loss reduction** — dollars saved, minus the cost of false
declines and analyst review time. Built from three explicit business inputs in
`evaluation.py` (recovery rate, review cost, false-decline cost) so the
assumptions are arguable rather than buried.

---

### Model terms

| Term | What it is | Why it's here |
|---|---|---|
| **Supervised** | learns from examples labelled fraud / not-fraud | LightGBM. Accurate, but only for fraud patterns already labelled |
| **Unsupervised** | learns what *normal* looks like; no labels | Isolation Forest + transformer. Labels lag an attack by 30–90 days, so these cover the gap |
| **Gradient boosting / LightGBM** | builds hundreds of small decision trees, each fixing the previous ones' mistakes | the standard winner on tabular data; fast and explainable |
| **Isolation Forest** | randomly splits the data; points that get isolated in few splits are odd | flags "unlike anything normal" without ever seeing fraud |
| **Transformer / sequence model** | reads a *sequence* and predicts what comes next | judges a transaction against **this card's rhythm** — the other two see one transaction at a time |
| **Ensemble** | combining several models' outputs | each covers the others' blind spots |
| **Stacking** | *learning* the combination with another model | deliberately **not** used — it would train on the same labels, inheriting the exact blind spot the unsupervised models exist to cover |
| **Class imbalance** | one class is far rarer than the other | 0.84% vs 99.16% — the central difficulty |
| **`scale_pos_weight`** | tell the model to treat each fraud as if it were ~118 rows | chosen over SMOTE-style oversampling, which invents cardholder behaviour that never happened |
| **Calibration** | making "0.9" actually mean "90% of these are fraud" | raw scores are ranked correctly but numerically wrong; without this, thresholds are arbitrary |
| **Isotonic regression** | calibration that only assumes "higher score = higher risk" | the mis-calibration here isn't S-shaped, so the more flexible method fits |
| **Log-odds / logit** | `log(p / (1-p))` — probability stretched onto an infinite scale | 0.001 → 0.002 is a *doubling* but looks like nothing on a 0–1 scale; log-odds makes evidence additive. See the blending section below |
| **Early stopping** | stop training when validation stops improving | prevents memorising the training set |
| **Chronological split** | train on January–October, test on November–December | a **random** split lets the model see a card's future transactions while predicting its past — metrics look great and production fails |
| **Leakage** | information at training time that won't exist at prediction time | the single easiest way to ship a broken fraud model |
| **Target encoding + shrinkage** | replace "merchant #4471" with its historical fraud rate, pulled toward the average when data is thin | one merchant, one chargeback shouldn't mean "100% fraud risk" |
| **Cold start** | a card/merchant with no history yet | scored **neutral**, never anomalous — otherwise every new customer is punished for being new |

### Serving terms

| Term | What it is | Why it's here |
|---|---|---|
| **p50 / p95 / p99** | the middle / 95th-worst / 99th-worst request out of 100 | averages hide the tail. p99 = 3.91 ms means 1 in 100 authorizations is slower than that — that's the one that times out |
| **Latency budget** | the time allowed before the card network gives up | ~50 ms for our slice; drives nearly every design choice |
| **ONNX** | a portable file format for a trained model | run the model as a fixed computation graph instead of through Python — same maths, far less per-call overhead |
| **Quantization / int8** | store weights as 8-bit integers instead of 32-bit decimals | 4× smaller, slightly faster; applied only to the transformer, since it changes tree split thresholds without helping |
| **Feature store** | database of pre-computed model inputs | "how many times has this card been used in the last hour?" must be *read* in milliseconds, not calculated |
| **Online vs offline store** | fast key-value lookup (Redis) vs historical archive (S3) | serving reads online; training reads offline; both must agree |
| **Train/serve skew** | features computed differently in training vs production | the most expensive bug class here — one shared implementation, asserted by test |
| **Velocity features** | counts and sums over recent time windows | "6 transactions in 10 minutes" is the card-testing signature |
| **SHAP** | splits a score into per-feature contributions that sum back to it | *why* a transaction was declined — required for adverse-action notices |
| **LIME** | fits a simple model near one prediction to see what would flip it | analyst-facing second opinion; too slow for the live path |

### Governance terms

| Term | What it is | Why it's here |
|---|---|---|
| **Drift** | the world changes and the model silently goes stale | fraud drifts fastest of all — attackers *adapt on purpose* |
| **PSI** (Population Stability Index) | a number for "how much did this distribution move?" | <0.10 stable, 0.10–0.25 watch, >0.25 investigate — thresholds risk teams already know |
| **Adverse action** | a decision against a customer, e.g. a decline | legally must come with a recorded reason — why declines always get SHAP codes |
| **Adverse impact ratio / four-fifths rule** | lowest group's decline rate ÷ highest group's | below 0.8 is the conventional disparate-impact red flag |
| **Equal opportunity / TPR gap** | is *recall* similar across customer groups? | a gap means some groups get **less fraud protection** |
| **Champion / challenger** | the model in production vs the candidate | a new model must *beat* the incumbent, not merely pass thresholds |
| **Audit trail** | tamper-evident record of every decision | "why did you block this in March?" must be answerable years later |

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
