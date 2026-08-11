# Architecture

## The constraint that shapes everything

A card authorization has a hard end-to-end budget — the network expects a
response in well under a second, and risk scoring is one of several steps
inside it. Our slice is ~50ms p99.

That single number drives most of the design:

* features must be **read**, not computed from history, at score time — hence a
  streaming writer and an online store rather than an on-demand query
* the model must be small enough to run in single-digit milliseconds — hence a
  120k-parameter transformer rather than a large sequence encoder
* explanations must be optional on the hot path — hence SHAP only on declines
  and explicit requests
* the service scales by **process count**, not threads per request — hence
  single-threaded ONNX sessions and `OMP_NUM_THREADS=1`

---

## Data flow

### Write path (continuous)

```
card network ──► Kafka topic (24 partitions, keyed by card_id)
                      │
                      ▼
              Feature writer pods (KEDA-scaled on consumer lag)
                      │  FeatureEngine.transform(txn)
                      ▼
              Redis: fp:card:{card_id} -> serialized card state
```

Partitioning by `card_id` is load-bearing, not an optimisation. The engine's
windows are per-card and order-dependent, so all of a card's events must be
processed in order by one consumer. This is also what makes the lock-free
read-modify-write in the consumer safe, and why the deployment uses a
`Recreate` strategy — a rolling update would put two writers in the group
mid-batch and interleave them on the same card.

Offsets are committed *after* the store write (at-least-once). A crash between
the two replays one message and double-counts a transaction in a velocity
window: a bounded error that decays out within 7 days. Committing first would
drop the event and leave a permanent hole.

### Read path (per authorization)

```
POST /score
   ├─ GET  fp:card:{card_id}          ~1-2ms
   ├─ FeatureEngine.transform          ~0.05ms   (31 features)
   ├─ LightGBM (ONNX)                  ~0.01ms
   ├─ IsolationForest (ONNX)           ~1.7ms
   ├─ Transformer (ONNX int8)          ~0.06ms
   ├─ logit blend + overlay rules      negligible
   ├─ SHAP  [declines / on request]    ~1.9ms
   └─ SETEX fp:card:{card_id}          ~1-2ms
```

`commit=false` skips the final write, so a pre-authorization quote cannot
pollute a cardholder's behavioural baseline.

---

## Components

### Feature engine

31 features in four groups: transaction-intrinsic, card behavioural baseline,
card velocity, and counterparty risk. Every window is strictly backward-looking
— a transaction never sees itself.

State per card is bounded: a 7-day event deque plus running moments, with
"ever-seen" novelty sets capped (512 merchants / 128 devices / 64 countries).
Without those caps a high-volume card's Redis value grows without limit.

The merchant risk encoding is smoothed toward the portfolio base rate with a
prior weight of 50, so a merchant with one chargeback does not get a 100% fraud
score. It is fitted on the training window only and frozen into the model
bundle — refreshing it under a fixed model would be a silent drift source.

### Models

| Model | Answers | Trained on | Weight |
|---|---|---|---|
| LightGBM | "does this match labelled fraud?" | labelled history | base probability |
| Isolation Forest | "is this unlike normal traffic?" | legitimate only | ≤1.6 log-odds |
| Transformer | "does this follow from this card's rhythm?" | legitimate only | ≤1.1 log-odds |

The unsupervised pair exists because label feedback lags an attack by 30–90
days. A stacked meta-learner would inherit exactly that blind spot, which is
why the blend uses fixed, auditable coefficients rather than a second learned
layer — and why it survives a model-risk review more easily.

LightGBM uses `scale_pos_weight` rather than SMOTE: at 0.85% prevalence,
synthetic oversampling manufactures cardholder behaviour that never happened
and pushes the velocity features out of distribution.

Early stopping is on validation **PR-AUC**. ROC-AUC plateaus happily while
precision in the alerting band collapses.

### Decisioning

Four actions — approve, step-up (3DS/OTP), review, decline — from three
thresholds, plus deterministic overlay rules that can only escalate.

Thresholds live in ConfigMap, not in the model artifact: retuning an operating
point is a risk decision that should not require a retrain or a code deploy.

### Model registry

A version is an immutable directory plus a manifest (data window, metrics,
feature list, git SHA). The `current` symlink is the only mutable thing, so
promotion and rollback are one atomic operation.

Loading validates the feature contract: if the code's `FEATURE_NAMES` no longer
matches the manifest, it refuses to serve rather than scoring a misaligned
vector.

---

## Failure modes and behaviour

| Failure | Behaviour | Rationale |
|---|---|---|
| Redis unreachable | falls back to in-process state; `/health` reports `backend: memory` | degraded velocity beats no authorization decisions |
| No ONNX build | falls back to native Python | an export is an optimisation, not a correctness requirement |
| No registered model | starts, fails `/ready`, `/health` explains | crash-looping hides the reason |
| Vector index down | assistant degrades to retrieval-only | a broken index should not take the tool offline |
| LLM unavailable | returns retrieved case IDs with a stated reason | analysts can still work the case |
| Feature writer lag | KEDA scales on consumer lag, not CPU | the writer is I/O-bound; CPU stays idle while lag grows |
| Poisoned Kafka message | logged, counted, stream continues | one bad message must not stop the feed |

---

## Scaling to 5M transactions/day

5M/day is ~58 txn/s average, with peaks several times that. At the measured
2.64 ms p50 a single core handles ~380 txn/s, so the *steady-state* compute
requirement is a fraction of one core — the replica count is driven by
redundancy, burst headroom and zone spread, not throughput.

The real constraints are elsewhere:

* **Online store memory.** ~4KB of serialised state per active card. At 2M
  active cards that is ~8GB, which sizes the ElastiCache node.
* **Kafka retention.** 8 days, one day beyond the longest feature window, so
  the online store can be fully rebuilt by replaying the topic after a total
  cache loss with no gap in card history.
* **Consumer parallelism.** Bounded by partition count (24). Replicas beyond
  that idle.

---

## Security posture

* Cardholder data never enters the model path — only derived features.
* The scoring service holds **read-only** S3 access to the registry, so a
  compromised pod cannot promote a model.
* The audit bucket grants `PutObject` without `DeleteObject`, with S3 Object
  Lock in COMPLIANCE mode for 7 years.
* Card identifiers in the audit log are salted hashes, with the salt in Secrets
  Manager.
* MSK and ElastiCache sit in subnets with no NAT route — no egress path.
* Containers run non-root with a read-only root filesystem and all capabilities
  dropped.
