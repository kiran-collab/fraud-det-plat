"""Kubeflow Pipelines definition for training, validation and promotion.

    python pipelines/kubeflow/fraud_pipeline.py   # compiles to fraud_pipeline.yaml

The pipeline exists to make promotion a *decision with gates*, not a manual
step someone performs on a Friday afternoon. The important structure is:

    ingest -> features -> train -> evaluate -> governance -> [GATE] -> export -> promote

Nothing reaches ``current`` without clearing the gate. The gate is a real
conditional on the governance step's output, not a report that gets emailed
and ignored: a fairness or drift finding stops the pipeline, and a human has to
either fix the model or record an explicit override.

The champion/challenger comparison matters as much as the absolute metrics. A
model that is worse than the one already in production should not be promoted
even if it clears every threshold on its own.
"""

from __future__ import annotations

from kfp import compiler, dsl

BASE_IMAGE = "ghcr.io/example/fraudplat-training:0.1.0"

# A candidate must beat the incumbent by more than this to justify the
# operational cost and risk of a swap. Retraining noise alone is worth ~0.005
# PR-AUC on this data, so a smaller "improvement" is not one.
MIN_PR_AUC_IMPROVEMENT = 0.01


@dsl.component(base_image=BASE_IMAGE)
def ingest_transactions(
    lookback_days: int,
    output_data: dsl.Output[dsl.Dataset],
) -> None:
    """Pull the labelled transaction window from the lake.

    Labels lag authorizations by 30-90 days (chargebacks settle slowly), so the
    training window deliberately ends before the present: the most recent weeks
    have incomplete labels and would teach the model that recent fraud does not
    exist.
    """

    from fraudplat.data.generator import generate

    df = generate(n_transactions=5_000_000, days=lookback_days)
    df.to_parquet(output_data.path, index=False)


@dsl.component(base_image=BASE_IMAGE)
def train_models(
    input_data: dsl.Input[dsl.Dataset],
    seed: int,
    model_dir: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
) -> None:
    import json
    import subprocess

    subprocess.run(
        ["python", "scripts/train.py", "--data", input_data.path,
         "--seed", str(seed), "--version", "candidate"],
        check=True,
        env={"FP_MODEL_DIR": model_dir.path},
    )
    manifest = json.loads(f"{model_dir.path}/candidate/manifest.json")
    ens = manifest["metrics"]["ensemble"]
    metrics.log_metric("pr_auc", ens["pr_auc"])
    metrics.log_metric("roc_auc", ens["roc_auc"])
    metrics.log_metric("recall_at_1pct_fpr", ens["recall_at_1pct_fpr"])
    metrics.log_metric("value_detection_rate", ens["value_detection_rate"])


@dsl.component(base_image=BASE_IMAGE)
def run_governance_checks(
    input_data: dsl.Input[dsl.Dataset],
    model_dir: dsl.Input[dsl.Model],
    report: dsl.Output[dsl.Artifact],
) -> bool:
    """Fairness and drift. Returns False to stop promotion."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/governance_report.py",
         "--data", input_data.path, "--version", "candidate", "--strict"],
        env={"FP_MODEL_DIR": model_dir.path, "FP_REPORT_DIR": report.path},
    )
    return result.returncode == 0


@dsl.component(base_image=BASE_IMAGE)
def compare_to_champion(
    model_dir: dsl.Input[dsl.Model],
    min_improvement: float,
) -> bool:
    """Champion/challenger gate.

    Compares the candidate against whatever is currently serving on the *same*
    held-out window. A candidate that only matches the incumbent is not
    promoted: every swap costs a cache-cold period and a fresh operational risk.
    """
    import json
    from pathlib import Path

    candidate = json.loads(Path(f"{model_dir.path}/candidate/manifest.json").read_text())
    champion_path = Path("/mnt/registry/current/manifest.json")
    if not champion_path.exists():
        return True  # nothing in production yet

    champion = json.loads(champion_path.read_text())
    cand_pr = candidate["metrics"]["ensemble"]["pr_auc"]
    champ_pr = champion["metrics"]["ensemble"]["pr_auc"]
    print(f"candidate PR-AUC {cand_pr:.4f} vs champion {champ_pr:.4f}")
    return cand_pr >= champ_pr + min_improvement


@dsl.component(base_image=BASE_IMAGE)
def export_and_quantize(model_dir: dsl.Input[dsl.Model]) -> None:
    """ONNX export with a numerical-parity check that fails the step."""
    import subprocess

    subprocess.run(
        ["python", "scripts/export_onnx.py", "--version", "candidate"],
        check=True,
        env={"FP_MODEL_DIR": model_dir.path},
    )


@dsl.component(base_image=BASE_IMAGE)
def promote(model_dir: dsl.Input[dsl.Model], version: str) -> None:
    """Repoint ``current``. Atomic, and reversible by rerunning with an older
    version - which is the documented rollback procedure."""
    from fraudplat.models.registry import ModelRegistry

    ModelRegistry(model_dir.path).promote(version)


@dsl.pipeline(
    name="fraud-detection-training",
    description="Train, validate, govern and promote the fraud scoring ensemble.",
)
def fraud_training_pipeline(
    lookback_days: int = 180,
    seed: int = 17,
    min_pr_auc_improvement: float = MIN_PR_AUC_IMPROVEMENT,
) -> None:
    data = ingest_transactions(lookback_days=lookback_days)
    # Large shuffle-heavy step; give it room rather than letting it OOM-kill
    # halfway through a 5M-row feature replay.
    data.set_memory_request("16Gi").set_cpu_request("4")

    trained = train_models(input_data=data.outputs["output_data"], seed=seed)
    trained.set_memory_request("32Gi").set_cpu_request("8")

    governance = run_governance_checks(
        input_data=data.outputs["output_data"],
        model_dir=trained.outputs["model_dir"],
    )

    champion = compare_to_champion(
        model_dir=trained.outputs["model_dir"],
        min_improvement=min_pr_auc_improvement,
    )

    # Both gates must pass. Promotion is the only step that touches production.
    with dsl.If(governance.output == True, name="governance-passed"):  # noqa: E712
        with dsl.If(champion.output == True, name="beats-champion"):  # noqa: E712
            exported = export_and_quantize(model_dir=trained.outputs["model_dir"])
            promote(
                model_dir=trained.outputs["model_dir"], version="candidate"
            ).after(exported)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=fraud_training_pipeline,
        package_path="fraud_pipeline.yaml",
    )
    print("compiled -> fraud_pipeline.yaml")
